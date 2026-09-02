Two more facts confirmed, and they're the load-bearing ones for this design:

- **`is_valid` never leaves the verification lambda.** It's a local in `fwd_post` (`conv_driver.cpp:1006-1078`), printed and optionally asserted, but never written into `result_t`. The selection loop (`conv_driver.cpp:562/573/585`) compares `result.duration_ms` only. **The search is structurally incapable of knowing whether the winner is correct.**
- **There is no production selector.** `heuristic_select_kernel` is a stub returning an empty tunable (`igemm_gtc_base.h:1002`). The only selection mechanism that exists is exhaustive runtime brute force.

---

# Replacement Architecture: MISA Tuning Pipeline

## 1. What's actually broken (and why it's architectural, not a set of bugs)

The current system collapses four independent concerns into one artifact (`.config`) and one process (`conv_driver.exe` runtime sweep):

| Concern                                             | Today                                                                    | Failure it causes                                                                        |
| --------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| **Feasibility** — is this tunable constructible?    | ~25 asserts scattered across 3 generators + a hand-written mapping table | PERF-003: fp16 lost 256×* tiles to a missing table row for months                        |
| **Applicability** — can it run *this* shape?        | `tunable_is_valid` in C++, duplicating Python asserts                    | Drift; large tiles silently never dispatch (PERF-002)                                    |
| **Correctness** — does it compute the right answer? | A local `bool` in a lambda                                               | **Invalid kernels win the search**                                                       |
| **Performance** — which is fastest?                 | `min(duration_ms)` over an unverified sweep, `-V 0`                      | COR-002: measurements are of uninitialized (zero) memory, and WMMA is 2× faster on zeros |

Two secondary couplings make it worse: the assembler's ±32 KB branch range forces `_all.config` files to be hand-partitioned per tile shape, and that packaging accident has leaked into the search *architecture* (scripts loop over config files); and `build_and_filter_configs.py --write` deletes any section that fails to assemble, conflating "genuinely infeasible" with "hit the COR-003 name collision".

**Design principle:** each concern becomes a separate stage with an explicit, cacheable artifact, and correctness becomes a structural gate rather than a printed diagnostic.

---

## 2. Layered architecture

```
   L0  Tunable Space Spec        tuning/space.yaml           (single source of truth)
        │  codegen ──────────────┬──────────────────────────┐
        ▼                        ▼                          ▼
   L1  Feasibility Oracle    python asserts            C++ tunable_is_valid
        │  (analytic VGPR/LDS model, no GPU, no build)
        ▼
   L2  Applicability Predicate   applicable(tunable, problem)
        │
        ▼
   L3  Analytic Cost Model       rank candidates, prune to top-N   (no GPU)
        │
        ▼
   L4  Measurement Harness       correctness-gated, real data, robust stats
        │
        ▼
   L5  Tuning Database           tuning/db/gfx1250.jsonl  (versioned, append-only)
        │
        ▼
   L6  Selection Policy          generated table → driver, no runtime search
        ▲
   L7  Search Orchestrator ──────┘  (drives L1→L5, batches hsaco under branch limit)
```

### L0 — Tunable Space Specification

One declarative file replaces the assert walls in all three WMMA generators plus the C++ validity checks.

```yaml
knobs:
  gemm_m_per_block:   {type: int, domain: [32,64,128,256]}
  lds_double_buffer:  {type: bool, default: 0}
  main_loop_interleave: {type: bool, default: 0}
  wmma_acc_high_bank: {type: bool, default: 0}
  # ...

constraints:
  - id: C001
    rule: main_loop_interleave implies lds_double_buffer
    evidence: "cross-wave LDS race, confirmed on hardware"
  - id: C002
    rule: precision == fp32 implies lds_double_buffer      # COR-001, structural
  - id: C003
    rule: mutually_exclusive(saddr_global_load, async_global_load, tdm_global_load)
  - id: C004
    rule: wmma_acc_high_bank implies not (wmma_m_tail or wmma_n_tail)
    status: LIMITATION          # ← tagged as a gap to close, not a law of nature
  - id: C005
    rule: wmma_epilogue_chunked implies not (wmma_m_tail or wmma_n_tail)
    status: LIMITATION
```

Two things this buys immediately. First, `status: LIMITATION` makes PERF-002's blocker *visible as a tracked item* rather than an assert message discovered by accident — the coupling `large tile ⇒ no tails ⇒ exact-divisibility only` becomes a queryable property of the space. Second, the Python asserts and the C++ `tunable_is_valid` are both **generated** from this file, so the two can no longer drift.

Kernel naming moves to **content addressing**: `igemm_fwd_gtcw_nhwc_fp16_bt128x128x32_<short-hash-of-canonical-tunable>`. Any knob that changes codegen changes the hash, so COR-003 becomes unrepresentable. Keep the human-readable prefix for grep-ability; the hash carries uniqueness.

### L1 — Feasibility Oracle (analytic, no build)

Today, discovering that `local_prefetch_num=2` needs 316 VGPRs costs a full codegen + assemble cycle and surfaces as 425 `register index is out of range` errors. Replace with a closed-form resource model:

```python
acc_per_lane = wmma_repeat_m * wmma_repeat_n * inst.num_v_c
vgpr = operands(local_prefetch_num, repeats) + addressing + staging \
     + (0 if wmma_acc_high_bank else acc_per_lane)
lds  = max(lds_single * lds_buffer_num, 0 if direct_store else epilogue_lds(...))
feasible = vgpr <= 256 and lds <= 65536
```

Two hard requirements. The model must be **validated against real builds in CI** — a sampled set of points is actually assembled, and any disagreement between predicted and `.amdhsa_next_free_vgpr` is a build failure, not a warning. And the mapping table (`ctrl_wmma_mapping_table`) becomes **generated** from the space + this model rather than hand-maintained, which is what makes PERF-003 structurally impossible.

Note the model also fixes PERF-007 for free: `epilogue_lds` must take `direct_store` as an input, which it currently does not (`igemm_fwd_gtc_wmma_nhwc.py:649-696`).

### L2 — Applicability Predicate

Generated into both languages from L0. Same content as `igemm_fwd_gtc_driver.h:497-502`, but derived rather than hand-written, and — critically — *queryable offline*, so the orchestrator can ask "which shapes in my benchmark set can this tunable even run?" before building anything. That question is what would have exposed PERF-002 immediately.

### L3 — Analytic Cost Model

Purpose is **pruning, not deciding**. The investigation supplies real constants:

```
t_global ≈ bytes_moved / bw(working_set)     # measured curve: 4MB→48.5, 64MB→33.0, 1GB→11.4 TB/s
t_lds    ≈ lds_traffic / lds_bw
t_wmma   ≈ flops / 2157e12                   # measured on RANDOM data, not the 629 TF the driver assumes
t_total  ≈ t_global + t_lds + t_wmma         # additive today (PERF-001: near-zero overlap)
                                             # → max(...) once pipelining lands
```

The additive-vs-max switch is a model *parameter*, so the same model describes the kernel before and after the PERF-001 fix, and its error is a measurable regression signal.

This also retires `get_theoritical_gpu_gflops` (COR-005), which reports 60% for a kernel at 17.8% of achievable peak and 117.38% for fp32. Efficiency is reported against the calibrated roofline or not at all.

**Explicit non-goal:** the cost model never selects. It orders candidates so L4 measures the top-N instead of all of them. Measurement remains the arbiter.

### L4 — Measurement Harness

The contract, stated as an invariant:

> A performance number exists only if the same launch that produced it was verified correct.

Concretely:
- `is_valid` becomes a field of `result_t`, populated by the post-lambda. Selection reads `if (result.valid && result.duration < best.duration)`. This is a ~10-line change and it is the highest-value fix in the entire redesign.
- Inputs are randomized and copied **unconditionally** (COR-002). The `-V` flag then controls only whether the *reference comparison* runs, never whether buffers are initialized. Given the measured 2.06× zero-operand fast path, zero-filled buffers are not a benign default — they are a systematically wrong one.
- Statistics: report **median and IQR** over N repeats, not `min`. `min` is the correct estimator only for a noiseless deterministic process; it maximally rewards the run that got lucky, and combined with the missing correctness gate it is exactly how "wrw 1.28–2.70× faster than MIOpen" was produced.
- Record `sclk` and package power per measurement so DVFS confounding is detectable after the fact (it was ruled out for the zero-data effect only because those were sampled).

### L5 — Tuning Database

Today, tuning results live in markdown files and are recomputed from scratch on every run. Replace with an append-only record store:

```json
{"schema": 1,
 "arch": "gfx1250", "generator_git": "a1b8889", "rocm": "10.1",
 "problem": {"dir":"fwd","prec":"fp16","layout":"nhwc",
             "n":128,"c":1024,"hi":14,"wi":14,"k":1024,"y":1,"x":1,
             "stride":[1,1],"pad":[0,0],"dil":[1,1],"group":1},
 "tunable_hash": "c3f1a9e2", "tunable": {...},
 "valid": true, "cost_ms": {"median":0.106,"iqr":0.002,"n":20},
 "sclk_mhz": 2278, "power_w": 1564, "ts": "2026-09-02T00:00:00Z"}
```

Keyed by `(arch, generator_git, problem_signature, tunable_hash)`. Keying on `generator_git` is what makes results safely invalidatable: a codegen change silently changing what a tunable *means* is the classic way tuning databases rot.

### L6 — Selection Policy

`heuristic_select_kernel` stops being a stub. The DB is compiled into a generated lookup table shipped with the driver:

1. Exact problem-signature hit → the recorded best *valid* tunable.
2. Miss → nearest-neighbour over normalized features (gemm_m/n/k, arithmetic intensity, divisibility flags for each candidate tile), restricted to tunables applicable to the query shape.
3. Miss → analytic model (L3) picks; the result is flagged `unvalidated` in the log so it's visible.

Production runs then perform **zero** runtime search. Runtime sweep remains available as an explicit tuning mode, not the default path.

### L7 — Search Orchestrator

Replaces `generate_all_configs.py` + `build_gfx1250_master_configs.py` + `build_and_filter_configs.py`.

- Enumerate space → L1 feasibility → L2 applicability against the target shape set → L3 ranking → build only survivors.
- **Batching is a packaging detail, not a config-authoring task.** The orchestrator packs kernels into `.hsaco` modules sized to stay under the ±32 KB branch limit. Nobody hand-partitions `_all.config` files by tile shape again.
- Search strategy: exhaustive where the feasible set is small (it usually is after L1+L2); successive halving for large spaces — short measurements on all candidates, full-precision measurement on survivors.
- A build failure is an **error**, never a silent drop. `build_and_filter_configs.py`'s current behaviour would have deleted a legitimate config on a name collision; under L0's content addressing collisions can't happen, and any remaining failure means the L1 model is wrong and must be fixed.

---

## 3. What this fixes

| Finding                                         | Fixed by                                                                   |
| ----------------------------------------------- | -------------------------------------------------------------------------- |
| Invalid kernels win the search                  | L4 correctness gate (`result_t.valid`)                                     |
| COR-002 `-V 0` measures zeros                   | L4 unconditional init                                                      |
| COR-001 fp32 `lds_double_buffer` (202 sections) | L0 constraint C002 — unrepresentable                                       |
| COR-003 name collision                          | L0 content-addressed naming                                                |
| COR-004 `tunables[0]` buffer width              | L4 per-kernel buffer sizing; removes the `ACCUMULATE_WIDTH_KEYS` exclusion |
| COR-005 broken efficiency model                 | L3 calibrated roofline                                                     |
| PERF-002 large tiles never dispatch             | L2 offline applicability + L0 `status: LIMITATION` tracking                |
| PERF-003 missing table rows                     | L1 generates the table                                                     |
| PERF-007 `direct_store` LDS over-reservation    | L1 resource model takes `direct_store` as input                            |
| `min(costs)` over noise                         | L4 median + IQR                                                            |
| No production selector                          | L6                                                                         |

PERF-001 and PERF-004 (pipelining, partial waits) are **not** addressed here — they are codegen work in `wmma_main_loop.py`, orthogonal to selection. This architecture's contribution to them is that it makes their effect measurable without the confounds that currently swamp it.

---

## 4. Migration path

Each step is independently valuable and independently shippable; none requires the next.

| #   | Step                                                                     | Effort | Unblocks                                                       |
| --- | ------------------------------------------------------------------------ | ------ | -------------------------------------------------------------- |
| 1   | `result_t.valid` + gate selection; unconditional buffer init             | XS     | Everything — all existing numbers are suspect until this lands |
| 2   | Median/IQR statistics; record sclk/power                                 | XS     | Trustworthy deltas                                             |
| 3   | Content-addressed kernel names (Python + C++ together)                   | S      | Removes name-collision class                                   |
| 4   | `tuning/space.yaml` + generate Python asserts and C++ `tunable_is_valid` | M      | Kills cross-language drift                                     |
| 5   | L1 resource model + CI validation vs real builds; generate mapping table | M      | Retires build-and-filter; makes PERF-003 impossible            |
| 6   | Tuning DB + record schema; port existing results                         | S      | Persistence                                                    |
| 7   | L3 cost model, calibrated and validated against the DB                   | M      | Pruned search                                                  |
| 8   | L6 selection policy; retire the stub                                     | M      | Production selection                                           |
| 9   | L7 orchestrator; retire the three config scripts                         | M      | Removes hand-partitioning                                      |

Steps 1–2 are a few dozen lines and should land before any further benchmarking, because every performance claim currently in the repo was produced by the mechanism they fix.

---

## 5. Risks and open questions

- **The L1 resource model must exactly match the generator.** If it under-predicts, valid configurations get silently pruned — the same failure mode as today's mapping table, just faster. Mitigation: CI assembles a sampled subset and fails on any prediction mismatch. This is non-negotiable; without it L1 is a liability.
- **Cost-model calibration is data-dependent.** The 2.06× zero-operand effect means the model's WMMA constant is only valid for realistic operand distributions. Real workloads with sparse or quantized activations may sit anywhere between 2157 and 4440 TFLOPS. The model should carry an explicit operand-distribution assumption.
- **DB rot.** Keying on `generator_git` is correct but coarse — an unrelated codegen commit invalidates everything. A per-kernel content hash of the emitted assembly would be tighter; worth doing if invalidation churn becomes painful.
- **Assumptions I made rather than asked about:** Python + YAML for L0 (matches the repo's stdlib-only, no-package-manager constraint); JSONL for the DB (greppable, diffable, no sqlite dependency); nearest-neighbour rather than a learned model for L6 (the DB will be small for a long time, and an interpretable selector is easier to debug than a regressor). Any of these is cheap to revisit.
- **Unresolved from the investigation and material here:** bwd and wrw generators remain unaudited, and their gaps (1.72× / 2.06×) are worse than fwd's (1.17×). If their tunable spaces have direction-specific constraints I haven't seen — the wrw stream-K path especially — L0's spec will need per-direction sections. I'd want to read those two generators before finalizing the L0 schema.