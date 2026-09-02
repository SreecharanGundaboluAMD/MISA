# MISA Tuning Pipeline — Follow-Up Work Plan

**Status:** handoff document for other agents.
**Supersedes the consumer assumption in** `tuning_rewrite.md` (repo root).
**Prerequisite reading:** `tuning_rewrite.md` (the L0–L7 layer design), `opus_review.md` (the COR-/PERF- findings this all derives from).

---

## 0. The correction that reframes the whole design

`tuning_rewrite.md` was written assuming `conv_driver.exe` is the thing that selects kernels. **It is not.** MIOpen is the consumer. `conv_driver.exe` is a developer benchmark harness that happens to contain a selection loop; nothing in production ever runs it.

That changes the target of L2 (applicability), L3 (cost model), L5 (tuning DB) and L6 (selection policy). Those four layers must land **inside MIOpen**, or feed MIOpen data, or they are dead code. L0/L1/L4/L7 stay in MISA and are unaffected.

### What MIOpen actually does with MISA output — confirmed

| Fact | Evidence |
|---|---|
| MISA kernels ship as **pre-assembled `.s` + `.inc` text checked into MIOpen** — 5,964 `.s`, 34 `.inc` | `~/rocm-libraries/projects/miopen/src/kernels/dynamic_igemm/igemm_gtc_xdlops_nhwc_<arch>/<dir>_<prec>/` |
| They are glob-compiled at MIOpen build time; no MISA dependency exists in MIOpen's build | `miopen/src/CMakeLists.txt:362-364` (`file(GLOB_RECURSE ... "kernels/dynamic_igemm/*.s")`) |
| The export step is a shell script in *this* repo that runs split-kernel codegen and `cp`s the output into a directory named exactly like MIOpen's | `script/gen_gfx950_conv_split_kernel.sh` (10 configs → `igemm_gtc_xdlops_nhwc_gfx950/{dir}_{prec}/`) |
| MIOpen's tunable list is a **hand-transcribed C++ copy** of MISA tunables — ~260 brace-init entries | `GetFwdXdlopsNHWCConfigList()`, `conv_asm_implicit_gemm_gtc_fwd_nhwc.cpp:51-318` |
| Kernel selection = reconstruct the mangled filename from the perf config | `ToKernelName`, `conv_asm_implicit_gemm_gtc_perf_config.cpp:261-299`; used at `conv_asm_implicit_gemm_gtc_fwd_nhwc.cpp:984-990` |
| **Default** path is a padding-minimizing heuristic, not a sweep | `HeuristicInit` → `HeuristicInitMacroTileNoPadGemmK(...)` exact-match scan, else `find_with_gemm_k_pad()` minimizing `ComputeMatrixPadSize` (`conv_asm_implicit_gemm_gtc_fwd_nhwc.cpp:525-640`) |
| Exhaustive enumeration happens **only in the spare set** | `SetNextValue`, `conv_asm_implicit_gemm_gtc_fwd_nhwc.cpp:636-666` — main set returns false immediately: *"always break generic search of main set (no spare), make sure we can use spare set"* |
| **gfx1250 is not accepted by any GTC asm solver** | `conv_asm_implicit_gemm_gtc_fwd_nhwc.cpp:903-905` gates on `gfx908 / gfx90a / gfx942 / gfx95*` only |

**Nuance on "MIOpen went through all the MISA kernels and picked the fastest."** That is true of the *tuning* run (`MIOPEN_FIND_ENFORCE=search`, spare set, results persisted to perf-db), and it is how the shipped perf-db entries were produced. It is **not** what a default Find call does — that uses `HeuristicInit` plus a perf-db lookup. Both paths matter and they need different follow-up work (Track C).

### The finding that changes sequencing

**MIOpen already serves gfx1250 convolution — via `hipconv`, not MISA.**

- `miopen/src/solver/conv/conv_hipconv.cpp` (456 lines), registered as a first-class tunable solver at `miopen/src/solver.cpp:813`.
- `miopen/src/hipconv/src/arch/cdna5/` — a full CDNA5 backend (`direct`, `grouped`, `depthwise`, `tdm_desc.h`, `bunnies_mi400.hpp`).

So MISA gfx1250 is not filling a vacuum; it has an incumbent inside the same library. Two consequences for this plan:

1. **`ConvHipConv` is the integration template to copy.** Its perf config has exactly *one* tuned field — an index into a library-ranked config list:
   ```cpp
   struct PerformanceConfigConvHipConv : PerfConfigBase<PerformanceConfigConvHipConv>
   { int index = -1; /* ... */ };          // solvers.hpp:4654-4686
   ```
   Ranking, applicability and workspace all delegate to the library (`hipconv::get_valid_configs / find_config / is_applicable / get_workspace_size / get_weighted_throughput_index`, `hipconv/include/hipconv/hipconv.hpp:56-114`). This is precisely the shape that makes the hand-transcribed `kernel_param_list` unnecessary — and it is *already merged and in production in MIOpen*, so it is not a speculative design.
2. **A MISA gfx1250 solver must justify itself against hipconv per shape**, not against MIOpen-as-a-whole. `GetWti` is the arbitration mechanism outside Find.

---

## 1. Prerequisite: verify the landed action-plan commit

Commit `16bbbfa` — *"gfx1250 WMMA: implement 11-item prioritized action plan from hardware review"* — claims all 11 items, 106 files, +1490/−1178. Its 10 numbered bullets map to action-plan items 1, 3, 4, 5, 6, 7, 8, 9, 10, 11; item 2 (fp16 256× mapping rows) predates it.

**W0 — Independently verify `16bbbfa` before building on it.** Owner: any agent. Effort: S.

Do not take the commit message as evidence. Specifically re-establish:

| Claim | How to check |
|---|---|
| Item 1: `-V 0` init fixed | Buffers randomized+copied unconditionally in `conv_driver.cpp`; then confirm `-V 0` and `-V 1` timings now agree within noise on a fp16 128×128 shape. The pre-fix gap was 0.099 vs 0.137 ms. |
| Item 1 second half: *"re-run every benchmark"* | **Probably not done.** Every TFLOPS/ratio number in `docs/` and in the benchmark scripts' output predates the fix and was measured on zero-filled operands (WMMA is 2.06× faster on zeros). Treat all of them as invalid until re-measured. |
| Item 4: pipelining | The commit says it is **gated off for single-buffered LDS** because it re-exposed the last-lane barrier race. So PERF-001's ~1.73× headroom is only realized when `lds_double_buffer=2`. Measure what fraction of shipped configs actually get it. |
| Item 5: tails + `acc_high_bank`/`epilogue_chunked` | This is the PERF-002 blocker. Re-run the previously-rejected 256×256 non-divisible shape and confirm `valid:y` **and** a win, not just that it builds. |
| Still missing: `result_t.valid` correctness gate | The commit does not list it. If `is_valid` is still a local in `fwd_post` (`conv_driver.cpp:1003-1080`), the selection loop still cannot know whether its winner is correct — this is migration step 1 in `tuning_rewrite.md` and the highest-value remaining fix. **Check this first.** |

Output: a short `docs/action_plan_verification.md` with per-item Confirmed / Partial / Not-done and the measurement that establishes it.

---

## 2. Work tracks

Each item is written to be picked up independently. Format: **Goal / Evidence / Deliverable / Acceptance / Depends on / Effort.**

### Track A — The MISA→MIOpen export contract

Today this contract is a `cp` in a shell script plus a human retyping tunables into C++. Everything else in this plan is blocked on making it a real, checkable interface.

---

**A1 — Write down the contract as it exists today.**

- **Goal:** one document that states exactly what MIOpen consumes, so later items can change it deliberately rather than by accident.
- **Evidence:** the table in §0. Also `script/gen_gfx950_conv_split_kernel.sh` (the only export path that exists) vs `script/gen_gfx1250_conv.sh` (same shape, but emits into `igemm_gtc_wmma_nhwc_gfx1250/` — a directory MIOpen has no knowledge of, and note it does **not** pass `-s`/split-kernel in its filename convention the way the gfx950 script's target layout requires).
- **Deliverable:** `docs/misa_miopen_contract.md` covering: directory layout, the `.s`/`.inc` split and why `.inc` files are shared, the glob in `src/CMakeLists.txt:362-364`, the filename↔`ToKernelName` correspondence, and the four places kernel naming is independently implemented.
- **Acceptance:** a reader can, from the doc alone, regenerate MIOpen's `igemm_gtc_xdlops_nhwc_gfx950/` tree from this repo and diff it byte-for-byte against what is checked in. **Actually do that diff and report the result** — if it does not reproduce, that itself is a finding (drift between what MIOpen ships and what MISA currently generates).
- **Depends on:** nothing.
- **Effort:** S.

---

**A2 — Generate MIOpen's tunable list instead of transcribing it.**

- **Goal:** kill `GetFwdXdlopsNHWCConfigList()`-style hand-transcription as a class of bug.
- **Evidence:** `conv_asm_implicit_gemm_gtc_fwd_nhwc.cpp:51-318` is ~260 brace-init entries, each a manual copy of a MISA tunable, with a bwd and wrw twin. Nothing checks that this list matches the `.s` files shipped alongside it. MISA already has an export hook: `igemm_codegen.py -output` (lines 78-86, 119-132) → `igemm_out_tunable_param()`.
- **Deliverable:** extend the `-output` path to emit a machine-readable manifest (JSON) per built config: one record per emitted kernel with the full canonical tunable, the mangled name, the emitted `.s`/`.inc` filenames, and the resource facts the assembler reported (`.amdhsa_next_free_vgpr`, LDS bytes, `.amdhsa_accum_offset` where applicable). Then a generator that turns the manifest into the C++ list.
- **Acceptance:** regenerating the gfx950 list from the manifest reproduces the current `kernel_param_list` semantically (order may differ; add a checker that compares as sets). Any discrepancy is reported, not silently reconciled — a discrepancy means MIOpen is currently advertising a tunable it does not ship, or vice versa, which is worth knowing on its own.
- **Depends on:** A1.
- **Effort:** M.
- **Note:** the manifest is also the natural L0/L1 validation artifact from `tuning_rewrite.md` §L1 — the "CI assembles a sampled subset and fails on prediction mismatch" requirement is just a diff against this manifest's recorded VGPR/LDS numbers. Build it once, use it twice.

---

**A3 — Single source of truth for kernel-name mangling.**

- **Goal:** four implementations → one generator plus three generated consumers.
- **Evidence:** the four copies are `igemm_gtc_encode_kernel_name` (`python/igemm/igemm_base.py`), the C++ twin in `driver/igemm_gtc_base.h`, `ToKernelName` (`conv_asm_implicit_gemm_gtc_perf_config.cpp:261-299`), and the on-disk filename convention. Prior incident: `direct_store` was absent from one copy since Phase 59 and `hipModuleGetFunction` silently ran the wrong kernel. Item 7 of the action plan just added two *more* fields (`wmma_epilogue_chunked`, `wmma_acc_high_bank`) to two of the four copies — MIOpen's copy does not know about either.
- **The schema mismatch that must be resolved here:** MIOpen's `ToKernelName` emits the XDLOPS schema — arch infix `_gtcx`/`_gtcx2`/`_gtcx3`/`_gtcx35`, then `_wt<m>x<n>x<k>_ws<m>x<n>_wr<m>x<n>`. MISA's WMMA schema is `_gtcw`, `_wt16x16` (two dims, no K), `_wr<m>x<n>`, **no `ws` field at all**, plus WMMA-only suffixes. These are not reconcilable by adding an arch infix; `ToKernelName` needs a WMMA branch.
- **Deliverable:** name mangling defined once (in the L0 spec / manifest), with the C++ driver's copy and MIOpen's `ToKernelName` both generated or both checked against it in CI.
- **Acceptance:** a test that, for every kernel in the A2 manifest, asserts MIOpen's `ToKernelName` on the corresponding perf config produces the shipped filename. This test would have caught the Phase 59 `direct_store` bug on the day it landed.
- **Depends on:** A2.
- **Effort:** M.

---

### Track B — Land gfx1250 in MIOpen

---

**B1 — Choose and document the integration shape. Do this before writing any solver code.**

- **Goal:** decide between (a) extending the existing `ConvAsmImplicitGemmGTC*XdlopsNHWC` solvers with a WMMA branch, (b) a new `ConvAsmImplicitGemmGTCWmmaNHWC` solver family, (c) the `ConvHipConv` index-into-ranked-list pattern.
- **Evidence for (c), which is the recommendation:**
  - `PerformanceConfigConvHipConv` carries one int (`solvers.hpp:4654-4686`); all knowledge lives library-side. Adding a WMMA knob later costs zero MIOpen changes.
  - By contrast `PerformanceConfigAsmImplicitGemmGTC`'s constructor takes ~24 positional parameters (`solvers.hpp:2785-2838`) — every new knob is a signature change across three direction subclasses, and it currently has **no** field for any WMMA knob: `wmma_tile_m/n`, `wmma_repeat_m/n`, `wmma_acc_high_bank`, `wmma_epilogue_chunked`, `wmma_m/n/k_tail`, `lds_double_buffer`, `main_loop_interleave`, `local_prefetch_num`, `direct_store`, `saddr_global_load`, `tdm_global_load`, `wrw_streamk`, `wmma_setprio`, `gsplit_stagger`.
  - `hipconv.hpp`'s `matches_descriptor()` (lines 96-107) is an existing, shipped answer to "how do I name a config without putting every knob in MIOpen's header": a comma-separated `key=value` descriptor string that the kernel family interprets, and *rejects tokens it does not understand*. MISA's tunable dict maps onto this almost directly.
- **Deliverable:** a decision record with the trade-offs, including the cost of (c): it implies MISA output is consumed through a small C++ shim library rather than as loose `.s` files, which is a build-system change on MIOpen's side. Weigh that against the fact that (a) and (b) both require growing the 24-parameter constructor by ~15 fields and adding a fourth-copy naming branch.
- **Acceptance:** a decision, with the rejected options' costs written down. Everything in B2–B4 is conditional on this.
- **Depends on:** A1. Should be informed by A3's schema-mismatch finding.
- **Effort:** S (analysis), but it is the highest-leverage decision in the plan.

---

**B2 — Implement the chosen perf config + solver.**

- **Goal:** a gfx1250 MISA solver MIOpen can call.
- **Deliverable (if B1 picks (c)):** a `PerformanceConfigConvAsmImplicitGemmMisaWmma { int index; }` plus a MISA-side ranked-config query, mirroring `conv_hipconv.cpp:108-160` including the `config_count` caching comment — that comment documents a real `generic_search.hpp` ordering constraint (`ComputedIterator` calls `IsValid` before `SetNextValue`, and `SetNextValue` has no `ExecutionContext`) that any new tunable solver will hit.
- **Deliverable (if B1 picks (a)/(b)):** the ~15 missing WMMA fields in the perf config, a WMMA branch in `ToKernelName`, WMMA-aware `HeuristicInit`, and `SetNextValue` coverage.
- **Acceptance:** `MIOPEN_FIND_ENFORCE=search` on gfx1250 selects a MISA kernel for at least one shape and the result verifies against MIOpen's own reference. **Not** "it compiles."
- **Depends on:** B1, A2, A3.
- **Effort:** L.

---

**B3 — Arch gate, registration, solver ID.**

- **Goal:** the solver is reachable.
- **Evidence:** the gate to amend is `conv_asm_implicit_gemm_gtc_fwd_nhwc.cpp:903-905` (and the bwd/wrw twins). Registration precedent: `miopen/src/solver.cpp:813` (`RegisterWithSolver(registry, ++id, conv::ConvHipConv{}, miopenConvolutionAlgoDirect)`), plus the two `mlo_dir_conv.cpp` inclusion sites (lines 65, 207).
- **Acceptance:** solver appears in `miopendriver`'s solver list on a gfx1250 machine; `IsApplicable` returns true for the intended shape class and false elsewhere.
- **Depends on:** B2.
- **Effort:** S.
- **Caution:** solver IDs are persisted in perf-db and find-db. Adding one is an append-only operation; do not renumber.

---

**B4 — Coexistence with `ConvHipConv` on gfx1250.**

- **Goal:** decide what happens when both MISA and hipconv are applicable to the same shape, in both Find mode and immediate mode.
- **Evidence:** Find benchmarks and ignores `GetWti`. Immediate mode ranks by `GetWti` — hipconv returns `hipconv::get_weighted_throughput_index(...)` (`conv_hipconv.cpp` `GetWti`, and see the comment above it: hipconv deliberately under-scores its small-channel fallback so other solvers can outrank it). A MISA solver returning a default `GetWti` will lose or win arbitrarily.
- **Deliverable:** a `GetWti` implementation grounded in the L3 cost model (Track C1), plus a documented head-to-head on a representative shape set.
- **Acceptance:** a table of shapes × {MISA, hipconv} × measured ms, and a statement of which shape classes MISA should claim. This is also the honest answer to "is the gfx1250 MISA work worth shipping" — collect it early even if B2 is incomplete, using `conv_driver.exe` vs `MIOpenDriver` (see D2).
- **Depends on:** B3 for the real version; a preliminary version can be done today with the two drivers.
- **Effort:** M.

---

### Track C — Re-point the tuning layers at MIOpen

This is the part of `tuning_rewrite.md` that changes most. **L6 "Selection Policy" as written — a generated table shipped with `conv_driver.exe`, replacing the `heuristic_select_kernel` stub at `driver/igemm_gtc_base.h:1002` — should be dropped.** MIOpen already owns selection, with a persisted perf-db and a Find mechanism. MISA's job is to supply the *data* and the *heuristic*, not the mechanism.

---

**C1 — L3 cost model as MIOpen's heuristic, not as a driver-side selector.**

- **Goal:** replace/augment `HeuristicInitMacroTileNoPadGemmK`'s padding-minimization with a model that knows about memory traffic.
- **Evidence:** the current heuristic (`conv_asm_implicit_gemm_gtc_fwd_nhwc.cpp:525-640`) scans for an exact `m/n/k` tile match and otherwise minimizes `ComputeMatrixPadSize`. It has no notion of bandwidth, arithmetic intensity, or occupancy. The investigation supplies calibrated constants: HBM 11.4 TB/s at 1 GB working set (48.5 at 4 MB, 33.0 at 64 MB), L2 4 MB, WMMA 2157 TFLOPS on random operands.
- **Deliverable:** the L3 model from `tuning_rewrite.md` §L3, implemented once (MISA-side, exported through the same shim as the config list) and consumed by MIOpen for both `HeuristicInit` ordering and `GetWti`.
- **Acceptance:** on a held-out shape set, the model's top-1 is within X% of the measured best more often than padding-minimization is. Pick and publish X before measuring.
- **Depends on:** B1 (determines where the model lives), E2 (needs valid measurements to calibrate against).
- **Effort:** L.
- **Explicit non-goal, restated:** the model prunes and orders. Measurement decides.

---

**C2 — L5 tuning DB → MIOpen perf-db.**

- **Goal:** MISA's tuning results reach production, which means reaching MIOpen's perf-db.
- **Evidence:** MIOpen persists tuning results keyed by solver ID + problem description. MISA's proposed JSONL record (`tuning_rewrite.md` §L5) has a compatible key set (`arch`, problem signature, tunable) plus fields MIOpen does not keep (`generator_git`, `sclk_mhz`, `power_w`, IQR).
- **Deliverable:** an exporter from the MISA tuning DB to MIOpen perf-db format, and the inverse importer so MIOpen's existing gfx950 perf-db entries can seed MISA's DB (they encode real tuning results nobody in MISA currently has).
- **Acceptance:** round-trip on the gfx950 entries: import → export → byte-identical. The importer's output is also immediately useful as a validation set for C1.
- **Depends on:** the MISA-side DB existing (`tuning_rewrite.md` migration step 6).
- **Effort:** M.
- **Note:** keep `generator_git` in the MISA DB even though MIOpen has no field for it. It is what makes results invalidatable when codegen semantics shift, and MIOpen's perf-db has no equivalent protection — which is a latent risk on MIOpen's side worth flagging to them separately.

---

**C3 — Reconcile applicability across the three implementations.**

- **Goal:** one applicability predicate, three consumers.
- **Evidence:** MISA has Python asserts (~25 rules across three generators), C++ `tunable_is_valid` (`driver/igemm_fwd_gtc_driver.h:450-503`), and MIOpen has `IsApplicable` + `PerformanceConfig::IsValid`. Three independent hand-written copies of overlapping logic.
- **Specific case to carry through:** the large-tile divisibility gate at `driver/igemm_fwd_gtc_driver.h:499-502`. Action-plan item 5 claims tails now work with `wmma_acc_high_bank`/`epilogue_chunked`; if so, **that gate must be relaxed correspondingly, on all three sides**, or the 256× tiles still will not dispatch and item 5's benefit is unrealized. Verify this as part of W0.
- **Deliverable:** applicability generated from the L0 spec into Python asserts, C++ `tunable_is_valid`, and the MIOpen predicate.
- **Acceptance:** a differential test — enumerate the cross product of (shipped tunables × a shape corpus) and assert all three implementations agree on every cell. Disagreements found this way are real bugs; report them rather than papering over them.
- **Depends on:** L0 existing (`tuning_rewrite.md` migration step 4).
- **Effort:** L.

---

### Track D — `conv_driver.exe`'s reduced role

---

**D1 — Demote the driver to a measurement harness (L4) and say so.**

- **Goal:** stop investing in driver-side selection.
- **Deliverable:** delete or explicitly deprecate `heuristic_select_kernel` (`driver/igemm_gtc_base.h:1002` — currently a stub returning an empty tunable). Keep and harden the L4 work from `tuning_rewrite.md`: `result_t.valid` correctness gate, unconditional buffer init, median+IQR instead of `min`, recorded sclk/power.
- **Acceptance:** the driver's selection loop either (i) is gated on validity and reports median/IQR, or (ii) is removed in favour of "measure every tunable, emit records, decide nothing."
- **Depends on:** W0 (to find out how much of this item 1 already landed).
- **Effort:** S.
- **Priority: high.** Every number Track C calibrates against comes from here.

---

**D2 — Cross-validate MISA driver measurements against MIOpenDriver.**

- **Goal:** establish that the two harnesses agree, before trusting either to arbitrate MISA vs hipconv.
- **Evidence needed because:** the two use different timing methodology, different warmup, different data initialization, and — as of pre-`16bbbfa` — MISA's `-V 0` measured zero-filled operands. The known benchmark trap (`script/benchmark_gfx1250_vs_miopen.py` using `-V 0` + `min(costs)`, letting invalid tunables win) is exactly this class of error.
- **Deliverable:** a harness that runs the same shape through both drivers and reports the delta, over a shape corpus.
- **Acceptance:** agreement within a stated tolerance, or a documented explanation of the systematic difference. Any MISA-vs-MIOpen performance claim published before this exists should be labelled provisional.
- **Depends on:** D1.
- **Effort:** M.

---

### Track E — Validation and re-baselining

---

**E1 — Re-measure everything post-`16bbbfa`.**

- **Goal:** a trustworthy baseline.
- **Evidence:** action-plan item 1 changed what the benchmark measures (random vs zero operands; the measured zero-operand speedup was 2.06×). Every TFLOPS figure and every "N× vs MIOpen" ratio in `docs/` and in script output predates it.
- **Deliverable:** a re-run of the standard shape corpus with `-V 1`, median+IQR, on the current HEAD, published as *the* baseline with the commit hash attached.
- **Acceptance:** old numbers are either reproduced or explicitly retracted. Retraction is a fine outcome; silently leaving them in place is not.
- **Depends on:** D1.
- **Effort:** S (mostly machine time).
- **Priority: high**, and it gates C1's calibration.

---

**E2 — Audit `bwd` and `wrw`, which were never reviewed.**

- **Goal:** close the biggest remaining evidence gap.
- **Evidence:** the original investigation audited `igemm_fwd_gtc_wmma_nhwc.py` only. The measured gaps are worse for the unaudited directions — bwd 1.72×, wrw 2.06×, vs fwd 1.17×. `16bbbfa` touched bwd (+11 lines) and wrw (+10) only incidentally, versus fwd's +118, so the action plan did not close this either. The wrw stream-K path is entirely unexamined.
- **Deliverable:** the same treatment fwd got — prologue/epilogue read, generated assembly inspected, ablation-based cost attribution.
- **Acceptance:** COR-/PERF- findings for bwd and wrw at the same evidentiary standard (Confirmed requires source, generated-code, measured, or ISA evidence).
- **Depends on:** E1 for valid measurements.
- **Effort:** L.
- **Blocks:** the L0 schema. If bwd/wrw have direction-specific constraints not present in fwd — the stream-K path especially — L0 needs per-direction sections and it is cheaper to know that before writing the schema than after.

---

**E3 — MIOpen-side CI.**

- **Goal:** the contract stays true.
- **Deliverable:** the A3 name test and the C3 differential applicability test wired into whatever CI covers `src/kernels/dynamic_igemm/`, plus a check that every advertised tunable has a corresponding shipped `.s`.
- **Acceptance:** deliberately breaking one copy of the name mangling fails CI.
- **Depends on:** A3, C3.
- **Effort:** M.

---

## 3. Sequencing

```
W0  verify 16bbbfa ────┬─────────────────────────────────────────────┐
                       │                                             │
D1  driver → L4 ───────┴──> E1 re-baseline ──> C1 cost model ──┐     │
                                     │                         │     │
A1 contract ──> A2 manifest ──> A3 naming ──> B1 decision ──> B2 solver ──> B3 gate ──> B4 vs hipconv
                                     │                         │
                                     └──> C2 perf-db ──────────┘
L0 spec ──> C3 applicability ──> E3 CI
E2 bwd/wrw audit ──> (feeds back into L0 schema)
```

**Start here, in this order, and they can run in parallel:**

1. **W0** — nothing downstream is trustworthy until the landed commit is independently verified. In particular find out whether `result_t.valid` exists yet.
2. **D1 + E1** — a valid baseline. Cheap, and every later number depends on it.
3. **A1** — the contract document, including the byte-for-byte regeneration diff. Cheap, and it is the prerequisite for the whole MIOpen track.
4. **B1** — the integration-shape decision. Analysis only, but it determines the shape of B2/C1/C2.

**Do not start** B2 before B1, or C1 before E1.

---

## 4. Explicit non-goals and things not to do

- **Do not build a driver-side selection table.** L6 as written in `tuning_rewrite.md` is superseded; MIOpen owns selection.
- **Do not add WMMA knobs to `PerformanceConfigAsmImplicitGemmGTC` before B1 decides.** That constructor is already 24 positional parameters across three subclasses; growing it is a one-way door.
- **Do not publish MISA-vs-MIOpen or MISA-vs-hipconv numbers before D2.** The existing ones were produced by a harness with a known systematic error.
- **Do not treat `16bbbfa`'s commit message as evidence** (per the standing evidence rules — a commit message is a claim, not a measurement).
- **Do not renumber MIOpen solver IDs.** They are persisted in shipped databases.

---

## 5. Open questions requiring information this repo does not contain

These need a human or a MIOpen-side owner; agents should not guess.

1. **Is MISA on gfx1250 intended to ship at all, given hipconv/CDNA5 already exists in MIOpen?** The answer determines whether Track B is production work or an evaluation exercise. B4's head-to-head data is the input to that decision, and is worth collecting regardless.
2. **Who owns the MIOpen-side change?** MISA and MIOpen are separate repos with separate review; Track B's items are MIOpen PRs, not MISA commits.
3. **Is the `.s`-checked-in delivery model negotiable?** B1 option (c) implies a shim library, which is a build-system change MIOpen may not accept. If it is non-negotiable, B1 collapses to (a)/(b) and A3's schema work becomes mandatory rather than advisory.
4. **Does an MI400/gfx1250 perf-db exist anywhere?** If MIOpen has already tuned hipconv on this part, those records are the best available validation set for C1 and should be imported (C2's inverse direction).
5. **What is the target shape corpus?** Every acceptance criterion above says "a shape corpus" without defining one. The corpus should be agreed once and reused across E1, B4, C1 and D2 — otherwise the numbers are not comparable and the whole plan's evidence base fragments.

---

## 6. Evidence index

Every non-obvious claim above, with its source. Re-verify before relying on any of these; the repos move.

| Claim | Source |
|---|---|
| MIOpen ships 5,964 `.s` + 34 `.inc` MISA kernels | `miopen/src/kernels/dynamic_igemm/` |
| Glob-compiled at build | `miopen/src/CMakeLists.txt:362-364` |
| Hand-transcribed tunable list | `miopen/src/solver/conv/conv_asm_implicit_gemm_gtc_fwd_nhwc.cpp:51-318` |
| Padding-minimizing default heuristic | same file, 525-640 |
| Exhaustive search only in spare set | same file, 636-666 |
| No gfx1250 in GTC asm solvers | same file, 903-905 |
| Name reconstruction → `.s` filename | same file, 984-990; `conv_asm_implicit_gemm_gtc_perf_config.cpp:261-299` |
| GTC perf config: 24 positional params, zero WMMA fields | `miopen/src/include/miopen/conv/solvers.hpp:2785-2838` (fwd), `:2989-` (bwd) |
| hipconv CDNA5 backend exists | `miopen/src/hipconv/src/arch/cdna5/` |
| hipconv solver registered | `miopen/src/solver.cpp:813`; `mlo_dir_conv.cpp:65,207` |
| hipconv perf config = one index | `solvers.hpp:4654-4686` |
| hipconv delegation API | `miopen/src/hipconv/include/hipconv/hipconv.hpp:56-114` |
| `matches_descriptor` config-spec mechanism | same header, 96-107 |
| `ComputedIterator` ordering constraint | `conv_hipconv.cpp` `IsValid` comment (~line 138-153) |
| MISA export script (gfx950) | `script/gen_gfx950_conv_split_kernel.sh` |
| MISA export script (gfx1250, no MIOpen target) | `script/gen_gfx1250_conv.sh` |
| MISA tunable export hook | `igemm_codegen.py:78-86,119-132` |
| `heuristic_select_kernel` is a stub | `driver/igemm_gtc_base.h:1002` |
| `is_valid` never leaves the lambda | `driver/conv_driver.cpp:1003-1080`; selection at `:562,573,585` |
| Large-tile divisibility gate | `driver/igemm_fwd_gtc_driver.h:497-502` |
| Action-plan commit | `16bbbfa`, 106 files, +1490/−1178 |
