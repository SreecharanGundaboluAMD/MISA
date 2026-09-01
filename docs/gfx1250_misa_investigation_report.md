# gfx1250 MISA Bring-up: Engineering Investigation Report

**Branch:** `users/SreecharanGundaboluAMD/gfx1250_bringup` @ `29818ef`
**Date:** 2026-09-01
**Scope:** Correctness and performance audit of gfx1250 WMMA convolution kernel code generation vs gfx950 XDLOPS, with autotuning design. `docs/` excluded per task rules.

---

## 1. Executive Summary

### Top findings (correctness risks separated from performance opportunities)

**Correctness risks:**
1. **[COR-001]** fp32 WMMA (`v_wmma_f32_16x16x4_f32`) silently corrupts at high occupancy without `lds_double_buffer=1` — confirmed defect, documented in AGENTS.md. All fp32 configs already set this, but the risk of new configs missing it remains.
2. **[COR-002]** `saddr_global_load + wmma_n_tail` is a confirmed correctness bug for fwd (excluded from config generation but not fixed at code level) — `script/generate_all_configs.py:127`.
3. **[COR-003]** int8 split-K atomic epilogue used `global_atomic_add_f32` (float bit-pattern reinterpretation) for int8 accumulators — code-level fix applied in Phase 57 (`global_atomic_add_u32`), but not yet hardware-validated with genuinely signed/large-magnitude int8 data — `igemm_base.py:414-434`.
4. **[COR-004]** `atomic_cascade=1` hangs on real hardware — hard-blocked by assert, but the code path exists and is a trap for future developers — `igemm_base.py:377-379`.

**Performance opportunities (ordered by expected impact):**
1. **[PERF-001]** Config coverage gap: gfx950 has 30+ tile shapes for fwd/fp16; gfx1250 has 5. This is the single largest contributor to the performance gap — many shapes have no well-fitting tile.
2. **[PERF-002]** WMMA main loop uses single-buffered LDS by default; XDLOPS defaults to double-buffered. Single-buffering forces a barrier stall every K-iteration with no overlap.
3. **[PERF-003]** K-substep drain loop (`emit_extra_substeps`) is sequential — substeps 1..N-1 have no load/compute overlap unless `main_loop_interleave=1` (opt-in, requires `lds_double_buffer=1`).
4. **[PERF-004]** `DISABLE_XDL_ARB_STALL` (SCHED_MODE bit[2]) is not used — prevents back-to-back WMMA issue, leaving WMMA pipeline bubbles.
5. **[PERF-005]** Register pressure (~235+ VGPRs for 128x128 fp16) limits occupancy to 1 wave/CU, capping parallelism.
6. **[PERF-006]** Epilogue uses per-lane scalar `global_store_dword` by default — `direct_store=1` skips the LDS reshuffle but is opt-in.
7. **[PERF-007]** WMMA instruction `cycle` field is `None` — codegen cannot make latency-aware scheduling decisions.
8. **[PERF-008]** wrw is 2-3.5x slower — primary path uses atomic-add split-K epilogue with scalar `global_atomic_add_f32`, no vectorization or packed-atomic optimization by default.
9. **[PERF-009]** wrw split-K benchmarking includes `hipMemset` (output zeroing for atomic accumulation) in every timed iteration — partially inflates wrw's apparent latency vs gfx950's non-atomic path.

### Limitations in available evidence
- No GPU access for runtime profiling — all performance conclusions are from static code analysis and existing benchmark results in the repo.
- WMMA instruction latency/throughput not available in the ISA doc prose (only hazard table); `cycle=None` in code.
- Three generator audit subagents (fwd/bwd/wrw) were still running when this report was written; their findings should be incorporated when available.
- Benchmark results compare MISA against MIOpen (which uses different solvers), not a controlled gfx1250-vs-gfx950 comparison on identical kernels.

---

## 2. Repository and Dispatch Map

### Convolution shape → kernel → config flow

```
.config file (INI)
  │
  ├─ [codegen] arch='gfx1250', code_object='cov3', mode='flat'
  │
  ├─ [igemm_{dir}_gtc] sections (one per kernel variant)
  │     │
  │     └─ igemm_codegen.py → igemm_flatten()
  │           │
  │           ├─ config_parser_t.parse() → config_content_t
  │           ├─ igemm_try_expand_tunable_content() (fan out list-valued tensor_layout)
  │           │
  │           └─ codegen_driver_t(mc, tunable_dicts)
  │                 │  __init__ dispatches on direction + 'wmma_tile_m' in tunable:
  │                 │    fwd → igemm_fwd_gtc_wmma_nhwc_t
  │                 │    bwd → igemm_bwd_gtc_wmma_nhwc_t
  │                 │    wrw → igemm_wrw_gtc_wmma_nhwc_t
  │                 │
  │                 └─ do_emit(): hsa_header → emit_global_macro (pass for gfx1250)
  │                       → emit_igemm_kernel (each kernel.emit_kernel_body())
  │                         → emit_kernel_prologue → emit_kernel_tap_loop → emit_kernel_epilogue
  │
  └─ do_compile(): compile_asm_t → clang++ -x assembler -mcpu=gfx1250 → .hsaco
```

**Runtime dispatch** (`conv_driver.exe`):
```
main() → hipModuleLoad(hsaco) → config_parser → igemm_gtc_tunable_from_config
  for each direction (forw bitmask):
    igemm_{dir}_gtc_t(module, ...)
      for each tunable:
        if tunable_is_valid(arg, tunable):  ← shape/config compatibility check
          driver->run() → hipModuleGetFunction → grid/block → launch_conv_driver
```

### Key dispatch files
| Component | File | Key symbols |
|---|---|---|
| Generator selection | `python/codegen_driver.py:48-86` | `codegen_driver_t.__init__` |
| Tunable contract | `python/igemm/igemm_base.py:206-603` | `igemm_gtc_tunable_parameter_t` |
| WMMA main loop | `python/operations/wmma_main_loop.py:184-542` | `wmma_main_loop_t.emit` |
| WMMA instruction table | `python/operations/wmma.py:76-106` | `v_wmma_f32_16x16x32_f16` etc. |
| WMMA tile mapping | `python/operations/wmma_mapping.py:311-428` | `ctrl_wmma_mapping_table` |
| C++ tunable | `driver/igemm_gtc_base.h:105-276` | `igemm_gtc_tunable_t` |
| C++ kernel name | `driver/igemm_gtc_base.h:434-624` | `igemm_gtc_encode_kernel_name` |
| C++ fwd validity | `driver/igemm_fwd_gtc_driver.h:398-504` | `tunable_is_valid` |
| C++ wrw validity | `driver/igemm_wrw_gtc_driver.h:253-339` | `tunable_is_valid` |
| Config generation | `script/generate_all_configs.py` | `is_valid`, `gen_combos` |

### How configurations are selected
1. `generate_all_configs.py` enumerates all valid flag combinations per tile shape (11 binary flags × ~5 tile shapes × 3 directions × 3-4 precisions), filtered by `is_valid()` which encodes ~25 mutual-exclusion rules.
2. `build_gfx1250_master_configs.py` unions per-tile configs into master `_all.config` files.
3. `build_and_filter_configs.py` builds each, drops sections that fail assembly (VGPR overflow).
4. At runtime, `conv_driver.exe` tries every tunable in the loaded config, calls `tunable_is_valid()` per shape, benchmarks valid ones, reports fastest.

**Silent fallback risk:** `tunable_is_valid()` returning `false` silently skips a kernel — no warning. If ALL kernels for a direction are invalid, the driver prints nothing for that direction. No explicit "no valid kernel found" error.

---

## 3. Correctness Findings

### COR-001: fp32 WMMA requires lds_double_buffer=1 unconditionally
- **Severity:** Critical
- **Confidence:** Confirmed (documented in AGENTS.md, standalone repro exists)
- **Evidence:** `AGENTS.md` Known Issues; `igemm_base.py:246` (defaults to 0); `wmma.py:106` (`v_wmma_f32_16x16x4_f32`, K=4)
- **Mechanism:** `s_barrier_signal`/`s_barrier_wait` doesn't reliably fence a wave's last lane's LDS write at high occupancy. Single-buffered LDS with the tight fp32 K=4 loop exposes this race.
- **Trigger:** fp32 WMMA with `lds_double_buffer=0` at ~1500+ workgroups (64x64) or ~12000+ (128x128).
- **Why tests miss it:** Small/benchmark-scale shapes pass. Only fails at production-scale occupancy.
- **Fix:** All fp32 configs already set `lds_double_buffer=1`. Enforce via assert:
  ```python
  if self.precision == 'fp32' and self.fma_type == 'WMMA':
      assert self.lds_double_buffer, "fp32 WMMA requires lds_double_buffer=1"
  ```
- **Risk:** None — assertion-only, catches misconfiguration at codegen time.

### COR-002: saddr_global_load + wmma_n_tail is broken for fwd
- **Severity:** High
- **Confidence:** Confirmed (hardware A/B in `generate_all_configs.py:118-127`)
- **Evidence:** `script/generate_all_configs.py:118-127` — excluded from config generation but not fixed at code level
- **Mechanism:** Likely fwd's B-operand N-boundary address computation doesn't account for saddr's different addressing path. Root cause not diagnosed.
- **Trigger:** fwd + `saddr_global_load=1` + `wmma_n_tail=1` on a shape with `gemm_n % gemm_n_per_block != 0`.
- **Why tests miss it:** Only excluded from combinatorial config generation; no assert prevents manual creation of this combination.
- **Fix:** Add assert in `igemm_fwd_gtc_wmma_nhwc.py.__init__`:
  ```python
  assert not (tunable.saddr_global_load and tunable.wmma_n_tail), \
      "saddr_global_load + wmma_n_tail is broken for fwd (confirmed valid:n on hardware)"
  ```
- **Risk:** None — prevents shipping a known-broken combination.

### COR-003: int8 split-K atomic epilogue (code-level fix applied, not HW-validated)
- **Severity:** Medium
- **Confidence:** Code fix applied (Phase 57), but not hardware-validated with signed/large data
- **Evidence:** `igemm_base.py:414-434` — `global_atomic_add_u32` now used for int8/int4
- **Mechanism:** Previously used `global_atomic_add_f32`, which is bit-exact for small non-negative integers (subnormal float range) but silently corrupts for negative or large-magnitude int8 accumulators.
- **Trigger:** int8/int4 + `gemm_k_global_split=1` with signed activations/weights or accumulators > 8.39M.
- **Why tests miss it:** Small positive test values pass (subnormal coincidence).
- **Fix:** Already applied at code level. Needs hardware validation with genuinely signed/large int8 data.
- **Risk:** None for the fix itself; remaining risk is unvalidated hardware behavior.

### COR-004: atomic_cascade=1 hangs on real hardware
- **Severity:** Medium (blocked, but a trap)
- **Confidence:** Confirmed (hangs on hardware)
- **Evidence:** `igemm_base.py:377-379` — `assert not self.atomic_cascade`
- **Mechanism:** Cascading atomic defers completion to a subsequent release that's never issued; `s_wait_storecnt 0x0` never completes.
- **Trigger:** `atomic_cascade=1` (currently blocked by assert).
- **Fix:** Assert is correct. The TODO in `igemm_base.py:328-376` documents the required release mechanism.
- **Risk:** None while assert is in place.

### COR-005: WMMA hazard avoidance relies on hardware arbiter stall only
- **Severity:** Low (correctness OK, but performance-limiting)
- **Confidence:** Hypothesis (based on ISA reading)
- **Evidence:** ISA §5.7.2.1 (line 2499-2507); `wmma_main_loop.py:238-277` (`emit_wmma_tile` — no V_NOPs between WMMA issues)
- **Mechanism:** The code issues back-to-back WMMA instructions without V_NOPs or S_DELAY_ALU. Correctness is maintained because the hardware arbiter stalls the wave between multicycle WMMA issues (default behavior). But this prevents back-to-back issue.
- **Trigger:** Always (every WMMA kernel).
- **Fix:** Not a correctness fix — see PERF-004 for the performance implication.

---

## 4. Performance Findings

### PERF-001: Config coverage gap (30+ vs 5 tile shapes)
- **Evidence:**
  - gfx950 fwd/fp16: 4500 lines, 30+ unique tile shapes (256x128, 256x64, 128x256, 64x256, 128x128, 64x64, 32x128, etc.)
  - gfx1250 fwd/fp16: 597 lines, 5 unique tile shapes (128x128, 128x64, 64x128, 64x64, all K=32/64)
  - gfx950 includes K=8,16,32,64,128; gfx1250 only K=32,64
  - gfx950 includes 256-wide tiles; gfx1250's largest is 128 (except experimental 256x256 f16acc)
- **Hypothesis:** Many convolution shapes land on poorly-fitting tiles, causing tail overhead, wasted work, or insufficient parallelism.
- **Confirmation experiment:** Build gfx1250 with additional tile shapes (256x128, 128x256, 64x256) and benchmark on shapes where current tiles are suboptimal.
- **Expected impact:** High — directly addresses the 1.2-3.5x slowdown.
- **Confounding factors:** VGPR pressure may prevent some larger tiles; WMMA's block_size==macro_tile constraint limits factorizations.
- **Implementation:** See §7 (MISA improvement plan).

### PERF-002: Single-buffered LDS by default (vs XDLOPS double-buffered)
- **Evidence:**
  - `wmma_main_loop.py:65` — `self.lds_buffer_num = 1` (default)
  - `main_loop_graph.py:42` — `self.lds_buffer_num = 2` (XDLOPS default)
  - `igemm_base.py:246` — `lds_double_buffer` defaults to 0
- **Hypothesis:** Single-buffering forces a barrier stall every K-iteration with no global-load/compute overlap. The next tile's global load can't start until the current tile's LDS read is complete (same buffer).
- **Confirmation experiment:** A/B test `lds_double_buffer=0` vs `1` on a compute-bound shape.
- **Expected impact:** Medium-High — double-buffering hides global memory latency.
- **Implementation:** Make `lds_double_buffer=1` the default for gfx1250 WMMA configs (already required for fp32).

### PERF-003: Sequential k-substep drain loop
- **Evidence:** `wmma_main_loop.py:303-316` (`emit_extra_substeps`):
  ```python
  for ks in range(1, num_k_substeps):
      self._emit(f_sld_a(v_a(), v_sld_a_os(), off_a))
      self._emit(f_sld_b(v_b(), v_sld_b_os(), off_b))
      self._emit(f"s_wait_dscnt 0x0")
      emit_wmma_tile()
  ```
  Substeps 1..N-1 are pure sequential load→wait→compute with no overlap.
- **Hypothesis:** When `gemm_k_per_block > inst_wmma.k` (k-sub-loop), substeps 1..N-1 expose full LDS-read latency with no overlap.
- **Confirmation experiment:** Compare `gemm_k_per_block=32` (1 substep) vs `64` (2 substeps) with/without `main_loop_interleave=1`.
- **Expected impact:** Medium — only affects k-sub-loop configs (gemm_k_per_block > inst_wmma.k).
- **Implementation:** Make `main_loop_interleave=1` the default when k-sub-loop is active and `lds_double_buffer=1`.

### PERF-004: DISABLE_XDL_ARB_STALL not used
- **Evidence:**
  - ISA §5.7.2.1 (line 2504): `DISABLE_XDL_ARB_STALL` "allows a wave to declare that it wants to be able to issue multiple WMMA ops back to back"
  - ISA §5.7.2.1 (line 2506-2507): "This can block co-execution opportunities so it is likely beneficial primarily when a single wave is running on a SIMD"
  - `wmma_main_loop.py:238-277` — no `SCHED_MODE` modification
- **Hypothesis:** With occupancy=1 (which is the case for 128x128 fp16 at ~235 VGPRs), `DISABLE_XDL_ARB_STALL` would allow back-to-back WMMA issue, improving instruction throughput.
- **Confirmation experiment:** Set `SCHED_MODE` bit[2] in kernel prologue, benchmark WMMA-heavy shapes.
- **Expected impact:** Medium — only helps when occupancy=1 (which is common for large tiles).
- **Confounding factors:** May hurt when occupancy>1 (blocks co-execution).
- **Implementation:** Gate on occupancy=1; emit `s_setreg_b32 hwreg(SCHED_MODE), ...` in prologue.

### PERF-005: Register pressure limits occupancy to 1 wave/CU
- **Evidence:**
  - `igemm_fwd_gtc_wmma_nhwc.py:492-507` — VGPR allocation: v_c=128, v_a=32, v_b=32, v_gld_a=16, v_gld_b=16, plus addressing/temp = ~235+ VGPRs
  - gfx1250 has 512 VGPRs/CU (256 per bank × 2 banks), but WMMA uses bank 0 only unless `wmma_acc_high_bank=1`
  - At 235 VGPRs/wave × 4 waves/block = 940 VGPRs/block > 512 → only 1 block (4 waves) per CU
- **Hypothesis:** Occupancy=1 means no latency hiding from wave switching; all latency stalls are exposed.
- **Confirmation experiment:** Compare 128x128 (occupancy=1) vs 64x64 (occupancy=2+) on memory-bound shapes.
- **Expected impact:** High for memory-bound shapes.
- **Implementation:**
  - Use `wmma_acc_f16`/`wmma_acc_bf16` (num_v_c=4 instead of 8) to halve accumulator footprint
  - Use `wmma_acc_high_bank=1` to move v_c to bank 1 (frees bank 0 for more waves)
  - Use smaller tiles (64x64) for memory-bound shapes
  - Use `async_global_load=1` to eliminate v_gld_a/v_gld_b staging buffers

### PERF-006: Scalar epilogue store by default
- **Evidence:** `coalescing_store_wmma.py:44` — "a per-lane scalar global_store_dword is already contiguous across each 16-lane half-wave"
- **Hypothesis:** The LDS-reshuffle epilogue adds overhead (ds_write + ds_read + global_store) that `direct_store=1` skips.
- **Confirmation experiment:** A/B `direct_store=0` vs `1`.
- **Expected impact:** Low-Medium — epilogue is a small fraction of total time for large-K shapes.
- **Implementation:** Make `direct_store=1` the default (already an option).

### PERF-007: WMMA instruction cycle field is None
- **Evidence:** `wmma.py:76-106` — all `inst_wmma_t` entries have `cycle=None`
- **Hypothesis:** Without latency information, the codegen cannot make informed scheduling decisions (e.g., how many independent instructions to issue between WMMA dependencies).
- **Confirmation experiment:** N/A (static analysis — the field is unused)
- **Expected impact:** Low (indirect — limits future optimization)
- **Implementation:** Fill in cycle counts from ISA doc or measurement. ISA §5.7.2.1 mentions "16-cycle WMMA" as an example.

### PERF-008: wrw atomic-add epilogue is scalar
- **Evidence:**
  - `coalescing_store_wmma.py:52-59` — split-K uses scalar `global_atomic_add_f32`, "no wide/packed fp32 variant on this ISA"
  - `atomic_pack_bf16` exists (`coalescing_store_wmma.py:125-139`) but is opt-in and bf16-only
  - wrw is 2-3.5x slower than gfx950 (benchmark data)
- **Hypothesis:** Scalar atomic-add for split-K wrw creates L2 contention; `atomic_pack_bf16` halves atomic count but is not default.
- **Confirmation experiment:** A/B `atomic_pack_bf16=0` vs `1` for bf16 wrw split-K.
- **Expected impact:** Medium for bf16 wrw.
- **Implementation:** Make `atomic_pack_bf16=1` default for bf16 wrw split-K configs.

---

### PERF-009: wrw split-K benchmarking includes hipMemset in timing
- **Evidence:**
  - `driver/igemm_wrw_gtc_driver.h:967-978` — `wrw_gsplit_prolog` calls `hipMemset(p_wei, 0, ...)` to zero the output before every dispatch
  - `driver/igemm_gtc_base.h:714` — `ms += prolog_kernel()` inside the timed `launch_kernels` lambda
  - The `hipMemset` zeroes `group * (k/group) * (c/group) * y * x * sizeof(float)` bytes every iteration
  - For a typical wrw shape (e.g. 128 channels, 128 output, 3x3 filter): 128*128*9*4 = 589,824 bytes = 576 KB per zero
  - Additionally, the ternary search over split-K divisors (`time_split` lambda, line 990) does O(log(divisor_count)) real timed launches per config, each including the hipMemset
- **Hypothesis:** The `hipMemset` overhead is included in every timed iteration, inflating wrw's apparent latency. gfx950's non-atomic path doesn't have this cost. The ternary search multiplies this overhead.
- **Confirmation experiment:** Time `hipMemset` alone for typical wrw output sizes; compare to total wrw kernel time. Compare wrw with `gemm_k_global_split=0` (no memset needed) vs `=1`.
- **Expected impact:** Medium — partially explains the 2-3.5x wrw slowdown.
- **Implementation:** This is correct behavior (zeroing IS required for atomic accumulation). The benchmark should report kernel-only time separately. For production, a separate zeroing kernel could overlap with other work. Consider `wrw_reduction_kernel=1` (plain stores to workspace + separate reduction) to avoid the per-iteration memset.

## 5. gfx1250 vs gfx950 Comparison

### Implementation differences explaining the performance gap

| Factor | gfx950 XDLOPS | gfx1250 WMMA | Impact |
|---|---|---|---|
| **Tile shape diversity** | 30+ shapes (256x128, 256x64, 128x256, 64x256, etc.) | 5 shapes (128x128, 128x64, 64x128, 64x64) | **High** — many shapes have no good tile |
| **LDS double-buffering** | Default ON (`lds_buffer_num=2`) | Default OFF (`lds_buffer_num=1`) | **High** — no load/compute overlap |
| **K-subloop pipelining** | Graph-based scheduler with interleaving | Fixed sequential drain loop | **Medium** — substep latency exposed |
| **Register pressure** | AGPR separate from VGPR (accumulator in AGPR) | All in VGPR (v_c=128 for 128x128) | **High** — limits occupancy |
| **Wave size** | 64 lanes | 32 lanes (WMMA requires wave32) | **Medium** — half the threads per wave |
| **Matrix instruction** | MFMA (various M×N×K) | WMMA (fixed 16×16×32) | **Medium** — less tile flexibility |
| **Epilogue** | Vectorized coalescing store | Per-lane scalar store (or direct_store opt-in) | **Low-Medium** |
| **Config search space** | 4500 lines (fwd/fp16) | 597 lines | **High** — fewer candidates searched |
| **Main loop schedule** | `dotx_core_loop_graph` (DAG-based, ~10 schedules) | Single fixed schedule | **Medium** — no schedule tuning |
| **wrw split-K overhead** | Non-atomic or separate reduction | `hipMemset` in every timed iteration + ternary search | **Medium** — inflates wrw apparent latency |

### Key architectural difference: AGPR vs VGPR accumulator
gfx950's XDLOPS uses AGPR (Accumulator VGPR) for matrix results — physically separate from VGPR file. This means the accumulator (which is the largest register consumer) doesn't compete with v_a/v_b/addressing for VGPR space. gfx1250's WMMA accumulates directly in v_c (plain VGPR), so the 128-element accumulator for a 128x128 tile consumes half the 256-VGPR bank by itself.

### Key architectural difference: wave64 vs wave32
gfx950 uses wave64 (64 lanes/wave); gfx1250 WMMA requires wave32 (ISA §7.12: "WMMA instructions are supported only for wave32"). This means:
- Half the threads per wave → half the per-wave global load bandwidth
- Double the waves for the same block size → double the wave-management overhead
- But WMMA's 16×16 tile is computed cooperatively across all 32 lanes, so per-lane work is similar

---

## 6. Targeted ISA Findings

### Selected TOC sections and rationale

| ISA Section | Page | Why selected |
|---|---|---|
| §7.12 WMMA | 94 | Core instruction set used by all gfx1250 kernels |
| §7.12.1 WMMA data hazards | 98 | Correctness: back-to-back WMMA hazard requirements |
| §7.12.2 Matrix Element Storage in VGPRs | 99 | Correctness: lane/VGPR → (row,col,k) mapping |
| §5.7.2.1 DISABLE_XDL_ARB_STALL | 56 | Performance: back-to-back WMMA issue |
| §5.8 S_DELAY_ALU | 57 | Performance: software scheduling hints |
| §10.9 WMMA Matrix Load Ops with Transpose | 137 | Performance: native transpose-load for bwd/wrw |
| §11.2.4 LDS to VGPR Matrix Load with Transpose | 154 | Performance: `ds_load_tr16_b128` for bwd/wrw |
| §3.3.2 VGPRs | 16 | Correctness: VGPR limits, banking |

### Relevant architectural facts

1. **WMMA hazard requirements (§7.12.1):** Dense WMMA F16/BF16 with same A/B matrix (RAW) needs 5 V_NOPs (or 1 NOP + 4 coexec slots). WMMA_IU8 needs 9 V_NOPs. The code does NOT insert these — relies on hardware arbiter stall.

2. **DISABLE_XDL_ARB_STALL (§5.7.2.1):** Allows back-to-back WMMA issue. "likely beneficial primarily when a single wave is running on a SIMD" — which is exactly the occupancy=1 case for 128x128 tiles. Not used by MISA.

3. **S_DELAY_ALU (§5.8):** Optional software scheduling hint. "XDL WMMA ops (16-bit data and smaller) are tracked as if they were TRANS ops." Not used by MISA.

4. **WMMA capabilities (§7.12):**
   - "WMMA instructions are supported only for wave32" — confirms MISA's wave32 usage
   - "EXEC must be set to all 1's" — MISA's tail masking uses EXEC masking around stores, not WMMA issues (correct)
   - "Mode.Denorm is ignored, and denorms are preserved" — no denormal concern
   - RA/RB reuse bits: "hints that either matrix is likely to be used again in the next instruction and should be cached" — NOT used by MISA

5. **LDS to VGPR Matrix Load with Transpose (§11.2.4):** `ds_load_tr16_b128` — native hardware transpose-load for 16-bit elements, wave32-only. MISA's Phase 63 uses this for bwd/wrw transposed B operand (`wmma_mapping.py:182-244`, `get_gemm_index_for_src_matrix_transposed_ds_tr16`).

6. **WMMA Matrix Load Ops with Transpose (§10.9):** `global_load_tr16_b128` — global memory variant of transpose-load. Not used by MISA (could eliminate the LDS staging step for transposed operands).

### Missing or incorrectly implemented opportunities

1. **RA/RB matrix reuse bits (§7.12):** The ISA provides reuse hints for A/B matrices. MISA's `inst_wmma_t.__call__` does not emit them. When consecutive WMMA instructions share the same A or B matrix (which happens in the `wave_repeat_m × wave_repeat_n` loop), setting RA or RB could improve cache utilization.

2. **DISABLE_XDL_ARB_STALL (§5.7.2.1):** Not set. At occupancy=1, this would allow back-to-back WMMA issue. See PERF-004.

3. **S_DELAY_ALU (§5.8):** Not used. Could reduce pipeline stalls by hinting dependencies to hardware.

4. **global_load_tr16_b128 (§10.9):** Not used. For bwd/wrw's transposed operand, this could load+transpose directly from global memory into VGPRs, skipping the LDS staging step entirely.

5. **WMMA F32 16×16×4 F32 (§7.12):** The only pure-fp32 WMMA form has K=4 (vs K=32 for fp16/bf16). This means 8× more main-loop iterations for the same K, with 8× more barrier overhead. The fp32 path is inherently less efficient — consider whether fp32 conv should use a different approach (e.g., fp16 compute with fp32 accumulation via cast).

### ISA ambiguities
- WMMA instruction latency/throughput is not stated in the ISA doc prose. §5.7.2.1 mentions "16-cycle WMMA" as an example, but per-instruction cycle counts are not in the doc. The `cycle=None` in `wmma.py` reflects this.
- The exact coexec slot semantics (how many independent VALU instructions count as "coexec slots") are not fully specified.

---

## 7. MISA Convolution Improvement Plan

### Quick wins (1-3 days each, high confidence)

1. **Make `lds_double_buffer=1` the default for all gfx1250 WMMA configs.** Currently opt-in; XDLOPS defaults to ON. This is the single highest-impact change for memory-bound shapes. Requires verifying LDS size fits (double the single-buffer size).

2. **Make `direct_store=1` the default.** The LDS-reshuffle epilogue is unnecessary overhead — 16 consecutive lanes already cover 16 consecutive columns. Already an option, just not default.

3. **Add assert for COR-002** (saddr_global_load + wmma_n_tail for fwd). Prevents shipping a known-broken combination.

4. **Make `atomic_pack_bf16=1` the default for bf16 wrw split-K.** Halves atomic count for the worst-performing direction.

5. **Add more tile shapes to the mapping table.** Specifically:
   - 256x128 and 128x256 (using `wmma_acc_high_bank=1` or `wmma_acc_f16/bf16acc` to fit VGPR budget)
   - 32x128 and 128x32 (for small-M or small-N shapes)
   - 32x64 and 64x32 (intermediate)

### Medium-term improvements (1-2 weeks each)

6. **Implement `DISABLE_XDL_ARB_STALL` for occupancy=1 kernels.** Set `SCHED_MODE` bit[2] in prologue when VGPR pressure limits to 1 wave/CU. Allows back-to-back WMMA issue.

7. **Make `main_loop_interleave=1` the default when k-sub-loop is active.** Currently opt-in and requires `lds_double_buffer=1`. The sequential drain loop (PERF-003) is a clear performance regression for k-sub-loop configs.

8. **Use RA/RB matrix reuse bits.** When consecutive WMMA instructions in `emit_wmma_tile` share the same A or B matrix, set the reuse hint. ISA §7.12: "hints that either matrix is likely to be used again in the next instruction and should be cached."

9. **Explore `global_load_tr16_b128` for transposed operands.** bwd's B and wrw's A/B are transposed. Currently: global_load → VGPR → ds_write → ds_read (or ds_load_tr16_b128) → WMMA. The global transpose-load could skip the LDS staging entirely for the transposed operand.

10. **Fill in WMMA cycle counts.** Measure or obtain from AMD the latency for each WMMA instruction variant. Populate `inst_wmma_t.cycle` to enable latency-aware scheduling.

11. **Implement 256-wide tiles for fp16/bf16.** The `wmma_epilogue_chunked=1` + `wmma_acc_f16/bf16acc` or `wmma_acc_high_bank=1` mechanisms already exist. Add 256x128, 128x256, 256x64, 64x256 tile shapes to the mapping table and generate configs.

### Longer-term experiments (1+ month each)

12. **Multi-schedule main loop.** XDLOPS has a DAG-based scheduler (`main_loop_graph.py`) with ~10 schedules. WMMA has one fixed schedule. Implement 2-3 alternative schedules (e.g., compute-first, load-first, interleave) and let the config search pick the best per shape.

13. **Persistent kernels for wrw.** The `wrw_streamk` mechanism exists but is a proof-of-concept. Extend to all wrw configs and benchmark against split-K atomic-add.

14. **Fused epilogue operations.** Add bias-add, ReLU, or output-quantization fused into the epilogue store, avoiding a separate kernel launch.

15. **Direct convolution path for 1x1 filters.** For 1x1 stride-1 conv (equivalent to GEMM), the implicit-GEMM lowering adds overhead. A direct path (like `igemm_fwd_gtc_nchwc.py`) could skip the tap loop entirely.

---

## 8. Autotuning and Configuration-Selection Design

### Search-space definition

#### Inventory of tunable parameters

From `igemm_base.py:206-603` and `generate_all_configs.py:68-73`:

| Parameter | Type | Values | Constraints |
|---|---|---|---|
| `gemm_m_per_block` | int | 32, 64, 128, 256 | Must match mapping table entry |
| `gemm_n_per_block` | int | 32, 64, 128, 256 | Must match mapping table entry |
| `gemm_k_per_block` | int | 4(fp32), 32(fp16/bf16), 64(int8) | Multiple of `inst_wmma.k` |
| `wmma_tile_m` | int | 16 (fixed) | ISA constraint |
| `wmma_tile_n` | int | 16 (fixed) | ISA constraint |
| `wmma_repeat_m` | int | 2, 4, 8 | Derived from tile/block |
| `wmma_repeat_n` | int | 2, 4, 8 | Derived from tile/block |
| `tensor_a/b_thread_lengths` | list | `[1, K, 1, 1]` or `[1, K, 1, R]` | K=gemm_k_per_block, R=row_repeat |
| `tensor_a/b_cluster_lengths` | list | `[1, 1, 1, block_size]` | Must equal block_size |
| `direction` | str | fwd, bwd, wrw | |
| `precision` | str | fp16, bf16, fp32, int8 | |
| `nxb` | int | 0 (nhwc) | |
| `nxe` | int | 0 (1x1), 1 (multi-tap) | |
| `lds_double_buffer` | bool | 0, 1 | Required for fp32; required for interleave |
| `async_global_load` | bool | 0, 1 | Excl. saddr, interleave, row_repeat>1 |
| `tdm_global_load` | bool | 0, 1 | 1x1-only; excl. interleave, k_tail, gsplit |
| `saddr_global_load` | bool | 0, 1 | Excl. async, tdm, interleave, gsplit, row_repeat>1 |
| `main_loop_interleave` | bool | 0, 1 | Requires lds_double_buffer; excl. async, gsplit |
| `wmma_setprio` | bool | 0, 1 | |
| `wmma_acc_f16` | bool | 0, 1 | fp16 only; excl. gsplit |
| `wmma_acc_bf16` | bool | 0, 1 | bf16 only; excl. gsplit |
| `wmma_m_tail` | bool | 0, 1 | Excl. chunked, f16acc/bf16acc |
| `wmma_n_tail` | bool | 0, 1 | Excl. chunked; requires row_repeat_b==1 |
| `wmma_k_tail` | bool | 0, 1 | Excl. tdm; excl. gsplit for fwd |
| `wmma_epilogue_chunked` | bool | 0, 1 | Excl. tails, f16acc/bf16acc, gsplit |
| `wmma_acc_high_bank` | bool | 0, 1 | Requires chunked for 256x256 |
| `direct_store` | bool | 0, 1 | Excl. epilogue_lds_pad |
| `epilogue_lds_pad` | bool | 0, 1 | Excl. direct_store; 64x64 only |
| `local_prefetch_num` | int | 1, 2 | Excl. interleave; requires k-sub-loop |
| `gemm_k_global_split` | bool | 0, 1 | Excl. direct_store, async, interleave (fwd) |
| `atomic_pack_bf16` | bool | 0, 1 | bf16 wrw only; excl. reduction_kernel |
| `wrw_reduction_kernel` | bool | 0, 1 | wrw only; excl. atomic_pack_bf16 |
| `wrw_streamk` | bool | 0, 1 | wrw only |
| `gsplit_stagger` | bool | 0, 1 | wrw gsplit only |
| `atomic_scope` | str | SCOPE_SYS, SCOPE_DEV | wrw gsplit only |
| `ds_load_tr_b` | bool | 0, 1 | bwd/wrw fp16/bf16 (default 1) |

**Effective search space size:** ~5 tile shapes × 2^11 flag combinations × 3 directions × 4 precisions ≈ 125,440 candidates, filtered to ~2,000-5,000 valid by `is_valid()`.

#### Parameters eliminable analytically
- `wmma_tile_m`, `wmma_tile_n`: always 16 (ISA constraint)
- `nxb`: always 0 for nhwc
- `ds_load_tr_b`: always 1 for bwd/wrw fp16/bf16 (promoted to default)
- `wavefront_size`: always 32 for gfx1250
- `wmma_acc_f16` only valid for fp16; `wmma_acc_bf16` only for bf16 — eliminate for other precisions
- `atomic_pack_bf16`, `wrw_reduction_kernel`, `wrw_streamk`, `gsplit_stagger`: wrw only
- `tdm_global_load`: 1x1-only — eliminate for multi-tap configs

### Search strategy

**Recommendation: Coarse-to-fine with successive halving, guided by an analytical cost model.**

#### Rationale
- Full Cartesian product (~5,000 valid configs × 200,000 shapes) is infeasible (10^9 benchmarks).
- Each benchmark takes ~0.1-10s → 10^8-10^10 seconds — clearly impractical.
- A cost model can eliminate obviously dominated configs and predict promising regions.
- Successive halving prunes poor configs early.

#### Design

**Stage 1: Analytical pre-filter (no benchmarking)**
- For each (shape, config) pair, compute:
  - VGPR count → occupancy (1 or 2+ waves/CU)
  - LDS size → fits in 64KB?
  - Tile fit waste: `ceil(gemm_m/tile_m)*tile_m*ceil(gemm_n/tile_n)*tile_n / (gemm_m*gemm_n)` — waste ratio
  - Split-K overhead: atomic count = `grid_m * grid_n * splits`
  - Arithmetic intensity: FLOPs / bytes loaded
- Eliminate configs with occupancy=0 (VGPR overflow), LDS overflow, or tile-fit waste > 2x.

**Stage 2: Coarse search (representative shapes)**
- Select ~50 representative shapes per direction (covering small/medium/large, 1x1/3x3, various C/K ratios).
- Benchmark ALL valid configs on these 50 shapes.
- Keep top 20% of configs per shape (successive halving round 1).

**Stage 3: Fine search (full shape set)**
- Benchmark surviving configs on all 200,000 shapes.
- Use successive halving: after N benchmarks per config, drop bottom 50%.
- Budget: ~10 benchmarks per (shape, config) pair, ~20 measurements for top candidates.

**Stage 4: Noise detection**
- Repeat top-10 configs 5× per shape.
- Use median + IQR for robust statistics.
- Flag configs with >10% variance across repeats.

### Scaling plan for ~200,000 shapes

#### Shape canonicalization and deduplication
```
canonical_key = (direction, precision, n, c_per_group, k_per_group,
                 hi, wi, y, x, stride_h, stride_w, dilation_h, dilation_w,
                 pad_h, pad_w, group, layout)
```
Shapes with identical GEMM dimensions (`gemm_m`, `gemm_n`, `gemm_k`) and identical tile-divisibility properties produce identical kernel performance — dedup on:
```
gemm_key = (direction, precision, gemm_m, gemm_n, gemm_k,
            gemm_m % tile_m, gemm_n % tile_n, gemm_k % gemm_k_per_block,
            group, layout)
```
Expected dedup: 200,000 → ~5,000-10,000 unique (gemm_m, gemm_n, gemm_k, tail-pattern) combinations.

#### Clustering
Cluster shapes by features:
- GEMM dimensions (M, N, K) — log-binned
- Tile divisibility (M%32, N%32, K%32)
- Arithmetic intensity (2*M*N*K / (M*K + N*K + M*N) bytes)
- Filter size (1x1 vs 3x3 vs larger)
- Stride (1 vs 2+)

#### Caching
- Cache compiled `.hsaco` files by kernel name (already done by codegen).
- Cache benchmark results by `(gemm_key, config_name, commit_hash, arch)`.
- Store results in SQLite: `results(direction, precision, gemm_m, gemm_n, gemm_k, config, ms, nrms, commit, arch, timestamp)`.

#### Parallelization
- Each (shape, config) benchmark is independent — embarrassingly parallel.
- Launch multiple `conv_driver.exe` processes on different GPU streams.
- Limit: GPU memory for concurrent kernel launches (each needs input/output tensors).
- Practical: 4-8 concurrent processes per GPU.

#### Resume capability
- Each benchmark result is written to SQLite immediately after completion.
- On restart, query SQLite for completed (gemm_key, config) pairs and skip them.
- Deterministic: same commit + arch + shapes → same results.

### Generalization strategy

#### Selector design: Compact lookup table with fallback hierarchy

```
select_config(shape, arch):
    1. Exact match: lookup (direction, precision, gemm_m, gemm_n, gemm_k, tail_pattern) in LUT
       → if found, return best config
    2. Nearest-neighbor: find closest (gemm_m, gemm_n, gemm_k) within 10% ratio
       → if found, return its best config
    3. Tile-fit heuristic: pick config with best tile-fit and highest occupancy
       → return default config
    4. Fallback: return a "generic" config (128x128, lds_double_buffer=1, direct_store=1)
```

#### Model features
- `gemm_m`, `gemm_n`, `gemm_k` (log-scaled)
- `gemm_m % 128`, `gemm_n % 128`, `gemm_k % 32` (tail indicators)
- `gemm_m / 128`, `gemm_n / 128` (tile count)
- `min(gemm_m, gemm_n) / max(gemm_m, gemm_n)` (aspect ratio)
- `gemm_k / 32` (K-depth)
- `2*gemm_m*gemm_n*gemm_k / (gemm_m*gemm_k + gemm_n*gemm_k + gemm_m*gemm_n)` (arithmetic intensity)
- `direction`, `precision` (categorical)
- `y*x` (filter size)
- `stride_h * stride_w` (stride product)

#### Overfitting prevention
- Split shape families (not random shapes) into train/validation/test:
  - Train: ResNet50 shapes, common CNN shapes
  - Validation: different ResNet variants, DenseNet shapes
  - Test: adversarial shapes (prime-number channels, extreme aspect ratios, tiny/large K)
- Ensure no near-duplicate shapes span train/test splits.

### Optimization objective

```
score(config, shape) = w_lat * median_latency + w_tail * p99_latency
                      + w_compile * compile_time + w_workspace * workspace_bytes
                      - w_correct * correctness_pass
```

Where:
- `w_lat = 1.0` (primary: median latency)
- `w_tail = 0.3` (secondary: tail latency)
- `w_compile = 0.01` (penalize slow-compiling configs)
- `w_workspace = 0.01` (penalize large workspace)
- `w_correct = ∞` (correctness is a hard gate — no config is accepted unless `valid:y`)

Use Pareto optimization: report the Pareto frontier of (latency, compile_time, workspace) and pick the knee point.

### Data and artifact format

```json
{
  "version": 1,
  "arch": "gfx1250",
  "commit": "29818ef",
  "generated": "2026-09-01",
  "configs": {
    "fwd_fp16": [
      {
        "gemm_m_range": [1, 256],
        "gemm_n_range": [1, 256],
        "gemm_k_range": [32, 1024],
        "config": "igemm_fwd_gtc_gfx1250_nhwc_fp16_all.config",
        "kernel": "igemm_fwd_gtcw_nhwc_fp16_..._128x128x32_...",
        "features": {"tile_m": 128, "tile_n": 128, "lds_dbuf": 1, ...}
      },
      ...
    ]
  },
  "fallback": {
    "fwd_fp16": "igemm_fwd_gtcw_nhwc_fp16_..._64x64x32_..."
  }
}
```

### End-to-end pseudocode

```python
def autotune(shapes, arch, commit):
    # Stage 1: Canonicalize and dedup
    unique_keys = dedup_shapes(shapes)  # ~5000-10000 from 200000

    # Stage 2: Generate all valid configs
    configs = generate_all_valid_configs(arch)  # ~2000-5000

    # Stage 3: Analytical pre-filter
    for shape_key in unique_keys:
        valid_configs = [c for c in configs if is_applicable(shape_key, c)]
        for c in valid_configs:
            c.occupancy = compute_occupancy(shape_key, c)
            c.tile_waste = compute_tile_waste(shape_key, c)
        valid_configs = [c for c in valid_configs if c.occupancy > 0 and c.tile_waste < 2.0]

    # Stage 4: Coarse search on representative shapes
    rep_shapes = select_representative(unique_keys, n=50)
    for shape_key in rep_shapes:
        for c in valid_configs:
            result = benchmark(shape_key, c, arch, commit)
            store_result(shape_key, c, result)
        # Successive halving round 1
        valid_configs = keep_top_percent(valid_configs, 20)

    # Stage 5: Fine search on all shapes
    for shape_key in unique_keys:
        for c in valid_configs:
            if already_benchmarked(shape_key, c):
                continue
            result = benchmark(shape_key, c, arch, commit)
            store_result(shape_key, c, result)

    # Stage 6: Build selector LUT
    lut = build_lookup_table(unique_keys, best_config_per_shape)
    save_artifact(lut, arch, commit)

def select_config(shape, arch):
    key = canonicalize(shape)
    if key in lut[arch]:
        return lut[arch][key]
    nn = nearest_neighbor(key, lut[arch])
    if nn and ratio(key, nn) < 0.1:
        return lut[arch][nn]
    return fallback_config(arch, shape.direction, shape.precision)
```

---

## 9. Validation and Benchmarking Plan

### Correctness validation

#### Differential testing
For each (direction, precision, shape, config):
1. Run MISA kernel with `-V 1` (verify against `naive_conv.hsaco` GPU reference).
2. Acceptance: `valid:y`, NRMS below precision-specific threshold.
3. Compare MISA gfx1250 result against gfx950 XDLOPS result on same shape (if semantically applicable).

#### Tolerances
| Precision | NRMS threshold | Notes |
|---|---|---|
| fp32 | 1e-6 | Bit-exact for same accumulation order |
| fp16 | 1e-3 | fp16 accumulation differences |
| bf16 | 1e-3 | bf16 accumulation differences |
| int8 | 0 (exact) | Integer accumulation is exact |

#### Property-based testing
- Random shape generation: N∈{1,2,4,8,16,32,64,128}, C∈{1,3,8,16,32,64,128,256,512,1024}, H,W∈{1,7,8,14,17,28,32,56,112,224}, K∈{1,3,8,16,32,64,128,256,512,1024}, Y,X∈{1,3,5,7}, stride∈{1,2}, dilation∈{1,2}, pad∈{0,1,2,3}, group∈{1,2,4,8}
- Adversarial shapes: prime-number channels (C=97, K=101), extreme aspect ratios (H=1, W=1000), tiny K (K=1), huge K (K=4096)

#### Boundary and adversarial shape families
- Tile boundaries: gemm_m = 127, 128, 129 (around 128x128 tile)
- K boundaries: gemm_k = 31, 32, 33 (around inst_wmma.k=32)
- Group boundaries: group=1 vs group=2 (channel interleaving)
- 1x1 vs 3x3 vs 7x7 filters
- Stride 1 vs stride 2
- Dilation 1 vs dilation 2
- Asymmetric filters: y=1, x=3 or y=3, x=1

#### Determinism
- Run each kernel 3× on the same shape; verify identical output (no non-deterministic atomics for non-split-K).
- For split-K: run 5× and verify atomic-add produces same result each time.

### Performance validation

#### Metrics
- Median latency (ms) over 20 repeats (after 5 warmup)
- P99 latency
- TFLOPS: `2 * M * N * K / median_latency / 1e9`
- Bandwidth: bytes_loaded / median_latency
- Occupancy: from `IGEMM_LOG_FASTEST_CONFIG` + VGPR count analysis
- Kernel count: number of valid configs searched

#### Shape families for benchmarking
1. **ResNet50 shapes**: standard CNN shapes (224x224, 56x56, 28x28, etc.)
2. **Large-channel 1x1**: N=128, C=1024, K=1024, 1x1
3. **Small-spatial large-channel**: N=4, C=512, H=W=8, K=256, 3x3
4. **Large-spatial small-channel**: N=1, C=64, H=W=512, K=128, 3x3
5. **Depthwise (group=C)**: N=64, C=128, H=W=32, K=128, 3x3, group=128
6. **Tail-heavy**: C=96, K=96 (not multiples of 32/64/128)

#### Acceptance criteria
- Correctness: `valid:y` for ALL test shapes
- Performance: median latency within 1.2x of gfx950 for fwd, 1.5x for bwd, 1.5x for wrw
- No regression: new configs must not be slower than the config they replace by more than 5%

#### Suggested commands
```bash
# Build and verify a single config
python3 igemm_codegen.py config/igemm_fwd_gtc_gfx1250_nhwc_fp16_all.config
./out/conv_driver.exe convfp16 -n 128 -c 1024 -H 17 -W 17 -k 1024 \
    -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F 1 -V 1 \
    --in_layout NHWC --fil_layout NHWC --out_layout NHWC

# Benchmark vs gfx950
python3 script/benchmark_gfx1250_vs_gfx950_diverse.py --direction all

# Benchmark vs MIOpen
python3 script/benchmark_gfx1250_vs_miopen.py --direction all --verify

# Find fastest config
IGEMM_LOG_FASTEST_CONFIG=1 ./out/conv_driver.exe convfp16 ...
```

---

## 10. Prioritized Action Table

| ID | Category | Expected Impact | Confidence | Effort | Validation Cost | Dependencies | Order |
|---|---|---|---|---|---|---|---|
| PERF-001 | Config coverage | High | High | Medium | Medium | None | 1 |
| PERF-002 | LDS double-buffer default | High | High | Low | Low | None | 2 |
| PERF-005 | Register pressure / occupancy | High | Medium | High | High | PERF-001 | 3 |
| PERF-003 | K-substep interleave default | Medium | High | Low | Low | PERF-002 | 4 |
| PERF-004 | DISABLE_XDL_ARB_STALL | Medium | Medium | Low | Medium | PERF-005 | 5 |
| PERF-006 | direct_store default | Low-Med | High | Low | Low | None | 6 |
| PERF-008 | atomic_pack_bf16 default | Medium | High | Low | Low | None | 7 |
| PERF-009 | wrw hipMemset in timing | Medium | High | Low | Low | None | 7 |
| COR-001 | fp32 dbuf assert | N/A (safety) | High | Low | Low | None | 8 |
| COR-002 | saddr+n_tail assert | N/A (safety) | High | Low | Low | None | 9 |
| PERF-007 | WMMA cycle counts | Low | Medium | Low | Low | None | 10 |
| COR-003 | int8 atomic HW validation | N/A (safety) | Medium | Low | Medium | GPU | 11 |
| §7.6 | RA/RB reuse bits | Medium | Low | Low | Medium | None | 12 |
| §7.7 | global_load_tr16_b128 | Medium | Low | Medium | Medium | None | 13 |
| §7.8 | Multi-schedule main loop | Medium | Low | High | High | PERF-007 | 14 |
| §8 | Autotuning system | High | High | High | High | PERF-001 | 15 |

---

## 11. Suggested Patches

### Patch 1: Make lds_double_buffer=1 the default for gfx1250 WMMA

**File:** `python/igemm/igemm_base.py`
**Location:** Line 246
```python
# Before:
self.lds_double_buffer = utility_dict_with_default_t(tunable_dict)('lds_double_buffer', 0)

# After:
# PERF-002: default to double-buffered for gfx1250 WMMA (mirrors XDLOPS's own default).
# fp32 already requires this (COR-001); making it the default for all precisions ensures
# load/compute overlap and eliminates the single-buffer barrier stall.
self.lds_double_buffer = utility_dict_with_default_t(tunable_dict)('lds_double_buffer', 1)
```

### Patch 2: Add assert for COR-002 (saddr + n_tail for fwd)

**File:** `python/igemm/igemm_fwd_gtc_wmma_nhwc.py`
**Location:** After line 275 (saddr_global_load asserts)
```python
assert not (tunable.saddr_global_load and tunable.wmma_n_tail), \
    "saddr_global_load + wmma_n_tail is confirmed broken for fwd (valid:n on hardware) -- " \
    "see script/generate_all_configs.py:118-127"
```

### Patch 3: Add fp32 lds_double_buffer assert

**File:** `python/igemm/igemm_base.py`
**Location:** After line 246 (lds_double_buffer read)
```python
if self.precision == 'fp32' and self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA:
    assert self.lds_double_buffer, \
        "fp32 WMMA requires lds_double_buffer=1 (silent corruption at high occupancy without it, see AGENTS.md)"
```

### Patch 4: Add more tile shapes to mapping table

**File:** `python/operations/wmma_mapping.py`
**Location:** After line 329 (fp16 table)
```python
# PERF-001: additional tile shapes for better shape coverage.
# 256x128: needs wmma_acc_high_bank=1 or wmma_acc_f16acc=1 to fit VGPR budget.
# Same wave_repeat_m/n as 128x128, just more waves_per_m.
ctrl_wmma_mapping_t(256, 128, 16, 16, 4, 4, 4, v_wmma_f32_16x16x32_f16),  # 4 waves_per_m, 2 waves_per_n
# 128x256: mirror
ctrl_wmma_mapping_t(128, 256, 16, 16, 4, 4, 4, v_wmma_f32_16x16x32_f16),  # 2 waves_per_m, 4 waves_per_n
# 32x128: for small-M shapes
ctrl_wmma_mapping_t(32,  128, 16, 16, 1, 2, 8, v_wmma_f32_16x16x32_f16),  # 1 wave_per_m, 4 waves_per_n
# 128x32: mirror
ctrl_wmma_mapping_t(128, 32,  16, 16, 4, 8, 2, v_wmma_f32_16x16x32_f16),  # 4 waves_per_m, 1 wave_per_n
```
**Note:** Each new shape also needs corresponding `tensor_a/b_thread/cluster_lengths` in config files and may need global-load generalization (row_repeat). This is a larger effort than the other patches.

### Patch 5: Set DISABLE_XDL_ARB_STALL for occupancy=1 kernels

**File:** `python/igemm/igemm_fwd_gtc_wmma_nhwc.py`
**Location:** In `emit_kernel_prologue`, after SGPR init
```python
# PERF-004: when occupancy is limited to 1 wave/CU by VGPR pressure, allow
# back-to-back WMMA issue by disabling the XDL arbiter stall.
# ISA §5.7.2.1: "likely beneficial primarily when a single wave is running on a SIMD"
if self.occupancy == 1:
    # SCHED_MODE is hwreg 0x7, bit 2 = DISABLE_XDL_ARB_STALL
    self._emit(f"s_getreg_b32 s[{self.sgpr.s_tmp()}], hwreg(0x7)")
    self._emit(f"s_or_b32 s[{self.sgpr.s_tmp()}], s[{self.sgpr.s_tmp()}], 0x4")
    self._emit(f"s_setreg_b32 hwreg(0x7), s[{self.sgpr.s_tmp()}]")
```
**Note:** `self.occupancy` needs to be computed from VGPR count. This is pseudocode — exact hwreg encoding should be verified against the ISA doc.

---

## 12. Open Questions and Missing Evidence

1. **WMMA instruction latency.** The ISA doc does not state per-instruction cycle counts for WMMA variants (only mentions "16-cycle WMMA" as an example in §5.7.2.1). The `cycle=None` field in `wmma.py` prevents latency-aware scheduling.
   - **How to answer:** Measure via a microbenchmark (issue N back-to-back WMMAs, measure cycles via `s_memtime`).

2. **Actual occupancy at runtime.** VGPR count is known at codegen time, but actual achieved occupancy depends on LDS size and CU count. The code computes `lds_single_size` but doesn't compute or assert occupancy.
   - **How to answer:** Add occupancy computation to `__init__` using `amdgpu_calculate_occupancy` (already exists in `amdgpu.py:246`). Compare with `IGEMM_SCLK_MHZ` / hardware profiler.

3. **Why is wrw specifically 2-3.5x slower?** The wrw path uses transposed operands (both A and B), atomic-add split-K, and has fewer tile shapes. But the relative contribution of each factor is unknown.
   - **How to answer:** Profile wrw kernels with ROCm profiler (rocprof) to identify the bottleneck (atomic contention, LDS bank conflicts, global memory bandwidth, or compute).

4. **Does the WMMA hazard table's "coexec slots" actually provide correctness?** The code relies on the hardware arbiter stall (default behavior) to satisfy WMMA hazard requirements. This is plausible but not explicitly confirmed.
   - **How to answer:** Verify by checking generated disassembly for V_NOP insertion, or test with `DISABLE_XDL_ARB_STALL` and no V_NOPs to see if results corrupt.

5. **What is the actual VGPR count per kernel?** The code allocates VGPRs via `gpr_sequencer_t` but doesn't print the final count. The 256-VGPR limit is mentioned in comments but not enforced.
   - **How to answer:** Add a print of `v_end` / `workitem_vgpr_count` after VGPR allocation in `kernel_vgpr_t.__init__`.

6. **gfx950 vs gfx1250 hardware comparison.** The benchmark data compares MISA/gfx1250 against MIOpen/gfx950 and MIOpen/gfx1250 (different solvers). A controlled comparison (same shapes, same verification) of MISA/gfx1250 vs MISA/gfx950 is needed to isolate codegen quality from hardware capability.
   - **How to answer:** Build both gfx950 and gfx1250 configs, run on the same shapes, compare TFLOPS and bandwidth utilization.

---

## Three Highest-Value Next Actions

1. **Add more tile shapes (256x128, 128x256, 32x128, 128x32) to the WMMA mapping table and generate configs.** This directly addresses the largest performance gap (PERF-001: 5 vs 30+ tile shapes). Start with shapes that fit the existing VGPR budget (use `wmma_acc_high_bank=1` or `wmma_acc_f16/bf16acc` for 256-wide tiles).

2. **Make `lds_double_buffer=1` the default for all gfx1250 WMMA configs.** This is a one-line change (Patch 1) with high confidence and high impact. XDLOPS already defaults to double-buffered; WMMA's single-buffered default is a legacy of the correctness-first milestone.

3. **Profile wrw kernels with rocprof to identify the bottleneck.** wrw is 2-3.5x slower than gfx950. The code has multiple potential bottlenecks (scalar atomic-add, transposed operand LDS traffic, sequential k-substep drain). A profiler trace would pinpoint which to address first, avoiding wasted optimization effort.
