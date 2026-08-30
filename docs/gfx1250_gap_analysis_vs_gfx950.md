# gfx1250 WMMA Convolution Gap Analysis vs gfx950 XDLOPs

*Source-backed analysis. Generated 2026-08-30. Every factual statement cites repository evidence. Hypotheses are labeled as such.*

---

## A. Executive Summary

### Five highest-value gfx1250 opportunities

**1. Fix the `direct_store` kernel-name suffix bug and re-benchmark (P0)**
The `direct_store` optimization (Phase 59) eliminates the LDS-reshuffle epilogue via per-lane `global_store_dword`, cross-validated by FlyDSL's production gfx1250 GEMM. However, the C++ driver's `igemm_gtc_encode_kernel_name()` never appended the `_direct` suffix, so `conv_driver.exe` could never actually *select* a direct_store kernel — every prior "direct_store wins" benchmark was measuring the wrong (non-direct) kernel. The bug is now fixed in the code, but **no valid performance data exists for direct_store**. Re-benchmarking is the single cheapest high-impact action.
- Evidence: `docs/gfx1250_optimization_backlog.md:77-94`; `docs/gfx1250_direct_store_plan.md`
- Impact: unknown (all prior data invalidated), but mechanism eliminates ~21-27% of instructions (LDS traffic in epilogue, per `docs/gfx1250_rocprof_profiling.md:191-275`)

**2. Port `main_loop_interleave` from fwd-only to bwd/wrw (P0)**
The `main_loop_interleave` optimization pairs each K-substep's compute with the next substep's global-load issue, hiding global-load latency behind WMMA compute. It is currently restricted to **fwd/fp16 only** (`script/generate_all_configs.py` is_valid rule 13: `mli AND dir!=fwd → invalid`). BWD and WRW have no interleave path at all. Given rocprof shows WMMA-busy at only 2-5% and wait-inst stalls at 16% for fwd, latency hiding is the highest-leverage pipeline lever.
- Evidence: `python/operations/wmma_main_loop.py:101-115` (Phase 15); `script/generate_all_configs.py` rule 13; `docs/gfx1250_rocprof_profiling.md:191-275`
- Impact: medium-high (latency hiding directly addresses the <5% WMMA-busy finding)

**3. Widen the WMMA pipeline schedule (P1)**
The WMMA path has exactly **one** software pipeline schedule (`wmma_main_loop_t.emit()`), explicitly described as "deliberately a single, correctness-first schedule" in its docstring. The gfx950 XDLOPs path has **~10 hand-unrolled schedules** dispatched by `wave_repeat_m × n × local_prefetch_num × interleave × lds_buffer_num`. Adding even 2-3 specialized schedules (e.g., for the common `wave_repeat_m=4, wave_repeat_n=4` case with double-buffer + prefetch) could significantly improve instruction-level parallelism.
- Evidence: `python/operations/wmma_main_loop.py:33-37` (docstring); `python/operations/mfma_main_loop.py:3430-3473` (~10 schedules)
- Impact: medium (pipeline depth is the main lever for <5% WMMA utilization)

**4. Extend `saddr_global_load` from fwd-only to bwd/wrw (P0)**
The SADDR global-load path (Phase 61) uses a scalar base register + 32-bit VGPR offset instead of a 64-bit VADDR carry chain, saving ~4 VGPRs per operand. It is currently fwd-only (all 4 precisions). BWD and WRW both need fresh designs for their transposed operands. With occupancy at 25% (128x128) and VGPR-limited, even 4 VGPRs saved could improve occupancy.
- Evidence: `config/igemm_fwd_gtc_gfx1250_nhwc_{fp16,bf16,fp32,int8}_saddr.config`; `docs/gfx1250_optimization_backlog.md:96-110` (P3, in-progress, fwd pilot done)
- Impact: low-medium (4 VGPRs → possibly 1 extra wave/CU)

**5. WRW addressing redesign for `gemm_k > gemm_m` (P1)**
WRW's global-load addressing requires `gemm_m_per_block >= gemm_k_per_block` with power-of-2 quotient (`num_col_groups = gemm_m_per_block // gemm_k_per_block`). This caps gemm_k_per_block at 64 when gemm_m=64. CK pairs 64x64 tiles with K=96/128/256, which MISA cannot reach. Design A (fractional col_group) is documented and estimated at ~3-4 sessions. This unlocks a richer tile-shape space for the direction with the largest performance gap (wrw avg 5.1x slower vs MIOpen).
- Evidence: `docs/gfx1250_wrw_addressing_redesign.md:1-120`; `docs/gfx1250_optimization_backlog.md:160-166`
- Impact: medium (enables CK-style tile/K pairings for wrw)

---

## B. Coverage Matrix

### B.1 Architecture × Direction × Datatype × Kernel-Family

| Arch | Direction | FP32 | FP16 | BF16 | INT8 | INT4 | Kernel Family | Config Files |
|------|-----------|------|------|------|------|------|---------------|--------------|
| **gfx1250** | fwd | ✅ WMMA | ✅ WMMA | ✅ WMMA | ✅ WMMA | ❌ | `igemm_fwd_gtc_wmma_nhwc_t` | ~50+ |
| **gfx1250** | bwd | ✅ WMMA | ✅ WMMA | ✅ WMMA | ✅ (base+tdm only) | ❌ | `igemm_bwd_gtc_wmma_nhwc_t` | ~35+ |
| **gfx1250** | wrw | ✅ WMMA | ✅ WMMA | ✅ WMMA | ✅ (64x64_k128 only) | ❌ | `igemm_wrw_gtc_wmma_nhwc_t` | ~40+ |
| **gfx950** | fwd | ✅ XDLOPs | ✅ XDLOPs | ✅ XDLOPs | ✅ XDLOPs | ❌ | `igemm_fwd_gtc_nhwc_t` | 4 |
| **gfx950** | bwd | ✅ XDLOPs | ✅ XDLOPs | ✅ XDLOPs | ❌ | ❌ | `igemm_bwd_gtc_nhwc_t` | 3 |
| **gfx950** | wrw | ✅ XDLOPs | ✅ XDLOPs | ✅ XDLOPs | ❌ | ❌ | `igemm_wrw_gtc_nhwc_t` | 3 |
| **gfx942** | fwd | ✅ XDLOPs | ✅ XDLOPs | ✅ XDLOPs | ✅ XDLOPs | ❌ | `igemm_fwd_gtc_nhwc_t` | 4 |
| **gfx942** | bwd | ✅ XDLOPs | ✅ XDLOPs | ✅ XDLOPs | ❌ | ❌ | `igemm_bwd_gtc_nhwc_t` | 3 |
| **gfx942** | wrw | ✅ XDLOPs | ✅ XDLOPs | ✅ XDLOPs | ❌ | ❌ | `igemm_wrw_gtc_nhwc_t` | 3 |
| **gfx940** | fwd | ❌ | ✅ XDLOPs | ✅ XDLOPs | ❌ | ❌ | `igemm_fwd_gtc_nhwc_t` | 2 |
| **gfx940** | bwd | ❌ | ✅ XDLOPs | ❌ | ❌ | ❌ | `igemm_bwd_gtc_nhwc_t` | 1 |
| **gfx940** | wrw | ✅ XDLOPs | ✅ XDLOPs | ❌ | ❌ | ❌ | `igemm_wrw_gtc_nhwc_t` | 2 |
| **gfx90a** | fwd | ✅ XDLOPs | ✅ XDLOPs | ✅ XDLOPs | ✅ XDLOPs | ❌ | `igemm_fwd_gtc_nhwc_t` | 4 |
| **gfx90a** | bwd | ✅ XDLOPs | ✅ XDLOPs | ✅ XDLOPs | ❌ | ❌ | `igemm_bwd_gtc_nhwc_t` | 3 |
| **gfx90a** | wrw | ✅ XDLOPs | ✅ XDLOPs | ❌ | ❌ | ❌ | `igemm_wrw_gtc_nhwc_t` | 2 |
| **gfx908** | fwd | ✅ DLOPs/XDLOPs | ✅ XDLOPs | ❌ | ✅ XDLOPs | ❌ | `igemm_fwd_gtc_nhwc_t` | 5+ |
| **gfx908** | bwd | ✅ DLOPs/XDLOPs | ✅ XDLOPs | ❌ | ❌ | ❌ | `igemm_bwd_gtc_nhwc_t` | 3+ |
| **gfx908** | wrw | ✅ DLOPs/XDLOPs | ✅ XDLOPs | ❌ | ❌ | ❌ | `igemm_wrw_gtc_nhwc_t` | 4+ |

*Evidence: `config/` directory enumeration by ConfigInventory agent. gfx950/942/940/90a/908 configs confirmed to exist (correcting the initial assumption). All gfx1250 configs use nxe=0 (1x1/GEMM-only). All gfx950+ configs use nxe=1 (general conv).*

### B.2 gfx1250 Tile × Optimization-Flag Coverage

| Tile | Direction | Precisions | direct_store | gsplit | tdm | dbuf | async | interleave | setprio | m/n/k_tail | k2x | f16/bf16acc | streamk | saddr | high_bank |
|------|-----------|------------|--------------|--------|-----|------|-------|------------|---------|------------|-----|-------------|---------|-------|-----------|
| 128x128 | fwd | fp16/bf16/fp32/int8 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅(fp16) | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅(bf16) |
| 128x128 | bwd | fp16/bf16/fp32 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅(bf16) |
| 128x128 | wrw | fp16/bf16/fp32 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| 64x64 | fwd | fp16/bf16/fp32/int8 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| 64x64 | bwd | fp16/bf16/fp32 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| 64x64 | wrw | fp16/bf16/fp32/int8 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| 128x64 | fwd | fp16/bf16/fp32 | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| 64x128 | fwd | fp16/bf16/fp32 | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| 32x32 | bwd | fp16/bf16/fp32 | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| 32x32 | wrw | fp32(k96) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| 256x256 | fwd | bf16 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ |
| 256x128 | bwd | bf16 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ |

*Evidence: ConfigInventory agent, `script/generate_all_configs.py` BASE_SECTIONS + is_valid() rules, individual config file inspection.*

### B.3 Key coverage gaps in the matrix

1. **No int8 in `generate_all_configs.py` BASE_SECTIONS** — int8 has no combinatorial sweep. Only base + direct + tdm + saddr configs exist for fwd int8; bwd int8 has only base+tdm; wrw int8 has only 64x64_k128.
2. **No `main_loop_interleave` for bwd/wrw** — is_valid rule 13: `mli AND dir!=fwd → invalid`.
3. **No `async_global_load` for wrw** — WRW generator lacks `v_off_a/v_zero/v_sst_tmp` VGPRs (`igemm_wrw_gtc_wmma_nhwc.py`).
4. **No `wmma_epilogue_chunked` or `wmma_acc_high_bank` for wrw** — asserted fwd/bwd only (`igemm_base.py:746,769`).
5. **No `saddr_global_load` for bwd/wrw** — fwd-only configs.
6. **No `streamk` for fwd/bwd** — wrw-only.
7. **No bwd/wrw 128x64 or 64x128 tiles** — fwd-only in BASE_SECTIONS.
8. **All gfx1250 configs are nxe=0** — no general-conv (nxe=1) gfx1250 configs, despite the generator code supporting it. All gfx950+ configs are nxe=1.

---

## C. Parity-Gap Table

| Priority | Existing optimization | Source path/symbol | Existing direction/datatype | Missing targets | Portability | Proposed change | Expected mechanism | Risk | Validation |
|----------|----------------------|--------------------|---------------------------|-----------------|-------------|----------------|-------------------|------|------------|
| **P0** | `direct_store` (Phase 59) | `coalescing_store_wmma.py:669` `_emit_direct_store`; `igemm_base.py` `direct_store` tunable | fwd/bwd/wrw × fp16/bf16/fp32/int8 | **All — bug fixed but no valid perf data exists** | Directly portable (already wired all directions) | Re-benchmark with confirmed-correct `_direct` kernel-name suffix | Eliminates LDS scatter+barrier+gather epilogue (~21-27% of instructions per rocprof) | Low (correctness already validated) | `conv_driver.exe -V 1` + timing; compare direct vs LDS-reshuffle on aligned/tail shapes |
| **P0** | `main_loop_interleave` (Phase 15) | `wmma_main_loop.py:101-115` `emit_interleaved_substeps`; `igemm_base.py` `main_loop_interleave` | fwd/fp16 only | **bwd (all prec), wrw (all prec), fwd/bf16/fp32** | Requires adaptation (transposed B operand in bwd/wrw changes interleave pattern) | Port `emit_interleaved_substeps` to bwd/wrw generators; relax is_valid rule 13; requires `lds_double_buffer=1` | Hides global-load latency behind WMMA compute (addresses 16% wait-inst stall for fwd) | Medium (transposed operand interleave is structurally different) | `conv_driver.exe -V 1` on K-heavy shapes; rocprof `SQ_INST_CYCLES_VALU_WMMA` before/after |
| **P0** | `saddr_global_load` (Phase 61) | `igemm_fwd_gtc_wmma_nhwc.py` `_emit_gld_chunk_load` saddr branch; `config/*_saddr.config` | fwd × all 4 precisions | **bwd (all), wrw (all)** | Requires adaptation (B operand is transposed in bwd/wrw; needs fresh SADDR design for transposed load) | Add saddr path to bwd/wrw `global_load_a/b_functor`; new configs | Saves ~4 VGPRs/operand (64-bit VADDR → 32-bit offset), may improve occupancy | Medium (transposed operand addressing with SADDR is non-trivial) | `conv_driver.exe -V 1`; check VGPR count via disassembly; occupancy via `gfx1250_occupancy_check.cpp` |
| **P0** | General-conv (nxe=1) gfx1250 configs | Generator code supports nxe=1 in all directions (`emit_kernel_tap_loop`) | **Code supports it, but NO configs exist** | All directions × all precisions | Directly portable (generator already handles it) | Create nxe=1 config files for common 3x3/stride2 shapes | Enables non-1x1 convolution on gfx1250 (currently only GEMM-only shapes work) | Low (generator already supports; just no config files) | `conv_driver.exe -V 1` with 3x3 conv shapes |
| **P1** | Multiple pipeline schedules | `mfma_main_loop.py:3430-3473` (~10 schedules) | gfx950 XDLOPs (all directions) | **gfx1250 WMMA (all directions)** | Requires adaptation (WMMA has different latency/hazard semantics than MFMA; no AGPR; wave32) | Add 2-3 specialized schedules for common `wave_repeat_m=4,n=4` case (single-buf, double-buf, double-buf+prefetch) | Better instruction-level parallelism; WMMA-busy currently <5% | High (hand-tuned schedules require hardware validation) | rocprof `SQ_INST_CYCLES_VALU_WMMA` before/after; timing on K-heavy shapes |
| **P1** | WRW addressing redesign (gemm_k > gemm_m) | `docs/gfx1250_wrw_addressing_redesign.md:1-120` Design A | Not implemented (any direction) | **wrw (all precisions)** | Architecture-specific (WRW's transposed A+B addressing) | Implement Design A (fractional col_group) in `emit_kernel_prologue`; ~30 lines | Unlocks K=96/128/256 for 64x64 tiles (CK pairing); enriches wrw tile space | Medium-High (new addressing scheme, needs careful LDS layout verification) | `conv_driver.exe -V 1` on wrw shapes with K>64; compare vs K=64 baseline |
| **P1** | `wmma_k_tail + gemm_k_global_split` composition | `igemm_wrw_gtc_wmma_nhwc.py:760-766` (last-shard clamp) | wrw only | **fwd, bwd** | Directly portable (same last-shard remainder logic) | Add last-shard `s_knum += s_gemm_k_tail` to fwd/bwd `run()` WMMA branch in driver; remove assert in `igemm_base.py:801-808` | Enables split-K on shapes where gemm_k is not a multiple of gemm_k_per_block (common for fwd/bwd) | Low-Medium (needs driver-side karg additions + kernel-side tail mask) | `conv_driver.exe -V 1` on shapes with gemm_k%gemm_k_per_block!=0 |
| **P1** | `atomic_pack_bf16` for fwd/bwd | `coalescing_store_wmma.py:561` (packed bf16x2 atomic); `igemm_base.py:615-622` | wrw/bf16 only | **fwd/bwd (bf16)** | Directly portable (same V_PERMLANE_XOR_B32 packing) | Wire `atomic_pack_bf16` in fwd/bwd coalescing_store ctrl; add asserts | Halves atomic count for bf16 split-K epilogue | Low (wrw measured net loss, but wrw's bottleneck was never atomics — fwd/bwd may differ) | `conv_driver.exe -V 1 -F 1/2` with bf16 split-K shapes |
| **P1** | `wmma_epilogue_chunked` for wrw | `coalescing_store_wmma.py:189` `_emit_chunked_non_atomic_store`; `igemm_base.py:744-751` | fwd/bwd only | **wrw** | Requires adaptation (WRW fires epilogue per tap; chunked store needs per-tap chunk management) | Lift assert in `igemm_base.py:746`; add per-tap chunked logic to wrw epilogue | Enables >128x128 tiles for wrw (currently capped at 128x128 by LDS+VGPR) | Medium (per-tap chunking adds complexity) | `conv_driver.exe -V 1` on wrw with 256x256 tile |
| **P1** | `wmma_acc_high_bank` for wrw | `igemm_base.py:769-782`; `igemm_fwd/bwd_gtc_wmma_nhwc.py` vgpr_msb_tracker | fwd/bwd only | **wrw** | Requires adaptation (WRW's per-tap v_c zero + epilogue interacts with VGPR-MSB banking) | Add `wmma_acc_high_bank` field to wrw tunable; wire vgpr_msb_tracker in wrw generator | Enables >128x128 tiles for wrw via VGPR bank-1 accumulator | High (VGPR-MSB had 7 bugs in fwd/bwd; wrw's per-tap reset adds another dimension) | `conv_driver.exe -V 1` on wrw with 256x256 tile; rocgdb for bank tracking |
| **P2** | `gsplit_stagger` for fwd/bwd | `igemm_wrw_gtc_wmma_nhwc.py:702-710`; `igemm_base.py:640-643` | wrw only | **fwd, bwd** | Directly portable (same `s_sleep_var` emission) | Add `gsplit_stagger` emission to fwd/bwd prologue | Desynchronizes shard launch bursts at high split counts (~3-4% measured in wrw) | Low (wrw measured wash at moderate splits) | `conv_driver.exe` timing with `IGEMM_GSPLIT_SWEEP` at high split counts |
| **P2** | Magic division for bwd/wrw hot paths | `igemm_fwd_gtc_wmma_nhwc.py:734,863,1454,...` (11 sites); `IGEMM_GTC_FEAT_MAGIC_DIVISION=1` | fwd only (Phase 60) | **bwd, wrw (group decode, stride_h/w)** | Directly portable (same magic-multiply technique) | Port `macro_mdiv_u32_rem_vs_t` calls to bwd/wrw coordinate paths | Replaces 15-instruction emulated division with 3-VALU magic multiply | Low (pure arithmetic optimization) | `conv_driver.exe` timing on group>1 / stride!=1 shapes |
| **P2** | `wrw_reduction_kernel` for fwd/bwd | `coalescing_store_wmma.py:646-647`; `igemm_base.py:824-832` | wrw only | **fwd, bwd** | Requires adaptation (fwd/bwd epilogue semantics differ from wrw's per-tap store) | Wire `wrw_reduction_kernel` path in fwd/bwd coalescing_store | Eliminates atomics for split-K (non-atomic store + separate reduction) | Medium (fwd/bwd accumulate across taps, so workspace layout differs) | `conv_driver.exe -V 1` with high split counts |
| **P2** | Deeper N-stage pipelining | Not implemented (any arch) | N/A | **All directions** | New work | Add `lds_buffer_num=3`+ support to `wmma_main_loop_t` | 3-stage pipeline hides more global-load latency | High (3-buffer LDS layout, barrier complexity, VGPR pressure) | Timing on memory-bound shapes; rocprof `SQ_INST_CYCLES_VMEM` |
| **P2** | Stream-K for fwd/bwd | `igemm_wrw_gtc_wmma_nhwc.py:1716-1860`; `driver/igemm_wrw_gtc_driver.h:1029-1140` | wrw only | **fwd, bwd** | Requires adaptation (fwd/bwd don't have per-tap epilogue; stream-K loop structure differs) | Port persistent-loop + atomic-claim to fwd/bwd generators | Predictability under contention (CV<1% vs gsplit 36-63%); but wrw measured 1.03-3.8x slower | High (wrw stream-K was 1.03-3.8x slower than gsplit; fwd/bwd may not benefit) | `conv_driver.exe` under contention; compare CV vs mean |

---

## D. gfx1250 WMMA versus gfx950 XDLOPs Comparison

| Area | gfx1250 WMMA | gfx950 XDLOPs | Evidence | Gap | WMMA improvement | Architectural limitation |
|------|-------------|---------------|----------|-----|------------------|--------------------------|
| **Instruction shape** | 16x16x32 (fp16/bf16), 16x16x64 (int8), 16x16x4 (fp32). Single shape per precision. | 5+ shapes per precision: 4x4, 16x16, 32x32 with varied K. ~80+ tile configs per precision. | `wmma.py:76-106` (7 insts); `mfma.py:96-134` (26+ insts); `xdlops_mapping.py:239-523` (~80 entries/table) | WMMA has 1 instruction shape (16x16) vs XDLOPs' 5+. No tile-size choice at instruction level. | N/A — WMMA instruction shape is ISA-fixed. Compensate via richer `wmma_repeat_m/n` and `wave_repeat` space. | WMMA ISA only defines 16x16 output tiles. Cannot match XDLOPs' 32x32 or 4x4. |
| **Wave size** | wave32 (mandatory). `ctrl_wmma_mapping_t.wave_size()=32`. | wave64 (mandatory). `AMDGPU_WAVE_SIZE=64`. | `wmma_mapping.py:56-60`; `amdgpu.py:54` | Different wave size affects occupancy math, VGPR budget, and lane-to-fragment mapping. | N/A — architectural difference. Wave32 halves block_size for same tile, potentially improving occupancy. | Cannot change. Wave32 is gfx1250 ISA requirement for WMMA. |
| **Accumulator storage** | Plain VGPR (`v_c`). No AGPR file. `wmma_main_loop.py:37-38` docstring. | AGPR (separate register file). `accvgpr_unified` mode on gfx90a+ shares VGPR file. `is_accvgpr_unified()` in all 3 generators. | `wmma_main_loop.py:37`; `igemm_fwd_gtc_nhwc.py:229-231`; `mfma.py:86-89` | WMMA accumulator competes with operands for VGPR space. XDLOPs has dedicated AGPR file (or unified mode). | N/A — WMMA has no AGPR. `wmma_acc_high_bank` (Phase 54) moves v_c to VGPR bank-1 as partial compensation. | WMMA ISA uses VGPR for accumulator. No AGPR file on gfx1250. |
| **Pipeline schedules** | **1** schedule (correctness-first). `wmma_main_loop_t.emit()`. | **~10** hand-unrolled schedules dispatched by `wave_repeat_m × n × local_prefetch_num × interleave × lds_buffer_num`. | `wmma_main_loop.py:33-37` (docstring: "Deliberately a single...schedule"); `mfma_main_loop.py:3430-3473` | WMMA has no schedule specialization. XDLOPs has per-tile-shape optimized schedules. | Add 2-3 specialized schedules for common `wave_repeat=4x4` case (single/double-buffer, with/without prefetch). | WMMA latency/hazard semantics differ from MFMA; schedules cannot be copied, must be re-tuned. |
| **Instruction interleaving** | Manual interleave in single schedule (Phase 15, fwd-only). | `create_scheduler()` with `INTERLEAVE_PTN_0/1` auto-interleaves MFMA+global_load+ds_read across machine basic blocks. | `wmma_main_loop.py:101-115`; `mfma_main_loop.py:837-848,1472-1484` | WMMA interleave is fwd-only and manually coded. XDLOPs has automatic MBB-based scheduler. | Port interleave to bwd/wrw (manual, since WMMA has no MBB scheduler). Long-term: port `create_scheduler`. | WMMA's `s_wait_dscnt`/`s_wait_asynccnt`/`s_wait_tensorcnt` counters differ from XDLOPs' `s_waitcnt lgkmcnt/vmcnt`. |
| **Double buffering** | Supported (Phase 2, `lds_buffer_num=1 or 2`). XOR ping-pong. | Default (`lds_buffer_num=2` in `ctrl_mfma_main_loop_t:46`). XOR ping-pong. | `wmma_main_loop.py:47-56`; `mfma_main_loop.py:46` | Both support double-buffer. WMMA defaults to single-buffer; XDLOPs defaults to double-buffer. | Change default to double-buffer in configs (already supported, just not default). | None. |
| **Local prefetch** | Supported (Phase 22, `local_prefetch_num=1 or 2`). | Supported (`local_prefetch_num=1 or 2` in `ctrl_mfma_main_loop_t:47`). | `wmma_main_loop.py:93-99`; `mfma_main_loop.py:47` | Both support 2-slot prefetch. Equivalent. | None needed. | None. |
| **K-sub-loop** | Supported (Phase 1, `unroll_k > inst_wmma.k`). | Not directly equivalent (XDLOPS uses `wave_tile_k` for sub-K tiling). | `wmma_main_loop.py:40-45,93-99` | WMMA has K-sub-loop (amortizes barrier over multiple WMMA issues). XDLOPs has wave_tile_k. Both achieve similar amortization. | None needed. | None. |
| **Global load mechanisms** | 4 paths: default (VGPR-staged), async (global→LDS direct, Phase 13), TDM (tensor_load_to_lds, Phase 28), SADDR (scalar base, Phase 61). | 1 path: default (VGPR-staged) + `tensor_a/b_pass_through` (single-pass-through for skinny gemm). | `igemm_fwd_gtc_wmma_nhwc.py:1361-1700`; `mfma_main_loop.py:88-89,102-552` | WMMA has MORE load mechanisms (async, TDM, SADDR). XDLOPs has pass-through (no LDS for one operand). | Port pass-through to WMMA (skip LDS for one operand in skinny-gemm case). | WMMA's per-lane addressing makes pass-through structurally different from XDLOPs' lanegroup model. |
| **Coalescing store / epilogue** | 4 paths: atomic_add, direct_store, LDS-reshuffle, chunked. `vector_write_out=4`. | LDS-reshuffle with `coalescing_groups` + `feat_vgpr_collapse`. `vector_write_out` up to 16. Atomic with macro switch (gfx940+). | `coalescing_store_wmma.py:466-700`; `coalescing_store.py:862-1500,1181` | WMMA has direct_store (no LDS). XDLOPs has coalescing_groups + VGPR collapse (reuse small VGPR set). XDLOPs has wider vector stores (up to 16 vs 4). | Investigate wider `vector_write_out` for WMMA (currently fixed at 4). | WMMA's per-lane accumulator layout (8 VGPRs/lane) limits vector store width vs XDLOPs' AGPR-grouped layout. |
| **Atomic epilogue** | Scalar `global_atomic_add_f32` per element. `atomic_pack_bf16` packs bf16x2 via V_PERMLANE_XOR_B32. | `buffer_atomic_add_dword_with_macro_switch_t` for gfx940+ (macro-switched encoding). `vector_write_out` up to 16. | `coalescing_store_wmma.py:597`; `coalescing_store.py:1181` | XDLOPs uses buffer atomics (address via SGPR resource descriptor). WMMA uses global atomics (address via VGPR pointer). Different addressing, same semantics. | N/A — different instruction encodings, not a performance gap. Atomic contention measured at zero (`docs/gfx1250_rocprof_profiling.md:39-72`). | gfx1250 ISA uses global_atomic (no buffer-atomic macro switch). |
| **Tile shape space** | 128x128, 64x64, 128x64, 64x128, 32x32, 256x256(fwd), 256x128(bwd). ~7 shapes. | 4x4 to 256x256, ~80+ configs per precision. | `script/generate_all_configs.py` BASE_SECTIONS (26 entries); `xdlops_mapping.py:239-523` | WMMA has far fewer tile shapes. No 4x4, 16x16, 32x32(as output tile), 48x48, 96x96, etc. | Add more `wmma_repeat_m/n` combinations (e.g., 3x4, 4x3, 2x2, 1x4). Requires non-power-of-2 wave grids. | WMMA `waves_per_m/n` must be integer; `wmma_repeat` must divide `gemm_m/n_per_block`. Fewer factorizations than XDLOPs' lanegroup model. |
| **Register pressure** | v_c in VGPR (8 VGPRs/lane for f32-acc, 4 for f16/bf16-acc). No AGPR. Occupancy 25% (128x128), 31% (64x64). | AGPR for accumulator (separate file). accvgpr_unified on gfx90a+. Occupancy higher due to AGPR offloading. | `docs/gfx1250_rocprof_profiling.md:116-155`; `igemm_fwd_gtc_nhwc.py:1014-1017` | WMMA's VGPR-only accumulator is the primary occupancy limiter. XDLOPs' AGPR doubles effective register file. | `wmma_acc_f16/bf16` (halves v_c VGPRs); `wmma_acc_high_bank` (bank-1). Both already implemented. | WMMA ISA has no AGPR. Cannot match XDLOPs' register file doubling. |
| **LDS bank conflicts** | Conflict-free (1.15-1.27 cyc/instr vs 1.0 min). `epilogue_lds_pad` available. | LDS padding via `get_lds_pad()` in `igemm_base.py:996-1024` (XDLOPS only, main loop). | `docs/gfx1250_rocprof_profiling.md:277-321`; `igemm_base.py:996-1024` | Both are near-conflict-free. XDLOPs has main-loop LDS padding; WMMA only has epilogue LDS padding. | None needed (LDS conflicts are NOT a bottleneck per rocprof). | None. |
| **NOP-based hazard resolution** | Not used. WMMA instructions have different latency/hazard semantics. | `get_nop_count_mfma_acc_raw()` emits NOPs after MFMA accumulation to resolve RAW hazards. | `mfma.py:91-93`; `mfma_main_loop.py:549,603` | XDLOPs has explicit NOP insertion for RAW hazards. WMMA relies on implicit pipeline behavior. | Investigate if WMMA has equivalent RAW hazards (unverified — may not need NOPs). | WMMA instruction latency model differs from MFMA. May not have same RAW hazard. |
| **Pass-through (skinny gemm)** | Not implemented. | `emit_single_pass_through()` (line 102) — one operand skips LDS. `pass_through_a/b_interleave_gld`. | `mfma_main_loop.py:88-89,102-552` | WMMA has no pass-through path. For skinny gemm (small K), skipping LDS saves bandwidth + latency. | Add pass-through path for K-small shapes (one operand loads global→VGPR→WMMA directly). | WMMA's per-lane addressing (lane%16 → row/col) makes pass-through structurally different. |
| **General conv (nxe=1) configs** | Generator supports nxe=1 (runtime tap loop), but **NO nxe=1 configs exist**. | All gfx950 configs are nxe=1. | `igemm_fwd_gtc_wmma_nhwc.py` (no nxe check); `config/igemm_*_gfx950_*` (all nxe=1) | gfx1250 has zero general-conv configs despite code support. Biggest coverage gap vs gfx950. | Create nxe=1 config files for common conv shapes (3x3, stride2, etc.). | None — generator already supports it. |
| **Compile-time specialization** | Config-file-driven (flat mode). Each `.config` section = one fully-specialized kernel. | Same (config-file-driven). Plus `seq` mode for combinatorial sweep. | `igemm_codegen.py` (both paths) | Both use compile-time specialization. WMMA lacks `seq` mode (gfx908-only). | N/A — `seq` mode is XDLOPS-specific (lanegroup enumeration). WMMA uses `generate_all_configs.py` instead. | `sequence_driver.py` asserts gfx908/XDLOPs only. |
| **Split-K search** | fwd/bwd: single heuristic. wrw: ternary search + occupancy candidate. | All: `get_gks_list()` returns `[0..max_split]`, each timed. | `driver/igemm_fwd_gtc_driver.h:740-770`; `driver/igemm_wrw_gtc_driver.h:660-1170`; `igemm_gtc_base.h:706-720` | WMMA fwd/bwd use single-heuristic (1 eval). XDLOPs times all splits. WMMA wrw is MORE thorough (ternary + occupancy). | Port wrw's ternary search to fwd/bwd (more launch overhead but better split selection). | None — pure driver-side change. |

---

## E. Ordered Implementation Plan

### Step 1: Re-benchmark direct_store (P0, no code change)
**Prerequisite**: The `_direct` suffix bug is already fixed in code.
- **Action**: Run `script/benchmark_gfx1250_vs_miopen.py --rebuild --direction all` with direct_store configs. Compare direct vs non-direct on same shapes.
- **Files**: No code changes. Uses existing `config/*_direct.config`.
- **Tests**: `conv_driver.exe -V 1` correctness on all direct_store configs (fwd/bwd/wrw × fp16/bf16/fp32/int8).
- **Dependency**: None.
- **Expected outcome**: Quantified direct_store benefit (or lack thereof). Prior data was all invalid.

### Step 2: Create nxe=1 gfx1250 config files (P0, config-only)
- **Action**: Write `config/igemm_{dir}_gtc_gfx1250_nhwc_{prec}_3x3.config` for fwd/bwd/wrw × fp16/bf16. Set `nxe=1`, `nxb=0`, use 128x128 tile.
- **Files**: New config files only.
- **Tests**: `conv_driver.exe convfp16 -n 128 -c 256 -H 28 -W 28 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F 1 -V 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC`
- **Dependency**: None (generator already supports nxe=1).
- **Expected outcome**: General convolution works on gfx1250.

### Step 3: Port `wmma_k_tail + gemm_k_global_split` to fwd/bwd (P0)
- **Action**: Remove asserts in `igemm_base.py:801-808`. Add last-shard remainder logic (`s_knum += s_gemm_k_tail if bz == num_splits-1`) to `driver/igemm_fwd_gtc_driver.h` and `driver/igemm_bwd_gtc_driver.h` WMMA `run()` branches. Add `s_gemm_k_tail`/`s_gemm_k_num_splits` to karg structs.
- **Files**: `python/igemm/igemm_base.py` (remove asserts), `driver/igemm_fwd_gtc_driver.h` (karg + run), `driver/igemm_bwd_gtc_driver.h` (karg + run).
- **Tests**: `conv_driver.exe -V 1` on shapes where `gemm_k % gemm_k_per_block != 0` with `-F 1` (fwd) and `-F 2` (bwd) and gsplit enabled.
- **Dependency**: None.
- **Expected outcome**: Split-K works on non-K-aligned shapes for fwd/bwd.

### Step 4: Port `main_loop_interleave` to bwd (P1)
- **Action**: Study `emit_interleaved_substeps` in `wmma_main_loop.py:330`. Adapt for BWD's transposed B operand (B advances whole K-rows, not contiguous). Add to `igemm_bwd_gtc_wmma_nhwc.py:emit_kernel_fma_main_loop()`. Relax `is_valid` rule 13 in `generate_all_configs.py` for bwd. Create `config/igemm_bwd_gtc_gfx1250_nhwc_fp16_interleave.config`.
- **Files**: `python/igemm/igemm_bwd_gtc_wmma_nhwc.py`, `script/generate_all_configs.py`, new config.
- **Tests**: `conv_driver.exe -V 1` on bwd K-heavy shapes; rocprof `SQ_INST_CYCLES_VALU_WMMA` before/after.
- **Dependency**: None.
- **Risk**: BWD's transposed B interleave is structurally different — B's shared_load reads at `step_bytes = wmma_tile_m * bytes_per_row` (row-to-row), not contiguous. The interleave must pair substep-ks compute with substep-(ks+1)'s global load, but B's global load and LDS store patterns differ from FWD's.

### Step 5: Port `saddr_global_load` to bwd (P1)
- **Action**: Study `_emit_gld_chunk_load` saddr branch in `igemm_fwd_gtc_wmma_nhwc.py`. Adapt for BWD's A operand (untransposed, same as FWD's A). B stays on default path (transposed). Create `config/igemm_bwd_gtc_gfx1250_nhwc_{prec}_saddr.config`.
- **Files**: `python/igemm/igemm_bwd_gtc_wmma_nhwc.py` (global_load_a_functor saddr branch), new configs.
- **Tests**: `conv_driver.exe -V 1`; check VGPR count via disassembly; `gfx1250_occupancy_check.cpp`.
- **Dependency**: None.
- **Expected outcome**: ~4 VGPRs saved on A operand, possibly +1 wave/CU occupancy.

### Step 6: Port `main_loop_interleave` to wrw (P1)
- **Action**: Further adapt Step 4's interleave for WRW's both-transposed operands. WRW's A and B both advance whole K-rows. Create `config/igemm_wrw_gtc_gfx1250_nhwc_fp16_interleave.config`.
- **Files**: `python/igemm/igemm_wrw_gtc_wmma_nhwc.py`, new config.
- **Tests**: `conv_driver.exe -V 1` on wrw K-heavy shapes.
- **Dependency**: Step 4 (bwd interleave informs wrw design).
- **Risk**: WRW has no `async_global_load` — interleave is the only latency-hiding mechanism. WRW's per-iteration B gather (GEMM_K is spatial) complicates the interleave pattern.

### Step 7: WRW addressing redesign — Design A (P1)
- **Action**: Implement fractional col_group design from `docs/gfx1250_wrw_addressing_redesign.md:26-80`. When `gemm_k > gemm_m`, invert the split: `row_stride = block_size / gemm_m_per_block`, multiple threads cooperate on each K-row. Modify `emit_kernel_prologue` in `igemm_wrw_gtc_wmma_nhwc.py` (~30 lines). Create `config/igemm_wrw_gtc_gfx1250_nhwc_{prec}_64x64_k128.config` (already exists for some precisions, extend to all).
- **Files**: `python/igemm/igemm_wrw_gtc_wmma_nhwc.py` (prologue addressing), `python/igemm/igemm_base.py` (remove asserts), new configs.
- **Tests**: `conv_driver.exe -V 1` on wrw shapes with K>64 (e.g., c=128, k=64, 64x64 tile, gemm_k_per_block=128).
- **Dependency**: None.
- **Expected outcome**: Unlocks K=96/128/256 for 64x64 wrw tiles.

### Step 8: Add specialized pipeline schedules (P1)
- **Action**: In `wmma_main_loop.py`, add a second schedule for the common case `wave_repeat_m=4, wave_repeat_n=4, lds_buffer_num=2, local_prefetch_num=1`. Dispatch based on ctrl fields (like `mfma_main_loop.py:3430-3473`). The new schedule should issue the next-tile global load BEFORE the current-tile WMMA burst (currently it's after).
- **Files**: `python/operations/wmma_main_loop.py` (new emit method + dispatch).
- **Tests**: `conv_driver.exe -V 1` on 128x128 tile shapes; rocprof `SQ_INST_CYCLES_VALU_WMMA` before/after.
- **Dependency**: None.
- **Risk**: High — hand-tuned schedules require hardware validation. WMMA latency model is not documented; must be empirically determined.

### Step 9: Port `wmma_epilogue_chunked` + `wmma_acc_high_bank` to wrw (P1)
- **Action**: Lift asserts in `igemm_base.py:746,769`. Add `vgpr_msb_tracker` to wrw generator. Handle per-tap v_c zero + epilogue interaction with VGPR-MSB banking. Create `config/igemm_wrw_gtc_gfx1250_nhwc_bf16_256x256.config`.
- **Files**: `python/igemm/igemm_base.py` (remove asserts), `python/igemm/igemm_wrw_gtc_wmma_nhwc.py` (vgpr_msb_tracker), `python/operations/coalescing_store_wmma.py` (per-tap chunked), new config.
- **Tests**: `conv_driver.exe -V 1` on wrw with 256x256 tile; rocgdb for bank tracking.
- **Dependency**: Step 7 (wrw addressing redesign enables larger K, which benefits from larger tiles).
- **Risk**: High — VGPR-MSB had 7 bugs in fwd/bwd. WRW's per-tap v_c reset adds another dimension. Phase 56 showed 256x256 is NOT a perf win for fwd (~2-3x slower), so wrw may not benefit either.

### Step 10: Port magic division to bwd/wrw (P2)
- **Action**: Port `macro_mdiv_u32_rem_vs_t` calls from fwd (11 sites) to bwd/wrw coordinate paths (group decode, stride_h/w, ho/wo decomposition).
- **Files**: `python/igemm/igemm_bwd_gtc_wmma_nhwc.py`, `python/igemm/igemm_wrw_gtc_wmma_nhwc.py`.
- **Tests**: `conv_driver.exe` timing on group>1 / stride!=1 shapes.
- **Dependency**: None.

---

## F. Unknowns

1. **direct_store performance**: All prior direct_store benchmark data is invalidated by the `_direct` suffix bug. No valid performance data exists. Re-benchmarking is required before any conclusion about its value. (`docs/gfx1250_optimization_backlog.md:77-94`)

2. **WMMA instruction latency model**: The WMMA path has no `get_nop_count_mfma_acc_raw()` equivalent. Whether WMMA instructions have RAW hazards requiring NOPs is **unverified**. The single-schedule pipeline may be leaving latency-hiding opportunities on the table. No ISA documentation consulted for WMMA latency (the `amd-instinct-cdna5-instruction-set-architecture.BIGFILE_DO_NOT_READ` was not read per its name).

3. **256x256 tile performance**: Phase 56 showed 256x256 is ~2-3x slower for fwd. The working theory is "VGPR-driven occupancy (512 vs ~250 VGPRs/wave)" but this is **not root-caused**. Whether wrw 256x256 would also be slower is unknown. (`docs/gfx1250_wmma_vgpr_msb_wip_status.md:342-400`)

4. **Stream-K for fwd/bwd**: WRW Stream-K is 1.03-3.8x slower than gsplit on idle GPU. Whether fwd/bwd would benefit is **unverified** — fwd/bwd have different GEMM axis roles (GEMM_K is channel, not spatial) and don't have per-tap epilogue. (`docs/gfx1250_streamk_design.md:592-615`)

5. **disable_xdl_arb_stall**: ISA doc documents the SCHED_MODE bit[2] semantics but gives no S_SETREG hardware-register ID. **Blocked** on missing register ID. Cannot be evaluated. (`docs/gfx1250_optimization_backlog.md:112-121`)

6. **gfx950 config tuning quality**: gfx950 configs exist but are sparse (3-4 per direction). They were NOT generated by `generate_all_configs.py` (gfx908-only). The gfx950 XDLOPs path's ~80+ tile configs per precision exist in `xdlops_mapping.py` but only a handful are exercised by checked-in configs. **The gfx950 path's tuning coverage is also thin** — comparing "maturity" is complicated by both paths being undertuned.

7. **rocprof data under contention**: All bwd/fwd tail-path rocprof data was measured under heavy GPU contention (100% from other tenants). Absolute WMMA-busy% is 3-8x lower than contention-free measurements. **Re-running on confirmed-idle GPU is an open backlog item** (`docs/gfx1250_optimization_backlog.md:35-38`).

8. **int8 hardware validation**: int8 gemm_k_global_split atomic epilogue (Phase 57, `global_atomic_add_u32`) is code-complete but **NOT hw-validated with signed/large-magnitude data**. (`docs/gfx1250_optimization_backlog.md:340-348`)

9. **bwd TDM large-K slowdown**: TDM is ~5-8% slower for bwd at K=1024 vs non-TDM. Root cause is **unexplained**. (`docs/gfx1250_optimization_backlog.md:196-202`)

10. **Hardware transpose-load**: No correct reference implementation exists for hardware transpose-load on gfx1250. Listed as open Tier 3 item. (`docs/gfx1250_optimization_backlog.md:447-450`)

11. **Phase 42 bwd TDM large-K**: The `s_tdm_k_remain` decrement and `tensor_dim0` rebuild per iteration may introduce a dependency chain that serializes with the TDM load. **Unverified** — no instruction-level analysis performed.

12. **Cross-architecture absolute throughput**: gfx1250 (CDNA5) and gfx950 (CDNA3a) have different CU counts, clock speeds, and memory bandwidth. **Normalized hardware utilization** (WMMA-busy%, occupancy, bandwidth utilization) should be compared, not absolute TFLOPS. No gfx950 rocprof data was collected for this analysis.
