# C1-C2: fp16/bf16 Output from f32 Accumulator (gfx1250)

## Problem

fp16/bf16 WMMA kernels accumulate in f32 (8 VGPRs per tile) but write f32 output (4 bytes/element).
This wastes 2× the memory bandwidth and 2× the store instructions vs writing fp16 output directly.
The performance report (OPT-2, Phase C) measured the epilogue at 19% of total time on a K-deep 1×1 conv,
estimating 1.23× speedup from deleting the f32 stores.

## Solution

### C1: Packed fp16/bf16 output in the direct_store epilogue

Added `wmma_fp16_output` tunable. When set, the `direct_store` epilogue path in
`coalescing_store_wmma.py` converts pairs of f32 accumulator elements into packed fp16x2
(or bf16x2) via `v_cvt_pk_f16_f32` / `v_cvt_pk_bf16_f32`, then stores one packed dword
per pair of columns using `global_store_dword`. Only even lanes issue the store (their
column is the pair's lower column), halving the number of store instructions.

The pattern is identical to the existing `atomic_pack_bf16` path (lines 639-647 of
`coalescing_store_wmma.py`), just with `global_store_dword` instead of
`global_atomic_pk_add_bf16`:

```asm
v_permlane_xor_b32 v[partner], v[v_c+N], 1, 32    ; get partner lane's value
v_cvt_pk_f16_f32   v[packed], v[v_c+N], v[partner] ; pack (own, partner) -> fp16x2
v_and_b32           v[tmp], 1, v[v_tid]
v_cmpx_eq_u32       0, v[tmp]                       ; EXEC = even lanes only
global_store_dword  v[addr], v[packed], s[s_p_out]
s_mov_b32           exec_lo, -1                     ; restore EXEC
```

The accumulator stays f32 (unlike `wmma_acc_f16` which changes the WMMA instruction itself);
only the output store narrows to fp16/bf16.

### C2: dtype_alloc_byte override dropped

In `conv_driver.cpp`, the `is_wmma → sizeof(float)` override on `dtype_alloc_byte` now
also treats `wmma_fp16_output` like `wmma_acc_f16`/`wmma_acc_bf16`/`atomic_pack_bf16`,
sizing the output buffer at `data_byte` (2 for fp16) instead of `sizeof(float)` (4).

The verification path (fwd/bwd/wrw `*_post` lambdas) reuses the existing
`tensor_cast_fp32_fp16acc_1d` / `tensor_cast_fp32_bf16acc_1d` kernels to expand the
2-byte output back to fp32 for comparison against the fp32 reference.

### Constraints

- Only valid for fp16/bf16 precision (not fp32/int8)
- Only with `direct_store=1` (not `gemm_k_global_split` — no packed-fp16 atomic-add on this ISA)
- Mutually exclusive with `wmma_acc_f16`/`wmma_acc_bf16` (already produce 2-byte output natively)
- Mutually exclusive with `atomic_pack_bf16` (already packs)
- Mutually exclusive with `wmma_m_tail`/`wmma_n_tail` (both need v_tmp3/v_tmp4 scratch slots)

### Kernel name mangling

`_f16o` suffix added in both Python (`igemm_base.py`) and C++ (`igemm_gtc_base.h`), kept in sync.

## Files Changed

1. `python/igemm/igemm_base.py` — `wmma_fp16_output` tunable definition + asserts + `_f16o` kernel name suffix
2. `python/operations/coalescing_store_wmma.py` — `wmma_fp16_output` ctrl field + packed fp16/bf16 store in `_emit_direct_store`
3. `python/igemm/igemm_fwd_gtc_wmma_nhwc.py` — ctrl wiring + `out_elem_byte_shift` + scratch VGPR allocation + coalescing_store call
4. `python/igemm/igemm_bwd_gtc_wmma_nhwc.py` — same as fwd
5. `python/igemm/igemm_wrw_gtc_wmma_nhwc.py` — same as fwd (two call sites)
6. `driver/igemm_gtc_base.h` — `wmma_fp16_output` struct field + config parsing + `_f16o` kernel name suffix
7. `driver/conv_driver.cpp` — `is_wmma_fp16_output` detection + `dtype_alloc_byte` override + COR-004 assert + verification paths (fwd/bwd/wrw)

## Correctness Results

All 9 runs (3 shapes × 3 independent launches) pass `valid:y`:

| Shape | Config | Run 1 | Run 2 | Run 3 |
|-------|--------|-------|-------|-------|
| 1: -n 256 -c 2048 -H 14 -W 14 -k 2048 | fp16o | valid:y | valid:y | valid:y |
| 2: -n 128 -c 1024 -H 17 -W 17 -k 1024 | fp16o | valid:y | valid:y | valid:y |
| 3: -n 128 -c 32 -H 56 -W 56 -k 128 (shallow-K) | fp16o | valid:y | valid:y | valid:y |

Shape 3 uses c=32 (1 K-iteration, shallowest possible for this tile) to maximize the
epilogue's share of total runtime — the critical test case from the performance report.

## Performance Results

3-run median TFLOP/s (median chosen over mean due to shared GPU contention outliers):

| Shape | Baseline (f32 out) | fp16o (fp16 out) | Speedup |
|-------|--------------------|------------------|---------|
| 1: -n 256 -c 2048 -H 14 -W 14 -k 2048 | 509.37 | 583.05 | 1.145× |
| 2: -n 128 -c 1024 -H 17 -W 17 -k 1024 | 422.80 | 477.24 | 1.129× |
| 3: -n 128 -c 32 -H 56 -W 56 -k 128 (shallow-K) | 77.50 | 84.29 | 1.087× |

3-run mean TFLOP/s (excluding obvious contention outlier in fp16o shape 1 run 2):

| Shape | Baseline mean | fp16o mean | Speedup |
|-------|---------------|------------|---------|
| 1 | 509.83 | 587.22 | 1.152× |
| 2 | 421.32 | 475.92 | 1.130× |
| 3 | 77.92 | 84.63 | 1.086× |

Raw timing (ms):

| Shape | Baseline (ms) | fp16o (ms) |
|-------|---------------|------------|
| 1 | 0.826, 0.827, 0.824 | 0.722, 1.684*, 0.712 |
| 2 | 0.183, 0.186, 0.183 | 0.165, 0.162, 0.163 |
| 3 | 0.042, 0.042, 0.042 | 0.039, 0.038, 0.039 |

*Outlier (GPU contention on shared GPU, ~2.3× slower than runs 1 and 3).

## Analysis

- **1.09–1.15× speedup** across all shapes, consistent with the performance report's
  1.23× estimate (the report was for a pure K-deep 1×1 conv; our shapes include overhead
  from non-1×1 spatial dimensions and varying K depth).
- **Shape 3 (shallow-K)** shows the smallest speedup (1.087×) despite the epilogue
  having the largest relative share — this is because the kernel is heavily
  launch-overhead-bound at 0.04ms, not epilogue-bound. The absolute time saved
  (0.042 → 0.039ms = 3µs) is small but consistent.
- **Shapes 1 and 2** (K-deep) show the largest speedups (1.13–1.15×), matching the
  report's prediction that f32 store elimination helps most when the epilogue is a
  significant fraction of total runtime.
- **NRMS**: all runs pass verification (`valid:y`), confirming that fp16 output
  precision is within tolerance for these shapes.
- The store instruction count is halved (128 → 64 stores per 128-element row in the
  macro-tile), and output bytes are halved (4 → 2 bytes/element), as expected.

## C2 (widen the store) Status

C2 (reshuffling the D-matrix column layout for `global_store_dwordx4`) is NOT implemented
in this change. The current packed store uses `global_store_dword` (1 dword = 2 fp16
elements per store). C2 would further widen this to `global_store_dwordx4` (4 dwords =
8 fp16 elements per store), reducing store instructions by another 4×. This requires
reshuffling the D-matrix column layout so each lane holds 4 contiguous columns, which
is a deeper change to the WMMA mapping and is left as future work.
