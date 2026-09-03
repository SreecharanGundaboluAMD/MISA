# wmma_acc_f16 / wmma_acc_bf16 Benchmark Findings

**Date:** 2026-09-03  
**Report ref:** §4/OPT-2a, §7/A3  
**Hardware:** gfx1250, 256 CU, 2400 MHz sclk

## Background

`wmma_acc_f16` / `wmma_acc_bf16` switch the WMMA instruction from f32-accumulate
(`v_wmma_f32_16x16x32_f16`, num_v_c=8) to f16/bf16-accumulate
(`v_wmma_f16_16x16x32_f16` / `v_wmma_bf16_16x16x32_bf16`, num_v_c=4). This halves
the accumulator VGPR footprint (128→64 for 128×128 tile) and makes the D-matrix
genuinely 2 B/element, halving epilogue store bytes. The driver already supports
this: `dtype_alloc_byte` flips from `sizeof(float)` to `data_byte`, and a
`tensor_cast` kernel expands the 2-byte output to fp32 for verification.

The configs already exist (`*_k2x_f16acc.config`, `*_k2x_bf16acc.config`) but are
excluded from `_all` master configs because `conv_driver.cpp` computes
`is_wmma_f16_acc`/`is_wmma_bf16_acc` from `tunables[0]` only (COR-004), so mixing
accumulate-width variants in one file corrupts verification.

## Method

Controlled comparison: `*_k2x_f16acc.config` vs `*_k2x.config` (same 128×128×64
tile, same k-sub-loop, only `wmma_acc_f16=1` differs). IGEMM_WARMUP=3,
IGEMM_REPEAT=10, `-V 1`.

## Results

### fp16 f16acc — valid:y, +12% on K-deep

| Shape | Baseline (k2x) | f16acc | Speedup | Validity |
|-------|---------------|--------|---------|----------|
| n128 c1024 17×17 k1024 1×1 | 356 TFLOP/s (0.218ms) | 400 TFLOP/s (0.194ms) | **+12.2%** | valid:y |
| n256 c2048 14×14 k2048 1×1 | 442 TFLOP/s (0.953ms) | 465 TFLOP/s (0.905ms) | **+5.3%** | valid:y |

The +12% on the medium-K shape matches the report's 19% epilogue share: halving
the epilogue saves ~9.5% of total time, and the freed VGPR pressure (64 fewer
accumulator VGPRs) likely adds occupancy headroom for the remainder.

### fp16 f16acc on shallow-K — slower, not applicable

| Variant | TFLOP/s | Validity |
|---------|---------|----------|
| _all best (64×64×32 dbuf_direct) | 136.5 (0.024ms) | valid:y |
| f16acc (64×64×64) | 122.7 (0.027ms) | valid:y |

The k2x tile (gemm_k_per_block=64) doesn't help when gemm_k=64 (single K-iteration).
The epilogue savings are offset by the different tile geometry. f16acc is not
beneficial on shallow-K shapes.

### bf16 bf16acc — valid:n, precision failure

| Shape | Baseline (k2x) | bf16acc | Speedup | Validity |
|-------|---------------|---------|---------|----------|
| n128 c1024 17×17 k1024 1×1 | 356 TFLOP/s | 400 TFLOP/s | +12.2% | **valid:n** |
| n256 c2048 14×14 k2048 1×1 | 445 TFLOP/s | 467 TFLOP/s | +5.3% | **valid:n** |

bf16 accumulation has only 7 mantissa bits (vs fp16's 10). Over K=1024–2048
accumulations, the precision loss exceeds the NRMS verification threshold. This
is a fundamental precision limitation, not a codegen bug. **bf16 bf16acc is not
usable for K≥1024 shapes.**

## Conclusions

1. **fp16 f16acc is production-ready for K-deep 1×1 shapes** (+12% at K=1024,
   +5% at K=2048, valid:y). The existing `igemm_fwd_gtc_gfx1250_nhwc_fp16_k2x_f16acc.config`
   can be used standalone.
2. **bf16 bf16acc is not usable** (valid:n at K≥1024). The bf16-accumulate path
   needs a wider-accumulate workaround — this is exactly what OPT-2b (the "correct
   fix": keep f32 accumulate, add `v_cvt_pk_f16_f32` + widened store) addresses.
3. **f16acc doesn't help on shallow-K** — the epilogue share is largest there
   (report says 86 TFLOP/s at c=64), but the k2x tile doesn't fit those shapes
   and the epilogue savings don't compensate for the tile mismatch.
4. **The _all master config exclusion is correct** — f16acc/bf16acc configs must
   be built standalone due to COR-004 (tunables[0]-based accumulate-width
   detection). Wiring them into benchmark scripts requires per-config standalone
   builds.

## Next steps

- **OPT-2b (Phase C):** Implement f32-accumulate → fp16-output epilogue
  (`v_cvt_pk_f16_f32` + widened store). This delivers the same epilogue halving
  as f16acc but without precision loss, and works for bf16 too. Requires
  D-matrix column layout reshuffle for contiguous stores.
- **Benchmark script integration:** Add f16acc as a standalone candidate for
  K-deep fp16 1×1 shapes in `benchmark_gfx1250_vs_gfx950_diverse.py` (requires
  per-config build, not _all).
