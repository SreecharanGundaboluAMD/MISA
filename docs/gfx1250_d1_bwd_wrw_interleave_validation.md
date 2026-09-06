# D1: main_loop_interleave Validation for BWD and WRW

## Foreword — Correction to Prior fwd Documentation

The document `docs/gfx1250_d1_main_loop_interleave_validation.md` (which validated fwd's main_loop_interleave mechanism) stated that "no shipped config enables main_loop_interleave." This was inaccurate.

**Fact:** `config/igemm_fwd_gtc_gfx1250_nhwc_fp16_interleave.config` already existed in the repo with `main_loop_interleave=1` and `lds_double_buffer=1` before that validation ran. The prior agent built an equivalent config from scratch in `/tmp` instead of discovering and using the pre-existing one. The measured performance numbers in that doc (fwd's +4.5–7.0% improvement) remain valid and correctly reflect the tunable settings tested, but the framing about "never enabled" was incorrect—the config was already shipped.

---

## What Was Tested

Two configs for **bwd** (backward/gradient-of-input propagation):

1. **Baseline:** `config/igemm_bwd_gtc_gfx1250_nhwc_fp16_k2x.config` — `main_loop_interleave=0` (default), `lds_double_buffer=0` (default)
2. **Interleave variant:** `config/igemm_bwd_gtc_gfx1250_nhwc_fp16_interleave.config` — `main_loop_interleave=1`, `lds_double_buffer=1`

Both: `gemm_m_per_block=128, gemm_n_per_block=128, gemm_k_per_block=64` (k2x), fp16, 1×1 convolutions only (plain GEMMs).

**WRW (weight-gradient propagation):** Pre-existing correctness bug blocks benchmarking (see "WRW Status" below).

---

## Design Note: A/B Operand Interleaving Strategy

Both bwd and wrw implement `main_loop_interleave` with an **asymmetric operand strategy**:

- **A operand interleaves:** In bwd, A is grad_output (untransposed); in wrw, A is grad_output (transposed). Interleaving directly overlaps K-loop global loads of A with WMMA compute of prior K-blocks.
- **B operand does NOT interleave:** In bwd, B is weight (transposed); in wrw, B is input (transposed). Both are accessed via a deferred bulk-LDS path and remain outside the interleaved K-loop schedule. This is an intentional, known, documented limitation.

See config header comments for implementation details. This asymmetry is by design, not a gap to fix.

---

## BWD Correctness Results

Both variants passed correctness validation (`valid:y`) on all tested shapes:

| Variant              | Shape 1 (128×1024×17) | Shape 2 (256×2048×14) |
|----------------------|-----------------------|------------------------|
| Baseline             | **valid:y**           | **valid:y**            |
| main_loop_interleave | **valid:y**           | **valid:y**            |

Test command template (mode `-F 2` for backward):
```
conv_driver.exe convfp16 -n {N} -c {C} -H {H} -W {W} -k {K} -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F 2 -V 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
```

---

## BWD Performance Results

All runs: `IGEMM_WARMUP=5 IGEMM_REPEAT=20`, 3 independent process launches each.

### Shape 1: `128×1024×17×17` (1×1 filter, K=1024, K-substeps=32)

| Variant              | Run 1  | Run 2  | Run 3  | Avg    | Min    | Max    |
|----------------------|--------|--------|--------|--------|--------|--------|
| Baseline             | 171.72 | 171.39 | 172.12 | 171.74 | 171.39 | 172.12 |
| main_loop_interleave | 177.52 | 177.68 | 178.54 | 177.91 | 177.52 | 178.54 |

**Δ = +3.59% average TFLOP/s** (171.74 → 177.91)

### Shape 2: `256×2048×14×14` (1×1 filter, K=2048, K-substeps=64)

| Variant              | Run 1  | Run 2  | Run 3  | Avg    | Min    | Max    |
|----------------------|--------|--------|--------|--------|--------|--------|
| Baseline             | 214.90 | 215.44 | 215.45 | 215.26 | 214.90 | 215.45 |
| main_loop_interleave | 223.95 | 223.63 | 224.08 | 223.89 | 223.63 | 224.08 |

**Δ = +4.01% average TFLOP/s** (215.26 → 223.89)

### Run-to-run Noise Analysis

| Variant   | Shape | Std dev (TFLOP/s) | Max range (min→max) |
|-----------|-------|-------------------|---------------------|
| Baseline  | 1     | 0.32              | 0.73                |
| Interleave| 1     | 0.43              | 1.02                |
| Baseline  | 2     | 0.25              | 0.55                |
| Interleave| 2     | 0.19              | 0.45                |

Both variants maintain tight run-to-run consistency (std dev ≤0.43 TFLOP/s). The interleave improvement (3.59–4.01%) is well outside the per-variant noise bands in both shapes.

---

## WRW Correctness Results

Both variants passed correctness validation (`valid:y`) on the required shape (K=2048/C=2048, 256-workgroup grid):

| Variant              | K=2048, C=2048 (256-CU grid) |
|----------------------|------------------------------|
| Baseline             | **valid:y** ✓                |
| main_loop_interleave | **valid:y** ✓                |

**Note:** This result required the correctness fix from commit `10e63c5` (missing `s_wait_loadcnt 0x0` in `_emit_sst_all_chunks`, `python/igemm/igemm_wrw_gtc_wmma_nhwc.py`). Prior builds of the k2x configs were non-functional. The fix was orthogonal to main_loop_interleave; both baseline and interleave failed before the fix, and both pass after it.

Test command template (mode `-F 4` for wrw):
```
conv_driver.exe convfp16 -n {N} -c {C} -H {H} -W {W} -k {K} -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F 4 -V 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
```

---

## WRW Performance Results

All runs: `IGEMM_WARMUP=5 IGEMM_REPEAT=20`, 3 independent process launches.

### Shape: `256×2048×14×14` (1×1 filter, K=2048, C=2048, K-substeps=64, full 256-workgroup grid)

| Variant              | Run 1  | Run 2  | Run 3  | Avg    | Min    | Max    |
|----------------------|--------|--------|--------|--------|--------|--------|
| Baseline             | 112.23 | 112.33 | 112.60 | 112.39 | 112.23 | 112.60 |
| main_loop_interleave | 121.60 | 121.97 | 122.21 | 121.93 | 121.60 | 122.21 |

**Δ = +8.49% average TFLOP/s** (112.39 → 121.93)

### Run-to-run Noise Analysis

| Variant   | Std dev (TFLOP/s) | Max range (min→max) |
|-----------|-------------------|---------------------|
| Baseline  | 0.17              | 0.37                |
| Interleave| 0.28              | 0.61                |

Both variants maintain tight run-to-run consistency (std dev ≤0.28 TFLOP/s). The interleave improvement (+8.49%) is well outside the per-variant noise bands.
---

## Summary and Conclusions

### BWD: Consistent Measurable Win

The `main_loop_interleave=1` mechanism produces a **statistically significant performance improvement for bwd** across both tested shapes:

- **Shape 1 (128×1024×17×17, K=1024):** +3.59% improvement
- **Shape 2 (256×2048×14×14, K=2048):** +4.01% improvement

Both improvements are consistent with the fwd results (+4.5–7.0%), validating that the interleave strategy generalizes well across the backward pass.

### WRW: Strong Win (After Fix)

The `main_loop_interleave=1` mechanism produces a **notably larger performance improvement for wrw** than for bwd:

- **Shape (256×2048×14×14, K=2048/C=2048):** +8.49% improvement

This is substantially higher than bwd (+3.59–4.01%), suggesting that wrw's memory-access patterns interact particularly favorably with the interleave mechanism (likely due to transposed-operand ordering in gradient computation).

**Note on correctness:** The wrw k2x results required the fix from commit `10e63c5` (missing `s_wait_loadcnt 0x0` in `_emit_sst_all_chunks`). Prior builds were non-functional, but this was a pre-existing correctness bug in k2x tuning, not a limitation of main_loop_interleave. Both baseline and interleave k2x configs pass correctness after the fix.

### Overall Assessment

**`main_loop_interleave=1` is a valid, measurable performance win for both bwd and wrw**, delivering:

- **BWD:** +3.6–4.0% throughput improvement (2 shapes)
- **WRW:** +8.5% throughput improvement (K=2048/C=2048 shape)

Both improvements are consistent, reproducible, and well outside run-to-run noise bands. The mechanism correctly interleaves A-operand loads with compute while leaving B-operand on the deferred path—this asymmetry is by design and works well for both directions.

**Recommendation:** Like fwd, `main_loop_interleave=1` should be promoted to a swappable performance tuning parameter in the master k2x config family for both bwd and wrw, with the required `lds_double_buffer=1` (already standard practice in other shipped tunings).

---

## Testing Metadata

- **GPU:** gfx1250, 256 CUs
- **ROCm:** /home/sgundabo/rocm-10.1
- **Build:** Generated via `igemm_codegen.py` from configs
- **Baseline configs:** `config/igemm_bwd_gtc_gfx1250_nhwc_fp16_k2x.config`, `config/igemm_wrw_gtc_gfx1250_nhwc_fp16_k2x.config`
- **Interleave configs:** `config/igemm_bwd_gtc_gfx1250_nhwc_fp16_interleave.config`, `config/igemm_wrw_gtc_gfx1250_nhwc_fp16_interleave.config`
- **Validation date:** 2026-09-06
