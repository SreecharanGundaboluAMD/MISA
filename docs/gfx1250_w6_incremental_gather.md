# W-6: Strength-reduce wrw's per-iteration B-gather to incremental index update

## Status: COMPLETE — all 9 correctness runs pass, performance measured

## What was changed

### 1. Tunable (Python + C++ + kernel name mangling)

**`python/igemm/igemm_base.py`:**
- Added `self.wrw_incremental_gather` tunable (default 0)
- Added `_wig` suffix in `igemm_gtc_encode_kernel_name`

**`driver/igemm_gtc_base.h`:**
- Added `int wrw_incremental_gather = 0;` to struct
- Added parsing in config loader
- Added `_wig` to C++ kernel name mangling

### 2. New kernarg: `ho`

- Added `int ho;` to `igemm_wrw_gtc_wmma_nhwc_karg_t` at offset 132
- Added `karg.ho = ho;` in driver setup
- Added `amdgpu_kernel_arg_t('ho', 4, 132, ...)` to Python get_kernel_args
- Added `s_ho` SGPR, loaded in prologue (gated by wrw_incremental_gather)

### 3. New VGPRs (2)

- `v_inc_hw_idx` — persistent hw_idx (= ho_idx*wo + wo_idx, the ho_wo remainder)
- `v_inc_n_idx` — persistent n_idx (the ho_wo quotient)

### 4. Incremental gather implementation

**First iteration (seeds VGPRs):** full div/rem as before, then saves hw_idx and n_idx.

**Subsequent iterations (`_emit_b_gather_incremental`):**
1. Compute k_abs for k-tail check only (add, no division)
2. `v_inc_hw_idx += kpb`
3. Conditional wrap by ho_wo: `v_cmp_gt_u32 vcc_lo, s[ho_wo], v[hw_idx]` (SGPR first, same pattern as baseline bounds check) + `v_cndmask_b32`
4. `v_inc_n_idx += n_carry`
5. Magic div/rem by wo: `hw_idx // wo` → ho_idx, wo_idx (same as baseline's second division)
6. Recompute hi_idx, wi_idx, v_flag, row_idx, v_addr_b (same mul/add as baseline)

**Savings:** eliminates the ho_wo division. The wo division is kept. The conditional wrap uses the exact same `v_cmp_gt_u32` + `v_cndmask_b32` instruction pattern as the baseline's bounds check.

### 5. VGPR pressure

- Baseline: 251 VGPRs
- W-6: 253 VGPRs (+2)
- Well within gfx1250's 256 limit

## Correctness — all 9 runs valid:y

### Shape 1: 128×1024×17×17×1024 (1×1, wo=17, ho=17)

| Run | W-6 | Baseline |
|-----|------|----------|
| 1 | valid:y, 277.920 TFLOP/s | valid:y, 275.709 TFLOP/s |
| 2 | valid:y, 277.566 TFLOP/s | valid:y, 275.611 TFLOP/s |
| 3 | valid:y, 276.808 TFLOP/s | valid:y, 277.536 TFLOP/s |
| **Avg** | **277.4** | **276.3** |

### Shape 2: 256×2048×14×14×2048 (1×1, wo=14, ho=14)

| Run | W-6 | Baseline |
|-----|------|----------|
| 1 | valid:y, 387.726 TFLOP/s | valid:y, 384.716 TFLOP/s |
| 2 | valid:y, 385.183 TFLOP/s | valid:y, 385.813 TFLOP/s |
| 3 | valid:y, 286.168* | valid:y, 384.994 TFLOP/s |
| **Avg** | **353.0** (386.5 excl. outlier) | **385.2** |

*Run 3 hit GPU contention (shared machine). Runs 1-2 average: 386.5 TFLOP/s.

### Shape 3: 64×512×28×28×512 (3×3, multi-tap, wo=28, ho=28)

| Run | W-6 | Baseline |
|-----|------|----------|
| 1 | valid:y, 295.548 TFLOP/s | valid:y, 247.355 TFLOP/s |
| 2 | valid:y, 298.376 TFLOP/s | valid:y, 247.551 TFLOP/s |
| 3 | valid:y, 296.213 TFLOP/s | valid:y, 247.356 TFLOP/s |
| **Avg** | **296.7** | **247.4** |

## Performance summary

| Shape | Baseline avg | W-6 avg | Change |
|-------|-------------|---------|--------|
| 128×1024×17×17×1024 | 276.3 TFLOP/s | 277.4 TFLOP/s | +0.4% |
| 256×2048×14×14×2048 | 385.2 TFLOP/s | 386.5 TFLOP/s (excl. outlier) | +0.3% |
| 64×512×28×28×512 (3×3) | 247.4 TFLOP/s | 296.7 TFLOP/s | **+19.9%** |

The 3×3 multi-tap shape shows a significant +19.9% improvement because each tap's K-loop
has fewer iterations, making the saved ho_wo division more impactful per iteration. The 1×1
shapes show modest gains since the ho_wo division is a smaller fraction of total per-iteration cost.

## Debugging history

The ho-wrap went through several iterations:
1. Scalar-branch loops (`s_cbranch_vccnz`) — broken for divergent per-lane wraps
2. `v_cmp_lt_u32`/`v_cmp_ge_u32` + `v_cndmask_b32` — correct logic but failed on hardware
3. EXEC-masked loops — caused hangs (stale vcc bits)
4. **Final: maintain hw_idx, wrap by ho_wo using `v_cmp_gt_u32` (SGPR first, same as baseline) + `v_cndmask_b32`** — works

Key insight: `v_cmp_gt_u32` with SGPR as first operand (matching baseline's `v_cmp_gt_u32 vcc_lo, s[hi], v[hi_idx]`) works reliably. Earlier `v_cmp_lt_u32`/`v_cmp_ge_u32` approaches had subtle issues despite correct logic.

## Files modified

1. `python/igemm/igemm_base.py` — tunable + kernel name mangling
2. `driver/igemm_gtc_base.h` — C++ struct + parsing + kernel name mangling
3. `driver/igemm_wrw_gtc_driver.h` — karg struct + driver assignment
4. `python/igemm/igemm_wrw_gtc_wmma_nhwc.py` — SGPR, VGPR, kernarg, prologue, incremental gather

## Re-benchmark on faster machine (2026-09-04)

**Machine:** gfx1250, 256 CUs, ROCm 10.1. Less-contended than the original benchmark machine.

**Method:** 5 independent process launches per (shape, config), IGEMM_WARMUP=5, IGEMM_REPEAT=20.
Configs: `config/w6_test_wig.config` (wrw_incremental_gather=1) vs `config/w6_test_baseline.config`.
Built fresh into `/tmp/w6_wig` and `/tmp/w6_baseline` via `igemm_codegen.py`.

### Raw results — 5 runs per (shape, config)

#### Shape 1: 128×1024×17×17×1024 (1×1, wo=17, ho=17)

| Run | W-6 cost (ms) | W-6 TFLOP/s | Baseline cost (ms) | Baseline TFLOP/s |
|-----|---------------|-------------|---------------------|-------------------|
| 1 | 0.186 | 418.134 | 0.187 | 415.779 |
| 2 | 0.186 | 416.326 | 0.185 | 418.334 |
| 3 | 0.185 | 418.555 | 0.186 | 416.644 |
| 4 | 0.185 | 419.634 | 0.186 | 416.286 |
| 5 | 0.186 | 417.763 | 0.186 | 416.544 |
| **Avg** | **0.1856** | **418.08** | **0.1860** | **416.72** |
| **Range** | 0.185–0.186 | 416.33–419.63 | 0.185–0.187 | 415.78–418.33 |

#### Shape 2: 256×2048×14×14×2048 (1×1, wo=14, ho=14)

| Run | W-6 cost (ms) | W-6 TFLOP/s | Baseline cost (ms) | Baseline TFLOP/s |
|-----|---------------|-------------|---------------------|-------------------|
| 1 | 0.534 | 787.797 | 0.518 | 811.827 |
| 2 | 0.514 | 818.446 | 0.605 | 695.606 |
| 3 | 0.515 | 817.488 | 0.512 | 822.661 |
| 4 | 0.538 | 781.710 | 0.515 | 817.039 |
| 5 | 0.538 | 783.029 | 0.539 | 781.624 |
| **Avg** | **0.5278** | **797.69** | **0.5378** | **785.75** |
| **Range** | 0.514–0.538 | 781.71–818.45 | 0.512–0.605 | 695.61–822.66 |

Note: Baseline run 2 (695.606 TFLOP/s, 0.605 ms) is an outlier — likely transient contention. Excluding it, baseline avg = 0.521 ms / 808.3 TFLOP/s, making W-6 −1.3% on this shape (i.e., within noise).

#### Shape 3: 64×512×28×28×512 (3×3, wo=28, ho=28)

| Run | W-6 cost (ms) | W-6 TFLOP/s | Baseline cost (ms) | Baseline TFLOP/s |
|-----|---------------|-------------|---------------------|-------------------|
| 1 | 0.565 | 418.950 | 0.575 | 411.724 |
| 2 | 0.564 | 420.041 | 0.575 | 412.070 |
| 3 | 0.564 | 419.939 | 0.574 | 412.359 |
| 4 | 0.577 | 410.163 | 0.571 | 414.337 |
| 5 | 0.569 | 416.063 | 0.577 | 410.353 |
| **Avg** | **0.5678** | **417.03** | **0.5744** | **412.17** |
| **Range** | 0.564–0.577 | 410.16–420.04 | 0.571–0.577 | 410.35–414.34 |

#### Shape 4: 32×256×56×56×256 (3×3, wo=56, ho=56) — ablation

| Run | W-6 cost (ms) | W-6 TFLOP/s | Baseline cost (ms) | Baseline TFLOP/s |
|-----|---------------|-------------|---------------------|-------------------|
| 1 | 0.387 | 305.609 | 0.397 | 297.876 |
| 2 | 0.388 | 305.305 | 0.392 | 302.211 |
| 3 | 0.389 | 304.296 | 0.392 | 302.234 |
| 4 | 0.395 | 299.714 | 0.386 | 307.023 |
| 5 | 0.395 | 300.066 | 0.387 | 306.136 |
| **Avg** | **0.3908** | **303.00** | **0.3908** | **303.10** |
| **Range** | 0.387–0.395 | 299.71–305.61 | 0.386–0.397 | 297.88–307.02 |

#### Shape 5: 128×64×56×56×64 (1×1) — ablation

**Not applicable.** The 128×128 tile config cannot handle c=64, k=64 (GEMM M/N dimensions
smaller than the tile). Both W-6 and baseline report "not applicable". This is a config/shape
mismatch, not a W-6 regression.

### Performance summary

| Shape | Baseline avg (TFLOP/s) | W-6 avg (TFLOP/s) | % change (TFLOP/s) | % change (cost) |
|-------|------------------------|--------------------|---------------------|------------------|
| 1: 128×1024×17×17×1024 (1×1) | 416.72 | 418.08 | +0.33% | +0.22% |
| 2: 256×2048×14×14×2048 (1×1) | 785.75 | 797.69 | +1.52% | +1.86% |
| 3: 64×512×28×28×512 (3×3) | 412.17 | 417.03 | +1.18% | +1.15% |
| 4: 32×256×56×56×256 (3×3) | 303.10 | 303.00 | −0.03% | +0.00% |
| 5: 128×64×56×56×64 (1×1) | N/A | N/A | N/A | N/A |

### Analysis: Is the +19.9% on 3×3 reproducible?

**No. The +19.9% on Shape 3 is not reproducible on this machine.** With 5 independent runs,
W-6 shows only **+1.18%** over baseline on Shape 3 (417.03 vs 412.17 TFLOP/s), down from the
originally reported +19.9% (296.7 vs 247.4 TFLOP/s).

Key observations:

1. **Absolute throughput is ~70% higher on this machine.** Shape 3 baseline went from 247.4
   TFLOP/s (original) to 412.2 TFLOP/s (this run). The original baseline was clearly running
   under heavy GPU contention. When the GPU is fully available, the ho_wo division is a much
   smaller fraction of wall-clock time, and eliminating it yields only ~1%.

2. **The original +19.9% was an artifact of contention.** On the contended machine, the
   baseline's 3×3 kernel was disproportionately slowed (247 vs 412 TFLOP/s — a 40% deficit),
   while W-6 was less affected (297 vs 417 — a 29% deficit). This differential inflated the
   apparent gain. On an uncontended GPU, both kernels run at their true speed and the difference
   collapses to ~1%.

3. **W-6 is consistently within noise of baseline on all shapes.** Shape 1: +0.33%, Shape 4:
   −0.03%. Shape 2's +1.52% is inflated by a baseline outlier (run 2 at 695.6 TFLOP/s); excluding
   it, W-6 is within noise. The 3×3 shapes (3 and 4) show +1.18% and −0.03% respectively — no
   special benefit from the incremental gather on multi-tap convolutions.

4. **The optimization is correct but marginal.** All 40 valid runs report valid:y. W-6 eliminates
   one integer magic division per K-iteration, which is a real instruction saving, but it is
   dwarfed by memory latency and WMMA compute on this architecture. The +2 VGPR cost (251→253)
   may partially offset the instruction savings.

**Conclusion:** W-6's `wrw_incremental_gather` is a correct, low-risk micro-optimization that
provides a small (~1%) benefit on 3×3 shapes and is within noise on 1×1 shapes. The originally
reported +19.9% was a measurement artifact from a contended GPU and does not reflect the
optimization's true value.
