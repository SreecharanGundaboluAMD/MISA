# W-4: 64×64 vs 128×128 WRW WMMA Tile Size Comparison (gfx1250)

**Date:** 2026-09-04
**GPU:** gfx1250 (256 CUs)
**Precision:** FP16
**Direction:** WRW (weight-recurrent)
**Layout:** NHWC/NHWC/NHWC
**Filter:** 3×3, stride 1, no padding

## Configurations Tested

| Label | Tile (M×N×K) | lds_row_pad |
|-------|-------------|-------------|
| A | 128×128×32 | 0 |
| B | 128×128×32 | 16 |
| C | 64×64×32 | 0 |
| D | 64×64×32 | 16 |

All configs: `gemm_k_global_split = 1`, `cumode = 0`, `nxb = nxe = 0`, `wavefront_size = 32`.

## Results

### Shape 1: Medium 1×1 (`-n 128 -c 1024 -H 17 -W 17 -k 1024`)

| Tile | lds_row_pad | TFLOP/s | Cost (ms) | Valid |
|------|-------------|---------|-----------|-------|
| 128×128 | 0 | 299.4 | 0.259 | y |
| 128×128 | 16 | 416.9 | 0.186 | y |
| 64×64 | 0 | 329.9 | 0.235 | y |
| 64×64 | 16 | 353.5 | 0.219 | y |

### Shape 2: Large 1×1 (`-n 256 -c 2048 -H 14 -W 14 -k 2048`)

| Tile | lds_row_pad | TFLOP/s | Cost (ms) | Valid |
|------|-------------|---------|-----------|-------|
| 128×128 | 0 | 480.8 | 0.875 | y |
| 128×128 | 16 | **824.2** | **0.511** | y |
| 64×64 | 0 | 470.0 | 0.895 | y |
| 64×64 | 16 | 513.7 | 0.819 | y |

## Analysis

### Does 128×128 with padding beat 64×64 with padding?

**Yes, decisively.** On the large shape (256/2048/14), 128×128+pad=16 achieves **824.2 TFLOP/s** versus 513.7 for 64×64+pad=16 — a **60.5% advantage** for the larger tile. On the medium shape (128/1024/17), the gap is smaller but still in favor of 128×128: **416.9 vs 353.5 TFLOP/s** (+17.9%).

### Key observations

1. **128×128+pad=16 dominates on large shape.** This is a 71% improvement over 128×128+pad=0 (480.8→824.2 TFLOP/s), making it by far the best configuration overall. The large tile benefits significantly from reduced bank conflicts when the shape has enough data to sustain the higher occupancy and memory bandwidth.

2. **64×64+pad=16 provides modest improvement.** Compared to 64×64+pad=0, the padding gives +8.4% on the medium shape and +9.3% on the large shape. This is far less dramatic than the 71% gain seen with 128×128.

3. **Pad=0 baseline:** On the medium shape, 128×128+pad=0 (299.4 TFLOP/s) was the worst performer. On the large shape, it improves to 480.8 TFLOP/s but still trails padded variants. Without padding, the larger tile suffers from LDS bank conflicts that reduce effective utilization.

4. **Cost profile:** On the large shape, 128×128+pad=16 achieves both the highest throughput AND the lowest latency (0.511ms vs 0.895ms for the next-best 64×64+pad=0). The padded 128×128 tile is faster and more efficient at every metric.

5. **The W-3 finding confirms this:** W-3 showed that `lds_row_pad=16` eliminates LDS bank conflicts for both tile sizes. The 128×128 tile, being 4× the compute capacity of 64×64, benefits far more from having its data moved off the critical path of bank conflict resolution, allowing it to sustain much higher utilization.

### Relative performance matrix (large shape)

|              | pad=0  | pad=16 |
|--------------|--------|--------|
| **128×128**  | 480.8  | **824.2** (+71%) |
| **64×64**    | 470.0  | 513.7 (+9.3%) |

The pad=16 benefit is ~7× larger for 128×128 than for 64×64.

### Relative performance matrix (medium shape)

|              | pad=0  | pad=16 |
|--------------|--------|--------|
| **128×128**  | 299.4  | 416.9 (+39%) |
| **64×64**    | 329.9  | 353.5 (+7.2%) |

Same pattern: pad=16 gives 128×128 a large uplift (39%) while 64×64 gets only a modest one (7%).

## Recommendation

**Use 128×128 with `lds_row_pad=16` as the primary tile for WRW on gfx1250.** It delivers:

- **71% higher throughput** than the unpadded 128×128 variant
- **60% higher throughput** than 64×64 with padding on large shapes
- **39% higher throughput** than unpadded 128×128 on medium shapes
- **Lowest latency** across all tested configurations

The 64×64 tile remains useful as a fallback for shapes that are not multiples of 128 or for very small workloads where the larger tile cannot fully occupy the GPU's 256 CUs. However, for the common large/batch convolution workloads that dominate inference and training compute, 128×128+pad=16 is the clear winner.

The optimal tile ordering strategy for WRW on gfx1250 should be:
1. **128×128 + lds_row_pad=16** — primary tile for most shapes
2. **64×64 + lds_row_pad=16** — secondary tile for small/narrow dimensions
3. Avoid unpadded configurations on gfx1250; bank conflicts consistently degrade performance.
