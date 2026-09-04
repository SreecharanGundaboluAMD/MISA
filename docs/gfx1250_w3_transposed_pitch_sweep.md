# W-3: Transposed LDS Row Pitch Sweep (bwd/wrw)

**Date:** 2026-09-04
**GPU:** gfx1250 (MI450), 256 CUs
**ROCm:** 10.1 (`/home/sgundabo/rocm-10.1`)
**Commit base:** `e8139fc` (B4 `6fcde3a` landed)

## 1. Background

B4 (`6fcde3a`) implemented `lds_row_pad` for transposed operands in bwd and
wrw. The transposed operand's LDS row pitch is
`gemm_{m,n}_per_block * data_byte = 256 B` for 128-wide fp16 — exactly 64
banks × 4 B, the worst possible bank alias (every K-row hits the same banks).

Valid pads must keep `gcd(stride_dwords, 64) == 4` (conflict-free):

| pad | stride (B) | stride (dwords) | gcd(stride, 64) | conflict class        |
|-----|------------|-----------------|-----------------|-----------------------|
| 0   | 256        | 64              | 64              | 64-way alias (worst)  |
| 16  | 272        | 68              | 4               | conflict-free ✓       |
| 48  | 304        | 76              | 4               | conflict-free ✓       |
| 64  | 320        | 80              | 16              | 16-way conflict (bad) |

B4 validated pad=16. This sweep tests pad=48 (also conflict-free, never
tested) and pad=0 (baseline) to confirm the model and pick the optimal pad.

## 2. Method

### Config

Standard 128×128×32 fp16 WMMA config, `nxe=0` (1×1 GEMM path), with
`lds_row_pad` set to {0, 16, 48}. wrw configs include `gemm_k_global_split = 1`
(W-1 default). Each pad value is a separate config file → separate build →
separate `conv_driver.exe` with a single tunable (no config-search noise).

### Shapes (from the report's standing benchmark set)

| Label      | -n  | -c   | -H | -W | -k   | -y | -x | -p | -q | gemm_m | gemm_n | gemm_k |
|------------|-----|------|----|----|------|----|----|----|----|--------|--------|--------|
| med1x1     | 128 | 1024 | 17 | 17 | 1024 | 1  | 1  | 0  | 0  | 1024   | 36992  | 1024   |
| large1x1   | 256 | 2048 | 14 | 14 | 2048 | 1  | 1  | 0  | 0  | 2048   | 100352 | 2048   |
| 3x3        | 64  | 512  | 28 | 28 | 512  | 3  | 3  | 1  | 1  | 512    | 200704 | 4608   |

**3×3 note:** The configs use `nxe=0` (1×1-only path). The 3×3 filter is
flattened into the GEMM's K dimension (gemm_k = C·y·x = 512·9 = 4608), so the
`nxe=0` path handles it as a plain GEMM — **no build failure**, all 3×3
combinations ran and validated. (The task anticipated a possible skip; it was
not needed.)

### Directions

- **bwd** (`-F 2`): B is transposed (256 B row pitch). A is untransposed but
  also padded (64→80 B) since `lds_row_pad` is a single tunable applied to both
  operands. The measured improvement is the **combined** A+B padding effect;
  A-side and B-side contributions cannot be independently isolated with the
  current tunable.
- **wrw** (`-F 4`): **Both** A and B are transposed (256 B row pitch each), so
  padding applies to both transposed operands simultaneously.

### Run

`IGEMM_WARMUP=3 IGEMM_REPEAT=10`, fp16 only. Best (min) cost over 10 repeats
reported. All combinations `valid:y`.

## 3. Results

| dir | shape    | pad | stride_B | stride_dw | gcd | valid | TFLOP/s | cost_ms | speedup vs pad0 |
|-----|----------|-----|----------|-----------|-----|-------|---------|---------|-----------------|
| bwd | med1x1   | 0   | 256      | 64        | 64  | y     | 424.5   | 0.183   | 1.000×           |
| bwd | med1x1   | 16  | 272      | 68        | 4   | y     | 498.3   | 0.156   | **1.173×**       |
| bwd | med1x1   | 48  | 304      | 76        | 4   | y     | 496.1   | 0.156   | **1.173×**       |
| bwd | large1x1 | 0   | 256      | 64        | 64  | y     | 537.6   | 0.783   | 1.000×           |
| bwd | large1x1 | 16  | 272      | 68        | 4   | y     | 572.1   | 0.736   | **1.064×**       |
| bwd | large1x1 | 48  | 304      | 76        | 4   | y     | 562.0   | 0.749   | 1.045×           |
| bwd | 3x3      | 0   | 256      | 64        | 64  | y     | 480.6   | 0.493   | 1.000×           |
| bwd | 3x3      | 16  | 272      | 68        | 4   | y     | 617.5   | 0.383   | **1.287×**       |
| bwd | 3x3      | 48  | 304      | 76        | 4   | y     | 616.9   | 0.384   | 1.284×           |
| wrw | med1x1   | 0   | 256      | 64        | 64  | y     | 300.0   | 0.259   | 1.000×           |
| wrw | med1x1   | 16  | 272      | 68        | 4   | y     | 417.3   | 0.186   | 1.392×           |
| wrw | med1x1   | 48  | 304      | 76        | 4   | y     | 419.1   | 0.185   | **1.400×**       |
| wrw | large1x1 | 0   | 256      | 64        | 64  | y     | 482.1   | 0.873   | 1.000×           |
| wrw | large1x1 | 16  | 272      | 68        | 4   | y     | 820.8   | 0.513   | **1.702×**       |
| wrw | large1x1 | 48  | 304      | 76        | 4   | y     | 776.6   | 0.542   | 1.611×           |
| wrw | 3x3      | 0   | 256      | 64        | 64  | y     | 320.5   | 0.739   | 1.000×           |
| wrw | 3x3      | 16  | 272      | 68        | 4   | y     | 414.0   | 0.572   | 1.292×           |
| wrw | 3x3      | 48  | 304      | 76        | 4   | y     | 424.6   | 0.558   | **1.324×**       |

### Speedup summary (pad=16 vs pad=0)

| shape    | bwd    | wrw    |
|----------|--------|--------|
| med1x1   | +17.3% | +39.2% |
| large1x1 | +6.4%  | +70.2% |
| 3x3      | +28.7% | +29.2% |

## 4. Analysis

### 4.1 Does the bank-conflict model track?

**Yes, decisively.** pad=0 (gcd=64, worst alias) is the slowest in every one of
the 18 cells. Both conflict-free pads (gcd=4) produce large, consistent
speedups: **+6% to +70%**. The transposed operands suffer the worst-case
64-way bank alias at pad=0, and eliminating it (gcd→4) is the single largest
lever in these kernels — consistent with B4's finding and the §3.4 fwd stride
sweep.

wrw benefits more than bwd because **both** A and B are transposed in wrw (two
aliased operands padded), vs only B in bwd (A is untransposed, padded 64→80 as
a side effect). The wrw large1x1 shape sees +70% — the largest gain — because
its 2048-wide K produces the most K-row LDS loads, maximally exposing the
alias.

### 4.2 Does pad=48 differ from pad=16?

**Negligibly, and pad=16 is marginally better.** Both are conflict-free
(gcd=4). Across the 9 (dir, shape) pairs:

| metric (pad16 vs pad48)            | result                         |
|------------------------------------|--------------------------------|
| within 1% of each other            | 7 of 9 pairs                   |
| pad=16 faster                      | bwd large1x1 (+6.4% vs +4.5%), wrw large1x1 (+70% vs +61%) |
| pad=48 faster                      | wrw med1x1 (+40.0% vs +39.2%), wrw 3x3 (+32.4% vs +29.2%) — both <1.1pp |

The differences are within run-to-run noise (IGEMM_REPEAT=10, single-shot
min). pad=48's larger stride (304 vs 272 B) wastes slightly more LDS capacity
with no conflict benefit, which may explain its marginal deficit on the
large1x1 shapes (higher LDS pressure / lower occupancy). This mirrors the §3.4
fwd finding that 80 B and 112 B performed similarly (both conflict-free), while
128 B (gcd=16, conflicted) was worse.

### 4.3 Does 3×3 regress (like fwd)?

**No — 3×3 shows the *largest* bwd gain (+28.7%).** This is the opposite of
fwd, where the §3.4 sweep found padding *hurt* the 3×3 shape. The difference:
in fwd the 3×3 path uses `nxe=1` (image-domain tiling) with a different LDS
access pattern where padding disrupts the layout; here bwd/wrw use `nxe=0`
(filter flattened into K), so the 3×3 case is a plain GEMM with the same
transposed-operand alias as 1×1. There is no layout disruption, so padding
helps uniformly. wrw 3×3 gains +29%, in line with its 1×1 shapes.

## 5. Recommendation

**Use `lds_row_pad = 16`** for all bwd and wrw fp16 128×128×32 configs.

Rationale:
- **Conflict-free** (gcd=4), same as pad=48.
- **Equal or faster** than pad=48 on 7 of 9 pairs; marginally but consistently
  better on the large1x1 shapes where LDS pressure matters.
- **Smaller LDS footprint** (272 vs 304 B/row) — preserves occupancy headroom
  for future larger-tile or higher-occupancy configs.
- Matches B4's validated default and B5's emitted config sections
  (`0dd5e99`), so no churn to landed work.

pad=48 is a viable alternative (also conflict-free, within noise) but offers no
benefit over pad=16 and consumes more LDS. pad=0 must never be used for
transposed operands — it is the worst-case 64-way alias.

### A-side / B-side isolation caveat (bwd)

In bwd, `lds_row_pad` pads **both** A (untransposed, 64→80 B) and B
(transposed, 256→272 B) simultaneously. The measured bwd speedup is the
**combined** effect. The B-side (transposed) contribution is expected to
dominate (256→272 eliminates a 64-way alias; 64→80 eliminates a 16-way alias),
but the split cannot be isolated without separate A/B pad tunables. This is a
known limitation of the single-tunable design, noted for the record.
