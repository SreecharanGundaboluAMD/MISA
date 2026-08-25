# gfx1250 WMMA vs. MIOpen: vendor-trace benchmark comparison (2026-08-25)

Benchmarked MISA's gfx1250 WMMA kernels against
`/home/sgundabo/vendor-benchmarking/tracelens_shapes_gfx1250.json`, a MIOpenDriver trace
recording the solver MIOpen chose and its measured time for each shape. Goal: for shapes MISA
can actually support, compare MISA's fastest tile variant against MIOpen's chosen solver/time.

**Status: results below are PROVISIONAL.** See the contention caveat near the end before
treating any specific multiple as accurate — re-benchmark on an exclusively-held GPU before
relying on these numbers.

## Shape triage

The trace has 112 total entries:

- **26 batch norm** entries — not a MISA op (igemm only does convolution), skipped.
- **86 conv entries**, of which:
  - **14 are depthwise** (`g == c`) — architecturally out of scope for MISA's igemm approach
    (a degenerate 1-wide GEMM per group has no efficient tile size; see
    `docs/gfx1250_wmma_layout.md`'s "What's NOT done" notes on this). Skipped.
  - Of the remaining **72 non-depthwise conv entries**, only **20** have GEMM_M/GEMM_N/GEMM_K
    dimensions that fit MISA's currently-available gfx1250 WMMA tile shapes: GEMM_M and GEMM_N
    both need to land on a multiple of 64 or 128 (128x128 or 64x64 tiles — the asymmetric
    128x64/64x128 shapes landed this session cover power-of-2 ratios, not arbitrary sizes), and
    GEMM_K must be an exact multiple of 32 (fp16/bf16 `inst_wmma.k` — the WMMA main loop has no
    K-boundary/padding handling, so a non-exact GEMM_K has no valid MISA config at all). The
    other 52 entries fail one of these two constraints and simply aren't buildable today.

All 20 buildable shapes are bf16/NHWC, batch size 42.

## Results

Built each direction's combined bf16 config (already has both 128x128 and 64x64 tile sections
in one file — built via `python3 igemm_codegen.py config/igemm_<dir>_gtc_gfx1250_nhwc_bf16.config`,
**not** `-s`, see the split-kernel macro-loss note in this doc's companion
`gfx1250_wmma_layout.md`). Ran `conv_driver.exe` in its default `driver_mode_normal`, which
searches every tunable present in the built module and reports the fastest — the same kind of
search MIOpen itself does over its solver database. `IGEMM_WARMUP=5 IGEMM_REPEAT=20` for each
shape, args taken directly from the trace's MIOpenDriver command line (`-F 1`=fwd, `-F 2`=bwd,
`-F 4`=wrw; bf16 mode string is `convbfp16`, matching MIOpenDriver's own naming).

| Direction | Shapes tested | MISA vs. MIOpen |
|---|---|---|
| fwd | 5 | 1.4x-1.9x slower — reasonable for a newer, less-tuned kernel path |
| bwd | 5 | 1.3x-3.2x slower — same story |
| wrw | 10 | **100x-1665x SLOWER** — a real problem, not noise |

Two representative wrw data points: `c=128,H=30,W=40,k=128,1x1` took 8.0ms on MISA vs 0.035ms on
MIOpen (227x). The worst, `c=128,H=120,W=160,k=128,3x3,pad=1`, took **1.18 seconds** on MISA vs
0.71ms on MIOpen (1665x).

## Root cause of the wrw gap (architectural, confirmed independent of any single measurement)

wrw's GEMM_M (=K_out) and GEMM_N (=C_in*Y*X) are both small for these shapes (often exactly 128,
up to 512), while GEMM_K (=N*Ho*Wo, batch*spatial) is huge (10,000s-800,000s). With only a
128x128 or 64x64 tile choice and **no `gemm_k_global_split` support anywhere in the WMMA path**
(that K-splitting mechanism exists for the XDLOPS track but was never ported to WMMA — this
whole gfx1250 WMMA initiative has so far been correctness, tile-shape, and main-loop-focused,
not K-split), the grid size for several of these shapes is **exactly one workgroup**
(gemm_m=128, gemm_n=128 -> one 128x128 tile) on a **256 compute-unit** GPU (confirmed via
`rocminfo`). That single workgroup then serially executes 1500+ main-loop iterations
(GEMM_K / 32) with no other concurrent workgroup to hide global-memory/barrier latency behind —
every iteration's stall is fully exposed rather than overlapped, consistent with the observed
~5us/iteration x ~1500 iterations ~= 8ms.

fwd/bwd don't hit this because their GEMM_M (=N*Ho*Wo, spatial) is naturally large
(100,000s+), giving thousands of workgroups regardless of tile choice — full occupancy, hence
only the "normal" 1.3-3x gap vs. mature XDLOPS kernels rather than a 100-1000x occupancy cliff.

**Practical implication**: MISA's gfx1250 WMMA wrw kernels are only viable today for problems
whose GEMM_M x GEMM_N tile count alone yields enough workgroups to fill 256 CUs — large
K_out/C_in — not the small-K_out/C_in-huge-batch-spatial shapes common in real backbones (which
is most of what this trace file contains). Porting `gemm_k_global_split` (or an equivalent
atomic-reduction K-split) to the WMMA wrw path is likely the single highest-leverage next step
for wrw performance on gfx1250 — bigger impact than any tile-shape or main-loop work landed so
far in this initiative.

## Contention caveat — GPU was shared during (at least part of) this measurement

Immediately after these results were first reported, it was flagged that other users might be
running concurrently on this GPU. Checked right away and **confirmed**: `rocm-smi --showuse
--showpids` showed `GPU use (%): 100` with a `python3` KFD process consuming VRAM, whose PID
changed between two consecutive checks seconds apart (2664415 -> 2664515) and was not visible
via `ps` in this session at all — i.e. a different user/namespace actively driving the GPU, not
anything from this session. This check happened only *after* the benchmark run above, not
during it, so it's unknown whether the same contention was present while the fwd/bwd/wrw
numbers were actually collected.

**Treat every number above as provisional, not final** — especially the extreme wrw slowdowns.
Contention could inflate MISA's measured times more than MIOpen's (MISA's already-tiny
1-workgroup wrw case has zero spare occupancy to absorb competing work; MIOpen's numbers came
from an earlier, separate trace run this session didn't collect and can't correct for). The
architectural explanation above (no `gemm_k_global_split`, 1-workgroup grids on a 256-CU part)
is sound independent of contention and likely still explains the bulk of the gap, but the exact
multiples (227x, 1665x, etc.) should be re-measured on an otherwise-idle GPU before being
treated as precise.

**Before any future re-benchmark**: run `rocm-smi --showuse --showpids` (and re-check a few
seconds later — a PID that changes between checks is itself a contention signal) and confirm
`GPU use (%)` is near 0 with no unfamiliar KFD process, before starting, not just before
concluding.

## How to reproduce

1. Build the direction's combined bf16 config: `python3 igemm_codegen.py
   config/igemm_<dir>_gtc_gfx1250_nhwc_bf16.config` (no `-s`/`--split_kernel` — that flag drops
   gfx1250's kernel-specific division macros and breaks assembly, see
   `docs/gfx1250_wmma_layout.md`'s Phase 11 "process discovery" note).
2. Confirm the GPU is idle (see the contention caveat above).
3. From `out/`, run e.g.:
   ```
   IGEMM_WARMUP=5 IGEMM_REPEAT=20 ./conv_driver.exe convbfp16 -n 42 -c <C> -H <H> -W <W> \
     -k <K> -y <Y> -x <X> -p <P> -q <Q> -u <U> -v <V> -l 1 -j 1 -g <G> -F <1|2|4> -t 1 \
     --in_layout NHWC --fil_layout NHWC --out_layout NHWC -V 0
   ```
   using the exact args from the trace's MIOpenDriver command line for that shape.
4. `driver_mode_normal` (the default) searches every tunable/tile-shape present in the built
   module and reports the fastest per direction — read off the lowest `cost:` line.
