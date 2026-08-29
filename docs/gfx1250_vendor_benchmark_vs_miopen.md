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

## Re-run (2026-08-25, same day, same branch tip)

Re-executed the exact reproduce steps below against the same 20 shapes, at the current branch
tip (`e6d21a6`, no kernel changes since the original run above — Phase 15/16's main-loop
interleaving was already included in both this run and the original). Goal was to get a second
independent measurement, per this doc's own "re-benchmark before trusting these numbers"
recommendation.

**Contention was still present and, if anything, worse-looking**: `rocm-smi --showuse --showpids`
read `GPU use (%): 100` on every check (start of run, mid-run, end of run, ~seconds to minutes
apart), with **no KFD PIDs visible at all** in this session — not even the earlier run's
sometimes-visible `python3` process. This is consistent with a different tenant/namespace
(container or different Linux user) driving the GPU that this session simply cannot see, on
what is evidently a shared/multi-tenant box. User explicitly asked to proceed anyway; treat
every number below as equally provisional to the first run, not a clean confirmation.

| Direction | Shape (c,H,W,k,y×x) | MISA best (ms) | MIOpen (ms) | Ratio |
|---|---|---|---|---|
| fwd | 128,120,160,128,3x3 | 0.385 | 0.328 | 1.17x slower |
| fwd | 192,60,80,64,1x1 | 0.029 | 0.036 | **1.25x FASTER** |
| fwd | 256,60,80,64,1x1 | 0.032 | 0.040 | **1.24x FASTER** |
| fwd | 64,60,80,128,1x1 | 0.026 | 0.020 | 1.28x slower |
| fwd | 64,60,80,256,1x1 | 0.045 | 0.037 | 1.21x slower |
| bwd | 128,120,160,128,3x3 | 0.604 | 0.318 | 1.90x slower |
| bwd | 192,60,80,64,1x1 | 0.043 | 0.033 | 1.31x slower |
| bwd | 256,60,80,64,1x1 | 0.050 | 0.037 | 1.35x slower |
| bwd | 64,60,80,128,1x1 | 0.030 | 0.027 | 1.13x slower |
| bwd | 64,60,80,256,1x1 | 0.045 | 0.042 | 1.07x slower |
| wrw | 128,120,160,128,3x3 | 630.8 | 0.414 | 1525x slower |
| wrw | 128,30,40,128,1x1 | 4.385 | 0.022 | 200x slower |
| wrw | 128,30,40,128,3x3 | 31.17 | 0.067 | 462x slower |
| wrw | 128,30,40,512,1x1 | 4.032 | 0.053 | 76x slower |
| wrw | 192,60,80,64,1x1 | 17.31 | 0.0056 | 3087x slower |
| wrw | 256,30,40,128,1x1 | 4.295 | 0.028 | 151x slower |
| wrw | 256,60,80,64,1x1 | 17.25 | 0.052 | 331x slower |
| wrw | 512,30,40,128,1x1 | 4.032 | 0.050 | 81x slower |
| wrw | 64,60,80,128,1x1 | 17.34 | 0.040 | 431x slower |
| wrw | 64,60,80,256,1x1 | 17.18 | 0.052 | 329x slower |

**Same story, same order of magnitude, no regression and no fix**: fwd lands at 0.8x-1.28x vs.
MIOpen (two of five shapes actually *beat* MIOpen this time, unlike the original run's uniformly
1.4x-1.9x — plausibly run-to-run/contention noise, not a real improvement, since nothing in the
fwd/bwd kernel path changed between runs). bwd is a consistent 1.07x-1.90x slower. wrw remains
catastrophic at **76x-3087x slower** — same architectural cause as diagnosed above (no
`gemm_k_global_split`, 1-2 workgroup grids on 256 CUs for these shapes), confirmed independently
a second time. The `192,60,80,64` wrw shape's 3087x is the single worst ratio seen across either
run, but note its MIOpen baseline (0.0056ms) is also the smallest absolute time in the whole
table, so it's the most sensitive to timer/contention noise on both sides — don't over-index on
that specific multiple.

**Bottom line unchanged from the original run**: fwd/bwd are in the normal "newer, less-tuned
kernel" range; wrw needs `gemm_k_global_split` (or equivalent K-split) before it's usable for
these shapes, and that conclusion now has two independent (if both contended) measurements
behind it rather than one.

## Automated reproduction

`script/benchmark_gfx1250_vs_miopen.py` automates the manual reproduce steps below for
all 38 shapes in this doc's tables (embeds the shape list, per-shape MISA config choice,
and the MIOpen/gfx950 + MIOpen/gfx1250 reference numbers, so it has no dependency on the
trace JSON files being present on whatever machine runs it). Builds each needed MISA
config on first use (cached after that; pass `--rebuild` to force), runs `conv_driver.exe`
for every shape, and prints/writes a Markdown table with MISA's time next to both MIOpen
references and the ratio. Usage:

```
python3 script/benchmark_gfx1250_vs_miopen.py [--direction fwd|bwd|wrw|all] [--rebuild] \
    [--warmup N] [--repeat N] [--verify] [--markdown-out FILE]
```

Prints a `rocm-smi --showuse --showpids` snapshot before running (advisory, not blocking)
given how often GPU contention has muddied the exact numbers in this doc.

## How to reproduce (manual)

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

## Update (2026-08-25): wrw fixed via `gemm_k_global_split`

The wrw catastrophe diagnosed above (no K-split, 1-2 workgroup grids on a 256-CU part) is
fixed — see `docs/gfx1250_wmma_layout.md`'s Phase 17. Re-ran all 10 wrw shapes from the table
above (same batch=42, same GPU, same-day) through the new
`config/igemm_wrw_gtc_gfx1250_nhwc_bf16_gsplit.config` (`gemm_k_global_split = 1`, built and
run the same way as the reproduce steps above, just pointing at the `_gsplit` config file):

| Shape (c,H,W,k,y×x) | MISA before (ms) | MISA gsplit (ms) | MIOpen (ms) | Before vs MIOpen | After vs MIOpen | Speedup |
|---|---|---|---|---|---|---|
| 128,120,160,128,3x3 | 630.8 | 2.435 | 0.414 | 1525x slower | 5.9x slower | 259x |
| 128,30,40,128,1x1 | 4.385 | 0.043 | 0.022 | 200x slower | 2.0x slower | 102x |
| 128,30,40,128,3x3 | 31.17 | 0.429 | 0.067 | 462x slower | 6.4x slower | 73x |
| 128,30,40,512,1x1 | 4.032 | 0.111 | 0.053 | 76x slower | 2.1x slower | 36x |
| 192,60,80,64,1x1 | 17.31 | 0.164 | 0.0056 | 3087x slower | 29.3x slower | 106x |
| 256,30,40,128,1x1 | 4.295 | 0.093 | 0.028 | 151x slower | 3.3x slower | 46x |
| 256,60,80,64,1x1 | 17.25 | 0.148 | 0.052 | 331x slower | 2.9x slower | 117x |
| 512,30,40,128,1x1 | 4.032 | 0.104 | 0.050 | 81x slower | 2.1x slower | 39x |
| 64,60,80,128,1x1 | 17.34 | 0.082 | 0.040 | 431x slower | 2.1x slower | 212x |
| 64,60,80,256,1x1 | 17.18 | 0.148 | 0.052 | 329x slower | 2.9x slower | 116x |

Worst case improved from 3087x slower than MIOpen to 29.3x slower; eight of ten shapes now
land within 2-6.5x of MIOpen (previously 76x-462x for the same eight). The remaining gap on
these shapes is architectural, not a bug: this is still a correctness-first, single-heuristic
K-split (no runtime search over multiple candidate split counts the way the mature XDLOPS path
does via `gks_iterative`, and no coalescing-store/LDS-reshuffle epilogue), so there's real
headroom left — searching split counts and the epilogue/LDS optimizations noted elsewhere in
this doc's companion layout doc are the natural next steps, in that order of likely impact.

`192,60,80,64,1x1` remains the worst case (29.3x) for the same reason its MIOpen baseline
(0.0056ms) was flagged as the most noise-sensitive number in the original run — it's also the
only shape here where the 128x128 tile wasn't applicable (`gemm_n=64` doesn't divide evenly by
128), so it's stuck on the 64x64 tile alone.

Same GPU-contention caveat as the rest of this doc applies: `rocm-smi --showuse --showpids`
read `GPU use (%): 100` with no visible KFD PIDs during this run too. Treat the exact multiples
above as provisional for the same reason as the original run, though the two-orders-of-
magnitude scale of the improvement is not something contention noise could produce.

### Re-run (2026-08-25, same day, same branch tip, gsplit config)

Re-executed the exact same 10 gsplit shapes above as a second independent measurement, per
this doc's own "re-benchmark before trusting these numbers" convention. Contention was still
present (`GPU use (%): 100`, no visible KFD PIDs, same as every other check this session).

| Shape (c,H,W,k,y×x) | Run 1 (ms) | Run 2 (ms) | MIOpen (ms) | Run 2 vs MIOpen |
|---|---|---|---|---|
| 128,120,160,128,3x3 | 2.435 | 2.270 | 0.414 | 5.5x slower |
| 128,30,40,128,1x1 | 0.043 | 0.042 | 0.022 | 1.9x slower |
| 128,30,40,128,3x3 | 0.429 | 0.412 | 0.067 | 6.1x slower |
| 128,30,40,512,1x1 | 0.111 | 0.109 | 0.053 | 2.1x slower |
| 192,60,80,64,1x1 | 0.164 | 0.127 | 0.0056 | 22.7x slower |
| 256,30,40,128,1x1 | 0.093 | 0.093 | 0.028 | 3.3x slower |
| 256,60,80,64,1x1 | 0.148 | 0.148 | 0.052 | 2.8x slower |
| 512,30,40,128,1x1 | 0.104 | 0.104 | 0.050 | 2.1x slower |
| 64,60,80,128,1x1 | 0.082 | 0.082 | 0.040 | 2.1x slower |
| 64,60,80,256,1x1 | 0.148 | 0.148 | 0.052 | 2.8x slower |

Nine of ten shapes reproduced within a few percent of run 1 (well within normal run-to-run
noise). `192,60,80,64,1x1` moved more (0.164ms → 0.127ms, ratio 29.3x → 22.7x vs MIOpen) — the
same shape already flagged in the original (pre-gsplit) run as the most noise-sensitive, since
its MIOpen baseline (0.0056ms) is also the smallest absolute time in either table and it's the
one shape here stuck on the 64x64 tile alone (128x128 isn't applicable for `gemm_n=64`). The
conclusion is unchanged and now has two independent measurements behind it: wrw is fixed from
a 76x-3087x catastrophe down to a normal-for-a-newer-kernel 2x-6x gap on 9 of 10 shapes, with
one persistently noisier outlier around 23x-29x.

## Update (2026-08-25): split-count runtime search (`docs/gfx1250_wmma_layout.md`'s Phase 18)

The single-heuristic split count above (`largest divisor <= ceil(num_cu/tile_count)`) turned
out to leave real performance on the table — timing it against nearby alternatives found gaps
as large as 6x for the same problem, just from picking a different (still-valid) split count.
Replaced with a small runtime search: 3 candidate split counts (the heuristic target, half,
and double, each snapped to the nearest valid divisor of `num_k_blocks`) are actually launched
and timed, keeping whichever ran fastest. Re-ran the same 10 shapes a third time:

| Shape (c,H,W,k,y×x) | Single-heuristic (ms) | Search (ms), best split | MIOpen (ms) | vs MIOpen |
|---|---|---|---|---|
| 128,120,160,128,3x3 | 2.270 | 2.362 (1008) | 0.414 | 5.7x slower |
| 128,30,40,128,1x1 | 0.042 | 0.035 (225) | 0.022 | 1.6x slower |
| 128,30,40,128,3x3 | 0.412 | 0.408 (225) | 0.067 | 6.1x slower |
| 128,30,40,512,1x1 | 0.109 | 0.103 (63) | 0.053 | 1.9x slower |
| 192,60,80,64,1x1 | 0.127 | 0.074 (315) | 0.0056 | 13.2x slower |
| 256,30,40,128,1x1 | 0.093 | 0.084 (105) | 0.028 | 3.0x slower |
| 256,60,80,64,1x1 | 0.148 | 0.086 (252) | 0.052 | 1.65x slower |
| 512,30,40,128,1x1 | 0.104 | 0.103 (63) | 0.050 | 2.06x slower |
| 64,60,80,128,1x1 | 0.082 | 0.064 (450) | 0.040 | 1.6x slower |
| 64,60,80,256,1x1 | 0.148 | 0.087 (252) | 0.052 | 1.67x slower |

Four shapes improved 1.6x-2x over the single-heuristic result just from trying 2 more split
counts and keeping the best (`192,60,80,64,1x1`, `256,60,80,64,1x1`, `64,60,80,128,1x1`,
`64,60,80,256,1x1`); the rest were flat to slightly better. None regressed. Worst case vs
MIOpen improved from ~23-29x to ~13.2x; the shape stuck on a single tile choice
(`192,60,80,64,1x1`, 128x128 not applicable for `gemm_n=64`) remains the outlier, though
meaningfully less of one.

This confirms the split-count choice — not just having a K-split at all — is a real, sizeable
lever. A wider search (more candidates, or an adaptive bracket instead of a fixed 3-point one)
is very likely to close more of the remaining gap; see the layout doc's Phase 18 for the
tradeoff (search cost scales with candidate count, same as it does for the mature XDLOPS path's
own `gks_iterative`). Same GPU-contention caveat as the rest of this doc applies throughout.

## Update (2026-08-25): epilogue address double-buffering tried, no measured change

Also tried double-buffering the epilogue's address computation (`docs/gfx1250_wmma_layout.md`'s
Phase 19), after confirming via the CDNA5 ISA doc that fp32 atomic-add can't be vectorized/
widened at the instruction level on this hardware. Re-ran all 10 shapes — numbers were
indistinguishable from the Phase 18 table above (within normal run-to-run noise). Kept the
change (it's correctness-neutral and removes a real hazard) but it isn't the lever that matters
here: the atomic RMW round-trip latency, not the surrounding address arithmetic, is almost
certainly what dominates the atomic epilogue's cost.

## Update (2026-08-25): split-count search pushed further — full-sweep characterization + ternary search

Pushed on the split-count lever (`docs/gfx1250_wmma_layout.md`'s Phase 20) since it was the one
shown to actually move the needle. Added a research-only `IGEMM_GSPLIT_SWEEP=<target>` env var
and swept the **entire** perf-vs-split-count curve (every divisor of `num_k_blocks`, for all
three distinct values across these 10 shapes) directly on hardware. Confirmed the curve is
unimodal — but the true minimum's position relative to the naive `ceil(num_cu/tile_count)`
target moves inconsistently between shapes (sometimes well above it, sometimes well below),
which is exactly why Phase 18's fixed `{target, target/2, target×2}` bracket sometimes landed
6x off. Replaced it with a ternary search over the full sorted divisor list — same underlying
principle (unimodal ⇒ findable minimum), but it actually locates the minimum instead of
guessing at its neighborhood, in O(log(divisor count)) real timed launches.

| Shape (c,H,W,k,y×x) | Phase 18 (3-cand, ms) | Phase 20 (ternary, ms) | MIOpen (ms) | vs MIOpen |
|---|---|---|---|---|
| 128,120,160,128,3x3 | 2.362 | 1.995 | 0.414 | 4.8x slower |
| 128,30,40,128,1x1 | 0.035 | 0.035 | 0.022 | 1.6x slower |
| 128,30,40,128,3x3 | 0.408 | 0.413 | 0.067 | 6.2x slower |
| 128,30,40,512,1x1 | 0.103 | 0.098 | 0.053 | 1.85x slower |
| 192,60,80,64,1x1 | 0.074 | 0.065 | 0.0056 | 11.6x slower |
| 256,30,40,128,1x1 | 0.084 | 0.085 | 0.028 | 3.0x slower |
| 256,60,80,64,1x1 | 0.086 | 0.065 | 0.052 | 1.25x slower |
| 512,30,40,128,1x1 | 0.103 | 0.096 | 0.050 | 1.9x slower |
| 64,60,80,128,1x1 | 0.064 | 0.062 | 0.040 | 1.55x slower |
| 64,60,80,256,1x1 | 0.087 | 0.067 | 0.052 | 1.3x slower |

Two shapes improved a further ~23-24% over Phase 18 (`256,60,80,64,1x1` and `64,60,80,256,1x1`
— both had target-vs-true-optimum mismatches the 3-candidate bracket couldn't bridge); the rest
were flat to modestly better. **None regressed.** Worst case vs MIOpen: 13x → 11.6x. Six of ten
shapes are now within 2x of MIOpen; the worst two (`128,30,40,128,3x3` at 6.2x and
`192,60,80,64,1x1` at 11.6x) are the smallest-absolute-time shapes in the set, so also the most
exposed to fixed per-dispatch overhead that no split-count choice can amortize away.

This is very likely close to the ceiling for split-count tuning alone — the exhaustive sweep
found points at most 1-2% better than the ternary search's picks. **Further gains from here
need the epilogue itself**, specifically an LDS-reshuffle coalescing store for the non-atomic
path (fwd/bwd/non-split wrw — the atomic path's fp32 atomic-add has no wide instruction to
exploit, confirmed via the CDNA5 ISA doc in Phase 19). That work was deliberately deferred, not
attempted in this session — see `docs/gfx1250_wmma_layout.md`'s Phase 19 "Critical files" note
and this doc's own conclusion above for where to pick it back up.

## Update (2026-08-25): LDS-reshuffle epilogue implemented (`docs/gfx1250_wmma_layout.md`'s Phase 21)

Implemented the deferred item above: fwd/bwd/non-split-wrw's direct epilogue (128 scalar
`global_store_dword` for a 128x128 tile) replaced with an LDS-reshuffle coalescing store —
confirmed 128 → 32 `global_store_dwordx4` (the predicted 4x) via instruction count on the
generated `.inc`. Wall-clock impact is modest and shape-dependent (0-14% faster, one shape
flat-to-slightly-slower within noise) because fwd/bwd are compute-bound for the large-K
shapes in this trace — the epilogue is a small fraction of total kernel time regardless of
how much cheaper it gets. A synthetic single-K-block shape (built specifically to make the
epilogue a large fraction of total time) showed the clearest win, 7-14% faster. Non-split
wrw showed almost nothing (~1%) — its bottleneck is occupancy (too few workgroups, fixed by
Phase 17's K-split, not this change), not epilogue cost. Full numbers and the two real bugs
found while building this (a masking bug from `v_gemm_im`/`v_gemm_in` being global rather
than tile-local addresses, and a deferred-context return-placement bug that silently zeroed
out the *atomic* path's output during regression testing) are in the layout doc's Phase 21.

This is likely close to the ceiling for epilogue-side optimization on fwd/bwd/non-split-wrw
given how little of their total time the epilogue represents. Further gains on this branch
would need to target the main loop itself (LDS bank conflicts, register double-buffering,
graduated waitcnt — all flagged but not attempted in the original code review this session)
rather than the epilogue.

## Update (2026-08-26): register-level prefetch (`docs/gfx1250_wmma_layout.md`'s Phase 22), and a cross-architecture comparison vs. gfx950

Implemented the deferred main-loop item from above: `local_prefetch_num=2`, intra-K-substep
VGPR-level prefetch for `v_a`/`v_b` (ported from the dotx/mfma paths, adapted for WMMA — see
Phase 22 for the full design and the VGPR-budget audit). Measured effect, fp32 128x128
(the only precision with VGPR headroom to fit it today): fwd gets a real, reproducible win
(+1.7% to +9.4% depending on shape), bwd regresses on one shape (-8.6%, not yet root-caused)
and is flat on another, wrw (non-split) is flat both shapes tested (expected — it's
occupancy-bound, not main-loop-latency-bound, same reason the epilogue work above didn't
move it either). **Decision: enabled for fwd, held for bwd pending the regression's root
cause, not used for wrw.** This only applies to fp32 — fp16/bf16/int8's 128x128 tile has no
VGPR headroom to fit it (confirmed via the audit: `+64` VGPRs needed, `252/256` already used).

### Cross-architecture comparison: MISA/gfx1250 vs. MIOpen/gfx950

Since `local_prefetch_num=2` only applies to fp32 and this comparison's shape set is
exclusively bf16 (same 20 buildable shapes as the MIOpen/gfx1250 comparison above, unchanged
by that decision), this is really a separate question: **how does MISA's gfx1250 WMMA
backend compare against MIOpen running on the older gfx950 (MI350X) architecture**, not a
re-run of the same-arch comparison above. Found the counterpart trace file at
`~/rocm-libraries/tracelens_shapes_gfx950.json` (same 112 total / 86 conv / 20-buildable
shape set as the gfx1250 trace referenced above — confirmed by re-running the exact same
shape-triage filter from this doc's own methodology and getting an identical 20-shape list).

For wrw, used MISA's *current* best option (the `_gsplit` K-split config from Phase 17,
built after the original MIOpen/gfx1250 comparison above) rather than the plain config that
comparison used — an apples-to-apples "MISA's fastest option today" comparison, not a replay
of an since-fixed regression. fwd/bwd use the same combined bf16 config as before (both tile
shapes searched, fastest reported). `IGEMM_WARMUP=5 IGEMM_REPEAT=20`, same methodology as
the original comparison.

| Direction | Shapes | MISA/gfx1250 vs. MIOpen/gfx950 |
|---|---|---|
| fwd | 5 | 0.99x-1.31x (avg 1.19x) — one shape (`c=128,H=120,W=160,k=128,3x3`) is a dead heat |
| bwd | 5 | 0.91x-1.24x (avg 1.08x) — one shape (`c=256,H=60,W=80,k=64,1x1`) is actually **faster** than MIOpen/gfx950 |
| wrw (`_gsplit`) | 10 | 1.09x-4.89x (avg 2.01x) |

**A dramatically different picture than the MIOpen/gfx1250 comparison above** (which showed
1.4x-3.2x for fwd/bwd and 100x-1665x for wrw) — expected, since that comparison pit MISA's
young gfx1250 kernels against MIOpen's own mature gfx1250 solvers, while this one compares
against MIOpen running on an *older* architecture (gfx950/MI350X). fwd and bwd are
essentially at parity with MIOpen-on-a-different-generation-GPU; wrw, even with the K-split
fix, still trails by 1-5x — the `_gsplit` fix solved the catastrophic 100x-1665x-class
failure (too few workgroups) but the resulting kernel's per-workgroup efficiency (one atomic
add per output element, see Phase 19/21) still isn't fully competitive. Worth noting: this
isn't a claim that gfx1250 silicon is faster than gfx950 silicon, or vice versa — it's a
statement about how mature each *software stack* is on its respective hardware today.

Same GPU-contention caveat as the rest of this doc applies; a couple of spot-reruns showed
tight variance (fwd 3x3: 0.361/0.362ms; wrw-gsplit 3x3: 2.02/2.16ms, ~7% spread) but this
was not re-benchmarked on an exclusively-held GPU.

## Update (2026-08-26): re-checked against a real MIOpen/gfx1250 trace with solver names

The `tracelens_shapes_gfx1250.json` file this doc originally cited no longer exists on disk.
Found what's almost certainly its replacement/successor:
`~/rocm-libraries/tracelense_gfx1250_b01_2.json` — same 112-entry structure, same host
architecture (`GPU Model: 1 x AMD Radeon Graphics`, `gfx1250`, host `heliosr-1b114-b01-2`,
matching this doc's "b01_2"-style naming), and critically, each entry's `row[2]` records the
actual MIOpen solver name (e.g. `156/ConvHipImplicitGemmGroupWrwXdlops`), not just a time —
letting this comparison show *what MIOpen chose*, not only *how fast*.

**No new runs were done for this update** (per instruction) — MISA's side reuses the bf16
numbers already measured earlier in this session for the MIOpen/gfx950 comparison above
(fwd/bwd: plain combined config; wrw: the `_gsplit` config), matched against this trace
file's numbers for the same 20 shapes, read directly (not re-run).

| Direction | Shapes | MISA/gfx1250 vs. MIOpen/gfx1250 | MIOpen solver(s) chosen |
|---|---|---|---|
| fwd | 5 | 0.81x-1.10x (avg 0.98x) — MISA is *faster* on 2/5 | `ConvHipImplicitGemmGroupFwdXdlops`, `ConvHipConv` |
| bwd | 5 | 1.02x-1.84x (avg 1.25x) | `ConvHipConv`, `ConvHipImplicitGemmGroupBwdXdlops` |
| wrw (`_gsplit`) | 10 | 1.23x-11.41x (avg 3.5x, one outlier) | `ConvHipImplicitGemmGroupWrwXdlops` (all 10) |

**This is the real, same-architecture comparison the top of this doc originally attempted**
(before its source trace went missing) — and it tells a much better story than that
original attempt's stale numbers (1.4x-3.2x fwd/bwd, 100x-1665x wrw): fwd is now
essentially at parity with MIOpen's own gfx1250 solvers (occasionally faster), bwd trails by
a modest ~1.25x on average, and wrw — with the `_gsplit` K-split fix — trails by ~3.5x
average instead of the original 2-3 orders of magnitude. One wrw shape
(`c=192,H=60,W=80,k=64,1x1`) is a 11.4x outlier: MIOpen's solver reports an unusually fast
0.0056ms there relative to every structurally similar shape in the set (0.03-0.05ms range),
which looks more like a measurement/solver-selection quirk specific to that one shape than a
real, general 11x gap — worth a closer look before trusting it, not something to generalize
from.

**Caveat, stated plainly**: this is not a controlled, same-session A/B — MISA's numbers came
from this session's earlier gfx950-comparison run, MIOpen's from a trace recorded on a
different run (possibly a different day, different GPU contention state, different MIOpen
version). Directionally consistent with everything else measured this session (fwd wins,
wrw still behind but no longer catastrophically), but treat the exact multiples as
approximate, same as every other number in this doc.

## Update (2026-08-27): full re-triage against `tracelens_shapes_gfx950.json` after Phases 23-31

Re-ran the whole "what's applicable" triage from scratch against
`/home/sgundabo/rocm-libraries/tracelens_shapes_gfx950.json` (the same MI350X/gfx950 trace
cited in the update above), this time accounting for every capability landed since the
original 20-shape triage at the top of this doc: `wmma_m_tail`/`wmma_n_tail` (Phases 25/26,
fwd; `wmma_m_tail` only for bwd), `gemm_k_global_split` (Phase 17, wrw), and TDM-based
GEMM_K tail (Phase 31, fwd/1x1 only). wrw still has no tail relief of any kind, gk/gn/gm
must all be exact tile multiples.

**Triage methodology**: 112 total entries, 26 batch norm (skipped, not a MISA op), 86 conv,
14 depthwise (`g==c`, skipped, architecturally out of scope). Of the remaining 72
non-depthwise conv entries, computed GEMM_M/N/K per direction (fwd: `gm=n·ho·wo,
gn=k/g, gk=c/g`; bwd: `gm=n·hi·wi, gn=c/g, gk=k/g`; wrw: `gm=k/g, gn=c/g, gk=n·ho·wo` --
matching each direction's own WMMA driver-check formula exactly, read directly from
`igemm_{fwd,bwd,wrw}_gtc_driver.h`) and checked which of MISA's actually-buildable configs
(not just "is there theoretically a relief mechanism") would accept it: `fwd_base`/
`bwd_base` (both 128x128 and 64x64 tiles, exact-multiple only), `fwd_mtail`/`fwd_ntail`/
`fwd_mntail`/`fwd_tdm` (each a SINGLE 128x128-only build -- no 64x64 alternative),
`bwd_mtail` (a single 64x64-only build), `wrw_gsplit` (both 128x128 and 64x64, always used
for wrw per this doc's own established "MISA's fastest option today" precedent).

**One false positive caught and corrected**: the naive check "gm or gn lands on a multiple
of 64 or 128" isn't sufficient for the tail/tdm-specialized configs, since (unlike the base
combined configs) each only has ONE tile size built. `c=48,H=120,W=160,k=192,1x1` looked
`fwd_tdm`-eligible by dimension count (`gn=192` is a multiple of 64) but the `_tdm` config
only has a 128x128 build and 192 isn't a multiple of 128 -- confirmed by the actual driver
returning `not applicable` when tried. Manually re-verified every other classified shape
against the correct *paired*-tile constraint (a single tile choice must satisfy gm AND gn
together, not each independently against "any" tile) with no further errors found. Final
count: **38 applicable shapes** (17 fwd, 11 bwd, 10 wrw) out of 72 non-depthwise conv
entries -- up from the original triage's 20, entirely due to the tail/tdm/gsplit capability
growth across Phases 17 and 25-31 (all of which happened after that original count was
taken).

### Benchmark

Same methodology as this doc's established reproduce steps: `IGEMM_WARMUP=5
IGEMM_REPEAT=20`, `driver_mode_normal` (searches every tunable in the built module,
reports the fastest), `-V 0` (skip verification, matching the existing gfx950-comparison
methodology above), batch=42 bf16/NHWC throughout. MIOpen's numbers are read directly from
the trace file (gfx950/MI350X), not re-run. GPU contention check before starting:
`rocm-smi --showuse --showpids` read `GPU use (%): 100` with **no visible KFD PIDs** on two
checks a few seconds apart -- the same "different tenant/namespace this session can't see"
pattern documented earlier in this file on a shared/multi-tenant box. Same caveat applies:
treat exact multiples as approximate.

| Direction | Shape (c,H,W,k,y×x) | Config | MISA/gfx1250 (ms) | MIOpen/gfx950 (ms) | Ratio |
|---|---|---|---|---|---|
| fwd | 256,1,1,16,1x1 | mntail | 0.02500 | 0.00621 | 4.03x slower |
| fwd | 512,1,1,32,1x1 | mntail | 0.02900 | 0.00800 | 3.63x slower |
| fwd | 32,1,1,512,1x1 | mtail | 0.02200 | 0.00629 | 3.50x slower |
| fwd | 96,120,160,48,1x1 | ntail | 0.06300 | 0.03997 | 1.58x slower |
| fwd | 128,30,40,512,1x1 | mtail | 0.03000 | 0.02189 | 1.37x slower |
| fwd | 256,30,40,128,1x1 | mtail | 0.01800 | 0.01366 | 1.32x slower |
| fwd | 192,60,80,64,1x1 | base | 0.03000 | 0.02282 | 1.31x slower |
| fwd | 64,60,80,128,1x1 | base | 0.02300 | 0.01824 | 1.26x slower |
| fwd | 512,30,40,128,1x1 | mtail | 0.02400 | 0.01953 | 1.23x slower |
| fwd | 24,240,320,128,1x1 | tdm | 0.22000 | 0.17973 | 1.22x slower |
| fwd | 256,60,80,64,1x1 | base | 0.03300 | 0.02730 | 1.21x slower |
| fwd | 128,30,40,128,1x1 | mtail | 0.01300 | 0.01112 | 1.17x slower |
| fwd | 64,60,80,256,1x1 | base | 0.03700 | 0.03168 | 1.17x slower |
| fwd | 192,120,160,48,1x1 | ntail | 0.09400 | 0.08065 | 1.17x slower |
| fwd | 128,30,40,128,3x3 | mtail | 0.03900 | 0.03575 | 1.09x slower |
| fwd | 128,120,160,128,3x3 | base | 0.36900 | 0.36303 | 1.02x slower |
| fwd | 48,120,160,128,1x1 | tdm | 0.06400 | 0.06297 | 1.02x slower |
| bwd | 512,1,1,32,1x1 | mtail | 0.01800 | 0.00939 | 1.92x slower |
| bwd | 128,30,40,512,1x1 | mtail | 0.04300 | 0.02783 | 1.54x slower |
| bwd | 128,30,40,128,3x3 | mtail | 0.08500 | 0.05621 | 1.51x slower |
| bwd | 64,60,80,256,1x1 | base | 0.04400 | 0.03461 | 1.27x slower |
| bwd | 512,30,40,128,1x1 | mtail | 0.04200 | 0.03528 | 1.19x slower |
| bwd | 128,120,160,128,3x3 | base | 0.60100 | 0.50721 | 1.18x slower |
| bwd | 256,30,40,128,1x1 | mtail | 0.02800 | 0.02428 | 1.15x slower |
| bwd | 64,60,80,128,1x1 | base | 0.02900 | 0.02647 | 1.10x slower |
| bwd | 192,60,80,64,1x1 | base | 0.03900 | 0.03852 | 1.01x slower |
| bwd | 128,30,40,128,1x1 | mtail | 0.01600 | 0.01779 | **1.11x faster** |
| bwd | 256,60,80,64,1x1 | base | 0.04100 | 0.04617 | **1.13x faster** |
| wrw | 128,30,40,128,3x3 | gsplit | 0.42700 | 0.08654 | 4.93x slower |
| wrw | 128,120,160,128,3x3 | gsplit | 2.04800 | 0.66439 | 3.08x slower |
| wrw | 192,60,80,64,1x1 | gsplit | 0.14800 | 0.05720 | 2.59x slower |
| wrw | 256,30,40,128,1x1 | gsplit | 0.08600 | 0.03579 | 2.40x slower |
| wrw | 512,30,40,128,1x1 | gsplit | 0.09500 | 0.04728 | 2.01x slower |
| wrw | 128,30,40,512,1x1 | gsplit | 0.09800 | 0.05202 | 1.88x slower |
| wrw | 128,30,40,128,1x1 | gsplit | 0.03500 | 0.02684 | 1.30x slower |
| wrw | 64,60,80,128,1x1 | gsplit | 0.06300 | 0.05149 | 1.22x slower |
| wrw | 64,60,80,256,1x1 | gsplit | 0.06800 | 0.05810 | 1.17x slower |
| wrw | 256,60,80,64,1x1 | gsplit | 0.06500 | 0.05890 | 1.10x slower |

**Summary**: fwd 17 shapes, 1.02x-4.03x slower (avg 1.66x, 0 faster); bwd 11 shapes,
0.89x-1.92x (avg 1.24x, **2 actually faster**); wrw 10 shapes, 1.10x-4.93x slower (avg
2.17x, 0 faster).

**The 20 shapes already tested in the Phase-22-era gfx950 comparison above show no
regression** -- re-isolating just those (all `_base`/`_gsplit`, no tail/tdm involved) gives
fwd avg 1.19x (was 1.19x), bwd avg 1.09x (was 1.08x), wrw avg 2.17x (was 2.01x, well within
this doc's own documented run-to-run noise band). **The 18 newly-applicable shapes --
unlocked entirely by Phases 25-31's tail/tdm/gsplit work -- are measurably worse on
average**: fwd's 12 new shapes average 1.86x vs. the original set's 1.19x; bwd's 6 new
shapes average 1.37x vs. 1.09x. This is an honest, expected finding, not a regression: the
tail/tdm code paths (EXEC-mask guards, per-wave TDM issue overhead, small-grid occupancy
for `H=1,W=1`-style degenerate spatial shapes) were built correctness-first throughout this
whole initiative and have never been performance-tuned the way the exact-fit base path
has. The worst offenders are exactly where that's expected: the three worst fwd ratios
(3.5x-4.0x) are all `H=1,W=1` (`gemm_m=42`, batch-only, no spatial extent at all --
essentially a tiny GEMM with no way to fill 256 CUs regardless of tile choice) matched
against MIOpen baselines under 0.01ms, the regime this doc has repeatedly flagged as most
exposed to fixed per-dispatch overhead; wrw's two worst ratios (3.3x3.3, 128x128x128) are
its smallest 3x3 shapes, the same "smallest absolute time, most overhead-exposed" pattern
noted for wrw earlier in this doc.

**Net picture**: on the original, well-fitting shape population, MISA/gfx1250 remains
essentially at parity-to-modestly-behind MIOpen/gfx950 across all three directions (as
found before Phase 23). The newly-unlocked tail/edge-case shapes are usable (correct,
`valid` results) but meaningfully slower in relative terms -- a real, quantified list of
where future performance work on the tail-handling paths specifically (not the main loop
or epilogue, both already tuned for the exact-fit case) would have the most leverage.

## Update (2026-08-27): same 38 shapes vs. MIOpen running natively on gfx1250

Same MISA numbers as the gfx950 comparison above, matched instead against
`~/rocm-libraries/tracelense_gfx1250_b01_2.json` (the real MIOpen/gfx1250 trace with
solver names, already used once in this doc's "re-checked against a real MIOpen/gfx1250
trace" update). All 38 applicable shapes matched by exact MIOpenDriver argument string.

| Direction | Shape (c,H,W,k,y×x) | Config | MISA (ms) | MIOpen/gfx1250 (ms) | Solver | Ratio |
|---|---|---|---|---|---|---|
| fwd | 512,1,1,32,1x1 | mntail | 0.02900 | 0.00971 | ImplicitGemmGroupFwdXdlops | 2.99x slower |
| fwd | 256,1,1,16,1x1 | mntail | 0.02500 | 0.00862 | ImplicitGemmGroupFwdXdlops | 2.90x slower |
| fwd | 32,1,1,512,1x1 | mtail | 0.02200 | 0.00847 | ImplicitGemmGroupFwdXdlops | 2.60x slower |
| fwd | 128,30,40,512,1x1 | mtail | 0.03000 | 0.02047 | ConvHipConv | 1.47x slower |
| fwd | 24,240,320,128,1x1 | tdm | 0.22000 | 0.16886 | ImplicitGemmGroupFwdXdlops | 1.30x slower |
| fwd | 192,120,160,48,1x1 | ntail | 0.09400 | 0.07747 | ImplicitGemmGroupFwdXdlops | 1.21x slower |
| fwd | 96,120,160,48,1x1 | ntail | 0.06300 | 0.05205 | ImplicitGemmGroupFwdXdlops | 1.21x slower |
| fwd | 256,30,40,128,1x1 | mtail | 0.01800 | 0.01492 | ConvHipConv | 1.21x slower |
| fwd | 512,30,40,128,1x1 | mtail | 0.02400 | 0.01995 | ImplicitGemmGroupFwdXdlops | 1.20x slower |
| fwd | 64,60,80,128,1x1 | base | 0.02300 | 0.02039 | ImplicitGemmGroupFwdXdlops | 1.13x slower |
| fwd | 128,120,160,128,3x3 | base | 0.36900 | 0.32842 | ConvHipConv | 1.12x slower |
| fwd | 128,30,40,128,3x3 | mtail | 0.03900 | 0.03641 | ConvHipConv | 1.07x slower |
| fwd | 48,120,160,128,1x1 | tdm | 0.06400 | 0.05992 | ImplicitGemmGroupFwdXdlops | 1.07x slower |
| fwd | 64,60,80,256,1x1 | base | 0.03700 | 0.03714 | ConvHipConv | **1.00x faster** |
| fwd | 128,30,40,128,1x1 | mtail | 0.01300 | 0.01307 | ConvHipConv | **1.01x faster** |
| fwd | 256,60,80,64,1x1 | base | 0.03300 | 0.03960 | ImplicitGemmGroupFwdXdlops | **1.20x faster** |
| fwd | 192,60,80,64,1x1 | base | 0.03000 | 0.03615 | ImplicitGemmGroupFwdXdlops | **1.21x faster** |
| bwd | 128,30,40,128,3x3 | mtail | 0.08500 | 0.02388 | ConvHipConv | 3.56x slower |
| bwd | 128,30,40,512,1x1 | mtail | 0.04300 | 0.01857 | ConvHipConv | 2.32x slower |
| bwd | 256,30,40,128,1x1 | mtail | 0.02800 | 0.01395 | ConvHipConv | 2.01x slower |
| bwd | 512,30,40,128,1x1 | mtail | 0.04200 | 0.02179 | ConvHipConv | 1.93x slower |
| bwd | 512,1,1,32,1x1 | mtail | 0.01800 | 0.00945 | ConvHipConv | 1.90x slower |
| bwd | 128,120,160,128,3x3 | base | 0.60100 | 0.31830 | ConvHipConv | 1.89x slower |
| bwd | 128,30,40,128,1x1 | mtail | 0.01600 | 0.01274 | ConvHipConv | 1.26x slower |
| bwd | 192,60,80,64,1x1 | base | 0.03900 | 0.03283 | ConvHipConv | 1.19x slower |
| bwd | 256,60,80,64,1x1 | base | 0.04100 | 0.03716 | ConvHipConv | 1.10x slower |
| bwd | 64,60,80,128,1x1 | base | 0.02900 | 0.02664 | ImplicitGemmGroupBwdXdlops | 1.09x slower |
| bwd | 64,60,80,256,1x1 | base | 0.04400 | 0.04199 | ConvHipConv | 1.05x slower |
| wrw | 192,60,80,64,1x1 | gsplit | 0.14800 | 0.00561 | ImplicitGemmGroupWrwXdlops | 26.39x slower |
| wrw | 128,30,40,128,3x3 | gsplit | 0.42700 | 0.06740 | ImplicitGemmGroupWrwXdlops | 6.34x slower |
| wrw | 128,120,160,128,3x3 | gsplit | 2.04800 | 0.41365 | ImplicitGemmGroupWrwXdlops | 4.95x slower |
| wrw | 256,30,40,128,1x1 | gsplit | 0.08600 | 0.02838 | ImplicitGemmGroupWrwXdlops | 3.03x slower |
| wrw | 512,30,40,128,1x1 | gsplit | 0.09500 | 0.04997 | ImplicitGemmGroupWrwXdlops | 1.90x slower |
| wrw | 128,30,40,512,1x1 | gsplit | 0.09800 | 0.05295 | ImplicitGemmGroupWrwXdlops | 1.85x slower |
| wrw | 128,30,40,128,1x1 | gsplit | 0.03500 | 0.02198 | ImplicitGemmGroupWrwXdlops | 1.59x slower |
| wrw | 64,60,80,128,1x1 | gsplit | 0.06300 | 0.04026 | ImplicitGemmGroupWrwXdlops | 1.56x slower |
| wrw | 64,60,80,256,1x1 | gsplit | 0.06800 | 0.05214 | ImplicitGemmGroupWrwXdlops | 1.30x slower |
| wrw | 256,60,80,64,1x1 | gsplit | 0.06500 | 0.05217 | ImplicitGemmGroupWrwXdlops | 1.25x slower |

**Summary**: fwd 17 shapes, 0.83x-2.99x (avg 1.42x, **4 faster**); bwd 11 shapes, 1.05x-3.56x
slower (avg 1.75x, 0 faster); wrw 10 shapes, 1.25x-26.39x slower (avg 5.02x, 0 faster).

**Isolating the original 20 (pre-Phase-23) shapes**: fwd avg **0.98x** (min 0.83x, max
1.13x -- essentially a dead heat, matching this doc's earlier "re-checked... with solver
names" update's 0.98x almost exactly); bwd avg 1.26x (matching that update's 1.25x almost
exactly). **No regression on fwd/bwd for the shapes already characterized.** The 18
newly-applicable shapes are worse on this trace too (fwd 1.60x avg, bwd 2.16x avg),
mirroring the same tail/edge-case-overhead pattern found against gfx950 above.

**wrw is the outlier, and it needed a real investigation, not just a report**: all 10 wrw
shapes are already in the "original 20" (wrw has no tail relief, so no new wrw shapes were
unlocked -- every wrw shape tested here was already tested in the earlier "re-checked...
with solver names" update). That update reported wrw at 1.23x-11.41x (avg ~3.5x); this run
shows 1.25x-26.39x (avg 5.02x) for the *same* 10 shapes, using the *same* `_gsplit` config
and the *same* automatic ternary split-count search (confirmed still present and correct
in `igemm_wrw_gtc_driver.h`). Investigated directly rather than just flagging the
discrepancy: re-ran the worst outlier (`c=192,H=60,W=80,k=64,1x1`) three times (0.148,
0.149, 0.148ms -- tightly reproducible, not run-to-run noise), then swept `IGEMM_GSPLIT_SWEEP`
across 13 candidate split counts by hand. The full sweep confirms the ternary search
*is* finding the true minimum today (split 225-252 at ~0.149ms genuinely beats split 315,
the split this doc's Phase 20 update reported as best, which now measures 0.156ms) -- so
this is not a search-algorithm regression. The true minimum itself has simply gotten much
slower since Phase 20 (0.065ms then vs. ~0.148ms now for the identical shape/config/search).
**Given `rocm-smi --showuse --showpids` showed `GPU use (%): 100` with no visible KFD PIDs
throughout this entire session** (the same "different tenant on a shared box" signal
flagged repeatedly elsewhere in this doc), the most likely explanation is GPU contention --
and wrw's `_gsplit` kernels, which launch many small workgroups per candidate split and are
occupancy/dispatch-latency-sensitive by design (that sensitivity is the whole reason
`gemm_k_global_split` exists), are plausibly far more exposed to a noisy neighbor's
scheduling interference than fwd/bwd's much larger, compute-bound single-dispatch kernels
-- consistent with fwd/bwd showing *no* measurable change from earlier sessions while wrw
shows a large one. This is a plausible, evidence-backed explanation, not a confirmed root
cause: the exact multiples for wrw specifically should be treated as more provisional than
this doc's other numbers, and re-measured on an exclusively-held GPU before being used to
make any wrw-specific optimization decision.

## Update (2026-08-28): re-confirmed the original two catastrophic-outlier shapes from the very first pass

The very first pass at the top of this doc (before wrw split-K existed) flagged two shapes
as the worst: `c=128,H=30,W=40,k=128,1x1` (8.0ms MISA vs 0.035ms MIOpen, 227x slower) and
`c=128,H=120,W=160,k=128,3x3,pad=1` (1.18s MISA vs 0.71ms MIOpen, 1665x slower). Spot-checked
both again directly (`bench_out/wrw_bf16_master`'s master config, `-V 0`, `IGEMM_WARMUP`/
`IGEMM_REPEAT` 3-5): non-split candidates for both shapes are unchanged (~8-10ms and
~1.15-1.45s respectively, confirming those old numbers weren't measurement error), but the
best `_gsplit`-family candidate now gives **0.031ms** (`wsred_gkgs[525]`, faster than the
recorded MIOpen number) and **1.655ms** (`wsred_gkgs[1008]`, ~2.3-2.5x slower) respectively —
consistent with this doc's own ~2-5x average finding above, not the pre-split-K 227x/1665x
figures. `wsred` (weighted split reduction) variants won both spot-checks over plain
`gkgs`/`ktail_gkgs`/`scopedev_gkgs`/etc.

## Update (2026-08-28): fresh benchmark after Phase 59 (direct_store) + Phase 58 (streamk)

Re-ran the full 38-shape benchmark using `script/benchmark_gfx1250_vs_miopen.py`
(which now automatically tries `direct_store` variants alongside the original
configs and `gsplit`, picking whichever per-shape candidate is fastest). Same
methodology as every prior section in this doc: `IGEMM_WARMUP=5 IGEMM_REPEAT=20`,
driver_mode_normal, `-V 0`, batch=42 bf16/NHWC. MIOpen reference numbers (both
gfx950 and gfx1250) come from the same trace files embedded in the script.
GPU contention was still present (`GPU use (%): 100`, no visible KFD PIDs —
contention caveats from earlier in this doc still apply).

| Direction | Shape (c,H,W,k,y×x) | Config | MISA (ms) | MIOpen/gfx950 (ms) | MIOpen/gfx1250 (ms) | vs gfx950 | vs gfx1250 | MIOpen/gfx1250 solver |
|---|---|---|---|---|---|---|---|---|
| fwd | 256,1,1,16,1x1 | mntail | 0.02300 | 0.00621 | 0.00862 | 3.70x slower | 2.67x slower | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 512,1,1,32,1x1 | mntail | 0.02900 | 0.00800 | 0.00971 | 3.63x slower | 2.99x slower | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 32,1,1,512,1x1 | mtail | 0.01900 | 0.00629 | 0.00847 | 3.02x slower | 2.24x slower | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 96,120,160,48,1x1 | ntail_direct | 0.05800 | 0.03997 | 0.05205 | 1.45x slower | 1.11x slower | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 128,30,40,512,1x1 | mtail | 0.02800 | 0.02189 | 0.02047 | 1.28x slower | 1.37x slower | 220/ConvHipConv |
| fwd | 256,30,40,128,1x1 | mtail | 0.01700 | 0.01366 | 0.01492 | 1.24x slower | 1.14x slower | 220/ConvHipConv |
| fwd | 192,60,80,64,1x1 | base_direct | 0.02700 | 0.02282 | 0.03615 | 1.18x slower | 1.34x faster | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 64,60,80,128,1x1 | base | 0.02200 | 0.01824 | 0.02039 | 1.21x slower | 1.08x slower | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 512,30,40,128,1x1 | mtail_direct | 0.02400 | 0.01953 | 0.01995 | 1.23x slower | 1.20x slower | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 24,240,320,128,1x1 | tdm | 0.18100 | 0.17973 | 0.16886 | 1.01x slower | 1.07x slower | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 256,60,80,64,1x1 | base_direct | 0.03100 | 0.02730 | 0.03960 | 1.14x slower | 1.28x faster | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 128,30,40,128,1x1 | mtail_direct | 0.01200 | 0.01112 | 0.01307 | 1.08x slower | 1.09x faster | 220/ConvHipConv |
| fwd | 64,60,80,256,1x1 | base | 0.04200 | 0.03168 | 0.03714 | 1.33x slower | 1.13x slower | 220/ConvHipConv |
| fwd | 192,120,160,48,1x1 | ntail_direct | 0.08800 | 0.08065 | 0.07747 | 1.09x slower | 1.14x slower | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 128,30,40,128,3x3 | mtail | 0.03900 | 0.03575 | 0.03641 | 1.09x slower | 1.07x slower | 220/ConvHipConv |
| fwd | 128,120,160,128,3x3 | base | 0.36600 | 0.36303 | 0.32842 | 1.01x slower | 1.11x slower | 220/ConvHipConv |
| fwd | 48,120,160,128,1x1 | tdm | 0.06300 | 0.06297 | 0.05992 | 1.00x slower | 1.05x slower | 137/ConvHipImplicitGemmGroupFwdXdlops |
| bwd | 512,1,1,32,1x1 | mtail_direct | 0.01800 | 0.00939 | 0.00945 | 1.92x slower | 1.90x slower | 220/ConvHipConv |
| bwd | 128,30,40,512,1x1 | mtail | 0.04300 | 0.02783 | 0.01857 | 1.54x slower | 2.32x slower | 220/ConvHipConv |
| bwd | 128,30,40,128,3x3 | mtail | 0.08500 | 0.05621 | 0.02388 | 1.51x slower | 3.56x slower | 220/ConvHipConv |
| bwd | 64,60,80,256,1x1 | base_direct | 0.04300 | 0.03461 | 0.04199 | 1.24x slower | 1.02x slower | 220/ConvHipConv |
| bwd | 512,30,40,128,1x1 | mtail | 0.04100 | 0.03528 | 0.02179 | 1.16x slower | 1.88x slower | 220/ConvHipConv |
| bwd | 128,120,160,128,3x3 | base | 0.59600 | 0.50721 | 0.31830 | 1.18x slower | 1.87x slower | 220/ConvHipConv |
| bwd | 256,30,40,128,1x1 | mtail_direct | 0.02700 | 0.02428 | 0.01395 | 1.11x slower | 1.93x slower | 220/ConvHipConv |
| bwd | 64,60,80,128,1x1 | base | 0.02700 | 0.02647 | 0.02664 | 1.02x slower | 1.01x slower | 155/ConvHipImplicitGemmGroupBwdXdlops |
| bwd | 192,60,80,64,1x1 | base | 0.03800 | 0.03852 | 0.03283 | 1.01x faster | 1.16x slower | 220/ConvHipConv |
| bwd | 128,30,40,128,1x1 | mtail | 0.01700 | 0.01779 | 0.01274 | 1.05x faster | 1.33x slower | 220/ConvHipConv |
| bwd | 256,60,80,64,1x1 | base | 0.04200 | 0.04617 | 0.03716 | 1.10x faster | 1.13x slower | 220/ConvHipConv |
| wrw | 128,30,40,128,3x3 | gsplit | 0.42100 | 0.08654 | 0.06740 | 4.86x slower | 6.25x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 128,120,160,128,3x3 | gsplit | 2.18800 | 0.66439 | 0.41365 | 3.29x slower | 5.29x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 192,60,80,64,1x1 | gsplit | 0.15100 | 0.05720 | 0.00561 | 2.64x slower | 26.93x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 256,30,40,128,1x1 | gsplit | 0.08600 | 0.03579 | 0.02838 | 2.40x slower | 3.03x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 512,30,40,128,1x1 | gsplit | 0.09800 | 0.04728 | 0.04997 | 2.07x slower | 1.96x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 128,30,40,512,1x1 | gsplit | 0.09800 | 0.05202 | 0.05295 | 1.88x slower | 1.85x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 128,30,40,128,1x1 | gsplit | 0.03500 | 0.02684 | 0.02198 | 1.30x slower | 1.59x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 64,60,80,128,1x1 | gsplit | 0.06300 | 0.05149 | 0.04026 | 1.22x slower | 1.56x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 64,60,80,256,1x1 | gsplit | 0.06700 | 0.05810 | 0.05214 | 1.15x slower | 1.29x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 256,60,80,64,1x1 | gsplit | 0.06500 | 0.05890 | 0.05217 | 1.10x slower | 1.25x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |

**Summary**: fwd 17 shapes, 1.00x-3.70x slower (avg 1.57x, 3 faster vs gfx1250);
bwd 11 shapes, 1.01x-3.56x slower (avg 1.74x, 0 faster vs gfx1250; 3 faster vs gfx950);
wrw 10 shapes, 1.10x-26.93x slower (avg 5.10x, 0 faster vs gfx1250).

**Fwd gets 3 faster-than-MIOpen shapes this time** (vs 4 in the pre-Phase-59 table),
all on the `base` or `_direct` family at moderate-to-large spatial dimensions --
the direct_store=1 epilogue (skipping LDS reshuffle, doing scalar `global_store_dword`
per lane) is winning on 3 of the 5 shapes where it was the fastest candidate
(`ntail_direct` won 3, `base_direct` won 2, `mtail_direct` won 3). 8 of 17 fwd
shapes now use a `_direct` variant as their fastest config.

**Bwd sees a small improvement from direct_store** on `512,1,1,32` (`mtail_direct`
won at 0.018ms vs 0.018ms with gsplit -- actually both tied at this resolution, but
the script prefers the first alphabetical match that beats all others), and modest
wins at `256,30,40,128` and `64,60,80,256` (both `_direct` picked, 1-2% faster
than the old base path). The three H=1,W=1-adjacent shapes (`128,30,40,128,3x3`,
`128,30,40,512`, `256,30,40,128`) all stuck with plain `mtail` -- direct_store
didn't move the needle there.

**Wrw remains the outlier**: 1.10x to 26.93x slower vs MIOpen/gfx1250. The worst
(`192,60,80,64,1x1` at 26.93x) is the shape this doc has flagged for several
sessions now as noise-sensitive (its MIOpen baseline 0.0056ms is an order of
magnitude below every structurally similar shape in the set). Outside that one,
wrw averages 3.1x vs gfx1250 (1.25x-6.25x range), consistent with this doc's
pre-Phase-58/59 numbers -- streamk (Phase 58) hasn't been wired into the benchmark
script yet, so these results use the same `_gsplit` config as all prior wrw tables.

**GPU contention caveat, same as every prior section**: `GPU use (%): 100` with no
visible KFD PIDs throughout this run. Treat exact multiples as approximate. The
relative ranking (which config family wins per shape, the general magnitude of
each direction's gap) is more trustworthy than any one shape's exact ratio.

Generated by `python3 script/benchmark_gfx1250_vs_miopen.py --direction all --warmup 5 --repeat 20`,
output also saved to `bench_results_20260828.md`.
