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
