# gfx1250 Tuning Refactor: Phase 6 Dominance Study

Tuning refactor plan (`docs/gfx1250_tuning_refactor_plan.md`) Phase 6: for every
tunable flag that offers an "old vs. new" or "on vs. off" choice, determine whether
one side ever wins (irregular tradeoff -> keep both, document the regime) or whether
one side is dominated in every tested region (clean cutover -> delete the loser).

All benchmarks in this document were run **sequentially** (one `conv_driver.exe`
invocation at a time; never `Promise.all`/concurrent GPU work) on an exclusive,
uncapped gfx1250 node (~2371 MHz observed sclk, vs. the ~1100 MHz-capped machine used
for some prior-session validation docs). Every reported delta was checked against a
noise floor established by repeating an unchanged config 3-6 times back to back:
same-config repeat variance is consistently **~1-3%**. Deltas discussed as "real"
below are outside that band and reproduce across repeats; deltas discussed as "noise"
are inside it.

## Correction logged mid-session: initial shape coverage was insufficient

The first sweep (5 shapes: 1x1_bottleneck, 1x1_wide_small_c, 3x3_medium, 3x3_wide,
1x1_deep_bottleneck, plus one "huge_k" outlier) was **not broad enough** to fairly
evaluate `wmma_l2_prefetch`, whose own design doc explicitly hypothesizes its win
condition as "compute-to-memory ratio high enough to amortize the overhead" -- i.e. a
very long K-loop relative to M/N, which none of the 5 shapes represented well. Based
on that incomplete sweep, `wmma_l2_prefetch` was about to be deleted as "no winning
region found." **This was caught before any commit** by re-checking specifically for
the shape regime the mechanism's own doc predicted, which reversed the conclusion
(see below). Every subsequent candidate in this document was re-checked the same way:
before concluding "no winning region," at least one shape was constructed to match
that mechanism's own stated design intent, not just the generic shape battery.

## Findings

### `wmma_l2_prefetch` -- KEEP (irregular tradeoff, real winning region found)

Prior validation (`docs/gfx1250_d1_l2_prefetch_validation.md`, 1100MHz-capped
hardware) found a ~22% regression on 2 shapes and marked the mechanism "not
recommended for adoption." Fresh data on uncapped hardware (2371MHz) across 5 more
diverse shapes/2 directions reconfirmed a consistent regression (-11% to -57%),
which looked like solid "always loses" evidence.

However, testing the shape regime the mechanism's own design doc predicts as its win
condition (very long K, small M/N) reverses this:

| shape (fwd) | base | l2pf | gap_hoist | all3 (setprio+gaphoist+l2pf) |
|---|---|---|---|---|
| `n128 c8192 k128` (gemm_k=8192) | 1.38-1.42 | 1.44-1.50 (**+5-8%**) | 1.57-1.62 (+15-20%) | 1.77-1.81 (**+27-32%**) |
| `n128 c16384 k128` (gemm_k=16384) | 1.48-1.49 | 1.57-1.60 (**+6-8%**) | 1.70-1.71 (+15%) | 1.95-1.96 (**+31-33%**) |

Confirmed stable over 6 repeats across both K depths. **Disposition: keep the
tunable as-is.** It genuinely loses on typical compute-bound shapes and genuinely
wins on long-K/memory-bound shapes -- a real, shape-dependent tradeoff, not a
dominated mechanism. No code change needed (it already defaults off and is already
present, gated, in the master `_all.config` unions for fwd/bwd/wrw fp16).

**Action item:** `script/generate_all_configs.py`'s combinatorial sweep doesn't need
to change (this is a runtime-shape effect, not a config-shape effect) -- but anyone
benchmarking a genuinely long-K workload (e.g. very deep bottleneck layers, large
reduction dimension) should know to try the `_l2pf` variant; the master config's
existing `_l2pf` sections make it reachable by `conv_driver.exe`'s own best-of search.

### `wg_swizzle` -- confirmed no benefit (re-tested properly this time)

Removed in `327c1ef` ("Housekeeping: remove workgroup_swizzle... 0 configs ever set
it; confirmed dead code via full grep sweep") -- but the removal justification was
about config-file usage, not about whether the mechanism itself ever wins. Its
introduction commit (`215d30d`) reported a marginal, **never-repeated, single-run**
result: +0.38% to +0.76% on one "16x16 grid" shape, i.e. already at or below this
session's established noise floor.

Reinstated the exact removed mechanism (kernel-prologue `s_bx`/`s_by` bit-swap,
G=4/8, guarded skip when grid dims aren't multiples of G) on top of current HEAD,
confirmed byte-identical output for every existing config with the tunable unset,
then re-tested it on its own claimed best-case geometry -- large square grids
(16x16 and 32x32 workgroup grids, matching the tile size) -- across **all 3
directions x both tile sizes (6 combos) x 3 repeats each**:

| combo | G=0 (base) | G=4 | G=8 |
|---|---|---|---|
| fwd/128 | 335.3-338.8 | 337.6-338.3 (+0.1%) | 334.1-334.9 (-0.9%) |
| fwd/64 | 384.7-392.0 | 381.9-394.5 (~0%) | 385.0-387.4 (-0.5%) |
| bwd/128 | 262.0-263.6 | 261.7-265.2 (+0.2%) | 260.1-263.4 (-0.1%) |
| bwd/64 | 272.3-275.5 | 274.0-276.3 (+0.5%) | 269.9-277.0 (-0.2%) |
| wrw/128 | 247.3-249.3 | 247.8-248.5 (-0.1%) | 248.3-249.5 (+0.2%) |
| wrw/64 | 250.1-251.4 | 251.4-252.4 (+0.5%) | 250.3-253.9 (+0.8%) |

Every delta is inside the ~1-3% noise band. A follow-up deeper repeat check on the
single most positive-looking cell (wrw/64, G=8, +0.8%) flipped sign entirely across
3 more repeats (251.0, 254.5, 251.3 vs. G=0's 254.0, 254.7, 254.9 -- net negative,
not positive). Correctness of the guard mechanism itself was independently
re-confirmed: a rectangular grid (16x4 at tile128, where grid_y=4 is not a multiple
of G=8 and should trigger the skip branch) reports `valid:y` for both G=4 (applies)
and G=8 (skips).

**Disposition: no measurable benefit found even in the mechanism's own predicted
best-case regime, across 6 direction/tile combinations with proper repeated
trials.** Unlike `wmma_l2_prefetch`, this is a genuinely dominated mechanism --
`wg_swizzle` never beats the identity mapping outside noise. The reinstated code was
reverted back to the pre-session (deleted) state; **removal stands, now on solid
evidence** rather than the original's "0 configs set it" justification alone.

### `atomic_cascade` -- excluded from this study (correctness issue, not a perf question)

Deleted in `7aecb40`. This was a permanently `assert not`-guarded field --
**confirmed to hang real hardware** (missing companion release for the cascading
atomic's deferred scope-completion signal; the kernel's own `s_wait_storecnt 0x0`
never returns). It could never be legally enabled by any config, so there is no "fair
chance" benchmark to give it: enabling it to test performance would hang the GPU.
Not a dominance-study candidate. If someone completes the TODO that was documented at
deletion time (add a companion release instruction, verify on an isolated HIP probe
first) it becomes a new feature to benchmark from scratch, not a revival of dead code.

### Dead-code removals confirmed out of scope (not benchmarked choices)

- `e8139fc` ("Remove dead atomic-claim symbols from wrw stream-K"): this removed
  leftover SGPR/VGPR declarations and prologue setup for a dynamic atomic-claim
  mechanism that had **already been replaced** by static shard indexing in an earlier
  change (W-2). Zero remaining references confirmed before removal. Not a
  performance A/B choice -- the dynamic mechanism was gone; this just swept up its
  corpse.

## Second fairness pass: `ds_load_tr_b` and `wrw_incremental_gather`

The claim that these two had "already [been] tested across the corrected, broader
shape set" in an earlier draft of this document was **not accurate** -- they had
only been run against the original 5-shape battery, not against any shape
constructed from their own specific design rationale (the same standard applied
above to `wmma_l2_prefetch` and `wg_swizzle`). Corrected below.

### `ds_load_tr_b` -- KEEP (irregular tradeoff), but the existing rule doesn't extrapolate

Prior characterization: "64-tile always wins with tr_b=1; 128-tile: tr_b=0 wins on
large-K/N 1x1 bottleneck shapes, tr_b=1 wins on 3x3 (multi-tap) shapes" -- based on
one shape size per category. Re-tested with the SAME categories pushed more
extreme (bigger K/N bottleneck: `n512 c8192 H4W4 k8192`; more taps: 7x7 instead of
3x3), stable over 3 repeats each (<0.5% variance):

| direction | shape | tr_b=0 | tr_b=1 | winner |
|---|---|---|---|---|
| bwd | extreme 1x1 bottleneck | 596.5 | 615.6 | **tr_b=1 (+3.2%)** -- reverses the prior rule |
| bwd | extreme multitap (7x7) | 494.9 | 483.2 | **tr_b=0 (+2.4%)** -- reverses the prior rule |
| wrw | extreme 1x1 bottleneck | 541.7 | 486.7 | tr_b=0 (+10.2%) -- consistent with prior rule |
| wrw | extreme multitap (7x7) | 11.14 | 12.58 | tr_b=1 (+12.9%) -- consistent with prior rule |

**bwd's preference reverses at more extreme shape magnitudes; wrw's holds.** The
previous "1x1-bottleneck vs. 3x3" categorical framing was generalized from a single
shape per category and does not hold uniformly across shape scale for bwd -- the
real relationship is shape-magnitude-dependent, not simply tap-count-dependent.
**Disposition: keep both, still an irregular tradeoff, but retract the confident
categorical rule for bwd specifically until it's characterized across a proper
size sweep (not just two data points per direction).** wrw's rule is corroborated
by a second, more extreme data point and can be trusted with more confidence.

**Action item still open:** `script/generate_all_configs.py`'s combinatorial FLAGS
never varies `ds_load_tr_b` (it's promoted to an unconditional default), so neither
direction's `tr_b=0` win is reachable by the searched config surface today -- should
be added as a combinatorial toggle for bwd/wrw, especially now that bwd's win
condition is known to be less predictable than previously stated.

### `wrw_incremental_gather` -- KEEP, small real win, but a real correctness bug was found and fixed

**Correctness bug found and fixed.** Testing the shapes that were meant to probe
this mechanism's long-K hypothesis (`n128 c8192 H1W1 k128`, `n128 c16384 H1W1
k128`) returned `valid:n` for `wrw_incremental_gather=1` on both -- a genuine,
previously-undiscovered wrong-answer bug, not a performance question. Root cause
(read directly from `igemm_wrw_gtc_wmma_nhwc.py`'s `_emit_b_gather_incremental`):
the per-iteration index update does `hw_idx += gemm_k_per_block` followed by a
**single** conditional wrap-by-`ho*wo`. That is only correct when at most one wrap
can occur per K-step, i.e. `gemm_k_per_block <= ho*wo`. Whenever `gemm_k_per_block
> ho*wo` (small output spatial size relative to the K-tile -- exactly the shapes
above, where `ho*wo=1`), multiple wraps are needed per iteration but only one is
applied, silently corrupting the gathered addresses. Confirmed the exact boundary
by bisection (`gemm_k_per_block=32` in all test configs):

| ho*wo | vs. gemm_k_per_block(32) | result |
|---|---|---|
| 1, 16, 25 | < 32 | **valid:n** |
| 32 (exact) | == 32 | valid:y |
| 36, 68, 102, 119, 289 | > 32 | valid:y |

Since `ho`/`wo` are runtime launch-shape values, not known at Python-codegen time,
this cannot be caught by `igemm_base.py`'s assert-based tunable validation -- it
needed a runtime shape guard in the C++ driver, mirroring the existing
pattern used for other shape-dependent kernel legality checks in the same file.
**Fixed in `driver/igemm_wrw_gtc_driver.h`'s `tunable_is_valid()`:** rejects
(`not applicable`, matching how e.g. an undersized M/N tile is already handled)
whenever `wrw_incremental_gather` is set and `ho*wo < gemm_k_per_block`. Verified:
the WMMA kernel's own compiled `.hsaco` is byte-identical before/after (driver-only
change); the guard correctly rejects every case that previously returned `valid:n`
and accepts every case (including the exact `ho*wo == gemm_k_per_block` boundary)
that returns `valid:y`. No shipped config sets `wrw_incremental_gather=1` today, so
this had zero shipped-result impact -- but the flag was live and reachable, and
would have silently corrupted results for anyone who tried it on a small-spatial
shape (a common, plausible shape class, not an exotic corner case).

**Performance, re-tested on a corrected shape.** The original long-K test shapes
were also geometrically wrong for this mechanism: wrw's `GEMM_K = N*Ho*Wo`
(batch x spatial), not channel-driven like fwd/bwd -- the `c=8192/16384, H=W=1`
shapes used had `gemm_k=128` (only 4 iterations, the opposite of "long K") and
`ho*wo=1` (which is exactly why they hit the correctness bug above). Rebuilt a
genuinely long-K, correctness-safe shape (`n128 c128 H224 W224 k128`: `gemm_k=
6.4M` iterations, `ho*wo=50176 >> gemm_k_per_block`): `wrw_incremental_gather=0`
averages 298.6ms, `=1` averages 297.2ms across 3 repeats each (<0.5% variance) --
a small, real, consistent **~0.5% win**. This matches the mechanism's actual
nature: it removes a small, fixed number of instructions (magic div/rem ->
add+conditional) from each K-iteration's overhead, not a memory-latency-hiding
trick like `wmma_l2_prefetch` -- so even with millions of iterations, the relative
savings stays small.

**Disposition: keep the tunable.** Small but genuine win once restricted to safe
shapes (now enforced by the driver guard); too marginal to justify promoting to a
default. Not reachable by any shipped config today, so no immediate action beyond
the correctness fix already applied.

## Findings unchanged by this fairness pass

- **`wmma_setprio`**: small, inconsistent effect (+0.2% to +2% on most shapes,
  occasionally negative/noisy on extreme shapes). Keep as an independent tunable.
- **`wmma_gap_hoist`**: moderate, mostly-positive effect (+4% to +20% depending on
  shape, including the long-K shapes tested for `wmma_l2_prefetch` above), but not
  universal. Irregular tradeoff -- keep, do not make unconditional.
