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

## Prior-session dominance findings (unchanged by this review)

Carried over from the same Phase 6 pass, all already tested across the corrected,
broader shape set including the long-K/large-grid additions above where applicable:

- **`ds_load_tr_b`** (bwd/wrw fp16/bf16 native 16-bit transpose load): irregular
  tradeoff, confirmed real (stable over repeats, effect sizes 2-38%). 64-tile: always
  wins with tr_b=1. 128-tile: tr_b=0 wins on large-K/N 1x1 bottleneck shapes (-7% to
  -9% for tr_b=1), tr_b=1 wins on 3x3 shapes (+5% to +9%). **Action item still open:**
  `script/generate_all_configs.py`'s combinatorial FLAGS never varies `ds_load_tr_b`
  (it's promoted to an unconditional default), so the 128-tile's tr_b=0 win is
  currently unreachable by the searched config surface -- should be added as a
  combinatorial toggle for bwd/wrw.
- **`wrw_incremental_gather`**: no measurable effect (deltas 0.1-1.6%, inside the
  noise floor). No action.
- **`wmma_setprio`**: small, inconsistent effect (+0.2% to +2% on most shapes,
  occasionally negative/noisy on extreme shapes). Keep as an independent tunable.
- **`wmma_gap_hoist`**: moderate, mostly-positive effect (+4% to +20% depending on
  shape, including the long-K shapes above), but not universal. Irregular tradeoff --
  keep, do not make unconditional.
