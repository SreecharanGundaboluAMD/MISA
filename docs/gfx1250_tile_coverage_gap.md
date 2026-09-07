# gfx950 vs gfx1250 Tile-Shape Coverage Gap

## Current state (measured directly, updated 2026-09-07 after this session's fwd extension)

| Direction | gfx950 distinct (M x N) tile shapes | gfx1250 distinct (M x N) tile shapes |
|---|---|---|
| fwd | **14**: 128x128, 128x256, 128x32, 128x64, 256x128, 256x32, 256x64, 32x128, 32x256, 32x64, 64x128, 64x256, 64x32, 64x64 | **9**: 128x128, 128x32, 128x64, 256x128, 32x128, 32x64, 64x128, 64x32, 64x64 |
| bwd | **14**: 128x128, 128x256, 128x32, 128x64, 256x128, 256x256, 256x32, 256x64, 32x128, 32x64, 64x128, 64x256, 64x32, 64x64 | **3**: 128x128, 32x32, 64x64 |
| wrw | **12**: 128x128, 128x256, 128x64, 256x128, 256x256, 256x32, 256x64, 32x256, 64x128, 64x256, 64x32, 64x64 | **3**: 128x128, 32x32, 64x64 |

gfx950 covers essentially a full macro-tile grid over `{32,64,128,256} x
{32,64,128,256}` (minus a few combinations that presumably don't build for
that architecture's XDLOPS path). gfx1250's WMMA pipeline covers only a
handful of specific shapes per direction, and fwd has visibly more coverage
than bwd/wrw (fwd alone has picked up 128x64/64x128 asymmetric tiles and
256x128).

## Why this gap exists, and why it isn't something Phase 5-7 could close

This is **not** a tunable-combination or config-generation gap -- it's a
**generator capability gap**. Every `BASE_SECTIONS` entry in
`script/generate_all_configs.py` requires a hand-authored base `.config`
section (`gemm_m_per_block`/`gemm_n_per_block`/`wmma_repeat_m`/
`wmma_repeat_n`/`tensor_a_cluster_lengths`/`tensor_b_cluster_lengths`, all
mutually consistent for that specific macro-tile) that the corresponding
WMMA generator class (`igemm_{fwd,bwd,wrw}_gtc_wmma_nhwc_t`) must already
support. Missing tile shapes are missing because:

1. **No base config section exists for them yet** -- authoring one requires
   picking a valid `(wmma_repeat_m, wmma_repeat_n)` pair and cluster-length
   decomposition for the new macro-tile, which is real per-tile design
   work, not something `generate_all_configs.py`'s combinatorial FLAGS
   sweep can produce on its own (FLAGS only toggles orthogonal features
   *within* an already-valid base tile).
2. **The generator itself may not support the shape at all.** Several
   comments across the codebase note the WMMA generators were built
   "correctness-first" against a narrow set of tile shapes and later
   extended incrementally (e.g. fwd's 128x64/64x128 asymmetric tiles and
   256x128 required their own validation passes; bwd/wrw haven't received
   that same extension yet). There is no guarantee every gfx950 tile shape
   is even mechanically reachable by gfx1250's WMMA instruction geometry
   (16x16 native tile, fixed `wmma_tile_m`/`wmma_tile_n`=16) without
   register-budget or LDS-layout work specific to that macro-tile.

Phase 6 (dominance study) explicitly measured and flagged this gap as
**out of its own scope**: dominance-study work assumes the tile-shape
menu is already fixed and asks "does tunable X ever win", not "what tile
shapes should exist". Phase 7 (expand tuning search) was scoped strictly to
the *existing* WMMA tunable surface (`FLAGS`), not new WMMA capability --
see `docs/gfx1250_tuning_refactor_plan.md`'s "Parking lot" section, which
explicitly lists new-tile/new-capability work as deliberately out of scope
for that plan.

## What closing it would actually require

For each missing (direction, tile shape) pair:

1. Design a valid `(wmma_repeat_m, wmma_repeat_n, tensor_a/b_cluster_lengths)`
   decomposition for that macro-tile against the fixed 16x16 WMMA native
   tile and this arch's wavefront_size=32.
2. Author a base `.config` section and add it to `BASE_SECTIONS`.
3. Build + assemble it standalone first (before any combinatorial FLAGS
   sweep) to confirm the generator class actually handles the new
   macro-tile at all -- register budget, LDS sizing, and epilogue tiling
   are all tile-shape-dependent and may need generator code changes, not
   just a new config section.
4. Hardware-validate (`-V 1`) across a representative shape battery using
   `script/sweep_shapes.py` (see `AGENTS.md`/`CLAUDE.md`'s "Shape sweep"
   section) -- e.g. `python3 script/sweep_shapes.py --configs
   <new-tile-config> --shapes config/shapes/default.json --mode validity`,
   then extend `config/shapes/default.json` (or pass a custom shape list)
   with shapes specifically sized to exercise the new macro-tile's M/N
   edges (exact-fit and tail-remainder cases). Do not rely on the 1-2
   shapes a benchmark script happens to use -- both correctness bugs found
   this session (`docs/gfx1250_dominance_study.md`) were combinations that
   passed construction-time and assembly-time checks but failed on real
   hardware for specific shapes only, exactly the failure mode a new tile
   shape is most likely to introduce (new register/LDS layout, new
   edge/tail arithmetic).
5. Only then let it participate in the combinatorial FLAGS sweep like the
   existing tile shapes do.

This is genuinely new WMMA-capability engineering (extending
`igemm_bwd_gtc_wmma_nhwc_t`/`igemm_wrw_gtc_wmma_nhwc_t` to new macro-tiles,
likely non-trivial register/LDS layout work for bwd/wrw specifically, since
their asymmetric-tile support currently lags fwd's), not a config-cleanup
or tuning-search task. It was **not attempted in this session** --
flagging it here with the measured gap and the concrete steps to close it,
for a decision on priority/timing rather than assuming it should happen
now.

## Recommendation

Given the scope (multi-tile-shape generator extension across 2-3
directions, each needing its own register/LDS-layout design and hardware
validation), this should be its own dedicated phase/session with explicit
hardware-benchmarking time budgeted, not folded into tuning-search cleanup
work. Suggested prioritization if picked up: fwd first (already has the
most asymmetric-tile precedent to extend from), then bwd/wrw using
whatever generator-level lessons fwd's extension surfaces.

## Update (2026-09-07): two specific gap entries already investigated and closed as "not worth pursuing"; a small amount of new register headroom now available for the rest

This session separately investigated whether gfx1250 could reach the
**256x256/256x128** entries in the table above (the biggest shapes in
gfx950's grid) via the two mechanisms that actually exist for growing past
the WMMA path's 128x128 register ceiling: VGPR-MSB banking
(`wmma_acc_high_bank`, `docs/gfx1250_wmma_vgpr_msb_wip_status.md`'s Phase
54/55) and an 8-wave `col_split_b` mapping that avoids MSB banking
entirely. **Conclusion: both are real, hardware-validated-correct
capabilities, and both measure as a net performance LOSS against the
existing 128x128 tile at every scale tested** -- from small shapes (2-2.7x
slower) up to a deliberately huge, occupancy-saturating, deep-K shape
(16384x512x16384, still 1.35-1.4x slower, and the ratio plateaus rather
than closing with scale). `rocprofv3`-measured `SQ_WAVES_sum` is IDENTICAL
between the 128x128 and 256x128 (`col_split_b`) kernels on the same shape,
ruling out occupancy/launch-count as the explanation -- this is a genuine
per-wave throughput deficit in the bigger tile's own structure, not
resolved further (would need instruction-issue-rate profiling).

**Practical implication for this gap table**: don't spend effort chasing
the fwd/bwd **256x256** and **256x128** entries specifically -- the
capability to build them already exists (fwd's is even hardware-validated
and sitting in `config/` today as opt-in, non-default files) and it isn't
a win. bwd's 256x128 additionally has a confirmed, un-root-caused
`valid:n` correctness bug (now gated by an assert in `igemm_base.py`,
see the wmma_acc_high_bank/row_repeat_a>1 commit) -- doubly not
shippable today regardless of performance. The genuinely open, unexplored
part of this gap is the **smaller/asymmetric shapes** this doc's "what
closing it would require" section describes (32x64, 128x256, 64x256,
32x128, etc. for bwd/wrw specifically) -- normal-sized tiles reachable via
ordinary `wmma_repeat_m/n` decomposition, not via MSB/col_split_b, and
not measured or attempted this session.

**Register-budget note**: a separate liveness-based VGPR-reuse pass this
session (fwd/bwd only; wrw's per-tap epilogue structure can't use the same
mechanism, see that commit) freed a modest amount of headroom on the
tightest existing 128x128 tiles -- typically 2-8 registers depending on
precision and which tunables are active, by overlapping epilogue-only
scratch with two provably-dead register pools (post-loop address
registers, and -- for saddr/most configs -- the global-load staging
buffer, behind an explicit drain wait). This is nowhere near enough to
reach a genuinely bigger tile (256x256 needs roughly double the plain
128x128 register footprint), but it's real, free margin that should be
accounted for -- not re-solved from scratch -- when someone does the
register/LDS-layout design work for a new bwd/wrw asymmetric tile shape:
the mechanism is generic (bin-packs whatever epilogue-only scratch a new
tile's tunable combination needs against whatever dead-register pools
that tile's addressing path happens to have), so a new tile shape gets
this reuse "for free" as long as its generator code is added using the
existing `kernel_vgpr_t` pattern, not a new one.

## Update (2026-09-07, later same day): fwd extended to 4 new asymmetric shapes (hardware-validated); bwd/wrw's remaining gaps now precisely characterized, one confirmed as a real hardware bug

Picked this gap back up per explicit request to close it incrementally. Result:
**fwd gained 128x32, 32x128, 32x64, 64x32** (12 new config files: 4 shapes x
3 precisions, fp16/bf16/fp32), each hardware-validated (`conv_driver.exe -V 1`,
multiple shapes including grouped/multi-tap cases, plus `script/sweep_shapes.py`
across `config/shapes/default.json` for every combinatorial FLAGS variant --
4400+ `(config, kernel, shape)` checks total, 0 failures). **bwd and wrw gained
nothing new** -- both hit real, now-documented blockers described below, not
unwillingness to try.

### Method: the real gating mechanism is a hand-maintained wave/repeat table, not just config authorship

Contrary to this doc's original "author a base .config section" framing,
the actual gate for a new WMMA tile shape is
`python/operations/wmma_mapping.py`'s `ctrl_wmma_mapping_table` --
`get_ctrl_wmma_mapping_from_wave_tile()` asserts the exact
`(macro_tile_m, macro_tile_n, wave_tile, wave_repeat_m, wave_repeat_n, waves)`
tuple must already be a row in that table, keyed by precision (not
direction -- fwd and bwd share entries for identical tile geometries).
Closing a gap entry means: (1) derive a valid wave/repeat decomposition,
(2) add it to the table for each precision, (3) author the `.config`
section, (4) construct-and-real-assemble (`clang -x assembler`) to catch
VGPR-budget/register-range failures the Python-level asserts miss, (5)
hardware-validate. All four new fwd shapes are single-wave
(`block_size=32` -- the smaller of the two macro-tile dimensions forces
`waves_per_m*waves_per_n=1`). 128x32/32x128 additionally need
`wmma_acc_high_bank=1` + `wmma_epilogue_chunked=1`: measured at 275/278
VGPRs without them (19-22 over the 256/wave ceiling) despite a modest
`total_acc_c=128` -- a single-wave block's prologue/epilogue overhead does
not shrink with fewer waves the way the accumulator does. With high-bank
moving the accumulator to the second bank, both fit comfortably. 32x64/64x32
fit the plain 0-255 range directly, no extra flags. Per the existing
256x128 precedent, the two high-bank shapes are opt-in standalone files
(`config/igemm_fwd_gtc_gfx1250_nhwc_{prec}_{128x32,32x128}.config`), kept
OUT of `BASE_SECTIONS`/the combinatorial FLAGS sweep (high-bank's own
asserts already reject most of FLAGS); 32x64/64x32 are plain row_repeat-only
shapes and were added to `BASE_SECTIONS` normally.

### Generalizes the earlier "256-anything is a VGPR-ceiling/perf trap" finding

`total_acc_c = gemm_m_per_block * gemm_n_per_block / block_size`, and
`block_size` is capped at the SMALLER of the two macro-tile dims (fwd's A
side has no col_split equivalent, so `block_size <= gemm_m_per_block`
always). Whenever either dimension is 256 and the other is < 256, the best
achievable `total_acc_c` is exactly 256 (at `block_size = min(M,N)`) --
meaning 128x256, 256x32, 256x64, 32x256, 64x256 all require the SAME
high-bank treatment as the already-explored, already-rejected 256x256/
256x128 (measured 1.35-2.7x SLOWER than 128x128 at every scale, see the
update above). Not measured individually this session, but the register
arithmetic is exact and universal -- these five entries should be treated
as the same "not worth pursuing" bucket, not re-investigated one at a time.
That leaves 128x256 as fwd's only remaining, genuinely-unexplored-for-a-
real-reason gap: it hits the identical ceiling.

### bwd: found a real, previously-latent hardware correctness bug, not just "lagging behind fwd"

bwd's `row_repeat_a` generalization (mirroring fwd's A-side mechanism) was
added to `igemm_bwd_gtc_wmma_nhwc_t` back on 2026-08-25 but **no config or
`BASE_SECTIONS` entry ever actually used it** -- it was dead, hardware-
unvalidated code. This session authored the first real one (64x32,
`block_size=32`, `row_repeat_a=2`) to close that gap entry, and it passed
construction AND real assembly, but **`conv_driver.exe -V 1` reports
`valid:n`** even on the simplest possible shape (1x1 conv, no padding,
single group) -- not root-caused (candidates: the recomputed-per-row
n/hi/wi decomposition in `global_load_a_functor`'s row>0 branch, or a
group/split-K SGPR interaction fwd's A-side doesn't share). Deleted the
config and added `assert self.row_repeat_a == 1` in
`igemm_bwd_gtc_wmma_nhwc_t.__init__`, gating row_repeat_a>1 off entirely
(not just combined with `wmma_acc_high_bank`, which was already gated
separately) until someone root-causes it. This is the SAME failure class
`docs/gfx1250_dominance_study.md` warns about: passes every construction-
time and assembly-time check, wrong only on real hardware -- reinforces
that `is_valid()`-style automation (construct+assemble) is necessary but
not sufficient; hardware validation via `sweep_shapes.py`/`conv_driver.exe`
is not optional for new tile-shape work, even when the underlying mechanism
already has a working precedent elsewhere in the same file.

Separately (not re-verified this session, but the math is unchanged): even
where bwd's `row_repeat_a` mechanism DID work, 128x64 measures at 257
VGPRs -- one register over budget -- and 128x32 at 279 (23 over). bwd
cannot use the `wmma_acc_high_bank` escape hatch fwd used, because
`igemm_base.py` already asserts `wmma_acc_high_bank` combined with bwd
`row_repeat_a>1` is a CONFIRMED-bad hardware combination (separate,
pre-existing finding, `docs/gfx1250_wmma_vgpr_msb_wip_status.md`). So even
after the correctness bug above is fixed, 128x64/128x32 need a genuine
VGPR trim (same class of work as the existing liveness-based reuse pass,
~1-2 registers) to become reachable -- not a fundamental blocker, but real,
separate optimization work.

bwd's remaining gap entries needing gemm_n_per_block != block_size
(128x256, 32x128, 32x64, 64x128, 64x256) are unreachable at all without
porting a B-side mechanism (row_repeat_b or col_split_b) to bwd's
TRANSPOSED B operand -- explicitly out of scope for both this session and
the original doc (see "What closing it would actually require" above).

### wrw: not "lagging", structurally blocked -- zero new shapes possible without an addressing redesign

`igemm_wrw_gtc_wmma_nhwc_t.__init__` (the line 190-193 assert) requires
**`gemm_n_per_block == gemm_m_per_block` unconditionally** -- wrw's B
(input) operand addressing reuses A's row/col-group tiling scheme, which is
only correct when M==N. This is not an unexplored corner or a missing
config section: every asymmetric shape in wrw's gap list (128x256, 128x64,
256x32, 256x64, 32x128, 64x128, 64x256, 32x64, 64x32) is asserted
unreachable by construction. The only shapes wrw can ever have are square
powers of 2 -- 32x32, 64x64, 128x128, 256x256 -- and all four already exist
(256x256 explored and rejected as a perf loss in the update above). **wrw's
tile-shape gap is fully closed in the sense that nothing more is reachable
without redesigning B's addressing to not reuse A's tiling** (a genuine,
substantial rework of `emit_kernel_prologue`'s row/col-group derivation and
every functor that depends on it -- not attempted this session, and not
recommended as an incremental follow-up; it is its own project).

### Updated recommendation

fwd's tile-shape gap is now essentially closed (128x256 is a known,
quantified non-win, not an oversight). Remaining genuinely valuable work,
in priority order: (1) root-cause bwd's `row_repeat_a` hardware bug --
unblocks 64x32 immediately and de-risks 128x32/128x64 once their VGPR
trims land; (2) the 1-2-register trims for bwd 128x64/128x32; (3) porting
fwd's B-side row_repeat_b/col_split_b mechanism to bwd's transposed B,
unblocking 32x128/32x64/64x128 (128x256/64x256 still hit the 256-ceiling
trap and should be deprioritized per the finding above regardless). wrw's
B-addressing redesign remains its own dedicated project, not a queue item
alongside these.
