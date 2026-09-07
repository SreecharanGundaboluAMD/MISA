# gfx950 vs gfx1250 Tile-Shape Coverage Gap

## Current state (measured directly, 2026-09-07)

| Direction | gfx950 distinct (M x N) tile shapes | gfx1250 distinct (M x N) tile shapes |
|---|---|---|
| fwd | **14**: 128x128, 128x256, 128x32, 128x64, 256x128, 256x32, 256x64, 32x128, 32x256, 32x64, 64x128, 64x256, 64x32, 64x64 | **5**: 128x128, 128x64, 256x128, 64x128, 64x64 |
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
