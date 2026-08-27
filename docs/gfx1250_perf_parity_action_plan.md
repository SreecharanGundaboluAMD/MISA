# Reaching (and beyond) parity on gfx1250: cross-source action plan (2026-08-27)

Synthesis of five independent investigations, each in its own doc:
- `docs/gfx1250_ck_deep_dive.md` — Composable Kernel
- `docs/gfx1250_hipconv_deep_dive.md` — hipconv (follow-up pass)
- `docs/gfx1250_flydsl_deep_dive.md` — FlyDSL (follow-up pass)
- `docs/gfx1250_rocke_deep_dive.md` — rocKE
- `docs/gfx1250_rocprof_profiling.md` — hardware-counter profiling of MISA's own kernels

Read those for full detail, code references, and caveats. This doc pulls out what's
**cross-validated by more than one independent source** (the strongest signal — different
codebases converging on the same technique means it's a real, general property of this
hardware/problem class, not one project's idiosyncrasy) and orders everything by expected
value against MISA's actual measured gap (`docs/gfx1250_vendor_benchmark_vs_miopen.md`):
wrw is 1.1x–26x slower than MIOpen (worst by far), fwd/bwd are close to parity on
well-fitting shapes but 1.5x–4x slower on tail/edge-case shapes.

## Tier 1 — cross-validated by 2+ independent sources, try these first

### 1. `s_setprio(1)`/`s_setprio(0)` bracketing around the WMMA-issue burst
**Confirmed as real, shipping code in both CK and hipconv, independently.** CK's v1 WMMA
pipeline (`blockwise_gemm_pipeline_wmmaops_v1.hpp`) and hipconv's `direct/kernel.hpp` both
raise instruction-issue priority immediately before a WMMA burst and drop it back
afterward — two unrelated codebases arrived at the identical two-instruction trick. It
directly targets the exact gap this project's own research already flagged as
untried (`disable_xdl_arb_stall`), but is cheaper and more universally safe: `s_setprio`
is a local, per-wave priority hint (no correctness risk, no interaction with other waves'
correctness), unlike `disable_xdl_arb_stall` which the ISA doc explicitly warns can
*hurt* well-occupied kernels (see Tier 2, item 5). **Action**: add
`s_setprio(1)`/`s_setprio(0)` around the WMMA issue sequence in
`python/operations/wmma_main_loop.py`'s `emit_wmma_tile()`. Cheapest, lowest-risk item in
this whole list — try it everywhere (fwd/bwd/wrw), not just wrw.

### 2. Packed 2-wide vector atomics for wrw's accumulate epilogue
**Confirmed as real, shipping code in both rocKE's wgrad conv and FlyDSL's gfx1250 MoE
split-K GEMM, independently.** MISA's `python/operations/coalescing_store_wmma.py`
(~line 206) emits one scalar `global_atomic_add_f32` per output element. Both other
sources instead build a 2-element vector from adjacent output columns and issue one
packed atomic, halving the atomic-op count for the same total data moved — directly
attacking the exact operation rocprof's profiling confirmed dominates wrw's non-WMMA
cycles. **Two independent sources doing the identical thing on the same problem shape
(reduction-heavy, atomic-accumulate epilogue) is unusually strong convergence for a
performance idea.** **Action**: verify gfx1250 has a packed 2x-fp32 (or 2x-bf16/fp16 with
a final fp32-promotion pass, trading precision for atomic count — needs an accuracy
check given wrw's numerics) atomic-add instruction via `llvm-mc`, the same verification
discipline used for TDM, then prototype in `coalescing_store_wmma.py`'s atomic path.
Highest-value item specifically for wrw.

### 3. Cross-check MISA's wrw split-K choice against alternative strategies before picking one
No single "right answer" emerged, but **three independent sources all point at MISA's
static, per-search split-K as leaving something on the table** — from three different
angles:
- CK computes its split count from a **closed-form occupancy formula**
  (`floor(hipOccupancyMaxActiveBlocksPerMultiprocessor-derived capacity / grid_size)`,
  capped by `ceil(gemmK/KPerBlock)` and an accuracy-driven max), not a search at all.
  Cheapest thing to try: port this formula as a *free, near-zero-cost second candidate*
  alongside MISA's existing ternary search in `driver/igemm_wrw_gtc_driver.h` — if they
  agree on the worst (26x-slower) shapes, the search isn't the problem; if they diverge,
  that's a direct, cheap-to-detect signal the search's cost model is wrong for tiny-grid
  wrw problems specifically.
- rocKE has a **working persistent-kernel/stream-K mechanism** (`helpers/streamk.py`,
  `helpers/persistent.py`) — a small, constant-size grid with an atomic tile-counter
  dynamically pulling work, rather than a fixed grid.z split decided before launch. Not
  wired to conv wgrad in rocKE today, but architecturally the most different (and
  potentially highest-ceiling) alternative to MISA's current design.
- hipconv's wgrad kernel offers **split-K via a separate, dedicated reduction kernel**
  (workspace partials + a second elementwise-sum pass) as an alternative to atomic
  accumulation, plus **block-diagonal channel packing across conv groups** to fill small
  WMMA tiles when the group count is high — a structurally different way to solve
  "GEMM_M/N too small to fill a tile."
**Action, in order of effort**: (a) the CK formula cross-check first (cheapest, most
diagnostic); (b) if that doesn't fully close the gap, prototype hipconv's
separate-reduction-kernel design next (avoids atomic contention entirely, moderate
effort); (c) rocKE's stream-K is the largest structural change and the longest-term bet,
worth prototyping only after (a)/(b) are exhausted.

## Tier 2 — single-source but concrete and well-grounded

4. **rocprof's own finding**: MISA's WMMA-busy-cycle-fraction is 2.3% (wrw) / 5.2% (fwd) of
   total active cycles — both far below rocKE's own stated "<40% is a red flag" heuristic.
   Atomic-address-conflict stalls measured at exactly zero, ruling out same-address atomic
   contention as wrw's bottleneck (a plausible a priori guess that hardware counters
   directly refute). This is the quantitative evidence *motivating* items 1-3 above, not a
   separate action — but it also means: **extend this profiling pass to bwd and to fwd's
   tail-handling paths** (mtail/ntail/tdm), which are unmeasured today and were flagged in
   the benchmark doc as similarly worse-than-baseline.
5. **`disable_xdl_arb_stall` (`SCHED_MODE` bit[2])** — untested by MISA, but the ISA doc's
   own text is unusually specific about when it helps ("beneficial primarily when a single
   wave is running on a SIMD") vs. hurts (blocks co-execution for well-occupied kernels).
   Combined with finding 4's occupancy data, this points specifically at wrw's low-occupancy
   split-K workgroups as the place to try it — test wrw in isolation, not fwd/bwd, and only
   after confirming per-SIMD wave occupancy is actually low there (not yet measured — see
   `gfx1250_rocprof_profiling.md`'s "Not yet done").
6. **Widen wrw's tile-shape space**: CK pairs small M/N tiles (64x64) with much larger
   KPerBlock (96/128/256, vs. MISA's 32/64) specifically so small-output-channel wrw
   problems get fewer, cheaper main-loop iterations without forcing a 128x128 tile that's
   mostly EXEC-masked. FlyDSL and rocKE separately show tile_m as low as 16/32/48/96 being
   legal, deliberately-supported shapes in this kernel family, not just MISA's current
   64/128 set. Both point the same direction: MISA's tunable space is coarser than what
   this hardware/problem class supports. Real codegen work (new `gemm_k_per_block`/
   `gemm_m_per_block` values need VGPR/LDS re-derivation, not a flag flip), medium-to-large
   effort, but the CK finding specifically targets wrw's exact failure mode.
7. **hipconv's staggered per-shard K-loop start phase** for `gemm_k_global_split` (rotate
   each split-K workgroup's first K-tile index by a per-shard offset, reducing
   simultaneous-burst memory contention at kernel launch) — small, low-risk, easy to A/B
   against the existing search.

## Tier 3 — exploratory / longer-term, lower immediate priority

8. Hardware transpose-load (`global_load_tr_b128_v8f16`, confirmed gfx1250-valid via CK's
   own audit, but no correct reference implementation exists anywhere for a 16x16x32-layout
   WMMA — would be new engineering, not a port).
9. Deeper main-loop pipelining (N-stage, beyond MISA's current double-buffer) gated by LDS
   headroom, per FlyDSL's `num_buffers` pattern — mechanical generalization, worth it once
   the higher-value wrw items above are exhausted.
10. Autotuning-with-build-cache infrastructure (FlyDSL's `conv3d_autotune.py` pattern) as a
    longer-term supplement to MISA's hand-curated `.config` files, specifically for shapes
    where static configs are demonstrably failing (wrw's tail cases).

## What NOT to do (confirmed dead ends / already-settled questions)

- CK's own WMMA main-loop pipeline is *shallower* than MISA's (2-stage, no LDS
  double-buffering at all) — nothing to adopt from CK on prefetch depth; if anything MISA
  is already ahead here.
- CK independently built and then **disabled** an instruction-interleave scheduler for its
  WMMA pipeline (`#if 0`'d out, "TODO: needs WMMA-specific rework") — a second, independent
  confirmation (alongside MISA's own measured regression from `main_loop_interleave`) that
  hand-scheduled interleaving is a genuinely hard problem for WMMA specifically, not
  something MISA got wrong in isolation. Don't re-attempt this without a fundamentally
  different scheduling model.
- MISA's epilogue design (LDS-reshuffle coalescing store) is mechanically the same as CK's
  default CShuffle epilogue — no adoptable idea there.
- MISA's fp32-always WMMA-output-accumulator design already sidesteps CK's own documented
  bf16-atomic correctness bug (0.35-0.50 error at small output-channel counts) — no action
  needed, already correct by construction.
