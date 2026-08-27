# hipconv deep dive: main-loop scheduling, occupancy, and wrw/split-K strategy

This is a follow-up pass over `/home/sgundabo/hipconv`, focused specifically on the
questions the first pass (`docs/gfx1250_external_research_findings.md`, "hipconv
findings" section) didn't cover in depth: main-loop pipelining/scheduling, wrw-style
occupancy strategy, barrier/sync tricks, non-atomic epilogue mechanics, and gfx1250-
specific tuning constants. Read that file first for TDM descriptor bit-packing, K-tail
handling, single-issuer-wave timing, DMA-engine parity, and hardware LDS padding --
none of that is repeated here.

Files examined this pass, all under `hipconv/src/arch/cdna5/` unless noted:
- `direct/kernel.hpp`, `direct/config.hpp` -- the fprop/dgrad ("direct") conv kernel.
- `grouped/grouped_multi_g/kernel.hpp`, `grouped/grouped_multi_g/config.hpp` -- grouped
  conv2d fprop for small group sizes (G in {4,8,16,32}).
- `grouped/grouped_multi_g_wgrad/kernel.hpp`, `.../config_table.hpp`,
  `grouped/reduction.{hpp,cpp}` -- the gfx1250 wgrad (weight-gradient) kernel and its
  split-K reduction pass. **This is the single most relevant file for MISA's wrw gap.**
- `hipconv/src/bunnies.hpp`, `arch/cdna5/bunnies_mi400.hpp` -- shared async-copy/matrix
  primitives used by all of the above.
- For contrast: `docs/algorithms/direct/direct-wgrad*.md` and the corresponding
  `arch/cdna4/direct/direct_wgrad/*` kernel -- NOT gfx1250 code (cdna4 = MI350/MI355X,
  MFMA, WAVE64), but its design is directly relevant and its own docs claim it
  generalizes to WMMA/WAVE32; flagged clearly below as unconfirmed-for-gfx1250.

## 1. Main-loop pipelining depth and scheduling

hipconv's actual shipped gfx1250 kernels split into two very different scheduling
styles depending on whether waves must cooperate on shared LDS data or not.

**`direct/kernel.hpp` (fprop/dgrad): role-specialized waves + staggered barriers.**
This kernel uses `tiles_k=2 x tiles_j=4 = 8` waves per workgroup and gives each wave a
*fixed role* for the whole kernel via `wave_id_k`/`wave_id_j`/`wave_rank`
(`wave_rank = wave_id/4`), not just "wave 0 issues, the rest wait":
- `wave_id_k==0` waves stage the input tensor into LDS; `wave_id_k==1` waves stage the
  weights (comment calls these "CU0"/"CU1", legacy naming, not literal compute units).
- Within a k-group, `wave_id_j==0` prefetches the *next* c-tile's TDM load mid-loop,
  while `wave_id_j==1` handles the H-ring buffer advance -- two different loader
  sub-roles, not a simple parity split.
- A **staggered/split barrier** opens the main loop (`direct/kernel.hpp:443-448`):
  `if (wave_rank == 1) s_barrier();` -- only half the waves barrier here, the other half
  fall through, deliberately putting the two wave groups out of phase so one group's
  load issuance overlaps the other's compute, mirroring (in spirit) the CDNA4 ping-pong
  design described in section 2.
- TDM loads are double-buffered (`num_buf=2`, `buf_in`/`buf_wei` toggled each
  iteration) and the *next* c-tile / next-(r,s) weight tile is prefetched from the
  *middle* of the current tile's compute (`part_c == c_parts/2`, `kernel.hpp:544-553`),
  not from the top or bottom of the loop body -- explicitly to land the load without
  stalling either the current WMMA sequence or the next one.
- `__builtin_amdgcn_s_setprio(1)` / `s_setprio(0)` bracket every `mma()` call
  (`kernel.hpp:557,559`), elevating instruction priority for the duration of the WMMA
  sequence, and `sched_barrier(0)` appears immediately around every `s_barrier()` to
  stop the compiler from hoisting/sinking instructions across it.
- Net pipeline depth: effectively a **2-deep, role-partitioned software pipeline**
  (current tile compute vs. next tile's TDM load-in-flight), similar order of depth to
  MISA's LDS double-buffering, but with *fixed wave roles* driving the schedule instead
  of every wave running the identical body.

**`grouped_multi_g/kernel.hpp` and `grouped_multi_g_wgrad/kernel.hpp`: no barriers at
all, ring-buffered TDM prefetch instead.** These two kernels give each wave a fully
independent unit of work (one wave computes one output tile / one wgrad tile with no
cross-wave LDS sharing), so there is **no `s_barrier` anywhere in either file** -- all
synchronization is `s_wait_tensorcnt` against that wave's own private TDM ring. The
depth is controlled by `cfg.prefetch_depth` (PF, config field, not hardcoded):
`PF=2` is a classic double buffer; `PF>2` keeps `PF-1` loads in flight via
`s_wait_tensorcnt(PF-2)` instead of draining to 0 every iteration
(`grouped_multi_g_wgrad/kernel.hpp:287`, `grouped_multi_g/kernel.hpp:981` and
similar) -- the same "don't over-drain" idea the first pass already flagged for the
depthwise kernel, confirmed here as a general pattern used by every cdna5 kernel that
has a K-reduction main loop, with the ring depth exposed as a tunable rather than fixed.

**Did hipconv find a working version of MISA's regressed `main_loop_interleave`?**
Not really as the same trick. MISA's `ctrl.interleave`
(`python/operations/wmma_main_loop.py:120`, used from
`python/igemm/igemm_fwd_gtc_wmma_nhwc.py`) interleaves the wait+store of one K-substep
with the *next* substep's compute inside a single wave's own instruction stream, and
regressed. hipconv's closest analogue in `direct/kernel.hpp` gets its overlap a
different way: by giving *different waves* different roles (loader vs. ring-updater)
so the loader wave's TDM issuance runs concurrently with the compute wave's WMMA
issuance on separate hardware queues, rather than by manually reordering wait/store
pairs within one wave's instruction stream. That's a structurally different lever
(cross-wave role division vs. intra-wave instruction reordering) and isn't a direct
counter-example to MISA's regression -- it doesn't prove MISA's specific interleave
would work if retried, but it does suggest that if MISA revisits this, splitting
*which wave issues which load* (as gfx1250's dual-DMA-engine parity trick from the
first pass already suggests) may be a more promising axis than reordering instructions
within a single wave's stream.

## 2. wrw / backward-weight-style occupancy strategy (highest priority: MISA's biggest gap)

`grouped_multi_g_wgrad/kernel.hpp` is hipconv's actual shipped gfx1250 wgrad kernel and
uses **two independent techniques**, neither of which is atomic contention tuning:

**(a) Block-diagonal WMMA channel packing to fill a tiny M/N tile.** For a grouped
conv with group size G in {4, 8, 16}, one 16x16 WMMA output tile is far wider than one
group's own (G x G) weight-gradient block. Instead of padding a single group up to 16
channels and wasting the rest of the tile (4x-16x wasted MACs for G=4), hipconv packs
`GPW = 16/G` *independent groups* diagonally into the same 16x16 tile
(`grouped_multi_g_wgrad/kernel.hpp:70-73`, comment block at the top of the file). The
WMMA computes all `GPW` groups' partial products in one instruction; the epilogue then
keeps only the diagonal blocks (`(c_off >> LOGG) != (mn >> LOGG)` guard, line 346) and
discards the off-diagonal cross terms, which are never written. G=32 is the inverse
case: it splits one group across `NTILE=2` sub-tiles per axis instead. This is a
**register-tile-level fix for small-M/N occupancy**, entirely orthogonal to grid-level
split-K -- it uses the wasted lanes of the WMMA op itself rather than trying to fill
the CU with more waves.

**(b) Split-K via a separate reduction kernel, not atomics, when the config calls for
it.** The launch computes `num_partitions = q_tiles * N` (grid.y * grid.z) from the
tiling itself, not from a search over split factors. Given that, the kernel then
statically picks one of three epilogue strategies at compile time
(`launch_impl`, `grouped_multi_g_wgrad/kernel.hpp:384-445`):
  - `num_partitions == 1`: plain (non-atomic) store.
  - `multi_contrib && !cfg.split_k`: atomic-accumulate into the real output
    (`NEEDS_ATOMIC=true`, `bn::cascade_atomic_add_f32`) -- this is the path that uses
    `TH_ATOMIC_CASCADE_RT`, already flagged by the first pass.
  - `multi_contrib && cfg.split_k`: **each partition writes to a private slice of a
    `workspace` buffer (`PARTITIONED=true`, no atomics at all in the main kernel), then
    a second, tiny, dead-simple reduction kernel
    (`grouped/reduction.cpp:conv2d_grouped_multi_g_wgrad_reduce_cdna5`) sums the
    partitions elementwise** (`dW[e] = sum_p partials[p][e]`, one thread per output
    element, no cross-partition atomics, no ordering games). This sidesteps the
    cascading-atomic hang/contention question entirely for the configs where it's
    picked.
  Which of the two multi-contributor strategies gets used is a **static per-shape
  config-table choice**, not an online search: `config_table.hpp` lists the exact same
  `(group_size, waves_per_wg)` pair twice, once without `split_k` (defaults to
  atomic-cascade) and once with `.split_k = true` explicitly, for every `group_size` in
  {4,8,16,32} and `waves_per_wg` in {4,8} -- i.e. hipconv ships *both* the atomic and
  the reduction-kernel variant for the same shape and leans on its generic
  config-ranking mechanism (already covered by the first pass) to pick between them,
  rather than deriving the split factor from a runtime heuristic or search the way
  MISA's ternary search over `gemm_k_global_split` counts does.

**Important caveat -- the CDNA4 (not gfx1250) ping-pong wgrad design.**
`docs/algorithms/direct/direct-wgrad.md` and `direct-wgrad-main-loop-blocks.md` (both
in `hipconv/docs/`, not under `src/arch/cdna5/`) describe a much more aggressive
8-wave "ping-pong" wgrad kernel: two 4-wave groups run staggered memory/compute phases
(`s_barrier` + priority `set_prio(0)`/`set_prio(1)`) so one wave-group's global-load
phase overlaps the other's MFMA compute phase every half-step, with a carefully
reasoned 2-4 row prefetch distance chosen against measured HBM latency (882 cycles on
MI355X) vs. the 576-cycle compute half-step. **This design is implemented under
`hipconv/src/arch/cdna4/direct/direct_wgrad/` -- i.e. it targets MI350/MI355X (CDNA4,
MFMA, WAVE64), not gfx1250.** The doc explicitly claims "with minor adjustments, the
algorithm maps to WAVE32 and MI450 WMMA," but a repo-wide search found **no
`arch/cdna5` kernel that actually implements this ping-pong schedule** -- the real
gfx1250 wgrad kernel (`grouped_multi_g_wgrad`) instead uses the much simpler
independent-per-wave design in (a)/(b) above, with zero `s_barrier`s. Treat the
ping-pong wgrad doc as a **credible but unconfirmed-for-gfx1250 design sketch** (same
epistemic status as the first pass's `TH_ATOMIC_CASCADE_RT` note) -- worth reading for
ideas (the explicit HBM-latency-vs-compute-phase arithmetic in particular is a good
template for reasoning about MISA's own prefetch depth), but do not assume it runs on
gfx1250 hardware as described; it may hit the same kind of WAVE32/WMMA-specific gap
that made hipconv build a separate, simpler cdna5 kernel instead of reusing it.

## 3. Barrier and synchronization strategy

- `direct/kernel.hpp` uses the **staggered/split barrier** already described in
  section 1 (`wave_rank`-conditional `s_barrier()` at loop entry/exit,
  `kernel.hpp:443-448, 572-576`) plus a `s_barrier()`/`sched_barrier(0)` pair around
  *every* WMMA-adjacent LDS hand-off inside the c-loop (`kernel.hpp:469, 554-555, 564-
  565`) -- this is a much higher barrier density than a single per-iteration
  workgroup barrier, but each one is immediately followed by `sched_barrier(0)`, which
  is a *compiler scheduling fence* (prevents instruction reordering across that point),
  not a second hardware barrier -- so the actual hardware `s_barrier` count per K-tile
  iteration is 2 (one per r/s sub-step transition), not inflated further.
- No split-phase "signal early / wait late" pattern (the FlyDSL
  `s_barrier_signal`/`s_barrier_wait` decoupling reported by the first pass) appears
  anywhere in `arch/cdna5/` -- hipconv's cdna5 kernels use the plain, non-split
  `__builtin_amdgcn_s_barrier()` throughout. The first pass's FlyDSL finding remains
  the only source for that specific technique.
- No `disable_xdl_arb_stall`/`SCHED_MODE` `S_SETREG_B32` write appears anywhere in
  `arch/cdna5/` either. Instead, hipconv uses **`s_setprio(1)` immediately before
  `mma()` and `s_setprio(0)` immediately after** (`direct/kernel.hpp:557,559`) -- a
  standard, ISA-documented instruction-priority bump, not a scheduler-mode register
  hack. This is a lower-risk mechanism than `disable_xdl_arb_stall` for achieving a
  similar goal (keep the WMMA sequence from losing issue slots to a neighboring wave
  mid-sequence) and is trivially encodable in hand-written `.s`.
- `grouped_multi_g_wgrad/kernel.hpp:309` uses a single `__builtin_amdgcn_sched_barrier(0x10)`
  right before the `mma()` call inside its per-row loop -- a narrower scheduling fence
  (mask `0x10`, restricting what the compiler may reorder across that specific point)
  rather than the blanket `sched_barrier(0)` used in `direct/kernel.hpp`. Exact bit
  semantics weren't verified against the ISA doc this pass; flagged for anyone
  adapting this literally.

## 4. Epilogue strategy for the non-atomic (dense conv fwd/dgrad) case

`direct/kernel.hpp`'s epilogue (`kernel.hpp:592-672`) is a three-step pattern:
1. Cast each WMMA accumulator sub-tile to the output type and store it into a padded
   LDS scratch buffer with `ds_store_b128` (`store_tile<arch::ds_store_b128>`,
   line 609) -- a per-lane, but already-vectorized (16B/lane), store into LDS. No
   scalar-then-reshuffle step here because the WMMA C-tile's lane layout is already
   contiguous enough for a direct 128-bit store (the same lane-layout observation the
   first pass reported independently from FlyDSL and from MISA's own
   `coalescing_store_wmma.py` docstring).
2. `__syncthreads()`.
3. **Store straight from LDS to global with an async, per-lane instruction that never
   passes through VGPRs at all**: `bn::global_store_async_from_lds<arch, store_cfg>`
   (line 661), which lowers to `__builtin_amdgcn_global_store_async_from_lds_b128`
   (`bunnies_mi400.hpp:473-508`) -- a *different* primitive from the TDM
   `tensor_store_from_lds` the first pass already covered: this is a per-lane-addressed
   async global store issued directly against an LDS source address, no whole-tile
   descriptor involved, but still skips the VGPR round-trip a plain `buffer_store`
   epilogue would need. Vector width is chosen once at compile time via
   `get_bytes_per_lane()` (line 643-652): 16B if the output's fastest-varying extent is
   16B-aligned, falling back to 8B, 4B, then 1B otherwise.
4. Boundary handling picks between two call sites at compile+runtime, not EXEC masks:
   `global_store_async_from_lds` (unchecked) when the whole tile is provably in-bounds
   (`out_view.in_bounds(...)` checked once, line 658), else
   `global_store_async_from_lds_checked`, which takes an `in_bounds(i,j)` lambda
   evaluated per store round (line 666-671) and simply skips the instruction when out
   of bounds -- a per-round branch, not a per-lane EXEC mask, but functionally similar
   cost/shape to MISA's `wmma_m_tail`/`wmma_n_tail` guards.

This differs from MISA's `python/operations/coalescing_store_wmma.py` in one concrete
way: MISA's LDS reshuffle stages scalar per-lane values into LDS, then *reads them back
into VGPRs* as `dwordx4` before issuing the final `buffer_store_dwordx4`. hipconv's
step 3 skips that read-back -- the async store instruction sources directly from LDS.
If MISA's WMMA C-tile lane layout is already contiguous enough to `ds_store_b128`
straight into LDS (worth re-checking, since the first pass already flagged this same
question from a different angle), the VGPR read-back stage may be pure overhead that
could be dropped the same way, provided gfx1250 exposes an equivalent per-lane
async-store-from-LDS builtin (unconfirmed this pass -- only found via hipconv's
`bunnies_mi400.hpp` wrapper, not independently verified against the ISA doc the way
TDM was in the first pass).

## 5. gfx1250/cdna5-specific tuning constants

From `direct/config.hpp` (fprop/dgrad) defaults: `tile_size_h = tile_size_w = 16`,
`tile_size_k = 256`, `tile_size_c = 128`, WMMA shape `16x16x32`, `tiles_j = 4`,
`tiles_k = 2` => **8 waves per workgroup** (`cfg.tiles() = 32`), with
`__launch_bounds__(cfg.tiles()*wave_size, 2)` (`kernel.hpp:676`) explicitly requesting
a minimum occupancy of **2 workgroups per CU** (16 waves/CU) from the compiler, not
left to register-pressure inference alone. LDS layouts add explicit dword padding
(`tile_size_c_pad_amount`, `tile_size_k_pad_amount`, both asserted to specific values:
4 and "8 dwords total across `tiles_k`") with an inline comment explaining the
bank-conflict reasoning (double-buffer interleaving pushes half a buffer into a
different "vertical bank" so reads of each half don't conflict) -- a from-first-
-principles derivation of a padding constant, not a swept/searched value.

From `grouped_multi_g/config.hpp` and `grouped_multi_g_wgrad/config_table.hpp`: no
fixed wave count -- `waves_per_wg` is swept over `{1, 2, 4, 8}` per `group_size` in the
config table (see section 2), and `prefetch_depth` (PF) is a free tunable, not a fixed
architectural constant, with an explicit static assert `PF >= 2`. The block-diagonal
channel-packing factor `GPW = 16/G` (`gpw_of()`) is derived purely from the WMMA tile
width (16) and the group size, i.e. it's a closed-form function of the WMMA ISA shape,
not a swept parameter.

## Recommendations for MISA

Ordered by priority; wrw/occupancy first since that's MISA's largest measured gap
(1.1x-26x slower than MIOpen on small/non-exact shapes per the background above).

1. **(High value, high risk, wrw) Consider a split-K-via-reduction-kernel path as an
   alternative to `gemm_k_global_split`'s atomic accumulate**, at least for the
   worst-case small/non-exact wrw shapes. MISA's current mechanism
   (`python/igemm/igemm_wrw_gtc_wmma_nhwc.py`, `ctrl_coalescing_store_wmma.gemm_k_global_split`
   / `atomic_cascade` wired at `igemm_wrw_gtc_wmma_nhwc.py:159-183`) always accumulates
   via atomics into the real output. hipconv's alternative -- write each K-split
   partition to a private workspace slice, then run a second, trivial elementwise-sum
   kernel -- removes atomic contention entirely for the configs it's used on. This is a
   bigger change than a tunable flag: it needs a workspace allocation path (MISA's
   driver would need to plumb a scratch buffer size, similar to what
   `get_workspace_size()` does in hipconv's `Grouped_Multi_G_WgradConvKernel`) and a
   new tiny reduction kernel. Effort: medium-high (new codegen path + driver plumbing
   + workspace sizing); risk: medium (reduction kernel itself is trivial and easy to
   validate independently of the main GEMM). Given wrw is MISA's worst gap and the
   26x-slower cases are exactly the small-M/N, atomic-heavy ones, this is the highest-
   value item on this list.

2. **(High value, medium risk, wrw) Block-diagonal WMMA channel packing for small
   GEMM_M/N.** MISA's wrw weak spot is specifically "small, non-exact shapes." If any
   of those small-shape cases have a GEMM_M or GEMM_N that's a fraction of the WMMA
   tile width (16 for gfx1250), hipconv's `gpw_of()`/`ntile_of()` packing
   (`grouped_multi_g_wgrad/kernel.hpp:70-78`, block-diagonal-keep epilogue guard at
   line 346) is a concrete template for using the *whole* WMMA tile instead of padding
   a narrow M/N up and wasting lanes. This would touch MISA's wrw tile-size selection
   logic and epilogue masking (`wmma_m_tail`/`wmma_n_tail` guards) rather than the main
   loop. Effort: medium (new packing/unpacking logic in the epilogue, plus a guard
   condition analogous to hipconv's `(c_off >> LOGG) != (mn >> LOGG)` check); risk:
   medium -- only applies where MISA's tail shapes are narrower than the WMMA tile
   width, needs a shape audit first to confirm it's the right lever for MISA's actual
   26x-slow cases (which may be narrow-K, not narrow-M/N -- verify before investing).

3. **(Medium value, low-medium risk) Direct async-store-from-LDS epilogue, skipping
   the VGPR read-back.** MISA's `python/operations/coalescing_store_wmma.py` reshuffles
   through LDS and then reads back into VGPRs before the final `buffer_store_dwordx4`.
   hipconv's `direct/kernel.hpp` epilogue (section 4 above) stores directly from LDS to
   global via `global_store_async_from_lds_b128`, skipping that read-back. Effort: low-
   medium (need to confirm gfx1250 exposes an equivalent per-lane async-store-from-LDS
   instruction beyond TDM's whole-tile `tensor_store_from_lds`, then swap the final
   stage of the existing coalescing-store codegen); risk: low if the instruction exists
   and behaves as hipconv's wrapper implies -- this is a narrow, mechanical change to
   an already-working path, not a new algorithm.

4. **(Medium value, low risk) `s_setprio(1)/(0)` bracketing around WMMA issue** as a
   cheaper alternative to trying `disable_xdl_arb_stall`. MISA doesn't use either
   today. `s_setprio` is a plain, ISA-documented instruction (no `SCHED_MODE` register
   write, no undocumented side effects to characterize) and hipconv brackets every
   `mma()` call with it in `direct/kernel.hpp:557-559`. Effort: very low (two
   instructions around the existing WMMA emission in
   `python/operations/wmma_main_loop.py`); risk: low -- easy to A/B on real hardware
   since it's a no-op if wrong (just an issue-priority hint).

5. **(Lower priority, exploratory) Role-specialized wave scheduling for the main
   loop**, i.e. giving fixed loader/compute roles to specific waves (as
   `direct/kernel.hpp`'s `wave_id_k`/`wave_id_j` split does) instead of every wave
   running an identical body, as a different axis to retry overlap on given that
   MISA's own `main_loop_interleave` (intra-wave instruction reordering,
   `python/operations/wmma_main_loop.py`'s `ctrl.interleave`) regressed. This is a much
   larger restructuring of the main-loop codegen than the other items here (effort:
   high; risk: high -- no direct evidence it helps on gfx1250 specifically, since even
   hipconv's own most-aggressive version of this idea, the CDNA4 ping-pong wgrad
   design in section 2, is unconfirmed to run on gfx1250 hardware at all). Only worth
   pursuing after items 1-4 land and are measured.
