# CK gfx1250/WMMA deep dive: performance techniques MISA could learn from

Investigation of `/home/sgundabo/rocm-libraries/projects/composablekernel`'s gfx1250
(RDNA/WMMA) code paths — `gridwise_gemm_wmma_cshuffle_v3*`, `blockwise_gemm_pipeline_wmmaops_*`,
`device_grouped_conv_{fwd,bwd_weight}_*_wmma_cshuffle_v3.hpp`, `wmma_gemm.hpp`, `amd_wmma.hpp`,
and the gfx1250 WMMA instance-library headers under
`library/include/ck/library/tensor_operation_instance/gpu/{grouped_conv_fwd,grouped_conv_bwd_weight,grouped_gemm}/*wmma*`.
All `_xdl`-suffixed and CDNA-gated code was ignored as irrelevant (gfx1250 has no MFMA/XDL).
The pre-existing `/home/sgundabo/rocm-libraries/gfx1250_ck_audit.md` (build/correctness audit,
not performance) was read first for architectural context and is cited where it already answers
a question in this brief.

No code was modified in either repository. This is a research note only.

---

## 1. Tile/pipeline shape choices — CK ships a much wider instance library than a single tuned config

CK does not pick one MPerBlock/NPerBlock/KPerBlock for gfx1250 WMMA convs the way MISA's
codegen currently emits a fixed 128x128/64x64 (K=32, or 64 in the "k2x" variant). Its instance
library (e.g.
`library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp`)
precompiles dozens of `DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<...>` instantiations
spanning:

- MPerBlock/NPerBlock: 64, 128, 256, 512 (both square and rectangular, e.g. 128x64, 256x64, 512x256)
- **KPerBlock: 32, 48, 64, 96, 128, 256** — not just one or two fixed K-tile depths
- BlockSize (workgroup thread count): 64, 128, 256, 512, tracking MWave×NWave×32
- Both `BlockGemmPipelineVersion::v1` and `::v3` are represented across instances (see §2)

At runtime, MIOpen (or CK's own `DeviceOp::GetInstances()` + `IsSupportedArgument` heuristic
ranking) walks this instance set and picks the best-fitting one for the actual problem shape,
falling back through the list when a candidate's `CheckValidity`/`IsSupportedArgument` rejects
it. This is a fundamentally different strategy from MISA's "one hand-tuned kernel per config,
selected mostly by tunable-parameter search offline" — CK amortizes the tile-shape search into
build-time instantiation + host-time dispatch, so a given wrw problem with small GEMM_M/N can
be routed to a small 64x64 tile (with a correspondingly large or small K-tile) instead of being
forced through a 128x128 tile that's mostly EXEC-masked away.

**Why this matters for MISA's worst gap (wrw, small/non-exact shapes 26x slower):** MISA's
`gemm_m_per_block`/`gemm_n_per_block` space (128x128, 64x64 only) is coarser than CK's, and
MISA has no KPerBlock options between 32 and 64. wrw's GEMM_M (output channels K) is frequently
small (e.g. K=3, K=64) — CK's 64x64 tile with KPerBlock=96/128/256 gives it a way to keep the
K-tile large (fewer main-loop iterations, less loop overhead) while keeping M/N small to match
a tiny output-channel count, a combination MISA's current tunable space doesn't have.

## 2. Main-loop software pipelining — CK's WMMA pipeline has only 2 stages (shallower than MISA's own techniques), and its instruction-interleave scheduler is present but disabled

`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmma_selector.hpp` only offers
two pipeline versions for WMMA: `v1` (`blockwise_gemm_pipeline_wmmaops_v1.hpp`, a simple
load/sync/compute/store loop with no software global-prefetch depth beyond 1) and `v3`
(`blockwise_gemm_pipeline_wmmaops_v3.hpp`, "compute optimized": `PrefetchStages = 2`,
`PrefillStages = 1`, `GlobalBufferNum = 1`). Unlike CK's XDL pipeline family (which goes up to
v4/v5 with deeper prefetch and double-buffered LDS), **CK's WMMA path tops out at a
2-stage global prefetch with a single LDS buffer** — i.e. no LDS double-buffering at all
(`GlobalBufferNum = 1`; the loop body does `block_sync_lds()` → `RunWrite` (LDS store) →
`RunRead` (next global load) → compute → `block_sync_lds()` → `LocalLoad`, all against the same
LDS buffer). MISA's own LDS double-buffering is therefore already ahead of CK's WMMA v3 pipeline
on this specific axis.

More interesting: `blockwise_gemm_pipeline_wmmaops_v3.hpp`'s `HotLoopScheduler()` (lines
182–288) contains a **fully worked-out `__builtin_amdgcn_sched_group_barrier`-based
instruction-interleave schedule** (splitting DS-write/DS-read/VMEM-load/WMMA issue into
explicit groups, computing per-instruction issue rates from `wmma_cycle`/`ds_read_*_wmma_rate`)
— but the entire body is `#if 0`'d out, with the comment "`TODO: Calculation of the number of
instructions may require changes for WMMA`". In other words: **CK's own engineers built the
compute/memory-interleaving scheduler for WMMA by porting it from the XDL pipeline, then
disabled it because the instruction-count math wasn't re-derived for WMMA's different cycle
counts.** This is a concrete, second data point (alongside MISA's own "interleaved schedule
was a regression") that hand-scheduling the WMMA hot loop is genuinely hard to get right, not
just something MISA got wrong — CK hasn't shipped a working version of it either.

**A real, working scheduling primitive CK *does* ship for WMMA** (in `v1`, not `v3`):
`blockwise_gemm_pipeline_wmmaops_v1.hpp` around lines 916–942 brackets the first WMMA issue of
each MAC-loop body with `__builtin_amdgcn_s_setprio(1)` and closes the body with
`__builtin_amdgcn_s_setprio(0)`, each wrapped in `__builtin_amdgcn_sched_barrier(0)` (a
compiler-only reordering barrier, not a hardware wait). This temporarily raises the issuing
wavefront's hardware arbitration priority for the duration of the compute-bound WMMA burst, then
drops it back to let other wavefronts' VMEM/LDS-latency-hiding work interleave — functionally
the same goal as the "avoid inter-instruction arbitration stalls between back-to-back matrix
instructions" idea behind `disable_xdl_arb_stall` that the task background flags as something
MISA doesn't use. This is real, shipping code (not disabled like the v3 scheduler above).

## 3. LDS layout / transpose-on-load — the gfx12 hardware transpose-load builtin is real on gfx1250, but CK's only user of it is disabled there

Per the audit doc (`gfx1250_ck_audit.md`, "VERIFIED NON-ISSUES" and "RESOLVED by A/B test"
sections) and confirmed by reading `include/ck/utility/amd_transpose_load.hpp` directly: the
`__gfx12__`-gated `amd_global_load_transpose_to_vgpr()` (wrapping
`__builtin_amdgcn_global_load_tr_b128_v8f16` / `_tr_b64_v2i32`) **does compile and is
HW-valid on gfx1250** — the feature it needs (`gfx12-insts`) is present on gfx1250, unlike the
`wmma-128b-insts`/`wmma-256b-insts` features gfx1200/1201 need for their own WMMA family, which
gfx1250 lacks. The only caller of this builtin in CK, however, is the "wave-transfer"/direct-store
path (`gridwise_gemm_wmma_cshuffle_v3_common.hpp:197`, `#if defined(__gfx120__)` —
`IsAWaveTransferApplicable`/`IsBWaveTransferApplicable`), which is compiled out entirely for
gfx1250 (`__gfx120__` excludes gfx1250) because — per the audit's real-hardware A/B test — that
path bakes in gfx1200's 16x16x**16** WMMA tile layout, and enabling it unmodified on gfx1250
(16x16x**32** WMMA) silently computes wrong results (99.4% mismatched elements, verified on
real HW) despite reporting ~1.9x higher TFLOPS.

**Why this matters for MISA:** the constraint is real but narrower than "don't use hardware
transpose-load on gfx1250" — it's "don't reuse 16x16x16-layout math with a 16x16x32 WMMA
instruction." The hardware transpose-load builtin itself is available and untested-but-plausible
for gfx1250 if paired with correct K=32 tile-layout math (which CK never attempted — it just
disabled the whole path rather than porting it). This is upside CK left on the table, not a
dead end: `global_load_tr_b128_v8f16` (16 fp16 elements, transposed, per lane group) could
plausibly replace part of MISA's own A/B global-load-to-thread staging in the non-TDM path, but
would need MISA to work out the correct 16x16x32-consistent thread-to-element mapping from
scratch — CK provides no reference implementation to crib from here.

## 4. Split-K occupancy heuristic and the two-stage fp32-workspace accumulate design (highest priority — wrw)

CK's wrw split-K count (`gemm_k_global_split`'s analogue, called `KBatch`/`k_batch_` in CK) is
chosen by a **closed-form occupancy formula**, not a search:
`include/ck/tensor_operation/gpu/device/impl/split_k_utils.hpp::get_best_occupancy_k_batch_value()`:

```
max_capacity  = max_occupancy_per_CU * num_CUs        // from one cached
                                                        // hipOccupancyMaxActiveBlocksPerMultiprocessor call
optimal_split = floor(max_capacity / grid_size)        // grid_size = M/MPerBlock * N/NPerBlock * G
k_batch       = optimal_split > 1 ? optimal_split : 1
```

then clamped to `ceil(gemmK / KPerBlock)` (can't split more ways than there are K-tiles) and,
separately, hard-capped at 128 "to avoid accuracy issues" (see
`device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp:541-550`). This is a single arithmetic
formula evaluated once per launch (the `hipOccupancyMaxActiveBlocksPerMultiprocessor` call is
cached in a static `ActiveWorkgroupsPerCU`), with no runtime search over candidate split counts —
cheaper than MISA's ternary search (`driver/igemm_wrw_gtc_driver.h`, per the "ternary" grep hits),
at the cost of being a heuristic rather than an exact-optimum search. Given MISA's ternary
search is described as producing genuinely poor results on the worst wrw shapes, it's worth
cross-checking CK's closed-form value against MISA's search result on the same shapes — if
CK's heuristic and MISA's search converge, the search isn't the problem; if they diverge, MISA's
search may be mis-modeling occupancy (e.g. not accounting for the actual `hipOccupancy`-style
active-CU-block count the way CK does).

**The more valuable find:** CK ships a *separate* wrw device op,
`device_grouped_conv_bwd_weight_two_stage_wmma_cshuffle_v3.hpp` ("TwoStage"), specifically
for the reduction-heavy, small-M/N wrw shape. Its design (confirmed by reading the file, e.g.
lines 826, 872-931, 1034-1152): stage 1 runs the split-K GEMM with
`InMemoryDataOperationEnum::AtomicAdd` into a **workspace buffer typed as `AccDataType` (fp32)**,
not the final `WeiDataType` — `AccDataType* p_c_grid = type_convert<AccDataType*>(arg.p_workspace_)`.
Stage 2 is a *separate*, cheap elementwise kernel (`kernel_elementwise`/
`kernel_batched_elementwise`, using `GridwiseElementwiseWeightTransposeCast`/`GridwiseElementwiseCast`)
that casts (and optionally transposes) the fp32 workspace down to the final `WeiDataType`
(fp16/bf16) in one pass. This avoids ever atomic-adding into a low-precision buffer (which CK's
own audit — see "SEPARATE ISSUE — bf16 WrW accuracy failures at K=3" in
`gfx1250_ck_audit.md` — shows is a *real, measured* correctness bug in CK's single-stage bf16
path at large k_batch: 0.35–0.50 error vs 0.13 tolerance) and, as a performance side effect, lets
the atomic-heavy GEMM kernel do only fp32-native atomics (no packed/rounded bf16 atomic-add
read-modify-write contention) while a fully separate, non-atomic, embarrassingly-parallel kernel
does the final cast.

**Cross-check against MISA:** MISA's `python/operations/coalescing_store_wmma.py` (line ~48-56)
already always allocates the WMMA output accumulator as fp32 in global memory regardless of the
tunable's nominal output precision ("there is no fp16/bf16 atomic-add precision concern and no
workspace/cast step needed" per its own comment) — so MISA is *not* exposed to CK's specific
bf16-atomic correctness bug, and doesn't need CK's two-stage split for that reason. The two-stage
*pattern* is still worth studying for a different reason: CK's stage-2 cast/transpose runs as an
**independent, fully-parallel elementwise kernel launch** decoupled from the split-K GEMM's grid
shape, whereas MISA's non-atomic epilogue (LDS-reshuffle coalescing store) and atomic epilogue
are both folded into the same kernel as the GEMM. For wrw's worst shapes (tiny grid from small
M/N, huge K-split-driven grid.z), a similarly separated final-cast/writeback pass might get
better occupancy/vectorization than doing it inline per-CTA, but this would be a bigger
structural change than the other recommendations below.

## 5. Occupancy/grid strategy for small-M×N/huge-K (the wrw shape) — same heuristic as §4, not a different mechanism

Covered under §4 — CK has no separate/different split-K selection strategy specifically for
grouped-conv wrw vs plain GEMM; `get_best_occupancy_k_batch_value` is shared verbatim across
`device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp`,
`device_grouped_conv_bwd_weight_two_stage_wmma_cshuffle_v3.hpp`,
`device_grouped_conv_bwd_weight_multiple_d_wmma_cshuffle_v3.hpp`, and their XDL equivalents (see
grep hits across `include/ck/tensor_operation/gpu/device/impl/*bwd_weight*.hpp`). CK relies on
its wide tile-shape instance library (§1) to give the split-K heuristic a good MPerBlock/NPerBlock
to split around, rather than on a smarter split-count algorithm.

## 6. gfx1250-specific tuning beyond ISA-family selection

Nothing beyond what the audit doc already documents as "COVERED" (the `__gfx125__`-gated
16x16x32/16x16x64 WMMA instruction family selection in `amd_wmma.hpp`/`wmma_gemm.hpp`, 320KB LDS
size, cluster-launch/async-LDS gating) was found to constitute an actual *different, tuned*
gfx1250 code path outside of "this instruction family replaces that one." No gfx1250-specific
tile-shape defaults, pipeline-version defaults, or scheduling differences (vs. gfx1200/1201) were
found in the instance-selection heuristics — CK treats gfx1250 as "just another WMMA target" for
tuning purposes once the ISA-level gating is done, leaning entirely on the wide static instance
library (§1) rather than per-arch tuning logic.

## 7. Barrier/scheduling primitives around WMMA issue

- `block_sync_lds()` (a plain `s_waitcnt lgkmcnt(0) + s_barrier`) is CK's only barrier primitive
  in both WMMA pipelines — no split-phase/early-signal-late-wait barrier construct was found
  anywhere in the WMMA gridwise/blockwise pipeline files (that idiom does exist elsewhere in the
  CK tree for other kernel families per the task background, but not here).
- `__builtin_amdgcn_sched_barrier(0)` (compiler-reordering fence, zero hardware cost) surrounds
  most compute regions in both v1 and v3 pipelines, used to stop the compiler moving instructions
  across the boundary — this is the "no functional cost, just makes the schedule predictable"
  primitive CK relies on instead of manual `sched_group_barrier` interleaving in the (disabled) v3
  scheduler.
- `s_setprio(1)/s_setprio(0)` bracketing the WMMA-issue burst in v1 (§2) is the one real,
  shipping arbitration-priority technique — CK's closest equivalent to
  `disable_xdl_arb_stall`, and adoptable in isolation (it's two scalar instructions, no
  restructuring needed).

---

## Recommendations for MISA, prioritized

1. **(High value, low-to-medium risk) Adopt `s_setprio(1)`/`s_setprio(0)` bracketing around the
   WMMA-issue burst in the main loop.** MISA's main loop emission lives in
   `python/operations/wmma_main_loop.py`; CK's exact pattern (raise priority right before the
   first WMMA issue of the MAC-loop body, drop it back to 0 at the end of that body, each edge
   wrapped in a `sched_barrier(0)`-equivalent — MISA would need the assembly-level analogue,
   `s_setprio`) is two scalar instructions per loop iteration and directly targets the "avoid
   inter-instruction arbitration stalls" gap the background explicitly calls out as unaddressed.
   Low code-size risk; the main uncertainty is whether it measurably helps on real gfx1250
   hardware (CK's own comments don't quantify its benefit) — treat as a cheap experiment, not a
   guaranteed win.

2. **(Highest priority given wrw is the biggest gap, medium effort) Cross-check MISA's wrw
   ternary split-K search against CK's closed-form occupancy heuristic on the worst-case shapes.**
   Port `get_best_occupancy_k_batch_value`'s formula (`floor(max_occupancy_per_CU * num_CUs /
   grid_size)`, capped at `ceil(gemmK/KPerBlock)` and at some accuracy-driven max) as a second,
   near-free candidate alongside the existing search in `driver/igemm_wrw_gtc_driver.h`. If the
   two disagree on the worst (26x-slower) shapes, that's a direct signal the ternary search's
   cost model is wrong for tiny-grid wrw problems, which is cheap to test and high-value to fix
   given wrw is called out as the single worst gap.

3. **(Medium priority, higher effort) Widen MISA's wrw tile-shape space, specifically smaller
   MPerBlock/NPerBlock paired with larger KPerBlock.** CK's instance library pairs 64x64 tiles
   with KPerBlock up to 256 (vs MISA's fixed 32/64) specifically to give small-output-channel wrw
   problems a large-K/small-M,N option. This means adding new `gemm_k_per_block` tunable values
   (96, 128, 256) to MISA's wrw config space in `python/igemm/igemm_wrw_gtc_wmma_nhwc.py` and
   `python/operations/wmma_main_loop.py` (which currently treats K=32/64 as the only cases) —
   a real codegen extension, not a flag flip, since VGPR/LDS budgets and the K-substep loop
   structure would need re-deriving for each new K depth. This is the CK finding most directly
   aimed at wrw's worst (small, non-exact) shapes.

4. **(Low priority, exploratory) Investigate hardware transpose-load
   (`global_load_tr_b128_v8f16`/`_tr_b64_v2i32`, feature `gfx12-insts`) for MISA's non-TDM
   global-load path.** Confirmed HW-valid on gfx1250 (audit doc + `amd_transpose_load.hpp`
   direct read), but CK has no working reference for the 16x16x32-consistent layout math (its
   only user assumes gfx1200's 16x16x16 layout and is disabled on gfx1250). This would be new
   engineering for MISA (likely touching the global-load functor path in
   `python/igemm/igemm_fwd_gtc_wmma_nhwc.py`/`igemm_bwd_gtc_wmma_nhwc.py`), not a port — treat as
   a research spike, not a scheduled task, and only worth it if profiling shows global-load
   staging (rather than the K-tail/EXEC-mask paths already identified as the bigger cost) is a
   bottleneck on well-fitting shapes.

5. **(No action needed, confirms parity) MISA's LDS-reshuffle coalescing store and CK's default
   CShuffle epilogue (`epilogue_cshuffle_v3_wmma_base.hpp`) use the same mechanism** (per-thread
   accumulator → LDS in shuffled layout → `block_sync` → vectorized LDS read + threadwise global
   store). No adoptable idea here beyond what MISA already does. Similarly, CK's WMMA main loop
   (§2) has *shallower* prefetch/no LDS double-buffering compared to MISA's own double-buffering
   — nothing to adopt from CK on that axis either; if anything MISA's main loop is already ahead
   of CK's shipped WMMA pipeline here.
