# Stream-K / persistent-kernel design for wrw split-K (2026-08-28)

**Status: Approach A (K-only) implemented and hardware-validated the same day**, once the
user returned and could supervise (the design below was written during an earlier,
unsupervised part of the session, when actually writing this device-side code was
deliberately deferred -- see the "why deferred" note preserved below for that reasoning).
Implementation, bugs found, and validation results are in a new section at the end of this
doc ("Implementation update"). The sections below are the original research/design output
and remain accurate as background.

This doc is the output of a research pass (reading rocKE's reference implementation in
full, plus MISA's current wrw split-K mechanism in full) intended to make implementation
fast and low-risk. See `docs/gfx1250_optimization_backlog.md`'s Stream-K item for the
current one-paragraph summary.

## 1. Motivation (concrete, evidence-backed, not speculative)

`docs/gfx1250_vendor_benchmark_vs_miopen.md`'s last ~100 lines document that wrw's
`_gsplit` (split-K) numbers are the most volatile of anything benchmarked in this project:
the *same* shape, config, and search reported 0.065ms in one session and ~0.148ms in a
later session — a >2x regression with zero code change. The most likely explanation
(confirmed not a search-algorithm bug via a manual `IGEMM_GSPLIT_SWEEP` sweep): wrw's
split-K kernels launch **many small workgroups per candidate split** (grid.z equals the
chosen split count, which can reach into the hundreds-to-thousands, capped at 4096) and
are therefore unusually exposed to scheduling interference from other tenants on a shared
GPU — fwd/bwd's much larger, single-dispatch kernels show no such session-to-session
variance.

Stream-K / persistent-kernel design (as used by CK Tile, FlashAttention-3, etc.) fixes
this at the architectural level: launch a **small, constant-size grid** (order of magnitude
= number of CUs), and have those workgroups dynamically self-balance via a global atomic
counter, rather than pre-deciding "how many workgroups" via an expensive, noise-sensitive
pre-launch timing search.

## 2. Reference implementation (rocKE)

Found at `~/rocm-libraries/dnn-providers/hip-kernel-provider/rocke/platform/python/rocke/helpers/streamk.py`
and `.../helpers/persistent.py` (read in full; also mirrored under `~/rocm-libs/...`).
This is CK-Tile-parity Python IR-emission code (targets rocKE's own `IRBuilder`, not
MISA's raw-asm-string emitter) — a legitimate architectural reference, not a drop-in port
target.

### 2a. Work partitioning (`streamk.py`)

- Decomposes the GEMM into `(m_tile, n_tile, k_iter)` triples. `num_macro_tiles =
  m_tiles * n_tiles * k_iters` is the total unit-of-work count — maximally fine-grained,
  every output tile split into exactly `k_iters` separate work items.
- `emit_streamk_decode`: given a linear work-item id, `k_iter = id % k_iters`, `nn = id //
  k_iters`, `n_tile = nn % n_tiles`, `m_tile = nn // n_tiles` — K-major within a fixed
  `(m,n)` tile. Also derives `is_first`/`is_last` (k_iter==0 / k_iter==k_iters-1).
- `compute_streamk_grid_size`: **grid size = `min(num_macro_tiles, num_cus *
  blocks_per_cu)`** — capped at (a small multiple of) the CU count, never scaled to the
  problem's total tile/K-iter count. Default `num_cus=304` (MI300X/MI355X).
- Two reduction strategies:
  - **Atomic** (fully implemented in rocKE's "v1"): each CTA atomically adds its partial
    K-sum (`global_atomic_add_f32`) into an f32 workspace slot for that `(m,n)` tile.
    Needs a separate finalization pass to cast f32→output dtype. Conceptually identical
    to what MISA already does for wrw's `gemm_k_global_split`, just phrased per-work-item
    instead of per-shard.
  - **Reduction** (only the *decode* surface shipped; the actual busy-wait/flag-table
    reduction pass is explicitly deferred in rocKE's own source — "lands with the
    StreamK GEMM kernel itself" per its docstring, and that kernel does not exist
    anywhere on this machine). Contributing CTAs would cooperate via a tile-major
    workspace + flag table; the first contributor stores, later ones atomic-add, every
    contributor bumps a flag counter; the CTA that observes `flag == k_iters_per_tile`
    finalizes in-kernel, avoiding a second launch. **Not usable as a reference today —
    rocKE's own implementation of the hard part doesn't exist yet.**

### 2b. Persistent-kernel loop mechanics (`persistent.py`)

- Launch a small, constant number of CTAs (~`num_cus * waves_per_cu`), each pulling
  successive work-item ids from a global atomic counter (`atomic_add(1)`) until the
  counter exhausts the total tile count.
- `build_persistent_counter_init`: fetches this CTA's first tile id. **Only thread 0
  issues the atomic**, then the result is **broadcast** to every lane:
  - Single-wave CTA (`block_size <= wave_size`): every lane issues the atomic with a
    per-lane increment (only lane 0 non-zero), then `ds_bpermute(addr=0, val)`
    broadcasts lane 0's result — pure wave-internal SIMD traffic, no `s_barrier` needed.
    This replaced an earlier LDS+`s_barrier` design after the compiler was found to
    **elide the barrier** on the single-wave path at `max_iters>1`, silently skipping
    ~1.7% of loop bodies — a real, hardware-observed correctness bug in the prior version.
  - Multi-wave CTA (`block_size > wave_size`): LDS slot + a real `s_barrier` (not elided,
    since other waves genuinely observe it).
- `persistent_tile_loop`: a **bounded** `scf.for` of `max_iters` trips (statically sized
  to `ceil(num_tiles_total / launch_grid_size)`), threading `tile_idx` as a loop-carried
  value. Each iteration yields `(tile_idx, in_range)`; the caller's body runs under
  `scf.if(in_range)`. After the body, the loop fetches the *next* tile id for the next
  iteration. Over-fetching past `num_tiles` is harmless (spurious atomic traffic only);
  **under-estimating `max_iters` is a latent, uncaught bug** the helper does not defend
  against.

### 2c. Launch geometry

Grid = `min(total_work_items, num_cus * blocks_per_cu)` — small, constant, decoupled from
problem size. This is the fundamental structural difference from MISA's current design
(grid.z scales with the *chosen split count*, up to 4096).

## 3. MISA's current wrw split-K mechanism (file:line references)

### 3a. Grid.z computation and shard K-range decode

- Host (`driver/igemm_wrw_gtc_driver.h`): `time_split(splits)` (~944-992) sets
  `karg.gemm_k_per_wg = (num_k_blocks / splits) * gemm_k_per_block` and
  `karg.gemm_k_num_splits = splits`, launches `grid = {grid_x*block_size, grid_y,
  splits}` (954). **grid.z literally equals the chosen split count** — no
  persistent/bounded-grid concept exists today. Cap: `MAX_GEMM_K_SPLITS=4096`
  (`igemm_gtc_base.h:725-732`, `igemm_gemm_k_global_split_cap`); observed candidates up
  to 1260+ splits historically.
- Device (`python/igemm/igemm_wrw_gtc_wmma_nhwc.py:688-713`, `emit_kernel_prologue`):
  gfx1250 packs `blockIdx.y`/`blockIdx.z` into `ttmp7` (low16/high16). Split variant
  decodes `s_and_b32 s_by, ttmp7, 0xffff` / `s_lshr_b32 s_bz, ttmp7, 16` (698-699) — the
  **only place in the whole WMMA codebase that decodes a 3rd grid dimension** today.
  `s_gemm_k_wg_off = s_bz * s_gemm_k_per_wg` (713) is this shard's K-slice base, folded
  into A's persistent base address (862) and re-added every main-loop iteration to B's
  gather (`_emit_b_gather`, ~1026, since wrw's GEMM_K is spatial and regathered every
  iteration, unlike fwd/bwd's constant K stride).
- `s_knum` (756) = `s_gemm_k_per_wg` (not true `gemm_k`) — **the WMMA main loop itself is
  unmodified**; shard size is communicated purely by shrinking the loop-counter target.
  With `wmma_k_tail` set, only the last shard (`bz == gemm_k_num_splits-1`) gets `s_knum`
  extended by the true remainder (757-766) — shard ranges are exact contiguous multiples
  of `gemm_k_per_block`, no gap/overlap logic needed.

### 3b. Atomic epilogue

`python/operations/coalescing_store_wmma.py`, `igemm_coalescing_store_wmma_t.__call__`,
the `elif ctrl.gemm_k_global_split:` branch (592-663). Per accumulator element:
`global_atomic_add_f32 ... scope:SCOPE_SYS` (657, default) or `global_atomic_add_u32` for
int8/int4 (656, Phase 57), or (if `wrw_reduction_kernel`) a plain non-atomic
`global_store_dword` (642) into a disjoint per-shard workspace slice. **`scope:` is
load-bearing** — a bare atomic silently drops cross-CU updates on this hardware (found
and fixed in Phase 17). Optional packed-bf16 atomic variant exists but measured a net
loss (Phase 34) since atomic contention was never actually the bottleneck here
(`TX_VMW_ATOMIC_SETCONFLICT_STALL` measured at exactly zero).

### 3c. Host-side split-count search (ternary search)

`driver/igemm_wrw_gtc_driver.h` ~773-1038. Only divisors of `num_k_blocks` are valid
candidates (WMMA main loop has no general K-tail handling); capped by
`igemm_gemm_k_global_split_cap` = `min(4096, num_k_blocks, gemm_k/32)`. One candidate
injected from CK Tile's occupancy heuristic (`compute_gemmk_global_splits`, real
`hipModuleOccupancyMaxActiveBlocksPerMultiprocessor` query). The actual search is a
**ternary search over the sorted divisor list**, each comparison a real timed launch
(cached), exploiting empirically-measured unimodality of cost-vs-split-count.
`IGEMM_GSPLIT_SWEEP` bypasses this for manual research. **This entire search — and its
sensitivity to per-launch timing noise — is exactly what Stream-K's constant-grid design
would make unnecessary.**

### 3d. `wrw_reduction_kernel` (closest existing analog to Stream-K's "Reduction" strategy)

Tunable `wrw_reduction_kernel`/`_wsred` (Phase 35, `igemm_base.py:795-803`). Each shard
does a **plain non-atomic store** into its own disjoint slice of a
`num_partitions x output_size` fp32 workspace; a separate reduction kernel
(`wrw_reduce_partials_f32`, `driver/gpu_tensor_cast/gpu_tensor_cast.cpp:263-281`, a
trivial grid-stride sum over `num_partitions` slices) runs afterward. This is essentially
CK-Tile's non-atomic disjoint-workspace idea, already shipped and hardware-validated —
but still driven by the same fixed-grid.z-decided-before-launch split count; only the
epilogue write mechanism changes, not the work-distribution/launch-grid model.

### 3e. `gsplit_stagger` (launch stagger) — the workaround Stream-K would obsolete

`gsplit_stagger` (`igemm_base.py:608-628`; emission at
`igemm_wrw_gtc_wmma_nhwc.py:702-710`) emits `s_sleep_var` right after `s_bz` decode — a
pure wall-clock perturbation to desynchronize simultaneously-launched shards' first
memory bursts. Measured ~3-4% faster only at very high split counts, no reliable benefit
at low/moderate splits. **This is exactly the class of workaround Stream-K obviates by
construction**: persistent CTAs pull work at genuinely different real times (each
finishes its current unit, then atomically claims the next), so there's no "N shards all
launched simultaneously, all starting at K-offset 0" pattern to stagger in the first place.

## 4. Concrete engineering gaps and risks

### 4a. No persistent-kernel primitive exists anywhere in MISA's codegen model

MISA hand-emits raw assembly text; there is no `IRBuilder`/`scf_for_iter`/`ds_bpermute`
abstraction. Every existing loop (main K-loop, wrw's own tap loop) is a hand-written
label + `s_cbranch_scc0/1` pair over compile-time-known register roles. A persistent
"claim from atomic counter, loop until exhausted" pattern needs: a new global-memory
counter kernarg/buffer (zeroed once per dispatch), a new looping macro (fetch tile id via
single-lane atomic + broadcast — MISA already has the `v_readfirstlane`-style broadcast
idiom used for `s_wave_id` derivation, `igemm_wrw_gtc_wmma_nhwc.py:686`), decode
`(m_tile,n_tile,k_iter)` via MISA's existing `macro_int_div_rem_vs_gfx1250_t` (already
used pervasively — this part is a genuine, low-risk reuse), then branch back to a loop
top that must **re-run the entire per-tile prologue** (block offsets, address
recomputation, LDS offset setup) every iteration, not once at kernel entry as today.

### 4b. `coalescing_store_wmma.py`/`wmma_main_loop.py` assume 1 workgroup ↔ 1 (tile,
K-range), fixed at launch

Epilogue addressing (`v_gemm_im`/`v_gemm_in`, `s_block_m_off`/`s_block_n_off`) is derived
once in the prologue from `s_bx`/`s_by` and treated as immutable for the kernel's
lifetime. A persistent kernel that claims a *different* output tile on a later iteration
needs to recompute these from scratch every iteration. **Useful precedent**: wrw's
*existing tap loop* already re-enters the main loop with fresh per-iteration addressing
(re-zero `v_c`, reset `v_addr_a`/`v_addr_b`, re-issue loads, re-invoke
`emit_kernel_fma_main_loop()`, ~lines 917-953) — but it only varies the *tap* (iy,ix)
today, not the full `(m_tile,n_tile,k_iter)` triple, and doesn't yet know how to pull that
triple from a runtime atomic counter rather than a compile-time-nested double loop.

### 4c. What the fixup step needs

The **Atomic** strategy is nearly free to build on MISA's existing atomic epilogue
(§3b) — the only change is *what decides* the K-range and tile (today: static
`bz`/`gemm_k_per_wg`; Stream-K: decoded from a claimed linear id). The **Reduction**
strategy needs a flag-table wait/finalize mechanism that exists in neither MISA nor
(completely) in rocKE's own reference — genuinely new engineering with no working
example to copy, and it introduces a busy-wait spin loop, a class of code MISA has zero
precedent for.

### 4d. gfx1250-specific risks

- `blockIdx.z` packing (`ttmp7` low16/high16) becomes irrelevant if Stream-K abandons
  grid.z entirely (likely, since persistent kernels want a flat grid sized to CU count)
  — this actually *removes* the one existing landmine here.
- Atomic `scope:SCOPE_SYS`/`SCOPE_DEV` must be preserved on any new atomic epilogue path
  (bare atomics silently drop cross-CU updates on this hardware, Phase 17).
- **Hang risk**: this exact project already hit an unrecoverable GPU hang (back-to-back
  same-register WMMA with zero interleaving; required a physical machine reboot — see
  `docs/claude_persistent_memory_notes.md`'s "gfx1250 WMMA hang risk"). wrw's kernels are
  always 4-wave (`block_size=128`, multi-wave), so rocKE's single-wave `ds_bpermute`
  broadcast bug class shouldn't recur verbatim, but MISA would need the LDS+`s_barrier`
  path instead — a brand-new cross-wave synchronization point per persistent-loop
  iteration with no precedent in this codebase. Any new barrier/wait logic here needs
  careful, hardware-supervised testing (not an unsupervised session) given this history.
- Wave32: WMMA kernels here are wave32-only; any new lane-0-broadcast logic should follow
  the existing `v_cmpx`/`exec_lo` idiom already used throughout the epilogue (e.g.
  `wmma_m_tail`/`wmma_n_tail` masking), not a 64-bit saveexec pattern.

## 5. Proposed implementation approaches

### Approach A — "Minimal Stream-K" (recommended starting point)

Launch a constant-size grid (≈ num_cu, or a small multiple, per
`compute_streamk_grid_size`) instead of `grid.z = splits`. Each persistent workgroup runs
a bounded loop (`max_iters = ceil(total_work_items / grid_size)`, compile/launch-time
known — **no data-dependent trip count, so no hang risk from the loop bound itself**)
that: (1) atomically claims the next linear work-item id (single lane + LDS+barrier
broadcast, per §4a/4d), (2) decodes `(m_tile,n_tile,k_slice)` via the existing div/rem
macro, (3) re-derives `s_block_m_off`/`s_block_n_off` and re-runs the tap-loop-style reset
(zero `v_c`, reset addresses, re-issue loads), (4) runs the existing WMMA main loop
unmodified over that K-slice, (5) runs the existing atomic epilogue unmodified.

**Reused nearly as-is**: atomic epilogue, div/rem macros, tap-loop's re-entry precedent,
per-dispatch zero-init discipline, `scope:SCOPE_SYS` atomics.
**New**: the persistent-loop control flow itself, the tile-claim atomic counter +
broadcast, generalizing per-tile addressing from prologue-once to recomputed-every-iteration.
**Effort**: Medium-High. **Risk**: Medium — main risks are getting the multi-wave
broadcast barrier genuinely correct (rocKE found a real bug in the analogous single-wave
case), correctly sizing `max_iters`/`in_range` so the tail isn't dropped or
double-processed, and auditing every prologue-computed register for whether it needs to
move into the per-iteration reset (e.g. `s_wei_row_c`, group-decode offsets).

### Approach B — "Full Stream-K" (defer)

Arbitrary/fractional K-slice-to-workgroup mapping (CK Tile's classic Stream-K, not just
exact-divisor shards), via either a generalized Atomic strategy (needs a finalize pass
per tile) or the Reduction strategy (flag-table + last-contributor finalize — genuinely
new on both sides of the port, since rocKE's own reference doesn't ship a complete
version either). **Effort**: High-to-very-high. **Risk**: High — new spin-wait
synchronization primitive class for this codebase, highest correctness-audit burden, and
a lost-wakeup/ordering bug here risks a hang, not just a wrong answer. **Not worth
attempting before Approach A is built and measured.**

### Approach C — Hybrid: persistent grid + existing `wrw_reduction_kernel` fixup

Keep Approach A's persistent/bounded-grid tile-claim loop, but reuse the already-shipped,
already-validated `wrw_reduction_kernel` disjoint-workspace + separate-reduction-kernel
path (§3d) as the fixup instead of the atomic epilogue. Avoids the atomic-scope/ordering
surface for the new mechanism (only the tile-claim counter itself is atomic, a much
simpler single-atomic-per-work-item pattern), at the cost of a second kernel launch (same
cost `wrw_reduction_kernel` already pays today) and reworking the workspace addressing
scheme to be indexed by claimed tile id rather than fixed shard id. **Effort**: Medium
(between A and B). **Risk**: Medium-Low on the fixup side (proven mechanism), Medium on
the persistent-loop side (same novel-mechanism risk as A).

### Recommendation

Start with **Approach A** — smallest change that captures Stream-K's core, evidence-backed
benefit (§1) while reusing the largest fraction of already-hardware-validated MISA
machinery. Approach C is a reasonable fallback if A's atomic-epilogue-under-dynamic-
tile-reassignment has correctness wrinkles not anticipated here. Approach B should be
deferred indefinitely unless A is measured and found insufficient.

## 6. Key files for implementation

- `python/igemm/igemm_wrw_gtc_wmma_nhwc.py` — `kernel_sgpr_t`/`kernel_vgpr_t` (247-420),
  `emit_kernel_prologue` (657-894), `emit_kernel_tap_loop` (895-1001, the closest existing
  "re-enter main loop, fresh per-iteration state" precedent), `_emit_b_gather` (1004+).
- `python/operations/coalescing_store_wmma.py` — `ctrl_coalescing_store_wmma_t` (31-169),
  the atomic branch (592-663).
- `python/igemm/igemm_base.py` — `get_igemm_gtc_gemm_k_global_split` (144-153),
  `gemm_k_global_split`/`atomic_scope`/`gsplit_stagger`/`wrw_reduction_kernel` tunables
  (~267-803).
- `driver/igemm_wrw_gtc_driver.h` — grid.z/karg construction and ternary search
  (760-1038), `if_gemm_k_global_split`/`compute_gemmk_global_splits` (422-509).
- `driver/igemm_gtc_base.h` — `igemm_gemm_k_global_split_cap` (725-732), `wrw_reduce_karg_t`
  (75-82).
- `driver/gpu_tensor_cast/gpu_tensor_cast.cpp:263-281` — `wrw_reduce_partials_f32`, the
  existing fixup-pass donor for Approach C.
- Reference: `~/rocm-libraries/dnn-providers/hip-kernel-provider/rocke/platform/python/rocke/helpers/streamk.py`
  and `.../helpers/persistent.py` (also mirrored at `~/rocm-libs/...`).
- Docs consulted: `docs/gfx1250_wmma_layout.md` Phase 17 (1733-1857), Phase 18/20/33
  (1857+, 3257-3320), Phase 34 (3321-3417), Phase 35 (3419+), Phase 41 (4076-4157), Phase
  50 (~4773+); `docs/gfx1250_vendor_benchmark_vs_miopen.md:574-674`;
  `docs/claude_persistent_memory_notes.md` (WMMA hang risk, VGPR-MSB, wrw status summary).

## Implementation update (2026-08-28, same day, user supervising)

Implemented Approach A exactly as scoped above (K-only, single-tap, grid.x/grid.y
unchanged), with one simplification: shard granularity is fixed at exactly one
`gemm_k_per_block` per claimed unit (`total_shards = num_k_blocks`, the finest granularity
rocKE's own `k_iter` uses) rather than a tunable coarser chunk size -- kept simple since
proving the mechanism was the goal, not tuning granularity.

### What changed

- **New tunable `wrw_streamk`** (`igemm_base.py`) — requires `gemm_k_global_split=1`,
  `nxe==0`, and asserts against `wmma_k_tail`/`tdm_global_load`/`wrw_reduction_kernel`
  (none of those combinations were attempted).
- **New kernarg fields** (`igemm_wrw_gtc_wmma_nhwc.py`'s `get_kernel_args()`,
  `igemm_wrw_gtc_driver.h`'s karg struct): `p_streamk_counter` (pointer to a
  host-zeroed `grid_x*grid_y*4`-byte int32 workspace, one atomic-claim counter per output
  tile), `streamk_max_iters`, `streamk_grid_y`. `gemm_k_num_splits` is reused (same
  meaning, "total shard count", now the persistent loop's in-range bound instead of the
  wmma_k_tail last-shard check).
- **New device method `emit_kernel_streamk_loop()`** — replaces
  `emit_kernel_tap_loop()` for `wrw_streamk` builds. A bounded loop (compile/launch-time
  `max_iters`, no data-dependent trip count): claim next shard index (flat-tid==0 issues
  `global_atomic_add_u32` with the SADDR form + `th:TH_ATOMIC_RETURN`, EXEC-masked via
  `v_cmpx_eq_u32`), broadcast via an LDS round-trip + a real `s_barrier_signal`/
  `s_barrier_wait` pair (this kernel is always 4-wave/`block_size=128`), in-range check,
  then the same per-shard body the static design already used (zero `v_c`, recompute
  `v_addr_a` from a now-split-K-offset-free `v_addr_a_base`, `_emit_b_gather`, main loop,
  atomic epilogue store).
- **Host driver** (`igemm_wrw_gtc_driver.h`): new `time_streamk()` closure, used instead
  of the ternary search entirely when `wrw_streamk` is set — sizes the persistent grid via
  the existing occupancy heuristic (`compute_gemmk_global_splits`), capped to
  `[1, total_shards]`, computes `max_iters`, allocates+zeros the counter workspace (zeroed
  every dispatch, same discipline as the atomic epilogue's own output zero-init).
- New config: `config/igemm_wrw_gtc_gfx1250_nhwc_bf16_streamk.config` (opt-in only, not in
  the master config union yet).

### Two real bugs found via hardware testing (both fixed)

1. **Missing `s_wait_dscnt 0x0` before the first barrier.** Every existing
   barrier-after-LDS-write in this codebase (`coalescing_store_wmma.py`) precedes
   `s_barrier_signal` with `s_wait_dscnt 0x0` to ensure the write has actually retired
   before the barrier releases other waves to read it — my first version omitted this.
   Caused flaky, run-to-run-varying wrong results at higher shard counts. Fixed by adding
   the wait (matching established convention exactly).
2. **`s_ix` never initialized (the actual root cause of most observed failures).**
   `emit_kernel_tap_loop()` normally zeroes `s_ix` at the top of its own loop; this new
   method bypasses the tap loop entirely and never set it, leaving it as whatever garbage
   occupied that SGPR at kernel entry. `_emit_b_gather` reads `s_ix` (dilation/pad offset
   for B's gather) unconditionally, so a garbage value pushed `wi_idx` out of
   `[0, wi)` for every thread, masking every B load to zero via the existing OOB-`v_flag`
   mechanism — main loop ran, but on all-zero B data, so `v_c` (and the final output)
   stayed exactly zero. This gave a *deterministic* zero output at small shard counts and
   a garbage-dependent (hence apparently "flaky") wrong output at others, since the
   specific garbage value in that SGPR slot varies with whatever prior kernel/context left
   it. Found via a hardware raw-dump diagnostic (temporary, since removed) that isolated
   each stage: claim mechanism confirmed correct first (every thread saw the right claimed
   tile_idx and total_shards), then a dump of `v_gld_b` immediately after the initial
   global loads showed it was unconditionally zero — pointing straight at the OOB mask.
   Fixed with one `s_mov_b32 s[s_ix], 0` at the top of the new method (`nxe==0` guarantees
   `y=x=1`, so this is valid for the entire kernel lifetime, not just per-tap).

### Hardware validation

All on `bf16`/wrw/NHWC, spot-checked shapes (not an exhaustive sweep — this is a proof of
mechanism, not a finished feature):

| Scenario | Shape | total_shards | persistent grid.z | max_iters | Result |
|---|---|---|---|---|---|
| Single shard (no contention at all) | n=1,c=128,H=4,W=8,k=128 | 1 | 1 | 1 | `valid:y`, nrms~0.0005 |
| Degenerate 1:1 (small) | n=2,c=128,H=8,W=8,k=128 | 4 | 4 | 1 | `valid:y` |
| Degenerate 1:1 (larger) | n=8,c=128,H=16,W=16,k=128 | 64 | 64 | 1 | `valid:y` |
| **Genuine multi-claim, exact multiple** | n=64,c=512,H=16,W=16,k=512 | 512 | 64 | 8 | `valid:y`, nrms~0.0005 |
| **Genuine multi-claim, non-exact (tail)** | n=50,c=512,H=16,W=16,k=512 | 400 | 64 | 7 | `valid:y`, nrms~0.0004 |

Every scenario repeated 2-5 times with fresh random data each run — no flakiness observed
after the two fixes above. Zero regression confirmed: every existing (non-`wrw_streamk`)
kernel in the full bf16 wrw master config is byte-identical before/after this change
(`id()`-based label-name noise aside).

### Performance comparison (2026-08-28, same day, GPU under contention -- see caveat)

Spot-checked against the existing `_gsplit` design's best-of-search result, real shapes
from `docs/gfx1250_vendor_benchmark_vs_miopen.md`'s wrw trace (n=42, 1x1, `bench_out/
wrw_bf16_master` vs. the `_streamk` config, `-V 0`, `IGEMM_WARMUP=5 IGEMM_REPEAT=10`):

| Shape (c,H,W,k) | `_gsplit` (best-of-search) | `wrw_streamk` | Ratio |
|---|---|---|---|
| 128,30,40,128 | 0.073ms (chosen split=315) | 0.297ms (grid.z=1024) | **~4.1x slower** |
| 256,30,40,128 | 0.128ms (chosen split=175) | 0.565ms (grid.z=512) | **~4.4x slower** |

**`wrw_streamk` is currently slower, not faster — and the reason is identifiable, not
mysterious.** `STREAMK_DEBUG=1` shows the persistent grid is NOT actually small: for the
128,30,40,128 shape it launched **1024 workgroups** (more than `_gsplit`'s own chosen 315),
each doing up to 2 claims (`total_shards=1575`). The bug is in the occupancy-sizing formula
reused from the existing heuristic (`compute_gemmk_global_splits` =
`num_cu * potential_occupancy / grid_size`): that formula was designed to answer "how many
splits should exist" and scales *up* as `grid_x*grid_y` (`grid_size`) shrinks — for these
shapes `grid_x*grid_y=1` (a single output tile), so it returns a large number. That's
backwards for a persistent grid, which should stay close to `num_cu` (~256) itself
regardless of how many output tiles exist. The result: `wrw_streamk` pays real per-shard
atomic-claim + LDS-broadcast + double-barrier overhead on top of a workgroup count that's
*similar in magnitude* to `_gsplit`'s own chosen split count, with none of Stream-K's
intended "small constant grid" benefit actually realized.

### Performance fix (same day, implemented immediately after the finding above)

Two changes to `time_streamk()` in `igemm_wrw_gtc_driver.h`, host-side only (zero
device-code changes needed):

1. **Grid.z sizing**: replaced `compute_gemmk_global_splits` (the splits-heuristic
   formula, which scales *up* as `grid_x*grid_y` shrinks) with a direct target: `num_cu *
   blocks_per_cu` (`blocks_per_cu=1`) total persistent workers, divided across
   `grid_x*grid_y` independent per-tile counters — matching rocKE's own
   `compute_streamk_grid_size` shape. Uses the *raw* `hipDeviceProp_t::multiProcessorCount`
   queried fresh in this closure, not `this->num_cu` (the base class doubles that for
   gfx10+, for the unrelated splits-heuristic's own purposes — reusing it would have
   re-inflated the exact worker count this fix is trying to shrink).
2. **Shard granularity** (the actual dominant cost, confirmed by testing grid.z-only
   shrinkage first and finding it barely moved the needle — see below): shards are no
   longer fixed at exactly one `gemm_k_per_block`. `time_streamk` now targets a total
   shard count of `min(per_tile_workers * 4, 256)` — a handful of claims per worker,
   capped in absolute terms since every claim costs a real atomic + LDS-broadcast +
   double-barrier round trip regardless of how many workers share the work — snapped down
   to the largest divisor of `num_k_blocks` (still required to be exact, no K-tail relief
   in this mode) via the same `largest_divisor_leq` helper the old ternary search already
   used. `karg.gemm_k_per_wg` is set to that shard's real size
   (`num_k_blocks/total_shards * gemm_k_per_block`) instead of a hardcoded single block.

**Why grid.z alone wasn't enough**: re-tested with only fix 1 applied (shard count still
`num_k_blocks`) — grid.z dropped from 1024 to 512 for the 128,30,40,128 shape, but cost
barely changed (0.297ms → 0.319ms, actually slightly worse). The total number of
atomic-claim/broadcast round trips across the whole dispatch is driven by *shard count*
(still ~1575 either way), not worker count — concentrating the same total claims into
fewer workers just doubles `max_iters` per worker, leaving the aggregate overhead
unchanged. Fix 2 (coarser shards) is what actually mattered.

**Result after both fixes** (same shapes, same methodology):

| Shape (c,H,W,k) | `_gsplit` (best-of-search) | `wrw_streamk` (fixed) | Ratio |
|---|---|---|---|
| 128,30,40,128 | 0.073ms (split=315) | 0.077ms (total_shards=225, grid.z=225) | **~1.05x — near parity** |
| 256,30,40,128 | 0.129ms (split=175) | 0.169ms (total_shards=128, grid.z=128) | **~1.3x slower** |

Down from ~4.1x/~4.4x slower before the fix. Note this comparison only measures the
*chosen candidate's* kernel time for `_gsplit` — it doesn't count the cost of the ternary
search itself (multiple real timed launches to find split=315/175), which `wrw_streamk`
doesn't need at all (one shot, no search). In a real deployment where the split count
isn't pre-cached, `wrw_streamk`'s total wall-clock cost could already be lower even at
~1.3x slower per-dispatch — not measured here, a natural follow-up.

Re-validated correctness after the fix on every scenario from the table above (all still
`valid:y`, including the tail case, which now lands at `total_shards=50, grid.z=16,
max_iters=4` instead of `400/64/7` — same qualitative shape, much cheaper).

**Caveat**: `rocm-smi` showed `GPU use (%): 100` from another tenant throughout both the
original comparison and the re-measurement — absolute numbers aren't final on an idle GPU,
but the *qualitative* improvement (near-parity instead of 4x slower) is large enough, and
confirmed via `STREAMK_DEBUG`'s direct shard/grid.z readout (not just wall-clock timing),
to trust directionally.

### What's NOT done (real scope gaps, not swept under the rug)

- **No tuning/search over the sizing constants at all** — this is the biggest gap. The
  existing `_gsplit` design finds its split count via a real ternary search over dozens of
  real-timed candidates per shape (see `time_split`'s search loop above `time_streamk` in
  `igemm_wrw_gtc_driver.h`). `time_streamk()` makes exactly ONE fixed heuristic choice
  (`blocks_per_cu=1`, `claims_per_worker_target=4`, `max_total_shards=256`, all hardcoded
  C++ constants, not config-file tunables or env-var-sweepable like
  `IGEMM_GSPLIT_SWEEP`) and never searches. These constants got the two measured shapes
  from ~4x slower to near-parity, but they are unvalidated guesses, not a tuned optimum —
  a real per-shape sweep would very likely do better in some regimes and worse in others.
- **The actual motivating question hasn't been measured**: is `wrw_streamk` more
  *contention-resilient* than `_gsplit`, not just similarly fast on one clean-ish run?
  `docs/gfx1250_vendor_benchmark_vs_miopen.md` documents `_gsplit` showing >2x
  session-to-session variance under contention specifically because it launches
  hundreds-to-thousands of simultaneously-dispatched workgroups — that's the entire reason
  Stream-K was worth building. Repeating the same shape many times (both designs) under
  real contention and comparing variance, not just mean, is the real test and hasn't been
  done.
- **Only one config exists**: `config/igemm_wrw_gtc_gfx1250_nhwc_bf16_streamk.config`,
  128x128x32, bf16 only, not in the master config union. To reach this via the normal
  driver search/benchmark flow, need at least a 64x64x32 variant (matching `_gsplit`'s
  existing tile-shape coverage) and fp16/fp32/int8 mirrors, then a decision on whether/when
  to fold into the master union (deliberately deferred pending the two items above).
- **Re-measure on a confirmed-idle GPU** — both the original 4x-slower finding and the
  fixed near-parity result were measured under contention from another tenant.
- **fp16/fp32/int8 untested** — only bf16 built and validated.
- **Only the 128x128 tile shape tested** — 64x64 and other existing wrw tile shapes not
  attempted.
- **Not combined with `wmma_m_tail`/`wmma_n_tail`** (M/N-tail masking) — only exact-multiple
  gemm_m/gemm_n tested. Blocks a real chunk of real shapes (anything whose gemm_m/gemm_n
  isn't an exact tile multiple).
- **`wsred`-equivalent (Approach C) not attempted** — this implements Approach A's atomic
  strategy only.

### Resuming on another machine — prioritized next steps

1. **Contention-stability measurement** (cheapest, and the load-bearing question this
   whole effort rests on) — repeat the same shape N times each for `_gsplit` and
   `wrw_streamk` under real contention, compare variance not just mean. If `wrw_streamk`
   isn't more stable, the rest of this list matters much less.
2. **Expose the sizing constants for tuning** — turn `blocks_per_cu`/
   `claims_per_worker_target`/`max_total_shards` into an env-var sweep (mirroring
   `IGEMM_GSPLIT_SWEEP`) at minimum, ideally a real per-shape search like `_gsplit`'s own
   ternary search.
3. **Re-measure on an idle GPU** to get a trustworthy absolute baseline.
4. **More configs** (64x64x32, fp16/fp32/int8) once 1-3 suggest it's worth the coverage.
5. **M/N-tail support**, then Approach C, roughly in that order of expected value.
