# gfx1250 WMMA performance optimization backlog

**Purpose**: a single tracked checklist for every optimization idea raised across
`docs/gfx1250_perf_parity_action_plan.md`, `docs/gfx1250_rocprof_profiling.md`, and
direct hardware investigation. Items are added when identified and **removed only when
actually implemented** (moved to "Done", with a link to the commit/doc section that did
it) — not when merely discussed or deferred. If an item is deferred, it stays listed as
open, not silently dropped. This doc is the source of truth for "what's left"; the two
docs above are the source of truth for "why" (full research, cross-source validation,
hardware findings).

Status values: `[ ]` open, `[~]` in progress / partially done, `[x]` done (see Done
section for the record).

## Tier 0 — cheap and diagnostic (measure, don't change code)

- [x] **Measure actual occupancy** (`SQ_WAVES`/`hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`
      vs. theoretical max waves/CU), across fwd/bwd/wrw and base/tail/split-K variants.
      Done 2026-08-27 — see `docs/gfx1250_rocprof_profiling.md` Finding 3 and
      `script/gfx1250_occupancy_check.cpp`. Result: occupancy is a pure function of tile
      shape (25.0% for 128x128, 31.2% for 64x64), flat across direction and mechanism.
      This **revises** (weakens, doesn't kill) the `disable_xdl_arb_stall` justification
      below — wrw is not uniquely low-occupancy relative to fwd/bwd.
- [x] **Extend rocprof hardware-counter profiling to bwd and fwd's tail paths**
      (mtail/ntail/ktail/tdm) — previously only fwd-base and wrw-gsplit were profiled.
      Done 2026-08-27 — see `docs/gfx1250_rocprof_profiling.md` Finding 4. Same
      qualitative story as the base paths (SQ busy ~93-94%, WMMA busy well under 1%, no
      tail mechanism stands out as different in kind) but the run was under confirmed
      heavy GPU contention (`rocm-smi` showed 100% GPU use from other tenants), so
      absolute WMMA-busy percentages are not comparable 1:1 to Finding 1/2's numbers.
- [ ] **Re-run Finding 1/2/4's profiling on a confirmed-idle GPU** to get absolute
      WMMA-busy-fraction numbers that are directly comparable across all profiled
      kernels (currently only within-run relative comparisons are trustworthy for
      Finding 4). Blocked on GPU availability, not effort — check `rocm-smi --showuse`
      shows 0% from other tenants before running.
- [ ] **No LDS-bank-conflict-specific counters collected** — dedicated counters exist on
      this GPU under the `TX_VMW_*`/`SQC_*` blocks (`rocprofv3 --list-avail`), not yet
      explored. Cheap to add to the existing `--pmc` invocation once picked.
- [ ] **Run `rocprof-compute`** (the roofline/comprehensive successor to Omniperf,
      confirmed present at `/home/sgundabo/rocm-10.1/bin/rocprof-compute` and presumably
      also under `/opt/rocm-10.1.0a20260820/`) for an automated roofline/occupancy report
      per kernel instead of hand-picked counters.

## Tier 1 — small effort (cross-validated or ISA-doc-motivated, cheap to try)

- [ ] **`disable_xdl_arb_stall` (`SCHED_MODE` bit[2]) A/B test on a wrw split-K shape.**
      **Attempted 2026-08-27, blocked — not a guess we should make.** The CDNA5 ISA doc
      (§5.7.2.1) documents this bit's *existence and semantics* but gives no `S_SETREG`
      hardware-register ID/encoding for `SCHED_MODE`, and it is conspicuously absent
      from the doc's own "Wave State Registers" table (§3.4, the complete list of
      `S_GETREG`/`S_SETREG`-addressable registers, indices 1-28) — every other
      documented writable register (`MODE`, `STATE_PRIV`, `EXCP_FLAG_USER`, etc.) is in
      that table with an explicit index; `SCHED_MODE` is not. Confirmed via `llvm-mc
      -mcpu=gfx1250`: `hwreg(HW_REG_SCHED_MODE, ...)` is not a recognized symbolic name
      (LLVM has no built-in constant for it either). Writing an arbitrary/guessed
      register ID via `s_setreg_b32` on real hardware is a correctness hazard, not a
      performance experiment — it can corrupt unrelated wave state. **This item stays
      open but is not actionable without the actual register ID/encoding** (from a more
      complete internal register-ID reference than what's currently available, or a
      vendor confirmation). Do not implement by guessing the hwreg ID.
- [x] ~~**hipconv's staggered per-shard K-loop start phase**~~ — **investigated
      2026-08-27, not implemented: could not verify the mechanism exists.** Searched
      the actual local hipconv source (`~/hipconv/hipconv`, `~/rocm-ck-hipconv`,
      `~/rocm-hipconv-pr`) directly for a split-K shard K-tile-start rotation. Found
      only hipconv's unrelated intra-workgroup wave-role barrier stagger and CK's/
      hipconv's ordinary contiguous split-K range assignment (same design MISA already
      uses via `s_gemm_k_wg_off`). See `docs/gfx1250_perf_parity_action_plan.md`
      Tier 2 item 7's correction for the full trail. Not re-opened unless a concrete
      reference implementation is found — do not implement a fabricated mechanism just
      to close this item.

## Tier 2 — medium effort (real codegen work, well-grounded)

- [ ] **Widen wrw's tile-shape space**: CK pairs small M/N tiles (64x64) with much
      larger `KPerBlock` (96/128/256, vs. MISA's 32/64) so small-output-channel wrw
      problems get fewer, cheaper main-loop iterations. FlyDSL/rocKE separately support
      tile_m as low as 16/32/48/96. Needs new VGPR/LDS-budget derivation per tile size,
      not a flag flip. Partially done via the existing `_k4x` (128x128x128) config —
      the smaller-tile-larger-K direction (64x64 with K=96/128/256) is still open.
- [ ] **Extend TDM beyond fwd/1x1** — to bwd/wrw and/or multi-tap (y/x>1) convolutions.
      Current TDM support is fwd-only, 1x1/unit-stride-only (`docs/gfx1250_wmma_layout.md`
      Phase 28-31). Would need the same hardware-OOB-zero-fill trick re-derived for
      bwd/wrw's different operand-addressing patterns.
- [ ] **Fix `script/classify_gfx1250_coverage.py`'s `gemm_n % 4 == 0` blind spot** — the
      static-analysis classifier doesn't model fwd/bwd's non-atomic epilogue's
      `gemm_n % 4 == 0` sub-constraint (needed when `wmma_n_tail` is active, from the
      vectorized 4-wide store), so it over-reports some very-small-`gemm_n` shapes as
      "supported" when they're actually unbuildable today. Confirmed via direct
      hardware testing (`fwd fp32 n=1,c=1,k=1,H=W=1760` and
      `fwd fp32 n=256,c=3,k=3,H=W=32`, both `gemm_n=k∈{1,3}`, all kernels report "not
      applicable"). Found during the benchmark exercise (2026-08-27), not yet fixed.

## Tier 3 — bigger bets (largest structural change, longest-term)

- [ ] **Stream-K / persistent-kernel design** for wrw's split-K (rocKE has a working
      reference: `helpers/streamk.py`, `helpers/persistent.py`) — a small, constant-size
      grid with an atomic tile-counter dynamically pulling work, instead of a fixed
      `grid.z` split decided before launch. Architecturally the most different, highest-
      ceiling alternative to MISA's current design; also the largest engineering lift.
- [ ] **hipconv's block-diagonal channel packing across conv groups** — fills small WMMA
      tiles when the group count is high, a structurally different way to solve
      "GEMM_M/N too small to fill a tile" than tail-masking.
- [ ] **Deeper main-loop pipelining** (N-stage, beyond MISA's current double-buffer),
      gated by LDS headroom, per FlyDSL's `num_buffers` pattern.
- [ ] **Autotuning-with-build-cache infrastructure** (FlyDSL's `conv3d_autotune.py`
      pattern) as a longer-term supplement to MISA's hand-curated `.config` files,
      specifically for shapes where static configs are demonstrably failing (wrw's tail
      cases).
- [ ] **Hardware transpose-load** (`global_load_tr_b128_v8f16`, confirmed gfx1250-valid
      via CK's own audit) to skip materializing an NHWC copy inside the hand-assembled
      kernel's own load functors — no correct reference implementation exists anywhere
      for a 16x16x32-layout WMMA yet; new engineering, not a port.

## Confirmed dead ends (kept for reference — do not re-attempt without new evidence)

- Hand-scheduled WMMA instruction interleaving: MISA's own `main_loop_interleave`
  regressed performance; CK independently built and then disabled the same idea
  (`#if 0`'d, "TODO: needs WMMA-specific rework"). Two independent confirmations.
- Adopting CK's main-loop pipeline depth: CK's is *shallower* (2-stage, no LDS
  double-buffering) than MISA's — nothing to adopt, if anything MISA is ahead.
- Redesigning the epilogue's LDS-reshuffle coalescing store: mechanically the same as
  CK's default CShuffle epilogue already.
- Switching wrw's WMMA-output accumulator away from fp32: MISA's fp32-always design
  already sidesteps CK's own documented bf16-atomic correctness bug (0.35-0.50 error at
  small output-channel counts).
- Blaming same-address atomic contention for wrw's slowness:
  `TX_VMW_ATOMIC_SETCONFLICT_STALL` measured at exactly zero (Finding 1).

## Done

- **`s_setprio(1)`/`s_setprio(0)` bracketing around WMMA issue** — Phase 32,
  `python/operations/wmma_main_loop.py`.
- **Packed 2-wide vector atomics for wrw's bf16 accumulate epilogue** — Phase 34,
  `python/operations/coalescing_store_wmma.py` (bf16 only; fp32/fp16 packed atomics not
  yet attempted — re-add as a new Tier 1/2 item if pursued).
- **wrw split-K choice cross-check against CK's occupancy formula** — Phase 33,
  `driver/igemm_wrw_gtc_driver.h`.
- **Occupancy measurement** and **rocprof extension to bwd/fwd-tail** — see Tier 0
  above, both closed 2026-08-27.
- **`V_PERMLANE_XOR_B32` swap for Phase 34's cross-lane exchange** — Phase 40,
  `python/operations/coalescing_store_wmma.py` /
  `python/igemm/igemm_wrw_gtc_wmma_nhwc.py`, closed 2026-08-27. Removes the per-iteration
  `s_wait_dscnt` and a kernel-lifetime VGPR; hardware-validated correct across 5 shapes,
  ~4-9% faster on the shapes tried (contention-noisy, directional only).
