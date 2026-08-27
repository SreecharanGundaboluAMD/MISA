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
- [ ] **No LDS-bank-conflict-specific counters collected** — `rocprof-compute`'s LDS
      Utilization/Bank-Conflict-Stall-Rate metrics (block 3.4) return N/A for gfx1250 in
      the currently-installed build (confirmed 2026-08-27, see
      `docs/gfx1250_rocprof_profiling.md` Finding 5 — a tooling gap, not a usage error).
      Plain `rocprofv3 --pmc` counters under the `TX_VMW_*`/`SQC_*` blocks
      (`rocprofv3 --list-avail`) not yet tried as an alternative path to the same data.
- [x] **Run `rocprof-compute`** — done 2026-08-27, see
      `docs/gfx1250_rocprof_profiling.md` Finding 5: real instruction-mix counters
      (Wave/VALU/VMEM/LDS Instruction Mix blocks), upgrading the "address computation,
      LDS traffic, bookkeeping" claim from architectural inference to direct
      measurement. Result: non-WMMA VALU is ~50% of all instructions in BOTH fwd and
      wrw (a striking, reproducible signature), LDS traffic is the second-largest
      category (21.5% fwd, 27.0% wrw), and WMMA's instruction-count share
      cross-validates Finding 1's cycle-count share via an independent counting method.
      Needed a `PATH` workaround (the tool's internal `rocminfo` call resolves to an
      incompatible system-wide install) and an isolated venv for `analyze`'s pinned
      dependencies (not installed system-wide, to avoid touching a shared box's Python
      environment).

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
      2026-08-27 across every hipconv branch (`~/hipconv/hipconv`'s `main` plus all 12
      wgrad/splitk-named branches), plus `~/rocm-ck-hipconv`/`~/rocm-hipconv-pr`: no
      reference implementation exists anywhere.** Implemented MISA's own version of the
      idea anyway (Phase 41, `docs/gfx1250_wmma_layout.md`): a
      `gsplit_stagger` tunable emitting one `S_SLEEP_VAR` at kernel entry
      (`(bz mod 128)*~64` cycles), pure timing perturbation, no addressing/masking
      changes. Hardware-validated correct. Controlled A/B (pinned split counts via
      `IGEMM_GSPLIT_SWEEP`, 3 repeats each): **small, consistent ~3-4% win at very high
      split counts (1260, heavily over-subscribed occupancy)**; a wash at moderate
      counts (525); too noisy to read at low counts (84, GPU under heavy external
      contention at measurement time). Kept as an opt-in-only config
      (`config/igemm_wrw_gtc_gfx1250_nhwc_bf16_gsplit_stagger.config`), NOT folded into
      the master config union or enabled by default — re-measure on an idle GPU before
      considering that.

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
- **Occupancy measurement**, **rocprof extension to bwd/fwd-tail**, and
  **`rocprof-compute` instruction-mix decomposition** — see Tier 0 above, all closed
  2026-08-27.
- **`V_PERMLANE_XOR_B32` swap for Phase 34's cross-lane exchange** — Phase 40,
  `python/operations/coalescing_store_wmma.py` /
  `python/igemm/igemm_wrw_gtc_wmma_nhwc.py`, closed 2026-08-27. Removes the per-iteration
  `s_wait_dscnt` and a kernel-lifetime VGPR; hardware-validated correct across 5 shapes,
  ~4-9% faster on the shapes tried (contention-noisy, directional only).
- **wrw split-K launch stagger (`gsplit_stagger`)** — Phase 41,
  `python/igemm/igemm_wrw_gtc_wmma_nhwc.py` / `python/igemm/igemm_base.py` /
  `driver/igemm_gtc_base.h`, closed 2026-08-27. MISA's own mechanism (no verified
  hipconv reference found despite a thorough search); ~3-4% faster at very high split
  counts, inconclusive at lower counts. Opt-in config only, not a default.
