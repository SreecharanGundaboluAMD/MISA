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
- [x] **LDS bank-conflict counters** — collected 2026-08-27 via `rocprofv3 --pmc` with
      `SQ_INST_CYCLES_LDS` (the `rocprof-compute` block 3.4 metrics returned N/A for
      gfx1250 but this direct counter is available and valid). See
      `docs/gfx1250_rocprof_profiling.md` Finding 6. **Result: LDS is essentially
      conflict-free** (1.15-1.27 cycles per LDS instruction vs. 1.0 theoretical minimum).
      LDS bank conflicts are NOT a meaningful bottleneck — the right axis for reducing LDS
      overhead is instruction COUNT reduction (via TDM, already done), not conflict
      reduction. This item is fully closed.
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

- [x] ~~**Widen wrw's tile-shape space to 64x64 with K=96/128/256**~~ — **attempted
      2026-08-27, corrected: not reachable as originally worded.** wrw's A/B addressing
      (`igemm_wrw_gtc_wmma_nhwc.py`) derives its per-thread split from
      `num_col_groups = gemm_m_per_block // gemm_k_per_block`, which requires
      `gemm_m_per_block >= gemm_k_per_block` — for a 64-wide tile, K=96/128/256 all break
      this (`64 // 128 == 0`, `utility_log2(0)` is undefined). The existing 128x128
      `_k2x`/`_k4x` configs only work because they sit at or below 128; they're not
      evidence this generalizes past `gemm_k_per_block == gemm_m_per_block`. Reaching
      CK's actual 96-256 range would need a genuine addressing redesign — Tier 3 effort,
      not a config-only change. See `docs/gfx1250_wmma_layout.md` Phase 43 for the full
      trail. **What was actually implemented**: the real ceiling this tile's mechanism
      supports, `gemm_k_per_block=64` (`num_col_groups=1`), for bf16/fp16/fp32 — new
      `_64x64_kmax` configs, folded into the master config union. LDS/VGPR budgets
      derived by hand and confirmed exactly against the compiled kernel's reported
      metadata; codegen+assembly clean (CPU-only); zero regression (git-worktree diff,
      every existing kernel byte-identical, only the one new kernel symbol appended).
      **Not hardware-validated** — implemented under an explicit no-GPU-execution
      constraint (a benchmark was running on the shared GPU); correctness is unconfirmed
      until the hardware battery every other phase in this doc has required actually
      runs. Do not treat as "done" in the sense every other closed item here is.
- [ ] **wrw addressing redesign to support `gemm_k_per_block > gemm_m_per_block`** —
      found while attempting the tile-widening above. The real blocker to CK's stated
      64x64-tile/large-K pairing (Tier 3 effort): `num_col_groups` would need to
      support values <1 (multiple threads cooperating on one K-row, or a lane owning
      more than one row) — a different per-thread addressing scheme for both A and B,
      not a config change. Not attempted.
- [x] **Extend TDM to bwd** (both operands, 1x1/unit-stride) — done 2026-08-27, Phase 42
      (`docs/gfx1250_wmma_layout.md`). Zero regression (45 configs byte-identical),
      hardware-validated correct across all 4 precisions, group>1, large-K, and a full
      K-tail-via-hardware-OOB battery. Also found and fixed a real waste in the port:
      bwd's expensive per-tap division gather is now explicitly skipped under TDM
      (fwd's own TDM never bothered, since fwd's equivalent waste is cheap). **Timing is
      a genuine trade-off, not a uniform win**: ~5-11% faster for shallow GEMM_K
      (K=128), ~5-8% *slower* for deep GEMM_K (K=1024) — the per-iteration hardware-OOB
      tensor_dim rebuild's fixed cost outweighs the one-time gather savings once K is
      deep enough. Added to the master config search (not a blanket replacement) so the
      driver picks whichever is actually faster per shape.
- [x] **Skip the per-iteration TDM tensor_dim rebuild when not needed** — implemented
      and hardware-validated 2026-08-27, Phase 44 (`docs/gfx1250_wmma_layout.md`). Guards
      the existing rebuild with `s_cmp_lt_i32 s[tdm_k_remain], {tile_dim0}` /
      `s_cbranch_scc0 {skip_label}` in both fwd's and bwd's `move_slice_window_a/b` --
      correct by construction (a materially different, previously-untested configuration
      from what Phase 31 found broken: only non-tail calls skip the update, the genuine
      tail call still gets the full rebuild). Hardware-validated: exact-multiple K, a
      full K-tail battery (K=1/31/33/63/65/100), large-K (32 iterations), group>1, all
      3 precisions tested directly (fp16/bf16/fp32), and composition with M/N-tail-via-TDM
      -- every case `valid:y`. Zero regression (every non-TDM kernel byte-identical, every
      TDM kernel shows only the expected additive guard). **Honest result: no measurable
      timing improvement** at either shallow (K=128) or deep (K=1024) GEMM_K -- kept as
      the new default anyway (correctness-neutral, no downside found), but this means
      Phase 42's large-K slowdown is **still unexplained** -- something other than this
      rebuild's SALU cost dominates at deep K. New open item below.
- [ ] **Root-cause Phase 42's still-unexplained bwd TDM large-K slowdown** — Phase 44
      ruled out the per-iteration tensor_dim rebuild as the cause (removing it produced
      no measurable timing change at K=1024 vs K=128). The ~5-8% slowdown at deep K
      relative to the non-TDM path (first measured in Phase 42) needs a fresh
      hypothesis and profiling pass (e.g. rocprof instruction-mix/cycle breakdown on the
      TDM vs non-TDM K=1024 dispatches specifically, mirroring Finding 5's methodology)
      -- not yet attempted.
- [ ] **Extend TDM to wrw and/or multi-tap (y/x>1) convolutions** — assessed 2026-08-27,
      **not attempted**. TDM support is now fwd+bwd, still 1x1/unit-stride-only.
      Structural analysis: wrw's GEMM_K (spatial, `n*ho*wo`) is the ROW axis for **BOTH**
      A and B (`num_col_groups = gemm_m_per_block // gemm_k_per_block` used identically
      for both operands, per Phase 43's investigation) -- unlike bwd, where only B needed
      the "swapped" TDM axis treatment (A's GEMM_K was already the natural contiguous
      axis). A TDM port for wrw would need the axis-swapped descriptor design (validated
      for bwd's B in Phase 42) applied to *both* operands, PLUS correctly handle
      split-K's per-shard K-slice offset interacting with the descriptor's `global_addr`
      and `tensor_dim1` -- a genuinely new interaction bwd never had to solve (bwd has no
      split-K at all). Phase 42's bwd port had a strong safety net every step of the way:
      each new formula could be cross-checked against bwd's own already-validated
      non-TDM stride/offset math before ever touching hardware. wrw's split-K
      per-shard-offset interaction has no equivalently strong existing reference to
      cross-check against, meaning more of the design would be genuinely novel and
      untested rather than a structural port. Combined with zero hardware validation
      ability this session, this was assessed as too high-risk to attempt blind --
      deferred in full (not even a partial/untested implementation) until GPU access
      returns. Multi-tap (any direction) is a separate, larger question (TDM's
      per-tap-gather-free assumption would need re-examining entirely) and wasn't
      assessed further this pass.
- [x] **Fix `script/classify_gfx1250_coverage.py`'s `gemm_n % 4 == 0` blind spot** — done
      2026-08-27 (CPU-only, no GPU execution needed — pure static analysis script).
      Added `gap_n_mod4_fwd`/`gap_n_mod4_bwd` categories, gated on `'N' in needs and
      gemm_n % 4 != 0`, for fwd/bwd only (wrw's scalar atomic epilogue has no vectorized
      grouping, unaffected). Re-ran against the full 95,066-entry corpus: 630 fwd + 336
      bwd distinct-entry-weighted shapes reclassified from "supported" to this new gap
      category (not just the tiny `gemm_n∈{1,3}` cases originally found — the fix also
      catches much larger non-multiple-of-4 values already in the corpus, e.g.
      `gemm_n`=486/510/1001 for bwd). Revises overall NHWC-assumed coverage from 99.40%
      to 98.38% — see `docs/gfx1250_wmma_coverage_gap_analysis.md`'s updated tables.
      Closing the underlying gap itself (not just the classifier) would need a new,
      finer-grained per-element epilogue masking primitive — tracked as a genuinely new
      Tier 2 item below (not yet attempted).
- [ ] **New epilogue masking granularity for `gemm_n % 4 != 0` N-tail shapes**
      (fwd/bwd) — found while fixing the classifier blind spot above. fwd/bwd's
      non-atomic vectorized-4-wide-store epilogue (`coalescing_store_wmma.py`) can't
      correctly handle a `gemm_n` that both needs N-tail relief AND isn't a multiple of
      4 — the EXEC-mask guard only checks a group's first column, so out-of-range tail
      columns within the same 4-wide group get written too. Closing this for real (not
      just in the classifier) needs a per-element (not per-4-wide-group) masking
      primitive in the shared epilogue, affecting both fwd and bwd. Real engineering
      effort (Tier C), not a config-only fix — not yet attempted.

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
