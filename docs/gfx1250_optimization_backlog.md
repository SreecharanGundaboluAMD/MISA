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

- [x] ~~**[P1] Host-precomputed Magic Division (`magic_div.h`) in hot coordinate paths**~~
      — done for fwd's GEMM_M decode (`igemm_fwd_gtc_wmma_nhwc.py`'s one-time
      `m_idx -> (n_idx, ho_idx, wo_idx)` prologue decomposition): replaced the emulated
      15-instruction `.v_u32_div_rem_vs_gfx1250` with the project's existing
      `macro_mdiv_u32_rem_vs_t` (host-computed magic multiplier + shift, 3 VALU
      instructions), magic values passed via 5 new kernargs. Implemented in a prior
      session but never actually hardware-validated (deferred). Picked back up
      2026-08-30: found `valid:n` on real hardware, root-caused to two bugs (an
      SMEM-load/`s_wait_kmcnt` ordering race reading `s_shift_pack` before its load
      landed, and a missing 4-SGPR-alignment requirement on `s_magic_ho_wo` that broke
      the master combinatorial config's assembly for some tunable combinations), both
      fixed. Hardware-validated `valid:y` across bf16/fp16/fp32/int8, every
      `wmma_m_tail`/`wmma_n_tail`/`wmma_k_tail` combination, `gemm_k_global_split`, and
      `group_count>1`, using non-power-of-2 spatial shapes (power-of-2 shapes make
      `magic=1` degenerate to a plain shift and don't actually exercise the magic
      multiply). See `docs/gfx1250_wmma_layout.md`'s Phase 60 for the full trail.
      **Not done**: group decoding's own division (still the old emulated macro, a
      separate call site not touched by this pass), `stride_h`/`stride_w` magic values
      (computed and loaded into kernargs/SGPRs but never consumed -- fwd WMMA has no
      multi-tap/strided config yet to exercise them), and bwd/wrw (not attempted).
      Performance not yet measured (this pass was scoped to correctness).
- [x] ~~**[P2] Complete Direct Store (`direct_store=1`) expansion across all master configs**~~
      — the codegen mechanism (`coalescing_store_wmma.py`'s `_emit_direct_store`) was already
      direction-agnostic and the combinatorial `_all.config` generator already covered
      `direct_store=1` for bf16/fp16/fp32 across fwd/bwd/wrw before this pass. Filled the real
      remaining gaps: int8 (`igemm_fwd_gtc_gfx1250_nhwc_int8_direct.config`, zero coverage
      anywhere before this), standalone hand-curated wrw configs (`*_direct`/`*_mntail_direct`/
      `*_ktail_direct` for bf16/fp16/fp32, wrw had none before — fwd/bwd already had them), and
      the missing 128x128 section in fwd/bwd's existing `*_direct.config` files (only 64x64 had
      `direct_store=1`). **Found and fixed a real, previously-undetected bug while validating
      this**: the driver's C++ kernel-name builder (`driver/igemm_gtc_base.h`'s
      `igemm_gtc_encode_kernel_name`, which must mirror `igemm_base.py`'s Python version byte
      for byte) never gained a `_direct` suffix when Phase 59 added `direct_store` — so
      `direct_store=1` was NEVER actually reachable through `conv_driver.exe`'s normal
      candidate-search path: `hipModuleGetFunction` either silently ran a same-named
      non-`_direct` sibling kernel (whenever one happened to coexist in the same build, e.g.
      every combinatorial `_all.config` — meaning **every prior "direct_store" benchmark
      number in `docs/gfx1250_vendor_benchmark_vs_miopen.md` was actually measuring the
      non-direct-store kernel**) or failed loudly with "named symbol not found" (every
      standalone `*_direct.config`, which have no same-named sibling). Fixed by adding the
      missing `direct_store` struct field, config-parse line, and `_direct` name-suffix line
      (matching `epilogue_lds_pad`'s exact pattern immediately above it). Hardware-validated
      `valid:y` after the fix, on odd (non-power-of-2) shapes, across bf16/fp16/fp32/int8 (fwd
      only for int8), fwd/bwd/wrw, both tile shapes, and wrw's mtail+ntail/ktail combinations.
      Zero regression on non-`direct_store` configs (the fix only appends a suffix when
      `direct_store=1`, which was never true for them). **Not done**: re-running the vendor
      benchmark now that direct_store is actually reachable (the existing "direct_store wins
      on N shapes" narrative needs to be re-measured, not trusted) — flagged as a new,
      separate follow-up, not silently absorbed into this item's scope.
- [x] ~~**Pre-existing bug: bwd's master `_all.config` fails to assemble for
      bf16/fp16/fp32**~~ — **FIXED 2026-09-01, hardware-validated.** Root-caused via
      `git log -S` bisection: commit `78af72c` ("Port saddr_global_load from fwd to bwd
      and wrw") accidentally DELETED two lines from `move_slice_window_b_functor`'s
      `tdm_global_load` branch while splicing in its own new `elif
      tunable.saddr_global_load` clause immediately after — the final
      `s_or_b32 s[s_tdm_g1_b(3)], ...` (combining tensor_dim1's hi16 with the
      compile-time tile_dim0 constant) and the `outer._emit_front(f"{skip_label}:")`
      label definition itself. This wasn't a label-uniqueness collision as first
      suspected — the label was simply never emitted anywhere, so ANY
      `tdm_global_load=1` bwd kernel (standalone or master-combined) has failed to
      assemble since that commit landed (2026-08-31 22:57), not just combined master
      configs; this session's master-config rebuild was apparently the first rebuild
      attempt since. Restored both deleted lines in their original position (before the
      new saddr `elif`, matching A's identically-structured, still-correct
      `move_slice_window_a_functor` sibling). Hardware-validated `-V 1` after the fix:
      bf16/fp16/fp32, a genuine K-tail shape (`gemm_k=100`, not a multiple of
      `gemm_k_per_block`, actually exercising the rebuilt `s_tdm_g1_b(3)` register --
      not just an assembly check), a multi-K-block+tail shape (`k=200`, 6 full blocks +
      partial), and the `tdm_direct` (TDM + `direct_store`) combo. All three
      directions' master configs (all three precisions) now build cleanly.
- [x] **[P3] 32-bit SADDR base offsets for in-loop global loads**
      — Replace 64-bit carry-chain address stepping in inner K-loops with 32-bit byte offset
      VGPRs + SADDR base SGPRs (`global_load_dwordx4 vdst, v_off, s_p_base offset:N`).
      Saves 1 VGPR and 1 VALU carry op per address step across all directions.
      **Fwd pilot done** (new `saddr_global_load` tunable, `igemm_fwd_gtc_wmma_nhwc.py`):
      the default (non-async, non-TDM) global-load path for A/B now shares
      `async_global_load`'s existing 32-bit-offset address computation (a ready-made
      pattern already in the same file), only changing the actual load instruction's
      addressing operands (SADDR + 32-bit offset instead of a 64-bit VADDR pair) --
      confirmed via `llvm-mc -mcpu=gfx1250` that plain `global_load_dwordx4` already
      supports this addressing mode (standard GLOBAL_* "GVS" form, ISA doc §5884), not
      something new to discover. Mutually exclusive with `async_global_load`/
      `tdm_global_load`/`main_loop_interleave`/`gemm_k_global_split`/`row_repeat_a/b>1`
      for this pass (narrowest-slice discipline, same as every other addressing
      mechanism in this file). Hardware-validated `valid:y` across bf16/fp16/fp32/int8,
      both tile shapes, `wmma_m_tail`+`wmma_n_tail`, `wmma_k_tail` (with a genuinely
      active tail boundary), and `group_count>1`. Confirmed ~4 fewer VGPRs
      (`.amdhsa_next_free_vgpr` 252→248 on the 128x128 tile) and zero regression
      (`saddr_global_load=0`'s generated `.s`/`.inc` byte-identical to before). Also
      fixed the new tunable's kernel-naming into BOTH Python and C++ driver-side name
      builders from the start (see the P2 entry above for why that specific gap is a
      known, previously-costly failure mode in this codebase). See
      `docs/gfx1250_wmma_layout.md`'s Phase 62.
      **Correction (2026-09-01, Phase 67)**: "bwd/wrw not done" above was stale --
      `saddr_global_load` had already been fully ported to bwd's and wrw's A+B operands
      (commit `78af72c`, landed before Phase 67 started) and was hardware-validated at
      that time. The only real remaining gap was config-coverage: no fp32 standalone
      `*_saddr.config` for bwd/wrw (added, `gemm_k_per_block=4` +
      `lds_double_buffer=1`), and bwd's/wrw's existing bf16/fp16 saddr configs were
      never folded into their master `_all.config` unions (fwd's was) -- re-ran
      `script/build_gfx1250_master_configs.py --write` (pure glob-and-union, no code
      changes) to fold all of them in. Hardware-validated `valid:y` for every
      newly-covered saddr section (bwd/wrw x bf16/fp16/fp32, both standalone and from
      inside the regenerated master configs). Performance still not separately
      measured. See `docs/gfx1250_wmma_layout.md`'s Phase 67.
      **Follow-up (2026-09-01, Phase 68)**: `saddr_global_load` added to
      `script/generate_all_configs.py`'s combinatorial `FLAGS` (previously only ever
      tested in its own single-feature file, never combined with `direct_store`/
      `wmma_m_tail`/`wmma_setprio`/`lds_double_buffer`/etc.). This surfaced a genuine,
      previously-unknown correctness bug:
- [x] **fwd `saddr_global_load` + `wmma_n_tail` produces `valid:n`** — found
      2026-09-01 (Phase 68) the moment the combinatorial generator tried this pair for
      the first time. Confirmed real (not a shape artifact): the identical exact-fit
      shape passes `valid:y` with plain `wmma_n_tail` alone AND with
      `saddr_global_load` + `wmma_m_tail` alone -- only the `saddr`+`n_tail` pairing on
      fwd fails. bwd's identical pairing hardware-validates fine (wrw structurally can't
      combine them: wrw requires `gemm_k_global_split` alongside any tail flag, which
      `saddr_global_load` already excludes). Not root-caused (plausibly fwd's B-operand
      N-boundary address computation not accounting for `saddr`'s different addressing
      path) -- excluded from the generated corpus via a new `is_valid()` rule
      (`if sa and nt and direction == 'fwd': return False`) rather than left broken in
      the searched space. Root-causing and fixing this is real, separate follow-up work.
      See `docs/gfx1250_wmma_layout.md`'s Phase 68.
- [ ] **`script/build_and_filter_configs.py`'s per-section failure isolation doesn't
      recognize the `register index is out of range` error class** — found 2026-09-01
      (Phase 68) while validating the saddr combinatorial expansion above. When a
      multi-section per-tile combo file fails to assemble as a whole, this script tries
      to identify and drop just the offending section(s) by pattern-matching a
      `kernel 'NAME'` string in the compiler output -- but a real, separate,
      PRE-EXISTING failure class (`<instantiation>:N:M: error: register index is out of
      range`, apparently a VGPR/register-macro budget overflow in some large multi-tap
      magic-division / tail-masking combinations) doesn't match that pattern, so the
      script falls back to "keeping all sections," silently leaving a config file that
      does NOT actually build via `igemm_codegen.py`. Confirmed via git-stash A/B that 7
      of 27 per-tile combo files (`bwd` bf16/fp16 128x128; fp32 128x128/64x64 for all
      three directions; `fwd` bf16/fp16 64x128) hit this and are PRE-EXISTING, not
      introduced by the saddr work. Not a silent correctness risk today (the benchmark
      script's `build_one_config` already handles a build failure gracefully and just
      skips that candidate) but it does silently narrow the searched corpus for those
      (direction, precision, tile) combinations, and the underlying VGPR/register-budget
      bug itself is still unknown. Two separate follow-ups here: (1) teach
      `extract_failing_sections()` to also parse this error format so it can actually
      isolate and drop just the bad sections, (2) root-cause the register-budget
      overflow itself. See `docs/gfx1250_wmma_layout.md`'s Phase 68.
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
      **Hardware-validated 2026-08-28**: exact-fit shape (n=4,c=64,H=32,W=32,k=64,
      gemm_k=4096, an exact multiple of 64) run standalone (single-config build, not the
      master union) with `-V 1` — `valid:y` for all three precisions (bf16/fp16/fp32).
      K-tail/M-tail were not exercised here (this narrow single-config build has no tail
      variant; tail handling lives in separate config sections combined at the master
      level) but the new tile's core WMMA mechanism is now confirmed correct on real
      hardware, same footing as every other closed item in this doc.
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
- [x] **Extend TDM to wrw, both operands, split-K aware** — done 2026-08-27 (Phase 45),
      hardware-validated. TDM support is now fwd+bwd+wrw, still 1x1/unit-stride-only.
      Both wrw operands needed the axis-swapped descriptor (GEMM_K is the row axis for
      BOTH A and B in wrw, unlike bwd where only B needed it). Split-K's per-shard offset
      interaction turned up one real, driver-side gap: `gemm_k_global_split`'s
      `karg.gemm_k_per_wg` computation rounds gemm_k UP to the next multiple of
      `gemm_k_per_block`, which TDM (no `wmma_k_tail` clamp, asserted mutually exclusive)
      has no way to correct for -- fixed by requiring an exact-multiple gemm_k whenever
      TDM is combined with split-K specifically (non-split TDM still gets the full
      K-tail-via-hardware-OOB relaxation). See `docs/gfx1250_wmma_layout.md`'s Phase 45
      for full validation battery and the (unrelated to this item, but discovered during
      it) `conv_driver.exe` mode-string gotcha. Multi-tap (any direction) remains
      unaddressed -- TDM's per-tap-gather-free assumption would need re-examining
      entirely and wasn't assessed this pass.
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
      The underlying gap itself (not just the classifier) is now closed too — see
      Phase 51 below. `script/classify_gfx1250_coverage.py`'s `gap_n_mod4_*`
      categories are now stale (would still report these shapes as unsupported) but
      were not updated in this pass — a small follow-up if the coverage doc is
      regenerated again.
- [x] **New epilogue masking granularity for `gemm_n % 4 != 0` N-tail shapes**
      (fwd/bwd/wrw) — done 2026-08-27 (Phase 51), hardware-validated. Found again (as
      "not applicable" on 2/20 shapes) via a diverse gfx950-baseline benchmark sweep.
      `coalescing_store_wmma.py`'s non-atomic epilogue now decomposes its vectorized
      store into per-element masked scalar stores via a **runtime** branch (only taken
      when `gemm_n` isn't actually a multiple of `vector_write_out` this pass) — a
      compile-time-only version was tried first and measured to regress an
      already-working exact-multiple-of-4 shape ~24%, which the runtime branch avoids
      entirely. `tunable_is_valid()`'s `gemm_n % 4 == 0` restriction is lifted in all
      three drivers. Scoped to the standard f32-accumulate case only —
      `wmma_acc_f16`/`bf16acc`'s packed-2-elements-per-register layout is a genuinely
      different addressing scheme, not audited (no existing config combines the two;
      now a hard `igemm_base.py` assert instead of a silent gap). See
      `docs/gfx1250_wmma_layout.md`'s Phase 51.
- [x] **Add a 32x32 bwd macro-tile for small-gemm_m/n occupancy** — done 2026-08-27
      (Phase 46), hardware-validated, all 3 precisions. Closes ~1.5x of a ~2.6x gap vs.
      CK found on a small-spatial/large-channel bwd shape. See
      `docs/gfx1250_wmma_layout.md`'s Phase 46 for the full derivation. Two follow-ups
      opened by this work, both below.
- [ ] **Generalize row_repeat_a/b for block_size > macro_tile in BOTH dimensions at
      once** — opened 2026-08-27 (Phase 46), downgraded in priority 2026-08-27 (Phase
      48): split-K (below) fully closed the gap on the Phase 46 target shape without
      needing this, so this is no longer required to match CK on *that* shape. Still a
      real, independently-motivated gap for shapes where split-K doesn't apply as
      cleanly (very small `gemm_k` with no useful divisor for any split count > 1, or
      shapes small enough that atomic-add overhead outweighs the occupancy win). The
      existing 128x64/64x128 asymmetric tile entries in `wmma_mapping.py` already
      generalize `block_size` exceeding `gemm_m_per_block` OR `gemm_n_per_block` (one
      dimension at a time, via `row_repeat_a`/`row_repeat_b`). A tile like
      32x32-with-2-waves (CK's own tuned instance for the Phase 46 shape uses
      `BlockSize=64` i.e. 2 waves per 32x32 tile, `MRepeat=2,NRepeat=1` per wave,
      splitting N across the 2 waves) needs `block_size` to exceed the macro-tile in
      **both** dimensions simultaneously relative to a single wave's own coverage — a
      case the existing row_repeat generalization doesn't cover; needs a new
      thread-folding scheme (2 threads cooperate per LDS row), not a simple extension.
      Real engineering effort (Tier C), touches `python/operations/wmma_mapping.py`'s
      wave-grid math and likely each of fwd/bwd/wrw's global-load thread-mapping.
- [x] **bwd has no `gemm_k_global_split` (split-K) support at all** — done 2026-08-27
      (Phase 48), hardware-validated. Ported from wrw's pattern, adapted for bwd's
      swapped GEMM roles (A's shard offset is a flat byte add, B's is a stride-multiply
      — mirror image of wrw's A/B). On the Phase 46 target shape: `valid:y`,
      0.017-0.022ms / 9.1-12.1 TFLOPS at `gkgs[8]` (bf16/fp16) — **matches or exceeds
      CK's own reported 0.0215ms/9382 GFLOPS on this exact shape**, fully closing the
      gap Phase 46 set out to close. fp32's driver-heuristic split count is
      over-aggressive (fp32's `gemm_k_per_block=4` is much finer-grained than
      bf16/fp16's 32) — `IGEMM_GSPLIT_SWEEP` finds a much better split count manually;
      a follow-up item (below) to make the heuristic `gemm_k_per_block`-aware.
      Not combined with `wmma_k_tail` or `tdm_global_load` in this pass (both
      explicitly asserted against) — see `docs/gfx1250_wmma_layout.md`'s Phase 48.
- [x] **bwd/fwd split-K driver heuristic over-splits for fp32** — fixed 2026-08-27
      (Phase 50), hardware-validated. Root cause confirmed: the `~512-total-workgroups`
      heuristic in `driver/igemm_bwd_gtc_driver.h`/`igemm_fwd_gtc_driver.h` targets a
      *split count* independent of `gemm_k_per_block`, so fp32 (K=4, 8x finer than
      bf16/fp16's K=32) got 8x-too-fine real shards for the same target. Fixed by
      clamping `target_splits` through a new shared `igemm_gemm_k_global_split_cap`
      helper (`driver/igemm_gtc_base.h`) that bounds K-elements-per-shard to a floor
      (32) regardless of `gemm_k_per_block` — bwd's Phase 46 target shape now
      auto-picks `gkgs[8]` (6.1-6.7 TFLOPS, ~2x better than the old `gkgs[64]`'s 3.3
      TFLOPS) with no manual `IGEMM_GSPLIT_SWEEP` needed; bf16's own already-good
      `gkgs[8]` choice is unaffected (the cap agrees with it exactly). Not a full
      per-precision launch sweep (wrw's fuller ternary-search approach remains a
      further option if the last ~15% to the hand-tuned peak ever matters) but a
      principled, non-overfit fix. See `docs/gfx1250_wmma_layout.md`'s Phase 50.
- [x] **split-K heuristics had no upper sanity bound on split count** — found and
      fixed 2026-08-27 (Phase 50), hardware-validated. An extreme wrw shape
      (`n=256,c=32,H=449,W=449,k=3`, tiny gemm_m, huge gemm_k) made wrw's existing
      real-launch ternary search try a **135000-way split**, taking minutes to time
      just that one candidate (massive atomic-add contention). Fixed via the same
      shared `igemm_gemm_k_global_split_cap` helper above (absolute ceiling 4096,
      comfortably above the largest previously-useful split count measured, 1260) —
      applied to bwd/fwd's heuristic target and to wrw's enumerated divisor list (plus
      its `wrw_reduction_kernel` workspace allocation, previously sized for the
      uncapped `num_k_blocks`, a latent multi-GB-allocation risk for this shape class).
      `IGEMM_GSPLIT_SWEEP`'s manual override remains intentionally uncapped. See
      `docs/gfx1250_wmma_layout.md`'s Phase 50.
- [x] **bwd `group>1` returns `valid:n`** — fixed 2026-08-27 (Phase 47). Root cause: a
      copy-paste-from-fwd bug in the weight operand's group-offset computation (used
      `gemm_n` instead of bwd's own `gemm_k` — see `docs/gfx1250_wmma_layout.md`'s Phase
      47). One-line fix, hardware-validated across group=1/2/4, all precisions, every
      tile.
- [x] **fwd has no `gemm_k_global_split` (split-K) support at all** — done 2026-08-27
      (Phase 49), hardware-validated. Simpler port than bwd's: fwd's A and B BOTH have
      GEMM_K as their own contiguous axis, so both shard offsets are flat adds folded
      once into the existing group-offset computation — no per-tap address change
      needed. bf16/fp16 both `valid:y` at `gkgs[16]`, 0.017-0.018ms/~11-12 TFLOPS — a
      ~3x speedup over the non-split 128x128 baseline on the session's target shape.
      Not combined with `wmma_k_tail`, `tdm_global_load`, `async_global_load`, or
      `main_loop_interleave` in this pass (all asserted against) — see
      `docs/gfx1250_wmma_layout.md`'s Phase 49.
- [x] **int8/int4 `gemm_k_global_split` atomic epilogue** — code-level fix landed
      2026-08-28 (Phase 57). `coalescing_store_wmma.py`'s atomic path now emits
      `global_atomic_add_u32` for int8/int4 (was always `global_atomic_add_f32`, which
      bit-reinterpreted the genuine int32 accumulator as float — only coincidentally
      correct for small non-negative sums). The blocking assert in `igemm_base.py` was
      removed accordingly. NOT hardware end-to-end validated with genuinely signed /
      large-magnitude int8 data — deprioritized, int8/int4 is not a current focus. See
      `docs/gfx1250_wmma_vgpr_msb_wip_status.md`'s Phase 57.

## Tier 3 — bigger bets (largest structural change, longest-term)

- [x] **VGPR-MSB register-allocator support** (`S_SET_VGPR_MSB`, up to 1024 VGPRs/wave)
      — DONE (Phase 54, 2026-08-28). `v_c` (the accumulator) now lives in a second,
      independently-addressed register bank via `wmma_acc_high_bank=1`. Two real bugs
      found and fixed along the way, both hardware-validated: (1) `dst` MSB left at
      bank 1 after every WMMA burst with nothing resetting it, corrupting every
      bank-0 VGPR write in between bursts; (2) `coalescing_store_wmma.py`'s per-row
      scatter advance (`v_add_u32 v_tmp1, stride, v_tmp1`) put the running address in
      the instruction's VSRC1 slot, which was still gated by `src1=1` (active for the
      neighboring `ds_write`'s `v_c` DATA operand) — silently read from bank 1
      instead of bank 0. `valid:y` across fwd bf16/fp16/int8/fp32 and bwd bf16, both
      epilogue implementations, at 128x128. See
      `docs/gfx1250_wmma_vgpr_msb_wip_status.md` for the full writeup.
- [~] **Bigger WMMA macro-tile via the chunked epilogue** (Phase 52/53/55/56) —
      CORRECTNESS DONE, PERFORMANCE NOT YET A WIN. The VGPR wall is closed (VGPR-MSB
      above): fwd 256x256 and bwd 256x128 build with full F32 accumulate
      (`wmma_acc_high_bank=1`, replacing the precision-losing `wmma_acc_bf16`
      workaround Phase 53 needed and which still didn't fit even with packing). Also
      found and fixed a genuine, separate, pre-existing `wmma_epilogue_chunked` bug
      along the way: the chunked scatter derived `wave_idx`/`lane_sub` directly from
      `v_gemm_im` without masking out the block-offset the prologue permanently folds
      into it, so every workgroup after the first (`grid_x>1`) scattered into the
      wrong LDS bytes entirely (invisible for `bx=0`, since `block_m_off=0` there).
      Found via a position-fingerprint diagnostic (`CHUNK_FINGERPRINT=1`) and fixed by
      masking to tile-local range, matching the unchunked path's existing discipline.
      **Hardware-validated `valid:y`** for both directions, single- and
      multi-workgroup, multi-K-block, multi-N-block. See Phase 55 in
      `docs/gfx1250_wmma_vgpr_msb_wip_status.md`.

      **Phase 56: tried to close the performance gap, couldn't yet.** Same-run
      comparison (256x256 vs. existing 128x128, both in one `conv_driver.exe`
      invocation) showed the new tile is the SLOWEST candidate at every scale tried —
      ~2-3x slower at small/medium problem sizes, still ~10-15% slower even at the
      largest tested (256x256 spatial, 256 channels). Tried `lds_double_buffer=1` (no
      consistent benefit, costs the full 64KB LDS budget) and packed accumulate
      (`wmma_acc_bf16`, drops VGPR/wave from 512 to 384 — consistent ~10-12%
      improvement over the plain entry, added as a separate opt-in
      `*_256x256_bf16acc.config`/`*_256x128_bf16acc.config` file per this project's
      accumulate-width-variant convention) — **neither closes the gap**. VGPR-driven
      occupancy explains part of it but not all; remaining candidates (chunked
      epilogue's per-group barrier overhead vs. 128x128's one-shot epilogue, or
      fewer/bigger workgroups reducing wave-level parallelism independent of per-wave
      VGPR count) not yet investigated — would need direct rocprof occupancy/
      instruction-mix measurement on a confirmed-idle GPU to separate. See Phase 56 in
      `docs/gfx1250_wmma_vgpr_msb_wip_status.md` for full numbers.

      Untested: int8/fp16/fp32 at 256-size tiles (only bf16 verified), wrw direction —
      **do not extend to these until the performance gap above is understood**, since
      that would just propagate a config that isn't winning yet.
- [~] **Stream-K / persistent-kernel design** for wrw's split-K — **Approach A implemented
      and hardware-validated 2026-08-28** (with the user supervising, after an earlier
      unsupervised pass deliberately stopped at the design stage — see
      `docs/gfx1250_streamk_design.md` for the full history, design rationale, two real
      bugs found and fixed, and the validation table). One-paragraph summary: instead of
      wrw's `gemm_k_global_split` launching `grid.z == chosen split count` (up to
      thousands of tiny simultaneously-dispatched workgroups — the documented
      contention-sensitivity culprit in `docs/gfx1250_vendor_benchmark_vs_miopen.md`),
      `wrw_streamk=1` (new tunable, `config/igemm_wrw_gtc_gfx1250_nhwc_bf16_streamk.config`)
      launches a small, occupancy-sized persistent grid whose workgroups dynamically claim
      K-shards (one `gemm_k_per_block` each) from a per-output-tile atomic counter in a
      bounded loop — no per-split-count search needed at all. Hardware-validated: single
      shard, degenerate 1:1 (small and large shard counts), genuine multi-claim at an exact
      multiple (512 shards / 64 persistent workers, 8 claims each), and multi-claim with a
      non-exact-multiple tail (400/64, 7 claims) all pass cleanly, repeated with fresh
      random data. Zero regression on every existing kernel.
      **Performance: found ~4-4.4x SLOWER, root-caused, and fixed same day (2026-08-28)**
      — now near parity. First measurement (`STREAMK_DEBUG=1`) found the persistent
      grid.z wasn't actually small (reused the existing splits-heuristic formula, which
      scales *up* as output-tile count shrinks — launched 1024 workgroups for a
      single-tile shape, MORE than `_gsplit`'s own chosen 315). Fixed grid.z sizing to a
      direct `num_cu`-based target (raw CU count, matching rocKE's
      `compute_streamk_grid_size`) — that alone barely helped (0.297ms→0.319ms), because
      the real cost is *shard count* (atomic-claim + LDS-broadcast + double-barrier per
      shard), not worker count. Fixed by also making shard granularity coarser (was fixed
      at exactly one `gemm_k_per_block`; now targets ~4 claims/worker capped at 256 total
      shards, snapped to an exact divisor of `num_k_blocks`) — **closed the gap from
      ~4.1x/4.4x slower to ~1.05x (near parity) / ~1.3x slower** on the same two real
      shapes, with zero device-code changes (host-side sizing only). Re-validated
      correctness on every prior scenario (still all `valid:y`). Full numbers, both fixes,
      and the reasoning for why grid.z alone wasn't enough are in
      `docs/gfx1250_streamk_design.md`'s "Performance fix" section.
      **Re-measured on a confirmed-idle GPU + found/fixed a real additional tuning bug
      (2026-08-28, same day)**: idle-GPU numbers matched the earlier contended ones almost
      exactly (confirms the original finding/fix weren't contention artifacts). Testing a
      third, multi-output-tile shape (`512,30,40,128`, 4 output tiles) found it at ~1.73x
      slower — worse than the two already-fixed shapes — because the persistent-worker
      target was being divided across output tiles (matching rocKE's own formula shape),
      starving multi-tile shapes of workers and forcing more loop iterations per worker.
      Exposed the sizing constants as env-var overrides
      (`STREAMK_BLOCKS_PER_CU`/`STREAMK_DIVIDE_BY_TILES`/`STREAMK_CLAIMS_PER_WORKER`/
      `STREAMK_MAX_SHARDS`, mirroring `IGEMM_GSPLIT_SWEEP`) and found
      `STREAMK_DIVIDE_BY_TILES=0` (don't divide — every tile independently targets the
      full `num_cu` worker count) strictly better on every shape tested — made it the new
      default. **All three measured shapes now sit at a consistent ~1.03-1.09x slower**
      (was 1.04x/1.31x/1.73x). Full numbers in `docs/gfx1250_streamk_design.md`.
      **Contention-resilience measured rigorously (2026-08-28, same day) — real
      trade-off found, not a clean win**: synthesized contention (background fwd conv,
      confirmed via `rocm-smi`). First two attempts used sequential-block sampling (all
      `_gsplit` repeats, then all `wrw_streamk` repeats) and gave *opposite* conclusions
      between an 8-repeat/2-shape run and a 10-repeat/4-shape run — sequential blocks
      confound "design A vs B" with "contention level during time window 1 vs 2." Fixed
      by **interleaving** (alternate A/B every repeat) across 3 shapes, 10-12 repeats
      each, both sides using their own best-of-search result. Result: **`wrw_streamk`'s
      timing is dramatically more consistent** (CV well under 1%, essentially
      deterministic, in 2 of 3 shapes) **than `_gsplit`'s** (CV 36-63%, including
      individual outlier runs 4-6x off its own mean) — strong, reproducible support for
      the core hypothesis. **But `wrw_streamk`'s mean is worse in all three shapes**
      (1.06x-3.20x slower) — contention hurts its absolute throughput more than
      `_gsplit`'s, even though it hurts `_gsplit`'s *predictability* far more. Separately
      (reproduced across all attempts): `_gsplit`'s own ternary search picks different
      split counts across otherwise-identical repeats under contention — an instability
      `wrw_streamk` cannot have by construction (no search at all). Tried reducing
      `STREAMK_MAX_SHARDS` as a contention-aware retune (hypothesis: less atomic/barrier
      overhead) — made it strictly worse (1.24ms→3.66ms as shards dropped 256→32);
      **don't repeat that experiment**. Full numbers and methodology discussion in
      `docs/gfx1250_streamk_design.md`.
      **Config coverage + M/N-tail masking done (2026-08-28, same day)**: added
      64x64x32 to the bf16 config, plus new fp16/fp32 configs (both tile sizes, matching
      `_gsplit`'s existing coverage; int8 skipped, not a current priority). M/N-tail
      masking needed zero code changes (the tail flags are prologue-computed constants,
      already wired into the epilogue call) — 128x128x32 doesn't fit (already at the
      256-VGPR wall, tail flags push it over), but 64x64x32 has headroom and
      hardware-validates cleanly (M-only, N-only, both, combined with multi-claim) — new
      `config/igemm_wrw_gtc_gfx1250_nhwc_bf16_streamk_mntail.config`. All still opt-in,
      not in the master config union.
     **Idle-GPU max_iters sweep: persistence is strictly worse (2026-08-28)**. Targeted
     sweep varying `STREAMK_GRID_Z` (new env var) independently to control `max_iters`
     directly, with `-V 1` verification, on 3 shapes (GPU confirmed idle — `rocm-smi`'s
     100% is a driver bug). Result: **per-claim overhead scales linearly with `max_iters`**:
     max_iters=1→0.068ms, 2→0.096ms, 4→0.161ms, 8→0.300ms (shape 128,30,40,128). Each
     persistent-loop iteration pays a ~10-instruction cross-wave synchronization sequence
     (atomic_add + ds_write + 2×barrier + ds_read + readfirstlane) that doesn't exist in
     the static `gsplit` design (shard index = `blockIdx.z`, two scalar instructions). The
     best streamk config is always `max_iters=1` — functionally identical to `gsplit` but
     with extra atomic-claim + barrier overhead per shard. **Conclusion: the persistent-grid
     design is fundamentally incompatible with this kernel's per-claim synchronization
     cost on an idle GPU.** See `docs/gfx1250_streamk_design.md`'s "max_iters sweep" section.
     **Critical comparison gap found (2026-08-28)**: the "~1.03x slower" figure above was
     vs `gsplit`'s atomic (`gkgs`) candidates only. Against `_gsplit`'s actual best
     (`wsred_gkgs` — non-atomic disjoint-workspace + separate reduction kernel, Phase 35),
     `wrw_streamk` is **2.5-3.8x slower**. The atomic epilogue (shared by both
     `wrw_streamk` and `gkgs`) is the dominant cost, not split-count search or grid sizing.
     The natural next step is **Approach C** (persistent grid + `wrw_reduction_kernel`'s
     non-atomic epilogue) — the only combination that could plausibly beat `wsred` on both
     predictability AND throughput. Atomic-epilogue streamk tuning is a dead end.
     **Still open, reprioritized**:
     1. **Approach C** (persistent grid + non-atomic `wrw_reduction_kernel` epilogue) —
        now highest priority, the only path to beating `wsred` on throughput while keeping
        Stream-K's predictability. Original engineering (rocKE's Reduction-strategy
        reference is incomplete), but `wrw_reduction_kernel` itself is already shipped and
        hardware-validated (Phase 35).
     2. Real (not synthesized) contention data to confirm the predictability trade-off.
     3. Master config union decision — deferred pending Approach C.
- [ ] **hipconv's block-diagonal channel packing across conv groups** — fills small WMMA
      tiles when the group count is high, a structurally different way to solve
      "GEMM_M/N too small to fill a tile" than tail-masking.
- [ ] **Deeper main-loop pipelining** (N-stage, beyond MISA's current double-buffer),
      gated by LDS headroom, per FlyDSL's `num_buffers` pattern.
- [ ] **Autotuning-with-build-cache infrastructure** (FlyDSL's `conv3d_autotune.py`
      pattern) as a longer-term supplement to MISA's hand-curated `.config` files,
      specifically for shapes where static configs are demonstrably failing (wrw's tail
      cases).
- [x] ~~**Hardware transpose-load for the WMMA B-operand (bwd) and A+B operands (wrw)**~~ —
      **DONE 2026-09-01 (Phase 63 bwd, Phase 64 wrw), hardware-validated, real measured
      win, tail-mask/split-K composition confirmed free.** Built a standalone hardware
      probe to reverse-engineer `ds_load_tr16_b128`'s exact per-lane addressing/lane-remap
      semantics (the ISA doc's diagram isn't text-extractable), confirmed the mechanism
      (8-lane-group in-flight transpose), derived and hardware-confirmed the exact address
      formula MISA's operands need, and wired it into bwd's `shared_load_b_functor` and
      wrw's `shared_load_a_functor`/`shared_load_b_functor` behind a new opt-in
      `ds_load_tr_b` tunable — 2 native instructions replace the entire ~160-instruction
      manual read+pack loop per wave_repeat step, per operand.
      **Measured** (rocprofv3, same shape before/after): bwd `SQ_INSTS_ALL` -43.1%
      (~28% faster wall-clock); wrw `SQ_INSTS_ALL` -57.1% (~17.4% faster wall-clock,
      larger win since both operands benefit). Hardware-validated `valid:y`: bwd and wrw,
      bf16/fp16, both tile shapes, multi-K-block, multi-block grids, `group_count>1`;
      wrw's `gemm_k_global_split=1` (its primary path, both tile shapes); bwd's
      `wmma_m_tail`+`wmma_n_tail`+`wmma_k_tail` combined and `wmma_n_tail` alone; wrw's
      `wmma_k_tail` and `wmma_m_tail`+`wmma_n_tail` combined. Zero regression (default
      config's generated `.s` has zero `ds_load_tr16_b128` occurrences). Kernel naming
      synced in both Python and C++ from the start. Tail-mask/split-K composition
      required NO new masking code at all -- code review confirmed all masking happens
      at LDS WRITE time (global-load `v_flag` + shared-store tail-dword AND-mask), not
      LDS READ time, so the transpose-load just reads whatever's already correctly-masked
      in LDS; the four mutual-exclusion asserts added in Phase 63 were removed. See
      `docs/gfx1250_wmma_layout.md`'s Phase 63/64 for the full derivation, probe
      methodology, and address formula.
      **Not done, tracked as follow-ups**:
      1. **bwd's own `gemm_k_global_split` could not be hardware-validated** — confirmed
         (via testing the unmodified pre-Phase-63 codebase too) that bwd's split-K driver
         path currently fails with an illegal-memory-access on every shape/split-count
         tried in this environment, a real pre-existing regression unrelated to this
         work (see the new backlog entry below). No code-level reason to expect it
         behaves differently once that's fixed (wrw's structurally similar split-K
         composes correctly).
      2. ~~**Full benchmark-suite sweep / folding into the master config union**~~ —
         **DONE 2026-09-01 (Phase 68).** Turned out `ds_load_tr_b` was reachable by
         ZERO config files in the entire repo (found via fresh rocprofv3 profiling that
         re-confirmed the win: bwd -35%/wrw -62% instructions on a fixed shape,
         real wall-clock gains too). Rather than adding more standalone configs,
         promoted it to a smart DEFAULT (`1` whenever `direction in ('bwd','wrw')` and
         `precision in ('fp16','bf16')`, unless `wrw_streamk`) in `igemm_base.py` --
         every existing bwd/wrw fp16/bf16 config now gets it automatically, matching
         Phase 64's wait-batching precedent. Validated broadly first (async, dbuf,
         interleave, ldspad, setprio, tdm(+direct), saddr, gsplit(+setprio),
         local_prefetch_num=2(+bf16acc), group_count>1 -- all `valid:y`). Also had to
         fix `driver/igemm_gtc_base.h` (the C++ driver's independent tunable parser) to
         mirror the identical default -- hit the known kernel-naming-desync failure
         mode (`hipModuleGetFunction`/"named symbol not found") when the Python and C++
         defaults briefly disagreed. See `docs/gfx1250_wmma_layout.md`'s Phase 68.
      3. ~~**`GLOBAL_LOAD_TR16_B128`**~~ — **investigated 2026-09-01, REJECTED with
         quantified reasoning (not merely deferred).** Hardware mechanism confirmed
         identical to the LDS variant via a second standalone probe. But the actual
         instruction count left to save turned out tiny: post-Phase-63, B's entire
         remaining global-load+LDS-store pipeline (everything this would skip) is only
         **8 instructions** per main-loop iteration (4 `global_load_dwordx4` + 4
         `ds_write_b128`, measured directly from the generated 128x128x32 bf16 kernel).
         Preserving the existing double-buffer's latency-hiding for a direct-to-global
         version needs either (a) a register-copy into `v_b`'s static destination each
         iteration -- **32 `v_mov_b32` instructions** for this tile shape, a net
         **+24/iteration regression**, or (b) main-loop-body duplication to avoid the
         copy -- large, invasive, touches every interacting main-loop mode, for an
         8-instruction prize. Neither is worth it. Not implementing. See
         `docs/gfx1250_wmma_layout.md`'s Phase 64 (section "4b") for the full numbers
         and reasoning -- keep this closed unless a future tile shape or main-loop
         redesign changes the underlying instruction-count math.
- [x] ~~**fp32 B-operand LDS-load overhead reduction / batch the per-element
      `s_wait_dscnt 0x0`**~~ — **DONE 2026-09-01 (Phase 64), hardware-validated.**
      `shared_load_b_functor`'s (bwd) and `shared_load_a_functor`'s (wrw) manual-loop
      fallback (used by fp32 always, since no `ds_load_tr16_b128` variant exists for it,
      and by int8 / any `ds_load_tr_b`-unset fp16/bf16 config) now batches reads up to
      the scratch buffer's real capacity (`chunk_num_dwords`: 16 for fp16/bf16/int8, 4
      for fp32) before a single wait, instead of one wait per vgpr index -- up to 8x
      fewer wait instructions. Not gated behind an opt-in flag (changes generated code
      for every existing fp32/int8 config and any `ds_load_tr_b`-unset fp16/bf16 config)
      -- hardware-validated `valid:y` across fp32, int8, and bf16-default (all both tile
      shapes) to confirm the reordering is safe. See `docs/gfx1250_wmma_layout.md`'s
      Phase 64.
- [ ] **Pre-existing bug: bwd's `gemm_k_global_split` crashes (illegal memory access) --
      extensively investigated 2026-09-01, NOT root-caused, real progress made.**
      Reproduces on the unmodified pre-Phase-63 codebase too. Ruled out with concrete
      evidence (see `docs/gfx1250_wmma_layout.md`'s Phase 65 for the full trail):
      search-heuristic instability (crashes at every pinned split count including the
      degenerate single-shard case), the shape/tile itself (identical shape works
      perfectly with split-K off), kernarg struct layout mismatch, missing per-launch
      zero-init, a HIP-event resource leak, and a generic host-side timing-harness bug
      (wrw's structurally similar split-K path, same harness, works correctly). **New,
      narrower findings**: (a) crash correlates with total dispatch COUNT specifically
      -- 5 repeats succeeds, 10+ (including the driver's true default) crashes, same
      pinned split/shape; (b) requires an actual multi-iteration K-loop -- a trivial
      single-K-block shape never crashes regardless of repeat count; (c) `rocgdb` shows
      all waves halted mid-main-loop with individually-plausible-looking addresses (no
      obviously-wrong value found by inspection); (d) applying the fix from this
      codebase's one other known barrier/LDS-visibility race
      (`docs/gfx1250_fp32_wmma_occupancy_race.md`'s `lds_double_buffer=1`) did NOT fix
      it. Next steps for whoever picks this up: rocgdb single-stepping (breakpoints on
      GPU kernel code didn't behave as expected this session, needs a different
      technique), testing whether the specific failing iteration varies run-to-run
      (confirming/refuting a genuine race), and an occupancy-controlled comparison
      (fewer workgroups, same repeat count).
- [x] **VOPD/VOPD3 dual-issue VALU** — CDNA5 ISA §7.8/7.8.1 (dual-issue VALU, two
      independent ops per cycle on eligible opcode pairs) was used nowhere in this
      codebase. **Scoped and (partially) implemented 2026-09-01 (Phase 67)**: the
      originally-suspected target (the tail-dword masking helper,
      `igemm_bwd_gtc_wmma_nhwc.py:1264-1290`) turned out to be a dead end -- its
      `v_cmp`/`v_and`/`v_or` opcodes aren't a VOPD-eligible pair and the chain is mostly
      sequentially dependent, so no amount of scoping would have made it pair. The real,
      safe target found instead: the `v_mov_b32 v[reg], 0` zero-init loops present in
      all three direction generators (accumulator clear, prologue `v_zero`, and the
      per-K-iteration tail-masked global-load zero-init) -- fully independent,
      fully VOPD-eligible, shared-literal-legal. New shared helper
      `emit_vopd_paired_zero_init` (`python/igemm/igemm_base.py`) applied at all 9
      matching call sites across fwd/bwd/wrw, unconditionally (bit-identical output,
      same category as Phase 64's wait-batching -- no new opt-in tunable). Confirmed
      safe with `wmma_acc_high_bank=1` (VGPR-MSB, Phase 54) on hardware. Does NOT cover
      every VALU chain in the codebase -- only the zero-init pattern; other independent
      VALU chains (if any exist) are still unscoped. See
      `docs/gfx1250_wmma_layout.md`'s Phase 67.
- [ ] **wrw's mandatory atomic-epilogue is a structural cost, not just a tuning gap** —
      found during the original profiling investigation (2026-09-01, before Phase 63),
      never turned into a tracked item until now. wrw's `gemm_k_global_split` path (its
      PRIMARY path per CLAUDE.md) always pays a scalar, per-element, non-coalesced
      `global_atomic_add_f32` epilogue (`coalescing_store_wmma.py:597-668`) — no
      vectorized store, no coalescing across lanes, unlike fwd/bwd's cheap
      `direct_store` epilogue (a plain per-lane `global_store_dword`, already confirmed
      LDS-free and cheap in Phase 64's investigation). `atomic_pack_bf16` doesn't reduce
      VALU either — it *adds* ~5 instructions per pair (`v_permlane_xor_b32`,
      `v_cvt_pk_bf16_f32`, `v_and_b32`, `v_cmpx_eq_u32`, `s_mov_b32` restore) purely to
      halve atomic-RMW traffic, trading VALU for less memory contention. This plausibly
      explains why wrw's benchmark ratios are consistently worse than fwd/bwd's
      (`bench_results.md`: wrw 1.09x-5.32x slower vs gfx1250 MIOpen, notably worse than
      fwd/bwd's typical ≤2x). **Not yet investigated**: whether shapes where GEMM_K fits
      in a single pass (no genuine need for a K-reduction split) could skip
      `gemm_k_global_split` entirely and reuse the cheap `direct_store` epilogue instead
      — would need checking `tunable_is_valid`'s current requirements and whether wrw's
      non-split-K path (already exists, per `igemm_wrw_gtc_gfx1250_nhwc_bf16.config`) is
      being under-used by the driver's shape-to-config selection for GEMM_K-small
      shapes that don't actually need splitting.
- [x] ~~**Address-hoist in `_emit_direct_store`'s epilogue `i_rm` loop**~~ — **DONE
      2026-09-01 (Phase 66), hardware-validated.** The outer `i_rm` loop used to
      recompute `row*stride+col` from scratch every iteration (4 instructions:
      row+base, mul by stride, add col, shift to bytes) even though consecutive
      blocks are separated by a compile-time-constant row gap. Precomputes a
      `s_row_gap = row_stride * gap_rows` SGPR once, then each subsequent i_rm
      transition is a single `v_add_u32` instead of the 4-instruction recompute --
      confirmed in the generated 128x128 bf16 kernel (`wave_repeat_m=4`, 3
      transitions, each dropping from 4 to 1 instruction). This was NOT as trivial
      as it looked when filed -- two real bugs found and fixed only by trusting
      `-V 1` over hand-derivation, not skipped:
      1. **Wrong source register.** The cur/nxt ping-pong's PYTHON aliases return to
         their block-starting binding (cur=v_tmp1) after an even number of swaps,
         but the CONTENT holding the last row's real address ends up in v_tmp2
         (nxt's binding), not v_tmp1 -- v_tmp1 holds a stale, one-row-behind value.
         An initial version read from v_tmp1 and got `valid:n`.
      2. **Off-by-one in the gap itself.** The gap from the last row a block
         touches (`row_off+num_v_c-1`) to the next block's first row
         (`row_off+wave_tile_m`) is `wave_tile_m-num_v_c+1`, not
         `wave_tile_m-num_v_c` -- missing the `+1` also produced `valid:n`, caught
         separately after fixing bug 1.
      Both root-caused by reading the actual generated disassembly line-by-line and
      hand-tracing register bindings against it, not by re-deriving from the
      formula alone. Needed threading a second scratch SGPR (`s_tmp2`, already
      existed as an optional parameter used only by the mutually-exclusive
      LDS-reshuffle path's `wmma_n_tail` branch, now always passed) through
      fwd/bwd/wrw's `coalescing_store` call sites (wrw actually DOES support
      `direct_store` -- an earlier claim in this conversation that it didn't was
      wrong; found while wiring this up, both of wrw's two call sites needed the
      same fix). Hardware-validated `valid:y`: fwd/bwd/wrw direct_store (both tile
      shapes), `wmma_m_tail`+direct_store (exercises the tail-row-tracker's own
      hoist), fp32 direct_store, int8 direct_store (fwd), and a zero-regression
      check on the non-direct_store `wmma_n_tail` LDS-reshuffle path (confirms
      sharing the scratch SGPR slot is safe). See
      `docs/gfx1250_wmma_layout.md`'s Phase 66.
- [ ] **hipconv's block-diagonal channel packing across conv groups** — fills small WMMA
      tiles when the group count is high, a structurally different way to solve
      "GEMM_M/N too small to fill a tile" than tail-masking.
- [ ] **Deeper main-loop pipelining** (N-stage, beyond MISA's current double-buffer),
      gated by LDS headroom, per FlyDSL's `num_buffers` pattern.
- [ ] **Autotuning-with-build-cache infrastructure** (FlyDSL's `conv3d_autotune.py`
      pattern) as a longer-term supplement to MISA's hand-curated `.config` files,
      specifically for shapes where static configs are demonstrably failing (wrw's tail
      cases).

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
