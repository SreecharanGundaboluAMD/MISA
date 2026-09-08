# gfx1250 WMMA main-loop scheduler: design proposal (not yet implemented)

## Status

Design only. No source changes in this file's scope. Implementation +
hardware validation across the full precision/tile/tail matrix is
substantial follow-on work — see "Why this is deliberately not implemented
yet" below.

## What XDLOPS already has that WMMA doesn't

`python/codegen/scheduler.py` (`simple_interleave_scheduler_t`,
`create_scheduler`) is a generic, arch-agnostic machine-basic-block (MBB)
interleaver, already used throughout `python/operations/mfma_main_loop.py`
(the gfx908/90a/940 XDLOPS main loop). It is NOT XDLOPS-specific code; it
operates on `mc_inst` lists grouped into MBBs by
`create_machine_basic_block(fn, group_mbb_by_end_of_inst_op="v_mfma")` (or
`"ds_write"`, etc.) — i.e. "cut the instruction stream into chunks, each
chunk ending right after the Nth occurrence of instruction X" — then
interleaves two such MBB lists according to a named pattern:

- **`INTERLEAVE_PTN_0`** ("mbb0 mfma+share_load, mbb1 global_load+move_slice_window"):
  ratio-interleaves a global-load/move-slice-window MBB stream into an
  mfma+shared-load MBB stream. The ratio is derived, not hand-tuned: count
  global-mem ops in mbb_1, grow `gmem_per_interval` until it fits within
  `2/3` of mbb_0's length, spread the rest evenly.
- **`INTERLEAVE_PTN_1`** ("mbb0 mfma, mbb1 share_store"): evenly spreads N
  shared-store MBBs across (mfma_count − reserved) mfma-anchored slots in
  the base stream.
- **`INTERLEAVE_PTN_2`** (declared, unused in current mfma_main_loop.py):
  global-store vs. share-store.

`wmma_main_loop.py` (gfx1250) has none of this. Its `emit()` (see
`python/operations/wmma_main_loop.py:249-833`) is one large hand-sequenced
function: fixed `if async_a: ... if tdm_a: ...` branches choosing between
mutually-exclusive *data-path* strategies (async/TDM/plain VADDR), plus a
handful of independently-gated hand-hoisting booleans
(`can_hoist`/`gap_hoist`/`l2_prefetch`, PERF-001/Phase 70) that reorder at
most one or two instructions each. There is no MBB abstraction and no
general interleaving pass — every schedule variation is a new hand-written
branch in the same function.

## Proposed schedules (reusing the existing MBB/scheduler machinery, unmodified)

All three reuse `create_scheduler`/`simple_interleave_scheduler_t` exactly
as XDLOPS does; no new scheduler code is needed, only the WMMA-side MBB
construction (the `create_machine_basic_block(fn, group_mbb_by_end_of_inst_op="v_wmma")`
calls) and the `ctrl_wmma_main_loop_t.emit()` call sites that build the two
MBB lists and invoke `.lower(interleave_pattern=...)`.

1. **`schedule=fixed` (default, byte-identical to today)** — current
   hand-sequenced emission path, untouched. Existing configs' generated
   `.s` stays byte-for-byte identical; this is the required fallback per
   the tuning refactor plan's Phase 4d ("default-off/byte-identical
   fallback").

2. **`schedule=mbb_gmem_interleave`** — group the WMMA main loop's
   mfma+shared-load instructions into one MBB list
   (`group_mbb_by_end_of_inst_op="v_wmma"`), the global-load +
   move-slice-window instructions into a second, and interleave with
   `INTERLEAVE_PTN_0`. This directly targets PERF-001's own finding
   (`docs/gfx1250_rocprof_profiling.md`: WMMA-busy is only 2.3-5.2%, i.e.
   95%+ idle) by spreading global-load issue slots evenly across the WMMA
   burst instead of hoisting only the single next-tile address/load
   (today's `can_hoist`). Highest expected value on fwd/bwd's plain-VADDR
   path (no async/TDM); async/TDM paths already get their loads issued
   before the wait, so the marginal benefit is smaller there but the
   mechanism is not path-specific — it would need per-path MBB lists
   supplied by the same `f_gld_a`/`f_gld_b` functors already in
   `ctrl_wmma_main_loop_t`.

3. **`schedule=mbb_store_interleave`** — group shared-store instructions
   into an MBB list, interleave with `INTERLEAVE_PTN_1` against the
   mfma-anchored base stream, instead of today's single
   `s_waitcnt lgkmcnt(0)` + batch store before the barrier. Directly
   targets the same idle-WMMA-unit finding for the *store* side of the
   pipeline (today's LDS double-buffer store is a single block, not spread
   across the WMMA burst at all).

Not proposed as a fourth option: combining both 2 and 3 in one pass. XDLOPS
itself never does this (every `mfma_main_loop.py` call site interleaves
gmem separately from store, in two sequential scheduler invocations,
`se_sub` then `se_last`) — mirror that structure rather than inventing a
3-way interleave with no working precedent to validate against.

## Selection mechanism sketch

- New tunable `wmma_schedule` (str, default `"fixed"`), read the same way
  every other optional tunable is (`utility_dict_with_default_t`,
  `igemm_base.py`).
- `ctrl_wmma_main_loop_t.emit()` dispatches at the top on this field: the
  existing body becomes the `"fixed"` branch verbatim (no behavior change
  for every existing config); `"mbb_gmem_interleave"` and
  `"mbb_store_interleave"` are new sibling branches built from the same
  functors (`f_gld_a/b`, `f_sst_a/b`, the WMMA-issue macro) already passed
  into `ctrl_wmma_main_loop_t`, just re-grouped into MBBs instead of
  emitted inline.
- Folded into the kernel name (like `wmma_acc_f16`, not like
  `atomic_scope`) since it changes emitted-instruction order/count, which
  is the kind of change this repo's naming convention reserves for
  driver-visible kernel identity — though note it does NOT change the
  driver-visible kernel *contract* (block/grid size, buffer layout), only
  which of several byte-identical-throughput-behavior variants was built;
  worth confirming against `igemm_gtc_encode_kernel_name`'s existing
  precedent for schedule-only tunables (`wmma_setprio` is a comparable
  precedent and IS folded in) before committing to this.

## Why this is deliberately not implemented yet

The tuning refactor plan's own Phase 4d calls pipeline-schedule
consolidation "Highest risk and do this last" for exactly the reason that
applies here: a scheduler bug is a silent wrong-answer risk on every
config that opts in, not a build failure. Landing it responsibly requires,
per this repo's own established validation bar (see this session's VGPR
reuse and tile-coverage work as precedent):

1. Implement both new schedule branches behind the `fixed` default.
2. Hardware-validate `valid:y` across the full precision × tile × tail
   combinator — fp16/bf16/fp32/int8, both tile families, every tail
   variant (`wmma_m_tail`/`wmma_n_tail`/`mntail`), on the actual target
   GPU, not just successful assembly.
3. Re-run the exact `rocprof-compute`/`rocprofv3 --pmc` methodology already
   used in this session's Reprofile phase, on both schedules, to confirm
   the WMMA-busy-fraction finding this design targets actually moves.
4. Full `script/build_and_filter_configs.py --jobs 16` regression (11504
   sections) to confirm zero collateral assembly/VGPR regressions.

That is a multi-session effort in its own right, not something to compress
into the remaining scope here without risking exactly the kind of
plausible-but-wrong "done" this project's own conventions (and this
session's own DISABLE_XDL_ARB_STALL result — a plausible-sounding lever
that measured as a regression once actually tried on hardware) argue
against assuming without measurement.
