# Phase 54 (VGPR-MSB) — WIP status, paused for a GPU hang/reboot

**Status as of this write-up: NOT WORKING YET, but very close.** The core mechanism is
hardware-verified. The full kernel no longer crashes. It still produces wrong results
(`valid:n`, no crash) for reasons not yet root-caused. A GPU hang was triggered by one
of the diagnostic tests below and the machine needs a reboot before further hardware
testing — see the safety warning at the end before running anything on real hardware
again.

## Goal

Grow gfx1250 WMMA's macro-tile past 128x128 by moving the accumulator (`v_c`) into a
second, independently-addressed 256-VGPR bank via `S_SET_VGPR_MSB` (ISA doc §3.3.2.3),
instead of trying to fit everything in the plain 256-register space (Phase 53 showed
that's mathematically impossible for a 256x256 tile — see Phase 52/53 in
`docs/gfx1250_wmma_layout.md`).

## What's proven (hardware-verified, high confidence)

Via ~15 minimal standalone kernels (source and launcher scripts under
`/tmp/msb_hwtest/` on the machine this was developed on — **not saved anywhere
persistent, will need to be rewritten on the next machine if wanted**, though the
technique is fully described below so recreating them is quick):

1. `S_SET_VGPR_MSB` itself works exactly as documented: plain `v0`-`v255` operand
   syntax + a preceding `s_set_vgpr_msb <imm>` retargets the physical register,
   independently per DST/SRC0/SRC1/SRC2 slot. Immediate bit layout confirmed:
   `imm[7:0] = {dst[7:6], src2[5:4], src1[3:2], src0[1:0]}`.
2. Plain VALU (`v_mov_b32`), DS (`ds_write_b32`/`ds_read_b32`), and VMEM
   (`global_store_dword`) all correctly read/write bank 1 through this mechanism.
3. `v_wmma_f32_16x16x32_bf16` (the actual f32-accumulate, 8-VGPR-C/D instruction this
   project's 128x128 baseline uses) correctly accumulates into a bank-1 C/D operand,
   for both a single call and the full 4x4=16-call `(i_rm,i_rn)` grid pattern this
   project's main loop actually emits — **as long as WMMA operands are 8-register-
   aligned** (an 8-register C/D operand at an unaligned base, e.g. `v[6:13]`, silently
   computes garbage even with NO banking involved at all — a real, separate finding,
   worth remembering for any future hand-written WMMA test).
4. Works correctly across 128 threads (4 waves) with real `s_barrier` synchronization
   between them.
5. `v_lshl_or_b32` is VOP3 with a genuine, MSB-sensitive SRC2 slot (not folded into
   SRC0/SRC1) — confirmed by direct disassembly. Easy to miss since the assembly
   syntax `v_lshl_or_b32 vdst, src0, shift_imm, src2` reads like a 2-source op.

**Conclusion: the ISA mechanism itself is solid on this hardware.** Every bug found so
far has been in this project's own codegen, not the hardware/toolchain.

## Bugs found and fixed this session (all real, all worth keeping regardless of Phase 54's fate)

1. **Phase 53 pre-existing bug**: `_emit_chunked_non_atomic_store`'s caller did
   `return self._get_deferred()` from *inside* the `with self._deferred_context():`
   block. `deferred_context_t.__exit__` (which populates `self.outter.deferred_buffer`)
   hadn't run yet when the return expression evaluated, so the chunked epilogue path
   silently emitted **nothing at all** — this had never been caught because the chunked
   epilogue was never actually build-tested before Phase 53 was parked. Fixed by
   restructuring to `if/else` so the single, correct, already-present return (after the
   `with` block naturally exits) handles it. Confirmed via diff that the plain
   (`wmma_epilogue_chunked=0`) path is still byte-identical.
2. **Phase 53 pre-existing addressing bug**, found via hardware validation after fixing
   #1: the chunked epilogue's gather loop advanced the *global* store address by a
   single fixed per-pass stride, silently assuming "compact row advances linearly"
   implies "true (uncompacted) row advances linearly." False whenever the compact row
   crosses a `wave_tile_m` boundary (a discontinuity in the compact→true mapping) —
   roughly half of every group's stored elements landed at the wrong row
   (`valid:n`, no crash). **Fixed and hardware-confirmed correct (`valid:y`)** by
   recomputing the true row fully from `v_tid` every pass instead of assuming linearity
   — see `_emit_chunked_non_atomic_store`'s per-pass loop and its docstring in
   `coalescing_store_wmma.py`. This fix needed one new persistent register
   (`v_chunked_col`, holds this thread's global column across all passes) — see
   `kernel_vgpr_t.__init__` in both `igemm_fwd_gtc_wmma_nhwc.py` and
   `igemm_bwd_gtc_wmma_nhwc.py`.
3. **New Phase 54 tracker design bug**: `vgpr_msb_tracker_t.ensure()`'s "skip if
   unchanged" memoization assumes textual/emission (Python codegen) order matches
   runtime execution order. False for `wmma_main_loop.py`'s `emit_wmma_tile()`, called
   from ~6 different places (early issue, loop body, drain/tail) stitched together by
   *real* runtime branches — a branch can reach the WMMA issue point without ever
   having executed whichever earlier textual call site the tracker "remembers" setting
   the state from. Fixed by adding `vgpr_msb_tracker_t.force()` (always emits,
   never skips) and using it at the WMMA issue site instead of `ensure()`.
4. **New Phase 54 bug (this is the one that caused the actual hardware crash)**: the
   accumulator zero-init sets `dst=1` (bank 1) and nothing ever reset it before the
   *rest* of the prologue (GEMM_M index decomposition, tap-loop address setup, etc.)
   continued writing ordinary bank-0 VGPRs. Every VGPR write in the prologue after the
   zero-init was silently landing in bank 1. **Found via `rocgdb`** (see below) — the
   crash's actual PC was in `..._tap_x`, at a completely unrelated, ordinary
   `global_load_b128` whose address register had never been correctly written, nothing
   to do with WMMA/epilogue at all (which is why bisecting by disabling the epilogue,
   then the main loop, kept "succeeding" right up until the prologue-only case, which
   finally isolated it). Fixed by adding an explicit
   `ensure(dst=0, src0=0, src1=0, src2=0)` immediately after the zero-init loop, in
   both `igemm_fwd_gtc_wmma_nhwc.py` and `igemm_bwd_gtc_wmma_nhwc.py`.
5. **New Phase 54 bug (ordering)**: in the *unchunked* epilogue's scatter loop, the
   `src1=1` (bank 1, for `v_c`'s DATA operand) toggle was placed *before* each `i_rm`
   iteration's own address computation (which reads `v_gemm_im`/`v_gemm_in`, needing
   `src1=0`) — backwards for iterations after the first. Fixed by moving the toggle to
   after the address computation, and adding an `src1=0` reset at the top of every
   `i_rm` iteration after the first (mirrors the chunked path's already-correct
   per-group toggle pattern). See `coalescing_store_wmma.py`'s unchunked
   (`else:`) branch.

After fixes #3 and #4, the **crash is gone**. `wmma_acc_high_bank=1` against the plain
unchunked epilogue path (128x128 tile, bf16, fwd) now runs to completion without
faulting. It still reports `valid:n` (wrong numerical results) on the same shape that
gives `valid:y` for the unmodified baseline. **Root cause not yet found.**

### How the crash was actually found (rocgdb — use this technique first, not more guessing)

Guessing-and-checking via standalone isolated repros ate an enormous amount of time
before this worked and kept passing (correctly, but unhelpfully) for every hypothesis
tried. What actually cracked it: running the real, failing `conv_driver.exe` invocation
directly under `rocgdb`:

```
export HSA_ENABLE_DEBUG=1
rocgdb -batch -x /tmp/gdb_cmds.txt --args ./conv_driver.exe convbfp16 -n 1 -c 32 -H 16 -W 8 \
  -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F 1 -t 1 \
  --in_layout NHWC --fil_layout NHWC --out_layout NHWC -V 0
```
where `/tmp/gdb_cmds.txt` contains:
```
set pagination off
run <same args as --args, gdb also needs `run` with them>
bt
info registers
thread apply all bt
```
The crashing wave's `pc` in `info registers` pointed straight at the exact faulting
instruction and its containing label (`..._tap_x`), which was then matched back to a
source line by disassembling the built `.hsaco`
(`llvm-objdump -d --mcpu=gfx1250 <name>.hsaco`) and finding that same byte offset.
**Do this immediately next time**, before spending hours on synthetic repros.

## Current unresolved question: why is it still `valid:n`?

Two live hypotheses, neither confirmed:

1. **Another ordering/reset gap** like bugs #4/#5 above, somewhere not yet found —
   plausible given how many were found by inspection once actually looked for. Worth a
   careful, instruction-by-instruction re-audit of every `ensure()`/`force()` call site
   against the *actual* `.inc` output (not just the Python source) before assuming
   anything deeper is wrong. The rocgdb technique above should be the first tool
   reached for, not last.
2. **A genuine WMMA back-to-back-same-register hazard** on this hardware (see the
   critical safety note below) that the *real* kernel's k-loop might also trigger in a
   subtler form even with intervening instructions between dependent WMMA issues
   (unlike the isolated repro, which had zero interleaving and hung outright). The ISA
   doc explicitly warns: *"Hardware does not detect data dependencies between
   co-executing multicycle instructions from the same wave... See Requirements for WMMA
   data hazards for more information on WMMA hazard avoidance"* — that referenced
   section was not located/read this session. **Read it first** before further
   hardware testing; it may describe a required NOP/delay pattern around WMMA that
   this project's *existing* (bank-0-only) codegen already satisfies incidentally
   (via the barrier/ds_read work between k-iterations) but that Phase 54's bank
   switching could be disrupting somehow (e.g. if `s_set_vgpr_msb` itself needs to be
   excluded from whatever hazard-covering instruction count the hardware expects,
   similar to how the doc separately notes `S_SET_VGPR_MSB` doesn't count toward
   `S_DELAY_ALU`'s SKIP count).

## CRITICAL SAFETY WARNING for whoever resumes this

A standalone test that issued **4 back-to-back `v_wmma_f32_16x16x32_bf16` calls with a
same-register dependency chain (each reading the previous call's C/D output) and zero
interleaving instructions** hung the GPU — `rocm-smi` showed 100% GPU utilization even
after the host-side process was killed, and a subsequent unrelated kernel launch
timed out after 2 minutes waiting on it. This required a machine reboot to clear.

**Do not repeat that exact pattern** (tight back-to-back WMMA depending on its own
immediately-preceding output, no other instructions between) without first reading the
ISA doc's WMMA-hazard section referenced above, and without a way to recover the GPU
(or permission/access to do a reset) if it hangs again. When testing anything WMMA-
related on real hardware:
- Always run under a killable, backgroundable process with a short timeout.
- Prefer testing patterns that match what the *real* generated kernel actually does
  (which always has a barrier + several `ds_read`s between one iteration's WMMA burst
  and the next touching the same registers) over maximally-tight synthetic repros —
  the tight repro is what triggered the hang, and does not resemble real usage.
- Have a rollback/reset plan (or explicit permission to run one) before running
  anything that stacks dependent multicycle WMMA instructions back to back.

## Exact state of the code (as committed in this session's WIP commit)

All changes described in bugs #1-#5 above are in the working tree / this commit.
Summary of touched files:

- `python/operations/utility.py` — new `vgpr_msb_tracker_t` class (`ensure()` +
  `force()`), placed right after `gpr_sequencer_t`.
- `python/operations/wmma_main_loop.py` — `ctrl_wmma_main_loop_t.vgpr_msb_tracker`
  field; `emit_wmma_tile()` uses `force()` (not `ensure()`) to set
  `dst=1,src0=0,src1=0,src2=1` before every WMMA burst.
- `python/operations/coalescing_store_wmma.py` — `ctrl_coalescing_store_wmma_t.
  vgpr_msb_tracker` field; both the chunked (`_emit_chunked_non_atomic_store`) and
  unchunked (`__call__`'s `else:` branch) non-atomic epilogue paths are wired with
  correctly-ordered `ensure()` calls (see bugs #2/#5 above for the exact
  per-iteration toggle pattern in each). New `v_chunked_col` parameter threaded
  through from the caller. The plain atomic (`gemm_k_global_split`) path and
  `atomic_pack_bf16` are **not** wired — `wmma_acc_high_bank` asserts against both.
- `python/igemm/igemm_base.py` — new `wmma_acc_high_bank` tunable (default 0);
  asserts `num_vgpr_accumulate_c <= 256` (single-bank-for-`v_c` scope, see Phase 54 in
  `gfx1250_wmma_layout.md`) and excludes `gemm_k_global_split`/`wmma_m_tail`/
  `wmma_n_tail` (not yet wired for MSB).
- `python/igemm/igemm_fwd_gtc_wmma_nhwc.py`, `igemm_bwd_gtc_wmma_nhwc.py` —
  `self.vgpr_msb_tracker` created in `__init__`; `v_c` allocated via its own
  bank-1 `gpr_sequencer_t` (`v_c_vseq`) instead of the shared bank-0 one when
  `wmma_acc_high_bank` is set; `kernel_code_dict`'s `workitem_vgpr_count` set to
  `256 + num_vgpr_accumulate_c` (bank 1 always starts at physical VGPR 256
  regardless of bank 0's actual usage — the wave must be granted that whole span);
  prologue's accumulator zero-init resets `dst=0` immediately after (bug #4); new
  `v_chunked_col` VGPR allocated when `wmma_epilogue_chunked` is set.
- `python/operations/wmma_mapping.py` — Phase 53's (parked, not yet revisited)
  256x256/256x128 table entries, unrelated to this session's Phase 54 debugging.
- `docs/gfx1250_wmma_layout.md`, `docs/gfx1250_optimization_backlog.md` — Phase
  52/53 corrections and Phase 54 mechanism write-up (this file supplements, not
  replaces, that write-up — check both).
- `config/igemm_fwd_gtc_gfx1250_nhwc_bf16_256x256.config`,
  `config/igemm_bwd_gtc_gfx1250_nhwc_bf16_256x128.config` — Phase 53's parked tile
  configs, parameter-consistent but not yet buildable/relevant until the VGPR-MSB
  mechanism itself is fully working.
- `amd-instinct-cdna5-instruction-set-architecture.md` — the local ISA reference
  doc this whole investigation depends on. Committed this session at the user's
  explicit request (machine migration) — this reverses the standing "never commit
  this file" convention from earlier in the project's history; if that convention
  had a real reason behind it (size, licensing/NDA), reconsider before pushing this
  commit anywhere shared.

## How to resume on a different machine

1. Clone/pull this branch (`users/SreecharanGundaboluAMD/gfx1250_bringup`) — this
   WIP commit has everything above.
2. Confirm the new machine has: a gfx1250 GPU (`rocminfo | grep gfx1250`), ROCm
   toolchain with `llvm-mc`/`clang++`/`llvm-objdump` targeting gfx1250 (this session
   used `/home/sgundabo/rocm-10.1/llvm/bin/` — path will differ), `rocgdb`, and
   `hipcc` (for any standalone repro kernels, if rebuilding them).
3. **First action**: read the ISA doc section on WMMA data hazards (search
   `amd-instinct-cdna5-instruction-set-architecture.md` for "WMMA data hazard" or
   similar — referenced in section 5.7 but not located/read this session).
4. Reproduce the current `valid:n` state: build
   `config/igemm_fwd_gtc_gfx1250_nhwc_bf16_128x128` — wait, there is no such
   pre-made config for the mechanism-only test; recreate it inline (this exact
   config was used throughout this session, not saved to a file):
   ```
   [codegen]
   arch = 'gfx1250'
   code_object = 'cov3'
   mode = 'flat'

   [igemm_fwd_gtc]
   gemm_m_per_block         = 128
   gemm_n_per_block         = 128
   gemm_k_per_block         = 32
   wmma_tile_m              = 16
   wmma_repeat_m            = 4
   wmma_tile_n              = 16
   wmma_repeat_n            = 4
   tensor_a_thread_lengths  = [1, 32, 1, 1]
   tensor_a_cluster_lengths = [1, 1, 1, 128]
   tensor_b_thread_lengths  = [1, 32, 1, 1]
   tensor_b_cluster_lengths = [1, 1, 1, 128]
   direction                = "fwd"
   precision                = "bf16"
   tensor_layout            = 'nhwc'
   nxb                      = 0
   nxe                      = 0
   wavefront_size           = 32
   cumode                   = 0
   wmma_acc_high_bank       = 1
   ```
   ```
   python3 igemm_codegen.py -d /tmp/p54_resume <path-to-above-config>
   cd /tmp/p54_resume
   IGEMM_WARMUP=2 IGEMM_REPEAT=2 ./conv_driver.exe convbfp16 -n 1 -c 128 -H 16 -W 8 \
     -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F 1 -t 1 \
     --in_layout NHWC --fil_layout NHWC --out_layout NHWC -V 1
   ```
   Expect `valid:n`, no crash, matching this session's end state.
5. Use `rocgdb` (see the technique above) on this exact repro *first*, before
   writing any new synthetic tests — it found the real bug in minutes once used,
   after hours of guessing didn't.
6. Once `valid:y` is achieved on this plain 128x128 mechanism-only test: re-run the
   **zero-regression check** (every existing config, `wmma_acc_high_bank` defaults
   off, confirm byte-identical `.inc` output) before touching anything else, then
   resume Phase 53's parked 256x256/256x128 tile work (configs already exist, see
   above) now that the mechanism should actually be provably correct.
