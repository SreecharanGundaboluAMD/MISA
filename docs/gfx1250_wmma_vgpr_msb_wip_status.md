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

## Update (2026-08-28, new machine, post-migration): RCA'd and fixed the dst=1 leak bug -- necessary but NOT sufficient

A second machine (the one this session had been running on) crashed before hardware
validation could finish, but its investigation correctly identified a real, previously-missed
bug in `wmma_main_loop.py`: `emit_wmma_tile()` sets `dst=1` (bank 1, for `v_c`) via
`vgpr_msb_tracker.force()` before every WMMA burst, but **nothing ever reset it back to 0
afterward**. Every ordinary bank-0 VGPR write that runs between one burst and the next --
`f_sld_a`/`f_sld_b`'s `ds_read_b128` (loads `v_a`/`v_b`), `f_gld_a`/`f_gld_b`'s
`global_load_dwordx4` (writes the persistent `v_gld_a`/`v_gld_b` staging registers), and
`emit_buffer_switch()`'s `v_xor_b32` (toggles `v_sst_a_os`/`v_sld_a_os`/`v_sld_b_os`) -- was
silently landing in bank 1 instead. This matches the "valid:n, no crash" symptom exactly (bank
1 is still a valid VGPR address, so the corruption never faults).

**Fix applied** (`python/operations/wmma_main_loop.py`'s `emit_wmma_tile()`): rather than
patching each individual consumer call site (which is exactly how the bug happened in the
first place -- easy to miss one), the reset (`vgpr_msb_tracker.force(dst=0, src0=0, src1=0,
src2=0)`) is now the LAST thing `emit_wmma_tile()` does, right after the burst, mirroring the
already-fixed prologue pattern (set dst=1 for the write, reset immediately after). This
guarantees every downstream consumer sees bank 0 by default regardless of which one runs next.
Uses `force()` not `ensure()`, same reasoning as the dst=1 call above (this point is reached via
a real loop-back branch, not just linear Python emission order).

**Verified on real hardware this session** (2026-08-28, gfx1250, idle-ish GPU apart from ~100%
utilization from two unrelated other-user `python3` processes -- proceeded carefully with
`timeout`, no hang):
1. Confirmed via generated `.inc`/disassembly: `s_set_vgpr_msb 0` now appears right after every
   WMMA burst, before the next `ds_read`/`global_load`/buffer-switch. Disassembly's own
   physical-address resolution (`v0 /*v256*/` etc.) confirms the bank-1 addressing for both the
   WMMA burst AND the epilogue's `ds_store_b32` (which independently toggles `src1=1` itself,
   already correct from the earlier bugs #3-#5 fixes) is being computed correctly by the
   assembler/hardware.
2. Zero regression: every existing (non-`wmma_acc_high_bank`) config produces byte-identical
   `.s`/`.inc` output before and after this fix (diffed `igemm_bwd_gtc_gfx1250_nhwc_bf16_32x32`
   directly) -- the fix is fully gated behind `ctrl.vgpr_msb_tracker is not None`.
3. **The crash-class symptom this bug caused is gone, but `valid:n` PERSISTS** on the same
   128x128 bf16 fwd mechanism-only test from this doc's resume config.
4. **Tested and REFUTED one hazard hypothesis**: added 4 real `v_nop` instructions right after
   the WMMA burst (testing whether reading a bank-1-addressed Matrix D needs more real-VALU-
   instruction separation than the ISA doc's dense-WMMA hazard table implies for bank-0 reads --
   doc section 7.12.1, table says 0 NOPs needed when XDL co-exec-stall isn't disabled, which this
   codegen never disables). **No change at all.** (Throwaway diagnostic, reverted, not in the
   working tree.)

### The remaining bug is precision-independent, and always faults at the exact same boundary

Initial per-pixel diffing on bf16 alone was misleading: `PER_PIXEL_CHECK_PRINT`'s diff-line
output caps at 100 printed mismatches, and bf16's first ~100 output elements (all within output
row/`gemm_m` index 0) already have small (~0.02%-9%) errors, consuming the whole cap before the
real story -- everything from output index 128 onward -- was ever visible. Re-ran with
`PRINT_EVERY_PIXEL=1` (prints every element unconditionally, no cap) across **all four
precisions** (bf16, fp16, int8, fp32) on matching 128x128 single-workgroup configs
(`gemm_m=n*ho*wo=128`, `gemm_n=k=128`, exactly one 128x128 output tile, so the flat output index
is `M_row*128 + N_col`):

- **Every precision shows the identical fault line**: output row `M=0` (flat indices 0-127) is
  either bit-exact (int8, fp32) or has only the small pre-existing noise (bf16, fp16 -- see
  "still open" note below); **every row `M>=1` (index 128 onward) is completely wrong** for
  every precision, not just a few percent off. int8 is the most legible: every element from
  index 128 on reads the exact constant `0x01010101` (a bit-pattern, not a plausible int32
  GEMM accumulation result -- exceeds the max possible sum for `c=128`). fp16 reads near-zero
  garbage (`0x0000003a`-ish tiny floats, unrelated to `ref`). fp32 reads plausible-looking but
  wrong small positive floats. bf16's `M=0` block is small-noise-wrong (a separate, much
  smaller-magnitude issue, still unexplained but clearly secondary) while its `M>=1` block is
  exactly the same kind of unrelated-to-`ref` garbage as the other three.
- **Confirmed this is NOT precision-specific and NOT an addressing bug**: disassembly for both
  bf16 and int8 shows byte-identical, correctly-resolved bank-1 physical addressing (via
  llvm-objdump's own `/*v256*/`-style annotations) for the accumulator zero-init (all 128
  registers, v256-v383), every WMMA burst call (all 16 `(i_rm,i_rn)` combinations, correct
  8-register strides), and the epilogue's `ds_store_b32` scatter (also correctly bank-resolved,
  same stride pattern, for both int8 and bf16). The *addresses* the generated code computes are
  provably correct at the instruction-encoding level for every row, not just row 0.
- **Confirmed this is NOT an epilogue-implementation bug**: `wmma_epilogue_chunked=1` (a
  completely different, independently-written LDS scatter/gather implementation from the default
  unchunked path -- different addressing scheme, different loop structure, shares no code) was
  tested against int8 and produces the **exact same failure**: same starting index (128), same
  exact corrupted value (`0x01010101`). Two structurally-unrelated consumers of `v_c` reading the
  *identical* wrong bit pattern at the *identical* index means the corruption is already present
  in `v_c`'s bank-1 physical registers (for every `(i_rm,i_rn)` group except the very first)
  before *either* epilogue implementation starts reading it -- ruling out both epilogue codepaths
  as the cause and pointing back at the main K-loop's WMMA accumulation into bank 1, or a genuine
  hardware behavior difference between the first bank-1 WMMA destination range used and every
  subsequent one.
- **Hazard-timing theories reconsidered and found wanting**: the ISA doc's accumulation-chain
  hazard entries require A/B/Index to match a *previous* instruction's D -- doesn't apply within
  one 16-call burst (every call in a burst targets a disjoint `v_c` sub-range, no intra-burst
  chaining), and the *cross-K-iteration* accumulation chain (same `v_c` sub-range read as C by
  the next outer K-block's WMMA) has a full barrier + several `ds_read`s of real separation
  already, for every precision identically -- yet only `i_rm=0`/row 0 survives. No hazard theory
  found so far explains why row 0 specifically would be exempt while every other row breaks
  identically across four unrelated instruction encodings (`v_wmma_f32_16x16x32_{f16,bf16}`,
  `v_wmma_i32_16x16x64_iu8`, `v_wmma_f32_16x16x4_f32`).

### RESOLVED (same session): it's the real epilogue's address formula, not v_c, not VGPR-MSB

Two independent lines of attack converged on the same answer:

**1. A standalone repro (`/tmp/msb_repro/`, gone -- ephemeral, recreate from this description)**
using the exact toolchain `clang++ -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 <f>.s -o
<f>.hsaco` + a `hipModuleLoad`/`hipModuleGetFunction`/`hipModuleLaunchKernel` host harness
(pattern: `test/twiddle/twiddle_test.cpp`, `test/persistent_workgroup/`). Built up incrementally,
matching the real kernel's exact scale each step -- **every single variant passed with zero
mismatches**: a minimal 2-group/1-wave test; the full 16-group/1-wave scale; the full
16-group/4-wave/128-register scale (bit-for-bit matching the real kernel's block/tile config);
adding distinct `v_a`/`v_b` sub-ranges per `(i_rm,i_rn)` (matching `emit_wmma_tile()`'s real
indexing exactly, not a shared/simplified input); and adding an in-flight `global_load` during
the WMMA burst (matching the real kernel's software-pipelined prefetch overlap). None of these
reproduced the bug -- the core VGPR-MSB + multi-group + multi-wave + K-loop mechanism is
correct at every scale tested.

**2. A direct raw-dump diagnostic in the REAL kernel** (`MSB_RAW_DUMP=1` env var, wired into
`igemm_fwd_gtc_wmma_nhwc.py`'s `emit_kernel_epilogue()` -- kept in the tree, off by default,
zero cost otherwise): bypasses `coalescing_store_wmma.py` entirely and dumps `v_c` (bank 1)
straight to `s_p_out` via a trivial flat per-thread `global_store`, no LDS round-trip, no
tile-transpose addressing. Run against the exact same failing int8 128x128 config:
**every one of the 16384 raw dwords across all 128 threads x 128 registers reads back exactly
correct** (`0x00000080` = 128, matching `ref`). Also tested two closer approximations of the
real epilogue's mechanics -- repeated `src1` 0→1→0→1 re-toggling per 32-register group
(`MSB_RAW_DUMP_REPEATED_TOGGLE=1`), and a full `ds_write`(bank1)→barrier→`ds_read`(bank0)→
`global_store` LDS round-trip with trivial flat addressing (`MSB_RAW_DUMP_LDS_ROUNDTRIP=1`) --
**both also come back 100% correct**.

**Conclusion**: `v_c` is provably, unconditionally correct right after the main loop ends, for
every group, every precision. VGPR-MSB itself, multi-group accumulation, multi-wave sync, the
K-loop structure, repeated bank re-toggling, and even a full LDS round-trip all work fine in
isolation. The bug is real but narrow: it's specifically in `coalescing_store_wmma.py`'s
**real per-`(i_rm, j, i_rn)` tile-transpose address computation** (the actual row/col math
using `v_gemm_im`/`v_gemm_in`, `row_off = i_rm*wave_tile_m`, `padded_stride`/`macro_tile_n`
shifts, and `wmma_mapping.py`'s `get_gemm_index_for_dst_matrix` per-lane row formula
`row = wave_m_idx*64 + i_rm*wave_tile_m + (lane/16)*8 + j`) -- an ordinary addressing/logic bug
in that formula (most likely a genuine LDS address collision where some `(i_rm,j,i_rn)`
combination overwrites another's slot before it's read back), NOT a hardware or VGPR-MSB-
mechanism issue at all. Hand-deriving the row formula algebraically didn't turn up an obvious
collision in the time available (it looks non-colliding on paper: `wave_m_idx∈{0,1} *64 +
i_rm∈{0..3}*16 + (lane/16)∈{0,1}*8 + j∈{0..7}` covers 0..127 without overlap) -- the bug is
subtle enough that it needs a live dump, not more hand algebra.

### FIXED (same session, immediately after RESOLVED above): the actual bug, found by hand-tracing the generated `.inc`

Rather than adding another live dump, hand-tracing the ACTUAL generated assembly for the failing
int8 config (not just the Python source) found it directly. In the unchunked scatter's inner
`for j in range(inst_wmma.num_v_c):` loop (`coalescing_store_wmma.py` ~line 727-729, and the
chunked path's structurally identical twin ~line 323-325), the per-`j` row-advance is:

```python
self._emit(f"v_add_u32 v[{v_tmp1}], {stride}, v[{v_tmp1}]   ; advance to row ...")
```

`v_add_u32` is a VOP2 instruction: `vdst, src0, vsrc1` -- the immediate (`stride`) lands in the
**SRC0** slot, and `v_tmp1` (the running LDS address, meant to be an ordinary bank-0 value) lands
in the **VSRC1** slot. VSRC1's bank is controlled by the SAME `src1` MSB bit that's holding
`src1=1` for the surrounding `ds_write_b32`'s `v_c` DATA operand (set once, before the whole
`j` loop, via `ensure(src1=1)`, and not reset until the loop ends). So this "add 512 to the
address" instruction **silently reads its own current value from bank 1 (garbage) instead of
bank 0**, corrupting the running address for every `j>=1` -- while `j=0` (no advance needed yet)
uses the address computed *before* entering the `src1=1` window, so it's unaffected. This
exactly matches every observed symptom: the first row/group is always correct, everything after
it is corrupted, identically regardless of precision or which of the two epilogue
implementations runs (both have the same bug, independently). The OUTER `i_rm` loop's own
address computation already correctly brackets itself with `ensure(src1=0)`/`ensure(src1=1)` --
this exact same discipline was just missing from the INNER `j` loop's row-advance.

**Fix**: bracket the row-advance with `ensure(src1=0)` before / `ensure(src1=1)` after, in both
the unchunked (`coalescing_store_wmma.py`'s `else:` branch, ~line 727) and chunked
(`_emit_chunked_non_atomic_store`, ~line 323) paths.

**Hardware-verified fixed**: `valid:y` for fwd bf16/fp16/int8/fp32 (both unchunked and
`wmma_epilogue_chunked=1`) and bwd bf16, all on the same 128x128 single-workgroup shape that
previously failed. bf16/fp16's earlier-reported "separate small `M=0` noise issue" was NOT a
separate bug -- it was the same root cause (the tiny errors happened to look like rounding noise
for bf16/fp16's smaller accumulated magnitudes, while int8's larger integer values made the
corruption obviously nonsensical); it's gone now too. Zero regression confirmed: existing
non-`wmma_acc_high_bank` configs (`igemm_bwd_gtc_gfx1250_nhwc_bf16_32x32`,
`igemm_fwd_gtc_gfx1250_nhwc_bf16`) produce byte-identical `.s` output before/after (the fix is
fully gated behind `ctrl.vgpr_msb_tracker is not None`).

**Phase 54's VGPR-MSB mechanism is now genuinely working end-to-end** for the 128x128
mechanism-only test across every precision and both epilogue implementations. Next real step is
resuming the originally-parked 256x256/256x128 tile work (Phase 53's configs already exist --
`config/igemm_fwd_gtc_gfx1250_nhwc_bf16_256x256.config`,
`config/igemm_bwd_gtc_gfx1250_nhwc_bf16_256x128.config`) now that the mechanism is provably
correct, not just mechanism-proven-in-isolation.

## Phase 55 (2026-08-28, same day): resumed the 256x256/256x128 tile work -- works for a single workgroup, found a SEPARATE pre-existing multi-workgroup bug

Resumed Phase 53's parked configs now that Phase 54's mechanism is hardware-proven correct.

**Switched from packed to full accumulation.** Phase 53's configs used `wmma_acc_bf16=1`
because at the time that was the only way to have any chance of fitting the 256-VGPR
ceiling -- and per Phase 53's own math it STILL didn't fit (284 registers, 28 over) even with
packing. Replaced `wmma_acc_bf16=1` with `wmma_acc_high_bank=1` in both configs and added the
missing full-F32-accumulate `ctrl_wmma_mapping_t` table entries (`python/operations/
wmma_mapping.py`'s `'bf16'` key -- previously only `'bf16_bf16acc'`/`'fp16_f16acc'` had
256x256/256x128 entries). With `v_c` in bank 1, full F32 accumulate needs
`wave_repeat_m*wave_repeat_n*8 = 256` registers (fits easily; total wave VGPR request is 512,
confirmed via the generated `.s`'s `.vgpr_count`), avoiding the precision-losing packed
workaround entirely -- matches this project's stated preference for F32 accumulation
elsewhere (see the optimization backlog's "confirmed dead ends").

**Hardware-validated `valid:y` for a SINGLE workgroup**: fwd 256x256 and bwd 256x128, both at
the exact single-tile problem size (`gemm_m<=256, gemm_n<=256`) and with multiple K-blocks
within that single workgroup (`c` up to 128, i.e. 4 K-iterations) -- both pass cleanly.

**Found a genuine, SEPARATE, pre-existing bug**: any problem needing MULTIPLE workgroups along
`gemm_m` (`grid_x>1`) reports `valid:n` -- for BOTH fwd and bwd. Bisected this down to
`wmma_epilogue_chunked` itself, completely independent of VGPR-MSB or tile size: it reproduces
identically at the ORIGINAL 128x128 tile (`wmma_acc_high_bank` not even set) with
`wmma_epilogue_chunked=1` and `grid_x>1`. Confirmed via `PRINT_NRMS=1`: single-workgroup nrms
is `~0.0006` (normal bf16 rounding, well under the `0.0082` tolerance); multi-workgroup nrms is
`~0.19-0.20` (250-300x over tolerance) -- consistently reproducible across multiple runs with
different random data each time (ruling out a flaky race, pointing at a deterministic logic
bug). Bisected further: multiple K-blocks alone is fine (`valid:y`); multiple N-blocks alone is
fine (tested at 256x256, `k=512`); it's specifically multiple **M**-blocks (`grid_x>1`) that
breaks it, for both directions.

**Root cause FOUND AND FIXED.** Initial algebraic check of the compact-row-to-native-row
recovery formula found no error -- because the formula itself is fine; the bug is in one of
its *inputs*. `wmma_main_loop.py`'s/kernel prologue's `v_gemm_im` gets `s_block_m_off`
**permanently folded in** right after it's computed (`v_add_u32 v_gemm_im, s_block_m_off,
v_gemm_im`) -- so from that point on, `v_gemm_im` is a GLOBAL row, not the tile-local value
the chunked scatter's compact-row derivation assumes. `block_m_off` is always an exact
multiple of `macro_tile_m` (hence of `wave_tile_m*wave_repeat_m` too), so for workgroup `bx=0`
(`block_m_off=0`) this was completely invisible -- but for `bx>=1`, right-shifting the
now-nonzero-in-those-bits `v_gemm_im` by `log2(wave_tile_m*wave_repeat_m)` computes a
`wave_idx` that's 2 or 3 too high (out of the valid 0/1 range), scattering into LDS bytes
entirely outside the small per-group chunked region. The sibling **unchunked** path was
already safe because it explicitly masks (`(v_gemm_im+row_off) & (macro_tile_m-1)`) before
using `v_gemm_im` for LDS addressing -- the chunked path's scatter was simply missing the
equivalent mask.

**Found via a position-fingerprint diagnostic**, more decisive than the raw LDS dump:
replaced the scatter's real accumulator DATA with a small decodable integer encoding
`(i_rm, j, i_rn)` (same value for every lane, e.g. `i_rm*1000+j*10+i_rn`), left the
unmodified real gather/store pipeline to carry it to global memory, then read back via
`DUMP_PRED`'s raw hex. Workgroup `bx=0`'s output showed clean, correctly-decodable
fingerprints matching hand-computed expectations at every position (e.g. index 128 = row 1 →
`0x0a` = `i_rm=0,j=1,i_rn=0`, exactly as predicted). Workgroup `bx=1`'s entire output showed
**zero clean fingerprints anywhere** -- not a wrong-but-decodable value (which would indicate
a formula error), but what looked like leftover/uninitialized memory, proving the scatter
was never landing its writes anywhere the gather could find them for that workgroup.

**Fix**: mask `v_gemm_im` to tile-local range (`& (macro_tile_m-1)`) before deriving
`wave_idx`/`lane_sub` in `coalescing_store_wmma.py`'s `_emit_chunked_non_atomic_store` scatter
setup, matching the unchunked path's existing discipline.

**Hardware-validated `valid:y`** across every combination tried: fwd 128x128
(`wmma_epilogue_chunked=1`, no VGPR-MSB) and fwd 256x256 / bwd 256x128 (both with
`wmma_acc_high_bank=1`), each with single-workgroup, multi-M-block (2 and 3+ blocks),
multi-N-block, multi-K-block, and combinations of all three. `PRINT_NRMS=1` confirms normal
bf16 rounding-level error (`~0.0004-0.0006`, previously `~0.19-0.20` for the broken
multi-workgroup case) in every case. int8 128x128 chunked + multi-workgroup also confirmed
`valid:y`. Zero regression confirmed (byte-identical `.s` for existing non-`wmma_epilogue_chunked`
configs).

**Config status**: both `config/igemm_fwd_gtc_gfx1250_nhwc_bf16_256x256.config` and
`config/igemm_bwd_gtc_gfx1250_nhwc_bf16_256x128.config` are updated (full F32 accumulate) and
hardware-validated correct for single- AND multi-workgroup problems. Phase 55 is DONE.

Two diagnostic hooks are kept in the tree (env-gated, off by default, zero cost): `CHUNK_RAW_DUMP=1`
(raw LDS dump, less decisive) and `CHUNK_FINGERPRINT=1` (position fingerprinting, the one that
actually found this bug) in `coalescing_store_wmma.py`'s `_emit_chunked_non_atomic_store`.

## How to resume on a different machine

Phase 54's 128x128 mechanism-only test is DONE (`valid:y` across fwd bf16/fp16/int8/fp32, both
epilogue implementations, and bwd bf16). Phase 55's 256x256/256x128 tile work is ALSO DONE, for
single- AND multi-workgroup problems (the `wmma_epilogue_chunked` masking bug is fixed). No need
to re-litigate either of the above -- this whole doc is now a closed investigation, kept for
reference and for the diagnostic techniques (rocgdb, raw-dump hooks, position fingerprinting)
that generalize to future gfx1250 debugging.

1. Clone/pull this branch (`users/SreecharanGundaboluAMD/gfx1250_bringup`).
2. Confirm the new machine has a gfx1250 GPU (`rocminfo | grep gfx1250`) and the ROCm toolchain
   (`clang++`/`llvm-objdump` targeting gfx1250; path was `/opt/rocm/llvm/bin/` this session).
3. Next real work is genuinely new, not a continuation of this doc's investigation -- e.g. wider
   tile-shape/precision coverage for `wmma_acc_high_bank`+`wmma_epilogue_chunked` (only bf16
   fwd/bwd tested this session; int8/fp16/fp32 at 256-size tiles, and wrw entirely, are
   untested), or performance tuning now that correctness is established (double-buffering the
   256x256 main loop, since the chunked epilogue's small LDS footprint may leave enough headroom
   -- see Phase 52's note that gfx950's own tuned instance sacrificed this).
4. `MSB_RAW_DUMP=1` (`igemm_fwd_gtc_wmma_nhwc.py`'s `emit_kernel_epilogue()`),
   `CHUNK_RAW_DUMP=1`, and `CHUNK_FINGERPRINT=1` (both in `coalescing_store_wmma.py`'s
   `_emit_chunked_non_atomic_store`) remain in the tree as reusable diagnostics, off by default --
   the fingerprint technique in particular (replace real data with a decodable per-position
   constant, let the real address/data-movement pipeline carry it, decode the output) is a
   generally useful pattern for any future addressing-bug hunt in this codebase.
