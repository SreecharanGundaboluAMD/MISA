# Claude persistent memory notes (session snapshot, 2026-08-28)

This file is a plain-text export of the persistent, cross-session memory Claude Code
built up while working on this repo (stored outside the repo, under
`~/.claude/projects/.../memory/`). It's committed here so the lessons travel with the
repo across machines, independent of any one assistant's local memory directory.

## Index

- **conv_driver.exe mode string** — always use `convfp16`/`convbfp16`/`convint8`, not
  plain `conv`, for non-fp32 kernels, or every result silently reports `valid:n`.
- **gfx1250 VGPR-MSB** — 128x128 WMMA tile ceiling is a codegen limit, not
  hardware/toolchain; VGPR-MSB works on real hardware, but Phase 54's kernel
  integration is WIP, paused for a machine migration + GPU reboot.
- **GPU hardware debug technique** — use `rocgdb` to find the faulting PC on
  real-hardware crashes before writing synthetic repro kernels.
- **gfx1250 WMMA hang risk** — back-to-back same-register WMMA with zero interleaving
  hung the GPU; read the ISA doc's WMMA-hazard section before retrying.

---

## conv_driver.exe mode string

*(feedback)*

`conv_driver.exe`'s positional mode argument selects verification precision:
`conv`=fp32, `convfp16`=fp16, `convbfp16`=bf16, `convint8`=int8, `convint4`=int4 (mirrors
MIOpenDriver's convention). Always match it to the tunable's `precision` field.

**Why:** running any fp16/bf16/int8/int4 config with the plain `conv` mode string does not
error — it silently generates/casts test data as fp32 and compares against a kernel that
actually reads packed lower-precision data, so every candidate reports `pred:-nan`,
`valid:n`. This is indistinguishable from a genuine kernel/hardware/driver bug from the
output alone. On 2026-08-27 this cost an entire multi-hour session chasing what looked like
a total environment-wide regression (WMMA, LDS, barriers, a machine reboot, a git bisect to
the project's first-ever commit) before the mode string was found to be the actual cause —
see `docs/gfx1250_wmma_layout.md`'s Phase 45 note and `README.md`'s conv_driver.exe section
in the MISA repo for the full writeup.

**How to apply:** before treating any `conv_driver.exe pred:-nan`/`valid:n` result as a real
bug, first double check the mode string in the exact command that produced it. If it's
`conv` for a config whose precision isn't fp32, that is very likely the whole story — rerun
with the correct mode string (`convfp16`/`convbfp16`/`convint8`/`convint4`) before
investigating further.

---

## gfx1250 VGPR-MSB

*(project)*

MISA's gfx1250 WMMA kernels (`python/igemm/igemm_{fwd,bwd,wrw}_gtc_wmma_nhwc.py`) are
capped at a 128x128 macro-tile. Root cause (Phase 52/53,
`docs/gfx1250_wmma_layout.md`): not the LDS ceiling (fixed via a chunked epilogue
mirroring XDLOPS's `coalescing_groups`) but a 256-VGPR-per-wave ceiling -- even with
packed accumulate (`wmma_acc_bf16`/`wmma_acc_f16`), the best-reachable 256x256 config
needs a minimum 284 VGPRs, 28 over budget, with no tile-shape/wave-split choice closing
the gap.

**Corrected finding (important -- an earlier version of this memory got this wrong):**
`S_SET_VGPR_MSB` (ISA doc §3.3.2.3, up to 1024 VGPRs/wave) IS fully usable through this
project's existing toolchain (ROCm 10.1, AMD LLVM 23.0.0git, gfx1250). The first
investigation tested it wrong -- tried literal `v256`/`v[256]` operand syntax, got
"register index is out of range," and wrongly concluded the toolchain didn't support
the extended range. That's not how the mechanism works: instruction encoding always
uses the plain 0-255 register field; `S_SET_VGPR_MSB` supplies the extra 2 bits via the
wave's MODE state at *runtime*, independently per DST/SRC0/SRC1/SRC2 slot. "Hardware
v300" is written as ordinary `v44` (300-256) with the right slot's MSB set to `01` --
never as literal `v300` syntax. Retested correctly (plain low-range syntax + a
preceding `s_set_vgpr_msb`) and confirmed via a full `llvm-mc` assemble +
`llvm-objdump` disassemble round-trip -- the disassembler even resolves and annotates
the true hardware address in comments (e.g. `v_add_f32_e32 v10 /*v522*/, ...`), and the
VDS ADDR/DATA0/DATA1-to-slot mapping matched the doc's table exactly.

**What this means**: the 256-register ceiling is a codegen limitation (this project's
register allocator and every instruction-emission site assume flat 0-255 addressing
with zero bank tracking), not a hardware or toolchain one. Implementing real support
means: (1) a register allocator that tracks a bank per symbol once allocation exceeds
256, (2) instruction-emission logic that knows each instruction format's DST/SRC0/
SRC1/SRC2-to-physical-operand mapping (VOP1/VOP2/VOP3/VOP3P/VOPD/VDS/VFLAT/VBUFFER/
VIMAGE all differ), and (3) emitting `S_SET_VGPR_MSB` only when the needed bank
combination actually changes from the previous instruction (not before every VALU op).
This is genuine, substantial codegen work, actively in progress as Phase 54
(`docs/gfx1250_wmma_layout.md`, `docs/gfx1250_optimization_backlog.md` Tier 3).

**Why**: user asked (2026-08-28) to drop the 256-VGPR tile-shape-fitting effort
entirely and focus on implementing VGPR-MSB directly, since it's the real fix for the
128x128 tile ceiling that's been holding gfx1250 WMMA back against gfx950's XDLOPS.
**How to apply**: don't re-derive the "is VGPR-MSB usable" question from scratch --
it's settled (yes, via plain-syntax + bank-select, confirmed by the round-trip above).
Check `docs/gfx1250_wmma_layout.md` Phase 54 for implementation status before starting
or resuming this work.

**Update (2026-08-28, end of session, paused for a machine migration + GPU reboot):**
implementation attempted, real progress, not yet working. The mechanism itself checked
out on ~15 standalone hardware tests (VALU/DS/VMEM/multi-wave/aligned-WMMA all
round-trip through bank 1 correctly) -- every bug found was in this project's own new
codegen (`vgpr_msb_tracker_t` + its call sites), not the hardware. Found and fixed:
a pre-existing Phase 53 bug that silently discarded the whole chunked epilogue's
output (a `return` inside a `with self._deferred_context():` block, evaluated before
`__exit__` populates the buffer), a related Phase 53 addressing bug (assumed compact-
row and true-row advance linearly per gather pass -- false across a `wave_tile_m`
boundary, now fixed and hardware-confirmed `valid:y` standalone), a tracker design
flaw (`ensure()`'s memoization assumes emission order == runtime order, unsafe across
`wmma_main_loop.py`'s real branches -- added `force()` for that call site), and the
actual crash's root cause: the accumulator zero-init sets `dst=1` and nothing reset it
before the rest of the prologue's ordinary bank-0 VGPR writes (found via `rocgdb`
pointing straight at the faulting PC in ~minutes, after hours of guessing via synthetic
repros found nothing -- **use rocgdb first next time**, see
`docs/gfx1250_wmma_vgpr_msb_wip_status.md` for the exact command). After that fix the
kernel no longer crashes but still reports `valid:n` (wrong results, root cause not yet
found). Session ended because a synthetic back-to-back-same-register WMMA repro (zero
interleaving, testing a possible hardware hazard the ISA doc warns about) hung the GPU,
requiring a reboot. Everything is committed (commit `1b0ec35` on
`users/SreecharanGundaboluAMD/gfx1250_bringup`) including the ISA doc (committed this
one time at explicit user request for the migration -- the "never commit this file"
convention below still stands for future sessions unless the user asks again).
Full resume instructions, the exact repro config, and a hardware-safety warning are in
`docs/gfx1250_wmma_vgpr_msb_wip_status.md` -- read that file first when picking this
back up, not this memory alone.

---

## GPU hardware debug technique

*(feedback)*

When a MISA-generated kernel crashes on real gfx1250 hardware (e.g.
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`), reach for `rocgdb` immediately instead of
guessing-and-checking with hand-written standalone repro kernels first.

```
export HSA_ENABLE_DEBUG=1
rocgdb -batch -x gdb_cmds.txt --args ./conv_driver.exe <same args that crash>
```
where `gdb_cmds.txt` contains:
```
set pagination off
run <same args, gdb also needs them after `run`>
bt
info registers
thread apply all bt
```
The crashing wave's `pc` in `info registers` points at the exact faulting instruction
and its containing label. Match that back to source by disassembling the built
`.hsaco` (`llvm-objdump -d --mcpu=gfx1250 <name>.hsaco`) and finding the same byte
offset.

**Why**: during Phase 54 (VGPR-MSB) debugging (2026-08-28), hours were spent writing
~15 standalone isolated repro kernels trying to reproduce a crash via hypothesis
(alignment, branch safety, async-load races, etc.) — all passed, none found the actual
bug. Running the real failing command directly under `rocgdb` found the true root
cause (a stale register-bank state left over from an earlier codegen phase) in
minutes: the PC pointed at a completely unrelated, ordinary instruction nowhere near
where the hypotheses were focused.

**How to apply**: for any future "runs but crashes on real hardware" bug in this
project, get the rocgdb backtrace/PC *first*, before writing any synthetic isolated
test. Only build synthetic repros once you know which code path is actually
implicated. See "gfx1250 VGPR-MSB" above and `docs/gfx1250_wmma_vgpr_msb_wip_status.md`
for the full worked example.

---

## gfx1250 WMMA hang risk

*(project)*

On this project's gfx1250 dev hardware (2026-08-28), a minimal standalone test kernel
that issued 4 `v_wmma_f32_16x16x32_bf16` instructions back to back, each reading the
immediately-preceding call's C/D output from the same registers with zero other
instructions in between, hung the GPU. `rocm-smi` showed 100% GPU utilization even
after the host-side process was killed; a subsequent, unrelated kernel launch timed
out after 2 minutes. Required a machine reboot to clear.

The ISA doc (`amd-instinct-cdna5-instruction-set-architecture.md`, section 5.7) warns:
*"Hardware does not detect data dependencies between co-executing multicycle
instructions from the same wave... See Requirements for WMMA data hazards for more
information on WMMA hazard avoidance."* That referenced section was not located/read
before this hang happened — read it first if testing anything like this again.

MISA's actual generated kernels (the real main loop) always have a barrier + several
`ds_read`s between one iteration's WMMA burst and the next iteration's WMMA burst that
touches the same accumulator registers, so this exact degenerate pattern (zero
interleaving) doesn't match real usage and may not be the actual risk in practice --
but it's untested whether the real kernel's existing interleaving is *provably*
sufficient, or just accidentally been enough so far.

**Why**: found while debugging Phase 54 (VGPR-MSB) — see "gfx1250 VGPR-MSB" above and
`docs/gfx1250_wmma_vgpr_msb_wip_status.md` for full context.

**How to apply**: before running any hand-written test that stacks multiple WMMA
instructions with a same-register dependency chain, (1) read the ISA doc's WMMA-hazard
section first, (2) always run such tests under a killable/backgroundable process with
a short timeout, (3) prefer patterns that include real interleaving (matching actual
generated kernels) over maximally-tight synthetic repros, and (4) have a GPU
reset/reboot plan or explicit permission before running anything like this on a
shared or hard-to-access machine.
