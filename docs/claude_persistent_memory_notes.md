# Claude persistent memory notes (session snapshot, 2026-08-28, updated)

This file is a plain-text export of the persistent, cross-session memory Claude Code
built up while working on this repo (stored outside the repo, under
`~/.claude/projects/.../memory/`). It's committed here so the lessons travel with the
repo across machines, independent of any one assistant's local memory directory.

## Index

- **conv_driver.exe mode string** — always use `convfp16`/`convbfp16`/`convint8`, not
  plain `conv`, for non-fp32 kernels, or every result silently reports `valid:n`.
- **gfx1250 VGPR-MSB** — 128x128 WMMA tile ceiling is a codegen limit, not
  hardware/toolchain. Phase 54 is DONE for 128x128: both the dst=1-never-reset bug and
  a `coalescing_store_wmma.py` row-advance MSB-bank bug are found, fixed, and
  hardware-verified `valid:y` across fwd bf16/fp16/int8/fp32 + bwd bf16, both epilogues.
  Phase 55: resumed 256x256/256x128 tile configs (full F32 accumulate, no more packed
  workaround) — also found+fixed a separate `wmma_epilogue_chunked` masking bug that
  broke every `grid_x>1` workgroup. Both configs now `valid:y` single+multi-workgroup.
  Phase 56: double-buffer + packed-accumulate tried for perf, neither closes the
  2-3x gap vs 128x128 — don't extend to wrw/other precisions yet. Phase 57: int8/int4
  split-K atomic epilogue fixed (code-level only, not hw-validated, deprioritized);
  64x64_kmax wrw tile now hw-validated `valid:y`. Also: wrw's old "100-1660x slower than
  MIOpen" number is OBSOLETE (predates wrw split-K) — current numbers already tracked in
  `docs/gfx1250_vendor_benchmark_vs_miopen.md` (~2-5x slower average, updated 2026-08-27),
  spot-confirmed 2026-08-28.
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

**Update (2026-08-28, new machine, post-migration): the dst=1-never-reset bug is fixed
and hardware-verified, but it was necessary, not sufficient.** A different machine had
independently RCA'd the exact bug described above (`emit_wmma_tile()` sets `dst=1` before
every WMMA burst, nothing ever reset it back to 0, so every bank-0 VGPR write in between
bursts -- `ds_read_b128` for `v_a`/`v_b`, `global_load_dwordx4` into `v_gld_a`/`v_gld_b`,
`emit_buffer_switch()`'s `v_xor_b32` -- was silently landing in bank 1) but crashed before
it could hardware-validate its own fix. Re-applied the fix on this machine (as the LAST
thing `emit_wmma_tile()` does, right after the burst, rather than patching each individual
consumer call site -- more robust, since there were three separate consumer classes, not
just the one the other machine had noticed). Verified via disassembly (`s_set_vgpr_msb 0`
now appears after every burst; llvm-objdump's own physical-address resolution, e.g.
`v0 /*v256*/`, confirms correct bank-1 addressing) and confirmed zero regression (every
existing non-`wmma_acc_high_bank` config's `.s`/`.inc` is byte-identical before/after).

**On real hardware: the crash-class symptom is gone, and the remaining `valid:n` is RESOLVED to
an ordinary software bug, NOT a hardware/VGPR-MSB issue.** Initial per-pixel diffing on bf16
alone was misleading (the diff-printer's 100-line cap was consumed by small errors in output row
`M=0` before ever showing row `M>=1`). Re-tested with `PRINT_EVERY_PIXEL=1` (no cap) across all
four precisions: every precision showed the identical fault line -- row `M=0` correct, every row
`M>=1` corrupted (int8: exact constant `0x01010101`; others: garbage unrelated to reference).

Two independent tests then proved `v_c` itself is NOT the problem: (1) a standalone repro built
up incrementally to the real kernel's exact scale (16 groups, 4 waves, 128 bank-1 registers,
distinct `v_a`/`v_b` per group, in-flight prefetch during the WMMA burst -- matching
`emit_wmma_tile()`'s real indexing exactly) passed with zero mismatches at every scale tested;
(2) a raw-dump diagnostic wired directly into the real kernel (`MSB_RAW_DUMP=1` env var in
`igemm_fwd_gtc_wmma_nhwc.py`'s `emit_kernel_epilogue()`, kept in the tree, off by default,
zero cost) that bypasses `coalescing_store_wmma.py` entirely and dumps `v_c` straight to
output -- every one of 16384 raw dwords came back exactly correct, for the SAME config that
fails through the real epilogue. Repeated bank-toggling and a full LDS round-trip with trivial
flat addressing also both came back 100% correct.

**Conclusion**: `v_c` is unconditionally correct after the main loop; VGPR-MSB, multi-group
accumulation, multi-wave sync, and even a full LDS round-trip all work fine standalone.

**FOUND AND FIXED the actual bug, same session, immediately after the above.** Hand-tracing the
generated `.inc` (not just the Python source) found it directly: in `coalescing_store_wmma.py`'s
inner `for j in range(inst_wmma.num_v_c):` scatter loop (both the unchunked path and the chunked
path's identical twin), the per-`j` row-advance `v_add_u32 v[v_tmp1], <stride>, v[v_tmp1]` is
VOP2 -- the immediate lands in SRC0, but `v_tmp1` (meant to be an ordinary bank-0 address) lands
in the VSRC1 slot, which is the SAME slot the surrounding `ds_write_b32`'s `v_c` DATA operand
uses via `src1=1` (set once before the whole `j` loop, not reset until it ends). So the address
advance silently reads its own current value from BANK 1 (garbage) instead of bank 0, for every
`j>=1` -- while `j=0` (no advance needed yet) is unaffected. Exactly matches every symptom: first
row/group always correct, everything after it corrupted, identically regardless of precision or
epilogue implementation (both have the bug independently). Fixed by bracketing the row-advance
with `ensure(src1=0)`/`ensure(src1=1)`, matching the discipline the OUTER `i_rm` loop already
used for its own address computation.

**Hardware-verified fixed**: `valid:y` for fwd bf16/fp16/int8/fp32 (both unchunked and
`wmma_epilogue_chunked=1`) and bwd bf16. bf16/fp16's earlier "separate small M=0 noise" was NOT a
separate bug -- same root cause, just looked like rounding noise at bf16's smaller magnitudes.
Zero regression confirmed. Phase 54's VGPR-MSB mechanism is now genuinely done and correct for
the 128x128 test across every precision and both epilogues.

**Phase 55 (same day): resumed the 256x256/256x128 tile configs -- DONE, including a second
real bug found and fixed.** Switched the configs from `wmma_acc_bf16=1` (packed,
precision-losing, and per Phase 53's own math STILL 28 registers short of fitting) to
`wmma_acc_high_bank=1` with full F32 accumulate (256 registers, fits easily in bank 1 -- total
wave VGPR request 512, confirmed via `.vgpr_count`). Added the missing full-F32
`ctrl_wmma_mapping_t` table entries (`wmma_mapping.py`'s `'bf16'` key only had packed-accumulate
256-size entries before).

Multi-workgroup (`grid_x>1`) initially reported `valid:n` for both directions -- a genuine,
separate, pre-existing bug, unrelated to VGPR-MSB or tile size (reproduced at the ORIGINAL
128x128 tile with just `wmma_epilogue_chunked=1`). This exact code path had never been
hardware-tested with `grid_x>1` before (Phase 53 was parked before ever running a
multi-workgroup test). **Root cause**: `v_gemm_im` has `s_block_m_off` permanently folded in by
the prologue, so it's a GLOBAL row from that point on -- but the chunked scatter's compact-row
derivation (`wave_idx = v_gemm_im >> log2(wave_tile_m*wave_repeat_m)`) used it directly with no
masking. `block_m_off` is always a multiple of `macro_tile_m`, so `bx=0` never exposed it;
`bx>=1` computed a `wave_idx` 2-3 too high, scattering into completely wrong LDS bytes. The
sibling unchunked path was already safe (masks `v_gemm_im` before use) -- the chunked path was
just missing the equivalent mask.

**Found via a position-fingerprint diagnostic** (`CHUNK_FINGERPRINT=1`), more decisive than a
raw LDS dump: replaced the scatter's DATA with a small decodable integer encoding
`(i_rm,j,i_rn)`, left the real gather/store pipeline untouched, decoded via `DUMP_PRED`'s raw
hex. `bx=0`'s output showed clean, correctly-decodable fingerprints matching hand-computed
expectations everywhere; `bx=1`'s showed ZERO clean fingerprints anywhere (leftover/
uninitialized-looking memory instead) -- proof the scatter was never landing writes where the
gather could find them for that workgroup. **Fixed** by masking `v_gemm_im` to tile-local range
before deriving `wave_idx`/`lane_sub`, matching the unchunked path's existing discipline.

**Hardware-validated `valid:y`**: fwd 128x128 (no MSB) and fwd 256x256/bwd 256x128 (with MSB),
each across single-workgroup, multi-M-block (2 and 3+), multi-N-block, multi-K-block, and
combinations. int8 128x128 chunked+multi-workgroup also confirmed. `PRINT_NRMS=1` back to
normal bf16 rounding levels (~0.0004-0.0006, was ~0.19-0.20 broken). Zero regression confirmed.
Both 256-size configs are DONE -- correct for single- and multi-workgroup problems, both
directions -- see `docs/gfx1250_wmma_vgpr_msb_wip_status.md`'s Phase 55 for full detail. The
`CHUNK_FINGERPRINT` technique (decodable per-position constant + real pipeline + decode output)
is a good reusable pattern for future addressing-bug hunts in this codebase.

**Phase 56** (same day): tried to make the 256x256/256x128 tile a performance win via
`lds_double_buffer=1` and packed accumulate (`wmma_acc_bf16`). Neither closes the gap --
still 2-3x slower than 128x128 at every scale tried. Packed accumulate alone gives a
consistent ~10-12% improvement over plain (kept as opt-in `*_bf16acc.config` files) but
doesn't get close to parity. Don't re-attempt either as a fix, and don't extend this tile
to wrw/int8/fp16/fp32 until the remaining gap (chunked-epilogue barrier overhead vs.
fewer/bigger-workgroup occupancy loss) is root-caused via rocprof on an idle GPU.

**Phase 57** (2026-08-28, separate item, not VGPR-MSB): int8/int4's `gemm_k_global_split`
atomic epilogue always emitted `global_atomic_add_f32`, silently bit-reinterpreting the
genuine int32 WMMA accumulator as float (only bit-exact for small non-negative sums).
Fixed to emit `global_atomic_add_u32` for int8/int4 (`coalescing_store_wmma.py`), removed
the blocking assert in `igemm_base.py`. Code-level fix only -- not hardware end-to-end
validated with genuinely signed/large-magnitude data, deprioritized per explicit user
direction (int8/int4 isn't a current focus). Also hardware-validated (this session) the
previously-untested 64x64_kmax wrw tile shape from an earlier, unrelated backlog item:
`valid:y` for bf16/fp16/fp32 on an exact-fit shape.

**Correction**: an earlier memory (`gfx1250-vendor-benchmark-vs-miopen`, 2026-08-25) found
wrw 100-1660x slower than MIOpen, root-caused to "no `gemm_k_global_split` support in the
WMMA path at all". That was correct when written (wrw's first split-K port landed ~12h
after that benchmark ran) but is now OBSOLETE. **This repo already has a much more
thorough, current answer**: `docs/gfx1250_vendor_benchmark_vs_miopen.md` (updated through
2026-08-27) re-benchmarked all 10 applicable wrw shapes multiple times with the gsplit fix
in place — current average is ~2.17x-5.02x slower than MIOpen (the spread attributed to
GPU contention, confirmed not a search-algorithm issue via manual sweep), not 100-1660x.
fwd/bwd remain at parity-to-modest-slowdown (~1.2-1.75x avg, several shapes faster).
Spot-checked 2 of the same shapes on 2026-08-28 (0.031ms vs. old 8.0ms/MIOpen's 0.035ms;
1.655ms vs. old 1.18s/MIOpen's ~0.7ms) — consistent with that doc's numbers. **Read
`docs/gfx1250_vendor_benchmark_vs_miopen.md` for current wrw-vs-MIOpen numbers, not the old
100-1660x figure.**

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
