# gfx1250 WMMA register layout (empirically verified)

Verified on real gfx1250 hardware (this machine) via `/tmp/wmma_probe/{probe.s,host.cpp}`:
an end-to-end round-trip test that loads random 16x32 (A) and 32x16 (B) fp16 matrices into
VGPRs per the formulas below, executes `v_wmma_f32_16x16x32_f16`, reads back the D operand
per the formula below, and compares against a CPU-computed reference. Matched exactly across
5 random-seed trials.

## `v_wmma_f32_16x16x32_f16 D, A, B, C` (M=16, N=16, K=32, fp16 in / fp32 accum)

Wave32. For lane `l` (0..31):

- **A operand** (8 VGPRs/lane, packed 2 fp16/dword): for vgpr index `a` (0..7), half `s` (0/1):
  `row = l % 16`, `k = (l / 16) * 16 + a*2 + s`
- **B operand** (8 VGPRs/lane, same packing): for vgpr index `a` (0..7), half `s` (0/1):
  `col = l % 16`, `k = (l / 16) * 16 + a*2 + s`
- **C/D operand** (8 VGPRs/lane, fp32): for vgpr index `j` (0..7):
  `row = (l / 16) * 8 + j`, `col = l % 16`

In words: the 32-lane wave splits into two 16-lane halves along `l/16`. Each half independently
covers all 16 rows (`l%16`) of A and all 16 columns (`l%16`) of B, but only half the K range
(k=0..15 for the lower half, k=16..31 for the upper half) — the two halves' partial products are
summed by the instruction itself. For the output, the two halves instead split the **M** range
(rows 0..7 vs 8..15), each producing the full 16 columns.

C and D may be distinct register ranges (no aliasing requirement) — confirmed via `llvm-mc`
accepting `v_wmma_f32_16x16x32_f16 v[40:47], v[8:15], v[16:23], v[0:7]`.

## `v_wmma_f32_16x16x32_bf16` (M=16, N=16, K=32, bf16 in / fp32 accum)

**Verified** — identical formula to the fp16 case above (same 8-VGPR/lane footprint, same 2
elements/dword packing). Confirmed via the same round-trip technique across 3 random-seed
trials (`/tmp/wmma_probe/probe_bf16.s`, `host_bf16.cpp`).

## Not yet re-verified for other instructions

Only `v_wmma_f32_16x16x32_f16` and `v_wmma_f32_16x16x32_bf16` (both K=32, identical footprint)
are verified. Do **not** assume the same formula for the K=64 (`v_wmma_i32_16x16x64_iu8`), K=128
(`v_wmma_f32_16x16x128_fp8_fp8`, note its A/B footprint is 16 VGPRs/lane, not 8 — a structurally
different layout, not just a wider `k` range), or K=4 (`v_wmma_f32_16x16x4_f32`) variants — the
D-operand (output) layout is expected to carry over (M=N=16, `num_v_c=8` for all of them), but
the A/B operand layout must be independently re-verified with the same round-trip probe
technique before trusting it for a new instruction.

## Workgroup ID delivery (gfx1250-specific, not WMMA-specific — relevant to any multi-workgroup kernel)

gfx1250 delivers `blockIdx.x`/`blockIdx.y` via **`ttmp9`/`ttmp7`** (trap-temporary registers),
**not** classical pre-loaded system SGPRs — regardless of the `.amdhsa_system_sgpr_workgroup_id_x/y`
kernel-descriptor flags, which are accepted by the assembler but do not correspond to where the
hardware actually places the value on this ROCm/LLVM build. Found by compiling and disassembling
a trivial HIP kernel that reads `blockIdx.x`/`.y` (`hipcc -x hip --offload-arch=gfx1250
--cuda-device-only -c -O0`, then `clang-offload-bundler --unbundle` + `llvm-objdump -d`) — the
compiler's own generated code is ground truth for the ABI it targets. The generated code also
does a runtime check (`s_getreg_b32 s4, hwreg(HW_REG_IB_STS2, 6, 4)` compared against 0) with a
fallback bitfield-decode path when nonzero — almost certainly gating between plain dispatch and
gfx1250's new "workgroup cluster" feature; `s_mov_b32 sX, ttmp9`/`ttmp7` is the simple,
non-clustered (default) case. Verified independently on real hardware via a multi-workgroup
atomic-slot dispatch test. No official documentation found for this — discovered purely by
disassembly; re-verify if targeting a different ROCm/LLVM version.

## Accumulator initialization

`v_wmma_*` computes `D = A@B + C`. If a kernel reuses the same VGPR range for both the C and D
operand across the whole main loop (the natural choice, since WMMA has no separate
accumulator-clear instruction), that range **must be explicitly zeroed before the first use** —
it is not zero at kernel entry. Skipping this produces a kernel that runs to completion and
looks plausible but silently accumulates onto garbage VGPR contents.

## GCN12.5 ISA quirks found along the way (relevant to any gfx1250 codegen, not just WMMA)

- `.amdhsa_ieee_mode`, `.amdhsa_dx10_clamp`, `.amdhsa_workgroup_processor_mode` kernel-descriptor
  directives are all rejected by the assembler on gfx1250 (fixed in `python/codegen/amdgpu.py`).
- Carry-out arithmetic uses `vcc_lo` (not bare `vcc`) and `v_add_co_ci_u32` (not `v_addc_co_u32`).
- Waitcnt is split into separate `s_wait_loadcnt` / `s_wait_storecnt` / `s_wait_dscnt` /
  `s_wait_kmcnt` instructions instead of a single `s_waitcnt vmcnt()/lgkmcnt()`.
- The asm→hsaco recipe must use `-target amdgcn-amd-amdhsa` (full triple with "amd" vendor);
  the vendor-less `amdgcn--amdhsa` triple used in some older MISA test scripts
  (`test/persistent_workgroup/build.sh`) produces a target-id mismatch against the
  `.amdgcn_target` directive and fails to assemble.
