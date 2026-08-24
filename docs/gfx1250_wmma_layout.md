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

## `v_wmma_i32_16x16x64_iu8` (M=16, N=16, K=64, int8 in / int32 accum)

**Verified** — same 8-VGPR/lane footprint as fp16/bf16, but 4 int8/dword (not 2), so each
vgpr/lane covers 4 k-values instead of 2:

- **A operand**: for vgpr index `a` (0..7), byte `s` (0..3): `row = l % 16`,
  `k = (l/16)*32 + a*4 + s`
- **B operand**: for vgpr index `a` (0..7), byte `s` (0..3): `col = l % 16`,
  `k = (l/16)*32 + a*4 + s`
- **D operand**: identical to the fp16/bf16 case (`row = (l/16)*8 + j, col = l % 16`), but the
  32-bit accumulator elements are `int32`, not `fp32`.

Confirmed via the same round-trip technique across 5 random-seed trials, using small
**unsigned** (0..8) test values (`/tmp/wmma_probe/probe_int8.s`, `host_int8.cpp`) — sidesteps
the `neg_lo` signedness modifier (confirmed via `llvm-mc`: the base encoding with no modifier
has `neg_lo` bits all 0, i.e. defaults to unsigned interpretation of both A and B; `clamp` and
`neg_lo:[a,b,c]` both assemble too, but haven't been exercised).

A pleasant consequence of `gemm_k_per_block` always being chosen to equal `inst_wmma.k`, and
`num_v_a`/`num_v_b`/`num_v_c` being 8/8/8 for every wired-up instruction so far: quantities like
"bytes per LDS tile row" (`gemm_k_per_block * data_byte`) and "bytes per wave_repeat step" come
out **precision-invariant** (always 64 bytes and 1024 bytes respectively) because element width
and per-block element count scale inversely. `igemm_fwd_gtc_wmma_nhwc_t` needed exactly two
literal-shift fixes (the A/B global-address stride multiplier) to add int8 support — see that
file's class docstring.

**Validated on real gfx1250 hardware** (`igemm_fwd_gtc_wmma_nhwc_t`, 128x128x64 tile): exact
match against a CPU reference (int32, no tolerance needed) across single-block, multi-block,
non-square, and multi-K-block configurations, multiple random seeds.

## Not yet re-verified for other instructions

`v_wmma_f32_16x16x32_f16`, `v_wmma_f32_16x16x32_bf16` (K=32), and `v_wmma_i32_16x16x64_iu8`
(K=64) are verified. Do **not** assume the same formula for K=128
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

## LDS-transpose read for operands whose natural memory layout doesn't match the WMMA
## operand orientation (bwd's weight operand, and wrw's grad_output/input operands)

Some GEMM role-assignments store an operand `[gemm_k rows][gemm_m or gemm_n contiguous]`
(k-major, row-contiguous in the non-contraction dim) instead of the `[gemm_m or
gemm_n][gemm_k]` (row-major, k-contiguous) orientation the fwd kernel's operands happen to
have naturally. bwd's weight operand is the first example: physically `[K_out][C_in]`
(the same buffer/layout fwd reads), but bwd binds `GEMM_K=K_out`, `GEMM_N=C_in` — the
opposite of what a WMMA B-operand load wants.

Rather than transposing through an extra LDS pass, global load and LDS store are left
completely unchanged (still contiguous, still writing the natural `[K][N]` layout into LDS);
only the WMMA-consumption-time LDS **read** changes, from 2x `ds_read_b128` (contiguous) to
16x `ds_read_u16` (strided, one per row, `row_pitch_bytes` apart) + 8x `v_lshl_or_b32` (pack
pairs into dwords) per wave_repeat step. This trades LDS-instruction count for correctness/
auditability — an explicitly deferred optimization, same tradeoff
`coalescing_store_wmma.py` already documents for its own epilogue.

New helper: `igemm_wmma_mapping_t.get_gemm_index_for_src_matrix_transposed()` in
`wmma_mapping.py` — sibling to `get_gemm_index_for_src_matrix`, returns a ready-to-use BYTE
offset (not an element-unit row/col left for the caller to scale, since the k-half
contribution here is a `row_pitch_bytes`-scaled jump, not a small fixed 32-byte add). See
`igemm_bwd_gtc_wmma_nhwc.py`'s `shared_load_b_functor` for the concrete worked example,
including how the caller adds the two remaining compile-time-constant deltas
(`i_rn * wave_tile_n * elem_bytes` for the wave_repeat step, `(a*2+s) * row_pitch_bytes` for
which of the 16 rows within a k-half block).

**Validated on real gfx1250 hardware** (`igemm_bwd_gtc_wmma_nhwc_t`, single-K-block config
128x128x32): exact match against a CPU reference across single-block, multi-block (4x3 and
other grids), and multi-K-block (2 and 4 iterations) configurations, multiple random seeds,
and a config where GEMM_N (total C_in) differs from `gemm_n_per_block` (catches a stride
computed from the wrong tunable field, since bwd's `move_slice_window_b` needs a
**runtime** per-K-block stride — `s_gemm_n * databyte * gemm_k_per_block` — unlike fwd's
compile-time-constant one, because C_in isn't known until kernel launch).

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
