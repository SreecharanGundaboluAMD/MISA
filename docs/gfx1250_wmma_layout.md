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

**wrw (grad-weight)** extends this to BOTH operands (`igemm_wrw_gtc_wmma_nhwc_t`): grad_output
is naturally `[GEMM_K][GEMM_M]` (K_out-major from the M side's perspective, needs
`side='m'`), input is naturally `[GEMM_K][GEMM_N]` (needs `side='n'`, same tensor shape as
bwd's weight operand). This surfaced a real bug in
`get_gemm_index_for_src_matrix_transposed`: the wave-index extraction for `side='m'` was
reusing `side='n'`'s formula (`wave_id & (waves_per_side-1)`, a low-bit mask), but
`get_gemm_index_for_dst_matrix`'s established convention encodes `wave_id` as
`m_idx*waves_per_n + n_idx` (m in the *high* bits, n in the *low* bits) — so `side='m'` needs
`wave_id >> log2(waves_per_n)` (a shift), not a mask. Since bwd only ever exercises `side='n'`
(where the mask happens to be correct), this was invisible until wrw's `side='m'` A operand
exposed it: symptom was a clean 64x64 quadrant swap in the output (waves whose true `m_idx`
happened to coincide with the buggy mask-derived value were accidentally correct; the other
two waves computed using the wrong operand's 64-column M-half). Fixed by branching the wave
index computation on `side` instead of reusing one formula for both. Re-verified bwd (only
uses `side='n'`, unaffected) and fwd (doesn't use this function at all) show no regression.

**Validated on real gfx1250 hardware** (`igemm_wrw_gtc_wmma_nhwc_t`, single-K-block config
128x128x32): exact match across single-block, multi-block (including non-square
`GEMM_M != GEMM_N` grids, which catch an A/B tensor swap), and multi-K-block configurations
(including a config with `GEMM_M != GEMM_N != gemm_*_per_block`, which distinguishes the two
independent per-K-block strides `s_a_k_stride`/`s_b_k_stride` from each other), multiple
random seeds.

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

## Driver-side WMMA integration (`driver/conv_driver.cpp` and `igemm_{fwd,bwd,wrw}_gtc_driver.h`)

All three WMMA milestone kernels (fwd fp16/bf16/int8, bwd fp16, wrw fp16) are now dispatchable
through the real `conv_driver.exe`, not just the standalone `/tmp/wmma_probe/` harnesses used
during Phases 1-4. Each direction's driver class gained: a `fma_type == WMMA` branch in
`get_block_size()` (wave32, no `wave_step`), an early WMMA branch in `tunable_is_valid()`
(degenerate-case-only checks, bypassing the XDLOPS/DLOPS-oriented nhwc checks entirely), a
minimal karg struct (`igemm_{fwd,bwd,wrw}_gtc_wmma_nhwc_karg_t`, 36 bytes for bwd/wrw, grown to
68 bytes for fwd by Phase 5a below), and a self-contained
`run()` branch that builds the karg and launches directly via `igemm_launch_kernels()` with a
genuine 2D `(grid_x, grid_y)` dispatch — unlike the general kernels, which flatten the block
index into one dimension and decode it via magic division inside the kernel, this kernel reads
`workgroup_id_x`/`workgroup_id_y` directly as the M-block/N-block indices (see the "Workgroup ID
delivery" section above), so `get_grid_size()` (which returns a single flattened count) is
reused for other purposes but NOT for the actual launch geometry.

Two non-obvious pitfalls hit during this integration, both fixed:

- **Output-buffer width.** The WMMA kernels always accumulate/store the D operand at full
  width (fp32, or int32 for int8) regardless of `tunable->precision` — a deliberate
  correctness-first simplification (see each kernel's docstring). But `conv_driver.cpp`'s
  generic `*_dtype` buffers (`device_input_dtype`/`device_weight_dtype`/`device_output_dtype`)
  are sized at `data_byte` (the tunable's precision width, e.g. 2 bytes for fp16) everywhere
  else in the file. Passing a 2-byte-per-element buffer to a kernel that writes 4 bytes/element
  is a real heap buffer overflow, not just a wrong-answer bug. Fixed by computing
  `is_wmma = tunables[0].fma_type == WMMA` once in `main()` and widening the three `*_dtype`
  allocations (host and device) to `sizeof(float)` when `is_wmma` — over-allocating is harmless
  for whichever buffers stay native-precision-width in practice — and by branching the
  memset/memcpy-back/comparison call for whichever buffer plays the "kernel-native-width
  output" role for that direction (output for fwd, input for bwd, weight for wrw) to use the
  wide size and (for int8) an `int32_t` comparison instead of `int8_t`.
- **p_in/p_wei/p_out role mapping differs per direction.** `run(arg, tunable, p_in, p_wei,
  p_out, gks)`'s parameter names follow the *general* NHWC kernels' convention, where `p_in`
  always refers to the GPU buffer conventionally called "input" in conv terminology (the
  computed/written tensor for bwd — grad_input) and `p_out` to the "output" buffer (the
  read-only tensor for bwd — grad_output). The Python WMMA kernels' own kernarg field names
  (`p_in`/`p_wei`/`p_out`) instead describe each operand's *role within that specific kernel*
  (A operand/B operand/output), which for bwd is the OPPOSITE of run()'s convention
  (`karg.p_in`=grad_output read, `karg.p_out`=grad_input write) and for wrw is a 3-way
  ROTATION (`karg.p_in`=grad_output read ← run()'s `p_out`; `karg.p_wei`=input read ← run()'s
  `p_in`; `karg.p_out`=grad_weight write ← run()'s `p_wei`). Getting this backwards produces a
  kernel that runs without error but silently reads/writes the wrong buffers — caught by
  cross-checking pointer values (not just contents) between `main()` and `run()` when hardware
  results didn't match the reference.

**Validated on real gfx1250 hardware** through `conv_driver.exe` itself (not just the
standalone harnesses): fwd (fp16/bf16/int8), bwd (fp16), and wrw (fp16), each across
single-block and multi-block/multi-K-block configurations, using the driver's own
`naive_conv_*_nhwc`/`gpu_naive_conv_*_nhwc_fp32` CPU/GPU reference and `valid_vector<T>`
comparison — no bespoke verification code, unlike Phases 1-4.

## Phase 5a: general stride/padding for fwd (still 1x1 filter, dilation=1, group=1)

`igemm_fwd_gtc_wmma_nhwc_t` was extended in place (not a new sibling class) to support
arbitrary `stride_h/w` and `pad_h/w` — the first departure from the pure-degenerate-GEMM
addressing every earlier phase used. Mechanism (full detail in the class's docstring and
`emit_kernel_prologue`):

- Kernarg grew from 36 to 68 bytes: added `hi, wi, stride_h, stride_w, pad_h, pad_w, wo,
  ho_wo` (`ho_wo` = `ho*wo`, host-precomputed — the kernel never needs `ho` on its own).
- Each thread's merged GEMM_M index is decomposed back into `(n_idx, ho_idx, wo_idx)` via
  two chained runtime `u32` divisions (by `ho_wo` then `wo`), **not** magic division —
  correctness-first, per the original plan's own guidance, since `ho`/`wo` are only known at
  launch time. `hi_idx = ho_idx*stride_h - pad_h` / `wi_idx` similarly (signed subtract, so a
  negative result is a valid two's-complement bit pattern, not clamped).
- Out-of-bounds (padding) detection reuses `igemm_fwd_gtc_nhwc.py`'s established unsigned-
  wraparound trick (`v_cmp_gt_u32` treats a negative signed index, reinterpreted as u32, as
  huge, so one comparison rejects "negative" and "too large" together) — but instead of that
  file's post-load `v_cndmask_b32` select, padding is enforced via **EXEC masking**
  (`v_cmpx_le_u32`/`s_mov_b32 exec_lo, -1`) around the global load itself, since for a 1x1
  filter a thread's padding-vs-real status is a compile-time-fixed-per-thread constant for
  the whole kernel (never depends on which K-chunk is loading) — computed once in the
  prologue, `v_gld_a` pre-zeroed once, and a padding lane's global load simply never executes
  again for the kernel's lifetime, no per-iteration recheck needed. Weight (B) and the output
  write are completely untouched — a 1x1 filter has no spatial extent on the weight side, and
  every GEMM_M index is a real (non-padding) *output* pixel regardless of input padding.

**Real bug found**: the codebase's existing plain-division macro (`macro_int_div_rem_vs_t` /
`macro_int_div_vs_t` in `python/operations/utility.py`) uses bare `vcc` and 64-bit SGPR-pair
compare destinations (`s[\s_tmp4:\s_tmp4+1]`) — both **fail to assemble on gfx1250** (confirmed
via `llvm-mc`: "operands are not valid for this GPU or mode" / "invalid operand for
instruction"), since gfx1250 is wave32 and needs `vcc_lo` plus single-SGPR (32-bit) condition
masks. This wasn't caught earlier because no prior WMMA kernel needed runtime division. Fixed
by adding **new**, wave32-specific sibling classes (`macro_int_div_vs_gfx1250_t`,
`macro_int_div_rem_vs_gfx1250_t`) rather than modifying the existing ones in place — zero risk
to other architectures' codegen, which keep using the original wave64-oriented classes
unchanged. The new division routine was independently verified on real gfx1250 hardware across
24 divisors × 32 numerators each (768 cases total, including 0, exact multiples, and
near-`UINT32_MAX` numerators) via a standalone probe (`/tmp/wmma_probe/probe_div.s`,
`host_div.cpp`) *before* being trusted inside the real kernel — same discipline as every other
new primitive in this project.

**Validated on real gfx1250 hardware** through `conv_driver.exe`: fp16/bf16/int8 fwd, each
across a stride-1/pad-0 regression case (unchanged behavior) plus multiple genuine stride
(1/2/3) × padding (0/1/2) combinations, multi-block grids, asymmetric stride_h≠stride_w, and
multi-K-block configs — all compared against the driver's own `naive_conv_fwd_nhwc` reference,
which already implements general stride/padding/dilation correctly (no separate reference
needed for this phase, unlike the WMMA-layout probes of Phases 1-4).

**Not yet done** (deliberately deferred, scoped as separate future steps): multi-tap filters
(`y,x > 1`, needs a real K-loop-coupled tap iteration, unlike 1x1's single-input-pixel-per-
output-pixel gather), dilation, `group > 1`. Stride/pad for bwd and wrw are covered next.

## Phase 5b: general stride/padding for bwd (still 1x1 filter, dilation=1, group=1)

Same idea as Phase 5a, but with a genuinely harder validity condition. fwd's GEMM_M (n*ho*wo)
enumerates *output* pixels, so its gather is a simple "is this input coordinate in bounds"
check. bwd's GEMM_M (n*hi*wi) enumerates *input* pixels — grad_input's own size — while the A
operand it reads (grad_output) is stored in grad_output's *different* pixel space (n*ho*wo),
which is smaller than GEMM_M whenever `stride > 1`. Concretely: `ho_idx = (hi_idx+pad_h) /
stride_h` is only meaningful when that division is *exact* — an input pixel that doesn't land
on a stride multiple has no corresponding output pixel at all (a "stride gap"), independent of
whether `ho_idx` would otherwise be in bounds. So each thread now runs 4 chained divisions
(not fwd's 2): decompose GEMM_M into `(n_idx, hi_idx, wi_idx)` via `hi*wi` then `wi` (same
shape as fwd's decomposition, just using the *input*'s own hi/wi as divisors instead of the
*output*'s ho/wo), then `(hi_idx+pad_h)/stride_h` and `(wi_idx+pad_w)/stride_w`, checking both
the remainder (must be exactly 0) and the quotient (must be `< ho`/`< wo`) — four conditions
ANDed into one flag, same EXEC-masking mechanism as Phase 5a. Weight (B) and the grad_input
output write are untouched, exactly as in fwd's case.

Kernarg grew from 36 to 68 bytes (`hi_wi, wi, stride_h, stride_w, pad_h, pad_w, ho, wo`) — note
`hi`/`wi` are needed as *divisors* here (unlike fwd, where they were *bounds*), while `ho`/`wo`
are the *bounds* (the reverse of fwd's role assignment, since bwd's roles are already generally
inverted relative to fwd — see the p_in/p_wei/p_out mapping note in the driver-integration
section above).

**Validated on real gfx1250 hardware** through `conv_driver.exe`: fp16 bwd, stride-1/pad-0
regression plus stride (1/2/3) × padding (0/1/2), multi-block, asymmetric stride_h≠stride_w,
and multi-K-block configs — all passed on the first hardware run (no bugs found this time,
unlike wrw's `side='m'` bug in Phase 4) — against the driver's own `naive_conv_bwd_nhwc`
reference.

## Phase 5c: general stride/padding for wrw (still 1x1 filter, dilation=1, group=1)

wrw's GEMM_K is `N·Ho·Wo` — batch×output-spatial, not a channel count — so unlike fwd/bwd
(where the gather-vs-bounds decision is made once per thread in the prologue and never
revisited), wrw's two operands sit on *opposite sides* of the K-loop and need opposite
treatment:

- **A (grad_output)** needs **no change at all**. GEMM_K is *defined* as grad_output's own
  pixel space (`n*ho*wo`), so every K-index the main loop walks through is, by construction, a
  real grad_output pixel — there is no padding/stride gather to do on this operand, in any
  direction.
- **B (input)** needs fwd's simpler multiply-based gather (`hi_idx = ho_idx*stride_h - pad_h`,
  2 bounds conditions, not bwd's 4-condition division-based one — the K-index here indexes
  *output* pixels, same role fwd's GEMM_M played), but it must be **recomputed every main-loop
  iteration**, not once in the prologue: each iteration consumes a different 32-row slice of
  K-space (`k_block_off = s_knum - s_kitr`, evaluated right after `wmma_main_loop.py`'s
  per-iteration `s_kitr -= unroll_k`, giving the K already consumed = the offset of the block
  about to be loaded), so the `(n_idx, ho_idx, wo_idx)` decomposition and resulting
  gather/flag are entirely different values each time.

Implementation consequences of the per-iteration gather:
- `global_load_b_functor` now **re-zeroes all 16 `v_gld_b` registers on every call**, not just
  once — a lane that was valid (in-bounds) on a previous iteration but is invalid on this one
  must not silently keep stale non-zero data from the earlier load; zero-then-maybe-load is the
  only way to guarantee that.
- The row/column split that used to be computed once (`row_local = tid>>2`, `col_group =
  tid&3`) is now split: `v_row_local` and `v_b_col_off` (the fixed byte offset from
  `block_n_off + col_start`) are still computed once (they don't depend on which K-block is
  loaded), but the row's *absolute* K-position (`k_abs = k_block_off + row_local`) and
  everything downstream of it (division, gather, flag, final address) live in a new shared
  method, `_emit_b_gather(s_k_block_off)`, called once from the prologue (`k_block_off=None`,
  i.e. 0) and once per iteration from `move_slice_window_b_functor`.
- `s_b_k_stride` (Phase 4's constant per-iteration address bump for B) is gone — B's address is
  fully recomputed from scratch each iteration instead of incrementally bumped, since the
  gather makes a constant stride meaningless.

Kernarg grew from 36 to 68 bytes (`ho_wo, wo, stride_h, stride_w, pad_h, pad_w, hi, wi`) — same
field *values* as fwd's Phase 5a (wrw's B/input gather is structurally fwd's A/input gather,
just driven by a runtime-varying K-offset instead of a compile-time-fixed M-offset).

**Real bug found (driver-side, not kernel-side)**: `driver/igemm_wrw_gtc_driver.h`'s
`tunable_is_valid()` has a long-standing generic guard, `fil_h_ext = y*dilation_h+1-y` (which
is exactly `1` for any 1x1 filter) combined with `if (pad_w >= fil_w_ext || pad_h >=
fil_h_ext) return false;` — this unconditionally rejected *every* padded 1x1-filter config,
WMMA or not, since `pad_h >= 1` always trips `1 >= 1`. The WMMA-specific validity branch was
being evaluated *after* this check, so it never got a chance to run for any padded case. Fixed
by relocating the WMMA branch to run immediately after the `need_wrw` check, before the
`fil_*_ext` guard — using `tunable->gemm_m_per_block` etc. directly (the function's local
aliases for those fields aren't declared yet at that point).

**Validated on real gfx1250 hardware** through `conv_driver.exe`: fp16 wrw, stride-1/pad-0
regression, stride=2/pad=1 (a 4-K-block case, `n=1,ho=16,wo=8 → gemm_k=128`, already exercising
the per-iteration gather across multiple main-loop iterations on the very first stride/pad
test), stride=3/pad=2, asymmetric stride_h≠stride_w, multi-block M/N, and a larger multi-K-block
case (`n=4` batch) — all passed, all against the driver's own `naive_conv_wrw_nhwc` reference.
No numerical bugs found in the per-iteration re-gather/re-zero mechanism itself; the only issue
was the pre-existing driver-side validity guard above.

With Phase 5c complete, all three directions (fwd/bwd/wrw) now support arbitrary stride and
padding for 1x1 filters, `group=1`, `dilation=1`, on fp16 (bf16/int8 stride/pad extension is
straightforward parametrization of the same mechanism, not attempted yet — see Phase 5d below).

**Not yet done**: multi-tap filters (`y,x > 1`), dilation, `group > 1`, and replicating
stride/pad support to bf16/int8 for bwd/wrw (only fwd has all three precisions with general
stride/pad so far). See Phase 5d below for multi-tap/dilation on fwd.

## Phase 5d: multi-tap filters (y,x >= 1) and dilation for fwd

`igemm_fwd_gtc_wmma_nhwc_t` was extended again (still 1x1-filter-only class from Phase 5a's
perspective, now generalized) to support arbitrary filter height/width and dilation, on top
of Phase 5a's stride/pad (`group=1` still required):

```
output[n,ho,wo,k] = sum_{iy,ix,c} input[n, ho*stride_h-pad_h+iy*dilation_h,
                                          wo*stride_w-pad_w+ix*dilation_w, c] * weight[k,iy,ix,c]
```

Weight's standard conv layout is `[K_out][Y][X][C_in]` (`C_in` innermost — confirmed against
`driver/naive_conv.h`'s `f_idx = k*fy*fx*c + iy*fx*c + ix*c + ic`), so its `K_out`-stride is
`y*x*c` elements (not just `c` as in the 1x1 case), and a given tap's `C_in`-column-block
starts `(iy*x+ix)*c` elements into that row.

**Design choice**: rather than porting `igemm_fwd_gtc_nhwc.py`'s `merge_e` machinery (a single
merged `GEMM_K = c*y*x` with carry-propagating address deltas for when the c-dimension
overflows into x, and x into y — see that file's `move_slice_window_accumulate` and its dozen
`s_diff_in_*` SGPRs), this kernel keeps `GEMM_K = c` unchanged from Phase 5a and wraps the
*entire* WMMA K-main-loop in a new, small, **runtime** (not compile-time-unrolled) outer loop
over the `y*x` taps:

```
s_iy = 0
L_tap_y: s_ix = 0
  L_tap_x: <recompute this tap's A address+flag, B address>
           <issue first A/B loads>
           <run the WMMA K-main-loop over C -- emitted EXACTLY ONCE; the runtime
            branch below re-enters this same code for every tap>
           s_ix++; branch to L_tap_x if s_ix < s_x
  s_iy++; branch to L_tap_y if s_iy < s_y
```

`v_c` (the accumulator) is zeroed once at kernel entry and never reset between taps, so
successive taps' `v_wmma_* D=A@B+C` calls accumulate naturally — the only thing that changes
per tap is which A/B addresses feed the same, unchanged K-loop body. All four waves in a block
execute this loop in lockstep (`s_iy`/`s_ix` and the `y`/`x` bounds are wave-uniform SGPR
values, not thread-varying), so the existing single-buffered-LDS barrier scheme is unaffected.
Emitting the K-main-loop's assembly only once (rather than once per tap) keeps code size
independent of filter size, at the cost of a few extra scalar instructions and one branch per
tap — a correctness-first tradeoff consistent with every other phase in this milestone.

Per-tap addressing: A's (input) `hi_idx`/`wi_idx`/flag/address are fully recomputed every tap
from this thread's *persistent* `n_idx`/`ho_idx`/`wo_idx` (decomposed from GEMM_M once, in the
prologue, via the same runtime-division sequence Phase 5a introduced) plus the current
`s_iy`/`s_ix`. Padding is masked the same way Phase 5a did (unsigned-wraparound bounds check +
EXEC-masked global load), but the flag is no longer a per-thread constant for the whole kernel
(it can flip between taps, since different taps look at different input pixels) —
`global_load_a_functor` therefore re-zeros all 16 `v_gld_a` registers on **every** call, not
just once (the same discipline Phase 5c/wrw established for its own per-iteration-varying
gather), so a lane valid on one tap can never leak stale data into a tap where it's invalid.
B's (weight) address has no padding concept — a tap index is always a real weight row as long
as `0<=iy<y, 0<=ix<x`, which the loop bounds guarantee — so it's just a different fixed byte
offset added to a per-thread base address (`v_addr_b_base`, computed once from the corrected
`y*x*c` row stride) each tap.

**Bug caught before hardware** (self-review, not a test failure): an early draft precomputed
`v_addr_a`'s high 32 bits once in the prologue (`v_addr_a(1) = p_in_hi`) on the theory that a
pointer's high half rarely changes. That's wrong here specifically because every tap computes
a **fresh, independent** address rather than incrementally bumping the previous tap's address
— reusing `v_addr_a(1)`'s value from the *previous* tap as an input to `v_add_co_ci_u32` would
silently accumulate carries across taps, drifting the high half upward. Fixed by resetting
`v_addr_a(1) = s_p_in(1)` fresh inside the per-tap gather, before every tap's carry-add pair
(B's `v_addr_b_base(1)` was already correct, since it's a genuine loop-invariant base that
per-tap code only ever reads from, never accumulates into).

Kernarg grew from 68 to 84 bytes (`y, x, dilation_h, dilation_w`). `driver/igemm_fwd_gtc_driver.h`'s
`tunable_is_valid()` dropped its `unit_conv_1x1` requirement entirely (now only `group==1` and
`tensor_layout=="nhwc"` are checked before the tile-multiple checks) — no other driver-side
mapping changes were needed since `y`/`x`/`dilation_h`/`dilation_w` were already being read
out of `arg` for logging/dumpheader purposes, just not yet wired into the WMMA karg.

**Validated on real gfx1250 hardware** through `conv_driver.exe`, fp16/bf16/int8, against the
driver's own `naive_conv_fwd_nhwc` reference: 1x1 stride-1/pad-0 regression (bit-for-bit same
kernel path, now going through one tap-loop iteration), 3x3 with and without padding, 3x3 with
dilation=2, 3x3 combined with stride=2, 5x5 with dilation=2, non-square 3x5 filters with
independent per-axis stride/pad/dilation, 7x7, multi-block M/N grids, and multi-K-block-per-tap
(large `C_in`) — all exact matches, largest case: n=2, C=256 (multiple K-blocks per tap), 5x5
filter, stride=2, pad=2, multi-block grid.

**Still not done**: `group > 1` (all directions) — see Phase 5f below for wrw's multi-tap/
dilation extension, which completes multi-tap+dilation support across all three directions.

## Phase 5e: multi-tap filters (y,x >= 1) and dilation for bwd

Mirrors fwd's Phase 5d exactly in structure (same runtime, not compile-time-unrolled, outer
loop over the y*x taps wrapping a single static WMMA K-main-loop emission), applied to bwd's A
operand (grad_output) and B operand (weight) instead of fwd's A/B:

```
grad_input[n,hi,wi,c] = sum_{iy,ix,k} grad_output[n, (hi+pad_h-iy*dilation_h)/stride_h,
                                                    (wi+pad_w-ix*dilation_w)/stride_w, k]
                                     * weight[k,iy,ix,c]
```

**A operand (grad_output)**: needs the harder per-tap gather. This thread's `(n_idx, hi_idx,
wi_idx)` are decomposed from GEMM_M exactly once (persistent VGPRs), then every tap recomputes
`numerator_h = hi_idx + pad_h - iy*dilation_h` — note the **subtract**, the opposite sign from
fwd's per-tap `+iy*dilation_h`, since bwd's formula is fwd's algebraically inverted relation
(confirmed against `driver/naive_conv.h`'s `cur_oh = ih + py - dy*ir` before dividing by `sy`).
A negative numerator wraps to a huge `u32`; the subsequent division's bounds check
(`ho_idx < s_ho`) naturally rejects it via quotient overflow, so no separate sign check is
needed — the same "unsigned wraparound rejects invalid" trick used everywhere else in this
milestone, just one division removed from the direct comparison. Since the flag can now flip
between taps, `global_load_a_functor` re-zeros all 16 `v_gld_a` registers on every call.

**B operand (weight)**: same treatment as fwd's Phase 5d — weight's `[K_out][Y][X][C_in]`
layout means the per-K_out-row element count becomes `y*x*c` instead of plain `c`. Since bwd
addresses weight via `row_local`/`col_group` (not a decomposed GEMM index like fwd), the new
`s_wei_row_c = gemm_n*x*y` scalar replaces the old plain `gemm_n` (=C_in) everywhere it was
used as a row multiplier: both the fixed `row_local*wei_row_c` base address AND
`move_slice_window_b`'s per-K-iteration stride (`s_wei_k_stride = wei_row_c*databyte*32`,
unchanged formula shape, just fed the corrected row-count). A given tap's C_in-column-block
still starts `(iy*x+ix)*c` elements into that row — a fixed byte offset added to a per-thread
base (`v_addr_b_base`) fresh every tap, identical idiom to fwd.

Weight's transposed LDS read and the grad_input output write are **unaffected** by taps — they
only ever see LDS-local byte offsets within whichever slice of the weight tile is currently
resident, populated correctly regardless of which absolute tap the global address pointed to.

Kernarg grew from 68 to 84 bytes (`y, x, dilation_h, dilation_w`, identical field additions to
fwd's Phase 5d). Driver's `tunable_is_valid()` dropped its `unit_conv_1x1` requirement.

**Validated on real gfx1250 hardware** through `conv_driver.exe` (fp16) against
`naive_conv_bwd_nhwc`: 1x1 stride-1/pad-0 regression, 3x3 with padding, 3x3 with dilation=2,
3x3 combined with stride=2 (exercising the stride-gap divide together with the per-tap
dilation subtraction for the first time), non-square 3x5 with independent per-axis
stride/pad, and a multi-block/multi-K-block 5x5 stride=2 case (C=256, spanning multiple
K-blocks per tap) — all exact matches, all passing on the first hardware run.

## Phase 5f: multi-tap filters (y,x >= 1) and dilation for wrw

**wrw's tap loop is structurally different from fwd/bwd's**, and this is the key thing to
remember before touching this code again: in fwd/bwd, every tap contributes to the SAME output
pixel (all y*x taps are summed into one accumulator, stored once at the very end). In wrw,
each tap produces a DIFFERENT, INDEPENDENT slice of the output tensor — confirmed against
`driver/naive_conv.h`'s `naive_conv_wrw_nhwc`: `filter_grad[...,ir,is,...] = value` is computed
FRESH per `(ir,is)` tap, summed only over the `(n,ho,wo)` reduction dimension (GEMM_K), never
accumulated across taps. So here, `v_c` is zeroed and the epilogue (`coalescing_store`) fires
**once per tap**, not once at kernel end:

```
s_iy = 0
L_tap_y: s_ix = 0
  L_tap_x: zero v_c
           reset A's address to its tap-independent base (v_addr_a_base)
           <B's initial per-tap gather + issue first A/B loads>
           <run the WMMA K-main-loop over N*Ho*Wo -- emitted EXACTLY ONCE>
           <store v_c to THIS TAP's own [iy,ix] output slice, s_wait_storecnt>
           s_ix++; branch to L_tap_x if s_ix < s_x
  s_iy++; branch to L_tap_y if s_iy < s_y
```

**Tensor A (grad_output)** needs no tap-dependence at all (same as Phase 5c): grad_output has
no Y,X extent in its own storage. Only its *address* needs resetting to a persistent
`v_addr_a_base` at the start of every tap, since `move_slice_window_a` incrementally bumps
`v_addr_a` across a tap's own K-loop and it would otherwise carry over into the next tap.

**Tensor B (input)** extends its existing per-iteration gather (`_emit_b_gather`, structurally
unchanged from Phase 5c) with a per-tap bias: `hi_idx = ho_idx*stride_h - pad_h +
iy*dilation_h` (same ADD sign as fwd's Phase 5d, since this is an output-index->input-index
gather via multiplication, not bwd's harder divide-based one). Since `_emit_b_gather` reads the
CURRENT `s_iy`/`s_ix` live every time it's invoked — once per tap from the new
`emit_kernel_tap_loop`, and once per K-iteration from `move_slice_window_b` — no special-casing
was needed beyond adding the bias terms; both existing call sites automatically pick up
whichever tap is active.

**Output (grad_weight)** is `[K_out][Y][X][C_in]` for a multi-tap filter (was plain
`[K_out][C_in]`), so its row stride becomes `y*x*c` (`s_wei_row_c = gemm_n*x*y`, the same
scalar fwd's Phase 5d / bwd's Phase 5e introduce for the *weight* tensor, reused here for
grad_weight since they share the on-disk layout convention). Each tap's C_in-column-block
starts `(iy*x+ix)*c` elements into that row — expressed as a byte offset added to a **fresh
per-tap base pointer** `s_p_out_tap` (not a per-thread VGPR offset), since
`coalescing_store_wmma.py`'s `s_p_out` argument is the literal store base address shared by
every thread in the block; adding the tap's constant offset to the base pointer once per tap
needed zero changes to that shared, direction-agnostic epilogue helper. `s_p_out_tap` needed
explicit 2-alignment (`sseq(2, 2)`) since it's used as a VADDR base by `global_store_dword` —
caught immediately by `llvm-mc`'s "invalid register alignment" on the first build attempt.

Kernarg grew from 68 to 84 bytes (`y, x, dilation_h, dilation_w`, identical field additions to
fwd/bwd). Driver's `tunable_is_valid()` dropped its `unit_conv_1x1` requirement (still checked
*before* the pre-existing `fil_h_ext`/pad guard, per Phase 5c's fix).

**Validated on real gfx1250 hardware** through `conv_driver.exe` (fp16) against
`naive_conv_wrw_nhwc`: 1x1 stride-1/pad-0 regression, 3x3 with padding, 3x3 with dilation=2,
3x3 combined with stride=2, non-square 3x5 with independent per-axis stride/pad, and a
multi-block/multi-K-block 5x5 stride=2 case (`n=4,c=256`, 25 taps each running their own
K-loop reduction and independent output store) — all exact matches, all passing on the first
hardware run.

With Phase 5f complete, all three directions (fwd/bwd/wrw) now support arbitrary multi-tap
filters, dilation, and stride/padding for fp16, `group=1`.

## Phase 6: bf16 and int8 for bwd/wrw's general addressing

**bf16 was already free**: `igemm_bwd_gtc_wmma_nhwc_t` and `igemm_wrw_gtc_wmma_nhwc_t` already
asserted `precision in ('fp16', 'bf16')` and use `data_byte` generically throughout (same
2-byte-element layout as fp16, `num_v_a=num_v_b=8` for both instructions) -- validated on
hardware with zero code changes, just new `.config` files.

**int8 required real fixes** in three layers, found by working through bwd first (fp16's
`gemm_k_per_block=32` vs int8's `=64` breaks several hardcoded assumptions that happened to
never matter until now):

1. **`gemm_k_per_block`-hardcoded literals**: `wmma_mapping.py`'s
   `get_gemm_index_for_src_matrix_transposed` hardcoded the "k_half" row-jump as `16` (half of
   32) -- fixed to `ctrl.inst_wmma.k // 2` (32 for int8). Both bwd's `s_wei_k_stride` and wrw's
   `s_a_k_stride` hardcoded their per-K-block stride shift as `+5` (i.e. `*32`) -- fixed to
   `log2(data_byte * gemm_k_per_block)`. Both bwd's B operand and wrw's A/B operands hardcoded
   their thread-to-LDS-tile `row_local=tid>>2`/`col_group=tid&3`/`col_start=col_group*32`
   partition -- this is only correct when `gemm_k_per_block==32`; generalized to
   `num_col_groups = 128/gemm_k_per_block`, `row_local=tid>>log2(num_col_groups)`,
   `col_start=col_group*gemm_k_per_block` (a clean identity: the per-thread global-load is
   always a fixed 64 bytes, and `gemm_k_per_block*data_byte==64` always, so the column-chunk
   width in *elements* always equals `gemm_k_per_block`, for every precision).
2. **Transpose-read packing width**: the shared LDS-transpose-read pack logic
   (`shared_load_b_functor` in bwd, `shared_load_a_functor`/`shared_load_b_functor` in wrw) was
   hardcoded to 2-way 16-bit packing (`ds_read_u16` x2 + one `<<16` OR). Generalized to
   `elem_per_dword = 4/data_byte` (2 for fp16/bf16, 4 for int8), `ds_read_u16`-or-`ds_read_u8`,
   and a chained `v_lshl_or_b32` pack with `shift = s*8*data_byte`. Naively batching all 8
   vgpr-indices' reads before packing (the original fp16-only structure) would need 32 scratch
   VGPRs for int8's 4-way packing -- exceeded the wave's VGPR budget (confirmed by `llvm-mc`
   "register index out of range" on the first build attempt) -- restructured to process one
   vgpr-index at a time (read `elem_per_dword` sub-elements, wait, pack, move on), trading some
   latency-hiding for staying within the existing 16-register scratch budget.
3. **A real, previously-latent signedness bug in the int8 WMMA instruction itself**: the base
   `v_wmma_i32_16x16x64_iu8` encoding (no modifier) defaults to **unsigned** interpretation of
   both A and B (confirmed via `llvm-mc`), but conv's int8 tensors are `int8_t` (signed)
   throughout the driver. fwd's original int8 validation only ever used constant value `1` for
   every element (`igemm_rand_int`'s weak-data branch) -- positive, so signed vs unsigned never
   differed, and the bug was invisible. bwd's driver harness uses genuinely random `-5..5` data,
   which immediately exposed it (`valid:n` with wildly wrong values). Fixed by adding the
   `neg_lo:[1,1,0]` modifier (confirmed via `llvm-mc` to assemble and, empirically, to produce
   correct signed results against the driver's own reference) to the instruction table entry in
   `wmma.py` -- this is a **retroactive fix to already-shipped fwd int8 support**, not just a
   bwd/wrw addition; re-verified fwd's existing tests still pass with the modifier applied.

Also found and fixed a **real driver gap** (not a kernel bug): `conv_driver.cpp`'s wrw
data-generation block only had `tensor_copy<T,float>` branches for `driverHalf`/`driverBFloat16`
-- there was no `driverInt8` branch at all, so `host_input_dtype`/`host_output_dtype` were never
populated for int8, and the kernel computed on stale/uninitialized device memory. This produced
a `valid:n` that looked exactly like a kernel bug and cost significant debugging time before the
missing branch was spotted; the kernel-side fixes above were all independently correct and
already passing once real data was actually being fed to them. Also added the matching
`is_wmma`+`int8_t` branch to `wrw_post`'s output comparison (mirroring fwd/bwd's, which already
had it) -- previously wrw's output comparison unconditionally read the int32 D-operand as
`float`, producing `-nan`.

**Validated on real gfx1250 hardware** through `conv_driver.exe`, both bf16 and int8, for both
bwd and wrw, against `naive_conv_{bwd,wrw}_nhwc`: 1x1 regression, combined stride+multi-tap
(3x3, stride=2, pad=1), and multi-block/multi-K-block cases -- all exact matches. A full
9-way (3 directions x 3 precisions) regression sweep at the simplest 1x1 configuration also
passes.

## Phase 7: group>1 (grouped convolution), all three directions

**Grid encoding**: group is folded into `grid_y` (`grid_y = group *
ceil(gemm_n/gemm_n_per_block)`) rather than a genuine 3rd grid dimension, since no gfx1250
kernel anywhere has a working `workgroup_id_z` delivery mechanism (only `ttmp9`/`ttmp7` for
x/y are verified). The kernel decodes `group_idx` back out of `s_by` on-device:
`blocks_per_group_n = ceil(gemm_n/128)` needs no division (`gemm_n_per_block=128` is a
compile-time power of 2, just `(gemm_n+127)>>7`), but the `group_idx`/corrected-`s_by` split
does need a genuine runtime division (`blocks_per_group_n` is only known at launch time) --
this reuses the existing, already-hardware-verified `macro_int_div_rem_vs_gfx1250_t` via a
broadcast-to-vgpr (`v_mov_b32`) + `v_readfirstlane_b32` round-trip, rather than introducing a
new scalar-scalar division macro variant. `s_by` is overwritten in place with the corrected
within-group N-block index, so `s_block_n_off` (computed right after) needed no changes.
`gemm_m`/`gemm_n`/`gemm_k` were already per-group values in every kernel (the driver divides
`c`/`k` by `group` before writing the karg, unchanged since Phase 5 prep) -- the only new
kernarg field needed across all three kernels is `group` itself.

**The real bug, caught by hardware validation on the very first attempt (fwd)**: group 0's
per-operand address offset is always zero, so a kernel with a genuine bug in its group-offset
math still produces CORRECT output for group 0 -- masking the bug until group 1+ is checked.
The bug: NHWC tensors store their group split INTERLEAVED within each pixel's channel
dimension (`[N,H,W,G*C_per_group]`), so the per-pixel/row memory stride between consecutive
spatial positions must be the tensor's TOTAL channel count (`gemm_k*group`, `gemm_n*group`,
etc, depending on which GEMM dimension that operand's channel axis plays), NOT the per-group
value already used for the K-reduction size and the base-pointer offset -- an early draft
reused the per-group value for the row stride too (correct only when `group=1`, where
total==per-group), producing wrong, uncorrelated-looking output for every group past the
first. Diagnosed by comparing the same test with `group=1` (grid_y>1 alone, no group, passed
cleanly) against `group=2` (failed) to isolate that the bug was group-specific, then bisecting
which pixel index the wrongness started at (exactly `k=gemm_n`, i.e. the group-1 boundary) via
`PRINT_EVERY_PIXEL=1`, and ruling out the division-macro/`v_readfirstlane_b32` round-trip as
the cause first (bypassing it with a hardcoded `group_idx=s_by` for the trivial
`blocks_per_group_n=1` case still failed identically) before finding the real row-stride issue.

Per direction, exactly one operand needs NO correction (the tensor whose group split is the
OUTERMOST/block-contiguous dimension, `[G][K_per_group][Y][X][C_per_group]` -- confirmed
against `driver/naive_conv.h` and the general XDLOPS kernel's own grouped-conv addressing);
the other operand(s) (whose group split is interleaved per-pixel) need both a base-pointer
offset (`group_idx * that operand's own per-group element count`) AND the total-channel-count
row-stride fix above:
- **fwd**: A (input) and output need the fix; B (weight) doesn't.
- **bwd**: A (grad_output) and output (grad_input) need the fix; B (weight) doesn't.
- **wrw**: A (grad_output) and B (input) BOTH need the fix (wrw is the one direction where
  both GEMM_M and GEMM_N are group-affected, since GEMM_K=N*Ho*Wo is spatial, not a channel);
  output (grad_weight) doesn't, but its group offset must be added directly to the persistent
  `s_p_out` (not the existing per-tap `s_p_out_tap`, which is recomputed fresh every tap FROM
  `s_p_out` -- see Phase 5f) so every tap's offset automatically inherits the group shift.
  A's `move_slice_window` per-K-block stride (`s_a_k_stride`) also needed the same
  total-vs-per-group correction, since it depends on the same row-width quantity.

**Validated on real gfx1250 hardware** through `conv_driver.exe`, all three directions x all
three precisions (fp16/bf16/int8), against `naive_conv_{fwd,bwd,wrw}_nhwc`: `group=1`
regression (bit-identical to pre-Phase-7 behavior), `group=2`, `group=4`, combined with
multi-tap+stride and multi-block M/N configs -- all exact matches. A full 9-way (3 directions
x 3 precisions) x 2 group-counts regression sweep also passes.

With Phase 7 complete, this milestone's ENTIRE originally-scoped surface (fwd/bwd/wrw x
fp16/bf16/int8 x general stride/pad/multi-tap/dilation x group>1) is now COMPLETE.

Remaining gaps: fp8/fp32 WMMA instruction layouts remain unverified (would need their own
hardware round-trip probes before use, per the same discipline that caught the int8 signedness
bug above -- do not assume any new instruction/precision is correct without independently
re-verifying against real random (not just small positive) test data).
