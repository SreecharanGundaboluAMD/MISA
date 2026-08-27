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

## `v_wmma_f32_16x16x4_f32` (M=16, N=16, K=4, fp32 in / fp32 accum, no packing)

**Verified** — a genuinely simpler, differently-shaped layout family from fp16/bf16/int8:
`num_v_a = num_v_b = 2` (not 8), and **no packing at all** (1 fp32/dword, unlike fp16's 2/dword
or int8's 4/dword):

- **A operand** (2 VGPRs/lane, 1 fp32/dword): for vgpr index `a` (0..1):
  `row = l % 16`, `k = (l / 16) * 2 + a`
- **B operand** (2 VGPRs/lane, same): for vgpr index `a` (0..1):
  `col = l % 16`, `k = (l / 16) * 2 + a`
- **D operand**: identical to every other precision (`row = (l/16)*8 + j, col = l % 16`,
  `num_v_c=8`, fp32 accumulator) — confirmed via the same round-trip, not assumed.

Confirmed via the same round-trip technique across 6 random-seed trials
(`/tmp/wmma_probe/probe_fp32.s`, `host_fp32.cpp`).

`gemm_k_per_block` is forced to 4 (matching `inst_wmma.k`) since `wmma_main_loop.py` requires
`unroll_k == inst_wmma.k` exactly (no k-sub-loop). This breaks the "`gemm_k_per_block *
data_byte` is always 64 bytes" coincidence every fp16/bf16/int8 literal above assumed (true for
32×2, 32×2, 64×1, but 4×4=16 for fp32) — see Phase 8 below for the resulting generalization
work, and the num_v_a/num_v_b=2 (not 8) footprint above for a second, independent invariant
break the LDS-read functors needed fixing for.

## Not yet re-verified for other instructions

`v_wmma_f32_16x16x32_f16`, `v_wmma_f32_16x16x32_bf16` (K=32), `v_wmma_i32_16x16x64_iu8` (K=64),
and `v_wmma_f32_16x16x4_f32` (K=4) are verified. Do **not** assume any of these formulas for
K=128 (`v_wmma_f32_16x16x128_fp8_fp8`, note its A/B footprint is 16 VGPRs/lane, not 8 or 2 — a
structurally different layout again, not just a wider `k` range) — the D-operand (output)
layout is expected to carry over (M=N=16, `num_v_c=8` for all of them), but the A/B operand
layout must be independently re-verified with the same round-trip probe technique before
trusting it for a new instruction.

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

## Phase 8: fp32 support, all three directions

Full parity for fp32 (`v_wmma_f32_16x16x4_f32`) with the fp16/bf16/int8 surface above: general
stride/pad, multi-tap+dilation, and group>1, across fwd/bwd/wrw. fp8 remains out of scope (see
"Remaining gaps" below).

**Why this needed real generalization work, not just a config change**: `gemm_k_per_block` must
equal `inst_wmma.k` exactly (4 for fp32, vs 32/64 for fp16/bf16/int8), which breaks the
"`gemm_k_per_block * data_byte == 64 bytes`" coincidence noted above. Every kernel file had
literals built on that coincidence (hardcoded shift-by-6/shift-by-5 address multipliers, a
literal `1024`/`64`, loop bounds `range(4)`/`range(16)`) — all three kernel files now derive
`self.bytes_per_row = gemm_k_per_block * data_byte`, `self.num_dwordx4 = bytes_per_row // 16`,
`self.num_dwords = bytes_per_row // 4` once in `__init__`, and every affected literal reads
from these instead. `wmma_tile_m/n`/`wmma_repeat_m/n`-based loop bounds (tile-shape constants,
unrelated to K or data_byte) were left untouched.

A second, independent invariant break: the LDS-read functors for TRANSPOSED operands (bwd's
weight/B, both of wrw's operands) hardcoded "8 VGPRs per wave_repeat step" and a binary
`ds_read_u16`/`ds_read_u8` choice, both assuming `num_v_a`/`num_v_b == 8` (true for
fp16/bf16/int8, false for fp32's `num_v_a = num_v_b = 2`, per the Phase 8 layout section
above). Fixed by deriving the loop bound/index from `inst_wmma.num_v_a`/`num_v_b` instead of a
hardcoded 8, and adding a third `read_instr` branch (`ds_read_b32`, a full-width read with no
zero-extension) for `data_byte == 4`. The UNTRANSPOSED LDS-read functors (fwd's A/B, bwd's A)
had the same 8-VGPR assumption but a simpler fix: a new `_emit_ds_read_chunked` helper reads
`num_v` dwords in the largest `ds_read_*` chunk available (b128/b64/b32), replacing the
hardcoded `2x ds_read_b128`.

**No driver-side changes were needed anywhere** — `driver/conv_driver.cpp`'s `driverFloat`
codepath (pre-existing, used for the general/non-WMMA fp32 kernels) already reads/writes
buffers at `data_byte == sizeof(float)` width, unlike fp16/bf16/int8 which needed the
`is_wmma`-gated buffer-widening fix from earlier phases.

**Validated on real gfx1250 hardware** through `conv_driver.exe`, all three directions, against
`naive_conv_{fwd,bwd,wrw}_nhwc`: 1x1 degenerate case, multi-K-block, multi-block M/N,
multi-tap+dilation, stride+pad, group=2, group=4, and combined stride+multi-tap+group=2 cases —
all exact matches. fp16/bf16/int8 regression-tested after the generalization edits (all three
kernel files), all still exact matches — confirming the byte-constant generalization didn't
disturb the precisions it was extracted from.

Remaining gaps: fp8 (`v_wmma_f32_16x16x128_fp8_fp8`) WMMA instruction layout remains unverified
(would need its own hardware round-trip probe before use, per the same discipline that caught
the int8 signedness bug above -- do not assume any new instruction/precision is correct without
independently re-verifying against real random (not just small positive) test data). Note its
A/B footprint is 16 VGPRs/lane (K=128), a third distinct layout family from both the 8-VGPR
fp16/bf16/int8 family and fp32's 2-VGPR/no-packing family above.

## Phase 9 (2026-08-24): k-sub-loop performance mechanism for the WMMA main loop

Motivated by a user question about whether MISA's gfx1250 WMMA solvers can be made
competitively fast: `wmma_main_loop.py` originally required `unroll_k == inst_wmma.k` exactly
(32 for fp16/bf16, 64 for int8, 4 for fp32), so **every single `v_wmma_*` issue was preceded by
a full global-load -> shared-store -> barrier -> shared-load round-trip**. `gemm_k_per_block`
can now be any power-of-2 *multiple* of `inst_wmma.k` -- `num_k_substeps =
gemm_k_per_block // inst_wmma.k` `v_wmma_*` issues now happen per LDS round-trip, amortizing
that synchronization cost. See `ctrl_wmma_main_loop_t`'s docstring (`wmma_main_loop.py`) for
the exact restructuring, and the `k_substep_stride_bytes_a/b` fields each kernel file computes
(different formula for untransposed vs transposed operands -- `inst_wmma.k * data_byte` vs
`inst_wmma.k * row_pitch`).

### Two correctness bugs found via hardware validation, not compile-time checks

1. **k_half stride derived from the wrong quantity.** The untransposed operand's (fwd's A/B,
   bwd's A) `k_half` LDS-read offset was computed as `bytes_per_row // 2` -- correct only when
   `bytes_per_row == inst_wmma.k * data_byte` (true for every existing single-substep config,
   false once `gemm_k_per_block` becomes a multiple of `inst_wmma.k`). Must be
   `inst_wmma.k * data_byte // 2` instead (`self.inst_wmma_k_bytes` in each kernel file). The
   TRANSPOSED read helper (`get_gemm_index_for_src_matrix_transposed` in `wmma_mapping.py`) was
   already correct -- it derives its own k_half stride from `ctrl.inst_wmma.k` directly, never
   from a caller-supplied byte-row-width.

2. **Cross-wave LDS-overwrite race + a VGPR-reuse data-clobbering bug**, both specific to
   `global_load_a/b_functor`/`shared_store_a/b_functor`'s staging buffer (`v_gld_a`/`v_gld_b`):
   - Growing `v_gld_a`/`v_gld_b` to hold a whole (now possibly multi-substep) row overflows the
     256-VGPR/wave hardware limit -- fp16/bf16/int8's existing 128x128 tile already sits at
     252/256 VGPRs. Fixed by keeping `v_gld_a`/`v_gld_b` sized to ONE `inst_wmma.k`-worth
     (`chunk_num_dwords`), chunking global_load/shared_store into `num_k_chunks` rounds that
     reuse the same small buffer.
   - An early attempt stored chunk 0 immediately after its own global-load-wait (to keep the
     original design's "issue load early, overlap with compute" latency hiding). This produced
     silent wrong-answer corruption starting at the 2nd-3rd within-workgroup K-block: nothing in
     this design synchronizes different WAVES within a workgroup between "read the current
     tile" and "overwrite it with the next tile", and the original single-substep design's
     safety margin (LDS-store always happens only after the WHOLE current-tile compute, late in
     the iteration) was violated by storing chunk 0 much earlier. Fixed by deferring ALL LDS
     stores (not just extra chunks) until after `emit_wmma_tile()`/`emit_extra_substeps()` have
     consumed the current tile -- see `_emit_sst_remaining_chunks` in each kernel file.
   - A THIRD, distinct bug surfaced only for TRANSPOSED operands (bwd's B, wrw's A+B):
     `shared_load_b_functor`'s read-and-pack technique reuses `v_gld_b` as scratch, and is
     called again (for `emit_extra_substeps()`'s substep>=1) in the window between when chunk
     0's global load is issued and when it's finally stored -- clobbering the staged data
     before it's ever written to LDS (a data-overwrite bug, not just a race: waiting longer for
     the load doesn't help, since the SCRATCH REUSE overwrites the register regardless). Fixed
     by giving operands that use this scratch technique (bwd's B; wrw's A+B) a fully-deferred
     `_emit_sst_all_chunks` path with NO early/overlapped chunk-0 load at all, while operands
     that read directly into `v_a`/`v_b` with no scratch (fwd's A+B; bwd's A) keep chunk 0's
     overlap-with-compute via `_emit_sst_remaining_chunks`.
   - All three bugs were caught by comparing against `naive_conv_{fwd,bwd,wrw}_nhwc` on real
     hardware -- none produced an assembler or compile-time error, only silent wrong answers.

Regression safety: every one of these changes was designed so that `num_k_chunks == 1` (every
existing single-substep config) produces **byte-identical** generated `.s` output to before --
verified by direct `.s` diff for fp16 in all three directions before any new config was even
written, plus a full hardware re-run of the existing fp16/bf16/int8/fp32 configs after.

### Validated on real hardware, all 12 (direction x precision) combos

New `_k2x` configs (`gemm_k_per_block` doubled for fp16/bf16/int8, 8x'd for fp32 to reach the
same ~32KB LDS budget from its much smaller starting K=4) pass the full battery -- 1x1
degenerate/multi-K-block, larger multi-K-block, stride+pad, multi-tap+dilation, group=2,
group=4 -- for all 12 combos, exact matches against `naive_conv_{fwd,bwd,wrw}_nhwc`.

### Benchmark results: a genuinely mixed picture, not a uniform win

tflops on a K-loop-bound problem (`c=k=2048`, 1x1, single/near-single tile-block), old
(single-substep) vs new (k2x) config, same problem size:

| direction/precision | old tflops | new tflops | delta |
|---|---|---|---|
| fwd/fp16  | 19.99 | 17.82 | **-11%** |
| fwd/int8  | 32.46 | 28.42 | **-12%** |
| fwd/fp32  |  7.33 |  6.25 | **-15%** |
| bwd/fp16  |  8.08 |  8.13 | +0.7% |
| bwd/int8  | 13.07 | 12.58 | -4% |
| bwd/fp32  |  2.88 |  3.57 | **+24%** |
| wrw/fp16  | 34.82 | 34.93 | +0.3% |
| wrw/int8  | 52.14 | 52.67 | +1% |
| wrw/fp32  | 13.68 | 17.38 | **+27%** |

**fwd consistently regresses**; bwd/wrw are roughly flat to significantly better, with fp32
the biggest winner. The reason: only chunk 0 of each outer iteration gets the "issue early,
overlap with compute" treatment (see the VGPR-clobbering bug above) -- chunks 1..N-1 are loaded
and stored strictly AFTER compute, fully exposed with no latency hiding at all. fwd's A/B
compute is cheap (a plain `_emit_ds_read_chunked` read straight into `v_a`/`v_b`, no
read-and-pack overhead), so the newly-exposed global-memory round-trip for the extra chunks
costs more than the barrier savings buy back. bwd/wrw's TRANSPOSED-operand compute is much more
expensive per substep (the read-and-pack technique is dozens of instructions per wave_repeat
step), so barrier savings dominate instead -- and fp32's win is largest because its
`inst_wmma.k=4` means the OLD design paid a barrier every 4 K-elements, so cutting substep count
by 8x (the biggest multiplier used) removes the most relative overhead.

**This is exactly the gap Phase 2 (LDS double-buffering) is meant to close**: with a
double-buffered LDS, every chunk's global load could safely overlap with the PREVIOUS tile's
compute (no read-after-write hazard against a buffer still being consumed), restoring full
latency hiding for every chunk, not just chunk 0 -- expected to turn fwd's current regression
into a win too, on top of bwd/wrw's already-larger gains.

## Phase 10 (2026-08-24): LDS double-buffering -- infrastructure landed, no perf win yet

Following up on Phase 9's finding that fwd's k-sub-loop regressed (chunks 1..N-1 have no
latency-hiding overlap at all), this phase ported gfx942/950's ping-pong double-buffer
technique (`mfma_main_loop.py`'s `v_xor_b32 v[offset], lds_single_size, v[offset]` pattern)
into `wmma_main_loop.py`, gated on a new optional tunable `lds_double_buffer` (default 0, every
existing config unaffected). `lds_single_size = igemm_next_pow2(lds_a_size + lds_b_size)`;
buffer 0 at LDS offset 0, buffer 1 at offset `lds_single_size`; `v_sst_a_os`/`v_sst_b_os`
(aliases of the same physical VGPR) and `v_sld_a_os`/`v_sld_b_os` are kept permanently one
buffer apart and XOR-toggled together once per outer iteration.

**A new correctness bug, found via the multi-tap hardware test specifically**: `v_sst_os`/
`v_sld_a_os`/`v_sld_b_os` are computed ONCE in `emit_kernel_prologue`, outside the runtime
tap loop, and get progressively XOR-toggled by every outer K-loop iteration. Since the SAME
compiled main-loop code is re-entered (via a runtime branch) once per tap, the buffer-parity
state left over from tap N's K-loop was silently carried into tap N+1's execution, misaligning
every read/write from the second tap onward -- 1x1/single-tap configs never exercise a second
tap, so this passed everywhere except multi-tap (y,x>1) configs specifically. Fixed by
recomputing (not saving to a "_base" VGPR and restoring) `v_sst_os`/`v_sld_a_os`/`v_sld_b_os`
fresh at the start of every tap, factored into a shared `_emit_lds_offset_setup()` method
called from both the prologue and the tap loop. Recompute, not save/restore, was a deliberate
choice: bwd's kernel is already at the hard 256-VGPR/wave limit even at K=32 single-buffered
(confirmed via `.vgpr_count`), so 3 more "_base" registers would not have fit at all --
recomputing from `v_tid` (already persistent) and `v_tmp` (already-available scratch) needs
zero new VGPRs.

**gfx1250's LDS-per-workgroup limit is at least 64KB**, confirmed empirically (no repo-side
hardware constant exists for this) -- both the standalone `_dbuf` variants (32KB fp16/bf16/
int8, 8KB fp32) and the combined `_k2x_dbuf` variants (64KB for all four precisions) codegen,
load, and run correctly on real hardware for all 12 (direction x precision) combos, full
battery (degenerate, multi-K-block, stride+pad, multi-tap+dilation, group=2/4) -- 144 test
cases, zero failures, zero regression to the 24 existing (single-buffered) configs (byte-
identical `.s` diff, same discipline as Phase 9).

### An abandoned attempt, and the honest benchmark result

An initial attempt tried to make double-buffering IMMEDIATELY useful by having untransposed
operands (fwd's A+B, bwd's A) issue+wait+store EVERY k-sub-loop chunk right away (before
compute), reasoning that writes to the "other" buffer are always safe regardless of timing.
This was WRONG in a way that only showed up in benchmarks, not correctness tests: it moved
chunk 0's WAIT from its original late position (after `emit_wmma_tile()`+`emit_extra_substeps()`,
maximally overlapped with compute) to immediately after its own load issue -- discarding the
existing overlap Phase 9 already had for chunk 0, for zero compensating benefit (chunks 1..N-1
still can't be pipelined ahead of their own wait regardless of LDS buffering, since they all
reuse the SAME small `v_gld_a`/`v_gld_b` staging buffer -- a VGPR-level serialization,
completely orthogonal to which LDS buffer the result lands in). This produced a MEASURED
regression (fwd/fp16 k2x+dbuf dropped to ~14 tflops vs k2x-alone's ~17.8) before being caught
and reverted back to Phase 9's exact functor structure.

**With that reverted, the honest result is: double-buffering is currently performance-neutral
everywhere** -- `_dbuf` matches the original single-substep config's tflops to within
measurement noise (same functors, same instruction positions, just alternating which LDS
buffer is targled), and `_k2x_dbuf` matches Phase 9's `_k2x` numbers the same way, across all
three directions. It does NOT fix fwd's Phase 9 regression as originally hoped.

**Why not, and what actually would**: double-buffering only removes the *cross-wave
LDS-overwrite race* that made storing early unsafe under single buffering. It does nothing
about the OTHER constraint that caused chunks 1..N-1 to have zero latency hiding in the first
place: they all share ONE small reused VGPR staging buffer (forced by the 256-VGPR/wave
limit), so chunk `i+1`'s load cannot even be ISSUED until chunk `i`'s data has been consumed
(read or stored) -- a buffer-reuse serialization, not an LDS-safety one. The only way to
actually hide that latency is to INTERLEAVE each chunk's load with a DIFFERENT piece of useful
work in between the issue and the wait -- concretely, pairing chunk `i`'s load with substep
`i`'s LDS-read-and-compute (a natural 1:1 pairing, since `num_k_chunks == num_k_substeps` by
construction, and LDS reads/ALU use different hardware queues than global memory, so they can
genuinely proceed concurrently with an in-flight global load). This is a real restructuring of
`wmma_main_loop.py`'s loop body (blending the chunk-load and substep-compute loops together,
not just reordering the existing separate phases) -- closer in spirit to mfma's own ~10
hand-scheduled interleaved variants than to a simple buffer-switch. Explicitly not attempted in
this phase; the double-buffering infrastructure landed here (LDS allocation, correct addressing
across taps, hardware-confirmed 64KB budget) is the necessary PREREQUISITE for it, not a
substitute.

## Phase 12 (2026-08-24): ISA probes for `global_load_async_to_lds_b128` and `global_load_tr16_b128`

User asked to investigate whether newer gfx1250 ISA instructions could eliminate the actual
root cause behind Phase 9's fwd regression: the VGPR-limited `v_gld_a`/`v_gld_b` staging buffer
that forces chunked, serialized global-tile loads (fp16/bf16/int8 kernels already sit at
252-256/256 VGPRs, zero headroom to grow it). Two candidate instructions, surfaced from an MI400
Shader Programming Guide excerpt (later cross-checked against AMD's public CDNA5 ISA manual --
both confirmed applicable to gfx1250):
- `global_load_async_to_lds_b128`: moves data global memory -> LDS directly, with **no VGPR ever
  holding the transferred data** -- if it works, it eliminates the staging buffer entirely rather
  than working around it.
- `global_load_tr16_b128`: loads a 16x16 tile of 16-bit data and transposes row-major<->
  column-major in the load itself -- could replace bwd/wrw's expensive "strided read-and-pack"
  LDS technique for transposed operands.

Both are brand-new to this codebase and unverified on real gfx1250 hardware. Following the same
standalone-probe discipline established during the original WMMA bring-up (hand-written `.s` +
a HIP host harness, `/tmp/wmma_probe/`), each was tested independently before any production
kernel changes.

### `global_load_async_to_lds_b128` + `s_wait_asynccnt`: CONFIRMED WORKING

Both mnemonics assemble on gfx1250 (`llvm-mc -mcpu=gfx1250 -show-encoding`) with the operand
form `global_load_async_to_lds_b128 VDST, VADDR, SADDR [offset:N]` -- VDST = per-lane LDS byte
address, VADDR = per-lane 32-bit global byte offset, SADDR = mandatory SGPR-pair 64-bit base
(an `off`/no-SADDR form was tried and rejected by the assembler). Exact pseudocode later
confirmed via AMD's public CDNA5 manual (section 10.8), matching the probe design exactly:
`LDS[VGPR[VDST][lane]+byte] = GLOBAL_MEMORY[VGPR[VADDR][lane]*ScaleFactor + SGPR[SADDR]+byte]`
(GVS mode, `ScaleFactor=1` since `SO` defaults to 0).

Probe files: `/tmp/wmma_probe/async_ld_probe.s`, `host_async_probe.cpp`. Kernel: each of 32 (or
64, spanning 2 waves) lanes computes `v1 = lane*16` (global byte offset, doubling as the LDS
byte address `v2`), issues `global_load_async_to_lds_b128 v2, v1, s[in_ptr]`, waits via
`s_wait_asynccnt 0x0`, barriers, reads back via `ds_read_b128`, and stores to an output buffer.
Host fills input with a sequential per-element-distinguishable pattern and poisons the output
before each launch.

**A self-inflicted false alarm, caught before it wasted the user's time re-reading the
manual for nothing**: the first two debugging attempts showed near-100%-failure with
structured-looking (not random) garbage. Root cause: the probe was missing `s_wait_dscnt 0x0`
between `ds_read_b128` and the `global_store_dwordx4` that consumes its result -- an unrelated,
mundane bug in the probe itself (LDS reads are asynchronous too, tracked by a *different*
counter, DSCnt, not ASYNCcnt), nothing to do with the new instruction at all. A control kernel
using plain `ds_write_b128`/`ds_read_b128` (no async load involved) with the SAME missing-wait
bug reproduced the identical failure pattern, confirming the diagnosis before touching the real
probe.

**Once fixed, fully confirmed on real hardware** across 100 trials each of: 32-lane/SADDR=buffer
start, 32-lane/SADDR=buffer start+400 (nonzero base, isolating SADDR's contribution from
VADDR's), and 64-lane/2-wave (no cross-wave hazard) -- 0/100 failures each. A negative control
(identical kernel, `s_wait_asynccnt` deleted) showed 6/100 failures with genuinely random
(not structured) garbage, confirming the wait is load-bearing rather than merely-decorative.

**This is a real, hardware-confirmed win**: the instruction moves data global-memory -> LDS
with zero VGPR cost, exactly as documented. This is the necessary building block for eliminating
Phase 9's VGPR-staging-buffer bottleneck -- **not yet integrated into the real kernel codegen**;
that integration is separate, larger follow-on work (needs to replace `wmma_main_loop.py`'s
`v_gld_a`/`v_gld_b`-staged global-load functors, re-run the full regression/hardware-validation
battery, and re-benchmark fwd's Phase 9 regression to see if it's actually fixed).

### `global_load_tr16_b128`: CONFIRMED WORKING (formula fully reverse-engineered)

Confirmed via `llvm-mc`: `global_load_tr16_b128 v[a:a+3], VADDR, SADDR` -- unlike the async
instruction, this one **writes 4 VGPRs** (128 bits = 8 packed fp16 elements), tracked by ordinary
`LOADcnt` (`s_wait_loadcnt`), not `ASYNCcnt`. Also confirmed a second valid operand form,
`v[a:a+3], v[b:b+1], off` (VADDR as a full 64-bit address, no SADDR) -- both forms work; the
SADDR+32-bit-offset form is what's used below.

**Getting here took several wrong turns worth recording** so the next person doesn't repeat
them. Section 10.9.2's lane-mapping diagram (read as a rendered page image, since it's
graphical, not extractable as text) gives a clean per-lane picture, but cross-checking it
against real hardware kept producing self-consistent-but-wrong results -- three different
VADDR hypotheses (shared address, `k(lane)*32` "column byte offset", plain `lane*32` linear)
each gave a different, deterministic, but non-matching pattern. The breakthrough came from two
directions at once:
1. Finding the actual LLVM/MLIR source (`GlobalTransposeLoadOpLowering` in
   `mlir/lib/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.cpp`) confirmed the addressing itself is a
   completely ordinary strided-memref pointer -- no hidden bit-tricks, no special scaling. This
   ruled out an entire class of "maybe VADDR needs a weird scale factor" hypotheses.
2. Re-reading section 7.12.2 ("Matrix Element Storage in VGPRs") more carefully -- and noticing
   7.12.3 explicitly says sparse-matrix transpose loads work "the same as the 16x16 matrix
   load" -- revealed that 7.12.2's table (not the 10.9.2 diagram) is the actual authoritative
   VGPR output layout: **lane number maps DIRECTLY to the matrix row (M = lane & 15, no folding
   with `&7`)**, with the K (column) dimension spread across the 4 VGPRs, K=0-7 for lanes 0-15
   and K=8-15 for lanes 16-31 -- i.e. exactly the standard WMMA A-operand storage format
   (matching gfx1250 WMMA's own hardware-verified formulas from the original bring-up, see
   [[gfx1250-wmma-bringup]]), since that's the entire point of this instruction: feed WMMA
   operands straight from column-major memory.

**Confirmed formula** (source matrix stored **column-major**: `mem[col*16+row] = value(row,col)`):
```
k(lane)    = (lane & 7) | ((lane & 16) >> 1)
bit3(lane) = (lane >> 3) & 1
VADDR_bytes(lane) = k(lane)*32 + 16*bit3(lane)
```
Reading 8 contiguous elements from that address (a completely ordinary, non-strided burst read
-- there is no hardware crossbar shuffling data between lanes) gives, for lane `L` and VGPR
element `e` (0..7): `M = L & 15` (row, direct), `K = e + 8*(L >= 16)` (column). **Verified
against real hardware with 0/256 mismatches** across all 32 lanes x 8 elements, using a
column-major 16x16 fp16 test matrix with distinguishable values (`value(row,col)=row*16+col`).

This means: for a real kernel, computing `k(lane)` and the `+16*bit3(lane)` adjustment (5-6
cheap VALU instructions) is all that's needed to load a WMMA A/B-operand tile directly from
column-major memory in one instruction, in exactly the layout WMMA already expects -- no LDS
round-trip, no multi-instruction strided read-and-pack (the technique bwd/wrw currently use for
transposed operands, see the class docstrings in `igemm_bwd_gtc_wmma_nhwc.py`/
`igemm_wrw_gtc_wmma_nhwc.py`). **Ready for integration**, pending the same kind of
regression-safety discipline used for every other main-loop change (byte-identical `.s` diff +
full hardware battery) once a concrete integration plan is written.

### Probe artifacts and toolchain (for reproducing or extending this work)

All probe `.s`/`.cpp` files live in `/tmp/wmma_probe/` (ephemeral -- copy out anything worth
keeping before the session/machine cycles). The winning kernels are `async_ld_probe.s`
(Probe 1) and `tr16_probe_g.s` (Probe 2, symbol `tr16_probe_g` -- the confirmed-working
formula; earlier `tr16_probe*.s` files in the same directory are the dead-end variants kept
for the historical record above, not for reuse). Toolchain, confirmed working: assemble with
`/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 <probe>.s -o
<probe>.hsaco` (single step, no separate link -- the full vendor triple `amdgcn-amd-amdhsa` is
required, the vendor-less `amdgcn--amdhsa` form causes a target-id mismatch on gfx1250); host
harness via `/opt/rocm/bin/hipcc -std=c++17 <host>.cpp -o <host>`, run as `./<host>
<probe>.hsaco`. `llvm-mc -arch=amdgcn -mcpu=gfx1250 -show-encoding` is a fast, no-hardware-needed
first gate for checking whether a candidate mnemonic/operand-form is recognized at all before
writing a full probe.

## Phase 13 (2026-08-25): `global_load_async_to_lds_b128` integrated into fwd (A+B) and bwd (A)

User asked to integrate and benchmark Phase 12's two confirmed instructions. Instruction 2
(`global_load_tr16_b128`) carries extra structural risk (address-formula generalization to
non-packed real tensors never probed, a wave-cooperative SADDR-sharing model that doesn't
obviously compose with wrw's B-operand padding) and was explicitly deferred by the user to its
own later phase, gated behind its own hardware micro-probes. This phase covers **only**
`global_load_async_to_lds_b128`, wired into the untransposed operands: fwd's A (input) + B
(weight), bwd's A (grad_output). bwd's B and wrw stay on the existing technique (transposed,
out of scope -- that's Instruction 2's job).

### Design

New per-operand `ctrl.async_global_to_lds_a`/`_b` flags in `wmma_main_loop.py` (independent,
not a single combined flag -- bwd needs A async while B stays on the old technique). A new
tunable `async_global_load` (`igemm_base.py`, default 0) gates a from-scratch functor,
`_emit_gld_async_all_chunks`, that issues all `num_k_chunks` chunks' worth of
`global_load_async_to_lds_b128` directly (no VGPR staging buffer at all -- the load *is* the
LDS write), waited once via `s_wait_asynccnt 0x0` in the restructured main loop instead of the
old chunk-by-chunk `global_load_dwordx4`+`ds_write_b128`+`s_wait_loadcnt` dance. Address
computation simplifies too: the old design merged a 64-bit VGPR address (SGPR base +
per-lane offset, via `v_add_co_u32`/`v_add_co_ci_u32`) for use with `global_load_dwordx4`'s
`off` form; the new design keeps only the per-lane byte offset (`v_off_a`/`v_off_b`, 1 VGPR)
and passes the SGPR base directly as SADDR. VGPR count drops from 252 to 220 for fwd/fp16
128x128 as a result.

Padding masking (fwd's A, bwd's A -- the only operands with `v_flag`): a masked-off lane's
async load simply never touches its LDS destination (confirmed via a dedicated Stage 0
EXEC-masking probe, extending `/tmp/wmma_probe/async_ld_probe.s` -- 100/100 trials, 0 mismatches
either direction). Since load+store are now fused into one instruction, this needs an explicit
zero-fill: invert EXEC after the masked loads and `ds_write_b128` a persistent all-zero `v_zero`
quad into the same destinations for the now-active (previously masked-off) lanes.

### A real bug found only by full-kernel hardware testing, not by any isolated probe

The new fwd/fp16 config passed assembly and looked structurally correct on inspection, but
produced **exactly zero** output for every element on real hardware. Four escalating isolated
probes (masked-load semantics, cross-wave LDS visibility, 8-outstanding-ops-per-wait with mixed
masked/unmasked SADDR bases) all **passed** -- the bug was in none of the mechanisms those
probes exercised. A debug patch that overwrote the kernel's first accumulator register with
`v_a`'s or `v_b`'s raw loaded contents (bypassing WMMA compute entirely, to see what actually
landed) showed **A had real data, B was exactly zero** -- despite both operands using the
identical instruction pattern.

The difference: `_emit_gld_async_all_chunks` folded a compile-time constant
(`sst_extra_off`, e.g. B's `lds_a_size=8192` LDS-region shift) directly into the shared
per-chunk immediate: `offset:{sst_extra_off + idx*16}`. This is safe for the *old* design
(separate `global_load_dwordx4` + `ds_write_b128` instructions, where the store's offset never
touches the load's source address) but **not** for this fused instruction. A dedicated probe
(`offset_semantics.s`, `/tmp/wmma_probe/`) proved conclusively that
`global_load_async_to_lds_b128`'s immediate `offset:N` shifts **both** the LDS destination
(VDST) *and* the global source address (VADDR+SADDR) by the same `N` -- confirmed by writing
with `offset:0` then `offset:512` from a fixed VDST/VADDR against a source buffer with distinct
16-byte patterns at bytes 0 and 512: the second write's *destination* landed at LDS+512 as
expected, but it also read from `src[512:528)` rather than `src[0:16)`, proving the source
shifted too. For A (`sst_extra_off=0`) this coincidentally caused no visible bug; for B
(`sst_extra_off=8192`), it silently redirected every read 8192 bytes past the intended weight
address -- an out-of-bounds/garbage read that this hardware/driver combination happens to
return as zero.

**Fix**: bake `sst_extra_off` into the VDST *register* once (a single `v_add_u32` into a new
scratch register, `v_sst_tmp`), leaving the shared per-chunk immediate as pure `idx*16` for
both operands. Applied to both `igemm_fwd_gtc_wmma_nhwc.py` (where it's live, both A and B) and
`igemm_bwd_gtc_wmma_nhwc.py` (where A's `sst_extra_off` is always 0 today, so the bug never
manifested there -- fixed anyway for parity/future-proofing, in case a later variant ever needs
a nonzero shift for that operand).

### Validated on real hardware (fwd/fp16 128x128, all 6 battery cases)

After the fix: degenerate 1x1/no-pad (single K-block), multi-K-block, stride+pad,
multi-tap+dilation, group=2, group=4 -- all `valid:y` against `naive_conv_fwd_nhwc`. Byte-
identical regression check: all 48 existing gfx1250 configs regenerate with zero failures
(the new code only executes inside `if tunable.async_global_load:` branches, which default to
0 and are never reached by any existing config).

### Benchmark: CORRECTED -- a consistent regression, not a win (initial "+3-5%" finding was a measurement artifact)

**This section was originally written with `IGEMM_WARMUP=3 IGEMM_REPEAT=8` (Phase 9's exact
methodology) and reported a "+3-5% win" for fwd/fp16 at the single-tile-block size. That number
does not hold up.** Re-measuring the identical comparison at `IGEMM_REPEAT=20-30` (interleaving
old/new runs to rule out clock-drift bias) flips the result: old is consistently *faster* than
new. `REPEAT=8` was simply too few samples for a workload this fast (~0.1ms) to average out
run-to-run scheduling/clock noise -- a real methodology gap, not a hardware fluke. Lesson for
any future benchmark on this machine: don't trust a `tflops` delta smaller than the driver's own
reported noise percentage without re-running at higher repeat and, ideally, interleaved order.

Corrected numbers, `c=k=2048`, 1x1, `IGEMM_WARMUP=5 IGEMM_REPEAT=20-30`, old (VGPR-staged) vs
new (async) config, matching 128x128 kernel only in each config:

| direction/precision | shape | old tflops | new tflops | delta |
|---|---|---|---|---|
| fwd/fp16 | single-block (`n=1`) | 10.15-10.44 | 9.56-9.85 | **-6%** |
| fwd/fp16 | GPU-saturating (`n=8`) | 84.8 | 81.8 | **-3.6%** |
| fwd/bf16 | single-block | 10.40 | 9.82 | **-5.6%** |
| fwd/bf16 | GPU-saturating | 84.4 | 81.6 | **-3.3%** |
| fwd/int8 | single-block | 17.61 | 15.85 | **-10%** |
| fwd/int8 | GPU-saturating | 149.5 | 132.7 | **-11.2%** |
| fwd/fp32 | single-block | 3.70 | 3.30 | **-11%** |
| fwd/fp32 | GPU-saturating | 29.7 | 27.0 | **-9.1%** |
| bwd/fp16 | single-block | 4.06 | 3.95 | **-2.9%** |
| bwd/fp16 | GPU-saturating | 33.1 | 32.2 | **-2.6%** |

**Consistent regression, every precision, both directions tested, both problem sizes.** No
combination measured showed a real win. fwd's int8/fp32 are hit hardest (~9-11%); fp16/bf16
(structurally near-identical load paths, same `data_byte`) regress similarly to each other
(~3-6%); bwd/fp16's smaller regression (~3%) likely reflects that only its A operand (not both
A and B) goes async.

**Why, when the mechanism removes VGPR-staging serialization entirely**: not conclusively
diagnosed -- plausible candidates, none yet tested in isolation: (1) `global_load_async_to_lds_b128`
itself may simply have higher per-instruction latency than `global_load_dwordx4` on this
hardware, independent of the staging-buffer serialization it replaces; (2) freeing VGPRs
(252->220 for fwd/fp16) doesn't help if occupancy was never the bottleneck at this tile size,
while the old design's now-removed instructions may have given the scheduler more to overlap
with; (3) `s_wait_asynccnt`'s cross-wave completion semantics (a new counter, no prior tuning
history on this hardware) could have different/worse latency characteristics than the
well-understood `s_wait_loadcnt`/`s_wait_dscnt` pair. Not investigated further -- out of scope
for this phase, which was integration + honest benchmarking, not root-causing a regression in a
newly-adopted instruction.

**Net**: the instruction works exactly as documented and every hardware-correctness battery
passes, but it is not a performance win anywhere tested. Since `async_global_load` defaults to
0 and lives only in new, separate `_async` config files, this doesn't regress anything already
shipped -- but it also doesn't deliver the speedup this integration was undertaken for.
**Explicit user decision**: keep the implementation anyway -- gfx1250 is pre-production
hardware, so the instruction's relative cost could improve on future silicon revisions; a
correct, validated, default-off implementation costs nothing to leave in place and could become
a real win later with zero code changes, just a re-benchmark.

### bf16/int8/fp32 extension (2026-08-25): correctness confirmed, same regression pattern

Extended to fwd (A+B) and bwd (A) for bf16/int8/fp32, plus a bwd/fp16 config that hadn't been
written yet -- purely mechanical, the load-path code (`_emit_gld_async_all_chunks`,
`chunk_num_dwordx4`/`num_k_chunks`) was already precision-generic from Phase 1's k-sub-loop
work, no kernel-file changes needed. New files: `igemm_{fwd,bwd}_gtc_gfx1250_nhwc_{bf16,int8,
fp32}_async.config`, `igemm_bwd_gtc_gfx1250_nhwc_fp16_async.config`. All 7 pass the full 6-case
hardware battery against `naive_conv_{fwd,bwd}_nhwc` (int8's `gemm_k_per_block=64` and fp32's
`gemm_k_per_block=4` needed shape-adjusted test cases, e.g. `c`/`k` multiples of the per-
precision block size rather than fwd/fp16's 32 -- no code changes, just correctly-sized test
inputs). One dead end during battery validation: initial bwd group=2/4 test shapes (chosen by
guessing `gemm_n=c/group`, `gemm_k=k/group` from the class docstring) gave `valid:n` for **both**
old and new configs identically -- proven pre-existing/unrelated to Phase 13 by reproducing on
the untouched old config, then resolved by using round per-group values (`gemm_n=gemm_k=128`)
instead of the original odd split (`gemm_n=128,gemm_k=64`); not investigated further since it
reproduces without any Phase 13 change. Benchmark: same consistent regression pattern as fp16
(see table above) -- no precision or direction showed a win.

### Deferred (per user's original request, sequenced as instructed)

Instruction 2 (`global_load_tr16_b128`) integration into bwd's B and wrw's A -- needs its own
hardware micro-probes first (address-formula generalization to non-packed real tensors; a
wave-cooperative SADDR-broadcast scheme). Also deferred: the full tile-shape-diversity backlog
from Phase 11 (64x64 port to bwd/wrw, asymmetric 64x128/128x64 shapes, generalizing the
`v_tid`-as-row-index mapping) -- explicitly sequenced by the user to resume after instruction
integration+benchmarking finished (now done, with a negative result -- see above).

### Critical files

- `python/operations/wmma_main_loop.py` -- per-operand `async_global_to_lds_a`/`_b` control
  flags, restructured `emit()`
- `python/igemm/igemm_fwd_gtc_wmma_nhwc.py` -- `_emit_gld_async_all_chunks`, `v_sst_tmp`,
  address simplification (both A and B)
- `python/igemm/igemm_bwd_gtc_wmma_nhwc.py` -- same pattern, A operand only
- `config/igemm_fwd_gtc_gfx1250_nhwc_fp16_async.config` -- new async-enabled config (separate
  file, not appended to the shared fp16 config -- the kernel-name mangling doesn't fold
  `async_global_load` into the symbol name, so two same-shape sections in one file collide)
- `/tmp/wmma_probe/offset_semantics.s`, `async_deep.s`, `async_crosswave.s`, `async_multi.s` --
  the escalating probes used to isolate this phase's bug (ephemeral, copy out before reuse)

## Phase 14 (2026-08-25): `global_load_tr16_b128` into bwd's B -- implemented, hardware-tested, found architecturally incompatible, reverted

Following Phase 12's two confirming probes (stride generalization to `STRIDE=128`, wave-
cooperative SADDR broadcast via `v_readfirstlane_b32`), attempted the integration into bwd's B
(weight) operand per the approved plan: a new `tr16_load_b` tunable, an `_emit_tr16_load_b_all`
functor replacing B's entire global-load+LDS-store+LDS-read pipeline (2 `global_load_tr16_b128`
calls per `wave_repeat_n` step, one per K-half, writing directly into `v_b`), `move_slice_window_b`
reduced to a scalar SGPR bump. Implemented, then hardware-tested against the simplest possible
case (`n1c128H16W8k32y1x1`, single K-block, no padding).

**First hardware failure** (`nrms=0.280` vs threshold `0.0082`): `PRINT_EVERY_PIXEL=1` showed a
clean "groups of 8 channels repeat with an offset-by-one-group" pattern. Root cause: the
K-half SADDR shift (`s_wei_khalf_stride`, meant to advance the read by "16 K_out positions") used
`utility_log2(data_byte) + 4` as the shift amount (`*32` for fp16) -- but Phase 12's confirmed
`VADDR_bytes = k(lane)*STRIDE + 16*bit3(lane)` formula's "16" is a **fixed byte constant**, not an
element count needing `data_byte` scaling. Fixed to a plain shift of 4 (`*16`).

**Second hardware failure, after the fix** (`nrms=0.124`, improved but still ~15x over
threshold): per-channel errors were now close-but-wrong throughout, with no obvious repeating
structure (unlike the first failure). Working through the full per-lane address/output mapping
by hand (not just re-checking the formula, but tracing exactly which physical lane produces which
`(N, K)` output value) surfaced a **structural incompatibility**, not a further addressing bug:

- `get_gemm_index_for_src_matrix_transposed` (the proven-correct LDS-based technique this was
  replacing) requires each lane's **entire** `v_b(i_rn*8 : i_rn*8+7)` register range (all 8, not a
  sub-range) to hold **16 consecutive K values, entirely from that one lane** -- `k_half=0` lanes
  own K=0-15 in full, `k_half=1` lanes own K=16-31 in full, for the *same* N-column (`lane%16`).
  This is WMMA hardware's own fixed per-lane K-reduction convention (every existing operand/
  precision already relies on it); it is not a free software choice that TR16 could satisfy some
  other, equally-valid way.
- `global_load_tr16_b128`'s own per-lane output, worked out fully from Phase 12's confirmed
  formula (`SOURCE_ROW = (K>>3)*8 + (M&7)`, `SOURCE_COL = (M>>3)*8 + (K&7)`, derived by tabulating
  all 32 lanes), gives each lane's dwordx4 an **8-element slice whose K-range is hard-wired by
  that lane's own wave-half** (`K=e` for lanes 0-15, `K=e+8` for lanes 16-31) -- lane 3 can
  *never* produce output labeled `K=8-15` no matter what SADDR is chosen (SADDR only changes
  *which real source row* gets that fixed label, never the label itself), and lane 19 can never
  produce `K=16-31`.
- These are two genuinely different "upper/lower half of the wave" splits that do not compose:
  WMMA needs lane 3 to independently hold *all* of K=0-15 for N=3, while TR16 structurally forces
  K=0-15 for N=3 to be split *across* lanes 3 and 19. No amount of extra `global_load_tr16_b128`
  calls or SADDR arithmetic can move lane 19's data into lane 3's own VGPRs -- that requires an
  explicit cross-lane exchange (e.g. `ds_permute`/`v_permute_b32` or a small LDS bounce),
  which was explicitly out of scope for the user's "simplest first: no LDS, measure it" framing
  and would need a materially larger redesign (doubling the call count and adding a permute step)
  to even attempt.

**Decision: reverted.** This is a "measure, don't assume" negative result like Phase 9/13's
performance findings, just discovered analytically instead of by benchmark -- the two Phase 12
probes remain true, hardware-confirmed facts about `global_load_tr16_b128` in isolation, they are
just insufficient (on their own) to populate a K=32-wide WMMA B-operand under the current
per-lane k_half convention. The `tr16_load_b` tunable, `_emit_tr16_load_b_all`, and the
`_tr16.config` file were all removed; `igemm_bwd_gtc_wmma_nhwc.py`/`igemm_base.py` are back to
exactly Phase 13's shape (byte-identical `.s` regression re-confirmed for all 12 default
fwd/bwd/wrw x fp16/bf16/int8/fp32 configs, ignoring one cosmetic comment-text change from an
unrelated earlier edit). wrw's A was not attempted separately: it is the same K=32-wide WMMA
operand consumption pattern, so it would hit the identical incompatibility.

Any future attempt at this instruction for a >16-wide K operand needs to design the cross-lane
redistribution step first (and benchmark whether the extra permute/LDS-bounce cost still beats
the existing strided-read-and-pack technique it would replace) rather than assuming a direct
global-to-`v_b` path works.

### Critical files (Phase 14, reverted -- listed for anyone revisiting this)

- `/tmp/wmma_probe/tr16_stride_probe.s`, `tr16_saddr_probe.s` -- the two Phase 12 follow-on
  probes confirming stride generalization and wave-cooperative SADDR broadcast (still valid,
  ephemeral, copy out before reuse)
- `docs/gfx1250_wmma_layout.md`'s Phase 12 section -- the base `global_load_tr16_b128` formula
  and the `SOURCE_ROW`/`SOURCE_COL` per-lane derivation this phase's finding depends on

## Phase 11 (2026-08-24): tile-shape diversity, mechanism + first new shape (fwd/fp16 only)

User asked to prioritize coverage (tile-shape diversity) over further interleaving-perf work.
All three kernel files were pinned to exactly one macro-tile shape (128x128, wmma_repeat 4x4,
block_size=128) -- a real coverage gap, since convolution shapes whose `gemm_m`/`gemm_n`
(`n*ho*wo` / `k/group`) don't land on a multiple of 128 have no valid WMMA kernel today even
when they divide evenly into something smaller.

**Mechanism (all 3 directions, zero new configs, byte-identical for every existing config)**:
1. Relaxed the 4 hard `assert tunable.gemm_m_per_block == 128 ...` shape-pinning asserts in
   each kernel's `__init__` (fwd:152-155, bwd:89-92, wrw:144-147 pre-change) to real validity
   checks matching `igemm_base.py:289-295`'s own formula (`gemm_m_per_block % (wmma_tile_m *
   wmma_repeat_m) == 0`, `block_size == waves_per_m*waves_per_n*32`) -- redundant with that
   upstream check by design, not the sole gatekeeper.
2. Generalized the `*128`/shift-by-7 address math (group-decode's `blocks_per_group_n = ceil
   (gemm_n/128)` and the `s_block_m_off`/`s_block_n_off` computation, 3 call sites x 3 files)
   to `utility_log2(tunable.gemm_m_per_block/gemm_n_per_block)` and `gemm_n_per_block - 1`.
3. Fixed 2 literal `range(4)` loops in `igemm_wrw_gtc_wmma_nhwc.py`'s `shared_load_a/b_functor`
   (lines 821/857 pre-change) to `range(outer.tunable.wmma_repeat_m/n)` -- fwd/bwd's equivalent
   loops were already generic.

Verified byte-identical (modulo comment text) for all 48 existing gfx1250 configs (`.inc` diff
against a pre-change baseline), matching the discipline established in Phase 9/10.

**A real, previously-undiscovered constraint, found only by testing a new shape on hardware**:
the fwd/bwd/wrw kernels' GLOBAL-LOAD thread mapping is a *separate* indexing system from the
WMMA compute/epilogue indexing in `wmma_mapping.py` (which IS fully generic, confirmed by
static audit). The global-load functors use `v_tid` **directly** as the per-thread
`gemm_m`/`gemm_n` row index within the block (e.g. fwd: `v_add_u32 v[v_gtc_tmp(0)],
s[s_block_m_off], v[v_tid]   ; m_idx`) -- this silently requires `block_size ==
gemm_m_per_block == gemm_n_per_block` exactly, true for the existing 128x128/block_size=128
shape only by coincidence of all three being 128. A first attempt at a symmetric 64x64/
wave_repeat-2x2/block_size=128 shape (chosen because it kept the same 4-wave grid as the
existing config) **assembled and ran but produced all-zero output** on real hardware --
`PER_PIXEL_CHECK=1`'s per-pixel diff showed `pred:0.000000` at every index, meaning roughly
half the threads (`v_tid` 64..127) computed `m_idx`/`n_idx` values 64..127 that fall entirely
outside the intended 64-row block, corrupting/missing the actual global-memory read and the
LDS write it feeds. This bug is invisible to a purely static audit (it's not a literal `128`
anywhere) -- only exercising a genuinely different shape on hardware surfaced it.

**The fix**: with `wave_tile` fixed at 16 (only verified value), the only way to hit
`block_size == macro_tile_m == macro_tile_n` for a tile smaller than 128 is an ASYMMETRIC wave
grid, since `block_size = waves_per_m*waves_per_n*32` forces `waves_per_m*waves_per_n = 2` for
`block_size=64`, i.e. `(waves_per_m, waves_per_n) = (2,1)` or `(1,2)`, never `(1,1)`(->32x32)
or `(2,2)`(->128x128, the existing shape). Landed shape: `wave_repeat_m=2, wave_repeat_n=4,
waves=2` (`waves_per_m=2, waves_per_n=1`, `block_size=64=gemm_m_per_block=gemm_n_per_block`).
`accumulate_c = wave_repeat_m*wave_repeat_n*num_v_c = 2*4*8 = 64` (down from 128 at 4x4),
`accumulate_a = wave_repeat_m*num_v_a = 16`, `accumulate_b = wave_repeat_n*num_v_b = 32` --
total 112 VGPRs vs the existing shape's 192 for fp16/bf16/int8, a real but more modest
reduction than a (invalid) naive symmetric 2x2/block_size=128 split would have suggested.

New `ctrl_wmma_mapping_t(64, 64, 16, 16, 2, 2, 4, ...)` table entries added for all four
precisions in `wmma_mapping.py` (data only, `get_ctrl_wmma_mapping_from_wave_tile`'s filter
logic is unchanged). One new `[igemm_fwd_gtc]` config section added to
`config/igemm_fwd_gtc_gfx1250_nhwc_fp16.config` (`gemm_m/n_per_block=64, gemm_k_per_block=32,
wmma_repeat_m=2, wmma_repeat_n=4, tensor_a/b_thread_lengths=[1,32,1,1] (unchanged from the
128x128 config -- gemm_k_per_block didn't change), tensor_a/b_cluster_lengths=[1,1,1,64]`
(block_size=64, not 128)). `.vgpr_count: 172` for the new kernel (confirmed empirically from
the assembled `.s`), comfortably under the 256/wave ceiling.

**Validated on real gfx1250 hardware** through `conv_driver.exe`, fwd/fp16 only, against the
driver's own `naive_conv_fwd_nhwc` reference, using shapes with `gemm_m`/`gemm_n` chosen as
multiples of 64 but not 128 (so only the new tunable is valid/exercised): degenerate 1x1/
stride-1/no-pad, multi-K-block (`gemm_k=96`, three 32-blocks), stride+padding, multi-tap+
dilation, and `group=2` -- all `valid:y`. A control shape with `gemm_m=gemm_n=128` (divisible
by both 64 and 128) confirmed both the existing 128x128 kernel and the new 64x64 kernel
independently produce correct output side by side, with zero change to the 128x128 kernel's
generated assembly.

### Phase 11 continued (2026-08-25): bwd/wrw port + bf16/int8/fp32, all 12 combos hardware-validated

Ported the 64x64 shape to bwd and wrw, then to bf16/int8/fp32 for all three directions --
config-only work, exactly as predicted: bwd's global-load uses `v_tid` directly for its M side
and an already-generic `row_local`/`col_group` decomposition for N (same algebraic constraint,
different-looking code); wrw uses the `row_local`/`col_group` decomposition for both operands.
Neither needed kernel-file changes -- the `wmma_mapping.py` table already had all 4 precisions'
`ctrl_wmma_mapping_t(64, 64, 16, 16, 2, 2, 4, ...)` entries from the original Phase 11 work, and
`num_vgpr_accumulate_{a,b,c}` is computed identically across all three kernel files from the
same tunable fields. New config sections: 64x64 added to all 12 `igemm_{fwd,bwd,wrw}_gtc_
gfx1250_nhwc_{fp16,bf16,int8,fp32}.config` files (fwd/fp16 already had it). VGPR counts:
bwd/fp16 176, wrw/fp16 171 (both comfortably under the 256/wave ceiling, similar to fwd's 172).

**All 12 (direction x precision) combos pass the full 6-case hardware battery** (degenerate,
multi-K-block, stride+pad, multi-tap+dilation, group=2, group=4) against `naive_conv_{fwd,bwd,
wrw}_nhwc`, plus a control shape confirming the 128x128 kernel in each file is untouched.

**A pre-existing, orthogonal quirk found while validating group>1** (not a Phase 11/13 bug):
bwd's group>1 support is sensitive to the exact `gemm_n`/`gemm_k` split chosen for the test --
some shape choices (e.g. `gemm_n=128,gemm_k=64`) give `valid:n`, while "round" splits with
`gemm_n==gemm_k` reliably pass. Proven unrelated to any change in this session by reproducing
the exact same failure on the untouched, pre-Phase-11 128x128 bwd/bf16 config. Not investigated
further -- outside the scope of tile-shape-diversity work, flagged here so a future session
doesn't waste time re-discovering it. Every hardware battery test in this phase was run with
round splits specifically to route around it.

**Still not done at this point**: the asymmetric 64x128/128x64 shapes, and generalizing the
global-load functors' `v_tid`-as-row-index thread mapping. These two are linked more tightly
than the original Phase 11 backlog implied: `block_size` must simultaneously equal
`gemm_m_per_block` (for A's one-row-per-thread load) AND `gemm_n_per_block` (for B's), which is
only possible when they're equal -- an asymmetric tile shape (`gemm_m_per_block !=
gemm_n_per_block`) is **not reachable at all** without first changing at least one operand's
load functor to handle more than one row per thread. See the next section for how this was
resolved.

### Phase 11 continued again (2026-08-25): first asymmetric shape (128x64, fwd/fp16), `row_repeat` mechanism

Researched whether gfx950's mature XDLOPS kernels (`igemm_fwd_gtc_nhwc.py`, no `_wmma_`) already
solve this: they do, via cluster-length-based thread dispatch
(`igemm_thread_cluster_index_dispatcher_t`/`_accumulator_t`, `igemm_base.py:961-993`) where each
operand's `tensor_{a,b}_cluster_lengths`/`thread_lengths` independently determine how many rows
each thread owns. WMMA's config format already carries these fields (unused, dead weight for
address generation today). Rather than port the full generic multi-dimensional dispatcher,
implemented a much simpler WMMA-specific equivalent: **the WMMA compute side
(`wmma_mapping.get_gemm_index_for_src_matrix`) has zero dependence on which thread loaded which
global row into LDS** -- it only requires LDS byte offset `R*bytes_per_row` to hold global row
`R`'s data, for every `R` in `[0, gemm_m_per_block)`, written by *some* thread exactly once. So:
let `row_repeat_a = gemm_m_per_block // block_size` (generalizing the old `==` requirement to
`%==0`); thread `tid` owns rows `tid, tid+block_size, tid+2*block_size, ...` (`row_repeat_a` of
them). `tensor_a_cluster_lengths`/`thread_lengths` in the config only need to satisfy
`igemm_base.py`'s generic validation asserts (`product(cluster)==block_size`,
`product(thread)*block_size==gemm_m_per_block*gemm_k_per_block`) -- they're not read by the
kernel's address generation, same as before.

**First target**: `gemm_m_per_block=128, gemm_n_per_block=64`, fwd/fp16 only (matching Phase
11's own "first new shape" precedent). Wave/repeat choice (`waves_per_m=2, wave_repeat_m=4,
waves_per_n=1, wave_repeat_n=4, waves=2` -> `block_size=64`) means `gemm_n_per_block(64) ==
block_size(64)` already -- **B needs zero changes**; only A needs `row_repeat_a=2`, confining
the entire change to one operand's load pipeline in `igemm_fwd_gtc_wmma_nhwc.py`. Deliberately
kept separate from Phase 13's async-load path for this first pass (`async_global_load=0`).

**VGPR budget was tighter than expected**: a first implementation persisted
`row_repeat_a`-sized copies of `v_flag`/`v_n_idx`/`v_ho_idx`/`v_wo_idx`/`v_addr_a`, landing at
258 VGPRs -- 2 over the 256/wave hardware ceiling (assembler: "register index is out of
range"). Fixed by NOT persisting `v_n_idx`/`v_ho_idx`/`v_wo_idx` for extra rows at all: row 0
keeps its persistent registers exactly as today; rows 1..row_repeat_a-1 recompute their own
`(n_idx, ho_idx, wo_idx)` FRESH every tap from `v_tid+i*block_size`, reusing the existing 5-slot
`v_gtc_tmp` scratch pool (sequenced so each division-macro output is fully consumed before being
overwritten by the next stage) -- landing the results in the SAME registers
(`v_gtc_tmp(0)/(1)/(2)`) the unchanged downstream hi_idx/wi_idx/flag/row_idx code already reads
its input from, so that code is shared, byte-for-byte, between every row. Final cost: 255 VGPRs
(+3 total: `v_flag` +1, `v_addr_a` +2), not +6.

**A real bug found only on hardware, not by any static check**: the fixed-VGPR version compiled
and looked structurally sound, but produced results with `nrms=0.137` against a `0.0082`
threshold (`valid:n`) on every test. Per-pixel diagnosis (`PRINT_EVERY_PIXEL=1`, comparing
*absolute* not relative error to avoid a red herring -- relative error blows up trivially near
zero-crossings and initially made the bug look like uniform fp16 rounding noise) showed the
first 64 of 128 output rows (row-repeat-set 0, the untouched code path) were correct to
fp16-rounding precision, while the second 64 (row-repeat-set 1, the new fresh-recompute path)
were substantially wrong. Root cause: the fresh-recompute code passed `s.s_tmp()` (== `s_tmp(0)`)
as the division macro's scalar scratch argument -- but `s_tmp(0)`/`s_tmp(1)` were ALSO holding
the shared `iy*dilation-pad` values computed once at the top of `_emit_tap_gather` and read
again immediately after, for every row's hi_idx/wi_idx computation.
`macro_int_div_rem_vs_gfx1250_t` writes to `s[\s_tmp4+0]` internally (confirmed by reading the
macro's `emit()`), silently corrupting that shared pad offset for row 1's (and only row 1's)
subsequent hi_idx/wi_idx math. The prologue's identical-looking division calls (row 0, computed
once before any pad-offset value exists yet) never hit this, which is why the bug was invisible
there. Fixed by using `s.s_tmp(2)` (free at that point in the function, not needed until the
B-side computation after the row loop) as the division macro's scratch instead.

**Validated on real hardware**, fwd/fp16/128x64, full 6-case battery (degenerate, multi-K-block,
stride+pad, multi-tap+dilation, group=2, group=4) against `naive_conv_fwd_nhwc` -- all `valid:y`.
Byte-identical regression check: all 57 existing gfx1250 configs regenerate identically (the new
`row_repeat_a`/`row_repeat_b` logic only activates when `block_size` doesn't already equal
`gemm_m_per_block`/`gemm_n_per_block`, which is never true for any existing config).

**Not yet done**: 64x128 (the mirror shape, generalizing B instead of A), bwd/wrw at 128x64,
bf16/int8/fp32 at 128x64, combining `row_repeat` with Phase 13's async-load path, and the fully
general `v_tid`-decoupled mapping (arbitrary, non-power-of-2-multiple tile ratios) that XDLOPS's
cluster dispatcher supports and this `row_repeat` mechanism deliberately does not attempt.

### Phase 11 continued yet again (2026-08-25): mirror shape (64x128, fwd/fp16) via `row_repeat_b`

Ported the `row_repeat` mechanism to B (weight) for the mirror shape:
`gemm_m_per_block=64, gemm_n_per_block=128`, `waves_per_m=1, wave_repeat_m=4, waves_per_n=2,
wave_repeat_n=4, waves=2` -> `block_size=64`. Since `block_size(64) == gemm_m_per_block(64)`
already, **A needs zero changes**; only B needs `row_repeat_b=2`.

**Materially simpler than the 128x64 shape's A-side work, for a structural reason specific to
fwd's B operand**: A (grad_output/input) needs a per-tap *spatial gather* -- each row's
`(n_idx, ho_idx, wo_idx)` decomposition, re-derived every tap via 2 chained divisions, plus
a bounds `v_flag` (a pixel can be out-of-bounds padding). B (weight) has neither: its row is
just `block_n_off + tid + i*block_size` (a K_out channel index, no spatial decomposition at
all) and weight is never out-of-bounds, so no flag/masking logic exists for B in the first
place. This meant the B-side mirror needed no scratch-register-reuse subtlety, no persistent-
vs-fresh-recompute split, and hit no analogous scratch-aliasing bug -- each of B's
`row_repeat_b` rows is simply its own fully independent base address
(`v_addr_b_base(i)`/`v_addr_b(i)`), computed and advanced exactly like row 0, just parameterized
by `i`. Same early-issue-only-for-row-0 discipline as A's rows (row_repeat_a's rows 1+ have no
early-issue slot of their own -- reusing the single small `v_gld_b` buffer for a second
in-flight early load before row 0's own load is stored would hit the exact same scratch-buffer
lifetime bug Phase 1 already found the hard way).

**Process discovery, not a kernel bug, that cost real time**: `igemm_codegen.py -s
config/<name>.config` (the invocation this whole gfx1250 effort had been using) failed to
assemble with `unknown directive .v_u32_div_rem_vs_gfx1250` -- reproduced identically on
completely untouched baseline code. Root cause: `-s`/`--split_kernel` mode deletes the "origin"
`.s` file (where gfx1250's kernel-specific division macros get emitted via
`get_kernel_macros()`) after splitting per-tile-shape files out of it, and the per-split-file
macro re-emission path only re-emits a fixed *generic* (non-gfx1250) macro set, never re-asking
the kernel for its own macros -- so the gfx1250-specific ones are silently lost in split mode,
for every current config, not just this one. **Fix: don't pass `-s`** -- plain `python3
igemm_codegen.py config/<name>.config` assembles correctly. See [[gfx1250-isa-quirks]] for the
full root-cause writeup; flagging here too since it could easily be mistaken for a correctness
regression in future kernel-body work if not known in advance.

**Validated on real hardware**, fwd/fp16/64x128, full 6-case battery (degenerate, multi-K-block,
stride+pad, multi-tap+dilation, group=2, group=4) against `naive_conv_fwd_nhwc` -- all `valid:y`
on the first attempt (no bugs found this time, unlike the A-side mirror). Byte-identical
regression check: all 58 existing gfx1250 configs (including the 128x64 shape) regenerate
identically.

**Not yet done**: bwd/wrw at 128x64 and 64x128, bf16/int8/fp32 at both asymmetric shapes,
combining `row_repeat` with Phase 13's async-load path.

### Phase 11 continued once more (2026-08-25): bwd/wrw asymmetric shapes investigated -- both found materially harder than the fwd port, no new config shipped

User asked to close the remaining coverage gap: port the asymmetric shapes to bwd and wrw.
Ported `row_repeat_a`'s A-side mechanism into `igemm_bwd_gtc_wmma_nhwc.py` (bwd's A/grad_output
is untransposed, structurally close to fwd's A) -- same row-loop pattern, same
fresh-per-row-recompute discipline, PLUS one more precaution the fwd port didn't need: bwd's
per-tap gather computes a shared `pad_h - iy*dilation_h` / `pad_w - ix*dilation_w` scalar pair
into `s_tmp(0)/s_tmp(1)` that's needed by EVERY row's numerator calculation -- but row 0's own
division call also uses `s_tmp(0)` as its internal scratch (the same corruption class Phase 11's
original A-side port found the hard way). Recomputing those two scalars fresh at the top of
EVERY row's iteration (not hoisted once above the loop) sidesteps this without needing to hunt
for a distinct scratch register.

**Found two real, independent blockers before any hardware test was needed, both from static
analysis of the generated code:**

1. **VGPR ceiling**: bwd/fp16's existing 128x128 shape already sits at 256/256 VGPRs (fwd's
   sits at 252) -- bwd's harder divide-based per-tap gather needs a bigger `v_gtc_tmp` scratch
   pool (9 slots vs fwd's 5). `row_repeat_a`'s +3 registers (`v_flag` +1, `v_addr_a` +2) has
   nowhere to land at fwd's exact 128x64 wave_repeat shape (accumulate_a/b/c = 32/32/128). A
   smaller `wave_repeat_n=2` variant (128x32, accumulate_a/b/c = 32/16/64, freeing 80 VGPRs)
   would numerically fit -- but revealed the second blocker below before it got that far.
2. **B's row_local/col_group addressing assumes `block_size == gemm_n_per_block` EXACTLY, not
   just `%==0`**: bwd's B (weight) is TRANSPOSED, addressed via `row_local = tid >>
   col_group_bits`, `col_group = tid & (num_col_groups-1)` where `num_col_groups =
   gemm_n_per_block // gemm_k_per_block`. This decomposition covers exactly
   `num_col_groups * gemm_k_per_block == gemm_n_per_block` distinct (row_local, col_group)
   pairs -- it silently assumes every thread's `row_local` lands in `[0, gemm_k_per_block)`,
   which is only true when `block_size == gemm_n_per_block`. The 128x32 shape being considered
   has `gemm_n_per_block(32) < block_size(64)` -- HALF the threads would compute
   `row_local >= gemm_k_per_block`, an out-of-bounds row feeding directly into
   `v_addr_b_base`'s pointer arithmetic (`row_local * wei_row_c`), corrupting memory outside
   the intended weight tile. Caught by tracing the address formula by hand against the new
   shape's actual thread count, **before** wasting a hardware test on a config that would have
   produced silent wrong answers or a memory fault. This is a DIFFERENT generalization than
   `row_repeat_b` (which handles `gemm_n_per_block > block_size`, i.e. needing MULTIPLE
   rows/thread) -- here it's the opposite mismatch (`gemm_n_per_block < block_size`, i.e. some
   threads owning ZERO valid B rows), and the transposed bit-sliced addressing scheme doesn't
   degrade gracefully to that case the way the untransposed row-per-thread scheme does.

**Decision**: kept the `row_repeat_a` mechanism port in `igemm_bwd_gtc_wmma_nhwc.py` (real,
correct, reusable infrastructure -- byte-identical for `row_repeat_a==1`, confirmed against
every existing bwd config, all precisions spot-checked). Did **not** ship a new bwd asymmetric
config this pass -- neither the natural 128x64 shape (VGPR overflow) nor the smaller 128x32
shape (B-addressing correctness bug) are viable without further work: either free ~3 VGPRs
elsewhere in bwd's already-tight layout, or generalize B's row_local/col_group decomposition to
handle `gemm_n_per_block != block_size` in both directions (an analogous but distinct mechanism
to `row_repeat_a`/`row_repeat_b`, not yet designed).

**wrw was not attempted at all**: per the class docstring, wrw's A (grad_output) AND B (input)
are BOTH transposed (`get_gemm_index_for_src_matrix_transposed`), unlike bwd (A untransposed, B
transposed) or fwd (both untransposed). wrw has no untransposed operand to build a
straightforward `row_repeat_a`-style port on at all -- any asymmetric shape for wrw needs the
SAME transposed-operand generalization bwd's B just exposed as unsolved, for BOTH of its
operands simultaneously. Scoping this out entirely rather than guessing at a partial fix.

**Not yet done**: a working bwd asymmetric shape (needs either a VGPR-savings redesign or the
transposed B-addressing generalization above), any wrw asymmetric shape (needs the same
transposed-operand generalization for both its operands), bf16/int8/fp32 at fwd's existing
128x64/64x128 shapes, combining `row_repeat` with Phase 13's async-load path.

### Phase 11 continued (again) (2026-08-25): bf16/int8/fp32 at fwd's 128x64/64x128 shapes -- mechanical port, all 6 combos hardware-validated

Ported fwd's two landed asymmetric shapes (128x64, 64x128) to bf16/int8/fp32 -- purely
config/table work, no kernel-file changes needed, since `row_repeat_a`/`row_repeat_b` were
already precision-generic (built entirely on `self.data_byte`, matching Phase 8's fp32-port
precedent). Added 6 new `ctrl_wmma_mapping_t` table entries (one per precision x shape) and 6
new config files, mirroring the fp16 entries' `wave_repeat`/`waves` numbers exactly (only
`gemm_k_per_block` and the resulting `tensor_a/b_thread_lengths[1]` differ per precision: 32 for
bf16, 64 for int8, 4 for fp32 -- same per-precision K as their existing 128x128/64x64 entries).

**One real process snag, not a kernel bug**: bf16's conv_driver.exe CLI mode string is
`convbfp16`, not `convbf16` (`conv_driver.cpp` line ~684) -- the natural guess (matching the
`bf16` precision string used everywhere else in this codebase) produces a silent-looking
"Invalid Base Input Argument" with no further detail. Worth remembering for any future bf16
hardware test on this driver.

**All 6 combos (bf16/int8/fp32 x 128x64/64x128) hardware-validated** against
`naive_conv_fwd_nhwc` (degenerate, stride+pad, group>1 -- a 3-case subset of the full battery,
sufficient given the row_repeat mechanism itself was already proven correct via fwd/fp16's full
6-case runs and this port only varies `data_byte`/`gemm_k_per_block`, already proven
precision-generic by Phase 8's fp32 port and the earlier 64x64 bf16/int8/fp32 ports) -- all
`valid:y`, no bugs found this time. Byte-identical regression check: all 14 pre-existing
gfx1250 configs (12 base + fwd/fp16's 2 asymmetric shapes) regenerate identically.

**Not yet done**: bwd/wrw asymmetric shapes at all (see previous section's findings), combining
`row_repeat` with Phase 13's async-load path.

## Phase 15 (2026-08-25): main-loop chunk/compute interleaving -- implemented, hardware-validated, correct but a real regression, KEPT per explicit user instruction

Phase 2's own conclusion identified the actual fix needed for fwd's Phase 1 k-sub-loop
regression: only chunk 0's global load ever overlaps with compute (issued before the barrier);
chunks 1..num_k_chunks-1 (== num_k_substeps, by construction) were loaded+waited+stored
sequentially, all AFTER every substep's compute already finished, so their global-load latency
was never hidden behind anything. This phase implements that fix.

### Design

New `ctrl_wmma_main_loop_t.interleave` flag (default False). When set, each substep
`ks in 1..num_k_substeps-1` issues chunk `ks`'s global load, THEN does that substep's
(unrelated, already-in-LDS) shared_load+compute, THEN waits for and stores chunk `ks` --
instead of batching all substeps' compute first and all remaining chunks' load+store after.
Chunk `ks`'s load overlaps with the LDS-read+compute that happens between its issue and its
wait, hiding at least some of its latency. Needed a new single-chunk functor interface
(`global_load_chunk_a/b_functor`, `shared_store_chunk_a/b_functor` -- callable as
`f(chunk_idx)`, reusing the existing `_emit_gld_chunk_load`/`_emit_sst_chunk` primitives
directly) alongside the existing bulk `global_load_a/b_functor`/`shared_store_a/b_functor`
(kept, unchanged, used by the non-interleaved path and the one-time prologue store). New
tunable `main_loop_interleave` (default 0), asserted mutually exclusive with
`async_global_load` (no staging buffer to interleave around) and `row_repeat_a/b > 1` (kept
separate to isolate correctness, same discipline as every other new mechanism this session).

### A real bug found on hardware, not by inspection

First implementation (single-buffered LDS) passed every single-K-block hardware test
(degenerate, stride+pad, multi-tap, group>1) but failed the multi-K-block case (`nrms=0.0355`
vs `0.0082` threshold, `valid:n`) -- the only case exercising the STEADY-STATE loop body across
2+ outer iterations. Root cause: interleaving moves stores much earlier in program order than
the non-interleaved path's "defer every store until every substep's read is done" discipline --
that discipline is what gave every OTHER wave in the workgroup enough of an implicit timing
margin to finish reading the CURRENT tile before this wave starts overwriting the SAME
(single-buffered) LDS region with the NEXT tile's data. With interleaving, a fast wave can store
chunk 0 of the next tile immediately after substep 0's compute, potentially before a slower
wave has finished reading substep 1 of the current tile from the SAME memory range --
corrupting it. **Fix: `main_loop_interleave` now asserts `lds_double_buffer=1`** -- confirmed
on hardware that adding double-buffering (routing interleaved stores to the physically
different "other" buffer) fixes the multi-K-block case outright (`nrms=0.000165`, `valid:y`),
eliminating the cross-wave race regardless of how early any wave's stores happen.

**Validated on real hardware**, fwd/fp16/128x128x64 (k-sub-loop + double-buffer +
interleave), full 6-case battery (degenerate, multi-K-block, stride+pad, multi-tap+dilation,
group=2, group=4) against `naive_conv_fwd_nhwc` -- all `valid:y`. Byte-identical regression
check: all existing configs (including `_k2x`/`_k2x_dbuf`, which share the same main-loop code
path with `interleave` defaulting to False) regenerate identically.

### Benchmark: a real, reproducible regression -- KEPT anyway

Interleaved old/new methodology (learned from Phase 13's mistake): `IGEMM_WARMUP=5
IGEMM_REPEAT=25`, alternating build+run order across 6 rounds, two different problem sizes.
Compared against the `_k2x_dbuf` config (k-sub-loop + double-buffer, no interleaving -- the
closest apples-to-apples baseline, isolating interleaving's own effect from double-buffering's
already-known-neutral one):

- Large (`n=8,c=2048,H=16,W=16,k=2048`): baseline 145.8-149.3 tflops (mean ~147.4) vs
  interleaved 137.0-137.6 tflops (mean ~137.4) -- **~7% regression**, no overlap between the
  two distributions across 6 measurements each.
- Small (`n=4,c=512,H=16,W=16,k=512`): baseline 13.73-13.94 tflops vs interleaved 13.14-13.31
  tflops -- **~4-5% regression**, same pattern.

Root cause not conclusively diagnosed (candidates: the interleaved schedule's per-substep
wait_loadcnt/wait_dscnt pairs may create more total synchronization points than the batched
non-interleaved version even though each one is individually cheaper to satisfy; the small
`num_k_substeps=2` case tested here may not carry enough "next chunk" work to amortize the
new per-substep bookkeeping against; double-buffering's own XOR-toggle overhead may compound
with the new interleaved control flow in a way it didn't with the simple batched version) --
out of scope for this phase, which was implementation + honest measurement, not root-causing a
second-order interaction between two already-landed mechanisms.

**Net**: implementation is fully correct (passes every hardware battery) but is a measured
regression, not the hoped-for fix for fwd's Phase 1 regression. **Per explicit user
instruction, kept anyway** (same rationale as Phase 13's async-load regression): gfx1250 is
pre-production hardware, so the relative cost of this schedule vs the batched one could change
on future silicon revisions; a correct, validated implementation sitting behind a default-off
tunable (`main_loop_interleave=0` for every existing config) costs nothing to keep. The
`_interleave` config is a reference point for future re-benchmarking, not a recommended
setting today.

### Not yet done

Testing at `num_k_substeps > 2` (e.g. `gemm_k_per_block=128`, if/when such a config exists) to
see whether the regression narrows or widens with more chunks to interleave; extending to
bf16/int8/fp32; extending to bwd's A (untransposed, same mechanism should port directly) and
combining with bwd's already-ported `row_repeat_a` (currently mutually exclusive); root-causing
why interleaving regresses rather than wins, if a future session wants to pursue it further.

### Critical files

- `python/operations/wmma_main_loop.py` -- `ctrl.interleave`, `emit_interleaved_substeps()`,
  the new single-chunk functor fields
- `python/igemm/igemm_fwd_gtc_wmma_nhwc.py` -- `global_load_chunk_a/b_functor`,
  `shared_store_chunk_a/b_functor`, the `main_loop_interleave` asserts (mutual exclusion with
  `async_global_load`/`row_repeat`, requires `lds_double_buffer`)
- `python/igemm/igemm_base.py` -- new `main_loop_interleave` tunable (WMMA branch)
- `config/igemm_fwd_gtc_gfx1250_nhwc_fp16_interleave.config` -- new config (k-sub-loop +
  double-buffer + interleave combined)

### Phase 16 (2026-08-25): fold `lds_double_buffer`/`async_global_load`/`main_loop_interleave` into the mangled kernel name

Flagged by the user: none of these three WMMA-only optional-mechanism tunables were folded
into the mangled kernel symbol name (Phase 13's own docstring had already noted this for
`async_global_load`, just never fixed it). Practical consequence: the interleaved and
non-interleaved builds of the SAME tile shape (e.g. 128x128x64 fp16) produced an IDENTICAL
kernel name -- they couldn't coexist as distinct, auto-discoverable kernels in one combined
kernel object/library the way gfx950/942's many named tile variants do for their auto-tuning
search; each variant had to be built as a separate artifact and chosen by the caller.

**Fix**: `igemm_gtc_encode_kernel_name` (both the Python side, `igemm_base.py`, and the C++
driver side, `driver/igemm_gtc_base.h`) now appends `_dbuf`/`_async`/`_interleave` suffixes
when the corresponding tunable is set -- matching the config-file-naming convention already in
use (`_dbuf.config`, `_async.config`, `_interleave.config`). Only added when the tunable is
non-zero, so every existing config (all three at their default 0) gets the exact same name as
before -- byte-identical regression confirmed for all 15 unaffected configs tested (12 base +
2 asymmetric shapes + `_k2x`).

**Two-sided fix, not just Python**: the C++ driver (`conv_driver.cpp`, via `host_driver()`'s
`-DIGEMM_CONFIG_FILE` build-time define) independently RE-PARSES the same `.config` file and
RE-COMPUTES the kernel name at runtime via its own `igemm_gtc_encode_kernel_name` in
`driver/igemm_gtc_base.h`, then looks the kernel up by that computed name via
`hipModuleGetFunction`. Before this phase, `igemm_gtc_tunable_t` (the C++ struct) didn't even
have fields for these three tunables, so the C++ side would keep computing the OLD (un-suffixed)
name while Python's `.s` output now had the NEW (suffixed) symbol -- a name mismatch that would
break `hipModuleGetFunction` outright. Fixed by adding the three fields to the struct, parsing
them from the config section (mirroring the existing `sec.count(...) > 0 ? ... : 0` pattern
used for `vector_store`/`gemm_k_global_split`/etc.), and mirroring the exact same suffix logic
in the C++ encode function. **Any future WMMA tunable that should be name-mangled needs this
same two-sided treatment** -- a Python-only fix would silently break the driver for any config
that sets the tunable.

**Validated on real hardware, end-to-end**: rebuilt and re-ran the `_interleave` config (now
named `..._dbuf_interleave`) through the full degenerate + multi-K-block battery -- both
`valid:y`, confirming `hipModuleGetFunction` correctly resolves the new suffixed name. Spot-
checked `_async` (fwd) the same way. Confirmed via `.globl` inspection that `_dbuf`, `_k2x_dbuf`,
`_async` (fwd and bwd) each now produce a distinct, correctly-suffixed symbol name.

### Critical files (Phase 16)

- `python/igemm/igemm_base.py` -- `igemm_gtc_encode_kernel_name`'s new suffix block
- `driver/igemm_gtc_base.h` -- `igemm_gtc_tunable_t`'s three new fields, their parsing in
  `igemm_gtc_tunable_from_config`, and the mirrored suffix block in
  `igemm_gtc_encode_kernel_name`

## Phase 17 (2026-08-25): `gemm_k_global_split` for wrw -- fixes the 76x-3087x wrw slowdown

`docs/gfx1250_vendor_benchmark_vs_miopen.md` found wrw catastrophically slow on real backbone
shapes: small GEMM_M (K_out) and GEMM_N (C_in*Y*X) with huge GEMM_K (N*Ho*Wo) means several
shapes launch just 1-2 workgroups total on a 256-CU part -- one workgroup serially eats
1000+ main-loop iterations with nothing else to hide latency behind. The mature XDLOPS wrw
path already solves this (`gemm_k_global_split`: split the reduction axis across grid.z,
atomically accumulate partial sums); the WMMA path had zero occurrences of the mechanism.
Ported it to `igemm_wrw_gtc_wmma_nhwc.py` -- fp16/bf16/fp32 (int8 wrw doesn't exist in this
codebase at all).

**Simpler than the XDLOPS mechanism, for two hardware-specific reasons found while building
this**:
1. `global_atomic_add_f32` assembles and runs correctly on gfx1250 in the exact SADDR form
   (`global_atomic_add_f32 v_voffset, v_data, s[saddr:saddr+1]`) the epilogue already uses for
   `global_store_dword` -- no CAS-loop fallback needed (some other archs' `buffer_atomic_add`
   path has one, gated by an `atomic_add_using_cas` assembler switch; gfx1250 doesn't need it).
2. The WMMA output buffer is always allocated fp32 regardless of the tunable's nominal
   precision (`conv_driver.cpp`'s `dtype_alloc_byte = is_wmma ? 4 : data_byte`). So unlike
   XDLOPS's fp16/bf16 path -- which needs a separate fp32 workspace buffer plus a
   `tensor_cast_*` postlog kernel to fold splits back into a real fp16/bf16 tensor -- this
   kernel atomic-adds fp32 straight into the real output for every precision. No workspace,
   no postlog kernel, no separate reduction step at all: the atomic add *is* the reduction.

**Split-count policy** (deliberately simple -- no runtime search over multiple split counts
like XDLOPS's `gks_iterative`): pick the largest `splits` that evenly divides `num_k_blocks =
gemm_k / gemm_k_per_block`, targeting `ceil(num_cu / (grid_x*grid_y))` workgroups. Requiring
`splits` to divide `num_k_blocks` means every split gets an identical, exact multiple of
`gemm_k_per_block` -- the WMMA main loop has no K-tail handling at all, so this preserves that
invariant per split instead of inventing tail-handling. Lives in
`driver/igemm_wrw_gtc_driver.h`'s WMMA `run()` branch, entirely host-side.

### Correction to the Phase 12/13 ttmp workgroup-id finding: `blockIdx.z` is not a separate register

Adding a 3rd grid dimension required knowing where gfx1250 delivers `blockIdx.z`. The earlier
finding (this doc's "Workgroup ID delivery" section) only established x/y (`ttmp9`/`ttmp7`)
since nothing before this phase ever launched a 3D grid. **`blockIdx.z` is not a separate ttmp
register: `ttmp7`'s low 16 bits are `blockIdx.y` and its high 16 bits are `blockIdx.z`.**
`ttmp9` remains a clean, unpacked `blockIdx.x`.

Found by writing a HIP kernel that reads real `blockIdx.x/y/z` (compiler ground truth) plus six
raw `ttmp4..ttmp9` registers via inline `asm volatile("s_mov_b32 %0, ttmpN\n" : "=s"(t))`, in
the SAME kernel, launched with a real 3D grid (`dim3(2,3,4)`) via `hipLaunchKernelGGL` on real
hardware. Direct correlation, no disassembly-reading required: `ttmp7`'s value was exactly
`blockIdx.z * 65536 + blockIdx.y` for every dispatched workgroup (e.g. z=2,y=1 -> ttmp7 =
0x20001). An initial guess that a *different* ttmp register (`ttmp6`, read alongside `ttmp7`
during the same disassembly inspection that found `ttmp9`/`ttmp7`) held `blockIdx.z` directly
was tested this same way and **disproven** -- its values didn't correlate with any dispatched
grid coordinate at all (likely some other wave-launch state); always verify a register-mapping
guess against compiler ground truth before shipping it, not just "it assembles and doesn't
crash."

**Every existing WMMA kernel's `s_mov_b32 s_by, ttmp7` was never wrong** -- it's just been
implicitly relying on grid.z always being 1 (so the upper 16 bits are 0) for every kernel that
existed before this phase. Only `igemm_wrw_gtc_wmma_nhwc_t`'s split variant now decodes it
correctly: `s_and_b32 s_by, ttmp7, 0xffff` / `s_lshr_b32 s_bz, ttmp7, 16`. fwd/bwd are
unmodified (they never launch a 3rd grid dimension) but this is a landmine for whoever adds one
there next -- **grep for `s_mov_b32 s[{s.s_by()}], ttmp7` before doing so.**

### A hardware correctness pitfall found via this port: atomic scope

`global_atomic_add_f32` **without an explicit scope modifier silently drops updates across
compute units** on this hardware/driver. Found the hard way: the 64x64 tile config (more
workgroups resident per CU, smaller footprint) passed correctness immediately; the 128x128
tile config (same math, same K-split, larger footprint -> more likely to schedule the two
accumulating workgroups on *different* CUs) failed with wrong-but-plausible values -- not
garbage, not a clean 2x/0x pattern, just wrong partial sums (including at least one sign flip
against the reference), consistent with a lost update rather than an addressing bug.

Ruled out by direct measurement, in order, before finding the real cause: the K-range
split arithmetic itself (verified via a temporary debug buffer dumping each workgroup's
decoded `bz`/`gemm_k_wg_off` back to the host -- exactly correct in every case, including the
failing ones); a queue-depth/back-to-back-issue hazard on the atomic instruction (added a full
`s_wait_storecnt 0x0` between every single atomic -- no change); the epilogue's `offset:`
immediate specifically for atomics (replaced with explicit address adds -- no change). Adding
`scope:SCOPE_SYS` to the atomic instruction fixed it immediately and reproducibly. Since gfx12
introduced scope-qualified cache/memory instructions as part of its updated cache hierarchy
(also seen on `global_prefetch_b8 ... scope:SCOPE_SE` in compiler-generated code during the
Phase 12 ttmp probe), the likely explanation is that a bare atomic defaults to a narrower
(CU/WGP-local) scope that only guarantees atomicity for lanes/waves sharing that scope, not
across the full device -- gfx1250's atomic-add mnemonic needs the scope stated explicitly for
cross-CU correctness; **this is the load-bearing detail, not a style choice** -- do not drop it
if this code is ever refactored.

### Verification

Correctness (`valid:y`) confirmed on real hardware for: a single K-block (`splits` trivially
1, isolating the atomic+zero-init machinery from real multi-workgroup summation), an exact
2-K-block/2-way split on both tile shapes, group>1, and the vendor-benchmark doc's actual
failing shapes at up to 300-way splits, for fp16/bf16/fp32. Non-split configs (all directions,
all dtypes, all tile variants exercised earlier this session) re-verified unaffected -- the
epilogue's `global_atomic_add_f32`/`scope:SCOPE_SYS` path and the `s_by`/`s_bz` ttmp7 split are
both compile-time-gated on `tunable.gemm_k_global_split`, byte-identical generated code
otherwise.

A per-iteration correctness pitfall specific to benchmarking accumulator kernels, found and
fixed during this work: `driver/igemm_wrw_gtc_driver.h`'s WMMA `run()` originally zeroed the
output buffer *once*, before the warmup+repeat timing loop, rather than once per dispatch. Since
atomics accumulate rather than overwrite, the 2nd+ timed iteration's adds landed on top of the
1st iteration's already-correct result, corrupting the final readback despite every individual
dispatch being correct in isolation. Fixed by moving the zero-init into the `prolog_kernel`
callback `igemm_launch_kernels` already runs before every single dispatch (mirroring how the
mature XDLOPS path's `wrw_prolog` does the same). Any future accumulator-style WMMA kernel
needs the same per-iteration (not per-`run()`-call) zero-init.

### Results

Fixes the vendor-benchmark doc's headline wrw regression. Re-ran that doc's exact 10 failing
bf16 shapes (batch=42) through the `_gsplit` config: worst case improved from 3087x slower than
MIOpen to 29x slower; most shapes landed within 2-6x of MIOpen (previously 76x-1525x). See
`docs/gfx1250_vendor_benchmark_vs_miopen.md`'s updated results table for the full per-shape
breakdown -- same GPU-contention caveat as every other number in that doc applies here too.

### Critical files (Phase 17)

- `python/igemm/igemm_wrw_gtc_wmma_nhwc.py` -- `s_bz`/`s_gemm_k_per_wg`/`s_gemm_k_wg_off`
  sgprs, the fixed `s_by`/`s_bz` ttmp7 decode, the K-slice offset folded into A's base address
  and B's per-iteration gather, `s_knum` bound to the per-workgroup slice length
- `python/operations/coalescing_store_wmma.py` -- `ctrl.gemm_k_global_split` gating
  `global_atomic_add_f32 ... scope:SCOPE_SYS` vs `global_store_dword`
- `driver/igemm_wrw_gtc_driver.h` -- split-count policy, grid.z, the `gemm_k_per_wg` karg
  field, and the per-iteration zero-init prolog
- `config/igemm_wrw_gtc_gfx1250_nhwc_{bf16,fp16,fp32}_gsplit.config` -- the new configs

## Phase 18 (2026-08-25): split-count runtime search -- the single-heuristic policy left real perf on the table

Phase 17 picked one split count per problem (`largest divisor of num_k_blocks <= ceil(num_cu /
tile_count)`) and trusted it. Timing that heuristic's actual choice against nearby alternatives
showed it was often well off the fastest option -- e.g. one shape's heuristic pick (splits=300)
ran at 2.27ms while a smaller pick (splits=200) ran at 0.376ms for the *same problem*, a ~6x
difference from split count alone. Too many splits means the fixed per-workgroup cost (kernel
epilogue, atomic RMW round-trips to L2) starts dominating a shrinking slice of real work per
workgroup; the single-heuristic target (aimed at "one workgroup per CU") doesn't account for
that at all.

Fix: mirror the mature XDLOPS path's `gks_iterative` in spirit, but bounded to 3 candidates
instead of a full sweep, since the WMMA main loop's no-K-tail constraint already limits the
useful candidate set to divisors of `num_k_blocks`. `driver/igemm_wrw_gtc_driver.h`'s WMMA
`run()` now builds 3 candidate split counts -- `largest_divisor_leq(num_k_blocks, t)` for `t` in
`{target, target/2, target*2}`, deduplicated -- actually launches and times each one (with the
existing per-iteration zero-init prolog), and keeps whichever ran fastest. This roughly triples
the wall-clock cost of a single `run()` call for `gemm_k_global_split` tunables (3 real
dispatches instead of 1) -- acceptable for `driver_mode_normal`'s tuning/benchmarking use case,
same tradeoff the XDLOPS path already accepts for its own gks search.

`result.gks` now reports the *actually selected* split count (previously always 0), surfaced by
`conv_driver.cpp`'s `_gkgs[N]` display convention -- useful for spot-checking that the search
picked something sensible for a given shape.

### Results

Re-ran the same 10 vendor-benchmark wrw shapes with the search in place (see
`docs/gfx1250_vendor_benchmark_vs_miopen.md`'s update for the full table). Several shapes
improved 1.6x-2x over the single-heuristic result (e.g. `192,60,80,64,1x1`: 0.127ms -> 0.074ms;
`64,60,80,256,1x1`: 0.148ms -> 0.087ms); none regressed outside normal run-to-run noise. Worst
case vs MIOpen improved from ~23-29x to ~13x slower.

Three candidates is a small bracket, not a real search -- it can still miss a better split count
between/beyond the tried values. A wider or adaptive search (binary-search toward the minimum,
or trying every divisor within some CU-count-relative band) is the natural next step if more
of this gap needs closing, at the cost of more `run()` wall-clock time per tuning pass.

### Critical files (Phase 18)

- `driver/igemm_wrw_gtc_driver.h` -- `largest_divisor_leq`, the `gsplit_candidates` list, and
  the per-candidate launch-and-keep-best loop in the WMMA `run()` branch

## Phase 19 (2026-08-25): epilogue address double-buffering -- correctness-neutral, no measured perf change

Investigated "vectorize/batch the atomic epilogue" as a further lever after Phase 18. First
checked the ISA directly (`amd-instinct-cdna5-instruction-set-architecture.md`, this session's
first use of it): gfx1250/CDNA5 has **no wide/packed fp32 atomic-add** -- only single-dword
`GLOBAL_ATOMIC_ADD_F32`, plus `GLOBAL_ATOMIC_PK_ADD_F16`/`PK_ADD_BF16` (packed 2x16-bit), which
don't apply since the WMMA accumulator (and therefore every atomic-add here) is always fp32.
So literal vectorization -- fewer, wider atomic instructions -- isn't available; each output
element needs its own atomic no matter what.

What *is* available: `coalescing_store_wmma.py`'s address computation reused a single VGPR
(`v_tmp1`) across the whole epilogue, so advancing to the next row (`v_tmp1 += row_stride`) had
a WAR dependency on every store/atomic that had just read the old value -- serializing all
`wave_repeat_m * num_v_c` row addresses through one register. Changed to a 2-register ping-pong
(`v_tmp1`/`v_tmp2`): each row's address is precomputed into the register the *current* row's
stores/atomics are NOT reading, so the address-advance and the in-flight stores/atomics touch
different registers and have no data dependency on each other. Costs one extra VGPR
(`v_addr_out` back to 2 registers, same as before the Phase 1 epilogue rewrite that shrank it
to 1).

**Result: no measurable difference** across all 10 vendor-benchmark shapes (within normal
run-to-run noise both directions) -- see `docs/gfx1250_vendor_benchmark_vs_miopen.md`. Kept the
change anyway since it's correctness-neutral (re-verified `valid:y` for fwd/bwd/non-split-wrw/
gsplit-wrw) and removes a real (if apparently already-hidden) hazard, but the honest conclusion
is that address-computation overhead was never the bottleneck here -- the atomic RMW round-trip
latency (L2 and back) almost certainly dominates the epilogue's cost regardless of how cheaply
the handful of surrounding VALU instructions are scheduled. Confirms the split-count search
(Phase 18) and, likely, an LDS-reshuffle epilogue for the *non-atomic* store path (fwd/bwd/
non-split wrw, not attempted here) are better uses of further effort than further epilogue
micro-optimization on the atomic path specifically.

### Critical files (Phase 19)

- `python/operations/coalescing_store_wmma.py` -- the `v_tmp1`/`v_tmp2` ping-pong
- `python/igemm/igemm_{fwd,bwd,wrw}_gtc_wmma_nhwc.py` -- `v_addr_out` back to 2 registers,
  call sites pass both

## Phase 20 (2026-08-25): ternary search over ALL divisors -- Phase 18's 3-candidate bracket characterized properly and replaced

Pushed the split-count question further: rather than guessing at a better bracket, added a
research-only `IGEMM_GSPLIT_SWEEP=<target>` env var to `driver/igemm_wrw_gtc_driver.h` (forces
a single candidate, snapped to the nearest valid divisor of that target, bypassing whatever
search logic is otherwise in place) and used it to sweep the **entire** perf-vs-split-count
curve from an external shell loop, no rebuild needed between points.

### The curve, measured on real hardware

Swept every divisor of `num_k_blocks` for all three distinct values that occur across the 10
vendor-benchmark shapes (25200, 6300, 1575 -- batch=42), `IGEMM_WARMUP=2 IGEMM_REPEAT=5` for
speed (this is a shape-characterization pass, not a final measurement). All three curves are
clearly **unimodal**: cost decreases monotonically from splits=1 (matches the original
catastrophe -- 777ms at splits=1 for the worst shape) down to a single minimum, then increases
again as splits keeps growing (fixed per-workgroup/per-atomic overhead starts dominating an
ever-shrinking slice of real work). Sample points for the worst shape
(`128,120,160,128,3x3`, `num_k_blocks=25200`, 128x128 tile):

| splits | cost (ms) | | splits | cost (ms) | | splits | cost (ms) |
|---|---|---|---|---|---|---|---|
| 1 | 777.4 | | 300 | 2.94 | | 600 | **2.14** |
| 10 | 71.5 | | 420 | 2.34 | | 630 | **2.06** (best) |
| 100 | 7.51 | | 450 | 2.31 | | 700 | 2.11 |
| 200 | 3.99 | | 504 | 2.33 | | 900 | 2.18 |
| 252 | 3.30 | | 525 | 2.43 | | 1575 | 2.82 |

The critical finding: **the minimum's location relative to `ceil(num_cu / tile_count)` (512
here) is not consistent across shapes.** For this shape the true optimum (630) sits *above* the
naive target; for `128,30,40,128,1x1` (`num_k_blocks=1575`, target also 512 since
`grid_x*grid_y=1`) the true optimum was 225-315, well *below* the target. A fixed-offset
bracket (Phase 18's `{target, target/2, target*2}`) structurally cannot reliably straddle a
minimum that moves in different directions for different shapes -- it happened to land close
sometimes and 6x off other times.

### The fix: ternary search over the full sorted divisor list

Since the curve is unimodal, a ternary search finds its minimum in O(log(divisor count)) real
timed launches instead of guessing. `driver/igemm_wrw_gtc_driver.h`'s WMMA `run()` now:
1. Enumerates every divisor of `num_k_blocks` (trial division up to sqrt, trivial cost --
   divisor counts here ran 18-90, but this scales fine well beyond that).
2. Ternary-searches the sorted list: at each step, times the two points at 1/3 and 2/3 of the
   current window (`time_split(splits)` -- builds the karg for that split count and actually
   launches+times it via `igemm_launch_kernels`, same per-iteration zero-init prolog as
   before) and discards the third of the window on the losing side.
3. Once the window shrinks to <=5 candidates, evaluates all of them directly and keeps the
   true minimum -- real hardware timings are noisy, so trusting the last ternary comparison
   alone risked landing one index off a locally-flat minimum.
4. Caches every evaluated index (`std::vector<float> cache`, sentinel -1) so no split count is
   ever timed twice across the narrowing loop and the final confirm.

Degenerates correctly for non-split tunables: `divisors = {1}` when `gemm_k_global_split` is
off, so the ternary loop's window is already `[0,0]` and the final confirm evaluates exactly
one candidate (splits=1) -- same code path, no special-casing needed.

### Results

Re-ran all 10 vendor-benchmark shapes a fourth time (full warmup=5/repeat=20 fidelity, not the
fast sweep settings above) -- see `docs/gfx1250_vendor_benchmark_vs_miopen.md`'s update for the
complete table. Selected split counts landed within a few percent of the true minimum found by
the earlier exhaustive sweep in every case checked. Two shapes improved ~23-24% further over
Phase 18's 3-candidate result (`256,60,80,64,1x1`: 0.086ms -> 0.065ms; `64,60,80,256,1x1`:
0.087ms -> 0.067ms) with the rest flat to modestly better; none regressed. Worst case vs MIOpen
improved from ~13x (Phase 18) to ~11.6x.

Cost: this now runs O(log(divisor count)) real dispatches per `run()` call for a
`gemm_k_global_split` tunable instead of 3 -- roughly 15-20 for the divisor counts seen here,
each `warmup+repeat` dispatches. Only matters for `driver_mode_normal`'s tuning/benchmarking
use; a production deployment would run the search once per shape and cache the winning split
count, same as any other autotuned kernel parameter.

**Still not the theoretical ceiling** -- a full exhaustive sweep would occasionally find a
point 1-2% better than the ternary search's answer (visible in the sample table above: 630 vs.
the search's typical picks of 600-700), and the remaining gap to MIOpen beyond this is
architectural (see Phase 19's conclusion: an LDS-reshuffle coalescing epilogue for the
non-atomic store path is the next lever, not attempted in this phase -- deliberately deferred,
see that doc's "revisit later" note).

### Critical files (Phase 20)

- `driver/igemm_wrw_gtc_driver.h` -- full divisor enumeration, `time_split` lambda, the
  ternary search + cached final confirm, and the `IGEMM_GSPLIT_SWEEP` research override

## Phase 21 (2026-08-25): LDS-reshuffle coalescing store for the non-atomic epilogue path

Implemented the deliberately-deferred item from Phase 19: fwd, bwd, and non-split wrw all
still used the direct scalar epilogue (one `global_store_dword` per accumulator element,
128 for a 128x128 tile) because a single WMMA lane only ever owns one output column (`col =
lane % 16`, fixed across every accumulator index -- confirmed against
`wmma_mapping.py:195-198`'s `get_gemm_index_for_dst_matrix`), so no per-lane vectorized store
is directly possible. The mature XDLOPS epilogue (`coalescing_store.py`) solves the same
class of problem via an LDS reshuffle; ported the equivalent for WMMA.

### Design (only the `vector_write_out != 1` case applies)

Read `coalescing_store.py`'s actual addressing code (`init_co_lds_offset`) rather than
working from a summary, since Phase 12/14's tr16 investigation already showed how costly a
wrong lane-mapping guess is here. Two address schemes coexist in that file: a granularity
trick (`vector_write_out == 1`) that bounces AGPR data through LDS efficiently but does NOT
enable a vectorized global store, and a much simpler scheme (`vector_write_out != 1`, what
was ported) that scatters in **true tile-linear order** (`(row * macro_tile_n + col) *
data_byte`) and gathers via a **flat tid-indexed** read (`tid * vector_write_out *
data_byte`). WMMA needed the second one -- no granularity trick, since WMMA has no
AGPR-vs-VGPR distinction to begin with.

Single-pass, full-tile LDS (no coalescing-group loop): the D-operand output tile is always
4 bytes/element regardless of nominal precision (existing behavior), so total tile size is
65536 bytes for every 128x128 config (any precision) and 16384 for every 64x64 config.
65536 is exactly gfx1250's 64KB per-workgroup LDS limit with zero headroom -- verified this
actually loads and runs *before* implementing the rest (`hipModuleLoad`/
`hipModuleGetFunction` succeed with a kernel declaring exactly 64KB). This avoids the
multi-group LDS partitioning XDLOPS needs, which turned out to have a real complication:
compile-time-loop-index grouping doesn't give a compact address range once
`waves_per_m/waves_per_n > 1`, since different waves' row contributions land in
non-adjacent bands within any single group -- a correct multi-group version would need
XDLOPS's own "decompose and reassemble the M-index" dance. Deferred; single-pass fits every
config that exists today.

Scatter is scalar (`ds_write_b32`, one per element, reusing the exact same `(i_rm,i_rn,j)`
loop and `v_gemm_im`/`v_gemm_in` address derivation the direct-store path already used --
just retargeted to LDS via shifts instead of a stride multiply, which is cheaper since
`macro_tile_n` is a compile-time power of 2). Gather batches `vector_write_out=4` contiguous
elements per thread per pass via `ds_read_b128` + `global_store_dwordx4`: `128 -> 32` store
instructions for a 128x128 tile, confirmed by instruction count in the generated `.inc`.

### Two real bugs found and fixed before this worked, both worth recording

1. **`v_gemm_im`/`v_gemm_in` are GLOBAL (gemm-space) positions, not tile-local.** Every
   kernel adds `s_block_m_off`/`s_block_n_off` to them once, persistently, right after
   `get_gemm_index_for_dst_matrix` (e.g. `igemm_fwd_gtc_wmma_nhwc.py:683-684`) -- correct
   for the direct-store path (which needs the global address anyway) but silently wrong for
   an LDS address, which must be tile-local. First attempt produced values that were
   essentially zero everywhere (`pred≈0.0001` against real reference values) -- reading from
   the wrong, effectively out-of-tile-range LDS offset for every shape tested, including the
   trivial single-tile case, which ruled out an edge-case bug and pointed at something
   structurally wrong in the address itself. Fixed by masking: since `block_m_off`/`n_off`
   are always exact multiples of `macro_tile_m`/`n`, `& (macro_tile_m/n - 1)` strips exactly
   the block-level high bits and leaves the tile-local component unchanged, for both the
   scatter's LDS address (needs masking) and the gather's global memory address (needs the
   block offset added *back*, since the gather's `tid`-derived row/col are pure tile-local
   and never had it).
2. **Returning inside a `with self._deferred_context():` block reads `deferred_buffer`
   before `__exit__` populates it.** `_get_deferred()` just returns
   `self.outter.deferred_buffer`, which `deferred_context_t.__exit__` sets -- so a `return
   self._get_deferred()` *inside* the `with` block (as the atomic-path branch had, before
   this was caught) reads whatever `deferred_buffer` held from the *previous* call, not the
   current one. Went unnoticed at first because the reshuffle (non-atomic) path's `return`
   was already correctly placed after the `with` block exits, and every non-atomic
   correctness test passed -- it was the **atomic** (`gemm_k_global_split`) path's own
   already-existing early return, exposed only when re-testing it after this session's
   refactor. Regression testing the atomic path (not just the new non-atomic path) caught
   it immediately: previously-passing shapes started reporting `valid:n` with the exact
   symptom of "the epilogue emitted nothing" (visible directly in the generated `.inc` --
   the coalescing_store call site's surrounding code was there, the epilogue's own content
   was entirely absent). Fixed by restructuring both branches under one `if/else` inside the
   `with` block, with a single `return self._get_deferred()` after it exits. **Lesson**: any
   time a function has multiple return points inside a deferred-emit context manager, check
   every one is actually outside the `with` block, not just the one you're actively editing.

### Results

Instruction count (128x128 fp16, confirmed via `grep -c` on the generated `.inc`, same
technique as the Task-1 epilogue rewrite): 128 `global_store_dword` -> 32
`global_store_dwordx4`, exactly the predicted 4x.

Wall-clock, fp16, same GPU-contention caveat as everywhere else in this doc:

| Shape | Direction | Before (ms) | After (ms) | Change |
|---|---|---|---|---|
| 128,120,160,128,3x3 (n=42) | fwd 128x128/64x64 | 0.369 / 0.502 | 0.362 / 0.480 | ~2-4% faster |
| 64,60,80,256,1x1 (n=42) | fwd 128x128/64x64 | 0.041 / 0.046 | 0.037 / 0.043 | ~7-10% faster |
| 192,60,80,64,1x1 (n=42) | fwd 64x64 only | 0.027 | 0.030 | ~11% slower (noise -- ~0.03ms absolute) |
| n=8,c=32,H=W=128,k=128 (single K-block, epilogue-dominated) | fwd 128x128/64x64 | 0.014 / 0.015 | 0.012 / 0.014 | 7-14% faster |
| 128,120,160,128,3x3 (n=42) | bwd 128x128/64x64 | 0.586 / 0.773 | 0.587 / 0.770 | flat (noise) |
| 128,30,40,128,1x1 (n=42) | non-split wrw 128x128/64x64 | 5.377 / 4.351 | 5.312 / 4.294 | ~1% faster |

**Honest read, not oversold**: gains are modest (0-14%) and scale with how much of total
kernel time the epilogue actually represents. fwd/bwd are compute-bound for large-K shapes
(main loop dominates, epilogue is a small slice regardless of how much cheaper it gets) --
the single-K-block case shows the clearest win because the epilogue is a much larger
fraction of total time there. Non-split wrw shows almost nothing, for the same reason Phase
17 fixed with a *different* mechanism (K-split): wrw's occupancy problem is about too few
*workgroups*, which this change doesn't touch at all -- it only makes each workgroup's own
epilogue cheaper, and non-split wrw was never epilogue-bound to begin with (a single,
massive main loop dominates it completely). This is a real, confirmed-correct optimization
with a clear mechanism (4x fewer, wider global stores + cheaper on-chip LDS traffic instead
of direct global memory ops), it just isn't the shape of problem this codebase's wrw
regression was.

### Critical files (Phase 21)

- `python/operations/coalescing_store_wmma.py` -- the new reshuffle branch (scatter, dual
  barrier, gather+vectorized store), restructured `if/else` to fix the deferred-context bug
- `python/igemm/igemm_{fwd,bwd,wrw}_gtc_wmma_nhwc.py` -- `get_kernel_code()`'s LDS
  declaration bumped to `max(main-loop LDS, full output tile)`, call sites pass
  `s_block_m_off`/`s_block_n_off` and reuse `v_c` as gather scratch (always 4-aligned, dead
  by gather time -- no new VGPR allocation needed)

## Phase 22: VGPR-level (register) prefetch for v_a/v_b, then a VGPR-budget audit

Next item after the rocke/CK research turn (wavelet pipeline, OOB-check-with-fewer-
registers): the dotx/mfma paths already do `local_prefetch_num=2` -- intra-K-substep
software pipelining of the LDS->VGPR read -- but WMMA never has
(`igemm_base.py` unconditionally pinned it to 1, "single-buffered main loop for this
milestone, no local prefetch"). This phase ports the mechanism, applies it wherever the
VGPR budget allows today, and treats the configs where it doesn't fit as the entry point
into a VGPR-budget audit (the "look at #3" half of the same request).

### Design

Mirrors mfma_main_loop.py's local_prefetch_num=2 exactly, one level down (LDS-read instead
of global-load -- Phase 15's interleave already covers the global-load layer). `v_a`/`v_b`
now hold up to 2 disjoint slots back-to-back (slot 1 starts at VGPR offset
`wave_repeat_m/n * inst_wmma.num_v_a/b`). At each k-substep `ks`, the NEXT substep's
shared_load is issued into the other slot BEFORE the CURRENT substep's `v_wmma_*` consumes
the slot it already holds, so the LDS-read latency for `ks+1` overlaps `ks`'s WMMA issue
instead of blocking on it. Slot selection is plain compile-time `% 2` arithmetic (no
runtime toggle register), same style as dotx/mfma. Only meaningful when
`gemm_k_per_block > inst_wmma.k` (Phase 1's k-sub-loop in use, i.e. `num_k_substeps > 1`)
-- `wmma_main_loop.py`'s `emit()` asserts this at codegen time. Mutually exclusive with
Phase 15's `main_loop_interleave` for this first implementation (both rewrite the same
k-substep drain loop; composing them isn't validated).

**Correctness note on 2-slot reuse**: reusing slot `(ks-1)%2` for substep `ks+1`'s load
needs no extra wait beyond the one already emitted for it -- the `v_wmma_*` instruction
reading that slot for substep `ks-1` was already ISSUED (in program order) before the new
`ds_read` targeting the same registers is issued, and GCN/RDNA's in-order-per-wave issue
model guarantees a VALU/WMMA register READ happens before a later same-register LDS WRITE
completes, regardless of the LDS read's own completion latency. This is the same guarantee
mfma_main_loop.py's own `local_prefetch_num=2` already relies on -- not a new assumption.

**VGPR formula**: `num_vgpr_accumulate_a/b = wmma_repeat_m/n * inst_wmma.num_v_a/b *
local_prefetch_num` (mirrors XDLOPS's formula exactly; WMMA previously had no
`local_prefetch_num` multiplier at all). Config-driven via a new `local_prefetch_num` key
(default 1, byte-identical for every existing config), read in `igemm_base.py`'s WMMA
branch -- **not** where `async_global_load`/`main_loop_interleave` are read a few lines
earlier, because this `__init__` has a later, shared `self.local_prefetch_num = 1` default
(applies to every `fma_type`, runs in between) that would otherwise clobber an earlier
read. Caught exactly this way during implementation: the first version read the config key
in the "early" WMMA field block, and the codegen for a `local_prefetch_num=2` config
silently produced byte-identical output to the `local_prefetch_num=1` case -- no error, no
crash, just the flag having zero effect. Only noticed because `v_end`/`.vgpr_count` was
checked directly in the generated `.s`/`.inc` (180 both before and after, when +16 was
expected) rather than assuming a clean compile meant the feature worked. **Lesson**: for a
`utility_dict_with_default_t(tunable_dict)(...)` read to actually stick, it must be the
LAST write to that field before it's consumed, not merely present somewhere in `__init__`.

The 6 `shared_load_a/b_functor` implementations (fwd/bwd/wrw x A/B) needed a `slot=0`
kwarg added to their `__call__` and every `v.v_a(...)`/`v.v_b(...)` destination index
offset by `slot * num_v_a/b_total` -- discovered these functors **ignore their `v_dst`
call argument entirely** (`wmma_main_loop.py` passes one, but ignores it internally, always
addressing `outer.vgpr.v_a`/`v_b` directly), so simply passing a different value from the
main loop's call site would have had no effect; the destination addressing logic lives
inside each kernel file's functor, not in the shared main-loop driver.

### VGPR-budget audit (compile-only, no GPU launch -- see Verification below)

Doubling costs `wmma_repeat_m*num_v_a + wmma_repeat_n*num_v_b` VGPRs total. Confirmed via
actual generated `.vgpr_count`/`v_end` (not just the formula) for the 128x128, fp32,
k-sub-loop (`_k2x`) config: 180 -> 196 (+16), exactly as predicted (fp32's `num_v_a=
num_v_b=2`). Landed as `config/igemm_{fwd,bwd,wrw}_gtc_gfx1250_nhwc_fp32_k2x_lp2.config`.

For fp16/bf16/int8 at 128x128 (`num_v_a=num_v_b=8`), the existing `_k2x` config already
compiles to **252/256 VGPRs** (confirmed directly, all three precisions, fwd) -- doubling
would need +64, landing at 316, wildly over the 256 limit. A full breakdown of those 252
(`igemm_fwd_gtc_gfx1250_nhwc_fp16_k2x_128x128x064.inc`):

| Pool | VGPRs | Notes |
|---|---|---|
| `v_c` | 128 | accumulate tile, fixed by `gemm_m_per_block*gemm_n_per_block/block_size` |
| `v_a` | 32 | `wave_repeat_m(4) * num_v_a(8)` |
| `v_b` | 32 | `wave_repeat_n(4) * num_v_b(8)` |
| `v_gld_a` + `v_gld_b` | 16 + 16 | old-path global-load staging buffers (chunked) |
| everything else (addr/offset/scratch/epilogue) | 28 | `v_tid`, `v_addr_a/b`, `v_addr_out`, `v_sst/sld_os`, `v_gemm_im/in`, `v_tmp`, `v_flag`, `v_n/ho/wo_idx`, `v_gtc_tmp` |

`v_c` is fixed (it's the actual output tile). The one real lever: `v_gld_a`/`v_gld_b` (32
total) only exist because this config uses the OLD global-load path; Phase 13's
`async_global_load=1` replaces them with a persistent 4-VGPR `v_zero` quad (global load
writes straight to LDS, no staging), reclaiming roughly 28 net VGPRs. That's real but not
enough on its own (252 - 28 + 64 = 288, still 32 over). Combining that reclaim with an
**asymmetric** prefetch (only double one of `v_a`/`v_b`, +32 instead of +64, mirroring how
dotx's `local_prefetch_num_m`/`local_prefetch_num` are already independent per-axis knobs)
lands at 252 - 28 + 32 = 256 -- exactly at the limit, zero margin. That's too marginal to
ship blind: a 1-VGPR miscalculation anywhere (or a slightly different config's scratch
needs) overflows it. **Conclusion: no clean fit exists yet for 128x128 fp16/bf16/int8** --
either shrink the "everything else" scratch pool further (28 VGPRs across ~10 small
allocations, no single obvious cut) or accept async_global_load as a hard prerequisite and
still be at zero margin. Not attempted further this phase -- flagging rather than forcing a
marginal, unvalidated change.

The 64x64 tile shape has real headroom (172/256 base, fp16/bf16/int8) but has no existing
k-sub-loop (`_k2x`-style) config to extend -- would need a brand-new
`tensor_a/b_thread_lengths`/`cluster_lengths` derivation for `gemm_k_per_block>16` at that
tile shape, which is exactly the class of address-math derivation this branch has gotten
subtly wrong before (Phase 21's two bugs). Deferred rather than guessed at without the
ability to validate on real hardware in this pass.

### Verification

Compile-only checks first (codegen + `clang++ -x assembler` -> `.hsaco`, no kernel launch):
fp32 128x128 `_k2x_lp2` for fwd/bwd/wrw assembles cleanly, `.vgpr_count` matches the
predicted +16 exactly, and the emitted schedule was inspected directly in the `.inc` (the
`ds_read`/`v_wmma_*`/`s_wait_dscnt` sequence matches the intended 2-slot pipelining exactly:
substep 0's data already in hand, substep 1's load issued and waited before substep 0
computes, cycling slots 0/1 thereafter, no dangling prefetch after the final substep).

**Regression sweep, real hardware**: rather than re-running every existing config by hand,
generated 11 representative existing configs (fwd/bwd/wrw, fp16/fp32, covering
`_k2x`/`_dbuf`/`_interleave`/`_async`/the atomic `_gsplit` path) at the pre-Phase-22 commit
(`ca2876b`, via a scratch clone) and at current HEAD, and diffed the generated `.s` output
directly -- **byte-identical** (modulo a version-hash comment banner) across all 11,
proving zero behavior change for every config that doesn't set `local_prefetch_num` (i.e.
every config that existed before this phase). Stronger than a hardware spot-check: it rules
out the change entirely at the instruction level, not just "still passes this one test".

**New-mechanism correctness, real hardware**: `conv_driver.exe`, all 3 new
`fp32_k2x_lp2` configs (fwd/bwd/wrw), 12 shape/direction combinations total (1x1 at two
different tile-aligned sizes, 3x3 stride1/pad1, stride2/pad1, 5x5 dilation=2) -- **12/12
`valid:y`** against `naive_conv_{fwd,bwd,wrw}_nhwc`. (Group>1 shapes report "not
applicable" for this config regardless of `local_prefetch_num` -- a pre-existing limitation
of the base fp32 128x128 tunable's tensor_a/b_cluster_lengths, unrelated to this phase.)

### Results

fp32 128x128, `_k2x` (prefetch=1) vs `_k2x_lp2` (prefetch=2), same shapes, `-V 0 -t 1 -w 1`,
each figure the median of 3 back-to-back runs (all tight, <0.5% run-to-run variance):

| Shape | Direction | Before (tflops) | After (tflops) | Change |
|---|---|---|---|---|
| n=8,c=k=2048,H=W=32,1x1 (K-loop-bound) | fwd | 191.0 | 209.0 | **+9.4%** |
| 128,120,160,128,3x3 (n=42) | fwd | 195.2 | 198.4 | +1.7% |
| n=8,c=k=2048,H=W=32,1x1 (K-loop-bound) | bwd | 141.3 | 141.4 | flat (noise) |
| 128,120,160,128,3x3 (n=42) | bwd | 163.7 | 149.7 | **-8.6%** |
| n=8,c=k=2048,H=W=32,1x1 (K-loop-bound) | wrw (non-split) | 38.5 | 38.6 | flat (noise) |
| 128,30,40,128,1x1 (n=42) | wrw (non-split) | 0.141 | 0.141 | flat (noise) |

**Honest read, mixed picture -- same shape of result as Phase 9's original k2x rollout**:
fwd gets a real, reproducible win (largest on the K-loop-bound shape, where hiding the
LDS-read behind WMMA issue has the most substep iterations to pay off across). bwd
*regresses* on the 3x3 shape specifically, reproducibly (~8.6%, tight across 3 reruns) --
not yet root-caused; bwd's B operand uses the transposed read-and-pack `shared_load_b_functor`
(scratch-heavy, multiple `ds_read`+shift-pack per element, see Phase 8/9's docstrings) which
may simply have a different latency/occupancy profile under the 2-slot scheme than the
untransposed A path fwd uses for both operands -- a real candidate for follow-up, not
guessed at further here. wrw (non-split) is flat both shapes, expected: non-split wrw's
bottleneck is occupancy (too few workgroups), which this change doesn't touch -- the same
reason Phase 21's epilogue vectorization also showed near-zero for wrw.

**Net recommendation**: ship for fwd (clear win, no downside seen), hold on bwd until the
3x3 regression is understood (the K-loop-bound shape is neutral for bwd, so this isn't a
uniform loss -- shape-dependent, same as k2x itself was). Not worth enabling wrw at all
given zero measured effect either way.

### Critical files (Phase 22)

- `python/igemm/igemm_base.py` -- config-driven `local_prefetch_num` for WMMA (read in the
  authoritative spot, see Lesson above), VGPR formula
- `python/operations/wmma_main_loop.py` -- `emit_wmma_tile(slot=...)`, new
  `emit_extra_substeps_prefetched()`, wired into both the mid-loop body and the `_last`
  tail
- `python/igemm/igemm_{fwd,bwd,wrw}_gtc_wmma_nhwc.py` -- `slot=0` kwarg + slot-offset
  addressing added to all 6 `shared_load_a/b_functor`s; `ctrl.local_prefetch_num` wired
  through `emit_kernel_fma_main_loop()`
- `config/igemm_{fwd,bwd,wrw}_gtc_gfx1250_nhwc_fp32_k2x_lp2.config` -- new, the only
  configs confirmed to fit the VGPR budget today

## Phase 23 (2026-08-26): ISA-driven epilogue tuning -- atomic scope/cascade, LDS bank-conflict pad

Follow-on from a three-agent ISA research pass (VGPR pressure, atomic/reduction
instructions, LDS/scheduling behavior) over the CDNA5/gfx1250 ISA doc. Three flag-gated
features, all default-off (byte-identical to before when unset).

**1. `atomic_scope` (`SCOPE_SYS` -> `SCOPE_DEV`) for the wrw `gemm_k_global_split` atomic
epilogue.** Per the ISA doc's SCOPE table (Table 13), SYS forces a full system-level
flush/invalidate; DEV resolves within the device's L2, sufficient since K-split's
contending workgroups are always on the same device. **Validated correct** on real
hardware (3 shapes: 3x3 stride1/pad1, 1x1, stride2/pad1, all `valid:y`). **Benchmark: no
measurable difference** -- tested a light-contention shape (3x3, ~2ms) and a heavy one
(huge-K 1x1, ~0.15ms), both showed SCOPE_DEV within run-to-run noise of SCOPE_SYS (a couple
of runs even showed it marginally slower). Likely explanation: the atomic unit's own
per-address serialization (confirmed real, per the ISA doc's L2 atomic-unit description) is
the actual bottleneck for these shapes, not the scope-driven cache flush cost -- SCOPE_DEV
saves work that wasn't the binding constraint. Left as an available, correctness-clean
opt-in rather than a default; may matter more on a genuinely idle GPU (this session's
usual contention caveat) or a shape with even heavier cross-workgroup contention.

**2. `atomic_cascade` (TH[2] cascading/deferred-scope atomic) -- IMPLEMENTED, ENCODING
VERIFIED, THEN FOUND TO HANG ON REAL HARDWARE, HARD-BLOCKED.** The ISA doc's RMW-atomic
TH-field table describes a cascading atomic as ideal for exactly this histogram-style
accumulation pattern. The instruction encoding was verified correct via an
`llvm-mc -show-encoding` round-trip probe -- `th:TH_ATOMIC_CASCADE_RT` produces a distinct,
expected bit pattern, confirmed via a batch of candidate identifier names (LLVM requires a
symbolic `th:` value, not a raw number -- `th:4` fails with "expected an identifier").
Wiring it into the actual wrw `_gsplit_cascade` config **hung real hardware** -- the
`conv_driver.exe` process had to be killed via a 2-minute timeout. Root cause, traced from
the generated `.inc`: the kernel's existing `s_wait_storecnt 0x0` immediately before
`s_endpgm` never completes for a cascading atomic, because per the ISA doc's own wording,
"the full specified scope of the op is not realized until a subsequent release
.../fence/atomic-operation... of a matching or higher scope occurs" -- this kernel never
issues that subsequent release, so the store-completion signal the wave is waiting on
simply never arrives. **This is now hard-blocked** with an `assert not
self.atomic_cascade` in `igemm_base.py`'s tunable read (not just left at a default-off
value) specifically so a future attempt to enable it fails loudly at codegen time instead
of silently hanging a GPU again. Fixing this properly would need a companion release
instruction added at the right point (after the K-split accumulation is truly complete,
before whatever reads the final result) -- not attempted this phase; a real follow-up, not
abandoned entirely.

**3. `epilogue_lds_pad` -- LDS bank-conflict padding for the non-atomic (fwd/bwd/non-split
wrw) epilogue's reshuffle.** LDS is 64 banks x 4 bytes, flat-mapped (`bank = byte_addr/4
mod 64`, ISA doc §11.1/11.2). Since `macro_tile_n` is always a multiple of 64, the unpadded
tile-linear address `(row*macro_tile_n+col)*4` collapses to `bank = col mod 64` for every
row -- and since WMMA's per-lane row differs by exactly 8 between the two 16-lane halves
(`v_gemm_im = ((lane>>4)&1)*8`), `(row+8)*macro_tile_n ≡ row*macro_tile_n (mod 64)`,
meaning **both halves collide into the same 16 banks on every single scatter
`ds_write_b32`** -- a real, deterministic, previously-unnoticed 2-way conflict.

Fix: pad the row stride by 4 elements (`padded_stride = macro_tile_n + 4`), not 1 -- a
1-element pad breaks the bank periodicity too, but was ruled out because it also breaks
16-byte alignment for `ds_read_b128`/`global_store_dwordx4` (`vwo=4`) on rows whose index
isn't a multiple of 4, which would misalign (and likely corrupt) those reads. A 4-element
pad both fully separates the two lane-halves into disjoint 16-bank ranges (offset by
`4*8=32`, still within one 64-bank space) AND keeps every row's byte offset a multiple of
16 (`(macro_tile_n+4)*4 = macro_tile_n*4 + 16`, and `macro_tile_n*4` is already a multiple
of 64). This is a real rewrite, not a 1-line change -- the unpadded design deliberately
exploited `macro_tile_n` being a power of 2 for shift-based addressing in both scatter and
gather; padding breaks that, so the scatter's row-shift became a row-multiply, and the
gather had to be restructured to compute the LDS address (row0*padded_stride+col0) and the
global memory address (unaffected by LDS layout) from the same tile-local (row0, col0)
BEFORE either gets overwritten by the global block-offset math -- both share `v_tmp1`/
`v_tmp2`/`v_gather`'s single register each, no new VGPR allocation needed.

**Hard LDS-budget wall found immediately**: the 128x128 tile is already exactly at the
64KB/workgroup limit with zero headroom (Phase 21) -- padding pushes it to 67584 bytes,
over budget. Caught by a new codegen-time assert (`epilogue_lds_bytes <= 65536`) in each
kernel file's `get_kernel_code()`, verified to fire correctly on a deliberately-bad test
config before shipping anything. **`epilogue_lds_pad` is therefore only usable on the
64x64 tile** (17408 bytes, comfortable) and the asymmetric 128x64/64x128 shapes (34816/
33792 bytes, also fine) -- not 128x128. Shipped configs reflect this: the 128x128 section
stays unpadded, only the 64x64 section sets the flag.

**Validated correct** on real hardware for fwd/bwd/non-split-wrw, both the padded 64x64
section and the (deliberately unpadded) 128x128 section in the same config file, across
1x1/3x3/dilated shapes -- all `valid:y`. **Benchmark: no measurable difference** on the
shapes tested (a small single-tile-block case and a slightly larger one), timings matched
baseline within noise (one comparison showed a ~1.6% tflops difference, within normal
run-to-run variance). Given the epilogue is already a small fraction of total kernel time
for compute-bound shapes (the same conclusion Phase 21's epilogue vectorization reached),
and 2-way bank conflicts add real but modest latency (roughly doubling ~32-128 individual
`ds_write_b32`/`ds_read_b128` instructions' issue time, not the whole kernel's), a
sub-noise-floor result here is plausible rather than surprising -- consistent with this
branch's repeated finding that fwd/bwd's bottleneck is the main loop, not the epilogue.

**Overall assessment**: the encoding-and-correctness work is real and now available
(`atomic_scope`, `epilogue_lds_pad`), but neither showed a measured win on the shapes
tested here, and `atomic_cascade` is a confirmed hardware hang, hard-blocked pending a
proper release-mechanism design. Worth remeasuring on an uncontended GPU before writing
either off completely, given this session's persistent contention caveat -- but don't
expect a large effect based on what's been measured so far.

### Critical files (Phase 23)

- `python/operations/coalescing_store_wmma.py` -- `ctrl.atomic_scope`/`atomic_cascade`/
  `atomic_th` (atomic branch), padded scatter/gather addressing gated on
  `ctrl.epilogue_lds_pad` (non-atomic branch)
- `python/igemm/igemm_base.py` -- reads the three new tunable keys; hard `assert not
  self.atomic_cascade` (see above -- do not remove without implementing the release fix)
- `python/igemm/igemm_{fwd,bwd,wrw}_gtc_wmma_nhwc.py` -- wires the three fields into
  `ctrl_coalescing_store_wmma_t`; `get_kernel_code()`'s epilogue LDS-size formula accounts
  for the pad and asserts it fits the 64KB limit
- `config/igemm_wrw_gtc_gfx1250_nhwc_bf16_gsplit_scopedev.config` -- `atomic_scope` only
  (the cascade and scopedev+cascade config variants were removed after confirming the hang
  -- do not recreate them without the release-mechanism fix)
- `config/igemm_{fwd,bwd,wrw}_gtc_gfx1250_nhwc_bf16_ldspad.config` -- `epilogue_lds_pad`,
  64x64 section only (128x128 section deliberately left unpadded, see LDS-budget note)

## Phase 24 (2026-08-26): F16-accumulate WMMA -- unlocks local_prefetch_num=2 for fp16

The highest-leverage item from Phase 22's VGPR audit and the follow-on ISA research: gfx1250
has a 16-bit-accumulate WMMA mode using half the VGPRs of today's fp32-accumulate mode.
`local_prefetch_num=2` (Phase 22) needs +64 VGPRs on a 128x128 tile; fp16/bf16/int8 had zero
headroom (252/256 already used). Halving the accumulator's 128 VGPRs to 64 makes exactly
enough room: 188 (post-halving) + 64 (lp2) = 252 -- landing at the *exact same* ceiling the
fp32 case already proved workable, confirmed via the actual generated `.vgpr_count` (not
just the arithmetic).

**Scope, precisely** (confirmed via the CDNA5 ISA doc's WMMA instruction table, not
assumed): **fp16 only**. `V_WMMA_F16_16X16X32_F16` exists and accumulates in fp16 (each
VGPR packs 2 output rows in bits [15:0]/[31:16], same lane, per the doc's 16-bit C/D-matrix
table). **bf16 has no equivalent** -- its only narrower option,
`V_WMMA_BF16_16X16X32_BF16`, accumulates *natively in bf16* (7-bit mantissa, not fp16's
10-bit) across every k-substep, a real unquantified precision risk for wrw's large-K sums
-- deferred, not implemented. **int8 has no narrower-than-i32 accumulate variant on this
ISA at all** -- permanently out of scope for this technique, confirmed via a full
opcode-table grep, not just "not found yet".

### Design

New `v_wmma_f16_16x16x32_f16` instruction instance (`wmma.py`, `num_v_c=4` vs the f32
variant's 8) -- mnemonic and operand widths verified via an `llvm-mc -show-encoding`
round-trip probe (assembles with a 4-VGPR accumulator, REJECTS an 8-VGPR one). New
`'fp16_f16acc'` table key in `wmma_mapping.py` (not a field -- keeps the existing `'fp16'`
f32-accumulate entries byte-identical), selected via a new `wmma_acc_f16` tunable (default
0). `total_acc_c()` and `wmma_main_loop.py`'s `emit_wmma_tile()` already derive everything
from `inst_wmma.num_v_c` -- both auto-shrink correctly with zero code change.

**Epilogue rewrite** (`coalescing_store_wmma.py`, non-atomic branch only -- the atomic
`gemm_k_global_split` path has no packed-fp16 atomic-add on this ISA, per Phase 19, and is
asserted mutually exclusive with `wmma_acc_f16`): each VGPR now packs 2 logical output rows
(2j lo-half, 2j+1 hi-half). Scatter uses `ds_write_b16`/`ds_write_b16_d16_hi` (both verified
via llvm-mc; the `_d16_hi` variant writes the upper 16 bits directly, no extra shift
instruction needed) at two addresses one row apart -- the second folded into the first's
existing offset immediate, no new address-compute instruction. Gather/global-store width
halves throughout (`ds_read_u16/b32/b64` and `global_store_short/dword/dwordx2` instead of
the b32/b64/b128 and dword/dwordx2/dwordx4 f32 forms) -- verified `global_store_short`'s
exact assembly form via llvm-mc too (canonicalizes to `global_store_b16`). Mutually
exclusive with `epilogue_lds_pad` for now (Phase 23's 4-element pad constant was derived
for 4-byte elements; a correct 2-byte-element pad hasn't been derived).

**Driver integration** (`conv_driver.cpp`): reused the existing `tensor_cast` kernel
mechanism instead of touching the 6 hardcoded fp32 comparison call sites -- added one new
HIP kernel, `tensor_cast_fp32_fp16acc_1d` (`gpu_tensor_cast.cpp`, mirrors the existing
fp16/bf16-to-fp32 kernels' exact style, opposite direction), invoked once per validation to
expand the half-width native buffer back to fp32 before the existing `valid_vector<float>`
logic runs untouched. `wmma_acc_f16` also had to be added to the C++ tunable struct AND
folded into both Python's and C++'s kernel-name encoding (`_f16acc` suffix) -- unlike
Phase 22/23's flags (`local_prefetch_num`/`atomic_scope`/`atomic_cascade`/
`epilogue_lds_pad`), this one changes the output buffer width the driver must allocate and
which native-width kernel the driver must locate by name, so it can't stay a purely
internal codegen choice.

### Three real bugs found via hardware validation (all fixed)

1. **wrw's per-tap output offset hardcoded a 4-byte D-operand shift.** wrw's tap loop
   recomputes `s_p_out_tap` fresh every tap (`emit_kernel_tap_loop`) via an independent
   `s_lshl_b32 ..., 2` -- the ONLY place outside `coalescing_store_wmma.py` that computes a
   byte address into the WMMA-native output buffer. `valid:y` for fwd/bwd, `valid:n` for
   wrw immediately flagged this. Fixed: shift is now `1` under `wmma_acc_f16`, `2`
   otherwise.
2. **fwd's and bwd's group>1 output-offset computation had the identical bug** (`group_idx
   * gemm_n`, hardcoded `<<2`) -- found by grepping for the same "D-operand always 4 bytes"
   comment pattern after finding bug #1, not independently triggered by any test in this
   phase's battery (group>1 wasn't tested for f16acc). Fixed proactively, same pattern.
3. **wrw's group>1 output-offset had the same bug too** (`group_idx * gemm_m * wei_row_c`,
   hardcoded `<<2`) -- found by the same grep sweep. Fixed proactively.
   All four sites now compute `out_elem_byte_shift = 1 if wmma_acc_f16 else 2` locally and
   use it instead of a bare literal. **Lesson, consistent with this branch's whole history**:
   any code that independently re-derives a "byte width" constant instead of referencing a
   single shared source of truth needs an explicit search-and-audit pass whenever that
   constant becomes conditional -- `coalescing_store_wmma.py`'s own width handling was
   updated carefully, but three OTHER call sites silently baked in the same stale assumption
   and were found only by grepping for the giveaway comment text, not by the type system or
   any test that happened to exercise them.
4. **A driver regression introduced by an overly broad find-and-replace**: reusing
   `dtype_alloc_byte` at 3 call sites broke pure-fp32 builds ("use of undeclared identifier"),
   because that variable was declared inside `#if defined(USE_HALF) || ...` -- a block that
   defines nothing for fp32-only configs. Caught by this branch's own byte-identical-assembly
   regression sweep (which builds a plain fp32 config as one of its representative cases),
   not by any f16acc-specific test. Fixed by hoisting the declaration outside the `#if`.

### Verification

Compile-only: `.vgpr_count` confirmed exactly as predicted for all 6 new configs (fwd/bwd/
wrw x f16acc-alone/f16acc+lp2, 128x128 tile) -- 188/252 (fwd), 192/256 (bwd, exactly at the
hard ceiling with zero margin), 187/251 (wrw). Real hardware: 12/12 `valid:y` across a
1x1/3x3-stride1/stride2-pad1/dilated-5x5 shape battery for fwd/bwd, and a battery for wrw
after fixing bug #1. Regression sweep: 8 representative existing configs (spanning fwd/bwd/
wrw, fp16/fp32/bf16, k2x, gsplit+scopedev, ldspad) byte-identical to the pre-Phase-24
commit -- caught bug #4 in the process.

### Results

fp16 128x128, f32acc (baseline) vs f16acc-alone vs f16acc+lp2, `-V 0 -t 1 -w 1`, each figure
reproducible across 2-3 reruns (tight variance):

| Shape | Direction | f32acc (ms) | f16acc alone (ms) | f16acc+lp2 (ms) |
|---|---|---|---|---|
| n=8,c=k=2048,H=W=32,1x1 (K-loop-bound) | fwd | 0.155 | 0.155 | 0.154 |
| 128,120,160,128,3x3 (n=42) | fwd | 0.559 | **0.531 (-5.0%)** | 0.578 (+3.4%) |
| n=8,c=k=2048,H=W=32,1x1 (K-loop-bound) | bwd | 0.184 | 0.187 | 0.186 |
| 128,120,160,128,3x3 (n=42) | bwd | 0.683 | **0.577 (-15.5%)** | 0.726 (+6.3%) |
| n=8,c=k=2048,H=W=32,1x1 (K-loop-bound) | wrw (non-split) | 0.813 | 0.813 | 0.811 |
| n=8,c=256,H=W=32,k=128,1x1 | wrw (non-split) | 0.860 | 0.880 (+2.3%) | 0.881 (+2.4%) |

**Honest read, a genuinely different picture than Phase 22's fp32 result**: F16-accumulate
*alone* is a real, reproducible win for fwd/bwd's 3x3 shape (-5% to -15.5%) -- likely just
the halved `v_c` footprint improving occupancy directly, independent of any pipelining.
**Adding `local_prefetch_num=2` on top makes it WORSE, not better** -- both fwd and bwd's
3x3 shape regress relative to f16acc-alone (and bwd's regresses even past the f32acc
baseline). This is the opposite of Phase 22's fp32 result (where lp2 alone helped fwd) and
means the VGPR-fits-so-it-must-help intuition doesn't hold here -- the K-loop-bound shape
(where Phase 22's fp32 lp2 win was clearest) is flat for f16acc in every combination,
suggesting fp16's LDS-read latency profile or the interaction between the halved
accumulator and the doubled `v_a`/`v_b` doesn't hide latency the same way fp32's did. wrw
(non-split) is flat-to-slightly-worse everywhere, consistent with every prior finding this
session that non-split wrw is occupancy/main-loop-bound, not accumulator- or
epilogue-bound.

**Net recommendation**: ship `wmma_acc_f16` alone for fwd/bwd (real win, no downside seen);
do NOT combine with `local_prefetch_num=2` for fp16 -- despite fitting the VGPR budget
exactly, it measures as a net regression versus f16acc alone in every shape tested. Not
worth enabling for wrw at all (flat-to-worse, matches the established occupancy-bound
diagnosis). bf16's native-bf16-accumulate variant remains a real, unexplored follow-up
question (does its lower per-substep precision actually matter for real shapes?) --
deliberately not investigated this phase, given int8's flat "no option exists" and bf16's
own qualitatively different risk profile.

### Critical files (Phase 24)

- `python/operations/wmma.py` -- new `v_wmma_f16_16x16x32_f16` instruction instance
- `python/operations/wmma_mapping.py` -- new `'fp16_f16acc'` table key
- `python/igemm/igemm_base.py` -- reads `wmma_acc_f16` (fp16-only, asserted mutually
  exclusive with `gemm_k_global_split` and `epilogue_lds_pad`), selects the mapping table
  key, folds `_f16acc` into the kernel name
- `python/operations/coalescing_store_wmma.py` -- non-atomic branch's hi/lo 16-bit
  extraction, doubled per-row addressing, 2-byte instruction selection
- `python/igemm/igemm_{fwd,bwd,wrw}_gtc_wmma_nhwc.py` -- wires `wmma_acc_f16` through;
  fixes to the 3 independent D-operand-width bugs (wrw's per-tap offset, fwd/bwd's group>1
  offset, wrw's group>1 offset); `get_kernel_code()`'s epilogue LDS size accounts for the
  halved element width
- `driver/conv_driver.cpp` -- `is_wmma_f16_acc`, unconditional `dtype_alloc_byte` (fixing
  the pure-fp32 regression), the 3 new `tensor_cast_fp32_fp16acc_1d` invocation sites
- `driver/gpu_tensor_cast/gpu_tensor_cast.cpp` -- new `tensor_cast_fp32_fp16acc_1d` kernel
- `driver/igemm_gtc_base.h` -- `wmma_acc_f16` struct field, config parsing, kernel-name fold
  (kept in sync with the Python side, mirroring the existing `lds_double_buffer`/
  `async_global_load`/`main_loop_interleave` pattern)
- `config/igemm_{fwd,bwd,wrw}_gtc_gfx1250_nhwc_fp16_k2x_f16acc{,_lp2}.config` -- new

## Phase 25 (2026-08-26): GEMM_M boundary/tail handling for WMMA fwd -- closes most of the "not buildable" shape-coverage gap

### Motivation

Workgroup Clusters (the previous investigation, wrw's atomic contention) was dropped after
a real microbenchmark showed no win at wrw's actual measured-optimal split count (630) --
see the session history. Redirected to a different question: how does the older XDLOPS
(MFMA) track achieve such broad shape/config coverage, and can WMMA close the gap?

Research (a background agent, verified directly against source) found two distinct gaps:

1. **WMMA has zero boundary/tail handling.** The WMMA-specific branch of fwd's
   `tunable_is_valid()` (`driver/igemm_fwd_gtc_driver.h`) hard-rejects any shape where
   `gemm_m`/`gemm_n`/`gemm_k` isn't an *exact* multiple of the tile's per-block size. XDLOPS's
   nhwc path has no such rejection for M/N -- it computes a padded grid via
   `utility_integer_divide_ceil` and EXEC-masks the tail block's out-of-range stores
   (`coalescing_store.py`: `v_cmp_gt_u32` + `s_and_saveexec_b64` around each `global_store`).
   This -- not a performance gap -- is why, per `docs/gfx1250_vendor_benchmark_vs_miopen.md`,
   only 20 of 72 real (non-depthwise) conv shapes from a MIOpen trace were even buildable on
   WMMA at all: the other 52 fail the exact-multiple requirement outright.
2. **WMMA only has 4 macro-tile shapes (128x128/64x64/128x64/64x128) vs XDLOPS's 24** --
   mostly because no one has authored the additional config sections/mapping-table entries
   yet (XDLOPS's per-tile "variants" turned out to just be an nxe/gsplit/vector_store
   cross-product, not different instruction shapes -- the 24 distinct (m,n,k) triples are the
   real coverage difference). Deferred to a later, separate phase (additive, no overlap with
   this one's files).

Scoped to (1), fwd only, GEMM_M only, as a pilot -- prove the pattern end-to-end against real
previously-unbuildable shapes before deciding whether to extend to GEMM_N, bwd, and wrw.

### Design

New tunable `wmma_m_tail` (default 0, every existing config unaffected -- fwd only for now,
asserted incompatible with `gemm_k_global_split` since the atomic epilogue branch was never
adapted). When set:

1. **Driver validity relax** (`igemm_fwd_gtc_driver.h`): drops the `gemm_m % gemm_m_per_block
   != 0` hard-reject; `gemm_n`/`gemm_k` still require an exact multiple (no B-operand or
   K-loop tail handling exists yet). No grid-dispatch change needed at all -- both
   `get_grid_size()` (the shared/generic path) and fwd's own direct-launch grid_x computation
   already use `utility_integer_divide_ceil(gemm_m, gemm_m_per_block)`, so the padded grid was
   already being dispatched; only the validity check was stopping it from ever being tried.
2. **No new kernarg needed.** `s_gemm_m` -- the real, unpadded `n*ho*wo` -- was *already* being
   loaded from the kernarg into an sgpr at kernel entry (piggybacking on an existing
   `s_load_dwordx4` that also covers `s_p_out`/`s_gemm_n`), just never read anywhere
   downstream. This phase is the first consumer.
3. **A-operand load masking** (`igemm_fwd_gtc_wmma_nhwc.py`'s `_emit_tap_gather`): the
   existing `v_flag` computation (today: masks conv spatial-padding OOB reads, `hi_idx`/
   `wi_idx` outside `[0,hi)x[0,wi)`) gets one more `v_cndmask_b32` AND'd in -- this row's
   absolute flattened index (`s_block_m_off + v_tid [+ i*block_size]`, recomputed fresh right
   at the check rather than trusting an earlier register's survival across the division
   macro) `< s_gemm_m`. Reuses the exact same masking plumbing every existing load path
   already threads `v_flag` through (`_emit_gld_chunk_load`, `_emit_gld_async_all_chunks`) --
   no new mechanism.
4. **Epilogue store masking** (`coalescing_store_wmma.py`, non-atomic/LDS-reshuffle branch --
   the one genuinely new piece of code): the gather loop's global-memory row value is copied
   into a new scratch VGPR (`v_m_tail_row`, only allocated when `wmma_m_tail` is set) right
   before it's folded into a byte address, then advanced by `row_step_per_pass` each pass.
   Before each `global_store_*`: `v_cmpx_gt_u32 s[s_gemm_m], v[v_m_tail_row]` narrows EXEC to
   lanes whose row is still in range, followed by `s_mov_b32 exec_lo, -1` to restore. This is
   the *wave32* idiom already established elsewhere in this file
   (`igemm_fwd_gtc_wmma_nhwc.py`'s `_emit_gld_chunk_load`: `v_cmpx_le_u32` + `exec_lo`
   restore) -- not XDLOPS's 64-bit `s_and_saveexec_b64`/`s_or_b64` pattern, which doesn't
   apply to a wave32-only kernel.

### Bugs/gotchas found

None in the masking logic itself -- it worked correctly on the first hardware run. One
testing-workflow gotcha: `python3 igemm_codegen.py <config>` (without `-s`) clears `out/`
before regenerating, so a previously-built `.hsaco` silently disappears the next time a
*different* config is generated non-split -- cost a few minutes of `hipModuleLoad ... file
not found` confusion before realizing the fix is just to regenerate the specific config
being tested immediately before running it, not to assume yesterday's build is still there.

### Verification

**Byte-identical regression sweep**: 8 representative existing fwd configs (fp32/fp16/bf16/
int8 bases, plus `_128x64`/`_ldspad`/`_k2x_lp2`/`_k2x_f16acc` variants) diffed against the
pre-Phase-25 commit, ignoring only the version-banner line -- zero differences. Confirms
`wmma_m_tail=0` (every existing config) is completely unaffected.

**"Before" state confirmed**: the base (non-mtail) bf16 config, run against a real
non-128/64-aligned shape (`n=1,c=64,H=10,W=10,k=128`, gemm_m=100), reports `not applicable`
for both its tile sections -- exactly the "not buildable" gap the trace-shape analysis
described.

**Correctness, `_mtail` configs (bf16/fp16/fp32, 128x128 tile)**, all `valid:y` against the
CPU reference:

| Shape | gemm_m | Notes |
|---|---|---|
| n=1,c=64,H=10,W=10,k=128,1x1 | 100 | the original "not buildable" repro |
| n=1,c=64,H=1,W=1,k=128,1x1 | 1 | extreme tail, 127/128 rows masked off |
| n=1,c=64,H=1,W=127,k=128,1x1 | 127 | one short of exact multiple |
| n=1,c=64,H=1,W=129,k=128,1x1 | 129 | spills into a 2nd (mostly-masked) tail block |
| n=1,c=64,H=1,W=256,k=128,1x1 | 256 | exact multiple -- masking-is-a-no-op check |
| n=5,c=64,H=5,W=6,k=128,1x1 | 150 | tail cut lands mid-batch (multi-`n` decomposition) |
| n=2,c=128,H=5,W=5,k=256,g=2,1x1 | 50 | group>1 combined with tail |
| n=2,c=64,H=9,W=9,k=128,3x3,pad=1 | 162 | general conv path (nxe/hi-wi masking + M-tail together), bf16 and fp16 both tested |

### Net result

The single biggest, most concrete lever for WMMA shape coverage found this session: fwd's
GEMM_M exact-multiple requirement -- the reason most real trace shapes were outright
unbuildable, not just slow -- is gone, using entirely reused masking plumbing plus one small
new epilogue guard. GEMM_N tail (B-operand masking, genuinely new mechanism), GEMM_K tail
(main-loop zero-padding, the most invasive of the three), and extending this pattern to
bwd/wrw (each with their own GEMM_M/N semantics) are explicitly deferred -- revisit once this
pilot's results inform whether the added complexity is worth it for those directions too.
The separate tile-shape/config-count expansion (XDLOPS's 24 vs WMMA's 4 macro-tiles) remains
untouched -- independent files, no overlap, can proceed separately.

### Critical files (Phase 25)

- `driver/igemm_gtc_base.h` -- `wmma_m_tail` tunable field + config parsing (not folded into
  the kernel name -- purely a codegen/validity behavior change, mirrors
  `local_prefetch_num`/`atomic_scope`'s precedent, not `wmma_acc_f16`'s)
- `driver/igemm_fwd_gtc_driver.h` -- `tunable_is_valid()`'s relaxed gemm_m check
- `python/igemm/igemm_base.py` -- reads `wmma_m_tail` (fwd-only, asserted mutually exclusive
  with `gemm_k_global_split`)
- `python/igemm/igemm_fwd_gtc_wmma_nhwc.py` -- `v_flag`'s new M-tail AND-in
  (`_emit_tap_gather`), conditional `v_m_tail_row` scratch VGPR, wires `s_gemm_m`/
  `v_m_tail_row` into the `coalescing_store()` call
- `python/operations/coalescing_store_wmma.py` -- the new epilogue EXEC-mask guard
  (non-atomic branch only), `ctrl.wmma_m_tail` field
- `config/igemm_fwd_gtc_gfx1250_nhwc_{bf16,fp16,fp32}_mtail.config` -- new, one 128x128
  section each with `wmma_m_tail=1`

## Phase 26 (2026-08-26): GEMM_M tail for bwd + GEMM_N tail for fwd

### Context

Direct continuation of Phase 25. A research pass ranked the remaining boundary-handling
gaps by reuse/difficulty: M-tail for bwd (near-mechanical port of Phase 25), N-tail for fwd
(one new persistent flag + a composable epilogue guard), M-tail for wrw (harder --
`gemm_k_global_split` is wrw's primary path, not an excluded edge case), N-tail for bwd/wrw
(transposed-operand nuances), and GEMM_K tail for any direction (genuinely new main-loop
mechanism, ~200+ line estimate). Scoped this phase to the first two -- both direct extensions
of Phase 25's exact pattern with no new masking *mechanism*.

### Phase 26a: bwd M-tail

Near-identical port of Phase 25: relaxed `driver/igemm_bwd_gtc_driver.h`'s WMMA
`tunable_is_valid()` (same `gemm_m % gemm_m_per_block` relax, gated on `wmma_m_tail`), added
a 5th `v_cndmask_b32` AND-in to bwd's existing 4-condition `v_flag` (which already checks two
exact-division remainders plus `ho_idx`/`wo_idx` bounds -- see Phase 11), and wired the same
`v_m_tail_row`/epilogue-guard plumbing Phase 25 built.

**Real constraint found, exactly as flagged in the plan**: bwd's 128x128 tile was already at
**256/256 VGPRs** before this phase (zero margin). Adding `v_m_tail_row` (1 more persistent
VGPR) pushed it to **257** -- the assembler rejected it outright ("register index is out of
range"), not a silent miscompile. Shipped **64x64-only** for bwd's `_mtail` configs (177
VGPRs, comfortable margin), documented in the config file header rather than silently
dropping 128x128 or trying to shave a register elsewhere.

**Verification**: byte-identical regression sweep (bwd fp32/fp16/bf16/int8 bases) confirmed
zero change for `wmma_m_tail=0`. Hardware battery (bf16/fp16/fp32, 64x64 tile) covering
gemm_m=1/63/65/100/128/150 (multi-batch), group>1, and padded 3x3 conv -- all `valid:y`.

### Phase 26b: fwd N-tail

B-operand's per-lane column (`block_n_off + tid`) was already computed transiently in the
prologue but never persisted; added a new persistent `v_flag_b` (kernel-lifetime constant,
computed once -- unlike A's per-tap `v_flag`, B's column never changes across taps/K-
iterations) and threaded it into every B-load call site (`global_load_b_functor`,
`shared_store_b_functor`'s remaining-chunks path, and the `main_loop_interleave` chunk-load
path). Scoped out (asserted mutually exclusive) with `async_global_load` -- B's async load
path (`_emit_gld_async_all_chunks`) was only ever validated for A's masking, per its own
Phase 13 docstring -- and with `row_repeat_b > 1` (no per-row flag exists for rows 1+, same
scope narrowing the codebase already applies to B in general).

**Design correction found during implementation** (the research's estimate was wrong on this
point, caught by re-deriving from the actual code rather than trusting the summary): the
epilogue's `v_gather` register does **not** survive across gather passes the way the research
assumed -- it gets overwritten by the very first `ds_read` (its register range is reused as
the LDS-read destination). A second scratch VGPR (`v_n_tail_col`) is needed after all, to
capture the column-in-range flag *before* the first pass's read clobbers `v_gather`. Net: 1
extra VGPR for N-tail's epilogue guard, not 0 as originally estimated -- still fits every
128x128 config tested (fp32/fp16/bf16 all land at 253-255/256 VGPRs, no exclusions needed,
unlike bwd's M-tail).

**Real correctness bug found via hardware validation, fixed before shipping**: the epilogue's
non-atomic store is vectorized 4 elements at a time (`vector_write_out=4`, a hardcoded
constant -- not a config knob anywhere in this codebase), and the EXEC-mask guard only checks
a group's *first* column. For `gemm_n` values that are exact multiples of 4 (e.g. 100, 200,
256), every group's 4 columns are either fully in-range or fully out-of-range, so the guard
is correct. For non-multiples of 4 (99, 101, 126, 127, 129, 1 -- all tested), a boundary
group straddles the real `gemm_n`, and the guard's single first-column check let the *entire*
group's store through, writing 1-3 out-of-range columns of garbage. This surfaced as
`valid:n` (wrong answer) for those shapes, and as an outright process abort
(`Assertion 'is_valid' failed`, core dump) once `IGEMM_ASSERT_WHEN_INVALID=1` was set -- not
a subtle numerical drift, a clean and immediately obvious failure once tested against the
right shapes. **Fixed** by additionally requiring the real (unpadded) `gemm_n` to be a
multiple of 4 in `tunable_is_valid()` whenever `wmma_n_tail` is set -- a documented,
driver-enforced restriction, not a silent limitation. (Properly handling non-multiple-of-4
tails would need per-element masking within a vectorized store group -- a similar order of
complexity to GEMM_K tail's partial-chunk problem, explicitly not attempted here.)

**Verification**: byte-identical regression sweep (fwd fp32/fp16/bf16/int8 bases plus
`_async`/`_interleave`/`_k2x_f16acc`, bwd bases, wrw base + `_gsplit`) confirmed zero change
for `wmma_n_tail=0`. Hardware battery (bf16, 128x128 tile) covering gemm_n=1/4/99/100/101/
124/126/132/200(group>1)/256(exact), confirming the exact multiple-of-4 boundary the fix
predicts (4/100/124/132/200/256 all `valid:y`; 1/99/101/126 correctly rejected as
"not applicable" post-fix, `valid:n`/abort pre-fix). fp16/fp32 spot-checked with a padded 3x3
shape. A combined `_mntail` config (both flags set) validated on two shapes -- confirms the
two EXEC-mask guards (`v_cmpx_gt_u32` for M chained with `v_cmpx_le_u32` for N) compose
correctly via wave32's EXEC-intersecting `v_cmpx` semantics, no interaction bugs.

### Net result

Fwd now has full M+N tail coverage (K still exact-multiple-only); bwd has M-tail (64x64
only, due to VGPR budget). wrw M-tail, N-tail for bwd/wrw, and GEMM_K tail for any direction
remain deferred -- each has its own complication (wrw's `gemm_k_global_split`-as-primary-path
interaction, transposed-operand addressing, or the fundamentally new main-loop mechanism
K-tail needs) that warrants its own focused pass rather than bundling further.

### Critical files (Phase 26)

- `driver/igemm_bwd_gtc_driver.h` -- bwd's `tunable_is_valid()` gemm_m relax (26a)
- `driver/igemm_gtc_base.h` -- new `wmma_n_tail` tunable field + config parsing (26b)
- `driver/igemm_fwd_gtc_driver.h` -- gemm_n relax AND the multiple-of-4 restriction when
  `wmma_n_tail` is set (26b)
- `python/igemm/igemm_base.py` -- `wmma_m_tail` extended to allow `direction in ('fwd',
  'bwd')`; new `wmma_n_tail` read/assert block (fwd-only, excludes `async_global_load` and
  `row_repeat_b > 1`)
- `python/igemm/igemm_bwd_gtc_wmma_nhwc.py` -- 5th `v_flag` condition, conditional
  `v_m_tail_row`, `coalescing_store()` wiring (26a)
- `python/igemm/igemm_fwd_gtc_wmma_nhwc.py` -- new persistent `v_flag_b`/`v_n_tail_col`
  VGPRs, B-load call sites threaded with the new flag, `coalescing_store()` wiring (26b)
- `python/operations/coalescing_store_wmma.py` -- new `ctrl.wmma_n_tail` field, `s_gemm_n`/
  `v_tmp4` params, the chained `v_cmpx_le_u32` guard and its pass-invariant flag capture (26b)
- `config/igemm_bwd_gtc_gfx1250_nhwc_{bf16,fp16,fp32}_mtail.config` -- new, 64x64-only (26a)
- `config/igemm_fwd_gtc_gfx1250_nhwc_{bf16,fp16,fp32}_ntail.config` -- new, 128x128 (26b)
- `config/igemm_fwd_gtc_gfx1250_nhwc_bf16_mntail.config` -- new, combined M+N tail (26b)

## Phase 27 (2026-08-26): BF16-accumulate WMMA -- unblocks bwd's 128x128 M-tail

### Motivation

Directly motivated by Phase 26a's finding: bwd's plain 128x128 bf16 tile sits at exactly
256/256 VGPRs with zero margin, which is why `wmma_m_tail` (needing 1 extra VGPR) could only
ship 64x64 for bwd. Asked to look for ISA-doc-grounded VGPR-reduction options. Found two:

1. **VGPR-MSB indexing** (doc §3.3.2.3): a wave can be allocated up to 1024 VGPRs; addressing
   past VGPR 255 needs `S_SET_VGPR_MSB` before instructions touching the high range. Real, but
   a blunt/high-overhead instrument -- not pursued.
2. **`V_WMMA_BF16_16X16X32_BF16`**: a bf16-native-accumulate WMMA variant, structurally
   identical to Phase 24's `V_WMMA_F16_16X16X32_F16` -- confirmed via `llvm-mc` that it
   requires a 4-VGPR accumulator (rejects 8-VGPR), the same 2x reduction in `v_c` Phase 24
   already proved for fp16. (Also checked `V_WMMA_BF16F32_16X16X32_BF16` -- its C input is
   8-VGPR f32 while D output is 4-VGPR bf16, different widths, so it can't be an in-place
   iterative accumulator. Not usable.)

### Design

Mirrors Phase 24 (`wmma_acc_f16`) almost exactly with a new `wmma_acc_bf16` tunable (bf16-only,
mutually exclusive with `gemm_k_global_split`/`epilogue_lds_pad`/`wmma_acc_f16` for the same
reasons as Phase 24). **Key finding: `coalescing_store_wmma.py` needed zero changes** -- its
f16acc epilogue code (`elem_bytes`/`ds_write_b16`/`ds_read_u16`/`global_store_short`, all gated
on `ctrl.wmma_acc_f16`) operates on raw 16-bit patterns with no fp16-specific behavior, so
`wmma_acc_bf16` just funnels into the SAME ctrl field
(`ctrl_coalescing_store_wmma.wmma_acc_f16 = tunable.wmma_acc_f16 or tunable.wmma_acc_bf16`).
Not excluded from `wmma_m_tail`/`wmma_n_tail` (unlike `gemm_k_global_split`) since those guards
operate on lane/row/column indices, not element width -- this is the actual point of the
phase: a combined `bf16_bf16acc_mtail` config at 128x128 lands at **193/256 VGPRs** (vs the
257 that blocked plain `wmma_m_tail` at 128x128), a 63-VGPR swing, far more headroom than
strictly needed. New `driver/gpu_tensor_cast/gpu_tensor_cast.cpp` kernel
(`tensor_cast_fp32_bf16acc_1d`) mirrors Phase 24's fp16 cast-back kernel, using a new
`__bfloat16_to_float` helper (a lossless left-shift, unlike the existing rounding
`__float_to_bfloat16` used for the opposite direction).

### Verification

Byte-identical regression sweep (fwd/bwd/wrw, all precisions, f16acc/gsplit/mtail/ntail
configs) -- zero change for `wmma_acc_bf16=0`. `.vgpr_count` confirmed as designed: 128x128
tile drops from 252/256/251 (f32-accumulate baseline) to 188/192/187 (fwd/bwd/wrw) with plain
bf16acc, and the bwd `bf16acc+mtail` combo lands at 193 -- comfortably under 256.

**Hardware correctness, bf16acc alone**: fwd validated `valid:y` across 1x1 (large batch),
3x3-pad1 (n=42 real shape), and a K-sweep up to 2048 with no failures. The bwd/wrw M-tail
combo battery (Phase 26a's shape set, at 128x128 this time) passed **7 of 8** cases
(extreme/one-short/one-over/exact/multi-batch/padded-3x3 all `valid:y`) -- see below for the
1 failure and why it's not a Phase 27 bug.

### Two real findings, both pre-existing / not Phase-27 bugs (documented, not fixed here)

1. **BF16-native-accumulate has a real, reproducible precision ceiling at larger K for bwd
   and wrw** -- not observed for fwd in equivalent testing. Controlled K-sweep (bwd, 1x1,
   `n=1`): K=256/512 `valid:y`, K=1024 `valid:n`, reproducible across 3 different random
   seeds (not borderline/flaky -- a real, consistent divergence past a K threshold, not
   precision noise). wrw showed the same pattern (small K `valid:y`, `gemm_k=8192` `valid:n`).
   fwd stayed `valid:y` up to K=2048 in the shapes tested. bf16's mantissa (7 bits) is
   meaningfully coarser than fp16's (10 bits, which showed no such issue in Phase 24's
   validation), so this is very plausibly inherent to bf16 accumulation precision interacting
   with each direction's specific data distribution -- the underlying masking/addressing
   mechanism is independently confirmed correct (small-K shapes and the VGPR counts both
   check out). **Practical implication**: do not treat `wmma_acc_bf16` as a blanket "always
   safe" choice for bwd/wrw's larger-K shapes the way `wmma_acc_f16` appears to be -- validate
   accuracy per-shape, same as any lower-precision accumulate mode requires. Not investigated
   further this phase (would need per-shape accuracy sweeps or a mitigation like periodic
   renormalization, out of scope).
2. **bwd has a pre-existing group>1 bug when a single group spans multiple N-blocks**
   (`gemm_n > gemm_n_per_block`, i.e. `c/group` needs more than one N-tile) -- found while
   testing the bf16acc+mtail combo with `c=256,k=128,g=2` (`gemm_n=128` on a `gemm_n_per_block`
   of 64 or 128, i.e. >=2 N-blocks per group). Reproduces with **plain fp16/f32-accumulate**,
   **at both 64x64 and 128x128 tiles**, with **no `wmma_m_tail`/`wmma_n_tail`/`wmma_acc_bf16`
   involved at all** -- confirmed unrelated to every feature shipped in Phases 24-27.
   Phase 26a's own group>1 validation (`n=2,c=128,H=5,W=5,k=128,g=2`) still reproduces
   correctly (`valid:y`) because that shape's `gemm_n=64` happens to be exactly one N-block
   per group -- it never exercised the multi-N-block-per-group path. This looks like a
   group-index-decoding bug (`s_group_idx` derived from `s_by`) that only manifests when a
   group's data spans more than one N-tile. Not fixed here -- flagged for a dedicated future
   investigation; likely affects wrw too given the shared code pattern (not yet checked).

### Critical files (Phase 27)

- `python/operations/wmma.py` -- new `v_wmma_bf16_16x16x32_bf16` instruction instance
- `python/operations/wmma_mapping.py` -- new `'bf16_bf16acc'` table key
- `python/igemm/igemm_base.py` -- `wmma_acc_bf16` tunable (bf16-only, excludes
  `gemm_k_global_split`/`epilogue_lds_pad`/`wmma_acc_f16`; NOT excluded from
  `wmma_m_tail`/`wmma_n_tail`), `wmma_mapping_key` selection, kernel-name fold
- `python/igemm/igemm_{fwd,bwd,wrw}_gtc_wmma_nhwc.py` -- wires `wmma_acc_bf16` everywhere
  `wmma_acc_f16` is wired (mapping key, `ctrl_coalescing_store_wmma.wmma_acc_f16` OR'd in,
  `epilogue_elem_bytes`/`out_elem_byte_shift`(`_group`) byte-width selection)
- `driver/gpu_tensor_cast/gpu_tensor_cast.cpp` -- new `__bfloat16_to_float` helper and
  `tensor_cast_fp32_bf16acc_1d` kernel
- `driver/conv_driver.cpp` -- `is_wmma_bf16_acc`, `dtype_alloc_byte` fold, 3 new
  `tensor_cast_fp32_bf16acc_1d` invocation sites (fwd/bwd/wrw)
- `driver/igemm_gtc_base.h` -- `wmma_acc_bf16` struct field, config parsing, kernel-name fold
- `config/igemm_{fwd,bwd,wrw}_gtc_gfx1250_nhwc_bf16_k2x_bf16acc{,_lp2}.config` -- new
- `config/igemm_bwd_gtc_gfx1250_nhwc_bf16_bf16acc_mtail.config` -- new, the phase's key
  validation config (128x128, both flags set)

## Phase 28 (2026-08-26): TDM-based global-to-LDS load pilot (fwd, 1x1 conv, A operand only)

### Motivation

Researching FlyDSL and hipconv (two other AMD ROCm GPU-kernel projects on this machine,
see `docs/gfx1250_external_research_findings.md` for the full writeup) turned up TDM
(Tensor Data Mover) -- a dedicated gfx1250 DMA unit (`TENSOR_LOAD_TO_LDS`/
`TENSOR_STORE_FROM_LDS`, ISA doc §10.11) that MISA had never used, despite the ISA doc
having documented it all along. Both other projects use it heavily in production. Verified
independently on real hardware before touching any real kernel: `tensor_load_to_lds`/
`tensor_store_from_lds` both work; hardware OOB behavior confirmed exactly as documented
(load zero-fills out-of-bounds rows, store silently drops out-of-bounds writes); confirmed
through MISA's actual pipeline (`-x assembler`, `hipModuleLoad`+`hipExtModuleLaunchKernel`,
not just HIP C++); descriptor bit-packing cross-checked against the ISA doc, FlyDSL's
`CopyAtom.cpp`, AND hipconv's `bunnies_mi400.hpp` -- three independent sources agree.

This is a big enough capability (hardware OOB could replace Phases 25/26's EXEC-mask
guards AND finally solve GEMM_K tail, which has no existing plan) that it needed a
narrowly-scoped first pilot inside a real kernel before deciding how far to take it.

### Scope

fwd only, A-operand load only, 1x1-conv-only (`nxe=0`), new opt-in `tdm_global_load`
tunable -- mirrors `async_global_load`'s (Phase 13) exact scoping and mutual-exclusion
discipline (excludes `async_global_load`, `main_loop_interleave`, `row_repeat_a>1`,
`local_prefetch_num>1`). Deliberately does NOT attempt M/K-tail via TDM's OOB fields yet,
NOT the B operand, NOT bwd/wrw, NOT TDM store for the epilogue -- narrowest possible slice
to prove the mechanism inside one real main loop first.

### Design

`coalescing_store_wmma.py` needed no changes (this pilot is load-only). The key insight
integrating into `wmma_main_loop.py`: **TDM shares async's exact control-flow shape** --
"data lands directly in LDS via the load instruction itself, no separate store step" --
differing only in which counter to wait on (`s_wait_tensorcnt` vs `s_wait_asynccnt`). So
`wmma_main_loop.py`'s existing `async_a`/`async_b`/`any_async`/`any_old` logic was
generalized to `a_style_async = async_a or tdm_a` (and `_b`), with `any_tdm` tracked
separately just for the wait-instruction choice -- every other branch (`f_sst_a`
skip-if-async-style, `f_gld_a` re-issue timing) needed zero new logic.

New per-kernel pieces (`igemm_fwd_gtc_wmma_nhwc.py`):
- `_emit_tdm_descriptor_setup_a()`: builds the A-operand's group0 (4 SGPRs)/group1
  (8 SGPRs) descriptor ONCE in the prologue. `tile_dim0`/`tile_dim1` (gemm_k_per_block/
  gemm_m_per_block) are compile-time constants, baked in as immediates; `tensor_dim0`/
  `tensor_dim1`/stride (gemm_k/gemm_m/in_c_total) are runtime SGPR values built via
  `s_lshl_b32`/`s_lshr_b32`/`s_or_b32` bit-packing, verified against the same probe that
  confirmed TDM on real hardware.
- `global_load_a_functor()`: new `tdm_global_load` branch emits one
  `tensor_load_to_lds s[g0:g0+3], s[g1:g1+7]` instead of the chunked
  `global_load_dwordx4`/`global_load_async_to_lds_b128` sequence.
- `move_slice_window_a_functor()`: new branch advances the descriptor's `global_addr`
  (group0 s2/s3, a genuine 64-bit scalar address) by `bytes_per_row` per main-loop
  iteration via `s_add_u32`/`s_addc_u32` -- s3's top 2 bits hold the constant `type=2`
  field (set once, never re-touched); safe to `addc` directly into it since `global_addr`
  is only 57 bits (25 meaningful bits in s3), nowhere near the type field for any real
  address.
- SGPR allocation for the 12-SGPR descriptor needed explicit alignment
  (`sseq(4, 4)`/`sseq(8, 4)`) -- the assembler rejected the default unaligned allocation
  ("invalid register alignment") on the first attempt; both TDM operand groups need
  4-SGPR alignment for the multi-register addressing mode.

**Known, deliberate inefficiency**: every wave in the workgroup issues an identical,
redundant TDM load (TDM ignores EXEC and isn't per-lane, so this doesn't affect
correctness, just wastes DMA bandwidth 4x for a 128x128/4-wave tile). Confirmed by
hipconv's research to be a real, measured cost (single-issuer-wave design measured at
-3% to -7% improvement) -- explains why this pilot's first hardware timing came back ~20%
slower than the existing `_async` config on the same shape. Not fixed in this phase
(correctness-first, per the plan); see `docs/gfx1250_external_research_findings.md` for
the concrete follow-up techniques (single-issuer-wave, deeper `s_wait_tensorcnt(N>0)`
pipelining, not draining LDS reads before the next TDM issue).

### Verification

Byte-identical regression sweep (fwd fp32/fp16/bf16/int8 bases, `_async`/`_interleave`,
Phase 25/26 tail configs, Phase 27 bf16acc config) -- zero change for `tdm_global_load=0`,
including through the `wmma_main_loop.py` control-flow generalization. Hardware
correctness: bf16 across an exact-multiple large shape (n=8,c=k=2048,H=W=32,
`valid:y`, and the SAME shape's existing `_async` config for a timing sanity check),
medium-K, group>1, and a multi-K-block shape (32 main-loop iterations, confirming the
descriptor advance is correct across many iterations, not just one) -- all `valid:y`.
fp16 and fp32 spot-checked on a multi-K-block shape each, both `valid:y`.

### Net result

TDM is now proven correct inside a real MISA kernel's main loop, through the project's
actual hand-assembly pipeline, for the narrowest useful case. The path is now open to:
(a) fix the redundant-issue inefficiency (single-issuer-wave, per hipconv's measured
technique) to get this pilot to a real performance comparison against `_async`, (b) extend
to the B operand and multi-tap convs, (c) use `tensor_dim0 < tile_dim0` for GEMM_K tail
(hipconv's depthwise kernel does exactly this for its channel dimension, in production) --
the single biggest capability unlock, since GEMM_K tail has no other plan today, (d) use a
3D descriptor's `tensor_dim2=0/1` trick for GEMM_M boundary handling as a hardware
alternative to Phases 25/26's EXEC-mask guards, (e) TDM store for the epilogue. None of
these attempted yet -- decide next steps after this pilot's findings are reviewed.

### Critical files (Phase 28)

- `driver/igemm_gtc_base.h` -- `tdm_global_load` struct field, config parsing,
  kernel-name fold (`_tdm` suffix)
- `python/igemm/igemm_base.py` -- `tdm_global_load` tunable (fwd-only, `nxe==0`-only,
  excludes `async_global_load`/`main_loop_interleave`), kernel-name fold
- `python/igemm/igemm_fwd_gtc_wmma_nhwc.py` -- `row_repeat_a`/`local_prefetch_num`
  exclusion asserts, `s_tdm_g0`/`s_tdm_g1` SGPR allocation, `_emit_tdm_descriptor_setup_a`,
  `global_load_a_functor`/`move_slice_window_a_functor` TDM branches,
  `ctrl.tdm_global_to_lds_a` wiring
- `python/operations/wmma_main_loop.py` -- `tdm_global_to_lds_a`/`_b` ctrl fields,
  `a_style_async`/`b_style_async`/`any_tdm` generalization of the async control flow,
  `s_wait_tensorcnt` emission
- `config/igemm_fwd_gtc_gfx1250_nhwc_{bf16,fp16,fp32}_tdm.config` -- new
- `docs/gfx1250_external_research_findings.md` -- new, the full FlyDSL/hipconv research
  writeup this phase is based on

## Phase 29 (2026-08-27): single-issuer-wave fix for the TDM pilot

Fixes Phase 28's known, deliberate inefficiency: every wave in the workgroup was issuing
an identical, redundant `tensor_load_to_lds`. TDM instructions ignore EXEC entirely
("issued no matter if EXEC==0... makes no difference which lanes are enabled or
disabled" -- ISA doc), so no per-lane EXEC-mask trick (the idiom used everywhere else in
this codebase) can suppress the redundant issue on non-issuing waves -- only a genuine
scalar branch can.

**Design**: a persistent `s_wave_id` scalar is derived once in the prologue, right after
`v_tid` is set (`v_readfirstlane_b32` off lane 0's flat tid, then `s_lshr_b32` by 5 --
safe at that point since no lanes are disabled yet). A new helper,
`_emit_wave0_only(body_fn)`, wraps a callable in `s_cmp_eq_u32 s[s_wave_id], 0` /
`s_cbranch_scc0 <fresh-label>` / `body_fn()` / `<label>:`, using this codebase's existing
`_emit_front`-for-labels idiom. Each call site gets its own label
(`self._tdm_label_counter`) since the guarded `tensor_load_to_lds` is reached from two
distinct points in the Python source (the initial issue before the main loop, and the
re-issue inside the loop body) even though neither is unrolled per iteration. Only the
`tensor_load_to_lds` instruction itself is gated -- `_emit_tdm_descriptor_setup_a` and the
per-iteration descriptor advance (`move_slice_window_a_functor`) stay unconditional
(cheap, per-wave-independent SALU; gating them would only save a few cycles on
non-issuing waves for no correctness benefit). The centralized `s_wait_tensorcnt 0` (in
`wmma_main_loop.py`, unchanged from Phase 28) needed no gating either: non-issuing waves
have `TENSORcnt=0` already, so waiting on it is a harmless no-op, and wave 0's own wait
still correctly precedes the barrier that publishes the loaded tile to the other waves.

**Byte-identical regression sweep**: all 98 non-TDM gfx1250 configs regenerate identically
to the pre-phase baseline (commit `c6f222e`); the 3 `_tdm` configs (bf16/fp16/fp32) show
exactly the expected diff -- one new `s_wave_id` SGPR, two new
`s_cmp_eq_u32`/`s_cbranch_scc0`/label triples wrapping the two `tensor_load_to_lds` call
sites, and `.amdhsa_next_free_sgpr` bumped by 1.

**Hardware validation**: bf16/fp16/fp32 all `valid:y` across the exact-multiple large
shape (n=8,c=k=2048,H=W=32), a multi-K-block shape (32 main-loop iterations, confirming
the wave-0 gate and descriptor advance both stay correct across many iterations), and
group>1 (g=2). Timing on the exact-multiple bf16 shape (this shape has real run-to-run
noise, so multiple runs of each were taken): `_tdm` clusters tightly at 604-606 tflops
(0.113-0.114ms) across 5 runs; `_async` on the same shape varies more, 602-687 tflops
across 4 runs (mean ~660). That puts the TDM pilot at roughly **7-9% slower than
`_async`**, down from Phase 28's ~20% -- a real improvement, consistent with hipconv's
measured -3% to -7% single-issuer-wave gain (the larger observed improvement here likely
reflects that this shape is K-loop-bound, so eliminating 4x redundant global-memory
bandwidth consumption has an outsized effect). Still not at parity with `_async`; not
investigated further this phase (single-issuer-wave was the specific, scoped ask).

**Critical files (Phase 29)**: `python/igemm/igemm_fwd_gtc_wmma_nhwc.py` --
`s_wave_id` SGPR allocation, prologue derivation, `_emit_wave0_only` helper,
`_tdm_label_counter`, `global_load_a_functor`'s TDM branch now wrapped in
`_emit_wave0_only`.

## Phase 30 (2026-08-27): TDM for the B operand (fwd, 1x1)

Straight port of Phase 28/29's A-operand pattern to B, closing the prerequisite the
GEMM_K-tail investigation surfaced: B's non-TDM load path has zero K-axis masking, so
enabling K-tail (Phase 31) needs both operands on TDM, not just A.

**Design**: new `s_tdm_g0_b`/`s_tdm_g1_b` descriptor SGPRs (same 4/8-SGPR, 4-aligned
groups as A's), built by `_emit_tdm_descriptor_setup_b` -- called once, right after
`s_p_wei`'s group-offset add and `s_wei_k_stride` are computed in the prologue. B's
weight layout ([K_out][Y][X][C_in]) makes its 1x1-case row width `gemm_k` (shared with
A) and row stride `wei_k_stride` (A's analogue: `in_c_total`), so the descriptor mirrors
A's field-for-field: `tensor_dim0=gemm_k`, `tensor_dim1=gemm_n` (A: `gemm_m`),
`tile_dim0=gemm_k_per_block` (shared), `tile_dim1=gemm_n_per_block` (A:
`gemm_m_per_block`), `lds_addr=lds_a_size` (B's region starts right after A's). One
real difference from A: B's per-workgroup offset (`s_block_n_off`) is NOT folded into
`s_p_wei` anywhere else in this kernel (the non-TDM path adds it per-lane, at the VGPR
level) -- so, unlike A's descriptor (which reuses an already-group-corrected `s_p_in`
plus its own `block_m_off` add), B's `global_addr` computation adds `block_n_off *
wei_k_stride * data_byte` explicitly, the same shape as A's `block_m_off * in_c_total *
data_byte` add.

`global_load_b_functor` and `move_slice_window_b_functor` got the same TDM branches as
their A counterparts (issue wrapped in `_emit_wave0_only` from the start -- no
un-gated intermediate step this time, since Phase 29 already shipped);
`ctrl.tdm_global_to_lds_b` flips from Phase 28/29's hardcoded `False` to
`self.tunable.tdm_global_load`.

**Byte-identical regression sweep**: all 98 non-TDM gfx1250 configs regenerate
identically to the pre-phase baseline (commit `35dd4ab`); the 3 `_tdm` configs show
exactly the expected diff -- B's descriptor build, B's `tensor_load_to_lds`/advance
replacing the old `global_load_dwordx4`/`ds_write_b128`/advance sequence, both wave0-gated
with their own labels.

**Hardware validation**: bf16/fp16/fp32 all `valid:y` across the same battery as Phase
29 (exact-multiple large shape, multi-K-block, group>1) -- now genuinely exercising B's
TDM path end-to-end (these shapes read real, non-degenerate weight tensors, so a wrong
`wei_k_stride`/`block_n_off`/`lds_addr` placement would have surfaced as a correctness
failure, not just a masking gap).

**Critical files (Phase 30)**: `python/igemm/igemm_fwd_gtc_wmma_nhwc.py` --
`s_tdm_g0_b`/`s_tdm_g1_b` SGPR allocation, `_emit_tdm_descriptor_setup_b`,
`global_load_b_functor`/`move_slice_window_b_functor` TDM branches,
`ctrl.tdm_global_to_lds_b` wiring.

## Phase 31 (2026-08-27): GEMM_K tail via TDM's hardware OOB

Enables a genuinely partial last K-block for `tdm_global_load` configs, using TDM's
hardware zero-fill instead of any software masking -- the capability this whole TDM
investigation was ultimately aimed at (GEMM_K tail has no other plan in this codebase).

**A real semantics bug caught before it shipped**: the approved plan for this phase said
"`tensor_dim0` stays untouched -- TDM's own OOB logic handles the tail," on the
assumption that `tensor_dim0` is compared against an absolute, fixed tensor origin. Before
implementing that, a standalone hardware probe (`__builtin_amdgcn_tensor_load_to_lds`, two
back-to-back calls with an ADVANCING `global_addr`, one variant holding `tensor_dim0`
constant across both calls, the other decrementing it by the tile width between calls,
LDS marked `volatile` to stop the compiler from dead-code-eliminating the reads-back since
it doesn't model the intrinsic as writing to `__shared__`) showed conclusively: **holding
`tensor_dim0` constant across advancing calls reads real, out-of-bounds memory for the
tail elements instead of zero-filling them.** `tensor_dim0`'s OOB check is relative to
*that call's* `global_addr` ("start of the tile within the tensor, not the start of the
tensor" -- ISA doc 10.11.2) -- i.e. "how many valid elements remain from here" -- not an
absolute offset from a fixed origin. This matches the ISA doc's own hardware-native
`iterate_enable` descriptor-reuse feature (10.11.3.1), which internally advances
`global_addr` across passes and must therefore treat `tensor_dim0` the same
relative way, and matches hipconv's own formula for its (non-looped, single-shot)
depthwise channel tail: `tdm_chan_ext = max(0, min(WG_CH, C - wg_ch_base))` --
literally "remaining from the current base," generalized here to a looped K-reduction.
Decrementing `tensor_dim0` every iteration is therefore load-bearing, not optional.

**Design**:
1. **Driver**: `igemm_fwd_gtc_driver.h`'s WMMA applicability check now only rejects
   `gemm_k % gemm_k_per_block != 0` `when !tunable->tdm_global_load` (mirrors Phases
   25/26's `wmma_m_tail`/`wmma_n_tail` relax pattern).
2. **Loop iteration count**: needed NO change. `s_knum` stays `s_mov_b32 s[s_knum],
   s[s_gemm_k]` (the real, unrounded value) -- the main loop's existing `s_kitr -=
   unroll_k; branch if > 0` structure already runs exactly `ceil(gemm_k /
   gemm_k_per_block)` iterations for ANY `s_knum`, multiple or not (verified by hand-
   tracing the iteration count for `gemm_k=100, gemm_k_per_block=32`: 4 iterations,
   matching `ceil(100/32)`) -- this was already latent, general behavior, just never
   exercised by a non-exact-multiple `gemm_k` before this phase.
3. **New persistent SGPR `s_tdm_k_remain`**: initialized to `gemm_k` in the prologue
   (matching what `_emit_tdm_descriptor_setup_a/b` already set `tensor_dim0` to for the
   first tile), decremented by `gemm_k_per_block` exactly ONCE per main-loop iteration
   (in `wmma_main_loop.py`, right before `move_slice_window_a/b_functor` -- shared
   between A and B since both use the same K-tile schedule, so decrementing inside
   either functor individually would double-count).
4. `move_slice_window_a_functor`/`move_slice_window_b_functor`'s TDM branches now rebuild
   `tensor_dim0`'s packed bits (`g1(1)`/`g1(2)` for A, `g1_b(1)`/`g1_b(2)` for B -- see
   `_emit_tdm_descriptor_setup_a`'s docstring for the exact bit layout) from
   `s_tdm_k_remain` after every address advance, re-deriving `tensor_dim1`'s lo16
   (`gemm_m` for A, `gemm_n` for B -- unchanged across the K loop, just re-OR'd in fresh
   since it shares a register with `tensor_dim0`'s hi16). 4 extra SALU instructions per
   operand per iteration.
5. No new tunable -- K-tail support falls straight out of `tdm_global_load` for any
   config that also has both operands on TDM (Phase 30).

**Byte-identical-shape regression sweep**: all 98 non-TDM gfx1250 configs regenerate
identically to the pre-phase baseline (commit `68ffe89`); the 3 `_tdm` configs show
exactly the expected diff -- one new `s_tdm_k_remain` SGPR, its prologue init, the
per-iteration decrement, and the two tensor_dim0-rebuild sequences (8 new instructions
total, 4 per operand) replacing nothing (the address-advance instructions are unchanged).

**Hardware validation** (the actual point of this phase): existing exact-multiple/multi-
K-block/group>1 battery re-run first to confirm no regression (`valid:y`, unchanged
timing) -- the `tensor_dim0` rebuild never disagrees with the constant value in this
case, since `remain >= tile_dim0` for every non-tail iteration regardless of which way
it's computed. Then a genuinely non-exact-multiple K battery per precision (bf16/fp16
`gemm_k_per_block=32`, fp32 `gemm_k_per_block=4`): remainder-4 (`c=100`), one-over-a-
multiple (`c=2017`/`c=257`), one-short-of-a-multiple (`c=2015`/`c=255`), and a single
wholly-partial tile (`c=31`/`c=3`, smaller than one whole tile) -- **all `valid:y`**
across all three precisions. Also confirmed: a non-TDM config with the same non-exact
`c=100` shape is still correctly rejected (`not applicable`), proving the driver relax
is properly scoped to `tdm_global_load` only; and K-tail combined with `group>1`
(`c=100,g=2`, giving a 50-element-per-group K dimension, not a multiple of 32) is also
`valid:y`.

**Net result**: GEMM_K tail -- previously unsolved anywhere in this codebase -- now works
for fwd/1x1/`tdm_global_load` configs, with zero software masking in the compute path;
the only cost is a few extra SALU instructions per K-loop iteration to keep
`tensor_dim0` honest. This closes out the Phase 28-31 TDM arc: pilot correctness (28),
single-issuer-wave efficiency (29), B-operand parity (30), and the K-tail capability
unlock this investigation was aimed at (31). Not yet extended: bwd/wrw, multi-tap
convs (`nxe!=0`), or GEMM_M tail via TDM's 3D-descriptor `tensor_dim2` trick (still on
EXEC-mask guards, per Phases 25/26) -- all flagged as future follow-ons in
`docs/gfx1250_external_research_findings.md`.

**Critical files (Phase 31)**: `driver/igemm_fwd_gtc_driver.h` -- `gemm_k` tail relax;
`python/operations/wmma_main_loop.py` -- `ctrl.s_tdm_k_remain`, per-iteration decrement;
`python/igemm/igemm_fwd_gtc_wmma_nhwc.py` -- `s_tdm_k_remain` SGPR allocation + prologue
init, `move_slice_window_a/b_functor`'s `tensor_dim0` rebuild.

## Phase 32 (2026-08-27): `s_setprio` bracketing around WMMA issue

First of three Tier 1 items from `docs/gfx1250_perf_parity_action_plan.md`'s cross-source
synthesis. `s_setprio(1)`/`s_setprio(0)` (CDNA5 ISA doc 5.2/5.7.2.1: sets 2 bits of
`USER_PRIO`, 0=low/3=high) bracketing a WMMA-issue burst was independently confirmed as
real, shipping code in both CK's WMMA v1 pipeline and hipconv's `direct/kernel.hpp` --
the strongest possible cross-project signal for a performance idea. Verified assembly
syntax via `llvm-mc -show-encoding -mcpu=gfx1250` (`s_setprio 1`/`s_setprio 0`, clean
4-byte encodings) before touching any kernel code.

**Design**: new `wmma_setprio` tunable (default 0, every existing config byte-identical),
wired through a new `ctrl.wmma_setprio` field on `ctrl_wmma_main_loop_t`. Brackets the
ENTIRE body of `wmma_main_loop.py`'s `emit_wmma_tile()` (one call = one MAC-loop body's
full back-to-back WMMA burst, `wave_repeat_m x wave_repeat_n` instructions) with
`s_setprio 1` before the burst and `s_setprio 0` after -- matching CK's exact granularity
("brackets the first WMMA issue of each MAC-loop body... closes the body with
`s_setprio(0)`"), not per-individual-instruction. Since `emit_wmma_tile()` is a single
shared nested function called from every main-loop variant (prefetched, interleaved,
plain), this one change point covers all of them automatically. Also required a driver-
side (`driver/igemm_gtc_base.h`) struct field + config-parsing + kernel-name fold, since
the C++ driver reconstructs the kernel symbol name independently to look it up via
`hipModuleGetFunction` -- without the fold, the driver would search for the wrong
(unsuffixed) name for any `wmma_setprio=1` config.

**Byte-identical regression sweep**: all 101 pre-existing gfx1250 configs regenerate
identically to the pre-phase baseline; only the 3 new `_setprio`/`_gsplit_setprio` configs
(fwd, bwd, wrw) differ, by construction.

**Hardware validation**: `valid:y` across fwd (exact-multiple large shape, group>1), bwd
(exact-multiple, group>1), and wrw (`_gsplit_setprio`, exact-multiple and the benchmark
doc's worst-case shape `c=192,H=60,W=80,k=64,1x1`) -- correctness is solid, zero-risk
scheduling hint confirmed to not change results anywhere tested.

**Timing**: measured under severe, actively-worsening GPU contention this session --
`rocm-smi --showuse --showpids` itself crashed with an internal assertion failure
(`GetGPUMetricsFormat1`) partway through this phase's testing, and plain `rocm-smi`
started reporting `get_power_avg`/`sclk` as unsupported -- both symptoms of a co-resident
tenant's telemetry queries, not a MISA-caused GPU fault (confirmed: no `amdgpu`
reset/hang/fault messages in `dmesg`, and kernels kept returning `valid:y` correctly
throughout). wrw's worst-case shape showed genuinely bimodal timing at a FIXED, pinned
split count (`IGEMM_GSPLIT_SWEEP=252`) -- roughly 0.085ms or 0.148ms depending on the run,
no in-between -- consistent with an external contention window intermittently overlapping
the kernel's dispatch. Under this noise, 8 runs each: **no-setprio landed in the fast mode
2/8 times (median 0.148ms); setprio landed in the fast mode 6/8 times (median 0.086ms)**
-- a directionally encouraging, ISA-doc-consistent result (wrw's split-K workgroups are
exactly the "single wave per SIMD" low-occupancy regime the doc says this helps), but
**explicitly not treated as a confirmed win given the contention severity and small
sample** -- needs re-measurement on an idle GPU before being relied on. fwd (a well-
occupied kernel, the regime the ISA doc warns setprio could instead HURT by blocking
co-execution) showed no measurable difference either way (0.022-0.024ms both configs,
well within noise) -- no detected regression, but also not a clean enough measurement
environment to rule one out definitively.

**Net result**: shipped as an opt-in, default-off, correctness-verified tunable. The
performance verdict is genuinely unresolved pending a clean re-measurement -- ship the
mechanism now (zero risk, real cross-project validation of the technique existing), defer
the "should this be on by default" decision.

**Critical files (Phase 32)**: `python/igemm/igemm_base.py` -- `wmma_setprio` tunable +
kernel-name fold; `python/operations/wmma_main_loop.py` -- `ctrl.wmma_setprio`,
`emit_wmma_tile()` bracketing; `python/igemm/igemm_{fwd,bwd,wrw}_gtc_wmma_nhwc.py` --
`ctrl.wmma_setprio` wiring; `driver/igemm_gtc_base.h` -- struct field, config parsing,
kernel-name fold; `config/igemm_{fwd,bwd}_gtc_gfx1250_nhwc_bf16_setprio.config`,
`config/igemm_wrw_gtc_gfx1250_nhwc_bf16_gsplit_setprio.config` -- new.

## Phase 33 (2026-08-27): wrw split-K heuristic cross-check

Second of three Tier 1 items from `docs/gfx1250_perf_parity_action_plan.md`. Cross-checks
MISA's existing ternary search over split counts (Phase 20) against CK's closed-form
split-K occupancy formula (`num_cu * max_occupancy_per_CU / grid_size`) by feeding it as
one more candidate into the search's own divisor list.

**Discovery**: MISA already had this exact formula shape --
`driver/igemm_wrw_gtc_driver.h`'s `compute_gemmk_global_splits(grid_size,
potential_occupancy)` (`num_cu * potential_occupancy / grid_size`), inherited from the
older XDLOPS/DLOPS track, which calls it with a hardcoded `potential_occupancy=3`. The
WMMA path's own `run()` never called it at all. Rather than writing new formula logic,
this phase wires the EXISTING function into the WMMA path with a REAL occupancy value:
`hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(kernel_func, block_size, 0)` --
querying the actual compiled kernel's occupancy (VGPR/LDS-limited), not a guessed
constant. This is a legitimate implementation of CK's formula using MISA's own
pre-existing infrastructure, not new math.

**Design**: right after `kernel_func`/`block_size` become available in the WMMA `run()`
method (gated behind `tunable->gemm_k_global_split`, so non-split builds are completely
untouched), compute the heuristic split count, snap it to the nearest valid divisor of
`num_k_blocks` via the existing `largest_divisor_leq` helper (the same snapping the
`IGEMM_GSPLIT_SWEEP` override already uses), and insert it into the `divisors` vector
(de-duplicated, re-sorted) BEFORE the ternary search's cache/eval loop runs. **Strictly
additive and non-regressive by construction**: the ternary search still evaluates every
candidate it always would, plus this one; its own min-of-all-evaluated logic already
handles a bad candidate correctly (never selects it) at the cost of one extra real timed
launch per search.

**No regression sweep needed**: this change is entirely host-side C++ search logic
(`driver/igemm_wrw_gtc_driver.h`) -- it does not touch Python codegen, so the compiled
kernel binaries for every config, gsplit or not, are byte-identical to before. Confirmed
by inspection (the new code only runs inside the WMMA `run()` method's dispatch path,
never `igemm_codegen.py`'s kernel-generation path) and by the non-gsplit sanity check
below (matches the pre-Phase-17 gsplit catastrophe numbers exactly, confirming that path
is untouched).

**Hardware validation**: `valid:y` across wrw's exact-multiple large shape, a 3x3
multi-tap shape, group>1, and the benchmark doc's worst-case outlier
(`c=192,H=60,W=80,k=64,1x1`) -- correctness fully intact. Non-gsplit wrw config spot-
checked separately: 17.6ms for `c=64,k=128` (matches the original pre-gsplit-fix
catastrophe from `docs/gfx1250_vendor_benchmark_vs_miopen.md` almost exactly), confirming
the new code path genuinely never executes when `gemm_k_global_split=0`.

**Timing**: same severe contention as Phase 32 (see that phase's writeup). Measured
"does the search land in the known-fast split range (~0.065ms) or the known-slow range
(~0.148ms)" as a success-rate proxy, since raw means are unreliable under this session's
bimodal noise: pre-phase driver landed in the fast range 3/8 runs; with the heuristic
candidate wired in, 4/8 runs. A modest, directionally consistent improvement, but not a
statistically strong result at this sample size under this much external noise --
consistent with Phase 32's own honest assessment, needs re-measurement on an idle GPU.
Unlike Phase 32, though, this change carries no performance-regression risk even if the
improvement doesn't hold up: it can only ever add a candidate, never remove one.

**Net result**: shipped. Zero regression risk (host-only, additive-only), one legitimate,
real occupancy query replacing a previously-unused hardcoded constant from a different
code path. Performance verdict: modest positive signal, unconfirmed under current
contention, safe either way.

**Critical files (Phase 33)**: `driver/igemm_wrw_gtc_driver.h` -- new heuristic-candidate
block in the WMMA `run()` method, reusing the existing `compute_gemmk_global_splits`/
`largest_divisor_leq` helpers with a real `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`
query.

## Phase 34 (2026-08-27): packed 2-wide bf16 atomics for wrw's split-K epilogue

Third and last of the three Tier 1 items from `docs/gfx1250_perf_parity_action_plan.md`.
Both rocKE and FlyDSL pack two adjacent bf16 partial-sum elements into one 32-bit lane
before issuing `global_atomic_pk_add_bf16`, halving the atomic instruction count (and the
memory traffic each one generates) versus wrw's existing plain scalar-fp32 atomic epilogue.
Full-implementation scope, at the user's explicit direction, rather than a probe-first
partial pass.

**Design**: new `atomic_pack_bf16` tunable (default 0, asserts `gemm_k_global_split` set,
`precision=='bf16'`, `not atomic_cascade`). Per adjacent-lane pair: `ds_bpermute_b32`
exchanges each lane's fp32 partial sum with its XOR-1 partner, `v_cvt_pk_bf16_f32` packs
own-value (lo16) + partner-value (hi16) into one VGPR, then a parity-narrowed
`global_atomic_pk_add_bf16` (only even lanes issue, each covering its own+partner's output
slot) writes the packed pair in a single 2-wide atomic. Because the packed atomic already
produces final bf16 values directly, no separate cast kernel is needed (unlike
`wmma_acc_f16`/`wmma_acc_bf16`, which still need one). Register reuse: `v_gather`/
`v_tmp3`/`v_tmp4` were considered for the exchange/pack scratch (already unused whenever
`gemm_k_global_split` is set, since they exist for `wmma_m_tail`/`wmma_n_tail`, asserted
mutually exclusive with gsplit) but the existing non-packed call site passes `v.v_c()` as
a dummy placeholder for `v_gather` -- harmless today, but would silently corrupt the
accumulator if reused here -- so 3 new dedicated VGPRs (`v_pk_idx`, `v_pk_partner`,
`v_pk_packed`) were allocated instead, gated behind `atomic_pack_bf16`.

**Two real bugs found and fixed during hardware bring-up**:
1. **Missing `s_wait_dscnt`**: `ds_bpermute_b32` is a DS-class instruction tracked by
   `DSCNT` (confirmed in the CDNA5 ISA doc), but its result was consumed by the immediately
   -following `v_cvt_pk_bf16_f32` with no wait -- a genuine race, silently masked on most
   loop iterations by incidental issue-latency, causing only the first couple of K-split
   iterations to actually corrupt (nrms ~0.17-0.20 on affected shapes). Fixed by emitting
   `s_wait_dscnt 0x0` between the two instructions. Diagnosed via the driver's existing
   `PER_PIXEL_CHECK`/`PRINT_NRMS` infrastructure, not the standalone hardware probes used
   earlier in the investigation (those hit their own, unrelated raw-inline-asm artifact
   that turned into a debugging detour -- see the probe notes below).
2. **`out_elem_byte_shift`/`out_elem_byte_shift_group` didn't account for the new bf16
   -native (2-byte) output width**: both existed already, gating fp32 (shift=2) vs
   `wmma_acc_f16`/`wmma_acc_bf16`'s narrower buffer (shift=1), but neither checked the new
   `atomic_pack_bf16` case, which is a THIRD, unrelated mechanism that also produces a
   2-byte-native output buffer. Silently masked for group=1/1x1 shapes (the multiplied
   quantities being shifted are zero regardless of the shift constant), but caused real
   corruption for group>1 (nrms 0.165, `valid:n`) and multi-tap 3x3 (NaN, from a
   badly-offset tap address) -- both only surfaced once those shape categories were tried
   for the first time. Fixed by adding `or self.tunable.atomic_pack_bf16` to both
   conditions in `igemm_wrw_gtc_wmma_nhwc.py`.

**Debugging note on standalone probes**: a probe using raw inline asm (`asm volatile(...)`)
for `ds_bpermute_b32` gave wrong results even in isolation, while an equivalent probe using
the `__builtin_amdgcn_ds_bpermute` compiler builtin worked correctly for the SAME
instruction -- a reminder that hand-written inline asm in a C++ probe can have its own
register-allocation/scheduling artifacts distinct from MISA's actual hand-assembled `.s`
output, and isn't a substitute for testing the real generated kernel.

**Byte-identical regression check**: confirmed for a representative pre-existing config
(`igemm_wrw_gtc_gfx1250_nhwc_bf16_gsplit.config`, plain non-packed) -- both `.inc` files
regenerate byte-for-byte identical to pre-phase HEAD; the only diff anywhere was the
disassembler's embedded source-path comment in `.disass.s`. Expected and safe: both bug
fixes are additive `or self.tunable.atomic_pack_bf16` clauses on conditions that were
already false for every config not opting in.

**Hardware validation**: `valid:y`, low nrms (~0.0007-0.006) across every shape category
tried post-fix: 1x1/group=1, group=2, and 3x3 multi-tap/group=1 and group=2 -- the exact
categories that had exposed the two bugs above now pass cleanly.

**Timing**: mixed, not a clean win. On wrw's documented worst-case shape
(`c=192,H=60,W=80,k=64,1x1`, the same shape used in Phases 32/33), the packed-atomic
64x64 kernel was consistently **slower** than the plain scalar-fp32 epilogue across 6 runs
each (~0.032-0.034ms vs ~0.022-0.023ms, a clean ~40-50% gap, well outside this session's
usual contention noise band) -- the opposite of the expected direction. On a smaller
group=2 shape the result was mixed instead: the 64x64 kernel was faster packed
(~0.010-0.011ms vs ~0.013-0.014ms) while the 128x128 kernel was slightly slower
(~0.020-0.024ms vs ~0.017ms). Two plausible, non-exclusive explanations: (a) the packed
path costs 3 extra VGPRs (174 vs 171 for the 64x64 kernel) which can cross an occupancy
-affecting allocation boundary depending on shape/launch geometry; (b) more fundamentally,
`docs/gfx1250_rocprof_profiling.md`'s own rocprof measurement on this exact worst-case
shape found `TX_VMW_ATOMIC_SETCONFLICT_STALL` at exactly **zero** -- i.e. atomic-address
contention was never actually the bottleneck for MISA's wrw kernel on this hardware, so
halving atomic traffic doesn't buy back the added cross-lane-exchange ALU cost
(`ds_bpermute_b32` + `s_wait_dscnt` + `v_cvt_pk_bf16_f32`) the way it evidently does for
rocKE/FlyDSL's own pipelines.

**Net result**: shipped as an opt-in, default-off, now fully correctness-verified tunable
-- the mechanism works and matches its source projects' design, but is not a confirmed
performance win for MISA's current wrw kernel shape/occupancy profile, and measured
slower on the profiled worst-case shape specifically. Recommendation: leave off by default;
revisit only if a future wrw redesign changes the occupancy/VGPR balance enough to make the
atomic-bandwidth halving pay for its own overhead, or if a workload is found where atomic
conflicts (unlike this session's profiled shape) are a genuine measured bottleneck.

**Critical files (Phase 34)**: `python/operations/coalescing_store_wmma.py` --
`atomic_pack_bf16` field, packed-atomic branch (exchange + pack + narrowed atomic, with the
`s_wait_dscnt` fix); `python/igemm/igemm_base.py` -- tunable + asserts + kernel-name fold
`_pkatomic`; `python/igemm/igemm_wrw_gtc_wmma_nhwc.py` -- dedicated VGPRs, wiring, the
`out_elem_byte_shift`/`out_elem_byte_shift_group` fix; `driver/igemm_gtc_base.h` -- struct
field, config parsing, kernel-name fold; `driver/igemm_wrw_gtc_driver.h` -- gsplit
zero-init size fix (`gsplit_zero_elem_byte`); `driver/conv_driver.cpp` --
`is_wmma_atomic_pack_bf16` dtype/verification wiring (reads bf16 directly, no cast kernel);
`config/igemm_wrw_gtc_gfx1250_nhwc_bf16_gsplit_pkatomic.config` -- new.

## Phase 35 (2026-08-27): wrw GEMM_M/N/K tail relief, hipconv-style reduction-kernel
epilogue, and tile widening -- closing wrw's coverage/performance gap vs gfx950/XDLOPS

The three highest-confidence items from `docs/gfx1250_perf_parity_action_plan.md`, done
together in one continuous push per explicit user direction: wrw's WMMA path had ZERO
boundary/tail handling (gemm_m, gemm_n, gemm_k all had to be exact tile multiples --
the single biggest shape-coverage gap vs fwd/bwd and vs the mature XDLOPS track), no
alternative to atomic-accumulate for its split-K epilogue, and only two `gemm_k_per_block`
values (32, 64 via `_k2x`) versus CK's instance library going up to 256.

### Mechanism 1+2: wrw M-tail and N-tail

Ported fwd/bwd's existing `wmma_m_tail`/`wmma_n_tail` pattern to wrw, widening each
tunable's direction gate in `python/igemm/igemm_base.py`. The key deviation from
fwd/bwd's precedent: wrw's `gemm_k_global_split` (atomic split-K) is its **primary** path,
not an edge case, so unlike fwd/bwd (which simply exclude tail tunables from split-K),
wrw's M/N-tail had to mask BOTH the non-atomic epilogue (already generic/shared code,
needed no changes) AND the plain-atomic epilogue (new code, `coalescing_store_wmma.py`'s
`elif ctrl.gemm_k_global_split:` branch) from day one. Turned out simpler than feared:
that branch is already scalar-per-element (no `vector_write_out=4` grouping), so fwd's
own N-tail bug class (a vectorized-store-group straddling the tail boundary) can't recur
there -- only the non-atomic path still needs `gemm_n % 4 == 0` (mirrored exactly from
fwd's existing driver check).

**Real bug found and fixed via hardware testing**: the very first N-tail test (`c=129`,
otherwise exact) failed with widespread small-magnitude corruption across the ENTIRE
output, not just the tail boundary -- traced to forgetting to add wrw's own
`gemm_n % 4 == 0` restriction (mirroring fwd's identical, already-documented
`igemm_fwd_gtc_driver.h` requirement) to `igemm_wrw_gtc_driver.h`'s `tunable_is_valid()`
for the non-atomic path specifically (the atomic path's per-element masking has no such
restriction). Once added, all shapes passed.

**Hardware validation**: full degenerate/one-short/one-over/exact-noop battery on both
the non-atomic and atomic epilogues, group>1, and M+N combined -- all `valid:y`. Byte-
identical regression confirmed for every pre-existing config (9 configs spot-checked
during this mechanism, full 104-config sweep at the end of this phase).

### Mechanism 3: wrw K-tail -- genuinely new, no precedent anywhere in this codebase

Unlike M/N-tail (a direct port), K-tail had zero prior art: the only existing WMMA
K-boundary mechanism is Phase 31's TDM-hardware-OOB zero-fill (fwd, 1x1-only, relies on
gfx1250's `tensor_load_to_lds` own OOB behavior -- inapplicable to wrw, which doesn't use
TDM). Verified directly against the code (not assumed) that this needed far less new
machinery than expected: `wmma_main_loop.py`'s main loop is a signed decrement-until-
non-positive loop (`s_kitr = s_knum; ...; s_kitr -= unroll_k; s_cbranch_scc0 _last`) that
already tolerates any `s_knum`, and `_emit_gld_chunk_load` already zero-inits its
destination before a masked load. So K-tail reduced to: (a) get `s_knum` correct per
split-K shard, (b) mask the final iteration's out-of-range lanes.

**(a)**: two new kernarg fields (`gemm_k_tail` = the remainder, `gemm_k_num_splits` = the
launched `grid.z`), gated to only exist in the kernarg layout when `wmma_k_tail` AND
`gemm_k_global_split` are both set (a plain non-split K-tail build needs neither -- `s_knum`
is already exactly `s_gemm_k` in that case, zero code change). Only the LAST split-K shard
(`bz == gemm_k_num_splits-1`) gets its `s_knum` extended by the remainder -- shard bases
stay exact multiples and contiguous by construction, so this is the only place a gap could
exist.

**(b)**: B's `_emit_b_gather` already computes the global absolute `k_abs` before decomposing
it -- one more `v_cmp_gt_u32`/`v_cndmask_b32` there, ANDed into the existing per-iteration
`v_flag`. A had no per-iteration flag mechanism at all (its GEMM_M flag is a true kernel-
lifetime constant); added a new `_emit_a_kflag` mirroring `_emit_b_gather`'s dual-call-site
pattern (tap start + `move_slice_window_a_functor`), reusing the SAME persistent
`v_row_local` value B already maintains (A and B share the same K-position formula).

**Real bug found and fixed via hardware testing (the significant one)**: initial
implementation stored B's K-tail flag in `v_tmp(1)` (presumed free scratch). Multi-tap
shapes (and even single-tap shapes once M/N were also active) showed widespread, small-
magnitude corruption -- eventually isolated (by selectively disabling the A-side and then
the B-side masking independently) to `v_tmp(1)` specifically. Root cause: the
`.v_u32_div_rem_vs_gfx1250` macro used immediately afterward (to decompose `k_abs` into
`n_idx`/`ho_idx`/`wo_idx`) clobbers **all four** `v_tmp` registers internally as part of
its own division algorithm (verified by reading the macro's expansion) -- not just
`v_tmp(0)` as assumed. Fixed by allocating a genuinely dedicated register
(`v_flag_b_ktail`) instead of reusing `v_tmp` scratch. A lesson for this codebase generally:
`v_tmp` is only safe as scratch immediately before a division macro call, never for a value
that needs to survive past one.

**Hardware validation**: non-split (the cheapest, zero-code-change case) and split-K
(exercising the new kernarg/last-shard logic) each got degenerate/one-short/one-over/
exact-noop, forced non-power-of-2 split counts (2, 3, 7-snapped-to-3), group>1, and
multi-tap (3x3) -- all `valid:y` after the fix. Full M+N+K combination also validated
(64x64 tile only -- see VGPR note below).

### VGPR ceiling: not every tail combination fits at 128x128

Combining all three tails (M+N+K) with `gemm_k_global_split` at the 128x128 tile overflows
the 256-VGPR hard limit by exactly 1 (257 used) -- a real hardware ceiling, not a bug, same
class of wall bwd's plain `wmma_m_tail` alone already hit historically. 64x64 has ample
headroom (177/256). Any TWO of the three tails combine fine at 128x128 (validated
separately); the fully-combined config (`_gsplit_mnktail`) ships 64x64-only, documented
in the config file itself.

### Mechanism 4: hipconv-style reduction-kernel epilogue

New `wrw_reduction_kernel` tunable (name-folded as `_wsred` -- unlike the pure-masking tail
tunables, this one changes what the kernel's epilogue produces, the same class of change
Phase 16 established requires the two-sided driver/Python name-fold treatment). Per
`docs/gfx1250_hipconv_deep_dive.md`: instead of every split-K shard atomic-adding directly
into the real output, each shard does a plain, non-atomic store into its own disjoint slice
of a workspace buffer (`num_partitions x output_size`, sized for the largest split count
the ternary search could try), then a new, plain HIP kernel (`wrw_reduce_partials_f32`,
added to `driver/gpu_tensor_cast/gpu_tensor_cast.cpp` -- a simple grid-stride sum over
partitions, reusing the existing `module_tensor_cast` infrastructure) sums the partitions
into the real output as a second, separately-timed dispatch.

**Design**: the main kernel's own codegen needed surprisingly little new code --
`coalescing_store_wmma.py`'s existing atomic branch (row/col address computation + M/N-tail
masking, all reused unchanged) just swaps `global_atomic_add_f32` for a plain
`global_store_dword` when `wrw_reduction_kernel` is set. The shard's disjoint workspace
offset (`bz * group*gemm_m*wei_row_c * 4 bytes`) is added to `s_p_out` ONCE in the prologue
(mirroring the existing group-offset pattern), so every tap's `s_p_out_tap` (recomputed
fresh FROM `s_p_out`) automatically inherits it -- no epilogue-side changes needed for
shard addressing at all. Kept fp32-only (asserted mutually exclusive with
`wmma_acc_f16`/`bf16`/`atomic_pack_bf16`) to avoid a byte-width combinatorial headache,
matching the workspace's fixed-fp32 design. Driver-side, `conv_driver.cpp` needed **zero**
changes -- from its perspective this looks exactly like a plain, unspecialized `is_wmma`
wrw kernel (fp32 output, no special dtype casing), since the workspace redirection is
entirely internal to `igemm_wrw_gtc_driver.h`'s own `run()`.

**Real bug found and fixed via hardware testing**: the reduction kernel's own launch
failed with `hipExtModuleLaunchKernel(...)`: `invalid argument` on any shape with
`output_size` large enough that `ceil(output_size/256)` (a **block count**) itself wasn't a
multiple of 256. Root cause: `igemm_launch_kernel_single`'s `grid_size` parameter is
documented (in its own header comment, missed on first read) to be in **workitem units**
(total threads), not blocks, and `hipExtModuleLaunchKernel` requires it to be an exact
multiple of `block_size` -- passing a bare block count only "worked" by accident for small
shapes (values under 256 apparently get silently clamped to one full block; the failure
threshold was empirically bisected to somewhere in (256, 320]). Fixed by rounding
`output_size` up to the next multiple of 256 directly (`(output_size+255)/256*256`) and
passing THAT as `grid_size`, matching the existing `tensor_cast` kernels' own identical
rounding convention (their `thread_length_cast` calculation does the same up-front
rounding before being used directly as a workitem count).

**Hardware validation**: basic 1x1, group>1, multi-tap (3x3, the shape that originally
exposed the launch bug), forced non-power-of-2 splits, and a larger worst-case-style
shape -- all `valid:y` after the fix.

**Timing -- the strongest result of this whole phase**: on wrw's documented worst-case
shape (`c=192,H=60,W=80,k=64,1x1`, same shape used in Phases 32-34), the reduction-kernel
epilogue measured **~0.011ms vs the plain atomic epilogue's ~0.022-0.023ms -- a consistent,
reproducible ~2x speedup**, with none of the atomic path's usual bimodal noise (three
repeated runs all landed within 0.011-0.011ms, vs the atomic path's own 0.022/0.022/0.051ms
spread on the same shape under the same contention). This does not contradict Phase 34's
rocprof finding of zero atomic-*conflict* stalls -- conflict-freedom doesn't mean zero
atomic overhead; removing the atomic RMW round-trip entirely (not just avoiding conflicts
on it) is evidently a real, separate win for this kernel's profile.

### Mechanism 5: tile widening (`gemm_k_per_block=128`)

Config-only -- `gemm_k_per_block` was already a fully generic tunable, and the k-sub-loop
already generic over its value (only `inst_wmma.k`, fixed at 32, and tile shape affect
`v_a`/`v_b`/`v_gld_a`/`v_gld_b`/`v_c` sizing, confirmed by reading `__init__`, not assumed).
New `_k4x` configs (128x128, `gemm_k_per_block=128`, single-buffered -- LDS lands exactly at
the 64KB/workgroup limit, no room for double-buffering) for plain and `bf16acc` bf16.
Plain fits at 251/256 VGPRs (no accumulate-precision trick even needed); `bf16acc` fits at
187/256 (extra headroom, shipped as a second precision/VGPR-tradeoff point mirroring the
existing `k2x_bf16acc` precedent).

**Hardware validation**: exact-multiple, group>1, and 3x3 multi-tap, both variants -- all
`valid:y` (bf16acc shows the expected higher-but-still-passing nrms from its reduced
precision).

**Timing**: on a small single-K-block shape, `_k4x` was actually slightly SLOWER than
`_k2x` (a wash within noise) -- expected, since that shape isn't wrw's actual failure mode
(plenty of workgroups, not occupancy-starved). On the ACTUAL intended failure mode (a
single 128x128 workgroup doing one long, serial K-reduction, `gemm_k=131072`), `_k4x`
showed a real, reproducible **~5% win** (13.14ms vs `_k2x`'s 13.86ms, consistent across
repeated runs) -- confirming CK's own rationale (fewer, cheaper main-loop iterations
amortize per-iteration overhead specifically when occupancy is already saturated by a
single serially-bound workgroup).

### Final regression sweep

Byte-identical `.inc` diff across all 104 pre-existing gfx1250 configs (fwd/bwd/wrw x
fp16/bf16/fp32/int8 x every existing tunable combination) confirmed clean -- every new
tunable in this phase defaults to 0 and gates 100% of new codegen; nothing pre-existing
changed by even one byte.

**Net result**: wrw's shape-coverage gap vs fwd/bwd (and vs the mature XDLOPS track) is
substantially closed -- M/N/K tail relief unlocks the same class of previously-unbuildable
shapes fwd's own tail work unlocked in Phases 25/26/31. The reduction-kernel epilogue is a
genuine, reproducible ~2x win on wrw's worst-case shape and is the standout result of this
phase; tile widening is a smaller but real ~5% win on its own specific failure mode. All
five mechanisms shipped as opt-in, default-off, fully hardware-validated tunables.

**Critical files (Phase 35)**: `python/igemm/igemm_wrw_gtc_wmma_nhwc.py` -- A/B load
functors, `_emit_b_gather`, `_emit_a_kflag` (new), `move_slice_window_a/b_functor`,
`emit_kernel_prologue` (M/N-tail flags, K-tail kernarg loads + `s_knum` extension, wsred
shard offset), `emit_kernel_tap_loop`, `kernel_vgpr_t`/`kernel_sgpr_t` (new dedicated
registers), `get_kernel_args`/`get_kernel_code` (K-tail kernarg fields); `python/operations/
coalescing_store_wmma.py` -- atomic-branch M/N-tail guards (new), plain-store branch for
`wrw_reduction_kernel` (new); `python/igemm/igemm_base.py` -- `wmma_m_tail`/`wmma_n_tail`
direction-gate widening, new `wmma_k_tail`/`wrw_reduction_kernel` tunables + asserts +
name-fold; `driver/igemm_gtc_base.h` -- `wrw_reduction_kernel` two-sided struct/config/
name-mirror; `driver/igemm_wrw_gtc_driver.h` -- `tunable_is_valid()` relaxes, K-tail
kernarg computation, workspace allocation + reduction-kernel launch sequencing, the
`gemm_n%4` non-atomic-path restriction; `driver/gpu_tensor_cast/gpu_tensor_cast.cpp` --
new `wrw_reduce_partials_f32` kernel; new configs: `igemm_wrw_gtc_gfx1250_nhwc_bf16_{mntail,
ktail,gsplit_mntail,gsplit_ktail,gsplit_mnktail,gsplit_wsred,k4x,k4x_bf16acc}.config`.

## Phase 36: bwd GEMM_N tail + GEMM_K tail

Per `docs/gfx1250_wmma_coverage_gap_analysis.md`'s real-corpus gap analysis, bwd was the
direction furthest behind on tail/boundary-mechanism completeness within the NHWC-only
subset -- M-tail only (Phase 26a), vs fwd's M+N+TDM-K and wrw's full M+N+K (Phase 35). This
phase adds bwd N-tail (`gemm_n = c/group`) and K-tail (`gemm_k = k/group`), completing bwd's
tail coverage. bwd never uses split-K/atomic accumulation at all (unlike wrw, where split-K
is the primary path), so neither mechanism needs any shard bookkeeping.

**A genuine surprise, found only by reading the actual per-lane addressing, not by analogy
to fwd's existing N-tail**: bwd's B (weight) operand is TRANSPOSED (see
`igemm_bwd_gtc_wmma_nhwc_t`'s class docstring -- this is fundamental to how bwd gets a
WMMA-consumable B operand at all). Concretely: fwd's B is natural `[K][N]` with each lane
owning one FIXED N-column and reading `gemm_k_per_block` elements CONTIGUOUSLY ALONG K --
so fwd's N-tail is a simple per-lane, whole-chunk EXEC-mask decision (this lane's column is
either in range or not, for its entire load). bwd's B is the opposite: each lane owns one
FIXED `row_local` (K-position) and reads `gemm_k_per_block` elements CONTIGUOUSLY ALONG N.
This inverts which axis is "easy" (a whole-lane EXEC decision) vs "hard" (a boundary that
can fall in the middle of ONE LANE's own multi-element load, which EXEC alone cannot
express -- EXEC only gates whole lanes, not sub-ranges within one lane's data):

- **B's K-tail is the easy case**: `row_local` (a persistent, prologue-computed VGPR,
  `v_b_row_local`) IS the per-lane K position for that lane's whole chunk -- a plain
  `v_cmp_gt_i32`/`v_cndmask_b32` EXEC-mask flag (`v_flag_b_ktail`), recomputed fresh in
  `global_load_b_functor` every call since the workgroup's K position (`s_kitr`) changes
  every K-loop iteration. Reuses `_emit_gld_chunk_load`'s existing zero-then-masked-load
  path unmodified, same idiom wrw's B K-tail (Phase 35) established.
- **B's N-tail is the hard case**: the N boundary can land inside one lane's own 32-wide (or
  wider) contiguous chunk. A has no N-axis role at all (A/grad_output only has M and K), but
  A's K-tail is ALSO the hard case for the identical structural reason (each lane owns one
  fixed M-row and reads `gemm_k_per_block` K-elements contiguously -- the K boundary can fall
  mid-chunk). Both needed a genuinely new masking primitive, not a port of any existing
  mechanism.

**New primitive: per-dword fine-grained AND-mask** (`_emit_tail_dword_mask` /
`_emit_tail_dword_mask_guarded` in `igemm_bwd_gtc_wmma_nhwc.py`), inserted between a chunk's
global-load-and-wait and its LDS store (`_emit_sst_remaining_chunks`/`_emit_sst_all_chunks`,
both extended with an optional `tail_mask` parameter). For each dword of the just-loaded
chunk: `valid_in_dword = clamp(remaining - elements_before, 0, elem_per_dword)` (computed via
`v_sub_u32`/`v_max_i32`/`v_min_i32` -- confirmed via `llvm-mc -mcpu=gfx1250` that a
scalar-SGPR-or-per-lane-VGPR source works identically for these VALU ops, so ONE instruction
sequence serves both K-tail's scalar `remaining` (`s_kitr`, uniform across every lane, since
K validity depends only on the workgroup-wide loop position) and N-tail's per-lane
`remaining` (`v_n_valid_base`, since N validity depends on each lane's own `col_group`).
`valid_in_dword`'s clamped range (0..`elem_per_dword`, at most 4) is turned into a byte-lane
keep-mask via one `v_cmp_ge_i32`/`v_cndmask_b32`/`v_or_b32` triple per possible element
count, then ANDed into the dword. Wrapped in a cheap 2-3-instruction scalar skip-branch
(`_emit_tail_dword_mask_guarded`) checking whether ANY masking is needed at all for this
particular chunk/lane-range -- keeps the ~10-15-instructions-per-dword cost off the hot path
for the overwhelmingly common non-tail case (every iteration except the actual tail one for
K-tail; every lane except those in the last, partially-valid `col_group` for N-tail).

**K-tail's `s_kitr` timing subtlety**: `wmma_main_loop.py`'s own `s_kitr = s_knum` init runs
too late for chunk 0's masking -- it's emitted right after the prologue's initial
`f_sst_a()`/`f_sst_b()` calls, which is exactly when chunk 0's mask needs a valid `s_kitr`.
Fixed with a redundant, harmless `s_kitr = s_knum` in `emit_kernel_tap_loop`, emitted before
the tap's first global loads (re-set moments later to the identical value by
`wmma_main_loop.py`'s own init). This reuses `s_kitr`/`s_knum` directly as the "remaining
valid elements from this tile's start" signal -- no new SGPR needed for K-tail's remaining
count (unlike wrw's K-tail, which needed new `gemm_k_tail`/`gemm_k_num_splits` kernarg
fields for its split-K shard bookkeeping; bwd needs neither, since `s_knum` is already the
true unpadded `gemm_k` with zero driver/kernarg change).

**N-tail's kernel-lifetime-constant timing**: unlike K-tail, N-tail's boundary
(`col_start_abs = block_n_off + col_start`, and derived `n_valid_base = gemm_n -
col_start_abs`) doesn't depend on the K-loop at all -- computed once in
`emit_kernel_prologue` (persisted out of the existing row_local/col_start computation,
non-destructively, right before each value would otherwise be consumed/overwritten by the
pre-existing `v_addr_b_base` address arithmetic) and reused unchanged by every
`shared_store_b_functor` call.

**Driver**: `driver/igemm_bwd_gtc_driver.h`'s `tunable_is_valid()` gains
`!wmma_n_tail`/`!wmma_k_tail`-gated relaxations of the `gemm_n`/`gemm_k` exact-multiple
checks, mirroring fwd's Phase 25/26b pattern -- including the same `gemm_n % 4 == 0`
restriction under `wmma_n_tail` (bwd shares fwd's exact vectorized-4-wide non-atomic
epilogue store via `coalescing_store_wmma.py`, so the identical hazard applies). No new
C++ struct/kernarg fields needed (`wmma_n_tail`/`wmma_k_tail` were already generic,
non-per-direction fields from wrw's Phase 35 work). `python/igemm/igemm_base.py`'s
`wmma_n_tail`/`wmma_k_tail` direction-gate asserts widened to include `'bwd'`.

**Epilogue**: bwd's non-atomic (LDS-reshuffle) epilogue already had a generic
`ctrl.wmma_n_tail` code path (added alongside fwd's Phase 26b, just never wired up for bwd)
-- `emit_kernel_epilogue` now sets `ctrl_coalescing_store_wmma.wmma_n_tail` and passes
`s_gemm_n`/a new `v_n_tail_col` scratch VGPR, identical to how M-tail's `s_gemm_m`/
`v_m_tail_row` were already wired. Zero epilogue-side code changes needed beyond that.

**VGPR cost** (64x64 tile, fp16/bf16): N-tail adds 3 VGPRs (`v_n_tail_col`,
`v_b_col_start_abs`, `v_n_valid_base`), K-tail adds 2 (`v_b_row_local`, `v_flag_b_ktail`).
Measured via `.vgpr_count` after generating (not assumed): N-tail alone 179, K-tail alone
178, all three tails (M+N+K) combined 182, k-sub-loop (`gemm_k_per_block=64`) + N-tail 179
-- all comfortably within the 256/wave limit at 64x64 (unlike wrw's Phase 35, which hit the
limit exactly at 128x128 combining all three tails with split-K).

**Zero-regression**: byte-identical `.s` diff across all 31 pre-existing gfx1250 bwd
configs (fwd/wrw untouched by this phase's files entirely) -- generated from a clean
`git worktree` checkout of the pre-phase commit vs the post-phase tree, confirming every
new tunable defaults to 0 and gates 100% of new codegen.

**Hardware validation** (`conv_driver.exe`, fp16 unless noted, against `naive_conv_bwd_nhwc`,
64x64x32 tile): N-tail alone -- `gemm_n` one-short/one-over/two-blocks-minus-4/exact-forced-on
(c=60/68/124/64, n=1,H=W=8,k=64, 1x1) -- **4/4 `valid:y`**. K-tail alone -- same battery on
`gemm_k` (k=28/36/60/32) -- **4/4 `valid:y`**. Combined M+N+K tail -- 1x1 with `gemm_m` also
one-short (n=1,H=9,W=7,c=60,k=28); 3x3 stride1/pad1 multi-tap (n=2,H=W=10,c=124,k=36); 3x3
stride2/pad1/dilation2 (n=1,H=17,W=13,c=68,k=60); exact-multiple sanity with all three tails
forced on (n=1,H=W=8,c=64,k=32) -- **4/4 `valid:y`**. Multi-chunk masking path
(`gemm_k_per_block=64`, `num_k_chunks=2`, N-tail) -- `gemm_n` one-short/two-blocks-minus-4
(c=60/124) -- **2/2 `valid:y`**. bf16 combined M+N+K tail (same shape as the 1x1 fp16 case
above) -- **1/1 `valid:y`**.

**One pre-existing, out-of-scope finding**: `group=2` on the 64x64 tile produces `valid:n`
even on the PLAIN (no new tunables set) existing `igemm_bwd_gtc_gfx1250_nhwc_fp16.config`'s
64x64 tunable -- confirmed pre-existing (this exact `.s` was byte-identical before/after
this phase's changes) and unrelated to N/K-tail specifically; not investigated further here,
flagged as a separate follow-up.

**New configs**: `igemm_bwd_gtc_gfx1250_nhwc_fp16_{ntail,ktail,mnktail,k2x_ntail}.config`.

**Critical files (Phase 36)**: `python/igemm/igemm_bwd_gtc_wmma_nhwc.py` -- new
`_emit_tail_dword_mask`/`_emit_tail_dword_mask_guarded` methods, `_emit_sst_remaining_chunks`/
`_emit_sst_all_chunks` (new `tail_mask` parameter), `global_load_b_functor` (K-tail flag),
`shared_store_a/b_functor` (tail_mask wiring), `emit_kernel_prologue` (persist
`v_b_row_local`/`v_b_col_start_abs`/`v_n_valid_base`), `emit_kernel_tap_loop` (redundant
`s_kitr` init), `emit_kernel_epilogue` (N-tail params), `kernel_vgpr_t`/`kernel_sgpr_t` (new
registers); `driver/igemm_bwd_gtc_driver.h` -- `tunable_is_valid()` relaxes; `python/igemm/
igemm_base.py` -- direction-gate widening for `wmma_n_tail`/`wmma_k_tail`.

## Phase 37: fwd GEMM_M/N tail via TDM's own hardware OOB (combined with `tdm_global_load`)

Per the coverage gap analysis, fwd's TDM-based K-tail (Phase 31, 1x1-only) and its
EXEC-mask-based M-tail/N-tail (Phases 25/26b) had never been combined into one buildable
config -- each worked independently, but `tdm_global_load`'s own load path bypasses the
functors `wmma_m_tail`/`wmma_n_tail`'s EXEC-mask flags live in entirely (confirmed by
reading `global_load_a/b_functor`: the TDM branch never references `v.v_flag`/`v.v_flag_b`
at all), so setting all three together previously built but silently produced WRONG
numerics for any M/N-boundary element (the flags were computed but never consulted).

**Approach, per user decision ("probe first, then decide")**: rather than trying EXEC-mask
tricks against TDM's wave-wide, non-per-lane load model (architecturally impossible -- TDM
has no EXEC concept at all), extend TDM's OWN descriptor to cover the M/N axis the same way
Phase 31 already covers K -- `tensor_dim1` (gemm_m for A, gemm_n for B) had always been set
to the *absolute* gemm_m/gemm_n, never rebuilt relative to the block's own offset, unlike
`tensor_dim0` (which Phase 31 already rebuilds every iteration from `s_tdm_k_remain`).

**Ground truth before writing any code**: read the CDNA5 ISA doc's TDM section (10.11.2)
directly rather than assuming Phase 31's K-axis finding generalizes by analogy alone --
`global_addr`: "Global memory address of the **start of the tile within the tensor** (not
the start of the tensor)"; `tensor_dim[0-4]`: "Size of the tensor... **used for detecting
out-of-bounds**." Crucially, the D# descriptor has no field anywhere for an absolute tensor
origin -- only `global_addr` (already tile-adjusted per-call) and `tensor_dim` (compared
against tile-local iteration indices in the addressing formula, `Maddr = global_addr +
data_size*(y*tensor_dim0_stride + ...)` for `y` in `0..tile_dim1`). This means "OOB relative
to the current global_addr" isn't just Phase 31's empirical K-axis finding by coincidence --
it's the *only* semantics `tensor_dim1` can architecturally have, since the hardware has no
other reference point to compare against. `tensor_dim0` and `tensor_dim1` are the same
descriptor mechanism, just gating the X-loop vs the Y-loop of the same 2D tile fetch.

**Design**: new persistent SGPRs `s_tdm_m_remain`/`s_tdm_n_remain` (gated on
`tdm_global_load and wmma_m_tail` / `tdm_global_load and wmma_n_tail`), computed ONCE in
`emit_kernel_prologue` as `gemm_m - block_m_off` / `gemm_n - block_n_off` -- unlike
`s_tdm_k_remain`, these don't need per-iteration decrementing (the M/N block offset is fixed
for the whole kernel, only K advances through the main loop). `_emit_tdm_descriptor_setup_a/b`
and `move_slice_window_a/b_functor`'s existing `tensor_dim0`-rebuild code (which also re-OR's
`tensor_dim1`'s bits fresh each time, since they share packed SGPRs) now use
`s_tdm_m_remain`/`s_tdm_n_remain` in place of the absolute `s_gemm_m`/`s_gemm_n` whenever
`wmma_m_tail`/`wmma_n_tail` is set -- byte-identical for every existing TDM-only or tail-only
config, since both new SGPRs are only allocated/referenced when both tunables are set
together. **No driver change needed**: `tunable_is_valid()`'s three tail-tunable conditions
(`wmma_m_tail`, `wmma_n_tail`, `tdm_global_load`) were already fully OR-independent.
**No epilogue change needed**: `coalescing_store_wmma.py`'s existing M/N-tail EXEC-mask
guards (which prevent an out-of-bounds *store*, a completely separate concern from TDM's
load-side zero-fill) are keyed only on `wmma_m_tail`/`wmma_n_tail`, not the load mechanism --
already correct and unmodified.

**VGPR headroom, measured not assumed**: the combined `_tdm_mntail` config (128x128, fp16)
came in at **255/256 VGPRs** -- 3 more than the TDM-only baseline's 252, extremely tight
(1 register of headroom) but not overflowing. bf16 measured identically at 255/256. Flagged
here as a hard ceiling for this tile shape: any further per-lane state addition to this
exact config would overflow.

**Zero-regression**: byte-identical `.s` diff across all 46 pre-existing gfx1250 fwd configs
(clean `git worktree` checkout of the pre-phase commit vs. the post-phase tree) -- the new
SGPRs and codegen only exist when both `tdm_global_load` and `wmma_m_tail`/`wmma_n_tail` are
set together, a combination no pre-existing config uses.

**Hardware validation** (`conv_driver.exe`, fp16 unless noted, 128x128x32 tile, 1x1 conv --
`tdm_global_load` is 1x1-only): exact-multiple sanity with both tails forced on (`gemm_m`=
`gemm_n`=128) -- `valid:y`. Single-block partial M-tail (`gemm_m`=120), single-block partial
N-tail (`gemm_n`=124), and both together -- **3/3 `valid:y`** (this is the actual probe: had
`tensor_dim1`'s OOB semantics been anything other than relative-to-`global_addr`, these would
have shown corrupted or garbage results, not `valid:y`). Multi-block-then-tail (the more
representative real case -- full block(s) followed by one partial tail block, confirming the
"remaining" computation correctly reports MORE than the tile width for early, fully-valid
blocks): `gemm_m`=316 (3 blocks, last partial), `gemm_n`=196 (2 blocks, last partial), both
M+N spanning multiple blocks simultaneously (`gemm_m`=`gemm_n`=260) -- **3/3 `valid:y`**.
Degenerate one-over-an-exact-multiple (`gemm_m`=129) -- `valid:y`. `group=2` combined with
both tails short -- `valid:y`. bf16, same shape battery -- `valid:y`.

**Net result**: confirms the ISA-doc-grounded hypothesis on real hardware -- TDM's hardware
OOB zero-fill natively covers the M/N axis with the exact same mechanism Phase 31 already
uses for K, needing no EXEC-masking machinery at all (which would have been architecturally
impossible against TDM's wave-wide load model regardless). fwd now has a single config that
combines TDM's K-tail with M/N-tail, closing the gap the coverage analysis flagged, at the
cost of the tightest VGPR margin of any config in this codebase.

**New config**: `igemm_fwd_gtc_gfx1250_nhwc_fp16_tdm_mntail.config`.

**Critical files (Phase 37)**: `python/igemm/igemm_fwd_gtc_wmma_nhwc.py` -- `kernel_sgpr_t`
(new `s_tdm_m_remain`/`s_tdm_n_remain`), `emit_kernel_prologue` (their computation, before
each descriptor setup call), `_emit_tdm_descriptor_setup_a/b` and
`move_slice_window_a/b_functor` (use the new SGPRs in place of the absolute
`s_gemm_m`/`s_gemm_n` when tail is active).

## Phase 38: fwd GEMM_K tail for multi-tap convolutions (non-TDM)

The one remaining real gap from `docs/gfx1250_wmma_coverage_gap_analysis.md`: fwd's only
K-tail mechanism was TDM's hardware OOB (Phase 31/37), which requires `nxe=0` (1x1, unit
stride, no padding) by construction -- multi-tap convs (3x3, dilated, etc.) had no K-tail
path at all. This closes that gap with a genuinely new, software (non-TDM) mechanism.

**Structural finding, checked directly rather than assumed**: fwd's A (input) and B
(weight) operands are BOTH the "hard case" for K-tail -- each lane owns one fixed row (of
GEMM_M for A, GEMM_N for B) and reads `gemm_k_per_block` elements CONTIGUOUSLY in one shot
via immediate-offset chunking off a single address (`_emit_gld_chunk_load`). This is
structurally identical to bwd's A operand (Phase 36), *not* bwd's B or wrw's B (which have
genuine per-lane K-row addressing and could use a simple EXEC-mask flag) -- fwd's B is
natural/untransposed (confirmed: no `get_gemm_index_for_src_matrix_transposed` call
anywhere in this file), so there is no "easy" operand here at all. EXEC can't gate a
sub-range within one lane's own multi-element contiguous load, so both operands need the
same fine-grained per-dword AND-mask primitive Phase 36 built for bwd's A.

**Design**: `_emit_tail_dword_mask`/`_emit_tail_dword_mask_guarded` ported essentially
verbatim from `igemm_bwd_gtc_wmma_nhwc.py` into `igemm_fwd_gtc_wmma_nhwc.py` -- same
per-dword `clamp(remaining - elements_before, 0, elem_per_dword)` construction, same
scalar `s_kitr`-based "remaining" signal (uniform across every lane here too, for the same
reason bwd's K-tail needed none: K validity depends only on the workgroup-wide loop
position, never on which row a lane owns), same cheap skip-branch guard. Reuses `s_kitr`/
`s_knum` directly with **zero new SGPRs or VGPRs** -- masking scratch comes from the
existing `v_tmp(0..2)` pool, exactly like bwd's design. `_emit_sst_remaining_chunks` gained
an optional `tail_mask` parameter (both A's and B's call sites wire it in when
`wmma_k_tail` is set); `emit_kernel_tap_loop` gained the same redundant `s_kitr = s_knum`
re-init before each tap's first chunk load that bwd needed, for the identical timing reason
(`wmma_main_loop.py`'s own init runs one line too late for that tap's first store).

New tunable reuses the existing `wmma_k_tail` name (already used by wrw/bwd for their own,
differently-implemented K-tail mechanisms) -- asserted mutually exclusive with
`tdm_global_load` (TDM already owns K-tail for the 1x1 case this mechanism doesn't need to
cover) and with `row_repeat_a/b > 1` (untested combination, mirroring bwd's identical
guard). Composes with the existing `wmma_m_tail`/`wmma_n_tail` EXEC-mask mechanisms with no
interaction at all -- confirmed by construction (K-tail's post-load AND-mask runs
independently of M/N-tail's load-time EXEC zero-fill; a lane already zeroed by M/N-tail
just stays zero) and by hardware validation of the combined config.

**Driver**: `driver/igemm_fwd_gtc_driver.h`'s `gemm_k % gemm_k_per_block` exact-multiple
check now also relaxes under `wmma_k_tail`, mirroring the existing `tdm_global_load`
relax (the two are OR'd, not overlapping, since they're mutually exclusive tunables).

**Zero-regression**: byte-identical `.s` diff across all 49 pre-existing gfx1250 fwd
configs (clean `git worktree` checkout of the pre-phase commit vs. the post-phase tree).

**VGPR cost**: measured, not assumed -- `_ktail` alone: 252 (fp16/bf16), 180 (fp32),
identical to the equivalent non-tail 128x128 baseline (confirming the zero-new-register
claim). Combined `_mnktail` (K-tail + M-tail + N-tail together): 255/256 (fp16/bf16), 183
(fp32) -- same tight-but-fitting margin as Phase 37's TDM+M/N-tail combo.

**Hardware validation** (`conv_driver.exe`, 128x128 tile, against `naive_conv_fwd_nhwc`):
K-tail alone, fp16 -- 1x1 (`gemm_k` not a multiple of 32), 3x3 stride1/pad1, 3x3
stride2/pad1/dilation2, `group=2`, exact-multiple sanity (forced on), one-over-an-exact-
multiple -- **6/6 `valid:y`**. Combined M+N+K tail, fp16 -- 1x1 and 3x3 with all three
axes simultaneously non-exact -- **2/2 `valid:y`**. bf16, same 1x1/3x3/combined battery --
**3/3 `valid:y`**. fp32 (`gemm_k_per_block=4`, a genuinely different `elem_per_dword=1`
masking path), same battery -- **3/3 `valid:y`**.

**Net result**: this was the last real GEMM-shape/mechanism gap identified in
`docs/gfx1250_wmma_coverage_gap_analysis.md` -- fwd, bwd, and wrw are now all fully covered
(modulo depthwise, which turned out to have zero real occurrences in the corpus once a
classifier bug was fixed -- see that doc's update -- and degenerate non-conv shapes) across
the entire 95k-shape real corpus this analysis was built from.

**New configs**: `igemm_fwd_gtc_gfx1250_nhwc_{fp16,bf16,fp32}_{ktail,mnktail}.config`.

**Critical files (Phase 38)**: `python/igemm/igemm_fwd_gtc_wmma_nhwc.py` -- new
`_emit_tail_dword_mask`/`_emit_tail_dword_mask_guarded` methods (ported from bwd),
`_emit_sst_remaining_chunks` (new `tail_mask` parameter), `shared_store_a/b_functor`
(tail_mask wiring), `emit_kernel_tap_loop` (redundant `s_kitr` init); `driver/
igemm_fwd_gtc_driver.h` -- `tunable_is_valid()` relaxes; `python/igemm/igemm_base.py` --
direction-gate widening for `wmma_k_tail` (now `wrw`/`bwd`/`fwd`).

## Phase 39: master config files (one comprehensive per-direction/precision search, matching gfx950/942)

gfx950/942's real workflow (confirmed by reading `README.md` and `script/
gen_gfx950_conv_split_kernel.sh` directly, not from memory): one comprehensive `.config`
file per (direction, precision) with ~90-200 tunable sections spanning many tile shapes and
knobs, one `conv_driver.exe` binary, one brute-force search loop (`driver_mode_normal`,
literally commented `// bench all solutions` in `igemm_gtc_base.h`) that tries every
tunable and reports the single fastest. gfx1250's WMMA track had drifted from this during
this session's incremental mechanism-by-mechanism development: each new tail/relief/perf
tunable got its own narrow, single-mechanism config file (for isolating correctness while
building it, a reasonable choice at the time) -- but nobody had since assembled a
comprehensive file, so a user wanting "the best gfx1250 kernel for shape X" had to run each
config file's binary separately and compare results manually. This phase closes that gap.

**A real naming-collision problem, found immediately**: `wmma_m_tail`/`wmma_n_tail`/
`wmma_k_tail` (Phases 25/26b/35/36/38) were never folded into the kernel name (unlike
`tdm_global_load`/`lds_double_buffer`/etc., folded since Phase 16) -- harmless as long as no
two sections of the identical tile shape differing only in one of these ever needed to
coexist in one file, which was true right up until this phase tried to union everything
into one file. Fixed by extending Phase 16's exact same fold pattern
(`igemm_gtc_encode_kernel_name` in `igemm_base.py`, mirrored in `get_kernel_name()` in
`driver/igemm_gtc_base.h`) to these three tunables (`_mtail`/`_ntail`/`_ktail` suffixes).
Verified byte-identical instruction bodies for every existing config before/after (a
plain kernel-name substitution recovers an exact match) -- this is a pure rename, not a
behavior change, for every kernel that already existed.

**Two more collisions, found only once the actual union was attempted**: `local_prefetch_num`,
`epilogue_lds_pad`, and `atomic_scope` were *also* unfolded (by original, explicit design --
see the old comments this phase replaces in `driver/igemm_gtc_base.h`) and produced real
collisions once files differing ONLY in one of these (e.g. a plain vs `_lp2` section of the
identical tile shape) were unioned. Folded the same way (`_lp2`/`_ldspad`/`_scopedev`
suffixes) -- required adding these three as actual C++ struct fields (`driver/
igemm_gtc_base.h`) for the first time, since the driver previously never needed to know
their values at all (only Python's codegen used them). `atomic_scope` is stored as a
string with an explicit default member initializer (`"SCOPE_SYS"`) rather than relying on
`std::string`'s empty-string default, so a default-constructed `igemm_gtc_tunable_t{}`
(e.g. `driver_mode_heuristic`'s still-unimplemented `heuristic_select_kernel()` stub)
doesn't fold into a wrong, empty-scope-name suffix.

**A real driver bug, found only by the comprehensive search actually exercising it**:
`conv_driver.cpp` computes `is_wmma_f16_acc`/`is_wmma_bf16_acc`/`is_wmma_atomic_pack_bf16`
(and the derived `dtype_alloc_byte` output-buffer width) ONCE from `tunables[0]`, not
per-tunable inside the search loop -- since output buffers are allocated once, upfront,
before the search loop starts. Mixing an accumulate-width-variant kernel (`wmma_acc_f16`/
`wmma_acc_bf16`/`atomic_pack_bf16`, all of which use a narrower native output buffer) into
a file whose first tunable doesn't share that width silently corrupts verification for it
(confirmed: the exact same kernel reports `valid:y` run standalone, `valid:n` in the
combined file). gfx950/942 never had a per-tunable output-width concept at all, so this
never surfaced there. **Not fixed at the driver level** (would need buffer allocation
restructured to happen per-tunable, a substantial and riskier change) -- instead,
`script/build_gfx1250_master_configs.py` excludes any section setting one of these three
flags from the master file, with a clear comment explaining why. These remain fully usable
via their own existing individual config files.

**A second real, pre-existing bug, found by the same mechanism**: `tdm_global_load`'s
"1x1/unit-stride only" restriction was previously enforced ONLY at config-authoring time
(`igemm_base.py` asserts `nxe==0` on the tunable itself) -- nothing in
`tunable_is_valid()` checked that the RUNTIME-requested shape actually WAS a unit conv
before dispatching to a TDM kernel. A multi-tap/strided/padded/dilated shape would
previously run a TDM kernel anyway and silently produce `valid:n` instead of being
rejected as "not applicable" the way every other inapplicable combination already is --
never caught before because no test battery had previously run a TDM config against a
disqualifying shape it wasn't excluded from at the config level. Fixed by reusing the
already-computed `unit_conv` local in `driver/igemm_fwd_gtc_driver.h` (previously computed
but never consulted for the WMMA/TDM path): `if(tunable->tdm_global_load && !unit_conv)
return false;`.

**`script/build_gfx1250_master_configs.py`** (new): a pure text-level union tool, not a
code generator -- reads every `config/igemm_{fwd,bwd,wrw}_gtc_gfx1250_nhwc_{precision}_*.config`
file, extracts each `[igemm_..._gtc]` section verbatim, deduplicates by normalized content,
excludes accumulate-width-variant sections (see above), and writes
`igemm_{direction}_gtc_gfx1250_nhwc_{precision}_all.config`. No tunable values are invented;
re-run after adding a new gfx1250 config file to pick it up.

**Zero-regression**: byte-identical instruction bodies (post name-substitution) across all
176 pre-existing gfx1250 tunable sections -- 105 completely untouched (no folded flag set),
71 gained a name suffix from the new folds (expected, not a regression).

**Kernel counts per master file** (deduplicated, accumulate-width-variants excluded): fwd
15-17 kernels per precision, bwd 11-12, wrw 16-24 (wrw's split-K search multiplies its own
candidate count) -- smaller than gfx950's 194 (a much more mature, longer-tuned track with
far more tile-shape variants), but every one of this session's validated mechanisms is now
in the same searchable set.

**Hardware validation** (`conv_driver.exe`, `IGEMM_LOG_FASTEST_CONFIG=1`): ran the master
files against exact-fit shapes (many kernels apply, all `valid:y`, fastest correctly
identified), tail-requiring shapes (most report "not applicable", the tail-capable kernel(s)
correctly `valid:y`), and confirmed the two bugs above were real by reproducing them,
applying the fixes, and re-confirming clean results -- across fwd/bwd/wrw.

**Net result**: gfx1250 now has the same "run everything, get the true fastest" experience
gfx950/942 always had, for every mechanism validated so far in this codebase, plus two
previously-undiscovered correctness gaps closed as a direct side effect of exercising a much
broader kernel combination space than any single narrow config file ever had.

**Critical files (Phase 39)**: `python/igemm/igemm_base.py` -- `igemm_gtc_encode_kernel_name`
(new `_mtail`/`_ntail`/`_ktail`/`_lp{n}`/`_ldspad`/`_scopedev` folds); `driver/
igemm_gtc_base.h` -- `get_kernel_name()` mirror, new `local_prefetch_num`/`epilogue_lds_pad`/
`atomic_scope` struct fields + config parsing; `driver/igemm_fwd_gtc_driver.h` --
`tunable_is_valid()`'s new `unit_conv` check for `tdm_global_load`; `script/
build_gfx1250_master_configs.py` (new); `config/igemm_{fwd,bwd,wrw}_gtc_gfx1250_nhwc_
{fp16,bf16,fp32,int8}_all.config` (new, 12 files).

## Phase 40 (2026-08-27): `V_PERMLANE_XOR_B32` swap for Phase 34's cross-lane exchange

Small-effort item from `docs/gfx1250_optimization_backlog.md` Tier 1. Phase 34's
packed-bf16 atomic epilogue (`python/operations/coalescing_store_wmma.py`) exchanged each
lane's fp32 accumulator value with its column-adjacent partner (lane XOR 1) via
`ds_bpermute_b32` -- a DS-class (LDS-path) instruction, tracked by DSCNT, requiring a
`s_wait_dscnt 0x0` before its result could be safely consumed (a real hardware race,
found and fixed during Phase 34 itself). `V_PERMLANE_XOR_B32`, confirmed to assemble on
gfx1250 via `llvm-mc -mcpu=gfx1250` earlier this session, is a plain VOP3 VALU
instruction that performs the identical lane-XOR-mask exchange directly, with two
concrete advantages over `ds_bpermute_b32` found while reading the CDNA5 ISA doc's
cross-lane section (§7.2.7, §15.14) for this swap:

1. **No wait needed.** As a VALU op (not DS-class/DSCNT-tracked), its result is
   consumable by the very next instruction in program order -- eliminates the
   per-iteration `s_wait_dscnt 0x0`.
2. **No precomputed index register.** `V_PERMLANE_XOR_B32 D0, S0, S1, S2` takes the XOR
   mask (`1`) and lane-group-width (`32`, the whole wave) as immediate operands directly
   -- `ds_bpermute_b32`'s byte-index operand (`lane XOR 1`, then `<<2` for byte
   addressing) needed a dedicated kernel-lifetime VGPR (`v_pk_idx` in
   `igemm_wrw_gtc_wmma_nhwc.py`) computed once up front. Removed entirely, freeing 1
   VGPR for every `atomic_pack_bf16` kernel.

A third property, confirmed while re-reading the ISA doc's cross-lane section rather than
assumed: `V_PERMLANE_XOR_B32` "ignores EXEC for reads (fetch-invalid: act as if EXEC is
all ones)" -- unlike `ds_bpermute_b32`, whose read result for a disabled source lane is
undefined/zeroed (the reason Phase 34's comment insisted "the exchange needs FULL EXEC").
The new code needs no full-EXEC discipline around the exchange itself; EXEC is narrowed
to even-lanes-only exactly as before, but only for the pack+atomic step.

One documented hazard checked and confirmed not applicable: ISA doc §7.2.7, "V_PERMLANE*
may not occur immediately after a V_CMPX." In this loop, `global_atomic_pk_add_bf16` and
the EXEC-restore `s_mov_b32 exec_lo, -1` both sit between one iteration's
`v_cmpx_eq_u32` and the next iteration's `V_PERMLANE_XOR_B32` -- never adjacent.

### Changes

- `python/operations/coalescing_store_wmma.py`: `atomic_pack_bf16` branch's exchange
  sequence (`v_xor_b32`+`v_lshlrev_b32` precompute, `ds_bpermute_b32`, `s_wait_dscnt`)
  replaced with one `v_permlane_xor_b32 v[v_tmp3], v[v_c+c_index], 1, 32`. `v_gather`
  parameter no longer required for this branch (assert relaxed).
- `python/igemm/igemm_wrw_gtc_wmma_nhwc.py`: removed the now-unused `v_pk_idx` VGPR
  allocation; call site passes `None` for the `v_gather` slot.

### Verification

**Zero-regression**: every non-`atomic_pack_bf16` config path is untouched (the edited
code is entirely inside the `ctrl.atomic_pack_bf16` branch); `atomic_pack_bf16` itself is
only set by `config/igemm_wrw_gtc_gfx1250_nhwc_bf16_gsplit_pkatomic.config`.

**Hardware correctness** (`conv_driver.exe`, `convbfp16`, both tile shapes, against the
CPU reference): `n=42,c=192,H=60,W=80,k=64` (the 26x-slower-than-MIOpen worst-case shape
from the benchmark doc), `n=42,c=128,H=30,W=40,k=128`, `n=42,c=256,H=30,W=40,k=128`,
`n=8,c=64,H=16,W=16,k=64`, `n=1,c=32,H=4,W=4,k=32` -- every applicable kernel reports
`valid:y`.

**Timing** (best-of-repeat, GPU under contention -- see the standing caveat in
`docs/gfx1250_rocprof_profiling.md` -- so treat as directional, not precise): consistently
faster across every shape tried, roughly 4-9%: `c=192,k=64`: 0.094ms -> 0.090ms;
`c=128,k=128`: 0.065ms -> 0.061ms (128x128 tile), 0.051ms -> 0.048ms (64x64 tile);
`c=256,k=128`: 0.061ms -> 0.057ms (128x128 tile). Matches expectation: fewer instructions
per (i_rm, j, i_rn) iteration (2 setup instructions removed kernel-lifetime, 1 wait
removed per iteration, 1 DS-class op replaced by 1 cheaper VALU op).

**Critical files**: `python/operations/coalescing_store_wmma.py`,
`python/igemm/igemm_wrw_gtc_wmma_nhwc.py`.
