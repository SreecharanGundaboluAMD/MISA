# bwd `gemm_k_global_split` memory fault — investigation notes (R5)

**Status: RESOLVED.** Root cause found and fixed: `igemm_bwd_gtc_wmma_nhwc.py`'s
`get_kernel_args()` was missing the `gemm_k_per_wg` argument declaration at kernarg offset 88
from the PAL metadata `.args` array. The driver-side C++ karg struct
(`igemm_bwd_gtc_wmma_nhwc_karg_t`) and the kernel's assembly prologue (`s_load_dword` at
offset 0x58=88) both correctly referenced this field, but the `.hsaco`'s PAL metadata declared
no argument at offset 88 — creating a 4-byte hole that the ROCm runtime did not reliably
populate, causing the device to read stale/garbage data at that offset. fwd and wrw both
correctly declared this argument; bwd's comment said "always present" but the actual
`kas.append(amdgpu_kernel_arg_t('gemm_k_per_wg', ...))` call was missing. The one-line fix
(adding the missing `kas.append`) has been hardware-validated on all repro shapes and the
standing regression set. The `tunable_is_valid` rejection has been removed.

## Symptom

`igemm_bwd_gtcw_..._dstrb_gkgs` (bwd's WMMA `gemm_k_global_split=1` atomic-epilogue kernel)
reliably crashes with `HSA_STATUS_ERROR_MEMORY_FAULT` / `hipEventSynchronize` "illegal memory
access", or — depending on shape/scheduling — silently produces `valid:n` with an impossible
efficiency reading instead of crashing. Reproduced on:

- The original report shape (`n128 c1024 17x17 k1024`, 1x1, real split count 1).
- A trivial, exactly-tile-aligned shape (`n32 c128 16x16 k128`, 1x1, group=1) with both a
  real split count of 4 (heuristic default) and split count forced to 1
  (`IGEMM_GSPLIT_SWEEP=1`) — i.e. **the crash does not require real multi-way K-splitting**;
  it reproduces even when every workgroup's shard index (`bz`) is 0.
- fwd's identical mechanism (same kernarg struct shape, same load sequence, same tile size, a
  numerically identical GEMM shape for this specific test case) was verified `valid:y` at
  real split count 4 on the same hardware in the same session — this is not a generic
  gsplit/atomic-epilogue bug, it is specific to bwd.

## What was ruled out (with evidence)

All of the following were checked directly against the generated assembly and/or live
hardware state via `rocgdb` (`set amdgpu precise-memory on`, which is required —  the default
mode reports an imprecise, often-wrong faulting PC) and were confirmed **not** to be the bug:

1. **Host-side struct/offset mismatch.** `igemm_bwd_gtc_wmma_nhwc_karg_t` is a packed struct;
   `offsetof(gemm_k_per_wg) == 88`, `sizeof == 112`, verified with a standalone
   `offsetof`/`sizeof` probe — matches the Python codegen's kernarg offset exactly.
2. **Host value at the point of assignment.** A temporary `fprintf` right before
   `kernel_launchers.push_back(...)` confirmed `karg.gemm_k_per_wg == 32` (the correct
   per-shard K length) for the real-split-count-4 case, and `== 128` (correct, full
   `gemm_k`) for the forced-split-count-1 case, every time.
3. **`karg_size`/`HIP_LAUNCH_PARAM_BUFFER_SIZE` mismatch.** Confirmed `sizeof(karg) == 112`,
   matches `kernarg_segment_byte_size` in the kernel's own `.amdhsa_kernel` metadata, and the
   launch path (`igemm_launch_kernel_single`, shared byte-for-byte with fwd/wrw) passes it
   straight through.
4. **Register-allocation collision.** `s_gemm_k_per_wg` (s50) and every register that could
   plausibly alias it (`s_magic_hi_wi`=s24, `s_shift_pack`=s28, `s_tmp`=s52-55) were checked
   against the full `.set` list for the exact kernel that crashed — no collision.
5. **Missing/misordered `s_wait_kmcnt`.** The single `s_wait_kmcnt 0x0` after all 19 scalar
   kernarg loads (identical count and structure to fwd's, which works) precedes every use of
   `s_gemm_k_per_wg`. No SGPR is written between the load and first use.
6. **The K-loop trip count itself.** Hardcoding `s_knum` to the known-correct value (32,
   bypassing `s_gemm_k_per_wg` for the loop bound specifically) does **not** fix the crash —
   it only fixes the forced-split-count-1 case (where the shard-offset multiply
   `bz * gemm_k_per_wg` is `0 * garbage = 0`, harmless) and does *not* fix the real
   split-count-4 case, where a different (`bz>0`) workgroup's `gemm_k_wg_off = bz *
   garbage_gemm_k_per_wg` still corrupts its own A/B/output addressing regardless of the
   loop-bound hardcode. Both observations are consistent with a single root cause: garbage
7. **sseq() SGPR allocation order (recommendation #3 probe).** Moved `s_gemm_k_per_wg`'s
   `sseq()` declaration to be immediately adjacent to `s_group`'s in
   `igemm_bwd_gtc_wmma_nhwc.py`'s SGPR allocator, changing the register number from s50 to
   s42 (now consecutive with `s_group`=s41) and reducing the gap between the two kernarg
   loads at offsets 84+88 to zero. The assembler did **not** merge them into a single
   `s_load_dwordx2` (still two separate `s_load_dword` instructions), but the absolute SGPR
   number changed substantially (s50→s42, closer to fwd's s38). Both repro shapes (`n32 c128
   16x16 k128` and `n128 c1024 17x17 k1024`) still crashed with
   `HSA_STATUS_ERROR_MEMORY_FAULT` / `hipEventSynchronize` illegal-memory-access on the same
   hardware. The corruption at kernarg offset 88 is **not** caused by the absolute SGPR
   number or the allocation-order adjacency between `s_group` and `s_gemm_k_per_wg`. This
   rules out recommendation #3's structural-workaround hypothesis and further supports the
   suspicion that the bug is in the ROCm/HIP runtime kernarg-copy path or hardware
   kernarg-preload, not in this codebase's Python/C++ source.

## What was found (and not further explained)

Reading the live kernarg buffer directly out of device memory via `rocgdb` (`x/28xw` on the
address computed from `s0:s1`, the kernarg segment pointer) at the crash point shows **every
byte correct except kernarg offset 88**:

```
offset  0- 84: matches every host-side field exactly (pointers, gemm_m/n/k, hi_wi, wi,
               stride/pad/ho/wo/y/x/dilation/group)
offset 88    : 0x5f317831  -- WRONG. Host wrote 32 (0x00000020) here.
offset 92-108: matches every host-side magic-division field exactly
```

`0x5f317831`, read as little-endian ASCII bytes, is `"1x1_"` — a plausible fragment of this
exact kernel's own mangled name (`..._ta1x32x1x1_1x1x1x128_tb1x32x1x1_...` contains the
substring `1x1_`). This strongly suggests the corrupted 4 bytes are leftover data from
*something else* (very possibly a completely unrelated allocation, since the kernarg pool is
runtime-managed and not zeroed between uses) rather than a stack/pointer-lifetime bug on our
side — but this was not conclusively traced further. In particular:

- The struct layout, kernarg size, and the SGPR destination number are all provably
  identical in every property checked, yet fwd's byte-for-byte-analogous field at the same
  kernarg offset (88) is delivered correctly under the same test.
- The only two properties that differ between the (working) fwd kernel and the (broken) bwd
  kernel that were **not** exhaustively ruled out are (a) the *absolute* SGPR number assigned
  to the field by the two kernels' independent register allocators (fwd: s38, bwd: s50) and
  (b) total SGPR count (`.amdhsa_next_free_sgpr`: fwd 53, bwd 56) — i.e. this could plausibly
  be a HIP/ROCr kernarg-copy or hardware kernarg-preload interaction that depends on exact
  SGPR pressure or register number, not a bug in this codebase's Python/C++ source. This
  hypothesis was not verified further (would require ROCm runtime source-level debugging or
- The above hypothesis (a) — that the absolute SGPR number or total SGPR count was the cause
  — was ruled out by item 7 above (commit 6cd1ce7: moving s_gemm_k_per_wg from s50 to s42
  changed nothing). The true root cause was found by a different approach: comparing the two
  kernels' compiled `.hsaco` PAL metadata (see "Root cause" below).

## Root cause (FOUND)

The initial hypothesis was that gfx1250 hardware kernarg preloading metadata differed between
fwd and bwd. That hypothesis was **wrong** — neither kernel's `.amdhsa_kernel` descriptor in
the `.s`/`.inc` source contains any `.amdhsa_user_sgpr_kernarg_preload_length` or
`.amdhsa_user_sgpr_count` directives, and the compiled `.hsaco`'s PAL metadata contains no
preload-related fields either. Both kernels rely entirely on manual `s_load_b32`/`s_load_b128`
sequences in their prologues to read kernargs from the segment pointer in `s[0:1]`.

The actual root cause was a **missing argument declaration in the PAL metadata `.args` array**.

`igemm_bwd_gtc_wmma_nhwc.py`'s `get_kernel_args()` builds the list of kernel arguments that
gets emitted into the `.hsaco`'s PAL metadata (the `.args` array seen by
`llvm-readobj --notes`). This list had a 4-byte hole at offset 88: after `group` (offset 84),
it jumped directly to `magic_hi_wi` (offset 92), **omitting `gemm_k_per_wg` (offset 88)**.
The comment at that location said "Always present in the karg layout (even for non-split
kernels, which never read it) so both variants share one struct on the driver side — mirrors
wrw's identical field" — but the actual `kas.append(amdgpu_kernel_arg_t('gemm_k_per_wg', 4,
88, 'by_value', 'i32'))` call was never written. fwd (line 701) and wrw (line 646) both have
it; bwd was the only one missing it.

Evidence from `llvm-readobj --notes --elf-output-style=GNU` on the compiled `.hsaco` files:

**fwd (working)** `.args` array — correctly declares `gemm_k_per_wg` at offset 88:
```
      - .name:           group
        .offset:         84
        .size:           4
        .value_kind:     by_value
      - .name:           gemm_k_per_wg    ← PRESENT
        .offset:         88
        .size:           4
        .value_kind:     by_value
      - .name:           magic_0
        .offset:         92
```

**bwd (broken)** `.args` array — skips offset 88 entirely:
```
      - .name:           group
        .offset:         84
        .size:           4
        .value_kind:     by_value
      - .name:           magic_hi_wi      ← NO gemm_k_per_wg at offset 88!
        .offset:         92
```

Both kernels' assembly prologues load from offset 0x58=88 via `s_load_b32`:
```
fwd: s_load_b32 s38, s[0:1], 0x58     ; gemm_k_per_wg
bwd: s_load_b32 s50, s[0:1], 0x58     ; gemm_k_per_wg
```

The ROCm runtime uses the PAL metadata `.args` array to determine which kernarg dwords to copy
to the device. With no argument declared at offset 88, the runtime does not reliably populate
that dword — the device reads stale/garbage data (the `0x5f317831` / `"1x1_"` fragment observed
in the rocgdb session), exactly matching the symptom.

## Fix

One-line fix in `python/igemm/igemm_bwd_gtc_wmma_nhwc.py`'s `get_kernel_args()`: add the
missing `kas.append(amdgpu_kernel_arg_t('gemm_k_per_wg', 4, 88, 'by_value', 'i32'))` between
the `group` and `magic_hi_wi` entries, matching fwd and wrw. The `tunable_is_valid` rejection
in `driver/igemm_bwd_gtc_driver.h` has been removed.

## Hardware validation

After the fix, all previously-crashing shapes now report `valid:y`:

```
# Trivial tile-aligned shape, 128x128 tile, real split count 4:
$ IGEMM_RUN_ONLY_KERNEL=igemm_bwd_gtcw_nhwc_fp16_bx0_ex0_bt128x128x32_wt16x16_wr4x4_ta1x32x1x1_1x1x1x128_tb1x32x1x1_1x1x1x128_dstrb_gkgs \
  ./conv_driver.exe convfp16 -n 32 -c 128 -H 16 -W 16 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 \
  -l 1 -j 1 -g 1 -F 2 -V 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
[bwd: 0] ..._dstrb_gkgs[4], cost:0.022ms, tflops:12.228(2.12%), valid:y

# Original report shape (n128 c1024 17x17 k1024), real split count 1:
$ IGEMM_RUN_ONLY_KERNEL=..._128x128x32_..._dstrb_gkgs \
  ./conv_driver.exe convfp16 -n 128 -c 1024 -H 17 -W 17 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 \
  -l 1 -j 1 -g 1 -F 2 -V 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
[bwd: 0] ..._dstrb_gkgs[1], cost:0.400ms, tflops:193.809(33.61%), valid:y

# 64x64 tile, trivial shape:
$ IGEMM_RUN_ONLY_KERNEL=..._64x64x32_..._dstrb_gkgs \
  ./conv_driver.exe convfp16 -n 32 -c 128 -H 16 -W 16 -k 128 ...
[bwd: 0] ..._dstrb_gkgs[2], cost:0.069ms, tflops:3.869(0.67%), valid:y

# Forced split count 1 (IGEMM_GSPLIT_SWEEP=1):
$ IGEMM_GSPLIT_SWEEP=1 IGEMM_RUN_ONLY_KERNEL=..._128x128x32_..._dstrb_gkgs \
  ./conv_driver.exe convfp16 -n 32 -c 128 -H 16 -W 16 -k 128 ...
[bwd: 0] ..._dstrb_gkgs[1], cost:0.026ms, tflops:10.389(1.80%), valid:y
```

Standing regression set (from `docs/gfx1250_wmma_perf_report_v2.md`) — all bwd shapes pass
with zero `valid:n` or crashes across all kernel variants in the master fp16 config:
```
-n 256 -c 2048 -H 14 -W 14 -k 2048 -y 1 -x 1   # 12 kernels, all valid:y
-n 128 -c 1024 -H 17 -W 17 -k 1024 -y 1 -x 1   # 24 kernels, all valid:y
-n 64  -c 512  -H 28 -W 28 -k 512  -y 3 -x 3   # 22 kernels, all valid:y
-n 32  -c 256  -H 56 -W 56 -k 256  -y 3 -x 3   # 22 kernels, all valid:y
-n 128 -c 64   -H 56 -W 56 -k 64   -y 1 -x 1   # 24 kernels, all valid:y
```

## Reproduction (post-fix; no longer crashes)

```
$ python3 igemm_codegen.py config/igemm_bwd_gtc_gfx1250_nhwc_fp16_gsplit.config -d /tmp/x
$ IGEMM_RUN_ONLY_KERNEL=igemm_bwd_gtcw_nhwc_fp16_bx0_ex0_bt128x128x32_wt16x16_wr4x4_ta1x32x1x1_1x1x1x128_tb1x32x1x1_1x1x1x128_dstrb_gkgs \
  ./conv_driver.exe convfp16 -n 32 -c 128 -H 16 -W 16 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 \
  -l 1 -j 1 -g 1 -F 2 -V 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
# Expected: valid:y (pre-fix: HSA_STATUS_ERROR_MEMORY_FAULT or silent corruption)
```
