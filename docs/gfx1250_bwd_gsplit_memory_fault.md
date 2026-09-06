# bwd `gemm_k_global_split` memory fault — investigation notes (R5)

**Status: unresolved, mitigated by rejection.** `driver/igemm_bwd_gtc_driver.h`'s WMMA
`tunable_is_valid` now unconditionally rejects `gemm_k_global_split=1` for bwd. No bwd config
in this repo sets it, so this is a no-op today; it exists to stop a *future* config or driver
change from silently re-exposing a crash. Do not remove the rejection without either fixing
the root cause below or getting a second, independent hardware confirmation that it's gone.

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
   at kernarg offset 88 as read by the device, not a loop-logic bug.

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
  a minimal non-igemm repro, out of scope for this pass).

## Recommendation for whoever picks this up next

1. Build the smallest possible standalone repro: a HIP kernel with a >84-byte kernarg struct
   where a lone `s_load_dword` sits between a `group`-of-fields region and a `dwordx4`
   region, launched via the exact `HIP_LAUNCH_PARAM_BUFFER_POINTER`/`_SIZE` mechanism this
   driver uses (not `hipLaunchKernelGGL`), to see if the corruption reproduces outside this
   codebase entirely — that would point conclusively at ROCm/HIP or the hardware rather than
   at anything in this repo.
2. If reproducible standalone, escalate to the ROCm/hardware team (same track as
   `docs/gfx1250_fp32_wmma_occupancy_race.md`).
3. If **not** reproducible standalone, the next suspect is `wmma_k_tail`/`row_repeat_a`-style
   codegen-time register-count interactions specific to `igemm_bwd_gtc_wmma_nhwc_t`'s
   `sseq()` allocation order — try moving `s_gemm_k_per_wg`'s declaration to be immediately
   adjacent to `s_group`'s (so the two loads can merge into one `s_load_dwordx2`) as a
   structural workaround, and confirm whether the corruption follows the register number or
   disappears.
4. Do not re-enable bwd `gemm_k_global_split` (remove the `tunable_is_valid` rejection in
   `driver/igemm_bwd_gtc_driver.h`) without a hardware-validated fix — this is a memory-fault
   crash, not a wrong-answer, and must not regress silently.

## Reproduction

```
$ python3 igemm_codegen.py config/igemm_bwd_gtc_gfx1250_nhwc_fp16_all.config -d /tmp/x
# (with the tunable_is_valid rejection above reverted, to rebuild the affected kernel)
$ IGEMM_RUN_ONLY_KERNEL=igemm_bwd_gtcw_nhwc_fp16_bx0_ex0_bt128x128x32_wt16x16_wr4x4_ta1x32x1x1_1x1x1x128_tb1x32x1x1_1x1x1x128_dstrb_gkgs \
  /opt/rocm/bin/rocgdb -q -ex 'set amdgpu precise-memory on' -ex run --args \
  ./conv_driver.exe convfp16 -n 32 -c 128 -H 16 -W 16 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 \
  -l 1 -j 1 -g 1 -F 2 -V 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
```
