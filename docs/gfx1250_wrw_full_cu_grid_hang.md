# wrw "full-CU-grid hang" — investigation notes (R6), DISPROVEN

**Status: resolved. Not a real hardware bug.** A prior session's perf report claimed a
non-split-K wrw WMMA tunable (`k2x`, `dbuf`, `bf16_k2x_bf16acc` all implicated) hangs the
device indefinitely whenever the shape's GEMM grid exactly saturates the CU count (256).
This session initially added a defensive rejection for the condition (commit `980dad1`)
without re-testing the raw path, out of caution given the reported reboot cost. On later
investigation (with the user's explicit direction to look into the suspiciously long
"hang" timings before assuming a real hang, and to test the raw path once that was
addressed) the claim did not hold up:

- **The real bug was a test-driver performance bug**, not a device hang: `gen_rand_vector`
  (`driver/conv_driver.cpp`) and `block_wise_tensor_copy` (`driver/tensor_copy_cpu.h`) filled
  host tensors with a **strided round-robin** access pattern (`for i = tid; i < n; i +=
  num_threads`). On this workstation, `std::thread::hardware_concurrency()` is **255** —
  with a stride of 255 elements, every 64-byte cache line (16 fp32 elements) gets written by
  up to 16 different threads in an interleaved order, causing severe cache-line-bouncing
  false sharing. For the report's regression shape (`n256 c2048 14x14 k2048`, ~103M elements
  per tensor), this made one-time host-side setup take **60+ seconds** — long enough, under
  a `timeout 30-45` guard (as the original report explicitly used), to look exactly like an
  unrecoverable device hang, especially since the first process output (`[wrw: N]
  kernel_name`) doesn't print until *after* this setup completes, so a killed process
  produces zero output, indistinguishable from a real hang without further investigation.
- **Fixed** (`driver/conv_driver.cpp`'s `block_wise_rand_generator`, `driver/
  tensor_copy_cpu.h`'s `block_wise_tensor_copy` and its `int4x2_t` specialization): each
  thread now gets a contiguous chunk (`[tid*chunk, (tid+1)*chunk)`) instead of a strided
  round-robin. Measured on the exact regression shape: **~63s → ~0.4s** host-side setup
  (isolated single-kernel run, no GPU work) — roughly 150x.
- **With the fix, the raw (previously "hanging") kernels were re-tested directly** — no
  guard, no rejection, the literal reported-as-dangerous dispatch — at the exact hang shape
  (`n256 c2048 14x14 k2048`, grid=16x16=256=CU count):
  - `dbuf` (`bt128x128x32`, no split-K): completed in ~2.6ms kernel time, `valid:y`,
    3x repeated, zero anomalies.
  - `k2x` (`bt128x128x64`, `gemm_k_per_block=64`, no split-K): completed in ~3.8ms,
    `valid:y`.
  - `bf16_k2x_bf16acc`: completed in ~3.8ms, but `valid:n` — this is a **separate,
    already-known, unrelated correctness issue** (see `docs/gfx1250_wmma_perf_report_v2.md`'s
    own housekeeping item "exclude known-bad bf16acc from master config search"), not a hang.
  - `sudo dmesg -T` was checked after every test: **zero `amdgpu` messages of any kind**
    (no page fault, no ring timeout, no GPU reset) across the entire investigation, except
    for the unrelated, already-resolved R5 memory-fault entries from earlier in the session.
    `rocm-smi` stayed fully responsive throughout.

**Conclusion: the reported hang never happened on the device.** It was a client-side
(test-driver) performance bug in this repo's own RNG/copy helpers, now fixed. The R6
rejection added in commit `980dad1` has been **reverted** (`driver/igemm_wrw_gtc_driver.h`) —
it was solving a problem that didn't exist and would have incorrectly blocked legitimate
future non-split-K wrw configs at exactly this grid size.

## Lesson for future investigations

- **A `timeout`-killed process with zero output is not proof of a device hang** —
  distinguish "no output because setup never finished" from "no output because the device
  is wedged" by (a) checking `dmesg` for actual `amdgpu` fault/reset/timeout messages, and
  (b) checking `rocm-smi`/`rocminfo` responsiveness, *before* concluding a hang and before
  taking any destructive recovery action. In this repo's own history, a `rocm-smi
  --gpureset` issued on an unconfirmed hang made a previous, unrelated situation worse
  (undetected by `rocminfo`/`hipGetDeviceCount` for 90+ seconds, requiring a full host
  reboot regardless) — the cost of guessing wrong is high, so confirm via `dmesg`/`rocm-smi`
  first.
- `hardware_concurrency()` on this specific workstation (255) is high enough that any
  strided-by-thread-count host-side loop over a large array is a likely false-sharing
  hotspot. Grep `driver/*.cpp`/`driver/*.h` for the `for (... i = tid; i < ...; i +=
  block_size)` pattern before adding new host-side parallel fill/copy/reduce loops; prefer
  contiguous per-thread chunks.

## Where the fix lives

- `driver/conv_driver.cpp`: `block_wise_rand_generator` (contiguous chunking).
- `driver/tensor_copy_cpu.h`: `block_wise_tensor_copy` (both the generic template and the
  `int4x2_t` specialization), contiguous chunking.
- `driver/igemm_wrw_gtc_driver.h`: R6 rejection reverted; only the original grid-starvation
  *warning* (unrelated to this investigation) remains.
