# wrw full-CU-grid hang without split-K — investigation notes (R6)

**Status: unresolved, mitigated by rejection.** `driver/igemm_wrw_gtc_driver.h`'s WMMA
`run()` now hard-rejects (`return_code = -1`, surfaces as "not applicable") any
`gemm_k_global_split=0` wrw tunable whose base grid (`grid_x*grid_y`, before any split-K
`grid.z` multiplier) is `>=` the device's CU count, **before** it ever calls
`hipModuleLaunchKernel`. This also currently catches `wrw_streamk` tunables (they don't set
`gemm_k_global_split` either, and streamk was not independently verified safe at this exact
grid size — see "Scope of the mitigation" below). No shipped wrw config in this repo
currently reaches this path with a shape that trips the guard (the shipped default sets
`gemm_k_global_split=1`), so this is a no-op today; it exists to stop a future config,
benchmarking script, or hand-edited debug tunable from silently reproducing the hang.

**Do not remove or narrow this guard without a hardware-validated fix, and do not attempt to
reproduce the raw hang without a strict process-level timeout AND acceptance that the GPU may
require a full host reboot to recover** (`rocm-smi --gpureset` was tried in a prior session
after a hang and made the device state worse — undetected by `rocminfo`/`hipGetDeviceCount`
for 90+ seconds — a reboot was required either way). This investigation deliberately did
**not** attempt to re-trigger the hang; see "Why this was not reproduced live" below.

## Symptom (as reported, not independently re-reproduced this session)

Any non-split-K wrw WMMA tunable (`k2x`, `dbuf`, `bf16_k2x_bf16acc` all independently
implicated) hangs indefinitely — not `valid:n`, an actual device-side hang requiring a process
`timeout` to even regain a shell prompt — whenever the shape's GEMM grid
(`ceil(k/group, gemm_m_per_block) * group * ceil(c/group, gemm_n_per_block)`) exactly equals
this GPU's CU count (256). The regression shape is `n256 c2048 14x14 k2048` (1x1): wrw's GEMM
is `M=k/group=2048, N=c/group=2048`, which tiles to exactly `16x16=256` 128x128 workgroups.
Smaller grids (64, 16 workgroups on the same shape family) complete normally (slow, as
expected — grid starvation — but not hung). The shipped default tunable (`gemm_k_global_split
=1`, i.e. `_gkgs`) completes normally on the identical shape, because splitting the reduction
axis across `grid.z` changes the total workgroup count away from the exact 256-workgroup
collision.

## Why this was not reproduced live this session

Reproducing this requires dispatching the exact hang-triggering kernel, which the report
says requires a full host reboot to recover from (not just `rocm-smi --gpureset`, which is
reported to have made a prior occurrence *worse* — an already-degraded GPU state that a
reboot was still required to clear). Given:

- The user's explicit prior experience needing a full machine reboot after hitting this,
- No available in-session mechanism to guarantee recovery short of a reboot the operator
  must perform manually,
- The fix (reject the dangerous dispatch before it ever launches) does not itself require
  triggering the hang to implement or to verify — `tunable_is_valid`/`run()`'s early-return
  path is ordinary host-side control flow, testable without ever reaching
  `hipModuleLaunchKernel`,

this investigation implemented the mitigation from static code analysis and repo precedent
(see below) and verified **only** that the guard fires (returns "not applicable") on the
documented hang shape, without ever letting a real dispatch reach the device. It did not
capture a live fault signature (no `rocgdb` session, no register dump) the way R5's crash
investigation did — there was no way to do that safely for a device-hanging condition using
the tools available in this environment (a `timeout`-killed *host process* does not
guarantee the *GPU queue* itself recovers; the R5 precedent used the same `rocgdb`/precise-
memory technique specifically because a memory *fault* — unlike a hang — reliably terminates
the offending wave and returns control to the host).

## Working hypothesis (not independently confirmed)

The perf report that found this hang frames it as "a plausible instance of the same class of
issue as `docs/gfx1250_fp32_wmma_occupancy_race.md`" — both triggered by "every CU
simultaneously saturated, no idle SIMD slack anywhere to break a dependency cycle." That
precedent's mechanism (a barrier/LDS-visibility staleness race, see that doc) is explicitly
about `v_wmma_f32_16x16x4_f32` (fp32, `gemm_k_per_block=4`) and was NOT reproduced for
fp16/bf16 (`v_wmma_f32_16x16x32_f16`/`bf16`, `gemm_k_per_block=32`) at up to ~50k concurrent
workgroups for fwd/bwd. wrw's hang shape here is only 256 workgroups — several orders of
magnitude below where that specific mechanism was ever observed to bite for fwd/bwd's fp16/
bf16 builds. That makes it unlikely (though not impossible — wrw's addressing and main-loop
functors are not byte-identical to fwd/bwd's) that this is literally the *same* race;
"exactly grid == CU count, breaks only there" is also a distinctly different signature (a
hang, not staleness) from that doc's reproducer (wrong data, never a hang). Code inspection
(this session) ruled out one specific alternative hypothesis: wrw's `wrw_streamk` persistent-
kernel/atomic-claim loop (which *would* plausibly deadlock under some interleavings) is
gated behind a completely separate emission path
(`igemm_wrw_gtc_wmma_nhwc_t.emit_kernel_streamk_loop`, only called when
`tunable.wrw_streamk` is set) from the plain tap-loop path
(`emit_kernel_tap_loop`) that `k2x`/`dbuf`/`bf16acc` actually use — there is no leftover
persistent-loop/spin-wait/atomic-claim code compiled into the non-streamk kernels that could
explain a deadlock by itself. The true mechanism remains uncharacterized.

## Scope of the mitigation

The rejection triggers on `base_grid = grid_x*grid_y >= CU count` for any
`gemm_k_global_split=0` WMMA wrw tunable — this also currently catches `wrw_streamk`
tunables (mutually exclusive with `gemm_k_global_split` at the tunable level, so they take
the same `!gemm_k_global_split` branch). `wrw_streamk` was **not** implicated by the original
report and was not independently tested at this exact grid size; it is included in the
rejection anyway because it was not tested and confirmed *safe* here either, and the
downside of a false-positive rejection (an experimental perf feature reports "not
applicable" at one specific grid size) is far cheaper than the downside of a false negative
(another hang). If `wrw_streamk` is later confirmed safe at exactly `grid == CU count`
(its own `grid.z` multiplier plausibly routes around the same collision the same way
`gemm_k_global_split` does), narrow the condition to exclude it explicitly.

## Recommendation for whoever picks this up next

1. Reproduce only on hardware where a hang's blast radius is acceptable (a dedicated,
   easily power-cycled test box, ideally with an out-of-band reset, not a shared/relied-upon
   machine) and with `rocgdb` attached *before* dispatch if at all possible, since a live
   hung-wave inspection (`info threads`, per-wave PC/registers, no `precise-memory` needed —
   a hang doesn't raise a fault signal, but the waves are still resident and inspectable
   while hung) is the most direct way to find which instruction every wave is actually
   stuck at.
2. Test whether the hang is specific to `grid == CU count` exactly, or persists for
   `grid > CU count` (oversubscribed, extra workgroups queued) too — the current guard
   conservatively rejects both (`>=`), but the report only directly evidences the exact-equal
   case.
3. Test with a synthetic fwd/bwd shape engineered to hit `grid == CU count` (per
   `docs/gfx1250_fp32_wmma_occupancy_race.md`'s wrw caveat, their GEMM mapping makes a
   naturally-occurring collision rare, but a shape can be constructed) to determine whether
   this is wrw-specific or a general full-occupancy issue affecting every direction.
4. Do not remove this guard without a hardware-validated fix, confirmed via the process in
   item 1, on a machine where a reboot is cheap.

## Where the mitigation lives

- `driver/igemm_wrw_gtc_driver.h`, WMMA branch of `run()`, immediately after `grid_x`/
  `grid_y` are computed and before any kernel is launched.
