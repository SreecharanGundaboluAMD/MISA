# gfx1250 split-barrier LDS visibility bug — standalone reproducer

Self-contained, MISA-independent repro for the root cause behind
`../gfx1250_fp32_wmma_occupancy_race.md`. No Python/codegen dependency — just
hand-written GCN assembly (`kernel.s`) plus a ~100-line HIP host driver
(`host.cpp`). Build with `./build.sh` (needs `clang++`/`hipcc` from a ROCm
install targeting gfx1250; set `ROCM_PATH` if it's not at `/opt/rocm`).

## What it tests

Whether `s_barrier_signal`/`s_barrier_wait` (CDNA5's split barrier), combined
with `s_wait_dscnt 0x0` on the writing wave, actually makes that wave's LDS
writes visible to *other* waves in the same workgroup by the time they resume
past the barrier — independent of WMMA, independent of any GEMM/conv
addressing. Per workgroup (64 threads = 2 wavefronts of 32):

1. Every thread writes a per-iteration-unique "magic" value
   (`iter<<16 | wave_id<<8 | tid`) to its own LDS slot (`LDS[tid*4]`).
2. `s_wait_dscnt 0x0` (wait for *this wave's own* writes to land).
3. `s_barrier_signal -1` / `s_barrier_wait -1`.
4. Every thread reads **all 64 slots** (both waves' data) and compares each
   against the value that iteration should have produced. A stale read (a
   value from a previous iteration) is only possible if the barrier let this
   wave proceed before some other wave's write was actually visible.
5. Repeated for 100 iterations per workgroup launch, across a large,
   configurable number of concurrent workgroups (`./repro [num_workgroups]`).

No floating point, no WMMA — this isolates the exact synchronization
primitive the real kernel relies on.

## Result

Below ~1000-3000 concurrently-resident workgroups: **zero mismatches**, every
run. Above that: mismatches appear and scale with workgroup count. Example
(`./repro 20000`, gfx1250):

```
total slot-read checks: 8192000000
total failures: 18924608

Aggregate over 618144 recorded (last-mismatch-per-thread) records:
  slot_i histogram (nonzero only):
    slot 31: 480
    slot 63: 617664
  staleness histogram (iters behind):
    1 iters behind: 618144
```

This is a **precisely characterized** failure, not vague flakiness:

- **100% of mismatches are exactly 1 loop iteration stale.** Never 2+, never
  garbage/uninitialized-looking data. A wave crosses the barrier and reads
  the *previous* iteration's value from some other wave's slot.
- **>99.9% of mismatches are on slot 63** (i.e. `tid=63`, lane 31 — the
  *last* lane of the *last* wave, tid range 32–63). The rare remainder
  (~0.08%) are on slot 31 (lane 31 of the *first* wave, tid range 0–31) — the
  same "last lane of a wave" pattern, just far less frequently the one that's
  actually stale when the check runs. Every other slot (0–30, 32–62): zero
  observed mismatches across every run.

This points at something specific to how the **last lane's** contribution to
a full-wavefront-width `ds_write_b32` is drained/acknowledged — as if
`s_wait_dscnt`/the barrier's completion signal for that wave can be satisfied
before the highest-lane-index write of a preceding LDS store instruction has
actually committed, while lanes 0–30 are reliably already visible.

## How the diagnostic capture works (and why not an interactive debugger)

Single-stepping this under `rocgdb` would almost certainly perturb the exact
timing the race depends on and could easily make it stop reproducing
(occupancy/scheduling-dependent races are classic heisenbugs under a
debugger). Instead, `kernel.s` captures its own evidence with no debugger
involved: every mismatching lane overwrites a **per-thread** diagnostic
record (`{slot_i, iter, actual, expected}`, 4 dwords) in global memory via
`v_cmpx_ne_u32` + `global_store_dword`, at full unmodified execution speed —
no atomics, no cross-thread contention on the diagnostic write itself, no
breakpoints. After the run, the host reads back every thread's *last*
mismatch (if any) and histograms it (`slot_i`, staleness). This gets the same
"actual vs. expected register content at the moment of failure" information
an interactive debugger session would, without the risk of masking the race.

## Files

- `kernel.s` — the kernel, hand-written GCN assembly (tracked in git despite
  the repo's blanket `*.s` ignore rule — see `.gitignore`).
- `host.cpp` — HIP host driver: loads `kernel.hsaco`, launches, verifies,
  reports the histograms above.
- `build.sh` — `clang++ -x assembler ... kernel.s -o kernel.hsaco` +
  `hipcc host.cpp -o repro`.
- `kernel.hsaco`, `repro` — prebuilt artifacts (also committed, so this can be
  handed off and run without rebuilding, though rebuilding is one command).

## For the hardware team

Run `./repro 20000` (or higher) a few times. Below the occupancy threshold
it's always clean; above it, expect the exact signature described above.
The fact that it is *always* exactly 1 iteration stale and *overwhelmingly*
localized to the last lane of a wave (not randomly distributed across
slots/staleness) should narrow this considerably compared to a generic
"barrier doesn't work" report.
