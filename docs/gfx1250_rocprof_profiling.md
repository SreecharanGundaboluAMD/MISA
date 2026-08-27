# rocprof-based profiling of MISA's gfx1250 WMMA kernels (2026-08-27)

Hardware-counter profiling of MISA's own kernels, aimed specifically at the biggest known
gap from `docs/gfx1250_vendor_benchmark_vs_miopen.md`: wrw is 1.1x-26x slower than MIOpen
(both on gfx950 and on MIOpen's own gfx1250 solvers), the worst of the three directions by
a wide margin. Goal: characterize WHERE wrw's cycles actually go (compute? memory? sync?
atomics?) using real hardware counters instead of guessing from architecture alone, and
cross-reference against the CDNA5 ISA doc for concrete, testable levers. This is a first
pass, not exhaustive -- see "Not yet done" at the end.

## Tooling and methodology

`/home/sgundabo/rocm-10.1/bin/rocprofv3` (rocprofv2 is not installed on this box; no
`rocprof` v1 binary either). Two invocation modes used:
- `--kernel-trace`: per-dispatch start/end timestamps, kernel name, grid/workgroup size,
  VGPR/SGPR/LDS usage -- written to a `rocpd` SQLite database (`rocpd_kernel_dispatch_*`,
  `rocpd_info_kernel_symbol_*` tables).
- `--pmc COUNTER1 COUNTER2 ...`: hardware performance counters, one value per counter per
  dispatch per HW instance (shader engine/SIMD/etc.), also into the same SQLite schema
  (`rocpd_pmc_event_*` joined to `rocpd_kernel_dispatch_*` via `event_id`). All requested
  counters fit in a single pass for this GPU (no multi-pass warning) — see the counter list
  below.

**Isolating a single kernel dispatch for clean measurement**: `conv_driver.exe`'s default
`driver_mode_normal` searches every tunable in the built module (multiple real kernel
launches per shape) and, for `gemm_k_global_split` configs, ALSO runs an internal ternary
search over split counts (`igemm_wrw_gtc_driver.h`, confirmed still present and correct as
of this session's earlier gfx1250-vs-gfx1250 benchmark investigation). Both of these are
real, separate dispatches that would otherwise mix together under one profiling run. Two
techniques used to get a single, deterministic kernel dispatch per profiling run:
1. A config file trimmed to ONE tile-shape section only (e.g. copy just the `64x64x32`
   block out of `igemm_wrw_gtc_gfx1250_nhwc_bf16_gsplit.config` into a scratch file) —
   avoids the tile-shape search.
2. `IGEMM_GSPLIT_SWEEP=<N>` (an existing, already-documented research-only env var in
   `igemm_wrw_gtc_driver.h`) — forces one specific split count, avoiding the ternary
   search's multiple internal launches.

**Counters collected** (all read successfully in one `--pmc` pass):
`SQ_BUSY_CYCLES`, `SQ_CYCLES`, `SQ_WAVES`, `SQ_INST_CYCLES_VALU_WMMA`, `SQ_INSTS_LDS`,
`GRBM_GUI_ACTIVE`, `GRBM_TA_BUSY`, `TX_VMW_ATOMIC_SETCONFLICT_STALL`, `GL2C_EA_ATOMIC`.
Values are summed across all reported HW instances per counter (consistent across
counters collected in the same run, so RATIOS between two counters from the same run are
meaningful even though the raw instance-count denominator isn't independently verified).

**Caveat**: `--pmc` collection forces additional serialization overhead vs. an
un-instrumented run — the driver's own reported `cost:` under `--pmc` was ~2-15x higher
than the same shape's un-instrumented time. This does not invalidate ratios computed
*within* one profiled run (both counters in a ratio see the same overhead), but absolute
timings collected under `--pmc` should not be compared against un-instrumented benchmark
numbers.

## Finding 1: wrw's worst-case shape spends only ~2.3% of cycles on WMMA compute

Profiled `c=192,H=60,W=80,k=64,1x1` (the shape measured at 26x slower than MIOpen/gfx1250
in the benchmark doc — the single worst outlier), using the 64x64 tile,
`gemm_k_global_split=1`, pinned to split=252 (the split this shape's ternary search
actually picks). 7 dispatches profiled (2 warmup + 5 repeat):

| Counter | Sum | Per-dispatch avg |
|---|---|---|
| `SQ_CYCLES` | 748,539,648 | 106,934,235 |
| `SQ_BUSY_CYCLES` | 617,612,307 | 88,230,330 |
| `SQ_INST_CYCLES_VALU_WMMA` | 16,934,400 | 2,419,200 |
| `SQ_INSTS_LDS` | 27,518,400 | 3,931,200 |
| `TX_VMW_ATOMIC_SETCONFLICT_STALL` | 0 | 0 |
| `GL2C_EA_ATOMIC` | 2,709,518 | 387,074 |

- **SQ busy fraction** (`SQ_BUSY_CYCLES / SQ_CYCLES`): **82.5%** — the SQ (instruction
  issue unit) is genuinely active most of the time, not stalled/idle waiting on something
  external. This kernel is not simply "waiting around."
- **WMMA busy fraction** (`SQ_INST_CYCLES_VALU_WMMA / SQ_CYCLES`): **2.26%** — of that
  active time, only about 1 in 44 cycles is actually spent executing WMMA matrix
  instructions. The other ~80% of active cycles are address computation, LDS traffic,
  atomics, barriers, and loop bookkeeping — instruction-issue overhead completely
  dwarfing the actual tensor-core work.
- **`TX_VMW_ATOMIC_SETCONFLICT_STALL` = 0**: this specifically rules out one plausible
  hypothesis — that same-address atomic-add contention between waves/workgroups
  targeting the same output element is the bottleneck. It measurably is not, at least not
  at this shape/split. (`GL2C_EA_ATOMIC`'s ~387k atomic ops/dispatch is simply the
  expected count — 252 splits × the output tensor's element count each contributing one
  atomic add — not evidence of a stall.)

## Finding 2: even a "healthy" fwd kernel is only ~5% WMMA-busy — but wrw is still ~2x worse

For contrast, profiled `c=64,H=60,W=80,k=128,1x1` fwd (measured at 1.13x-1.26x vs. MIOpen
in the benchmark doc — one of the *better*-performing shapes), both tile choices
(driver_mode_normal's search):

| Kernel | SQ busy % | WMMA busy % | SQ_WAVES (sum) |
|---|---|---|---|
| fwd 64x64 | 93.0% | 5.15% | 88,200 |
| fwd 128x128 | 94.1% | 5.16% | 44,100 |
| wrw 64x64 gsplit (from Finding 1) | 82.5% | **2.26%** | — |

Two things stand out:
1. **Neither direction is "WMMA-bound" in an absolute sense** — even MISA's best-behaving
   shapes spend the large majority of active cycles on non-WMMA instruction issue (VALU
   address math, LDS traffic, global loads/stores). This particular fwd shape is a 1x1
   conv with modest K (64), so it's plausibly genuinely memory/address-bound rather than
   under-optimized — a low WMMA fraction here isn't necessarily a red flag by itself.
2. **wrw's WMMA fraction is roughly HALF of fwd's on the same rough infrastructure**
   (WMMA main loop, same ISA, same GPU), while wrw's measured gap vs. MIOpen is 5-20x
   larger than fwd's. This is consistent with (and now quantifies) the already-documented
   architectural diagnosis: `gemm_k_global_split` turns each workgroup's useful compute
   into a small fraction of a much-larger, redundantly-executed K-reduction loop (many
   splits × full main-loop overhead each, for a shrinking useful-K-per-split as the split
   count grows) — the WMMA-busy-fraction gap is the first *direct hardware-counter*
   confirmation of this, not just an inference from grid-size arithmetic.

**External cross-check**: `docs/gfx1250_rocke_deep_dive.md`'s investigation of rocKE's own
GEMM-optimization playbook independently documents a bottleneck-diagnosis rule of thumb —
"low MFMA/WMMA-instruction fraction (<40% of total instructions) relative to
address/pack/mask instructions → compute-bound assumption is wrong, fix coordinate
arithmetic first." MISA's measured WMMA-busy-cycle-fraction (2.3% for wrw, 5.2% for fwd)
is far below even that already-conservative 40% threshold — both directions, not just
wrw, are unambiguously in "fix the surrounding arithmetic/overhead, not the matrix
instructions" territory by rocKE's own stated heuristic, though wrw is roughly 2x further
into that territory than fwd.

## ISA-doc cross-reference: a concrete, testable lever this profiling data motivates

CDNA5 ISA doc §5.7.2.1 ("Disable Multicycle XDL Stall"): normally, after a wave issues a
multi-cycle VALU op (e.g. a 16-cycle WMMA), the instruction arbiter stalls that wave from
issuing anything else until it completes — deliberately, so *other* waves on the same SIMD
get the opportunity to co-execute during that time. `SCHED_MODE` bit[2]
(`DISABLE_XDL_ARB_STALL`/`DISABLE_VALU_ARB_STALL`, set via `S_SETREG_B32`) lets a wave
issue multiple WMMAs back-to-back instead, at the cost of blocking that co-execution
opportunity for other waves. The doc states explicitly: **"this can block co-execution
opportunities so it is likely beneficial primarily when a single wave is running on a
SIMD."**

This is directly, precisely relevant to Finding 1/2's data: wrw's `gemm_k_global_split`
grid has many small workgroups (grid_z = split count, up to 252+ in the profiled case)
each doing comparatively little WMMA work — plausibly low occupancy-per-SIMD in practice
for some of these tiny-tile shapes, exactly the regime the ISA doc says this bit helps.
fwd/bwd's larger, well-occupied kernels are exactly the regime the same doc warns it could
hurt (blocking beneficial co-execution among many concurrent waves). **Recommendation:
an isolated A/B test of `disable_xdl_arb_stall` on a wrw gsplit shape specifically,
separate from any fwd/bwd test** — this was previously noted (via the FlyDSL research
pass) as an untested idea, but this profiling data is the first evidence pointing at
*which* MISA kernels it's actually likely to help vs. hurt. (Corrected an inaccuracy while
cross-referencing this: `docs/gfx1250_external_research_findings.md` previously recorded
this as `SCHED_MODE` "bit 4" — the ISA doc specifies bit[2]; fixed in that file.)

## Recommendations, prioritized

1. **Reduce per-split main-loop overhead for wrw's `gemm_k_global_split`, not just the
   split *count*.** The existing ternary search (Phase 20) already optimizes split count
   for wall-clock time, but this profiling shows the fundamental problem is that even the
   *best* split count still spends ~98% of active cycles on non-WMMA work. A search over
   split count alone cannot fix a fixed per-split overhead-to-compute ratio — the next
   lever has to reduce that ratio directly (fewer, cheaper per-iteration operations in the
   main loop for the small-tile case, or a fundamentally different occupancy strategy —
   see the parallel research into CK/hipconv/FlyDSL/rocKE's stream-K and persistent-kernel
   designs, which this doc's findings should be read alongside).
2. **A/B test `disable_xdl_arb_stall` on wrw specifically** (see above) — cheap
   (single `S_SETREG_B32` at kernel entry), testable in isolation, ISA-doc-motivated
   reasoning for why it should help wrw without needing to guess.
3. **Extend this profiling exercise to bwd and to fwd's tail-handling paths** (mtail/
   ntail/tdm), which the benchmark doc also flagged as measurably worse than the
   exact-fit base path — this pass only profiled fwd-base and wrw-gsplit; the
   tail-handling paths' WMMA-busy-fraction and SQ-busy-fraction are unmeasured and would
   likely show a similar (or different!) story worth knowing before prioritizing work.
4. **Profile a worse split count and a spatially-larger wrw shape for comparison** — this
   pass profiled one shape at its best split; confirming that a *deliberately worse* split
   choice shows an even lower WMMA fraction (not just a slower wall-clock time) would
   strengthen the "overhead scales with split count" diagnosis with direct counter
   evidence rather than inference from timing alone.

## Not yet done

- No LDS-bank-conflict-specific counters collected (there are dedicated ones on this GPU,
  e.g. under the `TX_VMW_*`/`SQC_*` blocks per `rocprofv3 --list-avail`, not yet explored).
- No `rocprof-compute` (the roofline/comprehensive successor to Omniperf, confirmed
  present at `/home/sgundabo/rocm-10.1/bin/rocprof-compute`) run yet — would give a more
  complete automated roofline/occupancy report per kernel instead of hand-picked counters.
- No occupancy-vs-theoretical-max computation (`SQ_WAVES` vs. `max_waves_per_simd`,
  both queryable) — would directly confirm or refute the "low occupancy per SIMD" premise
  behind the `disable_xdl_arb_stall` recommendation above, rather than leaving it as a
  plausible inference.
- GPU was under contention throughout this session (see
  `docs/gfx1250_vendor_benchmark_vs_miopen.md`'s repeated notes on this) — the *ratios*
  reported here (WMMA-busy-fraction etc.) should be far less contention-sensitive than
  absolute wall-clock numbers, but this hasn't been independently confirmed by re-running
  on an idle GPU.
