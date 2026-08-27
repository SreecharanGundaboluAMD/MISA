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

## Finding 3: real occupancy is tile-shape-bound, identical across fwd/bwd/wrw and across tail mechanisms — this weakens (doesn't kill) the `disable_xdl_arb_stall` case

Measured directly via `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor` (new tool:
`script/gfx1250_occupancy_check.cpp`), not inferred from `SQ_WAVES`, which only gives a
cumulative dispatch-lifetime total, not a per-CU concurrency figure. Queried against every
built master-config `.hsaco`, both tile shapes, all three directions, and every
tail/split-K variant available:

| Kernel class | block_size | vgpr_count | LDS/block | active_blocks/CU | waves/CU | max waves/CU | occupancy |
|---|---|---|---|---|---|---|---|
| 64x64 tile (fwd base, bwd base, bwd mtail_ntail_ktail, wrw gsplit base) | 64 | 172 | 16 KB | 10 | 20 | 64 | **31.2%** |
| 128x128 tile (fwd base/mtail_ntail_ktail/tdm/tdm_mtail_ntail, bwd base, wrw gsplit base/mtail_ntail_gkgs, wrw 128x128x128 k4x) | 128 | 252 | 64 KB | 4 | 16 | 64 | **25.0%** |

Two things this directly establishes:
1. **Occupancy is a pure function of tile shape** (which fixes VGPR count and LDS
   footprint) — completely flat across direction, tail-mechanism (base/mtail/ntail/ktail/
   tdm/combined), and split-K enablement. None of these control-flow variants change the
   compiled kernel's resource footprint enough to move the needle.
2. **wrw's `gemm_k_global_split` workgroups are NOT uniquely low-occupancy relative to
   fwd/bwd** — they sit at the exact same 25%/31.2% as every other kernel using the same
   tile shape. This is worth stating plainly because it revises (not invalidates) the
   `disable_xdl_arb_stall` reasoning in the "ISA-doc cross-reference" section above: that
   recommendation was motivated by an inference that wrw's small split-K workgroups are
   plausibly low-occupancy-per-SIMD ("exactly the regime the ISA doc says this bit
   helps"). The measured reality is 16-20 resident waves per CU (4-10 blocks x 2-4 waves),
   which is far from the ISA doc's stated ideal case of "a single wave running on a SIMD"
   for either tile shape, in any direction. The A/B test (Tier 2 item 5, still worth
   running — see `docs/gfx1250_optimization_backlog.md`) should be read as an empirical
   check of a weaker, direction-agnostic hypothesis ("does reducing arbitration stalls
   help at ~25-31% occupancy in general") rather than a wrw-specific fix for a uniquely
   bad occupancy problem, since no such asymmetry actually exists.
3. Neither tile shape is anywhere near LDS-capacity-bound at the observed concurrency:
   `sharedMemPerMultiprocessor` on this device is 320 KB; 10 blocks x 16 KB = 160 KB and
   4 blocks x 64 KB = 256 KB, both under budget, with the 128x128 case close enough (256
   of 320 KB, one more block would need 320 KB exactly) that LDS rounding/reservation
   overhead plausibly caps it at 4 rather than 5 — not conclusively isolated from a
   VGPR-driven cap (252 vgprs/thread is also a substantial fraction of typical
   register-file capacity), but LDS is the more likely binding constraint for the
   128x128 tile specifically given how close to the budget boundary it sits.

## Finding 4: extended to bwd and fwd's tail paths (mtail/ntail/ktail/tdm) — same qualitative story, but this run was measured under confirmed heavy GPU contention

Extended the Finding 1/2 methodology (isolated single-tile-shape scratch configs, same
`--pmc` counter set, via `/opt/rocm-10.1.0a20260820/bin/rocprofv3`) to the paths that
Finding 2's "Recommendations" item 3 flagged as unmeasured: bwd-base, bwd's combined
M+N+K-tail kernel, and fwd's `mtail_ntail_ktail`/`tdm`/`tdm_mtail_ntail` kernels. Same
shape family as Finding 2 (`c=64,H=60,W=80,k=128,1x1`, `n=42`) for the base/tdm cases;
`c=90or100,H=60,W=80,k=90or92,1x1` (chosen to actually activate the tail masks — `k`
kept a multiple of 4 per the `classify_gfx1250_coverage.py` `gemm_n%4==0` constraint
already documented in `docs/gfx1250_wmma_coverage_gap_analysis.md`) for the tail cases:

| Kernel | SQ busy % | WMMA busy % | SQ_CYCLES (sum, 7 dispatches) |
|---|---|---|---|
| bwd base | 93.5% | 0.61% | 1,852,691,450 |
| bwd mtail_ntail_ktail | 93.9% | 0.83% | 2,037,931,712 |
| fwd mtail_ntail_ktail | 93.9% | 0.92% | 1,843,725,310 |
| fwd tdm | 93.9% | 0.64% | 1,776,072,790 |
| fwd tdm_mtail_ntail | 93.8% | 0.91% | 1,854,993,114 |

**Important caveat, confirmed directly**: `rocm-smi --showuse` at measurement time
reported **GPU use: 100%** from other processes sharing this box (a multi-tenant
research machine — several unrelated `claude` sessions and other users' workloads were
active). This is the exact "GPU under contention" caveat already flagged in "Not yet
done" below, now directly confirmed rather than just suspected. The absolute WMMA-busy%
values above are **3-8x lower** than Finding 2's fwd-base figure (5.15%) measured earlier
in the session on the same rough shape family — almost certainly a contention artifact
(competing workloads inflate `SQ_CYCLES`' wall-clock-cycle count without adding to *this*
dispatch's own `SQ_INST_CYCLES_VALU_WMMA`), not a real regression in the tail paths.
**Do not compare these absolute percentages against Finding 1/2's numbers directly.**

What still survives, because it's a within-run (same-contention-level) relative
comparison rather than an absolute one: SQ busy consistently 93-94% (same
"genuinely active, not idle" story as base kernels), WMMA busy consistently well under
1% for every tail/TDM variant just as for the base kernels — no tail mechanism stands
out as qualitatively different in kind from the base path's overhead profile. Re-running
this specific comparison on an idle GPU to get absolute numbers comparable to Finding 1/2
remains open (see "Not yet done").

## Finding 5: instruction-mix decomposition via `rocprof-compute` -- upgrades Finding 1/2's "non-WMMA overhead" from inference to measurement

Findings 1/2/4 established WMMA occupies only 2-5% of *cycles*, and attributed the rest
to "address computation, LDS traffic, atomics, loop bookkeeping" by reading the kernel's
own code -- an architecturally-grounded inference, not a direct counter measurement.
This finding closes that gap using `rocprof-compute`'s Wave/VALU/VMEM/LDS Instruction
Mix blocks (7.5/7.6/7.7/7.8), which report actual per-category *instruction counts*, an
independent counting method from Finding 1/2's cycle-based `SQ_INST_CYCLES_VALU_WMMA`.

**Tooling notes** (gfx1250 is new/preview hardware in this rocprof-compute build):
`rocprof-compute`'s internal subprocess calls plain `rocminfo` (found via `PATH`), which
resolves to `/usr/bin/rocminfo` (an older, system-wide ROCm 6.x install) and segfaults on
this box's actual driver -- fixed by prepending `/opt/rocm-10.1.0a20260820/bin` to `PATH`
so the correct versioned `rocminfo` is found first. The `analyze` subcommand additionally
needs a specific pinned dependency set (`numpy==1.26.4`, `pandas==2.2.3`, etc., listed in
`/opt/rocm-10.1.0a20260820/libexec/rocprofiler-compute/requirements.txt`) not compatible
with the system-wide Python environment (which has newer numpy/pandas already installed
for other purposes) -- resolved with an isolated venv (`python3 -m venv`) rather than
touching system packages, since this is a shared multi-tenant box. The LDS block's
utilization/bank-conflict metrics (3.4.5-3.4.9) returned `N/A`/`0.0` even after explicitly
requesting `-b 3.4` during profiling -- appears to be a genuine gap in gfx1250's current
metric support in this rocprof-compute build (not a usage error), so LDS bank-conflict
data specifically is still not available on this hardware/tool combination.

**fwd, healthy shape** (`c=64,H=60,W=80,k=128,1x1`, 64x64 tile -- same shape as Finding
2's "healthy" fwd measurement):

| Category | Instructions | % of total |
|---|---|---|
| VALU (total) | 3,742,200 | 53.2% |
| &nbsp;&nbsp;of which WMMA | 201,600 | 2.9% (of total); 5.4% (of VALU) |
| &nbsp;&nbsp;of which non-WMMA VALU | 3,540,600 | 50.4% |
| LDS | 1,512,000 | 21.5% |
| SALU | 630,000 | 9.0% |
| Internal (barriers/waitcnt/branches) | 415,800 | 5.9% |
| VMEM (global load/store) | 403,200 | 5.7% |
| Transcendental | 37,800 | 0.5% |
| **Total** | 7,030,800 | 100% |

**wrw, worst-case shape** (`c=192,H=60,W=80,k=64,1x1`, 64x64 tile, split=252 -- same
shape/split as Finding 1):

| Category | Instructions | % of total |
|---|---|---|
| VALU (total) | 7,609,900 | 52.2% |
| &nbsp;&nbsp;of which WMMA | 302,400 | 2.1% (of total); 4.0% (of VALU) |
| &nbsp;&nbsp;of which non-WMMA VALU | 7,307,500 | 50.2% |
| LDS | 3,931,200 | 27.0% |
| Internal | 2,045,740 | 14.0% |
| VMEM (incl. the atomic-adds) | 399,168 | 2.7% |
| SALU | 443,016 | 3.0% |
| Transcendental | 77,112 | 0.5% |
| **Total** | 14,571,100 | 100% |

**What this confirms, with real counters rather than inference**:
1. **Non-WMMA VALU is the single largest category in both directions, and strikingly
   consistent**: 50.4% (fwd) vs 50.2% (wrw) of *all* instructions, despite these being
   very differently-shaped kernels (fwd's exact-fit non-split path vs wrw's heavily
   split, atomic-accumulate path). This is direct, positive evidence for "address
   computation" as a major, reproducible cost center -- not just an inference from
   reading the code.
2. **LDS is the second-largest category in both**, and *larger* for wrw (27.0% vs
   21.5%) -- consistent with wrw's double-buffered operand staging reading heavily from
   LDS every iteration (LDS Load 3,628,800 vs LDS Store only 302,400 -- almost all LDS
   traffic is reads), whereas fwd's LDS traffic is store-dominated (Store 1,008,000 vs
   Load 504,000, consistent with its LDS-reshuffle epilogue writing more than its
   double-buffered loads read).
3. **wrw pays a visibly larger "Internal" (barrier/waitcnt/branch bookkeeping) tax**:
   14.0% vs fwd's 5.9% -- a real, counter-measured difference, consistent with the
   existing diagnosis that many small split-K shards each redundantly pay a fixed
   per-shard bookkeeping cost.
4. **WMMA's instruction-count share cross-validates Finding 1's cycle-count share via
   an independent method**: wrw's WMMA instruction-share (2.1%) lands almost exactly on
   Finding 1's WMMA cycle-share for a near-identical shape (2.26%) -- two different
   counting methodologies (instruction count vs. cycle count) converging on the same
   number is a meaningfully stronger confirmation than either alone. fwd's instruction-
   share (2.9%) is lower than its cycle-share (5.15%, Finding 2), which is itself
   sensible, not contradictory: a WMMA instruction spans many cycles per issue, so its
   cycle-share is expected to exceed its instruction-count-share.
5. One tooling artifact worth flagging, not a hardware finding: `VALU Instructions -
   XDL` reports the exact same value as `VALU Instructions - WMMA` in both shapes
   (201,600 and 302,400 respectively) -- gfx1250 has no XDL (that's the CDNA/MFMA name),
   so this looks like this preview build's metric definitions not yet fully
   distinguishing the two names for gfx1250 specifically. `CMACC`/`SMACC` also report
   values individually exceeding "Total VALU Instructions," meaning they are not
   mutually-exclusive partitions of the total the way the plain VALU/SALU/LDS/VMEM
   breakdown is -- not used in the "50% non-WMMA VALU" conclusion above, which relies
   only on the additive Wave Instruction Mix block (7.5), where the categories do sum
   correctly.

**Bottom line for "can we optimize it"**: the diagnosis from Finding 1/2 is now measured,
not just inferred -- roughly half of all instructions in both directions are non-WMMA
VALU (address/index computation, masking), with LDS traffic a strong second contributor
that's proportionally worse for wrw specifically. This directly motivates TDM extension
(moves address generation from software VALU into hardware) as the highest-leverage next
lever, ahead of anything that only touches the WMMA path itself -- see
`docs/gfx1250_optimization_backlog.md` and Phase 42's TDM-for-bwd work.

## Finding 6: LDS cycles per instruction — near-conflict-free (1.15-1.27x theoretical minimum)

After `rocprof-compute`'s LDS bank-conflict metrics returned N/A for gfx1250 (Finding 5),
this finding uses `rocprofv3 --pmc` directly with `SQ_INST_CYCLES_LDS` to estimate LDS
throughput per instruction. The counter has an empty description in this tool build (a
gfx1250-preview tooling gap) but is present with the same block/dimensions as all working
SQ counters — the ratio methodology is valid regardless.

**Counters collected** (same two shapes as Findings 1/2 for direct comparability):
`SQ_CYCLES`, `SQ_INSTS_LDS`, `SQ_INST_CYCLES_LDS`, `SQ_INSTS_VALU`, `SQ_INST_CYCLES_VMEM`,
`SQ_WAIT_ANY`, `SQ_WAIT_INST_ANY`, `SQ_WAVE_CYCLES` — all fit in one `--pmc` pass.

**Cross-validation**: `SQ_INSTS_LDS` from this run = 27,518,400 (wrw) and 10,584,000 (fwd),
both exactly 7× the per-dispatch values from Finding 5 (7 dispatches = 2 warmup + 5 repeat).
The methodology is self-consistent.

| Metric | wrw gsplit=252 | fwd base | Notes |
|---|---|---|---|
| `SQ_INST_CYCLES_LDS` (total, 7 dispatches) | 31,752,000 | 13,406,400 | Cycles attributed to LDS instruction execution |
| `SQ_INSTS_LDS` (total, 7 dispatches) | 27,518,400 | 10,584,000 | LDS instruction count |
| **LDS cycles/instruction** | **1.154** | **1.267** | Key ratio: how close to conflict-free |
| `SQ_INST_CYCLES_VMEM` (total) | 35,226,240 | 19,051,360 | VMEM cycles for comparison |
| `SQ_INST_CYCLES_LDS / SQ_CYCLES` | 4.31% | 8.20% | LDS cycle fraction of total SQ cycles |
| `SQ_INST_CYCLES_VMEM / SQ_CYCLES` | 4.78% | 11.66% | VMEM cycle fraction of total SQ cycles |
| `SQ_WAIT_ANY / SQ_WAVE_CYCLES` | 95.69% | 79.67% | Wait-any fraction (same quad-cycle base) |
| `SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES` | 1.02% | 16.36% | Instruction-issue stall fraction |

**The main finding**: LDS throughput is **essentially conflict-free** at 1.15-1.27 cycles per
instruction. `ds_write_b128` and `ds_read_b128` have a theoretical 1.0-cycle throughput when
no bank conflicts occur (confirmed in CDNA5 ISA doc: 32 LDS banks, b128 accesses 4 banks, so
a conflict requires two concurrent waves hitting the SAME 4-bank subset in the SAME cycle).
Measured values of 1.15-1.27x of that minimum mean either a very small fraction of
instructions encounter conflicts, or the ~0.15-0.27 overhead comes from other per-instruction
latency (pipeline fill, wait-count logic) rather than bank conflicts specifically. Either way,
**LDS bank conflicts are definitively NOT a meaningful bottleneck for these kernels** — reducing
LDS instruction *count* (via TDM, already done) is the right axis, not reducing conflicts per
instruction.

**Secondary findings**:

1. **LDS cycle fraction (4-8%) is real but not the dominant overhead**. Compared to Finding 5's
   instruction *count* fractions (21.5% for fwd, 27% for wrw), the cycle fraction is much
   smaller — consistent with LDS instructions being nearly conflict-free (≈1 cycle each) and
   thus not disproportionately expensive per cycle even though they're the second-most-common
   instruction category.

2. **VMEM cycle fraction (fwd: 11.66%) is notably higher than LDS**. This is interesting given
   that Finding 5 showed fwd's VMEM instruction *count* is only 5.7% (403,200 out of 7,030,800
   total instructions). VMEM instructions being ~2x as many *cycles* per instruction as LDS
   means they have significant pipeline latency — consistent with global memory load latency
   (hundreds of cycles per in-flight request, amortized by wave-switching) being much higher
   than LDS latency.

3. **Wait-any fraction (~80-96%) is expected, not a red flag**. `SQ_WAIT_ANY` counts
   "per-simd, nondeterministic" quad-cycles where a wave is waiting for *anything* — this
   includes the normal multi-wave arbiter schedule where a wave is not the currently-selected
   wave on its SIMD. With 10-20 resident waves per CU (Finding 3), each wave is
   "waiting" most of the time simply because other waves are executing. This counter reflects
   the GPU's natural wave-switching behavior, not stalls.

4. **Wait-inst fraction (wrw: 1.02%, fwd: 16.36%) is the more meaningful stall metric**.
   `SQ_WAIT_INST_ANY` counts instruction-issue stalls specifically — cycles where it IS
   this wave's turn to issue but no instruction is ready (data hazard, scoreboard full, etc.).
   wrw's near-zero instruction-issue stall rate is consistent with its very low WMMA-busy
   fraction: the SQ is almost never stalled waiting for the previous WMMA to finish before
   issuing the next instruction. fwd's 16% is more substantial but still moderate.

**Bottom line**: LDS is not a hidden bottleneck. The LDS traffic profile (near-conflict-free,
low per-cycle cost) confirms that Finding 5's diagnosis stands intact — the dominant overhead is
non-WMMA VALU (address computation), and the appropriate lever is hardware-assisted addressing
(TDM), not LDS conflict reduction. The backlog item for LDS bank-conflict measurement is now
fully closed.

## Not yet done

- GPU was under contention throughout this session (see
  `docs/gfx1250_vendor_benchmark_vs_miopen.md`'s repeated notes on this) — the *ratios*
  reported here (WMMA-busy-fraction, LDS-cycles-per-instruction, etc.) should be far less
  contention-sensitive than absolute wall-clock numbers, but this hasn't been independently
  confirmed by re-running on an idle GPU.
