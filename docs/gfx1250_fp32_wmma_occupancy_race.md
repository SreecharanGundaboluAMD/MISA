# gfx1250 fp32 WMMA occupancy-dependent correctness bug (KNOWN ISSUE, mechanism characterized)

Status: **mitigated (lds_double_buffer=1 mandatory); underlying hardware
mechanism precisely characterized via a standalone, WMMA-independent
reproducer, but not yet root-caused at the silicon level.**
Affects: every gfx1250 WMMA config using `v_wmma_f32_16x16x4_f32` (i.e. any
`precision=fp32` WMMA config with `gemm_k_per_block=4`) -- both `fwd` and `bwd`
confirmed by direct testing; `wrw` presumed affected (identical code path, not
independently stress-tested -- see "wrw caveat" below). Both 64x64 and 128x128
tiles confirmed affected.

**Update:** a standalone, MISA-independent reproducer
(`docs/gfx1250_fp32_wmma_race_repro/`) isolated the exact mechanism, with no
WMMA involved at all: **`s_barrier_signal`/`s_barrier_wait`, combined with
`s_wait_dscnt 0x0` on the writing wave, does not reliably make the *last
lane's* (lane 31, i.e. the highest thread ID in a wave) LDS write visible to
other waves by the time the barrier releases them, once enough workgroups are
concurrently resident.** Every observed failure is stale by *exactly one*
loop iteration (never more, never garbage), and >99.9% of failures are reads
of the slot written by the last lane of a wave. See "Root mechanism, now
characterized" below and `docs/gfx1250_fp32_wmma_race_repro/README.md` for
the full repro and data. This supersedes item 1 in "What this is NOT" below --
that section is kept as-is for the historical trail (the WMMA-hazard/bank-
conflict/setprio investigation is all still valid and worth keeping), but its
conclusion on the barrier-fence question specifically was wrong; see the
update section for why.

## Summary

Single-buffered LDS (`lds_double_buffer` unset/0) main-loop kernels built from
`igemm_{fwd,bwd,wrw}_gtc_gfx1250_nhwc_fp32.config` produce silent wrong-answer
results (`valid:n`, NRMS ~1e-3) once enough workgroups are launched concurrently.
Below a size-dependent threshold, the exact same kernel binary is 100%
correct on every run. `lds_double_buffer=1` reliably fixes it in every
configuration tested. **All fp32 WMMA configs in this repo now set
`lds_double_buffer=1` unconditionally as a result of this investigation --
do not remove it without re-validating at the occupancy scale described below.**

## Reproduction

```bash
# BWD, 64x64 tile -- fails around ~1500 total workgroups
python3 igemm_codegen.py -d /tmp/repro config/igemm_bwd_gtc_gfx1250_nhwc_fp32.config
# (temporarily comment out both `lds_double_buffer = 1` lines to see the failure)
cd /tmp/repro
IGEMM_WARMUP=1 IGEMM_REPEAT=1 ./conv_driver.exe conv \
  -n 24 -c 128 -H 60 -W 80 -k 8 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 \
  -F 2 -V 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
# -> igemm_bwd_gtcw_..._bt64x64x4_..., valid:n (flaky: sometimes y, sometimes n, at n=24)
#    n=4..20 (same c/k/H/W/K): reliably valid:y, every run
#    n>=36: reliably valid:n
```

| Config | Tile | Fails at (total workgroups, roughly) | Passes reliably below |
|---|---|---|---|
| bwd fp32 | 64x64 | ~24 * 60 * 80 / 64 * 2 ≈ 1500 | n<=20 (≈1250 wg) |
| bwd fp32 | 128x128 | n=128 (c=128,k=8) ≈ 12000 wg | n=42 (≈3150 wg) |
| fwd fp32 | 64x64 | n=128 (c=8,k=128) | n=84 |
| fwd fp32 | 128x128 | not independently isolated -- fixed proactively | -- |

`k>=8` (2+ K-loop iterations) and `K=4` (1 iteration, no LDS-buffer re-read)
never reproduces this at any grid size tested -- the single-buffered
read-after-write on the SAME LDS address across loop iterations is necessary,
not just contention.

Random input data is regenerated every run (`conv_driver.cpp`'s RNG is seeded
from the system clock, not a fixed seed) -- run-to-run NRMS variance is
therefore not by itself proof of a hardware race, but the **grid-size
threshold** (reliably correct below it, unreliable/wrong above it, same code,
same shapes otherwise) is: a purely static/deterministic addressing bug would
fail identically regardless of how many workgroups are launched.

## What this is NOT (ruled out, with evidence)

1. **"CDNA5's split barrier provides no LDS fence" was the original theory offered
   for this bug (citing ISA doc S5.6/S15.5).** This is not supported by the ISA
   doc: S5.7.1.4 states LDS-write DScnt completion means "the data is written
   into LDS memory" -- the documented model treats `s_wait_dscnt` + barrier as a
   complete, sufficient fence. It's also empirically false: `fwd` (which uses
   the identical barrier/`s_wait_dscnt` scheme, and was originally believed
   unaffected) fails too once pushed to high enough occupancy. If barriers
   provided no fence at all, `fp16`/`bf16` builds using the exact same loop
   structure would fail at the same rate; they didn't reproduce in any test run
   (up to n=256, ≈50k workgroups).

2. **WMMA operand same-wave WAR hazard.** `v_wmma_f32_16x16x4_f32` (opcode 93)
   is conspicuously absent from the ISA doc's WMMA data-hazard table (S7.12.1) --
   that table only covers the "XDL" dense WMMA forms (F16/BF16/FP8/IU8/F4),
   each requiring V_NOPs before a subsequent instruction overwrites that WMMA's
   Matrix A/B/Index source. BWD's transposed-B read overwrites `v_b` via
   `v_mov_b32` right after the barrier, with zero VALU instructions between it
   and the previous iteration's last WMMA. **Tested**: inserted 4, then 20,
   `v_nop` at that exact point (`python/operations/wmma_main_loop.py`, right
   after `s_barrier_wait -1`) and rebuilt -- zero effect on failure rate at any
   grid size. This rules out "insufficient same-wave instruction spacing".

3. **Cross-wave matrix-pipe issue-arbitration.** Tested the existing
   `wmma_setprio=1` tunable (brackets the WMMA burst with `s_setprio 1`/`0` to
   stop this wave yielding mid-burst to other waves on the same execution
   unit). **Also zero effect** at every grid size tested.

4. **LDS bank conflicts as data corruption (not just slowdown).** The
   transposed-B read's `k_half * row_pitch_bytes` term is, by construction
   (`get_gemm_index_for_src_matrix_transposed` in `python/operations/
   wmma_mapping.py` implements this as a left-shift, i.e. it assumes a
   power-of-2), always an exact multiple of 256 bytes = 64 banks * 4B --
   meaning `k_half=0` and `k_half=1` are architecturally guaranteed to land in
   the same LDS bank. This looked promising as a mechanism, but ISA doc S11.2
   states explicitly that bank conflicts are "designed to prevent the
   attempted concurrent accesses to the same bank by turning them into serial
   accesses" -- a throughput cost, not a documented correctness hazard -- and
   this same shift-based formula/conflict is shared by the fp16/bf16 transpose
   read path, which does not reproduce the bug. Not disproven outright (not
   independently tested with a re-derived non-power-of-2 pitch), but the
   documented behavior and the fp16/bf16 comparison both argue against it as
   the sole mechanism.

## What's confirmed

- Not tile-size-specific (64x64 AND 128x128 both fail, just at different
  occupancy thresholds -- smaller tile => higher occupancy per WGP => fails at
  a smaller total workgroup count).
- Not direction-specific (fwd AND bwd both fail).
- Is specific to `v_wmma_f32_16x16x4_f32` -- fp16/bf16 (wider, ISA-table-listed
  "XDL" WMMA forms, `gemm_k_per_block=32`) did not reproduce up to n=256
  (~50k workgroups), well past where fp32 fails.
- Is occupancy/contention-triggered, not a static per-thread addressing bug
  (small grids pass 100% reliably; the same kernel, same shapes, larger grid
  fails).
- `lds_double_buffer=1` (disjoint read/write LDS regions, no same-address
  reuse across iterations) reliably fixes it in every configuration tested
  (bwd 64x64, bwd 128x128, fwd 64x64 -- all retested up to 2-4x their original
  failure threshold with zero failures after the fix).

## Root mechanism, now characterized

A standalone, MISA-independent kernel (`docs/gfx1250_fp32_wmma_race_repro/`,
`lds_barrier_visibility_test`) isolates the barrier/LDS-visibility question
with **no WMMA instruction at all**, ruling out every WMMA-specific theory
above as *necessary* for the underlying hazard (WMMA-F32's tight, low-latency
loop body just makes it easy to expose in the original kernel; it isn't the
cause). Per workgroup (64 threads = 2 wavefronts of 32): every thread writes
a per-iteration-unique magic value to its own LDS slot, `s_wait_dscnt 0x0`,
`s_barrier_signal`/`s_barrier_wait`, then every thread reads *all 64 slots*
(both waves' data) and checks each against the value that iteration should
have produced.

Below ~1000-3000 concurrent workgroups: zero mismatches, always. Above that,
mismatches appear and scale with occupancy -- matching the WMMA kernel's
occupancy-dependence exactly. But unlike the WMMA case, this kernel also
captures full diagnostic state per mismatch (`{slot_i, iter, actual,
expected}`, written by the mismatching lane itself at full speed, no
debugger/breakpoint involved -- see the repro's README for why an
interactive `rocgdb` session was deliberately avoided: single-stepping this
would almost certainly perturb the exact timing the race depends on and mask
it). Aggregated over 618k captured mismatches (`./repro 20000`):

```
slot_i histogram (nonzero only):
    slot 31: 480       (0.08%)
    slot 63: 617664     (99.92%)
staleness histogram (iters behind):
    1 iters behind: 618144   (100%)
```

**This is a precise, mechanistic signature, not generic flakiness:**

- **100% of mismatches are exactly 1 loop iteration stale** -- never 2+,
  never garbage. A wave crosses the barrier and reads the *previous*
  iteration's value from some other wave's slot.
- **>99.9% of mismatches are reads of slot 63** (`tid=63` = lane 31, the
  *last* lane, of the *last* wave, tid range 32-63). The rare remainder are
  slot 31 (lane 31 of the *first* wave) -- same "last lane of a wave"
  pattern, just far less frequently the one that's stale. Every other slot
  (0-30, 32-62): zero observed mismatches, ever.

This points at something specific to how the **last lane's** contribution to
a full-wavefront-width `ds_write_b32` is drained/acknowledged -- as if
`s_wait_dscnt`'s completion signal for a wave can be satisfied before the
highest-lane-index write of a preceding LDS store has actually committed,
while lanes 0-30 are reliably already visible by that point. This is
consistent with (and narrows down) the WMMA kernel's failure: BWD's transpose
read pattern and the tight fp32 K=4 loop simply made this specific,
low-probability-per-iteration hazard easy to hit at moderate occupancy;
fp16/bf16's wider loop bodies and FWD's different addressing evidently
needed much higher occupancy to expose the same underlying issue (FWD did
eventually reproduce too, at n=128 -- see the table above -- consistent with
this being a generic hazard rather than something WMMA/opcode-specific).

**Not yet answered:** *why* the last lane specifically -- whether this is a
DScnt-tracking artifact (e.g. the counter is decremented once a majority/
some-but-not-all lanes' banked writes land, rather than waiting for the
slowest bank), a genuine LDS pipeline drain-ordering issue at high
occupancy, or something else entirely. That needs AMD-side hardware/
microcode knowledge this investigation doesn't have access to.

## Reproducer for escalation (AMD / hardware team)

**Prefer `docs/gfx1250_fp32_wmma_race_repro/` for handoff** -- it's smaller,
has no MISA/Python/conv dependency at all, isolates the mechanism without any
WMMA instruction, and includes the precise last-lane/1-iteration-stale
characterization above with reproducible aggregate statistics. The
MISA-based repro below is kept for completeness/provenance (it's what this
investigation started from), but the standalone one is the better artifact
to actually send.

Minimal, self-contained MISA-based repro package:

```bash
# 1. Generate the kernel (from repo root)
python3 igemm_codegen.py -d /tmp/wmma_f32_repro \
    <(sed 's/lds_double_buffer.*=.*1//' config/igemm_bwd_gtc_gfx1250_nhwc_fp32.config)

# 2. Run at a passing grid size (sanity check -- should be 100% valid:y)
cd /tmp/wmma_f32_repro
for i in 1 2 3; do
  ./conv_driver.exe conv -n 4 -c 128 -H 60 -W 80 -k 8 -y 1 -x 1 -p 0 -q 0 \
    -u 1 -v 1 -l 1 -j 1 -g 1 -F 2 -V 1 \
    --in_layout NHWC --fil_layout NHWC --out_layout NHWC
done

# 3. Run at a failing grid size (same kernel binary, same K-loop depth,
#    only the number of workgroups changes) -- expect intermittent valid:n
for i in 1 2 3 4 5; do
  ./conv_driver.exe conv -n 24 -c 128 -H 60 -W 80 -k 8 -y 1 -x 1 -p 0 -q 0 \
    -u 1 -v 1 -l 1 -j 1 -g 1 -F 2 -V 1 \
    --in_layout NHWC --fil_layout NHWC --out_layout NHWC
done
```

**Comments for the reviewing team:**

- The 64x64 kernel's K-loop body (`igemm_bwd_gtc_gfx1250_nhwc_fp32_064x064x004.inc`,
  label `..._wmma_body`) is: `s_wait_dscnt 0x0` / `s_barrier_signal -1` /
  `s_barrier_wait -1` / read this iteration's A+B tile from LDS / 8x
  `v_wmma_f32_16x16x4_f32` / global-load next tile / `s_wait_loadcnt 0x0` /
  `ds_write_b128` the next tile into the *same* LDS addresses just read from /
  branch back. This is a standard single-buffered software-pipelined GEMM
  K-loop and (per S5.6/S5.7.1.4) should be a complete, correct fence.
- `V_WMMA_F32_16X16X4_F32` (opcode 93, Table 43) is the only WMMA form in
  Table 43 **not** listed in the WMMA data-hazard table (S7.12.1, "Dense WMMA
  Instructions") -- every other form (F16/BF16/FP8 variants/IU8/F4/SWMMAC
  variants) has an explicit row specifying required V_NOPs/coexec slots for
  RAW/WAW/WAR hazards around its Matrix A/B/D operands. It's unclear whether
  this omission means the form has no such hazard, or whether the table is
  simply incomplete for it.
- We inserted up to 20 `V_NOP` immediately after `s_barrier_wait -1` (before
  the LDS read that reuses the WMMA's prior source registers) and separately
  tried `s_setprio 1`/`s_setprio 0` bracketing the WMMA issue burst (per
  S5.7.2.1's `DISABLE_XDL_ARB_STALL` discussion) -- neither changed the
  failure rate at all, at any tested grid size. This argues against a simple
  same-wave instruction-spacing hazard of the kind Table in S7.12.1 describes
  for the other WMMA forms.
- The bug requires >=2 K-loop iterations (i.e. the LDS buffer must actually be
  read, then overwritten, then read again) and a sufficiently large grid
  (empirically ~1500+ concurrently-schedulable 64-thread workgroups for the
  64x64 tile; ~12000+ for the 128x128 tile). Below that, 100% reliably
  correct across many repeated runs with fresh random data each time.
- Switching to double-buffered LDS (disjoint read/write addresses per
  iteration, no barrier-mediated same-address reuse) makes the bug
  disappear entirely, retested well past the original failure threshold.

This looks consistent with either an unlisted hazard specific to
`V_WMMA_F32_16X16X4_F32`, or a genuine multi-wave contention bug in the
matrix execution unit for this opcode -- we were not able to distinguish
between these from software alone.

## wrw caveat

`wrw`'s grid size (workgroup count) is driven by `k`/`c` (the M/N GEMM
dimensions for weight-gradient), not by `n`/`ho`/`wo` (which only feed the
*contraction* dimension) -- so the `-n <big>` scaling trick used above to
raise fwd/bwd occupancy does not raise wrw's workgroup count the same way.
`wrw` was not independently pushed to a high-occupancy failure repro; it
shares byte-for-byte the same `wmma_main_loop.py` code path and the same
`v_wmma_f32_16x16x4_f32` opcode, so `lds_double_buffer=1` was applied to its
fp32 configs proactively rather than reactively. If revisiting this, scale
occupancy via large `-k`/`-c` (with `-g` group count also viable) instead of
`-n`.

## Where the fix lives

- `config/igemm_{fwd,bwd,wrw}_gtc_gfx1250_nhwc_fp32.config` -- both the 128x128
  and 64x64 sections now set `lds_double_buffer = 1` unconditionally, with a
  comment pointing back to this doc.
- Any *new* fp32 WMMA config (`gemm_k_per_block=4`, i.e. using
  `v_wmma_f32_16x16x4_f32`) must set `lds_double_buffer=1` too -- see
  `AGENTS.md`'s Known Issues pointer.

## Ideas for whoever picks this up next

1. ~~Attach `rocgdb` to a controlled failing run and compare actual register
   contents against expected~~ -- **done, via a safer non-debugger route**:
   see "Root mechanism, now characterized" above and
   `docs/gfx1250_fp32_wmma_race_repro/`. An interactive `rocgdb` session was
   deliberately not used for the *investigation* (see that repro's README for
   why -- perturbing timing risks masking an occupancy-dependent race), but
   now that the failure is precisely localized (last lane of a wave, always
   exactly 1 iteration stale), a *targeted* rocgdb session -- e.g. a hardware
   watchpoint on the last lane's specific LDS address, or correlating
   `s_barrier_wait` exit with the DS pipe's internal completion state if
   such register/perf-counter visibility exists -- is a much more tractable
   next step than the original open-ended "compare against expected" ask,
   now that we know exactly what to look for.
2. Try re-deriving the transposed-B LDS layout with a non-power-of-2 row pitch
   (breaks the guaranteed same-bank alignment described in "ruled out" item 4)
   as an independent variable -- would help separate "bank conflict adjacent"
   from "opcode-specific" as the trigger, even though the current evidence
   leans away from bank conflicts alone.
3. Check for gfx1250 microcode/firmware updates or known-errata lists from AMD
   specifically mentioning either `V_WMMA_F32_16X16X4_F32` or `S_BARRIER_WAIT`
   under high occupancy -- and specifically ask about last-lane/highest-
   bank DS-write completion signaling, given how precisely that's now
   localized.
4. If AMD confirms a hardware erratum, this doc's mitigation
   (`lds_double_buffer=1`) can likely stay permanent regardless -- double
   buffering is a reasonable default for a production K-loop schedule anyway,
   independent of this bug.
5. Try padding LDS layouts so that the write pattern's "last lane" doesn't
   land in the last bank/address of a burst (e.g. reorder which lane owns
   which slot, or add trailing padding) -- if the hazard is specifically
   about bank/address position rather than lane index per se, this would
   distinguish the two and might suggest a second, more targeted mitigation
   than full double-buffering.
