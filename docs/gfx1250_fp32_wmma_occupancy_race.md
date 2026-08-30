# gfx1250 fp32 WMMA occupancy-dependent correctness bug (KNOWN ISSUE, unresolved)

Status: **mitigated (lds_double_buffer=1 mandatory), root cause NOT confirmed**.
Affects: every gfx1250 WMMA config using `v_wmma_f32_16x16x4_f32` (i.e. any
`precision=fp32` WMMA config with `gemm_k_per_block=4`) -- both `fwd` and `bwd`
confirmed by direct testing; `wrw` presumed affected (identical code path, not
independently stress-tested -- see "wrw caveat" below). Both 64x64 and 128x128
tiles confirmed affected.

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

## Unconfirmed: what's actually happening in hardware

Given (2) and (3) above -- the two software-visible WMMA-hazard-avoidance
mechanisms ISA doc S7.12.1 documents -- both had zero effect, the corruption
is very likely not something fixable by CPU-visible instruction reordering or
issue-priority hints at all. This points at either:

- an unlisted/undocumented hazard specific to `V_WMMA_F32_16X16X4_F32` (its
  absence from the S7.12.1 hazard table, unlike every other WMMA form, may be
  a documentation gap rather than "this form has no hazard"), or
- a genuine multi-wave/multi-workgroup interference bug in the shared matrix
  execution unit specific to this opcode, only exposed under concurrent
  issue from many waves.

Neither has been confirmed. Confirming this needs register-level hardware
tracing (`rocgdb` attached to a controlled failing run, comparing the actual
values landing in `v_c`/`v_a`/`v_b` against wave-scheduling state -- see
[[gpu_hardware_debug_technique]] in this project's memory notes for the
general technique) or escalation to AMD with the reproducer below.

## Reproducer for escalation (AMD / hardware team)

Minimal, self-contained repro package:

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

1. Attach `rocgdb` to a controlled failing run (small enough to single-step,
   large enough to reproduce -- try n=24..40 range for bwd 64x64) and compare
   actual `v_c`/`v_a`/`v_b` register contents against expected, timed against
   `s_barrier_wait`/WMMA-issue events, per [[gpu_hardware_debug_technique]].
2. Try re-deriving the transposed-B LDS layout with a non-power-of-2 row pitch
   (breaks the guaranteed same-bank alignment described in "ruled out" item 4)
   as an independent variable -- would help separate "bank conflict adjacent"
   from "opcode-specific" as the trigger, even though the current evidence
   leans away from bank conflicts alone.
3. Check for gfx1250 microcode/firmware updates or known-errata lists from AMD
   specifically mentioning `V_WMMA_F32_16X16X4_F32` under high occupancy.
4. If AMD confirms a hardware erratum, this doc's mitigation
   (`lds_double_buffer=1`) can likely stay permanent regardless -- double
   buffering is a reasonable default for a production K-loop schedule anyway,
   independent of this bug.
