# bwd `lds_double_buffer` + `lds_row_pad` stacking produces `-nan` output (R7)

**Status: unresolved, mitigated by rejection.** `driver/igemm_bwd_gtc_driver.h`'s WMMA
`tunable_is_valid` now unconditionally rejects `lds_double_buffer=1 && lds_row_pad>0` for
bwd. No committed bwd config sets both today outside the new (also-excluded)
`igemm_bwd_gtc_gfx1250_nhwc_fp16_dbuf_ldsrp.config`, so this is a no-op for the master
`_all.config` union; it exists to stop a future config or driver change from silently
re-exposing wrong-answer output.

## Context

Discovered while executing `docs/gfx1250_wmma_perf_report_v2.md` §5/§7 P0.5's recommendation
("combine `lds_row_pad=16` with `lds_double_buffer=1`... for all three directions, after R1
is fixed for wrw"). New `_dbuf_ldsrp.config` files were added for fwd/bwd/wrw (128x128 and
64x64 tiles, same shapes as each direction's existing standalone `_ldsrp.config`), folded
into the three fp16 master configs, and hardware-validated on the standing regression set.

## Symptom

fwd's and wrw's `dbuf_ldsrp` (wrw: `dbuf_dstrb_ldsrp_gkgs`) kernels are `valid:y` across
every regression shape and both tile sizes — the two mechanisms are cleanly additive there.
bwd's `dbuf_dstrb_ldsrp` kernels (both 128x128 and 64x64 tiles) instead print `-nan` output
on **every** regression shape tested:

```
$ ./conv_driver.exe convfp16 -n 128 -c 1024 -H 17 -W 17 -k 1024 -y 1 -x 1 -p 0 -q 0 -F 2 -V 1 \
    --in_layout NHWC --fil_layout NHWC --out_layout NHWC
[bwd: 5] igemm_bwd_gtcw_..._dbuf_dstrb_ldsrp, cost:0.952ms, tflops:442.350(76.70%) invalid float at 1, ref:-0.325255, pred:-nan
[bwd: 6] igemm_bwd_gtcw_..._dbuf_dstrb_ldsrp, cost:1.591ms, tflops:264.479(45.86%) invalid float at 0, ref:14.800145, pred:-nan
```

Confirmed across all 5 standing-regression shapes (1x1 and 3x3, large and small), both tile
sizes (128x128x32 and 64x64x32) — not a single-shape edge case. `-nan`, not a crash: the
kernel dispatches and completes, just produces poisoned floating-point output, most likely
from an uninitialized-or-corrupted LDS read (an `s_barrier`-raced or wrongly-addressed read
landing on garbage, then propagating through the WMMA accumulate as NaN).

## What's already known (not yet exhaustively ruled out — this is a fresh finding)

- **Both mechanisms individually are already hardware-validated correct for bwd.**
  `lds_row_pad=16` alone (`igemm_bwd_gtc_gfx1250_nhwc_fp16_ldsrp.config`, already in the
  master union since before this session) is `valid:y`. `lds_double_buffer=1` alone
  (`igemm_bwd_gtc_gfx1250_nhwc_fp16_dbuf.config`) is `valid:y` (validated in `b542b08`,
  reconfirmed in this session's fwd/bwd/wrw regression sweep). Only the **combination**
  fails for bwd.
- **fwd and wrw's identical combination both work.** Ruled out: this is not a generic
  "these two mechanisms never compose" bug — bwd is the outlier.
- **The most structurally distinctive thing about bwd vs fwd/wrw**: bwd is the only
  direction with **asymmetric A/B transpose** — A (grad_output) is untransposed, B (weight)
  is transposed (`igemm_bwd_gtc_wmma_nhwc.py`'s `__init__`: `lds_a_size =
  gemm_m_per_block * lds_bytes_per_row`, `lds_b_size = gemm_k_per_block *
  lds_row_pitch_b` — A indexed by M-rows, B indexed by K-rows with a *different* padded
  row pitch). fwd has neither operand transposed (both indexed by M/N-rows, same padded
  row pitch); wrw has **both** operands transposed (both indexed by K-rows, matching
  padded row pitches). bwd's asymmetry means `lds_a_size` and `lds_b_size` are not
  computed the same way relative to each other the way fwd's and wrw's are — a
  double-buffer toggle/offset computation that implicitly assumes a symmetric relationship
  between the two regions' sizing could plausibly break only for bwd.
- **Not yet checked**: the actual double-buffer toggle/swap address computation (where the
  kernel flips between buffer 0 and buffer 1's base offset each K-iteration) for bwd
  specifically, cross-referenced against `lds_a_size`/`lds_b_size`/`lds_single_size`'s
  values when `lds_row_pad>0` is also active. This is the natural next investigative step
  but was not pursued this session (deprioritized in favor of shipping the two directions
  that do work, and not leaving a wrong-answer kernel reachable in the interim).

## Reproduction

```
# Requires temporarily reverting the tunable_is_valid rejection in
# driver/igemm_bwd_gtc_driver.h (WMMA branch, "R7" comment) to rebuild the affected kernel.
$ python3 igemm_codegen.py config/igemm_bwd_gtc_gfx1250_nhwc_fp16_all.config -d /tmp/x
$ IGEMM_RUN_ONLY_KERNEL=igemm_bwd_gtcw_nhwc_fp16_bx0_ex0_bt128x128x32_wt16x16_wr4x4_ta1x32x1x1_1x1x1x128_tb1x32x1x1_1x1x1x128_dbuf_dstrb_ldsrp \
  ./conv_driver.exe convfp16 -n 128 -c 1024 -H 17 -W 17 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 \
  -l 1 -j 1 -g 1 -F 2 -V 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
# Expected (bug present): "invalid float ..., pred:-nan"
```

## Recommendation for whoever picks this up next

1. Read `_emit_lds_offset_setup`'s double-buffer toggle logic (bwd:
   `igemm_bwd_gtc_wmma_nhwc.py`, called once per tap from `emit_kernel_tap_loop`) alongside
   `lds_a_size`/`lds_b_size`/`lds_single_size`'s derivation in `__init__`, specifically for
   the `lds_row_pad>0 and lds_double_buffer=1` combination — compare against fwd's and wrw's
   identically-named method to find where the asymmetric-transpose case diverges.
2. Given R5's PAL-metadata lesson this session (a bug that looked like a register/hardware
   issue turned out to be a missing metadata declaration two layers away from where the
   investigation started), do not assume the bug is in the addressing math without first
   checking simpler candidates: the `workgroup_group_segment_byte_size` computed LDS
   allocation size, and whether `ds_load_tr_b`'s transpose-load path (bwd's B always uses
   it when `ds_load_tr_b` is set — check whether the combined config exercises it) has its
   own padding/pitch assumption that doesn't account for the double-buffer's second-buffer
   offset.
3. Do not remove the `tunable_is_valid` rejection in `driver/igemm_bwd_gtc_driver.h` without
   a hardware-validated fix — this is silent wrong-answer output (`-nan`), not a crash, and
   must not regress silently back into the master config's search path.

## Disposition of this session's work

fwd's and wrw's `_dbuf_ldsrp.config` are hardware-validated correct and shipped (folded into
their respective master `_all.config` unions). bwd's `_dbuf_ldsrp.config` remains as a
standalone file (useful for reproducing this bug) but is excluded from the master union's
search path by the driver-side rejection above, and was NOT folded in a way that makes it
reachable through `conv_driver.exe`'s normal candidate search.
