# B4 Analysis: LDS Padding for Transposed Operands (bwd/wrw)

**Date:** 2026-09-03  
**Report ref:** §7/B4, OPT-1, W-3  
**Status:** Deferred — requires different mechanism than fwd's B2

## Background

B2 (commit `74bbdc3`) successfully threaded `lds_bytes_per_row` through the
fwd generator, delivering +23% on 1×1 shapes. B4 attempts to extend the same
mechanism to bwd and wrw, where one or both operands are **transposed** in LDS.

## Problem

The fwd padding works because the store and load both use the same row stride
(`bytes_per_row = gemm_k_per_block * data_byte`). Padding this one stride fixes
both store and load bank conflicts.

For transposed operands (bwd's B, wrw's A and B), the store and load use
**different strides**:

- **Store**: `v_sst_os = tid * bytes_per_row` — data laid out in thread order,
  each thread writes `bytes_per_row` bytes. K-rows are implicitly contiguous:
  tids 0..N/gemm_k_per_block-1 fill K-row 0, next group fills K-row 1, etc.
- **Load**: `row_pitch = gemm_{m,n}_per_block * data_byte` — data read as
  `[K rows][M or N cols]`, each K-row is `row_pitch` bytes apart.

When `bytes_per_row` (64 for fp16) is padded to 80, the store places 16 bytes
of padding after each 64-byte chunk. But the load expects K-rows at
`row_pitch = 256` byte intervals. With padding, K-row 0 should be [0, 256) but
tid 3's data lands at [240, 304) instead of [192, 256). **The store and load
layouts disagree.**

Padding `row_pitch` instead of `bytes_per_row` doesn't help either: the store
still places K-rows contiguously (at `gemm_n_per_block * data_byte` byte
intervals), so the load's padded `row_pitch` would read past K-row 0 into K-row 1.

## Root Cause

The transposed-operand store uses a 1D linear layout (thread order), while the
load uses a 2D transposed layout (K-row order). These two layouts are only
consistent when there's no padding — the thread-order store happens to fill
K-rows contiguously. Padding either stride breaks this coincidence.

## What Would Be Needed

To pad the transposed operand's LDS layout, the store must place data at
`K_row * padded_row_pitch + col_offset` instead of `tid * bytes_per_row`.
This is a fundamentally different store pattern — it requires:
1. Computing K_row and col_offset from tid (a decomposition, not a simple multiply)
2. A separate store offset VGPR (can't share `v_sst_os` with A's unpadded layout)
3. Changes to `_emit_sst_all_chunks` / `_emit_sst_remaining_chunks` to use the
   transposed store pattern

This is a deeper change than B2's simple stride substitution. It's closely
related to the report's W-3 ("Sweep the transposed LDS row pitch for wrw and
bwd") which is listed as a separate item.

## Recommendation

1. **B4 for the untransposed A operand in bwd**: could be done independently
   (A's store and load both use `bytes_per_row`, same as fwd). This would give
   partial benefit. However, bwd's A-side bank conflict is the same 64 B stride
   as fwd's, while the dominant conflict is B's 256 B transposed pitch (31% of
   wrw runtime per §9.3). Padding A alone gives marginal benefit.

2. **Full B4 (transposed operand padding)**: defer to W-3, which is the report's
   own separate item for this optimization. W-3 should implement the transposed
   store pattern described above.

3. **B5 (emit `lds_row_pad` sections from config scripts)**: can proceed for fwd
   only, since bwd/wrw padding isn't ready. The `_all` master config's
   fastest-tunable search will use padded fwd sections where they help and
   unpadded bwd/wrw sections elsewhere.
