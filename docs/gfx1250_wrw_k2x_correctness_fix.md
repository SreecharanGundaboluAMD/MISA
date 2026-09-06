# wrw k2x (gemm_k_per_block=64) Correctness Fix

## Symptom

`config/igemm_wrw_gtc_gfx1250_nhwc_fp16_k2x.config` (`gemm_m_per_block=128,
gemm_n_per_block=128, gemm_k_per_block=64`, plain k2x baseline — no
`main_loop_interleave`, no other special tunables) produced `valid:n` (numerically
wrong output vs the GPU naive-conv reference) on large-grid shapes such as
`n256 c2048 H14 W14 k2048 y1 x1` (grid = 256 workgroups = full 256-CU grid).

This was NOT the separate grid-starvation issue (which only affects small M×N
shapes, prints an explicit warning, and is expected behavior). The failing shape
showed no grid-starvation warning yet still reported `valid:n`.

The non-k2x wrw config (`igemm_wrw_gtc_gfx1250_nhwc_fp16.config`, default
`gemm_k_per_block=32`) passed correctness (`valid:y`) on the same shape, and
fwd/bwd k2x configs also passed — isolating the bug to wrw's k2x path.

## Root Cause

**File:** `python/igemm/igemm_wrw_gtc_wmma_nhwc.py`, method `_emit_sst_all_chunks`
(lines 1649–1652 pre-fix).

`_emit_sst_all_chunks` is called by `shared_store_a_functor` and
`shared_store_b_functor` when `num_k_chunks > 1` (i.e.,
`gemm_k_per_block > inst_wmma.k`, the k2x case). It loads each K-chunk from
global memory into the reused `v_gld` buffer via `_emit_gld_chunk_load`, then
immediately stores `v_gld` to LDS via `_emit_sst_chunk` — **without an
`s_wait_loadcnt 0x0` between the load and the store**.

The `ds_write_b128` (LDS store) therefore reads `v_gld` before the
`global_load_dwordx4` has completed, storing stale/undefined data to LDS. This
corrupts the K-data for all chunks, producing silently wrong WMMA results.

The pre-fix code was:

```python
for c in range(self.num_k_chunks):
    self._emit_gld_chunk_load(v_gld, v_addr, c, v_flag=v_flag, v_flag_col=v_flag_col, saddr=saddr)
    self._emit_sst_chunk(v_gld, v_sst_os, sst_extra_off, c)
```

Every analogous implementation in the codebase includes the wait:
- `igemm_bwd_gtc_wmma_nhwc.py` `_emit_sst_all_chunks` (line 1311): has `s_wait_loadcnt 0x0`
- `igemm_fwd_gtc_wmma_nhwc.py` `_emit_sst_all_chunks_row` (line 1581): has `s_wait_loadcnt 0x0`
- wrw's own `_emit_sst_remaining_chunks` (line 1632): has `s_wait_loadcnt 0x0`

Only wrw's `_emit_sst_all_chunks` was missing it.

When `num_k_chunks == 1` (the default `gemm_k_per_block=32` case),
`_emit_sst_all_chunks` is never called — `shared_store_a/b_functor` take the
`_emit_sst_remaining_chunks` path instead, which does wait. This is why the bug
only manifested at k2x (`gemm_k_per_block=64`).

## Fix

Added the missing `s_wait_loadcnt 0x0` between `_emit_gld_chunk_load` and
`_emit_sst_chunk` in `_emit_sst_all_chunks`:

```python
for c in range(self.num_k_chunks):
    self._emit_gld_chunk_load(v_gld, v_addr, c, v_flag=v_flag, v_flag_col=v_flag_col, saddr=saddr)
    self._emit(f"s_wait_loadcnt 0x0")
    self._emit_sst_chunk(v_gld, v_sst_os, sst_extra_off, c)
```

This is a one-line fix, matching the identical pattern in bwd's and fwd's
analogous methods, and in wrw's own `_emit_sst_remaining_chunks`.

## Validation

### wrw k2x (the fixed config)

All shapes: `conv_driver.exe convfp16 ... -F 4 -V 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC`

| Shape | Grid | Result |
|-------|------|--------|
| n256 c2048 H14 W14 k2048 y1 x1 | 256 (full) | **valid:y** |
| n128 c1024 H17 W17 k1024 y1 x1 | 64 | **valid:y** (grid-starvation warning, expected) |
| n64 c1024 H28 W28 k1024 y1 x1 | 64 | **valid:y** (grid-starvation warning, expected) |

### wrw non-k2x baseline (regression check)

| Shape | Result |
|-------|--------|
| n256 c2048 H14 W14 k2048 y1 x1 | **valid:y** (both 128x128x32 and 64x64x32 tunables) |

### fwd k2x regression check

| Shape | Result |
|-------|--------|
| n256 c2048 H14 W14 k2048 y1 x1 | **valid:y** |

### bwd k2x regression check

| Shape | Result |
|-------|--------|
| n256 c2048 H14 W14 k2048 y1 x1 | **valid:y** |

The fix touches only `igemm_wrw_gtc_wmma_nhwc.py`'s `_emit_sst_all_chunks` — a
wrw-specific method — so fwd/bwd are unaffected by construction. Verified anyway
as cheap insurance.
