# gfx1250 WMMA: XDLOPS → WMMA Porting Opportunities

**Scope**: Code-derived gap analysis only. No docs/ research was consulted; findings come from comparing the legacy XDLOPS generators (`python/igemm/igemm_{fwd,bwd,wrw}_gtc_nhwc.py`), the new WMMA generators (`python/igemm/igemm_{fwd,bwd,wrw}_gtc_wmma_nhwc.py`), and the shared operations/base classes.

**Date**: 2026-08-31

## Why this comparison matters

The WMMA path is a ground-up rewrite for gfx1250, but the legacy XDLOPS generators accumulated years of micro-optimizations that are still absent or only partially present in WMMA. Some of the most valuable near-term wins are not new ideas — they are features already proven in one WMMA direction that simply need to be ported to the others.

## Tier 1 — High impact, relatively contained

These are the cheapest wins because the implementation pattern already exists somewhere in the WMMA codebase.

### 1. `main_loop_interleave` → bwd and wrw  ✅ DONE (2026-08-31)
- **Status**: Implemented in all three directions. Fwd sets `interleave_a=interleave_b=True`; bwd and wrw set `interleave_a=True, interleave_b=False` (B is transposed, reuses `v_gld_b` as scratch — interleaving would clobber in-flight loads). wrw redirects A's `shared_load_a` scratch to a dedicated `v_scratch` VGPR.
- **Design**: Single `ctrl.interleave` replaced with per-operand `ctrl.interleave_a`/`ctrl.interleave_b`; `emit_interleaved_substeps` rewritten as `emit_mixed_substeps` handling mixed interleave/non-interleave per operand.
- **Performance**: No improvement measured on bwd or wrw. BWD is ~7-10% **slower** with interleave (0.90-0.93x baseline speed). WRW is flat (~1.0x, within noise). The overhead of per-chunk `s_wait_loadcnt`+`ds_write_b128` round-trips outweighs the latency-hiding benefit for these directions' memory access patterns. The feature is correct and available but not beneficial for current tile shapes.
- **What it does**: Overlaps global loads/stores with WMMA execution in the hot K-loop.
- **Impact**: High for fwd (where both operands are untransposed and benefit from full interleaving); neutral/negative for bwd/wrw with current shapes.

### 2. `saddr_global_load` (32-bit SADDR global loads) → bwd and wrw
- **Status**: Implemented in **fwd** only (`igemm_fwd_gtc_wmma_nhwc.py:263-275`).
- **Gap**: Missing in bwd and wrw.
- **What it does**: Replaces 64-bit VADDR pairs with a 32-bit VGPR byte offset plus an SGPR scalar base (`global_load_dwordx4 vdst, v_off, s_p_base offset:N`). Saves 1 VGPR per address pair and removes carry-chain overhead.
- **Impact**: High for wrw especially — it is near the VGPR ceiling. The user's own backlog already tracks this as P3 with fwd pilot done and bwd/wrw follow-up noted.

### 3. `async_global_load` → wrw (and B operand of bwd)
- **Status**: Fully implemented in **fwd**; A-only in **bwd** (`igemm_bwd_gtc_wmma_nhwc.py:1599` hardcodes `async_global_to_lds_b = False`); **not wired in wrw** (`igemm_wrw_gtc_wmma_nhwc.py:286` has only the mutual-exclusion assert).
- **What it does**: Uses gfx1250's async global-to-LDS / direct-to-register load path to hide global memory latency behind compute.
- **Impact**: High for wrw — wrw's GEMM_K is N·Ho·Wo, the largest of the three directions, making it the most memory-latency-bound.

### 4. `gsplit_stagger` → fwd and bwd
- **Status**: Implemented in **wrw** only (`igemm_wrw_gtc_wmma_nhwc.py:845-852`).
- **Gap**: Missing in fwd and bwd.
- **What it does**: Staggers the first global-load burst of each split-K shard by `(bz mod 128)` elements to scatter cache-line collisions.
- **Impact**: Medium — benefits split-K configurations at high split counts.

### 5. `local_prefetch_num > 1` validation and use in wrw
- **Status**: Wired in wrw (`igemm_wrw_gtc_wmma_nhwc.py:1749`) but untested and currently asserted incompatible with `tdm_global_load` in `wmma_main_loop.py:284`.
- **What it does**: Multi-slot LDS prefetch pipeline that issues LDS reads ahead of WMMA consumption.
- **Impact**: High if the main loop is LDS-read-latency bound. wrw's transposed strided LDS reads are especially latency-sensitive.

### 6. `wmma_acc_high_bank` / VGPR-MSB → wrw
- **Status**: Implemented in **fwd** and **bwd**; missing in **wrw**.
- **What it does**: Places the WMMA accumulator `v_c` in the high VGPR bank ([128..255]) to avoid bank conflicts with A/B operand registers.
- **Impact**: Medium — reduces accumulator-read bank conflicts during the WMMA burst.

## Tier 2 — Medium impact, more invasive

These require new WMMA-specific code but are grounded in proven XDLOPS or cross-arch patterns.

### 7. `tensor_a_pass_through` / `tensor_b_pass_through` (LDS bypass)
- **Status**: Full support in XDLOPS (`igemm_fwd_gtc_nhwc.py:69,256-300,851,940-976`); not implemented in any WMMA direction.
- **Gap**: `igemm_base.py` already declares the fields (`:358-359`) and correctly zeroes LDS allocation when set (`:907-908`), but `wmma_main_loop.py` has no pass-through branch.
- **What it does**: Skips LDS entirely for one operand, feeding global loads directly into the FMA operand registers.
- **Impact**: Medium-High for skinny shapes where LDS bandwidth is the bottleneck.

### 8. `merge_e` (fold c×y×x into GEMM_K)
- **Status**: Full support in XDLOPS fwd/bwd/wrw; not implemented in any WMMA direction.
- **Gap**: WMMA deliberately uses a runtime y×x tap loop (`emit_kernel_tap_loop`) for code-size reasons.
- **What it does**: Folds the full filter-tap extent into a single flat GEMM_K axis with carry-propagating address deltas, eliminating per-tap loop overhead.
- **Impact**: Medium-High for large filters (3×3, 5×5) where the tap-loop overhead matters.

### 9. `source_access_order` (M-major vs N-major grid dispatch)
- **Status**: Supported in XDLOPS; not implemented in WMMA.
- **Gap**: `igemm_base.py` declares the field (`:376`) and encodes it into kernel names (`:1335`), but WMMA generators use a fixed bx→M, by→N mapping.
- **What it does**: Allows launching blocks in M-major or N-major order to improve L2 reuse of input vs. weight.
- **Impact**: Medium for large convolutions where L2 locality dominates.

### 10. Configurable `vector_store` width in WMMA epilogue
- **Status**: Sophisticated logic in XDLOPS (`igemm_fwd_gtc_nhwc.py:140-183`); not tunable in WMMA.
- **Gap**: WMMA store width is hardcoded in `python/operations/coalescing_store_wmma.py`.
- **What it does**: Selects epilogue vector write width (1/2/4/8/16 elements) based on precision, split-K, and tile geometry.
- **Impact**: Medium — wider vector stores reduce store instruction count on aligned shapes.

### 11. `global_prefetch_a_num` / `global_prefetch_b_num` (VGPR double-staging)
- **Status**: Implemented in XDLOPS (`mfma_main_loop.py` uses `v_gld_a_gpf`/`v_gld_b_gpf`); absent in WMMA.
- **What it does**: Holds the next tile's in-flight global data while the current tile computes, hiding global latency across iterations.
- **Impact**: Medium — requires VGPR budget headroom but can significantly reduce stall cycles.

### 12. Main-loop LDS padding (`lds_pad_m` / `lds_pad_n`)
- **Status**: Implemented in XDLOPS (`igemm_base.py:906,913-914`; `mfma_main_loop.py:fctrl.lds_pad_m/n`); `igemm_base.py:get_lds_pad()` returns 0 for non-XDLOPS.
- **Gap**: WMMA only has `epilogue_lds_pad`; the main-loop A/B staging region is unpadded.
- **What it does**: Adds row-level padding in LDS to break periodic bank conflicts.
- **Impact**: Medium — the user's own profiling already found LDS conflicts are not currently a bottleneck, so this is speculative for future tile shapes.

## Tier 3 — Lower impact or niche

| # | Feature | Status | Notes |
|---|---|---|---|
| 13 | `move_slice_window_accumule_functor` (bwd) | Missing in WMMA | XDLOPS bwd uses an extra slice-window advance for multi-tap when `merge_e==0`. WMMA bwd handles it inline. |
| 14 | `wmma_epilogue_chunked` for wrw | Missing in wrw only | Reduces peak LDS for large tiles. fwd/bwd have it; wrw's `ctrl_coalescing_store_wmma` setup does not set it. |
| 15 | `multihead` on-device tap dispatch | Not in WMMA | XDLOPS computes tap indices from block index. WMMA's runtime tap loop largely subsumes this. |
| 16 | `precache_soffset` | Not in WMMA | Caches scalar offsets for multi-row global loads. WMMA's persistent VADDR design largely removes the need. |

## Features correctly absent from WMMA

These are XDLOPS-specific workarounds that do not apply to gfx1250 WMMA:

- **`bf16_1k_in_fp16`**: gfx90a-specific workaround; gfx1250 has native bf16 WMMA instructions.
- **AGPR-to-VGPR transfer loop in coalescing store**: WMMA accumulates directly in VGPRs, so this structural step is unnecessary.
- **SRD `buffer_load_dwordx4`**: XDLOPS's 4-SGPR buffer-descriptor model is not the gfx1250 global-load path. The SADDR pilot is the closer analogue.

## Bottom line

The fastest performance wins are **cross-porting already-proven WMMA features across directions**:

1. ~~`main_loop_interleave` → bwd, wrw~~ ✅ Done — implemented but no perf gain on bwd/wrw (see item #1 above)
2. `saddr_global_load` → bwd, wrw
3. `async_global_load` → wrw, and B operand of bwd
4. `gsplit_stagger` → fwd, bwd
5. `wmma_acc_high_bank` → wrw
6. `local_prefetch_num > 1` validation → wrw

After those are exhausted, the next most impactful XDLOPS-derived additions are:

- `tensor_a/b_pass_through` (new WMMA main-loop branch)
- `merge_e` (eliminate tap-loop overhead)
- `source_access_order` (L2 locality)
- configurable `vector_store` and generalized `coalescing_store_groups`

These are larger rewrites, but the modular structure of `wmma_main_loop.py` and `coalescing_store_wmma.py` means each can be added as a new control-field + branch rather than restructuring the whole generator.
