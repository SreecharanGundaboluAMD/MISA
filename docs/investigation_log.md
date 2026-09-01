# Investigation Log — gfx1250 MISA Bring-up Audit

## Branch / Commit
- Branch: `users/SreecharanGundaboluAMD/gfx1250_bringup`
- Commit: `29818ef162fe461a556f29bf90d005da5d03d78f`
- Date: 2026-09-01

## Files Inspected

### Python codegen framework
- `python/codegen_driver.py` (full, 430 lines) — dispatch, generator selection, emit/compile
- `python/codegen/amdgpu.py` (full, 638 lines) — arch config, precision enums, occupancy
- `python/codegen/igemm_base.py` → `python/igemm/igemm_base.py` (key ranges) — tunable contract, kernel name encoding

### WMMA generators (active gfx1250 path)
- `python/igemm/igemm_fwd_gtc_wmma_nhwc.py` (lines 1-606 read in detail) — fwd WMMA generator
- `python/igemm/igemm_bwd_gtc_wmma_nhwc.py` — delegated to BwdGenAudit subagent
- `python/igemm/igemm_wrw_gtc_wmma_nhwc.py` — delegated to WrwGenAudit subagent

### WMMA operations layer
- `python/operations/wmma.py` (full, 106 lines) — WMMA instruction definitions
- `python/operations/wmma_main_loop.py` (full, 542 lines) — main loop schedule
- `python/operations/wmma_mapping.py` (full, 439 lines) — thread/tile mapping, mapping table
- `python/operations/coalescing_store_wmma.py` (lines 1-175) — epilogue store

### gfx950 XDLOPS comparison path
- `python/operations/main_loop_graph.py` (lines 34-77, 567-680) — XDLOPS core loop graph
- `python/operations/dotx_main_loop.py` (header) — XDLOPS main loop

### C++ driver
- `driver/igemm_gtc_base.h` (lines 1-903) — tunable struct, kernel name encoding, launch, split-K
- `driver/igemm_fwd_gtc_driver.h` (lines 1-703) — fwd block/grid/validity/run
- `driver/igemm_wrw_gtc_driver.h` (lines 1-442) — wrw block/grid/validity

### Config files
- `config/igemm_fwd_gtc_gfx1250_nhwc_fp16_all.config` (full, 597 lines) — gfx1250 master fwd fp16
- `config/igemm_fwd_gtc_gfx950_nhwc_fp16.config` (lines 1-300) — gfx950 fwd fp16
- Config line/shape counts compared across all 6 direction×precision pairs

### Scripts
- `script/generate_all_configs.py` (lines 1-203) — config generation, validity rules

### Benchmark results (non-docs, in scope)
- `bench_results_gfx1250_vs_miopen_20260830.md` (full)
- `bench_results_diverse_combinatorial_20260828.md` (full)
- `bench_results_38_combinatorial_20260828.md` (full)
- `bench_results_20260831.md` (partial, in-progress)

### ISA document (targeted sections only)
- `amd-instinct-cdna5-instruction-set-architecture.md`:
  - Lines 75-293: Table of Contents
  - Lines 4173-4328: §7.12 WMMA instruction table, capabilities, restrictions
  - Lines 4328-4428: §7.12.1 WMMA data hazard requirements
  - Lines 2480-2583: §5.7.2 Dependency cases, §5.7.2.1 DISABLE_XDL_ARB_STALL, §5.8 S_DELAY_ALU
  - Lines 6627-6723: §11.2.4 LDS to VGPR Matrix Load with Transpose
  - Lines 4899-5003: §8.4 Alignment and bounds checking

## Files Intentionally Excluded
- Everything under `docs/` (per task rules)
- `amd-instinct-cdna5-instruction-set-architecture.md` read only in targeted sections (per task rules)
- `python/igemm/igemm_{fwd,bwd,wrw}_gtc.py` (legacy MAC/NCHW paths, not gfx1250)
- `python/igemm/igemm_{fwd,bwd,wrw}_gtc_nhwc.py` (XDLOPS/DLOPS nhwc paths, read only for comparison)
- `python/sequence_driver.py` (gfx908-only, not gfx1250)

## Commands Run
- `git rev-parse HEAD && git branch --show-current && git log --oneline -5`
- `ls -1` (top-level, python/, python/igemm/, python/codegen/, python/operations/, driver/, config/, script/, test/)
- `ls -1 config/ | grep -i gfx1250` and `grep -i gfx950`
- `wc -l` on 6 config files (gfx950 vs gfx1250 comparison)
- `grep -E '^gemm_m_per_block|^gemm_n_per_block|^gemm_k_per_block'` on gfx950 and gfx1250 fwd configs (tile shape diversity)

## Subagents Dispatched
- FwdGenAudit (scout) — full audit of igemm_fwd_gtc_wmma_nhwc.py
- BwdGenAudit (scout) — full audit of igemm_bwd_gtc_wmma_nhwc.py
- WrwGenAudit (scout) — full audit of igemm_wrw_gtc_wmma_nhwc.py
- Status: all three still running at time of report writing (reading ~2000-line files)

## Evidence Collected
1. Benchmark data showing MISA vs gfx950 performance gaps (fwd ~1.2x, bwd ~1.7x, wrw ~2-3.5x slower)
2. Config coverage gap: gfx950 has 30+ tile shapes vs gfx1250's 5 for fwd/fp16
3. WMMA main loop uses single fixed schedule with single-buffered LDS by default
4. XDLOPS path defaults to double-buffered LDS (lds_buffer_num=2)
5. WMMA instruction cycle field is None (latency unknown to codegen)
6. ISA hazard table requires 5 V_NOP for F16/BF16 WMMA with same A/B (RAW)
7. ISA DISABLE_XDL_ARB_STALL allows back-to-back WMMA issue (not used by MISA)
8. Register pressure: 128x128 fp16 config uses ~235+ VGPRs (approaching 256 limit)
9. WMMA k-substep drain loop is sequential (no overlap) unless main_loop_interleave=1
10. Epilogue uses per-lane scalar global_store_dword (not vectorized unless direct_store=1)
