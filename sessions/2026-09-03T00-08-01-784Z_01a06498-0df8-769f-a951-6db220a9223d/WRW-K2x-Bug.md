{
  "summary": "Root-caused and fixed wrw k2x (gemm_k_per_block=64) correctness bug on gfx1250.",
  "root_cause": "python/igemm/igemm_wrw_gtc_wmma_nhwc.py, _emit_sst_all_chunks (line 1649): missing s_wait_loadcnt 0x0 between _emit_gld_chunk_load (issues global load, does not wait) and _emit_sst_chunk (ds_write_b128 to LDS). The store reads v_gld before global_load_dwordx4 completes, storing stale/undefined data to LDS. Every analogous method in the codebase (bwd _emit_sst_all_chunks, fwd _emit_sst_all_chunks_row, wrw _emit_sst_remaining_chunks) includes the wait — only wrw _emit_sst_all_chunks omitted it. Only triggered when num_k_chunks>1 (gemm_k_per_block>inst_wmma.k), i.e. the k2x case.",
  "fix": "One-line fix: added s_wait_loadcnt 0x0 between _emit_gld_chunk_load and _emit_sst_chunk in _emit_sst_all_chunks, matching the identical pattern in bwd/fwd and wrw _emit_sst_remaining_chunks.",
  "validation": {
    "wrw_k2x": {
      "n256_c2048_H14_W14_k2048_grid256": "valid:y",
      "n128_c1024_H17_W17_k1024_grid64": "valid:y",
      "n64_c1024_H28_W28_k1024_grid64": "valid:y"
    },
    "wrw_non_k2x_baseline": "valid:y (no regression)",
    "fwd_k2x_regression": "valid:y",
    "bwd_k2x_regression": "valid:y"
  },
  "commit": "10e63c5 on branch users/SreecharanGundaboluAMD/gfx1250_bringup",
  "doc": "docs/gfx1250_wrw_k2x_correctness_fix.md"
}