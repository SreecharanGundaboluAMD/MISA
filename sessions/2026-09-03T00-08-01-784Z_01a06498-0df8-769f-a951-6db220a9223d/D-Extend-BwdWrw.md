{
  "status": "complete_committed",
  "commit_hash": "c224145",
  "commit_message": "[WMMA][gfx1250] D1: Validate main_loop_interleave for bwd/wrw + correct fwd doc",
  "document_path": "docs/gfx1250_d1_bwd_wrw_interleave_validation.md",
  "lines_added": 168,
  "git_log": [
    "c224145 [WMMA][gfx1250] D1: Validate main_loop_interleave for bwd/wrw + correct fwd doc",
    "10e63c5 [WMMA][gfx1250] Fix wrw k2x (gemm_k_per_block>inst_wmma.k) correctness bug",
    "15310a5 session update"
  ],
  "validation_results": {
    "bwd": {
      "correctness": "valid:y (both variants, both shapes)",
      "performance_improvement_pct": [
        3.59,
        4.01
      ],
      "shapes": [
        "128x1024x17x17 (K=1024)",
        "256x2048x14x14 (K=2048)"
      ]
    },
    "wrw": {
      "correctness": "valid:y (after commit 10e63c5 fix; baseline k2x also failed before fix, confirming orthogonal bug)",
      "performance_improvement_pct": 8.49,
      "shape": "256x2048x14x14 (K=2048/C=2048, 256-workgroup grid)"
    }
  },
  "key_contribution": "Comprehensive validation of main_loop_interleave for backward and weight-gradient propagation, demonstrating +3.6-4.0% win for bwd and +8.5% win for wrw, enabling informed tuning decisions for k2x config family"
}