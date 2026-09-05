{
  "analysis": "The +19.9% on 3×3 is NOT reproducible. On this faster, less-contended machine, W-6 shows only +1.18% on Shape 3. The original +19.9% was a contention artifact: the original baseline ran at 247 TFLOP/s (40% deficit vs 412 TFLOP/s here), while W-6 was less affected (297 vs 417). W-6 is a correct ~1% micro-optimization, not a 20% win.",
  "builds": {
    "baseline_dir": "/tmp/w6_baseline",
    "wig_dir": "/tmp/w6_wig"
  },
  "commit": "4ef933d",
  "findings_doc": "docs/gfx1250_w6_incremental_gather.md (new section: 'Re-benchmark on faster machine (2026-09-04)')",
  "git_log": [
    "4ef933d [WMMA][gfx1250] W-6: Re-benchmark on faster machine",
    "39a9e9c update session info",
    "215d30d [WMMA][gfx1250] E-4: Workgroup swizzle for L2 locality"
  ],
  "results": {
    "shape1_128x1024x17x17x1024_1x1": {
      "baseline_avg_tflops": 416.72,
      "baseline_cost_range_ms": "0.185-0.187",
      "baseline_tflops_range": "415.78-418.33",
      "pct_change_cost": 0.22,
      "pct_change_tflops": 0.33,
      "runs_valid": 5,
      "wig_avg_tflops": 418.08,
      "wig_cost_range_ms": "0.185-0.186",
      "wig_tflops_range": "416.33-419.63"
    },
    "shape2_256x2048x14x14x2048_1x1": {
      "baseline_avg_tflops": 785.75,
      "baseline_cost_range_ms": "0.512-0.605",
      "baseline_note": "Run 2 (695.6 TFLOP/s) is an outlier; excluding it baseline avg=808.3, W-6 is within noise",
      "baseline_tflops_range": "695.61-822.66",
      "pct_change_cost": 1.86,
      "pct_change_tflops": 1.52,
      "runs_valid": 5,
      "wig_avg_tflops": 797.69,
      "wig_cost_range_ms": "0.514-0.538",
      "wig_tflops_range": "781.71-818.45"
    },
    "shape3_64x512x28x28x512_3x3": {
      "baseline_avg_tflops": 412.17,
      "baseline_cost_range_ms": "0.571-0.577",
      "baseline_tflops_range": "410.35-414.34",
      "original_reported_pct": 19.9,
      "pct_change_cost": 1.15,
      "pct_change_tflops": 1.18,
      "runs_valid": 5,
      "wig_avg_tflops": 417.03,
      "wig_cost_range_ms": "0.564-0.577",
      "wig_tflops_range": "410.16-420.04"
    },
    "shape4_32x256x56x56x256_3x3_ablation": {
      "baseline_avg_tflops": 303.1,
      "baseline_cost_range_ms": "0.386-0.397",
      "baseline_tflops_range": "297.88-307.02",
      "pct_change_cost": 0,
      "pct_change_tflops": -0.03,
      "runs_valid": 5,
      "wig_avg_tflops": 303,
      "wig_cost_range_ms": "0.387-0.395",
      "wig_tflops_range": "299.71-305.61"
    },
    "shape5_128x64x56x56x64_1x1_ablation": {
      "note": "Not applicable: 128x128 tile too large for c=64,k=64. Both configs report 'not applicable'. Config/shape mismatch, not a W-6 regression.",
      "runs_valid": 0
    }
  },
  "total_runs": "40 valid (Shape 5 not applicable for both configs)",
  "total_valid_runs": 40
}