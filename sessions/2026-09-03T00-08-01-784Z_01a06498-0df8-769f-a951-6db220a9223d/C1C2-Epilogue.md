{
  "acceptance_criteria": {
    "1_tunable_added": "wmma_fp16_output tunable added in Python (igemm_base.py), C++ (igemm_gtc_base.h), and kernel name mangling (_f16o suffix in both, kept in sync)",
    "2_packed_epilogue": "direct_store epilogue in coalescing_store_wmma.py modified to pack pairs of f32 -> fp16x2/bf16x2 via v_permlane_xor_b32 + v_cvt_pk_f16_f32/v_cvt_pk_bf16_f32, then global_store_dword (only even lanes store)",
    "3_dtype_alloc_byte_override_dropped": "conv_driver.cpp dtype_alloc_byte now includes is_wmma_fp16_output in the data_byte branch; verification paths updated to use tensor_cast kernels for fp16->f32 expansion",
    "4_correctness": "All 9 runs (3 shapes x 3 independent launches) pass valid:y",
    "5_performance_measured": "3-run median speedup: 1.087x-1.145x vs f32 output baseline",
    "6_nrms": "All runs valid:y (fp16 output precision within tolerance)",
    "7_findings_doc": "docs/gfx1250_c1c2_fp16_output.md written and committed",
    "8_committed": "Commit 2af9f24 on branch users/SreecharanGundaboluAMD/gfx1250_bringup"
  },
  "commit": "2af9f24",
  "correctness": {
    "shape1_n256_c2048_H14_W14_k2048": {
      "all_valid": true,
      "runs": [
        "valid:y",
        "valid:y",
        "valid:y"
      ]
    },
    "shape2_n128_c1024_H17_W17_k1024": {
      "all_valid": true,
      "runs": [
        "valid:y",
        "valid:y",
        "valid:y"
      ]
    },
    "shape3_n128_c32_H56_W56_k128_shallow_K": {
      "all_valid": true,
      "runs": [
        "valid:y",
        "valid:y",
        "valid:y"
      ]
    }
  },
  "files_changed": [
    "python/igemm/igemm_base.py (tunable def + asserts + _f16o name)",
    "python/operations/coalescing_store_wmma.py (ctrl field + packed store in _emit_direct_store)",
    "python/igemm/igemm_fwd_gtc_wmma_nhwc.py (ctrl wiring + out_elem_byte_shift + VGPR alloc + call site)",
    "python/igemm/igemm_bwd_gtc_wmma_nhwc.py (same as fwd)",
    "python/igemm/igemm_wrw_gtc_wmma_nhwc.py (same, two call sites)",
    "driver/igemm_gtc_base.h (struct field + config parsing + _f16o name)",
    "driver/conv_driver.cpp (is_wmma_fp16_output + dtype_alloc_byte + COR-004 + verification fwd/bwd/wrw)",
    "docs/gfx1250_c1c2_fp16_output.md (findings)"
  ],
  "performance": {
    "baseline_median_tflops": {
      "shape1": 509.37,
      "shape2": 422.8,
      "shape3": 77.5
    },
    "fp16o_median_tflops": {
      "shape1": 583.05,
      "shape2": 477.24,
      "shape3": 84.29
    },
    "speedup": {
      "shape1": "1.145x",
      "shape2": "1.129x",
      "shape3": "1.087x"
    }
  }
}