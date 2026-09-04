{
  "acceptance_criteria": {
    "1_bwd_pad0_no_regression": "PASS — valid:y, 425.7 TFLOP/s (128x128), 358.3 (64x64)",
    "2_bwd_pad16_kernel_name_ldsrp": "PASS — kernel name includes _ldsrp",
    "3_bwd_pad16_valid": "PASS — valid:y, 498.3 TFLOP/s (35.2%), +17% vs pad=0",
    "4_wrw_pad0_no_regression": "PASS — valid:y, 299.6 TFLOP/s (128x128), 329.7 (64x64)",
    "5_wrw_pad16_kernel_name_ldsrp": "PASS — kernel name includes _ldsrp",
    "6_wrw_pad16_valid": "PASS — valid:y, 416.6 TFLOP/s (29.4%), +39% vs pad=0",
    "7_fwd_no_regression": "PASS — valid:y, 372.1 TFLOP/s (128x128), 332.5 (64x64), fwd generator unchanged"
  },
  "changes": {
    "python/operations/wmma_mapping.py": "Added _emit_mul_or_shift helper (v_lshlrev_b32 for power-of-2, v_mul_lo_u32 otherwise). Applied at 2 sites in get_gemm_index_for_src_matrix_transposed and get_gemm_index_for_src_matrix_transposed_ds_tr16 where row_pitch_bytes is multiplied. Byte-identical when lds_row_pad=0.",
    "python/igemm/igemm_bwd_gtc_wmma_nhwc.py": "Added lds_bytes_per_row (A untransposed, same as fwd B2), lds_row_pitch_b (B transposed, padded), threads_per_krow_b. A store/load use lds_bytes_per_row; B load uses lds_row_pitch_b. B store offset computed on-the-fly from tid*bytes_per_row + (tid//threads_per_krow)*lds_row_pad using v_tmp(2..3) scratch (no extra VGPR — bwd at 256 limit). shared_load_a step_bytes uses lds_bytes_per_row. shared_load_b row_pitch uses lds_row_pitch_b. ctrl k_substep_stride_bytes_b uses lds_row_pitch_b. LDS sizes use padded strides.",
    "python/igemm/igemm_wrw_gtc_wmma_nhwc.py": "Added lds_row_pitch_a/b (both padded). v_sst_os uses transposed formula directly (no separate VGPR — both operands transposed). shared_load_a/b use padded row_pitch. _emit_sst_chunk row_stride path uses lds_row_pitch_a. ctrl strides use padded row_pitch. LDS sizes and lds_single_size computed from padded values."
  },
  "test_config": "-n 128 -c 1024 -H 17 -W 17 -k 1024 -y 1 -x 1 -F 2(bwd)/4(wrw)/1(fwd), fp16, 128x128x32 tile, lds_row_pad=16"
}