{
  "summary": "Extended lds_row_pad tunable to the bwd WMMA generator (igemm_bwd_gtc_wmma_nhwc.py), mirroring B2's fwd implementation.",
  "files_changed": [
    "python/igemm/igemm_bwd_gtc_wmma_nhwc.py",
    "python/operations/wmma_mapping.py"
  ],
  "changes": {
    "igemm_bwd_gtc_wmma_nhwc.py": [
      "Added validation asserts (lds_row_pad % 16 == 0, gcd checks for both A and B padded strides)",
      "Added self.lds_bytes_per_row = bytes_per_row + lds_row_pad (LDS-only A-side stride)",
      "Added self.lds_row_pitch_b = gemm_n_per_block * data_byte + lds_row_pad (LDS-only transposed B-side stride)",
      "Updated lds_a_size to use lds_bytes_per_row",
      "Updated lds_b_size to use lds_row_pitch_b",
      "Store offset: v_lshlrev_b32 -> v_mul_lo_u32 with lds_bytes_per_row",
      "A load offset: v_lshlrev_b32 -> v_mul_lo_u32 with lds_bytes_per_row",
      "B load offset: gemm_n_per_block*data_byte -> lds_row_pitch_b (both ds_load_tr_b and non-tr paths)",
      "shared_store_a_functor row_off: bytes_per_row -> lds_bytes_per_row",
      "shared_load_a_functor step_bytes: bytes_per_row -> lds_bytes_per_row",
      "shared_load_b_functor row_pitch: gemm_n_per_block*data_byte -> lds_row_pitch_b",
      "k_substep_stride_bytes_b: uses lds_row_pitch_b",
      "move_slice_window functors: UNCHANGED (use bytes_per_row global stride)",
      "TDM descriptor advance: UNCHANGED (uses bytes_per_row global stride)",
      "lds_single_size: UNCHANGED (still next_pow2)"
    ],
    "wmma_mapping.py": [
      "Added _emit_mul_or_shift helper: emits v_lshlrev_b32 for power-of-2 multipliers, v_mul_lo_u32 otherwise (needed because lds_row_pad makes row_pitch_bytes non-power-of-2)",
      "get_gemm_index_for_src_matrix_transposed: uses _emit_mul_or_shift for half_k*row_pitch_bytes",
      "get_gemm_index_for_src_matrix_transposed_ds_tr16: uses _emit_mul_or_shift for row_pitch_bytes",
      "Backward-compatible: when row_pitch_bytes is power-of-2 (lds_row_pad=0), emits v_lshlrev_b32 identically to original"
    ]
  },
  "verification": {
    "lds_row_pad=0_build": "PASS - builds successfully, no _ldsrp in kernel name",
    "lds_row_pad=16_build": "PASS - builds successfully, _ldsrp in kernel name",
    "assembly_check_pad16": "PASS - v_mul_lo_u32 with 0x50=80 (A stride) and 0x110=272 (B stride) confirmed in disassembly",
    "assembly_check_pad0": "PASS - v_lshlrev_b32 used for B load (256 is power-of-2), v_mul_lo_u32 for A store/load (encoding change matching B2 fwd)",
    "fwd_no_regression": "PASS - fwd generator still builds",
    "wrw_no_regression": "PASS - wrw generator still builds",
    "conv_driver_valid_y": "SKIPPED - no AMD GPU available on this machine (rocminfo not found)"
  },
  "note": "wmma_mapping.py was modified (not prohibited by constraints) because bwd's B operand is TRANSPOSED, requiring get_gemm_index_for_src_matrix_transposed* to handle non-power-of-2 row pitches. The fwd B2 reference didn't need this because fwd's B is untransposed. The _emit_mul_or_shift helper is backward-compatible: power-of-2 values emit v_lshlrev_b32 (identical to original), so all existing configs with lds_row_pad=0 are unaffected."
}