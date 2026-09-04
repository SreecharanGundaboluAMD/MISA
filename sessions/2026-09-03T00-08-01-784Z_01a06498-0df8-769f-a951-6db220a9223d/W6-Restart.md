{
  "status": "WIP — correctness issue not fully resolved",
  "summary": "Implemented the wrw_incremental_gather tunable with incremental B-gather strength reduction. The incremental approach saves the ho_wo magic division (the more expensive of the two) by maintaining persistent wo_idx/ho_idx/n_idx VGPRs. The first iteration seeds via full div/rem; subsequent iterations use a wo-only magic div/rem + conditional ho-wrap via v_cndmask_b32.\n\nCorrectness: The incremental update logic is provably correct — exhaustive Python simulation over all 256 threads × all K-iterations for 14x14 shows 0/200448 mismatches. On hardware, shapes with single-wrap or no-wrap (1x33, 1x64, 14x1, 7x7, 17x17 with c=128) pass valid:y. But 14x14 (ho=14, wo=14, where both wo-wrap AND ho-wrap fire) and 128x1024x17x17 fail valid:n.\n\nRoot cause of the hardware failure is NOT yet identified. The generated assembly was verified instruction-by-instruction: magic div/rem encoding is correct, v_cmp/v_sub/v_cndmask encodings are correct, s_ho kernarg is loaded correctly (verified value=14 at offset 132 with s_wait_kmcnt before use). Three approaches were tried for the wrap logic: (1) scalar-branch loops with s_cbranch_vccnz (fundamentally broken for divergent per-lane wraps), (2) EXEC-masked loops with s_and_saveexec_b32 (caused hangs), (3) unconditional wo div/rem + v_cndmask_b32 ho-wrap (current approach — correct in simulation, fails on hardware for 14x14).\n\nChanges committed: tunable in Python+C++ + kernel name '_wig', new kernarg 'ho', new SGPR s_ho, 3 new VGPRs, _emit_b_gather_incremental method, move_slice_window_b modification. VGPR pressure: 251→254 (well within 256 limit). Findings document at docs/gfx1250_w6_incremental_gather.md. Not benchmarked due to correctness issue.",
  "files_modified": [
    "python/igemm/igemm_base.py",
    "driver/igemm_gtc_base.h",
    "driver/igemm_wrw_gtc_driver.h",
    "python/igemm/igemm_wrw_gtc_wmma_nhwc.py",
    "config/w6_test_wig.config",
    "config/w6_test_baseline.config",
    "docs/gfx1250_w6_incremental_gather.md"
  ],
  "git_commit": "d0ad423"
}