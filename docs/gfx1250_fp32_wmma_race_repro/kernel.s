// lds_barrier_visibility_test -- minimal reproducer for CDNA5 split-barrier
// not providing cross-wave LDS store visibility.
//
// Pattern (per workgroup, 2 waves × 32 lanes = 64 threads):
//   - LDS: 256 bytes (64 dwords), single-buffered
//   - Loop N times:
//     1. Wave 0 writes magic_A to LDS[tid], wave 1 writes magic_B to LDS[tid]
//     2. s_wait_dscnt 0x0  (wait for THIS wave's ds_write)
//     3. s_barrier_signal / s_barrier_wait  (wait for ALL waves)
//     4. Each wave reads ALL 64 LDS slots
//     5. Check: wave 0 must see wave 1's magic_B in slots 32-63,
//        wave 1 must see wave 0's magic_A in slots 0-31
//     6. If any slot has stale data from previous iteration, FAIL
//
// The magic values change per iteration (encoded as iteration|wave|slot),
// so stale reads from iteration N-1 are detectable.
//
// Build: clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx1250 kernel.s -o kernel.hsaco
//        hipcc -std=c++17 host.cpp -o repro
// Run:   ./repro [num_workgroups]

.set N_ITERS, 100    // loop iterations

.set s_ka,     0
.set s_bx,     2
.set s_p_out,  4   // output fail-count ptr (s4:s5)
.set s_p_diag, 6   // diagnostic buffer ptr (s6:s7) -- per-thread last-mismatch record
.set s_kitr,   8
.set s_tmp,    9

.set v_tid,    0
.set v_val,    1   // value to write
.set v_rd,     2   // read value
.set v_exp,    3   // expected value
.set v_fail,   4   // per-lane fail count
.set v_sst_os, 5   // LDS store offset
.set v_tmp,    6
.set v_outaddr,8   // 2 vgprs for global_store
.set v_idx,    10  // wg_id*64 + tid, persistent
.set v_diagaddr,12 // 2 vgprs (even): per-thread diagnostic record address
.set v_tmp2,   14
.set v_end,    15

.text
.globl lds_barrier_visibility_test
.p2align 8
.type lds_barrier_visibility_test,@function
lds_barrier_visibility_test:
    s_load_dwordx4 s[s_p_out:s_p_out+3], s[s_ka:s_ka+1], 0
    v_mov_b32 v[v_tid], v0
    s_mov_b32 s[s_bx], ttmp9
    s_wait_kmcnt 0x0

    v_mov_b32 v[v_fail], 0
    v_lshlrev_b32 v[v_sst_os], 2, v[v_tid]   // tid * 4 bytes (one dword per thread)

    // Per-thread diagnostic record address: idx = wg_id*64+tid, addr = p_diag + idx*16
    // (4 dwords/thread: slot_i, iter, actual, expected -- overwritten on every mismatch,
    // so after the run each thread's record is its LAST mismatch, if any. The host
    // zero-inits the whole buffer, so an all-zero record means "never mismatched".)
    s_lshl_b32 s[s_tmp], s[s_bx], 6
    v_mov_b32 v[v_idx], s[s_tmp]
    v_add_u32 v[v_idx], v[v_idx], v[v_tid]
    v_lshlrev_b32 v[v_tmp2], 4, v[v_idx]
    v_mov_b32 v[v_diagaddr+1], s[s_p_diag+1]
    v_add_co_u32 v[v_diagaddr], vcc_lo, s[s_p_diag], v[v_tmp2]
    v_add_co_ci_u32 v[v_diagaddr+1], vcc_lo, 0, v[v_diagaddr+1], vcc_lo

    s_mov_b32 s[s_kitr], N_ITERS

L_loop:
    // Step 1: write this iteration's magic value to LDS[tid]
    // magic = (iter << 16) | (wave_id << 8) | (tid & 0xFF)
    v_lshrrev_b32 v[v_tmp], 5, v[v_tid]       // wave_id = tid >> 5
    v_lshlrev_b32 v[v_val], 16, s[s_kitr]     // iter << 16
    v_lshl_or_b32 v[v_val], v[v_tmp], 8, v[v_val]  // | (wave_id << 8)
    v_lshl_or_b32 v[v_val], v[v_tid], 0, v[v_val]  // | tid  (tid < 64, fits in 8 bits... actually tid can be 0-63, and we want the low 8 bits)
    // Actually v_tid can be up to 63 which fits in 6 bits, fine for 8-bit field.
    // But v_lshl_or_b32 with shift 0 is just OR. Let me use v_or_b32 instead.
    // v_lshl_or_b32 v[v_val], v[v_tid], 0, v[v_val]  -- this is v_val = (v_tid << 0) | v_val = v_tid | v_val
    // That works if v_tid's upper bits are zero (they are, since tid < 64 and we shifted wave_id out).
    // Wait, v_tid is the FULL tid (0-63), not just the lane. We want the low 8 bits.
    // v_tid is 0-63, which fits in 7 bits. (v_tid << 0) | v_val works.
    // But we already have iter<<16 | wave_id<<8, and tid has bits 0-5 set.
    // tid = wave_id*32 + lane. So tid's bits 0-4 = lane, bits 5 = wave_id.
    // (tid << 0) | val would set bits 0-5 of val, but wave_id is already in bits 8+.
    // This means tid's bit 5 (wave_id) would be OR'd into bit 5, which is fine
    // since val's bits 0-7 are currently 0 (only iter<<16 and wave_id<<8 are set).
    // Actually val = (iter<<16) | (wave_id<<8). wave_id = tid>>5. So:
    // val = (iter<<16) | ((tid>>5)<<8) | tid
    // For tid=0: val = iter<<16 | 0 | 0 = iter<<16
    // For tid=32: val = iter<<16 | (1<<8) | 32 = iter<<16 | 256 | 32
    // This encodes iter, wave, and tid uniquely. Good.

    ds_write_b32 v[v_sst_os], v[v_val]

    // Step 2: wait for THIS wave's ds_write
    s_wait_dscnt 0x0

    // Step 3: barrier (signal + wait)
    s_barrier_signal -1
    s_barrier_wait -1

    // Step 4: read ALL 64 LDS slots and check
    // Each slot i should contain: (iter<<16) | ((i>>5)<<8) | i
    .irp i, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63
        // Read slot \i
        v_mov_b32 v[v_tmp], \i * 4
        ds_read_b32 v[v_rd], v[v_tmp]
        s_wait_dscnt 0x0

        // Expected: (iter<<16) | ((\i >> 5) << 8) | \i
        v_lshlrev_b32 v[v_exp], 16, s[s_kitr]
        v_lshl_or_b32 v[v_exp], (\i >> 5), 8, v[v_exp]
        v_or_b32 v[v_exp], \i, v[v_exp]

        // Check
        v_cmp_eq_u32 vcc_lo, v[v_rd], v[v_exp]
        v_cndmask_b32 v[v_tmp], 1, 0, vcc_lo   // v_tmp = (match ? 1 : 0)
        v_cmp_eq_u32 vcc_lo, 0, v[v_tmp]        // v_tmp == 0 means mismatch
        v_cndmask_b32 v[v_tmp], 1, 0, vcc_lo   // v_tmp = (mismatch ? 1 : 0)
        v_add_u32 v[v_fail], v[v_fail], v[v_tmp]

        // Diagnostic capture: on mismatch, overwrite this thread's record with
        // {slot_i, iter, actual, expected} -- EXEC-narrowed to mismatching lanes
        // only, so this costs nothing extra for lanes that matched.
        v_cmpx_ne_u32 v[v_tmp], 0
        v_mov_b32 v[v_tmp2], \i
        global_store_dword v[v_diagaddr:v_diagaddr+1], v[v_tmp2], off offset:0
        v_mov_b32 v[v_tmp2], s[s_kitr]
        global_store_dword v[v_diagaddr:v_diagaddr+1], v[v_tmp2], off offset:4
        global_store_dword v[v_diagaddr:v_diagaddr+1], v[v_rd], off offset:8
        global_store_dword v[v_diagaddr:v_diagaddr+1], v[v_exp], off offset:12
        s_mov_b32 exec_lo, -1
    .endr

    // Decrement and loop
    s_sub_i32 s[s_kitr], s[s_kitr], 1
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc1 L_loop

    // Epilogue: atomic-add fail count to global output
    v_lshlrev_b32 v[v_outaddr], 2, v[v_tid]
    v_mov_b32 v[v_outaddr+1], s[s_p_out+1]
    v_add_co_u32 v[v_outaddr], vcc_lo, s[s_p_out], v[v_outaddr]
    v_add_co_ci_u32 v[v_outaddr+1], vcc_lo, 0, v[v_outaddr+1], vcc_lo
    // Use flat atomic add (or global_atomic)
    global_atomic_add v[v_outaddr:v_outaddr+1], v[v_fail], off
    s_wait_storecnt 0x0
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel lds_barrier_visibility_test
    .amdhsa_group_segment_fixed_size 256
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 15
    .amdhsa_next_free_sgpr 16
    .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: lds_barrier_visibility_test
    .symbol: lds_barrier_visibility_test.kd
    .sgpr_count: 16
    .vgpr_count: 15
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 256
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .reqd_workgroup_size : [64, 1, 1]
    .max_flat_workgroup_size: 64
    .args:
    - { .name: p_out , .size: 8, .offset: 0, .value_kind: global_buffer, .value_type: i32, .address_space: global, .is_const: false}
    - { .name: p_diag, .size: 8, .offset: 8, .value_kind: global_buffer, .value_type: i32, .address_space: global, .is_const: false}
...
.end_amdgpu_metadata
