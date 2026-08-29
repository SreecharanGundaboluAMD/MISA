################################################################################
#
#  MIT License
#
#  Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#
################################################################################

import os
from .global_memory import *
from .wmma_mapping import *

class ctrl_coalescing_store_wmma_t(object):
    '''
    Direct (non-LDS-coalesced) epilogue for gfx1250 WMMA. Deliberately much simpler
    than coalescing_store_dotx.py/coalescing_store.py's xdlops classes: no
    "coalescing_groups" LDS-reshuffle pass, no vector_store_m/n folding, and (unlike
    the xdlops epilogue) no accvgpr_unified field at all -- WMMA accumulates directly
    in v_c (plain VGPR), so there is nothing to move out of an AGPR file first.

    This trades away some store coalescing/vectorization tuning for a much smaller,
    more auditable epilogue, appropriate for a correctness-first milestone (see
    docs/gfx1250_wmma_layout.md). It is still reasonably store-friendly in practice:
    per docs/gfx1250_wmma_layout.md, 16 consecutive lanes share one output row and
    cover 16 consecutive columns for a fixed accumulator vgpr index, so a per-lane
    scalar global_store_dword is already contiguous across each 16-lane half-wave.
    '''
    def __init__(self):
        self.cxm = None              # ctrl_wmma_mapping_t
        self.block_size = 256
        self.precision = 'fp32'      # accumulator precision as stored to global memory
        # when set, this thread's macro tile is only a PARTIAL sum over a slice of the
        # reduction (K) axis -- other workgroups hold the other slices and atomically add
        # into the same output elements, so the store must accumulate (global_atomic_add_f32)
        # rather than overwrite (global_store_dword). See gemm_k_global_split in
        # docs/gfx1250_wmma_layout.md -- correct for every precision because the WMMA output
        # buffer is always allocated fp32 regardless of the tunable's nominal precision
        # (conv_driver.cpp's is_wmma dtype_alloc_byte override), so there is no fp16/bf16
        # atomic-add precision concern and no workspace/cast step needed. Atomic-add has no
        # wide/packed fp32 variant on this ISA (confirmed against the CDNA5 ISA doc), so this
        # path stays scalar regardless of vector_write_out -- see docs/gfx1250_wmma_layout.md.
        self.gemm_k_global_split = False
        # non-atomic path only: target width (in elements) for the vectorized global store
        # after the LDS reshuffle. 4 = global_store_dwordx4. See docs/gfx1250_wmma_layout.md's
        # LDS-reshuffle phase for the derivation.
        self.vector_write_out = 4

        # Phase 23 (ISA-driven epilogue tuning): atomic path only. Default 'SCOPE_SYS' =
        # today's exact behavior. 'SCOPE_DEV' resolves within the device's L2 instead of
        # forcing a full system-level flush/invalidate -- sufficient since gemm_k_global_split's
        # contending workgroups are always on the same device. See docs/gfx1250_wmma_layout.md.
        self.atomic_scope = 'SCOPE_SYS'
        # atomic path only. 0 (default) = regular atomic. 1 = TH[2] cascading/deferred-scope
        # atomic. The `th:TH_ATOMIC_CASCADE_RT` ENCODING was confirmed correct via an
        # `llvm-mc -show-encoding` round-trip probe (Phase 23) -- but using it CONFIRMED HANGS
        # ON REAL HARDWARE: this kernel's `s_wait_storecnt 0x0` before `s_endpgm` never
        # completes, because the doc states a cascading atomic's full scope/completion is only
        # realized at "a subsequent release... of a matching or higher scope", which this
        # kernel never issues. Hard-blocked in igemm_base.py's tunable read (`assert not
        # self.atomic_cascade`) until a companion release mechanism is designed and added --
        # do not wire this up without that. See docs/gfx1250_wmma_layout.md's Phase 23.
        self.atomic_cascade = 0
        # the confirmed `th:` identifier to emit when atomic_cascade=1 -- a string, not a raw
        # number (llvm-mc requires a symbolic th value, rejects numeric immediates).
        self.atomic_th = 'TH_ATOMIC_CASCADE_RT'
        # non-atomic path only. 0 (default) = today's unpadded tile-linear LDS layout. 1 =
        # pad the row stride by one element to break a bank-conflict periodicity (macro_tile_n
        # is always a multiple of 64, so the unpadded layout puts every row of a given column
        # in the same LDS bank -- see docs/gfx1250_wmma_layout.md).
        self.epilogue_lds_pad = 0

        # Phase 24 (F16-accumulate WMMA): non-atomic path only (there is no packed-fp16
        # atomic-add on this ISA -- the gemm_k_global_split path always stays f32-accumulate
        # regardless of this flag). 0 (default) = v_c holds one fp32 element/VGPR, today's
        # exact behavior. 1 = v_c holds TWO packed fp16 elements/VGPR (rows 2j and 2j+1 of
        # the same column in bits [15:0]/[31:16], per the CDNA5 ISA doc's 16-bit C/D-matrix
        # table) -- the scatter must split each VGPR into two separate LDS writes at their
        # real row addresses (one row apart), and every element-width-dependent byte
        # shift/instruction-selection in the gather halves accordingly (2 bytes/element
        # instead of 4). See docs/gfx1250_wmma_layout.md's Phase 24.
        self.wmma_acc_f16 = 0

        # Phase 25 (GEMM_M tail): originally non-atomic-path-only (fwd/bwd, where it's
        # mutually exclusive with gemm_k_global_split -- that atomic epilogue branch was
        # never adapted for fwd/bwd). Phase 35 adds masking for the PLAIN atomic branch too
        # (wrw, whose gemm_k_global_split is its primary path) -- see that branch's
        # v_tmp3-based per-element row guard below; NOT implemented for the atomic_pack_bf16
        # branch (mutually exclusive at the igemm_base.py tunable level). 0 (default) =
        # today's unconditional store, every existing config unaffected. 1 = EXEC-mask each
        # store/atomic off for lanes whose absolute row (block_m_off + this lane's
        # tile-local row) is >= the real (unpadded) GEMM_M -- the tail block's out-of-range
        # rows. Mirrors XDLOPS's coalescing_store.py v_cmp_gt_u32/saveexec guard, but using
        # this file's existing v_cmpx/exec_lo idiom (see igemm_fwd_gtc_wmma_nhwc.py's
        # _emit_gld_chunk_load) since WMMA is wave32-only -- no 64-bit saveexec needed, a plain
        # exec_lo restore suffices. See docs/gfx1250_wmma_layout.md's Phase 25/35.
        self.wmma_m_tail = 0

        # Phase 26b (GEMM_N tail): analogous to wmma_m_tail but for the column, same Phase 35
        # atomic-branch extension. Non-atomic path: a second EXEC-mask guard chained right
        # after the M-tail one (wave32 v_cmpx intersects with the already-narrowed EXEC -- no
        # extra VGPR needed there, since `v_gather` already holds this lane's global column
        # for every pass). Atomic path: recomputes the column fresh per i_rn (column doesn't
        # depend on row/j, only on the compile-time-known i_rn index) into v_tmp4 before each
        # guard. Independent of wmma_m_tail -- either, both, or neither may be set.
        self.wmma_n_tail = 0

        # Phase 34 (packed atomics): atomic path only, precision=='bf16' only (asserted in
        # igemm_base.py). 0 (default) = today's scalar global_atomic_add_f32, one per lane.
        # 1 = pack ADJACENT columns (lane L and lane L^1, which per wmma_mapping.py's
        # `lane % 16 -> column` are always adjacent columns within the same 16-lane half)
        # into one global_atomic_pk_add_bf16, halving the number of actual atomic RMW ops
        # hitting memory -- independently confirmed as a real technique by both FlyDSL's
        # gfx1250 MoE split-K GEMM and rocKE's wgrad conv epilogue, see
        # docs/gfx1250_perf_parity_action_plan.md's Tier 1 item 2. Needs a genuine
        # cross-lane exchange (ds_bpermute_b32) since each lane only ever holds its OWN
        # column's fp32 accumulator value -- verified correct (exchange pairing, packing
        # byte order, multi-block atomic accumulation) via a standalone hardware probe
        # before landing here. Trades scalar-fp32-atomic precision for packed-bf16-atomic
        # precision on the K-split reduction -- a real numerical tradeoff, not free; see
        # docs/gfx1250_wmma_layout.md's Phase 34 for the accuracy validation.
        self.atomic_pack_bf16 = 0

        # Phase 35 (hipconv-style reduction-kernel epilogue): atomic (gemm_k_global_split)
        # path only, mutually exclusive with atomic_pack_bf16 (asserted in igemm_base.py).
        # 0 (default) = today's global_atomic_add_f32. 1 = plain global_store_dword instead
        # -- correct ONLY when the caller has arranged for s_p_out to already point at this
        # shard's own disjoint slice of a workspace buffer (no concurrent writer ever
        # targets the same address, so no atomic/ordering is needed at all) -- see
        # docs/gfx1250_wmma_layout.md's Phase 35 and igemm_wrw_gtc_driver.h's WMMA run().
        self.wrw_reduction_kernel = 0

        # Phase 53 (chunked epilogue, non-atomic path only): 0 (default) = today's
        # one-shot design, staging the WHOLE macro-tile in LDS before any store --
        # byte-identical codegen, this is what hard-caps the macro-tile at 128x128 (see
        # docs/gfx1250_wmma_layout.md's Phase 52/53). 1 = reuse a small,
        # tile-size-INVARIANT LDS region across `wave_repeat_m` sequential groups
        # instead -- the same principle XDLOPS's coalescing_store.py already uses
        # (`coalescing_groups`), adapted to WMMA's simpler addressing. Mutually
        # exclusive with wmma_m_tail/wmma_n_tail/wmma_acc_f16/bf16acc (masking/packed-
        # layout interaction not audited this pass, see igemm_base.py) and with
        # gemm_k_global_split (this flag only touches the non-atomic branch).
        self.wmma_epilogue_chunked = 0

        # Phase 54 (VGPR-MSB): shared vgpr_msb_tracker_t instance (None = mechanism
        # off, today's byte-identical behavior). When set, v_c lives in a separate
        # 256-VGPR bank -- every v_c read-out below (ds_write DATA/global-store-or-
        # atomic VSRC -> src1 slot; the bf16 cross-lane permlane/cvt_pk path's first
        # source -> src0 slot) asks the tracker to emit `s_set_vgpr_msb` only when the
        # required bank combination actually changes. See igemm_base.py's
        # wmma_acc_high_bank docstring.
        self.vgpr_msb_tracker = None
        # Phase 59: skip LDS reshuffle, store per-element directly.
        # 16 consecutive lanes cover 16 consecutive columns (lane%16 -> column per
        # wmma_mapping.py), so the half-wave's scalar stores are already contiguous
        # at the memory controller level -- the LDS reshuffle is unnecessary overhead.
        self.direct_store = False


class igemm_coalescing_store_wmma_t(mc_base_t):
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        assert type(ctrl) is ctrl_coalescing_store_wmma_t
        self.ctrl = ctrl
        # Phase 51: per-pass fast/slow store branch labels (see __call__'s non-atomic path)
        # -- id(self) keeps these unique across every kernel in an assembled multi-kernel
        # file (each kernel builds its own igemm_coalescing_store_wmma_t instance, called
        # exactly once, so a per-instance counter alone would collide across kernels
        # sharing the same small counter values).
        self._label_counter = 0

    def _emit_chunked_non_atomic_store(self, ctrl, cxm, inst_wmma, v_c, v_gemm_im, v_gemm_in,
            v_tid, v_gather, s_p_out, s_gemm_m_stride, v_tmp1, v_tmp2, s_tmp1,
            s_block_m_off, s_block_n_off, vwo, macro_tile_n, elem_bytes, elem_byte_shift,
            ds_read_inst, gst_inst, v_gather_range, pad, padded_stride, log2_n, v_chunked_col):
        '''
        Phase 53: chunked epilogue. Reuses one small, tile-size-INVARIANT LDS region
        across `wave_repeat_m` sequential groups instead of staging the whole macro-tile
        at once (the pre-Phase-53 design above, which is exactly what hard-caps the
        macro-tile at 128x128 -- see docs/gfx1250_wmma_layout.md's Phase 52/53). Same
        principle as XDLOPS's coalescing_store.py `coalescing_groups`, adapted to WMMA's
        much simpler (no thread-cluster sub-tiling) addressing.

        Chunks by `i_rm` (wave_repeat_m) only, not `i_rn` -- one group's scatter already
        covers the FULL macro_tile_n width (the `i_rn`/`j` loops are unconditionally
        nested inside `i_rm`), so the per-group LDS footprint is
        `(wave_tile_m*waves_per_m) x macro_tile_n` elements, independent of
        `wave_repeat_m`/`wave_repeat_n` -- i.e. independent of how big the macro-tile
        grows. No `i_rn` chunking is needed to keep this bounded.

        Key subtlety: `wmma_mapping.py`'s `get_gemm_index_for_dst_matrix` encodes
        `v_gemm_im = (wave_m_idx << log2(wave_tile_m*wave_repeat_m)) | (lane/16)*8` --
        note the wave index is shifted by the FULL `wave_tile_m*wave_repeat_m`, not just
        `wave_tile_m`. So for a FIXED `i_rm`, the absolute rows touched across all waves
        are scattered across the full macro-tile (stride `wave_tile_m*wave_repeat_m`
        apart), not contiguous -- reusing one small LDS region across groups requires a
        **compact** (group-local) address, not the tile-local one the unchunked path
        above uses. The compact row drops the `wave_repeat_m` factor: extract
        `wave_idx = v_gemm_im >> log2(wave_tile_m*wave_repeat_m)` and re-pack it at
        `log2(wave_tile_m)` instead of its native (higher) bit position. This is the
        SAME transform in both directions: SCATTER computes compact-from-native (once,
        group-invariant, since v_gemm_im itself never changes); GATHER needs the inverse
        (native-from-compact, since the compact row it derives from `tid` must be
        expanded back to a real tile-local row before adding `s_block_m_off` for the
        global store address) -- this direction ALSO needs `+ i_rm*wave_tile_m` folded
        in, since that's exactly the offset compact addressing dropped.

        Bugfix (found via hardware validation, -V 1, after this method's first real
        build+run -- it had never been hardware-tested before): the gather's per-pass
        loop originally advanced the GLOBAL store address by a single fixed stride
        per pass, assuming "compact row advances linearly" implies "true (uncompacted)
        row advances linearly". False: the compact->true mapping
        (`wave_idx = compact_row >> log2(wave_tile_m)`, re-inserted at
        `log2(wave_tile_m*wave_repeat_m)`) has a discontinuity every `wave_tile_m`
        compact rows -- crossing it jumps the true row by
        `wave_tile_m*wave_repeat_m - wave_tile_m`, not by the assumed per-pass step.
        Confirmed on hardware: roughly half of every group's stored elements landed at
        the wrong row (`valid:n`). Fixed by recomputing the true row (hence the global
        address) fully from scratch every pass, using `v_tid` (never clobbered) instead
        of trying to preserve a running value through the pass loop's own register
        reuse.

        One persistent register beyond the unchunked path's v_tmp1/v_tmp2/v_gather is
        genuinely needed for this: `v_chunked_col` holds this thread's GLOBAL column
        (invariant across every pass and every group, computed once below before the
        group loop) -- everything else the recomputation needs (compact row, wave_idx,
        lane_sub) is rederived fresh each pass from `v_tid`, so no other new register is
        required; `v_tmp1`/`v_tmp2`/`v_gather` keep their existing per-group/per-pass
        scratch roles.
        '''
        wave_tile_m = cxm.wave_tile_m
        wave_repeat_m = cxm.wave_repeat_m
        waves_per_m = cxm.waves_per_m()
        virtual_tile_m = wave_tile_m * waves_per_m   # per-group compact M range -- constant across groups
        log2_wave_tile_m = utility_log2(wave_tile_m)
        log2_true_tile_m = utility_log2(wave_tile_m * wave_repeat_m)   # bit position to re-insert wave_idx at (native/uncompacted)
        total_elements_g = virtual_tile_m * macro_tile_n
        elements_per_thread_g = total_elements_g // ctrl.block_size
        assert elements_per_thread_g % vwo == 0, f"elements_per_thread_g:{elements_per_thread_g} not divisible by vector_write_out:{vwo}"
        num_passes_g = elements_per_thread_g // vwo
        elems_per_pass_g = ctrl.block_size * vwo
        assert elems_per_pass_g % macro_tile_n == 0, f"elems_per_pass_g:{elems_per_pass_g} not divisible by macro_tile_n:{macro_tile_n}"
        row_step_per_pass_g = elems_per_pass_g // macro_tile_n

        if ctrl.vgpr_msb_tracker is not None:
            # Phase 54: every VALU instruction in this whole epilogue writes to a
            # bank-0 scratch register (v_tmp1/v_tmp2/v_gather/v_gather_range) -- v_c is
            # ONLY ever read here, never written -- so dst=0 holds for the entire
            # method; only src0/src1 change, at the specific points below where v_c is
            # actually read. src2 must ALSO be reset here (not just dst): the gather's
            # per-pass address recomputation uses `v_lshl_or_b32 vdst, src0, shift_imm,
            # src2` (confirmed VOP3 via llvm-mc -- the OR operand is a real, MSB-
            # sensitive SRC2 slot, not folded into src0/src1) with `v_gather` as BOTH
            # dst and src2 -- left at its main-loop value (src2=1, for v_c's WMMA C
            # operand), this silently read v_gather from the wrong bank (a real bug
            # found via hardware validation: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION,
            # not merely wrong results, since the corrupted value feeds directly into
            # the global store address).
            msb_line = ctrl.vgpr_msb_tracker.ensure(dst=0, src2=0)
            if msb_line:
                self._emit(msb_line)
        self._emit(f"; wmma CHUNKED LDS-reshuffle coalescing store, tile {cxm.macro_tile_m}x{macro_tile_n}, "
                   f"{wave_repeat_m} groups of {virtual_tile_m}x{macro_tile_n}, vector_write_out={vwo}, {num_passes_g} passes/group")
        # this thread's GLOBAL column: invariant across every pass and every group (only
        # the row varies), so computed once, here, into a register that survives the
        # whole method -- v_tmp1/v_gather both get reused as scratch below and can't
        # hold it themselves. Uses v_tmp1 as scratch first (its real per-group job,
        # the LDS address, is (re)computed fresh inside the group loop below anyway).
        self._emit(f"v_lshlrev_b32 v[{v_tmp1}], {utility_log2(vwo)}, v[{v_tid}]   ; tid*{vwo}")
        self._emit(f"v_and_b32 v[{v_chunked_col}], {macro_tile_n - 1}, v[{v_tmp1}]   ; col (tile-local)")
        self._emit(f"v_add_u32 v[{v_chunked_col}], s[{s_block_n_off}], v[{v_chunked_col}]   ; + block_n_off -> global col (persistent for the whole method)")
        self._emit(f"s_wait_dscnt 0x0")
        self._emit(f"s_barrier_signal -1")
        self._emit(f"s_barrier_wait -1   ; main loop's own LDS traffic must retire before group 0 reuses this LDS region")
        self._emit_empty_line()

        for i_rm in range(wave_repeat_m):
            self._emit(f"; --- chunked epilogue group {i_rm}/{wave_repeat_m} ---")
            if i_rm != 0:
                self._emit(f"s_wait_dscnt 0x0")
                self._emit(f"s_barrier_signal -1")
                self._emit(f"s_barrier_wait -1   ; group {i_rm-1}'s gather-reads must retire before this group's scatter reuses the same LDS bytes")

            # ---- scatter: this group's own wave_repeat_n*num_v_c accumulator slice, at the
            # COMPACT row (v_gemm_im's wave_idx re-packed at log2(wave_tile_m), no i_rm offset --
            # dropped by compaction, recovered on the gather side instead) ----
            # Bugfix (Phase 55, 2026-08-28): `v_gemm_im` has `s_block_m_off` PERMANENTLY
            # folded in by the prologue (`v_add_u32 v_gemm_im, s_block_m_off, v_gemm_im`),
            # so it's a GLOBAL row for every workgroup after the first, not the tile-local
            # value this compact-row derivation assumes. `block_m_off` is always an exact
            # multiple of `macro_tile_m` (hence of `wave_tile_m*wave_repeat_m` too), so for
            # workgroup bx=0 (block_m_off=0) this was invisible, but for bx>=1 the
            # right-shift below picks up block_m_off's now-nonzero high bits, computing a
            # `wave_idx` that's 2 or 3 too high (out of the valid 0/1 range) instead of the
            # intended native wave index -- silently scattering into the wrong LDS bytes
            # entirely (found via hardware validation: a `CHUNK_FINGERPRINT` diagnostic
            # showed a SECOND workgroup's output reading pure leftover/uninitialized memory
            # instead of any of the expected fingerprint values, for every single element).
            # The sibling unchunked path already masks (`(v_gemm_im+row_off) &
            # (macro_tile_m-1)`) before using v_gemm_im for LDS addressing -- this was
            # simply missing here. Mask to tile-local range first, same discipline.
            self._emit(f"v_and_b32 v[{v_tmp1}], {cxm.macro_tile_m - 1}, v[{v_gemm_im}]   ; tile-local v_gemm_im (strip any block offset folded in)")
            self._emit(f"v_lshrrev_b32 v[{v_tmp1}], {log2_true_tile_m}, v[{v_tmp1}]   ; wave_idx (native position)")
            self._emit(f"v_and_b32 v[{v_tmp2}], {wave_tile_m - 1}, v[{v_gemm_im}]   ; lane_sub = (lane/16)*8")
            self._emit(f"v_lshl_or_b32 v[{v_tmp2}], v[{v_tmp1}], {log2_wave_tile_m}, v[{v_tmp2}]   ; compact_row = (wave_idx<<log2(wave_tile_m)) | lane_sub")
            self._emit(f"v_and_b32 v[{v_tmp1}], {macro_tile_n - 1}, v[{v_gemm_in}]   ; col (unchanged -- N is never chunked)")
            if pad:
                self._emit(f"v_mul_lo_u32 v[{v_tmp2}], {padded_stride}, v[{v_tmp2}]   ; compact_row * padded_stride")
            else:
                self._emit(f"v_lshlrev_b32 v[{v_tmp2}], {log2_n}, v[{v_tmp2}]   ; compact_row << log2(macro_tile_n)")
            self._emit(f"v_add_u32 v[{v_tmp2}], v[{v_tmp1}], v[{v_tmp2}]   ; + col -> compact tile-linear index")
            self._emit(f"v_lshlrev_b32 v[{v_tmp2}], {elem_byte_shift}, v[{v_tmp2}]   ; byte address (compact, this group)")
            # Phase 53: for wmma_acc_f16/bf16acc, VGPR j packs TWO logical rows (2j lo-half,
            # 2j+1 hi-half) -- mirrors the unchunked path's identical Phase 24 handling
            # exactly (wave_tile_m/waves_per_m/log2_wave_tile_m above are unaffected by
            # accumulate width, since a WMMA instruction always covers wave_tile_m=16 rows
            # regardless of how densely they're packed into registers).
            row_step = 2 if ctrl.wmma_acc_f16 else 1
            if ctrl.vgpr_msb_tracker is not None:
                # Phase 54: v_c is always the DATA operand of ds_write_b16/b16_d16_hi/
                # b32 below (VDS -> src1 slot, confirmed via llvm-mc) -- one setting
                # covers this whole group's scatter, reset before the gather's global
                # store below (which reads v_gather_range, a bank-0 register, off the
                # SAME src1 slot).
                msb_line = ctrl.vgpr_msb_tracker.ensure(src1=1)
                if msb_line:
                    self._emit(msb_line)
            for j in range(inst_wmma.num_v_c):
                if j != 0:
                    # Bugfix (Phase 54, 2026-08-28): same class of bug as the unchunked
                    # path's identical row-advance (see its comment) -- v_tmp2 lands in
                    # v_add_u32's VOP2 VSRC1 slot, still gated by src1=1 (left active for
                    # the previous j's ds_write DATA operand), so it's silently read from
                    # bank 1 (garbage) instead of bank 0. Bracket at src1=0.
                    if ctrl.vgpr_msb_tracker is not None:
                        msb_line = ctrl.vgpr_msb_tracker.ensure(src1=0)
                        if msb_line:
                            self._emit(msb_line)
                    self._emit(f"v_add_u32 v[{v_tmp2}], {padded_stride * elem_bytes * row_step}, v[{v_tmp2}]   ; advance to compact row j={j}")
                    if ctrl.vgpr_msb_tracker is not None:
                        msb_line = ctrl.vgpr_msb_tracker.ensure(src1=1)
                        if msb_line:
                            self._emit(msb_line)
                for i_rn in range(cxm.wave_repeat_n):
                    c_index = (i_rm * cxm.wave_repeat_n + i_rn) * inst_wmma.num_v_c + j
                    col_off = i_rn * cxm.wave_tile_n * elem_bytes
                    offset_str = f" offset:{col_off}" if col_off != 0 else ""
                    if os.environ.get('CHUNK_FINGERPRINT'):
                        # TEMPORARY DIAGNOSTIC: replace the real accumulator DATA with a
                        # decodable integer fingerprint encoding (i_rm, j, i_rn) -- SAME for
                        # every lane, unlike the real per-lane v_c value -- then let the rest
                        # of the pipeline (unmodified barrier/gather/global_store) carry it
                        # to global memory. Isolates whether the ADDRESS math is right,
                        # completely independent of accumulator data correctness: read back
                        # via DUMP_PRED's raw hex and decode fingerprint = i_rm*1000+j*10+i_rn
                        # at every output position, compare against the position's OWN
                        # expected (i_rm,j,i_rn) per the row/col formula.
                        fingerprint = i_rm * 1000 + j * 10 + i_rn
                        self._emit(f"v_mov_b32 v[{v_tmp1}], {fingerprint}   ; CHUNK_FINGERPRINT: encodes (i_rm={i_rm},j={j},i_rn={i_rn})")
                        data_reg = v_tmp1
                    else:
                        data_reg = f"{v_c}+{c_index}"
                    if ctrl.wmma_acc_f16:
                        hi_off = col_off + padded_stride * elem_bytes
                        self._emit(f"ds_write_b16 v[{v_tmp2}], v[{data_reg}]{offset_str}   ; row j*2 (lo 16 bits)")
                        self._emit(f"ds_write_b16_d16_hi v[{v_tmp2}], v[{data_reg}] offset:{hi_off}   ; row j*2+1 (hi 16 bits)")
                    else:
                        self._emit(f"ds_write_b32 v[{v_tmp2}], v[{data_reg}]{offset_str}")
            self._emit_empty_line()

            self._emit(f"s_wait_dscnt 0x0")
            self._emit(f"s_barrier_signal -1")
            self._emit(f"s_barrier_wait -1   ; group {i_rm}'s scatter writes must be visible before this group's gather reads")

            if os.environ.get('CHUNK_RAW_DUMP') and i_rm == 0:
                # TEMPORARY DIAGNOSTIC: dump group 0's raw LDS content (flat tid-based
                # addressing, bypassing the compact->native row recovery entirely) straight
                # to s_p_out, then stop -- isolates whether the SCATTER wrote correct data
                # into LDS (independent of the GATHER's row-recovery math).
                elems_per_thread_raw = total_elements_g // ctrl.block_size
                self._emit(f"; CHUNK_RAW_DUMP: raw LDS dump, group 0, bypassing gather recovery")
                self._emit(f"v_lshlrev_b32 v[{v_tmp1}], {utility_log2(elems_per_thread_raw)}, v[{v_tid}]   ; tid*{elems_per_thread_raw}, flat element index")
                self._emit(f"v_lshlrev_b32 v[{v_tmp2}], {elem_byte_shift}, v[{v_tmp1}]   ; LDS byte address")
                self._emit(f"v_lshlrev_b32 v[{v_tmp1}], {elem_byte_shift}, v[{v_tmp1}]   ; global byte address (same flat layout)")
                for e in range(elems_per_thread_raw):
                    off = e * elem_bytes
                    self._emit(f"ds_read_b32 v[{v_gather}], v[{v_tmp2}] offset:{off}")
                    self._emit(f"s_wait_dscnt 0x0")
                    self._emit(f"global_store_dword v[{v_tmp1}], v[{v_gather}], s[{s_p_out}:{s_p_out}+1] offset:{off}")
                self._emit(f"s_wait_storecnt 0x0")
                self._emit(f"s_endpgm")
                return

            if ctrl.vgpr_msb_tracker is not None:
                # Phase 54: the ENTIRE gather phase below (address computation AND the
                # final store loop) only ever reads/writes bank-0 scratch registers
                # (v_tid/v_tmp1/v_tmp2/v_gather/v_gather_range) -- reset src1 back to
                # bank 0 here, before the FIRST gather instruction, not just before the
                # store loop, since e.g. v_tid/v_tmp1 appear as src1 in the address
                # computation too.
                msb_line = ctrl.vgpr_msb_tracker.ensure(src1=0)
                if msb_line:
                    self._emit(msb_line)

            # ---- gather: same tid-based scheme as the unchunked path, but over the
            # SMALLER virtual (compact) tile. LDS address (compact-linear, invariant
            # across this group's passes, exactly like the unchunked path) is computed
            # once, here. The GLOBAL store address is NOT invariant in the same way --
            # it must be recomputed fresh every pass (see this method's docstring
            # bugfix note) -- that happens inside the per-pass loop below instead. ----
            self._emit(f"v_lshlrev_b32 v[{v_tmp1}], {utility_log2(vwo)}, v[{v_tid}]   ; tid*{vwo}, compact tile-linear index for pass 0")
            self._emit(f"v_and_b32 v[{v_gather}], {macro_tile_n - 1}, v[{v_tmp1}]   ; col0 (tile-local, scratch only)")
            self._emit(f"v_lshrrev_b32 v[{v_tmp2}], {log2_n}, v[{v_tmp1}]   ; compact_row0 (0..{virtual_tile_m - 1}, scratch only)")
            if pad:
                self._emit(f"v_mul_lo_u32 v[{v_tmp1}], {padded_stride}, v[{v_tmp2}]   ; compact_row0 * padded_stride")
                self._emit(f"v_add_u32 v[{v_tmp1}], v[{v_gather}], v[{v_tmp1}]   ; + col0 -> padded compact tile-linear index")
                self._emit(f"v_lshlrev_b32 v[{v_tmp1}], {elem_byte_shift}, v[{v_tmp1}]   ; padded LDS byte address (invariant across this group's passes)")
            else:
                self._emit(f"v_lshlrev_b32 v[{v_tmp1}], {elem_byte_shift}, v[{v_tmp1}]   ; LDS byte address (invariant across this group's passes)")
            self._emit_empty_line()

            for it in range(num_passes_g):
                # recompute this pass's TRUE global row fresh from v_tid every time --
                # the compact->true mapping is discontinuous every wave_tile_m compact
                # rows, so it cannot be advanced by a single fixed per-pass stride (see
                # this method's docstring bugfix note). v_gather is pure scratch here:
                # safe, because its underlying register (v_gather_range/v_c) isn't
                # touched by the ds_read below until AFTER this address is fully
                # computed and stored in v_tmp2.
                self._emit(f"v_lshlrev_b32 v[{v_gather}], {utility_log2(vwo)}, v[{v_tid}]   ; tid*{vwo}")
                self._emit(f"v_lshrrev_b32 v[{v_gather}], {log2_n}, v[{v_gather}]   ; compact_row0")
                if it != 0:
                    self._emit(f"v_add_u32 v[{v_gather}], {it * row_step_per_pass_g}, v[{v_gather}]   ; + pass advance -> compact_row (pass {it})")
                self._emit(f"v_lshrrev_b32 v[{v_tmp2}], {log2_wave_tile_m}, v[{v_gather}]   ; wave_idx = compact_row >> log2(wave_tile_m)")
                self._emit(f"v_and_b32 v[{v_gather}], {wave_tile_m - 1}, v[{v_gather}]   ; lane_sub = compact_row & (wave_tile_m-1)")
                self._emit(f"v_lshl_or_b32 v[{v_gather}], v[{v_tmp2}], {log2_true_tile_m}, v[{v_gather}]   ; (wave_idx<<log2(wave_tile_m*wave_repeat_m)) | lane_sub")
                if i_rm != 0:
                    self._emit(f"v_or_b32 v[{v_gather}], {i_rm * wave_tile_m}, v[{v_gather}]   ; | (i_rm<<log2(wave_tile_m)) -- true row (tile-local, full macro_tile_m range)")
                self._emit(f"v_add_u32 v[{v_gather}], s[{s_block_m_off}], v[{v_gather}]   ; + block_m_off -> global row")
                self._emit(f"v_mul_lo_u32 v[{v_gather}], s[{s_gemm_m_stride}], v[{v_gather}]")
                self._emit(f"v_add_u32 v[{v_tmp2}], v[{v_chunked_col}], v[{v_gather}]   ; + global col")
                self._emit(f"v_lshlrev_b32 v[{v_tmp2}], {elem_byte_shift}, v[{v_tmp2}]   ; global memory byte address, pass {it}")

                self._emit(f"{ds_read_inst} v[{v_gather_range}], v[{v_tmp1}] offset:{it * row_step_per_pass_g * padded_stride * elem_bytes}")
                self._emit(f"s_wait_dscnt 0x0")
                self._emit(f"{gst_inst} v[{v_tmp2}], v[{v_gather_range}], s[{s_p_out}:{s_p_out}+1]")
            self._emit_empty_line()

    def __call__(self, v_c, v_gemm_im, v_gemm_in, s_p_out, s_gemm_m_stride, v_tmp1, v_tmp2, s_tmp1, v_tid=None, v_gather=None, s_block_m_off=None, s_block_n_off=None, s_gemm_m=None, v_tmp3=None, s_gemm_n=None, v_tmp4=None, s_tmp2=None, v_chunked_col=None):
        '''
        v_gemm_im/v_gemm_in: this thread's base (row, col) within the macro tile, from
            igemm_wmma_mapping_t.get_gemm_index_for_dst_matrix (row/col of wave_repeat
            iteration (0,0), vgpr index 0). Byte-address computation and the global
            block offset are left to the caller (this emits only the intra-macro-tile
            part) -- v_tmp1/v_tmp2 need 1 scratch VGPR each, s_tmp1 needs 1 scratch SGPR.
        s_gemm_m_stride: row stride of the output tensor, in elements (typically
            gemm_n, i.e. the full N extent, not just macro_tile_n).
        v_tid/v_gather: only needed for the non-atomic (LDS-reshuffle) path -- v_tid is the
            kernel's persistent flat thread id, v_gather is a vector_write_out-wide scratch
            VGPR range (unused by the atomic path).
        s_gemm_m/v_tmp3: needed when ctrl.wmma_m_tail is set, for EITHER the non-atomic path
            (Phase 25) or the plain-atomic path (Phase 35, wrw) -- s_gemm_m is the real
            (unpadded) GEMM_M scalar, v_tmp3 is 1 scratch VGPR used to track each
            pass/element's absolute row for the EXEC-mask guard.
        s_gemm_n/v_tmp4: needed when ctrl.wmma_n_tail is set, same two-path support as
            wmma_m_tail above -- s_gemm_n is the real (unpadded) GEMM_N scalar, v_tmp4 is 1
            scratch VGPR holding this lane's column-in-range check (non-atomic path: a flag,
            since `v_gather` itself gets reused as the gather's LDS-read destination and its
            column value doesn't survive to pass 0's guard; atomic path: the raw recomputed
            column value, compared fresh per i_rn).
            Phase 34 (ctrl.atomic_pack_bf16, atomic path only -- mutually exclusive with
            wmma_m_tail/wmma_n_tail at the igemm_base.py tunable level, so reusing these
            same three params here is safe): v_tmp3 holds the cross-lane-exchanged partner
            value (per-iteration scratch, produced directly by V_PERMLANE_XOR_B32 -- no
            precomputed index register needed, see below), v_tmp4 holds the packed bf16x2
            result (per-iteration scratch). v_tid is required (used for the even/odd
            EXEC-mask guard only -- v_gather is unused by this branch).

        ctrl.gemm_k_global_split selects between two structurally different epilogues:

        Atomic path (gemm_k_global_split=True): by default, one global_atomic_add_f32 per
        accumulator element, no LDS traffic (there is no wide/packed fp32 atomic-add on
        this ISA, confirmed against the CDNA5 ISA doc, so the plain fp32 path can't be
        vectorized). v_tmp1/v_tmp2 ping-pong as the "current row"/"next row" address so
        the row-advance add has no data dependency on the in-flight atomics (different
        register), instead of serializing all row addresses through one register. Column
        offset is folded into the atomic's immediate offset. When ctrl.atomic_pack_bf16 is
        set instead: every lane exchanges its fp32 value with its column-adjacent partner
        (V_PERMLANE_XOR_B32, lane XOR 1 -- valid since `lane % 16 -> column` in
        wmma_mapping.py always makes adjacent lanes adjacent columns within a 16-lane
        half), packs (own, partner) into one bf16x2 with the lower column in the packed
        value's low 16 bits (matching row-major memory layout), then only EVEN lanes
        issue one global_atomic_pk_add_bf16 covering both columns -- halving the number
        of actual atomic ops hitting memory at the cost of bf16 (not fp32) intermediate
        precision on the K-split reduction. V_PERMLANE_XOR_B32 "ignores EXEC for reads
        (fetch-invalid: act as if EXEC is all ones)" per the CDNA5 ISA doc's cross-lane
        section, so -- unlike the ds_bpermute_b32 this replaced (Phase 34's original
        version; DS-class reads DO depend on the source lane's EXEC bit) -- the exchange
        itself needs no full-EXEC widening at all; EXEC is narrowed to even-lanes-only
        only for the pack+atomic step and restored before the next iteration's exchange,
        same as before. Also a normal VALU op (VOP3), not a DS-class instruction tracked
        by DSCNT, so its result is available to the very next instruction with no
        `s_wait_dscnt` -- removes both the per-iteration wait and the one-time partner-
        byte-index precompute (`v_gather`) the ds_bpermute_b32 version needed, since
        V_PERMLANE_XOR_B32 takes the XOR mask directly as an immediate operand. (ISA doc
        7.2.7: "V_PERMLANE* may not occur immediately after a V_CMPX" -- not applicable
        here, since `global_atomic_pk_add_bf16` and the EXEC-restore `s_mov_b32` both sit
        between this loop's `v_cmpx_eq_u32` and the next iteration's V_PERMLANE_XOR_B32.)

        Non-atomic path (gemm_k_global_split=False): LDS-reshuffle coalescing store, see
        docs/gfx1250_wmma_layout.md's LDS-reshuffle phase for the full derivation. A single
        lane only ever owns one column (per docs/gfx1250_wmma_layout.md: col = lane % 16,
        fixed across every accumulator index), so no per-lane vectorized store is possible
        directly -- stage the whole macro-tile through LDS in true tile-linear order
        (scatter, one ds_write_b32 per element, same addressing style as the atomic path's
        row/col derivation just retargeted to LDS with shifts instead of a stride multiply),
        barrier, then every lane reads back `vector_write_out` contiguous elements via a
        flat tid-indexed mapping (gather) and issues one global_store_dwordN -- 4x fewer
        global stores and address multiplies than the direct/atomic path, using cheap
        on-chip LDS traffic to get there.
        '''
        ctrl = self.ctrl
        cxm = ctrl.cxm
        inst_wmma = cxm.inst_wmma
        assert cxm.wave_tile_m == 16 and cxm.wave_tile_n == 16
        if ctrl.wmma_m_tail:
            # Phase 35: now implemented for BOTH the non-atomic and the plain-atomic
            # (gemm_k_global_split, not atomic_pack_bf16) epilogue branches -- wrw's atomic
            # path is scalar-per-element already, so masking it needs no extra grouping
            # logic. atomic_pack_bf16 is separately excluded at the tunable level
            # (igemm_base.py) since it needs the same v_tmp3/v_tmp4 slots for its own,
            # unrelated cross-lane-exchange scratch.
            assert s_gemm_m is not None and v_tmp3 is not None
        if ctrl.wmma_n_tail:
            assert s_gemm_n is not None and v_tmp4 is not None
        if ctrl.atomic_pack_bf16:
            assert ctrl.gemm_k_global_split, "atomic_pack_bf16 only applies to the atomic epilogue branch"
            # v_gather not required here: V_PERMLANE_XOR_B32 takes its XOR mask as an
            # immediate, unlike ds_bpermute_b32's precomputed partner-byte-index operand
            # (the original Phase 34 mechanism this replaced).
            assert v_tid is not None and v_tmp3 is not None and v_tmp4 is not None

        with self._deferred_context():
            if ctrl.gemm_k_global_split and ctrl.atomic_pack_bf16:
                # ---- Phase 34: packed-bf16 atomic-add epilogue ----
                self._emit(f"; wmma packed-bf16 atomic-add epilogue, {cxm.wave_repeat_m}x{cxm.wave_repeat_n} tiles, "
                           f"{inst_wmma.num_v_c} rows/tile")
                self._emit(f"s_lshl_b32 s[{s_tmp1}], s[{s_gemm_m_stride}], 1   ; row-to-row byte stride (bf16, 2 bytes/elem)")
                self._emit_empty_line()
                for i_rm in range(cxm.wave_repeat_m):
                    row_off = i_rm * cxm.wave_tile_m
                    cur, nxt = v_tmp1, v_tmp2
                    self._emit(f"v_add_u32 v[{cur}], {row_off}, v[{v_gemm_im}]" if row_off != 0 else f"v_mov_b32 v[{cur}], v[{v_gemm_im}]")
                    self._emit(f"v_mul_lo_u32 v[{cur}], s[{s_gemm_m_stride}], v[{cur}]")
                    self._emit(f"v_add_u32 v[{cur}], v[{v_gemm_in}], v[{cur}]")
                    self._emit(f"v_lshlrev_b32 v[{cur}], 1, v[{cur}]  ; byte address (bf16), row {row_off}, col 0")
                    for j in range(inst_wmma.num_v_c):
                        if j != inst_wmma.num_v_c - 1:
                            self._emit(f"v_add_u32 v[{nxt}], v[{cur}], s[{s_tmp1}]   ; precompute row {row_off + j + 1} address")
                        for i_rn in range(cxm.wave_repeat_n):
                            c_index = (i_rm * cxm.wave_repeat_n + i_rn) * inst_wmma.num_v_c + j
                            col_off = i_rn * cxm.wave_tile_n * 2
                            offset_str = f" offset:{col_off}" if col_off != 0 else ""
                            # V_PERMLANE_XOR_B32: partner = this lane XOR 1, mask/group-width
                            # are immediates (no precomputed index register needed, unlike
                            # ds_bpermute_b32's byte-index operand). Ignores EXEC for reads
                            # (fetch-invalid), so no full-EXEC widening needed either. Plain
                            # VALU op, not DSCNT-tracked -- result is immediately consumable.
                            self._emit(f"v_permlane_xor_b32 v[{v_tmp3}], v[{v_c}+{c_index}], 1, 32   ; partner lane = this lane XOR 1")
                            self._emit(f"v_cvt_pk_bf16_f32 v[{v_tmp4}], v[{v_c}+{c_index}], v[{v_tmp3}]   ; lo16=this lane's col, hi16=partner's -- only correct if this lane is even")
                            # only even lanes issue the packed atomic (their own column is
                            # the pair's lower/base column); narrow EXEC, issue, restore.
                            self._emit(f"v_and_b32 v[{v_tmp3}], 1, v[{v_tid}]")
                            self._emit(f"v_cmpx_eq_u32 0, v[{v_tmp3}]   ; EXEC = (this lane is even)")
                            th_str = f" th:{ctrl.atomic_th}" if ctrl.atomic_cascade else ""
                            self._emit(f"global_atomic_pk_add_bf16 v[{cur}], v[{v_tmp4}], s[{s_p_out}:{s_p_out}+1]{offset_str} scope:{ctrl.atomic_scope}{th_str}")
                            self._emit(f"s_mov_b32 exec_lo, -1   ; restore full EXEC for the next iteration's exchange")
                        cur, nxt = nxt, cur
                self._emit_empty_line()
            elif ctrl.gemm_k_global_split:
                # Phase 35: wrw_reduction_kernel replaces the atomic add with a plain store
                # into this shard's own disjoint workspace slice (the shard offset is baked
                # into s_p_out by the caller, before this function is even invoked -- see
                # igemm_wrw_gtc_wmma_nhwc.py's prologue) -- simpler than the atomic case (no
                # scope/ordering concerns at all), and the SAME per-element address/masking
                # logic below applies unchanged either way.
                self._emit(f"; wmma direct {'store (reduction-kernel epilogue)' if ctrl.wrw_reduction_kernel else 'atomic-add'} epilogue, "
                           f"{cxm.wave_repeat_m}x{cxm.wave_repeat_n} tiles, {inst_wmma.num_v_c} rows/tile")
                self._emit(f"s_lshl_b32 s[{s_tmp1}], s[{s_gemm_m_stride}], 2   ; row-to-row byte stride")
                for i_rm in range(cxm.wave_repeat_m):
                    row_off = i_rm * cxm.wave_tile_m
                    cur, nxt = v_tmp1, v_tmp2
                    # cur = byte address of (row_off, col 0), i.e. row = v_gemm_im + row_off
                    self._emit(f"v_add_u32 v[{cur}], {row_off}, v[{v_gemm_im}]" if row_off != 0 else f"v_mov_b32 v[{cur}], v[{v_gemm_im}]")
                    if ctrl.wmma_m_tail:
                        # Phase 35: v_tmp3 tracks this element's absolute row value in
                        # PARALLEL with cur/nxt's byte-address ping-pong (cur/nxt get
                        # overwritten with a multiplied+shifted byte address below, so the
                        # raw row value must be captured separately, here, before that
                        # happens) -- advanced by +1 alongside cur/nxt's per-j row advance.
                        self._emit(f"v_mov_b32 v[{v_tmp3}], v[{cur}]   ; wmma_m_tail: absolute row (row {row_off}, j=0)")
                    self._emit(f"v_mul_lo_u32 v[{cur}], s[{s_gemm_m_stride}], v[{cur}]")
                    self._emit(f"v_add_u32 v[{cur}], v[{v_gemm_in}], v[{cur}]")
                    self._emit(f"v_lshlrev_b32 v[{cur}], 2, v[{cur}]  ; byte address, row {row_off}, col 0")
                    for j in range(inst_wmma.num_v_c):
                        if j != inst_wmma.num_v_c - 1:
                            self._emit(f"v_add_u32 v[{nxt}], v[{cur}], s[{s_tmp1}]   ; precompute row {row_off + j + 1} address")
                        for i_rn in range(cxm.wave_repeat_n):
                            c_index = (i_rm * cxm.wave_repeat_n + i_rn) * inst_wmma.num_v_c + j
                            col_off = i_rn * cxm.wave_tile_n * 4
                            offset_str = f" offset:{col_off}" if col_off != 0 else ""
                            masked = ctrl.wmma_m_tail or ctrl.wmma_n_tail
                            if ctrl.wmma_m_tail:
                                self._emit(f"v_cmpx_gt_u32 s[{s_gemm_m}], v[{v_tmp3}]   ; wmma_m_tail: row < real gemm_m")
                            if ctrl.wmma_n_tail:
                                col_val = i_rn * cxm.wave_tile_n
                                self._emit(f"v_add_u32 v[{v_tmp4}], {col_val}, v[{v_gemm_in}]" if col_val != 0 else f"v_mov_b32 v[{v_tmp4}], v[{v_gemm_in}]")
                                self._emit(f"v_cmpx_gt_u32 s[{s_gemm_n}], v[{v_tmp4}]   ; wmma_n_tail: col < real gemm_n")
                            # scope:SCOPE_SYS (or SCOPE_DEV, Phase 23) is load-bearing, not
                            # decoration: a bare global_atomic_add_f32 defaults to a narrower
                            # (CU/WGP-local) cache scope on gfx1250, which silently drops
                            # updates when the two accumulating workgroups land on different
                            # compute units -- confirmed on hardware. See
                            # docs/gfx1250_wmma_layout.md. th:{th} (Phase 23, optional) marks
                            # a cascading/deferred-scope atomic -- see ctrl.atomic_cascade.
                            # Phase 35: wrw_reduction_kernel needs neither -- it's a plain,
                            # non-atomic store (no concurrent writers ever target the same
                            # workspace address, so no ordering/scope concern exists).
                            if ctrl.wrw_reduction_kernel:
                                self._emit(f"global_store_dword v[{cur}], v[{v_c}+{c_index}], s[{s_p_out}:{s_p_out}+1]{offset_str}")
                            else:
                                th_str = f" th:{ctrl.atomic_th}" if ctrl.atomic_cascade else ""
                                # Phase 57: int8/int4's WMMA accumulator is a genuine int32
                                # value, not a float -- global_atomic_add_f32 would silently
                                # bit-reinterpret it as float and only coincidentally add
                                # correctly for small non-negative subnormal-range sums (see
                                # the assert this replaces in igemm_base.py for the full
                                # derivation). global_atomic_add_u32 is a plain 32-bit integer
                                # add, correct for BOTH signed and unsigned int32 bit patterns
                                # (two's-complement addition doesn't care how the result is
                                # later interpreted) -- no separate signed variant exists or is
                                # needed (confirmed: no GLOBAL_ATOMIC_ADD_I32 in the ISA doc's
                                # opcode table, only U32/F32/F64/U64).
                                atomic_add_inst = 'global_atomic_add_u32' if ctrl.precision in ('int8', 'int4') else 'global_atomic_add_f32'
                                self._emit(f"{atomic_add_inst} v[{cur}], v[{v_c}+{c_index}], s[{s_p_out}:{s_p_out}+1]{offset_str} scope:{ctrl.atomic_scope}{th_str}")
                            if masked:
                                self._emit(f"s_mov_b32 exec_lo, -1")
                        cur, nxt = nxt, cur
                        if ctrl.wmma_m_tail and j != inst_wmma.num_v_c - 1:
                            self._emit(f"v_add_u32 v[{v_tmp3}], 1, v[{v_tmp3}]   ; wmma_m_tail: advance to row {row_off + j + 1}")
                self._emit_empty_line()
            elif ctrl.direct_store:
                assert not ctrl.wrw_reduction_kernel, "direct_store + wrw_reduction_kernel not implemented"
                assert not ctrl.wmma_epilogue_chunked, "direct_store + wmma_epilogue_chunked not implemented"
                self._emit_direct_store(ctrl, cxm, inst_wmma, v_c, v_gemm_im, v_gemm_in,
                    s_p_out, s_gemm_m_stride, v_tmp1, v_tmp2, s_tmp1,
                    s_gemm_m, v_tmp3, s_gemm_n, v_tmp4)
            else:
                # ---- non-atomic: LDS-reshuffle coalescing store ----
                assert v_tid is not None and v_gather is not None and s_block_m_off is not None and s_block_n_off is not None
                vwo = ctrl.vector_write_out
                macro_tile_m = cxm.macro_tile_m
                macro_tile_n = cxm.macro_tile_n
                assert macro_tile_n % vwo == 0, f"macro_tile_n:{macro_tile_n} not divisible by vector_write_out:{vwo}"
                if ctrl.wmma_n_tail and vwo > 1 and not ctrl.wmma_acc_f16:
                    # Phase 51: dedicated scratch for the per-pass fast/slow store branch --
                    # see the pass loop below.
                    assert s_tmp2 is not None, "wmma_n_tail with vector_write_out>1 (the non-narrow-accumulate case) requires s_tmp2, see Phase 51"
                if ctrl.wmma_epilogue_chunked:
                    # Phase 53: wmma_acc_f16/bf16acc IS supported (needed to make a bigger
                    # tile's accumulator fit gfx1250's real 256-VGPR/wave ceiling at all --
                    # see docs/gfx1250_wmma_layout.md's Phase 53 correction) -- only
                    # wmma_m_tail/wmma_n_tail remain excluded (masking interaction with
                    # per-group chunking not audited this pass).
                    assert not ctrl.wmma_m_tail and not ctrl.wmma_n_tail, \
                        "wmma_epilogue_chunked is not yet combined with wmma_m_tail/wmma_n_tail, see docs/gfx1250_wmma_layout.md's Phase 53"
                else:
                    total_elements = macro_tile_m * macro_tile_n
                    elements_per_thread = total_elements // ctrl.block_size
                    assert elements_per_thread % vwo == 0, f"elements_per_thread:{elements_per_thread} not divisible by vector_write_out:{vwo}"
                    num_passes = elements_per_thread // vwo
                    elems_per_pass = ctrl.block_size * vwo
                    assert elems_per_pass % macro_tile_n == 0, f"elems_per_pass:{elems_per_pass} not divisible by macro_tile_n:{macro_tile_n}"
                    row_step_per_pass = elems_per_pass // macro_tile_n
                log2_n = utility_log2(macro_tile_n)
                # Phase 23: pad the LDS row stride by one dwordx4 (4 elements) to break a
                # bank-conflict periodicity -- macro_tile_n is always a multiple of 64 (LDS
                # bank count), so the unpadded tile-linear address puts every row of a given
                # column in the same bank (a guaranteed 2-way conflict on every scatter
                # ds_write_b32, since WMMA's per-lane row differs by exactly 8 between the
                # two 16-lane halves, and (row+8)*macro_tile_n === row*macro_tile_n (mod 64)
                # -- both halves collide into the same 16 banks). Padding by 4 elements
                # (not 1) is deliberate: it still fully separates the two halves into
                # disjoint 16-bank ranges (offset by 4*8=32, per the same math), while
                # keeping every row's start byte-address a multiple of 16 -- required for
                # ds_read_b128/global_store_dwordx4 (vwo=4) to stay naturally aligned. A
                # 1-element pad would break bank periodicity too but misalign every row
                # whose index isn't a multiple of 4, corrupting vwo=4 reads. See
                # docs/gfx1250_wmma_layout.md's Phase 23 for the full derivation.
                pad = 4 if ctrl.epilogue_lds_pad else 0
                padded_stride = macro_tile_n + pad
                # Phase 24: the scatter unpacks f16acc's packed accumulator into genuinely
                # 2-byte-per-element LDS storage (see the scatter loop below) -- so from the
                # gather's point of view, LDS (and the final global memory tensor) just holds
                # narrower elements, same tile-linear layout otherwise. vwo counts LOGICAL
                # elements regardless of width; only the instruction/byte-shift selection
                # needs to know the width.
                elem_bytes = 2 if ctrl.wmma_acc_f16 else 4
                elem_byte_shift = 1 if ctrl.wmma_acc_f16 else 2
                if ctrl.wmma_acc_f16:
                    ds_read_inst = {1: "ds_read_u16", 2: "ds_read_b32", 4: "ds_read_b64"}[vwo]
                    gst_inst = {1: "global_store_short", 2: "global_store_dword", 4: "global_store_dwordx2"}[vwo]
                else:
                    ds_read_inst = {1: "ds_read_b32", 2: "ds_read_b64", 4: "ds_read_b128"}[vwo]
                    gst_inst = {1: "global_store_dword", 2: "global_store_dwordx2", 4: "global_store_dwordx4"}[vwo]
                # register range width = dwords needed to hold vwo elements at elem_bytes
                # each (vwo*4 for f32 -- always vwo dwords/VGPRs; vwo*2 for f16acc -- half as
                # many VGPRs, floored at 1 since ds_read_u16/global_store_short both address
                # one full VGPR regardless of the 16-bit payload being smaller than a dword).
                v_gather_num_regs = max(1, (vwo * elem_bytes) // 4)
                v_gather_range = f"{v_gather}:{v_gather}+{v_gather_num_regs - 1}" if v_gather_num_regs > 1 else v_gather

                if ctrl.wmma_epilogue_chunked:
                    assert v_chunked_col is not None, "wmma_epilogue_chunked requires v_chunked_col (see kernel_vgpr_t)"
                    self._emit_chunked_non_atomic_store(ctrl, cxm, inst_wmma, v_c, v_gemm_im, v_gemm_in,
                        v_tid, v_gather, s_p_out, s_gemm_m_stride, v_tmp1, v_tmp2, s_tmp1,
                        s_block_m_off, s_block_n_off, vwo, macro_tile_n, elem_bytes, elem_byte_shift,
                        ds_read_inst, gst_inst, v_gather_range, pad, padded_stride, log2_n, v_chunked_col)
                else:

                    if ctrl.vgpr_msb_tracker is not None:
                        # Phase 54: same reasoning as the chunked path's identical reset --
                        # every scratch VALU instruction below writes to a bank-0 register,
                        # v_c is only ever READ (never written) here, so dst=0/src2=0 holds
                        # for this whole branch; only src1 toggles, once for the whole
                        # scatter block and once for the whole gather block (unlike the
                        # chunked path, this design does all scatter THEN all gather, not
                        # per-group, so no per-i_rm toggling is needed).
                        msb_line = ctrl.vgpr_msb_tracker.ensure(dst=0, src2=0)
                        if msb_line:
                            self._emit(msb_line)
                    self._emit(f"; wmma LDS-reshuffle coalescing store, tile {macro_tile_m}x{macro_tile_n}, "
                               f"vector_write_out={vwo}, {num_passes} passes")
                    # barrier: the main loop's own LDS traffic must be fully retired before we reuse
                    # this same physical LDS region for the reshuffle -- otherwise a fast wave could
                    # start overwriting it while a slow wave in this workgroup is still reading it.
                    self._emit(f"s_wait_dscnt 0x0")
                    self._emit(f"s_barrier_signal -1")
                    self._emit(f"s_barrier_wait -1")

                    # ---- scatter: same (i_rm,i_rn,j) addressing as the atomic path above, retargeted
                    # to a tile-linear LDS address via shifts (macro_tile_n is a compile-time power of
                    # 2) instead of a runtime-stride multiply -- cheaper than the global-memory case.
                    # v_gemm_im/v_gemm_in are GLOBAL (gemm-space) positions -- the caller adds
                    # s_block_m_off/s_block_n_off once, persistently, right after computing them (see
                    # e.g. igemm_fwd_gtc_wmma_nhwc.py's emit_kernel_prologue) -- so they must be masked
                    # back down to tile-local (0..macro_tile_m/n-1) before use as an LDS address: since
                    # block_m_off/n_off are always exact multiples of macro_tile_m/n, `& (macro_tile-1)`
                    # strips exactly that high part and leaves the tile-local component unchanged. ----
                    self._emit(f"v_and_b32 v[{v_tmp2}], {macro_tile_n - 1}, v[{v_gemm_in}]   ; tile-local col (persistent)")
                    for i_rm in range(cxm.wave_repeat_m):
                        row_off = i_rm * cxm.wave_tile_m
                        if ctrl.vgpr_msb_tracker is not None and i_rm != 0:
                            # Phase 54: this iteration's address computation (below)
                            # reads v_gemm_im as src1 -- reset back to bank 0 (the
                            # previous i_rm's ds_write block above left it at src1=1).
                            # i_rm==0 needs no reset: src1 is already 0 here, carried
                            # over from the main loop's own state.
                            msb_line = ctrl.vgpr_msb_tracker.ensure(src1=0)
                            if msb_line:
                                self._emit(msb_line)
                        self._emit(f"v_add_u32 v[{v_tmp1}], {row_off}, v[{v_gemm_im}]" if row_off != 0 else f"v_mov_b32 v[{v_tmp1}], v[{v_gemm_im}]")
                        self._emit(f"v_and_b32 v[{v_tmp1}], {macro_tile_m - 1}, v[{v_tmp1}]   ; tile-local row")
                        if pad:
                            self._emit(f"v_mul_lo_u32 v[{v_tmp1}], {padded_stride}, v[{v_tmp1}]   ; row * padded_stride")
                        else:
                            self._emit(f"v_lshlrev_b32 v[{v_tmp1}], {log2_n}, v[{v_tmp1}]   ; row << log2(macro_tile_n)")
                        self._emit(f"v_add_u32 v[{v_tmp1}], v[{v_tmp2}], v[{v_tmp1}]   ; + col -> tile-linear index, row {row_off}")
                        self._emit(f"v_lshlrev_b32 v[{v_tmp1}], {elem_byte_shift}, v[{v_tmp1}]   ; byte address")
                        if ctrl.vgpr_msb_tracker is not None:
                            # Phase 54: the address computation just above reads
                            # v_gemm_im/v_gemm_in (persistent bank-0 registers) as
                            # src1 -- must run at src1=0 (established once, before this
                            # loop, and restored at the end of every prior iteration's
                            # ds_write block below). v_c is the DATA operand of
                            # ds_write_b16/b16_d16_hi/b32 below (VDS -> src1 slot) --
                            # switch to src1=1 only now, after that address is done.
                            msb_line = ctrl.vgpr_msb_tracker.ensure(src1=1)
                            if msb_line:
                                self._emit(msb_line)
                        # Phase 24: for f16acc, VGPR j packs TWO logical rows (2j lo-half, 2j+1
                        # hi-half, per the ISA doc's 16-bit C/D-matrix table) -- v_tmp1 tracks the
                        # LO row's address, stepping 2 rows (not 1) per j; the HI row's write
                        # reuses the SAME v_tmp1 base with one extra row's stride folded into its
                        # offset immediate (no extra address-compute instruction needed).
                        row_step = 2 if ctrl.wmma_acc_f16 else 1
                        for j in range(inst_wmma.num_v_c):
                            if j != 0:
                                # Bugfix (Phase 54, 2026-08-28): this add's second operand
                                # (v_tmp1, the running address) lands in v_add_u32's VOP2
                                # VSRC1 slot -- which is STILL gated by src1's MSB, left at
                                # 1 (bank 1) by the PREVIOUS j's ds_write below. Reading
                                # v_tmp1 at src1=1 silently reads it from bank 1 (garbage)
                                # instead of its real bank-0 value, corrupting the running
                                # address for every j>=1 (and everything chained after it)
                                # -- while j=0 (no advance needed yet) stays correct. This
                                # exactly explains the "row 0 always right, every other row
                                # wrong" signature root-caused this session (see
                                # docs/gfx1250_wmma_vgpr_msb_wip_status.md). Bracket the
                                # advance at src1=0, same discipline as the outer i_rm
                                # loop's own address computation just above.
                                if ctrl.vgpr_msb_tracker is not None:
                                    msb_line = ctrl.vgpr_msb_tracker.ensure(src1=0)
                                    if msb_line:
                                        self._emit(msb_line)
                                self._emit(f"v_add_u32 v[{v_tmp1}], {padded_stride * elem_bytes * row_step}, v[{v_tmp1}]   ; advance to row {row_off + j * row_step}")
                                if ctrl.vgpr_msb_tracker is not None:
                                    msb_line = ctrl.vgpr_msb_tracker.ensure(src1=1)
                                    if msb_line:
                                        self._emit(msb_line)
                            for i_rn in range(cxm.wave_repeat_n):
                                c_index = (i_rm * cxm.wave_repeat_n + i_rn) * inst_wmma.num_v_c + j
                                col_off = i_rn * cxm.wave_tile_n * elem_bytes
                                offset_str = f" offset:{col_off}" if col_off != 0 else ""
                                if ctrl.wmma_acc_f16:
                                    hi_off = col_off + padded_stride * elem_bytes
                                    self._emit(f"ds_write_b16 v[{v_tmp1}], v[{v_c}+{c_index}]{offset_str}   ; row {row_off + j*2} (lo 16 bits)")
                                    self._emit(f"ds_write_b16_d16_hi v[{v_tmp1}], v[{v_c}+{c_index}] offset:{hi_off}   ; row {row_off + j*2 + 1} (hi 16 bits)")
                                else:
                                    self._emit(f"ds_write_b32 v[{v_tmp1}], v[{v_c}+{c_index}]{offset_str}")
                    self._emit_empty_line()

                    # ---- barrier: all scatter writes must be visible to every lane before any gather read ----
                    self._emit(f"s_wait_dscnt 0x0")
                    self._emit(f"s_barrier_signal -1")
                    self._emit(f"s_barrier_wait -1")

                    if ctrl.vgpr_msb_tracker is not None:
                        # Phase 54: the whole gather section below (address computation AND
                        # the per-pass store loop) only ever touches bank-0 registers
                        # (v_tid/v_tmp1/v_tmp2/v_gather/v_gather_range as pure scratch) --
                        # reset src1 back to bank 0 before the first gather instruction.
                        msb_line = ctrl.vgpr_msb_tracker.ensure(src1=0)
                        if msb_line:
                            self._emit(msb_line)
                    # ---- gather: lane `tid` owns `vwo` contiguous elements per pass, starting at
                    # tile-linear index tid*vwo (+ a compile-time pass offset folded into the ds_read's
                    # immediate). row0/col0 (this lane's position for pass 0) are derived from that same
                    # index via shift/mask (macro_tile_n is a power of 2) -- col never changes across
                    # passes since elems_per_pass is a multiple of macro_tile_n by construction, so only
                    # the row (and therefore the global memory address) advances, by one scalar add of a
                    # precomputed per-pass stride. ----
                    self._emit(f"v_lshlrev_b32 v[{v_tmp1}], {utility_log2(vwo)}, v[{v_tid}]   ; tid*{vwo}, tile-linear index for pass 0")
                    self._emit(f"v_and_b32 v[{v_gather}], {macro_tile_n - 1}, v[{v_tmp1}]   ; col0 (tile-local)")
                    self._emit(f"v_lshrrev_b32 v[{v_tmp2}], {log2_n}, v[{v_tmp1}]   ; row0 (tile-local)")
                    # Phase 23: compute the LDS byte address (padded or not) from row0/col0 while
                    # they're still tile-local -- must happen BEFORE the block-offset adds below
                    # overwrite v_gather/v_tmp2 with global col/row. v_tmp1's original tid*vwo
                    # value is dead after this (unpadded case reuses it directly; padded case
                    # recomputes from row0*padded_stride+col0, since padding breaks the direct
                    # shift relationship between the packed index and the LDS offset).
                    if pad:
                        self._emit(f"v_mul_lo_u32 v[{v_tmp1}], {padded_stride}, v[{v_tmp2}]   ; row0 * padded_stride")
                        self._emit(f"v_add_u32 v[{v_tmp1}], v[{v_gather}], v[{v_tmp1}]   ; + col0 -> padded tile-linear index")
                        self._emit(f"v_lshlrev_b32 v[{v_tmp1}], {elem_byte_shift}, v[{v_tmp1}]   ; padded LDS byte address (invariant across passes)")
                    else:
                        self._emit(f"v_lshlrev_b32 v[{v_tmp1}], {elem_byte_shift}, v[{v_tmp1}]   ; LDS byte address (invariant across passes)")
                    self._emit(f"v_add_u32 v[{v_gather}], s[{s_block_n_off}], v[{v_gather}]   ; + block_n_off -> global col")
                    self._emit(f"v_add_u32 v[{v_tmp2}], s[{s_block_m_off}], v[{v_tmp2}]   ; + block_m_off -> global row")
                    if ctrl.wmma_n_tail:
                        # Phase 26b/51: this lane's column-in-range state, captured NOW -- v_gather
                        # itself is about to be reused as the gather's LDS-read destination
                        # register (v_gather_range aliases v_gather, see below), so its "global
                        # column" value would otherwise be gone before pass 0's guard needs it.
                        # Pass-invariant (column never changes across passes), so this is computed
                        # once here, not per-pass. Phase 51: holds "remaining = gemm_n - col0"
                        # (SIGNED), not a plain 0/1 flag -- "remaining > i" for i=0..vwo-1 is what
                        # the per-element masking below needs (a per-GROUP flag, "remaining > 0",
                        # is just the i=0 case of the same check, so this subsumes the old Phase
                        # 26b behavior exactly for vwo==1 / non-straddling groups). Signed
                        # subtraction/compare also correctly masks off every element when
                        # gemm_n < vwo (remaining goes negative).
                        self._emit(f"v_sub_i32 v[{v_tmp4}], s[{s_gemm_n}], v[{v_gather}]   ; wmma_n_tail: remaining = gemm_n - col0")
                        if vwo > 1 and not ctrl.wmma_acc_f16:
                            # Phase 51: gemm_n % vwo, computed ONCE (pass-invariant) into its own
                            # dedicated scratch SGPR -- lets each pass cheaply branch straight to
                            # the pre-Phase-51 fast (single vectorized store) path whenever the
                            # real gemm_n happens to be an exact multiple of vwo, instead of always
                            # paying the slower per-element decomposition (see the pass loop
                            # below). vwo is a compile-time power of 2, so a plain AND suffices.
                            self._emit(f"s_and_b32 s[{s_tmp2}], s[{s_gemm_n}], {vwo - 1}   ; wmma_n_tail: gemm_n % {vwo}")
                    if ctrl.wmma_m_tail:
                        # Phase 25: this lane's absolute row for pass 0, preserved across passes --
                        # v_tmp2 itself gets folded into a byte address on the next line, so it can't
                        # double as the running row counter the way it does in the unmasked case.
                        self._emit(f"v_mov_b32 v[{v_tmp3}], v[{v_tmp2}]   ; wmma_m_tail: absolute row, pass 0")
                    self._emit(f"v_mul_lo_u32 v[{v_tmp2}], s[{s_gemm_m_stride}], v[{v_tmp2}]")
                    self._emit(f"v_add_u32 v[{v_tmp2}], v[{v_gather}], v[{v_tmp2}]")
                    self._emit(f"v_lshlrev_b32 v[{v_tmp2}], {elem_byte_shift}, v[{v_tmp2}]   ; global memory byte address for pass 0")
                    self._emit(f"s_lshl_b32 s[{s_tmp1}], s[{s_gemm_m_stride}], {utility_log2(row_step_per_pass) + elem_byte_shift}   ; per-pass memory stride")
                    self._emit_empty_line()
                    for it in range(num_passes):
                        # row_step_per_pass*padded_stride*elem_bytes == elems_per_pass*elem_bytes
                        # when pad=0 (since elems_per_pass is a multiple of macro_tile_n by
                        # construction) -- one formula, byte-identical to the old literal in the
                        # unpadded f32 case (elem_bytes=4).
                        self._emit(f"{ds_read_inst} v[{v_gather_range}], v[{v_tmp1}] offset:{it * row_step_per_pass * padded_stride * elem_bytes}")
                        self._emit(f"s_wait_dscnt 0x0")
                        if ctrl.wmma_n_tail and vwo > 1 and not ctrl.wmma_acc_f16:
                            # Phase 51: gemm_n is no longer guaranteed to be a multiple of vwo here
                            # (that restriction -- previously required by tunable_is_valid()
                            # whenever wmma_n_tail was set -- is exactly what this mechanism
                            # lifts), so a single EXEC mask covering the whole vwo-wide vectorized
                            # store can't correctly handle gemm_n falling STRICTLY INSIDE one
                            # lane's group (the old mask only checked the group's first column,
                            # silently writing up to vwo-1 out-of-range trailing columns too).
                            #
                            # A runtime scalar branch picks between the pre-Phase-51 fast path
                            # (single vectorized store, exact-multiple case -- byte-identical to
                            # before this phase) and a slow path decomposed into vwo individual
                            # masked scalar stores (only the specific straddling group's write
                            # differs from the fast path; every other lane's data is unaffected
                            # either way) -- measured on real hardware to matter: without this
                            # branch (always taking the slow path), an existing exact-multiple-of-4
                            # shape regressed ~24% (0.132ms -> 0.164ms). gemm_n % vwo is pass-
                            # invariant, precomputed once into s_tmp2 above. m_tail's row check
                            # (if any) is cheaply recomputed fresh for each slow-path element (1
                            # extra VALU instruction) rather than saving/restoring a partially-
                            # narrowed EXEC state across elements -- avoids needing yet another
                            # scratch register. Not yet extended to wmma_acc_f16/bf16acc's packed-
                            # 2-elements-per-register layout (no existing config combines the two,
                            # asserted against in igemm_base.py -- see
                            # docs/gfx1250_optimization_backlog.md).
                            self._label_counter += 1
                            label_slow = f"L_cstore_{id(self)}_{self._label_counter}_ntail_slow"
                            label_done = f"L_cstore_{id(self)}_{self._label_counter}_ntail_done"
                            self._emit(f"s_cmp_eq_u32 s[{s_tmp2}], 0")
                            self._emit(f"s_cbranch_scc0 {label_slow}   ; Phase 51: gemm_n % {vwo} != 0 this pass -> per-element masking")
                            if ctrl.wmma_m_tail:
                                self._emit(f"v_cmpx_gt_u32 s[{s_gemm_m}], v[{v_tmp3}]   ; wmma_m_tail: row < real gemm_m (fast path)")
                            self._emit(f"v_cmpx_gt_i32 v[{v_tmp4}], 0   ; wmma_n_tail: col < real gemm_n (fast path)")
                            self._emit(f"{gst_inst} v[{v_tmp2}], v[{v_gather_range}], s[{s_p_out}:{s_p_out}+1]")
                            self._emit(f"s_mov_b32 exec_lo, -1")
                            self._emit(f"s_branch {label_done}")
                            self._emit_front(f"{label_slow}:")
                            for i in range(vwo):
                                if ctrl.wmma_m_tail:
                                    self._emit(f"v_cmpx_gt_u32 s[{s_gemm_m}], v[{v_tmp3}]   ; wmma_m_tail: row < real gemm_m (elem {i})")
                                self._emit(f"v_cmpx_gt_i32 v[{v_tmp4}], {i}   ; wmma_n_tail: col0+{i} < real gemm_n")
                                self._emit(f"global_store_dword v[{v_tmp2}], v[{v_gather}+{i}], s[{s_p_out}:{s_p_out}+1] offset:{i * elem_bytes}")
                                self._emit(f"s_mov_b32 exec_lo, -1")
                            self._emit_front(f"{label_done}:")
                        else:
                            if ctrl.wmma_m_tail:
                                # Phase 25: EXEC-mask off lanes whose absolute row for this pass is in
                                # the tail block's out-of-range tail (>= real gemm_m). Wave32-only idiom
                                # (v_cmpx narrows EXEC directly, exec_lo restore afterward) -- mirrors
                                # _emit_gld_chunk_load's existing v_flag masking in
                                # igemm_fwd_gtc_wmma_nhwc.py, not XDLOPS's 64-bit saveexec/or pattern.
                                self._emit(f"v_cmpx_gt_u32 s[{s_gemm_m}], v[{v_tmp3}]   ; wmma_m_tail: row < real gemm_m")
                            if ctrl.wmma_n_tail:
                                # Phase 26b: chained right after the M-tail guard (if any) -- wave32
                                # v_cmpx intersects with the current EXEC rather than overwriting it,
                                # so this further narrows to lanes that are ALSO column-in-range.
                                # Phase 51: "remaining > 0" is exactly the old "col0 < gemm_n" flag.
                                self._emit(f"v_cmpx_gt_i32 v[{v_tmp4}], 0   ; wmma_n_tail: col < real gemm_n")
                            self._emit(f"{gst_inst} v[{v_tmp2}], v[{v_gather_range}], s[{s_p_out}:{s_p_out}+1]")
                            if ctrl.wmma_m_tail or ctrl.wmma_n_tail:
                                self._emit(f"s_mov_b32 exec_lo, -1")
                        if it != num_passes - 1:
                            self._emit(f"v_add_u32 v[{v_tmp2}], v[{v_tmp2}], s[{s_tmp1}]   ; advance to pass {it + 1}")
                            if ctrl.wmma_m_tail:
                                self._emit(f"v_add_u32 v[{v_tmp3}], {row_step_per_pass}, v[{v_tmp3}]   ; wmma_m_tail: advance absolute row")
                    self._emit_empty_line()
        return self._get_deferred()
    def _emit_direct_store(self, ctrl, cxm, inst_wmma, v_c, v_gemm_im, v_gemm_in,
        s_p_out, s_gemm_m_stride, v_tmp1, v_tmp2, s_tmp1,
        s_gemm_m, v_tmp3, s_gemm_n, v_tmp4):
        '''
        Phase 59: direct per-lane global_store_dword epilogue, no LDS reshuffle.
        16 consecutive lanes cover 16 consecutive columns (lane%16 -> column per
        wmma_mapping.py), so the half-wave's scalar stores are already contiguous
        at the memory controller level. Eliminates the scatter/barrier/gather round
        trip. The address computation mirrors the atomic path's per-element loop
        exactly (same cur/nxt ping-pong, same row*stride+col formula, same tail
        masking) — only the store opcode differs: global_store_dword instead of
        global_atomic_add_f32. See docs/gfx1250_direct_store_plan.md.
        '''
        self._emit(f"; wmma direct per-lane store epilogue (no LDS reshuffle), "
                   f"{cxm.wave_repeat_m}x{cxm.wave_repeat_n} tiles, {inst_wmma.num_v_c} rows/tile")
        self._emit(f"s_lshl_b32 s[{s_tmp1}], s[{s_gemm_m_stride}], 2   ; row-to-row byte stride")
        for i_rm in range(cxm.wave_repeat_m):
            row_off = i_rm * cxm.wave_tile_m
            cur, nxt = v_tmp1, v_tmp2
            # cur = byte address of (row_off, col 0): row = v_gemm_im + row_off
            self._emit(f"v_add_u32 v[{cur}], {row_off}, v[{v_gemm_im}]" if row_off != 0 else f"v_mov_b32 v[{cur}], v[{v_gemm_im}]")
            if ctrl.wmma_m_tail:
                self._emit(f"v_mov_b32 v[{v_tmp3}], v[{cur}]   ; wmma_m_tail: absolute row (row {row_off}, j=0)")
            self._emit(f"v_mul_lo_u32 v[{cur}], s[{s_gemm_m_stride}], v[{cur}]")
            self._emit(f"v_add_u32 v[{cur}], v[{v_gemm_in}], v[{cur}]")
            self._emit(f"v_lshlrev_b32 v[{cur}], 2, v[{cur}]  ; byte address, row {row_off}, col 0")
            for j in range(inst_wmma.num_v_c):
                if j != inst_wmma.num_v_c - 1:
                    self._emit(f"v_add_u32 v[{nxt}], v[{cur}], s[{s_tmp1}]   ; precompute row {row_off + j + 1} address")
                for i_rn in range(cxm.wave_repeat_n):
                    c_index = (i_rm * cxm.wave_repeat_n + i_rn) * inst_wmma.num_v_c + j
                    col_off = i_rn * cxm.wave_tile_n * 4
                    offset_str = f" offset:{col_off}" if col_off != 0 else ""
                    masked = ctrl.wmma_m_tail or ctrl.wmma_n_tail
                    if ctrl.wmma_m_tail:
                        self._emit(f"v_cmpx_gt_u32 s[{s_gemm_m}], v[{v_tmp3}]   ; wmma_m_tail: row < real gemm_m")
                    if ctrl.wmma_n_tail:
                        col_val = i_rn * cxm.wave_tile_n
                        self._emit(f"v_add_u32 v[{v_tmp4}], {col_val}, v[{v_gemm_in}]" if col_val != 0 else f"v_mov_b32 v[{v_tmp4}], v[{v_gemm_in}]")
                        self._emit(f"v_cmpx_gt_u32 s[{s_gemm_n}], v[{v_tmp4}]   ; wmma_n_tail: col < real gemm_n")
                    self._emit(f"global_store_dword v[{cur}], v[{v_c}+{c_index}], s[{s_p_out}:{s_p_out}+1]{offset_str}")
                    if masked:
                        self._emit(f"s_mov_b32 exec_lo, -1")
                cur, nxt = nxt, cur
                if ctrl.wmma_m_tail and j != inst_wmma.num_v_c - 1:
                    self._emit(f"v_add_u32 v[{v_tmp3}], 1, v[{v_tmp3}]   ; wmma_m_tail: advance to row {row_off + j + 1}")
        self._emit_empty_line()
