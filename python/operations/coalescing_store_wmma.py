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

        # Phase 25 (GEMM_M tail): non-atomic path only (see igemm_base.py's mutual-exclusion
        # assert with gemm_k_global_split -- the atomic epilogue was never adapted for this).
        # 0 (default) = today's unconditional store, every existing config unaffected. 1 =
        # EXEC-mask each pass's global store off for lanes whose absolute row (block_m_off +
        # this lane's tile-local row) is >= the real (unpadded) GEMM_M -- the tail block's
        # out-of-range rows. Mirrors XDLOPS's coalescing_store.py v_cmp_gt_u32/saveexec guard,
        # but using this file's existing v_cmpx/exec_lo idiom (see igemm_fwd_gtc_wmma_nhwc.py's
        # _emit_gld_chunk_load) since WMMA is wave32-only -- no 64-bit saveexec needed, a plain
        # exec_lo restore suffices. See docs/gfx1250_wmma_layout.md's Phase 25.
        self.wmma_m_tail = 0

        # Phase 26b (GEMM_N tail): analogous to wmma_m_tail but for the column. 0 (default) =
        # unaffected. 1 = a second EXEC-mask guard, chained right after the M-tail one (wave32
        # v_cmpx intersects with the already-narrowed EXEC -- no extra VGPR needed here, since
        # `v_gather` already holds this lane's global column for every pass, per its own
        # comment below). Independent of wmma_m_tail -- either, both, or neither may be set.
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


class igemm_coalescing_store_wmma_t(mc_base_t):
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        assert type(ctrl) is ctrl_coalescing_store_wmma_t
        self.ctrl = ctrl

    def __call__(self, v_c, v_gemm_im, v_gemm_in, s_p_out, s_gemm_m_stride, v_tmp1, v_tmp2, s_tmp1, v_tid=None, v_gather=None, s_block_m_off=None, s_block_n_off=None, s_gemm_m=None, v_tmp3=None, s_gemm_n=None, v_tmp4=None):
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
        s_gemm_m/v_tmp3: only needed when ctrl.wmma_m_tail is set (non-atomic path only) --
            s_gemm_m is the real (unpadded) GEMM_M scalar, v_tmp3 is 1 scratch VGPR used to
            track each pass's absolute row for the EXEC-mask guard.
        s_gemm_n/v_tmp4: only needed when ctrl.wmma_n_tail is set (non-atomic path only) --
            s_gemm_n is the real (unpadded) GEMM_N scalar, v_tmp4 is 1 scratch VGPR holding
            this lane's column-in-range flag (`v_gather` itself gets reused as the gather's
            LDS-read destination, so its column value doesn't survive to pass 0's guard).
            Phase 34 (ctrl.atomic_pack_bf16, atomic path only -- wmma_m_tail/wmma_n_tail are
            mutually exclusive with gemm_k_global_split already, so reusing these same
            three params here is safe): v_gather holds the partner-lane byte-index
            (computed once, kernel-lifetime constant), v_tmp3 holds the cross-lane-
            exchanged partner value (per-iteration scratch), v_tmp4 holds the packed
            bf16x2 result (per-iteration scratch). v_tid is required (used for both the
            partner-index computation and the even/odd EXEC-mask guard).

        ctrl.gemm_k_global_split selects between two structurally different epilogues:

        Atomic path (gemm_k_global_split=True): by default, one global_atomic_add_f32 per
        accumulator element, no LDS traffic (there is no wide/packed fp32 atomic-add on
        this ISA, confirmed against the CDNA5 ISA doc, so the plain fp32 path can't be
        vectorized). v_tmp1/v_tmp2 ping-pong as the "current row"/"next row" address so
        the row-advance add has no data dependency on the in-flight atomics (different
        register), instead of serializing all row addresses through one register. Column
        offset is folded into the atomic's immediate offset. When ctrl.atomic_pack_bf16 is
        set instead: every lane exchanges its fp32 value with its column-adjacent partner
        (ds_bpermute_b32, lane XOR 1 -- valid since `lane % 16 -> column` in
        wmma_mapping.py always makes adjacent lanes adjacent columns within a 16-lane
        half), packs (own, partner) into one bf16x2 with the lower column in the packed
        value's low 16 bits (matching row-major memory layout), then only EVEN lanes
        issue one global_atomic_pk_add_bf16 covering both columns -- halving the number
        of actual atomic ops hitting memory at the cost of bf16 (not fp32) intermediate
        precision on the K-split reduction. The exchange needs FULL EXEC (both lanes of
        every pair must be enabled for the gather to see real, not disabled-lane-zeroed,
        values), so EXEC is narrowed to even-lanes-only only for the pack+atomic step and
        restored before the next iteration's exchange -- this repeats every (i_rm, j,
        i_rn) iteration, a real per-iteration cost this phase accepts as correctness-
        first, not yet optimized (mirrors Phase 28's TDM pilot's own "known, deliberate
        inefficiency" framing).

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
            assert not ctrl.gemm_k_global_split, "wmma_m_tail's masking is only implemented for the non-atomic epilogue branch"
            assert s_gemm_m is not None and v_tmp3 is not None
        if ctrl.wmma_n_tail:
            assert not ctrl.gemm_k_global_split, "wmma_n_tail's masking is only implemented for the non-atomic epilogue branch"
            assert s_gemm_n is not None and v_tmp4 is not None
        if ctrl.atomic_pack_bf16:
            assert ctrl.gemm_k_global_split, "atomic_pack_bf16 only applies to the atomic epilogue branch"
            assert v_tid is not None and v_gather is not None and v_tmp3 is not None and v_tmp4 is not None

        with self._deferred_context():
            if ctrl.gemm_k_global_split and ctrl.atomic_pack_bf16:
                # ---- Phase 34: packed-bf16 atomic-add epilogue ----
                self._emit(f"; wmma packed-bf16 atomic-add epilogue, {cxm.wave_repeat_m}x{cxm.wave_repeat_n} tiles, "
                           f"{inst_wmma.num_v_c} rows/tile")
                self._emit(f"s_lshl_b32 s[{s_tmp1}], s[{s_gemm_m_stride}], 1   ; row-to-row byte stride (bf16, 2 bytes/elem)")
                # partner-lane byte-index for ds_bpermute_b32 (kernel-lifetime constant,
                # computed once): partner = this lane XOR 1, index is in bytes (*4).
                self._emit(f"v_xor_b32 v[{v_gather}], 1, v[{v_tid}]   ; partner lane = this lane XOR 1")
                self._emit(f"v_lshlrev_b32 v[{v_gather}], 2, v[{v_gather}]   ; ds_bpermute_b32 index is in bytes")
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
                            # exchange needs FULL EXEC -- both lanes of every pair must be
                            # enabled or the gather sees a disabled-lane zero, not the real
                            # partner value (CDNA5 ISA doc 11.2.3).
                            self._emit(f"ds_bpermute_b32 v[{v_tmp3}], v[{v_gather}], v[{v_c}+{c_index}]")
                            # ds_bpermute_b32 is a DS-class instruction tracked by DSCNT (CDNA5
                            # ISA doc: DS return-value ops "use DSCNT to determine when this
                            # instruction has executed") -- its result is NOT safely readable
                            # by the very next instruction the way a plain VALU-to-VALU
                            # dependency is. Found by a real hardware miscompare (only the
                            # FIRST couple of (i_rm,j,i_rn) iterations showed corrupted hi16
                            # values; later iterations happened to have enough natural
                            # instruction-issue delay to mask the race) before this wait was
                            # added -- see docs/gfx1250_wmma_layout.md's Phase 34.
                            self._emit(f"s_wait_dscnt 0x0")
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
                self._emit(f"; wmma direct atomic-add epilogue, {cxm.wave_repeat_m}x{cxm.wave_repeat_n} tiles, "
                           f"{inst_wmma.num_v_c} rows/tile")
                self._emit(f"s_lshl_b32 s[{s_tmp1}], s[{s_gemm_m_stride}], 2   ; row-to-row byte stride")
                for i_rm in range(cxm.wave_repeat_m):
                    row_off = i_rm * cxm.wave_tile_m
                    cur, nxt = v_tmp1, v_tmp2
                    # cur = byte address of (row_off, col 0), i.e. row = v_gemm_im + row_off
                    self._emit(f"v_add_u32 v[{cur}], {row_off}, v[{v_gemm_im}]" if row_off != 0 else f"v_mov_b32 v[{cur}], v[{v_gemm_im}]")
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
                            # scope:SCOPE_SYS (or SCOPE_DEV, Phase 23) is load-bearing, not
                            # decoration: a bare global_atomic_add_f32 defaults to a narrower
                            # (CU/WGP-local) cache scope on gfx1250, which silently drops
                            # updates when the two accumulating workgroups land on different
                            # compute units -- confirmed on hardware. See
                            # docs/gfx1250_wmma_layout.md. th:{th} (Phase 23, optional) marks
                            # a cascading/deferred-scope atomic -- see ctrl.atomic_cascade.
                            th_str = f" th:{ctrl.atomic_th}" if ctrl.atomic_cascade else ""
                            self._emit(f"global_atomic_add_f32 v[{cur}], v[{v_c}+{c_index}], s[{s_p_out}:{s_p_out}+1]{offset_str} scope:{ctrl.atomic_scope}{th_str}")
                        cur, nxt = nxt, cur
                self._emit_empty_line()
            else:
                # ---- non-atomic: LDS-reshuffle coalescing store ----
                assert v_tid is not None and v_gather is not None and s_block_m_off is not None and s_block_n_off is not None
                vwo = ctrl.vector_write_out
                macro_tile_m = cxm.macro_tile_m
                macro_tile_n = cxm.macro_tile_n
                assert macro_tile_n % vwo == 0, f"macro_tile_n:{macro_tile_n} not divisible by vector_write_out:{vwo}"
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
                    self._emit(f"v_add_u32 v[{v_tmp1}], {row_off}, v[{v_gemm_im}]" if row_off != 0 else f"v_mov_b32 v[{v_tmp1}], v[{v_gemm_im}]")
                    self._emit(f"v_and_b32 v[{v_tmp1}], {macro_tile_m - 1}, v[{v_tmp1}]   ; tile-local row")
                    if pad:
                        self._emit(f"v_mul_lo_u32 v[{v_tmp1}], {padded_stride}, v[{v_tmp1}]   ; row * padded_stride")
                    else:
                        self._emit(f"v_lshlrev_b32 v[{v_tmp1}], {log2_n}, v[{v_tmp1}]   ; row << log2(macro_tile_n)")
                    self._emit(f"v_add_u32 v[{v_tmp1}], v[{v_tmp2}], v[{v_tmp1}]   ; + col -> tile-linear index, row {row_off}")
                    self._emit(f"v_lshlrev_b32 v[{v_tmp1}], {elem_byte_shift}, v[{v_tmp1}]   ; byte address")
                    # Phase 24: for f16acc, VGPR j packs TWO logical rows (2j lo-half, 2j+1
                    # hi-half, per the ISA doc's 16-bit C/D-matrix table) -- v_tmp1 tracks the
                    # LO row's address, stepping 2 rows (not 1) per j; the HI row's write
                    # reuses the SAME v_tmp1 base with one extra row's stride folded into its
                    # offset immediate (no extra address-compute instruction needed).
                    row_step = 2 if ctrl.wmma_acc_f16 else 1
                    for j in range(inst_wmma.num_v_c):
                        if j != 0:
                            self._emit(f"v_add_u32 v[{v_tmp1}], {padded_stride * elem_bytes * row_step}, v[{v_tmp1}]   ; advance to row {row_off + j * row_step}")
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
                    # Phase 26b: this lane's column-in-range flag, captured NOW -- v_gather
                    # itself is about to be reused as the gather's LDS-read destination
                    # register (v_gather_range aliases v_gather, see below), so its "global
                    # column" value would otherwise be gone before pass 0's guard needs it.
                    # Pass-invariant (column never changes across passes), so this is computed
                    # once here, not per-pass.
                    self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s_gemm_n}], v[{v_gather}]")
                    self._emit(f"v_cndmask_b32 v[{v_tmp4}], 0, 1, vcc_lo   ; wmma_n_tail: col < real gemm_n")
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
                        # v_tmp4 holds the pass-invariant column flag computed once above (see
                        # the flag idiom used by _emit_gld_chunk_load elsewhere in this
                        # codebase: v_cmpx_le_u32 1, v[flag]).
                        self._emit(f"v_cmpx_le_u32 1, v[{v_tmp4}]   ; wmma_n_tail: col < real gemm_n")
                    self._emit(f"{gst_inst} v[{v_tmp2}], v[{v_gather_range}], s[{s_p_out}:{s_p_out}+1]")
                    if ctrl.wmma_m_tail or ctrl.wmma_n_tail:
                        self._emit(f"s_mov_b32 exec_lo, -1")
                    if it != num_passes - 1:
                        self._emit(f"v_add_u32 v[{v_tmp2}], v[{v_tmp2}], s[{s_tmp1}]   ; advance to pass {it + 1}")
                        if ctrl.wmma_m_tail:
                            self._emit(f"v_add_u32 v[{v_tmp3}], {row_step_per_pass}, v[{v_tmp3}]   ; wmma_m_tail: advance absolute row")
                self._emit_empty_line()
        return self._get_deferred()
