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
# pylint: disable=maybe-no-member
from ..codegen import *
from ..operations import *
import os
from .igemm_base import *

class igemm_bwd_gtc_wmma_nhwc_t(mc_base_t):
    '''
    gfx1250 WMMA kernel for backward-data (grad-input) convolution. Phase 5b added arbitrary
    stride/padding (1x1 filter only). Phase 5e now adds multi-tap filters (y,x >= 1) and
    dilation, mirroring igemm_fwd_gtc_wmma_nhwc_t's Phase 5d design (a runtime, not compile-
    time-unrolled, outer loop over the y*x taps wrapping a single static WMMA K-main-loop
    emission -- see that class's docstring for the full rationale):

        grad_input[n,hi,wi, c] = sum_{iy,ix,k} grad_output[n, (hi+pad_h-iy*dilation_h)/stride_h,
                                                            (wi+pad_w-ix*dilation_w)/stride_w, k]
                                              * weight[k, iy, ix, c]
                                  (0 if either division isn't exact, or the resulting ho/wo
                                   index is out of bounds -- a "stride gap": for stride>1, most
                                   input pixels have NO corresponding output pixel at all, and
                                   this is now evaluated fresh per tap since different taps
                                   probe different candidate output pixels)

    where m enumerates (n_batch, hi, wi) INPUT pixels (GEMM_M = n*hi*wi -- this is
    grad_input's own pixel count, note the input, not output, spatial extent), k enumerates
    output channels PER TAP (GEMM_K = k_out, unchanged from Phase 5b; the y*x taps are a
    separate outer loop, not folded into GEMM_K), n enumerates input channels (GEMM_N = c_in).

    Weight is stored [K_out, Y, X, C_in] (C_in innermost, standard conv layout -- see
    driver/naive_conv.h and igemm_fwd_gtc_wmma_nhwc_t's Phase 5d docstring for the exact
    reference formula), so its K_out-stride is `y*x*c` elements (not just `c` as in the 1x1
    case): `s_wei_row_c = gemm_n * x * y` replaces the old plain `gemm_n` (=C_in) as the
    per-K_out-row element multiplier everywhere it's used (both the fixed row_local/col_group
    base address and the per-K-iteration `move_slice_window_b` stride). A given tap's
    C_in-column-block starts `(iy*x+ix)*c` elements into that row -- a fixed byte offset added
    to a per-thread base (`v_addr_b_base`) fresh every tap, exactly like fwd's B operand.

    Tensor A (grad_output) needs the harder per-tap gather: this thread's (n_idx, hi_idx,
    wi_idx) are decomposed from GEMM_M exactly ONCE (persistent VGPRs, same division sequence
    Phase 5b introduced), then EVERY tap recomputes `numerator_h = hi_idx + pad_h -
    iy*dilation_h` (note the SUBTRACT -- opposite sign from fwd's per-tap ADD, since bwd's
    formula is fwd's inverted; a negative numerator wraps to a huge u32, which the subsequent
    division's bounds check naturally rejects via quotient overflow, the same "unsigned
    wraparound rejects invalid" trick used everywhere else in this milestone, just one
    division removed from the comparison) and re-derives ho_idx/flag via the same
    divide-and-check-both-remainder-and-bounds sequence Phase 5b used. Since the flag can now
    flip between taps, `global_load_a_functor` re-zeros all 16 `v_gld_a` registers on EVERY
    call (Phase 5c/wrw's discipline), not just once.

    Weight's transposed LDS read (shared_load_b_functor, get_gemm_index_for_src_matrix_transposed)
    and the output (grad_input) write are UNAFFECTED by taps -- they only ever see LDS-local
    byte offsets within whichever 32-row-by-128-col slice of the weight tile is currently
    resident, populated correctly by global_load_b_functor+shared_store_b_functor regardless of
    which absolute tap the global address pointed to.

    Same tile-shape constraints as the fwd kernel: 128x128x32, wmma_repeat 4x4,
    block_size 128, precision fp16/bf16. gemm_m/gemm_n multiples of 128, gemm_k
    (K_out) a multiple of 32.
    '''
    def __init__(self, mc, tunable):
        mc_base_t.__init__(self, mc)
        assert tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA
        assert tunable.precision in ('fp16', 'bf16', 'int8', 'fp32'), f'unsupported precision:{tunable.precision}'
        assert tunable.tensor_layout == 'nhwc'
        # tile shape is not pinned to 128x128/4x4/block_size=128 -- any shape is accepted as
        # long as it is internally consistent. igemm_gtc_tunable_parameter_t.__init__
        # (igemm_base.py) already derives/validates tunable.block_size generically from these
        # same fields before we ever get here; this is a second, kernel-local check with the
        # same formula (gfx1250 WMMA is always wave32).
        assert tunable.gemm_m_per_block % (tunable.wmma_tile_m * tunable.wmma_repeat_m) == 0
        assert tunable.gemm_n_per_block % (tunable.wmma_tile_n * tunable.wmma_repeat_n) == 0
        waves_per_m = tunable.gemm_m_per_block // (tunable.wmma_tile_m * tunable.wmma_repeat_m)
        waves_per_n = tunable.gemm_n_per_block // (tunable.wmma_tile_n * tunable.wmma_repeat_n)
        assert tunable.block_size == waves_per_m * waves_per_n * 32
        self.tunable = tunable
        self.data_byte = amdgpu_precision_data_byte(tunable.precision)
        # Phase 54 (VGPR-MSB): see igemm_fwd_gtc_wmma_nhwc.py's identical comment --
        # one tracker shared by the prologue's v_c zero-init, main loop, and epilogue.
        self.vgpr_msb_tracker = vgpr_msb_tracker_t() if tunable.wmma_acc_high_bank else None

        # Asymmetric tile shapes (2026-08-25): row_repeat_a generalizes A (grad_output,
        # untransposed) the same way igemm_fwd_gtc_wmma_nhwc_t's row_repeat_a does -- thread
        # tid owns global rows tid, tid+block_size, ..., row_repeat_a of them. B (weight) is
        # TRANSPOSED here (see class docstring / get_gemm_index_for_src_matrix_transposed) --
        # its per-thread row_local/col_group bit-sliced addressing is a fundamentally
        # different mechanism from A's/fwd's simple "row=tid" mapping, so row_repeat_b (B
        # needing multiple rows/thread) is NOT yet implemented; only shapes where
        # gemm_n_per_block == block_size (row_repeat_b==1) are supported for now.
        assert tunable.gemm_m_per_block % tunable.block_size == 0, \
            f"gemm_m_per_block({tunable.gemm_m_per_block}) must be a multiple of block_size({tunable.block_size})"
        assert tunable.gemm_n_per_block % tunable.block_size == 0, \
            f"gemm_n_per_block({tunable.gemm_n_per_block}) must be a multiple of block_size({tunable.block_size})"
        self.row_repeat_a = tunable.gemm_m_per_block // tunable.block_size
        self.row_repeat_b = tunable.gemm_n_per_block // tunable.block_size
        assert self.row_repeat_b == 1, \
            "row_repeat_b > 1 (B needing multiple rows/thread) is not implemented for bwd's transposed B"
        assert not (tunable.async_global_load and self.row_repeat_a > 1), \
            "row_repeat_a > 1 is not yet supported together with async_global_load"
        # Phase 42 (TDM global load, bwd): mirrors igemm_fwd_gtc_wmma_nhwc_t's identical
        # Phase 28 asserts -- narrowest correctness-first slice.
        assert not (tunable.tdm_global_load and self.row_repeat_a > 1), \
            "tdm_global_load is not yet supported together with row_repeat_a > 1"
        assert not (tunable.tdm_global_load and tunable.local_prefetch_num > 1), \
            "tdm_global_load is not yet supported together with local_prefetch_num > 1"
        assert not (tunable.tdm_global_load and tunable.async_global_load), \
            "tdm_global_load and async_global_load are mutually exclusive -- two different load mechanisms for the same operand"
        # Phase 15 (main-loop interleaving, bwd port): A (grad_output, untransposed)
        # interleaves -- same as fwd's A/B. B (weight, TRANSPOSED) does NOT interleave: its
        # shared_load_b_functor reuses v_gld_b as scratch for the read-and-pack technique,
        # and interleaving would issue chunk N+1's load into v_gld_b between substep N's
        # shared_load (scratch clobber) and chunk N's store -- the same clobbering risk that
        # already forces global_load_b_functor to issue nothing when num_k_chunks>1 (see
        # _emit_sst_all_chunks). interleave_b stays False unconditionally. Requires
        # lds_double_buffer=1 (same cross-wave LDS race as fwd, confirmed on hardware) and
        # is mutually exclusive with async_global_load/tdm_global_load (A's path -- no
        # staging buffer to interleave around) and row_repeat_a>1 (untested combination).
        assert not (tunable.main_loop_interleave and tunable.async_global_load), \
            "main_loop_interleave is not supported together with async_global_load"
        assert not (tunable.main_loop_interleave and tunable.tdm_global_load), \
            "main_loop_interleave is not supported together with tdm_global_load"
        assert not (tunable.main_loop_interleave and self.row_repeat_a > 1), \
            "main_loop_interleave is not yet supported together with row_repeat_a > 1"
        assert not (tunable.main_loop_interleave and not tunable.lds_double_buffer), \
            "main_loop_interleave requires lds_double_buffer=1 (single-buffered interleaving races across waves, confirmed on hardware)"
        assert not (tunable.main_loop_interleave and tunable.gemm_k_global_split), \
            "gemm_k_global_split is not yet combined with main_loop_interleave for bwd -- not audited together"
        # Phase 61 (32-bit SADDR global loads, bwd port): same discipline as fwd's pilot --
        # mutually exclusive with async/tdm (both have their own non-VADDR-pair addressing),
        # main_loop_interleave (not audited together), gemm_k_global_split (not audited against
        # the base-pointer shard offset), and row_repeat_a>1 (untested combination).
        assert not (tunable.saddr_global_load and tunable.async_global_load), \
            "saddr_global_load and async_global_load are mutually exclusive -- both are alternatives to the default 64-bit VADDR-pair path"
        assert not (tunable.saddr_global_load and tunable.tdm_global_load), \
            "saddr_global_load and tdm_global_load are mutually exclusive -- both are alternatives to the default 64-bit VADDR-pair path"
        assert not (tunable.saddr_global_load and tunable.main_loop_interleave), \
            "saddr_global_load is not yet combined with main_loop_interleave for bwd -- not audited together"
        assert not (tunable.saddr_global_load and tunable.gemm_k_global_split), \
            "saddr_global_load is not yet combined with gemm_k_global_split for bwd -- not audited against the base-pointer shard offset"
        assert not (tunable.saddr_global_load and self.row_repeat_a > 1), \
            "saddr_global_load is not yet supported together with row_repeat_a > 1 (untested combination)"
        # Phase 48 (gemm_k_global_split, bwd): not yet combined with TDM -- TDM's own
        # s_tdm_k_remain init reads s_gemm_k directly (the true, un-sharded total), not
        # s_knum/s_gemm_k_per_wg, so combining the two would silently give every shard the
        # FULL K range instead of its own slice. Also not yet combined with wmma_k_tail (no
        # last-shard clamp implemented for bwd's split-K yet, unlike wrw's).
        assert not (tunable.tdm_global_load and tunable.gemm_k_global_split), \
            "gemm_k_global_split is not yet combined with tdm_global_load for bwd -- TDM's s_tdm_k_remain init would need to use s_knum, not the un-sharded s_gemm_k"
        assert not (tunable.wmma_k_tail and tunable.gemm_k_global_split), \
            "gemm_k_global_split is not yet combined with wmma_k_tail for bwd -- no last-shard remainder clamp implemented yet (see wrw's s_gemm_k_tail/s_gemm_k_num_splits for the pattern to port)"
        # Phase 29/42: fresh-label counter for _emit_wave0_only (mirrors fwd).
        self._tdm_label_counter = 0

        # Phase 24/27: 'fp16_f16acc'/'bf16_bf16acc' are separate table keys (not fields), see
        # wmma_mapping.py -- pick the num_v_c=4 narrow-accumulate instruction instead of the
        # num_v_c=8 f32-accumulate one.
        if tunable.wmma_acc_f16:
            wmma_mapping_key = tunable.precision + '_f16acc'
        elif tunable.wmma_acc_bf16:
            wmma_mapping_key = tunable.precision + '_bf16acc'
        else:
            wmma_mapping_key = tunable.precision
        ctrl_wmma_mapping = get_ctrl_wmma_mapping_from_wave_tile(tunable.gemm_m_per_block, tunable.gemm_n_per_block,
                tunable.wmma_tile_m, tunable.wmma_tile_n, tunable.wmma_repeat_m, tunable.wmma_repeat_n,
                tunable.block_size // tunable.wave_size, wmma_mapping_key)
        # gemm_k_per_block must be a multiple of the wired-up instruction's K (32 for
        # fp16/bf16, 64 for int8, 4 for fp32) -- Phase 1's k-sub-loop (wmma_main_loop.py)
        # issues gemm_k_per_block // inst_wmma.k v_wmma_* calls per LDS round-trip.
        assert tunable.gemm_k_per_block % ctrl_wmma_mapping.inst_wmma.k == 0, \
            f"gemm_k_per_block({tunable.gemm_k_per_block}) must be a multiple of inst_wmma.k({ctrl_wmma_mapping.inst_wmma.k}) for precision {tunable.precision}"
        self.wmma_mapping = igemm_wmma_mapping_t(self.mc, ctrl_wmma_mapping)

        ctrl_coalescing_store_wmma = ctrl_coalescing_store_wmma_t()
        ctrl_coalescing_store_wmma.cxm = ctrl_wmma_mapping
        ctrl_coalescing_store_wmma.block_size = tunable.block_size
        ctrl_coalescing_store_wmma.precision = tunable.precision
        ctrl_coalescing_store_wmma.atomic_scope = tunable.atomic_scope
        ctrl_coalescing_store_wmma.atomic_cascade = tunable.atomic_cascade
        ctrl_coalescing_store_wmma.epilogue_lds_pad = tunable.epilogue_lds_pad
        # Phase 27: see igemm_fwd_gtc_wmma_nhwc.py's identical comment -- the ctrl field's
        # actual behavior is precision-agnostic (2-byte-packed accumulator), so both tunables
        # funnel into it.
        ctrl_coalescing_store_wmma.wmma_acc_f16 = tunable.wmma_acc_f16 or tunable.wmma_acc_bf16
        ctrl_coalescing_store_wmma.wmma_m_tail = tunable.wmma_m_tail
        ctrl_coalescing_store_wmma.wmma_n_tail = tunable.wmma_n_tail
        # Phase 48: switches the shared epilogue from its direct (LDS-reshuffle) store path
        # to an atomic-add path -- direction-agnostic, mirrors wrw's identical wiring.
        ctrl_coalescing_store_wmma.gemm_k_global_split = tunable.gemm_k_global_split
        ctrl_coalescing_store_wmma.direct_store          = tunable.direct_store
        # Phase 53: see igemm_fwd_gtc_wmma_nhwc.py's identical comment.
        ctrl_coalescing_store_wmma.wmma_epilogue_chunked = tunable.wmma_epilogue_chunked
        ctrl_coalescing_store_wmma.vgpr_msb_tracker = self.vgpr_msb_tracker
        self.coalescing_store = igemm_coalescing_store_wmma_t(self.mc, ctrl_coalescing_store_wmma)
        # K/N-tail (bwd-specific): bwd's B (weight) operand is TRANSPOSED (see class
        # docstring) -- unlike fwd's B, where each lane owns one fixed N-column and reads
        # gemm_k_per_block elements CONTIGUOUSLY ALONG K, bwd's B has each lane own one
        # fixed row_local (K-position) and read gemm_k_per_block elements contiguously
        # ALONG N. This inverts which axis's tail is "easy" (a whole-chunk, per-lane EXEC
        # decision) vs "hard" (a fine-grained, sub-lane decision that can fall in the middle
        # of one lane's own multi-element load, needing a per-dword AND-mask instead of
        # EXEC): B's K-tail is easy (row_local IS the per-lane K position), B's N-tail is
        # hard (N spans multiple consecutive elements within one lane's own chunk). A has no
        # N-axis role at all; A's K-tail is hard for the same reason wrw's K-tail needed new
        # machinery -- unlike wrw though, A's per-lane K range here is a flat linear index
        # (no spatial n/ho/wo decomposition), so the mask itself is simpler even though the
        # masking MECHANISM (fine-grained, not EXEC) is the same new primitive as B's N-tail.
        # row_repeat_a > 1 is not supported together with wmma_k_tail (untested combination,
        # every existing config has row_repeat_a==1 anyway -- see __init__'s own docstring).
        assert not (tunable.wmma_k_tail and self.row_repeat_a > 1), \
            "wmma_k_tail is not supported together with row_repeat_a > 1"
        self._tail_mask_label_id = 0

        # gemm_k_per_block*data_byte happens to equal 64 bytes for fp16/bf16/int8 (32*2, 32*2,
        # 64*1), but fp32 forces gemm_k_per_block=4 (matching inst_wmma.k), giving 4*4=16 --
        # see igemm_fwd_gtc_wmma_nhwc_t's Phase 8 docstring for the full rationale. Every
        # literal built on the 64-byte coincidence is now derived from these instead.
        self.bytes_per_row = tunable.gemm_k_per_block * self.data_byte
        self.num_dwordx4   = self.bytes_per_row // 16
        self.num_dwords    = self.bytes_per_row // 4
        # Phase 1 (k-sub-loop): the WMMA "k_half" wave-split (lane>>4) only ever applies
        # WITHIN one inst_wmma.k-wide instruction, never across the whole (possibly
        # multi-substep) gemm_k_per_block row -- must be derived from inst_wmma.k, NOT
        # bytes_per_row (which now can be a multiple of inst_wmma.k*data_byte).
        self.inst_wmma_k_bytes = ctrl_wmma_mapping.inst_wmma.k * self.data_byte
        # Phase 1 (k-sub-loop): global_load/shared_store are chunked into num_k_chunks
        # rounds of one inst_wmma.k-worth each, reusing the same small v_gld_a/b buffer
        # across chunks -- see igemm_fwd_gtc_wmma_nhwc_t's __init__ docstring.
        self.chunk_num_dwordx4 = self.inst_wmma_k_bytes // 16
        self.chunk_num_dwords  = self.inst_wmma_k_bytes // 4
        self.num_k_chunks      = self.num_dwordx4 // self.chunk_num_dwordx4

        # A-region (grad_output): natural [GEMM_M][GEMM_K] tile, same shape as fwd's A/B.
        self.lds_a_size = tunable.gemm_m_per_block * tunable.gemm_k_per_block * self.data_byte
        # B-region (weight): natural [GEMM_K][GEMM_N] tile -- same total byte size (32*128*databyte
        # either way), just a transposed interpretation of the same linear LDS layout.
        self.lds_b_size = tunable.gemm_k_per_block * tunable.gemm_n_per_block * self.data_byte
        # Phase 2 (double-buffering): see igemm_fwd_gtc_wmma_nhwc_t's identically-named
        # fields for the full rationale.
        self.lds_single_size = igemm_next_pow2(self.lds_a_size + self.lds_b_size)
        self.lds_buffer_num  = 2 if tunable.lds_double_buffer else 1

        self.sgpr = self.kernel_sgpr_t(mc, self)
        self.vgpr = self.kernel_vgpr_t(mc, self)

    def name(self):
        return igemm_gtc_encode_kernel_name(self.tunable, self.mc.arch_config.arch)

    def get_kernel_macros(self):
        # plain (non-magic) runtime u32 division, wave32-safe (see class docstring / Phase 5a).
        return [macro_int_div_vs_gfx1250_t(self.mc), macro_int_div_rem_vs_gfx1250_t(self.mc)]

    class kernel_sgpr_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            sseq = gpr_sequencer_t()
            self.s_ka          = sym_t('s_ka'          , sseq(2))
            self.s_bx          = sym_t('s_bx'          , sseq(1))    # workgroup_id_x -> gemm_m block index
            self.s_by          = sym_t('s_by'          , sseq(1))    # workgroup_id_y -> gemm_n block index
            self.s_p_in        = sym_t('s_p_in'        , sseq(2))    # grad_output (A operand)
            self.s_p_wei       = sym_t('s_p_wei'       , sseq(2))    # weight (B operand)
            self.s_p_out       = sym_t('s_p_out'       , sseq(2))    # grad_input (output)
            self.s_gemm_m      = sym_t('s_gemm_m'      , sseq(1))    # N*Hi*Wi
            self.s_gemm_n      = sym_t('s_gemm_n'      , sseq(1))    # C_in
            self.s_gemm_k      = sym_t('s_gemm_k'      , sseq(1))    # K_out
            # stride/pad kernarg fields (Phase 5b)
            self.s_hi_wi       = sym_t('s_hi_wi'       , sseq(1))    # = hi*wi, divisor for (n, hi*wi) decomposition
            self.s_wi          = sym_t('s_wi'          , sseq(1))    # divisor for (hi_idx, wi_idx) decomposition
            self.s_stride_h    = sym_t('s_stride_h'    , sseq(1))
            self.s_stride_w    = sym_t('s_stride_w'    , sseq(1))
            self.s_pad_h       = sym_t('s_pad_h'       , sseq(1))
            self.s_pad_w       = sym_t('s_pad_w'       , sseq(1))
            self.s_ho          = sym_t('s_ho'          , sseq(1))    # bound + row-pitch factor for grad_output address
            self.s_wo          = sym_t('s_wo'          , sseq(1))    # bound + divisor for grad_output address
            self.s_ho_wo       = sym_t('s_ho_wo'       , sseq(1))    # = s_ho*s_wo, computed on-device once
            # Phase 60 (Magic Division): host-precomputed magic multipliers + packed shifts
            # for the hi_wi/wi/stride_h/stride_w divisors. Loaded from kernargs (computed
            # driver-side via magic_div_u32_gen), replacing the ~24-instruction emulated
            # macro_int_div_rem_vs_gfx1250_t with 5-instruction magic multiply.
            # s_magic_hi_wi is 4-aligned (sseq(1, 4)) for s_load_dwordx4.
            self.s_magic_hi_wi     = sym_t('s_magic_hi_wi'    , sseq(1, 4))
            self.s_magic_wi         = sym_t('s_magic_wi'      , sseq(1))
            self.s_magic_stride_h   = sym_t('s_magic_stride_h', sseq(1))
            self.s_magic_stride_w   = sym_t('s_magic_stride_w', sseq(1))
            self.s_shift_pack       = sym_t('s_shift_pack'    , sseq(1))
            self.s_shift_hi_wi      = sym_t('s_shift_hi_wi'   , sseq(1))
            self.s_shift_wi         = sym_t('s_shift_wi'      , sseq(1))
            self.s_shift_stride_h   = sym_t('s_shift_stride_h', sseq(1))
            self.s_shift_stride_w   = sym_t('s_shift_stride_w', sseq(1))
            # Phase 5e (multi-tap + dilation) kernarg fields, contiguous, matching
            # get_kernel_args()'s trailing layout.
            self.s_y           = sym_t('s_y'           , sseq(1))
            self.s_x           = sym_t('s_x'           , sseq(1))
            self.s_dilation_h  = sym_t('s_dilation_h'  , sseq(1))
            self.s_dilation_w  = sym_t('s_dilation_w'  , sseq(1))
            self.s_wei_row_c   = sym_t('s_wei_row_c'   , sseq(1))    # = gemm_n*x*y, weight's per-K_out-row element count
            self.s_iy          = sym_t('s_iy'          , sseq(1))   # runtime tap-loop counters
            self.s_ix          = sym_t('s_ix'          , sseq(1))
            self.s_group_idx   = sym_t('s_group_idx'   , sseq(1))   # decoded from s_by (group folded into grid_y)
            self.s_group       = sym_t('s_group'       , sseq(1))   # kernarg: total group count
            self.s_a_k_total   = sym_t('s_a_k_total'   , sseq(1))   # = gemm_k*group, A(grad_output)'s per-pixel row stride
            self.s_out_c_total = sym_t('s_out_c_total' , sseq(1))   # = gemm_n*group, output(grad_input)'s per-pixel row stride
            self.s_block_m_off = sym_t('s_block_m_off' , sseq(1))
            self.s_block_n_off = sym_t('s_block_n_off' , sseq(1))
            self.s_wei_k_stride = sym_t('s_wei_k_stride', sseq(1))   # s_wei_row_c*databyte*gemm_k_per_block: weight's per-K-block global stride
            self.s_kitr        = sym_t('s_kitr'        , sseq(1))
            self.s_knum        = sym_t('s_knum'        , sseq(1))
            # Phase 48 (gemm_k_global_split, K-split across grid.z): only loaded/used when
            # outer.tunable.gemm_k_global_split is set, but always declared for a uniform
            # register layout between split and non-split kernel variants -- mirrors wrw's
            # identical Phase 35-era fields (igemm_wrw_gtc_wmma_nhwc.py).
            self.s_bz             = sym_t('s_bz'             , sseq(1))   # workgroup_id_z -> this workgroup's K-slice index
            self.s_gemm_k_per_wg  = sym_t('s_gemm_k_per_wg'  , sseq(1))   # kernarg: this workgroup's K-slice length
            self.s_gemm_k_wg_off  = sym_t('s_gemm_k_wg_off'  , sseq(1))   # = s_bz * s_gemm_k_per_wg
            if outer.tunable.tdm_global_load:
                # Phase 42: TDM descriptor for A (grad_output) -- structurally identical
                # port of igemm_fwd_gtc_wmma_nhwc_t's Phase 28 A-operand descriptor (same
                # NHWC-contiguous-per-pixel property), just using bwd's own SGPR names.
                self.s_tdm_g0  = sym_t('s_tdm_g0'      , sseq(4, 4))
                self.s_tdm_g1  = sym_t('s_tdm_g1'      , sseq(8, 4))
                # TDM descriptor for B (weight) -- tensor_dim0/tensor_dim1's roles are
                # SWAPPED relative to fwd's B descriptor: bwd's GEMM_K (K_out) is weight's
                # ROW axis here (not its contiguous axis, unlike fwd where GEMM_K=C_in is
                # the contiguous axis) -- see _emit_tdm_descriptor_setup_b's docstring.
                self.s_tdm_g0_b = sym_t('s_tdm_g0_b'   , sseq(4, 4))
                self.s_tdm_g1_b = sym_t('s_tdm_g1_b'   , sseq(8, 4))
                self.s_wave_id = sym_t('s_wave_id'     , sseq(1))
                # Remaining valid K (K_out) elements from the tile about to be issued --
                # mirrors fwd's identical Phase 31 field.
                self.s_tdm_k_remain = sym_t('s_tdm_k_remain', sseq(1))
            if outer.tunable.wmma_n_tail:
                # N-tail's skip-branch (see _emit_tail_dword_mask_guarded) needs a scratch
                # SGPR to hold "this block's exclusive N end" -- kept dedicated rather than
                # reusing s_tmp to avoid any risk of clobbering a live s_tmp value at one of
                # this check's several call sites (global_load_b_functor, shared_store_b_functor).
                self.s_tail_tmp = sym_t('s_tail_tmp'   , sseq(1))
            self.s_tmp         = sym_t('s_tmp'         , sseq(4))
            self.s_end         = sym_t('s_end'         , sseq())

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('s_'):
                    self._emit(v.declare())

    class kernel_vgpr_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            vseq = gpr_sequencer_t()
            # Phase 54 (VGPR-MSB): see igemm_fwd_gtc_wmma_nhwc.py's identical comment --
            # v_c gets its own bank-1 sequencer instead of sharing bank 0's vseq.
            v_c_vseq = gpr_sequencer_t() if outer.tunable.wmma_acc_high_bank else vseq
            self.v_c           = sym_t('v_c'           , v_c_vseq(outer.tunable.num_vgpr_accumulate_c))     # 128
            self.v_a           = sym_t('v_a'           , vseq(outer.tunable.num_vgpr_accumulate_a))     # 32
            self.v_b           = sym_t('v_b'           , vseq(outer.tunable.num_vgpr_accumulate_b))     # 32
            # Phase 1 (k-sub-loop): sized to outer.num_dwords (=bytes_per_row//4), NOT a
            # hardcoded 16 -- bytes_per_row can now exceed 64B once gemm_k_per_block is a
            # multiple (not just equal to) inst_wmma.k, so the old fixed-16 allocation
            # silently overflowed into whatever VGPR followed it.
            if outer.tunable.async_global_load:
                # Phase 13: A (untransposed) has no VGPR staging buffer at all -- see
                # igemm_fwd_gtc_wmma_nhwc.py's kernel_vgpr_t for the full rationale. B stays
                # on the old technique (transposed, reuses v_gld_b as scratch -- out of
                # scope for this phase), so v_gld_b is always declared.
                self.v_zero    = sym_t('v_zero'        , vseq(4))
            else:
                self.v_gld_a   = sym_t('v_gld_a'       , vseq(outer.chunk_num_dwords))
            self.v_gld_b   = sym_t('v_gld_b'       , vseq(outer.chunk_num_dwords))   # also reused as scratch by the transposed shared_load_b
            self.v_tid         = sym_t('v_tid'         , vseq(1))
            # 64-bit VADDR pairs must be even-aligned on gfx1250 (verified with llvm-mc)
            if outer.tunable.async_global_load or outer.tunable.saddr_global_load:
                # Phase 13/61: plain 32-bit per-lane byte OFFSET (SADDR carries the 64-bit base
                # separately) -- see igemm_fwd_gtc_wmma_nhwc.py's kernel_vgpr_t docstring.
                self.v_off_a   = sym_t('v_off_a'        , vseq(1))
                # Phase 13 bugfix: see igemm_fwd_gtc_wmma_nhwc.py's v_sst_tmp declaration --
                # not currently exercised here (this operand's sst_extra_off is always 0,
                # B stays on the old technique), kept for parity/future-proofing.
                self.v_sst_tmp = sym_t('v_sst_tmp'      , vseq(1))
            else:
                # row_repeat_a copies, mirroring igemm_fwd_gtc_wmma_nhwc_t's v_addr_a -- see
                # that file's __init__ docstring. row_repeat_a==1 for every existing config
                # (byte-identical).
                self.v_addr_a  = sym_t('v_addr_a'       , vseq(2 * outer.row_repeat_a, 2))    # persistent global A address(es) (64-bit each)
            if outer.tunable.saddr_global_load:
                # Phase 61: B also uses 32-bit offset (SADDR carries s_p_wei separately).
                # v_off_b_base is B's tap-independent base offset (reset per tap, same as
                # v_addr_b_base in the VADDR path).
                self.v_off_b      = sym_t('v_off_b'      , vseq(1))
                self.v_off_b_base = sym_t('v_off_b_base' , vseq(1))
            else:
                self.v_addr_b      = sym_t('v_addr_b'      , vseq(2, 2))
                # Phase 5e: B's fixed per-thread base (before this tap's column offset is added) --
                # computed once from the now-correct y*x*c row stride, reset into v_addr_b fresh
                # every tap (then move_slice_window_b bumps v_addr_b across K-iterations within a tap).
                self.v_addr_b_base = sym_t('v_addr_b_base' , vseq(2, 2))
            self.v_addr_out    = sym_t('v_addr_out'    , vseq(2))    # scratch used by coalescing_store_wmma (ping-pong pair)
            if outer.tunable.wmma_epilogue_chunked:
                # Phase 53 bugfix: see igemm_fwd_gtc_wmma_nhwc.py's identical comment --
                # persistent per-lane global column, needed across every pass of the
                # chunked epilogue's per-group gather.
                self.v_chunked_col = sym_t('v_chunked_col' , vseq(1))
            if outer.tunable.wmma_m_tail:
                # Phase 26a: extra scratch for coalescing_store_wmma's per-pass absolute-row
                # EXEC-mask guard -- only allocated when wmma_m_tail is set (every existing
                # config is byte-identical, this register simply doesn't exist otherwise).
                # See igemm_fwd_gtc_wmma_nhwc_t's identical Phase 25 addition -- this kernel
                # is already at the hard 256-VGPR/wave limit for some tile shapes (see
                # _emit_lds_offset_setup's docstring), so this 1 extra VGPR needs checking
                # against .vgpr_count per shape, not assumed to always fit.
                self.v_m_tail_row = sym_t('v_m_tail_row' , vseq(1))
            if outer.tunable.wmma_n_tail:
                # N-tail (B/weight, hard case -- see __init__'s docstring): v_n_tail_col is
                # the epilogue's scratch (mirrors v_m_tail_row); v_b_col_start_abs/
                # v_n_valid_base are load-side, kernel-lifetime constants (col_start_abs
                # doesn't depend on the K-loop, unlike K-tail's remaining count) computed
                # once in the prologue and reused by every shared_store_b_functor call.
                self.v_n_tail_col      = sym_t('v_n_tail_col'      , vseq(1))
                self.v_b_col_start_abs = sym_t('v_b_col_start_abs' , vseq(1))
                self.v_n_valid_base    = sym_t('v_n_valid_base'    , vseq(1))
            if outer.tunable.wmma_k_tail:
                # K-tail (A's hard fine-grained mask + B's easy per-lane EXEC flag -- see
                # __init__'s docstring). v_b_row_local is a kernel-lifetime constant
                # (persisted out of the existing row_local computation in
                # emit_kernel_prologue, since it would otherwise be overwritten by the very
                # next instruction); v_flag_b_ktail is recomputed fresh every
                # global_load_b_functor call (s_kitr changes every K-loop iteration).
                self.v_b_row_local  = sym_t('v_b_row_local'  , vseq(1))
                self.v_flag_b_ktail = sym_t('v_flag_b_ktail' , vseq(1))
            self.v_sst_os      = sym_t('v_sst_os'      , vseq(1))    # shared store offset (same for A/B region)
            self.v_sld_a_os    = sym_t('v_sld_a_os'    , vseq(1))
            self.v_sld_b_os    = sym_t('v_sld_b_os'    , vseq(1))    # transposed byte offset (see get_gemm_index_for_src_matrix_transposed)
            self.v_gemm_im     = sym_t('v_gemm_im'     , vseq(1))
            self.v_gemm_in     = sym_t('v_gemm_in'     , vseq(1))
            self.v_tmp         = sym_t('v_tmp'         , vseq(4))
            # Phase 5e: v_flag is now recomputed every TAP (not a whole-kernel constant like
            # Phase 5b, since different taps probe different candidate output pixels) -- see
            # global_load_a_functor's re-zero-every-call discipline. v_n_idx/v_hi_idx/v_wi_idx
            # are this thread's GEMM_M decomposition, computed once and kept persistent (every
            # tap re-derives ho_idx/wo_idx/flag from the SAME hi_idx/wi_idx). v_gtc_tmp is
            # scratch reused fresh every tap for the divide/flag/row_idx computation (needs
            # more registers than fwd's Phase 5d: 4 chained divisions here, not 2).
            # v_flag: row_repeat_a copies (must persist across the K-loop for every row this
            # thread owns), mirroring igemm_fwd_gtc_wmma_nhwc_t's v_flag. v_n_idx/v_hi_idx/
            # v_wi_idx stay SINGLE registers even for row_repeat_a>1 -- only row 0's
            # decomposition is persisted; rows 1..row_repeat_a-1 recompute their own FRESH
            # every tap inside _emit_tap_gather, reusing v_gtc_tmp's existing scratch slots.
            self.v_flag        = sym_t('v_flag'        , vseq(outer.row_repeat_a))
            self.v_n_idx       = sym_t('v_n_idx'       , vseq(1))
            self.v_hi_idx      = sym_t('v_hi_idx'      , vseq(1))
            self.v_wi_idx      = sym_t('v_wi_idx'      , vseq(1))
            self.v_gtc_tmp     = sym_t('v_gtc_tmp'     , vseq(9))
            self.v_end         = sym_t('v_end'         , vseq())

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('v_'):
                    self._emit(v.declare())

    def get_kernel_args(self):
        kas = []
        kas.append(amdgpu_kernel_arg_t('p_in'      , 8,  0, 'global_buffer', 'f32', address_space='global', is_const='false'))
        kas.append(amdgpu_kernel_arg_t('p_wei'     , 8,  8, 'global_buffer', 'f32', address_space='global', is_const='false'))
        kas.append(amdgpu_kernel_arg_t('p_out'     , 8, 16, 'global_buffer', 'f32', address_space='global', is_const='false'))
        kas.append(amdgpu_kernel_arg_t('gemm_m'    , 4, 24, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('gemm_n'    , 4, 28, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('gemm_k'    , 4, 32, 'by_value', 'i32'))
        # Phase 5b (stride/pad): declaration order here must exactly match kernel_sgpr_t's
        # s_hi_wi..s_wo declaration order.
        kas.append(amdgpu_kernel_arg_t('hi_wi'     , 4, 36, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('wi'        , 4, 40, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('stride_h'  , 4, 44, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('stride_w'  , 4, 48, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('pad_h'     , 4, 52, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('pad_w'     , 4, 56, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('ho'        , 4, 60, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('wo'        , 4, 64, 'by_value', 'i32'))
        # Phase 5e (multi-tap + dilation)
        kas.append(amdgpu_kernel_arg_t('y'         , 4, 68, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('x'         , 4, 72, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dilation_h', 4, 76, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dilation_w', 4, 80, 'by_value', 'i32'))
        # Phase 7 (group>1): the only new kernarg needed -- see fwd's Phase 7 docstring for the
        # full rationale (A/output need the tensor's TOTAL channel count for their per-pixel
        # row stride, which requires knowing group itself; weight needs no equivalent fix).
        kas.append(amdgpu_kernel_arg_t('group'     , 4, 84, 'by_value', 'i32'))
        # Phase 48 (gemm_k_global_split): this workgroup's K-slice length. Always present in
        # the karg layout (even for non-split kernels, which never read it) so both variants
        # share one struct on the driver side -- mirrors wrw's identical field.
        # Phase 60 (Magic Division): host-precomputed magic multipliers for the
        # hi_wi/wi/stride_h/stride_w divisors. Always present so all variants share one
        # karg struct. Placed after gemm_k_per_wg to avoid shifting existing offsets.
        kas.append(amdgpu_kernel_arg_t('magic_hi_wi'   , 4, 92, 'by_value', 'u32'))
        kas.append(amdgpu_kernel_arg_t('magic_wi'      , 4, 96, 'by_value', 'u32'))
        kas.append(amdgpu_kernel_arg_t('magic_stride_h', 4, 100, 'by_value', 'u32'))
        kas.append(amdgpu_kernel_arg_t('magic_stride_w', 4, 104, 'by_value', 'u32'))
        kas.append(amdgpu_kernel_arg_t('shift_pack'    , 4, 108, 'by_value', 'u32'))
        return kas

    def get_kernel_code(self):
        # see igemm_fwd_gtc_wmma_nhwc_t's identically-named field for the rationale: the
        # LDS-reshuffle epilogue needs the whole output tile's worth of LDS, which can exceed
        # what the main loop alone reserved.
        # Phase 23: epilogue_lds_pad adds 4 padding elements per row to break a bank-conflict
        # periodicity (see coalescing_store_wmma.py) -- reflect that in the LDS size too.
        epilogue_pad = 4 if self.tunable.epilogue_lds_pad else 0
        # Phase 24: f16acc's epilogue stages genuinely 2-byte-per-element LDS data (see
        # coalescing_store_wmma.py's scatter), half the f32 case's footprint.
        epilogue_elem_bytes = 2 if (self.tunable.wmma_acc_f16 or self.tunable.wmma_acc_bf16) else 4
        # Phase 48: the atomic (gemm_k_global_split) epilogue never touches LDS at all
        # (no reshuffle needed for a scalar atomic-add-per-element store) -- mirrors wrw's
        # identical gating. Phase 53: see igemm_fwd_gtc_wmma_nhwc.py's identical comment --
        # the chunked epilogue only ever needs (wmma_tile_m*waves_per_m) rows resident at
        # once, not the full gemm_m_per_block.
        epilogue_m_rows = (self.wmma_mapping.ctrl.wave_tile_m * self.wmma_mapping.ctrl.waves_per_m()) \
            if self.tunable.wmma_epilogue_chunked else self.tunable.gemm_m_per_block
        epilogue_lds_bytes = 0 if self.tunable.gemm_k_global_split else \
            epilogue_m_rows * (self.tunable.gemm_n_per_block + epilogue_pad) * epilogue_elem_bytes
        # Phase 23: the 128x128 tile is already exactly at the 64KB/workgroup hardware limit
        # with ZERO headroom (Phase 21) -- padding pushes it over (128*132*4 = 67584 > 65536).
        # Fail loudly at codegen time, not silently at kernel-load time on real hardware.
        assert epilogue_lds_bytes <= 65536, \
            f"epilogue LDS ({epilogue_lds_bytes} bytes) exceeds the 64KB/workgroup hardware limit -- " \
            f"epilogue_lds_pad is not usable for this tile shape ({self.tunable.gemm_m_per_block}x{self.tunable.gemm_n_per_block})"
        kernel_code_dict = {
            'enable_sgpr_kernarg_segment_ptr'  :   1,
            'enable_sgpr_workgroup_id_x'       :   1,
            'enable_sgpr_workgroup_id_y'       :   1,
            'enable_vgpr_workitem_id'          :   0,
            'workgroup_group_segment_byte_size':   max(self.lds_single_size * self.lds_buffer_num, epilogue_lds_bytes),
            'kernarg_segment_byte_size'         :   112,  # Phase 60: always full struct size (C++ struct is always 112 bytes with magic fields)
            'wavefront_sgpr_count'              :   self.sgpr.s_end.value + 2 * 3,
            # Phase 54 (VGPR-MSB): see igemm_fwd_gtc_wmma_nhwc.py's identical comment --
            # bank 1 always starts at physical VGPR 256, so the wave must be granted
            # that whole span regardless of bank 0's (v_end's) actual usage.
            'workitem_vgpr_count'               :   (256 + self.tunable.num_vgpr_accumulate_c) \
                if self.tunable.wmma_acc_high_bank else self.vgpr.v_end.value,
            'wavefront_size'                    :   32,
            'cumode'                            :   0,
        }
        return amdgpu_kernel_code_t(kernel_code_dict)

    def get_kernel_info(self):
        kernel_code = self.get_kernel_code()
        kernel_args = self.get_kernel_args()
        return amdgpu_kernel_info_t(kernel_code, self.name(), self.tunable.block_size, kernel_args)

    def emit_kernel_symbol(self):
        self.sgpr.emit()
        self._emit_empty_line()
        self.vgpr.emit()
        self._emit_empty_line()

    def emit_kernel_header(self):
        kernel_name = self.name()
        self._emit('.text')
        self._emit('.globl {}'.format(kernel_name))
        self._emit('.p2align 8')
        self._emit('.type {},@function'.format(kernel_name))
        self._emit('{}:'.format(kernel_name))

    def emit_kernel_amd_kernel_code_t(self):
        amd_kernel_code_t(self.mc, self.get_kernel_info()).emit()

    def emit_kernel_end(self):
        self._emit('s_endpgm')

    def emit_kernel_footer(self):
        self._emit_empty_line()

    def _emit_lds_offset_setup(self):
        ''' See igemm_fwd_gtc_wmma_nhwc_t's identically-named method for the full
        rationale (recompute, not save/restore, to avoid needing new VGPRs -- this
        kernel is already at the hard 256-VGPR/wave limit before Phase 2). '''
        v = self.vgpr
        # ---- fixed per-thread shared-memory store offset: this thread owns byte-chunk
        # tid*bytes_per_row ---- (A: row tid's full gemm_k_per_block-element row; B:
        # (row_local,col_group)'s gemm_k_per_block-element chunk -- the linear tid-order
        # enumeration happens to coincide exactly for both, see class docstring)
        self._emit(f"v_lshlrev_b32 v[{v.v_sst_os()}], {utility_log2(self.bytes_per_row)}, v[{v.v_tid()}]   ; tid*{self.bytes_per_row} bytes")
        self._emit_empty_line()

        # ---- shared-memory load offset for A (untransposed, same as fwd) ----
        # this call also computes a "gemm_in" (column) side-output that bwd's A-only path
        # doesn't need; parked in v_gemm_im()/v_gemm_in() is unsafe (aliases the function's own
        # internal tmp2+2 scratch when waves_per_m!=1), so a genuinely dead destination is used --
        # it is fully overwritten below by get_gemm_index_for_dst_matrix before anything reads it.
        self._emit(self.wmma_mapping.get_gemm_index_for_src_matrix(v.v_tmp(3), v.v_sld_a_os(), v.v_tid(), v.v_tmp()))
        self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}], 4, v[{v.v_tid()}]")
        self._emit(f"v_and_b32 v[{v.v_tmp()}], 1, v[{v.v_tmp()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {utility_log2(self.inst_wmma_k_bytes // 2)}, v[{v.v_tmp()}]      ; k_half * {self.inst_wmma_k_bytes // 2} bytes")
        self._emit(f"v_lshlrev_b32 v[{v.v_sld_a_os()}], {utility_log2(self.bytes_per_row)}, v[{v.v_sld_a_os()}]  ; row * {self.bytes_per_row} byte row-stride")
        self._emit(f"v_add_u32 v[{v.v_sld_a_os()}], v[{v.v_tmp()}], v[{v.v_sld_a_os()}]")
        self._emit_empty_line()

        # ---- shared-memory load offset for B (TRANSPOSED -- weight is [K][N] in LDS) ----
        # row_pitch_bytes = gemm_n_per_block * databyte (128 * databyte)
        self._emit(self.wmma_mapping.get_gemm_index_for_src_matrix_transposed(v.v_sld_b_os(), v.v_tid(), v.v_tmp(),
                self.tunable.gemm_n_per_block * self.data_byte, self.data_byte, 'n'))
        self._emit_empty_line()

    def _emit_tdm_descriptor_setup_a(self):
        '''
        Phase 42: TDM descriptor for A (grad_output). A structurally IDENTICAL port of
        igemm_fwd_gtc_wmma_nhwc_t's `_emit_tdm_descriptor_setup_a` -- grad_output is an
        NHWC tensor, so a fixed (n, ho, wo) pixel's K_out channels are contiguous in
        memory, exactly the same "row = fixed spatial+batch position, contiguous channel
        run" property fwd's A relies on. tensor_dim0=gemm_k (K_out, contiguous),
        tensor_dim1=gemm_m (n*hi*wi, row axis), tensor_dim0_stride=a_k_total (bwd's name
        for the same "TOTAL per-pixel channel count" quantity fwd calls in_c_total).

        This is only correct for the 1x1/unit-stride/no-pad/no-dilation case (asserted at
        the tunable level via nxe==0) -- bwd's general per-tap gather (_emit_tap_gather)
        is genuinely harder than fwd's for multi-tap (division-based, not just a
        multiply), but for y=x=1 with pad=0/stride=1/dilation=1 it collapses to the
        trivial identity ho=hi, wo=wi, valid always -- exactly what TDM's flat,
        gather-free load assumes.
        '''
        s = self.sgpr
        data_size_code = utility_log2(self.data_byte)
        tile_dim0 = self.tunable.gemm_k_per_block
        tile_dim1 = self.tunable.gemm_m_per_block
        assert tile_dim0 < 65536 and tile_dim1 < 65536, "TDM tile_dim0/1 are 16-bit fields"

        self._emit(f"; --- Phase 42: TDM descriptor for A operand (grad_output) ---")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g0(0)}], 1   ; group0: pred=1 (valid tensor)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g0(1)}], 0   ; group0: lds_addr (A's LDS region starts at byte 0)")
        self._emit(f"; group0: global_addr = p_in + block_m_off * a_k_total * data_byte")
        self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_block_m_off()}], s[{s.s_a_k_total()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], {data_size_code}")
        self._emit(f"s_add_u32 s[{s.s_tdm_g0(2)}], s[{s.s_p_in()}], s[{s.s_tmp(0)}]")
        self._emit(f"s_addc_u32 s[{s.s_tmp(1)}], s[{s.s_p_in(1)}], 0")
        self._emit(f"s_or_b32 s[{s.s_tdm_g0(3)}], s[{s.s_tmp(1)}], 0x80000000   ; | type=2 (image) in bits[31:30]")
        self._emit_empty_line()

        self._emit(f"; group1: data_size={data_size_code}, workgroup_mask=0 (not clustered)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1(0)}], {data_size_code << 16}")
        self._emit(f"s_lshl_b32 s[{s.s_tdm_g1(1)}], s[{s.s_gemm_k()}], 16   ; tensor_dim0 (gemm_k) lo16 -> [31:16]")
        self._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_gemm_k()}], 16   ; tensor_dim0 hi16")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(1)}], s[{s.s_gemm_m()}], 16   ; tensor_dim1 (gemm_m) lo16")
        self._emit(f"s_or_b32 s[{s.s_tdm_g1(2)}], s[{s.s_tmp(0)}], s[{s.s_tmp(1)}]")
        self._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_gemm_m()}], 16   ; tensor_dim1 hi16")
        self._emit(f"s_or_b32 s[{s.s_tdm_g1(3)}], s[{s.s_tmp(0)}], {tile_dim0 << 16}   ; | tile_dim0 (compile-time)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1(4)}], {tile_dim1}   ; tile_dim1 (compile-time), tile_dim2 unused")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1(5)}], s[{s.s_a_k_total()}]   ; tensor_dim0_stride lo32 (elements)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1(6)}], 0   ; tensor_dim0_stride hi16 (assume < 2^32 elements)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1(7)}], 0   ; tensor_dim1_stride unused (2D tensor)")
        self._emit_empty_line()

    def _emit_tdm_descriptor_setup_b(self):
        '''
        Phase 42: TDM descriptor for B (weight). Weight's GLOBAL memory layout is
        [K_out][Y][X][C_in] (C_in innermost), the SAME physical tensor/layout fwd's B
        reads -- but bwd's GEMM_K is K_out (weight's ROW axis here), while fwd's GEMM_K is
        C_in (weight's contiguous axis). This means tensor_dim0/tensor_dim1's roles are
        SWAPPED relative to fwd's B descriptor:
          tensor_dim0 = gemm_n (C_in, CONTIGUOUS axis)      tile_dim0 = gemm_n_per_block
          tensor_dim1 = gemm_k (K_out, ROW axis)            tile_dim1 = gemm_k_per_block
          tensor_dim0_stride = wei_row_c (= x*y*gemm_n, the per-K_out-row element count --
              bwd's existing name for the same quantity fwd calls wei_k_stride)
        Confirmed dimensionally consistent with bwd's own EXISTING (already-validated)
        non-TDM code: `move_slice_window_b_functor` advances v_addr_b by `s_wei_k_stride`
        (= wei_row_c*databyte*gemm_k_per_block) once per K-tile, i.e. by
        gemm_k_per_block ROWS, not gemm_k_per_block contiguous elements -- exactly the
        "K_out is the row axis" structure this descriptor encodes. Because GEMM_K is now
        tensor_dim1 (not tensor_dim0 as in fwd's B), the K-tail-via-hardware-OOB rebuild
        (see move_slice_window_b_functor's tdm_global_load branch) rebuilds tensor_dim1
        every iteration, not tensor_dim0 -- the one genuine structural difference from
        fwd's B-TDM code, forced by this axis swap, not an arbitrary design choice.

        global_addr's tile-level base (row=0, i.e. this block's own K-tile start, which
        move_slice_window_b advances) is `p_wei + block_n_off*databyte` -- block_n_off is
        a column (contiguous-axis) offset here, matching bwd's own existing
        `v_addr_b_base` formula's `+ block_n_off` term (added directly, not multiplied by
        wei_row_c, unlike the `row_local*wei_row_c` term in that same formula).
        '''
        s = self.sgpr
        data_size_code = utility_log2(self.data_byte)
        tile_dim0 = self.tunable.gemm_n_per_block
        tile_dim1 = self.tunable.gemm_k_per_block
        assert tile_dim0 < 65536 and tile_dim1 < 65536, "TDM tile_dim0/1 are 16-bit fields"

        self._emit(f"; --- Phase 42: TDM descriptor for B operand (weight) ---")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g0_b(0)}], 1   ; group0: pred=1 (valid tensor)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g0_b(1)}], {self.lds_a_size}   ; group0: lds_addr (B's LDS region starts after A's)")
        self._emit(f"; group0: global_addr = p_wei + block_n_off * data_byte (block_n_off is a column/contiguous-axis offset)")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(0)}], s[{s.s_block_n_off()}], {data_size_code}")
        self._emit(f"s_add_u32 s[{s.s_tdm_g0_b(2)}], s[{s.s_p_wei()}], s[{s.s_tmp(0)}]")
        self._emit(f"s_addc_u32 s[{s.s_tmp(1)}], s[{s.s_p_wei(1)}], 0")
        self._emit(f"s_or_b32 s[{s.s_tdm_g0_b(3)}], s[{s.s_tmp(1)}], 0x80000000   ; | type=2 (image) in bits[31:30]")
        self._emit_empty_line()

        self._emit(f"; group1: data_size={data_size_code}, workgroup_mask=0 (not clustered)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1_b(0)}], {data_size_code << 16}")
        self._emit(f"s_lshl_b32 s[{s.s_tdm_g1_b(1)}], s[{s.s_gemm_n()}], 16   ; tensor_dim0 (gemm_n) lo16 -> [31:16]")
        self._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_gemm_n()}], 16   ; tensor_dim0 hi16")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(1)}], s[{s.s_gemm_k()}], 16   ; tensor_dim1 (gemm_k) lo16")
        self._emit(f"s_or_b32 s[{s.s_tdm_g1_b(2)}], s[{s.s_tmp(0)}], s[{s.s_tmp(1)}]")
        self._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_gemm_k()}], 16   ; tensor_dim1 hi16")
        self._emit(f"s_or_b32 s[{s.s_tdm_g1_b(3)}], s[{s.s_tmp(0)}], {tile_dim0 << 16}   ; | tile_dim0 (compile-time)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1_b(4)}], {tile_dim1}   ; tile_dim1 (compile-time), tile_dim2 unused")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1_b(5)}], s[{s.s_wei_row_c()}]   ; tensor_dim0_stride lo32 (elements)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1_b(6)}], 0   ; tensor_dim0_stride hi16 (assume < 2^32 elements)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1_b(7)}], 0   ; tensor_dim1_stride unused (2D tensor)")
        self._emit_empty_line()

    def _emit_wave0_only(self, body_fn):
        '''
        Phase 42: direct port of igemm_fwd_gtc_wmma_nhwc_t's identical Phase 29 helper --
        see there for the full rationale (TDM ignores EXEC entirely, so only a genuine
        scalar branch, not EXEC-masking, can suppress a redundant per-wave TDM issue).
        '''
        s = self.sgpr
        self._tdm_label_counter += 1
        label = f"L_{self.name()}_wave0_only_{self._tdm_label_counter}"
        self._emit(f"s_cmp_eq_u32 s[{s.s_wave_id()}], 0")
        self._emit(f"s_cbranch_scc0 {label}")
        body_fn()
        self._emit_front(f"{label}:")

    def emit_kernel_prologue(self):
        s = self.sgpr
        v = self.vgpr
        if self.vgpr_msb_tracker is not None:
            # Phase 54: see igemm_fwd_gtc_wmma_nhwc.py's identical comment -- establish
            # a known VGPR-MSB state before the first VGPR-writing instruction.
            msb_line = self.vgpr_msb_tracker.ensure(dst=0, src0=0, src1=0, src2=0)
            if msb_line:
                self._emit(msb_line)
        self._emit(f"s_load_dwordx4 s[{s.s_p_in()}:{s.s_p_in(3)}], s[{s.s_ka()}:{s.s_ka(1)}], 0")
        self._emit(f"s_load_dwordx4 s[{s.s_p_out()}:{s.s_p_out(3)}], s[{s.s_ka()}:{s.s_ka(1)}], 16")
        self._emit(f"s_load_dword s[{s.s_gemm_k()}], s[{s.s_ka()}:{s.s_ka(1)}], 32")
        # individual s_load_dword (not dwordx4) -- these SGPRs aren't guaranteed 4-aligned
        self._emit(f"s_load_dword s[{s.s_hi_wi()}], s[{s.s_ka()}:{s.s_ka(1)}], 36")
        self._emit(f"s_load_dword s[{s.s_wi()}], s[{s.s_ka()}:{s.s_ka(1)}], 40")
        self._emit(f"s_load_dword s[{s.s_stride_h()}], s[{s.s_ka()}:{s.s_ka(1)}], 44")
        self._emit(f"s_load_dword s[{s.s_stride_w()}], s[{s.s_ka()}:{s.s_ka(1)}], 48")
        self._emit(f"s_load_dword s[{s.s_pad_h()}], s[{s.s_ka()}:{s.s_ka(1)}], 52")
        self._emit(f"s_load_dword s[{s.s_pad_w()}], s[{s.s_ka()}:{s.s_ka(1)}], 56")
        self._emit(f"s_load_dword s[{s.s_ho()}], s[{s.s_ka()}:{s.s_ka(1)}], 60")
        self._emit(f"s_load_dword s[{s.s_wo()}], s[{s.s_ka()}:{s.s_ka(1)}], 64")
        self._emit(f"s_load_dword s[{s.s_y()}], s[{s.s_ka()}:{s.s_ka(1)}], 68")
        self._emit(f"s_load_dword s[{s.s_x()}], s[{s.s_ka()}:{s.s_ka(1)}], 72")
        self._emit(f"s_load_dword s[{s.s_dilation_h()}], s[{s.s_ka()}:{s.s_ka(1)}], 76")
        self._emit(f"s_load_dword s[{s.s_dilation_w()}], s[{s.s_ka()}:{s.s_ka(1)}], 80")
        self._emit(f"s_load_dword s[{s.s_group()}], s[{s.s_ka()}:{s.s_ka(1)}], 84")
        if self.tunable.gemm_k_global_split:
            self._emit(f"s_load_dword s[{s.s_gemm_k_per_wg()}], s[{s.s_ka()}:{s.s_ka(1)}], 88")
        # Phase 60 (Magic Division): load magic multipliers + packed shift from kernargs
        self._emit(f"s_load_dwordx4 s[{s.s_magic_hi_wi()}:{s.s_magic_hi_wi(3)}], s[{s.s_ka()}:{s.s_ka(1)}], 92")
        self._emit(f"s_load_dword s[{s.s_shift_pack()}], s[{s.s_ka()}:{s.s_ka(1)}], 108")
        self._emit(f"v_mov_b32 v[{v.v_tid()}], v0")
        if self.tunable.tdm_global_load:
            # Phase 42: mirrors igemm_fwd_gtc_wmma_nhwc_t's identical Phase 29 computation --
            # this wave's index within the workgroup, derived while EXEC is still fully
            # enabled (kernel entry, before any lane-disabling branch).
            self._emit(f"v_readfirstlane_b32 s[{s.s_wave_id()}], v[{v.v_tid()}]   ; Phase 29/42: lane 0's flat tid = this wave's base")
            self._emit(f"s_lshr_b32 s[{s.s_wave_id()}], s[{s.s_wave_id()}], 5   ; wave index within workgroup")
        # gfx1250 delivers workgroup id via ttmp9/ttmp7 -- see docs/gfx1250_wmma_layout.md.
        # ttmp9 is a clean, unpacked blockIdx.x. ttmp7 PACKS blockIdx.y (low 16 bits) and
        # blockIdx.z (high 16 bits) -- Phase 48 mirrors wrw's identical gemm_k_global_split
        # decode (a plain `s_mov_b32 s_by, ttmp7` is only correct when grid.z is always 1).
        self._emit(f"s_mov_b32 s[{s.s_bx()}], ttmp9")
        if self.tunable.gemm_k_global_split:
            self._emit(f"s_and_b32 s[{s.s_by()}], ttmp7, 0xffff")
            self._emit(f"s_lshr_b32 s[{s.s_bz()}], ttmp7, 16")
        else:
            self._emit(f"s_mov_b32 s[{s.s_by()}], ttmp7")
        self._emit(f"s_wait_kmcnt 0x0")
        # Phase 60 (Magic Division): unpack the per-divisor shifts from the packed shift
        # word, now that s_wait_kmcnt has guaranteed s_shift_pack's s_load_dword has landed.
        # shift_pack layout: [7:0]=hi_wi, [15:8]=wi, [23:16]=stride_h, [31:24]=stride_w
        self._emit(f"s_and_b32 s[{s.s_shift_hi_wi()}], s[{s.s_shift_pack()}], 0xff")
        self._emit(f"s_lshr_b32 s[{s.s_shift_wi()}], s[{s.s_shift_pack()}], 8")
        self._emit(f"s_lshr_b32 s[{s.s_shift_stride_h()}], s[{s.s_shift_pack()}], 16")
        self._emit(f"s_lshr_b32 s[{s.s_shift_stride_w()}], s[{s.s_shift_pack()}], 24")
        if self.tunable.gemm_k_global_split:
            self._emit(f"s_mul_i32 s[{s.s_gemm_k_wg_off()}], s[{s.s_bz()}], s[{s.s_gemm_k_per_wg()}]   ; this workgroup's K-slice base")
        self._emit_empty_line()

        m_int_div_rem_vs = macro_int_div_rem_vs_gfx1250_t(self.mc)

        # ---- group>1: decode group_idx out of s_by and correct s_by in place BEFORE
        # s_block_n_off is computed from it below -- see igemm_fwd_gtc_wmma_nhwc_t's Phase 7
        # docstring for the full rationale (grid_y folding, division-macro reuse). A
        # (grad_output) and the output (grad_input) are NHWC tensors with group INTERLEAVED
        # within each pixel's channel dimension, so their per-pixel row stride must be the
        # TOTAL channel count (gemm_k*group / gemm_n*group), not the per-group gemm_k/gemm_n
        # used for the K-reduction size and base-pointer offset. Weight needs no equivalent
        # fix (its group split is the outermost/block-contiguous dimension). ----
        self._emit(f"; --- group>1: decode group_idx, correct s_by, offset A/output base pointers ---")
        self._emit(f"s_add_u32 s[{s.s_tmp(0)}], s[{s.s_gemm_n()}], {self.tunable.gemm_n_per_block - 1}")
        self._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], {utility_log2(self.tunable.gemm_n_per_block)}   ; blocks_per_group_n = ceil(gemm_n/gemm_n_per_block)")
        self._emit(f"v_mov_b32 v[{v.v_gtc_tmp(0)}], s[{s.s_by()}]")
        self._emit(m_int_div_rem_vs(v.v_gtc_tmp(1), v.v_gtc_tmp(2), v.v_gtc_tmp(0), s.s_tmp(0), v.v_tmp(), s.s_tmp(1)))
        self._emit(f"v_readfirstlane_b32 s[{s.s_group_idx()}], v[{v.v_gtc_tmp(2)}]   ; group_idx")
        self._emit(f"v_readfirstlane_b32 s[{s.s_by()}], v[{v.v_gtc_tmp(1)}]   ; s_by <- corrected within-group N-block index")
        self._emit_empty_line()

        self._emit(f"s_mul_i32 s[{s.s_a_k_total()}], s[{s.s_gemm_k()}], s[{s.s_group()}]")
        self._emit(f"s_mul_i32 s[{s.s_out_c_total()}], s[{s.s_gemm_n()}], s[{s.s_group()}]")
        self._emit_empty_line()

        self._emit(f"; A (grad_output): group offset = group_idx * gemm_k elements")
        self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_group_idx()}], s[{s.s_gemm_k()}]")
        if self.tunable.gemm_k_global_split:
            # Phase 48: A's GEMM_K is its own CONTIGUOUS axis (unlike wrw's A, where GEMM_K
            # is the row axis needing a stride-multiply) -- this shard's K-slice base is a
            # flat element add, exactly like the group offset immediately above (both
            # advance along the same contiguous axis).
            self._emit(f"s_add_u32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], s[{s.s_gemm_k_wg_off()}]   ; += this workgroup's K-slice base")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], {utility_log2(self.data_byte)}")
        self._emit(f"s_add_u32 s[{s.s_p_in()}], s[{s.s_p_in()}], s[{s.s_tmp(0)}]")
        self._emit(f"s_addc_u32 s[{s.s_p_in(1)}], s[{s.s_p_in(1)}], 0")

        # Phase 24: shift must follow the D-operand's real width (4 bytes normally, 2 under
        # wmma_acc_f16) -- see the identical bug found and fixed in wrw's per-tap output
        # offset (igemm_wrw_gtc_wmma_nhwc.py's emit_kernel_tap_loop).
        out_elem_byte_shift = 1 if (self.tunable.wmma_acc_f16 or self.tunable.wmma_acc_bf16) else 2
        self._emit(f"; output (grad_input): group offset = group_idx * gemm_n elements (D-operand is fp32/int32 (4B) normally, fp16 (2B) under wmma_acc_f16)")
        self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_group_idx()}], s[{s.s_gemm_n()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], {out_elem_byte_shift}")
        self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp(0)}]")
        self._emit(f"s_addc_u32 s[{s.s_p_out(1)}], s[{s.s_p_out(1)}], 0")
        self._emit_empty_line()

        # zero the accumulator -- v_wmma_* does D = A@B + C, and v_c is used as both C and D.
        self._emit(f"; clear accumulator")
        if self.vgpr_msb_tracker is not None:
            msb_line = self.vgpr_msb_tracker.ensure(dst=1)
            if msb_line:
                self._emit(msb_line)
        for i in range(self.tunable.num_vgpr_accumulate_c):
            self._emit(f"v_mov_b32 v[{v.v_c(i)}], 0")
        if self.vgpr_msb_tracker is not None:
            # Phase 54 bugfix: see igemm_fwd_gtc_wmma_nhwc.py's identical comment --
            # reset dst back to bank 0 immediately, before the rest of the prologue's
            # ordinary bank-0 VGPR writes.
            msb_line = self.vgpr_msb_tracker.ensure(dst=0, src0=0, src1=0, src2=0)
            if msb_line:
                self._emit(msb_line)
        self._emit_empty_line()

        if self.tunable.async_global_load:
            self._emit(f"; Phase 13: persistent zero quad, used to zero-fill padding lanes' LDS")
            self._emit(f"; destinations after a masked global_load_async_to_lds_b128 (see global_load_a_functor)")
            for i in range(4):
                self._emit(f"v_mov_b32 v[{v.v_zero(i)}], 0")
            self._emit_empty_line()

        self._emit(f"s_lshl_b32 s[{s.s_block_m_off()}], s[{s.s_bx()}], {utility_log2(self.tunable.gemm_m_per_block)}   ; *gemm_m_per_block")
        self._emit(f"s_lshl_b32 s[{s.s_block_n_off()}], s[{s.s_by()}], {utility_log2(self.tunable.gemm_n_per_block)}   ; *gemm_n_per_block")
        if self.tunable.gemm_k_global_split:
            self._emit(f"s_mov_b32 s[{s.s_knum()}], s[{s.s_gemm_k_per_wg()}]   ; this workgroup only reduces its own K-slice")
        else:
            self._emit(f"s_mov_b32 s[{s.s_knum()}], s[{s.s_gemm_k()}]")
        if self.tunable.tdm_global_load:
            # Phase 42: mirrors fwd's identical Phase 31 init -- starts equal to gemm_k
            # (matching what _emit_tdm_descriptor_setup_a's tensor_dim0 is set to directly).
            self._emit(f"s_mov_b32 s[{s.s_tdm_k_remain()}], s[{s.s_gemm_k()}]")
            self._emit_tdm_descriptor_setup_a()
        # weight's per-K_out-row element count is y*x*c (not just c as in the 1x1 case) --
        # computed once, used both for the fixed row_local/col_group base below and for
        # move_slice_window_b's per-K-iteration stride.
        self._emit(f"s_mul_i32 s[{s.s_wei_row_c()}], s[{s.s_x()}], s[{s.s_y()}]")
        self._emit(f"s_mul_i32 s[{s.s_wei_row_c()}], s[{s.s_wei_row_c()}], s[{s.s_gemm_n()}]")
        # NOTE: gemm_k_per_block is 32 for fp16/bf16 but 64 for int8 -- must NOT hardcode a
        # "+5" (*32) shift here, or B's per-K-block stride silently undercounts by 2x for int8.
        self._emit(f"s_lshl_b32 s[{s.s_wei_k_stride()}], s[{s.s_wei_row_c()}], {utility_log2(self.data_byte * self.tunable.gemm_k_per_block)}   ; wei_row_c * databyte * {self.tunable.gemm_k_per_block}")
        self._emit_empty_line()

        # Phase 47: fixed a copy-paste-from-fwd bug -- weight's group split is along
        # bwd's own GEMM_K (K_out per group, the row-count dimension for the weight
        # tensor's [G][K_per_group][Y][X][C_per_group] layout), NOT gemm_n (C_in per
        # group). fwd's identical-looking code correctly uses gemm_n there because
        # fwd's own GEMM_N happens to equal k/group -- bwd's GEMM roles are swapped
        # (gemm_k = k/group here), so blindly reusing "gemm_n" silently scaled the
        # group offset by the wrong (and differently-valued) per-group count,
        # producing valid:n for any group>1 shape where gemm_n != gemm_k.
        self._emit(f"; group>1: B (weight) group offset = group_idx * gemm_k * wei_row_c elements")
        self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_group_idx()}], s[{s.s_gemm_k()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], s[{s.s_wei_row_c()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], {utility_log2(self.data_byte)}")
        self._emit(f"s_add_u32 s[{s.s_p_wei()}], s[{s.s_p_wei()}], s[{s.s_tmp(0)}]")
        self._emit(f"s_addc_u32 s[{s.s_p_wei(1)}], s[{s.s_p_wei(1)}], 0")
        self._emit_empty_line()

        if self.tunable.tdm_global_load:
            self._emit_tdm_descriptor_setup_b()

        # ---- one-time decomposition of this thread's GEMM_M index into (n_idx, hi_idx,
        # wi_idx), kept persistent: every tap re-derives ho_idx/wo_idx/flag from these ----
        self._emit(f"s_mul_i32 s[{s.s_ho_wo()}], s[{s.s_ho()}], s[{s.s_wo()}]")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], s[{s.s_block_m_off()}], v[{v.v_tid()}]   ; m_idx")
        # Phase 60 (Magic Division): replace ~24-instruction emulated divide with
        # 5-instruction magic multiply+shift for hi_wi and wi divisors
        m_mdiv_rem_vs = macro_mdiv_u32_rem_vs_t(self.mc)
        self._emit(m_mdiv_rem_vs(v.v_gtc_tmp(1), v.v_n_idx(), v.v_gtc_tmp(0), s.s_magic_hi_wi(), s.s_shift_hi_wi(), s.s_hi_wi(), v.v_tmp()))
        self._emit(f"; v_gtc_tmp(1)=hw_idx (rem), v_n_idx=n_idx (quo)")
        self._emit(m_mdiv_rem_vs(v.v_wi_idx(), v.v_hi_idx(), v.v_gtc_tmp(1), s.s_magic_wi(), s.s_shift_wi(), s.s_wi(), v.v_tmp()))
        self._emit(f"; v_wi_idx=wi_idx (rem), v_hi_idx=hi_idx (quo)")
        self._emit_empty_line()

        # ---- B's fixed per-thread row base (this tap's column offset is added fresh every
        # tap in _emit_tap_gather) -- thread tid owns row_local (one of gemm_k_per_block K_out
        # rows WITHIN this K-block) and col_group (one of num_col_groups chunks of the
        # 128-wide C_in tile, each chunk exactly gemm_k_per_block elements wide -- since the
        # global load is always a fixed 64 bytes/thread, and gemm_k_per_block*databyte==64
        # always, the chunk width in ELEMENTS equals gemm_k_per_block for every precision: 32
        # for fp16/bf16 (4 col_groups), 64 for int8 (2 col_groups) -- NOT the hardcoded
        # tid>>2/tid&3/col_group*32 an fp16-only version of this code once had) -- see class
        # docstring / wmma_mapping.py ----
        num_col_groups = self.tunable.gemm_n_per_block // self.tunable.gemm_k_per_block
        col_group_bits = utility_log2(num_col_groups)
        col_start_shift = utility_log2(self.tunable.gemm_k_per_block)
        self._emit(f"; v_addr_b_base = p_wei + (row_local*wei_row_c + block_n_off + col_start) * databyte")
        self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}], {col_group_bits}, v[{v.v_tid()}]        ; row_local = tid>>{col_group_bits}")
        if self.tunable.wmma_k_tail:
            # K-tail: persist row_local BEFORE the very next line multiplies it by
            # wei_row_c in place -- see kernel_vgpr_t's v_b_row_local docstring.
            self._emit(f"v_mov_b32 v[{v.v_b_row_local()}], v[{v.v_tmp()}]   ; K-tail: persist row_local")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wei_row_c()}], v[{v.v_tmp()}]  ; row_local * wei_row_c")
        self._emit(f"v_and_b32 v[{v.v_tmp(1)}], {num_col_groups - 1}, v[{v.v_tid()}]           ; col_group = tid&{num_col_groups - 1}")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp(1)}], {col_start_shift}, v[{v.v_tmp(1)}]      ; col_start = col_group*{self.tunable.gemm_k_per_block}")
        if self.tunable.wmma_n_tail:
            # N-tail: persist col_start_abs (= block_n_off + col_start) BEFORE the very next
            # line consumes v_tmp(1) by merging it into v_tmp(0) -- see kernel_vgpr_t's
            # v_b_col_start_abs docstring. Kernel-lifetime constant (col_start doesn't change
            # across the K-loop or taps), so n_valid_base can be derived once, here too.
            self._emit(f"v_add_u32 v[{v.v_b_col_start_abs()}], s[{s.s_block_n_off()}], v[{v.v_tmp(1)}]   ; N-tail: col_start_abs")
            self._emit(f"v_sub_u32 v[{v.v_n_valid_base()}], s[{s.s_gemm_n()}], v[{v.v_b_col_start_abs()}]   ; N-tail: n_valid_base = gemm_n - col_start_abs")
        self._emit(f"v_add_u32 v[{v.v_tmp()}], v[{v.v_tmp(1)}], v[{v.v_tmp()}]")
        self._emit(f"v_add_u32 v[{v.v_tmp()}], s[{s.s_block_n_off()}], v[{v.v_tmp()}]")
        if self.tunable.gemm_k_global_split:
            # Phase 48: B's GEMM_K is its own ROW axis (structurally identical to wrw's A) --
            # this shard's K-slice base, in B-row units, is s_gemm_k_wg_off*wei_row_c
            # elements, added into the same flat row-index accumulator every other per-row
            # term (row_local*wei_row_c, col_start, block_n_off) already feeds.
            self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_gemm_k_wg_off()}], s[{s.s_wei_row_c()}]   ; this workgroup's K-slice base, in B row units")
            self._emit(f"v_add_u32 v[{v.v_tmp()}], s[{s.s_tmp(2)}], v[{v.v_tmp()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {utility_log2(self.data_byte)}, v[{v.v_tmp()}]   ; * databyte (elements -> bytes)")
        if self.tunable.saddr_global_load:
            # Phase 61: byte OFFSET only -- s_p_wei is passed separately as SADDR
            self._emit(f"v_mov_b32 v[{v.v_off_b_base()}], v[{v.v_tmp()}]   ; v_off_b_base = (row_local*wei_row_c + block_n_off + col_start) * databyte")
        else:
            self._emit(f"v_mov_b32 v[{v.v_addr_b_base(1)}], s[{s.s_p_wei(1)}]")
            self._emit(f"v_add_co_u32 v[{v.v_addr_b_base()}], vcc_lo, s[{s.s_p_wei()}], v[{v.v_tmp()}]")
            self._emit(f"v_add_co_ci_u32 v[{v.v_addr_b_base(1)}], vcc_lo, 0, v[{v.v_addr_b_base(1)}], vcc_lo")
        self._emit_empty_line()

        self._emit_lds_offset_setup()

        # ---- persistent (im, in) for the epilogue, converted to global output indices ----
        self._emit(self.wmma_mapping.get_gemm_index_for_dst_matrix(v.v_gemm_in(), v.v_gemm_im(), v.v_tid(), v.v_tmp()))
        self._emit(f"v_add_u32 v[{v.v_gemm_im()}], s[{s.s_block_m_off()}], v[{v.v_gemm_im()}]")
        self._emit(f"v_add_u32 v[{v.v_gemm_in()}], s[{s.s_block_n_off()}], v[{v.v_gemm_in()}]")
        self._emit_empty_line()

        # ---- Phase 5e: runtime tap-loop counter, initialized once before the loop ----
        self._emit(f"s_mov_b32 s[{s.s_iy()}], 0")
        self._emit_empty_line()

    def _emit_tap_gather(self):
        '''
        Recomputes this tap's A operand (grad_output) address+flag and B operand (weight)
        address, using the current s_iy/s_ix (runtime tap-loop counters) plus the persistent
        v_n_idx/v_hi_idx/v_wi_idx (this thread's GEMM_M decomposition, computed once in
        emit_kernel_prologue). See class docstring for the harder stride-gap divide this
        kernel needs (vs fwd's simpler bounds-only check).

        A's computation is looped row_repeat_a times (once per row this thread owns -- see
        __init__'s row_repeat_a docstring), mirroring igemm_fwd_gtc_wmma_nhwc_t's identical
        row-loop structure. Row 0 uses the persistent v_hi_idx/v_wi_idx/v_n_idx exactly as
        today. Rows 1..row_repeat_a-1 recompute (n_idx, hi_idx, wi_idx) FRESH from
        v_tid+i*block_size, using v_gtc_tmp(0)/(1)/(2) as scratch (division scratch s_tmp(2),
        free during this loop -- B's per-tap address computation after the loop is the only
        other user, matching fwd's identical reasoning). The `pad_h - iy*dilation_h` /
        `pad_w - ix*dilation_w` scalar values are recomputed fresh into s_tmp(0)/s_tmp(1) at
        the TOP of every row's iteration (not hoisted once above the loop) -- row 0's own
        numerator-division call uses s_tmp(0) as ITS scratch (see below), which would
        otherwise silently corrupt the shared pad value before row 1 gets a chance to read
        it (the exact bug igemm_fwd_gtc_wmma_nhwc_t's asymmetric-shape work found the hard
        way, see docs/gfx1250_wmma_layout.md's Phase 11 section) -- recomputing fresh per row
        sidesteps this entirely instead of hunting for a distinct scratch register.
        row_repeat_a==1 for every existing config, so this loop runs once with i=0 and takes
        the row-0 branch, byte-identical.
        '''
        s = self.sgpr
        v = self.vgpr
        m_mdiv_rem_vs = macro_mdiv_u32_rem_vs_t(self.mc)

        for i in range(self.row_repeat_a):
            tag = '' if i == 0 else f'({i})'
            self._emit(f"; --- per-tap gather (row{tag}): numerator_h = hi_idx + pad_h - iy*dilation_h ---")
            self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_iy()}], s[{s.s_dilation_h()}]")
            self._emit(f"s_sub_i32 s[{s.s_tmp(0)}], s[{s.s_pad_h()}], s[{s.s_tmp(0)}]   ; pad_h - iy*dilation_h")
            self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_ix()}], s[{s.s_dilation_w()}]")
            self._emit(f"s_sub_i32 s[{s.s_tmp(1)}], s[{s.s_pad_w()}], s[{s.s_tmp(1)}]   ; pad_w - ix*dilation_w")

            if i == 0:
                hi_src, wi_src, n_src = v.v_hi_idx(), v.v_wi_idx(), v.v_n_idx()
            else:
                self._emit(f"v_add_u32 v[{v.v_gtc_tmp(4)}], {i * self.tunable.block_size}, v[{v.v_tid()}]")
                self._emit(f"v_add_u32 v[{v.v_gtc_tmp(4)}], s[{s.s_block_m_off()}], v[{v.v_gtc_tmp(4)}]   ; m_idx (row {i})")
                # Phase 60 (Magic Division): magic multiply+shift for hi_wi and wi divisors
                self._emit(m_mdiv_rem_vs(v.v_gtc_tmp(3), v.v_gtc_tmp(2), v.v_gtc_tmp(4), s.s_magic_hi_wi(), s.s_shift_hi_wi(), s.s_hi_wi(), v.v_tmp()))
                self._emit(f"; v_gtc_tmp(0)=hi_idx({i}), v_gtc_tmp(1)=wi_idx({i}), v_gtc_tmp(2)=n_idx({i})")
                hi_src, wi_src, n_src = v.v_gtc_tmp(0), v.v_gtc_tmp(1), v.v_gtc_tmp(2)
            self._emit(f"v_add_u32 v[{v.v_gtc_tmp(4)}], v[{hi_src}], s[{s.s_tmp(0)}]   ; numerator_h{tag}")
            self._emit(f"v_add_u32 v[{v.v_gtc_tmp(3)}], v[{wi_src}], s[{s.s_tmp(1)}]   ; numerator_w{tag}")
            self._emit_empty_line()

            self._emit(f"; ho_idx{tag} = numerator_h/stride_h (valid iff exact division & in bounds --")
            self._emit(f"; a negative numerator wraps to a huge u32, which the bounds check below")
            self._emit(f"; naturally rejects via quotient overflow, no separate sign check needed)")
            # Phase 60 (Magic Division): magic multiply+shift for stride_h and stride_w divisors
            self._emit(m_mdiv_rem_vs(v.v_gtc_tmp(5), v.v_gtc_tmp(6), v.v_gtc_tmp(4), s.s_magic_stride_h(), s.s_shift_stride_h(), s.s_stride_h(), v.v_tmp()))
            self._emit(f"; v_gtc_tmp(5)=rem_h, v_gtc_tmp(6)=ho_idx{tag}")
            self._emit(m_mdiv_rem_vs(v.v_gtc_tmp(7), v.v_gtc_tmp(8), v.v_gtc_tmp(3), s.s_magic_stride_w(), s.s_shift_stride_w(), s.s_stride_w(), v.v_tmp()))
            self._emit(f"; v_gtc_tmp(7)=rem_w, v_gtc_tmp(8)=wo_idx{tag}")
            self._emit_empty_line()

            self._emit(f"; v_flag{tag} = 1 iff both divisions are exact AND (ho_idx,wo_idx) in bounds")
            self._emit(f"v_cmp_eq_u32 vcc_lo, 0, v[{v.v_gtc_tmp(5)}]")
            self._emit(f"v_cndmask_b32 v[{v.v_flag(i)}], 0, 1, vcc_lo")
            self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_ho()}], v[{v.v_gtc_tmp(6)}]")
            self._emit(f"v_cndmask_b32 v[{v.v_flag(i)}], 0, v[{v.v_flag(i)}], vcc_lo")
            self._emit(f"v_cmp_eq_u32 vcc_lo, 0, v[{v.v_gtc_tmp(7)}]")
            self._emit(f"v_cndmask_b32 v[{v.v_flag(i)}], 0, v[{v.v_flag(i)}], vcc_lo")
            self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_wo()}], v[{v.v_gtc_tmp(8)}]")
            self._emit(f"v_cndmask_b32 v[{v.v_flag(i)}], 0, v[{v.v_flag(i)}], vcc_lo")
            self._emit_empty_line()

            if self.tunable.wmma_m_tail:
                # Phase 26a: v_flag also gates on this row's absolute GEMM_M index, mirroring
                # igemm_fwd_gtc_wmma_nhwc_t's identical Phase 25 addition. v_gtc_tmp(4) is free
                # here (its i>0 pre-division value, if any, is not trusted to have survived the
                # division macro calls above -- recomputed fresh, same discipline as Phase 25).
                self._emit(f"; wmma_m_tail: v_flag{tag} &= (this row's absolute GEMM_M index < real gemm_m)")
                if i == 0:
                    self._emit(f"v_add_u32 v[{v.v_gtc_tmp(4)}], s[{s.s_block_m_off()}], v[{v.v_tid()}]")
                else:
                    self._emit(f"v_add_u32 v[{v.v_gtc_tmp(4)}], {i * self.tunable.block_size}, v[{v.v_tid()}]")
                    self._emit(f"v_add_u32 v[{v.v_gtc_tmp(4)}], s[{s.s_block_m_off()}], v[{v.v_gtc_tmp(4)}]")
                self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_gemm_m()}], v[{v.v_gtc_tmp(4)}]")
                self._emit(f"v_cndmask_b32 v[{v.v_flag(i)}], 0, v[{v.v_flag(i)}], vcc_lo")
                self._emit_empty_line()

            self._emit(f"; row_idx{tag} = n_idx*(ho*wo) + ho_idx*wo + wo_idx (meaningless but harmless if")
            self._emit(f"; v_flag==0 -- that lane's global_load_a is EXEC-masked off, see global_load_a_functor)")
            self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(0)}], s[{s.s_ho_wo()}], v[{n_src}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(2)}], s[{s.s_wo()}], v[{v.v_gtc_tmp(6)}]")
            self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(2)}]")
            self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(8)}]")
            self._emit_empty_line()

            if self.tunable.async_global_load or self.tunable.saddr_global_load:
                self._emit(f"; v_off_a = row_idx * a_k_total * databyte (Phase 13/61: byte OFFSET only --")
                self._emit(f"; s_p_in is passed separately as SADDR; a_k_total = gemm_k*group, see class docstring)")
                self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(0)}], s[{s.s_a_k_total()}], v[{v.v_gtc_tmp(0)}]")
                self._emit(f"v_lshlrev_b32 v[{v.v_off_a()}], {utility_log2(self.data_byte)}, v[{v.v_gtc_tmp(0)}]")
                self._emit_empty_line()
            else:
                self._emit(f"; v_addr_a{tag} = p_in + row_idx * a_k_total * databyte (a_k_total = gemm_k*group --")
                self._emit(f"; grad_output's pixel-to-pixel stride is its TOTAL K_out count, not the per-group gemm_k, see class docstring)")
                self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(0)}], s[{s.s_a_k_total()}], v[{v.v_gtc_tmp(0)}]")
                self._emit(f"v_lshlrev_b32 v[{v.v_gtc_tmp(0)}], {utility_log2(self.data_byte)}, v[{v.v_gtc_tmp(0)}]")
                self._emit(f"v_mov_b32 v[{v.v_addr_a(i*2+1)}], s[{s.s_p_in(1)}]   ; reset high half fresh -- this")
                self._emit(f"                                                ; tap's address is NOT a continuation of the previous tap's")
                self._emit(f"v_add_co_u32 v[{v.v_addr_a(i*2)}], vcc_lo, s[{s.s_p_in()}], v[{v.v_gtc_tmp(0)}]")
                self._emit(f"v_add_co_ci_u32 v[{v.v_addr_a(i*2+1)}], vcc_lo, 0, v[{v.v_addr_a(i*2+1)}], vcc_lo")
                self._emit_empty_line()

        self._emit(f"; --- per-tap B address: v_addr_b = v_addr_b_base + (iy*x+ix)*gemm_n*{self.data_byte} bytes ---")
        self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_iy()}], s[{s.s_x()}]")
        self._emit(f"s_add_i32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], s[{s.s_ix()}]   ; tap linear index")
        self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], s[{s.s_gemm_n()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], {utility_log2(self.data_byte)}   ; tap byte offset")
        if self.tunable.saddr_global_load:
            self._emit(f"v_add_u32 v[{v.v_off_b()}], s[{s.s_tmp(2)}], v[{v.v_off_b_base()}]   ; Phase 61: byte OFFSET only")
        else:
            self._emit(f"v_add_co_u32 v[{v.v_addr_b()}], vcc_lo, s[{s.s_tmp(2)}], v[{v.v_addr_b_base()}]")
            self._emit(f"v_add_co_ci_u32 v[{v.v_addr_b(1)}], vcc_lo, 0, v[{v.v_addr_b_base(1)}], vcc_lo")
        self._emit_empty_line()

    def emit_kernel_tap_loop(self):
        '''
        Runtime (not compile-time-unrolled) outer loop over the y*x filter taps, wrapping a
        SINGLE static emission of the WMMA K-main-loop -- see class docstring and
        igemm_fwd_gtc_wmma_nhwc_t's Phase 5d docstring for the full rationale. v_c is never
        reset between taps, so successive taps' D=A@B+C calls accumulate naturally.
        '''
        s = self.sgpr
        label_tap_y = f"L_{self.name()}_tap_y"
        label_tap_x = f"L_{self.name()}_tap_x"
        self._emit_front(f"{label_tap_y}:")
        self._emit(f"s_mov_b32 s[{s.s_ix()}], 0")
        self._emit_front(f"{label_tap_x}:")
        if self.lds_buffer_num == 2:
            # Phase 2 (double-buffering): recompute the fresh, untoggled state at the
            # start of EVERY tap -- see _emit_lds_offset_setup's docstring for why.
            self._emit_lds_offset_setup()
        if not self.tunable.tdm_global_load:
            # Phase 42: skipped entirely under TDM -- _emit_tap_gather's whole purpose is
            # computing v_addr_a/v_flag (A) and v_addr_b (B's per-tap column offset), none
            # of which global_load_a/b_functor's tdm_global_load branches reference at all
            # (TDM's own descriptor, rebuilt via move_slice_window, replaces both). Unlike
            # fwd (where the equivalent unused-under-TDM computation is a few cheap
            # multiplies), bwd's A-side gather is genuinely expensive (two integer-division
            # macro calls per tap) -- leaving it in would silently defeat much of the point
            # of this extension, so it is skipped explicitly here rather than left as
            # dead-but-emitted code the way fwd's original Phase 28 pilot did.
            self._emit_tap_gather()
        if self.tunable.wmma_k_tail:
            # K-tail: this tap's first chunk (below) gets stored via shared_store_a/b_functor
            # BEFORE wmma_main_loop.py's own "s_kitr = s_knum" init runs (that init happens
            # right after this prologue block's f_sst_a()/f_sst_b() calls, per
            # wmma_main_loop.py's emit()) -- so the masking code (which reads s_kitr as "how
            # many valid elements remain from this tile's own start") needs it set fresh
            # here first. Harmless/redundant once wmma_main_loop.py sets it again moments
            # later to the same value.
            self._emit(f"s_mov_b32 s[{s.s_kitr()}], s[{s.s_knum()}]   ; K-tail: needed before this tap's first chunk store")
        # ---- issue the first global loads for this tap (main loop expects this precondition;
        # global_load_a_functor re-zeros v_gld_a on every call, see its own docstring) ----
        self._emit(self.global_load_a_functor()())
        self._emit(self.global_load_b_functor()())
        self._emit_empty_line()

        # ---- the WMMA K-main-loop over K_out, emitted EXACTLY ONCE here; the runtime branches
        # below re-enter this same code for every tap ----
        self.emit_kernel_fma_main_loop()
        self._emit_empty_line()

        self._emit(f"s_add_u32 s[{s.s_ix()}], s[{s.s_ix()}], 1")
    def _emit_gld_chunk_load(self, v_gld, v_addr, chunk_idx, v_flag=None, saddr=None):
        ''' Phase 1 (k-sub-loop): issues (does not wait) ONE inst_wmma.k-wide chunk's
        global load into the small, reused v_gld buffer.

        saddr (Phase 61, 32-bit SADDR global loads): optional scalar base symbol
        (s.s_p_in/s.s_p_wei). When set, v_addr is a single 32-bit byte-offset VGPR
        (v_off_a/v_off_b) and the load uses the SADDR form. '''
        if v_flag is not None:
            for i in range(self.chunk_num_dwords):
                self._emit(f"v_mov_b32 v[{v_gld(i)}], 0")
            self._emit(f"v_cmpx_le_u32 1, v[{v_flag()}]")
        for i in range(self.chunk_num_dwordx4):
            idx = chunk_idx * self.chunk_num_dwordx4 + i
            if saddr is not None:
                self._emit(f"global_load_dwordx4 v[{v_gld(i*4)}:{v_gld(i*4+3)}], v[{v_addr()}], s[{saddr()}:{saddr(1)}] offset:{idx*16}")
            else:
                self._emit(f"global_load_dwordx4 v[{v_gld(i*4)}:{v_gld(i*4+3)}], v[{v_addr()}:{v_addr(1)}], off offset:{idx*16}")
        if v_flag is not None:
            self._emit(f"s_mov_b32 exec_lo, -1")

    def _emit_sst_chunk(self, v_gld, v_sst_os, sst_extra_off, chunk_idx):
        ''' Phase 1 (k-sub-loop): stores ONE already-loaded-and-waited chunk to LDS. '''
        for i in range(self.chunk_num_dwordx4):
            idx = chunk_idx * self.chunk_num_dwordx4 + i
            self._emit(f"ds_write_b128 v[{v_sst_os()}], v[{v_gld(i*4)}:{v_gld(i*4+3)}] offset:{sst_extra_off + idx*16}")

    def _emit_sst_remaining_chunks(self, v_gld, v_addr, v_sst_os, sst_extra_off, v_flag=None, tail_mask=None, saddr=None):
        '''
        Phase 1 (k-sub-loop): stores chunk 0 (already loaded+waited via the existing
        global_load_a/b_functor + outer s_wait_loadcnt call sequence in
        wmma_main_loop.py), then load+wait+stores chunks 1..num_k_chunks-1 sequentially,
        reusing the same small v_gld buffer. Deliberately does NOT overlap these
        remaining chunks' loads with wmma compute -- ALL of these stores happen only
        after shared_store_a/b_functor is called, i.e. only after every read of the
        CURRENT tile is done, preserving the single-buffered-LDS safety invariant the
        original design relies on (no wave may overwrite a tile's LDS storage until
        every wave has finished reading it). See igemm_fwd_gtc_wmma_nhwc_t's identically-
        named method for the full incident writeup (an early attempt stored chunk 0
        immediately after its own load and produced silent wrong-answer corruption on
        real hardware starting at the 3rd within-workgroup K-block).

        tail_mask (K/N-tail, new): optional (remaining_operand, skip_check_fn) tuple -- see
        _emit_tail_dword_mask_guarded. When set, applied to each chunk's data right after
        it's loaded+waited, before that chunk is stored to LDS.
        '''
        elem_per_dword = 4 // self.data_byte
        if tail_mask is not None:
            self._emit_tail_dword_mask_guarded(v_gld, self.chunk_num_dwords, 0, tail_mask)
        self._emit_sst_chunk(v_gld, v_sst_os, sst_extra_off, 0)
        for c in range(1, self.num_k_chunks):
            self._emit_gld_chunk_load(v_gld, v_addr, c, v_flag=v_flag, saddr=saddr)
            self._emit(f"s_wait_loadcnt 0x0")
            if tail_mask is not None:
                self._emit_tail_dword_mask_guarded(v_gld, self.chunk_num_dwords, c * self.chunk_num_dwords * elem_per_dword, tail_mask)
            self._emit_sst_chunk(v_gld, v_sst_os, sst_extra_off, c)

    def _emit_sst_all_chunks(self, v_gld, v_addr, v_sst_os, sst_extra_off, v_flag=None, tail_mask=None, saddr=None):
        '''
        Phase 1 (k-sub-loop): like _emit_sst_remaining_chunks, but load+wait+stores ALL
        num_k_chunks chunks here (including chunk 0 -- global_load_b_functor issues no
        early load at all for this operand). Needed specifically for B: its
        shared_load_b_functor reuses v_gld_b as scratch for the transposed read-and-pack
        technique, called again (for substep>=1) via emit_extra_substeps() -- which runs
        AFTER global_load_b_functor but BEFORE shared_store_b_functor. An early chunk-0
        load surviving in v_gld_b across that window would get silently clobbered by that
        scratch reuse before ever being stored (caught on real hardware as silent
        wrong-answer corruption, distinct from the cross-wave LDS-overwrite race
        _emit_sst_remaining_chunks itself guards against). A is unaffected (untransposed,
        read directly into v_a by _emit_ds_read_chunked -- no scratch, no clobbering risk),
        so it keeps chunk 0's overlap-with-compute via _emit_sst_remaining_chunks.

        tail_mask: see _emit_sst_remaining_chunks above.
        '''
        elem_per_dword = 4 // self.data_byte
        for c in range(self.num_k_chunks):
            self._emit_gld_chunk_load(v_gld, v_addr, c, v_flag=v_flag, saddr=saddr)
            self._emit(f"s_wait_loadcnt 0x0")
            if tail_mask is not None:
                self._emit_tail_dword_mask_guarded(v_gld, self.chunk_num_dwords, c * self.chunk_num_dwords * elem_per_dword, tail_mask)
            self._emit_sst_chunk(v_gld, v_sst_os, sst_extra_off, c)

    def _emit_tail_dword_mask(self, v_gld, num_dwords, elements_before, remaining_operand):
        '''
        K/N-tail (bwd-specific, new -- see __init__'s docstring for why this fine-grained,
        sub-lane masking is needed at all, unlike M-tail's simple per-lane EXEC-mask): zeros
        the invalid trailing sub-elements of `num_dwords` dwords of already-loaded (not yet
        stored to LDS) data in `v_gld`, given `remaining_operand` (a ready-to-use VALU source
        operand string, e.g. "s[...]" for K-tail -- uniform across every lane, since K
        validity only depends on the workgroup-wide loop position -- or "v[...]" for N-tail --
        per-lane, since N validity depends on each lane's own col_group) = how many valid
        elements remain from `elements_before`'s reference point (elements_before is this
        dword's own offset from that same reference point, in elements). Both scalar and
        vector sources are legal operands for the VALU instructions used here (verified via
        llvm-mc against gfx1250), so one instruction sequence serves both callers.

        For each dword, `valid_in_dword = clamp(remaining - elements_before, 0, elem_per_dword)`
        is built into a byte-lane mask via one v_cmp_ge_i32/v_cndmask_b32 pair per possible
        element count (elem_per_dword is at most 4, so this never becomes a large unrolled
        sequence) OR'd together, then ANDed into that dword. Uses v_tmp(0..2) as scratch
        (safe: not live across this call at either call site -- inserted between a chunk's
        load+wait and its LDS store). Always emits the full sequence; callers wrap this in
        _emit_tail_dword_mask_guarded to skip it cheaply in the overwhelmingly common
        (no tail on this particular tile/lane) case.
        '''
        v = self.vgpr
        elem_per_dword = 4 // self.data_byte
        bits_per_element = self.data_byte * 8
        for d in range(num_dwords):
            base = elements_before + d * elem_per_dword
            self._emit(f"v_sub_u32 v[{v.v_tmp(0)}], {remaining_operand}, {base}   ; valid_raw, dword {d}")
            self._emit(f"v_max_i32 v[{v.v_tmp(0)}], 0, v[{v.v_tmp(0)}]")
            self._emit(f"v_min_i32 v[{v.v_tmp(0)}], {elem_per_dword}, v[{v.v_tmp(0)}]   ; valid_in_dword, clamped [0,{elem_per_dword}]")
            for k in range(1, elem_per_dword + 1):
                element_mask = (((1 << bits_per_element) - 1) << ((k - 1) * bits_per_element)) & 0xffffffff
                self._emit(f"v_cmp_ge_i32 vcc_lo, v[{v.v_tmp(0)}], {k}")
                dst = v.v_tmp(1) if k == 1 else v.v_tmp(2)
                self._emit(f"v_cndmask_b32 v[{dst}], 0, {hex(element_mask)}, vcc_lo   ; element {k-1} keep-mask")
                if k != 1:
                    self._emit(f"v_or_b32 v[{v.v_tmp(1)}], v[{v.v_tmp(1)}], v[{v.v_tmp(2)}]")
            self._emit(f"v_and_b32 v[{v_gld(d)}], v[{v.v_tmp(1)}], v[{v_gld(d)}]   ; apply mask")
        self._emit_empty_line()

    def _emit_tail_dword_mask_guarded(self, v_gld, num_dwords, elements_before, tail_mask):
        '''
        tail_mask = (remaining_operand, skip_check_fn): skip_check_fn(label) returns a list
        of instruction strings ending in a conditional scalar branch to `label` when NO
        masking is needed for this call at all -- e.g. K-tail checks s_kitr against
        gemm_k_per_block (workgroup-wide, changes every K-loop iteration); N-tail checks
        s_block_n_off+gemm_n_per_block against s_gemm_n (kernel-lifetime constant, still
        recomputed fresh here for simplicity -- correctness over the last few cycles for
        this new mechanism). Keeps the ~10-15-instructions-per-dword mask machinery off the
        hot path for the overwhelmingly common non-tail case.
        '''
        remaining_operand, skip_check_fn = tail_mask
        self._tail_mask_label_id += 1
        label = f"L_{self.name()}_tail_mask_{self._tail_mask_label_id}"
        for line in skip_check_fn(label):
            self._emit(line)
        self._emit_tail_dword_mask(v_gld, num_dwords, elements_before, remaining_operand)
        self._emit_front(f"{label}:")

    def _emit_gld_async_all_chunks(self, v_off, v_sst_os, sst_extra_off, s_saddr, v_flag=None):
        ''' Phase 13: see igemm_fwd_gtc_wmma_nhwc.py's identically-named method -- A
        (untransposed) only; B stays on the existing transposed read-and-pack technique.
        sst_extra_off folded into VDST via v_sst_tmp, not the shared immediate -- see
        v_sst_tmp's declaration (fwd hit this as a real bug for its B operand; this
        operand's sst_extra_off is always 0 today, but the fix is applied for parity). '''
        v = self.vgpr
        if sst_extra_off != 0:
            self._emit(f"v_add_u32 v[{v.v_sst_tmp()}], {sst_extra_off}, v[{v_sst_os()}]")
            dst = v.v_sst_tmp
        else:
            dst = v_sst_os
        if v_flag is not None:
            self._emit(f"v_cmpx_le_u32 1, v[{v_flag()}]")
        for c in range(self.num_k_chunks):
            for i in range(self.chunk_num_dwordx4):
                idx = c * self.chunk_num_dwordx4 + i
                self._emit(f"global_load_async_to_lds_b128 v[{dst()}], v[{v_off()}], "
                           f"s[{s_saddr()}:{s_saddr(1)}] offset:{idx*16}")
        if v_flag is not None:
            self._emit(f"s_xor_b32 exec_lo, exec_lo, -1")
            for c in range(self.num_k_chunks):
                for i in range(self.chunk_num_dwordx4):
                    idx = c * self.chunk_num_dwordx4 + i
                    self._emit(f"ds_write_b128 v[{dst()}], v[{v.v_zero(0)}:{v.v_zero(3)}] "
                               f"offset:{idx*16}")
            self._emit(f"s_mov_b32 exec_lo, -1")

    def global_load_a_functor(self):
        '''
        Zeros v_gld_a on EVERY call (not just once, since Phase 5e's per-tap flag can flip
        between taps -- same discipline Phase 5c/wrw and Phase 5d/fwd established for
        their own per-iteration/per-tap-varying gathers), then EXEC-masks the load itself
        so a stride-gap/oob lane's v_gld_a simply stays zero for this tap/K-out chunk.
        Only chunk 0's load is issued here -- see igemm_fwd_gtc_wmma_nhwc_t's identically
        named method for why double-buffering does NOT change this (moving chunk 0's
        wait+store earlier was tried and regressed performance by discarding its
        overlap-with-compute).

        Phase 13 (async_global_load=1): see igemm_fwd_gtc_wmma_nhwc.py's identically named
        method -- issues ALL chunks directly to LDS via global_load_async_to_lds_b128, no
        scratch buffer. shared_store_a_functor is not called at all in this mode.
        '''
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                s = outer.sgpr
                with outer._deferred_context():
                    if outer.tunable.tdm_global_load:
                        # Phase 42: mirrors igemm_fwd_gtc_wmma_nhwc_t's identical TDM branch --
                        # one TDM instruction moves the whole gemm_m_per_block x
                        # gemm_k_per_block tile straight into LDS, wave-0-only issue.
                        outer._emit_wave0_only(lambda: outer._emit(f"tensor_load_to_lds s[{s.s_tdm_g0()}:{s.s_tdm_g0(3)}], s[{s.s_tdm_g1()}:{s.s_tdm_g1(7)}]"))
                    elif outer.tunable.async_global_load:
                        outer._emit_gld_async_all_chunks(v.v_off_a, v.v_sst_os, 0, outer.sgpr.s_p_in, v_flag=v.v_flag)
                    elif outer.tunable.saddr_global_load:
                        outer._emit_gld_chunk_load(v.v_gld_a, v.v_off_a, 0, v_flag=v.v_flag, saddr=s.s_p_in)
                    else:
                        outer._emit_gld_chunk_load(v.v_gld_a, v.v_addr_a, 0, v_flag=v.v_flag)
                return outer._get_deferred()
            def get_issues(self):
                return outer.chunk_num_dwordx4
        return functor_t()

    def global_load_b_functor(self):
        '''
        B is TRANSPOSED and its shared_load_b_functor reuses v_gld_b as scratch for the
        read-and-pack technique. When num_k_chunks>1, emit_extra_substeps() calls
        shared_load_b_functor AGAIN (for substep>=1) between when this functor issues
        chunk 0's global load and when shared_store_b_functor finally stores it --
        clobbering chunk 0's staged data via that scratch reuse before it's ever stored
        (caught on real hardware as silent wrong-answer corruption; a plain "wait for the
        load" fix is NOT enough, since the data itself gets overwritten, not just raced).
        So for num_k_chunks>1, this issues NOTHING here -- shared_store_b_functor's
        _emit_sst_all_chunks handles every chunk (including chunk 0) itself, fully after
        emit_extra_substeps() is done with v_gld_b. For num_k_chunks==1 (every existing
        config), emit_extra_substeps() never runs, so there's no clobbering risk and this
        keeps the original single-chunk load-early/overlap-with-compute behavior exactly.
        '''
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                s = outer.sgpr
                with outer._deferred_context():
                    if outer.tunable.tdm_global_load:
                        # Phase 42: mirrors global_load_a_functor's TDM branch above.
                        outer._emit_wave0_only(lambda: outer._emit(f"tensor_load_to_lds s[{s.s_tdm_g0_b()}:{s.s_tdm_g0_b(3)}], s[{s.s_tdm_g1_b()}:{s.s_tdm_g1_b(7)}]"))
                    else:
                        v_flag = None
                        if outer.tunable.wmma_k_tail:
                            # K-tail, easy case (see __init__'s docstring): this lane's
                            # row_local (persistent, prologue-computed) IS the per-lane K
                            # position for its whole chunk -- a plain per-lane EXEC-mask check,
                            # recomputed fresh every call since s_kitr changes every K-loop
                            # iteration (unlike N-tail's col_start_abs, which doesn't).
                            outer._emit(f"v_cmp_gt_i32 vcc_lo, s[{s.s_kitr()}], v[{v.v_b_row_local()}]   ; K-tail: row_local < remaining")
                            outer._emit(f"v_cndmask_b32 v[{v.v_flag_b_ktail()}], 0, 1, vcc_lo")
                            v_flag = v.v_flag_b_ktail
                        if outer.num_k_chunks == 1:
                            if outer.tunable.saddr_global_load:
                                outer._emit_gld_chunk_load(v.v_gld_b, v.v_off_b, 0, v_flag=v_flag, saddr=s.s_p_wei)
                            else:
                                outer._emit_gld_chunk_load(v.v_gld_b, v.v_addr_b, 0, v_flag=v_flag)
                return outer._get_deferred()
            def get_issues(self):
                return outer.chunk_num_dwordx4
        return functor_t()

    def shared_store_a_functor(self):
        ''' row_repeat_a > 1: row 0 uses the exact row_repeat_a==1 code path (unchanged);
        rows 1..row_repeat_a-1 (no early-issue slot of their own -- see global_load_a_functor's
        docstring) are fully deferred via _emit_sst_all_chunks (loads+stores every chunk,
        including chunk 0 -- already exactly what that method does for B's num_k_chunks>1
        case), storing into LDS shifted by i*block_size*bytes_per_row. wmma_k_tail is
        asserted incompatible with row_repeat_a>1 (see __init__), so rows 1+ never need
        tail_mask. '''
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                s = outer.sgpr
                with outer._deferred_context():
                    k_tail = None
                    if outer.tunable.wmma_k_tail:
                        def _skip(label, s=s, outer=outer):
                            return [f"s_cmp_ge_i32 s[{s.s_kitr()}], {outer.tunable.gemm_k_per_block}",
                                    f"s_cbranch_scc1 {label}"]
                        k_tail = (f"s[{s.s_kitr()}]", _skip)
                    if outer.tunable.saddr_global_load:
                        outer._emit_sst_remaining_chunks(v.v_gld_a, v.v_off_a, v.v_sst_os, 0, v_flag=v.v_flag, tail_mask=k_tail, saddr=s.s_p_in)
                    else:
                        outer._emit_sst_remaining_chunks(v.v_gld_a, v.v_addr_a, v.v_sst_os, 0, v_flag=v.v_flag, tail_mask=k_tail)
                    for i in range(1, outer.row_repeat_a):
                        row_addr = lambda idx=0, i=i: v.v_addr_a(i * 2 + idx)
                        row_flag = lambda i=i: v.v_flag(i)
                        row_off  = i * outer.tunable.block_size * outer.bytes_per_row
                        outer._emit_sst_all_chunks(v.v_gld_a, row_addr, v.v_sst_os, row_off, v_flag=row_flag)
                return outer._get_deferred()
            def get_issues(self):
                return outer.chunk_num_dwordx4
        return functor_t()

    def shared_store_b_functor(self):
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                s = outer.sgpr
                with outer._deferred_context():
                    n_tail = None
                    if outer.tunable.wmma_n_tail:
                        # N-tail, hard case (see __init__'s docstring): col_start_abs is a
                        # kernel-lifetime constant, so the skip check (is this WHOLE
                        # workgroup's N-block entirely in-bounds) is too -- still recomputed
                        # fresh each call for simplicity, cheap either way.
                        def _skip(label, s=s, outer=outer):
                            return [f"s_add_u32 s[{s.s_tail_tmp()}], s[{s.s_block_n_off()}], {outer.tunable.gemm_n_per_block}",
                                    f"s_cmp_le_u32 s[{s.s_tail_tmp()}], s[{s.s_gemm_n()}]",
                                    f"s_cbranch_scc1 {label}"]
                        n_tail = (f"v[{v.v_n_valid_base()}]", _skip)
                    v_b_addr = v.v_off_b if outer.tunable.saddr_global_load else v.v_addr_b
                    s_b_saddr = s.s_p_wei if outer.tunable.saddr_global_load else None
                    if outer.num_k_chunks == 1:
                        outer._emit_sst_remaining_chunks(v.v_gld_b, v_b_addr, v.v_sst_os, outer.lds_a_size, v_flag=None, tail_mask=n_tail, saddr=s_b_saddr)
                    else:
                        outer._emit_sst_all_chunks(v.v_gld_b, v_b_addr, v.v_sst_os, outer.lds_a_size, v_flag=None, tail_mask=n_tail, saddr=s_b_saddr)
                return outer._get_deferred()
            def get_issues(self):
                return outer.chunk_num_dwordx4
        return functor_t()

    def global_load_chunk_a_functor(self):
        ''' Phase 15 (bwd port): single-chunk primitive for the interleaved main loop --
        issues ONE chunk's global load of A (grad_output, untransposed). Reuses the exact
        same helper (_emit_gld_chunk_load) and v_flag masking as global_load_a_functor's
        chunk-0 call, just parameterized by chunk_idx instead of hardcoded to 0. '''
        outer = self
        class functor_t:
            def __call__(self, chunk_idx):
                v = outer.vgpr
                s = outer.sgpr
                with outer._deferred_context():
                    if outer.tunable.saddr_global_load:
                        outer._emit_gld_chunk_load(v.v_gld_a, v.v_off_a, chunk_idx, v_flag=v.v_flag, saddr=s.s_p_in)
                    else:
                        outer._emit_gld_chunk_load(v.v_gld_a, v.v_addr_a, chunk_idx, v_flag=v.v_flag)
                return outer._get_deferred()
        return functor_t()

    def shared_store_chunk_a_functor(self):
        ''' Phase 15 (bwd port): single-chunk primitive for the interleaved main loop --
        stores ONE already-loaded-and-waited chunk of A to LDS (reuses _emit_sst_chunk). '''
        outer = self
        class functor_t:
            def __call__(self, chunk_idx):
                v = outer.vgpr
                with outer._deferred_context():
                    outer._emit_sst_chunk(v.v_gld_a, v.v_sst_os, 0, chunk_idx)
                return outer._get_deferred()
        return functor_t()

    def _emit_ds_read_chunked(self, v_base_sym, v_os_sym, base_off, num_v):
        '''
        Reads `num_v` contiguous dwords from LDS starting at `base_off`, into
        `v_base_sym(0..num_v-1)`, in the largest chunks ds_read_* supports (b128=4 dwords,
        b64=2, b32=1). Generalizes the old fp16/bf16/int8-only "always 2x ds_read_b128 (8
        dwords)" pattern, which hardcoded num_v_a/num_v_b=8 -- fp32 has num_v_a/num_v_b=2, so
        this must read a different (smaller) chunk shape. See igemm_fwd_gtc_wmma_nhwc_t's
        identical helper / Phase 8 docstring.
        '''
        remaining = num_v
        idx = 0
        off = 0
        while remaining > 0:
            if remaining >= 4:
                self._emit(f"ds_read_b128 v[{v_base_sym(idx)}:{v_base_sym(idx+3)}], v[{v_os_sym()}] offset:{base_off+off}")
                idx += 4; off += 16; remaining -= 4
            elif remaining >= 2:
                self._emit(f"ds_read_b64 v[{v_base_sym(idx)}:{v_base_sym(idx+1)}], v[{v_os_sym()}] offset:{base_off+off}")
                idx += 2; off += 8; remaining -= 2
            else:
                self._emit(f"ds_read_b32 v[{v_base_sym(idx)}], v[{v_os_sym()}] offset:{base_off+off}")
                idx += 1; off += 4; remaining -= 1

    def shared_load_a_functor(self):
        outer = self
        num_v_a = outer.wmma_mapping.ctrl.inst_wmma.num_v_a
        num_v_a_total = outer.tunable.wmma_repeat_m * num_v_a   # Phase 22: one local_prefetch_num slot's worth
        step_bytes = outer.tunable.wmma_tile_m * outer.bytes_per_row   # was hardcoded 1024
        class functor_t:
            def __call__(self, v_dst, v_os, extra_off, slot=0):
                v = outer.vgpr
                slot_off = slot * num_v_a_total
                with outer._deferred_context():
                    for i_rm in range(outer.tunable.wmma_repeat_m):
                        base = i_rm * step_bytes + extra_off
                        outer._emit_ds_read_chunked(lambda k, i_rm=i_rm: v.v_a(slot_off+i_rm*num_v_a+k), v.v_sld_a_os, base, num_v_a)
                return outer._get_deferred()
        return functor_t()

    def shared_load_b_functor(self):
        '''
        Transposed read for the weight operand: LDS holds it as natural [K rows][N cols]
        (row_pitch = gemm_n_per_block*databyte bytes), but the WMMA B operand needs
        col-major/k-contiguous packed elements. For each wave_repeat step i_rn and each of the
        `num_v_b` vgpr indices a (k-half-relative for fp16/bf16/int8; fp32 has no k-half split
        at all since num_v_b=2 exactly matches K/2), read the `elem_per_dword` k-sub-elements
        (s=0..elem_per_dword-1) as separate sub-dword LDS loads (they are row_pitch_bytes
        apart, not adjacent) into a small slice of v_gld_b (reused as scratch here -- safe,
        since this iteration's real global load into v_gld_b happens later in the main loop,
        after shared_load completes), then pack them into one dword with a v_lshl_or_b32 chain
        -- deliberately correctness-over-speed, see class docstring. `elem_per_dword` is 2 for
        fp16/bf16 (16-bit reads, one shift-16 pack), 4 for int8 (8-bit reads, three chained
        shift-8/16/24 packs), or 1 for fp32 (a plain full-dword read, no packing at all -- fp32
        has no sub-dword elements to begin with) -- see docs/gfx1250_wmma_layout.md's
        per-precision A/B operand layout facts. Processes one vgpr index `a` at a time (wait,
        then pack, before moving to the next `a`) rather than batching all reads-then-all-packs
        like the fp16-only version this replaced: int8's elem_per_dword=4 would need 32 scratch
        registers to batch all 8 (v_gld_b is only 16, sized for its OTHER use as the real
        global-load staging buffer) -- trading some latency-hiding for staying within the
        existing VGPR budget. `num_v_b` (not hardcoded 8) is also new for Phase 8/fp32 -- every
        other precision has num_v_b=8, but fp32 has num_v_b=2.
        '''
        outer = self
        class functor_t:
            def __call__(self, v_dst, v_os, extra_off, slot=0):
                v = outer.vgpr
                num_v_b = outer.wmma_mapping.ctrl.inst_wmma.num_v_b
                num_v_b_total = outer.tunable.wmma_repeat_n * num_v_b   # Phase 22: one local_prefetch_num slot's worth
                slot_off = slot * num_v_b_total
                row_pitch = outer.tunable.gemm_n_per_block * outer.data_byte
                elem_per_dword = 4 // outer.data_byte
                if outer.data_byte == 2:
                    read_instr = 'ds_read_u16'
                elif outer.data_byte == 4:
                    read_instr = 'ds_read_b32'   # fp32: full-dword read, no zero-extension needed
                else:
                    read_instr = 'ds_read_u8'
                with outer._deferred_context():
                    for i_rn in range(outer.tunable.wmma_repeat_n):
                        col_off = i_rn * outer.tunable.wmma_tile_n * outer.data_byte
                        for a in range(num_v_b):
                            # B region starts at outer.lds_a_size within the shared LDS tile;
                            # v_sld_b_os only carries the offset local to B's own region.
                            for s in range(elem_per_dword):
                                off = outer.lds_a_size + col_off + (a * elem_per_dword + s) * row_pitch
                                outer._emit(f"{read_instr} v[{v.v_gld_b(s)}], v[{v.v_sld_b_os()}] offset:{extra_off + off}")
                            outer._emit(f"s_wait_dscnt 0x0")
                            outer._emit(f"v_mov_b32 v[{v.v_b(slot_off+i_rn*num_v_b+a)}], v[{v.v_gld_b(0)}]")
                            for s in range(1, elem_per_dword):
                                shift = s * 8 * outer.data_byte
                                outer._emit(f"v_lshl_or_b32 v[{v.v_b(slot_off+i_rn*num_v_b+a)}], v[{v.v_gld_b(s)}], {shift}, v[{v.v_b(slot_off+i_rn*num_v_b+a)}]")
                return outer._get_deferred()
        return functor_t()

    def move_slice_window_a_functor(self):
        ''' row_repeat_a > 1: every row's v_addr_a pair advances independently by the same
        per-K-substep stride, mirroring igemm_fwd_gtc_wmma_nhwc_t's identically-named method. '''
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                s = outer.sgpr
                with outer._deferred_context():
                    if outer.tunable.tdm_global_load:
                        # Phase 42: advance the TDM descriptor's global_addr by one K-chunk's
                        # worth of bytes (mirrors igemm_fwd_gtc_wmma_nhwc_t's identical A-side
                        # Phase 28/31 branch exactly -- A's tensor_dim0=gemm_k is the
                        # contiguous axis here too, so bytes_per_row is the right advance),
                        # then rebuild tensor_dim0 from s_tdm_k_remain (already decremented
                        # for this tile by wmma_main_loop.py) so TDM's hardware OOB correctly
                        # zero-fills a genuinely partial last K-tile.
                        outer._emit(f"s_add_u32 s[{s.s_tdm_g0(2)}], s[{s.s_tdm_g0(2)}], {outer.bytes_per_row}")
                        outer._emit(f"s_addc_u32 s[{s.s_tdm_g0(3)}], s[{s.s_tdm_g0(3)}], 0")
                        # Phase 44: skip the rebuild unless this call is genuinely preparing
                        # the tail tile -- see igemm_fwd_gtc_wmma_nhwc.py's identically-named
                        # functor for the full reasoning (this call only ever prepares the
                        # NEXT tile, so at most one call per K-loop needs the real rebuild;
                        # every other call's tensor_dim0 staying at whatever it was left at
                        # -- always >= tile_dim0 until the genuine tail -- is a direct
                        # structural consequence of the OOB check, not a new assumption).
                        skip_label = f"L_{outer.name()}_tdm_a_skip_rebuild"
                        outer._emit(f"s_cmp_lt_i32 s[{s.s_tdm_k_remain()}], {outer.tunable.gemm_k_per_block}   ; Phase 44: is the tile now being prepared genuinely partial?")
                        outer._emit(f"s_cbranch_scc0 {skip_label}   ; not partial -- skip the rebuild, tensor_dim0 stays >= tile_dim0")
                        outer._emit(f"s_lshl_b32 s[{s.s_tdm_g1(1)}], s[{s.s_tdm_k_remain()}], 16   ; tensor_dim0 (remaining K) lo16 -> [31:16]")
                        outer._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_tdm_k_remain()}], 16   ; tensor_dim0 hi16")
                        outer._emit(f"s_lshl_b32 s[{s.s_tmp(1)}], s[{s.s_gemm_m()}], 16   ; tensor_dim1 (gemm_m) lo16")
                        outer._emit(f"s_or_b32 s[{s.s_tdm_g1(2)}], s[{s.s_tmp(0)}], s[{s.s_tmp(1)}]")
                        outer._emit_front(f"{skip_label}:")
                    elif outer.tunable.async_global_load or outer.tunable.saddr_global_load:
                        # Phase 13/61: v_off_a is a plain 32-bit byte OFFSET (no base pointer
                        # folded in), so advancing it is a single add, no carry chain needed.
                        outer._emit(f"v_add_u32 v[{v.v_off_a()}], {outer.bytes_per_row}, v[{v.v_off_a()}]")
                    else:
                        for i in range(outer.row_repeat_a):
                            outer._emit(f"v_add_co_u32 v[{v.v_addr_a(i*2)}], vcc_lo, {outer.bytes_per_row}, v[{v.v_addr_a(i*2)}]")
                            outer._emit(f"v_add_co_ci_u32 v[{v.v_addr_a(i*2+1)}], vcc_lo, 0, v[{v.v_addr_a(i*2+1)}], vcc_lo")
                return outer._get_deferred()
        return functor_t()

    def move_slice_window_b_functor(self):
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                s = outer.sgpr
                with outer._deferred_context():
                    if outer.tunable.tdm_global_load:
                        # Phase 42: advance by s_wei_k_stride (one K-tile's worth of ROWS,
                        # NOT gemm_k_per_block contiguous bytes -- see
                        # _emit_tdm_descriptor_setup_b's docstring for why B's axes are
                        # swapped relative to fwd's B). K-tail-via-OOB rebuild targets
                        # tensor_dim1 (gemm_k, the row axis here), not tensor_dim0 -- the one
                        # structural difference from fwd's B move_slice_window.
                        outer._emit(f"s_add_u32 s[{s.s_tdm_g0_b(2)}], s[{s.s_tdm_g0_b(2)}], s[{s.s_wei_k_stride()}]")
                        outer._emit(f"s_addc_u32 s[{s.s_tdm_g0_b(3)}], s[{s.s_tdm_g0_b(3)}], 0")
                        # Phase 44: skip the rebuild unless this call is genuinely preparing
                        # the tail tile -- see igemm_fwd_gtc_wmma_nhwc.py's identically-named
                        # functor for the full reasoning. K is tensor_dim1 here (not
                        # tensor_dim0), but the same "only the genuine tail call needs the
                        # real rebuild" property applies unchanged.
                        skip_label = f"L_{outer.name()}_tdm_b_skip_rebuild"
                        outer._emit(f"s_cmp_lt_i32 s[{s.s_tdm_k_remain()}], {outer.tunable.gemm_k_per_block}   ; Phase 44: is the tile now being prepared genuinely partial?")
                        outer._emit(f"s_cbranch_scc0 {skip_label}   ; not partial -- skip the rebuild, tensor_dim1 stays >= tile_dim1")
                        outer._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_gemm_n()}], 16   ; tensor_dim0 (gemm_n) hi16 -- unchanged, re-derived fresh")
                        outer._emit(f"s_lshl_b32 s[{s.s_tmp(1)}], s[{s.s_tdm_k_remain()}], 16   ; tensor_dim1 (remaining K) lo16")
                        outer._emit(f"s_or_b32 s[{s.s_tdm_g1_b(2)}], s[{s.s_tmp(0)}], s[{s.s_tmp(1)}]")
                        outer._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_tdm_k_remain()}], 16   ; tensor_dim1 (remaining K) hi16")
                    elif outer.tunable.saddr_global_load:
                        # Phase 61: v_off_b is a plain 32-bit byte OFFSET, advance by
                        # s_wei_k_stride (one K-tile's worth of weight rows) -- no carry chain.
                        outer._emit(f"v_add_u32 v[{v.v_off_b()}], s[{s.s_wei_k_stride()}], v[{v.v_off_b()}]")
                    else:
                        outer._emit(f"v_add_co_u32 v[{v.v_addr_b()}], vcc_lo, s[{s.s_wei_k_stride()}], v[{v.v_addr_b()}]")
                        outer._emit(f"v_add_co_ci_u32 v[{v.v_addr_b(1)}], vcc_lo, 0, v[{v.v_addr_b(1)}], vcc_lo")
                return outer._get_deferred()
        return functor_t()

    def emit_kernel_fma_main_loop(self):
        ctrl = ctrl_wmma_main_loop_t()
        ctrl.wmma_m           = self.wmma_mapping.ctrl
        ctrl.unroll_k         = self.tunable.gemm_k_per_block
        ctrl.label_prefix     = self.name()
        ctrl.precision        = self.tunable.precision
        ctrl.lds_single_size  = self.lds_single_size
        ctrl.lds_buffer_num   = self.lds_buffer_num
        # Phase 13: A (untransposed) may go async; B stays on the old technique (transposed,
        # out of scope -- see class docstring / global_load_b_functor).
        ctrl.async_global_to_lds_a = self.tunable.async_global_load
        ctrl.async_global_to_lds_b = False
        ctrl.tdm_global_to_lds_a = self.tunable.tdm_global_load
        ctrl.tdm_global_to_lds_b = self.tunable.tdm_global_load
        ctrl.local_prefetch_num = self.tunable.local_prefetch_num
        # Phase 15 (bwd port): A (untransposed) interleaves; B (transposed, v_gld_b reused
        # as scratch by shared_load_b) stays on the deferred bulk path -- interleave_b=False.
        ctrl.interleave_a = self.tunable.main_loop_interleave
        ctrl.interleave_b = False
        ctrl.wmma_setprio = self.tunable.wmma_setprio
        ctrl.vgpr_msb_tracker = self.vgpr_msb_tracker
        # Phase 1 (k-sub-loop): A (grad_output/input, untransposed) advances K-contiguous
        # bytes; B (weight, TRANSPOSED -- [K rows][N cols] in LDS) advances whole K-rows,
        # i.e. inst_wmma.k * row_pitch (row_pitch = gemm_n_per_block*data_byte), matching
        # shared_load_b_functor's own row_pitch computation.
        inst_wmma_k = self.wmma_mapping.ctrl.inst_wmma.k
        ctrl.k_substep_stride_bytes_a    = inst_wmma_k * self.data_byte
        ctrl.k_substep_stride_bytes_b    = inst_wmma_k * (self.tunable.gemm_n_per_block * self.data_byte)
        ctrl.global_load_a_functor       = self.global_load_a_functor()
        ctrl.global_load_b_functor       = self.global_load_b_functor()
        ctrl.shared_store_a_functor      = self.shared_store_a_functor()
        ctrl.shared_store_b_functor      = self.shared_store_b_functor()
        ctrl.shared_load_a_functor       = self.shared_load_a_functor()
        ctrl.shared_load_b_functor       = self.shared_load_b_functor()
        ctrl.move_slice_window_a_functor = self.move_slice_window_a_functor()
        ctrl.move_slice_window_b_functor = self.move_slice_window_b_functor()
        if self.tunable.main_loop_interleave:
            ctrl.global_load_chunk_a_functor  = self.global_load_chunk_a_functor()
            ctrl.shared_store_chunk_a_functor = self.shared_store_chunk_a_functor()
        ctrl.v_a       = sym_t(self.vgpr.v_a.label)
        ctrl.v_b       = sym_t(self.vgpr.v_b.label)
        ctrl.v_c       = sym_t(self.vgpr.v_c.label)
        ctrl.v_sst_a_os = sym_t(self.vgpr.v_sst_os.label)
        ctrl.v_sld_a_os = sym_t(self.vgpr.v_sld_a_os.label)
        ctrl.v_sst_b_os = sym_t(self.vgpr.v_sst_os.label)
        ctrl.v_sld_b_os = sym_t(self.vgpr.v_sld_b_os.label)
        ctrl.s_kitr    = sym_t(self.sgpr.s_kitr.label)
        ctrl.s_knum    = sym_t(self.sgpr.s_knum.label)
        if self.tunable.tdm_global_load:
            ctrl.s_tdm_k_remain = sym_t(self.sgpr.s_tdm_k_remain.label)

        # first global load for this tap already issued by emit_kernel_tap_loop(), which is
        # also the sole caller of this method (see class docstring, Phase 5e)
        wmma_main_loop_t(self.mc, ctrl).emit()

    def emit_kernel_epilogue(self):
        v = self.vgpr
        s = self.sgpr
        if os.environ.get('BWD_RAW_DUMP'):
            # DIAGNOSTIC: bypass coalescing_store, dump v_c raw to s_p_out (one dword per
            # register per thread). Proves whether v_c is already correct after the K-loop
            # (bug is in epilogue) or wrong (bug is in K-loop). Mirrors fwd's MSB_RAW_DUMP.
            # Layout: thread tid writes v_c[0..num_vgpr_accumulate_c-1] to
            # p_out + tid * num_vgpr_accumulate_c * 4, giving a flat raw dump.
            self._emit(f"; BWD_RAW_DUMP: raw v_c dump, bypassing coalescing_store")
            self._emit(f"v_lshlrev_b32 v[{v.v_addr_out()}], {utility_log2(self.tunable.num_vgpr_accumulate_c * 4)}, v[{v.v_tid()}]")
            self._emit(f"v_mov_b32 v[{v.v_addr_out(1)}], s[{s.s_p_out(1)}]")
            self._emit(f"v_add_co_u32 v[{v.v_addr_out()}], vcc_lo, s[{s.s_p_out()}], v[{v.v_addr_out()}]")
            self._emit(f"v_add_co_ci_u32 v[{v.v_addr_out(1)}], vcc_lo, 0, v[{v.v_addr_out(1)}], vcc_lo")
            for i in range(self.tunable.num_vgpr_accumulate_c):
                self._emit(f"global_store_dword v[{v.v_addr_out()}:{v.v_addr_out(1)}], v[{v.v_c(i)}] offset:{i*4}")
            self._emit(f"s_wait_storecnt 0x0")
            return
        # s_out_c_total (=gemm_n*group) is grad_input's TOTAL row stride (NOT K_out, and not
        # just the per-group gemm_n either once group>1 -- see class docstring's Phase 7 note).
        self._emit(self.coalescing_store(v.v_c.label, v.v_gemm_im(), v.v_gemm_in(), s.s_p_out.label, s.s_out_c_total.label, v.v_addr_out(), v.v_addr_out(1), s.s_tmp(), v.v_tid(), v.v_c(), s.s_block_m_off(), s.s_block_n_off(),
                    s.s_gemm_m.label if self.tunable.wmma_m_tail else None, v.v_m_tail_row() if self.tunable.wmma_m_tail else None,
                    s.s_gemm_n.label if self.tunable.wmma_n_tail else None, v.v_n_tail_col() if self.tunable.wmma_n_tail else None,
                    s.s_tmp(1) if self.tunable.wmma_n_tail else None,
                    v_chunked_col=v.v_chunked_col() if self.tunable.wmma_epilogue_chunked else None))
        self._emit(f"s_wait_storecnt 0x0")

    def emit_kernel_body(self):
        self.emit_kernel_prologue()
        self.emit_kernel_tap_loop()
        self.emit_kernel_epilogue()
