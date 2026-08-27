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
from .igemm_base import *

class igemm_wrw_gtc_wmma_nhwc_t(mc_base_t):
    '''
    gfx1250 WMMA kernel for backward-weight (grad-weight) convolution. Phase 5c added arbitrary
    stride and padding (1x1 filter only). Phase 5f now adds multi-tap filters (y,x >= 1) and
    dilation -- see igemm_fwd_gtc_wmma_nhwc_t's Phase 5d and igemm_bwd_gtc_wmma_nhwc_t's Phase
    5e docstrings for the sibling extensions this partially mirrors, and read on for why wrw's
    version is structurally different from both.

    **Critical difference from fwd/bwd's tap loop**: in fwd/bwd, every tap contributes to the
    SAME output pixel (all y*x taps are summed into one accumulator, stored once at the very
    end). In wrw, each tap produces a DIFFERENT, INDEPENDENT slice of the output tensor --
    confirmed against driver/naive_conv.h's naive_conv_wrw_nhwc: `filter_grad[...,ir,is,...] =
    value` is a value computed FRESH per (ir,is) tap, summed only over the (n,ho,wo) reduction
    dimension, NOT accumulated across taps. So here, `v_c` must be zeroed and the epilogue
    (coalescing_store) must fire ONCE PER TAP, not once at kernel end -- the tap loop's body is
    "zero accumulator; run one full K-main-loop reduction; store to this tap's own [Y,X] output
    slice", repeated y*x times via the same runtime (not compile-time-unrolled) branch loop
    fwd/bwd use to wrap a single static K-main-loop emission.

    Tensor A (grad_output) needs NO tap-dependence at all (same as the non-tap Phase 5c): its
    addressing is a plain [K_out] row read, and grad_output has no Y,X extent in its own
    storage. Only its address needs resetting to a persistent `v_addr_a_base` at the start of
    EVERY tap (move_slice_window_a incrementally bumps `v_addr_a` across a tap's K-loop, so it
    must be restored before the next tap's K-loop starts fresh).

    Tensor B (input) needs its existing per-iteration gather (`_emit_b_gather`, unchanged in
    structure from Phase 5c) extended with a per-tap bias: `hi_idx = ho_idx*stride_h - pad_h +
    iy*dilation_h` (same ADD sign as fwd's Phase 5d, since this is an output-index->input-index
    gather via multiplication, not bwd's harder divide-based one). Since `_emit_b_gather` reads
    the CURRENT `s_iy`/`s_ix` live every time it's invoked (from the prologue once per tap, and
    from `move_slice_window_b` every K-iteration within that tap), no special-casing is needed
    beyond adding the bias terms -- the existing call sites automatically pick up whichever tap
    is currently active.

    Output (grad_weight) is `[K_out][Y][X][C_in]` for a multi-tap filter (was plain
    `[K_out][C_in]` for 1x1) -- so its row stride becomes `y*x*c` (`s_wei_row_c = gemm_n*x*y`,
    same scalar fwd's Phase 5d/bwd's Phase 5e introduce for the weight tensor, reused here for
    the OUTPUT tensor since grad_weight and weight share the same on-disk layout convention),
    and each tap's C_in-column-block starts `(iy*x+ix)*c` elements into that row -- expressed as
    a byte offset added to a FRESH per-tap base pointer `s_p_out_tap` (not a VGPR offset, since
    `coalescing_store_wmma.py`'s `s_p_out` argument is the literal store base address for
    EVERY thread in the block; adding the tap's constant offset to the base pointer once per
    tap, rather than perturbing per-thread column indices, needed zero changes to that shared,
    direction-agnostic epilogue helper).

    Unlike fwd/bwd, GEMM_K here (N*Ho*Wo) is spatial, not a channel count, so the gather this
    phase adds must be recomputed EVERY main-loop iteration (a fresh (n,ho,wo) triple each
    K-block), not once in the prologue like fwd/bwd's per-thread-constant flag. Tensor A
    (grad_output) needs NO change at all: GEMM_K is *defined* as grad_output's own pixel count
    (N*Ho*Wo), so every k-index is trivially valid in grad_output's own natural storage,
    regardless of stride/pad -- exactly like fwd's weight operand or bwd's weight operand
    needed no stride/pad awareness (no spatial dependence on the affected operand). Tensor B
    (input), however, needs the SAME kind of gather Phase 5a added to fwd's input operand
    (multiply-based, output-index -> input-index, simple bounds check -- NOT bwd's harder
    divide-based stride-gap check, since here we go output->input via multiplication): each
    thread's absolute k-index (`k_block_off + row_local`, where `row_local` is now a
    PERSISTENT per-thread constant computed once, and `k_block_off` advances by
    `gemm_k_per_block` every iteration, tracked via `s_knum - s_kitr`) is decomposed into
    (n_idx, ho_idx, wo_idx) via chained division (see get_kernel_macros() /
    macro_int_div_rem_vs_gfx1250_t), gathered to (hi_idx, wi_idx) via the stride/pad formula,
    and masked via EXEC (`v_cmpx_le_u32`/restore) around B's global load -- but because this
    flag now changes every iteration (unlike fwd/bwd's kernel-lifetime constant), v_gld_b must
    be explicitly re-zeroed before EVERY masked load (not just once), or a lane that was valid
    two iterations ago and is invalid now would silently reuse stale data instead of zero.

    Same tile-shape constraints as fwd/bwd: 128x128x32, wmma_repeat 4x4, block_size
    128, precision fp16/bf16. gemm_m/gemm_n multiples of 128, gemm_k (N*Ho*Wo) a
    multiple of 32.

    Original degenerate-case structure this closely mirrors:

        grad_weight[m, n] = sum_k grad_output[k, m] * input[k, n]

    where m enumerates output channels (GEMM_M = k_out), n enumerates input channels
    (GEMM_N = c_in), k enumerates (n_batch, h, w) pixels (GEMM_K = n*ho*wo, the
    contraction dim -- batch*spatial, typically large, unlike fwd/bwd where K was a
    channel count).

    Both operands need the LDS-transpose treatment bwd introduced for its weight
    operand (see igemm_bwd_gtc_wmma_nhwc.py / wmma_mapping.py's
    get_gemm_index_for_src_matrix_transposed), because BOTH are physically stored
    pixel-major/channel-contiguous ([GEMM_K][GEMM_M or N]), the transpose of what a
    WMMA A/B operand load wants ([GEMM_M or N][GEMM_K], contiguous over K):

      - Tensor A (grad_output) is naturally [N,Ho,Wo,K_out] i.e. [GEMM_K][GEMM_M],
        K_out-contiguous -- GEMM_M's row pitch is K_out (s_gemm_m).
      - Tensor B (input) is naturally [N,Hi,Wi,C_in] i.e. [GEMM_K][GEMM_N],
        C_in-contiguous -- GEMM_N's row pitch is C_in (s_gemm_n). This is the exact
        same tensor/layout bwd's B operand (weight) used, just a different tensor
        (input here, not weight) with the same [K][contiguous] shape.

    Global load/LDS store for both operands stay in their natural (untransposed)
    [K rows][M or N contiguous] orientation, using the same row_local=tid>>2/
    col_group=tid&3 per-thread chunk assignment bwd's B operand introduced (now
    applied to A as well, since A needs identical treatment here). Only the
    WMMA-consumption LDS *read* (shared_load_a_functor/shared_load_b_functor) does
    the transpose, via get_gemm_index_for_src_matrix_transposed with side='m' for A
    and side='n' for B.

    Output (grad_weight) is written through the exact [K_out][C_in] layout/buffer
    the forward kernel READS as its weight input -- reuses coalescing_store_wmma.py's
    epilogue unmodified, with s_gemm_m_stride bound to GEMM_N=C_in (same pattern bwd
    used for its own, differently-shaped, output).

    GEMM_K (batch*spatial) is typically large and needs a real multi-iteration K-loop
    (already generically supported by wmma_main_loop.py / this class's s_kitr/s_knum,
    validated by bwd's multi-K-block hardware tests -- no new mechanism here).
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

        # Phase 24: 'fp16_f16acc' is a separate table key (not a field), see wmma_mapping.py --
        # picks v_wmma_f16_16x16x32_f16 (num_v_c=4) instead of v_wmma_f32_16x16x32_f16 (num_v_c=8).
        # Mutually exclusive with gemm_k_global_split (asserted in igemm_base.py) since the
        # atomic epilogue branch below was never adapted for a packed accumulator.
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
        ctrl_coalescing_store_wmma.gemm_k_global_split = tunable.gemm_k_global_split
        ctrl_coalescing_store_wmma.atomic_scope = tunable.atomic_scope
        ctrl_coalescing_store_wmma.atomic_cascade = tunable.atomic_cascade
        ctrl_coalescing_store_wmma.epilogue_lds_pad = tunable.epilogue_lds_pad
        # Phase 27: see igemm_fwd_gtc_wmma_nhwc.py's identical comment -- the ctrl field's
        # actual behavior is precision-agnostic (2-byte-packed accumulator), so both tunables
        # funnel into it.
        ctrl_coalescing_store_wmma.wmma_acc_f16 = tunable.wmma_acc_f16 or tunable.wmma_acc_bf16
        ctrl_coalescing_store_wmma.atomic_pack_bf16 = tunable.atomic_pack_bf16
        self.coalescing_store = igemm_coalescing_store_wmma_t(self.mc, ctrl_coalescing_store_wmma)

        # A-region (grad_output) and B-region (input): both natural [GEMM_K rows][M or N
        # contiguous] tiles -- same total byte size either way (32*128*databyte).
        self.lds_a_size = tunable.gemm_k_per_block * tunable.gemm_m_per_block * self.data_byte
        self.lds_b_size = tunable.gemm_k_per_block * tunable.gemm_n_per_block * self.data_byte
        # Phase 2 (double-buffering): see igemm_fwd_gtc_wmma_nhwc_t's identically-named
        # fields for the full rationale. NOTE: wrw's A+B are BOTH transposed (read-and-
        # pack scratch reuse), so unlike fwd/bwd's A, wrw's functors do NOT gain the
        # early-store optimization from double buffering -- see global_load_a/b_functor.
        # This still needs to be wired up correctly so the buffer-switch bookkeeping in
        # wmma_main_loop.py addresses LDS correctly whenever a wrw _dbuf config is built.
        self.lds_single_size = igemm_next_pow2(self.lds_a_size + self.lds_b_size)
        self.lds_buffer_num  = 2 if tunable.lds_double_buffer else 1

        # gemm_k_per_block*data_byte happens to equal 64 bytes for fp16/bf16/int8 (32*2, 32*2,
        # 64*1), but fp32 forces gemm_k_per_block=4 (matching inst_wmma.k), giving 4*4=16 --
        # see igemm_fwd_gtc_wmma_nhwc_t's Phase 8 docstring for the full rationale. Every
        # literal built on the 64-byte coincidence is now derived from these instead.
        self.bytes_per_row = tunable.gemm_k_per_block * self.data_byte
        self.num_dwordx4   = self.bytes_per_row // 16
        self.num_dwords    = self.bytes_per_row // 4
        # Phase 1 (k-sub-loop): global_load/shared_store are chunked into num_k_chunks
        # rounds of one inst_wmma.k-worth each, reusing the same small v_gld_a/b buffer
        # across chunks -- see igemm_fwd_gtc_wmma_nhwc_t's __init__ docstring.
        inst_wmma_k_bytes = ctrl_wmma_mapping.inst_wmma.k * self.data_byte
        self.chunk_num_dwordx4 = inst_wmma_k_bytes // 16
        self.chunk_num_dwords  = inst_wmma_k_bytes // 4
        self.num_k_chunks      = self.num_dwordx4 // self.chunk_num_dwordx4

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
            self.s_p_wei       = sym_t('s_p_wei'       , sseq(2))    # input (B operand)
            self.s_p_out       = sym_t('s_p_out'       , sseq(2))    # grad_weight (output)
            self.s_gemm_m      = sym_t('s_gemm_m'      , sseq(1))    # K_out
            self.s_gemm_n      = sym_t('s_gemm_n'      , sseq(1))    # C_in
            self.s_gemm_k      = sym_t('s_gemm_k'      , sseq(1))    # N*Ho*Wo
            # stride/pad kernarg fields (Phase 5c) -- for B's (n,ho,wo)->(hi,wi) gather.
            self.s_ho_wo       = sym_t('s_ho_wo'       , sseq(1))    # = ho*wo, divisor for (n, ho*wo) decomposition
            self.s_wo          = sym_t('s_wo'          , sseq(1))    # divisor for (ho_idx, wo_idx) decomposition
            self.s_stride_h    = sym_t('s_stride_h'    , sseq(1))
            self.s_stride_w    = sym_t('s_stride_w'    , sseq(1))
            self.s_pad_h       = sym_t('s_pad_h'       , sseq(1))
            self.s_pad_w       = sym_t('s_pad_w'       , sseq(1))
            self.s_hi          = sym_t('s_hi'          , sseq(1))    # bound for hi_idx
            self.s_wi          = sym_t('s_wi'          , sseq(1))    # bound for wi_idx, and multiplier for hi_idx*wi
            self.s_hi_wi       = sym_t('s_hi_wi'       , sseq(1))    # = s_hi*s_wi, computed on-device once
            # Phase 5f (multi-tap + dilation) kernarg fields, contiguous, matching
            # get_kernel_args()'s trailing layout.
            self.s_y           = sym_t('s_y'           , sseq(1))
            self.s_x           = sym_t('s_x'           , sseq(1))
            self.s_dilation_h  = sym_t('s_dilation_h'  , sseq(1))
            self.s_dilation_w  = sym_t('s_dilation_w'  , sseq(1))
            self.s_wei_row_c   = sym_t('s_wei_row_c'   , sseq(1))    # = gemm_n*x*y, grad_weight's per-K_out-row element count
            self.s_iy          = sym_t('s_iy'          , sseq(1))   # runtime tap-loop counters
            self.s_ix          = sym_t('s_ix'          , sseq(1))
            self.s_p_out_tap   = sym_t('s_p_out_tap'   , sseq(2, 2))   # p_out + this tap's column byte offset, recomputed every tap; even-aligned like s_p_out (used as a VADDR base)
            self.s_group_idx   = sym_t('s_group_idx'   , sseq(1))   # decoded from s_by (group folded into grid_y)
            self.s_group       = sym_t('s_group'       , sseq(1))   # kernarg: total group count
            self.s_a_m_total   = sym_t('s_a_m_total'   , sseq(1))   # = gemm_m*group, A(grad_output)'s per-pixel row stride
            self.s_b_n_total   = sym_t('s_b_n_total'   , sseq(1))   # = gemm_n*group, B(input)'s per-pixel row stride
            self.s_block_m_off = sym_t('s_block_m_off' , sseq(1))
            self.s_block_n_off = sym_t('s_block_n_off' , sseq(1))
            self.s_a_k_stride  = sym_t('s_a_k_stride'  , sseq(1))   # K_out*databyte*gemm_k_per_block: grad_output's per-K-block global stride
            self.s_kitr        = sym_t('s_kitr'        , sseq(1))
            self.s_knum        = sym_t('s_knum'        , sseq(1))
            # gemm_k_global_split (K-split across grid.z): only loaded/used when
            # outer.tunable.gemm_k_global_split is set, but always declared for a uniform
            # register layout between split and non-split kernel variants -- see class
            # docstring / docs/gfx1250_wmma_layout.md.
            self.s_bz             = sym_t('s_bz'             , sseq(1))   # workgroup_id_z -> this workgroup's K-slice index
            self.s_gemm_k_per_wg  = sym_t('s_gemm_k_per_wg'  , sseq(1))   # kernarg: this workgroup's K-slice length
            self.s_gemm_k_wg_off  = sym_t('s_gemm_k_wg_off'  , sseq(1))   # = s_bz * s_gemm_k_per_wg
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
            self.v_c           = sym_t('v_c'           , vseq(outer.tunable.num_vgpr_accumulate_c))     # 128
            self.v_a           = sym_t('v_a'           , vseq(outer.tunable.num_vgpr_accumulate_a))     # 32
            self.v_b           = sym_t('v_b'           , vseq(outer.tunable.num_vgpr_accumulate_b))     # 32
            # Phase 1 (k-sub-loop): sized to outer.chunk_num_dwords (one inst_wmma.k-worth),
            # NOT outer.num_dwords (the whole, possibly multi-substep, row) -- see
            # igemm_fwd_gtc_wmma_nhwc_t's __init__ docstring for why.
            self.v_gld_a       = sym_t('v_gld_a'       , vseq(outer.chunk_num_dwords))   # also
                                                                       # reused as scratch by the transposed shared_load_a
            self.v_gld_b       = sym_t('v_gld_b'       , vseq(outer.chunk_num_dwords))   # ditto, reused by the transposed shared_load_b
            self.v_tid         = sym_t('v_tid'         , vseq(1))
            # 64-bit VADDR pairs must be even-aligned on gfx1250 (verified with llvm-mc)
            self.v_addr_a      = sym_t('v_addr_a'      , vseq(2, 2))    # persistent global A address (64-bit)
            self.v_addr_b      = sym_t('v_addr_b'      , vseq(2, 2))
            # Phase 5f: A's tap-independent base address (row_local/block_m_off/col_start only
            # -- no Y,X dependence), reset into v_addr_a fresh at the start of every tap since
            # move_slice_window_a incrementally bumps v_addr_a across a tap's own K-loop.
            self.v_addr_a_base = sym_t('v_addr_a_base' , vseq(2, 2))
            self.v_addr_out    = sym_t('v_addr_out'    , vseq(2))    # scratch used by coalescing_store_wmma (ping-pong pair)
            self.v_sst_os      = sym_t('v_sst_os'      , vseq(1))    # shared store offset (same for A/B region)
            self.v_sld_a_os    = sym_t('v_sld_a_os'    , vseq(1))    # transposed byte offset (side='m')
            self.v_sld_b_os    = sym_t('v_sld_b_os'    , vseq(1))    # transposed byte offset (side='n')
            self.v_gemm_im     = sym_t('v_gemm_im'     , vseq(1))
            self.v_gemm_in     = sym_t('v_gemm_in'     , vseq(1))
            self.v_tmp         = sym_t('v_tmp'         , vseq(4))
            # Phase 5c (stride/pad): B's (n,ho,wo)->(hi,wi) gather must be recomputed every
            # main-loop iteration (unlike fwd/bwd's kernel-lifetime-constant flag), since
            # GEMM_K here is spatial -- see class docstring. v_row_local/v_b_col_off are
            # persistent (computed once); v_flag is recomputed by move_slice_window_b every
            # iteration; v_gtc_tmp is scratch for that recomputation.
            self.v_row_local   = sym_t('v_row_local'   , vseq(1))    # tid>>2, this thread's fixed row-within-K-block
            self.v_b_col_off   = sym_t('v_b_col_off'   , vseq(1))    # (block_n_off+col_start)*databyte, fixed
            self.v_flag        = sym_t('v_flag'        , vseq(1))
            self.v_gtc_tmp     = sym_t('v_gtc_tmp'     , vseq(5))
            if outer.tunable.atomic_pack_bf16:
                # Phase 34: packed-bf16 atomic epilogue scratch -- v_pk_idx (partner-lane
                # byte-index for ds_bpermute_b32, kernel-lifetime constant once computed),
                # v_pk_partner (cross-lane-exchanged value, per-iteration scratch),
                # v_pk_packed (packed bf16x2 result, per-iteration scratch). Only allocated
                # when atomic_pack_bf16 is set (every existing config byte-identical).
                self.v_pk_idx     = sym_t('v_pk_idx'     , vseq(1))
                self.v_pk_partner = sym_t('v_pk_partner' , vseq(1))
                self.v_pk_packed  = sym_t('v_pk_packed'  , vseq(1))
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
        # Phase 5c (stride/pad): declaration order here must exactly match kernel_sgpr_t's
        # s_ho_wo..s_wi declaration order.
        kas.append(amdgpu_kernel_arg_t('ho_wo'     , 4, 36, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('wo'        , 4, 40, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('stride_h'  , 4, 44, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('stride_w'  , 4, 48, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('pad_h'     , 4, 52, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('pad_w'     , 4, 56, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('hi'        , 4, 60, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('wi'        , 4, 64, 'by_value', 'i32'))
        # Phase 5f (multi-tap + dilation)
        kas.append(amdgpu_kernel_arg_t('y'         , 4, 68, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('x'         , 4, 72, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dilation_h', 4, 76, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dilation_w', 4, 80, 'by_value', 'i32'))
        # Phase 7 (group>1): the only new kernarg needed -- see fwd's Phase 7 docstring for
        # the full rationale (A/B need the tensor's TOTAL channel count for their per-pixel
        # row stride, which requires knowing group itself; output needs no equivalent fix).
        kas.append(amdgpu_kernel_arg_t('group'     , 4, 84, 'by_value', 'i32'))
        # gemm_k_global_split: this workgroup's K-slice length (N*Ho*Wo/splits, an exact
        # multiple of gemm_k_per_block by construction -- see driver-side split-count
        # policy). Always present in the karg layout (even for non-split kernels, which
        # simply never load it) so both variants share one struct on the driver side.
        kas.append(amdgpu_kernel_arg_t('gemm_k_per_wg', 4, 88, 'by_value', 'i32'))
        return kas

    def get_kernel_code(self):
        # see igemm_fwd_gtc_wmma_nhwc_t's identically-named field for the rationale. Only
        # the non-atomic (non-split) epilogue uses the LDS reshuffle -- the gemm_k_global_split
        # atomic path never touches LDS in the epilogue, so it doesn't need the boost.
        # Phase 23: epilogue_lds_pad adds 4 padding elements per row to break a bank-conflict
        # periodicity (see coalescing_store_wmma.py) -- reflect that in the LDS size too.
        epilogue_pad = 4 if self.tunable.epilogue_lds_pad else 0
        # Phase 24: f16acc's epilogue stages genuinely 2-byte-per-element LDS data (see
        # coalescing_store_wmma.py's scatter), half the f32 case's footprint.
        epilogue_elem_bytes = 2 if (self.tunable.wmma_acc_f16 or self.tunable.wmma_acc_bf16) else 4
        epilogue_lds_bytes = 0 if self.tunable.gemm_k_global_split else \
            self.tunable.gemm_m_per_block * (self.tunable.gemm_n_per_block + epilogue_pad) * epilogue_elem_bytes
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
            'kernarg_segment_byte_size'         :   92,
            'wavefront_sgpr_count'              :   self.sgpr.s_end.value + 2 * 3,
            'workitem_vgpr_count'               :   self.vgpr.v_end.value,
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
        rationale (recompute, not save/restore, to avoid needing new VGPRs). '''
        v = self.vgpr
        # ---- fixed per-thread shared-memory store offset: this thread owns byte-chunk
        # tid*bytes_per_row ---- (both A's and B's (row_local,col_group) gemm_k_per_block-
        # element chunk land at the same linear tid position, see class docstring)
        self._emit(f"v_lshlrev_b32 v[{v.v_sst_os()}], {utility_log2(self.bytes_per_row)}, v[{v.v_tid()}]   ; tid*{self.bytes_per_row} bytes")
        self._emit_empty_line()

        # ---- shared-memory load offset for A (TRANSPOSED -- grad_output is [K][M] in LDS) ----
        # row_pitch_bytes = gemm_m_per_block * databyte (128 * databyte); A's region starts at
        # byte 0 within the shared LDS tile (no lds_a_size-style base needed, unlike B below).
        self._emit(self.wmma_mapping.get_gemm_index_for_src_matrix_transposed(v.v_sld_a_os(), v.v_tid(), v.v_tmp(),
                self.tunable.gemm_m_per_block * self.data_byte, self.data_byte, 'm'))
        self._emit_empty_line()

        # ---- shared-memory load offset for B (TRANSPOSED -- input is [K][N] in LDS) ----
        # row_pitch_bytes = gemm_n_per_block * databyte (128 * databyte)
        self._emit(self.wmma_mapping.get_gemm_index_for_src_matrix_transposed(v.v_sld_b_os(), v.v_tid(), v.v_tmp(),
                self.tunable.gemm_n_per_block * self.data_byte, self.data_byte, 'n'))
        self._emit_empty_line()

    def emit_kernel_prologue(self):
        s = self.sgpr
        v = self.vgpr
        self._emit(f"s_load_dwordx4 s[{s.s_p_in()}:{s.s_p_in(3)}], s[{s.s_ka()}:{s.s_ka(1)}], 0")
        self._emit(f"s_load_dwordx4 s[{s.s_p_out()}:{s.s_p_out(3)}], s[{s.s_ka()}:{s.s_ka(1)}], 16")
        self._emit(f"s_load_dword s[{s.s_gemm_k()}], s[{s.s_ka()}:{s.s_ka(1)}], 32")
        # individual s_load_dword (not dwordx4) -- these SGPRs aren't guaranteed 4-aligned
        self._emit(f"s_load_dword s[{s.s_ho_wo()}], s[{s.s_ka()}:{s.s_ka(1)}], 36")
        self._emit(f"s_load_dword s[{s.s_wo()}], s[{s.s_ka()}:{s.s_ka(1)}], 40")
        self._emit(f"s_load_dword s[{s.s_stride_h()}], s[{s.s_ka()}:{s.s_ka(1)}], 44")
        self._emit(f"s_load_dword s[{s.s_stride_w()}], s[{s.s_ka()}:{s.s_ka(1)}], 48")
        self._emit(f"s_load_dword s[{s.s_pad_h()}], s[{s.s_ka()}:{s.s_ka(1)}], 52")
        self._emit(f"s_load_dword s[{s.s_pad_w()}], s[{s.s_ka()}:{s.s_ka(1)}], 56")
        self._emit(f"s_load_dword s[{s.s_hi()}], s[{s.s_ka()}:{s.s_ka(1)}], 60")
        self._emit(f"s_load_dword s[{s.s_wi()}], s[{s.s_ka()}:{s.s_ka(1)}], 64")
        self._emit(f"s_load_dword s[{s.s_y()}], s[{s.s_ka()}:{s.s_ka(1)}], 68")
        self._emit(f"s_load_dword s[{s.s_x()}], s[{s.s_ka()}:{s.s_ka(1)}], 72")
        self._emit(f"s_load_dword s[{s.s_dilation_h()}], s[{s.s_ka()}:{s.s_ka(1)}], 76")
        self._emit(f"s_load_dword s[{s.s_dilation_w()}], s[{s.s_ka()}:{s.s_ka(1)}], 80")
        self._emit(f"s_load_dword s[{s.s_group()}], s[{s.s_ka()}:{s.s_ka(1)}], 84")
        if self.tunable.gemm_k_global_split:
            self._emit(f"s_load_dword s[{s.s_gemm_k_per_wg()}], s[{s.s_ka()}:{s.s_ka(1)}], 88")
        self._emit(f"v_mov_b32 v[{v.v_tid()}], v0")
        # gfx1250 delivers workgroup id via ttmp9/ttmp7 -- see docs/gfx1250_wmma_layout.md.
        # ttmp9 is a clean, unpacked blockIdx.x. ttmp7 PACKS blockIdx.y (low 16 bits) and
        # blockIdx.z (high 16 bits) -- confirmed by an inline-asm probe comparing raw ttmp
        # reads against the compiler's own blockIdx ground truth on real hardware (see
        # docs/gfx1250_wmma_layout.md's gemm_k_global_split phase). A plain `s_mov_b32
        # s_by, ttmp7` (as every other WMMA kernel still does) is only correct when grid.z
        # is always 1 -- true everywhere except this kernel's split variant, which is why
        # only this decode needs the mask/shift split.
        self._emit(f"s_mov_b32 s[{s.s_bx()}], ttmp9")
        if self.tunable.gemm_k_global_split:
            self._emit(f"s_and_b32 s[{s.s_by()}], ttmp7, 0xffff")
            self._emit(f"s_lshr_b32 s[{s.s_bz()}], ttmp7, 16")
        else:
            self._emit(f"s_mov_b32 s[{s.s_by()}], ttmp7")
        self._emit(f"s_wait_kmcnt 0x0")
        if self.tunable.gemm_k_global_split:
            self._emit(f"s_mul_i32 s[{s.s_gemm_k_wg_off()}], s[{s.s_bz()}], s[{s.s_gemm_k_per_wg()}]   ; this workgroup's K-slice base")
        self._emit_empty_line()

        m_int_div_rem_vs = macro_int_div_rem_vs_gfx1250_t(self.mc)

        # ---- group>1: decode group_idx out of s_by and correct s_by in place BEFORE
        # s_block_n_off is computed from it below -- see igemm_fwd_gtc_wmma_nhwc_t's Phase 7
        # docstring for the full rationale. A (grad_output, GEMM_M=K_per_group) and B (input,
        # GEMM_N=C_per_group) are BOTH NHWC tensors with group INTERLEAVED within each pixel's
        # channel dimension (wrw is the one direction where BOTH operands need this, since
        # both its GEMM_M and GEMM_N are per-group), so their per-pixel row stride must be the
        # TOTAL channel count (gemm_m*group / gemm_n*group), not the per-group gemm_m/gemm_n
        # used for the base-pointer offset. Output (grad_weight) needs no equivalent fix --
        # its group split is the outermost/block-contiguous dimension. ----
        self._emit(f"; --- group>1: decode group_idx, correct s_by, offset A/B base pointers ---")
        self._emit(f"s_add_u32 s[{s.s_tmp(0)}], s[{s.s_gemm_n()}], {self.tunable.gemm_n_per_block - 1}")
        self._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], {utility_log2(self.tunable.gemm_n_per_block)}   ; blocks_per_group_n = ceil(gemm_n/gemm_n_per_block)")
        self._emit(f"v_mov_b32 v[{v.v_gtc_tmp(0)}], s[{s.s_by()}]")
        self._emit(m_int_div_rem_vs(v.v_gtc_tmp(1), v.v_gtc_tmp(2), v.v_gtc_tmp(0), s.s_tmp(0), v.v_tmp(), s.s_tmp(1)))
        self._emit(f"v_readfirstlane_b32 s[{s.s_group_idx()}], v[{v.v_gtc_tmp(2)}]   ; group_idx")
        self._emit(f"v_readfirstlane_b32 s[{s.s_by()}], v[{v.v_gtc_tmp(1)}]   ; s_by <- corrected within-group N-block index")
        self._emit_empty_line()

        self._emit(f"s_mul_i32 s[{s.s_a_m_total()}], s[{s.s_gemm_m()}], s[{s.s_group()}]")
        self._emit(f"s_mul_i32 s[{s.s_b_n_total()}], s[{s.s_gemm_n()}], s[{s.s_group()}]")
        self._emit_empty_line()

        self._emit(f"; A (grad_output): group offset = group_idx * gemm_m elements")
        self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_group_idx()}], s[{s.s_gemm_m()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], {utility_log2(self.data_byte)}")
        self._emit(f"s_add_u32 s[{s.s_p_in()}], s[{s.s_p_in()}], s[{s.s_tmp(0)}]")
        self._emit(f"s_addc_u32 s[{s.s_p_in(1)}], s[{s.s_p_in(1)}], 0")

        self._emit(f"; B (input): group offset = group_idx * gemm_n elements")
        self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_group_idx()}], s[{s.s_gemm_n()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], {utility_log2(self.data_byte)}")
        self._emit(f"s_add_u32 s[{s.s_p_wei()}], s[{s.s_p_wei()}], s[{s.s_tmp(0)}]")
        self._emit(f"s_addc_u32 s[{s.s_p_wei(1)}], s[{s.s_p_wei(1)}], 0")
        self._emit_empty_line()

        self._emit(f"s_lshl_b32 s[{s.s_block_m_off()}], s[{s.s_bx()}], {utility_log2(self.tunable.gemm_m_per_block)}   ; *gemm_m_per_block")
        self._emit(f"s_lshl_b32 s[{s.s_block_n_off()}], s[{s.s_by()}], {utility_log2(self.tunable.gemm_n_per_block)}   ; *gemm_n_per_block")
        if self.tunable.gemm_k_global_split:
            self._emit(f"s_mov_b32 s[{s.s_knum()}], s[{s.s_gemm_k_per_wg()}]   ; this workgroup only reduces its own K-slice")
        else:
            self._emit(f"s_mov_b32 s[{s.s_knum()}], s[{s.s_gemm_k()}]")
        # A's per-K-block (32 rows) global stride depends on a runtime tensor extent (K_out) --
        # unlike fwd's compile-time-constant +64 bytes. B has no equivalent constant stride
        # anymore (Phase 5c): its address is recomputed fresh every iteration (see
        # move_slice_window_b_functor), since the (n,ho,wo)->(hi,wi) gather changes every block.
        # NOTE: gemm_k_per_block is 32 for fp16/bf16 but 64 for int8 -- must NOT hardcode a
        # "+5" (*32) shift here, or A's per-K-block stride silently undercounts by 2x for int8.
        # group>1: uses s_a_m_total (=gemm_m*group), NOT plain gemm_m -- A's per-row width in
        # its real physical [K][M_total] storage is the TOTAL K_out count, same reasoning as
        # the per-pixel row stride above.
        self._emit(f"s_lshl_b32 s[{s.s_a_k_stride()}], s[{s.s_a_m_total()}], {utility_log2(self.data_byte * self.tunable.gemm_k_per_block)}   ; a_m_total * databyte * {self.tunable.gemm_k_per_block}")
        self._emit(f"s_mul_i32 s[{s.s_hi_wi()}], s[{s.s_hi()}], s[{s.s_wi()}]")
        # grad_weight's per-K_out-row element count is y*x*c (not just c as in the 1x1 case) --
        # reused for the epilogue's row stride (Phase 5f). B (input) itself has no y/x
        # dependence in its own storage, only in which pixel the per-iteration gather reads.
        self._emit(f"s_mul_i32 s[{s.s_wei_row_c()}], s[{s.s_x()}], s[{s.s_y()}]")
        self._emit(f"s_mul_i32 s[{s.s_wei_row_c()}], s[{s.s_wei_row_c()}], s[{s.s_gemm_n()}]")
        self._emit_empty_line()

        # ---- group>1: output (grad_weight) group offset = group_idx * gemm_m * wei_row_c
        # elements (the per-group output tensor size, K_per_group*Y*X*C_per_group =
        # gemm_m*wei_row_c) -- applied here (not alongside A/B above) since it needs
        # s_wei_row_c, just computed. Added directly to s_p_out (not s_p_out_tap, which is
        # recomputed fresh every tap FROM s_p_out -- see emit_kernel_tap_loop), so every tap's
        # per-tap offset automatically inherits this block-constant group shift. ----
        # Phase 24: shift must follow the D-operand's real width (4 bytes normally, 2 under
        # wmma_acc_f16) -- same bug class as emit_kernel_tap_loop's per-tap offset above.
        # Phase 34: atomic_pack_bf16 is a THIRD case with a 2-byte-native output buffer,
        # same shift as wmma_acc_f16/bf16 even though it's a completely different
        # mechanism (packed-atomic epilogue, not a packed-accumulator VGPR layout) --
        # found via a real hardware miscompare on group>1 (g=1 has a zero group offset
        # regardless of this shift's value, masking the bug entirely).
        out_elem_byte_shift_group = 1 if (self.tunable.wmma_acc_f16 or self.tunable.wmma_acc_bf16 or self.tunable.atomic_pack_bf16) else 2
        self._emit(f"; output (grad_weight): group offset = group_idx * gemm_m * wei_row_c elements")
        self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_group_idx()}], s[{s.s_gemm_m()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], s[{s.s_wei_row_c()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], {out_elem_byte_shift_group}   ; D-operand is fp32/int32 (4B) normally, fp16 (2B) under wmma_acc_f16")
        self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp(0)}]")
        self._emit(f"s_addc_u32 s[{s.s_p_out(1)}], s[{s.s_p_out(1)}], 0")
        self._emit_empty_line()

        # ---- global address for this thread's chunk of the A tile (grad_output, natural [GEMM_K][GEMM_M]) ----
        # thread tid owns row_local (one of gemm_k_per_block rows of this K-block) and
        # col_group (one of num_col_groups chunks of the 128-wide K_out tile, each chunk
        # exactly gemm_k_per_block elements wide -- see bwd's Phase-int8 fix docstring for why
        # this can't stay hardcoded at tid>>2/tid&3/col_group*32: that's only correct when
        # gemm_k_per_block==32, i.e. fp16/bf16, not int8's 64). Tap-independent (grad_output
        # has no Y,X extent), so computed once into the persistent v_addr_a_base and reset into
        # v_addr_a fresh at the start of every tap (Phase 5f).
        num_col_groups = self.tunable.gemm_m_per_block // self.tunable.gemm_k_per_block
        col_group_bits = utility_log2(num_col_groups)
        col_start_shift = utility_log2(self.tunable.gemm_k_per_block)
        self._emit(f"; v_addr_a_base = p_in + (row_local*K_out + block_m_off + col_start) * databyte")
        self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}], {col_group_bits}, v[{v.v_tid()}]        ; row_local = tid>>{col_group_bits}")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_a_m_total()}], v[{v.v_tmp()}]  ; row_local * a_m_total (K_out TOTAL, not per-group)")
        self._emit(f"v_and_b32 v[{v.v_tmp(1)}], {num_col_groups - 1}, v[{v.v_tid()}]           ; col_group = tid&{num_col_groups - 1}")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp(1)}], {col_start_shift}, v[{v.v_tmp(1)}]      ; col_start = col_group*{self.tunable.gemm_k_per_block}")
        self._emit(f"v_add_u32 v[{v.v_tmp()}], v[{v.v_tmp(1)}], v[{v.v_tmp()}]")
        self._emit(f"v_add_u32 v[{v.v_tmp()}], s[{s.s_block_m_off()}], v[{v.v_tmp()}]")
        if self.tunable.gemm_k_global_split:
            self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_gemm_k_wg_off()}], s[{s.s_a_m_total()}]   ; this workgroup's K-slice base, in A row units")
            self._emit(f"v_add_u32 v[{v.v_tmp()}], s[{s.s_tmp(2)}], v[{v.v_tmp()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {utility_log2(self.data_byte)}, v[{v.v_tmp()}]")
        self._emit(f"v_mov_b32 v[{v.v_addr_a_base(1)}], s[{s.s_p_in(1)}]")
        self._emit(f"v_add_co_u32 v[{v.v_addr_a_base()}], vcc_lo, s[{s.s_p_in()}], v[{v.v_tmp()}]")
        self._emit(f"v_add_co_ci_u32 v[{v.v_addr_a_base(1)}], vcc_lo, 0, v[{v.v_addr_a_base(1)}], vcc_lo")
        self._emit_empty_line()

        # ---- persistent setup for B's per-iteration gather (input, gathered through
        # stride/pad -- see class docstring). Same num_col_groups/col_group_bits/
        # col_start_shift as A above (A and B are both 128-wide tiles here). ----
        self._emit(f"v_lshrrev_b32 v[{v.v_row_local()}], {col_group_bits}, v[{v.v_tid()}]        ; row_local = tid>>{col_group_bits} (fixed for the whole kernel)")
        self._emit(f"; v_b_col_off = (block_n_off + col_start) * databyte (fixed for the whole kernel)")
        self._emit(f"v_and_b32 v[{v.v_tmp()}], {num_col_groups - 1}, v[{v.v_tid()}]            ; col_group = tid&{num_col_groups - 1}")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {col_start_shift}, v[{v.v_tmp()}]        ; col_start = col_group*{self.tunable.gemm_k_per_block}")
        self._emit(f"v_add_u32 v[{v.v_tmp()}], s[{s.s_block_n_off()}], v[{v.v_tmp()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_b_col_off()}], {utility_log2(self.data_byte)}, v[{v.v_tmp()}]")
        self._emit_empty_line()

        self._emit_lds_offset_setup()

        # ---- persistent (im, in) for the epilogue, converted to global output indices ----
        self._emit(self.wmma_mapping.get_gemm_index_for_dst_matrix(v.v_gemm_in(), v.v_gemm_im(), v.v_tid(), v.v_tmp()))
        self._emit(f"v_add_u32 v[{v.v_gemm_im()}], s[{s.s_block_m_off()}], v[{v.v_gemm_im()}]")
        self._emit(f"v_add_u32 v[{v.v_gemm_in()}], s[{s.s_block_n_off()}], v[{v.v_gemm_in()}]")
        self._emit_empty_line()

        # ---- Phase 5f: runtime tap-loop counter, initialized once before the loop ----
        self._emit(f"s_mov_b32 s[{s.s_iy()}], 0")
        self._emit_empty_line()

    def emit_kernel_tap_loop(self):
        '''
        Runtime (not compile-time-unrolled) outer loop over the y*x filter taps -- see class
        docstring for why wrw's version is structurally different from fwd/bwd's: each tap
        produces a DIFFERENT, INDEPENDENT slice of the output (grad_weight), so v_c must be
        zeroed and the epilogue must fire ONCE PER TAP, not once at the very end. The WMMA
        K-main-loop is still emitted EXACTLY ONCE (the runtime branch at the bottom re-enters
        it for every tap), same mechanism as fwd/bwd.
        '''
        s = self.sgpr
        v = self.vgpr
        label_tap_y = f"L_{self.name()}_tap_y"
        label_tap_x = f"L_{self.name()}_tap_x"
        self._emit_front(f"{label_tap_y}:")
        self._emit(f"s_mov_b32 s[{s.s_ix()}], 0")
        self._emit_front(f"{label_tap_x}:")

        if self.lds_buffer_num == 2:
            # Phase 2 (double-buffering): recompute the fresh, untoggled state at the
            # start of EVERY tap -- see _emit_lds_offset_setup's docstring for why.
            self._emit_lds_offset_setup()

        # ---- zero the accumulator fresh for this tap (grad_weight's [iy,ix] slice is an
        # independent reduction, not a running sum across taps) ----
        self._emit(f"; clear accumulator (fresh per tap, see class docstring)")
        for i in range(self.tunable.num_vgpr_accumulate_c):
            self._emit(f"v_mov_b32 v[{v.v_c(i)}], 0")
        self._emit_empty_line()

        # ---- reset A's address to its tap-independent base (move_slice_window_a bumped it
        # across the PREVIOUS tap's K-loop) ----
        self._emit(f"v_mov_b32 v[{v.v_addr_a()}], v[{v.v_addr_a_base()}]")
        self._emit(f"v_mov_b32 v[{v.v_addr_a(1)}], v[{v.v_addr_a_base(1)}]")
        self._emit_empty_line()

        # ---- B's initial (k_block_off=0) gather for this tap's first global load -- reads
        # the CURRENT s_iy/s_ix live, so no special-casing needed here ----
        self._emit_b_gather(None)
        self._emit_empty_line()

        # ---- issue the first global loads for this tap (main loop expects this precondition) ----
        self._emit(self.global_load_a_functor()())
        self._emit(self.global_load_b_functor()())
        self._emit_empty_line()

        # ---- the WMMA K-main-loop over N*Ho*Wo, emitted EXACTLY ONCE here; the runtime
        # branches below re-enter this same code for every tap ----
        self.emit_kernel_fma_main_loop()
        self._emit_empty_line()

        # ---- store this tap's v_c to grad_weight's own [iy,ix] slice: row stride becomes
        # wei_row_c (y*x*c) instead of plain c, and the tap's column-block offset is folded
        # into a fresh per-tap base pointer (s_p_out_tap) rather than perturbing per-thread
        # column indices, so coalescing_store_wmma.py itself needs no changes ----
        # Phase 24 bug found via hardware validation (wrw f16acc gave valid:n while fwd/bwd
        # passed): this shift was hardcoded to 2 (WMMA D-operand assumed always fp32/int32,
        # 4 bytes) -- true for every direction EXCEPT wrw's own per-tap output offset, which
        # is the only place outside coalescing_store_wmma.py that independently computes a
        # byte address into the WMMA-native output buffer. f16acc halves that buffer's
        # element width to 2 bytes, so this must shift by 1, not 2, in that case.
        # atomic_pack_bf16 (Phase 34) is the same 2-byte-native case for a different reason
        # (packed-atomic epilogue writes bf16 directly) -- see out_elem_byte_shift_group above.
        out_elem_byte_shift = 1 if (self.tunable.wmma_acc_f16 or self.tunable.wmma_acc_bf16 or self.tunable.atomic_pack_bf16) else 2
        self._emit(f"; s_p_out_tap = p_out + (iy*x+ix)*gemm_n*{1 << out_elem_byte_shift} bytes (WMMA D-operand is")
        self._emit(f"; fp32/int32 (4B) normally, or fp16 (2B) under wmma_acc_f16 -- see coalescing_store_wmma.py)")
        self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_iy()}], s[{s.s_x()}]")
        self._emit(f"s_add_i32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], s[{s.s_ix()}]   ; tap linear index")
        self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], s[{s.s_gemm_n()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], {out_elem_byte_shift}   ; tap byte offset")
        self._emit(f"s_add_u32 s[{s.s_p_out_tap()}], s[{s.s_p_out()}], s[{s.s_tmp(2)}]")
        self._emit(f"s_addc_u32 s[{s.s_p_out_tap(1)}], s[{s.s_p_out(1)}], 0")
        # Phase 34: atomic_pack_bf16 needs genuine scratch for v_gather/v_tmp3/v_tmp4 (the
        # non-packed atomic path never touches them, so v_c() is passed as a harmless
        # unused placeholder there -- see coalescing_store_wmma.py's docstring).
        if self.tunable.atomic_pack_bf16:
            self._emit(self.coalescing_store(v.v_c.label, v.v_gemm_im(), v.v_gemm_in(), s.s_p_out_tap.label, s.s_wei_row_c.label, v.v_addr_out(), v.v_addr_out(1), s.s_tmp(), v.v_tid(), v.v_pk_idx(), s.s_block_m_off(), s.s_block_n_off(), None, v.v_pk_partner(), None, v.v_pk_packed()))
        else:
            self._emit(self.coalescing_store(v.v_c.label, v.v_gemm_im(), v.v_gemm_in(), s.s_p_out_tap.label, s.s_wei_row_c.label, v.v_addr_out(), v.v_addr_out(1), s.s_tmp(), v.v_tid(), v.v_c(), s.s_block_m_off(), s.s_block_n_off()))
        self._emit(f"s_wait_storecnt 0x0")
        self._emit_empty_line()

        self._emit(f"s_add_u32 s[{s.s_ix()}], s[{s.s_ix()}], 1")
        self._emit(f"s_cmp_lt_u32 s[{s.s_ix()}], s[{s.s_x()}]")
        self._emit(f"s_cbranch_scc1 {label_tap_x}")
        self._emit(f"s_add_u32 s[{s.s_iy()}], s[{s.s_iy()}], 1")
        self._emit(f"s_cmp_lt_u32 s[{s.s_iy()}], s[{s.s_y()}]")
        self._emit(f"s_cbranch_scc1 {label_tap_y}")
        self._emit_empty_line()

    def _emit_b_gather(self, s_k_block_off):
        '''
        Computes this thread's absolute K-index (k_block_off + row_local, where row_local is
        the persistent per-thread constant computed in the prologue), decomposes it into
        (n_idx, ho_idx, wo_idx) [output/grad_output space, matching GEMM_K's own definition],
        gathers the corresponding (possibly out-of-bounds) input pixel (hi_idx, wi_idx) via the
        same stride/pad formula fwd's Phase 5a uses, sets v_flag, and writes v_addr_b. Called
        once from emit_kernel_tap_loop at the start of every tap (s_k_block_off=None, meaning
        k_block_off=0) and once per main-loop iteration from move_slice_window_b_functor
        (s_k_block_off = the SGPR holding the freshly-computed s_knum-s_kitr) -- see class
        docstring for why this must be recomputed every iteration. Reads the CURRENT s_iy/s_ix
        live (Phase 5f's tap bias), so no special-casing is needed for either call site to
        pick up whichever tap is currently active.
        '''
        s = self.sgpr
        v = self.vgpr
        m_int_div_rem_vs = macro_int_div_rem_vs_gfx1250_t(self.mc)
        if s_k_block_off is not None:
            self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], s[{s_k_block_off}], v[{v.v_row_local()}]   ; k_abs (within this workgroup's K-slice)")
        else:
            self._emit(f"v_mov_b32 v[{v.v_gtc_tmp(0)}], v[{v.v_row_local()}]   ; k_abs (k_block_off=0, within this workgroup's K-slice)")
        if self.tunable.gemm_k_global_split:
            self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(0)}], s[{s.s_gemm_k_wg_off()}]   ; += this workgroup's K-slice base")
        self._emit(m_int_div_rem_vs(v.v_gtc_tmp(1), v.v_gtc_tmp(2), v.v_gtc_tmp(0), s.s_ho_wo(), v.v_tmp(), s.s_tmp()))
        self._emit(f"; v_gtc_tmp(1)=hw_idx (rem), v_gtc_tmp(2)=n_idx (quo)")
        self._emit(m_int_div_rem_vs(v.v_gtc_tmp(3), v.v_gtc_tmp(4), v.v_gtc_tmp(1), s.s_wo(), v.v_tmp(), s.s_tmp()))
        self._emit(f"; v_gtc_tmp(3)=wo_idx (rem), v_gtc_tmp(4)=ho_idx (quo)")
        self._emit_empty_line()

        self._emit(f"; hi_idx = ho_idx*stride_h - pad_h + iy*dilation_h ; wi_idx symmetric (Phase 5f tap bias)")
        self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_iy()}], s[{s.s_dilation_h()}]")
        self._emit(f"s_sub_i32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], s[{s.s_pad_h()}]   ; iy*dilation_h - pad_h")
        self._emit(f"s_mul_i32 s[{s.s_tmp(3)}], s[{s.s_ix()}], s[{s.s_dilation_w()}]")
        self._emit(f"s_sub_i32 s[{s.s_tmp(3)}], s[{s.s_tmp(3)}], s[{s.s_pad_w()}]   ; ix*dilation_w - pad_w")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(4)}], s[{s.s_stride_h()}], v[{v.v_gtc_tmp(4)}]")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(4)}], v[{v.v_gtc_tmp(4)}], s[{s.s_tmp(2)}]   ; hi_idx")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(3)}], s[{s.s_stride_w()}], v[{v.v_gtc_tmp(3)}]")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(3)}], v[{v.v_gtc_tmp(3)}], s[{s.s_tmp(3)}]   ; wi_idx")
        self._emit_empty_line()

        self._emit(f"; v_flag = 1 iff (hi_idx, wi_idx) in [0,hi)x[0,wi)")
        self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_hi()}], v[{v.v_gtc_tmp(4)}]")
        self._emit(f"v_cndmask_b32 v[{v.v_flag()}], 0, 1, vcc_lo")
        self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_wi()}], v[{v.v_gtc_tmp(3)}]")
        self._emit(f"v_cndmask_b32 v[{v.v_flag()}], 0, v[{v.v_flag()}], vcc_lo")
        self._emit_empty_line()

        self._emit(f"; row_idx = n_idx*(hi*wi) + hi_idx*wi + wi_idx (meaningless but harmless if")
        self._emit(f"; v_flag==0 -- that lane's global_load_b is EXEC-masked off, see global_load_b_functor)")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(0)}], s[{s.s_hi_wi()}], v[{v.v_gtc_tmp(2)}]")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(2)}], s[{s.s_wi()}], v[{v.v_gtc_tmp(4)}]")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(2)}]")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(3)}]")
        self._emit_empty_line()

        self._emit(f"; v_addr_b = p_wei + row_idx * b_n_total * databyte + v_b_col_off (b_n_total = gemm_n*group --")
        self._emit(f"; input's pixel-to-pixel stride is its TOTAL C_in count, not the per-group gemm_n, see class docstring)")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(0)}], s[{s.s_b_n_total()}], v[{v.v_gtc_tmp(0)}]   ; row_idx * b_n_total")
        self._emit(f"v_lshlrev_b32 v[{v.v_gtc_tmp(0)}], {utility_log2(self.data_byte)}, v[{v.v_gtc_tmp(0)}]")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(0)}], v[{v.v_b_col_off()}]")
        self._emit(f"v_mov_b32 v[{v.v_addr_b(1)}], s[{s.s_p_wei(1)}]")
        self._emit(f"v_add_co_u32 v[{v.v_addr_b()}], vcc_lo, s[{s.s_p_wei()}], v[{v.v_gtc_tmp(0)}]")
        self._emit(f"v_add_co_ci_u32 v[{v.v_addr_b(1)}], vcc_lo, 0, v[{v.v_addr_b(1)}], vcc_lo")

    def _emit_gld_chunk_load(self, v_gld, v_addr, chunk_idx, v_flag=None):
        ''' Phase 1 (k-sub-loop): issues (does not wait) ONE inst_wmma.k-wide chunk's
        global load into the small, reused v_gld buffer. '''
        if v_flag is not None:
            for i in range(self.chunk_num_dwords):
                self._emit(f"v_mov_b32 v[{v_gld(i)}], 0")
            self._emit(f"v_cmpx_le_u32 1, v[{v_flag()}]")
        for i in range(self.chunk_num_dwordx4):
            idx = chunk_idx * self.chunk_num_dwordx4 + i
            self._emit(f"global_load_dwordx4 v[{v_gld(i*4)}:{v_gld(i*4+3)}], v[{v_addr()}:{v_addr(1)}], off offset:{idx*16}")
        if v_flag is not None:
            self._emit(f"s_mov_b32 exec_lo, -1")

    def _emit_sst_chunk(self, v_gld, v_sst_os, sst_extra_off, chunk_idx):
        ''' Phase 1 (k-sub-loop): stores ONE already-loaded-and-waited chunk to LDS. '''
        for i in range(self.chunk_num_dwordx4):
            idx = chunk_idx * self.chunk_num_dwordx4 + i
            self._emit(f"ds_write_b128 v[{v_sst_os()}], v[{v_gld(i*4)}:{v_gld(i*4+3)}] offset:{sst_extra_off + idx*16}")

    def _emit_sst_remaining_chunks(self, v_gld, v_addr, v_sst_os, sst_extra_off, v_flag=None):
        '''
        Phase 1 (k-sub-loop): stores chunk 0 (already loaded+waited via the existing
        global_load_a/b_functor + outer s_wait_loadcnt call sequence in
        wmma_main_loop.py), then load+wait+stores chunks 1..num_k_chunks-1 sequentially,
        reusing the same small v_gld buffer -- see igemm_fwd_gtc_wmma_nhwc_t's identically-
        named method for why (preserves the single-buffered-LDS safety invariant: no wave
        may overwrite a tile's LDS storage until every wave has finished reading it).
        '''
        self._emit_sst_chunk(v_gld, v_sst_os, sst_extra_off, 0)
        for c in range(1, self.num_k_chunks):
            self._emit_gld_chunk_load(v_gld, v_addr, c, v_flag=v_flag)
            self._emit(f"s_wait_loadcnt 0x0")
            self._emit_sst_chunk(v_gld, v_sst_os, sst_extra_off, c)

    def _emit_sst_all_chunks(self, v_gld, v_addr, v_sst_os, sst_extra_off, v_flag=None):
        '''
        Phase 1 (k-sub-loop): like _emit_sst_remaining_chunks, but load+wait+stores ALL
        num_k_chunks chunks here (including chunk 0 -- global_load_a/b_functor issues no
        early load at all for this operand). Needed here since BOTH of wrw's operands are
        TRANSPOSED and reuse v_gld_a/v_gld_b as scratch for their read-and-pack technique,
        called again (for substep>=1) via emit_extra_substeps() -- which runs AFTER
        global_load_a/b_functor but BEFORE shared_store_a/b_functor. An early chunk-0 load
        surviving in v_gld across that window would get silently clobbered by that scratch
        reuse before ever being stored -- see igemm_bwd_gtc_wmma_nhwc_t's identically-named
        method for the full incident writeup (caught on real hardware as silent
        wrong-answer corruption, distinct from the cross-wave LDS-overwrite race
        _emit_sst_remaining_chunks itself guards against).
        '''
        for c in range(self.num_k_chunks):
            self._emit_gld_chunk_load(v_gld, v_addr, c, v_flag=v_flag)
            self._emit(f"s_wait_loadcnt 0x0")
            self._emit_sst_chunk(v_gld, v_sst_os, sst_extra_off, c)

    def global_load_a_functor(self):
        ''' Phase 1: only issues chunk 0's load when num_k_chunks==1 (no clobbering risk
        then, since emit_extra_substeps() never runs) -- see _emit_sst_all_chunks. '''
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                with outer._deferred_context():
                    if outer.num_k_chunks == 1:
                        outer._emit_gld_chunk_load(v.v_gld_a, v.v_addr_a, 0, v_flag=None)
                return outer._get_deferred()
            def get_issues(self):
                return outer.chunk_num_dwordx4
        return functor_t()

    def global_load_b_functor(self):
        '''
        Unlike fwd/bwd's Phase 5a/5b (where a padding lane's flag is constant for the whole
        kernel, so pre-zeroing v_gld_a ONCE in the prologue is correct), B's gather here is
        recomputed every iteration (see class docstring), so v_gld_b must be explicitly
        re-zeroed EVERY call before the masked load -- otherwise a lane that was valid on a
        previous iteration but is invalid (out of bounds) on this one would silently reuse its
        stale (non-zero) data instead of contributing zero. Phase 1: only issues chunk 0's
        load when num_k_chunks==1 -- see _emit_sst_all_chunks.
        '''
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                with outer._deferred_context():
                    if outer.num_k_chunks == 1:
                        outer._emit_gld_chunk_load(v.v_gld_b, v.v_addr_b, 0, v_flag=v.v_flag)
                return outer._get_deferred()
            def get_issues(self):
                return outer.chunk_num_dwordx4
        return functor_t()

    def shared_store_a_functor(self):
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                with outer._deferred_context():
                    if outer.num_k_chunks == 1:
                        outer._emit_sst_remaining_chunks(v.v_gld_a, v.v_addr_a, v.v_sst_os, 0, v_flag=None)
                    else:
                        outer._emit_sst_all_chunks(v.v_gld_a, v.v_addr_a, v.v_sst_os, 0, v_flag=None)
                return outer._get_deferred()
            def get_issues(self):
                return outer.chunk_num_dwordx4
        return functor_t()

    def shared_store_b_functor(self):
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                with outer._deferred_context():
                    if outer.num_k_chunks == 1:
                        outer._emit_sst_remaining_chunks(v.v_gld_b, v.v_addr_b, v.v_sst_os, outer.lds_a_size, v_flag=v.v_flag)
                    else:
                        outer._emit_sst_all_chunks(v.v_gld_b, v.v_addr_b, v.v_sst_os, outer.lds_a_size, v_flag=v.v_flag)
                return outer._get_deferred()
            def get_issues(self):
                return outer.chunk_num_dwordx4
        return functor_t()

    def shared_load_a_functor(self):
        '''
        Transposed read for grad_output: LDS holds it as natural [K rows][M cols]
        (row_pitch = gemm_m_per_block*databyte bytes). Same technique as bwd's
        shared_load_b_functor (see that file for the detailed explanation, including why
        int8's elem_per_dword=4 case processes one vgpr index `a` at a time with a small
        scratch slice instead of batching all 8 reads before waiting, and why fp32 needs a
        third ds_read_b32 read_instr branch plus a `num_v_a`-derived loop bound/index instead
        of the hardcoded 8 that fp16/bf16/int8 all happen to share), applied here to the A
        operand instead of B, and with A's region based at LDS byte 0 (not lds_a_size).
        '''
        outer = self
        class functor_t:
            def __call__(self, v_dst, v_os, extra_off, slot=0):
                v = outer.vgpr
                num_v_a = outer.wmma_mapping.ctrl.inst_wmma.num_v_a
                num_v_a_total = outer.tunable.wmma_repeat_m * num_v_a   # Phase 22: one local_prefetch_num slot's worth
                slot_off = slot * num_v_a_total
                row_pitch = outer.tunable.gemm_m_per_block * outer.data_byte
                elem_per_dword = 4 // outer.data_byte
                if outer.data_byte == 2:
                    read_instr = 'ds_read_u16'
                elif outer.data_byte == 4:
                    read_instr = 'ds_read_b32'   # fp32: full-dword read, no zero-extension needed
                else:
                    read_instr = 'ds_read_u8'
                with outer._deferred_context():
                    for i_rm in range(outer.tunable.wmma_repeat_m):
                        col_off = i_rm * outer.tunable.wmma_tile_m * outer.data_byte
                        for a in range(num_v_a):
                            for s in range(elem_per_dword):
                                off = col_off + (a * elem_per_dword + s) * row_pitch
                                outer._emit(f"{read_instr} v[{v.v_gld_a(s)}], v[{v.v_sld_a_os()}] offset:{extra_off + off}")
                            outer._emit(f"s_wait_dscnt 0x0")
                            outer._emit(f"v_mov_b32 v[{v.v_a(slot_off+i_rm*num_v_a+a)}], v[{v.v_gld_a(0)}]")
                            for s in range(1, elem_per_dword):
                                shift = s * 8 * outer.data_byte
                                outer._emit(f"v_lshl_or_b32 v[{v.v_a(slot_off+i_rm*num_v_a+a)}], v[{v.v_gld_a(s)}], {shift}, v[{v.v_a(slot_off+i_rm*num_v_a+a)}]")
                return outer._get_deferred()
        return functor_t()

    def shared_load_b_functor(self):
        '''
        Transposed read for input: LDS holds it as natural [K rows][N cols] (row_pitch =
        gemm_n_per_block*databyte bytes). Identical to bwd's shared_load_b_functor (same
        tensor shape/role: [K][C_in], contiguous over C_in) -- see that file for the
        detailed explanation of the strided-read-and-pack technique, including the
        `num_v_b`-derived loop bound/index and the ds_read_b32 branch fp32 needs.
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
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                s = outer.sgpr
                with outer._deferred_context():
                    outer._emit(f"v_add_co_u32 v[{v.v_addr_a()}], vcc_lo, s[{s.s_a_k_stride()}], v[{v.v_addr_a()}]")
                    outer._emit(f"v_add_co_ci_u32 v[{v.v_addr_a(1)}], vcc_lo, 0, v[{v.v_addr_a(1)}], vcc_lo")
                return outer._get_deferred()
        return functor_t()

    def move_slice_window_b_functor(self):
        '''
        Unlike fwd/bwd (and unlike A's own move_slice_window, and this kernel's own
        degenerate-case predecessor), B's address can no longer be advanced by a constant/
        runtime-but-fixed stride each iteration: the (n,ho,wo)->(hi,wi) gather must be
        recomputed from scratch for the new K-block's absolute k-index -- see class docstring
        and _emit_b_gather. k_block_off = s_knum - s_kitr, evaluated here (i.e. AFTER
        wmma_main_loop.py's per-iteration s_kitr -= unroll_k), correctly gives the K already
        consumed = the offset of the block about to be loaded.
        '''
        outer = self
        class functor_t:
            def __call__(self):
                s = outer.sgpr
                with outer._deferred_context():
                    outer._emit(f"s_sub_i32 s[{s.s_tmp()}], s[{s.s_knum()}], s[{s.s_kitr()}]   ; k_block_off")
                    outer._emit_b_gather(s.s_tmp())
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
        ctrl.local_prefetch_num = self.tunable.local_prefetch_num
        ctrl.wmma_setprio = self.tunable.wmma_setprio
        # Phase 1 (k-sub-loop): both A (grad_output) and B (input) are TRANSPOSED here
        # ([K rows][M or N cols] in LDS), so advancing inst_wmma.k K-elements means
        # advancing inst_wmma.k whole K-rows, i.e. inst_wmma.k * row_pitch, matching each
        # shared_load_a/b_functor's own row_pitch computation.
        inst_wmma_k = self.wmma_mapping.ctrl.inst_wmma.k
        ctrl.k_substep_stride_bytes_a    = inst_wmma_k * (self.tunable.gemm_m_per_block * self.data_byte)
        ctrl.k_substep_stride_bytes_b    = inst_wmma_k * (self.tunable.gemm_n_per_block * self.data_byte)
        ctrl.global_load_a_functor       = self.global_load_a_functor()
        ctrl.global_load_b_functor       = self.global_load_b_functor()
        ctrl.shared_store_a_functor      = self.shared_store_a_functor()
        ctrl.shared_store_b_functor      = self.shared_store_b_functor()
        ctrl.shared_load_a_functor       = self.shared_load_a_functor()
        ctrl.shared_load_b_functor       = self.shared_load_b_functor()
        ctrl.move_slice_window_a_functor = self.move_slice_window_a_functor()
        ctrl.move_slice_window_b_functor = self.move_slice_window_b_functor()
        ctrl.v_a       = sym_t(self.vgpr.v_a.label)
        ctrl.v_b       = sym_t(self.vgpr.v_b.label)
        ctrl.v_c       = sym_t(self.vgpr.v_c.label)
        ctrl.v_sst_a_os = sym_t(self.vgpr.v_sst_os.label)
        ctrl.v_sld_a_os = sym_t(self.vgpr.v_sld_a_os.label)
        ctrl.v_sst_b_os = sym_t(self.vgpr.v_sst_os.label)
        ctrl.v_sld_b_os = sym_t(self.vgpr.v_sld_b_os.label)
        ctrl.s_kitr    = sym_t(self.sgpr.s_kitr.label)
        ctrl.s_knum    = sym_t(self.sgpr.s_knum.label)

        # first global load for this tap already issued by emit_kernel_tap_loop(), which is
        # also the sole caller of this method and also emits the per-tap epilogue store
        # (see class docstring, Phase 5f)
        wmma_main_loop_t(self.mc, ctrl).emit()

    def emit_kernel_body(self):
        self.emit_kernel_prologue()
        self.emit_kernel_tap_loop()
