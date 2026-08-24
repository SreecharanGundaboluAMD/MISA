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
    gfx1250 WMMA kernel for backward-weight (grad-weight) convolution. Phase 5c extends this
    beyond the original degenerate 1x1/stride1/no-pad GEMM to support arbitrary stride and
    padding (still 1x1 filter, dilation=1, group=1) -- see igemm_fwd_gtc_wmma_nhwc_t's Phase
    5a and igemm_bwd_gtc_wmma_nhwc_t's Phase 5b docstrings for the sibling extensions.

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
        assert tunable.precision in ('fp16', 'bf16'), f'unsupported precision:{tunable.precision}'
        assert tunable.tensor_layout == 'nhwc'
        assert tunable.gemm_m_per_block == 128 and tunable.gemm_n_per_block == 128 and tunable.gemm_k_per_block == 32
        assert tunable.wmma_tile_m == 16 and tunable.wmma_tile_n == 16
        assert tunable.wmma_repeat_m == 4 and tunable.wmma_repeat_n == 4
        assert tunable.block_size == 128
        self.tunable = tunable
        self.data_byte = amdgpu_precision_data_byte(tunable.precision)

        ctrl_wmma_mapping = get_ctrl_wmma_mapping_from_wave_tile(tunable.gemm_m_per_block, tunable.gemm_n_per_block,
                tunable.wmma_tile_m, tunable.wmma_tile_n, tunable.wmma_repeat_m, tunable.wmma_repeat_n,
                tunable.block_size // tunable.wave_size, tunable.precision)
        self.wmma_mapping = igemm_wmma_mapping_t(self.mc, ctrl_wmma_mapping)

        ctrl_coalescing_store_wmma = ctrl_coalescing_store_wmma_t()
        ctrl_coalescing_store_wmma.cxm = ctrl_wmma_mapping
        ctrl_coalescing_store_wmma.block_size = tunable.block_size
        ctrl_coalescing_store_wmma.precision = tunable.precision
        self.coalescing_store = igemm_coalescing_store_wmma_t(self.mc, ctrl_coalescing_store_wmma)

        # A-region (grad_output) and B-region (input): both natural [GEMM_K rows][M or N
        # contiguous] tiles -- same total byte size either way (32*128*databyte).
        self.lds_a_size = tunable.gemm_k_per_block * tunable.gemm_m_per_block * self.data_byte
        self.lds_b_size = tunable.gemm_k_per_block * tunable.gemm_n_per_block * self.data_byte

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
            self.s_block_m_off = sym_t('s_block_m_off' , sseq(1))
            self.s_block_n_off = sym_t('s_block_n_off' , sseq(1))
            self.s_a_k_stride  = sym_t('s_a_k_stride'  , sseq(1))   # K_out*databyte*gemm_k_per_block: grad_output's per-K-block global stride
            self.s_kitr        = sym_t('s_kitr'        , sseq(1))
            self.s_knum        = sym_t('s_knum'        , sseq(1))
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
            self.v_gld_a       = sym_t('v_gld_a'       , vseq(16))   # 32 elements staged before LDS store; also
                                                                       # reused as scratch by the transposed shared_load_a
            self.v_gld_b       = sym_t('v_gld_b'       , vseq(16))   # ditto, reused by the transposed shared_load_b
            self.v_tid         = sym_t('v_tid'         , vseq(1))
            # 64-bit VADDR pairs must be even-aligned on gfx1250 (verified with llvm-mc)
            self.v_addr_a      = sym_t('v_addr_a'      , vseq(2, 2))    # persistent global A address (64-bit)
            self.v_addr_b      = sym_t('v_addr_b'      , vseq(2, 2))
            self.v_addr_out    = sym_t('v_addr_out'    , vseq(2, 2))    # scratch used by coalescing_store_wmma
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
        return kas

    def get_kernel_code(self):
        kernel_code_dict = {
            'enable_sgpr_kernarg_segment_ptr'  :   1,
            'enable_sgpr_workgroup_id_x'       :   1,
            'enable_sgpr_workgroup_id_y'       :   1,
            'enable_vgpr_workitem_id'          :   0,
            'workgroup_group_segment_byte_size':   self.lds_a_size + self.lds_b_size,
            'kernarg_segment_byte_size'         :   68,
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
        self._emit(f"v_mov_b32 v[{v.v_tid()}], v0")
        # gfx1250 delivers workgroup id via ttmp9/ttmp7 -- see docs/gfx1250_wmma_layout.md.
        self._emit(f"s_mov_b32 s[{s.s_bx()}], ttmp9")
        self._emit(f"s_mov_b32 s[{s.s_by()}], ttmp7")
        self._emit(f"s_wait_kmcnt 0x0")
        self._emit_empty_line()

        # zero the accumulator -- v_wmma_* does D = A@B + C, and v_c is used as both C and D.
        self._emit(f"; clear accumulator")
        for i in range(self.tunable.num_vgpr_accumulate_c):
            self._emit(f"v_mov_b32 v[{v.v_c(i)}], 0")
        self._emit_empty_line()

        self._emit(f"s_lshl_b32 s[{s.s_block_m_off()}], s[{s.s_bx()}], 7   ; *128")
        self._emit(f"s_lshl_b32 s[{s.s_block_n_off()}], s[{s.s_by()}], 7   ; *128")
        self._emit(f"s_mov_b32 s[{s.s_knum()}], s[{s.s_gemm_k()}]")
        # A's per-K-block (32 rows) global stride depends on a runtime tensor extent (K_out) --
        # unlike fwd's compile-time-constant +64 bytes. B has no equivalent constant stride
        # anymore (Phase 5c): its address is recomputed fresh every iteration (see
        # move_slice_window_b_functor), since the (n,ho,wo)->(hi,wi) gather changes every block.
        self._emit(f"s_lshl_b32 s[{s.s_a_k_stride()}], s[{s.s_gemm_m()}], {utility_log2(self.data_byte) + 5}   ; K_out * databyte * 32")
        self._emit(f"s_mul_i32 s[{s.s_hi_wi()}], s[{s.s_hi()}], s[{s.s_wi()}]")
        self._emit_empty_line()

        # ---- global address for this thread's chunk of the A tile (grad_output, natural [GEMM_K][GEMM_M]) ----
        # thread tid owns row_local=tid>>2 (one of 32 rows of this K-block), col_group=tid&3
        # (one of 4 32-wide chunks of the 128-wide K_out tile) -- same scheme bwd's B operand
        # used, now applied to A since A also needs the transposed treatment here.
        self._emit(f"; v_addr_a = p_in + (row_local*K_out + block_m_off + col_start) * databyte")
        self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}], 2, v[{v.v_tid()}]        ; row_local = tid>>2")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_gemm_m()}], v[{v.v_tmp()}]  ; row_local * K_out")
        self._emit(f"v_and_b32 v[{v.v_tmp(1)}], 3, v[{v.v_tid()}]           ; col_group = tid&3")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp(1)}], 5, v[{v.v_tmp(1)}]      ; col_start = col_group*32")
        self._emit(f"v_add_u32 v[{v.v_tmp()}], v[{v.v_tmp(1)}], v[{v.v_tmp()}]")
        self._emit(f"v_add_u32 v[{v.v_tmp()}], s[{s.s_block_m_off()}], v[{v.v_tmp()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {utility_log2(self.data_byte)}, v[{v.v_tmp()}]")
        self._emit(f"v_mov_b32 v[{v.v_addr_a(1)}], s[{s.s_p_in(1)}]")
        self._emit(f"v_add_co_u32 v[{v.v_addr_a()}], vcc_lo, s[{s.s_p_in()}], v[{v.v_tmp()}]")
        self._emit(f"v_add_co_ci_u32 v[{v.v_addr_a(1)}], vcc_lo, 0, v[{v.v_addr_a(1)}], vcc_lo")
        self._emit_empty_line()

        # ---- persistent setup for B's per-iteration gather (input, gathered through
        # stride/pad -- see class docstring) ----
        self._emit(f"v_lshrrev_b32 v[{v.v_row_local()}], 2, v[{v.v_tid()}]        ; row_local = tid>>2 (fixed for the whole kernel)")
        self._emit(f"; v_b_col_off = (block_n_off + col_start) * databyte (fixed for the whole kernel)")
        self._emit(f"v_and_b32 v[{v.v_tmp()}], 3, v[{v.v_tid()}]            ; col_group = tid&3")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], 5, v[{v.v_tmp()}]        ; col_start = col_group*32")
        self._emit(f"v_add_u32 v[{v.v_tmp()}], s[{s.s_block_n_off()}], v[{v.v_tmp()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_b_col_off()}], {utility_log2(self.data_byte)}, v[{v.v_tmp()}]")
        self._emit_empty_line()

        # ---- initial (k_block_off=0) gather + address + flag for B's first global load ----
        self._emit_b_gather(None)
        self._emit_empty_line()

        # ---- fixed per-thread shared-memory store offset: this thread owns byte-chunk tid*64 ----
        # (both A's and B's (row_local,col_group) 32-element chunk land at the same linear tid
        # position, see class docstring)
        self._emit(f"v_lshlrev_b32 v[{v.v_sst_os()}], 6, v[{v.v_tid()}]   ; tid*64 bytes")
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

        # ---- persistent (im, in) for the epilogue, converted to global output indices ----
        self._emit(self.wmma_mapping.get_gemm_index_for_dst_matrix(v.v_gemm_in(), v.v_gemm_im(), v.v_tid(), v.v_tmp()))
        self._emit(f"v_add_u32 v[{v.v_gemm_im()}], s[{s.s_block_m_off()}], v[{v.v_gemm_im()}]")
        self._emit(f"v_add_u32 v[{v.v_gemm_in()}], s[{s.s_block_n_off()}], v[{v.v_gemm_in()}]")
        self._emit_empty_line()

        # ---- issue the first global loads (main loop expects this precondition) ----
        self._emit(self.global_load_a_functor()())
        self._emit(self.global_load_b_functor()())
        self._emit_empty_line()

    def _emit_b_gather(self, s_k_block_off):
        '''
        Computes this thread's absolute K-index (k_block_off + row_local, where row_local is
        the persistent per-thread constant computed in the prologue), decomposes it into
        (n_idx, ho_idx, wo_idx) [output/grad_output space, matching GEMM_K's own definition],
        gathers the corresponding (possibly out-of-bounds) input pixel (hi_idx, wi_idx) via the
        same stride/pad formula fwd's Phase 5a uses, sets v_flag, and writes v_addr_b. Called
        once from the prologue (s_k_block_off=None, meaning k_block_off=0) and once per
        main-loop iteration from move_slice_window_b_functor (s_k_block_off = the SGPR holding
        the freshly-computed s_knum-s_kitr) -- see class docstring for why this must be
        recomputed every iteration, unlike fwd/bwd's one-time prologue computation.
        '''
        s = self.sgpr
        v = self.vgpr
        m_int_div_rem_vs = macro_int_div_rem_vs_gfx1250_t(self.mc)
        if s_k_block_off is not None:
            self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], s[{s_k_block_off}], v[{v.v_row_local()}]   ; k_abs")
        else:
            self._emit(f"v_mov_b32 v[{v.v_gtc_tmp(0)}], v[{v.v_row_local()}]   ; k_abs (k_block_off=0)")
        self._emit(m_int_div_rem_vs(v.v_gtc_tmp(1), v.v_gtc_tmp(2), v.v_gtc_tmp(0), s.s_ho_wo(), v.v_tmp(), s.s_tmp()))
        self._emit(f"; v_gtc_tmp(1)=hw_idx (rem), v_gtc_tmp(2)=n_idx (quo)")
        self._emit(m_int_div_rem_vs(v.v_gtc_tmp(3), v.v_gtc_tmp(4), v.v_gtc_tmp(1), s.s_wo(), v.v_tmp(), s.s_tmp()))
        self._emit(f"; v_gtc_tmp(3)=wo_idx (rem), v_gtc_tmp(4)=ho_idx (quo)")
        self._emit_empty_line()

        self._emit(f"; hi_idx = ho_idx*stride_h - pad_h ; wi_idx = wo_idx*stride_w - pad_w")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(4)}], s[{s.s_stride_h()}], v[{v.v_gtc_tmp(4)}]")
        self._emit(f"v_sub_i32 v[{v.v_gtc_tmp(4)}], v[{v.v_gtc_tmp(4)}], s[{s.s_pad_h()}]   ; hi_idx")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(3)}], s[{s.s_stride_w()}], v[{v.v_gtc_tmp(3)}]")
        self._emit(f"v_sub_i32 v[{v.v_gtc_tmp(3)}], v[{v.v_gtc_tmp(3)}], s[{s.s_pad_w()}]   ; wi_idx")
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

        self._emit(f"; v_addr_b = p_wei + row_idx * C_in * databyte + v_b_col_off")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(0)}], s[{s.s_gemm_n()}], v[{v.v_gtc_tmp(0)}]   ; row_idx * C_in")
        self._emit(f"v_lshlrev_b32 v[{v.v_gtc_tmp(0)}], {utility_log2(self.data_byte)}, v[{v.v_gtc_tmp(0)}]")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(0)}], v[{v.v_b_col_off()}]")
        self._emit(f"v_mov_b32 v[{v.v_addr_b(1)}], s[{s.s_p_wei(1)}]")
        self._emit(f"v_add_co_u32 v[{v.v_addr_b()}], vcc_lo, s[{s.s_p_wei()}], v[{v.v_gtc_tmp(0)}]")
        self._emit(f"v_add_co_ci_u32 v[{v.v_addr_b(1)}], vcc_lo, 0, v[{v.v_addr_b(1)}], vcc_lo")

    def global_load_a_functor(self):
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                with outer._deferred_context():
                    for i in range(4):
                        outer._emit(f"global_load_dwordx4 v[{v.v_gld_a(i*4)}:{v.v_gld_a(i*4+3)}], v[{v.v_addr_a()}:{v.v_addr_a(1)}], off offset:{i*16}")
                return outer._get_deferred()
            def get_issues(self):
                return 4
        return functor_t()

    def global_load_b_functor(self):
        '''
        Unlike fwd/bwd's Phase 5a/5b (where a padding lane's flag is constant for the whole
        kernel, so pre-zeroing v_gld_a ONCE in the prologue is correct), B's gather here is
        recomputed every iteration (see class docstring), so v_gld_b must be explicitly
        re-zeroed EVERY call before the masked load -- otherwise a lane that was valid on a
        previous iteration but is invalid (out of bounds) on this one would silently reuse its
        stale (non-zero) data instead of contributing zero.
        '''
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                with outer._deferred_context():
                    for i in range(16):
                        outer._emit(f"v_mov_b32 v[{v.v_gld_b(i)}], 0")
                    outer._emit(f"v_cmpx_le_u32 1, v[{v.v_flag()}]")
                    for i in range(4):
                        outer._emit(f"global_load_dwordx4 v[{v.v_gld_b(i*4)}:{v.v_gld_b(i*4+3)}], v[{v.v_addr_b()}:{v.v_addr_b(1)}], off offset:{i*16}")
                    outer._emit(f"s_mov_b32 exec_lo, -1")
                return outer._get_deferred()
            def get_issues(self):
                return 4
        return functor_t()

    def shared_store_a_functor(self):
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                with outer._deferred_context():
                    for i in range(4):
                        outer._emit(f"ds_write_b128 v[{v.v_sst_os()}], v[{v.v_gld_a(i*4)}:{v.v_gld_a(i*4+3)}] offset:{i*16}")
                return outer._get_deferred()
            def get_issues(self):
                return 4
        return functor_t()

    def shared_store_b_functor(self):
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                with outer._deferred_context():
                    for i in range(4):
                        outer._emit(f"ds_write_b128 v[{v.v_sst_os()}], v[{v.v_gld_b(i*4)}:{v.v_gld_b(i*4+3)}] offset:{outer.lds_a_size + i*16}")
                return outer._get_deferred()
            def get_issues(self):
                return 4
        return functor_t()

    def shared_load_a_functor(self):
        '''
        Transposed read for grad_output: LDS holds it as natural [K rows][M cols]
        (row_pitch = gemm_m_per_block*databyte bytes). Same technique as bwd's
        shared_load_b_functor (see that file for the detailed explanation), applied here to
        the A operand instead of B, and with A's region based at LDS byte 0 (not lds_a_size).
        '''
        outer = self
        class functor_t:
            def __call__(self, v_dst, v_os, extra_off):
                v = outer.vgpr
                row_pitch = outer.tunable.gemm_m_per_block * outer.data_byte
                with outer._deferred_context():
                    for i_rm in range(4):
                        col_off = i_rm * outer.tunable.wmma_tile_m * outer.data_byte
                        for a in range(8):
                            off_lo = col_off + (a * 2 + 0) * row_pitch
                            off_hi = col_off + (a * 2 + 1) * row_pitch
                            outer._emit(f"ds_read_u16 v[{v.v_gld_a(a*2)}], v[{v.v_sld_a_os()}] offset:{extra_off + off_lo}")
                            outer._emit(f"ds_read_u16 v[{v.v_gld_a(a*2+1)}], v[{v.v_sld_a_os()}] offset:{extra_off + off_hi}")
                        outer._emit(f"s_wait_dscnt 0x0")
                        for a in range(8):
                            outer._emit(f"v_lshl_or_b32 v[{v.v_a(i_rm*8+a)}], v[{v.v_gld_a(a*2+1)}], 16, v[{v.v_gld_a(a*2)}]")
                return outer._get_deferred()
        return functor_t()

    def shared_load_b_functor(self):
        '''
        Transposed read for input: LDS holds it as natural [K rows][N cols] (row_pitch =
        gemm_n_per_block*databyte bytes). Identical to bwd's shared_load_b_functor (same
        tensor shape/role: [K][C_in], contiguous over C_in) -- see that file for the
        detailed explanation of the strided-read-and-pack technique.
        '''
        outer = self
        class functor_t:
            def __call__(self, v_dst, v_os, extra_off):
                v = outer.vgpr
                row_pitch = outer.tunable.gemm_n_per_block * outer.data_byte
                with outer._deferred_context():
                    for i_rn in range(4):
                        col_off = i_rn * outer.tunable.wmma_tile_n * outer.data_byte
                        for a in range(8):
                            # B region starts at outer.lds_a_size within the shared LDS tile;
                            # v_sld_b_os only carries the offset local to B's own region.
                            off_lo = outer.lds_a_size + col_off + (a * 2 + 0) * row_pitch
                            off_hi = outer.lds_a_size + col_off + (a * 2 + 1) * row_pitch
                            outer._emit(f"ds_read_u16 v[{v.v_gld_b(a*2)}], v[{v.v_sld_b_os()}] offset:{extra_off + off_lo}")
                            outer._emit(f"ds_read_u16 v[{v.v_gld_b(a*2+1)}], v[{v.v_sld_b_os()}] offset:{extra_off + off_hi}")
                        outer._emit(f"s_wait_dscnt 0x0")
                        for a in range(8):
                            outer._emit(f"v_lshl_or_b32 v[{v.v_b(i_rn*8+a)}], v[{v.v_gld_b(a*2+1)}], 16, v[{v.v_gld_b(a*2)}]")
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
        ctrl.lds_single_size  = self.lds_a_size + self.lds_b_size
        ctrl.lds_buffer_num   = 1
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

        # first global load already issued by emit_kernel_prologue()
        wmma_main_loop_t(self.mc, ctrl).emit()

    def emit_kernel_epilogue(self):
        v = self.vgpr
        s = self.sgpr
        # s_gemm_m_stride bound to GEMM_N=C_in here -- grad_weight's row stride, same
        # [K_out][C_in] layout the fwd kernel reads as its weight input.
        self._emit(self.coalescing_store(v.v_c.label, v.v_gemm_im(), v.v_gemm_in(), s.s_p_out.label, s.s_gemm_n.label, v.v_addr_out.label))
        self._emit(f"s_wait_storecnt 0x0")

    def emit_kernel_body(self):
        self.emit_kernel_prologue()
        self.emit_kernel_fma_main_loop()
        self.emit_kernel_epilogue()
