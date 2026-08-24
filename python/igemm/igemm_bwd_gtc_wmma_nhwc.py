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

class igemm_bwd_gtc_wmma_nhwc_t(mc_base_t):
    '''
    gfx1250 WMMA kernel for backward-data (grad-input) convolution. Phase 5b extends this
    beyond the original degenerate 1x1/stride1/no-pad GEMM to support arbitrary stride and
    padding (still 1x1 filter, dilation=1, group=1) -- see igemm_fwd_gtc_wmma_nhwc_t's Phase
    5a docstring for the sibling fwd extension this closely mirrors.

        grad_input[n,hi,wi, c] = sum_k grad_output[n, (hi+pad_h)/stride_h, (wi+pad_w)/stride_w, k] * weight[k, c]
                                  (0 if (hi+pad_h) isn't an exact multiple of stride_h, same
                                   for w, or if the resulting ho/wo index is out of bounds --
                                   a "stride gap": for stride>1, most input pixels have NO
                                   corresponding output pixel at all)

    where m enumerates (n_batch, hi, wi) INPUT pixels (GEMM_M = n*hi*wi -- this is
    grad_input's own pixel count, note the input, not output, spatial extent), k enumerates
    output channels (GEMM_K = k_out, the contraction dim -- swapped vs fwd, where GEMM_K was
    input channels), n enumerates input channels (GEMM_N = c_in).

    Weight (B) and the output (grad_input) write are completely unaffected by stride/padding
    (weight has no spatial extent for a 1x1 filter; every GEMM_M index is a real grad_input
    pixel regardless of stride/pad) and stay exactly as in the degenerate-case kernel.

    Tensor A (grad_output), however, needs the SAME kind of gather Phase 5a added to fwd's
    input operand, but with a genuinely different (harder) validity condition: unlike fwd's
    "in bounds or not" padding check, an input pixel here maps to a grad_output pixel only if
    `(spatial_idx + pad) / stride` divides EXACTLY -- a "stride gap" position (when stride>1)
    contributes zero regardless of bounds, since grad_output was never computed for it. GEMM_M
    is decomposed into (n_idx, hi_idx, wi_idx) via two chained runtime divisions (by hi*wi then
    wi -- see get_kernel_macros()/macro_int_div_rem_vs_gfx1250_t in operations/utility.py, the
    wave32-safe plain-division primitive Phase 5a introduced), then TWO MORE division calls
    compute `(hi_idx+pad_h)/stride_h` (quotient=ho_idx, remainder checked for exact
    divisibility) and the w equivalent. The resulting flag ANDs four conditions (h divisible,
    h in bounds, w divisible, w in bounds) and gates grad_output's global load via the same
    EXEC-masking (`v_cmpx_le_u32`/restore around a pre-zeroed v_gld_a) Phase 5a used --
    correct for the same reason: an input pixel's stride-gap/bounds status is a per-thread
    constant for the whole kernel (1x1 filter, so it never depends on which K-out chunk is
    loading).

    Tensor A (grad_output) is naturally row-major [GEMM_M'][GEMM_K] where GEMM_M' is
    grad_output's OWN pixel count (n*ho*wo) -- for the degenerate case GEMM_M'==GEMM_M
    (identity gather), which is why the original kernel could reuse fwd's A-operand path
    verbatim; for general stride/pad these differ in size and the gather above is required.

    Tensor B (weight) is physically stored [K_out][C_in] (same buffer/layout the
    forward kernel reads), which for THIS kernel's role assignment (GEMM_K=K_out,
    GEMM_N=C_in) means weight is naturally [GEMM_K][GEMM_N] -- K-major, N-contiguous,
    the TRANSPOSE of what a WMMA B-operand load wants ([GEMM_N][GEMM_K], contiguous
    over K). Global load/LDS store keep weight in its natural (untransposed)
    orientation; only the WMMA-consumption LDS *read* (shared_load_b_functor) does
    the transpose, via wmma_mapping.py's get_gemm_index_for_src_matrix_transposed --
    16 strided ds_read_u16 + 8 v_lshl_or_b32 packs per wave_repeat step instead of 2
    contiguous ds_read_b128, a deliberate correctness-over-speed tradeoff for this
    milestone (see that function's docstring).

    Output (grad_input) write reuses coalescing_store_wmma.py's epilogue unmodified,
    with s_gemm_m_stride bound to GEMM_N=C_in (NOT K_out -- easy to get backwards
    since the variable name is the same as fwd's, but bound to a different tensor).

    Same tile-shape constraints as the fwd kernel: 128x128x32, wmma_repeat 4x4,
    block_size 128, precision fp16/bf16. gemm_m/gemm_n multiples of 128, gemm_k
    (K_out) a multiple of 32.
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

        # A-region (grad_output): natural [GEMM_M][GEMM_K] tile, same shape as fwd's A/B.
        self.lds_a_size = tunable.gemm_m_per_block * tunable.gemm_k_per_block * self.data_byte
        # B-region (weight): natural [GEMM_K][GEMM_N] tile -- same total byte size (32*128*databyte
        # either way), just a transposed interpretation of the same linear LDS layout.
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
            self.s_block_m_off = sym_t('s_block_m_off' , sseq(1))
            self.s_block_n_off = sym_t('s_block_n_off' , sseq(1))
            self.s_wei_k_stride = sym_t('s_wei_k_stride', sseq(1))   # C_in*databyte*gemm_k_per_block: weight's per-K-block global stride
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
            self.v_gld_a       = sym_t('v_gld_a'       , vseq(16))   # 32 elements staged before LDS store
            self.v_gld_b       = sym_t('v_gld_b'       , vseq(16))   # also reused as scratch by the transposed shared_load_b
            self.v_tid         = sym_t('v_tid'         , vseq(1))
            # 64-bit VADDR pairs must be even-aligned on gfx1250 (verified with llvm-mc)
            self.v_addr_a      = sym_t('v_addr_a'      , vseq(2, 2))    # persistent global A address (64-bit)
            self.v_addr_b      = sym_t('v_addr_b'      , vseq(2, 2))
            self.v_addr_out    = sym_t('v_addr_out'    , vseq(2, 2))    # scratch used by coalescing_store_wmma
            self.v_sst_os      = sym_t('v_sst_os'      , vseq(1))    # shared store offset (same for A/B region)
            self.v_sld_a_os    = sym_t('v_sld_a_os'    , vseq(1))
            self.v_sld_b_os    = sym_t('v_sld_b_os'    , vseq(1))    # transposed byte offset (see get_gemm_index_for_src_matrix_transposed)
            self.v_gemm_im     = sym_t('v_gemm_im'     , vseq(1))
            self.v_gemm_in     = sym_t('v_gemm_in'     , vseq(1))
            self.v_tmp         = sym_t('v_tmp'         , vseq(4))
            # Phase 5b (stride/pad): v_flag is a per-thread CONSTANT (see class docstring),
            # computed once in the prologue and reused by every main-loop reload of A.
            # v_gtc_tmp is scratch for the one-time (n,hi,wi)->(ho,wo)-or-invalid decomposition
            # (needs more registers than fwd's Phase 5a: 4 chained divisions here, not 2).
            self.v_flag        = sym_t('v_flag'        , vseq(1))
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
        self._emit(f"s_load_dword s[{s.s_hi_wi()}], s[{s.s_ka()}:{s.s_ka(1)}], 36")
        self._emit(f"s_load_dword s[{s.s_wi()}], s[{s.s_ka()}:{s.s_ka(1)}], 40")
        self._emit(f"s_load_dword s[{s.s_stride_h()}], s[{s.s_ka()}:{s.s_ka(1)}], 44")
        self._emit(f"s_load_dword s[{s.s_stride_w()}], s[{s.s_ka()}:{s.s_ka(1)}], 48")
        self._emit(f"s_load_dword s[{s.s_pad_h()}], s[{s.s_ka()}:{s.s_ka(1)}], 52")
        self._emit(f"s_load_dword s[{s.s_pad_w()}], s[{s.s_ka()}:{s.s_ka(1)}], 56")
        self._emit(f"s_load_dword s[{s.s_ho()}], s[{s.s_ka()}:{s.s_ka(1)}], 60")
        self._emit(f"s_load_dword s[{s.s_wo()}], s[{s.s_ka()}:{s.s_ka(1)}], 64")
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
        # weight's per-K-block (32 rows) global stride = C_in * databyte * 32 -- runtime value,
        # since C_in (s_gemm_n) isn't known at compile time (unlike fwd's constant +64 bytes).
        self._emit(f"s_lshl_b32 s[{s.s_wei_k_stride()}], s[{s.s_gemm_n()}], {utility_log2(self.data_byte) + 5}   ; C_in * databyte * 32")
        self._emit_empty_line()

        # ---- global address for this thread's row of the A tile (grad_output, gathered
        # through stride/pad -- see class docstring) ----
        m_int_div_rem_vs = macro_int_div_rem_vs_gfx1250_t(self.mc)
        self._emit(f"s_mul_i32 s[{s.s_ho_wo()}], s[{s.s_ho()}], s[{s.s_wo()}]")
        self._emit(f"; decode this thread's absolute GEMM_M index into (n_idx, hi_idx, wi_idx)")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], s[{s.s_block_m_off()}], v[{v.v_tid()}]   ; m_idx")
        self._emit(m_int_div_rem_vs(v.v_gtc_tmp(1), v.v_gtc_tmp(2), v.v_gtc_tmp(0), s.s_hi_wi(), v.v_tmp(), s.s_tmp()))
        self._emit(f"; v_gtc_tmp(1)=hw_idx (rem), v_gtc_tmp(2)=n_idx (quo)")
        self._emit(m_int_div_rem_vs(v.v_gtc_tmp(3), v.v_gtc_tmp(4), v.v_gtc_tmp(1), s.s_wi(), v.v_tmp(), s.s_tmp()))
        self._emit(f"; v_gtc_tmp(3)=wi_idx (rem), v_gtc_tmp(4)=hi_idx (quo)")
        self._emit_empty_line()

        self._emit(f"; ho_idx = (hi_idx+pad_h)/stride_h (valid iff exact division & in bounds)")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(4)}], v[{v.v_gtc_tmp(4)}], s[{s.s_pad_h()}]   ; numerator_h")
        self._emit(m_int_div_rem_vs(v.v_gtc_tmp(5), v.v_gtc_tmp(6), v.v_gtc_tmp(4), s.s_stride_h(), v.v_tmp(), s.s_tmp()))
        self._emit(f"; v_gtc_tmp(5)=rem_h, v_gtc_tmp(6)=ho_idx")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(3)}], v[{v.v_gtc_tmp(3)}], s[{s.s_pad_w()}]   ; numerator_w")
        self._emit(m_int_div_rem_vs(v.v_gtc_tmp(7), v.v_gtc_tmp(8), v.v_gtc_tmp(3), s.s_stride_w(), v.v_tmp(), s.s_tmp()))
        self._emit(f"; v_gtc_tmp(7)=rem_w, v_gtc_tmp(8)=wo_idx")
        self._emit_empty_line()

        self._emit(f"; v_flag = 1 iff both divisions are exact AND (ho_idx,wo_idx) in bounds")
        self._emit(f"v_cmp_eq_u32 vcc_lo, 0, v[{v.v_gtc_tmp(5)}]")
        self._emit(f"v_cndmask_b32 v[{v.v_flag()}], 0, 1, vcc_lo")
        self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_ho()}], v[{v.v_gtc_tmp(6)}]")
        self._emit(f"v_cndmask_b32 v[{v.v_flag()}], 0, v[{v.v_flag()}], vcc_lo")
        self._emit(f"v_cmp_eq_u32 vcc_lo, 0, v[{v.v_gtc_tmp(7)}]")
        self._emit(f"v_cndmask_b32 v[{v.v_flag()}], 0, v[{v.v_flag()}], vcc_lo")
        self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_wo()}], v[{v.v_gtc_tmp(8)}]")
        self._emit(f"v_cndmask_b32 v[{v.v_flag()}], 0, v[{v.v_flag()}], vcc_lo")
        self._emit_empty_line()

        self._emit(f"; row_idx = n_idx*(ho*wo) + ho_idx*wo + wo_idx (meaningless but harmless if")
        self._emit(f"; v_flag==0 -- that lane's global_load_a is EXEC-masked off, see global_load_a_functor)")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(0)}], s[{s.s_ho_wo()}], v[{v.v_gtc_tmp(2)}]")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(2)}], s[{s.s_wo()}], v[{v.v_gtc_tmp(6)}]")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(2)}]")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(8)}]")
        self._emit_empty_line()

        self._emit(f"; v_addr_a = p_in + row_idx * gemm_k * databyte")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(0)}], s[{s.s_gemm_k()}], v[{v.v_gtc_tmp(0)}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_gtc_tmp(0)}], {utility_log2(self.data_byte)}, v[{v.v_gtc_tmp(0)}]")
        self._emit(f"v_mov_b32 v[{v.v_addr_a(1)}], s[{s.s_p_in(1)}]")
        self._emit(f"v_add_co_u32 v[{v.v_addr_a()}], vcc_lo, s[{s.s_p_in()}], v[{v.v_gtc_tmp(0)}]")
        self._emit(f"v_add_co_ci_u32 v[{v.v_addr_a(1)}], vcc_lo, 0, v[{v.v_addr_a(1)}], vcc_lo")
        self._emit_empty_line()

        # ---- global address for this thread's chunk of the B tile (weight, natural [GEMM_K][GEMM_N]) ----
        # thread tid owns row_local=tid>>2 (one of 32 K_out rows), col_group=tid&3 (one of 4
        # 32-wide chunks of the 128-wide C_in tile) -- see class docstring / wmma_mapping.py.
        self._emit(f"; v_addr_b = p_wei + (row_local*C_in + block_n_off + col_start) * databyte")
        self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}], 2, v[{v.v_tid()}]        ; row_local = tid>>2")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_gemm_n()}], v[{v.v_tmp()}]  ; row_local * C_in")
        self._emit(f"v_and_b32 v[{v.v_tmp(1)}], 3, v[{v.v_tid()}]           ; col_group = tid&3")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp(1)}], 5, v[{v.v_tmp(1)}]      ; col_start = col_group*32")
        self._emit(f"v_add_u32 v[{v.v_tmp()}], v[{v.v_tmp(1)}], v[{v.v_tmp()}]")
        self._emit(f"v_add_u32 v[{v.v_tmp()}], s[{s.s_block_n_off()}], v[{v.v_tmp()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {utility_log2(self.data_byte)}, v[{v.v_tmp()}]")
        self._emit(f"v_mov_b32 v[{v.v_addr_b(1)}], s[{s.s_p_wei(1)}]")
        self._emit(f"v_add_co_u32 v[{v.v_addr_b()}], vcc_lo, s[{s.s_p_wei()}], v[{v.v_tmp()}]")
        self._emit(f"v_add_co_ci_u32 v[{v.v_addr_b(1)}], vcc_lo, 0, v[{v.v_addr_b(1)}], vcc_lo")
        self._emit_empty_line()

        # ---- fixed per-thread shared-memory store offset: this thread owns byte-chunk tid*64 ----
        # (A: row tid's full 32-element row; B: (row_local,col_group)'s 32-element chunk -- the
        # linear tid-order enumeration happens to coincide exactly for both, see class docstring)
        self._emit(f"v_lshlrev_b32 v[{v.v_sst_os()}], 6, v[{v.v_tid()}]   ; tid*64 bytes")
        self._emit_empty_line()

        # ---- shared-memory load offset for A (untransposed, same as fwd) ----
        # this call also computes a "gemm_in" (column) side-output that bwd's A-only path
        # doesn't need; parked in v_gemm_im()/v_gemm_in() is unsafe (aliases the function's own
        # internal tmp2+2 scratch when waves_per_m!=1), so a genuinely dead destination is used --
        # it is fully overwritten below by get_gemm_index_for_dst_matrix before anything reads it.
        self._emit(self.wmma_mapping.get_gemm_index_for_src_matrix(v.v_tmp(3), v.v_sld_a_os(), v.v_tid(), v.v_tmp()))
        self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}], 4, v[{v.v_tid()}]")
        self._emit(f"v_and_b32 v[{v.v_tmp()}], 1, v[{v.v_tmp()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], 5, v[{v.v_tmp()}]      ; k_half * 32 bytes")
        self._emit(f"v_lshlrev_b32 v[{v.v_sld_a_os()}], 6, v[{v.v_sld_a_os()}]  ; row * 64 byte row-stride")
        self._emit(f"v_add_u32 v[{v.v_sld_a_os()}], v[{v.v_tmp()}], v[{v.v_sld_a_os()}]")
        self._emit_empty_line()

        # ---- shared-memory load offset for B (TRANSPOSED -- weight is [K][N] in LDS) ----
        # row_pitch_bytes = gemm_n_per_block * databyte (128 * databyte)
        self._emit(self.wmma_mapping.get_gemm_index_for_src_matrix_transposed(v.v_sld_b_os(), v.v_tid(), v.v_tmp(),
                self.tunable.gemm_n_per_block * self.data_byte, self.data_byte, 'n'))
        self._emit_empty_line()

        # ---- persistent (im, in) for the epilogue, converted to global output indices ----
        self._emit(self.wmma_mapping.get_gemm_index_for_dst_matrix(v.v_gemm_in(), v.v_gemm_im(), v.v_tid(), v.v_tmp()))
        self._emit(f"v_add_u32 v[{v.v_gemm_im()}], s[{s.s_block_m_off()}], v[{v.v_gemm_im()}]")
        self._emit(f"v_add_u32 v[{v.v_gemm_in()}], s[{s.s_block_n_off()}], v[{v.v_gemm_in()}]")
        self._emit_empty_line()

        # ---- pre-zero v_gld_a: a stride-gap/oob lane's global_load_a is EXEC-masked off on
        # every call (including this first one), so without this it would carry garbage forever ----
        for i in range(16):
            self._emit(f"v_mov_b32 v[{v.v_gld_a(i)}], 0")
        self._emit_empty_line()

        # ---- issue the first global loads (main loop expects this precondition) ----
        self._emit(self.global_load_a_functor()())
        self._emit(self.global_load_b_functor()())
        self._emit_empty_line()

    def global_load_a_functor(self):
        '''
        Padding/stride-gap lanes (v_flag==0) never execute the load (see class docstring):
        this thread's v_gld_a is pre-zeroed once (emit_kernel_prologue) and, for such a lane,
        simply never written again for the life of the kernel -- correct, since a 1x1-filter
        stride-gap/oob input pixel contributes zero regardless of which K-out chunk is loaded.
        '''
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                with outer._deferred_context():
                    outer._emit(f"v_cmpx_le_u32 1, v[{v.v_flag()}]")
                    for i in range(4):
                        outer._emit(f"global_load_dwordx4 v[{v.v_gld_a(i*4)}:{v.v_gld_a(i*4+3)}], v[{v.v_addr_a()}:{v.v_addr_a(1)}], off offset:{i*16}")
                    outer._emit(f"s_mov_b32 exec_lo, -1")
                return outer._get_deferred()
            def get_issues(self):
                return 4
        return functor_t()

    def global_load_b_functor(self):
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                with outer._deferred_context():
                    for i in range(4):
                        outer._emit(f"global_load_dwordx4 v[{v.v_gld_b(i*4)}:{v.v_gld_b(i*4+3)}], v[{v.v_addr_b()}:{v.v_addr_b(1)}], off offset:{i*16}")
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
        outer = self
        class functor_t:
            def __call__(self, v_dst, v_os, extra_off):
                v = outer.vgpr
                with outer._deferred_context():
                    for i_rm in range(4):
                        base = i_rm * 1024 + extra_off
                        outer._emit(f"ds_read_b128 v[{v.v_a(i_rm*8)}:{v.v_a(i_rm*8+3)}], v[{v.v_sld_a_os()}] offset:{base}")
                        outer._emit(f"ds_read_b128 v[{v.v_a(i_rm*8+4)}:{v.v_a(i_rm*8+7)}], v[{v.v_sld_a_os()}] offset:{base+16}")
                return outer._get_deferred()
        return functor_t()

    def shared_load_b_functor(self):
        '''
        Transposed read for the weight operand: LDS holds it as natural [K rows][N cols]
        (row_pitch = gemm_n_per_block*databyte bytes), but the WMMA B operand needs
        col-major/k-contiguous packed pairs. For each wave_repeat step i_rn and each of the 8
        vgpr indices a (k-half-relative), read the two k-sub-elements (s=0,1) as separate
        16-bit LDS loads (they are row_pitch_bytes apart, not adjacent) into v_gld_b (reused as
        scratch here -- safe, since this iteration's real global load into v_gld_b happens
        later in the main loop, after shared_load completes), then pack them into one dword
        with v_lshl_or_b32 -- deliberately correctness-over-speed, see class docstring.
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
                with outer._deferred_context():
                    outer._emit(f"v_add_co_u32 v[{v.v_addr_a()}], vcc_lo, 64, v[{v.v_addr_a()}]")
                    outer._emit(f"v_add_co_ci_u32 v[{v.v_addr_a(1)}], vcc_lo, 0, v[{v.v_addr_a(1)}], vcc_lo")
                return outer._get_deferred()
        return functor_t()

    def move_slice_window_b_functor(self):
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                s = outer.sgpr
                with outer._deferred_context():
                    outer._emit(f"v_add_co_u32 v[{v.v_addr_b()}], vcc_lo, s[{s.s_wei_k_stride()}], v[{v.v_addr_b()}]")
                    outer._emit(f"v_add_co_ci_u32 v[{v.v_addr_b(1)}], vcc_lo, 0, v[{v.v_addr_b(1)}], vcc_lo")
                return outer._get_deferred()
        return functor_t()

    def emit_kernel_fma_main_loop(self):
        ctrl = ctrl_wmma_main_loop_t()
        ctrl.wmma_m           = self.wmma_mapping.ctrl
        ctrl.unroll_k         = 32
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
        # s_gemm_m_stride bound to GEMM_N=C_in here (NOT K_out) -- grad_input's row stride.
        self._emit(self.coalescing_store(v.v_c.label, v.v_gemm_im(), v.v_gemm_in(), s.s_p_out.label, s.s_gemm_n.label, v.v_addr_out.label))
        self._emit(f"s_wait_storecnt 0x0")

    def emit_kernel_body(self):
        self.emit_kernel_prologue()
        self.emit_kernel_fma_main_loop()
        self.emit_kernel_epilogue()
