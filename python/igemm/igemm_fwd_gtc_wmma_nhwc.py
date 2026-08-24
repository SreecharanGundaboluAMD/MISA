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

class igemm_fwd_gtc_wmma_nhwc_t(mc_base_t):
    '''
    gfx1250 WMMA fwd kernel -- still NOT a fully general NHWC convolution kernel like
    igemm_fwd_gtc_nhwc_t (no multi-tap filter, no dilation, no groups), but Phase 5a
    extends it beyond the original degenerate 1x1/stride1/no-pad GEMM to support
    arbitrary stride and padding (dilation=1, y=x=1, group=1 still required):

        output[n,ho,wo, k] = sum_c input[n, ho*stride_h-pad_h, wo*stride_w-pad_w, c] * weight[k, c]
                              (0 if the input pixel above is out of [0,hi)x[0,wi) bounds)

    m enumerates (n,ho,wo) output pixels (GEMM_M = n*ho*wo), k enumerates input
    channels (GEMM_K = c), n enumerates output channels (GEMM_N = k_out). Weight
    ([K_out, C_in], the standard conv weight layout with y=x=1) is still exactly
    row-major [GEMM_N][GEMM_K] with no transpose needed and NO stride/pad dependence
    at all (weight has no spatial extent for a 1x1 filter) -- its addressing is
    completely unchanged from the degenerate-case kernel.

    Input, however, is no longer simply row-major-by-M: each output pixel's GEMM_M
    index must be decomposed back into (n, ho, wo) (via runtime integer division --
    see get_kernel_macros()/macro_int_div_rem_vs_gfx1250_t in operations/utility.py,
    a wave32-safe sibling of the codebase's existing "plain division" fallback
    macro_int_div_rem_vs_t -- the original uses bare `vcc` and 64-bit SGPR-pair
    compare destinations that fail to assemble on gfx1250's wave32, see that class's
    docstring. Deliberately NOT magic division, since ho/wo are only known at
    kernel-launch time and this is still correctness-first, not perf-tuned),
    then mapped through the stride/pad formula to an (hi, wi) input pixel coordinate,
    which may be out of bounds (padding). Out-of-bounds reads are masked to zero via
    the same unsigned-wraparound trick igemm_fwd_gtc_nhwc.py's macro_set_flag_nhw
    uses (a negative signed index reinterpreted as u32 is huge, so a single unsigned
    `>` comparison rejects both "negative" and "too large" in one instruction) plus
    EXEC masking (`v_cmpx_le_u32`/restore) around the global load itself, rather than
    a post-load select -- since this thread's padding-vs-real-pixel status is a
    per-thread CONSTANT for the whole kernel (1x1 filter means it never depends on
    which K-chunk is being loaded), the flag is computed once in the prologue and
    reused for every main-loop iteration's reload, and the masked-off VGPRs simply
    keep holding whatever they were initialized to (zero, from never being written)
    for the entire kernel -- no repeated re-masking needed.

    This is still a new, purpose-built prologue rather than an adaptation of
    igemm_fwd_gtc_nhwc_t's general-conv machinery: that machinery's k_pack/
    thread-cluster addressing is tightly coupled to XDLOPS/DLOPS register-packing
    assumptions that don't match WMMA's operand layout, and reconciling the two
    was judged riskier than writing new, small, auditable addressing logic here.

    Supports exactly one macro-tile shape: gemm_m_per_block == gemm_n_per_block == 128,
    block_size == 128 (4 waves), wmma_tile_m == wmma_tile_n == 16, wmma_repeat_m ==
    wmma_repeat_n == 4. gemm_k_per_block must equal the wired-up instruction's K (32 for
    fp16/bf16, 64 for int8 -- see ctrl_wmma_mapping_table in wmma_mapping.py); gemm_k
    (total) must be a multiple of that; gemm_m/gemm_n must be multiples of 128.
    precision: 'fp16', 'bf16' (both 2 bytes/element, K=32), or 'int8' (1 byte/element, K=64,
    int32 accumulate) -- see docs/gfx1250_wmma_layout.md for the empirically-verified
    per-lane layout of all three. Despite int8's different K/element-width, almost none of
    this kernel's byte-level address math needed a precision-specific branch: gemm_k_per_block
    is always chosen to equal inst_wmma.k, and inst_wmma.num_v_a/num_v_b/num_v_c are 8/8/8 for
    every wired-up instruction, so quantities like "bytes per tile row" (=64, always) and
    "bytes per wave_repeat step" turn out precision-invariant. Only the A/B global-address
    stride (gemm_k * data_byte) genuinely varies and is computed from self.data_byte.
    '''
    def __init__(self, mc, tunable):
        mc_base_t.__init__(self, mc)
        assert tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA
        assert tunable.precision in ('fp16', 'bf16', 'int8'), f'unsupported precision:{tunable.precision}'
        assert tunable.tensor_layout == 'nhwc'
        assert tunable.gemm_m_per_block == 128 and tunable.gemm_n_per_block == 128
        assert tunable.wmma_tile_m == 16 and tunable.wmma_tile_n == 16
        assert tunable.wmma_repeat_m == 4 and tunable.wmma_repeat_n == 4
        assert tunable.block_size == 128
        self.tunable = tunable
        self.data_byte = amdgpu_precision_data_byte(tunable.precision)

        ctrl_wmma_mapping = get_ctrl_wmma_mapping_from_wave_tile(tunable.gemm_m_per_block, tunable.gemm_n_per_block,
                tunable.wmma_tile_m, tunable.wmma_tile_n, tunable.wmma_repeat_m, tunable.wmma_repeat_n,
                tunable.block_size // tunable.wave_size, tunable.precision)
        # gemm_k_per_block must equal the wired-up instruction's K (32 for fp16/bf16, 64 for
        # int8) -- wmma_main_loop.py requires unroll_k == inst_wmma.k exactly (no k-sub-loop).
        assert tunable.gemm_k_per_block == ctrl_wmma_mapping.inst_wmma.k, \
            f"gemm_k_per_block({tunable.gemm_k_per_block}) must equal inst_wmma.k({ctrl_wmma_mapping.inst_wmma.k}) for precision {tunable.precision}"
        self.wmma_mapping = igemm_wmma_mapping_t(self.mc, ctrl_wmma_mapping)

        ctrl_coalescing_store_wmma = ctrl_coalescing_store_wmma_t()
        ctrl_coalescing_store_wmma.cxm = ctrl_wmma_mapping
        ctrl_coalescing_store_wmma.block_size = tunable.block_size
        ctrl_coalescing_store_wmma.precision = tunable.precision
        self.coalescing_store = igemm_coalescing_store_wmma_t(self.mc, ctrl_coalescing_store_wmma)

        # int8 added: the only two byte-width-dependent literals (the A/B global-address
        # stride multiplier in emit_kernel_prologue) were generalized to self.data_byte; every
        # other literal shift (tid*64 store offset, k_half*32, row*64 stride, wave_repeat*1024)
        # turned out precision-invariant per the class docstring, so no other changes were
        # needed for int8 support.
        self.lds_a_size = tunable.gemm_m_per_block * tunable.gemm_k_per_block * self.data_byte
        self.lds_b_size = tunable.gemm_n_per_block * tunable.gemm_k_per_block * self.data_byte

        self.sgpr = self.kernel_sgpr_t(mc, self)
        self.vgpr = self.kernel_vgpr_t(mc, self)

    def name(self):
        return igemm_gtc_encode_kernel_name(self.tunable, self.mc.arch_config.arch)

    def get_kernel_macros(self):
        # plain (non-magic) runtime u32 division, used to decompose the merged GEMM_M lane
        # index back into (n, ho, wo) -- see class docstring. Both macros must be registered:
        # the "_rem_" one's body invokes the plain one as a nested .macro call.
        return [macro_int_div_vs_gfx1250_t(self.mc), macro_int_div_rem_vs_gfx1250_t(self.mc)]

    class kernel_sgpr_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            sseq = gpr_sequencer_t()
            self.s_ka          = sym_t('s_ka'          , sseq(2))
            self.s_bx          = sym_t('s_bx'          , sseq(1))    # workgroup_id_x -> gemm_m block index
            self.s_by          = sym_t('s_by'          , sseq(1))    # workgroup_id_y -> gemm_n block index
            self.s_p_in        = sym_t('s_p_in'        , sseq(2))
            self.s_p_wei       = sym_t('s_p_wei'       , sseq(2))
            self.s_p_out       = sym_t('s_p_out'       , sseq(2))
            self.s_gemm_m      = sym_t('s_gemm_m'      , sseq(1))
            self.s_gemm_n      = sym_t('s_gemm_n'      , sseq(1))
            self.s_gemm_k      = sym_t('s_gemm_k'      , sseq(1))
            # stride/pad kernarg fields (Phase 5a) -- declared contiguously in this exact
            # order to match get_kernel_args()'s layout, so two s_load_dwordx4 cover all 8.
            self.s_hi          = sym_t('s_hi'          , sseq(1))
            self.s_wi          = sym_t('s_wi'          , sseq(1))
            self.s_stride_h    = sym_t('s_stride_h'    , sseq(1))
            self.s_stride_w    = sym_t('s_stride_w'    , sseq(1))
            self.s_pad_h       = sym_t('s_pad_h'       , sseq(1))
            self.s_pad_w       = sym_t('s_pad_w'       , sseq(1))
            self.s_wo          = sym_t('s_wo'          , sseq(1))    # divisor for (ho,wo) decomposition
            self.s_ho_wo       = sym_t('s_ho_wo'       , sseq(1))    # divisor for (n,ho*wo) decomposition
            self.s_hi_wi       = sym_t('s_hi_wi'       , sseq(1))    # = s_hi*s_wi, computed on-device once
            self.s_block_m_off = sym_t('s_block_m_off' , sseq(1))
            self.s_block_n_off = sym_t('s_block_n_off' , sseq(1))
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
            self.v_gld_a       = sym_t('v_gld_a'       , vseq(16))   # gemm_k_per_block*data_byte = 64B row (any precision), staged before LDS store
            self.v_gld_b       = sym_t('v_gld_b'       , vseq(16))
            self.v_tid         = sym_t('v_tid'         , vseq(1))
            # 64-bit VADDR pairs must be even-aligned on gfx1250 (verified with llvm-mc)
            self.v_addr_a      = sym_t('v_addr_a'      , vseq(2, 2))    # persistent global A address (64-bit)
            self.v_addr_b      = sym_t('v_addr_b'      , vseq(2, 2))
            self.v_addr_out    = sym_t('v_addr_out'    , vseq(2, 2))    # scratch used by coalescing_store_wmma
            self.v_sst_os      = sym_t('v_sst_os'      , vseq(1))    # shared store offset (same for A/B region)
            self.v_sld_a_os    = sym_t('v_sld_a_os'    , vseq(1))
            self.v_sld_b_os    = sym_t('v_sld_b_os'    , vseq(1))
            self.v_gemm_im     = sym_t('v_gemm_im'     , vseq(1))
            self.v_gemm_in     = sym_t('v_gemm_in'     , vseq(1))
            self.v_tmp         = sym_t('v_tmp'         , vseq(4))
            # Phase 5a (stride/pad): v_flag is a per-thread CONSTANT (1x1 filter -> padding
            # status never depends on which K-chunk is loaded), computed once in the prologue
            # and reused by every main-loop reload of the A operand. v_gtc_tmp is scratch used
            # only during that one-time (n,ho,wo)->(hi,wi) decomposition -- see emit_kernel_prologue.
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
        # Phase 5a (stride/pad): declaration order here must exactly match kernel_sgpr_t's
        # s_hi..s_ho_wo declaration order, so two s_load_dwordx4 cover all 8 fields.
        kas.append(amdgpu_kernel_arg_t('hi'        , 4, 36, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('wi'        , 4, 40, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('stride_h'  , 4, 44, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('stride_w'  , 4, 48, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('pad_h'     , 4, 52, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('pad_w'     , 4, 56, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('wo'        , 4, 60, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('ho_wo'     , 4, 64, 'by_value', 'i32'))
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
        self._emit(f"s_load_dword s[{s.s_hi()}], s[{s.s_ka()}:{s.s_ka(1)}], 36")
        self._emit(f"s_load_dword s[{s.s_wi()}], s[{s.s_ka()}:{s.s_ka(1)}], 40")
        self._emit(f"s_load_dword s[{s.s_stride_h()}], s[{s.s_ka()}:{s.s_ka(1)}], 44")
        self._emit(f"s_load_dword s[{s.s_stride_w()}], s[{s.s_ka()}:{s.s_ka(1)}], 48")
        self._emit(f"s_load_dword s[{s.s_pad_h()}], s[{s.s_ka()}:{s.s_ka(1)}], 52")
        self._emit(f"s_load_dword s[{s.s_pad_w()}], s[{s.s_ka()}:{s.s_ka(1)}], 56")
        self._emit(f"s_load_dword s[{s.s_wo()}], s[{s.s_ka()}:{s.s_ka(1)}], 60")
        self._emit(f"s_load_dword s[{s.s_ho_wo()}], s[{s.s_ka()}:{s.s_ka(1)}], 64")
        self._emit(f"v_mov_b32 v[{v.v_tid()}], v0")
        # gfx1250 delivers workgroup id via ttmp9/ttmp7 (verified empirically against a
        # disassembled HIP-compiled kernel using blockIdx.x/.y on this hardware/toolchain --
        # NOT via classical pre-loaded system SGPRs, regardless of the
        # .amdhsa_system_sgpr_workgroup_id_x/y kernel-descriptor flags). See
        # docs/gfx1250_wmma_layout.md.
        self._emit(f"s_mov_b32 s[{s.s_bx()}], ttmp9")
        self._emit(f"s_mov_b32 s[{s.s_by()}], ttmp7")
        self._emit(f"s_wait_kmcnt 0x0")
        self._emit_empty_line()

        # zero the accumulator -- v_wmma_* does D = A@B + C, and v_c is used as both C and D
        # (see wmma_main_loop.py), so it must start at zero rather than whatever garbage was
        # left in these VGPRs at kernel entry.
        self._emit(f"; clear accumulator")
        for i in range(self.tunable.num_vgpr_accumulate_c):
            self._emit(f"v_mov_b32 v[{v.v_c(i)}], 0")
        self._emit_empty_line()

        self._emit(f"s_lshl_b32 s[{s.s_block_m_off()}], s[{s.s_bx()}], 7   ; *128")
        self._emit(f"s_lshl_b32 s[{s.s_block_n_off()}], s[{s.s_by()}], 7   ; *128")
        self._emit(f"s_mov_b32 s[{s.s_knum()}], s[{s.s_gemm_k()}]")
        self._emit_empty_line()

        # ---- global address for this thread's row of the A tile (input, gathered through
        # stride/pad -- see class docstring) ----
        m_int_div_rem_vs = macro_int_div_rem_vs_gfx1250_t(self.mc)
        self._emit(f"s_mul_i32 s[{s.s_hi_wi()}], s[{s.s_hi()}], s[{s.s_wi()}]")
        self._emit(f"; decode this thread's absolute GEMM_M index into (n_idx, ho_idx, wo_idx)")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], s[{s.s_block_m_off()}], v[{v.v_tid()}]   ; m_idx")
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

        self._emit(f"; v_flag = 1 iff (hi_idx, wi_idx) in [0,hi)x[0,wi) -- unsigned compare rejects")
        self._emit(f"; negative indices too (they wrap to a huge u32), no separate >=0 check needed")
        self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_hi()}], v[{v.v_gtc_tmp(4)}]")
        self._emit(f"v_cndmask_b32 v[{v.v_flag()}], 0, 1, vcc_lo")
        self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_wi()}], v[{v.v_gtc_tmp(3)}]")
        self._emit(f"v_cndmask_b32 v[{v.v_flag()}], 0, v[{v.v_flag()}], vcc_lo")
        self._emit_empty_line()

        self._emit(f"; row_idx = n_idx*(hi*wi) + hi_idx*wi + wi_idx (meaningless but harmless if")
        self._emit(f"; v_flag==0 -- that lane's global_load_a is EXEC-masked off, see global_load_a_functor)")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(0)}], s[{s.s_hi_wi()}], v[{v.v_gtc_tmp(2)}]")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(2)}], s[{s.s_wi()}], v[{v.v_gtc_tmp(4)}]")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(2)}]")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(3)}]")
        self._emit_empty_line()

        self._emit(f"; v_addr_a = p_in + row_idx * gemm_k * {self.data_byte} bytes")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(0)}], s[{s.s_gemm_k()}], v[{v.v_gtc_tmp(0)}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_gtc_tmp(0)}], {utility_log2(self.data_byte)}, v[{v.v_gtc_tmp(0)}]")
        self._emit(f"v_mov_b32 v[{v.v_addr_a(1)}], s[{s.s_p_in(1)}]")
        self._emit(f"v_add_co_u32 v[{v.v_addr_a()}], vcc_lo, s[{s.s_p_in()}], v[{v.v_gtc_tmp(0)}]")
        self._emit(f"v_add_co_ci_u32 v[{v.v_addr_a(1)}], vcc_lo, 0, v[{v.v_addr_a(1)}], vcc_lo")
        self._emit_empty_line()

        # ---- global address for this thread's row of the B tile (weight, [GEMM_N][GEMM_K]) ----
        self._emit(f"; v_addr_b = p_wei + (block_n_off + tid) * gemm_k * {self.data_byte} bytes")
        self._emit(f"v_add_u32 v[{v.v_tmp(1)}], s[{s.s_block_n_off()}], v[{v.v_tid()}]")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_gemm_k()}], v[{v.v_tmp(1)}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp(1)}], {utility_log2(self.data_byte)}, v[{v.v_tmp(1)}]")
        self._emit(f"v_mov_b32 v[{v.v_addr_b(1)}], s[{s.s_p_wei(1)}]")
        self._emit(f"v_add_co_u32 v[{v.v_addr_b()}], vcc_lo, s[{s.s_p_wei()}], v[{v.v_tmp(1)}]")
        self._emit(f"v_add_co_ci_u32 v[{v.v_addr_b(1)}], vcc_lo, 0, v[{v.v_addr_b(1)}], vcc_lo")
        self._emit_empty_line()

        # ---- fixed per-thread shared-memory store offset: this thread owns row `tid` ----
        self._emit(f"v_lshlrev_b32 v[{v.v_sst_os()}], 6, v[{v.v_tid()}]   ; tid*64 bytes")
        self._emit_empty_line()

        # ---- shared-memory load offsets (WMMA operand layout, see docs/gfx1250_wmma_layout.md) ----
        # v_gemm_in/v_gemm_im outputs land directly in v_sld_b_os/v_sld_a_os (element-unit row/col,
        # contiguous wave-block base already folded in) -- kept distinct from the v_tmp scratch
        # triple the function itself uses internally, to avoid the two aliasing.
        self._emit(self.wmma_mapping.get_gemm_index_for_src_matrix(v.v_sld_b_os(), v.v_sld_a_os(), v.v_tid(), v.v_tmp()))
        self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}], 4, v[{v.v_tid()}]")
        self._emit(f"v_and_b32 v[{v.v_tmp()}], 1, v[{v.v_tmp()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], 5, v[{v.v_tmp()}]      ; k_half * 32 bytes")
        self._emit(f"v_lshlrev_b32 v[{v.v_sld_a_os()}], 6, v[{v.v_sld_a_os()}]  ; row * 64 byte row-stride")
        self._emit(f"v_add_u32 v[{v.v_sld_a_os()}], v[{v.v_tmp()}], v[{v.v_sld_a_os()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_sld_b_os()}], 6, v[{v.v_sld_b_os()}]")
        self._emit(f"v_add_u32 v[{v.v_sld_b_os()}], v[{v.v_tmp()}], v[{v.v_sld_b_os()}]")
        self._emit_empty_line()

        # ---- persistent (im, in) for the epilogue, converted to global output indices ----
        self._emit(self.wmma_mapping.get_gemm_index_for_dst_matrix(v.v_gemm_in(), v.v_gemm_im(), v.v_tid(), v.v_tmp()))
        self._emit(f"v_add_u32 v[{v.v_gemm_im()}], s[{s.s_block_m_off()}], v[{v.v_gemm_im()}]")
        self._emit(f"v_add_u32 v[{v.v_gemm_in()}], s[{s.s_block_n_off()}], v[{v.v_gemm_in()}]")
        self._emit_empty_line()

        # ---- pre-zero v_gld_a: a padding lane's global_load_a is EXEC-masked off on every
        # call (including this first one), so without this it would carry garbage forever ----
        for i in range(16):
            self._emit(f"v_mov_b32 v[{v.v_gld_a(i)}], 0")
        self._emit_empty_line()

        # ---- issue the first global loads (main loop expects this precondition) ----
        self._emit(self.global_load_a_functor()())
        self._emit(self.global_load_b_functor()())
        self._emit_empty_line()

    def global_load_a_functor(self):
        '''
        Padding lanes (v_flag==0) never execute the load (see class docstring): this thread's
        v_gld_a is pre-zeroed once (emit_kernel_prologue) and, for a padding lane, simply never
        written again for the life of the kernel -- correct, since a 1x1-filter padding pixel
        contributes zero regardless of which K-chunk is being loaded.
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
        outer = self
        class functor_t:
            def __call__(self, v_dst, v_os, extra_off):
                v = outer.vgpr
                with outer._deferred_context():
                    for i_rn in range(4):
                        base = outer.lds_a_size + i_rn * 1024 + extra_off  # B region starts after A's region
                        outer._emit(f"ds_read_b128 v[{v.v_b(i_rn*8)}:{v.v_b(i_rn*8+3)}], v[{v.v_sld_b_os()}] offset:{base}")
                        outer._emit(f"ds_read_b128 v[{v.v_b(i_rn*8+4)}:{v.v_b(i_rn*8+7)}], v[{v.v_sld_b_os()}] offset:{base+16}")
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
                with outer._deferred_context():
                    outer._emit(f"v_add_co_u32 v[{v.v_addr_b()}], vcc_lo, 64, v[{v.v_addr_b()}]")
                    outer._emit(f"v_add_co_ci_u32 v[{v.v_addr_b(1)}], vcc_lo, 0, v[{v.v_addr_b(1)}], vcc_lo")
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
        self._emit(self.coalescing_store(v.v_c.label, v.v_gemm_im(), v.v_gemm_in(), s.s_p_out.label, s.s_gemm_n.label, v.v_addr_out.label))
        self._emit(f"s_wait_storecnt 0x0")

    def emit_kernel_body(self):
        self.emit_kernel_prologue()
        self.emit_kernel_fma_main_loop()
        self.emit_kernel_epilogue()
