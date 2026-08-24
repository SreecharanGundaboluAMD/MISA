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
    gfx1250 WMMA fwd kernel. Phase 5a added arbitrary stride/pad (1x1 filter only).
    Phase 5d added multi-tap filters (y,x >= 1) and dilation. Phase 7 adds group>1:

        output[n,ho,wo, g*Kpg+k] = sum_{iy,ix,c} input[n, ho*stride_h-pad_h+iy*dilation_h,
                                                    wo*stride_w-pad_w+ix*dilation_w, g*Cpg+c]
                                                * weight[g*Kpg+k, iy, ix, c]
                              (0 if the input pixel above is out of [0,hi)x[0,wi) bounds;
                               g = group index, Kpg/Cpg = K_out/C_in per group)

    group>1 (Phase 7): `gemm_m`/`gemm_n`/`gemm_k` are already per-group values (the driver
    divides `k`/`c` by `group` before writing the karg, unchanged from every earlier phase);
    `group` itself is folded into `grid_y` by the driver (`grid_y = group *
    ceil(gemm_n/gemm_n_per_block)`) rather than a genuine 3rd grid dimension (gfx1250 has no
    working `workgroup_id_z` delivery for these kernels). The kernel decodes `group_idx` back
    out of `s_by` on-device (`blocks_per_group_n = ceil(gemm_n/128)`, needs no division since
    128 is a compile-time power of 2; the `group_idx`/corrected-`s_by` split reuses the same
    wave32-safe division macro as everywhere else via a broadcast-to-vgpr +
    `v_readfirstlane_b32` round-trip, rather than a new scalar-scalar macro variant), then
    overwrites `s_by` in place so `s_block_n_off` (computed right after) is unchanged.

    Two DIFFERENT per-operand corrections are needed, because NHWC group-splitting works
    differently for each tensor:
    - **Input and output**: group is INTERLEAVED within each pixel's channel dimension (NHWC
      layout is `[N,H,W,G*C_per_group]`/`[N,Ho,Wo,G*K_per_group]`), so the per-pixel/row
      memory stride must be the tensor's TOTAL channel count (`gemm_k*group`/`gemm_n*group`,
      new `s_in_c_total`/`s_out_k_total`), NOT the per-group `gemm_k`/`gemm_n` used for the
      K-reduction size and the base-pointer offset (those stay unchanged) -- **this was a real
      bug caught by hardware validation**: an early draft reused `gemm_k`/`gemm_n` for the row
      stride too (correct only for `group=1`, where total==per-group), producing wrong,
      uncorrelated-looking output for every group except group 0 (group 0's offset is always
      zero, so the row-stride bug was invisible there) until traced down to exactly this.
    - **Weight**: needs NO equivalent correction. Its group split is the OUTERMOST dimension
      (`[G][K_per_group][Y][X][C_per_group]`, confirmed against `driver/naive_conv.h` and the
      general XDLOPS kernel's own grouped-conv addressing), so a group's entire weight
      sub-tensor is self-contained/block-contiguous -- the existing `wei_k_stride`-based
      addressing, with `group_idx * gemm_n * wei_k_stride` added to the base pointer once, is
      already correct unchanged.

    m enumerates (n,ho,wo) output pixels (GEMM_M = n*ho*wo), k enumerates input
    channels PER TAP (GEMM_K = c -- unchanged from Phase 5a; the y*x taps are a
    separate outer loop, not folded into GEMM_K), n enumerates output channels
    (GEMM_N = k_out). Weight is the standard conv layout [K_out, Y, X, C_in]
    (C_in innermost, confirmed against driver/naive_conv.h's
    `f_idx = k*fy*fx*c + iy*fx*c + ix*c + ic`), so weight's K_out-stride is
    `y*x*c` elements (not just `c` as in the 1x1 case), and a given tap's
    C_in-column-block starts `(iy*x+ix)*c` elements into that row.

    Rather than porting igemm_fwd_gtc_nhwc.py's merge_e machinery (a single
    merged GEMM_K = c*y*x with carry-propagating address deltas for when the
    c-dimension overflows into x, and x into y -- deliberately not reused here,
    see the Phase 2 design notes: that machinery is tightly coupled to XDLOPS/
    DLOPS thread-cluster assumptions that don't match WMMA's much simpler
    layout), this kernel keeps GEMM_K = c and wraps the *entire* WMMA K-main-loop
    (unchanged from Phase 5a) in a new, small, RUNTIME (not compile-time-unrolled)
    outer loop over the y*x taps:

        s_iy = 0
        L_tap_y: s_ix = 0
          L_tap_x: <recompute this tap's A operand address+flag, B operand address>
                   <issue first A/B loads>
                   <run the WMMA K-main-loop over C -- textually emitted EXACTLY
                    ONCE; the runtime branch below re-enters the same code>
                   s_ix++; branch to L_tap_x if s_ix < s_x
          s_iy++; branch to L_tap_y if s_iy < s_y

    v_c (the accumulator) is zeroed once at kernel entry and never reset between
    taps, so successive taps' `v_wmma_* D=A@B+C` calls naturally accumulate --
    the only thing that changes per tap is which A/B addresses feed the same
    K-loop body. All four waves in the block execute this loop in lockstep
    (s_iy/s_ix and the y/x bounds are wave-uniform SGPR values, not thread-
    varying), so the existing single-buffered-LDS barrier scheme inside the
    K-main-loop is unaffected.

    Per-tap gather (A operand): `hi_idx = ho_idx*stride_h - pad_h + iy*dilation_h`,
    `wi_idx` symmetric -- ho_idx/wo_idx/n_idx are decomposed from this thread's
    GEMM_M index exactly once (Phase 5a's division sequence, now kept in
    persistent VGPRs `v_n_idx`/`v_ho_idx`/`v_wo_idx` instead of being consumed
    in-place, since every tap needs to recompute hi_idx/wi_idx fresh from the
    same ho_idx/wo_idx). Padding is masked the same way Phase 5a did (unsigned-
    wraparound bounds check + EXEC-masked global load) but the flag is no longer
    a per-thread CONSTANT for the whole kernel (it can flip between taps, since
    different taps look at different input pixels) -- `global_load_a_functor`
    therefore re-zeros all 16 `v_gld_a` registers on EVERY call (same discipline
    Phase 5c/wrw already established for its own per-iteration-varying gather),
    not just once, so a lane valid on a previous tap can never leak stale data
    into a tap where it's invalid.

    Per-tap gather (B operand): no padding concept (a tap index is always a
    real weight row as long as `0<=iy<y, 0<=ix<x`, which the loop bounds
    guarantee) -- just a different fixed byte offset added to a per-thread base
    address (`v_addr_b_base`, computed once from the *now-correct* `y*x*c`
    row stride) for each tap. `move_slice_window_b_functor` (the plain +64-byte
    bump across the K-sub-loop within one tap) is unchanged from Phase 5a.

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
        assert tunable.precision in ('fp16', 'bf16', 'int8', 'fp32'), f'unsupported precision:{tunable.precision}'
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
        self.coalescing_store = igemm_coalescing_store_wmma_t(self.mc, ctrl_coalescing_store_wmma)

        # int8 added: the only two byte-width-dependent literals (the A/B global-address
        # stride multiplier in emit_kernel_prologue) were generalized to self.data_byte; every
        # other literal shift (tid*64 store offset, k_half*32, row*64 stride, wave_repeat*1024)
        # turned out precision-invariant for fp16/bf16/int8, since gemm_k_per_block*data_byte
        # happens to equal 64 bytes for all three (32*2, 32*2, 64*1). fp32 breaks this
        # coincidence (gemm_k_per_block is forced to 4, matching inst_wmma.k, so
        # 4*4=16 bytes, not 64) -- every one of those literals is now derived from
        # `self.bytes_per_row` instead, see their call sites below.
        self.bytes_per_row = tunable.gemm_k_per_block * self.data_byte   # per-thread A/B row width
        self.num_dwordx4   = self.bytes_per_row // 16   # global_load/shared_store loop bound (was hardcoded 4)
        self.num_dwords    = self.bytes_per_row // 4    # v_gld_a/b zero-init loop bound (was hardcoded 16)
        # Phase 1 (k-sub-loop): the WMMA "k_half" wave-split (lane>>4) only ever applies
        # WITHIN one inst_wmma.k-wide instruction, never across the whole (possibly
        # multi-substep) gemm_k_per_block row -- must be derived from inst_wmma.k, NOT
        # bytes_per_row (which now can be a multiple of inst_wmma.k*data_byte).
        self.inst_wmma_k_bytes = ctrl_wmma_mapping.inst_wmma.k * self.data_byte
        # Phase 1 (k-sub-loop): global_load/shared_store are chunked into num_k_chunks
        # rounds of one inst_wmma.k-worth each, REUSING the same small v_gld_a/b buffer
        # (sized to chunk_num_dword{,x4}, not the whole row) across chunks -- growing
        # v_gld_a/b to hold the whole (now possibly multi-substep) row would exceed the
        # 256-VGPR/wave hardware limit for fp16/bf16/int8, which already sit at 252/256
        # with the single-substep tile. See docs/gfx1250_wmma_layout.md.
        self.chunk_num_dwordx4 = self.inst_wmma_k_bytes // 16
        self.chunk_num_dwords  = self.inst_wmma_k_bytes // 4
        self.num_k_chunks      = self.num_dwordx4 // self.chunk_num_dwordx4
        self.lds_a_size = tunable.gemm_m_per_block * tunable.gemm_k_per_block * self.data_byte
        self.lds_b_size = tunable.gemm_n_per_block * tunable.gemm_k_per_block * self.data_byte
        # Phase 2 (double-buffering): lds_single_size must be a power of 2 for the
        # v_xor_b32 ping-pong switch (wmma_main_loop.py) to correctly alternate between
        # exactly two adjacent buffers -- same rounding igemm_base.py's XDLOPS/DLOPS
        # tracks already use for their own (long-since-validated) double buffering.
        self.lds_single_size = igemm_next_pow2(self.lds_a_size + self.lds_b_size)
        self.lds_buffer_num  = 2 if tunable.lds_double_buffer else 1

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
            # Phase 5d (multi-tap + dilation) kernarg fields, contiguous, matching
            # get_kernel_args()'s trailing layout so two more s_load_dword pairs cover them.
            self.s_y           = sym_t('s_y'           , sseq(1))
            self.s_x           = sym_t('s_x'           , sseq(1))
            self.s_dilation_h  = sym_t('s_dilation_h'  , sseq(1))
            self.s_dilation_w  = sym_t('s_dilation_w'  , sseq(1))
            self.s_wei_k_stride = sym_t('s_wei_k_stride', sseq(1))   # = y*x*gemm_k, computed on-device once
            self.s_iy          = sym_t('s_iy'          , sseq(1))   # runtime tap-loop counters
            self.s_ix          = sym_t('s_ix'          , sseq(1))
            self.s_group_idx   = sym_t('s_group_idx'   , sseq(1))   # decoded from s_by (group folded into grid_y)
            self.s_group       = sym_t('s_group'       , sseq(1))   # kernarg: total group count
            self.s_in_c_total  = sym_t('s_in_c_total'  , sseq(1))   # = gemm_k*group, A's per-pixel row stride
            self.s_out_k_total = sym_t('s_out_k_total' , sseq(1))   # = gemm_n*group, output's per-pixel row stride
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
            # Phase 1 (k-sub-loop): sized to outer.chunk_num_dwords (one inst_wmma.k-worth),
            # NOT outer.num_dwords (the whole, possibly multi-substep, row) -- global_load/
            # shared_store are now chunked and reuse this same small buffer across chunks,
            # since growing it to hold the whole row would exceed the 256-VGPR/wave limit
            # for fp16/bf16/int8. See self.chunk_num_dwords in __init__.
            self.v_gld_a       = sym_t('v_gld_a'       , vseq(outer.chunk_num_dwords))
            self.v_gld_b       = sym_t('v_gld_b'       , vseq(outer.chunk_num_dwords))
            self.v_tid         = sym_t('v_tid'         , vseq(1))
            # 64-bit VADDR pairs must be even-aligned on gfx1250 (verified with llvm-mc)
            self.v_addr_a      = sym_t('v_addr_a'      , vseq(2, 2))    # persistent global A address (64-bit)
            self.v_addr_b      = sym_t('v_addr_b'      , vseq(2, 2))
            # Phase 5d: B's fixed per-thread row base (before this tap's column offset is
            # added) -- computed once from the *y*x*c* row stride, reused every tap.
            self.v_addr_b_base = sym_t('v_addr_b_base' , vseq(2, 2))
            self.v_addr_out    = sym_t('v_addr_out'    , vseq(2, 2))    # scratch used by coalescing_store_wmma
            self.v_sst_os      = sym_t('v_sst_os'      , vseq(1))    # shared store offset (same for A/B region)
            self.v_sld_a_os    = sym_t('v_sld_a_os'    , vseq(1))
            self.v_sld_b_os    = sym_t('v_sld_b_os'    , vseq(1))
            self.v_gemm_im     = sym_t('v_gemm_im'     , vseq(1))
            self.v_gemm_in     = sym_t('v_gemm_in'     , vseq(1))
            self.v_tmp         = sym_t('v_tmp'         , vseq(4))
            # Phase 5d: v_flag is now recomputed every TAP (not a whole-kernel constant like
            # Phase 5a, since different taps look at different input pixels) -- see
            # global_load_a_functor's re-zero-every-call discipline. v_n_idx/v_ho_idx/v_wo_idx
            # are this thread's GEMM_M decomposition, computed once and kept persistent (every
            # tap re-derives hi_idx/wi_idx from the SAME ho_idx/wo_idx). v_gtc_tmp is scratch
            # reused fresh every tap for the hi_idx/wi_idx/flag/row_idx computation.
            self.v_flag        = sym_t('v_flag'        , vseq(1))
            self.v_n_idx       = sym_t('v_n_idx'       , vseq(1))
            self.v_ho_idx      = sym_t('v_ho_idx'      , vseq(1))
            self.v_wo_idx      = sym_t('v_wo_idx'      , vseq(1))
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
        # Phase 5d (multi-tap + dilation)
        kas.append(amdgpu_kernel_arg_t('y'         , 4, 68, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('x'         , 4, 72, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dilation_h', 4, 76, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dilation_w', 4, 80, 'by_value', 'i32'))
        # Phase 7 (group>1): the only new kernarg needed -- gemm_m/gemm_n/gemm_k are already
        # per-group values (host divides by group before writing them), but A (input) and the
        # output (D) need the tensor's TOTAL channel count for their per-pixel row stride,
        # which requires knowing group itself (see class docstring).
        kas.append(amdgpu_kernel_arg_t('group'     , 4, 84, 'by_value', 'i32'))
        return kas

    def get_kernel_code(self):
        kernel_code_dict = {
            'enable_sgpr_kernarg_segment_ptr'  :   1,
            'enable_sgpr_workgroup_id_x'       :   1,
            'enable_sgpr_workgroup_id_y'       :   1,
            'enable_vgpr_workitem_id'          :   0,
            'workgroup_group_segment_byte_size':   self.lds_single_size * self.lds_buffer_num,
            'kernarg_segment_byte_size'         :   88,
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
        '''
        Computes v_sst_os/v_sld_a_os/v_sld_b_os fresh from v_tid (and v_tmp scratch) --
        called once from emit_kernel_prologue, and AGAIN at the start of every tap when
        double-buffered (see emit_kernel_tap_loop). Recomputing (instead of saving/
        restoring a "_base" copy in dedicated VGPRs) avoids needing any new persistent
        registers: bwd's kernel is already at the hard 256-VGPR/wave limit before Phase
        2, so 3 more registers for save/restore would not have fit. Without this reset,
        v_sst_os/v_sld_a_os/v_sld_b_os -- XOR-toggled once per outer K-loop iteration by
        wmma_main_loop.py's double-buffer switch -- would carry whatever buffer-parity
        the PREVIOUS tap's K-loop happened to leave them in into the next tap, silently
        misaligning reads/writes for every tap after the first (caught on real hardware
        as wrong answers specific to multi-tap (y,x>1) configs -- 1x1/single-tap configs
        never exercise a second tap, so they passed).
        '''
        v = self.vgpr
        # ---- fixed per-thread shared-memory store offset: this thread owns row `tid` ----
        # bytes_per_row (=gemm_k_per_block*data_byte) is 64 for fp16/bf16/int8, but 16 for
        # fp32 (gemm_k_per_block forced to 4) -- see class docstring / self.bytes_per_row.
        self._emit(f"v_lshlrev_b32 v[{v.v_sst_os()}], {utility_log2(self.bytes_per_row)}, v[{v.v_tid()}]   ; tid*{self.bytes_per_row} bytes")
        self._emit_empty_line()

        # ---- shared-memory load offsets (WMMA operand layout, see docs/gfx1250_wmma_layout.md) ----
        # v_gemm_in/v_gemm_im outputs land directly in v_sld_b_os/v_sld_a_os (element-unit row/col,
        # contiguous wave-block base already folded in) -- kept distinct from the v_tmp scratch
        # triple the function itself uses internally, to avoid the two aliasing.
        self._emit(self.wmma_mapping.get_gemm_index_for_src_matrix(v.v_sld_b_os(), v.v_sld_a_os(), v.v_tid(), v.v_tmp()))
        self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}], 4, v[{v.v_tid()}]")
        self._emit(f"v_and_b32 v[{v.v_tmp()}], 1, v[{v.v_tmp()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {utility_log2(self.inst_wmma_k_bytes // 2)}, v[{v.v_tmp()}]      ; k_half * {self.inst_wmma_k_bytes // 2} bytes")
        self._emit(f"v_lshlrev_b32 v[{v.v_sld_a_os()}], {utility_log2(self.bytes_per_row)}, v[{v.v_sld_a_os()}]  ; row * {self.bytes_per_row} byte row-stride")
        self._emit(f"v_add_u32 v[{v.v_sld_a_os()}], v[{v.v_tmp()}], v[{v.v_sld_a_os()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_sld_b_os()}], {utility_log2(self.bytes_per_row)}, v[{v.v_sld_b_os()}]")
        self._emit(f"v_add_u32 v[{v.v_sld_b_os()}], v[{v.v_tmp()}], v[{v.v_sld_b_os()}]")
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
        self._emit(f"s_load_dword s[{s.s_y()}], s[{s.s_ka()}:{s.s_ka(1)}], 68")
        self._emit(f"s_load_dword s[{s.s_x()}], s[{s.s_ka()}:{s.s_ka(1)}], 72")
        self._emit(f"s_load_dword s[{s.s_dilation_h()}], s[{s.s_ka()}:{s.s_ka(1)}], 76")
        self._emit(f"s_load_dword s[{s.s_dilation_w()}], s[{s.s_ka()}:{s.s_ka(1)}], 80")
        self._emit(f"s_load_dword s[{s.s_group()}], s[{s.s_ka()}:{s.s_ka(1)}], 84")
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

        m_int_div_rem_vs = macro_int_div_rem_vs_gfx1250_t(self.mc)

        # ---- group>1: decode group_idx out of s_by (group is folded into grid_y by the
        # driver: grid_y = group * ceil(gemm_n/gemm_n_per_block)), and correct s_by in place
        # to the within-group N-block index BEFORE s_block_n_off is computed from it below --
        # blocks_per_group_n needs no division (gemm_n_per_block=128 is a compile-time power
        # of 2), but the group_idx/corrected-s_by split does need a genuine runtime division
        # (blocks_per_group_n is only known at launch time). Reuses the same wave32-safe
        # division macro as everywhere else in this kernel (broadcast s_by to a vgpr, divide,
        # v_readfirstlane_b32 both outputs back to scalar) rather than a new scalar-scalar
        # variant. Applies A's and the output's per-group address offset here (both only need
        # gemm_k/gemm_n, already loaded); B's offset is applied further below, once
        # s_wei_k_stride is available. ----
        self._emit(f"; --- group>1: decode group_idx, correct s_by, offset A/output base pointers ---")
        self._emit(f"s_add_u32 s[{s.s_tmp(0)}], s[{s.s_gemm_n()}], 127")
        self._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], 7   ; blocks_per_group_n = ceil(gemm_n/128)")
        self._emit(f"v_mov_b32 v[{v.v_gtc_tmp(0)}], s[{s.s_by()}]")
        self._emit(m_int_div_rem_vs(v.v_gtc_tmp(1), v.v_gtc_tmp(2), v.v_gtc_tmp(0), s.s_tmp(0), v.v_tmp(), s.s_tmp(1)))
        self._emit(f"v_readfirstlane_b32 s[{s.s_group_idx()}], v[{v.v_gtc_tmp(2)}]   ; group_idx")
        self._emit(f"v_readfirstlane_b32 s[{s.s_by()}], v[{v.v_gtc_tmp(1)}]   ; s_by <- corrected within-group N-block index")
        self._emit_empty_line()

        # ---- group>1: A (input) and the output (D) are NHWC tensors with the group split
        # INTERLEAVED within each pixel's channel dimension, so the per-pixel/row memory
        # stride must be the TOTAL channel count (gemm_k*group / gemm_n*group), not the
        # per-group gemm_k/gemm_n the rest of this kernel uses for the K-reduction size and
        # the base-pointer offset -- those stay unchanged. B (weight) needs no equivalent
        # correction: its group split is the OUTERMOST/block-contiguous dimension ([G][K_out
        # per group][Y][X][C_in per group]), so a group's entire weight sub-tensor is
        # self-contained and the existing wei_k_stride-based addressing is already correct.
        self._emit(f"s_mul_i32 s[{s.s_in_c_total()}], s[{s.s_gemm_k()}], s[{s.s_group()}]")
        self._emit(f"s_mul_i32 s[{s.s_out_k_total()}], s[{s.s_gemm_n()}], s[{s.s_group()}]")
        self._emit_empty_line()

        self._emit(f"; A (input): group offset = group_idx * gemm_k elements")
        self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_group_idx()}], s[{s.s_gemm_k()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], {utility_log2(self.data_byte)}")
        self._emit(f"s_add_u32 s[{s.s_p_in()}], s[{s.s_p_in()}], s[{s.s_tmp(0)}]")
        self._emit(f"s_addc_u32 s[{s.s_p_in(1)}], s[{s.s_p_in(1)}], 0")

        self._emit(f"; output: group offset = group_idx * gemm_n elements (D-operand is always fp32/int32, 4 bytes)")
        self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_group_idx()}], s[{s.s_gemm_n()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], 2")
        self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp(0)}]")
        self._emit(f"s_addc_u32 s[{s.s_p_out(1)}], s[{s.s_p_out(1)}], 0")
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

        # ---- one-time decomposition of this thread's GEMM_M index into (n_idx, ho_idx, wo_idx),
        # kept persistent: every tap re-derives hi_idx/wi_idx from the SAME ho_idx/wo_idx ----
        self._emit(f"s_mul_i32 s[{s.s_hi_wi()}], s[{s.s_hi()}], s[{s.s_wi()}]")
        self._emit(f"; decode this thread's absolute GEMM_M index into (n_idx, ho_idx, wo_idx)")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], s[{s.s_block_m_off()}], v[{v.v_tid()}]   ; m_idx")
        self._emit(m_int_div_rem_vs(v.v_gtc_tmp(1), v.v_n_idx(), v.v_gtc_tmp(0), s.s_ho_wo(), v.v_tmp(), s.s_tmp()))
        self._emit(f"; v_gtc_tmp(1)=hw_idx (rem), v_n_idx=n_idx (quo)")
        self._emit(m_int_div_rem_vs(v.v_wo_idx(), v.v_ho_idx(), v.v_gtc_tmp(1), s.s_wo(), v.v_tmp(), s.s_tmp()))
        self._emit(f"; v_wo_idx=wo_idx (rem), v_ho_idx=ho_idx (quo)")
        self._emit_empty_line()

        # ---- B's row stride is y*x*gemm_k elements for a multi-tap filter (weight layout is
        # [K_out][Y][X][C_in], C_in innermost -- see class docstring), computed once ----
        self._emit(f"; s_wei_k_stride = y * x * gemm_k")
        self._emit(f"s_mul_i32 s[{s.s_wei_k_stride()}], s[{s.s_x()}], s[{s.s_y()}]")
        self._emit(f"s_mul_i32 s[{s.s_wei_k_stride()}], s[{s.s_wei_k_stride()}], s[{s.s_gemm_k()}]")
        self._emit_empty_line()

        # ---- group>1: B's group offset = group_idx * gemm_n * wei_k_stride elements (the
        # per-group weight tensor size, K_per_group*Y*X*C_per_group = gemm_n*wei_k_stride) --
        # applied here (not alongside A/output above) since it needs s_wei_k_stride ----
        self._emit(f"; B (weight): group offset = group_idx * gemm_n * wei_k_stride elements")
        self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_group_idx()}], s[{s.s_gemm_n()}]")
        self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], s[{s.s_wei_k_stride()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], {utility_log2(self.data_byte)}")
        self._emit(f"s_add_u32 s[{s.s_p_wei()}], s[{s.s_p_wei()}], s[{s.s_tmp(0)}]")
        self._emit(f"s_addc_u32 s[{s.s_p_wei(1)}], s[{s.s_p_wei(1)}], 0")
        self._emit_empty_line()

        # ---- B's fixed per-thread row base (this tap's column offset is added fresh every
        # tap in _emit_tap_gather -- see class docstring) ----
        self._emit(f"; v_addr_b_base = p_wei + (block_n_off + tid) * wei_k_stride * {self.data_byte} bytes")
        self._emit(f"v_add_u32 v[{v.v_tmp(1)}], s[{s.s_block_n_off()}], v[{v.v_tid()}]")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_wei_k_stride()}], v[{v.v_tmp(1)}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_tmp(1)}], {utility_log2(self.data_byte)}, v[{v.v_tmp(1)}]")
        self._emit(f"v_mov_b32 v[{v.v_addr_b_base(1)}], s[{s.s_p_wei(1)}]")
        self._emit(f"v_add_co_u32 v[{v.v_addr_b_base()}], vcc_lo, s[{s.s_p_wei()}], v[{v.v_tmp(1)}]")
        self._emit(f"v_add_co_ci_u32 v[{v.v_addr_b_base(1)}], vcc_lo, 0, v[{v.v_addr_b_base(1)}], vcc_lo")
        self._emit_empty_line()


        self._emit_lds_offset_setup()

        # ---- persistent (im, in) for the epilogue, converted to global output indices ----
        self._emit(self.wmma_mapping.get_gemm_index_for_dst_matrix(v.v_gemm_in(), v.v_gemm_im(), v.v_tid(), v.v_tmp()))
        self._emit(f"v_add_u32 v[{v.v_gemm_im()}], s[{s.s_block_m_off()}], v[{v.v_gemm_im()}]")
        self._emit(f"v_add_u32 v[{v.v_gemm_in()}], s[{s.s_block_n_off()}], v[{v.v_gemm_in()}]")
        self._emit_empty_line()

        # ---- Phase 5d: runtime tap-loop counters, initialized once before the loop ----
        self._emit(f"s_mov_b32 s[{s.s_iy()}], 0")
        self._emit_empty_line()

    def _emit_tap_gather(self):
        '''
        Recomputes this tap's A operand address+flag and B operand address, using the
        current s_iy/s_ix (runtime tap-loop counters) plus the persistent v_n_idx/v_ho_idx/
        v_wo_idx (this thread's GEMM_M decomposition, computed once in emit_kernel_prologue).
        Called once per tap iteration (see emit_kernel_tap_loop) -- NOT a compile-time-unrolled
        helper, so its emitted instructions execute y*x times via the runtime branch, not y*x
        times in the binary.
        '''
        s = self.sgpr
        v = self.vgpr
        self._emit(f"; --- per-tap gather: hi_idx = ho_idx*stride_h - pad_h + iy*dilation_h ---")
        self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_iy()}], s[{s.s_dilation_h()}]")
        self._emit(f"s_sub_i32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], s[{s.s_pad_h()}]   ; iy*dilation_h - pad_h")
        self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_ix()}], s[{s.s_dilation_w()}]")
        self._emit(f"s_sub_i32 s[{s.s_tmp(1)}], s[{s.s_tmp(1)}], s[{s.s_pad_w()}]   ; ix*dilation_w - pad_w")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(0)}], s[{s.s_stride_h()}], v[{v.v_ho_idx()}]")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(0)}], s[{s.s_tmp(0)}]   ; hi_idx")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(1)}], s[{s.s_stride_w()}], v[{v.v_wo_idx()}]")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(1)}], v[{v.v_gtc_tmp(1)}], s[{s.s_tmp(1)}]   ; wi_idx")
        self._emit_empty_line()

        self._emit(f"; v_flag = 1 iff (hi_idx, wi_idx) in [0,hi)x[0,wi)")
        self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_hi()}], v[{v.v_gtc_tmp(0)}]")
        self._emit(f"v_cndmask_b32 v[{v.v_flag()}], 0, 1, vcc_lo")
        self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_wi()}], v[{v.v_gtc_tmp(1)}]")
        self._emit(f"v_cndmask_b32 v[{v.v_flag()}], 0, v[{v.v_flag()}], vcc_lo")
        self._emit_empty_line()

        self._emit(f"; row_idx = n_idx*(hi*wi) + hi_idx*wi + wi_idx (meaningless but harmless if")
        self._emit(f"; v_flag==0 -- that lane's global_load_a is EXEC-masked off, see global_load_a_functor)")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(2)}], s[{s.s_hi_wi()}], v[{v.v_n_idx()}]")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(3)}], s[{s.s_wi()}], v[{v.v_gtc_tmp(0)}]")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(2)}], v[{v.v_gtc_tmp(2)}], v[{v.v_gtc_tmp(3)}]")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(2)}], v[{v.v_gtc_tmp(2)}], v[{v.v_gtc_tmp(1)}]")
        self._emit_empty_line()

        self._emit(f"; v_addr_a = p_in + row_idx * in_c_total * {self.data_byte} bytes (in_c_total = gemm_k*group --")
        self._emit(f"; the pixel-to-pixel stride is the TENSOR's total channel count, not the per-group gemm_k, see class docstring)")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(2)}], s[{s.s_in_c_total()}], v[{v.v_gtc_tmp(2)}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_gtc_tmp(2)}], {utility_log2(self.data_byte)}, v[{v.v_gtc_tmp(2)}]")
        self._emit(f"v_mov_b32 v[{v.v_addr_a(1)}], s[{s.s_p_in(1)}]   ; reset high half fresh -- this")
        self._emit(f"                                                ; tap's address is NOT a continuation of the previous tap's")
        self._emit(f"v_add_co_u32 v[{v.v_addr_a()}], vcc_lo, s[{s.s_p_in()}], v[{v.v_gtc_tmp(2)}]")
        self._emit(f"v_add_co_ci_u32 v[{v.v_addr_a(1)}], vcc_lo, 0, v[{v.v_addr_a(1)}], vcc_lo")
        self._emit_empty_line()

        self._emit(f"; --- per-tap B address: v_addr_b = v_addr_b_base + (iy*x+ix)*gemm_k*{self.data_byte} bytes ---")
        self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_iy()}], s[{s.s_x()}]")
        self._emit(f"s_add_i32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], s[{s.s_ix()}]   ; tap linear index")
        self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], s[{s.s_gemm_k()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], {utility_log2(self.data_byte)}   ; tap byte offset")
        self._emit(f"v_add_co_u32 v[{v.v_addr_b()}], vcc_lo, s[{s.s_tmp(2)}], v[{v.v_addr_b_base()}]")
        self._emit(f"v_add_co_ci_u32 v[{v.v_addr_b(1)}], vcc_lo, 0, v[{v.v_addr_b_base(1)}], vcc_lo")
        self._emit_empty_line()

    def emit_kernel_tap_loop(self):
        '''
        Runtime (not compile-time-unrolled) outer loop over the y*x filter taps, wrapping a
        SINGLE static emission of the WMMA K-main-loop -- see class docstring. v_c is never
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
        self._emit_tap_gather()
        # ---- issue the first global loads for this tap (main loop expects this precondition;
        # global_load_a_functor re-zeros v_gld_a on every call, see its own docstring) ----
        self._emit(self.global_load_a_functor()())
        self._emit(self.global_load_b_functor()())
        self._emit_empty_line()

        # ---- the WMMA K-main-loop over C, emitted EXACTLY ONCE here; the runtime branches
        # below re-enter this same code for every tap ----
        self.emit_kernel_fma_main_loop()
        self._emit_empty_line()

        self._emit(f"s_add_u32 s[{s.s_ix()}], s[{s.s_ix()}], 1")
        self._emit(f"s_cmp_lt_u32 s[{s.s_ix()}], s[{s.s_x()}]")
        self._emit(f"s_cbranch_scc1 {label_tap_x}")
        self._emit(f"s_add_u32 s[{s.s_iy()}], s[{s.s_iy()}], 1")
        self._emit(f"s_cmp_lt_u32 s[{s.s_iy()}], s[{s.s_y()}]")
        self._emit(f"s_cbranch_scc1 {label_tap_y}")
        self._emit_empty_line()

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
        Phase 1 (k-sub-loop): stores chunk 0 (already loaded+waited by the caller, via the
        EXISTING global_load_a/b_functor + outer s_wait_loadcnt call sequence in
        wmma_main_loop.py -- unchanged from the original design), then load+wait+stores
        chunks 1..num_k_chunks-1 sequentially, reusing the SAME small v_gld buffer (sized
        to chunk_num_dword{,x4}, one inst_wmma.k-worth -- growing it to hold the whole row
        would exceed the 256-VGPR/wave hardware limit for fp16/bf16/int8).

        Deliberately does NOT overlap remaining chunks' global loads with wmma compute the
        way chunk 0's does: ALL of these stores happen only after shared_store_a/b_functor
        is called, which itself is only reached after emit_wmma_tile()+emit_extra_substeps()
        (ALL of the current tile's LDS reads) have completed -- see class docstring. This
        preserves the single-buffered-LDS safety invariant the original (pre-k-sub-loop)
        design relies on: no wave may overwrite a tile's LDS storage until every wave has
        finished reading it. An early attempt stored chunk 0 immediately after ITS OWN load
        (right after global_load_a/b_functor, well before the other waves' reads of that
        same region were guaranteed done) and produced silent wrong-answer corruption on
        real hardware starting at the 3rd within-workgroup K-block -- caught by comparing
        against naive_conv_fwd_nhwc, not by any assembler/compile-time check.
        '''
        self._emit_sst_chunk(v_gld, v_sst_os, sst_extra_off, 0)
        for c in range(1, self.num_k_chunks):
            self._emit_gld_chunk_load(v_gld, v_addr, c, v_flag=v_flag)
            self._emit(f"s_wait_loadcnt 0x0")
            self._emit_sst_chunk(v_gld, v_sst_os, sst_extra_off, c)

    def global_load_a_functor(self):
        '''
        Zeros v_gld_a on EVERY call (not just once, since Phase 5d's per-tap flag can flip
        between taps -- same discipline Phase 5c/wrw established for its own
        per-iteration-varying gather), then EXEC-masks the load itself so a padding lane's
        v_gld_a simply stays zero for this tap/K-chunk. Only chunk 0's load is issued
        here (matching the original single-chunk design's overlap-with-compute
        placement -- issued early, waited/stored much later in shared_store_a_functor);
        chunks 1..N-1 are loaded+stored later too -- see _emit_sst_remaining_chunks's
        docstring for why. NOTE (Phase 2): double-buffering does NOT change this --
        moving chunk 0's wait+store earlier (right after this call) was tried and
        actively REGRESSED performance, since it throws away the very overlap-with-
        compute this placement exists for; the buffer-reuse constraint (one small
        v_gld_a for all chunks) means chunks 1..N-1 can't be pipelined ahead of their
        own wait regardless of how many LDS buffers exist -- only a true interleaved
        schedule (pairing each chunk's load with a DIFFERENT substep's compute, not yet
        implemented) could change that. Double buffering here exists purely so a
        future interleaved schedule has a safe place to land its early stores.
        '''
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                with outer._deferred_context():
                    outer._emit_gld_chunk_load(v.v_gld_a, v.v_addr_a, 0, v_flag=v.v_flag)
                return outer._get_deferred()
            def get_issues(self):
                return outer.chunk_num_dwordx4
        return functor_t()

    def global_load_b_functor(self):
        ''' See global_load_a_functor's docstring -- B is untransposed too, same treatment. '''
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                with outer._deferred_context():
                    outer._emit_gld_chunk_load(v.v_gld_b, v.v_addr_b, 0, v_flag=None)
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
                    outer._emit_sst_remaining_chunks(v.v_gld_a, v.v_addr_a, v.v_sst_os, 0, v_flag=v.v_flag)
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
                    outer._emit_sst_remaining_chunks(v.v_gld_b, v.v_addr_b, v.v_sst_os, outer.lds_a_size, v_flag=None)
                return outer._get_deferred()
            def get_issues(self):
                return outer.chunk_num_dwordx4
        return functor_t()

    def _emit_ds_read_chunked(self, v_base_sym, v_os_sym, base_off, num_v):
        '''
        Reads `num_v` contiguous dwords from LDS starting at `base_off`, into
        `v_base_sym(0..num_v-1)`, in the largest chunks ds_read_* supports (b128=4 dwords,
        b64=2, b32=1). Generalizes the old fp16/bf16/int8-only "always 2x ds_read_b128 (8
        dwords)" pattern, which hardcoded num_v_a/num_v_b=8 -- fp32 has num_v_a/num_v_b=2, so
        this must read a different (smaller) chunk shape. See class docstring.
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
        step_bytes = outer.tunable.wmma_tile_m * outer.bytes_per_row   # was hardcoded 1024
        class functor_t:
            def __call__(self, v_dst, v_os, extra_off):
                v = outer.vgpr
                with outer._deferred_context():
                    for i_rm in range(outer.tunable.wmma_repeat_m):
                        base = i_rm * step_bytes + extra_off
                        outer._emit_ds_read_chunked(lambda k, i_rm=i_rm: v.v_a(i_rm*num_v_a+k), v.v_sld_a_os, base, num_v_a)
                return outer._get_deferred()
        return functor_t()

    def shared_load_b_functor(self):
        outer = self
        num_v_b = outer.wmma_mapping.ctrl.inst_wmma.num_v_b
        step_bytes = outer.tunable.wmma_tile_n * outer.bytes_per_row   # was hardcoded 1024
        class functor_t:
            def __call__(self, v_dst, v_os, extra_off):
                v = outer.vgpr
                with outer._deferred_context():
                    for i_rn in range(outer.tunable.wmma_repeat_n):
                        base = outer.lds_a_size + i_rn * step_bytes + extra_off  # B region starts after A's region
                        outer._emit_ds_read_chunked(lambda k, i_rn=i_rn: v.v_b(i_rn*num_v_b+k), v.v_sld_b_os, base, num_v_b)
                return outer._get_deferred()
        return functor_t()

    def move_slice_window_a_functor(self):
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                with outer._deferred_context():
                    outer._emit(f"v_add_co_u32 v[{v.v_addr_a()}], vcc_lo, {outer.bytes_per_row}, v[{v.v_addr_a()}]")
                    outer._emit(f"v_add_co_ci_u32 v[{v.v_addr_a(1)}], vcc_lo, 0, v[{v.v_addr_a(1)}], vcc_lo")
                return outer._get_deferred()
        return functor_t()

    def move_slice_window_b_functor(self):
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                with outer._deferred_context():
                    outer._emit(f"v_add_co_u32 v[{v.v_addr_b()}], vcc_lo, {outer.bytes_per_row}, v[{v.v_addr_b()}]")
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
        # Phase 1 (k-sub-loop): both A and B are untransposed here (K contiguous within
        # an LDS row), so advancing inst_wmma.k K-elements is just inst_wmma.k*data_byte.
        ctrl.k_substep_stride_bytes_a    = self.wmma_mapping.ctrl.inst_wmma.k * self.data_byte
        ctrl.k_substep_stride_bytes_b    = self.wmma_mapping.ctrl.inst_wmma.k * self.data_byte
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
        # also the sole caller of this method (see class docstring, Phase 5d)
        wmma_main_loop_t(self.mc, ctrl).emit()

    def emit_kernel_epilogue(self):
        v = self.vgpr
        s = self.sgpr
        # s_out_k_total (=gemm_n*group) is the output tensor's TOTAL row stride (see class
        # docstring's group>1 note) -- s_gemm_n alone (per-group) is only correct for group=1.
        self._emit(self.coalescing_store(v.v_c.label, v.v_gemm_im(), v.v_gemm_in(), s.s_p_out.label, s.s_out_k_total.label, v.v_addr_out.label))
        self._emit(f"s_wait_storecnt 0x0")

    def emit_kernel_body(self):
        self.emit_kernel_prologue()
        self.emit_kernel_tap_loop()
        self.emit_kernel_epilogue()
