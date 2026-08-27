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

        # Asymmetric tile shapes (2026-08-25): the global-load thread mapping historically
        # required block_size == gemm_m_per_block == gemm_n_per_block (one row per thread,
        # addressed directly by v_tid). Generalized here to allow block_size to merely DIVIDE
        # both evenly -- each thread then owns `row_repeat_{a,b}` rows, strided by block_size
        # (thread `tid` owns global rows `tid, tid+block_size, tid+2*block_size, ...`). The
        # WMMA *compute* side (wmma_mapping.get_gemm_index_for_src_matrix) has no dependence on
        # which thread loaded which row -- it only requires LDS byte offset R*bytes_per_row to
        # hold global row R's data, regardless of which thread wrote it -- so this is purely a
        # load-side change. For every existing config block_size==gemm_m/n_per_block exactly,
        # so row_repeat_a==row_repeat_b==1 and every row-repeat loop below degenerates to
        # exactly today's single-iteration code (byte-identical).
        assert tunable.gemm_m_per_block % tunable.block_size == 0, \
            f"gemm_m_per_block({tunable.gemm_m_per_block}) must be a multiple of block_size({tunable.block_size})"
        assert tunable.gemm_n_per_block % tunable.block_size == 0, \
            f"gemm_n_per_block({tunable.gemm_n_per_block}) must be a multiple of block_size({tunable.block_size})"
        self.row_repeat_a = tunable.gemm_m_per_block // tunable.block_size
        self.row_repeat_b = tunable.gemm_n_per_block // tunable.block_size
        # row_repeat > 1 combined with Phase 13's async load is deliberately out of scope for
        # now (kept separate to isolate correctness of each mechanism).
        assert not (tunable.async_global_load and (self.row_repeat_a > 1 or self.row_repeat_b > 1)), \
            "row_repeat > 1 is not yet supported together with async_global_load"
        # Phase 15 (main-loop interleaving): mutually exclusive with async_global_load (no
        # staging buffer to interleave around) and, for this first pass, with row_repeat>1
        # (kept separate to isolate correctness of each mechanism, same discipline as above).
        # REQUIRES lds_double_buffer=1 -- confirmed on real hardware (multi-K-block battery
        # case, nrms=0.035 vs 0.0082 threshold) that single-buffered interleaving races
        # across waves: a wave that stores an early chunk of the NEXT tile can overwrite LDS
        # a slower wave is still reading as part of the CURRENT tile's LATER substeps, since
        # interleaving moves stores much earlier in program order than the non-interleaved
        # path's "defer every store until every substep's read is done" discipline (the
        # implicit cross-wave safety margin that discipline relies on). Double-buffering
        # routes the interleaved stores to the OTHER physical LDS buffer, eliminating the
        # overlap entirely regardless of how early any wave's stores happen.
        assert not (tunable.main_loop_interleave and tunable.async_global_load), \
            "main_loop_interleave is not supported together with async_global_load"
        assert not (tunable.main_loop_interleave and (self.row_repeat_a > 1 or self.row_repeat_b > 1)), \
            "main_loop_interleave is not yet supported together with row_repeat > 1"
        assert not (tunable.main_loop_interleave and not tunable.lds_double_buffer), \
            "main_loop_interleave requires lds_double_buffer=1 (single-buffered interleaving races across waves, confirmed on hardware)"
        # Phase 28 (TDM global load pilot): narrowest correctness-first slice, same discipline
        # as the asserts above -- row_repeat_a>1 and local_prefetch_num>1 are kept separate
        # for now to isolate correctness of each mechanism.
        assert not (tunable.tdm_global_load and self.row_repeat_a > 1), \
            "tdm_global_load is not yet supported together with row_repeat_a > 1"
        assert not (tunable.tdm_global_load and tunable.local_prefetch_num > 1), \
            "tdm_global_load is not yet supported together with local_prefetch_num > 1"
        # Phase 29: fresh-label counter for _emit_wave0_only -- the TDM issue site is
        # reached from more than one place in the Python source (initial issue before the
        # loop, re-issue inside the loop body), each needing its own unique skip-label.
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
        # Phase 27: coalescing_store_wmma.py's ctrl field is named wmma_acc_f16 but its actual
        # behavior is precision-agnostic ("is the accumulator 2-byte-packed"), proven by
        # bf16-accumulate needing zero changes there -- both tunables funnel into this one
        # ctrl field rather than adding a second, identical branch.
        ctrl_coalescing_store_wmma.wmma_acc_f16 = tunable.wmma_acc_f16 or tunable.wmma_acc_bf16
        ctrl_coalescing_store_wmma.wmma_m_tail = tunable.wmma_m_tail
        ctrl_coalescing_store_wmma.wmma_n_tail = tunable.wmma_n_tail
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
            if outer.tunable.tdm_global_load:
                # Phase 28: TDM descriptor for the A operand -- group0 (4 SGPRs: pred,
                # lds_addr, global_addr lo/hi|type) and group1 (8 SGPRs: config/mask,
                # tensor_dim0/1, tile_dim0/1/2, tensor_dim0/1_stride). Only allocated when
                # tdm_global_load is set (every existing config byte-identical).
                self.s_tdm_g0  = sym_t('s_tdm_g0'      , sseq(4, 4))
                self.s_tdm_g1  = sym_t('s_tdm_g1'      , sseq(8, 4))
                # Phase 29: this wave's index within the workgroup (0, 1, 2, ...), used to
                # gate TDM issue to a single wave -- TDM instructions ignore EXEC entirely
                # (not per-lane), so a scalar branch on this is the only way to suppress
                # redundant per-wave issues.
                self.s_wave_id = sym_t('s_wave_id'     , sseq(1))
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
            if outer.tunable.async_global_load:
                # Phase 13: no VGPR staging buffer needed at all -- global_load_async_to_lds_b128
                # writes straight to LDS. v_zero is a persistent all-zero quad used to explicitly
                # zero-fill padding lanes' LDS destinations (see global_load_a_functor).
                self.v_zero    = sym_t('v_zero'        , vseq(4))
            else:
                # Phase 1 (k-sub-loop): sized to outer.chunk_num_dwords (one inst_wmma.k-worth),
                # NOT outer.num_dwords (the whole, possibly multi-substep, row) -- global_load/
                # shared_store are now chunked and reuse this same small buffer across chunks,
                # since growing it to hold the whole row would exceed the 256-VGPR/wave limit
                # for fp16/bf16/int8. See self.chunk_num_dwords in __init__.
                self.v_gld_a   = sym_t('v_gld_a'       , vseq(outer.chunk_num_dwords))
                self.v_gld_b   = sym_t('v_gld_b'       , vseq(outer.chunk_num_dwords))
            self.v_tid         = sym_t('v_tid'         , vseq(1))
            if outer.tunable.async_global_load:
                # Phase 13: global_load_async_to_lds_b128's VADDR is a plain 32-bit per-lane
                # byte OFFSET (SADDR carries the 64-bit base separately) -- no need for a
                # 2-VGPR-aligned full address pair like the old global_load_dwordx4 path.
                self.v_off_a      = sym_t('v_off_a'      , vseq(1))
                self.v_off_b      = sym_t('v_off_b'      , vseq(1))
                self.v_off_b_base = sym_t('v_off_b_base' , vseq(1))
                # Phase 13 bugfix: global_load_async_to_lds_b128's immediate `offset:N` shifts
                # BOTH the LDS destination (VDST) and the global source address (VADDR+SADDR)
                # by the same N (verified on real hardware -- unlike the old design's separate
                # global_load+ds_write, where the store's offset never touched the load's
                # source). So a nonzero sst_extra_off (e.g. B's lds_a_size region shift) must be
                # baked into the VDST *register* once, not the shared per-chunk immediate --
                # otherwise it also shifts the source read address, reading garbage/OOB memory.
                self.v_sst_tmp    = sym_t('v_sst_tmp'    , vseq(1))
            else:
                # 64-bit VADDR pairs must be even-aligned on gfx1250 (verified with llvm-mc).
                # row_repeat_a copies (one pair per row this thread owns -- see __init__'s
                # row_repeat_a docstring); vseq(2*row_repeat_a, 2) keeps EVERY pair
                # (v_addr_a(i*2), v_addr_a(i*2+1)) even-aligned since the whole block starts
                # even-aligned. row_repeat_a==1 for every existing config (byte-identical).
                self.v_addr_a      = sym_t('v_addr_a'      , vseq(2 * outer.row_repeat_a, 2))    # persistent global A address(es) (64-bit each)
                # row_repeat_b copies, mirroring v_addr_a/row_repeat_a above -- B needs no
                # flag/masking (weight is never out of bounds), so this is the whole story:
                # each row's address pair just advances independently. row_repeat_b==1 for
                # every existing config (byte-identical).
                self.v_addr_b      = sym_t('v_addr_b'      , vseq(2 * outer.row_repeat_b, 2))
                # Phase 5d: B's fixed per-thread row base(s) (before this tap's column offset
                # is added) -- computed once from the *y*x*c* row stride, reused every tap.
                self.v_addr_b_base = sym_t('v_addr_b_base' , vseq(2 * outer.row_repeat_b, 2))
            self.v_addr_out    = sym_t('v_addr_out'    , vseq(2))    # scratch used by coalescing_store_wmma (ping-pong pair)
            if outer.tunable.wmma_m_tail:
                # Phase 25: extra scratch for coalescing_store_wmma's per-pass absolute-row
                # EXEC-mask guard -- only allocated when wmma_m_tail is set (every existing
                # config is byte-identical, this register simply doesn't exist otherwise).
                self.v_m_tail_row = sym_t('v_m_tail_row' , vseq(1))
            if outer.tunable.wmma_n_tail:
                # Phase 26b: extra scratch for coalescing_store_wmma's pass-invariant
                # column-in-range flag -- only allocated when wmma_n_tail is set.
                self.v_n_tail_col = sym_t('v_n_tail_col' , vseq(1))
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
            # v_flag: row_repeat_a copies (must persist across the K-loop for every row this
            # thread owns -- see __init__'s row_repeat_a docstring). ==1 for every existing
            # config (byte-identical). v_n_idx/v_ho_idx/v_wo_idx stay SINGLE registers even
            # for row_repeat_a>1 -- only row 0's decomposition is persisted (exactly as
            # today); rows 1..row_repeat_a-1 recompute their own (n_idx,ho_idx,wo_idx) FRESH
            # every tap inside _emit_tap_gather, reusing v_gtc_tmp's existing 5 scratch slots
            # (no extra persistent VGPRs) -- see that function's docstring. This keeps the
            # asymmetric shape's VGPR cost to +3 total (v_flag +1, v_addr_a +2) instead of +6.
            self.v_flag        = sym_t('v_flag'        , vseq(outer.row_repeat_a))
            if outer.tunable.wmma_n_tail:
                # Phase 26b: B's column (block_n_off + tid) is a kernel-lifetime constant
                # (unlike A's per-tap v_flag) -- computed once in emit_kernel_prologue, not
                # per-tap. Only allocated when wmma_n_tail is set (every existing config byte-
                # identical).
                self.v_flag_b  = sym_t('v_flag_b'      , vseq(1))
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
        # the LDS-reshuffle epilogue (coalescing_store_wmma.py) reuses the main loop's LDS
        # region for the whole output tile (post-barrier, temporally disjoint use) -- needs
        # more than the main loop alone if the tile is bigger than what the main loop reserved
        # (always true here: main-loop LDS is sized by A/B *input* staging, epilogue LDS by the
        # *output* tile, which is a fixed 4 bytes/element regardless of precision).
        # Phase 23: epilogue_lds_pad adds 4 padding elements per row to break a bank-conflict
        # periodicity (see coalescing_store_wmma.py) -- reflect that in the LDS size too.
        epilogue_pad = 4 if self.tunable.epilogue_lds_pad else 0
        # Phase 24: f16acc's epilogue stages genuinely 2-byte-per-element LDS data (see
        # coalescing_store_wmma.py's scatter), half the f32 case's footprint.
        epilogue_elem_bytes = 2 if (self.tunable.wmma_acc_f16 or self.tunable.wmma_acc_bf16) else 4
        epilogue_lds_bytes = self.tunable.gemm_m_per_block * (self.tunable.gemm_n_per_block + epilogue_pad) * epilogue_elem_bytes
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

    def _emit_tdm_descriptor_setup_a(self):
        '''
        Phase 28 (TDM global-load pilot, fwd/1x1 only): builds the A-operand's TDM
        descriptor (group0: 4 SGPRs, group1: 8 SGPRs) ONCE, before the main loop --
        `s_block_m_off`/`s_p_in`/`s_in_c_total`/`s_gemm_m`/`s_gemm_k` must already be set
        (this is called right after `s_knum` in emit_kernel_prologue). Bit-packing verified
        against the CDNA5 ISA doc (10.11.4, Tables 62-63) and cross-checked against FlyDSL's
        own GFX1250 CopyAtom.cpp, then confirmed correct on real hardware via a standalone
        probe (load + store, both directions' OOB behavior) compiled through MISA's actual
        `-x assembler` pipeline and launched via `hipModuleLoad`+`hipExtModuleLaunchKernel`
        (this project's real dispatch call) -- see docs/gfx1250_wmma_layout.md's Phase 28.

        tile_dim0 (gemm_k_per_block) and tile_dim1 (gemm_m_per_block) are compile-time
        constants (baked in as immediates); tensor_dim0 (gemm_k), tensor_dim1 (gemm_m), and
        tensor_dim0_stride (in_c_total) are runtime SGPR values (real conv shape). Every
        wave in the workgroup issues an identical, redundant TDM load (TDM ignores EXEC and
        isn't per-lane) -- correctness-first for this pilot; only-wave-0 issuing is a
        follow-up optimization, not attempted here.
        '''
        s = self.sgpr
        data_size_code = utility_log2(self.data_byte)
        tile_dim0 = self.tunable.gemm_k_per_block
        tile_dim1 = self.tunable.gemm_m_per_block
        assert tile_dim0 < 65536 and tile_dim1 < 65536, "TDM tile_dim0/1 are 16-bit fields"

        self._emit(f"; --- Phase 28: TDM descriptor for A operand ---")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g0(0)}], 1   ; group0: pred=1 (valid tensor)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g0(1)}], 0   ; group0: lds_addr (A's LDS region starts at byte 0)")
        self._emit(f"; group0: global_addr = p_in + block_m_off * in_c_total * data_byte")
        self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_block_m_off()}], s[{s.s_in_c_total()}]")
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
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1(5)}], s[{s.s_in_c_total()}]   ; tensor_dim0_stride lo32 (elements)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1(6)}], 0   ; tensor_dim0_stride hi16 (assume < 2^32 elements)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1(7)}], 0   ; tensor_dim1_stride unused (2D tensor)")
        self._emit_empty_line()

    def _emit_wave0_only(self, body_fn):
        '''
        Phase 29 (single-issuer-wave): wraps body_fn() -- a callable that emits
        instructions via self._emit -- in a scalar branch so only wave 0 (of however many
        waves make up this workgroup) executes it. TDM instructions ignore EXEC entirely
        (confirmed against the CDNA5 ISA doc: "issued no matter if EXEC==0... makes no
        difference which lanes are enabled or disabled"), so an EXEC-mask/v_cmpx approach
        -- which works for every OTHER per-lane masking mechanism in this kernel -- cannot
        suppress a redundant TDM issue on non-issuing waves; only a genuine s_cbranch can.

        Every call gets a FRESH label (self._tdm_label_counter) even though this method may
        be invoked multiple times across the kernel's source (initial issue before the main
        loop, re-issue inside the loop body) -- each is a distinct physical instruction
        stream (the loop body's copy executes many times at runtime via s_branch, but the
        Python-level emission happens once per call site), so each needs its own unique
        skip-target label to avoid an assembler "duplicate symbol" error.
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
        if self.tunable.tdm_global_load:
            # Phase 29: derive this wave's index within the workgroup, once, while EXEC is
            # still fully enabled (kernel entry, before any lane-disabling branch) -- lane 0
            # is guaranteed to be the "first active lane" v_readfirstlane_b32 reads here.
            self._emit(f"v_readfirstlane_b32 s[{s.s_wave_id()}], v[{v.v_tid()}]   ; Phase 29: lane 0's flat tid = this wave's base")
            self._emit(f"s_lshr_b32 s[{s.s_wave_id()}], s[{s.s_wave_id()}], 5   ; wave index within workgroup")
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
        self._emit(f"s_add_u32 s[{s.s_tmp(0)}], s[{s.s_gemm_n()}], {self.tunable.gemm_n_per_block - 1}")
        self._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], {utility_log2(self.tunable.gemm_n_per_block)}   ; blocks_per_group_n = ceil(gemm_n/gemm_n_per_block)")
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

        # Phase 24: shift must follow the D-operand's real width (4 bytes normally, 2 under
        # wmma_acc_f16) -- see the identical bug found and fixed in wrw's per-tap output
        # offset (igemm_wrw_gtc_wmma_nhwc.py's emit_kernel_tap_loop).
        out_elem_byte_shift = 1 if (self.tunable.wmma_acc_f16 or self.tunable.wmma_acc_bf16) else 2
        self._emit(f"; output: group offset = group_idx * gemm_n elements (D-operand is fp32/int32 (4B) normally, fp16 (2B) under wmma_acc_f16)")
        self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_group_idx()}], s[{s.s_gemm_n()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], {out_elem_byte_shift}")
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

        if self.tunable.async_global_load:
            self._emit(f"; Phase 13: persistent zero quad, used to zero-fill padding lanes' LDS")
            self._emit(f"; destinations after a masked global_load_async_to_lds_b128 (see global_load_a_functor)")
            for i in range(4):
                self._emit(f"v_mov_b32 v[{v.v_zero(i)}], 0")
            self._emit_empty_line()

        self._emit(f"s_lshl_b32 s[{s.s_block_m_off()}], s[{s.s_bx()}], {utility_log2(self.tunable.gemm_m_per_block)}   ; *gemm_m_per_block")
        self._emit(f"s_lshl_b32 s[{s.s_block_n_off()}], s[{s.s_by()}], {utility_log2(self.tunable.gemm_n_per_block)}   ; *gemm_n_per_block")
        self._emit(f"s_mov_b32 s[{s.s_knum()}], s[{s.s_gemm_k()}]")
        self._emit_empty_line()

        if self.tunable.tdm_global_load:
            self._emit_tdm_descriptor_setup_a()

        # ---- one-time decomposition of this thread's GEMM_M index into (n_idx, ho_idx, wo_idx),
        # kept persistent: every tap re-derives hi_idx/wi_idx from the SAME ho_idx/wo_idx.
        # row_repeat_a>1 (asymmetric tile shapes): this is ONLY row 0's decomposition --
        # rows 1..row_repeat_a-1 have no persistent registers of their own (to keep VGPR cost
        # down) and instead recompute their own (n_idx, ho_idx, wo_idx) FRESH every tap
        # inside _emit_tap_gather, using v_tid+i*block_size -- see that function's docstring.
        # row_repeat_a==1 for every existing config, so this is exactly today's code ----
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
        if self.tunable.async_global_load:
            self._emit(f"; v_off_b_base = (block_n_off + tid) * wei_k_stride * {self.data_byte} bytes")
            self._emit(f"; (Phase 13: byte OFFSET only -- s_p_wei is passed separately as SADDR)")
            self._emit(f"v_add_u32 v[{v.v_off_b_base()}], s[{s.s_block_n_off()}], v[{v.v_tid()}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_off_b_base()}], s[{s.s_wei_k_stride()}], v[{v.v_off_b_base()}]")
            self._emit(f"v_lshlrev_b32 v[{v.v_off_b_base()}], {utility_log2(self.data_byte)}, v[{v.v_off_b_base()}]")
            self._emit_empty_line()
        else:
            # row_repeat_b>1 (asymmetric tile shapes, B side): thread tid owns rows
            # tid, tid+block_size, tid+2*block_size, ... -- B needs no flag/masking (weight
            # is never out of bounds), so each row is simply its own independent base
            # address, no persistent decomposition or scratch-reuse subtlety like A's rows
            # needed. row_repeat_b==1 for every existing config: the loop runs once with
            # i=0, producing v_addr_b_base(0)/(1) == v_addr_b_base()/(1), byte-identical.
            for i in range(self.row_repeat_b):
                tag = '' if i == 0 else f'({i})'
                self._emit(f"; v_addr_b_base{tag} = p_wei + (block_n_off + tid + {i}*block_size) * wei_k_stride * {self.data_byte} bytes")
                if i == 0:
                    self._emit(f"v_add_u32 v[{v.v_tmp(1)}], s[{s.s_block_n_off()}], v[{v.v_tid()}]")
                else:
                    self._emit(f"v_add_u32 v[{v.v_tmp(1)}], {i * self.tunable.block_size}, v[{v.v_tid()}]")
                    self._emit(f"v_add_u32 v[{v.v_tmp(1)}], s[{s.s_block_n_off()}], v[{v.v_tmp(1)}]")
                if i == 0 and self.tunable.wmma_n_tail:
                    # Phase 26b: v_flag_b = 1 iff this lane's absolute column < real gemm_n --
                    # a kernel-lifetime constant, computed once here (only row 0; row_repeat_b
                    # > 1 has no masking support at all today, matching the pre-existing scope
                    # narrowing this loop's own docstring already describes for B).
                    self._emit(f"; wmma_n_tail: v_flag_b = 1 iff this lane's absolute column < real gemm_n")
                    self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_gemm_n()}], v[{v.v_tmp(1)}]")
                    self._emit(f"v_cndmask_b32 v[{v.v_flag_b()}], 0, 1, vcc_lo")
                self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_wei_k_stride()}], v[{v.v_tmp(1)}]")
                self._emit(f"v_lshlrev_b32 v[{v.v_tmp(1)}], {utility_log2(self.data_byte)}, v[{v.v_tmp(1)}]")
                self._emit(f"v_mov_b32 v[{v.v_addr_b_base(i*2+1)}], s[{s.s_p_wei(1)}]")
                self._emit(f"v_add_co_u32 v[{v.v_addr_b_base(i*2)}], vcc_lo, s[{s.s_p_wei()}], v[{v.v_tmp(1)}]")
                self._emit(f"v_add_co_ci_u32 v[{v.v_addr_b_base(i*2+1)}], vcc_lo, 0, v[{v.v_addr_b_base(i*2+1)}], vcc_lo")
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

        A's computation is looped row_repeat_a times (once per row this thread owns -- see
        __init__'s row_repeat_a docstring). Row 0 uses the persistent v_n_idx/v_ho_idx/
        v_wo_idx exactly as today. Rows 1..row_repeat_a-1 have no persistent registers of
        their own -- they recompute (n_idx, ho_idx, wo_idx) FRESH here from
        v_tid+i*block_size, using v_gtc_tmp(3)/(4) as scratch for the two division-macro
        calls, landing the results in v_gtc_tmp(0)/(1)/(2) -- the SAME registers the
        unchanged downstream hi_idx/wi_idx/flag/row_idx code already reads its
        ho_idx/wo_idx/n_idx input from, just substituting the source symbol per row (an
        in-place read-then-write on gtc_tmp(0)/(1)/(2) for rows 1+, a fresh write from the
        persistent registers for row 0) -- so the entire downstream sequence is shared,
        unmodified, between all rows. This keeps VGPR cost to +3 total for row_repeat_a=2
        (v_flag +1, v_addr_a +2) instead of +6 (no extra persistent n_idx/ho_idx/wo_idx
        registers). row_repeat_a==1 for every existing config, so this loop runs once with
        i=0 and takes the row-0 branch (byte-identical). B is untouched -- row_repeat_b==1
        always for this phase (see __init__'s assert).
        '''
        s = self.sgpr
        v = self.vgpr
        m_int_div_rem_vs = macro_int_div_rem_vs_gfx1250_t(self.mc)
        self._emit(f"; --- per-tap gather: hi_idx = ho_idx*stride_h - pad_h + iy*dilation_h ---")
        self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_iy()}], s[{s.s_dilation_h()}]")
        self._emit(f"s_sub_i32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], s[{s.s_pad_h()}]   ; iy*dilation_h - pad_h")
        self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_ix()}], s[{s.s_dilation_w()}]")
        self._emit(f"s_sub_i32 s[{s.s_tmp(1)}], s[{s.s_tmp(1)}], s[{s.s_pad_w()}]   ; ix*dilation_w - pad_w")

        for i in range(self.row_repeat_a):
            tag = '' if i == 0 else f'({i})'
            if i == 0:
                ho_src, wo_src, n_src = v.v_ho_idx(), v.v_wo_idx(), v.v_n_idx()
            else:
                self._emit(f"v_add_u32 v[{v.v_gtc_tmp(4)}], {i * self.tunable.block_size}, v[{v.v_tid()}]")
                self._emit(f"v_add_u32 v[{v.v_gtc_tmp(4)}], s[{s.s_block_m_off()}], v[{v.v_gtc_tmp(4)}]   ; m_idx (row {i})")
                # NOTE: scalar scratch here MUST NOT be s.s_tmp()/s.s_tmp(1) (==s_tmp(0)/(1)) --
                # those hold the shared iy*dilation-pad values computed once above and read
                # again below for hi_idx/wi_idx (bug found on real hardware: the division
                # macro internally overwrites its passed s_tmp4+0, silently corrupting the
                # pad offset for every row using this fresh-recompute path). s.s_tmp(2) is
                # free until the B computation after this loop, so it's safe scratch here.
                self._emit(m_int_div_rem_vs(v.v_gtc_tmp(3), v.v_gtc_tmp(2), v.v_gtc_tmp(4), s.s_ho_wo(), v.v_tmp(), s.s_tmp(2)))
                self._emit(m_int_div_rem_vs(v.v_gtc_tmp(1), v.v_gtc_tmp(0), v.v_gtc_tmp(3), s.s_wo(), v.v_tmp(), s.s_tmp(2)))
                self._emit(f"; v_gtc_tmp(0)=ho_idx({i}), v_gtc_tmp(1)=wo_idx({i}), v_gtc_tmp(2)=n_idx({i})")
                ho_src, wo_src, n_src = v.v_gtc_tmp(0), v.v_gtc_tmp(1), v.v_gtc_tmp(2)
            self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(0)}], s[{s.s_stride_h()}], v[{ho_src}]")
            self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(0)}], s[{s.s_tmp(0)}]   ; hi_idx{tag}")
            self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(1)}], s[{s.s_stride_w()}], v[{wo_src}]")
            self._emit(f"v_add_u32 v[{v.v_gtc_tmp(1)}], v[{v.v_gtc_tmp(1)}], s[{s.s_tmp(1)}]   ; wi_idx{tag}")
            self._emit_empty_line()

            self._emit(f"; v_flag{tag} = 1 iff (hi_idx, wi_idx) in [0,hi)x[0,wi)")
            self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_hi()}], v[{v.v_gtc_tmp(0)}]")
            self._emit(f"v_cndmask_b32 v[{v.v_flag(i)}], 0, 1, vcc_lo")
            self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_wi()}], v[{v.v_gtc_tmp(1)}]")
            self._emit(f"v_cndmask_b32 v[{v.v_flag(i)}], 0, v[{v.v_flag(i)}], vcc_lo")
            self._emit_empty_line()

            if self.tunable.wmma_m_tail:
                # Phase 25: v_flag also gates on this row's absolute GEMM_M index (the same
                # kind of OOB condition as hi/wi above, just against the tail block's real
                # gemm_m instead of the input tensor's spatial extent). v_gtc_tmp(4) is free
                # here for i==0 (unused until this point in that branch) and is recomputed
                # fresh rather than reused from i>0's pre-division value at line ~754, to
                # avoid depending on whether the division macro above preserves its dividend.
                self._emit(f"; wmma_m_tail: v_flag{tag} &= (this row's absolute GEMM_M index < real gemm_m)")
                if i == 0:
                    self._emit(f"v_add_u32 v[{v.v_gtc_tmp(4)}], s[{s.s_block_m_off()}], v[{v.v_tid()}]")
                else:
                    self._emit(f"v_add_u32 v[{v.v_gtc_tmp(4)}], {i * self.tunable.block_size}, v[{v.v_tid()}]")
                    self._emit(f"v_add_u32 v[{v.v_gtc_tmp(4)}], s[{s.s_block_m_off()}], v[{v.v_gtc_tmp(4)}]")
                self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_gemm_m()}], v[{v.v_gtc_tmp(4)}]")
                self._emit(f"v_cndmask_b32 v[{v.v_flag(i)}], 0, v[{v.v_flag(i)}], vcc_lo")
                self._emit_empty_line()

            self._emit(f"; row_idx = n_idx*(hi*wi) + hi_idx*wi + wi_idx (meaningless but harmless if")
            self._emit(f"; v_flag==0 -- that lane's global_load_a is EXEC-masked off, see global_load_a_functor)")
            self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(2)}], s[{s.s_hi_wi()}], v[{n_src}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(3)}], s[{s.s_wi()}], v[{v.v_gtc_tmp(0)}]")
            self._emit(f"v_add_u32 v[{v.v_gtc_tmp(2)}], v[{v.v_gtc_tmp(2)}], v[{v.v_gtc_tmp(3)}]")
            self._emit(f"v_add_u32 v[{v.v_gtc_tmp(2)}], v[{v.v_gtc_tmp(2)}], v[{v.v_gtc_tmp(1)}]")
            self._emit_empty_line()

            if self.tunable.async_global_load:
                self._emit(f"; v_off_a = row_idx * in_c_total * {self.data_byte} bytes (Phase 13: byte OFFSET only --")
                self._emit(f"; s_p_in is passed separately as SADDR; in_c_total = gemm_k*group, see class docstring)")
                self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(2)}], s[{s.s_in_c_total()}], v[{v.v_gtc_tmp(2)}]")
                self._emit(f"v_lshlrev_b32 v[{v.v_off_a()}], {utility_log2(self.data_byte)}, v[{v.v_gtc_tmp(2)}]")
                self._emit_empty_line()
            else:
                self._emit(f"; v_addr_a{tag} = p_in + row_idx * in_c_total * {self.data_byte} bytes (in_c_total = gemm_k*group --")
                self._emit(f"; the pixel-to-pixel stride is the TENSOR's total channel count, not the per-group gemm_k, see class docstring)")
                self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(2)}], s[{s.s_in_c_total()}], v[{v.v_gtc_tmp(2)}]")
                self._emit(f"v_lshlrev_b32 v[{v.v_gtc_tmp(2)}], {utility_log2(self.data_byte)}, v[{v.v_gtc_tmp(2)}]")
                self._emit(f"v_mov_b32 v[{v.v_addr_a(i*2+1)}], s[{s.s_p_in(1)}]   ; reset high half fresh -- this")
                self._emit(f"                                                ; tap's address is NOT a continuation of the previous tap's")
                self._emit(f"v_add_co_u32 v[{v.v_addr_a(i*2)}], vcc_lo, s[{s.s_p_in()}], v[{v.v_gtc_tmp(2)}]")
                self._emit(f"v_add_co_ci_u32 v[{v.v_addr_a(i*2+1)}], vcc_lo, 0, v[{v.v_addr_a(i*2+1)}], vcc_lo")
                self._emit_empty_line()

        if self.tunable.async_global_load:
            self._emit(f"; --- per-tap B offset: v_off_b = v_off_b_base + (iy*x+ix)*gemm_k*{self.data_byte} bytes ---")
            self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_iy()}], s[{s.s_x()}]")
            self._emit(f"s_add_i32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], s[{s.s_ix()}]   ; tap linear index")
            self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], s[{s.s_gemm_k()}]")
            self._emit(f"s_lshl_b32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], {utility_log2(self.data_byte)}   ; tap byte offset")
            self._emit(f"v_add_u32 v[{v.v_off_b()}], s[{s.s_tmp(2)}], v[{v.v_off_b_base()}]")
            self._emit_empty_line()
        else:
            self._emit(f"; --- per-tap B address(es): v_addr_b = v_addr_b_base + (iy*x+ix)*gemm_k*{self.data_byte} bytes ---")
            self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_iy()}], s[{s.s_x()}]")
            self._emit(f"s_add_i32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], s[{s.s_ix()}]   ; tap linear index")
            self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], s[{s.s_gemm_k()}]")
            self._emit(f"s_lshl_b32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], {utility_log2(self.data_byte)}   ; tap byte offset")
            for i in range(self.row_repeat_b):
                self._emit(f"v_add_co_u32 v[{v.v_addr_b(i*2)}], vcc_lo, s[{s.s_tmp(2)}], v[{v.v_addr_b_base(i*2)}]")
                self._emit(f"v_add_co_ci_u32 v[{v.v_addr_b(i*2+1)}], vcc_lo, 0, v[{v.v_addr_b_base(i*2+1)}], vcc_lo")
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

    def _emit_sst_all_chunks_row(self, v_gld, v_addr, v_sst_os, sst_extra_off, v_flag=None):
        '''
        Like _emit_sst_remaining_chunks, but loads+stores chunk 0 too (no early-issue
        precondition) -- used for row_repeat_a's "extra" rows (1..row_repeat_a-1, see
        __init__'s row_repeat_a docstring), which have no early-overlap slot of their own
        (only row 0 does, matching today's exact single-row code path -- see
        global_load_a_functor's docstring for why giving a second row its own early slot
        would reuse v_gld_a while row 0's early load is still in flight). Fully sequential:
        each chunk is loaded, waited, and stored before the next starts, exactly like
        _emit_sst_remaining_chunks's chunks 1..N-1 -- just starting from chunk 0 instead.
        Only reached for row_repeat_a > 1 (non-async only), never for any existing config.
        '''
        for c in range(self.num_k_chunks):
            self._emit_gld_chunk_load(v_gld, v_addr, c, v_flag=v_flag)
            self._emit(f"s_wait_loadcnt 0x0")
            self._emit_sst_chunk(v_gld, v_sst_os, sst_extra_off, c)

    def _emit_gld_async_all_chunks(self, v_off, v_sst_os, sst_extra_off, s_saddr, v_flag=None):
        '''
        Phase 13: issues ALL num_k_chunks worth of global_load_async_to_lds_b128 for this
        operand directly (no VGPR staging, no separate store step -- the load IS the LDS
        write). Since there's no small reused buffer to serialize on (unlike the old
        _emit_gld_chunk_load/_emit_sst_remaining_chunks split), every chunk fires
        independently; the caller waits once via s_wait_asynccnt (see wmma_main_loop.py).

        Masking (v_flag, A operand only): a masked-off (EXEC-inactive) lane's async load
        simply never touches its LDS destination -- confirmed on real hardware via a
        dedicated probe (see docs/gfx1250_wmma_layout.md's Phase 13 section) -- so unlike
        the old design (which pre-zeroed v_gld_a so an unconditional store would write
        zero), here we must explicitly zero-fill the SAME destinations for the
        now-inactive lanes with an inverted-EXEC ds_write_b128 from the persistent
        v_zero quad, after the masked async loads.

        sst_extra_off (e.g. B's lds_a_size region shift) is folded into the VDST register
        once via v_sst_tmp, NOT into the per-chunk immediate -- see v_sst_tmp's declaration
        for why (the immediate shifts the global source address too, on this instruction).
        '''
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

        Phase 13 (async_global_load=1): entirely different design -- issues ALL chunks
        directly to LDS via global_load_async_to_lds_b128 (see _emit_gld_async_all_chunks),
        no scratch buffer, no early/overlapped chunk-0 trick (there's nothing to overlap
        with -- the load is the store). shared_store_a_functor is not called at all in
        this mode; wmma_main_loop.py only invokes this functor, at a different position
        than the non-async path (after the current tile's compute, not before).

        row_repeat_a > 1 (asymmetric tile shapes, non-async only -- see __init__'s
        row_repeat_a docstring and its assert forbidding this combined with async):
        ONLY row 0 gets this early-issue/overlap treatment, using v_addr_a/v_flag exactly as
        before (row_repeat_a==1's code path, unchanged). Rows 1..row_repeat_a-1 have no early
        slot of their own -- reusing v_gld_a for a second in-flight early load before row 0's
        is stored would be exactly the kind of scratch-buffer lifetime conflict Phase 1's
        bug #3 already found the hard way (see docs/gfx1250_wmma_layout.md's Phase 9 section)
        -- so they are handled entirely inside shared_store_a_functor instead (fully
        deferred, no overlap, via _emit_sst_all_chunks_row).
        '''
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                s = outer.sgpr
                with outer._deferred_context():
                    if outer.tunable.tdm_global_load:
                        # Phase 28/29: one TDM instruction moves the whole gemm_m_per_block x
                        # gemm_k_per_block tile straight into LDS -- no VGPR staging, no
                        # per-lane masking (EXEC is ignored by tensor instructions). Phase 29:
                        # only wave 0 issues it -- see _emit_wave0_only's docstring for why a
                        # scalar branch, not EXEC-masking, is required to suppress this on
                        # non-issuing waves.
                        outer._emit_wave0_only(lambda: outer._emit(f"tensor_load_to_lds s[{s.s_tdm_g0()}:{s.s_tdm_g0(3)}], s[{s.s_tdm_g1()}:{s.s_tdm_g1(7)}]"))
                    elif outer.tunable.async_global_load:
                        outer._emit_gld_async_all_chunks(v.v_off_a, v.v_sst_os, 0, outer.sgpr.s_p_in, v_flag=v.v_flag)
                    else:
                        outer._emit_gld_chunk_load(v.v_gld_a, v.v_addr_a, 0, v_flag=v.v_flag)
                return outer._get_deferred()
            def get_issues(self):
                return outer.chunk_num_dwordx4
        return functor_t()

    def global_load_b_functor(self):
        ''' See global_load_a_functor's docstring -- B is untransposed too, same treatment
        (Phase 13: no masking needed for B, mirrors the non-async path's v_flag=None).

        row_repeat_b > 1 (non-async only, asserted mutually exclusive in __init__): only row
        0 gets this early-issue/overlap slot -- rows 1..row_repeat_b-1 have no v_gld_b of
        their own to reuse safely while row 0's early load is still in flight (same reasoning
        as row_repeat_a's rows 1+, see global_load_a_functor's docstring), so they are handled
        entirely inside shared_store_b_functor instead. '''
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                with outer._deferred_context():
                    if outer.tunable.async_global_load:
                        outer._emit_gld_async_all_chunks(v.v_off_b, v.v_sst_os, outer.lds_a_size, outer.sgpr.s_p_wei, v_flag=None)
                    else:
                        outer._emit_gld_chunk_load(v.v_gld_b, v.v_addr_b, 0, v_flag=(v.v_flag_b if outer.tunable.wmma_n_tail else None))
                return outer._get_deferred()
            def get_issues(self):
                return outer.chunk_num_dwordx4
        return functor_t()

    def shared_store_a_functor(self):
        ''' row_repeat_a > 1: row 0 uses the exact row_repeat_a==1 code path (unchanged);
        rows 1..row_repeat_a-1 (no early-issue slot of their own -- see global_load_a_functor's
        docstring) are fully deferred via _emit_sst_all_chunks_row, storing into LDS shifted
        by i*block_size*bytes_per_row (a pure destination-immediate shift -- ds_write_b128's
        offset never touches the load's source address, unlike Phase 13's async instruction,
        so no v_sst_tmp-style register trick is needed here). '''
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                with outer._deferred_context():
                    outer._emit_sst_remaining_chunks(v.v_gld_a, v.v_addr_a, v.v_sst_os, 0, v_flag=v.v_flag)
                    for i in range(1, outer.row_repeat_a):
                        row_addr = lambda idx=0, i=i: v.v_addr_a(i * 2 + idx)
                        row_flag = lambda i=i: v.v_flag(i)
                        row_off  = i * outer.tunable.block_size * outer.bytes_per_row
                        outer._emit_sst_all_chunks_row(v.v_gld_a, row_addr, v.v_sst_os, row_off, v_flag=row_flag)
                return outer._get_deferred()
            def get_issues(self):
                return outer.chunk_num_dwordx4
        return functor_t()

    def shared_store_b_functor(self):
        ''' row_repeat_b > 1: row 0 uses the exact row_repeat_b==1 code path (unchanged);
        rows 1..row_repeat_b-1 (no early-issue slot of their own -- see global_load_b_functor's
        docstring) are fully deferred via _emit_sst_all_chunks_row, storing into LDS shifted
        by i*block_size*bytes_per_row past B's own region base (lds_a_size). '''
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                with outer._deferred_context():
                    outer._emit_sst_remaining_chunks(v.v_gld_b, v.v_addr_b, v.v_sst_os, outer.lds_a_size, v_flag=(v.v_flag_b if outer.tunable.wmma_n_tail else None))
                    for i in range(1, outer.row_repeat_b):
                        row_addr = lambda idx=0, i=i: v.v_addr_b(i * 2 + idx)
                        row_off  = outer.lds_a_size + i * outer.tunable.block_size * outer.bytes_per_row
                        # rows 1..row_repeat_b-1 have no flag of their own (row_repeat_b==1 is
                        # asserted in __init__ whenever wmma_n_tail is set -- see there).
                        outer._emit_sst_all_chunks_row(v.v_gld_b, row_addr, v.v_sst_os, row_off, v_flag=None)
                return outer._get_deferred()
            def get_issues(self):
                return outer.chunk_num_dwordx4
        return functor_t()

    def global_load_chunk_a_functor(self):
        ''' Phase 15: single-chunk primitive for the interleaved main loop -- issues ONE
        chunk's global load (row 0 only; main_loop_interleave is asserted mutually exclusive
        with row_repeat_a>1 in __init__). Reuses the exact same helper (_emit_gld_chunk_load)
        and v_flag masking as global_load_a_functor's chunk-0 call, just parameterized by
        chunk_idx instead of hardcoded to 0. '''
        outer = self
        class functor_t:
            def __call__(self, chunk_idx):
                v = outer.vgpr
                with outer._deferred_context():
                    outer._emit_gld_chunk_load(v.v_gld_a, v.v_addr_a, chunk_idx, v_flag=v.v_flag)
                return outer._get_deferred()
        return functor_t()

    def global_load_chunk_b_functor(self):
        ''' Phase 15: see global_load_chunk_a_functor's docstring -- B is untransposed too,
        same treatment (no masking needed for B, mirrors global_load_b_functor's v_flag=None). '''
        outer = self
        class functor_t:
            def __call__(self, chunk_idx):
                v = outer.vgpr
                with outer._deferred_context():
                    outer._emit_gld_chunk_load(v.v_gld_b, v.v_addr_b, chunk_idx, v_flag=(v.v_flag_b if outer.tunable.wmma_n_tail else None))
                return outer._get_deferred()
        return functor_t()

    def shared_store_chunk_a_functor(self):
        ''' Phase 15: single-chunk primitive for the interleaved main loop -- stores ONE
        already-loaded-and-waited chunk (reuses _emit_sst_chunk, the same helper
        _emit_sst_remaining_chunks calls internally). '''
        outer = self
        class functor_t:
            def __call__(self, chunk_idx):
                v = outer.vgpr
                with outer._deferred_context():
                    outer._emit_sst_chunk(v.v_gld_a, v.v_sst_os, 0, chunk_idx)
                return outer._get_deferred()
        return functor_t()

    def shared_store_chunk_b_functor(self):
        outer = self
        class functor_t:
            def __call__(self, chunk_idx):
                v = outer.vgpr
                with outer._deferred_context():
                    outer._emit_sst_chunk(v.v_gld_b, v.v_sst_os, outer.lds_a_size, chunk_idx)
                return outer._get_deferred()
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
        outer = self
        num_v_b = outer.wmma_mapping.ctrl.inst_wmma.num_v_b
        num_v_b_total = outer.tunable.wmma_repeat_n * num_v_b   # Phase 22: one local_prefetch_num slot's worth
        step_bytes = outer.tunable.wmma_tile_n * outer.bytes_per_row   # was hardcoded 1024
        class functor_t:
            def __call__(self, v_dst, v_os, extra_off, slot=0):
                v = outer.vgpr
                slot_off = slot * num_v_b_total
                with outer._deferred_context():
                    for i_rn in range(outer.tunable.wmma_repeat_n):
                        base = outer.lds_a_size + i_rn * step_bytes + extra_off  # B region starts after A's region
                        outer._emit_ds_read_chunked(lambda k, i_rn=i_rn: v.v_b(slot_off+i_rn*num_v_b+k), v.v_sld_b_os, base, num_v_b)
                return outer._get_deferred()
        return functor_t()

    def move_slice_window_a_functor(self):
        ''' row_repeat_a > 1: every row's v_addr_a pair is advanced independently by the
        same per-K-substep stride (no early/deferred asymmetry here -- this is pure address
        bookkeeping, needed for every row every K-substep regardless of row 0 vs "extra"
        rows). row_repeat_a==1 for every existing config: the loop runs once with i=0,
        producing v_addr_a(0)/v_addr_a(1) == v_addr_a()/v_addr_a(1), byte-identical. '''
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                s = outer.sgpr
                with outer._deferred_context():
                    if outer.tunable.tdm_global_load:
                        # Phase 28: advance the TDM descriptor's global_addr (group0 s2/s3,
                        # a genuine 64-bit scalar address, unlike async's plain VGPR offset)
                        # by one K-chunk's worth of bytes for the next tile's load. s3's top
                        # 2 bits hold the constant "type=2" field (set once in the prologue,
                        # never re-touched) -- safe to addc directly into s3 since global_addr
                        # is only 57 bits (25 meaningful bits in s3), nowhere near the type
                        # field at bits[31:30] for any real address.
                        outer._emit(f"s_add_u32 s[{s.s_tdm_g0(2)}], s[{s.s_tdm_g0(2)}], {outer.bytes_per_row}")
                        outer._emit(f"s_addc_u32 s[{s.s_tdm_g0(3)}], s[{s.s_tdm_g0(3)}], 0")
                    elif outer.tunable.async_global_load:
                        # Phase 13: v_off_a is a plain 32-bit byte OFFSET (no base pointer
                        # folded in), so advancing it is a single add, no carry chain needed.
                        outer._emit(f"v_add_u32 v[{v.v_off_a()}], {outer.bytes_per_row}, v[{v.v_off_a()}]")
                    else:
                        for i in range(outer.row_repeat_a):
                            outer._emit(f"v_add_co_u32 v[{v.v_addr_a(i*2)}], vcc_lo, {outer.bytes_per_row}, v[{v.v_addr_a(i*2)}]")
                            outer._emit(f"v_add_co_ci_u32 v[{v.v_addr_a(i*2+1)}], vcc_lo, 0, v[{v.v_addr_a(i*2+1)}], vcc_lo")
                return outer._get_deferred()
        return functor_t()

    def move_slice_window_b_functor(self):
        ''' row_repeat_b > 1: every row's v_addr_b pair advances independently by the same
        per-K-substep stride -- see move_slice_window_a_functor's docstring for the identical
        reasoning on the A side. row_repeat_b==1 for every existing config: the loop runs
        once with i=0, byte-identical. '''
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                with outer._deferred_context():
                    if outer.tunable.async_global_load:
                        outer._emit(f"v_add_u32 v[{v.v_off_b()}], {outer.bytes_per_row}, v[{v.v_off_b()}]")
                    else:
                        for i in range(outer.row_repeat_b):
                            outer._emit(f"v_add_co_u32 v[{v.v_addr_b(i*2)}], vcc_lo, {outer.bytes_per_row}, v[{v.v_addr_b(i*2)}]")
                            outer._emit(f"v_add_co_ci_u32 v[{v.v_addr_b(i*2+1)}], vcc_lo, 0, v[{v.v_addr_b(i*2+1)}], vcc_lo")
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
        ctrl.async_global_to_lds_a = self.tunable.async_global_load
        ctrl.async_global_to_lds_b = self.tunable.async_global_load
        ctrl.tdm_global_to_lds_a = self.tunable.tdm_global_load
        ctrl.tdm_global_to_lds_b = False   # Phase 28 pilot: A operand only
        ctrl.interleave = self.tunable.main_loop_interleave
        ctrl.local_prefetch_num = self.tunable.local_prefetch_num
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
        if self.tunable.main_loop_interleave:
            ctrl.global_load_chunk_a_functor  = self.global_load_chunk_a_functor()
            ctrl.global_load_chunk_b_functor  = self.global_load_chunk_b_functor()
            ctrl.shared_store_chunk_a_functor = self.shared_store_chunk_a_functor()
            ctrl.shared_store_chunk_b_functor = self.shared_store_chunk_b_functor()
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
        self._emit(self.coalescing_store(v.v_c.label, v.v_gemm_im(), v.v_gemm_in(), s.s_p_out.label, s.s_out_k_total.label, v.v_addr_out(), v.v_addr_out(1), s.s_tmp(), v.v_tid(), v.v_c(), s.s_block_m_off(), s.s_block_n_off(),
                    s.s_gemm_m.label if self.tunable.wmma_m_tail else None, v.v_m_tail_row() if self.tunable.wmma_m_tail else None,
                    s.s_gemm_n.label if self.tunable.wmma_n_tail else None, v.v_n_tail_col() if self.tunable.wmma_n_tail else None))
        self._emit(f"s_wait_storecnt 0x0")

    def emit_kernel_body(self):
        self.emit_kernel_prologue()
        self.emit_kernel_tap_loop()
        self.emit_kernel_epilogue()
