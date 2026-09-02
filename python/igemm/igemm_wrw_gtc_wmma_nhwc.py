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

        # ---- row_stride redesign: supports gemm_k_per_block > gemm_m_per_block (see
        # docs/gfx1250_wrw_addressing_redesign.md). The original addressing
        # (num_col_groups = gemm_m_per_block // gemm_k_per_block, below in
        # emit_kernel_prologue) requires gemm_m_per_block >= gemm_k_per_block with a
        # power-of-2 quotient -- utility_log2(0) blows up otherwise. When
        # gemm_k_per_block > gemm_m_per_block, each thread instead owns row_stride FULL
        # GEMM_K rows (each gemm_m_per_block elements wide) rather than one row's
        # column-fragment; row_stride = gemm_k_per_block // gemm_m_per_block.
        #
        # B's addressing below already silently reuses A's num_col_groups/col_group_bits
        # (derived only from gemm_m_per_block) for its own gemm_n_per_block-wide tile --
        # only correct when gemm_n_per_block == gemm_m_per_block. That was an undocumented
        # assumption before; asserted explicitly now (doesn't change behavior for any
        # existing, already-passing config).
        assert tunable.gemm_n_per_block == tunable.gemm_m_per_block, \
            f"wrw WMMA's B (input) addressing reuses A's row/col-group tiling scheme " \
            f"(derived from gemm_m_per_block) -- requires gemm_n_per_block({tunable.gemm_n_per_block}) " \
            f"== gemm_m_per_block({tunable.gemm_m_per_block})"
        if tunable.gemm_k_per_block > tunable.gemm_m_per_block:
            assert tunable.gemm_k_per_block % tunable.gemm_m_per_block == 0, \
                f"gemm_k_per_block({tunable.gemm_k_per_block}) must be a multiple of " \
                f"gemm_m_per_block({tunable.gemm_m_per_block}) when gemm_k_per_block > gemm_m_per_block " \
                f"(row_stride addressing needs each thread's owned rows to tile the block exactly)"
            # row_stride addressing needs one thread per M-column of a K-row (see
            # emit_kernel_prologue) -- true of every existing config (block_size is
            # already derived from wave geometry that happens to equal gemm_m_per_block
            # for these square tiles), asserted explicitly here since it's now load-bearing.
            assert tunable.block_size == tunable.gemm_m_per_block, \
                f"row_stride addressing requires block_size({tunable.block_size}) == " \
                f"gemm_m_per_block({tunable.gemm_m_per_block})"
            # split-K/streamk/TDM combined with row_stride are still out of scope (see
            # docs/gfx1250_wrw_addressing_redesign.md) -- not wired through the new
            # per-row addressing below. M/N/K-tail ARE supported (see the row_stride>1
            # branches in emit_kernel_prologue, _emit_a_kflag, and the v_flag_col
            # parameter threaded through _emit_gld_chunk_load).
            assert not (tunable.gemm_k_global_split or tunable.wrw_streamk), \
                "row_stride addressing does not yet support split-K/streamk -- deferred follow-up"
            assert not tunable.tdm_global_load, \
                "row_stride addressing does not yet support tdm_global_load -- deferred follow-up"
            self.row_stride = tunable.gemm_k_per_block // tunable.gemm_m_per_block
            self.chunks_per_row = tunable.gemm_m_per_block // ctrl_wmma_mapping.inst_wmma.k
        else:
            self.row_stride = 1
            self.chunks_per_row = None

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
        # Phase 35: wire wrw's M/N-tail into the epilogue (fwd/bwd already do this).
        ctrl_coalescing_store_wmma.wmma_m_tail = tunable.wmma_m_tail
        ctrl_coalescing_store_wmma.wmma_n_tail = tunable.wmma_n_tail
        ctrl_coalescing_store_wmma.wrw_reduction_kernel = tunable.wrw_reduction_kernel
        ctrl_coalescing_store_wmma.direct_store          = tunable.direct_store
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
        # Phase 58 (wrw_streamk): a dedicated 4-byte LDS slot for the persistent loop's
        # tile-claim broadcast (lane 0 writes the atomic result, s_barrier, every lane
        # reads it back) -- placed right after the A/B staging LDS so it never collides
        # with it. gemm_k_global_split's own epilogue uses zero LDS (see
        # get_kernel_code's epilogue_lds_bytes), so this is the only extra LDS this mode needs.
        self.streamk_lds_off = self.lds_single_size * self.lds_buffer_num

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

        # Phase 45 (TDM global load, wrw): both A and B are "128-wide tiles" sharing the
        # same num_col_groups formula (GEMM_K is the row axis for BOTH operands here,
        # unlike bwd where only B needed axis-swapping) -- base case only (no M/N-tail
        # combined with TDM yet, mirroring fwd's original Phase 28 scope before Phase 37
        # extended it).
        assert not (tunable.tdm_global_load and (tunable.wmma_m_tail or tunable.wmma_n_tail or tunable.wmma_k_tail)), \
            "tdm_global_load is not yet combined with wmma_m_tail/n_tail/k_tail for wrw -- TDM's own hardware OOB replaces wmma_k_tail; M/N-tail-via-TDM is a separate, not-yet-attempted extension"
        assert not (tunable.tdm_global_load and tunable.local_prefetch_num > 1), \
            "tdm_global_load is not yet supported together with local_prefetch_num > 1"
        assert not (tunable.tdm_global_load and tunable.async_global_load), \
            "tdm_global_load and async_global_load are mutually exclusive -- two different load mechanisms for the same operand"
        self._tdm_label_counter = 0
        # Phase 15 (main-loop interleaving, wrw port): A (grad_output, TRANSPOSED)
        # interleaves -- its chunk loads occupy v_gld_a in-flight during compute, and
        # shared_load_a is redirected to v_scratch (allocated above) instead of v_gld_a for
        # its read-and-pack scratch. B (input, TRANSPOSED) does NOT interleave: its
        # shared_load_b reuses v_gld_b as scratch, and interleaving would issue chunk N+1's
        # load into v_gld_b between substep N's shared_load (scratch clobber) and chunk N's
        # store. interleave_b stays False. Requires lds_double_buffer=1 (same cross-wave LDS
        # race as fwd, confirmed on hardware) and is mutually exclusive with
        # async_global_load/tdm_global_load and row_stride>1 (untested combinations).
        assert not (tunable.main_loop_interleave and tunable.async_global_load), \
            "main_loop_interleave is not supported together with async_global_load"
        assert not (tunable.main_loop_interleave and tunable.tdm_global_load), \
            "main_loop_interleave is not supported together with tdm_global_load"
        assert not (tunable.main_loop_interleave and self.row_stride > 1), \
            "main_loop_interleave is not yet supported together with row_stride > 1"
        assert not (tunable.main_loop_interleave and not tunable.lds_double_buffer), \
            "main_loop_interleave requires lds_double_buffer=1 (single-buffered interleaving races across waves, confirmed on hardware)"
        assert not (tunable.main_loop_interleave and (tunable.gemm_k_global_split or tunable.wrw_streamk)), \
            "gemm_k_global_split/wrw_streamk is not yet combined with main_loop_interleave for wrw -- not audited together"

        # Phase 61 (32-bit SADDR global loads, wrw port): same discipline as fwd/bwd --
        # mutually exclusive with async/tdm, main_loop_interleave, gemm_k_global_split/
        # wrw_streamk, and row_stride>1 (untested combinations).
        assert not (tunable.saddr_global_load and tunable.async_global_load), \
            "saddr_global_load and async_global_load are mutually exclusive -- both are alternatives to the default 64-bit VADDR-pair path"
        assert not (tunable.saddr_global_load and tunable.tdm_global_load), \
            "saddr_global_load and tdm_global_load are mutually exclusive -- both are alternatives to the default 64-bit VADDR-pair path"
        assert not (tunable.saddr_global_load and tunable.main_loop_interleave), \
            "saddr_global_load is not yet combined with main_loop_interleave for wrw -- not audited together"
        assert not (tunable.saddr_global_load and (tunable.gemm_k_global_split or tunable.wrw_streamk)), \
            "saddr_global_load is not yet combined with gemm_k_global_split/wrw_streamk for wrw -- not audited together"
        assert not (tunable.saddr_global_load and self.row_stride > 1), \
            "saddr_global_load is not yet supported together with row_stride > 1 (untested combination)"
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
            # Phase 60 (Magic Division): host-precomputed magic multipliers + packed shifts
            # for the ho_wo/wo divisors in _emit_b_gather_one_row. Loaded from kernargs
            # (computed driver-side via magic_div_u32_gen), replacing the ~24-instruction
            # emulated macro_int_div_rem_vs_gfx1250_t with 5-instruction magic multiply.
            # s_magic_ho_wo is 4-aligned (sseq(1, 4)) so it can be loaded via s_load_dwordx4
            # alongside s_magic_wo -- mirrors fwd's identical layout.
            self.s_magic_ho_wo  = sym_t('s_magic_ho_wo'  , sseq(1, 4))
            self.s_magic_wo     = sym_t('s_magic_wo'     , sseq(1))
            self.s_shift_pack   = sym_t('s_shift_pack'   , sseq(1))
            self.s_shift_ho_wo  = sym_t('s_shift_ho_wo'  , sseq(1))
            self.s_shift_wo     = sym_t('s_shift_wo'     , sseq(1))
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
            if outer.tunable.wmma_k_tail and outer.tunable.gemm_k_global_split:
                # Phase 35: only meaningful (and only loaded) when K-tail must compose with
                # split-K -- a plain (non-split) wmma_k_tail build needs neither kernarg,
                # since s_knum is already exactly s_gemm_k (the true, unpadded value) in that
                # case. gemm_k_tail = the remainder R (gemm_k - num_k_blocks*gemm_k_per_block,
                # computed driver-side); gemm_k_num_splits = the launched grid.z, so "am I the
                # last shard" is a simple s_bz == s_gemm_k_num_splits-1 compare.
                self.s_gemm_k_tail       = sym_t('s_gemm_k_tail'       , sseq(1))
                self.s_gemm_k_num_splits = sym_t('s_gemm_k_num_splits' , sseq(1))
            if outer.tunable.wrw_streamk:
                # Phase 58: gemm_k_num_splits here means "total shard count" (reused,
                # mutually exclusive with the wmma_k_tail branch above -- see igemm_base.py's
                # assert). p_streamk_counter/streamk_max_iters/streamk_grid_y are the new
                # kernargs; s_streamk_tile_idx is the currently-claimed shard index (replaces
                # s_bz's role for address derivation); s_streamk_iter is the persistent loop's
                # own bounded counter.
                self.s_gemm_k_num_splits   = sym_t('s_gemm_k_num_splits'   , sseq(1))
                self.s_streamk_counter_ptr = sym_t('s_streamk_counter_ptr' , sseq(2, 2))
                self.s_streamk_max_iters   = sym_t('s_streamk_max_iters'   , sseq(1))
                self.s_streamk_grid_y      = sym_t('s_streamk_grid_y'      , sseq(1))
                self.s_streamk_tile_idx    = sym_t('s_streamk_tile_idx'    , sseq(1))
                self.s_streamk_iter        = sym_t('s_streamk_iter'        , sseq(1))
                self.s_streamk_addr        = sym_t('s_streamk_addr'        , sseq(2, 2))
            if outer.tunable.tdm_global_load:
                # Phase 45: TDM descriptors for A (grad_output) and B (input) -- both
                # operands are "128-wide tiles" here (GEMM_K is the row axis for BOTH),
                # see _emit_tdm_descriptor_setup_a/b.
                self.s_tdm_g0  = sym_t('s_tdm_g0'      , sseq(4, 4))
                self.s_tdm_g1  = sym_t('s_tdm_g1'      , sseq(8, 4))
                self.s_tdm_g0_b = sym_t('s_tdm_g0_b'   , sseq(4, 4))
                self.s_tdm_g1_b = sym_t('s_tdm_g1_b'   , sseq(8, 4))
                self.s_wave_id = sym_t('s_wave_id'     , sseq(1))
                # Remaining valid K (this shard's own spatial reduction range) --
                # initialized from s_knum (already the correct per-shard length,
                # including split-K, since tdm_global_load asserts not wmma_k_tail so
                # s_knum needs no separate tail-extension logic here).
                self.s_tdm_k_remain = sym_t('s_tdm_k_remain', sseq(1))
                # B's per-K-block global stride (mirrors s_a_k_stride, using b_n_total
                # instead of a_m_total) -- needed for TDM's per-iteration global_addr
                # advance (move_slice_window_b), since B's per-K-block byte stride isn't
                # otherwise computed anywhere (the non-TDM path re-derives B's address
                # from scratch every iteration via the stride/pad gather instead).
                self.s_b_k_stride = sym_t('s_b_k_stride', sseq(1))
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
            # Phase 15 (main-loop interleaving, wrw port): when main_loop_interleave is
            # active, A's chunk loads occupy v_gld_a in-flight during compute, so
            # shared_load_a can no longer reuse v_gld_a as scratch for its transposed
            # read-and-pack. v_scratch provides an alternative scratch (sized to
            # elem_per_dword=4, the int8 worst case). Only allocated when interleave is on.
            if outer.tunable.main_loop_interleave:
                self.v_scratch     = sym_t('v_scratch'     , vseq(4))
            self.v_tid         = sym_t('v_tid'         , vseq(1))
            # 64-bit VADDR pairs must be even-aligned on gfx1250 (verified with llvm-mc).
            # row_stride redesign: when row_stride>1 a thread owns row_stride separate
            # K-rows, each needing its own persistent address -- sized 2*row_stride (row r's
            # pair is registers 2*r/2*r+1); row_stride==1 (every existing config) keeps this
            # byte-identical to a single pair.
            if outer.tunable.saddr_global_load:
                # Phase 61: 32-bit byte offsets (SADDR carries s_p_in/s_p_wei separately).
                # row_stride==1 asserted when saddr is on.
                self.v_off_a      = sym_t('v_off_a'      , vseq(1))
                self.v_off_b      = sym_t('v_off_b'      , vseq(1))
                self.v_off_a_base = sym_t('v_off_a_base' , vseq(1))
            else:
                self.v_addr_a      = sym_t('v_addr_a'      , vseq(2 * outer.row_stride, 2))    # persistent global A address(es) (64-bit each)
                self.v_addr_b      = sym_t('v_addr_b'      , vseq(2 * outer.row_stride, 2))
                # Phase 5f: A's tap-independent base address (row_local/block_m_off/col_start only
                # -- no Y,X dependence), reset into v_addr_a fresh at the start of every tap since
                # move_slice_window_a incrementally bumps v_addr_a across a tap's own K-loop.
                self.v_addr_a_base = sym_t('v_addr_a_base' , vseq(2 * outer.row_stride, 2))
            self.v_addr_out    = sym_t('v_addr_out'    , vseq(2))    # scratch used by coalescing_store_wmma (ping-pong pair)
            if outer.tunable.wmma_m_tail:
                # Phase 35: extra scratch for coalescing_store_wmma's per-element absolute-row
                # EXEC-mask guard (both the non-atomic and atomic epilogue branches) -- only
                # allocated when wmma_m_tail is set, mirrors fwd's identically-named register.
                self.v_m_tail_row = sym_t('v_m_tail_row' , vseq(1))
            if outer.tunable.wmma_n_tail:
                # Phase 35: extra scratch for coalescing_store_wmma's column-in-range guard --
                # only allocated when wmma_n_tail is set, mirrors fwd's v_n_tail_col.
                self.v_n_tail_col = sym_t('v_n_tail_col' , vseq(1))
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
            # row_stride redesign: one flag per owned row (B's gather is redone per row);
            # row_stride==1 keeps this a single register, byte-identical to before.
            self.v_flag        = sym_t('v_flag'        , vseq(outer.row_stride))
            self.v_gtc_tmp     = sym_t('v_gtc_tmp'     , vseq(5))
            if outer.tunable.wmma_m_tail:
                # Phase 35: A's (grad_output) absolute-GEMM_M-index-in-range flag. Unlike B's
                # v_flag (recomputed every iteration, GEMM_K being spatial), A's GEMM_M
                # position is a true per-lane kernel-lifetime constant (move_slice_window_a
                # only advances the K-axis stride) -- computed once in emit_kernel_prologue.
                # row_stride redesign: every thread loads the SAME full-width row, so M-tail
                # validity depends only on which sub_chunk (inst_wmma.k-wide slice of the
                # row) is being loaded, not on tid/row_offset -- one broadcast (lane-uniform)
                # flag per sub_chunk instead of one per thread. row_stride==1 keeps this a
                # single register, byte-identical to before.
                self.v_flag_a_mtail = sym_t('v_flag_a_mtail', vseq(outer.chunks_per_row if outer.row_stride > 1 else 1))
            if outer.tunable.wmma_n_tail:
                # Phase 35: B's (input) absolute-GEMM_N-index-in-range flag -- also a
                # kernel-lifetime constant (B's column position never changes), ANDed into
                # B's own per-iteration v_flag inside _emit_b_gather (not a replacement).
                # row_stride redesign: same sub_chunk-indexed broadcast-flag reasoning as
                # v_flag_a_mtail above.
                self.v_flag_b_ntail = sym_t('v_flag_b_ntail', vseq(outer.chunks_per_row if outer.row_stride > 1 else 1))
            if outer.tunable.wmma_k_tail:
                # Phase 35: A's per-iteration GEMM_K-in-range flag -- unlike M-tail's, this
                # MUST be recomputed every iteration (mirrors B's own v_flag). If wmma_m_tail
                # is also set, v_flag_a_mtail is ANDed into this each time it's recomputed.
                # row_stride redesign: one flag per owned row (mirrors v_flag's sizing
                # above); row_stride==1 keeps this a single register, byte-identical.
                self.v_flag_a_ktail = sym_t('v_flag_a_ktail', vseq(outer.row_stride))
                # Phase 35: B's per-iteration GEMM_K-in-range flag, computed in _emit_b_gather.
                # MUST be a dedicated register, not `v_tmp` scratch -- the div/rem macro
                # (.v_u32_div_rem_vs_gfx1250) clobbers ALL FOUR v_tmp registers internally as
                # part of its own division algorithm, which silently destroyed this flag when
                # it was (incorrectly) computed into v_tmp(1) before the div/rem calls that
                # follow it in _emit_b_gather -- a real bug found via multi-tap hardware
                # testing (single-tap shapes happened to not expose it, depending on what
                # garbage the division scratch left behind).
                self.v_flag_b_ktail = sym_t('v_flag_b_ktail', vseq(1))
            if outer.tunable.atomic_pack_bf16:
                # Phase 34: packed-bf16 atomic epilogue scratch -- v_pk_partner (cross-lane-
                # exchanged value, per-iteration scratch), v_pk_packed (packed bf16x2 result,
                # per-iteration scratch). Only allocated when atomic_pack_bf16 is set (every
                # existing config byte-identical). No dedicated partner-index register needed
                # since the V_PERMLANE_XOR_B32 swap (docs/gfx1250_optimization_backlog.md) --
                # it takes the XOR mask as an immediate, unlike ds_bpermute_b32's precomputed
                # byte-index operand (formerly v_pk_idx, removed).
                self.v_pk_partner = sym_t('v_pk_partner' , vseq(1))
                self.v_pk_packed  = sym_t('v_pk_packed'  , vseq(1))
            if outer.tunable.wrw_streamk:
                # Phase 58: persistent-loop tile-claim scratch. The atomic uses the SADDR
                # form (s_streamk_addr, an SGPR pair -- see emit_kernel_streamk_loop), not a
                # VADDR pair, so no dedicated VGPR address register is needed here (VGPR
                # budget for this tile shape is already tight -- an earlier VADDR-pair
                # version overflowed past register 255). v_streamk_one is a constant 1 (the
                # atomic's increment operand, also doubles as its required VOFFSET=0... no,
                # see v_streamk_zero for that); v_streamk_claim holds the atomic's returned
                # pre-increment value, then (after the LDS broadcast round-trip) every lane's
                # copy of the SAME claimed shard index.
                self.v_streamk_one   = sym_t('v_streamk_one'   , vseq(1))
                self.v_streamk_claim = sym_t('v_streamk_claim' , vseq(1))
                # Dedicated always-zero VGPR: doubles as (a) the atomic's required VOFFSET
                # (must be 0 -- the real address is entirely in SADDR) and (b)
                # ds_write_b32/ds_read_b32's per-lane address (every lane must target the
                # SAME LDS word for the broadcast to work; the actual LDS byte position is
                # the instruction's `offset:` immediate instead -- see
                # emit_kernel_streamk_loop).
                self.v_streamk_zero  = sym_t('v_streamk_zero'  , vseq(1))
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
        # Phase 35: only needed when K-tail must compose with split-K (a plain, non-split
        # wmma_k_tail build needs neither -- see kernel_sgpr_t's identical comment). Gated
        # (not always-present like gemm_k_per_wg above) so every existing config's
        # kernarg_segment_byte_size stays byte-identical.
        if self.tunable.wmma_k_tail and self.tunable.gemm_k_global_split:
            kas.append(amdgpu_kernel_arg_t('gemm_k_tail', 4, 92, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('gemm_k_num_splits', 4, 96, 'by_value', 'i32'))
        if self.tunable.wrw_streamk:
            # Phase 58: mutually exclusive with the wmma_k_tail branch above (asserted in
            # igemm_base.py), so offset 92 (gemm_k_tail's slot there) is simply unused here --
            # reuse gemm_k_num_splits at its EXISTING offset 96 (same host-side struct field,
            # same meaning it already has: "how many shards exist" -- previously only used
            # for the wmma_k_tail last-shard check, now the persistent loop's in-range test
            # too) rather than colliding with it. New fields append after it, 8-byte-aligned
            # for the pointer.
            kas.append(amdgpu_kernel_arg_t('gemm_k_num_splits', 4, 96, 'by_value', 'i32'))
            kas.append(amdgpu_kernel_arg_t('p_streamk_counter', 8, 104, 'global_buffer', 'u32', address_space='global', is_const='false'))
            kas.append(amdgpu_kernel_arg_t('streamk_max_iters', 4, 112, 'by_value', 'i32'))
        # Phase 60 (Magic Division): host-precomputed magic multipliers for the ho_wo/wo
        # divisors in _emit_b_gather_one_row's K-decomposition. Always present (even for
        # 1x1 configs where ho_wo=1, wo=1 -- magic_div_u32_gen(1) produces valid values)
        # so all variants share one karg struct on the driver side. Placed after all
        # existing fields to avoid shifting any gated field's offset.
        kas.append(amdgpu_kernel_arg_t('magic_ho_wo', 4, 120, 'by_value', 'u32'))
        kas.append(amdgpu_kernel_arg_t('magic_wo',    4, 124, 'by_value', 'u32'))
        kas.append(amdgpu_kernel_arg_t('shift_pack',  4, 128, 'by_value', 'u32'))
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
        # PERF-007 (2026-09-02): see igemm_fwd_gtc_wmma_nhwc.py's identical comment --
        # direct_store (Phase 59) skips the LDS-reshuffle epilogue entirely, so it needs
        # zero epilogue LDS just like gemm_k_global_split's atomic path.
        epilogue_lds_bytes = 0 if (self.tunable.gemm_k_global_split or self.tunable.direct_store) else \
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
            'workgroup_group_segment_byte_size':   max(self.lds_single_size * self.lds_buffer_num, epilogue_lds_bytes) + (4 if self.tunable.wrw_streamk else 0),
            'kernarg_segment_byte_size'         :   132,  # Phase 60: always full struct size (C++ struct is always 132 bytes with magic fields)
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
        # row_stride redesign: bytes_per_row (=gemm_k_per_block*data_byte) is no longer
        # guaranteed a power of 2 (e.g. K=96 with a 32-wide tile) -- v_mul_lo_u32 by the
        # compile-time constant works for both cases, unlike the v_lshlrev_b32/utility_log2
        # this replaced (which asserted on K=96).
        self._emit(f"v_mul_lo_u32 v[{v.v_sst_os()}], {self.bytes_per_row}, v[{v.v_tid()}]   ; tid*{self.bytes_per_row} bytes")
        self._emit_empty_line()

        # ---- shared-memory load offset for A (TRANSPOSED -- grad_output is [K][M] in LDS) ----
        # row_pitch_bytes = gemm_m_per_block * databyte (128 * databyte); A's region starts at
        # byte 0 within the shared LDS tile (no lds_a_size-style base needed, unlike B below).
        # Phase 64: ds_load_tr_b uses a different per-lane address formula (hardware
        # transpose-load) -- see wmma_mapping.py's
        # get_gemm_index_for_src_matrix_transposed_ds_tr16 docstring.
        if self.tunable.ds_load_tr_b:
            self._emit(self.wmma_mapping.get_gemm_index_for_src_matrix_transposed_ds_tr16(v.v_sld_a_os(), v.v_tid(), v.v_tmp(),
                    self.tunable.gemm_m_per_block * self.data_byte, self.data_byte, 'm'))
        else:
            self._emit(self.wmma_mapping.get_gemm_index_for_src_matrix_transposed(v.v_sld_a_os(), v.v_tid(), v.v_tmp(),
                    self.tunable.gemm_m_per_block * self.data_byte, self.data_byte, 'm'))
        self._emit_empty_line()

        # ---- shared-memory load offset for B (TRANSPOSED -- input is [K][N] in LDS) ----
        # row_pitch_bytes = gemm_n_per_block * databyte (128 * databyte)
        if self.tunable.ds_load_tr_b:
            self._emit(self.wmma_mapping.get_gemm_index_for_src_matrix_transposed_ds_tr16(v.v_sld_b_os(), v.v_tid(), v.v_tmp(),
                    self.tunable.gemm_n_per_block * self.data_byte, self.data_byte, 'n'))
        else:
            self._emit(self.wmma_mapping.get_gemm_index_for_src_matrix_transposed(v.v_sld_b_os(), v.v_tid(), v.v_tmp(),
                    self.tunable.gemm_n_per_block * self.data_byte, self.data_byte, 'n'))
        self._emit_empty_line()

    def _emit_tdm_descriptor_setup_a(self):
        '''
        Phase 45: TDM descriptor for A (grad_output). GEMM_K (n*ho*wo, spatial) is the
        ROW axis here (not the contiguous axis, unlike fwd's A/bwd's A) -- each row is a
        single spatial reduction position, GEMM_M (K_out) is the contiguous axis within
        that row (grad_output is NHWC, K_out innermost). This is the SAME axis-swap
        bwd's B needed (Phase 42), just applied to wrw's A: tensor_dim0=gemm_m (K_out,
        contiguous), tensor_dim1=gemm_k (spatial, row axis), tensor_dim0_stride=a_m_total
        (this project's already-established, non-TDM-validated per-pixel row stride --
        see s_a_k_stride's own derivation, which uses the identical a_m_total base).

        Only correct for the 1x1/unit-stride/no-pad/no-dilation case (asserted via
        nxe==0 at the tunable level) -- for y=x=1 with stride=1/pad=0/dilation=1, B's
        per-tap gather (`_emit_b_gather`, decomposing the absolute K-index into
        (n,ho,wo) then remapping through the stride/pad formula to (hi,wi)) collapses to
        an IDENTITY (hi=ho, wi=wo, and critically `hi_wi == ho_wo` too since input and
        output spatial extents are equal with no padding/stride reduction) -- so the
        recomposed `row_idx` used for B's address is provably equal to the same flat K
        index that was decomposed, meaning B's address is ALSO a simple linear function
        of the K index for this restricted case, exactly like every other TDM operand
        this project has ported. A's own tap-independence (grad_output has no Y,X extent
        at all) makes this even more direct for A.
        '''
        s = self.sgpr
        data_size_code = utility_log2(self.data_byte)
        tile_dim0 = self.tunable.gemm_m_per_block
        tile_dim1 = self.tunable.gemm_k_per_block
        assert tile_dim0 < 65536 and tile_dim1 < 65536, "TDM tile_dim0/1 are 16-bit fields"

        self._emit(f"; --- Phase 45: TDM descriptor for A operand (grad_output) ---")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g0(0)}], 1   ; group0: pred=1 (valid tensor)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g0(1)}], 0   ; group0: lds_addr (A's LDS region starts at byte 0)")
        self._emit(f"; group0: global_addr = p_in + (block_m_off + gemm_k_wg_off*a_m_total) * data_byte")
        if self.tunable.gemm_k_global_split:
            self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_gemm_k_wg_off()}], s[{s.s_a_m_total()}]")
            self._emit(f"s_add_u32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], s[{s.s_block_m_off()}]")
        else:
            self._emit(f"s_mov_b32 s[{s.s_tmp(0)}], s[{s.s_block_m_off()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], {data_size_code}")
        self._emit(f"s_add_u32 s[{s.s_tdm_g0(2)}], s[{s.s_p_in()}], s[{s.s_tmp(0)}]")
        self._emit(f"s_addc_u32 s[{s.s_tmp(1)}], s[{s.s_p_in(1)}], 0")
        self._emit(f"s_or_b32 s[{s.s_tdm_g0(3)}], s[{s.s_tmp(1)}], 0x80000000   ; | type=2 (image) in bits[31:30]")
        self._emit_empty_line()

        self._emit(f"; group1: data_size={data_size_code}, workgroup_mask=0 (not clustered)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1(0)}], {data_size_code << 16}")
        self._emit(f"s_lshl_b32 s[{s.s_tdm_g1(1)}], s[{s.s_gemm_m()}], 16   ; tensor_dim0 (gemm_m) lo16 -> [31:16]")
        self._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_gemm_m()}], 16   ; tensor_dim0 hi16")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(1)}], s[{s.s_tdm_k_remain()}], 16   ; tensor_dim1 (this shard's remaining K) lo16")
        self._emit(f"s_or_b32 s[{s.s_tdm_g1(2)}], s[{s.s_tmp(0)}], s[{s.s_tmp(1)}]")
        self._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_tdm_k_remain()}], 16   ; tensor_dim1 hi16")
        self._emit(f"s_or_b32 s[{s.s_tdm_g1(3)}], s[{s.s_tmp(0)}], {tile_dim0 << 16}   ; | tile_dim0 (compile-time)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1(4)}], {tile_dim1}   ; tile_dim1 (compile-time), tile_dim2 unused")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1(5)}], s[{s.s_a_m_total()}]   ; tensor_dim0_stride lo32 (elements)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1(6)}], 0   ; tensor_dim0_stride hi16 (assume < 2^32 elements)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1(7)}], 0   ; tensor_dim1_stride unused (2D tensor)")
        self._emit_empty_line()

    def _emit_tdm_descriptor_setup_b(self):
        '''
        Phase 45: TDM descriptor for B (input). Structurally identical to A's -- input is
        also NHWC (C_in innermost, contiguous), GEMM_K (spatial) is the row axis, GEMM_N
        (C_in) is the contiguous axis. tensor_dim0=gemm_n, tensor_dim1=gemm_k,
        tensor_dim0_stride=b_n_total. See _emit_tdm_descriptor_setup_a's docstring for
        why the 1x1/unit-stride restriction makes B's normally-gathered address a simple
        linear function of the K index here too.
        '''
        s = self.sgpr
        data_size_code = utility_log2(self.data_byte)
        tile_dim0 = self.tunable.gemm_n_per_block
        tile_dim1 = self.tunable.gemm_k_per_block
        assert tile_dim0 < 65536 and tile_dim1 < 65536, "TDM tile_dim0/1 are 16-bit fields"

        self._emit(f"; --- Phase 45: TDM descriptor for B operand (input) ---")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g0_b(0)}], 1   ; group0: pred=1 (valid tensor)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g0_b(1)}], {self.lds_a_size}   ; group0: lds_addr (B's LDS region starts after A's)")
        self._emit(f"; group0: global_addr = p_wei + (block_n_off + gemm_k_wg_off*b_n_total) * data_byte")
        if self.tunable.gemm_k_global_split:
            self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_gemm_k_wg_off()}], s[{s.s_b_n_total()}]")
            self._emit(f"s_add_u32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], s[{s.s_block_n_off()}]")
        else:
            self._emit(f"s_mov_b32 s[{s.s_tmp(0)}], s[{s.s_block_n_off()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], {data_size_code}")
        self._emit(f"s_add_u32 s[{s.s_tdm_g0_b(2)}], s[{s.s_p_wei()}], s[{s.s_tmp(0)}]")
        self._emit(f"s_addc_u32 s[{s.s_tmp(1)}], s[{s.s_p_wei(1)}], 0")
        self._emit(f"s_or_b32 s[{s.s_tdm_g0_b(3)}], s[{s.s_tmp(1)}], 0x80000000   ; | type=2 (image) in bits[31:30]")
        self._emit_empty_line()

        self._emit(f"; group1: data_size={data_size_code}, workgroup_mask=0 (not clustered)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1_b(0)}], {data_size_code << 16}")
        self._emit(f"s_lshl_b32 s[{s.s_tdm_g1_b(1)}], s[{s.s_gemm_n()}], 16   ; tensor_dim0 (gemm_n) lo16 -> [31:16]")
        self._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_gemm_n()}], 16   ; tensor_dim0 hi16")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(1)}], s[{s.s_tdm_k_remain()}], 16   ; tensor_dim1 (this shard's remaining K) lo16")
        self._emit(f"s_or_b32 s[{s.s_tdm_g1_b(2)}], s[{s.s_tmp(0)}], s[{s.s_tmp(1)}]")
        self._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_tdm_k_remain()}], 16   ; tensor_dim1 hi16")
        self._emit(f"s_or_b32 s[{s.s_tdm_g1_b(3)}], s[{s.s_tmp(0)}], {tile_dim0 << 16}   ; | tile_dim0 (compile-time)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1_b(4)}], {tile_dim1}   ; tile_dim1 (compile-time), tile_dim2 unused")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1_b(5)}], s[{s.s_b_n_total()}]   ; tensor_dim0_stride lo32 (elements)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1_b(6)}], 0   ; tensor_dim0_stride hi16 (assume < 2^32 elements)")
        self._emit(f"s_mov_b32 s[{s.s_tdm_g1_b(7)}], 0   ; tensor_dim1_stride unused (2D tensor)")
        self._emit_empty_line()

    def _emit_wave0_only(self, body_fn):
        ''' Phase 45: direct port of igemm_fwd_gtc_wmma_nhwc_t's identical Phase 29 helper. '''
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
        if self.tunable.wmma_k_tail and self.tunable.gemm_k_global_split:
            self._emit(f"s_load_dword s[{s.s_gemm_k_tail()}], s[{s.s_ka()}:{s.s_ka(1)}], 92")
            self._emit(f"s_load_dword s[{s.s_gemm_k_num_splits()}], s[{s.s_ka()}:{s.s_ka(1)}], 96")
        if self.tunable.wrw_streamk:
            self._emit(f"s_load_dword s[{s.s_gemm_k_num_splits()}], s[{s.s_ka()}:{s.s_ka(1)}], 96   ; Phase 58: total shard count")
            self._emit(f"s_load_dwordx2 s[{s.s_streamk_counter_ptr()}:{s.s_streamk_counter_ptr(1)}], s[{s.s_ka()}:{s.s_ka(1)}], 104")
            self._emit(f"s_load_dword s[{s.s_streamk_max_iters()}], s[{s.s_ka()}:{s.s_ka(1)}], 112")
            self._emit(f"s_load_dword s[{s.s_streamk_grid_y()}], s[{s.s_ka()}:{s.s_ka(1)}], 116")
        # Phase 60 (Magic Division): load magic multipliers + packed shift from kernargs
        self._emit(f"s_load_dwordx2 s[{s.s_magic_ho_wo()}:{s.s_magic_ho_wo(1)}], s[{s.s_ka()}:{s.s_ka(1)}], 120")
        self._emit(f"s_load_dword s[{s.s_shift_pack()}], s[{s.s_ka()}:{s.s_ka(1)}], 128")
        self._emit(f"v_mov_b32 v[{v.v_tid()}], v0")
        if self.tunable.tdm_global_load:
            # Phase 45: mirrors fwd/bwd's identical Phase 29/42 computation -- this wave's
            # index within the workgroup, derived while EXEC is still fully enabled.
            self._emit(f"v_readfirstlane_b32 s[{s.s_wave_id()}], v[{v.v_tid()}]   ; Phase 29/42/45: lane 0's flat tid = this wave's base")
            self._emit(f"s_lshr_b32 s[{s.s_wave_id()}], s[{s.s_wave_id()}], 5   ; wave index within workgroup")
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
        if self.tunable.gemm_k_global_split and self.tunable.gsplit_stagger:
            # Phase 41: stagger each split-K shard's first memory burst by (bz mod 128)
            # * ~64 cycles via S_SLEEP_VAR -- as early as possible (right after bz is
            # decoded, before ANY of the group-decode/pointer-offset prologue work or
            # the first global load), so the entire rest of the prologue's wall-clock
            # position is shifted too, not just the loop body. Pure timing -- doesn't
            # touch s_bz itself, any address computation, or K-tail masking.
            self._emit(f"s_and_b32 s[{s.s_tmp()}], s[{s.s_bz()}], 0x7f   ; gsplit_stagger: bz mod 128")
            self._emit(f"s_sleep_var s[{s.s_tmp()}]")
        self._emit(f"s_wait_kmcnt 0x0")
        # Phase 60 (Magic Division): unpack the per-divisor shifts from the packed shift
        # word, now that s_wait_kmcnt above has guaranteed s_shift_pack's s_load_dword
        # has actually landed. shift_pack layout: [7:0] = ho_wo shift, [15:8] = wo shift.
        self._emit(f"s_and_b32 s[{s.s_shift_ho_wo()}], s[{s.s_shift_pack()}], 0xff")
        self._emit(f"s_lshr_b32 s[{s.s_shift_wo()}], s[{s.s_shift_pack()}], 8")
        if self.tunable.wrw_streamk:
            # Phase 58: s_gemm_k_wg_off is NOT computed here -- unlike the static-shard
            # design, this workgroup's shard index isn't known until the persistent loop
            # claims one (see emit_kernel_streamk_loop). s_bz is decoded above but otherwise
            # unused in this mode (it's just "which of the small persistent pool", not a
            # shard index). What CAN be computed once here (constant for the kernel's
            # lifetime): this tile's counter address = counter_ptr + (bx*grid_y + by)*4.
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_bx()}], s[{s.s_streamk_grid_y()}]")
            self._emit(f"s_add_u32 s[{s.s_tmp()}], s[{s.s_tmp()}], s[{s.s_by()}]")
            self._emit(f"s_lshl_b32 s[{s.s_tmp()}], s[{s.s_tmp()}], 2   ; *4 bytes")
            self._emit(f"s_add_u32 s[{s.s_streamk_addr()}], s[{s.s_streamk_counter_ptr()}], s[{s.s_tmp()}]")
            self._emit(f"s_addc_u32 s[{s.s_streamk_addr(1)}], s[{s.s_streamk_counter_ptr(1)}], 0")
            self._emit(f"v_mov_b32 v[{v.v_streamk_one()}], 1")
            self._emit(f"v_mov_b32 v[{v.v_streamk_zero()}], 0")
        elif self.tunable.gemm_k_global_split:
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
            if self.tunable.wmma_k_tail:
                # Phase 35: shard bases (s_gemm_k_wg_off = bz*gemm_k_per_wg) are exact
                # multiples of gemm_k_per_block and contiguous by construction -- only the
                # LAST shard's range needs extending to cover gemm_k's true (possibly
                # non-exact-multiple) end. No overlap, no gap: every other shard's range is
                # already provably within [0, gemm_k).
                self._emit(f"s_add_u32 s[{s.s_tmp(0)}], s[{s.s_bz()}], 1")
                self._emit(f"s_cmp_eq_u32 s[{s.s_tmp(0)}], s[{s.s_gemm_k_num_splits()}]   ; am I the last shard?")
                self._emit(f"s_cselect_b32 s[{s.s_tmp(0)}], s[{s.s_gemm_k_tail()}], 0")
                self._emit(f"s_add_u32 s[{s.s_knum()}], s[{s.s_knum()}], s[{s.s_tmp(0)}]   ; wmma_k_tail: extend last shard's range by the remainder")
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
        # row_stride redesign: gemm_k_per_block is no longer guaranteed a power of 2 (e.g.
        # K=96 with a 32-wide tile) -- a plain s_mul_i32 by the compile-time constant
        # data_byte*gemm_k_per_block works for both the power-of-2 and non-power-of-2 case,
        # unlike the s_lshl_b32/utility_log2 this replaced (which asserted on K=96).
        self._emit(f"s_mul_i32 s[{s.s_a_k_stride()}], s[{s.s_a_m_total()}], {self.data_byte * self.tunable.gemm_k_per_block}   ; a_m_total * databyte * {self.tunable.gemm_k_per_block}")
        if self.tunable.tdm_global_load:
            # Phase 45: B's per-K-block global stride (mirrors s_a_k_stride's derivation,
            # using b_n_total instead of a_m_total) -- only needed for TDM's B descriptor
            # advance, since the non-TDM path never has a constant B stride (its address
            # is re-gathered from scratch every iteration instead).
            self._emit(f"s_lshl_b32 s[{s.s_b_k_stride()}], s[{s.s_b_n_total()}], {utility_log2(self.data_byte * self.tunable.gemm_k_per_block)}   ; b_n_total * databyte * {self.tunable.gemm_k_per_block}")
            self._emit(f"s_mov_b32 s[{s.s_tdm_k_remain()}], s[{s.s_knum()}]   ; Phase 45: this shard's own remaining K (s_knum already reflects split-K)")
            self._emit_tdm_descriptor_setup_a()
            self._emit_tdm_descriptor_setup_b()
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
        if self.tunable.wrw_reduction_kernel and not self.tunable.wrw_streamk:
            self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_group()}], s[{s.s_gemm_m()}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], s[{s.s_wei_row_c()}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], s[{s.s_bz()}]")
            self._emit(f"s_lshl_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], 2   ; workspace is always fp32 (4B)")
            self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp(0)}]")
            self._emit(f"s_addc_u32 s[{s.s_p_out(1)}], s[{s.s_p_out(1)}], 0")
        self._emit_empty_line()

        if self.row_stride == 1:
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
            if self.tunable.wmma_m_tail:
                # Phase 35: v_tmp(1) still holds col_start (this thread's position within the
                # GEMM_M block, untouched by the row-major flat-index accumulation above, which
                # only ever writes v_tmp()) -- block_m_off + col_start is this thread's absolute
                # GEMM_M index, independent of which K-row (row_local) it's on.
                self._emit(f"v_add_u32 v[{v.v_tmp(2)}], s[{s.s_block_m_off()}], v[{v.v_tmp(1)}]   ; wmma_m_tail: absolute GEMM_M index")
                self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_gemm_m()}], v[{v.v_tmp(2)}]")
                self._emit(f"v_cndmask_b32 v[{v.v_flag_a_mtail()}], 0, 1, vcc_lo")
            if self.tunable.gemm_k_global_split and not self.tunable.wrw_streamk:
                self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_gemm_k_wg_off()}], s[{s.s_a_m_total()}]   ; this workgroup's K-slice base, in A row units")
                self._emit(f"v_add_u32 v[{v.v_tmp()}], s[{s.s_tmp(2)}], v[{v.v_tmp()}]")
            # Phase 58 (wrw_streamk): the K-slice offset is NOT folded in here -- unlike the
            # static-shard design, it isn't known until the persistent loop claims a shard.
            # v_addr_a_base stays shard-independent; emit_kernel_streamk_loop() adds the current
            # iteration's offset directly onto v_addr_a (not v_addr_a_base) after each claim.
            self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {utility_log2(self.data_byte)}, v[{v.v_tmp()}]")
            if self.tunable.saddr_global_load:
                # Phase 61: byte OFFSET only -- s_p_in passed separately as SADDR
                self._emit(f"v_mov_b32 v[{v.v_off_a_base()}], v[{v.v_tmp()}]")
            else:
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
            if self.tunable.wmma_n_tail:
                self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_gemm_n()}], v[{v.v_tmp()}]")
                self._emit(f"v_cndmask_b32 v[{v.v_flag_b_ntail()}], 0, 1, vcc_lo   ; wmma_n_tail: persistent, ANDed into per-iteration v_flag in _emit_b_gather")
            self._emit(f"v_lshlrev_b32 v[{v.v_b_col_off()}], {utility_log2(self.data_byte)}, v[{v.v_tmp()}]")
            self._emit_empty_line()
        else:
            # ---- row_stride redesign (gemm_k_per_block > gemm_m_per_block): thread tid owns
            # row_stride FULL K-rows starting at row_local_base = tid*row_stride -- no
            # col_group at all (each owned row is the entire gemm_m_per_block/gemm_n_per_block
            # width), since a "row" here no longer fits within a single thread's old
            # gemm_k_per_block-wide column-fragment. Each owned row gets its own persistent
            # VADDR pair -- v_addr_a_base(2*row_offset)/(2*row_offset+1) -- since A's address is
            # a genuine runtime function of which physical K-row this thread owns (unlike the
            # LDS side, where all row_stride owned rows share a single compile-time-derived
            # store offset -- see _emit_sst_chunk). See docs/gfx1250_wrw_addressing_redesign.md.
            self._emit(f"; row_stride={self.row_stride}: thread owns {self.row_stride} full K-rows, starting at tid*{self.row_stride}")
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], {self.row_stride}, v[{v.v_tid()}]   ; row_local_base = tid*{self.row_stride}")
            for row_offset in range(self.row_stride):
                if row_offset == 0:
                    self._emit(f"v_mov_b32 v[{v.v_tmp(1)}], v[{v.v_tmp()}]   ; row_local for row_offset=0")
                else:
                    self._emit(f"v_add_u32 v[{v.v_tmp(1)}], {row_offset}, v[{v.v_tmp()}]   ; row_local for row_offset={row_offset}")
                self._emit(f"v_mul_lo_u32 v[{v.v_tmp(2)}], s[{s.s_a_m_total()}], v[{v.v_tmp(1)}]  ; row_local * a_m_total (K_out TOTAL, not per-group)")
                self._emit(f"v_add_u32 v[{v.v_tmp(2)}], s[{s.s_block_m_off()}], v[{v.v_tmp(2)}]")
                self._emit(f"v_lshlrev_b32 v[{v.v_tmp(2)}], {utility_log2(self.data_byte)}, v[{v.v_tmp(2)}]")
                self._emit(f"v_mov_b32 v[{v.v_addr_a_base(2 * row_offset + 1)}], s[{s.s_p_in(1)}]")
                self._emit(f"v_add_co_u32 v[{v.v_addr_a_base(2 * row_offset)}], vcc_lo, s[{s.s_p_in()}], v[{v.v_tmp(2)}]")
                self._emit(f"v_add_co_ci_u32 v[{v.v_addr_a_base(2 * row_offset + 1)}], vcc_lo, 0, v[{v.v_addr_a_base(2 * row_offset + 1)}], vcc_lo")
            self._emit_empty_line()

            if self.tunable.wmma_m_tail:
                # ---- wmma_m_tail (row_stride redesign): every thread loads the SAME
                # full-width row, so M-in-range validity depends only on sub_chunk (which
                # inst_wmma.k-wide slice of the row), identically for every lane -- one
                # broadcast flag per sub_chunk (computed once, lane-uniform) instead of one
                # per thread. Combined with A's per-row K-tail flag (independent axis) on
                # the fly by _emit_gld_chunk_load's v_flag_col -- see
                # docs/gfx1250_wrw_addressing_redesign.md. ----
                inst_wmma_k = self.wmma_mapping.ctrl.inst_wmma.k
                self._emit(f"; wmma_m_tail: broadcast per-sub_chunk M-in-range flag (same for every lane)")
                for sub_chunk in range(self.chunks_per_row):
                    self._emit(f"v_mov_b32 v[{v.v_tmp(2)}], s[{s.s_block_m_off()}]")
                    if sub_chunk:
                        self._emit(f"v_add_u32 v[{v.v_tmp(2)}], {sub_chunk * inst_wmma_k}, v[{v.v_tmp(2)}]   ; += sub_chunk*{inst_wmma_k}")
                    self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_gemm_m()}], v[{v.v_tmp(2)}]")
                    self._emit(f"v_cndmask_b32 v[{v.v_flag_a_mtail(sub_chunk)}], 0, 1, vcc_lo   ; wmma_m_tail: sub_chunk={sub_chunk}")
                self._emit_empty_line()

            # ---- persistent setup for B's per-iteration gather. v_row_local is the BASE row
            # (row_offset=0); _emit_b_gather adds each row_offset in 0..row_stride-1 itself.
            # col_start is always 0 here (a thread's row is the full gemm_n_per_block width). ----
            self._emit(f"v_mov_b32 v[{v.v_row_local()}], v[{v.v_tmp()}]   ; row_local_base = tid*{self.row_stride} (fixed for the whole kernel; per-row offset added in _emit_b_gather)")
            self._emit(f"; v_b_col_off = block_n_off * databyte (fixed for the whole kernel; col_start=0, full N-width row per thread)")
            self._emit(f"v_mov_b32 v[{v.v_tmp()}], s[{s.s_block_n_off()}]")
            self._emit(f"v_lshlrev_b32 v[{v.v_b_col_off()}], {utility_log2(self.data_byte)}, v[{v.v_tmp()}]")
            self._emit_empty_line()

            if self.tunable.wmma_n_tail:
                # ---- wmma_n_tail (row_stride redesign): same sub_chunk-indexed broadcast-
                # flag reasoning as wmma_m_tail above, for B's GEMM_N (input channel) tail. ----
                inst_wmma_k = self.wmma_mapping.ctrl.inst_wmma.k
                self._emit(f"; wmma_n_tail: broadcast per-sub_chunk N-in-range flag (same for every lane)")
                for sub_chunk in range(self.chunks_per_row):
                    self._emit(f"v_mov_b32 v[{v.v_tmp(2)}], s[{s.s_block_n_off()}]")
                    if sub_chunk:
                        self._emit(f"v_add_u32 v[{v.v_tmp(2)}], {sub_chunk * inst_wmma_k}, v[{v.v_tmp(2)}]   ; += sub_chunk*{inst_wmma_k}")
                    self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_gemm_n()}], v[{v.v_tmp(2)}]")
                    self._emit(f"v_cndmask_b32 v[{v.v_flag_b_ntail(sub_chunk)}], 0, 1, vcc_lo   ; wmma_n_tail: sub_chunk={sub_chunk}")
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
        emit_vopd_paired_zero_init(self._emit, v.v_c, self.tunable.num_vgpr_accumulate_c)
        self._emit_empty_line()

        if not self.tunable.tdm_global_load:
            # ---- reset A's address(es) to its tap-independent base (move_slice_window_a
            # bumped it across the PREVIOUS tap's K-loop). row_stride>1: one pair per owned
            # row; row_stride==1 is a single pair, byte-identical to before. ----
            if self.tunable.saddr_global_load:
                self._emit(f"v_mov_b32 v[{v.v_off_a()}], v[{v.v_off_a_base()}]   ; Phase 61: reset A byte offset to base")
            else:
                for row_offset in range(self.row_stride):
                    self._emit(f"v_mov_b32 v[{v.v_addr_a(2 * row_offset)}], v[{v.v_addr_a_base(2 * row_offset)}]")
                    self._emit(f"v_mov_b32 v[{v.v_addr_a(2 * row_offset + 1)}], v[{v.v_addr_a_base(2 * row_offset + 1)}]")
            self._emit_empty_line()

            # ---- B's initial (k_block_off=0) gather for this tap's first global load --
            # reads the CURRENT s_iy/s_ix live, so no special-casing needed here ----
            # Phase 45: skipped entirely under TDM -- for 1x1 (TDM's only supported case,
            # nxe==0), there is exactly one tap (y=x=1), and _emit_b_gather's whole
            # purpose (the 2-division (n,ho,wo) decomposition + stride/pad remap) is
            # provably a no-op identity for that case (see
            # _emit_tdm_descriptor_setup_a's docstring) -- global_load_b_functor's TDM
            # branch never references its output (v_addr_b/v_flag) at all. Skipping it
            # avoids paying for two integer-division macro calls per kernel launch for
            # nothing, mirroring bwd's identical Phase 42 optimization.
            self._emit_b_gather(None)
            if self.tunable.wmma_k_tail:
                self._emit_a_kflag(None)
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
        # unused placeholder there -- see coalescing_store_wmma.py's docstring). Phase 35:
        # wmma_m_tail/wmma_n_tail reuse those same v_tmp3/v_tmp4 slots for their own
        # row/col-in-range scratch (v_m_tail_row/v_n_tail_col) -- mutually exclusive with
        # atomic_pack_bf16 for now (asserted in igemm_base.py) to avoid a slot conflict and
        # the unreviewed partner-lane-out-of-range wrinkle a packed cross-lane exchange would
        # introduce for tail masking.
        if self.tunable.atomic_pack_bf16:
            self._emit(self.coalescing_store(v.v_c.label, v.v_gemm_im(), v.v_gemm_in(), s.s_p_out_tap.label, s.s_wei_row_c.label, v.v_addr_out(), v.v_addr_out(1), s.s_tmp(), v.v_tid(), None, s.s_block_m_off(), s.s_block_n_off(), None, v.v_pk_partner(), None, v.v_pk_packed()))
        else:
            self._emit(self.coalescing_store(v.v_c.label, v.v_gemm_im(), v.v_gemm_in(), s.s_p_out_tap.label, s.s_wei_row_c.label, v.v_addr_out(), v.v_addr_out(1), s.s_tmp(), v.v_tid(), v.v_c(),
                    s.s_block_m_off(), s.s_block_n_off(),
                    s.s_gemm_m.label if self.tunable.wmma_m_tail else None, v.v_m_tail_row() if self.tunable.wmma_m_tail else None,
                    s.s_gemm_n.label if self.tunable.wmma_n_tail else None, v.v_n_tail_col() if self.tunable.wmma_n_tail else None,
                    # Phase 66: always passed now (not just under wmma_n_tail) --
                    # direct_store's outer-loop address hoist reuses this scratch
                    # SGPR too; mutually exclusive with the LDS-reshuffle path's own
                    # use of it, so sharing the slot is safe.
                    s.s_tmp(1)))
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

        row_stride redesign: when row_stride>1, this thread owns row_stride consecutive
        absolute K-positions (v_row_local is the first, i.e. row_offset=0); the whole
        gather is repeated once per owned row into that row's own v_flag(row_offset)/
        v_addr_b(2*row_offset:2*row_offset+1) slot. row_stride==1 is a single iteration,
        byte-identical to before.
        '''
        for row_offset in range(self.row_stride):
            self._emit_b_gather_one_row(s_k_block_off, row_offset)

    def _emit_b_gather_one_row(self, s_k_block_off, row_offset):
        s = self.sgpr
        v = self.vgpr
        m_mdiv_rem_vs = macro_mdiv_u32_rem_vs_t(self.mc)
        row_note = f" + row_offset({row_offset})" if row_offset else ""
        if s_k_block_off is not None:
            self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], s[{s_k_block_off}], v[{v.v_row_local()}]   ; k_abs (within this workgroup's K-slice){row_note}")
        else:
            self._emit(f"v_mov_b32 v[{v.v_gtc_tmp(0)}], v[{v.v_row_local()}]   ; k_abs (k_block_off=0, within this workgroup's K-slice){row_note}")
        if row_offset:
            self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(0)}], {row_offset}   ; += row_offset (this thread's {row_offset}-th owned row)")
        if self.tunable.gemm_k_global_split:
            self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(0)}], s[{s.s_gemm_k_wg_off()}]   ; += this workgroup's K-slice base")
        if self.tunable.wmma_k_tail:
            # Phase 35: capture the K-in-range check NOW, while v_gtc_tmp(0) still holds the
            # raw global k_abs -- it's about to be destroyed by the div/rem decomposition
            # below. MUST use the dedicated v_flag_b_ktail register, NOT v_tmp scratch --
            # the magic division macro below clobbers v_tmp(0) internally (Phase 60: only
            # 1 VGPR now, vs 4 in the old emulated macro), which would silently destroy a
            # v_tmp-based flag before it's consumed by the v_flag AND further down
            # (original bug found via multi-tap hardware testing).
            self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_gemm_k()}], v[{v.v_gtc_tmp(0)}]")
            self._emit(f"v_cndmask_b32 v[{v.v_flag_b_ktail()}], 0, 1, vcc_lo   ; wmma_k_tail: k_abs < real gemm_k")
        # Phase 60 (Magic Division): replace ~24-instruction emulated divide with
        # 5-instruction magic multiply+shift. The magic macro clobbers only a SINGLE
        # VGPR (v_tmp), not 4 like the old macro -- but the row_stride redesign's
        # s_k_block_off preservation concern still applies (s.s_tmp() is live across
        # this function for row_stride>1), and s.s_tmp() is NOT clobbered by the magic
        # macro (which has no SGPR scratch argument), so the div_rem_scratch workaround
        # is no longer needed. Kept here as a no-op for minimal diff.
        # Phase 60: magic multiply+shift for ho_wo and wo divisors
        self._emit(m_mdiv_rem_vs(v.v_gtc_tmp(1), v.v_gtc_tmp(2), v.v_gtc_tmp(0), s.s_magic_ho_wo(), s.s_shift_ho_wo(), s.s_ho_wo(), v.v_tmp()))
        self._emit(f"; v_gtc_tmp(1)=hw_idx (rem), v_gtc_tmp(2)=n_idx (quo)")
        self._emit(m_mdiv_rem_vs(v.v_gtc_tmp(3), v.v_gtc_tmp(4), v.v_gtc_tmp(1), s.s_magic_wo(), s.s_shift_wo(), s.s_wo(), v.v_tmp()))
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

        self._emit(f"; v_flag({row_offset}) = 1 iff (hi_idx, wi_idx) in [0,hi)x[0,wi)")
        self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_hi()}], v[{v.v_gtc_tmp(4)}]")
        self._emit(f"v_cndmask_b32 v[{v.v_flag(row_offset)}], 0, 1, vcc_lo")
        self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_wi()}], v[{v.v_gtc_tmp(3)}]")
        self._emit(f"v_cndmask_b32 v[{v.v_flag(row_offset)}], 0, v[{v.v_flag(row_offset)}], vcc_lo")
        if self.tunable.wmma_n_tail and self.row_stride == 1:
            # Phase 35: AND in B's persistent (kernel-lifetime-constant) column-in-range flag
            # -- v_flag_b_ntail was computed once in the prologue, not recomputed here.
            # row_stride redesign: N-tail becomes a sub_chunk-indexed (not row_offset-
            # indexed) axis when row_stride>1 -- combined separately via
            # _emit_gld_chunk_load's v_flag_col instead (see docs/gfx1250_wrw_addressing_redesign.md).
            self._emit(f"v_and_b32 v[{v.v_flag(row_offset)}], v[{v.v_flag(row_offset)}], v[{v.v_flag_b_ntail()}]   ; wmma_n_tail")
        if self.tunable.wmma_k_tail:
            # Phase 35: AND in the K-in-range check captured into v_flag_b_ktail above.
            self._emit(f"v_and_b32 v[{v.v_flag(row_offset)}], v[{v.v_flag(row_offset)}], v[{v.v_flag_b_ktail()}]   ; wmma_k_tail")
        self._emit_empty_line()

        self._emit(f"; row_idx = n_idx*(hi*wi) + hi_idx*wi + wi_idx (meaningless but harmless if")
        self._emit(f"; v_flag({row_offset})==0 -- that lane's global_load_b is EXEC-masked off, see global_load_b_functor)")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(0)}], s[{s.s_hi_wi()}], v[{v.v_gtc_tmp(2)}]")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(2)}], s[{s.s_wi()}], v[{v.v_gtc_tmp(4)}]")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(2)}]")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(3)}]")
        self._emit_empty_line()

        self._emit(f"; v_addr_b({row_offset}) = p_wei + row_idx * b_n_total * databyte + v_b_col_off (b_n_total = gemm_n*group --")
        self._emit(f"; input's pixel-to-pixel stride is its TOTAL C_in count, not the per-group gemm_n, see class docstring)")
        self._emit(f"v_mul_lo_u32 v[{v.v_gtc_tmp(0)}], s[{s.s_b_n_total()}], v[{v.v_gtc_tmp(0)}]   ; row_idx * b_n_total")
        self._emit(f"v_lshlrev_b32 v[{v.v_gtc_tmp(0)}], {utility_log2(self.data_byte)}, v[{v.v_gtc_tmp(0)}]")
        self._emit(f"v_add_u32 v[{v.v_gtc_tmp(0)}], v[{v.v_gtc_tmp(0)}], v[{v.v_b_col_off()}]")
        if self.tunable.saddr_global_load:
            # Phase 61: byte OFFSET only -- s_p_wei passed separately as SADDR.
            # row_stride==1 asserted when saddr is on, so row_offset is always 0.
            self._emit(f"v_mov_b32 v[{v.v_off_b()}], v[{v.v_gtc_tmp(0)}]   ; v_off_b = row_idx * b_n_total * databyte + v_b_col_off")
        else:
            self._emit(f"v_mov_b32 v[{v.v_addr_b(2 * row_offset + 1)}], s[{s.s_p_wei(1)}]")
            self._emit(f"v_add_co_u32 v[{v.v_addr_b(2 * row_offset)}], vcc_lo, s[{s.s_p_wei()}], v[{v.v_gtc_tmp(0)}]")
            self._emit(f"v_add_co_ci_u32 v[{v.v_addr_b(2 * row_offset + 1)}], vcc_lo, 0, v[{v.v_addr_b(2 * row_offset + 1)}], vcc_lo")

    def _emit_gld_chunk_load(self, v_gld, v_addr, chunk_idx, v_flag=None, v_flag_col=None, saddr=None):
        ''' Phase 1 (k-sub-loop): issues (does not wait) ONE inst_wmma.k-wide chunk's
        global load into the small, reused v_gld buffer.

        row_stride redesign: when row_stride>1, chunk_idx (a compile-time Python loop
        variable, fully unrolled -- see _emit_sst_all_chunks) is decoded into
        (row_offset, sub_chunk) = divmod(chunk_idx, chunks_per_row): row_offset selects
        which of this thread's row_stride owned VADDR pairs to read
        (v_addr(2*row_offset)/(2*row_offset+1)), sub_chunk is the compile-time byte
        immediate within that row -- see docs/gfx1250_wrw_addressing_redesign.md.
        row_stride==1 always resolves row_offset=0, byte-identical to before.

        v_flag_col (row_stride redesign only): a SECOND, independent masking input
        indexed by sub_chunk instead of row_offset (M/N-tail, which depends on which
        slice of the shared full-width row this is, not which owned row) -- ANDed
        together with v_flag (row-indexed: K-tail/spatial validity) into scratch when
        both are present, since the two axes are independent (avoids a wasteful
        row_stride*chunks_per_row cross-product of precombined flags). '''
        if self.row_stride > 1:
            row_offset, sub_chunk = divmod(chunk_idx, self.chunks_per_row)
        else:
            row_offset, sub_chunk = 0, chunk_idx
        a_lo, a_hi = v_addr(2 * row_offset), v_addr(2 * row_offset + 1)
        if v_flag is not None or v_flag_col is not None:
            emit_vopd_paired_zero_init(self._emit, v_gld, self.chunk_num_dwords)
            if v_flag is not None and v_flag_col is not None:
                self._emit(f"v_and_b32 v[{self.vgpr.v_tmp(3)}], v[{v_flag(row_offset)}], v[{v_flag_col(sub_chunk)}]   ; combine row-indexed and sub_chunk-indexed validity")
                self._emit(f"v_cmpx_le_u32 1, v[{self.vgpr.v_tmp(3)}]")
            elif v_flag is not None:
                self._emit(f"v_cmpx_le_u32 1, v[{v_flag(row_offset)}]")
            else:
                self._emit(f"v_cmpx_le_u32 1, v[{v_flag_col(sub_chunk)}]")
        for i in range(self.chunk_num_dwordx4):
            idx = sub_chunk * self.chunk_num_dwordx4 + i
            if saddr is not None:
                self._emit(f"global_load_dwordx4 v[{v_gld(i*4)}:{v_gld(i*4+3)}], v[{v_addr()}], s[{saddr()}:{saddr(1)}] offset:{idx*16}")
            else:
                self._emit(f"global_load_dwordx4 v[{v_gld(i*4)}:{v_gld(i*4+3)}], v[{a_lo}:{a_hi}], off offset:{idx*16}")
        if v_flag is not None or v_flag_col is not None:
            self._emit(f"s_mov_b32 exec_lo, -1")

    def _emit_sst_chunk(self, v_gld, v_sst_os, sst_extra_off, chunk_idx):
        ''' Phase 1 (k-sub-loop): stores ONE already-loaded-and-waited chunk to LDS.

        row_stride redesign: row_offset*row_pitch_bytes is folded into the `offset:`
        immediate -- both row_pitch_bytes (gemm_m_per_block*data_byte) and row_offset
        are compile-time constants, so this needs no new runtime arithmetic (see
        docs/gfx1250_wrw_addressing_redesign.md for the derivation: v_sst_os=tid*bytes_per_row
        is already exactly row_offset=0's slot; further owned rows are a fixed further
        offset). row_stride==1 always resolves row_offset=0, byte-identical to before. '''
        if self.row_stride > 1:
            row_offset, sub_chunk = divmod(chunk_idx, self.chunks_per_row)
            row_pitch_bytes = self.tunable.gemm_m_per_block * self.data_byte
        else:
            row_offset, sub_chunk = 0, chunk_idx
            row_pitch_bytes = 0
        for i in range(self.chunk_num_dwordx4):
            idx = sub_chunk * self.chunk_num_dwordx4 + i
            self._emit(f"ds_write_b128 v[{v_sst_os()}], v[{v_gld(i*4)}:{v_gld(i*4+3)}] offset:{sst_extra_off + row_offset * row_pitch_bytes + idx*16}")

    def _emit_sst_remaining_chunks(self, v_gld, v_addr, v_sst_os, sst_extra_off, v_flag=None, v_flag_col=None, saddr=None):
        '''
        Phase 1 (k-sub-loop): stores chunk 0 (already loaded+waited via the existing
        global_load_a/b_functor + outer s_wait_loadcnt call sequence in
        wmma_main_loop.py), then load+wait+stores chunks 1..num_k_chunks-1 sequentially,
        reusing the same small v_gld buffer -- see igemm_fwd_gtc_wmma_nhwc_t's identically-
        named method for why (preserves the single-buffered-LDS safety invariant: no wave
        may overwrite a tile's LDS storage until every wave has finished reading it).

        v_flag_col is only ever non-None for row_stride>1 (see _emit_gld_chunk_load), which
        always has num_k_chunks>1 and so never reaches this function (see
        shared_store_a/b_functor) -- accepted here purely for a uniform call signature.
        '''
        self._emit_sst_chunk(v_gld, v_sst_os, sst_extra_off, 0)
        for c in range(1, self.num_k_chunks):
            self._emit_gld_chunk_load(v_gld, v_addr, c, v_flag=v_flag, v_flag_col=v_flag_col, saddr=saddr)
            self._emit(f"s_wait_loadcnt 0x0")
            self._emit_sst_chunk(v_gld, v_sst_os, sst_extra_off, c)

    def _emit_sst_all_chunks(self, v_gld, v_addr, v_sst_os, sst_extra_off, v_flag=None, v_flag_col=None, saddr=None):
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
            self._emit_gld_chunk_load(v_gld, v_addr, c, v_flag=v_flag, v_flag_col=v_flag_col, saddr=saddr)
            self._emit_sst_chunk(v_gld, v_sst_os, sst_extra_off, c)

    def _a_flag_symbol(self):
        '''
        Phase 35: which VGPR (if any) masks A's (grad_output) load. When wmma_k_tail is set,
        v_flag_a_ktail already has wmma_m_tail's flag ANDed into it every time it's
        recomputed (see _emit_a_kflag), so it alone is the correct combined flag whenever
        K-tail is active, regardless of whether M-tail is also active. When only wmma_m_tail
        is set, its own persistent flag is used directly. Returns None (no masking) when
        neither is set -- every existing config's byte-identical behavior.

        row_stride redesign: M-tail is a sub_chunk-indexed (not row_offset-indexed) axis
        here, so it's combined separately via _emit_gld_chunk_load's v_flag_col instead of
        being pre-anded into v_flag_a_ktail (see _emit_a_kflag) -- this accessor only ever
        returns the row-indexed K-tail flag (or None) when row_stride>1.
        '''
        if self.row_stride > 1:
            if self.tunable.wmma_k_tail:
                return self.vgpr.v_flag_a_ktail
            return None
        if self.tunable.wmma_k_tail:
            return self.vgpr.v_flag_a_ktail
        if self.tunable.wmma_m_tail:
            return self.vgpr.v_flag_a_mtail
        return None

    def _a_flag_col_symbol(self):
        ''' row_stride redesign: sub_chunk-indexed M-tail flag for A's load masking
        (_emit_gld_chunk_load's v_flag_col) -- None when row_stride==1 (M-tail is already
        folded into _a_flag_symbol's row-indexed flag there instead) or wmma_m_tail unset. '''
        if self.row_stride > 1 and self.tunable.wmma_m_tail:
            return self.vgpr.v_flag_a_mtail
        return None

    def _b_flag_col_symbol(self):
        ''' row_stride redesign: sub_chunk-indexed N-tail flag for B's load masking. '''
        if self.row_stride > 1 and self.tunable.wmma_n_tail:
            return self.vgpr.v_flag_b_ntail
        return None

    def global_load_a_functor(self):
        ''' Phase 1: only issues chunk 0's load when num_k_chunks==1 (no clobbering risk
        then, since emit_extra_substeps() never runs) -- see _emit_sst_all_chunks.
        Phase 15 (main_loop_interleave): issues chunk 0's load even when num_k_chunks>1 --
        the interleaved main loop needs chunk 0 in-flight early, and shared_load_a no longer
        clobbers v_gld_a (redirected to v_scratch). '''
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                s = outer.sgpr
                with outer._deferred_context():
                    if outer.tunable.tdm_global_load:
                        # Phase 45: mirrors fwd/bwd's identical TDM branch -- one TDM
                        # instruction moves the whole gemm_k_per_block x gemm_m_per_block
                        # tile straight into LDS, wave-0-only issue.
                        outer._emit_wave0_only(lambda: outer._emit(f"tensor_load_to_lds s[{s.s_tdm_g0()}:{s.s_tdm_g0(3)}], s[{s.s_tdm_g1()}:{s.s_tdm_g1(7)}]"))
                    elif outer.num_k_chunks == 1 or outer.tunable.main_loop_interleave:
                        if outer.tunable.saddr_global_load:
                            outer._emit_gld_chunk_load(v.v_gld_a, v.v_off_a, 0, v_flag=outer._a_flag_symbol(), saddr=s.s_p_in)
                        else:
                            outer._emit_gld_chunk_load(v.v_gld_a, v.v_addr_a, 0, v_flag=outer._a_flag_symbol())
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
                s = outer.sgpr
                with outer._deferred_context():
                    if outer.tunable.tdm_global_load:
                        # Phase 45: mirrors global_load_a_functor's TDM branch above.
                        outer._emit_wave0_only(lambda: outer._emit(f"tensor_load_to_lds s[{s.s_tdm_g0_b()}:{s.s_tdm_g0_b(3)}], s[{s.s_tdm_g1_b()}:{s.s_tdm_g1_b(7)}]"))
                    elif outer.num_k_chunks == 1:
                        if outer.tunable.saddr_global_load:
                            outer._emit_gld_chunk_load(v.v_gld_b, v.v_off_b, 0, v_flag=v.v_flag, saddr=s.s_p_wei)
                        else:
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
                s = outer.sgpr
                with outer._deferred_context():
                    v_a_addr = v.v_off_a if outer.tunable.saddr_global_load else v.v_addr_a
                    s_a_saddr = s.s_p_in if outer.tunable.saddr_global_load else None
                    if outer.num_k_chunks == 1 or outer.tunable.main_loop_interleave:
                        outer._emit_sst_remaining_chunks(v.v_gld_a, v_a_addr, v.v_sst_os, 0, v_flag=outer._a_flag_symbol(), v_flag_col=outer._a_flag_col_symbol(), saddr=s_a_saddr)
                    else:
                        outer._emit_sst_all_chunks(v.v_gld_a, v_a_addr, v.v_sst_os, 0, v_flag=outer._a_flag_symbol(), v_flag_col=outer._a_flag_col_symbol(), saddr=s_a_saddr)
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
                    v_b_addr = v.v_off_b if outer.tunable.saddr_global_load else v.v_addr_b
                    s_b_saddr = s.s_p_wei if outer.tunable.saddr_global_load else None
                    if outer.num_k_chunks == 1:
                        outer._emit_sst_remaining_chunks(v.v_gld_b, v_b_addr, v.v_sst_os, outer.lds_a_size, v_flag=v.v_flag, v_flag_col=outer._b_flag_col_symbol(), saddr=s_b_saddr)
                    else:
                        outer._emit_sst_all_chunks(v.v_gld_b, v_b_addr, v.v_sst_os, outer.lds_a_size, v_flag=v.v_flag, v_flag_col=outer._b_flag_col_symbol(), saddr=s_b_saddr)
                return outer._get_deferred()
            def get_issues(self):
                return outer.chunk_num_dwordx4
        return functor_t()

    def global_load_chunk_a_functor(self):
        ''' Phase 15 (wrw port): single-chunk primitive for the interleaved main loop --
        issues ONE chunk's global load of A (grad_output, transposed). Reuses _emit_gld_chunk_load
        with the same flag/flag_col as global_load_a_functor's chunk-0 call. row_stride==1
        asserted when interleave is on, so v_flag_col is always None here. '''
        outer = self
        class functor_t:
            def __call__(self, chunk_idx):
                v = outer.vgpr
                s = outer.sgpr
                with outer._deferred_context():
                    if outer.tunable.saddr_global_load:
                        outer._emit_gld_chunk_load(v.v_gld_a, v.v_off_a, chunk_idx, v_flag=outer._a_flag_symbol(), v_flag_col=outer._a_flag_col_symbol(), saddr=s.s_p_in)
                    else:
                        outer._emit_gld_chunk_load(v.v_gld_a, v.v_addr_a, chunk_idx, v_flag=outer._a_flag_symbol(), v_flag_col=outer._a_flag_col_symbol())
                return outer._get_deferred()
        return functor_t()

    def shared_store_chunk_a_functor(self):
        ''' Phase 15 (wrw port): single-chunk primitive for the interleaved main loop --
        stores ONE already-loaded-and-waited chunk of A to LDS (reuses _emit_sst_chunk). '''
        outer = self
        class functor_t:
            def __call__(self, chunk_idx):
                v = outer.vgpr
                with outer._deferred_context():
                    outer._emit_sst_chunk(v.v_gld_a, v.v_sst_os, 0, chunk_idx)
                return outer._get_deferred()
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
                if outer.tunable.ds_load_tr_b:
                    # Phase 64: native ds_load_tr16_b128 replaces the entire manual
                    # read+pack loop below -- see bwd's shared_load_b_functor (Phase 63)
                    # for the concrete derivation; identical mechanism, side='m' here.
                    assert outer.data_byte == 2 and num_v_a == 8
                    with outer._deferred_context():
                        for i_rm in range(outer.tunable.wmma_repeat_m):
                            col_off = i_rm * outer.tunable.wmma_tile_m * outer.data_byte
                            base = extra_off + col_off
                            dst0 = slot_off + i_rm * num_v_a
                            dst4 = dst0 + num_v_a // 2
                            k_shift = outer.wmma_mapping.ctrl.inst_wmma.k // 4   # = half_k // 2 = 8
                            outer._emit(f"ds_load_tr16_b128 v[{v.v_a(dst0)}:{v.v_a(dst0+3)}], v[{v.v_sld_a_os()}] offset:{base}")
                            outer._emit(f"ds_load_tr16_b128 v[{v.v_a(dst4)}:{v.v_a(dst4+3)}], v[{v.v_sld_a_os()}] offset:{base + k_shift * row_pitch}")
                    return outer._get_deferred()
                elem_per_dword = 4 // outer.data_byte
                if outer.data_byte == 2:
                    read_instr = 'ds_read_u16'
                elif outer.data_byte == 4:
                    read_instr = 'ds_read_b32'   # fp32: full-dword read, no zero-extension needed
                else:
                    read_instr = 'ds_read_u8'
                # Phase 15 (wrw port): when main_loop_interleave is active, v_gld_a holds
                # in-flight A chunk loads during compute -- use v_scratch instead of v_gld_a
                # for the read-and-pack scratch. v_scratch is sized to elem_per_dword (max 4).
                if outer.tunable.main_loop_interleave:
                    v_scratch = lambda s: v.v_scratch(s)
                else:
                    v_scratch = lambda s: v.v_gld_a(s)
                # Phase 63 (wait-batching): see bwd's shared_load_b_functor for the
                # rationale -- batch reads up to the scratch buffer's actual capacity
                # before a single wait, instead of one wait per `a`.
                batch_cap = outer.chunk_num_dwords if not outer.tunable.main_loop_interleave else 4
                with outer._deferred_context():
                    for i_rm in range(outer.tunable.wmma_repeat_m):
                        col_off = i_rm * outer.tunable.wmma_tile_m * outer.data_byte
                        total_slots = num_v_a * elem_per_dword
                        slot = 0
                        while slot < total_slots:
                            n_batch = min(batch_cap, total_slots - slot)
                            for j in range(n_batch):
                                abs_slot = slot + j
                                off = col_off + abs_slot * row_pitch
                                outer._emit(f"{read_instr} v[{v_scratch(j)}], v[{v.v_sld_a_os()}] offset:{extra_off + off}")
                            outer._emit(f"s_wait_dscnt 0x0")
                            for j in range(n_batch):
                                abs_slot = slot + j
                                a, s = abs_slot // elem_per_dword, abs_slot % elem_per_dword
                                dst = v.v_a(slot_off+i_rm*num_v_a+a)
                                if s == 0:
                                    outer._emit(f"v_mov_b32 v[{dst}], v[{v_scratch(j)}]")
                                else:
                                    shift = s * 8 * outer.data_byte
                                    outer._emit(f"v_lshl_or_b32 v[{dst}], v[{v_scratch(j)}], {shift}, v[{dst}]")
                            slot += n_batch
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
                if outer.tunable.ds_load_tr_b:
                    # Phase 64: native ds_load_tr16_b128, identical to bwd's Phase 63
                    # (see igemm_bwd_gtc_wmma_nhwc.py's shared_load_b_functor).
                    assert outer.data_byte == 2 and num_v_b == 8
                    with outer._deferred_context():
                        for i_rn in range(outer.tunable.wmma_repeat_n):
                            col_off = i_rn * outer.tunable.wmma_tile_n * outer.data_byte
                            base = extra_off + outer.lds_a_size + col_off
                            dst0 = slot_off + i_rn * num_v_b
                            dst4 = dst0 + num_v_b // 2
                            k_shift = outer.wmma_mapping.ctrl.inst_wmma.k // 4   # = half_k // 2 = 8
                            outer._emit(f"ds_load_tr16_b128 v[{v.v_b(dst0)}:{v.v_b(dst0+3)}], v[{v.v_sld_b_os()}] offset:{base}")
                            outer._emit(f"ds_load_tr16_b128 v[{v.v_b(dst4)}:{v.v_b(dst4+3)}], v[{v.v_sld_b_os()}] offset:{base + k_shift * row_pitch}")
                    return outer._get_deferred()
                elem_per_dword = 4 // outer.data_byte
                if outer.data_byte == 2:
                    read_instr = 'ds_read_u16'
                elif outer.data_byte == 4:
                    read_instr = 'ds_read_b32'   # fp32: full-dword read, no zero-extension needed
                else:
                    read_instr = 'ds_read_u8'
                batch_cap = outer.chunk_num_dwords
                with outer._deferred_context():
                    for i_rn in range(outer.tunable.wmma_repeat_n):
                        col_off = i_rn * outer.tunable.wmma_tile_n * outer.data_byte
                        total_slots = num_v_b * elem_per_dword
                        slot = 0
                        while slot < total_slots:
                            n_batch = min(batch_cap, total_slots - slot)
                            for j in range(n_batch):
                                abs_slot = slot + j
                                # B region starts at outer.lds_a_size within the shared LDS
                                # tile; v_sld_b_os only carries the offset local to B's own
                                # region.
                                off = outer.lds_a_size + col_off + abs_slot * row_pitch
                                outer._emit(f"{read_instr} v[{v.v_gld_b(j)}], v[{v.v_sld_b_os()}] offset:{extra_off + off}")
                            outer._emit(f"s_wait_dscnt 0x0")
                            for j in range(n_batch):
                                abs_slot = slot + j
                                a, s = abs_slot // elem_per_dword, abs_slot % elem_per_dword
                                dst = v.v_b(slot_off+i_rn*num_v_b+a)
                                if s == 0:
                                    outer._emit(f"v_mov_b32 v[{dst}], v[{v.v_gld_b(j)}]")
                                else:
                                    shift = s * 8 * outer.data_byte
                                    outer._emit(f"v_lshl_or_b32 v[{dst}], v[{v.v_gld_b(j)}], {shift}, v[{dst}]")
                            slot += n_batch
                return outer._get_deferred()
        return functor_t()

    def _emit_a_kflag(self, s_k_block_off):
        '''
        Phase 35: A's (grad_output) per-iteration GEMM_K-in-range flag, mirroring
        _emit_b_gather's k_abs computation but using A's own persistent row-within-K-block
        value (v_row_local -- the SAME formula/value B uses, since A and B share the same
        num_col_groups/col_group_bits tiling scheme, see class docstring). Unlike B, A has no
        spatial decomposition to do, so this only ever needs the boolean flag itself. Writes
        into v_flag_a_ktail; if wmma_m_tail is also set, ANDs its persistent flag in here too
        (rather than merging the two elsewhere) so v_flag_a_ktail is always the single,
        ready-to-use combined flag for A's loads (see _a_flag_symbol()).

        row_stride redesign: when row_stride>1, this thread owns row_stride consecutive
        absolute K-positions (v_row_local is the first); loops row_offset exactly like
        _emit_b_gather, writing each row's result into the row_stride-sized
        v_flag_a_ktail(row_offset). M-tail is deliberately NOT anded in here for
        row_stride>1 -- M-tail's validity depends on sub_chunk (which slice of the full
        row), not row_offset, an independent axis combined on the fly by
        _emit_gld_chunk_load's v_flag_col instead (see docs/gfx1250_wrw_addressing_redesign.md).
        row_stride==1 is a single iteration, byte-identical to before.
        '''
        s = self.sgpr
        v = self.vgpr
        for row_offset in range(self.row_stride):
            if s_k_block_off is not None:
                self._emit(f"v_add_u32 v[{v.v_flag_a_ktail(row_offset)}], s[{s_k_block_off}], v[{v.v_row_local()}]   ; k_abs (within this workgroup's K-slice)")
            else:
                self._emit(f"v_mov_b32 v[{v.v_flag_a_ktail(row_offset)}], v[{v.v_row_local()}]   ; k_abs (k_block_off=0, within this workgroup's K-slice)")
            if row_offset:
                self._emit(f"v_add_u32 v[{v.v_flag_a_ktail(row_offset)}], v[{v.v_flag_a_ktail(row_offset)}], {row_offset}   ; += row_offset (this thread's {row_offset}-th owned row)")
            if self.tunable.gemm_k_global_split:
                self._emit(f"v_add_u32 v[{v.v_flag_a_ktail(row_offset)}], v[{v.v_flag_a_ktail(row_offset)}], s[{s.s_gemm_k_wg_off()}]   ; += this workgroup's K-slice base")
            self._emit(f"v_cmp_gt_u32 vcc_lo, s[{s.s_gemm_k()}], v[{v.v_flag_a_ktail(row_offset)}]")
            self._emit(f"v_cndmask_b32 v[{v.v_flag_a_ktail(row_offset)}], 0, 1, vcc_lo   ; wmma_k_tail: k_abs < real gemm_k")
            if self.tunable.wmma_m_tail and self.row_stride == 1:
                self._emit(f"v_and_b32 v[{v.v_flag_a_ktail(row_offset)}], v[{v.v_flag_a_ktail(row_offset)}], v[{v.v_flag_a_mtail()}]   ; wmma_m_tail")

    def move_slice_window_a_functor(self):
        outer = self
        class functor_t:
            def __call__(self):
                v = outer.vgpr
                s = outer.sgpr
                with outer._deferred_context():
                    if outer.tunable.tdm_global_load:
                        # Phase 45: advance by s_a_k_stride (one K-tile's worth of ROWS --
                        # GEMM_K is the row axis for A here, same reasoning as bwd's B in
                        # Phase 42). K-tail-via-hardware-OOB rebuild mirrors Phase 44's
                        # skip-unless-genuinely-partial guard from the very start (no
                        # separate unconditional-rebuild step, since Phase 44 already
                        # validated this pattern for fwd/bwd).
                        outer._emit(f"s_add_u32 s[{s.s_tdm_g0(2)}], s[{s.s_tdm_g0(2)}], s[{s.s_a_k_stride()}]")
                        outer._emit(f"s_addc_u32 s[{s.s_tdm_g0(3)}], s[{s.s_tdm_g0(3)}], 0")
                        skip_label = f"L_{outer.name()}_tdm_a_skip_rebuild"
                        outer._emit(f"s_cmp_lt_i32 s[{s.s_tdm_k_remain()}], {outer.tunable.gemm_k_per_block}   ; Phase 44/45: is the tile now being prepared genuinely partial?")
                        outer._emit(f"s_cbranch_scc0 {skip_label}   ; not partial -- skip the rebuild, tensor_dim1 stays >= tile_dim1")
                        outer._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_gemm_m()}], 16   ; tensor_dim0 (gemm_m) hi16 -- unchanged, re-derived fresh")
                        outer._emit(f"s_lshl_b32 s[{s.s_tmp(1)}], s[{s.s_tdm_k_remain()}], 16   ; tensor_dim1 (remaining K) lo16")
                        outer._emit(f"s_or_b32 s[{s.s_tdm_g1(2)}], s[{s.s_tmp(0)}], s[{s.s_tmp(1)}]")
                        outer._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_tdm_k_remain()}], 16   ; tensor_dim1 (remaining K) hi16")
                        outer._emit(f"s_or_b32 s[{s.s_tdm_g1(3)}], s[{s.s_tmp(0)}], {outer.tunable.gemm_m_per_block << 16}   ; | tile_dim0 (compile-time)")
                        outer._emit_front(f"{skip_label}:")
                    elif outer.tunable.saddr_global_load:
                        # Phase 61: v_off_a is a plain 32-bit byte OFFSET, advance by
                        # s_a_k_stride (one K-tile's worth of rows) -- no carry chain.
                        outer._emit(f"v_add_u32 v[{v.v_off_a()}], s[{s.s_a_k_stride()}], v[{v.v_off_a()}]")
                        if outer.tunable.wmma_k_tail:
                            outer._emit(f"s_sub_i32 s[{s.s_tmp(1)}], s[{s.s_knum()}], s[{s.s_kitr()}]   ; k_block_off")
                            outer._emit_a_kflag(s.s_tmp(1))
                    else:
                        # row_stride redesign: every owned row advances by the SAME
                        # s_a_k_stride (one whole K-block's worth of rows) each iteration --
                        # row_stride==1 is a single pair, byte-identical to before.
                        for row_offset in range(outer.row_stride):
                            outer._emit(f"v_add_co_u32 v[{v.v_addr_a(2 * row_offset)}], vcc_lo, s[{s.s_a_k_stride()}], v[{v.v_addr_a(2 * row_offset)}]")
                            outer._emit(f"v_add_co_ci_u32 v[{v.v_addr_a(2 * row_offset + 1)}], vcc_lo, 0, v[{v.v_addr_a(2 * row_offset + 1)}], vcc_lo")
                        if outer.tunable.wmma_k_tail:
                            outer._emit(f"s_sub_i32 s[{s.s_tmp(1)}], s[{s.s_knum()}], s[{s.s_kitr()}]   ; k_block_off")
                            outer._emit_a_kflag(s.s_tmp(1))
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

        Phase 45 (tdm_global_load): this whole non-TDM gather-recompute is bypassed --
        TDM's B descriptor uses a genuine constant per-K-block stride (s_b_k_stride,
        mirroring A's s_a_k_stride) instead, since the 1x1/unit-stride restriction makes
        B's gather a provable identity (see _emit_tdm_descriptor_setup_a's docstring).
        '''
        outer = self
        class functor_t:
            def __call__(self):
                s = outer.sgpr
                with outer._deferred_context():
                    if outer.tunable.tdm_global_load:
                        outer._emit(f"s_add_u32 s[{s.s_tdm_g0_b(2)}], s[{s.s_tdm_g0_b(2)}], s[{s.s_b_k_stride()}]")
                        outer._emit(f"s_addc_u32 s[{s.s_tdm_g0_b(3)}], s[{s.s_tdm_g0_b(3)}], 0")
                        skip_label = f"L_{outer.name()}_tdm_b_skip_rebuild"
                        outer._emit(f"s_cmp_lt_i32 s[{s.s_tdm_k_remain()}], {outer.tunable.gemm_k_per_block}   ; Phase 44/45: is the tile now being prepared genuinely partial?")
                        outer._emit(f"s_cbranch_scc0 {skip_label}   ; not partial -- skip the rebuild, tensor_dim1 stays >= tile_dim1")
                        outer._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_gemm_n()}], 16   ; tensor_dim0 (gemm_n) hi16 -- unchanged, re-derived fresh")
                        outer._emit(f"s_lshl_b32 s[{s.s_tmp(1)}], s[{s.s_tdm_k_remain()}], 16   ; tensor_dim1 (remaining K) lo16")
                        outer._emit(f"s_or_b32 s[{s.s_tdm_g1_b(2)}], s[{s.s_tmp(0)}], s[{s.s_tmp(1)}]")
                        outer._emit(f"s_lshr_b32 s[{s.s_tmp(0)}], s[{s.s_tdm_k_remain()}], 16   ; tensor_dim1 (remaining K) hi16")
                        outer._emit(f"s_or_b32 s[{s.s_tdm_g1_b(3)}], s[{s.s_tmp(0)}], {outer.tunable.gemm_n_per_block << 16}   ; | tile_dim0 (compile-time)")
                        outer._emit_front(f"{skip_label}:")
                    else:
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
        # Phase 15 (wrw port): A (transposed, but redirected to v_scratch) interleaves;
        # B (transposed, v_gld_b reused as scratch by shared_load_b) stays deferred.
        ctrl.interleave_a = self.tunable.main_loop_interleave
        ctrl.interleave_b = False
        ctrl.wmma_setprio = self.tunable.wmma_setprio
        ctrl.tdm_global_to_lds_a = self.tunable.tdm_global_load
        ctrl.tdm_global_to_lds_b = self.tunable.tdm_global_load
        # Phase 71 (PERF-004): both A and B are transposed (pack/wait-batched
        # shared_load technique) -- not eligible for wmma_main_loop.py's
        # partial-wait schedule (defaults already False; explicit for clarity).
        ctrl.ds_read_plain_a = False
        ctrl.ds_read_plain_b = False
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
        # also the sole caller of this method and also emits the per-tap epilogue store
        # (see class docstring, Phase 5f)
        wmma_main_loop_t(self.mc, ctrl).emit()

    def emit_kernel_streamk_loop(self):
        '''
        Phase 58: persistent-kernel proof of mechanism (docs/gfx1250_streamk_design.md,
        "Approach A", scoped to K-only / single-tap). Replaces emit_kernel_tap_loop()'s
        (nxe==0, single-iteration) per-shard body with a BOUNDED loop (max_iters is a
        host-computed, launch-time-constant kernarg -- no data-dependent trip count, so no
        hang risk from the loop bound itself) that dynamically claims K-shard indices from a
        per-(bx,by) global atomic counter, instead of the static workgroup_id_z-derived
        shard index every other gemm_k_global_split kernel uses. Only called when
        wrw_streamk is set (nxe==0, wmma_k_tail=0, tdm_global_load=0 all asserted in
        igemm_base.py, so this method never needs to handle those cases).
        '''
        s = self.sgpr
        v = self.vgpr
        # nxe==0 guarantees y=x=1 (asserted in igemm_base.py), so ix is always 0 -- normally
        # zeroed inside emit_kernel_tap_loop()'s label_tap_y top (s_iy itself IS zeroed
        # unconditionally in emit_kernel_prologue), but this method never calls the tap loop
        # at all, so s_ix was otherwise left holding whatever garbage occupied that SGPR at
        # kernel entry -- found via a hardware raw-dump diagnostic (v_gld_b consistently
        # read back as exactly 0 for every thread/shard, tracing back to _emit_b_gather's
        # OOB v_flag masking every single B load because wi_idx, derived from garbage s_ix,
        # was always out of [0, wi)).
        self._emit(f"s_mov_b32 s[{s.s_ix()}], 0")
        label_loop = f"L_{self.name()}_streamk_loop"
        label_skip = f"L_{self.name()}_streamk_skip"

        self._emit(f"s_mov_b32 s[{s.s_streamk_iter()}], 0")
        self._emit_front(f"{label_loop}:")

        # ---- claim the next shard index. Only flat-tid==0 issues the atomic (EXEC-masked,
        # not a per-wave check -- v_tid is the flat 0..127 thread id, so this is exactly one
        # lane across the whole 4-wave workgroup); the result is broadcast to every lane via
        # an LDS round-trip + a REAL cross-wave barrier (this kernel is always block_size
        # 128 / 4-wave -- see docs/gfx1250_streamk_design.md's note on why rocKE's
        # single-wave ds_bpermute broadcast doesn't apply here and an actual barrier is
        # required and cannot be elided by a compiler that never exists in this hand-
        # assembled pipeline in the first place). gfx1250's VOPC "x" form takes no explicit
        # exec destination (implicit); the atomic uses the SADDR form (s_streamk_addr) with
        # a VOFFSET of 0 (v_streamk_zero) and needs an explicit th:TH_ATOMIC_RETURN to get a
        # return value at all -- both confirmed via llvm-mc against this exact ISA. gfx1250
        # has no plain s_barrier -- it's always the split s_barrier_signal/s_barrier_wait
        # form (see docs/claude_persistent_memory_notes.md's ISA quirks). ----
        self._emit(f"; Phase 58: claim next K-shard index (only flat tid==0 issues the atomic)")
        self._emit(f"v_cmpx_eq_u32 0, v[{v.v_tid()}]")
        self._emit(f"global_atomic_add_u32 v[{v.v_streamk_claim()}], v[{v.v_streamk_zero()}], v[{v.v_streamk_one()}], s[{s.s_streamk_addr()}:{s.s_streamk_addr(1)}] scope:SCOPE_SYS th:TH_ATOMIC_RETURN")
        self._emit(f"s_wait_loadcnt 0x0")
        self._emit(f"ds_write_b32 v[{v.v_streamk_zero()}], v[{v.v_streamk_claim()}] offset:{self.streamk_lds_off}")
        self._emit(f"s_mov_b32 exec_lo, -1")
        self._emit(f"s_wait_dscnt 0x0   ; the ds_write must retire before the barrier releases other waves to read it")
        self._emit(f"s_barrier_signal -1")
        self._emit(f"s_barrier_wait -1")
        self._emit(f"ds_read_b32 v[{v.v_streamk_claim()}], v[{v.v_streamk_zero()}] offset:{self.streamk_lds_off}")
        self._emit(f"s_wait_dscnt 0x0")
        self._emit(f"s_barrier_signal -1   ; ensure every lane has read before the next iteration can reuse this LDS slot")
        self._emit(f"s_barrier_wait -1")
        self._emit(f"v_readfirstlane_b32 s[{s.s_streamk_tile_idx()}], v[{v.v_streamk_claim()}]")
        self._emit_empty_line()

        self._emit(f"s_cmp_lt_u32 s[{s.s_streamk_tile_idx()}], s[{s.s_gemm_k_num_splits()}]   ; in_range: did we claim a real shard?")
        self._emit(f"s_cbranch_scc0 {label_skip}")
        self._emit_empty_line()

        # ---- process the claimed shard: this workgroup's K-slice base, in gemm_k_per_wg
        # units, is now (claimed tile_idx) instead of the static s_bz -- everything else
        # downstream (s_gemm_k_wg_off feeds _emit_b_gather's per-iteration gather AND, via
        # the fresh v_addr_a recompute below, A's address) is unchanged from the static
        # design's per-shard body. ----
        self._emit(f"s_mul_i32 s[{s.s_gemm_k_wg_off()}], s[{s.s_streamk_tile_idx()}], s[{s.s_gemm_k_per_wg()}]")
        self._emit_empty_line()

        # ---- Phase 58 Approach C: per-iteration workspace-shard offset for non-atomic
        # wrw_reduction_kernel epilogue. The existing prologue computes s_p_out += bz *
        # group * gemm_m * wei_row_c once from the static blockIdx.z -- wrong when the
        # shard index changes each iteration. Recomputed here from s_streamk_tile_idx
        # instead, into a fresh per-iteration copy (s_p_out_tap -- which already exists and
        # is normally used for per-tap column offsets; since nxe==0 guarantees y=x=1, the
        # column offset is always 0, so s_p_out_tap is freely reusable as the shard-offset
        # base). The coalescing-store call below passes s_p_out_tap rather than s_p_out,
        # matching what emit_kernel_tap_loop() already does for its own non-atomic path.
        # For the atomic-epilogue path (wrw_reduction_kernel=0), the original prologue
        # offset (bz=0, i.e. no workspace) is untouched, and the atomic-add address
        # mechanism is shard-agnostic anyway -- but for consistency the coalescing-store
        # call always receives the freshly-computed s_p_out_tap. ----
        if self.tunable.wrw_reduction_kernel:
            self._emit(f"; wrw_reduction_kernel + streamk: shard offset = tile_idx * group * gemm_m * wei_row_c elements")
            self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_group()}], s[{s.s_gemm_m()}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], s[{s.s_wei_row_c()}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], s[{s.s_streamk_tile_idx()}]")
            self._emit(f"s_lshl_b32 s[{s.s_tmp(0)}], s[{s.s_tmp(0)}], 2   ; workspace is always fp32 (4B)")
            self._emit(f"s_add_u32 s[{s.s_p_out_tap()}], s[{s.s_p_out()}], s[{s.s_tmp(0)}]")
            self._emit(f"s_addc_u32 s[{s.s_p_out_tap(1)}], s[{s.s_p_out(1)}], 0")
        else:
            # Atomic epilogue: s_p_out_tap == s_p_out (no shard-offset workspace needed).
            # The copy is still done to keep the coalescing-store argument uniform
            # regardless of the reduction strategy -- and to match emit_kernel_tap_loop()'s
            # own convention (always passes s_p_out_tap).
            self._emit(f"s_mov_b32 s[{s.s_p_out_tap()}], s[{s.s_p_out()}]")
            self._emit(f"s_mov_b32 s[{s.s_p_out_tap(1)}], s[{s.s_p_out(1)}]")
        self._emit_empty_line()

        self._emit(f"; clear accumulator (fresh per claimed shard)")
        emit_vopd_paired_zero_init(self._emit, v.v_c, self.tunable.num_vgpr_accumulate_c)
        self._emit_empty_line()

        if self.lds_buffer_num == 2:
            self._emit_lds_offset_setup()

        # ---- v_addr_a = v_addr_a_base + (gemm_k_wg_off * a_m_total) * databyte, recomputed
        # fresh every claimed shard (v_addr_a_base itself has NO split-K offset folded in --
        # see emit_kernel_prologue's Phase 58 note) ----
        self._emit(f"s_mul_i32 s[{s.s_tmp(2)}], s[{s.s_gemm_k_wg_off()}], s[{s.s_a_m_total()}]")
        self._emit(f"s_lshl_b32 s[{s.s_tmp(2)}], s[{s.s_tmp(2)}], {utility_log2(self.data_byte)}")
        self._emit(f"v_add_co_u32 v[{v.v_addr_a()}], vcc_lo, s[{s.s_tmp(2)}], v[{v.v_addr_a_base()}]")
        self._emit(f"v_add_co_ci_u32 v[{v.v_addr_a(1)}], vcc_lo, 0, v[{v.v_addr_a_base(1)}], vcc_lo")
        self._emit_empty_line()

        # ---- B's initial gather (reads s_gemm_k_wg_off live, already updated above) ----
        self._emit_b_gather(None)
        self._emit_empty_line()

        self._emit(self.global_load_a_functor()())
        self._emit(self.global_load_b_functor()())
        self._emit_empty_line()

        self.emit_kernel_fma_main_loop()
        self._emit_empty_line()

        # ---- epilogue store this shard's partial sum (iy=ix=0 always -- nxe==0 -- so the
        # per-tap column offset is always 0, but the shard-offset has been computed into
        # s_p_out_tap above for both the non-atomic wrw_reduction_kernel path and the
        # atomic path). ----
        self._emit(self.coalescing_store(v.v_c.label, v.v_gemm_im(), v.v_gemm_in(), s.s_p_out_tap.label, s.s_wei_row_c.label, v.v_addr_out(), v.v_addr_out(1), s.s_tmp(), v.v_tid(), v.v_c(),
                s.s_block_m_off(), s.s_block_n_off(),
                s.s_gemm_m.label if self.tunable.wmma_m_tail else None, v.v_m_tail_row() if self.tunable.wmma_m_tail else None,
                s.s_gemm_n.label if self.tunable.wmma_n_tail else None, v.v_n_tail_col() if self.tunable.wmma_n_tail else None,
                # Phase 66: always passed now (not just under wmma_n_tail) --
                # direct_store's outer-loop address hoist reuses this scratch SGPR
                # too; mutually exclusive with the LDS-reshuffle path's own use of
                # it, so sharing the slot is safe.
                s.s_tmp(1)))
        self._emit(f"s_wait_storecnt 0x0")
        self._emit_empty_line()

        self._emit_front(f"{label_skip}:")
        self._emit(f"s_add_u32 s[{s.s_streamk_iter()}], s[{s.s_streamk_iter()}], 1")
        self._emit(f"s_cmp_lt_u32 s[{s.s_streamk_iter()}], s[{s.s_streamk_max_iters()}]")
        self._emit(f"s_cbranch_scc1 {label_loop}")
        self._emit_empty_line()

    def emit_kernel_body(self):
        self.emit_kernel_prologue()
        if self.tunable.wrw_streamk:
            self.emit_kernel_streamk_loop()
        else:
            self.emit_kernel_tap_loop()
