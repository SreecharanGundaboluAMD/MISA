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
from .utility import *
from .wmma import *

class ctrl_wmma_mapping_t(object):
    '''
    wmma mapping for gfx1250. Unlike ctrl_xdlops_mapping_t there is no AGPR-lanegroup
    sub-structure to model: a single v_wmma_* instruction already computes the entire
    wave_tile_m x wave_tile_n (16x16) output tile cooperatively across all 32 lanes of
    one wave (wave32 is mandatory here, not a software tiling choice like gfx1030's
    dlops/lanegroup path). The physical lane/vgpr -> (row,col,k) formulas are empirically
    verified on real gfx1250 hardware; see docs/gfx1250_wmma_layout.md.

    A macro tile is assembled from wave-tiles via a waves_per_m x waves_per_n grid of
    waves, further repeated wave_repeat_m x wave_repeat_n times per wave (no wave_step
    concept -- WMMA's __call__ has no cbsz/abid/blgp-style sub-addressing to drive one).
    '''
    def __init__(self, macro_tile_m, macro_tile_n, wave_tile_m, wave_tile_n, waves, wave_repeat_m, wave_repeat_n, inst_wmma):
        assert wave_tile_m == inst_wmma.m and wave_tile_n == inst_wmma.n
        self.macro_tile_m = macro_tile_m
        self.macro_tile_n = macro_tile_n
        self.wave_tile_m = wave_tile_m
        self.wave_tile_n = wave_tile_n
        self.waves = waves
        self.wave_repeat_m = wave_repeat_m
        self.wave_repeat_n = wave_repeat_n
        self.inst_wmma = inst_wmma

    def wave_size(self):
        # gfx1250 WMMA genuinely requires wave32 -- not computed from tiling factors
        # the way ctrl_dotx_mapping_t.wave_size() is, since there is no wave64-compat
        # ambiguity to parameterize away here.
        return 32

    def block_size(self):
        return self.waves * self.wave_size()

    def waves_per_m(self):
        return self.macro_tile_m // (self.wave_repeat_m * self.wave_tile_m)

    def waves_per_n(self):
        return self.macro_tile_n // (self.wave_repeat_n * self.wave_tile_n)

    def macro_tile_validate(self):
        assert self.macro_tile_m == self.wave_tile_m * self.wave_repeat_m * self.waves_per_m()
        assert self.macro_tile_n == self.wave_tile_n * self.wave_repeat_n * self.waves_per_n()
        assert self.waves == self.waves_per_m() * self.waves_per_n()

    def acc_c_per_thread_m(self):
        # each instruction call produces inst_wmma.num_v_c VGPRs already covering the
        # full wave_tile_m rows for this thread's fixed column; wave_repeat_m stacks more
        # such calls to cover further rows of the macro tile.
        return self.wave_repeat_m, self.inst_wmma.num_v_c

    def acc_c_per_thread_n(self):
        return self.wave_repeat_n, 1

    def total_acc_c(self):
        return self.wave_repeat_m * self.wave_repeat_n * self.inst_wmma.num_v_c


class igemm_wmma_mapping_t(mc_base_t):
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        assert type(ctrl) is ctrl_wmma_mapping_t
        self.ctrl = ctrl

    def get_gemm_index_for_src_matrix(self, v_gemm_in, v_gemm_im, v_thread_id, v_tmp2, **options):
        '''
        compute this thread's (im, ik) / (in, ik) position for reading the A / B operand
        out of an LDS tile laid out row-major as [macro_tile_m or macro_tile_n][gemm_k],
        i.e. row-contiguous in k (k innermost). v_gemm_im/v_gemm_in are the row indices
        (element units, not yet scaled by the k-row stride); the k-half selection
        (lane/16 -> upper/lower 16 of the 32 k-values) is a fixed 32-byte (16 fp16)
        additional byte offset within that row, applied by the caller since it is
        identical for every thread and every wave-tile (not part of the wave/macro-tile
        indexing hierarchy at all -- see docs/gfx1250_wmma_layout.md).

        Only wave_tile_m == wave_tile_n == 16 (inst_wmma.m == inst_wmma.n == 16) is
        supported, matching the only instruction wired up so far.
        '''
        ctrl = self.ctrl
        assert ctrl.wave_tile_m == 16 and ctrl.wave_tile_n == 16
        with self._deferred_context():
            self._emit(f"; wmma mapping, get source matrix gemm index (row-index only; k-half byte offset is added by the caller)")
            self._emit(f"v_and_b32 v[{v_gemm_in}], 15, v[{v_thread_id}]          ; lane % 16 -> wave-tile column/row index")
            self._emit(f"v_and_b32 v[{v_gemm_im}], 15, v[{v_thread_id}]          ; lane % 16 -> wave-tile column/row index")
            self._emit(f"v_lshrrev_b32 v[{v_tmp2}], 5, v[{v_thread_id}]          ; wave id")
            if ctrl.waves_per_n() != 1:
                self._emit(f"v_and_b32 v[{v_tmp2}+1], {ctrl.waves_per_n() - 1}, v[{v_tmp2}]  ; waves_per_n index")
                self._emit(f"v_lshl_or_b32 v[{v_gemm_in}], v[{v_tmp2}+1], {utility_log2(ctrl.wave_tile_n * ctrl.wave_repeat_n)}, v[{v_gemm_in}]")
            if ctrl.waves_per_m() != 1:
                self._emit(f"v_lshrrev_b32 v[{v_tmp2}+2], {utility_log2(ctrl.waves_per_n())}, v[{v_tmp2}]  ; waves_per_m index")
                self._emit(f"v_lshl_or_b32 v[{v_gemm_im}], v[{v_tmp2}+2], {utility_log2(ctrl.wave_tile_m * ctrl.wave_repeat_m)}, v[{v_gemm_im}]")
            self._emit_empty_line()
        return self._get_deferred()

    def get_gemm_index_for_src_matrix_transposed(self, v_byte_offset, v_thread_id, v_tmp2, row_pitch_bytes, elem_bytes, side, **options):
        '''
        Sibling to get_gemm_index_for_src_matrix, for an operand whose LDS storage is the
        TRANSPOSE of what that function assumes: [gemm_k rows][macro_tile_m or n contiguous]
        (k-major, row-contiguous in the M/N direction) instead of [M or N][gemm_k] (row-major,
        contiguous in k). This is the case for e.g. bwd's weight operand: physically stored
        [K_out][C_in] (K_out-major, C_in-contiguous), while GEMM_K=K_out and GEMM_N=C_in --
        i.e. the WMMA operand's contraction dimension (k) is the memory-contiguous one is
        false here, unlike the untransposed case.

        Unlike get_gemm_index_for_src_matrix (which returns an ELEMENT-unit row/col index and
        leaves final byte-scaling and the k-half offset to the caller), this function returns
        a single, ready-to-use BYTE offset directly, because the transposed case's k-half
        contribution is itself a large, row_pitch_bytes-scaled jump (not a small fixed 32-byte
        addition), so it cannot be cleanly separated from byte-scaling the way the untransposed
        formula does. The caller still adds two purely compile-time-constant deltas via the
        `ds_read ... offset:` immediate: `i_rn_or_im * wave_tile_n_or_m * elem_bytes` for the
        wave_repeat column/row, and `(a*2+s) * row_pitch_bytes` for which of the 8 vgpr/2 half
        sub-elements this read is for -- see igemm_bwd_gtc_wmma_nhwc.py's shared_load_b_functor
        for a concrete worked example.

        side: 'm' or 'n' -- which wave axis (waves_per_m/wave_tile_m vs waves_per_n/wave_tile_n)
        this operand's non-contraction dimension belongs to.
        '''
        ctrl = self.ctrl
        assert ctrl.wave_tile_m == 16 and ctrl.wave_tile_n == 16
        assert side in ('m', 'n')
        waves_per_side  = ctrl.waves_per_m() if side == 'm' else ctrl.waves_per_n()
        wave_tile_side  = ctrl.wave_tile_m if side == 'm' else ctrl.wave_tile_n
        wave_repeat_side = ctrl.wave_repeat_m if side == 'm' else ctrl.wave_repeat_n
        with self._deferred_context():
            self._emit(f"; wmma mapping, get transposed source matrix byte offset (side={side})")
            self._emit(f"v_and_b32 v[{v_byte_offset}], 15, v[{v_thread_id}]   ; lane % 16 -> local index")
            self._emit(f"v_lshrrev_b32 v[{v_tmp2}], 5, v[{v_thread_id}]       ; wave id")
            if waves_per_side != 1:
                # wave_id's bits are n-major-low/m-major-high (wave_id = m_idx*waves_per_n +
                # n_idx), matching get_gemm_index_for_dst_matrix's convention -- side='n' wants
                # the low bits (mask), side='m' wants the high bits (shift right by
                # log2(waves_per_n)), NOT the same mask reused for both.
                if side == 'n':
                    self._emit(f"v_and_b32 v[{v_tmp2}+1], {waves_per_side - 1}, v[{v_tmp2}]  ; waves_per_n index")
                else:
                    self._emit(f"v_lshrrev_b32 v[{v_tmp2}+1], {utility_log2(ctrl.waves_per_n())}, v[{v_tmp2}]  ; waves_per_m index")
                self._emit(f"v_lshl_or_b32 v[{v_byte_offset}], v[{v_tmp2}+1], {utility_log2(wave_tile_side * wave_repeat_side)}, v[{v_byte_offset}]")
            self._emit(f"v_lshlrev_b32 v[{v_byte_offset}], {utility_log2(elem_bytes)}, v[{v_byte_offset}]   ; local index * elem_bytes")
            self._emit(f"v_lshrrev_b32 v[{v_tmp2}], 4, v[{v_thread_id}]")
            self._emit(f"v_and_b32 v[{v_tmp2}], 1, v[{v_tmp2}]                ; k_half (0/1)")
            # k_half spans inst_wmma.k/2 rows of the transposed [K][M or N] LDS tile (16 for
            # fp16/bf16's K=32, 32 for int8's K=64 -- NOT a hardcoded 16, since the A/B operand
            # layout's "upper vs lower 16 lanes" split always covers half of K regardless of
            # how many k-values a lane's vgprs cover per half. See docs/gfx1250_wmma_layout.md.
            half_k = ctrl.inst_wmma.k // 2
            self._emit(f"v_lshlrev_b32 v[{v_tmp2}], {utility_log2(half_k * row_pitch_bytes)}, v[{v_tmp2}]  ; k_half * {half_k} * row_pitch_bytes")
            self._emit(f"v_add_u32 v[{v_byte_offset}], v[{v_tmp2}], v[{v_byte_offset}]")
            self._emit_empty_line()
        return self._get_deferred()

    def get_gemm_index_for_dst_matrix(self, v_gemm_in, v_gemm_im, v_thread_id, v_tmp2):
        '''
        compute this thread's base (im, in) position in the macro-tile output, i.e. the
        (row, col) of vgpr-index 0 of the C/D operand for wave_repeat iteration (0,0).
        Per-instruction-call vgpr j (0..inst_wmma.num_v_c-1) adds j to the row (see
        docs/gfx1250_wmma_layout.md: row = (lane/16)*8 + j), and each further
        wave_repeat_m/wave_repeat_n iteration adds a multiple of wave_tile_m/wave_tile_n
        -- both handled by the caller (coalescing_store_wmma.py), not here.
        '''
        ctrl = self.ctrl
        assert ctrl.wave_tile_m == 16 and ctrl.wave_tile_n == 16
        with self._deferred_context():
            self._emit(f"; wmma mapping, get dst matrix gemm index")
            self._emit(f"v_and_b32 v[{v_gemm_in}], 15, v[{v_thread_id}]          ; lane % 16 -> column")
            self._emit(f"v_lshrrev_b32 v[{v_gemm_im}], 4, v[{v_thread_id}]")
            self._emit(f"v_and_b32 v[{v_gemm_im}], 1, v[{v_gemm_im}]")
            self._emit(f"v_lshlrev_b32 v[{v_gemm_im}], 3, v[{v_gemm_im}]         ; (lane/16)*8 -> row base")
            self._emit(f"v_lshrrev_b32 v[{v_tmp2}], 5, v[{v_thread_id}]          ; wave id")
            if ctrl.waves_per_n() != 1:
                self._emit(f"v_and_b32 v[{v_tmp2}+1], {ctrl.waves_per_n() - 1}, v[{v_tmp2}]  ; waves_per_n index")
                self._emit(f"v_lshl_or_b32 v[{v_gemm_in}], v[{v_tmp2}+1], {utility_log2(ctrl.wave_tile_n * ctrl.wave_repeat_n)}, v[{v_gemm_in}]")
            if ctrl.waves_per_m() != 1:
                self._emit(f"v_lshrrev_b32 v[{v_tmp2}+2], {utility_log2(ctrl.waves_per_n())}, v[{v_tmp2}]  ; waves_per_m index")
                self._emit(f"v_lshl_or_b32 v[{v_gemm_im}], v[{v_tmp2}+2], {utility_log2(ctrl.wave_tile_m * ctrl.wave_repeat_m)}, v[{v_gemm_im}]")
            self._emit_empty_line()
        return self._get_deferred()


# seeded with exactly one entry per precision for the correctness-first milestones:
# block_size = 4 waves * 32 = 128, macro_tile 128x128, wave_repeat 4x4, waves_per_m=waves_per_n=2
#   macro_tile_m = wave_tile_m(16) * wave_repeat_m(4) * waves_per_m(2) = 128  (same for n)
# fp16/bf16 share the same K=32, 8-VGPR/lane instruction footprint (both verified, see
# docs/gfx1250_wmma_layout.md) so they share the same tile-shape table.
#
# A second, smaller 64x64 entry was added for tile-shape-diversity coverage: convolution
# shapes whose gemm_m/gemm_n aren't a multiple of 128 have no valid 128x128 kernel today even
# when they divide evenly into 64. IMPORTANT constraint discovered while adding this: the fwd/
# bwd/wrw kernels' GLOBAL-LOAD thread mapping (separate from this file's wave/repeat compute
# indexing, which is already fully generic) uses v_tid directly as the per-thread gemm_m/n row
# index within the block -- i.e. it requires block_size == gemm_m_per_block == gemm_n_per_block
# exactly (true for the existing 128x128/block_size=128 entry only by coincidence of both being
# 128). With wave_tile fixed at 16 (only verified shape), this forces an ASYMMETRIC wave grid
# for any smaller square macro-tile: waves_per_m*waves_per_n*32 (=block_size) must equal
# macro_tile_m (=macro_tile_n), which for macro_tile=64 means waves_per_m/n=(2,1) or (1,2), not
# (1,1)(->32x32) or (2,2)(->128x128, the existing shape). Chosen here: waves_per_m=2,
# waves_per_n=1 -> wave_repeat_m=2 (macro_tile_m=16*2*2=64), wave_repeat_n=4
# (macro_tile_n=16*4*1=64), waves=2 (block_size=64). accumulate_c=2*4*8=64 (down from 128),
# accumulate_a=2*8=16, accumulate_b=4*8=32 -- total 112 VGPRs vs 192 for fp16/bf16/int8's
# existing 128x128/4x4 shape, a real if more modest reduction than a naive symmetric 2x2 split
# would have given (which is NOT valid here -- see above).
#
# A 32x32 entry (2026-08-27, bwd occupancy investigation): waves_per_m=waves_per_n=1 (the
# ONLY valid factorization at macro_tile=32 under the same block_size==macro_tile_m==
# macro_tile_n constraint above -- (1,1) is the (1,1)(->32x32) case the 64x64 comment
# calls out as the excluded alternative there) -> wave_repeat_m=wave_repeat_n=2
# (macro_tile=16*2*1=32 each), waves=1 (block_size=32, a single wave, no row_repeat_a/b
# generalization needed at all since block_size already equals both macro_tile dims
# directly -- same structural category as the existing 64x64/128x128 entries, just
# smaller). Motivation: CK's tuned instance for a small-spatial/large-channel bwd shape
# (n=4,c=512,H=W=8,k=256, see docs/gfx1250_wmma_layout.md's Phase 46) uses an MPerBlock=
# NPerBlock=32 tile specifically to maximize workgroup count (occupancy) on gfx1250's
# 256 CUs when gemm_m/gemm_n are individually too small for 64x64 to produce enough
# workgroups -- this entry is the equivalent occupancy lever available to MISA's own
# tiling scheme without inventing new machinery. accumulate_c=2*2*8=32, accumulate_a=
# accumulate_b=2*8=16 -- 64 VGPRs total, half of 64x64's 112.
ctrl_wmma_mapping_table = {
    'fp16': [
        ctrl_wmma_mapping_t(128, 128, 16, 16, 4, 4, 4, v_wmma_f32_16x16x32_f16),
        ctrl_wmma_mapping_t(64,  64,  16, 16, 2, 2, 4, v_wmma_f32_16x16x32_f16),
        # 32x32, single wave (see table-level comment above for the derivation).
        ctrl_wmma_mapping_t(32,  32,  16, 16, 1, 2, 2, v_wmma_f32_16x16x32_f16),
        # Asymmetric first shape (2026-08-25): waves_per_m=2 (128/(4*16)), waves_per_n=1
        # (64/(4*16)), waves=2 -> block_size=64. block_size == gemm_n_per_block(64) already
        # (B needs zero changes), but block_size < gemm_m_per_block(128) -- A's global-load
        # needs the new row_repeat_a=2 generalization (see igemm_fwd_gtc_wmma_nhwc.py). Same
        # accumulate_a/b/c magnitudes as the 128x128 entry (same wave_repeat_m/n=4/4), just a
        # different waves_per_n split.
        ctrl_wmma_mapping_t(128, 64,  16, 16, 2, 4, 4, v_wmma_f32_16x16x32_f16),
        # Mirror shape (2026-08-25): waves_per_m=1, waves_per_n=2, waves=2 -> block_size=64.
        # block_size == gemm_m_per_block(64) already (A needs zero changes), but block_size <
        # gemm_n_per_block(128) -- B's global-load needs the row_repeat_b=2 generalization
        # (see igemm_fwd_gtc_wmma_nhwc.py). B needs no flag/masking (weight is never OOB), so
        # this mirror was materially simpler to implement than the 128x64 entry's A-side work.
        ctrl_wmma_mapping_t(64,  128, 16, 16, 2, 4, 4, v_wmma_f32_16x16x32_f16),
    ],
    # Phase 24 (F16-accumulate WMMA): separate table key (not a field on ctrl_wmma_mapping_t)
    # so the existing f32-accumulate 'fp16' entries stay byte-identical -- the caller
    # (igemm_base.py) picks this key instead of 'fp16' when tunable.wmma_acc_f16=1. Same
    # tile shapes as 'fp16' above (accumulate width doesn't affect tiling), just
    # v_wmma_f16_16x16x32_f16 instead of v_wmma_f32_16x16x32_f16 -- total_acc_c() naturally
    # halves (num_v_c=4 instead of 8) with no other change needed.
    'fp16_f16acc': [
        ctrl_wmma_mapping_t(128, 128, 16, 16, 4, 4, 4, v_wmma_f16_16x16x32_f16),
        ctrl_wmma_mapping_t(64,  64,  16, 16, 2, 2, 4, v_wmma_f16_16x16x32_f16),
        ctrl_wmma_mapping_t(128, 64,  16, 16, 2, 4, 4, v_wmma_f16_16x16x32_f16),
        ctrl_wmma_mapping_t(64,  128, 16, 16, 2, 4, 4, v_wmma_f16_16x16x32_f16),
        # 256x256, Phase 53 (chunked epilogue): a bigger-than-128x128 macro-tile's
        # accumulator (total_acc_c = macro_tile_m*macro_tile_n/block_size) does not fit
        # gfx1250's real, hardware-verified 256-VGPR/wave ceiling at f32-accumulate width
        # (num_v_c=8) for ANY block_size choice that keeps block_size<=macro_tile (the
        # existing row_repeat_a/b precondition -- see Phase 53's correction to Phase 52,
        # which originally over-cited an external "1024 VGPR" figure this project's plain
        # v[N]-addressed assembly can't actually use). f16acc's num_v_c=4 halves it back
        # into range: block_size=256 (waves_per_m=4,waves_per_n=2, chosen so
        # block_size==macro_tile_m==macro_tile_n exactly -- no row_repeat needed at all,
        # the simplest case) gives total_acc_c = 256*256/256 * (4/8) = 128, matching the
        # EXISTING 128x128 f32 entry's own accumulator size exactly. fwd-only -- see the
        # 256x128 entry below for bwd (transposed B requires row_repeat_b==1). Requires
        # `wmma_epilogue_chunked=1` (chunked epilogue, extended to handle f16acc's packed
        # 2-elements-per-register layout) -- see docs/gfx1250_wmma_layout.md's Phase 53.
        ctrl_wmma_mapping_t(256, 256, 16, 16, 8, 4, 8, v_wmma_f16_16x16x32_f16),
        # 256x128, Phase 53 (chunked epilogue, bwd): bwd's B is TRANSPOSED
        # (igemm_bwd_gtc_wmma_nhwc_t's own docstring) and asserts row_repeat_b==1 --
        # gemm_n_per_block must stay == block_size(128, matching the existing 128x128
        # entry's own waves_per_m/n=2/2 split exactly), so only gemm_m_per_block grows
        # (wave_repeat_m 4->8, row_repeat_a=2, already-generalized -- no new global-load
        # mechanism). total_acc_c = 256*128/128 * (4/8) = 128, same budget as fwd's
        # 256x256 above. Requires `wmma_epilogue_chunked=1`.
        ctrl_wmma_mapping_t(256, 128, 16, 16, 4, 8, 4, v_wmma_f16_16x16x32_f16),
    ],
    'bf16': [
        ctrl_wmma_mapping_t(128, 128, 16, 16, 4, 4, 4, v_wmma_f32_16x16x32_bf16),
        ctrl_wmma_mapping_t(64,  64,  16, 16, 2, 2, 4, v_wmma_f32_16x16x32_bf16),
        # 32x32, single wave (see table-level comment above the 'fp16' entry for the derivation).
        ctrl_wmma_mapping_t(32,  32,  16, 16, 1, 2, 2, v_wmma_f32_16x16x32_bf16),
        # Asymmetric shapes (2026-08-25): mechanical port of fp16's 128x64/64x128 entries --
        # the row_repeat mechanism operates purely on gemm_m/n_per_block vs block_size and is
        # already precision-generic (uses self.data_byte throughout), same as the 64x64 port.
        ctrl_wmma_mapping_t(128, 64,  16, 16, 2, 4, 4, v_wmma_f32_16x16x32_bf16),
        ctrl_wmma_mapping_t(64,  128, 16, 16, 2, 4, 4, v_wmma_f32_16x16x32_bf16),
        # 256x256/256x128, Phase 54 (VGPR-MSB): full F32-accumulate versions of the
        # 'bf16_bf16acc' entries below -- Phase 53 needed the packed accumulator
        # (num_v_c=4) to have any chance of fitting the 256-VGPR/wave ceiling, and it
        # STILL didn't fit (284 registers needed, 28 over) until Phase 54 moved v_c
        # into a second, independently-addressed bank via `wmma_acc_high_bank=1`. With
        # v_c in bank 1, full F32 accumulate (total_acc_c = wave_repeat_m*wave_repeat_n*8
        # = 256) fits easily -- no precision tradeoff needed. Same tile shapes as the
        # 'bf16_bf16acc' entries (accumulate width doesn't affect tiling). Requires
        # `wmma_epilogue_chunked=1` (a 256x256 F32-accumulate tile needs 262144 bytes of
        # LDS to stage one-shot, 4x the 64KB ceiling). See
        # docs/gfx1250_wmma_vgpr_msb_wip_status.md's Phase 54 fix.
        ctrl_wmma_mapping_t(256, 256, 16, 16, 8, 4, 8, v_wmma_f32_16x16x32_bf16),
        ctrl_wmma_mapping_t(256, 128, 16, 16, 4, 8, 4, v_wmma_f32_16x16x32_bf16),
    ],
    # Phase 27 (BF16-accumulate WMMA): mirrors 'fp16_f16acc' above exactly -- same tile shapes
    # as 'bf16', just v_wmma_bf16_16x16x32_bf16 instead of v_wmma_f32_16x16x32_bf16.
    'bf16_bf16acc': [
        ctrl_wmma_mapping_t(128, 128, 16, 16, 4, 4, 4, v_wmma_bf16_16x16x32_bf16),
        ctrl_wmma_mapping_t(64,  64,  16, 16, 2, 2, 4, v_wmma_bf16_16x16x32_bf16),
        ctrl_wmma_mapping_t(128, 64,  16, 16, 2, 4, 4, v_wmma_bf16_16x16x32_bf16),
        ctrl_wmma_mapping_t(64,  128, 16, 16, 2, 4, 4, v_wmma_bf16_16x16x32_bf16),
        # 256x256/256x128, Phase 53 (chunked epilogue): see 'fp16_f16acc''s identical
        # entries for the full derivation.
        ctrl_wmma_mapping_t(256, 256, 16, 16, 8, 4, 8, v_wmma_bf16_16x16x32_bf16),
        ctrl_wmma_mapping_t(256, 128, 16, 16, 4, 8, 4, v_wmma_bf16_16x16x32_bf16),
    ],
    # int8: K=64 (not 32), but num_v_a/num_v_b/num_v_c are still 8/8/8 (same as fp16/bf16 --
    # only elements/dword differs: 4 int8/dword vs 2 fp16/dword), so the same 128x128 macro
    # tile / 4x4 wave_repeat shape carries over unchanged. Verified via hardware round-trip
    # probe (/tmp/wmma_probe/probe_int8.s, host_int8.cpp), see docs/gfx1250_wmma_layout.md.
    'int8': [
        ctrl_wmma_mapping_t(128, 128, 16, 16, 4, 4, 4, v_wmma_i32_16x16x64_iu8),
        ctrl_wmma_mapping_t(64,  64,  16, 16, 2, 2, 4, v_wmma_i32_16x16x64_iu8),
        # Asymmetric shapes (2026-08-25): mechanical port, see fp16/bf16's entries above.
        ctrl_wmma_mapping_t(128, 64,  16, 16, 2, 4, 4, v_wmma_i32_16x16x64_iu8),
        ctrl_wmma_mapping_t(64,  128, 16, 16, 2, 4, 4, v_wmma_i32_16x16x64_iu8),
    ],
    # fp32: K=4 (much shorter than the others), num_v_a=num_v_b=2 (not 8 -- no packing at all,
    # 1 fp32/dword, unlike fp16's 2/dword or int8's 4/dword). D operand (num_v_c=8) carries
    # over unchanged, confirmed via hardware round-trip probe (/tmp/wmma_probe/probe_fp32.s,
    # host_fp32.cpp, 6 random seeds), see docs/gfx1250_wmma_layout.md. Same 128x128 macro tile
    # / 4x4 wave_repeat shape as every other precision -- only gemm_k_per_block (forced to 4,
    # matching inst_wmma.k) and the resulting byte-width-per-thread-row differ.
    'fp32': [
        ctrl_wmma_mapping_t(128, 128, 16, 16, 4, 4, 4, v_wmma_f32_16x16x4_f32),
        ctrl_wmma_mapping_t(64,  64,  16, 16, 2, 2, 4, v_wmma_f32_16x16x4_f32),
        # 32x32, single wave (see table-level comment above the 'fp16' entry for the derivation).
        ctrl_wmma_mapping_t(32,  32,  16, 16, 1, 2, 2, v_wmma_f32_16x16x4_f32),
        # Asymmetric shapes (2026-08-25): mechanical port, see fp16/bf16's entries above.
        ctrl_wmma_mapping_t(128, 64,  16, 16, 2, 4, 4, v_wmma_f32_16x16x4_f32),
        ctrl_wmma_mapping_t(64,  128, 16, 16, 2, 4, 4, v_wmma_f32_16x16x4_f32),
    ],
}

def get_ctrl_wmma_mapping_from_wave_tile(macro_tile_m, macro_tile_n, wave_tile_m, wave_tile_n, wave_repeat_m, wave_repeat_n, waves, precision):
    assert precision in ctrl_wmma_mapping_table, f'wmma mapping table not yet populated for precision:{precision}'
    target_wmma_tiling = [c for c in ctrl_wmma_mapping_table[precision] if
                            c.macro_tile_m == macro_tile_m and c.macro_tile_n == macro_tile_n and
                            c.wave_tile_m == wave_tile_m and c.wave_tile_n == wave_tile_n and
                            c.wave_repeat_m == wave_repeat_m and c.wave_repeat_n == wave_repeat_n and
                            c.waves == waves]
    assert len(target_wmma_tiling) != 0, f"not found for precision:{precision}, macro_tile:{macro_tile_m}x{macro_tile_n}, wave_tile:{wave_tile_m}x{wave_tile_n}, " + \
            f"wave_repeat:{wave_repeat_m}x{wave_repeat_n}, waves:{waves}"
    return target_wmma_tiling[0]
