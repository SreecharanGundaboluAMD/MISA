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


# seeded with exactly one entry for the fwd/nhwc/fp16 correctness-first milestone:
# block_size = 4 waves * 32 = 128, macro_tile 128x128, wave_repeat 2x2 -> waves_per_m=waves_per_n=4?
# waves = waves_per_m*waves_per_n must equal 4: pick waves_per_m=2, waves_per_n=2, wave_repeat_m=wave_repeat_n=4
#   macro_tile_m = wave_tile_m(16) * wave_repeat_m(4) * waves_per_m(2) = 128  (same for n)
ctrl_wmma_mapping_fp16 = [
    ctrl_wmma_mapping_t(128, 128, 16, 16, 4, 4, 4, v_wmma_f32_16x16x32_f16),
]

def get_ctrl_wmma_mapping_from_wave_tile(macro_tile_m, macro_tile_n, wave_tile_m, wave_tile_n, wave_repeat_m, wave_repeat_n, waves, precision):
    assert precision == 'fp16', f'wmma mapping table not yet populated for precision:{precision}'
    target_wmma_tiling = [c for c in ctrl_wmma_mapping_fp16 if
                            c.macro_tile_m == macro_tile_m and c.macro_tile_n == macro_tile_n and
                            c.wave_tile_m == wave_tile_m and c.wave_tile_n == wave_tile_n and
                            c.wave_repeat_m == wave_repeat_m and c.wave_repeat_n == wave_repeat_n and
                            c.waves == waves]
    assert len(target_wmma_tiling) != 0, f"not found for macro_tile:{macro_tile_m}x{macro_tile_n}, wave_tile:{wave_tile_m}x{wave_tile_n}, " + \
            f"wave_repeat:{wave_repeat_m}x{wave_repeat_n}, waves:{waves}"
    return target_wmma_tiling[0]
