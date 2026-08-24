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

from .global_memory import *
from .wmma_mapping import *

class ctrl_coalescing_store_wmma_t(object):
    '''
    Direct (non-LDS-coalesced) epilogue for gfx1250 WMMA. Deliberately much simpler
    than coalescing_store_dotx.py/coalescing_store.py's xdlops classes: no
    "coalescing_groups" LDS-reshuffle pass, no vector_store_m/n folding, and (unlike
    the xdlops epilogue) no accvgpr_unified field at all -- WMMA accumulates directly
    in v_c (plain VGPR), so there is nothing to move out of an AGPR file first.

    This trades away some store coalescing/vectorization tuning for a much smaller,
    more auditable epilogue, appropriate for a correctness-first milestone (see
    docs/gfx1250_wmma_layout.md). It is still reasonably store-friendly in practice:
    per docs/gfx1250_wmma_layout.md, 16 consecutive lanes share one output row and
    cover 16 consecutive columns for a fixed accumulator vgpr index, so a per-lane
    scalar global_store_dword is already contiguous across each 16-lane half-wave.
    '''
    def __init__(self):
        self.cxm = None              # ctrl_wmma_mapping_t
        self.block_size = 256
        self.precision = 'fp32'      # accumulator precision as stored to global memory


class igemm_coalescing_store_wmma_t(mc_base_t):
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        assert type(ctrl) is ctrl_coalescing_store_wmma_t
        self.ctrl = ctrl

    def __call__(self, v_c, v_gemm_im, v_gemm_in, s_p_out, s_gemm_m_stride, v_tmp2):
        '''
        v_gemm_im/v_gemm_in: this thread's base (row, col) within the macro tile, from
            igemm_wmma_mapping_t.get_gemm_index_for_dst_matrix (row/col of wave_repeat
            iteration (0,0), vgpr index 0). Byte-address computation and the global
            block offset are left to the caller (this emits only the intra-macro-tile
            part) -- v_tmp2 needs 2 scratch VGPRs.
        s_gemm_m_stride: row stride of the output tensor, in elements (typically
            gemm_n, i.e. the full N extent, not just macro_tile_n).

        Emits one global_store_dword per accumulator element -- num_v_c * wave_repeat_m
        * wave_repeat_n stores per thread, no LDS traffic.
        '''
        ctrl = self.ctrl
        cxm = ctrl.cxm
        inst_wmma = cxm.inst_wmma
        assert cxm.wave_tile_m == 16 and cxm.wave_tile_n == 16

        with self._deferred_context():
            self._emit(f"; wmma direct store epilogue, {cxm.wave_repeat_m}x{cxm.wave_repeat_n} tiles, "
                       f"{inst_wmma.num_v_c} rows/tile")
            for i_rm in range(cxm.wave_repeat_m):
                for i_rn in range(cxm.wave_repeat_n):
                    c_index = (i_rm * cxm.wave_repeat_n + i_rn) * inst_wmma.num_v_c
                    for j in range(inst_wmma.num_v_c):
                        # row = v_gemm_im + i_rm*wave_tile_m + j ; col = v_gemm_in + i_rn*wave_tile_n
                        row_off = i_rm * cxm.wave_tile_m + j
                        col_off = i_rn * cxm.wave_tile_n
                        self._emit(f"v_add_u32 v[{v_tmp2}], {row_off}, v[{v_gemm_im}]" if row_off != 0 else f"v_mov_b32 v[{v_tmp2}], v[{v_gemm_im}]")
                        self._emit(f"v_mul_lo_u32 v[{v_tmp2}], s[{s_gemm_m_stride}], v[{v_tmp2}]")
                        if col_off != 0:
                            self._emit(f"v_add_u32 v[{v_tmp2}], {col_off}, v[{v_tmp2}]")
                        self._emit(f"v_add_u32 v[{v_tmp2}], v[{v_gemm_in}], v[{v_tmp2}]")
                        self._emit(f"v_lshlrev_b32 v[{v_tmp2}], 2, v[{v_tmp2}]  ; *4 bytes (fp32 out)")
                        self._emit(f"global_store_dword v[{v_tmp2}], v[{v_c}+{c_index + j}], s[{s_p_out}:{s_p_out}+1]")
            self._emit_empty_line()
        return self._get_deferred()
