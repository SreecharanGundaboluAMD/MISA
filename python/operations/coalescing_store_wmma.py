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
        # when set, this thread's macro tile is only a PARTIAL sum over a slice of the
        # reduction (K) axis -- other workgroups hold the other slices and atomically add
        # into the same output elements, so the store must accumulate (global_atomic_add_f32)
        # rather than overwrite (global_store_dword). See gemm_k_global_split in
        # docs/gfx1250_wmma_layout.md -- correct for every precision because the WMMA output
        # buffer is always allocated fp32 regardless of the tunable's nominal precision
        # (conv_driver.cpp's is_wmma dtype_alloc_byte override), so there is no fp16/bf16
        # atomic-add precision concern and no workspace/cast step needed.
        self.gemm_k_global_split = False


class igemm_coalescing_store_wmma_t(mc_base_t):
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        assert type(ctrl) is ctrl_coalescing_store_wmma_t
        self.ctrl = ctrl

    def __call__(self, v_c, v_gemm_im, v_gemm_in, s_p_out, s_gemm_m_stride, v_tmp1, v_tmp2, s_tmp1):
        '''
        v_gemm_im/v_gemm_in: this thread's base (row, col) within the macro tile, from
            igemm_wmma_mapping_t.get_gemm_index_for_dst_matrix (row/col of wave_repeat
            iteration (0,0), vgpr index 0). Byte-address computation and the global
            block offset are left to the caller (this emits only the intra-macro-tile
            part) -- v_tmp1/v_tmp2 need 1 scratch VGPR each, s_tmp1 needs 1 scratch SGPR.
        s_gemm_m_stride: row stride of the output tensor, in elements (typically
            gemm_n, i.e. the full N extent, not just macro_tile_n).

        Emits one global_store_dword/global_atomic_add_f32 per accumulator element --
        num_v_c * wave_repeat_m * wave_repeat_n per thread, no LDS traffic. Column offset
        within a wave_repeat_m row (0/64/128/192 bytes for wave_tile_n=16) is folded into
        the store's immediate offset instead of a per-store address add; consecutive rows
        within a wave_repeat_m tile are reached by a precomputed byte row-stride add
        instead of re-deriving the address with a (quarter-rate) multiply.

        v_tmp1/v_tmp2 ping-pong as the "current row" / "next row" address: the next row's
        address is computed into the register NOT currently being read by this row's
        stores/atomics, so that add has no data dependency on the in-flight stores at all
        (different register) and can be issued/scheduled independently of them, instead of
        serializing through a single reused address register. Neither instruction blocks
        on atomic completion either way (the wave doesn't wait for the RMW round-trip
        merely by issuing the next instruction) -- this only removes the same-register
        WAR hazard between "advance the address" and "the previous stores/atomics that
        just read it", which otherwise chains all `wave_repeat_m * num_v_c` row addresses
        into one serial dependency through a single register.
        '''
        ctrl = self.ctrl
        cxm = ctrl.cxm
        inst_wmma = cxm.inst_wmma
        assert cxm.wave_tile_m == 16 and cxm.wave_tile_n == 16

        with self._deferred_context():
            self._emit(f"; wmma direct store epilogue, {cxm.wave_repeat_m}x{cxm.wave_repeat_n} tiles, "
                       f"{inst_wmma.num_v_c} rows/tile")
            self._emit(f"s_lshl_b32 s[{s_tmp1}], s[{s_gemm_m_stride}], 2   ; row-to-row byte stride")
            for i_rm in range(cxm.wave_repeat_m):
                row_off = i_rm * cxm.wave_tile_m
                cur, nxt = v_tmp1, v_tmp2
                # cur = byte address of (row_off, col 0), i.e. row = v_gemm_im + row_off
                self._emit(f"v_add_u32 v[{cur}], {row_off}, v[{v_gemm_im}]" if row_off != 0 else f"v_mov_b32 v[{cur}], v[{v_gemm_im}]")
                self._emit(f"v_mul_lo_u32 v[{cur}], s[{s_gemm_m_stride}], v[{cur}]")
                self._emit(f"v_add_u32 v[{cur}], v[{v_gemm_in}], v[{cur}]")
                self._emit(f"v_lshlrev_b32 v[{cur}], 2, v[{cur}]  ; byte address, row {row_off}, col 0")
                for j in range(inst_wmma.num_v_c):
                    if j != inst_wmma.num_v_c - 1:
                        self._emit(f"v_add_u32 v[{nxt}], v[{cur}], s[{s_tmp1}]   ; precompute row {row_off + j + 1} address")
                    for i_rn in range(cxm.wave_repeat_n):
                        c_index = (i_rm * cxm.wave_repeat_n + i_rn) * inst_wmma.num_v_c + j
                        col_off = i_rn * cxm.wave_tile_n * 4
                        offset_str = f" offset:{col_off}" if col_off != 0 else ""
                        inst = "global_atomic_add_f32" if ctrl.gemm_k_global_split else "global_store_dword"
                        # scope:SCOPE_SYS is load-bearing, not decoration: a bare
                        # global_atomic_add_f32 defaults to a narrower (CU/WGP-local) cache
                        # scope on gfx1250, which silently drops updates when the two
                        # accumulating workgroups land on different compute units --
                        # confirmed on hardware (small tiles, which pack more workgroups per
                        # CU, "happened" to pass without it; large tiles, spread across more
                        # CUs, lost updates). See docs/gfx1250_wmma_layout.md.
                        scope_str = " scope:SCOPE_SYS" if ctrl.gemm_k_global_split else ""
                        self._emit(f"{inst} v[{cur}], v[{v_c}+{c_index}], s[{s_p_out}:{s_p_out}+1]{offset_str}{scope_str}")
                    cur, nxt = nxt, cur
            self._emit_empty_line()
        return self._get_deferred()
