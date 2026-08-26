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
        # atomic-add precision concern and no workspace/cast step needed. Atomic-add has no
        # wide/packed fp32 variant on this ISA (confirmed against the CDNA5 ISA doc), so this
        # path stays scalar regardless of vector_write_out -- see docs/gfx1250_wmma_layout.md.
        self.gemm_k_global_split = False
        # non-atomic path only: target width (in elements) for the vectorized global store
        # after the LDS reshuffle. 4 = global_store_dwordx4. See docs/gfx1250_wmma_layout.md's
        # LDS-reshuffle phase for the derivation.
        self.vector_write_out = 4

        # Phase 23 (ISA-driven epilogue tuning): atomic path only. Default 'SCOPE_SYS' =
        # today's exact behavior. 'SCOPE_DEV' resolves within the device's L2 instead of
        # forcing a full system-level flush/invalidate -- sufficient since gemm_k_global_split's
        # contending workgroups are always on the same device. See docs/gfx1250_wmma_layout.md.
        self.atomic_scope = 'SCOPE_SYS'
        # atomic path only. 0 (default) = regular atomic. 1 = TH[2] cascading/deferred-scope
        # atomic. The `th:TH_ATOMIC_CASCADE_RT` ENCODING was confirmed correct via an
        # `llvm-mc -show-encoding` round-trip probe (Phase 23) -- but using it CONFIRMED HANGS
        # ON REAL HARDWARE: this kernel's `s_wait_storecnt 0x0` before `s_endpgm` never
        # completes, because the doc states a cascading atomic's full scope/completion is only
        # realized at "a subsequent release... of a matching or higher scope", which this
        # kernel never issues. Hard-blocked in igemm_base.py's tunable read (`assert not
        # self.atomic_cascade`) until a companion release mechanism is designed and added --
        # do not wire this up without that. See docs/gfx1250_wmma_layout.md's Phase 23.
        self.atomic_cascade = 0
        # the confirmed `th:` identifier to emit when atomic_cascade=1 -- a string, not a raw
        # number (llvm-mc requires a symbolic th value, rejects numeric immediates).
        self.atomic_th = 'TH_ATOMIC_CASCADE_RT'
        # non-atomic path only. 0 (default) = today's unpadded tile-linear LDS layout. 1 =
        # pad the row stride by one element to break a bank-conflict periodicity (macro_tile_n
        # is always a multiple of 64, so the unpadded layout puts every row of a given column
        # in the same LDS bank -- see docs/gfx1250_wmma_layout.md).
        self.epilogue_lds_pad = 0


class igemm_coalescing_store_wmma_t(mc_base_t):
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        assert type(ctrl) is ctrl_coalescing_store_wmma_t
        self.ctrl = ctrl

    def __call__(self, v_c, v_gemm_im, v_gemm_in, s_p_out, s_gemm_m_stride, v_tmp1, v_tmp2, s_tmp1, v_tid=None, v_gather=None, s_block_m_off=None, s_block_n_off=None):
        '''
        v_gemm_im/v_gemm_in: this thread's base (row, col) within the macro tile, from
            igemm_wmma_mapping_t.get_gemm_index_for_dst_matrix (row/col of wave_repeat
            iteration (0,0), vgpr index 0). Byte-address computation and the global
            block offset are left to the caller (this emits only the intra-macro-tile
            part) -- v_tmp1/v_tmp2 need 1 scratch VGPR each, s_tmp1 needs 1 scratch SGPR.
        s_gemm_m_stride: row stride of the output tensor, in elements (typically
            gemm_n, i.e. the full N extent, not just macro_tile_n).
        v_tid/v_gather: only needed for the non-atomic (LDS-reshuffle) path -- v_tid is the
            kernel's persistent flat thread id, v_gather is a vector_write_out-wide scratch
            VGPR range (unused by the atomic path).

        ctrl.gemm_k_global_split selects between two structurally different epilogues:

        Atomic path (gemm_k_global_split=True): unchanged from before -- one
        global_atomic_add_f32 per accumulator element, no LDS traffic. There is no wide/
        packed fp32 atomic-add on this ISA (confirmed against the CDNA5 ISA doc), so this
        can't be vectorized; v_tmp1/v_tmp2 ping-pong as the "current row"/"next row"
        address so the row-advance add has no data dependency on the in-flight atomics
        (different register), instead of serializing all row addresses through one
        register. Column offset is folded into the atomic's immediate offset.

        Non-atomic path (gemm_k_global_split=False): LDS-reshuffle coalescing store, see
        docs/gfx1250_wmma_layout.md's LDS-reshuffle phase for the full derivation. A single
        lane only ever owns one column (per docs/gfx1250_wmma_layout.md: col = lane % 16,
        fixed across every accumulator index), so no per-lane vectorized store is possible
        directly -- stage the whole macro-tile through LDS in true tile-linear order
        (scatter, one ds_write_b32 per element, same addressing style as the atomic path's
        row/col derivation just retargeted to LDS with shifts instead of a stride multiply),
        barrier, then every lane reads back `vector_write_out` contiguous elements via a
        flat tid-indexed mapping (gather) and issues one global_store_dwordN -- 4x fewer
        global stores and address multiplies than the direct/atomic path, using cheap
        on-chip LDS traffic to get there.
        '''
        ctrl = self.ctrl
        cxm = ctrl.cxm
        inst_wmma = cxm.inst_wmma
        assert cxm.wave_tile_m == 16 and cxm.wave_tile_n == 16

        with self._deferred_context():
            if ctrl.gemm_k_global_split:
                self._emit(f"; wmma direct atomic-add epilogue, {cxm.wave_repeat_m}x{cxm.wave_repeat_n} tiles, "
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
                            # scope:SCOPE_SYS (or SCOPE_DEV, Phase 23) is load-bearing, not
                            # decoration: a bare global_atomic_add_f32 defaults to a narrower
                            # (CU/WGP-local) cache scope on gfx1250, which silently drops
                            # updates when the two accumulating workgroups land on different
                            # compute units -- confirmed on hardware. See
                            # docs/gfx1250_wmma_layout.md. th:{th} (Phase 23, optional) marks
                            # a cascading/deferred-scope atomic -- see ctrl.atomic_cascade.
                            th_str = f" th:{ctrl.atomic_th}" if ctrl.atomic_cascade else ""
                            self._emit(f"global_atomic_add_f32 v[{cur}], v[{v_c}+{c_index}], s[{s_p_out}:{s_p_out}+1]{offset_str} scope:{ctrl.atomic_scope}{th_str}")
                        cur, nxt = nxt, cur
                self._emit_empty_line()
            else:
                # ---- non-atomic: LDS-reshuffle coalescing store ----
                assert v_tid is not None and v_gather is not None and s_block_m_off is not None and s_block_n_off is not None
                vwo = ctrl.vector_write_out
                macro_tile_m = cxm.macro_tile_m
                macro_tile_n = cxm.macro_tile_n
                assert macro_tile_n % vwo == 0, f"macro_tile_n:{macro_tile_n} not divisible by vector_write_out:{vwo}"
                total_elements = macro_tile_m * macro_tile_n
                elements_per_thread = total_elements // ctrl.block_size
                assert elements_per_thread % vwo == 0, f"elements_per_thread:{elements_per_thread} not divisible by vector_write_out:{vwo}"
                num_passes = elements_per_thread // vwo
                elems_per_pass = ctrl.block_size * vwo
                assert elems_per_pass % macro_tile_n == 0, f"elems_per_pass:{elems_per_pass} not divisible by macro_tile_n:{macro_tile_n}"
                row_step_per_pass = elems_per_pass // macro_tile_n
                log2_n = utility_log2(macro_tile_n)
                # Phase 23: pad the LDS row stride by one dwordx4 (4 elements) to break a
                # bank-conflict periodicity -- macro_tile_n is always a multiple of 64 (LDS
                # bank count), so the unpadded tile-linear address puts every row of a given
                # column in the same bank (a guaranteed 2-way conflict on every scatter
                # ds_write_b32, since WMMA's per-lane row differs by exactly 8 between the
                # two 16-lane halves, and (row+8)*macro_tile_n === row*macro_tile_n (mod 64)
                # -- both halves collide into the same 16 banks). Padding by 4 elements
                # (not 1) is deliberate: it still fully separates the two halves into
                # disjoint 16-bank ranges (offset by 4*8=32, per the same math), while
                # keeping every row's start byte-address a multiple of 16 -- required for
                # ds_read_b128/global_store_dwordx4 (vwo=4) to stay naturally aligned. A
                # 1-element pad would break bank periodicity too but misalign every row
                # whose index isn't a multiple of 4, corrupting vwo=4 reads. See
                # docs/gfx1250_wmma_layout.md's Phase 23 for the full derivation.
                pad = 4 if ctrl.epilogue_lds_pad else 0
                padded_stride = macro_tile_n + pad
                ds_read_inst = {1: "ds_read_b32", 2: "ds_read_b64", 4: "ds_read_b128"}[vwo]
                gst_inst = {1: "global_store_dword", 2: "global_store_dwordx2", 4: "global_store_dwordx4"}[vwo]
                v_gather_range = f"{v_gather}:{v_gather}+{vwo - 1}" if vwo > 1 else v_gather

                self._emit(f"; wmma LDS-reshuffle coalescing store, tile {macro_tile_m}x{macro_tile_n}, "
                           f"vector_write_out={vwo}, {num_passes} passes")
                # barrier: the main loop's own LDS traffic must be fully retired before we reuse
                # this same physical LDS region for the reshuffle -- otherwise a fast wave could
                # start overwriting it while a slow wave in this workgroup is still reading it.
                self._emit(f"s_wait_dscnt 0x0")
                self._emit(f"s_barrier_signal -1")
                self._emit(f"s_barrier_wait -1")

                # ---- scatter: same (i_rm,i_rn,j) addressing as the atomic path above, retargeted
                # to a tile-linear LDS address via shifts (macro_tile_n is a compile-time power of
                # 2) instead of a runtime-stride multiply -- cheaper than the global-memory case.
                # v_gemm_im/v_gemm_in are GLOBAL (gemm-space) positions -- the caller adds
                # s_block_m_off/s_block_n_off once, persistently, right after computing them (see
                # e.g. igemm_fwd_gtc_wmma_nhwc.py's emit_kernel_prologue) -- so they must be masked
                # back down to tile-local (0..macro_tile_m/n-1) before use as an LDS address: since
                # block_m_off/n_off are always exact multiples of macro_tile_m/n, `& (macro_tile-1)`
                # strips exactly that high part and leaves the tile-local component unchanged. ----
                self._emit(f"v_and_b32 v[{v_tmp2}], {macro_tile_n - 1}, v[{v_gemm_in}]   ; tile-local col (persistent)")
                for i_rm in range(cxm.wave_repeat_m):
                    row_off = i_rm * cxm.wave_tile_m
                    self._emit(f"v_add_u32 v[{v_tmp1}], {row_off}, v[{v_gemm_im}]" if row_off != 0 else f"v_mov_b32 v[{v_tmp1}], v[{v_gemm_im}]")
                    self._emit(f"v_and_b32 v[{v_tmp1}], {macro_tile_m - 1}, v[{v_tmp1}]   ; tile-local row")
                    if pad:
                        self._emit(f"v_mul_lo_u32 v[{v_tmp1}], {padded_stride}, v[{v_tmp1}]   ; row * padded_stride")
                    else:
                        self._emit(f"v_lshlrev_b32 v[{v_tmp1}], {log2_n}, v[{v_tmp1}]   ; row << log2(macro_tile_n)")
                    self._emit(f"v_add_u32 v[{v_tmp1}], v[{v_tmp2}], v[{v_tmp1}]   ; + col -> tile-linear index, row {row_off}")
                    self._emit(f"v_lshlrev_b32 v[{v_tmp1}], 2, v[{v_tmp1}]   ; byte address")
                    for j in range(inst_wmma.num_v_c):
                        if j != 0:
                            self._emit(f"v_add_u32 v[{v_tmp1}], {padded_stride * 4}, v[{v_tmp1}]   ; advance to row {row_off + j}")
                        for i_rn in range(cxm.wave_repeat_n):
                            c_index = (i_rm * cxm.wave_repeat_n + i_rn) * inst_wmma.num_v_c + j
                            col_off = i_rn * cxm.wave_tile_n * 4
                            offset_str = f" offset:{col_off}" if col_off != 0 else ""
                            self._emit(f"ds_write_b32 v[{v_tmp1}], v[{v_c}+{c_index}]{offset_str}")
                self._emit_empty_line()

                # ---- barrier: all scatter writes must be visible to every lane before any gather read ----
                self._emit(f"s_wait_dscnt 0x0")
                self._emit(f"s_barrier_signal -1")
                self._emit(f"s_barrier_wait -1")

                # ---- gather: lane `tid` owns `vwo` contiguous elements per pass, starting at
                # tile-linear index tid*vwo (+ a compile-time pass offset folded into the ds_read's
                # immediate). row0/col0 (this lane's position for pass 0) are derived from that same
                # index via shift/mask (macro_tile_n is a power of 2) -- col never changes across
                # passes since elems_per_pass is a multiple of macro_tile_n by construction, so only
                # the row (and therefore the global memory address) advances, by one scalar add of a
                # precomputed per-pass stride. ----
                self._emit(f"v_lshlrev_b32 v[{v_tmp1}], {utility_log2(vwo)}, v[{v_tid}]   ; tid*{vwo}, tile-linear index for pass 0")
                self._emit(f"v_and_b32 v[{v_gather}], {macro_tile_n - 1}, v[{v_tmp1}]   ; col0 (tile-local)")
                self._emit(f"v_lshrrev_b32 v[{v_tmp2}], {log2_n}, v[{v_tmp1}]   ; row0 (tile-local)")
                # Phase 23: compute the LDS byte address (padded or not) from row0/col0 while
                # they're still tile-local -- must happen BEFORE the block-offset adds below
                # overwrite v_gather/v_tmp2 with global col/row. v_tmp1's original tid*vwo
                # value is dead after this (unpadded case reuses it directly; padded case
                # recomputes from row0*padded_stride+col0, since padding breaks the direct
                # shift relationship between the packed index and the LDS offset).
                if pad:
                    self._emit(f"v_mul_lo_u32 v[{v_tmp1}], {padded_stride}, v[{v_tmp2}]   ; row0 * padded_stride")
                    self._emit(f"v_add_u32 v[{v_tmp1}], v[{v_gather}], v[{v_tmp1}]   ; + col0 -> padded tile-linear index")
                    self._emit(f"v_lshlrev_b32 v[{v_tmp1}], 2, v[{v_tmp1}]   ; padded LDS byte address (invariant across passes)")
                else:
                    self._emit(f"v_lshlrev_b32 v[{v_tmp1}], 2, v[{v_tmp1}]   ; LDS byte address (invariant across passes)")
                self._emit(f"v_add_u32 v[{v_gather}], s[{s_block_n_off}], v[{v_gather}]   ; + block_n_off -> global col")
                self._emit(f"v_add_u32 v[{v_tmp2}], s[{s_block_m_off}], v[{v_tmp2}]   ; + block_m_off -> global row")
                self._emit(f"v_mul_lo_u32 v[{v_tmp2}], s[{s_gemm_m_stride}], v[{v_tmp2}]")
                self._emit(f"v_add_u32 v[{v_tmp2}], v[{v_gather}], v[{v_tmp2}]")
                self._emit(f"v_lshlrev_b32 v[{v_tmp2}], 2, v[{v_tmp2}]   ; global memory byte address for pass 0")
                self._emit(f"s_lshl_b32 s[{s_tmp1}], s[{s_gemm_m_stride}], {utility_log2(row_step_per_pass) + 2}   ; per-pass memory stride")
                self._emit_empty_line()
                for it in range(num_passes):
                    # row_step_per_pass*padded_stride*4 == elems_per_pass*4 when pad=0 (since
                    # elems_per_pass is a multiple of macro_tile_n by construction) -- one
                    # formula, byte-identical to the old literal in the unpadded case.
                    self._emit(f"{ds_read_inst} v[{v_gather_range}], v[{v_tmp1}] offset:{it * row_step_per_pass * padded_stride * 4}")
                    self._emit(f"s_wait_dscnt 0x0")
                    self._emit(f"{gst_inst} v[{v_tmp2}], v[{v_gather_range}], s[{s_p_out}:{s_p_out}+1]")
                    if it != num_passes - 1:
                        self._emit(f"v_add_u32 v[{v_tmp2}], v[{v_tmp2}], s[{s_tmp1}]   ; advance to pass {it + 1}")
                self._emit_empty_line()
        return self._get_deferred()
