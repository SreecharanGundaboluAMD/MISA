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
from .wmma_mapping import *

class ctrl_wmma_main_loop_t(object):
    '''
    Deliberately a single, correctness-first software pipeline schedule -- unlike
    mfma_main_loop.py's ~10 hand-unrolled schedules (which exist to tune CDNA
    occupancy/interleave), this milestone needs exactly one working schedule.
    Also unlike the mfma track there is no AGPR/accvgpr_unified field at all: WMMA
    accumulates directly in v_c (plain VGPR), same register model as the DLOPS track.

    Phase 1 (k-sub-loop): `unroll_k` may now be any power-of-2 MULTIPLE of
    `wmma_m.inst_wmma.k` (not just equal to it) -- one `v_wmma_*` call only ever
    consumes `inst_wmma.k` worth of K, so `unroll_k // inst_wmma.k` k-sub-steps are
    issued against the SAME already-in-LDS tile before the next global-load/shared-
    store/barrier round-trip, amortizing that synchronization cost over more useful
    FLOPs. See `k_substep_stride_bytes_a/b` below for the per-substep LDS offset.
    '''
    def __init__(self):
        self.wmma_m                      = None   # ctrl_wmma_mapping_t
        self.unroll_k                    = 0
        self.label_prefix                = ''
        self.precision                   = 'fp16'

        self.lds_single_size             = 0       # in byte, should be power of 2
        self.lds_buffer_num              = 1        # single-buffered for this milestone

        # Phase 1 (k-sub-loop): byte offset between consecutive inst_wmma.k-wide K-slices
        # within one already-in-LDS tile, one value per operand since transposed vs
        # untransposed operands advance through LDS differently (contiguous-within-a-row
        # vs row-to-row) -- computed by the kernel file that knows each operand's actual
        # LDS layout, not derivable here. Only used when unroll_k > inst_wmma.k.
        self.k_substep_stride_bytes_a    = 0
        self.k_substep_stride_bytes_b    = 0

        # functor
        self.global_load_a_functor       = None
        self.global_load_b_functor       = None
        self.shared_store_a_functor      = None
        self.shared_store_b_functor      = None
        self.shared_load_a_functor       = None
        self.shared_load_b_functor       = None
        self.move_slice_window_a_functor = None
        self.move_slice_window_b_functor = None

        # symbol type
        self.v_a                         = None
        self.v_b                         = None
        self.v_c                         = None
        self.v_sst_a_os                  = None
        self.v_sld_a_os                  = None
        self.v_sst_b_os                  = None
        self.v_sld_b_os                  = None
        self.s_kitr                      = None
        self.s_knum                      = None


class wmma_main_loop_t(mc_base_t):
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        self.ctrl = ctrl

    def emit(self):
        ctrl = self.ctrl
        wmma_m = ctrl.wmma_m
        inst_wmma = wmma_m.inst_wmma
        assert ctrl.unroll_k % inst_wmma.k == 0, \
            f"wmma main loop requires unroll_k({ctrl.unroll_k}) to be a multiple of " \
            f"inst_wmma.k({inst_wmma.k}) -- one v_wmma_* call only ever consumes inst_wmma.k worth of K"
        assert ctrl.lds_buffer_num == 1, "only single-buffered LDS is implemented for this milestone"
        num_k_substeps = ctrl.unroll_k // inst_wmma.k

        label_body     = f'L_{ctrl.label_prefix}_wmma_body'
        label_end      = f'L_{ctrl.label_prefix}_wmma_end'

        f_gld_a = ctrl.global_load_a_functor
        f_gld_b = ctrl.global_load_b_functor
        f_sst_a = ctrl.shared_store_a_functor
        f_sst_b = ctrl.shared_store_b_functor
        f_sld_a = ctrl.shared_load_a_functor
        f_sld_b = ctrl.shared_load_b_functor
        f_move_slice_window_a = ctrl.move_slice_window_a_functor
        f_move_slice_window_b = ctrl.move_slice_window_b_functor

        v_a, v_b, v_c = ctrl.v_a, ctrl.v_b, ctrl.v_c
        v_sst_a_os, v_sld_a_os = ctrl.v_sst_a_os, ctrl.v_sld_a_os
        v_sst_b_os, v_sld_b_os = ctrl.v_sst_b_os, ctrl.v_sld_b_os
        s_kitr, s_knum = ctrl.s_kitr, ctrl.s_knum

        def emit_wmma_tile():
            self._emit(f"; wmma compute, {wmma_m.wave_repeat_m}x{wmma_m.wave_repeat_n} instruction issues")
            for i_rm in range(wmma_m.wave_repeat_m):
                for i_rn in range(wmma_m.wave_repeat_n):
                    c_index = (i_rm * wmma_m.wave_repeat_n + i_rn) * inst_wmma.num_v_c
                    a_index = i_rm * inst_wmma.num_v_a
                    b_index = i_rn * inst_wmma.num_v_b
                    self._emit(inst_wmma(
                        v_c((c_index, c_index + inst_wmma.num_v_c - 1)),
                        v_a((a_index, a_index + inst_wmma.num_v_a - 1)),
                        v_b((b_index, b_index + inst_wmma.num_v_b - 1)),
                        v_c((c_index, c_index + inst_wmma.num_v_c - 1))))

        def emit_extra_substeps():
            # k-sub-step 0's shared_load+wmma is always emitted at its original position
            # (see below) so the pre-existing global-load-latency-hiding overlap is
            # preserved byte-for-byte; substeps 1..N-1 (this drain) have no such overlap
            # opportunity left (the next tile's global load is already in flight by the
            # time this runs), so they're a plain sequential load+wait+compute sequence
            # against the same already-in-LDS tile -- no new barrier/global-load needed.
            for ks in range(1, num_k_substeps):
                off_a = ks * ctrl.k_substep_stride_bytes_a
                off_b = ks * ctrl.k_substep_stride_bytes_b
                self._emit(f_sld_a(v_a(), v_sld_a_os(), off_a))
                self._emit(f_sld_b(v_b(), v_sld_b_os(), off_b))
                self._emit(f"s_wait_dscnt 0x0")
                emit_wmma_tile()

        # ---- prologue: caller has already issued the first global A/B load ----
        # Structure note: the LDS-load + compute for a tile always happens at the TOP
        # of the loop, before checking whether another tile remains -- this ensures the
        # last (or only) tile is always consumed, unlike a naive "check-then-loop"
        # structure which would skip compute entirely for a single-k-block problem.
        self._emit(f"; start WMMA loop, unroll_k:{ctrl.unroll_k}")
        self._emit(f"s_wait_loadcnt 0x0")
        self._emit(f_sst_a())
        self._emit(f_sst_b())
        self._emit_empty_line()

        self._emit(f"s_mov_b32 s[{s_kitr()}], s[{s_knum()}]")
        self._emit_empty_line()

        self._emit_front(f"{label_body}:")
        self._emit(f"s_wait_dscnt 0x0")
        self._emit(f"s_barrier_signal -1")
        self._emit(f"s_barrier_wait -1")
        self._emit_empty_line()

        self._emit(f_sld_a(v_a(), v_sld_a_os(), 0))
        self._emit(f_sld_b(v_b(), v_sld_b_os(), 0))
        self._emit(f"s_wait_dscnt 0x0")
        self._emit_empty_line()

        self._emit(f"s_sub_i32 s[{s_kitr()}], s[{s_kitr()}], {ctrl.unroll_k}")
        self._emit(f"s_cmp_gt_i32 s[{s_kitr()}], 0")
        self._emit(f"s_cbranch_scc0 {label_body}_last")
        self._emit_empty_line()

        self._emit(f_move_slice_window_a())
        self._emit(f_move_slice_window_b())
        self._emit(f_gld_a())
        self._emit(f_gld_b())
        self._emit_empty_line()

        emit_wmma_tile()
        emit_extra_substeps()
        self._emit_empty_line()

        self._emit(f"s_wait_loadcnt 0x0")
        self._emit(f_sst_a())
        self._emit(f_sst_b())
        self._emit(f"s_branch {label_body}")
        self._emit_empty_line()

        self._emit_front(f"{label_body}_last:")
        emit_wmma_tile()
        emit_extra_substeps()
        self._emit_empty_line()

        self._emit_front(f"{label_end}:")
