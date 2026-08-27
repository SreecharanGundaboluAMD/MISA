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

    Phase 2 (double-buffering): `lds_buffer_num` may now be 1 (default, unchanged) or
    2. When 2, `v_sst_a_os`/`v_sst_b_os` (aliases of the SAME physical store-offset
    VGPR -- see each kernel file's emit_kernel_fma_main_loop) are kept permanently
    `lds_single_size` bytes ahead of `v_sld_a_os`/`v_sld_b_os`, i.e. always pointing at
    the OTHER of the two adjacent, power-of-2-sized/aligned LDS buffers -- so a tile
    being prefetched into the "next" buffer never overwrites the tile still being read
    out of the "current" one, without needing a barrier between them. Mirrors
    mfma_main_loop.py's `v_xor_b32 v[offset], lds_single_size, v[offset]` ping-pong
    technique. `lds_single_size` must be a power of 2 for the XOR toggle to correctly
    alternate between exactly two adjacent buffers.
    '''
    def __init__(self):
        self.wmma_m                      = None   # ctrl_wmma_mapping_t
        self.unroll_k                    = 0
        self.label_prefix                = ''
        self.precision                   = 'fp16'

        self.lds_single_size             = 0       # in byte, should be power of 2
        self.lds_buffer_num              = 1        # single-buffered for this milestone

        # Phase 13: per-operand (independent, since e.g. bwd's A goes async while its B
        # stays on the old technique -- B is transposed, out of scope for this phase).
        # When True for an operand, that operand's global_load_a/b_functor issues its data
        # global-memory -> LDS directly via global_load_async_to_lds_b128 (no VGPR staging
        # buffer, no separate shared_store step -- the load IS the LDS write), and that
        # operand's shared_store_a/b_functor is never called. Completion is tracked by
        # ASYNCcnt; s_wait_asynccnt is emitted (alongside s_wait_dscnt, if the OTHER
        # operand still uses the old ds_write_b128-based technique) right before the
        # barrier. Default False for both = today's exact byte-identical behavior.
        self.async_global_to_lds_a       = False
        self.async_global_to_lds_b       = False
        # Phase 28: TDM (Tensor Data Mover)-based load -- shares async's "data lands
        # directly in LDS, no separate store step" control-flow shape (both flags feed the
        # same a_style_async/b_style_async checks below), but waits on a DIFFERENT counter
        # (TENSORcnt via s_wait_tensorcnt, not ASYNCcnt via s_wait_asynccnt), so it's tracked
        # as its own pair of flags rather than folded into async_global_to_lds_a/b. Default
        # False = today's exact byte-identical behavior.
        self.tdm_global_to_lds_a         = False
        self.tdm_global_to_lds_b         = False
        # Phase 31: persistent SGPR tracking "remaining valid K elements from the tile
        # about to be issued" -- decremented by unroll_k once per loop iteration (shared
        # between A and B, since both use the same K tile size), and consumed by
        # move_slice_window_a/b_functor to rebuild each descriptor's tensor_dim0 before
        # every re-issue. None unless tdm_global_load is set.
        self.s_tdm_k_remain              = None

        # Phase 1 (k-sub-loop): byte offset between consecutive inst_wmma.k-wide K-slices
        # within one already-in-LDS tile, one value per operand since transposed vs
        # untransposed operands advance through LDS differently (contiguous-within-a-row
        # vs row-to-row) -- computed by the kernel file that knows each operand's actual
        # LDS layout, not derivable here. Only used when unroll_k > inst_wmma.k.
        self.k_substep_stride_bytes_a    = 0
        self.k_substep_stride_bytes_b    = 0

        # Phase 15 (chunk/compute interleaving): the actual fix for fwd's Phase 1 k-sub-loop
        # regression, per Phase 2's own conclusion. Phase 1 already amortizes the barrier/LDS
        # round-trip over num_k_substeps v_wmma_* issues, but only substep 0's global load
        # (issued before the barrier, see `global_load_a/b_functor`) ever overlaps with
        # compute -- chunks 1..num_k_chunks-1 (== num_k_substeps, by construction: both count
        # inst_wmma.k-wide slices of the same row) were previously loaded+waited+stored
        # SEQUENTIALLY, all AFTER every substep's compute was already done, so their global-
        # load latency was never hidden behind anything. When True (requires
        # num_k_chunks==num_k_substeps > 1, i.e. Phase 1's k-sub-loop feature is in use, and
        # is mutually exclusive with async_global_to_lds_a/b -- the async instruction has no
        # small reused staging buffer to interleave around, see igemm_fwd_gtc_wmma_nhwc.py's
        # global_load_a_functor docstring), each substep ks's compute is paired with chunk
        # (ks+1)'s global-load issue instead: issue chunk ks+1's load, THEN do substep ks's
        # (unrelated, already-in-LDS) shared_load+compute, THEN wait+store chunk ks+1 -- the
        # compute gives the just-issued load real time to complete in the background before
        # its wait is reached. Requires the new global_load_chunk_a/b_functor and
        # shared_store_chunk_a/b_functor (single-chunk primitives, chunk_idx explicit)
        # instead of the bulk global_load_a/b_functor/shared_store_a/b_functor used by the
        # non-interleaved path. Default False = today's exact byte-identical behavior.
        self.interleave                  = False

        # functor
        self.global_load_a_functor       = None
        self.global_load_b_functor       = None
        self.shared_store_a_functor      = None
        self.shared_store_b_functor      = None
        self.shared_load_a_functor       = None
        self.shared_load_b_functor       = None
        self.move_slice_window_a_functor = None
        self.move_slice_window_b_functor = None
        # Phase 15 (interleaving): single-chunk primitives, only used when ctrl.interleave.
        # Callable as f(chunk_idx) -- see docstring above.
        self.global_load_chunk_a_functor  = None
        self.global_load_chunk_b_functor  = None
        self.shared_store_chunk_a_functor = None
        self.shared_store_chunk_b_functor = None

        # Phase 22 (VGPR-level prefetch): 1 (default, unchanged) or 2. When 2, v_a/v_b hold
        # TWO disjoint slots back-to-back (slot 1 starts at VGPR offset
        # wave_repeat_m/n*inst_wmma.num_v_a/b -- see igemm_base.py's num_vgpr_accumulate_a/b
        # formula, doubled to match). Each k-substep's shared_load is issued into the NEXT
        # substep's slot BEFORE the CURRENT substep's compute consumes the slot it already
        # holds -- classic 2-slot software pipelining, intra-K-substep only (mirrors
        # mfma_main_loop.py's local_prefetch_num=2 exactly, one level down at the LDS-read
        # layer instead of the global-load layer that Phase 15's interleave already covers).
        # Mutually exclusive with ctrl.interleave for this first implementation -- both
        # rewrite the same k-substep drain loop, and composing them isn't validated yet.
        self.local_prefetch_num          = 1

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
        assert ctrl.lds_buffer_num in (1, 2), "wmma main loop supports single (1) or double (2) buffered LDS only"
        double_buffer = ctrl.lds_buffer_num == 2
        num_k_substeps = ctrl.unroll_k // inst_wmma.k
        if ctrl.interleave:
            assert num_k_substeps > 1, \
                "interleave requires num_k_substeps>1 (Phase 1's k-sub-loop) -- nothing to interleave otherwise"
            assert not (ctrl.async_global_to_lds_a or ctrl.async_global_to_lds_b), \
                "interleave is not supported together with async_global_to_lds_a/b (no staging buffer to interleave around)"
        assert ctrl.local_prefetch_num in (1, 2), "wmma main loop supports local_prefetch_num of 1 or 2 only"
        prefetch = ctrl.local_prefetch_num == 2
        if prefetch:
            assert num_k_substeps > 1, \
                "local_prefetch_num=2 requires num_k_substeps>1 (Phase 1's k-sub-loop) -- nothing to prefetch into otherwise"
            assert not ctrl.interleave, \
                "local_prefetch_num=2 is not supported together with interleave (both rewrite the k-substep drain loop)"
        num_v_a_total = wmma_m.wave_repeat_m * inst_wmma.num_v_a   # one local_prefetch_num slot's worth
        num_v_b_total = wmma_m.wave_repeat_n * inst_wmma.num_v_b

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
        f_gld_chunk_a = ctrl.global_load_chunk_a_functor
        f_gld_chunk_b = ctrl.global_load_chunk_b_functor
        f_sst_chunk_a = ctrl.shared_store_chunk_a_functor
        f_sst_chunk_b = ctrl.shared_store_chunk_b_functor

        v_a, v_b, v_c = ctrl.v_a, ctrl.v_b, ctrl.v_c
        v_sst_a_os, v_sld_a_os = ctrl.v_sst_a_os, ctrl.v_sld_a_os
        v_sst_b_os, v_sld_b_os = ctrl.v_sst_b_os, ctrl.v_sld_b_os
        s_kitr, s_knum = ctrl.s_kitr, ctrl.s_knum

        def emit_wmma_tile(slot=0):
            self._emit(f"; wmma compute, {wmma_m.wave_repeat_m}x{wmma_m.wave_repeat_n} instruction issues")
            a_slot_off = slot * num_v_a_total
            b_slot_off = slot * num_v_b_total
            for i_rm in range(wmma_m.wave_repeat_m):
                for i_rn in range(wmma_m.wave_repeat_n):
                    c_index = (i_rm * wmma_m.wave_repeat_n + i_rn) * inst_wmma.num_v_c
                    a_index = a_slot_off + i_rm * inst_wmma.num_v_a
                    b_index = b_slot_off + i_rn * inst_wmma.num_v_b
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

        def emit_extra_substeps_prefetched():
            '''
            Phase 22: replaces emit_wmma_tile()+emit_extra_substeps() (both the mid-loop
            call and the `_last` tail's) when ctrl.local_prefetch_num==2. Substep 0's
            shared_load already landed in slot 0 (the unconditional prologue load, issued
            and waited on above, before this function is ever called). At each step ks,
            issue substep (ks+1)'s shared_load into the OTHER slot BEFORE computing on
            substep ks's already-in-hand slot, then wait once for (ks+1)'s data -- so its
            LDS-read latency overlaps substep ks's WMMA issue instead of blocking on it.
            Only 2 physical slots exist regardless of num_k_substeps; reusing slot (ks-1)%2
            for substep ks+1 needs no extra wait beyond the one already emitted for it:
            the wmma instruction reading that slot for substep ks-1 was already ISSUED (in
            program order) before this ds_read is issued, and GCN/RDNA's in-order-per-wave
            issue model guarantees a VALU/WMMA read happens before a later same-register
            LDS write completes, regardless of the LDS read's own completion latency --
            the same guarantee mfma_main_loop.py's local_prefetch_num=2 already relies on.
            '''
            lp = ctrl.local_prefetch_num
            for ks in range(num_k_substeps):
                nxt = ks + 1
                if nxt < num_k_substeps:
                    off_a = nxt * ctrl.k_substep_stride_bytes_a
                    off_b = nxt * ctrl.k_substep_stride_bytes_b
                    self._emit(f_sld_a(v_a(), v_sld_a_os(), off_a, slot=nxt % lp))
                    self._emit(f_sld_b(v_b(), v_sld_b_os(), off_b, slot=nxt % lp))
                emit_wmma_tile(slot=ks % lp)
                if nxt < num_k_substeps:
                    self._emit(f"s_wait_dscnt 0x0")

        def emit_interleaved_substeps():
            '''
            Phase 15: replaces emit_extra_substeps() + the later bulk wait+store when
            ctrl.interleave. Substep 0's compute (using v_a/v_b already read at the top of
            the loop body) has already happened by the time this is called; chunk 0's global
            load (issued via f_gld_a()/f_gld_b() before this, unchanged from the
            non-interleaved path) is waited+stored here, THEN each remaining substep ks
            (1..num_k_substeps-1) issues chunk ks's load BEFORE its own (unrelated,
            already-in-LDS) shared_load+compute -- giving that load real time to complete in
            the background -- and only waits+stores it after the compute. The final chunk
            (num_k_substeps-1) still gets this same treatment; its wait+store simply lands
            right before the loop's own trailing bookkeeping (buffer switch / branch), same
            as the non-interleaved path's bulk store did.
            '''
            self._emit(f"s_wait_loadcnt 0x0   ; chunk 0's load")
            self._emit(f_sst_chunk_a(0))
            self._emit(f_sst_chunk_b(0))
            for ks in range(1, num_k_substeps):
                off_a = ks * ctrl.k_substep_stride_bytes_a
                off_b = ks * ctrl.k_substep_stride_bytes_b
                self._emit(f_gld_chunk_a(ks))
                self._emit(f_gld_chunk_b(ks))
                self._emit(f_sld_a(v_a(), v_sld_a_os(), off_a))
                self._emit(f_sld_b(v_b(), v_sld_b_os(), off_b))
                self._emit(f"s_wait_dscnt 0x0")
                emit_wmma_tile()
                self._emit(f"s_wait_loadcnt 0x0   ; chunk {ks}'s load")
                self._emit(f_sst_chunk_a(ks))
                self._emit(f_sst_chunk_b(ks))

        def emit_buffer_switch():
            # Phase 2 (double-buffering): toggle the store offset (v_sst_a_os and
            # v_sst_b_os alias the same physical VGPR, so one toggle covers both) and
            # both read offsets to the OTHER LDS buffer, maintaining the invariant that
            # store and read always point at different buffers.
            self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {ctrl.lds_single_size}, v[{v_sst_a_os()}]")
            self._emit(f"v_xor_b32 v[{v_sld_a_os()}], {ctrl.lds_single_size}, v[{v_sld_a_os()}]")
            self._emit(f"v_xor_b32 v[{v_sld_b_os()}], {ctrl.lds_single_size}, v[{v_sld_b_os()}]")

        # ---- prologue: caller has already issued the first global A/B load ----
        # Structure note: the LDS-load + compute for a tile always happens at the TOP
        # of the loop, before checking whether another tile remains -- this ensures the
        # last (or only) tile is always consumed, unlike a naive "check-then-loop"
        # structure which would skip compute entirely for a single-k-block problem.
        self._emit(f"; start WMMA loop, unroll_k:{ctrl.unroll_k}")

        async_a = ctrl.async_global_to_lds_a
        async_b = ctrl.async_global_to_lds_b
        tdm_a = ctrl.tdm_global_to_lds_a
        tdm_b = ctrl.tdm_global_to_lds_b
        # Phase 28: TDM shares async's "data lands directly in LDS, no separate store step"
        # control-flow shape -- a_style_async/b_style_async gate every "is this operand
        # already in LDS" check below, identically for async and TDM. Only the actual WAIT
        # instruction differs (s_wait_asynccnt vs s_wait_tensorcnt, see label_body below).
        a_style_async = async_a or tdm_a
        b_style_async = async_b or tdm_b
        any_async = async_a or async_b
        any_tdm   = tdm_a or tdm_b
        any_old   = (not a_style_async) or (not b_style_async)

        # Phase 13: an operand on the async path already issued its first tile's data
        # straight into LDS (global_load_async_to_lds_b128, no VGPR staging, no separate
        # store step) -- nothing to wait on or store here for it; that wait happens at
        # the top of label_body, same slot as every other iteration's wait for the
        # PREVIOUS iteration's prefetch. An operand still on the old path needs its
        # usual s_wait_loadcnt + deferred store, same as always.
        if any_old:
            self._emit(f"s_wait_loadcnt 0x0")
            if not a_style_async:
                self._emit(f_sst_a())
            if not b_style_async:
                self._emit(f_sst_b())
            self._emit_empty_line()

        if double_buffer:
            # Phase 2: the tile just stored above landed in buffer 0 (v_sst_a_os's
            # initial position, same as v_sld_a/b_os's) -- advance the STORE offset
            # alone to buffer 1 so the first loop iteration's read (buffer 0, this
            # tile) and prefetch-store (buffer 1, the next tile) target different
            # buffers from the very first iteration onward.
            self._emit(f"v_xor_b32 v[{v_sst_a_os()}], {ctrl.lds_single_size}, v[{v_sst_a_os()}]")
            self._emit_empty_line()

        self._emit(f"s_mov_b32 s[{s_kitr()}], s[{s_knum()}]")
        self._emit_empty_line()

        self._emit_front(f"{label_body}:")
        # Gates THIS wave's own writes into the tiles about to be read below -- an async
        # operand's LDS write (ASYNCcnt) and/or an old-path operand's ds_write_b128
        # (DSCnt) -- before this wave tells the workgroup it's safe to cross the barrier
        # and read. Both waits coexist when one operand is async and the other isn't
        # (e.g. bwd: A async, B still on the old technique).
        if any_async:
            self._emit(f"s_wait_asynccnt 0x0")
        if any_tdm:
            self._emit(f"s_wait_tensorcnt 0x0")
        if any_old:
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

        if any_tdm:
            # Phase 31: decremented exactly ONCE per iteration here (not inside
            # move_slice_window_a/b_functor themselves, which both run every iteration --
            # decrementing in either of those would double-count) -- both A's and B's
            # descriptor rebuild below read this same post-decrement value.
            self._emit(f"s_sub_i32 s[{ctrl.s_tdm_k_remain()}], s[{ctrl.s_tdm_k_remain()}], {ctrl.unroll_k}   ; Phase 31: remaining valid K for the tile about to be issued")
        self._emit(f_move_slice_window_a())
        self._emit(f_move_slice_window_b())
        if ctrl.interleave:
            # Phase 15: chunk 0 issued exactly as the non-interleaved path does; substep 0's
            # compute happens next, then emit_interleaved_substeps() takes over chunk 0's
            # wait+store AND every remaining chunk/substep, interleaved.
            self._emit(f_gld_a())
            self._emit(f_gld_b())
            self._emit_empty_line()
            emit_wmma_tile()
            emit_interleaved_substeps()
            self._emit_empty_line()
        else:
            if not a_style_async:
                self._emit(f_gld_a())
            if not b_style_async:
                self._emit(f_gld_b())
            self._emit_empty_line()

            if prefetch:
                emit_extra_substeps_prefetched()
            else:
                emit_wmma_tile()
                emit_extra_substeps()
            self._emit_empty_line()

            # Phase 13: an async operand's NEXT-tile load-to-LDS is issued only now, after
            # this tile's ds_reads (f_sld_a/f_sld_b above) are fully consumed -- the async
            # load IS the LDS write, so (unlike the old design's chunk-0 early-issue trick)
            # there is no safe-to-start-early sub-step to preserve; deferring ALL of it is
            # what keeps the cross-wave LDS-write-visibility invariant from Phase 9 intact.
            # An old-path operand keeps its usual s_wait_loadcnt + deferred store.
            if any_old:
                self._emit(f"s_wait_loadcnt 0x0")
                if not a_style_async:
                    self._emit(f_sst_a())
                if not b_style_async:
                    self._emit(f_sst_b())
            if a_style_async:
                self._emit(f_gld_a())
            if b_style_async:
                self._emit(f_gld_b())
        if double_buffer:
            emit_buffer_switch()
        self._emit(f"s_branch {label_body}")
        self._emit_empty_line()

        self._emit_front(f"{label_body}_last:")
        if prefetch:
            emit_extra_substeps_prefetched()
        else:
            emit_wmma_tile()
            emit_extra_substeps()
        self._emit_empty_line()

        self._emit_front(f"{label_end}:")
