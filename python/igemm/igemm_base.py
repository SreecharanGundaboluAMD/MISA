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
from __future__ import print_function
import sys
import math
from ..codegen import *
from ..operations import *


IGEMM_GTC_FEAT_ALLOW_LDS_REORDER = 0
IGEMM_GTC_FEAT_PRECACHE_SOFFSET = 1
IGEMM_GTC_FEAT_LOCAL_PREFETCH = 1
IGEMM_GTC_FEAT_FMA_INTERLEAVE = 1
IGEMM_GTC_FEAT_MAGIC_DIVISION = 1
IGEMM_GTC_FEAT_SOURCE_ACCESS_ENCODING_KERNEL_NAME = 0

# IGEMM_GTC_TENSOR_LAYOUT_NCHW = ((1 << 4) | 0)
# IGEMM_GTC_TENSOR_LAYOUT_NHWC = ((1 << 4) | 1)
# IGEMM_GTC_TENSOR_LAYOUT_CNHW = ((1 << 4) | 2)


IGEMM_GTC_TUNABLE_FMA_TYPE_MAC               = 'mac'
IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS             = 'dlops'
IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS            = 'xdlops'
IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA              = 'wmma'


IGEMM_GTC_TUNABLE_SOURCE_ACCESS_ORDER_GEMM_M_GEMM_N       = 0    # m*n, load gemm_n first
IGEMM_GTC_TUNABLE_SOURCE_ACCESS_ORDER_GEMM_N_GEMM_M       = 1    # n*m, load gemm_m first

def igemm_get_vector_size(v):
    vec_size = 1
    if v % 4 == 0:
        vec_size = 4
    elif v % 2 == 0:
        vec_size = 2
    else:
        pass
    return vec_size

# compute next power of 2
def igemm_next_pow2(n):
    if n == 0:
        return 1
    if n & (n - 1) == 0:
        return n
    while n & (n - 1) > 0:
        n &= (n - 1)
    return n << 1

def igemm_next_mul(n, mul):
    d = n // mul
    d = d + (1 if (n % mul != 0) else 0)
    return d * mul

def igemm_is_pow2(v):
    return v and (not(v & (v - 1)))

def igemm_log2(v):
    assert (v and (not(v & (v - 1)))), 'v:{} must be power of 2'.format(v)
    return int(math.log2(v))

def igemm_division_magic(divisor):
    '''
    compute magic num for fast int divison
    divisor <= pow(2, 31)
    '''
    assert(divisor <= pow(2, 31))
    magic_shift = 0
    for i in range(31):
        if pow(2, i) >= divisor:
            magic_shift = i
            break
    magic_num = int(pow(2, 32) * (pow(2, magic_shift) - divisor) / divisor) + 1
    return magic_num, magic_shift

def igemm_get_epack_length(precision):
        # GetEPackLength
        epack = 1
        if precision == AMDGPU_PRECISION_FP16:
            # todo: xdlops check
            epack = 2
        elif precision == AMDGPU_PRECISION_BF16:
            epack = 2
        return epack

def igemm_gcd(a, b):
    # math.gcd new in python 3.5
    return math.gcd(a, b)

def igemm_lcm(a, b):
    return abs(a * b) // math.gcd(a, b)

def igemm_flatten_list_product(x):
    assert type(x) is list
    from functools import reduce
    return reduce(lambda a, b: a*b, x)

def igemm_flatten_list_accumulate(x):
    assert type(x) is list
    from functools import reduce
    return reduce(lambda a, b: a+b, x)

def emit_vopd_paired_zero_init(emit_fn, reg_at, count):
    '''
    Phase 67: pairs consecutive `v_mov_b32 v[reg], 0` zero-inits into
    `v_dual_mov_b32 ... :: v_dual_mov_b32 ...` (CDNA5 dual-issue VALU, ISA doc
    section 7.8) -- halves the instruction count wherever a contiguous run of
    same-value (0) VGPR zero-inits is emitted. Safe unconditionally (not gated
    behind an opt-in tunable, unlike most codegen variants in this project):
    `v_mov_b32` is both X- and Y-slot eligible, the immediate 0 is a single
    literal SHARED by both halves (VOPD's encoding allows exactly one literal,
    shared or single-use -- two DIFFERENT literals in one pair would NOT be
    legal, which is why this helper must not be reused for non-uniform-value
    zero/const-inits), and any two CONSECUTIVE VGPR indices always differ in
    destination parity, satisfying VOPD's even/odd-dest-bank constraint for
    free regardless of the starting register's own parity. Confirmed
    `v_dual_mov_b32 vN, 0 :: v_dual_mov_b32 vN+1, 0` assembles cleanly on the
    pinned toolchain for multiple N (both even and odd starting points) via a
    standalone llvm-mc/clang smoke test -- see docs/gfx1250_wmma_layout.md's
    Phase 67.

    `reg_at(i)` returns the register label (string, as used inside an
    f-string's `v[...]`) for zero-init index i (0..count-1); `emit_fn` is
    typically `self._emit`. Falls back to a single plain `v_mov_b32` for a
    trailing odd count.
    '''
    i = 0
    while i + 1 < count:
        emit_fn(f"v_dual_mov_b32 v[{reg_at(i)}], 0 :: v_dual_mov_b32 v[{reg_at(i+1)}], 0")
        i += 2
    if i < count:
        emit_fn(f"v_mov_b32 v[{reg_at(i)}], 0")

def get_igemm_gtc_fma_type(tunable_dict):
    assert type(tunable_dict) is dict
    if 'gemm_m_per_thread' in tunable_dict and 'gemm_n_per_thread' in tunable_dict:
        if tunable_dict['arch'] == 'gfx900':
            return IGEMM_GTC_TUNABLE_FMA_TYPE_MAC
        if tunable_dict['arch'] in ('gfx906', 'gfx908', 'gfx90a', 'gfx940', 'gfx942', 'gfx950', 'gfx1030'):
            return IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS
    if 'wave_tile_m' in tunable_dict and 'wave_tile_n' in tunable_dict:
        assert tunable_dict['arch'] in ('gfx908', 'gfx90a', 'gfx940', 'gfx942', 'gfx950')
        return IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS
    if 'wmma_tile_m' in tunable_dict and 'wmma_tile_n' in tunable_dict:
        assert tunable_dict['arch'] in ('gfx1250',)
        return IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA
    if 'lanegroup_tile_m' in tunable_dict and 'lanegroup_tile_n' in tunable_dict:
        return IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS
    assert False

def get_igemm_gtc_gemm_k_global_split(tunable_dict):
    assert type(tunable_dict) is dict
    if tunable_dict['arch'] in ('gfx908', 'gfx90a', 'gfx940', 'gfx942', 'gfx950'):
        gemm_k_global_split = utility_dict_with_default_t(tunable_dict)('gemm_k_global_split', 0)
        if gemm_k_global_split > 0:
            return 1
        else:
            return 0
    else:
        return 0

def igemm_get_fma_type_from_arch_config(arch_config):
    assert type(arch_config) is amdgpu_arch_config_t
    if arch_config.use_xdlops:
        return IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS
    if getattr(arch_config, 'use_wmma', False):
        return IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA
    if arch_config.use_dlops:
        return IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS
    return IGEMM_GTC_TUNABLE_FMA_TYPE_MAC

def igemm_use_lanegroup_thread_mapping(tunable):
    attrs = ('lanegroup_tile_m', 'lanegroup_tile_n', 'lanegroup_wave_m', 'lanegroup_wave_n')
    if type(tunable) is dict:
        if all(key in tunable for key in attrs):
            return True
    elif type(tunable) is igemm_gtc_tunable_parameter_t:
        if all(hasattr(tunable, attr) for attr in attrs):
            return True
    return False

class igemm_gtc_tunable_parameter_t(object):
    '''
    generic tensor contraction
    '''
    def __init__(self, tunable_dict):
        self.tensor_layout                      = utility_dict_with_default_t(tunable_dict)('tensor_layout', 'nchw')
        self.gemm_m_per_block                   = tunable_dict['gemm_m_per_block']
        self.gemm_n_per_block                   = tunable_dict['gemm_n_per_block']
        self.gemm_k_per_block                   = tunable_dict['gemm_k_per_block']
        self.fma_type                           = get_igemm_gtc_fma_type(tunable_dict)
        if self.fma_type in (IGEMM_GTC_TUNABLE_FMA_TYPE_MAC, IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS):
            if igemm_use_lanegroup_thread_mapping(tunable_dict):
                self.lanegroup_tile_m           = tunable_dict['lanegroup_tile_m']
                self.lanegroup_wave_m           = tunable_dict['lanegroup_wave_m']
                self.lanegroup_repeat_m         = tunable_dict['lanegroup_repeat_m']
                self.lanegroup_tile_n           = tunable_dict['lanegroup_tile_n']
                self.lanegroup_wave_n           = tunable_dict['lanegroup_wave_n']
                self.lanegroup_repeat_n         = tunable_dict['lanegroup_repeat_n']
            else:
                self.gemm_m_per_thread          = tunable_dict['gemm_m_per_thread']
                self.gemm_m_level0_cluster      = tunable_dict['gemm_m_level0_cluster']
                self.gemm_m_level1_cluster      = tunable_dict['gemm_m_level1_cluster']
                self.gemm_n_per_thread          = tunable_dict['gemm_n_per_thread']
                self.gemm_n_level0_cluster      = tunable_dict['gemm_n_level0_cluster']
                self.gemm_n_level1_cluster      = tunable_dict['gemm_n_level1_cluster']
        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            self.wave_tile_m                    = tunable_dict['wave_tile_m']
            self.wave_step_m                    = tunable_dict['wave_step_m']
            self.wave_repeat_m                  = tunable_dict['wave_repeat_m']
            self.wave_tile_n                    = tunable_dict['wave_tile_n']
            self.wave_step_n                    = tunable_dict['wave_step_n']
            self.wave_repeat_n                  = tunable_dict['wave_repeat_n']
            self.wave_tile_k                    = utility_dict_with_default_t(tunable_dict)('wave_tile_k', 1)
        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA:
            self.wmma_tile_m                    = tunable_dict['wmma_tile_m']
            self.wmma_repeat_m                  = tunable_dict['wmma_repeat_m']
            self.wmma_tile_n                    = tunable_dict['wmma_tile_n']
            self.wmma_repeat_n                  = tunable_dict['wmma_repeat_n']
            # Phase 2 (gfx1250 WMMA LDS double-buffering): optional, defaults to 0
            # (single-buffered, every existing config) so old configs are unaffected.
            self.lds_double_buffer              = utility_dict_with_default_t(tunable_dict)('lds_double_buffer', 0)
            # Phase 13: optional, defaults to 0 (today's VGPR-staged global_load_dwordx4 +
            # ds_write_b128 path, every existing config unaffected). When 1, untransposed
            # operands (fwd A/B, bwd A) use global_load_async_to_lds_b128 instead -- no VGPR
            # staging buffer, global memory -> LDS directly.
            self.async_global_load              = utility_dict_with_default_t(tunable_dict)('async_global_load', 0)
            # Phase 61: optional, defaults to 0 (today's 64-bit-VADDR-pair global_load_dwordx4
            # path, every existing config unaffected). When 1, the default (non-async,
            # non-TDM) global-load path for fwd's A/B operands uses a 32-bit byte-offset VGPR
            # + a scalar SADDR base (s_p_in/s_p_wei) instead of a per-thread 64-bit VGPR
            # address pair -- same GLOBAL_* "GVS" addressing mode
            # (addr = IOFFSET + SADDR[63:0] + VADDR[31:0]) async_global_load's
            # global_load_async_to_lds_b128 already uses, just applied to the ordinary
            # VGPR-staged global_load_dwordx4 the ASYNC/TDM paths don't take. Saves 1 VGPR
            # (2*row_repeat -> 1) and 1 VALU carry op per address step. fwd-only pilot for
            # now -- mutually exclusive with async_global_load/tdm_global_load/
            # main_loop_interleave/gemm_k_global_split/row_repeat_a>1/row_repeat_b>1, asserted
            # in igemm_fwd_gtc_wmma_nhwc.py's __init__, same "narrowest correctness-first
            # slice" discipline as every other addressing mechanism there. See
            # docs/gfx1250_wmma_layout.md's Phase 62.
            self.saddr_global_load              = utility_dict_with_default_t(tunable_dict)('saddr_global_load', 0)
            # Phase 28: TDM (Tensor Data Mover)-based global-to-LDS load for the A operand --
            # optional, defaults to 0 (today's exact byte-identical behavior). When 1, uses
            # tensor_load_to_lds (the dedicated TDM hardware unit) instead of
            # global_load_async_to_lds_b128. fwd-only, 1x1-conv-only (nxe==0), and mutually
            # exclusive with async_global_load/main_loop_interleave for this first pilot --
            # row_repeat_a>1 and local_prefetch_num>1 exclusions are asserted in
            # igemm_fwd_gtc_wmma_nhwc.py's __init__ (mirroring async_global_load's identical
            # row_repeat_a assert there). See docs/gfx1250_wmma_layout.md's Phase 28.
            self.tdm_global_load                = utility_dict_with_default_t(tunable_dict)('tdm_global_load', 0)
            if self.tdm_global_load:
                # NOTE: self.direction/self.nxe aren't set yet at this point in __init__ (this
                # fma_type-specific branch runs before them) -- read the raw dict instead.
                # Phase 42: bwd added -- grad_output (A) is NHWC-contiguous per pixel exactly
                # like fwd's A for the 1x1/unit-stride case (bwd's "harder per-tap gather"
                # only matters for multi-tap; y=x=1 collapses it to a trivial identity), and
                # weight (B) is read in the SAME physical layout fwd's B reads, just with
                # tensor_dim0/tensor_dim1's roles swapped (bwd's GEMM_K is weight's ROW axis,
                # not its contiguous axis) -- see docs/gfx1250_wmma_layout.md's Phase 42.
                # Phase 45: wrw added -- GEMM_K (spatial n*ho*wo) is the ROW axis for BOTH
                # operands here (grad_output and input are both NHWC, channel-contiguous
                # per pixel), unlike bwd where only B needed axis-swapping. Base case only
                # (no M/N-tail combined with TDM yet) -- see igemm_wrw_gtc_wmma_nhwc.py's
                # __init__ assert and docs/gfx1250_wmma_layout.md's Phase 45.
                assert tunable_dict['direction'] in ('fwd', 'bwd', 'wrw'), "tdm_global_load is only implemented for fwd/bwd/wrw so far, see docs/gfx1250_wmma_layout.md's Phase 28/42/45"
                assert tunable_dict['nxe'] == 0, "tdm_global_load is only implemented for 1x1/unit-stride convs (nxe=0) so far, see docs/gfx1250_wmma_layout.md's Phase 28"
                assert not self.async_global_load, "tdm_global_load and async_global_load are mutually exclusive -- they're two different load mechanisms for the same operand"
                assert not utility_dict_with_default_t(tunable_dict)('main_loop_interleave', 0), \
                    "tdm_global_load and main_loop_interleave are mutually exclusive for now, see docs/gfx1250_wmma_layout.md's Phase 28"
            # Phase 15: optional, defaults to 0 (today's exact byte-identical main loop).
            # When 1, interleaves each remaining k-sub-loop chunk's global load with the
            # PREVIOUS chunk/substep's (unrelated, already-in-LDS) compute instead of loading
            # + storing all remaining chunks sequentially after every substep's compute is
            # done -- the actual fix for fwd's Phase 1 k-sub-loop regression, per Phase 2's
            # own conclusion. Requires gemm_k_per_block > inst_wmma.k (k-sub-loop in use) and
            # is mutually exclusive with async_global_load.
            self.main_loop_interleave            = utility_dict_with_default_t(tunable_dict)('main_loop_interleave', 0)
            # Phase 32: brackets each emit_wmma_tile() call's WMMA-issue burst with
            # s_setprio 1 (before) / s_setprio 0 (after) -- a pure instruction-issue-priority
            # hint (CDNA5 ISA doc 5.2, SYS_PRIO/USER_PRIO), no correctness or register
            # implications, so no mutual-exclusion asserts needed. Independently confirmed as
            # real, shipping code in both CK's WMMA v1 pipeline and hipconv's direct/kernel.hpp
            # -- see docs/gfx1250_perf_parity_action_plan.md's Tier 1 item 1. Default 0 = every
            # existing config byte-identical.
            self.wmma_setprio                    = utility_dict_with_default_t(tunable_dict)('wmma_setprio', 0)
            # Phase 22 (VGPR-level prefetch): local_prefetch_num is read further below,
            # in the num_vgpr_accumulate_a/b section -- this __init__ has a later, shared
            # `self.local_prefetch_num = 1` default (for every fma_type) that runs AFTER
            # this point and would otherwise clobber a value read here.
            # Phase 23 (ISA-driven epilogue tuning): all three default to today's exact
            # byte-identical behavior, every existing config unaffected. See
            # coalescing_store_wmma.py and docs/gfx1250_wmma_layout.md's Phase 23 for the
            # ISA citations behind each.
            # atomic_scope: SCOPE_SYS (default) or SCOPE_DEV for the wrw gemm_k_global_split
            # atomic-add epilogue -- SYS forces a full system-level flush/invalidate; DEV
            # resolves within the device's L2, sufficient since K-split workgroups are always
            # on the same device.
            self.atomic_scope                   = utility_dict_with_default_t(tunable_dict)('atomic_scope', 'SCOPE_SYS')
            # atomic_cascade: 0 (default, regular atomic) or 1 (TH[2] cascading/deferred-scope
            # atomic). CONFIRMED HANGS ON REAL HARDWARE (Phase 23) -- DO NOT ENABLE without
            # completing the TODO below.
            #
            # TODO (atomic_cascade resumption checklist):
            # 1. ROOT CAUSE: a cascading atomic defers its full-scope completion signal to
            #    "a subsequent release/fence/atomic-operation of a matching or higher scope"
            #    (CDNA5 ISA doc §4.1 Table 12). The existing `s_wait_storecnt 0x0` at kernel
            #    end (wrw_gtc_wmma_nhwc.py's emit_kernel_fma_end) waits for that signal, which
            #    never arrives because no such release is ever issued -- GPU hangs indefinitely.
            #
            # 2. FIX REQUIRED BEFORE ENABLING: add a release (one of the following, each
            #    chosen from ISA doc §4.1 "VMEM Policies for Writeback and Invalidate Ops"):
            #    (a) Simplest: emit a matching `global_store_b32` (or zero-sized FLAT with
            #        `cpol:SCOPE_DEV`) with TH_STORE_WB (write-back) at the same scope level
            #        as atomic_scope, after the last atomic but before the existing storecnt wait.
            #        This is a "release" in the ISA's sense -- completes the deferred scope.
            #    (b) Alternative: replace the existing `s_wait_storecnt 0x0` with an explicit
            #        `buffer_gl2_wb` (writeback to L2) + `s_wait_storecnt`, which forces the
            #        same scope-promotion the release above would provide.
            #    (c) Check whether `s_wait_storecnt` alone is sufficient after removing the
            #        TH_ATOMIC_CASCADE_RT bit -- the hang is specifically from the deferred-scope
            #        semantic, so removing that bit would revert to a normal atomic.
            #
            # 3. VERIFY ENCODING (already done, no re-work needed): llvm-mc -show-encoding
            #    confirmed `th:TH_ATOMIC_CASCADE_RT` on `global_atomic_add_f32` produces the
            #    correct bit pattern -- this part is correct and need not be re-probed.
            #
            # 4. TEST SEQUENCE after implementing the release:
            #    (a) Add a single `global_atomic_add_f32 ... th:TH_ATOMIC_CASCADE_RT` with
            #        a companion release to a minimal HIP probe (not the full igemm kernel)
            #        and confirm it completes without a hang -- isolate the fix before wiring
            #        into the full codegen.
            #    (b) Wire into coalescing_store_wmma.py's atomic branch; re-enable this assert
            #        (change `assert not` to `assert in (0, 1)`), create the gsplit_cascade
            #        config variant (was deleted in Phase 23 -- recreate from
            #        igemm_wrw_gtc_gfx1250_nhwc_bf16_gsplit.config + atomic_cascade = 1).
            #    (c) Full correctness battery: conv_driver.exe -V 1 for wrw gsplit across all
            #        shape battery shapes. Then benchmark vs baseline, comparing both the
            #        cascade-only and cascade+SCOPE_DEV combinations.
            #
            # 5. EXPECTED BENEFIT (if it works): the ISA doc describes the cascade pattern
            #    as purpose-built for "histogram (non-returning) type atomic ops" -- exactly
            #    wrw's K-split accumulation pattern. The expected benefit is reduced per-atomic
            #    L2-coherence overhead: instead of each non-returning atomic-add fully committing
            #    at the requested scope before the next begins, the hardware can batch/pipeline
            #    the scope promotion until the eventual release. May help most for the
            #    small-K-dimension wrw shapes with many K-split workgroups contending on
            #    the same output elements (e.g. the shapes measured as 2-5x slower than
            #    MIOpen's wrw solver in the Phase 23 gfx1250 trace comparison).
            #
            # See docs/gfx1250_wmma_layout.md's Phase 23 for the full hang diagnosis and the
            # llvm-mc probe transcript.
            self.atomic_cascade                 = utility_dict_with_default_t(tunable_dict)('atomic_cascade', 0)
            assert not self.atomic_cascade, \
                "atomic_cascade=1 hangs on real hardware (s_wait_storecnt never completes without a companion release) -- not usable yet; see TODO in igemm_base.py and docs/gfx1250_wmma_layout.md Phase 23"
            # epilogue_lds_pad: 0 (default, unpadded) or 1 (pad the non-atomic LDS-reshuffle
            # epilogue's row stride by one element to break a bank-conflict periodicity --
            # macro_tile_n is always a multiple of 64, so the unpadded tile-linear address
            # puts every row of a given column in the same LDS bank). Only affects the
            # non-atomic (fwd/bwd/non-split-wrw) epilogue branch.
            self.epilogue_lds_pad                = utility_dict_with_default_t(tunable_dict)('epilogue_lds_pad', 0)
            # lds_row_pad: bytes of padding added to each LDS row in the main-loop A/B
            # tile (NOT the epilogue -- see epilogue_lds_pad above). The default 64 B row
            # stride aliases 32 lanes onto 4 of 64 LDS bank groups; +16 B makes the stride
            # 80 B (gcd(20,64)==4, conflict-free). Measured +9-27% on 1x1 shapes (perf
            # report 2026-09-02 OPT-1/Phase B). Must be a multiple of 16 and produce
            # gcd(stride_dwords, 64)==4. Default 0 (unpadded, every existing config
            # unaffected). Only affects the main-loop LDS layout, NOT the global stride
            # (bytes_per_row stays the global stride for move_slice_window).
            self.lds_row_pad                    = utility_dict_with_default_t(tunable_dict)('lds_row_pad', 0)
        else:
            assert False

        self.tensor_a_pass_through              = utility_dict_with_default_t(tunable_dict)('tensor_a_pass_through', 0)
        self.tensor_b_pass_through              = utility_dict_with_default_t(tunable_dict)('tensor_b_pass_through', 0)
        self.tensor_a_thread_lengths            = tunable_dict['tensor_a_thread_lengths']     # list!
        self.tensor_a_cluster_lengths           = tunable_dict['tensor_a_cluster_lengths']    # list!
        self.tensor_b_thread_lengths            = tunable_dict['tensor_b_thread_lengths']     # list!
        self.tensor_b_cluster_lengths           = tunable_dict['tensor_b_cluster_lengths']    # list!
        self.direction                          = tunable_dict['direction']
        self.precision                          = tunable_dict['precision']
        self.nxb                                = tunable_dict['nxb']           # multiplier of b
        self.nxe                                = tunable_dict['nxe']           # muptiplier of e. here if 0, means x=y=1
        default_mh                              = 1 if (self.direction == 'bwd' and self.tensor_layout == "nhwc" and self.nxe != 0) else 0
        self.multihead                          = utility_dict_with_default_t(tunable_dict)('multihead', default_mh)
        self.gemm_k_global_split                = get_igemm_gtc_gemm_k_global_split(tunable_dict)
        self.allow_lds_reorder                  = utility_dict_with_default_t(tunable_dict)('allow_lds_reorder', IGEMM_GTC_FEAT_ALLOW_LDS_REORDER)
        self.precache_soffset                   = utility_dict_with_default_t(tunable_dict)('precache_soffset', IGEMM_GTC_FEAT_PRECACHE_SOFFSET)

        default_source_access_order             = IGEMM_GTC_TUNABLE_SOURCE_ACCESS_ORDER_GEMM_N_GEMM_M if (self.direction == 'fwd' and self.tensor_layout == 'nchw') \
                                                        else IGEMM_GTC_TUNABLE_SOURCE_ACCESS_ORDER_GEMM_M_GEMM_N
        self.source_access_order                = utility_dict_with_default_t(tunable_dict)('source_access_order', default_source_access_order)

        self.gemm_m_unmerge_cluster             = utility_dict_with_default_t(tunable_dict)('gemm_m_unmerge_cluster', 0)
        self.gemm_n_unmerge_cluster             = utility_dict_with_default_t(tunable_dict)('gemm_n_unmerge_cluster', 0)
        self.gemm_k_unmerge_cluster             = utility_dict_with_default_t(tunable_dict)('gemm_k_unmerge_cluster', 0)     # maybe no need support for 1
        self.vector_store                       = utility_dict_with_default_t(tunable_dict)('vector_store', 0)
        self.gemm_k_global_split                = utility_dict_with_default_t(tunable_dict)('gemm_k_global_split', 0)
        # Phase 57: gfx1250 WMMA's gemm_k_global_split atomic epilogue used to be ALWAYS
        # global_atomic_add_f32 (coalescing_store_wmma.py) -- for int8/int4, v_c holds a
        # genuine int32 accumulate, not a float, so that atomic add was a bit-pattern-
        # reinterpreted float addition, not an integer one. It was EXACT (bit-for-bit
        # correct) only when every partial-sum element was non-negative and stayed within
        # the ~8.39M subnormal-float range (IEEE754 subnormal addition of small non-negative
        # integers, stored as raw int32 bit patterns, happens to be bit-exact integer
        # addition with no rounding) -- confirmed empirically while porting fwd's Phase 49
        # split-K: a small (~512-magnitude, all-positive) int8 test shape passed
        # nrms:0.000000 even with a real 8-way cross-shard split, which initially looked
        # like proof of correctness but was actually a subnormal-arithmetic coincidence
        # specific to small non-negative sums. Realistic int8 conv accumulators are commonly
        # NEGATIVE (signed activations/weights) or larger than 8.39M in magnitude, where that
        # reinterpretation would have silently corrupted the result. Fixed in Phase 57:
        # coalescing_store_wmma.py's atomic path now emits global_atomic_add_u32 (a plain
        # 32-bit integer add, correct for both signed and unsigned int32 bit patterns --
        # two's-complement addition doesn't care how the result is later interpreted) for
        # int8/int4 precision, gated on ctrl.precision. NOTE: this is a code-level fix only --
        # not yet hardware-validated end-to-end with genuinely signed/large-magnitude int8
        # data (deprioritized; int8/int4 is not a current focus -- see
        # docs/gfx1250_wmma_vgpr_msb_wip_status.md's Phase 57).
        self.merge_e                            = utility_dict_with_default_t(tunable_dict)('merge_e', 0)   # indicate if merge c*y*x for gemm_k (fwd), useful in nhwc fwd
        #  x -(unmerge)-> x0*x1, if set to 1, means cluster first iterate all x1
        # hence stride of x0 should not be x1, but be total number of x divide by x0

        # if self.tensor_layout == "nchwc":
        self.vector_c                           = utility_dict_with_default_t(tunable_dict)('vector_c', 1)
        self.wavefront_size                     = utility_dict_with_default_t(tunable_dict)('wavefront_size', 64)
        self.cumode                             = utility_dict_with_default_t(tunable_dict)('cumode', 0)

        assert type(self.tensor_a_thread_lengths) is list and type(self.tensor_a_cluster_lengths) is list
        assert type(self.tensor_b_thread_lengths) is list and type(self.tensor_b_cluster_lengths) is list
        # assert type(self.opt_1x1) is bool
        assert self.direction in ('fwd', 'bwd', 'wrw')
        assert self.precision in ('fp32', 'fp16', 'bf16', 'int8', 'int4')
        if self.tensor_layout == "nchw":
            assert self.nxb in (1,4,8,16,32,64,128,256)
        elif self.tensor_layout == "nhwc":
            assert self.nxb == 0, 'nhwc now no need have different nxb value'
        elif self.tensor_layout[0:5] == "nchwc":
            assert self.vector_c in (4, 8, 16, 32), 'do not support arbitary vector_c'
        else:
            assert False
        assert self.nxe in (0,1)

        self.wave_size = AMDGPU_WAVE_SIZE
        # TODO: better specify
        if self.fma_type in (IGEMM_GTC_TUNABLE_FMA_TYPE_MAC, IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS):
            if igemm_use_lanegroup_thread_mapping(self):
                self.wave_size                  = self.lanegroup_wave_m * self.lanegroup_wave_n * LANEGROUP_SIZE
                assert self.wave_size in (32, 64)
                assert self.gemm_m_per_block % (self.lanegroup_tile_m * self.lanegroup_wave_m * self.lanegroup_repeat_m) == 0
                assert self.gemm_n_per_block % (self.lanegroup_tile_n * self.lanegroup_wave_n * self.lanegroup_repeat_n) == 0
                waves_per_m = self.gemm_m_per_block // (self.lanegroup_tile_m * self.lanegroup_wave_m * self.lanegroup_repeat_m)
                waves_per_n = self.gemm_n_per_block // (self.lanegroup_tile_n * self.lanegroup_wave_n * self.lanegroup_repeat_n)
                self.block_size                 = waves_per_m * waves_per_n * self.wave_size
            else:
                self.block_size                 = self.gemm_m_level0_cluster * self.gemm_n_level0_cluster * self.gemm_m_level1_cluster * self.gemm_n_level1_cluster

        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            assert self.gemm_m_per_block % (self.wave_tile_m * self.wave_step_m * self.wave_repeat_m) == 0
            assert self.gemm_n_per_block % (self.wave_tile_n * self.wave_step_n * self.wave_repeat_n) == 0
            waves_per_m = self.gemm_m_per_block // (self.wave_tile_m * self.wave_step_m * self.wave_repeat_m)
            waves_per_n = self.gemm_n_per_block // (self.wave_tile_n * self.wave_step_n * self.wave_repeat_n)
            self.block_size                     = waves_per_m * waves_per_n * self.wave_size

        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA:
            self.wave_size                      = 32   # gfx1250 WMMA requires wave32, not a software tiling choice
            assert self.gemm_m_per_block % (self.wmma_tile_m * self.wmma_repeat_m) == 0
            assert self.gemm_n_per_block % (self.wmma_tile_n * self.wmma_repeat_n) == 0
            waves_per_m = self.gemm_m_per_block // (self.wmma_tile_m * self.wmma_repeat_m)
            waves_per_n = self.gemm_n_per_block // (self.wmma_tile_n * self.wmma_repeat_n)
            self.block_size                     = waves_per_m * waves_per_n * self.wave_size

        assert self.block_size == igemm_flatten_list_product(self.tensor_a_cluster_lengths), f"block_size:{self.block_size}, a_cluster_lengths:{self.tensor_a_cluster_lengths}, {self.gemm_m_per_block}x{self.gemm_n_per_block}"
        assert self.block_size == igemm_flatten_list_product(self.tensor_b_cluster_lengths), f"block_size:{self.block_size}, b_cluster_lengths:{self.tensor_b_cluster_lengths}, {self.gemm_m_per_block}x{self.gemm_n_per_block}"

        def _unmerge_x1_from_e(unroll_k, nxe):
            if nxe == 0:
                return unroll_k # not used, 1x1 special
            if unroll_k % nxe == 0:
                return unroll_k // nxe
            return unroll_k     # not used

        if self.direction == 'fwd':
            if self.tensor_layout == 'nchw':
                assert self.gemm_n_per_block % self.nxb == 0
                self.unmerge_sub_n = self.gemm_n_per_block // self.nxb
                self.unmerge_sub_k = 1                          # not used
                self.unmerge_sub_c = _unmerge_x1_from_e(self.gemm_k_per_block, self.nxe)
            elif self.tensor_layout == 'nhwc':
                self.unmerge_sub_n = 1                          # not used
                self.unmerge_sub_k = 1                          # not used
                self.unmerge_sub_c = 1                          # not used
            elif self.tensor_layout[0:5] == "nchwc":
                pass
            else:
                assert False
        elif self.direction == 'bwd':
            if self.tensor_layout == 'nchw':
                assert self.gemm_n_per_block % self.nxb == 0
                self.unmerge_sub_n = self.gemm_n_per_block // self.nxb
                self.unmerge_sub_k = _unmerge_x1_from_e(self.gemm_k_per_block, self.nxe)
                self.unmerge_sub_c = 1                             # not used
            elif self.tensor_layout == 'nhwc':
                self.unmerge_sub_n = 1                          # not used
                self.unmerge_sub_k = 1                          # not used
                self.unmerge_sub_c = 1                          # not used
            else:
                assert False
        else:
            if self.tensor_layout == 'nchw':
                assert self.gemm_k_per_block % self.nxb == 0
                self.unmerge_sub_n = _unmerge_x1_from_e(self.gemm_k_per_block, self.nxb)
                self.unmerge_sub_k = 1
                self.unmerge_sub_c = self.gemm_n_per_block
            elif self.tensor_layout == 'nhwc':
                self.unmerge_sub_n = 1                          # not used
                self.unmerge_sub_k = 1                          # not used
                self.unmerge_sub_c = 1                          # not used

        self.tensor_a_pass_through_interleave_gld = 0 if self.tensor_layout == 'nhwc' else 1
        self.tensor_b_pass_through_interleave_gld = 0 if self.tensor_layout == 'nhwc' else 1

        self.fma_interleave = IGEMM_GTC_FEAT_FMA_INTERLEAVE
        self.local_prefetch_num = 1
        # vector global/lds implicit here
        if self.fma_type in (IGEMM_GTC_TUNABLE_FMA_TYPE_MAC, IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS):
            if igemm_use_lanegroup_thread_mapping(self):
                # register for a,b,c buffer
                dotx_mapping = get_ctrl_dotx_mapping_from_lanegroup_tile(self.gemm_m_per_block, self.gemm_n_per_block,
                                        self.lanegroup_tile_m, self.lanegroup_tile_n, self.lanegroup_wave_m, self.lanegroup_wave_n, self.block_size // self.wave_size,
                                        self.lanegroup_repeat_m, self.lanegroup_repeat_n, self.precision, get_dotx_fma_instruction(mc_get_current().arch_config.arch, self.precision))
                self.local_prefetch_num         = 2 if IGEMM_GTC_FEAT_LOCAL_PREFETCH else 1 # TODO: other local prefetch
                #if dotx_mapping.lanegroup_repeat_n > self.local_prefetch_num:
                self.local_prefetch_num_m = dotx_mapping.lanegroup_repeat_m
                if self.direction == 'fwd':
                    assert self.tensor_a_thread_lengths[1] == self.tensor_b_thread_lengths[1]
                self.num_vgpr_accumulate_c  = dotx_mapping.total_acc_c()

                # TODO: try use prefetch
                self.num_vgpr_accumulate_a  = self.local_prefetch_num_m * dotx_mapping.thread_m()
                self.num_vgpr_accumulate_b  = self.local_prefetch_num * dotx_mapping.thread_n()
            else:
                self.gemm_m_repeat              = self.gemm_m_per_block // (self.gemm_m_per_thread * self.gemm_m_level0_cluster * self.gemm_m_level1_cluster)
                self.gemm_n_repeat              = self.gemm_n_per_block // (self.gemm_n_per_thread * self.gemm_n_level0_cluster * self.gemm_n_level1_cluster)
                # register for a,b,c buffer
                self.num_vgpr_accumulate_c      = (self.gemm_m_repeat * self.gemm_m_per_thread * self.gemm_n_repeat * self.gemm_n_per_thread)
                self.num_vgpr_accumulate_a      = (self.gemm_m_repeat * self.gemm_m_per_thread)
                self.num_vgpr_accumulate_b      = (self.gemm_n_repeat * self.gemm_n_per_thread)

        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            self.local_prefetch_num             = 2 if IGEMM_GTC_FEAT_LOCAL_PREFETCH else 1
            if (self.tensor_a_pass_through and self.wave_repeat_n == 2) or (self.tensor_b_pass_through and self.wave_repeat_m == 2):
                self.local_prefetch_num         = 1
            # register for a,b,c buffer
            xdlops_mapping = get_ctrl_xdlops_mapping_from_wave_tile(self.gemm_m_per_block, self.gemm_n_per_block, self.wave_tile_m, self.wave_tile_n, self.wave_tile_k, 
                    self.wave_repeat_m, self.wave_repeat_n, self.wave_step_m, self.wave_step_n, self.block_size // self.wave_size, self.precision)
            self.num_agpr_accumulate_c          = xdlops_mapping.total_acc_c()
            assert self.num_agpr_accumulate_c == self.gemm_m_per_block * self.gemm_n_per_block // self.block_size, f"block_size:{self.block_size}, {self.gemm_m_per_block}x{self.gemm_n_per_block}x{self.gemm_k_per_block}"
            self.num_vgpr_accumulate_a          = self.wave_step_m * self.wave_repeat_m * xdlops_mapping.inst_mfma.num_v_a * self.local_prefetch_num
            self.num_vgpr_accumulate_b          = self.wave_step_n * self.wave_repeat_n * xdlops_mapping.inst_mfma.num_v_b * self.local_prefetch_num

        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA:
            # Phase 22 (VGPR-level prefetch): optional, defaults to 1 (today's exact
            # byte-identical single-buffered v_a/v_b, every existing config unaffected).
            # When 2, doubles num_vgpr_accumulate_a/b below, and wmma_main_loop.py issues
            # the NEXT k-substep's shared_load into the other slot before the CURRENT
            # substep's compute -- mirrors XDLOPS's local_prefetch_num=2, but intra-K-
            # substep, not cross-main-loop-iteration (see docs/gfx1250_wmma_layout.md).
            # Only meaningful when gemm_k_per_block > inst_wmma.k (k-sub-loop in use);
            # each kernel file's wmma_main_loop.py emit() asserts this at codegen time.
            # Read here (not where main_loop_interleave/async_global_load are, above) since
            # this __init__'s later shared `self.local_prefetch_num = 1` default (every
            # fma_type) runs in between and would otherwise clobber an earlier read.
            self.local_prefetch_num = utility_dict_with_default_t(tunable_dict)('local_prefetch_num', 1)
            if self.local_prefetch_num == 2:
                assert self.main_loop_interleave == 0, "local_prefetch_num=2 and main_loop_interleave are mutually exclusive for now"
            # Phase 24 (F16-accumulate WMMA): optional, defaults to 0 (today's f32-accumulate,
            # every existing config unaffected). When 1, selects the WMMA variant that
            # accumulates directly in fp16 (num_v_c=4, half the VGPRs of f32-accumulate's
            # num_v_c=8) -- fp16 only; bf16/int8 have no equivalent on this ISA (see
            # wmma.py's v_wmma_f16_16x16x32_f16 comment). Only affects the non-atomic
            # epilogue branch (coalescing_store_wmma.py) -- there is no packed-fp16
            # atomic-add on this ISA, so gemm_k_global_split stays f32-accumulate regardless.
            self.wmma_acc_f16 = utility_dict_with_default_t(tunable_dict)('wmma_acc_f16', 0)
            if self.wmma_acc_f16:
                assert self.precision == 'fp16', \
                    f"wmma_acc_f16=1 is only implemented for fp16 (got precision={self.precision}) -- " \
                    f"bf16/int8 have no equivalent halved-accumulate WMMA variant, see docs/gfx1250_wmma_layout.md's Phase 24"
                # epilogue_lds_pad's pad amount (4 elements) was derived for 4-byte-wide
                # accumulator elements specifically (breaks bank conflicts while preserving
                # 16-byte alignment for ds_read_b128/global_store_dwordx4). f16acc's LDS
                # elements are 2 bytes wide -- the SAME pad constant would misalign the
                # narrower ds_read_b64/global_store_dwordx2 the gather uses, and a correct
                # 2-byte-element pad amount hasn't been derived. Mutually exclusive for now.
                assert not self.epilogue_lds_pad, \
                    "wmma_acc_f16 and epilogue_lds_pad are mutually exclusive for now -- the Phase 23 pad constant was derived for 4-byte elements, not f16acc's 2-byte elements, see docs/gfx1250_wmma_layout.md's Phase 24"
                # The WMMA instruction (and its num_v_c/accumulator width) is selected ONCE
                # per kernel, shared by both epilogue branches -- there is no way for the
                # atomic (gemm_k_global_split) branch to keep reading f32-width VGPRs while
                # the non-atomic branch reads packed f16 ones from the SAME v_c allocation.
                # coalescing_store_wmma.py's atomic branch was never adapted for a packed
                # accumulator (it has no packed atomic-add to target anyway, per Phase 19),
                # so this combination is unsupported, not just unimplemented -- block it.
                assert not self.gemm_k_global_split, \
                    "wmma_acc_f16 and gemm_k_global_split are mutually exclusive -- the atomic epilogue branch was never adapted to read a packed f16 accumulator, see docs/gfx1250_wmma_layout.md's Phase 24"
            # Phase 27 (BF16-accumulate WMMA): mirrors wmma_acc_f16 exactly, using
            # v_wmma_bf16_16x16x32_bf16 instead of v_wmma_f16_16x16x32_f16. A separate flag
            # (not a shared "wmma_acc_narrow" concept) since each is precision-gated to its
            # own dtype and independently mutually exclusive with gemm_k_global_split/
            # epilogue_lds_pad for the identical reasons -- but see below, NOT excluded from
            # wmma_m_tail/wmma_n_tail: those EXEC-mask guards operate on lane/row/column
            # indices, not element width, so they compose with a 2-byte-packed accumulator
            # with no changes needed (this is the actual motivation for this phase -- see
            # docs/gfx1250_wmma_layout.md's Phase 27).
            self.wmma_acc_bf16 = utility_dict_with_default_t(tunable_dict)('wmma_acc_bf16', 0)
            if self.wmma_acc_bf16:
                assert self.precision == 'bf16', \
                    f"wmma_acc_bf16=1 is only implemented for bf16 (got precision={self.precision})"
                assert not self.epilogue_lds_pad, \
                    "wmma_acc_bf16 and epilogue_lds_pad are mutually exclusive for now -- see wmma_acc_f16's identical Phase 23/24 reasoning"
                assert not self.gemm_k_global_split, \
                    "wmma_acc_bf16 and gemm_k_global_split are mutually exclusive -- the atomic epilogue branch was never adapted to read a packed accumulator, see wmma_acc_f16's identical Phase 24 reasoning"
                assert not self.wmma_acc_f16, "wmma_acc_bf16 and wmma_acc_f16 are mutually exclusive (different precisions)"
            # Phase 34 (packed-bf16 atomics): atomic path only, bf16 precision only (packed
            # bf16 atomics need bf16-native memory, and the accuracy tradeoff of packing was
            # only measured for bf16 -- see docs/gfx1250_wmma_layout.md's Phase 34). Halves
            # the number of actual atomic ops the K-split epilogue issues, at the cost of
            # bf16 (not fp32) precision on the K-split reduction itself. Defined here (before
            # wmma_m_tail/wmma_n_tail) since wrw's tail tunables below need to reference it
            # in their own mutual-exclusion asserts.
            self.atomic_pack_bf16 = utility_dict_with_default_t(tunable_dict)('atomic_pack_bf16', 0)
            if self.atomic_pack_bf16:
                assert self.gemm_k_global_split, \
                    "atomic_pack_bf16 only applies to the atomic (gemm_k_global_split) epilogue, see docs/gfx1250_wmma_layout.md's Phase 34"
                assert self.precision == 'bf16', \
                    "atomic_pack_bf16 is only implemented for bf16 precision so far, see docs/gfx1250_wmma_layout.md's Phase 34"
                assert not self.atomic_cascade, \
                    "atomic_pack_bf16 and atomic_cascade are mutually exclusive for now -- not tested together"
            # Phase 41 (gsplit_stagger): optional, defaults to 0 (today's behavior, every
            # existing config unaffected). When 1, emits one S_SLEEP_VAR at kernel entry
            # (right after blockIdx.z is decoded, before any of the group-decode/pointer-
            # offset work) sleeping for (bz mod 128)*~64 cycles. Hypothesis: with many
            # split-K shards launched close together in wall-clock time (this project's own
            # wrw benchmark shapes observed up to 300+ splits), every shard's very first
            # main-loop global load lands at the SAME relative offset within its own
            # (disjoint) K-slice -- if that offset's low address bits happen to alias onto
            # the same DRAM channel/bank subset across shards, the first iteration's loads
            # could burst-contend even though each shard's overall address range is
            # distinct. Spreading each shard's start in TIME (not touching which data it
            # reads, its tile-visitation order, or K-tail masking at all) is a purely
            # timing-side perturbation with no correctness surface -- see
            # docs/gfx1250_optimization_backlog.md for the honest status: this project
            # could not find a working reference for this idea in hipconv's actual source
            # after a real search, so it is MISA's own untested hypothesis, not a ported
            # technique; A/B measurement decides whether it earns a permanent home.
            self.gsplit_stagger = utility_dict_with_default_t(tunable_dict)('gsplit_stagger', 0)
            if self.gsplit_stagger:
                assert self.gemm_k_global_split, \
                    "gsplit_stagger only applies to the atomic (gemm_k_global_split) epilogue -- there is only one shard otherwise, nothing to stagger"
            # Phase 59 (experimental): skip the LDS-reshuffle epilogue entirely and store each
            # v_c element directly via global_store_dword. 16 consecutive lanes cover 16
            # consecutive columns per wmma_mapping.py's lane%16->column derivation, so the
            # half-wave's scalar stores are already coalesced at the memory controller level.
            # FlyDSL and MISA's own coalescing_store_wmma.py docstring confirm this adjacency.
            # Only applies to the non-atomic (non-split-K) epilogue path.
            self.direct_store = utility_dict_with_default_t(tunable_dict)('direct_store', 0)
            # Phase 59: Phase 4 perf A/B across 5 fwd shapes showed direct_store is 1.07-1.25x
            # faster than LDS-reshuffle on all tested shapes (n=8,c=128,H=30,W=40,k=128/512,
            # n=4,c=256,H=56,W=56,k=256) with zero regressions — strong evidence for the
            # new epilogue. LEFT AS OPT-IN (default 0) because LDS-reshuffle may win on
            # memory-bandwidth-bound shapes or future precision/tile combinations not yet
            # tested. The master config union should include both direct_store=0 and
            # direct_store=1 variants so the driver's per-shape search picks the best.
            # Phase 25 (GEMM_M tail): optional, defaults to 0 (today's exact-gemm_m-multiple-
            # only requirement, every existing config unaffected). When 1, the driver's
            # tunable_is_valid() allows gemm_m % gemm_m_per_block != 0, and this kernel emits
            # extra masking: the A-operand v_flag computation also checks the lane's absolute
            # flattened row index against the real (unpadded) GEMM_M, and the epilogue
            # EXEC-masks stores whose absolute row index is out of range. fwd (Phase 25) and
            # bwd (Phase 26a) so far. Phase 35 adds wrw: unlike fwd/bwd, wrw's split-K
            # (gemm_k_global_split) is its PRIMARY path, not an edge case, so wrw's M-tail
            # must (and does) also mask the atomic epilogue branch -- no exclusion needed.
            self.wmma_m_tail = utility_dict_with_default_t(tunable_dict)('wmma_m_tail', 0)
            if self.wmma_m_tail:
                assert self.direction in ('fwd', 'bwd', 'wrw'), "wmma_m_tail is only implemented for fwd/bwd/wrw so far, see docs/gfx1250_wmma_layout.md's Phase 25/26/35"
                if self.direction != 'wrw':
                    assert not self.gemm_k_global_split, \
                        "wmma_m_tail and gemm_k_global_split are mutually exclusive for now -- the atomic epilogue branch has no M-tail masking, see docs/gfx1250_wmma_layout.md's Phase 25"
                else:
                    assert not self.atomic_pack_bf16, \
                        "wmma_m_tail and atomic_pack_bf16 are mutually exclusive for now -- both need the same coalescing_store v_tmp3/v_tmp4 scratch slots for unrelated purposes, and tail-masking a packed cross-lane exchange (a lane's partner may be in-range while it isn't) hasn't been reviewed, see docs/gfx1250_wmma_layout.md's Phase 35"
            # Phase 26b (GEMM_N tail): analogous to wmma_m_tail but for GEMM_N -- the
            # B-operand load gains a persistent (kernel-lifetime-constant, not per-tap) flag
            # checking this lane's absolute column against the real GEMM_N, and the epilogue
            # gets a second EXEC-mask guard chained after the M-tail one (wave32 v_cmpx
            # intersects with the already-narrowed EXEC). fwd only so far, Phase 35 adds wrw
            # (same split-K-must-compose reasoning as wmma_m_tail above -- wrw's atomic
            # epilogue is scalar-per-element already, so the vectorized-store granularity
            # issue below that forces fwd's gemm_n%4==0 restriction does not apply to wrw's
            # atomic branch). NOTE: the driver's tunable_is_valid() additionally requires the
            # real gemm_n to be a multiple of 4 for fwd/bwd's non-atomic epilogue -- that
            # epilogue's store is 4-elements-per-group vectorized, and the guard only checks a
            # group's first column, so a group straddling a non-multiple-of-4 tail would
            # silently write past the real gemm_n (confirmed on hardware). See
            # docs/gfx1250_wmma_layout.md's Phase 26b. bwd (new): B/weight is TRANSPOSED
            # (see igemm_bwd_gtc_wmma_nhwc_t's docstring) -- each lane's own chunk load
            # spans multiple consecutive N-values (not a single fixed column like fwd's
            # natural-layout B), so bwd's N-tail masking is a fine-grained per-dword AND-mask
            # applied to the loaded data, NOT the simple per-lane EXEC-mask fwd/wrw use --
            # a different (new) mechanism sharing only the tunable name and the driver-side/
            # epilogue-side plumbing.
            self.wmma_n_tail = utility_dict_with_default_t(tunable_dict)('wmma_n_tail', 0)
            if self.wmma_n_tail:
                assert self.direction in ('fwd', 'bwd', 'wrw'), "wmma_n_tail is only implemented for fwd/bwd/wrw so far, see docs/gfx1250_wmma_layout.md's Phase 26b/35"
                if self.direction == 'wrw':
                    assert not self.atomic_pack_bf16, \
                        "wmma_n_tail and atomic_pack_bf16 are mutually exclusive for now -- same reasoning as wmma_m_tail, see docs/gfx1250_wmma_layout.md's Phase 35"
                    # Phase 51: only matters when wrw takes the shared NON-atomic epilogue
                    # (gemm_k_global_split=0) -- the atomic path (scalar-per-element already)
                    # is unaffected by this regardless of accumulate width. See the identical
                    # assert/reasoning in the fwd/bwd branch below.
                    assert self.gemm_k_global_split or not (self.wmma_acc_f16 or self.wmma_acc_bf16), \
                        "wmma_n_tail's per-element epilogue masking is not yet implemented for wmma_acc_f16/bf16acc's packed 2-elements-per-register layout, see docs/gfx1250_wmma_layout.md's Phase 51"
                else:
                    assert not self.gemm_k_global_split, \
                        "wmma_n_tail and gemm_k_global_split are mutually exclusive for now -- the atomic epilogue branch has no N-tail masking, see docs/gfx1250_wmma_layout.md's Phase 26b"
                    assert not self.async_global_load, \
                        "wmma_n_tail and async_global_load are mutually exclusive for now -- global_load_async_to_lds_b128's masking was only ever validated for the A operand, see docs/gfx1250_wmma_layout.md's Phase 13/26b"
                    assert self.gemm_n_per_block // self.block_size == 1, \
                        "wmma_n_tail requires row_repeat_b == 1 -- rows 1+ have no flag of their own, see docs/gfx1250_wmma_layout.md's Phase 26b"
                    # Phase 51: the new per-element (not per-vwo-group) epilogue masking that
                    # lifts the gemm_n%4==0 restriction (coalescing_store_wmma.py) was only
                    # implemented for the standard f32-accumulate layout, where each of the
                    # vector_write_out elements occupies one whole VGPR -- wmma_acc_f16/
                    # bf16acc pack TWO elements per register (see the scatter's ds_write_b16/
                    # _d16_hi split), a genuinely different addressing scheme not audited
                    # here. No existing config combines the two, so this is a forward-looking
                    # guard, not a real regression.
                    assert not (self.wmma_acc_f16 or self.wmma_acc_bf16), \
                        "wmma_n_tail's per-element epilogue masking is not yet implemented for wmma_acc_f16/bf16acc's packed 2-elements-per-register layout, see docs/gfx1250_wmma_layout.md's Phase 51"
            # Phase 53 (chunked epilogue): non-atomic path only. 0 (default) = today's
            # one-shot design (stage the whole macro-tile in LDS at once), byte-identical
            # for every existing config -- this is what hard-caps the macro-tile at 128x128
            # (see docs/gfx1250_wmma_layout.md's Phase 52). 1 = reuse a small,
            # tile-size-invariant LDS region across wave_repeat_m groups instead, unlocking
            # bigger macro-tiles (e.g. the new 256x256/256x128 wmma_mapping.py entries)
            # within the same 64KB/workgroup LDS limit. This alone only solves the LDS
            # ceiling -- gfx1250's real, hardware-verified 256-VGPR/wave budget (NOT the
            # 1024 figure Phase 52 originally cited from external research, which turned
            # out to describe a capability this project's plain v[N]-addressed assembly
            # doesn't use -- see Phase 53's correction) independently caps how big a tile's
            # ACCUMULATOR can be, regardless of LDS. wmma_acc_f16/bf16acc (halves the
            # accumulator width) is therefore a load-bearing combination for any tile
            # bigger than 128x128, not an optional extra -- supported here. Narrowest
            # correctness-first slice: not yet combined with wmma_m_tail/wmma_n_tail
            # (masking interaction with per-group chunking not audited) or
            # gemm_k_global_split (this flag only touches the non-atomic branch -- the
            # atomic path has no LDS staging or tile-size ceiling to begin with). See
            # docs/gfx1250_wmma_layout.md's Phase 53.
            self.wmma_epilogue_chunked = utility_dict_with_default_t(tunable_dict)('wmma_epilogue_chunked', 0)
            if self.wmma_epilogue_chunked:
                assert self.direction in ('fwd', 'bwd'), \
                    "wmma_epilogue_chunked is only implemented for fwd/bwd so far, see docs/gfx1250_wmma_layout.md's Phase 53"
                # Phase 68 (2026-09-02, PERF-002): relaxed from an outright ban -- see
                # coalescing_store_wmma.py's _emit_chunked_non_atomic_store for the actual
                # masking wiring this needed (the gather phase's per-pass loop gained the
                # same v_cmpx_gt_u32/v_cmpx_gt_i32 EXEC-mask guards the unchunked gather
                # already used, keyed off the same v_tmp3/v_tmp4 absolute row/col state,
                # recomputed per-group/per-pass from v_tid like every other chunked
                # address value). Hardware-validated (-V 1, valid:y) standalone at 128x128
                # and combined with wmma_acc_high_bank at 256x256/256x128 (non-divisible
                # gemm_m).
                assert not self.gemm_k_global_split, \
                    "wmma_epilogue_chunked only applies to the non-atomic epilogue -- gemm_k_global_split's atomic path has no LDS staging to chunk"
            # Phase 54 (VGPR-MSB): moves ONLY the accumulator (v_c) into a second,
            # independently-addressed 256-VGPR bank (S_SET_VGPR_MSB, doc Sec 3.3.2.3),
            # freeing its entire footprint out of the plain 0-255 address space every
            # other register (v_a/v_b/global-load buffers/addresses/temps) still lives
            # in. Confirmed via a direct llvm-mc assemble+objdump disassemble round-trip
            # that this project's toolchain fully supports the mechanism -- an earlier
            # investigation tested it wrong (tried literal v256+ operand syntax, which
            # was never real) and wrongly concluded it was unusable; see
            # docs/gfx1250_wmma_layout.md's Phase 53 correction and Phase 54. This is
            # the actual fix for the VGPR wall Phase 53 hit trying to grow past 128x128
            # by fitting everything in one bank. Scope of this phase: v_c ITSELF must
            # still fit inside a single 256-register bank (asserted below once
            # num_vgpr_accumulate_c is known) -- v_c spanning multiple banks (needing
            # per-c_index-range MSB switching inside the main loop's WMMA-issue inner
            # loop) is a natural follow-on, not implemented here. Default 0 = today's
            # exact byte-identical behavior (v_c allocated from the same flat 0-255
            # sequencer as everything else).
            self.wmma_acc_high_bank = utility_dict_with_default_t(tunable_dict)('wmma_acc_high_bank', 0)
            if self.wmma_acc_high_bank:
                # coalescing_store_wmma.py's chunked non-atomic path
                # (_emit_chunked_non_atomic_store) AND its plain (unchunked) non-atomic
                # path are both wired for v_c living in a separate bank. The plain atomic
                # (gemm_k_global_split) path and atomic_pack_bf16 are not -- both still
                # read v_c directly (global_atomic_add_f32/global_atomic_pk_add_bf16,
                # v_permlane_xor_b32) with no tracker awareness.
                assert not self.gemm_k_global_split, \
                    "wmma_acc_high_bank's atomic epilogue path (gemm_k_global_split) is not yet wired for v_c living in a separate VGPR bank"
                # Phase 68 (2026-09-02, PERF-002): wmma_m_tail/wmma_n_tail's masked
                # slow-path store (indexing v_gather+i directly, bypassing
                # v_gather_range) only ever reads/writes v_gather/v_tmp3/v_tmp4 -- pure
                # bank-0 scratch registers, never v_c -- and the ENTIRE gather phase
                # (address computation and every store variant) runs with the MSB
                # tracker's src1 pinned to bank 0 for its whole duration (see
                # coalescing_store_wmma.py's __call__, right before the gather section
                # begins), so it composes with v_c living in bank 1 with no code changes
                # needed. Hardware-validated (-V 1, valid:y) on a 128x128 bf16 fwd config
                # with wmma_acc_high_bank=1+wmma_m_tail=1 standalone.
            # Phase 35 (GEMM_K tail, wrw): no precedent anywhere else in this codebase --
            # unlike the TDM-hardware-OOB K-tail fwd/1x1 uses (Phase 31), this is a genuine
            # software EXEC-mask mechanism, since wrw doesn't use TDM. Composes with
            # gemm_k_global_split by construction (only the LAST split-K shard's loop range
            # gets extended to cover the true, non-padded gemm_k -- see driver's
            # gemm_k_tail/gemm_k_num_splits kernarg fields). See docs/gfx1250_wmma_layout.md's
            # Phase 35. bwd (new): needs a genuinely different masking mechanism from
            # wrw's: bwd's A(grad_output) operand loads gemm_k_per_block elements
            # CONTIGUOUSLY per lane (one lane, one M-row, all K at once) -- unlike wrw's
            # per-lane-per-K-row B addressing, EXEC can't gate a sub-range within one
            # lane's own load, so A's K-tail uses the same new fine-grained per-dword
            # AND-mask as bwd's N-tail (see wmma_n_tail's docstring above). B's K-tail (bwd's
            # B is TRANSPOSED, so row_local really is a fixed per-lane K position) stays the
            # simple per-lane EXEC-mask case.
            self.wmma_k_tail = utility_dict_with_default_t(tunable_dict)('wmma_k_tail', 0)
            if self.wmma_k_tail:
                assert self.direction in ('wrw', 'bwd', 'fwd'), "wmma_k_tail is only implemented for wrw/bwd/fwd so far, see docs/gfx1250_wmma_layout.md's Phase 35/36/38"
                if self.direction == 'bwd':
                    assert not self.gemm_k_global_split, \
                        "wmma_k_tail is not implemented together with gemm_k_global_split for bwd yet -- bwd got split-K support in Phase 48 but without a last-shard remainder clamp (no wrw-style s_gemm_k_tail/s_gemm_k_num_splits pattern ported yet), see docs/gfx1250_wmma_layout.md's Phase 48"
                    # Phase 42: mirrors fwd's identical mutual-exclusion -- TDM already
                    # handles K-tail via hardware OOB for the 1x1-only case.
                    assert not self.tdm_global_load, \
                        "wmma_k_tail (new, non-TDM) and tdm_global_load are mutually exclusive -- TDM already has its own K-tail mechanism"
                if self.direction == 'fwd':
                    # fwd (new): this is a genuinely DIFFERENT mechanism from TDM's K-tail
                    # (Phase 31/37) -- TDM already handles K-tail via hardware OOB for the
                    # 1x1-only case; this new software mechanism is for multi-tap convs,
                    # which TDM was never extended to cover. Mutually exclusive by
                    # construction (also asserted kernel-side).
                    assert not self.tdm_global_load, \
                        "wmma_k_tail (new, non-TDM) and tdm_global_load are mutually exclusive -- TDM already has its own K-tail mechanism"
            # Phase 63: replace shared_load_b_functor's (bwd) / shared_load_a_functor's and
            # shared_load_b_functor's (wrw) manual per-element LDS-transpose read+pack loop
            # (ds_read_u16/u8 -> s_wait_dscnt -> v_mov_b32 -> v_lshl_or_b32, ~160
            # instructions/K-iteration, see docs/gfx1250_rocprof_profiling.md's Finding 7)
            # with the native ds_load_tr16_b128 hardware transpose-load instruction (2
            # calls/wave_repeat_n step, zero packing/waits). fp16/bf16 only -- no fp32
            # variant of this instruction exists (element sizes supported are 16/8/6/4-bit).
            # See docs/gfx1250_wmma_layout.md's Phase 63 for the empirically-confirmed
            # hardware addressing mechanism this relies on (reverse-engineered via a
            # standalone hardware probe, since the ISA doc's exact per-lane semantics are
            # only in an unextracted diagram).
            # Phase 64: composes freely with wmma_m_tail/wmma_n_tail/wmma_k_tail/
            # gemm_k_global_split -- confirmed by code inspection that all tail/split-K
            # masking happens at LDS WRITE time (global_load's v_flag + shared_store's
            # tail-dword mask), not LDS READ time, so shared_load_b/a_functor (whichever
            # mechanism) just reads whatever's already correctly-masked in LDS regardless.
            # The original narrower asserts here were overly conservative -- removed after
            # this review, hardware-validated per combination (see Phase 64 in
            # docs/gfx1250_wmma_layout.md).
            # Phase 67: promoted to DEFAULT-ON for every bwd/wrw fp16/bf16 config (same
            # treatment as Phase 64's wait-batching) -- every combination tried
            # (direct_store, async_global_load, lds_double_buffer, main_loop_interleave,
            # epilogue_lds_pad, wmma_setprio, tdm_global_load(+direct), saddr_global_load,
            # gemm_k_global_split(+wmma_setprio), local_prefetch_num=2(+wmma_acc_bf16),
            # group_count>1, multi-K-block, both tile shapes) hardware-validated `valid:y`
            # with zero regression; the only two failures found (wrw's k2x/gemm_k_per_block=64
            # `interleave`/`k2x_dbuf` sections) reproduce identically with ds_load_tr_b=0,
            # confirmed via A/B against the unmodified config -- a pre-existing k2x bug,
            # unrelated to this change. `wrw_streamk` is excluded from the default (checked
            # via the raw tunable_dict, not self.wrw_streamk, since that field is parsed
            # later in __init__) -- stream-K's history of faulting tunables (see the
            # benchmark-script-validn-trap finding) makes it the one combination not
            # exercised here; a config can still explicitly opt out (or into, for fwd/fp32/
            # int8 it stays a no-op) by setting ds_load_tr_b explicitly.
            _dstrb_default = 1 if (self.direction in ('bwd', 'wrw') and self.precision in ('fp16', 'bf16')
                                    and not tunable_dict.get('wrw_streamk', 0)) else 0
            self.ds_load_tr_b = utility_dict_with_default_t(tunable_dict)('ds_load_tr_b', _dstrb_default)
            if self.ds_load_tr_b:
                assert self.direction in ('bwd', 'wrw'), "ds_load_tr_b is bwd/wrw only (Phase 63/64) -- fwd's operands aren't LDS-transposed to begin with"
                assert self.precision in ('fp16', 'bf16'), "ds_load_tr16_b128 is a 16-bit-element instruction only -- no fp32 variant exists"
            # Phase 35 (hipconv-style reduction-kernel epilogue): replaces the atomic epilogue
            # entirely for wrw's split-K path -- each shard writes a plain, non-atomic store
            # into its own disjoint slice of a workspace buffer (num_splits x output_size),
            # then a separate reduction kernel (driver/gpu_tensor_cast/gpu_tensor_cast.cpp)
            # sums the partitions into the real output. No atomics anywhere in the main
            # kernel. This is a buffer-layout-changing tunable (the kernel writes partials,
            # not final output) so -- unlike wmma_m_tail/wmma_n_tail/wmma_k_tail, which are
            # pure masking -- it IS folded into the kernel name (see
            # igemm_gtc_encode_kernel_name below). See docs/gfx1250_wmma_layout.md's Phase 35.
            self.wrw_reduction_kernel = utility_dict_with_default_t(tunable_dict)('wrw_reduction_kernel', 0)
            if self.wrw_reduction_kernel:
                assert self.direction == 'wrw', "wrw_reduction_kernel is only implemented for wrw, see docs/gfx1250_wmma_layout.md's Phase 35"
                assert self.gemm_k_global_split, \
                    "wrw_reduction_kernel only makes sense when gemm_k_global_split is set (it replaces that path's atomic epilogue)"
                assert not self.atomic_pack_bf16, \
                    "wrw_reduction_kernel and atomic_pack_bf16 are mutually exclusive -- this mode has no atomics at all"
                assert not self.wmma_acc_f16 and not self.wmma_acc_bf16, \
                    "wrw_reduction_kernel keeps the D-operand plain fp32 (matching the workspace's fixed fp32 layout) -- not combined with the packed-accumulator precision tricks for now"
            # Phase 58 (wrw_streamk): a persistent-kernel / Stream-K proof of mechanism --
            # see docs/gfx1250_streamk_design.md for the full design and rationale. Scoped
            # deliberately narrow for a first pass: grid.x/grid.y (tile assignment) are
            # completely unchanged -- only the K-split axis (grid.z) becomes dynamic. Instead
            # of grid.z == the chosen split count (today's model, up to 4096 tiny
            # simultaneously-launched workgroups -- the documented contention-sensitivity
            # culprit in docs/gfx1250_vendor_benchmark_vs_miopen.md), the driver launches a
            # small, constant-size grid.z; each of those persistent workgroups loops,
            # atomically claiming the next not-yet-processed K-shard index from a per-tile
            # global counter (one counter slot per (bx,by) output tile, zeroed before each
            # dispatch) until the shard count is exhausted. Requires gemm_k_global_split=1
            # (builds on top of its existing atomic epilogue, per-shard K-range derivation,
            # and kernarg fields -- this tunable only changes WHERE the shard index comes
            # from, not how a shard is processed once claimed). Requires nxe==0 (single-tap,
            # y=x=1 -- same restriction every existing gsplit config already has): the
            # persistent loop replaces the tap loop's single (iy=0,ix=0) body with N
            # dynamically-claimed-shard iterations of that same body shape (zero v_c, reset
            # addresses, run the main loop, atomic-epilogue-store) -- generalizing this to
            # real multi-tap (y*x>1) filters is future work, not attempted here.
            self.wrw_streamk = utility_dict_with_default_t(tunable_dict)('wrw_streamk', 0)
            if self.wrw_streamk:
                assert self.gemm_k_global_split, \
                    "wrw_streamk builds on top of the atomic (gemm_k_global_split) epilogue -- set gemm_k_global_split=1 too"
                assert self.nxe == 0, \
                    "wrw_streamk's first pass only supports nxe==0 (single-tap, y=x=1) -- see docs/gfx1250_streamk_design.md"
                assert not self.gsplit_stagger, \
                    "gsplit_stagger staggers simultaneously-launched shards' first burst -- meaningless once shards are claimed at different real times by a persistent loop, and untested together"
                assert not self.wmma_k_tail, \
                    "wrw_streamk's first pass doesn't yet compose with wmma_k_tail's last-shard-remainder-extension logic -- untested, not attempted"
                assert not self.tdm_global_load, \
                    "wrw_streamk's first pass doesn't compose with tdm_global_load (both declare s_wave_id independently) -- untested, not attempted"
                # Phase 58 Approach C: wrw_reduction_kernel is now compatible with wrw_streamk --
# the per-iteration workspace-shard offset is computed from s_streamk_tile_idx
# inside emit_kernel_streamk_loop() instead of from the static blockIdx.z in the
# prologue. See igemm_wrw_gtc_wmma_nhwc.py's emit_kernel_streamk_loop() and
# docs/gfx1250_streamk_design.md.
            if self.wmma_acc_f16:
                wmma_mapping_key = self.precision + '_f16acc'
            elif self.wmma_acc_bf16:
                wmma_mapping_key = self.precision + '_bf16acc'
            else:
                wmma_mapping_key = self.precision
            wmma_mapping = get_ctrl_wmma_mapping_from_wave_tile(self.gemm_m_per_block, self.gemm_n_per_block, self.wmma_tile_m, self.wmma_tile_n,
                    self.wmma_repeat_m, self.wmma_repeat_n, self.block_size // self.wave_size, wmma_mapping_key)
            self.num_vgpr_accumulate_c          = wmma_mapping.total_acc_c()
            self.num_vgpr_accumulate_a          = self.wmma_repeat_m * wmma_mapping.inst_wmma.num_v_a * self.local_prefetch_num
            self.num_vgpr_accumulate_b          = self.wmma_repeat_n * wmma_mapping.inst_wmma.num_v_b * self.local_prefetch_num
            if self.wmma_acc_high_bank:
                assert self.num_vgpr_accumulate_c <= 256, \
                    f"wmma_acc_high_bank (Phase 54) only moves v_c into a SINGLE extra bank -- " \
                    f"num_vgpr_accumulate_c:{self.num_vgpr_accumulate_c} must fit within one 256-register " \
                    f"bank; a bigger accumulator needs multi-bank MSB switching inside the main loop, not implemented"
            # COR-001 (2026-09-02): fp32 WMMA tunables measured `valid:n` against the GPU
            # naive-conv reference on real gfx1250 hardware when single-buffered --
            # fp32's WMMA A/B operand loads are 4x wider than fp16/bf16's per K-element
            # (4 bytes vs 1), and without LDS double-buffering the main loop can't hide
            # that load latency behind compute, corrupting results on real hardware (not
            # just a perf regression). precision/fma_type/lds_double_buffer are all
            # already set above (lines 215/246/396) by this point. Fail loudly here, at
            # codegen time, rather than silently on hardware.
            if self.precision == 'fp32':
                assert self.lds_double_buffer == 1, \
                    f"fp32 WMMA tunables require lds_double_buffer=1 (see COR-001) -- " \
                    f"gemm_m_per_block:{self.gemm_m_per_block}x{self.gemm_n_per_block}x{self.gemm_k_per_block} is missing it"

        self.global_prefetch_a_num              = 2 if self.tensor_a_pass_through and not self.tensor_a_pass_through_interleave_gld else 1
        self.global_prefetch_b_num              = 2 if self.tensor_b_pass_through and not self.tensor_b_pass_through_interleave_gld else 1

        self.num_global_load_a                  = igemm_flatten_list_product(self.tensor_a_thread_lengths)
        self.num_global_load_b                  = igemm_flatten_list_product(self.tensor_b_thread_lengths)

        if self.fma_type in (IGEMM_GTC_TUNABLE_FMA_TYPE_MAC, IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS):
            if igemm_use_lanegroup_thread_mapping(self):
                gemm_msg = f"gemm_m_per_block:{self.gemm_m_per_block} - {self.lanegroup_tile_m}x{self.lanegroup_wave_m}x{self.lanegroup_repeat_m}, gemm_n_per_block:{self.gemm_n_per_block} - {self.lanegroup_tile_n}x{self.lanegroup_wave_n}x{self.lanegroup_repeat_n}, gemm_k_per_block:{self.gemm_k_per_block}"
            else:
                gemm_msg = f"gemm_m_per_block:{self.gemm_m_per_block} - {self.gemm_m_per_thread}x{self.gemm_m_level0_cluster}x{self.gemm_m_level1_cluster}, gemm_n_per_block:{self.gemm_n_per_block} - {self.gemm_n_per_thread}x{self.gemm_n_level0_cluster}x{self.gemm_n_level1_cluster}, gemm_k_per_block:{self.gemm_k_per_block}"
        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            gemm_msg = f"gemm_m_per_block:{self.gemm_m_per_block} - {self.wave_tile_m}x{self.wave_step_m}x{self.wave_repeat_m}, gemm_n_per_block:{self.gemm_n_per_block} - {self.wave_tile_n}x{self.wave_step_n}x{self.wave_repeat_n}, gemm_k_per_block:{self.gemm_k_per_block}"
        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA:
            gemm_msg = f"gemm_m_per_block:{self.gemm_m_per_block} - {self.wmma_tile_m}x{self.wmma_repeat_m}, gemm_n_per_block:{self.gemm_n_per_block} - {self.wmma_tile_n}x{self.wmma_repeat_n}, gemm_k_per_block:{self.gemm_k_per_block}"

        assert self.num_global_load_a * self.block_size == self.gemm_m_per_block * self.gemm_k_per_block, gemm_msg
        assert self.num_global_load_b * self.block_size == self.gemm_n_per_block * self.gemm_k_per_block, gemm_msg

        # LDS size
        self.lds_pad_m, self.lds_pad_n = self.get_lds_pad() # LDS pad
        self.lds_a                     = amdgpu_precision_data_byte(self.precision) * self.gemm_k_per_block * self.gemm_m_per_block if not self.tensor_a_pass_through else 0
        self.lds_b                     = amdgpu_precision_data_byte(self.precision) * self.gemm_k_per_block * self.gemm_n_per_block if not self.tensor_b_pass_through else 0
        self.lds_a                     = int(self.lds_a)
        self.lds_b                     = int(self.lds_b)
        self.lds_a_np2                 = igemm_next_pow2( self.lds_a) if self.lds_a != 0 else 0
        self.lds_b_np2                 = igemm_next_pow2( self.lds_b) if self.lds_b != 0 else 0
        lds_a_pad                      = self.lds_a_np2 // 32 * (32 + self.lds_pad_m)
        lds_b_pad                      = self.lds_b_np2 // 32 * (32 + self.lds_pad_n)
        self.lds_single                = igemm_next_pow2( self.lds_a_np2 + self.lds_b_np2) if (self.lds_a_np2 + self.lds_b_np2 != 0) else 0
        # wmma_main_loop.py only implements a single-buffered LDS schedule for this milestone
        self.lds_buffer_num            = 1 if self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA else 2
        self.lds_total                 = self.lds_buffer_num * self.lds_single

        # for case whose tile size is like 128x128x32, the top priority is to keep the occupancy bigger than 2
        # TODO: need to make some compromise in occupancy and lds double buffer
        if self.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA and self.is_occupancy_decreased():
            self.lds_buffer_num                 = 1 if self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS else 2
            self.lds_total                      = self.lds_buffer_num * self.lds_single
        if self.lds_total > 32 * 1024:
            self.lds_buffer_num                 = 1
            self.lds_total                      = self.lds_buffer_num * self.lds_single
        # print(f"lds_a:{self.lds_a}, lds_b:{self.lds_b}, lds_a_np2:{self.lds_a_np2}, lds_b_np2:{self.lds_b_np2}, lds_single:{self.lds_single}, lds_total:{self.lds_total}")
        # TODO: LDS size check

        # some parameter not in modular_conv
        if self.fma_type in (IGEMM_GTC_TUNABLE_FMA_TYPE_MAC, IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS):
            if igemm_use_lanegroup_thread_mapping(self):
                pass
            else:
                self.thread_tile_m                  = self.gemm_m_repeat * self.gemm_m_per_thread
                self.thread_tile_n                  = self.gemm_n_repeat * self.gemm_n_per_thread
                self.thread_sub_tile_m              = self.gemm_m_per_thread
                self.thread_sub_tile_n              = self.gemm_n_per_thread

        # number of loops at least needed for final coalescing store, dicided by LDS size
        # self.coalescing_store_groups            = (self.gemm_m_per_block * self.gemm_n_per_block) // \
        #         (self.lds_buffer_num * igemm_next_pow2(igemm_next_pow2(self.gemm_k_per_block * self.gemm_m_per_block) + igemm_next_pow2(self.gemm_k_per_block * self.gemm_n_per_block) ))
        if self.direction == "wrw" and (self.tensor_b_thread_lengths[3] == 1 or self.vector_store == 1) and self.gemm_k_global_split == 1 and self.precision == 'fp16':
            self.use_fp32_atomic_add_for_fp16_data = 1
        else:
            self.use_fp32_atomic_add_for_fp16_data = 0

        if (self.direction == "fwd" or self.direction == "bwd") and self.vector_store == 1 and self.gemm_k_global_split == 1 and self.precision == 'fp16':
            self.use_fp32_atomic_add_for_fp16_data = 1

        if self.gemm_k_global_split == 1 and self.precision == 'bf16':
            self.use_fp32_atomic_add_for_fp16_data = 1

        self.coalescing_store_groups = math.ceil((self.gemm_m_per_block * self.gemm_n_per_block) / (self.lds_total // (amdgpu_precision_data_byte(self.precision) if self.use_fp32_atomic_add_for_fp16_data == 0 else 4)))

        if self.coalescing_store_groups == 0:
            self.coalescing_store_groups = 1        # this means LDS size is already bigger than c matrix all pixel. just use one group is ok
        #if self.coalescing_store_groups < 2:
        #    self.coalescing_store_groups = 2
        if self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA:
            # coalescing_store_wmma.py is a direct-store epilogue with no LDS-reshuffle
            # grouping at all for this milestone -- always exactly one group.
            self.coalescing_store_groups = 1
        shrinked_lds_buffer_num = self.lds_buffer_num
        if self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            # check on grouping
            xdlops_mapping = get_ctrl_xdlops_mapping_from_wave_tile(self.gemm_m_per_block, self.gemm_n_per_block, self.wave_tile_m, self.wave_tile_n, self.wave_tile_k, 
                    self.wave_repeat_m, self.wave_repeat_n, self.wave_step_m, self.wave_step_n, self.block_size // self.wave_size, self.precision)
            length_in_m =  xdlops_mapping.wave_repeat_m * xdlops_mapping.wave_step_m * xdlops_mapping.lanegroup_m_per_wave() * xdlops_mapping.lanegroup_m_per_block() # no need xdlops_mapping.lanegroup_m_per_thread()
            if length_in_m % self.coalescing_store_groups != 0:
                # we still asume both value are power of 2
                assert self.coalescing_store_groups % length_in_m == 0
                shrink_in_co_group = self.coalescing_store_groups // length_in_m

                # TODO: this may affect occupancy!
                shrinked_lds_buffer_num = shrinked_lds_buffer_num * shrink_in_co_group
                self.lds_total = shrinked_lds_buffer_num * self.lds_single
                self.coalescing_store_groups = self.coalescing_store_groups // shrink_in_co_group
                assert length_in_m % self.coalescing_store_groups == 0
        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS:
            dotx_mapping = get_ctrl_dotx_mapping_from_lanegroup_tile(self.gemm_m_per_block, self.gemm_n_per_block,
                                        self.lanegroup_tile_m, self.lanegroup_tile_n, self.lanegroup_wave_m, self.lanegroup_wave_n, self.block_size // self.wave_size,
                                        self.lanegroup_repeat_m, self.lanegroup_repeat_n, self.precision, get_dotx_fma_instruction(mc_get_current().arch_config.arch, self.precision))
            c_vgpr_sst = dotx_mapping.lanegroup_repeat_m * dotx_mapping.lanegroup_m_per_thread() // self.coalescing_store_groups
            dotx_length_in_m = utility_gcd(self.vector_c, dotx_mapping.lanegroup_m_per_thread()) # TODO: lanegroup size may differ
            assert c_vgpr_sst >= dotx_length_in_m, f"v sst is smaller than length in m"
            if c_vgpr_sst % dotx_length_in_m != 0:
                self.coalescing_store_groups = dotx_mapping.lanegroup_repeat_m * dotx_mapping.lanegroup_m_per_thread() // dotx_length_in_m
        if self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            if self.lds_total >= lds_a_pad + lds_b_pad:
                pass
            else:
                self.lds_total += (lds_a_pad - self.lds_a_np2 + lds_b_pad - self.lds_b_np2)

    def get_lds_pad(self):
        if self.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            return 0, 0
        if self.direction == 'wrw' and self.precision in ('fp16', 'bf16'):
            if self.gemm_k_per_block == 32 and self.gemm_m_per_block >= 128 and self.gemm_n_per_block >= 128 and self.tensor_b_thread_lengths[1] >= 4:
                return 4, 4
            else:
                return 0, 0
        if self.direction == 'bwd' and self.precision in ('fp16', 'bf16'):
            if self.gemm_k_per_block == 32 and self.gemm_m_per_block >= 128 and self.gemm_n_per_block >= 128 and self.tensor_b_thread_lengths[1] >= 8:
                if self.gemm_m_per_block == 128 and self.gemm_n_per_block == 128:
                    return 0, 0
                else:
                    return 0, 4
            else:
                return 0, 0
        else:
            return 0, 0

    def is_occupancy_decreased(self):
        is_decreased = False
        is_lds_decreased = False
        is_agpr_decreased = False
        is_vgpr_decreased = False
        if self.lds_single <= 16 * 1024 and self.lds_single > 8 * 1024:
            is_lds_decreased = True

        if self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            if self.num_agpr_accumulate_c < 128:
                is_agpr_decreased = True

        a_data_per_vgpr = 1 

        # for fwd and bwd pass, return true directly, because they do not use lds double buffer
        if self.direction == 'fwd':
            return True

        elif self.direction == "wrw":
            if self.precision == "fp32":
                return True
            elif self.tensor_a_thread_lengths[3] > 1:
                a_data_per_vgpr = 2
            else:
                a_data_per_vgpr = 1
        else:
            return True

        if self.num_global_load_a // a_data_per_vgpr <= 8:
            is_vgpr_decreased = True

        is_decreased = is_lds_decreased and is_agpr_decreased and is_vgpr_decreased
        return is_decreased

    def output(self):
        def to_miopen_prec(precision):
            if precision == 'fp32':
                return 'miopenFloat'
            if precision == 'fp16':
                return 'miopenHalf'
            if precision == 'bf16':
                return 'miopenBFloat16'
            else:
                assert False, "unkown data type"

        if False:
            brace_left='   {'
            brace_right='}'
            direction = "\"" + self.direction + "\""
            precision = "\"" + self.precision + "\""
            out_str = (f"\t\t{'{':2}{direction}{',':2}{precision},{self.nxb:4},{self.nxe:4},{self.gemm_m_per_block:4},{self.gemm_n_per_block:4},{self.gemm_k_per_block:4},")
            out_str += (f"{self.wave_tile_m:4},{self.wave_tile_n:4},{self.wave_tile_k:4},{self.wave_step_m:4},{self.wave_step_n:4},{self.wave_repeat_m:4},{self.wave_repeat_n:4},")
            out_str += (f"{brace_left}{self.tensor_a_thread_lengths[0]},{self.tensor_a_thread_lengths[1]:4},{self.tensor_a_thread_lengths[2]:4},{self.tensor_a_thread_lengths[3]:4}{brace_right},")
            out_str += (f"{brace_left}{self.tensor_a_cluster_lengths[0]},{self.tensor_a_cluster_lengths[1]:4},{self.tensor_a_cluster_lengths[2]:4},{self.tensor_a_cluster_lengths[3]:4}{brace_right},")
            out_str += (f"{brace_left}{self.tensor_b_thread_lengths[0]},{self.tensor_b_thread_lengths[1]:4},{self.tensor_b_thread_lengths[2]:4},{self.tensor_b_thread_lengths[3]:4}{brace_right},")
            out_str += (f"{brace_left}{self.tensor_b_cluster_lengths[0]},{self.tensor_b_cluster_lengths[1]:4},{self.tensor_b_cluster_lengths[2]:4},{self.tensor_b_cluster_lengths[3]:4}{brace_right},")
            out_str += (f"{self.gemm_k_global_split:4}{brace_right},")
        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS:
            brace_left='{'
            brace_right='}'
            direction = "\"" + self.direction + "\""
            tensor_layout = "\"" + self.tensor_layout + "\""
            precision = to_miopen_prec(self.precision) + 'x' + str(self.vector_c)
            out_str = (f"        {'{'}{direction}, {tensor_layout}, {precision}, {self.nxb:2},{self.nxe:2},{self.gemm_m_per_block:4},{self.gemm_n_per_block:4},{self.gemm_k_per_block:4},")
            out_str += (f"{self.lanegroup_tile_m:3},{self.lanegroup_tile_n:3},{self.lanegroup_wave_m:2},{self.lanegroup_wave_n:2},{self.lanegroup_repeat_m:2},{self.lanegroup_repeat_n:2},")
            out_str += (f"{self.vector_c:2},")
            out_str += (f" {brace_left}{self.tensor_a_thread_lengths[0]:2},{self.tensor_a_thread_lengths[1]:2},{self.tensor_a_thread_lengths[2]:2},{self.tensor_a_thread_lengths[3]:2}{brace_right},")
            out_str += (f" {brace_left}{self.tensor_a_cluster_lengths[0]:3},{self.tensor_a_cluster_lengths[1]:3},{self.tensor_a_cluster_lengths[2]:3},{self.tensor_a_cluster_lengths[3]:3}{brace_right},")
            out_str += (f" {brace_left}{self.tensor_b_thread_lengths[0]:2},{self.tensor_b_thread_lengths[1]:2},{self.tensor_b_thread_lengths[2]:2},{self.tensor_b_thread_lengths[3]:2}{brace_right},")
            out_str += (f" {brace_left}{self.tensor_b_cluster_lengths[0]:3},{self.tensor_b_cluster_lengths[1]:3},{self.tensor_b_cluster_lengths[2]:3},{self.tensor_b_cluster_lengths[3]:3}{brace_right}")
            out_str += f"{brace_right},"
        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA:
            brace_left='{'
            brace_right='}'
            direction = "\"" + self.direction + "\""
            tensor_layout = "\"" + self.tensor_layout + "\""
            precision = to_miopen_prec(self.precision)
            out_str = (f"        {'{'}{direction}, {tensor_layout}, {precision}, {self.nxb:2},{self.nxe:2},{self.gemm_m_per_block:4},{self.gemm_n_per_block:4},{self.gemm_k_per_block:4},")
            out_str += (f"{self.wmma_tile_m:3},{self.wmma_tile_n:3},{self.wmma_repeat_m:2},{self.wmma_repeat_n:2},")
            out_str += (f"{self.multihead:2},{self.vector_store:2},{self.gemm_k_global_split:2},{self.merge_e:2},{self.vector_c:2},{self.tensor_a_pass_through:2},")
            out_str += (f" {brace_left}{self.tensor_a_thread_lengths[0]:2},{self.tensor_a_thread_lengths[1]:2},{self.tensor_a_thread_lengths[2]:2},{self.tensor_a_thread_lengths[3]:2}{brace_right},")
            out_str += (f" {brace_left}{self.tensor_a_cluster_lengths[0]:3},{self.tensor_a_cluster_lengths[1]:3},{self.tensor_a_cluster_lengths[2]:3},{self.tensor_a_cluster_lengths[3]:3}{brace_right},")
            out_str += (f" {brace_left}{self.tensor_b_thread_lengths[0]:2},{self.tensor_b_thread_lengths[1]:2},{self.tensor_b_thread_lengths[2]:2},{self.tensor_b_thread_lengths[3]:2}{brace_right},")
            out_str += (f" {brace_left}{self.tensor_b_cluster_lengths[0]:3},{self.tensor_b_cluster_lengths[1]:3},{self.tensor_b_cluster_lengths[2]:3},{self.tensor_b_cluster_lengths[3]:3}{brace_right}")
            out_str += f"{brace_right},"
        else:
            brace_left='{'
            brace_right='}'
            direction = "\"" + self.direction + "\""
            tensor_layout = "\"" + self.tensor_layout + "\""
            precision = to_miopen_prec(self.precision)
            out_str = (f"        {'{'}{direction}, {tensor_layout}, {precision}, {self.nxb:2},{self.nxe:2},{self.gemm_m_per_block:4},{self.gemm_n_per_block:4},{self.gemm_k_per_block:4},")
            out_str += (f"{self.wave_tile_m:3},{self.wave_tile_n:3},{self.wave_tile_k:3},{self.wave_step_m:2},{self.wave_step_n:2},{self.wave_repeat_m:2},{self.wave_repeat_n:2},")
            out_str += (f"{self.multihead:2},{self.vector_store:2},{self.gemm_k_global_split:2},{self.merge_e:2},{self.vector_c:2},{self.tensor_a_pass_through:2},")
            out_str += (f" {brace_left}{self.tensor_a_thread_lengths[0]:2},{self.tensor_a_thread_lengths[1]:2},{self.tensor_a_thread_lengths[2]:2},{self.tensor_a_thread_lengths[3]:2}{brace_right},")
            out_str += (f" {brace_left}{self.tensor_a_cluster_lengths[0]:3},{self.tensor_a_cluster_lengths[1]:3},{self.tensor_a_cluster_lengths[2]:3},{self.tensor_a_cluster_lengths[3]:3}{brace_right},")
            out_str += (f" {brace_left}{self.tensor_b_thread_lengths[0]:2},{self.tensor_b_thread_lengths[1]:2},{self.tensor_b_thread_lengths[2]:2},{self.tensor_b_thread_lengths[3]:2}{brace_right},")
            out_str += (f" {brace_left}{self.tensor_b_cluster_lengths[0]:3},{self.tensor_b_cluster_lengths[1]:3},{self.tensor_b_cluster_lengths[2]:3},{self.tensor_b_cluster_lengths[3]:3}{brace_right}")
            out_str += f"{brace_right},"
            
        return out_str

    def to_dict(self):
        tunable_dict = {}
        tunable_dict['tensor_layout']                   = self.tensor_layout
        tunable_dict['fma_type']                        = self.fma_type
        tunable_dict['gemm_m_per_block']                = self.gemm_m_per_block
        tunable_dict['gemm_n_per_block']                = self.gemm_n_per_block
        tunable_dict['gemm_k_per_block']                = self.gemm_k_per_block
        if self.fma_type in (IGEMM_GTC_TUNABLE_FMA_TYPE_MAC, IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS):
            if igemm_use_lanegroup_thread_mapping(self):
                 tunable_dict['lanegroup_tile_m']       = self.lanegroup_tile_m
                 tunable_dict['lanegroup_wave_m']       = self.lanegroup_wave_m
                 tunable_dict['lanegroup_repeat_n']     = self.lanegroup_repeat_n
                 tunable_dict['lanegroup_tile_n']       = self.lanegroup_tile_n
                 tunable_dict['lanegroup_wave_n']       = self.lanegroup_wave_n
                 tunable_dict['lanegroup_repeat_n']     = self.lanegroup_repeat_n
            else:
                tunable_dict['gemm_m_per_thread']       = self.gemm_m_per_thread
                tunable_dict['gemm_m_level0_cluster']   = self.gemm_m_level0_cluster
                tunable_dict['gemm_m_level1_cluster']   = self.gemm_m_level1_cluster
                tunable_dict['gemm_n_per_thread']       = self.gemm_n_per_thread
                tunable_dict['gemm_n_level0_cluster']   = self.gemm_n_level0_cluster
                tunable_dict['gemm_n_level1_cluster']   = self.gemm_n_level1_cluster
        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            tunable_dict['wave_tile_m']                 = self.wave_tile_m
            tunable_dict['wave_step_m']                 = self.wave_step_m
            tunable_dict['wave_repeat_m']               = self.wave_repeat_m
            tunable_dict['wave_tile_n']                 = self.wave_tile_n
            tunable_dict['wave_step_n']                 = self.wave_step_n
            tunable_dict['wave_repeat_n']               = self.wave_repeat_n
            tunable_dict['wave_tile_k']                 = self.wave_tile_k
        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA:
            tunable_dict['wmma_tile_m']                 = self.wmma_tile_m
            tunable_dict['wmma_repeat_m']               = self.wmma_repeat_m
            tunable_dict['wmma_tile_n']                 = self.wmma_tile_n
            tunable_dict['wmma_repeat_n']               = self.wmma_repeat_n
        else:
            assert False
        tunable_dict['tensor_a_pass_through']           = self.tensor_a_pass_through
        tunable_dict['tensor_b_pass_through']           = self.tensor_b_pass_through
        tunable_dict['tensor_a_thread_lengths']         = self.tensor_a_thread_lengths
        tunable_dict['tensor_a_cluster_lengths']        = self.tensor_a_cluster_lengths
        tunable_dict['tensor_b_thread_lengths']         = self.tensor_b_thread_lengths
        tunable_dict['tensor_b_cluster_lengths']        = self.tensor_b_cluster_lengths
        tunable_dict['direction']                       = self.direction
        tunable_dict['precision']                       = self.precision
        tunable_dict['nxb']                             = self.nxb
        tunable_dict['nxe']                             = self.nxe
        tunable_dict['source_access_order']             = self.source_access_order
        tunable_dict['gemm_k_global_split']             = self.gemm_k_global_split
        tunable_dict['merge_e']                         = self.merge_e
        tunable_dict['vector_c']                        = self.vector_c
        tunable_dict['multihead']                       = self.multihead
        tunable_dict['allow_lds_reorder']               = self.allow_lds_reorder
        tunable_dict['precache_soffset']                = self.precache_soffset

        tunable_dict['local_prefetch_num']              = self.local_prefetch_num
        tunable_dict['global_prefetch_a_num']           = self.global_prefetch_a_num
        tunable_dict['global_prefetch_b_num']           = self.global_prefetch_b_num
        tunable_dict['fma_interleave']                  = self.fma_interleave

        tunable_dict['gemm_m_unmerge_cluster']          = self.gemm_m_unmerge_cluster
        tunable_dict['gemm_n_unmerge_cluster']          = self.gemm_n_unmerge_cluster
        tunable_dict['gemm_k_unmerge_cluster']          = self.gemm_k_unmerge_cluster
        tunable_dict['vector_store']                    = self.vector_store
        tunable_dict['wavefront_size']                  = self.wavefront_size
        tunable_dict['cumode']                          = self.cumode

        return tunable_dict

    def serialize(self, **options):
        def get_dict_with_default(some_dict, key, default_value):
            if key in some_dict:
                return some_dict[key]
            return default_value

        section_name = get_dict_with_default(options, 'section_name', False)
        line_start = get_dict_with_default(options, 'line_start', '; ')
        new_line = get_dict_with_default(options, 'new_line', '\n')
        equal = get_dict_with_default(options, 'equal', ':')
        extra_info = get_dict_with_default(options, 'extra_info', True)
        sstr = ''

        if section_name:
            sstr += \
                line_start + f'[igemm_{self.direction}_gtc]' + new_line

        sstr += line_start + 'tensor_layout              {} {}'.format(equal, '\'' + self.tensor_layout + '\'') + new_line + \
                line_start + 'gemm_m_per_block           {} {}'.format(equal, self.gemm_m_per_block) + new_line + \
                line_start + 'gemm_n_per_block           {} {}'.format(equal, self.gemm_n_per_block) + new_line + \
                line_start + 'gemm_k_per_block           {} {}'.format(equal, self.gemm_k_per_block) + new_line
        if self.fma_type in (IGEMM_GTC_TUNABLE_FMA_TYPE_MAC, IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS):
            if igemm_use_lanegroup_thread_mapping(self):
                sstr += \
                line_start + 'lanegroup_tile_m           {} {}'.format(equal, self.lanegroup_tile_m) + new_line + \
                line_start + 'lanegroup_wave_m           {} {}'.format(equal, self.lanegroup_wave_m) + new_line + \
                line_start + 'lanegroup_repeat_m         {} {}'.format(equal, self.lanegroup_repeat_m) + new_line + \
                line_start + 'lanegroup_tile_n           {} {}'.format(equal, self.lanegroup_tile_n) + new_line + \
                line_start + 'lanegroup_wave_n           {} {}'.format(equal, self.lanegroup_wave_n) + new_line + \
                line_start + 'lanegroup_repeat_n         {} {}'.format(equal, self.lanegroup_repeat_n) + new_line
            else:
                sstr += \
                line_start + 'gemm_m_per_thread          {} {}'.format(equal, self.gemm_m_per_thread) + new_line + \
                line_start + 'gemm_m_level0_cluster      {} {}'.format(equal, self.gemm_m_level0_cluster) + new_line + \
                line_start + 'gemm_m_level1_cluster      {} {}'.format(equal, self.gemm_m_level1_cluster) + new_line + \
                line_start + 'gemm_n_per_thread          {} {}'.format(equal, self.gemm_n_per_thread) + new_line + \
                line_start + 'gemm_n_level0_cluster      {} {}'.format(equal, self.gemm_n_level0_cluster) + new_line + \
                line_start + 'gemm_n_level1_cluster      {} {}'.format(equal, self.gemm_n_level1_cluster) + new_line
        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
            sstr += \
                line_start + 'wave_tile_m                {} {}'.format(equal, self.wave_tile_m) + new_line + \
                line_start + 'wave_step_m                {} {}'.format(equal, self.wave_step_m) + new_line + \
                line_start + 'wave_repeat_m              {} {}'.format(equal, self.wave_repeat_m) + new_line + \
                line_start + 'wave_tile_n                {} {}'.format(equal, self.wave_tile_n) + new_line + \
                line_start + 'wave_step_n                {} {}'.format(equal, self.wave_step_n) + new_line + \
                line_start + 'wave_repeat_n              {} {}'.format(equal, self.wave_repeat_n) + new_line + \
                line_start + 'wave_tile_k                {} {}'.format(equal, self.wave_tile_k) + new_line
        elif self.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA:
            sstr += \
                line_start + 'wmma_tile_m                {} {}'.format(equal, self.wmma_tile_m) + new_line + \
                line_start + 'wmma_repeat_m              {} {}'.format(equal, self.wmma_repeat_m) + new_line + \
                line_start + 'wmma_tile_n                {} {}'.format(equal, self.wmma_tile_n) + new_line + \
                line_start + 'wmma_repeat_n              {} {}'.format(equal, self.wmma_repeat_n) + new_line
        if self.tensor_a_pass_through:
            sstr += \
                line_start + 'tensor_a_pass_through      {} {}'.format(equal, self.tensor_a_pass_through) + new_line
        if self.tensor_b_pass_through:
            sstr += \
                line_start + 'tensor_b_pass_through      {} {}'.format(equal, self.tensor_b_pass_through) + new_line
        sstr += \
                line_start + 'tensor_a_thread_lengths    {} {}'.format(equal, self.tensor_a_thread_lengths) + new_line + \
                line_start + 'tensor_a_cluster_lengths   {} {}'.format(equal, self.tensor_a_cluster_lengths) + new_line + \
                line_start + 'tensor_b_thread_lengths    {} {}'.format(equal, self.tensor_b_thread_lengths) + new_line + \
                line_start + 'tensor_b_cluster_lengths   {} {}'.format(equal, self.tensor_b_cluster_lengths) + new_line + \
                line_start + 'direction                  {} {}'.format(equal, '\'' + self.direction + '\'') + new_line + \
                line_start + 'precision                  {} {}'.format(equal, '\'' + self.precision + '\'') + new_line + \
                line_start + 'nxb                        {} {}'.format(equal, self.nxb) + new_line + \
                line_start + 'nxe                        {} {}'.format(equal, self.nxe) + new_line
        if self.gemm_k_global_split:
            sstr += \
                line_start + 'gemm_k_global_split        {} {}'.format(equal, self.gemm_k_global_split) + new_line
        if self.merge_e:
            sstr += \
                line_start + 'merge_e                    {} {}'.format(equal, self.merge_e) + new_line
        if self.vector_c:
            sstr += \
                line_start + 'vector_c                   {} {}'.format(equal, self.vector_c) + new_line
        if self.vector_store:
            sstr += \
                line_start + 'vector_store               {} {}'.format(equal, self.vector_store) + new_line
        if extra_info:
            sstr += \
                line_start + new_line + \
                line_start + 'block_size                 {} {}'.format(equal, self.block_size) + new_line
            if self.fma_type in (IGEMM_GTC_TUNABLE_FMA_TYPE_MAC, IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS):
                if igemm_use_lanegroup_thread_mapping(self):
                    pass
                else:
                    sstr += \
                    line_start + 'thread_tile                {} {}x{}'.format(equal, self.thread_tile_m, self.thread_tile_n) + new_line
            sstr += \
                line_start + 'lds_total                  {} {}'.format(equal, self.lds_total) + new_line + \
                line_start + 'lds_buffer_num             {} {}'.format(equal, self.lds_buffer_num) + new_line + \
                line_start
        return sstr

    def serialize_as_section(self):
        return self.serialize(section_name=True, line_start='', equal='=', extra_info=False)

def igemm_gtc_encode_kernel_base_name(tunable, arch):
    assert type(tunable) is igemm_gtc_tunable_parameter_t

    kernel_name = f"igemm_{tunable.direction}_"
    if type(arch) is not str:
        arch_str = amdgpu_arch_to_string(arch)
    else:
        arch_str = arch

    if tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC:
        kernel_name += 'gtcm_'                                  # generic tensor contraction with mac
    elif tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS:
        if arch_str == 'gfx1030':
            kernel_name += 'gtcn2_'
        else:
            kernel_name += 'gtc_'                               # generic tensor contraction with dlops
    elif tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
        if arch_str == 'gfx908':
            kernel_name += 'gtcx_'                              # generic tensor contraction with xdlops
        elif arch_str == 'gfx90a':
            kernel_name += 'gtcx2_'
        elif arch_str == 'gfx940' or arch_str == 'gfx942':
            kernel_name += 'gtcx3_'
        elif arch_str == 'gfx950':
            kernel_name += 'gtcx35_'
        else:
            assert False
    elif tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA:
        assert arch_str == 'gfx1250'
        kernel_name += 'gtcw_'                                  # generic tensor contraction with wmma

    vector_c_str = ""
    if tunable.vector_c > 1:
        vector_c_str = f"x{tunable.vector_c}"

    kernel_name += f"{tunable.tensor_layout}_{tunable.precision}{vector_c_str}"

    return kernel_name

def igemm_gtc_encode_kernel_name(tunable, arch):
    def lengths_str(lengths):
        assert type(lengths) is list
        return "x".join( [f"{x}" for x in lengths] )

    assert type(tunable) is igemm_gtc_tunable_parameter_t

    kernel_name = igemm_gtc_encode_kernel_base_name(tunable, arch) + '_'

    kernel_name += f"bx{tunable.nxb}_ex{tunable.nxe}_"
    if IGEMM_GTC_FEAT_SOURCE_ACCESS_ENCODING_KERNEL_NAME:
        kernel_name += f"sa{tunable.source_access_order}_"
    kernel_name += f"bt{tunable.gemm_m_per_block}x{tunable.gemm_n_per_block}x{tunable.gemm_k_per_block}_"
    if tunable.fma_type in (IGEMM_GTC_TUNABLE_FMA_TYPE_MAC, IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS):
        if igemm_use_lanegroup_thread_mapping(tunable):
            kernel_name += f'lt{tunable.lanegroup_tile_m}x{tunable.lanegroup_tile_n}_' +\
                            f'lw{tunable.lanegroup_wave_m}x{tunable.lanegroup_wave_n}_' +\
                            f'lr{tunable.lanegroup_repeat_m}x{tunable.lanegroup_repeat_n}_'
        else:
            kernel_name +=   f"tt{tunable.thread_tile_m}x{tunable.thread_tile_n}_" +\
                         f"gm{tunable.gemm_m_repeat}x{tunable.gemm_m_level0_cluster}x{tunable.gemm_m_level1_cluster}_" +\
                         f"gn{tunable.gemm_n_repeat}x{tunable.gemm_n_level0_cluster}x{tunable.gemm_n_level1_cluster}_"
    elif tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS:
        kernel_name +=   f'wt{tunable.wave_tile_m}x{tunable.wave_tile_n}x{tunable.wave_tile_k}_' +\
                         f'ws{tunable.wave_step_m}x{tunable.wave_step_n}_' +\
                         f'wr{tunable.wave_repeat_m}x{tunable.wave_repeat_n}_'
    elif tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA:
        kernel_name +=   f'wt{tunable.wmma_tile_m}x{tunable.wmma_tile_n}_' +\
                         f'wr{tunable.wmma_repeat_m}x{tunable.wmma_repeat_n}_'

    kernel_name +=       "ta" + lengths_str(tunable.tensor_a_thread_lengths) + "_" + lengths_str(tunable.tensor_a_cluster_lengths) + "_" +\
                         "tb" + lengths_str(tunable.tensor_b_thread_lengths) + "_" + lengths_str(tunable.tensor_b_cluster_lengths)

    # Phase 16: gfx1250 WMMA optional-mechanism tunables were previously NOT folded into the
    # mangled kernel name (Phase 13's own docstring already flagged this for async_global_load)
    # -- meaning e.g. the interleaved and non-interleaved builds of the SAME tile shape produced
    # IDENTICAL kernel names, so they couldn't coexist as distinct, auto-discoverable kernels in
    # one combined kernel object the way gfx950/942's many named tile variants do. Suffixes
    # match the established _dbuf/_async/_interleave config-file-naming convention already in
    # use. Only added when the tunable is actually set (every existing config has all three at
    # their default 0), so every existing kernel name is unaffected -- this only changes the
    # name for configs that opt into one of these mechanisms, which by construction (Phase 13/
    # 15's docstrings) is only the small number of _dbuf/_async/_k2x_dbuf/_interleave configs.
    if tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA:
        if tunable.lds_double_buffer:
            kernel_name += "_dbuf"
        if tunable.async_global_load:
            kernel_name += "_async"
        if tunable.tdm_global_load:
            kernel_name += "_tdm"
        if tunable.saddr_global_load:
            kernel_name += "_saddr"
        if tunable.main_loop_interleave:
            kernel_name += "_interleave"
        if tunable.wmma_acc_f16:
            kernel_name += "_f16acc"
        if tunable.wmma_acc_bf16:
            kernel_name += "_bf16acc"
        if tunable.wmma_setprio:
            kernel_name += "_setprio"
        if tunable.atomic_pack_bf16:
            kernel_name += "_pkatomic"
        if tunable.wrw_reduction_kernel:
            kernel_name += "_wsred"
        if tunable.gsplit_stagger:
            kernel_name += "_stagger"
        if tunable.wrw_streamk:
            kernel_name += "_streamk"
        # Extends Phase 16's fold above to the M/N/K-tail EXEC-mask/fine-grained-mask
        # mechanisms (Phases 25/26b/35/36/38) -- these were pure masking additions with no
        # buffer-layout change, so folding them was skipped originally (every existing
        # config keeps a distinct name per tile shape regardless). That becomes a real
        # problem once multiple tail-flag combinations of the SAME tile shape need to
        # coexist as separate, simultaneously-searchable kernels in one combined "master"
        # config file (mirroring gfx950/942's single comprehensive per-direction/precision
        # file, see docs/gfx1250_wmma_layout.md's master-config phase) -- without this,
        # they'd collide on symbol name and fail to assemble together. Only added when set
        # (every existing config has all three at their default 0), so no existing kernel
        # name changes for non-tail configs; tail-enabled configs' names DO change (gain a
        # suffix) -- a one-time, intentional rename, not a regression (the compiled
        # instruction bodies are unaffected).
        if tunable.wmma_m_tail:
            kernel_name += "_mtail"
        if tunable.wmma_n_tail:
            kernel_name += "_ntail"
        if tunable.wmma_k_tail:
            kernel_name += "_ktail"
        # Found while assembling the master config files (new): three more WMMA-only
        # tunables were unfolded and produced real, previously-undetected kernel-name
        # collisions between separate config files that happened to differ ONLY in one of
        # these (e.g. igemm_wrw_gtc_gfx1250_nhwc_bf16_k2x_bf16acc.config vs its own _lp2
        # sibling; igemm_bwd_gtc_gfx1250_nhwc_bf16.config's 64x64 section vs
        # _ldspad.config's). Gated to WMMA only, matching every other fold in this block --
        # gfx908/90a/942/950's XDLOPS/DLOPS/MAC kernel names are completely unaffected
        # regardless of these tunables' values there.
        if tunable.epilogue_lds_pad:
            kernel_name += "_ldspad"
        if tunable.direct_store:
            kernel_name += "_direct"
        if tunable.ds_load_tr_b:
            kernel_name += "_dstrb"
        if tunable.lds_row_pad:
            kernel_name += "_ldsrp"
        # Phase 68 (2026-09-02): wmma_epilogue_chunked/wmma_acc_high_bank (see this
        # tunable's own definitions above) change the epilogue's generated code (chunked
        # LDS staging, v_c living in a second VGPR bank respectively) just like every
        # other flag folded into this block, so must be folded into the kernel name for
        # the same hipModuleGetFunction-lookup reason -- otherwise a chunked/hibank build
        # and its plain sibling of the same tile shape collide on symbol name.
        if tunable.wmma_epilogue_chunked:
            kernel_name += "_chunked"
        if tunable.wmma_acc_high_bank:
            kernel_name += "_hibank"
        if tunable.local_prefetch_num != 1:
            kernel_name += f"_lp{tunable.local_prefetch_num}"
        if tunable.atomic_scope != 'SCOPE_SYS':
            kernel_name += "_scopedev" if tunable.atomic_scope == 'SCOPE_DEV' else f"_ascope{tunable.atomic_scope}"

    if tunable.tensor_a_pass_through:
        kernel_name += "_pta"

    if tunable.tensor_b_pass_through:
        kernel_name += "_ptb"

    if tunable.gemm_m_unmerge_cluster:
        kernel_name += "_mc"

    if tunable.gemm_n_unmerge_cluster:
        kernel_name += "_nc"

    if tunable.gemm_k_unmerge_cluster:
        kernel_name += "_kc"

    if tunable.multihead:
        kernel_name += "_mh"

    if tunable.merge_e:
        kernel_name += "_me"

    if tunable.vector_store:
        kernel_name += f"_vs{tunable.vector_store}"

    if tunable.gemm_k_global_split:
        kernel_name += "_gkgs"

    return kernel_name


class igemm_kernel_detail_base_t(object):
    # gemm problem details
    def __init__(self):
        self.vgpr_total = 0
        self.sgpr_total = 0

        self.thread_m = 0
        self.thread_n = 0
        self.block_m = 0
        self.block_n = 0
        self.unroll_k = 0
        self.block_size = 0

        self.vgpr_c_accumulate = 0
        self.vgpr_a_accumulate = 0
        self.vgpr_b_accumulate = 0
        self.vgpr_a_global_fetch = 0
        self.vgpr_b_global_fetch = 0
        # if local fetch to accumulate directly, no extra local fetch gpr is needed
        self.vgpr_a_local_fetch = 0
        self.vgpr_b_local_fetch = 0
        self.vgpr_other = 0

        self.lds_total = 0
        self.lds_buffers = 1        # single buffer, double buffer...
        self.occupancy = 1

        
        # now hard code v4r1 tiling stratagy. in the future, this should be more flex
        # wei->tensor_a, input->tensor_b
        # wei: e_k, e is unroll_k

        self.msg = ''

    def getattrs(self):
        attrs = [i for i in dir(self) if not callable(getattr(self,i)) and not i.startswith("__") and not i == 'msg']
        return attrs

    def key(self):
        attrs = self.getattrs()
        return '-'.join( [ str(getattr(self, attr)) for attr in attrs] ) 

    def serialize(self):
        return  'thread_mxn          : {}x{}'.format(self.thread_m, self.thread_n) + '\n' + \
                'block_mxn           : {}x{}'.format(self.block_m, self.block_n) + '\n' + \
                'unroll_k            : {}'.format(self.unroll_k) + '\n' + \
                'block_size          : {}'.format(self.block_size) + '\n' + \
                'vgpr_total          : {}'.format(self.vgpr_total) + '\n' + \
                'sgpr_total          : {}'.format(self.sgpr_total) + '\n' + \
                'lds_total           : {}'.format(self.lds_total) + '\n' + \
                'lds_buffers         : {}'.format(self.lds_buffers) + '\n' + \
                'occupancy           : {}'.format(self.occupancy) + '\n' + \
                'vgpr_c_accumulate   : {}'.format(self.vgpr_c_accumulate) + '\n' + \
                'vgpr_a_accumulate   : {}'.format(self.vgpr_a_accumulate) + '\n' + \
                'vgpr_b_accumulate   : {}'.format(self.vgpr_b_accumulate) + '\n' + \
                'vgpr_a_global_fetch : {}'.format(self.vgpr_a_global_fetch) + '\n' + \
                'vgpr_b_global_fetch : {}'.format(self.vgpr_b_global_fetch) + '\n' + \
                'vgpr_a_local_fetch  : {}'.format(self.vgpr_a_local_fetch) + '\n' + \
                'vgpr_b_local_fetch  : {}'.format(self.vgpr_b_local_fetch) + '\n' + \
                'vgpr_other          : {}'.format(self.vgpr_other) + '\n'

class igemm_thread_cluster_index_dispatcher_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    
    def __call__(self, v_x, v_tid_shifter, c_x, t_x, is_last = False):
        with self._deferred_context():
            if c_x == 1:
                self._emit(f"v_mov_b32 v[{v_x}], 0")
            else:
                self._emit(f"v_and_b32 v[{v_x}], {c_x - 1}, v[{v_tid_shifter}]")
                if t_x != 1:
                    self._emit(f"v_lshlrev_b32 v[{v_x}], {igemm_log2(t_x)}, v[{v_x}]")
                if not is_last:
                    self._emit(f"v_lshrrev_b32 v[{v_tid_shifter}], {igemm_log2(c_x)}, v[{v_tid_shifter}]")
        return self._get_deferred()

class igemm_thread_cluster_index_accumulator_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)

    def __call__(self, v_dst, v_x0, v_x1, c_x0, c_x1, n_x0, n_x1):
        assert not (c_x0 == 1 and c_x1 == 1)
        with self._deferred_context():
            if c_x0 != 1 and c_x1 != 1:
                self._emit(f"v_lshl_or_b32 v[{v_dst}], v[{v_x0}], {igemm_log2(n_x1)}, v[{v_x1}]")
            elif c_x0 == 1 and c_x1 != 1:
                self._emit(f"v_mov_b32 v[{v_dst}], v[{v_x1}]")
            elif c_x0 != 1 and c_x1 == 1:
                if n_x1 == 1:
                    self._emit(f"v_mov_b32 v[{v_dst}], v[{v_x0}]")
                else:
                    self._emit(f"v_lshlrev_b32 v[{v_dst}], {igemm_log2(n_x1)}, v[{v_x0}]")
        return self._get_deferred()
