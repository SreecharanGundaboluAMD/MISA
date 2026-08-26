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

def inst_wmma_data_type_to_string(data_type):
    if data_type == AMDGPU_PRECISION_FP32:
        return 'fp32'
    if data_type == AMDGPU_PRECISION_FP16:
        return 'fp16'
    if data_type == AMDGPU_PRECISION_BF16:
        return 'bf16'
    if data_type == AMDGPU_PRECISION_INT8:
        return 'int8'
    assert False

class inst_wmma_t(object):
    '''
    gfx1250 WMMA. Unlike MFMA/XDLOPS, there is no AGPR register file on this
    hardware (verified: v_accvgpr_write_b32 is rejected by the assembler on
    gfx1250) -- all four operands (D, A, B, C) are plain VGPRs, always. Do not
    add an "accvgpr_unified"-style branch here; there is nothing to unify.

    num_v_c counts VGPRs (not AGPRs, unlike inst_mfma_t.num_a_c).
    '''
    def __init__(self, m, n, k, data_type, cycle, num_v_a, num_v_b, num_v_c, **options):
        self.m = m
        self.n = n
        self.k = k
        self.data_type = data_type
        self.cycle = cycle
        self.num_v_a = num_v_a
        self.num_v_b = num_v_b
        self.num_v_c = num_v_c
        self.options = options

    def name(self):
        if 'name' in self.options and self.options['name'] != None:
            return self.options['name']
        assert False, "unverified WMMA mnemonics must be given an explicit 'name=' " \
                       "option once confirmed with llvm-mc on gfx1250 -- do not guess"

    def __call__(self, reg_d, reg_a, reg_b, reg_c):
        wmma_inst = self.name()
        modifier = self.options.get('modifier', '')
        return f"{wmma_inst} v[{reg_d}], v[{reg_a}], v[{reg_b}], v[{reg_c}]{modifier}"


#                                        m,  n,  k,   precision,              cycle, v_a, v_b, v_c
# verified with llvm-mc -mcpu=gfx1250 (register footprints are the ground truth; see
# docs/gfx1250_wmma_layout.md for the lane/vgpr -> (row,col,k) formula, verified end-to-end
# on real hardware for the fp16 entry below).
v_wmma_f32_16x16x32_f16   = inst_wmma_t(16, 16, 32,  AMDGPU_PRECISION_FP16,  None,  8,   8,   8,  name='v_wmma_f32_16x16x32_f16')
# Phase 24: F16-accumulate variant -- num_v_c=4 (half of the f32-accumulate entry above),
# each VGPR packing 2 output rows (bits [15:0]/[31:16] of the same lane/column, per the
# CDNA5 ISA doc's 16-bit C/D-matrix table). Verified via llvm-mc -show-encoding: assembles
# with a 4-VGPR (v[d:d+3]) accumulator, and REJECTS an 8-VGPR accumulator ("invalid operand
# for instruction") -- num_v_c=4 is assembler-enforced, not assumed. bf16/int8 have no
# equivalent halved-accumulate variant on this ISA (bf16's only narrower option,
# v_wmma_bf16_16x16x32_bf16, accumulates natively in bf16 -- a real precision tradeoff, not
# implemented here; int8 has no narrower-than-i32 accumulate at all) -- see
# docs/gfx1250_wmma_layout.md's Phase 24.
v_wmma_f16_16x16x32_f16   = inst_wmma_t(16, 16, 32,  AMDGPU_PRECISION_FP16,  None,  8,   8,   4,  name='v_wmma_f16_16x16x32_f16')
v_wmma_f32_16x16x32_bf16  = inst_wmma_t(16, 16, 32,  AMDGPU_PRECISION_BF16,  None,  8,   8,   8,  name='v_wmma_f32_16x16x32_bf16')
# neg_lo:[1,1,0] selects SIGNED interpretation of A and B (the base encoding with no modifier
# defaults to unsigned, confirmed via llvm-mc -- see docs/gfx1250_wmma_layout.md). The original
# fwd-only round-trip probe used small UNSIGNED (0..8) test values, so this never mattered
# until bwd/wrw's driver validation exercised genuinely signed (-5..5) random data and exposed
# it: without this modifier, a negative int8 input is reinterpreted as a large unsigned byte,
# producing silently wrong results whenever real data (not just small positive probe values)
# is used. Conv's int8 tensors are signed (int8_t) throughout the driver, so this must always
# be signed*signed, never the unsigned default.
v_wmma_i32_16x16x64_iu8   = inst_wmma_t(16, 16, 64,  AMDGPU_PRECISION_INT8,  None,  8,   8,   8,  name='v_wmma_i32_16x16x64_iu8', modifier=' neg_lo:[1,1,0]')
# NOTE: no AMDGPU_PRECISION_FP8 constant exists yet in codegen/amdgpu.py; this stretch entry
# is not wired into any mapping table yet, so the placeholder FP32 data_type is unused for now.
v_wmma_f32_16x16x128_fp8_fp8 = inst_wmma_t(16, 16, 128, AMDGPU_PRECISION_FP32, None, 16, 16,   8,  name='v_wmma_f32_16x16x128_fp8_fp8')
v_wmma_f32_16x16x4_f32    = inst_wmma_t(16, 16, 4,   AMDGPU_PRECISION_FP32,  None,  2,   2,   8,  name='v_wmma_f32_16x16x4_f32')
