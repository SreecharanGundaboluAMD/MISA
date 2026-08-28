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

from ..codegen import *
import math

DBG_USE_PACK_F16_FOR_BF16 = 0

class macro_int_div_vv_t(macro_base_t):
    '''
    integer divide to compute `v_q = v_n / v_d`, v_q, v_n, v_d all vgpr
    '''
    def name(self):
        return '.v_u32_div'
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)
    def __call__(self, v_q, v_n, v_d, v_tmp4, s_tmp4):
        return '{} {}, {}, {}, {}, {}'.format(self.name(), v_q, v_n, v_d, v_tmp4, s_tmp4)
    def emit(self):
        with self._emit_macro_indented(".macro {} v_q, v_n, v_d, v_tmp4, s_tmp4".format(self.name())):
            if self.mc.arch_config.arch >= 940 and self.mc.arch_config.arch < 1000:
                self._emit("v_cvt_f32_u32 v[\\v_tmp4+0], v[\\v_d]")
                self._emit("v_sub_i32 v[\\v_tmp4+3], 0, v[\\v_d]")
                self._emit("v_rcp_iflag_f32 v[\\v_tmp4+0], v[\\v_tmp4+0]")
                self._emit("s_nop 0")
                self._emit("v_mul_f32 v[\\v_tmp4+0], 0x4f7ffffe, v[\\v_tmp4+0]")
                self._emit("v_cvt_u32_f32 v[\\v_tmp4+0], v[\\v_tmp4+0]")
                self._emit("v_mul_lo_u32 v[\\v_tmp4+1], v[\\v_tmp4+3], v[\\v_tmp4+0]")
                self._emit("v_mul_hi_u32 v[\\v_tmp4+1], v[\\v_tmp4+0], v[\\v_tmp4+1]")
                self._emit("v_add_u32 v[\\v_tmp4+0], v[\\v_tmp4+0], v[\\v_tmp4+1]")
                self._emit("v_mul_hi_u32 v[\\v_tmp4+0], v[\\v_n], v[\\v_tmp4+0]")
                self._emit("v_mul_lo_u32 v[\\v_tmp4+1], v[\\v_tmp4+0], v[\\v_d]")
                self._emit("v_sub_u32 v[\\v_q], v[\\v_n], v[\\v_tmp4+1]")
                self._emit("v_add_u32 v[\\v_tmp4+2], 1, v[\\v_tmp4+0]")
                self._emit("v_cmp_le_u32 vcc, v[\\v_d], v[\\v_q]")
                self._emit("v_subrev_u32 v[\\v_tmp4+1], v[\\v_d], v[\\v_q]")
                self._emit("s_nop 0")
                self._emit("v_cndmask_b32 v[\\v_tmp4+0], v[\\v_tmp4+0], v[\\v_tmp4+2], vcc")
                self._emit("v_cndmask_b32 v[\\v_q], v[\\v_q], v[\\v_tmp4+1], vcc")
                self._emit("v_add_u32 v[\\v_tmp4+1], 1, v[\\v_tmp4+0]")
                self._emit("v_cmp_le_u32 vcc, v[\\v_d], v[\\v_q]")
                self._emit("s_nop 1")
                self._emit("v_cndmask_b32 v[\\v_q], v[\\v_tmp4+0], v[\\v_tmp4+1], vcc")
            else:
                self._emit("v_cvt_f32_u32     v[\\v_tmp4+0],   v[\\v_d]")
                self._emit("v_rcp_f32         v[\\v_tmp4+0],   v[\\v_tmp4+0]")
                self._emit("v_mul_f32         v[\\v_tmp4+0],   0x4f800000, v[\\v_tmp4+0]")
                self._emit("v_cvt_u32_f32     v[\\v_tmp4+0],   v[\\v_tmp4+0]")
                self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   v[\\v_d],      v[\\v_tmp4+0]")
                self._emit("v_mul_hi_u32      v[\\v_tmp4+2],   v[\\v_d],      v[\\v_tmp4+0]")
                self._emit("v_sub_co_u32      v[\\v_tmp4+3],   vcc, 0,     v[\\v_tmp4+1]")
                self._emit("v_cmp_ne_i32      s[\\s_tmp4:\\s_tmp4+1], 0,          v[\\v_tmp4+2]")
                self._emit("v_cndmask_b32     v[\\v_tmp4+1],   v[\\v_tmp4+3],   v[\\v_tmp4+1],   s[\\s_tmp4:\\s_tmp4+1]")
                self._emit("v_mul_hi_u32      v[\\v_tmp4+1],   v[\\v_tmp4+1],   v[\\v_tmp4+0]")
                self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
                self._emit("v_add_co_u32      v[\\v_tmp4+0],   vcc,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
                self._emit("v_cndmask_b32     v[\\v_tmp4+0],   v[\\v_tmp4+0],   v[\\v_tmp4+2],   s[\\s_tmp4:\\s_tmp4+1]")
                self._emit("v_mul_hi_u32      v[\\v_tmp4+0],   v[\\v_tmp4+0],   v[\\v_n]")
                self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   v[\\v_tmp4+0],   v[\\v_d]")
                self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc,        v[\\v_n],      v[\\v_tmp4+1]")
                self._emit("v_cmp_ge_u32      s[\\s_tmp4:\\s_tmp4+1], v[\\v_n],      v[\\v_tmp4+1]")
                self._emit("v_cmp_ge_u32      s[\\s_tmp4+2:\\s_tmp4+3], v[\\v_tmp4+2],   v[\\v_d]")
                self._emit("v_add_co_u32      v[\\v_tmp4+2],   vcc, 1, v[\\v_tmp4+0]")
                self._emit("s_and_b64         s[\\s_tmp4+2:\\s_tmp4+3], s[\\s_tmp4:\\s_tmp4+1], s[\\s_tmp4+2:\\s_tmp4+3]")
                self._emit("v_add_co_u32      v[\\v_tmp4+1],   vcc, -1,    v[\\v_tmp4+0]")
                self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+0],   v[\\v_tmp4+2],      s[\\s_tmp4+2:\\s_tmp4+3]")
                self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+1],   v[\\v_tmp4+2],      s[\\s_tmp4:\\s_tmp4+1]")
                self._emit("v_cmp_ne_i32      vcc,          0,          v[\\v_d]")
                self._emit("v_cndmask_b32     v[\\v_q],      -1,         v[\\v_tmp4+2],      vcc")

class macro_int_div_rem_vv_t(macro_base_t):
    '''
    integer divide to compute `v_q = v_n / v_d, v_r = v_n % v_d`, v_r, v_q, v_n, v_d all vgpr
    '''
    def name(self):
        return '.v_u32_div_rem'
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)
    def __call__(self, v_r, v_q, v_n, v_d, v_tmp4, s_tmp4):
        return '{} {}, {}, {}, {}, {}, {}'.format(self.name(), v_r, v_q, v_n, v_d, v_tmp4, s_tmp4)
    def emit(self):
        int_div_vv = macro_int_div_vv_t(self.mc)
        with self._emit_macro_indented(".macro {} v_r, v_q, v_n, v_d, v_tmp4, s_tmp4".format(self.name())):
            self._emit(int_div_vv("\\v_q", "\\v_n", "\\v_d", "\\v_tmp4", "\\s_tmp4"))
            self._emit(f"v_mul_lo_u32 v[\\v_tmp4], v[\\v_d], v[\\v_q]")
            self._emit(f"v_sub_u32 v[\\v_r], v[\\v_n], v[\\v_tmp4]")

class macro_int_div_vs_t(macro_base_t):
    '''
    integer divide to compute `v_q = v_n / s_d`, v_q, v_n are vgpr, s_d is sgpr
    '''
    def name(self):
        return '.v_u32_div_vs'
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)
    def __call__(self, v_q, v_n, s_d, v_tmp4, s_tmp4):
        return '{} {}, {}, {}, {}, {}'.format(self.name(), v_q, v_n, s_d, v_tmp4, s_tmp4)
    def emit(self):
        with self._emit_macro_indented(".macro {} v_q, v_n, s_d, v_tmp4, s_tmp4".format(self.name())):
            if self.mc.arch_config.arch >= 940 and self.mc.arch_config.arch < 1000:
                self._emit("v_cvt_f32_u32 v[\\v_tmp4+0], s[\\s_d]")
                self._emit("s_sub_i32 s[\\s_tmp4+0], 0, s[\\s_d]")
                self._emit("v_rcp_iflag_f32 v[\\v_tmp4+0], v[\\v_tmp4+0]")
                self._emit("s_nop 0")
                self._emit("v_mul_f32 v[\\v_tmp4+0], 0x4f7ffffe, v[\\v_tmp4+0]")
                self._emit("v_cvt_u32_f32 v[\\v_tmp4+0], v[\\v_tmp4+0]")
                self._emit("v_mul_lo_u32 v[\\v_tmp4+1], s[\\s_tmp4+0], v[\\v_tmp4+0]")
                self._emit("v_mul_hi_u32 v[\\v_tmp4+1], v[\\v_tmp4+0], v[\\v_tmp4+1]")
                self._emit("v_add_u32 v[\\v_tmp4+0], v[\\v_tmp4+0], v[\\v_tmp4+1]")
                self._emit("v_mul_hi_u32 v[\\v_tmp4+0], v[\\v_n], v[\\v_tmp4+0]")
                self._emit("v_mul_lo_u32 v[\\v_tmp4+1], v[\\v_tmp4+0], s[\\s_d]")
                self._emit("v_sub_u32 v[\\v_q], v[\\v_n], v[\\v_tmp4+1]")
                self._emit("v_add_u32 v[\\v_tmp4+2], 1, v[\\v_tmp4+0]")
                self._emit("v_cmp_le_u32 vcc, s[\\s_d], v[\\v_q]")
                self._emit("v_subrev_u32 v[\\v_tmp4+1], s[\\s_d], v[\\v_q]")
                self._emit("s_nop 0")
                self._emit("v_cndmask_b32 v[\\v_tmp4+0], v[\\v_tmp4+0], v[\\v_tmp4+2], vcc")
                self._emit("v_cndmask_b32 v[\\v_q], v[\\v_q], v[\\v_tmp4+1], vcc")
                self._emit("v_add_u32 v[\\v_tmp4+1], 1, v[\\v_tmp4+0]")
                self._emit("v_cmp_le_u32 vcc, s[\\s_d], v[\\v_q]")
                self._emit("s_nop 1")
                self._emit("v_cndmask_b32 v[\\v_q], v[\\v_tmp4+0], v[\\v_tmp4+1], vcc")
            else:
                self._emit("v_cvt_f32_u32     v[\\v_tmp4+0],   s[\\s_d]")
                self._emit("v_rcp_f32         v[\\v_tmp4+0],   v[\\v_tmp4+0]")
                self._emit("v_mul_f32         v[\\v_tmp4+0],   0x4f800000, v[\\v_tmp4+0]")
                self._emit("v_cvt_u32_f32     v[\\v_tmp4+0],   v[\\v_tmp4+0]")
                self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   s[\\s_d],      v[\\v_tmp4+0]")
                self._emit("v_mul_hi_u32      v[\\v_tmp4+2],   s[\\s_d],      v[\\v_tmp4+0]")
                self._emit("v_sub_co_u32      v[\\v_tmp4+3],   vcc, 0,     v[\\v_tmp4+1]")
                self._emit("v_cmp_ne_i32      s[\\s_tmp4:\\s_tmp4+1], 0,          v[\\v_tmp4+2]")
                self._emit("v_cndmask_b32     v[\\v_tmp4+1],   v[\\v_tmp4+3],   v[\\v_tmp4+1],   s[\\s_tmp4:\\s_tmp4+1]")
                self._emit("v_mul_hi_u32      v[\\v_tmp4+1],   v[\\v_tmp4+1],   v[\\v_tmp4+0]")
                self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
                self._emit("v_add_co_u32      v[\\v_tmp4+0],   vcc,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
                self._emit("v_cndmask_b32     v[\\v_tmp4+0],   v[\\v_tmp4+0],   v[\\v_tmp4+2],   s[\\s_tmp4:\\s_tmp4+1]")
                self._emit("v_mul_hi_u32      v[\\v_tmp4+0],   v[\\v_tmp4+0],   v[\\v_n]")
                self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   s[\\s_d],     v[\\v_tmp4+0]")
                self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc,        v[\\v_n],      v[\\v_tmp4+1]")
                self._emit("v_cmp_ge_u32      s[\\s_tmp4:\\s_tmp4+1], v[\\v_n],      v[\\v_tmp4+1]")
                self._emit("v_cmp_le_u32      s[\\s_tmp4+2:\\s_tmp4+3],  s[\\s_d],    v[\\v_tmp4+2]")
                self._emit("v_add_co_u32      v[\\v_tmp4+2],   vcc, 1, v[\\v_tmp4+0]")
                self._emit("s_and_b64         s[\\s_tmp4+2:\\s_tmp4+3], s[\\s_tmp4:\\s_tmp4+1], s[\\s_tmp4+2:\\s_tmp4+3]")
                self._emit("v_add_co_u32      v[\\v_tmp4+1],   vcc, -1,    v[\\v_tmp4+0]")
                self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+0],   v[\\v_tmp4+2],      s[\\s_tmp4+2:\\s_tmp4+3]")
                self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+1],   v[\\v_tmp4+2],      s[\\s_tmp4:\\s_tmp4+1]")
                self._emit("v_cmp_ne_i32      vcc,          s[\\s_d],   0")
                self._emit("v_cndmask_b32     v[\\v_q],      -1,         v[\\v_tmp4+2],      vcc")

class macro_int_div_vs_gfx1250_t(macro_base_t):
    '''
    gfx1250 (wave32)-specific plain u32 division: v_q = v_n / s_d, v_q/v_n vgpr, s_d sgpr.
    Algorithm identical to macro_int_div_vs_t's else-branch (works for any u32 divisor >= 1,
    including the non-power-of-2 conv dimensions this is used for), but using vcc_lo (not bare
    vcc) and single-SGPR (not 64-bit-pair) condition masks, since gfx1250 is wave32 and rejects
    both wave64-only forms outright (confirmed via llvm-mc). Verified independently on real
    gfx1250 hardware across many (numerator, divisor) pairs before being trusted in any kernel
    -- see /tmp/wmma_probe/{probe_div.s,host_div.cpp}.
    '''
    def name(self):
        return '.v_u32_div_vs_gfx1250'
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)
    def __call__(self, v_q, v_n, s_d, v_tmp4, s_tmp4):
        return '{} {}, {}, {}, {}, {}'.format(self.name(), v_q, v_n, s_d, v_tmp4, s_tmp4)
    def emit(self):
        with self._emit_macro_indented(".macro {} v_q, v_n, s_d, v_tmp4, s_tmp4".format(self.name())):
            self._emit("v_cvt_f32_u32     v[\\v_tmp4+0],   s[\\s_d]")
            self._emit("v_rcp_f32         v[\\v_tmp4+0],   v[\\v_tmp4+0]")
            self._emit("v_mul_f32         v[\\v_tmp4+0],   0x4f800000, v[\\v_tmp4+0]")
            self._emit("v_cvt_u32_f32     v[\\v_tmp4+0],   v[\\v_tmp4+0]")
            self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   s[\\s_d],      v[\\v_tmp4+0]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+2],   s[\\s_d],      v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+3],   vcc_lo, 0,     v[\\v_tmp4+1]")
            self._emit("v_cmp_ne_i32      s[\\s_tmp4+0], 0,          v[\\v_tmp4+2]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+1],   v[\\v_tmp4+3],   v[\\v_tmp4+1],   s[\\s_tmp4+0]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+1],   v[\\v_tmp4+1],   v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc_lo,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
            self._emit("v_add_co_u32      v[\\v_tmp4+0],   vcc_lo,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+0],   v[\\v_tmp4+0],   v[\\v_tmp4+2],   s[\\s_tmp4+0]")
            self._emit("v_mul_hi_u32      v[\\v_tmp4+0],   v[\\v_tmp4+0],   v[\\v_n]")
            self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   s[\\s_d],     v[\\v_tmp4+0]")
            self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc_lo,        v[\\v_n],      v[\\v_tmp4+1]")
            self._emit("v_cmp_ge_u32      s[\\s_tmp4+0], v[\\v_n],      v[\\v_tmp4+1]")
            self._emit("v_cmp_le_u32      s[\\s_tmp4+1],  s[\\s_d],    v[\\v_tmp4+2]")
            self._emit("v_add_co_u32      v[\\v_tmp4+2],   vcc_lo, 1, v[\\v_tmp4+0]")
            self._emit("s_and_b32         s[\\s_tmp4+1], s[\\s_tmp4+0], s[\\s_tmp4+1]")
            self._emit("v_add_co_u32      v[\\v_tmp4+1],   vcc_lo, -1,    v[\\v_tmp4+0]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+0],   v[\\v_tmp4+2],      s[\\s_tmp4+1]")
            self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+1],   v[\\v_tmp4+2],      s[\\s_tmp4+0]")
            self._emit("v_cmp_ne_i32      vcc_lo,          s[\\s_d],   0")
            self._emit("v_cndmask_b32     v[\\v_q],      -1,         v[\\v_tmp4+2],      vcc_lo")

class macro_int_div_rem_vs_gfx1250_t(macro_base_t):
    '''
    gfx1250 (wave32) counterpart to macro_int_div_rem_vs_t: v_q = v_n / s_d, v_r = v_n % s_d.
    '''
    def name(self):
        return '.v_u32_div_rem_vs_gfx1250'
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)
    def __call__(self, v_r, v_q, v_n, s_d, v_tmp4, s_tmp4):
        return '{} {}, {}, {}, {}, {}, {}'.format(self.name(), v_r, v_q, v_n, s_d, v_tmp4, s_tmp4)
    def emit(self):
        int_div_vs = macro_int_div_vs_gfx1250_t(self.mc)
        with self._emit_macro_indented(".macro {} v_r, v_q, v_n, s_d, v_tmp4, s_tmp4".format(self.name())):
            self._emit(int_div_vs("\\v_q", "\\v_n", "\\s_d", "\\v_tmp4", "\\s_tmp4"))
            self._emit(f"v_mul_lo_u32 v[\\v_tmp4], s[\\s_d], v[\\v_q]")
            self._emit(f"v_sub_u32 v[\\v_r], v[\\v_n], v[\\v_tmp4]")
            self._emit("s_nop 0")

class macro_int_div_rem_vs_t(macro_base_t):
    '''
    integer divide to compute `v_q = v_n / s_d, v_r = v_n % s_d`, v_r, v_q, v_n are vgpr, s_d is sgpr
    '''
    def name(self):
        return '.v_u32_div_rem_vs'
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)
    def __call__(self, v_r, v_q, v_n, s_d, v_tmp4, s_tmp4):
        return '{} {}, {}, {}, {}, {}, {}'.format(self.name(), v_r, v_q, v_n, s_d, v_tmp4, s_tmp4)
    def emit(self):
        int_div_vs = macro_int_div_vs_t(self.mc)
        with self._emit_macro_indented(".macro {} v_r, v_q, v_n, s_d, v_tmp4, s_tmp4".format(self.name())):
            self._emit(int_div_vs("\\v_q", "\\v_n", "\\s_d", "\\v_tmp4", "\\s_tmp4"))
            self._emit(f"v_mul_lo_u32 v[\\v_tmp4], s[\\s_d], v[\\v_q]")
            self._emit(f"v_sub_u32 v[\\v_r], v[\\v_n], v[\\v_tmp4]")
            self._emit("s_nop 0")

class macro_int_div_ss_t(macro_base_t):
    '''
    integer divide to compute `s_q = s_n / s_d`, s_q, s_n, s_d all sgpr
    '''
    def name(self):
        return '.v_u32_div_ss'
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)
    def __call__(self, v_q, s_n, s_d, v_tmp4, s_tmp4):
        return '{} {}, {}, {}, {}, {}'.format(self.name(), v_q, s_n, s_d, v_tmp4, s_tmp4)
    def emit(self):
        with self._emit_macro_indented(".macro .v_u32_div_ss v_q, s_n, s_d, v_tmp4, s_tmp4"):
            if self.mc.arch_config.arch >= 940 and self.mc.arch_config.arch < 1000:
                self._emit("v_cvt_f32_u32 v[\\v_tmp4+0], s[\\s_d]")
                self._emit("s_sub_i32 s[\\s_tmp4+0], 0, s[\\s_d]")
                self._emit("v_rcp_iflag_f32 v[\\v_tmp4+0], v[\\v_tmp4+0]")
                self._emit("s_nop 0")
                self._emit("v_mul_f32 v[\\v_tmp4+0], 0x4f7ffffe, v[\\v_tmp4+0]")
                self._emit("v_cvt_u32_f32 v[\\v_tmp4+0], v[\\v_tmp4+0]")
                self._emit("v_mul_lo_u32 v[\\v_tmp4+1], s[\\s_tmp4+0], v[\\v_tmp4+0]")
                self._emit("v_mul_hi_u32 v[\\v_tmp4+1], v[\\v_tmp4+0], v[\\v_tmp4+1]")
                self._emit("v_add_u32 v[\\v_tmp4+0], v[\\v_tmp4+0], v[\\v_tmp4+1]")
                self._emit("v_mul_hi_u32 v[\\v_tmp4+0], s[\\s_n], v[\\v_tmp4+0]")
                self._emit("v_mul_lo_u32 v[\\v_tmp4+1], v[\\v_tmp4+0], s[\\s_d]")
                self._emit("v_sub_u32 v[\\v_q], s[\\s_n], v[\\v_tmp4+1]")
                self._emit("v_add_u32 v[\\v_tmp4+2], 1, v[\\v_tmp4+0]")
                self._emit("v_cmp_le_u32 vcc, s[\\s_d], v[\\v_q]")
                self._emit("v_subrev_u32 v[\\v_tmp4+1], s[\\s_d], v[\\v_q]")
                self._emit("s_nop 0")
                self._emit("v_cndmask_b32 v[\\v_tmp4+0], v[\\v_tmp4+0], v[\\v_tmp4+2], vcc")
                self._emit("v_cndmask_b32 v[\\v_q], v[\\v_q], v[\\v_tmp4+1], vcc")
                self._emit("v_add_u32 v[\\v_tmp4+1], 1, v[\\v_tmp4+0]")
                self._emit("v_cmp_le_u32 vcc, s[\\s_d], v[\\v_q]")
                self._emit("s_nop 1")
                self._emit("v_cndmask_b32 v[\\v_q], v[\\v_tmp4+0], v[\\v_tmp4+1], vcc")
            else:
                self._emit("v_cvt_f32_u32     v[\\v_tmp4+0],   s[\\s_d]")
                self._emit("v_rcp_f32         v[\\v_tmp4+0],   v[\\v_tmp4+0]")
                self._emit("v_mul_f32         v[\\v_tmp4+0],   0x4f800000, v[\\v_tmp4+0]")
                self._emit("v_cvt_u32_f32     v[\\v_tmp4+0],   v[\\v_tmp4+0]")
                self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   s[\\s_d],      v[\\v_tmp4+0]")
                self._emit("v_mul_hi_u32      v[\\v_tmp4+2],   s[\\s_d],      v[\\v_tmp4+0]")
                self._emit("v_sub_co_u32      v[\\v_tmp4+3],   vcc, 0,     v[\\v_tmp4+1]")
                self._emit("v_cmp_ne_i32      s[\\s_tmp4:\\s_tmp4+1], 0,          v[\\v_tmp4+2]")
                self._emit("v_cndmask_b32     v[\\v_tmp4+1],   v[\\v_tmp4+3],   v[\\v_tmp4+1],   s[\\s_tmp4:\\s_tmp4+1]")
                self._emit("v_mul_hi_u32      v[\\v_tmp4+1],   v[\\v_tmp4+1],   v[\\v_tmp4+0]")
                self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
                self._emit("v_add_co_u32      v[\\v_tmp4+0],   vcc,        v[\\v_tmp4+0],   v[\\v_tmp4+1]")
                self._emit("v_cndmask_b32     v[\\v_tmp4+0],   v[\\v_tmp4+0],   v[\\v_tmp4+2],   s[\\s_tmp4:\\s_tmp4+1]")
                self._emit("v_mul_hi_u32      v[\\v_tmp4+0],   s[\\s_n],   v[\\v_tmp4+0]")
                self._emit("v_mul_lo_u32      v[\\v_tmp4+1],   s[\\s_d],     v[\\v_tmp4+0]")
                self._emit("v_sub_co_u32      v[\\v_tmp4+2],   vcc,        s[\\s_n],      v[\\v_tmp4+1]")
                self._emit("v_cmp_ge_u32      s[\\s_tmp4:\\s_tmp4+1], s[\\s_n],      v[\\v_tmp4+1]")
                self._emit("v_cmp_le_u32      s[\\s_tmp4+2:\\s_tmp4+3],  s[\\s_d],    v[\\v_tmp4+2]")
                self._emit("v_add_co_u32      v[\\v_tmp4+2],   vcc, 1, v[\\v_tmp4+0]")
                self._emit("s_and_b64         s[\\s_tmp4+2:\\s_tmp4+3], s[\\s_tmp4:\\s_tmp4+1], s[\\s_tmp4+2:\\s_tmp4+3]")
                self._emit("v_add_co_u32      v[\\v_tmp4+1],   vcc, -1,    v[\\v_tmp4+0]")
                self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+0],   v[\\v_tmp4+2],      s[\\s_tmp4+2:\\s_tmp4+3]")
                self._emit("v_cndmask_b32     v[\\v_tmp4+2],   v[\\v_tmp4+1],   v[\\v_tmp4+2],      s[\\s_tmp4:\\s_tmp4+1]")
                self._emit("v_cmp_ne_i32      vcc,          s[\\s_d],   0")
                self._emit("v_cndmask_b32     v[\\v_q],      -1,         v[\\v_tmp4+2],      vcc")

class macro_int_div_rem_ss_t(macro_base_t):
    '''
    integer divide to compute `s_q = s_n / s_d, s_r = s_n % s_d`, s_r, s_q, s_n, s_d all sgpr
    '''
    def name(self):
        return '.v_u32_div_rem_ss'
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)

    def __call__(self, s_r, s_q, s_n, s_d, v_q, v_tmp4, s_tmp4):
        return '{} {}, {}, {}, {}, {}, {}, {}'.format(self.name(), s_r, s_q, s_n, s_d, v_q, v_tmp4, s_tmp4) 

    def emit(self):
        int_div_ss = macro_int_div_ss_t(self.mc)
        with self._emit_macro_indented(".macro {} s_r, s_q, s_n, s_d, v_q, v_tmp4, s_tmp4".format(self.name())):
            self._emit(int_div_ss("\\v_q", "\\s_n", "\\s_d", "\\v_tmp4", "\\s_tmp4"))
            if self.mc.arch_config.arch >= 940 and self.mc.arch_config.arch < 1000:
                self._emit(f"s_nop 0")  # must manually insert nop when valu write vgprn, then readlane that vgpr
            self._emit(f"v_readfirstlane_b32 s[\\s_q], v[\\v_q]")
            self._emit(f"s_mul_i32 s[\\s_tmp4], s[\\s_d], s[\\s_q]")
            self._emit(f"s_sub_i32 s[\\s_r], s[\\s_n], s[\\s_tmp4]")


class macro_mdiv_u32_si_t(macro_base_t):
    def name(self):
        return '.mdiv_u32_si'
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("s_quot")
        self.declare_arg("s_numer")
        self.declare_arg("magic")
        self.declare_arg("shift")
        self.declare_arg("s_tmp")   # can be the same as s_quot
    def expr(self):
        self._emit(f"s_mul_hi_u32 s[{self.s_tmp()}], {self.magic()}, s[{self.s_numer()}]")
        self._emit(f"s_add_u32 s[{self.s_tmp()}], s[{self.s_tmp()}], s[{self.s_numer()}]")
        self._emit(f"s_lshr_b32 s[{self.s_quot()}], s[{self.s_tmp()}], {self.shift()}")
        

class macro_mdiv_u32_vi_t(macro_base_t):
    def name(self):
        return '.mdiv_u32_vi'
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_quot")
        self.declare_arg("v_numer")
        self.declare_arg("magic")
        self.declare_arg("shift")
        self.declare_arg("v_tmp")
        self.mc = mc
    def expr(self):
        self._emit(f"v_mul_hi_u32 v[{self.v_tmp()}], {self.magic()}, v[{self.v_numer()}]")
        self._emit(v_add_nc_u32(f"{self.v_tmp()}", f"{self.v_tmp()}", f"{self.v_numer()}"))
        self._emit(f"v_lshrrev_b32 v[{self.v_quot()}], {self.shift()}, v[{self.v_tmp()}]")
        
class div_u32_vi_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    
    def __call__(self, v_quot, v_numer, denom, v_tmp):
        assert isinstance(denom, int)
        with self._deferred_context():
            if utility_is_pow2(denom):
                self._emit(f"v_lshrrev_b32 v[{v_quot}], {utility_log2(denom)}, v[{v_numer}]")
            else:
                d_magic, d_shift = utility_division_magic(denom)
                mdiv_u32_vi = macro_mdiv_u32_vi_t(self.mc)
                self._emit(mdiv_u32_vi(v_quot, v_numer, str(d_magic), str(d_shift), v_tmp))
        return self._get_deferred()

class div_u32_si_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, s_quot, s_numer, denom, s_tmp):
        assert isinstance(denom, int)
        with self._deferred_context():
            if utility_is_pow2(denom):
                self._emit(f"s_lshr_b32 s[{s_quot}], s[{s_numer}], {utility_log2(denom)}")
            else:
                d_magic, d_shift = utility_division_magic(denom)
                mdiv_u32_si = macro_mdiv_u32_si_t(self.mc)
                self._emit(mdiv_u32_si(s_quot, s_numer, str(d_magic), str(d_shift), s_tmp))
        return self._get_deferred()

class mul_u32_si_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, s_o, s_i, multiplier):
        assert isinstance(multiplier, int) or math.log2(multiplier).is_integer()
        with self._deferred_context():
            if math.log2(multiplier).is_integer():
                if math.log2(multiplier) > 0:
                    self._emit(f"s_lshl_b32 s[{s_o}], s[{s_i}], {utility_log2(multiplier)}")
                else:
                    self._emit(f"s_lshr_b32 s[{s_o}], s[{s_i}], {-utility_log2(multiplier)}")
            else:
                self._emit(f"s_mul_i32 s[{s_o}], s[{s_i}], {multiplier}")
        return self._get_deferred()
    
class mul_u32_vi_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, v_o, v_i, multiplier):
        assert isinstance(multiplier, int) or math.log2(multiplier).is_integer()
        with self._deferred_context():
            if math.log2(multiplier).is_integer():
                if math.log2(multiplier) > 0:
                    self._emit(f"v_lshlrev_b32 v[{v_o}], {utility_log2(multiplier)}, v[{v_i}]")
                else:
                    self._emit(f"v_lshrrev_b32 v[{v_o}], {-utility_log2(multiplier)}, v[{v_i}]")
            else:
                self._emit(f"v_mul_lo_u32 v[{v_o}], v[{v_i}], {multiplier}")
        return self._get_deferred()
    
class add_lshl_u32_vi_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def __call__(self, v_o, v_i_0, v_i_1, shift):
        assert isinstance(shift, int)
        with self._deferred_context():
            if shift > 0:
                self._emit(f"v_add_lshl_u32 v[{v_o}], v[{v_i_0}], v[{v_i_1}], {shift}")
            else:
                self._emit(f"v_add_nc_u32 v[{v_o}], v[{v_i_0}], v[{v_i_1}]")
                self._emit(f"v_ashrrev_i32 v[{v_o}], {-shift}, v[{v_o}]")
        return self._get_deferred()


class macro_mdiv_u32_rem_vi_t(macro_base_t):
    def name(self):
        return '.mdiv_u32_rem_vi'
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_rem")
        self.declare_arg("v_quot")
        self.declare_arg("v_numer")
        self.declare_arg("magic")
        self.declare_arg("shift")
        self.declare_arg("denom")
        self.declare_arg("v_tmp")
    def expr(self):
        mdiv_u32_vi = macro_mdiv_u32_vi_t(self.mc, self.inline)
        self._emit(mdiv_u32_vi( self.v_quot(), self.v_numer(), self.magic(), self.shift(), self.v_tmp()  ))
        self._emit(f"v_mul_lo_u32 v[{self.v_tmp()}], {self.denom()}, v[{self.v_quot()}]")
        self._emit(v_sub_nc_u32(f"{self.v_rem()}", f"{self.v_numer()}", f"{self.v_tmp()}"))

class div_rem_u32_vi_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    
    def __call__(self, v_rem, v_quot, v_numer, denom, v_tmp):
        assert isinstance(denom, int)
        with self._deferred_context():
            if utility_is_pow2(denom):
                if v_quot != None:
                    self._emit(f"v_lshrrev_b32 v[{v_quot}], {utility_log2(denom)}, v[{v_numer}]")
                self._emit(f"v_and_b32 v[{v_rem}], {denom - 1}, v[{v_numer}]")
            else:
                if v_quot == None:
                    v_quot = v_tmp + "+1"
                d_magic, d_shift = utility_division_magic(denom)
                mdiv_rem_u32_vi = macro_mdiv_u32_rem_vi_t(self.mc)
                self._emit(mdiv_rem_u32_vi(v_rem, v_quot, v_numer, str(d_magic), str(d_shift), str(denom), v_tmp))
        return self._get_deferred()

class macro_mdiv_u32_ss_t(macro_base_t):
    def name(self):
        return '.mdiv_u32_ss'
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("s_quot")
        self.declare_arg("s_numer")
        self.declare_arg("s_magic")
        self.declare_arg("s_shift")
        self.declare_arg("s_tmp")
    def expr(self):
        self._emit(f"s_mul_hi_u32 s[{self.s_tmp()}], s[{self.s_magic()}], s[{self.s_numer()}]")
        self._emit(f"s_add_u32 s[{self.s_tmp()}], s[{self.s_tmp()}], s[{self.s_numer()}]")
        self._emit(f"s_lshr_b32 s[{self.s_quot()}], s[{self.s_tmp()}], s[{self.s_shift()}]")


class macro_mdiv_u32_rem_ss_t(macro_base_t):
    def name(self):
        return '.mdiv_u32_rem_ss'
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("s_rem")
        self.declare_arg("s_quot")
        self.declare_arg("s_numer")
        self.declare_arg("s_magic")
        self.declare_arg("s_shift")
        self.declare_arg("s_denom")
        self.declare_arg("s_tmp")
    def expr(self):
        mdiv_u32_ss = macro_mdiv_u32_ss_t(self.mc, self.inline)
        self._emit(mdiv_u32_ss(self.s_quot(), self.s_numer(), self.s_magic(), self.s_shift(), self.s_tmp()))
        self._emit(f"s_mul_i32 s[{self.s_tmp()}], s[{self.s_denom()}], s[{self.s_quot()}]")
        self._emit(f"s_sub_u32 s[{self.s_rem()}], s[{self.s_numer()}], s[{self.s_tmp()}]")


class macro_mdiv_u32_vs_t(macro_base_t):
    def name(self):
        return '.mdiv_u32_vs'
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_quot")
        self.declare_arg("v_numer")
        self.declare_arg("s_magic")
        self.declare_arg("s_shift")
        self.declare_arg("v_tmp")
        self.mc = mc
    def expr(self):
        self._emit(f"v_mul_hi_u32 v[{self.v_tmp()}], s[{self.s_magic()}], v[{self.v_numer()}]")
        if self.mc.arch_config.arch == AMDGPU_ARCH_GFX1030:
            self._emit(f"v_add_nc_u32 v[{self.v_tmp()}], v[{self.v_tmp()}], v[{self.v_numer()}]")
        else:
            self._emit(f"v_add_u32 v[{self.v_tmp()}], v[{self.v_tmp()}], v[{self.v_numer()}]")
        self._emit(f"v_lshrrev_b32 v[{self.v_quot()}], s[{self.s_shift()}], v[{self.v_tmp()}]")

class macro_mdiv_u32_rem_vs_t(macro_base_t):
    def name(self):
        return '.mdiv_u32_rem_vs'
    def __init__(self, mc, inline = False):
        macro_base_t.__init__(self, mc, inline)
        self.declare_arg("v_rem")
        self.declare_arg("v_quot")
        self.declare_arg("v_numer")
        self.declare_arg("s_magic")
        self.declare_arg("s_shift")
        self.declare_arg("s_denom")
        self.declare_arg("v_tmp")
    def expr(self):
        mdiv_u32_vs = macro_mdiv_u32_vs_t(self.mc, self.inline)
        self._emit(mdiv_u32_vs( self.v_quot(), self.v_numer(), self.s_magic(), self.s_shift(), self.v_tmp()  ))
        self._emit(f"v_mul_lo_u32 v[{self.v_tmp()}], s[{self.s_denom()}], v[{self.v_quot()}]")
        if self.mc.arch_config.arch == AMDGPU_ARCH_GFX1030:
            self._emit(f"v_sub_nc_u32 v[{self.v_rem()}], v[{self.v_numer()}], v[{self.v_tmp()}]")
        else:
            self._emit(f"v_sub_u32 v[{self.v_rem()}], v[{self.v_numer()}], v[{self.v_tmp()}]")


class macro_c_clear_t(macro_base_t):
    def name(self):
        return '.v_clear_nc'
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)
    def __call__(self, vid, num):
        return '{} {}, {}'.format(self.name(), vid, num)
    def emit(self):
        with self._emit_macro_indented(".macro {} vid, num".format(self.name())):
            self._emit("_v = \\vid")
            self._emit(".rept \\num")
            with self._indent_context():
                self._emit("v_mov_b32 v[_v], 0")
                self._emit("_v = _v + 1")
            self._emit(".endr")

class macro_acc_c_clear_t(macro_base_t):
    '''
    gfx908 RAW harzard attention!
    '''
    def name(self):
        return '.v_clear_acc_c'
    def __init__(self, mc):
        macro_base_t.__init__(self, mc)
    def __call__(self, a, num):
        return '{} {}, {}'.format(self.name(), a, num)
    def emit(self):
        with self._emit_macro_indented(".macro {} a, num".format(self.name())):
            self._emit("_a = \\a")
            self._emit(".rept \\num")
            with self._indent_context():
                self._emit("v_accvgpr_write_b32 a[_a], 0")
                self._emit("_a = _a + 1")
            self._emit(".endr")

class gpr_sequencer_t(object):
    def __init__(self, cnt = 0):
        self.cnt = cnt
    def __call__(self, step = 0, alignment = 0):
        previous_cnt = self.cnt
        if alignment:
            aligned_cnt = ((previous_cnt + alignment - 1) // alignment) * alignment
            self.cnt = aligned_cnt
            previous_cnt = aligned_cnt
        self.cnt += step
        return previous_cnt
    def get(self):
        return self.cnt

class vgpr_msb_tracker_t(object):
    '''
    Phase 54 (gfx1250 VGPR-MSB, doc Sec 3.3.2.3): tracks the wave's currently-active
    S_SET_VGPR_MSB state -- one 2-bit bank field per DST/SRC0/SRC1/SRC2 operand slot --
    and returns a new `s_set_vgpr_msb` line only when the combination required by the
    next instruction differs from what's already active. Mirrors ds_waitcnt_t's
    (main_loop_graph.py) redundancy-suppression pattern: S_SET_VGPR_MSB sets all four
    fields at once and the hardware MODE state persists until the next
    S_SET_VGPR_MSB, so a slot not touched by the current instruction keeps whatever
    bank was last programmed for it -- callers only pass the slots their instruction
    actually uses; the tracker fills in the rest from its retained state.

    Immediate bit layout confirmed via a direct llvm-mc assemble + llvm-objdump
    disassemble round-trip against this project's actual toolchain (see
    docs/gfx1250_wmma_layout.md's Phase 53/54): immediate[7:0] =
    {dst[7:6], src2[5:4], src1[3:2], src0[1:0]}.

    The very first `ensure()` call always emits, regardless of whether the requested
    banks are all 0 -- the ISA doc doesn't state MODE's VGPR-MSB reset value at wave
    launch, so this deliberately never assumes it.
    '''
    def __init__(self):
        self.dst, self.src0, self.src1, self.src2 = 0, 0, 0, 0
        self.initialized = False

    def ensure(self, dst=None, src0=None, src1=None, src2=None):
        new_dst  = self.dst  if dst  is None else dst
        new_src0 = self.src0 if src0 is None else src0
        new_src1 = self.src1 if src1 is None else src1
        new_src2 = self.src2 if src2 is None else src2
        if self.initialized and (new_dst, new_src0, new_src1, new_src2) == (self.dst, self.src0, self.src1, self.src2):
            return None
        self.dst, self.src0, self.src1, self.src2 = new_dst, new_src0, new_src1, new_src2
        self.initialized = True
        imm = (new_dst << 6) | (new_src2 << 4) | (new_src1 << 2) | new_src0
        return f"s_set_vgpr_msb {imm}"

    def force(self, dst=None, src0=None, src1=None, src2=None):
        '''
        Like ensure(), but ALWAYS emits, never skips based on remembered state, and
        never assumes the caller's textual/program order matches runtime execution
        order. Required at any call site that can be reached via more than one
        control-flow path (a real runtime branch or loop, not a compile-time-unrolled
        Python for-loop) -- e.g. wmma_main_loop.py's emit_wmma_tile(), called from
        several different places (early issue, loop body, drain/tail) stitched
        together by actual branches. ensure()'s cross-call memoization silently
        assumes straight-line code: it can wrongly believe a bank is already set
        because some OTHER textual call site set it earlier in the Python script,
        when the runtime path that actually reached this point never executed that
        prior call at all. Found via hardware validation (a real, reproducible
        HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION) after every straight-line-only
        micro-test passed -- see docs/gfx1250_wmma_layout.md's Phase 54.
        '''
        new_dst  = self.dst  if dst  is None else dst
        new_src0 = self.src0 if src0 is None else src0
        new_src1 = self.src1 if src1 is None else src1
        new_src2 = self.src2 if src2 is None else src2
        self.dst, self.src0, self.src1, self.src2 = new_dst, new_src0, new_src1, new_src2
        self.initialized = True
        imm = (new_dst << 6) | (new_src2 << 4) | (new_src1 << 2) | new_src0
        return f"s_set_vgpr_msb {imm}"

class macro_packlo_b32_t(macro_base_t):
    def __init__(self, mc):
        macro_base_t.__init__(self, mc, True)
        self.declare_arg("v_dst")
        self.declare_arg("v_a")
        self.declare_arg("v_b")

    def name(self):
        return '.v_packlo_b32'

    def expr(self):
        if DBG_USE_PACK_F16_FOR_BF16:
            self._emit(f"v_pack_b32_f16 v[{self.v_dst()}], v[{self.v_a()}], v[{self.v_b()}]")
        else:
            self._emit(f"v_lshlrev_b32  v[{self.v_dst()}], 16, v[{self.v_a()}]")
            self._emit(f"v_alignbit_b32 v[{self.v_dst()}], v[{self.v_b()}], v[{self.v_dst()}], 16")

class macro_packhi_b32_t(macro_base_t):
    def __init__(self, mc):
        macro_base_t.__init__(self, mc, True)
        self.declare_arg("v_dst")
        self.declare_arg("v_a")
        self.declare_arg("v_b")

    def name(self):
        return '.v_packhi_b32'

    def expr(self):
        if DBG_USE_PACK_F16_FOR_BF16:
            self._emit(f"v_pack_b32_f16 v[{self.v_dst()}], v[{self.v_a()}], v[{self.v_b()}] op_sel:[1, 1]")
        else:
            self._emit(f"v_lshrrev_b32  v[{self.v_dst()}], 16, v[{self.v_b()}]")
            self._emit(f"v_alignbit_b32 v[{self.v_dst()}], v[{self.v_dst()}], v[{self.v_a()}], 16")


class macro_packed_fp16_to_bf16_t(macro_base_t):
    def __init__(self, mc, **options):
        macro_base_t.__init__(self, mc, True)
        self.options = options
        self.declare_arg("v_packed_f16")
        self.declare_arg("v_tmp")
        assert 'num_vgpr' in options

    def name(self):
        return '.v_packed_fp16_to_bf16'

    def expr(self):
        num_vgpr = self.options["num_vgpr"]
        for i in range(num_vgpr):
            self._emit(f"v_cvt_f32_f16 v[{self.v_tmp()}], v[{self.v_packed_f16(i)}]")
            self._emit(f"v_cvt_f32_f16 v[{self.v_packed_f16(i)}], v[{self.v_packed_f16(i)}] src0_sel:WORD_1")
            self._emit(macro_packhi_b32_t(self.v_packed_f16(i), self.v_tmp(), self.v_packed_f16(i)))

def utility_list_to_string(arr):
    assert type(arr) is list
    return 'x'.join(f'{itm}' for itm in arr)

class utility_dict_with_default_t(object):
    def __init__(self, d):
        self.d = d
    def __call__(self, key, default_value):
        if self.d is None:
            return default_value
        if key in self.d:
            return self.d[key]
        return default_value

# compute next power of 2
def utility_next_pow2(n):
    if n == 0:
        return 1
    if n & (n - 1) == 0:
        return n
    while n & (n - 1) > 0:
        n &= (n - 1)
    return n << 1

def utility_next_mul(n, mul):
    d = n // mul
    d = d + (1 if (n % mul != 0) else 0)
    return d * mul

def utility_is_pow2(v):
    return v and (not(v & (v - 1)))

def utility_log2(v):
    assert math.log2(v).is_integer(), f'v:{v} must be power of 2'
    return int(math.log2(v))

def utility_get_epack_length(precision):
        # GetEPackLength
        epack = 1
        if precision == AMDGPU_PRECISION_FP16:
            # todo: xdlops check
            epack = 2
        elif precision == AMDGPU_PRECISION_BF16:
            epack = 2
        return epack

def utility_gcd(a, b):
    # math.gcd new in python 3.5
    return math.gcd(a, b)

def utility_lcm(a, b):
    return abs(a * b) // math.gcd(a, b)

def utility_flatten_list_product(x):
    assert type(x) is list
    from functools import reduce
    return reduce(lambda a, b: a*b, x, 1)

def utility_flatten_list_accumulate(x):
    assert type(x) is list
    from functools import reduce
    return reduce(lambda a, b: a+b, x, 0)

def utility_division_magic(divisor):
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