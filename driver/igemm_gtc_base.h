/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef __IGEMM_GTC_BASE_H
#define __IGEMM_GTC_BASE_H

#ifdef USE_HALF
#include "half.hpp"
using float16 = half_float::half;
#else
using float16 = int16_t;
#endif
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include "config_parser.h"
#include "utility.h"
#include <string>
#include <unistd.h>
#include <vector>
#include <assert.h>
#include <math.h>
#include <functional>
#include <stdint.h>
#include <numeric>
#include <algorithm>
#include "magic_div.h"
#include "shisa_dumps.h"

#define IGEMM_GTC_TUNABLE_FMA_TYPE_MAC              "mac"
#define IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS            "dlops"
#define IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS           "xdlops"
#define IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA              "wmma"
#define IGEMM_GTC_TUNABLE_FMA_TYPE_NA               "fma_na"
#define AMDGPU_WAVE_SIZE        64

typedef enum {
    driverHalf  = 0, /*!< 16-bit floating point (Fully supported) */
    driverFloat = 1, /*!< 32-bit floating point (Fully supported) */
    driverInt8  = 3,
    driverBFloat16 = 5, /*!< 16-bit binary floating point (8-bit exponent, 7-bit fraction)
                           (Partially supported) */
    driverInt4  = 7,
} driverDataType_t;

typedef struct {
    void* output;
    void* input;
    int total_length;
} __attribute__((packed)) tensor_cast_karg_t;

// Phase 35: karg for wrw_reduce_partials_f32 (driver/gpu_tensor_cast/gpu_tensor_cast.cpp) --
// sums num_partitions disjoint output_size-element fp32 slices of workspace into output.
typedef struct {
    void* output;
    void* workspace;
    int num_partitions;
    int output_size;
} __attribute__((packed)) wrw_reduce_karg_t;

static inline size_t get_data_byte(driverDataType_t dtype)
{
    if(dtype == driverHalf)
        return 2;
    if(dtype == driverFloat)
        return 4;
    if(dtype == driverInt8)
        return 1;
    if(dtype == driverBFloat16)
        return 2;
    if(dtype == driverInt4)
        return 1;
    assert(0);
    return 0;
}

typedef enum {
    driver_mode_normal      = 0,    // bench all solutions
    driver_mode_heuristic   = 1,    // find suitable heuristic
} driver_mode_t;

typedef struct {
    std::string tensor_layout;
    int gemm_m_per_block;
    int gemm_n_per_block;
    int gemm_k_per_block;
    std::string fma_type;
    union{
        struct{
            int lanegroup_tile_m;
            int lanegroup_wave_m;
            int lanegroup_repeat_m;
            int lanegroup_tile_n;
            int lanegroup_wave_n;
            int lanegroup_repeat_n;
            int dummy_0;
        };
        struct{
            int gemm_m_per_thread;
            int gemm_m_level0_cluster;
            int gemm_m_level1_cluster;
            int gemm_n_per_thread;
            int gemm_n_level0_cluster;
            int gemm_n_level1_cluster;
            int dummy;
        };
        struct{
            // also reused for WMMA (fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA): wave_tile_m/n
            // hold wmma_tile_m/n, wave_repeat_m/n hold wmma_repeat_m/n; wave_step_m/n and
            // wave_tile_k are unused/left zero for WMMA (no step/k-tile concept there).
            int wave_tile_m;
            int wave_step_m;
            int wave_repeat_m;
            int wave_tile_n;
            int wave_step_n;
            int wave_repeat_n;
            int wave_tile_k;
        };
    };
    int tensor_a_pass_through;
    int tensor_b_pass_through;
    std::vector<int> tensor_a_thread_lengths;
    std::vector<int> tensor_a_cluster_lengths;
    std::vector<int> tensor_b_thread_lengths;
    std::vector<int> tensor_b_cluster_lengths;
    std::string direction;
    std::string precision;
    int nxb;
    int nxe;
    int gemm_m_unmerge_cluster;
    int gemm_n_unmerge_cluster;
    int gemm_k_unmerge_cluster;
    int multihead;
    int source_access_order;
    int vector_store;
    int gemm_k_global_split;
    int merge_e;
    int vector_c;
    // gfx1250 WMMA-only optional mechanisms (Phase 13/15) -- default 0, unused/ignored for
    // every other fma_type. Must be folded into igemm_gtc_encode_kernel_name below (mirroring
    // igemm_base.py's igemm_gtc_encode_kernel_name) so the C++ driver's name computation
    // agrees with what the Python codegen actually named the kernel symbol -- otherwise
    // hipModuleGetFunction looks up the wrong (un-suffixed) name and fails to find it.
    int lds_double_buffer;
    int async_global_load;
    // Phase 28: TDM (Tensor Data Mover)-based global-to-LDS load for the A operand --
    // gfx1250-only, an alternative to async_global_load's global_load_async_to_lds_b128
    // path using the dedicated TENSOR_LOAD_TO_LDS hardware unit instead. Folded into the
    // kernel name (matches async_global_load's treatment) since it's the same category of
    // load-mechanism flag, even though it doesn't change buffer layout.
    int tdm_global_load;
    // Phase 61: 32-bit SADDR-based global loads for fwd's default (non-async, non-TDM)
    // path -- an alternative addressing mode, same category as async_global_load/
    // tdm_global_load above, folded into the kernel name for the same
    // hipModuleGetFunction-lookup reason (see this project's Phase 61 postmortem on
    // direct_store: a flag added to Python's kernel naming but missed here made
    // direct_store unreachable through the normal driver path for an entire phase --
    // added here from the start specifically to avoid repeating that).
    int saddr_global_load;
    int main_loop_interleave;
    // Phase 24: also folded into the kernel name (unlike local_prefetch_num/atomic_scope/
    // atomic_cascade/epilogue_lds_pad, which are purely internal-codegen choices that don't
    // change anything the driver needs to know) -- wmma_acc_f16 changes the WMMA kernel's
    // native-width output-role buffer from fp32 to fp16, which the driver's buffer allocation/
    // comparison logic (conv_driver.cpp's is_wmma handling) must match, and which must be
    // able to coexist as a distinct, separately-named kernel alongside the f32-accumulate
    // build of the same tile shape.
    int wmma_acc_f16;
    // Phase 27: bf16 analog of wmma_acc_f16 above -- same reasoning (folded into kernel name,
    // driver buffer-allocation impact), just gated to precision=='bf16' instead of 'fp16'.
    int wmma_acc_bf16;
    // Phase 32: s_setprio bracketing around the WMMA-issue burst -- a pure instruction-issue
    // scheduling hint with no buffer-allocation/dispatch impact, but still folded into the
    // kernel name (like tdm_global_load/main_loop_interleave) so the driver's own
    // get_kernel_name() reconstructs the same suffixed symbol name Python's codegen actually
    // emitted -- otherwise hipModuleGetFunction would look up the wrong (unsuffixed) name.
    int wmma_setprio;
    // Phase 34: packed-bf16 atomic epilogue for wrw's gemm_k_global_split path -- changes
    // the OUTPUT (grad_weight) buffer's native width from fp32 to bf16 (2 bytes), so the
    // driver's dtype_alloc_byte override and wrw_post's verification path need to know
    // about it (same category as wmma_acc_f16/wmma_acc_bf16 above), and it's folded into
    // the kernel name for the same hipModuleGetFunction-lookup reason as wmma_setprio.
    int atomic_pack_bf16;
    // Phase 25: when set, relaxes fwd's WMMA-only gemm_m%gemm_m_per_block==0 validity
    // requirement (tunable_is_valid()) and turns on the kernel's GEMM_M tail-masking codegen
    // (v_flag OOB check on the A-operand load, EXEC-masked store in the epilogue). Master-
    // config phase (new): NOW folded into the kernel name -- see local_prefetch_num/
    // atomic_scope/epilogue_lds_pad below for why this changed from the original "not
    // folded" design (real name collisions once multiple tail-flag combinations of the
    // same tile shape needed to coexist in one comprehensive config file).
    int wmma_m_tail;
    // Phase 26b: analogous to wmma_m_tail but for GEMM_N (fwd only so far) -- relaxes
    // gemm_n%gemm_n_per_block==0 and turns on B-operand load masking + a second epilogue
    // EXEC-mask guard. Independent flag (not folded into wmma_m_tail) so M-only, N-only, and
    // M+N configs can all exist. Now folded into the kernel name, same reason as wmma_m_tail.
    int wmma_n_tail;
    // Phase 35: GEMM_K tail for wrw -- relaxes gemm_k%gemm_k_per_block==0. Unlike
    // wmma_m_tail/wmma_n_tail this composes with gemm_k_global_split by construction (only
    // the last split-K shard's loop range gets extended); see gemm_k_tail/gemm_k_num_splits
    // in the wrw karg struct. Now folded into the kernel name, same reason as wmma_m_tail.
    int wmma_k_tail;
    // Phase 35: hipconv-style reduction-kernel epilogue for wrw's gemm_k_global_split path --
    // replaces the atomic epilogue with plain per-shard stores into a workspace buffer plus a
    // separate reduction kernel pass. Changes what the main kernel's epilogue produces
    // (partials, not final output), so IS folded into the kernel name (same category as
    // atomic_pack_bf16/wmma_setprio above).
    int wrw_reduction_kernel;
    // Phase 41: optional S_SLEEP_VAR stagger at kernel entry for wrw's gemm_k_global_split
    // path, scaled by (blockIdx.z mod 128) -- pure timing perturbation (no buffer/dispatch
    // impact), folded into the kernel name for the same hipModuleGetFunction-lookup reason
    // as every other folded flag above.
    int gsplit_stagger;
    // Phase 58: persistent-kernel / Stream-K proof of mechanism for wrw split-K -- see
    // docs/gfx1250_streamk_design.md. Changes how the driver launches gemm_k_global_split
    // (small constant grid.z + a claimed-shard counter workspace, instead of grid.z ==
    // the chosen split count) -- folded into the kernel name like every other flag above.
    int wrw_streamk;
    // Master-config phase (new): local_prefetch_num/atomic_scope/epilogue_lds_pad were
    // "purely internal-codegen choices" the driver never needed to know about, by the
    // ORIGINAL design -- true only as long as no two config sections of the SAME tile shape
    // differing ONLY in one of these ever needed to coexist in one file. Assembling the
    // first comprehensive per-(direction,precision) master config surfaced exactly that
    // collision (e.g. a plain vs _lp2 section of the identical tile shape). Now tracked and
    // folded into the kernel name for the same hipModuleGetFunction-lookup reason as every
    // other folded flag above. atomic_scope is a string ('SCOPE_SYS' default, 'SCOPE_DEV'
    // for wrw's _scopedev configs) -- stored as-is, not reduced to a bool, so the fold logic
    // can match Python's exact string comparison.
    int local_prefetch_num = 1;
    int epilogue_lds_pad = 0;
    // Phase 59: direct per-lane global_store_dword epilogue (skips the LDS-reshuffle
    // gather/scatter). Changes the epilogue's generated code just like
    // wrw_reduction_kernel/atomic_pack_bf16 above, so must be folded into the kernel name
    // -- this was missed when Phase 59 landed: kernel_name never gained a "_direct" suffix
    // here (see below), so hipModuleGetFunction either silently found a same-named
    // non-direct sibling kernel (whenever one happened to exist in the same build, e.g.
    // every master combinatorial config) or failed outright with "named symbol not found"
    // (whenever it didn't, e.g. every standalone hand-curated *_direct.config) -- direct
    // store was never actually exercised through the normal driver search path. Found
    // while hardware-validating the P2 (direct_store config expansion) backlog item.
    int direct_store = 0;
    // Phase 63: ds_load_tr16_b128-based B-operand LDS transpose read (bwd, fp16/bf16
    // only) -- replaces shared_load_b_functor's manual per-element read+pack loop.
    // Changes the generated kernel's code (not just addressing internals the driver is
    // ignorant of), so folded into the kernel name from the start -- see this project's
    // own direct_store postmortem right above for why that specific omission was costly.
    int ds_load_tr_b = 0;
    // Default-member-initialized (unlike the int fields above, which zero-init safely via
    // aggregate-init anyway) so a default-constructed igemm_gtc_tunable_t{} -- e.g.
    // driver_mode_heuristic's still-unimplemented heuristic_select_kernel() stub -- doesn't
    // produce an empty-string atomic_scope that would incorrectly fold into "_ascope" with
    // no scope name.
    std::string atomic_scope = "SCOPE_SYS";
} igemm_gtc_tunable_t;

static inline std::string get_igemm_gtc_fma_type(std::string arch_string, const config_section_t &sec){
    if(sec.count("lanegroup_tile_m") > 0 && sec.count("lanegroup_tile_n") > 0){
        if(arch_string == "gfx900")
            return IGEMM_GTC_TUNABLE_FMA_TYPE_MAC;
        if(arch_string == "gfx906" || arch_string == "gfx1030")
            return IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS;
        if(arch_string == "gfx908" || arch_string == "gfx90a" || arch_string == "gfx940" || arch_string == "gfx942" || arch_string == "gfx950")
            return IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS;
    }else if(sec.count("wave_tile_m") > 0 && sec.count("wave_tile_n") > 0){
        assert(arch_string == "gfx908" || arch_string == "gfx90a" || arch_string == "gfx940" || arch_string == "gfx942" || arch_string == "gfx950");
        return IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS;
    }else if(sec.count("wmma_tile_m") > 0 && sec.count("wmma_tile_n") > 0){
        assert(arch_string == "gfx1250");
        return IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA;
    }
    return IGEMM_GTC_TUNABLE_FMA_TYPE_NA;
}

static inline std::vector<igemm_gtc_tunable_t>
igemm_gtc_tunable_from_config(const config_content_t &content) {
    std::vector<igemm_gtc_tunable_t> tunables;
    config_section_t codegen_sec = content.get_section("codegen");
    assert(codegen_sec.get_name() == "codegen");
    for (const auto &sec : content) {
        if (sec.get_name() == "igemm_fwd_gtc" ||
            sec.get_name() == "igemm_bwd_gtc" || 
            sec.get_name() == "igemm_wrw_gtc")
        {
            igemm_gtc_tunable_t tunable;
            tunable.tensor_layout            = sec.count("tensor_layout") > 0 ? sec.at("tensor_layout").get_string() : "nchw";
            tunable.gemm_m_per_block         = sec.at("gemm_m_per_block").get_int();
            tunable.gemm_n_per_block         = sec.at("gemm_n_per_block").get_int();
            tunable.gemm_k_per_block         = sec.at("gemm_k_per_block").get_int();
            tunable.fma_type                 = get_igemm_gtc_fma_type(codegen_sec.at("arch").get_string(), sec);
            assert(tunable.fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_NA);
            if(tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC){
                tunable.gemm_m_per_thread        = sec.at("gemm_m_per_thread").get_int();
                tunable.gemm_m_level0_cluster    = sec.at("gemm_m_level0_cluster").get_int();
                tunable.gemm_m_level1_cluster    = sec.at("gemm_m_level1_cluster").get_int();
                tunable.gemm_n_per_thread        = sec.at("gemm_n_per_thread").get_int();
                tunable.gemm_n_level0_cluster    = sec.at("gemm_n_level0_cluster").get_int();
                tunable.gemm_n_level1_cluster    = sec.at("gemm_n_level1_cluster").get_int();
            }else if(tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS){
                tunable.lanegroup_tile_m        = sec.at("lanegroup_tile_m").get_int();
                tunable.lanegroup_wave_m    = sec.at("lanegroup_wave_m").get_int();
                tunable.lanegroup_repeat_m    = sec.at("lanegroup_repeat_m").get_int();
                tunable.lanegroup_tile_n        = sec.at("lanegroup_tile_n").get_int();
                tunable.lanegroup_wave_n    = sec.at("lanegroup_wave_n").get_int();
                tunable.lanegroup_repeat_n    = sec.at("lanegroup_repeat_n").get_int();
            }
            else if(tunable.fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA){
                tunable.wave_tile_m              = sec.at("wmma_tile_m").get_int();
                tunable.wave_step_m              = 0;
                tunable.wave_repeat_m            = sec.at("wmma_repeat_m").get_int();
                tunable.wave_tile_n              = sec.at("wmma_tile_n").get_int();
                tunable.wave_step_n              = 0;
                tunable.wave_repeat_n            = sec.at("wmma_repeat_n").get_int();
                tunable.wave_tile_k              = 0;
                tunable.lds_double_buffer        = sec.count("lds_double_buffer") > 0 ? sec.at("lds_double_buffer").get_int() : 0;
                tunable.async_global_load        = sec.count("async_global_load") > 0 ? sec.at("async_global_load").get_int() : 0;
                tunable.tdm_global_load          = sec.count("tdm_global_load") > 0 ? sec.at("tdm_global_load").get_int() : 0;
                tunable.saddr_global_load         = sec.count("saddr_global_load") > 0 ? sec.at("saddr_global_load").get_int() : 0;
                tunable.main_loop_interleave     = sec.count("main_loop_interleave") > 0 ? sec.at("main_loop_interleave").get_int() : 0;
                tunable.wmma_acc_f16             = sec.count("wmma_acc_f16") > 0 ? sec.at("wmma_acc_f16").get_int() : 0;
                tunable.wmma_acc_bf16             = sec.count("wmma_acc_bf16") > 0 ? sec.at("wmma_acc_bf16").get_int() : 0;
                tunable.wmma_m_tail               = sec.count("wmma_m_tail") > 0 ? sec.at("wmma_m_tail").get_int() : 0;
                tunable.wmma_n_tail               = sec.count("wmma_n_tail") > 0 ? sec.at("wmma_n_tail").get_int() : 0;
                tunable.wmma_setprio               = sec.count("wmma_setprio") > 0 ? sec.at("wmma_setprio").get_int() : 0;
                tunable.atomic_pack_bf16           = sec.count("atomic_pack_bf16") > 0 ? sec.at("atomic_pack_bf16").get_int() : 0;
                tunable.wmma_k_tail                = sec.count("wmma_k_tail") > 0 ? sec.at("wmma_k_tail").get_int() : 0;
                tunable.wrw_reduction_kernel       = sec.count("wrw_reduction_kernel") > 0 ? sec.at("wrw_reduction_kernel").get_int() : 0;
                tunable.gsplit_stagger              = sec.count("gsplit_stagger") > 0 ? sec.at("gsplit_stagger").get_int() : 0;
                tunable.wrw_streamk                 = sec.count("wrw_streamk") > 0 ? sec.at("wrw_streamk").get_int() : 0;
                tunable.local_prefetch_num         = sec.count("local_prefetch_num") > 0 ? sec.at("local_prefetch_num").get_int() : 1;
                tunable.epilogue_lds_pad           = sec.count("epilogue_lds_pad") > 0 ? sec.at("epilogue_lds_pad").get_int() : 0;
                tunable.direct_store               = sec.count("direct_store") > 0 ? sec.at("direct_store").get_int() : 0;
                tunable.ds_load_tr_b                = sec.count("ds_load_tr_b") > 0 ? sec.at("ds_load_tr_b").get_int() : 0;
                tunable.atomic_scope               = sec.count("atomic_scope") > 0 ? sec.at("atomic_scope").get_string() : "SCOPE_SYS";
            }
            else{
                tunable.wave_tile_m              = sec.at("wave_tile_m").get_int();
                tunable.wave_step_m              = sec.at("wave_step_m").get_int();
                tunable.wave_repeat_m            = sec.at("wave_repeat_m").get_int();
                tunable.wave_tile_n              = sec.at("wave_tile_n").get_int();
                tunable.wave_step_n              = sec.at("wave_step_n").get_int();
                tunable.wave_repeat_n            = sec.at("wave_repeat_n").get_int();
                tunable.wave_tile_k              = sec.count("wave_tile_k") > 0 ? sec.at("wave_tile_k").get_int() : 1;
            }
            tunable.tensor_a_pass_through    = sec.count("tensor_a_pass_through") > 0 ? sec.at("tensor_a_pass_through").get_int() : 0;
            tunable.tensor_b_pass_through    = sec.count("tensor_b_pass_through") > 0 ? sec.at("tensor_b_pass_through").get_int() : 0;
            tunable.tensor_a_thread_lengths  = sec.at("tensor_a_thread_lengths").get_list_int();
            tunable.tensor_a_cluster_lengths = sec.at("tensor_a_cluster_lengths").get_list_int();
            tunable.tensor_b_thread_lengths  = sec.at("tensor_b_thread_lengths").get_list_int();
            tunable.tensor_b_cluster_lengths = sec.at("tensor_b_cluster_lengths").get_list_int();
            tunable.direction                = sec.at("direction").get_string();
            tunable.precision                = sec.at("precision").get_string();
            tunable.nxb                      = sec.at("nxb").get_int();
            tunable.nxe                      = sec.at("nxe").get_int();
            tunable.gemm_m_unmerge_cluster   = sec.count("gemm_m_unmerge_cluster") > 0 ? sec.at("gemm_m_unmerge_cluster").get_int() : 0;
            tunable.gemm_n_unmerge_cluster   = sec.count("gemm_n_unmerge_cluster") > 0 ? sec.at("gemm_n_unmerge_cluster").get_int() : 0;
            tunable.gemm_k_unmerge_cluster   = sec.count("gemm_k_unmerge_cluster") > 0 ? sec.at("gemm_k_unmerge_cluster").get_int() : 0;
            int default_mh                   = tunable.direction == "bwd" && tunable.tensor_layout == "nhwc" && tunable.nxe != 0 ? 1 : 0;
            tunable.multihead                = sec.count("multihead") > 0 ? sec.at("multihead").get_int() : default_mh;
            int default_source_access_order  = tunable.direction == "fwd" ? 1 : 0;
            tunable.source_access_order      = sec.count("source_access_order") > 0 ? sec.at("source_access_order").get_int() : default_source_access_order;
            tunable.vector_store             = sec.count("vector_store") > 0 ? sec.at("vector_store").get_int() : 0;
            tunable.gemm_k_global_split      = sec.count("gemm_k_global_split") > 0 ? sec.at("gemm_k_global_split").get_int() : 0;
            tunable.merge_e                  = sec.count("merge_e") > 0 ? sec.at("merge_e").get_int() : 0;
            tunable.vector_c                 = sec.count("vector_c") > 0 ? sec.at("vector_c").get_int() : 1;
            tunables.push_back(tunable);
        }
    }
    return tunables;
}

static inline int get_gcn_arch(char* archname)
{
    int gcn_arch = 1000;
    if (!strncmp("gfx908", archname, 6)){
        gcn_arch = 908;
    }
    else if (!strncmp("gfx90a", archname, 6)){
        gcn_arch = 910;
    }
    else if (!strncmp("gfx940", archname, 6)){
        gcn_arch = 940;
    }
    else if (!strncmp("gfx941", archname, 6)){
        gcn_arch = 941;
    }
    else if (!strncmp("gfx942", archname, 6)){
        gcn_arch = 942;
    }
    else if (!strncmp("gfx950", archname, 6)){
        gcn_arch = 950;
    }
    else if (!strncmp("gfx1250", archname, 7)){
        gcn_arch = 1250;
    }
    return gcn_arch;
}

static inline std::string
igemm_gtc_encode_kernel_name(const igemm_gtc_tunable_t *tunable) {
    auto tensor_layout            = tunable->tensor_layout;
    auto gemm_m_per_block         = tunable->gemm_m_per_block;
    auto gemm_n_per_block         = tunable->gemm_n_per_block;
    auto gemm_k_per_block         = tunable->gemm_k_per_block;
    auto fma_type                 = tunable->fma_type;
    // auto gemm_m_per_thread        = tunable->gemm_m_per_thread;
    // auto gemm_m_level0_cluster    = tunable->gemm_m_level0_cluster;
    // auto gemm_m_level1_cluster    = tunable->gemm_m_level1_cluster;
    // auto gemm_n_per_thread        = tunable->gemm_n_per_thread;
    // auto gemm_n_level0_cluster    = tunable->gemm_n_level0_cluster;
    // auto gemm_n_level1_cluster    = tunable->gemm_n_level1_cluster;
    auto tensor_a_pass_through    = tunable->tensor_a_pass_through;
    auto tensor_b_pass_through    = tunable->tensor_b_pass_through;
    auto tensor_a_thread_lengths  = tunable->tensor_a_thread_lengths;
    auto tensor_a_cluster_lengths = tunable->tensor_a_cluster_lengths;
    auto tensor_b_thread_lengths  = tunable->tensor_b_thread_lengths;
    auto tensor_b_cluster_lengths = tunable->tensor_b_cluster_lengths;
    auto direction                = tunable->direction;
    auto precision                = tunable->precision;
    auto nxb                      = tunable->nxb;
    auto nxe                      = tunable->nxe;
    auto gemm_m_unmerge_cluster   = tunable->gemm_m_unmerge_cluster;
    auto gemm_n_unmerge_cluster   = tunable->gemm_n_unmerge_cluster;
    auto gemm_k_unmerge_cluster   = tunable->gemm_k_unmerge_cluster;
    auto source_access_order      = tunable->source_access_order;
    auto multihead                = tunable->multihead;
    auto vector_store             = tunable->vector_store;
    auto gemm_k_global_split      = tunable->gemm_k_global_split;
    auto merge_e                  = tunable->merge_e;

    static int gcn_arch = -1;
    if(gcn_arch == -1){
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice(&dev));
        HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
        gcn_arch = get_gcn_arch(dev_prop.gcnArchName);
    }

    std::string kernel_name = std::string("igemm_") + direction + "_";
    if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC)
        kernel_name += "gtcm_";
    else if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS)
        if(gcn_arch == 1030)
            kernel_name += "gtcn2_";
        else
            kernel_name += "gtc_";
    else if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS){
        if(gcn_arch == 908)
            kernel_name += "gtcx_";
        else if(gcn_arch == 910)
            kernel_name += "gtcx2_";
        else if(gcn_arch == 940 || gcn_arch == 941 || gcn_arch == 942)
            kernel_name += "gtcx3_";
        else if(gcn_arch == 950)
            kernel_name += "gtcx35_";
    }
    else if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA){
        if(gcn_arch == 1250)
            kernel_name += "gtcw_";
    }
    std::string vector_c_str = "";
    if(tunable->vector_c > 1)
        vector_c_str += std::string("x") + std::to_string(tunable->vector_c);

    kernel_name += tensor_layout + std::string("_") + precision + vector_c_str +
        std::string("_bx") + std::to_string(nxb) + 
        std::string("_ex") + std::to_string(nxe) +
#if USE_SOURCE_ACCESS_ENCODING_KERNEL_NAME
        std::string("_sa") + std::to_string(source_access_order) + "_";
#else
        "_";
#endif

    kernel_name += std::string("bt") +
            std::to_string(gemm_m_per_block) + "x" +
            std::to_string(gemm_n_per_block) + "x" +
            std::to_string(gemm_k_per_block) + "_";

    if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC){
        auto gemm_m_per_thread        = tunable->gemm_m_per_thread;
        auto gemm_m_level0_cluster    = tunable->gemm_m_level0_cluster;
        auto gemm_m_level1_cluster    = tunable->gemm_m_level1_cluster;
        auto gemm_n_per_thread        = tunable->gemm_n_per_thread;
        auto gemm_n_level0_cluster    = tunable->gemm_n_level0_cluster;
        auto gemm_n_level1_cluster    = tunable->gemm_n_level1_cluster;
        assert(gemm_m_per_block % (gemm_m_per_thread * gemm_m_level0_cluster * gemm_m_level1_cluster) == 0);
        assert(gemm_n_per_block % (gemm_n_per_thread * gemm_n_level0_cluster * gemm_n_level1_cluster) == 0);
        int gemm_m_repeat = gemm_m_per_block / (gemm_m_per_thread * gemm_m_level0_cluster * gemm_m_level1_cluster);
        int gemm_n_repeat = gemm_n_per_block / (gemm_n_per_thread * gemm_n_level0_cluster * gemm_n_level1_cluster);

        int thread_tile_m = gemm_m_repeat * gemm_m_per_thread;
        int thread_tile_n = gemm_n_repeat * gemm_n_per_thread;
        kernel_name += std::string("tt") +
            std::to_string(thread_tile_m) + "x" +
            std::to_string(thread_tile_n) + "_" +
            "gm" + 
            std::to_string(gemm_m_repeat) + "x" +
            std::to_string(gemm_m_level0_cluster) + "x" +
            std::to_string(gemm_m_level1_cluster) + "_" +
            "gn" + 
            std::to_string(gemm_n_repeat) + "x" +
            std::to_string(gemm_n_level0_cluster) + "x" +
            std::to_string(gemm_n_level1_cluster) + "_";
    }else if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS){
        kernel_name +=   std::string("lt") + std::to_string(tunable->lanegroup_tile_m) + "x" + std::to_string(tunable->lanegroup_tile_n) + "_" + 
                         "lw" + std::to_string(tunable->lanegroup_wave_m) + "x" + std::to_string(tunable->lanegroup_wave_n) + "_" +
                         "lr" + std::to_string(tunable->lanegroup_repeat_m) + "x" + std::to_string(tunable->lanegroup_repeat_n) + "_";
    }else if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS){
        kernel_name +=   std::string("wt") + std::to_string(tunable->wave_tile_m) + "x" + std::to_string(tunable->wave_tile_n) + "x" + std::to_string(tunable->wave_tile_k) + "_" +
                         "ws" + std::to_string(tunable->wave_step_m) + "x" + std::to_string(tunable->wave_step_n) + "_" +
                         "wr" + std::to_string(tunable->wave_repeat_m) + "x" + std::to_string(tunable->wave_repeat_n) + "_";
    }else if (tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA){
        // wave_tile_m/n and wave_repeat_m/n hold wmma_tile_m/n and wmma_repeat_m/n (see union comment above)
        kernel_name +=   std::string("wt") + std::to_string(tunable->wave_tile_m) + "x" + std::to_string(tunable->wave_tile_n) + "_" +
                         "wr" + std::to_string(tunable->wave_repeat_m) + "x" + std::to_string(tunable->wave_repeat_n) + "_";
    }

    kernel_name +=
            "ta" + utility_int_list_to_string(tensor_a_thread_lengths) + "_" +
                    utility_int_list_to_string(tensor_a_cluster_lengths)+ "_" +
            "tb" + utility_int_list_to_string(tensor_b_thread_lengths) + "_" +
                    utility_int_list_to_string(tensor_b_cluster_lengths);
    // printf("[%s]\n",kernel_name.c_str());
    // Phase 16: mirrors igemm_base.py's igemm_gtc_encode_kernel_name -- must stay in sync,
    // see this struct's lds_double_buffer/async_global_load/main_loop_interleave comment.
    if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA){
        if(tunable->lds_double_buffer)
            kernel_name += std::string("_dbuf");
        if(tunable->async_global_load)
            kernel_name += std::string("_async");
        if(tunable->tdm_global_load)
            kernel_name += std::string("_tdm");
        if(tunable->saddr_global_load)
            kernel_name += std::string("_saddr");
        if(tunable->main_loop_interleave)
            kernel_name += std::string("_interleave");
        if(tunable->wmma_acc_f16)
            kernel_name += std::string("_f16acc");
        if(tunable->wmma_acc_bf16)
            kernel_name += std::string("_bf16acc");
        if(tunable->wmma_setprio)
            kernel_name += std::string("_setprio");
        if(tunable->atomic_pack_bf16)
            kernel_name += std::string("_pkatomic");
        if(tunable->wrw_reduction_kernel)
            kernel_name += std::string("_wsred");
        if(tunable->gsplit_stagger)
            kernel_name += std::string("_stagger");
        if(tunable->wrw_streamk)
            kernel_name += std::string("_streamk");
        // mirrors igemm_base.py's identical extension -- must stay in sync.
        if(tunable->wmma_m_tail)
            kernel_name += std::string("_mtail");
        if(tunable->wmma_n_tail)
            kernel_name += std::string("_ntail");
        if(tunable->wmma_k_tail)
            kernel_name += std::string("_ktail");
        // mirrors igemm_base.py's identical extension -- must stay in sync.
        if(tunable->epilogue_lds_pad)
            kernel_name += std::string("_ldspad");
        if(tunable->direct_store)
            kernel_name += std::string("_direct");
        if(tunable->ds_load_tr_b)
            kernel_name += std::string("_dstrb");
        if(tunable->local_prefetch_num != 1)
            kernel_name += std::string("_lp") + std::to_string(tunable->local_prefetch_num);
        if(tunable->atomic_scope != "SCOPE_SYS")
            kernel_name += (tunable->atomic_scope == "SCOPE_DEV") ? std::string("_scopedev") : (std::string("_ascope") + tunable->atomic_scope);
    }
    if(tensor_a_pass_through)
        kernel_name += std::string("_pta");
    if(tensor_b_pass_through)
        kernel_name += std::string("_ptb");
    if(gemm_m_unmerge_cluster)
        kernel_name += std::string("_mc");
    if(gemm_n_unmerge_cluster)
        kernel_name += std::string("_nc");
    if(gemm_k_unmerge_cluster)
        kernel_name += std::string("_kc");
    if(multihead)
        kernel_name += std::string("_mh");
    if(merge_e)
        kernel_name += std::string("_me");
    // when split in gemmk, we need call atomic add function
    if(vector_store)
        kernel_name += std::string("_vs") + std::to_string(vector_store);
    if(gemm_k_global_split > 0)
        kernel_name += std::string("_gkgs");
    return kernel_name;
}

static inline float igemm_launch_kernel_single(hipFunction_t kernel_func, void* args, size_t arg_size, std::vector<size_t> grid_size, std::vector<size_t> block_size)
{
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
                        HIP_LAUNCH_PARAM_END};
    float ms = .0;

    hipEvent_t start;
    hipEvent_t stop;
    
    HIP_CALL(hipEventCreate(&start));
    HIP_CALL(hipEventCreate(&stop));

    // for hipHccModuleLaunchKernel/hipExtModuleLaunchKernel, the grid_size is in unit of workitem
    HIP_CALL(hipExtModuleLaunchKernel(kernel_func, grid_size[0], grid_size[1], grid_size[2],
                                        block_size[0], block_size[1], block_size[2], 0, 0, NULL,
                                        (void **)&config, start, stop,0));


    HIP_CALL(hipEventSynchronize(stop));
    HIP_CALL(hipEventElapsedTime(&ms, start, stop));
    HIP_CALL(hipEventDestroy(start));
    HIP_CALL(hipEventDestroy(stop));

    return ms;
}

static inline float igemm_launch_kernel(hipFunction_t kernel_func, void* args, size_t arg_size, std::vector<size_t> grid_size, std::vector<size_t> block_size, int warmup, int repeat)
{
    assert(repeat > 2);
    std::vector<float> duration_list;
    for (int i = 0; i < warmup; i++) {
        igemm_launch_kernel_single(kernel_func, args, arg_size, grid_size, block_size);
    }

    for (int i = 0; i < repeat; i++) {
        float d = igemm_launch_kernel_single(kernel_func, args, arg_size, grid_size, block_size);
        duration_list.push_back(d);
    }
    // remove min and max from list, then do average
    auto imin = std::min_element(begin(duration_list), end(duration_list));
    duration_list.erase(imin);
    auto imax = std::max_element(begin(duration_list), end(duration_list));
    duration_list.erase(imax);

    assert(duration_list.size() == (repeat - 2));
    float avg_duration = std::accumulate(duration_list.begin(), duration_list.end(), (float).0) / duration_list.size();
    return avg_duration;
}

struct igemm_launch_kernel_t {
    hipFunction_t           kernel_func;
    void *                  args;
    size_t                  arg_size;
    std::vector<size_t>     grid_size;
    std::vector<size_t>     block_size;

    operator dispatchinfo_t() const {
        if (grid_size.size() != 3 || block_size.size() != 3) {
            std::cout << "ERROR: dispatch size n_dims !=3 (grid_size.size() != 3 || block_size.size() != 3)" << std::endl;
            assert(0);
        }
        if (arg_size > MAX_KARG_DUMP_BYTES) {
            std::cout << "ERROR: arg_size > MAX_KARG_DUMP_BYTES" << std::endl;
            assert(0);
        }
        dispatchinfo_t di;
        di.gi.gsize = {static_cast<uint32_t>(grid_size[0]),
                       static_cast<uint32_t>(grid_size[1]),
                       static_cast<uint32_t>(grid_size[2])};
        di.gi.wsize = {static_cast<uint32_t>(block_size[0]),
                       static_cast<uint32_t>(block_size[1]),
                       static_cast<uint32_t>(block_size[2])};
        di.karg_size = arg_size;
        di.ktype = kargtype_t::unknown;
        memset(di.karg_dump, 0, MAX_KARG_DUMP_BYTES);
        memcpy(di.karg_dump, args, arg_size);
        
        return di;
    }
};

template<typename prolog_kernel_t, typename postlog_kernel_t>
static inline float igemm_launch_kernels(const std::vector<igemm_launch_kernel_t> & kernels, prolog_kernel_t prolog_kernel, postlog_kernel_t postlog_kernel, int warmup, int repeat)
{
    auto launch_kernels = [&]() -> float{
        float ms = .0;
        ms += prolog_kernel();
        for(const auto & ker :  kernels){
            float t = igemm_launch_kernel_single(ker.kernel_func, ker.args, ker.arg_size, ker.grid_size, ker.block_size);
            //std::cout << ker.kernel_func << ": " << t << std::endl;
            ms += t;
        }
        ms += postlog_kernel();
        return ms;
    };

    std::vector<float> duration_list;
    for (int i = 0; i < warmup; i++) {
        launch_kernels();
    }

    for (int i = 0; i < repeat; i++) {
        float d = launch_kernels();
        duration_list.push_back(d);
    }
    if(repeat > 2){
        // remove min and max from list, then do average
        auto imin = std::min_element(begin(duration_list), end(duration_list));
        duration_list.erase(imin);
        auto imax = std::max_element(begin(duration_list), end(duration_list));
        duration_list.erase(imax);
    }
    
    float avg_duration = std::accumulate(duration_list.begin(), duration_list.end(), (float).0) / duration_list.size();
    return avg_duration;
}

static inline int igemm_get_max_gks(int gemm_k, int gemm_k_per_block, int max_log2_splits)
{
    if(gemm_k % gemm_k_per_block != 0)
        return 0;
    int rem = gemm_k / gemm_k_per_block;
    // to find the highest power of 2 value that can divide rem
    // https://www.geeksforgeeks.org/highest-power-of-two-that-divides-a-given-number/
    int rem_pow2 = rem & (~(rem - 1));
    int gks = (int)log2(rem_pow2);
    if(gks > max_log2_splits)
        gks = max_log2_splits;
    return gks;
}

// Phase 50: sane upper bound on gfx1250 WMMA gemm_k_global_split's split count, shared by
// bwd/fwd/wrw's WMMA launch paths. Splitting the reduction axis across grid.z provides real
// parallelism/occupancy benefit only up to a point; beyond it, atomic-add contention
// dominates and further splitting can only hurt. Two independent caps combine (the tighter
// one applies):
//  - an absolute ceiling (gfx1250 has 256 CUs; splitting far beyond a modest multiple of
//    that has no more real parallelism left to exploit). 4096 is comfortably above the
//    largest split count previously measured-and-still-beneficial (wrw's Phase 41
//    gsplit_stagger testing went up to 1260 splits), while cutting off genuinely runaway
//    values -- an extreme small-gemm_m/n, huge-gemm_k wrw shape was observed picking a
//    135000-way split (and taking minutes just to real-launch-time that one candidate)
//    before this cap existed, see docs/gfx1250_wmma_layout.md's Phase 50.
//  - a minimum K-elements-per-shard floor: each shard's own K-slice should still cover a
//    "real" amount of reduction work to amortize the atomic-add's fixed per-shard overhead,
//    independent of gemm_k_per_block. This is what actually fixes fp32
//    (gemm_k_per_block=4) over-splitting relative to bf16/fp16 (gemm_k_per_block=32): a
//    total-workgroup-count-only heuristic can't see that fp32's num_k_blocks is 8x larger
//    for the same gemm_k, so the same "target split count" gives fp32 8x-finer real shards
//    than bf16/fp16 at the identical shape -- bounding K elements per shard directly (32,
//    matching bf16/fp16's own native gemm_k_per_block) closes this precision-dependent
//    blind spot without needing a real per-precision launch sweep to discover it.
static inline int igemm_gemm_k_global_split_cap(int gemm_k, int gemm_k_per_block)
{
    constexpr int MAX_GEMM_K_SPLITS = 4096;
    constexpr int MIN_GEMM_K_PER_SHARD = 32;
    int num_k_blocks = std::max(1, gemm_k / gemm_k_per_block);
    int max_by_min_shard = std::max(1, gemm_k / MIN_GEMM_K_PER_SHARD);
    return std::min({MAX_GEMM_K_SPLITS, num_k_blocks, max_by_min_shard});
}

// this is to support big tensor > 4G. need to decide how many splits needed
// return the number of splits
static inline size_t igemm_split_batch_size(const args_t *arg, int data_byte)
{
    int hi = arg->get_int("in_h");
    int wi = arg->get_int("in_w");
    int n = arg->get_int("batchsize");
    int k = arg->get_int("out_channels");
    int c = arg->get_int("in_channels");

    int stride_h = arg->get_int("conv_stride_h");
    int stride_w = arg->get_int("conv_stride_w");
    int dilation_h = arg->get_int("dilation_h");
    int dilation_w = arg->get_int("dilation_w");
    int pad_h = arg->get_int("pad_h");
    int pad_w = arg->get_int("pad_w");
    int y = arg->get_int("fil_h");
    int x = arg->get_int("fil_w");
    int ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
    int wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);

    // int data_byte = utility_string_to_data_byte(tunable->precision);
    size_t image_size_input = static_cast<size_t>(c) * hi * wi * data_byte;
    size_t image_size_output = static_cast<size_t>(k) * ho * wo * data_byte;
    size_t size_4g = 0xffffffffUL;
    if(image_size_input >= size_4g || image_size_output >= size_4g)
        return 0;

    size_t image_size = image_size_input >= image_size_output ? image_size_input : image_size_output;
    size_t splited_n = size_4g / image_size;

    // round up splits, we must match
    // 1. splited_n * image_size < size_4g
    // 2. n % splited_n == 0
    // if(splited_n >= n)
    //     return 1;
    assert(splited_n != 0);
    while(splited_n >= 1){
        // printf("n:%d, splited_n:%d\n", n, splited_n);
        if(n % splited_n == 0 && splited_n * image_size < size_4g)
            break;
        splited_n--;
    }
    assert(splited_n * image_size < size_4g && n % splited_n == 0);
    return static_cast<size_t>(n) / splited_n;
}

#define SPATIAL_TILING_FLAG_TLE     0   // input section size should <= a value in var (hi|lo, hi->tile in h, lo->tile in w)
#define SPATIAL_TILING_FLAG_TEQ     1   // tile size equal to value in var (hi|lo, hi->tile in h, lo->tile in w)

struct igemm_spatial_tiling_t {
    uint32_t tile_w {0};
    uint32_t tile_h {0};
};

static inline uint32_t
igemm_find_tile_size_with_upper_bound(uint32_t out_size, size_t upper_bound,
                uint32_t stride, uint32_t dilation, uint32_t filter)
{
    // return tile size so that the required input tile(sec_in) is no larger than upper_bound
    uint32_t n_tiles = 1; 
    for( ; n_tiles <= out_size ; n_tiles++){
        uint32_t tile_size = (out_size + n_tiles - 1) / n_tiles;
        uint32_t sec_in = (tile_size - 1) * stride + 1 + dilation * (filter - 1);
        if(sec_in <= upper_bound)
            break;
    }

    return (out_size + n_tiles - 1) / n_tiles;
}

static inline igemm_spatial_tiling_t
igemm_spatial_tiling(const args_t *arg, uint32_t flag, uint32_t var)
{
    int hi = arg->get_int("in_h");
    int wi = arg->get_int("in_w");

    int stride_h = arg->get_int("conv_stride_h");
    int stride_w = arg->get_int("conv_stride_w");
    int dilation_h = arg->get_int("dilation_h");
    int dilation_w = arg->get_int("dilation_w");
    int pad_h = arg->get_int("pad_h");
    int pad_w = arg->get_int("pad_w");
    int y = arg->get_int("fil_h");
    int x = arg->get_int("fil_w");
    int ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
    int wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);

    igemm_spatial_tiling_t tiling;

    int tile_y = env_get_int("TILE_Y", 0);
    int tile_x = env_get_int("TILE_X", 0);
    if(tile_x != 0 && tile_y != 0){
        flag = SPATIAL_TILING_FLAG_TEQ;
        var = (tile_y << 16) | tile_x;
    }

    if(flag == SPATIAL_TILING_FLAG_TLE){
        uint32_t size_h = var >> 16;
        uint32_t size_w = var & 0xffff;

        tiling.tile_h = igemm_find_tile_size_with_upper_bound(ho, size_h, stride_h, dilation_h, y);
        tiling.tile_w = igemm_find_tile_size_with_upper_bound(wo, size_w, stride_w, dilation_w, x);
    }
    else if(flag == SPATIAL_TILING_FLAG_TEQ){
        uint32_t size_h = var >> 16;
        uint32_t size_w = var & 0xffff;

        assert(size_h <= ho);
        assert(size_w <= wo);

        tiling.tile_h = size_h;
        tiling.tile_w = size_w;
    }

    return tiling;
}

class igemm_driver_base_t{
public:
    igemm_driver_base_t(hipModule_t module_tensor_cast_, hipModule_t module_, driver_mode_t driver_mode_, driverDataType_t data_type_, int warmup_, int repeat_, bool verbose_) : 
        module_tensor_cast(module_tensor_cast_), module(module_), driver_mode(driver_mode_), data_type(data_type_), warmup(warmup_), repeat(repeat_), verbose(verbose_)
    {
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice(&dev));
        HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
        this->num_cu = dev_prop.multiProcessorCount;
        gcn_arch = get_gcn_arch(dev_prop.gcnArchName);
        if(this->gcn_arch >= 1000)
            this->num_cu *= 2;
        max_mpb = -1;
        max_npb = -1;
        max_kpb = -1;
        max_gks = -1;
        vector_c = 1;
    }
    std::string get_kernel_name(const igemm_gtc_tunable_t *tunable) {
        return igemm_gtc_encode_kernel_name(tunable);
    }

    size_t get_workspace_size(const args_t *arg, const igemm_gtc_tunable_t *tunable){
        int hi = arg->get_int("in_h");
        int wi = arg->get_int("in_w");
        int n = arg->get_int("batchsize");
        int k = arg->get_int("out_channels");
        int c = arg->get_int("in_channels");

        int stride_h = arg->get_int("conv_stride_h");
        int stride_w = arg->get_int("conv_stride_w");
        int dilation_h = arg->get_int("dilation_h");
        int dilation_w = arg->get_int("dilation_w");
        int pad_h = arg->get_int("pad_h");
        int pad_w = arg->get_int("pad_w");
        int y = arg->get_int("fil_h");
        int x = arg->get_int("fil_w");
        int ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
        int wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);
        int group = arg->get_int("group_count");
        int forw = arg->get_int("forw");

        size_t workspace_size = 0;
        if(forw & 1) // forward ws size
        {
            if(tunable->precision == "fp16" && tunable->gemm_k_global_split == 1 && tunable->vector_store == 1)
                workspace_size = static_cast<size_t>(n) * k * ho * wo;
            else if(tunable->precision == "bf16" && tunable->gemm_k_global_split == 1)
                workspace_size = static_cast<size_t>(n) * k * ho * wo;
        }
        else if(forw & 2) // backward data ws size
        {
            if(tunable->precision == "fp16" && tunable->gemm_k_global_split == 1 && tunable->vector_store == 1)
                workspace_size = static_cast<size_t>(n) * c * hi * wi;
            else if(tunable->precision == "bf16" && tunable->gemm_k_global_split == 1)
                workspace_size = static_cast<size_t>(n) * c * hi * wi;
        }
        else if(forw & 4) // backward weights ws size
        {
            if(tunable->precision == "fp16" && tunable->gemm_k_global_split == 1 && (tunable->tensor_b_thread_lengths[3] == 1 || tunable->vector_store == 1))
                workspace_size = static_cast<size_t>(group) * (k / group) * (c / group) * y * x;
            else if(tunable->precision == "bf16" && tunable->gemm_k_global_split == 1)
                workspace_size = static_cast<size_t>(group) * (k / group) * (c / group) * y * x;
        }
        else if(forw == 0) // all dirs
        {
            std::cout << "not support direction" << std::endl;
            assert(false);
        }
        else
        {
            std::cout << "wrong direction" << std::endl;
            assert(false);
        }
        return workspace_size * sizeof(float);
    }

    void set_block_tile_boundary(int max_mpb_, int max_npb_, int max_kpb_, int max_gks_){
        // CAUSTION! when setting this value to none -1, you need to understand what will happen
        this->max_mpb = max_mpb_;
        this->max_npb = max_npb_;
        this->max_kpb = max_kpb_;
        this->max_gks = max_gks_;
    }

    void set_vector_c(int vector_c_){
        this->vector_c = vector_c_;
    }

    virtual size_t get_block_size(const igemm_gtc_tunable_t *tunable) = 0;
    virtual size_t get_grid_size(const args_t *arg, const igemm_gtc_tunable_t *tunable) = 0;
    virtual bool tunable_is_valid(const args_t *arg, const igemm_gtc_tunable_t *tunable) = 0;
    virtual result_t run(const args_t *arg, const igemm_gtc_tunable_t *tunable, void *p_in, void *p_wei, void *p_out, int current_gks) = 0;
    virtual std::vector<int> get_gks_list(const args_t *arg, const igemm_gtc_tunable_t *tunable) = 0;
    virtual igemm_spatial_tiling_t get_spatial_tiling(const args_t *arg) = 0;

    virtual igemm_gtc_tunable_t heuristic_select_kernel(const args_t *arg) {return igemm_gtc_tunable_t{}; }
    virtual int heuristic_select_gks(const args_t *arg, const igemm_gtc_tunable_t *tunable) {return 0; }

    hipModule_t         module_tensor_cast;
    hipModule_t         module;         // not used in IGEMM_SPLIT_KERNEL case
    driver_mode_t       driver_mode;
    driverDataType_t    data_type;
    int                 warmup;
    int                 repeat;
    bool                verbose;

    int                 num_cu;
    int                 gcn_arch;

    int                 max_mpb;
    int                 max_npb;
    int                 max_kpb;
    int                 max_gks;
    int                 vector_c;
};

static inline config_content_t
igemm_try_expand_tunable_content(const config_content_t & content)
{
    config_content_t expanded_content;
    std::vector<std::string> tunable_field_support_expand = {"tensor_layout"};
    for(const auto & sec : content){
        for(const auto f : tunable_field_support_expand){
            if(sec.get_name().compare(0, 6, "igemm_") == 0 &&
                    sec.at(f).get_type() == config_section_value_type_enum::config_section_value_type_list_string){
                for(auto & field_item : sec.at(f).get_list_string()){
                    auto new_sec = sec;
                    std::string field_item_str = std::string("\'") + field_item + std::string("\'");
                    new_sec.at(f) = config_section_value_t::parse_value(field_item_str);
                    expanded_content.add_section(new_sec);
                }
            }
            else{
                expanded_content.add_section(sec);
            }
        }
    }
    return expanded_content;
}

#endif