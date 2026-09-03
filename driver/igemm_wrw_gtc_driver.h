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

#ifndef __IGEMM_WRW_GTC_DRIVER_H
#define __IGEMM_WRW_GTC_DRIVER_H

#include "igemm_gtc_base.h"
#include "config_parser.h"
#include "utility.h"
#include <string>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <numeric>

#include "shisa_dumps.h"

#define WRW_MAX_GEMM_K_SPLITS 10

typedef struct {
    void *p_in;
    void *p_wei;
    void *p_out;
    int hi;
    int wi;
    int n;
    int k;                      // this is indeed k_per_group
    int c;                      // this is indeed c_per_group
    int ho;
    int wo;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int pad_h;
    int pad_w;
    int y;
    int x;
    int gemm_k_global_split;
    int group;
    int gemm_k_per_wg;
} __attribute__((packed)) igemm_wrw_gtc_karg_t;

// Karg for the gfx1250 WMMA wrw kernel (igemm_wrw_gtc_wmma_nhwc_t). Phase 5c added arbitrary
// stride/pad (for the B/input operand's gather -- A/grad_output needs no stride/pad
// awareness), Phase 5f added multi-tap filters (y,x>=1) + dilation, Phase 7 added group>1
// (the only new field is `group` itself -- see igemm_fwd_gtc_wmma_nhwc_t's Phase 7 docstring
// for the rationale). Layout must match get_kernel_args() in
// python/igemm/igemm_wrw_gtc_wmma_nhwc.py exactly (3 pointers + 16 ints, 88 bytes). NOTE the
// kernel's own field semantics: p_in=grad_output (READ), p_wei=input
// (READ), p_out=grad_weight (WRITE) -- run()'s conventional p_in/p_wei/p_out for wrw are
// p_in=input(READ), p_wei=grad_weight(WRITE), p_out=grad_output(READ), so the mapping is a
// 3-way ROTATION, not a simple swap -- see the WMMA branch in run().
typedef struct {
    void *p_in;
    void *p_wei;
    void *p_out;
    int   gemm_m;
    int   gemm_n;
    int   gemm_k;
    int   ho_wo;
    int   wo;
    int   stride_h;
    int   stride_w;
    int   pad_h;
    int   pad_w;
    int   hi;
    int   wi;
    int   y;
    int   x;
    int   dilation_h;
    int   dilation_w;
    int   group;
    int   gemm_k_per_wg;    // gemm_k_global_split: this workgroup's K-slice length. Always
                             // present (even for non-split kernels, which never read it) --
                             // see python/igemm/igemm_wrw_gtc_wmma_nhwc.py's karg comment.
    int   gemm_k_tail;         // Phase 35: wmma_k_tail remainder R = gemm_k - num_k_blocks*
                                // gemm_k_per_block. Always present in this C++ struct (same
                                // "always present" convention as gemm_k_per_wg above), but
                                // only declared/read on the device side when wmma_k_tail AND
                                // gemm_k_global_split are both set -- see get_kernel_args().
    int   gemm_k_num_splits;   // Phase 35: the launched grid.z (== splits), so the device can
                                // tell "am I the last shard" via bz == gemm_k_num_splits-1.
                                // Phase 58 (wrw_streamk): reused as "total shard count" for
                                // the persistent loop's in-range test (same meaning, offset
                                // 96 either way -- see get_kernel_args()'s Phase 58 note).
    int   _streamk_pad;        // Originally padding to align p_streamk_counter to 8 bytes.
                                // Now repurposed to carry streamk_persistent_grid_z (the
                                // launched grid.z) for the static shard-index computation:
                                // tile_idx = bz + iter * persistent_grid_z. Host always
                                // carries it regardless of wrw_streamk (same "always
                                // present" convention as every other field here).
    void *p_streamk_counter;   // Phase 58: pointer to a host-zeroed grid_x*grid_y*4-byte
                                // int32 workspace, one atomic-claim counter per output tile.
                                // Only present/read when wrw_streamk is set.
    int   streamk_max_iters;   // Phase 58: this persistent workgroup's bounded loop trip
                                // count = ceil(total_shards / launched_grid_z).
    int   streamk_grid_y;      // Phase 58: grid.y (N-block count), for this tile's counter
                                // index = (bx*grid_y + by).
    // Phase 60 (Magic Division): host-computed magic multipliers for the ho_wo/wo
    // divisors in _emit_b_gather_one_row's K-decomposition. Always present.
    uint32_t magic_ho_wo;    // magic for ho_wo
    uint32_t magic_wo;       // magic for wo
    uint32_t shift_pack;     // packed shifts: ho_wo[7:0], wo[15:8]
} __attribute__((packed)) igemm_wrw_gtc_wmma_nhwc_karg_t;

static void dump_wrw_karg(igemm_wrw_gtc_karg_t * karg){
    std::cout<<"p_in:"         <<karg->p_in<<",";
    std::cout<<"p_wei:"        <<karg->p_wei<<",";
    std::cout<<"p_out:"        <<karg->p_out<<",";
    std::cout<<"hi:"           <<karg->hi<<",";
    std::cout<<"wi:"           <<karg->wi<<",";
    std::cout<<"n:"            <<karg->n<<",";
    std::cout<<"k:"            <<karg->k<<",";
    std::cout<<"c:"            <<karg->c<<",";
    std::cout<<"ho:"           <<karg->ho<<",";
    std::cout<<"wo:"           <<karg->wo<<",";
    std::cout<<"stride_h:"     <<karg->stride_h<<",";
    std::cout<<"stride_w:"     <<karg->stride_w<<",";
    std::cout<<"dilation_h:"   <<karg->dilation_h<<",";
    std::cout<<"dilation_w:"   <<karg->dilation_w<<",";
    std::cout<<"pad_h:"        <<karg->pad_h<<",";
    std::cout<<"pad_w:"        <<karg->pad_w<<",";
    std::cout<<"y:"            <<karg->y<<",";
    std::cout<<"x:"            <<karg->x<<",";
    std::cout<<"gemm_k_global_split:" <<karg->gemm_k_global_split<<",";
    std::cout<<"group:"        <<karg->group;
    std::cout<<"gemm_k_per_wg:"        <<karg->gemm_k_per_wg;
    std::cout<<std::endl;
}

class igemm_wrw_gtc_t : public igemm_driver_base_t {
public:
    igemm_wrw_gtc_t(hipModule_t module_tensor_cast_, hipModule_t module_, driver_mode_t driver_mode_, driverDataType_t data_type_, int warmup_, int repeat_, bool verbose_)
        : igemm_driver_base_t(module_tensor_cast_, module_, driver_mode_, data_type_, warmup_, repeat_, verbose_) {}
    ~igemm_wrw_gtc_t(){}

    size_t get_block_size(const igemm_gtc_tunable_t *tunable) override {
        if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_MAC || tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_DLOPS){
            return tunable->gemm_m_level0_cluster * tunable->gemm_n_level0_cluster *
               tunable->gemm_m_level1_cluster * tunable->gemm_n_level1_cluster;
        }
        else if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_XDLOPS){
            int waves_per_m = tunable->gemm_m_per_block / (tunable->wave_tile_m * tunable->wave_step_m * tunable->wave_repeat_m);
            int waves_per_n = tunable->gemm_n_per_block / (tunable->wave_tile_n * tunable->wave_step_n * tunable->wave_repeat_n);
            return waves_per_m * waves_per_n * AMDGPU_WAVE_SIZE;
        }
        else if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA){
            int waves_per_m = tunable->gemm_m_per_block / (tunable->wave_tile_m * tunable->wave_repeat_m);
            int waves_per_n = tunable->gemm_n_per_block / (tunable->wave_tile_n * tunable->wave_repeat_n);
            return waves_per_m * waves_per_n * 32;
        }
        else{
            std::cout << "not valid fma_type: " << tunable->fma_type << std::endl;
            assert(false);
            return 0;
        }
    }
    size_t get_grid_size(const args_t *arg,
                      const igemm_gtc_tunable_t *tunable) override {
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

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int gemm_k_per_block         = tunable->gemm_k_per_block;
        int gemm_k_global_split      = tunable->gemm_k_global_split;
		
        int block_size               = get_block_size(tunable);
        int c_vec_min                = tunable->tensor_layout == "nchw" ? 1 : (tunable->tensor_b_thread_lengths[3]);
        int max_grid_size            = 1200;

        int gemm_m = k / group;
        int c_padded = ((c / group) + c_vec_min - 1) / c_vec_min * c_vec_min;
        int gemm_n = (c_padded * y * x  + gemm_n_per_block - 1) / gemm_n_per_block * gemm_n_per_block;
        size_t grid_size = static_cast<size_t>(group) * utility_integer_divide_ceil(gemm_m, gemm_m_per_block) *
                                    utility_integer_divide_ceil(gemm_n, gemm_n_per_block);

        int splits = igemm_split_batch_size(arg, utility_string_to_data_byte(tunable->precision));
        if(splits == 0){
            printf("image size (c*h*w or k*h*w) is bigger than 4g, which is not supported now\n");
            return false;
        }
        n = n/splits;   // split batch size here

        int min_n_per_block = 1;
        if(tunable->tensor_layout == "nhwc" && tunable->nxe == 1)
            min_n_per_block = tunable->tensor_a_thread_lengths[1];

        int b = ho * wo;
        if(tunable->tensor_layout == "nchw")
            b = tunable->nxe == 0 ? (ho * wo) : ((ho * wo + tunable->nxb - 1) / tunable->nxb) * tunable->nxb;

        if(tunable->tensor_layout == "nchw"){
            int gemm_k_global_splits = gemm_k_global_split == 1 ? compute_log2_gemmk_global_splits(grid_size, max_grid_size, n / min_n_per_block, b, gemm_k_per_block)
                                                                 : 0;

            int num_of_gemm = 1 << gemm_k_global_splits;
            grid_size *= num_of_gemm;
        }

        assert(grid_size <= 0xffffffffUL);
        return grid_size;
    }

    int get_lds_size(const igemm_gtc_tunable_t *tunable) {
        // TODO: fp16/bf16, xdlops
        int lds_a = utility_string_to_data_byte(tunable->precision) * tunable->gemm_k_per_block * tunable->gemm_m_per_block;
        int lds_b = utility_string_to_data_byte(tunable->precision) * tunable->gemm_k_per_block * tunable->gemm_n_per_block;
        return 2 * utility_next_pow2(utility_next_pow2(lds_a) + utility_next_pow2(lds_b));
    }

    bool tunable_is_valid(const args_t *arg,
                          const igemm_gtc_tunable_t *tunable) override
    {
        // TODO:
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

        int need_wrw = (forw == 0 ? 1 : (forw & 4 ? 1 : 0));
        if(need_wrw == 0)
            return false;

        if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA){
            // igemm_wrw_gtc_wmma_nhwc_t requires a single fixed 128x128 macro-tile shape --
            // see igemm_fwd_gtc_driver.h's identical WMMA branch for the rationale. Phase 5c
            // added arbitrary stride/pad, Phase 5f added arbitrary y/x (multi-tap filters) and
            // dilation, Phase 7 added group>1. Checked here, BEFORE the fil_h_ext/pad sanity
            // check below: that check assumes the general kernel's addressing (padding >= a
            // 1x1 filter's extent-of-1 is nonsensical for it) and would incorrectly reject
            // every padded WMMA config, which this kernel's gather+mask mechanism handles
            // correctly regardless of filter size.
            if(tunable->tensor_layout != "nhwc")
                return false;
            if(group < 1 || c % group != 0 || k % group != 0)
                return false;
            int wmma_gemm_m = k / group;
            int wmma_gemm_n = c / group;
            int wmma_gemm_k = n * ho * wo;
            // Phase 35: wmma_m_tail/wmma_n_tail/wmma_k_tail each independently relax their own
            // axis's exact-multiple requirement -- mirrors fwd's identical relax pattern in
            // igemm_fwd_gtc_driver.h. Unlike fwd/bwd, wrw's M/N-tail must also work under
            // gemm_k_global_split (checked/asserted at the tunable-construction level in
            // igemm_base.py, not here), so no additional restriction is needed here for that.
            // Phase 51: gemm_n%4==0 previously mirrored fwd's identical (now-fixed)
            // requirement for the NON-atomic epilogue -- its store is vector_write_out=4-
            // wide, and the EXEC-mask guard only checked a group's first column, so a group
            // straddling a non-multiple-of-4 tail silently wrote past the real gemm_n. Fixed
            // with genuine per-element masking (coalescing_store_wmma.py's Phase 51 comment)
            // -- lifted for the same reason fwd's was. The atomic epilogue
            // (gemm_k_global_split) issues one scalar atomic per element (no 4-wide
            // grouping) and was never affected by this in the first place.
            // Phase 45: tdm_global_load relaxes the gemm_k (wmma_gemm_k = n*ho*wo) exact-
            // multiple requirement the same way fwd's Phase 39/bwd's Phase 42 do -- TDM's
            // hardware OOB zero-fills the tail row on both A and B. Also mirrors those same
            // phases' runtime-shape check: tdm_global_load's "1x1/unit-stride only"
            // restriction was previously enforced only at CONFIG level (igemm_base.py
            // asserts nxe==0 for the tunable itself), never against the actual
            // runtime-requested shape.
            //
            // That relaxation is NOT extended to gemm_k_global_split: this driver's
            // karg.gemm_k_per_wg (this workgroup's own K-slice length) is computed as
            // (num_k_blocks / splits) * gemm_k_per_block -- i.e. gemm_k rounded UP to the
            // next multiple of gemm_k_per_block before being divided among shards. wrw's
            // wmma_k_tail mechanism (a separate, explicit kernarg + on-device flag) is what
            // normally clamps the LAST shard back down to the true gemm_k under split-K --
            // TDM asserts mutual exclusivity with wmma_k_tail (see igemm_wrw_gtc_wmma_nhwc.py's
            // __init__), so it has no such clamp. Confirmed on real hardware: a non-exact
            // gemm_k with tdm_global_load + gemm_k_global_split silently reads past the true
            // gemm_k (TDM's own tensor_dim1 gets set to the ROUNDED-UP per-shard length, not
            // the true remainder) -- valid:n. Exact-multiple gemm_k under split-K has no such
            // rounding (num_k_blocks divides evenly), so it stays valid:y and is left enabled.
            bool unit_conv = (x==1)&&(y==1)&&(stride_h==1)&&(stride_w==1)&&(dilation_h==1)&&(dilation_w==1)&&(pad_h==0)&&(pad_w==0);
            if(tunable->tdm_global_load && !unit_conv)
                return false;
            if(tunable->tdm_global_load && tunable->gemm_k_global_split && wmma_gemm_k % tunable->gemm_k_per_block != 0)
                return false;
            if((!tunable->wmma_m_tail && wmma_gemm_m % tunable->gemm_m_per_block != 0) ||
               (!tunable->wmma_n_tail && wmma_gemm_n % tunable->gemm_n_per_block != 0) ||
               (!tunable->tdm_global_load && !tunable->wmma_k_tail && wmma_gemm_k % tunable->gemm_k_per_block != 0))
                return false;
            return true;
        }

        int fil_h_ext = y * dilation_h + 1 - y;
        int fil_w_ext = x * dilation_w + 1 - x;
        if (pad_w >= fil_w_ext || pad_h >= fil_h_ext)
            return false;


        int nxb = tunable->nxb == 0 ? 1 : tunable->nxb;
        int b  = tunable->nxe == 0 ? (ho * wo) : ((ho * wo + nxb - 1) / nxb) * nxb;   // pad to nxb modulo when nxe != 0
        int data_byte = utility_string_to_data_byte(tunable->precision);
        assert(c % group == 0 && k % group == 0);

        int splits = igemm_split_batch_size(arg, utility_string_to_data_byte(tunable->precision));
        if(splits == 0){
            printf("image size (c*h*w or k*h*w) is bigger than 4g, which is not supported now\n");
            return false;
        }
        n = n/splits;   // split batch size here

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int gemm_k_per_block         = tunable->gemm_k_per_block;

        int gemm_k_global_split      = tunable->gemm_k_global_split;
        int gemmk_blocks             = 1 << gemm_k_global_split;

        int n_per_block = n >> gemm_k_global_split;

        int gemm_n = (c / group) * y * x;
        int gemm_k = n * b;

        int nxe = tunable->nxe == 0 ? 1 : tunable->nxe;
        bool unit_conv = (x==1)&&(y==1)&&(stride_h==1)&&(stride_w==1)&&(dilation_h==1)&&(dilation_w==1)&&(pad_h==0)&&(pad_w==0);


        if(splits > 1 && gemm_k_global_split == 0)
        {
            // large tensor can only used for gkgs kernel
            return false;
        }

        if(tunable->tensor_layout == "nchw"){
            if (n % gemmk_blocks != 0){
                return false;
            }
            if(((c / group) % (gemm_n_per_block / nxe) != 0) || (((x * y) % nxe) != 0))
            {
                return false;
            }
            if (gemm_k % gemm_k_per_block != 0){
                //std::cout << __func__ << " false: gemm_n is " << gemm_n << ", gemm_n_per_block is " << gemm_n_per_block << ", gemm_m is " << gemm_m << ", gemm_m_per_block is " << gemm_m_per_block << std::endl;
                return false;
            }

            if (gemm_k_per_block % nxb != 0){
                //std::cout << __func__ << " false: gemm_n_per_block is " << gemm_n_per_block << ", nxb is " << nxb << std::endl;
                return false;
            }

            int n_n0 = tunable->tensor_a_cluster_lengths[0] * tunable->tensor_a_thread_lengths[0];
        
            if (n_n0 > 1){
                if (n_per_block % (tunable->tensor_a_thread_lengths[1] * tunable->tensor_a_cluster_lengths[1] * n_n0) != 0){
                    return false;
                }
            }
            else {
                if (n_per_block * b % gemm_k_per_block !=0){
                    return false;
                }
            }

            // input vector load limitation, n1b
            if(tunable->tensor_b_thread_lengths[1] > 1 && (
                !unit_conv ||
                unit_conv && (hi * wi) % tunable->tensor_b_thread_lengths[1] != 0)) {
                return false;
            }

            // output vector load limitation, n1b
            if(tunable->tensor_a_thread_lengths[1] > 1 && (
                !unit_conv ||
                unit_conv && (ho * wo) % tunable->tensor_a_thread_lengths[1] != 0)) {
                return false;
            }
            if (b % nxb != 0){
                //std::cout << __func__ << " false: (ho * wo) is " << (ho * wo) << ", nxb is " << nxb << std::endl;
                return false;
            }
        }
        else{
            if(data_byte == 2){
                if((c / group) % tunable->tensor_b_thread_lengths[3] != 0){
                    return false;
                }
            }
        }

        if (!unit_conv && (tunable->nxe == 0))
            return false;

        return true;
    }

    // calculate log2_gemm_k_global_splits
    static inline int compute_gemmk_global_splits(const int& grid_size,
                                                  const int& potential_occupancy)
    {
        int num_cu;
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice(&dev));
        HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
        num_cu = dev_prop.multiProcessorCount;
        int gemm_k_global_splits = num_cu * potential_occupancy / grid_size;
        
        return gemm_k_global_splits;
    }

    // calculate log2_gemm_k_global_splits
    static inline int compute_log2_gemmk_global_splits(const int& grid_size,
                                                       const int& max_grid_size,
                                                       const int& n,
                                                       const int& b,
                                                       const int& gemm_k_per_block)
    {
        int log2_gemm_k_global_splits = 0;
        for(int gs = 0; gs < 9; gs++)
        {
            if((grid_size << gs) > max_grid_size)
                break;

            if((n % (1 << gs)) != 0)
                break;

            //if((n >> gs) * b % gemm_k_per_block != 0)
            //    break;
            log2_gemm_k_global_splits = gs;
        }
        return log2_gemm_k_global_splits;
    }

    static int if_gemm_k_global_split(const args_t *arg,
                                  const int gemm_m_per_block,
                                  const int gemm_n_per_block,
                                  const int gemm_k_per_block,
                                  const int data_byte,
                                  const std::string tensor_layout)
    {
        int gemm_k_global_split = 0;
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
        
        assert(c % group == 0 && k % group == 0);

        int splits = igemm_split_batch_size(arg, data_byte);
        assert(splits != 0);
        n = n/splits;   // split batch size here

        int gemm_m = k / group;
        int block_size = 256;
        int c_vec_min = tensor_layout == "nchw" ? 1 : (gemm_n_per_block * gemm_k_per_block / block_size);
        int c_padded = ((c / group) + c_vec_min - 1) / c_vec_min * c_vec_min;
        int gemm_n = (c_padded * y * x  + gemm_n_per_block - 1) / gemm_n_per_block * gemm_n_per_block;
        int gemm_k = n * ho * wo;

        int grid_size;
        grid_size = group * utility_integer_divide_ceil(gemm_m, gemm_m_per_block) *
                                    utility_integer_divide_ceil(gemm_n, gemm_n_per_block);
        if ((n % 2 == 0) && (grid_size < 512) && ((n >> 1) * ho * wo % gemm_k_per_block == 0)){
            gemm_k_global_split = 1;
        }
        else {
            gemm_k_global_split = 0;
        }
        return gemm_k_global_split;
    }

    static inline int find_tunable(const std::vector<igemm_gtc_tunable_t> tunables, 
                                    const int gemm_m_per_block,
                                    const int gemm_n_per_block,
                                    const int gemm_k_per_block,
                                    const int gemm_k_global_split,
                                    const int nxb,
                                    const int nxe)
    {
        int i;
        for (i = 0; i < tunables.size(); i++) {
            if ((tunables[i].gemm_m_per_block == gemm_m_per_block) &&
                (tunables[i].gemm_n_per_block == gemm_n_per_block) &&
                (tunables[i].gemm_k_per_block == gemm_k_per_block) &&
                (tunables[i].gemm_k_global_split == gemm_k_global_split) &&
                (tunables[i].nxb == nxb) &&
                (tunables[i].nxe == nxe)){
                break;
            }
        }
        return i;
    }

    std::string select_kernel(const args_t *arg, const std::vector<igemm_gtc_tunable_t> tunables)
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
        int group = arg->get_int("group_count");
        int data_byte = utility_string_to_data_byte(tunables[0].precision);
        std::string data_layout = tunables[0].tensor_layout;
        if(data_layout == "nhwc")
            return std::string("NONE");
        assert(c % group == 0 && k % group == 0);

        int gemm_m_per_block = 0;
        int gemm_n_per_block = 0;
        int gemm_k_per_block = 0;

        int gemm_k_global_split = 0;

        int gemm_m = k / group;
        int gemm_n = (c / group) * y * x;

        int grid_size;
        int block_size;
        int max_grid_size                 = 1200;
        int sel_index                     = -1;
        int sel_block_size                = 0;
        int sel_grid_size                 = 0;
        int sel_log2_gemm_k_global_splits = 0;
        int num_cu                        = 120;
        std::vector<int> nxb_list         = {16, 8, 4, 1};
        std::vector<int> nxe_list         = {0, 1};

        // i=log2(gemm_m_per_block*gemm_n_per_block)  to find largest kernel
        // when pack=0, means no need to search with pack image size. when pack=1, we need pack
        for(int pack = 0; pack < 2; pack++)
        {
            for (int i = 15; i > 7; i--){
                int r, l;
                r = (i + 1) >> 1;
                l = i - r;
                while (l > 1 && r < 9){
                    for (int swap = 0; swap < 2; swap++){

                        const auto gemm_m_per_block = swap == 0 ? 1 << r : 1 << l;
                        const auto gemm_n_per_block = swap == 0 ? 1 << l : 1 << r;
                    
                        if (gemm_n % gemm_n_per_block != 0)
                            continue;

                        for (int j = 5; j > 1; j--){
                            gemm_k_per_block = 1 << j;
                            for(const auto& nxe : nxe_list)
                            {
                                for(const auto& nxb : nxb_list)
                                {
                                    const auto b = pack == 0
                                        ? ho * wo
                                        : (nxe == 0 ? ho * wo : ((ho * wo + nxb - 1) / nxb) * nxb);
                                    const auto gemm_k = n * b;
                                    if(c % (gemm_n_per_block / (nxe == 0 ? 1 : nxe)) != 0)
                                        continue;
                                    if(gemm_k % gemm_k_per_block != 0)
                                        continue;

                                    if(nxe == 0)
                                    {
                                        if((x != 1) || (y != 1) || (dilation_h != 1) ||
                                            (dilation_w != 1) || (pad_h != 0) || (pad_w != 0))
                                            continue;
                                        if(stride_h != 1 || stride_w != 1)
                                        {
                                            if(nxb != 1)
                                                continue;
                                        }
                                        else
                                        {
                                            // nxe==0 case, need vector check(in nxe==0 case, nxb means
                                            // vector length)
                                            if(ho * wo % nxb != 0)
                                                continue;
                                        }
                                    }

                                    gemm_k_global_split = if_gemm_k_global_split(arg, 
                                        gemm_m_per_block, 
                                        gemm_n_per_block,
                                        gemm_k_per_block,
                                        data_byte,
                                        data_layout);

                                    int tunable_index = find_tunable(tunables, gemm_m_per_block, gemm_n_per_block, gemm_k_per_block, gemm_k_global_split, nxb, nxe);
                                    if (tunable_index < 0 || tunable_index >= tunables.size())
                                        continue;

                                    int log2_gemm_k_global_splits = 0;
                                    int grid_size = group * utility_integer_divide_ceil(gemm_m, gemm_m_per_block) * utility_integer_divide_ceil(gemm_n, gemm_n_per_block);
                                    int block_size = get_block_size(&tunables[tunable_index]);
                                    log2_gemm_k_global_splits = compute_log2_gemmk_global_splits(grid_size, max_grid_size, n, b, gemm_k_per_block);
                                    if (gemm_k_global_split == 0)
                                        log2_gemm_k_global_splits = 0;

                                    // in nxe==1 cases, wo%tb[1] need to be 0; when tb[1] > 1, need (pad_h+pad_w)==0
                                    if(nxe != 0)
                                    {
                                        if(wo % tunables[tunable_index].tensor_b_thread_lengths[1] != 0)
                                            continue;
                                        if(tunables[tunable_index].tensor_b_thread_lengths[1] > 1 &&
                                            (pad_h != 0 || pad_w != 0))
                                            continue;
                                    }

                                    grid_size = grid_size << log2_gemm_k_global_splits;

                                    if(block_size >= sel_block_size && grid_size > sel_grid_size)
                                    {
                                        sel_block_size                = block_size;
                                        sel_grid_size                 = grid_size;
                                        sel_index                     = tunable_index;
                                        sel_log2_gemm_k_global_splits = log2_gemm_k_global_splits;
                                        break;
                                    }
                                }
                            }
                            if (sel_grid_size > num_cu * 2)
                                break;
                        }
                        if (sel_grid_size > num_cu * 2)
                            break;
                    }
                    if (sel_grid_size > num_cu * 2)
                        break;
                    r++;
                    l--;
                }
                if (sel_grid_size > num_cu)
                    break;
            }
            //std::cout << "sel_index:" << sel_index << std::endl;
            if (sel_index < 0 || sel_index >= tunables.size())
            {
                return std::string("NONE");
            }
            else
            {
                const igemm_gtc_tunable_t *tunable_return = &tunables[sel_index];
                // std::cout << get_kernel_name(tunable_return) <<std::endl;
                return get_kernel_name(tunable_return);
            }
        }
        assert(false);
    }

    // get grid size without gks
    size_t get_cur_grid_size(const args_t *arg, const igemm_gtc_tunable_t *tunable){
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

        int gemm_m_per_block         = tunable->gemm_m_per_block;
        int gemm_n_per_block         = tunable->gemm_n_per_block;
        int gemm_k_per_block         = tunable->gemm_k_per_block;

        size_t block_size            = get_block_size(tunable);
        int c_vec_min                = tunable->tensor_layout == "nchw" ? 1 : (tunable->tensor_b_thread_lengths[3]);

        int gemm_m = k / group ;
        int c_padded = ((c / group) + c_vec_min - 1) / c_vec_min * c_vec_min;
        int gemm_n = (c_padded * y * x  + gemm_n_per_block - 1) / gemm_n_per_block * gemm_n_per_block;
        size_t grid_size = static_cast<size_t>(group) * utility_integer_divide_ceil(gemm_m, gemm_m_per_block) *
                                    utility_integer_divide_ceil(gemm_n, gemm_n_per_block);

        return grid_size;
    }

    result_t run(const args_t *arg, const igemm_gtc_tunable_t *tunable,
                 void *p_in, void *p_wei, void *p_out, int current_gks) override {
        if (!tunable_is_valid(arg, tunable)) {
            result_t result;
            result.return_code = -1;
            // std::cout << "not valid tunable config." << std::endl;
            return result;
        }

        if(tunable->fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA){
            // Self-contained launch path, mirroring igemm_fwd_gtc_driver.h's WMMA branch.
            // p_in/p_wei/p_out must be ROTATED relative to their names -- see the
            // igemm_wrw_gtc_wmma_nhwc_karg_t struct comment above.
            int hi = arg->get_int("in_h");
            int wi = arg->get_int("in_w");
            int n  = arg->get_int("batchsize");
            int k  = arg->get_int("out_channels");
            int c  = arg->get_int("in_channels");
            int group = arg->get_int("group_count");
            int y = arg->get_int("fil_h");
            int x = arg->get_int("fil_w");
            int stride_h = arg->get_int("conv_stride_h");
            int stride_w = arg->get_int("conv_stride_w");
            int dilation_h = arg->get_int("dilation_h");
            int dilation_w = arg->get_int("dilation_w");
            int pad_h = arg->get_int("pad_h");
            int pad_w = arg->get_int("pad_w");
            int ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
            int wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);

            int gemm_m = k / group;
            int gemm_n = c / group;
            int gemm_k = n * ho * wo;

            size_t grid_x = utility_integer_divide_ceil(gemm_m, tunable->gemm_m_per_block);
            // group is folded into grid_y, decoded on-device -- see igemm_fwd_gtc_driver.h's
            // identical WMMA branch for the rationale.
            size_t grid_y = static_cast<size_t>(group) * utility_integer_divide_ceil(gemm_n, tunable->gemm_n_per_block);
            // W-1 (perf report 2026-09-02): wrw's GEMM is M=K/group, N=C/group, GEMM_K=N*Ho*Wo
            // -- a small-MN, enormous-K GEMM. Without gemm_k_global_split the grid is just
            // grid_x*grid_y, which can be far smaller than the CU count (e.g. 16 WGs on 256
            // CUs = 6% utilization for a 512x512 64x64-tile shape). The default wrw configs
            // now set gemm_k_global_split=1; this warning catches any non-gsplit tunable
            // that would starve the GPU, so a stale config or a hand-edited debug tunable
            // doesn't silently regress to single-digit TFLOP/s. Uses dev_prop directly
            // (not this->num_cu, which is doubled for gfx10+ and would fire spuriously).
            if (!tunable->gemm_k_global_split) {
                hipDeviceProp_t _dev_prop;
                hipDevice_t _dev;
                HIP_CALL(hipGetDevice(&_dev));
                HIP_CALL(hipGetDeviceProperties(&_dev_prop, _dev));
                size_t base_grid = grid_x * grid_y;
                if (base_grid < static_cast<size_t>(_dev_prop.multiProcessorCount)) {
                    printf("[wrw] WARNING: gemm_k_global_split=0 with grid=%zu (< %d CUs); "
                           "grid starvation will cause severe underutilization. "
                           "Set gemm_k_global_split=1 in the config.\n",
                           base_grid, _dev_prop.multiProcessorCount);
                }
            }

            // gemm_k_global_split: split the reduction axis across grid.z, atomically
            // accumulating partial sums (see igemm_wrw_gtc_wmma_nhwc.py / docs/
            // gfx1250_wmma_layout.md). A 3-candidate bracket around a heuristic target was
            // tried first and left real perf on the table (see docs/gfx1250_wmma_layout.md's
            // Phase 20): timing the FULL sweep of divisors of num_k_blocks (every split gets
            // an identical, exact multiple of gemm_k_per_block -- the WMMA main loop has no
            // K-tail handling, so only divisors are valid candidates at all) showed cost vs.
            // split count is unimodal -- decreasing monotonically to a single minimum, then
            // increasing -- but the minimum's LOCATION relative to the naive
            // ceil(num_cu/tile_count) target isn't consistent (sometimes well above it,
            // sometimes well below), so a fixed-offset bracket around that target
            // structurally can't reliably find it. A ternary search over the sorted divisor
            // list can, in O(log(divisor count)) real timed launches, exploiting that same
            // unimodality.
            auto largest_divisor_leq = [](int num_k_blocks, int target) -> int {
                for (int s = std::min(std::max(target, 1), num_k_blocks); s >= 1; s--) {
                    if (num_k_blocks % s == 0) return s;
                }
                return 1;
            };
            int num_k_blocks = gemm_k / tunable->gemm_k_per_block;
            // Phase 35: wmma_k_tail's remainder -- the portion of gemm_k not covered by any
            // exact gemm_k_per_block-sized block. Only the LAST split-K shard's s_knum gets
            // extended by this (see igemm_wrw_gtc_wmma_nhwc.py's emit_kernel_prologue); 0 when
            // wmma_k_tail is unset (gemm_k is then required to be an exact multiple already,
            // per tunable_is_valid()).
            int gemm_k_tail = tunable->wmma_k_tail ? (gemm_k - num_k_blocks * tunable->gemm_k_per_block) : 0;
            // Phase 50: sane upper bound on how far the search below is allowed to split --
            // see igemm_gemm_k_global_split_cap's own comment (igemm_gtc_base.h). Without
            // this, an extreme small-gemm_m/n (tiny output), huge-gemm_k (huge n*ho*wo)
            // shape can produce num_k_blocks in the millions, and the ternary search below
            // would REAL-LAUNCH-time a pathological divisor near it (a 135000-way split was
            // observed taking minutes for a single candidate, dominated by atomic-add
            // contention across that many workgroup-z shards all targeting a handful of
            // output elements) -- found via a diverse gfx950-baseline benchmark sweep.
            int split_cap = tunable->gemm_k_global_split
                ? igemm_gemm_k_global_split_cap(gemm_k, tunable->gemm_k_per_block) : 1;
            std::vector<int> divisors;
            if (tunable->gemm_k_global_split) {
                for (int i = 1; static_cast<long long>(i) * i <= num_k_blocks; i++) {
                    if (num_k_blocks % i == 0) {
                        if (i <= split_cap) divisors.push_back(i);
                        int j = num_k_blocks / i;
                        if (j != i && j <= split_cap) divisors.push_back(j);
                    }
                }
                std::sort(divisors.begin(), divisors.end());
                if (divisors.empty()) divisors.push_back(1);
            } else {
                divisors.push_back(1);
            }

            igemm_wrw_gtc_wmma_nhwc_karg_t karg;
            karg.p_in     = p_out;   // grad_output (read)
            karg.p_wei    = p_in;    // input (read)
            // Phase 35 (hipconv-style reduction-kernel epilogue): when set, the main kernel
            // writes disjoint per-shard partial sums into a workspace buffer instead of
            // atomic-accumulating directly into the real output -- a separate reduction
            // kernel (wrw_reduce_partials_f32) sums them afterward. Workspace is sized for
            // min(num_k_blocks, split_cap) partitions (Phase 50: the largest split count the
            // ternary search below could now ever try, since its divisor list is capped --
            // previously sized for the uncapped num_k_blocks, which could be a multi-GB
            // allocation for an extreme small-gemm_m/n, huge-gemm_k shape), fp32-native
            // regardless of the tunable's nominal precision (same rationale as the atomic
            // path's always-fp32 output, see docs/gfx1250_wmma_layout.md's Phase 34/17), so
            // every candidate fits without reallocating mid-search.
            void *p_wei_workspace_wsred = nullptr;
            size_t wsred_output_size = static_cast<size_t>(group) * (k / group) * (c / group) * y * x;
            if (tunable->wrw_reduction_kernel) {
                size_t wsred_partitions = static_cast<size_t>(std::min(num_k_blocks, split_cap));
                HIP_CALL(hipMalloc(&p_wei_workspace_wsred, wsred_partitions * wsred_output_size * sizeof(float)));
                karg.p_out = p_wei_workspace_wsred;
            } else {
                karg.p_out = p_wei;   // grad_weight (write)
            }
            karg.gemm_m   = gemm_m;
            karg.gemm_n   = gemm_n;
            karg.gemm_k   = gemm_k;
            karg.ho_wo    = ho * wo;
            karg.wo       = wo;
            karg.stride_h = stride_h;
            karg.stride_w = stride_w;
            karg.pad_h    = pad_h;
            karg.pad_w    = pad_w;
            karg.hi       = hi;
            karg.wi       = wi;
            karg.y          = y;
            karg.x          = x;
            karg.dilation_h = dilation_h;
            karg.dilation_w = dilation_w;
            karg.group      = group;

            // Phase 60 (Magic Division): precompute magic multipliers for the ho_wo/wo
            // divisors in _emit_b_gather_one_row's K-decomposition, replacing ~24-instruction
            // emulated division with 5-instruction magic multiply.
            {
                magic_div_u32_t mdiv_ho_wo = magic_div_u32_gen(ho * wo);
                magic_div_u32_t mdiv_wo    = magic_div_u32_gen(wo);
                karg.magic_ho_wo = mdiv_ho_wo.magic;
                karg.magic_wo    = mdiv_wo.magic;
                karg.shift_pack  = magic_div_u32_pack_shift(mdiv_ho_wo.shift, mdiv_wo.shift, 0, 0);
            }
            size_t karg_size = sizeof(karg);

            hipFunction_t kernel_func;
            std::string kernel_name = get_kernel_name(tunable);
#ifdef IGEMM_SPLIT_KERNEL
            hipModule_t cur_kernel_module;
            std::string cur_kernel_hsaco = kernel_name + ".hsaco";
            HIP_CALL(hipModuleLoad(&cur_kernel_module, cur_kernel_hsaco.c_str()));
            HIP_CALL(hipModuleGetFunction(&kernel_func, cur_kernel_module, kernel_name.c_str()));
#else
            HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));
#endif

            size_t block_size = get_block_size(tunable);

            // Phase 33: cross-check the ternary search against MISA's existing (originally
            // XDLOPS-only) closed-form occupancy heuristic, compute_gemmk_global_splits --
            // fed a REAL hipModuleOccupancyMaxActiveBlocksPerMultiprocessor value instead of
            // that path's hardcoded potential_occupancy=3. This is exactly CK's own split-K
            // formula shape (num_cu * max_occupancy_per_CU / grid_size), see
            // docs/gfx1250_perf_parity_action_plan.md's Tier 1 item 3. Added as one MORE
            // candidate into the divisor list the ternary search below already evaluates --
            // strictly non-regressive: worst case is one extra real timed launch; if the
            // heuristic's candidate turns out worse, the search's own min-of-all-evaluated
            // logic simply never selects it.
            if (tunable->gemm_k_global_split) {
                int potential_occupancy = 1;
                HIP_CALL(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&potential_occupancy, kernel_func, static_cast<int>(block_size), 0));
                int heuristic_splits = std::min(compute_gemmk_global_splits(static_cast<int>(grid_x * grid_y), potential_occupancy), split_cap);
                int heuristic_candidate = largest_divisor_leq(num_k_blocks, heuristic_splits);
                if (std::find(divisors.begin(), divisors.end(), heuristic_candidate) == divisors.end()) {
                    divisors.push_back(heuristic_candidate);
                    std::sort(divisors.begin(), divisors.end());
                }
            }

            result_t result;
            result.kernel_name = kernel_name;
            memset(&result.dumpheader, 0, sizeof(result.dumpheader));
            result.dumpheader.conv.hi = hi;
            result.dumpheader.conv.wi = wi;
            result.dumpheader.conv.n  = n;
            result.dumpheader.conv.k  = k;
            result.dumpheader.conv.c  = c;
            result.dumpheader.conv.group = group;
            result.dumpheader.conv.dir = convdir_t::WRW;
            result.dumpheader.conv.dtype = dtype(tunable->precision);

            auto noop = std::function<float()>{[&]() -> float { return .0; }};
            // Atomics accumulate rather than overwrite -- the output must be re-zeroed before
            // EVERY dispatch (warmup and each timed repeat), not just once before the whole
            // benchmark loop, or the 2nd+ iteration's atomic-adds land on top of the 1st
            // iteration's already-written result. WMMA output is always allocated/written as
            // fp32 regardless of the tunable's nominal precision (conv_driver.cpp's is_wmma
            // dtype_alloc_byte override, and this kernel's D-operand-always-4-bytes epilogue),
            // so zero by element count * sizeof(float), not data_byte -- EXCEPT Phase 34's
            // atomic_pack_bf16, which writes the output at its native bf16 (2-byte) width
            // instead (conv_driver.cpp's dtype_alloc_byte override handles this the same way
            // it already does for wmma_acc_f16/wmma_acc_bf16) -- zeroing at the fp32 element
            // count there would write past the actual (half-sized) allocation.
            size_t gsplit_zero_elem_byte = tunable->atomic_pack_bf16 ? 2 : sizeof(float);
            // Phase 58 (wrw_streamk): one atomic-claim counter per (bx,by) output tile.
            // Zeroed every dispatch below (same "must re-zero every dispatch" discipline as
            // the atomic epilogue's own output buffer -- otherwise the 2nd+ warmup/repeat
            // iteration would start claiming from a stale nonzero counter and every shard
            // would read as already-exhausted). See docs/gfx1250_streamk_design.md.
            void *p_streamk_counter = nullptr;
            if (tunable->wrw_streamk) {
                HIP_CALL(hipMalloc(&p_streamk_counter, grid_x * grid_y * sizeof(int)));
            }
            auto wrw_gsplit_prolog = std::function<float()>{[&]() -> float {
                // Phase 35: wrw_reduction_kernel's main kernel does plain (non-atomic)
                // per-shard stores, not accumulation -- every partition slot gets fully
                // overwritten every dispatch, so no re-zero is needed (and zeroing at
                // wsred_output_size here would be the wrong size anyway -- the workspace is
                // num_k_blocks*wsred_output_size).
                if (tunable->gemm_k_global_split && !tunable->wrw_reduction_kernel)
                    HIP_CALL(hipMemset(p_wei, 0, static_cast<size_t>(group) * (k / group) * (c / group) * y * x * gsplit_zero_elem_byte));
                if (tunable->wrw_streamk)
                    HIP_CALL(hipMemset(p_streamk_counter, 0, grid_x * grid_y * sizeof(int)));
                return .0;
            }};

            // Phase 35: reduction kernel function, loaded once (reused across every
            // candidate split count the search below tries).
            hipFunction_t wrw_reduce_func;
            if (tunable->wrw_reduction_kernel) {
                HIP_CALL(hipModuleGetFunction(&wrw_reduce_func, module_tensor_cast, "wrw_reduce_partials_f32"));
            }

            std::string dump_dir = env_get_str("IGEMM_DUMPDIR_ALL", "");
            // Actually launches and times `splits` for real (karg.gemm_k_per_wg + grid.z),
            // via the same per-iteration zero-init prolog every other launch path uses.
            auto time_split = [&](int splits) -> float {
                karg.gemm_k_per_wg = (num_k_blocks / splits) * tunable->gemm_k_per_block;
                karg.gemm_k_tail = gemm_k_tail;
                karg.gemm_k_num_splits = splits;
                size_t grid_z = static_cast<size_t>(splits);

                result.dumpdata.clear();
                std::vector<igemm_launch_kernel_t> kernel_launchers;
                kernel_launchers.push_back({kernel_func, &karg, karg_size, {grid_x * block_size, grid_y, grid_z}, {block_size, 1, 1}});
                result.dumpheader.n_dispatches = kernel_launchers.size();
                result.dumpheader.gks = splits;
                result.dumpdata.push_back(kernel_launchers.back());
                result.dumpdata.back().ktype = kargtype_t::igemm_wrw_gtc_wmma_nhwc_karg_t;

                // Phase 35: reduction pass, timed together with the main kernel (same
                // pattern as the XDLOPS path's wrw_postlog/tensor_cast_func) -- this
                // `splits` is the number of partitions this specific candidate actually
                // wrote (workspace slots beyond it, up to num_k_blocks, are simply unused
                // for this launch).
                auto wrw_reduce_postlog = std::function<float()>{[&, splits]() -> float {
                    if (tunable->wrw_reduction_kernel) {
                        wrw_reduce_karg_t karg_reduce;
                        karg_reduce.output = p_wei;
                        karg_reduce.workspace = p_wei_workspace_wsred;
                        karg_reduce.num_partitions = splits;
                        karg_reduce.output_size = static_cast<int>(wsred_output_size);
                        size_t karg_reduce_size = sizeof(karg_reduce);
                        // igemm_launch_kernel_single's grid_size is in units of WORKITEMS
                        // (total threads), not blocks, and hipExtModuleLaunchKernel requires
                        // it to be an EXACT multiple of block_size -- round output_size up to
                        // the next multiple of 256, matching the existing tensor_cast kernel
                        // call sites' identical rounding pattern (see wrw_postlog's
                        // thread_length_cast above). A bare (output_size+255)/256 (a BLOCK
                        // count, not a workitem count) is NOT valid here -- confirmed on real
                        // hardware ("invalid argument" whenever that block count itself isn't
                        // also a multiple of 256).
                        size_t grid_reduce = (wsred_output_size + 255) / 256 * 256;
                        return igemm_launch_kernel_single(wrw_reduce_func, &karg_reduce, karg_reduce_size, {grid_reduce, 1, 1}, {256, 1, 1});
                    }
                    return .0;
                }};

                float duration = igemm_launch_kernels(kernel_launchers, wrw_gsplit_prolog, wrw_reduce_postlog, this->warmup, this->repeat);
                if (dump_dir.size())
                    dump_shader_args(dump_dir, result.dumpheader, result.dumpdata, result.kernel_name);
                return duration;
            };

            // Phase 58 (wrw_streamk): total_shards = num_k_blocks (finest granularity -- one
            // gemm_k_per_block-sized shard per claimable unit, matching rocKE's k_iter
            // granularity). The persistent grid.z is sized via the SAME occupancy heuristic
            // used for heuristic_candidate above (num_cu * max_occupancy_per_CU / grid_size),
            // capped to [1, total_shards] -- deliberately NO per-split-count search: that
            // search (and its real-timed-launch noise sensitivity) is exactly what a
            // persistent, self-balancing grid is meant to make unnecessary. See
            // docs/gfx1250_streamk_design.md.
            int streamk_persistent_grid_z = 1;
            auto time_streamk = [&]() -> float {
                // Phase 58 perf fix, take 2: shrinking grid.z alone (take 1) barely moved
                // the needle, because the total number of atomic-claim + LDS-broadcast +
                // double-barrier round trips is driven by the SHARD COUNT, not the worker
                // count -- with one gemm_k_per_block per shard, 1575 total_shards meant
                // ~1575 claims across the whole dispatch regardless of how many workers
                // shared them, and concentrating them into fewer workers (grid.z 1024->512)
                // just doubled max_iters, leaving the aggregate overhead unchanged (measured
                // no real improvement -- see docs/gfx1250_streamk_design.md). The actual
                // fix: make each CLAIMED UNIT cover multiple gemm_k_per_block blocks
                // (coarser granularity), cutting the total claim count proportionally while
                // keeping enough units for the persistent workers to still load-balance
                // across (aim for a handful of claims per worker, not one, and not
                // thousands).
                //
                // Grid.z target: ~one resident workgroup per CU total (each one already
                // self-refills via the atomic counter -- no need to oversubscribe),
                // spread across however many output tiles exist -- matches rocKE's own
                // compute_streamk_grid_size (min(num_macro_tiles, num_cus*blocks_per_cu)).
                // Uses the RAW device CU count, not this->num_cu -- the base class doubles
                // that for gfx10+ for the (unrelated) splits-heuristic's own purposes,
                // which just re-inflates the same worker count this fix is trying to shrink.
                hipDeviceProp_t dev_prop;
                hipDevice_t dev;
                HIP_CALL(hipGetDevice(&dev));
                HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
                // Research-only overrides for sweeping these constants externally without
                // rebuilding -- mirrors IGEMM_GSPLIT_SWEEP's role for the old design's split
                // count. Defaults match the hand-picked values that closed most of the
                // original ~4x gap; not yet swept/tuned per-shape (see
                // docs/gfx1250_streamk_design.md's "Resuming on another machine" list).
                const int blocks_per_cu = env_get_int("STREAMK_BLOCKS_PER_CU", 1);
                int grid_size = static_cast<int>(grid_x * grid_y);
                int target_total_workers = dev_prop.multiProcessorCount * blocks_per_cu;
                // Tuning update (idle-GPU sweep, same day): dividing target_total_workers by
                // grid_size (matching rocKE's own compute_streamk_grid_size shape) was the
                // default, but measured strictly worse on every multi-output-tile shape
                // tested -- fewer workers per tile means more max_iters (loop iterations)
                // per persistent workgroup, which shows up directly as wall-clock latency
                // even though total GPU-wide parallelism is technically similar. NOT
                // dividing (every tile independently targets target_total_workers) closed a
                // 512,30,40,128 shape from ~1.73x slower than _gsplit to ~1.07x, and
                // improved a 256,30,40,128 shape (which also has grid_y=2, not 1) from
                // ~1.31x to at/near parity -- with zero change on genuinely single-tile
                // shapes (grid_size=1 makes the two paths identical there). Made this the
                // default; STREAMK_DIVIDE_BY_TILES=1 restores the old rocKE-shaped behavior
                // for comparison.
                int per_tile_workers = env_get_int("STREAMK_DIVIDE_BY_TILES", 0)
                    ? std::max(1, (target_total_workers + grid_size - 1) / grid_size)
                    : target_total_workers;

                // Shard count: snapped to an exact divisor of num_k_blocks (every shard
                // must be an exact multiple of gemm_k_per_block -- no K-tail relief here).
                // Default max_shards=256 caps total claims (each claim costs a real atomic
                // + LDS-broadcast + double-barrier round trip). The idle-GPU max_iters sweep
                // (docs/gfx1250_streamk_design.md) proved max_iters=1 is always best --
                // per-claim barrier overhead scales linearly with max_iters. So the default
                // sizing targets grid_z == total_shards (max_iters=1): functionally identical
                // to _gsplit's one-workgroup-per-shard launch, but with the atomic-claim
                // mechanism in place for the contention-resilience benefit. Override via
                // STREAMK_MAX_SHARDS / STREAMK_CLAIMS_PER_WORKER / STREAMK_GRID_Z for
                // experimentation; max_iters>1 is expected to be slower but more predictable
                // under contention (see the contention-resilience section of the design doc).
                const int claims_per_worker_target = env_get_int("STREAMK_CLAIMS_PER_WORKER", 1);
                const int max_total_shards = env_get_int("STREAMK_MAX_SHARDS", 256);
                int shard_count_target = std::min(target_total_workers * claims_per_worker_target, max_total_shards);
                int total_shards = largest_divisor_leq(num_k_blocks, shard_count_target);

                // Grid.z: defaults to total_shards (max_iters=1, proven best on idle GPU).
                // STREAMK_GRID_Z override forces a smaller persistent grid (max_iters>1)
                // for contention-resilience experimentation -- see design doc's max_iters
                // sweep section for why this is slower on idle GPU but potentially more
                // predictable under contention.
                int grid_z_override = env_get_int("STREAMK_GRID_Z", 0);
                streamk_persistent_grid_z = grid_z_override > 0
                    ? std::min(grid_z_override, total_shards)
                    : total_shards;
                int max_iters = (total_shards + streamk_persistent_grid_z - 1) / streamk_persistent_grid_z;

                karg.gemm_k_per_wg = (num_k_blocks / total_shards) * tunable->gemm_k_per_block;
                karg.gemm_k_tail = 0;                              // asserted not wmma_k_tail
                karg.gemm_k_num_splits = total_shards;
                karg._streamk_pad = streamk_persistent_grid_z;  // repurposed: persistent grid.z for static shard indexing
                karg.p_streamk_counter = p_streamk_counter;
                karg.streamk_max_iters = max_iters;
                karg.streamk_grid_y = static_cast<int>(grid_y);
                size_t grid_z = static_cast<size_t>(streamk_persistent_grid_z);
                if (env_get_int("STREAMK_DEBUG", 0)) {
                    std::cout << "STREAMK_DEBUG: total_shards=" << total_shards
                              << " grid_z=" << streamk_persistent_grid_z
                              << " max_iters=" << max_iters
                              << " gemm_k_per_wg=" << karg.gemm_k_per_wg
                              << " grid_x=" << grid_x << " grid_y=" << grid_y
                              << std::endl;
                }

                result.dumpdata.clear();
                std::vector<igemm_launch_kernel_t> kernel_launchers;
                kernel_launchers.push_back({kernel_func, &karg, karg_size, {grid_x * block_size, grid_y, grid_z}, {block_size, 1, 1}});
                result.dumpheader.n_dispatches = kernel_launchers.size();
                result.dumpheader.gks = streamk_persistent_grid_z;
                result.dumpdata.push_back(kernel_launchers.back());
                result.dumpdata.back().ktype = kargtype_t::igemm_wrw_gtc_wmma_nhwc_karg_t;

                // Phase 58 Approach C: when wrw_reduction_kernel is set alongside wrw_streamk,
                // the main kernel writes non-atomic partial sums into disjoint workspace
                // slices (one per claimed shard, indexed by s_streamk_tile_idx computed
                // per-iteration inside emit_kernel_streamk_loop() -- NOT the static bz of
                // the old design). The postlog reduces them with the same
                // wrw_reduce_partials_f32 kernel time_split() already uses, parameterized
                // on total_shards (the actual partition count, guaranteed exact by the
                // divisor-snap logic above).
                std::function<float()> streamk_postlog = noop;
                if (tunable->wrw_reduction_kernel) {
                    streamk_postlog = [&, total_shards]() -> float {
                        wrw_reduce_karg_t karg_reduce;
                        karg_reduce.output = p_wei;
                        karg_reduce.workspace = p_wei_workspace_wsred;
                        karg_reduce.num_partitions = total_shards;
                        karg_reduce.output_size = static_cast<int>(wsred_output_size);
                        size_t karg_reduce_size = sizeof(karg_reduce);
                        size_t grid_reduce = (wsred_output_size + 255) / 256 * 256;
                        return igemm_launch_kernel_single(wrw_reduce_func, &karg_reduce, karg_reduce_size, {grid_reduce, 1, 1}, {256, 1, 1});
                    };
                }

                float duration = igemm_launch_kernels(kernel_launchers, wrw_gsplit_prolog, streamk_postlog, this->warmup, this->repeat);
                if (dump_dir.size())
                    dump_shader_args(dump_dir, result.dumpheader, result.dumpdata, result.kernel_name);
                return duration;
            };

            float min_duration = FLT_MAX;
            int selected_splits = 1;
            // Research-only override for sweeping the perf-vs-split-count curve externally
            // without rebuilding: IGEMM_GSPLIT_SWEEP=<target> forces a single candidate
            // snapped to the nearest valid divisor of that target, bypassing the search.
            int sweep_target = tunable->gemm_k_global_split ? env_get_int("IGEMM_GSPLIT_SWEEP", 0) : 0;
            if (tunable->wrw_streamk) {
                min_duration = time_streamk();
                selected_splits = streamk_persistent_grid_z;
            } else if (sweep_target > 0) {
                selected_splits = largest_divisor_leq(num_k_blocks, sweep_target);
                min_duration = time_split(selected_splits);
            } else {
                // Ternary search over the sorted divisor list, exploiting the measured
                // unimodality (see comment above `divisors`). Caches each evaluated index so
                // the narrowing loop and the final exhaustive confirm never re-launch the
                // same split count twice. Degenerates to a single evaluation of divisors[0]
                // (==1) when gemm_k_global_split is off, since `divisors` is just {1} then.
                std::vector<float> cache(divisors.size(), -1.0f);
                auto eval = [&](int idx) -> float {
                    if (cache[idx] < 0.0f) cache[idx] = time_split(divisors[idx]);
                    return cache[idx];
                };
                int lo = 0, hi = static_cast<int>(divisors.size()) - 1;
                while (hi - lo > 4) {
                    int m1 = lo + (hi - lo) / 3;
                    int m2 = hi - (hi - lo) / 3;
                    if (eval(m1) <= eval(m2)) hi = m2; else lo = m1;
                }
                // Final small window (<=5 candidates): confirm the true minimum directly
                // rather than trusting the last ternary comparison, since real hardware
                // timings are noisy and the curve can be locally flat near its minimum.
                for (int idx = lo; idx <= hi; idx++) {
                    float d = eval(idx);
                    if (d < min_duration) { min_duration = d; selected_splits = divisors[idx]; }
                }
            }

            result.return_code = 0;
            result.duration_ms = min_duration;
            result.gks = selected_splits;
            if (p_wei_workspace_wsred != nullptr)
                HIP_CALL(hipFree(p_wei_workspace_wsred));
            if (p_streamk_counter != nullptr)
                HIP_CALL(hipFree(p_streamk_counter));
#ifdef IGEMM_SPLIT_KERNEL
            HIP_CALL(hipModuleUnload(cur_kernel_module));
#endif
            return result;
        }

        if(this->driver_mode == driver_mode_heuristic)
            current_gks = tunable->gemm_k_global_split ? current_gks : 0;

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
        int data_byte = utility_string_to_data_byte(tunable->precision);
        assert(c % group == 0 && k % group == 0);

        size_t splits = igemm_split_batch_size(arg, utility_string_to_data_byte(tunable->precision));
        assert(splits != 0);
        n = n/splits;   // split batch size here

        int gemm_k_per_block         = tunable->gemm_k_per_block;

        size_t block_size            = get_block_size(tunable);
        int gemm_k_global_split      = tunable->gemm_k_global_split;

        size_t cur_grid_size = get_cur_grid_size(arg, tunable);

        int b                        = ho * wo;
        if(tunable->tensor_layout == "nchw")
            b  = tunable->nxe == 0 ? (ho * wo) : ((ho * wo + tunable->nxb - 1) / tunable->nxb) * tunable->nxb;
        int max_grid_size = 1200;
        int min_n_per_block = 1;
        if(tunable->tensor_layout == "nhwc" && tunable->nxe == 1)
            min_n_per_block = tunable->tensor_a_thread_lengths[1];

        int nb_per_block = tunable->gemm_k_per_block;
        if(tunable->tensor_layout == "nhwc" && tunable->nxe == 1)
            nb_per_block = tunable->tensor_a_cluster_lengths[1];

        size_t gemm_k_global_splits;
        if(tunable->tensor_layout == "nchw"){
            gemm_k_global_splits = gemm_k_global_split == 1 ? compute_log2_gemmk_global_splits(cur_grid_size, max_grid_size, n / min_n_per_block, b, gemm_k_per_block)
                                                                 : 0;
        }else{        
            gemm_k_global_splits = gemm_k_global_split == 1 ? compute_gemmk_global_splits(cur_grid_size, 3)
                                                                 : 0;
        }

        int use_workspace = 0;

        if(gemm_k_global_split == 1 && tunable->precision == "fp16" && (tunable->tensor_b_thread_lengths[3] == 1 || tunable->vector_store == 1))
            use_workspace = 1;
        else if(gemm_k_global_split == 1 && tunable->precision == "bf16")
            use_workspace = 1;
        else
            use_workspace = 0;

        size_t workspace_size = get_workspace_size(arg, tunable);
        void *p_wei_workspace;
        if(workspace_size == 0)
            p_wei_workspace = nullptr;
        else
            HIP_CALL(hipMalloc(&p_wei_workspace, workspace_size));

        kargtype_t ktype = kargtype_t::igemm_wrw_gtc_karg_t;
        igemm_wrw_gtc_karg_t karg;
        size_t karg_size = sizeof(karg);
        karg.p_in          = p_in;
        if(use_workspace == 1){
            karg.p_wei     = p_wei_workspace;
        } else{
            karg.p_wei     = p_wei;
        }
        karg.p_out         = p_out;
        karg.hi            = hi;
        karg.wi            = wi;
        karg.n             = n;
        karg.k             = k / group;
        karg.c             = c / group;
        karg.ho            = ho;
        karg.wo            = wo;

        karg.stride_h      = stride_h;
        karg.stride_w      = stride_w;
        karg.dilation_h    = dilation_h;
        karg.dilation_w    = dilation_w;
        karg.pad_h         = pad_h;
        karg.pad_w         = pad_w;
        karg.y             = y;
        karg.x             = x;
        karg.gemm_k_global_split = gemm_k_global_splits;
        karg.group         = group;
        //karg.gemm_k_per_wg = (int)(ceil((n / min_n_per_block) * b / (float)gemm_k_global_splits));
        //karg.gemm_k_per_wg = (karg.gemm_k_per_wg + nb_per_block - 1) / nb_per_block * nb_per_block;

        //gemm_k_global_splits = (int)(ceil((n / min_n_per_block) * b / (float)(karg.gemm_k_per_wg)));

        // tensor cast kernel args
        tensor_cast_karg_t karg_tensor_cast;
        karg_tensor_cast.output = p_wei;
        karg_tensor_cast.input = p_wei_workspace; 
        karg_tensor_cast.total_length = group * (k / group) * (c / group) * y * x;

        size_t karg_tensor_cast_size = sizeof(karg_tensor_cast);

        //int block_size = get_block_size(tunable);
        size_t grid_size = get_grid_size(arg, tunable);

        hipFunction_t kernel_func;
        std::string kernel_name = get_kernel_name(tunable);
        //dump_wrw_karg(&karg);
        //printf("kernel:%s\n, block:%d, grid:%d, gemm_k_global_split:%d\n", kernel_name.c_str(), block_size, grid_size, gemm_k_global_split);
        
#ifdef IGEMM_SPLIT_KERNEL
        hipModule_t cur_kernel_module;
        std::string cur_kernel_hsaco = kernel_name + ".hsaco";
        HIP_CALL(hipModuleLoad(&cur_kernel_module, cur_kernel_hsaco.c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, cur_kernel_module, kernel_name.c_str()));
#else
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));
#endif

        hipFunction_t tensor_cast_func;
        if(use_workspace == 1){
            std::string tensor_cast_kernel_name = tunable->precision == "fp16" ? "tensor_cast_fp16_fp32_1d" : "tensor_cast_bf16_fp32_1d";
            HIP_CALL(hipModuleGetFunction(&tensor_cast_func, module_tensor_cast, tensor_cast_kernel_name.c_str()));
        }

        auto wrw_prolog = gemm_k_global_split ? 
            std::function<float()>{[&]() -> float{
                if(use_workspace == 1)
                    HIP_CALL(hipMemset(p_wei_workspace, 0x0, group * (k / group) * (c / group) * y * x * sizeof(float)));
                else
                    HIP_CALL(hipMemset(p_wei, 0x0, group * (k / group) * (c / group) * y * x * data_byte));
                return .0;
            }} : 
            std::function<float()>{[&]() -> float{
                return .0;
            }};
        const size_t thread_length_cast = (static_cast<size_t>(group) * (k / group) * (c / group) * y * x + 8 * 256) / (8 * 256) * (8 * 256) / 8;
        auto wrw_postlog = use_workspace == 1 ?
            std::function<float()>{[&]() -> float{
                return igemm_launch_kernel_single(tensor_cast_func, &karg_tensor_cast, karg_tensor_cast_size, {thread_length_cast, 1, 1}, {256, 1, 1});
            }} :
            std::function<float()>{[&]() -> float{
                return .0;
            }};

        result_t result;
        result.kernel_name = kernel_name;

        std::string dump_dir = env_get_str("IGEMM_DUMPDIR_ALL", "");
        if (dump_dir.size()) {
            std::cout << "DEBUG: Dumping all dispatches to " << dump_dir
                      << std::endl;
        }
        result.dumpdata.clear();
        memset(&result.dumpheader, 0, sizeof(result.dumpheader));
        result.dumpheader.workspace_size = workspace_size;
        result.dumpheader.gi_postlog.gsize = {
            static_cast<uint32_t>(thread_length_cast), 1, 1};
        result.dumpheader.gi_postlog.wsize = {256, 1, 1};
        result.dumpheader.use_prolog = gemm_k_global_split;
        result.dumpheader.use_postlog = use_workspace == 1;
        result.dumpheader.cast_total_length = karg_tensor_cast.total_length;
        result.dumpheader.conv.hi = hi;
        result.dumpheader.conv.wi = wi;
        result.dumpheader.conv.n = n;
        result.dumpheader.conv.k = k;
        result.dumpheader.conv.c = c;
        result.dumpheader.conv.ho = ho;
        result.dumpheader.conv.wo = wo;
        result.dumpheader.conv.stride_h = stride_h;
        result.dumpheader.conv.stride_w = stride_w;
        result.dumpheader.conv.ddilation_h = 1;
        result.dumpheader.conv.ddilation_w = 1;
        result.dumpheader.conv.fdilation_h = dilation_h;
        result.dumpheader.conv.fdilation_w = dilation_w;
        result.dumpheader.conv.pad_h = pad_h;
        result.dumpheader.conv.pad_w = pad_w;
        result.dumpheader.conv.y = y;
        result.dumpheader.conv.x = x;
        result.dumpheader.conv.group = group;
        result.dumpheader.conv.dir = convdir_t::WRW;
        result.dumpheader.conv.dtype = dtype(tunable->precision);

        int max_split_num = tunable->gemm_k_global_split == 0 ? 1 : (this->max_gks == -1 ? WRW_MAX_GEMM_K_SPLITS : this->max_gks);
        float min_duration = FLT_MAX;
        int selected_gkgs = 0;
        int selected_grid_size = 0;
        //max_split_num = 1;
        auto run_with_gks = [&](int _gks){
            if(tunable->tensor_layout == "nhwc"){
                //for(int gkgs = 0; gkgs < max_split_num; gkgs++){
                std::vector<igemm_launch_kernel_t> kernel_launchers;

                // This is hacky, but in MIOpen we prefer a heuristic way to set gks, so ok now. 
                gemm_k_global_splits = _gks == 0 ? 1 : compute_gemmk_global_splits(cur_grid_size, _gks);
                if(gemm_k_global_splits == 0){
                    gemm_k_global_splits = 1;
                }
                int tmp_gemm_k_per_wg = (int)(ceil(ceil(n / (float)min_n_per_block) * b / (float)gemm_k_global_splits));
                tmp_gemm_k_per_wg = (tmp_gemm_k_per_wg + nb_per_block - 1) / nb_per_block * nb_per_block;
                gemm_k_global_splits = (int)(ceil(ceil(n / (float)min_n_per_block) * b / (float)(tmp_gemm_k_per_wg)));
                karg.gemm_k_global_split = gemm_k_global_splits;
                karg.gemm_k_per_wg = tmp_gemm_k_per_wg;
                // printf("gemm_k_global_splits=%d, tmp_gemm_k_per_wg=%d\n", gemm_k_global_splits, tmp_gemm_k_per_wg);
                // fflush(stdout);

                kernel_launchers.push_back({kernel_func, &karg, karg_size, {grid_size * block_size, splits, gemm_k_global_splits}, {block_size, 1, 1}});
                // if(use_workspace == 1){
                //     size_t thread_length_cast = (static_cast<size_t>(group) * (k / group) * (c / group) * y * x + 8 * 256) / (8 * 256) * (8 * 256) / 8;
                //     kernel_launchers.push_back({tensor_cast_func, &karg_tensor_cast, karg_tensor_cast_size, {thread_length_cast, 1, 1}, {256, 1, 1}});
                // }
                result.dumpheader.n_dispatches = kernel_launchers.size();
                result.dumpheader.gks = gemm_k_global_splits;
                result.dumpdata.push_back(kernel_launchers.back());
                result.dumpdata.back().ktype = ktype;

                float duration = igemm_launch_kernels({
                        kernel_launchers
                    }, wrw_prolog, wrw_postlog, this->warmup, this->repeat);
                if(min_duration > duration){
                    min_duration = duration;
                    selected_gkgs = _gks;
                    selected_grid_size = grid_size * gemm_k_global_splits;
                }
                // printf("block:%d, grid:%d, split:%d, duration:%f\n", block_size, grid_size, gemm_k_global_splits, duration);
                // fflush(stdout);

            }else{
                // nchw do not search for gemmksplit
                std::vector<igemm_launch_kernel_t> kernel_launchers = {
                    {kernel_func,
                     &karg,
                     karg_size,
                     {grid_size * block_size, 1, 1},
                     {block_size, 1, 1}}};

                result.dumpheader.n_dispatches = kernel_launchers.size();
                result.dumpheader.gks = gemm_k_global_splits;
                result.dumpdata.push_back(kernel_launchers.back());
                result.dumpdata.back().ktype = ktype;

                float duration = igemm_launch_kernels(kernel_launchers, wrw_prolog, wrw_postlog, this->warmup, this->repeat);
                min_duration = duration;
                selected_gkgs = gemm_k_global_splits;
                selected_grid_size = grid_size;

            }
        };
        if(current_gks != -1){
            run_with_gks(current_gks);
            if (dump_dir.size())
                    dump_shader_args(dump_dir, result.dumpheader, result.dumpdata, result.kernel_name);
            result.dumpdata.clear();
        }else{
            std::vector<int> all_gks = get_gks_list(arg, tunable);
            for(int gks : all_gks){
                run_with_gks(gks);
                if (dump_dir.size())
                    dump_shader_args(dump_dir, result.dumpheader, result.dumpdata, result.kernel_name);
                result.dumpdata.clear();
            }
        }

        result.return_code = 0;
        result.duration_ms = min_duration;
        result.gks         = selected_gkgs;
        result.grid_size   = selected_grid_size;
		// debug section of code
#if 0
        printf("workspace debug \r\n");
        float* gemmc_host_check = (float* )malloc((1 << gemm_k_global_split) * k * c * y * x * sizeof(float));
        printf("gemmc_host_check size=%d\n", (1 << gemm_k_global_split) * k * c * y * x * sizeof(float));
        if(gemm_k_global_split > 0){
            printf("copy workspace\n");
            //hipMemcpy(gemmc_host_check, p_wei_workspace, (1 << gemm_k_global_split) * k * c * y * x * sizeof(float), hipMemcpyDeviceToHost);
            hipMemcpy(gemmc_host_check, p_wei, group * (k / group) * (c / group) * y * x * sizeof(float16), hipMemcpyDeviceToHost);
        }
        else{
            printf("copy weight\n");
            hipMemcpy(gemmc_host_check, p_wei, group * (k / group) * (c / group) * y * x * sizeof(float16), hipMemcpyDeviceToHost);
        }
        for (int i_check = 0; i_check < (0+block_size); i_check++)
        {
            float16 *gemmc_host_check_fp16 = (float16 *)gemmc_host_check;
            float16 check_num0 = gemmc_host_check_fp16[i_check*2];
            float16 check_num1 = gemmc_host_check_fp16[i_check*2+1];
            float check_num0_fp32 = (float)check_num0;
            float check_num1_fp32 = (float)check_num1;
            printf("[%d]th var to monitor:[%f, %d, fp16(%f, %f)]\r\n", i_check, gemmc_host_check[i_check], ((int *)gemmc_host_check)[i_check], check_num0_fp32, check_num1_fp32);
        }
        printf("s_p_in=%x\n", p_in);
        printf("workspace debug end \r\n");
        free(gemmc_host_check);
#endif
#ifdef IGEMM_SPLIT_KERNEL
        HIP_CALL(hipModuleUnload(cur_kernel_module));
#endif
        if(workspace_size > 0)
            HIP_CALL(hipFree(p_wei_workspace));
        return result;
    }
    std::vector<int> get_gks_list(const args_t *arg, const igemm_gtc_tunable_t *tunable) override
    {
        if (!tunable_is_valid(arg, tunable)) {
            return std::vector<int>{0};
        }
        size_t cur_grid_size_t = get_cur_grid_size(arg, tunable);
        int hi = arg->get_int("in_h");
        int wi = arg->get_int("in_w");
        int n = arg->get_int("batchsize");

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

        int min_n_per_block = 1;
        if(tunable->tensor_layout == "nhwc" && tunable->nxe == 1)
            min_n_per_block = tunable->tensor_a_thread_lengths[1];

        int nb_per_block = tunable->gemm_k_per_block;
        if(tunable->tensor_layout == "nhwc" && tunable->nxe == 1)
            nb_per_block = tunable->tensor_a_cluster_lengths[1];

        int b = ho * wo;
        if(tunable->tensor_layout == "nchw")
            b = tunable->nxe == 0 ? (ho * wo) : ((ho * wo + tunable->nxb - 1) / tunable->nxb) * tunable->nxb;

        if(tunable->gemm_k_global_split == 0)
            return std::vector<int>{0};
        else{
            int max_split_num = tunable->gemm_k_global_split == 0 ? 1 : (this->max_gks == -1 ? WRW_MAX_GEMM_K_SPLITS : this->max_gks);

            std::vector<int> gks_list;
            std::vector<int> real_gks_list;
            for(int gks = 0; gks <= max_split_num; gks++){
                auto real_gks = compute_gemmk_global_splits(cur_grid_size_t, gks);
                if(real_gks == 0){
                    real_gks = 1;
                }
                int tmp_gemm_k_per_wg = (int)(ceil(ceil(n / (float)min_n_per_block) * b / (float)real_gks));
                tmp_gemm_k_per_wg = (tmp_gemm_k_per_wg + nb_per_block - 1) / nb_per_block * nb_per_block;
                real_gks = (int)(ceil(ceil(n / (float)min_n_per_block) * b / (float)(tmp_gemm_k_per_wg)));
                if(std::find(real_gks_list.begin(), real_gks_list.end(), real_gks) != real_gks_list.end()){
                    continue;
                }
                else{
                    real_gks_list.push_back(real_gks);
                    gks_list.push_back(gks);
                }
            }
            assert(gks_list.size() != 0);
            return gks_list;
        }
    }
    igemm_spatial_tiling_t get_spatial_tiling(const args_t *arg) override
    {
        return igemm_spatial_tiling_t{};
    }
};

#endif