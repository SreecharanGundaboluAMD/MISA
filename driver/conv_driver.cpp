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
#include <chrono>
#include <functional>

#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <time.h>
#include <vector>
#include <float.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <iostream>

#ifndef USE_EXT_MODULE_LAUNCH
#define USE_EXT_MODULE_LAUNCH 1
#endif

#ifndef USE_MAGIC_DIV
#define USE_MAGIC_DIV 0
#endif

#ifndef USE_SOURCE_ACCESS_ENCODING_KERNEL_NAME
#define USE_SOURCE_ACCESS_ENCODING_KERNEL_NAME 0
#endif

#ifdef USE_GPU_NAIVE_CONV
#   include "gpu_naive_conv.h"
#   ifndef IGEMM_GPU_NAIVE_CONV_HSACO
#       define  IGEMM_GPU_NAIVE_CONV_HSACO "naive_conv.hsaco"
#   endif
#else
#   define NAIVE_CONV_THREADED
#   include "naive_conv.h"
#endif

#ifndef USE_MIOPEN_NRMS
#define USE_MIOPEN_NRMS 1
#endif

#include "common.h"
#include "args.h"
#include "shisa_dumps.h"
#include "config_parser.h"
#include "perf.h"
#include "tensor_transpose.h"
#include "tensor_copy_cpu.h"
#include "tensor_validation_cpu.h"
#include "igemm_gtc_base.h"
#include "igemm_fwd_gtc_driver.h"
#include "igemm_bwd_gtc_driver.h"
#include "igemm_wrw_gtc_driver.h"

void dump_shader_args(std::string dump_dir,
                      const dumpheader_t &const_header,
                      const std::vector<dispatchinfo_t> &dispatches,
                      std::string kernel_name)
{
    dumpheader_t header = const_header;
    header.version = DUMPFILE_VERSION;
    if (header.n_dispatches != dispatches.size()) {
        std::cout << "ERROR: header.n_dispatches != dispatches.size()" << std::endl;
        assert(0);
    }

    std::string dump_path = dump_dir + kernel_name + ".gks" + std::to_string(header.gks) + ".dump";
    std::ofstream fs(dump_path, std::ios::out | std::ios::binary);
    fs.write(reinterpret_cast<const char *>(&header), sizeof(dumpheader_t));
    for (auto &di : dispatches) {
        fs.write(reinterpret_cast<const char *>(&di), sizeof(di));
    }
    fs.write(kernel_name.c_str(), kernel_name.size());
    fs.close();
}

misadatatype_t dtype(const std::string &s) {
    if (!s.compare("fp32"))
        return misadatatype_t::FP32;
    if (!s.compare("fp16"))
        return misadatatype_t::FP16;
    if (!s.compare("bf16"))
        return misadatatype_t::BF16;
    if (!s.compare("int8"))
        return misadatatype_t::INT8;
    if (!s.compare("int4"))
        return misadatatype_t::INT4;
    return misadatatype_t::UNKNOWN;
}

static inline double theoritical_gflops(double sclk_ghz, size_t cu,
                                             size_t simd) {
    return 2 * sclk_ghz * cu * simd;
}
static inline double
theoritical_fp32_conv_flop(size_t n, size_t c, size_t hi, size_t wi, size_t k,
                           size_t y, size_t x, size_t stride_h, size_t stride_w,
                           size_t dilation_h, size_t dilation_w, size_t pad_h,
                           size_t pad_w, size_t ngroups) {
    size_t ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
    size_t wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);

    double flop = (double)n * c * ho * wo * k * y * x * 2 / ngroups;
    return flop;
}
static inline double
measured_fp32_conv_gflops(double time_ms, size_t n, size_t c, size_t hi,
                          size_t wi, size_t k, size_t y, size_t x,
                          size_t stride_h, size_t stride_w, size_t dilation_h,
                          size_t dilation_w, size_t pad_h, size_t pad_w, size_t ngroups) {
    double flop =
        theoritical_fp32_conv_flop(n, c, hi, wi, k, y, x, stride_h, stride_w,
                                   dilation_h, dilation_w, pad_h, pad_w, ngroups);
    return flop / (time_ms * 1e6);
}


static inline double get_theoritical_conv_flop(const args_t * conv_args)
{
    int hi = conv_args->get_int("in_h");
    int wi = conv_args->get_int("in_w");
    int n = conv_args->get_int("batchsize");
    int k = conv_args->get_int("out_channels");
    int c = conv_args->get_int("in_channels");

    int stride_h = conv_args->get_int("conv_stride_h");
    int stride_w = conv_args->get_int("conv_stride_w");
    int dilation_h = conv_args->get_int("dilation_h");
    int dilation_w = conv_args->get_int("dilation_w");
    int pad_h = conv_args->get_int("pad_h");
    int pad_w = conv_args->get_int("pad_w");
    int y = conv_args->get_int("fil_h");
    int x = conv_args->get_int("fil_w");
    int ngroups = conv_args->get_int("group_count");
    int ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
    int wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);

    return theoritical_fp32_conv_flop(n, c, hi, wi, k, y, x, stride_h, stride_w,
                                   dilation_h, dilation_w, pad_h, pad_w, ngroups);
}

static inline double get_theoritical_gpu_gflops(int sclk_mhz, driverDataType_t data_type)
{
    int num_cu;
    int gcn_arch = 0;
    int num_simd = 4 * 16;
    hipDeviceProp_t dev_prop;
    hipDevice_t dev;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    num_cu = dev_prop.multiProcessorCount;
    gcn_arch = get_gcn_arch(dev_prop.gcnArchName);
    // gfx10/RDNA reports WGP count (each = 2 CUs), so double for those.
    // gfx1250/CDNA5 reports actual CU count, so don't double.
    if(gcn_arch >= 1000 && gcn_arch != 1250)
        num_cu *= 2;

    // If the caller didn't override sclk via IGEMM_SCLK_MHZ (left it at the
    // hardcoded default), use the device's actual clock rate instead.
    if(sclk_mhz == 1283)
        sclk_mhz = dev_prop.clockRate / 1000;

    int fp_factor = 1;
    if(data_type == driverHalf){
        if(gcn_arch == 950 || gcn_arch == 942 || gcn_arch == 941 || gcn_arch == 940)
            fp_factor = 8;  // xdlops
        else if(gcn_arch == 908 || gcn_arch == 910)
            fp_factor = 4;  // xdlops
        else if(gcn_arch == 1250)
            fp_factor = 8;  // wmma, v_wmma_f32_16x16x32_f16
        else
            fp_factor = 2;  // dlops
        if(gcn_arch >= 1000 && gcn_arch != 1250)
            fp_factor = 2;
    }
    // bf16 WMMA has the same matrix dimensions and throughput as fp16 WMMA
    // (v_wmma_f32_16x16x32_bf16 vs v_wmma_f32_16x16x32_f16), so the bf16
    // fp_factor must match fp16. Without this branch, bf16 fell through to
    // fp_factor=1, understating peak by 8x and inflating efficiency 8x.
    if(data_type == driverBFloat16){
        if(gcn_arch == 950 || gcn_arch == 942 || gcn_arch == 941 || gcn_arch == 940)
            fp_factor = 8;  // xdlops
        else if(gcn_arch == 908 || gcn_arch == 910)
            fp_factor = 4;  // xdlops
        else if(gcn_arch == 1250)
            fp_factor = 8;  // wmma, v_wmma_f32_16x16x32_bf16
        else
            fp_factor = 2;  // dlops
        if(gcn_arch >= 1000 && gcn_arch != 1250)
            fp_factor = 2;
    }
    if(data_type == driverInt8){
        if(gcn_arch == 950 || gcn_arch == 942 || gcn_arch == 941 || gcn_arch == 940)
            fp_factor = 8;  // xdlops
        else if(gcn_arch == 908 || gcn_arch == 910)
            fp_factor = 4;  // xdlops
        else if(gcn_arch == 1250)
            fp_factor = 16; // wmma, v_wmma_i32_16x16x64_iu8
        else
            fp_factor = 4;  // dlops
    }
    if(data_type == driverInt4){
        if(gcn_arch >= 1000)
            fp_factor = 8;  // xdlops
    }
    // else if(data_type == driverInt8){
    //     if(gcn_arch == 908)
    //     fp_factor = 4;
    // }

    // COR-005 (2026-09-02): gfx1250 was silently falling through to the generic/
    // xdlops-era default of num_simd = 4*16 = 64 lanes/CU set at this function's top,
    // because it wasn't in this arch list -- unlike fp_factor just above (lines 190-232),
    // which already special-cases gcn_arch == 1250. rocminfo/hipDeviceProperties on the
    // real gfx1250 part report "SIMDs per CU: 4" with a wave32 (32-lane) SIMD width, i.e.
    // 4*32 = 128 base FMA lanes/CU -- the SAME physical shape as the CDNA parts already
    // listed here, not the RDNA/gfx10-era 4*16 default (that default predates gfx1250 and
    // assumed the WGP-doubled CU counting gfx1250 explicitly opts out of above, line 182).
    // Silently using 64 instead of 128 halved the computed peak, which combined with
    // fp16/bf16's fp_factor=8 was reporting impossible >100% efficiency on real fp32
    // kernels and understating the true fp16 peak (60.41% of a too-low peak, rather than
    // the correct, much smaller percentage of the true, doubled peak).
    if(gcn_arch == 908 || gcn_arch == 910 || gcn_arch == 950 || gcn_arch == 942 || gcn_arch == 941 || gcn_arch == 940 || gcn_arch == 1250){
        num_simd = 4 * 32 ; // 4x miSIMD, 32x mac unit
    }

    return theoritical_gflops(((double)sclk_mhz) / 1000.0, num_cu, num_simd * fp_factor);
}

#ifndef IGEMM_HSACO
#define IGEMM_HSACO "igemm_gtc.hsaco"
#endif

#ifndef IGEMM_TENSOR_CAST_HSACO
#define IGEMM_TENSOR_CAST_HSACO "igemm_gtc_tensor_cast.hsaco"
#endif

#ifndef IGEMM_CONFIG_FILE
#define IGEMM_CONFIG_FILE "igemm_gtc.config"
#endif

#define IGEMM_RUN_ONLY_KERNEL_DEFAULT "off"

#define WARMUP 3
#define REPEAT 8
#define SCLK_MHZ 1283

template <typename T>
struct distribution_t{
};

template <>
struct distribution_t<int>{
    distribution_t(int min, int max) : distribution(min, max) {}
    template<class URNG>
    int operator()(URNG & rng){ return distribution(rng);}
    std::uniform_int_distribution<int> distribution;
};
template <>
struct distribution_t<float>{
    distribution_t(float min, float max) : distribution(min, max) {}
    template<class URNG>
    float operator()(URNG & rng){ return distribution(rng);}
    std::uniform_real_distribution<float> distribution;
};

template <typename Dst_T, typename Src_T>
void block_wise_rand_generator(Dst_T *p, int tid, int block_size, size_t total_size, Src_T min, Src_T max, Src_T scale)
{
    std::mt19937 rng(std::chrono::system_clock::now()
                        .time_since_epoch()
                        .count() +
                    std::hash<std::thread::id>()(std::this_thread::get_id()));
    distribution_t<Src_T> distribution(min,max);
    for (size_t i = tid; i < total_size; i += block_size) {
        p[i] = static_cast<Dst_T>(scale * distribution(rng));
    }
}

template <typename Dst_T, typename Src_T>
void gen_rand_vector(Dst_T *vec, size_t vec_size, Src_T fmin, Src_T fmax, Src_T scale = 1) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads < 4)
        num_threads = 4;
    // printf("total threads:%d\n",num_threads);
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.push_back(std::thread(block_wise_rand_generator<Dst_T, Src_T>,
            vec, t, num_threads, vec_size, fmin, fmax, scale));
    }
    for (auto &th : threads)
        th.join();
}

void dump_arg(const args_t *arg) {
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
    int ngroups = arg->get_int("group_count");

    printf("n:%d, c:%d, h:%d, w:%d, k:%d, y:%d, x:%d, sy:%d, sx:%d, dy:%d, "
           "dx:%d, py:%d, px:%d, ho:%d, wo:%d, group:%d\n",
           n, c, hi, wi, k, y, x, stride_h, stride_w, dilation_h, dilation_w,
           pad_h, pad_w, ho, wo, ngroups);
}

int string_to_dir(std::string direction)
{
    if(direction == "fwd")
        return 1;
    if(direction == "bwd")
        return 2;
    if(direction == "wrw")
        return 4;
    assert(0);
}

std::string log_cmd(const args_t *arg, driverDataType_t driver_data_type, std::string direction)
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
    int ngroups = arg->get_int("group_count");
    std::string in_layout = arg->get_str("in_layout");
    std::string out_layout = arg->get_str("out_layout");
    std::string fil_layout = arg->get_str("fil_layout");

    std::stringstream ss;
    if(driver_data_type == driverHalf)
    {
        ss << "convfp16";
    }
    else if(driver_data_type == driverFloat)
    {
        ss << "conv";
    }

    ss << " -n " << n
        << " -c " << c
        << " -H " << hi
        << " -W " << wi
        << " -k " << k
        << " -y " << y
        << " -x " << x
        << " -p " << pad_h
        << " -q " << pad_w
        << " -u " << stride_h
        << " -v " << stride_w
        << " -l " << dilation_h
        << " -j " << dilation_w;

    if(in_layout != "NCHW")
        ss << " --in_layout " << in_layout;
    if(fil_layout != "NCHW")
        ss << " --fil_layout " << fil_layout;
    if(out_layout != "NCHW")
        ss << " --out_layout " << out_layout;

    ss << " -g " << ngroups
            << " -F " << std::to_string(string_to_dir(direction))
            << " -t 1";

    return ss.str();
}

template<typename driver_t>
std::string get_tiling_string(driver_t * driver, const args_t *conv_args)
{
    int hi = conv_args->get_int("in_h");
    int wi = conv_args->get_int("in_w");

    int stride_h = conv_args->get_int("conv_stride_h");
    int stride_w = conv_args->get_int("conv_stride_w");
    int dilation_h = conv_args->get_int("dilation_h");
    int dilation_w = conv_args->get_int("dilation_w");
    int pad_h = conv_args->get_int("pad_h");
    int pad_w = conv_args->get_int("pad_w");
    int y = conv_args->get_int("fil_h");
    int x = conv_args->get_int("fil_w");

    int ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
    int wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);

    igemm_spatial_tiling_t tiling = driver->get_spatial_tiling(conv_args);
    if((tiling.tile_h == 0 && tiling.tile_w == 0) || 
        (tiling.tile_h == ho && tiling.tile_w == wo))
        return "";
    else{
        return std::string("[") + std::to_string(tiling.tile_w) + "x" + std::to_string(tiling.tile_h) + "]";
    }
}

template<typename driver_t, typename pre_func_t, typename post_func_t>
void launch_conv_driver(driver_t * driver, const args_t *conv_args, const std::vector<igemm_gtc_tunable_t> & tunables, std::string direction,
                    driverDataType_t driver_data_type, FILE * p_bcsv,
                    void* device_input, void* device_weight, void* device_output,
                    pre_func_t && pre_func, post_func_t && post_func)
{
    int sclk_mhz = env_get_int("IGEMM_SCLK_MHZ", SCLK_MHZ);
    std::string run_only_kernel = env_get_str("IGEMM_RUN_ONLY_KERNEL", IGEMM_RUN_ONLY_KERNEL_DEFAULT);
    int log_fastest_config = env_get_int("IGEMM_LOG_FASTEST_CONFIG", 0);
    int sleep_ms = env_get_int("IGEMM_SLEEP_MS", 0);
    int dump_gmap = env_get_int("IGEMM_DUMP_GMAP", 0);
    int gks_iterative = env_get_int("IGEMM_GKS_ITERATIVE", 0);
    int max_mpb = env_get_int("IGEMM_MAX_MPB", -1);
    int max_npb = env_get_int("IGEMM_MAX_NPB", -1);
    int max_kpb = env_get_int("IGEMM_MAX_KPB", -1);
    int max_gks = env_get_int("IGEMM_MAX_GKS", -1);
    int silent_not_applicable_level0 = env_get_int("IGEMM_SILENT_NA_L0", 1);  // ignore kernel that has different direction & layout
    std::string in_layout = conv_args->get_str("in_layout");
    std::string fil_layout = conv_args->get_str("fil_layout");

    double theo_conv_flop  = get_theoritical_conv_flop(conv_args);
    double theo_gpu_gflops = get_theoritical_gpu_gflops(sclk_mhz, driver->data_type);

    auto launch = [&](const igemm_gtc_tunable_t * tunable, int index, int current_gks, bool is_tunable_predicted = false) -> result_t {
        igemm_gtc_tunable_t predicted_tunable;
        const igemm_gtc_tunable_t * current_tunable = tunable;
        if(is_tunable_predicted){
            predicted_tunable = *tunable;
            // in prediction, the gks will be 0, 1, 2... if tunable support gks, other wise it is -1.
            // here we restore the gemm_k_global_split inside the tunable
            predicted_tunable.gemm_k_global_split = current_gks >= 0 ? 1 : 0;
            current_tunable = &predicted_tunable;
        }
        if(run_only_kernel != IGEMM_RUN_ONLY_KERNEL_DEFAULT){
            if(run_only_kernel != driver->get_kernel_name(current_tunable))
                {result_t result; result.return_code = -2; return result;}
        }
        if(silent_not_applicable_level0){
            if(direction != current_tunable->direction)
                {result_t result; result.return_code = -2; return result;}
            if(in_layout == "NCHW"){
                if(current_tunable->tensor_layout != "nchw")
                    {result_t result; result.return_code = -2; return result;}
            }else if(in_layout == "NHWC"){
                if(current_tunable->tensor_layout != "nhwc")
                    {result_t result; result.return_code = -2; return result;}
            }else if(in_layout == "NCHWC"){
                if(current_tunable->tensor_layout.compare(0, 5, "nchwc") != 0)
                    {result_t result; result.return_code = -2; return result;}
                auto wei_layout_config = current_tunable->tensor_layout.substr(6);
                if((fil_layout == "NCHWC" && wei_layout_config != "kcyxc") || 
                    (fil_layout == "CHWNC" && wei_layout_config != "cyxkc"))
                    {result_t result; result.return_code = -2; return result;}
            }
        }

        std::string current_kernel_name = driver->get_kernel_name(current_tunable).c_str();
        std::string single_kernel_name = env_get_str("IGEMM_KVALID_TARGET", "");
        
        if(single_kernel_name == "" || single_kernel_name == current_kernel_name){
            printf("[%s:%2d] %s", direction.c_str(), index, driver->get_kernel_name(current_tunable).c_str());
            fflush(stdout);
            pre_func();
            result_t result = driver->run(conv_args, current_tunable, device_input, device_weight, device_output, current_gks);
            std::string gks_string = "";
            if(current_tunable->gemm_k_global_split){
                gks_string = "[" + std::to_string(result.gks) + "]";
            }
            printf("%s", gks_string.c_str());
            std::string tiling_string = get_tiling_string(driver, conv_args);
            printf("%s", tiling_string.c_str());

            printf(", ");
            fflush(stdout);

            if (result.return_code != 0){
                printf("not applicable\n");
                return result_t{};
            }
            double gflops = theo_conv_flop / (result.duration_ms * 1e6);
            printf("cost:%.3fms, tflops:%.3f(%.2f%%)", result.duration_ms,
                    gflops / 1000 , (gflops / theo_gpu_gflops) * 100);

            post_func();

            printf("\n");
            result.gflops = gflops;
            result.efficiency = (gflops / theo_gpu_gflops) * 100;

            if(dump_gmap)
                gmap_dump(conv_args, current_tunable, result.gks);
            return result;
        }
        else{
            return result_t{};
        }

        
    };

    auto need_skip_due_to_macro_tile_boundary = [&](const igemm_gtc_tunable_t * tunable){
        if(max_mpb != -1 && tunable->gemm_m_per_block > max_mpb)
            return true;
        if(max_npb != -1 && tunable->gemm_n_per_block > max_npb)
            return true;
        if(max_kpb != -1 && tunable->gemm_k_per_block > max_kpb)
            return true;
        return false;
    };

    driver->set_block_tile_boundary(max_mpb, max_npb, max_kpb, max_gks);
    result_t fastest_result;
    fastest_result.duration_ms = FLT_MAX;
    int fastest_id = -1;
    if(driver->driver_mode == driver_mode_normal){
        int unique_index = 0;
        std::vector<igemm_gtc_tunable_t> unique_tunables;
        for(int i=0; i<tunables.size(); i++){
            if(need_skip_due_to_macro_tile_boundary(&tunables[i]))
                continue;
            if(gks_iterative){
                if(tunables[i].gemm_k_global_split != 0){
                    std::vector<int> gks_list = driver->get_gks_list(conv_args, &tunables[i]);
                    for(int gks : gks_list){
                        result_t result = launch(&tunables[i], unique_index, gks);
                        if(result.return_code == -2) continue;
                        unique_tunables.push_back(tunables[i]);
                        unique_tunables.back().gemm_k_global_split = gks;
                        if(result.duration_ms < fastest_result.duration_ms){
                            fastest_result = result;
                            fastest_id = unique_index;
                        }
                        unique_index++;
                    }
                }else{
                    result_t result = launch(&tunables[i], unique_index, 0);
                    if(result.return_code == -2) continue;
                    unique_tunables.push_back(tunables[i]);
                    unique_tunables.back().gemm_k_global_split = 0;
                    if(result.duration_ms < fastest_result.duration_ms){
                        fastest_result = result;
                        fastest_id = unique_index;
                    }
                    unique_index++;
                }
            }
            else{
                result_t result = launch(&tunables[i], unique_index, -1);
                if(result.return_code == -2) continue;
                unique_tunables.push_back(tunables[i]);
                unique_tunables.back().gemm_k_global_split = result.gks;
                if(result.duration_ms < fastest_result.duration_ms){
                    fastest_result = result;
                    fastest_id = unique_index;
                }
                unique_index++;
            }
        }

        if(log_fastest_config){
            dump_arg(conv_args);
            if(fastest_id == -1)
                printf("  fastest: no suitable kernel\n");
            else{
                std::string kernel_name_mock = fastest_result.kernel_name;
                std::string gks_kernel_ending = "_gkgs";
                if(fastest_result.kernel_name.compare(fastest_result.kernel_name.length() - gks_kernel_ending.length(),
                                            gks_kernel_ending.length(), gks_kernel_ending) == 0){
                    kernel_name_mock += "[" + std::to_string(fastest_result.gks) + "]";
                }
                printf("  fastest: [%d]%s, cost:%.3fms, tflops:%.3f(%.2f%%)\n",
                    fastest_id,
                    kernel_name_mock.c_str(),
                    fastest_result.duration_ms,
                    fastest_result.gflops / 1000,
                    fastest_result.efficiency);
            }
        }
    }else if(driver->driver_mode == driver_mode_heuristic){
        igemm_gtc_tunable_t selected_tunable = driver->heuristic_select_kernel(conv_args);
        if(run_only_kernel != IGEMM_RUN_ONLY_KERNEL_DEFAULT)
            if(run_only_kernel != driver->get_kernel_name(&selected_tunable)){
                printf("heuristic selected tunable not match your request\n");
                return;
            }

        result_t result = launch(&selected_tunable, 0, -1);
        fastest_result = result;
        fastest_id = 0;
    }else{
        assert(0);
    }

    if(p_bcsv){
        fprintf(p_bcsv, "%.3f,%.3f,%.2f%%,%s,",
            fastest_result.duration_ms, fastest_result.gflops/1000, fastest_result.efficiency, fastest_result.kernel_name.c_str());
        std::string conv_cmd = log_cmd(conv_args, driver_data_type, direction);
        fprintf(p_bcsv, "%s\n", conv_cmd.c_str());
        fflush(p_bcsv);
    }

    if(sleep_ms != 0)
        usleep(1000 * sleep_ms);
}

int main(int argc, char **argv) {
    std::string hsaco = env_get_str("IGEMM_HSACO", IGEMM_HSACO);
    std::string config_file = env_get_str("IGEMM_CONFIG_FILE", IGEMM_CONFIG_FILE);
    std::string run_only_kernel = env_get_str("IGEMM_RUN_ONLY_KERNEL", IGEMM_RUN_ONLY_KERNEL_DEFAULT);
    int warmup = env_get_int("IGEMM_WARMUP", WARMUP);
    int repeat = env_get_int("IGEMM_REPEAT", REPEAT);
    int assert_when_invalid = env_get_int("IGEMM_ASSERT_WHEN_INVALID", 0);
    int verbose     = env_get_int("IGEMM_VERBOSE", 0);
    int igemm_rand_int = env_get_int("IGEMM_RAND_INT", 0);
    int igemm_bench_csv = env_get_int("IGEMM_BENCH_CSV", 0);
    driver_mode_t driver_mode = static_cast<driver_mode_t>(env_get_int("IGEMM_MODE", 0));
    config_parser_t config_parser(config_file);
    auto unexpanded_content = config_parser.parse();
    auto content = igemm_try_expand_tunable_content(unexpanded_content);
    //content.dump();
    FILE * p_bcsv = nullptr;
    if(igemm_bench_csv){
        p_bcsv = fopen ("bench_model.csv", "a");
        assert(p_bcsv);
    }

#ifdef USE_GPU_NAIVE_CONV
    std::string gpu_naive_conv_hsaco = env_get_str("IGEMM_GPU_NAIVE_CONV_HSACO", IGEMM_GPU_NAIVE_CONV_HSACO);
    gpu_naive_conv_init(gpu_naive_conv_hsaco.c_str());
#endif

    auto tunables = igemm_gtc_tunable_from_config(content);
    if(tunables.size() == 0){
        printf("no tunable specified, may not work\n");
        return 0;
    }
    // printf("tunables:%d, hsaco:%s\n", tunables.size(), hsaco);

    // COR-004 (2026-09-02): everything below this point (is_wmma_f16_acc,
    // is_wmma_bf16_acc, is_wmma_atomic_pack_bf16, and every buffer-allocation/
    // verification decision derived from them further down in main()/
    // launch_conv_driver) reads ONLY tunables[0]'s accumulate-width flags and applies
    // that single choice to every kernel in the vector -- a combined/master config file
    // that mixed e.g. a wmma_acc_f16 section with a plain fp32-accumulate section of the
    // same direction/precision would silently size and verify every OTHER tunable's
    // buffers against tunables[0]'s width instead of its own, corrupting results without
    // any diagnostic. Only meaningful for WMMA (XDLOPS/DLOPS/MAC never read these flags).
    if(tunables[0].fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA){
        for(size_t i = 1; i < tunables.size(); i++){
            assert(tunables[i].fma_type != IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA ||
                   (tunables[i].wmma_acc_f16 == tunables[0].wmma_acc_f16 &&
                    tunables[i].wmma_acc_bf16 == tunables[0].wmma_acc_bf16 &&
                    tunables[i].atomic_pack_bf16 == tunables[0].atomic_pack_bf16) &&
                   "all WMMA tunables in one config must share the same accumulate-width "
                   "flags (wmma_acc_f16/wmma_acc_bf16/atomic_pack_bf16) as tunables[0] -- "
                   "see COR-004: the driver's buffer sizing/verification below only reads "
                   "tunables[0]'s choice and applies it to every kernel in the vector");
        }
    }

    // The gfx1250 WMMA milestone kernels always accumulate/store the D operand at full
    // width (fp32, or int32 for int8 -- see e.g. igemm_fwd_gtc_wmma_nhwc_t's docstring)
    // regardless of tunable->precision, unlike the XDLOPS/DLOPS/MAC kernels below which
    // natively write "precision"-width output. The generic *_dtype buffers are sized by
    // `data_byte` (precision width) everywhere else in this file; for WMMA tunables the
    // buffer that plays the "kernel-native-width" role for whichever direction is under
    // test (output for fwd, input for bwd, weight for wrw) must instead be sized/read/
    // compared at 4 bytes/element, or a real WMMA kernel would silently overrun it.
    bool is_wmma = tunables.size() > 0 && tunables[0].fma_type == IGEMM_GTC_TUNABLE_FMA_TYPE_WMMA;
    // Phase 24: wmma_acc_f16 (fp16 only) makes the WMMA kernel's D operand genuinely 2
    // bytes/element instead of the usual 4 -- the "native role" buffer (output for fwd,
    // input for bwd, weight for wrw) needs its OWN width, separate from is_wmma's blanket
    // fp32 override below. The other two buffers stay ordinary fp16 either way, so no
    // special-casing needed for them (they were already correctly sized by data_byte, or
    // harmlessly over-allocated to fp32 by is_wmma's dtype_alloc_byte).
    bool is_wmma_f16_acc = is_wmma && tunables[0].wmma_acc_f16 != 0;
    // Phase 27: bf16 analog of is_wmma_f16_acc above -- same 2-byte-native-role-buffer
    // situation, just gated to wmma_acc_bf16.
    bool is_wmma_bf16_acc = is_wmma && tunables[0].wmma_acc_bf16 != 0;
    // Phase 34: wrw's gemm_k_global_split atomic epilogue, packed-bf16 variant -- same
    // 2-byte-native-role-buffer situation as is_wmma_f16_acc/is_wmma_bf16_acc above (the
    // kernel writes grad_weight directly at bf16 width, no fp32 workspace), just gated to
    // atomic_pack_bf16 instead of an accumulate-precision tunable.
    bool is_wmma_atomic_pack_bf16 = is_wmma && tunables[0].atomic_pack_bf16 != 0;

    hipModule_t module;
#ifndef IGEMM_SPLIT_KERNEL
    HIP_CALL(hipModuleLoad(&module, hsaco.c_str()));
#endif

    std::string base_arg = create_base_args(argc, argv);
    args_t conv_args = create_conv_args(argc, argv);
    // dump_arg(&conv_args);
    driverDataType_t driver_data_type;
    auto vec_found = base_arg.find("x");
    std::string base_type = base_arg.substr(0, vec_found);
    int vector_c = find_vector_c_from_base_arg(base_arg);
    vector_c = env_get_int("VECTOR_C", vector_c);

    if(base_type == "conv"){
        driver_data_type = driverFloat;
    }
    else if(base_type == "convfp16"){
        driver_data_type = driverHalf;
    }
    else if(base_type == "convbfp16") {
        driver_data_type = driverBFloat16;
    }
    else if(base_type == "convint8") {
        driver_data_type = driverInt8;
    }
    else if(base_type == "convint4") {
        driver_data_type = driverInt4;
    }
    else{
        printf("invalid base type:%s\n", base_type.c_str());
        exit(0);
    }

    size_t data_byte = get_data_byte(driver_data_type);

    int hi = conv_args.get_int("in_h");
    int wi = conv_args.get_int("in_w");
    int n = conv_args.get_int("batchsize");
    int k = conv_args.get_int("out_channels");
    int c = conv_args.get_int("in_channels");

    int stride_h = conv_args.get_int("conv_stride_h");
    int stride_w = conv_args.get_int("conv_stride_w");
    int dilation_h = conv_args.get_int("dilation_h");
    int dilation_w = conv_args.get_int("dilation_w");
    int pad_h = conv_args.get_int("pad_h");
    int pad_w = conv_args.get_int("pad_w");
    int y = conv_args.get_int("fil_h");
    int x = conv_args.get_int("fil_w");
    int ngroups = conv_args.get_int("group_count");
    int ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
    int wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);
    int forw = conv_args.get_int("forw");
    std::string in_layout = conv_args.get_str("in_layout");
    std::string out_layout = conv_args.get_str("out_layout");
    std::string fil_layout = conv_args.get_str("fil_layout");

    int need_fwd = (forw == 0 ? 1 : (forw & 1 ? 1 : 0));
    int need_bwd = (forw == 0 ? 1 : (forw & 2 ? 1 : 0));
    int need_wrw = (forw == 0 ? 1 : (forw & 4 ? 1 : 0));

    //assert(in_layout == out_layout && in_layout == fil_layout); // currently only support all layout is the same
    assert(in_layout == out_layout); 
    assert(in_layout == "NCHW" || in_layout == "NHWC" || in_layout == "NCHWC");
    assert((in_layout == "NCHW" && tunables[0].tensor_layout == "nchw") || 
           (in_layout == "NHWC" && tunables[0].tensor_layout == "nhwc") ||
           (in_layout == "NCHWC" && tunables[0].tensor_layout.compare(0, 5, "nchwc") == 0));

    float *host_input = (float *)malloc(static_cast<size_t>(n) * c * hi * wi * sizeof(float));
    float *host_weight = (float *)malloc(static_cast<size_t>(k) * c * y * x * sizeof(float));
    float *host_output = (float *)malloc(static_cast<size_t>(n) * k * ho * wo * sizeof(float));

    float *device_input;
    float *device_weight;
    float *device_output;

    HIP_CALL(hipMalloc(&device_input, static_cast<size_t>(n) * c * hi * wi * sizeof(float)));
    HIP_CALL(hipMalloc(&device_weight, static_cast<size_t>(k) * c * y * x * sizeof(float)));
    HIP_CALL(hipMalloc(&device_output, static_cast<size_t>(n) * k * ho * wo * sizeof(float)));

    void *host_input_dtype;
    void *host_weight_dtype;
    void *host_output_dtype;

    void *device_input_dtype;
    void *device_weight_dtype;
    void *device_output_dtype;
    // over-allocate to fp32 width for WMMA (see is_wmma comment above) -- harmless for the
    // roles that stay native-precision, since only the actually-used prefix of each buffer
    // is ever read/written.
    // Phase 24: when wmma_acc_f16 is active, the "native role" buffer is genuinely 2
    // bytes/element (not 4) -- and since the OTHER two buffers are ordinary fp16 anyway
    // (data_byte==sizeof(half) for fp16 tests), sizing ALL THREE at data_byte is correct
    // for every one of them, not just an accepted over-allocation like the plain is_wmma
    // case below. Declared unconditionally (not inside the #if below) since fwd_pre/bwd_pre/
    // wrw_pre further down reference it regardless of which USE_* macro (if any) is defined
    // -- e.g. a pure fp32 build defines none of them, and previously (a Phase 24 regression,
    // caught by this branch's own byte-identical-assembly regression sweep) those lambdas
    // failed to compile with "use of undeclared identifier 'dtype_alloc_byte'".
    size_t dtype_alloc_byte = (is_wmma_f16_acc || is_wmma_bf16_acc || is_wmma_atomic_pack_bf16) ? data_byte : (is_wmma ? sizeof(float) : data_byte);
#if defined(USE_HALF) || defined(USE_INT8) || defined(USE_BF16) || defined(USE_INT4)
    host_input_dtype  = malloc(n * c * hi * wi * dtype_alloc_byte);
    host_weight_dtype = malloc(k * c * y * x * dtype_alloc_byte);
    host_output_dtype = malloc(n * k * ho * wo * dtype_alloc_byte);

    HIP_CALL(hipMalloc(&device_input_dtype, n * c * hi * wi * dtype_alloc_byte));
    HIP_CALL(hipMalloc(&device_weight_dtype, k * c * y * x * dtype_alloc_byte));
    HIP_CALL(hipMalloc(&device_output_dtype, n * k * ho * wo * dtype_alloc_byte));
#endif


    int need_verify = conv_args.get_int("verify");
    if(p_bcsv){
        fprintf(p_bcsv, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,",
                         n,  c, hi, wi,  k,  y,  x, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, ngroups);
        fprintf(p_bcsv, "%.2f,", get_theoritical_conv_flop(&conv_args)/1e9);
        fflush(p_bcsv);
    }

    if(driver_data_type == driverInt8 || driver_data_type == driverInt4)
        igemm_rand_int = 1;

    hipModule_t module_tensor_cast;
    std::string hsaco_tensor_cast = env_get_str("IGEMM_TENSOR_CAST_HSACO", IGEMM_TENSOR_CAST_HSACO);
    HIP_CALL(hipModuleLoad(&module_tensor_cast, hsaco_tensor_cast.c_str()));

    if (need_fwd){
        int fastest_id = -1;
        void *device_output_to_host = NULL;
        // gen rand -- always initialize inputs, even when not verifying (-V 0): WMMA is
        // measured ~2x faster on all-zero operands than on random data (COR-002), so
        // leaving inputs uninitialized under -V 0 silently invalidates timing results.
        if(!igemm_rand_int){
            gen_rand_vector<float, float>(host_input, static_cast<size_t>(n) * c * hi * wi, 0.0, 1.0);
            gen_rand_vector<float, float>(host_weight, static_cast<size_t>(k) * c * y * x, -0.5, 0.5);
        }else{
            gen_rand_vector<float, int>(host_input, static_cast<size_t>(n) * c * hi * wi, 1, 1);
            gen_rand_vector<float, int>(host_weight, static_cast<size_t>(k) * c * y * x, 1, 1);
        }
        if(driver_data_type == driverHalf){
            tensor_copy<float16, float>(static_cast<float16*>(host_input_dtype), host_input, static_cast<size_t>(n) * c * hi * wi);
            tensor_copy<float16, float>(static_cast<float16*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * y * x);
        }
        else if(driver_data_type == driverBFloat16){
            tensor_copy<bfloat16, float>(static_cast<bfloat16*>(host_input_dtype), host_input, static_cast<size_t>(n) * c * hi * wi);
            tensor_copy<bfloat16, float>(static_cast<bfloat16*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * y * x);
        }
        else if(driver_data_type == driverInt8){
            tensor_copy<int8_t, float>(static_cast<int8_t*>(host_input_dtype), host_input, static_cast<size_t>(n) * c * hi * wi);
            tensor_copy<int8_t, float>(static_cast<int8_t*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * y * x);
        }
        else if(driver_data_type == driverInt4)
        {
            tensor_copy<int4x2_t, float>(static_cast<int4x2_t*>(host_input_dtype), host_input, static_cast<size_t>(n) * c * hi * wi);
            tensor_copy<int4x2_t, float>(static_cast<int4x2_t*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * y * x);
        }

        if (need_verify) {
#ifdef USE_GPU_NAIVE_CONV
            HIP_CALL(hipMemcpy(device_input, host_input,
                       static_cast<size_t>(n) * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight, host_weight,
                       static_cast<size_t>(k) * c * y * x * sizeof(float), hipMemcpyHostToDevice));

            if(in_layout == "NCHW")
                gpu_naive_conv_fwd_nchw_fp32(device_input, device_weight, device_output,
                                n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            else if(in_layout == "NHWC")
                gpu_naive_conv_fwd_nhwc_fp32(device_input, device_weight, device_output,
                                n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            else if(in_layout == "NCHWC"){
                if(((c / ngroups) % vector_c != 0) || ((k / ngroups) % vector_c != 0)){
                    dump_arg(&conv_args);
                    printf("can't support c:%d k:%d with vec_c:%d\n", c, k, vector_c);
                    if(p_bcsv)
                    {
                        fprintf(p_bcsv, "\n");
                        fflush(p_bcsv);
                    }
                    exit(-1);
                }
                float* aux_in = (float*)malloc(static_cast<size_t>(n) * c * hi * wi * sizeof(float));
                float* aux_wei = (float*)malloc(static_cast<size_t>(k) * c * y * x * sizeof(float));
                float* aux_out = (float*)malloc(static_cast<size_t>(n) * k * ho * wo * sizeof(float));

                tensor_transpose_nchwc_2_nchw<float*>(aux_in, host_input, n, c, hi, wi, vector_c);
                for(int i_groups = 0; i_groups < ngroups; i_groups++){
                    int group_offset = i_groups * (k / ngroups) * (c / ngroups) * y * x;
                    if(fil_layout == "CHWNC")
                        tensor_transpose_chwnc_2_nchw<float*>(aux_wei + group_offset, host_weight + group_offset, k / ngroups, c / ngroups, y, x, vector_c);
                    else if(fil_layout == "NCHWC")
                        tensor_transpose_nchwc_2_nchw<float*>(aux_wei + group_offset, host_weight + group_offset, k / ngroups, c / ngroups, y, x, vector_c);
                }

                if(env_get_int("IGEMM_CHECK_TRNASPOSE", 0)){
                    float* aux_wei_check = (float*)malloc(static_cast<size_t>(k) * c * y * x * sizeof(float));
                    float* aux_in_check = (float*)malloc(static_cast<size_t>(n) * c * hi * wi * sizeof(float));

                    tensor_transpose_nchw_2_nchwc<float*>(aux_in_check, aux_in, n, c, hi, wi, vector_c);
                    if(fil_layout == "CHWNC")
                        tensor_transpose_nchw_2_chwnc<float*>(aux_wei_check, aux_wei, k, c, y, x, vector_c);
                    else if(fil_layout == "NCHWC")
                        tensor_transpose_nchw_2_nchwc<float*>(aux_wei_check, aux_wei, k, c, y, x, vector_c);

                    double transpose_nrms = get_nrms("fwd", driver_data_type);
                    valid_vector<float>(host_input, aux_in_check, static_cast<size_t>(n) * c * hi * wi, transpose_nrms);
                    valid_vector<float>(host_weight, aux_wei_check, static_cast<size_t>(k) * c * y * x, transpose_nrms);

                    free(aux_in_check);
                    free(aux_wei_check);
                }
                
                HIP_CALL(hipMemcpy(device_input, aux_in,
                       static_cast<size_t>(n) * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
                HIP_CALL(hipMemcpy(device_weight, aux_wei,
                       static_cast<size_t>(k) * c * y * x * sizeof(float), hipMemcpyHostToDevice));

                gpu_naive_conv_fwd_nchw_fp32(device_input, device_weight, device_output,
                        n, wi, hi, c,
                        k, x, y, pad_w, pad_h, stride_w, stride_h,
                        dilation_w, dilation_h, ngroups);

                HIP_CALL(hipMemcpy(host_output, device_output,
                                   static_cast<size_t>(n) * k * ho * wo * sizeof(float),
                                   hipMemcpyDeviceToHost));

                tensor_transpose_nchw_2_nchwc<float*>(aux_out, host_output, n, k, ho, wo, vector_c);

                HIP_CALL(hipMemcpy(device_input, host_input,
                       static_cast<size_t>(n) * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
                HIP_CALL(hipMemcpy(device_weight, host_weight,
                       static_cast<size_t>(k) * c * y * x * sizeof(float), hipMemcpyHostToDevice));

                HIP_CALL(hipMemcpy(device_output, aux_out,
                       static_cast<size_t>(n) * k * ho * wo * sizeof(float), hipMemcpyHostToDevice));

                free(aux_in);
                free(aux_wei);
                free(aux_out);
                // exit(1);
            }
            else
                assert(0);
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipMemcpy(host_output, device_output,
                                   static_cast<size_t>(n) * k * ho * wo * sizeof(float),
                                   hipMemcpyDeviceToHost));
#else
            if(in_layout == "NCHW")
                naive_conv_fwd_nchw(host_input, host_weight, host_output, n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            else if(in_layout == "NHWC")
                naive_conv_fwd_nhwc(host_input, host_weight, host_output, n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            else
                assert(0);
#endif
            if(driver_data_type != driverHalf && !is_wmma){
                device_output_to_host = malloc((static_cast<size_t>(n) * k * ho * wo * data_byte + 3) / 4 * 4);
            }
            else{
                device_output_to_host = malloc(static_cast<size_t>(n) * k * ho * wo * sizeof(float));
            }
        }

        if(driver_data_type == driverFloat)
        {
            HIP_CALL(hipMemcpy(device_input, host_input,
                        static_cast<size_t>(n) * c * hi * wi * data_byte, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight, host_weight,
                        static_cast<size_t>(k) * c * y * x * data_byte, hipMemcpyHostToDevice));
        }
        else if(driver_data_type == driverInt4)
        {
            HIP_CALL(hipMemcpy(device_input_dtype, host_input_dtype,
                        static_cast<size_t>(n) * c * hi * wi / 2, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight_dtype, host_weight_dtype,
                        static_cast<size_t>(k) * c * y * x / 2, hipMemcpyHostToDevice));
        }
        else
        {
            HIP_CALL(hipMemcpy(device_input_dtype, host_input_dtype,
                        static_cast<size_t>(n) * c * hi * wi * data_byte, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight_dtype, host_weight_dtype,
                        static_cast<size_t>(k) * c * y * x * data_byte, hipMemcpyHostToDevice));
        }

        igemm_fwd_gtc_t conv_fwd_driver(module_tensor_cast, module, driver_mode, driver_data_type, warmup, repeat, verbose);
        conv_fwd_driver.set_vector_c(vector_c);

        auto fwd_pre = [&](){
            if (need_verify)
                HIP_CALL(hipMemset(driver_data_type == driverFloat ? device_output : device_output_dtype,
                    0, static_cast<size_t>(n) * k * ho * wo * dtype_alloc_byte));
        };

        auto fwd_post = [&](){
            if (need_verify) {
                double nrms = get_nrms("fwd", driver_data_type);
                bool is_valid = false;
                if(driver_data_type == driverFloat){
                    HIP_CALL(hipMemcpy(device_output_to_host, device_output,
                                   static_cast<size_t>(n) * k * ho * wo * data_byte,
                                   hipMemcpyDeviceToHost));
                    is_valid = valid_vector<float>(host_output, static_cast<float*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * ho * wo, nrms);
                }else if(is_wmma_f16_acc){
                    // Phase 24: device_output_dtype holds the WMMA kernel's genuinely
                    // 2-byte/element (packed-then-unpacked, see coalescing_store_wmma.py)
                    // fp16 output -- expand it to fp32 via the new tensor_cast kernel
                    // (reusing device_output, free by this point -- its data was already
                    // copied out to host_output for the reference comparison above) so the
                    // existing valid_vector<float> comparison logic runs unchanged.
                    hipFunction_t f16acc_cast_func;
                    HIP_CALL(hipModuleGetFunction(&f16acc_cast_func, module_tensor_cast, "tensor_cast_fp32_fp16acc_1d"));
                    tensor_cast_karg_t karg_f16acc_cast;
                    karg_f16acc_cast.output = device_output;
                    karg_f16acc_cast.input = device_output_dtype;
                    karg_f16acc_cast.total_length = n * k * ho * wo;
                    const size_t thread_length_f16acc_cast = (static_cast<size_t>(n) * k * ho * wo + 8 * 256) / (8 * 256) * (8 * 256) / 8;
                    igemm_launch_kernel_single(f16acc_cast_func, &karg_f16acc_cast, sizeof(karg_f16acc_cast), {thread_length_f16acc_cast, 1, 1}, {256, 1, 1});
                    HIP_CALL(hipMemcpy(device_output_to_host, device_output,
                                   static_cast<size_t>(n) * k * ho * wo * sizeof(float),
                                   hipMemcpyDeviceToHost));
                    is_valid = valid_vector<float>(host_output, static_cast<float*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * ho * wo, nrms);
                }else if(is_wmma_bf16_acc){
                    // Phase 27: bf16 analog of the is_wmma_f16_acc branch above.
                    hipFunction_t bf16acc_cast_func;
                    HIP_CALL(hipModuleGetFunction(&bf16acc_cast_func, module_tensor_cast, "tensor_cast_fp32_bf16acc_1d"));
                    tensor_cast_karg_t karg_bf16acc_cast;
                    karg_bf16acc_cast.output = device_output;
                    karg_bf16acc_cast.input = device_output_dtype;
                    karg_bf16acc_cast.total_length = n * k * ho * wo;
                    const size_t thread_length_bf16acc_cast = (static_cast<size_t>(n) * k * ho * wo + 8 * 256) / (8 * 256) * (8 * 256) / 8;
                    igemm_launch_kernel_single(bf16acc_cast_func, &karg_bf16acc_cast, sizeof(karg_bf16acc_cast), {thread_length_bf16acc_cast, 1, 1}, {256, 1, 1});
                    HIP_CALL(hipMemcpy(device_output_to_host, device_output,
                                   static_cast<size_t>(n) * k * ho * wo * sizeof(float),
                                   hipMemcpyDeviceToHost));
                    is_valid = valid_vector<float>(host_output, static_cast<float*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * ho * wo, nrms);
                }else if(is_wmma){
                    // WMMA always accumulates/stores D at full width (fp32, or int32 for
                    // int8) regardless of tunable->precision -- see comment near is_wmma.
                    HIP_CALL(hipMemcpy(device_output_to_host, device_output_dtype,
                                   static_cast<size_t>(n) * k * ho * wo * sizeof(float),
                                   hipMemcpyDeviceToHost));
                    if(driver_data_type == driverInt8)
                        is_valid = valid_vector<int32_t>(host_output, static_cast<int32_t*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * ho * wo, nrms);
                    else
                        is_valid = valid_vector<float>(host_output, static_cast<float*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * ho * wo, nrms);
                }else{
                    HIP_CALL(hipMemcpy(device_output_to_host, device_output_dtype,
                                   static_cast<size_t>(n) * k * ho * wo * data_byte,
                                   hipMemcpyDeviceToHost));
                    if(driver_data_type == driverHalf)
                        is_valid = valid_vector<float16>(host_output, static_cast<float16*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * ho * wo, nrms);
                    else if(driver_data_type == driverBFloat16)
                        is_valid = valid_vector<bfloat16>(host_output, static_cast<bfloat16*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * ho * wo, nrms);
                    else if (driver_data_type == driverInt8)
                        is_valid = valid_vector<int8_t>(host_output, static_cast<int8_t*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * ho * wo, nrms);
                    else if (driver_data_type == driverInt4)
                        is_valid = valid_vector<int4x2_t>(host_output, static_cast<int4x2_t*>(device_output_to_host),
                                            static_cast<size_t>(n) * k * ho * wo, nrms);
                }
                printf(", valid:%s", is_valid ? "y" : "n");
                if(assert_when_invalid) assert(is_valid);
            }
        };

        if(driver_data_type == driverFloat)
            launch_conv_driver(&conv_fwd_driver, &conv_args, tunables, "fwd", driver_data_type, p_bcsv, device_input, device_weight, device_output, fwd_pre, fwd_post);
        else
            launch_conv_driver(&conv_fwd_driver, &conv_args, tunables, "fwd", driver_data_type, p_bcsv, device_input_dtype, device_weight_dtype, device_output_dtype, fwd_pre, fwd_post);

        if (need_verify)
            free(device_output_to_host);
    }

    if (need_bwd){
        void *device_input_to_host = NULL;
        result_t fastest_result_bwd;
        fastest_result_bwd.duration_ms = FLT_MAX;
        int fastest_id = -1;
        // gen rand -- always initialize inputs, even when not verifying (-V 0): WMMA is
        // measured ~2x faster on all-zero operands than on random data (COR-002), so
        // leaving inputs uninitialized under -V 0 silently invalidates timing results.
        if(!igemm_rand_int){
            gen_rand_vector<float, float>(host_output, static_cast<size_t>(n) * k * ho * wo, 0.0, 1.0);
            gen_rand_vector<float, float>(host_weight, static_cast<size_t>(k) * c * y * x, -0.5, 0.5);
        }
        else{
            gen_rand_vector<float, int>(host_output, static_cast<size_t>(n) * k * ho * wo, -5, 5);
            gen_rand_vector<float, int>(host_weight, static_cast<size_t>(k) * c * y * x, -5, 5);
        }
        gen_rand_vector<float, float>(host_input, static_cast<size_t>(n) * c * hi * wi, 999999., 9999999.);  // manually poison input value to a very large number

        if(driver_data_type == driverHalf){
            tensor_copy<float16, float>(static_cast<float16*>(host_output_dtype), host_output, static_cast<size_t>(n) * k * ho * wo);
            tensor_copy<float16, float>(static_cast<float16*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * y * x);
        }
        else if(driver_data_type == driverBFloat16){
            tensor_copy<bfloat16, float>(static_cast<bfloat16*>(host_output_dtype), host_output, static_cast<size_t>(n) * k * ho * wo);
            tensor_copy<bfloat16, float>(static_cast<bfloat16*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * y * x);
        }
        else if(driver_data_type == driverInt8){
            tensor_copy<int8_t, float>(static_cast<int8_t*>(host_output_dtype), host_output, static_cast<size_t>(n) * k * ho * wo);
            tensor_copy<int8_t, float>(static_cast<int8_t*>(host_weight_dtype), host_weight, static_cast<size_t>(k) * c * y * x);
        }

        if (need_verify) {
#ifdef USE_GPU_NAIVE_CONV
            HIP_CALL(hipMemcpy(device_output, host_output,
                       static_cast<size_t>(n) * k * ho * wo * sizeof(float), hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight, host_weight,
                       static_cast<size_t>(k) * c * y * x * sizeof(float), hipMemcpyHostToDevice));
            if(in_layout == "NCHW")
                gpu_naive_conv_bwd_nchw_fp32(device_input, device_weight, device_output,
                                n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            else if(in_layout == "NHWC")
                gpu_naive_conv_bwd_nhwc_fp32(device_input, device_weight, device_output,
                                n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            else
                assert(0);
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipMemcpy(host_input, device_input,
                                   static_cast<size_t>(n) * c * hi * wi * sizeof(float),
                                   hipMemcpyDeviceToHost));
#else
            if(in_layout == "NCHW")
                naive_conv_bwd_nchw(host_input, host_weight, host_output, n,
                                         wi, hi, c, k, x, y, pad_w,
                                         pad_h, stride_w, stride_h, dilation_w, dilation_h, ngroups);
            else if(in_layout == "NHWC")
                naive_conv_bwd_nhwc(host_input, host_weight, host_output, n,
                                         wi, hi, c, k, x, y, pad_w,
                                         pad_h, stride_w, stride_h, dilation_w, dilation_h, ngroups);
            else
                assert(0);
#endif
            if(driver_data_type != driverFloat && !is_wmma){
                device_input_to_host = malloc((static_cast<size_t>(n) * c * hi * wi * data_byte + 3) / 4 * 4 );
            }
            else{
                device_input_to_host = malloc(static_cast<size_t>(n) * c * hi * wi * sizeof(float));
            }
            // printf("len:%d\n", n * c * hi * wi * sizeof(float) );
        }

        if(driver_data_type == driverFloat){
            HIP_CALL(hipMemcpy(device_output, host_output,
                        static_cast<size_t>(n) * k * ho * wo * data_byte, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight, host_weight,
                        static_cast<size_t>(k) * c * y * x * data_byte, hipMemcpyHostToDevice));
        }else{
            HIP_CALL(hipMemcpy(device_output_dtype, host_output_dtype,
                        static_cast<size_t>(n) * k * ho * wo * data_byte, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_weight_dtype, host_weight_dtype,
                        static_cast<size_t>(k) * c * y * x * data_byte, hipMemcpyHostToDevice));
        }

        igemm_bwd_gtc_t conv_bwd_driver(module_tensor_cast, module, driver_mode, driver_data_type, warmup, repeat, verbose);
        conv_bwd_driver.set_vector_c(vector_c);

        auto bwd_pre = [&](){
            if (need_verify)
                HIP_CALL(hipMemset(driver_data_type == driverFloat ? device_input : device_input_dtype,
                    0x7f, static_cast<size_t>(n) * c * hi * wi * dtype_alloc_byte)); // 0x7f7f7f7f ~= 7.41e+28, a very large number
        };

        auto bwd_post = [&](){
            if (need_verify) {
                double nrms = get_nrms("bwd", driver_data_type);
                bool is_valid = false;
                if(driver_data_type == driverFloat){
                    HIP_CALL(hipMemcpy(device_input_to_host, device_input,
                                    static_cast<size_t>(n) * c * hi * wi * data_byte,
                                    hipMemcpyDeviceToHost));
                    is_valid = valid_vector<float>(host_input, static_cast<float*>(device_input_to_host),
                                                static_cast<size_t>(n) * c * hi * wi, nrms);
                } else if(is_wmma_f16_acc){
                    // Phase 24: see the equivalent fwd comment near is_wmma_f16_acc.
                    hipFunction_t f16acc_cast_func;
                    HIP_CALL(hipModuleGetFunction(&f16acc_cast_func, module_tensor_cast, "tensor_cast_fp32_fp16acc_1d"));
                    tensor_cast_karg_t karg_f16acc_cast;
                    karg_f16acc_cast.output = device_input;
                    karg_f16acc_cast.input = device_input_dtype;
                    karg_f16acc_cast.total_length = n * c * hi * wi;
                    const size_t thread_length_f16acc_cast = (static_cast<size_t>(n) * c * hi * wi + 8 * 256) / (8 * 256) * (8 * 256) / 8;
                    igemm_launch_kernel_single(f16acc_cast_func, &karg_f16acc_cast, sizeof(karg_f16acc_cast), {thread_length_f16acc_cast, 1, 1}, {256, 1, 1});
                    HIP_CALL(hipMemcpy(device_input_to_host, device_input,
                                    static_cast<size_t>(n) * c * hi * wi * sizeof(float),
                                    hipMemcpyDeviceToHost));
                    is_valid = valid_vector<float>(host_input, static_cast<float*>(device_input_to_host),
                                                static_cast<size_t>(n) * c * hi * wi, nrms);
                } else if(is_wmma_bf16_acc){
                    // Phase 27: bf16 analog of the is_wmma_f16_acc branch above.
                    hipFunction_t bf16acc_cast_func;
                    HIP_CALL(hipModuleGetFunction(&bf16acc_cast_func, module_tensor_cast, "tensor_cast_fp32_bf16acc_1d"));
                    tensor_cast_karg_t karg_bf16acc_cast;
                    karg_bf16acc_cast.output = device_input;
                    karg_bf16acc_cast.input = device_input_dtype;
                    karg_bf16acc_cast.total_length = n * c * hi * wi;
                    const size_t thread_length_bf16acc_cast = (static_cast<size_t>(n) * c * hi * wi + 8 * 256) / (8 * 256) * (8 * 256) / 8;
                    igemm_launch_kernel_single(bf16acc_cast_func, &karg_bf16acc_cast, sizeof(karg_bf16acc_cast), {thread_length_bf16acc_cast, 1, 1}, {256, 1, 1});
                    HIP_CALL(hipMemcpy(device_input_to_host, device_input,
                                    static_cast<size_t>(n) * c * hi * wi * sizeof(float),
                                    hipMemcpyDeviceToHost));
                    is_valid = valid_vector<float>(host_input, static_cast<float*>(device_input_to_host),
                                                static_cast<size_t>(n) * c * hi * wi, nrms);
                } else if(is_wmma){
                    // WMMA always writes grad_input at full width (fp32, or int32 for int8).
                    HIP_CALL(hipMemcpy(device_input_to_host, device_input_dtype,
                                    static_cast<size_t>(n) * c * hi * wi * sizeof(float),
                                    hipMemcpyDeviceToHost));
                    if(driver_data_type == driverInt8)
                        is_valid = valid_vector<int32_t>(host_input, static_cast<int32_t*>(device_input_to_host),
                                                static_cast<size_t>(n) * c * hi * wi, nrms);
                    else
                        is_valid = valid_vector<float>(host_input, static_cast<float*>(device_input_to_host),
                                                static_cast<size_t>(n) * c * hi * wi, nrms);
                } else {
                    HIP_CALL(hipMemcpy(device_input_to_host, device_input_dtype,
                                    static_cast<size_t>(n) * c * hi * wi * data_byte,
                                    hipMemcpyDeviceToHost));
                    if(driver_data_type == driverHalf)
                        is_valid = valid_vector<float16>(host_input, static_cast<float16*>(device_input_to_host),
                                                static_cast<size_t>(n) * c * hi * wi, nrms);
                    else if (driver_data_type == driverBFloat16)
                        is_valid = valid_vector<bfloat16>(host_input, static_cast<bfloat16*>(device_input_to_host),
                                                static_cast<size_t>(n) * c * hi * wi, nrms);
                    else if (driver_data_type == driverInt8)
                        is_valid = valid_vector<int8_t>(host_input, static_cast<int8_t*>(device_input_to_host),
                                                static_cast<size_t>(n) * c * hi * wi, nrms);
                }
                printf(", valid:%s", is_valid ? "y" : "n");
                if(assert_when_invalid) assert(is_valid);
            }
        };

        if(driver_data_type == driverFloat)
            launch_conv_driver(&conv_bwd_driver, &conv_args, tunables, "bwd",  driver_data_type, p_bcsv, device_input, device_weight, device_output, bwd_pre, bwd_post);
        else
            launch_conv_driver(&conv_bwd_driver, &conv_args, tunables, "bwd",  driver_data_type, p_bcsv, device_input_dtype, device_weight_dtype, device_output_dtype, bwd_pre, bwd_post);

        if (need_verify) 
            free(device_input_to_host);
    }

    if (need_wrw){
        void *device_weight_to_host = NULL;

        // gen rand -- always initialize inputs, even when not verifying (-V 0): WMMA is
        // measured ~2x faster on all-zero operands than on random data (COR-002), so
        // leaving inputs uninitialized under -V 0 silently invalidates timing results.
        if(!igemm_rand_int){
            gen_rand_vector<float, float>(host_input, static_cast<size_t>(n) * c * hi * wi, 0.0, 1.0);
            gen_rand_vector<float, float>(host_output, static_cast<size_t>(n) * k * ho * wo, -0.5, 0.5);
        }else{
            gen_rand_vector<float, int>(host_input, static_cast<size_t>(n) * c * hi * wi, -5, 5);
            gen_rand_vector<float, int>(host_output, static_cast<size_t>(n) * k * ho * wo, -5, 5);
        }
        if(driver_data_type == driverHalf){
            tensor_copy<float16, float>(static_cast<float16*>(host_input_dtype), host_input, static_cast<size_t>(n) * c * hi * wi);
            tensor_copy<float16, float>(static_cast<float16*>(host_output_dtype), host_output, static_cast<size_t>(n) * k * ho * wo);
        }
        else if(driver_data_type == driverBFloat16){
            tensor_copy<bfloat16, float>(static_cast<bfloat16*>(host_input_dtype), host_input, static_cast<size_t>(n) * c * hi * wi);
            tensor_copy<bfloat16, float>(static_cast<bfloat16*>(host_output_dtype), host_output, static_cast<size_t>(n) * k * ho * wo);
        }
        else if(driver_data_type == driverInt8){
            tensor_copy<int8_t, float>(static_cast<int8_t*>(host_input_dtype), host_input, static_cast<size_t>(n) * c * hi * wi);
            tensor_copy<int8_t, float>(static_cast<int8_t*>(host_output_dtype), host_output, static_cast<size_t>(n) * k * ho * wo);
        }

        if (need_verify) {
#ifdef USE_GPU_NAIVE_CONV
            HIP_CALL(hipMemcpy(device_input, host_input,
                       static_cast<size_t>(n) * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_output, host_output,
                       static_cast<size_t>(n) * k * ho * wo * sizeof(float), hipMemcpyHostToDevice));
            if(in_layout == "NCHW")
                gpu_naive_conv_wrw_nchw_fp32(device_input, device_weight, device_output,
                                n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            else if(in_layout == "NHWC")
                gpu_naive_conv_wrw_nhwc_fp32(device_input, device_weight, device_output,
                                n, wi, hi, c,
                                k, x, y, pad_w, pad_h, stride_w, stride_h,
                                dilation_w, dilation_h, ngroups);
            else
                assert(0);
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipMemcpy(host_weight, device_weight,
                                   static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x * sizeof(float),
                                   hipMemcpyDeviceToHost));
#else
            if(in_layout == "NCHW")
                naive_conv_wrw_nchw(host_input, host_weight, host_output, n,
                                         wi, hi, c, k, x, y, pad_w,
                                         pad_h, stride_w, stride_h, dilation_w, dilation_h, ngroups);
            else if(in_layout == "NHWC")
                naive_conv_wrw_nhwc(host_input, host_weight, host_output, n,
                                         wi, hi, c, k, x, y, pad_w,
                                         pad_h, stride_w, stride_h, dilation_w, dilation_h, ngroups);
            else
                assert(0);
#endif
            if(driver_data_type == driverHalf && !is_wmma){
                device_weight_to_host = malloc((static_cast<size_t>(k) * c * y * x * data_byte + 3) / 4 * 4);
            }
            else{
                device_weight_to_host = malloc(static_cast<size_t>(k) * c * y * x * sizeof(float));
            }
        }

        if(driver_data_type == driverFloat){
            HIP_CALL(hipMemcpy(device_input, host_input,
                       static_cast<size_t>(n) * c * hi * wi * sizeof(float), hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_output, host_output,
                       static_cast<size_t>(n) * k * ho * wo * sizeof(float), hipMemcpyHostToDevice));
        }else{
            HIP_CALL(hipMemcpy(device_input_dtype, host_input_dtype,
                        static_cast<size_t>(n) * c * hi * wi * data_byte, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpy(device_output_dtype, host_output_dtype,
                        static_cast<size_t>(n) * k * ho * wo * data_byte, hipMemcpyHostToDevice));
        }

#if 0
        printf("input\r\n");
        for (int i_check = 0; i_check < (0+32); i_check++)
        {
            printf("[%d]th var to monitor:[%f, %d]\r\n", i_check*hi*wi, host_input[i_check*hi*wi], ((int *)host_input)[i_check*hi*wi]);
        }
        printf("output\r\n");
        for (int i_check = 0; i_check < (0+32); i_check++)
        {
            printf("[%d]th var to monitor:[%f, %d]\r\n", i_check*ho*wo, host_output[i_check*ho*wo], ((int *)host_output)[i_check*ho*wo]);
        }
        printf("input\r\n");
        for (int i_check = 0; i_check < (0+32); i_check++)
        {
            printf("[%d]th var to monitor:[%f, %d]\r\n", i_check, host_input[i_check], ((int *)host_input)[i_check]);
        }
        printf("output\r\n");
        for (int i_check = 0; i_check < (0+32); i_check++)
        {
            printf("[%d]th var to monitor:[%f, %d]\r\n", i_check, host_output[i_check], ((int *)host_output)[i_check]);
        }
        printf("workspace debug end \r\n");
#endif   


        igemm_wrw_gtc_t conv_wrw_driver(module_tensor_cast, module, driver_mode, driver_data_type, warmup, repeat, verbose);
        conv_wrw_driver.set_vector_c(vector_c);
        
        auto wrw_pre = [&](){
            if (need_verify)
                HIP_CALL(hipMemset(driver_data_type == driverFloat ? device_weight : device_weight_dtype,
                    0, static_cast<size_t>(k) * c * y * x * dtype_alloc_byte));
        };

        auto wrw_post = [&](){
            if (need_verify) {
                double nrms = get_nrms("wrw", driver_data_type);
                bool is_valid;
                if(driver_data_type == driverFloat){
                    HIP_CALL(hipMemcpy(device_weight_to_host, device_weight,
                                   static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x * sizeof(float),
                                   hipMemcpyDeviceToHost));
                    is_valid = valid_vector<float>(host_weight, static_cast<float*>(device_weight_to_host),
                                    static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x, nrms);
                }else if(is_wmma_f16_acc){
                    // Phase 24: see the equivalent fwd comment near is_wmma_f16_acc.
                    hipFunction_t f16acc_cast_func;
                    HIP_CALL(hipModuleGetFunction(&f16acc_cast_func, module_tensor_cast, "tensor_cast_fp32_fp16acc_1d"));
                    tensor_cast_karg_t karg_f16acc_cast;
                    karg_f16acc_cast.output = device_weight;
                    karg_f16acc_cast.input = device_weight_dtype;
                    karg_f16acc_cast.total_length = ngroups * (k / ngroups) * (c / ngroups) * y * x;
                    const size_t thread_length_f16acc_cast = (static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x + 8 * 256) / (8 * 256) * (8 * 256) / 8;
                    igemm_launch_kernel_single(f16acc_cast_func, &karg_f16acc_cast, sizeof(karg_f16acc_cast), {thread_length_f16acc_cast, 1, 1}, {256, 1, 1});
                    HIP_CALL(hipMemcpy(device_weight_to_host, device_weight,
                                   static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x * sizeof(float),
                                   hipMemcpyDeviceToHost));
                    is_valid = valid_vector<float>(host_weight, static_cast<float*>(device_weight_to_host),
                                        static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x, nrms);
                }else if(is_wmma_bf16_acc){
                    // Phase 27: bf16 analog of the is_wmma_f16_acc branch above.
                    hipFunction_t bf16acc_cast_func;
                    HIP_CALL(hipModuleGetFunction(&bf16acc_cast_func, module_tensor_cast, "tensor_cast_fp32_bf16acc_1d"));
                    tensor_cast_karg_t karg_bf16acc_cast;
                    karg_bf16acc_cast.output = device_weight;
                    karg_bf16acc_cast.input = device_weight_dtype;
                    karg_bf16acc_cast.total_length = ngroups * (k / ngroups) * (c / ngroups) * y * x;
                    const size_t thread_length_bf16acc_cast = (static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x + 8 * 256) / (8 * 256) * (8 * 256) / 8;
                    igemm_launch_kernel_single(bf16acc_cast_func, &karg_bf16acc_cast, sizeof(karg_bf16acc_cast), {thread_length_bf16acc_cast, 1, 1}, {256, 1, 1});
                    HIP_CALL(hipMemcpy(device_weight_to_host, device_weight,
                                   static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x * sizeof(float),
                                   hipMemcpyDeviceToHost));
                    is_valid = valid_vector<float>(host_weight, static_cast<float*>(device_weight_to_host),
                                        static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x, nrms);
                }else if(is_wmma_atomic_pack_bf16){
                    // Phase 34: the packed-bf16 atomic epilogue writes grad_weight directly
                    // at native bf16 width -- no fp32 workspace, no cast kernel needed,
                    // unlike is_wmma_f16_acc/is_wmma_bf16_acc above (those exist because
                    // WMMA's plain epilogue accumulates in a wider VGPR role that still
                    // needs casting down; here the packed atomic already produced the
                    // final bf16 values in memory).
                    HIP_CALL(hipMemcpy(device_weight_to_host, device_weight_dtype,
                                   static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x * data_byte,
                                   hipMemcpyDeviceToHost));
                    is_valid = valid_vector<bfloat16>(host_weight, static_cast<bfloat16*>(device_weight_to_host),
                                    static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x, nrms);
                }else if(is_wmma){
                    // WMMA always writes grad_weight at full width (fp32, or int32 for int8) --
                    // see the equivalent fwd/bwd comment near is_wmma.
                    HIP_CALL(hipMemcpy(device_weight_to_host, device_weight_dtype,
                                   static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x * sizeof(float),
                                   hipMemcpyDeviceToHost));
                    if(driver_data_type == driverInt8)
                        is_valid = valid_vector<int32_t>(host_weight, static_cast<int32_t*>(device_weight_to_host),
                                        static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x, nrms);
                    else
                        is_valid = valid_vector<float>(host_weight, static_cast<float*>(device_weight_to_host),
                                        static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x, nrms);
                }else{
                    HIP_CALL(hipMemcpy(device_weight_to_host, device_weight_dtype,
                                   static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x * data_byte,
                                   hipMemcpyDeviceToHost));
                    if(driver_data_type == driverHalf)
                        is_valid = valid_vector<float16>(host_weight, static_cast<float16*>(device_weight_to_host),
                                    static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x, nrms);
                    else if(driver_data_type == driverBFloat16)
                        is_valid = valid_vector<bfloat16>(host_weight, static_cast<bfloat16*>(device_weight_to_host),
                                    static_cast<size_t>(ngroups) * (k / ngroups) * (c / ngroups) * y * x, nrms);
                }
                printf(", valid:%s", is_valid ? "y" : "n");
                if(assert_when_invalid) assert(is_valid);
            }
        };

        if(driver_data_type == driverFloat)
            launch_conv_driver(&conv_wrw_driver, &conv_args, tunables, "wrw", driver_data_type, p_bcsv, device_input, device_weight, device_output, wrw_pre, wrw_post);
        else
            launch_conv_driver(&conv_wrw_driver, &conv_args, tunables, "wrw", driver_data_type, p_bcsv, device_input_dtype, device_weight_dtype, device_output_dtype, wrw_pre, wrw_post);

        if (need_verify) 
            free(device_weight_to_host);
    }

    if(p_bcsv)
        fclose(p_bcsv);

    free(host_input);
    free(host_weight);
    free(host_output);

    HIP_CALL(hipFree(device_input));
    HIP_CALL(hipFree(device_weight));
    HIP_CALL(hipFree(device_output));

#if defined(USE_HALF) || defined(USE_INT8) || defined(USE_BF16) || defined(USE_INT4)
    free(host_input_dtype);
    free(host_weight_dtype);
    free(host_output_dtype);

    HIP_CALL(hipFree(device_input_dtype));
    HIP_CALL(hipFree(device_weight_dtype));
    HIP_CALL(hipFree(device_output_dtype));
#endif
}
