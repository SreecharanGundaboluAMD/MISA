// Standalone occupancy diagnostic for gfx1250 WMMA kernels -- part of this session's
// "cheap and diagnostic" optimization backlog item (see docs/gfx1250_optimization_backlog.md).
// Queries hipModuleOccupancyMaxActiveBlocksPerMultiprocessor directly (the same API
// Phase 33 already uses in driver/igemm_wrw_gtc_driver.h for its split-K heuristic cross-
// check) instead of inferring occupancy from SQ_WAVES counters, which only gives a
// cumulative total across a dispatch's lifetime, not a per-CU concurrency figure.
//
// Usage: ./occupancy_check <hsaco_path> <kernel_name> <block_size> [dynamic_lds_bytes]
//
// Prints: active blocks/CU (the API's direct answer), waves/CU (blocks * (block_size/32),
// wave32 throughout on gfx1250), and the device's per-CU wave-slot ceiling (queried via
// hipDeviceptr attributes) so a "% of theoretical max concurrency" figure can be read off
// directly -- this is what makes the number actionable rather than just a raw count.
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <cstdio>
#include <cstdlib>
#include <string>

#define HIP_CALL(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <hsaco_path> <kernel_name> <block_size> [dynamic_lds_bytes]\n", argv[0]);
        return 1;
    }
    std::string hsaco_path = argv[1];
    std::string kernel_name = argv[2];
    int block_size = atoi(argv[3]);
    size_t dyn_lds = argc > 4 ? (size_t)atoi(argv[4]) : 0;

    HIP_CALL(hipSetDevice(0));
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, 0));

    hipModule_t module;
    HIP_CALL(hipModuleLoad(&module, hsaco_path.c_str()));
    hipFunction_t kernel_func;
    hipError_t rc = hipModuleGetFunction(&kernel_func, module, kernel_name.c_str());
    if (rc != hipSuccess) {
        fprintf(stderr, "kernel '%s' not found in %s: %s\n", kernel_name.c_str(), hsaco_path.c_str(), hipGetErrorString(rc));
        return 1;
    }

    int active_blocks = 0;
    HIP_CALL(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel_func, block_size, dyn_lds));

    int waves_per_block = (block_size + 31) / 32;  // wave32 throughout gfx1250
    int waves_per_cu = active_blocks * waves_per_block;
    // maxThreadsPerMultiProcessor / 32 is the hardware's absolute wave-slot ceiling per CU
    // (independent of any particular kernel's VGPR/LDS footprint) -- the denominator for a
    // meaningful "% of theoretical max concurrency" figure.
    int max_waves_per_cu = prop.maxThreadsPerMultiProcessor / 32;
    double pct = max_waves_per_cu > 0 ? (100.0 * waves_per_cu / max_waves_per_cu) : 0.0;

    printf("kernel=%s block_size=%d active_blocks_per_cu=%d waves_per_cu=%d max_waves_per_cu=%d occupancy_pct=%.1f%%\n",
           kernel_name.c_str(), block_size, active_blocks, waves_per_cu, max_waves_per_cu, pct);
    return 0;
}
