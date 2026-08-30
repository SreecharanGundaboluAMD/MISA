#include <map>
// Minimal HIP host driver for lds_barrier_visibility_test.
//
// Launches a 64-thread kernel that tests whether s_barrier_signal/s_barrier_wait
// provides cross-wave LDS store visibility on gfx1250.
//
// The kernel writes distinct per-iteration magic values to LDS, barriers, then
// reads ALL slots. If any wave sees stale data from a previous iteration,
// the read fails the check and increments the fail counter.
//
// Build: see build.sh
// Usage: ./repro [num_workgroups]   (default: 20000)

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(expr) do { \
    hipError_t _e = (expr); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "HIP error %d (%s) at %s:%d: %s\n", \
                _e, hipGetErrorString(_e), __FILE__, __LINE__, #expr); \
        exit(1); \
    } \
} while (0)

static const int THREADS_PER_WG = 64;

int main(int argc, char** argv) {
    int num_wg = (argc > 1) ? atoi(argv[1]) : 20000;
    printf("lds_barrier_visibility_test: launching %d workgroups (%d threads each)\n",
           num_wg, THREADS_PER_WG);
    printf("Pattern: wave0+wave1 write distinct magic → s_wait_dscnt → s_barrier → read ALL slots → check\n");
    printf("Each iteration uses a unique magic value, so stale reads from iter N-1 are detectable.\n\n");

    HIP_CHECK(hipInit(0));
    hipDevice_t dev;
    HIP_CHECK(hipDeviceGet(&dev, 0));
    hipCtx_t ctx;
    HIP_CHECK(hipCtxCreate(&ctx, 0, dev));

    hipModule_t module;
    HIP_CHECK(hipModuleLoad(&module, "kernel.hsaco"));
    hipFunction_t kernel;
    HIP_CHECK(hipModuleGetFunction(&kernel, module, "lds_barrier_visibility_test"));

    // Output: one int32 per thread (fail count), zeroed
    size_t out_count = (size_t)num_wg * THREADS_PER_WG;
    size_t out_bytes = out_count * sizeof(int32_t);
    int32_t* d_out;
    HIP_CHECK(hipMalloc(&d_out, out_bytes));
    HIP_CHECK(hipMemset(d_out, 0, out_bytes));

    // Diagnostic: 4 int32/thread {slot_i, iter, actual, expected} -- last mismatch seen
    // by that thread, or all-zero if it never mismatched.
    size_t diag_count = out_count * 4;
    size_t diag_bytes = diag_count * sizeof(int32_t);
    int32_t* d_diag;
    HIP_CHECK(hipMalloc(&d_diag, diag_bytes));
    HIP_CHECK(hipMemset(d_diag, 0, diag_bytes));

    struct { void* out; void* diag; } args = { d_out, d_diag };
    size_t arg_size = sizeof(args);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        HIP_LAUNCH_PARAM_END
    };

    HIP_CHECK(hipModuleLaunchKernel(kernel,
        num_wg, 1, 1,          // grid
        THREADS_PER_WG, 1, 1,  // block
        0, 0, nullptr, config));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<int32_t> h_out(out_count);
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, out_bytes, hipMemcpyDeviceToHost));

    // Sum all fail counts
    int64_t total_fails = 0;
    int max_fails = 0;
    for (size_t i = 0; i < out_count; i++) {
        total_fails += h_out[i];
        if (h_out[i] > max_fails) max_fails = h_out[i];
    }

    printf("total slot-read checks: %zu (64 slots × 100 iters × %d threads × %d workgroups)\n",
           (size_t)64 * 100 * THREADS_PER_WG * num_wg, num_wg);
    printf("total failures: %lld\n", (long long)total_fails);
    printf("max failures per thread: %d\n", max_fails);

    if (total_fails > 0) {
        printf("\nRESULT: BUG REPRODUCED — s_barrier does NOT guarantee cross-wave LDS visibility.\n");
        printf("A wave read stale data from LDS after s_wait_dscnt + s_barrier_signal + s_barrier_wait,\n");
        printf("even though the writing wave had completed s_wait_dscnt before signaling the barrier.\n");
        printf("This confirms CDNA5's split barrier is purely a wave-arrival counter with no implicit LDS fence.\n");

        std::vector<int32_t> h_diag(diag_count);
        HIP_CHECK(hipMemcpy(h_diag.data(), d_diag, diag_bytes, hipMemcpyDeviceToHost));
        printf("\nSample mismatch records (thread's LAST observed mismatch; up to 20 shown):\n");
        printf("  wg   tid  slot  iter  actual(hex)  expected(hex)  staleness(iters behind)\n");
        int shown = 0;
        for (size_t t = 0; t < out_count && shown < 20; t++) {
            int32_t slot_i = h_diag[t*4+0];
            int32_t iter   = h_diag[t*4+1];
            int32_t actual = h_diag[t*4+2];
            int32_t expected = h_diag[t*4+3];
            if (slot_i == 0 && iter == 0 && actual == 0 && expected == 0) continue; // never mismatched
            int32_t actual_iter = (actual >> 16) & 0xFFFF;
            int staleness = iter - actual_iter;
            printf("  %-4zu %-4zu %-5d %-5d 0x%08x   0x%08x     %d\n",
                   t / THREADS_PER_WG, t % THREADS_PER_WG, slot_i, iter,
                   (unsigned)actual, (unsigned)expected, staleness);
            shown++;
        }
        if (shown == 0) printf("  (no non-zero diagnostic records found -- unexpected given total_fails>0)\n");

        // Aggregate: histogram of (slot_i, staleness) across every thread's last
        // recorded mismatch, to see if the pattern above is universal or one sample.
        std::vector<int> slot_hist(64, 0);
        std::map<int,int> staleness_hist;
        int total_records = 0;
        for (size_t t = 0; t < out_count; t++) {
            int32_t slot_i = h_diag[t*4+0];
            int32_t iter   = h_diag[t*4+1];
            int32_t actual = h_diag[t*4+2];
            int32_t expected = h_diag[t*4+3];
            if (slot_i == 0 && iter == 0 && actual == 0 && expected == 0) continue;
            total_records++;
            slot_hist[slot_i]++;
            int32_t actual_iter = (actual >> 16) & 0xFFFF;
            staleness_hist[iter - actual_iter]++;
        }
        printf("\nAggregate over %d recorded (last-mismatch-per-thread) records:\n", total_records);
        printf("  slot_i histogram (nonzero only):\n");
        for (int s = 0; s < 64; s++) if (slot_hist[s]) printf("    slot %2d: %d\n", s, slot_hist[s]);
        printf("  staleness histogram (iters behind):\n");
        for (auto& kv : staleness_hist) printf("    %d iters behind: %d\n", kv.first, kv.second);
    } else {
        printf("\nRESULT: No failures detected. The barrier provided LDS visibility at this scale.\n");
        printf("Try increasing num_workgroups, or the bug may be specific to WMMA pipeline timing.\n");
    }

    hipFree(d_out);
    hipFree(d_diag);
    hipModuleUnload(module);
    hipCtxDestroy(ctx);
    return total_fails > 0 ? 1 : 0;
}
