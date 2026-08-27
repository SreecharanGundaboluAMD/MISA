#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define MIOPEN_USE_RNE_BFLOAT16

typedef union _cvt_bf16_fp32
{
    uint u32;
    ushort2 ushortx2;
    ushort ushortvec[2];
    float f32;
} _cvt_bf16_fp32_t;

// Phase 27: opposite direction of __float_to_bfloat16 below -- bf16 is just the upper 16 bits
// of an fp32 value (truncated on the way in, per __float_to_bfloat16's rounding), so widening
// back to fp32 is an exact, lossless left-shift into the high half with a zero low half. No
// rounding needed (unlike the narrowing direction).
inline __device__ __host__ float __bfloat16_to_float(ushort src_val)
{
    _cvt_bf16_fp32_t target_val;
    target_val.ushortvec[0] = 0;
    target_val.ushortvec[1] = src_val;
    return target_val.f32;
}

inline __device__ __host__ ushort __float_to_bfloat16(float src_val)
{
    _cvt_bf16_fp32_t target_val;
    target_val.f32 = src_val;

    if((~target_val.u32 & 0x7f800000) == 0) // Inf or NaN
    {
        if((target_val.u32 & 0xffff) != 0)
        {
            target_val.u32 |= 0x10000; // Preserve signaling NaN
        }
    }
    else
    {
#ifdef MIOPEN_USE_RNE_BFLOAT16
        target_val.u32 += (0x7fff + (target_val.ushortvec[1] & 1));
#endif // MIOPEN_USE_RNE_BFLOAT16
    }
    return target_val.ushortvec[1];
}

extern "C"
__global__ __launch_bounds__(256,2)
void tensor_cast_fp16_fp32_1d(half* output, float* input, int total_length)
{
    constexpr auto unroll_length = 8;
    float vec_in_data[unroll_length];
    half vec_out_data[unroll_length];
    float *tmp_in;
    half *tmp_out;

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int block_size = blockDim.x;

    int offset = bid * unroll_length * 256;
    int block_end = offset + unroll_length * 256; 

    if(block_end <= total_length)
    {
        tmp_in = input + offset + tid;
        #pragma unroll
        for(int i = 0; i < unroll_length; i++){
            vec_in_data[i] = *(tmp_in);
            tmp_in += 1 * 256;
        }

        #pragma unroll
        for(int i = 0; i < unroll_length; i++){
            vec_out_data[i] = (half)(vec_in_data[i]);
        }
        
        tmp_out = output + offset + tid * 1;
        #pragma unroll
        for(int i = 0; i < unroll_length; i++){
            *(tmp_out) = vec_out_data[i];
            tmp_out += 1 * 256;
        }
    }
    else
    {
        float in_data;
        half out_data;
        for(int i = offset; i < total_length; i += 256)
        {
            int index = min(i + tid, total_length - 1);
            in_data = input[index];
            out_data = (half)(in_data);
            *(output + index) = out_data;
        }
    }
}

extern "C"
__global__ __launch_bounds__(256,2)
void tensor_cast_bf16_fp32_1d(ushort* output, float* input, int total_length)
{
    constexpr auto unroll_length = 8;
    float vec_in_data[unroll_length];
    ushort vec_out_data[unroll_length];
    float *tmp_in;
    ushort *tmp_out;

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int block_size = blockDim.x;

    int offset = bid * unroll_length * 256;
    int block_end = offset + unroll_length * 256; 

    if(block_end <= total_length)
    {
        tmp_in = input + offset + tid;
        #pragma unroll
        for(int i = 0; i < unroll_length; i++){
            vec_in_data[i] = *(tmp_in);
            tmp_in += 1 * 256;
        }

        #pragma unroll
        for(int i = 0; i < unroll_length; i++){
            vec_out_data[i] = __float_to_bfloat16(vec_in_data[i]);
        }
        
        tmp_out = output + offset + tid * 1;
        #pragma unroll
        for(int i = 0; i < unroll_length; i++){
            *(tmp_out) = vec_out_data[i];
            tmp_out += 1 * 256;
        }
    }
    else
    {
        float in_data;
        ushort out_data;
        for(int i = offset; i < total_length; i += 256)
        {
            int index = min(i + tid, total_length - 1);
            in_data = input[index];
            out_data = __float_to_bfloat16(in_data);
            *(output + index) = out_data;
        }
    }
}

// Phase 24 (gfx1250 WMMA F16-accumulate): opposite direction from the two kernels above --
// expands the WMMA kernel's packed-fp16-accumulator native output buffer (2 bytes/element)
// back up to fp32, so the driver's existing fp32 comparison/validation logic
// (conv_driver.cpp's valid_vector<float>(...)) runs completely unchanged. See
// docs/gfx1250_wmma_layout.md's Phase 24 -- this is the ONLY new code the driver needs for
// wmma_acc_f16 output-buffer handling; the comparison call sites themselves are untouched.
extern "C"
__global__ __launch_bounds__(256,2)
void tensor_cast_fp32_fp16acc_1d(float* output, half* input, int total_length)
{
    constexpr auto unroll_length = 8;
    half vec_in_data[unroll_length];
    float vec_out_data[unroll_length];
    half *tmp_in;
    float *tmp_out;

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int block_size = blockDim.x;

    int offset = bid * unroll_length * 256;
    int block_end = offset + unroll_length * 256;

    if(block_end <= total_length)
    {
        tmp_in = input + offset + tid;
        #pragma unroll
        for(int i = 0; i < unroll_length; i++){
            vec_in_data[i] = *(tmp_in);
            tmp_in += 1 * 256;
        }

        #pragma unroll
        for(int i = 0; i < unroll_length; i++){
            vec_out_data[i] = __half2float(vec_in_data[i]);
        }

        tmp_out = output + offset + tid * 1;
        #pragma unroll
        for(int i = 0; i < unroll_length; i++){
            *(tmp_out) = vec_out_data[i];
            tmp_out += 1 * 256;
        }
    }
    else
    {
        half in_data;
        float out_data;
        for(int i = offset; i < total_length; i += 256)
        {
            int index = min(i + tid, total_length - 1);
            in_data = input[index];
            out_data = __half2float(in_data);
            *(output + index) = out_data;
        }
    }
}

// Phase 27 (gfx1250 WMMA BF16-accumulate): bf16 analog of tensor_cast_fp32_fp16acc_1d above --
// same structure, ushort (raw bf16 bit pattern) input instead of half, __bfloat16_to_float
// instead of __half2float. See docs/gfx1250_wmma_layout.md's Phase 27.
extern "C"
__global__ __launch_bounds__(256,2)
void tensor_cast_fp32_bf16acc_1d(float* output, ushort* input, int total_length)
{
    constexpr auto unroll_length = 8;
    ushort vec_in_data[unroll_length];
    float vec_out_data[unroll_length];
    ushort *tmp_in;
    float *tmp_out;

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int block_size = blockDim.x;

    int offset = bid * unroll_length * 256;
    int block_end = offset + unroll_length * 256;

    if(block_end <= total_length)
    {
        tmp_in = input + offset + tid;
        #pragma unroll
        for(int i = 0; i < unroll_length; i++){
            vec_in_data[i] = *(tmp_in);
            tmp_in += 1 * 256;
        }

        #pragma unroll
        for(int i = 0; i < unroll_length; i++){
            vec_out_data[i] = __bfloat16_to_float(vec_in_data[i]);
        }

        tmp_out = output + offset + tid * 1;
        #pragma unroll
        for(int i = 0; i < unroll_length; i++){
            *(tmp_out) = vec_out_data[i];
            tmp_out += 1 * 256;
        }
    }
    else
    {
        ushort in_data;
        float out_data;
        for(int i = offset; i < total_length; i += 256)
        {
            int index = min(i + tid, total_length - 1);
            in_data = input[index];
            out_data = __bfloat16_to_float(in_data);
            *(output + index) = out_data;
        }
    }
}

// Phase 35 (hipconv-style reduction-kernel epilogue): sums `num_partitions` disjoint
// fp32 slices of `workspace` (each `output_size` elements, laid out contiguously --
// partition p occupies workspace[p*output_size : (p+1)*output_size)) into `output`. Used
// by wrw's gemm_k_global_split + wrw_reduction_kernel mode as a non-atomic alternative to
// accumulating partial sums directly into the real output buffer -- see
// docs/gfx1250_wmma_layout.md's Phase 35 and igemm_wrw_gtc_driver.h's WMMA run(). A plain
// grid-stride loop (not the unroll-8/tail-remainder shape the cast kernels above use) is
// sufficient here: no vectorized load/store width concern, plain fp32 in and out.
extern "C"
__global__ __launch_bounds__(256,2)
void wrw_reduce_partials_f32(float* output, float* workspace, int num_partitions, int output_size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < output_size; i += gridDim.x * blockDim.x) {
        float sum = 0.f;
        for (int p = 0; p < num_partitions; p++)
            sum += workspace[static_cast<size_t>(p) * output_size + i];
        output[i] = sum;
    }
}
