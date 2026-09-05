# Practical GEMM Optimization for gfx1250 / GFX12

## Purpose

This document distills the important technical content from the **Practical GEMMs, their Performance, and how MI400 Addresses them** presentation and its transcribed discussion. It is structured so that it can be supplied to an LLM as architectural and implementation context when optimizing GEMMs, implicit-GEMM convolutions, or other tiled matrix workloads for `gfx1250`.

The examples below are educational kernel fragments from the slides. They are intentionally simplified and may omit production requirements such as complete boundary handling, descriptor definitions, launch configuration, datatype specializations, error checking, and compiler-version guards.

## Source and fidelity notes

- Primary slide deck: `UnderstandingSystolicMMAs.pptx`, authored by Hashem Hashemi.
- Related meeting: `MI400 Training 004: Practical GEMM, Presentation`, transcribed.
- The deck links to a separate step-by-step code walkthrough. That linked document was not retrievable through the available enterprise search, so this guide includes the code visible in the slides and additional implementation details stated in the meeting transcript.
- Preserve architecture guards when using these examples. Some examples explain MI300 or gfx950 behavior before showing the gfx1250 equivalent.
- Validate builtin names and signatures against the exact ROCm/Clang toolchain being used. The examples reflect the presentation, not a stable public API contract.

---

# 1. GEMM model and parallelism

For `C = A x B`:

- `A` is conceptually `M x K`.
- `B` is conceptually `K x N`, although examples may use a K-major or transposed physical layout.
- `C` is `M x N`.
- `K` is the reduction or shared dimension.
- Each `C[m,n]` is a dot product over `K`.

There are two major sources of algorithmic parallelism:

1. **MN split:** independent output positions or output tiles execute in parallel.
2. **K split:** different portions of one dot product execute in parallel and are reduced.

There are also two important architectural levels:

1. **Wave-level parallelism:** waves have independent program counters.
2. **Data-level parallelism:** lanes within a wave execute the same instruction on different data.

The first design question is therefore: **Which algorithmic dimension should map to waves, and which should map to lanes?** The answer changes with shape, layout, datatype, and whether the kernel is memory-bound or compute-bound.

---

# 2. Optimization journey at a glance

The presentation builds the kernel progressively:

1. Naive N-split kernel.
2. Wider loads along K.
3. Wave-coalesced access using transpose, swizzle, LDS staging, or intra-wave Split-K.
4. N tiling for reuse of A.
5. M tiling for reuse of B.
6. Dot2, then systolic MFMA/WMMA.
7. Macro-tile cooperative staging.
8. Async direct-to-LDS copies.
9. Bifurcated wait counters.
10. Tensor Direct Memory loads.
11. Split barriers and load/compute pipelining.
12. Multicast TDM with cluster launches.
13. Cluster barriers to limit peer drift.
14. Coalesced or combined output stores.
15. Temporal hints and scope.
16. L2 prefetch.

The target steady state for a compute-bound gfx1250 GEMM is a trace with back-to-back WMMAs and little or no exposed address-generation, memory, or synchronization latency.

---

# 3. Step 1: naive N-split

Start with `M = 1`, a vector-matrix multiply. Each lane computes one output along N.

```cpp
uint n = blockIdx.x * WvPrGrp * WAVESIZE
       + threadIdx.y * WAVESIZE + threadIdx.x;
if(n >= N) return;

float sum = 0;
for(uint k = 0; k < K; k++)
    sum += __half2float(A[k])
         * __half2float(B[n * K + k]); // stride K: uncoalesced across lanes

C[n] = __float2half(sum);
```

## Why it is slow

At a fixed loop iteration `k`, neighboring lanes access `B[n*K+k]` for different values of `n`. Their addresses are separated by K, so the wave touches many cache lines while consuming only a small part of each line. Requests are scattered, return buses are underutilized, and the kernel wastes bandwidth even though its scalar code is simple.

**Rule:** inspect addresses across lanes, not only addresses within one lane.

---

# 4. Step 2: wider loads along K

A partial remedy is to fetch a wider vector per lane.

```cpp
union bigType {
    half h[8];
    half8 h8;
};

for(uint k = 0; k < K; k += 8) {
    bigType _B = *((bigType*)(&B[n * K + k])); // 128-bit load
    for(uint _k = 0; _k < 8; _k++)
        sum += __half2float(A[k + _k])
             * __half2float(_B.h[_k]);
}
```

This uses more of every memory transaction and can improve performance. It does **not** fix the separation between neighboring lanes, and it increases register pressure.

Tune vector width against:

- alignment and divisibility,
- register use,
- tail handling,
- achieved occupancy,
- and actual memory transactions.

---

# 5. Step 3: make accesses coalesced across the wave

The goal is for neighboring lanes to reference neighboring addresses. The deck presents four approaches:

1. Pre-transpose B.
2. Partially transpose or swizzle B in chunks.
3. Cooperatively stage through LDS.
4. Map Split-K to lanes and perform a cross-lane reduction.

These are alternatives, not a universal sequence. Pick based on whether layout conversion can be amortized, whether K is large enough for Split-K, and whether LDS/register resources can support staging.

## 5.1 Pre-transpose B

```python
weights = weights.transpose(0, 1).contiguous()
```

```cpp
for(uint k = 0; k < K; k++)
    sum += __half2float(A[k])
         * __half2float(B[n + k * N]); // coalesced across n
```

Advantages:

- Simple kernel indexing.
- Natural coalescing.
- Attractive for weights that are reused many times.

Costs:

- Layout conversion and additional storage.
- Rigidity if consumers expect the original layout.
- Difficult to justify for transient activations unless the surrounding graph produces or consumes the transformed format.

## 5.2 Partial transpose / chunk swizzle

The slide uses chunks of eight so lanes remain coalesced while each lane performs a wide K load.

```python
# Slide shorthand. Use the corresponding contiguous operation in real code.
weights.view(N, K // 8, 8).transpose(0, 1).cntgs()
```

```cpp
constexpr int chnk_size = 8;
union chnk {
    DTYPE h[chnk_size];
};

for(uint k = 0; k < K; k += chnk_size) {
    chnk a_c = *((chnk*)&A[k]);
    chnk b_c = *((chnk*)&B[n * chnk_size + k * N]); // coalesced + wide
    for(int e = 0; e < chnk_size; e++)
        sum += __half2float(a_c.h[e])
             * __half2float(b_c.h[e]);
}
```

The tensor transformation and kernel indexing must agree exactly. This trades layout complexity for coalescing plus wide accesses.

## 5.3 LDS staging

Cooperatively load global memory in a coalesced pattern, place it in LDS, synchronize, and read it back in the per-lane layout needed by computation.

```cpp
__shared__ half S[WAVESIZE * WAVESIZE];

for(uint k = 0; k < K; k += WAVESIZE) {
    // Phase 1: coalesced global -> LDS
    for(int s = 0; s < WAVESIZE; s++)
        S[threadIdx.x * WAVESIZE + s] =
            B[(nBase + s) * K + k + threadIdx.x];

    __syncthreads();

    // Phase 2: each lane reads its column from LDS
    for(int s = 0; s < WAVESIZE; s++)
        sum += __half2float(A[k + s])
             * __half2float(S[WAVESIZE * s + threadIdx.x]);
}
```

LDS is functioning as a layout transformation buffer. It lets global loads be coalesced even when the compute layout is not.

Important tradeoffs:

- Naive staging can serialize memory operations and underperform.
- Unrolling and wider chunk loads can improve overlap.
- LDS consumes capacity and can reduce occupancy.
- Barriers and bounds logic add overhead.
- The transcript explicitly notes the need to avoid LDS bank conflicts, commonly through padding or an appropriate swizzle.

## 5.4 Intra-wave Split-K

One wave owns an output. Lanes walk different K positions, then reduce their partial sums.

```cpp
uint n = blockIdx.x * WvPrGrp + threadIdx.y; // wave owns one output

for(uint k = 0; k < K; k += WAVESIZE) {
    int _k = k + threadIdx.x;
    sum += __half2float(A[_k])
         * __half2float(B[n * K + _k]); // coalesced
}

for(int l = WAVESIZE / 2; l > 0; l /= 2)
    sum += __shfl_down(sum, l);
```

This is effective for skinny GEMMs, especially small M. Its costs are the reduction and poor scaling as M grows and compute/reuse bottlenecks shift.

---

# 6. Step 4: N tiling

If each lane or wave computes only one N output, it repeatedly reloads A. Compute multiple neighboring N outputs so each A value is reused.

```cpp
constexpr int n_tile = 8;
uint n = (blockIdx.x * WvPrGrp + threadIdx.y) * n_tile;
float sum[n_tile] = {0};

for(uint k = 0; k < K; k += WAVESIZE) {
    float a = __half2float(A[k + threadIdx.x]); // loaded once
    for(int nt = 0; nt < n_tile; nt++)
        sum[nt] += a * __half2float(
            B[(n + nt) * K + k + threadIdx.x]);
}
```

Larger `n_tile` improves A reuse but raises accumulator/register use and serial work per wave. It can reduce active waves and available parallelism. The best value is shape- and architecture-dependent and must be tuned empirically.

---

# 7. Step 5: increase M and reuse B

Moving from vector-matrix to full GEMM introduces M parallelism. B values can be reused across M rows.

```cpp
float sum[M] = {0};
bigType _A[M];

for(uint k = 0; k < K; k += WAVESIZE) {
    for(int m = 0; m < M; m++)
        _A[m] = *((bigType*)(&A[k + K * m]));

    for(int s = 0; s < WAVESIZE; s++) {
        float b = __half2float(S[WAVESIZE * s + nOfst]); // loaded once
        for(int m = 0; m < M; m++)
            sum[m] += __half2float(_A[m].h[s]) * b;       // reused M times
    }
}
```

As M and N tiles grow:

- arithmetic intensity rises,
- more accumulators are needed,
- narrow A loads become limiting,
- register pressure grows,
- and the kernel transitions from memory-bound toward compute-bound.

At this point scalar FMA designs waste register-file bandwidth because A and B operands are duplicated or repeatedly read across output positions.

---

# 8. Step 6a: Dot2

Dot2 packs two FP16 products into one operation.

```cpp
union bigType {
    half h[WAVESIZE];
    half2 h2[WAVESIZE / 2];
};
union hf2 {
    half h[2];
    half2 h2;
};

for(int s = 0; s < WAVESIZE / 2; s++)
    for(int m = 0; m < M; m++) {
        hf2 bF = {.h = {
            S[WAVESIZE * (s * 2)     + nOfst],
            S[WAVESIZE * (s * 2 + 1) + nOfst]
        }};
        sum[m] = __builtin_amdgcn_fdot2(
            _A[m].h2[s], bF.h2, sum[m], 0);
    }
```

It doubles work per issue relative to scalar FMA for this packing, but it does not solve the wider register-file scaling problem. It is a bridge to systolic matrix operations rather than the final design for large dense GEMMs.

---

# 9. Step 6b: systolic MFMA and gfx1250 WMMA

## 9.1 Why systolic matrix instructions

Systolic instructions distribute input fragments across lanes so one instruction computes a matrix tile while sharing register-file bandwidth across both M and N. This avoids the operand duplication inherent in scalar FMA kernels.

For the wave64 MFMA example:

- operation: `mfma_f32_16x16x16f16`,
- output tile: `16 x 16`,
- 64 lanes,
- four FP16 source elements per lane,
- four FP32 output elements per lane.

```cpp
// Stage A into MFMA register layout
for(int m = 0; m < M; m++)
    S[WAVESIZE * m + tx] = _A[m];

for(int k16 = 0; k16 < WAVESIZE / 16; k16++)
    _A[k16].h4 = *((half4*)(&S[
        WAVESIZE * (tx % 16) + (tx / 16) * 4 + k16 * 16]));

// Identical swizzle for _B, then accumulate
for(int k16 = 0; k16 < WAVESIZE / 16; k16++)
    sum4 = __builtin_amdgcn_mfma_f32_16x16x16f16(
        _A[k16].h4, _B[k16].h4, sum4, 0, 0, 0);
```

## 9.2 gfx1250 wave32 WMMA layout

For the gfx1250/GFX12 slide example:

- operation: `wmma_f32_16x16x16_f16`,
- wave size: 32,
- output tile: `16 x 16`,
- each lane holds eight FP16 source elements (`half8`),
- each lane receives eight FP32 results (`float8`),
- `tx / 16` selects lane group 0 or 1,
- output indexing uses `c + (tx / 16) * 8`, rather than the wave64 MFMA factor of four.

```cpp
// Stage A into WMMA register layout (wave32)
for(int m = 0; m < M; m++)
    S[WAVESIZE * m + tx] = _A[m];

for(int k16 = 0; k16 < WAVESIZE / 16; k16++)
    _A[k16].h8 = *((half8*)(&S[
        WAVESIZE * (tx % 16) + (tx / 16) * 8 + k16 * 16]));

// Eight fp32 results per lane
for(int k16 = 0; k16 < WAVESIZE / 16; k16++)
    sum8 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
        _A[k16].h8, _B[k16].h8, sum8);
```

Build command shown in the deck:

```bash
PYTORCH_ROCM_ARCH=gfx1250 WAVESIZE=32 python setup.py develop
```

**Critical porting rule:** do not take a wave64 MFMA kernel, only replace the builtin, and expect good or correct behavior. Retune the fragment layout, lane mapping, output mapping, tiles, pipeline, and resource use for wave32 WMMA.

---

# 10. Step 7: macro-tiles and cooperative staging

Multiple waves in one workgroup cooperatively load a larger A/B macro-tile into LDS. Each wave then computes its own 16x16 WMMA/MFMA subtile while reusing the shared operands.

The slide example describes `WAVES_M x WAVES_N = 16` waves sharing a `64 x 64` LDS tile, loaded by all lanes.

```cpp
__shared__ half As[BLOCK_M * BLOCK_K];
__shared__ half Bs[BLOCK_N * BLOCK_K];

for(uint k = 0; k < K; k += BLOCK_K) {
    for(uint idx = threadIdx.y * WAVESIZE + threadIdx.x;
        idx < BLOCK_M * BLOCK_K;
        idx += WvPrGrp * WAVESIZE)
    {
        As[idx] = A[(mBase + idx / BLOCK_K) * K
                    + k + idx % BLOCK_K];
    }

    // Identical cooperative load for Bs[]
    __syncthreads();

    // Each wave performs WMMA/MFMA on its 16x16 slice of As/Bs.
}
```

Tradeoffs:

- higher global-memory reuse,
- less redundant staging,
- two workgroup-wide barriers per K stage in the described structure,
- LDS pressure,
- barrier cost that can grow with wave count and the SIMDs involved,
- and the old global-to-register-to-LDS hop still consumes register-file bandwidth.

The next steps remove or hide these costs.

---

# 11. Step 8: async direct-to-LDS

Instead of loading global data into a VGPR and then storing to LDS, copy directly:

```text
Global memory -> LDS
```

```cpp
for(uint idx = threadIdx.y * WAVESIZE + threadIdx.x;
    idx < BLOCK_M * BLOCK_K;
    idx += WvPrGrp * WAVESIZE)
{
    __builtin_amdgcn_global_load_lds(
        &A[(mBase + idx / BLOCK_K) * K + k + idx % BLOCK_K],
        &As[idx],       // LDS destination, no data VGPR hop
        /*size=*/2,
        /*voffset=*/0,
        /*aux=*/0);
}

// The slide shows the older broad wait form in the compilable fragment.
__builtin_amdgcn_s_waitcnt(0);
__syncthreads();
```

Architecture distinction from the presentation:

- gfx950/MI350 tracks completion through `vmcnt`, requiring `s_waitcnt(vmcnt=0)` before the barrier.
- gfx1250 provides a separate async counter, allowing `s_wait_asynccnt(0)` so unrelated traffic need not be drained.

---

# 12. Step 9: bifurcated wait counters

The presentation identifies five independent MI450/gfx1250 wait domains:

```asm
s_wait_loadcnt   0   // ordinary register loads
s_wait_storecnt  0   // ordinary stores
s_wait_dscnt     0   // LDS operations
s_wait_asynccnt  0   // async-to-LDS transfers
s_wait_tensorcnt 0   // tensor/TDM transfers
```

The benefit is dependency-precise waiting. A wave waiting for a tensor or async-to-LDS transfer should not stall behind unrelated loads or stores.

Use the narrowest wait that satisfies the actual dependency. The slides state that the old general `S_WAITCNT` is deprecated on MI450 in favor of selective waits.

---

# 13. Step 10: Tensor Direct Memory loads

Per-lane global-to-LDS loops still consume instructions for index arithmetic and issuing many transfers. TDM describes a tensor tile and lets hardware move it to LDS.

The descriptor is built in scalar registers and encodes the base addresses, tensor dimensions, tile dimensions, and related transfer properties.

```cpp
// Build descriptor outside the hot K loop when fields are invariant.
gfx1250_TDM_GROUP0 g0A(
    (uintptr_t)As,
    (uintptr_t)&A[mBase * K + k]);

gfx1250_TDM_GROUP1 g1A;
g1A.dataSize(1);
g1A.tensorDim0(K);
g1A.tensorDim1(M);
g1A.tileDim0(BLOCK_K);
g1A.tileDim1(BLOCK_M);

// One instruction moves the full BLOCK_M x BLOCK_K tile.
__builtin_amdgcn_tensor_load_to_lds(g0A, g1A, g2, g3, g4, 0);
__builtin_amdgcn_s_wait_tensorcnt(0);
__syncthreads();
```

Benefits:

- removes repeated per-lane address arithmetic,
- frees SIMD issue capacity for WMMA and other useful work,
- bypasses the data VGPR hop,
- and has its own `TENSORcnt` wait domain.

Limitations called out in the deck:

- `TENSORcnt` indicates full transfer completion; it does not separately report global-read completion and LDS-write completion.
- A normal workgroup barrier still creates a serialized load-complete to compute-begins transition unless the kernel uses split barriers and pipelining.
- The exact high-level representation of the descriptor may evolve, so validate against the active compiler interface.

---

# 14. Step 11: split barriers and load pipelining

A traditional barrier combines signaling arrival and waiting. A split barrier separates them:

```text
signal arrival -> perform independent work -> wait for peers
```

The deck states that roughly 100 clocks of useful work may fit in the signal-to-wait gap in the illustrated pattern. The next K-stage TDM can be issued there while current-stage work proceeds.

```cpp
for(uint k = 0; k < K; k += BLOCK_K) {
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __builtin_amdgcn_sched_barrier(0);      // ordering fence on gfx1250
    __builtin_amdgcn_s_barrier_signal(-1);  // workgroup arrival
    __builtin_amdgcn_sched_barrier(0);

    // Signal-to-wait gap: issue the next K-stage transfer.
    issue_wave_stage(k + BLOCK_K);

    __builtin_amdgcn_s_barrier_wait(-1);

    // WMMA on the now-ready As[] and Bs[] stage.
}
```

The slide specifically requires `sched_barrier(0)` fences around the tensor wait and barrier signal sequence on gfx1250 to prevent compiler reordering.

Production pipelining normally requires double or multi-buffered LDS storage so that the producer does not overwrite operands still being consumed by WMMA.

---

# 15. Step 12: multicast TDM and cluster launch

Adjacent workgroups may request identical rows or columns of A or B. A multicast transfer fetches once from L2 and delivers the data to LDS in multiple cluster peers.

The launch sets a 2D cluster dimension. The TDM descriptor's workgroup mask selects recipients.

```cpp
hipLaunchAttribute attrs[] = {
    {hipLaunchAttributeClusterDimension,
     {CLUSTER_N, CLUSTER_M, 1}}
};

hipLaunchKernelExC(&cfg, kernel, args);
```

Inside the kernel:

```cpp
g1A.workgroupMask(a_mask);
__builtin_amdgcn_tensor_load_to_lds(g0A, g1A, g2, g3, g4, 0);
__builtin_amdgcn_s_wait_tensorcnt(0);
```

Important details from the slides and Q&A:

- Cluster workgroups are launched together.
- The cluster dimensions indicate how workgroups are grouped and permit identifying reuse along M and/or N.
- Every participating peer must execute `s_wait_tensorcnt(0)` because the tensor counter is incremented when the multicast arrives at each peer.
- The multicast mechanism waits for matching requests from peers. If requests do not align within the timeout window, it falls back to separate transfers rather than causing a functional failure.
- Multicast performance therefore depends on peers remaining sufficiently synchronized.

---

# 16. Step 13: cluster barriers

Cluster workgroups run on different CUs/WGPs and can drift. The slides state that if one peer becomes more than roughly 1024 cycles ahead, the multicast can time out and fall back to redundant loads. The meeting discussion described the default timeout as approximately 1000 clocks and noted an implementation detail involving a register value of 512 with a multiplier.

The cluster barrier pattern is:

```cpp
// 1. Synchronize waves within the current workgroup.
__builtin_amdgcn_s_barrier_signal(-1);
__builtin_amdgcn_s_barrier_wait(-1);

// 2. Exactly one wave per workgroup signals cluster arrival.
if(threadIdx.y == 0)
    __builtin_amdgcn_s_barrier_signal(-3); // -3 means cluster barrier

// 3. All waves wait for cluster peers.
__builtin_amdgcn_s_barrier_wait(-3);
```

Semantics highlighted in the training:

- `-1` represents a regular workgroup barrier.
- `-3` represents a cluster barrier.
- One wave per workgroup signals the cluster barrier, otherwise the cluster receives duplicate arrival signals.
- All waves wait.
- In this GEMM usage, the cluster barrier is primarily a performance mechanism to preserve multicast alignment, not a correctness requirement for the mathematical result.
- Too-frequent barriers add overhead. Too-infrequent barriers allow drift and multicast timeouts.
- The speaker gave a rough starting point of every low number of thousands of clocks, but emphasized that frequency is workload-dependent and should be tuned.
- Functional divergence between cluster peers can increase drift substantially.

---

# 17. Step 14: output-store optimization

WMMA outputs are lane-swizzled. Directly storing each lane's results can generate scattered partial-cache-line writes and stress the memory path, especially for small-K GEMMs where epilogue/store time is a larger fraction of runtime.

The presentation gives three strategies.

## 17.1 Claused stores

Compute store addresses first, then issue the stores as a consecutive clause. The write-combining buffer can observe adjacent partial writes, merge them, and avoid unnecessary read-modify-write traffic.

Conceptual sequence:

```cpp
// Pseudocode only: exact clause encoding depends on the compiler/ISA interface.
compute_all_output_addresses();
begin_store_clause();
issue_scattered_output_stores_back_to_back();
end_store_clause();
```

## 17.2 Stage output through LDS

```text
WMMA lane-swizzled results
    -> scattered writes into LDS
    -> synchronization
    -> cooperative coalesced LDS reads
    -> coalesced global stores
```

This exchanges LDS traffic and synchronization for cleaner global stores.

## 17.3 Async LDS-to-global

Stage the output into LDS and hand it to an asynchronous store engine. This can avoid reading staged output through normal VGPRs before writing global memory. The presentation associates this path with `ASYNCcnt` tracking.

Benchmark all three approaches. The best path depends on K, epilogue complexity, output layout, tile shape, and available LDS.

---

# 18. Step 15: temporal hints and scope

The slides state that VMEM operations carry:

- **SCOPE:** coherence domain such as WGP, SE, DEV, or SYS.
- **TH:** temporal behavior such as RT, NT, HT, LU, including near/far cache combinations.

High-level meanings given in the material:

- `RT`: regular temporal caching.
- `NT`: low or no expected reuse.
- `HT`: higher temporal priority.
- `LU`: last use. The cache can treat the line as non-temporal, and the training notes architecture-specific dirty-data behavior.

Suggested GEMM usage from the slide:

- Cooperative macro-tile A/B loads: `RT_NT`, temporal in the near cache during the cooperative load but non-temporal in L2 if they are not reread after staging.
- C output stores: `NT` when no reuse is expected.
- Last-use loads: consider `LU` where supported and semantically appropriate.

Do not apply hints mechanically. Classify reuse at each cache level and benchmark. Streaming traffic with poor reuse should not evict data with meaningful reuse.

---

# 19. Step 16: L2 prefetch

`GLOBAL_PREFETCH_B8` is presented as a fire-and-forget hint that pulls a cache line into L2:

- no wait-counter tracking,
- no pipeline stall if the prefetch misses,
- and no LDS or VGPR allocation for the prefetched tile.

The slide issues it two K stages ahead:

```text
prefetch address = k + 2 * BLOCK_K
```

This adds another latency-hiding tier beyond double-buffered LDS/TDM pipelining.

The presentation distinguishes:

- **speculative prefetch:** quietly drops a bad address, useful near tile boundaries;
- **non-speculative prefetch:** can walk page tables, and the programmer guarantees the address is valid.

Use prefetch for compute-bound kernels with visible load latency between compute blocks. It is not expected to help a kernel already limited by memory bandwidth.

---

# 20. Shape-dependent design guidance

## Skinny GEMM or GEMV-like shapes

Consider:

- intra-wave Split-K,
- wide and coalesced loads,
- smaller tiles to retain parallelism,
- store cost as a meaningful fraction of total runtime,
- and avoiding heavyweight macro-tile machinery when reuse is insufficient.

## Large M and N, large K

Prioritize:

- WMMA-centered compute,
- macro-tile reuse,
- TDM/direct-to-LDS,
- multi-stage pipelines,
- split barriers,
- multicast when adjacent workgroups share operands,
- and prefetch if thread traces show exposed memory gaps.

## Small K

Pay disproportionate attention to:

- launch and synchronization overhead,
- output permutation and store coalescing,
- epilogue fusion,
- and avoiding a pipeline whose setup cost exceeds the saved memory latency.

## Irregular M or N

The Q&A emphasized wave tails. A tile size that is ideal under a continuous performance model can become worse when a small number of leftover tiles requires another scheduling round. Model full tiles and tail waves separately, and benchmark nearby tile sizes rather than relying only on arithmetic-intensity estimates.

---

# 21. Translating these principles to convolution

Many high-performance convolutions use implicit GEMM or a GEMM-like tiled kernel. Apply the same reasoning after defining the convolution-to-GEMM mapping.

1. Define the logical M, N, and K for the convolution direction and layout.
2. Determine whether activation or weight addresses are naturally coalesced across lanes.
3. Avoid materializing `im2col` unless its conversion cost is amortized; prefer an implicit address mapping when practical.
4. Stage activation and weight tiles into LDS in a layout directly consumable by WMMA.
5. Reuse weights across output positions and activations across output channels.
6. Treat stride, dilation, padding, and boundary predicates as potential divergence and address-generation costs.
7. Evaluate TDM descriptors for regular multidimensional regions. Irregular boundary regions may require a separate path.
8. Use cluster multicast when neighboring workgroups consume the same weight or activation tile.
9. Keep boundary or tail work from disrupting the hot interior path.
10. Optimize the output transform and fused epilogue, since lane-swizzled accumulator stores can become expensive.

When an LLM is optimizing convolution code, it should not blindly reinterpret every convolution as a dense contiguous GEMM. It must preserve convolution indexing and verify that the proposed data movement remains valid for stride, dilation, groups, padding, and tensor layout.

---

# 22. Tuning dimensions and search space

The training explicitly states that there is no single performance uplift number for TDM, WMMA, or the other features. Benefit depends on datatype, M/N/K, tile shape, and the current bottleneck.

Candidate tuning parameters include:

- `BLOCK_M`, `BLOCK_N`, `BLOCK_K`,
- number of waves in M and N,
- per-wave output tile count,
- LDS padding/swizzle,
- vector width,
- number of pipeline stages,
- prefetch distance,
- Split-K factor,
- cluster dimensions,
- multicast masks,
- cluster-barrier interval,
- output-store strategy,
- temporal hints,
- and epilogue fusion.

For every candidate, record at least:

- correctness,
- median and tail latency,
- achieved throughput,
- VGPR and LDS use,
- occupancy or active waves,
- global load/store efficiency,
- LDS bank conflict indicators,
- WMMA issue density,
- wait/synchronization stalls,
- and tail-tile behavior.

The meeting discussion recommends empirical tuning because non-linear effects, especially leftover tiles and scheduling rounds, are difficult to capture completely in an analytical model. An LLM or tuning agent can generate candidates, run benchmarks, inspect traces, and refine ranges, but must not select configurations from theory alone.

---

# 23. Diagnostic decision tree

## Global loads are scattered

- Check addresses across lanes.
- Try layout transpose/swizzle if reusable.
- Otherwise stage cooperatively through LDS.
- For skinny shapes, evaluate intra-wave Split-K.

## Loads are coalesced but SIMD issue is consumed by address generation

- Use wider loads where valid.
- Move to direct-to-LDS.
- Evaluate TDM to offload tensor address generation.

## WMMA is not issued back-to-back in a compute-heavy trace

- Identify waits between WMMAs.
- Pipeline next-stage TDM while computing the current stage.
- Replace broad waits with the correct bifurcated counter.
- Use split barriers and required scheduling fences.
- Consider L2 prefetch when LDS/register capacity prevents earlier real loads.

## Multicast benefit is inconsistent

- Confirm all peers issue matching transfers.
- Verify cluster dimensions and recipient masks.
- Check for control-flow divergence.
- Tune cluster-barrier frequency.
- Look for fallback caused by peer drift/timeouts.

## Small-K kernels remain slow

- Inspect output stores and epilogue.
- Test claused stores.
- Test LDS output staging.
- Test async LDS-to-global where available.
- Reduce setup, pipeline, and synchronization overhead.

## Larger tiles regress

- Check VGPR pressure and occupancy.
- Check LDS capacity.
- Check reduction in waves.
- Check tail-wave count.
- Reduce M/N tile or pipeline depth and benchmark nearby configurations.

---

# 24. Correctness and safety checklist

Before accepting an optimization:

- Verify transpose flags and physical strides.
- Verify M/N/K tails.
- Verify vector alignment and load validity.
- Verify lane-to-fragment mapping for wave32 gfx1250 WMMA.
- Verify output index mapping for all eight FP32 values per lane.
- Verify LDS padding and bank behavior.
- Verify producer/consumer ordering around async/TDM transfers.
- Verify the exact counter being waited on.
- Verify `sched_barrier(0)` placement where required by the gfx1250 sequence.
- Verify only one wave signals a cluster barrier.
- Verify every required wave/peer executes matching waits and barriers without divergent omission.
- Verify multicast timeout fallback is only a performance effect, not relied upon for synchronization.
- Verify speculative versus non-speculative prefetch behavior at boundaries.
- Verify cache hints do not change required coherence semantics.
- Run numerical validation appropriate to the datatype and accumulation mode.

---

# 25. Compact prompt for an optimization LLM

Use the following as the task header when giving this guide and a kernel to an LLM:

```text
Target architecture: gfx1250, GFX12 wave32.

Optimize the supplied GEMM or implicit-GEMM convolution implementation while preserving exact indexing semantics and numerical requirements.

Reason explicitly about:
1. Lane-level global-memory coalescing.
2. Operand reuse across M and N.
3. LDS layout, padding, and bank conflicts.
4. Wave32 WMMA fragment and output mapping.
5. VGPR/LDS pressure and occupancy.
6. Direct-to-LDS or TDM opportunities.
7. Fine-grained wait counters and dependency correctness.
8. Double/multi-buffering and split-barrier pipelining.
9. Cluster multicast reuse and peer drift.
10. Output-store coalescing or write combining.
11. Shape tails and nonlinear wave-tail effects.
12. Cache hints and L2 prefetch only when supported by the bottleneck.

Do not merely substitute a WMMA builtin into an MFMA/FMA kernel. Retune layout, tile sizes, lane mapping, output mapping, pipeline, and resource use for wave32.

For every proposed change, provide:
- the bottleneck addressed,
- exact code modifications,
- correctness conditions,
- expected resource tradeoffs,
- benchmark cases that should improve or regress,
- and profiler or trace evidence needed to validate the claim.

Treat slide code as simplified educational pseudocode. Confirm builtin names/signatures with the active ROCm compiler and guard architecture-specific paths.
```

---

# 26. Final optimization priorities

A practical priority order is:

1. Correct mapping and boundary behavior.
2. Coalesced global memory access.
3. Reuse through M/N tiling and cooperative staging.
4. Native wave32 WMMA fragment mapping.
5. Reduction of register-hop and address-generation overhead through direct-to-LDS/TDM.
6. Continuous WMMA issue through pipelining, split barriers, and narrow waits.
7. Cross-workgroup reuse through multicast when shapes justify it.
8. Efficient output permutation and stores.
9. Cache hints and prefetch after primary bottlenecks are resolved.
10. Empirical tuning across representative shapes, including tails.

The key success criterion is not simply using every new instruction. It is selecting only the mechanisms that address the measured bottleneck for a particular shape and maintaining a dense WMMA schedule without losing performance to memory layout, register pressure, LDS pressure, barriers, tails, or stores.
