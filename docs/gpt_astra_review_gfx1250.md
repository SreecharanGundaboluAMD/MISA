 Bottom line                                                                                                                                    
                                                                                                                                                
 The branch has most of the basic gfx1250 mechanisms, but they are not yet composed into consistently efficient kernels. The highest-value      
 remaining work is:                                                                                                                             
                                                                                                                                                
 1. Fix feature-composition and tuning bugs.                                                                                                    
 2. Finish the TDM/async software pipeline and remove obsolete register allocations.                                                            
 3. Exploit WMMA operand-reuse hints and additional native transpose instructions.                                                              
 4. Add clustered multicast and asynchronous output stores where measurements justify them.                                                     
 5. Extend the fast paths beyond their current shape and precision restrictions.                                                                
                                                                                                                                                
 I found two concrete bugs, reproduced both, and tested two scheduling/data-layout experiments on the available gfx1250 device.                 
                                                                                                                                                
 I used source/configuration files, the permitted optimization guide, and targeted ISA sections. I did not read other documentation. Repository 
 files were not modified; experiments stayed under /tmp.                                                                                        
                                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                                               
                                                                                                                                                
 What is already implemented                                                                                                                    
                                                                                                                                                
 These should not be treated as missing bring-up work:                                                                                          
                                                                                                                                                
 ┌─────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ 
 │ Area            │ Existing implementation                                                                                                  │ 
 ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ 
 │ Matrix compute  │ Wave32 WMMA for FP16, BF16, FP32 and INT8; optional FP16/BF16 accumulation                                               │ 
 ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ 
 │ Input movement  │ VGPR-staged loads, scalar-base addressing, async global-to-LDS, TDM                                                      │ 
 ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ 
 │ Scheduling      │ K-substeps, double-buffered LDS, load/compute interleaving, local operand prefetch, partial LDS waits, split-barrier gap │ 
 │                 │ hoisting                                                                                                                 │ 
 ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ 
 │ LDS layout      │ Input-row padding, epilogue padding, native 16-bit transpose loads for backward and weight-gradient                      │ 
 ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ 
 │ Outputs         │ Direct stores, LDS-coalesced stores, chunked epilogues, narrowed outputs, packed BF16 atomics                            │ 
 ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ 
 │ Parallelism     │ Split-K in all three directions; WRW workspace reduction and Stream-K                                                    │ 
 ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ 
 │ Resources/cache │ High-bank accumulator addressing, temporal hints, speculative L2 prefetch                                                │ 
 └─────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ 
                                                                                                                                                
 The important distinction is implemented versus usable together versus selected effectively.                                                   
                                                                                                                                                
 1. Fix TDM’s incompatibility with padded LDS rows — confirmed correctness bug                                                                  
                                                                                                                                                
 This is the clearest immediate hardware-exploitation gap.                                                                                      
                                                                                                                                                
 Forward computes padded LDS read addresses using:                                                                                              
                                                                                                                                                
 ```text                                                                                                                                        
lds_bytes_per_row = bytes_per_row + lds_row_pad
 ```                                                                                                                                            
                                                                                                                                                
 But its TDM descriptor sets the first control word to only:                                                                                    
                                                                                                                                                
 ```text                                                                                                                                        
data_size_code << 16
 ```                                                                                                                                            
                                                                                                                                                
 The hardware’s pad_enable, pad_interval, and pad_amount fields remain zero. TDM therefore writes tightly packed rows while the WMMA operand    
 loads read padded rows.                                                                                                                        
                                                                                                                                                
 Sources:                                                                                                                                       
 - LDS layout construction (python/igemm/igemm_fwd_gtc_wmma_nhwc.py#L360)                                                                       
 - Padded operand read addresses (python/igemm/igemm_fwd_gtc_wmma_nhwc.py#L852)                                                                 
 - TDM A descriptor (python/igemm/igemm_fwd_gtc_wmma_nhwc.py#L894), B descriptor (python/igemm/igemm_fwd_gtc_wmma_nhwc.py#L950)                 
 - ISA padding fields (amd-instinct-cdna5-instruction-set-architecture.md#L6194)                                                                
                                                                                                                                                
 ### Hardware evidence                                                                                                                          
                                                                                                                                                
 A temporary FP16 forward configuration with:                                                                                                   
                                                                                                                                                
 ```text                                                                                                                                        
tile = 128×128×32
lds_double_buffer = 1
tdm_global_load = 1
lds_row_pad = 16
 ```                                                                                                                                            
                                                                                                                                                
 built successfully and was accepted by the driver. For batch=1, C=64, H=W=16, Kout=128, it produced:                                           
                                                                                                                                                
 ```text                                                                                                                                        
invalid float at 118, ref:-0.340968, pred:-nan
valid:n
 ```                                                                                                                                            
                                                                                                                                                
 Changing only the two descriptor control words in temporary assembly to enable the corresponding hardware padding made that case pass. Three   
 additional cases passed, including a reduction tail, C=1016, and a larger batch/spatial case.                                                  
                                                                                                                                                
 Remaining work: generate TDM padding fields from the same layout definition used by LDS reads and buffer sizing. Validate each                 
 operand/direction separately, including tails and buffer switches. Rejecting the combination would hide the opportunity rather than implement  
 it.                                                                                                                                            
                                                                                                                                                
 This is performance-relevant: on batch=32, C=1024, H=W=16, Kout=1024, the temporary padded-TDM kernel measured approximately 51 µs, versus 67  
 µs for unpadded TDM. However, the existing master’s best padded non-TDM kernel was still faster at approximately 46 µs.                        
                                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                                               
                                                                                                                                                
 2. Repair split-K search before trusting tuning results — confirmed driver bug                                                                 
                                                                                                                                                
 Forward and backward WMMA choose a split count from a heuristic targeting roughly 512 workgroups, then round down to an exact divisor.         
                                                                                                                                                
 More importantly, their WMMA run() branches ignore current_gks. They return before the legacy code that consumes that argument.                
                                                                                                                                                
 Consequently, IGEMM_GKS_ITERATIVE=1 does not actually explore different split counts for these WMMA paths.                                     
                                                                                                                                                
 Sources:                                                                                                                                       
 - Forward split selection (driver/igemm_fwd_gtc_driver.h#L725)                                                                                 
 - Backward split selection (driver/igemm_bwd_gtc_driver.h#L778)                                                                                
 - Outer iterative search (driver/conv_driver.cpp#L577)                                                                                         
                                                                                                                                                
 ### Hardware evidence                                                                                                                          
                                                                                                                                                
 For the forward 64×64 split-K kernel, batch=1, C=1024, H=W=16, Kout=128, iterative mode produced five entries:                                 
                                                                                                                                                
 ```text                                                                                                                                        
_gkgs[32]
_gkgs[32]
_gkgs[32]
_gkgs[32]
_gkgs[32]
 ```                                                                                                                                            
                                                                                                                                                
 All passed verification—but all used the same actual split factor.                                                                             
                                                                                                                                                
 Remaining work:                                                                                                                                
                                                                                                                                                
 - Make the requested split factor an effective, tested launch contract.                                                                        
 - Search valid factors using actual kernel resource limits and workload size, rather than only the fixed workgroup target.                     
 - Extend split-K to currently unsupported reduction remainders and tail combinations.                                                          
 - Compare complete execution costs.                                                                                                            
                                                                                                                                                
 That last point matters: the launch timing helper (driver/igemm_gtc_base.h#L765) sums durations returned by prologue/kernel/epilogue           
 callbacks, but forward’s zeroing callback (driver/igemm_fwd_gtc_driver.h#L792) performs hipMemset and returns zero. The reported kernel cost   
 therefore does not include that initialization cost.                                                                                           
                                                                                                                                                
 Do not select an execution plan solely from timings that omit its required zeroing, conversion, or reduction work.                             
                                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                                               
                                                                                                                                                
 3. Finish TDM/async load–compute overlap                                                                                                       
                                                                                                                                                
 The scheduler’s aggressive can_hoist path excludes:                                                                                            
                                                                                                                                                
 - TDM;                                                                                                                                         
 - async direct-to-LDS;                                                                                                                         
 - interleaved loads;                                                                                                                           
 - FP32;                                                                                                                                        
 - single-buffered LDS.                                                                                                                         
                                                                                                                                                
 The ordinary double-buffered load path consequently has a better-developed schedule than the dedicated transfer-engine paths.                  
                                                                                                                                                
 Sources:                                                                                                                                       
 - Eligibility and safety gates (python/operations/wmma_main_loop.py#L586)                                                                      
 - Legacy TDM/async schedule (python/operations/wmma_main_loop.py#L729)                                                                         
                                                                                                                                                
 For the generated 128×128×32 TDM kernel, the steady-state issue order is:                                                                      
                                                                                                                                                
 ```text                                                                                                                                        
wait for TDM
signal barrier
wait barrier
load operands from LDS
fully wait for LDS reads
update descriptors
issue 16 WMMAs
issue next TDM transfers
switch buffers
repeat
 ```                                                                                                                                            
                                                                                                                                                
 The next transfers are issued after the WMMA burst. Merely allocating two LDS buffers does not provide the intended load-ahead schedule.       
                                                                                                                                                
 ### Hardware experiment                                                                                                                        
                                                                                                                                                
 In temporary assembly, I moved the two next-stage TDM issues before the WMMA burst, retaining completed current-stage LDS reads and double     
 buffering.                                                                                                                                     
                                                                                                                                                
 All five tested shapes passed, across three process runs each.                                                                                 
                                                                                                                                                
 Representative medians:                                                                                                                        
                                                                                                                                                
 ┌────────────────────────────┬─────────────┬───────────────────┐                                                                               
 │ Shape: batch, C, H=W, Kout │ Current TDM │ Earlier TDM issue │                                                                               
 ├────────────────────────────┼─────────────┼───────────────────┤                                                                               
 │ 32, 1024, 16, 1024         │ 67 µs       │ 64 µs             │                                                                               
 ├────────────────────────────┼─────────────┼───────────────────┤                                                                               
 │ 1, 1024, 16, 128           │ 59 µs       │ 56 µs             │                                                                               
 ├────────────────────────────┼─────────────┼───────────────────┤                                                                               
 │ 128, 1024, 17, 1024        │ 286 µs      │ 284 µs            │                                                                               
 └────────────────────────────┴─────────────┴───────────────────┘                                                                               
                                                                                                                                                
 The last difference is small; this is evidence supporting further scheduling work, not a general speedup claim or production-ready patch.      
                                                                                                                                                
 Remaining work:                                                                                                                                
                                                                                                                                                
 - Develop a genuine double-buffered TDM/async schedule with explicit buffer ownership.                                                         
 - Compose it with operand prefetch and dependency-specific LDS waits.                                                                          
 - Separate steady-state handling from final-stage descriptor/tail work.                                                                        
 - Consider deeper buffering or producer/consumer specialization only after the two-stage schedule is effective.                                
                                                                                                                                                
 The ISA’s TDM completion-to-LDS-barrier mechanism is another candidate for that design: atomic_barrier_enable                                  
 (amd-instinct-cdna5-instruction-set-architecture.md#L6105) exists, but is not wired into these descriptors.                                    
                                                                                                                                                
 Preserve the FP32 safety gates. Current source deliberately requires double buffering and disables FP32 hoisting; the FP16 experiment does not 
 establish safety for FP32.                                                                                                                     
                                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                                               
                                                                                                                                                
 4. Remove TDM’s obsolete VGPR lifetimes                                                                                                        
                                                                                                                                                
 The forward register allocator removes global-load staging registers only when async_global_load is enabled. TDM falls into the ordinary       
 allocation branch and reserves v_gld_a and v_gld_b, despite bypassing that data path.                                                          
                                                                                                                                                
 Source: register allocation (python/igemm/igemm_fwd_gtc_wmma_nhwc.py#L541).                                                                    
                                                                                                                                                
 Generated resource counts for the same FP16 128×128×32 tile:                                                                                   
                                                                                                                                                
 ┌───────────────────────────────────┬────────────┬────────┐                                                                                    
 │ Variant                           │ VGPR count │ LDS    │                                                                                    
 ├───────────────────────────────────┼────────────┼────────┤                                                                                    
 │ Ordinary loads, double buffer     │ 252        │ 64 KiB │                                                                                    
 ├───────────────────────────────────┼────────────┼────────┤                                                                                    
 │ Async loads, double buffer        │ 221        │ 64 KiB │                                                                                    
 ├───────────────────────────────────┼────────────┼────────┤                                                                                    
 │ TDM, single buffer                │ 252        │ 64 KiB │                                                                                    
 ├───────────────────────────────────┼────────────┼────────┤                                                                                    
 │ TDM, double buffer                │ 252        │ 64 KiB │                                                                                    
 ├───────────────────────────────────┼────────────┼────────┤                                                                                    
 │ TDM, double buffer, direct output │ 252        │ 32 KiB │                                                                                    
 └───────────────────────────────────┴────────────┴────────┘                                                                                    
                                                                                                                                                
 Remaining work: eliminate unused staging allocations, associated initialization, and obsolete vector-address calculations from TDM-only        
 paths—not just change one allocation condition.                                                                                                
                                                                                                                                                
 Also examine register lifetime overlap between prologue, main loop, and epilogue before expanding high-bank usage.                             
                                                                                                                                                
 Important qualification: the HIP occupancy API reported four resident blocks for all seven compared variants. Removing registers does not      
 automatically cross an occupancy threshold. Any resulting speedup is [INFERENCE] until measured.                                               
                                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                                               
                                                                                                                                                
 5. Exploit WMMA operand-reuse hints and INT8 transpose loads                                                                                   
                                                                                                                                                
 ### WMMA operand reuse                                                                                                                         
                                                                                                                                                
 The WMMA loop repeats A across its inner N loop, but the instruction encoder emits no matrix-reuse modifiers.                                  
                                                                                                                                                
 Sources:                                                                                                                                       
 - WMMA issue order (python/operations/wmma_main_loop.py#L359)                                                                                  
 - Instruction encoder (python/operations/wmma.py#L66)                                                                                          
 - ISA reuse semantics (amd-instinct-cdna5-instruction-set-architecture.md#L4286)                                                               
                                                                                                                                                
 I verified that the installed assembler accepts:                                                                                               
                                                                                                                                                
 ```asm                                                                                                                                         
matrix_a_reuse
matrix_b_reuse
 ```                                                                                                                                            
                                                                                                                                                
 and produces distinct encodings. Neither modifier appears in the implementation.                                                               
                                                                                                                                                
 Remaining work: microbenchmark reuse-aware issue sequences and then apply the hints where the operand/opcode sequence satisfies the ISA        
 restrictions. Do not set both everywhere: the ISA explicitly constrains reuse across instruction transitions.                                  
                                                                                                                                                
 ### Native INT8 transpose loads                                                                                                                
                                                                                                                                                
 Backward/WRW already use ds_load_tr16_b128 for FP16/BF16. INT8 still uses the manual transpose/load-and-pack path.                             
                                                                                                                                                
 The relevant missing instruction is ds_load_tr8_b64, not the 16-bit form.                                                                      
                                                                                                                                                
 Sources:                                                                                                                                       
 - Current transpose selection (python/igemm/igemm_base.py#L959)                                                                                
 - ISA transpose instruction family (amd-instinct-cdna5-instruction-set-architecture.md#L6627)                                                  
                                                                                                                                                
 This is a focused next optimization for INT8 backward/WRW. There is no corresponding FP32 transpose form in that ISA table; FP32 needs a       
 different strategy.                                                                                                                            
                                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                                               
                                                                                                                                                
 6. Add clustered multicast for workloads with cross-workgroup reuse                                                                            
                                                                                                                                                
 This is a genuinely absent major hardware capability.                                                                                          
                                                                                                                                                
 Current TDM descriptors explicitly set workgroup_mask=0, and the launch helper uses ordinary module launches without a cluster contract.       
                                                                                                                                                
 Sources:                                                                                                                                       
 - Zero multicast mask (python/igemm/igemm_fwd_gtc_wmma_nhwc.py#L894)                                                                           
 - Launch helper (driver/igemm_gtc_base.h#L684)                                                                                                 
 - ISA cluster semantics (amd-instinct-cdna5-instruction-set-architecture.md#L644)                                                              
                                                                                                                                                
 Remaining work:                                                                                                                                
                                                                                                                                                
 - Cluster-aware launch configuration.                                                                                                          
 - Mapping adjacent GEMM tiles to cluster peers.                                                                                                
 - Operand-specific multicast masks.                                                                                                            
 - Correct participation/completion rules.                                                                                                      
 - Cluster-barrier cadence to control peer drift.                                                                                               
 - Non-cluster handling for unsuitable/tail grids.                                                                                              
                                                                                                                                                
 Start with regular, large 1×1 GEMMs where neighboring output tiles demonstrably share activation or weight tiles. Benchmark cluster size and   
 synchronization frequency rather than maximizing them.                                                                                         
                                                                                                                                                
 This should follow a sound single-workgroup TDM pipeline; multicast will not fix an internally serialized kernel.                              
                                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                                               
                                                                                                                                                
 7. Expand TDM beyond the current unit-convolution envelope                                                                                     
                                                                                                                                                
 TDM is explicitly restricted to nxe=0, with runtime checks rejecting non-unit convolutions.                                                    
                                                                                                                                                
 Sources:                                                                                                                                       
 - Tunable restrictions (python/igemm/igemm_base.py#L275)                                                                                       
 - Backward runtime restriction (driver/igemm_bwd_gtc_driver.h#L547)                                                                            
                                                                                                                                                
 The ISA supports more than the current 2D pilots: multidimensional tensors, descriptor iteration, gather, and padding.                         
                                                                                                                                                
 Remaining work:                                                                                                                                
                                                                                                                                                
 - Map regular convolution interiors onto richer descriptors.                                                                                   
 - Reduce repeated software address generation for stride/dilation/tap traversal.                                                               
 - Handle boundaries separately where a regular interior transfer is possible.                                                                  
 - Validate grouped layouts and physical tensor strides.                                                                                        
 - Preserve hardware gather’s out-of-bounds ordering requirements.                                                                              
                                                                                                                                                
 This is not permission to treat convolution as a contiguous GEMM. The useful target is removing address-generation work while preserving the   
 real convolution mapping.                                                                                                                      
                                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                                               
                                                                                                                                                
 8. Finish the output side of the hardware pipeline                                                                                             
                                                                                                                                                
 Direct stores and LDS reshuffling already exist. What is absent is an output path using:                                                       
                                                                                                                                                
 ```asm                                                                                                                                         
global_store_async_from_lds_*
 ```                                                                                                                                            
                                                                                                                                                
 or:                                                                                                                                            
                                                                                                                                                
 ```asm                                                                                                                                         
tensor_store_from_lds
 ```                                                                                                                                            
                                                                                                                                                
 The current coalesced path stages to LDS, reloads into VGPRs, then stores to global memory.                                                    
                                                                                                                                                
 Source: coalescing epilogue (python/operations/coalescing_store_wmma.py#L532).                                                                 
 Hardware: async store semantics (amd-instinct-cdna5-instruction-set-architecture.md#L5849).                                                    
                                                                                                                                                
 Remaining work:                                                                                                                                
                                                                                                                                                
 - Evaluate asynchronous output transfer for non-atomic, coalesced epilogues.                                                                   
 - Preserve LDS lifetime until transfer completion, especially with chunking or persistent loops.                                               
 - Evaluate address-precomputed, consecutive store sequences against existing reshuffling.                                                      
 - Compose native-width output conversion with more epilogue/resource configurations.                                                           
 - Add fused operations only where the consuming workload needs them.                                                                           
                                                                                                                                                
 Async stores are not a replacement for split-K atomic reduction.                                                                               
                                                                                                                                                
 My small-K comparison reinforces the need for alternatives: direct output took approximately 14 µs, versus approximately 11–12 µs for the      
 coalesced alternatives. Removing LDS traffic was not automatically faster.                                                                     
                                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                                               
                                                                                                                                                
 9. Make the tuning space reflect the implemented capability                                                                                    
                                                                                                                                                
 I parsed 316 sections across the 12 direction/precision master files. There are material coverage/composition gaps:                            
                                                                                                                                                
 - All 12 masters contain zero wmma_l2_prefetch entries, although narrow configurations exist.                                                  
 - The combinatorial generator exposes only 11 flags.                                                                                           
 - Larger/high-bank/chunked families exist, but are not broadly explored by that generator.                                                     
 - INT8 is omitted from its base-section enumeration.                                                                                           
 - Output/accumulation-width variants are explicitly excluded from masters.                                                                     
                                                                                                                                                
 Sources:                                                                                                                                       
 - Combinatorial flags and restrictions (script/generate_all_configs.py#L68)                                                                    
 - Width-family exclusions (script/build_gfx1250_master_configs.py#L107)                                                                        
 - Driver’s homogeneous-width requirement (driver/conv_driver.cpp#L695)                                                                         
                                                                                                                                                
 The last issue needs an execution-contract solution: partition candidates by compatible input/output semantics and reuse appropriate buffers.  
 Do not allocate per timed launch or silently compare reduced-precision accumulation as though it were numerically equivalent.                  
                                                                                                                                                
 ### Remaining parallelism work                                                                                                                 
                                                                                                                                                
 Split-K exists and helped the small-grid case I tested. The gaps are selection, remainder handling, and composition.                           
                                                                                                                                                
 WRW Stream-K also exists, but its default launch sets grid_z=total_shards, yielding max_iters=1. That is not yet evidence of an effective      
 general persistent scheduler.                                                                                                                  
                                                                                                                                                
 Source: WRW worker policy (driver/igemm_wrw_gtc_driver.h#L1140).                                                                               
                                                                                                                                                
 Benchmark genuinely persistent scheduling against ordinary split-K before porting the mechanism to other directions. For skinny/grouped        
 workloads, also consider smaller or asymmetric tiles and wave-local reduction designs rather than forcing large macro-tiles.                   
                                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                                               
                                                                                                                                                
 10. Treat FP8, block-scaled formats, and sparsity as separate feature programs                                                                 
                                                                                                                                                
 The hardware has FP8/BF8, block-scaled FP4/FP6/FP8, and structured-sparse matrix instructions.                                                 
                                                                                                                                                
 The repository’s FP8 WMMA entry is explicitly a placeholder using an FP32 datatype tag, not an end-to-end supported path.                      
                                                                                                                                                
 Source: FP8 placeholder (python/operations/wmma.py#L103).                                                                                      
                                                                                                                                                
 If those workloads matter, remaining work includes datatype/storage contracts, scales, conversion, register mappings, memory paths, epilogues, 
 driver support, and numerical validation.                                                                                                      
                                                                                                                                                
 These are significant hardware capabilities—but not substitutes for finishing efficient dense FP16/BF16 convolution, and not safe transparent  
 conversions of existing workloads.                                                                                                             
                                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                                               
                                                                                                                                                
 Correctness gates that still constrain optimization                                                                                            
                                                                                                                                                
 Several exclusions deserve root-cause work rather than permanent proliferation of incompatible flags:                                          
                                                                                                                                                
 - Backward rejects lds_double_buffer && lds_row_pad.                                                                                           
 - Forward config generation excludes scalar-base loads combined with N-tail handling.                                                          
 - FP32 requires double buffering and cannot use the current hoisted schedule.                                                                  
 - Cascading atomics are hard-disabled pending a valid completion/release protocol.                                                             
                                                                                                                                                
 Do not remove these guards merely because a small case passes.                                                                                 
                                                                                                                                                
 Also, the guide’s compiler scheduling barriers are not missing assembly instructions here. This generator emits assembly directly;             
 compiler-only scheduling intrinsics should not be copied into it as supposed hardware fences.                                                  
                                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                                               
                                                                                                                                                
 Verification and recommended order                                                                                                             
                                                                                                                                                
 The current FP16 forward master built successfully. Across three shapes, 29, 26, and 29 applicable candidates passed, with no valid:n results. 
 Its winning choices differed: padded 128×128, padded 64×64, and 64×64 split-K.                                                                 
                                                                                                                                                
 The seven-variant comparison and temporary TDM experiments are illustrative, not a full performance campaign. I also attempted                 
 hardware-counter collection twice; the selected SQ/GRBM counters returned zero despite valid execution. I cannot attribute measured            
 differences to specific stall counters.                                                                                                        
                                                                                                                                                
 ### Recommended sequence                                                                                                                       
                                                                                                                                                
 1. Fix TDM padding and the ignored WMMA split-count override.                                                                                  
 2. Prune TDM register lifetimes and finish its double-buffered schedule, comparing against the strongest padded non-TDM baseline.              
 3. Microbenchmark WMMA reuse hints and INT8 transpose loads.                                                                                   
 4. Expand tuning composition and measure complete execution-plan cost.                                                                         
 5. Develop multicast, richer convolution TDM, and async output paths according to the remaining measured bottlenecks.                          
 6. Add new numerical formats only when required by target workloads.                                                                           
                                                                                                                                                
 The branch’s main remaining challenge is making its existing mechanisms cooperate correctly and profitably—not accumulating more isolated      
 optimization flags.                                                                                                                            
                                                                                                                                                
 Evidence, configurations, generated assembly, and temporary code objects are retained in:                                                      
                                                                                                                                                
 /tmp/misa_gfx1250_analysis_rfkzq2v0/