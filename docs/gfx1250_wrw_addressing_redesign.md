 WRW Addressing Redesign: Supporting gemm_k_per_block > gemm_m_per_block
                                                                                                                                                                                   
 Problem Statement                                                                                                                                                                 
                                                                                                                                                                                   
 The wrw (backward-weight) kernel's global-load addressing for both operands A (grad_output) and B (input) derives a per-thread (row_local, col_group) split from:                 
                                                                                                                                                                                   
 ```                                                                                                                                                                               
num_col_groups = gemm_m_per_block // gemm_k_per_block
row_local = tid >> log2(num_col_groups)
col_group = tid & (num_col_groups - 1)
col_start = col_group * gemm_k_per_block
 ```                                                                                                                                                                               
                                                                                                                                                                                   
 This requires gemm_m_per_block >= gemm_k_per_block with a power-of-2 quotient, because:                                                                                           
 1. utility_log2(0) is undefined when gemm_k_per_block > gemm_m_per_block                                                                                                          
 2. The bit-slicing tid & (num_col_groups-1) assumes each thread owns exactly one column-chunk                                                                                     
 3. When num_col_groups < 1, threads would need to cooperate on a single K-row or one thread would own >1 K-row — a fundamentally different addressing scheme                      
                                                                                                                                                                                   
 The ceiling gemm_k_per_block == gemm_m_per_block == 64 was implemented in Phase 43. CK supports 64x64 tiles with K=96/128/256; MISA cannot reach these without a redesign.        
                                                                                                                                                                                   
 ────────────────────────────────────────────────────────────────────────────────                                                                                                  
                                                                                                                                                                                   
 Two Viable Designs                                                                                                                                                                
                                                                                                                                                                                   
 ### Design A: Fractional col_group (threads share rows, each owns a fractional chunk)                                                                                             
                                                                                                                                                                                   
 Idea: When gemm_k_per_block > gemm_m_per_block, invert the split. Instead of num_col_groups chunks of gemm_k_per_block rows each, use num_col_groups = 1 but let threads own      
 fractional portions of a single K-row via integer math:                                                                                                                           
                                                                                                                                                                                   
 ```                                                                                                                                                                               
# When gemm_k_per_block >= gemm_m_per_block:
# block_size threads cover gemm_m_per_block rows * gemm_k_per_block cols
# Each thread owns 1 (gemm_m_per_block * gemm_k_per_block / block_size) elements
# in a transposed [gemm_k_per_block][gemm_m_per_block] LDS tile

# Thread tid's position in the (k, m) grid:
tile_elements = gemm_k_per_block * gemm_m_per_block / block_size  # elements per thread
row_idx = (tid * tile_elements + element_within_thread) / gemm_m_per_block  # which K-row
col_idx = (tid * tile_elements + element_within_thread) % gemm_m_per_block  # which M-col

# Or more efficiently (avoid multiplication):
# Precompute: row_stride = block_size / gemm_m_per_block  # threads per K-row (always integer when gemm_m <= gemm_k)
row_idx = tid / row_stride  # which K-row this thread starts on
thread_in_row = tid % row_stride  # position within this row's thread group

# If row_stride > 1: multiple threads cooperate on one K-row, each owning gemm_m_per_block / row_stride columns
# If row_stride == 1: each thread owns a full K-row (current design, num_col_groups == 1)
 ```                                                                                                                                                                               
                                                                                                                                                                                   
 Key insight: row_stride = block_size / gemm_m_per_block is always an integer because block_size == gemm_m_per_block for the square-tile wrw case (or at least divides it via the  
 wave-grid constraints). When gemm_k_per_block > gemm_m_per_block, row_stride can exceed 1, meaning threads cooperate on each K-row.                                               
                                                                                                                                                                                   
 For A (grad_output): Each thread loads gemm_k_per_block / row_stride contiguous rows from global memory. The per-row stride is s_a_m_total() (K_out total width).                 
                                                                                                                                                                                   
 For B (input): Same decomposition but B's column indices go through the spatial gather (stride/pad computation). The col_start formula changes to account for fractional          
 ownership.                                                                                                                                                                        
                                                                                                                                                                                   
 Pros:                                                                                                                                                                             
 - Minimal VGPR overhead (just 1-2 extra for row_stride, row_idx)                                                                                                                  
 - Coalesced global loads within each row (threads in the same row group read contiguous memory)                                                                                   
 - Compatible with existing LDS storage layout — no change to double-buffering or tiling                                                                                           
 - The transposed LDS read (get_gemm_index_for_src_matrix_transposed) already handles arbitrary thread counts correctly                                                            
                                                                                                                                                                                   
 Cons:                                                                                                                                                                             
 - gemm_k_per_block / row_stride must be an integer (or the load becomes fragmented, losing coalescing)                                                                            
 - Requires careful bounds-checking when the tile doesn't divide evenly                                                                                                            
 - Need to verify the LDS double-buffer indexing works when more threads write each LDS row                                                                                        
                                                                                                                                                                                   
 Implementation scope:                                                                                                                                                             
 - igemm_wrw_gtc_wmma_nhwc.py: emit_kernel_prologue() — replace num_col_groups formula with row_stride-based addressing for both A and B operands                                  
 - igemm_wmma_mapping.py: get_gemm_index_for_src_matrix_transposed — already generic, but verify the byte-offset math handles gemm_k_per_block / row_stride per-thread correctly   
 - Config files: add new _large_k configs with gemm_k_per_block > gemm_m_per_block                                                                                                 
                                                                                                                                                                                   
 ────────────────────────────────────────────────────────────────────────────────                                                                                                  
                                                                                                                                                                                   
 ### Design B: Row ownership swap (threads own columns, a warp cooperates on rows)                                                                                                 
                                                                                                                                                                                   
 Idea: Swap the decomposition. When gemm_k_per_block > gemm_m_per_block, make threads own columns of the [gemm_k_per_block][gemm_m_per_block] tile and use a warp (16 lanes) to    
 cooperatively cover each row:                                                                                                                                                     
                                                                                                                                                                                   
 ```                                                                                                                                                                               
# When gemm_k_per_block > gemm_m_per_block:
# Each thread owns a contiguous column chunk of gemm_m_per_block / col_elements elements
# Each column is covered by 16 threads (a warp) cooperatively reading gemm_k_per_block rows

col_elements = gemm_m_per_block / 16  # elements per thread per column (always 4 for gemm_m=64)
col_start = (tid / 16) * col_elements  # which column chunk
row_offset = (tid % 16) * gemm_k_per_block  # which rows (16 lanes cover all K rows)
 ```                                                                                                                                                                               
                                                                                                                                                                                   
 Pros:                                                                                                                                                                             
 - Clean 16-lane-per-row structure matches WMMA's inherent lane grouping                                                                                                           
 - Each thread reads exactly gemm_k_per_block contiguous elements (highly coalesced)                                                                                               
                                                                                                                                                                                   
 Cons:                                                                                                                                                                             
 - Requires restructuring the global load to use warp-cooperative row iteration                                                                                                    
 - LDS double-buffering changes: instead of one lane writing one row, 16 lanes write different rows of the same column                                                             
 - B operand's per-iteration gather becomes more complex (each of the 16 lanes in a warp has a different row_offset that interacts with stride/pad)                                
 - Higher risk of breaking existing assumptions in wmma_main_loop.py                                                                                                               
                                                                                                                                                                                   
 ────────────────────────────────────────────────────────────────────────────────                                                                                                  
                                                                                                                                                                                   
 Recommended Approach: Design A                                                                                                                                                    
                                                                                                                                                                                   
 Design A is preferred because:                                                                                                                                                    
 1. Incremental risk: It changes only the global-load address computation in the prologue, not the LDS read path, WMMA compute, or epilogue                                        
 2. Compatible with existing tiling: The LDS tile remains [gemm_k_per_block][gemm_m_per_block] for A and [gemm_k_per_block][gemm_n_per_block] for B                                
 3. Minimal code churn: The num_col_groups formula is replaced by a row_stride formula in exactly one location per operand (A and B in emit_kernel_prologue)                       
 4. CK compatibility: Matches CK's approach of having multiple warps cooperate on large-K tiles                                                                                    
                                                                                                                                                                                   
 ────────────────────────────────────────────────────────────────────────────────                                                                                                  
                                                                                                                                                                                   
 Detailed Implementation Plan                                                                                                                                                      
                                                                                                                                                                                   
 ### Phase 1: Refactor num_col_groups to row_stride (wrw WMMA kernel)                                                                                                              
                                                                                                                                                                                   
 File: python/igemm/igemm_wrw_gtc_wmma_nhwc.py                                                                                                                                     
                                                                                                                                                                                   
 Location: emit_kernel_prologue(), lines ~905-943                                                                                                                                  
                                                                                                                                                                                   
 Change: Replace the num_col_groups decomposition with a row_stride decomposition:                                                                                                 
                                                                                                                                                                                   
 ```python                                                                                                                                                                         
# OLD:
num_col_groups = self.tunable.gemm_m_per_block // self.tunable.gemm_k_per_block
col_group_bits = utility_log2(num_col_groups)
col_start_shift = utility_log2(self.tunable.gemm_k_per_block)

# NEW:
if self.tunable.gemm_k_per_block <= self.tunable.gemm_m_per_block:
    # Current path: each thread owns one column-chunk
    self._emit_num_col_groups_path(num_col_groups, col_group_bits, col_start_shift)
else:
    # New path: threads share rows, row_stride > 1
    row_stride = self.tunable.gemm_k_per_block // self.tunable.gemm_m_per_block
    # Each thread owns 1/gemm_m_per_block * block_size / row_stride column-elements
    # Multiple threads (row_stride worth) share each K-row
    self._emit_row_stride_path(row_stride)
 ```                                                                                                                                                                               
                                                                                                                                                                                   
 Key new address computation for row_stride path (A operand, grad_output):                                                                                                         
                                                                                                                                                                                   
 ```asm                                                                                                                                                                            
; row_stride = block_size / gemm_m_per_block (e.g. 2 when K=128, M=64)
; Each thread owns gemm_m_per_block / block_size * gemm_k_per_block column elements
; = gemm_k_per_block / row_stride column elements

; v_row_stride_bits = log2(row_stride)
; v_row_idx = tid >> v_row_stride_bits      ; which K-row (or group of K-rows)
; v_thread_in_row = tid & (row_stride - 1)  ; position within row group

; Column stride within this thread's owned range:
; col_offset = v_thread_in_row * (gemm_m_per_block / block_size) * gemm_k_per_block
;            = v_thread_in_row * (gemm_k_per_block / row_stride) / block_size * block_size
; Simplify: col_offset = v_thread_in_row * (gemm_k_per_block / row_stride)

; Total flat index: (v_row_idx * K_out + block_m_off + col_offset) * databyte
 ```                                                                                                                                                                               
                                                                                                                                                                                   
 For B operand (input, with gather): Same row_stride decomposition but the column indices flow through the stride/pad gather (_emit_b_gather). The key difference: B's col_start   
 becomes col_offset + block_n_off where block_n_off = s.s_block_n_off().                                                                                                           
                                                                                                                                                                                   
 ### Phase 2: Refactor WMMA LDS transposed-read mapping                                                                                                                            
                                                                                                                                                                                   
 File: python/operations/wmma_mapping.py                                                                                                                                           
                                                                                                                                                                                   
 Location: get_gemm_index_for_src_matrix_transposed(), lines 125-180                                                                                                               
                                                                                                                                                                                   
 Change: Verify (and if needed, adjust) the byte-offset computation to handle the case where gemm_k_per_block / row_stride elements are read per thread per K-iteration.           
                                                                                                                                                                                   
 The critical formula in this function is the row_pitch_bytes * half_k * k_half jump. This needs to be verified for the case where half_k * row_pitch_bytes may no longer align    
 with a clean power-of-2 when row_stride > 1. In most cases it should work unchanged because the LDS storage layout itself doesn't change — only the global-load pattern changes.  
                                                                                                                                                                                   
 ### Phase 3: Update global load macros                                                                                                                                            
                                                                                                                                                                                   
 Files: python/codegen/amdgpu.py, python/operations/global_memory.py                                                                                                               
                                                                                                                                                                                   
 Change: The 2D global load macros (macro_igemm_2d_global_load_t) already handle arbitrary vectorization widths via vector_d0/vector_d1. The row_stride design requires each       
 thread to load gemm_k_per_block / row_stride rows. This maps naturally to:                                                                                                        
 - length_d0 = gemm_k_per_block / row_stride (K dimension — now > 1 for row_stride > 1)                                                                                            
 - length_d1 = gemm_m_per_block / block_size (M dimension — number of column-chunks per thread)                                                                                    
                                                                                                                                                                                   
 The existing 2D load infrastructure should handle this with minimal changes.                                                                                                      
                                                                                                                                                                                   
 ### Phase 4: Config and validation                                                                                                                                                
                                                                                                                                                                                   
 Files: config/igemm_wrw_gtc_gfx1250_nhwc_*_64x64_kmax.config (new)                                                                                                                
                                                                                                                                                                                   
 Change: Add new config sections with gemm_k_per_block = 96, 128, 256 paired with gemm_m_per_block = gemm_n_per_block = 64.                                                        
                                                                                                                                                                                   
 These require:                                                                                                                                                                    
 - block_size adjustment (if gemm_k_per_block / row_stride changes VGPR usage)                                                                                                     
 - lds_a/lds_b recalculation: 64 * 96 * 2 = 12288B each for bf16 K=96, 64 * 128 * 2 = 16384B for K=128                                                                             
 - Verify VGPR budget still fits within 256                                                                                                                                        
                                                                                                                                                                                   
 ### Phase 5: Hardware validation                                                                                                                                                  
                                                                                                                                                                                   
 Battery:                                                                                                                                                                          
 - Exact-fit shapes for K=96, 128, 256 with gemm_m = gemm_n = 64                                                                                                                   
 - K-tail and M-tail boundaries                                                                                                                                                    
 - Multi-group (group > 1)                                                                                                                                                         
 - All 4 precisions (bf16/fp16/fp32/int8)                                                                                                                                          
 - N-tail for B operand                                                                                                                                                            
 - Compare output NRMS against naive convolution                                                                                                                                   
 - Profiling: verify coalescing improves with the new address pattern (compare SDSA_USAGE vs existing)                                                                             
                                                                                                                                                                                   
 ────────────────────────────────────────────────────────────────────────────────                                                                                                  
                                                                                                                                                                                   
 Risk Assessment                                                                                                                                                                   
                                                                                                                                                                                   
 ┌──────────────────────────────────┬──────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐ 
 │ Risk                             │ Severity                                             │ Mitigation                                                                          │ 
 ├──────────────────────────────────┼──────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤ 
 │ gemm_k_per_block % row_stride != │ HIGH — would produce fractional elements, breaking   │ Assert gemm_k_per_block >= gemm_m_per_block && gemm_k_per_block % row_stride == 0   │ 
 │ 0                                │ coalescing                                           │ in tunable validation; or fall back to Design B for non-divisible cases             │ 
 ├──────────────────────────────────┼──────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤ 
 │ LDS bank conflicts with multiple │ MEDIUM — could negate coalescing gains               │ Verify via LDS bank-conflict counters (already have rocprof tooling); add LDS       │ 
 │ threads per row                  │                                                      │ padding if needed                                                                   │ 
 ├──────────────────────────────────┼──────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤ 
 │ B-gather path breaks with        │ MEDIUM — stride/pad computation must correctly       │ Test each (y, x) tap coordinate independently; compare with naive                   │ 
 │ row_stride > 1                   │ handle multiple lanes reading different K-rows       │                                                                                     │ 
 ├──────────────────────────────────┼──────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤ 
 │ Double-buffer indexing           │ LOW — existing LDS layout unchanged                  │ Verify lds_a_np2 and buffer-switch logic                                            │ 
 │ collisions                       │                                                      │                                                                                     │ 
 ├──────────────────────────────────┼──────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤ 
 │ WMMA LDS read mismatch           │ LOW — transposed read must agree with new            │ Unit test with -V 1 correctness check before profiling                              │ 
 │                                  │ global-load write pattern                            │                                                                                     │ 
 └──────────────────────────────────┴──────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘ 
                                                                                                                                                                                   
 ────────────────────────────────────────────────────────────────────────────────                                                                                                  
                                                                                                                                                                                   
 Files to Touch                                                                                                                                                                    
                                                                                                                                                                                   
 ┌───────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────┐             
 │ File                                                  │ Change                                                                                                    │             
 ├───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤             
 │ python/igemm/igemm_wrw_gtc_wmma_nhwc.py               │ emit_kernel_prologue(): new row_stride addressing for A and B operands; guards for both old and new paths │             
 ├───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤             
 │ python/operations/wmma_mapping.py                     │ get_gemm_index_for_src_matrix_transposed(): verify byte-offset math                                       │             
 ├───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤             
 │ python/codegen/amdgpu.py                              │ 2D global load macro: verify length_d0 = gemm_k_per_block / row_stride                                    │             
 ├───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤             
 │ python/operations/global_memory.py                    │ 2D global load controller: verify vectorization for new dimensions                                        │             
 ├───────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────��──┤            
 │ python/igemm/igemm_base.py                            │ Tunable validation: assert gemm_k_per_block % row_stride == 0 when K > M                                  │             
 ├───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤             
 │ config/igemm_wrw_gtc_gfx1250_nhwc_*_large_k.config    │ New configs for K=96, 128, 256                                                                            │             
 ├───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤             
 │ config/igemm_wrw_gtc_gfx1250_nhwc_*_64x64_kmax.config │ Add large-K variants to existing configs                                                                  │             
 ├───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤             
 │ driver/igemm_gtc_base.h                               │ Kernel name builder: add _large_k suffix (same pattern as _64x64_kmax)                                    │             
 ├───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤             
 │ python/igemm/igemm_wrw_gtc_nhwc.py                    │ Same changes for XDLOPS wrw path (if applicable)                                                          │             
 └───────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────┘             
                                                                                                                                                                                   
 ────────────────────────────────────────────────────────────────────────────────                                                                                                  
                                                                                                                                                                                   
 Effort Estimate                                                                                                                                                                   
                                                                                                                                                                                   
 - Phase 1 (addressing refactor): 1-2 sessions — the core logic change is focused to ~30 lines in emit_kernel_prologue                                                             
 - Phase 2 (LDS mapping verification): 1 session — likely zero changes, mostly verification                                                                                        
 - Phase 3 (global load macros): 0.5 session — mostly additive, existing infrastructure                                                                                            
 - Phase 4 (config + validation): 0.5 session — config math and driver-side naming                                                                                                 
 - Phase 5 (hardware validation): 1 session — run -V 1 battery, compare NRMS                                                                                                       
                                                                                                                                                                                   
 Total: ~3-4 sessions, assuming Design A works cleanly. Design B would add 1-2 sessions for the warp-cooperative restructuring.                                                    