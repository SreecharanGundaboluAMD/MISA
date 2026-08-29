# Direct per-lane `global_store_dword` epilogue — eliminate LDS reshuffle

## Motivation

rocprof-compute instruction-mix (Finding 5): non-WMMA VALU is ~50% of all instructions,
LDS is 21-27%. The non-atomic epilogue's LDS-reshuffle contributes to all three top
categories: LDS traffic (ds_write scatter + ds_read gather), barriers (s_barrier between
scatter and gather), and VALU address arithmetic for the scatter/gather indices. The
reshuffle was correctness-first: a single lane only owns one column, so vectorized
per-lane stores weren't possible directly. But 16 consecutive lanes in a half-wave
cover 16 consecutive columns — the half-wave's scalar stores already coalesce at the
memory controller level. The LDS round trip adds cost without benefit.

Cross-validated: FlyDSL (production gfx1250 GEMM) uses direct per-lane stores, no LDS
reshuffle. MISA's own docstring (coalescing_store_wmma.py:42-44) independently notes
the adjacency: "16 consecutive lanes share one output row and cover 16 consecutive
columns".

## Hardware-verified lane geometry (wmma_mapping.py:195-198)

- lane % 16 → column (fixed across all v_c indices)
- (lane/16) * 8 + j → row (j = v_c register index, 0..inst_wmma.num_v_c-1)
- wave_tile_n = 16 (all configs) — exactly one 16-lane half-wave covers one tile row

## Plan

### Phase 0 — Verify adjacency from generated assembly (CPU-only)

Read generated .inc for a real kernel and confirm the store address pattern.

### Phase 1 — Implement `_emit_direct_store` in coalescing_store_wmma.py

New method: for each v_c index j, each lane issues global_store_dword to its
(column_from_lane, row_from_lane_plus_j) address. No LDS, no barrier.

Tunable: `direct_store` (opt-in, default 0).

### Phase 2 — CPU-level zero-regression (build + diff)

Existing (direct_store=0) path must produce byte-identical output.

### Phase 3 — Hardware correctness (valid:y across directions/tiles/precisions)

Full battery: fwd/bwd/wrw, 128x128/64x64, bf16/fp16/fp32/int8, tail combinations,
multi-workgroup.

### Phase 4 — Hardware performance (A/B vs LDS-reshuffle)

Shapes that exercise non-atomic epilogue. Same-run comparison via driver_mode_normal.

### Phase 5 — Decision

| Result | Action |
|---|---|
| Faster or within noise | Enable by default |
| Modestly slower (2-10%) | Keep opt-in only |
| Significantly slower (>10%) | Abandon, document as dead end |

## Key files

- `python/operations/coalescing_store_wmma.py` — the epilogue codegen
- `python/igemm/igemm_base.py` — tunable definition
- `python/operations/wmma_mapping.py` — lane→(row,col) derivation

 Plan: Direct buffer_store_b128 Epilogue (Eliminate LDS Reshuffle)                                                                                                                         
                                                                                                                                                                                           
 ### Why this is the right next item                                                                                                                                                       
                                                                                                                                                                                           
 From rocprof Finding 5 (instruction-mix, measured on real hardware):                                                                                                                      
                                                                                                                                                                                           
 ┌───────────────┬──────────────────────────────┬─────────────────────────────────────────────┐                                                                                            
 │ Category      │ Current count                │ Source                                      │                                                                                            
 ├───────────────┼──────────────────────────────┼─────────────────────────────────────────────┤                                                                                            
 │ LDS traffic   │ 21.5% (fwd) / 27.0% (wrw)    │ LDS reshuffle's ds_write_b32 / ds_read_b128 │                                                                                            
 ├───────────────┼──────────────────────────────┼─────────────────────────────────────────────┤                                                                                            
 │ Barriers      │ part of 5.9-14.0% "Internal" │ s_barrier between scatter and gather        │                                                                                            
 ├───────────────┼──────────────────────────────┼─────────────────────────────────────────────┤                                                                                            
 │ Non-WMMA VALU │ part of 50.4% (fwd)          │ Scatter/gather address arithmetic           │                                                                                            
 └───────────────┴──────────────────────────────┴─────────────────────────────────────────────┘                                                                                            
                                                                                                                                                                                           
 The LDS reshuffle contributes to ALL three categories. Eliminating it removes the scatter LDS writes, the barrier, the gather LDS reads, and their address arithmetic — replacing them    
 with just global_store_dword from v_c directly.                                                                                                                                           
                                                                                                                                                                                           
 The cost: scalar stores instead of vectorized ones. But 16 consecutive scalar stores from adjacent lanes going to consecutive addresses already coalesce at the memory controller level   
 (wave32 → two 16-lane half-waves, each doing 16 contiguous global_store_dword). And the tradeoff is removing ~128 ds_write + ~32 ds_read + 2 s_barrier + address math.                    
                                                                                                                                                                                           
 Cross-validated by two independent sources:                                                                                                                                               
 - FlyDSL: "gfx1250's WMMA C layout already has 16 consecutive lanes covering 16 consecutive columns, contiguous enough for direct stores"                                                 
 - MISA's own docstring (coalescing_store_wmma.py:42-44): "per-lane scalar global_store_dword is already contiguous across each 16-lane half-wave"                                         
                                                                                                                                                                                           
 ────────────────────────────────────────────────────────────────────────────────                                                                                                          
                                                                                                                                                                                           
 ### Phase 0: Verify the adjacency claim (CPU-only, ~30 min)                                                                                                                               
                                                                                                                                                                                           
 Step 0a — Read generated assembly for a known kernel:                                                                                                                                     
 Build fwd bf16 128x128 (the simplest config that exercises the non-atomic epilogue):                                                                                                      
                                                                                                                                                                                           
 ```                                                                                                                                                                                       
cd /home/sgundabo/MISA && python3 igemm_codegen.py config/igemm_fwd_gtc_gfx1250_nhwc_bf16.config -d out/audit_fwd
 ```                                                                                                                                                                                       
                                                                                                                                                                                           
 Then examine out/audit_fwd/*.inc — look for the epilogue section (labeled LDS-reshuffle). Extract the actual lane→address mapping by tracing v_thread_id through v_and_b32 v[gemm_in],    
 15, v[thread_id] and the subsequent store addresses.                                                                                                                                      
                                                                                                                                                                                           
 Step 0b — Hand-derive the address pattern:                                                                                                                                                
 For a fixed v_c index j:                                                                                                                                                                  
 - Lane L's column = L % 16 (from wmma_mapping.py:195)                                                                                                                                     
 - Row base = (L / 16) * 8 + j (from wmma_mapping.py:197-198 — (lane/16)*8 + j)                                                                                                            
 - Row base depends on L/16, so rows 0-7 come from half-wave 0 (lanes 0-15), rows 8-15 from half-wave 1 (lanes 16-31)                                                                      
 - Per row: each lane stores to its column → addresses within a row are contiguous (col 0 from lane 0, col 1 from lane 1, ... col 15 from lane 15)                                         
                                                                                                                                                                                           
 Verification target: confirm that for each j, lanes 0-15 together write 16 contiguous global_store_dword covering columns 0-15 at row j (for half-wave 0's rows). If true, the LDS        
 reshuffle adds no coalescing benefit — the raw scalar stores are already fully coalesced at the 16-lane half-wave granularity.                                                            
                                                                                                                                                                                           
 Gate: if verification fails (lanes map to non-adjacent columns), abandon direct store and report why.                                                                                     
                                                                                                                                                                                           
 ────────────────────────────────────────────────────────────────────────────────                                                                                                          
                                                                                                                                                                                           
 ### Phase 1: Prototype the direct path in coalescing_store_wmma.py (~2-3 hours)                                                                                                           
                                                                                                                                                                                           
 Step 1a — Add direct_store tunable:                                                                                                                                                       
 In igemm_base.py, add self.direct_store = utility_dict_with_default_t(tunable_dict)('direct_store', 0). This is the opt-in gate — existing behavior is untouched by default.              
                                                                                                                                                                                           
 Step 1b — Add the direct-store branch:                                                                                                                                                    
 In ctrl_coalescing_store_wmma_t, add a direct_store field. In the __call__ method, add an early branch before the LDS-reshuffle section (before line 665):                                
                                                                                                                                                                                           
 ```python                                                                                                                                                                                 
if ctrl.direct_store:
    self._emit_direct_store(...)
    return
 ```                                                                                                                                                                                       
                                                                                                                                                                                           
 Step 1c — Implement _emit_direct_store:                                                                                                                                                   
 The simplest correct implementation:                                                                                                                                                      
 1. For each v_c index j in range(num_v_c): emit global_store_dword v[v_c_base + j], v[v_addr] with each lane computing its own column-scaled address.                                     
 2. The address computation reuses the existing v_gemm_im/v_gemm_in (row/col base), plus j*stride for the intra-tile row offset.                                                           
 3. No barrier, no LDS, no gather. Just N scalar stores (N = num_v_c × lanes_per_halfwave / coalescing_width).                                                                             
                                                                                                                                                                                           
 Structure: mirror the unchunked non-atomic path's per-j loop, but skip the LDS scatter and just issue global_store_dword directly from v_v_c[j] to the computed address.                  
                                                                                                                                                                                           
 Step 1d — Wire wmma_m_tail/wmma_n_tail masking:                                                                                                                                           
 The existing tail-mask logic in the LDS-reshuffle path applies EXEC masks to clip out-of-range rows/columns. The direct store path needs the same EXEC-mask guards — these are            
 lane-level, so they apply identically regardless of whether the store goes through LDS or direct.                                                                                         
                                                                                                                                                                                           
 ────────────────────────────────────────────────────────────────────────────────                                                                                                          
                                                                                                                                                                                           
 ### Phase 2: CPU-level verification (zero regression, ~30 min)                                                                                                                            
                                                                                                                                                                                           
 Step 2a — Build both paths:                                                                                                                                                               
                                                                                                                                                                                           
 ```                                                                                                                                                                                       
python3 igemm_codegen.py config/igemm_fwd_gtc_gfx1250_nhwc_bf16.config -d out/audit_base
# Then manually add direct_store=1 to the config section and rebuild:
python3 igemm_codegen.py ... -d out/audit_direct
 ```                                                                                                                                                                                       
                                                                                                                                                                                           
 Step 2b — Diff generated assembly:                                                                                                                                                        
                                                                                                                                                                                           
 ```diff                                                                                                                                                                                   
diff out/audit_base/*.s out/audit_direct/*.s
 ```                                                                                                                                                                                       
                                                                                                                                                                                           
 Expected: only the epilogue section changes. The LDS-reshuffle scatter/barrier/gather is replaced by direct stores. All other sections (prologue, main loop, prolog) must be              
 byte-identical.                                                                                                                                                                           
                                                                                                                                                                                           
 Step 2c — Verify diff is structural, not corrupting:                                                                                                                                      
 - LDS reshuffle code: ds_write_b32, s_barrier_signal, ds_read_b128, s_barrier_wait → gone                                                                                                 
 - Direct store code: global_store_dword for each v_c index → present                                                                                                                      
 - No new register allocations or spills                                                                                                                                                   
 - Address computation uses the same v_gemm_im/v_gemm_in as before                                                                                                                         
                                                                                                                                                                                           
 ────────────────────────────────────────────────────────────────────────────────                                                                                                          
                                                                                                                                                                                           
 ### Phase 3: Hardware correctness validation (~2 hours)                                                                                                                                   
                                                                                                                                                                                           
 Build and test across all non-atomic (non-split-K) config combinations:                                                                                                                   
                                                                                                                                                                                           
 ┌───────────┬─────────┬───────────┬──────────────┬────────────────────────────────┐                                                                                                       
 │ Direction │ Tile    │ Precision │ Direct store │ Valid?                         │                                                                                                       
 ├───────────┼─────────┼───────────┼──────────────┼────────────────────────────────┤                                                                                                       
 │ fwd       │ 128x128 │ bf16      │ 1            │ valid:y check                  │                                                                                                       
 ├───────────┼─────────┼───────────┼──────────────┼────────────────────────────────┤                                                                                                       
 │ fwd       │ 128x128 │ fp16      │ 1            │ valid:y check                  │                                                                                                       
 ├───────────┼─────────┼───────────┼──────────────┼────────────────────────────────┤                                                                                                       
 │ fwd       │ 128x128 │ fp32      │ 1            │ valid:y check                  │                                                                                                       
 ├───────────┼─────────┼───────────┼──────────────┼────────────────────────────────┤                                                                                                       
 │ fwd       │ 128x128 │ int8      │ 1            │ valid:y check                  │                                                                                                       
 ├───────────┼─────────┼───────────┼──────────────┼────────────────────────────────┤                                                                                                       
 │ fwd       │ 64x64   │ bf16      │ 1            │ valid:y check                  │                                                                                                       
 ├───────────┼─────────┼───────────┼──────────────┼────────────────────────────────┤                                                                                                       
 │ bwd       │ 128x128 │ bf16      │ 1            │ valid:y check                  │                                                                                                       
 ├───────────┼─────────┼───────────┼──────────────┼────────────────────────────────┤                                                                                                       
 │ bwd       │ 64x64   │ bf16      │ 1            │ valid:y check                  │                                                                                                       
 ├───────────┼─────────┼───────────┼──────────────┼────────────────────────────────┤                                                                                                       
 │ wrw       │ 128x128 │ bf16      │ 1            │ valid:y check (non-split only) │                                                                                                       
 ├───────────┼─────────┼───────────┼──────────────┼────────────────────────────────┤                                                                                                       
 │ wrw       │ 64x64   │ bf16      │ 1            │ valid:y check                  │                                                                                                       
 └───────────┴─────────┴───────────┴──────────────┴────────────────────────────────┘                                                                                                       
                                                                                                                                                                                           
 Plus tail tests (fwd wmma_m_tail=1 and wmma_n_tail=1 with the direct store path):                                                                                                         
 | fwd | 128x128 | bf16 | mtail | 1 | valid:y |                                                                                                                                            
 | fwd | 128x128 | bf16 | ntail | 1 | valid:y |                                                                                                                                            
 | fwd | 64x64 | bf16 | mtail+ntail | 1 | valid:y |                                                                                                                                        
                                                                                                                                                                                           
 Multi-workgroup tests (grid_x>1 or grid_y>1):                                                                                                                                             
 | fwd | 128x128 | bf16 | c=256 (multi-N) | 1 | valid:y |                                                                                                                                  
 | fwd | 128x128 | bf16 | n=2,H=256,W=256 (multi-M) | 1 | valid:y |                                                                                                                        
                                                                                                                                                                                           
 Gate: every test above must pass valid:y before proceeding to performance measurement. If any fails, debug the direct-address formula before continuing.                                  
                                                                                                                                                                                           
 ────────────────────────────────────────────────────────────────────────────────                                                                                                          
                                                                                                                                                                                           
 ### Phase 4: Zero-regression confirmation (≥1 hour)                                                                                                                                       
                                                                                                                                                                                           
 Rebuild every EXISTING config (without direct_store=1) and verify the generated .s/.inc is byte-identical to the committed version:                                                       
                                                                                                                                                                                           
 ```bash                                                                                                                                                                                   
git stash  # revert code changes
python3 igemm_codegen.py ... -d out/baseline  # all configs
git stash pop  # re-apply direct_store changes
python3 igemm_codegen.py ... -d out/test  # same configs, no direct_store
diff -r out/baseline out/test  # must be empty
 ```                                                                                                                                                                                       
                                                                                                                                                                                           
 Gate: the direct_store=0 path must produce identical output to ensure the code changes are fully gated and don't break existing paths.                                                    
                                                                                                                                                                                           
 ────────────────────────────────────────────────────────────────────────────────                                                                                                          
                                                                                                                                                                                           
 ### Phase 5: Hardware performance comparison (~1 hour)                                                                                                                                    
                                                                                                                                                                                           
 For the shapes that exercise the non-atomic epilogue (i.e. NOT split-K/wrw_streamk/wsred):                                                                                                
                                                                                                                                                                                           
 ┌─────────────────────────────┬──────────────────┬──────────────────────┬───────────────────┬───────┐                                                                                     
 │ Shape                       │ Config           │ Without direct_store │ With direct_store │ Delta │                                                                                     
 ├─────────────────────────────┼──────────────────┼──────────────────────┼───────────────────┼───────┤                                                                                     
 │ c=128,H=30,W=40,k=128,1x1   │ fwd bf16 128x128 │ cost:                │ cost:             │ ±X%   │                                                                                     
 ├─────────────────────────────┼──────────────────┼──────────────────────┼───────────────────┼───────┤                                                                                     
 │ c=128,H=30,W=40,k=128,1x1   │ fwd bf16 64x64   │ cost:                │ cost:             │ ±X%   │                                                                                     
 ├─────────────────────────────┼──────────────────┼──────────────────────┼───────────────────┼───────┤                                                                                     
 │ c=128,H=120,W=160,k=128,3x3 │ fwd bf16 128x128 │ cost:                │ cost:             │ ±X%   │                                                                                     
 ├─────────────────────────────┼──────────────────┼──────────────────────┼───────────────────┼───────┤                                                                                     
 │ c=128,H=30,W=40,k=128,1x1   │ bwd bf16 128x128 │ cost:                │ cost:             │ ±X%   │                                                                                     
 ├─────────────────────────────┼──────────────────┼──────────────────────┼───────────────────┼───────┤                                                                                     
 │ c=192,H=60,W=80,k=64,1x1    │ wrw bf16 64x64   │ cost:                │ cost:             │ ±X%   │                                                                                     
 └─────────────────────────────┴──────────────────┴──────────────────────┴───────────────────┴───────┘                                                                                     
                                                                                                                                                                                           
 Use IGEMM_WARMUP=5 IGEMM_REPEAT=10 and driver_mode_normal (searches all candidates). Compare the MINIMUM cost across all candidates — the direct-store kernel vs the LDS-reshuffle kernel 
 competing within the same conv_driver.exe invocation (fair same-run comparison).                                                                                                          
                                                                                                                                                                                           
 The performance question: does eliminating LDS traffic + barrier save enough to offset losing vectorized global stores? The rocprof data suggests yes (LDS cycles/instruction is low but  
 27% instruction count is large; the barrier is expensive for 4-wave workgroups).                                                                                                          
                                                                                                                                                                                           
 ────────────────────────────────────────────────────────────────────────────────                                                                                                          
                                                                                                                                                                                           
 ### Phase 6: Decision gate                                                                                                                                                                
                                                                                                                                                                                           
 ┌─────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ 
 │ Result                              │ Action                                                                                                                                          │ 
 ├─────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ 
 │ Direct store is faster or within    │ Enable by default. Set direct_store=1 as the new default in the master config union. Document in the backlog as closed.                         │ 
 │ noise (±2%)                         │                                                                                                                                                 │ 
 ├─────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ 
 │ Direct store is modestly slower     │ Keep as opt-in only. Document the tradeoff (direct store wins on shapes where the barrier is costly, LDS reshuffle wins on                      │ 
 │ (2-10%)                             │ memory-bandwidth-limited shapes).                                                                                                               │ 
 ├─────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ 
 │ Direct store is significantly       │ Abandon. Document why (likely the LDS reshuffle's vectorized global_store_dwordx4 reduces global-memory traffic more than the LDS overhead      │ 
 │ slower (>10%)                       │ costs). Add to "confirmed dead ends" section of the backlog.                                                                                    │ 
 └─────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ 
                                                                                                                                                                                           
 ────────────────────────────────────────────────────────────────────────────────                                                                                                          
                                                                                                                                                                                           
 ### Risk assessment                                                                                                                                                                       
                                                                                                                                                                                           
 ┌─────────────────────────────────────────────────────────────────┬────────────────────────────────┬────────────────────────┬──────────────────────────────────────────────────────┐      
 │ Risk                                                            │ Likelihood                     │ Impact                 │ Mitigation                                           │      
 ├─────────────────────────────────────────────────────────────────┼────────────────────────────────┼────────────────────────┼──────────────────────────────────────────────────────┤      
 │ Lane-to-column adjacency doesn't hold for all configs           │ Low (verified by mapping code) │ High (wrong results)   │ Phase 0 verification, Phase 3 exhaustive testing     │      
 ├─────────────────────────────────────────────────────────────────┼────────────────────────────────┼────────────────────────┼──────────────────────────────────────────────────────┤      
 │ Address formula bug (off-by-one in row/col to global address)   │ Medium                         │ Wrong results          │ Phase 3 testing across all directions and tile sizes │      
 ├─────────────────────────────────────────────────────────────────┼────────────────────────────────┼────────────────────────┼──────────────────────────────────────────────────────┤      
 │ Direct store is slower on GPU                                   │ Medium                         │ Abandon change         │ Phase 5 measurement before making it default         │      
 ├─────────────────────────────────────────────────────────────────┼────────────────────────────────┼────────────────────────┼──────────────────────────────────────────────────────┤      
 │ VGPR spill from extra address register                          │ Low                            │ Performance regression │ Diff VGPR count before/after                         │      
 ├─────────────────────────────────────────────────────────────────┼────────────────────────────────┼────────────────────────┼──────────────────────────────────────────────────────┤      
 │ global_store_dword vs global_store_dwordx4 bandwidth difference │ Medium                         │ Performance regression │ Measure in Phase 5                                   │      
 └─────────────────────────────────────────────────────────────────┴────────────────────────────────┴────────────────────────┴──────────────────────────────────────────────────────┘      
                                                                                                                                                                                           
 ────────────────────────────────────────────────────────────────────────────────                                                                                                          
                                                                                                                                                                                           
 ### Total effort estimate                                                                                                                                                                 
                                                                                                                                                                                           
 ┌──────────────────────┬───────────┬─────────────────┐                                                                                                                                    
 │ Phase                │ Time      │ Dependencies    │                                                                                                                                    
 ├──────────────────────┼───────────┼─────────────────┤                                                                                                                                    
 │ 0 — Verify adjacency │ 30 min    │ None (CPU-only) │                                                                                                                                    
 ├──────────────────────┼───────────┼─────────────────┤                                                                                                                                    
 │ 1 — Prototype        │ 2-3 hours │ Phase 0         │                                                                                                                                    
 ├──────────────────────┼───────────┼─────────────────┤                                                                                                                                    
 │ 2 — CPU verify       │ 30 min    │ Phase 1         │                                                                                                                                    
 ├──────────────────────┼───────────┼─────────────────┤                                                                                                                                    
 │ 3 — HW correctness   │ 2 hours   │ Phase 2         │                                                                                                                                    
 ├──────────────────────┼───────────┼─────────────────┤                                                                                                                                    
 │ 4 — Zero regression  │ 1 hour    │ Phase 3         │                                                                                                                                    
 ├──────────────────────┼───────────┼─────────────────┤                                                                                                                                    
 │ 5 — HW performance   │ 1 hour    │ Phase 3         │                                                                                                                                    
 ├──────────────────────┼───────────┼─────────────────┤                                                                                                                                    
 │ 6 — Decision gate    │ 0         │ Phase 5         │                                                                                                                                    
 ├──────────────────────┼───────────┼─────────────────┤                                                                                                                                    
 │ Total                │ 7-9 hours │                 │                                                                                                                                    
 └──────────────────────┴───────────┴─────────────────┘                                                                                                                                    
