 Yes—but remove knobs that represent implementation details or correctness requirements, not knobs whose benefit genuinely depends on the 
 workload.                                                                                                                                
                                                                                                                                          
 The goal should be a small set of meaningful kernel choices, rather than dozens of independent booleans. “Everything tunable” and        
 “everything enabled” are both poor endpoints.                                                                                            
                                                                                                                                          
 1. Separate four different kinds of knobs                                                                                                
                                                                                                                                          
 They are currently mixed together.                                                                                                       
                                                                                                                                          
 ┌─────────────────────────────────┬─────────────────────────────────────┬──────────────────────────────────────────────────────────────┐ 
 │ Category                        │ Treatment                           │ Examples here                                                │ 
 ├─────────────────────────────────┼─────────────────────────────────────┼──────────────────────────────────────────────────────────────┤ 
 │ Correctness requirements        │ Enforce or derive; never search     │ FP32’s required double buffering; TDM descriptor padding     │ 
 │                                 │                                     │ matching the LDS layout                                      │ 
 ├─────────────────────────────────┼─────────────────────────────────────┼──────────────────────────────────────────────────────────────┤ 
 │ Implementation improvements     │ Make unconditional after            │ Native 16-bit LDS transpose replacing manual transpose for   │ 
 │                                 │ validation; remove flag             │ eligible FP16/BF16 paths                                     │ 
 ├─────────────────────────────────┼─────────────────────────────────────┼──────────────────────────────────────────────────────────────┤ 
 │ Workload-dependent tradeoffs    │ Retain as tuning choices            │ Tile dimensions, split-K factor, direct versus coalesced     │ 
 │                                 │                                     │ output, pipeline depth                                       │ 
 ├─────────────────────────────────┼─────────────────────────────────────┼──────────────────────────────────────────────────────────────┤ 
 │ Experimental or incomplete      │ Keep outside the normal tuning      │ Cascading atomics without a working completion protocol;     │ 
 │ mechanisms                      │ space                               │ unvalidated feature combinations                             │ 
 └─────────────────────────────────┴─────────────────────────────────────┴──────────────────────────────────────────────────────────────┘ 
                                                                                                                                          
 This distinction answers most of the question: not every field in a configuration deserves to be a tuning dimension.                     
                                                                                                                                          
 2. What should become automatic?                                                                                                         
                                                                                                                                          
 ### Correctness fixes: immediately, without a performance flag                                                                           
                                                                                                                                          
 The TDM padding issue is a good example.                                                                                                 
                                                                                                                                          
 There should not be independent decisions for:                                                                                           
                                                                                                                                          
 - whether LDS reads use padding;                                                                                                         
 - whether TDM writes insert matching padding;                                                                                            
 - how large each LDS buffer is;                                                                                                          
 - how buffer switching advances the descriptor.                                                                                          
                                                                                                                                          
 Those are consequences of one selected LDS layout.                                                                                       
                                                                                                                                          
 Conceptually:                                                                                                                            
                                                                                                                                          
 ```text                                                                                                                                  
Selected LDS layout
    ├── ordinary LDS write addresses
    ├── WMMA operand read addresses
    ├── TDM padding fields
    ├── buffer size
    └── buffer-switch offsets
 ```                                                                                                                                      
                                                                                                                                          
 Making those independently configurable creates invalid kernels, not useful tuning flexibility.                                          
                                                                                                                                          
 Likewise, removing unused TDM staging registers should be an unconditional implementation improvement. It does not need a                
 remove_unused_tdm_registers knob.                                                                                                        
                                                                                                                                          
 ### Better instruction implementations: usually automatic within an eligible family                                                      
                                                                                                                                          
 The native FP16/BF16 transpose-load path is a strong candidate. The code already defaults it on for backward/WRW.                        
                                                                                                                                          
 Once its supported domain is established, the public choice should not be:                                                               
                                                                                                                                          
 ```text                                                                                                                                  
manual transpose versus hardware transpose
 ```                                                                                                                                      
                                                                                                                                          
 It should simply use the appropriate implementation for that datatype/layout. Retain a comparison mechanism during development if        
 needed, then remove the production tuning flag when the old implementation is no longer useful.                                          
                                                                                                                                          
 Importantly, “automatic” can be conditional on the kernel family. It does not mean globally enabled for every precision and direction.   
                                                                                                                                          
 3. What should remain tunable?                                                                                                           
                                                                                                                                          
 Choices that exchange one resource or cost for another:                                                                                  
                                                                                                                                          
 - Larger tiles: more reuse, but fewer workgroups and more registers.                                                                     
 - More buffering: more overlap, but more LDS.                                                                                            
 - Direct stores: less reshuffling, but potentially worse global transactions.                                                            
 - Split-K: more parallelism, but initialization and reduction costs.                                                                     
 - Prefetch: potentially hides latency, but adds requests and instructions.                                                               
 - Narrow accumulation: different numerical behavior, not merely a faster implementation.                                                 
                                                                                                                                          
 The measurements from this analysis are concrete counterexamples to universal defaults:                                                  
                                                                                                                                          
 - L2 prefetch changed one large-K comparison from approximately 60 to 78 µs.                                                             
 - Direct stores were worse on the small-K comparison.                                                                                    
 - Padding won in the forward master search, but backward currently rejects padding combined with double buffering.                       
                                                                                                                                          
 I would not make any of those globally unconditional from the current evidence.                                                          
                                                                                                                                          
 4. Replace interacting booleans with coherent choices                                                                                    
                                                                                                                                          
 Some current flags encode alternatives to the same operation:                                                                            
                                                                                                                                          
 ```text                                                                                                                                  
async_global_load
tdm_global_load
saddr_global_load
 ```                                                                                                                                      
                                                                                                                                          
 Others jointly describe a schedule:                                                                                                      
                                                                                                                                          
 ```text                                                                                                                                  
lds_double_buffer
main_loop_interleave
local_prefetch_num
wmma_gap_hoist
 ```                                                                                                                                      
                                                                                                                                          
 Treating all of these as independent switches creates combinations that the implementation cannot actually execute.                      
                                                                                                                                          
 A better configuration expresses choices such as:                                                                                        
                                                                                                                                          
 ```text                                                                                                                                  
tile
input transfer strategy
LDS layout
pipeline schedule
output strategy
reduction strategy
 ```                                                                                                                                      
                                                                                                                                          
 Each choice resolves to the necessary low-level details.                                                                                 
                                                                                                                                          
 For example, a pipeline schedule should specify where transfers, waits, buffer switches, and operand loads occur. It should not be       
 assembled accidentally by enabling several flags that rewrite overlapping portions of the loop.                                          
                                                                                                                                          
 I would start with a handful of existing, validated schedules—not introduce a general scheduling framework or constraint solver.         
                                                                                                                                          
 5. Establish one authoritative legality contract                                                                                         
                                                                                                                                          
 Today legality is distributed across:                                                                                                    
                                                                                                                                          
 - Python tunable assertions;                                                                                                             
 - generator-specific assertions;                                                                                                         
 - configuration-generation exclusions;                                                                                                   
 - C++ runtime eligibility checks;                                                                                                        
 - master-config exclusions.                                                                                                              
                                                                                                                                          
 Those layers can disagree. A configuration can assemble, be rejected at runtime, or execute incorrectly.                                 
                                                                                                                                          
 The contract should distinguish:                                                                                                         
                                                                                                                                          
 ### Build-time legality                                                                                                                  
                                                                                                                                          
 Can this kernel be generated correctly?                                                                                                  
                                                                                                                                          
 Examples:                                                                                                                                
                                                                                                                                          
 - instruction supports the datatype;                                                                                                     
 - transfer strategy supports the selected layout;                                                                                        
 - schedule supports the selected buffering;                                                                                              
 - resource limits are satisfied;                                                                                                         
 - output strategy supports the accumulator representation.                                                                               
                                                                                                                                          
 ### Runtime applicability                                                                                                                
                                                                                                                                          
 Can this generated kernel handle this convolution?                                                                                       
                                                                                                                                          
 Examples:                                                                                                                                
                                                                                                                                          
 - unit convolution versus general convolution;                                                                                           
 - channel alignment;                                                                                                                     
 - group count;                                                                                                                           
 - dimension tails;                                                                                                                       
 - permitted reduction partitions.                                                                                                        
                                                                                                                                          
 Do not maintain multiple handwritten copies of the same restriction. Resolve build-time choices in one place and emit the resulting      
 kernel capabilities alongside the kernel. The driver should consume that contract for runtime filtering.                                 
                                                                                                                                          
 Also provide explicit rejection reasons. “Unsupported because TDM currently requires unit convolution” is much more useful than a silent 
 skip.                                                                                                                                    
                                                                                                                                          
 6. Remove knobs using evidence of dominance                                                                                              
                                                                                                                                          
 “Sure to improve performance” is usually too strong. A practical removal rule is:                                                        
                                                                                                                                          
 ▏ Within a clearly defined supported domain, the new implementation preserves the required numerical contract and makes the alternative  
 ▏ unnecessary.                                                                                                                           
                                                                                                                                          
 Before deleting a performance-related choice:                                                                                            
                                                                                                                                          
 1. Validate the relevant precision, direction, layout, tail, and resource-boundary cases.                                                
 2. Compare against the best applicable baseline, not a convenient narrow configuration.                                                  
 3. Measure complete execution cost, including required initialization and reductions.                                                    
 4. Repeat measurements sufficiently to distinguish regressions from noise.                                                               
 5. Look for a workload region where the old choice still wins.                                                                           
                                                                                                                                          
 Then:                                                                                                                                    
                                                                                                                                          
 - No useful winning region for the old implementation: remove the knob and old path.                                                     
 - A predictable winning region: select automatically by a simple rule.                                                                   
 - An irregular but meaningful tradeoff: retain it in tuning.                                                                             
                                                                                                                                          
 Avoid a universal percentage threshold. A small regression in a common workload can matter more than a large win in a rare one.          
                                                                                                                                          
 7. Reachability should be mechanically checked                                                                                           
                                                                                                                                          
 For every retained tuning dimension, require evidence that it actually does something:                                                   
                                                                                                                                          
 - It changes the resolved kernel or launch plan.                                                                                         
 - At least one supported workload can reach each retained alternative.                                                                   
 - It is included in an appropriate search family.                                                                                        
 - Selecting a value produces the requested behavior.                                                                                     
                                                                                                                                          
 The ignored WMMA current_gks argument is exactly why this matters: the outer loop appeared to explore alternatives while repeatedly      
 launching the same split factor.                                                                                                         
                                                                                                                                          
 Also deduplicate configurations after resolution. If two configurations resolve to the same implementation, keeping both only wastes     
 build and tuning work.                                                                                                                   
                                                                                                                                          
 What I would do in this branch                                                                                                           
                                                                                                                                          
 In this order:                                                                                                                           
                                                                                                                                          
 1. Inventory the current fields and classify each into the four categories above.                                                        
 2. Fix the demonstrated contract failures: TDM padding and ineffective split-count selection.                                            
 3. Make derived details internal: descriptor padding, resource sizing, unused-register elimination, mandatory safety properties.         
 4. Consolidate conflicting booleans into a few validated strategy choices.                                                               
 5. Unify legality and runtime capability information.                                                                                    
 6. Run a dominance study and delete redundant alternatives.                                                                              
 7. Expand the tuning space only where a measured tradeoff remains.                                                                       
                                                                                                                                          
 When removing a flag, migrate the Python contract, C++ mirror, configuration files, kernel naming, and search scripts together. Leaving  
 obsolete flags accepted but ignored would recreate the reachability problem.                                                             
                                                                                                                                          
 My recommendation: fewer public tuning choices, more automatically derived implementation details, and stricter contracts between them.  
 Defaults help usability; deleting a redundant dimension is the real simplification.                                                      