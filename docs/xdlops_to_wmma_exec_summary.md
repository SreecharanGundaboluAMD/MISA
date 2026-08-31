  Executive summary
  
  The WMMAepath isnnotsjust-"XDLOPSewithddifferent FMA instructions." It is a ground-up rewrite that already includes several gfx1250-specific optimizations the XDLOPS
  path never had (async globalcloads, TDM, SADDR, WMMA-tail handling, direct st re). However, t e XDLOPS g nerators accumulated years of micro-optimizations that are
  stillnabsent or only partially present in the WMMA generators. Theohighest-valueiports fall into t ree buckets:

  1.eWithin-WMMA cross-port: FeaturescalreadyuworkingninmonemWMMAldirectiondbutamissing_inothewothers (saddr_global_load,hmain_loop_interleave,agsplit_stagger,-field +
     async_global_loadeforuwrw).ng the whole generator.
  2. XDLOPS→WMMA architectural ports: Features that would need new WMMA code but are proven valuable elsewhere (tensor_a/b_pass_through, merge_e, source_access_order,
✻    global prefetch buffers, configurable vector_store).
  3. Shared-operation plumbing: Improvements to the WMMA main-loop / coalescing-store control objects (move_slice_window_accumule_functor, main-loop LDS padding,
❯    generalized coalescing_store_groups).
  ⎿  Stats dialog dismissed
   --
❯ 
  Ranked porting opportunities

❯ Tier 1 — High impact, relatively containedents into the docs folder including this summary.                                                                             

  ┌─────┬───────────────────────────┬──────────────────────────┬──────────────────────────────────────────────┬──────────────────────────────────────────────────────┐
  │  #oc│/gfx1250_oFeaturetion_backl│g.md  Status in WMMA      │                Why it matters                │                   Code references                    │
  ├─────┼───────────────────────────┼──────────────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
✻ │r1   │ main_loop_interleave  )   │ ✅ DONE in all 3 dirs      │ Overlaps global loads/stores with WMMA       │ All 3 generators now set ctrl.interleave_a/b;       │
  │     │                           │ (fwd: A+B; bwd/wrw: A only)│ execution, hiding latency in the hot K-loop. │ bwd/wrw: interleave_a=True, interleave_b=False      │
  ├─────┼───────────────────────────┼──────────────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ WRit│ it into a new file        │                          │ Saves 1 VGPR per 64-bit VADDR pair and       │                                                      │  
──│ 2  ─│ saddr_global_load (32-bit │ Implemented in fwd only; │ removes─carry-chain overhead. wrw is─near    │ igemm_fwd_gtc_wmma_nhwc.py:263-275;─bwd/wrw n─t      │
❯ │     │  SADDR global loads)      │  missin  in bwd and wrw  │ the VGPR ceiling, so this matters most       │ referenced                                           │
──│─────│───────────────────────────│──────────────────────────│─there.───────────────────────────────────────│──────────────────────────────────────────────────────│────
  ├─────┼───────────────────────────┼──────────────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │     │                           │ Fully implemented in     │ wrw has the largest GEMM_K (N·Ho·Wo) and is  │ igemm_wrw_gtc_wmma_nhwc.py:286 (only assert); fwd    │
  │ 3   │ async_global_load for wrw │ fwd; A-only in bwd; not  │ the most memory-latency-bound direction.     │ fully implemented                                    │
  │     │                           │ wired in wrw             │                                              │                                                      │
  ├─────┼───────────────────────────┼──────────────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ 4   │ gsplit_stagger            │ Implemented in wrw only; │ Scatters split-K shard global-load cache     │ igemm_wrw_gtc_wmma_nhwc.py:845-852; absent in        │
  │     │                           │  missing in fwd and bwd  │ collisions by staggering start offsets.      │ fwd/bwd                                              │
  ├─────┼───────────────────────────┼──────────────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │     │ local_prefetch_num > 1    │ Wired but                │ XDLOPS uses multi-slot LDS prefetch to hide  │ igemm_wrw_gtc_wmma_nhwc.py:1749;                     │
  │ 5   │ validation/use in wrw     │ untested/unused          │ ds_read latency; wrw's strided transposed    │ wmma_main_loop.py:284 asserts incompatibility with   │
  │     │                           │                          │ LDS reads are especially latency-sensitive.  │ TDM                                                  │
  ├─────┼───────────────────────────┼──────────────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │     │ wmma_acc_high_bank /      │ Implemented in fwd/bwd   │ Places accumulator v_c in high VGPR range to │                                                      │
  │ 6   │ VGPR-MSB                  │ only; missing in wrw     │  avoid bank conflicts with A/B operands      │ fwd/bwd implemented; wrw not referenced              │
  │     │                           │                          │ during WMMA burst.                           │                                                      │
  └─────┴───────────────────────────┴──────────────────────────┴──────────────────────────────────────────────┴──────────────────────────────────────────────────────┘

  Tier 2 — Medium impact, more invasive

  ┌─────┬──────────────────────────────┬─────────────────┬──────────────────────────────────────────────────────────┬────────────────────────────────────────────────┐
  │  #  │           Feature            │ Status in WMMA  │                      Why it matters                      │                Code references                 │
  ├─────┼──────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │     │ tensor_a_pass_through /      │ Not implemented │ Skips LDS for one operand, saving LDS bandwidth and      │                                                │
  │ 7   │ tensor_b_pass_through (LDS   │  in any WMMA    │ barrier cycles on skinny shapes. igemm_base.py already   │ igemm_base.py:358-359,907-908;                 │
  │     │ bypass)                      │ direction       │ declares the fields and zeroes LDS when set, but         │ wmma_main_loop.py lacks pass-through           │
  │     │                              │                 │ wmma_main_loop.py has no pass-through branch.            │                                                │
  ├─────┼──────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │     │ merge_e (fold c×y×x into     │ Not implemented │ Eliminates runtime y×x tap-loop overhead for multi-tap   │ igemm_base.py:404;                             │
  │ 8   │ GEMM_K)                      │  in any WMMA    │ filters. XDLOPS has full support in fwd/bwd/wrw.         │ igemm_fwd_gtc_nhwc.py:500-621 (fwd XDLOPS      │
  │     │                              │ direction       │                                                          │ reference)                                     │
  ├─────┼──────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │     │ source_access_order (M-major │ Not implemented │ Controls L2 reuse of input vs weight; important for      │ igemm_base.py:376,1335; WMMA generators use    │
  │ 9   │  vs N-major grid)            │  in any WMMA    │ large convolutions. The field exists in igemm_base.py    │ fixed bx→M, by→N                               │
  │     │                              │ direction       │ and is encoded into kernel names.                        │                                                │
  ├─────┼──────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │     │ Configurable vector_store    │ Not implemented │ XDLOPS chooses epilogue vector width (1/2/4/8/16) per    │ igemm_base.py:381,942-955;                     │
  │ 10  │ width                        │  in WMMA        │ precision and split-K mode. WMMA store width is          │ python/operations/coalescing_store_wmma.py     │
  │     │                              │                 │ hardcoded in coalescing_store_wmma.py.                   │                                                │
  ├─────┼──────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │     │ global_prefetch_a_num /      │ Not implemented │ Holds next tile's in-flight global data while current    │ igemm_base.py:886-887; mfma_main_loop.py uses  │
  │ 11  │ global_prefetch_b_num (VGPR  │  in WMMA        │ tile computes. Derived from pass-through mode in XDLOPS. │ v_gld_a_gpf/v_gld_b_gpf                        │
  │     │ double-staging)              │                 │                                                          │                                                │
  ├─────┼──────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ 12  │ Main-loop LDS padding        │ Not implemented │ Breaks bank conflicts in main-loop ds_read/ds_write.     │ igemm_base.py:906,913-914; mfma_main_loop.py   │
  │     │ (lds_pad_m/lds_pad_n)        │  in WMMA        │ igemm_base.py:get_lds_pad() returns 0 for non-XDLOPS.    │ uses fctrl.lds_pad_m/n                         │
  └─────┴──────────────────────────────┴─────────────────┴──────────────────────────────────────────────────────────┴────────────────────────────────────────────────┘

  Tier 3 — Lower impact or niche

  ┌─────┬─────────────────────────────────────┬───────────────────┬─────────────────────────────────────────────────────┬────────────────────────────────────────────┐
  │  #  │               Feature               │  Status in WMMA   │                   Why it matters                    │              Code references               │
  ├─────┼─────────────────────────────────────┼───────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │     │ move_slice_window_accumule_functor  │ Missing in WMMA   │ XDLOPS bwd uses an extra slice-window advance for   │ mfma_main_loop.py:60;                      │
  │ 13  │ in bwd                              │ main-loop control │ multi-tap when merge_e==0; WMMA bwd handles it      │ igemm_bwd_gtc_nhwc.py:2941                 │
  │     │                                     │                   │ inline.                                             │                                            │
  ├─────┼─────────────────────────────────────┼───────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │     │                                     │ Missing in wrw    │ Reduces peak LDS usage for large tiles by chunking  │ fwd/bwd implemented; wrw                   │
  │ 14  │ wmma_epilogue_chunked for wrw       │ only              │ the epilogue.                                       │ ctrl_coalescing_store_wmma setup does not  │
  │     │                                     │                   │                                                     │ set it                                     │
  ├─────┼─────────────────────────────────────┼───────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │     │                                     │ Not implemented   │ XDLOPS computes tap indices from block index for    │ igemm_base.py:369;                         │
  │ 15  │ multihead on-device tap dispatch    │ in WMMA           │ single-tap-per-workgroup kernels. WMMA's runtime    │ igemm_bwd_gtc_nhwc.py:2024-2079            │
  │     │                                     │                   │ tap loop subsumes this.                             │                                            │
  ├─────┼─────────────────────────────────────┼───────────────────┼─────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │     │                                     │ Not implemented   │ Caches scalar offsets for multi-row global loads.   │                                            │
  │ 16  │ precache_soffset                    │ in WMMA           │ WMMA's persistent VADDR design largely removes the  │ igemm_base.py:372                          │
  │     │                                     │                   │ need.                                               │                                            │
  └─────┴─────────────────────────────────────┴───────────────────┴─────────────────────────────────────────────────────┴────────────────────────────────────────────┘

  ---

  Notable architectural differences (not port opportunities)

  - AGPR vs VGPR accumulators: XDLOPS uses AGPRs; WMMA accumulates directly in VGPRs. The AGPR→VGPR transfer loop in coalescing_store_xdlops.py has no WMMA equivalent and
    is not needed.
  - bf16_1k_in_fp16: This is a gfx90a-specific workaround and correctly absent on gfx1250, which has native bf16 WMMA instructions.
  - SRD buffer_load_dwordx4 vs flat global_load_dwordx4: XDLOPS uses 4-SGPR buffer descriptors with hardware OOB clamping. WMMA uses flat/global loads with explicit EXEC
    masking. The SADDR pilot in fwd WMMA is the closest gfx1250 analogue, not a direct SRD port.

  ---

  Bottom line

  The fastest performance wins for WMMA are cross-porting features already proven in one WMMA direction to the others:

  - ~~main_loop_interleave → bwd, wrw~~ ✅ Done — implemented but no perf gain on bwd/wrw (bwd ~7-10% slower, wrw flat)
  - saddr_global_load → bwd, wrw
  - async_global_load → wrw, and B operand of bwd
  - gsplit_stagger → fwd, bwd
  - wmma_acc_high_bank → wrw

  After that, the most impactful XDLOPS-derived additions would be:

  - tensor_a/b_pass_through (new WMMA main-loop branch)
  - merge_e (eliminates tap-loop overhead)
  - source_access_order (L2 locality)
  - main-loop LDS padding + generalized coalescing_store_groups (memory-system efficiency)

  These are larger rewrites, but the code structure in wmma_main_loop.py and coalescing_store_wmma.py is modular enough that each can be added as a new control-field +
  branch rather than restructuring the whole generator.