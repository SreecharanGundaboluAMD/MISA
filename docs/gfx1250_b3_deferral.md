# B3 Deferral: v_xor_b32 → ±delta Buffer Toggle

**Date:** 2026-09-03  
**Report ref:** §7/B3, OPT-1  
**Status:** Deferred

## Background

B2 (commit `74bbdc3`) threaded `lds_bytes_per_row` through the fwd generator,
making `lds_row_pad` functional. However, `lds_single_size` still uses
`igemm_next_pow2(lds_a_size + lds_b_size)` because the double-buffer toggle
in `wmma_main_loop.py:478-485` uses `v_xor_b32` with `lds_single_size`, which
only works for power-of-2 values.

With `lds_row_pad=16` on the 128×128×32 fp16 tile:
- Padded sum: `128*80 + 128*80 = 20480`
- `next_pow2(20480) = 32768` (60% waste)
- Double-buffered LDS: `32768 * 2 = 65536` (vs 32768 unpadded)

B3 would replace `v_xor_b32` with `±delta` add, allowing `lds_single_size`
to be the exact padded sum (20480), saving 24 KB/workgroup.

## Why Deferred

1. **No scratch VGPR available.** The `emit_buffer_switch()` function in
   `wmma_main_loop.py` operates on `v_sst_a_os`, `v_sld_a_os`, `v_sld_b_os`.
   The `±delta` approach needs a scratch VGPR for the `v_cndmask` result (to
   select between +K and -K). `v_a[0]` is technically free at the switch
   point (WMMA compute is done, next iteration's LDS load hasn't happened),
   but this is fragile — future `local_prefetch_num=2` or interleave changes
   could make `v_a` live at the switch point.

2. **No occupancy benefit.** The `next_pow2` rounding wastes 24 KB LDS, but
   occupancy is VGPR-bound (252/256 VGPRs, 4 blocks/CU per
   `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`). The wasted LDS
   doesn't reduce occupancy. It would only matter if a future config reduces
   VGPR usage below the 256 limit, making LDS the new limiter.

3. **Report says "prefer" not "must".** The report's OPT-1 risk/tradeoff
   section frames B3 as a preference, not a requirement: "Prefer replacing
   the `v_xor_b32` toggle with a `±delta` add so a non-power-of-2 buffer
   stride works (1 extra VALU/iteration, saves 24 KB)."

4. **Complexity cost.** The `±delta` approach requires either:
   - A scratch VGPR (not available without aliasing `v_a[0]`), or
   - An SGPR pair holding `K` and `-K` (consuming 2 SGPRs), or
   - A multi-instruction sequence using `v_cmp` + `v_cndmask` + `v_add_u32`
     (5 instructions vs 3 `v_xor_b32`, +2 VALU/iteration)

   The report's "1 extra VALU/iteration" estimate appears optimistic — the
   minimal correct implementation is +2 VALU/iteration (shared `v_cmp` +
   shared `v_cndmask` + 3×`v_add_u32` vs 3×`v_xor_b32`).

5. **Large immediate limitation.** `v_cndmask_b32 dst, src0, src1, vcc`
   requires both sources as VGPRs or inline constants. For non-power-of-2 K
   (e.g., 20480), neither K nor -K fits in an inline constant, so both must
   be pre-loaded into VGPRs or SGPRs.

## Revisit Criteria

B3 should be revisited when any of the following becomes true:
- LDS capacity becomes the occupancy limiter (VGPR usage drops below ~240)
- A config needs `lds_row_pad` on a tile where `next_pow2` would exceed the
  64 KB/workgroup hardware LDS limit
- `local_prefetch_num=2` or interleave makes `v_a` live at the switch point,
  requiring a dedicated scratch VGPR anyway (at which point the marginal
  cost of using it for `±delta` is zero)
