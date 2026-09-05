I'll start by reading the key files to understand the codebase structure and the patterns I need to follow.Let me read more specific sections to understand the patterns.The file is large and the structural summary is eliding the ranges I need. Let me read the specific line ranges.Now let me read the rest of the dispatch logic and the non-atomic path to see how direct_store is selected:Now let me read the tunable parameter definitions in igemm_base.py to find where wmma_acc_f16 etc. are defined, and the kernel name encoding:Now let me see more context around the tunable definitions and the kernel name encoding:Now let me look at the C++ side for kernel name encoding and the `is_wmma_f16_acc` detection in conv_driver.cpp:Now let me see the full verification path in conv_driver.cpp for is_wmma_f16_acc and is_wmma_bf16_acc, plus the C++ kernel name encoding in igemm_gtc_base.h:Now let me look at how the config parser works in C++ and see the tunable parsing section:Let me also check the wrw verification path and how tensor_cast kernels work:Now let me also check the bwd verification path to see if it has a similar structure:Now let me check the bwd post verification path to see if it has the same structure:Now I have a thorough understanding of the codebase. Let me also check how `direct_store` is set on the ctrl in the Python codegen:Now let me check how ctrl fields like `atomic_pack_bf16` and `wmma_acc_f16` are set from the tunable:Now let me also check how the fwd kernel calls coalescing_store to see if it passes v_tid etc:Now I have a comprehensive understanding of the codebase. Let me also look at how the fwd kernel calls the coalescing_store, specifically the v_tid argument and the output stride:Now I have all the context I need. Let me also check how the output stride and the byte shift for output address is computed in fwd:Let me check the bwd and wrw coalescing_store calls:Now let me check the bwd ctrl setup and out_elem_byte_shift:Now I have a thorough understanding of the codebase. Let me search for any other places where the output element byte width is computed:Now I have a complete understanding. The key insight is:

For `wmma_fp16_output`, the **accumulator stays f32** (unlike `wmma_acc_f16` which changes the WMMA instruction itself), but the **output store** is packed fp16/bf16. This means:
- The output buffer is 2 bytes/element (like `wmma_acc_f16`)
- The `out_elem_byte_shift` in the prologue needs to be 1 (2 bytes/elem)
- The epilogue's `direct_store` path needs to pack pairs of f32 → fp16x2/bf16x2

The approach is: `wmma_fp16_output` should be treated exactly like `wmma_acc_f16`/`wmma_acc_bf16` for:
1. `dtype_alloc_byte` override in conv_driver.cpp
2. `out_elem_byte_shift` in fwd/bwd/wrw prologue
3. Verification path (use tensor_cast to expand fp16→fp32)

But the epilogue code is different — we use `v_cvt_pk_f16_f32`/`v_cvt_pk_bf16_f32` in the `_emit_direct_store` path.

Let me now implement all the changes. I'll start with the Python side:Now let me implement all the changes. Starting with Step 1: Add the tunable in igemm_base.py. I'll add it right after `direct_store`:Now add the kernel name suffix. Let me re-read the kernel name encoding section:Now update the `out_elem_byte_shift` in fwd, bwd, and wrw to include `wmma_fp16_output`. Also update the ctrl wiring. Let's start with fwd:Wait, I need to check the line numbers more carefully. The `ctrl_coalescing_store_wmma.wmma_acc_f16 = tunable.wmma_acc_f16 or tunable.wmma_acc_bf16` is line 331. Let me re-read:Wait, the edit replaced line 331 (which was `ctrl_coalescing_store_wmma.wmma_acc_f16 = tunable.wmma_acc_f16 or tunable.wmma_acc_bf16`) with the new line. That's wrong - I need to ADD the new line, not replace the existing one. Let me fix this:Good. Now do the same for bwd:Now update bwd's out_elem_byte_shift:Now update wrw. Let me read the wrw ctrl setup:Now update wrw's out_elem_byte_shift (two places):Now add the `wmma_fp16_output` field to the ctrl class in coalescing_store_wmma.py:Now Step 2: Implement the packed epilogue in `_emit_direct_store`. I need to modify the method to handle the `wmma_fp16_output` case. Let me re-read the current `_emit_direct_store` method to get the exact current line numbers:Now I understand the full `_emit_direct_store` method. The key changes needed for `wmma_fp16_output`:

1. Row-to-row byte stride: `s_lshl_b32 s[{s_tmp1}], s[{s_gemm_m_stride}], 1` (bf16/fp16 = 2 bytes/elem, not 4)
2. Byte address shift: `v_lshlrev_b32 v[{cur}], 1, v[{cur}]` (shift by 1, not 2)
3. Column offset: `i_rn * cxm.wave_tile_n * 2` (2 bytes/elem, not 4)
4. Pack pairs of f32 → fp16x2/bf16x2 using `v_permlane_xor_b32` + `v_cvt_pk_f16_f32`/`v_cvt_pk_bf16_f32`
5. Only even lanes store: EXEC mask, `global_store_dword`, restore EXEC
6. Need `v_tid` for the even/odd check — but `_emit_direct_store` doesn't currently receive `v_tid`!

Let me check how `_emit_direct_store` is called:I see that `_emit_direct_store` is called with `v_tmp3` and `v_tmp4` which are used for tail masking. For `wmma_fp16_output`, I need `v_tid` for the even/odd lane check, plus `v_tmp3`/`v_tmp4` for the cross-lane exchange scratch. Since `wmma_fp16_output` is mutually exclusive with `wmma_m_tail`/`wmma_n_tail` (I should add that assert), `v_tmp3`/`v_tmp4` are available for the packing scratch.

Wait, actually I need to think about this more carefully. The `wmma_fp16_output` is NOT mutually exclusive with `wmma_m_tail`/`wmma_n_tail` per the task spec. But the `atomic_pack_bf16` path IS mutually exclusive with them. Let me check if I should make `wmma_fp16_output` mutually exclusive with tails too.

Looking at the `atomic_pack_bf16` path: it uses `v_tmp3`/`v_tmp4` for the cross-lane exchange, and is mutually exclusive with `wmma_m_tail`/`wmma_n_tail` because those also need `v_tmp3`/`v_tmp4`. The same conflict applies to `wmma_fp16_output` in the direct_store path. So I should assert mutual exclusivity with tails for now.

But wait — the `_emit_direct_store` function already receives `v_tmp3`/`v_tmp4`. And the `__call__` method passes `v_tid` as a parameter. I need to pass `v_tid` to `_emit_direct_store` when `wmma_fp16_output` is set. Let me modify the call and the function signature.

Actually, let me think about this differently. Looking at the `atomic_pack_bf16` path pattern: it uses `v_tmp3` for the partner value (from `v_permlane_xor_b32`) and `v_tmp4` for the packed result. The even/odd check uses `v_tid`.

For the `_emit_direct_store` path with `wmma_fp16_output`:
- `v_tmp3` → partner value (from `v_permlane_xor_b32`)
- `v_tmp4` → packed result (from `v_cvt_pk_f16_f32`)
- `v_tid` → needed for even/odd check

Since `wmma_fp16_output` is mutually exclusive with `wmma_m_tail`/`wmma_n_tail` (both need `v_tmp3`/`v_tmp4`), I can use those slots. And I need to pass `v_tid` to `_emit_direct_store`.

Let me also check what the `__call__` method's `v_tid` parameter is in the direct_store case — it's passed but currently not forwarded to `_emit_direct_store`. Let me add it.

Now let me implement the modified `_emit_direct_store`. I'll add the `v_tid` parameter and a `precision` parameter (to know whether to use f16 or bf16 cvt):Now modify the `_emit_direct_store` method signature and implementation. Let me re-read the current state:Now I need to rewrite `_emit_direct_store` to handle the `wmma_fp16_output` case. The approach:

1. Add `v_tid=None` to the signature
2. When `ctrl.wmma_fp16_output`:
   - Use shift 1 (not 2) for byte stride and address computation
   - Column offset uses `* 2` (not `* 4`)
   - For each (i_rn, j): exchange partner value, pack, only even lanes store
   - Use `v_cvt_pk_f16_f32` for fp16, `v_cvt_pk_bf16_f32` for bf16
   - Use `v_tmp3` for partner value, `v_tmp4` for packed result
   - Use `v_tid` for even/odd checkNow Step 4: Update the C++ side. First, add the tunable field and parsing in `igemm_gtc_base.h`:Now add the parsing. Let me read the parsing section:Now add the kernel name suffix in the C++ encoding:Now update `conv_driver.cpp`: add `is_wmma_fp16_output` detection, update `dtype_alloc_byte`, update the COR-004 assertion, and update the verification paths.Now update the COR-004 assertion to include `wmma_fp16_output`, add the `is_wmma_fp16_output` bool, and update `dtype_alloc_byte`:Now add `is_wmma_fp16_output` and update `dtype_alloc_byte`: