# W-6: Strength-reduce wrw's per-iteration B-gather to incremental index update

## Status: IN PROGRESS — correctness issue identified but not yet resolved

## What was changed

### 1. Tunable added (Python + C++ + kernel name mangling)

**`python/igemm/igemm_base.py`:**
- Added `self.wrw_incremental_gather = utility_dict_with_default_t(tunable_dict)('wrw_incremental_gather', 0)` (after `wrw_streamk`, line ~955)
- Added `if tunable.wrw_incremental_gather: kernel_name += "_wig"` in `igemm_gtc_encode_kernel_name` (line ~1508)

**`driver/igemm_gtc_base.h`:**
- Added `int wrw_incremental_gather = 0;` to `igemm_gtc_tunable_t` struct (line ~244)
- Added parsing: `tunable.wrw_incremental_gather = sec.count("wrw_incremental_gather") > 0 ? sec.at("wrw_incremental_gather").get_int() : 0;` (line ~369)
- Added kernel name mangling: `if(tunable->wrw_incremental_gather) kernel_name += std::string("_wig");` (line ~607)

### 2. New kernarg: `ho`

The incremental ho-wrap needs `ho` (= ho_wo/wo) as a runtime value. There is no existing `s_ho` SGPR.

**`driver/igemm_wrw_gtc_driver.h`:**
- Added `int ho;` to `igemm_wrw_gtc_wmma_nhwc_karg_t` struct (offset 132)
- Added `karg.ho = ho;` in the karg setup

**`python/igemm/igemm_wrw_gtc_wmma_nhwc.py`:**
- Added `kas.append(amdgpu_kernel_arg_t('ho', 4, 132, 'by_value', 'i32'))` to `get_kernel_args`
- Added `self.s_ho = sym_t('s_ho', sseq(1))` to `kernel_sgpr_t`
- Added `s_load_dword s[s_ho], ... 132` in prologue (gated by `wrw_incremental_gather and row_stride == 1`)

### 3. New VGPRs

Added to `kernel_vgpr_t.__init__` before `v_end`, gated by `wrw_incremental_gather and row_stride == 1`:
- `v_inc_wo_idx` — persistent wo_idx
- `v_inc_ho_idx` — persistent ho_idx  
- `v_inc_n_idx` — persistent n_idx

### 4. Incremental gather implementation

**`_emit_b_gather(s_k_block_off, is_incremental=False)`:** Added `is_incremental` parameter.

**`_emit_b_gather_one_row`:** When `use_incremental` is True, delegates to `_emit_b_gather_incremental`. When False (first iteration), does the full div/rem as before, then seeds `v_inc_*` VGPRs.

**`_emit_b_gather_incremental`:** New method that:
1. Computes k_abs for k-tail check only (no division)
2. Increments `v_inc_wo_idx += kpb`
3. Magic div/rem by `wo` → carry, new_wo
4. `v_inc_wo_idx = new_wo; v_inc_ho_idx += carry`
5. Conditional ho-wrap (unrolled twice) using `v_cmp_lt_u32` + `v_cndmask_b32`
6. Recomputes hi_idx, wi_idx, v_flag, row_idx, v_addr_b from updated indices

**`move_slice_window_b_functor`:** Non-TDM path now calls `_emit_b_gather(s.s_tmp(), is_incremental=True)`.

### 5. VGPR pressure

- Baseline: 251 VGPRs (`.amdhsa_next_free_vgpr 251`)
- W-6: 254 VGPRs (`.amdhsa_next_free_vgpr 254`) — +3 VGPRs for v_inc_wo_idx/ho_idx/n_idx
- Well within gfx1250's 256 VGPR limit

## Correctness results

The incremental update logic is **provably correct** — exhaustive Python simulation over all 256 threads × all K-iterations for 14x14 (ho=14, wo=14, kpb=32) shows 0/200448 mismatches vs the full div/rem path.

### Hardware test results

| Shape | wo | ho | Baseline | W-6 |
|-------|-----|-----|----------|-----|
| 128×128×1×33 | 33 | 1 | valid:y | valid:y |
| 128×128×1×64 | 64 | 1 | valid:y | valid:y |
| 128×128×14×1 | 1 | 14 | valid:y | valid:y |
| 128×128×7×7 | 7 | 7 | valid:y | valid:y |
| 128×128×17×17 | 17 | 17 | valid:y | valid:y |
| 128×128×14×14 | 14 | 14 | valid:y | **valid:n** |
| 128×1024×17×17 | 17 | 17 | valid:y | **valid:n** |

### Root cause analysis

The failure occurs specifically when **both wo-wrap AND ho-wrap fire** in the same iteration. Shapes where only one type of wrap occurs (or neither) pass.

The original implementation used scalar-branch loops (`s_cbranch_vccnz`) which are **fundamentally incorrect** for per-lane divergent wrap counts — `s_cbranch_vccnz` branches if ANY lane's vcc bit is set, skipping the wrap for ALL lanes including those that need it.

The current implementation uses `v_cndmask_b32` for the ho-wrap (per-lane conditional), which should be correct. The wo-wrap uses a magic div/rem (unconditional, per-lane). The Python simulation confirms the logic is correct.

**The remaining hardware failure for 14×14 is not yet diagnosed.** The generated assembly was verified instruction-by-instruction:
- Magic div/rem encoding: correct (v_mul_hi_u32 + v_add + v_lshrrev + v_mul_lo_u32 + v_sub)
- v_cmp_lt_u32 with SGPR first operand: assembled correctly to e64 form
- v_cndmask_b32 with immediate operands: same encoding as baseline's v_flag computation
- s_ho kernarg: loaded at offset 132, s_wait_kmcnt 0x0 before use, value verified as 14

Possible causes not yet investigated:
1. Subtle VOP3 encoding issue with SGPR operands in v_cmp/v_sub on gfx1250
2. Instruction scheduling hazard between the magic div/rem and the ho-wrap comparisons
3. Interaction with the deferred context emission ordering for labels

## Performance

Not yet benchmarked due to the correctness issue.

## Files modified

1. `python/igemm/igemm_base.py` — tunable definition + kernel name mangling
2. `driver/igemm_gtc_base.h` — C++ struct field + parsing + kernel name mangling
3. `driver/igemm_wrw_gtc_driver.h` — karg struct field + driver assignment
4. `python/igemm/igemm_wrw_gtc_wmma_nhwc.py` — SGPR, VGPR, kernarg, prologue load, incremental gather implementation, move_slice_window_b modification
5. `config/w6_test_wig.config` — test config (incremental gather)
6. `config/w6_test_baseline.config` — test config (baseline)
7. `config/w6_test_wig_mtap.config` — test config (multi-tap, incremental)
8. `config/w6_test_baseline_mtap.config` — test config (multi-tap, baseline)
