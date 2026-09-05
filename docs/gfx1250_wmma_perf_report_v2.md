# gfx1250 WMMA Convolution — Performance Report v2

**Supersedes** `docs/gfx1250_wmma_perf_report_2026-09-02.md` (v1).
**Date:** 2026-09-03 · **HEAD:** `2af9f24` · **Hardware:** gfx1250, 256 CU, 4 SIMD/CU, 1024 VGPR/SIMD (wave32), 320 KB LDS + 64 KB WGP$/CU, 64 LDS banks × 4 B.
**New input:** `docs/gfx1250_gemm_optimization_guide.md` (AMD "Practical GEMMs / MI400" deck + transcript).

Everything below is measured on this machine unless tagged `[INFERENCE]`.

---

## 1. Executive summary

The v1 recommendations were implemented across 20 commits (`3dd26ce..2af9f24`). **They worked**, measured against the configs a user actually got before the campaign:

| shape | dir | before campaign | HEAD best | Δ |
|---|---|---|---|---|
| `n64 c512 28×28 k512 3×3` | wrw | 23.3 | **412.3** | **17.7×** |
| `n128 c1024 17×17 k1024 1×1` | wrw | 72.4 | **340.7** | **4.7×** |
| `n128 c1024 17×17 k1024 1×1` | fwd | 397 | **502.3** | **+26 %** |
| `n128 c1024 17×17 k1024 1×1` | bwd | 436 | **492.1** | **+13 %** |
| `n256 c2048 14×14 k2048 1×1` | fwd | 640 | 647.3 | +1 % |
| `n64 c512 28×28 k512 3×3` | fwd | ~690 | 705.4 | +2 % |

Against the measured WMMA burst ceiling (1411 TFLOP/s) that is 46 % (fwd), 42 % (bwd), 35 % (wrw) — up from 33 %/31 %/1–16 %.

**But four defects were introduced or left behind, one of them a blocker:**

| # | Defect | Severity |
|---|---|---|
| **R1** | `config/igemm_wrw_gtc_gfx1250_nhwc_fp16_all.config` **does not build at HEAD** (5 sections fail an unconditional assert added by `6fcde3a`). AGENTS.md was simultaneously repointed at the `_all` configs by `81f1e08`. | **blocker** |
| **R2** | `wmma_fp16_output` (+8.7–14.5 %, validated) is **reachable from zero configs** — implemented, measured, inert. | high |
| **R3** | `fp_factor` 8 → 9 (`d91a7ae`) makes reported efficiency non-physical and clock-dependent. | medium |
| **R4** | wrw stream-K is now correct but **40 % slower than plain split-K**, and still emits `valid:n` on multi-tap shapes. | medium |

**The single biggest opportunity is unchanged and untouched: compute/memory overlap.** v1 measured `no_wmma (0.134 ms) + wmma_only (0.055 ms) ≈ baseline (0.195 ms)` — overlap ≈ 3 %, i.e. **1.45× on the table**. Guide §14 names the mechanism precisely, and the repo already emits the primitive incorrectly (see §5.1).

---

## 2. Commit-by-commit review

| # | commit | verdict | note |
|---|---|---|---|
| 1 | `3dd26ce` wrw `gemm_k_global_split` default | **OK** | 4 default configs + a real driver warning when `grid < CU count`. This is the 4.7–17.7× win. |
| 2 | `d91a7ae` peak denominator 8 → 9 | **SUSPECT** | see §3.1 |
| 3 | `81f1e08` repoint quick-start to `_all` | **OK-BUT** | correct direction, but see R1 and §3.2 |
| 4 | `ba2ecd0` benchmark `wmma_acc_f16/bf16` | **OK** (docs) | fp16 f16acc +12.2 %/+5.3 % `valid:y`; **bf16 bf16acc `valid:n`** — a known-bad path is still shipped in configs |
| 5 | `5a0f370` `lds_row_pad` tunable | **OK** | Python + C++ struct, parser, and `_ldsrp` mangling all in sync; default 0 |
| 6 | `74bbdc3` thread `lds_bytes_per_row` (fwd) | **OK** | exactly the v1 design, plus `%16==0` and `gcd(stride_dwords,64)==4` asserts. `bytes_per_row` correctly retained as the *global* stride |
| 7 | `d1135f6` B3 deferral doc | **OK** (docs) | `next_pow2(20480)=32768`, 60 % LDS waste — deferral is justified since occupancy is VGPR-bound |
| 8 | `c93e7db` B4 analysis doc | **STALE** | deferred B4; commit 9 then implemented B4 and deleted this doc. Net effect fine, but the intermediate commit is noise |
| 9 | `6fcde3a` `lds_row_pad` for bwd/wrw | **BROKEN** | correct *design* (separate `lds_row_pitch_b` for the transposed [K][N] region, gcd asserts on both A and B pitches) but the `threads_per_krow` assert is **outside** the `if lds_row_pad > 0` guard → breaks every tile with `gemm_k_per_block > gemm_n_per_block`. See §3.3 |
| 10 | `0dd5e99` emit `lds_row_pad=16` sections | **OK-BUT-THIN** | 2 padded sections per `_all` config; **0** in `igemm_fwd_gtc_gfx1250_nhwc_fp16_128x128_all.config` (84 sections). Also leaves 4 stray `config/w6_test_*.config` artifacts |
| 11 | `1fd71fb` stream-K static shard indexing | **OK-BUT** | fixes `valid:n` (verified), but see §3.4 — it is no longer stream-K, and it is now slower than split-K |
| 12 | `e8139fc` remove dead atomic-claim symbols | **OK** | pure cleanup following 11 |
| 13 | `442d0e0` transposed pitch sweep (W-3) | **OK** (docs) | measurement commit |
| 14 | `094c9eb` 64×64 vs 128×128 wrw (W-4) | **OK** (docs) | measurement commit |
| 15 | `1b95803` un-gate `ds_load_tr_b` for streamk | **OK** | −0.98 % / +2.69 %; +5–8 % on the 64×64 tile. Small but correct and validated 3× |
| 16 | `6fe6977` W-6 gather (WIP, ho-wrap bug) | **OK** (superseded) | self-labelled WIP; should ideally have been squashed into 17 |
| 17 | `08443f3` W-6 incremental gather | **OK** | +0.4 %/+0.3 % on 1×1, **+19.9 % on 3×3 multi-tap**, all `valid:y`. v1 predicted 7 % — correct for 1×1, and multi-tap benefits much more |
| 18 | `215d30d` E-4 workgroup swizzle | **OK-BUT-INERT** | measurements are noise: +0.76 %, −1.14 %, −1.41 %. 0 configs enable it. Correctly left off; **recommend removing** rather than carrying dead tunable surface |
| 19 | `4ef933d` W-6 re-benchmark | **OK** (docs) | |
| 20 | `2af9f24` C1–C2 fp16/bf16 output | **OK-BUT-INERT** | real work: `v_permlane_xor_b32` + `v_cvt_pk_f16_f32`, even lanes only → **halves both store bytes and store instructions** (128 → 64 `global_store_dword`); `dtype_alloc_byte` override dropped; `_f16o` mangling in sync; 1.087–1.145× validated. **But 0 configs emit `wmma_fp16_output`** |

---

## 3. Defects in detail

### 3.1 R3 — `fp_factor = 9` is not a physical rate

`driver/conv_driver.cpp:197,212`. Peak `= sclk × 256 CU × 128 lanes × 2 × fp_factor`. `fp_factor` is *MACs per lane per clock*:

```
fp_factor = 8  ->  8192 MAC / 32 lanes / 8 = 32.00 cycles per v_wmma_f32_16x16x32_f16   <- clean
fp_factor = 9  ->  8192 MAC / 32 lanes / 9 = 28.44 cycles                               <- not realizable
```

The value was chosen to make the denominator equal v1's *measured* 1411 TFLOP/s at the clock the driver happened to read (`dev_prop.clockRate` = 2391 MHz). That is a software result and a clock reading, not a hardware constant. Consequences: efficiency now drifts with clock, and any kernel that beats 1411 will report > 100 %.

**Fix:** revert to `fp_factor = 8` (peak 1254 TFLOP/s at 2.391 GHz). Report the 1411 burst separately as "best achieved", and if the discrepancy matters, fix the clock source (`IGEMM_SCLK_MHZ` already exists) rather than the FLOP constant.

### 3.2 R1 — the wrw master config does not build (blocker)

```
$ python3 igemm_codegen.py config/igemm_wrw_gtc_gfx1250_nhwc_fp16_all.config
AssertionError: threads_per_krow(0) must be > 0 and a power of 2
```
Bisected to `6fcde3a`; `6fcde3a~1` builds cleanly. Five sections fail — all the deep-K wrw tiles, which are exactly the tiles wrw needs:

| section | M | N | K | `bytes_per_row` | `N·data_byte` | `threads_per_krow` |
|---|---|---|---|---|---|---|
| 2 | 32 | 32 | 96 | 192 | 64 | **0** |
| 3–5 | 64 | 64 | 128 | 256 | 128 | **0** |
| 6 | 64 | 64 | 256 | 512 | 128 | **0** |

Root cause (`igemm_wrw_gtc_wmma_nhwc.py`): `threads_per_krow_{a,b} = (gemm_{m,n}_per_block * data_byte) // bytes_per_row` truncates to 0 whenever `gemm_k_per_block > gemm_{m,n}_per_block`, and the assert sits **outside** the `if tunable.lds_row_pad > 0:` guard, so it fires on unpadded configs that previously worked.

**Fix (small):** move the `threads_per_krow` computation and both asserts inside `if tunable.lds_row_pad > 0:`; separately, either support `threads_per_krow < 1` (several K-rows per thread) or assert `lds_row_pad == 0` for those tile shapes with a clear message. Then re-run `script/build_and_filter_configs.py`.

### 3.3 R2 — the epilogue win is unreachable

`wmma_fp16_output` is validated at 1.087–1.145× across three shapes and halves both store bytes and store instructions — and **no config sets it**:

```
lds_row_pad         16 configs / 28 sections
wmma_fp16_output     0 configs /  0 sections   <-- inert
workgroup_swizzle    0 configs /  0 sections   <-- inert (correctly: its measurements are noise)
```
`lds_row_pad` coverage is also thin: 2 sections per `_all` config, 0 in the 84-section `_128x128_all`.

**Fix:** emit `wmma_fp16_output=1` and `lds_row_pad=16` variants from `script/generate_all_configs.py` / `build_gfx1250_master_configs.py` for every tile shape, then let the driver's fastest-tunable search decide. This is the cheapest remaining win in the repo.

### 3.4 R4 — stream-K: correct now, but slower than what it replaced

| | pre-fix (`valid:n`) | HEAD (`valid:y`) | split-K at HEAD |
|---|---|---|---|
| `n128 c1024 17×17 1×1` | 457.2 | **290.9** | **340.7** |
| `n256 c2048 14×14 1×1` | — | **229.3** | **493.7** |
| `n64 c512 28×28 3×3` | n/a | **2 of 2 sections `valid:n`** | 412.3 |

Two problems. First, `tile_idx = bz + iter * grid_z` is *static round-robin persistent split-K*, not Stream-K — the dynamic tail rebalancing that gives Stream-K its name is gone, and with it the performance. Second, the path still produces wrong answers on multi-tap shapes: `wrw_streamk` requires `nxe==0` (a Python-side assert in `igemm_base.py`), but the **driver's `tunable_is_valid` does not reject those tunables at runtime for `y,x > 1`**, so they run and silently fail.

**Fix:** (a) add the `nxe==0` rejection to `igemm_wrw_gtc_driver.h::tunable_is_valid`; (b) either restore dynamic claiming with a correct broadcast (the ISA has `ds_bpermute` and, better, `DS_ATOMIC_ASYNC_BARRIER_ARRIVE` — guide §16), or delete the path and keep split-K, which is faster today.

---

## 4. Where HEAD actually stands

Best `valid:y` over each master config, fp16, `-V 1`:

| shape | fwd | bwd | wrw |
|---|---|---|---|
| `n256 c2048 14×14 k2048 1×1` | 647.3 | 592.6 | 493.7 |
| `n128 c1024 17×17 k1024 1×1` | 502.3 | 492.1 | 340.7 |
| `n64 c512 28×28 k512 3×3` | 705.4 | 616.5 | 412.3 |

vs the 1411 TFLOP/s WMMA burst ceiling: **35–50 %.** Half the machine is still idle in the best case.

Two data-quality issues worth triaging: `igemm_fwd_gtc_gfx1250_nhwc_fp16_128x128_all.config` reports **16 of 84 sections `valid:n`** on both 1×1 shapes, and the bf16 `bf16acc` configs are known-bad per `ba2ecd0`'s own findings. Shipping master configs with known-failing sections erodes trust in the fastest-tunable search.

---

## 5. New opportunities from the optimization guide

The guide independently confirms the v1 diagnosis ("WMMA is not issued back-to-back → identify waits between WMMAs, pipeline next-stage TDM while computing, replace broad waits with the correct bifurcated counter, **use split barriers and required scheduling fences**", §23). Three items are new, cheap, and directly grounded in the current generated code.

### 5.1 ★ Split barriers are emitted but the gap is empty (guide §14)

Every barrier site in the generated kernels:
```asm
s_barrier_signal -1
s_barrier_wait   -1     ; <-- immediately adjacent, at all 3 sites
```
The repo pays for the split-barrier primitive and captures none of its benefit. Guide §14: *"roughly 100 clocks of useful work may fit in the signal-to-wait gap … the next K-stage TDM can be issued there"*, and it requires `sched_barrier(0)` fences around the sequence on gfx1250.

This is a **targeted, low-risk subset of v1's OPT-4** — no MBB scheduler rewrite needed. Move the next tile's `move_slice_window` + `global_load` issue (already hoisted, `wmma_main_loop.py:626-629`) into the signal→wait gap. Expected: partial recovery of the 1.45× overlap gap. **Try this before D1.**

Files: `python/operations/wmma_main_loop.py:588-589` (the barrier pair) and `:592-629` (`can_hoist` block).

### 5.2 Temporal hints and scope are entirely unused (guide §18)

Generated main-loop loads carry no cache policy:
```asm
global_load_dwordx4 v[v_gld_a:v_gld_a+3], v[v_addr_a:v_addr_a+1], off offset:0
```
No `th:` or `scope:` anywhere in `igemm_fwd_gtc_wmma_nhwc.py` (2 occurrences exist in `coalescing_store_wmma.py`, atomic path only). Guide §18 recommends `RT_NT` for cooperative A/B macro-tile loads (temporal near, non-temporal in L2 since they are not re-read after LDS staging) and `NT` for C stores. Given global loads are 46 % of fwd runtime and the epilogue writes a stream that is never re-read, this is a **one-line-per-emitter change** with real upside on L2 pressure. Must be benchmarked, not applied mechanically.

### 5.3 L2 prefetch is unused — and the assembler is already asking for it (guide §19)

`GLOBAL_PREFETCH_B8` is fire-and-forget: no wait counter, no stall on miss, no VGPR/LDS cost; the guide issues it two K-stages ahead. Nothing in the WMMA path emits it. Separately, `clang` warns on **every** generated kernel:

> *kernel … does not begin with the required prologue sequence: S_MOV_B64 followed by V_NOP and GLOBAL_PREFETCH_B8*

plus ISA §2.4 requires 256 bytes of `S_CODE_END` padding past every shader. Both are unaddressed. Fix the prologue/padding first (robustness), then evaluate prefetch — the guide notes it helps compute-bound kernels with visible load latency, which matches the fwd 1×1 profile.

### 5.4 Guide items that confirm existing choices — no action

- **§12 bifurcated wait counters:** already done well (`s_wait_dscnt`/`loadcnt`/`asynccnt`/`tensorcnt` all used correctly).
- **§20 "larger tiles regress → check VGPR pressure, tail waves":** matches v1's measured 256×256 null result (639 vs 635).
- **§9.2 wave32 fragment mapping:** already empirically verified in this repo.

---

## 6. Revised priority list

**P0 — defects (do first, all small)**
1. **R1** Fix the `threads_per_krow` assert scope; restore the wrw master-config build. *Blocker.*
2. **R2** Emit `wmma_fp16_output=1` and broaden `lds_row_pad=16` coverage from the config scripts. *Validated 1.087–1.145×, currently unreachable.*
3. **R3** Revert `fp_factor` to 8; fix the clock source instead.
4. **R4** Reject `wrw_streamk` for `nxe!=0` in `tunable_is_valid`; then either restore dynamic claiming or delete the path (split-K is faster today).
5. Triage the 16 `valid:n` sections in `fwd_..._128x128_all`; drop bf16 `bf16acc` from shipped configs; delete `config/w6_test_*.config`; remove the inert `workgroup_swizzle` tunable.

**P1 — the remaining big win (1.45×, still untouched)**
6. **Split-barrier pipelining (guide §14)** — move the hoisted global load into the `signal → wait` gap with `sched_barrier(0)` fences. Small, targeted, try before 7.
7. **v1 OPT-4 / D1–D3** — MBB + `simple_interleave_scheduler_t` in `wmma_main_loop.py`, mirroring `mfma_main_loop.py`; plus making `local_prefetch_num=2` reachable at `gemm_k_per_block == inst_wmma.k`. Acceptance test: `baseline → max(no_wmma, wmma_only)` instead of their sum.

**P2 — cheap, guide-derived**
8. Temporal hints `RT_NT` on A/B loads, `NT` on C stores (§5.2).
9. gfx1250 shader prologue + `S_CODE_END` padding (§5.3), then `GLOBAL_PREFETCH_B8` two stages ahead.
10. True C2: widen the epilogue store beyond `global_store_dword` (packing halved the count; a lane-permute to 4 contiguous columns would enable `dwordx4`), or test claused stores (guide §17.1).

**P3 — larger mechanisms, unchanged from v1**
11. `async_global_load` into fwd configs (removes the `ds_write` phase, frees 32 VGPRs).
12. TDM beyond `nxe==0`, with `D#.pad_amount`/`pad_interval` — hardware LDS padding would subsume `lds_row_pad` at zero instruction cost (guide §13).
13. Cluster multicast + cluster barriers (guide §15/§16) — targets the 46 % global-load term; needs cluster dispatch.
14. bwd `stride==1 && dilation==1` fast path.
15. Small-channel shapes (`c=64` still ~86 TFLOP/s) — dominated by prologue, tap scaffolding and epilogue, not the main loop.

---

## 7. Validation protocol for the next round

Unchanged from v1 and still the highest-signal tool: the main-loop ablation harness (copy a generated output dir, delete instruction classes from **only** the `_wmma_body` region, re-assemble, run `-V 0`). Land it as `script/ablate_main_loop.py`.

Standing regression set:
```
-n 256 -c 2048 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0   # large 1x1, feed-limited
-n 128 -c 1024 -H 17 -W 17 -k 1024 -y 1 -x 1 -p 0 -q 0   # medium 1x1
-n 64  -c 512  -H 28 -W 28 -k 512  -y 3 -x 3 -p 1 -q 1   # 3x3, compute-limited
-n 32  -c 256  -H 56 -W 56 -k 256  -y 3 -x 3 -p 1 -q 1   # 3x3, large spatial
-n 128 -c 64   -H 56 -W 56 -k 64   -y 1 -x 1 -p 0 -q 0   # small-channel / tail-heavy
```
Per guide §22, every candidate must record correctness, median **and tail** latency, VGPR/LDS use, occupancy, and tail-tile behaviour — not throughput alone. Two process rules that would have caught R1 and R2:

- **Every commit that adds a tunable must also add a config that sets it**, or state explicitly that it is deferred.
- **`script/build_and_filter_configs.py` over all `config/*.config` must pass before merge.** R1 would have been caught immediately.
