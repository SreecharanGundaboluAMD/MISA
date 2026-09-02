# gfx1250 WMMA Convolution Pipeline — Performance Engineering Report

**Date:** 2026-09-02  **Hardware:** gfx1250, 256 CU, 4 SIMD/CU, 1024 VGPR/SIMD (wave32), 320 KB LDS + 64 KB WGP$ per CU, 64 LDS banks × 4 B.
**Method:** every number below is *measured on this machine*, either with `conv_driver.exe` or by surgically ablating the generated `.inc` assembly and re-assembling with the repo's own toolchain. Ablated kernels are numerically wrong by construction; only their timing is used. Claims not backed by a measurement are tagged `[INFERENCE]`.

---

## 1. Executive summary

**The kernel is not WMMA limited, not latency limited, not occupancy limited, and not bandwidth limited. It is limited by an almost total absence of compute/memory overlap.**

The decisive measurement (fwd fp16 128×128×32, `n128 c1024 17×17 k1024 1×1`):

| variant | cost | note |
|---|---|---|
| baseline | 0.195 ms | 397 TFLOP/s |
| WMMA burst deleted, all memory kept | 0.134 ms | memory pipeline alone |
| all memory deleted, WMMA burst kept | 0.055 ms | compute alone |
| **0.134 + 0.055 = 0.189 ms** | | **≈ baseline (0.195)** |

Compute and memory are **additive, not overlapped** — measured overlap ≈ 3 %. If they overlapped perfectly the same kernel would run at 0.134 ms (**1.45×**).

Second decisive measurement: a pure back-to-back `v_wmma_f32_16x16x32_f16` 4×4 burst reaches **1411 TFLOP/s** (fwd) / **1423 TFLOP/s** (bwd) on this part. Best *tuned* result today over the full `_all` config space is 640 TFLOP/s (large 1×1) and 462 TFLOP/s (`n128 c1024 17×17`) — i.e. **33–45 % of the achievable WMMA rate**. (Note: `driver/conv_driver.cpp` assumes a 1254 TFLOP/s peak, so its reported efficiency % is ~12 % optimistic.)

Ranked findings, all measured:

1. **LDS bank conflicts are real and free to fix** — the 64 B row stride aliases 32 lanes onto 4 bank groups. Padding to 80 B: **+9.4 %** (128×128 1×1), **+20.7 % / +26.6 %** (64×64). Proven by a 4-point stride sweep that tracks the bank model exactly.
2. **The epilogue costs 19 %** of a K=1024 1×1 conv: 128 scalar `global_store_dword` per lane, writing **fp32 output from an fp16 kernel**.
3. **`lds_double_buffer` swings performance by ±27–37 %, shape-dependently**, and the canonical config named in `AGENTS.md` picks the wrong side on large 1×1 shapes (448 vs 616 TFLOP/s).
4. **No instruction-level interleaving of memory into the WMMA burst.** `python/codegen/scheduler.py` + the MBB IR already do this for XDLOPS (`mfma_main_loop.py`); `wmma_main_loop.py` never adopted it. This is the 1.45× headroom.
5. **Cluster/multicast loads (ISA §10.7) — a GEMM-specific hardware feature — are entirely unused**, while global loads account for **46 %** of runtime.

Explicitly ruled out by measurement (do **not** spend effort here): load latency, LDS latency, occupancy, LDS capacity, address coalescing, `DISABLE_XDL_ARB_STALL`, larger macro-tiles, deeper `gemm_k_per_block`. See §3.2.

---

## 2. WMMA pipeline mapping

### 2.1 Shape under study
`igemm_fwd_gtcw_nhwc_fp16_..._bt128x128x32_wt16x16_wr4x4`: macro-tile 128×128×32, `block_size=128` (4 waves), wave tile 64×64, `wave_repeat 4×4`, `waves_per_m = waves_per_n = 2`. 252 VGPR, 4 blocks/CU, **4 waves/SIMD** (measured via `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`).

### 2.2 Data path
```
global (NHWC)
  → global_load_dwordx4 ×8   [+ v_cmpx EXEC predication + 16-dword VGPR zero-fill]
  → v_gld_a/v_gld_b (32 VGPR staging)
  → ds_write_b128 ×8         [LDS, 64 B row stride, UNPADDED, XOR double-buffered]
  → ds_read_b128 ×16
  → v_a/v_b (32+32 VGPR)
  → v_wmma_f32_16x16x32_f16 ×16
  → v_c (128 VGPR, live across the whole tap loop)
  → epilogue: 128 × global_store_dword (fp32)
```

### 2.3 Steady-state main loop (`wmma_main_loop.py:592-650`, `can_hoist` path)
```
body:
  s_wait_dscnt 0x0 ; s_barrier_signal -1 ; s_barrier_wait -1
  s_sub_i32 / s_cmp_gt_i32 / s_cbranch_scc0 -> _last
  4×  v_add_co_u32/v_add_co_ci_u32          ; move_slice_window A+B (64-bit carry chain)
  8×  v_dual_mov_b32 v[v_gld_a+n], 0        ; zero-fill (no buffer-resource OOB clamp)
  1×  v_cmpx_le_u32 1, v[v_flag]
  8×  global_load_dwordx4                   ; next tile
  1×  s_mov_b32 exec_lo, -1
  16× ds_read_b128                          ; this tile, from LDS
  4×  s_wait_dscnt 0x6/0x4/0x2/0x0          ; Phase-71 partial-wait ladder
  16× v_wmma_f32_16x16x32_f16
  s_wait_loadcnt 0x0
  8×  ds_write_b128                         ; next tile into LDS
  3×  v_xor_b32 ..., 16384                  ; double-buffer switch
  s_branch body
```
**Structural observation:** the memory instructions and the WMMA instructions occupy *disjoint program regions*, and the region boundary is a workgroup-wide barrier. A wave can never fill its own WMMA execution shadow with its own memory work, and the barrier re-synchronises all 4 waves of a workgroup into the same phase every iteration.

### 2.4 LDS addressing (`igemm_fwd_gtc_wmma_nhwc.py:792, :819-821`)
```
v_sst_os   = tid * 64                                        ; thread tid owns LDS row tid
v_sld_a_os = (wave_m*64 + (tid&15)) * 64 + ((tid>>4)&1) * 32
v_sld_b_os = (wave_n*64 + (tid&15)) * 64 + ((tid>>4)&1) * 32
```
No padding, no swizzle, no permutation anywhere in the main loop. The only `xor` is the double-buffer base toggle.

### 2.5 Direction differences
- **fwd** — A and B both untransposed, plain `ds_read_b128`. Only direction that can use the Phase-71 partial-wait ladder.
- **bwd** — B (weight) transposed via hardware `ds_load_tr16_b128` (zero packing VALU). Expensive divide-based gather hoisted **per tap**; in-loop cost is 4 VALU pointer bumps.
- **wrw** — **both** operands transposed (16 × `ds_load_tr16_b128`/iteration), and `move_slice_window_b_functor` re-runs the **entire spatial gather every K-iteration** (~29 VALU + 5 SALU incl. two magic div/rem) because GEMM_K is spatial. wrw is decisively the most overhead-heavy direction. Every `wrw_streamk` config is additionally excluded from the `ds_load_tr_b` default (`igemm_base.py:893-895`) and therefore pays the manual 128-VALU pack path.

---

## 3. Root causes

### 3.1 Cycle attribution (fwd fp16 128×128, `n128 c1024 17×17 k1024 1×1`, baseline 0.195 ms)

| ablation | cost (ms) | speed-up | attributed share |
|---|---|---|---|
| 8 × `global_load_dwordx4` + zero-fill + `v_cmpx` | 0.106 | **1.84×** | **45.6 %** |
| 128 × epilogue `global_store_dword` | 0.159 | **1.23×** | **19 %** |
| all LDS (16 read + 8 write) | 0.163 | 1.20× | 16.4 % |
| 16 × `ds_read_b128` only | 0.175 | 1.11× | 10.3 % |
| `s_barrier_signal`/`s_barrier_wait` | 0.188 | 1.04× | 3.6 % |
| 8 × `ds_write_b128` only | 0.191 | 1.02× | 2.1 % |
| 16-dword zero-fill only | 0.193 | 1.01× | 1.0 % |
| *everything except WMMA + loop control* | 0.055 | 3.55× | 71.8 % |

bwd (same shape, `-F 2`, baseline 0.178 ms): global loads 23 %, LDS reads 12.4 % (A 15.7 %, B `ds_load_tr16` 9.6 %), `ds_write` 7.3 %, WMMA-only 0.054 ms → **1423 TFLOP/s ceiling reproduced independently**.

### 3.2 Hypotheses tested and **rejected** — do not pursue

| hypothesis | test | result |
|---|---|---|
| Global-load latency exposed at `s_wait_loadcnt 0x0` | delete the wait | **1.026×** — latency is already covered |
| LDS load-to-use latency | delete all `s_wait_dscnt` | **1.010×** |
| Uncoalesced global loads (32 divergent lines/instr) | force lane-contiguous addresses + widen instruction span | **1.065×** — divergence is a minor term |
| Occupancy / LDS capacity | rebuild identical code with 32 KB → 64 KB LDS | **1.00×** (399.4 vs 398.5); occupancy stays 4 blocks/CU (VGPR-bound) |
| ISA §5.7.2.1 `DISABLE_XDL_ARB_STALL` | `s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE,2,1), 1` at kernel entry | **0 %** on both fwd (396.1→395.9) and bwd (436.4→433.4). Matches the ISA's own caveat ("beneficial primarily when a single wave is running on a SIMD"); we run 4. |
| Larger macro-tile raises arithmetic intensity | bf16 256×256 vs 128×128, `n256 c2048 14×14 k2048` | **639.4 vs 634.6** — no gain |
| Deeper `gemm_k_per_block` amortises the barrier | fp16 128×128, k=32 → 64 (± `main_loop_interleave`) | **614 → 442 / 462** (−28 %/−25 %) at identical VGPR (252) and LDS |

### 3.3 The primary root cause: zero compute/memory overlap
Since latency, occupancy and bandwidth are all excluded, and since `no_wmma + wmma_only ≈ baseline`, the loss is **structural scheduling**: the barrier-delimited phase layout of §2.3. A wave issues 16 WMMA back-to-back (arbiter stalls it for the full burst, ISA §5.7.2.1), then issues its memory work; cross-workgroup filling cannot compensate because all resident workgroups are rate-limited by the same per-CU LDS and TA/L1 units and are re-synchronised by the barrier each iteration.

### 3.4 Secondary root cause: LDS bank aliasing
64 banks × 4 B; the 64 B (16-dword) row stride puts lane `L`'s first dword in bank `(L*16) mod 64` → only 4 distinct bank groups across 32 lanes. Model predicts 4-way conflict on `ds_read_b128` and 8-way on `ds_write_b128`. **Verified by a stride sweep at constant LDS allocation (64 KB) and constant instruction count:**

| row stride | dwords | gcd(stride,64) | model | 1×1 TFLOP/s | 3×3 TFLOP/s |
|---|---|---|---|---|---|
| 64 B (shipped) | 16 | 16 | 4-way conflict | 397.1 | **689.9** |
| **80 B** | 20 | 4 | **conflict-free** | **435.5** | 553.4 |
| **112 B** | 28 | 4 | **conflict-free** | **433.0** | 561.6 |
| 128 B | 32 | 32 | 8-way (worse) | 395.2 | 416.8 |

The 1×1 column tracks the bank model exactly. The 3×3 column does **not** — the shipped 64 B stride wins there, for a reason this session did **not** isolate (LDS *allocation* size was independently ruled out by the 64 KB control). This is why padding must ship as a **tunable, not a default**.

### 3.5 Tertiary root cause: the epilogue writes fp32
`driver/conv_driver.cpp:~831`: `dtype_alloc_byte = ... (is_wmma ? sizeof(float) : data_byte)`. An fp16 fwd conv therefore writes a **fp32** output tensor — 2× the store bytes of a real fp16 convolution. Combined with the WMMA D-matrix layout (each lane holds 8 elements down a *column*, so consecutive `v_c` registers are consecutive rows and cannot be merged), the epilogue emits **128 scalar `global_store_dword`** per lane (512 B/lane, 65536 B/workgroup), each producing two disjoint 64 B segments. Measured: **19 % of runtime** on a K=1024 shape — and proportionally far worse on shallow-K shapes.

---

## 4. Top optimisation opportunities (ranked by measured impact)

### OPT-1 — LDS row padding as a tunable  ★ measured +9 % to +27 %
- **Why:** §3.4. 64 B stride aliases 32 lanes onto 16 of 64 banks.
- **Evidence:** stride sweep table §3.4; `+9.4 %` (128×128 1×1), `+20.7 %` (64×64, `n128 c1024`), `+26.6 %` (64×64, `n32 c256 56×56 3×3`), `+3.5 %` (64×64 3×3 28×28). All `valid:y`.
- **Files:** `python/igemm/igemm_fwd_gtc_wmma_nhwc.py:353` (`bytes_per_row`), `:377-378` (`lds_a_size`/`lds_b_size`), `:792`/`:819-821` (offset emission), `:1868`/`:1884` (`step_bytes`), `:1745`/`:1783` (`row_off`); same structure in the bwd/wrw generators; `python/operations/wmma_main_loop.py:479-486` (buffer toggle).
- **Approach:** introduce `self.lds_bytes_per_row = bytes_per_row + tunable.lds_row_pad`, used **only** on the LDS side (`bytes_per_row` must stay the *global* stride for `move_slice_window`). Replace the `v_lshlrev_b32` row-scaling with `v_mul_lo_u32` (prologue-only, cost irrelevant). Valid pads keep 16 B alignment and make `gcd(stride_dwords, 64) == 4`: **+16 B** and **+48 B** both qualify; **+64 B is actively harmful** (measured).
- **Risk / tradeoff:** `lds_single_size = next_pow2(a+b)` rounds 20480 → 32768, doubling the LDS allocation. Occupancy was unaffected here (VGPR-bound at 4 blocks/CU) but will not always be. Prefer replacing the `v_xor_b32` toggle with a `±delta` add so a non-power-of-2 buffer stride works (1 extra VALU/iteration, saves 24 KB). **Measured −20 % on 3×3 128×128** — hence default `0`.
- **Validation:** `script/build_and_filter_configs.py` to prove assembly; `-V 1` across the shape set; then let `script/build_gfx1250_master_configs.py` emit both `lds_row_pad=0` and `=16` sections so the driver's own fastest-tunable selection resolves it per shape.

### OPT-2 — Stop writing fp32 output from fp16/bf16 kernels  ★ measured epilogue = 19 %
- **Why / evidence:** §3.5. Deleting the epilogue stores is worth **1.23×** on a K-deep 1×1 conv.
- **Files:** `driver/conv_driver.cpp:~831` (`dtype_alloc_byte`), `python/operations/coalescing_store_wmma.py:1054-1160` (`_emit_direct_store`), `python/igemm/igemm_base.py` (`wmma_acc_f16`, `wmma_acc_bf16`).
- **Approach:** two independent halves.
  (a) *Cheap, already built:* `wmma_acc_f16` / `wmma_acc_bf16` already make D genuinely 2 B/element and already flip `dtype_alloc_byte` back to `data_byte`. Benchmark those variants first — they halve epilogue bytes with no new code. Cost: accumulation precision.
  (b) *Correct fix:* keep f32 accumulate, add a `v_cvt_pk_f16_f32` + widened store epilogue so the kernel emits fp16 output from an f32 accumulator. This also halves bytes **without** precision loss, and matches what MIOpen actually produces.
- **Risk:** (a) changes numerics — re-validate NRMS thresholds. (b) requires the D-matrix column layout to be reshuffled to get contiguous elements per lane (`ds_bpermute`/`v_permlane`, or the existing non-`direct_store` LDS reshuffle path).
- **Validation:** `-V 1`; compare fp16-output vs fp32-output NRMS; measure on shallow-K shapes (`c=64`), where the epilogue share is largest.

### OPT-3 — Fix config/tunable selection  ★ measured up to +37 %, zero codegen risk
- **Why:** `lds_double_buffer` is a coin flip whose sign depends on the shape, and it gates `can_hoist` in `wmma_main_loop.py:546` (i.e. it silently selects between two *different* loop schedules).
- **Evidence** (fp16 128×128, single-variable isolation, all `valid:y`):

  | variant | `n256 c2048 14×14 1×1` | `n128 c1024 17×17 1×1` | `n64 c512 28×28 3×3` |
  |---|---|---|---|
  | none | **616.0** | **409.7** | 585.6 |
  | `lds_double_buffer=1` | 450.7 (−27 %) | 389.8 | **684.0** (+17 %) |
  | `direct_store=1` | 612.0 | 393.1 | 594.8 |
  | both | 447.8 | 397.7 | 675.3 |

  The canonical config named in `AGENTS.md` (`igemm_fwd_gtc_gfx1250_nhwc_fp16_direct.config`) sets **both**, i.e. it is 27 % off the best available tunable on large 1×1 shapes. The `_all` master config *does* already carry all 4 combinations (22/22/20/20 of 84 sections), so the driver's fastest-tunable search recovers this — but only if the `_all` config is the one being built and benchmarked.
- **Files:** `config/igemm_fwd_gtc_gfx1250_nhwc_fp16_direct.config`, `AGENTS.md` (the "canonical minimal config" recommendation), `script/benchmark_gfx1250_vs_miopen.py`.
- **Approach:** point all benchmarking and the documented quick-start at the `_all` master configs; keep the narrow configs for codegen debugging only. Optionally add a shape-based prior in `igemm_fwd_gtc_driver.h` so the runtime search starts from the likely winner.
- **Risk:** none to correctness; longer benchmark wall-clock.
- **Validation:** re-run `script/benchmark_gfx1250_vs_miopen.py` against `_all` and confirm the reported best ≥ current best.

### OPT-4 — Interleave memory into the WMMA burst  ★ theoretical 1.45×, largest effort
- **Why:** §3.3. This is the only lever that attacks the dominant loss.
- **Evidence:** `no_wmma (0.134) + wmma_only (0.055) ≈ baseline (0.195)`; overlap ≈ 3 %.
- **Files:** `python/operations/wmma_main_loop.py` (`emit_wmma_tile`, `emit()`), `python/codegen/mbb.py`, `python/codegen/scheduler.py` (`simple_interleave_scheduler_t`), modelled on `python/operations/mfma_main_loop.py` which already does exactly this for XDLOPS.
- **Approach:** build the loop body as typed `machine_basic_block_t`s (SALU/VALU/DS/GLOBAL/WMMA) instead of raw strings, then schedule: distribute the 8 `global_load_dwordx4`, 8 `ds_write_b128` and next-tile `ds_read_b128` **between** the 16 `v_wmma`, respecting (i) the ISA §7.12.1 WMMA hazard table — 5 V_NOP-equivalents when a WMMA's A/B matches the previous WMMA's D (not violated by the current accumulator pattern, but scheduling must preserve that), and (ii) WAR on `v_a`/`v_b` (`local_prefetch_num=2`'s two-slot register double-buffer already exists for this, but is inert because `num_k_substeps == 1` at `gemm_k_per_block == inst_wmma.k == 32`).
- **Risk:** high. Rewrites the hot loop for all three directions; interacts with the barrier-visibility race already documented for fp32 (`can_hoist` excludes fp32 for exactly this reason). Land it behind a tunable and validate per precision.
- **Validation:** re-run the §3.1 ablation table on the new kernel — success is `baseline → max(no_wmma, wmma_only)` rather than their sum. Then `-V 1` across the full shape set, fp32 last.

### OPT-5 — Cluster / multicast loads (ISA §10.7)  ★ attacks the 46 % term, unused today
- **Why:** ISA §10.7 opens with *"In GEMM applications, it is common to have multiple workgroups request the same data from memory. Multicast loads allow a single wave to request that data be loaded from memory and broadcast to multiple WGPs in the same cluster."* In the measured shape, each A-tile is re-read by 8 N-blocks and each B-tile by 289 M-blocks.
- **Evidence:** global loads = **45.6 %** of runtime (fwd) / 23 % (bwd); repo-wide grep finds no `CLUSTER_LOAD_*`, no `workgroup_mask`, no cluster usage at all.
- **Files:** new emitter in `python/codegen/instruction.py`; `python/igemm/igemm_*_gtc_wmma_nhwc.py` global-load functors; workgroup-cluster dispatch in `driver/igemm_gtc_base.h`.
- **Approach:** `CLUSTER_LOAD_ASYNC_TO_LDS_B128` (ASYNCcnt-tracked) with `M0` carrying the multicast mask + timeout, for the operand shared across the cluster's workgroups. Requires launching workgroups **in a cluster** (§2.3) — a host-side dispatch change.
- **Risk:** substantial new mechanism; timeout semantics mean a workgroup that arrives late gets a separate broadcast (correct, just slower). Needs cluster-capable dispatch in HIP.
- **Validation:** start with a standalone microbenchmark proving cluster dispatch + multicast works at all, then wire into fwd 1×1 only.

---

## 5. ISA features that appear underutilised

| ISA § | Feature | Status in repo | Assessment |
|---|---|---|---|
| §10.7 | `CLUSTER_LOAD_*` multicast, explicitly motivated by GEMM | **absent** | **Highest-value unused feature** — see OPT-5 |
| §10.11 | TDM (`TENSOR_LOAD_TO_LDS`): hardware OOB→0, **hardware LDS padding** via `D#.pad_amount`/`pad_interval`, descriptor iteration, cluster multicast | `tdm_global_load` exists but gated to `nxe==0` and off in every benchmarked config | Would deliver OPT-1 **for free** (hardware LDS padding) *and* remove the zero-fill/`v_cmpx` predication *and* remove the `ds_write` phase. Best single ISA lever after clusters. |
| §10.8 | `GLOBAL_LOAD_ASYNC_TO_LDS_B128` | `async_global_load` exists; not used by the benchmarked fwd configs | Removes the `ds_write` phase (2.1 % fwd, 7.3 % bwd) **and** frees the 32 staging VGPRs → headroom for a larger wave tile |
| §9 / §14.5.1 | `buffer_load_*` with a V# SRD (free hardware OOB→0) | **absent everywhere in the WMMA path** | Would delete the per-iteration 16-dword zero-fill + `v_cmpx` + `s_mov exec_lo`. **Measured worth only 1.0 %** — low priority despite being an obvious gap |
| §5.7.2.1 | `SCHED_MODE[2]` `DISABLE_XDL_ARB_STALL` | absent; **`hwreg(HW_REG_WAVE_SCHED_MODE)` = hwreg 26 confirmed assembler-reachable on gfx1250** | **Tested: 0 % effect** at 4 waves/SIMD. Revisit only if a future config runs 1 wave/SIMD |
| §5.8 | `S_DELAY_ALU` | absent | Low value — measurements show no dependency-stall term |
| §8.5 / §10.5 | `S_PREFETCH_DATA`, VMEM prefetch | absent | Low value — latency is already covered (§3.2) |
| §2.4 | Shader padding: 64 DWORDs of `S_CODE_END` past every shader; gfx1250 also wants an `S_MOV_B64`+`V_NOP`+`GLOBAL_PREFETCH_B8` prologue | absent — the assembler emits *"kernel … does not begin with the required prologue sequence"* for **every** generated kernel | Correctness/robustness item worth resolving regardless of performance |
| §7.12 | `V_WMMA_F32_16X16X4_F32` is the **only** pure-fp32 form (K=4 vs fp16's K=32) | in use | Structural: fp32 can never approach fp16 throughput here. Consider bf16/fp16 emulation for fp32 conv |
| §7.12.3 | `SWMMAC` structured sparsity | absent | Not applicable to dense convolution |

---

## 6. Convolution-specific recommendations

1. **1×1 (`nxe==0`) has no compile-time specialisation.** `emit_kernel_tap_loop` (`igemm_fwd_gtc_wmma_nhwc.py:1405-1449`) always emits the full runtime `y`/`x` two-level loop because `s_y`/`s_x` are kernarg SGPRs — a 1×1 conv still executes the loop scaffolding, one round of per-tap LDS-offset recompute (15 VALU) and per-tap gather (~19 VALU + 8 SALU), and both `s_cmp`/`s_cbranch` pairs. One-time per kernel, so *small*, but it also blocks constant-folding the address math. Worth doing as part of any tap-loop rework, not on its own.

2. **3×3 already performs best (690 TFLOP/s = 49 % of ceiling)** precisely because the 9 taps re-read overlapping input through L1/L2, cutting the dominant global-load term. This is strong corroboration that fwd 1×1 is feed-limited. Any optimisation must be validated on **both** 1×1 and 3×3 — OPT-1 helps 1×1 and hurts 3×3 at 128×128.

3. **bwd has no `stride==1 && dilation==1` fast path.** `_emit_tap_gather` (`igemm_bwd_gtc_wmma_nhwc.py:995-1128`) unconditionally emits **two magic div/rem macros** (10 VALU) plus exact-division checks per tap. Per-tap, so bounded — but for many-tap bwd it is pure waste on the overwhelmingly common stride-1 case. Cheap win: emit a stride-1 variant selected by a kernel-name suffix, exactly as `nxe` already discriminates elsewhere.

4. **wrw is structurally the worst and should be the next target after fwd.** `move_slice_window_b_functor` (`igemm_wrw_gtc_wmma_nhwc.py:1893-1929` → `_emit_b_gather_one_row:1325-1420`) re-runs the **whole spatial gather every K-iteration** (~29 VALU + 5 SALU, two magic div/rem), versus bwd's 4 VALU pointer bumps, because wrw's GEMM_K is spatial. Fix: strength-reduce the gather to an incremental update (the `wo`/`ho` indices advance by a constant per iteration; a magic division is only needed at wrap). Additionally, `wrw_streamk` configs are excluded from the `ds_load_tr_b` default (`igemm_base.py:893-895`) and therefore pay 128 packing VALU + 8 extra `s_wait_dscnt` per iteration — that exclusion should be revisited.

5. **Grouped convolution** works only by folding `group` into `grid_y` (`igemm_fwd_gtc_driver.h:337-396`). For `group > 1`, `gemm_n = K/group` shrinks; once `K/group < gemm_n_per_block` the N dimension is padded and WMMA lanes go idle. The 32×32 and rectangular 64×128 / 128×64 mappings exist and partially cover this, but there is no automatic small-N tile selection.

6. **Depthwise convolution is effectively unsupported.** Depthwise is `group == C`, giving `gemm_n = 1` per group — a 16-wide WMMA N tile is then 1/16 utilised at best. No specialised path exists. If depthwise matters, it needs a separate non-WMMA (VALU/dotx) kernel; forcing it through this pipeline will always be ~6 % efficient.

7. **No L2-aware workgroup swizzle anywhere.** The kernel reads `ttmp9`/`ttmp7` straight into `s_bx`/`s_by` (`igemm_fwd_gtc_wmma_nhwc.py:1017-1022`) and block offsets are a plain shift (`:1133-1134`). For the measured 289 × 8 grid, consecutive workgroups walk M fastest, so the weight tile's temporal locality across the M sweep is poor. A grouped/"threadblock swizzle" rasterisation is a well-understood 5–15 % win on large GEMMs and is a ~20-line change (remap `s_bx`/`s_by` in the prologue; no driver change needed). **Not yet measured here — recommend as experiment E-4.**

8. **Tail shapes** are covered by `wmma_m_tail`/`wmma_n_tail`/`wmma_k_tail` tunables, but `atomic_pack_bf16` is mutually exclusive with M/N tail (`coalescing_store_wmma.py:1271-1279`), so split-K wrw on a tail-heavy shape loses the packed-atomic halving.

---

## 7. Implementation plan

**Phase A — zero-risk, config/driver only (est. +20–37 % on the shapes measured)**
- A1. Repoint `AGENTS.md`'s quick-start, `script/benchmark_gfx1250_vs_miopen.py` and `..._vs_gfx950_diverse.py` at the `_all` master configs instead of the narrow `_direct` config. *(OPT-3)*
- A2. Correct the gfx1250 peak constant in `driver/conv_driver.cpp` (currently implies 1254 TFLOP/s; measured achievable 1411–1423) so tuning decisions are not calibrated against a wrong denominator.
- A3. Benchmark the existing `wmma_acc_f16` / `wmma_acc_bf16` configs on shallow-K shapes; they already halve epilogue bytes. *(OPT-2a)*

**Phase B — LDS padding tunable (est. +9–27 % on 1×1 and all 64×64 tiles)**
- B1. Add `lds_row_pad` to `igemm_gtc_tunable_parameter_t` (`python/igemm/igemm_base.py`) **and** its C++ twin `igemm_gtc_tunable_t` (`driver/igemm_gtc_base.h`) plus the name-mangling on both sides — these are manually synchronised, per `docs/architecture_map.md`.
- B2. Thread `lds_bytes_per_row` through the fwd generator (§4/OPT-1 file list). Assert `lds_bytes_per_row % 16 == 0` and `gcd(lds_bytes_per_row//4, 64) == 4`.
- B3. Replace the `v_xor_b32` double-buffer toggle in `wmma_main_loop.py:479-486` with a `±delta` add so `lds_single_size` need not be a power of two (saves 24 KB/workgroup at pad=16).
- B4. Repeat for bwd and wrw. **Note:** their transposed operand's LDS row pitch is `gemm_n_per_block * data_byte` = **256 B for the 128-wide fp16 tile = exactly 64 banks × 4 B**, an even more degenerate alias than fwd's 64 B. `[INFERENCE]` this predicts a far worse conflict than fwd's; measure before and after (bwd's LDS reads are already 12.4 % of runtime).
- B5. Emit `lds_row_pad ∈ {0, 16}` sections from `script/generate_all_configs.py` / `build_gfx1250_master_configs.py` so the runtime search picks per shape.

**Phase C — epilogue (est. up to +19 % on K-deep, more on shallow-K)**
- C1. Emit fp16/bf16 output from an f32 accumulator (`v_cvt_pk_*`), and drop the `is_wmma → sizeof(float)` override in `conv_driver.cpp`. *(OPT-2b)*
- C2. Widen the store: reshuffle the D-matrix column layout so each lane holds 4 contiguous columns, enabling `global_store_dwordx4` (32 instructions instead of 128). Reuse the existing non-`direct_store` LDS-reshuffle machinery or `v_permlane`.

**Phase D — the main-loop scheduler (est. up to 1.45 ×, high risk)**
- D1. Port `machine_basic_block_t` + `simple_interleave_scheduler_t` into `wmma_main_loop.py`, mirroring `mfma_main_loop.py`. Gate behind a tunable; fp16/bf16 first, fp32 last (the barrier-visibility race in `can_hoist` applies).
- D2. Make `local_prefetch_num=2` reachable at `gemm_k_per_block == inst_wmma.k` (today it requires `num_k_substeps > 1`, which never holds for the shipped 32-deep fp16 tiles) so `v_a`/`v_b` register double-buffering can cover the WAR hazard the interleaving creates.
- D3. Re-run the §3.1 ablation as the acceptance test.

**Phase E — ISA mechanisms (research)**
- E1. Wire `async_global_load` into the fwd configs and measure; it removes the `ds_write` phase and frees 32 VGPRs.
- E2. Extend TDM beyond `nxe==0` and enable `D#.pad_amount`/`pad_interval` — hardware LDS padding subsumes Phase B at zero instruction cost.
- E3. Prototype cluster dispatch + `CLUSTER_LOAD_ASYNC_TO_LDS_B128`. *(OPT-5)*
- E4. Add the workgroup swizzle (§6.7) — cheapest untested idea in this report.
- E5. Add the §2.4 shader padding / `GLOBAL_PREFETCH_B8` prologue the assembler is asking for.

---

## 8. Suggested experiments and benchmarks

**Reusable harness.** The ablation method used throughout this report is the highest-signal tool available here and should be kept: copy a generated output dir, delete instruction classes from *only* the `_wmma_body` region of the `.inc` (leaving `_wmma_body_last` intact so the loop structure and trip count are unchanged), re-assemble with `$ROCM_PATH/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx1250`, and run with `-V 0`. Results are numerically wrong and timing-only, which is exactly what is needed for attribution. Recommend landing it as `script/ablate_main_loop.py`.

| # | Experiment | Measures | Success criterion |
|---|---|---|---|
| E-1 | Ablation table (§3.1) re-run after every change | where cycles actually go | `baseline → max(no_wmma, wmma_only)` instead of their sum |
| E-2 | Stride sweep 64/80/112/128 B on bwd and wrw | whether the 256 B transposed row pitch is the predicted 16-way alias | ≥ the +10 % fwd saw; bwd LDS-read share drops below 12.4 % |
| E-3 | fp16-output vs fp32-output epilogue, shallow-K (`c=64,128`) | OPT-2 upside where the epilogue dominates | ≥ +15 % at `c=64`, NRMS within threshold |
| E-4 | Workgroup swizzle (group-M rasterisation), sweep group width 1/4/8/16 | L2 locality of the weight tile | ≥ +5 % on `n256 c2048 14×14 k2048` |
| E-5 | `async_global_load=1` fwd configs | `ds_write` elimination + 32 VGPR headroom | ≥ +2 % and VGPR count ≤ 220 |
| E-6 | Full `_all` sweep before/after each phase, fwd+bwd+wrw, fp16/bf16/fp32 | regression safety | no shape regresses > 2 % |
| E-7 | Roofline check: sweep `c ∈ {64,128,256,512,1024,2048}` at fixed spatial size | confirms the feed-limited → compute-limited transition | efficiency rises monotonically with `c` |

**Benchmark shape set used in this report** (keep as the standing regression set — it spans feed-limited, compute-limited and tail-heavy regimes):
```
-n 256 -c 2048 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0     # large 1x1, feed-limited
-n 128 -c 1024 -H 17 -W 17 -k 1024 -y 1 -x 1 -p 0 -q 0     # medium 1x1
-n 64  -c 512  -H 28 -W 28 -k 512  -y 3 -x 3 -p 1 -q 1     # 3x3, compute-limited (best today)
-n 32  -c 256  -H 56 -W 56 -k 256  -y 3 -x 3 -p 1 -q 1     # 3x3, large spatial
-n 128 -c 64   -H 56 -W 56 -k 64   -y 1 -x 1 -p 0 -q 0     # small-channel / tail-heavy (86 TFLOP/s today)
```
Note the last shape runs at **86 TFLOP/s (6 % of ceiling)** — small-channel shapes are an entirely separate, unaddressed problem: with `c=64`, `gemm_k=64` gives only 2 K-iterations, so prologue, tap-loop scaffolding and the fp32 epilogue dominate completely. OPT-2 and a `nxe==0` specialised prologue are the relevant levers there, not anything in the main loop.

---

## 9. Addendum — wrw measured (2026-09-02, same session)

§6.4 of this report inferred from source that wrw's per-K-iteration spatial gather was "the decisive extra cost". **That inference is wrong.** Measured, the gather is 7 %; the transposed LDS reads are 31 %. Corrected findings below.

### 9.1 wrw is 20–30× slower than fwd/bwd, and the dominant cause is grid starvation

wrw's GEMM is `M = K/group`, `N = C/group`, `GEMM_K = N·Ho·Wo` — a *small-MN, enormous-K* GEMM. Without split-K the grid is tiny:

| shape | M×N | grid @128×128 | % of 256 CUs | measured (default cfg) | measured efficiency |
|---|---|---|---|---|---|
| `n128 c1024 17×17 k1024 1×1` | 1024×1024 | 64 WGs | 25 % | 52.5 TFLOP/s | 4.17 % |
| `n64 c512 28×28 k512 3×3` | 512×512 | **16 WGs** | **6 %** | 13.4 TFLOP/s | 1.06 % |
| `n256 c2048 14×14 k2048 1×1` | 2048×2048 | 256 WGs | 100 % | 197.5 TFLOP/s | 15.7 % |

Efficiency ≈ `(grid/256) × ~17 %` in every row — i.e. **per-CU efficiency is a roughly constant ~16–21 %, and everything else is idle CUs.**

`config/igemm_wrw_gtc_gfx1250_nhwc_fp16.config` — the plain wrw config — sets **no `gemm_k_global_split`**. Enabling it (config-only, no codegen change):

| shape | default cfg | `_gsplit` | `_64x64_all` | speed-up |
|---|---|---|---|---|
| `n128 c1024 17×17 1×1` | 72.4 | 338.3 | **341.5** | **4.7×** |
| `n64 c512 28×28 3×3` | 23.3 | 353.5 | **403.3** | **17.3×** |

### 9.2 wrw stream-K is the fastest path *and* is numerically wrong

`config/igemm_wrw_gtc_gfx1250_nhwc_fp16_streamk.config`, `n128 c1024 17×17 k1024 1×1`:
```
bt128x128x32 ..._streamk_gkgs[68], cost:0.170ms, tflops:457.233(36.34%), valid:n
bt64x64x32   ..._streamk_gkgs[68], cost:0.205ms, tflops:378.181(30.06%), valid:n
```
**457 TFLOP/s — 35 % faster than the best split-K result (338) and 6.3× the default config** — but wrong. Reproducible and tile/shape dependent: at `n8 c128 14×14 k128 1×1` the 128×128 tile is `valid:y` while the 64×64 tile is `valid:n`. Suspects (unverified): the claim-broadcast sequence (`igemm_wrw_gtc_wmma_nhwc.py:2030-2042`, system-scope `global_atomic_add_u32` → LDS broadcast → 2 barrier pairs → `v_readfirstlane_b32`), or the per-claim `s_gemm_k_wg_off = tile_idx * gemm_k_per_wg` shard mapping (`:2054`) at partial tiles. Note `wrw_streamk` requires `nxe==0` (`igemm_base.py:940-941`), so 3×3 cannot use it at all.

### 9.3 Ablation at full grid (`n256 c2048 14×14 k2048 1×1`, 256 WGs, baseline 2.124 ms / 198 TFLOP/s)

| ablation | cost | speed-up | attributed |
|---|---|---|---|
| 16 × `ds_load_tr16_b128` | 1.468 ms | **1.447×** | **30.9 %** |
| spatial gather (`.mdiv_u32_rem_vs` ×2 + `v_mul_lo_u32` + bounds flag) | 1.975 ms | 1.075× | **7.0 %** |
| 16-dword `v_gld_b` zero-fill | 2.111 ms | 1.006× | 0.6 % |

Loop-body instruction census (generated `.inc`, 116 instructions): 16 `v_wmma`, 16 `ds_load_tr16_b128`, 8 `global_load_dwordx4`, 8 `ds_write_b128`, 8 `v_dual_mov_b32`, 2 `.mdiv_u32_rem_vs` macros + ~23 VALU of gather. So ~43 VALU of address/predication overhead per 16 WMMA (vs fwd's 13) — but it costs only 7 %, because it is not the binding resource.

### 9.4 The transposed LDS pitch is the leading suspect for the 31 %

`[INFERENCE]`, but with three converging measurements:
1. wrw's transposed row pitch is `gemm_{m,n}_per_block * data_byte` = **256 B at the 128-wide tile = exactly 64 banks × 4 B**, so every K-row aliases the identical bank set (`igemm_wrw_gtc_wmma_nhwc.py:718-737`). At the 64-wide tile the pitch is 128 B — half as degenerate.
2. **wrw's 128×128 tile is *slower* than its 64×64 tile (198 vs 259 TFLOP/s)** — inverted relative to fwd and bwd, and consistent with the wider tile having the worse alias.
3. `ds_load_tr_b=0` (the manual path: 128 × `ds_read_u16` + 128 packing VALU) is **exactly as fast as** 16 × `ds_load_tr16_b128` at 128×128 (197.9 vs 198.1), while being 1.32× *slower* at 64×64 (195.9 vs 259.4). Sixteen wide reads should crush 128 narrow reads + 128 VALU; that they tie is only explicable if the wide reads are severely serialised.

### 9.5 Revised wrw priority list

| # | Item | Measured / expected | Effort |
|---|---|---|---|
| W-1 | **Enable `gemm_k_global_split` in the default wrw configs**; make the driver never emit a wrw grid < CU count without split-K | **4.7×–17.3×** | config + a driver guard |
| W-2 | **Fix wrw stream-K correctness** (`valid:n`) | already **+35 %** over split-K where it runs | debug; repro in §9.2 |
| W-3 | **Sweep the transposed LDS row pitch** (256 B → 272 B / 320 B) for wrw *and* bwd | up to 31 % of wrw runtime is in these reads | same mechanism as OPT-1 |
| W-4 | Prefer 64×64 over 128×128 for wrw until W-3 lands | 259 vs 198 (**+31 %**) | config ordering |
| W-5 | Un-gate `ds_load_tr_b` for `wrw_streamk` (`igemm_base.py:893-895`) after W-2 | removes 128 packing VALU + 8 waits/iteration | 1-line + validation |
| W-6 | Strength-reduce the per-iteration gather to an incremental index update | **7 %** — *not* the priority §6.4 claimed | moderate |
