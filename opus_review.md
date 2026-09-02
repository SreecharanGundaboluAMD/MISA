All investigation phases are complete. Full report follows.

---

# gfx1250 MISA Investigation — Root Cause Discovery

## Executive Summary

The gfx1250 WMMA path is **not compute-limited and not occupancy-limited**. Ablation on real hardware decomposes the shipping fp16 128×128 kernel into three phases whose isolated costs sum to 0.139 ms against a measured total of 0.137 ms — **the main loop is essentially perfectly serialized, with near-zero overlap between global load, LDS traffic, and WMMA**. The WMMA phase itself already runs at ~98% of achievable WMMA throughput. Perfect overlap alone is worth **~1.73×**.

Three structural causes explain the serialization and the gap vs gfx950:

1. **The main loop has no software pipelining.** One barrier and three *full* counter drains per iteration; `lds_double_buffer=1` adds exactly three `v_xor_b32` and removes no barrier. Every wait emitted anywhere in `python/` is `0x0` — not one partial wait, despite ISA §5.7 documenting `S_WAIT_LOADCNT <= N` and a fused `S_WAIT_LOADCNT_DSCNT`.
2. **Large macro-tiles are reachable and faster, but almost never dispatch.** I must correct my own earlier reading here: gfx1250 is *not* capped at 128×128. A 256×256 tile exists, is hardware-validated, and I measured it **1.21× (bf16) / 1.32× (fp16) faster** than 128×128 — directly contradicting the Phase 56 note in the config itself. The real blocker is that `wmma_acc_high_bank` and `wmma_epilogue_chunked` are each mutually exclusive with `wmma_m_tail`/`wmma_n_tail`, so large tiles only dispatch when `gemm_m % 256 == 0`.
3. **Reachable tiles are a hand-written table.** `ctrl_wmma_mapping_table` gave 256×* entries to `bf16` but not to plain `fp16`, `int8`, or `fp32`. Adding two lines for fp16 yielded a measured **1.07–1.35× speedup, `valid:y` on every shape tested**. That patch is applied and validated (see Suggested Patches).

Separately, two findings invalidate parts of the existing evidence base: **`-V 0` benchmarks are meaningless** (input buffers are never written, and WMMA is ~2× faster on all-zero data — measured), and **202 of 721 gfx1250 fp32 config sections lack `lds_double_buffer=1`**, contradicting CLAUDE.md; I reproduced a live `valid:n` and confirmed the flag fixes it.

---

## Repository Map

### Dispatch path
```
conv_driver.cpp:705-731   mode string → driver_data_type      (convfp16/convbfp16/convint8/conv)
conv_driver.cpp:750-757   -F bitmask  → need_fwd/need_bwd/need_wrw
conv_driver.cpp:680-695   is_wmma / acc-width flags read ONCE from tunables[0]   ← COR-004
conv_driver.cpp:994       igemm_fwd_gtc_t(module, ...)
conv_driver.cpp:1083-1085 launch_conv_driver(driver, args, tunables, "fwd", ...)
igemm_gtc_base.h:653-673  igemm_launch_kernel: warmup loop, repeat loop, drop min+max, average
                          (verify-independent — the -V discrepancy is NOT in the timing harness)
```

### Configuration-selection path
```
driver/igemm_fwd_gtc_driver.h:450-503   tunable_is_valid, WMMA branch (early-out, bypasses XDLOPS checks)
   :497-498   tdm_global_load  → require unit conv
   :499-502   gemm_m % gemm_m_per_block  unless wmma_m_tail
              gemm_n % gemm_n_per_block  unless wmma_n_tail
              gemm_k % gemm_k_per_block  unless tdm_global_load or wmma_k_tail
   invalid tunables are silently skipped (IGEMM_ASSERT_WHEN_INVALID=1 to abort)
script/build_gfx1250_master_configs.py:106  ACCUMULATE_WIDTH_KEYS excluded from master union
script/build_and_filter_configs.py           drops sections that fail assembly   ← COR-003 hazard
```

### Kernel-generation path
```
codegen_driver.py:48-83     direction + presence of 'wmma_tile_m' → igemm_{dir}_gtc_wmma_nhwc_t
igemm_base.py:484-489       waves_per_m/n = gemm_{m,n}_per_block / (wmma_tile * wmma_repeat)
                            block_size = waves_per_m * waves_per_n * 32
                            asserts block_size == prod(a_cluster) == prod(b_cluster)
igemm_base.py:946-950       get_ctrl_wmma_mapping_from_wave_tile → ENUMERATED TABLE
python/operations/wmma_mapping.py:312-430   ctrl_wmma_mapping_table, 34 entries
emit_kernel_body = prologue → tap_loop → epilogue
   tap loop → ctrl_wmma_main_loop_t → wmma_main_loop_t.emit()   (ONE schedule, all directions)
   epilogue → coalescing_store_wmma.py
```
All three directions share `wmma_main_loop.py`, so every main-loop finding below applies to fwd, bwd and wrw alike.

### Reachable macro-tiles (measured from the table)
| precision    | tiles                                                                      |
| ------------ | -------------------------------------------------------------------------- |
| fp16         | 32×32, 64×64, 64×128, 128×64, 128×128 *(+256×256, 256×128 after my patch)* |
| fp16_f16acc  | 64×64, 64×128, 128×64, 128×128, 256×128, 256×256                           |
| bf16         | 32×32, 64×64, 64×128, 128×64, 128×128, 256×128, 256×256                    |
| bf16_bf16acc | same as bf16 minus 32×32                                                   |
| **int8**     | 64×64, 64×128, 128×64, **128×128 max**                                     |
| **fp32**     | 32×32, 64×64, 64×128, 128×64, **128×128 max**                              |

---

## Correctness Findings

### Confirmed defects

**COR-001 — 202 of 721 gfx1250 fp32 config sections omit the mandatory `lds_double_buffer=1`.**
*Severity: High. Confidence: Confirmed (measured).*
CLAUDE.md states "All fp32 configs in `config/` already set `lds_double_buffer=1`". They do not. `config/igemm_fwd_gtc_gfx1250_nhwc_fp32_direct.config` contains no `lds_double_buffer` line at all. Worst offenders: `igemm_bwd_..._fp32_32x32_all` (50), `igemm_wrw_..._fp32_all` (28), `igemm_fwd_..._fp32_all` (20), `igemm_bwd_..._fp32_all` (16) — i.e. the master `_all` configs the build and benchmark workflows actually consume.
*Trigger:* any fp32 WMMA kernel at sufficient occupancy (the documented last-lane LDS-write visibility race).
*Reproducer:*
```bash
python3 igemm_codegen.py -d /tmp/f32 config/igemm_fwd_gtc_gfx1250_nhwc_fp32_direct.config
cd /tmp/f32 && ./conv_driver.exe conv -n 128 -c 1024 -H 14 -W 14 -k 1024 \
  -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F 1 -V 1 \
  --in_layout NHWC --fil_layout NHWC --out_layout NHWC
# bt64x64x4 ... valid:n     (reproduced 2/2)
```
Adding `lds_double_buffer = 1` to both sections → `valid:y` 2/2, at **identical cost** (0.805/0.811 ms vs 0.810/0.844 ms), so the fix is free here.
*Fix:* enforce it in code, not by convention — assert in `igemm_base.py` that `precision=='fp32' and fma_type==WMMA` implies `lds_double_buffer`, then repair the 202 sections.

**COR-002 — `-V 0` benchmark numbers are invalid, and the invalidity is data-dependent.**
*Severity: High (invalidates published results). Confidence: Confirmed (measured, mechanism isolated).*
`driver/conv_driver.cpp:827-863` randomizes and `hipMemcpy`s the input/weight buffers **only inside `if (need_verify)`**. There is no `hipMemset` anywhere in the file. Under `-V 0` the device buffers are `hipMalloc`'d and never written. This matters because I measured **`v_wmma_f32_16x16x32_f16` is 2.06× faster on all-zero operands**: 4440 TFLOPS (zeros) vs 2157 (random), with *identical* sclk (2278 MHz) and package power (1564 vs 1565 W) — a hardware zero/low-activity fast path, not DVFS.
*Observed:* same kernel, `-V 0` → 0.099 ms / 528.9 TFLOPS; `-V 1` → 0.137 ms / 384.8 TFLOPS (2 runs each). Ablation confirms the mechanism: removing the WMMA instructions collapses the gap (0.099 → 0.103, +4%), while the pure-global-load ablation shows almost none.
*Impact:* `script/benchmark_gfx1250_vs_miopen.py` uses `-V 0` **and** `min(costs)`, so the "wrw 1.28–2.70× faster than MIOpen" result in `bench_results_gfx1250_vs_miopen_20260830.md` is not supported.
*Fix:* move the randomize+copy out of the `need_verify` guard, or `hipMemset` to a non-zero pattern unconditionally.

**COR-003 — Kernel-name collision: `wmma_epilogue_chunked` and `wmma_acc_high_bank` are read by codegen but not encoded in the kernel name.**
*Severity: Medium. Confidence: Confirmed (reproduced).*
A programmatic diff of tunables-read vs tunables-encoded (`igemm_gtc_encode_kernel_name`, `igemm_base.py`) leaves exactly these two unencoded. Two sections differing only in `wmma_epilogue_chunked` emit the same symbol:
```
error: symbol 'igemm_fwd_gtcw_nhwc_fp16_bx0_ex0_bt128x128x32_...' is already defined
```
with `.amdhsa_group_segment_fixed_size 65536` vs `16384`. `-s` split mode collides too (same filename). This is currently *loud*, but `script/build_and_filter_configs.py --write` drops sections that fail assembly — so a legitimate config gets silently deleted from the search set.
*Fix:* append `_chunked` / `_hibank` suffixes in both the Python mangler and its C++ twin in `driver/igemm_gtc_base.h` (these must be edited together — they are independently maintained and drift silently).

**COR-004 — Output-buffer width is derived from `tunables[0]`, so mixing accumulate widths in one search corrupts verification.**
*Severity: Medium (latent; mitigated by convention). Confidence: Confirmed (source).*
`conv_driver.cpp:680-695` computes `is_wmma_f16_acc` / `is_wmma_bf16_acc` / `is_wmma_atomic_pack_bf16` once from `tunables[0]`, then sizes the output buffer from them. Any kernel later in the same search with a different accumulate width reads/writes the wrong element width. `script/build_gfx1250_master_configs.py:106` works around this by excluding `ACCUMULATE_WIDTH_KEYS` from the master union — a script-level convention a hand-written config can violate freely.
*Fix:* reject at load time — assert all tunables in one run share an accumulate width.

### Likely defects

**COR-005 — The driver's efficiency model is wrong for gfx1250, and prints impossible values.**
*Severity: Low (reporting only). Confidence: Confirmed arithmetic, Likely impact on decisions.*
`get_theoritical_gpu_gflops` uses `num_simd = 4*16` (gfx1250 is absent from the `4*32` list) and `fp_factor = 8`, implying ~629 TFLOPS fp16 peak. It reported **60.41%** for a kernel at 385 TFLOPS (really ~17.8% of the 2157 TFLOPS achievable peak) and a physically impossible **117.38%** for the fp32 kernel. Any tuning steered by these percentages is steered wrong.

### Suspicious code requiring validation

**COR-006 — Every gfx1250 kernel omits the LLVM-mandated prologue.**
*Severity: Unknown. Confidence: Confirmed absent; impact Unknown.*
`hipcc --cuda-device-only -S` for a trivial gfx1250 kernel emits `s_mov_b64 s[64:65],0` / `v_nop` / `global_prefetch_b8` and `s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1` (MODE.REPLAY_MODE). MISA emits **zero** `global_prefetch` and **zero** `setreg`; every kernel triggers the assembler warning *"does not begin with the required prologue sequence"*.
**Refuted for performance:** base 0.138, +setreg 0.138, +prefetch 0.139, +both 0.139 ms, all `valid:y`. It remains open as a correctness/compliance question only, because the ISA does not document what REPLAY_MODE does (see ISA Findings).

---

## Performance Findings

**PERF-001 — The main loop is fully serialized; there is no software pipelining. Impact: XL (~1.73×). Confidence: Confirmed.**
*Category: Kernel Generation / Synchronization. Evidence: Generated Code + Measured.*
Shape `n128 c1024 14×14 k1024` fp16 128×128, `-V 1`:

| ablation                  | cost (ms) |     | ablation | cost (ms) |
| ------------------------- | --------- | --- | -------- | --------- |
| base                      | 0.137     |     | nowmma   | 0.103     |
| empty (loop body removed) | 0.020     |     | nogld    | 0.076     |
| only_wmma                 | 0.045     |     | noldsr   | 0.106     |
| only_lds                  | 0.075     |     | noldsw   | 0.120     |
| only_gld                  | 0.079     |     | nobar    | 0.125     |
| gld_wmma                  | 0.110     |     | nodswait | 0.126     |

Isolated costs above the 0.020 floor: WMMA 0.025, LDS 0.055, global 0.059 — **sum 0.139 ≈ base 0.137**. Perfectly overlapped would be `max(0.059,0.055,0.025)+0.020 = 0.079 ms`. The WMMA phase runs 52.6 GFLOP in 0.025 ms ≈ 2100 TFLOPS ≈ **98% of the 2157 TFLOPS achievable peak** — compute is not the problem.
*Competing hypothesis (refuted):* insufficient occupancy to hide latency. A standalone microbenchmark reaches ~3160 TFLOPS at **1 wave/SIMD** with only 2 independent accumulator chains; 1/2/4/8 waves/SIMD all land at ~3150–3160 with N_ACC≥4.
*Distinguishing experiment already run:* the pipelining knobs are inert — dbuf / k64 / k64+interleave / baseline = 389.5 / 364.2 / 383.6 / 388.6 TFLOPS, within 6%.

**PERF-002 — Large macro-tiles are faster but rarely dispatch. Impact: L–XL. Confidence: Confirmed.**
*Category: Configuration Selection. Evidence: Measured + Source.*
bf16, `-V 1`, both `valid:y`:

| shape                  | 256×256      | 128×128  | speedup |
| ---------------------- | ------------ | -------- | ------- |
| n128 c1024 14×14 k1024 | 0.108 ms     | 0.128 ms | 1.19×   |
| n128 c512 28×28 k512   | 0.124        | 0.156    | 1.26×   |
| n64 c512 28×28 k512    | 0.071        | 0.075    | 1.06×   |
| n256 c1024 14×14 k1024 | 0.206        | 0.200    | 0.97×   |
| n128 c2048 7×7 k2048   | **REJECTED** | 0.105    | —       |

This refutes the config's own Phase 56 comment ("NOT a performance win on any shape tested… ~2-3x slower at small scale"). The rejection is `driver/igemm_fwd_gtc_driver.h:499`: `gemm_m = 128·49 = 6272`, `6272 % 256 = 128 ≠ 0`, and the config cannot set `wmma_m_tail` because of two hard asserts:
```
AssertionError: wmma_epilogue_chunked is not yet combined with wmma_m_tail/wmma_n_tail
AssertionError: wmma_acc_high_bank is not yet combined with wmma_m_tail/wmma_n_tail's masked slow-path store
```
A 256×256 f32-accumulate tile *requires* both (262144 B of epilogue LDS otherwise; 256 acc VGPRs/lane otherwise), so **large tile ⇒ no tail handling ⇒ exact-divisibility only**. That coupling, not throughput, is why large tiles look useless.

**PERF-003 — Reachable tiles are a hand-written table; fp16/int8/fp32 were missing rows. Impact: L (measured 1.07–1.35× for fp16). Confidence: Confirmed.**
*Category: Kernel Generation. Evidence: Source + Measured.*
`ctrl_wmma_mapping_table` gave 256×256/256×128 to `bf16`, `bf16_bf16acc` and `fp16_f16acc` but not to plain `fp16`, `int8`, or `fp32`. The in-file comment describes the bf16 rows as a "mechanical port" — only the mnemonic differs. Adding two fp16 rows built clean at `.amdhsa_next_free_vgpr 512` and measured, all `valid:y`:

| shape                  | fp16 128×128        | fp16 256×256  | speedup   |
| ---------------------- | ------------------- | ------------- | --------- |
| n128 c1024 14×14 k1024 | 0.140 ms / 375.5 TF | 0.106 / 494.3 | **1.32×** |
| n128 c512 28×28 k512   | 0.143 / 368.9       | 0.124 / 423.1 | **1.15×** |
| n64 c512 28×28 k512    | 0.077 / 340.0       | 0.072 / 363.3 | **1.07×** |
| n256 c1024 14×14 k1024 | 0.281 / 374.8       | 0.208 / 505.1 | **1.35×** |

**PERF-004 — All waits are full drains; not one partial wait exists. Impact: M–L. Confidence: Confirmed (source) / Likely (impact).**
*Category: Synchronization. Evidence: Generated Code + ISA.*
Exhaustive grep of `python/`: 19× `s_wait_dscnt 0x0`, 8× `s_wait_loadcnt 0x0`, 7× `s_wait_storecnt 0x0`, 3× `s_wait_kmcnt 0x0`, 1× `s_wait_tensorcnt 0x0`, 1× `s_wait_asynccnt 0x0`. Zero partial waits. The generated body drains **all 16** `ds_read_b128` before the first WMMA even though the first WMMA needs only the first four. ISA §5.7 documents `S_WAIT_LOADcnt <= 1` and Table 25 lists the fused `S_WAIT_LOADCNT_DSCNT`; LOADcnt is a 6-bit counter (63 outstanding).
*Impact is Likely, not Confirmed:* the `nodswait` ablation (0.126 vs 0.137) bounds the *removable* drain cost at ~8%, but partial waits also unlock the reordering PERF-001 needs, so the two are coupled.

**PERF-005 — Register pressure blocks the standard fix at 128×128. Impact: L if resolved. Confidence: Confirmed.**
*Category: Register Pressure.* `local_prefetch_num=2` — register double-buffering of `ds_read`s, exactly what gfx950 does — needs **316 VGPRs** on the 128×128 tile and fails to assemble (`error: register index is out of range` ×425). Accumulators alone are 128 VGPRs, because wave32 × 4 waves = 128 lanes for a 128×128 tile and WMMA has no separate accumulator file. This is precisely what `wmma_acc_high_bank` (VGPR-MSB) exists to solve, and it is already working — but it is mutually exclusive with tail handling (PERF-002).

**PERF-006 — fp32 pays ~8× the per-K loop overhead and has no large tiles. Impact: XL for fp32. Confidence: Confirmed.**
*Category: Instruction Selection.* The only pure-fp32 form is `v_wmma_f32_16x16x4_f32` (K=4) vs fp16's K=32. The generated fp32 kernel is `bt128x128x4` with the *same* single barrier and three full drains per iteration but 1/8 the FLOPs — a 46-instruction body of which 16 are WMMA, 8 `ds_read_b64`, 2 `global_load_dwordx4`. Measured 0.570 ms / 92.3 TFLOPS where fp16 got 0.137 ms. This quantitatively explains the worst measured gaps (bwd fp32 5.90×, wrw fp32 5.83×). fp32 also caps at 128×128 in the mapping table, compounding it.

**PERF-007 — `direct_store` over-reserves LDS. Impact: XS today, blocking later. Confidence: Confirmed.**
`get_kernel_code` (`igemm_fwd_gtc_wmma_nhwc.py:649-696`) computes `epilogue_lds_bytes` without consulting `direct_store`, so a direct_store kernel that never uses epilogue LDS still declares 65536 B while its main loop needs 16384. Currently latent — VGPR 252 caps occupancy at 4 blocks/CU regardless — but it will cancel any future VGPR reduction.

### Refuted hypotheses (negative results)
- **Occupancy limits the WMMA pipe** — refuted; ~3160 TFLOPS at 1 wave/SIMD.
- **Uncoalesced lane mapping (32 lanes × 2048 B stride) costs bandwidth** — refuted; MISA's mapping 3.17 TB/s vs gfx950's 4-lanes-per-row mapping 3.36 TB/s (6%, and the grid was the limiter).
- **Missing LLVM prologue / REPLAY_MODE costs performance** — refuted (COR-006).

---

## gfx1250 vs gfx950 Comparison

Evidence-backed implementation differences only (both built from this repo's own generator):

|                        | gfx1250 WMMA                                            | gfx950 XDLOPS                                                          |
| ---------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------- |
| wave                   | 32                                                      | 64                                                                     |
| accumulators           | plain arch VGPRs (`v_c+0..127`)                         | separate AGPR file (`.amdhsa_accum_offset`)                            |
| 256×128 fp16 kernel    | n/a (table row absent pre-patch)                        | `next_free_vgpr 212`, `accum_offset 84` → 84 arch + 128 acc, LDS 32768 |
| 128×128 fp16 kernel    | `next_free_vgpr 252`, LDS 65536                         | —                                                                      |
| 256×256 bf16 kernel    | `next_free_vgpr 512`, LDS 65536 → 1 wave/SIMD           | —                                                                      |
| main loop              | `wmma_main_loop.py`: **one** correctness-first schedule | `mfma_main_loop.py`: ~10 hand-unrolled schedules, 60 `s_barrier` sites |
| barriers/iteration     | 1 (`s_barrier_signal -1`/`s_barrier_wait -1`)           | scheduled per variant                                                  |
| waits                  | full drains only                                        | `s_waitcnt lgkmcnt(N)`/`vmcnt(N)` partials                             |
| tile shapes in configs | 32×32 … 256×256 (256× bf16-only pre-patch)              | 32×256 … 256×256, 12 distinct shapes                                   |

**The structural asymmetry:** at equal accumulator cost per lane (128 VGPRs), wave64 buys gfx950 a 256×128 tile while wave32 buys gfx1250 only 128×128 — half the arithmetic intensity (64 vs 128 FLOP/byte) for the same register budget. gfx1250 can recover it by using 8 waves instead of 4, but the mapping table has no such entry: the existing 256×128 row is `wr4x8 w4` (4 waves → 256 acc/lane → forces VGPR-MSB) rather than `wr4x4 w8` (8 waves → 128 acc/lane → no VGPR-MSB, tails allowed). I attempted that entry; it fails the table lookup by construction, and adding it needs a matching `wmma_mapping` derivation, not just a row.

---

## ISA Findings

Sections were selected only from concrete code evidence, per instruction.

- **§5.7 "Data Dependency Resolution"** — selected because every emitted wait is `0x0` (PERF-004). **Establishes:** partial waits are supported and idiomatic (`S_WAIT_LOADcnt <= 1` in the ISA's own example); Table 25 provides fused `S_WAIT_LOADCNT_DSCNT` (SIMM16[15:8]=load/store, [7:0]=DS); LOADcnt is 6-bit. This confirms PERF-004 is a codegen choice, not an architectural constraint.
- **§3.4.3 MODE register** — selected because `hipcc` emits `s_setreg hwreg(HW_REG_WAVE_MODE, 25, 1)` and MISA does not (COR-006). **Establishes:** bit 25 is `REPLAY_MODE`, `0 = single-VMEM-group mode, 1 = multi-VMEM-group mode`; also DST/SRC0/SRC1/SRC2_VGPR_MSB at 12:19 (the mechanism behind `wmma_acc_high_bank`). **Does NOT establish:** what REPLAY_MODE actually changes. Grep for "replay" / "VMEM group" across the whole document returns only this one definitional line plus two unrelated XNACK mentions. No behavioral or performance semantics are documented, so I cannot classify COR-006's severity from the ISA.

**Where the ISA does not establish behavior, explicitly:**
- No throughput/latency table for `v_wmma_*`. The measured **2.06× data-dependent throughput difference** (all-zero vs random operands, at identical clock and power) is **not documented anywhere** in the ISA. This is the single most consequential undocumented behavior found, since it silently invalidates any zero-initialized benchmark.
- No documented WMMA→WMMA or LDS→WMMA hazard latencies, so the correct partial-wait distances for PERF-004 must be found empirically.
- REPLAY_MODE semantics, as above.

---

## Prioritized Action Plan

| #   | Action                                                                                                           | Impact                                        | Conf.              | Effort | Validation cost |
| --- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | ------------------ | ------ | --------------- |
| 1   | Fix `-V 0` input init (COR-002) and re-run every benchmark                                                       | Invalidates/re-bases all existing perf claims | Confirmed          | XS     | Low             |
| 2   | Land fp16 256×* table rows (PERF-003) — **done, validated**                                                      | 1.07–1.35× fp16                               | Confirmed          | XS     | Done            |
| 3   | Enforce `lds_double_buffer` for fp32 WMMA + repair 202 sections (COR-001)                                        | Correctness                                   | Confirmed          | S      | Low             |
| 4   | Software-pipeline `wmma_main_loop.py`: hoist global loads one full iteration, partial waits (PERF-001, PERF-004) | up to ~1.73×                                  | Confirmed headroom | L      | Medium          |
| 5   | Make tails compatible with `wmma_acc_high_bank`/`epilogue_chunked` (PERF-002)                                    | L — unlocks large tiles on most shapes        | Confirmed blocker  | M–L    | Medium          |
| 6   | Add `256×128 wr4x4 w8` mapping (8 waves, 128 acc/lane, no VGPR-MSB)                                              | L — large tile *with* tails                   | Likely             | M      | Medium          |
| 7   | Encode `wmma_epilogue_chunked`/`wmma_acc_high_bank` in kernel name, both sides (COR-003)                         | Prevents silent config loss                   | Confirmed          | XS     | Low             |
| 8   | Add int8/fp32 large-tile rows (PERF-003)                                                                         | M                                             | Likely             | S      | Low             |
| 9   | Assert uniform accumulate width per run (COR-004)                                                                | Correctness                                   | Confirmed          | XS     | Low             |
| 10  | Fix `get_theoritical_gpu_gflops` for gfx1250 (COR-005)                                                           | Reporting only                                | Confirmed          | XS     | Low             |
| 11  | Gate `epilogue_lds_bytes` on `direct_store` (PERF-007)                                                           | XS now, unblocks later                        | Confirmed          | XS     | Low             |

---

## Suggested Patches

**Applied and hardware-validated** — this is the one repository file I modified (`git checkout python/operations/wmma_mapping.py` to revert; backup at `/tmp/wmma_mapping.py.bak`):

```diff
--- a/python/operations/wmma_mapping.py
+++ b/python/operations/wmma_mapping.py
@@ -327,6 +327,11 @@ ctrl_wmma_mapping_table = {
         ctrl_wmma_mapping_t(64,  128, 16, 16, 2, 4, 4, v_wmma_f32_16x16x32_f16),
+        # 256x256/256x128: mechanical port of the 'bf16' entries below -- identical
+        # tiling, only the WMMA instruction differs. Requires wmma_acc_high_bank=1
+        # (VGPR-MSB) and wmma_epilogue_chunked=1, same as bf16.
+        ctrl_wmma_mapping_t(256, 256, 16, 16, 8, 4, 8, v_wmma_f32_16x16x32_f16),
+        ctrl_wmma_mapping_t(256, 128, 16, 16, 4, 8, 4, v_wmma_f32_16x16x32_f16),
     ],
```

**Proposed, not applied** (each small and independent):

```diff
# driver/conv_driver.cpp — COR-002: initialize device buffers regardless of -V
-    if (need_verify) {
-        if(!igemm_rand_int){
+    if (true) {                       /* inputs must be initialized even when not verifying:
+                                         WMMA is ~2x faster on all-zero operands (measured) */
+        if(!igemm_rand_int){
```

```python
# python/igemm/igemm_base.py — COR-001: make the fp32 rule structural
assert not (self.precision == 'fp32'
            and self.get_igemm_gtc_fma_type() == 'wmma'
            and not self.lds_double_buffer), \
    "fp32 WMMA requires lds_double_buffer=1 (last-lane LDS visibility race)"
```

```python
# python/igemm/igemm_base.py, igemm_gtc_encode_kernel_name — COR-003
# (mirror the same two suffixes in driver/igemm_gtc_base.h)
if tunable.wmma_epilogue_chunked: kernel_name += '_chunked'
if tunable.wmma_acc_high_bank:    kernel_name += '_hibank'
```

---

## Missing Evidence

Questions that materially block firmer conclusions:

1. **Why is WMMA 2× faster on zeros?** No ISA documentation. Until characterized, I cannot state the true achievable peak for real data — 2157 TFLOPS is measured on random-normal operands only, and the answer changes every efficiency figure in this report.
2. **What does MODE.REPLAY_MODE do?** Undocumented; blocks classifying COR-006's severity.
3. **Correct partial-wait distances.** No documented WMMA/LDS hazard latencies; PERF-004's real payoff must be measured, not derived.
4. **bwd/wrw generator source unread.** I read only their assert walls and main-loop wiring, confirming they share `wmma_main_loop.py` (so PERF-001/004/006 transfer). Their direction-specific prologues/epilogues — and the wrw stream-K path — are unaudited, despite bwd (1.72×) and wrw (2.06×) having the worst gaps vs fwd (1.17×).
5. **Whether a `256×128 wr4x4 w8` mapping is derivable.** Action 6 assumes the mapping generalizes to 8 waves; the table lookup rejects it today and I did not trace whether the derivation supports it.
6. **All published benchmark numbers** are `-V 0` and therefore unusable; nothing can be concluded about MIOpen parity until Action 1 lands and the suites are re-run.

---

## Investigation Log

**Files inspected:** `python/operations/wmma_mapping.py`, `wmma_main_loop.py`, `coalescing_store_wmma.py`; `python/igemm/igemm_base.py`, `igemm_fwd_gtc_wmma_nhwc.py`, and the assert walls of `igemm_bwd_gtc_wmma_nhwc.py` / `igemm_wrw_gtc_wmma_nhwc.py`; `python/codegen_driver.py`; `driver/conv_driver.cpp`, `driver/igemm_gtc_base.h`, `driver/igemm_fwd_gtc_driver.h`; `script/build_gfx1250_master_configs.py`; ~30 `config/*.config` files across both architectures; generated `.inc`/`.s` for fp16 128×128 (base, dbuf, k64, interleave), fp32 128×128, bf16 256×256, fp16 256×256, gfx950 256×128.

**Files excluded per instruction:** all of `docs/` except `docs/architecture_map.md` (navigation only). The ISA document was never read sequentially — only §5.7 and §3.4.3, each selected from a specific code observation, via its TOC.

**Commands executed:** ~35 `igemm_codegen.py` builds; ~40 `conv_driver.exe` runs under `-V 1`; ablation builds via `clang++ -x assembler -I<dir> -target amdgcn--amdhsa -mcpu=gfx1250` + `IGEMM_HSACO=`; `hipcc --cuda-device-only -S` for the LLVM prologue reference; five standalone HIP microbenchmarks (`/tmp/wmma_occ2.hip`, `bw.hip`, `bw3.hip`, `wmma_data.hip`, `wmma_clk.hip`); `rocm-smi` sampling under sustained load.

**Environment:** gfx1250, 256 CUs, 4 MB L2, sclk 2278 MHz sustained, 1564 W package. Measured global-read bandwidth vs working set: 4 MB → 48.5 TB/s, 16 MB → 42.0, 64 MB → 33.0, 256 MB → 11.7, 1 GB → 11.4 (HBM roof ≈ 11.4 TB/s). The kernel's global phase moves 802 MB in ~0.079 ms ≈ 10.6 TB/s — i.e. **already at ~93% of the HBM roof**, which is why tile size (arithmetic intensity) is the dominant lever.