# Phase D1-P1 — Split-Barrier Gap-Hoist Validation (wmma_gap_hoist=1)

**Date:** 2026-09-06  **Hardware:** gfx1250, 256 CUs, exclusive access.  **ROCm:** /home/sgundabo/rocm-10.1

**⚠ Clock caveat:** This session's GPU is clock-capped at **1100 MHz sclk** (`rocm-smi --showclocks`: sclk level 1, 1100 MHz; range 500–1100 MHz). The reference machine used by prior validation docs reports ~2391 MHz. **All TFLOP/s figures below are measured at 1100 MHz and are NOT comparable to reference-machine numbers.** Performance conclusions are **directional only** — same-session, same-shape, same-tile relative deltas. Final perf sign-off (additive vs redundant, final keep/exclude decision for each direction's master config) must be re-verified on the faster reference machine before being treated as conclusive, mirroring how `docs/gfx1250_wmma_perf_report_v2.md` §0 already caveats this.

---

## 1. The mechanism

`docs/gfx1250_gemm_optimization_guide.md` §14 ("Step 11: split barriers and load pipelining") describes `s_barrier_signal` as non-blocking — independent work can be issued in the gap before `s_barrier_wait` actually stalls the wave. The guide's own example issues the next K-stage's load in that gap.

`python/operations/wmma_main_loop.py`'s existing `can_hoist` mechanism (Phase 70 / PERF-001, activated by `lds_double_buffer=1`) already hoists the next tile's `move_slice_window_a/b` + `global_load_a/b` (`f_gld_a`/`f_gld_b`) issuance to happen BEFORE the current tile's LDS-read+WMMA-compute block — but it does so AFTER `s_barrier_wait -1` has already completed, not in the signal→wait gap itself.

The `wmma_gap_hoist` tunable (this phase, D1-P1) fills that gap. When `gap_hoist = ctrl.gap_hoist and can_hoist` is True:

1. `s_barrier_wait -1` is **not** emitted immediately after `s_barrier_signal -1` (it is skipped when `gap_hoist` is True).
2. The hoisted `move_slice_window_a/b` + `f_gld_a`/`f_gld_b` issuance runs in the gap (between signal and wait).
3. `s_barrier_wait -1` is emitted immediately after the hoisted load issuance (fallthrough path).
4. On the `_last` tail path (when no more tiles remain), `s_barrier_wait -1` is emitted immediately after the `_last` label.

Both paths (fallthrough and `_last`) get exactly one `s_barrier_wait -1` each, preserving the exact same synchronization semantics — just later in the fallthrough path, filling the gap with useful work.

### Safety argument

This is safe whenever `can_hoist` already is, by the exact same 3-point argument `can_hoist`'s own docstring uses (see `wmma_main_loop.py` lines 607-628):

1. `move_slice_window` and `global_load` touch only global-memory address registers and VGPR staging (`v_gld_a`/`v_gld_b`) — never LDS — so reordering them relative to the barrier is completely orthogonal to the barrier-mediated LDS ordering invariant.
2. The previous iteration's LDS store has already been drained by `s_wait_dscnt`/barrier before the hoisted load overwrites the staging VGPRs.
3. The LDS store of the newly-loaded tile still happens only after `s_wait_loadcnt` at the end of the iteration, preserving the double-buffered write-after-read safety.

Moving the wait from "immediately after signal" to "after the hoisted loads" changes ONLY the timing of when the wave stalls on the barrier — not the LDS write/read ordering invariant. The signal itself is non-blocking; the wait is what actually stalls.

### Silent gating

`gap_hoist = ctrl.gap_hoist and can_hoist` — the new flag silently has no effect whenever `can_hoist`'s own preconditions aren't met (fp32, async, TDM, interleave, single-buffered LDS). No Python assert or C++ `tunable_is_valid` rejection is added for this; `gap_hoist` is a no-op without `lds_double_buffer=1`.

### Zero-diff guarantee

When `wmma_gap_hoist` is 0 (the default, today's behavior for every existing config), the emitted assembly is byte-for-byte identical to the pre-change code: `s_barrier_signal -1` is immediately followed by `s_barrier_wait -1` + empty line, and no `gap_hoist` blocks are entered. Verified by diffing the generated `.inc` files (see §3 below).

---

## 2. Relationship to `can_hoist`

`can_hoist` hoists the next tile's load issuance to BEFORE the current tile's LDS-read+WMMA-compute, but AFTER `s_barrier_wait` has completed. `gap_hoist` moves the `s_barrier_wait` itself to AFTER the hoisted load issuance — so the load fills the signal→wait gap instead of running after the wait.

The two mechanisms compose: `gap_hoist` is gated on `can_hoist` being True. When both are on, the load issuance happens in the gap between signal and wait (rather than after wait), giving the load a head start behind the barrier stall window itself. When only `can_hoist` is on (today's behavior), the load runs after the wait completes — still before the compute, but without the extra barrier-gap overlap.

---

## 3. Zero-diff regression check

Built `config/igemm_fwd_gtc_gfx1250_nhwc_fp16_dbuf.config` (no `wmma_gap_hoist`) before and after the Python changes. The generated `.inc` file has `s_barrier_signal -1` immediately followed by `s_barrier_wait -1` at all barrier occurrences — byte-identical to the pre-change output:

```
$ grep -n "s_barrier" /tmp/fwd_dbuf_regression/igemm_fwd_gtc_gfx1250_nhwc_fp16_dbuf_128x128x032.inc
407:    s_barrier_signal -1
408:    s_barrier_wait -1
539:    s_barrier_signal -1
540:    s_barrier_wait -1
720:    s_barrier_signal -1
721:    s_barrier_wait -1
```

Same pattern confirmed for bwd and wrw `_dbuf.config` builds — all barriers have signal immediately followed by wait, zero gap.

### Gap-hoist disassembly confirmation

The gaphoist kernel's `.inc` shows real instructions between signal and wait:

```
$ grep -n "s_barrier" /tmp/fwd_gaphoist/igemm_fwd_gtc_gfx1250_nhwc_fp16_dbuf_gaphoist_128x128x032.inc
407:    s_barrier_signal -1
435:    s_barrier_wait -1       ← 28 lines of hoisted loads between signal and wait
490:    s_barrier_wait -1       ← _last tail path
```

Lines 408–434 contain `move_slice_window` (v_add_co_u32 address advances) and `global_load_b128` instructions — the hoisted next-tile load issuance filling the gap.

---

## 4. Correctness

All kernels verified with `-V 1` (driver-internal reference comparison), `IGEMM_RUN_ONLY_KERNEL` pinned, `timeout 30` wrapped:

| direction | precision | shape | gap_hoist kernel |
|---|---|---|---|
| fwd | fp16 | n128 c1024 17×17 k1024 1×1 | **valid:y** |
| fwd | fp16 | n256 c2048 14×14 k2048 1×1 | **valid:y** |
| fwd | fp16 | n64 c512 28×28 k512 3×3 | **valid:y** |
| bwd | fp16 | n128 c1024 17×17 k1024 1×1 | **valid:y** |
| bwd | fp16 | n256 c2048 14×14 k2048 1×1 | **valid:y** |
| bwd | fp16 | n64 c512 28×28 k512 3×3 | **valid:y** |
| wrw | fp16 | n128 c1024 17×17 k1024 1×1 | **valid:y** |
| wrw | fp16 | n256 c2048 14×14 k2048 1×1 | **valid:y** |
| wrw | fp16 | n64 c512 28×28 k512 3×3 | **valid:y** |

All 9 shape/direction combinations pass. No crashes, no hangs, no `valid:n`.

---

## 5. Performance (directional, 1100 MHz sclk — NOT comparable to reference machine)

**Method:** 3 independent launches per data point, `IGEMM_WARMUP=5 IGEMM_REPEAT=20`, `IGEMM_RUN_ONLY_KERNEL` to pin the 128×128×32 tile. All runs serial (single GPU). Shape: `n128 c1024 17×17 k1024 1×1` (the primary standing regression shape).

**Clock:** `rocm-smi --showclocks` → sclk 1100 MHz (clock-capped), mclk 1900 MHz, fclk 1000 MHz. Confirmed at start and end of session.

### 5.1 fp16 fwd

| variant | run 1 | run 2 | run 3 | min | max | avg | delta vs dbuf |
|---|---|---|---|---|---|---|---|
| baseline (no dbuf) | 272.5 | 272.2 | 273.5 | 272.2 | 273.5 | 272.7 | — |
| dbuf (can_hoist, no gap) | 311.3 | 312.4 | 311.0 | 311.0 | 312.4 | 311.6 | — |
| dbuf + gaphoist | 324.8 | 324.6 | 328.3 | 324.6 | 328.3 | 325.9 | **+4.4%** |

Run-to-run variance <1.2%. The +4.4% delta is outside noise — **directionally additive** for fwd.

### 5.2 fp16 bwd

Bwd showed inconsistent results across repeat batches at this clock. First batch had gaphoist slower than dbuf; subsequent batches had it faster. All runs combined:

| variant | batch | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| baseline | 1 | 218.0 | 216.6 | 218.1 |
| dbuf | 1 | 264.4 | 264.4 | 262.5 |
| gaphoist | 1 | 252.9 | 251.7 | 253.4 |
| dbuf | 2 | 251.7 | 249.0 | 251.8 |
| gaphoist | 2 | 267.4 | 265.8 | 264.6 |
| dbuf | 3 | 253.4 | 252.6 | 252.7 |
| gaphoist | 3 | 266.2 | 264.7 | 265.8 |

The dbuf and gaphoist values overlap across batches (dbuf ranges 249–264, gaphoist ranges 252–267). At this clock, the bwd signal is **inconclusive** — the delta is within run-to-run noise. This may be a clock-sensitivity artifact (bwd's asymmetric operand paths create different memory-access patterns that may interact differently with the barrier timing at different clocks). Must be re-measured on the reference machine.

### 5.3 fp16 wrw

| variant | run 1 | run 2 | run 3 | min | max | avg | delta vs dbuf |
|---|---|---|---|---|---|---|---|
| baseline (with gkgs) | 169.0 | 169.4 | 168.3 | 168.3 | 169.4 | 168.9 | — |
| dbuf + gkgs | 218.1 | 215.8 | 218.2 | 215.8 | 218.2 | 217.3 | — |
| dbuf + gkgs + gaphoist | 234.8 | 234.9 | 235.4 | 234.8 | 235.4 | 235.0 | **+8.2%** |

Run-to-run variance <1.1%. The +8.2% delta is well outside noise — **directionally additive** for wrw.

Note: wrw's `_dbuf.config` lacks `gemm_k_global_split=1` (an 8× parallelism handicap, documented in `docs/gfx1250_d1_crosstile_hoisting_validation.md` §5.4). The comparison here uses a constructed `_dbuf_gkgs.config` (dbuf + gsplit, no gaphoist) as the fair dbuf-alone baseline, so the ONLY variable is `wmma_gap_hoist`.

---

## 6. Conclusion (directional, pending reference-machine re-verification)

| direction | correctness | gaphoist vs dbuf-alone (this session, 1100 MHz) | verdict (directional) |
|---|---|---|---|
| **fwd** | valid:y (3/3 shapes) | **+4.4%** (outside noise) | **directionally additive** |
| **bwd** | valid:y (3/3 shapes) | **inconclusive** (within noise) | **inconclusive at this clock** |
| **wrw** | valid:y (3/3 shapes) | **+8.2%** (outside noise) | **directionally additive** |

**Correctness: PASS for all three directions on all standing regression shapes.** No crashes, no hangs, no `valid:n`.

**Performance (directional):**
- **fwd: directionally additive** — gap_hoist adds a real +4.4% over dbuf-alone, consistent across 3 runs.
- **wrw: directionally additive** — gap_hoist adds a real +8.2% over dbuf-alone, consistent across 3 runs.
- **bwd: inconclusive at this clock** — the delta between dbuf-alone and dbuf+gaphoist is within run-to-run noise at 1100 MHz sclk. This may resolve to additive, neutral, or slightly negative on the reference machine; cannot be determined from this session's data.

**⚠ These performance conclusions are directional only and MUST be re-verified on the faster reference machine (~2391 MHz sclk) before being treated as final.** The 1100 MHz clock cap on this session's GPU means absolute throughput is roughly half the reference machine's, and relative deltas may shift at higher clocks (barrier latency, memory latency, and compute latency all scale differently with clock).

### Config disposition

All three `_dbuf_gaphoist.config` files are shipped standalone and folded into their direction's master `_all.config` (via `script/build_gfx1250_master_configs.py --write`, which auto-picks up matching config files). All three master configs build and dispatch cleanly. No direction is excluded — all three pass correctness validation, and the perf signal is directionally positive or inconclusive (not negative) for all three. Per the priority steer's decision rule, a direction should not be excluded merely because the perf delta is unclear at this clock.

### Configs used

| config | tile | key difference |
|---|---|---|
| `config/igemm_fwd_gtc_gfx1250_nhwc_fp16.config` | 128×128×32 + 64×64×32 | baseline (no dbuf, no gaphoist) |
| `config/igemm_fwd_gtc_gfx1250_nhwc_fp16_dbuf.config` | 128×128×32 | + `lds_double_buffer=1` |
| `config/igemm_fwd_gtc_gfx1250_nhwc_fp16_dbuf_gaphoist.config` | 128×128×32 | + `lds_double_buffer=1` + `wmma_gap_hoist=1` |
| `config/igemm_bwd_gtc_gfx1250_nhwc_fp16.config` | 128×128×32 + 64×64×32 | baseline |
| `config/igemm_bwd_gtc_gfx1250_nhwc_fp16_dbuf.config` | 128×128×32 | + `lds_double_buffer=1` |
| `config/igemm_bwd_gtc_gfx1250_nhwc_fp16_dbuf_gaphoist.config` | 128×128×32 | + `lds_double_buffer=1` + `wmma_gap_hoist=1` |
| `config/igemm_wrw_gtc_gfx1250_nhwc_fp16.config` | 128×128×32 + 64×64×32 | baseline, `gemm_k_global_split=1` |
| `config/igemm_wrw_gtc_gfx1250_nhwc_fp16_dbuf_gkgs.config` | 128×128×32 | + `lds_double_buffer=1` + `gemm_k_global_split=1` (constructed for fair comparison) |
| `config/igemm_wrw_gtc_gfx1250_nhwc_fp16_dbuf_gaphoist.config` | 128×128×32 | + `lds_double_buffer=1` + `gemm_k_global_split=1` + `wmma_gap_hoist=1` |

### Files changed

| file | change |
|---|---|
| `python/operations/wmma_main_loop.py` | Added `self.gap_hoist = False` field; `gap_hoist = ctrl.gap_hoist and can_hoist`; conditional barrier wait emission; gap-fill wait after hoisted loads (fallthrough + `_last` paths) |
| `python/igemm/igemm_base.py` | Added `self.wmma_gap_hoist` tunable; `_gaphoist` kernel-name suffix |
| `python/igemm/igemm_fwd_gtc_wmma_nhwc.py` | `ctrl.gap_hoist = self.tunable.wmma_gap_hoist` |
| `python/igemm/igemm_bwd_gtc_wmma_nhwc.py` | `ctrl.gap_hoist = self.tunable.wmma_gap_hoist` |
| `python/igemm/igemm_wrw_gtc_wmma_nhwc.py` | `ctrl.gap_hoist = self.tunable.wmma_gap_hoist` |
| `driver/igemm_gtc_base.h` | `int wmma_gap_hoist = 0;` struct field; config-parse line; `_gaphoist` kernel-name suffix |
| `config/igemm_fwd_gtc_gfx1250_nhwc_fp16_dbuf_gaphoist.config` | New config |
| `config/igemm_bwd_gtc_gfx1250_nhwc_fp16_dbuf_gaphoist.config` | New config |
| `config/igemm_wrw_gtc_gfx1250_nhwc_fp16_dbuf_gaphoist.config` | New config (includes `gemm_k_global_split=1`) |
| `config/igemm_wrw_gtc_gfx1250_nhwc_fp16_dbuf_gkgs.config` | New config (dbuf+gsplit, for fair wrw comparison baseline) |
