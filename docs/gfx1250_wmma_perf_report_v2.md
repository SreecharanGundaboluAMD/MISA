# gfx1250 WMMA Convolution — Performance Report v2 (updated)

**Supersedes** `docs/gfx1250_wmma_perf_report_2026-09-02.md` (v1). This revision updates v2
(originally dated 2026-09-03, HEAD `2af9f24`) with a review of the 6 commits landed since
(`2af9f24..b542b08`), fresh hardware verification of every still-open defect, two newly
discovered critical defects, and a re-derived priority plan cross-checked against
`docs/gfx1250_gemm_optimization_guide.md` and the current repo state.

**Date:** 2026-09-06 · **HEAD:** `b542b08` · **Hardware:** gfx1250, 256 CU, wave32.

---

## Resolution status (2026-09-06 follow-up session, HEAD `12d3c0d`)

All six defects below (R1–R6) plus the three Housekeeping items (§7 item 7) were
addressed in a follow-up session the same day. Hardware-validated on the full standing
regression set (§9) across fwd/bwd/wrw, all three `_all.config` masters rebuilding and
dispatching cleanly (`valid:y` or correctly-rejected `not applicable`, zero crashes, zero
hangs). Per-defect disposition:

- **R1** (`b367d93`) — fixed. `threads_per_krow_{a,b}` computation/assert moved inside
  `if tunable.lds_row_pad > 0:`. All three `_all.config` masters build.
- **R2** (`12d3c0d`) — fixed. `wmma_fp16_output=1` config variants added for fwd/bwd/wrw
  fp16 (`*_f16o.config`), excluded from the `_all.config` union via the same
  accumulate-width-hazard mechanism as `wmma_acc_f16`/`bf16`/`atomic_pack_bf16` (buffer
  width computed once from `tunables[0]`, not per-tunable) — reachable and build+dispatch
  verified standalone.
- **R3** (`175f455`) — fixed. `fp_factor` reverted to the physical value 8 for WMMA
  fp16/bf16 (matches XDLOPS' existing use of 8); comments corrected.
- **R4** (`472f274`) — fixed. `wrw_streamk` on a non-unit-conv (3×3) shape now correctly
  reports "not applicable" instead of dispatching and returning `valid:n` at impossible
  >100% "efficiency".
- **R5** (`ed870e6`) — **fixed, root-caused.** `igemm_bwd_gtc_wmma_nhwc.py`'s
  `get_kernel_args()` was missing the `gemm_k_per_wg` argument declaration at kernarg
  offset 88 from the PAL metadata `.args` array (fwd and wrw both declare it; bwd's own
  comment claimed it was "always present" but the `kas.append()` call was never written).
  The 4-byte hole meant the ROCm runtime did not reliably populate that dword on the
  device, which read stale/garbage data instead — exactly the observed
  `HSA_STATUS_ERROR_MEMORY_FAULT`/silent-corruption symptom. One-line fix (add the missing
  `kas.append`), hardware-validated on every previously-crashing shape plus the full
  standing regression set, `tunable_is_valid`'s rejection removed. Two earlier hypotheses
  (SGPR allocation order/adjacency, gfx1250 hardware kernarg preloading) were tested and
  ruled out with hardware evidence before the real cause was found — see
  `docs/gfx1250_bwd_gsplit_memory_fault.md` for the full trail.
- **R6** (`980dad1`, then `a3b1476`) — **disproven, not a real hardware bug.** The original
  hang report was itself a test-driver artifact: `conv_driver.cpp`'s host-side RNG fill
  (`gen_rand_vector`) and `tensor_copy_cpu.h`'s tensor copy both had severe false-sharing
  in their multi-threaded chunking (per-index round-robin writes across threads, not
  contiguous per-thread ranges), which at large shapes inflated *host-side setup time*
  to 60+ seconds — indistinguishable from a device hang under a fixed `timeout N` with no
  progress output. Fixed at the actual root cause (`c56fb4e`, contiguous per-thread chunking)
  instead: the documented "hang" shape now completes correctly and near-instantly for every
  previously-implicated kernel (`k2x`, `dbuf`, `bf16_k2x_bf16acc`) with repeated runs and no
  GPU health issues. The defensive grid-rejection guard added in `980dad1` was reverted in
  `a3b1476` as unnecessary product-behavior narrowing for a bug that does not exist; see
  `docs/gfx1250_wrw_full_cu_grid_hang.md` for the full account.
- Housekeeping — fixed. `config/w6_test_*.config` (4 stray files) deleted; the
  `wg_swizzle`/`workgroup_swizzle` tunable removed entirely (Python generators, kernel-name
  encoding, C++ driver struct/parse/encode — 0 configs used it); `wmma_acc_bf16` was
  already excluded from every master `_all.config`'s "recommended" search path by the
  pre-existing accumulate-width-hazard exclusion (confirmed: 0 `wmma_acc_bf16=1` sections in
  any `*_all.config` today) — no further action needed.

**Net effect:** R1's fix alone unblocks wrw's entire master-config search (previously
couldn't build at all), which in turn makes §5's `lds_row_pad` finding reachable for wrw for
the first time. R5 turned out to be a genuine, fully-root-caused fix (a missing PAL metadata
argument declaration), not merely a mitigation — bwd `gemm_k_global_split` is safe to use
again and its 2 sections are already reachable in the master `_all.config`. R6 turned out to
require no product change at all once the measurement artifact was fixed.

---

## 0. Machine caveat — read this before comparing any numbers

**This verification session's GPU is clock-capped at 1100 MHz sclk** (`rocm-smi
--showsclkrange`: 500–1100 MHz; confirmed via `rocm-smi --showclocks`). The v1/v2 reports and
every `docs/gfx1250_d1_*_validation.md`/`docs/gfx1250_w*.md` doc were measured on a different,
faster machine (v2 cites `dev_prop.clockRate` = 2391 MHz; commit `4ef933d` is titled
"re-benchmark on faster machine"). **Absolute TFLOP/s figures are not comparable across
machines or across documents in this repo.** This is compounded by R3 below (`fp_factor=9`
makes the driver's own efficiency-% denominator track whichever clock it happens to read).

Every new number in this revision is a **same-session, same-shape, same-tile relative
delta** (variant vs. baseline, both measured back-to-back on this machine at a pinned
1100 MHz) — that comparison is valid regardless of absolute clock. Historical absolute
numbers from other docs are cited only when explicitly attributed to their source. Where
this revision's relative deltas disagree with a prior doc's relative deltas for the *same*
mechanism (they do, substantially, for `lds_row_pad` — see §5), that disagreement is itself
evidence that this codebase's performance characteristics are machine-sensitive and every
future finding needs the actual clock recorded (`IGEMM_SCLK_MHZ` or `rocm-smi --showclocks`)
alongside the number.

---

## 1. Executive summary

Since v2 (`2af9f24`), 6 commits landed: one real correctness fix (wrw k2x), one epilogue
extension (C2 widen), and three pure-validation commits confirming pre-existing but
never-benchmarked mechanisms are real wins. **None of v2's four defects (R1–R4) were
fixed.** All four are reconfirmed below with fresh, live reproductions on this machine.
**Two new critical defects were found during this review** — one is a GPU memory-fault
crash, the other is an indefinite hang — neither previously documented anywhere in this
repo.

| id | defect | status | severity |
|---|---|---|---|
| R1 | `igemm_wrw_gtc_gfx1250_nhwc_fp16_all.config` still fails to build (`threads_per_krow(0)` assert) | **FIXED `b367d93`** — see Resolution status above | **blocker** |
| R2 | `wmma_fp16_output` still reachable from 0 configs (now with more code behind it: `c52944a` extended it to the non-atomic epilogue too) | **FIXED `12d3c0d`** — see Resolution status above | high |
| R3 | `fp_factor=9` still non-physical; comment added acknowledging it, value unchanged | **FIXED `175f455`** — see Resolution status above | medium |
| R4 | `wrw_streamk`'s missing `nxe==0` runtime rejection still absent from the WMMA branch of `tunable_is_valid` | **FIXED `472f274`** — see Resolution status above | medium-high |
| **R5 (new)** | **bwd `gemm_k_global_split=1` GPU memory-fault crash** on `n128 c1024 17×17 k1024` (`HSA_STATUS_ERROR_MEMORY_FAULT`, `hipEventSynchronize` returns illegal-memory-access) | **FIXED, root-caused `ed870e6`** — see Resolution status above and `docs/gfx1250_bwd_gsplit_memory_fault.md` | ~~critical~~ n/a |
| **R6 (new)** | **wrw hangs indefinitely** (not `valid:n` — an unrecoverable device-side hang requiring `timeout`) on any non-split-K tunable (`k2x`, `dbuf`, `bf16acc`, and by extension `interleave`) whenever the shape's GEMM grid exactly equals 256 (= CU count) | **DISPROVEN — test-driver artifact, not a hardware bug (`a3b1476`, root-caused fixed by `c56fb4e`)** — see Resolution status above and `docs/gfx1250_wrw_full_cu_grid_hang.md` | ~~critical~~ n/a |

The **single biggest opportunity is not the one v2 named.** v2's "compute/memory overlap"
diagnosis is now real code (`b542b08` validated it as `lds_double_buffer=1`'s existing
`can_hoist` path), delivering a genuine but modest **+10–20%** relative win, confirmed again
in this session. But **`lds_row_pad` (LDS bank-conflict padding, shipped since before v2) is
a far larger, already-implemented, still-barely-exploited lever**: measured today at
**+41% to +98%** relative over baseline on the exact same tile shape, across fwd, bwd, *and*
wrw — roughly 2–5× the size of the hoisting win — and it is **not combined with
`lds_double_buffer` in a single shipped section anywhere in the repo**, nor is it reachable
for wrw at all (blocked by R1). Stacking the two mechanisms is untested and is this report's
top concrete recommendation (§5, §7 P0.5).

---

## 2. Commit-by-commit review since v2 (`2af9f24..b542b08`)

| # | commit | verdict | note |
|---|---|---|---|
| 1 | `c52944a` C2: widen fp16 output store to `dwordx4` | **OK, still inert** | Extends `wmma_fp16_output` from the `direct_store` epilogue (C1) to the default non-atomic LDS-reshuffle epilogue too — real, more general code. Does not change R2: still 0 configs set `wmma_fp16_output=1`, so none of this is reachable from `conv_driver.exe`'s normal candidate search. |
| 2 | `5d1d601` D1: validate `main_loop_interleave`/`local_prefetch_num` (fwd) | **OK (docs), self-correcting** | Confirms fwd's existing `_interleave.config` (already in `_all.config` since 2026-08-25) is a real +4.5–7.0% win; confirms `local_prefetch_num=2` is VGPR-infeasible for plain fp32-accumulate fp16, only reachable combined with `wmma_acc_f16`. No code change. |
| 3 | `10e63c5` **Fix wrw k2x `s_wait_loadcnt` correctness bug** | **real fix, OK** | One-line fix: `_emit_sst_all_chunks` (wrw, `gemm_k_per_block>inst_wmma.k` path) was missing `s_wait_loadcnt 0x0` between issuing the global load and storing it to LDS, reading stale VGPR data. Matches the identical pattern already present in every analogous fwd/bwd/wrw method. Fixes `valid:n` on `n256 c2048 14×14 k2048` (full 256-CU grid) for wrw k2x. **However — see R6: on this machine, the exact shape this fix was validated against now hangs instead of running `valid:n`**, for `k2x` and unrelated non-split-K tunables alike. The correctness fix is real and necessary but did not, by itself, make this shape reliably runnable. |
| 4 | `c224145` D1: validate `main_loop_interleave` for bwd/wrw | **OK (docs)**, but see §6 contradiction | bwd: +3.6–4.0% (2 shapes). wrw: +8.5% (1 shape), gated on commit 3's fix. Also corrects an inaccurate claim in the prior fwd validation doc (a config already existed; the agent had rebuilt an equivalent one from scratch instead of finding it — a discovery-before-authoring process gap worth naming explicitly for future validation tasks). |
| 5 | `b542b08` D1: validate cross-tile hoisting (`lds_double_buffer=1`'s `can_hoist` path) | **OK, real win, reconfirmed independently today** | +10.2–15.7% across fwd/bwd fp16/bf16 on the reference machine; this session independently reproduces a comparable +10.6–19.5% relative delta on a *different* machine (§5). The wrw comparison in that doc is explicitly flagged by its own author as confounded (dbuf config lacks `gemm_k_global_split=1`, an 8× parallelism handicap) — correctly not read as a wrw regression. |
| — | `15310a5`, `ea1ed03` | session bookkeeping (transcript exports) | no code/doc content requiring review |

None of these six commits touch `driver/conv_driver.cpp` (R3), `driver/igemm_wrw_gtc_driver.h`
(R4), `python/igemm/igemm_wrw_gtc_wmma_nhwc.py`'s `threads_per_krow` assert (R1), or add any
`wmma_fp16_output=1` config (R2). All four are exactly where v2 left them.

---

## 3. Defects reconfirmed live, with today's evidence

### 3.1 R1 — wrw master config still does not build

```
$ python3 igemm_codegen.py config/igemm_wrw_gtc_gfx1250_nhwc_fp16_all.config -d /tmp/x
AssertionError: threads_per_krow(0) must be > 0 and a power of 2
  at python/igemm/igemm_wrw_gtc_wmma_nhwc.py:290
```
Root cause unchanged from v2 (`igemm_wrw_gtc_wmma_nhwc.py:285-291`): `threads_per_krow_{a,b}`
is computed and asserted **unconditionally for every wrw WMMA section**, not just when
`lds_row_pad>0` — it was written as part of the B4 padding feature but the assert isn't
gated on the feature being in use. The current `_all.config` unions in 3 deep-K sections
(`_32x32_k96.config`, `_64x64_k128*.config` ×3, `_64x64_k256.config`) where
`gemm_k_per_block > gemm_{m,n}_per_block` makes `threads_per_krow` truncate to 0 —
independent of whether that section also sets `lds_row_pad`. Confirmed today: these deep-K
files build fine **standalone** (they predate `lds_row_pad`'s existence and are unaffected by
it in isolation) — the failure only appears once the union pulls every wrw section into one
`__init__` pass, which is the master config's entire purpose. **Every downstream benefit in
this report that depends on the fastest-tunable search — including the `lds_row_pad`
finding in §5 — is unreachable for wrw until this is fixed.** This remains the single
highest-leverage fix in the repo: one file, ~6 lines, unblocks correctness-search coverage
for an entire direction.

**Fix** (unchanged recommendation from v2, still not applied): move the
`threads_per_krow_{a,b}` computation and both its asserts inside `if tunable.lds_row_pad > 0:`
(lines 285-291 must move below the existing `if` at line 275); for the `lds_row_pad==0` case,
either don't compute `threads_per_krow` at all (nothing downstream needs it unpadded — grep
confirms its only other use, line 744, is inside the same `if lds_row_pad>0` block already),
or explicitly assert `gemm_k_per_block <= min(gemm_m_per_block, gemm_n_per_block)` with a
clear message if you want to keep the invariant documented. Then re-run
`script/build_and_filter_configs.py` and confirm all three `_all.config` files build.

### 3.2 R2 — `wmma_fp16_output` still unreachable, now with more code behind it

```
$ grep -rl "wmma_fp16_output" config/*.config | wc -l
0
```
`c52944a` (since v2) made the mechanism *more* complete — both epilogue paths (direct-store
and the default LDS-reshuffle) now support it — which makes non-adoption a larger sunk cost,
not a smaller one. Still validated at 1.087–1.145× in its own commit's measurements, still
zero configs enable it.

### 3.3 R3 — `fp_factor=9` still non-physical

Unchanged (`driver/conv_driver.cpp:197,213`). A comment block was added (lines 238-249,
pre-existing from before v2) explaining a *different*, already-fixed bug (`num_simd`
defaulting to 64 instead of 128 for gfx1250) but does not touch `fp_factor`. Given §0's
finding that this session's machine reports a completely different clock (1100 MHz vs.
2391 MHz), the practical impact of this defect is now directly visible: the same kernel's
printed "efficiency %" is meaningless across machines, and — see R5/R6 below — a crashing or
hung kernel can print a *higher* "efficiency" than a correct one right up until it faults,
because the % is computed from wall-clock time with no correctness gate.

### 3.4 R4 — `wrw_streamk`'s `nxe==0` rejection still missing, reproduced live

```
$ ./conv_driver.exe convfp16 -n 64 -c 512 -H 28 -W 28 -k 512 -y 3 -x 3 -p 1 -q 1 \
    -u 1 -v 1 -F 4 -V 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
[wrw:0] ..._streamk_dstrb_gkgs[224], cost:0.382ms, tflops:619.075(95.42%), valid:n
[wrw:1] ..._streamk_dstrb_gkgs[224], cost:0.328ms, tflops:722.854(111.41%), valid:n
```
Confirmed by code inspection this session: `driver/igemm_wrw_gtc_driver.h`'s
`tunable_is_valid`, WMMA branch (lines 280-339), unconditionally `return true;` at line 338
once its own tail/tdm checks pass — it never reaches the generic `if (!unit_conv &&
tunable->nxe==0) return false;` guard at line 438 (confirmed via `git blame`: that guard
predates gfx1250 WMMA entirely, added 2025-07-21 for the NCHW/scalar paths, and the WMMA
branch's early return simply bypasses it). `wrw_streamk` is asserted `nxe==0` at
config-construction time (`igemm_base.py:995`), so it is always exactly the case this guard
exists to catch — it just never runs for WMMA tunables. Also note the printed >100%
"efficiency" on a `valid:n` result, a second, independent illustration of R3's practical
cost.

**Fix** (unchanged from v2): add `if (!unit_conv) return false;` inside the WMMA branch when
`tunable->wrw_streamk` is set (or, more generally, when `tunable->nxe==0`, matching the
existing generic-path semantics), before line 338's `return true;`.

---

## 4. New defects found this session

### 4.1 R5 — bwd `gemm_k_global_split=1` crashes with a GPU memory fault

Reproduced twice independently (once inside a 3-shape sweep, once isolated):

```
$ IGEMM_RUN_ONLY_KERNEL=igemm_bwd_gtcw_..._dstrb_gkgs ./conv_driver.exe convfp16 \
    -n 128 -c 1024 -H 17 -W 17 -k 1024 -y 1 -x 1 -p 0 -q 0 -F 2 -V 1 ...
Warning: Queue error - HSA_STATUS_ERROR_MEMORY_FAULT
:0:rocdevice.cpp:3891: ... Memory Fault Error [... faulting addr: 0x754eeda07000,
    kernel: igemm_bwd_gtcw_nhwc_fp16_..._dstrb_gkgs]
[hiperror](700) fail to call hipEventSynchronize(stop),(an illegal memory access was encountered)
```

The same kernel on a *different* shape (`n256 c2048 14×14 k2048`, also inside the sweep)
independently printed `valid:n` at an impossible 227–303% "efficiency" rather than crashing
— i.e. this is not a single-shape edge case, it is bwd's `gemm_k_global_split` epilogue
producing genuinely undefined behavior (out-of-bounds write, given the memory fault) that
manifests as either a crash or silent wrong-answer depending on shape. **Not present in any
prior report or backlog doc.** `bwd`'s atomic/split epilogue is comparatively new
(`3dd26ce` enabled `gemm_k_global_split` by default for **wrw**, not bwd — bwd's own gsplit
sections appear to have shipped without an equivalent correctness pass). GPU health was
confirmed intact after the fault (a subsequent unrelated fwd kernel ran and validated
normally) — the fault is confined to this kernel/shape pairing, not a permanent device
failure.

**Recommendation:** treat every `bwd *_gkgs*` config as untrusted until root-caused. Given
the memory-fault signature (not merely a wrong numeric result), this is higher severity than
R4 — a user benchmarking bwd today can crash their process. Bisect against `bwd`'s
`gemm_k_global_split` epilogue in `coalescing_store_wmma.py`/`igemm_bwd_gtc_wmma_nhwc.py`
the same way `10e63c5` bisected wrw's k2x bug: compare against the working non-split bwd
path and fwd's own (working) gsplit epilogue for the missing wait/bounds check. Reproduction
shape: `-n 128 -c 1024 -H 17 -W 17 -k 1024 -y 1 -x 1 -F 2 -V 1`, kernel name pattern
`*_dstrb_gkgs` (`IGEMM_RUN_ONLY_KERNEL` to isolate).

### 4.2 R6 — wrw hangs (not `valid:n`) at exactly grid == CU count without split-K

Reproduced independently on **three** different tunables, all on the identical shape
(`n256 c2048 14×14 k2048`, 1×1 — wrw's GEMM is `M=k/group=2048, N=c/group=2048`, which tiles
to exactly `16×16=256` 128×128 workgroups, matching this GPU's 256 CUs 1:1):

| tunable | `gemm_k_global_split` | result |
|---|---|---|
| `fp16_k2x` (`gemm_k_per_block=64`) | not set | **hang** (`timeout 45`, zero output, even at `-V 0`, `IGEMM_WARMUP=0 IGEMM_REPEAT=1`) |
| `fp16_dbuf` (`gemm_k_per_block=32`) | not set | **hang** (`timeout 30`, zero output) |
| `bf16_k2x_bf16acc` | not set | **hang** (`timeout 30`, zero output) |
| `fp16` (default, shipped) | **set** (`gkgs[4]`) | completes normally, `valid:y` |
| `fp16_dbuf` on smaller grids (64, 16 workgroups) | not set | completes (slow, grid-starvation warning as expected), `valid:y` |

This is a genuine device-side hang, not a slow kernel: `-V 0` with zero warmup and a single
repeat still produces no output within 30-45 seconds, versus ~1-2 seconds for every other
kernel at this shape. `rocm-smi` showed the GPU still reporting 100% utilization after the
hung process was killed by `timeout`; a subsequent, unrelated fwd kernel run immediately
after confirmed the device itself was not wedged (it completed normally) — so the hang is
confined to this exact (shape, no-split-K) pairing, reproducibly, not a one-off driver fluke.

This was not previously documented anywhere in this repo, including the `wrw_streamk`
design doc and the grid-starvation warning path (which handles `grid < CU count`, not
`grid == CU count` exactly). It is a plausible instance of the same class of issue as
`docs/gfx1250_fp32_wmma_occupancy_race.md` (barrier-visibility failure at high occupancy) —
that doc's mechanism is a **silent** staleness bug at occupancy, not a hang, but both are
triggered by "every CU simultaneously saturated, no idle SIMD slack anywhere to break a
dependency cycle." Both symptoms are consistent with a workgroup barrier or an
epilogue synchronization point that assumes at least one other CU can always make forward
progress — which stops being true when literally every CU is running an identical,
lockstep-synchronized kernel with no split-K stagger to desynchronize them.

**Practical implication, immediately actionable:** `3dd26ce`'s decision to default-enable
`gemm_k_global_split=1` for wrw was already justified on performance grounds (v2's
4.7–17.7× headline win) — this session shows it also happens to route around R6 entirely,
because split-K changes the grid shape away from the exact 256-workgroup collision. **Any
wrw config or future tunable that does *not* set `gemm_k_global_split=1` should not be
trusted on full-CU-grid shapes until this is root-caused** — that includes `k2x`, `dbuf`,
`interleave` (gated the same way as `k2x`, same `num_k_chunks>1` code path,
untested-but-suspect), and every `*_bf16acc`/`*_f16acc` variant that doesn't separately
carry gsplit. `script/build_and_filter_configs.py` and any future benchmarking script MUST
wrap every dispatch in a timeout (this session used `timeout 30-45`) — an un-timeout-guarded
sweep across the master config's candidate list would have hung indefinitely on this shape.

**Recommendation:** reproduce under `rocgdb`/`rocprof` with a short timeout to see which
instruction address the hung wave(s) are stuck at (barrier wait is the prime suspect given
the fp32-race precedent); check whether the same collision (`grid == CU count`) reproduces
for fwd/bwd's own gsplit-free tunables at a shape engineered to hit it (fwd/bwd's GEMM
mapping makes this much rarer — their grid is `ho*wo*n` × `k`, not `k/group × c/group`, so
naturally-occurring 1:1 collisions are far less likely, but a synthetic shape should be
constructed to test whether this is wrw-specific or a general full-occupancy barrier issue).

---

## 5. `lds_row_pad`: the largest lever in the repo, still barely exploited

All numbers in this table are same-session relative deltas at pinned 1100 MHz sclk, same
tile shape (`128×128×32` unless noted), baseline vs. `lds_row_pad=16` only (no other tunable
differs):

| direction | shape | baseline TFLOP/s | `lds_row_pad=16` TFLOP/s | relative Δ |
|---|---|---|---|---|
| fwd | `n256 c2048 14×14 k2048` (1×1) | 353.2 | 581.6 | **+64.7%** |
| fwd | `n128 c1024 17×17 k1024` (1×1) | 275.6 | 404.7 | **+46.8%** |
| fwd | `n64 c512 28×28 k512` (3×3) | 307.0 | 432.0 | **+40.7%** |
| bwd | `n256 c2048 14×14 k2048` (1×1) | 267.9 | 530.9 | **+98.2%** |
| bwd | `n64 c512 28×28 k512` (3×3) | 240.7 | 476.5 | **+98.0%** |
| wrw | `n128 c1024 17×17 k1024` (1×1, both w/ `gkgs`) | 167.0 | 279.8 | **+67.5%** |
| wrw | `n64 c512 28×28 k512` (3×3, both w/ `gkgs`) | 173.1 | 283.0 | **+63.5%** |

For comparison, this session's independent re-measurement of `b542b08`'s cross-tile-hoisting
(`lds_double_buffer=1`) win on the identical shapes was **+10.6% to +19.5%** — real, but
4-9× smaller than `lds_row_pad`'s effect. The mechanism (breaking the 64 B unpadded LDS row
stride's alias onto 4-of-64 bank groups, per the tunable's own docstring) predates this
report — it shipped before v2, cited there as "measured +9-27% on 1x1 shapes" on the faster
reference machine. **The 41-98% seen here vs. the 9-27% seen there, for the identical
mechanism, is itself a finding**: LDS-bank-conflict cost is apparently far more expensive
relative to compute at this machine's clock/timing than at the reference machine's — exactly
the kind of machine-sensitivity §0 warns about, and a concrete reason every future
performance claim in this codebase should travel with its measured clock.

**Why this matters more than any single item in v2's priority list:** despite being the
largest win measured to date in either report, coverage is minimal and two mechanisms that
should compose are never tested together:

- fwd/bwd `_all.config`: exactly 2 `lds_row_pad` sections each (128×128 and 64×64,
  `gemm_k_per_block=32` only) — not combined with `dbuf`, `interleave`, `k2x`, `saddr`, or
  `tdm`.
- wrw: **zero** reachable sections (blocked entirely by R1 — the `_ldsrp.config` file exists
  and builds standalone, but can never be searched by `conv_driver.exe` until the master
  config builds).
- **No config anywhere sets both `lds_row_pad=16` and `lds_double_buffer=1`.** These are
  different mechanisms (bank-conflict-free row stride vs. cross-tile load/compute overlap)
  with no obvious interaction — grep of `wmma_main_loop.py`'s `can_hoist` gate (§ correctness
  conditions, `docs/gfx1250_d1_crosstile_hoisting_validation.md` §1) shows no dependency on
  row pitch at all. Stacking them is the single cheapest, highest-expected-value experiment
  this report can point at: if the two effects are even partially additive, this is a
  larger win than everything else in this document combined.

**Recommendation (P0.5, do this before anything else in §7):** add `lds_row_pad=16` to the
existing `_dbuf.config` / `_interleave.config` / `_k2x.config` family for all three
directions (after R1 is fixed for wrw), hardware-validate correctness on the standard
regression shapes, and benchmark. If additive, fold into `generate_all_configs.py`'s
combinatorial matrix so every future tile family gets both by default.

---

## 6. A stale cross-source claim: `main_loop_interleave` is not a dead end

`docs/gfx1250_perf_parity_action_plan.md` (2026-08-27, predates the D1 validation commits)
states under "What NOT to do":

> CK independently built and then **disabled** an instruction-interleave scheduler for its
> WMMA pipeline ... a second, independent confirmation (alongside **MISA's own measured
> regression from `main_loop_interleave`**) that hand-scheduled interleaving is a genuinely
> hard problem for WMMA specifically ... Don't re-attempt this without a fundamentally
> different scheduling model.

This is now **contradicted by this repo's own later data**: `docs/gfx1250_d1_main_loop_interleave_validation.md` (fwd, +4.5–7.0%) and `docs/gfx1250_d1_bwd_wrw_interleave_validation.md`
(bwd +3.6–4.0%, wrw +8.5%) — all three directions, all `valid:y`, all outside run-to-run
noise bands, landed after the action plan was written. Whatever "MISA's own measured
regression" the action plan is referring to is either a different, unnamed experiment, or
stale information that was never corrected when the D1 docs superseded it. **This should be
fixed in `gfx1250_perf_parity_action_plan.md` directly** (out of scope for this doc to edit,
flagged here so it doesn't keep misleading future readers) — the correct, current state is
"`main_loop_interleave` is a validated small win in all three directions; CK's own disabled
scheduler is architecturally different (full MBB-level instruction interleaving) from this
repo's coarser K-substep-granularity interleave, so CK's experience doesn't actually
transfer as a warning against it."

---

## 7. Guide cross-check (`docs/gfx1250_gemm_optimization_guide.md`) — re-audited against current code

Re-verified every claim from v2's §5 against `HEAD` rather than trusting it was still
accurate:

- **§14 split barriers still have an empty gap.** Confirmed via disassembly of the current
  `fwd_all` build: every `s_barrier_signal -1` is immediately followed by `s_barrier_wait`
  with zero intervening instructions (`objdump` byte offsets 4 bytes apart — the minimum
  possible). Unchanged from v2. The `can_hoist` mechanism validated in `b542b08` (§2, §5)
  hoists the *entire next-tile load phase* to before the compute block, which is a different,
  coarser-grained fix than filling this specific gap — both are legitimate but distinct;
  neither has been tried in combination with the other in the same kernel.
- **§18 temporal hints (`th:`/`scope:`) still unset by codegen.** Re-confirmed: 0 occurrences
  of `th:` anywhere in generated `.s` source; the 192 `scope:` occurrences visible in
  `objdump` disassembly are the disassembler's default `SCOPE_SYS` annotation on unrelated
  scalar kernarg loads, not anything the WMMA generators intentionally emit on the main-loop
  `global_load_dwordx4`s. No codegen change since v2.
- **§19 L2 prefetch (`GLOBAL_PREFETCH_B8`) still unused.** Confirmed: no occurrence in
  `python/igemm/` or `python/operations/`. The only `prefetch`-named field in the codebase
  (`global_prefetch_a/b_num` in `igemm_base.py`) is an unrelated, pre-existing
  VGPR-level double-buffer count for the old XDLOPS pass-through path, not the guide's L2
  cache-line hint.
- **§15/§16 cluster multicast and cluster barriers are entirely unimplemented** — confirmed
  by grep (`multicast`, `clusterDim`, `workgroupMask`, `ClusterDimension`): zero matches
  anywhere in `python/` or `driver/`. Not mentioned as attempted in any backlog doc either.
  This remains the largest unaddressed mechanism from the guide, appropriately deferred (it
  needs a cluster-aware launch path this codebase has no scaffolding for at all) but should
  stay on the radar for the next major push once the P0/P1 items below are closed.
- **§17.1 claused stores are unimplemented** — confirmed by grep (no `s_clause` anywhere in
  the codebase). §17.2/17.3 (LDS-staged / async-to-global stores) are effectively what the
  existing non-atomic epilogue (`coalescing_store_wmma.py`) already does — no new work
  needed there, but claused stores for the atomic (split-K) epilogue's scalar
  `global_atomic_add_f32` sequence remain untried; this is the same code path
  `docs/gfx1250_perf_parity_action_plan.md`'s item 2 (packed 2-wide vector atomics,
  cross-validated against CK and FlyDSL) already targets — the two ideas compose (pack
  first, then clause the packed stores) and neither has been attempted.
- **No `sched_barrier` calls exist anywhere in the codebase** (confirmed by grep). This is
  expected, not a gap: the guide's `sched_barrier(0)` requirement is a *compiler* reordering
  fence for HIP C++ source compiled through LLVM's machine scheduler. This codebase emits
  raw assembly text directly (no compiler backend reordering pass sits between the Python
  emitter and the final `.s`), so there is nothing for `sched_barrier` to fence against —
  the guide's requirement doesn't transfer to this codegen model. Worth stating explicitly
  so nobody spends effort adding no-op fences to already-linear hand-emitted assembly.

---

## 8. Revised priority plan

**P0 — defects (all FIXED in the 2026-09-06 follow-up session; see "Resolution status" near
the top of this document)**

1. ~~**R1**~~ **DONE** (`b367d93`) — moved `threads_per_krow` computation inside
   `if lds_row_pad > 0:` in `igemm_wrw_gtc_wmma_nhwc.py`. wrw's master-config search
   (including item P0.5 below) is now unblocked.
2. ~~**R5 (new)**~~ **DONE, root-caused** (`ed870e6`) — the crash was a missing
   `gemm_k_per_wg` PAL metadata argument declaration at kernarg offset 88 in bwd's
   `get_kernel_args()` (fwd/wrw both declare it; bwd's `kas.append()` call was simply never
   written). Fixed with a one-line addition; `tunable_is_valid`'s rejection removed;
   hardware-validated on every crashing shape plus the full standing regression set.
3. ~~**R6 (new)**~~ **DISPROVEN** (`a3b1476`) — the "hang" was a test-driver host-side
   false-sharing bug in RNG fill / tensor copy inflating setup time past the sweep timeout,
   fixed at its actual root cause (`c56fb4e`). No wrw kernel/driver change was needed; see
   `docs/gfx1250_wrw_full_cu_grid_hang.md`.
4. ~~**R4**~~ **DONE** (`472f274`) — added the missing `!unit_conv` rejection to
   `tunable_is_valid`'s WMMA branch for `wrw_streamk`.
5. ~~**R2**~~ **DONE** (`12d3c0d`) — `wmma_fp16_output=1` config variants added for
   fwd/bwd/wrw fp16, build+dispatch verified.
6. ~~**R3**~~ **DONE** (`175f455`) — `fp_factor` reverted to 8 (physical value) for WMMA.
7. ~~Housekeeping~~ **DONE** — `config/w6_test_*.config` (4 stray files) deleted; the
   `workgroup_swizzle` tunable removed entirely (Python + C++); `wmma_acc_bf16` confirmed
   already excluded from every master `_all.config`'s search path (pre-existing
   accumulate-width-hazard mechanism — no code change needed).

**P0.5 — DONE (this session), mixed result — additivity is direction-dependent, not universal**

8. ~~**Combine `lds_row_pad=16` with `lds_double_buffer=1`**~~ Added `_dbuf_ldsrp.config` for
   all three directions (128×128 and 64×64 tiles), hardware-validated on the standing
   regression set. Result, measured at `n128 c1024 17×17 k1024`, 128×128 tile, pinned
   1100 MHz (this session's machine is clock-capped, see §0 — **these numbers are directional
   only, from a single shape/tile, and MUST be re-measured on the faster reference machine
   before being treated as a real perf sign-off; further perf collection on this machine was
   stopped once this was flagged mid-session**), warmed-up (`IGEMM_WARMUP=5 IGEMM_REPEAT=20`):

   | direction | baseline | `dbuf` alone | `ldsrp` alone | `dbuf`+`ldsrp` | incremental `dbuf` gain over `ldsrp` alone |
   |---|---|---|---|---|---|
   | fwd | 275.2 TFLOP/s | 312.3 (+13.5%) | 404.5 (+47.0%) | 401.9 (+46.0%) | **~0%, noise-level** |
   | wrw | 169.1 TFLOP/s | *(not comparable — wrw's `_dbuf.config` lacks `gemm_k_global_split=1`, an 8× parallelism handicap that dominates its number)* | 280.2 (+65.7%) | 310.5 (+83.7%) | **+10.8%, real** |
   | bwd | — | — | — | **`-nan`, broken** | see R7 below |

   **fwd: not additive** — `lds_row_pad` alone already captures essentially the entire win
   at this shape/tile; stacking `lds_double_buffer` on top adds nothing measurable (within
   run-to-run noise). **wrw: genuinely additive** — `lds_double_buffer` contributes a real
   further +10.8% once `lds_row_pad` is already applied. **bwd: broken, not shippable** —
   new defect **R7** (`docs/gfx1250_bwd_dbuf_ldsrp_nan.md`): the combination produces
   silent `-nan` output on every regression shape/tile tested, despite each mechanism
   individually being hardware-validated correct for bwd — plausibly related to bwd being
   the only direction with asymmetric A/B transpose (A untransposed, B transposed, unlike
   fwd's neither/wrw's both), not yet root-caused. Mitigated by a `tunable_is_valid`
   rejection (mirroring R5's pattern) so no wrong-answer bwd kernel is reachable through
   the master config.

   fwd's and wrw's `_dbuf_ldsrp.config` are shipped, folded into their master
   `_all.config` unions. bwd's is not (excluded by the driver-side rejection).
   **Takeaway for future stacking experiments**: "two independently-good mechanisms compose"
   is not a safe default assumption in this codebase — verify per-direction, not just
   per-mechanism.

**P1 — guide-derived, cheap, unchanged from v2 (still not attempted)**

9. Fill the split-barrier signal→wait gap (guide §14) with the next tile's hoisted load
   issue — a finer-grained version of what `can_hoist` already does at block granularity;
   evaluate whether it's additive with `can_hoist` or redundant.
10. Temporal hints (`RT_NT` on A/B cooperative loads, `NT` on C stores, guide §18) — still a
    one-line-per-emitter change, still unbenchmarked.
11. gfx1250 shader prologue / `S_CODE_END` padding, then `GLOBAL_PREFETCH_B8` two K-stages
    ahead (guide §19) for compute-bound 1×1 shapes.
12. Packed 2-wide vector atomics for wrw's split-K epilogue
    (`docs/gfx1250_perf_parity_action_plan.md` item 2, cross-validated against CK and
    FlyDSL) — compose with claused stores (guide §17.1) once the atomic op itself is packed.

**P2 — larger mechanisms, unchanged from v2/backlog**

13. `async_global_load` broadened beyond its current 8-config footprint.
14. TDM beyond `nxe==0`/unit-stride; hardware LDS padding (`D#.pad_amount`) to subsume
    `lds_row_pad` at zero instruction cost (guide §13) — now more valuable given §5's
    finding that padding is the single largest lever measured.
15. Cluster multicast + cluster barriers (guide §15/§16) — needs new launch-path
    scaffolding; highest structural cost, deferred appropriately.
16. wrw addressing redesign to support `gemm_k_per_block > gemm_m_per_block`
    (`docs/gfx1250_optimization_backlog.md`, Tier 2) — unlocks CK-parity tile shapes for
    small-output-channel wrw, the repo's worst-measured direction per
    `docs/gfx1250_vendor_benchmark_vs_miopen.md`.

**P3 — cross-source items already tracked, not duplicated here**

`docs/gfx1250_optimization_backlog.md` (805 lines, actively maintained, items added/removed
only on actual implementation) and `docs/gfx1250_perf_parity_action_plan.md` (modulo §6's
correction) remain the source of truth for the exhaustive wrw-vs-MIOpen gap-closing list —
`s_setprio` bracketing (already implemented, `wmma_setprio`, 3 configs), split-K strategy
alternatives (CK closed-form occupancy formula, rocKE persistent stream-K, hipconv
separate-reduction-kernel), `disable_xdl_arb_stall` (blocked on missing hwreg ID, correctly
not guessed), and the register-budget overflow silently narrowing
`build_and_filter_configs.py`'s combinatorial search. Nothing in this session's review
changes those items' status; consult the backlog directly rather than this report for their
current state.

---

## 9. Validation protocol updates for the next round

Unchanged from v2's core recommendation (main-loop ablation harness, `script/ablate_main_loop.py`,
still not landed) plus, learned this session:

- **Every benchmark run must record the actual clock** (`rocm-smi --showclocks` or
  `IGEMM_SCLK_MHZ`) alongside every number. §0/§5 show the same mechanism's relative benefit
  can differ by 3-4× across machines — a number without its clock is not reproducible
  evidence.
- **Every dispatch in an automated sweep must be wrapped in a timeout.** R6 is an
  indefinite hang, not a slow kernel; an un-timeout-guarded `build_and_filter_configs.py`-style
  sweep across a master config's full candidate list will stall forever the first time it
  hits a full-CU-grid, non-split-K wrw shape.
- **A `valid:n` or crash must not be treated as merely "excluded from the search."** R4 and
  R5 both print numerically impossible ">100% efficiency" for their broken results — R3
  means the tool cannot self-flag this, so any script consuming `conv_driver.exe`'s output
  must explicitly gate on `valid:y` AND a sane efficiency bound before trusting a "fastest"
  result.
- After any observed crash or hang, **confirm device health with an unrelated known-good
  kernel before trusting subsequent measurements** (this session: bwd's memory fault and
  wrw's hang were each followed by a fwd sanity run before continuing) — cheap insurance
  against silently attributing a wedged device's slow numbers to the wrong kernel.

Standing regression set (unchanged from v2):
```
-n 256 -c 2048 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0   # large 1x1, feed-limited; wrw's full-CU-grid R6 shape
-n 128 -c 1024 -H 17 -W 17 -k 1024 -y 1 -x 1 -p 0 -q 0   # medium 1x1; bwd's R5 crash shape
-n 64  -c 512  -H 28 -W 28 -k 512  -y 3 -x 3 -p 1 -q 1   # 3x3, compute-limited
-n 32  -c 256  -H 56 -W 56 -k 256  -y 3 -x 3 -p 1 -q 1   # 3x3, large spatial
-n 128 -c 64   -H 56 -W 56 -k 64   -y 1 -x 1 -p 0 -q 0   # small-channel / tail-heavy
```
Two process rules from v2, still valid and still not automated:
- **Every commit that adds a tunable must also add a config that sets it**, or state
  explicitly that it is deferred.
- **`script/build_and_filter_configs.py` over all `config/*.config` must pass before
  merge.** R1 would have been caught immediately, twice now (v2 and this revision).
