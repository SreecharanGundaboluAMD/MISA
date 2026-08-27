# FlyDSL deep dive: pipeline depth, tile selection, small-M/huge-K, scheduling

Follow-up investigation of `/home/sgundabo/FlyDSL`, going deeper than the first pass. Read
`docs/gfx1250_external_research_findings.md`'s "FlyDSL findings" section first -- this file
does not repeat anything from there (cluster multicast via TDM, narrow-accumulate WMMA being
unused, epilogue TDM-store/direct-store, K-tail being unsolved except host-side padding,
split-phase barriers, `disable_xdl_arb_stall`). Everything below is new.

Scope note: FlyDSL's WMMA-family compiler stack (MLIR `FlyROCDL` dialect, `rocdl.wmma_*`
intrinsics, wave32 execution model) spans gfx1103/gfx11 (RDNA3), gfx120x (RDNA4 desktop),
and gfx1250 ("CDNA5"/RDNA4-class, same as MISA's target). A few of the findings below come
from gfx11/gfx120x kernels rather than gfx1250 kernels specifically -- these are called out
explicitly, since they are not directly gfx1250-confirmed, but they exercise the same
wave32/WMMA instruction family MISA targets and are structurally portable. The gfx1250-only
findings (main-loop pipeline depth, MoE split-K, non-power-of-two tile widths, instruction
scheduling groups) come straight from files under `kernels/gemm/*_gfx1250.py` and
`kernels/moe_gemm_2stage_*_gfx1250.py`, most of which live on unmerged branches
(`gfx1250_moe_splitK_new`, branched from `main` at a point after the code the first pass
saw) rather than FlyDSL's `main` branch.

## 1. Main-loop pipeline depth: N-stage software pipelining, not fixed at 2

`kernels/gemm/gemm_a8w8_gfx1250.py` and `kernels/gemm/gemm_a8w4_mxscale_gfx1250.py` (branch
`gfx1250_moe_splitK_new`) take `num_buffers: Constexpr[int]` as a genuine compile-time
kernel parameter, not a hardcoded 2. It controls three things simultaneously:
- LDS arena size: `ARENA_B = max(num_buffers * PITCH, C_STORE_B)`, gated by
  `check_smem_capacity` (so a shape only gets a deep pipeline if it fits in LDS).
- The TDM prologue/steady-state/drain loop shape (lines ~467-494 of
  `gemm_a8w8_gfx1250.py`):
  ```python
  for i in range_constexpr(num_buffers - 1):
      issue(i, i)                                    # fill num_buffers-1 stages
  n_steady = K_TILES - (num_buffers - 1)
  for kt in range(n_steady):                          # steady state
      pipeline_fence(outstanding=(num_buffers - 2))    # s_wait_tensorcnt(N-2), not 0
      compute_ktile(buf, pbuf, kt + (num_buffers - 1))  # compute one, issue one more
  for j in range_constexpr(num_buffers - 1):           # drain
      pipeline_fence(outstanding=(num_buffers - 2 - j))
      compute_ktile(buf, pbuf, None)                    # compute only, issue nothing
  ```
  This is a real N-deep software pipeline: `num_buffers - 1` TDM loads are always in flight
  during steady state, and `pipeline_fence`'s `outstanding` argument threads a
  `s_wait_tensorcnt(N)` (N decreasing only in the drain tail) through
  `kernels/gemm/gemm_common_gfx1250.py`'s `pipeline_fence`/`tdm_ops.tensor_wait`.
- Tested/shipped depths: `tests/kernels/test_gemm_fp8fp4_gfx1250.py`'s parametrized cases use
  `num_buffers` = 2, 3, and 4 for different (M, N, K, tile) combinations on gfx1250 (e.g. a
  `256x256x512` tile-256 case uses `num_buffers=4`; a `128x128x1024` tile-128 case with more
  K-tiles available uses `num_buffers=3`). This is chosen per-shape by hand in the test
  matrix, not auto-searched, but it proves depths beyond simple double-buffering are real,
  compile, and pass correctness checks on gfx1250 hardware.

**Why this helps**: MISA's LDS double-buffering is fixed at depth 2 (one buffer being
consumed, one being filled). A 3-4 stage pipeline hides more of the TDM/global-load latency
behind compute when a K-tile's load takes longer than one K-tile's worth of WMMA + LDS-read
work to issue, which becomes more likely as MISA's `gemm_k_per_block` shrinks (32) relative
to TDM's larger transfer granularity, or when occupancy is low (fewer waves running to fill
the gap otherwise).

**Difference from MISA**: MISA's main loop (`python/igemm/igemm_*_gtc_wmma_nhwc.py`) has a
single double-buffer depth, plus a separate, unrelated VGPR-level "local prefetch" (2-deep,
fp32-only) within a K-substep -- it does not have a parametrized N-stage LDS pipeline whose
depth is chosen by LDS-capacity headroom, the way `num_buffers`/`ARENA_B` do here.

## 2. Tile-size selection: real autotuning exists (arch-agnostic), and non-power-of-two N tiles are used on gfx1250

Two distinct mechanisms found, neither reported before:

- **A genuine on-device brute-force autotuner**, `kernels/conv/conv3d_autotune.py`
  (`autotune_conv3d`): for a given (shape, dtype) key it compiles and benchmarks every
  candidate in a curated `(TILE_M, TILE_N, WAVE_M, WAVE_N)` list (plus an optional
  `WGM_VALUES = [1, 4, 8]` workgroup-swizzle sweep) via `flydsl.autotune.do_bench`, keeps the
  fastest, and caches the winner to `~/.flydsl/autotune/conv3d_*.json` keyed by
  `(kind, shape, dtype, device_fingerprint, toolchain_fingerprint, schema_version,
  candidates)`. Illegal/spilling configs are skipped with a bare `try/except` around the
  compile+bench call. This is architecture-generic (keyed by whatever `get_rocm_arch()`
  returns), so it is not gfx1250-exclusive, but nothing prevents it from running there.
  This is a materially different mechanism from MISA's static, hand-curated `.config` files
  -- it is a real search-and-cache loop that runs at the first call for a new shape, not a
  human-picked table.
- **For the actual gfx1250 GEMM kernels** (`gemm_a8w8_gfx1250.py`,
  `gemm_a8w4_mxscale_gfx1250.py`), tile shape is NOT auto-searched -- `tile_m/tile_n/tile_k/
  m_warp/n_warp/num_buffers/cluster_m/cluster_n` are all explicit `Constexpr` kernel
  parameters, hand-picked per test shape in `tests/kernels/test_gemm_fp8fp4_gfx1250.py`'s
  parametrize lists. But that same test file's `_PTPC_CASES` includes
  `(M=128, N=96, K=512, tile_m=128, tile_n=96, ...)` -- **a tile_n=96 tile**, which is neither
  a power of two nor 128-aligned. The kernel's own `ALIGNED_N = tile_n % 128 == 0 or
  128 % tile_n == 0` guard (gfx1250 `gemm_a8w8_gfx1250.py` line ~68) explicitly branches to a
  more general `N_BLOCKS` computation (`math.lcm(tile_n, 128)`-based) specifically to support
  non-128-aligned N widths, i.e. arbitrary tile_n is a designed-for case, not an accident.
  MISA's WMMA tile enumeration is currently 128x128, 64x64, 128x64, 64x128 -- all powers of
  two and all multiples of the 128-lane alignment MISA otherwise assumes. tile_n=96 (6x
  WMMA_N=16) is a genuinely different shape family MISA hasn't tried.
- Also notable: gfx1250 K-tiles here are 128 or 256 (`tile_k`), 4-8x larger than MISA's
  `gemm_k_per_block` of 32 (64 in the k2x variant) -- enabled by TDM's bulk-transfer
  granularity and the deeper `num_buffers` pipeline above, meaning fewer, larger LDS
  round-trips per K reduction rather than many small ones.

**Why this helps**: a real autotuner amortizes away the need to hand-derive the best tile
per shape (relevant to MISA's biggest gap -- wrw's failure mode is precisely "shapes MISA's
static config heuristics don't fit well"). The tile_n=96 case shows the compiler and its
address-generation math were deliberately built to not assume power-of-two/128-aligned tiles
-- worth knowing MISA's current constraint (multiples of 128 or exact powers of two) is a
self-imposed simplification, not a hardware requirement.

## 3. Small-M/huge-K (wrw-shaped) strategies -- the highest-value section

Three separate, concrete mechanisms, escalating in how directly they transfer to MISA's wrw.

### 3a. A real gfx1250 split-K GEMM with packed-vector atomics (most directly relevant)

Commit `7213c282` ("Add split-K support for MoE stage1 MXScale GEMM on gfx1250", branch
`gfx1250_moe_splitK_new`) adds `k_batch` (grid.z K-splitting, structurally identical to
MISA's `gemm_k_global_split`) to the MoE stage-1 gate/up GEMM -- whose shape (few tokens per
expert at decode time, `model_dim=7168` reduced against a small per-expert output) is the
same "small M, huge K" family as MISA's wrw. Concrete design, in
`kernels/moe_gemm_2stage_common_gfx1250.py`'s new
`_emit_stage1_gate_up_splitk_epilogue`:

- **Packed 2-wide atomic add**, not scalar-per-element:
  `rocdl.raw_ptr_buffer_atomic_fadd(frag_g, out_rsrc, byte_off_g, ...)` where `frag_g` is a
  `vector<2 x f16>` built from two adjacent output columns (`idx_g_even = idx_g0 &
  0xFFFFFFFE` aligns to the even column first). This issues **half as many atomic RMW ops**
  as one-atomic-per-element.
  - **MISA comparison, checked directly**: MISA's wrw atomic epilogue
    (`python/operations/coalescing_store_wmma.py` line ~206) emits
    `global_atomic_add_f32 v[{cur}], v[{v_c}+{c_index}], s[...]` -- one scalar f32 atomic
    per accumulator element, always f32. MISA has no packed/vector atomic path today.
- **Host pre-zeroes the output buffer** (`out_kernel.zero_()` in the test harness) rather
  than the kernel doing an in-kernel zero-then-flag handshake. Simpler than the
  gfx942/gfx950 `hgemm_splitk.py` sibling kernel (see 3b), which *does* zero and
  synchronize in-kernel via a polling flag -- worth knowing both patterns exist in the same
  codebase family, i.e. FlyDSL doesn't consider in-kernel zero+flag mandatory when the
  caller can pre-zero cheaply.
- **Split-K forces other fast paths off**: the commit message and code explicitly disable
  TDM-store, wave-specialized TDM, and the fused `doweight_stage1` epilogue whenever
  `k_batch > 1`, and push the final silu*up reduction out to a separate host-side (not even
  device-side) pass. This confirms TDM-store and atomic-accumulate are treated as mutually
  exclusive in FlyDSL's own design (TDM store is a plain overwrite, not an accumulate), and
  that fusing an activation into a split-K epilogue was judged not worth the complexity here.
- **Empirically calibrated, and the calibration result itself is a useful data point**: the
  commit message reports testing `k_batch` in `{1, 2, 7, 14}` on a real DeepSeek-shaped MoE
  (`model_dim=7168`), and picking **k_batch=2 as best, for only ~1.11x over baseline** --
  i.e. on real gfx1250 hardware, this small-M/huge-K split-K GEMM saw sharply diminishing
  (near-zero) returns past a very small split factor, not a large win from aggressive
  splitting. This is a useful outside data point for MISA's own wrw K-split ternary search:
  a modest split count being the empirical optimum is plausible and consistent with
  MISA's own findings, not an outlier.

### 3b. Wavefront-level pre-reduction before the atomic (gfx950, not gfx1250 -- transferable idea only)

An old, abandoned decode-GEMM prototype, `kernels/gemm_decode.py` (commit `4f37916a`, branch
`GEMM_small_M_SILOTIGER-669`, explicitly gfx950/MI355X, titled "2 times slower than Sami's CK
C++ kernel" -- i.e. FlyDSL's own author considered this attempt a failure, so treat the
overall kernel as a cautionary example, not a template). Still, one isolated technique in it
is worth lifting out on its own: before the single atomic add per output element, each wave
does a **butterfly reduction across its own lanes** via `ds_bpermute` (`wavefront_reduce_sum_f32`,
6 `xor`-shuffle stages) so that only one atomic op is issued per wave per output element
instead of one per lane. This only applies where multiple lanes in a wave compute partial
sums of the *same* output element (a "warp-per-scalar" layout) -- for a standard WMMA tile
where each lane already owns a distinct accumulator element this reduction is unnecessary,
but if any future MISA wrw variant used a reduction-heavy per-lane accumulation layout for
very small N, this is the applicable technique. Not gfx1250-confirmed; noted for completeness
per the task's request to look at small-M/huge-K strategies broadly.

### 3c. Non-square, sub-64 tile_m dispatch for small-M shapes (gfx950/942, not gfx1250)

The same `hgemm_splitk.py` file (also on branch `gfx1250_moe_splitK_new`, but its own test
gate `test_hgemm_splitk.py` restricts it to `ARCH in ["gfx950", "gfx942"]`) has a hardcoded
shape-dispatch table (`get_default_kwargs`) that drops `TILE_M` as low as **16** (equal to
`WMMA_M`, the smallest possible non-degenerate M tile) paired with `SPLIT_K=8` for specific
small-M/huge-K shapes, and exposes a `selections` search space of
`TILE_M in [16, 32, 48, 64, 96, 128]`. This shows the tile-size floor these kernels are
willing to go to for small-M problems is well below MISA's current 64x64 floor -- again not
gfx1250-confirmed, but a data point that a WMMA-based (wave32) compiler stack finds tile_m=16
to be a legal, useful configuration for small-M problems, worth MISA considering for wrw's
worst (small, non-exact) shapes specifically.

### 3d. Persistent kernel with per-workgroup K-phase rotation (RDNA3, not gfx1250 -- but same WMMA/wave32 family)

`kernels/gemm/rdna3_f16_gemm.py` (gfx11/RDNA3 only -- raises at compile time on anything
else) supports `persistent_wgs`: either 0 (plain one-workgroup-per-tile grid) or exactly
`num_tiles` (**whole-tile persistent**: launch a fixed, occupancy-sized number of
workgroups, and have each one loop, in-kernel, over its statically-assigned contiguous shard
of `[t_first, t_last]` output tiles):
```python
t_first = pid32 * num_tiles // persist_wgs
t_last  = (pid32 + 1) * num_tiles // persist_wgs - 1
for t, _carry in range(t_first, t_last + 1, 1, init=[...]):
    ...
    if const_expr(persist_rot_step):
        rot = pid32 * persist_rot_step % fx.Int32(num_k_tiles)   # per-WG K-phase stagger
    accs = _accumulate(pA_g, pB_g, rot)
```
Two things bundled together here: (1) a classic persistent-kernel work list, decoupling
grid size from `num_tiles` entirely, and (2) `persist_rot_step`, which staggers *where in
the K loop* each persistent workgroup starts (`rot`), so workgroups launched at the same
instant don't all issue their first global load to the same K-offset simultaneously --
explicitly commented as reducing synchronized memory-controller bursts at kernel start.

**Why this matters for wrw specifically**: this doesn't solve "M x N too small" by itself
(the loop is over M/N tiles, and wrw's problem is that GEMM_M/GEMM_N are already tiny), but
the *pattern* -- decouple launched-workgroup count from problem-shape-derived tile count,
and stagger each persistent workgroup's starting phase -- maps directly onto MISA's
`gemm_k_global_split`: today MISA's K-split workgroups are grid.z-indexed and (presumably)
all begin their K sub-range's global load at the same wall-clock instant. Applying
`persist_rot_step`'s idea to MISA's K-split axis (stagger each split-shard workgroup's first
load by a small per-shard phase offset) is a plausible, low-risk knob to test for reducing
memory-controller contention specifically in the K-split case, independent of adopting
"persistent" launch semantics at all. Not gfx1250-confirmed in this codebase (gfx120x, the
RDNA4 sibling file, does not carry this feature over), but it is exercised on the same
wave32/WMMA instruction family MISA uses, unlike the CDNA/MFMA-only findings above.

## 4. Instruction scheduling: explicit ds_read/WMMA issue-ratio scheduling groups, on gfx1250 itself

Beyond `disable_xdl_arb_stall` (already reported), the real gfx1250 kernels
(`gemm_a8w8_gfx1250.py`, `gemm_a8w4_mxscale_gfx1250.py`) emit **explicit instruction
scheduling-group hints** around every K-substep, via `rocdl.sched_barrier(0)` (schedule-group
boundary), `rocdl.sched_dsrd(N)` ("schedule N ds_read ops next"), and `rocdl.sched_mfma(N)`
("schedule N matrix ops next") -- these lower to AMDGPU's `sched_group_barrier`-style
intrinsics that force the backend's scheduler to interleave instructions in a specific,
hand-picked ratio rather than leaving it to the default list scheduler. Concretely
(`_kstep`/`compute_ktile_row`, lines ~284-345):
- The M-dimension accumulator rows are split into a "front half" and "back half"
  (`front_wm = (wmma_m_rep + 1) // 2`).
- The front half's operand loads are issued, then only the **back half's** count is waited
  on (`rocdl.s_wait_dscnt(len(_BACK) * DS_A)`) before starting the front half's WMMAs --
  i.e. front-half compute overlaps back-half loads still in flight, a register-level
  software pipeline *within* a single K-substep (distinct from, and layered on top of, the
  cross-K-tile `num_buffers` pipeline in section 1).
- After all substeps, a second pass re-emits the same ds_read/WMMA sequence wrapped in
  `sched_dsrd(N)`/`sched_mfma(N)` pairs with explicit counts
  (`_BS_DS if _ks == 0 else 0) + front_wm * DS_A`, etc.) -- this is a second, deliberate
  scheduling pass that only sets instruction-group boundaries, it does not change what's
  computed.

**Why this helps**: RDNA/CDNA backends' default scheduler doesn't always find the ideal
ds_read/matrix-op interleave ratio on its own, especially across a hand-unrolled K loop with
async loads in flight; explicit `sched_group_barrier`-style counts pin the interleave
mid-K-substep as well as across substeps.

**Difference from MISA**: MISA's "interleaved chunk/compute schedule" (mentioned in the
background as a measured regression, kept only per instruction) is MISA's closest analog,
but it's a hand-written instruction interleaving in the emitted `.s` text, not a
compiler-scheduling-group hint layered on top of an otherwise-default schedule. FlyDSL's
version is narrower in scope (it only pins ds_read-vs-WMMA issue counts, not a full
instruction reorder) which may be why it doesn't carry the same regression risk MISA found
-- worth treating as a structurally different (smaller-blast-radius) kind of scheduling
control than what MISA already tried and rejected, not a retry of the same idea.

## Recommendations for MISA, prioritized

1. **(High value, low-to-medium risk) Pack wrw's atomic accumulate into 2-wide vector
   atomics.** MISA's `python/operations/coalescing_store_wmma.py` (~line 206) emits one
   scalar `global_atomic_add_f32` per accumulator element. FlyDSL's gfx1250 MoE split-K
   epilogue (section 3a) instead builds a 2-element vector from two adjacent output columns
   and issues one packed atomic (`raw_ptr_buffer_atomic_fadd` on a `vector<2xf16>`),
   halving atomic op count. MISA's wrw accumulates in f32 today (per the same file's
   comments); the direct analog would need either (a) a packed 2xf32 atomic if gfx1250's
   ISA supports one (verify against the ISA doc/probe, the same way TDM was verified), or
   (b) accumulating the K-split partials in fp16/bf16 (packed 2-wide atomic, hardware-common)
   with a final fp32 promotion pass, trading some intermediate precision for half the atomic
   traffic -- needs an accuracy check given wrw's already-tight numerics. Effort: small
   (isolated to the atomic-emit line + a dtype/precision decision); risk: correctness
   (precision, ISA support) needs verification before landing, same rigor as TDM's hardware
   probe.

2. **(High value, low risk) Test a staggered per-shard K-loop start phase for
   `gemm_k_global_split`.** Section 3d's `persist_rot_step` idea: each grid.z K-split
   workgroup currently likely starts loading its K-sub-range at the same instant as every
   other shard for the same output tile. Add a small, cheap per-shard phase offset (e.g.
   rotate the first-iteration K-tile index by `shard_id * step`) to reduce
   simultaneous-burst memory contention at kernel launch, specifically for wrw's K-split
   path. Effort: small (touches only the K-split loop's initial offset computation, in
   whichever wrw `.py` file owns `gemm_k_global_split`'s main loop -- likely
   `python/igemm/igemm_wrw_gtc_wmma_nhwc.py`); risk: low, purely a scheduling/ordering
   change with no correctness implications, easy to A/B against the existing ternary search.

3. **(Medium value, medium risk) Try tile_m below 64 (down to 32 or 16) for wrw's worst
   (small, non-exact) shapes.** Sections 2 and 3c both show WMMA-based compilers in this
   family treating tile_m=16/32/48/96 as legal, deliberately-supported shapes, not just
   powers of two >=64. MISA's tail-handling cost (EXEC masks) is presumably worse the
   further GEMM_M/GEMM_N are from the tile size; a smaller floor tile might reduce the tail
   fraction for wrw's smallest shapes even if occupancy per-workgroup drops. Effort: medium
   (MISA's tile-shape enumeration and code paths assume today's specific set --
   `wmma_m_per_wave`/`gemm_m_per_block` plumbing throughout
   `python/igemm/igemm_wrw_gtc_wmma_nhwc.py` and `driver/igemm_wrw_gtc_driver.h`'s config
   validation would need a new floor value added and swept); risk: medium, since it's a new
   tile shape family, not a tuning knob on an existing one.

4. **(Medium value, low risk) Deepen the main-loop pipeline beyond double-buffering, gated
   by LDS headroom.** Section 1's `num_buffers` pattern (N-stage pipeline with a
   prologue/steady/drain loop shape, `s_wait_tensorcnt(num_buffers-2)` in steady state) is a
   direct, mechanical generalization of what MISA's double-buffer already does; parametrize
   MISA's LDS buffer count similarly (2, 3, ...) wherever `check_smem_capacity`-equivalent
   headroom exists, and sweep it the same way FlyDSL's test matrix does (per-shape, not
   auto-searched). Most valuable for K-heavy shapes (which includes wrw) where TDM/global-load
   latency is more likely to exceed one K-tile's compute time. Effort: medium (touches
   MISA's main-loop codegen and LDS allocation sizing in whichever file emits the double-
   buffer today); risk: low-to-medium, since it changes the main loop's control flow shape,
   needs re-validation against MISA's existing regression benchmarks.

5. **(Lower value, low risk, longer-term) Consider a real autotuning-with-cache mechanism
   for MISA's `.config` selection**, modeled on `conv3d_autotune.py`'s
   compile-benchmark-cache loop (keyed by device+toolchain+shape fingerprint, JSON-cached).
   This doesn't have to replace MISA's existing config files; it could supplement them for
   exactly the shapes flagged as "worst cases" in the background (wrw's small/non-exact
   shapes) where hand-curated configs are demonstrably failing. Effort: larger (new
   infrastructure, not a kernel change); risk: low technically, but needs product/workflow
   buy-in since it changes how configs are produced, not just what they contain -- lowest
   priority of the five, included for completeness since it directly targets "tile-size
   selection for hard shapes," the same problem MISA is worst at today.
