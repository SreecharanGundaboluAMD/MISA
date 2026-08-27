# rocKE deep dive: performance techniques transferable to MISA's gfx1250 WMMA convolution kernels

Source repo investigated: `/home/sgundabo/rocm-libraries/dnn-providers/hip-kernel-provider/rocke`
(read-only; nothing in that repo or in MISA's code was modified).

## What rocKE is

rocKE is a Python-embedded kernel-authoring DSL/compiler stack (sibling name in its own
docs: "CK DSL"). A kernel is written as a Python `spec` dataclass, a `build_*()` function
turns it into a typed SSA `KernelDef`, and a lowering engine (`core/lower_llvm.py`, or an
equivalent C++20 engine that must emit byte-identical LLVM IR) turns that into AMDGPU LLVM
IR, which `libamd_comgr` compiles in-process to HSACO and HIP launches — no template
metaprogramming, no external `.s` emission; MISA's raw-assembly codegen is a fundamentally
different implementation strategy for a conceptually similar goal (implicit-GEMM
GEMM/attention/conv kernels tuned per target). rocKE's primary target family today is CDNA
(gfx942/gfx950, MFMA); gfx1250 (WMMA) support exists (`Gfx1250Backend`, WMMA lane maps,
a couple of attention and toy-GEMM instances) but is newer and thinner — its own
`known_gaps.md` states gfx1250 has no C++ instance-builder family yet, and there is no
`dsl_docs/optimization/arch/gfx1250.md` (only `gfx942.md`/`gfx950.md` exist). Most of the
concrete, quantified optimization playbook below is therefore CDNA/MFMA-proven, not
gfx1250-proven — but nearly all of it is architecture-transferable technique, and a few
findings (marked below) are explicitly gfx1250-specific.

## 1. GEMM optimization playbook (`gemm-optimization-rocke.md`)

rocKE's own skill doc prescribes a fixed lever order, each with a quantified expected
gain, validated on a 4096³ fp16 GEMM (gfx942 → gfx950):

1. `pipeline='basic_v1'` (single-stage) baseline: ~350 TFLOPS (53% of peak).
2. Add 2-stage double-buffer (`pipeline='compv4'`): **+20%** (~420 TFLOPS).
3. Add async global→LDS copy (`use_async_copy=True`, bypasses VGPR staging): **+10%**
   (~460 TFLOPS) — MISA already does this (`global_load_async_to_lds_b128`).
4. Move to the "abundant LDS" target and switch bank-conflict mitigation from XOR-swizzle
   to padding (`lds_k_pad=8`) plus a cshuffle epilogue: **+13%** (~520 TFLOPS, 80% of peak).

The doc's own bottleneck table (section 11) is a reusable checklist MISA's profiling work
could lift directly: high `vmcnt` stalls → add prefetch/async copy; high `lgkmcnt` stalls
→ check LDS bank conflicts; low MFMA/WMMA-instruction fraction (<40% of total
instructions) relative to address/pack/mask instructions → compute-bound assumption is
wrong, fix coordinate arithmetic first.

Does the actual code follow the doc? Only partially for GEMM proper: `gemm_universal.py`
implements `pipeline ∈ {mem, compv3, compv4}` and `epilogue ∈ {default, cshuffle}` as
documented, and `helpers/schedule.py`'s `SchedulePolicy.for_pipeline` maps each pipeline
name to a concrete `sched_group_barrier`/`s_setprio` hint recipe (see §3/§6 below). The
`lds_k_pad` "padding vs. XOR" decision from the LDS skill (§2) is not wired as a GEMM-level
knob in the code paths inspected — it appears to live as documented guidance more than a
shipped, arch-selectable spec field for gfx1250.

## 2. LDS optimization playbook (`lds-optimization-rocke.md`)

The padding-vs-XOR decision is architecture-conditioned on LDS bank count, not just LDS
capacity, and this is the one part of the whole investigation with the most quantitative
detail:

- gfx942: 32 banks, 4 bytes/bank, 64KB LDS, stride=128B → **full** conflict (all threads
  hit the same bank).
- gfx950: 64 banks, 4 bytes/bank, 160KB LDS, stride=128B → only **2-way** conflict (banks
  double, so the same stride pattern that fully conflicts on gfx942 only half-conflicts on
  gfx950); stride=256B is needed to fully conflict on gfx950.
- Decision rule: `LDS_total < 96KB → XOR swizzle` (capacity-constrained); `LDS_avail ≥
  128KB → padding swizzle` (simpler addressing, 1-2 ALU ops vs 5-7 for XOR); measured
  gain on gfx950 ResNet50 conv3_1: **XOR 494 TFLOPS → padding 705 TFLOPS (+43%)**, because
  the simpler address math schedules better even though it wastes ~2% of LDS.

MISA's comparison point: MISA already uses a simple LDS padding scheme (mentioned in the
task background as its epilogue LDS padding for wrw). This finding suggests the choice of
*which* mitigation (padding vs. XOR-style swizzle) should be a first-class, per-arch/per-
LDS-budget decision rather than a single fixed scheme — worth explicitly checking whether
MISA's current fwd/bwd K-tile LDS layout (K=32 or 64, fp16/bf16, 2-byte elements) sits at a
stride that actually conflicts on gfx1250's bank count, since gfx1250's bank width/count is
not established anywhere in rocKE's own arch tables (RDNA/gfx12 LDS parameters are absent
from `lds-optimization-rocke.md`, which only documents gfx942/gfx950). This is a "verify on
hardware first" item, not a ready-made fix.

## 3. Prefetch strategy (`prefetch-data-load-rocke.md`)

The doc's rule of thumb: `stages_needed = ceil(memory_latency_cycles /
compute_duration_cycles)`, empirically capped at 4 stages (diminishing returns beyond
that, VGPR pressure grows linearly with stage count: `vgpr_prefetch = vgpr_per_tile *
prefetch_stages`). The worked "grouped conv" example stacks four independent levers and
reports cumulative, measured-style gains: `basic_v1` baseline 350 TFLOPS (60% MFMA
utilization, memory-bound) → `compv4` 2-stage prefetch +20% (420 TFLOPS) but occupancy
*drops* to 1 wave/SIMD → `use_async_copy=True` recovers occupancy to 2 waves/SIMD *and*
adds gain, +31% cumulative (460 TFLOPS) → LDS double-buffer +44% cumulative (505 TFLOPS)
→ padding-swizzle + cshuffle epilogue +48% cumulative (520 TFLOPS). The key lesson worth
importing as methodology: **each lever's gain was re-validated against occupancy, not just
runtime** — a lever that improves latency-hiding but silently drops waves/SIMD is not
free, and the doc explicitly calls this out as the reason async-copy was added *after*
compv4, not instead of it.

Compared to MISA's prefetch: MISA's local-prefetch is a **register-level** (VGPR
ping-pong) 2-deep prefetch within the K-substep loop, fp32-only today due to VGPR budget,
plus separate LDS double-buffering. rocKE's model treats "prefetch stages" and "LDS
double-buffer" as two independently tunable dials (`prefetch_stages` for how many future
global-load issues are in flight; `double_smem_buffer` for whether LDS itself is
ping-ponged) rather than coupling them — worth checking whether MISA's fp16/bf16 WMMA path
could get a *partial* register prefetch (1 extra in-flight global read, not a full
ping-pong) at lower VGPR cost than the full 2-deep scheme that's currently fp32-restricted,
since rocKE's data shows even a small amount of prefetch depth captures most of the gain
when compute duration is short (`stages_needed` rounds to 1-2 for the shapes in its
examples).

## 4. Stream-K / persistent-kernel mechanisms — the highest-value finding

rocKE ships two composable, tested pieces (`platform/python/rocke/helpers/persistent.py`,
`platform/python/rocke/helpers/streamk.py`, plus a demo consumer,
`platform/python/rocke/instances/common/streamk_gemm.py`), both **CK-Tile-parity ports**,
i.e. modeled directly on CK Tile's C++ `StreamKTilePartitioner` and persistent-DP
dispatcher. Important caveat up front: **neither is wired into rocKE's convolution wgrad
builder** (`conv_implicit_gemm_wgrad.py`) — wgrad instead uses a much simpler static
grid.z split-K (see below), the same shape of mechanism MISA already has. Stream-K/
persistent is only demonstrated end-to-end on a plain square GEMM. So this is a transplant
of a *generic, working mechanism*, not a "rocKE already solved your wrw problem" result.

**Persistent-kernel helper** (`helpers/persistent.py`):
- Pattern: launch a small, constant grid (~`num_cus * blocks_per_cu`, not sized to problem
  size). Each CTA does `tile_idx = atomic_add(Counter, 1)` to fetch its first tile, then
  loops `persistent_tile_for_each(counter, num_tiles, max_iters, body)`, refetching a new
  tile index after each iteration until `tile_idx >= num_tiles`. This directly decouples
  grid size from the (M,N,K)-derived tile count — exactly the property MISA's wrw problem
  needs, since wrw's GEMM_M×GEMM_N is tiny (poor natural occupancy) while GEMM_K is huge.
- A real, documented hardware bug fix is baked into the broadcast path (worth noting as a
  correctness landmine if MISA ever implements something similar): the original
  LDS-broadcast-plus-`s_barrier` design for single-wave CTAs let the AMDGPU optimizer elide
  the barrier, causing ~1.7% of in-range iterations to silently skip processing. The fix
  replaces it with `ds_bpermute(addr=0, val)` for single-wave (≤64-lane, i.e. ≤1 wave)
  CTAs — a wave-internal cross-lane broadcast that needs no barrier at all because SIMD
  pipeline ordering guarantees correctness. Multi-wave CTAs still use the LDS+barrier path
  (there the barrier is a *real* cross-wave sync so the optimizer cannot elide it).
- `compute_streamk_grid_size(spec, num_cus=304, blocks_per_cu=1)` = `min(num_macro_tiles,
  num_cus*blocks_per_cu)` — the grid-sizing formula, trivial but exactly what a persistent
  wrw kernel would need instead of MISA's current ternary search over a small set of
  discrete split_k values.

**Stream-K partitioner** (`helpers/streamk.py`):
- `StreamKPartition(m_tiles, n_tiles, k_iters)` with `num_macro_tiles = m_tiles * n_tiles *
  k_iters`. `emit_streamk_decode` turns a linear macro-tile id into `(m_tile, n_tile,
  k_iter, is_first, is_last)` via mod/div (K-major within an (m,n) tile — i.e. every
  macro-tile is one K-slice of one output tile, and consecutive linear ids walk K before
  moving to the next output tile). This is a strictly more dynamic/flexible version of
  MISA's fixed `gemm_k_global_split` count: instead of choosing one split factor ahead of
  time via ternary search, work is chunked into many small K-iteration units and CTAs pull
  them from a global counter, so the actual K-parallelism naturally adapts to how many CTAs
  show up (round-robin over a persistent grid) rather than being fixed at launch.
- Two reduction strategies are modeled: `Atomic` (each contributing CTA does
  `global_atomic_add` of its partial f32 sum into a workspace at the output's (m,n)
  position — shipped, this is what `streamk_gemm.py` uses end-to-end) and `Reduction`
  (cooperative flag-table + last-writer-reduces, avoids a second finalization kernel/pass —
  decode-only, not fully shipped). MISA's wrw atomic-K-split is the `Atomic` strategy
  already (grid.z + `global_atomic_add`), so MISA is not missing this idea; what MISA is
  missing is the *persistent, dynamically-chunked* grid dispatch on top of it.
- SGPR-pinning discipline worth copying regardless of stream-K: every wave-uniform i32
  derived from the macro-tile id (`m_tile`, `n_tile`, `k_iter`, `*_base`) is explicitly
  wrapped through `amd_wave_read_first_lane`-equivalent scalarization so the compiler keeps
  per-tile address arithmetic in SGPRs instead of re-materializing it in VGPRs on every use
  inside the K loop and atomic-store epilogue. This is a narrow, low-risk ISA-inspection
  target (see §5) independent of adopting stream-K itself.

**What rocKE's wgrad conv actually does today** (`conv_implicit_gemm_wgrad.py`,
`WgradConvSpec`), for direct comparison against MISA's mechanism: static
`split_k` parameter, grid `(ceil(N_wg/tile_n), ceil(M/tile_m), split_k)`, caller
zero-initializes `dW`, kernel always atomic-adds (never a plain store) when `split_k>1`.
One genuinely new, concrete, low-effort idea here: **rocKE's split-K atomic reduction
supports native packed atomics per output dtype**, not just fp32 scalar atomics —
`global_atomic_add_pk_bf16` (`<2 x bfloat>`) and `global_atomic_add_pk_f16` (`<2 x half>`)
for bf16/fp16 outputs, alongside the fp32 scalar `global_atomic_add`. MISA's own codebase
(`python/operations/global_memory.py`) already *lists* `buffer_atomic_pk_add_f16` as an
instruction and MISA's fwd/bwd MFMA-path comments (`igemm_fwd_gtc_nhwc.py`,
`igemm_bwd_gtc_nhwc.py`, both CDNA/gcn, not the gfx1250 WMMA path) say "prefer use
buffer_atomic_pk_add_f16" — but the gfx1250 WMMA wrw store path
(`python/operations/coalescing_store_wmma.py`) only emits `global_atomic_add_f32`, one
scalar atomic per accumulator element, unconditionally. If `v_pk_add`-style packed atomic
add exists on gfx1250/RDNA4 hardware (this needs verification — rocKE's own header
comments say "gfx940+", i.e. CDNA, and do not claim gfx1250/RDNA4 support), switching
MISA's wrw fp16/bf16 K-split epilogue to a packed 2-element atomic would halve the number
of atomic transactions in the highest-contention part of the kernel (the K-split path is
already MISA's worst-performing case, up to 26x slower on small non-exact shapes, so atomic
contention is plausibly part of that gap).

## 5. ISA-inspection / kernel-trace-capture methodology — directly reusable for MISA's upcoming rocprof work

`isa-inspection-rocke.md` and `capture-kernel-trace-rocke.md` describe a workflow that is
tooling-specific to rocKE (its own probe scripts, its own kernel-naming convention) but the
*methodology* underneath is architecture- and tool-agnostic and directly liftable:

- **Never trust an ISA change without checking the surrounding instruction mix.** The
  doc's core discipline: confirm the intended matrix instruction (`v_wmma_*`) appears, then
  separately count "hidden conversion tax" opcodes in the hot loop (`v_cvt_*`, `v_pk_*`,
  `v_perm_b32`, `v_lshrrev`/`v_ashrrev`, `v_and_b32`) — a native low-bit/packed instruction
  can still leave the kernel VALU-bound if the surrounding pack/unpack/clamp code didn't
  shrink. For MISA specifically (RDNA/WMMA section of the doc is explicit about this): "a
  native WMMA opcode alone does not imply the kernel is matrix-bound" and RDNA coordinate
  arithmetic (address decomposition for conv's non-power-of-2 tails) is called out as
  "especially visible" compared to CDNA. This maps directly onto MISA's known weak spot:
  tail-handling paths (EXEC masks, non-exact shapes) being 1.5x-4x (fwd/bwd) to up to 26x
  (wrw) slower — the prescribed next step is literally to count address/mask instructions
  (`v_cmp`/`v_cndmask`/shift-mask address decomposition) in the tail path ISA and check
  whether they, not the WMMA instructions themselves, dominate.
- **Reporting template** (isa-inspection doc, "Reporting Template" section) is a ready-to-
  reuse checklist: expected op present/count, old-path-removed check, waitcnt/barrier
  count before→after, memory/LDS op count before→after, VGPR/SGPR/LDS + spill status
  before→after, and an explicit verdict line ("keep/revert/measure more"). This is
  low-effort to adopt verbatim as a template for MISA's own optimization-change writeups.
- **rocprofv3 PMC counter set** (`capture-kernel-trace-rocke.md`, "Alternative: PMC
  Profiling" — relevant since ATT requires installing `rocprof-trace-decoder` separately,
  which may not be available in every environment): a two-pass PMC config using
  `MfmaUtil, VALUBusy, MemUnitBusy, MemUnitStalled, ALUStalledByLDS, LDSBankConflict,
  MeanOccupancyPerActiveCU` in pass 1 and `FetchSize, WriteSize, VFetchInsts, VWriteInsts`
  in pass 2 gives high-level bottleneck categorization without needing the full ATT
  decoder — a reasonable first counter set for MISA to start with before committing to ATT
  tooling. (Note: `MfmaUtil` is MFMA-specific; MISA would need the WMMA-equivalent counter
  name, which this doc does not give — worth checking `rocprofv3 --list-counters` on the
  target gfx1250 box.)
- **A subtle but important gotcha documented from real debugging**: `code.json`'s
  `Latency`/`Stall` columns from the ATT decoder are **hit-weighted totals summed over
  every execution**, not per-instance averages — dividing by the `Hit` column is required
  to get a real per-execution cost, and skipping this step produces stall numbers that
  exceed the kernel's whole wall-clock time. This is exactly the kind of misread that would
  silently corrupt an early rocprof-based analysis; worth flagging to whoever starts MISA's
  ATT work.
- **ISA-extraction workflow** for a case with no available disassembler: lower to LLVM IR
  and run `amdclang++ -O3` directly, then `rg` for
  `global_load_lds|global_load|buffer_load|ds_read|ds_write|v_mfma|v_wmma|s_waitcnt|
  s_barrier|v_cvt|global_atomic` on the resulting `.s`. Straightforward and reusable as-is
  since MISA already emits raw `.s`.

## 6. gfx1250/RDNA4-specific architecture facts found in rocKE

Two concrete, gfx1250-specific facts (not generic CDNA carryover) surfaced during the
`grep -rl "gfx125|gfx1250|RDNA4"` sweep:

- **gfx1250's native fp16/bf16 WMMA atom is K=32 per instruction, not K=16.**
  `core/arch/target.py` documents gfx1250 as "wave32/WMMA like gfx12 RDNA but the primary
  fp16/bf16 atom is deeper-K (`16x16x32`), matching CDNA MFMA's K-depth rather than
  gfx1201/RDNA4's `16x16x16`" (comment explicitly flags this as a *hypothesis* verified
  empirically by `examples/gfx1250/wmma_probe.py`, i.e. rocKE reverse-engineered this by
  testing, not from public docs). This matches MISA's own tile/K assumptions (K=32 default
  for fp16/bf16 in MISA's WMMA path) — a cross-check, not a new idea, but useful
  confirmation that MISA's basic tiling assumption is aligned with independently-verified
  hardware behavior. rocKE's fp8/bf8 gfx1250 WMMA atom is `16x16x64` (K=64) — if MISA ever
  adds fp8/bf8 support this is the atom shape to target.
- **A gfx1250-specific "wavelet" pipeline: dedicated load-waves + dedicated math-waves,
  exploiting separate hardware VMEM/WMMA issue slots for real concurrency.** This is the
  single most novel, concretely-implemented (not just documented) gfx1250-specific
  mechanism found, in `instances/common/conv_implicit_gemm.py` +
  `helpers/schedule.py`'s `SchedulePolicy.for_pipeline("wavelet")`. Mechanism: the launched
  workgroup is *oversized* — `launch_block_size = block_size + num_load_waves * wave_size`
  (default `num_load_waves=4`) — and the extra waves are a *different role*, not more
  compute parallelism: they exclusively issue `DRAM→LDS` transfers (`CoalescedTileLoader`
  sized to `num_load_waves * wave_size` threads) via `scf_if_else` (an LLVM `br i1` role
  branch, not exec-masking), while the original `warp_m × warp_n` "math waves" only ever
  do LDS reads + WMMA. A single (non-double-buffered) LDS region is shared by both roles.
  rocKE's own comment is explicit about *why* this works on gfx1250 specifically: "separate
  VMEM and WMMA issue slots provide hardware concurrency without exec-masking" — i.e. this
  is not software pipelining/interleaving within one wave's instruction stream (which is
  what MISA's chunk/compute interleave attempt was, and which MISA measured as a
  regression); it is *spatial* role separation across physically different waves, so the
  hardware's own dual issue paths for VMEM vs. matrix ops provide the overlap instead of
  the compiler/scheduler having to interleave instructions. `is_valid_spec` explicitly
  rejects this pipeline on MFMA/CDNA targets ("the single-buffer LDS is overwritten each K
  iteration and load/math waves execute sequentially rather than truly concurrently" on
  those targets) — i.e. rocKE itself asserts this is a gfx1250/WMMA-only technique, not a
  generic one. There is a hard compile-time guard (`_WMMA_COST_LIMIT = 4096` on
  `k_iters * mfmas_m * mfmas_n`, because the WMMA K-loop is fully unrolled and comgr
  compile time blows up past that) worth knowing about if MISA prototypes something
  similar with a Python-driven sweep, though it's not applicable to MISA's own
  assembly-emission path.
- No arch-specific LDS bank-count/width facts for gfx1250 exist anywhere in rocKE (the LDS
  skill doc only covers gfx942/gfx950 numbers) — this is a gap in rocKE too, not something
  to copy, but confirms MISA would need to establish gfx1250's real LDS bank count/width
  from hardware or ISA docs itself if it wants to replicate the padding-vs-XOR analysis in
  §2.

## Recommendations for MISA, prioritized

1. **(Highest priority — wrw occupancy) Prototype a persistent, dynamically-chunked K-split
   for wrw instead of (or as a variant alongside) the current static ternary-search
   `gemm_k_global_split`.** Concretely: replace the fixed split-factor grid.z with a small
   constant grid (sized to CU count) where each workgroup pulls `(m_tile, n_tile, k_chunk)`
   triples from a global atomic counter (mirroring
   `rocke/platform/python/rocke/helpers/persistent.py` +
   `rocke/platform/python/rocke/helpers/streamk.py`'s decode math — both are pure
   integer/SSA logic, easy to reimplement directly in MISA's Python codegen without
   depending on rocKE). This targets exactly MISA's stated worst case (wrw's small-M/N,
   huge-K, non-exact shapes, up to 26x slower). MISA file/mechanism to change:
   `gemm_k_global_split` machinery in the wrw codegen path (the ternary-search split-count
   selection) plus the wrw grid/kernarg setup. Effort: **high** — this is a structural
   change to wrw's launch/grid model and needs a new persistent-loop control-flow shape in
   hand-written assembly (no IRBuilder convenience layer to lean on, unlike rocKE). Risk:
   **high** — correctness-sensitive (atomic-accumulate ordering, tail/leftover-tile
   handling, the documented barrier-elision hardware footgun in §4 is a warning that this
   class of bug is real and has bitten a mature codebase). Worth prototyping on a narrow
   shape set first before generalizing.

2. **(Second priority, same wrw problem, much lower effort) Investigate whether packed
   `v_pk_add`-style atomic add (2x bf16/fp16 per atomic) is available on gfx1250, and if so
   switch the wrw K-split epilogue's atomic accumulation from
   `global_atomic_add_f32`-per-element to a packed 2-wide atomic for bf16/fp16 outputs.**
   MISA file/mechanism: `python/operations/coalescing_store_wmma.py` (`_emit_scalar_atomic`
   equivalent path, currently unconditionally `global_atomic_add_f32`); the instruction
   already exists as `buffer_atomic_pk_add_f16` in `python/operations/global_memory.py` and
   is already used conditionally in the MFMA (gcn) fwd/bwd paths, so the codegen plumbing
   partially exists — the gap is specifically in the gfx1250 WMMA wrw store path. Effort:
   **low-to-medium** (mostly wiring an existing instruction into an existing epilogue,
   pairing adjacent accumulator slots). Risk: **medium** — needs an ISA-inspection check
   (§5 methodology) to confirm the instruction is legal/correct on gfx1250 hardware before
   trusting it, since rocKE's own docs only claim gfx940+ (CDNA) support for this.

3. **(Worth a focused experiment on MISA's known interleaving failure) Try gfx1250's
   "wavelet" role-separation pattern (dedicated load-waves vs. dedicated math-waves) as an
   alternative to MISA's chunk/compute interleaving, which was measured as a regression.**
   Since rocKE's own reasoning is that gfx1250 has separate hardware VMEM/WMMA issue paths
   that give real concurrency across *waves* (not within one wave's interleaved instruction
   stream), this is mechanistically different from what MISA already tried and rejected —
   it is not guaranteed to help, but it targets the same goal (overlap load latency with
   WMMA compute) via a different hardware mechanism, so the prior negative result doesn't
   necessarily predict this one. MISA file/mechanism: the main K-loop structure in the
   fwd/bwd WMMA codegen (would require restructuring the wave/thread role assignment within
   a workgroup, i.e. oversizing the launched block and branching a subset of waves into a
   pure-loader role). Effort: **high** (new control-flow shape, new LDS-lifetime
   reasoning since both roles share one region). Risk: **medium** — no atomics/reduction
   correctness risk like #1, but real risk of another measured regression given MISA
   already has one negative data point on a conceptually related idea; treat as a
   time-boxed experiment, not a committed rewrite.

4. **(Low effort, no correctness risk) Adopt the SGPR-pinning discipline for macro-tile
   address arithmetic.** rocKE's stream-K decode explicitly scalarizes every wave-uniform
   value derived from a tile/K-split index before it's used in per-thread address math or
   atomic-store epilogues (§4). MISA file/mechanism: the wrw K-split epilogue's address
   computation (wherever the current `gemm_k_global_split` derives per-CTA base offsets) —
   worth an ISA-inspection pass (§5 method) to check whether MISA's current codegen already
   keeps these in SGPRs or is re-materializing them in VGPRs on every use. Effort:
   **low** (an ISA-grep check first, then a targeted fix only if the check finds a real
   problem). Risk: **low**.

5. **(Low effort methodology adoption, not a code change) Bring rocKE's ISA-inspection
   reporting template and PMC counter set into MISA's rocprof workflow as it stands up.**
   Specifically: (a) use the "expected op / old path removed / waits+barriers before→after
   / memory+LDS before→after / VGPR+SGPR+LDS+spills before→after / verdict" template
   verbatim for MISA's own optimization-change writeups (§5); (b) start MISA's rocprof
   counter set with `VALUBusy, MemUnitBusy, MemUnitStalled, ALUStalledByLDS,
   LDSBankConflict, MeanOccupancyPerActiveCU` plus `FetchSize/WriteSize/VFetchInsts/
   VWriteInsts` as a PMC-only first pass before committing to full ATT tooling
   (`rocprof-trace-decoder` is a separate, non-ROCm-shipped install); (c) specifically
   target MISA's tail-handling ISA (EXEC-mask paths, TDM boundary code) with the
   "conversion/coordinate-arithmetic tax" checklist from §5 — count `v_cmp`/`v_cndmask`/
   shift-mask address-decomposition instructions in the tail path and compare against WMMA
   instruction count, since this is the most direct, lowest-risk way to get evidence on
   *why* MISA's tail-handling paths are 1.5x-4x (fwd/bwd) to 26x (wrw) slower before
   committing effort to a specific fix. Effort: **low**. Risk: **none** (pure measurement).
   Remember the hit-weighted-totals-vs-averages gotcha in `code.json` (§5) if/when MISA
   does get ATT working.

6. **(Speculative, needs hardware verification first, no immediate action) Re-derive
   gfx1250's actual LDS bank count/width and re-run the padding-vs-XOR-swizzle analysis
   from §2 against MISA's current K-tile stride.** rocKE has no gfx1250 LDS bank data to
   borrow (its skill doc only covers gfx942/gfx950), so this is not a transplant — it is a
   suggestion to run the *same kind of analysis* rocKE ran to get its measured +43% padding
   win on gfx950, using gfx1250's real (currently unknown-to-rocKE) bank parameters. Effort:
   low to check, unknown to fix depending on outcome. Risk: none to check.
