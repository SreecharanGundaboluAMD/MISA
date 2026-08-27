# External research findings: FlyDSL and hipconv (gfx1250 / TDM)

Working notes from researching two other AMD ROCm repos on this machine
(`/home/sgundabo/FlyDSL`, `/home/sgundabo/hipconv`) for lessons portable to MISA's gfx1250
WMMA work. Written so this survives a context reset -- if you're picking this up cold, read
`docs/gfx1250_wmma_layout.md`'s Phase 27/28 entries first for what MISA has actually shipped;
this file is background research, not a record of MISA's own changes.

## Why this research happened

After Phase 27 (BF16-accumulate WMMA) fixed a VGPR ceiling, the user asked to look at
FlyDSL (`/home/sgundabo/FlyDSL`, a newer MLIR-based GPU kernel DSL) and hipconv
(`/home/sgundabo/hipconv`, a HIP C++ convolution kernel library) for portable ideas.
gfx1250 is referred to as "CDNA5" in AMD's own ISA doc naming (confusingly, since it's
RDNA4-class hardware) -- both other repos use "CDNA5"/gfx1250 interchangeably too.

## Headline finding: TDM (Tensor Data Mover)

A dedicated gfx1250 DMA unit (`TENSOR_LOAD_TO_LDS`/`TENSOR_STORE_FROM_LDS`, ISA doc
§10.11, lines ~6004-6362) that MISA had never used before this research. Both FlyDSL and
hipconv use it extensively in production gfx1250 kernels. Capabilities documented and/or
confirmed:

- Whole-tile global-memory <-> LDS transfer in one async instruction (`S_WAIT_TENSORCNT` to
  synchronize), no VGPR staging.
- **Hardware OOB handling**: loads zero-fill out-of-bounds rows in LDS; stores silently
  drop out-of-bounds writes. Controlled by the descriptor's `tensor_dim`/`tile_dim` fields
  -- no EXEC masking needed at all.
- **Hardware LDS padding** (`pad_interval`/`pad_amount`) for bank-conflict avoidance --
  no software pad-stride computation needed.
- **Workgroup Cluster multicast**: a nonzero `workgroup_mask` on a load causes the TDM to
  fill multiple workgroups' LDS from one HBM read (`CLUSTER_LOAD_ASYNC` instead of
  `GLOBAL_LOAD_ASYNC`).
- **Gather/scatter mode**: descriptor groups 2/3 redefined to carry row indices (8x i32 or
  16x i16) for non-contiguous row access.

### Confirmed on real hardware by MISA (this session, verified independently of both other repos)

- `__builtin_amdgcn_tensor_load_to_lds`/`_store_from_lds` both work on this GPU.
- Hardware OOB zero-fill (load) and drop-on-write (store) both confirmed exactly as
  documented, via a standalone probe (declared a tensor smaller than the requested tile,
  checked the extra rows came back zero / left a sentinel value untouched).
- **Confirmed through MISA's actual pipeline**, not just HIP C++: extracted the compiled
  probe's `.s`, assembled it via `/opt/rocm/llvm/bin/clang++ -x assembler -target
  amdgcn--amdhsa -mcpu=gfx1250` (MISA's real `python/codegen/compile.py` command), loaded
  via `hipModuleLoad`, launched via `hipExtModuleLaunchKernel` (MISA's real
  `driver/igemm_gtc_base.h:470` dispatch call) -- both load and store passed.
- Real assembly syntax: `tensor_load_to_lds s[d0:d0+3], s[d1:d1+7]` /
  `tensor_store_from_lds s[d0:d0+3], s[d1:d1+7]` for a 2D tensor (group0=4 SGPRs,
  group1=8 SGPRs; VADDR2/VADDR3 omitted from the mnemonic when NULL/unused), plus
  `s_wait_tensorcnt N`.
- Descriptor bit-packing (group0: `[pred, lds_addr, global_addr_lo, global_addr_hi|type]`;
  group1: `[data_size<<16|workgroup_mask, tensor_dim0_lo16<<16, tensor_dim0_hi16|tensor_dim1_lo16<<16, tensor_dim1_hi16|tile_dim0<<16, tile_dim1|tile_dim2<<16, stride0_lo32, stride0_hi16|stride1_lo16<<16, stride1_hi16]`)
  cross-checked against both the ISA doc's Tables 62-63 AND FlyDSL's own
  `lib/Dialect/FlyROCDL/GFX1250/CopyAtom.cpp` -- all three agree.

**MISA status as of this writing**: Phase 28 (in progress) is piloting a TDM-based A-operand
load for fwd, 1x1-conv-only, as an opt-in `tdm_global_load` tunable alongside (not replacing)
the existing `async_global_load` path. See `docs/gfx1250_wmma_layout.md`'s Phase 28 entry
for the actual shipped design once it lands.

## FlyDSL findings (two research agents, full detail in conversation history if needed)

### Cluster multicast is the real use for Workgroup Clusters (not atomic contention)

MISA separately investigated Workgroup Clusters for wrw's K-split atomic-reduction
contention and found no win (dropped -- see `docs/gfx1250_wmma_layout.md`'s cluster
investigation notes). FlyDSL uses clusters for a completely different purpose: TDM
multicast data reuse. `feat/gfx1250-cluster-launch` branch, `python/flydsl/expr/rocdl/cluster.py`
-- `compute_mcast_masks()` builds workgroup_mask bitmasks so all workgroups sharing an
M-tile row (or N-tile column) receive the SAME A-tile (or B-tile) load via one HBM read.
Production 256x256 GEMM kernels (`gfx1250/gemm-opt-0805` branch) use 4x4 clusters (the
16-workgroup hardware cap) specifically for this. Does NOT use `map_shared_rank`/cross-WG
LDS sharing (confirmed absent from this toolchain, same finding MISA already had).
Cluster sync: `s_barrier_signal(-3)`/`s_barrier_wait(-3)` (cluster-scope barrier, distinct
from the regular `-1` workgroup barrier).

### VGPR-MSB indexing is real but actively avoided, not used deliberately

FlyDSL commit `b98dbe2d` ("pre-calc epilogue addresses to eliminate all s_set_vgpr_msb")
treats the compiler's automatic `S_SET_VGPR_MSB` insertion (when live VGPRs exceed 256) as
something to eliminate via scheduling (precompute addresses in the WMMA execution shadow),
not a mechanism to use on purpose. Confirms real kernels do occasionally exceed 256 VGPRs
(up toward 1024) but treat it as a bug to schedule away, matching MISA's own decision not
to pursue MSB-indexing in Phase 27.

### Narrow-accumulate WMMA (bf16/f16-native) exists in FlyDSL's compiler but is UNUSED

`lib/Dialect/FlyROCDL/GFX1250/MmaAtom.cpp` has the same 4-VGPR narrow-accumulate layout
MISA just shipped (Phase 24/27), but no FlyDSL GEMM kernel actually uses it yet -- all use
8-VGPR f32 accumulators. MISA is ahead of FlyDSL on this specific optimization.

### Epilogue: TDM store is the norm; LDS reshuffle is sometimes skipped entirely

Two patterns found, both different from MISA's `coalescing_store_wmma.py` LDS-reshuffle
design:
- **Direct per-lane `buffer_store_b128`**, no LDS at all -- FlyDSL found gfx1250's WMMA C
  layout already has 16 consecutive lanes covering 16 consecutive columns, contiguous
  enough for direct stores without a reshuffle stage. (MISA's own
  `coalescing_store_wmma.py` docstring independently makes the same observation about lane
  layout -- worth re-examining whether the LDS reshuffle is actually necessary.)
- **LDS scatter + single TDM store** for the general case -- `tensor_store_from_lds` moves
  the whole tile in one instruction, with boundary handling via the store descriptor's
  `mn_oob`/`tensor_dim` fields (hardware-clips, no EXEC masking).

### K-tail: no GEMM kernel handles it; the one example is host-side zero-padding

Every FlyDSL gfx1250 GEMM kernel requires `K % tile_k == 0`, hard assert, no exceptions.
The only K-tail handling anywhere in the repo is in a conv kernel
(`conv3d_implicit_fp8.py`): pads the weight tensor's K dimension up to a tile boundary
**on the host**, allocating a zero-padded copy before launch, rather than handling a
partial tile in the kernel at all. This is a real, simple option MISA hasn't considered:
if the caller controls allocation, host-side padding sidesteps the K-tail problem entirely
without any kernel change -- but only works when MISA's driver/test harness can control
input allocation, which may not hold for arbitrary caller-supplied conv tensors in
production use. Still worth keeping in mind as a cheap fallback if TDM's own K-axis OOB
handling (untested on MISA's side so far -- Phase 28 doesn't attempt K-tail yet) turns out
not to cover the main-loop's chunked-load pattern cleanly.

### Split-phase barriers and back-to-back WMMA issue (smaller, easy wins)

- `s_barrier_signal(-1)` issued early (right after TDM loads), `s_barrier_wait(-1)` issued
  only when the data is actually needed (FlyDSL: 7 WMMAs later) -- overlaps barrier
  propagation latency with compute. Directly encodable in MISA's hand-written assembly.
- `disable_xdl_arb_stall()` -- a `S_SETREG_B32` write to `SCHED_MODE` bit 4 -- lets WMMAs
  issue back-to-back without a multicycle arbitration stall. FlyDSL's hgemm calls this
  once at kernel entry. Untested by MISA; worth a quick llvm-mc + hardware check.

## hipconv findings

`arch/cdna5/depthwise/depthwise_1d_toeplitz/` is a REAL, shipped gfx1250 depthwise
convolution kernel using "TDM row-streaming" (per hipconv's README), documented at
`docs/algorithms/toeplitz/depthwise-1d-toeplitz-cdna5.md` -- the single most directly
relevant prior art found across both repos, since it's TDM + convolution + gfx1250
specifically (FlyDSL's TDM usage is all generic GEMM, not convolution). Also examined:
`hipconv/src/arch/cdna5/tdm_desc.h`, `bunnies_mi400.hpp` (gfx1250 primitives), the
`direct/` and `grouped/` kernel families, and `config_table.h`/`docs/config-ranking.md`.

### Descriptor bit-packing: independently confirms MISA's implementation

hipconv's `bunnies_mi400.hpp` (lines 736-1015) breaks down group0/group1 the same way
MISA's Phase 28 code does and the same way FlyDSL's `CopyAtom.cpp` does -- three
independent sources now agree exactly on the bit layout. One structural note: hipconv's
`__builtin_amdgcn_tensor_load_to_lds` call passes all 5 descriptor groups (zeroing
groups 2-4 when only 2D is needed); MISA's hand-written `tensor_load_to_lds
s[d0:d0+3], s[d1:d1+7]` 2-operand form (VADDR2/VADDR3 omitted from the mnemonic) is the
correct hand-assembly equivalent for 2D -- confirmed working on real hardware already.

### GEMM_K tail: solved exactly the way expected, and confirmed in production

hipconv's depthwise kernel sets `tensor_dim0` (the channel/K extent) to the ACTUAL
remaining channel count while `tile_dim0` stays at the full workgroup tile width --
`tdm_chan_ext = max(0, min(WG_CH, C - wg_ch_base))`. A wave whose channel slice runs past
the real channel count gets hardware zero-fill instead of a software guard. **This is a
production-proven answer to MISA's currently-unsolved GEMM_K tail**: set `tensor_dim0 =
s_gemm_k` (the real, unpadded K) instead of the padded/tile value, and the TDM engine
handles the rest. MISA's Phase 28 pilot already sets `tensor_dim0` correctly (to
`s_gemm_k`) but doesn't yet test or rely on the OOB path for K -- confirmed by an
independent, shipped kernel that this is exactly the right lever to pull next.

**Correction/refinement, found while implementing Phase 31**: hipconv's depthwise
kernel's `tdm_chan_ext = C - wg_ch_base` is a SINGLE-SHOT computation (depthwise has no
K-reduction loop -- each output channel depends only on its own input channel, so there's
no advancing `global_addr` across multiple calls to the same descriptor). MISA's fwd
kernel DOES loop over K with an advancing `global_addr`, and a standalone hardware probe
(see `docs/gfx1250_wmma_layout.md`'s Phase 31) confirmed `tensor_dim0`'s OOB check is
relative to *that call's* `global_addr`, not a fixed tensor origin -- so holding
`tensor_dim0` constant at the original `s_gemm_k` across iterations (which is what "sets
`tensor_dim0` correctly" above was describing, and what Phase 28-30 actually shipped) is
only correct for iteration 0; it must be DECREMENTED by the tile width every iteration
(mirroring `wg_ch_base`'s role) for a real K-tail to zero-fill correctly instead of
reading past-the-end memory. Phase 31 implements this; the depthwise formula's shape
(`remaining = total - current_base`) is exactly right, just needs re-deriving every loop
iteration for a looped consumer rather than computing it once.

### GEMM_M/spatial tail: three-dimension technique, no EXEC masking anywhere

The depthwise kernel handles ALL of its boundaries (channel, right spatial edge,
top/bottom spatial edge) purely through TDM descriptor extents, never through per-lane
EXEC masking:
- Right edge: `tensor_dim1 < tile_dim1` on the column dimension zero-fills the halo.
- Top/bottom: a 3D descriptor's `tensor_dim2` is set to 0 for an entirely out-of-bounds
  row (`tile_dim2=1` always; `tensor_dim2 = 1 if row in-bounds else 0`) -- the whole tile
  reads as zero when the row doesn't exist, no need to special-case it.
- The ONE boundary that can't be expressed this way is the **left halo** -- zeroed once in
  LDS before streaming begins, since TDM only clips at the trailing/high end of an extent,
  not the leading edge.
This is a strictly more general (and more hardware-offloaded) technique than Phases
25/26's per-lane EXEC-mask guards -- worth revisiting those phases once TDM covers more of
the kernel, though that's a large follow-on, not something to retrofit casually.

### Measured, concrete tuning findings (post-correctness, for later phases)

- **Single-issuer-wave**: only one wave should issue a given TDM transfer, not every wave
  redundantly (which is exactly what MISA's Phase 28 pilot currently does, deliberately,
  as its "correctness first" simplification). hipconv measures round-robin issuing at
  -3% to -7% over naive all-waves-issue. Directly explains why Phase 28's first hardware
  timing came back slower than the existing `_async` path -- expected, not a red flag.
- **Don't drain LDS reads before the next TDM issue**: hipconv measured 1.4-5.3% lost by
  waiting on `s_wait_dscnt` before issuing the next `tensor_load_to_lds`. MISA's main loop
  should issue the next TDM load before/without waiting on the current tile's `ds_read`s.
- **DMA engine parity**: gfx1250 WGPs have two DMA engines, selected by wave parity (even
  vs odd). Assigning the load-issuer and store-issuer roles to opposite parities measured
  19% -- a concrete, cheap-to-implement idea once MISA has both a TDM load and a TDM store
  path.
- **Hardware LDS padding** (`pad_interval`/`pad_amount`) used in three different kernels
  for three different bank-conflict shapes, all computed as simple closed-form functions
  of tile width in DWORDs. A direct, hardware-offloaded replacement for MISA's Phase 23
  software `epilogue_lds_pad`, applicable to the main-loop LDS as soon as TDM lands there
  more broadly (Phase 28 doesn't need it yet -- unpadded layout, byte-identical to today).
- **`s_wait_tensorcnt(N)` with N>0** (not just 0): hipconv's ring keeps `prefetch_depth - 2`
  loads in flight rather than draining to 0 every iteration -- deeper pipelining than
  Phase 28's single-outstanding-load design. A natural next step once single-issuer-wave
  and multi-buffering are in place.
- **TDM store for the epilogue**: `tensor_store_from_lds` with the same descriptor-extent
  boundary clipping, no EXEC masking. Confirms FlyDSL's independent finding that TDM store
  is the production norm for the epilogue, not MISA's current software LDS-reshuffle.
  Subtlety hipconv's doc calls out explicitly: the store's `tile_dim` must stay at the full
  staged width even when the actual write is clipped narrower, because "the LDS cursor
  advances by the tile shape regardless of clipping" -- a narrowed tile_dim would silently
  desync the LDS read cursor from the intended row stride.

### Not immediately relevant to the current pilot, noted for later feature work

- `B_reuse` WMMA flag (skip re-reading an unchanged B operand across consecutive WMMA
  calls) -- relevant once MISA's K-loop has an invariant-B pattern to exploit.
- `ds_load_tr16_b128`/`global_load_tr16_b128` (hardware transpose-on-load) -- relevant if
  MISA ever needs a transposed WMMA operand without a software LDS transpose (bwd/wrw
  already do their own transpose addressing in software; worth a look if that ever becomes
  a bottleneck).
- `TH_ATOMIC_CASCADE_RT` used successfully in hipconv's grouped wgrad kernel
  (`global_atomic_add_f32 ... th:TH_ATOMIC_CASCADE_RT scope:SCOPE_DEV`) -- directly
  contradicts MISA's own Phase 23 finding that this hangs on real hardware! Worth a
  focused follow-up: MISA's cascading-atomic hang was traced to a missing companion
  release/fence instruction (see `docs/gfx1250_wmma_layout.md`'s Phase 23 TODO) -- hipconv
  evidently issues something MISA doesn't. Re-examine hipconv's exact usage context before
  reviving Phase 23's blocked cascading-atomic work.
- LDS-scoped (not workgroup-scoped) fences (`__ATOMIC_RELEASE`/`ACQUIRE` with `"local"`
  scope) for cross-wave LDS hand-off without retiring in-flight global stores onto the
  barrier's critical path -- a possible small win for MISA's existing barrier usage,
  low priority.

### Config/kernel selection (lower relevance)

hipconv uses compile-time `constexpr` config tables (not MISA's text `.config` files) with
a weighted-throughput-index ranking. Conceptually similar occupancy reasoning (e.g.
preferring 3 groups of 4 waves over 1 group of 8 when VGPR-limited) but a fundamentally
different mechanism (C++ template generation vs. Python-generated assembly) -- not
directly portable, more a confirmation that MISA's own occupancy-driven tile-shape
reasoning (Phase 22 VGPR audits, Phase 24/27 accumulate-width work) is the right instinct.
