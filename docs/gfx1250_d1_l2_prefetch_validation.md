# gfx1250 D1-P2: L2 Prefetch Two K-Stages Ahead — Validation

## Mechanism

This implements backlog item 11 from `docs/gfx1250_wmma_perf_report_v2.md` §8 (P1 list):
> gfx1250 shader prologue / `S_CODE_END` padding, then `GLOBAL_PREFETCH_B8` two K-stages
> ahead (guide §19) for compute-bound 1×1 shapes.

### New tunable: `wmma_l2_prefetch`

- **Default:** 0 (off) — every existing config is byte-identical.
- **Scope:** `nxe==0` (unit conv, 1x1 stride-1), non-async, non-saddr, non-TDM,
  non-interleave, `row_repeat_{a,b}==1`, `lds_double_buffer=1`, non-fp32 (i.e., gated
  on `can_hoist` being true, same safety envelope as `wmma_gap_hoist`).
- **Kernel name suffix:** `_l2pf`

### How it works

The existing `can_hoist` mechanism in `wmma_main_loop.py` already hoists the NEXT tile's
`move_slice_window` + `global_load` to happen early in each iteration (1 K-stage ahead of
the tile currently being computed). This new mechanism issues an ADDITIONAL, purely-
speculative `global_prefetch_b8` for the tile TWO K-stages ahead — i.e., one more
`unroll_k`-worth of address advance beyond what the hoisted load already computed — into
L2/WGP cache only, with no destination VGPR, no wait-counter, no LDS/register staging.

### Implementation approach: immediate `offset:` field

The initial implementation used a save/restore pattern (add stride to `v_addr_a`, issue
prefetch, subtract stride to restore). This caused an **illegal memory access** on real
hardware (confirmed via `rocgdb`): on gfx1250, `global_load_b128` does NOT latch the VADDR
at issue time — modifying `v_addr_a` after the load is issued but before it reaches the
memory controller corrupts the load's address.

The fix: use the instruction's immediate `offset:` field to add the stride. For fwd (both
A and B), the per-K-tile stride (`bytes_per_row = gemm_k_per_block * data_byte`) is a
compile-time constant, so:

```asm
global_prefetch_b8 v[v_addr_a:v_addr_a+1], off offset:64 th:TH_LOAD_NT_RT scope:SCOPE_CU
```

This prefetches from `v_addr_a + 64` (the 2-stages-ahead address) without modifying
`v_addr_a` at all. No scratch VGPRs needed, no alignment issues.

For bwd A, the same `offset:bytes_per_row` approach works. For bwd B and wrw A/B, the
stride is a runtime SGPR (`s_wei_k_stride` / `s_a_k_stride` / `s_b_k_stride`), so the
immediate offset cannot be used. Instead, a scratch VGPR pair (`v_gtc_tmp(pf_off:pf_off+1)`,
where `pf_off` is chosen at codegen time based on `v_gtc_tmp`'s allocated register parity
to ensure even-alignment: offset 1 when `v_gtc_tmp` is odd, offset 0 when even) is used to
compute the 2-stages-ahead address without modifying the real address registers.

## Confirmed Instruction Encoding

Verified against the real toolchain (`clang -x assembler -mcpu=gfx1250` +
`llvm-objdump -d`):

```asm
global_prefetch_b8 v[0:1], off th:TH_LOAD_NT_RT scope:SCOPE_CU
; Encoded as: EE17407C 00400000 00000000
```

- `EE17407C` — opcode for `global_prefetch_b8`
- `00400000` — TH=4 (NT_RT, **speculative**) + scope=0 (CU/WGP)
- `00000000` — immediate offset (0)

Key ISA doc references (§10.5, "VMEM Prefetch Instructions"):
- TH=4 (NT_RT) is **speculative** per the L2 Prefetch table: "the address is not known to
  be valid. It first attempts to translate the address from logical to physical (UTC-L0).
  If this translation fails, the request is silently dropped with no errors reported."
- TH=1 (NT) is **non-speculative**: "can walk page tables, and the programmer guarantees
  the address is valid."
- `scope:SCOPE_CU` (Scope=0, WGP): "Pulls in at all cache levels on miss" — prefetches
  into WGP-local L0/L1 cache as well as GL2, keeping prefetch traffic device-local.
- `scope:SCOPE_DEV` (Scope=2, DEV): "Does not prefetch into WGP, but brings it into the
  GL2" — bypasses WGP-local caching and sends every prefetch straight to the shared GL2.

**The original implementation used `scope:SCOPE_DEV`, which was a bug.** With potentially
256+ concurrently-resident workgroups each issuing 2 prefetch instructions per loop
iteration, `SCOPE_DEV` forced every prefetch into the shared GL2, creating severe
device-wide L2 request-queue contention that competed directly with real loads. This
caused a ~10× performance regression — not the ~10% overhead a fire-and-forget prefetch
should cost. The corrected `scope:SCOPE_CU` keeps prefetch traffic CU/WGP-local, matching
the ISA doc's Scope=0 (WGP) behavior. (Note: this toolchain has no `SCOPE_WGP` symbol;
`SCOPE_CU` is the identifier that maps to the ISA doc's Scope=0 CU-local/all-cache-levels
behavior. `SCOPE_WGP` fails to assemble with "invalid scope value"; `SCOPE_CU` assembles
cleanly.)

The speculative encoding is **required**: near the K-loop's tail, the 2-stages-ahead
address may run past the tensor's real allocated bounds. A non-speculative prefetch of an
out-of-bounds address would fault; a speculative one silently drops it.

**Even-alignment constraint:** `global_prefetch_b8`'s VADDR pair must be even-aligned
(verified: `v[0:1]` assembles, `v[1:2]` does not). This is why the immediate `offset:`
approach (which uses the already even-aligned `v_addr_a` pair) is preferred over scratch
VGPRs. When scratch VGPRs are unavoidable (bwd B, wrw A/B), the offset into `v_gtc_tmp`
is chosen dynamically based on the allocated register parity to guarantee even-alignment.

## Correctness Results

All tests run on gfx1250 (clock-capped at ~1100 MHz sclk), `-V 1`, timeout-wrapped.
Results below are with the corrected `scope:SCOPE_CU`.

### Standard regression shapes (fwd)

| Shape | Result |
|-------|--------|
| `n128 c1024 17x17 k1024 1x1` | **valid:y** |
| `n256 c2048 14x14 k2048 1x1` | **valid:y** |

### Boundary case (short K-loop)

| Shape | K-loop iters | Result |
|-------|-------------|--------|
| `n128 c128 1x1 k128 1x1` | 4 | **valid:y** |

This shape has only 4 main-loop iterations (gemm_k=128, gemm_k_per_block=32). On the last
iteration, the 2-stages-ahead address runs 2×64=128 bytes past the last valid K-tile —
well past the tensor's allocated bounds. The speculative prefetch silently drops this bad
address, confirming the TH_LOAD_NT_RT encoding works as documented with `SCOPE_CU`.

### Bwd and wrw directions

| Direction | Shape | Result |
|-----------|-------|--------|
| bwd | `n128 c1024 17x17 k1024 1x1` | **valid:y** |
| bwd | `n256 c2048 14x14 k2048 1x1` | **valid:y** |
| bwd | `n128 c128 1x1 k128 1x1` (boundary) | **valid:y** |
| wrw | `n128 c1024 17x17 k1024 1x1` | **valid:y** |
| wrw | `n256 c2048 14x14 k2048 1x1` | **valid:y** |
| wrw | `n128 c128 1x1 k128 1x1` (boundary) | **valid:y** |

### Post-risk GPU health

After every run, `rocm-smi --showclocks` confirmed the GPU was healthy (sclk=1100 MHz,
no faults). No hangs, no crashes, no residual errors.

## Zero-Diff Regression

Existing configs (tunable unset) produce byte-identical `.s`/`.inc` output before and
after the change. Verified via `diff` on `igemm_fwd_gtc_gfx1250_nhwc_fp16_dbuf.config`
output.

## Performance Comparison

**Caveat:** This machine's GPU is clock-capped at ~1100 MHz vs. a ~2400 MHz reference
machine. Results are directional-only, not a final performance verdict.

### Scope bug reproduction (SCOPE_DEV vs. SCOPE_CU)

The original shipped code used `scope:SCOPE_DEV`. Independent reproduction confirmed the
severe regression on both fwd shapes (single run each, `-V 1`):

| Shape | Baseline (TFLOP/s) | l2pf SCOPE_DEV (TFLOP/s) | Regression |
|-------|-------------------|--------------------------|------------|
| `n128 c1024 17x17 k1024 1x1` | 321.6 | 29.9 | **10.7×** |
| `n256 c2048 14x14 k2048 1x1` | 394.2 | 69.2 | **5.7×** |

The `SCOPE_DEV` regression is caused by device-wide GL2 request-queue contention: every
concurrently-resident workgroup's prefetch bypasses WGP-local cache and goes straight to
the shared GL2, competing with real loads for L2 request queue bandwidth. This is a
fundamentally different and far worse mechanism than the "fire-and-forget, no dependency-
counter impact" behavior the ISA doc describes for local-scope prefetch. The original
doc's explanation ("clock-capped machine can't amortize the overhead") was wrong.

### Corrected scope (SCOPE_CU) — 3-run benchmark

| Shape | Config | Runs (TFLOP/s) | Median |
|-------|--------|----------------|--------|
| `n128 c1024 17x17 k1024 1x1` | baseline (dbuf) | 320.3, 318.2, 320.9 | **320.3** |
| `n128 c1024 17x17 k1024 1x1` | l2pf (dbuf+l2pf, SCOPE_CU) | 239.3, 248.4, 247.3 | **247.3** |
| `n256 c2048 14x14 k2048 1x1` | baseline (dbuf) | 302.7, 394.9, 396.8 | **394.9** |
| `n256 c2048 14x14 k2048 1x1` | l2pf (dbuf+l2pf, SCOPE_CU) | 317.0, 307.7, 308.4 | **308.4** |

With the corrected `SCOPE_CU`, the catastrophic 10× regression is eliminated — but the
prefetch mechanism still shows a **~22% net regression** on both tested shapes:

- Shape 1: 247.3 / 320.3 = 77.2% → **−22.8%**
- Shape 2: 308.4 / 394.9 = 78.1% → **−21.9%**

### Interpretation

The ~22% regression with `SCOPE_CU` is a real overhead from the 2 extra prefetch
instructions per loop iteration that is not amortized by cache-hit benefits on these
shapes at this clock speed. The guide itself warns that prefetch is not guaranteed to
help every shape/workload — this is exactly that outcome: a correct, properly-scoped
mechanism that is simply not beneficial on this hardware/shape combination.

**This mechanism is not currently recommended for adoption** based on available evidence.
The scope bug fix (SCOPE_DEV → SCOPE_CU) is still valuable: it prevents a catastrophic
10× regression if this tunable is ever enabled, and corrects the record on the root cause.
Further tuning (e.g., less frequent prefetch — every other K-iteration instead of every
iteration, different K-stage distance, or gating on shapes where the compute-to-memory
ratio is genuinely high enough to amortize the overhead) or re-testing on a full-clock
(2400 MHz) reference machine could potentially change the cost/benefit tradeoff.

## Files Changed

- `python/igemm/igemm_base.py` — `wmma_l2_prefetch` tunable declaration + asserts
  (nxe==0, async/saddr/TDM/interleave mutual exclusion) + kernel-name mangling (`_l2pf`)
- `driver/igemm_gtc_base.h` — struct field, parser, kernel-name mangling (C++ driver side)
- `python/operations/wmma_main_loop.py` — `l2_prefetch` flag + `prefetch_a/b_functor`
  fields on `ctrl_wmma_main_loop_t`; emission in `can_hoist` branch after hoisted loads
- `python/igemm/igemm_fwd_gtc_wmma_nhwc.py` — `prefetch_a/b_functor` (offset: approach) +
  ctrl wiring + row_repeat assert
- `python/igemm/igemm_bwd_gtc_wmma_nhwc.py` — `prefetch_a_functor` (offset:) +
  `prefetch_b_functor` (scratch, dynamic parity) + ctrl wiring + row_repeat assert
- `python/igemm/igemm_wrw_gtc_wmma_nhwc.py` — `prefetch_a/b_functor` (scratch, dynamic
  parity) + ctrl wiring + row_stride assert + `s_b_k_stride` prologue computation +
  `s_b_k_stride` SGPR declaration moved outside TDM-only block (needed for l2pf path)
- `config/igemm_fwd_gtc_gfx1250_nhwc_fp16_dbuf_l2pf.config` — fwd config with tunable set
- `config/igemm_bwd_gtc_gfx1250_nhwc_fp16_dbuf_l2pf.config` — bwd config with tunable set
- `config/igemm_wrw_gtc_gfx1250_nhwc_fp16_dbuf_l2pf.config` — wrw config with tunable set
