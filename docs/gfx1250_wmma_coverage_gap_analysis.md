# gfx1250 WMMA coverage vs. a real MISA-solver-won shape corpus (2026-08-27, updated)

## What this is

Three real MIOpen solver-search traces (`convasmimplicitgemmgtcdynamic_{fwd,bwd,wrw}.txt`,
repo root, gitignored -- not committed, too large) were provided: every shape in them is one
where a MISA-authored solver (`ConvAsmImplicitGemmGTCDynamic{Fwd,Bwd,Wrw}XdlopsNHWC` --
MISA's older MFMA/XDLOPS codegen, integrated into MIOpen as a solver family) **won**
MIOpen's own solver search on whatever architecture the trace was recorded on. This is a
real, large, validated-fast usage corpus, not synthetic stress shapes.

**Question asked**: categorize these shapes and classify each against what MISA's gfx1250
WMMA backend can build *today*, and for what can't be built, identify concretely whether
it's (a) already supported, (b) closable with new `.config` files only, (c) closable with
moderate driver/codegen integration work, or (d) needs genuinely new kernel mechanisms --
and track the gap list here.

**Scale**: 95,066 total entries, 88,041 distinct shapes (deduped by full parameter tuple)
across fwd/bwd/wrw. Running each one on real hardware is impractical and unnecessary for a
*coverage* question -- this is answered by pure static analysis: compute each shape's
GEMM_M/N/K per direction, and check it against MISA's known, already-documented mechanism
support (from `docs/gfx1250_wmma_layout.md`'s Phase 5-37 history) via
`script/classify_gfx1250_coverage.py`.

**This revision** reflects two rounds of follow-on work since the original analysis:
(1) bwd GEMM_N/K-tail (Phase 36) and fwd's TDM+GEMM_M/N-tail combination (Phase 37) closed
the two biggest GEMM-shape mechanism gaps the original analysis found; (2) every remaining
*config-only* gap (a mechanism that already existed in code but only had `.config` files
for one precision) has now been closed with new, hardware-validated config files across
fp16/bf16/fp32 for fwd/bwd/wrw. Layout conversion (NCHW<->NHWC) was **explicitly decided
against** as a MISA-side feature -- see below.

## Headline finding: layout is still the dominant gap in the real corpus, but it's an
explicit non-goal for MISA

| Direction | Total distinct shapes | Non-NHWC layout | NHWC-only (comparable) |
|---|---|---|---|
| fwd | 17,772 | 17,675 (99.5%) | 97 |
| bwd | 18,750 | 18,376 (98.0%) | 374 |
| wrw | 51,519 | 51,262 (99.5%) | 257 |

Over 99% of this real-world corpus uses **NCHW** (MIOpenDriver's own default when no
`--in_layout`/`--fil_layout`/`--out_layout` flags are given) or a **mixed NHWC-input/
NCHW-filter-and-output** layout. MISA's gfx1250 WMMA kernels are NHWC-only by construction.

**Decision, this session**: MISA will not transpose NCHW<->NHWC itself. Historically this
conversion is MIOpen's own responsibility (the layer that wraps a solver and can insert a
transpose before/after dispatch), not the solver's -- and that division of responsibility is
being kept for gfx1250 rather than duplicating it inside MISA's driver. Two things were
investigated and explicitly ruled out as MISA-side substitutes:
- **`driver/gpu_nchw_nhwc_transpose.h`'s existing `gpu_nchw2nhwc`/`gpu_nhwc2nchw` kernels**
  are real but were found to be dead code (never invoked from `conv_driver.cpp`), and their
  kernel-selection helper has an unhandled-case fallthrough (undefined behavior) for shapes
  no candidate tile size divides evenly -- a real correctness hazard, not just unwired.
- **gfx1250's `GLOBAL_LOAD_TR16_B128`/`_TR8_B64`/etc. transpose-load instructions**
  (confirmed assembling via `llvm-mc -mcpu=gfx1250`) only transpose a single 16x16 (or
  16x32) tile into WMMA-operand VGPR layout on load -- no store-side equivalent exists in
  the ISA at all, and the granularity doesn't match a full-tensor NCHW<->NHWC permutation.

Given this, layout is treated here as **out of scope for MISA**, not as a gap MISA is
expected to close. The rest of this document therefore evaluates coverage two ways:

- **As requested** (real layout, matching what the corpus actually asks for) -- layout
  dominates, exactly as before.
- **Assuming layout is resolved upstream** (by MIOpen or whatever wraps MISA) -- isolates
  the GEMM-shape/mechanism coverage question, which is what actually improved this session.

## Coverage assuming layout is resolved upstream (`--assume-nhwc`)

This is the relevant view for judging MISA's own GEMM-shape coverage, and the one that
changed substantially this session:

| Direction | Total entries | Supported today | Gap: real mechanism missing | Out of scope / degenerate |
|---|---|---|---|---|
| fwd | 17,772 | 14,335 (80.66%) | 2,936 (16.52%) | 501 (2.82%) |
| bwd | 21,055 | 20,776 (98.67%) | 0 (0%) | 279 (1.33%) |
| wrw | 56,239 | 55,772 (99.18%) | 0 (0%) | 467 (0.83%) |
| **Overall** | **95,066** | **90,883 (95.60%)** | **2,936 (3.09%)** | **1,247 (1.31%)** |

**bwd and wrw are now fully covered** (modulo depthwise/degenerate shapes, which aren't
real coverage gaps -- see Methodology). **fwd's only remaining gap is K-tail for multi-tap
(non-1x1) convolutions** (16.52% of fwd, 3.09% overall) -- fwd's only K-tail mechanism is
TDM's hardware OOB (Phase 31/37), which is 1x1-conv-only by construction; a genuinely new
software K-tail mechanism (mirroring wrw's Phase 35 design) would be needed to close it.

Before this session's work, the same computation (real mechanism state as of the original
analysis, still assuming layout resolved) gave: fwd 79.98%, bwd 85.69% (bwd's N/K-tail gap
was 12.99% then, since Phase 36 hadn't landed), wrw 62.07% (37.12% was a bf16-only-config
gap). **The 24.96-point jump to 95.60% overall came entirely from closing config-only
gaps** (new `.config` files for mechanisms that already existed in code) -- no new codegen
was needed for that portion; only bwd's N/K-tail and fwd's TDM+M/N-tail combination
(Phases 36/37) required new mechanism work.

### fwd -- the one real remaining gap

| Category | Distinct shapes | Corpus entries | Effort |
|---|---|---|---|
| Supported (M/N-tail, `_mtail`/`_ntail`/`_mntail`) | 8,410 | 8,410 | - |
| Supported (exact tile fit) | 4,744 | 4,744 | - |
| **GAP: K-tail for multi-tap (non-1x1) convs** | 2,936 | 2,936 | **D** |
| Supported (TDM K/M/N-tail combined, Phase 31/37) | 1,181 | 1,181 | - |
| GAP: depthwise (g==c, out of scope) | 458 | 458 | D (architectural) |
| Degenerate (zero valid output positions) | 43 | 43 | n/a |

fwd's *only* K-tail mechanism is TDM's hardware OOB zero-fill, which requires `nxe=0`
(1x1, unit stride, no padding) by construction -- Phase 37 combined it with M/N-tail, but
that doesn't help multi-tap convs at all, since TDM itself was never extended past 1x1.
Closing this would need a genuinely new *software* K-tail for fwd (fwd currently has no
non-TDM K-tail mechanism whatsoever) -- wrw's Phase 35 K-tail (EXEC-mask based, works for
any tap count) is the closest in-tree template, though fwd's non-transposed operand
addressing differs enough from wrw's that it wouldn't be a direct port (see
`docs/gfx1250_wmma_layout.md`'s Phase 36 for a worked example of how "looks like a port"
can turn out to need new masking machinery once the actual per-lane addressing is checked).

### bwd and wrw -- fully covered

Both directions now show `0` remaining GEMM-shape/mechanism gaps in this corpus (aside from
depthwise and degenerate shapes, which are not real coverage gaps -- see Methodology).
bwd's coverage came from Phase 36 (new GEMM_N/K-tail mechanism) plus new bf16/fp32 config
files; wrw's came entirely from new fp16/fp32 config files for Phase 35's already-generic
M/N/K-tail mechanism (no new codegen).

## What changed this session

1. **bwd GEMM_N-tail + GEMM_K-tail** (`docs/gfx1250_wmma_layout.md`'s Phase 36): bwd
   previously had M-tail only. bwd's B (weight) operand is TRANSPOSED (unlike fwd's),
   which inverts which axis is the "easy" per-lane EXEC-mask case vs. the "hard" new
   fine-grained per-dword mask case -- discovered only by reading the actual per-lane
   addressing, not by assuming fwd's N-tail pattern would port directly. New configs:
   `igemm_bwd_gtc_gfx1250_nhwc_{fp16,bf16,fp32}_{ntail,ktail,mnktail}.config` (fp16's
   `_k2x_ntail` additionally smoke-tests the multi-chunk masking path).
2. **fwd TDM + GEMM_M/N-tail combined** (Phase 37): TDM's own descriptor now covers M/N
   tail natively (`tensor_dim1` rebuilt relative to the block offset, mirroring how
   `tensor_dim0` already covers K), grounded in the CDNA5 ISA doc's TDM section and
   confirmed as a real hardware probe. New configs:
   `igemm_fwd_gtc_gfx1250_nhwc_{fp16,bf16,fp32}_tdm_mntail.config`.
3. **All remaining config-only gaps closed**: wrw's `_mntail`/`_ktail`/`_gsplit_mntail`/
   `_gsplit_ktail`/`_gsplit_mnktail` mechanisms (Phase 35, previously bf16-only) now also
   have fp16/fp32 configs; bwd's new N/K-tail mechanisms (previously fp16-only) now also
   have bf16/fp32 configs; fwd's new TDM+M/N-tail combination (previously fp16-only) now
   also has bf16/fp32 configs. All hardware-validated (`valid:y` against
   `naive_conv_{fwd,bwd,wrw}_nhwc`) before being committed -- see each config's own header
   comment and `docs/gfx1250_wmma_layout.md`'s Phase 35-37 for the validation batteries.

## Precision coverage

This corpus is fp32/fp16/bf16 only (no int8 or int4 entries at all) -- confirmed by
counting `MIOpenDriver conv`/`convbfp16`/`convfp16`/`convint8` lines directly in the three
`.txt` files (zero `convint8` lines in any of them). int8 gaps (fwd/bwd/wrw tail configs,
wrw's int8 split-K epilogue) remain real, separately-tracked gaps -- see the catalog below
-- but have **zero weight in this specific corpus**, so closing them would not move any of
the percentages above. They were intentionally left open this round for that reason (this
session's config-writing effort targeted the gaps this corpus actually shows).

## Effort tiers (used consistently across this doc)

- **Tier A** -- already works, no action needed.
- **Tier B** -- the underlying code mechanism exists and is precision-generic; only a new
  `.config` file plus hardware validation is needed (config-file gap, cheapest to close).
- **Tier C** -- needs combining or extending existing, independently-working mechanisms
  (driver integration, new address-math variant, etc.) -- moderate effort, no fundamentally
  new kernel design.
- **Tier D** -- needs a genuinely new mechanism with no existing in-tree precedent for that
  direction (though often a sibling direction's existing mechanism is a strong template),
  or is architecturally out of scope entirely.

## Full gap catalog (ranked by real-corpus weight, i.e. how many actual entries -- not just
distinct shapes -- fall into each; assumes layout resolved, since layout itself is a
declared non-goal, not a ranked gap)

| Rank | Gap | Distinct shapes | Corpus entries | Effort | Where to start |
|---|---|---|---|---|---|
| 1 | fwd K-tail for multi-tap convs | 2,936 | 2,936 | **D** | New software EXEC-mask K-tail for fwd (fwd currently only has TDM's 1x1-only K-tail) -- wrw's Phase 35 K-tail is the closest template, though check actual per-lane addressing before assuming it ports directly (see Phase 36's bwd surprise) |
| 2 | Depthwise (g==c), all directions | 754 | 754 | D | Architectural -- not gfx1250-specific, out of scope for MISA's igemm approach on any architecture; not recommended to pursue |
| - | Layout (NCHW/mixed), all directions | 87,313 (of the real, non-`--assume-nhwc` corpus) | ~94,335 | n/a | Explicit non-goal for MISA -- left to the caller (e.g. MIOpen's own solver-wrapping), see "Headline finding" above |
| - | int8 tail/gsplit configs (fwd/bwd/wrw) | not in this corpus | 0 | B (tail configs) / D (wrw int8 gsplit specifically) | Config-only for most cases; wrw's int8 split-K epilogue was already flagged (Phase 17) as needing real new work, not just a config; bwd's int8 tail is Tier C (the elem_per_dword=4 masking case is wholly untested for bwd) |

**Recommendation**: fwd's multi-tap K-tail (rank 1) is now the *only* remaining GEMM-shape
gap with real corpus weight. It is a genuine Tier D (new mechanism) item, unlike everything
closed this session -- worth scoping as its own follow-on rather than assuming it's a quick
config addition.

## Reproduction

```
# As requested (real layout) -- shows the layout gap as it actually is:
python3 script/classify_gfx1250_coverage.py \
    convasmimplicitgemmgtcdynamic_fwd.txt \
    convasmimplicitgemmgtcdynamic_bwd.txt \
    convasmimplicitgemmgtcdynamic_wrw.txt \
    --csv-out /tmp/coverage_full.csv --md-out /tmp/coverage_report.md

# Assuming layout is resolved upstream -- isolates GEMM-shape/mechanism coverage:
python3 script/classify_gfx1250_coverage.py \
    convasmimplicitgemmgtcdynamic_fwd.txt \
    convasmimplicitgemmgtcdynamic_bwd.txt \
    convasmimplicitgemmgtcdynamic_wrw.txt \
    --assume-nhwc \
    --csv-out /tmp/coverage_nhwc.csv --md-out /tmp/coverage_nhwc_report.md
```

The three corpus `.txt` files are gitignored (large, machine-specific) and not committed --
this doc's numbers were generated from the versions present at analysis time; a rerun
against a different or extended corpus will naturally produce different counts. The CSV
output (one row per distinct shape, with its category, human-readable reason, and the
original reference solver/time) is the right starting point for picking concrete shapes to
validate once a gap is closed -- it is also gitignored (large) and was not committed;
regenerate on demand.

## Notes on methodology / caveats

- Classification is **pure GEMM-dimension arithmetic** against known, already-hardware-
  validated mechanism support -- it does not itself run anything on hardware. A shape
  marked "supported" is asserting the same GEMM-shape/mechanism combination has been
  validated elsewhere (this session's Phase 35-37 work, or earlier phases in
  `docs/gfx1250_wmma_layout.md`), not that this *exact* shape has been individually tested.
- `--assume-nhwc` models "every shape gets transposed to NHWC by the caller" -- it is a
  hypothetical for isolating GEMM-shape coverage, not a claim about MISA's actual behavior
  (MISA does not transpose layouts itself; see "Headline finding" above).
- "Exact tile fit" checks `gemm_m % 128 == 0 or gemm_m % 64 == 0` (and symmetrically for
  `gemm_n`), matching that MISA's combined base configs search both 128x128 and 64x64
  tiles. `gemm_k` fit is checked against the precision's base WMMA K-instruction width (32
  for fp16/bf16, 4 for fp32) -- `_k2x`/`_k4x`-style wider K-per-block variants are a
  performance lever, not a coverage one (they only ever make the exact-fit divisibility
  requirement *stricter*), so they're intentionally not modeled here.
- Depthwise detection uses `g == c` (each group has exactly one input channel) -- the same
  check used throughout this session's own investigation.
- "Degenerate" shapes (computed `ho <= 0` or `wo <= 0`) were cross-checked against MISA's
  own `conv_out_size` formula (`driver/common.h`) -- the same formula the actual driver
  uses -- not a separate approximation.
