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
support (from `docs/gfx1250_wmma_layout.md`'s Phase 5-38 history) via
`script/classify_gfx1250_coverage.py`.

**This revision** reflects three rounds of follow-on work since the original analysis:
(1) bwd GEMM_N/K-tail (Phase 36) and fwd's TDM+GEMM_M/N-tail combination (Phase 37) closed
the two biggest GEMM-shape mechanism gaps the original analysis found; (2) every
*config-only* gap (a mechanism that already existed in code but only had `.config` files
for one precision) was closed with new, hardware-validated config files across
fp16/bf16/fp32 for fwd/bwd/wrw; (3) fwd's GEMM_K tail for multi-tap convolutions (Phase 38)
-- the one remaining real mechanism gap at the time -- was closed. A classifier bug
(depthwise detection) was also found and fixed along the way -- see below. Layout
conversion (NCHW<->NHWC) remains an explicit non-goal for MISA -- see "Headline finding"
below.

**Update (2026-08-27, later)**: a second classifier blind spot was found and fixed --
fwd/bwd's non-atomic vectorized-4-wide-store epilogue requires `gemm_n % 4 == 0` whenever
`wmma_n_tail` is active (confirmed unbuildable on real hardware, not just inferred), and
the classifier previously didn't model this sub-constraint at all, silently over-counting
coverage for every N-tail-needing shape whose `gemm_n` isn't a multiple of 4. This is a
real, currently-unbuildable gap (not a config-only one -- closing it needs a new,
finer-grained epilogue masking primitive), now tracked as `gap_n_mod4_fwd`/
`gap_n_mod4_bwd` in `script/classify_gfx1250_coverage.py` and in
`docs/gfx1250_optimization_backlog.md`. This revises the "zero remaining GEMM-shape gaps"
claim below -- see the updated table.

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

This is the relevant view for judging MISA's own GEMM-shape coverage:

| Direction | Total entries | Supported today | GAP: `gemm_n%4!=0` (new) | Degenerate (not a real conv) |
|---|---|---|---|---|
| fwd | 17,772 | 17,099 (96.21%) | 630 (3.54%) | 43 (0.24%) |
| bwd | 21,055 | 20,514 (97.43%) | 336 (1.60%) | 205 (0.97%) |
| wrw | 56,239 | 55,915 (99.42%) | n/a (scalar atomic epilogue, no `%4` constraint) | 324 (0.58%) |
| **Overall** | **95,066** | **93,528 (98.38%)** | **966 (1.02%)** | **572 (0.60%)** |

**One real GEMM-shape/mechanism gap remains, found after the original "zero gaps" claim
below was written**: fwd/bwd's non-atomic epilogue needs `gemm_n % 4 == 0` whenever
`wmma_n_tail` is active (a real hardware constraint, not a classifier quirk -- confirmed
unbuildable via direct hardware testing), and the classifier didn't model it until this
pass. wrw is unaffected (its atomic epilogue is scalar-per-element, no vectorized-store
grouping at all). See "A second classifier blind spot" below for the fix and its effort
tier. Everything else below (the "3.09% -> 0%" mechanism-gap-closing narrative) is
unchanged and still accurate for the mechanisms it covers -- this is an *additional*,
previously-unmodeled constraint on top of that work, not a regression in it. Three
follow-on rounds this session had, at the time, closed every OTHER known GEMM-shape
mechanism gap:

1. **bwd GEMM_N-tail + GEMM_K-tail** (Phase 36) -- bwd previously had M-tail only.
2. **fwd TDM + GEMM_M/N-tail combined** (Phase 37) -- these existed independently but had
   never been combined into one buildable config.
3. **Every config-only gap closed** -- new fp16/bf16/fp32 config files for mechanisms that
   already existed in code (wrw's M/N/K-tail was previously bf16-only; bwd's new N/K-tail
   and fwd's new TDM+M/N-tail combo were initially fp16-only).
4. **fwd GEMM_K-tail for multi-tap convolutions** (Phase 38) -- fwd's only K-tail mechanism
   was TDM's hardware OOB, which is 1x1-only by construction; this added a genuinely new
   software (non-TDM) K-tail mechanism for multi-tap convs.

See `docs/gfx1250_wmma_layout.md`'s Phase 35-38 for full designs, bugs found, and hardware
validation batteries for each.

### A second classifier blind spot: `gemm_n % 4 == 0` for fwd/bwd's vectorized-store epilogue

fwd/bwd's non-atomic epilogue (`coalescing_store_wmma.py`'s LDS-reshuffle path) stores in
vectorized 4-wide chunks. When `wmma_n_tail` is active, the EXEC-mask guard only checks a
group's FIRST column -- a 4-column group straddling a non-multiple-of-4 `gemm_n` silently
writes the out-of-range tail columns too. Confirmed unbuildable on real hardware
(`fwd fp32 n=1,c=1,k=1,H=W=1760` and `n=256,c=3,k=3,H=W=32`, both `gemm_n=k` in `{1,3}` --
every kernel in the master config reports "not applicable"). The classifier never modeled
this sub-constraint, so every N-tail-needing shape was reported "supported" regardless of
`gemm_n mod 4` -- not just the tiny `gemm_n` cases above: the fix also catches much larger
non-multiple-of-4 values already present in this corpus (e.g. `gemm_n` = 486, 510, 1001 for
bwd). wrw's atomic epilogue is scalar-per-element (no vectorized grouping), so it's
unaffected. Fixed in `script/classify_gfx1250_coverage.py` (`gap_n_mod4_fwd`/
`gap_n_mod4_bwd` categories); tracked as an open Tier 2 item in
`docs/gfx1250_optimization_backlog.md` (effort tier C -- needs a new, finer-grained
per-element epilogue masking primitive, not just a config file).

### A classifier bug found and fixed along the way: "depthwise" was over-triggering

The original analysis flagged 754 shapes across all three directions as "depthwise (g==c),
architecturally out of scope." Checking directly: **every single one of those 754 shapes
has `g=1`** (zero have `g>1`). A `g=1, c=1` shape technically satisfies the literal `g==c`
check, but with only one group it's an entirely ordinary, non-grouped convolution -- the
architectural concern behind excluding real depthwise (many tiny independent per-group
GEMMs, a fundamentally different pattern from what an igemm approach is good at) only
applies once `g` is actually large. The check was fixed to require `g == c and g > 1`;
after the fix, the "depthwise" gap category disappears entirely from this corpus (0 shapes
in all three directions) -- every one of the 754 shapes was correctly ordinary and mostly
already covered by existing mechanisms. Whether *genuine* multi-group depthwise was ever
exercised on MISA's gfx950 track could not be confirmed either way -- no corpus examined
this session (including an earlier, separate 112-shape trace referenced in
`docs/gfx1250_vendor_benchmark_vs_miopen.md`, which used the same unguarded check and may
have the identical bug) contains a real `g>1` depthwise shape to test against. MISA's group
handling is generic (`c/g`, `k/g`), so a real depthwise shape wouldn't hit a hard assert --
it would just produce an extremely small `gemm_n` (or `gemm_m`), more a severe efficiency
problem (wasting nearly the whole tile) than a correctness one.

### What's left in "out of scope / degenerate"

After the depthwise fix, only one bucket remains across all three directions: **degenerate
zero-output shapes** -- `ho<=0` or `wo<=0` (computed via MISA's own `conv_out_size`
formula), meaning the filter can't fit the input even once (e.g. a 3x3 filter on a 1x1
input with no padding). These aren't real workloads -- they look like MIOpen
solver-robustness/bounds-checking test cases, not throughput-relevant shapes. ~572 shapes
total (0.60% of the corpus), spread fairly evenly across fwd/bwd/wrw.

## Precision coverage

This corpus is fp32/fp16/bf16 only (no int8 or int4 entries at all) -- confirmed by
counting `MIOpenDriver conv`/`convbfp16`/`convfp16`/`convint8` lines directly in the three
`.txt` files (zero `convint8` lines in any of them). int8 gaps (fwd/bwd/wrw tail configs,
wrw's int8 split-K epilogue) remain real, separately-tracked gaps -- see the catalog below
-- but have **zero weight in this specific corpus**, so closing them would not move any of
the percentages above. They were intentionally left open (this session's work targeted the
gaps this corpus actually shows).

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

## Full gap catalog (nothing with real corpus weight remains)

| Rank | Gap | Distinct shapes | Corpus entries | Effort | Where to start |
|---|---|---|---|---|---|
| - | Layout (NCHW/mixed), all directions | 87,313 (of the real, non-`--assume-nhwc` corpus) | ~94,335 | n/a | Explicit non-goal for MISA -- left to the caller (e.g. MIOpen's own solver-wrapping), see "Headline finding" above |
| - | int8 tail/gsplit/TDM/K-tail configs (fwd/bwd/wrw) | not in this corpus | 0 | B (config only, most cases) / D (wrw int8 split-K gsplit specifically) / C (bwd int8 tail -- the `elem_per_dword=4` masking case is wholly untested) | Config-only for most cases; wrw's int8 split-K epilogue was already flagged (Phase 17) as needing real new work, not just a config |

Every GEMM-shape/mechanism gap this analysis originally found (fwd K-tail multi-tap gap,
bwd's missing N/K-tail, every bf16-only/fp16-only config gap) has been closed. The only
entries left in the catalog have zero weight in this corpus (int8) or are an explicit
non-goal (layout).

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
  validated elsewhere (this session's Phase 35-38 work, or earlier phases in
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
- Depthwise detection uses `g == c and g > 1` (each group has exactly one input channel,
  AND there is more than one group -- see the classifier-bug note above for why the `g > 1`
  qualifier is load-bearing, not defensive).
- "Degenerate" shapes (computed `ho <= 0` or `wo <= 0`) were cross-checked against MISA's
  own `conv_out_size` formula (`driver/common.h`) -- the same formula the actual driver
  uses -- not a separate approximation.
