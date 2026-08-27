# gfx1250 WMMA coverage vs. a real MISA-solver-won shape corpus (2026-08-27)

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
support (from `docs/gfx1250_wmma_layout.md`'s Phase 5-35 history) via
`script/classify_gfx1250_coverage.py`.

## Headline finding: layout, not GEMM-shape, is the dominant gap

| Direction | Total distinct shapes | Non-NHWC layout | NHWC-only (comparable) |
|---|---|---|---|
| fwd | 17,772 | 17,675 (99.5%) | 97 |
| bwd | 18,750 | 18,376 (98.0%) | 374 |
| wrw | 51,519 | 51,262 (99.5%) | 257 |

Over 99% of this real-world corpus uses **NCHW** (the vast majority, no `--in_layout`/
`--fil_layout`/`--out_layout` flags at all -- MIOpenDriver's own default is NCHW) or a
**mixed NHWC-input/NCHW-filter-and-output** layout (a smaller slice, ~1-3%). MISA's gfx1250
WMMA kernels are NHWC-only by construction (every `tensor_layout = 'nhwc'` in every
gfx1250 config) -- `conv_driver.cpp` hard-asserts the requested layout matches the
compiled tunable's layout (`assert(in_layout=="NCHW" && tensor_layout=="nchw") || ...`,
`driver/conv_driver.cpp:744`) rather than converting, so **none** of these non-NHWC shapes
can run on the gfx1250 WMMA path today, regardless of their GEMM dimensions.

**This is not a fundamental blocker.** MISA already has working NCHW<->NHWC transpose
kernels (`driver/gpu_nchw_nhwc_transpose.h`: `gpu_nchw2nhwc`/`gpu_nhwc2nchw`, built on
`driver/gpu_batched_transpose.h`) -- they're simply never invoked by the WMMA dispatch path.
Closing this gap means wrapping a layout-mismatched WMMA request with a pre-transpose
(input, and filter if needed) and a post-transpose (output back to the requested layout),
similar in spirit to the existing workspace/reduction-kernel two-dispatch pattern from
`docs/gfx1250_wmma_layout.md`'s Phase 35. This is **driver/host-code integration work, not
new kernel math** -- classified as effort Tier C below. It is, by a wide margin, the single
highest-leverage gap in this entire analysis: closing it alone would make the other
GEMM-shape gaps below relevant to ~50-100x more of this real corpus.

## Per-direction breakdown, NHWC-only subset (the fair, apples-to-apples comparison)

Restricting to the shapes that actually request NHWC (the only layout gfx1250 WMMA
supports at all today) -- this is the set worth reading GEMM-shape gaps out of:

| Direction | NHWC shapes | Supported today | Gaps | Degenerate (not real convs) |
|---|---|---|---|---|
| fwd | 97 | 62 (63.9%) | 35 (36.1%) | 0 |
| bwd | 374 | 257 (68.7%) | 116 (31.0%) | 1 (0.3%) |
| **wrw** | 257 | **255 (99.2%)** | 0 (0.0%) | 2 (0.8%) |

**wrw is essentially fully covered** for this corpus's NHWC shapes -- a direct, concrete
validation of this session's Phase 35 work (`docs/gfx1250_wmma_layout.md`): the M/N/K-tail
relief and reduction-kernel epilogue landed there closed almost the entire real-shape gap
for that direction. fwd and bwd both still have real, characterized gaps (below).

### fwd (97 NHWC shapes)

| Category | Count | Effort |
|---|---|---|
| Supported (M/N-tail, `_mtail`/`_ntail`/`_mntail`) | 35 | - |
| Supported (exact tile fit) | 27 | - |
| **GAP: TDM K-tail + M/N-tail never combined** | 28 | C |
| **GAP: K-tail for multi-tap (non-1x1) convs** | 7 | D |

fwd's `_tdm` config (Phase 31's GEMM_K-tail via TDM hardware OOB) is 1x1-conv-only and has
never been combined with `wmma_m_tail`/`wmma_n_tail` into one config -- each mechanism
works independently and is understood to be additive in principle (per Phase 35's own
"no interaction" experience combining M/N/K-tail for wrw), just not yet built/validated
together for fwd. This is the largest fwd gap (28 of 35 gap shapes). The smaller K-tail-
for-multi-tap gap (7 shapes, all 3x3 filters) has no existing mechanism at all -- fwd's
*only* K-tail path is TDM, which is 1x1-only; a genuinely new K-tail (software EXEC-mask,
the same class of mechanism Phase 35 built for wrw from scratch) would be needed for
multi-tap fwd K-tail.

### bwd (374 NHWC shapes)

| Category | Count | Effort |
|---|---|---|
| Supported (exact tile fit) | 172 | - |
| **GAP: no N-tail or K-tail mechanism at all** | 113 | D |
| Supported (M-tail, `_mtail`) | 85 | - |
| **GAP: depthwise (g==c)** | 3 | D (architectural, out of scope) |
| Degenerate (zero valid output positions) | 1 | n/a |

bwd's single biggest gap: it never got N-tail or K-tail at all (only M-tail, Phase 26a) --
unlike fwd (which has both) and wrw (which now has all three, Phase 35). Porting Phase
35's wrw K-tail design (and fwd's existing N-tail design) to bwd is the natural next step
and is well-precedented -- two working reference implementations already exist in this
codebase to copy the pattern from.

### wrw (257 NHWC shapes)

| Category | Count | Effort |
|---|---|---|
| Supported (M/N/K-tail, Phase 35) | 163 | - |
| Supported (exact tile fit) | 92 | - |
| Degenerate (zero valid output positions) | 2 | n/a |

No gaps found in this corpus. The 2 degenerate shapes (`H=W=1, y=x=3, p=0` -- a 3x3 filter
that cannot fit a 1x1 input with no padding, `ho=wo=-1`) are not real convolutions; MISA's
own naive-conv reference agrees they produce no valid output, consistent with these being
MIOpen solver-robustness corner cases rather than throughput-relevant shapes.

## Precision coverage

This corpus is fp32/fp16/bf16 only (no int8 or int4 entries at all) -- so it exercises
none of the previously-documented int8-specific gaps (`docs/gfx1250_wmma_layout.md`'s
Phase 17 note that wrw's split-K/atomic path has no int8 support at all, and this
session's own Phase 35 tail-relief configs being bf16-only so far). Those remain real,
separately-tracked gaps (see "Not exercised by this corpus" below) -- just not weighted by
anything in this particular dataset.

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
distinct shapes -- fall into each)

| Rank | Gap | Distinct shapes | Corpus entries | Effort | Where to start |
|---|---|---|---|---|---|
| 1 | Non-NHWC layout (all directions) | 87,313 | ~94,335 | **C** | Wire `gpu_nchw2nhwc`/`gpu_nhwc2nchw` (`driver/gpu_nchw_nhwc_transpose.h`) into `conv_driver.cpp`'s WMMA dispatch path as pre/post transpose steps when the requested layout doesn't match `tensor_layout`, mirroring Phase 35's two-dispatch (main kernel + reduction kernel) sequencing pattern |
| 2 | bwd has no N-tail or K-tail | 113 | 113 | D | Port `docs/gfx1250_wmma_layout.md`'s Phase 26b (fwd N-tail) and Phase 35 (wrw K-tail) designs to bwd -- both are documented, working references |
| 3 | fwd TDM K-tail + M/N-tail uncombined | 28 | 28 | C | Combine two already-working, independent mechanisms into one config + validate (no new address math expected, unlike rank 4) |
| 4 | fwd K-tail for multi-tap convs | 7 | 7 | D | New software EXEC-mask K-tail for fwd (fwd currently only has TDM's 1x1-only K-tail) -- Phase 35's wrw K-tail design is the closest template |
| 5 | Depthwise (g==c), all directions | 3 (this corpus; likely more once NHWC gap is closed) | 3 | D | Architectural -- not gfx1250-specific, out of scope for MISA's igemm approach on any architecture; not recommended to pursue |
| - | int8 tail/gsplit configs (fwd/bwd/wrw) | not in this corpus | 0 | B (tail configs) / D (wrw int8 gsplit specifically) | Config-only for most cases; wrw's int8 split-K epilogue was already flagged (Phase 17) as needing real new work, not just a config |

**Recommendation, in priority order**: (1) the NHWC/NCHW transpose-wrapping gap dwarfs
everything else in real-corpus weight by roughly 3 orders of magnitude -- closing it is the
single highest-leverage change identifiable from this data, independent of any further
GEMM-shape tail work; (2) bwd N/K-tail, since bwd is now the direction furthest behind
fwd/wrw in mechanism completeness and two working reference designs already exist to copy;
(3) fwd's TDM+M/N-tail combination, cheaper than it looks since both halves already work
independently.

## Reproduction

```
python3 script/classify_gfx1250_coverage.py \
    convasmimplicitgemmgtcdynamic_fwd.txt \
    convasmimplicitgemmgtcdynamic_bwd.txt \
    convasmimplicitgemmgtcdynamic_wrw.txt \
    --csv-out /tmp/coverage_full.csv --md-out /tmp/coverage_report.md
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
  validated elsewhere (this session's Phase 35 work, or earlier phases in
  `docs/gfx1250_wmma_layout.md`), not that this *exact* shape has been individually tested.
- "Exact tile fit" checks `gemm_m % 128 == 0 or gemm_m % 64 == 0` (and symmetrically for
  `gemm_n`), matching that MISA's combined base configs search both 128x128 and 64x64
  tiles. `gemm_k` fit is checked against the precision's base WMMA K-instruction width (32
  for fp16/bf16, 4 for fp32) -- `_k2x`-style wider K-per-block variants are a performance
  lever, not a coverage one (they only ever make the exact-fit divisibility requirement
  *stricter*), so they're intentionally not modeled here.
- Depthwise detection uses `g == c` (each group has exactly one input channel) -- the same
  check used throughout this session's own investigation.
- "Degenerate" shapes (computed `ho <= 0` or `wo <= 0`) were cross-checked against MISA's
  own `conv_out_size` formula (`driver/common.h`) -- the same formula the actual driver
  uses -- not a separate approximation.
