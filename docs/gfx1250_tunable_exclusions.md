# gfx1250 WMMA Tunable Combination Exclusions — Catalog

**Purpose.** A single, mechanically-derived list of every tunable *combination*
gfx1250's WMMA pipeline currently refuses to build, and why — genuine hardware/
architectural incompatibility, an untested-but-plausible gap, a resource limit,
or a bug that's since been fixed. Written per `docs/gfx1250_tuning_refactor_plan.md`
Phase 5/5b, and per direct request: previously this reasoning was scattered
across three independent, drifting copies (Python asserts, C++ `tunable_is_valid`,
and `script/generate_all_configs.py`'s own hand-rolled `is_valid()`). As of this
pass, all three have collapsed into **one** source of truth for build-time
legality — this document is a snapshot of what that one source currently
rejects, not a fourth copy of the logic itself.

## How this list was produced (and how to regenerate it)

`script/generate_all_configs.py`'s `is_valid()` no longer hand-encodes rules. For
every candidate tunable combination it:

1. Constructs the real `igemm_gtc_tunable_parameter_t` (catches Category A
   cross-tunable asserts — `igemm_base.py`).
2. Constructs the real direction-specific WMMA generator class,
   `igemm_{fwd,bwd,wrw}_gtc_wmma_nhwc_t` (catches generator-level asserts —
   e.g. `row_repeat_a/b`, `col_split_b` interactions).
3. Emits the **full kernel body** in-memory (catches emission-time-only
   asserts never reached by construction alone — e.g. `wmma_main_loop.py`'s
   "interleave requires num_k_substeps>1").
4. Actually **assembles** that one kernel with the real ROCm toolchain
   (`clang++ -x assembler -mcpu=gfx1250`) — catches genuine assembler-only
   failures neither Python layer can see (e.g. "register index is out of
   range").

Any `AssertionError` or non-zero assembler exit at any of these four steps is
a rejection; anything else is accepted. This is exactly what `igemm_codegen.py`
itself does before shipping a kernel — "rejected here" and "would fail a real
build" are the same statement, and every combination this document calls
"excluded" is being excluded by the same mechanism, not a second copy of it.

Regenerate this table's *counts* at any time: the sweep below is the one
`script/generate_all_configs.py` itself performs across its `FLAGS` list × every
`BASE_SECTIONS` tile shape (11 boolean/enum flags × up to 27 base tile/precision
combinations = 55,296 raw candidates as of this writing; int8/fp8 intentionally
excluded from this sweep — zero priority per current direction). Of those,
**4,080** pass all four steps.

## Category A/B — Cross-tunable exclusions found by the real constructor

These are genuine, currently-real rejections as of this pass — every one is a
live `assert` (or, for the last row, a live assembler failure) that fires
whenever the combination is attempted, independent of tile shape or precision
unless noted. Counts are how many of the 55,296 raw candidates hit each one
(a single candidate can be counted by more than one rule if several would
independently reject it — `is_valid()` stops at the first).

| Rejection (verbatim assert text or failure) | Hit count | Where | Disposition |
|---|---:|---|---|
| `tdm_global_load` and `main_loop_interleave` are mutually exclusive for now | 13,824 | `igemm_base.py` | Untested combination, not a proven incompatibility — TDM's own async load already overlaps with compute differently; revisit if TDM gains its own interleave story |
| `local_prefetch_num=2` and `main_loop_interleave` are mutually exclusive for now | 6,912 | `igemm_base.py` | Untested combination |
| `wmma_m_tail` and `gemm_k_global_split` are mutually exclusive for now (fwd/bwd only — wrw's M-tail *does* mask the atomic branch, Phase 35) | 6,720 | `igemm_base.py` Phase 25 | Real gap: fwd/bwd's atomic epilogue has no M-tail masking implemented |
| `wmma_n_tail` and `gemm_k_global_split` are mutually exclusive for now (fwd/bwd only) | 3,360 | `igemm_base.py` Phase 26b | Real gap: fwd/bwd's atomic epilogue has no N-tail masking implemented |
| `local_prefetch_num=2` requires `num_k_substeps>1` — nothing to prefetch into otherwise | 3,312 | `wmma_main_loop.py` | Structural: only reachable at all for tiles/precisions with a multi-substep K loop; not a flaw, a real prerequisite |
| `tdm_global_load` is not yet supported together with `local_prefetch_num > 1` | 3,072 | generator `__init__` (all 3 directions) | Untested combination |
| `tdm_global_load` is not yet combined with `wmma_m_tail`/`n_tail`/`k_tail` for wrw | 2,304 | `igemm_wrw_gtc_wmma_nhwc.py` | TDM's own hardware OOB already replaces `wmma_k_tail` for wrw; M/N-tail-via-TDM is a separate, not-yet-attempted extension |
| `saddr_global_load` is not yet combined with `gemm_k_global_split`/`wrw_streamk` for wrw | 1,536 | `igemm_wrw_gtc_wmma_nhwc.py` | Untested — not audited against the shard/persistent-loop base-pointer offset |
| `saddr_global_load` and `tdm_global_load` are mutually exclusive | 1,392 | all 3 generators | Both are alternatives to the same default 64-bit VADDR-pair path — mutually exclusive by design, not a gap |
| `main_loop_interleave` requires `lds_double_buffer=1` | 1,312 | all 3 generators | **Genuine hardware constraint, confirmed on real hardware**: single-buffered interleaving races across waves |
| `gemm_k_global_split`/`wrw_streamk` is not yet combined with `main_loop_interleave` for wrw | 1,024 | `igemm_wrw_gtc_wmma_nhwc.py` | Untested combination |
| `tdm_global_load` is not yet supported together with `row_repeat_a > 1` | 960 | fwd/bwd generators | Untested — TDM's flat load assumes one row per thread |
| `wmma_n_tail` requires `row_repeat_b == 1` | 960 | `igemm_base.py` Phase 26b | Structural: rows 1+ have no per-row tail flag of their own |
| `main_loop_interleave` is not yet supported together with `row_repeat > 1` | 768 | all 3 generators | Untested combination |
| `interleave requires num_k_substeps>1` (emission-time) | 640 | `wmma_main_loop.py` | Same structural prerequisite as the `local_prefetch_num=2` row above, hit independently for `main_loop_interleave` |
| `saddr_global_load` is not yet supported together with `row_repeat_a/b > 1` | 576 | all 3 generators | Untested — asymmetric (128x64/64x128) tiles only |
| `saddr_global_load` is not yet combined with `gemm_k_global_split` for fwd | 384 | `igemm_fwd_gtc_wmma_nhwc.py` | Untested — not audited against the shard base-pointer offset |
| `saddr_global_load` is not yet combined with `main_loop_interleave` for bwd | 384 | `igemm_bwd_gtc_wmma_nhwc.py` | Untested combination |
| epilogue LDS exceeds the 64KB/workgroup hardware limit for `epilogue_lds_pad` on a 128x128 tile | 288 | `igemm_fwd_gtc_wmma_nhwc.py` `get_kernel_code()` | **Genuine hardware resource limit** (ISA-enforced 64KB/workgroup LDS cap) — `epilogue_lds_pad` stays usable on smaller tiles |
| `gemm_k_global_split` is not yet combined with `tdm_global_load` for fwd | 288 | `igemm_fwd_gtc_wmma_nhwc.py` | Real gap: TDM's `tensor_dim0` setup reads the un-sharded `s_gemm_k` directly |
| `saddr_global_load` is not yet combined with `gemm_k_global_split` for bwd | 288 | `igemm_bwd_gtc_wmma_nhwc.py` | Untested combination |
| `saddr_global_load` is not yet combined with `main_loop_interleave` for fwd | 256 | `igemm_fwd_gtc_wmma_nhwc.py` | Untested combination |
| `gemm_k_global_split` is not yet combined with `main_loop_interleave` for bwd | 192 | `igemm_bwd_gtc_wmma_nhwc.py` | Untested combination |
| `gemm_k_global_split` is not yet combined with `tdm_global_load` for bwd | 144 | `igemm_bwd_gtc_wmma_nhwc.py` | Real gap: `s_tdm_k_remain` init would need `s_knum`, not the un-sharded `s_gemm_k` |
| `gemm_k_global_split` is not yet combined with `main_loop_interleave` for fwd | 128 | `igemm_fwd_gtc_wmma_nhwc.py` | Untested combination |
| **fwd, tile 128x64 (asymmetric) + `wmma_n_tail=1`, every precision — real assembler failure: "register index is out of range"** | 192 | assembler (all other flags irrelevant once this base combination is present) | **New finding, this pass.** Passes every Python-level assert (`row_repeat_b==1` holds for 128x64), so this was invisible to a constructor-only check — only caught by actually assembling. Not yet root-caused; excluded mechanically rather than shipped broken. Candidate for a future root-cause pass (likely: some N-tail-masking register formula assumes the *symmetric*-tile addressing fwd's 128x128/64x64 base sections use, and doesn't generalize to 128x64's different `block_size`/column layout) |

Read the "Untested combination" dispositions literally: none of these are
proven bugs, they are asserts put in place when a feature was built and only
ever validated in isolation. Per `gpt_astra_tuning.md`'s own dominance-study
discipline (Phase 6 of the tuning refactor plan), lifting any one of them is
real, hardware-validation-required follow-up work, not a documentation fix —
listed here so the *next* person doesn't have to re-derive "is this asserted
because it's broken, or because nobody tried it yet" from scratch.

## Category A — resolved this pass (previously excluded, now fixed)

Three items previously excluded are **no longer excluded** — root-caused and
fixed under `docs/gfx1250_tuning_refactor_plan.md` Phase 2, hardware-validated:

- **`atomic_cascade`** — deleted entirely (was hard-blocked, confirmed hangs
  real hardware; the field, `atomic_th`, and the dead `th:` branches are gone,
  not just asserted off). See `docs/gfx1250_misa_investigation_report.md` COR-004.
- **bwd `lds_double_buffer && lds_row_pad` (R7)** — root-caused (B's padded
  store offset never picked up the double-buffer toggle) and fixed; no longer
  excluded. See `docs/gfx1250_bwd_dbuf_ldsrp_nan.md`.
- **fwd `saddr_global_load` + `wmma_n_tail`** — root-caused (`v_flag_b` was
  never computed on the saddr/async B-address branch) and fixed; no longer
  excluded. See `docs/gfx1250_optimization_backlog.md`.

## Category C — a "wrw tail without split-K" restriction that turned out to be a heuristic, not a rule

`script/generate_all_configs.py` previously hand-excluded `wmma_m_tail`/
`wmma_n_tail` for wrw whenever `gemm_k_global_split=0` ("wrw's M/N-tail was only
ever validated under split-K"). No Python assert and no C++ `tunable_is_valid`
check anywhere actually requires this — confirmed by grep and by the real
constructor accepting the combination. It is **no longer excluded**: the
combinatorial sweep above now includes wrw tail-without-split-K sections, and
this session's full hardware regression (every per-tile config, two shapes,
zero `valid:n`) exercised them with no failures. Recorded here so it isn't
mistaken for a still-open gap.

## Category D — runtime applicability (not a build-time tunable exclusion)

`wmma_m_tail`/`wmma_n_tail`/`wmma_k_tail` themselves are not "excluded"
combinations at all — every direction/precision/tile builds both the
tail-enabled and tail-disabled variant, and the **driver** (C++
`tunable_is_valid` in `driver/igemm_{fwd,bwd,wrw}_gtc_driver.h`) picks whichever
variant actually fits a given runtime shape at dispatch time. This is a
different kind of exclusion — "does this compiled kernel accept this shape",
not "can this tunable combination be built at all" — and is out of this
document's scope; see `gpt_astra_tuning.md` §5 and
`gfx1250_tuning_refactor_plan.md`'s Category D note for why the two are kept
conceptually separate.

## Legacy/out-of-scope fields (Category F, unrelated to WMMA)

`tensor_a_pass_through`, `tensor_b_pass_through`, `multihead`,
`allow_lds_reorder`, `precache_soffset`, `source_access_order`,
`gemm_{m,n,k}_unmerge_cluster`, `vector_store`, `merge_e`, `vector_c` are read
only by the legacy MAC/DLOPS/XDLOPS generators (gfx908/90a/940/950), never by
any `fma_type == WMMA` code path — see
`gfx1250_tuning_refactor_plan.md`'s Category F table for the full disposition
of each. Not part of this catalog; listed there, not repeated here.
