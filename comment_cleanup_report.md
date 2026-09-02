# Comment Cleanup Report — MISA/iGEMMgen Codebase

**Date:** 2026-09-01  
**Scope:** All `.py` files in `python/` (codegen, igemm, operations, root) and all `.h`/`.cpp`/`.hpp` files in `driver/`  
**Excluded:** `docs/` folder, all `.md` files, MIT license headers (boilerplate lines 1–24 in every file)

---

## Summary

| Metric | Count |
|--------|-------|
| Files analyzed | 67 source files |
| Files edited | 28 files |
| Comments removed | ~600+ comment lines |
| Comments compressed | ~80 comment lines condensed |
| Comments preserved | ~2,400+ comment lines |
| Overall removable % | ~18–20% |

**Verification:** All Python files pass `ast.parse()`. All module imports succeed (`from python import *`, all WMMA generators, all operations). All C++ files readable. No commented-out code, TODO/FIXME/HACK/WARNING, design rationale, or hardware-specific comments were removed.

---

## Per-File Report

### python/codegen/

#### `python/codegen/amdgpu.py`
- **Removable:** ~17% (3 comments removed)
- **Removed:** `# in byte` (L224), `# default set to 8` (L592), `# hard code to 0` (L595) — obvious restatements
- **Preserved:** 15 comments — TODOs, hardware-specific gfx assembler notes for gfx1250/gfx1170+/gfx1030, SGPR/VGPR counting rules, pylint directive, LLVM doc URLs, code object v3 details
- **Compressed:** None

#### `python/codegen/compile.py`
- **Removable:** ~15% (2 comments removed)
- **Removed:** `# for multiple files` (L200, L226) — obvious from list check
- **Preserved:** 11 comments — commented-out debug prints, TODO about cov3 default, hipclang design rationale, hipconfig provenance note

#### `python/codegen/config_parser.py`
- **Removable:** ~33% (3 comments removed)
- **Removed:** 3 comments that restated visible code (L77, L218, L222)
- **Preserved:** 6 comments — value-list/value-range syntax docs, TODO recursive dict, commented-out `__main__` block

#### `python/codegen/mbb.py`
- **Removable:** 100% (1 comment removed)
- **Removed:** `# note: copy to here!` (L120) — restates `list()` constructor
- **Preserved:** None

#### `python/codegen/mc.py`
- **Removable:** ~10% (1 comment removed)
- **Removed:** `# manage the indent here` (L241) — restates assignment
- **Preserved:** 9 comments — WARNING about debug flags breaking correctness, TODOs, commented-out alternative debug filter, uniqueness/sort ordering notes

#### `python/codegen/instruction.py`
- **Removable:** 0%
- **Preserved:** 1 HACK marker — `# like macro_c_clear_t. this is a hack`

#### `python/codegen/macro.py`
- **Removable:** 0%
- **Preserved:** 3 comments — critical 3-step macro expansion ordering invariant (overwrite → emit → restore)

#### `python/codegen/scheduler.py`
- **Removable:** 0%
- **Preserved:** 40 comments — all commented-out debug prints, alternative algorithm implementations, MFMA hardware limit note, 2/3 ratio heuristic explanation, state machine enum docs, TODO

#### `python/codegen/symbol.py`, `python/codegen/__init__.py`
- **Removable:** N/A (0 non-license comments)

---

### python/igemm/

All 13 files in `python/igemm/` were analyzed. The vast majority of comments are **Preserve** — phase-tracked design rationale, hardware ISA references, register allocation notes, stride/offset formula documentation, and commented-out code from iterative development.

- **Overall removable:** ~5–8% (very low — these are the most critical generator files)
- **Preserved patterns:**
  - Phase-tracked design comments referencing ISA docs and hardware behavior (Phase N: ...)
  - Commented-out code from A/B implementation paths
  - TODO/FIXME markers for non-optimal paths
  - Inline stride/offset formula documentation
  - Hardware-specific register allocation notes (VGPR/AGPR/SGPR)
  - Magic division optimization comments

No edits were needed in this directory — the comments are overwhelmingly design-critical.

---

### python/operations/

#### `python/operations/utility.py`
- **Removable:** ~29% (2 removed)
- **Removed:** `# compute next power of 2` (L734), `# GetEpackLength` (L757)
- **Preserved:** hardware nop rationale, register aliasing hint, TODO xdlops, py3.5 compat note, pylint directive

#### `python/operations/shared_memory.py`
- **Removable:** ~17% (7 removed)
- **Removed:** `# swap and record indice`, `# stride` (2x), precision examples (2x), case labels (2x)
- **Preserved:** debug prints, commented-out asserts/code, design convention note, TODO bug, performance rationale

#### `python/operations/mfma_main_loop.py`
- **Removable:** ~15% (~170 removed/compressed)
- **Removed:** 141 `# Nth fma` labels, `# global load` comments
- **Compressed:** `# Label: start/finishing/end of fma body` → `# fma body start/end` (27 instances)
- **Preserved:** pass-through design, TODO items, hack note, interleave rationale, k_pack math

#### `python/operations/mfma.py`
- **Compressed:** `# in unit of passes, aka 4 cycle` → `# in passes (4 cycles each)`
- **Preserved:** large commented-out class block, TODO int8 accumulate type, column header

#### `python/operations/global_memory.py`
- **Removable:** ~20% (6 removed)
- **Removed:** `# if d0 is 1`, precision examples, `# start to emit init` (2x), `# write out is ignored`
- **Preserved:** precache pattern documentation, TODO items, L1 bypass note, CAS loop explanation

#### `python/operations/dotx_main_loop.py`
- **Removable:** ~30%
- **Removed:** 6× `# compute index for three matrice`, `# Nth fma` labels
- **Compressed:** `# Label: start/finishing/end of fma body` → `# fma body start/end`
- **Preserved:** dpp8 limitation, double-buffer switch comments

#### `python/operations/coalescing_store_dotx.py`
- **Removable:** ~25%
- **Removed:** 12× `# this is among different thread`, `# vdata, vaddr...`
- **Preserved:** CAUSION notes, TODO items, lgkmcnt limitation explanation

#### `python/operations/coalescing_store.py`
- **Removable:** ~30%
- **Removed:** `# ctrl_thread_mapping_t`, `# g_m1 always 1`, `# do some assert`, `# need use v_co_sub...` (4x), constructor arg docs (3x), `# vdata, vaddr...` (2x), `# m0/m1 accumulate` (2x)
- **Preserved:** large commented-out code blocks, debug prints, NOTE markers

#### `python/operations/coalescing_store_wmma.py`
- **Removable:** ~5%
- **Preserved:** Phase-based design rationale (Phases 23–59), hardware validation notes, VGPR-MSB banking, atomic cascade hang warnings

#### `python/operations/wmma_main_loop.py`
- **Removable:** ~5% (2 removed)
- **Removed:** `# functor` (L134), `# symbol type` (L167)
- **Preserved:** Phase-based documentation (Phases 1–54), async loads, TDM, k-sub-loop, chunk/compute interleaving, VGPR prefetch, setprio, VGPR-MSB banking

#### `python/operations/wmma_mapping.py`
- **Removable:** 0%
- **Preserved:** All tile-shape constraints, block_size==macro_tile invariant, VGPR budget calculations

#### `python/operations/wmma.py`
- **Removable:** 0%
- **Preserved:** Hardware verification notes, Phase 24/27 F16/BF16 accumulate variants, signed int8 modifier rationale, FP8 placeholder

#### `python/operations/dotx_mapping.py`
- **Removable:** ~40%
- **Removed:** All `# 256`/`# 128`/`# 64` size annotation comments on mapping table entries
- **Preserved:** DPP8 assumption, TODO, commented-out code, algorithm step labels

#### `python/operations/generic_tensor_transformation.py`
- **Removable:** ~15% (2 removed)
- **Removed:** `# upper dims, or visible dims`, `# return a list of flatterned item`
- **Preserved:** tensor transformation chain diagram, coordinate iteration example, TODO

#### `python/operations/main_loop_graph.py`
- **Removable:** ~40%
- **Removed:** 4× `# compute index for three matrice`, `# first barrier and waitcnt`, `# global load before loop`, `# last unroll k`
- **Compressed:** `# sst a/b double buffer switch` → `# double buffer switch` (2x)
- **Preserved:** dpp8 limitation, loop structure comments

#### `python/operations/fma_main_loop.py`
- **Removable:** ~35%
- **Removed:** All `# Nth fma` comments, `# start emit`
- **Compressed:** `# Label: start/finishing/end of fma body` → `# fma body start/end`
- **Preserved:** `#self._emit_empty_line()` (commented-out code)

#### Other operations files (conv.py, nop.py, thread_mapping.py, spatial_tiling.py, xdlops_mapping.py, dotx.py, __init__.py)
- **Removable:** 0–15% (minimal or no removals)
- **Preserved:** All direction labels, pylint directives, lanegroup layout explanations, TODO items, SALU spatial-slice math annotations

---

### python/ (root)

#### `python/codegen_driver.py`
- **Removable:** ~18% (13 removed)
- **Removed:** `# fwd/bwd/wrw gtc` section comments (4x), `# emit global macro` headers (3x), `# emit the kernel`, `# give a flag for current target direction` (3x), `# build host`
- **Preserved:** WARNING about never opening file in one thread/process, concurrency notes, all commented-out code (upsampling clear, emit_write_4d_strided, dotx macro selection, emit_v4r1_dynamic_*, mp.set_start_method, os.chmod, debug prints)

#### `python/host_driver.py`, `python/sequence_driver.py`, `python/perf_advisor.py`, `python/__init__.py`
- **Removable:** ~0% (all comments are pylint directives, design rationale, debug, or commented-out code)

---

### driver/

#### `driver/conv_driver.cpp`
- **Removable:** ~29% (12 removed)
- **Removed:** `// direction`/`// layout` section labels, `//skip`, CSV header comments (3x), `// init host side`, `// launch tensor cast module`, `// begin wrw`, assert/layout check comments
- **Preserved:** gfx10/RDNA WGP vs CU count hardware notes, sclk override rationale, fp_factor per-arch throughput, bf16 WMMA throughput rationale, WMMA D-operand width design rationale, Phase 24/27/34 wmma_acc/bf16/atomic_pack rationale, TODO pre-clear, all commented-out debug code

#### `driver/igemm_gtc_base.h`
- **Removable:** ~9% (7 removed)
- **Removed:** driver_mode enum descriptions (2x), `// remove min and max...` (2x), workspace size direction comments (4x)
- **Preserved:** driverDataType enum descriptions, Phase 35 karg rationale, WMMA union reuse rationale, extensive Phase NN tunable field rationale (L162–275), kernel-naming-sync rationale, CAUTION warning, split cap rationale

#### `driver/igemm_fwd_gtc_driver.h`
- **Removable:** ~20%
- **Preserved:** magic division denom comments, WMMA karg Phase 5a/5d/7 rationale, Phase 49/60 magic division, Phase 25/26b/51/38/39 wmma_m_tail/n_tail/tdm rationale, WMMA 2D grid launch rationale

#### `driver/igemm_bwd_gtc_driver.h`
- **Removable:** ~18%
- **Preserved:** WMMA karg Phase 5b/5e/7 + p_in/p_out swap rationale, Phase 48/60 gemm_k_per_wg + magic division, Phase 26a/51/42 tail/tdm rationale, WMMA p_in/p_out swap launch rationale

#### `driver/igemm_wrw_gtc_driver.h`
- **Removable:** ~13%
- **Preserved:** WMMA karg Phase 5c/5f/7 + 3-way ROTATION rationale (critical), Phase 35/58 gemm_k_tail/num_splits/streamk/persistent kernel rationale, Phase 35/51/45 tail/tdm rationale, gemm_k_global_split ternary search rationale, Phase 33 occupancy heuristic, Phase 34/58 streamk rationale, Phase 35 reduction kernel rationale

#### `driver/args.h`
- **Removable:** ~14% (1 removed)
- **Removed:** `// we are looking for the "x" character`
- **Preserved:** TODO not safe, TODO try...catch, commented-out printf, return value docs

#### `driver/config_parser.h`
- **Removable:** ~40% (1 removed)
- **Removed:** `// return first section with name sec_name`
- **Preserved:** commented-out null-termination, config file unix format WARNING + dos2unix workaround, commented-out printf

#### `driver/common.h`
- **Removable:** 0%
- **Preserved:** All 15 comments — BF16 NaN preservation rationale + rocBLAS reference URL, RNE rounding algorithm explanation

#### `driver/magic_div.h`
- **Removable:** 0% (1 compressed)
- **Compressed:** 24-line algorithm explanation + example code block → 4-line summary: `Magic number integer division for uint32 (limited to INT32_MAX to avoid branching on GPU). Host: magic_div_u32_gen(d) -> {magic, shift}. GPU: (mulhi(magic, numer) + numer) >> shift.`

#### `driver/naive_conv.h`
- **Removable:** ~50% (5 removed)
- **Removed:** 4× `// sliding window for this filter`, `/************************** nhwc ****************************/` separator
- **Preserved:** `// #define NAIVE_CONV_THREADED`, `// if use threaded conv need c++11`, all `cur_h`/`cur_w` inline math, commented-out `naive_conv_blockwise_in_parallel`

#### `driver/naive_tiled_conv.h`
- **Removable:** ~57% (8 removed)
- **Removed:** tile size/pad calculation comments that restate variable assignments
- **Preserved:** 2D spatial-tile iterator rationale, `// tx ty is used to tile output h w`, commented-out printf

#### `driver/gpu_general_tensor_reorder.h`
- **Removable:** ~93% (14 removed)
- **Removed:** `//reorder kernel`, 12× order-label comments `//(0, ...)` through `//(3,...)`, `//loop over`
- **Preserved:** commented-out 32x64 pack_2x4 case, TODO need find better way to decide transpose tile size

#### `driver/gpu_batched_transpose/batched_transpose.cpp`
- **Removable:** ~78% (140 lines removed)
- **Removed:** All parameter-name comments on empty function stubs (`/*dst*/`, `/*src*/`, `/*height*/`, etc.) across 14 stub functions
- **Preserved:** cppcheck-suppress directives, layout assumption comments, transpose ASCII art, commented-out `__shared__` code, TODO

#### `driver/gpu_naive_conv/naive_conv.cpp`
- **Removable:** ~50% (234 lines removed)
- **Removed:** 33 multi-line block comments `/* need to compute total output/input/filter pixel... */` — restate obvious grid_size calculation
- **Preserved:** `// hcc seems need __device__ __host__ together`, `// design block_size 256` (design decision), all `cur_h`/`cur_w` inline math

#### `driver/gpu_tensor_reorder/general_tensor_reorder.cpp`
- **Removable:** ~17% (5 removed)
- **Removed:** 5× `//unroll k         block          thread`
- **Preserved:** cppcheck-suppress directives, reorder sequence comments with layout names

#### `driver/perf/gmap.cpp`
- **Removable:** ~22% (9 removed)
- **Removed:** `// get nd indices from a linear index`, `// get offset from nd indices`, `// nd range check`, `// serialize block request`, `// valid all record`, `// position of this block in ndim space` (2x), `// input`/`// wei`/`// out` section labels
- **Preserved:** struct field comments, overlap corner case note, convolution input pixel rationale, TODO ugly solution, commented-out code, global memory access pattern file purpose, commented-out WARNING

#### Other driver files with 0% removable
- `driver/utility.h`, `driver/gpu_naive_conv.h`, `driver/gpu_nchw_nhwc_transpose.h`, `driver/use_tracker.hpp`, `driver/perf.h`, `driver/shisa_dumps.h`, `driver/tensor_transpose.h`, `driver/tensor_copy_cpu.h`, `driver/tensor_validation_cpu.h`, `driver/gpu_tensor_reorder.h`, `driver/gpu_tensor_reorder/sequence.hpp`, `driver/gpu_tensor_cast/gpu_tensor_cast.cpp` — all comments are design rationale, compiler notes, or commented-out code

---

## Summary of Important Architectural Rationale Discovered in Comments

### 1. Phase-Tracking Development Convention
The codebase uses a "Phase N" comment convention throughout `python/igemm/`, `python/operations/`, `driver/igemm_gtc_base.h`, and `driver/igemm_*_gtc_driver.h` to document incremental feature development with hardware ISA citations. Each phase references specific hardware behavior, validation results, and design decisions. **These are the primary architectural documentation and must never be removed.**

### 2. Python↔C++ Kernel Name Synchronization
The tunable parameter object and kernel name mangling function exist in both Python (`igemm_base.py`) and C++ (`igemm_gtc_base.h`) with no code generation bridge. Any new tunable field, name-mangling suffix, or semantic change requires synchronized edits to both sides. The C++ driver uses `igemm_gtc_encode_kernel_name()` to reconstruct the exact kernel symbol Python emitted, for `hipModuleGetFunction` lookup — a mismatch causes a silent `hipModuleGetFunction` failure. (Preserved in `igemm_gtc_base.h` L162–275.)

### 3. WMMA D-Operand Buffer Width Decision
`conv_driver.cpp` L677–700 documents the critical architectural decision about WMMA D-operand width — when to use fp32 vs narrowed (f16/bf16) accumulator buffers, and the Phase 24/27/34 rationale for `wmma_acc_f16`/`wmma_acc_bf16`/`atomic_pack_bf16` output-buffer handling.

### 4. WRW 3-Way Pointer Rotation
`driver/igemm_wrw_gtc_driver.h` L68–77 documents that wrw's kernel field semantics differ from `run()`'s conventional pointer mapping: `p_in=grad_output (READ)`, `p_wei=input (READ)`, `p_out=grad_weight (WRITE)`, but `run()`'s conventional `p_in=input(READ)`, `p_wei=grad_weight(WRITE)`, `p_out=grad_output(READ)` — so the mapping is a 3-way ROTATION, not a simple swap.

### 5. Magic Division for GPU-Side Integer Division
`driver/magic_div.h` documents the magic number technique for replacing runtime integer division with multiply+shift on GPU (avoiding branching). Limited to INT32_MAX to avoid branch-based algorithms. Used pervasively in the C++ driver karg structs (magic_0 through magic_6 fields with documented denominators).

### 6. BF16 RNE Rounding and NaN Preservation
`driver/common.h` L69–108 and `driver/gpu_naive_conv/naive_conv.cpp` L47–55 document the BF16 conversion algorithm matching rocBLAS, including signaling NaN preservation and round-to-nearest-even rounding.

### 7. gfx10/RDNA vs gfx1250/CDNA5 CU Counting
`conv_driver.cpp` L180–181: gfx10/RDNA reports WGP count (each = 2 CUs), so double for those. gfx1250/CDNA5 reports actual CU count, so don't double. This affects all TFLOPS efficiency calculations.

### 8. fp32 WMMA Occupancy Race (Known Issue)
Documented in AGENTS.md and referenced in code comments: fp32 WMMA (`gemm_k_per_block=4`) requires `lds_double_buffer=1` unconditionally — single-buffered LDS produces silent wrong-answer results at high occupancy due to a barrier visibility race on the last lane's LDS write.

### 9. Macro Expansion 3-Step Invariant
`python/codegen/macro.py` L69/78/83: macro inline expansion must follow overwrite args → emit → restore args. Violating this order corrupts the macro state.

### 10. Debug Flags Break Correctness (WARNING)
`python/codegen/mc.py`: WARNING that debug flags strip LDS/global IO — produces incorrect results. These are debugging-only flags, not safe for production.

### 11. Concurrency: Never Open File in Multiple Processes
`python/codegen_driver.py` L269–270: WARNING about never opening the output file in more than one thread/process. The emit and compile phases must happen in the same process.

### 12. Stream-K Persistent Kernel (wrw)
`driver/igemm_wrw_gtc_driver.h` L98–130: Phase 35/58 documents the stream-K persistent kernel design for wrw, including `gemm_k_tail` remainder handling, `num_splits` ternary search, and the reduction kernel as a non-atomic alternative to atomic-add epilogue.

### 13. VGPR-MSB Banking
`python/operations/coalescing_store_wmma.py` and `python/operations/wmma_main_loop.py`: Phase-based documentation of VGPR-MSB banking for WMMA D-operand layout — critical for register allocation and avoiding bank conflicts.

### 14. DPP8 Limitation
`python/operations/dotx_main_loop.py` L286, `python/operations/main_loop_graph.py` L286/L457: DPP8 (Data Parallel Processing) assumption documented — limits certain dotx operations to specific thread-group patterns.

### 15. Compiler Workarounds
- `driver/gpu_tensor_reorder/sequence.hpp` L40: dummy array element to prevent compiler error on zero-size array
- `driver/gpu_naive_conv/naive_conv.cpp` L29: `hcc` requires `__device__ __host__` together
- `cppcheck-suppress invalidPointerCast` directives throughout batched_transpose.cpp and general_tensor_reorder.cpp
