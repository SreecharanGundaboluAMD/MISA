# MISA / iGEMMgen — Architecture Map

> Generated 2026-09-01. Grounded in source via parallel scout analysis of every major directory.
> Use this map as the basis for all subsequent reviews and findings.

## 1. Directory-by-Directory Breakdown

---

### `python/codegen/` — Low-Level Assembler Framework

**Purpose:** Generic AMDGPU assembly code-generation toolkit: symbol management, machine-code printer with pluggable emitter backends, arch-aware instruction encoders, macro definitions, machine basic block (MBB) construction/scheduling, `.config` file parsing, kernel code/metadata emission, and ROCm toolchain compilation wrappers.

**Key Classes:**

| Class | File | Role |
|---|---|---|
| `mc_asm_printer_t` | `mc.py` | Central MC printer — holds emitter + arch_config; provides `emit()`, `inject()`, `deferred_context()` |
| `mc_base_t` | `mc.py` | Mixin base for all emitters; `__init__` calls `mc.inject(self)` to gain `_emit`/`_deferred_context` |
| `mc_emit_to_file_t` | `mc.py` | File emitter backend with license header |
| `mc_emit_to_string_t` | `mc.py` | String buffer emitter backend |
| `mc_deferred_emit_t` | `mc.py` | Deferred capture emitter for macro/scheduler text fragments |
| `macro_base_t` | `macro.py` | `.macro/.endm` definition or inline expansion framework |
| `sym_t` / `msym_t` | `symbol.py` | ASM symbol abstraction (`.set` label + register-range expressions) |
| `inst_base_t` / `inst_*_t` | `instruction.py` | Per-instruction encoder singletons branching on arch |
| `amdgpu_arch_config_t` | `amdgpu.py` | Arch capability config (dlops/xdlops/wmma flags, arch, code_object) |
| `amd_kernel_code_t` | `amdgpu.py` | Emits `.amdhsa_kernel` (cov3) or `.amd_kernel_code_t` (cov2) descriptor |
| `amdgpu_metadata_t` | `amdgpu.py` | Emits `.amdgpu_metadata` YAML block |
| `hsa_header_t` / `hsa_footer_t` | `amdgpu.py` | cov2 HSA header/footer |
| `config_parser_t` | `config_parser.py` | INI-style `.config` → `config_content_t` (sections + typed values) |
| `config_content_t` / `config_section_t` | `config_parser.py` | Parsed config container (list of sections) |
| `compile_asm_t` | `compile.py` | `.s` → `.hsaco` via `clang++ -x assembler -target amdgcn--amdhsa` |
| `compile_hip_t` | `compile.py` | `.hip` → `.hsaco` via `hipcc --cuda-device-only` |
| `compile_host_t` | `compile.py` | C++ → `.exe` via `hipcc -std=c++17` |
| `compile_disass_t` | `compile.py` | `.hsaco` → `.disass.s` via `llvm-objdump --disassemble` |
| `machine_basic_block_t` | `mbb.py` | Ordered list of typed instructions (SALU/VALU/DS/GLOBAL/etc.) |
| `simple_interleave_scheduler_t` | `scheduler.py` | MBB interleaving (global-mem loads between compute, shared-mem after MFMA) |

**Key Interfaces:**
- `mc.inject(other)` — copies 11 emit/indent/deferred methods onto `other` as bound methods (dependency injection)
- `mc_set_current(mc)` / `mc_get_current()` — global MC pointer for stateless instruction encoders
- `config_parser_t(file).parse()` → `config_content_t`
- `compile_asm_t.compile()` / `compile_host_t.compile()` — return `bool`

**Dependencies:**
- Internal layering (acyclic): `symbol.py`, `mc.py`, `config_parser.py` (Layer 0, no intra-package deps) → `macro.py`, `amdgpu.py`, `mbb.py` (Layer 1, ← mc) → `instruction.py`, `scheduler.py` (Layer 2) → `compile.py` (Layer 3, ← amdgpu) → `__init__.py` (facade)
- External: `subprocess`, `os`, `sys`, `inspect`, `re`, `copy`

**External Integrations:**
- ROCm toolchain at `ROCM_PATH=/home/sgundabo/rocm-10.1`: `clang++` (assembler), `hipcc` (HIP host/device), `llvm-objdump` (disassembly)
- `git rev-parse HEAD` for version embedding in generated files

**Data Flow:**
```
.config file → config_parser_t → config_content_t (sections)
                                    ↓
                    [consumed by igemm_codegen.py / codegen_driver_t]
                                    ↓
mc_asm_printer_t(emitter, arch_config) ← mc_set_current()
                                    ↓
              kernel generators call self._emit("...") → emitter buffer
                                    ↓
              compile_asm_t: .s → .hsaco | compile_disass_t: .hsaco → .disass.s
```

---

### `python/igemm/` — Kernel Generators

**Purpose:** Per-direction, per-layout convolution kernel generator classes that produce AMDGPU assembly text from tunable parameters. The heart of the codegen pipeline.

**Key Classes:**

| Class | File | Role |
|---|---|---|
| `igemm_gtc_tunable_parameter_t` | `igemm_base.py` | Central tunable object; dispatches on FMA type (MAC/DLOPS/XDLOPS/WMMA) to read fma-specific fields |
| `igemm_fwd_gtc_wmma_nhwc_t` | `igemm_fwd_gtc_wmma_nhwc.py` | Active gfx1250 fwd WMMA generator (2051 lines) |
| `igemm_bwd_gtc_wmma_nhwc_t` | `igemm_bwd_gtc_wmma_nhwc.py` | Active gfx1250 bwd WMMA generator (1817 lines) |
| `igemm_wrw_gtc_wmma_nhwc_t` | `igemm_wrw_gtc_wmma_nhwc.py` | Active gfx1250 wrw WMMA generator (2133 lines; includes stream-K) |
| `igemm_fwd_gtc_t` | `igemm_fwd_gtc.py` | Legacy NCHW fwd (MAC/DLOPS/XDLOPS) |
| `igemm_bwd_gtc_t` | `igemm_bwd_gtc.py` | Legacy NCHW bwd |
| `igemm_wrw_gtc_t` | `igemm_wrw_gtc.py` | Legacy NCHW wrw |
| `igemm_fwd_gtc_nhwc_t` | `igemm_fwd_gtc_nhwc.py` | Legacy NHWC fwd (DLOPS/XDLOPS) |
| `igemm_bwd_gtc_nhwc_t` | `igemm_bwd_gtc_nhwc.py` | Legacy NHWC bwd (largest file, 3086 lines) |
| `igemm_wrw_gtc_nhwc_t` | `igemm_wrw_gtc_nhwc.py` | Legacy NHWC wrw |
| `igemm_fwd_gtc_nchwc_t` | `igemm_fwd_gtc_nchwc.py` | NCHWc vectorized fwd (DLOPS only) |
| `igemm_upsampling_clear_t` | `igemm_upsampling_clear.py` | Utility kernel for bwd upsampling zero-init |

**Key Interfaces (WMMA generators — the active target):**

All three WMMA generators share this emit flow:
```
emit_kernel_body()
  → emit_kernel_prologue()      # SGPR/VGPR setup, address computation, accumulator zero-init
  → emit_kernel_tap_loop()      # Runtime y×x tap loop wrapping static K-main-loop
  → emit_kernel_epilogue()      # s_wait_storecnt, s_endpgm
```

Functor methods (returned as callables, consumed by `wmma_main_loop_t`):
- `global_load_a_functor()` / `global_load_b_functor()` — issue global loads
- `shared_store_a_functor()` / `shared_store_b_functor()` — ds_write to LDS
- `shared_load_a_functor()` / `shared_load_b_functor()` — ds_read from LDS (transposed for bwd/wrw)
- `move_slice_window_a_functor()` / `move_slice_window_b_functor()` — advance global addresses

Direction-specific semantics:
- **fwd**: A=input (untransposed), B=weight (untransposed); v_c accumulated across all taps
- **bwd**: A=grad_output (untransposed), B=weight (transposed in LDS); v_c accumulated across taps
- **wrw**: A=grad_output (transposed), B=input (transposed); v_c zeroed + epilogue fired once per tap; supports `wrw_streamk` persistent kernel

**Key Functions:**
- `igemm_gtc_encode_kernel_name(tunable, arch)` → str — mangled kernel symbol name (must match C++ side)
- `get_igemm_gtc_fma_type(tunable_dict)` → str — dispatches MAC/DLOPS/XDLOPS/WMMA from dict keys
- `utility_dict_with_default_t(d)` — callable returning defaults for missing keys (used pervasively)

**Dependencies:**
- `from ..codegen import *` — `mc_base_t`, `macro_base_t`, `sym_t`, `gpr_sequencer_t`, arch/precision helpers, compile classes, MBB/scheduler
- `from ..operations import *` — `wmma_main_loop_t`, `ctrl_wmma_mapping_t`, `igemm_coalescing_store_wmma_t`, `utility_dict_with_default_t`, division macros
- `from .igemm_base import *` — sole intra-package dependency (all generators import only from `igemm_base`)

**External Integrations:** None directly (all through `python/codegen/compile.py`)

**Data Flow:**
```
tunable_dict (from .config) → igemm_gtc_tunable_parameter_t(tunable_dict)
    → generator.__init__(mc, tunable)
    → generator.emit_kernel_body()
        → emit_kernel_prologue() [SGPR/VGPR setup]
        → emit_kernel_tap_loop() [wraps emit_kernel_fma_main_loop()]
            → wmma_main_loop_t(mc, ctrl).emit() [calls functors]
        → emit_kernel_epilogue() [wait + endpgm]
    → assembly text via self._emit("...") → mc emitter buffer → .s file
```

---

### `python/operations/` — Operation Helpers (Main-Loop Drivers)

**Purpose:** Functor-based main-loop driver library and operation primitives. Each FMA variant (WMMA, MFMA, FMA, dotx) has a main-loop class that emits the software-pipelined K-iteration loop, calling direction-specific functors at load/store/compute points.

**Key Classes:**

| Class | File | Role |
|---|---|---|
| `ctrl_wmma_main_loop_t` | `wmma_main_loop.py` | WMMA loop config: 12 functor slots, buffer count, interleave, setprio, vgpr_msb |
| `wmma_main_loop_t` | `wmma_main_loop.py` | Emits WMMA K-loop: prologue → body (barrier/load/compute/store/switch) → tail |
| `ctrl_wmma_mapping_t` | `wmma_mapping.py` | WMMA thread-to-tile mapping (wave32, 16×16 instruction tiles) |
| `igemm_wmma_mapping_t` | `wmma_mapping.py` | Emits address-computation assembly for A/B/C/D matrix indices |
| `inst_wmma_t` | `wmma.py` | WMMA instruction descriptor; `__call__` emits `v_wmma_*` string |
| `ctrl_coalescing_store_wmma_t` | `coalescing_store_wmma.py` | WMMA epilogue config: atomic/non-atomic, tail, direct_store, chunked |
| `igemm_coalescing_store_wmma_t` | `coalescing_store_wmma.py` | Emits output epilogue: LDS-reshuffle+store or atomic_add |
| `ctrl_mfma_main_loop_t` / `mfma_main_loop_t` | `mfma_main_loop.py` | MFMA K-loop for gfx908+ (3478 lines, uses MBB IR) |
| `ctrl_xdlops_mapping_t` | `xdlops_mapping.py` | XDLOPS thread-to-tile mapping for gfx908+ |
| `ctrl_dotx_main_loop_t` / `dotx_main_loop_t` | `dotx_main_loop.py` | DLP dotx K-loop for gfx1030 (graph-based IR) |
| `ctrl_coalescing_store_t` | `coalescing_store.py` | Legacy XDLOPS coalescing store (1478 lines) |
| `conv_param_t` | `conv.py` | Convolution domain constants and parameter model |

**Key Interfaces:**
- `wmma_main_loop_t(mc, ctrl).emit()` — emits the full WMMA K-loop
- `inst_wmma_t.__call__(reg_d, reg_a, reg_b, reg_c)` — returns `v_wmma_*` instruction string
- `igemm_wmma_mapping_t.get_gemm_index_for_src_matrix(...)` / `_transposed(...)` / `_for_dst_matrix(...)` — address computation
- `get_ctrl_wmma_mapping_from_wave_tile(...)` — looks up pre-validated tile mapping from `ctrl_wmma_mapping_table`
- 12 functor slots on `ctrl_wmma_main_loop_t`: `global_load_a/b_functor`, `shared_store_a/b_functor`, `shared_load_a/b_functor`, `move_slice_window_a/b_functor`, `global_load_chunk_a/b_functor`, `shared_store_chunk_a/b_functor`

**Dependencies:**
- `from ..codegen import *` — `mc_base_t`, `macro_base_t`, `sym_t`, arch/precision constants
- Internal: `wmma_main_loop → wmma_mapping → wmma`; `mfma_main_loop → xdlops_mapping → mfma`; `dotx_main_loop → dotx_mapping → dotx → main_loop_graph`; `coalescing_store → thread_mapping + global_memory + shared_memory`; `coalescing_store_wmma → global_memory + wmma_mapping`
- No imports from `python/igemm/` (operations is lower-level than igemm)

**External Integrations:** None

**Data Flow:**
```
Kernel generator creates functors (closures capturing tunable params)
    → constructs ctrl_wmma_main_loop_t (functor slots + config)
    → wmma_main_loop_t(mc, ctrl).emit()
        → prologue: wait + store first tile
        → loop body: barrier → shared_load A/B → emit_wmma_tile() → global_load next → switch buffer
        → tail: final compute
    → assembly text via self._emit("...")
```

---

### `python/` (root) — Orchestrators

**Purpose:** Top-level orchestration: generator selection, emission, compilation, host driver build, and CLI entry point.

**Key Classes/Functions:**

| Class/Function | File | Role |
|---|---|---|
| `codegen_driver_t` | `codegen_driver.py` | Top orchestrator: selects generator by direction+tunable fields, emits, compiles |
| `host_driver()` | `host_driver.py` | Builds `conv_driver.exe` + `naive_conv.hsaco` + `igemm_gtc_tensor_cast.hsaco` |
| `igemm_sequence_driver()` | `sequence_driver.py` | Seq mode (gfx908-only): exhaustive tunable enumeration |
| `perf_advisor_t` | `perf_advisor.py` | Stub placeholder (empty `advise_occupancy()`) |
| `igemm_flatten()` | `igemm_codegen.py` | Flat mode: creates MC printer, calls `codegen_driver_t` |
| `igemm_try_expand_tunable_content()` | `igemm_codegen.py` | Expands list-valued `tensor_layout` into multiple sections |

**Key Interfaces:**
- `codegen_driver_t(mc, tunable_dicts).__call__(split_kernel=...)` — emits + compiles
  - `__init__`: dispatches on `direction` + `wmma_tile_m`/`tensor_layout` to select generator class
  - `do_emit()`: `emit_hsa_header()` → `emit_global_macro()` (gfx1250 WMMA: no-op) → `emit_igemm_macro()` → `emit_igemm_kernel()` → `emit_metadata()`
  - `do_compile()`: `compile_asm_t` then `compile_disass_t`
- `host_driver(arch, has_fp16/int8/bf16/int4_config, out_dir, config_file, ...)` — compiles C++ driver with `-DIGEMM_CONFIG_FILE` / `-DIGEMM_HSACO` / `-DUSE_{HALF,INT8,BF16,INT4}`

**Dependencies:**
- `codegen_driver.py` → `.igemm` (generator classes, tunable), `.codegen` (mc, compile, amdgpu), `.operations` (macros, instructions)
- `host_driver.py` → `.codegen` (compile_host_t, compile_hip_t, amdgpu_arch_config_t)
- `sequence_driver.py` → `.igemm`, `.codegen`, `.codegen_driver.codegen_driver_t`, `.host_driver.host_driver`
- `igemm_codegen.py` → `from python import *` (pulls everything)

**External Integrations:**
- `compile_host_t` shells out to `hipcc` / `g++` with ROCm include paths
- `compile_hip_t` shells out to `hipcc --cuda-device-only`

**Data Flow (full pipeline):**
```
.config file
  → config_parser_t → config_content
  → igemm_try_expand_tunable_content (expand list-valued fields)
  → [flat mode]:
      → host_driver() → conv_driver.exe + naive_conv.hsaco + tensor_cast.hsaco
      → igemm_flatten():
          → amdgpu_arch_config_t + mc_asm_printer_t
          → codegen_driver_t(mc, tunable_dicts)
              → do_emit(): hsa_header → global_macro → igemm_macro → per-kernel .inc
                  → kernel.emit_kernel_body()
                      → wmma_main_loop_t.emit() [calls functors]
                      → igemm_coalescing_store_wmma_t [epilogue]
              → do_compile(): .s → .hsaco → .disass.s
  → [seq mode] (gfx908 only):
      → igemm_sequence_driver(): enumerate tunables → codegen_driver_t → host_driver()
```

**Known issue:** `igemm_codegen.py` seq-mode branch calls `sequence_driver(...)` but the function is `igemm_sequence_driver` — latent name mismatch (seq mode is gfx908-only, not the active path).

---

### `driver/` — C++ Host Driver

**Purpose:** Loads HSACO binaries, parses `.config` tunables, constructs per-direction driver objects that benchmark and verify each kernel against a naive reference convolution, and reports timing in TFLOPS.

**Key Classes:**

| Class | File | Role |
|---|---|---|
| `igemm_gtc_tunable_t` | `igemm_gtc_base.h` | C++ mirror of Python's `igemm_gtc_tunable_parameter_t` (manual sync) |
| `igemm_driver_base_t` | `igemm_gtc_base.h` | Abstract base: pure virtuals `get_block_size`/`get_grid_size`/`tunable_is_valid`/`run`/`get_gks_list`/`get_spatial_tiling` |
| `igemm_fwd_gtc_t` | `igemm_fwd_gtc_driver.h` | Fwd driver: GEMM M=N·ho·wo, N=K/group; WMMA block=waves×32 |
| `igemm_bwd_gtc_t` | `igemm_bwd_gtc_driver.h` | Bwd driver: GEMM M=N·hi·wi, N=K/group; swaps p_in/p_out |
| `igemm_wrw_gtc_t` | `igemm_wrw_gtc_driver.h` | Wrw driver: GEMM M=K/group, N=C/group; rotates p_in/p_wei/p_out; ternary-search split-K |
| `args_t` | `args.h` | MIOpenDriver-compatible CLI parser (~40 args) |
| `config_parser_t` | `config_parser.h` | C++ .config parser (mirrors Python's) |
| `bfloat16` | `common.h` | Software bf16 with RNE rounding |
| `magic_div_u32_t` | `magic_div.h` | Magic division (multiply+shift replacing runtime `div`) |
| `DumpWriter_t` | `shisa_dumps.h` | Kernel dispatch dump writer (`.gks{N}.dump`) |

**Key Interfaces:**
- `igemm_driver_base_t::run(arg, tunable, p_in, p_wei, p_out, current_gks)` → `result_t` — launch + benchmark + verify
- `igemm_gtc_encode_kernel_name(tunable)` — reconstructs kernel symbol for `hipModuleGetFunction` (must match Python)
- `igemm_gtc_tunable_from_config(config_content)` → `vector<igemm_gtc_tunable_t>`
- `igemm_launch_kernels(kernels, prolog, postlog, warmup, repeat)` — multi-kernel dispatch with warmup
- `launch_conv_driver()` template — iterates tunables, filters, calls `run()`, tracks fastest

**GEMM Dimension Mapping (WMMA NHWC):**

| Direction | GEMM M | GEMM N | GEMM K | A operand | B operand | C operand |
|---|---|---|---|---|---|---|
| fwd | N·ho·wo | K/group | C/group | input | weight | output |
| bwd | N·hi·wi | K/group | C/group | grad_output | weight | grad_input |
| wrw | K/group | C/group | N·ho·wo | grad_output | input | grad_weight |

**Dependencies:**
- Include graph (acyclic): `common.h` / `utility.h` / `magic_div.h` (leaf) ← `config_parser.h` ← `igemm_gtc_base.h` ← `igemm_{fwd,bwd,wrw}_gtc_driver.h` ← `conv_driver.cpp`
- `naive_conv.h` (CPU reference) or `gpu_naive_conv.h` (GPU reference via HSACO)
- `tensor_validation_cpu.h` (NRMS comparison), `tensor_copy_cpu.h` (dtype conversion)
- `gpu_tensor_cast/`, `gpu_tensor_reorder/`, `gpu_batched_transpose/`, `gpu_naive_conv/` — GPU support kernel sources compiled to separate HSACOs

**External Integrations:**
- HIP runtime: `hipModuleLoad`, `hipModuleGetFunction`, `hipExtModuleLaunchKernel`, `hipMalloc`, `hipMemcpy`, `hipEvent*`, `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`
- `half.hpp` (external, pfultz2/half 1.12.0) for fp16 type when `USE_HALF` defined
- No direct ROCm/comgr link — driver consumes pre-assembled HSACOs at runtime

**Data Flow:**
```
conv_driver.exe <mode> -n .. -c .. -H .. -W .. -k .. -y .. -x .. ... -F <bitmask> -V <0|1>
  → hipModuleLoad(hsaco) → config_parser → igemm_gtc_tunable_from_config
  → <mode> arg → driverDataType_t (PRECISION GOTCHA: must match kernel precision)
  → allocate host(fp32) + device(dtype) buffers
  → for each direction (fwd/bwd/wrw per -F bitmask):
      → generate random input, compute naive reference (CPU or GPU)
      → create driver object (igemm_{dir}_gtc_t)
      → launch_conv_driver(): for each tunable:
          → driver->run() → hipExtModuleLaunchKernel (warmup + repeat)
          → optional verify: valid_vector<T>(NRMS tolerance)
          → track fastest, print cost/tflops/efficiency/valid:y|n
```

---

### `config/` — Configuration Files

**Purpose:** ~220 INI-format `.config` files defining tunable kernel parameters. Each file has a `[codegen]` header + one or more `[igemm_{dir}_gtc]` sections.

**Key Format:**
```ini
[codegen]
arch = 'gfx1250'
code_object = 'cov3'
mode = 'flat'

[igemm_fwd_gtc]
gemm_m_per_block = 128
gemm_n_per_block = 128
gemm_k_per_block = 32
wmma_tile_m = 16
wmma_repeat_m = 4
wmma_tile_n = 16
wmma_repeat_n = 4
tensor_a_thread_lengths = [1, 32, 1, 1]
tensor_a_cluster_lengths = [1, 1, 1, 128]
tensor_b_thread_lengths = [1, 32, 1, 1]
tensor_b_cluster_lengths = [1, 1, 1, 128]
direction = "fwd"
precision = "fp16"
tensor_layout = 'nhwc'
nxb = 0
nxe = 0
wavefront_size = 32
direct_store = 1          # optional, default 0
lds_double_buffer = 1     # optional, default 0
```

**Categorization (gfx1250, ~195 files):**
- 12 master `_all.config` (direction × precision) — auto-generated unions by `build_gfx1250_master_configs.py`
- 27 per-tile combinatorial `_all.config` — generated by `generate_all_configs.py` with ~25 mutual-exclusion rules
- ~150 narrow per-mechanism configs (direct, dbuf, async, tdm, saddr, interleave, k2x, gsplit, streamk, tail variants, acc variants, etc.)
- Legacy: gfx950 (10), gfx942 (7), gfx940 (5), gfx90a (7), gfx908 (7), gfx1030 (8)

**Dependencies:** Consumed by both Python (`config_parser_t`) and C++ (`config_parser_t` in `driver/config_parser.h`) — shared schema, no code bridge.

---

### `script/` — Automation Scripts

**Purpose:** Four roles: (1) config generation, (2) building, (3) post-build filtering, (4) benchmarking.

**Key Scripts:**

| Script | Role |
|---|---|
| `generate_all_configs.py` | Combinatorial per-tile configs (2^11 flags × ~25 mutual-exclusion rules) |
| `build_gfx1250_master_configs.py` | Unions narrow configs → master `_all.config` |
| `build_and_filter_configs.py` | Builds each config, drops sections failing assembly (VGPR overflow) |
| `benchmark_gfx1250_vs_miopen.py` | A/B benchmark vs MIOpen on 38 shapes |
| `benchmark_gfx1250_vs_gfx950_diverse.py` | A/B benchmark on 60 diverse shapes |
| `classify_gfx1250_coverage.py` | Static analysis: classifies 95K MIOpen shapes against MISA coverage |
| `sweep_streamk_*.py` / `bench_streamk_vs_gsplit.py` | Stream-K parameter sweeps |
| `gen_gfx{1250,950,940,942,90a}_conv*.sh` | Shell wrappers for `igemm_codegen.py` |
| `gtc_conv_{model,resnet50,ssd,gemm_big_size}.sh` | Benchmark scripts running `conv_driver.exe` on model shapes |
| `smoke_test.sh` | Randomized `conv_driver.exe` smoke test |
| `gfx1250_occupancy_check.cpp` | Standalone HIP occupancy measurement tool |

**Dependencies:**
- Config generation scripts: pure text I/O, no `python/` imports
- Benchmark scripts: invoke `igemm_codegen.py` (subprocess) + `conv_driver.exe` (subprocess) + `rocm-smi`
- `build_and_filter_configs.py`: invokes `igemm_codegen.py`, regexes build output for kernel names
- No direct imports from `python/codegen` or `python/igemm`

**External Integrations:** `igemm_codegen.py`, `conv_driver.exe`, `rocm-smi`, `hipcc` (for occupancy check)

---

### `test/` — Test Infrastructure

**Purpose:** Python codegen unit tests + C++ HIP kernel verification tests.

**Python Tests:**

| File | Tests | Run |
|---|---|---|
| `unittest.py` | mc_asm_printer_t emission, coalescing store, thread mapping, xdlops mapping, dotx mapping, macros | `python3 test/unittest.py` |
| `tensor_transformation/tensor_transformation_test.py` | Tensor descriptor algebra (split, merge, grouped_slice, vectorize) | `python3 test/...` |
| `twiddle/twiddle_test.py` | End-to-end: FFT codegen → compile → GPU run vs CPU Cooley-Tukey | `python3 test/...` (needs ROCm) |

**C++ Tests:**

| Test | Verifies | Arch |
|---|---|---|
| `naive_conv/` | 2D/3D naive conv (fwd/bwd/wrw) | gfx908 |
| `naive_tiled_conv/` | Tiled naive conv | gfx908 |
| `nchw_nhwc_transpose/` | NCHW↔NHWC transpose | gfx908 |
| `tensor_reorder/` | General 4D tensor reorder (24 permutations) | gfx1030 |
| `persistent_workgroup/` | Persistent workgroup fwd conv (NCHWC) | gfx1030 |
| `inference/` | Inference fwd-btm conv (fp16/int8, batch=1) | gfx1030 |

**Dependencies:**
- Python tests: `from python import *` (in-process, no subprocess)
- C++ tests: include from `driver/` (`naive_conv.h`, `gpu_naive_conv.h`, `args.h`, etc.), link HIP runtime
- `test/common/utility.h` + `test/common/fft.h` — shared C++ utilities

---

### `docs/` — Design Documentation

**Purpose:** ~25 markdown files: optimization log, backlog, profiling, benchmarking, known issues, external research, design plans.

**Key Documents:**
- `gfx1250_wmma_layout.md` (6029 lines) — master engineering log, 68+ phases, empirically verified WMMA register layouts
- `gfx1250_optimization_backlog.md` — tracked checklist (Tiers 0-3)
- `gfx1250_fp32_wmma_occupancy_race.md` — known issue: fp32 WMMA requires `lds_double_buffer=1` unconditionally
- `gfx1250_streamk_design.md` — Stream-K/persistent-kernel design for wrw
- `gfx1250_rocprof_profiling.md` — hardware-counter profiling
- External research: CK, hipconv, FlyDSL, rocKE deep dives

---

## 2. Architectural Analysis

### Core Modules

| Module | Location | Role |
|---|---|---|
| **Assembler framework** | `python/codegen/` | MC printer, instruction encoders, symbols, macros, MBB, scheduler, config parser, compile wrappers |
| **Kernel generators** | `python/igemm/` | Per-direction WMMA/legacy generators that emit assembly from tunables |
| **Main-loop drivers** | `python/operations/` | Functor-based K-loop emission (WMMA, MFMA, FMA, dotx) + coalescing store + thread mapping |
| **Top orchestrator** | `python/codegen_driver.py` | Generator selection + emit + compile pipeline |
| **C++ host driver** | `driver/` | HSACO loading, benchmarking, verification, split-K search |

### Utility Modules

| Module | Location | Role |
|---|---|---|
| `python/codegen/symbol.py` | ASM symbol abstraction | Leaf, no deps |
| `python/codegen/config_parser.py` | INI parser | Leaf, no deps |
| `driver/common.h` | bfloat16, HIP_CALL, conv_out_size | Leaf |
| `driver/utility.h` | gcd, ceil/floor div, pow2 | Leaf |
| `driver/magic_div.h` | Magic division | Leaf |
| `python/operations/utility.py` | Integer division macros, gcd, log2 | Leaf |
| `python/operations/conv.py` | Conv domain constants, conv_param_t | Leaf, no codegen import |
| `python/perf_advisor.py` | Stub placeholder | Unused |

### Shared Infrastructure

| Infrastructure | Python Side | C++ Side | Sync Mechanism |
|---|---|---|---|
| **Tunable parameter object** | `igemm_gtc_tunable_parameter_t` (`igemm_base.py`) | `igemm_gtc_tunable_t` (`igemm_gtc_base.h`) | **Manual** — no code generation bridges them |
| **Kernel name mangling** | `igemm_gtc_encode_kernel_name()` (`igemm_base.py`) | `igemm_gtc_encode_kernel_name()` (`igemm_gtc_base.h`) | **Manual** — must produce identical strings |
| **Config parsing** | `config_parser_t` (`python/codegen/config_parser.py`) | `config_parser_t` (`driver/config_parser.h`) | **Manual** — same INI format, independent implementations |
| **MC printer / dependency injection** | `mc_asm_printer_t.inject()` / `mc_base_t` (`mc.py`) | — | Python-only |
| **Global MC state** | `mc_set_current()` / `mc_get_current()` (`mc.py`) | — | Python-only |
| **Magic division** | `igemm_division_magic()` (`igemm_base.py`) | `magic_div_u32_gen()` (`magic_div.h`) | **Manual** — concept shared, implementations independent |

### Cyclic Dependencies

**None found.** The dependency graph is a strict DAG:

```
python/codegen/  (Layer 0: symbol, mc, config_parser)
       ↑
python/operations/  (Layer 1: imports from codegen)
       ↑
python/igemm/  (Layer 2: imports from codegen + operations + igemm_base)
       ↑
python/codegen_driver.py  (Layer 3: imports from igemm + codegen + operations)
       ↑
igemm_codegen.py  (Layer 4: imports from python)

driver/  (independent C++ tree, connected only by .config schema + kernel name convention)
script/  (subprocess invocations of igemm_codegen.py + conv_driver.exe, no python/ imports)
test/  (python tests: `from python import *`; C++ tests: include driver/ headers)
```

Within `python/operations/`, there are sibling-level cross-imports (e.g., `coalescing_store → thread_mapping + global_memory + shared_memory`) but no cycles.

**Latent issue (not a cycle):** `igemm_codegen.py` seq-mode calls `sequence_driver(...)` but the function is `igemm_sequence_driver` — name mismatch. Not triggered on the active gfx1250 path.

### Architectural Boundaries

```
┌─────────────────────────────────────────────────────────────────────┐
│                        igemm_codegen.py (CLI)                        │
├─────────────────────────────────────────────────────────────────────┤
│  python/codegen/          │  python/operations/  │  python/igemm/    │
│  (assembler framework)    │  (main-loop drivers) │  (kernel gens)    │
│  ──────────────────────── │  ─────────────────── │  ──────────────── │
│  mc.py, instruction.py    │  wmma_main_loop.py   │  igemm_{fwd,bwd,  │
│  amdgpu.py, compile.py    │  coalescing_store_*  │    wrw}_gtc_wmma_ │
│  config_parser.py         │  *_mapping.py        │    nhwc.py        │
│  mbb.py, scheduler.py     │  global_memory.py    │  igemm_base.py    │
│                           │  shared_memory.py    │                   │
│  [Layer 0]                │  [Layer 1]           │  [Layer 2]        │
├───────────────────────────┴─────────────────────┴───────────────────┤
│              python/codegen_driver.py (orchestrator) [Layer 3]       │
│              python/host_driver.py (C++ driver builder)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    ┌─── Shared schema boundary ───────────────────────────────┐    │
│    │  .config file format  │  kernel name mangling convention  │    │
│    │  (Python parser)       │  (Python + C++ must match)       │    │
│    └────────────────────────┴──────────────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────┴────────────────────────────────┐    │
│  │                    driver/ (C++ host driver)                │    │
│  │  conv_driver.cpp  │  igemm_gtc_base.h  │  igemm_*_driver.h  │    │
│  │  args.h           │  config_parser.h   │  naive_conv.h      │    │
│  │  magic_div.h      │  common.h          │  gpu_naive_conv/   │    │
│  │  gpu_tensor_cast/ │  gpu_tensor_reorder/│  gpu_batched_     │    │
│  │                   │                    │    transpose/      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                    hipModuleLoad(hsaco)                              │
│                    hipExtModuleLaunchKernel                          │
├─────────────────────────────────────────────────────────────────────┤
│  script/ (subprocess → igemm_codegen.py + conv_driver.exe)          │
│  test/   (python tests: from python import *; C++ tests: driver/)   │
│  config/ (.config files, consumed by both Python + C++)             │
│  docs/   (design docs, no code)                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key boundaries:**

1. **Python ↔ C++**: Connected only by (a) `.config` file format (shared schema, independent parsers) and (b) kernel symbol name convention (`igemm_gtc_encode_kernel_name` must match). No code generation bridges them — **manual synchronization is required** for any tunable field or name-mangling change.

2. **Codegen ↔ ROCm toolchain**: `python/codegen/compile.py` shells out to `clang++`/`hipcc`/`llvm-objdump` at `ROCM_PATH=/home/sgundabo/rocm-10.1`. Hardcoded path.

3. **Driver ↔ HIP runtime**: All GPU operations via HIP module API (`hipModule*`). No compile-time ROCm link — driver loads pre-assembled HSACOs at runtime.

4. **Script ↔ Codegen/Driver**: Subprocess invocations only (`igemm_codegen.py`, `conv_driver.exe`, `rocm-smi`). No `python/` imports.

5. **Test ↔ Python/Driver**: Python tests import `from python import *` (in-process). C++ tests include `driver/` headers at compile time. Neither direction reverses.

6. **Operations ↔ Igemm**: Strictly one-directional — `python/igemm/` imports from `python/operations/`, never the reverse. Operations defines functor *slots*; igemm fills them with direction-specific closures.

7. **WMMA (gfx1250) ↔ Legacy (gfx908–950)**: Within `python/igemm/`, all generators inherit directly from `mc_base_t` — no shared intermediate base class. Shared logic lives in `igemm_base.py` as free functions. WMMA generators use `wmma_main_loop_t`; legacy generators use `mfma_main_loop_t` / `dotx_main_loop_t` or inline loops. The `codegen_driver_t.__init__` dispatches on `wmma_tile_m` key presence.

---

## 3. Key Design Patterns

### mc_base_t Dependency Injection

All generators, macros, and emitters inherit from `mc_base_t(mc)`. The constructor calls `mc.inject(self)`, which copies 11 method references from the central `mc_asm_printer_t` printer as bound methods:

```
other._emit               → printer.emit
other._emit_empty_line     → printer.emit_empty_line
other._emit_macro_indented → printer.emit_macro_indented
other._emit_macro_desc     → printer.emit_macro_desc
other._emit_front          → printer.emit_front
other._inc_indent          → printer.inc_indent
other._dec_indent          → printer.dec_indent
other._indent_context      → printer.indent_context
other._deferred_context    → printer.deferred_context
other._get_deferred        → printer.get_deferred
other._insert_unique       → printer.insert_unique
```

This gives any subclass direct emit access to the shared printer without inheriting from it. All emit calls go through one emitter, ensuring consistent indent and output ordering. The `_deferred_context()` / `_get_deferred()` mechanism allows macros and schedulers to capture text fragments for later placement.

### Global MC State

`mc_set_current(mc)` / `mc_get_current()` maintain a process-global pointer to the active `mc_asm_printer_t`. Instruction encoder singletons in `instruction.py` are stateless functors — they query `mc_get_current().arch_config.arch` to branch on architecture (< 1000 → pre-gfx10 syntax with explicit `vcc,`; >= 1000 → gfx10+ syntax). This avoids threading `mc` through every instruction call site.

### Functor-Based Main Loop

Kernel generators define closure-returning methods (`global_load_a_functor()`, `shared_store_a_functor()`, etc.) that capture tunable-specific state. These closures are installed as slots on `ctrl_wmma_main_loop_t`. The generic `wmma_main_loop_t.emit()` drives the K-loop, calling the functors at the correct pipeline stages (load → store to LDS → barrier → load from LDS → WMMA compute → advance address). This separates loop *structure* (in `operations/`) from loop *content* (in `igemm/`).

### Manual Python ↔ C++ Mirror

The tunable parameter object and kernel name mangling function exist in both Python (`igemm_base.py`) and C++ (`igemm_gtc_base.h`) with no code generation bridge. Any new tunable field, name-mangling suffix, or semantic change requires synchronized edits to both sides. The C++ driver uses `igemm_gtc_encode_kernel_name()` to reconstruct the exact kernel symbol Python emitted, for `hipModuleGetFunction` lookup — a mismatch causes a silent `hipModuleGetFunction` failure.
