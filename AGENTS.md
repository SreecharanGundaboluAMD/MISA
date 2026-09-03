# Repository Guidelines

## Project Overview

**MISA** (a.k.a. **iGEMMgen**) is a Python-based code generator for **implicit-GEMM convolution kernels** targeting AMD GPUs. It emits hand-optimized AMD GPU assembly (GFX ISA) for the three convolution directions:

- `fwd` — forward convolution
- `bwd` — backward-input (data) gradient
- `wrw` — backward-weight gradient

The active development target is **gfx1250 / CDNA5** (MI350), using **WMMA** matrix-multiply-accumulate instructions (`v_wmma_f32_16x16x32_f16` / `_bf16`, `v_wmma_i32_16x16x64_iu8`). The repo also retains older paths for gfx908/gfx90a/gfx940/gfx950 (MAC, DLOPS, XDLOPS).

Pipeline: `.config` (tunable parameters) → `igemm_codegen.py` → assembly `.s` + per-tile `.inc` + `.hsaco` code object + `conv_driver.exe` host driver. The driver benchmarks and verifies kernels, accepting MIOpenDriver-compatible CLI arguments.

## Architecture & Data Flow

```
.config file  (INI: [codegen] header + [igemm_{dir}_gtc] tunable sections)
  │
  ├─[flat mode]  python3 igemm_codegen.py <config>
  │    │  config_parser_t.parse() → config_content_t
  │    │  igemm_try_expand_tunable_content()  (fan out list-valued tensor_layout)
  │    │
  │    ├─ host_driver():  compile_host_t → conv_driver.exe
  │    │    (+ compile_hip_t: naive_conv.hsaco, igemm_gtc_tensor_cast.hsaco)
  │    │    injects -DIGEMM_CONFIG_FILE / -DIGEMM_HSACO / -DUSE_{HALF,INT8,BF16,INT4}
  │    │
  │    └─ igemm_flatten():
  │         mc_asm_printer_t(emitter, arch) → mc_set_current()
  │         codegen_driver_t(mc, tunable_dicts)(split_kernel=...)
  │           __init__: dispatch on direction + wmma_tile_m → generator class list
  │           do_emit:  hsa_header → global_macro (gfx1250: no-op) → igemm_macro
  │                     → emit_igemm_kernel  (each kernel.emit_kernel_body())
  │                       → raw asm strings via mc emitter → .s + .inc files
  │           do_compile: compile_asm_t  →  clang++ -x assembler -mcpu=gfx1250  → .hsaco
  │                       compile_disass_t →  llvm-objdump --disassemble         → .disass.s
  │
  └─[seq mode]  sequence_driver()  — gfx908/XDLOPS only, not used for gfx1250

Runtime:  ./conv_driver.exe <mode> -n .. -c .. -H .. -W .. -k .. -y .. -x .. ... -F <bitmask> -V <0|1>
  main(): hipModuleLoad(hsaco) → config_parser → igemm_gtc_tunable_from_config
    <mode> arg → driver_data_type   ← PRECISION GOTCHA (see below)
    for each direction (forw bitmask):
      igemm_{dir}_gtc_t(module, ...) → launch_conv_driver(...)
        for each tunable: driver->run()  → hipModuleGetFunction → grid/block
          → igemm_launch_kernels (warmup + repeat)  → optional verify vs naive_conv
```

**Key architectural points:**

- **Layered assembly-string emitter.** The codegen does not use a compiler IR — it emits raw AMDGCN assembly text. `python/codegen/` is a generic assembler framework (symbols, macros, instruction encoders, machine-code printer, basic-block scheduler). `python/igemm/` contains the convolution-specific generators that call into it.
- **Dependency injection via `mc.inject()`.** `mc_base_t.__init__(mc)` calls `mc.inject(self)`, which copies `_emit`/`_deferred_context` from the central `mc_asm_printer_t` onto any object. This is how kernel generators gain `_emit(...)` without inheritance from the printer.
- **Process-global MC.** `mc_set_current(mc)`/`mc_get_current()` set a global current printer, read by `python/codegen/instruction.py` to branch on arch version (`< 1000` → gfx9 naming, else gfx10+).
- **Generator selection** (`codegen_driver_t.__init__`): inspects `tunable_dicts[0]['direction']` + presence of `wmma_tile_m` + `tensor_layout` to pick the class — WMMA (gfx1250) → `igemm_{dir}_gtc_wmma_nhwc_t`; nhwc → `_{dir}_gtc_nhwc_t`; nchwc → `_nchwc_t`; else base `_{dir}_gtc_t`.
- **WMMA generators emit everything inline** — `emit_global_macro()` is a no-op (`pass`) for gfx1250; no shared global macros. Older archs emit shared FMA/dotx macros.
- **C++ driver mirrors Python.** `driver/igemm_gtc_base.h` defines `igemm_gtc_tunable_t` (C++ twin of `igemm_gtc_tunable_parameter_t`) and `igemm_driver_base_t` (abstract). Direction subclasses `igemm_{fwd,bwd,wrw}_gtc_t` implement `get_block_size`/`get_grid_size`/`run`. GEMM dimension mapping: fwd M=N·ho·wo N=K/group; bwd M=K/group N=N·hi·wi; wrw M=K/group N=C/group.

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `python/codegen/` | Low-level assembler framework: `config_parser`, `amdgpu` (arch/precision enums), `mc` (machine-code printer + emitters), `instruction` (per-instruction encoders), `macro`, `symbol`, `mbb` (machine basic blocks), `scheduler` (interleaving), `compile` (shells out to ROCm toolchain) |
| `python/igemm/` | Kernel generators per direction/layout. Active gfx1250: `igemm_{fwd,bwd,wrw}_gtc_wmma_nhwc.py`, `igemm_base.py`. Legacy: `_{dir}_gtc.py` (NCHW/MAC), `_{dir}_gtc_nhwc.py` (DLOPS/XDLOPS), `igemm_fwd_gtc_nchwc.py`, `igemm_upsampling_clear.py` |
| `python/operations/` | Operation helpers (e.g. `coalescing_store_wmma.py`, `wmma_main_loop.py`) — functor-based main-loop drivers called by the generators |
| `python/` (root) | `codegen_driver.py` (top orchestrator), `sequence_driver.py` (seq mode, gfx908-only), `host_driver.py` (builds conv_driver.exe), `perf_advisor.py` |
| `driver/` | C++ host driver: `conv_driver.cpp` (main + launch loop), `igemm_gtc_base.h` (tunable + abstract base), `igemm_{fwd,bwd,wrw}_gtc_driver.h` (per-direction drivers), `args.h` (MIOpenDriver CLI parser), `config_parser.h`, `common.h`, helpers (`tensor_transpose.h`, `utility.h`, `use_tracker.hpp`), `perf/gmap.cpp` |
| `config/` | `.config` files (INI-like). Naming: `igemm_{dir}_gtc_{arch}_{layout}_{precision}_{variant}.config` |
| `script/` | Python automation: config generation, master-config unioning, build filtering, benchmarking, coverage analysis, stream-K sweeps |
| `test/` | Python codegen unit tests + C++ HIP kernel verification tests |
| `docs/` | Design docs (gfx1250 optimization phases, WMMA layout, stream-K, profiling, backlog) |
| `out/` | Generated artifacts (gitignored): `.s`, `.inc`, `.hsaco`, `conv_driver.exe` |

## Development Commands

### Build a kernel + driver (flat mode — the gfx1250 path)

```bash
python3 igemm_codegen.py config/igemm_fwd_gtc_gfx1250_nhwc_fp16_all.config
# optional: -d <out_dir>  (default 'out'), -s (split-kernel: one .s per kernel)
python3 igemm_codegen.py -d /tmp/my_out config/igemm_wrw_gtc_gfx1250_nhwc_bf16_all.config
```

Output in `out/`: `<configname>.s`, per-kernel `<configname>_<M>x<N>x<K>.inc`, `<configname>.hsaco`, `conv_driver.exe`, `naive_conv.hsaco`, `igemm_gtc_tensor_cast.hsaco`.

### Run / benchmark a kernel

```bash
# CRITICAL: mode arg must match kernel precision!
#   conv=fp32  convfp16=fp16  convbfp16=bf16  convint8=int8  convint4=int4
./out/conv_driver.exe convfp16  -n 128 -c 1024 -H 17 -W 17 -k 1024 \
    -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F 1 -V 1 \
    --in_layout NHWC --fil_layout NHWC --out_layout NHWC
#   -F bitmask: 1=fwd 2=bwd 4=wrw (or: fwd=1, bwd=2, wrw=4)
#   -V 1 = verify correctness, -V 0 = timing only
#   -t 1 = verify on (alias in some test drivers)
```

Useful env vars (`driver/ENVIRONMENT.md` has the full list):

```bash
IGEMM_WARMUP=5 IGEMM_REPEAT=20 ./out/conv_driver.exe convbfp16 ...
IGEMM_LOG_FASTEST_CONFIG=1 ./out/conv_driver.exe ...        # print fastest config
IGEMM_HSACO=/path/to/custom.hsaco ./out/conv_driver.exe ...
IGEMM_RUN_ONLY_KERNEL=<kernel_name> ./out/conv_driver.exe ...
IGEMM_SCLK_MHZ=1700 ./out/conv_driver.exe ...               # efficiency calc
```

### Config generation & filtering workflow (gfx1250)

```bash
python3 script/generate_all_configs.py --write          # per-tile combinatorial _all.config (encodes ~25 mutual-exclusion rules)
python3 script/build_gfx1250_master_configs.py --write  # union narrow configs → master _all.config (re-run after adding any config)
python3 script/build_and_filter_configs.py --write      # build each, drop sections that fail assembly (VGPR overflow)
```

### Benchmarking

```bash
python3 script/benchmark_gfx1250_vs_miopen.py [--direction {fwd,bwd,wrw,all}] [--rebuild] [--verify] [--markdown-out FILE]
python3 script/benchmark_gfx1250_vs_gfx950_diverse.py [--direction {fwd,bwd,wrw,all}]
```

### Tests

```bash
# Python codegen unit tests (no GPU)
python3 test/unittest.py
python3 test/tensor_transformation/tensor_transformation_test.py
python3 test/twiddle/twiddle_test.py            # needs ROCm toolchain

# C++ kernel verification tests (GPU + ROCm), run from repo root
sh test/naive_conv/build.sh
sh test/nchw_nhwc_transpose/build.sh && ./out/nchw_nhwc_transpose.exe <N> <C> <H> <W>
sh test/tensor_reorder/build.sh
sh test/persistent_workgroup/build.sh && sh test/persistent_workgroup/run.sh
sh test/inference/build.sh && sh test/inference/run_inference.sh
```

## Code Conventions & Common Patterns

### Config format

Custom INI parser (`python/codegen/config_parser.py`), **not** Python's `configparser`. One `[codegen]` header + one or more `[igemm_{dir}_gtc]` sections. Value types auto-inferred: int, float, quoted string, `[a,b,c]` list, `(start,end,step)` range, `{k=v}` dict. Comments: `#` or `;` (trailing stripped). Duplicate keys within a section are rejected.

**`[codegen]` fields:** `arch` (e.g. `'gfx1250'`), `code_object` (`'cov3'` = AMDGPU Code Object V3, the `.amdhsa_kernel` rodata format; `'cov2'` = legacy), `mode` (`'flat'`/`'flatten'` for gfx1250; `'seq'` is gfx908-only).

**`[igemm_*_gtc]` required fields:** `gemm_m_per_block`, `gemm_n_per_block`, `gemm_k_per_block`, `wmma_tile_m`, `wmma_tile_n` (always 16), `wmma_repeat_m`, `wmma_repeat_n`, `tensor_a_thread_lengths`, `tensor_a_cluster_lengths`, `tensor_b_thread_lengths`, `tensor_b_cluster_lengths`, `direction`, `precision`, `tensor_layout`, `nxb`, `nxe`.

**Optional fields** (default 0 via `utility_dict_with_default_t`, in `python/igemm/igemm_base.py`): `wavefront_size` (gfx1250 forces 32), `cumode`, `direct_store`, `gemm_k_global_split`, `lds_double_buffer`, `async_global_load`, `tdm_global_load`, `main_loop_interleave`, `wmma_setprio`, `wmma_acc_f16`, `wmma_acc_bf16`, `atomic_pack_bf16`, `gsplit_stagger`, `wmma_m_tail`, `wmma_n_tail`, `wmma_k_tail`, `wmma_epilogue_chunked`, `wmma_acc_high_bank`, `wrw_streamk`, `wrw_reduction_kernel`, `multihead`, `merge_e`, `local_prefetch_num`.

**Variant glossary** (suffix in config filename): `direct` (direct_store=1), `_all` (master union), `_64x64`/`_128x128`/`_32x32` (tile family), `streamk` (wrw_streamk=1), `mntail`/`mtail`/`ntail`/`ktail` (tail handling), `tdm` (tdm_global_load=1), `saddr`, `wsred` (wrw_reduction_kernel=1), `gsplit` (gemm_k_global_split=1), `k2x`/`k4x` (k-sub-loop), `bf16acc`/`f16acc` (narrowed accumulate), `setprio`, `stagger`, `pkatomic`.

### Naming conventions

- **Configs:** `igemm_{dir}_gtc_{arch}_{layout}_{precision}_{variant}.config`
- **Kernel symbols** (mangled, Python + C++ mirror): `igemm_{dir}_gtcw_nhwc_{prec}_bx{nxb}_ex{nxe}_bt{M}x{N}x{K}_wt{tm}x{tn}_wr{rm}x{rn}_ta{...}_tb{...}_...` — see `igemm_gtc_encode_kernel_name` in `python/igemm/igemm_base.py` and `driver/igemm_gtc_base.h`.
- **Generated files:** `<configname>.s`, `<configname>_<M:03>x<N:03>x<K:03>.inc`, `<configname>.hsaco`, `<configname>.disass.s`.

### Python codegen patterns

- **`mc_base_t` mixin:** all generators subclass `mc_base_t` and receive `_emit`/`_deferred_context` via `mc.inject(self)`. Call `self._emit("v_add_nc_u32 ...")` to emit assembly lines.
- **Functor-based main loop:** generators define `global_load_a_functor`/`shared_store_a_functor`/`shared_load_a_functor`/`move_slice_window_a_functor` (and `_b_` variants). These are called by `ctrl_wmma_main_loop_t` from `python/operations/wmma_main_loop.py`.
- **Kernel body structure:** `emit_kernel_body()` = `emit_kernel_prologue()` → `emit_kernel_tap_loop()` → `emit_kernel_epilogue()`. The tap loop wraps a static WMMA K-main-loop in a runtime y·x tap iteration (multi-tap, stride, dilation, group>1).
- **Tunable parameter object:** `igemm_gtc_tunable_parameter_t(tunable_dict)` — dispatches on `get_igemm_gtc_fma_type()` (MAC/DLOPS/XDLOPS/WMMA) to read fma-specific fields. All optional fields default to 0 for byte-identical backward compatibility.
- **Instruction encoders** (`python/codegen/instruction.py`): module-level singletons (e.g. `v_madmk`, `v_add_nc_u32`) called as `v_madmk(vdst, src0, imm32, vsrc1)`. They branch on `mc_get_current().arch_config.arch` for arch-specific syntax.

### C++ driver patterns

- **Abstract base + direction subclasses:** `igemm_driver_base_t` (igemm_gtc_base.h) with pure virtuals `get_block_size`/`get_grid_size`/`tunable_is_valid`/`run`/`get_gks_list`. Subclasses per direction.
- **WMMA block size:** `waves_per_m * waves_per_n * 32` (wave32 on gfx1250). Grid folds group into grid_y (no workgroup_id_z on gfx1250).
- **Split-K (`gemm_k_global_split`):** atomic-add epilogue into `p_out`; `p_out` is re-zeroed before each launch. wrw's primary path.
- **Verification:** `USE_GPU_NAIVE_CONV` (default) → `naive_conv.hsaco` GPU reference; compared with per-precision NRMS tolerance.

### Error handling

- Python codegen: `assert` for invariant violations (e.g. `assert type(mc) is mc_asm_printer_t`). `sys.exit(-1)` on config parse errors.
- Build failures: `compile_*_t.compile()` returns `bool`; `assert False` on failure. Build output is printed to stdout.
- Driver: tunables failing `tunable_is_valid` are silently skipped. `IGEMM_ASSERT_WHEN_INVALID=1` makes it abort instead.

### Formatting

- `driver/.clang-format`: `IndentWidth: 4` (applies to C++).
- Python: no enforced formatter; 4-space indent, snake_case, `_t` suffix on classes (e.g. `igemm_fwd_gtc_wmma_nhwc_t`).

## Important Files

| File | Role |
|------|------|
| `igemm_codegen.py` | **CLI entry point.** `python3 igemm_codegen.py <config> [-d dir] [-s] [-output list]` |
| `python/codegen_driver.py` | `codegen_driver_t` — top orchestrator: generator selection + emit + compile |
| `python/host_driver.py` | `host_driver()` — builds `conv_driver.exe` + `naive_conv.hsaco` + `tensor_cast.hsaco` |
| `python/sequence_driver.py` | `sequence_driver()` — seq mode (gfx908/XDLOPS only) |
| `python/igemm/igemm_base.py` | `igemm_gtc_tunable_parameter_t` (tunable contract), kernel name mangling, shared helpers |
| `python/igemm/igemm_fwd_gtc_wmma_nhwc.py` | Active gfx1250 fwd WMMA generator (~2046 lines) |
| `python/igemm/igemm_bwd_gtc_wmma_nhwc.py` | Active gfx1250 bwd WMMA generator (~1616 lines) |
| `python/igemm/igemm_wrw_gtc_wmma_nhwc.py` | Active gfx1250 wrw WMMA generator (~1911 lines; includes stream-K path) |
| `python/codegen/compile.py` | `compile_asm_t`/`compile_hip_t`/`compile_host_t`/`compile_disass_t` — ROCm toolchain invocations |
| `python/codegen/mc.py` | `mc_asm_printer_t` (central printer), `mc_base_t` (inject mixin), emitters |
| `python/codegen/amdgpu.py` | Arch/precision enums, `amdgpu_arch_config_t`, code-object constants |
| `python/codegen/config_parser.py` | `config_parser_t` — INI → `config_content_t` |
| `driver/conv_driver.cpp` | C++ host driver `main()` + `launch_conv_driver` benchmark loop |
| `driver/igemm_gtc_base.h` | `igemm_gtc_tunable_t` (C++ tunable), `igemm_driver_base_t` (abstract base), `igemm_launch_kernels` |
| `driver/igemm_{fwd,bwd,wrw}_gtc_driver.h` | Per-direction driver subclasses: block/grid/run |
| `driver/args.h` | MIOpenDriver-style CLI arg parser (`args_t`) |
| `config/igemm_fwd_gtc_gfx1250_nhwc_fp16_all.config` | Master gfx1250 WMMA flat config (union of all validated fp16 fwd tunable combinations; use this for benchmarking and quick-start — the narrow `_direct` config picks the wrong `lds_double_buffer`/`direct_store` combo on large 1×1 shapes, see `docs/gfx1250_wmma_perf_report_2026-09-02.md` OPT-3) |
| `driver/ENVIRONMENT.md` | Full env-var reference for `conv_driver.exe` |
| `docs/gfx1250_wmma_layout.md` | **Master reference** (368KB): empirically-verified WMMA register layout + full phase-by-phase dev history (Phases 1–61+) |
| `docs/gfx1250_fp32_wmma_occupancy_race.md` | **Known issue, mechanism characterized.** fp32 WMMA (`v_wmma_f32_16x16x4_f32`) silently corrupts output at high occupancy unless `lds_double_buffer=1` — root cause traced to a standalone repro showing `s_barrier_wait` doesn't reliably fence a wave's *last lane's* LDS write — see Known Issues below and `docs/gfx1250_fp32_wmma_race_repro/` |

## Known Issues

- **fp32 WMMA (`gemm_k_per_block=4`) requires `lds_double_buffer=1`, unconditionally, on every config.** Single-buffered LDS with `v_wmma_f32_16x16x4_f32` (this arch's only pure-fp32 WMMA form) produces silent wrong-answer results (`valid:n`) once enough workgroups run concurrently (empirically ~1500+ for 64x64 tiles, ~12000+ for 128x128) — small/benchmark-scale shapes will pass and hide the bug. Confirmed on both `fwd` and `bwd`, both tile sizes; `wrw` is presumed affected (same code path) and patched proactively. A standalone, WMMA-independent repro (`docs/gfx1250_fp32_wmma_race_repro/`) traced the mechanism: `s_barrier_signal`/`s_barrier_wait` (with `s_wait_dscnt 0x0` on the writer) does not reliably make a wave's **last lane's** (lane 31) LDS write visible to other waves by the time they resume past the barrier, once occupancy is high enough — every observed failure is stale by *exactly* one loop iteration and >99.9% land on the last-lane slot. Not fixed by `V_NOP`/`s_setprio` around the WMMA burst (both tried, zero effect) — it isn't WMMA-specific at all, WMMA's tight fp32 K=4 loop just makes it easy to expose. All fp32 configs in `config/` already set `lds_double_buffer=1` as a result — **do not remove it, and add it to any new fp32 WMMA config you create.** Full writeup and the standalone reproducer for hardware-team escalation: `docs/gfx1250_fp32_wmma_occupancy_race.md`, `docs/gfx1250_fp32_wmma_race_repro/README.md`.

## Runtime/Tooling Preferences

- **Python ≥ 3.6** (f-strings used throughout). No package manager / no `requirements.txt` — standard library only.
- **ROCm toolchain required** at `ROCM_PATH` (hardcoded `/home/sgundabo/rocm-10.1` in `python/codegen/compile.py`). Provides: `clang++` (assembler: `-x assembler -target amdgcn--amdhsa -mcpu=gfx1250`), `hipcc` (host + HIP: `-std=c++17`), `llvm-objdump` (disassembly), `clang-offload-bundler`, `llvm-readobj`.
- **`half.hpp` required for fp16:** install from [pfultz2/half 1.12.0](https://github.com/pfultz2/half/archive/1.12.0.tar.gz) → `$ROCM_PATH/include/half/half.hpp` (or `/usr/local/include/half.hpp`). Gated by `-DUSE_HALF` (set automatically when config contains fp16 sections).
- **Target arch:** gfx1250 (CDNA5). All gfx1250 configs use `code_object='cov3'`, `tensor_layout='nhwc'`, `wavefront_size=32`, WMMA 16×16×32 instructions.
- **No Node/Bun.** No `package.json`. Pure Python + C++/HIP.
- **`amd-instinct-cdna5-instruction-set-architecture.BIGFILE_DO_NOT_READ`** — the gfx1250 ISA reference; per its name, do not read it inline (1.3MB).

## Testing & QA

### No unified test runner

There is no `make test`, no pytest config, no CI script. Each test is invoked manually. The Python `test/unittest.py` has a `run_all_unittest()` dispatcher but it's a developer scratch-pad (most tests commented out; verification is primarily visual via `print(mc.emitter.get_buffer())`).

### Python codegen tests (no GPU)

- `test/unittest.py` — plain functions (no framework). Exercises `mc_asm_printer_t`, `inst_ds_read2_likely_t`, `ctrl_coalescing_store_t`, `igemm_coalescing_store_t`, xdlops/dotx variants, `macro_base_t` subclassing. Run: `python3 test/unittest.py`.
- `test/tensor_transformation/tensor_transformation_test.py` — pure-Python tensor-descriptor algebra with asserts. Run: `python3 test/tensor_transformation/tensor_transformation_test.py`.

### C++ HIP kernel tests (GPU + ROCm required)

Each has a `build.sh` (run from repo root); all use relative paths and write to `./out/`.

| Test | Verifies | Arch |
|------|----------|------|
| `test/naive_conv/` | GPU naive-conv reference kernel (the golden other tests use) | gfx908 |
| `test/naive_tiled_conv/` | Tiled conv = bit-exact vs naive (CPU-only) | gfx908 |
| `test/nchw_nhwc_transpose/` | NCHW↔NHWC transpose kernels | gfx908 |
| `test/tensor_reorder/` | General tensor-reorder kernels (23 dim permutations) | gfx1030 |
| `test/persistent_workgroup/` | Persistent-workgroup fwd conv (checked-in `.s`, not codegen'd at test time) | gfx1030 |
| `test/inference/` | Inference fwd-btm kernels (fp16/int8, batch=1) | gfx1030 |
| `test/twiddle/` | FFT butterfly codegen (`python.cellfft`) → compile → GPU run vs CPU Cooley-Tukey | gfx908 |

### Verification approach

- C++ tests compare GPU output against CPU reference (`naive_conv_*`, `fft_cooley_tukey_r`) via `valid_vector<T>` (NRMS or abs-delta tolerance, typically 1e-6 to 1e-4).
- `test/common/utility.h` provides shared `rand_vec`/`valid_vector`/`valid_vector_nrms`, but most tests redefine their own local copies (duplicated boilerplate).
- The main `conv_driver.exe` verifies against `naive_conv.hsaco` (GPU reference, `-DUSE_GPU_NAIVE_CONV`) with `-V 1`.

### Coverage expectations

No formal coverage tracking. Correctness is the gate: a kernel is "done" when `conv_driver.exe -V 1` reports `valid:y` across the target shape set. Benchmarking scripts (`script/benchmark_gfx1250_vs_*.py`) report TFLOPS and ratios vs MIOpen but do not assert performance thresholds.
