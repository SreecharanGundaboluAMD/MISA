#!/usr/bin/env python3
"""
Generate comprehensive _all.config files per tile shape with every valid tunable
combination for gfx1250 WMMA.

Each output file covers ONE tile shape for ONE (direction, precision) pair,
keeping each built .hsaco under the assembler's branch-range limit (±32KB
s_cbranch/s_branch immediate — the old monolithic _all.config with hundreds of
sections hit `branch size exceeds simm16`).

Both benchmark scripts are updated to try all tile-shape _all.config files
per (direction, precision) → conv_driver.exe searches each file's candidates
independently and the script reports the overall fastest.

Usage:
    python3 script/generate_all_configs.py [--write]
"""
import argparse
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import product


# Sentinel: a small number of flags have a NONZERO construct-time default for part
# of their domain (e.g. ds_load_tr_b defaults to 1 for bwd/wrw fp16/bf16) -- the
# generic "vals[k]==0 -> omit the override line, rely on the constructor's own
# default" convention used by every other flag below would then make BOTH bit
# values resolve to the SAME tunable there (found the hard way: a real assembler
# "symbol already defined" collision from two combos producing byte-identical
# kernels under different names). _FORCE_ZERO marks "explicitly write/merge the
# literal value 0" instead of "omit"; see the ds_load_tr_b remap in gen_combos().
_FORCE_ZERO = object()
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, 'config')

# Phase 5 step 1 (gfx1250_tuning_refactor_plan.md): legality is determined by
# actually constructing the real Python objects (igemm_gtc_tunable_parameter_t,
# then the direction's WMMA generator class) and catching AssertionError --
# not a hand-maintained parallel copy of the same rules. This is the exact
# same construction igemm_codegen.py itself performs before emitting a kernel,
# so "constructs without error" here means the same thing it means for a real
# build. See docs/gfx1250_tunable_exclusions.md for the resulting catalog of
# what actually gets rejected and why.
sys.path.insert(0, REPO_ROOT)
from python.igemm.igemm_base import igemm_gtc_tunable_parameter_t
from python.igemm.igemm_fwd_gtc_wmma_nhwc import igemm_fwd_gtc_wmma_nhwc_t
from python.igemm.igemm_bwd_gtc_wmma_nhwc import igemm_bwd_gtc_wmma_nhwc_t
from python.igemm.igemm_wrw_gtc_wmma_nhwc import igemm_wrw_gtc_wmma_nhwc_t
from python.codegen.mc import mc_asm_printer_t, mc_emit_to_string_t, mc_set_current
from python.codegen.amdgpu import (amdgpu_arch_config_t, amdgpu_string_to_arch,
                                    amdgpu_string_to_codeobj, AMDGPU_PRECISION_FP32)
from python.codegen.config_parser import config_parser_t
from python.operations.utility import (macro_mdiv_u32_vs_t, macro_mdiv_u32_rem_vs_t,
                                        macro_mdiv_u32_ss_t, macro_mdiv_u32_rem_ss_t)
import subprocess
import tempfile

ROCM_PATH = '/home/sgundabo/rocm-10.1'
_COMMON_MACROS = [macro_mdiv_u32_vs_t, macro_mdiv_u32_rem_vs_t, macro_mdiv_u32_ss_t,
                  macro_mdiv_u32_rem_ss_t]

_GEN_CLASS = {
    'fwd': igemm_fwd_gtc_wmma_nhwc_t,
    'bwd': igemm_bwd_gtc_wmma_nhwc_t,
    'wrw': igemm_wrw_gtc_wmma_nhwc_t,
}

_ARCH = amdgpu_arch_config_t({
    'arch'          :   amdgpu_string_to_arch('gfx1250'),
    'data_type'     :   AMDGPU_PRECISION_FP32,
    'code_object'   :   amdgpu_string_to_codeobj('cov3'),
})
_MC = mc_asm_printer_t(mc_emit_to_string_t(), _ARCH)
mc_set_current(_MC)

# (direction, precision, tile_m, tile_n, gemm_k, source_config)
BASE_SECTIONS = [
    # fwd
    ('fwd', 'fp16', 128, 128, 32, 'config/igemm_fwd_gtc_gfx1250_nhwc_fp16.config'),
    ('fwd', 'fp16', 64,  64,  32, 'config/igemm_fwd_gtc_gfx1250_nhwc_fp16.config'),
    ('fwd', 'fp16', 128, 64,  32, 'config/igemm_fwd_gtc_gfx1250_nhwc_fp16_128x64.config'),
    ('fwd', 'fp16', 64,  128, 32, 'config/igemm_fwd_gtc_gfx1250_nhwc_fp16_64x128.config'),
    ('fwd', 'bf16', 128, 128, 32, 'config/igemm_fwd_gtc_gfx1250_nhwc_bf16.config'),
    ('fwd', 'bf16', 64,  64,  32, 'config/igemm_fwd_gtc_gfx1250_nhwc_bf16.config'),
    ('fwd', 'bf16', 128, 64,  32, 'config/igemm_fwd_gtc_gfx1250_nhwc_bf16_128x64.config'),
    ('fwd', 'bf16', 64,  128, 32, 'config/igemm_fwd_gtc_gfx1250_nhwc_bf16_64x128.config'),
    ('fwd', 'fp32', 128, 128, 4,  'config/igemm_fwd_gtc_gfx1250_nhwc_fp32.config'),
    ('fwd', 'fp32', 64,  64,  4,  'config/igemm_fwd_gtc_gfx1250_nhwc_fp32.config'),
    ('fwd', 'fp32', 128, 64,  4,  'config/igemm_fwd_gtc_gfx1250_nhwc_fp32_128x64.config'),
    ('fwd', 'fp32', 64,  128, 4,  'config/igemm_fwd_gtc_gfx1250_nhwc_fp32_64x128.config'),
    # bwd
    ('bwd', 'fp16', 128, 128, 32, 'config/igemm_bwd_gtc_gfx1250_nhwc_fp16.config'),
    ('bwd', 'fp16', 64,  64,  32, 'config/igemm_bwd_gtc_gfx1250_nhwc_fp16.config'),
    ('bwd', 'fp16', 32,  32,  32, 'config/igemm_bwd_gtc_gfx1250_nhwc_fp16_32x32.config'),
    ('bwd', 'bf16', 128, 128, 32, 'config/igemm_bwd_gtc_gfx1250_nhwc_bf16.config'),
    ('bwd', 'bf16', 64,  64,  32, 'config/igemm_bwd_gtc_gfx1250_nhwc_bf16.config'),
    ('bwd', 'bf16', 32,  32,  32, 'config/igemm_bwd_gtc_gfx1250_nhwc_bf16_32x32.config'),
    ('bwd', 'fp32', 128, 128, 4,  'config/igemm_bwd_gtc_gfx1250_nhwc_fp32.config'),
    ('bwd', 'fp32', 64,  64,  4,  'config/igemm_bwd_gtc_gfx1250_nhwc_fp32.config'),
    ('bwd', 'fp32', 32,  32,  4,  'config/igemm_bwd_gtc_gfx1250_nhwc_fp32_32x32.config'),
    # wrw
    ('wrw', 'fp16', 128, 128, 32, 'config/igemm_wrw_gtc_gfx1250_nhwc_fp16.config'),
    ('wrw', 'fp16', 64,  64,  32, 'config/igemm_wrw_gtc_gfx1250_nhwc_fp16.config'),
    ('wrw', 'bf16', 128, 128, 32, 'config/igemm_wrw_gtc_gfx1250_nhwc_bf16.config'),
    ('wrw', 'bf16', 64,  64,  32, 'config/igemm_wrw_gtc_gfx1250_nhwc_bf16.config'),
    ('wrw', 'fp32', 128, 128, 4,  'config/igemm_wrw_gtc_gfx1250_nhwc_fp32.config'),
    ('wrw', 'fp32', 64,  64,  4,  'config/igemm_wrw_gtc_gfx1250_nhwc_fp32.config'),
]

# Binary tunables to toggle combinatorially
# Phase 67: added saddr_global_load -- was previously only ever tested in its own
# bespoke single-feature _saddr.config, never combined with direct_store/tail-relief/
# lds_double_buffer/wmma_setprio/local_prefetch_num/epilogue_lds_pad in the searched
# corpus.
# Phase 7 (gfx1250_tuning_refactor_plan.md, per Phase 6's dominance study): added
# ds_load_tr_b, wmma_gap_hoist, wmma_l2_prefetch. Phase 6 (docs/gfx1250_dominance_study.md)
# found all three are genuine, non-trivial, shape-dependent tradeoffs -- not dominated,
# not redundant with the existing FLAGS combinations:
#   - ds_load_tr_b was previously an unconditional default for bwd/wrw fp16/bf16 (no
#     escape hatch searched); Phase 6 found real wins for BOTH values depending on shape
#     magnitude (bwd's own win region even reverses direction at more extreme shapes,
#     see the doc), so the =0 escape hatch needs to be reachable by the search.
#   - wmma_gap_hoist/wmma_l2_prefetch previously only existed via bespoke standalone
#     config files (never combined with tail-relief/saddr/setprio in the searched
#     corpus); Phase 6 found wmma_l2_prefetch's best measured result (+27-33% on a
#     long-K shape) was the COMBINATION of setprio+gaphoist+l2pf together, which the
#     old bespoke-file approach could never produce. is_valid()'s real construction
#     (not a hand-maintained rule) transparently rejects the illegal combinations for
#     all three (e.g. l2_prefetch + async/saddr/tdm/interleave, ds_load_tr_b on fwd or
#     fp32) via AssertionError -- no special-casing needed here, same as every other
#     flag below.
FLAGS = [
    'direct_store', 'gemm_k_global_split', 'wmma_m_tail', 'wmma_n_tail',
    'tdm_global_load', 'lds_double_buffer', 'wmma_setprio',
    'local_prefetch_num', 'main_loop_interleave', 'epilogue_lds_pad',
    'saddr_global_load', 'ds_load_tr_b', 'wmma_gap_hoist', 'wmma_l2_prefetch',
]


def parse_sections(path):
    """Return [(section_name, [lines]), ...]."""
    with open(os.path.join(REPO_ROOT, path)) as f:
        text = f.read()
    sections = []
    for m in re.finditer(r'^\[(igemm_\w+)\]\s*\n(.*?)(?=^\[igemm_\w+\]|\Z)', text,
                         re.MULTILINE | re.DOTALL):
        sections.append((m.group(1), m.group(2).strip().split('\n')))
    return sections


def find_base(sections, direction, tile_m):
    """Find the section body with gemm_m_per_block == tile_m."""
    for name, body in sections:
        for line in body:
            m = re.match(r'gemm_m_per_block\s*=\s*(\d+)', line.strip())
            if m and int(m.group(1)) == tile_m:
                return name, [l + '\n' for l in body if l.strip()]
    return None, None


_typed_section_cache = {}   # src path -> {gemm_m_per_block: typed dict}

def get_base_typed_dict(direction, tile_m, src):
    '''Parse `src` once via the REAL config_parser_t (the same parser
    igemm_codegen.py itself uses) and cache every igemm_{direction}_gtc
    section's fully value-typed dict, keyed by gemm_m_per_block.'''
    if src not in _typed_section_cache:
        content = config_parser_t(os.path.join(REPO_ROOT, src)).parse()
        by_tile_m = {}
        for section in content.get_section(f'igemm_{direction}_gtc'):
            d = section.to_dict()
            if 'gemm_m_per_block' in d:
                by_tile_m.setdefault(d['gemm_m_per_block'], d)
        _typed_section_cache[src] = by_tile_m
    return _typed_section_cache[src].get(tile_m)


def _assembles(kernel):
    '''Actually assemble THIS ONE kernel via clang (hsa_header_t is a cov2-only
    no-op for our cov3 configs, so no other preamble is needed) -- fast (~30-
    40ms), catches real assembler-only failures a Python-level check cannot
    (confirmed real-world hit: "register index is out of range" for fwd's
    128x64 asymmetric tile + wmma_n_tail, on every precision -- passes every
    Python-level assert, `row_repeat_b==1` holds for this tile, but a register
    formula somewhere still overflows). _COMMON_MACROS mirrors what
    codegen_driver.py's emit_igemm_macro() would otherwise register once
    per-file from each kernel's get_kernel_macros() plus the shared magic-
    division macros every WMMA generator uses but does not re-declare
    (igemm_fwd_gtc_wmma_nhwc_t.get_kernel_macros()'s own docstring: "already
    registered globally... do not need re-registration here").'''
    _MC.emitter.string_buffer = ''
    for cls in _COMMON_MACROS:
        cls(_MC).emit()
    if hasattr(kernel, 'get_kernel_macros'):
        for macro in kernel.get_kernel_macros():
            macro.emit()
    kernel.emit_kernel_symbol()
    kernel.emit_kernel_header()
    kernel.emit_kernel_body()
    kernel.emit_kernel_end()
    kernel.emit_kernel_amd_kernel_code_t()
    kernel.emit_kernel_footer()
    asm_text = _MC.emitter.get_buffer()
    _MC.emitter.string_buffer = ''
    fd, asm_path = tempfile.mkstemp(suffix='.s')
    hsaco_path = asm_path[:-2] + '.hsaco'
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(asm_text)
        cmd = [f'{ROCM_PATH}/llvm/bin/clang++', '-x', 'assembler',
               '-target', 'amdgcn--amdhsa', '-mcpu=gfx1250', asm_path, '-o', hsaco_path]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return p.returncode == 0
    finally:
        for pth in (asm_path, hsaco_path):
            if os.path.exists(pth):
                os.unlink(pth)


def is_valid(direction, base_dict, vals):
    '''Phase 5 step 1 (gfx1250_tuning_refactor_plan.md): legality is
    determined by actually constructing the real Python objects
    (igemm_gtc_tunable_parameter_t, then the direction's WMMA generator
    class), emitting the full kernel body, AND assembling it with the real
    ROCm toolchain (_assembles above) -- not a hand-maintained parallel copy
    of the same rules. This is the exact same construct-emit-assemble
    sequence igemm_codegen.py itself performs before shipping a kernel, so
    "succeeds here" means the same thing it means for a real build. Replaces
    a hand-rolled rule set that had already drifted out of sync more than
    once (ds_load_tr_b's promotion to unconditional; the saddr_global_load +
    wmma_n_tail bug this copy only caught two phases after the real
    constructor could have; a genuinely unbuildable wrw saddr_global_load
    section -- inheriting gemm_k_global_split from its base -- silently
    deemed "valid" by the old rules) and catches failure classes at three
    different layers: construction-time asserts, emission-time-only asserts
    (e.g. "interleave requires num_k_substeps>1"), and real assembler
    failures (e.g. "register index is out of range", found only by actually
    assembling every candidate). See docs/gfx1250_tunable_exclusions.md for
    the resulting catalog of what actually gets rejected and why.

    Mirrors main()'s actual text-generation merge below: a combinatorial
    bit=0 does NOT emit an override line into the generated .config (so it
    must not force-zero an already-nonzero base default either -- e.g. fp32
    base sections already set lds_double_buffer=1, and bit=0 must leave that
    alone, not silently violate COR-001).'''
    merged = dict(base_dict)
    merged['arch'] = 'gfx1250'
    for k, v in vals.items():
        if v is _FORCE_ZERO:
            merged[k] = 0
        elif v not in (0, 'SCOPE_SYS'):
            merged[k] = v
    try:
        tunable = igemm_gtc_tunable_parameter_t(merged)
        kernel = _GEN_CLASS[direction](mc_asm_printer_t(_MC.emitter, _MC.arch_config), tunable)
        if not _assembles(kernel):
            return False, None
        return True, kernel.name()
    except AssertionError:
        return False, None
    finally:
        # Never actually consumed (validation-only) -- reset so thousands of
        # combos don't grow one shared string buffer unboundedly.
        _MC.emitter.string_buffer = ''


def _check_combo(task):
    direction, precision, tile_m, tile_n, gemm_k, base_dict, vals = task
    ok, kname = is_valid(direction, base_dict, vals)
    return (direction, precision, tile_m, tile_n, gemm_k, vals, kname) if ok else None


def gen_combos():
    """Return all (dir, prec, tm, tn, gk, vals_dict) that pass is_valid(),
    checked in PARALLEL across CPUs -- each check independently constructs,
    emits, and assembles one candidate kernel (~30-40ms, dominated by the
    clang subprocess) -- embarrassingly parallel, no shared state between
    candidates (each worker process gets its own forked copy of _MC)."""
    tasks = []
    for direction, precision, tile_m, tile_n, gemm_k, src in BASE_SECTIONS:
        base_dict = get_base_typed_dict(direction, tile_m, src)
        if base_dict is None:
            print(f"WARNING: no {tile_m} section in {src}", file=sys.stderr)
            continue
        for bits in product([0, 1], repeat=len(FLAGS)):
            vals = {FLAGS[i]: bits[i] for i in range(len(FLAGS))}
            # local_prefetch_num: bit 0 -> 1, bit 1 -> 2
            vals['local_prefetch_num'] = 2 if vals['local_prefetch_num'] == 1 else 1
            # ds_load_tr_b: bit 0 -> 0 (omit, preserves the pre-Phase-7 default-ON
            # behavior for bwd/wrw fp16/bf16 byte-identically); bit 1 -> _FORCE_ZERO
            # (explicit `ds_load_tr_b = 0` override, the escape-hatch Phase 6 found
            # real wins for). See _FORCE_ZERO's module-level docstring.
            vals['ds_load_tr_b'] = _FORCE_ZERO if vals['ds_load_tr_b'] == 1 else 0
            tasks.append((direction, precision, tile_m, tile_n, gemm_k, base_dict, vals))

    nproc = min(64, os.cpu_count() or 1)
    results = []
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        for res in ex.map(_check_combo, tasks, chunksize=16):
            if res is not None:
                results.append(res)
    return results



def _extra_lines(base_body, vals):
    '''Non-default tunable override lines to append after the cloned base body.
    Skips any flag the base body ALREADY sets (regardless of value) -- config
    files are plain INI, and config_parser_t (like every real consumer,
    igemm_codegen.py included) rejects a duplicate key outright. This is a
    real, previously-latent bug: wrw's own base sections default
    gemm_k_global_split=1 and fp32's default lds_double_buffer=1 (COR-001) --
    toggling either flag ON via the combinatorial FLAGS loop used to try to
    emit a second, duplicate line for a key the base already set to the exact
    same value, which config_parser_t rejects at parse time. Never triggered
    before Phase 5 step 1 (gfx1250_tuning_refactor_plan.md) because the old,
    stricter is_valid() happened to never generate those specific combinations.'''
    base_keys = set()
    for line in base_body:
        s = line.strip()
        if not s or s.startswith('#') or s.startswith(';') or '=' not in s:
            continue
        base_keys.add(s.split('=', 1)[0].strip())
    extra = []
    for flag in FLAGS:
        val = vals.get(flag)
        if flag in base_keys:
            continue
        if val is _FORCE_ZERO:
            extra.append(f"{flag:25s} = 0\n")
        elif val is not None and val != 0 and val != 'SCOPE_SYS':
            extra.append(f"{flag:25s} = {val}\n")
    return extra


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()

    # Pre-parse all source configs
    src_cache = {}
    for entry in BASE_SECTIONS:
        _, _, _, _, _, src = entry
        if src not in src_cache:
            src_cache[src] = parse_sections(src)

    base_cache = {}
    for direction, precision, tile_m, tile_n, gemm_k, src in BASE_SECTIONS:
        key = (direction, precision, tile_m, tile_n, gemm_k)
        if key in base_cache:
            continue
        sname, sbody = find_base(src_cache[src], direction, tile_m)
        if sbody is None:
            print(f"WARNING: no {tile_m}x{tile_n} section in {src}", file=sys.stderr)
            continue
        base_cache[key] = sbody

    # Generate all combinations
    combos = list(gen_combos())
    print(f"Total valid combinations: {len(combos)}")

    # Group by (direction, precision, tile_m, tile_n, gemm_k)
    per_tile = defaultdict(list)
    for direction, precision, tile_m, tile_n, gemm_k, vals, kname in combos:
        per_tile[(direction, precision, tile_m, tile_n, gemm_k)].append((vals, kname))

    total_files = 0
    total_sections = 0
    for (direction, precision, tile_m, tile_n, gemm_k), combos_list in sorted(per_tile.items()):
        # Output file name: igemm_{dir}_gtc_gfx1250_nhwc_{prec}_{tm}x{tn}_all.config
        out_name = f'igemm_{direction}_gtc_gfx1250_nhwc_{precision}_{tile_m}x{tile_n}_all.config'
        out_path = os.path.join(CONFIG_DIR, out_name)

        # Read existing file header if present
        codegen_header = [
            '[codegen]\n', "arch = 'gfx1250'\n", "code_object = 'cov3'\n", "mode = 'flat'\n"
        ]

        base_body = base_cache.get((direction, precision, tile_m, tile_n, gemm_k))
        if base_body is None:
            continue

        out_lines = list(codegen_header)
        out_lines.append('\n')
        out_lines.append(f"{'#' * 89}\n")
        out_lines.append(f"# Master config: all valid tunable combos for {direction}/{precision} "
                         f"{tile_m}x{tile_n}x{gemm_k}\n")
        out_lines.append(f"# Generated by script/generate_all_configs.py\n")
        out_lines.append(f"# {len(combos_list)} combinatorial variants\n")
        out_lines.append(f"{'#' * 89}\n\n")

        seen = set()
        for vals, kname in sorted(combos_list, key=lambda vk: (vk[1], sorted((k, repr(val)) for k, val in vk[0].items()))):
            # Dedup by the REAL resolved kernel name (not raw config text): some
            # flag combinations are no-ops under certain other tunables (e.g.
            # local_prefetch_num's value is irrelevant once tdm_global_load=1 uses
            # its own descriptor-based prefetch instead) -- these produce
            # byte-identical kernels under the identical name from textually
            # DIFFERENT config sections, which the old raw-text section_key dedup
            # didn't catch, causing a real "symbol already defined" assembler
            # collision. Mirrors the same fix already applied in
            # build_gfx1250_master_configs.py's ACCUMULATE_WIDTH_KEYS-adjacent
            # kernel-name dedup (Phase 5b).
            if kname in seen:
                continue
            seen.add(kname)

            # Clone base body and append non-default tunable flags
            new_body = list(base_body)

            extra = _extra_lines(base_body, vals)
            if extra:
                # Insert after last non-comment body line
                insert_at = len(new_body)
                for i in range(len(new_body) - 1, -1, -1):
                    s = new_body[i].strip()
                    if s and not s.startswith('#') and not s.startswith(';'):
                        insert_at = i + 1
                        break
                for i, el in enumerate(extra):
                    new_body.insert(insert_at + i, el)


            # Build label
            active = [k for k, v in sorted(vals.items()) if v not in (0, 'SCOPE_SYS')]
            label = '+'.join(active) if active else 'base'

            section_name = f'igemm_{direction}_gtc'
            out_lines.append(f"# --- {tile_m}x{tile_n}x{gemm_k} +{label} ---\n")
            out_lines.append(f"[{section_name}]\n")
            out_lines.extend(new_body)
            out_lines.append('\n')

        total_files += 1
        total_sections += len(seen)
        print(f"{out_name}: {len(seen)} sections ({len(out_lines)} lines)")

        if args.write:
            with open(out_path, 'w') as f:
                f.writelines(out_lines)

    print(f"\n{total_files} files, {total_sections} total sections")

    # Also write a combined _all.config with the original naming for backward compat
    # with the diverse benchmark script — it expects one file per (dir, prec)
    # named igemm_{dir}_gtc_gfx1250_nhwc_{prec}_all.config
    print("\n--- backward-compat combined files ---")
    combined = defaultdict(list)
    for (direction, precision, tile_m, tile_n, gemm_k), combos_list in sorted(per_tile.items()):
        combined[(direction, precision)].append((tile_m, tile_n, gemm_k, combos_list))

    for (direction, precision), tile_list in sorted(combined.items()):
        out_name = f'igemm_{direction}_gtc_gfx1250_nhwc_{precision}_all.config'
        out_path = os.path.join(CONFIG_DIR, out_name)
        out_lines = list(codegen_header)
        out_lines.append('\n')
        out_lines.append(f"{'#' * 89}\n")
        out_lines.append(f"# Combined master config for {direction}/{precision}\n")
        out_lines.append(f"# Generated by script/generate_all_configs.py\n")
        out_lines.append(f"# Per-tile-shape variants also available as:\n")
        for tm, tn, gk, _ in tile_list:
            out_lines.append(f"#   igemm_{direction}_gtc_gfx1250_nhwc_{precision}_{tm}x{tn}_all.config\n")
        out_lines.append(f"{'#' * 89}\n\n")

        total = 0
        for tile_m, tile_n, gemm_k, combos_list in tile_list:
            base_body = base_cache.get((direction, precision, tile_m, tile_n, gemm_k))
            if base_body is None:
                continue
            seen = set()
            for vals, kname in sorted(combos_list, key=lambda vk: (vk[1], sorted((k, repr(val)) for k, val in vk[0].items()))):
                if kname in seen:
                    continue
                seen.add(kname)
                new_body = list(base_body)
                extra = _extra_lines(base_body, vals)
                if extra:
                    insert_at = len(new_body)
                    for i in range(len(new_body) - 1, -1, -1):
                        s = new_body[i].strip()
                        if s and not s.startswith('#') and not s.startswith(';'):
                            insert_at = i + 1
                            break
                    for i, el in enumerate(extra):
                        new_body.insert(insert_at + i, el)
                active = [k for k, v in sorted(vals.items()) if v not in (0, 'SCOPE_SYS')]
                label = '+'.join(active) if active else 'base'
                section_name = f'igemm_{direction}_gtc'
                out_lines.append(f"# --- {tile_m}x{tile_n}x{gemm_k} +{label} ---\n")
                out_lines.append(f"[{section_name}]\n")
                out_lines.extend(new_body)
                out_lines.append('\n')
                total += 1

        print(f"{out_name}: {total} sections ({len(out_lines)} lines)")
        if args.write:
            with open(out_path, 'w') as f:
                f.writelines(out_lines)

    if not args.write:
        print("\nDry run -- use --write to actually generate files.")


if __name__ == '__main__':
    main()