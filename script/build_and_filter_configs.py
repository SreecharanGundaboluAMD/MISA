#!/usr/bin/env python3
"""
Build each generated per-tile _all.config, identify sections that fail assembly
(VGPR overflow, etc.), and rewrite the config file with only the passing sections.

This replaces the earlier hand-coded VGPR-budget heuristics in the generator --
the assembler is the authoritative source for which combinations actually build.

Usage:
    python3 script/build_and_filter_configs.py [--write]
"""
import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, 'config')


def find_per_tile_configs():
    """Return list of per-tile _all.config file paths."""
    configs = []
    for fn in sorted(os.listdir(CONFIG_DIR)):
        if not fn.endswith('_all.config'):
            continue
        # Must match per-tile pattern: _NNNxNNN_all or just _all
        if re.search(r'_\d+x\d+_all\.config$', fn):
            configs.append(os.path.join(CONFIG_DIR, fn))
    return configs


def parse_sections(path):
    """Return codegen_header_lines and list of (section_idx, section_text)."""
    with open(path) as f:
        lines = f.readlines()

    # Find codegen header (everything before first [igemm_])
    header_end = 0
    for i, line in enumerate(lines):
        if re.match(r'^\[igemm_\w+\]\s*$', line):
            header_end = i
            break

    header = lines[:header_end]

    # Collect sections
    sections = []
    in_section = False
    cur_section_text = []
    for i in range(header_end, len(lines)):
        line = lines[i]
        if re.match(r'^\[igemm_\w+\]\s*$', line):
            if cur_section_text:
                sections.append(cur_section_text)
            cur_section_text = [line]
            in_section = True
        elif in_section:
            cur_section_text.append(line)
    if cur_section_text:
        sections.append(cur_section_text)

    return header, sections


def build_and_capture_errors(config_path):
    """Build the config file, return the build log output."""
    import tempfile
    build_dir = tempfile.mkdtemp(prefix='cfg_test_')
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, 'igemm_codegen.py'),
             '-d', build_dir, config_path],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, '', 'TIMEOUT'
    finally:
        import shutil
        shutil.rmtree(build_dir, ignore_errors=True)


def extract_failing_sections(build_stdout, build_stderr):
    """Parse build output to identify which section names failed.

    Returns set of kernel base names that caused errors.
    We use the pattern: on a successful build, every section in the config
    produces a unique kernel name. Errors cite specific kernel names (e.g.,
    'igemm_fwd_gtcw_nhwc_bf16_bx0_...'). We extract those names.
    """
    failing = set()
    for line in (build_stdout + '\n' + build_stderr).split('\n'):
        # Match: error: ... igemm_<dir>_gtcw_nhwc_<prec>_bx0_...
        m = re.search(r'igemm_\w+_gtcw_nhwc_\w+_bx\d+_ex\d+_bt(\d+x\d+x\d+)_wt(\d+x\d+)_wr(\d+x\d+)', line)
        if m:
            tile_key = f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
            failing.add(tile_key)
    return failing


def section_to_signature(section_lines):
    """Extract the unique identifying features of a section."""
    sig_parts = []
    for line in section_lines:
        line = line.strip()
        if line.startswith('[') or not line:
            continue
        if line.startswith('#') or line.startswith(';'):
            continue
        sig_parts.append(line)
    return '\n'.join(sorted(sig_parts))


def build_kernel_name_from_section(section_lines):
    """Reconstruct what the kernel name would be from a config section.

    The kernel name encodes: gemm_m, gemm_n, gemm_k, wmma_tile, wmma_repeat,
    and the suffix flags (_dbuf, _gkgs, _mtail, _ntail, _direct, _setprio, etc.).
    We extract enough to match against the error messages.
    """
    vals = {}
    for line in section_lines:
        line = line.strip()
        if '=' in line and not line.startswith('#') and not line.startswith(';'):
            k, v = line.split('=', 1)
            k, v = k.strip(), v.strip()
            try:
                vals[k] = int(v)
            except ValueError:
                vals[k] = v.strip('"').strip("'")

    gm = vals.get('gemm_m_per_block', 0)
    gn = vals.get('gemm_n_per_block', 0)
    gk = vals.get('gemm_k_per_block', 0)
    wt_m = vals.get('wmma_tile_m', 16)
    wr_m = vals.get('wmma_repeat_m', 4)
    wr_n = vals.get('wmma_repeat_n', 4)

    return f"bt{gm}x{gn}x{gk}_wt{wt_m}x{wt_m}_wr{wr_m}x{wr_n}"


def process_config(config_path, write=False):
    """Build a config, identify failing sections, optionally rewrite."""
    header, sections = parse_sections(config_path)
    if not sections:
        return len(sections), len(sections)

    print(f"  building {os.path.basename(config_path)} ({len(sections)} sections)...", end=' ', flush=True)
    rc, stdout, stderr = build_and_capture_errors(config_path)

    if rc == 0:
        print(f"OK ({len(sections)} sections pass)")
        return len(sections), len(sections)

    failing = extract_failing_sections(stdout, stderr)

    if not failing:
        # Failed but couldn't parse which sections -- keep all
        print(f"FAILED (unparseable, keeping all {len(sections)})")
        return len(sections), len(sections)

    # Identify failing sections by matching kernel name patterns in their lines
    passing_sections = []
    removed = 0
    for sec in sections:
        sig = build_kernel_name_from_section(sec)
        fails = any(f in sig for f in failing)
        if fails:
            removed += 1
        else:
            passing_sections.append(sec)

    if removed == 0:
        print(f"FAILED (couldn't isolate, keeping all {len(sections)})")
        return len(sections), len(sections)

    print(f"FAILED ({removed} removed, {len(passing_sections)} kept)")
    # Rebuild with only passing sections
    if write and passing_sections:
        with open(config_path, 'w') as f:
            f.writelines(header)
            f.write('\n')
            for sec in passing_sections:
                f.writelines(sec)
                if not sec[-1].endswith('\n'):
                    f.write('\n')
                f.write('\n')
        # Re-verify
        rc2, _, _ = build_and_capture_errors(config_path)
        if rc2 != 0:
            print(f"  WARNING: filtered config still fails to build!")
            return len(sections), 0

    return len(sections), len(passing_sections)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true',
                    help='Rewrite failing config files with only passing sections')
    ap.add_argument('--jobs', type=int, default=4,
                    help='Number of parallel build jobs (default: 4)')
    args = ap.parse_args()

    configs = find_per_tile_configs()
    print(f"Found {len(configs)} per-tile configs to verify\n")

    # Build sequentially (each igemm_codegen.py already uses Python multiprocessing
    # internally for kernel emission -- parallelizing at this level would create
    # nested process pools which deadlock or thrash)
    total_orig = 0
    total_pass = 0
    for config_path in configs:
        orig, passing = process_config(config_path, write=args.write)
        total_orig += orig
        total_pass += passing

    print(f"\nTotal: {total_pass}/{total_orig} sections pass assembly")
    if not args.write:
        print("Dry run -- use --write to actually filter configs.")


if __name__ == '__main__':
    main()