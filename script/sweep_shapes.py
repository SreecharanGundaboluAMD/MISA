#!/usr/bin/env python3
"""
Standardized shape sweep for gfx1250 (and gfx950) igemm configs.

Builds one or more .config files (via igemm_codegen.py), then runs
conv_driver.exe sequentially -- one shape, one invocation at a time, never
concurrently, since this always targets a single shared GPU -- across a
user-supplied list of conv shapes. Reports every (config, kernel, shape)
combination that fails correctness (-V 1, "valid:n"), and optionally the
per-kernel timing (tflops) for perf comparisons across shapes.

This exists because ad-hoc one-off benchmarking sessions kept re-inventing
this exact loop (build -> run each shape sequentially -> parse "valid:"/
"tflops:" -> summarize) with slightly different, throwaway code each time.
Two real correctness bugs (wrw_incremental_gather's ho*wo < gemm_k_per_block
wrong-answer case, and ds_load_tr_b=0 + lds_double_buffer=1's wrong-answer
case) were found this way -- this script makes that process repeatable and
auditable instead of re-derived from scratch per session.

Usage:
    # Validity sweep across a glob of configs, using a shape list file
    python3 script/sweep_shapes.py --configs 'config/igemm_bwd_gtc_gfx1250_nhwc_fp16_*_all.config' \\
        --shapes shapes/default.json --mode validity

    # Perf sweep (timing only, no -V 1) across one config, one shape
    python3 script/sweep_shapes.py --configs config/igemm_fwd_gtc_gfx1250_nhwc_fp16.config \\
        --shapes shapes/default.json --mode perf

    # Both validity and timing, JSON report to a file
    python3 script/sweep_shapes.py --configs 'config/*_all.config' --shapes shapes/default.json \\
        --mode both --report /tmp/sweep_report.json

Shape file format (JSON): a list of objects, each either:
    {"name": "1x1_bottleneck", "n": 128, "c": 1024, "H": 17, "W": 17, "k": 1024,
     "y": 1, "x": 1, "p": 0, "q": 0, "u": 1, "v": 1, "l": 1, "j": 1, "g": 1}
"y"/"x"/"p"/"q"/"u"/"v"/"l"/"j"/"g" default to 1/1/0/0/1/1/1/1/1 if omitted.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DEFAULTS = {'y': 1, 'x': 1, 'p': 0, 'q': 0, 'u': 1, 'v': 1, 'l': 1, 'j': 1, 'g': 1}
_DIR_TO_F = {'fwd': 1, 'bwd': 2, 'wrw': 4}
_PREC_TO_MODE = {'fp16': 'convfp16', 'bf16': 'convbfp16', 'fp32': 'conv', 'int8': 'convint8'}


def load_shapes(path):
    with open(path) as f:
        shapes = json.load(f)
    out = []
    for i, sh in enumerate(shapes):
        merged = dict(_DEFAULTS)
        merged.update(sh)
        name = merged.pop('name', f'shape{i}')
        for req in ('n', 'c', 'H', 'W', 'k'):
            if req not in merged:
                raise ValueError(f"shape '{name}' missing required field '{req}'")
        out.append((name, merged))
    return out


def detect_direction_precision(config_path):
    """Best-effort (direction, precision) from filename; falls back to parsing
    the file's first igemm_*_gtc section if the filename doesn't say."""
    base = os.path.basename(config_path)
    m = re.search(r'igemm_(fwd|bwd|wrw)_gtc_\w*_(fp16|bf16|fp32|int8)', base)
    if m:
        return m.group(1), m.group(2)
    text = open(config_path).read()
    dm = re.search(r'direction\s*=\s*[\'"](\w+)[\'"]', text)
    pm = re.search(r'precision\s*=\s*[\'"](\w+)[\'"]', text)
    if dm and pm:
        return dm.group(1), pm.group(1)
    raise ValueError(f"could not determine direction/precision for {config_path}")


def build_config(config_path, build_dir):
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, 'igemm_codegen.py'), config_path, '-d', build_dir],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    return proc.returncode == 0, proc.stdout + proc.stderr


_REC_RE = re.compile(r'\]\s*(igemm_[a-zA-Z0-9_]+)')
_COST_RE = re.compile(r'cost:([\d.]+)ms')
_TFLOPS_RE = re.compile(r'tflops:([\d.]+)')
_VALID_RE = re.compile(r'valid:(\w)')

def run_shape(build_dir, mode, F, args, verify, timeout=1800):
    full = ['./conv_driver.exe', mode] + args + [
        '-u', '1', '-v', '1', '-l', '1', '-j', '1', '-g', '1',
        '-F', str(F), '-V', '1' if verify else '0',
        '--in_layout', 'NHWC', '--fil_layout', 'NHWC', '--out_layout', 'NHWC',
    ]
    proc = subprocess.run(full, cwd=build_dir, capture_output=True, text=True, timeout=timeout)
    out = proc.stdout + '\n' + proc.stderr
    recs = re.split(r'(?=\[\w+:\s*\d+\])', out)
    results = []
    for rec in recs:
        if 'igemm_' not in rec:
            continue
        nm = _REC_RE.search(rec)
        cm = _COST_RE.search(rec)
        tm = _TFLOPS_RE.search(rec)
        vm = _VALID_RE.search(rec)
        results.append({
            'name': nm.group(1) if nm else None,
            'cost_ms': float(cm.group(1)) if cm else None,
            'tflops': float(tm.group(1)) if tm else None,
            'valid': vm.group(1) if vm else None,
        })
    return results


def args_for_shape(shape):
    return ['-n', str(shape['n']), '-c', str(shape['c']), '-H', str(shape['H']), '-W', str(shape['W']),
            '-k', str(shape['k']), '-y', str(shape['y']), '-x', str(shape['x']),
            '-p', str(shape['p']), '-q', str(shape['q'])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--configs', required=True, help='glob pattern or single .config path')
    ap.add_argument('--shapes', required=True, help='path to a JSON shape-list file')
    ap.add_argument('--mode', choices=['validity', 'perf', 'both'], default='both',
                     help='validity: -V 1 only; perf: -V 0 timing only; both: -V 1 (timing comes free)')
    ap.add_argument('--repeats', type=int, default=1,
                     help='repeat each (config, shape) this many times (perf noise-floor check)')
    ap.add_argument('--timeout', type=int, default=1800,
                     help='per-shape conv_driver.exe subprocess timeout in seconds (default 1800; '
                          'large combinatorial _all.config files with hundreds of kernels need this high)')
    ap.add_argument('--report', help='write full JSON report to this path (always prints a summary to stdout)')
    ap.add_argument('--keep-build', action='store_true', help='keep the temporary build directories')
    args = ap.parse_args()

    config_paths = sorted(glob.glob(args.configs)) if any(c in args.configs for c in '*?[') else [args.configs]
    if not config_paths:
        print(f"No configs matched {args.configs!r}", file=sys.stderr)
        sys.exit(1)
    shapes = load_shapes(args.shapes)
    print(f"Sweeping {len(config_paths)} config(s) x {len(shapes)} shape(s), mode={args.mode}, "
          f"repeats={args.repeats}. Running SEQUENTIALLY (single shared GPU).")

    report = {'configs': {}}
    total_checked = 0
    total_bad = 0
    bad_list = []

    for cfg in config_paths:
        cfg_name = os.path.basename(cfg)
        try:
            direction, precision = detect_direction_precision(cfg)
        except ValueError as e:
            print(f"SKIP {cfg_name}: {e}", file=sys.stderr)
            continue
        mode = _PREC_TO_MODE.get(precision)
        F = _DIR_TO_F.get(direction)
        if mode is None or F is None:
            print(f"SKIP {cfg_name}: unknown direction/precision {direction}/{precision}", file=sys.stderr)
            continue

        build_dir = tempfile.mkdtemp(prefix='sweep_')
        ok, log = build_config(cfg, build_dir)
        if not ok:
            print(f"BUILD FAIL {cfg_name}:\n{log[-2000:]}", file=sys.stderr)
            if not args.keep_build:
                shutil.rmtree(build_dir, ignore_errors=True)
            continue

        cfg_report = {'direction': direction, 'precision': precision, 'shapes': {}}
        print(f"\n=== {cfg_name} ({direction}/{precision}) ===")
        for shape_name, shape in shapes:
            do_verify = args.mode in ('validity', 'both')
            all_runs = []
            for _ in range(max(1, args.repeats)):
                results = run_shape(build_dir, mode, F, args_for_shape(shape), do_verify, timeout=args.timeout)
                all_runs.append(results)
            base_results = all_runs[0]
            total_checked += len(base_results)
            bad_here = [r for r in base_results if r['valid'] == 'n']
            total_bad += len(bad_here)
            for r in bad_here:
                bad_list.append({'config': cfg_name, 'shape': shape_name, 'kernel': r['name']})
            if args.repeats > 1 and args.mode in ('perf', 'both'):
                # attach per-repeat tflops list per kernel, keyed by kernel name
                by_kernel = {}
                for run in all_runs:
                    for r in run:
                        if r['name']:
                            by_kernel.setdefault(r['name'], []).append(r['tflops'])
                cfg_report['shapes'][shape_name] = {'results': base_results, 'tflops_by_kernel': by_kernel}
            else:
                cfg_report['shapes'][shape_name] = {'results': base_results}
            status = 'OK' if not bad_here else f'{len(bad_here)} BAD'
            print(f"  {shape_name}: {len(base_results)} kernels, {status}")
        report['configs'][cfg_name] = cfg_report
        if not args.keep_build:
            shutil.rmtree(build_dir, ignore_errors=True)
        else:
            print(f"  (build kept at {build_dir})")

    print(f"\n{'='*70}")
    print(f"TOTAL: {total_checked} (config, kernel, shape) checks, {total_bad} bad")
    if bad_list:
        print("\nProblematic combinations:")
        for b in bad_list:
            print(f"  {b['config']} | {b['shape']} | {b['kernel']}")

    if args.report:
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=1)
        print(f"\nFull report written to {args.report}")

    sys.exit(1 if total_bad else 0)


if __name__ == '__main__':
    main()
