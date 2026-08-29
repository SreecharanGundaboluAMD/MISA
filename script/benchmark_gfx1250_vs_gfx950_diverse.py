#!/usr/bin/env python3
"""
Benchmark MISA's gfx1250 WMMA kernels against gfx950 MIOpen reference times, on a
diverse 60-shape set (20 each fwd/bwd/wrw) sampled from this repo's
convasmimplicitgemmgtcdynamic_{fwd,bwd,wrw}.txt dumps -- real MIOpenDriver command
lines paired with their measured ConvAsmImplicitGemmGTCDynamic*XdlopsNHWC time on
gfx950. Shapes were picked by sorting all (group==1) entries in each file by a FLOPs
proxy (n*c*k*ho*wo*y*x) and taking one representative per bucket across 20 equal-
population buckets, rotating through bf16/fp16/fp32 as the preferred precision per
bucket -- giving coverage across problem size (tiny 1x1 pointwise up to 1536x1536
spatial), filter size (1x1 up to 7x7), stride/dilation, and precision, without pure
randomness (reproducible; see script/build_gfx1250_master_configs.py's sibling
tooling for how the underlying configs are kept comprehensive).

Unlike script/benchmark_gfx1250_vs_miopen.py (which hardcodes ONE MISA config per
shape from a small curated set), this script builds ONE master config per
(direction, precision) via the *_all.config files (config/build_gfx1250_master_configs.py's
output, which already unions every validated tunable including gsplit/32x32 as of
Phase 48/49) and lets conv_driver.exe search the FULL candidate set per shape,
reporting the true fastest MISA kernel -- so split-K, 32x32, TDM, tail-relief, etc.
are all automatically in play without needing to guess which one a given shape wants.

Usage:
    python3 script/benchmark_gfx1250_vs_gfx950_diverse.py [options]

Options:
    --direction {fwd,bwd,wrw,all}   only benchmark one direction (default: all)
    --rebuild                       force-rebuild every master config even if already built
    --warmup N                      IGEMM_WARMUP (default 5)
    --repeat N                      IGEMM_REPEAT (default 20)
    --verify                        run with -V 1 (correctness check) instead of -V 0 (timing only)
    --build-dir DIR                 where to build MISA configs (default: bench_out/, gitignored)
    --markdown-out FILE             also write the results table to FILE

Before trusting absolute numbers, check GPU contention (this script prints a
rocm-smi snapshot before running, but does not block on it).
"""
import argparse
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FORW_FLAG = {'fwd': 1, 'bwd': 2, 'wrw': 4}
MODE_STRING = {'fp32': 'conv', 'fp16': 'convfp16', 'bf16': 'convbfp16'}

MASTER_CONFIG_FILES = {
    (d, p): f'config/igemm_{d}_gtc_gfx1250_nhwc_{p}_all.config'
    for d in ('fwd', 'bwd', 'wrw') for p in ('fp16', 'bf16', 'fp32')
}

# Per-tile _all.config files with combinatorial tunable coverage.
# Each covers one tile shape with up to ~68 valid tunable combos.
# Used as alternate candidates per shape alongside the monolithic config.
def find_per_tile_configs(direction, precision):
    """Return list of (tile_m, tile_n, config_path) for this direction/precision."""
    import glob
    results = []
    pattern = f'config/igemm_{direction}_gtc_gfx1250_nhwc_{precision}_*x*_all.config'
    for path in sorted(glob.glob(pattern)):
        m = re.search(r'_(\d+)x(\d+)_all\.config$', path)
        if m:
            results.append((int(m.group(1)), int(m.group(2)), path))
    return results

# (direction, n, c, H, W, k, y, x, p, q, u, v, l, j, precision, gfx950_ms)
# H/W are the INPUT spatial dims (MIOpenDriver convention, uniform across fwd/bwd/wrw);
# p/q = pad_h/pad_w, u/v = stride_h/stride_w, l/j = dilation_h/dilation_w, group=1 always
# (none of these three dump files had group>1 entries survive the group==1 filter).
SHAPES = [
    ('bwd', 4, 896, 1, 1, 112, 1, 1, 0, 0, 1, 1, 1, 1, 'bf16', 0.0117),
    ('bwd', 4, 96, 18, 18, 96, 3, 3, 1, 1, 2, 2, 1, 1, 'fp16', 0.0239),
    ('bwd', 1, 512, 20, 31, 1024, 1, 1, 0, 0, 2, 2, 1, 1, 'fp32', 0.0295),
    ('bwd', 12, 256, 14, 14, 128, 3, 3, 1, 1, 2, 2, 1, 1, 'bf16', 0.0312),
    ('bwd', 1, 128, 38, 173, 64, 5, 5, 1, 1, 2, 2, 1, 1, 'fp16', 0.0341),
    ('bwd', 2, 1024, 46, 46, 512, 1, 1, 0, 0, 2, 2, 1, 1, 'fp32', 0.0888),
    ('bwd', 800, 1024, 1, 1, 1024, 1, 1, 0, 0, 1, 1, 1, 1, 'bf16', 0.0163),
    ('bwd', 4, 256, 19, 87, 128, 5, 5, 1, 1, 2, 2, 1, 1, 'fp16', 0.0564),
    ('bwd', 128, 608, 14, 14, 128, 1, 1, 0, 0, 1, 1, 1, 1, 'fp32', 0.0961),
    ('bwd', 64, 256, 38, 38, 128, 1, 1, 0, 0, 1, 1, 1, 1, 'bf16', 0.0238687),
    ('bwd', 64, 256, 32, 32, 256, 1, 1, 0, 0, 1, 1, 1, 1, 'fp16', 0.0293),
    ('bwd', 204, 2048, 7, 7, 1024, 1, 1, 0, 0, 2, 2, 1, 1, 'fp32', 0.4797),
    ('bwd', 96, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1, 'bf16', 0.0958),
    ('bwd', 64, 96, 71, 71, 64, 3, 3, 0, 0, 1, 1, 1, 1, 'fp16', 0.175),
    ('bwd', 128, 512, 25, 25, 256, 3, 3, 0, 0, 2, 2, 1, 1, 'fp32', 1.8958),
    ('bwd', 22, 2048, 1, 256, 2048, 1, 2, 0, 0, 1, 2, 1, 1, 'bf16', 0.2042),
    ('bwd', 36, 768, 1, 682, 768, 1, 3, 0, 1, 1, 1, 1, 1, 'fp16', 0.2298),
    ('bwd', 100, 512, 28, 28, 512, 3, 3, 1, 1, 2, 2, 1, 1, 'fp32', 3.442),
    ('bwd', 256, 256, 18, 86, 128, 5, 5, 1, 1, 2, 2, 1, 1, 'bf16', 1.6698),
    ('bwd', 1, 512, 512, 512, 256, 3, 3, 1, 1, 1, 1, 1, 1, 'fp16', 1.1379),

    ('fwd', 64, 2, 5, 5, 1, 3, 3, 0, 0, 1, 1, 1, 1, 'bf16', 0.0152),
    ('fwd', 4, 1024, 1, 1, 64, 1, 1, 0, 0, 1, 1, 1, 1, 'fp16', 0.0078),
    ('fwd', 2, 256, 88, 100, 3, 1, 1, 0, 0, 1, 1, 1, 1, 'fp32', 0.0236),
    ('fwd', 4, 256, 14, 14, 256, 1, 1, 0, 0, 1, 1, 1, 1, 'bf16', 0.0156),
    ('fwd', 1, 256, 56, 56, 60, 3, 3, 1, 1, 2, 2, 1, 1, 'fp16', 0.023),
    ('fwd', 1, 64, 224, 224, 256, 1, 1, 0, 0, 2, 2, 1, 1, 'fp32', 0.0306),
    ('fwd', 37, 64, 38, 38, 64, 1, 3, 0, 1, 1, 2, 1, 1, 'bf16', 0.0256),
    ('fwd', 32, 608, 14, 14, 128, 1, 1, 0, 0, 1, 1, 1, 1, 'fp16', 0.0291),
    ('fwd', 2, 256, 13, 19, 720, 3, 3, 1, 1, 1, 1, 1, 1, 'fp32', 0.0443),
    ('fwd', 29, 1, 958, 80, 256, 3, 3, 0, 0, 2, 2, 1, 1, 'bf16', 0.3266),
    ('fwd', 4, 32, 112, 112, 512, 3, 3, 1, 1, 2, 2, 1, 1, 'fp16', 0.0362),
    ('fwd', 12, 256, 55, 55, 128, 3, 3, 1, 1, 2, 2, 1, 1, 'fp32', 0.0956),
    ('fwd', 7, 64, 225, 225, 64, 1, 3, 0, 1, 1, 1, 1, 1, 'bf16', 0.1219),
    ('fwd', 4, 512, 28, 28, 512, 3, 3, 1, 1, 1, 1, 1, 1, 'fp16', 0.0588),
    ('fwd', 256, 160, 17, 17, 768, 1, 1, 0, 0, 1, 1, 1, 1, 'fp32', 0.3343),
    ('fwd', 543, 80, 1, 92, 768, 1, 3, 0, 1, 1, 1, 1, 1, 'bf16', 0.1215),
    ('fwd', 128, 3, 225, 225, 64, 7, 7, 3, 3, 2, 2, 1, 1, 'fp16', 0.4298),
    ('fwd', 16, 128, 223, 223, 128, 3, 3, 1, 1, 2, 2, 1, 1, 'fp32', 0.973),
    ('fwd', 1, 96, 768, 768, 128, 3, 3, 1, 1, 1, 1, 1, 1, 'bf16', 0.3819),
    ('fwd', 1, 96, 1536, 1536, 128, 3, 3, 1, 1, 1, 1, 1, 1, 'fp16', 1.5088),

    ('wrw', 2, 576, 1, 30, 66, 1, 1, 0, 0, 1, 1, 1, 1, 'bf16', 0.0156),
    ('wrw', 64, 120, 6, 6, 120, 3, 3, 0, 0, 2, 2, 1, 1, 'fp16', 0.0257),
    ('wrw', 4, 128, 14, 14, 800, 1, 1, 0, 0, 1, 1, 1, 1, 'fp32', 0.0166),
    ('wrw', 5, 1728, 1, 30, 576, 1, 1, 0, 0, 1, 1, 1, 1, 'bf16', 0.0164),
    ('wrw', 128, 64, 56, 56, 3, 7, 7, 2, 2, 4, 4, 1, 1, 'fp16', 0.1051),
    ('wrw', 2, 64, 192, 240, 64, 1, 1, 0, 0, 1, 1, 1, 1, 'fp32', 0.0371),
    ('wrw', 16, 352, 28, 28, 128, 1, 1, 0, 0, 1, 1, 1, 1, 'bf16', 0.0338),
    ('wrw', 32, 256, 28, 28, 128, 1, 1, 0, 0, 1, 1, 1, 1, 'fp16', 0.0316),
    ('wrw', 42, 32, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1, 'fp32', 0.0563),
    ('wrw', 43, 128, 56, 56, 96, 1, 1, 0, 0, 1, 1, 1, 1, 'bf16', 0.0563),
    ('wrw', 64, 128, 28, 28, 384, 1, 1, 0, 0, 1, 1, 1, 1, 'fp16', 0.0625),
    ('wrw', 32, 32, 54, 54, 128, 3, 3, 1, 1, 1, 1, 1, 1, 'fp32', 0.1213),
    ('wrw', 48, 512, 29, 29, 256, 1, 1, 0, 0, 1, 1, 1, 1, 'bf16', 0.0965),
    ('wrw', 51, 768, 1, 963, 80, 1, 3, 0, 1, 1, 1, 1, 1, 'fp16', 0.1289),
    ('wrw', 256, 32, 449, 449, 3, 3, 3, 1, 1, 2, 2, 1, 1, 'fp32', 6.9182),
    ('wrw', 153, 768, 1, 163, 768, 1, 1, 0, 0, 1, 1, 1, 1, 'bf16', 0.139),
    ('wrw', 52, 768, 1, 477, 768, 1, 3, 0, 1, 1, 2, 1, 1, 'fp16', 0.2169),
    ('wrw', 64, 128, 113, 113, 128, 3, 3, 0, 0, 2, 2, 1, 1, 'fp32', 2.9065),
    ('wrw', 40, 768, 1, 622, 768, 1, 3, 0, 1, 1, 1, 1, 1, 'bf16', 0.2294),
    ('wrw', 512, 510, 10, 10, 512, 3, 3, 1, 1, 1, 1, 1, 1, 'fp16', 0.4809),
]

COST_RE = re.compile(r'\[(fwd|bwd|wrw):\s*\d+\][^\n]*?cost:([0-9.]+)ms')


def check_gpu_contention():
    try:
        out = subprocess.run(['rocm-smi', '--showuse', '--showpids'], capture_output=True,
                              text=True, timeout=10).stdout
        print("--- rocm-smi --showuse --showpids (contention snapshot) ---")
        print(out.strip())
        print("--- absolute numbers below are not trustworthy if this shows load from ---")
        print("--- another tenant (no local conv_driver.exe process running yet) ---\n")
    except Exception as e:
        print(f"(rocm-smi check skipped: {e})\n")


def build_one_config(config_file, out_dir, rebuild):
    """Build a single config file. Returns out_dir on success, None on failure."""
    exe = os.path.join(out_dir, 'conv_driver.exe')
    if os.path.exists(exe) and not rebuild:
        return out_dir
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    # Only print first-time builds
    if not os.path.exists(exe):
        label = os.path.basename(config_file)
        print(f"building {label} -> {out_dir}")
    try:
        subprocess.run([sys.executable, 'igemm_codegen.py', '-d', out_dir, config_file],
                        cwd=REPO_ROOT, check=True, capture_output=True, timeout=600)
        return out_dir
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def build_all_configs_parallel(direction, precision, build_dir, rebuild):
    """Build all configs for a (direction, precision) pair in parallel.
    Returns list of (out_dir, config_label) for successfully built configs.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    configs_to_build = []

    # Per-tile combinatorial configs only (the monolithic _all.config is too big
    # for the assembler's branch-range limit with 168 sections, so we search each
    # tile shape independently and aggregate the best result per shape)
    for tile_m, tile_n, tile_cfg in find_per_tile_configs(direction, precision):
        tile_out = os.path.join(build_dir, f'{direction}_{precision}_{tile_m}x{tile_n}_combo')
        configs_to_build.append((tile_cfg, tile_out, f'{tile_m}x{tile_n}_combo'))

    results = []
    futures = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for cfg, out_dir, label in configs_to_build:
            futures[pool.submit(build_one_config, cfg, out_dir, rebuild)] = (out_dir, label)

    for future in as_completed(futures):
        out_dir, label = futures[future]
        result = future.result()
        if result:
            results.append((result, label))
        else:
            print(f"  SKIP: {label} failed to build (VGPR overflow or other error)")

    return results


def run_shape(out_dir, direction, precision, n, c, H, W, k, y, x, p, q, u, v, l, j,
              warmup, repeat, verify):
    args = [
        './conv_driver.exe', MODE_STRING[precision],
        '-n', str(n), '-c', str(c), '-H', str(H), '-W', str(W), '-k', str(k),
        '-y', str(y), '-x', str(x), '-p', str(p), '-q', str(q),
        '-u', str(u), '-v', str(v), '-l', str(l), '-j', str(j),
        '-m', 'conv', '-g', '1', '-F', str(FORW_FLAG[direction]), '-t', '1',
        '--in_layout', 'NHWC', '--fil_layout', 'NHWC', '--out_layout', 'NHWC',
        '-V', '1' if verify else '0',
    ]
    env = dict(os.environ, IGEMM_WARMUP=str(warmup), IGEMM_REPEAT=str(repeat))
    try:
        result = subprocess.run(args, cwd=out_dir, env=env, capture_output=True, text=True,
                                 timeout=480)
    except subprocess.TimeoutExpired as e:
        def _decode(x):
            return x.decode('utf-8', 'replace') if isinstance(x, bytes) else (x or '')
        partial = _decode(e.stdout) + _decode(e.stderr)
        return None, f"TIMED OUT after {e.timeout}s\n{partial}", ''
    costs = [float(m.group(2)) for m in COST_RE.finditer(result.stdout) if m.group(1) == direction]
    if not costs:
        return None, result.stdout + result.stderr, ''
    return min(costs), None, result.stdout


def fmt_ratio(misa_ms, ref_ms):
    if misa_ms is None or ref_ms is None or ref_ms <= 0:
        return 'n/a'
    ratio = misa_ms / ref_ms
    return f"{ratio:.2f}x slower" if ratio >= 1 else f"{1/ratio:.2f}x faster"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--direction', choices=['fwd', 'bwd', 'wrw', 'all'], default='all')
    ap.add_argument('--rebuild', action='store_true', help='force-rebuild every master config even if already built')
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--repeat', type=int, default=20)
    ap.add_argument('--verify', action='store_true', help='run with -V 1 instead of -V 0')
    ap.add_argument('--build-dir', default=os.path.join(REPO_ROOT, 'bench_out'))
    ap.add_argument('--markdown-out', default=None)
    ap.add_argument('--log-dir', default=None, help='Directory to write full per-shape conv_driver.exe output logs')
    args = ap.parse_args()

    check_gpu_contention()

    shapes = [s for s in SHAPES if args.direction == 'all' or s[0] == args.direction]
    needed = sorted(set((s[0], s[14]) for s in shapes))

    # Build all configs in parallel first (ThreadPoolExecutor, 8 workers)
    print("=== Building all configs (parallel, up to 8 workers) ===\n")
    all_configs = {}
    for direction, precision in needed:
        print(f"\n--- {direction}/{precision} ---")
        configs = build_all_configs_parallel(direction, precision, args.build_dir, args.rebuild)
        all_configs[(direction, precision)] = configs
        print(f"  {len(configs)} configs built successfully")
    print(f"\n=== Build complete. Starting benchmark ===\n")

    header = ("| Direction | Precision | Shape (n,c,H,W,k,y×x,p,q,u,v) | MISA (ms) "
               "| gfx950 (ms) | vs gfx950 |")
    sep = "|---|---|---|---|---|---|"
    rows = []
    per_dir_ratios = {}

    for (direction, n, c, H, W, k, y, x, p, q, u, v, l, j, precision, ref950) in shapes:
        configs = all_configs.get((direction, precision), [])
        best_ms, best_label = None, None
        all_candidates = []
        for out_dir, label in configs:
            misa_ms, err, stdout = run_shape(out_dir, direction, precision, n, c, H, W, k, y, x,
                                      p, q, u, v, l, j, args.warmup, args.repeat, args.verify)
            all_candidates.append((label, misa_ms))
            if args.log_dir and stdout:
                os.makedirs(args.log_dir, exist_ok=True)
                logfile = os.path.join(args.log_dir,
                    f"{direction}_{precision}_{n}x{c}x{H}x{W}_{k}_{y}x{x}.log")
                with open(logfile, 'w') as f:
                    f.write(stdout)
            if misa_ms is not None and (best_ms is None or misa_ms < best_ms):
                best_ms, best_label = misa_ms, label
        misa_ms = best_ms
        shape_str = f"{n},{c},{H},{W},{k},{y}x{x},{p},{q},{u},{v}"
        # Log all candidates
        cand_str = " | ".join(f"{cn}={cm:.4f}ms" if cm else f"{cn}=N/A"
                              for cn, cm in sorted(all_candidates, key=lambda x: (x[1] if x[1] else 999.0)))
        print(f"[{direction} {precision} {shape_str}] candidates: {cand_str}")
        if misa_ms is None:
            label = 'not applicable'
            rows.append(f"| {direction} | {precision} | {shape_str} | {label} | {ref950} | - |")
            print(f"[{direction} {precision} {shape_str}] not applicable")
        else:
            r950 = fmt_ratio(misa_ms, ref950)
            rows.append(f"| {direction} | {precision} | {shape_str} | {misa_ms:.5f} | {ref950:.5f} | {r950} |")
            per_dir_ratios.setdefault(direction, []).append(misa_ms / ref950)
            print(f"[{direction} {precision} {shape_str}] BEST={misa_ms:.5f}ms  gfx950={ref950:.5f}ms ({r950})")
        # Write incremental markdown after each shape
        if args.markdown_out:
            lines_sofar = [header, sep] + rows
            lines_sofar.append("")
            lines_sofar.append("(benchmark in progress...)")
            with open(args.markdown_out, 'w') as f:
                f.write("\n".join(lines_sofar) + "\n")

    lines = [header, sep] + rows
    lines.append("")
    lines.append("Summary (avg ratio, MISA/gfx950 -- >1 means MISA is slower; "
                  "'not applicable' shapes excluded from the average):")
    for d in ['fwd', 'bwd', 'wrw']:
        if d in per_dir_ratios:
            n_ok = len(per_dir_ratios[d])
            n_total = len([s for s in shapes if s[0] == d])
            avg = sum(per_dir_ratios[d]) / n_ok
            lines.append(f"- {d}: avg {avg:.2f}x ({n_ok}/{n_total} shapes ran)")

    table_md = "\n".join(lines)
    print("\n" + table_md)

    if args.markdown_out:
        with open(args.markdown_out, 'w') as f:
            f.write(table_md + "\n")
        print(f"\nwrote {args.markdown_out}")


if __name__ == '__main__':
    main()
