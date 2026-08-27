#!/usr/bin/env python3
"""
Benchmark MISA's gfx1250 WMMA kernels against MIOpen (gfx950 and gfx1250), on the
same 38-shape set characterized in docs/gfx1250_vendor_benchmark_vs_miopen.md
(17 fwd, 11 bwd, 10 wrw). Automates what that doc's many "Update" sections did by
hand: build the right MISA config for each shape, run conv_driver.exe, and report
MISA's time next to the MIOpen reference numbers already recorded from
~/rocm-libraries/tracelens_shapes_gfx950.json and
~/rocm-libraries/tracelense_gfx1250_b01_2.json (embedded below, not re-read from
those files, so this script has no dependency on them being present).

Usage:
    python3 script/benchmark_gfx1250_vs_miopen.py [options]

Options:
    --direction {fwd,bwd,wrw,all}   only benchmark one direction (default: all)
    --rebuild                       force-rebuild every MISA config even if already built
    --warmup N                      IGEMM_WARMUP (default 5, matches the doc's methodology)
    --repeat N                      IGEMM_REPEAT (default 20, matches the doc's methodology)
    --verify                        run with -V 1 (correctness check) instead of -V 0 (timing only)
    --build-dir DIR                 where to build MISA configs (default: bench_out/, gitignored)
    --markdown-out FILE             also write the results table to FILE

Before trusting absolute numbers, check GPU contention (this script prints a
rocm-smi snapshot before running, but does not block on it -- see
docs/gfx1250_vendor_benchmark_vs_miopen.md's repeated contention caveats).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FORW_FLAG = {'fwd': 1, 'bwd': 2, 'wrw': 4}

# Maps (direction, config) -> config file (relative to repo root). "config" names
# match the Config column in docs/gfx1250_vendor_benchmark_vs_miopen.md's 38-shape
# tables: base = combined 128x128+64x64 exact-fit, mtail/ntail/mntail = M/N/both
# tail-relief (128x128-only for fwd, 64x64-only for bwd), tdm = TDM-based GEMM_K
# tail (128x128-only, 1x1 conv only), gsplit = K-split (128x128+64x64 both tiles,
# all three directions as of Phase 49 -- wrw always had it, bwd/fwd got it in
# Phase 48/49). Every shape's hardcoded per-row config (below) is tried alongside
# 'gsplit' automatically for bwd/fwd (see GSPLIT_CANDIDATE in main()) -- split-K
# wasn't available when most of these rows were first characterized, so it's
# always worth trying as a second candidate, not just when a row explicitly says so.
CONFIG_FILES = {
    ('fwd', 'base'):   'config/igemm_fwd_gtc_gfx1250_nhwc_bf16.config',
    ('fwd', 'mtail'):  'config/igemm_fwd_gtc_gfx1250_nhwc_bf16_mtail.config',
    ('fwd', 'ntail'):  'config/igemm_fwd_gtc_gfx1250_nhwc_bf16_ntail.config',
    ('fwd', 'mntail'): 'config/igemm_fwd_gtc_gfx1250_nhwc_bf16_mntail.config',
    ('fwd', 'tdm'):    'config/igemm_fwd_gtc_gfx1250_nhwc_bf16_tdm.config',
    ('fwd', 'gsplit'): 'config/igemm_fwd_gtc_gfx1250_nhwc_bf16_gsplit.config',
    ('bwd', 'base'):   'config/igemm_bwd_gtc_gfx1250_nhwc_bf16.config',
    ('bwd', 'mtail'):  'config/igemm_bwd_gtc_gfx1250_nhwc_bf16_mtail.config',
    ('bwd', 'gsplit'): 'config/igemm_bwd_gtc_gfx1250_nhwc_bf16_gsplit.config',
    ('wrw', 'gsplit'): 'config/igemm_wrw_gtc_gfx1250_nhwc_bf16_gsplit.config',
}

# gsplit wasn't available (or, for wrw, wasn't necessarily the best choice) when most
# SHAPES rows below were characterized -- always try it as a second candidate for
# bwd/fwd (wrw rows already use gsplit as their primary config) and report whichever
# is faster, rather than only running the row's originally-hardcoded config.
GSPLIT_CANDIDATE = {'fwd': 'gsplit', 'bwd': 'gsplit'}

# The 38 shapes from docs/gfx1250_vendor_benchmark_vs_miopen.md's "full re-triage"
# and "vs. MIOpen running natively on gfx1250" tables (2026-08-27 updates), batch=42
# bf16/NHWC throughout, group=1. MIOpen numbers are read directly from
# ~/rocm-libraries/tracelens_shapes_gfx950.json and
# ~/rocm-libraries/tracelense_gfx1250_b01_2.json (row[3], milliseconds) and
# ~row[2]~ (solver name for gfx1250) -- embedded here so this script doesn't
# depend on those files existing on whatever machine runs it.
SHAPES = [
    # direction, c, H, W, k, y, x, p, config, miopen_gfx950_ms, miopen_gfx1250_ms, solver_gfx1250
    ('fwd', 256, 1,   1,   16,  1, 1, 0, 'mntail', 0.006209, 0.008622, '137/ConvHipImplicitGemmGroupFwdXdlops'),
    ('fwd', 512, 1,   1,   32,  1, 1, 0, 'mntail', 0.007995, 0.009712, '137/ConvHipImplicitGemmGroupFwdXdlops'),
    ('fwd', 32,  1,   1,   512, 1, 1, 0, 'mtail',  0.006293, 0.008475, '137/ConvHipImplicitGemmGroupFwdXdlops'),
    ('fwd', 96,  120, 160, 48,  1, 1, 0, 'ntail',  0.039973, 0.052051, '137/ConvHipImplicitGemmGroupFwdXdlops'),
    ('fwd', 128, 30,  40,  512, 1, 1, 0, 'mtail',  0.021889, 0.020466, '220/ConvHipConv'),
    ('fwd', 256, 30,  40,  128, 1, 1, 0, 'mtail',  0.013662, 0.014924, '220/ConvHipConv'),
    ('fwd', 192, 60,  80,  64,  1, 1, 0, 'base',   0.022817, 0.036151, '137/ConvHipImplicitGemmGroupFwdXdlops'),
    ('fwd', 64,  60,  80,  128, 1, 1, 0, 'base',   0.018240, 0.020390, '137/ConvHipImplicitGemmGroupFwdXdlops'),
    ('fwd', 512, 30,  40,  128, 1, 1, 0, 'mtail',  0.019533, 0.019954, '137/ConvHipImplicitGemmGroupFwdXdlops'),
    ('fwd', 24,  240, 320, 128, 1, 1, 0, 'tdm',    0.179727, 0.168864, '137/ConvHipImplicitGemmGroupFwdXdlops'),
    ('fwd', 256, 60,  80,  64,  1, 1, 0, 'base',   0.027297, 0.039596, '137/ConvHipImplicitGemmGroupFwdXdlops'),
    ('fwd', 128, 30,  40,  128, 1, 1, 0, 'mtail',  0.011124, 0.013073, '220/ConvHipConv'),
    ('fwd', 64,  60,  80,  256, 1, 1, 0, 'base',   0.031679, 0.037143, '220/ConvHipConv'),
    ('fwd', 192, 120, 160, 48,  1, 1, 0, 'ntail',  0.080648, 0.077467, '137/ConvHipImplicitGemmGroupFwdXdlops'),
    ('fwd', 128, 30,  40,  128, 3, 3, 1, 'mtail',  0.035751, 0.036412, '220/ConvHipConv'),
    ('fwd', 128, 120, 160, 128, 3, 3, 1, 'base',   0.363031, 0.328420, '220/ConvHipConv'),
    ('fwd', 48,  120, 160, 128, 1, 1, 0, 'tdm',    0.062972, 0.059919, '137/ConvHipImplicitGemmGroupFwdXdlops'),
    ('bwd', 512, 1,   1,   32,  1, 1, 0, 'mtail',  0.009391, 0.009454, '220/ConvHipConv'),
    ('bwd', 128, 30,  40,  512, 1, 1, 0, 'mtail',  0.027835, 0.018570, '220/ConvHipConv'),
    ('bwd', 128, 30,  40,  128, 3, 3, 1, 'mtail',  0.056213, 0.023885, '220/ConvHipConv'),
    ('bwd', 64,  60,  80,  256, 1, 1, 0, 'base',   0.034608, 0.041991, '220/ConvHipConv'),
    ('bwd', 512, 30,  40,  128, 1, 1, 0, 'mtail',  0.035275, 0.021788, '220/ConvHipConv'),
    ('bwd', 128, 120, 160, 128, 3, 3, 1, 'base',   0.507215, 0.318301, '220/ConvHipConv'),
    ('bwd', 256, 30,  40,  128, 1, 1, 0, 'mtail',  0.024280, 0.013954, '220/ConvHipConv'),
    ('bwd', 64,  60,  80,  128, 1, 1, 0, 'base',   0.026471, 0.026644, '155/ConvHipImplicitGemmGroupBwdXdlops'),
    ('bwd', 192, 60,  80,  64,  1, 1, 0, 'base',   0.038520, 0.032831, '220/ConvHipConv'),
    ('bwd', 128, 30,  40,  128, 1, 1, 0, 'mtail',  0.017791, 0.012744, '220/ConvHipConv'),
    ('bwd', 256, 60,  80,  64,  1, 1, 0, 'base',   0.046168, 0.037164, '220/ConvHipConv'),
    ('wrw', 128, 30,  40,  128, 3, 3, 1, 'gsplit', 0.086537, 0.067399, '156/ConvHipImplicitGemmGroupWrwXdlops'),
    ('wrw', 128, 120, 160, 128, 3, 3, 1, 'gsplit', 0.664386, 0.413649, '156/ConvHipImplicitGemmGroupWrwXdlops'),
    ('wrw', 192, 60,  80,  64,  1, 1, 0, 'gsplit', 0.057199, 0.005608, '156/ConvHipImplicitGemmGroupWrwXdlops'),
    ('wrw', 256, 30,  40,  128, 1, 1, 0, 'gsplit', 0.035790, 0.028380, '156/ConvHipImplicitGemmGroupWrwXdlops'),
    ('wrw', 512, 30,  40,  128, 1, 1, 0, 'gsplit', 0.047279, 0.049966, '156/ConvHipImplicitGemmGroupWrwXdlops'),
    ('wrw', 128, 30,  40,  512, 1, 1, 0, 'gsplit', 0.052017, 0.052947, '156/ConvHipImplicitGemmGroupWrwXdlops'),
    ('wrw', 128, 30,  40,  128, 1, 1, 0, 'gsplit', 0.026844, 0.021975, '156/ConvHipImplicitGemmGroupWrwXdlops'),
    ('wrw', 64,  60,  80,  128, 1, 1, 0, 'gsplit', 0.051492, 0.040261, '156/ConvHipImplicitGemmGroupWrwXdlops'),
    ('wrw', 64,  60,  80,  256, 1, 1, 0, 'gsplit', 0.058101, 0.052136, '156/ConvHipImplicitGemmGroupWrwXdlops'),
    ('wrw', 256, 60,  80,  64,  1, 1, 0, 'gsplit', 0.058897, 0.052172, '156/ConvHipImplicitGemmGroupWrwXdlops'),
]

COST_RE = re.compile(r'\[(fwd|bwd|wrw):\s*\d+\][^\n]*?cost:([0-9.]+)ms')


def check_gpu_contention():
    try:
        out = subprocess.run(['rocm-smi', '--showuse', '--showpids'], capture_output=True,
                              text=True, timeout=10).stdout
        print("--- rocm-smi --showuse --showpids (contention snapshot) ---")
        print(out.strip())
        print("--- see docs/gfx1250_vendor_benchmark_vs_miopen.md for why this matters ---\n")
    except Exception as e:
        print(f"(rocm-smi check skipped: {e})\n")


def build_config(direction, config, build_dir, rebuild):
    config_file = CONFIG_FILES[(direction, config)]
    out_dir = os.path.join(build_dir, f'{direction}_{config}')
    exe = os.path.join(out_dir, 'conv_driver.exe')
    if os.path.exists(exe) and not rebuild:
        return out_dir
    os.makedirs(build_dir, exist_ok=True)
    print(f"building {direction}/{config} ({config_file}) -> {out_dir}")
    subprocess.run([sys.executable, 'igemm_codegen.py', '-d', out_dir, config_file],
                    cwd=REPO_ROOT, check=True)
    return out_dir


def run_shape(out_dir, direction, c, H, W, k, y, x, p, warmup, repeat, verify):
    args = [
        './conv_driver.exe', 'convbfp16',
        '-n', '42', '-c', str(c), '-H', str(H), '-W', str(W), '-k', str(k),
        '-y', str(y), '-x', str(x), '-p', str(p), '-q', str(p),
        '-u', '1', '-v', '1', '-l', '1', '-j', '1',
        '-m', 'conv', '-g', '1', '-F', str(FORW_FLAG[direction]), '-t', '1',
        '--in_layout', 'NHWC', '--fil_layout', 'NHWC', '--out_layout', 'NHWC',
        '-V', '1' if verify else '0',
    ]
    env = dict(os.environ, IGEMM_WARMUP=str(warmup), IGEMM_REPEAT=str(repeat))
    result = subprocess.run(args, cwd=out_dir, env=env, capture_output=True, text=True,
                             timeout=120)
    costs = [float(m.group(2)) for m in COST_RE.finditer(result.stdout) if m.group(1) == direction]
    if not costs:
        return None, result.stdout + result.stderr
    return min(costs), None


def fmt_ratio(misa_ms, ref_ms):
    if misa_ms is None or ref_ms is None or ref_ms <= 0:
        return 'n/a'
    ratio = misa_ms / ref_ms
    return f"{ratio:.2f}x slower" if ratio >= 1 else f"{1/ratio:.2f}x faster"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--direction', choices=['fwd', 'bwd', 'wrw', 'all'], default='all')
    ap.add_argument('--rebuild', action='store_true', help='force-rebuild every config even if already built')
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--repeat', type=int, default=20)
    ap.add_argument('--verify', action='store_true', help='run with -V 1 instead of -V 0')
    ap.add_argument('--build-dir', default=os.path.join(REPO_ROOT, 'bench_out'))
    ap.add_argument('--markdown-out', default=None)
    args = ap.parse_args()

    check_gpu_contention()

    shapes = [s for s in SHAPES if args.direction == 'all' or s[0] == args.direction]
    # candidate configs per shape = its own hardcoded config, plus 'gsplit' for
    # bwd/fwd (see GSPLIT_CANDIDATE) if not already the same and if a gsplit config
    # exists for that direction -- report whichever candidate is fastest.
    def candidates_for(direction, config):
        cands = [config]
        gsplit = GSPLIT_CANDIDATE.get(direction)
        if gsplit and gsplit != config and (direction, gsplit) in CONFIG_FILES:
            cands.append(gsplit)
        return cands

    needed_configs = sorted(set((s[0], cfg) for s in shapes for cfg in candidates_for(s[0], s[8])))
    built_dirs = {}
    for direction, config in needed_configs:
        built_dirs[(direction, config)] = build_config(direction, config, args.build_dir, args.rebuild)

    header = ("| Direction | Shape (c,H,W,k,y×x) | Config | MISA (ms) | MIOpen/gfx950 (ms) "
               "| MIOpen/gfx1250 (ms) | vs gfx950 | vs gfx1250 | MIOpen/gfx1250 solver |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    rows = []
    per_dir_ratios_950 = {}
    per_dir_ratios_1250 = {}

    for (direction, c, H, W, k, y, x, p, config, ref950, ref1250, solver1250) in shapes:
        best_ms, best_config, best_err = None, config, None
        for cand_config in candidates_for(direction, config):
            out_dir = built_dirs[(direction, cand_config)]
            cand_ms, cand_err = run_shape(out_dir, direction, c, H, W, k, y, x, p,
                                           args.warmup, args.repeat, args.verify)
            if cand_ms is not None and (best_ms is None or cand_ms < best_ms):
                best_ms, best_config, best_err = cand_ms, cand_config, None
            elif best_ms is None:
                best_err = cand_err
        misa_ms, config, err = best_ms, best_config, best_err
        shape_str = f"{c},{H},{W},{k},{y}x{x}"
        if misa_ms is None:
            rows.append(f"| {direction} | {shape_str} | {config} | FAILED | {ref950} | {ref1250} | - | - | {solver1250} |")
            print(f"[{direction} {shape_str}] FAILED to parse cost -- raw output:\n{err}", file=sys.stderr)
            continue
        r950 = fmt_ratio(misa_ms, ref950)
        r1250 = fmt_ratio(misa_ms, ref1250)
        rows.append(f"| {direction} | {shape_str} | {config} | {misa_ms:.5f} | {ref950:.5f} | "
                     f"{ref1250:.5f} | {r950} | {r1250} | {solver1250} |")
        per_dir_ratios_950.setdefault(direction, []).append(misa_ms / ref950)
        per_dir_ratios_1250.setdefault(direction, []).append(misa_ms / ref1250)
        print(f"[{direction} {shape_str} {config}] MISA={misa_ms:.5f}ms  "
              f"gfx950={ref950:.5f}ms ({r950})  gfx1250={ref1250:.5f}ms ({r1250})")

    lines = [header, sep] + rows
    lines.append("")
    lines.append("Summary (avg ratio, MISA/MIOpen -- >1 means MISA is slower):")
    for d in ['fwd', 'bwd', 'wrw']:
        if d in per_dir_ratios_950:
            avg950 = sum(per_dir_ratios_950[d]) / len(per_dir_ratios_950[d])
            avg1250 = sum(per_dir_ratios_1250[d]) / len(per_dir_ratios_1250[d])
            lines.append(f"- {d}: vs gfx950 avg {avg950:.2f}x, vs gfx1250 avg {avg1250:.2f}x "
                          f"({len(per_dir_ratios_950[d])} shapes)")

    table_md = "\n".join(lines)
    print("\n" + table_md)

    if args.markdown_out:
        with open(args.markdown_out, 'w') as f:
            f.write(table_md + "\n")
        print(f"\nwrote {args.markdown_out}")


if __name__ == '__main__':
    main()
