#!/usr/bin/env python3
"""
A/B benchmark: wrw_streamk vs _gsplit (ternary search) on real wrw shapes.

Runs each shape N times on each config, reports per-run timings so variance
(not just mean) is visible. The GPU is reported as idle (100% use is a driver
bug), so these should be trustworthy absolute numbers.

Usage:
    python3 script/bench_streamk_vs_gsplit.py [--repeats N] [--warmup N]
"""
import subprocess, os, re, statistics, sys, argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STREAMK_DIR = os.path.join(REPO, "out", "wrw_bf16_streamk")
GSPLIT_DIR  = os.path.join(REPO, "out", "wrw_bf16_all")

# Shapes from docs/gfx1250_vendor_benchmark_vs_miopen.md's wrw trace (batch=42, 1x1)
# These are the small-grid shapes where _gsplit launches many tiny workgroups
SHAPES = [
    # (c, H, W, k, y, x, pad, stride)  -- all batch=42, 1x1
    (128, 30, 40, 128, 1, 1, 0, 1),   # gemm_m=128, gemm_n=128, gemm_k=50400
    (256, 30, 40, 128, 1, 1, 0, 1),   # gemm_m=128, gemm_n=256, gemm_k=50400
    (512, 30, 40, 128, 1, 1, 0, 1),   # gemm_m=128, gemm_n=512, gemm_k=50400
    (128, 30, 40, 512, 1, 1, 0, 1),   # gemm_m=512, gemm_n=128, gemm_k=50400
    (64,  60, 80, 128, 1, 1, 0, 1),   # gemm_m=128, gemm_n=64,  gemm_k=201600
    (256, 60, 80, 64,  1, 1, 0, 1),   # gemm_m=64,  gemm_n=256, gemm_k=201600
    (128, 120,160, 128, 1, 1, 0, 1),  # gemm_m=128, gemm_n=128, gemm_k=806400
    # 3x3 shapes (still nxe==0 so streamk-compatible)
    (128, 30, 40, 128, 3, 3, 1, 1),   # gemm_m=128, gemm_n=1152, gemm_k=50400
]

def run_shape(exe_dir, shape, warmup, repeat):
    c, H, W, k, y, x, p, u = shape
    args = [
        os.path.join(exe_dir, "conv_driver.exe"), "convbfp16",
        "-n", "42", "-c", str(c), "-H", str(H), "-W", str(W), "-k", str(k),
        "-y", str(y), "-x", str(x), "-p", str(p), "-q", str(p),
        "-u", str(u), "-v", str(u), "-l", "1", "-j", "1", "-g", "1",
        "-F", "4", "-t", "1",
        "--in_layout", "NHWC", "--fil_layout", "NHWC", "--out_layout", "NHWC",
        "-V", "1",
    ]
    env = os.environ.copy()
    env["IGEMM_WARMUP"] = str(warmup)
    env["IGEMM_REPEAT"] = str(repeat)
    result = subprocess.run(args, capture_output=True, text=True, timeout=120, env=env, cwd=exe_dir)
    # Parse cost lines — keep only valid:y lines so a fast-but-wrong kernel
    # can never win the min(costs) comparison.
    costs = []
    valid = None
    for line in result.stdout.splitlines():
        m = re.search(r'cost:([\d.]+)ms', line)
        if m:
            vm = re.search(r'valid:(\w+)', line)
            line_valid = vm.group(1) if vm else None
            if line_valid:
                valid = line_valid
            if line_valid == 'y':
                costs.append(float(m.group(1)))
    best = min(costs) if costs else float('inf')
    return best, valid, costs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10, help="number of repeated runs per shape per config")
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations per run")
    args = parser.parse_args()

    print(f"{'Shape':<30} {'Config':<8} {'Runs (ms)':>60} {'Mean':>8} {'StdDev':>8} {'Min':>8} {'Max':>8} {'CV%':>6}")
    print("-" * 140)

    for shape in SHAPES:
        c, H, W, k, y, x, p, u = shape
        shape_str = f"c={c},H={H},W={W},k={k},{y}x{x}"
        row_data = {}

        for config_name, exe_dir in [("gsplit", GSPLIT_DIR), ("streamk", STREAMK_DIR)]:
            if not os.path.exists(os.path.join(exe_dir, "conv_driver.exe")):
                print(f"  {config_name} binary not found at {exe_dir}, skipping")
                continue
            runs = []
            for i in range(args.repeats):
                try:
                    best, valid, _ = run_shape(exe_dir, shape, args.warmup, 1)
                    runs.append(best)
                except Exception as e:
                    print(f"  ERROR on {config_name} {shape_str} run {i}: {e}")
                    runs.append(float('nan'))

            mean = sum(runs) / len(runs)
            variance = sum((r - mean) ** 2 for r in runs) / (len(runs) - 1) if len(runs) > 1 else 0
            stddev = variance ** 0.5
            cv = (stddev / mean * 100) if mean > 0 else 0
            runs_str = " ".join(f"{r:.3f}" for r in runs)
            print(f"{shape_str:<30} {config_name:<8} {runs_str:>60} {mean:>8.3f} {stddev:>8.3f} {min(runs):>8.3f} {max(runs):>8.3f} {cv:>5.1f}%")
            row_data[config_name] = {"runs": runs, "mean": mean, "stddev": stddev, "cv": cv}

        # Summary comparison
        if "gsplit" in row_data and "streamk" in row_data:
            g = row_data["gsplit"]
            s = row_data["streamk"]
            ratio = s["mean"] / g["mean"] if g["mean"] > 0 else float('inf')
            print(f"{'':>30} {'RATIO':<8} streamk/gsplit mean={ratio:.2f}x  gsplit CV={g['cv']:.1f}%  streamk CV={s['cv']:.1f}%")
        print()

if __name__ == "__main__":
    main()
