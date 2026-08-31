#!/usr/bin/env python3
"""
Sweep STREAMK_* env vars to find the best sizing for each shape.
Tests the hypothesis that the persistent grid needs to be SMALL (≈num_cu) and
shards need to be FEWER than workers (max_iters > 1) for persistence to help.
"""
import subprocess, os, re, sys, itertools

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STREAMK_DIR = os.path.join(REPO, "out", "wrw_bf16_streamk")

SHAPES = [
    (128, 30, 40, 128, 1, 1, 0, 1),
    (256, 30, 40, 128, 1, 1, 0, 1),
    (128, 120, 160, 128, 1, 1, 0, 1),
    (128, 30, 40, 128, 3, 3, 1, 1),
]

# Sweep grid: blocks_per_cu x claims_per_worker x max_shards x divide_by_tiles
SWEEPS = [
    # (label, env_dict)
    ("default", {}),
    ("bpc=1,cpw=1,ms=256", {"STREAMK_CLAIMS_PER_WORKER": "1", "STREAMK_MAX_SHARDS": "256"}),
    ("bpc=1,cpw=2,ms=256", {"STREAMK_CLAIMS_PER_WORKER": "2", "STREAMK_MAX_SHARDS": "256"}),
    ("bpc=1,cpw=1,ms=128", {"STREAMK_CLAIMS_PER_WORKER": "1", "STREAMK_MAX_SHARDS": "128"}),
    ("bpc=1,cpw=1,ms=64",  {"STREAMK_CLAIMS_PER_WORKER": "1", "STREAMK_MAX_SHARDS": "64"}),
    ("bpc=2,cpw=1,ms=256", {"STREAMK_BLOCKS_PER_CU": "2", "STREAMK_CLAIMS_PER_WORKER": "1", "STREAMK_MAX_SHARDS": "256"}),
    ("bpc=2,cpw=2,ms=256", {"STREAMK_BLOCKS_PER_CU": "2", "STREAMK_CLAIMS_PER_WORKER": "2", "STREAMK_MAX_SHARDS": "256"}),
    ("bpc=4,cpw=1,ms=256", {"STREAMK_BLOCKS_PER_CU": "4", "STREAMK_CLAIMS_PER_WORKER": "1", "STREAMK_MAX_SHARDS": "256"}),
    ("bpc=1,cpw=4,ms=256,div", {"STREAMK_CLAIMS_PER_WORKER": "4", "STREAMK_MAX_SHARDS": "256", "STREAMK_DIVIDE_BY_TILES": "1"}),
    ("bpc=1,cpw=1,ms=256,div", {"STREAMK_CLAIMS_PER_WORKER": "1", "STREAMK_MAX_SHARDS": "256", "STREAMK_DIVIDE_BY_TILES": "1"}),
    # Force small grid, many iters
    ("bpc=1,cpw=1,ms=32",  {"STREAMK_CLAIMS_PER_WORKER": "1", "STREAMK_MAX_SHARDS": "32"}),
    ("bpc=1,cpw=1,ms=16",  {"STREAMK_CLAIMS_PER_WORKER": "1", "STREAMK_MAX_SHARDS": "16"}),
    # Very few shards, big K per shard
    ("bpc=1,cpw=1,ms=8",   {"STREAMK_CLAIMS_PER_WORKER": "1", "STREAMK_MAX_SHARDS": "8"}),
]

def run_shape(shape, env_overrides, warmup=3, repeat=5):
    c, H, W, k, y, x, p, u = shape
    args = [
        os.path.join(STREAMK_DIR, "conv_driver.exe"), "convbfp16",
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
    env["STREAMK_DEBUG"] = "1"
    env.update(env_overrides)
    result = subprocess.run(args, capture_output=True, text=True, timeout=120, env=env, cwd=STREAMK_DIR)
    costs = []
    debug_line = ""
    for line in result.stdout.splitlines():
        m = re.search(r'cost:([\d.]+)ms', line)
        if m and 'valid:y' in line:
            costs.append(float(m.group(1)))
        if "STREAMK_DEBUG" in line:
            debug_line = line.strip()
    best = min(costs) if costs else float('inf')
    return best, debug_line

def main():
    for shape in SHAPES:
        c, H, W, k, y, x, p, u = shape
        shape_str = f"c={c},H={H},W={W},k={k},{y}x{x}"
        print(f"\n{'='*100}")
        print(f"SHAPE: {shape_str}  (gemm_m={k}, gemm_n={c*y*x}, gemm_k=42*H*W)")
        print(f"{'='*100}")
        print(f"{'Config':<30} {'Time(ms)':>10} {'Debug Info'}")
        print("-" * 100)

        results = []
        for label, env_dict in SWEEPS:
            try:
                best, debug = run_shape(shape, env_dict)
                # Extract just the key parts from debug
                debug_short = re.sub(r'p_streamk_counter=0x[0-9a-f]+ ', '', debug)
                debug_short = debug_short.replace('STREAMK_DEBUG: ', '')
                results.append((best, label, debug_short))
                print(f"{label:<30} {best:>10.3f} {debug_short}")
            except Exception as e:
                print(f"{label:<30} {'ERROR':>10} {e}")

        results.sort()
        print(f"\nBest: {results[0][1]} = {results[0][0]:.3f}ms  ({results[0][2]})")

if __name__ == "__main__":
    main()
