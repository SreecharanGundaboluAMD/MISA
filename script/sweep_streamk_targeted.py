#!/usr/bin/env python3
"""
Targeted sweep: vary total_shards (via STREAMK_MAX_SHARDS) and grid_z (via STREAMK_GRID_Z)
independently, with verification, to find whether any combination makes streamk competitive.
"""
import subprocess, os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STREAMK_DIR = os.path.join(REPO, "out", "wrw_bf16_streamk")
GSPLIT_DIR  = os.path.join(REPO, "out", "wrw_bf16_all")

# Only 1x1 shapes (nxe=0, streamk-compatible)
SHAPES = [
    (128, 30, 40, 128, 1, 1, 0, 1),   # num_k_blocks=1575, divisors: 1,3,5,7,9,15,21,25,35,45,63,75,105,175,225,315,525,1575
    (256, 30, 40, 128, 1, 1, 0, 1),   # same num_k_blocks
    (128, 120, 160, 128, 1, 1, 0, 1), # num_k_blocks=25200, many divisors
]

# (max_shards, grid_z) combos -- grid_z=0 means "auto" (target_total_workers)
COMBOS = [
    (225, 0),    # old default behavior (grid_z=225, max_iters=1)
    (225, 128),  # fewer workers, same shards, max_iters=2
    (225, 64),   # even fewer workers, max_iters=4
    (225, 32),   # minimal grid, max_iters=8
    (525, 0),    # more shards, auto grid
    (525, 256),  # more shards, full CU grid, max_iters=3
    (525, 128),  # more shards, half grid, max_iters=5
    (1575, 256), # finest shards, full grid, max_iters=7
    (1575, 128), # finest shards, half grid, max_iters=13
    (315, 0),    # medium shards
    (315, 128),  # medium shards, half grid
    (175, 0),    # fewer shards, bigger K per shard
    (175, 128),  # fewer shards, half grid
    (105, 0),    # even fewer
    (105, 64),   # even fewer, small grid
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
    valid = None
    for line in result.stdout.splitlines():
        m = re.search(r'cost:([\d.]+)ms', line)
        vm = re.search(r'valid:(\w+)', line)
        if m:
            line_valid = vm.group(1) if vm else None
            if line_valid:
                valid = line_valid
            if line_valid == 'y':
                costs.append(float(m.group(1)))
        if "STREAMK_DEBUG" in line:
            debug_line = re.sub(r'p_streamk_counter=0x[0-9a-f]+ ', '', line.strip()).replace('STREAMK_DEBUG: ', '')
    best = min(costs) if costs else float('inf')
    return best, valid, debug_line

def main():
    for shape in SHAPES:
        c, H, W, k, y, x, p, u = shape
        num_k_blocks = (42 * H * W) // 32
        shape_str = f"c={c},H={H},W={W},k={k},{y}x{x}"
        print(f"\n{'='*120}")
        print(f"SHAPE: {shape_str}  (num_k_blocks={num_k_blocks})")
        print(f"{'='*120}")
        print(f"{'max_shards':>10} {'grid_z':>7} {'Time(ms)':>10} {'Valid':>6} {'Debug'}")
        print("-" * 120)

        for max_shards, grid_z in COMBOS:
            env = {"STREAMK_MAX_SHARDS": str(max_shards), "STREAMK_CLAIMS_PER_WORKER": "1"}
            if grid_z > 0:
                env["STREAMK_GRID_Z"] = str(grid_z)
            try:
                best, valid, debug = run_shape(shape, env)
                print(f"{max_shards:>10} {grid_z:>7} {best:>10.3f} {valid:>6} {debug}")
            except Exception as e:
                print(f"{max_shards:>10} {grid_z:>7} {'ERR':>10} {'':>6} {e}")

    # Also print gsplit best for reference
    print(f"\n{'='*120}")
    print("GSPLIT (_all config) best for reference:")
    print(f"{'='*120}")
    for shape in SHAPES:
        c, H, W, k, y, x, p, u = shape
        shape_str = f"c={c},H={H},W={W},k={k},{y}x{x}"
        args = [
            os.path.join(GSPLIT_DIR, "conv_driver.exe"), "convbfp16",
            "-n", "42", "-c", str(c), "-H", str(H), "-W", str(W), "-k", str(k),
            "-y", str(y), "-x", str(x), "-p", str(p), "-q", str(p),
            "-u", str(u), "-v", str(u), "-l", "1", "-j", "1", "-g", "1",
            "-F", "4", "-t", "1",
            "--in_layout", "NHWC", "--fil_layout", "NHWC", "--out_layout", "NHWC",
            "-V", "1",
        ]
        env = os.environ.copy()
        env["IGEMM_WARMUP"] = "5"
        env["IGEMM_REPEAT"] = "10"
        result = subprocess.run(args, capture_output=True, text=True, timeout=120, env=env, cwd=GSPLIT_DIR)
        costs = []
        for line in result.stdout.splitlines():
            m = re.search(r'cost:([\d.]+)ms', line)
            if m and 'valid:y' in line:
                costs.append(float(m.group(1)))
        best = min(costs) if costs else float('inf')
        print(f"  {shape_str}: gsplit best = {best:.3f}ms")

if __name__ == "__main__":
    main()
