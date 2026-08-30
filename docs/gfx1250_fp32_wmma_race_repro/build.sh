#!/usr/bin/env bash
# Build script for the standalone wmma_f32_race_repro reproducer.
# No MISA/Python dependency -- just a ROCm toolchain (clang++ + hipcc).
set -euo pipefail
cd "$(dirname "$0")"

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

echo "assembling kernel.s -> kernel.hsaco"
"$ROCM_PATH/llvm/bin/clang++" -x assembler -target amdgcn--amdhsa \
    -mcpu=gfx1250 kernel.s -o kernel.hsaco

echo "compiling host.cpp -> repro"
"$ROCM_PATH/bin/hipcc" -std=c++17 host.cpp -o repro

echo "done. run with: ./repro [num_workgroups]"
