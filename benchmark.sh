#!/bin/bash

# Ant Colony Simulation Benchmark Script

echo "=== Ant Colony Simulation Benchmark ==="
echo ""

# Build all versions
echo "Building all versions..."
make cpu openmp 2>/dev/null

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo ""
echo "=== Configuration ==="
echo "Grid size: 256 x 256"
echo "Number of ants: 500"
echo "Food sources: 10"
echo "Food per source: 20"
echo ""

SEED=${1:-42}
echo "Random seed: $SEED"
echo ""

echo "=== Running Benchmarks ==="
echo ""

# CPU Version
echo "--- CPU (Sequential) ---"
./build/ant_cpu $SEED 2>&1 | grep -E "(Execution time|Time per tick|Food collected|Total ticks)"
echo ""

# OpenMP with different thread counts
for threads in 2 4 8; do
    echo "--- OpenMP ($threads threads) ---"
    ./build/ant_openmp $SEED $threads 2>&1 | grep -E "(Execution time|Time per tick|Food collected|Total ticks)"
    echo ""
done

# CUDA if available
if command -v ./build/ant_cuda &> /dev/null; then
    echo "--- CUDA (GPU) ---"
    ./build/ant_cuda $SEED 2>&1 | grep -E "(Execution time|Time per tick|Food collected|Total ticks|CUDA Device)"
    echo ""
fi

echo "=== Benchmark Complete ==="
