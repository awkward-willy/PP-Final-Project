#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH -A ACD114118
#SBATCH -t 00:30:00
#SBATCH -J cuda_block_bench

# CUDA Block Size Benchmark
# Tests different block sizes to find optimal configuration

echo "=== CUDA Block Size Benchmark ==="
echo ""

module load cmake
module load cuda

cd ~/pp_fp

# Block sizes to test (must be powers of 2 or multiples of 32)
BLOCK_SIZES=(32 64 128 256 512 1024)

SEED=42

echo "Testing block sizes: ${BLOCK_SIZES[*]}"
echo "Seed: $SEED"
echo ""
echo "============================================"

for block_size in "${BLOCK_SIZES[@]}"; do
    echo ""
    echo "--- Block Size: $block_size ---"
    
    # Rebuild with new block size
    rm -rf build
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON \
          -DCMAKE_CUDA_FLAGS="-DBLOCK_SIZE=$block_size" .. > /dev/null 2>&1
    make -j$(nproc) ant_cuda > /dev/null 2>&1
    
    if [ ! -f "./ant_cuda" ]; then
        echo "Build failed for block_size=$block_size"
        cd ~/pp_fp
        continue
    fi
    
    cd ~/pp_fp
    
    # Run 3 times and average
    total_time=0
    for run in 1 2 3; do
        OUTPUT=$(./build/ant_cuda $SEED 2>&1)
        TIME=$(echo "$OUTPUT" | grep "Execution time" | sed 's/.*: \([0-9]*\) ms/\1/')
        TICKS=$(echo "$OUTPUT" | grep "Total ticks" | sed 's/.*: \([0-9]*\)/\1/')
        FOOD=$(echo "$OUTPUT" | grep "Food collected" | sed 's/.*: \([0-9]*\/[0-9]*\)/\1/')
        total_time=$((total_time + TIME))
    done
    
    avg_time=$((total_time / 3))
    echo "Avg time: ${avg_time}ms | Ticks: $TICKS | Food: $FOOD"
done

echo ""
echo "============================================"
echo "Benchmark Complete"
