#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH -A ACD114118
#SBATCH -t 00:10:00
#SBATCH -J cuda_test

# CUDA Correctness Test Script
# Tests multiple seeds to verify food collection is accurate

echo "=== CUDA Correctness Test ==="
echo ""

# Load required modules
module load cmake
module load cuda

# Build
cd ~/pp_fp
mkdir -p build
cd build
if [ ! -f "Makefile" ]; then
    cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON ..
fi
make -j$(nproc) ant_cuda 2>&1 | tail -5

if [ ! -f "./ant_cuda" ]; then
    echo "Build failed!"
    exit 1
fi

cd ~/pp_fp

echo ""
echo "Testing CUDA version with multiple seeds..."
echo "============================================"
echo ""

PASSED=0
FAILED=0
SEEDS=(42 123 456 789 1000 2024 3141 9999 12345 99999)

for seed in "${SEEDS[@]}"; do
    echo -n "Seed $seed: "
    
    # Run and capture output
    OUTPUT=$(./build/ant_cuda $seed 2>&1)
    
    # Extract food collected line
    FOOD_LINE=$(echo "$OUTPUT" | grep "Food collected")
    
    if [ -z "$FOOD_LINE" ]; then
        echo "FAILED (no output)"
        ((FAILED++))
        continue
    fi
    
    # Parse "Food collected: X/Y"
    COLLECTED=$(echo "$FOOD_LINE" | sed 's/.*: \([0-9]*\)\/\([0-9]*\)/\1/')
    TOTAL=$(echo "$FOOD_LINE" | sed 's/.*: \([0-9]*\)\/\([0-9]*\)/\2/')
    
    if [ "$COLLECTED" -eq "$TOTAL" ]; then
        # Extract execution time
        TIME=$(echo "$OUTPUT" | grep "Execution time" | sed 's/.*: \([0-9]*\) ms/\1/')
        echo "PASSED ($COLLECTED/$TOTAL) - ${TIME}ms"
        ((PASSED++))
    else
        echo "FAILED ($COLLECTED/$TOTAL) - Food mismatch!"
        ((FAILED++))
    fi
done

echo ""
echo "============================================"
echo "Results: $PASSED passed, $FAILED failed"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "All tests PASSED!"
    exit 0
else
    echo "Some tests FAILED!"
    exit 1
fi
