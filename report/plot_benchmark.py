#!/usr/bin/env python3
"""
CUDA Block Size Benchmark Plotter
Plots execution time vs block size from benchmark results
"""

import matplotlib.pyplot as plt
import numpy as np

# Benchmark data
block_sizes = [32, 64, 128, 256, 512, 1024]
avg_times_ms = [23450, 23419, 23541, 23807, 23676, 23476]
ticks = [505636, 504909, 527078, 519953, 489610, 527146]

# Calculate time per tick
time_per_tick = [t / tk * 1000 for t, tk in zip(avg_times_ms, ticks)]  # microseconds

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Execution Time vs Block Size
ax1.bar(range(len(block_sizes)), avg_times_ms, color='steelblue', edgecolor='black')
ax1.set_xticks(range(len(block_sizes)))
ax1.set_xticklabels(block_sizes)
ax1.set_xlabel('Block Size (threads per block)', fontsize=12)
ax1.set_ylabel('Execution Time (ms)', fontsize=12)
ax1.set_title('CUDA Block Size vs Execution Time\n(512x512 grid, 1000 ants, 14000 food)', fontsize=12)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(avg_times_ms):
    ax1.text(i, v + 100, f'{v}', ha='center', va='bottom', fontsize=10)

# Highlight best
min_idx = avg_times_ms.index(min(avg_times_ms))
ax1.bar(min_idx, avg_times_ms[min_idx], color='green', edgecolor='black', label=f'Best: {block_sizes[min_idx]}')
ax1.legend()

# Plot 2: Time per Tick vs Block Size
ax2.bar(range(len(block_sizes)), time_per_tick, color='coral', edgecolor='black')
ax2.set_xticks(range(len(block_sizes)))
ax2.set_xticklabels(block_sizes)
ax2.set_xlabel('Block Size (threads per block)', fontsize=12)
ax2.set_ylabel('Time per Tick (μs)', fontsize=12)
ax2.set_title('CUDA Block Size vs Time per Tick', fontsize=12)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for i, v in enumerate(time_per_tick):
    ax2.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('block_size_benchmark.png', dpi=150, bbox_inches='tight')
print("Saved: block_size_benchmark.png")

# Print summary
print("\n=== Summary ===")
print(f"{'Block Size':<12} {'Time (ms)':<12} {'Ticks':<12} {'μs/tick':<12}")
print("-" * 48)
for bs, t, tk, tpt in zip(block_sizes, avg_times_ms, ticks, time_per_tick):
    print(f"{bs:<12} {t:<12} {tk:<12} {tpt:<12.2f}")

best_idx = avg_times_ms.index(min(avg_times_ms))
print(f"\nBest block size: {block_sizes[best_idx]} ({avg_times_ms[best_idx]} ms)")

# Show plot
plt.show()
