#!/usr/bin/env python3
"""
CUDA Grid Size Benchmark Plot
"""

import matplotlib.pyplot as plt
import numpy as np

# Benchmark data from grid_size.txt
grid_sizes = [256, 512, 1024, 2048]
grid_labels = ['256×256', '512×512', '1024×1024', '2048×2048']
grid_cells = [s*s for s in grid_sizes]  # Total cells

execution_time_ms = [2328, 3432, 8325, 22039]
time_per_tick_ms = [0.0357956, 0.0424569, 0.0929544, 0.24859]
total_ticks = [65036, 80835, 89560, 88656]

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Color gradient
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(grid_sizes)))

# Plot 1: Total Execution Time vs Grid Size
ax1 = axes[0]
bars1 = ax1.bar(grid_labels, execution_time_ms, color=colors, edgecolor='black', linewidth=1.2)
ax1.set_xlabel('Grid Size', fontsize=12)
ax1.set_ylabel('Execution Time (ms)', fontsize=12)
ax1.set_title('Total Execution Time vs Grid Size', fontsize=14, fontweight='bold')
ax1.set_ylim(0, max(execution_time_ms) * 1.15)
for bar, val in zip(bars1, execution_time_ms):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300, 
             f'{val:,} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Time per Tick vs Grid Size
ax2 = axes[1]
bars2 = ax2.bar(grid_labels, time_per_tick_ms, color=colors, edgecolor='black', linewidth=1.2)
ax2.set_xlabel('Grid Size', fontsize=12)
ax2.set_ylabel('Time per Tick (ms)', fontsize=12)
ax2.set_title('Time per Tick vs Grid Size', fontsize=14, fontweight='bold')
ax2.set_ylim(0, max(time_per_tick_ms) * 1.15)
for bar, val in zip(bars2, time_per_tick_ms):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Calculate scaling factor
scaling_256_to_2048 = execution_time_ms[3] / execution_time_ms[0]
cells_ratio = grid_cells[3] / grid_cells[0]

# Add configuration info
config_text = (
    f"Configuration:\n"
    f"Food: 7 sources × 2000 = 14000\n"
    f"Ants: 5000 | GPU: Tesla V100\n"
    f"Scaling: {scaling_256_to_2048:.1f}x time for {cells_ratio:.0f}x cells"
)
fig.text(0.98, 0.02, config_text, fontsize=9, family='monospace',
         ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('CUDA Performance vs Grid Size', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0.12, 1, 1])

# Save
plt.savefig('grid_size_benchmark.png', dpi=150, bbox_inches='tight')
plt.savefig('grid_size_benchmark.pdf', bbox_inches='tight')
print("Saved: grid_size_benchmark.png, grid_size_benchmark.pdf")

# Print summary
print("\n" + "="*70)
print("GRID SIZE BENCHMARK SUMMARY")
print("="*70)
print(f"{'Grid Size':<15} {'Cells':<12} {'Time (ms)':<12} {'ms/tick':<12} {'Ticks':<10}")
print("-"*70)
for gs, gl, c, t, tpt, tk in zip(grid_sizes, grid_labels, grid_cells, execution_time_ms, time_per_tick_ms, total_ticks):
    print(f"{gl:<15} {c:<12,} {t:<12,} {tpt:<12.4f} {tk:<10,}")
print("="*70)
print(f"\nScaling from 256×256 to 2048×2048:")
print(f"  Grid cells: {cells_ratio:.0f}x increase")
print(f"  Execution time: {scaling_256_to_2048:.1f}x increase")
print(f"  Efficiency: {cells_ratio/scaling_256_to_2048:.2f}x better than linear")

plt.show()
