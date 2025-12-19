#!/usr/bin/env python3
"""
CPU vs CUDA Benchmark Comparison Plot
"""

import matplotlib.pyplot as plt
import numpy as np

# Benchmark data
versions = ['CPU (Sequential)', 'CUDA (GPU)']
execution_time_ms = [52643, 2900]
time_per_tick_ms = [0.838051, 0.0390736]
total_ticks = [62816, 74219]

# Calculate speedup
speedup_total = execution_time_ms[0] / execution_time_ms[1]
speedup_per_tick = time_per_tick_ms[0] / time_per_tick_ms[1]

# Create figure with subplots - adjusted canvas size
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Color scheme
colors = ['#3498db', '#2ecc71']  # Blue for CPU, Green for CUDA

# Plot 1: Total Execution Time
ax1 = axes[0]
bars1 = ax1.bar(versions, execution_time_ms, color=colors, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Execution Time (ms)', fontsize=12)
ax1.set_title('Total Execution Time', fontsize=14, fontweight='bold')
ax1.set_ylim(0, max(execution_time_ms) * 1.15)
for i, (bar, val) in enumerate(zip(bars1, execution_time_ms)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
             f'{val:,} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.axhline(y=execution_time_ms[1], color='gray', linestyle='--', alpha=0.5)
ax1.text(0.5, execution_time_ms[0] * 0.5, f'⚡ {speedup_total:.1f}x faster', 
         ha='center', fontsize=16, color='white', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#27ae60', edgecolor='none', alpha=0.9))

# Plot 2: Time per Tick
ax2 = axes[1]
bars2 = ax2.bar(versions, time_per_tick_ms, color=colors, edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Time per Tick (ms)', fontsize=12)
ax2.set_title('Time per Tick', fontsize=14, fontweight='bold')
ax2.set_ylim(0, max(time_per_tick_ms) * 1.15)
for i, (bar, val) in enumerate(zip(bars2, time_per_tick_ms)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.4f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.axhline(y=time_per_tick_ms[1], color='gray', linestyle='--', alpha=0.5)
ax2.text(0.5, time_per_tick_ms[0] * 0.5, f'⚡ {speedup_per_tick:.1f}x faster', 
         ha='center', fontsize=16, color='white', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#27ae60', edgecolor='none', alpha=0.9))

# Plot 3: Speedup Summary
ax3 = axes[2]
speedup_labels = ['Total Time\nSpeedup', 'Per-Tick\nSpeedup']
speedup_values = [speedup_total, speedup_per_tick]
bars3 = ax3.bar(speedup_labels, speedup_values, color=['#e74c3c', '#9b59b6'], edgecolor='black', linewidth=1.2)
ax3.set_ylabel('Speedup (x times faster)', fontsize=12)
ax3.set_title('CUDA Speedup vs CPU', fontsize=14, fontweight='bold')
ax3.set_ylim(0, max(speedup_values) * 1.15)
for bar, val in zip(bars3, speedup_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.1f}x', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add configuration info - positioned at bottom right to avoid overlap
config_text = (
    "Configuration:\n"
    "Grid: 256×256 | Ants: 500\n"
    "Food: 10 sources × 20 = 200\n"
    "Seed: 42 | GPU: Tesla V100"
)
fig.text(0.98, 0.02, config_text, fontsize=9, family='monospace',
         ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Ant Colony Simulation: CPU vs CUDA Performance', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0.15, 1, 1])  # More space at bottom for config

# Save
plt.savefig('cpu_vs_cuda_benchmark.png', dpi=150, bbox_inches='tight')
plt.savefig('cpu_vs_cuda_benchmark.pdf', bbox_inches='tight')
print("Saved: cpu_vs_cuda_benchmark.png, cpu_vs_cuda_benchmark.pdf")

# Print summary table
print("\n" + "="*60)
print("BENCHMARK SUMMARY")
print("="*60)
print(f"{'Metric':<25} {'CPU':<15} {'CUDA':<15} {'Speedup':<10}")
print("-"*60)
print(f"{'Execution Time (ms)':<25} {execution_time_ms[0]:<15,} {execution_time_ms[1]:<15,} {speedup_total:.1f}x")
print(f"{'Time per Tick (ms)':<25} {time_per_tick_ms[0]:<15.4f} {time_per_tick_ms[1]:<15.4f} {speedup_per_tick:.1f}x")
print(f"{'Total Ticks':<25} {total_ticks[0]:<15,} {total_ticks[1]:<15,}")
print("="*60)

plt.show()
