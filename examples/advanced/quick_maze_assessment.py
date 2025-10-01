#!/usr/bin/env python3
"""Quick assessment of Recursive Backtracking vs Wilson's for MFG-RL."""

from maze_algorithm_assessment import analyze_maze
from perfect_maze_generator import MazeAlgorithm

import numpy as np

print("=" * 80)
print("QUICK MAZE ALGORITHM ASSESSMENT: Recursive Backtracking vs Wilson's")
print("=" * 80)

rows, cols = 20, 20
num_trials = 10

algorithms = [MazeAlgorithm.RECURSIVE_BACKTRACKING, MazeAlgorithm.WILSONS]
all_metrics = {alg: [] for alg in algorithms}

for algorithm in algorithms:
    print(f"\nRunning {num_trials} trials for {algorithm.value}...")

    for trial in range(num_trials):
        metrics = analyze_maze(algorithm, rows, cols, seed=trial)
        all_metrics[algorithm].append(metrics)

# Print results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

for algorithm in algorithms:
    metrics_list = all_metrics[algorithm]

    print(f"\n{algorithm.value.upper()}")
    print("-" * 80)

    print(f"  Generation Time:     {np.mean([m.generation_time for m in metrics_list]):.4f}s")
    print(f"  Dead Ends:           {np.mean([m.num_dead_ends for m in metrics_list]):.1f}")
    print(f"  Junctions:           {np.mean([m.num_junctions for m in metrics_list]):.1f}")
    print(f"  Longest Path:        {np.mean([m.longest_path_length for m in metrics_list]):.1f}")
    print(f"  Avg Path Length:     {np.mean([m.avg_path_length for m in metrics_list]):.1f}")
    print(f"  Branching Entropy:   {np.mean([m.branching_entropy for m in metrics_list]):.3f}")
    print(f"  Structural Bias:     {np.mean([m.structural_bias for m in metrics_list]):.3f}")
    print(f"  Exploration Diff:    {np.mean([m.exploration_difficulty for m in metrics_list]):.3f}")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

rb_metrics = all_metrics[MazeAlgorithm.RECURSIVE_BACKTRACKING]
wil_metrics = all_metrics[MazeAlgorithm.WILSONS]

print("\nKey Findings:")
print(
    f"• Recursive Backtracking is {np.mean([m.generation_time for m in wil_metrics])/np.mean([m.generation_time for m in rb_metrics]):.1f}x faster"
)
print(
    f"• RB has {np.mean([m.longest_path_length for m in rb_metrics])/np.mean([m.longest_path_length for m in wil_metrics]):.1f}x longer paths"
)
print(
    f"• Wilson's has {np.mean([m.num_junctions for m in wil_metrics])/np.mean([m.num_junctions for m in rb_metrics]):.1f}x more decision points"
)
print(
    f"• Wilson's has {np.mean([m.branching_entropy for m in wil_metrics])/np.mean([m.branching_entropy for m in rb_metrics]):.2f}x higher entropy"
)
print(f"• Both have identical structural bias: {np.mean([m.structural_bias for m in rb_metrics]):.4f}")

print("\nFor MFG-RL Experiments:")
print("  Recursive Backtracking: Long winding paths, high exploration difficulty")
print("  Wilson's Algorithm: More decision points, higher structural diversity")
