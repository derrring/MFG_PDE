#!/usr/bin/env python3
"""Quick visualization of perfect maze algorithms."""

from perfect_maze_generator import (
    MazeAlgorithm,
    PerfectMazeGenerator,
    verify_perfect_maze,
)

import matplotlib.pyplot as plt

# Generate and compare algorithms
algorithms = [
    MazeAlgorithm.RECURSIVE_BACKTRACKING,
    MazeAlgorithm.BINARY_TREE,
    MazeAlgorithm.SIDEWINDER,
    MazeAlgorithm.WILSONS,
]

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.flatten()

seed = 42
rows, cols = 20, 20

for i, algorithm in enumerate(algorithms):
    print(f"Generating {algorithm.value}...")

    generator = PerfectMazeGenerator(rows, cols, algorithm)
    grid = generator.generate(seed=seed)
    maze_array = generator.to_numpy_array()

    # Verify
    verification = verify_perfect_maze(grid)

    # Visualize
    axes[i].imshow(maze_array, cmap="binary", interpolation="nearest")

    title = f"{algorithm.value.replace('_', ' ').title()}"
    if verification["is_perfect"]:
        title += "\nPERFECT"
    axes[i].set_title(title, fontsize=12, fontweight="bold")
    axes[i].set_xticks([])
    axes[i].set_yticks([])

    # Add verification info
    info_text = (
        f"Connected: {verification['is_connected']}\n"
        f"Loop-free: {verification['is_no_loops']}\n"
        f"Passages: {verification['passage_count']}/{verification['expected_passages']}"
    )
    axes[i].text(
        0.02,
        0.98,
        info_text,
        transform=axes[i].transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

fig.suptitle(
    "Perfect Maze Algorithms Comparison\nAll Guaranteed: Fully Connected + No Loops", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("perfect_maze_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved to: perfect_maze_comparison.png")
plt.show()
