#!/usr/bin/env python3
"""
Rigorous Assessment: Recursive Backtracking vs Wilson's Algorithm for MFG-RL

Analyzes structural properties, computational cost, and RL suitability
to determine the optimal maze generation algorithm for MFG experiments.

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

from perfect_maze_generator import (
    Grid,
    MazeAlgorithm,
    PerfectMazeGenerator,
)

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


@dataclass
class MazeMetrics:
    """Comprehensive maze quality metrics."""

    algorithm: str
    generation_time: float

    # Structural complexity
    num_dead_ends: int
    num_junctions: int
    num_corridors: int

    # Path metrics
    longest_path_length: int
    avg_path_length: float
    path_length_variance: float

    # RL-specific metrics
    decision_points: int
    exploration_difficulty: float
    bottleneck_count: int

    # Statistical measures
    branching_entropy: float
    structural_bias: float


class MazeAnalyzer:
    """Analyzes maze structural properties for RL suitability."""

    def __init__(self, grid: Grid):
        self.grid = grid

    def count_dead_ends(self) -> int:
        """Count cells with only one passage (dead ends)."""
        count = 0
        for cell in self.grid.all_cells():
            passages = sum([cell.north, cell.south, cell.east, cell.west])
            if passages == 1:
                count += 1
        return count

    def count_junctions(self) -> int:
        """Count cells with 3+ passages (decision points)."""
        count = 0
        for cell in self.grid.all_cells():
            passages = sum([cell.north, cell.south, cell.east, cell.west])
            if passages >= 3:
                count += 1
        return count

    def count_corridors(self) -> int:
        """Count cells with exactly 2 passages (corridors)."""
        count = 0
        for cell in self.grid.all_cells():
            passages = sum([cell.north, cell.south, cell.east, cell.west])
            if passages == 2:
                count += 1
        return count

    def find_longest_path(self) -> tuple[int, list]:
        """Find longest path in maze using BFS from all cells."""
        max_length = 0
        longest_path = []

        # Try from each cell
        for start_cell in self.grid.all_cells():
            distances, paths = self._bfs_distances(start_cell)

            for cell, distance in distances.items():
                if distance > max_length:
                    max_length = distance
                    longest_path = paths[cell]

        return max_length, longest_path

    def compute_average_path_lengths(self, sample_size: int = 100) -> tuple[float, float]:
        """Compute average path length and variance via sampling."""
        path_lengths = []

        cells = self.grid.all_cells()
        np.random.seed(42)

        for _ in range(sample_size):
            start = np.random.choice(cells)
            end = np.random.choice(cells)

            if start != end:
                path = self._find_path(start, end)
                if path:
                    path_lengths.append(len(path) - 1)

        if not path_lengths:
            return 0.0, 0.0

        return np.mean(path_lengths), np.var(path_lengths)

    def compute_branching_entropy(self) -> float:
        """
        Compute entropy of branching decisions.
        Higher entropy = more unpredictable structure = better for RL.
        """
        passage_counts = defaultdict(int)

        for cell in self.grid.all_cells():
            num_passages = sum([cell.north, cell.south, cell.east, cell.west])
            passage_counts[num_passages] += 1

        total = len(self.grid.all_cells())
        entropy = 0.0

        for count in passage_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)

        return entropy

    def compute_structural_bias(self) -> float:
        """
        Measure directional bias in maze structure.
        0 = perfectly unbiased, 1 = maximally biased.
        """
        # Count passages in each direction
        north_count = sum(1 for cell in self.grid.all_cells() if cell.north)
        south_count = sum(1 for cell in self.grid.all_cells() if cell.south)
        east_count = sum(1 for cell in self.grid.all_cells() if cell.east)
        west_count = sum(1 for cell in self.grid.all_cells() if cell.west)

        total_passages = north_count + south_count + east_count + west_count

        if total_passages == 0:
            return 0.0

        # Expected count for unbiased maze
        expected = total_passages / 4.0

        # Compute chi-square statistic as bias measure
        chi_square = sum(
            (count - expected) ** 2 / expected for count in [north_count, south_count, east_count, west_count]
        )

        # Normalize to [0, 1]
        max_chi_square = 3 * expected  # Maximum possible deviation
        bias = min(chi_square / max_chi_square, 1.0) if max_chi_square > 0 else 0.0

        return bias

    def _bfs_distances(self, start_cell) -> tuple[dict, dict]:
        """BFS to compute distances from start cell."""
        distances = {start_cell: 0}
        paths = {start_cell: [start_cell]}
        queue = [start_cell]

        while queue:
            current = queue.pop(0)
            current_dist = distances[current]

            for neighbor in self._get_linked_neighbors(current):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    paths[neighbor] = [*paths[current], neighbor]
                    queue.append(neighbor)

        return distances, paths

    def _find_path(self, start, end):
        """Find path between two cells using BFS."""
        if start == end:
            return [start]

        queue = [(start, [start])]
        visited = {start}

        while queue:
            current, path = queue.pop(0)

            for neighbor in self._get_linked_neighbors(current):
                if neighbor == end:
                    return [*path, neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, [*path, neighbor]))

        return []

    def _get_linked_neighbors(self, cell):
        """Get neighbors that have passages to this cell."""
        neighbors = []

        if cell.north:
            neighbor = self.grid.get_cell(cell.row - 1, cell.col)
            if neighbor:
                neighbors.append(neighbor)

        if cell.south:
            neighbor = self.grid.get_cell(cell.row + 1, cell.col)
            if neighbor:
                neighbors.append(neighbor)

        if cell.east:
            neighbor = self.grid.get_cell(cell.row, cell.col + 1)
            if neighbor:
                neighbors.append(neighbor)

        if cell.west:
            neighbor = self.grid.get_cell(cell.row, cell.col - 1)
            if neighbor:
                neighbors.append(neighbor)

        return neighbors


def analyze_maze(algorithm: MazeAlgorithm, rows: int, cols: int, seed: int = 42) -> MazeMetrics:
    """Perform complete analysis of a maze algorithm."""
    print(f"Analyzing {algorithm.value}...")

    # Generate maze and time it
    generator = PerfectMazeGenerator(rows, cols, algorithm)

    start_time = time.time()
    grid = generator.generate(seed=seed)
    generation_time = time.time() - start_time

    # Create analyzer
    analyzer = MazeAnalyzer(grid)

    # Structural analysis
    dead_ends = analyzer.count_dead_ends()
    junctions = analyzer.count_junctions()
    corridors = analyzer.count_corridors()

    # Path analysis
    longest_path_length, _ = analyzer.find_longest_path()
    avg_path_length, path_variance = analyzer.compute_average_path_lengths(sample_size=200)

    # RL-specific metrics
    decision_points = junctions
    exploration_difficulty = avg_path_length / (rows + cols)  # Normalized by grid size
    bottleneck_count = corridors  # Corridors can be bottlenecks

    # Statistical measures
    branching_entropy = analyzer.compute_branching_entropy()
    structural_bias = analyzer.compute_structural_bias()

    return MazeMetrics(
        algorithm=algorithm.value,
        generation_time=generation_time,
        num_dead_ends=dead_ends,
        num_junctions=junctions,
        num_corridors=corridors,
        longest_path_length=longest_path_length,
        avg_path_length=avg_path_length,
        path_length_variance=path_variance,
        decision_points=decision_points,
        exploration_difficulty=exploration_difficulty,
        bottleneck_count=bottleneck_count,
        branching_entropy=branching_entropy,
        structural_bias=structural_bias,
    )


def comparative_assessment(rows: int = 20, cols: int = 20, num_trials: int = 10):
    """Run comprehensive comparative assessment."""
    print("=" * 80)
    print("MAZE ALGORITHM ASSESSMENT: Recursive Backtracking vs Wilson's")
    print("=" * 80)
    print(f"Grid size: {rows}x{cols}")
    print(f"Trials per algorithm: {num_trials}")
    print()

    algorithms = [MazeAlgorithm.RECURSIVE_BACKTRACKING, MazeAlgorithm.WILSONS]

    all_metrics = {alg: [] for alg in algorithms}

    # Run trials
    for algorithm in algorithms:
        print(f"\nRunning {num_trials} trials for {algorithm.value}...")

        for trial in range(num_trials):
            metrics = analyze_maze(algorithm, rows, cols, seed=trial)
            all_metrics[algorithm].append(metrics)

    # Aggregate statistics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    comparison_data = {}

    for algorithm in algorithms:
        metrics_list = all_metrics[algorithm]

        print(f"\n{algorithm.value.upper()}")
        print("-" * 80)

        # Compute statistics
        stats_dict = {
            "generation_time": [m.generation_time for m in metrics_list],
            "dead_ends": [m.num_dead_ends for m in metrics_list],
            "junctions": [m.num_junctions for m in metrics_list],
            "corridors": [m.num_corridors for m in metrics_list],
            "longest_path": [m.longest_path_length for m in metrics_list],
            "avg_path_length": [m.avg_path_length for m in metrics_list],
            "path_variance": [m.path_length_variance for m in metrics_list],
            "decision_points": [m.decision_points for m in metrics_list],
            "exploration_difficulty": [m.exploration_difficulty for m in metrics_list],
            "branching_entropy": [m.branching_entropy for m in metrics_list],
            "structural_bias": [m.structural_bias for m in metrics_list],
        }

        comparison_data[algorithm] = stats_dict

        # Print statistics
        for metric_name, values in stats_dict.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {metric_name:25s}: {mean_val:8.3f} ± {std_val:.3f}")

    # Statistical comparison
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON (t-tests)")
    print("=" * 80)

    rb_data = comparison_data[MazeAlgorithm.RECURSIVE_BACKTRACKING]
    wil_data = comparison_data[MazeAlgorithm.WILSONS]

    print(f"\n{'Metric':<25s} {'RB Mean':>10s} {'Wilson Mean':>12s} {'p-value':>10s} {'Significant?':>12s}")
    print("-" * 80)

    for metric_name in rb_data:
        rb_values = rb_data[metric_name]
        wil_values = wil_data[metric_name]

        _t_stat, p_value = stats.ttest_ind(rb_values, wil_values)
        significant = "YES" if p_value < 0.05 else "no"

        print(
            f"{metric_name:<25s} {np.mean(rb_values):10.3f} {np.mean(wil_values):12.3f} "
            f"{p_value:10.4f} {significant:>12s}"
        )

    # Visualization
    visualize_comparison(comparison_data, rows, cols)

    return comparison_data


def visualize_comparison(comparison_data: dict, rows: int, cols: int):
    """Create comprehensive visualization of comparison."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    rb_data = comparison_data[MazeAlgorithm.RECURSIVE_BACKTRACKING]
    wil_data = comparison_data[MazeAlgorithm.WILSONS]

    metrics = [
        ("generation_time", "Generation Time (s)", False),
        ("dead_ends", "Dead Ends", False),
        ("junctions", "Junctions", False),
        ("longest_path", "Longest Path", False),
        ("avg_path_length", "Avg Path Length", False),
        ("branching_entropy", "Branching Entropy", True),
        ("structural_bias", "Structural Bias", False),
        ("exploration_difficulty", "Exploration Difficulty", True),
    ]

    for idx, (metric_key, metric_label, higher_is_better) in enumerate(metrics):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])

        rb_values = rb_data[metric_key]
        wil_values = wil_data[metric_key]

        positions = [1, 2]
        data = [rb_values, wil_values]

        bp = ax.boxplot(
            data, positions=positions, widths=0.6, patch_artist=True, labels=["Recursive\nBacktracking", "Wilson's"]
        )

        # Color by performance
        if higher_is_better:
            better_idx = 0 if np.mean(rb_values) > np.mean(wil_values) else 1
        else:
            better_idx = 0 if np.mean(rb_values) < np.mean(wil_values) else 1

        colors = ["lightgreen" if i == better_idx else "lightcoral" for i in range(2)]
        for patch, color in zip(bp["boxes"], colors, strict=False):
            patch.set_facecolor(color)

        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title(metric_label, fontweight="bold")

    # Add recommendation panel
    ax_rec = fig.add_subplot(gs[2, 2])
    ax_rec.axis("off")

    # Compute scores
    rb_score = 0
    wil_score = 0

    for metric_key, _, higher_is_better in metrics:
        rb_mean = np.mean(rb_data[metric_key])
        wil_mean = np.mean(wil_data[metric_key])

        if higher_is_better:
            if rb_mean > wil_mean:
                rb_score += 1
            else:
                wil_score += 1
        else:
            if rb_mean < wil_mean:
                rb_score += 1
            else:
                wil_score += 1

    # Compute key insights
    faster_alg = "RB" if np.mean(rb_data["generation_time"]) < np.mean(wil_data["generation_time"]) else "Wilson"
    more_unbiased = "RB" if np.mean(rb_data["structural_bias"]) < np.mean(wil_data["structural_bias"]) else "Wilson"
    more_challenging = (
        "RB" if np.mean(rb_data["exploration_difficulty"]) > np.mean(wil_data["exploration_difficulty"]) else "Wilson"
    )

    if rb_score > wil_score:
        final_rec = "Recursive Backtracking (DFS)"
    elif wil_score > rb_score:
        final_rec = "Wilson's Algorithm"
    else:
        final_rec = "BOTH (tied)"

    recommendation = f"""
ASSESSMENT RECOMMENDATION

Metric Wins:
  Recursive Backtracking: {rb_score}
  Wilson's Algorithm: {wil_score}

Key Insights:
• Generation Speed: {faster_alg} faster
• Structural Bias: {more_unbiased} more unbiased
• RL Difficulty: {more_challenging} more challenging

RECOMMENDATION:
{final_rec}
    """

    ax_rec.text(
        0.05,
        0.95,
        recommendation,
        transform=ax_rec.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
    )

    fig.suptitle(f"Maze Algorithm Comparison ({rows}x{cols} grid)", fontsize=16, fontweight="bold")

    plt.savefig("maze_algorithm_assessment.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved: maze_algorithm_assessment.png")
    plt.show()


if __name__ == "__main__":
    comparative_assessment(rows=20, cols=20, num_trials=10)
