#!/usr/bin/env python3
"""
Comprehensive Maze Configuration Examples

Demonstrates all controllable parameters in the enhanced maze generator:
1. Grid dimensions (discrete cells)
2. Physical dimensions (continuous space)
3. Entry/exit positions (multiple strategies)
4. Multi-goal configurations
5. Position placement strategies

Author: MFG_PDE Team
Date: October 2025
"""

import matplotlib.pyplot as plt

from mfg_pde.alg.reinforcement.environments import (
    MazeAlgorithm,
    MazeConfig,
    PerfectMazeGenerator,
    PlacementStrategy,
    compute_position_metrics,
    create_continuous_maze_config,
    create_multi_goal_config,
    place_positions,
)


def example_1_basic_parameters():
    """Example 1: Basic controllable parameters."""
    print("=" * 70)
    print("Example 1: Basic Controllable Parameters")
    print("=" * 70)

    config = MazeConfig(
        rows=20,  # Grid height (cells)
        cols=30,  # Grid width (cells)
        algorithm="recursive_backtracking",  # or "wilsons"
        seed=42,  # Reproducibility
        wall_thickness=1,  # Pixel rendering thickness
    )

    print(f"\nGrid dimensions: {config.rows} x {config.cols} cells")
    print(f"Algorithm: {config.algorithm}")
    print(f"Seed: {config.seed}")

    pixel_dims = config.get_pixel_dimensions()
    print(f"Pixel dimensions: {pixel_dims}")


def example_2_continuous_dimensions():
    """Example 2: Physical (continuous) dimensions."""
    print("\n" + "=" * 70)
    print("Example 2: Physical (Continuous) Dimensions")
    print("=" * 70)

    config = create_continuous_maze_config(
        physical_width=10.0,  # 10 meters
        physical_height=15.0,  # 15 meters
        cell_density=20,  # 20 cells per meter
    )

    print(f"\nPhysical dimensions: {config.physical_dims.width}m x {config.physical_dims.height}m")
    print(f"Grid dimensions: {config.rows} x {config.cols} cells")
    print(f"Cell size: {config.physical_dims.cell_size}m")

    cell_pos = (100, 150)
    continuous_pos = config.cell_to_continuous(*cell_pos)
    print(f"\nCell {cell_pos} -> Continuous {continuous_pos}")

    continuous_coord = (5.0, 7.5)
    cell_coord = config.continuous_to_cell(*continuous_coord)
    print(f"Continuous {continuous_coord} -> Cell {cell_coord}")


def example_3_position_strategies():
    """Example 3: Position placement strategies."""
    print("\n" + "=" * 70)
    print("Example 3: Position Placement Strategies")
    print("=" * 70)

    generator = PerfectMazeGenerator(20, 20, MazeAlgorithm.RECURSIVE_BACKTRACKING)
    grid = generator.generate(seed=42)

    strategies = [
        PlacementStrategy.RANDOM,
        PlacementStrategy.CORNERS,
        PlacementStrategy.EDGES,
        PlacementStrategy.FARTHEST,
        PlacementStrategy.CLUSTERED,
    ]

    for strategy in strategies:
        positions = place_positions(grid, num_positions=4, strategy=strategy, seed=42)
        metrics = compute_position_metrics(grid, positions)

        print(f"\n{strategy.value}:")
        print(f"  Positions: {positions}")
        print(f"  Min distance: {metrics['min_distance']}")
        print(f"  Max distance: {metrics['max_distance']}")
        print(f"  Avg distance: {metrics['avg_distance']:.1f}")


def example_4_multi_goal_config():
    """Example 4: Multiple entry/exit points."""
    print("\n" + "=" * 70)
    print("Example 4: Multiple Entry/Exit Points")
    print("=" * 70)

    config = create_multi_goal_config(
        rows=30,
        cols=30,
        num_goals=5,  # 5 exit points
        goal_strategy="farthest",  # Maximize distance between goals
    )

    generator = PerfectMazeGenerator(config.rows, config.cols, MazeAlgorithm(config.algorithm))
    grid = generator.generate(seed=42)

    goal_positions = place_positions(grid, config.num_goals, config.placement_strategy, seed=42)

    print(f"\nNumber of goals: {config.num_goals}")
    print(f"Goal placement strategy: {config.placement_strategy.value}")
    print(f"Goal positions: {goal_positions}")

    metrics = compute_position_metrics(grid, goal_positions)
    print("\nGoal separation metrics:")
    print(f"  Min distance: {metrics['min_distance']}")
    print(f"  Max distance: {metrics['max_distance']}")
    print(f"  Avg distance: {metrics['avg_distance']:.1f}")


def example_5_custom_positions():
    """Example 5: Custom user-specified positions."""
    print("\n" + "=" * 70)
    print("Example 5: Custom User-Specified Positions")
    print("=" * 70)

    generator = PerfectMazeGenerator(25, 25, MazeAlgorithm.WILSONS)
    grid = generator.generate(seed=42)

    custom_starts = [(2, 2), (2, 22), (22, 2)]
    custom_goals = [(22, 22), (12, 12)]

    start_positions = place_positions(
        grid,
        num_positions=len(custom_starts),
        strategy=PlacementStrategy.CUSTOM,
        custom_positions=custom_starts,
    )

    goal_positions = place_positions(
        grid,
        num_positions=len(custom_goals),
        strategy=PlacementStrategy.CUSTOM,
        custom_positions=custom_goals,
    )

    print(f"\nStart positions: {start_positions}")
    print(f"Goal positions: {goal_positions}")

    start_metrics = compute_position_metrics(grid, start_positions)
    goal_metrics = compute_position_metrics(grid, goal_positions)

    print(f"\nStart separation: min={start_metrics['min_distance']}, avg={start_metrics['avg_distance']:.1f}")
    print(f"Goal separation: min={goal_metrics['min_distance']}, avg={goal_metrics['avg_distance']:.1f}")


def example_6_comprehensive_config():
    """Example 6: All parameters combined."""
    print("\n" + "=" * 70)
    print("Example 6: Comprehensive Configuration")
    print("=" * 70)

    from mfg_pde.alg.reinforcement.environments import PhysicalDimensions

    config = MazeConfig(
        rows=40,
        cols=60,
        algorithm="wilsons",
        seed=123,
        physical_dims=PhysicalDimensions(width=30.0, height=20.0, cell_size=0.5),
        num_starts=2,
        num_goals=4,
        placement_strategy=PlacementStrategy.FARTHEST,
        wall_thickness=1,
        verify_perfect=True,
    )

    print("\nGrid Configuration:")
    print(f"  Dimensions: {config.rows} x {config.cols} cells")
    print(f"  Algorithm: {config.algorithm}")
    print(f"  Seed: {config.seed}")

    print("\nPhysical Properties:")
    print(f"  Physical size: {config.physical_dims.width}m x {config.physical_dims.height}m")
    print(f"  Cell size: {config.physical_dims.cell_size}m")

    print("\nEntry/Exit Configuration:")
    print(f"  Number of starts: {config.num_starts}")
    print(f"  Number of goals: {config.num_goals}")
    print(f"  Placement strategy: {config.placement_strategy.value}")

    generator = PerfectMazeGenerator(config.rows, config.cols, MazeAlgorithm(config.algorithm))
    grid = generator.generate(seed=config.seed)

    starts = place_positions(grid, config.num_starts, config.placement_strategy, seed=42)
    goals = place_positions(grid, config.num_goals, config.placement_strategy, seed=43)

    print("\nGenerated Positions:")
    print(f"  Starts: {starts}")
    print(f"  Goals: {goals}")


def visualize_position_strategies():
    """Visualize different position placement strategies."""
    print("\n" + "=" * 70)
    print("Visualizing Position Placement Strategies")
    print("=" * 70)

    generator = PerfectMazeGenerator(20, 20, MazeAlgorithm.RECURSIVE_BACKTRACKING)
    grid = generator.generate(seed=42)
    maze_array = generator.to_numpy_array()

    strategies = [
        PlacementStrategy.CORNERS,
        PlacementStrategy.EDGES,
        PlacementStrategy.FARTHEST,
        PlacementStrategy.CLUSTERED,
    ]

    _fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()

    for idx, strategy in enumerate(strategies):
        positions = place_positions(grid, num_positions=5, strategy=strategy, seed=42)

        axes[idx].imshow(maze_array, cmap="binary", interpolation="nearest")
        axes[idx].set_title(f"{strategy.value.replace('_', ' ').title()}", fontweight="bold")

        for r, c in positions:
            pixel_r = r * 3 + 1
            pixel_c = c * 3 + 1
            axes[idx].plot(pixel_c, pixel_r, "ro", markersize=8)

        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

        metrics = compute_position_metrics(grid, positions)
        info_text = f"Min dist: {metrics['min_distance']}\nAvg dist: {metrics['avg_distance']:.1f}"
        axes[idx].text(
            0.02,
            0.98,
            info_text,
            transform=axes[idx].transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    plt.suptitle("Position Placement Strategies (Red = Positions)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("position_strategies_comparison.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved: position_strategies_comparison.png")
    plt.show()


def main():
    """Run all examples."""
    example_1_basic_parameters()
    example_2_continuous_dimensions()
    example_3_position_strategies()
    example_4_multi_goal_config()
    example_5_custom_positions()
    example_6_comprehensive_config()

    print("\n" + "=" * 70)
    print("All Examples Complete")
    print("=" * 70)

    visualize_position_strategies()


if __name__ == "__main__":
    main()
