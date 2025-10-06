#!/usr/bin/env python3
"""
Recursive Division Maze Generation Demo

Demonstrates variable-width mazes with rooms, open spaces, and controllable parameters:
1. Basic Recursive Division
2. Different room sizes
3. Door width variations
4. Loop addition (braided mazes)
5. Comparison with perfect mazes

Author: MFG_PDE Team
Date: October 2025
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.reinforcement.environments import (
    MazeAlgorithm,
    PerfectMazeGenerator,
    RecursiveDivisionConfig,
    RecursiveDivisionGenerator,
    add_loops,
    create_room_based_config,
)


def demo_basic_recursive_division():
    """Demonstrate basic Recursive Division generation."""
    print("=" * 70)
    print("Demo 1: Basic Recursive Division")
    print("=" * 70)

    config = RecursiveDivisionConfig(
        rows=30,
        cols=40,
        min_room_width=5,
        min_room_height=5,
        door_width=2,
        seed=42,
    )

    generator = RecursiveDivisionGenerator(config)
    maze = generator.generate()

    open_cells = np.sum(maze == 0)
    total_cells = maze.size
    print(f"\nMaze dimensions: {maze.shape}")
    print(f"Open space: {open_cells}/{total_cells} ({100*open_cells/total_cells:.1f}%)")
    print("Parameters: min_room=5x5, door_width=2")


def demo_room_size_variations():
    """Demonstrate different room sizes."""
    print("\n" + "=" * 70)
    print("Demo 2: Room Size Variations")
    print("=" * 70)

    room_sizes = ["small", "medium", "large"]

    for room_size in room_sizes:
        config = create_room_based_config(30, 40, room_size=room_size, seed=42)
        generator = RecursiveDivisionGenerator(config)
        maze = generator.generate()

        open_pct = 100 * np.sum(maze == 0) / maze.size
        print(f"\n{room_size.capitalize()} rooms:")
        print(f"  Min room size: {config.min_room_width}x{config.min_room_height}")
        print(f"  Open space: {open_pct:.1f}%")


def demo_door_width_variations():
    """Demonstrate different door widths (corridor widths)."""
    print("\n" + "=" * 70)
    print("Demo 3: Door Width (Corridor Width) Variations")
    print("=" * 70)

    corridor_widths = ["narrow", "medium", "wide"]

    for corridor in corridor_widths:
        config = create_room_based_config(30, 40, corridor_width=corridor, seed=42)
        generator = RecursiveDivisionGenerator(config)
        maze = generator.generate()

        print(f"\n{corridor.capitalize()} corridors:")
        print(f"  Door width: {config.door_width}")
        print(f"  Open space: {100*np.sum(maze == 0)/maze.size:.1f}%")


def demo_loop_addition():
    """Demonstrate loop addition (braided mazes)."""
    print("\n" + "=" * 70)
    print("Demo 4: Loop Addition (Braided Mazes)")
    print("=" * 70)

    config = RecursiveDivisionConfig(rows=30, cols=40, seed=42)
    generator = RecursiveDivisionGenerator(config)
    original_maze = generator.generate()

    original_open = np.sum(original_maze == 0)

    print("\nOriginal maze:")
    print(f"  Open space: {100*original_open/original_maze.size:.1f}%")

    for density in [0.1, 0.2, 0.3]:
        braided = add_loops(original_maze, loop_density=density, seed=42)
        braided_open = np.sum(braided == 0)
        added_pct = 100 * (braided_open - original_open) / original_maze.size

        print(f"\nLoop density {density:.1f}:")
        print(f"  Open space: {100*braided_open/braided.size:.1f}% (+{added_pct:.1f}%)")
        print(f"  Added {braided_open - original_open} connections")


def visualize_comparison():
    """Compare different maze generation approaches."""
    print("\n" + "=" * 70)
    print("Visual Comparison: Perfect vs Recursive Division")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Perfect maze (unit-width)
    perfect_gen = PerfectMazeGenerator(20, 20, MazeAlgorithm.RECURSIVE_BACKTRACKING)
    perfect_maze = perfect_gen.to_numpy_array()

    axes[0, 0].imshow(perfect_maze, cmap="binary", interpolation="nearest")
    axes[0, 0].set_title("Perfect Maze\n(Unit-width corridors)", fontweight="bold")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    # Small rooms, narrow corridors
    config1 = create_room_based_config(20, 20, "small", "narrow", seed=42)
    gen1 = RecursiveDivisionGenerator(config1)
    maze1 = gen1.generate()

    axes[0, 1].imshow(maze1, cmap="binary", interpolation="nearest")
    axes[0, 1].set_title("Small Rooms\nNarrow Corridors", fontweight="bold")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    # Medium rooms, medium corridors
    config2 = create_room_based_config(20, 20, "medium", "medium", seed=42)
    gen2 = RecursiveDivisionGenerator(config2)
    maze2 = gen2.generate()

    axes[0, 2].imshow(maze2, cmap="binary", interpolation="nearest")
    axes[0, 2].set_title("Medium Rooms\nMedium Corridors", fontweight="bold")
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])

    # Large rooms, wide corridors
    config3 = create_room_based_config(20, 20, "large", "wide", seed=42)
    gen3 = RecursiveDivisionGenerator(config3)
    maze3 = gen3.generate()

    axes[1, 0].imshow(maze3, cmap="binary", interpolation="nearest")
    axes[1, 0].set_title("Large Rooms\nWide Corridors", fontweight="bold")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    # Braided maze (low density)
    braided_low = add_loops(maze2, loop_density=0.15, seed=42)
    axes[1, 1].imshow(braided_low, cmap="binary", interpolation="nearest")
    axes[1, 1].set_title("Braided Maze\nLoop Density 0.15", fontweight="bold")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    # Braided maze (high density)
    braided_high = add_loops(maze2, loop_density=0.3, seed=42)
    axes[1, 2].imshow(braided_high, cmap="binary", interpolation="nearest")
    axes[1, 2].set_title("Braided Maze\nLoop Density 0.30", fontweight="bold")
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])

    fig.suptitle(
        "Maze Generation Comparison: Variable-Width Corridors & Rooms",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("recursive_division_comparison.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved: recursive_division_comparison.png")
    plt.show()


def demo_mfg_applications():
    """Demonstrate MFG-relevant scenarios."""
    print("\n" + "=" * 70)
    print("Demo 5: MFG Applications")
    print("=" * 70)

    scenarios = [
        ("Building Evacuation", "medium", "medium", 0.0),
        ("Concert Venue", "large", "wide", 0.1),
        ("Traffic Intersections", "small", "wide", 0.2),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (name, room_size, corridor, loop_density) in enumerate(scenarios):
        config = create_room_based_config(30, 40, room_size=room_size, corridor_width=corridor, seed=42)
        generator = RecursiveDivisionGenerator(config)
        maze = generator.generate()

        if loop_density > 0:
            maze = add_loops(maze, loop_density=loop_density, seed=42)

        axes[idx].imshow(maze, cmap="binary", interpolation="nearest")
        axes[idx].set_title(f"{name}\n({room_size} rooms, {corridor} corridors)", fontweight="bold")
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

        info_text = f"Open: {100*np.sum(maze==0)/maze.size:.1f}%\nLoops: {loop_density}"
        axes[idx].text(
            0.02,
            0.98,
            info_text,
            transform=axes[idx].transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            fontsize=9,
        )

    fig.suptitle("MFG Application Scenarios", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("mfg_application_scenarios.png", dpi=150, bbox_inches="tight")
    print("\nMFG scenarios saved: mfg_application_scenarios.png")
    plt.show()


def main():
    """Run all demonstrations."""
    demo_basic_recursive_division()
    demo_room_size_variations()
    demo_door_width_variations()
    demo_loop_addition()

    print("\n" + "=" * 70)
    print("Generating Visualizations...")
    print("=" * 70)

    visualize_comparison()
    demo_mfg_applications()

    print("\n" + "=" * 70)
    print("All Demonstrations Complete")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Variable-width corridors (1-3 cells)")
    print("  ✓ Rooms of varying sizes (small/medium/large)")
    print("  ✓ Controllable door widths")
    print("  ✓ Loop addition for route diversity")
    print("  ✓ MFG-relevant scenarios")
    print("\nGenerated Images:")
    print("  - recursive_division_comparison.png")
    print("  - mfg_application_scenarios.png")


if __name__ == "__main__":
    main()
