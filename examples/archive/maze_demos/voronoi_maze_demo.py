#!/usr/bin/env python3
"""
Voronoi Diagram Maze Generation Demo

Demonstrates organic, room-based maze generation using Voronoi diagrams
and Delaunay triangulation for realistic building layouts.

Features:
1. Basic Voronoi maze generation
2. Lloyd's relaxation for uniform distribution
3. Varying room densities (number of points)
4. Comparison with other maze algorithms
5. MFG application scenarios

Author: MFG_PDE Team
Date: October 2025
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.reinforcement.environments import (
    CellularAutomataGenerator,
    RecursiveDivisionGenerator,
    VoronoiMazeConfig,
    VoronoiMazeGenerator,
)


def demo_basic_voronoi():
    """Demonstrate basic Voronoi maze generation."""
    print("=" * 70)
    print("Demo 1: Basic Voronoi Maze Generation")
    print("=" * 70)

    config = VoronoiMazeConfig(rows=50, cols=50, num_points=15, seed=42)

    generator = VoronoiMazeGenerator(config)
    maze = generator.generate()

    open_cells = np.sum(maze == 0)
    total_cells = maze.size
    print(f"\nMaze dimensions: {maze.shape}")
    print(f"Number of rooms: {config.num_points}")
    print(f"Open space: {open_cells}/{total_cells} ({100*open_cells/total_cells:.1f}%)")
    print(f"Spanning tree edges: {len(generator.spanning_edges)} (should be {config.num_points - 1})")


def demo_lloyds_relaxation():
    """Demonstrate Lloyd's relaxation effect."""
    print("\n" + "=" * 70)
    print("Demo 2: Lloyd's Relaxation Effect")
    print("=" * 70)

    relaxation_levels = [0, 2, 5]

    for iterations in relaxation_levels:
        config = VoronoiMazeConfig(rows=40, cols=40, num_points=12, relaxation_iterations=iterations, seed=42)

        generator = VoronoiMazeGenerator(config)
        maze = generator.generate()

        open_pct = 100 * np.sum(maze == 0) / maze.size
        print(f"\nRelaxation iterations: {iterations}")
        print(f"  Open space: {open_pct:.1f}%")
        print(f"  Effect: {'Random distribution' if iterations == 0 else f'More uniform (smoothed {iterations}x)'}")


def demo_room_density_variations():
    """Demonstrate different room densities."""
    print("\n" + "=" * 70)
    print("Demo 3: Room Density Variations")
    print("=" * 70)

    densities = [8, 15, 25]

    for num_points in densities:
        config = VoronoiMazeConfig(rows=50, cols=50, num_points=num_points, seed=42)
        generator = VoronoiMazeGenerator(config)
        maze = generator.generate()

        open_pct = 100 * np.sum(maze == 0) / maze.size
        avg_room_area = (maze.size * open_pct / 100) / num_points

        print(f"\nNumber of rooms: {num_points}")
        print(f"  Open space: {open_pct:.1f}%")
        print(f"  Average room size: {avg_room_area:.0f} cells")


def visualize_voronoi_features():
    """Visualize Voronoi maze features and components."""
    print("\n" + "=" * 70)
    print("Visual Demonstration: Voronoi Maze Components")
    print("=" * 70)

    config = VoronoiMazeConfig(rows=50, cols=50, num_points=15, relaxation_iterations=2, seed=42)
    generator = VoronoiMazeGenerator(config)
    maze = generator.generate()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Basic Voronoi maze
    axes[0, 0].imshow(maze, cmap="binary", interpolation="nearest")
    axes[0, 0].set_title(f"Voronoi Maze\n{config.num_points} rooms", fontweight="bold")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    # Add seed points overlay
    if generator.points is not None:
        axes[0, 0].scatter(generator.points[:, 0], generator.points[:, 1], c="red", s=30, marker="o", alpha=0.7)

    # 2. No relaxation
    config_no_relax = VoronoiMazeConfig(rows=50, cols=50, num_points=15, relaxation_iterations=0, seed=42)
    gen_no_relax = VoronoiMazeGenerator(config_no_relax)
    maze_no_relax = gen_no_relax.generate()

    axes[0, 1].imshow(maze_no_relax, cmap="binary", interpolation="nearest")
    axes[0, 1].set_title("No Relaxation\nRandom Distribution", fontweight="bold")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    # 3. High relaxation
    config_relax = VoronoiMazeConfig(rows=50, cols=50, num_points=15, relaxation_iterations=5, seed=42)
    gen_relax = VoronoiMazeGenerator(config_relax)
    maze_relax = gen_relax.generate()

    axes[0, 2].imshow(maze_relax, cmap="binary", interpolation="nearest")
    axes[0, 2].set_title("Lloyd's Relaxation (5 iter)\nUniform Distribution", fontweight="bold")
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])

    # 4. Few rooms (sparse)
    config_sparse = VoronoiMazeConfig(rows=50, cols=50, num_points=8, seed=42)
    gen_sparse = VoronoiMazeGenerator(config_sparse)
    maze_sparse = gen_sparse.generate()

    axes[1, 0].imshow(maze_sparse, cmap="binary", interpolation="nearest")
    axes[1, 0].set_title("Sparse Layout\n8 Large Rooms", fontweight="bold")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    # 5. Many rooms (dense)
    config_dense = VoronoiMazeConfig(rows=50, cols=50, num_points=30, seed=42)
    gen_dense = VoronoiMazeGenerator(config_dense)
    maze_dense = gen_dense.generate()

    axes[1, 1].imshow(maze_dense, cmap="binary", interpolation="nearest")
    axes[1, 1].set_title("Dense Layout\n30 Small Rooms", fontweight="bold")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    # 6. Thick walls
    config_thick = VoronoiMazeConfig(rows=50, cols=50, num_points=15, wall_thickness=2, seed=42)
    gen_thick = VoronoiMazeGenerator(config_thick)
    maze_thick = gen_thick.generate()

    axes[1, 2].imshow(maze_thick, cmap="binary", interpolation="nearest")
    axes[1, 2].set_title("Thick Walls\nThickness = 2", fontweight="bold")
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])

    fig.suptitle("Voronoi Maze Generation: Feature Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("voronoi_maze_features.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved: voronoi_maze_features.png")
    plt.show()


def compare_maze_algorithms():
    """Compare Voronoi with other maze generation algorithms."""
    print("\n" + "=" * 70)
    print("Algorithm Comparison: Voronoi vs Recursive Division vs Cellular Automata")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Voronoi
    voronoi_config = VoronoiMazeConfig(rows=50, cols=50, num_points=15, seed=42)
    voronoi_gen = VoronoiMazeGenerator(voronoi_config)
    voronoi_maze = voronoi_gen.generate()

    axes[0].imshow(voronoi_maze, cmap="binary", interpolation="nearest")
    axes[0].set_title("Voronoi Diagram\nOrganic Rooms", fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    voronoi_open = 100 * np.sum(voronoi_maze == 0) / voronoi_maze.size
    axes[0].text(
        0.02,
        0.98,
        f"Open: {voronoi_open:.1f}%\nRooms: {voronoi_config.num_points}",
        transform=axes[0].transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        fontsize=9,
    )

    # Recursive Division
    from mfg_pde.alg.reinforcement.environments import RecursiveDivisionConfig

    rd_config = RecursiveDivisionConfig(rows=50, cols=50, min_room_width=5, seed=42)
    rd_gen = RecursiveDivisionGenerator(rd_config)
    rd_maze = rd_gen.generate()

    axes[1].imshow(rd_maze, cmap="binary", interpolation="nearest")
    axes[1].set_title("Recursive Division\nStructured Rooms", fontweight="bold")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    rd_open = 100 * np.sum(rd_maze == 0) / rd_maze.size
    axes[1].text(
        0.02,
        0.98,
        f"Open: {rd_open:.1f}%\nStructured",
        transform=axes[1].transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        fontsize=9,
    )

    # Cellular Automata
    from mfg_pde.alg.reinforcement.environments import CellularAutomataConfig

    ca_config = CellularAutomataConfig(rows=50, cols=50, initial_wall_prob=0.45, num_iterations=5, seed=42)
    ca_gen = CellularAutomataGenerator(ca_config)
    ca_maze = ca_gen.generate()

    axes[2].imshow(ca_maze, cmap="binary", interpolation="nearest")
    axes[2].set_title("Cellular Automata\nCave-like", fontweight="bold")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    ca_open = 100 * np.sum(ca_maze == 0) / ca_maze.size
    axes[2].text(
        0.02,
        0.98,
        f"Open: {ca_open:.1f}%\nOrganic",
        transform=axes[2].transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        fontsize=9,
    )

    fig.suptitle("Maze Generation Algorithm Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("algorithm_comparison.png", dpi=150, bbox_inches="tight")
    print("\nComparison saved: algorithm_comparison.png")
    plt.show()


def demo_mfg_applications():
    """Demonstrate MFG-relevant application scenarios."""
    print("\n" + "=" * 70)
    print("Demo: MFG Application Scenarios")
    print("=" * 70)

    scenarios = [
        ("Museum Layout", 12, 0, 2),  # (name, num_rooms, relaxation, wall_thickness)
        ("Conference Center", 20, 3, 1),
        ("Shopping Mall", 25, 2, 2),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (name, num_points, relaxation, wall_thickness) in enumerate(scenarios):
        config = VoronoiMazeConfig(
            rows=60,
            cols=60,
            num_points=num_points,
            relaxation_iterations=relaxation,
            wall_thickness=wall_thickness,
            seed=42,
        )

        generator = VoronoiMazeGenerator(config)
        maze = generator.generate()

        axes[idx].imshow(maze, cmap="binary", interpolation="nearest")
        axes[idx].set_title(f"{name}\n{num_points} Rooms", fontweight="bold")
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

        open_pct = 100 * np.sum(maze == 0) / maze.size
        info_text = f"Open: {open_pct:.1f}%\nRelax: {relaxation}\nWalls: {wall_thickness}"
        axes[idx].text(
            0.02,
            0.98,
            info_text,
            transform=axes[idx].transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            fontsize=9,
        )

    fig.suptitle("MFG Application Scenarios: Realistic Building Layouts", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("mfg_application_scenarios.png", dpi=150, bbox_inches="tight")
    print("\nMFG scenarios saved: mfg_application_scenarios.png")
    plt.show()


def main():
    """Run all demonstrations."""
    demo_basic_voronoi()
    demo_lloyds_relaxation()
    demo_room_density_variations()

    print("\n" + "=" * 70)
    print("Generating Visualizations...")
    print("=" * 70)

    visualize_voronoi_features()
    compare_maze_algorithms()
    demo_mfg_applications()

    print("\n" + "=" * 70)
    print("All Demonstrations Complete")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Voronoi diagram-based room generation")
    print("  ✓ Delaunay triangulation for connectivity")
    print("  ✓ Kruskal's spanning tree algorithm")
    print("  ✓ Lloyd's relaxation for uniform distribution")
    print("  ✓ Variable room densities and sizes")
    print("  ✓ Comparison with other algorithms")
    print("  ✓ MFG application scenarios")
    print("\nGenerated Images:")
    print("  - voronoi_maze_features.png")
    print("  - algorithm_comparison.png")
    print("  - mfg_application_scenarios.png")


if __name__ == "__main__":
    main()
