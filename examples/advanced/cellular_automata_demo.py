#!/usr/bin/env python3
"""
Cellular Automata Maze Generation Demo

Demonstrates organic, cave-like maze generation using cellular automata:
1. Basic CA maze generation
2. Effect of wall probability
3. Effect of iterations (smoothing)
4. Preset styles comparison
5. Comparison with structured mazes

Author: MFG_PDE Team
Date: October 2025
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.reinforcement.environments import (
    CellularAutomataConfig,
    CellularAutomataGenerator,
    MazeAlgorithm,
    PerfectMazeGenerator,
    RecursiveDivisionGenerator,
    create_preset_ca_config,
    create_room_based_config,
)


def demo_basic_ca():
    """Demonstrate basic CA maze generation."""
    print("=" * 70)
    print("Demo 1: Basic Cellular Automata Generation")
    print("=" * 70)

    config = CellularAutomataConfig(
        rows=50,
        cols=50,
        initial_wall_prob=0.45,
        num_iterations=5,
        seed=42,
    )

    generator = CellularAutomataGenerator(config)
    maze = generator.generate()

    open_cells = np.sum(maze == 0)
    total_cells = maze.size
    print(f"\nMaze dimensions: {maze.shape}")
    print(f"Open space: {open_cells}/{total_cells} ({100*open_cells/total_cells:.1f}%)")
    print("Parameters: wall_prob=0.45, iterations=5")


def demo_wall_probability():
    """Demonstrate effect of wall probability."""
    print("\n" + "=" * 70)
    print("Demo 2: Wall Probability Effect")
    print("=" * 70)

    probabilities = [0.30, 0.45, 0.60]

    for prob in probabilities:
        config = CellularAutomataConfig(rows=50, cols=50, initial_wall_prob=prob, num_iterations=5, seed=42)
        generator = CellularAutomataGenerator(config)
        maze = generator.generate()

        open_pct = 100 * np.sum(maze == 0) / maze.size
        print(f"\nWall probability {prob:.2f}:")
        print(f"  Open space: {open_pct:.1f}%")


def demo_iterations():
    """Demonstrate effect of smoothing iterations."""
    print("\n" + "=" * 70)
    print("Demo 3: Smoothing Iterations Effect")
    print("=" * 70)

    iterations_list = [0, 3, 5, 8]

    for iters in iterations_list:
        config = CellularAutomataConfig(rows=50, cols=50, initial_wall_prob=0.45, num_iterations=iters, seed=42)
        generator = CellularAutomataGenerator(config)
        maze = generator.generate()

        print(f"\n{iters} iterations:")
        print(f"  Open space: {100*np.sum(maze == 0)/maze.size:.1f}%")


def demo_preset_styles():
    """Demonstrate preset style configurations."""
    print("\n" + "=" * 70)
    print("Demo 4: Preset Styles")
    print("=" * 70)

    styles = ["cave", "cavern", "maze", "dense", "sparse"]

    for style in styles:
        config = create_preset_ca_config(50, 50, style=style, seed=42)
        generator = CellularAutomataGenerator(config)
        maze = generator.generate()

        print(f"\n{style.capitalize()} style:")
        print(f"  Wall probability: {config.initial_wall_prob:.2f}")
        print(f"  Iterations: {config.num_iterations}")
        print(f"  Open space: {100*np.sum(maze == 0)/maze.size:.1f}%")


def visualize_ca_comparison():
    """Visualize different CA configurations."""
    print("\n" + "=" * 70)
    print("Visual Comparison: CA Configurations")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Low iterations (noisy)
    config1 = CellularAutomataConfig(rows=60, cols=60, initial_wall_prob=0.45, num_iterations=1, seed=42)
    gen1 = CellularAutomataGenerator(config1)
    maze1 = gen1.generate()
    axes[0, 0].imshow(maze1, cmap="binary", interpolation="nearest")
    axes[0, 0].set_title("Noisy (1 iteration)", fontweight="bold")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    # 2. Medium iterations
    config2 = CellularAutomataConfig(rows=60, cols=60, initial_wall_prob=0.45, num_iterations=5, seed=42)
    gen2 = CellularAutomataGenerator(config2)
    maze2 = gen2.generate()
    axes[0, 1].imshow(maze2, cmap="binary", interpolation="nearest")
    axes[0, 1].set_title("Smooth (5 iterations)", fontweight="bold")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    # 3. High iterations
    config3 = CellularAutomataConfig(rows=60, cols=60, initial_wall_prob=0.45, num_iterations=10, seed=42)
    gen3 = CellularAutomataGenerator(config3)
    maze3 = gen3.generate()
    axes[0, 2].imshow(maze3, cmap="binary", interpolation="nearest")
    axes[0, 2].set_title("Very Smooth (10 iterations)", fontweight="bold")
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])

    # 4. Sparse (low wall prob)
    config4 = create_preset_ca_config(60, 60, style="sparse", seed=42)
    gen4 = CellularAutomataGenerator(config4)
    maze4 = gen4.generate()
    axes[1, 0].imshow(maze4, cmap="binary", interpolation="nearest")
    axes[1, 0].set_title("Sparse (35% walls)", fontweight="bold")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    # 5. Balanced
    config5 = create_preset_ca_config(60, 60, style="cave", seed=42)
    gen5 = CellularAutomataGenerator(config5)
    maze5 = gen5.generate()
    axes[1, 1].imshow(maze5, cmap="binary", interpolation="nearest")
    axes[1, 1].set_title("Cave (45% walls)", fontweight="bold")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    # 6. Dense (high wall prob)
    config6 = create_preset_ca_config(60, 60, style="dense", seed=42)
    gen6 = CellularAutomataGenerator(config6)
    maze6 = gen6.generate()
    axes[1, 2].imshow(maze6, cmap="binary", interpolation="nearest")
    axes[1, 2].set_title("Dense (55% walls)", fontweight="bold")
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])

    fig.suptitle("Cellular Automata: Configuration Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("cellular_automata_comparison.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved: cellular_automata_comparison.png")
    plt.show()


def visualize_algorithm_comparison():
    """Compare CA with other maze generation algorithms."""
    print("\n" + "=" * 70)
    print("Visual Comparison: Different Maze Algorithms")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # 1. Perfect maze (structured, unit-width)
    gen1 = PerfectMazeGenerator(30, 30, MazeAlgorithm.RECURSIVE_BACKTRACKING)
    gen1.generate(seed=42)
    maze1 = gen1.to_numpy_array()
    axes[0, 0].imshow(maze1, cmap="binary", interpolation="nearest")
    axes[0, 0].set_title("Perfect Maze\n(Recursive Backtracking)", fontweight="bold")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    # 2. Recursive Division (rooms and corridors)
    config_rd = create_room_based_config(30, 30, room_size="medium", corridor_width="medium", seed=42)
    gen2 = RecursiveDivisionGenerator(config_rd)
    maze2 = gen2.generate()
    axes[0, 1].imshow(maze2, cmap="binary", interpolation="nearest")
    axes[0, 1].set_title("Recursive Division\n(Rooms & Corridors)", fontweight="bold")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    # 3. Cellular Automata - Cave
    config_cave = create_preset_ca_config(30, 30, style="cave", seed=42)
    gen3 = CellularAutomataGenerator(config_cave)
    maze3 = gen3.generate()
    axes[1, 0].imshow(maze3, cmap="binary", interpolation="nearest")
    axes[1, 0].set_title("Cellular Automata\n(Cave Style)", fontweight="bold")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    # 4. Cellular Automata - Cavern (open)
    config_cavern = create_preset_ca_config(30, 30, style="cavern", seed=42)
    gen4 = CellularAutomataGenerator(config_cavern)
    maze4 = gen4.generate()
    axes[1, 1].imshow(maze4, cmap="binary", interpolation="nearest")
    axes[1, 1].set_title("Cellular Automata\n(Cavern Style)", fontweight="bold")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    fig.suptitle("Maze Generation Algorithms: Side-by-Side Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("algorithm_comparison.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved: algorithm_comparison.png")
    plt.show()


def demo_mfg_scenarios():
    """Demonstrate CA for MFG scenarios."""
    print("\n" + "=" * 70)
    print("Demo 5: MFG Application Scenarios")
    print("=" * 70)

    scenarios = [
        ("Park/Plaza (Sparse)", "sparse"),
        ("Natural Terrain (Cave)", "cave"),
        ("Irregular Urban Space (Dense)", "dense"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (name, style) in enumerate(scenarios):
        config = create_preset_ca_config(40, 50, style=style, seed=42)
        generator = CellularAutomataGenerator(config)
        maze = generator.generate()

        axes[idx].imshow(maze, cmap="binary", interpolation="nearest")
        axes[idx].set_title(name, fontweight="bold")
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

        open_pct = 100 * np.sum(maze == 0) / maze.size
        info_text = f"Open: {open_pct:.1f}%\nStyle: {style}"
        axes[idx].text(
            0.02,
            0.98,
            info_text,
            transform=axes[idx].transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            fontsize=9,
        )

    fig.suptitle("MFG Scenarios with Cellular Automata", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("ca_mfg_scenarios.png", dpi=150, bbox_inches="tight")
    print("\nMFG scenarios saved: ca_mfg_scenarios.png")
    plt.show()


def main():
    """Run all demonstrations."""
    demo_basic_ca()
    demo_wall_probability()
    demo_iterations()
    demo_preset_styles()

    print("\n" + "=" * 70)
    print("Generating Visualizations...")
    print("=" * 70)

    visualize_ca_comparison()
    visualize_algorithm_comparison()
    demo_mfg_scenarios()

    print("\n" + "=" * 70)
    print("All Demonstrations Complete")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Organic, cave-like maze generation")
    print("  ✓ Configurable wall probability and smoothing")
    print("  ✓ 5 preset styles (cave, cavern, maze, dense, sparse)")
    print("  ✓ Comparison with structured algorithms")
    print("  ✓ MFG application scenarios")
    print("\nGenerated Images:")
    print("  - cellular_automata_comparison.png")
    print("  - algorithm_comparison.png")
    print("  - ca_mfg_scenarios.png")


if __name__ == "__main__":
    main()
