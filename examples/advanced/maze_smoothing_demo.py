"""
Maze Smoothing and Adaptive Connectivity Demonstration

Demonstrates wall smoothing and adaptive door width features for organic mazes.
Shows how to:
1. Generate organic mazes (CA, Voronoi)
2. Apply morphological/Gaussian smoothing to reduce zigzag boundaries
3. Ensure connectivity with adaptive door widths
4. Analyze maze connectivity properties

MFG Research Value:
- Smooth boundaries create more realistic building layouts
- Adaptive door widths model realistic bottleneck dynamics
- Connectivity analysis validates environment quality

Author: MFG_PDE Team
Date: October 2025
"""

import matplotlib.pyplot as plt

from mfg_pde.alg.reinforcement.environments import (
    CellularAutomataConfig,
    CellularAutomataGenerator,
    VoronoiMazeConfig,
    VoronoiMazeGenerator,
    analyze_maze_connectivity,
    connect_regions_adaptive,
    smooth_walls_gaussian,
    smooth_walls_morphological,
)


def visualize_maze_processing(original, smoothed, connected, title_prefix):
    """Visualize maze processing pipeline."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(original, cmap="binary", interpolation="nearest")
    axes[0].set_title(f"{title_prefix}: Original")
    axes[0].axis("off")

    # Smoothed
    axes[1].imshow(smoothed, cmap="binary", interpolation="nearest")
    axes[1].set_title(f"{title_prefix}: Smoothed")
    axes[1].axis("off")

    # Connected
    axes[2].imshow(connected, cmap="binary", interpolation="nearest")
    axes[2].set_title(f"{title_prefix}: Connected")
    axes[2].axis("off")

    plt.tight_layout()
    return fig


def demo_ca_smoothing():
    """Demonstrate CA maze smoothing and connectivity."""
    print("=" * 60)
    print("Cellular Automata Maze: Smoothing & Adaptive Connectivity")
    print("=" * 60)

    # Generate CA maze (may have disconnections)
    config = CellularAutomataConfig(
        rows=60,
        cols=60,
        initial_wall_prob=0.45,
        num_iterations=5,
        ensure_connectivity=False,  # Test connectivity repair
        seed=42,
    )
    generator = CellularAutomataGenerator(config)
    original = generator.generate()

    # Analyze original
    original_analysis = analyze_maze_connectivity(original)
    print("\nOriginal Maze:")
    print(f"  - Regions: {original_analysis['num_regions']}")
    print(f"  - Largest region: {original_analysis['largest_region_size']} cells")
    print(f"  - Connectivity ratio: {original_analysis['connectivity_ratio']:.1%}")
    print(f"  - Connected: {original_analysis['is_connected']}")

    # Apply morphological smoothing
    smoothed = smooth_walls_morphological(original, iterations=2, method="opening")
    print("\nAfter Morphological Smoothing (opening, 2 iterations):")
    smoothed_analysis = analyze_maze_connectivity(smoothed)
    print(f"  - Regions: {smoothed_analysis['num_regions']}")
    print(f"  - Connected: {smoothed_analysis['is_connected']}")

    # Ensure connectivity with adaptive doors
    connected = connect_regions_adaptive(smoothed, min_door_width=2, max_door_width=4)
    final_analysis = analyze_maze_connectivity(connected)
    print("\nAfter Adaptive Connectivity:")
    print(f"  - Regions: {final_analysis['num_regions']}")
    print(f"  - Connected: {final_analysis['is_connected']}")
    print(f"  - Connectivity ratio: {final_analysis['connectivity_ratio']:.1%}")

    # Visualize
    visualize_maze_processing(original, smoothed, connected, "Cellular Automata")
    plt.savefig("ca_maze_smoothing.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved: ca_maze_smoothing.png")
    plt.close()


def demo_voronoi_smoothing():
    """Demonstrate Voronoi maze smoothing."""
    print("\n" + "=" * 60)
    print("Voronoi Maze: Gaussian Smoothing")
    print("=" * 60)

    # Generate Voronoi maze
    config = VoronoiMazeConfig(
        rows=60,
        cols=60,
        num_points=15,
        wall_thickness=1,
        door_width=2,
        seed=42,
    )
    generator = VoronoiMazeGenerator(config)
    original = generator.generate()

    # Analyze original
    original_analysis = analyze_maze_connectivity(original)
    print("\nOriginal Voronoi Maze:")
    print(f"  - Regions: {original_analysis['num_regions']}")
    print(f"  - Connected: {original_analysis['is_connected']}")
    print(f"  - Open space: {original_analysis['total_open_space']} cells")

    # Apply Gaussian smoothing
    smoothed = smooth_walls_gaussian(original, sigma=1.5, threshold=0.5)
    smoothed_analysis = analyze_maze_connectivity(smoothed)
    print("\nAfter Gaussian Smoothing (σ=1.5):")
    print(f"  - Regions: {smoothed_analysis['num_regions']}")
    print(f"  - Connected: {smoothed_analysis['is_connected']}")

    # Ensure connectivity if needed
    if not smoothed_analysis["is_connected"]:
        connected = connect_regions_adaptive(smoothed, min_door_width=2, max_door_width=4)
        final_analysis = analyze_maze_connectivity(connected)
        print("\nAfter Adaptive Connectivity:")
        print(f"  - Connected: {final_analysis['is_connected']}")
    else:
        connected = smoothed
        print("\n✓ Already connected after smoothing!")

    # Visualize
    visualize_maze_processing(original, smoothed, connected, "Voronoi Diagram")
    plt.savefig("voronoi_maze_smoothing.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved: voronoi_maze_smoothing.png")
    plt.close()


def demo_smoothing_comparison():
    """Compare morphological vs Gaussian smoothing."""
    print("\n" + "=" * 60)
    print("Smoothing Methods Comparison")
    print("=" * 60)

    # Generate test maze
    config = CellularAutomataConfig(rows=60, cols=60, initial_wall_prob=0.45, num_iterations=5, seed=123)
    generator = CellularAutomataGenerator(config)
    original = generator.generate()

    # Apply different smoothing methods
    morph_opening = smooth_walls_morphological(original, iterations=1, method="opening")
    morph_closing = smooth_walls_morphological(original, iterations=1, method="closing")
    morph_both = smooth_walls_morphological(original, iterations=1, method="both")
    gaussian = smooth_walls_gaussian(original, sigma=1.0)

    # Visualize comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(original, cmap="binary", interpolation="nearest")
    axes[0, 0].set_title("Original CA Maze")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(morph_opening, cmap="binary", interpolation="nearest")
    axes[0, 1].set_title("Morphological: Opening")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(morph_closing, cmap="binary", interpolation="nearest")
    axes[0, 2].set_title("Morphological: Closing")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(morph_both, cmap="binary", interpolation="nearest")
    axes[1, 0].set_title("Morphological: Both")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gaussian, cmap="binary", interpolation="nearest")
    axes[1, 1].set_title("Gaussian Smoothing (σ=1.0)")
    axes[1, 1].axis("off")

    # Hide last subplot
    axes[1, 2].axis("off")

    plt.suptitle("Smoothing Methods Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("smoothing_comparison.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved: smoothing_comparison.png")
    plt.close()

    # Print analysis
    print("\nSmoothing Method Analysis:")
    for name, maze in [
        ("Opening", morph_opening),
        ("Closing", morph_closing),
        ("Both", morph_both),
        ("Gaussian", gaussian),
    ]:
        analysis = analyze_maze_connectivity(maze)
        print(f"  {name:12s}: {analysis['num_regions']} regions, " f"{analysis['connectivity_ratio']:.1%} connected")


def demo_adaptive_door_widths():
    """Demonstrate adaptive door width based on zone sizes."""
    print("\n" + "=" * 60)
    print("Adaptive Door Width Demonstration")
    print("=" * 60)

    from mfg_pde.alg.reinforcement.environments import compute_adaptive_door_width

    # Test different region sizes
    test_cases = [
        (50, 60, "Small regions"),
        (500, 600, "Medium regions"),
        (2000, 2500, "Large regions"),
        (10000, 12000, "Very large regions"),
    ]

    print("\nAdaptive Door Width Calculation:")
    print("-" * 50)
    for size_a, size_b, description in test_cases:
        width = compute_adaptive_door_width(size_a, size_b, min_width=1, max_width=5)
        avg_size = (size_a + size_b) / 2
        print(f"{description:20s} (avg={avg_size:6.0f}): door width = {width} cells")

    print("\n✓ Larger zones automatically get wider doors for better flow!")


if __name__ == "__main__":
    # Run all demonstrations
    demo_ca_smoothing()
    demo_voronoi_smoothing()
    demo_smoothing_comparison()
    demo_adaptive_door_widths()

    print("\n" + "=" * 60)
    print("Maze Smoothing & Connectivity Demo Complete!")
    print("=" * 60)
    print("\nGenerated Files:")
    print("  - ca_maze_smoothing.png")
    print("  - voronoi_maze_smoothing.png")
    print("  - smoothing_comparison.png")
    print("\nKey Takeaways:")
    print("  1. Morphological smoothing reduces zigzag artifacts")
    print("  2. Gaussian smoothing creates very smooth boundaries")
    print("  3. Adaptive door widths improve MFG flow realism")
    print("  4. Connectivity analysis validates environment quality")
