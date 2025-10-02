"""
Maze Post-Processing Demonstration

Demonstrates wall smoothing and adaptive door width enhancements for organic mazes.

Features:
1. Wall smoothing for CA and Voronoi mazes (reduces zigzags)
2. Comparison of smoothing methods
3. Adaptive door width for connectivity
4. Before/after visual comparisons

Run: python examples/advanced/maze_postprocessing_demo.py
"""

import numpy as np

# Check scipy availability
try:
    import scipy.ndimage  # noqa: F401

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available. Post-processing features disabled.")
    print("Install with: pip install scipy")

import matplotlib.pyplot as plt

from mfg_pde.alg.reinforcement.environments import (
    CellularAutomataConfig,
    CellularAutomataGenerator,
    VoronoiMazeConfig,
    VoronoiMazeGenerator,
)

if SCIPY_AVAILABLE:
    from mfg_pde.alg.reinforcement.environments import (
        enhance_organic_maze,
        smooth_walls_combined,
        smooth_walls_gaussian,
        smooth_walls_morphological,
    )


def demo_wall_smoothing_methods():
    """Compare different wall smoothing methods on CA maze."""
    print("=" * 60)
    print("Wall Smoothing Methods Comparison")
    print("=" * 60)

    if not SCIPY_AVAILABLE:
        print("Skipping: scipy required for post-processing")
        return

    # Generate organic CA maze
    config = CellularAutomataConfig(
        rows=50,
        cols=50,
        initial_wall_prob=0.45,
        num_iterations=5,
        seed=42,
    )
    generator = CellularAutomataGenerator(config)
    original = generator.generate()

    print(f"\n✓ Generated CA maze: {config.rows}×{config.cols}")

    # Apply different smoothing methods
    smoothed_morph = smooth_walls_morphological(original, iterations=1, operation="open")
    smoothed_gaussian = smooth_walls_gaussian(original, sigma=1.0, threshold=0.5)
    smoothed_combined = smooth_walls_combined(original, morph_iterations=1, gaussian_sigma=0.8)

    print("✓ Applied 3 smoothing methods:")
    print("  - Morphological (opening)")
    print("  - Gaussian blur (σ=1.0)")
    print("  - Combined (morph + Gaussian)")

    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    axes[0, 0].imshow(original, cmap="binary", interpolation="nearest")
    axes[0, 0].set_title("Original CA Maze\n(Zigzag Boundaries)", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(smoothed_morph, cmap="binary", interpolation="nearest")
    axes[0, 1].set_title("Morphological Smoothing\n(Erosion → Dilation)", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(smoothed_gaussian, cmap="binary", interpolation="nearest")
    axes[1, 0].set_title("Gaussian Smoothing\n(Blur + Threshold)", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(smoothed_combined, cmap="binary", interpolation="nearest")
    axes[1, 1].set_title("Combined Smoothing ⭐\n(Recommended)", fontsize=12, fontweight="bold", color="green")
    axes[1, 1].axis("off")

    plt.suptitle(
        "Wall Smoothing Methods for Organic Mazes",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig("maze_smoothing_comparison.png", dpi=300, bbox_inches="tight")
    print("\n✓ Saved: maze_smoothing_comparison.png (300 DPI)")
    plt.close()


def demo_smoothing_strengths():
    """Show effect of different smoothing strengths."""
    print("\n" + "=" * 60)
    print("Smoothing Strength Comparison")
    print("=" * 60)

    if not SCIPY_AVAILABLE:
        print("Skipping: scipy required for post-processing")
        return

    # Generate Voronoi maze (more zigzags than CA)
    config = VoronoiMazeConfig(
        rows=60,
        cols=60,
        num_points=15,
        wall_thickness=1,
        seed=42,
    )
    generator = VoronoiMazeGenerator(config)
    original = generator.generate()

    print(f"\n✓ Generated Voronoi maze: {config.rows}×{config.cols}")

    # Apply different strengths
    light = enhance_organic_maze(original, smoothing_strength="light")
    medium = enhance_organic_maze(original, smoothing_strength="medium")
    strong = enhance_organic_maze(original, smoothing_strength="strong")

    print("✓ Applied 3 smoothing strengths:")
    print("  - Light (morph=1, σ=0.5)")
    print("  - Medium (morph=1, σ=0.8) [RECOMMENDED]")
    print("  - Strong (morph=2, σ=1.2)")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    axes[0, 0].imshow(original, cmap="binary", interpolation="nearest")
    axes[0, 0].set_title("Original Voronoi Maze\n(Rasterization Artifacts)", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(light, cmap="binary", interpolation="nearest")
    axes[0, 1].set_title("Light Smoothing\n(Subtle Refinement)", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(medium, cmap="binary", interpolation="nearest")
    axes[1, 0].set_title("Medium Smoothing ⭐\n(Recommended)", fontsize=12, fontweight="bold", color="green")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(strong, cmap="binary", interpolation="nearest")
    axes[1, 1].set_title("Strong Smoothing\n(Aggressive)", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")

    plt.suptitle(
        "Smoothing Strength Effect on Voronoi Mazes",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig("smoothing_strength_comparison.png", dpi=300, bbox_inches="tight")
    print("\n✓ Saved: smoothing_strength_comparison.png (300 DPI)")
    plt.close()


def demo_before_after_zoom():
    """Zoomed comparison showing detail improvement."""
    print("\n" + "=" * 60)
    print("Detailed Before/After Comparison (Zoomed)")
    print("=" * 60)

    if not SCIPY_AVAILABLE:
        print("Skipping: scipy required for post-processing")
        return

    # Generate CA maze
    config = CellularAutomataConfig(
        rows=80,
        cols=80,
        initial_wall_prob=0.45,
        num_iterations=6,
        seed=123,
    )
    generator = CellularAutomataGenerator(config)
    original = generator.generate()

    # Enhance
    enhanced = enhance_organic_maze(original, smoothing_strength="medium")

    print(f"\n✓ Generated and enhanced CA maze: {config.rows}×{config.cols}")

    # Zoom into interesting region
    zoom_slice = (slice(20, 50), slice(20, 50))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(original[zoom_slice], cmap="binary", interpolation="nearest")
    axes[0].set_title("BEFORE: Zigzag Boundaries", fontsize=14, fontweight="bold", color="red")
    axes[0].axis("off")
    # Add grid to show pixels
    axes[0].grid(True, which="both", color="gray", linewidth=0.5, alpha=0.3)

    axes[1].imshow(enhanced[zoom_slice], cmap="binary", interpolation="nearest")
    axes[1].set_title("AFTER: Smooth Boundaries", fontsize=14, fontweight="bold", color="green")
    axes[1].axis("off")
    axes[1].grid(True, which="both", color="gray", linewidth=0.5, alpha=0.3)

    plt.suptitle(
        "Wall Smoothing: Pixel-Level Detail (Zoomed 30×30 Region)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("smoothing_detail_zoom.png", dpi=300, bbox_inches="tight")
    print("\n✓ Saved: smoothing_detail_zoom.png (300 DPI)")
    plt.close()


def demo_adaptive_door_width():
    """Demonstrate adaptive door width concept with visualization."""
    print("\n" + "=" * 60)
    print("Adaptive Door Width Demonstration")
    print("=" * 60)

    print("\nConcept: Door width adapts to zone sizes")
    print("  - Small zones (10×10): width = 1-2 cells")
    print("  - Medium zones (20×20): width = 2-3 cells")
    print("  - Large zones (40×40): width = 3-5 cells")

    # Create example with manual zone assignments
    maze = np.ones((60, 80), dtype=np.int32)
    zone_map = np.zeros((60, 80), dtype=np.int32)

    # Create 3 zones of different sizes
    # Zone 0: Small (10×10)
    maze[5:15, 5:15] = 0
    zone_map[5:15, 5:15] = 0

    # Zone 1: Medium (20×20)
    maze[5:25, 25:45] = 0
    zone_map[5:25, 25:45] = 1

    # Zone 2: Large (30×30)
    maze[25:55, 5:35] = 0
    zone_map[25:55, 5:35] = 2

    # Zone 3: Medium (20×30)
    maze[25:45, 45:75] = 0
    zone_map[25:45, 45:75] = 3

    print("\n✓ Created 4 zones:")
    print("  Zone 0 (small): 10×10 = 100 cells")
    print("  Zone 1 (medium): 20×20 = 400 cells")
    print("  Zone 2 (large): 30×30 = 900 cells")
    print("  Zone 3 (medium): 20×30 = 600 cells")

    # Apply adaptive doors
    if SCIPY_AVAILABLE:
        from mfg_pde.alg.reinforcement.environments import adaptive_door_carving

        # Connect zones with adaptive doors
        maze = adaptive_door_carving(maze, zone_map, zone_i=0, zone_j=1, base_door_width=2)
        maze = adaptive_door_carving(maze, zone_map, zone_i=1, zone_j=3, base_door_width=2)
        maze = adaptive_door_carving(maze, zone_map, zone_i=0, zone_j=2, base_door_width=2)
        maze = adaptive_door_carving(maze, zone_map, zone_i=2, zone_j=3, base_door_width=2)

        print("✓ Added adaptive doors between zones")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Show zones
    axes[0].imshow(zone_map, cmap="tab10", interpolation="nearest")
    axes[0].set_title("Zone Map (4 Different Sizes)", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Show maze with adaptive doors
    axes[1].imshow(maze, cmap="binary", interpolation="nearest")
    axes[1].set_title("Adaptive Door Widths\n(Larger zones → Wider doors)", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    plt.suptitle(
        "Adaptive Door Width Based on Zone Size",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("adaptive_door_width_demo.png", dpi=300, bbox_inches="tight")
    print("\n✓ Saved: adaptive_door_width_demo.png (300 DPI)")
    plt.close()


def demo_production_workflow():
    """Recommended production workflow for organic mazes."""
    print("\n" + "=" * 60)
    print("Production Workflow: CA → Enhanced → Ready for MFG")
    print("=" * 60)

    if not SCIPY_AVAILABLE:
        print("Skipping: scipy required for post-processing")
        return

    # Step 1: Generate organic maze
    config = CellularAutomataConfig(
        rows=70,
        cols=70,
        initial_wall_prob=0.42,
        num_iterations=6,
        seed=456,
    )
    generator = CellularAutomataGenerator(config)
    original = generator.generate()

    print("\nStep 1: Generate organic CA maze")
    print(f"  ✓ Maze size: {config.rows}×{config.cols}")
    print(f"  ✓ Open cells: {np.sum(original == 0)}")

    # Step 2: Apply enhancement
    enhanced = enhance_organic_maze(
        original,
        smoothing_strength="medium",
        preserve_connectivity=True,
    )

    print("\nStep 2: Apply enhancement pipeline")
    print("  ✓ Wall smoothing (medium)")
    print("  ✓ Thickness normalization")
    print("  ✓ Connectivity verification")

    # Step 3: Verify improvement
    original_edges = _count_boundary_edges(original)
    enhanced_edges = _count_boundary_edges(enhanced)
    smoothness_improvement = 100 * (1 - enhanced_edges / original_edges)

    print("\nStep 3: Quality metrics")
    print(f"  ✓ Boundary edges: {original_edges} → {enhanced_edges}")
    print(f"  ✓ Smoothness improvement: {smoothness_improvement:.1f}%")
    print(f"  ✓ Open cells preserved: {np.sum(enhanced == 0)}")

    # Visualize workflow
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(original, cmap="binary", interpolation="nearest")
    axes[0].set_title("Step 1: Raw Generation\n(Zigzag Artifacts)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(enhanced, cmap="binary", interpolation="nearest")
    axes[1].set_title("Step 2: Enhanced\n(Smooth Boundaries)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Show difference (where changes occurred)
    diff = np.abs(original.astype(int) - enhanced.astype(int))
    axes[2].imshow(diff, cmap="Reds", interpolation="nearest")
    axes[2].set_title(
        f"Step 3: Changes Made\n({smoothness_improvement:.1f}% smoother)",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].axis("off")

    plt.suptitle(
        "Production Workflow: Organic Maze Enhancement",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("production_workflow.png", dpi=300, bbox_inches="tight")
    print("\n✓ Saved: production_workflow.png (300 DPI)")
    plt.close()

    print("\n" + "=" * 60)
    print("✓ Ready for MFG simulation!")
    print("=" * 60)


def _count_boundary_edges(maze: np.ndarray) -> int:
    """Count number of wall-open boundaries (proxy for roughness)."""
    rows, cols = maze.shape
    edge_count = 0

    for r in range(rows - 1):
        for c in range(cols - 1):
            # Check if wall-open boundary exists
            if maze[r, c] != maze[r + 1, c]:
                edge_count += 1
            if maze[r, c] != maze[r, c + 1]:
                edge_count += 1

    return edge_count


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("MAZE POST-PROCESSING DEMONSTRATION")
    print("=" * 60)
    print("\nFeatures:")
    print("1. Wall smoothing methods comparison")
    print("2. Smoothing strength effects")
    print("3. Detailed before/after zoom")
    print("4. Adaptive door width")
    print("5. Production workflow")
    print()

    if not SCIPY_AVAILABLE:
        print("ERROR: scipy required for post-processing demonstrations")
        print("Install with: pip install scipy")
        return

    # Run all demos
    demo_wall_smoothing_methods()
    demo_smoothing_strengths()
    demo_before_after_zoom()
    demo_adaptive_door_width()
    demo_production_workflow()

    print("\n" + "=" * 60)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - maze_smoothing_comparison.png")
    print("  - smoothing_strength_comparison.png")
    print("  - smoothing_detail_zoom.png")
    print("  - adaptive_door_width_demo.png")
    print("  - production_workflow.png")
    print("\nRecommended for production:")
    print("  enhance_organic_maze(maze, smoothing_strength='medium')")
    print()


if __name__ == "__main__":
    main()
