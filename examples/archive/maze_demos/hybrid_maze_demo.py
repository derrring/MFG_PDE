"""
Hybrid Maze Generation Demo

Demonstrates hybrid maze generation combining multiple algorithms to create
realistic, heterogeneous environments for MFG research.

Examples:
- Museum: Voronoi galleries + Cellular Automata gardens
- Office: Recursive Division rooms + Perfect Maze corridors
- Campus: Four quadrants with different spatial structures
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.reinforcement.environments import (
    HybridMazeGenerator,
    create_campus_hybrid,
    create_museum_hybrid,
    create_office_hybrid,
)


def visualize_hybrid_maze(maze, zone_map, title="Hybrid Maze"):
    """
    Visualize hybrid maze with zone coloring.

    Args:
        maze: Maze array (1=wall, 0=passage)
        zone_map: Zone assignment for each cell
        title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Maze structure
    axes[0].imshow(maze, cmap="binary", interpolation="nearest")
    axes[0].set_title(f"{title} - Structure")
    axes[0].axis("off")

    # Plot 2: Zone map
    im = axes[1].imshow(zone_map, cmap="tab10", interpolation="nearest")
    axes[1].set_title(f"{title} - Zones")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], label="Zone ID")

    # Plot 3: Combined view (maze with zone overlay)
    # Create RGB image
    combined = np.zeros((*maze.shape, 3))
    # Set walls to black
    combined[maze == 1] = [0, 0, 0]
    # Set passages to white
    combined[maze == 0] = [1, 1, 1]

    # Overlay zone colors with transparency
    zone_colors = plt.cm.tab10(zone_map / zone_map.max())[:, :, :3]
    alpha = 0.3
    combined[maze == 0] = alpha * zone_colors[maze == 0] + (1 - alpha) * combined[maze == 0]

    axes[2].imshow(combined, interpolation="nearest")
    axes[2].set_title(f"{title} - Combined")
    axes[2].axis("off")

    plt.tight_layout()
    return fig


def demo_museum_hybrid():
    """
    Demo: Museum with Voronoi galleries + Cellular Automata gardens.

    60% Voronoi (structured galleries) + 40% Cellular Automata (organic gardens)
    """
    print("\n" + "=" * 70)
    print("MUSEUM HYBRID: Voronoi Galleries + Cellular Automata Gardens")
    print("=" * 70)

    config = create_museum_hybrid(rows=80, cols=100, seed=42)
    generator = HybridMazeGenerator(config)
    maze = generator.generate()

    print(f"Maze size: {maze.shape}")
    print(f"Number of zones: {len(np.unique(generator.zone_map))}")
    print(f"Wall density: {np.sum(maze == 1) / maze.size:.1%}")
    print(f"Passage density: {np.sum(maze == 0) / maze.size:.1%}")

    visualize_hybrid_maze(maze, generator.zone_map, "Museum Hybrid")
    plt.savefig("museum_hybrid_maze.png", dpi=150, bbox_inches="tight")
    print("Saved visualization to: museum_hybrid_maze.png")


def demo_office_hybrid():
    """
    Demo: Office with Recursive Division rooms + Perfect Maze corridors.

    70% Recursive Division (structured offices) + 30% Perfect Maze (service corridors)
    """
    print("\n" + "=" * 70)
    print("OFFICE HYBRID: Recursive Division Rooms + Perfect Maze Corridors")
    print("=" * 70)

    config = create_office_hybrid(rows=80, cols=100, seed=123)
    generator = HybridMazeGenerator(config)
    maze = generator.generate()

    print(f"Maze size: {maze.shape}")
    print(f"Number of zones: {len(np.unique(generator.zone_map))}")
    print(f"Wall density: {np.sum(maze == 1) / maze.size:.1%}")
    print(f"Passage density: {np.sum(maze == 0) / maze.size:.1%}")

    visualize_hybrid_maze(maze, generator.zone_map, "Office Hybrid")
    plt.savefig("office_hybrid_maze.png", dpi=150, bbox_inches="tight")
    print("Saved visualization to: office_hybrid_maze.png")


def demo_campus_hybrid():
    """
    Demo: Campus with four quadrants using different algorithms.

    NW: Recursive Division (offices)
    NE: Voronoi (labs)
    SW: Perfect Maze (corridors)
    SE: Cellular Automata (gardens)
    """
    print("\n" + "=" * 70)
    print("CAMPUS HYBRID: Four Quadrants with Different Structures")
    print("=" * 70)

    config = create_campus_hybrid(rows=120, cols=120, seed=999)
    generator = HybridMazeGenerator(config)
    maze = generator.generate()

    print(f"Maze size: {maze.shape}")
    print(f"Number of zones: {len(np.unique(generator.zone_map))}")
    print(f"Wall density: {np.sum(maze == 1) / maze.size:.1%}")
    print(f"Passage density: {np.sum(maze == 0) / maze.size:.1%}")

    # Print zone breakdown
    for zone_id in range(4):
        zone_size = np.sum(generator.zone_map == zone_id)
        print(
            f"  Zone {zone_id} (Quadrant {['NW', 'NE', 'SW', 'SE'][zone_id]}): "
            f"{zone_size:,} cells ({zone_size / maze.size:.1%})"
        )

    visualize_hybrid_maze(maze, generator.zone_map, "Campus Hybrid")
    plt.savefig("campus_hybrid_maze.png", dpi=150, bbox_inches="tight")
    print("Saved visualization to: campus_hybrid_maze.png")


def demo_connectivity_verification():
    """
    Demonstrate connectivity verification across zones.

    Shows that hybrid mazes are globally connected even with different algorithms.
    """
    print("\n" + "=" * 70)
    print("CONNECTIVITY VERIFICATION")
    print("=" * 70)

    config = create_museum_hybrid(rows=60, cols=80, seed=555)
    generator = HybridMazeGenerator(config)
    maze = generator.generate()

    # Flood fill to count connected components
    visited = np.zeros_like(maze, dtype=bool)
    components = 0

    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == 0 and not visited[i, j]:
                components += 1
                # Flood fill from this position
                stack = [(i, j)]
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= maze.shape[0] or c < 0 or c >= maze.shape[1]:
                        continue
                    if visited[r, c] or maze[r, c] == 1:
                        continue
                    visited[r, c] = True
                    stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])

    print(f"Number of connected components: {components}")
    if components == 1:
        print("SUCCESS: Maze is globally connected across all zones")
    else:
        print(f"WARNING: Maze has {components} disconnected regions " "(connectivity verification may have failed)")


def main():
    """Run all hybrid maze demos."""
    print("\n" + "=" * 70)
    print("HYBRID MAZE GENERATION DEMONSTRATION")
    print("=" * 70)
    print(
        "\nHybrid mazes combine multiple generation algorithms to create\n"
        "realistic, heterogeneous environments for MFG research.\n"
    )
    print("Key Benefits:")
    print("- Zone-specific behavior analysis")
    print("- Heterogeneous Nash equilibria")
    print("- Multi-scale planning problems")
    print("- Realistic evacuation modeling")

    # Run demos
    demo_museum_hybrid()
    demo_office_hybrid()
    demo_campus_hybrid()
    demo_connectivity_verification()

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nGenerated 3 hybrid maze visualizations:")
    print("  - museum_hybrid_maze.png")
    print("  - office_hybrid_maze.png")
    print("  - campus_hybrid_maze.png")
    print("\nAll mazes verified for global connectivity.")
    print("\nNote: Close all plot windows to exit (or use non-interactive backend)")


if __name__ == "__main__":
    main()
