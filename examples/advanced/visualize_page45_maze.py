#!/usr/bin/env python3
"""
Quick visualization of the Page 45 maze layout using matplotlib.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from mfg_maze_layouts import CustomMazeLoader, MazeAnalyzer


def visualize_page45_maze():
    """Visualize the page 45 maze layout."""
    print("🎨 Visualizing Page 45 Maze Layout")

    # Load the maze
    loader = CustomMazeLoader()
    maze = loader.get_predefined_layout("paper_page45")

    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Basic maze layout
    ax1.imshow(maze, cmap="binary", interpolation="nearest")
    ax1.set_title("Page 45 Maze Layout\n(Black = Walls, White = Paths)", fontsize=14, fontweight="bold")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Add grid for better visibility
    height, width = maze.shape
    for i in range(height + 1):
        ax1.axhline(i - 0.5, color="gray", linewidth=0.2, alpha=0.3)
    for j in range(width + 1):
        ax1.axvline(j - 0.5, color="gray", linewidth=0.2, alpha=0.3)

    # 2. Maze with custom colors
    # Create a colorful version
    colored_maze = maze.copy().astype(float)
    colors = ["black", "lightblue"]  # Wall, Empty
    cmap = ListedColormap(colors)

    ax2.imshow(colored_maze, cmap=cmap, interpolation="nearest")
    ax2.set_title("Colored Maze Layout\n(Black = Walls, Blue = Navigable Paths)", fontsize=14, fontweight="bold")
    ax2.set_xticks([])
    ax2.set_yticks([])

    # 3. Maze analysis - bottlenecks
    analyzer = MazeAnalyzer(maze)
    bottlenecks_info = analyzer.analyze_bottlenecks()

    # Create bottleneck visualization
    bottleneck_maze = maze.copy().astype(float)

    # Mark regular bottlenecks
    for bottleneck in bottlenecks_info["bottlenecks"]:
        bottleneck_maze[bottleneck] = 0.5

    # Mark critical bottlenecks
    for critical in bottlenecks_info["critical_bottlenecks"]:
        bottleneck_maze[critical] = 0.2

    bottleneck_colors = ["black", "white", "orange", "red"]  # Wall, Empty, Bottleneck, Critical
    bottleneck_cmap = ListedColormap(bottleneck_colors)

    ax3.imshow(bottleneck_maze, cmap=bottleneck_cmap, interpolation="nearest")
    ax3.set_title(
        "Bottleneck Analysis\n(Orange = Bottlenecks, Red = Critical Bottlenecks)", fontsize=14, fontweight="bold"
    )
    ax3.set_xticks([])
    ax3.set_yticks([])

    # 4. Statistics and path analysis
    stats = analyzer.get_maze_statistics()

    # Create text summary
    stats_text = f"""
Page 45 Maze Statistics:

Dimensions: {stats['dimensions'][0]} × {stats['dimensions'][1]}
Total Cells: {stats['total_cells']}
Empty Cells: {stats['empty_cells']}
Wall Density: {stats['wall_density']:.1%}

Connectivity Analysis:
• Connected Components: {stats['connectivity']['connected_components']}
• Largest Component: {stats['connectivity']['largest_component_size']} cells
• Fully Connected: {'Yes' if stats['connectivity']['is_fully_connected'] else 'No'}

Bottleneck Analysis:
• Total Bottlenecks: {len(stats['bottlenecks']['bottlenecks'])}
• Critical Bottlenecks: {len(stats['bottlenecks']['critical_bottlenecks'])}
• Bottleneck Density: {stats['bottlenecks']['bottleneck_density']:.1%}

Mean Field Game Properties:
• High congestion expected at bottlenecks
• Strategic path choice important
• Natural mean field interactions
• Suitable for multi-agent navigation
"""

    ax4.text(
        0.05,
        0.95,
        stats_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgray", "alpha": 0.8},
    )
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis("off")

    # Overall title
    fig.suptitle("Page 45 Paper-Style Maze for Mean Field Games RL", fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    # Print maze dimensions
    print(f"📐 Maze Dimensions: {height} × {width}")
    print(f"🧱 Wall Cells: {np.sum(maze == 0)}")
    print(f"🚶 Empty Cells: {np.sum(maze == 1)}")
    print(f"📊 Wall Density: {stats['wall_density']:.1%}")

    plt.show()

    return maze


def print_ascii_maze():
    """Print the ASCII representation of the maze."""
    print("\n📄 ASCII Representation of Page 45 Maze:")
    print("=" * 50)

    loader = CustomMazeLoader()
    maze_ascii = loader.predefined_layouts["paper_page45"]

    print(maze_ascii)
    print("=" * 50)
    print("Legend: # = Wall, . = Empty path")


if __name__ == "__main__":
    print("🎮 Page 45 Maze Visualization")
    print("=" * 40)

    # Print ASCII version
    print_ascii_maze()

    # Show matplotlib visualization
    maze = visualize_page45_maze()

    print("\n✅ Visualization complete!")
    print("💡 This maze layout is perfect for testing mean field interactions")
    print("   through spatial congestion and bottleneck effects.")
