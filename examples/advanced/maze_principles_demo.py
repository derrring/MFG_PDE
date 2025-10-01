#!/usr/bin/env python3
"""
Maze Design Principles Demonstration

This script demonstrates the key principles for designing proper mazes for MFG RL:
1. Connectivity - all cells reachable
2. Perimeter entrances/exits
3. Strategic bottlenecks
4. Multiple path options

Author: MFG_PDE Team
Date: October 2025
"""

import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class MazePrinciplesDemo:
    """Demonstrates maze design principles with clear examples."""

    def __init__(self):
        pass

    def create_connected_maze_simple(self, width: int = 21, height: int = 21) -> np.ndarray:
        """Create a simple but properly connected maze."""
        # Start with all walls
        maze = np.zeros((height, width), dtype=int)

        # Create a guaranteed connected structure
        # Start with a main corridor
        mid_row = height // 2
        mid_col = width // 2

        # Horizontal main corridor
        maze[mid_row, 1 : width - 1] = 1

        # Vertical main corridor
        maze[1 : height - 1, mid_col] = 1

        # Add some branches to create choices
        for i in range(3, height - 3, 4):
            # Left branches
            for j in range(1, mid_col):
                if j % 3 == 0:
                    maze[i, j] = 1

            # Right branches
            for j in range(mid_col + 1, width - 1):
                if j % 3 == 0:
                    maze[i, j] = 1

        # Add some vertical connections
        for j in range(3, width - 3, 4):
            for i in range(1, height - 1):
                if i % 3 == 0:
                    maze[i, j] = 1

        # Add random connections for complexity
        for _ in range(20):
            i = random.randint(1, height - 2)
            j = random.randint(1, width - 2)
            maze[i, j] = 1

        # Ensure perimeter walls except entrances
        maze[0, :] = 0
        maze[-1, :] = 0
        maze[:, 0] = 0
        maze[:, -1] = 0

        # Add entrances on different sides
        entrances = [
            (0, width // 4),  # Top
            (height - 1, 3 * width // 4),  # Bottom
            (height // 4, 0),  # Left
            (3 * height // 4, width - 1),  # Right
        ]

        for r, c in entrances:
            maze[r, c] = 1

        return maze

    def analyze_connectivity(self, maze: np.ndarray) -> dict:
        """Analyze maze connectivity."""
        height, width = maze.shape

        # Find all connected components using BFS
        visited = np.zeros_like(maze, dtype=bool)
        components = []

        def bfs_component(start_r: int, start_c: int) -> set[tuple[int, int]]:
            component = set()
            queue = deque([(start_r, start_c)])

            while queue:
                r, c = queue.popleft()
                if (r, c) in component or visited[r, c] or maze[r, c] == 0:
                    continue

                visited[r, c] = True
                component.add((r, c))

                # Check 4-connected neighbors
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        queue.append((nr, nc))

            return component

        # Find all components
        for i in range(height):
            for j in range(width):
                if maze[i, j] == 1 and not visited[i, j]:
                    component = bfs_component(i, j)
                    if component:
                        components.append(component)

        return {
            "num_components": len(components),
            "components": components,
            "largest_component_size": max(len(c) for c in components) if components else 0,
            "total_empty_cells": sum(len(c) for c in components),
            "is_connected": len(components) == 1,
        }

    def find_bottlenecks(self, maze: np.ndarray) -> list[tuple[int, int]]:
        """Find bottleneck cells (cells with few neighbors)."""
        height, width = maze.shape
        bottlenecks = []

        for i in range(height):
            for j in range(width):
                if maze[i, j] == 1:  # Empty cell
                    # Count empty neighbors
                    neighbors = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = i + dr, j + dc
                        if 0 <= nr < height and 0 <= nc < width and maze[nr, nc] == 1:
                            neighbors += 1

                    # Bottleneck if <= 2 neighbors (corridor or dead end)
                    if neighbors <= 2:
                        bottlenecks.append((i, j))

        return bottlenecks

    def find_entrances(self, maze: np.ndarray) -> list[tuple[int, int]]:
        """Find entrance/exit points on perimeter."""
        height, width = maze.shape
        entrances = []

        # Check perimeter
        for j in range(width):
            if maze[0, j] == 1:  # Top
                entrances.append((0, j))
            if maze[height - 1, j] == 1:  # Bottom
                entrances.append((height - 1, j))

        for i in range(height):
            if maze[i, 0] == 1:  # Left
                entrances.append((i, 0))
            if maze[i, width - 1] == 1:  # Right
                entrances.append((i, width - 1))

        return entrances

    def demonstrate_principles(self):
        """Demonstrate all maze design principles."""
        print("🏗️ Maze Design Principles for MFG RL")
        print("=" * 50)

        # Create example mazes
        mazes = {
            "Connected Maze": self.create_connected_maze_simple(21, 21),
            "Bad: Disconnected": self.create_bad_disconnected_maze(),
            "Bad: Poor Design": self.create_bad_no_entrances_maze(),
            "Good: Strategic": self.create_strategic_maze(),
        }

        _fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        for idx, (name, maze) in enumerate(mazes.items()):
            row = idx // 2
            col = (idx % 2) * 2

            # Original maze
            ax1 = axes[row, col]
            ax1.imshow(maze, cmap="binary", interpolation="nearest")
            ax1.set_title(f"{name}\nOriginal Layout")
            ax1.set_xticks([])
            ax1.set_yticks([])

            # Analysis
            connectivity = self.analyze_connectivity(maze)
            bottlenecks = self.find_bottlenecks(maze)
            entrances = self.find_entrances(maze)

            # Analysis visualization
            ax2 = axes[row, col + 1]
            analysis_maze = maze.copy().astype(float)

            # Mark bottlenecks
            for r, c in bottlenecks:
                analysis_maze[r, c] = 0.5

            # Mark entrances
            for r, c in entrances:
                analysis_maze[r, c] = 0.3

            colors = ["black", "red", "orange", "white"]  # Wall, Entrance, Bottleneck, Path
            cmap = ListedColormap(colors)

            ax2.imshow(analysis_maze, cmap=cmap, interpolation="nearest")
            ax2.set_title(f'Analysis\n{connectivity["num_components"]} component(s)')
            ax2.set_xticks([])
            ax2.set_yticks([])

            # Print analysis
            print(f"\n🔍 {name}:")
            print(f"   Connected: {'✅' if connectivity['is_connected'] else '❌'}")
            print(f"   Components: {connectivity['num_components']}")
            print(f"   Entrances: {len(entrances)}")
            print(f"   Bottlenecks: {len(bottlenecks)}")
            print(f"   Empty cells: {connectivity['total_empty_cells']}")

        plt.tight_layout()
        plt.suptitle("Maze Design Principles: Good vs Bad Examples", fontsize=16, y=0.98)
        plt.show()

        # Show design principles summary
        self.show_principles_summary()

    def create_bad_disconnected_maze(self) -> np.ndarray:
        """Create a bad example with disconnected components."""
        maze = np.zeros((21, 21), dtype=int)

        # Create separate disconnected regions
        # Region 1
        maze[2:8, 2:8] = 1

        # Region 2 (disconnected)
        maze[12:18, 12:18] = 1

        # Some walls in between to keep them separate
        maze[8:12, :] = 0
        maze[:, 8:12] = 0

        return maze

    def create_bad_no_entrances_maze(self) -> np.ndarray:
        """Create a connected maze with poor strategic design (too linear, no choices)."""
        maze = np.zeros((21, 21), dtype=int)

        # Create a simple linear path - connected but strategically terrible
        # This demonstrates poor design: no route choices, single bottleneck

        # Main cross-shaped corridor (ensures connectivity)
        maze[10, 1:20] = 1  # Horizontal main corridor
        maze[1:20, 10] = 1  # Vertical main corridor

        # Add a few simple linear branches (no strategic value)
        maze[5, 8:13] = 1  # Short horizontal branch
        maze[15, 8:13] = 1  # Short horizontal branch
        maze[8:13, 5] = 1  # Short vertical branch
        maze[8:13, 15] = 1  # Short vertical branch

        # Add perimeter entrances
        maze[0, 10] = 1  # Top center
        maze[20, 10] = 1  # Bottom center
        maze[10, 0] = 1  # Left center
        maze[10, 20] = 1  # Right center

        return maze

    def create_strategic_maze(self) -> np.ndarray:
        """Create a good strategic maze with bottlenecks and choices."""
        maze = np.zeros((21, 21), dtype=int)

        # Create main paths with strategic bottlenecks
        # Main corridor with branches
        maze[10, 1:20] = 1  # Main horizontal
        maze[1:20, 10] = 1  # Main vertical

        # Create strategic bottlenecks and alternate paths
        # Upper section
        maze[5, 3:8] = 1
        maze[5, 13:18] = 1
        maze[3:8, 5] = 1
        maze[13:18, 5] = 1

        # Lower section
        maze[15, 3:8] = 1
        maze[15, 13:18] = 1
        maze[3:8, 15] = 1
        maze[13:18, 15] = 1

        # Add some connecting corridors
        maze[5:16, 7] = 1
        maze[5:16, 13] = 1
        maze[7, 5:16] = 1
        maze[13, 5:16] = 1

        # Perimeter walls except entrances
        maze[0, :] = 0
        maze[-1, :] = 0
        maze[:, 0] = 0
        maze[:, -1] = 0

        # Strategic entrance placement
        entrances = [(0, 10), (20, 10), (10, 0), (10, 20)]  # One on each side, centered
        for r, c in entrances:
            maze[r, c] = 1

        return maze

    def show_principles_summary(self):
        """Show summary of design principles."""
        principles_text = """
🎯 MAZE DESIGN PRINCIPLES FOR MFG RL:

1. 🔗 CONNECTIVITY (Most Critical)
   • All empty cells must be reachable from all other empty cells
   • Single connected component only
   • No isolated regions or "islands"

2. 🚪 ENTRANCE/EXIT PLACEMENT
   • Place on perimeter walls (border of maze)
   • Random positions, not fixed corners
   • Avoid connections where internal walls meet perimeter
   • Multiple entrances for flexibility

3. 🛣️ STRATEGIC DESIGN
   • Multiple path options between any two points
   • Strategic bottlenecks for congestion dynamics
   • Varying corridor widths
   • Choice points creating decision opportunities

4. 🎮 MEAN FIELD GAME PROPERTIES
   • Bottlenecks create natural congestion points
   • Multiple routes enable strategic decisions
   • Path length vs congestion trade-offs
   • Emergent coordination requirements

5. ⚖️ COMPLEXITY BALANCE
   • Not too sparse (boring, no interactions)
   • Not too dense (impossible navigation)
   • ~40-60% empty space typically good
   • Scalable with agent population (20-200 agents)

✅ VALIDATION CHECKLIST:
□ Single connected component
□ Perimeter entrances exist
□ Multiple paths between key points
□ Strategic bottlenecks present
□ Reasonable empty space ratio
"""

        plt.figure(figsize=(12, 10))
        plt.text(
            0.05,
            0.95,
            principles_text,
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round,pad=1", "facecolor": "lightblue", "alpha": 0.8},
        )
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis("off")
        plt.title("MFG Maze Design Principles Reference", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    demo = MazePrinciplesDemo()
    demo.demonstrate_principles()
