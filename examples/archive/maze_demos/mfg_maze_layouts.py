#!/usr/bin/env python3
"""
Custom Maze Layouts for MFG RL Experiments

This module provides predefined maze layouts and utilities for loading custom
labyrinths, including configurations similar to those used in research papers.

Features:
- Predefined research-grade maze layouts
- ASCII art maze loader
- Page 45 paper-style labyrinth configurations
- Maze analysis and validation tools
- Export/import functionality

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

from pathlib import Path

from mfg_maze_environment import CellType, MazeConfig, MFGMazeEnvironment

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.utils.logging import get_logger

logger = get_logger(__name__)


class CustomMazeLoader:
    """Loads and manages custom maze layouts."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.predefined_layouts = self._load_predefined_layouts()

    def load_maze_from_ascii(self, ascii_layout: str) -> np.ndarray:
        """
        Load maze from ASCII art representation.

        ASCII Format:
        '#' = Wall
        ' ' or '.' = Empty space
        'S' = Start position
        'G' = Goal position

        Args:
            ascii_layout: Multi-line string representing the maze

        Returns:
            Maze array with CellType values
        """
        lines = ascii_layout.strip().split("\n")
        lines = [line.rstrip() for line in lines if line.strip()]

        if not lines:
            raise ValueError("Empty maze layout")

        # Determine dimensions
        height = len(lines)
        width = max(len(line) for line in lines)

        # Initialize maze
        maze = np.full((height, width), CellType.EMPTY.value, dtype=int)

        # Character mapping
        char_map = {
            "#": CellType.WALL.value,
            " ": CellType.EMPTY.value,
            ".": CellType.EMPTY.value,
            "S": CellType.START.value,
            "G": CellType.GOAL.value,
        }

        # Fill maze
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                # Default to empty for unknown characters
                maze[i, j] = char_map.get(char, CellType.EMPTY.value)

        self.logger.info(f"Loaded maze from ASCII: {height}x{width}")
        return maze

    def load_maze_from_file(self, filepath: str | Path) -> np.ndarray:
        """Load maze from text file."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Maze file not found: {filepath}")

        with open(filepath) as f:
            ascii_layout = f.read()

        return self.load_maze_from_ascii(ascii_layout)

    def save_maze_to_file(self, maze: np.ndarray, filepath: str | Path):
        """Save maze to ASCII text file."""
        filepath = Path(filepath)

        # Character mapping
        char_map = {
            CellType.WALL.value: "#",
            CellType.EMPTY.value: ".",
            CellType.START.value: "S",
            CellType.GOAL.value: "G",
        }

        lines = []
        for row in maze:
            line = "".join(char_map.get(cell, ".") for cell in row)
            lines.append(line)

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        self.logger.info(f"Saved maze to: {filepath}")

    def get_predefined_layout(self, layout_name: str) -> np.ndarray:
        """Get a predefined maze layout."""
        if layout_name not in self.predefined_layouts:
            available = list(self.predefined_layouts.keys())
            raise ValueError(f"Unknown layout: {layout_name}. Available: {available}")

        return self.load_maze_from_ascii(self.predefined_layouts[layout_name])

    def list_predefined_layouts(self) -> list[str]:
        """List available predefined layouts."""
        return list(self.predefined_layouts.keys())

    def _load_predefined_layouts(self) -> dict[str, str]:
        """Load predefined maze layouts."""
        layouts = {}

        # Paper-style labyrinth (Page 45 inspired)
        layouts["paper_page45"] = """
#####################
#...........#.......#
#.#########.#.#####.#
#.#.......#...#...#.#
#.#.#####.#####.#.#.#
#.#.#...#.......#.#.#
#.#.#.#.#########.#.#
#.#.#.#...........#.#
#.#.#.#############.#
#.#.#...............#
#.#.#################
#.#.................#
#.#.#################
#...#...............#
#####.#.#############
#.....#.............#
#.#####.#############
#.......#...........#
#########.#.#########
#.........#.........#
#####################
"""

        # Classic research maze
        layouts["research_classic"] = """
#################
#...#.........#.#
#.#.#.#######.#.#
#.#...#.....#...#
#.#####.###.###.#
#.......#.#.....#
#########.#.#####
#.........#.....#
#.#############.#
#.#...........#.#
#.#.#########.#.#
#.#.#.......#.#.#
#.#.#.#####.#.#.#
#.#...#...#...#.#
#.#####.#.#####.#
#.......#.......#
#################
"""

        # Corridor maze (bottleneck focused)
        layouts["corridor_bottleneck"] = """
###################
#.................#
#.###############.#
#.#.............#.#
#.#.###########.#.#
#.#.#.........#.#.#
#.#.#.#######.#.#.#
#.#.#.......#.#.#.#
#.#.#######.#.#.#.#
#.#.........#.#.#.#
#.###########.#.#.#
#.............#.#.#
###############.#.#
#...............#.#
#.###############.#
#.................#
###################
"""

        # Cross-shaped maze
        layouts["cross_intersection"] = """
#####################
#.........#.........#
#.#######.#.#######.#
#.#.....#.#.#.....#.#
#.#.###.#.#.#.###.#.#
#.#.#.#.#.#.#.#.#.#.#
#.#.#.#.#.#.#.#.#.#.#
#.#.#.#.......#.#.#.#
#.#.#.#########.#.#.#
#.#.#...........#.#.#
#.#.#############.#.#
#.#...............#.#
#.#.#############.#.#
#.#.#...........#.#.#
#.#.#.#########.#.#.#
#.#.#.......#.#.#.#.#
#.#.#.#.#.#.#.#.#.#.#
#.#.###.#.#.#.###.#.#
#.#.....#.#.#.....#.#
#.#######.#.#######.#
#####################
"""

        # U-shaped maze
        layouts["u_shape"] = """
#############
#...........#
#.#########.#
#.#.......#.#
#.#.#####.#.#
#.#.#...#.#.#
#.#.#.#.#.#.#
#.#.#.#.#.#.#
#.#.#.#.#.#.#
#.#.#.#.#.#.#
#.#.#...#.#.#
#.#.#####.#.#
#.#.......#.#
#.#########.#
#...........#
#############
"""

        # Simple test maze
        layouts["simple_test"] = """
###########
#.........#
#.#######.#
#.#.....#.#
#.#.###.#.#
#.#...#.#.#
#.###.#.#.#
#.....#...#
#.#######.#
#.........#
###########
"""

        # Large open area with strategic walls
        layouts["open_strategic"] = """
#########################
#.......................#
#.#####################.#
#.#...................#.#
#.#.#################.#.#
#.#.#...............#.#.#
#.#.#.#############.#.#.#
#.#.#.#...........#.#.#.#
#.#.#.#.#########.#.#.#.#
#.#.#.#.#.......#.#.#.#.#
#.#.#.#.#.#####.#.#.#.#.#
#.#.#.#.#.....#.#.#.#.#.#
#.#.#.#.#####.#.#.#.#.#.#
#.#.#.#.......#.#.#.#.#.#
#.#.#.#########.#.#.#.#.#
#.#.#...........#.#.#.#.#
#.#.#############.#.#.#.#
#.#...............#.#.#.#
#.#################.#.#.#
#...................#.#.#
#.#####################.#
#.......................#
#########################
"""

        return layouts


class MazeAnalyzer:
    """Analyzes maze properties for MFG experiments."""

    def __init__(self, maze: np.ndarray):
        self.maze = maze
        self.height, self.width = maze.shape
        self.logger = get_logger(__name__)

    def analyze_connectivity(self) -> dict:
        """Analyze maze connectivity properties."""
        empty_cells = self._get_empty_cells()
        connected_components = self._find_connected_components(empty_cells)

        return {
            "total_empty_cells": len(empty_cells),
            "connected_components": len(connected_components),
            "largest_component_size": max(len(comp) for comp in connected_components) if connected_components else 0,
            "is_fully_connected": len(connected_components) == 1,
        }

    def analyze_bottlenecks(self) -> dict:
        """Identify bottlenecks in the maze."""
        bottlenecks = []
        empty_cells = self._get_empty_cells()

        for cell in empty_cells:
            neighbors = self._get_empty_neighbors(cell)
            if len(neighbors) <= 2:  # Potential bottleneck
                bottlenecks.append(cell)

        # Find critical bottlenecks (removal disconnects the maze)
        critical_bottlenecks = []
        for bottleneck in bottlenecks:
            if self._is_critical_bottleneck(bottleneck, empty_cells):
                critical_bottlenecks.append(bottleneck)

        return {
            "bottlenecks": bottlenecks,
            "critical_bottlenecks": critical_bottlenecks,
            "bottleneck_density": len(bottlenecks) / len(empty_cells) if empty_cells else 0,
        }

    def compute_path_metrics(self, start: tuple[int, int], goal: tuple[int, int]) -> dict:
        """Compute path metrics between start and goal."""
        shortest_path = self._find_shortest_path(start, goal)

        if not shortest_path:
            return {
                "shortest_path_length": float("inf"),
                "path_exists": False,
                "manhattan_distance": abs(start[0] - goal[0]) + abs(start[1] - goal[1]),
                "detour_ratio": float("inf"),
            }

        manhattan_dist = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        path_length = len(shortest_path) - 1

        return {
            "shortest_path_length": path_length,
            "path_exists": True,
            "manhattan_distance": manhattan_dist,
            "detour_ratio": path_length / manhattan_dist if manhattan_dist > 0 else 1.0,
            "shortest_path": shortest_path,
        }

    def get_maze_statistics(self) -> dict:
        """Get comprehensive maze statistics."""
        total_cells = self.height * self.width
        wall_cells = np.sum(self.maze == CellType.WALL.value)
        empty_cells = np.sum(self.maze == CellType.EMPTY.value)

        connectivity = self.analyze_connectivity()
        bottlenecks = self.analyze_bottlenecks()

        return {
            "dimensions": (self.height, self.width),
            "total_cells": total_cells,
            "wall_cells": wall_cells,
            "empty_cells": empty_cells,
            "wall_density": wall_cells / total_cells,
            "empty_density": empty_cells / total_cells,
            "connectivity": connectivity,
            "bottlenecks": bottlenecks,
        }

    def _get_empty_cells(self) -> list[tuple[int, int]]:
        """Get list of empty cell coordinates."""
        empty_cells = []
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i, j] != CellType.WALL.value:
                    empty_cells.append((i, j))
        return empty_cells

    def _get_empty_neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        """Get empty neighboring cells."""
        i, j = cell
        neighbors = []

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.height and 0 <= nj < self.width and self.maze[ni, nj] != CellType.WALL.value:
                neighbors.append((ni, nj))

        return neighbors

    def _find_connected_components(self, empty_cells: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
        """Find connected components using DFS."""
        visited = set()
        components = []
        empty_set = set(empty_cells)

        for cell in empty_cells:
            if cell not in visited:
                component = []
                stack = [cell]

                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)

                        neighbors = self._get_empty_neighbors(current)
                        for neighbor in neighbors:
                            if neighbor in empty_set and neighbor not in visited:
                                stack.append(neighbor)

                components.append(component)

        return components

    def _is_critical_bottleneck(self, cell: tuple[int, int], empty_cells: list[tuple[int, int]]) -> bool:
        """Check if removing a cell disconnects the maze."""
        # Create temporary maze without this cell
        temp_empty = [c for c in empty_cells if c != cell]
        components = self._find_connected_components(temp_empty)
        return len(components) > 1

    def _find_shortest_path(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        """Find shortest path using BFS."""
        if self.maze[start[0], start[1]] == CellType.WALL.value or self.maze[goal[0], goal[1]] == CellType.WALL.value:
            return []

        from collections import deque

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()

            if current == goal:
                return path

            for neighbor in self._get_empty_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, [*path, neighbor]))

        return []  # No path found


def create_custom_maze_environment(layout_name: str, num_agents: int = 50, max_steps: int = 500) -> MFGMazeEnvironment:
    """
    Create maze environment with custom layout.

    Args:
        layout_name: Name of predefined layout or path to maze file
        num_agents: Number of agents
        max_steps: Maximum episode steps

    Returns:
        Configured MFGMazeEnvironment
    """
    loader = CustomMazeLoader()

    # Try to load as predefined layout first
    try:
        maze = loader.get_predefined_layout(layout_name)
    except ValueError:
        # Try to load as file
        try:
            maze = loader.load_maze_from_file(layout_name)
        except FileNotFoundError as exc:
            raise ValueError(f"Layout not found: {layout_name}") from exc

    # Create configuration
    height, width = maze.shape
    config = MazeConfig(
        width=width, height=height, num_agents=num_agents, max_episode_steps=max_steps, maze_type="custom"
    )

    # Create environment
    env = MFGMazeEnvironment(config)

    # Override the generated maze with custom layout
    env.maze = maze
    env._setup_start_goal_positions()

    return env


def visualize_maze_analysis(maze: np.ndarray, analysis: dict | None = None):
    """Visualize maze with analysis overlays."""
    if analysis is None:
        analyzer = MazeAnalyzer(maze)
        analysis = analyzer.get_maze_statistics()

    _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Original maze
    ax1.imshow(maze, cmap="binary", interpolation="nearest")
    ax1.set_title("Original Maze")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Connectivity visualization
    analyzer = MazeAnalyzer(maze)
    empty_cells = analyzer._get_empty_cells()
    components = analyzer._find_connected_components(empty_cells)

    component_maze = np.copy(maze).astype(float)
    for i, component in enumerate(components):
        for cell in component:
            component_maze[cell] = i + 2

    ax2.imshow(component_maze, cmap="tab10", interpolation="nearest")
    ax2.set_title(f"Connected Components ({len(components)})")
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Bottlenecks visualization
    bottlenecks_info = analyzer.analyze_bottlenecks()
    bottleneck_maze = np.copy(maze).astype(float)

    for bottleneck in bottlenecks_info["bottlenecks"]:
        bottleneck_maze[bottleneck] = 2

    for critical in bottlenecks_info["critical_bottlenecks"]:
        bottleneck_maze[critical] = 3

    ax3.imshow(bottleneck_maze, cmap="RdYlBu_r", interpolation="nearest")
    ax3.set_title(f"Bottlenecks (Critical: {len(bottlenecks_info['critical_bottlenecks'])})")
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Statistics text
    stats_text = f"""
Maze Statistics:
Dimensions: {analysis['dimensions']}
Total cells: {analysis['total_cells']}
Wall density: {analysis['wall_density']:.2f}
Empty cells: {analysis['empty_cells']}

Connectivity:
Components: {analysis['connectivity']['connected_components']}
Largest component: {analysis['connectivity']['largest_component_size']}
Fully connected: {analysis['connectivity']['is_fully_connected']}

Bottlenecks:
Total: {len(analysis['bottlenecks']['bottlenecks'])}
Critical: {len(analysis['bottlenecks']['critical_bottlenecks'])}
Density: {analysis['bottlenecks']['bottleneck_density']:.3f}
"""

    ax4.text(
        0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10, verticalalignment="top", fontfamily="monospace"
    )
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis("off")

    plt.tight_layout()
    plt.show()


def demo_custom_layouts():
    """Demonstrate custom maze layouts and analysis."""
    print("🏗️  Custom Maze Layouts Demo")
    print("=" * 40)

    loader = CustomMazeLoader()
    available_layouts = loader.list_predefined_layouts()

    print(f"Available layouts: {available_layouts}")

    # Test each layout
    for layout_name in available_layouts:
        print(f"\n🎯 Testing layout: {layout_name}")

        try:
            # Load maze
            maze = loader.get_predefined_layout(layout_name)

            # Analyze maze
            analyzer = MazeAnalyzer(maze)
            stats = analyzer.get_maze_statistics()

            print(f"   Dimensions: {stats['dimensions']}")
            print(f"   Wall density: {stats['wall_density']:.2f}")
            print(f"   Connected components: {stats['connectivity']['connected_components']}")
            print(f"   Bottlenecks: {len(stats['bottlenecks']['bottlenecks'])}")

            # Test path finding
            empty_cells = analyzer._get_empty_cells()
            if len(empty_cells) >= 2:
                start = empty_cells[0]
                goal = empty_cells[-1]
                path_metrics = analyzer.compute_path_metrics(start, goal)
                print(f"   Path length: {path_metrics['shortest_path_length']}")
                print(f"   Detour ratio: {path_metrics['detour_ratio']:.2f}")

            # Create environment
            env = create_custom_maze_environment(layout_name, num_agents=20, max_steps=100)

            # Run quick test
            _obs = env.reset()
            for _ in range(5):
                actions = np.random.randint(0, 5, size=20)
                _obs, _rewards, _done, _info = env.step(actions)

            print("   Environment test passed ✓")

        except Exception as e:
            print(f"   Error testing {layout_name}: {e}")

    print("\n🎨 Layout Analysis Demo")
    print("Testing 'paper_page45' layout...")

    # Detailed analysis of paper layout
    paper_maze = loader.get_predefined_layout("paper_page45")
    visualize_maze_analysis(paper_maze)


if __name__ == "__main__":
    demo_custom_layouts()
