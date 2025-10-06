#!/usr/bin/env python3
"""
Perfect Maze Generation for MFG-RL Experiments

Implements classic maze generation algorithms from "Mazes for Programmers" by Jamis Buck.
All algorithms produce perfect mazes with two critical properties:
1. Fully Connected: Path exists between any two cells
2. No Loops: Exactly one unique path between any pair of cells

Author: MFG_PDE Team
Date: October 2025
Reference: Jamis Buck, "Mazes for Programmers" (2015)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.utils.logging import get_logger

logger = get_logger(__name__)


class MazeAlgorithm(Enum):
    """Available maze generation algorithms."""

    RECURSIVE_BACKTRACKING = "recursive_backtracking"
    BINARY_TREE = "binary_tree"
    SIDEWINDER = "sidewinder"
    ALDOUS_BRODER = "aldous_broder"
    WILSONS = "wilsons"


@dataclass(frozen=False, eq=True)
class Cell:
    """Represents a cell in the maze grid."""

    row: int
    col: int
    north: bool = False  # Has passage to north
    south: bool = False  # Has passage to south
    east: bool = False  # Has passage to east
    west: bool = False  # Has passage to west
    visited: bool = False

    def __hash__(self):
        """Make cell hashable based on position only."""
        return hash((self.row, self.col))

    def __eq__(self, other):
        """Equality based on position only."""
        if not isinstance(other, Cell):
            return False
        return self.row == other.row and self.col == other.col

    def link(self, other: Cell, bidirectional: bool = True):
        """Create passage to another cell."""
        if other.row == self.row - 1:  # Other is north
            self.north = True
            if bidirectional:
                other.south = True
        elif other.row == self.row + 1:  # Other is south
            self.south = True
            if bidirectional:
                other.north = True
        elif other.col == self.col - 1:  # Other is west
            self.west = True
            if bidirectional:
                other.east = True
        elif other.col == self.col + 1:  # Other is east
            self.east = True
            if bidirectional:
                other.west = True

    def is_linked(self, other: Cell) -> bool:
        """Check if there is a passage to another cell."""
        if other.row == self.row - 1:
            return self.north
        elif other.row == self.row + 1:
            return self.south
        elif other.col == self.col - 1:
            return self.west
        elif other.col == self.col + 1:
            return self.east
        return False


class Grid:
    """Grid of cells for maze generation."""

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.cells: list[list[Cell]] = []
        self._initialize_cells()

    def _initialize_cells(self):
        """Initialize the grid with cells."""
        self.cells = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                row.append(Cell(r, c))
            self.cells.append(row)

    def get_cell(self, row: int, col: int) -> Cell | None:
        """Get cell at position."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.cells[row][col]
        return None

    def get_neighbors(self, cell: Cell) -> list[Cell]:
        """Get all neighboring cells (4-connected)."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = self.get_cell(cell.row + dr, cell.col + dc)
            if neighbor is not None:
                neighbors.append(neighbor)
        return neighbors

    def get_unvisited_neighbors(self, cell: Cell) -> list[Cell]:
        """Get unvisited neighboring cells."""
        return [n for n in self.get_neighbors(cell) if not n.visited]

    def random_cell(self) -> Cell:
        """Get a random cell."""
        row = random.randint(0, self.rows - 1)
        col = random.randint(0, self.cols - 1)
        return self.cells[row][col]

    def all_cells(self) -> list[Cell]:
        """Get all cells in the grid."""
        cells = []
        for row in self.cells:
            cells.extend(row)
        return cells

    def reset_visited(self):
        """Reset visited flags for all cells."""
        for cell in self.all_cells():
            cell.visited = False


class PerfectMazeGenerator:
    """
    Perfect maze generator using classic algorithms.

    Reference: Jamis Buck, "Mazes for Programmers" (2015)
    """

    def __init__(self, rows: int, cols: int, algorithm: MazeAlgorithm = MazeAlgorithm.RECURSIVE_BACKTRACKING):
        """
        Initialize maze generator.

        Args:
            rows: Number of rows in maze
            cols: Number of columns in maze
            algorithm: Algorithm to use for generation
        """
        self.rows = rows
        self.cols = cols
        self.algorithm = algorithm
        self.grid = Grid(rows, cols)
        self.logger = get_logger(__name__)

    def generate(self, seed: int | None = None) -> Grid:
        """
        Generate a perfect maze.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Generated maze grid
        """
        if seed is not None:
            random.seed(seed)

        self.logger.info(f"Generating {self.rows}x{self.cols} maze using {self.algorithm.value}")

        if self.algorithm == MazeAlgorithm.RECURSIVE_BACKTRACKING:
            self._recursive_backtracking()
        elif self.algorithm == MazeAlgorithm.BINARY_TREE:
            self._binary_tree()
        elif self.algorithm == MazeAlgorithm.SIDEWINDER:
            self._sidewinder()
        elif self.algorithm == MazeAlgorithm.ALDOUS_BRODER:
            self._aldous_broder()
        elif self.algorithm == MazeAlgorithm.WILSONS:
            self._wilsons()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        self.logger.info("Maze generation complete")
        return self.grid

    def _recursive_backtracking(self):
        """
        Recursive Backtracking (Depth-First Search) algorithm.

        Creates mazes with long, winding passages and few dead ends.
        Ideal for RL experiments requiring complex exploration.
        """
        stack = []
        start_cell = self.grid.random_cell()
        start_cell.visited = True
        stack.append(start_cell)

        while stack:
            current = stack[-1]
            unvisited = self.grid.get_unvisited_neighbors(current)

            if unvisited:
                # Choose random unvisited neighbor
                neighbor = random.choice(unvisited)

                # Link cells (create passage)
                current.link(neighbor)

                # Mark as visited and add to stack
                neighbor.visited = True
                stack.append(neighbor)
            else:
                # Backtrack
                stack.pop()

    def _binary_tree(self):
        """
        Binary Tree algorithm.

        Fast but creates strong diagonal bias (northwest).
        North and west walls will be completely open corridors.
        """
        for row in self.grid.cells:
            for cell in row:
                neighbors = []

                # Only consider north and west
                north = self.grid.get_cell(cell.row - 1, cell.col)
                west = self.grid.get_cell(cell.row, cell.col - 1)

                if north is not None:
                    neighbors.append(north)
                if west is not None:
                    neighbors.append(west)

                if neighbors:
                    neighbor = random.choice(neighbors)
                    cell.link(neighbor)

    def _sidewinder(self):
        """
        Sidewinder algorithm.

        Processes row by row. Creates bias toward top row (open corridor)
        but solves diagonal bias of Binary Tree.
        """
        for row in self.grid.cells:
            run = []

            for cell in row:
                run.append(cell)

                at_eastern_boundary = cell.col == self.grid.cols - 1
                at_northern_boundary = cell.row == 0

                should_close_out = at_eastern_boundary or (not at_northern_boundary and random.choice([True, False]))

                if should_close_out:
                    # Pick random cell from run and carve north
                    member = random.choice(run)
                    if member.row > 0:
                        north = self.grid.get_cell(member.row - 1, member.col)
                        if north is not None:
                            member.link(north)
                    run = []
                else:
                    # Carve east
                    east = self.grid.get_cell(cell.row, cell.col + 1)
                    if east is not None:
                        cell.link(east)

    def _aldous_broder(self):
        """
        Aldous-Broder algorithm.

        Unbiased random walk. Produces truly unbiased mazes but can be slow.
        Every possible maze has equal probability of being generated.
        """
        cell = self.grid.random_cell()
        cell.visited = True
        unvisited_count = self.rows * self.cols - 1

        while unvisited_count > 0:
            neighbors = self.grid.get_neighbors(cell)
            neighbor = random.choice(neighbors)

            if not neighbor.visited:
                cell.link(neighbor)
                neighbor.visited = True
                unvisited_count -= 1

            cell = neighbor

    def _wilsons(self):
        """
        Wilson's algorithm.

        Loop-erased random walk. Unbiased like Aldous-Broder but much faster.
        Generally the preferred algorithm for unbiased mazes.
        """
        # Mark one cell as part of maze
        unvisited = self.grid.all_cells()
        first = random.choice(unvisited)
        first.visited = True
        unvisited.remove(first)

        while unvisited:
            # Start random walk from random unvisited cell
            cell = random.choice(unvisited)
            path = [cell]

            # Random walk until we hit the maze
            while not cell.visited:
                cell = random.choice(self.grid.get_neighbors(cell))

                # Loop erasure
                if cell in path:
                    # Erase loop
                    position = path.index(cell)
                    path = path[: position + 1]
                else:
                    path.append(cell)

            # Carve the path into the maze
            for i in range(len(path) - 1):
                path[i].link(path[i + 1])
                path[i].visited = True
                if path[i] in unvisited:
                    unvisited.remove(path[i])

    def to_numpy_array(self, wall_thickness: int = 1) -> np.ndarray:
        """
        Convert maze to numpy array.

        Args:
            wall_thickness: Thickness of walls in cells

        Returns:
            Numpy array where 1 = wall, 0 = passage
        """
        # Each cell becomes (2*wall_thickness + 1) x (2*wall_thickness + 1) array
        cell_size = 2 * wall_thickness + 1
        height = self.rows * cell_size + wall_thickness
        width = self.cols * cell_size + wall_thickness

        # Initialize with all walls
        maze = np.ones((height, width), dtype=int)

        # Carve passages
        for row in self.grid.cells:
            for cell in row:
                # Carve cell center
                r_start = cell.row * cell_size + wall_thickness
                c_start = cell.col * cell_size + wall_thickness

                maze[r_start : r_start + wall_thickness, c_start : c_start + wall_thickness] = 0

                # Carve passages based on links
                if cell.north:
                    maze[r_start - wall_thickness : r_start, c_start : c_start + wall_thickness] = 0
                if cell.south:
                    maze[
                        r_start + wall_thickness : r_start + 2 * wall_thickness, c_start : c_start + wall_thickness
                    ] = 0
                if cell.west:
                    maze[r_start : r_start + wall_thickness, c_start - wall_thickness : c_start] = 0
                if cell.east:
                    maze[
                        r_start : r_start + wall_thickness, c_start + wall_thickness : c_start + 2 * wall_thickness
                    ] = 0

        return maze

    def visualize(self, title: str | None = None, figsize: tuple[int, int] = (10, 10)):
        """Visualize the generated maze."""
        maze_array = self.to_numpy_array()

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(maze_array, cmap="binary", interpolation="nearest")

        if title is None:
            title = f"Perfect Maze: {self.algorithm.value}"
        ax.set_title(title, fontsize=14, fontweight="bold")

        ax.set_xticks([])
        ax.set_yticks([])

        # Add algorithm info
        info_text = f"Size: {self.rows}x{self.cols}\nAlgorithm: {self.algorithm.value}"
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        plt.tight_layout()
        return fig, ax


def compare_algorithms(rows: int = 15, cols: int = 15, seed: int = 42):
    """Compare different maze generation algorithms."""
    algorithms = [
        MazeAlgorithm.RECURSIVE_BACKTRACKING,
        MazeAlgorithm.BINARY_TREE,
        MazeAlgorithm.SIDEWINDER,
        MazeAlgorithm.WILSONS,
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()

    for i, algorithm in enumerate(algorithms):
        generator = PerfectMazeGenerator(rows, cols, algorithm)
        generator.generate(seed=seed)
        maze_array = generator.to_numpy_array()

        axes[i].imshow(maze_array, cmap="binary", interpolation="nearest")
        axes[i].set_title(f"{algorithm.value.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    fig.suptitle("Comparison of Perfect Maze Algorithms", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def verify_perfect_maze(grid: Grid) -> dict:
    """
    Verify that a maze is perfect (fully connected, no loops).

    Args:
        grid: Maze grid to verify

    Returns:
        Dictionary with verification results
    """
    all_cells = grid.all_cells()

    # Check connectivity via BFS
    grid.reset_visited()
    start = all_cells[0]
    queue = [start]
    start.visited = True
    visited_count = 1

    while queue:
        current = queue.pop(0)
        for neighbor in grid.get_neighbors(current):
            if not neighbor.visited and current.is_linked(neighbor):
                neighbor.visited = True
                visited_count += 1
                queue.append(neighbor)

    total_cells = len(all_cells)
    is_connected = visited_count == total_cells

    # Count passages
    passage_count = 0
    for cell in all_cells:
        if cell.north:
            passage_count += 1
        if cell.east:
            passage_count += 1

    # Perfect maze has exactly (n-1) passages for n cells
    expected_passages = total_cells - 1
    is_no_loops = passage_count == expected_passages

    return {
        "is_perfect": is_connected and is_no_loops,
        "is_connected": is_connected,
        "is_no_loops": is_no_loops,
        "visited_cells": visited_count,
        "total_cells": total_cells,
        "passage_count": passage_count,
        "expected_passages": expected_passages,
    }


def demo_perfect_mazes():
    """Demonstrate perfect maze generation."""
    print("Perfect Maze Generation Demo")
    print("=" * 60)

    # Test each algorithm
    for algorithm in MazeAlgorithm:
        print(f"\nTesting {algorithm.value}...")

        generator = PerfectMazeGenerator(20, 20, algorithm)
        grid = generator.generate(seed=42)

        # Verify maze is perfect
        verification = verify_perfect_maze(grid)

        print(f"  Is perfect: {verification['is_perfect']}")
        print(f"  Is connected: {verification['is_connected']}")
        print(f"  Is loop-free: {verification['is_no_loops']}")
        print(f"  Passages: {verification['passage_count']}/{verification['expected_passages']}")

        if not verification["is_perfect"]:
            print("  WARNING: Maze is not perfect!")

    print("\n" + "=" * 60)
    print("Visualizing algorithm comparison...")
    compare_algorithms(rows=15, cols=15, seed=42)


if __name__ == "__main__":
    demo_perfect_mazes()
