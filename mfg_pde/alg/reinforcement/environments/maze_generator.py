"""
Perfect Maze Generation for MFG-RL Environments

Implements classic maze generation algorithms for creating guaranteed-solvable
maze environments for reinforcement learning experiments.

All algorithms produce perfect mazes with two critical properties:
1. Fully Connected: Path exists between any two cells
2. No Loops: Exactly one unique path between any pair of cells

Implemented Algorithms:
- Recursive Backtracking (DFS): Long winding paths, high exploration difficulty
- Wilson's Algorithm: Unbiased sampling, structural diversity

Mathematical Foundation:
Perfect mazes are minimal spanning trees on grid graphs, ensuring:
- Connectivity: |V| vertices connected by |V|-1 edges
- Acyclicity: No loops in the graph structure
- Uniqueness: Exactly one path between any two vertices

Reference: Jamis Buck, "Mazes for Programmers" (2015)

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

import numpy as np


class MazeAlgorithm(Enum):
    """Available perfect maze generation algorithms."""

    RECURSIVE_BACKTRACKING = "recursive_backtracking"
    WILSONS = "wilsons"


@dataclass(frozen=False, eq=True)
class Cell:
    """
    Represents a cell in the maze grid.

    Attributes:
        row: Row index in grid
        col: Column index in grid
        north: Passage exists to north neighbor
        south: Passage exists to south neighbor
        east: Passage exists to east neighbor
        west: Passage exists to west neighbor
        visited: Temporary flag for generation algorithms
    """

    row: int
    col: int
    north: bool = False
    south: bool = False
    east: bool = False
    west: bool = False
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
        """
        Create passage to another cell.

        Args:
            other: Neighboring cell to link to
            bidirectional: Whether to create bidirectional link
        """
        if other.row == self.row - 1:
            self.north = True
            if bidirectional:
                other.south = True
        elif other.row == self.row + 1:
            self.south = True
            if bidirectional:
                other.north = True
        elif other.col == self.col - 1:
            self.west = True
            if bidirectional:
                other.east = True
        elif other.col == self.col + 1:
            self.east = True
            if bidirectional:
                other.west = True


class Grid:
    """Grid of cells for maze generation."""

    def __init__(self, rows: int, cols: int):
        """
        Initialize grid.

        Args:
            rows: Number of rows
            cols: Number of columns
        """
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
        """
        Get cell at position.

        Args:
            row: Row index
            col: Column index

        Returns:
            Cell if valid position, None otherwise
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.cells[row][col]
        return None

    def get_neighbors(self, cell: Cell) -> list[Cell]:
        """
        Get all neighboring cells (4-connected).

        Args:
            cell: Cell to get neighbors for

        Returns:
            List of neighboring cells
        """
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = self.get_cell(cell.row + dr, cell.col + dc)
            if neighbor is not None:
                neighbors.append(neighbor)
        return neighbors

    def get_unvisited_neighbors(self, cell: Cell) -> list[Cell]:
        """
        Get unvisited neighboring cells.

        Args:
            cell: Cell to get unvisited neighbors for

        Returns:
            List of unvisited neighbors
        """
        return [n for n in self.get_neighbors(cell) if not n.visited]

    def random_cell(self) -> Cell:
        """Get a random cell from the grid."""
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

    Generates mazes that are minimal spanning trees on grid graphs,
    guaranteeing connectivity and acyclicity.

    Algorithms:
    - Recursive Backtracking: DFS-based, creates long winding paths
    - Wilson's: Loop-erased random walk, unbiased sampling

    Reference: Jamis Buck, "Mazes for Programmers" (2015)
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        algorithm: MazeAlgorithm = MazeAlgorithm.RECURSIVE_BACKTRACKING,
    ):
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

        if self.algorithm == MazeAlgorithm.RECURSIVE_BACKTRACKING:
            self._recursive_backtracking()
        elif self.algorithm == MazeAlgorithm.WILSONS:
            self._wilsons()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return self.grid

    def _recursive_backtracking(self):
        """
        Recursive Backtracking (Depth-First Search) algorithm.

        Creates mazes with long, winding passages and few dead ends.
        Ideal for RL experiments requiring complex exploration.

        Algorithm:
        1. Start at random cell, mark as visited
        2. While unvisited neighbors exist:
           - Choose random unvisited neighbor
           - Link cells (create passage)
           - Move to neighbor, add to stack
        3. Backtrack when stuck (pop stack)

        Characteristics:
        - Fast: O(n) where n = number of cells
        - Biased toward long corridors
        - Few dead ends (10-15% of cells)
        - High exploration difficulty
        """
        stack = []
        start_cell = self.grid.random_cell()
        start_cell.visited = True
        stack.append(start_cell)

        while stack:
            current = stack[-1]
            unvisited = self.grid.get_unvisited_neighbors(current)

            if unvisited:
                neighbor = random.choice(unvisited)
                current.link(neighbor)
                neighbor.visited = True
                stack.append(neighbor)
            else:
                stack.pop()

    def _wilsons(self):
        """
        Wilson's algorithm using loop-erased random walks.

        Produces truly unbiased mazes where every possible maze
        for a given grid size has equal probability of being generated.

        Algorithm:
        1. Mark one random cell as part of maze
        2. For each unvisited cell:
           - Perform loop-erased random walk until hitting maze
           - Add discovered path to maze
        3. Repeat until all cells visited

        Characteristics:
        - Unbiased: All mazes equally likely
        - Moderate speed: O(n log n) expected
        - Higher dead-end count (25-30% of cells)
        - More decision points (junctions)
        - Greater structural diversity
        """
        unvisited = self.grid.all_cells()
        first = random.choice(unvisited)
        first.visited = True
        unvisited.remove(first)

        while unvisited:
            cell = random.choice(unvisited)
            path = [cell]

            while not cell.visited:
                cell = random.choice(self.grid.get_neighbors(cell))

                if cell in path:
                    position = path.index(cell)
                    path = path[: position + 1]
                else:
                    path.append(cell)

            for i in range(len(path) - 1):
                path[i].link(path[i + 1])
                path[i].visited = True
                if path[i] in unvisited:
                    unvisited.remove(path[i])

    def to_numpy_array(self, wall_thickness: int = 1) -> np.ndarray:
        """
        Convert maze to numpy array representation.

        Args:
            wall_thickness: Thickness of walls in cells

        Returns:
            Numpy array where 1 = wall, 0 = passage
        """
        cell_size = 2 * wall_thickness + 1
        height = self.rows * cell_size + wall_thickness
        width = self.cols * cell_size + wall_thickness

        maze = np.ones((height, width), dtype=np.int32)

        for row in self.grid.cells:
            for cell in row:
                r_start = cell.row * cell_size + wall_thickness
                c_start = cell.col * cell_size + wall_thickness

                maze[r_start : r_start + wall_thickness, c_start : c_start + wall_thickness] = 0

                if cell.north:
                    maze[
                        r_start - wall_thickness : r_start,
                        c_start : c_start + wall_thickness,
                    ] = 0
                if cell.south:
                    maze[
                        r_start + wall_thickness : r_start + 2 * wall_thickness,
                        c_start : c_start + wall_thickness,
                    ] = 0
                if cell.west:
                    maze[
                        r_start : r_start + wall_thickness,
                        c_start - wall_thickness : c_start,
                    ] = 0
                if cell.east:
                    maze[
                        r_start : r_start + wall_thickness,
                        c_start + wall_thickness : c_start + 2 * wall_thickness,
                    ] = 0

        return maze


def verify_perfect_maze(grid: Grid) -> dict:
    """
    Verify that a maze is perfect (fully connected, no loops).

    A perfect maze must satisfy:
    1. Connectivity: All cells reachable from any cell
    2. Acyclicity: Exactly (n-1) passages for n cells

    Args:
        grid: Maze grid to verify

    Returns:
        Dictionary with verification results including:
        - is_perfect: Overall validity
        - is_connected: Connectivity check
        - is_no_loops: Acyclicity check
        - visited_cells: Number of reachable cells
        - total_cells: Total number of cells
        - passage_count: Number of passages
        - expected_passages: Expected passages for perfect maze
    """
    all_cells = grid.all_cells()

    grid.reset_visited()
    start = all_cells[0]
    queue = [start]
    start.visited = True
    visited_count = 1

    while queue:
        current = queue.pop(0)
        for neighbor in grid.get_neighbors(current):
            if not neighbor.visited:
                if (
                    (neighbor.row == current.row - 1 and current.north)
                    or (neighbor.row == current.row + 1 and current.south)
                    or (neighbor.col == current.col - 1 and current.west)
                    or (neighbor.col == current.col + 1 and current.east)
                ):
                    neighbor.visited = True
                    visited_count += 1
                    queue.append(neighbor)

    total_cells = len(all_cells)
    is_connected = visited_count == total_cells

    passage_count = 0
    for cell in all_cells:
        if cell.north:
            passage_count += 1
        if cell.east:
            passage_count += 1

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


def generate_maze(
    rows: int,
    cols: int,
    algorithm: str = "recursive_backtracking",
    seed: int | None = None,
) -> np.ndarray:
    """
    High-level function to generate a perfect maze.

    Args:
        rows: Number of rows
        cols: Number of columns
        algorithm: Algorithm name ('recursive_backtracking' or 'wilsons')
        seed: Random seed for reproducibility

    Returns:
        Numpy array maze representation (1 = wall, 0 = passage)

    Example:
        >>> maze = generate_maze(20, 20, algorithm='recursive_backtracking', seed=42)
        >>> print(f"Maze shape: {maze.shape}")
        Maze shape: (41, 41)
    """
    alg_enum = MazeAlgorithm(algorithm)
    generator = PerfectMazeGenerator(rows, cols, alg_enum)
    grid = generator.generate(seed=seed)

    verification = verify_perfect_maze(grid)
    if not verification["is_perfect"]:
        raise RuntimeError(f"Generated maze is not perfect: {verification}")

    return generator.to_numpy_array()
