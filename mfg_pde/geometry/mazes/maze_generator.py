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
- Eller's Algorithm: Efficient row-by-row generation, O(width) memory
- Growing Tree: Flexible framework, balanced characteristics

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
from typing import TYPE_CHECKING

import numpy as np

from mfg_pde.geometry.geometry_protocol import GeometryType

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MazeAlgorithm(Enum):
    """Available perfect maze generation algorithms."""

    RECURSIVE_BACKTRACKING = "recursive_backtracking"
    WILSONS = "wilsons"
    ELLERS = "ellers"
    GROWING_TREE = "growing_tree"


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

    def link(self, other: Cell, bidirectional: bool = True) -> None:
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

    # GeometryProtocol implementation
    @property
    def dimension(self) -> int:
        """Spatial dimension of the maze (always 2 for grid-based mazes)."""
        return 2

    @property
    def geometry_type(self) -> GeometryType:
        """Type of geometry (MAZE for maze grids)."""
        return GeometryType.MAZE

    @property
    def num_spatial_points(self) -> int:
        """Total number of discrete spatial points (maze cells)."""
        return self.rows * self.cols

    def get_spatial_grid(self) -> NDArray:
        """
        Get spatial grid representation (cell center coordinates).

        Returns:
            numpy array of shape (rows * cols, 2) containing (x, y) coordinates
            of cell centers in the maze grid
        """
        # Generate cell center coordinates
        # Convention: (col + 0.5, row + 0.5) for cell centers
        x_coords = []
        y_coords = []
        for row in range(self.rows):
            for col in range(self.cols):
                x_coords.append(col + 0.5)
                y_coords.append(row + 0.5)

        return np.column_stack([x_coords, y_coords])


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
        elif self.algorithm == MazeAlgorithm.ELLERS:
            self._ellers()
        elif self.algorithm == MazeAlgorithm.GROWING_TREE:
            self._growing_tree()
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

    def _ellers(self):
        """
        Eller's algorithm for efficient row-by-row maze generation.

        Generates mazes one row at a time with minimal memory usage,
        making it ideal for very large mazes or streaming generation.

        Algorithm:
        1. Process rows from top to bottom
        2. Assign each cell to a set (initially unique sets)
        3. Randomly join adjacent cells in same row (merge sets)
        4. Create vertical connections ensuring each set has >= 1 passage down
        5. Continue sets to next row

        Characteristics:
        - Memory efficient: O(width) instead of O(width × height)
        - Can generate infinite mazes (streaming)
        - Fast: O(n) where n = number of cells
        - Moderate exploration difficulty
        - Balanced mixture of horizontal/vertical passages

        Reference: Eller (1982), "An Efficient Method for Generating Mazes"
        """
        # Each cell tracks its set membership
        current_row_sets = list(range(self.cols))
        next_set_id = self.cols

        for row_idx in range(self.rows):
            row_cells = [self.grid.cells[row_idx][col] for col in range(self.cols)]

            # Mark all cells in current row as visited
            for cell in row_cells:
                cell.visited = True

            # Step 1: Randomly connect adjacent cells in same row (merge sets)
            for col in range(self.cols - 1):
                cell = row_cells[col]
                neighbor = row_cells[col + 1]

                # Join if different sets and (random choice OR last row)
                should_join = row_idx == self.rows - 1 or random.random() > 0.5
                if current_row_sets[col] != current_row_sets[col + 1] and should_join:
                    # Link cells horizontally
                    cell.link(neighbor)

                    # Merge sets
                    old_set = current_row_sets[col + 1]
                    new_set = current_row_sets[col]
                    current_row_sets = [new_set if s == old_set else s for s in current_row_sets]

            # Step 2: Create vertical connections (if not last row)
            if row_idx < self.rows - 1:
                # Group cells by their set
                sets_in_row: dict[int, list[int]] = {}
                for col in range(self.cols):
                    set_id = current_row_sets[col]
                    if set_id not in sets_in_row:
                        sets_in_row[set_id] = []
                    sets_in_row[set_id].append(col)

                # For each set, create at least one vertical connection
                next_row_sets = [-1] * self.cols

                for set_id, cols_in_set in sets_in_row.items():
                    # Choose how many vertical connections for this set
                    num_connections = random.randint(1, max(1, len(cols_in_set)))
                    chosen_cols = random.sample(cols_in_set, num_connections)

                    for col in chosen_cols:
                        cell = row_cells[col]
                        below = self.grid.cells[row_idx + 1][col]

                        # Link vertically
                        cell.link(below)

                        # Carry set membership down
                        next_row_sets[col] = set_id

                # Assign new sets to unconnected cells in next row
                for col in range(self.cols):
                    if next_row_sets[col] == -1:
                        next_row_sets[col] = next_set_id
                        next_set_id += 1

                current_row_sets = next_row_sets

    def _growing_tree(self, selection_strategy: str = "mixed") -> None:
        """
        Growing Tree algorithm - generalized framework for maze generation.

        A flexible algorithm that produces different maze characteristics
        based on how cells are selected from the active set.

        Algorithm:
        1. Start with one cell in active set
        2. While active set not empty:
           - Choose cell from active set (strategy-dependent)
           - If cell has unvisited neighbors:
               * Choose random neighbor
               * Link and add neighbor to active set
           - Else remove cell from active set

        Selection Strategies:
        - newest: Always choose most recent cell (→ Recursive Backtracking)
        - oldest: Always choose oldest cell (→ Prim's algorithm)
        - random: Choose random cell (→ Simplified Prim's)
        - mixed: 50% newest, 50% random (balanced characteristics)

        Characteristics (mixed strategy):
        - Balanced: Combines benefits of DFS and Prim's
        - Moderate branching and dead ends
        - Good exploration difficulty
        - Versatile maze structure

        Reference: Jamis Buck, "Mazes for Programmers" (2015)
        """
        active = []
        start_cell = self.grid.random_cell()
        start_cell.visited = True
        active.append(start_cell)

        while active:
            # Select cell based on strategy
            if selection_strategy == "newest":
                # Like Recursive Backtracking (DFS)
                current = active[-1]
            elif selection_strategy == "oldest":
                # Like Prim's algorithm
                current = active[0]
            elif selection_strategy == "random":
                # Simplified Prim's
                current = random.choice(active)
            elif selection_strategy == "mixed":
                # 50% newest (DFS-like), 50% random (Prim's-like)
                if random.random() < 0.5:
                    current = active[-1]
                else:
                    current = random.choice(active)
            else:
                raise ValueError(f"Unknown selection strategy: {selection_strategy}")

            unvisited = self.grid.get_unvisited_neighbors(current)

            if unvisited:
                neighbor = random.choice(unvisited)
                current.link(neighbor)
                neighbor.visited = True
                active.append(neighbor)
            else:
                active.remove(current)

    def to_numpy_array(self, wall_thickness: int = 1) -> NDArray[np.int32]:
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
) -> NDArray[np.int32]:
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
