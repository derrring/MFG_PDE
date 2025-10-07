"""
Position Placement Strategies for Maze Environments

Implements various strategies for placing start/goal positions in mazes:
- Random placement
- Corner placement
- Edge placement
- Farthest point placement
- Custom user-specified positions

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

import random
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.alg.reinforcement.environments.maze_config import PlacementStrategy

if TYPE_CHECKING:
    from mfg_pde.alg.reinforcement.environments.maze_generator import Grid


def place_positions(
    grid: Grid,
    num_positions: int,
    strategy: PlacementStrategy,
    custom_positions: list[tuple[int, int]] | None = None,
    seed: int | None = None,
) -> list[tuple[int, int]]:
    """
    Place positions in maze according to strategy.

    Args:
        grid: Maze grid
        num_positions: Number of positions to place
        strategy: Placement strategy
        custom_positions: Custom positions (required if strategy=CUSTOM)
        seed: Random seed

    Returns:
        List of (row, col) positions

    Raises:
        ValueError: If custom positions required but not provided
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if strategy == PlacementStrategy.CUSTOM:
        if custom_positions is None or len(custom_positions) != num_positions:
            raise ValueError(f"CUSTOM strategy requires exactly {num_positions} custom_positions")
        return custom_positions

    elif strategy == PlacementStrategy.RANDOM:
        return _place_random(grid, num_positions)

    elif strategy == PlacementStrategy.CORNERS:
        return _place_corners(grid, num_positions)

    elif strategy == PlacementStrategy.EDGES:
        return _place_edges(grid, num_positions)

    elif strategy == PlacementStrategy.FARTHEST:
        return _place_farthest(grid, num_positions)

    elif strategy == PlacementStrategy.CLUSTERED:
        return _place_clustered(grid, num_positions)

    else:
        raise ValueError(f"Unknown placement strategy: {strategy}")


def _place_random(grid: Grid, num_positions: int) -> list[tuple[int, int]]:
    """Place positions randomly in valid cells."""
    all_cells = grid.all_cells()
    selected = random.sample(all_cells, num_positions)
    return [(cell.row, cell.col) for cell in selected]


def _place_corners(grid: Grid, num_positions: int) -> list[tuple[int, int]]:
    """
    Place positions in corners.

    Priority order: top-left, top-right, bottom-left, bottom-right
    """
    corners = [
        (0, 0),  # Top-left
        (0, grid.cols - 1),  # Top-right
        (grid.rows - 1, 0),  # Bottom-left
        (grid.rows - 1, grid.cols - 1),  # Bottom-right
    ]

    if num_positions <= 4:
        return corners[:num_positions]
    else:
        return corners + _place_random(grid, num_positions - 4)


def _place_edges(grid: Grid, num_positions: int) -> list[tuple[int, int]]:
    """Place positions randomly on edges."""
    edges = []

    edges.extend([(0, c) for c in range(grid.cols)])
    edges.extend([(grid.rows - 1, c) for c in range(grid.cols)])
    edges.extend([(r, 0) for r in range(1, grid.rows - 1)])
    edges.extend([(r, grid.cols - 1) for r in range(1, grid.rows - 1)])

    selected = random.sample(edges, min(num_positions, len(edges)))

    if num_positions > len(edges):
        remaining = _place_random(grid, num_positions - len(edges))
        selected.extend(remaining)

    return selected


def _place_farthest(grid: Grid, num_positions: int) -> list[tuple[int, int]]:
    """
    Place positions to maximize minimum pairwise distance.

    Uses greedy algorithm:
    1. Place first position randomly
    2. Each subsequent position placed at point farthest from all previous
    """
    if num_positions == 1:
        return _place_random(grid, 1)

    positions = []
    all_cells = grid.all_cells()

    first_cell = random.choice(all_cells)
    positions.append((first_cell.row, first_cell.col))

    for _ in range(num_positions - 1):
        max_min_dist = -1
        best_cell = None

        for cell in all_cells:
            if (cell.row, cell.col) in positions:
                continue

            min_dist = min(_maze_distance(grid, (cell.row, cell.col), pos) for pos in positions)

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_cell = (cell.row, cell.col)

        if best_cell is not None:
            positions.append(best_cell)

    return positions


def _place_clustered(grid: Grid, num_positions: int) -> list[tuple[int, int]]:
    """
    Place positions in a cluster.

    Useful for multi-agent scenarios where agents start together.
    """
    center_cell = random.choice(grid.all_cells())
    center = (center_cell.row, center_cell.col)

    if num_positions == 1:
        return [center]

    neighbors = _get_neighbors_within_radius(grid, center, radius=3)

    if len(neighbors) >= num_positions - 1:
        selected = random.sample(neighbors, num_positions - 1)
        return [center, *selected]
    else:
        remaining = _place_random(grid, num_positions - 1 - len(neighbors))
        return [center, *neighbors, *remaining]


def _maze_distance(grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> int:
    """
    Compute maze distance between two positions using BFS.

    Args:
        grid: Maze grid
        start: Starting (row, col)
        goal: Goal (row, col)

    Returns:
        Distance in maze (number of steps), or infinity if unreachable
    """
    start_cell = grid.get_cell(start[0], start[1])
    goal_cell = grid.get_cell(goal[0], goal[1])

    if start_cell is None or goal_cell is None:
        return float("inf")

    if start == goal:
        return 0

    grid.reset_visited()
    queue = deque([(start_cell, 0)])
    start_cell.visited = True

    while queue:
        current, dist = queue.popleft()

        for neighbor in _get_linked_neighbors(grid, current):
            if neighbor.row == goal[0] and neighbor.col == goal[1]:
                return dist + 1

            if not neighbor.visited:
                neighbor.visited = True
                queue.append((neighbor, dist + 1))

    return float("inf")


def _get_linked_neighbors(grid: Grid, cell: Any) -> list:
    """Get neighbors that have passages to this cell."""
    neighbors = []

    if cell.north:
        neighbor = grid.get_cell(cell.row - 1, cell.col)
        if neighbor:
            neighbors.append(neighbor)

    if cell.south:
        neighbor = grid.get_cell(cell.row + 1, cell.col)
        if neighbor:
            neighbors.append(neighbor)

    if cell.east:
        neighbor = grid.get_cell(cell.row, cell.col + 1)
        if neighbor:
            neighbors.append(neighbor)

    if cell.west:
        neighbor = grid.get_cell(cell.row, cell.col - 1)
        if neighbor:
            neighbors.append(neighbor)

    return neighbors


def _get_neighbors_within_radius(grid: Grid, center: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    """Get all cells within Manhattan distance radius from center."""
    neighbors = []
    center_row, center_col = center

    for r in range(max(0, center_row - radius), min(grid.rows, center_row + radius + 1)):
        for c in range(max(0, center_col - radius), min(grid.cols, center_col + radius + 1)):
            if (r, c) != center:
                manhattan_dist = abs(r - center_row) + abs(c - center_col)
                if manhattan_dist <= radius:
                    neighbors.append((r, c))

    return neighbors


def compute_position_metrics(grid: Grid, positions: list[tuple[int, int]]) -> dict:
    """
    Compute metrics about position placement.

    Args:
        grid: Maze grid
        positions: List of (row, col) positions

    Returns:
        Dictionary with metrics:
        - min_distance: Minimum pairwise maze distance
        - max_distance: Maximum pairwise maze distance
        - avg_distance: Average pairwise maze distance
        - total_pairs: Number of position pairs
    """
    if len(positions) < 2:
        return {
            "min_distance": 0,
            "max_distance": 0,
            "avg_distance": 0.0,
            "total_pairs": 0,
        }

    distances = []
    for i, pos1 in enumerate(positions):
        for pos2 in positions[i + 1 :]:
            dist = _maze_distance(grid, pos1, pos2)
            if dist != float("inf"):
                distances.append(dist)

    if not distances:
        return {
            "min_distance": float("inf"),
            "max_distance": float("inf"),
            "avg_distance": float("inf"),
            "total_pairs": len(positions) * (len(positions) - 1) // 2,
        }

    return {
        "min_distance": min(distances),
        "max_distance": max(distances),
        "avg_distance": np.mean(distances),
        "total_pairs": len(distances),
    }
