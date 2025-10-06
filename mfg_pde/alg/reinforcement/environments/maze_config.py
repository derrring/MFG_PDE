"""
Maze Configuration for MFG-RL Environments

Provides comprehensive configuration for maze generation with control over:
- Physical dimensions (continuous vs discrete)
- Entry/exit positions and counts
- Obstacle density and distribution
- Maze topology parameters

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


class PlacementStrategy(Enum):
    """Strategy for placing start/goal positions."""

    RANDOM = "random"  # Random valid positions
    CORNERS = "corners"  # Opposite corners
    EDGES = "edges"  # Opposite edges (random on edge)
    CUSTOM = "custom"  # User-specified positions
    FARTHEST = "farthest"  # Maximally distant points
    CLUSTERED = "clustered"  # Near each other (for multi-agent)


class MazeTopology(Enum):
    """Maze topology type."""

    GRID_2D = "grid_2d"  # Standard 2D rectangular grid
    TORUS = "torus"  # Wrap-around boundaries
    CYLINDER = "cylinder"  # Wrap one dimension
    HEX_GRID = "hex_grid"  # Hexagonal tiling (future)


@dataclass
class PhysicalDimensions:
    """
    Physical dimensions for continuous-space interpretation.

    Attributes:
        width: Physical width (e.g., meters, continuous units)
        height: Physical height
        cell_size: Physical size of each grid cell
    """

    width: float
    height: float
    cell_size: float | None = None

    def __post_init__(self):
        """Calculate cell size if not provided."""
        if self.cell_size is None:
            self.cell_size = min(self.width, self.height) / 100.0


@dataclass
class MazeConfig:
    """
    Comprehensive maze generation configuration.

    This configuration provides fine-grained control over maze structure,
    physical properties, and entry/exit placement for MFG-RL experiments.

    Attributes:
        # Grid dimensions (discrete)
        rows: Number of cell rows
        cols: Number of cell columns

        # Algorithm parameters
        algorithm: Generation algorithm ('recursive_backtracking' or 'wilsons')
        seed: Random seed for reproducibility

        # Physical dimensions (continuous interpretation)
        physical_dims: Optional physical dimensions for continuous space

        # Entry/exit configuration
        num_starts: Number of entry points
        num_goals: Number of exit points
        start_positions: Explicit start positions (row, col) or None for auto
        goal_positions: Explicit goal positions (row, col) or None for auto
        placement_strategy: Strategy for auto-placement

        # Maze topology
        topology: Maze topology (grid, torus, cylinder)

        # Rendering parameters
        wall_thickness: Thickness of walls in pixel representation
        include_boundary: Whether to add outer boundary walls

        # Validation
        verify_perfect: Whether to verify perfect maze properties
    """

    # Grid dimensions
    rows: int
    cols: int

    # Algorithm
    algorithm: Literal["recursive_backtracking", "wilsons"] = "recursive_backtracking"
    seed: int | None = None

    # Physical dimensions (optional)
    physical_dims: PhysicalDimensions | None = None

    # Entry/exit configuration
    num_starts: int = 1
    num_goals: int = 1
    start_positions: list[tuple[int, int]] | None = None
    goal_positions: list[tuple[int, int]] | None = None
    placement_strategy: PlacementStrategy = PlacementStrategy.CORNERS

    # Topology
    topology: MazeTopology = MazeTopology.GRID_2D

    # Rendering
    wall_thickness: int = 1
    include_boundary: bool = True

    # Validation
    verify_perfect: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.rows < 2 or self.cols < 2:
            raise ValueError(f"Maze must be at least 2x2, got {self.rows}x{self.cols}")

        if self.num_starts < 1:
            raise ValueError(f"Must have at least 1 start, got {self.num_starts}")

        if self.num_goals < 1:
            raise ValueError(f"Must have at least 1 goal, got {self.num_goals}")

        if self.start_positions is not None:
            if len(self.start_positions) != self.num_starts:
                raise ValueError(f"num_starts={self.num_starts} but got {len(self.start_positions)} start_positions")

        if self.goal_positions is not None:
            if len(self.goal_positions) != self.num_goals:
                raise ValueError(f"num_goals={self.num_goals} but got {len(self.goal_positions)} goal_positions")

    def get_pixel_dimensions(self) -> tuple[int, int]:
        """
        Get maze dimensions in pixel/array representation.

        Returns:
            (height, width) in pixels for numpy array
        """
        cell_size = 2 * self.wall_thickness + 1
        height = self.rows * cell_size + self.wall_thickness
        width = self.cols * cell_size + self.wall_thickness
        return height, width

    def get_continuous_dimensions(self) -> tuple[float, float] | None:
        """
        Get maze dimensions in continuous space.

        Returns:
            (width, height) in physical units, or None if not configured
        """
        if self.physical_dims is None:
            return None
        return self.physical_dims.width, self.physical_dims.height

    def cell_to_continuous(self, row: int, col: int) -> tuple[float, float] | None:
        """
        Convert cell coordinates to continuous space.

        Args:
            row: Cell row index
            col: Cell column index

        Returns:
            (x, y) in continuous coordinates, or None if not configured
        """
        if self.physical_dims is None:
            return None

        cell_width = self.physical_dims.width / self.cols
        cell_height = self.physical_dims.height / self.rows

        x = (col + 0.5) * cell_width
        y = (row + 0.5) * cell_height

        return x, y

    def continuous_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        """
        Convert continuous coordinates to cell indices.

        Args:
            x: Continuous x coordinate
            y: Continuous y coordinate

        Returns:
            (row, col) cell indices, or None if not configured
        """
        if self.physical_dims is None:
            return None

        cell_width = self.physical_dims.width / self.cols
        cell_height = self.physical_dims.height / self.rows

        col = int(x / cell_width)
        row = int(y / cell_height)

        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))

        return row, col


def create_default_config(
    rows: int = 20,
    cols: int = 20,
    algorithm: str = "recursive_backtracking",
    **kwargs: Any,
) -> MazeConfig:
    """
    Create maze configuration with sensible defaults.

    Args:
        rows: Number of rows
        cols: Number of columns
        algorithm: Generation algorithm
        **kwargs: Additional configuration parameters

    Returns:
        MazeConfig instance

    Example:
        >>> config = create_default_config(20, 20, num_goals=3)
        >>> print(config.rows, config.cols, config.num_goals)
        20 20 3
    """
    return MazeConfig(rows=rows, cols=cols, algorithm=algorithm, **kwargs)


def create_continuous_maze_config(
    physical_width: float,
    physical_height: float,
    cell_density: int = 20,
    **kwargs: Any,
) -> MazeConfig:
    """
    Create maze with continuous physical dimensions.

    Args:
        physical_width: Physical width in continuous units (e.g., meters)
        physical_height: Physical height in continuous units
        cell_density: Approximate number of cells per unit length
        **kwargs: Additional configuration parameters

    Returns:
        MazeConfig with physical dimensions configured

    Example:
        >>> # Create 10m x 10m maze with ~20 cells/meter
        >>> config = create_continuous_maze_config(10.0, 10.0, cell_density=20)
        >>> print(config.rows, config.cols)
        200 200
    """
    rows = int(physical_height * cell_density)
    cols = int(physical_width * cell_density)

    physical_dims = PhysicalDimensions(
        width=physical_width,
        height=physical_height,
        cell_size=1.0 / cell_density,
    )

    return MazeConfig(rows=rows, cols=cols, physical_dims=physical_dims, **kwargs)


def create_multi_goal_config(
    rows: int,
    cols: int,
    num_goals: int,
    goal_strategy: str = "edges",
    **kwargs: Any,
) -> MazeConfig:
    """
    Create maze configuration with multiple goals.

    Args:
        rows: Number of rows
        cols: Number of columns
        num_goals: Number of goal positions
        goal_strategy: Placement strategy for goals
        **kwargs: Additional configuration parameters

    Returns:
        MazeConfig with multiple goals

    Example:
        >>> config = create_multi_goal_config(30, 30, num_goals=4, goal_strategy='corners')
    """
    strategy_map = {
        "random": PlacementStrategy.RANDOM,
        "corners": PlacementStrategy.CORNERS,
        "edges": PlacementStrategy.EDGES,
        "farthest": PlacementStrategy.FARTHEST,
    }

    strategy = strategy_map.get(goal_strategy, PlacementStrategy.EDGES)

    return MazeConfig(
        rows=rows,
        cols=cols,
        num_goals=num_goals,
        placement_strategy=strategy,
        **kwargs,
    )
