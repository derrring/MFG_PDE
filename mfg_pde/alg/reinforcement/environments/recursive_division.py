"""
Recursive Division Maze Generation

Generates mazes with variable-width corridors, rooms, and open spaces.
Unlike traditional maze algorithms that "carve" paths, Recursive Division
starts with an empty space and adds walls, creating structured layouts
ideal for MFG scenarios with rooms and controllable bottlenecks.

Algorithm:
1. Start with completely open space (one large room)
2. Recursively divide space with walls
3. Add doors (passages) in each wall
4. Stop when rooms reach minimum size

Properties:
- Variable-width corridors (controllable)
- Structured, building-like layouts
- Rooms of varying sizes
- Controllable bottlenecks (door widths)
- Perfect for crowd dynamics, building layouts

Reference: Classic dungeon generation technique
Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np


class SplitOrientation(Enum):
    """Orientation for dividing a room."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@dataclass
class RecursiveDivisionConfig:
    """
    Configuration for Recursive Division maze generation.

    Attributes:
        rows: Grid height
        cols: Grid width
        min_room_width: Minimum room width before stopping division
        min_room_height: Minimum room height before stopping division
        door_width: Width of passages (1 = narrow, 3 = wide)
        num_doors_per_wall: Number of doors in each dividing wall
        split_bias: Bias toward horizontal (0.0) vs vertical (1.0) splits
        wall_thickness: Thickness of dividing walls
        seed: Random seed for reproducibility
    """

    rows: int
    cols: int
    min_room_width: int = 5
    min_room_height: int = 5
    door_width: int = 2
    num_doors_per_wall: int = 1
    split_bias: float = 0.5
    wall_thickness: int = 1
    seed: int | None = None

    def __post_init__(self):
        """Validate configuration."""
        if self.rows < self.min_room_height:
            raise ValueError(f"rows ({self.rows}) must be >= min_room_height ({self.min_room_height})")
        if self.cols < self.min_room_width:
            raise ValueError(f"cols ({self.cols}) must be >= min_room_width ({self.min_room_width})")
        if self.door_width < 1:
            raise ValueError(f"door_width must be >= 1, got {self.door_width}")
        if self.num_doors_per_wall < 1:
            raise ValueError(f"num_doors_per_wall must be >= 1, got {self.num_doors_per_wall}")
        if not 0.0 <= self.split_bias <= 1.0:
            raise ValueError(f"split_bias must be in [0, 1], got {self.split_bias}")


class RecursiveDivisionGenerator:
    """
    Generates mazes using Recursive Division algorithm.

    Creates structured layouts with rooms and variable-width corridors,
    ideal for MFG scenarios requiring open spaces and controllable bottlenecks.
    """

    def __init__(self, config: RecursiveDivisionConfig):
        """
        Initialize generator.

        Args:
            config: Configuration for maze generation
        """
        self.config = config
        self.maze: np.ndarray | None = None

    def generate(self, seed: int | None = None) -> np.ndarray:
        """
        Generate maze using Recursive Division.

        Args:
            seed: Random seed (overrides config seed)

        Returns:
            Numpy array where 1 = wall, 0 = open space
        """
        if seed is not None:
            random.seed(seed)
        elif self.config.seed is not None:
            random.seed(self.config.seed)

        # Initialize with all open space
        self.maze = np.zeros((self.config.rows, self.config.cols), dtype=np.int32)

        # Add outer boundary walls
        self.maze[0, :] = 1
        self.maze[-1, :] = 1
        self.maze[:, 0] = 1
        self.maze[:, -1] = 1

        # Recursively divide the interior
        self._divide(
            1,  # top
            1,  # left
            self.config.rows - 2,  # bottom
            self.config.cols - 2,  # right
        )

        return self.maze

    def _divide(self, top: int, left: int, bottom: int, right: int):
        """
        Recursively divide a chamber.

        Args:
            top: Top boundary (inclusive)
            left: Left boundary (inclusive)
            bottom: Bottom boundary (inclusive)
            right: Right boundary (inclusive)
        """
        height = bottom - top + 1
        width = right - left + 1

        # Stop if room too small
        if height < self.config.min_room_height or width < self.config.min_room_width:
            return

        # Decide split orientation
        if height < self.config.min_room_height * 2:
            orientation = SplitOrientation.VERTICAL
        elif width < self.config.min_room_width * 2:
            orientation = SplitOrientation.HORIZONTAL
        else:
            # Use bias to decide
            orientation = (
                SplitOrientation.HORIZONTAL if random.random() < self.config.split_bias else SplitOrientation.VERTICAL
            )

        if orientation == SplitOrientation.HORIZONTAL:
            self._divide_horizontally(top, left, bottom, right, height, width)
        else:
            self._divide_vertically(top, left, bottom, right, height, width)

    def _divide_horizontally(self, top: int, left: int, bottom: int, right: int, height: int, width: int):
        """Add horizontal wall with door(s)."""
        # Choose where to place wall (leave room for recursion)
        min_wall_pos = top + self.config.min_room_height - 1
        max_wall_pos = bottom - self.config.min_room_height + 1

        if min_wall_pos > max_wall_pos:
            return

        wall_row = random.randint(min_wall_pos, max_wall_pos)

        # Add wall (extend beyond chamber to fill corners/intersections)
        for thickness_offset in range(self.config.wall_thickness):
            if wall_row + thickness_offset <= bottom:
                # Extend wall slightly beyond chamber boundaries to fill intersections
                wall_left = max(0, left - 1)
                wall_right = min(self.config.cols - 1, right + 1)
                self.maze[wall_row + thickness_offset, wall_left : wall_right + 1] = 1

        # Add door(s)
        self._add_doors(wall_row, left, right, orientation=SplitOrientation.HORIZONTAL)

        # Recursively divide sub-chambers
        self._divide(top, left, wall_row - 1, right)
        self._divide(wall_row + self.config.wall_thickness, left, bottom, right)

    def _divide_vertically(self, top: int, left: int, bottom: int, right: int, height: int, width: int):
        """Add vertical wall with door(s)."""
        # Choose where to place wall
        min_wall_pos = left + self.config.min_room_width - 1
        max_wall_pos = right - self.config.min_room_width + 1

        if min_wall_pos > max_wall_pos:
            return

        wall_col = random.randint(min_wall_pos, max_wall_pos)

        # Add wall (extend beyond chamber to fill corners/intersections)
        for thickness_offset in range(self.config.wall_thickness):
            if wall_col + thickness_offset <= right:
                # Extend wall slightly beyond chamber boundaries to fill intersections
                wall_top = max(0, top - 1)
                wall_bottom = min(self.config.rows - 1, bottom + 1)
                self.maze[wall_top : wall_bottom + 1, wall_col + thickness_offset] = 1

        # Add door(s)
        self._add_doors(wall_col, top, bottom, orientation=SplitOrientation.VERTICAL)

        # Recursively divide sub-chambers
        self._divide(top, left, bottom, wall_col - 1)
        self._divide(top, wall_col + self.config.wall_thickness, bottom, right)

    def _add_doors(
        self,
        wall_pos: int,
        start: int,
        end: int,
        orientation: SplitOrientation,
    ):
        """
        Add door(s) in a wall.

        Args:
            wall_pos: Position of wall (row for horizontal, col for vertical)
            start: Start of wall span
            end: End of wall span
            orientation: Wall orientation
        """
        wall_length = end - start + 1

        # Determine number of doors (respect minimum room size)
        max_doors = wall_length // (self.config.min_room_width + 1)
        num_doors = min(self.config.num_doors_per_wall, max_doors)

        if num_doors < 1:
            num_doors = 1

        # Place doors randomly along wall
        door_positions = []
        for _ in range(num_doors):
            # Ensure door has enough space
            max_door_start = end - self.config.door_width + 1
            if max_door_start < start:
                max_door_start = start

            door_start = random.randint(start, max_door_start)
            door_positions.append(door_start)

        # Create doors
        for door_start in door_positions:
            for door_offset in range(self.config.door_width):
                door_pos = door_start + door_offset
                if door_pos > end:
                    break

                # Remove wall to create door
                for thickness_offset in range(self.config.wall_thickness):
                    if orientation == SplitOrientation.HORIZONTAL:
                        if wall_pos + thickness_offset < self.maze.shape[0]:
                            self.maze[wall_pos + thickness_offset, door_pos] = 0
                    else:  # VERTICAL
                        if wall_pos + thickness_offset < self.maze.shape[1]:
                            self.maze[door_pos, wall_pos + thickness_offset] = 0


def add_loops(
    maze: np.ndarray,
    loop_density: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Add loops to a maze by removing random internal walls.

    Creates "braided" mazes with multiple paths between points,
    essential for MFG scenarios with route choice and congestion.

    Args:
        maze: Input maze (1 = wall, 0 = open)
        loop_density: Fraction of removable walls to remove (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        Maze with added loops

    Example:
        >>> perfect_maze = generator.generate()
        >>> braided_maze = add_loops(perfect_maze, loop_density=0.15)
    """
    if seed is not None:
        random.seed(seed)

    maze_copy = maze.copy()
    height, width = maze.shape

    # Find internal walls that can be safely removed
    removable_walls = []

    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if maze[r, c] == 1:  # Is wall
                # Check if wall separates two open spaces
                # Count open neighbors
                open_neighbors = 0
                if r > 0 and maze[r - 1, c] == 0:
                    open_neighbors += 1
                if r < height - 1 and maze[r + 1, c] == 0:
                    open_neighbors += 1
                if c > 0 and maze[r, c - 1] == 0:
                    open_neighbors += 1
                if c < width - 1 and maze[r, c + 1] == 0:
                    open_neighbors += 1

                # Wall between two open spaces is removable
                if open_neighbors >= 2:
                    removable_walls.append((r, c))

    # Remove random subset of walls
    num_to_remove = int(len(removable_walls) * loop_density)
    if num_to_remove > 0:
        walls_to_remove = random.sample(removable_walls, num_to_remove)

        for r, c in walls_to_remove:
            maze_copy[r, c] = 0

    return maze_copy


def create_room_based_config(
    rows: int = 40,
    cols: int = 60,
    room_size: Literal["small", "medium", "large"] = "medium",
    corridor_width: Literal["narrow", "medium", "wide"] = "medium",
    **kwargs,
) -> RecursiveDivisionConfig:
    """
    Create configuration with preset room sizes and corridor widths.

    Args:
        rows: Grid height
        cols: Grid width
        room_size: Preset room size (small/medium/large)
        corridor_width: Preset corridor width (narrow/medium/wide)
        **kwargs: Additional configuration parameters

    Returns:
        RecursiveDivisionConfig

    Example:
        >>> config = create_room_based_config(40, 60, "large", "wide")
        >>> generator = RecursiveDivisionGenerator(config)
        >>> maze = generator.generate()
    """
    room_sizes = {
        "small": (3, 3),
        "medium": (5, 5),
        "large": (8, 8),
    }

    corridor_widths = {
        "narrow": 1,
        "medium": 2,
        "wide": 3,
    }

    min_width, min_height = room_sizes[room_size]
    door_width = corridor_widths[corridor_width]

    return RecursiveDivisionConfig(
        rows=rows,
        cols=cols,
        min_room_width=min_width,
        min_room_height=min_height,
        door_width=door_width,
        **kwargs,
    )
