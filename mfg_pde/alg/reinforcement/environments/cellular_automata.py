"""
Cellular Automata Maze Generation for MFG Environments

Generates organic, cave-like mazes using cellular automata rules.
Unlike structured algorithms (Recursive Backtracking, Recursive Division),
CA produces natural-looking, unpredictable layouts with variable-width passages.

Algorithm:
1. Random initialization with wall probability
2. Iteratively apply smoothing rules (4-5 rule typical) - VECTORIZED
3. Post-process to ensure connectivity
4. Optional: Add openings and clean small regions

Properties:
- Organic, cave-like appearance
- Variable-width passages emerge naturally
- Unpredictable layouts (high replayability)
- Configurable density and smoothness
- Ideal for natural environments, parks, irregular urban spaces

MFG Suitability (EXCELLENT):
- Variable-width corridors model realistic congestion dynamics
- Multiple path options enable route choice behavior
- Organic structure mimics real urban environments
- Controllable density matches crowd scenarios
- Natural bottlenecks create interesting equilibrium patterns

Performance: ~100x faster than nested loops via NumPy vectorization

Reference: Stephen Wolfram, "A New Kind of Science"
Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    from scipy.ndimage import convolve as scipy_convolve

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SCIPY_AVAILABLE = False
    scipy_convolve = None  # type: ignore


@dataclass
class CellularAutomataConfig:
    """
    Configuration for Cellular Automata maze generation.

    Attributes:
        rows: Grid height
        cols: Grid width
        initial_wall_prob: Initial probability of wall (0.0-1.0)
        num_iterations: Number of CA smoothing iterations
        birth_limit: Neighbors needed for cell to become wall
        death_limit: Neighbors needed for wall to stay wall
        use_moore_neighborhood: Use 8-connected (True) or 4-connected (False)
        ensure_connectivity: Post-process to connect regions
        min_region_size: Remove regions smaller than this
        seed: Random seed for reproducibility
    """

    rows: int
    cols: int
    initial_wall_prob: float = 0.45
    num_iterations: int = 5
    birth_limit: int = 5  # If >= 5 neighbors are walls, become wall
    death_limit: int = 4  # If < 4 neighbors are walls, become open
    use_moore_neighborhood: bool = True  # 8-connected
    ensure_connectivity: bool = True
    min_region_size: int = 10
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.rows < 5:
            raise ValueError(f"rows must be >= 5, got {self.rows}")
        if self.cols < 5:
            raise ValueError(f"cols must be >= 5, got {self.cols}")
        if not 0.0 <= self.initial_wall_prob <= 1.0:
            raise ValueError(f"initial_wall_prob must be in [0, 1], got {self.initial_wall_prob}")
        if self.num_iterations < 0:
            raise ValueError(f"num_iterations must be >= 0, got {self.num_iterations}")
        if self.birth_limit < 0:
            raise ValueError(f"birth_limit must be >= 0, got {self.birth_limit}")
        if self.death_limit < 0:
            raise ValueError(f"death_limit must be >= 0, got {self.death_limit}")


class CellularAutomataGenerator:
    """
    Generates organic, cave-like mazes using cellular automata.

    The algorithm uses simple rules to create natural-looking spaces:
    - Start with random noise (initial_wall_prob)
    - Apply smoothing rules iteratively
    - Cells with many wall neighbors become walls
    - Cells with few wall neighbors become open
    - Post-process to ensure connectivity

    Example:
        >>> config = CellularAutomataConfig(
        ...     rows=50, cols=50,
        ...     initial_wall_prob=0.45,
        ...     num_iterations=5,
        ... )
        >>> generator = CellularAutomataGenerator(config)
        >>> maze = generator.generate(seed=42)
    """

    def __init__(self, config: CellularAutomataConfig):
        """
        Initialize CA generator.

        Args:
            config: Configuration for CA maze generation
        """
        self.config = config
        self.maze: NDArray | None = None

    def generate(self, seed: int | None = None) -> NDArray:
        """
        Generate maze using cellular automata.

        Args:
            seed: Random seed (overrides config seed)

        Returns:
            Numpy array where 1 = wall, 0 = open space
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        elif self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

        # Initialize with random walls
        self.maze = self._initialize_random()

        # Apply CA rules iteratively
        for _ in range(self.config.num_iterations):
            self.maze = self._apply_ca_step()

        # Add boundary walls
        self._add_boundary_walls()

        # Post-processing
        if self.config.ensure_connectivity:
            self._ensure_connectivity()

        if self.config.min_region_size > 0:
            self._remove_small_regions()

        return self.maze

    def _initialize_random(self) -> NDArray:
        """Initialize maze with random walls."""
        maze = np.random.random((self.config.rows, self.config.cols))
        return (maze < self.config.initial_wall_prob).astype(np.int32)

    def _apply_ca_step(self) -> NDArray:
        """
        Apply one step of cellular automata rules using vectorized NumPy operations.

        This optimized version uses convolution to count neighbors efficiently,
        providing ~100x speedup over naive nested loops for large mazes.
        Falls back to manual counting if scipy unavailable.
        """
        if SCIPY_AVAILABLE:
            return self._apply_ca_step_vectorized()
        else:
            return self._apply_ca_step_manual()

    def _apply_ca_step_vectorized(self) -> NDArray:
        """Vectorized CA step using scipy convolution (fast)."""
        # Create convolution kernel based on neighborhood type
        if self.config.use_moore_neighborhood:
            # 8-connected (Moore) neighborhood
            kernel = np.ones((3, 3), dtype=np.int32)
            kernel[1, 1] = 0  # Don't count center cell
        else:
            # 4-connected (von Neumann) neighborhood
            kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32)

        # Pad maze to handle boundary conditions (edges count as walls)
        padded_maze = np.pad(self.maze, pad_width=1, mode="constant", constant_values=1)

        # Count wall neighbors using convolution
        neighbor_counts = scipy_convolve(padded_maze, kernel, mode="constant", cval=0)

        # Remove padding
        neighbor_counts = neighbor_counts[1:-1, 1:-1]

        # Apply CA rules vectorized
        new_maze = np.zeros_like(self.maze)

        # For currently wall cells: stay wall if >= death_limit neighbors
        wall_mask = self.maze == 1
        new_maze[wall_mask] = (neighbor_counts[wall_mask] >= self.config.death_limit).astype(np.int32)

        # For currently open cells: become wall if >= birth_limit neighbors
        open_mask = self.maze == 0
        new_maze[open_mask] = (neighbor_counts[open_mask] >= self.config.birth_limit).astype(np.int32)

        return new_maze

    def _apply_ca_step_manual(self) -> NDArray:
        """Manual CA step with nested loops (fallback when scipy unavailable)."""
        new_maze = np.zeros_like(self.maze)

        for r in range(self.config.rows):
            for c in range(self.config.cols):
                wall_count = self._count_wall_neighbors_manual(r, c)

                if self.maze[r, c] == 1:
                    # Currently a wall
                    new_maze[r, c] = 1 if wall_count >= self.config.death_limit else 0
                else:
                    # Currently open
                    new_maze[r, c] = 1 if wall_count >= self.config.birth_limit else 0

        return new_maze

    def _count_wall_neighbors_manual(self, row: int, col: int) -> int:
        """Count wall neighbors around a cell (manual fallback)."""
        count = 0

        if self.config.use_moore_neighborhood:
            # 8-connected neighborhood
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            # 4-connected neighborhood
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in offsets:
            r, c = row + dr, col + dc

            # Count edges as walls
            if r < 0 or r >= self.config.rows or c < 0 or c >= self.config.cols or self.maze[r, c] == 1:
                count += 1

        return count

    def _add_boundary_walls(self) -> None:
        """Add walls around the boundary."""
        self.maze[0, :] = 1
        self.maze[-1, :] = 1
        self.maze[:, 0] = 1
        self.maze[:, -1] = 1

    def _ensure_connectivity(self) -> None:
        """
        Ensure maze has one connected open region.

        Uses flood fill to find largest region, removes others.
        """
        # Find all open regions
        regions = self._find_regions()

        if not regions:
            # No open space, create a central opening
            center_r, center_c = self.config.rows // 2, self.config.cols // 2
            self.maze[center_r - 2 : center_r + 3, center_c - 2 : center_c + 3] = 0
            return

        # Keep only the largest region
        largest_region = max(regions, key=len)

        # Mark all cells not in largest region as walls
        for r in range(self.config.rows):
            for c in range(self.config.cols):
                if self.maze[r, c] == 0 and (r, c) not in largest_region:
                    self.maze[r, c] = 1

    def _find_regions(self) -> list[set[tuple[int, int]]]:
        """
        Find all connected open regions using flood fill.

        Returns:
            List of sets, each containing positions in a region
        """
        visited = np.zeros_like(self.maze, dtype=bool)
        regions = []

        for r in range(self.config.rows):
            for c in range(self.config.cols):
                if self.maze[r, c] == 0 and not visited[r, c]:
                    # Found new region, flood fill
                    region = self._flood_fill(r, c, visited)
                    regions.append(region)

        return regions

    def _flood_fill(self, start_r: int, start_c: int, visited: NDArray) -> set[tuple[int, int]]:
        """
        Flood fill to find connected region.

        Args:
            start_r: Starting row
            start_c: Starting column
            visited: Boolean array tracking visited cells

        Returns:
            Set of positions in this region
        """
        region = set()
        stack = [(start_r, start_c)]

        while stack:
            r, c = stack.pop()

            if r < 0 or r >= self.config.rows or c < 0 or c >= self.config.cols:
                continue
            if visited[r, c] or self.maze[r, c] == 1:
                continue

            visited[r, c] = True
            region.add((r, c))

            # Add 4-connected neighbors
            stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])

        return region

    def _remove_small_regions(self) -> None:
        """Remove regions smaller than min_region_size."""
        regions = self._find_regions()

        for region in regions:
            if len(region) < self.config.min_region_size:
                # Fill in small region
                for r, c in region:
                    self.maze[r, c] = 1


def create_preset_ca_config(
    rows: int,
    cols: int,
    style: str = "cave",
    seed: int | None = None,
) -> CellularAutomataConfig:
    """
    Create preset CA configurations for common styles.

    Args:
        rows: Grid height
        cols: Grid width
        style: Preset style ("cave", "cavern", "maze", "dense", "sparse")
        seed: Random seed

    Returns:
        CellularAutomataConfig

    Styles:
        - cave: Classic cave (45% walls, 5 iterations)
        - cavern: Large open spaces (40% walls, 4 iterations)
        - maze: More maze-like (50% walls, 6 iterations)
        - dense: Dense passages (55% walls, 5 iterations)
        - sparse: Open areas (35% walls, 3 iterations)
    """
    presets = {
        "cave": {
            "initial_wall_prob": 0.45,
            "num_iterations": 5,
            "birth_limit": 5,
            "death_limit": 4,
        },
        "cavern": {
            "initial_wall_prob": 0.40,
            "num_iterations": 4,
            "birth_limit": 5,
            "death_limit": 3,
        },
        "maze": {
            "initial_wall_prob": 0.50,
            "num_iterations": 6,
            "birth_limit": 5,
            "death_limit": 4,
        },
        "dense": {
            "initial_wall_prob": 0.55,
            "num_iterations": 5,
            "birth_limit": 6,
            "death_limit": 4,
        },
        "sparse": {
            "initial_wall_prob": 0.35,
            "num_iterations": 3,
            "birth_limit": 5,
            "death_limit": 3,
        },
    }

    if style not in presets:
        raise ValueError(f"Unknown style '{style}'. Choose from: {list(presets.keys())}")

    params = presets[style]

    return CellularAutomataConfig(
        rows=rows,
        cols=cols,
        seed=seed,
        **params,
    )
