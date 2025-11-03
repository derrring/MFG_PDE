"""
Maze Generation Utilities for MFG Environments

Provides post-processing utilities for maze generation:
- Wall smoothing for organic algorithms (CA, Voronoi)
- Adaptive door width for connectivity
- Connectivity analysis and repair

These utilities enhance maze quality by:
- Reducing zigzag artifacts in rasterized boundaries
- Creating realistic door widths based on zone sizes
- Ensuring robust multi-zone connectivity

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Optional scipy for advanced smoothing
try:
    from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def smooth_walls_morphological(
    maze: NDArray,
    iterations: int = 1,
    method: str = "opening",
) -> NDArray:
    """
    Smooth wall boundaries using morphological operations.

    Reduces zigzag artifacts in organic mazes (CA, Voronoi) by applying
    erosion/dilation operations. This is the recommended method for
    cleaning up rasterized boundaries.

    Args:
        maze: Input maze (1 = wall, 0 = open)
        iterations: Number of smoothing iterations (1-3 typical)
        method: Smoothing method - "opening" (default), "closing", or "both"

    Returns:
        Smoothed maze

    Methods:
        - "opening": Erode then dilate (removes protrusions)
        - "closing": Dilate then erode (fills gaps)
        - "both": Apply both operations

    Example:
        >>> from mfg_pde.alg.reinforcement.environments import CellularAutomataGenerator
        >>> from mfg_pde.alg.reinforcement.environments.maze_utils import smooth_walls_morphological
        >>>
        >>> maze = generator.generate()
        >>> smoothed = smooth_walls_morphological(maze, iterations=2)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for morphological smoothing. Install with: pip install scipy")

    walls = maze == 1

    if method == "opening":
        # Remove small protrusions
        smoothed = binary_erosion(walls, iterations=iterations)
        smoothed = binary_dilation(smoothed, iterations=iterations)
    elif method == "closing":
        # Fill small gaps
        smoothed = binary_dilation(walls, iterations=iterations)
        smoothed = binary_erosion(smoothed, iterations=iterations)
    elif method == "both":
        # Apply both for maximum smoothing
        smoothed = binary_erosion(walls, iterations=iterations)
        smoothed = binary_dilation(smoothed, iterations=iterations)
        smoothed = binary_dilation(smoothed, iterations=iterations)
        smoothed = binary_erosion(smoothed, iterations=iterations)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: opening, closing, both")

    return smoothed.astype(np.int32)


def smooth_walls_gaussian(
    maze: NDArray,
    sigma: float = 1.0,
    threshold: float = 0.5,
) -> NDArray:
    """
    Smooth wall boundaries using Gaussian blur and re-thresholding.

    Creates very smooth boundaries by blurring then re-thresholding.
    Good for creating aesthetically pleasing mazes, but may alter
    topology more than morphological methods.

    Args:
        maze: Input maze (1 = wall, 0 = open)
        sigma: Gaussian kernel standard deviation (0.5-2.0 typical)
        threshold: Re-threshold value (0.5 default)

    Returns:
        Smoothed maze

    Example:
        >>> smoothed = smooth_walls_gaussian(maze, sigma=1.5)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for Gaussian smoothing. Install with: pip install scipy")

    # Blur the maze
    blurred = gaussian_filter(maze.astype(float), sigma=sigma)

    # Re-threshold
    return (blurred > threshold).astype(np.int32)


def find_disconnected_regions(maze: NDArray) -> list[set[tuple[int, int]]]:
    """
    Find all disconnected open regions in a maze.

    Args:
        maze: Input maze (1 = wall, 0 = open)

    Returns:
        List of sets, each containing (row, col) positions in a region

    Example:
        >>> regions = find_disconnected_regions(maze)
        >>> print(f"Found {len(regions)} disconnected regions")
        >>> print(f"Largest region has {len(max(regions, key=len))} cells")
    """
    rows, cols = maze.shape
    visited = np.zeros_like(maze, dtype=bool)
    regions = []

    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == 0 and not visited[r, c]:
                # Found new region, flood fill
                region = _flood_fill(maze, r, c, visited)
                regions.append(region)

    return regions


def _flood_fill(maze: NDArray, start_r: int, start_c: int, visited: NDArray) -> set[tuple[int, int]]:
    """
    Flood fill to find connected region.

    Args:
        maze: Input maze
        start_r: Starting row
        start_c: Starting column
        visited: Boolean array tracking visited cells

    Returns:
        Set of positions in this region
    """
    rows, cols = maze.shape
    region = set()
    stack = [(start_r, start_c)]

    while stack:
        r, c = stack.pop()

        if r < 0 or r >= rows or c < 0 or c >= cols:
            continue
        if visited[r, c] or maze[r, c] == 1:
            continue

        visited[r, c] = True
        region.add((r, c))

        # Add 4-connected neighbors
        stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])

    return region


def find_region_boundary(
    region_a: set[tuple[int, int]],
    region_b: set[tuple[int, int]],
    maze: NDArray,
) -> list[tuple[int, int]]:
    """
    Find wall cells on boundary between two regions.

    Args:
        region_a: First region (set of (row, col) positions)
        region_b: Second region (set of (row, col) positions)
        maze: Input maze (1 = wall, 0 = open)

    Returns:
        List of (row, col) positions of wall cells on boundary

    Example:
        >>> regions = find_disconnected_regions(maze)
        >>> boundary = find_region_boundary(regions[0], regions[1], maze)
        >>> # boundary contains wall cells separating the two regions
    """
    rows, cols = maze.shape
    boundary = []

    # For each cell in region A, check if it's adjacent to region B through a wall
    for r, c in region_a:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols:
                # If neighbor is a wall
                if maze[nr, nc] == 1:
                    # Check if wall is adjacent to region B
                    for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr2, nc2 = nr + dr2, nc + dc2
                        if (nr2, nc2) in region_b:
                            boundary.append((nr, nc))
                            break

    return list(set(boundary))  # Remove duplicates


def compute_adaptive_door_width(
    region_a_size: int,
    region_b_size: int,
    min_width: int = 1,
    max_width: int = 5,
) -> int:
    """
    Compute adaptive door width based on region sizes.

    Larger zones get wider doors for better flow dynamics in MFG scenarios.

    Args:
        region_a_size: Number of cells in first region
        region_b_size: Number of cells in second region
        min_width: Minimum door width (default 1)
        max_width: Maximum door width (default 5)

    Returns:
        Recommended door width (in cells)

    Formula:
        width = min(max_width, max(min_width, sqrt(avg_size) / 10))

    Example:
        >>> width = compute_adaptive_door_width(1000, 800)
        >>> print(f"Recommended door width: {width} cells")
    """
    avg_size = (region_a_size + region_b_size) / 2
    width = int(np.sqrt(avg_size) / 10)
    return min(max_width, max(min_width, width))


def connect_regions_adaptive(
    maze: NDArray,
    min_door_width: int = 1,
    max_door_width: int = 5,
    max_iterations: int = 100,
) -> NDArray:
    """
    Connect all disconnected regions with adaptive door widths.

    Automatically finds disconnected regions and creates doors between them.
    Door width adapts to region sizes for realistic flow dynamics.

    Args:
        maze: Input maze (1 = wall, 0 = open)
        min_door_width: Minimum door width (default 1)
        max_door_width: Maximum door width (default 5)
        max_iterations: Maximum connection attempts (default 100)

    Returns:
        Connected maze

    Example:
        >>> from mfg_pde.alg.reinforcement.environments import CellularAutomataGenerator
        >>> from mfg_pde.alg.reinforcement.environments.maze_utils import connect_regions_adaptive
        >>>
        >>> config = CellularAutomataConfig(rows=60, cols=60, ensure_connectivity=False)
        >>> maze = generator.generate()
        >>> connected = connect_regions_adaptive(maze, min_door_width=2, max_door_width=4)
    """
    maze_copy = maze.copy()

    for _ in range(max_iterations):
        regions = find_disconnected_regions(maze_copy)

        if len(regions) <= 1:
            break  # Fully connected

        # Sort regions by size (largest first)
        regions.sort(key=len, reverse=True)

        # Connect largest region to next largest
        region_a = regions[0]
        region_b = regions[1]

        # Find boundary
        boundary = find_region_boundary(region_a, region_b, maze_copy)

        if not boundary:
            # No direct boundary - find closest points between regions
            min_dist = float("inf")
            closest_pair = None

            for pos_a in list(region_a)[:: max(1, len(region_a) // 20)]:  # Sample to avoid O(n²)
                for pos_b in list(region_b)[:: max(1, len(region_b) // 20)]:
                    dist = abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (pos_a, pos_b)

            if closest_pair:
                # Carve corridor between closest points
                maze_copy = _carve_corridor(maze_copy, closest_pair[0], closest_pair[1], width=min_door_width)
            else:
                break  # No way to connect

        else:
            # Compute adaptive door width
            door_width = compute_adaptive_door_width(
                len(region_a),
                len(region_b),
                min_width=min_door_width,
                max_width=max_door_width,
            )

            # Create door at a random boundary location
            door_center = boundary[len(boundary) // 2]  # Use middle boundary point
            maze_copy = _carve_door(maze_copy, door_center, width=door_width)

    return maze_copy


def _carve_door(maze: NDArray, center: tuple[int, int], width: int) -> NDArray:
    """
    Carve a door of specified width centered at a position.

    Args:
        maze: Input maze
        center: (row, col) center of door
        width: Door width in cells

    Returns:
        Maze with door carved
    """
    rows, cols = maze.shape
    r, c = center
    half_width = width // 2

    for dr in range(-half_width, half_width + 1):
        for dc in range(-half_width, half_width + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                maze[nr, nc] = 0

    return maze


def _carve_corridor(maze: NDArray, start: tuple[int, int], end: tuple[int, int], width: int) -> NDArray:
    """
    Carve a corridor between two points using Bresenham's line algorithm.

    Args:
        maze: Input maze
        start: (row, col) starting position
        end: (row, col) ending position
        width: Corridor width in cells

    Returns:
        Maze with corridor carved
    """
    rows, cols = maze.shape
    r0, c0 = start
    r1, c1 = end

    # Bresenham's line algorithm
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0

    while True:
        # Carve corridor at current position
        half_width = width // 2
        for dr_off in range(-half_width, half_width + 1):
            for dc_off in range(-half_width, half_width + 1):
                nr, nc = r + dr_off, c + dc_off
                if 0 <= nr < rows and 0 <= nc < cols:
                    maze[nr, nc] = 0

        if r == r1 and c == c1:
            break

        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

    return maze


def analyze_maze_connectivity(maze: NDArray) -> dict:
    """
    Analyze connectivity properties of a maze.

    Args:
        maze: Input maze (1 = wall, 0 = open)

    Returns:
        Dictionary with connectivity analysis:
        - num_regions: Number of disconnected regions
        - largest_region_size: Size of largest region
        - total_open_space: Total open cells
        - connectivity_ratio: Fraction of open space in largest region
        - is_connected: True if maze has single connected component

    Example:
        >>> from mfg_pde.alg.reinforcement.environments.maze_utils import analyze_maze_connectivity
        >>>
        >>> analysis = analyze_maze_connectivity(maze)
        >>> print(f"Regions: {analysis['num_regions']}")
        >>> print(f"Connected: {analysis['is_connected']}")
        >>> print(f"Connectivity: {analysis['connectivity_ratio']:.1%}")
    """
    regions = find_disconnected_regions(maze)
    total_open = np.sum(maze == 0)

    if not regions:
        return {
            "num_regions": 0,
            "largest_region_size": 0,
            "total_open_space": total_open,
            "connectivity_ratio": 0.0,
            "is_connected": False,
        }

    largest_size = len(max(regions, key=len))

    return {
        "num_regions": len(regions),
        "largest_region_size": largest_size,
        "total_open_space": total_open,
        "connectivity_ratio": largest_size / total_open if total_open > 0 else 0.0,
        "is_connected": len(regions) == 1,
    }
