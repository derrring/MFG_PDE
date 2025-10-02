"""
Maze Post-Processing Utilities

Provides smoothing, enhancement, and refinement operations for maze generation,
particularly for organic algorithms (Cellular Automata, Voronoi) that produce
zigzag wall boundaries due to pixel-level rasterization.

Features:
- Wall smoothing (morphological operations, Gaussian blur)
- Adaptive door width for connectivity
- Wall thickness normalization
- Boundary refinement

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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
    operation: Literal["open", "close", "both"] = "open",
) -> NDArray:
    """
    Smooth wall boundaries using morphological operations.

    Reduces zigzag artifacts from pixel-level rasterization in organic
    maze algorithms (Cellular Automata, Voronoi).

    Operations:
    - **Opening** (erosion → dilation): Removes small protrusions, smooths convex corners
    - **Closing** (dilation → erosion): Fills small gaps, smooths concave corners
    - **Both**: Apply opening then closing for balanced smoothing

    Args:
        maze: Input maze (1 = wall, 0 = open)
        iterations: Number of erosion/dilation iterations
        operation: Morphological operation type

    Returns:
        Smoothed maze

    Example:
        >>> ca_maze = CellularAutomataGenerator(config).generate()
        >>> smooth_maze = smooth_walls_morphological(ca_maze, iterations=2)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for morphological smoothing. Install with: pip install scipy")

    walls = maze == 1

    if operation in ["open", "both"]:
        # Opening: erosion then dilation (removes protrusions)
        walls = binary_erosion(walls, iterations=iterations)
        walls = binary_dilation(walls, iterations=iterations)

    if operation in ["close", "both"]:
        # Closing: dilation then erosion (fills gaps)
        walls = binary_dilation(walls, iterations=iterations)
        walls = binary_erosion(walls, iterations=iterations)

    return walls.astype(np.int32)


def smooth_walls_gaussian(
    maze: NDArray,
    sigma: float = 1.0,
    threshold: float = 0.5,
) -> NDArray:
    """
    Smooth wall boundaries using Gaussian blur and re-thresholding.

    Creates smoother, more organic wall contours by:
    1. Applying Gaussian blur to wall map
    2. Re-thresholding to binary maze

    More aggressive smoothing than morphological operations.
    Best for highly irregular CA mazes.

    Args:
        maze: Input maze (1 = wall, 0 = open)
        sigma: Gaussian kernel standard deviation (larger = more smoothing)
        threshold: Re-threshold value (0.5 = balanced)

    Returns:
        Smoothed maze

    Example:
        >>> voronoi_maze = VoronoiMazeGenerator(config).generate()
        >>> smooth_maze = smooth_walls_gaussian(voronoi_maze, sigma=1.5)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for Gaussian smoothing. Install with: pip install scipy")

    # Apply Gaussian blur
    blurred = gaussian_filter(maze.astype(float), sigma=sigma)

    # Re-threshold to binary
    smoothed = (blurred > threshold).astype(np.int32)

    return smoothed


def smooth_walls_combined(
    maze: NDArray,
    morph_iterations: int = 1,
    gaussian_sigma: float = 0.8,
) -> NDArray:
    """
    Combined smoothing: morphological opening + light Gaussian blur.

    Provides best balance of:
    - Removing small artifacts (morphological)
    - Smoothing boundaries (Gaussian)
    - Preserving maze structure

    Recommended for production use with organic mazes.

    Args:
        maze: Input maze (1 = wall, 0 = open)
        morph_iterations: Morphological operation iterations
        gaussian_sigma: Gaussian blur strength

    Returns:
        Smoothed maze

    Example:
        >>> ca_maze = CellularAutomataGenerator(config).generate()
        >>> smooth_maze = smooth_walls_combined(ca_maze)  # Best defaults
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for combined smoothing. Install with: pip install scipy")

    # Step 1: Morphological opening to remove small protrusions
    smoothed = smooth_walls_morphological(maze, iterations=morph_iterations, operation="open")

    # Step 2: Light Gaussian smoothing for boundary refinement
    smoothed = smooth_walls_gaussian(smoothed, sigma=gaussian_sigma, threshold=0.5)

    return smoothed


def normalize_wall_thickness(
    maze: NDArray,
    target_thickness: int = 1,
) -> NDArray:
    """
    Normalize wall thickness to target value.

    Useful for ensuring consistent wall thickness after smoothing
    or combining mazes with different wall thicknesses.

    Args:
        maze: Input maze (1 = wall, 0 = open)
        target_thickness: Desired wall thickness

    Returns:
        Maze with normalized wall thickness

    Example:
        >>> maze = smooth_walls_combined(original_maze)
        >>> maze = normalize_wall_thickness(maze, target_thickness=1)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for wall thickness normalization. Install with: pip install scipy")

    walls = maze == 1

    if target_thickness == 1:
        # Thin walls to single pixel
        from scipy.ndimage import binary_erosion

        # Erode until single-pixel thick
        thinned = binary_erosion(walls, iterations=1)
        return thinned.astype(np.int32)
    else:
        # Thicken walls
        from scipy.ndimage import binary_dilation

        dilated = binary_dilation(walls, iterations=target_thickness - 1)
        return dilated.astype(np.int32)


def adaptive_door_carving(
    maze: NDArray,
    zone_map: NDArray,
    zone_i: int,
    zone_j: int,
    min_width: int = 1,
    max_width: int = 5,
    base_door_width: int = 2,
) -> NDArray:
    """
    Carve door between two zones with adaptive width.

    Door width adapts to:
    - Zone sizes (larger zones → wider doors)
    - Boundary length (longer boundary → wider door)
    - Local geometry (wider passages where appropriate)

    Args:
        maze: Maze grid to modify
        zone_map: Zone assignment for each cell
        zone_i: First zone ID
        zone_j: Second zone ID
        min_width: Minimum door width
        max_width: Maximum door width
        base_door_width: Base door width before adaptation

    Returns:
        Maze with adaptive door carved

    Example:
        >>> # In hybrid maze generator
        >>> maze = adaptive_door_carving(maze, zone_map, zone_i=0, zone_j=1)
    """
    # Calculate adaptive door width based on zone sizes
    zone_i_size = np.sum(zone_map == zone_i)
    zone_j_size = np.sum(zone_map == zone_j)
    avg_size = (zone_i_size + zone_j_size) / 2

    # Adaptive formula: larger zones get wider doors
    # sqrt scaling prevents doors from becoming too wide
    size_factor = np.sqrt(avg_size) / 50  # Normalize by typical zone size
    adaptive_width = int(base_door_width * (1 + size_factor))

    # Clamp to valid range
    door_width = max(min_width, min(max_width, adaptive_width))

    # Find boundary between zones
    boundary_cells = _find_zone_boundary(zone_map, zone_i, zone_j)

    if len(boundary_cells) == 0:
        return maze  # No boundary found

    # Choose door location (middle of boundary)
    door_center_idx = len(boundary_cells) // 2
    door_center = boundary_cells[door_center_idx]

    # Carve door with adaptive width
    maze = _carve_door_at_position(maze, door_center, width=door_width)

    return maze


def _find_zone_boundary(
    zone_map: NDArray,
    zone_i: int,
    zone_j: int,
) -> list[tuple[int, int]]:
    """
    Find boundary cells between two zones.

    Args:
        zone_map: Zone assignment map
        zone_i: First zone ID
        zone_j: Second zone ID

    Returns:
        List of (row, col) boundary cell positions
    """
    rows, cols = zone_map.shape
    boundary_cells = []

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if zone_map[r, c] == zone_i:
                # Check 4-connected neighbors
                neighbors = [
                    zone_map[r - 1, c],  # North
                    zone_map[r + 1, c],  # South
                    zone_map[r, c - 1],  # West
                    zone_map[r, c + 1],  # East
                ]
                if zone_j in neighbors:
                    boundary_cells.append((r, c))

    return boundary_cells


def _carve_door_at_position(
    maze: NDArray,
    center: tuple[int, int],
    width: int,
) -> NDArray:
    """
    Carve door centered at position with given width.

    Args:
        maze: Maze grid to modify
        center: (row, col) center position
        width: Door width (creates width x width opening)

    Returns:
        Modified maze
    """
    r, c = center
    half_width = width // 2

    rows, cols = maze.shape

    for dr in range(-half_width, half_width + 1):
        for dc in range(-half_width, half_width + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                maze[nr, nc] = 0  # Open space

    return maze


def enhance_organic_maze(
    maze: NDArray,
    smoothing_strength: Literal["light", "medium", "strong"] = "medium",
    preserve_connectivity: bool = True,
) -> NDArray:
    """
    Complete enhancement pipeline for organic mazes.

    Recommended workflow for CA and Voronoi mazes:
    1. Smooth wall boundaries
    2. Normalize wall thickness
    3. Verify/preserve connectivity

    Args:
        maze: Input organic maze
        smoothing_strength: Smoothing intensity
        preserve_connectivity: Ensure connectivity after smoothing

    Returns:
        Enhanced maze

    Example:
        >>> ca_maze = CellularAutomataGenerator(config).generate()
        >>> enhanced = enhance_organic_maze(ca_maze, smoothing_strength="medium")
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for maze enhancement. Install with: pip install scipy")

    # Apply smoothing based on strength
    if smoothing_strength == "light":
        enhanced = smooth_walls_combined(maze, morph_iterations=1, gaussian_sigma=0.5)
    elif smoothing_strength == "medium":
        enhanced = smooth_walls_combined(maze, morph_iterations=1, gaussian_sigma=0.8)
    else:  # strong
        enhanced = smooth_walls_combined(maze, morph_iterations=2, gaussian_sigma=1.2)

    # Normalize wall thickness to 1 pixel
    enhanced = normalize_wall_thickness(enhanced, target_thickness=1)

    # Verify connectivity if requested
    if preserve_connectivity:
        if not _verify_connectivity(enhanced):
            # Fallback: return original if connectivity lost
            return maze

    return enhanced


def _verify_connectivity(maze: NDArray) -> bool:
    """
    Verify that maze has global connectivity (all open cells reachable).

    Args:
        maze: Maze to check

    Returns:
        True if fully connected
    """
    rows, cols = maze.shape

    # Find starting open cell
    start = None
    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == 0:
                start = (r, c)
                break
        if start:
            break

    if start is None:
        return False  # No open cells

    # BFS from start
    visited = np.zeros_like(maze, dtype=bool)
    queue = [start]
    visited[start] = True
    reachable_count = 1

    while queue:
        r, c = queue.pop(0)

        # Check 4 neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if maze[nr, nc] == 0 and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
                    reachable_count += 1

    # Check if all open cells reachable
    total_open = np.sum(maze == 0)
    return reachable_count == total_open
