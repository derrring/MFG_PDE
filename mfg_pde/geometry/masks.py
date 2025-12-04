"""
Mask generation for boundary condition regions.

This module provides utilities for creating boolean masks that define regions
on grids, supporting mixed boundary conditions with spatially-varying types.

Mask Types:
- **boundary_segment**: Segment on domain boundary edge (e.g., exit door on wall)
- **circle**: Circular region (absorbing region, obstacle)
- **rectangle**: Rectangular region (exit zone, obstacle zone)
- **polygon**: General polygonal region
- **mask_file**: Load from external file (.npy)

Usage:
    # Create mask for exit door on right wall
    exit_mask = boundary_segment_mask(
        grid_shape=(80, 40),
        domain_bounds=np.array([[0, 20], [0, 10]]),
        edge="right",
        range_min=4.0,
        range_max=6.0
    )

    # Create mask from config dict
    config = {"type": "boundary_segment", "edge": "right", "range": [4.0, 6.0]}
    mask = create_mask(config, grid_shape, domain_bounds)

    # Use with BoundaryConditions
    from mfg_pde.geometry.boundary import BCSegment, BCType, mixed_bc
    exit_seg = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0)
    # Apply mask to identify which boundary points belong to exit
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Mask Generators
# =============================================================================


def boundary_segment_mask(
    grid_shape: tuple[int, ...],
    domain_bounds: NDArray[np.floating],
    edge: Literal["left", "right", "bottom", "top", "x_min", "x_max", "y_min", "y_max"],
    range_min: float,
    range_max: float,
) -> NDArray[np.bool_]:
    """
    Create a mask for a segment on a domain boundary edge.

    The mask marks grid points that lie on the specified boundary edge
    within the given coordinate range.

    Args:
        grid_shape: Grid shape as (Ny, Nx) for 2D or (Nz, Ny, Nx) for 3D
        domain_bounds: Domain bounds array of shape (dim, 2) where
                      bounds[i] = [min_i, max_i]
        edge: Boundary edge identifier:
              - "left"/"x_min": x = x_min boundary
              - "right"/"x_max": x = x_max boundary
              - "bottom"/"y_min": y = y_min boundary
              - "top"/"y_max": y = y_max boundary
        range_min: Minimum coordinate along the edge
        range_max: Maximum coordinate along the edge

    Returns:
        Boolean mask of shape grid_shape, True where mask applies

    Example:
        # Exit door on right wall from y=4 to y=6
        mask = boundary_segment_mask(
            grid_shape=(40, 80),
            domain_bounds=np.array([[0, 20], [0, 10]]),
            edge="right",
            range_min=4.0,
            range_max=6.0
        )
    """
    domain_bounds = np.asarray(domain_bounds)

    if len(grid_shape) == 2:
        return _boundary_segment_mask_2d(grid_shape, domain_bounds, edge, range_min, range_max)
    elif len(grid_shape) == 3:
        return _boundary_segment_mask_3d(grid_shape, domain_bounds, edge, range_min, range_max)
    else:
        raise ValueError(f"boundary_segment_mask only supports 2D and 3D grids, got {len(grid_shape)}D")


def _boundary_segment_mask_2d(
    grid_shape: tuple[int, int],
    domain_bounds: NDArray[np.floating],
    edge: str,
    range_min: float,
    range_max: float,
) -> NDArray[np.bool_]:
    """2D implementation of boundary segment mask."""
    Ny, Nx = grid_shape
    x_min, x_max = domain_bounds[0]
    y_min, y_max = domain_bounds[1]

    # Create coordinate arrays
    x_coords = np.linspace(x_min, x_max, Nx)
    y_coords = np.linspace(y_min, y_max, Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")

    mask = np.zeros((Nx, Ny), dtype=bool)

    # Normalize edge names
    edge_map = {
        "left": "x_min",
        "right": "x_max",
        "bottom": "y_min",
        "top": "y_max",
    }
    edge_norm = edge_map.get(edge.lower(), edge.lower())

    # Small tolerance for floating point comparison
    tol = min((x_max - x_min) / (Nx - 1), (y_max - y_min) / (Ny - 1)) * 0.1

    if edge_norm == "x_min":
        # Left boundary: x = x_min, select y in [range_min, range_max]
        mask[0, :] = (Y[0, :] >= range_min - tol) & (Y[0, :] <= range_max + tol)
    elif edge_norm == "x_max":
        # Right boundary: x = x_max, select y in [range_min, range_max]
        mask[-1, :] = (Y[-1, :] >= range_min - tol) & (Y[-1, :] <= range_max + tol)
    elif edge_norm == "y_min":
        # Bottom boundary: y = y_min, select x in [range_min, range_max]
        mask[:, 0] = (X[:, 0] >= range_min - tol) & (X[:, 0] <= range_max + tol)
    elif edge_norm == "y_max":
        # Top boundary: y = y_max, select x in [range_min, range_max]
        mask[:, -1] = (X[:, -1] >= range_min - tol) & (X[:, -1] <= range_max + tol)
    else:
        raise ValueError(f"Unknown edge: {edge}. Use 'left', 'right', 'bottom', 'top'.")

    # Transpose to (Ny, Nx) convention used elsewhere
    return mask.T


def _boundary_segment_mask_3d(
    grid_shape: tuple[int, int, int],
    domain_bounds: NDArray[np.floating],
    edge: str,
    range_min: float,
    range_max: float,
) -> NDArray[np.bool_]:
    """3D implementation of boundary segment mask (face segment)."""
    # For 3D, edge now specifies a face
    # This is a simplified implementation - full 3D would need more parameters
    # Note: grid_shape, domain_bounds, edge, range_min, range_max are intentionally
    # unused as this is not yet implemented
    _ = (grid_shape, domain_bounds, edge, range_min, range_max)
    raise NotImplementedError(
        "3D boundary_segment_mask requires additional parameters for face specification. "
        "Use circle_mask or rectangle_mask for 3D regions."
    )


def circle_mask(
    grid_shape: tuple[int, ...],
    domain_bounds: NDArray[np.floating],
    center: tuple[float, ...],
    radius: float,
) -> NDArray[np.bool_]:
    """
    Create a circular (2D) or spherical (3D) mask.

    Args:
        grid_shape: Grid shape (Ny, Nx) for 2D or (Nz, Ny, Nx) for 3D
        domain_bounds: Domain bounds array of shape (dim, 2)
        center: Center coordinates (x, y) for 2D or (x, y, z) for 3D
        radius: Radius of the circle/sphere

    Returns:
        Boolean mask of shape grid_shape

    Example:
        # Circular absorbing region at (15, 5) with radius 2
        mask = circle_mask(
            grid_shape=(40, 80),
            domain_bounds=np.array([[0, 20], [0, 10]]),
            center=(15.0, 5.0),
            radius=2.0
        )
    """
    domain_bounds = np.asarray(domain_bounds)
    center = np.asarray(center)

    if len(grid_shape) == 2:
        Ny, Nx = grid_shape
        x_coords = np.linspace(domain_bounds[0, 0], domain_bounds[0, 1], Nx)
        y_coords = np.linspace(domain_bounds[1, 0], domain_bounds[1, 1], Ny)
        X, Y = np.meshgrid(x_coords, y_coords, indexing="xy")

        # Distance from center
        dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        return dist <= radius

    elif len(grid_shape) == 3:
        Nz, Ny, Nx = grid_shape
        x_coords = np.linspace(domain_bounds[0, 0], domain_bounds[0, 1], Nx)
        y_coords = np.linspace(domain_bounds[1, 0], domain_bounds[1, 1], Ny)
        z_coords = np.linspace(domain_bounds[2, 0], domain_bounds[2, 1], Nz)
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing="xy")

        dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2)
        return dist <= radius

    else:
        raise ValueError(f"circle_mask only supports 2D and 3D grids, got {len(grid_shape)}D")


def rectangle_mask(
    grid_shape: tuple[int, ...],
    domain_bounds: NDArray[np.floating],
    bounds: NDArray[np.floating] | tuple[tuple[float, float], ...],
) -> NDArray[np.bool_]:
    """
    Create a rectangular mask.

    Args:
        grid_shape: Grid shape (Ny, Nx) for 2D or (Nz, Ny, Nx) for 3D
        domain_bounds: Domain bounds array of shape (dim, 2)
        bounds: Rectangle bounds as array of shape (dim, 2) or tuple of
               ((x_min, x_max), (y_min, y_max), ...)

    Returns:
        Boolean mask of shape grid_shape

    Example:
        # Rectangular exit region
        mask = rectangle_mask(
            grid_shape=(40, 80),
            domain_bounds=np.array([[0, 20], [0, 10]]),
            bounds=np.array([[18, 20], [4, 6]])  # x in [18,20], y in [4,6]
        )
    """
    domain_bounds = np.asarray(domain_bounds)
    bounds = np.asarray(bounds)

    if len(grid_shape) == 2:
        Ny, Nx = grid_shape
        x_coords = np.linspace(domain_bounds[0, 0], domain_bounds[0, 1], Nx)
        y_coords = np.linspace(domain_bounds[1, 0], domain_bounds[1, 1], Ny)
        X, Y = np.meshgrid(x_coords, y_coords, indexing="xy")

        # Check if point is inside rectangle
        mask = (bounds[0, 0] <= X) & (bounds[0, 1] >= X) & (bounds[1, 0] <= Y) & (bounds[1, 1] >= Y)
        return mask

    elif len(grid_shape) == 3:
        Nz, Ny, Nx = grid_shape
        x_coords = np.linspace(domain_bounds[0, 0], domain_bounds[0, 1], Nx)
        y_coords = np.linspace(domain_bounds[1, 0], domain_bounds[1, 1], Ny)
        z_coords = np.linspace(domain_bounds[2, 0], domain_bounds[2, 1], Nz)
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing="xy")

        mask = (
            (bounds[0, 0] <= X)
            & (bounds[0, 1] >= X)
            & (bounds[1, 0] <= Y)
            & (bounds[1, 1] >= Y)
            & (bounds[2, 0] <= Z)
            & (bounds[2, 1] >= Z)
        )
        return mask

    else:
        raise ValueError(f"rectangle_mask only supports 2D and 3D grids, got {len(grid_shape)}D")


def polygon_mask(
    grid_shape: tuple[int, int],
    domain_bounds: NDArray[np.floating],
    vertices: NDArray[np.floating],
) -> NDArray[np.bool_]:
    """
    Create a polygon mask (2D only).

    Uses ray casting algorithm to determine point-in-polygon.

    Args:
        grid_shape: Grid shape (Ny, Nx)
        domain_bounds: Domain bounds array of shape (2, 2)
        vertices: Polygon vertices as array of shape (n_vertices, 2)
                 Vertices should be ordered (clockwise or counter-clockwise)

    Returns:
        Boolean mask of shape grid_shape

    Example:
        # L-shaped region
        vertices = np.array([
            [0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]
        ])
        mask = polygon_mask(grid_shape, domain_bounds, vertices)
    """
    if len(grid_shape) != 2:
        raise ValueError(f"polygon_mask only supports 2D grids, got {len(grid_shape)}D")

    domain_bounds = np.asarray(domain_bounds)
    vertices = np.asarray(vertices)

    Ny, Nx = grid_shape
    x_coords = np.linspace(domain_bounds[0, 0], domain_bounds[0, 1], Nx)
    y_coords = np.linspace(domain_bounds[1, 0], domain_bounds[1, 1], Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing="xy")

    # Flatten for point-in-polygon test
    points = np.column_stack([X.ravel(), Y.ravel()])

    # Ray casting algorithm (vectorized)
    n = len(vertices)
    inside = np.zeros(len(points), dtype=bool)

    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]

        # Check if ray from point crosses edge
        cond1 = (yi > points[:, 1]) != (yj > points[:, 1])
        slope = (xj - xi) / (yj - yi + 1e-15)
        cond2 = points[:, 0] < (slope * (points[:, 1] - yi) + xi)

        inside ^= cond1 & cond2
        j = i

    return inside.reshape(Ny, Nx)


def load_mask(
    path: str | Path,
    expected_shape: tuple[int, ...] | None = None,
) -> NDArray[np.bool_]:
    """
    Load a mask from a numpy file.

    Args:
        path: Path to .npy or .npz file
        expected_shape: If provided, validate loaded mask has this shape

    Returns:
        Boolean mask array

    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If shape mismatch or invalid format
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")

    if path.suffix == ".npz":
        data = np.load(path)
        # Look for 'mask' key, or use first array
        if "mask" in data:
            mask = data["mask"]
        else:
            mask = data[data.files[0]]
    else:
        mask = np.load(path)

    # Convert to boolean if needed
    mask = mask.astype(bool)

    if expected_shape is not None and mask.shape != expected_shape:
        raise ValueError(f"Mask shape mismatch: expected {expected_shape}, got {mask.shape}")

    return mask


def save_mask(
    mask: NDArray[np.bool_],
    path: str | Path,
    compressed: bool = True,
) -> None:
    """
    Save a mask to a numpy file.

    Args:
        mask: Boolean mask array
        path: Output path (.npy or .npz)
        compressed: If True and path is .npz, use compression
    """
    path = Path(path)

    if path.suffix == ".npz":
        if compressed:
            np.savez_compressed(path, mask=mask)
        else:
            np.savez(path, mask=mask)
    else:
        np.save(path, mask)


# =============================================================================
# Factory Function
# =============================================================================


def create_mask(
    config: dict,
    grid_shape: tuple[int, ...],
    domain_bounds: NDArray[np.floating],
) -> NDArray[np.bool_]:
    """
    Create a mask from a configuration dictionary.

    Factory function that dispatches to the appropriate mask generator
    based on the 'type' field in the config.

    Args:
        config: Configuration dictionary with 'type' field and type-specific params
        grid_shape: Grid shape
        domain_bounds: Domain bounds array

    Returns:
        Boolean mask array

    Config Examples:
        # Boundary segment (exit door)
        {
            "type": "boundary_segment",
            "edge": "right",
            "range": [4.0, 6.0]
        }

        # Circular region
        {
            "type": "circle",
            "center": [15.0, 5.0],
            "radius": 2.0
        }

        # Rectangular region
        {
            "type": "rectangle",
            "bounds": [[18, 20], [4, 6]]
        }

        # From file
        {
            "type": "mask_file",
            "path": "exit_mask.npy"
        }
    """
    mask_type = config.get("type", "").lower()

    if mask_type == "boundary_segment":
        # Extract range as [min, max] or separate range_min, range_max
        if "range" in config:
            range_min, range_max = config["range"]
        else:
            range_min = config.get("range_min", 0.0)
            range_max = config.get("range_max", 1.0)

        return boundary_segment_mask(
            grid_shape=grid_shape,
            domain_bounds=domain_bounds,
            edge=config["edge"],
            range_min=range_min,
            range_max=range_max,
        )

    elif mask_type == "circle":
        return circle_mask(
            grid_shape=grid_shape,
            domain_bounds=domain_bounds,
            center=tuple(config["center"]),
            radius=config["radius"],
        )

    elif mask_type == "rectangle":
        bounds = np.asarray(config["bounds"])
        return rectangle_mask(
            grid_shape=grid_shape,
            domain_bounds=domain_bounds,
            bounds=bounds,
        )

    elif mask_type == "polygon":
        vertices = np.asarray(config["vertices"])
        return polygon_mask(
            grid_shape=grid_shape,
            domain_bounds=domain_bounds,
            vertices=vertices,
        )

    elif mask_type in ["mask_file", "file"]:
        mask = load_mask(config["path"])
        # Optionally resize if shapes don't match
        if mask.shape != grid_shape:
            from scipy.ndimage import zoom

            # Compute zoom factors
            factors = tuple(gs / ms for gs, ms in zip(grid_shape, mask.shape, strict=True))
            mask = zoom(mask.astype(float), factors, order=0) > 0.5
        return mask

    else:
        raise ValueError(
            f"Unknown mask type: '{mask_type}'. "
            f"Supported types: boundary_segment, circle, rectangle, polygon, mask_file"
        )


# =============================================================================
# Mask Utilities
# =============================================================================


def combine_masks(
    masks: list[NDArray[np.bool_]],
    operation: Literal["union", "intersection"] = "union",
) -> NDArray[np.bool_]:
    """
    Combine multiple masks using boolean operations.

    Args:
        masks: List of boolean mask arrays (same shape)
        operation: "union" (OR) or "intersection" (AND)

    Returns:
        Combined mask
    """
    if not masks:
        raise ValueError("At least one mask required")

    result = masks[0].copy()

    for mask in masks[1:]:
        if mask.shape != result.shape:
            raise ValueError(f"Shape mismatch: {result.shape} vs {mask.shape}")

        if operation == "union":
            result = result | mask
        else:
            result = result & mask

    return result


def invert_mask(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Invert a mask (logical NOT)."""
    return ~mask


def mask_to_indices(
    mask: NDArray[np.bool_],
) -> tuple[NDArray[np.int_], ...]:
    """
    Convert a mask to index arrays.

    Args:
        mask: Boolean mask

    Returns:
        Tuple of index arrays (one per dimension)
    """
    return np.where(mask)


def indices_to_mask(
    indices: tuple[NDArray[np.int_], ...],
    shape: tuple[int, ...],
) -> NDArray[np.bool_]:
    """
    Convert index arrays to a mask.

    Args:
        indices: Tuple of index arrays
        shape: Output shape

    Returns:
        Boolean mask
    """
    mask = np.zeros(shape, dtype=bool)
    mask[indices] = True
    return mask


def get_boundary_mask(
    grid_shape: tuple[int, ...],
    which: Literal["all", "left", "right", "bottom", "top", "interior"] = "all",
) -> NDArray[np.bool_]:
    """
    Get a mask for boundary points on a grid.

    Args:
        grid_shape: Grid shape
        which: Which boundary to mask:
              - "all": All boundary points
              - "left", "right", "bottom", "top": Specific edge (2D)
              - "interior": Interior points (not on boundary)

    Returns:
        Boolean mask
    """
    mask = np.zeros(grid_shape, dtype=bool)

    if len(grid_shape) == 2:
        # Validate 2D grid (individual dimensions not needed for slicing)
        if len(grid_shape) != 2:
            raise ValueError(f"Expected 2D grid, got {len(grid_shape)}D")

        if which == "all":
            mask[0, :] = True  # Bottom
            mask[-1, :] = True  # Top
            mask[:, 0] = True  # Left
            mask[:, -1] = True  # Right
        elif which == "left":
            mask[:, 0] = True
        elif which == "right":
            mask[:, -1] = True
        elif which == "bottom":
            mask[0, :] = True
        elif which == "top":
            mask[-1, :] = True
        elif which == "interior":
            mask[1:-1, 1:-1] = True
        else:
            raise ValueError(f"Unknown boundary: {which}")

    elif len(grid_shape) == 3:
        # Unpack to validate but don't need individual values
        if len(grid_shape) != 3:
            raise ValueError(f"Expected 3D grid, got {len(grid_shape)}D")

        if which == "all":
            # All 6 faces
            mask[0, :, :] = True
            mask[-1, :, :] = True
            mask[:, 0, :] = True
            mask[:, -1, :] = True
            mask[:, :, 0] = True
            mask[:, :, -1] = True
        elif which == "interior":
            mask[1:-1, 1:-1, 1:-1] = True
        else:
            raise ValueError("3D boundary selection by name not supported. Use 'all' or 'interior'.")

    else:
        raise ValueError(f"get_boundary_mask supports 2D and 3D, got {len(grid_shape)}D")

    return mask


__all__ = [
    # Mask generators
    "boundary_segment_mask",
    "circle_mask",
    "rectangle_mask",
    "polygon_mask",
    # File I/O
    "load_mask",
    "save_mask",
    # Factory
    "create_mask",
    # Utilities
    "combine_masks",
    "invert_mask",
    "mask_to_indices",
    "indices_to_mask",
    "get_boundary_mask",
]
