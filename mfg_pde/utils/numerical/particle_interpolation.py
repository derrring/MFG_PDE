#!/usr/bin/env python3
"""
Particle Interpolation Utilities

This module provides utilities for converting between particle and grid representations,
essential for hybrid particle-grid methods in MFG problems.

Key Functions:
- interpolate_grid_to_particles: Sample grid values at particle positions (supports arbitrary nD)
- interpolate_particles_to_grid: Estimate density/values on grid from particles (supports arbitrary nD)
- adaptive_bandwidth_selection: Optimal KDE bandwidth selection

Dimension Support:
- Fully dimension-agnostic (1D, 2D, 3D, 4D+)
- Uses scipy.interpolate.RegularGridInterpolator for nD interpolation
- Uses scipy.stats.gaussian_kde for nD kernel density estimation
- Uses np.meshgrid for arbitrary-dimensional grid generation

Use Cases:
- Hybrid particle-grid solvers
- Initial condition generation
- Visualization of particle simulations
- Coupling different solver types

Example:
    >>> from mfg_pde.utils import interpolate_particles_to_grid
    >>>
    >>> # Estimate density on grid from particle positions
    >>> density = interpolate_particles_to_grid(
    ...     particle_positions=particles,
    ...     grid_points=x_grid,
    ...     method="kde"
    ... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Optional dependencies
try:
    from scipy.interpolate import RegularGridInterpolator, interp1d
    from scipy.stats import gaussian_kde

    SCIPY_AVAILABLE = True
except ImportError:
    RegularGridInterpolator = None
    interp1d = None
    gaussian_kde = None
    SCIPY_AVAILABLE = False


# =============================================================================
# GRID TO PARTICLES INTERPOLATION
# =============================================================================


def interpolate_grid_to_particles(
    grid_values: NDArray[np.float64],
    grid_bounds: tuple[tuple[float, float], ...],
    particle_positions: NDArray[np.float64],
    method: Literal["linear", "cubic", "nearest"] = "linear",
) -> NDArray[np.float64]:
    """
    Interpolate values from regular grid to particle positions.

    Supports arbitrary-dimensional grids (1D, 2D, 3D, 4D, ...) with multiple interpolation methods.

    Args:
        grid_values: Grid values
            - 1D: shape (Nx,) or (Nt, Nx)
            - 2D: shape (Nx, Ny) or (Nt, Nx, Ny)
            - 3D: shape (Nx, Ny, Nz) or (Nt, Nx, Ny, Nz)
            - nD: shape (Nx1, Nx2, ..., Nxd) or (Nt, Nx1, Nx2, ..., Nxd)
        grid_bounds: Spatial bounds for each dimension
            - 1D: ((xmin, xmax),)
            - 2D: ((xmin, xmax), (ymin, ymax))
            - 3D: ((xmin, xmax), (ymin, ymax), (zmin, zmax))
            - nD: ((x1_min, x1_max), (x2_min, x2_max), ..., (xd_min, xd_max))
        particle_positions: Particle coordinates, shape (N_particles, d)
        method: Interpolation method
            - "linear": Linear interpolation (fast, C0 continuous)
            - "cubic": Cubic spline (smooth, C2 continuous, 1D only)
            - "nearest": Nearest neighbor (fastest, discontinuous)

    Returns:
        Interpolated values at particle positions, shape (N_particles,)
            If grid_values has time dimension, returns (Nt, N_particles)

    Raises:
        ValueError: If scipy not available or invalid method
        ValueError: If dimensions don't match

    Example:
        >>> # 1D grid to particles
        >>> x_grid = np.linspace(0, 1, 50)
        >>> u_grid = np.sin(2*np.pi*x_grid)
        >>> particles = np.array([[0.25], [0.5], [0.75]])
        >>> u_particles = interpolate_grid_to_particles(
        ...     u_grid, ((0, 1),), particles, method="cubic"
        ... )

        >>> # 2D grid to particles
        >>> x = np.linspace(0, 1, 30)
        >>> y = np.linspace(0, 1, 30)
        >>> X, Y = np.meshgrid(x, y, indexing='ij')
        >>> u_grid = np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
        >>> particles = np.array([[0.3, 0.4], [0.7, 0.2]])
        >>> u_particles = interpolate_grid_to_particles(
        ...     u_grid, ((0, 1), (0, 1)), particles, method="linear"
        ... )
    """
    if not SCIPY_AVAILABLE:
        raise ValueError("scipy is required for grid-to-particle interpolation")

    # Detect dimension
    dimension = len(grid_bounds)

    # Check if grid_values has time dimension
    has_time = grid_values.ndim == dimension + 1

    if has_time:
        # Time-dependent: interpolate each time step
        Nt = grid_values.shape[0]
        N_particles = particle_positions.shape[0]
        result = np.zeros((Nt, N_particles))

        for t in range(Nt):
            result[t] = interpolate_grid_to_particles(grid_values[t], grid_bounds, particle_positions, method=method)
        return result

    # Validate dimensions
    if particle_positions.shape[1] != dimension:
        raise ValueError(
            f"Particle positions dimension ({particle_positions.shape[1]}) doesn't match grid dimension ({dimension})"
        )

    # 1D special case: can use faster interp1d
    if dimension == 1:
        if method == "cubic":
            interpolator = interp1d(
                np.linspace(grid_bounds[0][0], grid_bounds[0][1], grid_values.shape[0]),
                grid_values,
                kind="cubic",
                bounds_error=False,
                fill_value=0.0,
            )
        elif method == "linear":
            interpolator = interp1d(
                np.linspace(grid_bounds[0][0], grid_bounds[0][1], grid_values.shape[0]),
                grid_values,
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
            )
        elif method == "nearest":
            interpolator = interp1d(
                np.linspace(grid_bounds[0][0], grid_bounds[0][1], grid_values.shape[0]),
                grid_values,
                kind="nearest",
                bounds_error=False,
                fill_value=0.0,
            )
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        return interpolator(particle_positions[:, 0])

    # Multi-dimensional: use RegularGridInterpolator
    # Create coordinate arrays for each dimension
    coords = []
    for i, (bmin, bmax) in enumerate(grid_bounds):
        n_points = grid_values.shape[i]
        coords.append(np.linspace(bmin, bmax, n_points))

    # Map method names
    scipy_method = {
        "linear": "linear",
        "nearest": "nearest",
        "cubic": "cubic",  # Note: scipy only supports linear/nearest for nD
    }.get(method, "linear")

    if dimension > 1 and method == "cubic":
        import warnings

        warnings.warn(f"Cubic interpolation not available for {dimension}D, using linear", stacklevel=2)
        scipy_method = "linear"

    interpolator = RegularGridInterpolator(coords, grid_values, method=scipy_method, bounds_error=False, fill_value=0.0)

    return interpolator(particle_positions)


# =============================================================================
# PARTICLES TO GRID INTERPOLATION
# =============================================================================


def interpolate_particles_to_grid(
    particle_positions: NDArray[np.float64],
    grid_points: NDArray[np.float64] | tuple[NDArray[np.float64], ...],
    particle_values: NDArray[np.float64] | None = None,
    method: Literal["kde", "nearest", "histogram"] = "kde",
    bandwidth: float | str = "scott",
    normalize: bool = True,
) -> NDArray[np.float64]:
    """
    Interpolate particle data to regular grid.

    Primary use: Estimate density on grid from particle positions using KDE.
    Also supports: Value interpolation from particles to grid.

    Args:
        particle_positions: Particle coordinates
            - 1D: shape (N_particles,) or (N_particles, 1)
            - nD: shape (N_particles, d)
        grid_points: Grid coordinates
            - 1D: array of shape (Nx,)
            - 2D: tuple (x_coords, y_coords) each shape (Nx,) and (Ny,)
            - 3D: tuple (x_coords, y_coords, z_coords)
            - nD: tuple (x1_coords, x2_coords, ..., xd_coords)
        particle_values: Optional values at particles, shape (N_particles,)
            If None, estimates density (default behavior)
        method: Interpolation method
            - "kde": Gaussian kernel density estimation (smooth, accurate)
            - "nearest": Nearest neighbor (fast, discontinuous)
            - "histogram": Histogram binning (fast, piecewise constant)
        bandwidth: KDE bandwidth (only for method="kde")
            - float: explicit bandwidth value
            - "scott": Scott's rule (default)
            - "silverman": Silverman's rule
        normalize: If True, normalize result to unit integral (for densities)

    Returns:
        Grid values
            - 1D: shape (Nx,)
            - 2D: shape (Nx, Ny)
            - 3D: shape (Nx, Ny, Nz)

    Raises:
        ValueError: If scipy not available (for KDE method)

    Example:
        >>> # Density estimation from particles (most common use)
        >>> particles = np.random.randn(1000) * 0.2 + 0.5  # Gaussian blob
        >>> x_grid = np.linspace(0, 1, 100)
        >>> density = interpolate_particles_to_grid(
        ...     particles.reshape(-1, 1),
        ...     x_grid,
        ...     method="kde"
        ... )

        >>> # Value interpolation from particles
        >>> particle_temps = np.sin(2*np.pi*particles)  # Temperature at particles
        >>> temp_grid = interpolate_particles_to_grid(
        ...     particles.reshape(-1, 1),
        ...     x_grid,
        ...     particle_values=particle_temps,
        ...     method="kde"
        ... )
    """
    # Ensure particle_positions is 2D
    if particle_positions.ndim == 1:
        particle_positions = particle_positions.reshape(-1, 1)

    N_particles, dimension = particle_positions.shape

    # Parse grid_points
    if isinstance(grid_points, tuple):
        # Multi-dimensional grid
        if dimension != len(grid_points):
            raise ValueError(f"Grid dimension ({len(grid_points)}) doesn't match particle dimension ({dimension})")
        grid_shape = tuple(len(g) for g in grid_points)

        # Create meshgrid for evaluation (dimension-agnostic)
        # np.meshgrid works for arbitrary dimensions
        grids = np.meshgrid(*grid_points, indexing="ij")
        eval_points = np.column_stack([grid.ravel() for grid in grids])
    else:
        # 1D grid
        if dimension != 1:
            raise ValueError("For 1D grid, particle_positions must have dimension 1")
        grid_shape = (len(grid_points),)
        eval_points = grid_points.reshape(-1, 1)

    # Compute grid values based on method
    if method == "kde":
        if not SCIPY_AVAILABLE or gaussian_kde is None:
            raise ValueError("scipy is required for KDE method")

        if particle_values is None:
            # Density estimation (standard KDE)
            kde = gaussian_kde(particle_positions.T, bw_method=bandwidth)
            grid_values = kde(eval_points.T)
        else:
            # Weighted KDE for value interpolation
            # Use KDE weights to interpolate values
            kde = gaussian_kde(particle_positions.T, bw_method=bandwidth)
            kde(eval_points.T)

            # Weighted average of particle values
            # This is approximate but reasonable for scattered data
            from scipy.spatial import cKDTree

            tree = cKDTree(particle_positions)
            distances, indices = tree.query(eval_points, k=min(10, N_particles))

            # Gaussian weighting based on distances
            sigma = bandwidth if isinstance(bandwidth, float) else 1.0
            gaussian_weights = np.exp(-0.5 * (distances / sigma) ** 2)
            gaussian_weights /= gaussian_weights.sum(axis=1, keepdims=True)

            grid_values = (gaussian_weights * particle_values[indices]).sum(axis=1)

    elif method == "nearest":
        from scipy.spatial import cKDTree

        tree = cKDTree(particle_positions)
        _, indices = tree.query(eval_points)

        if particle_values is None:
            # Count particles in Voronoi cells (approximate density)
            grid_values = np.bincount(indices, minlength=len(eval_points)).astype(float)
        else:
            grid_values = particle_values[indices]

    elif method == "histogram":
        if dimension == 1:
            counts, _ = np.histogram(particle_positions[:, 0], bins=grid_points, weights=particle_values)
            # Extend to grid points (piecewise constant)
            grid_values = np.concatenate([counts, [counts[-1]]])
        else:
            # Multi-dimensional histogram
            bins = [grid_points[i] for i in range(dimension)]
            counts, _ = np.histogramdd(particle_positions, bins=bins, weights=particle_values)
            # Histogram has shape (n-1, n-1, ..., n-1) for nD
            # Update grid_shape to match histogram output
            grid_shape = tuple(len(grid_points[i]) - 1 for i in range(dimension))
            grid_values = counts.ravel()

    else:
        raise ValueError(f"Unknown method: {method}")

    # Reshape to grid shape
    if len(grid_shape) > 1:
        grid_values = grid_values.reshape(grid_shape)

    # Normalize if requested (for density estimation)
    if normalize and particle_values is None:
        if dimension == 1:
            dx = grid_points[1] - grid_points[0] if len(grid_points) > 1 else 1.0
            total_mass = np.sum(grid_values) * dx
        else:
            # Compute cell volumes for multi-dimensional grids
            cell_volume = np.prod([(grid_points[i][1] - grid_points[i][0]) for i in range(dimension)])
            total_mass = np.sum(grid_values) * cell_volume

        if total_mass > 1e-12:
            grid_values = grid_values / total_mass

    return grid_values


# =============================================================================
# BANDWIDTH SELECTION
# =============================================================================


def adaptive_bandwidth_selection(
    data: NDArray[np.float64], method: Literal["scott", "silverman", "isj"] = "scott"
) -> float:
    """
    Select optimal bandwidth for kernel density estimation.

    Args:
        data: Sample data, shape (N_samples, d) or (N_samples,)
        method: Bandwidth selection method
            - "scott": Scott's rule (1.06 * σ * N^(-1/5))
            - "silverman": Silverman's rule (0.9 * σ * N^(-1/5))
            - "isj": Improved Sheather-Jones (scipy required)

    Returns:
        Optimal bandwidth value

    Example:
        >>> particles = np.random.randn(1000) * 0.2
        >>> bandwidth = adaptive_bandwidth_selection(particles, method="scott")
        >>> kde = gaussian_kde(particles, bw_method=bandwidth)
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    N, d = data.shape

    if method == "scott":
        # Scott's rule: h = σ * N^(-1/(d+4))
        sigma = np.std(data, axis=0)
        return float(np.mean(sigma) * N ** (-1.0 / (d + 4)))

    elif method == "silverman":
        # Silverman's rule: h = (4/(d+2))^(1/(d+4)) * σ * N^(-1/(d+4))
        sigma = np.std(data, axis=0)
        factor = (4.0 / (d + 2)) ** (1.0 / (d + 4))
        return float(factor * np.mean(sigma) * N ** (-1.0 / (d + 4)))

    elif method == "isj":
        # Improved Sheather-Jones (requires scipy)
        if not SCIPY_AVAILABLE:
            raise ValueError("scipy required for ISJ bandwidth selection")

        # For now, fall back to Scott's rule
        # Full ISJ implementation is complex
        import warnings

        warnings.warn("ISJ method not fully implemented, using Scott's rule", stacklevel=2)
        return adaptive_bandwidth_selection(data, method="scott")

    else:
        raise ValueError(f"Unknown bandwidth selection method: {method}")


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "adaptive_bandwidth_selection",
    "interpolate_grid_to_particles",
    "interpolate_particles_to_grid",
]
