"""
1D Cartesian Grid for MFG Problems.

This module provides 1D regular grid specification for finite difference methods.
Boundary conditions are managed in boundary_conditions_1d.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfg_pde.geometry.base import CartesianGrid
from mfg_pde.geometry.boundary.fdm_bc_1d import BoundaryConditions  # noqa: TC001
from mfg_pde.geometry.protocol import GeometryType

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class SimpleGrid1D(CartesianGrid):
    """
    1D regular Cartesian grid for finite difference methods.

    This class provides a 1D uniform grid with boundary conditions,
    implementing finite difference operators for MFG solvers.
    """

    def __init__(self, xmin: float, xmax: float, boundary_conditions: BoundaryConditions):
        """
        Initialize 1D domain.

        Args:
            xmin: Left boundary of domain
            xmax: Right boundary of domain
            boundary_conditions: Boundary condition specification
        """
        if xmax <= xmin:
            raise ValueError("xmax must be greater than xmin")

        self.xmin = xmin
        self.xmax = xmax
        self.length = xmax - xmin
        self.boundary_conditions = boundary_conditions

        # Validate boundary conditions
        self.boundary_conditions.validate_values()

        # Cache for grid storage
        self._cached_grid: list[float] | None = None
        self._cached_num_points: int | None = None

    # GeometryProtocol implementation
    @property
    def dimension(self) -> int:
        """Spatial dimension of the geometry (always 1 for SimpleGrid1D)."""
        return 1

    @property
    def geometry_type(self) -> GeometryType:
        """Type of geometry (Cartesian grid for SimpleGrid1D)."""
        return GeometryType.CARTESIAN_GRID

    @property
    def num_spatial_points(self) -> int:
        """Total number of discrete spatial points."""
        if self._cached_num_points is None:
            raise ValueError("Grid not yet created. Call create_grid() first.")
        return self._cached_num_points

    def get_spatial_grid(self) -> np.ndarray:
        """
        Get spatial grid representation.

        Returns:
            numpy array of grid points

        Raises:
            ValueError: If grid has not been created yet
        """
        if self._cached_grid is None:
            raise ValueError("Grid not yet created. Call create_grid() first.")
        return np.array(self._cached_grid)

    def create_grid(self, num_points: int) -> tuple[float, list[float]]:
        """
        Create spatial grid for the domain.

        Args:
            num_points: Number of grid points (including boundaries)

        Returns:
            Tuple of (grid_spacing, grid_points)
        """
        if num_points < 2:
            raise ValueError("num_points must be at least 2")

        dx = self.length / (num_points - 1)
        x_points = [self.xmin + i * dx for i in range(num_points)]

        # Cache grid for GeometryProtocol
        self._cached_grid = x_points
        self._cached_num_points = num_points

        return dx, x_points

    def get_matrix_size(self, num_interior_points: int) -> int:
        """Get system matrix size for this domain's boundary conditions."""
        return self.boundary_conditions.get_matrix_size(num_interior_points)

    def __str__(self) -> str:
        """String representation of grid."""
        return f"SimpleGrid1D([{self.xmin}, {self.xmax}], {self.boundary_conditions})"

    def __repr__(self) -> str:
        """Detailed representation of grid."""
        return f"SimpleGrid1D(xmin={self.xmin}, xmax={self.xmax}, length={self.length}, bc={self.boundary_conditions})"

    # ============================================================================
    # Geometry ABC implementation
    # ============================================================================

    def get_bounds(self) -> tuple[NDArray, NDArray]:
        """
        Return bounding box of 1D domain.

        Returns:
            (min_coords, max_coords) as 1D arrays
        """
        return np.array([self.xmin]), np.array([self.xmax])

    @property
    def coordinates(self) -> list[NDArray]:
        """
        Get coordinate arrays for each dimension (TensorProductGrid compatible).

        Returns:
            List containing single 1D coordinate array [x_coords]

        Raises:
            ValueError: If grid has not been created yet
        """
        if self._cached_grid is None:
            raise ValueError("Grid not yet created. Call create_grid() first.")
        return [np.array(self._cached_grid)]

    def get_multi_index(self, flat_index: int) -> tuple[int]:
        """
        Convert flat index to 1D grid index (trivial for 1D).

        For 1D grids, the flat index is identical to the spatial index.

        Args:
            flat_index: Flat index in [0, num_points)

        Returns:
            Tuple (i,) of grid index (single element tuple for consistency with nD)

        Raises:
            ValueError: If grid not yet created or flat_index out of range
        """
        if self._cached_num_points is None:
            raise ValueError("Grid not yet created. Call create_grid() first.")

        if flat_index < 0 or flat_index >= self._cached_num_points:
            raise ValueError(f"flat_index {flat_index} out of range [0, {self._cached_num_points})")

        return (flat_index,)

    def get_index(self, multi_index: tuple[int, ...]) -> int:
        """
        Convert 1D grid index tuple to flat index (trivial for 1D).

        Inverse of get_multi_index().

        Args:
            multi_index: Tuple (i,) of grid index

        Returns:
            Flat index (equal to i for 1D)

        Raises:
            ValueError: If grid not yet created or index out of range
        """
        if self._cached_num_points is None:
            raise ValueError("Grid not yet created. Call create_grid() first.")

        (i,) = multi_index
        if i < 0 or i >= self._cached_num_points:
            raise ValueError(f"Grid index {i} out of range [0, {self._cached_num_points})")

        return i

    def get_problem_config(self) -> dict:
        """
        Return configuration dict for MFGProblem initialization.

        Returns:
            Dictionary with 1D domain configuration
        """
        if self._cached_num_points is None:
            raise ValueError("Grid not yet created. Call create_grid() first.")

        dx = self.length / (self._cached_num_points - 1)

        return {
            "num_spatial_points": self._cached_num_points,
            "spatial_shape": (self._cached_num_points,),
            "spatial_bounds": [(self.xmin, self.xmax)],
            "spatial_discretization": [self._cached_num_points - 1],  # Number of intervals
            "legacy_1d_attrs": {
                "xmin": self.xmin,
                "xmax": self.xmax,
                "Lx": self.length,
                "Nx": self._cached_num_points - 1,
                "dx": dx,
                "xSpace": self.get_spatial_grid(),
            },
        }

    # ============================================================================
    # CartesianGrid ABC implementation
    # ============================================================================

    def get_grid_spacing(self) -> list[float]:
        """
        Get grid spacing.

        Returns:
            [dx] as single-element list

        Raises:
            ValueError: If grid not yet created
        """
        if self._cached_num_points is None:
            raise ValueError("Grid not yet created. Call create_grid() first.")

        dx = self.length / (self._cached_num_points - 1)
        return [dx]

    def get_grid_shape(self) -> tuple[int, ...]:
        """
        Get grid shape.

        Returns:
            (num_points,) as single-element tuple

        Raises:
            ValueError: If grid not yet created
        """
        if self._cached_num_points is None:
            raise ValueError("Grid not yet created. Call create_grid() first.")

        return (self._cached_num_points,)

    # ============================================================================
    # Solver Operation Interface
    # ============================================================================

    def get_laplacian_operator(self) -> Callable:
        """
        Return finite difference Laplacian operator for 1D.

        Returns:
            Function with signature: (u: NDArray, idx: tuple[int]) -> float
        """
        if self._cached_num_points is None:
            raise ValueError("Grid not yet created. Call create_grid() first.")

        dx = self.length / (self._cached_num_points - 1)

        def laplacian_1d(u: NDArray, idx: tuple[int]) -> float:
            """
            Compute 1D Laplacian: d²u/dx².

            Args:
                u: Solution array of shape (num_points,)
                idx: Grid index as single-element tuple (i,)

            Returns:
                Laplacian value at idx
            """
            i = idx[0]
            n = len(u)

            # Handle boundaries with clamping
            i_plus = min(i + 1, n - 1)
            i_minus = max(i - 1, 0)

            # Central difference: (u[i+1] - 2*u[i] + u[i-1]) / dx²
            laplacian = (u[i_plus] - 2.0 * u[i] + u[i_minus]) / (dx**2)

            return float(laplacian)

        return laplacian_1d

    def get_gradient_operator(self) -> Callable:
        """
        Return finite difference gradient operator for 1D.

        Returns:
            Function with signature: (u: NDArray, idx: tuple[int]) -> NDArray
        """
        if self._cached_num_points is None:
            raise ValueError("Grid not yet created. Call create_grid() first.")

        dx = self.length / (self._cached_num_points - 1)

        def gradient_1d(u: NDArray, idx: tuple[int]) -> NDArray:
            """
            Compute 1D gradient: du/dx.

            Args:
                u: Solution array of shape (num_points,)
                idx: Grid index as single-element tuple (i,)

            Returns:
                Gradient as 1D array [du/dx]
            """
            i = idx[0]
            n = len(u)

            # Handle boundaries with clamping
            i_plus = min(i + 1, n - 1)
            i_minus = max(i - 1, 0)

            # Central difference: (u[i+1] - u[i-1]) / (2*dx)
            gradient = (u[i_plus] - u[i_minus]) / (2.0 * dx)

            return np.array([gradient])

        return gradient_1d

    def get_interpolator(self) -> Callable:
        """
        Return linear interpolator for 1D.

        Returns:
            Function with signature: (u: NDArray, point: NDArray) -> float
        """
        if self._cached_grid is None:
            raise ValueError("Grid not yet created. Call create_grid() first.")

        x_grid = np.array(self._cached_grid)

        def interpolate_1d(u: NDArray, points: NDArray) -> NDArray | float:
            """
            Linear interpolation in 1D.

            Args:
                u: Solution array of shape (num_points,)
                points: Physical coordinate(s) - scalar, array [x], or array of coords (N,)

            Returns:
                Interpolated value(s) - float if single point, array if multiple
            """
            # Handle different input shapes
            if points.ndim == 0:  # Scalar
                x = float(points)
                return float(np.interp(x, x_grid, u))
            elif points.shape == (1,):  # Single point [x]
                x = points[0]
                return float(np.interp(x, x_grid, u))
            else:  # Multiple points (N,) or (N, 1)
                if points.ndim == 2:
                    # (N, 1) case - flatten to (N,)
                    x = points[:, 0]
                else:
                    # (N,) case
                    x = points

                # Vectorized interpolation
                return np.interp(x, x_grid, u)

        return interpolate_1d

    def get_boundary_handler(self):
        """
        Return boundary condition handler for 1D domain.

        Returns:
            BoundaryConditions object for this domain
        """
        return self.boundary_conditions
