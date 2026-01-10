"""
Tensor product grid infrastructure for efficient 2D/3D structured grids.

This module provides memory-optimized tensor product grids for multi-dimensional
MFG problems, enabling efficient storage and computation on regular grid structures.

Mathematical Background:
    A tensor product grid in d dimensions is formed by:
        Ω = [x₁_min, x₁_max] × [x₂_min, x₂_max] × ... × [xₐ_min, xₐ_max]

    Grid points: (x₁ᵢ, x₂ⱼ, ..., xₐₖ) where i,j,...,k index along each dimension

    Storage efficiency: O(∑Nᵢ) instead of O(∏Nᵢ) for structured grids

References:
    - Strikwerda (2004): Finite Difference Schemes and Partial Differential Equations
    - LeVeque (2007): Finite Difference Methods for Ordinary and PDEs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfg_pde.geometry.base import CartesianGrid
from mfg_pde.geometry.protocol import GeometryType

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from mfg_pde.geometry.boundary.conditions import BoundaryConditions


class TensorProductGrid(CartesianGrid):
    """
    Tensor product grid for multi-dimensional structured domains.

    Provides memory-efficient representation of d-dimensional regular grids
    using 1D coordinate arrays. Supports uniform and non-uniform spacing.

    Supports arbitrary dimensions, though O(N^d) complexity limits practical
    use to d≤3 for dense grids. For high dimensions (d>3), consider meshfree methods.

    Attributes:
        dimension: Spatial dimension (any positive integer)
        bounds: List of (min, max) tuples for each dimension
        Nx: Number of intervals along each dimension (list)
        Nx_points: Number of grid points along each dimension (Nx + 1)
        num_points: Alias for Nx_points (deprecated, use Nx_points instead)
        coordinates: List of 1D coordinate arrays
        spacing: Grid spacing along each dimension (if uniform)
        is_uniform: Whether grid has uniform spacing in each dimension
        boundary_conditions: Spatial BC specification (SSOT for MFG solvers)

    Example:
        >>> # Create 2D grid: [0,10] × [0,5] with 100×50 intervals (101×51 points)
        >>> grid = TensorProductGrid(
        ...     dimension=2,
        ...     bounds=[(0.0, 10.0), (0.0, 5.0)],
        ...     Nx=[100, 50]  # intervals
        ... )
        >>> grid.Nx          # [100, 50] - intervals
        >>> grid.Nx_points   # [101, 51] - points

        >>> # Alternative: specify points directly
        >>> grid = TensorProductGrid(
        ...     dimension=2,
        ...     bounds=[(0.0, 10.0), (0.0, 5.0)],
        ...     Nx_points=[101, 51]  # points
        ... )

        >>> x, y = grid.meshgrid()  # Get coordinate matrices
        >>> flat_points = grid.flatten()  # Get all grid points as (N,2) array

        >>> # Create 4D grid (with performance warning)
        >>> grid_4d = TensorProductGrid(
        ...     dimension=4,
        ...     bounds=[(0.0, 1.0)] * 4,
        ...     Nx=[9] * 4  # 9 intervals per dim = 10^4 = 10,000 points
        ... )
    """

    def __init__(
        self,
        dimension: int,
        bounds: Sequence[tuple[float, float]],
        *,
        Nx: Sequence[int] | None = None,
        Nx_points: Sequence[int] | None = None,
        num_points: Sequence[int] | None = None,  # Deprecated alias for Nx_points
        spacing_type: str = "uniform",
        custom_coordinates: Sequence[NDArray] | None = None,
        boundary_conditions: BoundaryConditions | None = None,
    ):
        """
        Initialize tensor product grid.

        Args:
            dimension: Spatial dimension (any positive integer).
                Note: For d>3, grid requires O(N^d) memory/computation.
                Consider meshfree methods for high-dimensional problems.
            bounds: List of (min, max) bounds for each dimension
            Nx: Number of intervals along each dimension.
                For example, Nx=[100, 50] creates 101×51 grid points.
            Nx_points: Number of grid points along each dimension.
                For example, Nx_points=[101, 51] creates 100×50 intervals.
            num_points: Deprecated alias for Nx_points. Use Nx_points instead.
            spacing_type: "uniform" or "custom"
            custom_coordinates: Optional list of 1D coordinate arrays
            boundary_conditions: Spatial BC specification (default: no-flux).
                This is the SSOT for spatial BC - both HJB and FP solvers
                query this geometry for consistent boundary conditions.

        Note:
            Must specify exactly one of: Nx, Nx_points, or num_points.
            - Nx (intervals): Nx_points = Nx + 1
            - Nx_points (points): Nx = Nx_points - 1

        Raises:
            ValueError: If dimension < 1
            ValueError: If bounds length != dimension
            ValueError: If none or multiple of Nx/Nx_points/num_points specified
            UserWarning: If dimension > 3 (performance warning)
            DeprecationWarning: If num_points is used (use Nx_points instead)
        """
        if dimension < 1:
            raise ValueError(f"Dimension must be positive, got {dimension}")

        # Handle Nx vs Nx_points vs num_points
        specified = sum(x is not None for x in [Nx, Nx_points, num_points])
        if specified == 0:
            raise ValueError("Must specify one of: Nx (intervals) or Nx_points (points)")
        if specified > 1:
            raise ValueError("Cannot specify multiple of: Nx, Nx_points, num_points")

        if num_points is not None:
            import warnings

            warnings.warn(
                "num_points is deprecated, use Nx_points instead. Will be removed in v1.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            Nx_points = num_points

        # Convert to internal storage (always store as points)
        if Nx is not None:
            self._Nx_points: list[int] = [n + 1 for n in Nx]
        else:
            assert Nx_points is not None
            self._Nx_points = list(Nx_points)

        # Warn about performance for high dimensions
        if dimension > 3:
            import warnings

            total_points = 1
            for n in self._Nx_points:
                total_points *= n

            warnings.warn(
                f"TensorProductGrid with dimension={dimension} requires O(N^d) memory/computation. "
                f"Total grid points: {total_points:,}. "
                f"For high dimensions (d>3), consider meshfree methods.",
                category=UserWarning,
                stacklevel=2,
            )

        if len(bounds) != dimension or len(self._Nx_points) != dimension:
            raise ValueError(
                f"bounds and Nx/Nx_points must have length {dimension}, got {len(bounds)} and {len(self._Nx_points)}"
            )

        self._dimension = dimension
        self.bounds = list(bounds)
        self.spacing_type = spacing_type

        # Create coordinate arrays
        if spacing_type == "uniform":
            self.coordinates = [
                np.linspace(bounds[i][0], bounds[i][1], self._Nx_points[i]) for i in range(self._dimension)
            ]
            self.spacing = [
                (bounds[i][1] - bounds[i][0]) / (self._Nx_points[i] - 1) if self._Nx_points[i] > 1 else 0.0
                for i in range(self._dimension)
            ]
            self.is_uniform = True

        elif spacing_type == "custom":
            if custom_coordinates is None:
                raise ValueError("custom_coordinates required for spacing_type='custom'")
            if len(custom_coordinates) != self._dimension:
                raise ValueError(f"custom_coordinates must have length {self._dimension}")

            self.coordinates = [np.asarray(coords) for coords in custom_coordinates]
            self.spacing = [None] * self._dimension  # Variable spacing
            self.is_uniform = False

        else:
            raise ValueError(f"Unknown spacing_type: {spacing_type}")

        # Validate coordinates
        for i, coords in enumerate(self.coordinates):
            if len(coords) != self._Nx_points[i]:
                raise ValueError(f"Coordinate array {i} has length {len(coords)}, expected {self._Nx_points[i]}")

        # Store boundary conditions (SSOT for spatial BC)
        # Use lazy binding to set/validate dimension
        if boundary_conditions is not None:
            # bind_dimension returns a new BC with dimension set, or validates if already set
            boundary_conditions = boundary_conditions.bind_dimension(dimension)
        self._boundary_conditions = boundary_conditions

        # Cache for flattened grid (computed lazily, perf optimization)
        self._flattened_cache: NDArray | None = None

    # Geometry ABC implementation - properties
    @property
    def dimension(self) -> int:
        """Spatial dimension of the grid."""
        return self._dimension

    @property
    def geometry_type(self) -> GeometryType:
        """Type of geometry (always CARTESIAN_GRID for tensor product grids)."""
        return GeometryType.CARTESIAN_GRID

    @property
    def num_spatial_points(self) -> int:
        """Total number of discrete spatial points."""
        return self.total_points()

    # Nx/Nx_points consistent naming
    @property
    def Nx(self) -> list[int]:
        """Number of intervals along each dimension (Nx_points - 1)."""
        return [n - 1 for n in self._Nx_points]

    @property
    def Nx_points(self) -> list[int]:
        """Number of grid points along each dimension."""
        return self._Nx_points.copy()

    @property
    def num_points(self) -> list[int]:
        """Deprecated: Use Nx_points instead. Number of grid points along each dimension."""
        import warnings

        warnings.warn(
            "num_points is deprecated, use Nx_points instead. Will be removed in v1.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._Nx_points.copy()

    def get_spatial_grid(self) -> NDArray:
        """
        Get spatial grid representation.

        Returns:
            numpy array of all grid points (N, dimension)
        """
        return self.flatten()

    def get_problem_config(self) -> dict:
        """
        Return configuration dict for MFGProblem initialization.

        This polymorphic method provides TensorProductGrid-specific configuration
        for MFGProblem, including structured grid information.

        Returns:
            Dictionary with keys:
                - num_spatial_points: Total number of points
                - spatial_shape: Shape tuple (Nx_points[0], Nx_points[1], ...)
                - spatial_bounds: Bounds [(xmin, xmax), (ymin, ymax), ...]
                - spatial_discretization: Number of points [Nx_points[0], ...]
                - legacy_1d_attrs: Legacy 1D attributes (xmin, xmax, etc.) if 1D

        Added in v0.10.1 for polymorphic geometry handling.
        """
        config = {
            "num_spatial_points": self.total_points(),
            "spatial_shape": tuple(self._Nx_points),
            "spatial_bounds": tuple(self.bounds),
            "spatial_discretization": tuple(self._Nx_points),
        }

        # Legacy 1D attributes (for backward compatibility with 1D solvers)
        if self._dimension == 1:
            config["legacy_1d_attrs"] = {
                "xmin": self.bounds[0][0],
                "xmax": self.bounds[0][1],
                "Lx": self.bounds[0][1] - self.bounds[0][0],
                "Nx": self._Nx_points[0] - 1,  # Nx = intervals (Nx_points - 1)
                "Dx": self.spacing[0],
                "xSpace": self.coordinates[0],
            }
        else:
            config["legacy_1d_attrs"] = None

        return config

    def meshgrid(self, indexing: str = "ij") -> tuple[NDArray, ...]:
        """
        Create meshgrid from 1D coordinate arrays.

        Args:
            indexing: 'ij' (matrix indexing) or 'xy' (Cartesian indexing)

        Returns:
            Tuple of coordinate matrices (X, Y, Z, ...)

        Example:
            >>> grid = TensorProductGrid(2, [(0,1), (0,1)], [11, 11])
            >>> X, Y = grid.meshgrid()
            >>> X.shape, Y.shape
            ((11, 11), (11, 11))
        """
        return np.meshgrid(*self.coordinates, indexing=indexing)

    def flatten(self) -> NDArray:
        """
        Get all grid points as flat (N, dimension) array.

        Returns:
            Array of shape (N, dimension) where N = ∏num_points[i]

        Note:
            Result is cached for performance. Returns a copy of the cached array.

        Example:
            >>> grid = TensorProductGrid(2, [(0,1), (0,1)], [3, 3])
            >>> points = grid.flatten()
            >>> points.shape
            (9, 2)
        """
        if self._flattened_cache is None:
            mesh = self.meshgrid(indexing="ij")
            self._flattened_cache = np.column_stack([m.ravel() for m in mesh])
        return self._flattened_cache.copy()

    def total_points(self) -> int:
        """
        Get total number of grid points.

        Returns:
            N = ∏Nx_points[i]
        """
        return int(np.prod(self._Nx_points))

    def get_index(self, multi_index: Sequence[int]) -> int:
        """
        Convert multi-dimensional index to flat index.

        Args:
            multi_index: Tuple (i, j, k) of indices in each dimension

        Returns:
            Flat index for accessing 1D arrays

        Example:
            >>> grid = TensorProductGrid(2, [(0,1), (0,1)], Nx=[10, 10])
            >>> flat_idx = grid.get_index((5, 3))  # Point (i=5, j=3)
        """
        if len(multi_index) != self._dimension:
            raise ValueError(f"multi_index must have length {self._dimension}")

        flat_idx = 0
        stride = 1
        for i in reversed(range(self._dimension)):
            flat_idx += multi_index[i] * stride
            stride *= self._Nx_points[i]

        return flat_idx

    def get_multi_index(self, flat_index: int) -> tuple[int, ...]:
        """
        Convert flat index to multi-dimensional index.

        Args:
            flat_index: Flat index in [0, total_points)

        Returns:
            Tuple (i, j, k) of indices in each dimension

        Example:
            >>> grid = TensorProductGrid(2, [(0,1), (0,1)], Nx=[10, 10])
            >>> i, j = grid.get_multi_index(53)
        """
        if flat_index < 0 or flat_index >= self.total_points():
            raise ValueError(f"flat_index {flat_index} out of range [0, {self.total_points()})")

        indices = []
        remaining = flat_index
        # Process dimensions in row-major order (C-order)
        for i in range(self._dimension):
            stride = int(np.prod(self._Nx_points[i + 1 :])) if i < self._dimension - 1 else 1
            idx = remaining // stride
            indices.append(idx)
            remaining %= stride

        return tuple(indices)

    def get_spacing(self, dimension_idx: int = 0) -> float | NDArray:
        """
        Get grid spacing for specified dimension.

        Args:
            dimension_idx: Dimension index (0 for x, 1 for y, 2 for z)

        Returns:
            Scalar spacing (uniform) or array of spacings (non-uniform)
        """
        if dimension_idx >= self._dimension:
            raise ValueError(f"dimension_idx {dimension_idx} >= dimension {self._dimension}")

        if self.is_uniform:
            return self.spacing[dimension_idx]
        else:
            # Compute local spacing from coordinates
            coords = self.coordinates[dimension_idx]
            return np.diff(coords)

    def refine(self, factor: int | Sequence[int]) -> TensorProductGrid:
        """
        Create refined grid with more points.

        Args:
            factor: Refinement factor (same for all dims) or per-dimension factors

        Returns:
            New TensorProductGrid with refined resolution

        Example:
            >>> grid = TensorProductGrid(2, [(0,1), (0,1)], Nx=[10, 10])
            >>> fine_grid = grid.refine(2)  # Now 21×21 intervals (22×22 points)
        """
        if isinstance(factor, int):
            factors = [factor] * self._dimension
        else:
            factors = list(factor)

        # Refine intervals: (Nx_points - 1) * factor + 1 = new Nx_points
        new_Nx_points = [(n - 1) * f + 1 for n, f in zip(self._Nx_points, factors, strict=False)]

        return TensorProductGrid(
            dimension=self._dimension,
            bounds=self.bounds,
            Nx_points=new_Nx_points,
            spacing_type=self.spacing_type,
            boundary_conditions=self._boundary_conditions,
        )

    def coarsen(self, factor: int | Sequence[int]) -> TensorProductGrid:
        """
        Create coarsened grid with fewer points.

        Args:
            factor: Coarsening factor (same for all dims) or per-dimension factors

        Returns:
            New TensorProductGrid with coarser resolution
        """
        if isinstance(factor, int):
            factors = [factor] * self._dimension
        else:
            factors = list(factor)

        # Coarsen intervals: (Nx_points - 1) // factor + 1 = new Nx_points
        new_Nx_points = [(n - 1) // f + 1 for n, f in zip(self._Nx_points, factors, strict=False)]

        return TensorProductGrid(
            dimension=self._dimension,
            bounds=self.bounds,
            Nx_points=new_Nx_points,
            spacing_type=self.spacing_type,
            boundary_conditions=self._boundary_conditions,
        )

    def volume_element(self, multi_index: Sequence[int] | None = None) -> float:
        """
        Compute volume element (dx·dy·dz) at grid point.

        Args:
            multi_index: Optional grid point index (for non-uniform grids)

        Returns:
            Volume element (1D: dx, 2D: dx·dy, 3D: dx·dy·dz)
        """
        if self.is_uniform:
            return float(np.prod(self.spacing))
        else:
            # For non-uniform grids, need local spacing
            if multi_index is None:
                raise ValueError("multi_index required for non-uniform grids")

            vol = 1.0
            for i in range(self._dimension):
                spacings = self.get_spacing(i)
                idx = multi_index[i]
                # Use average of left and right spacing
                if idx == 0:
                    local_spacing = spacings[0]
                elif idx == self._Nx_points[i] - 1:
                    local_spacing = spacings[-1]
                else:
                    local_spacing = 0.5 * (spacings[idx - 1] + spacings[idx])
                vol *= local_spacing

            return vol

    # ============================================================================
    # Geometry ABC implementation (data interface)
    # ============================================================================

    def get_bounds(self) -> tuple[NDArray, NDArray]:
        """
        Return bounding box of grid.

        Returns:
            (min_coords, max_coords) tuple of arrays

        Examples:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,2)], Nx_points=[10,20])
            >>> min_coords, max_coords = grid.get_bounds()
            >>> min_coords
            array([0., 0.])
            >>> max_coords
            array([1., 2.])
        """
        min_coords = np.array([b[0] for b in self.bounds])
        max_coords = np.array([b[1] for b in self.bounds])
        return min_coords, max_coords

    def get_boundary_conditions(self):
        """
        Get spatial boundary conditions for this grid.

        Returns stored BC if provided at construction, otherwise falls back
        to parent class default (no-flux BC for mass conservation).

        This is the Single Source of Truth (SSOT) for spatial BC in MFG systems.

        Returns:
            BoundaryConditions: Spatial BC specification

        Examples:
            >>> # Default BC (no-flux)
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1),(0,1)], Nx=[10,10])
            >>> bc = grid.get_boundary_conditions()
            >>> bc.is_uniform  # True

            >>> # Custom BC
            >>> from mfg_pde.geometry.boundary import dirichlet_bc
            >>> grid = TensorProductGrid(..., boundary_conditions=dirichlet_bc(0.0, dimension=2))
            >>> bc = grid.get_boundary_conditions()
            >>> bc.bc_type.value  # 'dirichlet'
        """
        if self._boundary_conditions is not None:
            return self._boundary_conditions
        # Fall back to parent class default (no-flux)
        return super().get_boundary_conditions()

    def has_explicit_boundary_conditions(self) -> bool:
        """
        Check if this grid has explicitly specified boundary conditions.

        Returns:
            True if BC were provided in constructor, False if using default
        """
        return self._boundary_conditions is not None

    # ============================================================================
    # CartesianGrid ABC implementation (grid-specific utilities)
    # ============================================================================

    def get_grid_spacing(self) -> list[float]:
        """
        Get grid spacing per dimension.

        Returns:
            [dx1, dx2, ...] where dxi = (xmax_i - xmin_i) / (Ni - 1)

        Examples:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,2)], Nx_points=[11,21])
            >>> dx = grid.get_grid_spacing()
            >>> dx
            [0.1, 0.1]
        """
        if not self.is_uniform:
            raise ValueError("get_grid_spacing() only valid for uniform grids. Use get_spacing() for non-uniform.")
        return self.spacing

    def get_grid_shape(self) -> tuple[int, ...]:
        """
        Get number of grid points per dimension.

        Returns:
            (Nx_points[0], Nx_points[1], ...) tuple of grid points

        Examples:
            >>> grid = TensorProductGrid(dimension=2, Nx_points=[10, 20])
            >>> shape = grid.get_grid_shape()
            >>> shape
            (10, 20)
        """
        return tuple(self._Nx_points)

    # ============================================================================
    # Solver Operation Interface (NEW - from Geometry ABC)
    # ============================================================================

    def get_laplacian_operator(self) -> Callable:
        """
        Return finite difference Laplacian operator.

        Returns:
            Function with signature: (u: NDArray, idx: tuple[int, ...]) -> float

        Examples:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx_points=[10,10])
            >>> laplacian = grid.get_laplacian_operator()
            >>> u = np.random.rand(10, 10)
            >>> lap_value = laplacian(u, (5, 5))  # Laplacian at grid point (5,5)
        """
        if not self.is_uniform:
            raise NotImplementedError("Laplacian operator not yet implemented for non-uniform grids")

        def laplacian_fd(u: NDArray, idx: tuple[int, ...]) -> float:
            """
            Compute Laplacian at grid point idx using central finite differences.

            Args:
                u: Solution array of shape self.grid_shape
                idx: Grid index tuple (i, j, k, ...)

            Returns:
                Laplacian value: Δu = Σ (∂²u/∂xi²)
            """
            if len(idx) != self._dimension:
                raise ValueError(f"Index must have length {self._dimension}, got {len(idx)}")

            laplacian = 0.0
            for dim in range(self._dimension):
                dx = self.spacing[dim]

                # Handle boundaries with clamping (Neumann-like BC)
                idx_plus = list(idx)
                idx_minus = list(idx)
                idx_plus[dim] = min(idx[dim] + 1, self._Nx_points[dim] - 1)
                idx_minus[dim] = max(idx[dim] - 1, 0)

                # Central difference: (u[i+1] - 2*u[i] + u[i-1]) / dx²
                laplacian += (u[tuple(idx_plus)] - 2.0 * u[idx] + u[tuple(idx_minus)]) / (dx**2)

            return float(laplacian)

        return laplacian_fd

    def get_gradient_operator(self) -> Callable:
        """
        Return finite difference gradient operator.

        Returns:
            Function with signature: (u: NDArray, idx: tuple[int, ...]) -> NDArray

        Examples:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx_points=[10,10])
            >>> gradient = grid.get_gradient_operator()
            >>> u = np.random.rand(10, 10)
            >>> grad_u = gradient(u, (5, 5))  # Returns [du/dx, du/dy]
            >>> grad_u.shape
            (2,)
        """
        if not self.is_uniform:
            raise NotImplementedError("Gradient operator not yet implemented for non-uniform grids")

        def gradient_fd(u: NDArray, idx: tuple[int, ...]) -> NDArray:
            """
            Compute gradient at grid point idx using central finite differences.

            Args:
                u: Solution array of shape self.grid_shape
                idx: Grid index tuple (i, j, k, ...)

            Returns:
                Gradient vector [∂u/∂x1, ∂u/∂x2, ...]
            """
            if len(idx) != self._dimension:
                raise ValueError(f"Index must have length {self._dimension}, got {len(idx)}")

            gradient = np.zeros(self._dimension)
            for dim in range(self._dimension):
                dx = self.spacing[dim]

                # Handle boundaries with clamping
                idx_plus = list(idx)
                idx_minus = list(idx)
                idx_plus[dim] = min(idx[dim] + 1, self._Nx_points[dim] - 1)
                idx_minus[dim] = max(idx[dim] - 1, 0)

                # Central difference: (u[i+1] - u[i-1]) / (2*dx)
                gradient[dim] = (u[tuple(idx_plus)] - u[tuple(idx_minus)]) / (2.0 * dx)

            return gradient

        return gradient_fd

    def get_interpolator(self) -> Callable:
        """
        Return linear interpolator for arbitrary points.

        Returns:
            Function with signature: (u: NDArray, points: NDArray) -> NDArray | float
            - Single point (1D array of length dim): returns float
            - Multiple points (2D array of shape (N, dim)): returns NDArray of shape (N,)

        Examples:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx_points=[10,10])
            >>> interpolate = grid.get_interpolator()
            >>> u = np.random.rand(10, 10)
            >>> # Single point
            >>> value = interpolate(u, np.array([0.5, 0.3]))  # Returns float
            >>> # Multiple points (batched)
            >>> values = interpolate(u, np.array([[0.5, 0.3], [0.2, 0.8]]))  # Returns NDArray
        """

        def interpolate_linear(u: NDArray, points: NDArray) -> NDArray | float:
            """
            Linear interpolation at arbitrary point(s).

            Args:
                u: Solution array of shape self.grid_shape
                points: Physical coordinates, either:
                    - 1D array of length dim for single point
                    - 2D array of shape (N, dim) for N points

            Returns:
                Interpolated value(s): float for single point, NDArray for multiple
            """
            points = np.asarray(points)

            # Handle single point (1D array)
            if points.ndim == 1:
                if len(points) != self._dimension:
                    raise ValueError(f"Point must have length {self._dimension}, got {len(points)}")
                single_point = True
                points = points.reshape(1, -1)
            else:
                # Batched points (2D array)
                if points.shape[1] != self._dimension:
                    raise ValueError(f"Points must have shape (N, {self._dimension}), got {points.shape}")
                single_point = False

            # Use scipy's RegularGridInterpolator for nD linear interpolation
            try:
                from scipy.interpolate import RegularGridInterpolator
            except ImportError as err:
                raise ImportError("scipy required for interpolation") from err

            interpolator = RegularGridInterpolator(
                self.coordinates, u, method="linear", bounds_error=False, fill_value=0.0
            )
            result = interpolator(points)

            return float(result[0]) if single_point else result

        return interpolate_linear

    def get_boundary_handler(self, bc_type: str = "periodic", custom_conditions: dict | None = None):
        """
        Get or create dimension-agnostic boundary condition handler.

        Resolution order:
        1. Return stored BC if explicitly set via constructor or attribute
        2. Create new BC from bc_type parameter

        This ensures that custom BC (e.g., mixed BC with absorbing exits) are
        properly propagated to solvers that call this method.

        Args:
            bc_type: Standard boundary condition type (only used if no BC stored):
                - "dirichlet_zero": Zero Dirichlet on all boundaries
                - "neumann_zero": Zero Neumann (no-flux) on all boundaries
                - "periodic": Periodic on all boundaries
                - "periodic_x", "periodic_y", "periodic_z": Periodic in one direction
                - "periodic_both": Periodic in all directions (2D/3D)
                - "mixed": Mixed conditions (implementation-dependent)
            custom_conditions: Optional dict with custom BC specifications

        Returns:
            Boundary condition handler appropriate for grid dimension

        Example:
            >>> # Stored BC takes priority
            >>> grid = TensorProductGrid(2, [(0, 1), (0, 1)], [11, 11])
            >>> grid.boundary_conditions = my_custom_bc
            >>> grid.get_boundary_handler()  # Returns my_custom_bc

            >>> # Falls back to bc_type if no BC stored
            >>> grid2 = TensorProductGrid(2, [(0, 1), (0, 1)], [11, 11])
            >>> grid2.get_boundary_handler("dirichlet_zero")  # Creates Dirichlet BC

        Note:
            Boundary regions indexed 0 to 2D-1:
            - Region 2i: x_i_min (left hyperface in dimension i)
            - Region 2i+1: x_i_max (right hyperface in dimension i)
        """
        # Priority 1: Return stored BC if available
        if self._boundary_conditions is not None:
            return self._boundary_conditions

        # Priority 2: Create BC from parameters
        # Special case: 1D uses different BC interface
        if self._dimension == 1:
            return self._create_bc_1d(bc_type, custom_conditions)

        # Generic nD dispatch (D >= 2)
        return self._create_bc_nd(bc_type)

    def _create_bc_1d(self, bc_type: str, custom_conditions: dict | None):
        """Create 1D boundary conditions (uses dataclass interface)."""
        from mfg_pde.geometry.boundary.fdm_bc_1d import BoundaryConditions

        bc_map = {
            "periodic": {"type": "periodic"},
            "dirichlet_zero": {"type": "dirichlet", "left_value": 0.0, "right_value": 0.0},
            "neumann_zero": {"type": "neumann", "left_value": 0.0, "right_value": 0.0},
            "no_flux": {"type": "neumann", "left_value": 0.0, "right_value": 0.0},
        }

        if bc_type in bc_map:
            return BoundaryConditions(**bc_map[bc_type])
        elif custom_conditions:
            return BoundaryConditions(**custom_conditions)
        else:
            return BoundaryConditions(type="periodic")  # Safe default

    def _create_bc_nd(self, bc_type: str):
        """
        Create nD boundary conditions (D >= 2) using factory pattern.

        Dimension-agnostic: Flattens bounds and dispatches to appropriate factory.
        """
        # Flatten bounds to tuple format expected by factories
        flat_bounds = tuple(coord for min_max in self.bounds for coord in min_max)

        # Dimension-specific factory dispatch
        factory_map = {
            2: ("boundary.fem_bc_2d", "create_rectangle_boundary_conditions"),
            3: ("boundary.fem_bc_3d", "create_box_boundary_conditions"),
        }

        if self._dimension in factory_map:
            module_name, factory_name = factory_map[self._dimension]
            module = __import__(f"mfg_pde.geometry.{module_name}", fromlist=[factory_name])
            factory = getattr(module, factory_name)
            return factory(flat_bounds, bc_type)

        # Dimension > 3: Use 3D infrastructure as fallback (generalizes to nD hyperboxes)
        import warnings

        warnings.warn(
            f"Using 3D BC manager as fallback for dimension={self._dimension}. "
            f"BC handler will treat {self._dimension}D hypercube as generalized 3D box. "
            "This works for rectangular domains but may not handle all nD edge cases.",
            UserWarning,
            stacklevel=2,
        )
        from mfg_pde.geometry.boundary.fem_bc_3d import BoundaryConditionManager3D

        return BoundaryConditionManager3D()

    def __repr__(self) -> str:
        """String representation of grid."""
        return (
            f"TensorProductGrid(\n"
            f"  dimension={self._dimension},\n"
            f"  bounds={self.bounds},\n"
            f"  Nx_points={self._Nx_points},\n"
            f"  spacing_type='{self.spacing_type}',\n"
            f"  total_points={self.total_points()}\n"
            f")"
        )


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing TensorProductGrid...")

    import numpy as np

    # Test 2D grid creation with Nx (intervals)
    grid_2d = TensorProductGrid(dimension=2, bounds=[(0.0, 10.0), (0.0, 5.0)], Nx=[10, 5])

    assert grid_2d.dimension == 2
    assert grid_2d.Nx == [10, 5]  # intervals
    assert grid_2d.Nx_points == [11, 6]  # points
    assert grid_2d.total_points() == 11 * 6
    assert len(grid_2d.coordinates) == 2

    print(f"  2D grid (Nx=[10, 5]): {grid_2d.Nx_points[0]}x{grid_2d.Nx_points[1]} = {grid_2d.total_points()} points")

    # Test with Nx_points directly
    grid_2d_alt = TensorProductGrid(dimension=2, bounds=[(0.0, 10.0), (0.0, 5.0)], Nx_points=[11, 6])
    assert grid_2d_alt.Nx == [10, 5]
    assert grid_2d_alt.Nx_points == [11, 6]

    print(f"  2D grid (Nx_points=[11, 6]): Nx={grid_2d_alt.Nx}")

    # Test meshgrid
    X, Y = grid_2d.meshgrid()
    assert X.shape == (11, 6)
    assert Y.shape == (11, 6)
    assert X[0, 0] == 0.0
    assert X[-1, 0] == 10.0
    assert Y[0, 0] == 0.0
    assert Y[0, -1] == 5.0

    print(f"  Meshgrid: X shape {X.shape}, range [{X.min():.1f}, {X.max():.1f}]")

    # Test flatten
    points = grid_2d.flatten()
    assert points.shape == (66, 2)

    print(f"  Flattened: {points.shape[0]} points in {points.shape[1]}D")

    # Test spacing
    assert np.allclose(grid_2d.spacing[0], 1.0)  # 10/(11-1) = 1.0
    assert np.allclose(grid_2d.spacing[1], 1.0)  # 5/(6-1) = 1.0

    print(f"  Spacing: dx={grid_2d.spacing[0]:.2f}, dy={grid_2d.spacing[1]:.2f}")

    print("Smoke tests passed!")
