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
from mfg_pde.geometry.protocols import (
    SupportsAdvection,
    SupportsBoundaryDistance,
    SupportsBoundaryNormal,
    SupportsBoundaryProjection,
    SupportsDivergence,
    SupportsGradient,
    SupportsInterpolation,
    SupportsLaplacian,
    SupportsLipschitz,
    SupportsManifold,
    SupportsPeriodic,
    SupportsRegionMarking,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from mfg_pde.geometry.boundary.conditions import BoundaryConditions


class TensorProductGrid(
    CartesianGrid,
    # Boundary traits (Issue #590 Phase 1.2)
    SupportsBoundaryNormal,
    SupportsBoundaryProjection,
    SupportsBoundaryDistance,
    # Topology traits (Issue #590 Phase 1.2)
    SupportsManifold,
    SupportsLipschitz,
    SupportsPeriodic,
    # Operator traits (Issue #590 Phase 1.2, Issue #595)
    SupportsLaplacian,
    SupportsGradient,
    SupportsDivergence,
    SupportsAdvection,
    SupportsInterpolation,
    # Region traits (Issue #590 Phase 1.3)
    SupportsRegionMarking,
):
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

        # Region registry for named boundary/subdomain marking (Issue #590 Phase 1.3)
        self._regions: dict[str, NDArray[np.bool_]] = {}

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

    def get_collocation_points(self) -> np.ndarray:
        """
        Get flattened grid points as collocation points.

        Returns:
            Array of shape (N, d) where N is total number of grid points
            and d is the spatial dimension.

        Examples:
            >>> grid = TensorProductGrid(dimension=1, bounds=[(0, 1)], Nx_points=[5])
            >>> points = grid.get_collocation_points()
            >>> points.shape
            (5, 1)
            >>> points[:,0]  # x-coordinates
            array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        """
        # Get spatial grid - already returns (N, d) for all dimensions
        spatial_grid = self.get_spatial_grid()

        # For 1D, ensure (N, 1) shape
        if self.dimension == 1:
            return spatial_grid.reshape(-1, 1)
        else:
            # For d>1, get_spatial_grid() already returns (N, d)
            return spatial_grid

    # ============================================================================
    # Solver Operation Interface (NEW - from Geometry ABC)
    # ============================================================================

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

    # ============================================================================
    # Boundary Trait Implementations (Issue #590 - Phase 1.2)
    # ============================================================================

    def get_outward_normal(
        self,
        points: NDArray,
        boundary_name: str | None = None,
    ) -> NDArray:
        """
        Compute outward unit normal vectors at boundary points.

        For rectangular tensor product grids, normals are axis-aligned:
        - x_min boundary: [-1, 0, 0, ...]
        - x_max boundary: [+1, 0, 0, ...]
        - y_min boundary: [0, -1, 0, ...]
        - etc.

        Args:
            points: Points at which to evaluate normal, shape (num_points, dimension)
                    or (dimension,) for single point
            boundary_name: Optional boundary region name (e.g., "x_min", "y_max").
                           If None, infers boundary from point location.

        Returns:
            Outward unit normals, shape (num_points, dimension) or (dimension,)

        Raises:
            ValueError: If points not on boundary (when boundary_name=None)
            ValueError: If boundary_name not recognized

        Example:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx=[10,10])
            >>> normal = grid.get_outward_normal(np.array([0.0, 0.5]), boundary_name="x_min")
            >>> assert np.allclose(normal, [-1.0, 0.0])
        """
        # Handle single point
        single_point = points.ndim == 1
        if single_point:
            points = points.reshape(1, -1)

        if points.shape[1] != self._dimension:
            raise ValueError(f"Points dimension {points.shape[1]} != grid dimension {self._dimension}")

        num_points = points.shape[0]
        normals = np.zeros((num_points, self._dimension))

        if boundary_name is not None:
            # Parse boundary name: "x_min", "x_max", "y_min", etc.
            if "_" not in boundary_name:
                raise ValueError(f"Invalid boundary_name: {boundary_name}. Expected format: 'x_min', 'y_max', etc.")

            dim_name, side = boundary_name.rsplit("_", 1)
            dim_names = ["x", "y", "z", "w"]  # Extend as needed for high dimensions
            if self._dimension > len(dim_names):
                dim_names += [f"x{i}" for i in range(len(dim_names), self._dimension)]

            if dim_name not in dim_names[: self._dimension]:
                raise ValueError(f"Unknown dimension name: {dim_name}")

            dim_idx = dim_names.index(dim_name)
            if side == "min":
                normals[:, dim_idx] = -1.0
            elif side == "max":
                normals[:, dim_idx] = 1.0
            else:
                raise ValueError(f"Unknown side: {side}. Expected 'min' or 'max'")
        else:
            # Infer boundary from point location (find closest boundary)
            tolerance = 1e-10
            for i, point in enumerate(points):
                min_coords, max_coords = self.get_bounds()

                # Find which boundary this point is closest to
                dist_to_min = np.abs(point - min_coords)
                dist_to_max = np.abs(point - max_coords)

                # Check if on any boundary
                on_min = dist_to_min < tolerance
                on_max = dist_to_max < tolerance

                if not np.any(on_min | on_max):
                    raise ValueError(f"Point {point} not on boundary (tolerance={tolerance})")

                # Prioritize first boundary found (for corner points)
                if np.any(on_min):
                    dim_idx = np.argmax(on_min)
                    normals[i, dim_idx] = -1.0
                else:
                    dim_idx = np.argmax(on_max)
                    normals[i, dim_idx] = 1.0

        if single_point:
            return normals[0]
        return normals

    def project_to_boundary(
        self,
        points: NDArray,
        boundary_name: str | None = None,
    ) -> NDArray:
        """
        Project points onto domain boundary.

        For rectangular domains, this clamps coordinates to [xmin, xmax] × [ymin, ymax] × ...

        Args:
            points: Points to project, shape (num_points, dimension) or (dimension,)
            boundary_name: Optional specific boundary to project onto (e.g., "x_min").
                           If None, project to closest boundary.

        Returns:
            Projected points on boundary, same shape as input

        Example:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx=[10,10])
            >>> x_outside = np.array([1.2, 0.5])
            >>> x_boundary = grid.project_to_boundary(x_outside)
            >>> assert np.allclose(x_boundary, [1.0, 0.5])
        """
        # Handle single point
        single_point = points.ndim == 1
        if single_point:
            points = points.reshape(1, -1)

        projected = points.copy()
        min_coords, max_coords = self.get_bounds()

        if boundary_name is not None:
            # Project to specific boundary
            dim_name, side = boundary_name.rsplit("_", 1)
            dim_names = ["x", "y", "z", "w"]
            if self._dimension > len(dim_names):
                dim_names += [f"x{i}" for i in range(len(dim_names), self._dimension)]

            dim_idx = dim_names.index(dim_name)
            if side == "min":
                projected[:, dim_idx] = min_coords[dim_idx]
            elif side == "max":
                projected[:, dim_idx] = max_coords[dim_idx]
        else:
            # Project to closest boundary (clamp all coordinates)
            projected = np.clip(projected, min_coords, max_coords)

            # For points inside, project to closest face
            for i in range(len(projected)):
                point = points[i]
                # Find closest boundary
                dist_to_min = point - min_coords
                dist_to_max = max_coords - point

                # Find minimum distance to any boundary
                all_dists = np.concatenate([dist_to_min, dist_to_max])
                min_dist_idx = np.argmin(np.abs(all_dists))

                if min_dist_idx < self._dimension:
                    # Closest to min boundary
                    projected[i, min_dist_idx] = min_coords[min_dist_idx]
                else:
                    # Closest to max boundary
                    dim_idx = min_dist_idx - self._dimension
                    projected[i, dim_idx] = max_coords[dim_idx]

        if single_point:
            return projected[0]
        return projected

    def project_to_interior(
        self,
        points: NDArray,
        tolerance: float = 1e-10,
    ) -> NDArray:
        """
        Project points from outside domain into interior.

        For rectangular domains, clamps to [xmin + tol, xmax - tol] × ...

        Args:
            points: Points to project, shape (num_points, dimension) or (dimension,)
            tolerance: Distance to move inside boundary

        Returns:
            Projected points in interior, same shape as input

        Example:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx=[10,10])
            >>> x_outside = np.array([1.05, 0.5])
            >>> x_inside = grid.project_to_interior(x_outside, tolerance=1e-3)
            >>> assert x_inside[0] <= 1.0
        """
        # Handle single point
        single_point = points.ndim == 1
        if single_point:
            points = points.reshape(1, -1)

        min_coords, max_coords = self.get_bounds()
        projected = np.clip(points, min_coords + tolerance, max_coords - tolerance)

        if single_point:
            return projected[0]
        return projected

    def get_signed_distance(
        self,
        points: NDArray,
    ) -> NDArray:
        """
        Compute signed distance to boundary for rectangular domain.

        Uses box SDF formula:
            φ(x) = max(max(xmin - x), max(x - xmax))

        Negative inside, zero on boundary, positive outside.

        Args:
            points: Query points, shape (num_points, dimension) or (dimension,)

        Returns:
            Signed distances, shape (num_points,) or scalar

        Example:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx=[10,10])
            >>> phi = grid.get_signed_distance(np.array([[0.5, 0.5], [1.0, 0.5], [1.5, 0.5]]))
            >>> assert phi[0] < 0  # Inside
            >>> assert np.isclose(phi[1], 0, atol=1e-10)  # On boundary
            >>> assert phi[2] > 0  # Outside
        """
        # Handle single point
        single_point = points.ndim == 1
        if single_point:
            points = points.reshape(1, -1)

        min_coords, max_coords = self.get_bounds()

        # Box SDF: φ(x) = max(max(xmin - x), max(x - xmax))
        dist_from_min = min_coords - points  # Positive if outside min boundary
        dist_from_max = points - max_coords  # Positive if outside max boundary

        # For each point, find maximum violation
        sdf = np.maximum(np.max(dist_from_min, axis=1), np.max(dist_from_max, axis=1))

        if single_point:
            return sdf[0]
        return sdf

    # ============================================================================
    # Manifold Trait Implementation (Issue #590 - Phase 1.2)
    # ============================================================================

    @property
    def manifold_dimension(self) -> int:
        """
        Intrinsic dimension of the manifold.

        For TensorProductGrid, manifold dimension == spatial dimension (flat Euclidean space).
        """
        return self._dimension

    def get_metric_tensor(
        self,
        points: NDArray,
    ) -> NDArray:
        """
        Compute Riemannian metric tensor at given points.

        For flat Euclidean space (TensorProductGrid), metric is identity: g = I.

        Args:
            points: Query points, shape (num_points, dimension) or (dimension,)

        Returns:
            Metric tensor(s):
                - Single point: (dimension, dimension) identity matrix
                - Multiple points: (num_points, dimension, dimension) stack of identities

        Example:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx=[10,10])
            >>> g = grid.get_metric_tensor(np.array([0.5, 0.5]))
            >>> assert np.allclose(g, np.eye(2))
        """
        # Handle single point
        single_point = points.ndim == 1
        if single_point:
            return np.eye(self._dimension)

        num_points = points.shape[0]
        metric = np.zeros((num_points, self._dimension, self._dimension))
        for i in range(num_points):
            metric[i] = np.eye(self._dimension)

        return metric

    def get_tangent_space_basis(
        self,
        points: NDArray,
    ) -> NDArray:
        """
        Compute orthonormal basis for tangent space at given points.

        For flat Euclidean space, tangent space is ℝ^d with canonical basis.

        Args:
            points: Query points, shape (num_points, dimension) or (dimension,)

        Returns:
            Tangent basis vectors:
                - Single point: (dimension, dimension) canonical basis
                - Multiple points: (num_points, dimension, dimension)

        Example:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx=[10,10])
            >>> basis = grid.get_tangent_space_basis(np.array([0.5, 0.5]))
            >>> assert np.allclose(basis, np.eye(2))
        """
        # Canonical basis (same as metric for flat space)
        return self.get_metric_tensor(points)

    def compute_christoffel_symbols(
        self,
        points: NDArray,
    ) -> NDArray:
        """
        Compute Christoffel symbols for flat Euclidean space.

        For flat metric g = I, all Christoffel symbols are zero: Γᵢⱼᵏ = 0.

        Args:
            points: Query points, shape (num_points, dimension) or (dimension,)

        Returns:
            Christoffel symbols (all zeros):
                - Single point: (dimension, dimension, dimension) zeros
                - Multiple points: (num_points, dimension, dimension, dimension) zeros

        Example:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx=[10,10])
            >>> Gamma = grid.compute_christoffel_symbols(np.array([0.5, 0.5]))
            >>> assert np.allclose(Gamma, 0)
        """
        # Handle single point
        single_point = points.ndim == 1
        if single_point:
            return np.zeros((self._dimension, self._dimension, self._dimension))

        num_points = points.shape[0]
        return np.zeros((num_points, self._dimension, self._dimension, self._dimension))

    # ============================================================================
    # Lipschitz Trait Implementation (Issue #590 - Phase 1.2)
    # ============================================================================

    def get_lipschitz_constant(
        self,
        region: str | None = None,
    ) -> float:
        """
        Return Lipschitz constant for boundary representation.

        For axis-aligned rectangular boundaries, L = 0 (piecewise constant).

        Args:
            region: Optional boundary region name

        Returns:
            Lipschitz constant L = 0 for axis-aligned boundaries

        Example:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx=[10,10])
            >>> L = grid.get_lipschitz_constant()
            >>> assert L == 0.0
        """
        return 0.0  # Axis-aligned boundaries are piecewise constant

    def validate_lipschitz_regularity(
        self,
        tolerance: float = 1e-6,
    ) -> tuple[bool, str]:
        """
        Validate that boundary satisfies Lipschitz condition.

        Rectangular tensor product grids always have Lipschitz boundaries
        (piecewise smooth, axis-aligned).

        Args:
            tolerance: Numerical tolerance (unused for rectangular domains)

        Returns:
            (True, "") - Always valid for rectangular grids

        Example:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx=[10,10])
            >>> valid, msg = grid.validate_lipschitz_regularity()
            >>> assert valid and msg == ""
        """
        return True, ""

    # ============================================================================
    # Periodic Trait Implementation (Issue #590 - Phase 1.2)
    # ============================================================================

    @property
    def periodic_dimensions(self) -> tuple[int, ...]:
        """
        Get dimensions with periodic topology.

        Checks boundary conditions to determine which dimensions are periodic.

        Returns:
            Tuple of dimension indices that are periodic (empty if none)

        Example:
            >>> from mfg_pde.geometry.boundary import BoundaryConditions, BCType
            >>> # Periodic in x (dimension 0)
            >>> bc = BoundaryConditions(dimension=2, bc_type=BCType.PERIODIC)
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,2*np.pi),(0,1)],
            ...                           Nx=[20,10], boundary_conditions=bc)
            >>> # Note: Current BC doesn't track per-dimension periodicity yet
        """
        # Check boundary conditions for periodicity
        # Note: Current BC system doesn't have per-dimension periodic flags yet
        # This will be enhanced when SupportsRegionMarking is implemented
        if self._boundary_conditions is None:
            return ()

        from mfg_pde.geometry.boundary.types import BCType

        if self._boundary_conditions.is_uniform:
            if self._boundary_conditions.bc_type == BCType.PERIODIC:
                # All dimensions periodic
                return tuple(range(self._dimension))

        return ()  # Default: no periodicity

    def get_periods(self) -> dict[int, float]:
        """
        Get period lengths for periodic dimensions.

        Returns:
            Dictionary mapping dimension index → period length (L = xmax - xmin)

        Example:
            >>> # Assuming periodic BC
            >>> grid = TensorProductGrid(dimension=1, bounds=[(0, 2*np.pi)], Nx=[100])
            >>> periods = grid.get_periods()
            >>> # Returns {0: 2*pi} if periodic in x
        """
        periodic_dims = self.periodic_dimensions
        periods = {}
        for dim_idx in periodic_dims:
            xmin, xmax = self.bounds[dim_idx]
            periods[dim_idx] = xmax - xmin
        return periods

    def wrap_coordinates(
        self,
        points: NDArray,
    ) -> NDArray:
        """
        Wrap coordinates to canonical fundamental domain.

        For periodic dimension i: x_wrapped = xmin + (x - xmin) mod L

        Args:
            points: Points to wrap, shape (num_points, dimension) or (dimension,)

        Returns:
            Wrapped coordinates in [xmin, xmax), same shape as input

        Example:
            >>> # Assuming 1D periodic domain [0, 2π)
            >>> grid = TensorProductGrid(dimension=1, bounds=[(0, 2*np.pi)], Nx=[100])
            >>> x_wrapped = grid.wrap_coordinates(np.array([3*np.pi]))
            >>> # Returns [π] if periodic
        """
        # Handle single point
        single_point = points.ndim == 1
        if single_point:
            points = points.reshape(1, -1)

        wrapped = points.copy()
        periodic_dims = self.periodic_dimensions

        for dim_idx in periodic_dims:
            xmin, xmax = self.bounds[dim_idx]
            period = xmax - xmin
            wrapped[:, dim_idx] = xmin + np.mod(wrapped[:, dim_idx] - xmin, period)

        if single_point:
            return wrapped[0]
        return wrapped

    def compute_periodic_distance(
        self,
        points1: NDArray,
        points2: NDArray,
    ) -> NDArray:
        """
        Compute distance accounting for periodic topology.

        For periodic dim i: d_i = min(|x1_i - x2_i|, L_i - |x1_i - x2_i|)
        Total distance: d = sqrt(∑ d_i²)

        Args:
            points1: First set of points, shape (num_points, dimension) or (dimension,)
            points2: Second set of points, same shape as points1

        Returns:
            Distances, shape (num_points,) or scalar

        Example:
            >>> # 1D periodic [0, 1)
            >>> grid = TensorProductGrid(dimension=1, bounds=[(0, 1)], Nx=[100])
            >>> x1 = np.array([0.1])
            >>> x2 = np.array([0.9])
            >>> dist = grid.compute_periodic_distance(x1, x2)
            >>> # Returns 0.2 (wrapped distance) if periodic, 0.8 otherwise
        """
        # Handle single point
        single_point = points1.ndim == 1
        if single_point:
            points1 = points1.reshape(1, -1)
            points2 = points2.reshape(1, -1)

        diff = points1 - points2
        periodic_dims = self.periodic_dimensions

        for dim_idx in periodic_dims:
            period = self.bounds[dim_idx][1] - self.bounds[dim_idx][0]
            # Shortest distance on circle: min(|d|, L - |d|)
            abs_diff = np.abs(diff[:, dim_idx])
            diff[:, dim_idx] = np.minimum(abs_diff, period - abs_diff)

        distances = np.linalg.norm(diff, axis=1)

        if single_point:
            return distances[0]
        return distances

    # =========================================================================
    # Operator Trait Implementations (Issue #590 Phase 1.2, Issue #595)
    # =========================================================================

    def get_laplacian_operator(
        self,
        order: int = 2,
        bc: BoundaryConditions | None = None,
    ):
        """
        Return discrete Laplacian operator for this grid.

        Implements SupportsLaplacian protocol.

        Args:
            order: Discretization order (currently only order=2 supported)
            bc: Boundary conditions (None uses grid's default BC)

        Returns:
            LaplacianOperator: scipy LinearOperator for Laplacian

        Example:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx=[50, 50])
            >>> L = grid.get_laplacian_operator(order=2)
            >>> u = np.random.rand(51, 51)
            >>> Lu = L(u)  # Apply Laplacian
            >>> # Or use @ syntax for flattened arrays
            >>> Lu_flat = L @ u.ravel()
        """
        from mfg_pde.geometry.operators import LaplacianOperator

        # Use grid's BC if not provided
        if bc is None:
            bc = self.get_boundary_conditions()

        return LaplacianOperator(
            spacings=self.spacing,
            field_shape=tuple(self.Nx_points),
            bc=bc,
            order=order,
        )

    def get_gradient_operator(
        self,
        direction: int | None = None,
        order: int = 2,
        scheme: str = "central",
        time: float = 0.0,
    ):
        """
        Return discrete gradient operator(s) for this grid.

        Implements SupportsGradient protocol.

        Args:
            direction: Specific direction (0=x, 1=y, 2=z). If None, return all directions.
            order: Discretization order (not used yet, reserved for future)
            scheme: Difference scheme ("central", "upwind", "one_sided")
            time: Time for time-dependent boundary conditions (default 0.0)

        Returns:
            If direction is None:
                tuple of GradientComponentOperator for all dimensions
            If direction specified:
                Single GradientComponentOperator for that direction

        Example:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx=[50, 50])
            >>> grad_x, grad_y = grid.get_gradient_operator()
            >>> u = np.random.rand(51, 51)
            >>> du_dx = grad_x(u)
            >>> du_dy = grad_y(u)
        """
        from mfg_pde.geometry.operators import create_gradient_operators

        # Use grid's BC
        bc = self.get_boundary_conditions()

        # Create all gradient operators
        operators = create_gradient_operators(
            spacings=self.spacing,
            field_shape=tuple(self.Nx_points),
            scheme=scheme,
            bc=bc,
            time=time,
        )

        # Return specific direction or all
        if direction is not None:
            if direction >= self.dimension:
                raise ValueError(f"direction {direction} >= dimension {self.dimension}")
            return operators[direction]
        return operators

    def get_divergence_operator(
        self,
        order: int = 2,
    ):
        """
        Return discrete divergence operator for this grid.

        Implements SupportsDivergence protocol.

        Args:
            order: Discretization order (not used yet, reserved for future)

        Returns:
            DivergenceOperator (scipy LinearOperator) that computes ∇·F

        Example:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx=[50, 50])
            >>> div_op = grid.get_divergence_operator()
            >>> F = np.random.rand(2, 51, 51)  # Vector field (Fx, Fy)
            >>> div_F = div_op(F)  # Shape: (51, 51)
            >>> # Or use scipy interface:
            >>> div_F_flat = div_op @ F.ravel()  # Shape: (2601,)
        """
        from mfg_pde.geometry.operators import DivergenceOperator

        # Use grid's BC
        bc = self.get_boundary_conditions()

        return DivergenceOperator(
            spacings=self.spacing,
            field_shape=tuple(self.Nx_points),
            bc=bc,
        )

    def get_advection_operator(
        self,
        velocity_field: NDArray,
        scheme: str = "upwind",
        conservative: bool = True,
    ):
        """
        Return discrete advection operator for given velocity field.

        Implements SupportsAdvection protocol.

        Args:
            velocity_field: Velocity/drift field, shape (dimension, Nx_points...)
            scheme: Advection scheme ("upwind", "centered")
            conservative: If True, compute ∇·(vm). If False, compute v·∇m.

        Returns:
            AdvectionOperator (scipy LinearOperator) for transport

        Example:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx=[50, 50])
            >>> v = np.random.rand(2, 51, 51)  # Velocity field
            >>> adv_op = grid.get_advection_operator(v, scheme='upwind', conservative=True)
            >>> m = np.random.rand(51, 51)  # Density
            >>> div_mv = adv_op(m)  # Conservative advection
            >>> # Or use scipy interface:
            >>> div_mv_flat = adv_op @ m.ravel()
        """
        from mfg_pde.geometry.operators import AdvectionOperator

        # Use grid's BC
        bc = self.get_boundary_conditions()

        # Map conservative flag to form parameter
        form = "divergence" if conservative else "gradient"

        return AdvectionOperator(
            velocity_field=velocity_field,
            spacings=self.spacing,
            field_shape=tuple(self.Nx_points),
            scheme=scheme,
            form=form,
            bc=bc,
        )

    def get_interpolation_operator(
        self,
        query_points: NDArray,
        order: int = 1,
        extrapolation_mode: str = "boundary",
    ):
        """
        Return interpolation operator for given query points.

        Implements SupportsInterpolation protocol.

        Args:
            query_points: Points at which to interpolate, shape (num_query, dimension)
            order: Interpolation order (1=linear, 3=cubic)
            extrapolation_mode: How to handle points outside domain
                - "constant": Use fill_value (NaN)
                - "nearest": Use nearest boundary value
                - "boundary": Project to boundary and use boundary value

        Returns:
            InterpolationOperator (scipy LinearOperator) for grid-to-point evaluation

        Example:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx=[50, 50])
            >>> query_pts = np.random.rand(100, 2)  # 100 random points in [0,1]²
            >>> interp = grid.get_interpolation_operator(query_pts, order=1)
            >>> u = np.random.rand(51, 51)
            >>> u_interp = interp(u)  # Shape: (100,)
            >>> # Or use scipy interface:
            >>> u_interp_flat = interp @ u.ravel()  # Shape: (100,)
        """
        from mfg_pde.geometry.operators import InterpolationOperator

        return InterpolationOperator(
            grid_points=tuple(self.coordinates),
            query_points=query_points,
            order=order,
            extrapolation_mode=extrapolation_mode,
        )

    # ========================================================================
    # Region Marking Protocol (Issue #590 Phase 1.3)
    # ========================================================================

    def mark_region(
        self,
        name: str,
        predicate: Callable[[NDArray], NDArray[np.bool_]] | None = None,
        mask: NDArray[np.bool_] | None = None,
        boundary: str | None = None,
    ) -> None:
        """
        Mark a named spatial region for later reference.

        Implements SupportsRegionMarking protocol.

        Args:
            name: Unique name for this region (e.g., "inlet", "obstacle", "safe_zone")
            predicate: Function taking points (N, dimension) → bool mask (N,)
                       True where point is in region
            mask: Boolean mask directly specifying region (total_points,)
            boundary: Standard boundary name (e.g., "x_min", "x_max") for rectangular domains

        Raises:
            ValueError: If name already exists
            ValueError: If neither predicate, mask, nor boundary provided
            ValueError: If mask has wrong shape

        Example:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx=[50, 50])
            >>>
            >>> # Mark inlet region using predicate
            >>> grid.mark_region(
            ...     "inlet",
            ...     predicate=lambda x: np.all((x[:, 0] < 0.1) & (x[:, 1] > 0.4) & (x[:, 1] < 0.6), axis=0)
            ... )
            >>>
            >>> # Mark obstacle using direct mask
            >>> obstacle_mask = ...  # Boolean array of shape (51*51,)
            >>> grid.mark_region("obstacle", mask=obstacle_mask)
            >>>
            >>> # Mark boundary region
            >>> grid.mark_region("left_wall", boundary="x_min")

        Note:
            Issue #590 Phase 1.3: Enables mixed BC and localized constraints.
        """
        if name in self._regions:
            raise ValueError(
                f"Region '{name}' already exists. Use a different name or remove the existing region first."
            )

        # Validate: exactly one specification method
        specified = sum(x is not None for x in [predicate, mask, boundary])
        if specified == 0:
            raise ValueError("Must specify one of: predicate, mask, or boundary")
        if specified > 1:
            raise ValueError("Cannot specify multiple of: predicate, mask, boundary")

        # Compute mask based on specification method
        total_pts = self.total_points()

        if mask is not None:
            # Direct mask specification
            if mask.shape != (total_pts,):
                raise ValueError(f"Mask must have shape ({total_pts},), got {mask.shape}")
            region_mask = mask

        elif predicate is not None:
            # Predicate-based specification
            # Evaluate predicate at all grid points
            points = self.flatten()  # Shape: (total_points, dimension)
            region_mask = predicate(points)

            if region_mask.shape != (total_pts,):
                raise ValueError(
                    f"Predicate must return boolean array of shape ({total_pts},), got {region_mask.shape}"
                )

        elif boundary is not None:
            # Boundary name specification (e.g., "x_min", "x_max")
            # Parse boundary name: "x_min", "y_max", etc.
            import re

            match = re.match(r"([xyz])_(min|max)", boundary)
            if not match and self._dimension <= 3:
                raise ValueError(
                    f"Invalid boundary name '{boundary}'. Expected format: 'x_min', 'x_max', 'y_min', 'y_max', etc."
                )

            # For higher dimensions, support generic format
            if not match:
                # Try generic format: "dim0_min", "dim1_max", etc.
                match = re.match(r"dim(\d+)_(min|max)", boundary)
                if not match:
                    raise ValueError(f"Invalid boundary name '{boundary}'. Expected format: 'x_min', 'dimN_min', etc.")
                dim_idx = int(match.group(1))
            else:
                # Convert x/y/z to dimension index
                dim_char = match.group(1)
                dim_idx = {"x": 0, "y": 1, "z": 2}[dim_char]

            side = match.group(2)  # "min" or "max"

            if dim_idx >= self._dimension:
                raise ValueError(f"Dimension index {dim_idx} out of range for {self._dimension}D grid")

            # Create mask for specified boundary
            region_mask = self._get_boundary_mask(dim_idx, side)

        else:
            # Should never reach here due to validation above
            raise RuntimeError("Internal error: no specification method provided")

        # Store region
        self._regions[name] = region_mask

    def _get_boundary_mask(self, dim_idx: int, side: str) -> NDArray[np.bool_]:
        """
        Get boolean mask for boundary face.

        Args:
            dim_idx: Dimension index (0=x, 1=y, 2=z, ...)
            side: "min" or "max"

        Returns:
            Boolean mask of shape (total_points,)
        """
        # Create multi-index for boundary
        shape = tuple(self._Nx_points)
        total_pts = self.total_points()
        mask = np.zeros(total_pts, dtype=bool)

        # Flatten indices for the boundary
        # For axis-aligned boundaries in tensor product grid:
        # - x_min: all points where x-index = 0
        # - x_max: all points where x-index = Nx_points[0] - 1
        # etc.

        if side == "min":
            boundary_idx = 0
        else:  # "max"
            boundary_idx = self._Nx_points[dim_idx] - 1

        # Generate all multi-indices for this boundary face
        # Use np.ndindex to iterate over all indices in the grid
        for idx, multi_idx in enumerate(np.ndindex(*shape)):
            if multi_idx[dim_idx] == boundary_idx:
                mask[idx] = True

        return mask

    def get_region_mask(self, name: str) -> NDArray[np.bool_]:
        """
        Get boolean mask for named region.

        Implements SupportsRegionMarking protocol.

        Args:
            name: Region name (from mark_region call)

        Returns:
            Boolean mask of shape (total_points,)
            True at grid points in region

        Raises:
            KeyError: If region name not found

        Example:
            >>> grid.mark_region("inlet", predicate=lambda x: x[:, 0] < 0.1)
            >>> inlet_mask = grid.get_region_mask("inlet")
            >>> u_flat = np.zeros(grid.total_points())
            >>> u_flat[inlet_mask] = 1.0  # Set value in inlet region
        """
        if name not in self._regions:
            raise KeyError(f"Region '{name}' not found. Available regions: {list(self._regions.keys())}")
        return self._regions[name]

    def intersect_regions(self, *names: str) -> NDArray[np.bool_]:
        """
        Get intersection of multiple regions (boolean AND).

        Implements SupportsRegionMarking protocol.

        Args:
            *names: Region names to intersect

        Returns:
            Boolean mask: True where all regions overlap

        Raises:
            KeyError: If any region name not found

        Example:
            >>> # Points that are in both "inlet" and "high_priority"
            >>> mask = grid.intersect_regions("inlet", "high_priority")
            >>> u_flat[mask] = 2.0  # Set special value in intersection
        """
        if not names:
            raise ValueError("Must provide at least one region name")

        masks = [self.get_region_mask(name) for name in names]
        return np.logical_and.reduce(masks)

    def union_regions(self, *names: str) -> NDArray[np.bool_]:
        """
        Get union of multiple regions (boolean OR).

        Implements SupportsRegionMarking protocol.

        Args:
            *names: Region names to union

        Returns:
            Boolean mask: True where any region is True

        Raises:
            KeyError: If any region name not found

        Example:
            >>> # All exit points (multiple exit boundaries)
            >>> mask = grid.union_regions("exit_top", "exit_bottom", "exit_sides")
            >>> apply_exit_bc(u_flat[mask])
        """
        if not names:
            raise ValueError("Must provide at least one region name")

        masks = [self.get_region_mask(name) for name in names]
        return np.logical_or.reduce(masks)

    def get_region_names(self) -> list[str]:
        """
        Get list of all registered region names.

        Implements SupportsRegionMarking protocol.

        Returns:
            List of region names in registration order

        Example:
            >>> names = grid.get_region_names()
            >>> print("Available regions:", names)
            Available regions: ['inlet', 'exit', 'obstacle', 'walls']
        """
        return list(self._regions.keys())

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
