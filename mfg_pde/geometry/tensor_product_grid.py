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

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


class TensorProductGrid:
    """
    Tensor product grid for multi-dimensional structured domains.

    Provides memory-efficient representation of d-dimensional regular grids
    using 1D coordinate arrays. Supports uniform and non-uniform spacing.

    Supports arbitrary dimensions, though O(N^d) complexity limits practical
    use to d≤3 for dense grids. For high dimensions (d>3), consider meshfree methods.

    Attributes:
        dimension: Spatial dimension (any positive integer)
        bounds: List of (min, max) tuples for each dimension
        num_points: List of grid points along each dimension
        coordinates: List of 1D coordinate arrays
        spacing: Grid spacing along each dimension (if uniform)
        is_uniform: Whether grid has uniform spacing in each dimension

    Example:
        >>> # Create 2D grid: [0,10] × [0,5] with 101×51 points
        >>> grid = TensorProductGrid(
        ...     dimension=2,
        ...     bounds=[(0.0, 10.0), (0.0, 5.0)],
        ...     num_points=[101, 51]
        ... )
        >>> x, y = grid.meshgrid()  # Get coordinate matrices
        >>> flat_points = grid.flatten()  # Get all grid points as (N,2) array

        >>> # Create 4D grid (with performance warning)
        >>> grid_4d = TensorProductGrid(
        ...     dimension=4,
        ...     bounds=[(0.0, 1.0)] * 4,
        ...     num_points=[10] * 4  # 10^4 = 10,000 points
        ... )
    """

    def __init__(
        self,
        dimension: int,
        bounds: Sequence[tuple[float, float]],
        num_points: Sequence[int],
        spacing_type: str = "uniform",
        custom_coordinates: Sequence[NDArray] | None = None,
    ):
        """
        Initialize tensor product grid.

        Args:
            dimension: Spatial dimension (any positive integer).
                Note: For d>3, grid requires O(N^d) memory/computation.
                Consider meshfree methods for high-dimensional problems.
            bounds: List of (min, max) bounds for each dimension
            num_points: Number of grid points along each dimension
            spacing_type: "uniform" or "custom"
            custom_coordinates: Optional list of 1D coordinate arrays

        Raises:
            ValueError: If dimension < 1
            ValueError: If bounds/num_points length != dimension
            UserWarning: If dimension > 3 (performance warning)
        """
        if dimension < 1:
            raise ValueError(f"Dimension must be positive, got {dimension}")

        # Warn about performance for high dimensions
        if dimension > 3:
            import warnings

            total_points = 1
            for n in num_points:
                total_points *= n

            warnings.warn(
                f"TensorProductGrid with dimension={dimension} requires O(N^d) memory/computation. "
                f"Total grid points: {total_points:,}. "
                f"For high dimensions (d>3), consider meshfree methods.",
                category=UserWarning,
                stacklevel=2,
            )

        if len(bounds) != dimension or len(num_points) != dimension:
            raise ValueError(
                f"bounds and num_points must have length {dimension}, got {len(bounds)} and {len(num_points)}"
            )

        self.dimension = dimension
        self.bounds = list(bounds)
        self.num_points = list(num_points)
        self.spacing_type = spacing_type

        # Create coordinate arrays
        if spacing_type == "uniform":
            self.coordinates = [np.linspace(bounds[i][0], bounds[i][1], num_points[i]) for i in range(dimension)]
            self.spacing = [
                (bounds[i][1] - bounds[i][0]) / (num_points[i] - 1) if num_points[i] > 1 else 0.0
                for i in range(dimension)
            ]
            self.is_uniform = True

        elif spacing_type == "custom":
            if custom_coordinates is None:
                raise ValueError("custom_coordinates required for spacing_type='custom'")
            if len(custom_coordinates) != dimension:
                raise ValueError(f"custom_coordinates must have length {dimension}")

            self.coordinates = [np.asarray(coords) for coords in custom_coordinates]
            self.spacing = [None] * dimension  # Variable spacing
            self.is_uniform = False

        else:
            raise ValueError(f"Unknown spacing_type: {spacing_type}")

        # Validate coordinates
        for i, coords in enumerate(self.coordinates):
            if len(coords) != num_points[i]:
                raise ValueError(f"Coordinate array {i} has length {len(coords)}, expected {num_points[i]}")

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

        Example:
            >>> grid = TensorProductGrid(2, [(0,1), (0,1)], [3, 3])
            >>> points = grid.flatten()
            >>> points.shape
            (9, 2)
        """
        mesh = self.meshgrid(indexing="ij")
        return np.column_stack([m.ravel() for m in mesh])

    def total_points(self) -> int:
        """
        Get total number of grid points.

        Returns:
            N = ∏num_points[i]
        """
        return int(np.prod(self.num_points))

    def get_index(self, multi_index: Sequence[int]) -> int:
        """
        Convert multi-dimensional index to flat index.

        Args:
            multi_index: Tuple (i, j, k) of indices in each dimension

        Returns:
            Flat index for accessing 1D arrays

        Example:
            >>> grid = TensorProductGrid(2, [(0,1), (0,1)], [10, 10])
            >>> flat_idx = grid.get_index((5, 3))  # Point (i=5, j=3)
        """
        if len(multi_index) != self.dimension:
            raise ValueError(f"multi_index must have length {self.dimension}")

        flat_idx = 0
        stride = 1
        for i in reversed(range(self.dimension)):
            flat_idx += multi_index[i] * stride
            stride *= self.num_points[i]

        return flat_idx

    def get_multi_index(self, flat_index: int) -> tuple[int, ...]:
        """
        Convert flat index to multi-dimensional index.

        Args:
            flat_index: Flat index in [0, total_points)

        Returns:
            Tuple (i, j, k) of indices in each dimension

        Example:
            >>> grid = TensorProductGrid(2, [(0,1), (0,1)], [10, 10])
            >>> i, j = grid.get_multi_index(53)
        """
        if flat_index < 0 or flat_index >= self.total_points():
            raise ValueError(f"flat_index {flat_index} out of range [0, {self.total_points()})")

        indices = []
        remaining = flat_index
        # Process dimensions in row-major order (C-order)
        for i in range(self.dimension):
            stride = int(np.prod(self.num_points[i + 1 :])) if i < self.dimension - 1 else 1
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
        if dimension_idx >= self.dimension:
            raise ValueError(f"dimension_idx {dimension_idx} >= dimension {self.dimension}")

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
            >>> grid = TensorProductGrid(2, [(0,1), (0,1)], [10, 10])
            >>> fine_grid = grid.refine(2)  # Now 20×20 points
        """
        if isinstance(factor, int):
            factors = [factor] * self.dimension
        else:
            factors = list(factor)

        new_num_points = [(n - 1) * f + 1 for n, f in zip(self.num_points, factors, strict=False)]

        return TensorProductGrid(
            dimension=self.dimension,
            bounds=self.bounds,
            num_points=new_num_points,
            spacing_type=self.spacing_type,
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
            factors = [factor] * self.dimension
        else:
            factors = list(factor)

        new_num_points = [(n - 1) // f + 1 for n, f in zip(self.num_points, factors, strict=False)]

        return TensorProductGrid(
            dimension=self.dimension,
            bounds=self.bounds,
            num_points=new_num_points,
            spacing_type=self.spacing_type,
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
            for i in range(self.dimension):
                spacings = self.get_spacing(i)
                idx = multi_index[i]
                # Use average of left and right spacing
                if idx == 0:
                    local_spacing = spacings[0]
                elif idx == self.num_points[i] - 1:
                    local_spacing = spacings[-1]
                else:
                    local_spacing = 0.5 * (spacings[idx - 1] + spacings[idx])
                vol *= local_spacing

            return vol

    def __repr__(self) -> str:
        """String representation of grid."""
        return (
            f"TensorProductGrid(\n"
            f"  dimension={self.dimension},\n"
            f"  bounds={self.bounds},\n"
            f"  num_points={self.num_points},\n"
            f"  spacing_type='{self.spacing_type}',\n"
            f"  total_points={self.total_points()}\n"
            f")"
        )
