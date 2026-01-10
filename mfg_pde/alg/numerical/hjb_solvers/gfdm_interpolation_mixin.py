"""
GFDM Grid-Collocation Interpolation Mixin for HJB Solvers.

This module provides grid-to-collocation and collocation-to-grid mapping:
- Grid to collocation: Uses RegularGridInterpolator for smooth mapping
- Collocation to grid: Uses Delaunay triangulation with barycentric interpolation
- Batch versions for time-series data

These methods are extracted from HJBGFDMSolver to improve maintainability.
"""

from __future__ import annotations

import numpy as np

from mfg_pde.utils.numerical.particle.interpolation import (
    interpolate_grid_to_particles,
)


class GFDMInterpolationMixin:
    """
    Mixin providing grid-collocation interpolation methods for GFDM solvers.

    This mixin enables seamless conversion between:
    - Regular grid representation (for visualization, coupling with FDM solvers)
    - Collocation point representation (for GFDM computation)

    The collocation-to-grid direction uses a pre-computed sparse interpolation
    matrix based on Delaunay triangulation for efficiency.

    Expected attributes from host class:
    - collocation_points: np.ndarray of shape (N, dim)
    - n_points: int
    - domain_bounds: list of (min, max) tuples
    - _output_spatial_shape: tuple (optional, for output grid shape)
    - _n_spatial_grid_points: int (fallback for 1D output shape)
    """

    # Type hints for attributes expected from host class
    collocation_points: np.ndarray
    n_points: int
    domain_bounds: list

    def _map_grid_to_collocation(self, u_grid: np.ndarray) -> np.ndarray:
        """
        Map values from regular grid to collocation points using interpolation.

        Uses scipy RegularGridInterpolator via particle interpolation utilities.

        Args:
            u_grid: Values on regular grid, flattened to 1D

        Returns:
            Values at collocation points, shape (n_points,)
        """
        # Get spatial shape from stored attribute or infer from grid size
        if hasattr(self, "_output_spatial_shape"):
            spatial_shape = self._output_spatial_shape
        else:
            # Assume 1D if not set
            spatial_shape = (len(u_grid),)

        # Reshape to spatial dimensions
        u_grid_shaped = u_grid.reshape(spatial_shape)

        # Squeeze singleton dimensions for proper interpolation
        # e.g., (21, 1) -> (21,) for 1D problems
        u_grid_squeezed = np.squeeze(u_grid_shaped)

        # Format bounds for interpolation utility (matches squeezed dimensions)
        grid_bounds = self._format_grid_bounds_for_interpolation()

        return interpolate_grid_to_particles(
            u_grid_squeezed,
            grid_bounds=grid_bounds,
            particle_positions=self.collocation_points,
            method="linear",
        )

    def _map_collocation_to_grid(self, u_collocation: np.ndarray) -> np.ndarray:
        """
        Map values from collocation points to regular grid using cached interpolation.

        Uses pre-computed interpolation matrix for efficiency (O(n*m) vs O(n^3)).

        Args:
            u_collocation: Values at collocation points, shape (n_points,)

        Returns:
            Values on regular grid, flattened to 1D
        """
        # Use cached interpolation matrix if available
        if hasattr(self, "_interp_matrix") and self._interp_matrix is not None:
            return self._interp_matrix @ u_collocation

        # Fallback: build interpolation matrix on first call
        self._build_interpolation_matrix()
        return self._interp_matrix @ u_collocation

    def _build_interpolation_matrix(self) -> None:
        """
        Pre-compute interpolation matrix for collocation->grid mapping.

        Uses scipy LinearNDInterpolator with barycentric weights for efficiency.
        The matrix is sparse due to triangulation locality.
        """
        from scipy.spatial import Delaunay

        # Get spatial shape
        if hasattr(self, "_output_spatial_shape"):
            spatial_shape = self._output_spatial_shape
        else:
            Nx_grid = self._n_spatial_grid_points
            spatial_shape = (Nx_grid + 1,)

        non_singleton = tuple(s for s in spatial_shape if s > 1)
        squeezed_shape = non_singleton if non_singleton else (spatial_shape[0],)

        grid_bounds = self._format_grid_bounds_for_interpolation()
        ndim = len(squeezed_shape)

        # Create grid points
        if ndim == 1:
            xmin, xmax = grid_bounds if not isinstance(grid_bounds[0], (tuple, list)) else grid_bounds[0]
            x = np.linspace(xmin, xmax, squeezed_shape[0])
            grid_points = x.reshape(-1, 1)
        elif ndim == 2:
            (xmin, xmax), (ymin, ymax) = grid_bounds
            x = np.linspace(xmin, xmax, squeezed_shape[0])
            y = np.linspace(ymin, ymax, squeezed_shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")
            grid_points = np.column_stack([X.ravel(), Y.ravel()])
        else:
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = grid_bounds
            x = np.linspace(xmin, xmax, squeezed_shape[0])
            y = np.linspace(ymin, ymax, squeezed_shape[1])
            z = np.linspace(zmin, zmax, squeezed_shape[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        n_grid = grid_points.shape[0]
        n_coll = self.n_points

        # Build interpolation matrix using Delaunay triangulation
        # Each grid point is expressed as weighted sum of collocation points
        try:
            tri = Delaunay(self.collocation_points)
            simplex_indices = tri.find_simplex(grid_points)

            # Build sparse matrix
            from scipy.sparse import lil_matrix

            interp_matrix = lil_matrix((n_grid, n_coll))

            for i, (pt, simplex_idx) in enumerate(zip(grid_points, simplex_indices, strict=True)):
                if simplex_idx >= 0:
                    # Point inside triangulation - use barycentric coords
                    simplex = tri.simplices[simplex_idx]
                    b = tri.transform[simplex_idx, :ndim].dot(pt - tri.transform[simplex_idx, ndim])
                    bary = np.append(b, 1 - b.sum())
                    for j, weight in zip(simplex, bary, strict=True):
                        interp_matrix[i, j] = weight
                else:
                    # Point outside - use nearest neighbor
                    from scipy.spatial import cKDTree

                    if not hasattr(self, "_coll_tree"):
                        self._coll_tree = cKDTree(self.collocation_points)
                    _, nearest_idx = self._coll_tree.query(pt)
                    interp_matrix[i, nearest_idx] = 1.0

            self._interp_matrix = interp_matrix.tocsr()
        except Exception as e:
            raise RuntimeError(f"Failed to build interpolation matrix: {e}") from e

    def _format_grid_bounds_for_interpolation(self) -> tuple:
        """Format domain_bounds for interpolation utility."""
        if len(self.domain_bounds) == 1:
            # 1D: return (xmin, xmax)
            return self.domain_bounds[0]
        else:
            # nD: return tuple of (min, max) tuples
            return tuple(self.domain_bounds)

    def _map_grid_to_collocation_batch(self, U_grid: np.ndarray) -> np.ndarray:
        """
        Batch version of _map_grid_to_collocation.

        Args:
            U_grid: Shape (Nt, ...) where ... represents spatial dimensions

        Returns:
            U_collocation: Shape (Nt, n_points)
        """
        Nt = U_grid.shape[0]
        U_collocation = np.zeros((Nt, self.n_points))

        for n in range(Nt):
            u_grid_flat = U_grid[n].flatten()
            U_collocation[n, :] = self._map_grid_to_collocation(u_grid_flat)

        return U_collocation

    def _map_collocation_to_grid_batch(self, U_collocation: np.ndarray) -> np.ndarray:
        """
        Batch version of _map_collocation_to_grid.

        Args:
            U_collocation: Shape (Nt, n_points)

        Returns:
            U_grid: Shape (Nt, ...) where ... matches original spatial shape
        """
        Nt = U_collocation.shape[0]

        # Get the original spatial shape
        if hasattr(self, "_output_spatial_shape"):
            spatial_shape = self._output_spatial_shape
        else:
            Nx_grid = self._n_spatial_grid_points
            spatial_shape = (Nx_grid + 1,)

        n_spatial_points = int(np.prod(spatial_shape))

        # Map each timestep
        U_grid_flat = np.zeros((Nt, n_spatial_points))
        for n in range(Nt):
            U_grid_flat[n, :] = self._map_collocation_to_grid(U_collocation[n, :])

        # Reshape to original spatial dimensions
        output_shape = (Nt, *tuple(spatial_shape))
        return U_grid_flat.reshape(output_shape)
