"""
Grid-Collocation Mapping Component for GFDM Solvers.

This component provides bidirectional mapping between:
- Regular grid representation (for visualization, coupling with FDM solvers)
- Collocation point representation (for GFDM computation)

Extracted from GFDMInterpolationMixin as part of Issue #545 (mixin refactoring).
Uses composition pattern instead of inheritance for better testability and reusability.
"""

from __future__ import annotations

import numpy as np

from mfg_pde.utils.numerical.particle.interpolation import (
    interpolate_grid_to_particles,
)


class GridCollocationMapper:
    """
    Handles grid-to-collocation and collocation-to-grid interpolation for GFDM.

    This component enables seamless conversion between:
    - Regular grid representation (structured, for output/visualization)
    - Collocation point representation (unstructured, for GFDM computation)

    Grid → Collocation: Uses scipy RegularGridInterpolator for smooth mapping
    Collocation → Grid: Uses Delaunay triangulation with barycentric interpolation

    The collocation-to-grid direction uses a pre-computed sparse interpolation
    matrix for efficiency (O(n*m) vs O(n^3) for repeated mappings).

    Composition Pattern (Issue #545):
        This component is injected into HJBGFDMSolver instead of using mixin inheritance.
        Benefits: Testable independently, reusable for FDM/FEM solvers, clear dependencies.

    Parameters
    ----------
    collocation_points : np.ndarray
        Collocation points, shape (n_points, dimension)
    grid_shape : tuple[int, ...]
        Shape of output grid (Nx, Ny, Nz, ...)
    domain_bounds : list[tuple[float, float]]
        Domain bounds per dimension: [(xmin, xmax), (ymin, ymax), ...]

    Attributes
    ----------
    collocation_points : np.ndarray
        Collocation points (N, dimension)
    n_points : int
        Number of collocation points
    grid_shape : tuple[int, ...]
        Output grid shape
    domain_bounds : list[tuple[float, float]]
        Domain bounds per dimension
    _interp_matrix : scipy.sparse matrix | None
        Cached sparse interpolation matrix (collocation → grid)
    _coll_tree : scipy.spatial.cKDTree | None
        KDTree for nearest neighbor fallback

    Examples
    --------
    >>> # Create mapper for 2D GFDM solver
    >>> collocation = np.random.uniform(0, 1, (500, 2))
    >>> mapper = GridCollocationMapper(
    ...     collocation_points=collocation,
    ...     grid_shape=(21, 21),
    ...     domain_bounds=[(0, 1), (0, 1)]
    ... )
    >>>
    >>> # Map grid values to collocation points
    >>> u_grid = np.random.randn(21, 21).flatten()
    >>> u_coll = mapper.map_grid_to_collocation(u_grid)
    >>> assert u_coll.shape == (500,)
    >>>
    >>> # Map collocation values back to grid
    >>> u_grid_reconstructed = mapper.map_collocation_to_grid(u_coll)
    >>> assert u_grid_reconstructed.shape == (441,)  # 21*21
    """

    def __init__(
        self,
        collocation_points: np.ndarray,
        grid_shape: tuple[int, ...],
        domain_bounds: list[tuple[float, float]],
    ):
        """Initialize grid-collocation mapper."""
        self.collocation_points = np.asarray(collocation_points)
        self.n_points = len(collocation_points)
        self.grid_shape = grid_shape
        self.domain_bounds = domain_bounds

        # Cached interpolation matrix (built lazily on first use)
        self._interp_matrix = None
        self._coll_tree = None  # KDTree for nearest neighbor fallback

    def map_grid_to_collocation(self, u_grid: np.ndarray) -> np.ndarray:
        """
        Map values from regular grid to collocation points using interpolation.

        Uses scipy RegularGridInterpolator via particle interpolation utilities.

        Parameters
        ----------
        u_grid : np.ndarray
            Values on regular grid, flattened to 1D

        Returns
        -------
        np.ndarray
            Values at collocation points, shape (n_points,)
        """
        # Reshape to spatial dimensions
        u_grid_shaped = u_grid.reshape(self.grid_shape)

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

    def map_collocation_to_grid(self, u_collocation: np.ndarray) -> np.ndarray:
        """
        Map values from collocation points to regular grid using cached interpolation.

        Uses pre-computed interpolation matrix for efficiency (O(n*m) vs O(n^3)).

        Parameters
        ----------
        u_collocation : np.ndarray
            Values at collocation points, shape (n_points,)

        Returns
        -------
        np.ndarray
            Values on regular grid, flattened to 1D
        """
        # Build interpolation matrix on first use (lazy initialization)
        if self._interp_matrix is None:
            self._build_interpolation_matrix()

        return self._interp_matrix @ u_collocation

    def map_grid_to_collocation_batch(self, U_grid: np.ndarray) -> np.ndarray:
        """
        Batch version of map_grid_to_collocation for time-series data.

        Parameters
        ----------
        U_grid : np.ndarray
            Shape (Nt, ...) where ... represents spatial dimensions

        Returns
        -------
        np.ndarray
            U_collocation with shape (Nt, n_points)
        """
        Nt = U_grid.shape[0]
        U_collocation = np.zeros((Nt, self.n_points))

        for n in range(Nt):
            u_grid_flat = U_grid[n].flatten()
            U_collocation[n, :] = self.map_grid_to_collocation(u_grid_flat)

        return U_collocation

    def map_collocation_to_grid_batch(self, U_collocation: np.ndarray) -> np.ndarray:
        """
        Batch version of map_collocation_to_grid for time-series data.

        Parameters
        ----------
        U_collocation : np.ndarray
            Shape (Nt, n_points)

        Returns
        -------
        np.ndarray
            U_grid with shape (Nt, ...) where ... matches original spatial shape
        """
        Nt = U_collocation.shape[0]
        n_spatial_points = int(np.prod(self.grid_shape))

        # Map each timestep
        U_grid_flat = np.zeros((Nt, n_spatial_points))
        for n in range(Nt):
            U_grid_flat[n, :] = self.map_collocation_to_grid(U_collocation[n, :])

        # Reshape to original spatial dimensions
        output_shape = (Nt, *self.grid_shape)
        return U_grid_flat.reshape(output_shape)

    def _build_interpolation_matrix(self) -> None:
        """
        Pre-compute sparse interpolation matrix for collocation → grid mapping.

        Uses scipy LinearNDInterpolator with barycentric weights for efficiency.
        The matrix is sparse due to triangulation locality.

        For 1D: Uses linear interpolation between sorted collocation points
        For 2D+: Uses Delaunay triangulation with barycentric coordinates
        """
        from scipy.spatial import Delaunay

        # Remove singleton dimensions from grid shape
        non_singleton = tuple(s for s in self.grid_shape if s > 1)
        squeezed_shape = non_singleton if non_singleton else (self.grid_shape[0],)

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
            # Special handling for 1D: Delaunay doesn't work for 1D points
            if ndim == 1:
                # For 1D, use linear interpolation weights
                from scipy.sparse import lil_matrix

                interp_matrix = lil_matrix((n_grid, n_coll))

                # Sort collocation points for 1D interpolation
                coll_x = self.collocation_points[:, 0]
                sort_idx = np.argsort(coll_x)
                coll_x_sorted = coll_x[sort_idx]
                grid_x = grid_points[:, 0]

                # For each grid point, find surrounding collocation points and compute weights
                for i, x in enumerate(grid_x):
                    # Find where x would be inserted to maintain sorted order
                    idx_right = np.searchsorted(coll_x_sorted, x)

                    if idx_right == 0:
                        # x is before all collocation points - use nearest
                        interp_matrix[i, sort_idx[0]] = 1.0
                    elif idx_right == n_coll:
                        # x is after all collocation points - use nearest
                        interp_matrix[i, sort_idx[-1]] = 1.0
                    else:
                        # x is between two collocation points - use linear interpolation
                        idx_left = idx_right - 1
                        x_left = coll_x_sorted[idx_left]
                        x_right = coll_x_sorted[idx_right]

                        # Linear interpolation weights
                        weight_right = (x - x_left) / (x_right - x_left)
                        weight_left = 1.0 - weight_right

                        interp_matrix[i, sort_idx[idx_left]] = weight_left
                        interp_matrix[i, sort_idx[idx_right]] = weight_right

                self._interp_matrix = interp_matrix.tocsr()
            else:
                # For 2D+, use Delaunay triangulation
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

                        if self._coll_tree is None:
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
