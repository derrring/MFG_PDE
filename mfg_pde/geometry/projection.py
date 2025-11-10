"""
Geometry projection for hybrid MFG solvers.

This module provides infrastructure for projecting solution values between
different geometry discretizations, enabling hybrid solvers where HJB and FP
equations use different spatial representations.

Key Components:
    GeometryProjector: Base class for inter-geometry projections
    ParticleToGridProjector: Specialized for particle → grid (KDE-based)
    GridToParticleProjector: Specialized for grid → particle (interpolation)
    GridToGridProjector: For different grid resolutions

Example:
    >>> # Particle FP + Grid HJB hybrid solver
    >>> from mfg_pde.geometry import SimpleGrid2D
    >>> from mfg_pde.geometry.projection import GeometryProjector
    >>>
    >>> hjb_grid = SimpleGrid2D(bounds=(0, 1, 0, 1), resolution=(50, 50))
    >>> # Assume particle_geometry exists with particles
    >>>
    >>> projector = GeometryProjector.create(
    ...     hjb_geometry=hjb_grid,
    ...     fp_geometry=particle_geometry
    ... )
    >>>
    >>> # HJB solver produces U on grid
    >>> U_on_grid = hjb_solver.solve(...)
    >>>
    >>> # Project to particle locations for FP solver
    >>> U_at_particles = projector.project_hjb_to_fp(U_on_grid)
    >>>
    >>> # FP solver produces density at particles
    >>> M_at_particles = fp_solver.solve(U_at_particles, ...)
    >>>
    >>> # Project back to grid for next HJB iteration
    >>> M_on_grid = projector.project_fp_to_hjb(M_at_particles)

References:
    - Issue #257: Dual Geometry Architecture for Hybrid MFG Solvers
    - Achdou & Capuzzo-Dolcetta (2010): Mean field games: numerical methods
    - Particle-mesh methods in computational physics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.backends.backend_protocol import BaseBackend
    from mfg_pde.geometry.base import BaseGeometry


class GeometryProjector:
    """
    Base class for projecting solution values between different geometries.

    Handles mapping of:
    - Value functions U(x): HJB geometry → FP geometry
    - Density functions m(x): FP geometry → HJB geometry

    The projector automatically selects appropriate methods based on
    geometry types (particle, grid, implicit, etc.).
    """

    def __init__(
        self,
        hjb_geometry: BaseGeometry,
        fp_geometry: BaseGeometry,
        projection_method: Literal["auto", "kde", "interpolation", "nearest"] = "auto",
    ):
        """
        Initialize geometry projector.

        Args:
            hjb_geometry: Geometry used by HJB solver
            fp_geometry: Geometry used by FP solver
            projection_method: Projection strategy
                - "auto": Auto-detect based on geometry types (default)
                - "kde": Kernel density estimation (particle → grid)
                - "interpolation": Linear/bilinear interpolation (grid → particle)
                - "nearest": Nearest neighbor (generic fallback)
        """
        self.hjb_geometry = hjb_geometry
        self.fp_geometry = fp_geometry
        self.projection_method = projection_method

        # Cache for geometry properties
        self._hjb_points: NDArray | None = None
        self._fp_points: NDArray | None = None

        # Detect projection methods if auto
        if projection_method == "auto":
            self._setup_projection_methods()
        else:
            self.hjb_to_fp_method = projection_method
            self.fp_to_hjb_method = projection_method

    def _setup_projection_methods(self) -> None:
        """Auto-detect appropriate projection methods based on geometry types."""
        from mfg_pde.geometry.base import CartesianGrid

        hjb_is_grid = isinstance(self.hjb_geometry, CartesianGrid)
        fp_is_grid = isinstance(self.fp_geometry, CartesianGrid)

        # Check for particle-like geometry (has num_particles attribute)
        fp_is_particles = hasattr(self.fp_geometry, "num_particles")

        if hjb_is_grid and fp_is_particles:
            # Most common: Grid HJB + Particle FP
            self.hjb_to_fp_method = "interpolation"  # Grid → Particle
            self.fp_to_hjb_method = "kde"  # Particle → Grid
        elif hjb_is_grid and fp_is_grid:
            # Both grids (possibly different resolution)
            self.hjb_to_fp_method = "grid_interpolation"
            self.fp_to_hjb_method = "grid_restriction"
        else:
            # Generic fallback
            self.hjb_to_fp_method = "interpolation"
            self.fp_to_hjb_method = "nearest"

    def project_hjb_to_fp(self, U_on_hjb_geometry: NDArray, backend: BaseBackend | None = None) -> NDArray:
        """
        Project HJB value function from HJB geometry to FP geometry.

        This maps the solution U(x) from the HJB solver's discretization
        to the FP solver's discretization, typically via interpolation.

        Args:
            U_on_hjb_geometry: Value function on HJB geometry
                Shape depends on HJB geometry (e.g., (Nx, Ny) for 2D grid)
            backend: Optional backend for accelerated operations

        Returns:
            Value function on FP geometry
                Shape depends on FP geometry (e.g., (N_particles,) for particles)

        Examples:
            >>> # Grid → Particles: Interpolate at particle locations
            >>> U_on_grid = np.random.rand(50, 50)
            >>> U_at_particles = projector.project_hjb_to_fp(U_on_grid)
            >>> U_at_particles.shape  # (10000,) for 10k particles
        """
        # Same geometry: no projection needed
        if self.hjb_geometry is self.fp_geometry:
            return U_on_hjb_geometry

        # Get FP spatial points (where to evaluate U)
        if self._fp_points is None:
            self._fp_points = self.fp_geometry.get_spatial_grid()

        # Apply appropriate projection method
        if self.hjb_to_fp_method == "interpolation":
            return self._interpolate_grid_to_points(U_on_hjb_geometry, self._fp_points)
        elif self.hjb_to_fp_method == "grid_interpolation":
            return self._interpolate_grid_to_grid(U_on_hjb_geometry)
        elif self.hjb_to_fp_method == "nearest":
            return self._nearest_neighbor_projection(U_on_hjb_geometry, self._fp_points)
        else:
            raise ValueError(f"Unknown HJB→FP projection method: {self.hjb_to_fp_method}")

    def project_fp_to_hjb(
        self,
        M_on_fp_geometry: NDArray,
        bandwidth: str | float = "scott",
        backend: BaseBackend | None = None,
    ) -> NDArray:
        """
        Project FP density from FP geometry to HJB geometry.

        This maps the density m(x) from the FP solver's discretization
        to the HJB solver's discretization, typically via KDE or restriction.

        Args:
            M_on_fp_geometry: Density on FP geometry
                Shape depends on FP geometry (e.g., (N_particles,) for particles)
            bandwidth: KDE bandwidth for particle → grid projection
                - "scott": Scott's rule (default)
                - "silverman": Silverman's rule
                - float: Manual bandwidth factor
            backend: Optional backend for GPU-accelerated KDE

        Returns:
            Density on HJB geometry
                Shape depends on HJB geometry (e.g., (Nx, Ny) for 2D grid)

        Examples:
            >>> # Particles → Grid: Kernel density estimation
            >>> M_particles = np.random.rand(10000)  # Particle masses
            >>> M_on_grid = projector.project_fp_to_hjb(M_particles, bandwidth="scott")
            >>> M_on_grid.shape  # (50, 50) for 50×50 grid
        """
        # Same geometry: no projection needed
        if self.hjb_geometry is self.fp_geometry:
            return M_on_fp_geometry

        # Apply appropriate projection method
        if self.fp_to_hjb_method == "kde":
            return self._project_particles_to_grid_kde(M_on_fp_geometry, bandwidth, backend)
        elif self.fp_to_hjb_method == "grid_restriction":
            return self._restrict_grid_to_grid(M_on_fp_geometry)
        elif self.fp_to_hjb_method == "nearest":
            if self._hjb_points is None:
                self._hjb_points = self.hjb_geometry.get_spatial_grid()
            return self._nearest_neighbor_projection(M_on_fp_geometry, self._hjb_points)
        else:
            raise ValueError(f"Unknown FP→HJB projection method: {self.fp_to_hjb_method}")

    def _interpolate_grid_to_points(self, values_on_grid: NDArray, target_points: NDArray) -> NDArray:
        """
        Interpolate grid values to arbitrary points.

        Uses linear/bilinear/trilinear interpolation for Cartesian grids.
        """
        from mfg_pde.geometry.base import CartesianGrid

        if not isinstance(self.hjb_geometry, CartesianGrid):
            raise ValueError("Source geometry must be CartesianGrid for grid interpolation")

        # Get interpolation function from geometry
        interpolator = self.hjb_geometry.get_interpolator()
        return interpolator(values_on_grid, target_points)

    def _interpolate_grid_to_grid(self, values_on_source_grid: NDArray) -> NDArray:
        """Interpolate from one grid to another grid (possibly different resolution)."""
        from mfg_pde.geometry.base import CartesianGrid

        # Get target grid points
        target_points = self.fp_geometry.get_spatial_grid()

        # Use standard grid→points interpolation
        values_flat = self._interpolate_grid_to_points(values_on_source_grid, target_points)

        # Reshape to target grid shape
        if isinstance(self.fp_geometry, CartesianGrid):
            target_shape = self.fp_geometry.get_grid_shape()
            return values_flat.reshape(target_shape)

        return values_flat

    def _project_particles_to_grid_kde(
        self, particle_values: NDArray, bandwidth: str | float, backend: BaseBackend | None
    ) -> NDArray:
        """
        Project particle density to grid using kernel density estimation.

        Leverages existing GPU-accelerated KDE from density_estimation.py.
        """
        from mfg_pde.alg.numerical.density_estimation import gaussian_kde_gpu
        from mfg_pde.backends import create_backend
        from mfg_pde.geometry.base import CartesianGrid

        if backend is None:
            backend = create_backend("numpy")

        if not isinstance(self.hjb_geometry, CartesianGrid):
            raise ValueError("Target geometry must be CartesianGrid for KDE projection")

        # Get particle positions
        particle_positions = self.fp_geometry.get_spatial_grid()  # (N_particles, d)

        # Get target grid
        grid_shape = self.hjb_geometry.get_grid_shape()
        hjb_points = self.hjb_geometry.get_spatial_grid()  # (N_grid, d)

        dimension = particle_positions.shape[1] if particle_positions.ndim > 1 else 1

        # Handle bandwidth string to numeric conversion
        if isinstance(bandwidth, str):
            if bandwidth == "scott":
                bw_factor = len(particle_positions) ** (-1.0 / (dimension + 4))
            elif bandwidth == "silverman":
                bw_factor = (len(particle_positions) * (dimension + 2) / 4.0) ** (-1.0 / (dimension + 4))
            else:
                bw_factor = 1.0
        else:
            bw_factor = bandwidth

        # 1D case: use optimized GPU KDE
        if dimension == 1:
            density = gaussian_kde_gpu(
                particles=particle_positions.ravel(),
                grid=hjb_points.ravel(),
                bandwidth=bw_factor,
                backend=backend,
            )
            return density.reshape(grid_shape)

        # Multi-dimensional: use histogram-based density (fast fallback)
        # TODO: Implement multi-D GPU KDE in future
        return self._histogram_density_nd(particle_positions, particle_values, grid_shape)

    def _histogram_density_nd(
        self, particle_positions: NDArray, particle_weights: NDArray, grid_shape: tuple[int, ...]
    ) -> NDArray:
        """
        Multi-dimensional histogram-based density estimation.

        Fast fallback for nD particle → grid projection.
        """
        # Get grid bounds
        bounds = self.hjb_geometry.bounds  # (min_coords, max_coords)

        dimension = len(grid_shape)

        # Build range list for histogramdd
        hist_range = [(bounds[0][i], bounds[1][i]) for i in range(dimension)]

        # Compute weighted histogram
        hist, _ = np.histogramdd(particle_positions, bins=grid_shape, range=hist_range, weights=particle_weights)

        # Normalize to probability density
        cell_volume = np.prod([(bounds[1][i] - bounds[0][i]) / grid_shape[i] for i in range(dimension)])

        total_mass = np.sum(hist)
        if total_mass > 1e-12:
            return hist / (total_mass * cell_volume)
        else:
            # Uniform density if no particles
            return np.ones(grid_shape) / (np.prod(grid_shape) * cell_volume)

    def _restrict_grid_to_grid(self, values_on_fine_grid: NDArray) -> NDArray:
        """
        Restrict values from fine grid to coarse grid.

        Uses averaging for mass-conserving restriction.
        """
        from mfg_pde.geometry.base import CartesianGrid

        if not (isinstance(self.fp_geometry, CartesianGrid) and isinstance(self.hjb_geometry, CartesianGrid)):
            raise ValueError("Both geometries must be CartesianGrid for grid restriction")

        # For now, use interpolation (proper restriction operator would be more accurate)
        # TODO: Implement proper restriction/prolongation operators
        target_points = self.hjb_geometry.get_spatial_grid()
        interpolator = self.fp_geometry.get_interpolator()
        values_flat = interpolator(values_on_fine_grid, target_points)

        # Reshape to target grid shape
        if isinstance(self.hjb_geometry, CartesianGrid):
            target_shape = self.hjb_geometry.get_grid_shape()
            return values_flat.reshape(target_shape)

        return values_flat

    def _nearest_neighbor_projection(self, source_values: NDArray, target_points: NDArray) -> NDArray:
        """
        Generic nearest neighbor projection.

        Fallback method for arbitrary geometry pairs.
        """
        from scipy.spatial import cKDTree

        # Get source points
        source_points = (
            self.fp_geometry.get_spatial_grid()
            if source_values.shape == self.fp_geometry.get_spatial_grid().shape[:1]
            else self.hjb_geometry.get_spatial_grid()
        )

        # Build KD-tree for fast nearest neighbor lookup
        tree = cKDTree(source_points)
        _, indices = tree.query(target_points)

        return source_values.ravel()[indices]

    @classmethod
    def create(
        cls,
        hjb_geometry: BaseGeometry,
        fp_geometry: BaseGeometry,
        projection_method: Literal["auto", "kde", "interpolation", "nearest"] = "auto",
    ) -> GeometryProjector:
        """
        Factory method to create appropriate projector based on geometry types.

        Automatically detects specialized projector classes when available.

        Args:
            hjb_geometry: Geometry used by HJB solver
            fp_geometry: Geometry used by FP solver
            projection_method: Override auto-detection if needed

        Returns:
            GeometryProjector instance (or specialized subclass)

        Examples:
            >>> projector = GeometryProjector.create(grid, particles)
            >>> # Returns specialized ParticleToGridProjector if available
        """
        # For now, return base class (can add specialized subclasses later)
        return cls(hjb_geometry, fp_geometry, projection_method)


# Future: Specialized projector subclasses can be added here
# class ParticleToGridProjector(GeometryProjector):
#     """Optimized projector for particle → grid conversions."""
#     ...
#
# class GridToParticleProjector(GeometryProjector):
#     """Optimized projector for grid → particle conversions."""
#     ...
