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

from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfg_pde.backends.backend_protocol import BaseBackend
    from mfg_pde.geometry.base import BaseGeometry


class ProjectionRegistry:
    """
    Registry for specialized geometry projectors with hierarchical fallback.

    Allows registration of optimized projector functions for specific geometry
    type pairs, avoiding O(N²) implementations while supporting extensibility.

    Registration Pattern:
        @ProjectionRegistry.register(SimpleGrid2D, SimpleGrid2D)
        def optimized_grid2d_projector(source, target, values):
            # Optimized 2D grid-to-grid projection
            return projected_values

    Lookup Hierarchy:
        1. Exact type match: SimpleGrid2D → SimpleGrid2D
        2. Category match: CartesianGrid → CartesianGrid (base classes)
        3. None if no match found (caller uses generic fallback)

    This pattern scales to O(N) specialized projectors, not O(N²).
    """

    _registry: ClassVar[dict[tuple[type, type, str], Callable]] = {}

    @classmethod
    def register(cls, source_type: type, target_type: type, direction: Literal["hjb_to_fp", "fp_to_hjb"] = "hjb_to_fp"):
        """
        Decorator to register a specialized projector function.

        Args:
            source_type: Source geometry type (e.g., SimpleGrid2D)
            target_type: Target geometry type (e.g., SimpleGrid2D)
            direction: Projection direction ("hjb_to_fp" or "fp_to_hjb")

        Returns:
            Decorator function

        Example:
            @ProjectionRegistry.register(SimpleGrid2D, SimpleGrid2D, "hjb_to_fp")
            def fast_grid2d_interpolation(source, target, values, **kwargs):
                # Optimized implementation
                return projected_values
        """

        def decorator(func: Callable) -> Callable:
            key = (source_type, target_type, direction)
            cls._registry[key] = func
            return func

        return decorator

    @classmethod
    def get_projector(
        cls, source: BaseGeometry, target: BaseGeometry, direction: Literal["hjb_to_fp", "fp_to_hjb"]
    ) -> Callable | None:
        """
        Get specialized projector with hierarchical fallback.

        Lookup order:
        1. Exact type match: (type(source), type(target), direction)
        2. Category match: isinstance checks against registered base types
        3. None if no match

        Args:
            source: Source geometry instance
            target: Target geometry instance
            direction: Projection direction

        Returns:
            Projector function or None if no specialized projector found
        """
        # Try exact type match first
        exact_key = (type(source), type(target), direction)
        if exact_key in cls._registry:
            return cls._registry[exact_key]

        # Try category match (isinstance checks)
        for (src_type, tgt_type, reg_direction), func in cls._registry.items():
            if reg_direction == direction:
                if isinstance(source, src_type) and isinstance(target, tgt_type):
                    return func

        # No specialized projector found
        return None

    @classmethod
    def list_registered(cls) -> list[tuple[type, type, str]]:
        """
        List all registered projector type pairs.

        Returns:
            List of (source_type, target_type, direction) tuples
        """
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered projectors (mainly for testing)."""
        cls._registry.clear()


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

        # Cache for registered projectors
        self._hjb_to_fp_func: Callable | None = None
        self._fp_to_hjb_func: Callable | None = None

        # Detect projection methods if auto
        if projection_method == "auto":
            self._setup_projection_methods()
        else:
            self.hjb_to_fp_method = projection_method
            self.fp_to_hjb_method = projection_method

    def _setup_projection_methods(self) -> None:
        """Auto-detect appropriate projection methods based on geometry types."""
        from mfg_pde.geometry.base import CartesianGrid

        # Check registry first for specialized projectors
        self._hjb_to_fp_func = ProjectionRegistry.get_projector(self.hjb_geometry, self.fp_geometry, "hjb_to_fp")
        self._fp_to_hjb_func = ProjectionRegistry.get_projector(self.fp_geometry, self.hjb_geometry, "fp_to_hjb")

        # If registered projector found, use registry method
        if self._hjb_to_fp_func is not None:
            self.hjb_to_fp_method = "registry"
        if self._fp_to_hjb_func is not None:
            self.fp_to_hjb_method = "registry"

        # Otherwise, fall back to auto-detection
        if self._hjb_to_fp_func is None or self._fp_to_hjb_func is None:
            hjb_is_grid = isinstance(self.hjb_geometry, CartesianGrid)
            fp_is_grid = isinstance(self.fp_geometry, CartesianGrid)

            # Check for particle-like geometry (has num_particles attribute)
            fp_is_particles = hasattr(self.fp_geometry, "num_particles")

            if self._hjb_to_fp_func is None:
                if hjb_is_grid and fp_is_particles:
                    self.hjb_to_fp_method = "interpolation"  # Grid → Particle
                elif hjb_is_grid and fp_is_grid:
                    self.hjb_to_fp_method = "grid_interpolation"
                else:
                    self.hjb_to_fp_method = "interpolation"

            if self._fp_to_hjb_func is None:
                if hjb_is_grid and fp_is_particles:
                    self.fp_to_hjb_method = "kde"  # Particle → Grid
                elif hjb_is_grid and fp_is_grid:
                    self.fp_to_hjb_method = "grid_restriction"
                else:
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

        # Try registered projector first
        if self.hjb_to_fp_method == "registry" and self._hjb_to_fp_func is not None:
            return self._hjb_to_fp_func(self.hjb_geometry, self.fp_geometry, U_on_hjb_geometry, backend=backend)

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

        # Try registered projector first
        if self.fp_to_hjb_method == "registry" and self._fp_to_hjb_func is not None:
            return self._fp_to_hjb_func(
                self.fp_geometry, self.hjb_geometry, M_on_fp_geometry, bandwidth=bandwidth, backend=backend
            )

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


# =============================================================================
# FEM Mesh Projections (Delaunay Interpolation)
# =============================================================================


def _mesh_to_grid_delaunay(mesh_geo: BaseGeometry, grid_geo: BaseGeometry, mesh_values: NDArray, **kwargs) -> NDArray:
    """
    Project from FEM unstructured mesh to regular grid using Delaunay interpolation.

    This provides higher accuracy than nearest neighbor by respecting the mesh's
    triangulation structure for linear interpolation.

    Mathematical Formulation:
        For mesh vertices V = {v_i} with values u_i, project to grid points G = {g_j}:
        u(g_j) = ∑_i λ_i(g_j) u_i  where λ_i are barycentric coordinates

    Args:
        mesh_geo: Source FEM mesh (Mesh2D, Mesh3D, etc.)
        grid_geo: Target regular grid (SimpleGrid2D, SimpleGrid3D, etc.)
        mesh_values: Values at mesh vertices (N_vertices,)
        **kwargs: Additional arguments (unused)

    Returns:
        Values on grid (nx+1, ny+1, ...) or (N_grid,)

    Raises:
        ImportError: If scipy not available

    Notes:
        - Uses scipy.interpolate.LinearNDInterpolator internally
        - Points outside mesh convex hull use nearest neighbor extrapolation
        - Complexity: O(N log N) setup + O(log N) per query
        - Preserves C⁰ continuity (unlike nearest neighbor)

    Examples:
        >>> from mfg_pde.geometry import Mesh2D, SimpleGrid2D
        >>> mesh = Mesh2D(domain_type="rectangle", bounds=(0,1,0,1), mesh_size=0.05)
        >>> mesh.generate_mesh()
        >>> grid = SimpleGrid2D(bounds=(0,1,0,1), resolution=(50,50))
        >>> U_mesh = ...  # Values at mesh vertices
        >>> U_grid = _mesh_to_grid_delaunay(mesh, grid, U_mesh)
    """
    try:
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import KDTree
    except ImportError as e:
        raise ImportError(
            "scipy required for Delaunay interpolation. "
            "Install: pip install scipy. "
            "Alternatively, use nearest neighbor fallback (automatic)."
        ) from e

    # Get mesh vertex positions
    vertices = mesh_geo.get_spatial_grid()  # (N_vertices, dimension)

    # Create Delaunay interpolator
    # LinearNDInterpolator automatically uses Delaunay triangulation
    interpolator = LinearNDInterpolator(vertices, mesh_values)

    # Evaluate at grid points
    grid_points = grid_geo.get_spatial_grid()  # (N_grid, dimension)
    grid_values_flat = interpolator(grid_points)

    # Handle extrapolation: points outside mesh convex hull return NaN
    # Use nearest neighbor for these points
    nan_mask = np.isnan(grid_values_flat)
    if np.any(nan_mask):
        tree = KDTree(vertices)
        _, nearest_indices = tree.query(grid_points[nan_mask])
        grid_values_flat[nan_mask] = mesh_values[nearest_indices]

    # Reshape to grid shape if target is CartesianGrid
    from mfg_pde.geometry.base import CartesianGrid

    if isinstance(grid_geo, CartesianGrid):
        grid_shape = grid_geo.get_grid_shape()
        return grid_values_flat.reshape(grid_shape)

    return grid_values_flat


def _grid_to_mesh_interpolation(
    grid_geo: BaseGeometry, mesh_geo: BaseGeometry, grid_values: NDArray, **kwargs
) -> NDArray:
    """
    Project from regular grid to FEM mesh using bilinear/trilinear interpolation.

    This is optimal - grid's built-in interpolator is already efficient for
    evaluating at arbitrary points.

    Mathematical Formulation:
        For grid with bilinear/trilinear basis functions φ_i(x):
        u(v_j) = ∑_i u_i φ_i(v_j)  where v_j are mesh vertices

    Args:
        grid_geo: Source regular grid (SimpleGrid2D, SimpleGrid3D, etc.)
        mesh_geo: Target FEM mesh (Mesh2D, Mesh3D, etc.)
        grid_values: Values on grid (nx+1, ny+1, ...) or (N_grid,)
        **kwargs: Additional arguments (unused)

    Returns:
        Values at mesh vertices (N_vertices,)

    Notes:
        - Uses grid's built-in RegularGridInterpolator
        - Complexity: O(log N) per query with regular grid structure
        - Bilinear (2D) or trilinear (3D) interpolation
        - Points outside grid domain use fill_value=0.0

    Examples:
        >>> from mfg_pde.geometry import SimpleGrid2D, Mesh2D
        >>> grid = SimpleGrid2D(bounds=(0,1,0,1), resolution=(50,50))
        >>> mesh = Mesh2D(domain_type="rectangle", bounds=(0,1,0,1), mesh_size=0.05)
        >>> mesh.generate_mesh()
        >>> U_grid = ...  # Values on grid
        >>> U_mesh = _grid_to_mesh_interpolation(grid, mesh, U_grid)
    """
    # Get mesh vertex positions
    vertices = mesh_geo.get_spatial_grid()  # (N_vertices, dimension)

    # Use grid's built-in interpolator (bilinear/trilinear)
    interpolator = grid_geo.get_interpolator()

    # Evaluate at mesh vertices
    mesh_values = interpolator(grid_values, vertices)

    return mesh_values


# Automatic registration of FEM mesh projections if scipy available
def _register_fem_mesh_projections():
    """
    Automatically register specialized FEM mesh projections.

    This function is called on module import if scipy is available.
    It registers Delaunay-based projections for all UnstructuredMesh types
    (Mesh2D, Mesh3D, TriangularAMRMesh, etc.).

    Registered Projections:
        - UnstructuredMesh → CartesianGrid: Delaunay interpolation
        - CartesianGrid → UnstructuredMesh: Bilinear/trilinear interpolation

    Notes:
        - Uses category-based registration (isinstance checks)
        - Falls back to nearest neighbor if scipy not available
        - Silent failure if scipy missing (uses existing fallbacks)
    """
    try:
        # Only register if scipy available
        import scipy.interpolate
        import scipy.spatial  # noqa: F401

        from mfg_pde.geometry.base import CartesianGrid, UnstructuredMesh

        # Register mesh → grid (Delaunay interpolation)
        ProjectionRegistry.register(UnstructuredMesh, CartesianGrid, "fp_to_hjb")(_mesh_to_grid_delaunay)

        # Register grid → mesh (bilinear/trilinear interpolation)
        ProjectionRegistry.register(CartesianGrid, UnstructuredMesh, "hjb_to_fp")(_grid_to_mesh_interpolation)

    except ImportError:
        # scipy not available - will use fallback methods
        pass


# Register on module import
_register_fem_mesh_projections()
