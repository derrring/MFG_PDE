"""
Unified geometry base classes for MFG_PDE.

This module consolidates GeometryProtocol and BaseGeometry into a single
Geometry ABC that provides both data interface and solver operation interface.

Created: 2025-11-09
Part of: Issue #245 Phase 2 - Geometry Architecture Consolidation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

from .meshes.mesh_data import MeshVisualizationMode
from .protocol import GeometryType


class Geometry(ABC):
    """
    Unified abstract base class for all MFG geometries.

    Provides both data interface and solver operation interface,
    eliminating the need for separate protocol/adapter layers.

    All geometry types inherit from this class:
    - Cartesian grids: TensorProductGrid (all dimensions)
    - Unstructured meshes: Domain2D, Domain3D
    - Networks: NetworkGeometry
    - Adaptive meshes: AMR classes
    - Implicit domains: Level set, SDF

    Design Principles:
    1. Geometry objects are responsible for their own discretization
    2. Solvers request operations (Laplacian, gradient), not raw data
    3. Type system enforces solver-geometry compatibility

    Examples:
        >>> # Type-safe solver initialization
        >>> from mfg_pde.geometry import TensorProductGrid, CartesianGrid
        >>> from mfg_pde.solvers import HJBFDMSolver
        >>>
        >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx_points=[50,50])
        >>> problem = MFGProblem(geometry=grid, T=1.0, Nt=100)
        >>>
        >>> # Solver checks geometry type
        >>> if isinstance(problem.geometry, CartesianGrid):
        ...     solver = HJBFDMSolver(problem)  # OK
        ... else:
        ...     raise TypeError("FDM requires Cartesian grid")
    """

    # ============================================================================
    # Data Interface (what GeometryProtocol had)
    # ============================================================================

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Spatial dimension of the geometry.

        Returns:
            int: Dimension (1, 2, 3, ..., or 0 for networks)

        Examples:
            >>> grid = TensorProductGrid(dimension=2, ...)
            >>> grid.dimension
            2
        """
        ...

    @property
    @abstractmethod
    def geometry_type(self) -> GeometryType:
        """
        Type of geometry (enum).

        Returns:
            GeometryType: CARTESIAN_GRID, NETWORK, DOMAIN_2D, etc.

        Examples:
            >>> grid = TensorProductGrid(...)
            >>> grid.geometry_type
            <GeometryType.CARTESIAN_GRID: 'cartesian_grid'>
        """
        ...

    # ============================================================================
    # Type Helper Properties (eliminates isinstance checks)
    # ============================================================================

    @property
    def is_cartesian(self) -> bool:
        """
        Check if this is a Cartesian grid geometry.

        Returns:
            True if geometry_type is CARTESIAN_GRID

        Examples:
            >>> grid = TensorProductGrid(dimension=2, ...)
            >>> grid.is_cartesian
            True
            >>> network = create_network(NetworkType.RANDOM, 10)
            >>> network.is_cartesian
            False
        """
        return self.geometry_type == GeometryType.CARTESIAN_GRID

    @property
    def is_network(self) -> bool:
        """
        Check if this is a network/graph geometry.

        Returns:
            True if geometry_type is NETWORK

        Examples:
            >>> network = create_network(NetworkType.RANDOM, 10)
            >>> network.is_network
            True
            >>> grid = TensorProductGrid(dimension=2, ...)
            >>> grid.is_network
            False
        """
        return self.geometry_type == GeometryType.NETWORK

    @property
    def is_mesh(self) -> bool:
        """
        Check if this is an unstructured mesh geometry.

        Returns:
            True if geometry_type is DOMAIN_2D or DOMAIN_3D

        Examples:
            >>> mesh = Domain2D(...)
            >>> mesh.is_mesh
            True
        """
        return self.geometry_type in (GeometryType.DOMAIN_2D, GeometryType.DOMAIN_3D)

    @property
    def is_implicit(self) -> bool:
        """
        Check if this is an implicit geometry (SDF-based).

        Returns:
            True if geometry_type is IMPLICIT

        Examples:
            >>> domain = Hyperrectangle(...)
            >>> domain.is_implicit
            True
        """
        return self.geometry_type == GeometryType.IMPLICIT

    @property
    @abstractmethod
    def num_spatial_points(self) -> int:
        """
        Total number of discrete spatial points.

        Returns:
            int: Number of grid points / mesh vertices / graph nodes

        Examples:
            >>> grid = TensorProductGrid(dimension=2, Nx_points=[10, 20])
            >>> grid.num_spatial_points
            200
        """
        ...

    @abstractmethod
    def get_spatial_grid(self) -> NDArray:
        """
        Get spatial grid representation.

        Returns:
            For Cartesian grids: (N, d) array of grid points
            For meshes: (N_vertices, d) array of mesh vertices
            For networks: (N_nodes,) array or adjacency representation

        Examples:
            >>> grid = TensorProductGrid(dimension=2, ...)
            >>> points = grid.get_spatial_grid()
            >>> points.shape
            (200, 2)
        """
        ...

    @abstractmethod
    def get_bounds(self) -> tuple[NDArray, NDArray] | None:
        """
        Get bounding box of geometry.

        Returns:
            (min_coords, max_coords) tuple of arrays, or None if unbounded/not applicable

        Examples:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,2)])
            >>> min_coords, max_coords = grid.get_bounds()
            >>> min_coords
            array([0., 0.])
            >>> max_coords
            array([1., 2.])
        """
        ...

    @abstractmethod
    def get_problem_config(self) -> dict:
        """
        Return configuration dict for MFGProblem initialization.

        Polymorphic method allowing each geometry type to specify how it
        configures MFGProblem, avoiding hasattr checks and duck typing.

        Returns:
            Dictionary with keys (geometry-dependent):
                - num_spatial_points: int
                - spatial_shape: tuple
                - spatial_bounds: list[tuple] or None
                - spatial_discretization: list[int] or None
                - Additional geometry-specific data

        Examples:
            >>> grid = TensorProductGrid(...)
            >>> config = grid.get_problem_config()
            >>> config['num_spatial_points']
            200
        """
        ...

    # ============================================================================
    # Solver Operation Interface (NEW - eliminates hasattr patterns)
    # ============================================================================

    @abstractmethod
    def get_laplacian_operator(self) -> Callable:
        """
        Return discretized Laplacian operator for this geometry.

        Returns:
            Function with signature: (u: NDArray, point_idx) -> float

        Examples:
            >>> grid = TensorProductGrid(dimension=2, ...)
            >>> laplacian = grid.get_laplacian_operator()
            >>> u = np.random.rand(10, 20)
            >>> lap_value = laplacian(u, (5, 10))  # Laplacian at grid point (5,10)

        Implementation notes:
            - Cartesian grids: Finite difference Laplacian
            - Unstructured meshes: Finite element Laplacian (mass matrix)
            - Networks: Graph Laplacian
        """
        ...

    @abstractmethod
    def get_gradient_operator(self) -> Callable:
        """
        Return discretized gradient operator for this geometry.

        Returns:
            Function with signature: (u: NDArray, point_idx) -> NDArray

        Examples:
            >>> grid = TensorProductGrid(dimension=2, ...)
            >>> gradient = grid.get_gradient_operator()
            >>> u = np.random.rand(10, 20)
            >>> grad_u = gradient(u, (5, 10))  # Returns [du/dx, du/dy]
            >>> grad_u.shape
            (2,)

        Implementation notes:
            - Cartesian grids: Finite difference gradient
            - Unstructured meshes: Finite element gradient
            - Networks: Discrete gradient along edges
        """
        ...

    @abstractmethod
    def get_interpolator(self) -> Callable:
        """
        Return interpolation function for this geometry.

        Returns:
            Function with signature: (u: NDArray, point: NDArray) -> float

        Examples:
            >>> grid = TensorProductGrid(dimension=2, ...)
            >>> interpolate = grid.get_interpolator()
            >>> u = np.random.rand(10, 20)
            >>> value = interpolate(u, np.array([0.5, 0.3]))  # Interpolate at (0.5, 0.3)

        Implementation notes:
            - Cartesian grids: Linear/bilinear/trilinear interpolation
            - Unstructured meshes: Barycentric interpolation
            - Networks: Nearest node or graph-based interpolation
        """
        ...

    @abstractmethod
    def get_boundary_handler(self) -> Any:
        """
        Return boundary condition handler for this geometry.

        Returns:
            BoundaryHandler: Object that applies boundary conditions

        Examples:
            >>> grid = TensorProductGrid(...)
            >>> bc_handler = grid.get_boundary_handler()
            >>> u_bc = bc_handler.apply(u, bc_type="periodic")

        Implementation notes:
            - Handles Dirichlet, Neumann, periodic, Robin BCs
            - Geometry-specific boundary detection and application
        """
        ...

    # ============================================================================
    # Grid/Mesh Utilities (for geometries that need them)
    # ============================================================================

    def get_grid_spacing(self) -> list[float] | None:
        """
        Get grid spacing for regular Cartesian grids.

        Returns:
            [dx1, dx2, ...] or None if not a regular grid

        Default implementation returns None (override in Cartesian grid classes).

        Examples:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,2)], Nx_points=[10,20])
            >>> dx = grid.get_grid_spacing()
            >>> dx
            [0.111..., 0.105...]
        """
        return None

    def get_grid_shape(self) -> tuple[int, ...] | None:
        """
        Get grid shape for regular Cartesian grids.

        Returns:
            (Nx, Ny, ...) or None if not a regular grid

        Default implementation returns None (override in Cartesian grid classes).

        Examples:
            >>> grid = TensorProductGrid(dimension=2, Nx_points=[10, 20])
            >>> shape = grid.get_grid_shape()
            >>> shape
            (10, 20)
        """
        return None

    # ============================================================================
    # Boundary Methods (mandatory - every domain has boundary)
    # ============================================================================

    def is_on_boundary(
        self,
        points: NDArray,
        tolerance: float = 1e-10,
    ) -> NDArray:
        """
        Check if points are on the domain boundary.

        Args:
            points: Array of shape (n, d) - points to check
            tolerance: Distance tolerance for boundary detection

        Returns:
            Boolean array of shape (n,) - True if point is on boundary

        Default implementation uses bounds-based detection.
        Override for more accurate geometry-specific detection.
        """
        points = np.atleast_2d(points)
        bounds_result = self.get_bounds()
        if bounds_result is None:
            # Unbounded domain - no boundary
            return np.zeros(len(points), dtype=bool)

        min_coords, max_coords = bounds_result

        # Vectorized boundary check: any dimension at min or max boundary
        near_min = np.abs(points - min_coords) < tolerance  # shape: (n, d)
        near_max = np.abs(points - max_coords) < tolerance  # shape: (n, d)
        on_boundary = np.any(near_min | near_max, axis=1)  # shape: (n,)

        return on_boundary

    def get_boundary_normal(
        self,
        points: NDArray,
    ) -> NDArray:
        """
        Get outward normal vectors at boundary points.

        Args:
            points: Array of shape (n, d) - boundary points

        Returns:
            Array of shape (n, d) - unit outward normal at each point

        Default implementation assumes axis-aligned boundaries.
        Override for curved or complex boundaries.
        """
        points = np.atleast_2d(points)
        normals = np.zeros_like(points)
        bounds_result = self.get_bounds()

        if bounds_result is None:
            return normals

        min_coords, max_coords = bounds_result
        tolerance = 1e-10

        # Vectorized boundary detection
        near_min = np.abs(points - min_coords) < tolerance  # shape: (n, d)
        near_max = np.abs(points - max_coords) < tolerance  # shape: (n, d)

        # For each point, find first dimension at boundary (prioritize min over max)
        # Use argmax to get first True along dimension axis
        min_boundary_dim = np.argmax(near_min, axis=1)  # First dim at min boundary
        max_boundary_dim = np.argmax(near_max, axis=1)  # First dim at max boundary
        has_min_boundary = np.any(near_min, axis=1)
        has_max_boundary = np.any(near_max, axis=1)

        # Set normals for points at min boundary
        rows_min = np.where(has_min_boundary)[0]
        if len(rows_min) > 0:
            normals[rows_min, min_boundary_dim[rows_min]] = -1.0

        # Set normals for points at max boundary (only if not already at min)
        rows_max = np.where(has_max_boundary & ~has_min_boundary)[0]
        if len(rows_max) > 0:
            normals[rows_max, max_boundary_dim[rows_max]] = 1.0

        return normals

    def project_to_boundary(
        self,
        points: NDArray,
    ) -> NDArray:
        """
        Project points onto the domain boundary.

        Args:
            points: Array of shape (n, d) - points to project

        Returns:
            Array of shape (n, d) - projected points on boundary

        Default implementation projects to nearest axis-aligned boundary.
        Override for curved or complex boundaries.
        """
        points = np.atleast_2d(points)
        bounds_result = self.get_bounds()

        if bounds_result is None:
            return points.copy()

        min_coords, max_coords = bounds_result
        projected = points.copy()
        n_points = len(points)
        d = self.dimension

        # Compute distances to all 2d boundaries (d min + d max) - shape: (n, 2d)
        dist_to_min = np.abs(points - min_coords)  # shape: (n, d)
        dist_to_max = np.abs(points - max_coords)  # shape: (n, d)
        all_distances = np.hstack([dist_to_min, dist_to_max])  # shape: (n, 2d)

        # Find nearest boundary for each point
        nearest_idx = np.argmin(all_distances, axis=1)  # shape: (n,)

        # Determine which dimension and which side (min or max)
        is_min_boundary = nearest_idx < d
        dim_idx = np.where(is_min_boundary, nearest_idx, nearest_idx - d)

        # Apply projection: set the relevant coordinate to boundary value
        row_indices = np.arange(n_points)
        projected[row_indices, dim_idx] = np.where(is_min_boundary, min_coords[dim_idx], max_coords[dim_idx])

        return projected

    def project_to_interior(
        self,
        points: NDArray,
    ) -> NDArray:
        """
        Project points outside the domain back into the interior.

        For points already inside, returns them unchanged.
        For points outside, clips to domain bounds.

        Args:
            points: Array of shape (n, d) - points to project

        Returns:
            Array of shape (n, d) - points guaranteed to be inside domain

        Default implementation clips to bounding box.
        Override for non-convex or implicit domains.
        """
        points = np.atleast_2d(points)
        bounds_result = self.get_bounds()

        if bounds_result is None:
            return points.copy()

        min_coords, max_coords = bounds_result
        return np.clip(points, min_coords, max_coords)

    def get_boundary_regions(self) -> dict[str, dict]:
        """
        Get information about distinct boundary regions.

        Returns:
            Dictionary mapping region names to region info.

        Default implementation returns axis-aligned boundary regions.
        Override for complex domain boundaries.
        """
        bounds_result = self.get_bounds()
        if bounds_result is None:
            return {"all": {}}

        min_coords, max_coords = bounds_result
        regions = {}

        axis_names = ["x", "y", "z", "w", "v"]  # Extend as needed
        for d in range(self.dimension):
            axis = axis_names[d] if d < len(axis_names) else f"dim{d}"
            regions[f"{axis}_min"] = {"axis": d, "side": "min", "value": float(min_coords[d])}
            regions[f"{axis}_max"] = {"axis": d, "side": "max", "value": float(max_coords[d])}

        return regions


# ============================================================================
# Intermediate Abstract Base Classes
# ============================================================================


class CartesianGrid(Geometry):
    """
    Abstract base class for regular Cartesian grids.

    Extends Geometry with grid-specific properties that are guaranteed
    to exist for structured grids (TensorProductGrid).

    Grid Properties:
        - Regular spacing in each dimension
        - Tensor product structure
        - Known grid shape and spacing

    Use Cases:
        - Finite difference methods (FDM)
        - WENO schemes
        - Spectral methods
        - Any solver requiring structured grid

    Examples:
        >>> # Type-safe solver check
        >>> if isinstance(geometry, CartesianGrid):
        ...     dx = geometry.get_grid_spacing()  # Guaranteed to work
        ...     shape = geometry.get_grid_shape()  # Guaranteed to work
    """

    # Override to make non-optional for Cartesian grids
    @abstractmethod
    def get_grid_spacing(self) -> list[float]:
        """
        Get grid spacing per dimension.

        Returns:
            [dx1, dx2, ...] where dxi = (xmax_i - xmin_i) / (Ni - 1)

        Note: This is abstract for CartesianGrid (must be implemented by subclasses).
        """
        ...

    @abstractmethod
    def get_grid_shape(self) -> tuple[int, ...]:
        """
        Get number of grid points per dimension.

        Returns:
            (Nx1, Nx2, ...) tuple of grid points

        Note: This is abstract for CartesianGrid (must be implemented by subclasses).
        """
        ...


class UnstructuredMesh(Geometry):
    """
    Abstract base class for unstructured FEM-style meshes.

    Extends Geometry with mesh-specific properties for geometries
    generated via Gmsh → Meshio → PyVista pipeline.

    Mesh Properties:
        - Unstructured vertices and elements
        - Triangular (2D) or tetrahedral (3D) elements
        - Boundary tags for BC application
        - Mesh quality metrics

    Use Cases:
        - Finite element methods (FEM)
        - Complex geometries with obstacles
        - Adaptive mesh refinement
        - Domains with curved boundaries

    Examples:
        >>> if isinstance(geometry, UnstructuredMesh):
        ...     mesh_data = geometry.mesh_data
        ...     vertices = mesh_data.vertices
        ...     elements = mesh_data.elements
    """

    def __init__(self, dimension: int):
        """
        Initialize unstructured mesh.

        Args:
            dimension: Spatial dimension (2 or 3)
        """
        self._dimension = dimension
        self.mesh_data: Any | None = None  # MeshData from meshes.mesh_data
        self._gmsh_model: Any = None

    @property
    def dimension(self) -> int:
        """Spatial dimension (2 or 3 for meshes)."""
        return self._dimension

    @abstractmethod
    def create_gmsh_geometry(self) -> Any:
        """
        Create geometry using Gmsh API.

        Must be implemented by subclasses (Domain2D, Domain3D).
        """
        ...

    @abstractmethod
    def generate_mesh(self) -> Any:
        """
        Generate mesh using Gmsh → Meshio pipeline.

        Returns:
            MeshData: Mesh data structure

        Must be implemented by subclasses.
        """
        ...

    @abstractmethod
    def set_mesh_parameters(self, **kwargs) -> None:
        """
        Set mesh generation parameters.

        Args:
            **kwargs: Mesh parameters (mesh_size, refinement criteria, etc.)

        Must be implemented by subclasses.
        """
        ...

    # ============================================================================
    # Geometry ABC implementation
    # ============================================================================

    @property
    def geometry_type(self) -> GeometryType:
        """
        Type of geometry.

        Returns DOMAIN_2D or DOMAIN_3D based on dimension.
        Subclasses can override for more specific types.
        """
        if self._dimension == 2:
            return GeometryType.DOMAIN_2D
        elif self._dimension == 3:
            return GeometryType.DOMAIN_3D
        else:
            return GeometryType.CUSTOM

    @property
    def num_spatial_points(self) -> int:
        """Total number of discrete spatial points (mesh vertices)."""
        if self.mesh_data is None:
            raise ValueError("Mesh not yet generated. Call generate_mesh() first.")
        return self.mesh_data.num_vertices

    def get_spatial_grid(self) -> NDArray:
        """
        Get spatial grid representation (mesh vertices).

        Returns:
            Numpy array of mesh vertex coordinates (N_vertices, dimension)

        Raises:
            ValueError: If mesh has not been generated yet
        """
        if self.mesh_data is None:
            raise ValueError("Mesh not yet generated. Call generate_mesh() first.")
        return self.mesh_data.vertices

    def get_bounds(self) -> tuple[NDArray, NDArray]:
        """
        Get geometry bounding box from mesh.

        Returns:
            (min_coords, max_coords) tuple

        Raises:
            ValueError: If mesh not yet generated
        """
        if self.mesh_data is None:
            raise ValueError("Mesh not yet generated. Call generate_mesh() first.")
        return self.mesh_data.bounds

    def get_problem_config(self) -> dict:
        """
        Return configuration for MFGProblem.

        Returns:
            Dictionary with mesh configuration
        """
        if self.mesh_data is None:
            raise ValueError("Mesh not yet generated. Call generate_mesh() first.")

        return {
            "num_spatial_points": self.mesh_data.num_vertices,
            "spatial_shape": (self.mesh_data.num_vertices,),  # Unstructured
            "spatial_bounds": list(zip(*self.mesh_data.bounds, strict=True)),
            "spatial_discretization": None,  # No regular discretization
            "legacy_1d_attrs": None,
            "mesh_data": self.mesh_data,
        }

    # ============================================================================
    # Solver Operation Interface (FEM-style for unstructured meshes)
    # ============================================================================

    def get_laplacian_operator(self) -> Callable:
        """
        Return FEM Laplacian operator for unstructured mesh.

        Returns:
            Function with signature: (u: NDArray, vertex_idx: int) -> float

        Note: This is a placeholder implementation. Full FEM Laplacian requires
        assembly of mass and stiffness matrices.
        """

        def laplacian_fem_placeholder(u: NDArray, vertex_idx: int) -> float:
            """
            Placeholder FEM Laplacian (returns 0.0).

            TODO: Implement full FEM assembly with mass/stiffness matrices.

            Args:
                u: Solution vector at mesh vertices
                vertex_idx: Vertex index

            Returns:
                Laplacian value (currently 0.0)
            """
            return 0.0

        return laplacian_fem_placeholder

    def get_gradient_operator(self) -> Callable:
        """
        Return FEM gradient operator for unstructured mesh.

        Returns:
            Function with signature: (u: NDArray, vertex_idx: int) -> NDArray

        Note: This is a placeholder implementation. Full FEM gradient requires
        element-wise gradient reconstruction.
        """

        def gradient_fem_placeholder(u: NDArray, vertex_idx: int) -> NDArray:
            """
            Placeholder FEM gradient (returns zeros).

            TODO: Implement FEM gradient with element-wise reconstruction.

            Args:
                u: Solution vector at mesh vertices
                vertex_idx: Vertex index

            Returns:
                Gradient vector (currently zeros)
            """
            return np.zeros(self._dimension)

        return gradient_fem_placeholder

    def get_interpolator(self) -> Callable:
        """
        Return barycentric interpolator for unstructured mesh.

        Returns:
            Function with signature: (u: NDArray, point: NDArray) -> float

        Note: This is a placeholder. Full implementation requires finding
        containing element and computing barycentric coordinates.
        """

        def interpolate_barycentric_placeholder(u: NDArray, point: NDArray) -> float:
            """
            Placeholder barycentric interpolation (nearest neighbor).

            TODO: Implement proper barycentric interpolation with element search.

            Args:
                u: Solution vector at mesh vertices
                point: Physical coordinates

            Returns:
                Interpolated value (currently nearest neighbor)
            """
            if self.mesh_data is None:
                raise ValueError("Mesh not generated")

            # Simple nearest neighbor for now
            vertices = self.mesh_data.vertices
            distances = np.linalg.norm(vertices - point, axis=1)
            nearest_idx = int(np.argmin(distances))
            return float(u[nearest_idx])

        return interpolate_barycentric_placeholder

    def get_boundary_handler(self):
        """
        Return boundary condition handler.

        Returns:
            Dict with boundary information (placeholder)
        """
        return {"type": "unstructured_mesh", "implementation": "placeholder"}

    # ============================================================================
    # Boundary Methods (FEM-specific overrides)
    # ============================================================================

    def get_boundary_normal(self, points: NDArray) -> NDArray:
        """
        Get outward normal vectors at boundary points for unstructured mesh.

        For FEM meshes, boundary normals are computed from boundary face geometry:
        - 2D: Normal to boundary edges
        - 3D: Normal to boundary triangles

        Args:
            points: Array of shape (n, d) - boundary points

        Returns:
            Array of shape (n, d) - unit outward normal at each point

        Note:
            This is a placeholder implementation that falls back to the base class
            axis-aligned normal computation. Full implementation requires:
            1. Identify boundary faces from mesh topology
            2. Find closest boundary face to each query point
            3. Compute face normal from vertex positions

            TODO: Implement proper mesh-based boundary normal computation
        """
        # Placeholder: fall back to base class axis-aligned implementation
        return super().get_boundary_normal(points)

    def project_to_boundary(self, points: NDArray) -> NDArray:
        """
        Project points onto the mesh boundary.

        For FEM meshes, boundary projection finds the closest point on boundary faces:
        - 2D: Project to nearest boundary edge
        - 3D: Project to nearest boundary triangle

        Args:
            points: Array of shape (n, d) - points to project

        Returns:
            Array of shape (n, d) - projected points on boundary

        Note:
            This is a placeholder implementation that falls back to the base class
            axis-aligned projection. Full implementation requires:
            1. Build boundary face spatial index (e.g., BVH tree)
            2. Find closest boundary face to each query point
            3. Project point onto that face

            TODO: Implement proper mesh-based boundary projection
        """
        # Placeholder: fall back to base class axis-aligned implementation
        return super().project_to_boundary(points)

    def is_on_boundary(self, points: NDArray, tolerance: float = 1e-10) -> NDArray:
        """
        Check if points are on the mesh boundary.

        For FEM meshes, boundary detection checks distance to boundary faces.

        Args:
            points: Array of shape (n, d) - points to check
            tolerance: Distance tolerance for boundary detection

        Returns:
            Boolean array of shape (n,) - True if point is on boundary

        Note:
            This is a placeholder implementation that falls back to the base class
            bounds-based detection. Full implementation requires:
            1. Compute distance to all boundary faces
            2. Return True if min distance < tolerance

            TODO: Implement proper mesh-based boundary detection
        """
        # Placeholder: fall back to base class bounds-based implementation
        return super().is_on_boundary(points, tolerance)

    # ============================================================================
    # Mesh Utilities (from old BaseGeometry)
    # ============================================================================

    @abstractmethod
    def export_mesh(self, file_format: str, filename: str) -> None:
        """
        Export mesh in specified file format.

        Must be implemented by subclasses.
        """
        ...

    def visualize_mesh(
        self,
        mode: MeshVisualizationMode | str = MeshVisualizationMode.WITH_EDGES,
        *,
        show_edges: bool | None = None,
        show_quality: bool | None = None,
    ):
        """
        Visualize mesh using PyVista.

        Parameters
        ----------
        mode : MeshVisualizationMode or str, default=WITH_EDGES
            Visualization mode: SURFACE, WITH_EDGES, QUALITY, or QUALITY_WITH_EDGES
            Can pass strings: "surface", "with_edges", "quality", "quality_with_edges"
        show_edges : bool, optional (deprecated)
            Deprecated: Use mode parameter instead
        show_quality : bool, optional (deprecated)
            Deprecated: Use mode parameter instead

        Examples
        --------
        >>> # New API (recommended)
        >>> geom.visualize_mesh(mode=MeshVisualizationMode.QUALITY_WITH_EDGES)
        >>> geom.visualize_mesh(mode="quality")
        >>>
        >>> # Old API (deprecated but still works)
        >>> geom.visualize_mesh(show_edges=True, show_quality=False)
        """
        if self.mesh_data is None:
            self.generate_mesh()

        try:
            import pyvista as pv
        except ImportError as err:
            raise ImportError("pyvista is required for mesh visualization") from err

        if self.mesh_data is None:
            raise RuntimeError("Mesh data is None")

        # Handle backward compatibility
        if show_edges is not None or show_quality is not None:
            import warnings

            warnings.warn(
                "Parameters 'show_edges' and 'show_quality' are deprecated. "
                "Use 'mode' parameter instead: MeshVisualizationMode.SURFACE, .WITH_EDGES, "
                ".QUALITY, or .QUALITY_WITH_EDGES",
                DeprecationWarning,
                stacklevel=2,
            )
            # Convert old API to new
            edges = show_edges if show_edges is not None else True
            quality = show_quality if show_quality is not None else False
            if quality and edges:
                mode = MeshVisualizationMode.QUALITY_WITH_EDGES
            elif quality:
                mode = MeshVisualizationMode.QUALITY
            elif edges:
                mode = MeshVisualizationMode.WITH_EDGES
            else:
                mode = MeshVisualizationMode.SURFACE

        # Handle string mode
        if isinstance(mode, str):
            mode_map = {
                "surface": MeshVisualizationMode.SURFACE,
                "with_edges": MeshVisualizationMode.WITH_EDGES,
                "quality": MeshVisualizationMode.QUALITY,
                "quality_with_edges": MeshVisualizationMode.QUALITY_WITH_EDGES,
            }
            mode = mode_map.get(mode.lower(), MeshVisualizationMode.WITH_EDGES)

        # Determine display settings from mode
        show_edges_flag = mode in (MeshVisualizationMode.WITH_EDGES, MeshVisualizationMode.QUALITY_WITH_EDGES)
        show_quality_flag = mode in (MeshVisualizationMode.QUALITY, MeshVisualizationMode.QUALITY_WITH_EDGES)

        mesh = self.mesh_data.to_pyvista()
        plotter = pv.Plotter()

        if show_quality_flag:
            # Color by mesh quality if available
            if self.mesh_data.quality_metrics and "quality" in self.mesh_data.quality_metrics:
                mesh.cell_data["quality"] = self.mesh_data.quality_metrics["quality"]
                plotter.add_mesh(mesh, scalars="quality", show_edges=show_edges_flag)
            else:
                plotter.add_mesh(mesh, show_edges=show_edges_flag)
        else:
            plotter.add_mesh(mesh, show_edges=show_edges_flag)

        plotter.show()

    def compute_mesh_quality(self) -> dict[str, float]:
        """Compute mesh quality metrics."""
        if self.mesh_data is None:
            self.generate_mesh()

        if self.mesh_data is None:
            raise RuntimeError("Mesh data is None")

        quality_metrics = {}

        if self.mesh_data.element_type == "triangle":
            quality_metrics.update(self._compute_triangle_quality())
        elif self.mesh_data.element_type == "tetrahedron":
            quality_metrics.update(self._compute_tetrahedron_quality())

        if self.mesh_data.quality_metrics is not None:
            self.mesh_data.quality_metrics.update(quality_metrics)
        return quality_metrics

    def _compute_triangle_quality(self) -> dict[str, float]:
        """Compute quality metrics for triangular elements."""
        if self.mesh_data is None:
            raise RuntimeError("Mesh data is None")
        areas = self.mesh_data.compute_element_volumes()

        # Compute aspect ratios
        aspect_ratios = []
        for _i, element in enumerate(self.mesh_data.elements):
            v0, v1, v2 = self.mesh_data.vertices[element]

            # Edge lengths
            e1 = np.linalg.norm(v1 - v0)
            e2 = np.linalg.norm(v2 - v1)
            e3 = np.linalg.norm(v0 - v2)

            # Aspect ratio = (longest edge) / (shortest edge)
            aspect_ratios.append(max(float(e1), float(e2), float(e3)) / min(float(e1), float(e2), float(e3)))

        return {
            "min_area": float(np.min(areas)),
            "max_area": float(np.max(areas)),
            "mean_area": float(np.mean(areas)),
            "min_aspect_ratio": float(np.min(aspect_ratios)),
            "max_aspect_ratio": float(np.max(aspect_ratios)),
            "mean_aspect_ratio": float(np.mean(aspect_ratios)),
        }

    def _compute_tetrahedron_quality(self) -> dict[str, float]:
        """Compute quality metrics for tetrahedral elements."""
        if self.mesh_data is None:
            raise RuntimeError("Mesh data is None")
        volumes = self.mesh_data.compute_element_volumes()

        # Basic volume statistics
        return {
            "min_volume": float(np.min(volumes)),
            "max_volume": float(np.max(volumes)),
            "mean_volume": float(np.mean(volumes)),
        }


class ImplicitGeometry(Geometry):
    """
    Abstract base class for implicit geometries via signed distance functions (SDFs).

    Implicit geometries represent domains D ⊂ ℝ^d by a function φ: ℝ^d → ℝ where:
        φ(x) < 0  ⟺  x ∈ D    (interior)
        φ(x) = 0  ⟺  x ∈ ∂D   (boundary)
        φ(x) > 0  ⟺  x ∉ D    (exterior)

    Advantages:
        - Memory: O(d) parameters instead of O(N^d) mesh vertices
        - Dimension-agnostic: Same code works for 2D, 3D, ..., 100D
        - Obstacles: Free via CSG operations (union, intersection, difference)
        - Particle-friendly: Natural sampling and boundary handling

    Solver Methods:
        - Meshfree collocation methods (RBF, SPH)
        - Particle methods (Lagrangian)
        - Level set methods

    Use Cases:
        - High-dimensional MFG (d > 3)
        - Problems with complex obstacles
        - Particle-based simulations
        - Moving boundary problems

    Subclasses must implement:
        - signed_distance(x): Core SDF evaluation
        - get_bounding_box(): For rejection sampling

    Examples:
        >>> from mfg_pde.geometry.implicit import Hyperrectangle
        >>> domain = Hyperrectangle(bounds=np.array([[0, 1], [0, 1]]))
        >>> domain.contains(np.array([0.5, 0.5]))  # True
        >>> particles = domain.sample_uniform(1000)
    """

    @abstractmethod
    def signed_distance(self, x: NDArray) -> float | NDArray:
        """
        Compute signed distance function φ(x).

        Convention: φ(x) < 0 inside, φ(x) = 0 on boundary, φ(x) > 0 outside

        Args:
            x: Point(s) to evaluate - shape (d,) or (N, d)

        Returns:
            Signed distance(s) - scalar or array of shape (N,)
        """

    @abstractmethod
    def get_bounding_box(self) -> NDArray:
        """
        Get axis-aligned bounding box.

        Returns:
            bounds: Array of shape (d, 2) where bounds[i] = [min_i, max_i]
        """

    def contains(self, x: NDArray, tol: float = 0.0) -> bool | NDArray:
        """Check if point(s) are inside domain."""
        return self.signed_distance(x) <= tol

    def sample_uniform(self, n_samples: int, max_attempts: int = 100, seed: int | None = None) -> NDArray:
        """
        Sample particles uniformly inside domain via rejection sampling.

        Args:
            n_samples: Number of particles to sample
            max_attempts: Maximum rejection attempts per particle
            seed: Random seed for reproducibility

        Returns:
            particles: Array of shape (n_samples, d)
        """
        if seed is not None:
            np.random.seed(seed)

        bounds = self.get_bounding_box()
        dim = self.dimension

        particles = []
        attempts = 0
        max_total_attempts = n_samples * max_attempts

        while len(particles) < n_samples and attempts < max_total_attempts:
            batch_size = min(n_samples - len(particles), 1000)
            candidates = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(batch_size, dim))

            inside = self.contains(candidates)
            if np.isscalar(inside):
                inside = np.array([inside])

            valid = candidates[inside]
            particles.append(valid)
            attempts += batch_size

        if len(particles) == 0 or sum(len(p) for p in particles) < n_samples:
            raise RuntimeError(
                f"Rejection sampling failed: only found {sum(len(p) for p in particles)}/{n_samples} "
                f"valid particles after {attempts} attempts"
            )

        all_particles = np.vstack(particles)
        return all_particles[:n_samples]

    # ============================================================================
    # Geometry ABC implementation
    # ============================================================================

    def get_bounds(self) -> tuple[NDArray, NDArray]:
        """Return bounding box as (min_coords, max_coords)."""
        bounds = self.get_bounding_box()
        return bounds[:, 0], bounds[:, 1]

    def get_problem_config(self) -> dict:
        """Return configuration dict for MFGProblem initialization."""
        n_points = self.num_spatial_points
        bounds = self.get_bounding_box()

        return {
            "num_spatial_points": n_points,
            "spatial_shape": (n_points,),  # Flattened for implicit
            "spatial_bounds": [tuple(b) for b in bounds],
            "spatial_discretization": None,  # No regular grid
            "implicit_domain": True,
        }

    # ============================================================================
    # Solver Operation Interface (Meshfree methods)
    # ============================================================================

    def get_laplacian_operator(self) -> Callable:
        """
        Return Laplacian operator for meshfree methods.

        For implicit geometries, this uses finite differences on sampled
        collocation points or radial basis function (RBF) approximations.

        Returns:
            Function with signature: (u: NDArray, idx: tuple[int, ...]) -> float

        Note: This is a placeholder. Full RBF implementation requires
        assembling differentiation matrices.
        """

        def laplacian_meshfree(u: NDArray, idx: tuple[int, ...]) -> float:
            """
            Placeholder Laplacian for meshfree methods.

            TODO: Implement proper RBF-based Laplacian
            """
            raise NotImplementedError(
                "Meshfree Laplacian requires RBF differentiation matrices. Use particle-collocation solvers instead."
            )

        return laplacian_meshfree

    def get_gradient_operator(self) -> Callable:
        """
        Return gradient operator for meshfree methods.

        Returns:
            Function with signature: (u: NDArray, idx: tuple[int, ...]) -> NDArray
        """

        def gradient_meshfree(u: NDArray, idx: tuple[int, ...]) -> NDArray:
            """
            Placeholder gradient for meshfree methods.

            TODO: Implement proper RBF-based gradient
            """
            raise NotImplementedError(
                "Meshfree gradient requires RBF differentiation matrices. Use particle-collocation solvers instead."
            )

        return gradient_meshfree

    def get_interpolator(self) -> Callable:
        """
        Return RBF interpolator for meshfree methods.

        Returns:
            Function with signature: (u: NDArray, point: NDArray) -> float
        """

        def interpolate_rbf(u: NDArray, point: NDArray) -> float:
            """
            Placeholder RBF interpolation.

            TODO: Implement proper RBF interpolation
            """
            raise NotImplementedError("RBF interpolation not yet implemented. Use nearest-neighbor as fallback.")

        return interpolate_rbf

    def get_boundary_handler(self):
        """
        Return SDF-based boundary condition handler.

        For implicit domains, boundary handling uses the signed distance
        function to project particles back into the domain.
        """
        return {"type": "sdf_projection", "sdf": self.signed_distance}


class GraphGeometry(Geometry):
    """
    Abstract base class for graph geometries (networks and mazes).

    Unified interface for both abstract networks and spatially-embedded graphs
    (mazes, grid graphs). Distinction between discrete/continuous state spaces
    should be handled by solvers, not geometry.

    Graph Properties:
        - Node connectivity via adjacency matrix
        - Graph Laplacian for diffusion: L = D - A
        - Edge weights for transport costs
        - Optional spatial embedding (x,y coordinates)
        - Optional grid structure (for mazes)

    Use Cases:
        - MFG on abstract networks (social, transportation)
        - MFG on spatial graphs (mazes, grid worlds)
        - Discrete navigation problems
        - Continuous diffusion on graphs

    Design Philosophy:
        - Mazes ARE graphs (grid graphs with some edges removed)
        - Networks ARE graphs (abstract or spatially embedded)
        - Solver decides discrete vs continuous treatment
        - Geometry provides graph structure + optional spatial info

    Examples:
        >>> # Abstract network (no spatial embedding)
        >>> network = SomeNetworkGraph(...)
        >>> adj = network.get_adjacency_matrix()
        >>> pos = network.get_node_positions()  # Returns None
        >>>
        >>> # Maze (spatially embedded grid graph)
        >>> maze = SomeMazeGraph(...)
        >>> adj = maze.get_adjacency_matrix()
        >>> pos = maze.get_node_positions()  # Returns (N, 2) array
        >>> grid = maze.get_maze_array()  # Returns 2D grid array
    """

    @property
    def dimension(self) -> int:
        """
        Dimension for graphs (convention: 0).

        Returns:
            0 (graphs don't have Euclidean dimension as primary structure)

        Note: Spatially-embedded graphs may have node positions in ℝ^d,
        but graph structure is topological (dimension 0).
        """
        return 0

    @property
    def geometry_type(self) -> GeometryType:
        """Type of geometry (always NETWORK for all graph types)."""
        return GeometryType.NETWORK

    # ============================================================================
    # GeometryProtocol implementation
    # ============================================================================

    @property
    @abstractmethod
    def num_spatial_points(self) -> int:
        """Total number of nodes in the graph."""
        ...

    @abstractmethod
    def get_spatial_grid(self) -> NDArray:
        """
        Get node representation.

        Returns:
            For spatially-embedded graphs: (N, d) array of node positions
            For abstract graphs: (N,) array of node indices

        Note: For abstract graphs without spatial embedding, return node indices.
        """
        ...

    def get_bounds(self) -> tuple[NDArray, NDArray]:
        """
        Get bounding box for spatially-embedded graphs.

        Returns:
            (min_coords, max_coords) tuple

        Raises:
            NotImplementedError: If graph has no spatial embedding
        """
        positions = self.get_node_positions()
        if positions is None:
            raise NotImplementedError("Abstract graphs have no spatial bounds")

        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        return min_coords, max_coords

    def get_problem_config(self) -> dict:
        """
        Return configuration for MFGProblem.

        Returns:
            Dictionary with graph configuration
        """
        config = {
            "num_spatial_points": self.num_spatial_points,
            "spatial_shape": (self.num_spatial_points,),
            "spatial_discretization": None,
            "legacy_1d_attrs": None,
            "graph_data": {
                "adjacency_matrix": self.get_adjacency_matrix(),
                "laplacian_matrix": self.get_graph_laplacian(),
            },
        }

        # Add spatial bounds if embedded
        positions = self.get_node_positions()
        if positions is not None:
            min_coords, max_coords = self.get_bounds()
            config["spatial_bounds"] = list(zip(min_coords, max_coords, strict=True))
            config["graph_data"]["node_positions"] = positions
        else:
            config["spatial_bounds"] = None

        # Add maze array if grid-based
        maze_array = self.get_maze_array()
        if maze_array is not None:
            config["graph_data"]["maze_array"] = maze_array

        return config

    # ============================================================================
    # Graph-specific abstract methods
    # ============================================================================

    @abstractmethod
    def get_adjacency_matrix(self) -> NDArray:
        """
        Get adjacency matrix for the graph.

        Returns:
            Adjacency matrix A of shape (N, N) where:
                A[i,j] = weight of edge from node i to node j
                A[i,j] = 0 if no edge exists

        Note: For unweighted graphs, use A[i,j] = 1 for edges.
        """
        ...

    # ============================================================================
    # Optional methods for different graph types
    # ============================================================================

    def get_node_positions(self) -> NDArray | None:
        """
        Get physical coordinates for spatially-embedded graphs.

        Returns:
            (N, d) array of node positions in ℝ^d, or None if abstract graph

        Note: Override this for spatially-embedded graphs (mazes, road networks).
        Abstract graphs (social networks, etc.) return None.
        """
        return None

    def get_maze_array(self) -> NDArray | None:
        """
        Get 2D grid array for grid-based graphs (mazes).

        Returns:
            2D array where:
                0 = wall (no node)
                1 = free space (node exists)
            Or None if not a grid-based graph

        Note: Override this for maze-type graphs that have grid structure.
        """
        return None

    # ============================================================================
    # Graph utilities (concrete implementations)
    # ============================================================================

    def get_graph_laplacian(self, normalized: bool = False) -> NDArray:
        """
        Compute graph Laplacian matrix.

        Args:
            normalized: If True, compute normalized Laplacian L_norm = D^(-1/2) L D^(-1/2)

        Returns:
            Laplacian matrix L = D - A of shape (N, N)

        Note: Graph Laplacian is used for diffusion on graphs:
            ∂m/∂t = -L m (diffusion equation on graph)
        """
        A = self.get_adjacency_matrix()
        D = np.diag(np.sum(A, axis=1))  # Degree matrix

        if normalized:
            # Normalized Laplacian: L_norm = I - D^(-1/2) A D^(-1/2)
            D_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
            L_norm = np.eye(len(D)) - D_sqrt_inv @ A @ D_sqrt_inv
            return L_norm
        else:
            # Standard Laplacian: L = D - A
            return D - A

    def get_neighbors(self, node_idx: int) -> list[int]:
        """
        Get neighbor indices for a node.

        Args:
            node_idx: Node index

        Returns:
            List of neighbor node indices
        """
        A = self.get_adjacency_matrix()
        return [int(j) for j in range(len(A)) if A[node_idx, j] > 0]

    # ============================================================================
    # Solver Operation Interface (graph-specific)
    # ============================================================================

    def get_laplacian_operator(self) -> Callable:
        """
        Return graph Laplacian operator for discrete diffusion.

        Returns:
            Function with signature: (u: NDArray, node_idx: int) -> float
            Computing: (L u)[node_idx] = sum_j L[node_idx, j] * u[j]
        """
        L = self.get_graph_laplacian()

        def graph_laplacian_op(u: NDArray, node_idx: int) -> float:
            """
            Apply graph Laplacian to node.

            Args:
                u: Solution vector at nodes (N,)
                node_idx: Node index

            Returns:
                Laplacian value: (L u)[node_idx]
            """
            return float(np.dot(L[node_idx, :], u))

        return graph_laplacian_op

    def get_gradient_operator(self) -> Callable:
        """
        Return discrete gradient operator for graphs.

        Returns:
            Function with signature: (u: NDArray, node_idx: int) -> NDArray
            Computing: ∇u[node_idx] ≈ differences to neighbors
        """
        A = self.get_adjacency_matrix()

        def graph_gradient_op(u: NDArray, node_idx: int) -> NDArray:
            """
            Compute discrete gradient on graph (differences to neighbors).

            Args:
                u: Solution vector at nodes (N,)
                node_idx: Node index

            Returns:
                Gradient as 1D array with differences to each neighbor
            """
            neighbors = [j for j in range(len(A)) if A[node_idx, j] > 0]
            if not neighbors:
                return np.array([0.0])

            # Compute differences to neighbors
            diffs = np.array([u[j] - u[node_idx] for j in neighbors])
            return diffs

        return graph_gradient_op

    def get_interpolator(self) -> Callable:
        """
        Return interpolator for graphs (nearest neighbor or barycentric).

        Returns:
            Function with signature: (u: NDArray, point: NDArray) -> float
        """
        positions = self.get_node_positions()

        if positions is not None:
            # Spatially-embedded graph: use nearest neighbor in physical space
            def interpolate_nearest_neighbor(u: NDArray, point: NDArray) -> float:
                """
                Nearest-neighbor interpolation for spatially-embedded graph.

                Args:
                    u: Solution vector at nodes (N,)
                    point: Physical coordinates

                Returns:
                    Value at nearest node
                """
                distances = np.linalg.norm(positions - point, axis=1)
                nearest_idx = int(np.argmin(distances))
                return float(u[nearest_idx])

            return interpolate_nearest_neighbor
        else:
            # Abstract graph: interpret point as node index
            def interpolate_node_index(u: NDArray, point: NDArray) -> float:
                """
                Interpolation for abstract graph (point = node index).

                Args:
                    u: Solution vector at nodes (N,)
                    point: Node index as array [idx]

                Returns:
                    Value at node
                """
                node_idx = int(point[0])
                return float(u[node_idx])

            return interpolate_node_index

    def get_boundary_handler(self):
        """
        Return boundary condition handler for graphs.

        Returns:
            Dict with boundary information

        Note: For graphs, "boundary" typically means boundary nodes
        (nodes with fewer connections, or explicitly marked boundary nodes).
        """
        return {"type": "graph", "implementation": "none"}
