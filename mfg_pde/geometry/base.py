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

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

from .geometry_protocol import GeometryType


class Geometry(ABC):
    """
    Unified abstract base class for all MFG geometries.

    Provides both data interface and solver operation interface,
    eliminating the need for separate protocol/adapter layers.

    All geometry types inherit from this class:
    - Cartesian grids: Domain1D, TensorProductGrid, SimpleGrid2D/3D
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
        >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], num_points=[50,50])
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

    @property
    @abstractmethod
    def num_spatial_points(self) -> int:
        """
        Total number of discrete spatial points.

        Returns:
            int: Number of grid points / mesh vertices / graph nodes

        Examples:
            >>> grid = TensorProductGrid(dimension=2, num_points=[10, 20])
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
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,2)], num_points=[10,20])
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
            >>> grid = TensorProductGrid(dimension=2, num_points=[10, 20])
            >>> shape = grid.get_grid_shape()
            >>> shape
            (10, 20)
        """
        return None


# ============================================================================
# Intermediate Abstract Base Classes
# ============================================================================


class CartesianGrid(Geometry):
    """
    Abstract base class for regular Cartesian grids.

    Extends Geometry with grid-specific properties that are guaranteed
    to exist for structured grids (TensorProductGrid, Domain1D, SimpleGrid).

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
        self.mesh_data: Any | None = None  # MeshData from base_geometry
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


class NetworkGraph(Geometry):
    """
    Abstract base class for network/graph geometries.

    Extends Geometry with graph-specific properties for MFG problems
    on discrete network topologies.

    Graph Properties:
        - Node connectivity via adjacency matrix
        - Graph Laplacian for diffusion
        - Edge weights for transport costs
        - Network topology (grid, scale-free, small-world, etc.)

    Use Cases:
        - MFG on networks
        - Traffic flow on road networks
        - Opinion dynamics on social networks
        - Multi-agent systems on graphs

    Examples:
        >>> if isinstance(geometry, NetworkGraph):
        ...     adj_matrix = geometry.network_data.adjacency_matrix
        ...     laplacian = geometry.network_data.laplacian_matrix
    """

    @property
    def dimension(self) -> int:
        """
        Dimension for networks (convention: 0).

        Returns:
            0 (networks don't have Euclidean dimension)
        """
        return 0

    @property
    def geometry_type(self) -> GeometryType:
        """Type of geometry (always NETWORK for graphs)."""
        return GeometryType.NETWORK
