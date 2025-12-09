"""
Geometry package for MFG_PDE: Professional mesh generation and complex domain support.

This package implements comprehensive geometry management for MFG problems:
- Cartesian grids (TensorProductGrid): Regular finite difference grids for any dimension
- Unstructured meshes (Mesh2D/3D): FEM/FVM triangular/tetrahedral meshes via Gmsh
- Implicit domains: High-dimensional meshfree domains with signed distance functions
- Gmsh → Meshio → PyVista pipeline for professional mesh generation
- Advanced boundary condition management for complex domains

Key Components:
- TensorProductGrid: Unified regular Cartesian grids for finite difference methods (1D-nD)
- Mesh2D/3D: Unstructured meshes for FEM/FVM (d≤3)
- implicit: Meshfree geometry infrastructure for any dimension
  - Hyperrectangle: Axis-aligned boxes (O(d) sampling, no rejection!)
  - Hypersphere: Balls/circles for obstacles
  - CSG operations: Union, Intersection, Difference for complex domains
- Geometry: Unified ABC for all geometry types (in base.py)
- MeshData: Universal mesh data container
- MeshPipeline: Complete Gmsh → Meshio → PyVista workflow orchestration
- MeshManager: High-level mesh management for multiple geometries
- BoundaryManager: Advanced boundary condition management

Discretization Methods:
- Use TensorProductGrid for finite difference solvers (all dimensions)
- Use Mesh* (Gmsh) for FEM/FVM problems (d≤3)
- Use implicit.* (SDF) for high-dimensional particle-collocation (d≥4)
"""

from __future__ import annotations

# AMR imports (from subdirectory - canonical locations)
from .amr.amr_1d import Interval1D, OneDimensionalAMRMesh, OneDimensionalErrorEstimator, create_1d_amr_mesh
from .amr.amr_triangular_2d import (
    TriangleElement,
    TriangularAMRMesh,
    TriangularMeshErrorEstimator,
    create_triangular_amr_mesh,
)

# Boundary conditions from subdirectories
# Factory functions for uniform BCs
# FEM boundary condition classes (2D/3D)
from .boundary import (
    BCSegment,
    BCType,
    # 2D aliases
    BoundaryCondition2D,
    # 3D aliases
    BoundaryCondition3D,
    # Base FEM class
    BoundaryConditionFEM,
    BoundaryConditionManager2D,
    BoundaryConditionManager3D,
    BoundaryConditions,
    BoundaryManager,
    DirichletBC2D,
    DirichletBC3D,
    GeometricBoundaryCondition,
    MFGBoundaryHandler2D,
    MFGBoundaryHandler3D,
    MixedBoundaryConditions,
    NeumannBC2D,
    NeumannBC3D,
    PeriodicBC2D,
    PeriodicBC3D,
    RobinBC2D,
    RobinBC3D,
    create_box_boundary_conditions,
    create_circle_boundary_conditions,
    create_rectangle_boundary_conditions,
    create_sphere_boundary_conditions,
    create_standard_boundary_names,
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    periodic_bc,
    robin_bc,
)

# Graph-based geometry (networks + mazes)
from .graph import (
    BaseNetworkGeometry,
    GridNetwork,
    HybridMazeGenerator,
    MazeAlgorithm,
    MazeConfig,
    MazeGeometry,
    NetworkData,
    NetworkType,
    RandomNetwork,
    ScaleFreeNetwork,
    VoronoiMazeGenerator,
    maze_Algorithm,
    maze_CellularAutomataConfig,
    maze_CellularAutomataGenerator,
    maze_Config,
    maze_Geometry,
    maze_HybridGenerator,
    maze_RecursiveDivisionConfig,
    maze_RecursiveDivisionGenerator,
    maze_VoronoiGenerator,
)

# Network backend (from graph subdirectory - canonical location)
from .graph.network_backend import NetworkBackendType, OperationType, get_backend_manager, set_preferred_backend

# Legacy network imports (from old file names - now in graph subdirectory)
from .graph.network_geometry import compute_network_statistics, create_network

# Grid geometry - Import from subdirectories (canonical locations)
from .grids.tensor_grid import TensorProductGrid

# Implicit geometry
from .implicit import (
    ComplementDomain,
    DifferenceDomain,
    Hyperrectangle,
    Hypersphere,
    ImplicitDomain,
    IntersectionDomain,
    PointCloudGeometry,
    UnionDomain,
)

# Mask generation utilities
from .masks import (
    boundary_segment_mask,
    circle_mask,
    combine_masks,
    create_mask,
    get_boundary_mask,
    indices_to_mask,
    invert_mask,
    load_mask,
    mask_to_indices,
    polygon_mask,
    rectangle_mask,
    save_mask,
)

# Mesh data structures (from meshes subdirectory)
# Mesh geometry
from .meshes import Mesh1D, Mesh2D, Mesh3D, MeshData, MeshManager, MeshPipeline, MeshVisualizationMode

# Geometric operators
from .operators import GeometryProjector

# Legacy projection imports (from old file names - now in operators subdirectory)
from .operators.projection import ProjectionRegistry

# Unified geometry protocol
from .protocol import (
    # Boundary-aware protocol (for unified BC handling)
    BoundaryAwareProtocol,
    BoundaryType,
    # Core geometry protocol
    GeometryProtocol,
    GeometryType,
    detect_geometry_type,
    is_boundary_aware,
    is_geometry_compatible,
    validate_boundary_aware,
    validate_geometry,
)

# Legacy grid imports (from old file names)

__all__ = [
    # Multi-dimensional geometry components
    "BaseNetworkGeometry",
    # Boundary condition components
    "BCSegment",
    "BCType",
    "BoundaryCondition2D",
    "BoundaryCondition3D",
    "BoundaryConditionFEM",
    "BoundaryConditionManager2D",
    "BoundaryConditionManager3D",
    "BoundaryConditions",
    "BoundaryManager",
    # Specific boundary condition types
    "DirichletBC2D",
    "DirichletBC3D",
    # Geometry components
    "TensorProductGrid",  # Unified Cartesian grid for all dimensions
    "Mesh1D",
    "Mesh2D",
    "Mesh3D",
    "GeometricBoundaryCondition",
    # Unified geometry protocol
    "GeometryProtocol",
    "GeometryType",
    # Boundary-aware protocol
    "BoundaryAwareProtocol",
    "BoundaryType",
    "is_boundary_aware",
    "validate_boundary_aware",
    # Geometry projection (Issue #257)
    "GeometryProjector",
    "ProjectionRegistry",
    "GridNetwork",
    "Interval1D",
    "MFGBoundaryHandler2D",
    "MFGBoundaryHandler3D",
    "MeshData",
    "MeshManager",
    "MeshPipeline",
    "MeshVisualizationMode",
    "MixedBoundaryConditions",
    # Network backend components
    "NetworkBackendType",
    # Network geometry components
    "NetworkData",
    "NetworkType",
    "NeumannBC2D",
    "NeumannBC3D",
    # AMR components (legacy names from old file structure)
    "OneDimensionalAMRMesh",
    "OneDimensionalErrorEstimator",
    "OperationType",
    "PeriodicBC2D",
    "PeriodicBC3D",
    "PointCloudGeometry",
    "RandomNetwork",
    "RobinBC2D",
    "RobinBC3D",
    "ScaleFreeNetwork",
    "TriangleElement",
    # Legacy triangular AMR components
    "TriangularAMRMesh",
    "TriangularMeshErrorEstimator",
    # Implicit geometry (CSG operations)
    "ComplementDomain",
    "DifferenceDomain",
    "Hyperrectangle",
    "Hypersphere",
    "ImplicitDomain",
    "IntersectionDomain",
    "UnionDomain",
    # Factory and utility functions
    "compute_network_statistics",
    "create_1d_amr_mesh",
    # Boundary condition factory functions
    "create_box_boundary_conditions",
    "create_circle_boundary_conditions",
    "create_network",
    "create_rectangle_boundary_conditions",
    "create_sphere_boundary_conditions",
    "create_standard_boundary_names",
    "create_triangular_amr_mesh",
    "detect_geometry_type",
    "dirichlet_bc",
    "get_backend_manager",
    "is_geometry_compatible",
    # Maze geometry (primary names)
    "MazeGeometry",
    "MazeAlgorithm",
    "MazeConfig",
    "HybridMazeGenerator",
    "VoronoiMazeGenerator",
    # Maze with maze_ prefix
    "maze_Geometry",
    "maze_Algorithm",
    "maze_CellularAutomataConfig",
    "maze_CellularAutomataGenerator",
    "maze_Config",
    "maze_HybridGenerator",
    "maze_RecursiveDivisionConfig",
    "maze_RecursiveDivisionGenerator",
    "maze_VoronoiGenerator",
    "neumann_bc",
    "no_flux_bc",
    "periodic_bc",
    "robin_bc",
    "set_preferred_backend",
    "validate_geometry",
    # Mask generation utilities
    "boundary_segment_mask",
    "circle_mask",
    "combine_masks",
    "create_mask",
    "get_boundary_mask",
    "indices_to_mask",
    "invert_mask",
    "load_mask",
    "mask_to_indices",
    "polygon_mask",
    "rectangle_mask",
    "save_mask",
]

# Version information
__version__ = "1.0.0"


# Optional dependency checks with helpful error messages
def _check_optional_dependencies():
    """Check for optional dependencies and provide helpful messages."""
    dependencies = {
        "gmsh": "pip install gmsh",
        "meshio": "pip install meshio",
        "pyvista": "pip install pyvista",
    }

    missing = []
    for dep, install_cmd in dependencies.items():
        try:
            __import__(dep)
        except ImportError:
            missing.append(f"{dep} ({install_cmd})")

    if missing:
        import warnings

        warnings.warn(
            f"Optional dependencies missing for full geometry functionality: {', '.join(missing)}",
            ImportWarning,
        )


# Check dependencies on import
_check_optional_dependencies()
