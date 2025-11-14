"""
Geometry package for MFG_PDE: Professional mesh generation and complex domain support.

This package implements comprehensive geometry management for MFG problems:
- Cartesian grids (SimpleGrid1D/2D/3D): Regular finite difference grids
- Unstructured meshes (Mesh2D/3D): FEM/FVM triangular/tetrahedral meshes via Gmsh
- Implicit domains: High-dimensional meshfree domains with signed distance functions
- Gmsh → Meshio → PyVista pipeline for professional mesh generation
- Advanced boundary condition management for complex domains

Key Components:
- SimpleGrid1D/2D/3D: Regular Cartesian grids for finite difference methods
- Mesh2D/3D: Unstructured meshes for FEM/FVM (d≤3)
- implicit: Meshfree geometry infrastructure for any dimension
  - Hyperrectangle: Axis-aligned boxes (O(d) sampling, no rejection!)
  - Hypersphere: Balls/circles for obstacles
  - CSG operations: Union, Intersection, Difference for complex domains
- BaseGeometry: Abstract base class for all geometry types
- MeshData: Universal mesh data container
- MeshPipeline: Complete Gmsh → Meshio → PyVista workflow orchestration
- MeshManager: High-level mesh management for multiple geometries
- BoundaryManager: Advanced boundary condition management

Discretization Methods:
- Use SimpleGrid* for finite difference solvers
- Use Mesh* (Gmsh) for FEM/FVM problems (d≤3)
- Use implicit.* (SDF) for high-dimensional particle-collocation (d≥4)
"""

from __future__ import annotations

import warnings as _warnings

# AMR imports (from subdirectory - canonical locations)
from .amr.amr_1d import Interval1D, OneDimensionalAMRMesh, OneDimensionalErrorEstimator, create_1d_amr_mesh
from .amr.amr_triangular_2d import (
    TriangleElement,
    TriangularAMRMesh,
    TriangularMeshErrorEstimator,
    create_triangular_amr_mesh,
)

# Base geometry classes
from .base_geometry import BaseGeometry, MeshData

# Boundary conditions from subdirectories
from .boundary import (
    BoundaryConditions,
    BoundaryManager,
    GeometricBoundaryCondition,
)

# Legacy boundary condition imports (from old file names)
from .boundary_conditions_1d import dirichlet_bc, neumann_bc, no_flux_bc, periodic_bc, robin_bc
from .boundary_conditions_2d import (
    BoundaryCondition2D,
    BoundaryConditionManager2D,
    DirichletBC2D,
    MFGBoundaryHandler2D,
    NeumannBC2D,
    PeriodicBC2D,
    RobinBC2D,
    create_circle_boundary_conditions,
    create_rectangle_boundary_conditions,
)
from .boundary_conditions_3d import (
    BoundaryCondition3D,
    BoundaryConditionManager3D,
    DirichletBC3D,
    MFGBoundaryHandler3D,
    NeumannBC3D,
    PeriodicBC3D,
    RobinBC3D,
    create_box_boundary_conditions,
    create_sphere_boundary_conditions,
)

# Graph-based geometry (networks + mazes)
from .graph import (
    BaseNetworkGeometry,
    GridNetwork,
    HybridMazeGenerator,
    MazeAlgorithm,
    MazeConfig,
    NetworkData,
    NetworkType,
    PerfectMazeGenerator,
    RandomNetwork,
    ScaleFreeNetwork,
    VoronoiMazeGenerator,
    maze_Algorithm,
    maze_CellularAutomataConfig,
    maze_CellularAutomataGenerator,
    maze_Config,
    maze_HybridGenerator,
    maze_PerfectMazeGenerator,
    maze_RecursiveDivisionConfig,
    maze_RecursiveDivisionGenerator,
    maze_VoronoiGenerator,
)

# Legacy network imports (from old file names - now in graph subdirectory)
from .graph.network import compute_network_statistics, create_network

# Network backend (from graph subdirectory - canonical location)
from .graph.network_backend import NetworkBackendType, OperationType, get_backend_manager, set_preferred_backend

# Grid geometry - Import from subdirectories (canonical locations)
from .grids.grid_1d import SimpleGrid1D
from .grids.grid_2d import SimpleGrid2D, SimpleGrid3D
from .grids.tensor_grid import TensorProductGrid

# Implicit geometry
from .implicit import (
    ComplementDomain,
    DifferenceDomain,
    Hyperrectangle,
    Hypersphere,
    ImplicitDomain,
    IntersectionDomain,
    UnionDomain,
)

# Mesh geometry
from .meshes import Mesh1D, Mesh2D, Mesh3D, MeshManager, MeshPipeline

# Geometric operators
from .operators import GeometryProjector

# Point cloud geometry for particle-based solvers (Issue #269)
from .point_cloud import PointCloudGeometry

# Legacy projection imports (from old file names)
from .projection import ProjectionRegistry

# Unified geometry protocol
from .protocol import (
    GeometryProtocol,
    GeometryType,
    detect_geometry_type,
    is_geometry_compatible,
    validate_geometry,
)

# Legacy grid imports (from old file names)

# Backward compatibility aliases (DEPRECATED - will be removed in v0.12.0)
Domain1D = SimpleGrid1D  # Use SimpleGrid1D instead
Domain2D = Mesh2D  # Use Mesh2D instead
Domain3D = Mesh3D  # Use Mesh3D instead

# Issue deprecation warning for Domain* aliases
_warnings.warn(
    "Domain1D, Domain2D, and Domain3D aliases are deprecated and will be removed in v0.12.0. "
    "Please use SimpleGrid1D, Mesh2D, and Mesh3D directly:\n"
    "  from mfg_pde.geometry import SimpleGrid1D, Mesh2D, Mesh3D",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    # Multi-dimensional geometry components
    "BaseGeometry",
    "BaseNetworkGeometry",
    # Boundary condition components
    "BoundaryCondition2D",
    "BoundaryCondition3D",
    "BoundaryConditionManager2D",
    "BoundaryConditionManager3D",
    "BoundaryConditions",
    "BoundaryConditions2D",
    "BoundaryConditions3D",
    "BoundaryManager",
    "BoundaryRegion2D",
    "BoundaryRegion3D",
    # Specific boundary condition types
    "DirichletBC2D",
    "DirichletBC3D",
    # Geometry components (new naming convention)
    "SimpleGrid1D",
    "SimpleGrid2D",
    "SimpleGrid3D",
    "Mesh1D",
    "Mesh2D",
    "Mesh3D",
    # Backward compatibility aliases
    "Domain1D",
    "Domain2D",
    "Domain3D",
    "GeometricBoundaryCondition",
    # Unified geometry protocol
    "GeometryProtocol",
    "GeometryType",
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
    "TensorProductGrid",
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
    "create_triangular_amr_mesh",
    "detect_geometry_type",
    "dirichlet_bc",
    "get_backend_manager",
    "is_geometry_compatible",
    # Maze generation (original names for backward compatibility)
    "HybridMazeGenerator",
    "MazeAlgorithm",
    "MazeConfig",
    "PerfectMazeGenerator",
    "VoronoiMazeGenerator",
    # Maze generation (with maze_ prefix)
    "maze_Algorithm",
    "maze_CellularAutomataConfig",
    "maze_CellularAutomataGenerator",
    "maze_Config",
    "maze_HybridGenerator",
    "maze_PerfectMazeGenerator",
    "maze_RecursiveDivisionConfig",
    "maze_RecursiveDivisionGenerator",
    "maze_VoronoiGenerator",
    "neumann_bc",
    "no_flux_bc",
    "periodic_bc",
    "robin_bc",
    "set_preferred_backend",
    "validate_geometry",
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
