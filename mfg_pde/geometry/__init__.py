"""
Geometry package for MFG_PDE: Professional mesh generation and complex domain support.

This package implements comprehensive domain management for 1D, 2D, and 3D MFG problems:
- Domain1D: 1D domains with boundary conditions (foundation for MFG solvers)
- Domain2D: 2D domains with complex geometry support
- Gmsh → Meshio → PyVista pipeline for professional mesh generation
- Advanced boundary condition management for complex domains

Key Components:
- Domain1D: 1D domain and boundary condition management
- BaseGeometry: Abstract base class for all geometry types
- MeshData: Universal mesh data container
- Domain2D: 2D domain implementation with complex geometry support
- MeshPipeline: Complete Gmsh → Meshio → PyVista workflow orchestration
- MeshManager: High-level mesh management for multiple geometries
- BoundaryManager: Advanced boundary condition management
"""

from __future__ import annotations

from .base_geometry import BaseGeometry, MeshData
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
from .boundary_manager import BoundaryManager, GeometricBoundaryCondition
from .domain_1d import BoundaryConditions, Domain1D, dirichlet_bc, neumann_bc, no_flux_bc, periodic_bc, robin_bc
from .domain_2d import Domain2D
from .domain_3d import Domain3D
from .mesh_manager import MeshManager, MeshPipeline
from .network_backend import NetworkBackendType, OperationType, get_backend_manager, set_preferred_backend
from .network_geometry import (
    BaseNetworkGeometry,
    GridNetwork,
    NetworkData,
    NetworkType,
    RandomNetwork,
    ScaleFreeNetwork,
    compute_network_statistics,
    create_network,
)
from .one_dimensional_amr import Interval1D, OneDimensionalAMRMesh, OneDimensionalErrorEstimator, create_1d_amr_mesh
from .triangular_amr import TriangleElement, TriangularAMRMesh, TriangularMeshErrorEstimator, create_triangular_amr_mesh

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
    "BoundaryManager",
    # Specific boundary condition types
    "DirichletBC2D",
    "DirichletBC3D",
    "MFGBoundaryHandler2D",
    "MFGBoundaryHandler3D",
    "NeumannBC2D",
    "NeumannBC3D",
    "PeriodicBC2D",
    "PeriodicBC3D",
    "RobinBC2D",
    "RobinBC3D",
    # Boundary condition factory functions
    "create_box_boundary_conditions",
    "create_circle_boundary_conditions",
    "create_rectangle_boundary_conditions",
    "create_sphere_boundary_conditions",
    # Domain components
    "Domain1D",
    "Domain2D",
    "Domain3D",
    "GeometricBoundaryCondition",
    "GridNetwork",
    "Interval1D",
    "MeshData",
    "MeshManager",
    "MeshPipeline",
    # Network backend components
    "NetworkBackendType",
    # Network geometry components
    "NetworkData",
    "NetworkType",
    # 1D AMR components
    "OneDimensionalAMRMesh",
    "OneDimensionalErrorEstimator",
    "OperationType",
    "RandomNetwork",
    "ScaleFreeNetwork",
    "TriangleElement",
    # Triangular AMR components
    "TriangularAMRMesh",
    "TriangularMeshErrorEstimator",
    "compute_network_statistics",
    "create_1d_amr_mesh",
    "create_network",
    "create_triangular_amr_mesh",
    "dirichlet_bc",
    "get_backend_manager",
    "neumann_bc",
    "no_flux_bc",
    "periodic_bc",
    "robin_bc",
    "set_preferred_backend",
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
