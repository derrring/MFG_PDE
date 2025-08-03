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

from .base_geometry import BaseGeometry, MeshData
from .boundary_manager import BoundaryManager, GeometricBoundaryCondition
from .domain_1d import (
    BoundaryConditions,
    dirichlet_bc,
    Domain1D,
    neumann_bc,
    no_flux_bc,
    periodic_bc,
    robin_bc,
)
from .domain_2d import Domain2D
from .mesh_manager import MeshManager, MeshPipeline
from .network_backend import (
    get_backend_manager,
    NetworkBackendType,
    OperationType,
    set_preferred_backend,
)
from .network_geometry import (
    BaseNetworkGeometry,
    compute_network_statistics,
    create_network,
    GridNetwork,
    NetworkData,
    NetworkType,
    RandomNetwork,
    ScaleFreeNetwork,
)
from .one_dimensional_amr import (
    create_1d_amr_mesh,
    Interval1D,
    OneDimensionalAMRMesh,
    OneDimensionalErrorEstimator,
)
from .triangular_amr import (
    create_triangular_amr_mesh,
    TriangleElement,
    TriangularAMRMesh,
    TriangularMeshErrorEstimator,
)

__all__ = [
    # 1D domain components
    "BoundaryConditions",
    "Domain1D",
    "periodic_bc",
    "dirichlet_bc",
    "neumann_bc",
    "no_flux_bc",
    "robin_bc",
    # Multi-dimensional geometry components
    "BaseGeometry",
    "MeshData",
    "Domain2D",
    "MeshManager",
    "MeshPipeline",
    "BoundaryManager",
    "GeometricBoundaryCondition",
    # Network geometry components
    "NetworkData",
    "NetworkType",
    "BaseNetworkGeometry",
    "GridNetwork",
    "RandomNetwork",
    "ScaleFreeNetwork",
    "create_network",
    "compute_network_statistics",
    # Network backend components
    "NetworkBackendType",
    "OperationType",
    "get_backend_manager",
    "set_preferred_backend",
    # Triangular AMR components
    "TriangularAMRMesh",
    "TriangleElement",
    "TriangularMeshErrorEstimator",
    "create_triangular_amr_mesh",
    # 1D AMR components
    "OneDimensionalAMRMesh",
    "Interval1D",
    "OneDimensionalErrorEstimator",
    "create_1d_amr_mesh",
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
