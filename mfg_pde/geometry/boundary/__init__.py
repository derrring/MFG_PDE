"""
Boundary condition management for MFG problems.

This module provides boundary condition specifications for 1D, 2D, and 3D domains,
supporting Dirichlet, Neumann, Robin, periodic, and no-flux conditions.

Architecture:
- **Specification Layer** (dimension-agnostic):
  - types.py: BCType enum, BCSegment dataclass
  - conditions.py: Unified BoundaryConditions class and factory functions

- **Application Layer** (dimension-specific for performance):
  - applicator_base.py: Abstract protocols and base classes
  - applicator_fdm.py: FDM ghost cell BC application (1D/2D/3D/nD)
  - applicator_fem.py: FEM mesh-based BC application (2D/3D)
  - bc_2d.py: Optimized 2D FEM classes
  - bc_3d.py: Optimized 3D FEM classes

- **Legacy**:
  - legacy.py: Legacy 1D BoundaryConditions for backward compatibility

Usage:
    # Uniform BC (same type on all boundaries)
    bc = neumann_bc(dimension=2)

    # Mixed BC (different types on different segments)
    exit = BCSegment(name="exit", bc_type=BCType.DIRICHLET, boundary="x_max")
    wall = BCSegment(name="wall", bc_type=BCType.NEUMANN)
    bc = mixed_bc([exit, wall], dimension=2, domain_bounds=bounds)

    # FDM application
    padded = apply_boundary_conditions_2d(field, bc, bounds)
    # or
    applicator = FDMApplicator(dimension=2)
    padded = applicator.apply(field, bc, domain_bounds=bounds)

    # FEM application
    applicator = FEMApplicator(dimension=2)
    applicator.add_dirichlet(region=0, value=0.0)
    matrix, rhs = applicator.apply(matrix, rhs, mesh)
"""

# =============================================================================
# Base Classes and Protocols (applicator hierarchy)
# =============================================================================

from .applicator_base import (
    # Base classes
    BaseBCApplicator,
    BaseGraphApplicator,
    BaseMeshfreeApplicator,
    BaseStructuredApplicator,
    BaseUnstructuredApplicator,
    # Protocols
    BCApplicatorProtocol,
    BoundaryCalculator,
    BoundaryCapable,
    # Topology implementations (Issue #516)
    BoundedTopology,
    # Calculator implementations (physics-based naming)
    DirichletCalculator,
    # Enums
    DiscretizationType,
    FPNoFluxCalculator,  # -> ZeroFluxCalculator
    GridType,
    LinearExtrapolationCalculator,
    NeumannCalculator,
    # Backward compatibility aliases
    NoFluxCalculator,  # -> ZeroGradientCalculator
    PeriodicTopology,
    QuadraticExtrapolationCalculator,
    RobinCalculator,
    Topology,
    ZeroFluxCalculator,  # J·n = 0 (mass conservation)
    ZeroGradientCalculator,  # du/dn = 0 (edge extension)
    # Physics-aware ghost cell (for advection-diffusion/FP)
    ghost_cell_advection_diffusion_no_flux,
    # Ghost cell helpers (2nd-order)
    ghost_cell_dirichlet,
    ghost_cell_fp_no_flux,
    # Extrapolation ghost cell (for unbounded domains)
    ghost_cell_linear_extrapolation,
    ghost_cell_neumann,
    ghost_cell_quadratic_extrapolation,
    ghost_cell_robin,
    # High-order ghost cell extrapolation (4th/5th order for WENO)
    high_order_ghost_dirichlet,
    high_order_ghost_neumann,
)

# =============================================================================
# FDM Applicator (ghost cell method for structured grids)
# =============================================================================
from .applicator_fdm import (
    FDMApplicator,
    GhostBuffer,
    GhostCellConfig,
    PreallocatedGhostBuffer,
    apply_boundary_conditions_1d,
    apply_boundary_conditions_2d,
    apply_boundary_conditions_3d,
    apply_boundary_conditions_nd,
    bc_to_topology_calculator,
    create_boundary_mask_2d,
    create_ghost_buffer_from_bc,
    get_ghost_values_nd,
)

# =============================================================================
# FEM Applicator (mesh-based method for unstructured grids)
# =============================================================================
from .applicator_fem import (
    # 1D classes (optimized, from fem_bc_1d.py)
    BoundaryCondition1D,
    # 2D classes (optimized, from fem_bc_2d.py)
    BoundaryCondition2D,
    # 3D classes (optimized, from fem_bc_3d.py)
    BoundaryCondition3D,
    # Base FEM class alias
    BoundaryConditionFEM,
    BoundaryConditionManager1D,
    BoundaryConditionManager2D,
    BoundaryConditionManager3D,
    # Geometric BC manager
    BoundaryManager,
    DirichletBC1D,
    DirichletBC2D,
    DirichletBC3D,
    # Unified dispatchers
    FEMApplicator,
    GeometricBoundaryCondition,
    MFGBoundaryHandler1D,
    MFGBoundaryHandler2D,
    MFGBoundaryHandler3D,
    MFGBoundaryHandlerFEM,
    NeumannBC1D,
    NeumannBC2D,
    NeumannBC3D,
    PeriodicBC1D,
    PeriodicBC2D,
    PeriodicBC3D,
    RobinBC1D,
    RobinBC2D,
    RobinBC3D,
    create_box_boundary_conditions,
    create_circle_boundary_conditions,
    create_interval_boundary_conditions,
    create_rectangle_boundary_conditions,
    create_sphere_boundary_conditions,
    # Helper functions
    get_bc_class,
    get_manager_class,
)

# =============================================================================
# Graph Applicator (graph, network, and maze domains)
# =============================================================================
from .applicator_graph import (
    EdgeBC,
    GraphApplicator,
    GraphBCConfig,
    GraphBCType,
    NodeBC,
    create_graph_applicator,
    create_maze_applicator,
)

# =============================================================================
# Meshfree Applicator (geometry-based, for collocation methods)
# =============================================================================
from .applicator_meshfree import (
    MeshfreeApplicator,
    ParticleReflector,
    SDFParticleBCHandler,
)

# =============================================================================
# Particle Applicator (BC-segment-based, for Lagrangian particle solvers)
# =============================================================================
from .applicator_particle import ParticleApplicator

# Unified BoundaryConditions class and factory functions
from .conditions import (
    BoundaryConditions,
    MixedBoundaryConditions,  # Alias for backward compatibility
    dirichlet_bc,
    mixed_bc,
    neumann_bc,
    no_flux_bc,
    periodic_bc,
    robin_bc,
    uniform_bc,
)

# =============================================================================
# BC Dispatch (unified entry point for solvers)
# =============================================================================
from .dispatch import (
    apply_bc,
    get_applicator_for_geometry,
    validate_bc_compatibility,
)

# =============================================================================
# Legacy/Deprecated (backward compatibility - will be removed in v1.0.0)
# =============================================================================
# 1D FDM boundary conditions (simple left/right specification)
# DEPRECATED: Use conditions.BoundaryConditions with dimension=1 instead
from .fdm_bc_1d import BoundaryConditions as BoundaryConditions1DFDM

# =============================================================================
# Core Types (dimension-agnostic BC specification)
# =============================================================================
from .types import (
    BCSegment,
    BCType,
    create_standard_boundary_names,
)

# Backward compatibility alias (DEPRECATED)
LegacyBoundaryConditions1D = BoundaryConditions1DFDM

__all__ = [
    # Base classes and protocols
    "DiscretizationType",
    "GridType",
    "BCApplicatorProtocol",
    "BoundaryCapable",
    "Topology",
    "BoundaryCalculator",
    "BaseBCApplicator",
    "BaseStructuredApplicator",
    "BaseUnstructuredApplicator",
    "BaseMeshfreeApplicator",
    "BaseGraphApplicator",
    # Topology implementations (Issue #516)
    "PeriodicTopology",
    "BoundedTopology",
    # Calculator implementations (Issue #516)
    "DirichletCalculator",
    "NeumannCalculator",
    "RobinCalculator",
    "NoFluxCalculator",
    "LinearExtrapolationCalculator",
    "QuadraticExtrapolationCalculator",
    "FPNoFluxCalculator",
    # Ghost cell helper functions
    "ghost_cell_dirichlet",
    "ghost_cell_neumann",
    "ghost_cell_robin",
    # High-order ghost cell extrapolation (4th/5th order for WENO)
    "high_order_ghost_dirichlet",
    "high_order_ghost_neumann",
    # Physics-aware ghost cell (for advection-diffusion/FP)
    "ghost_cell_fp_no_flux",
    "ghost_cell_advection_diffusion_no_flux",
    # Extrapolation ghost cell (for unbounded domains)
    "ghost_cell_linear_extrapolation",
    "ghost_cell_quadratic_extrapolation",
    # Core types
    "BCType",
    "BCSegment",
    "create_standard_boundary_names",
    # Unified BC class
    "BoundaryConditions",
    "MixedBoundaryConditions",
    # Factory functions (uniform BCs)
    "uniform_bc",
    "periodic_bc",
    "dirichlet_bc",
    "neumann_bc",
    "no_flux_bc",
    "robin_bc",
    "mixed_bc",
    # Physics-based Calculator names (preferred)
    "ZeroGradientCalculator",  # du/dn = 0 (edge extension)
    "ZeroFluxCalculator",  # J·n = 0 (mass conservation)
    # FDM Applicator (Topology/Calculator composition - Issue #516)
    "GhostBuffer",
    "bc_to_topology_calculator",
    "create_ghost_buffer_from_bc",
    "FDMApplicator",
    "GhostCellConfig",
    "PreallocatedGhostBuffer",
    "apply_boundary_conditions_1d",
    "apply_boundary_conditions_2d",
    "apply_boundary_conditions_3d",
    "apply_boundary_conditions_nd",
    "create_boundary_mask_2d",
    "get_ghost_values_nd",
    # FEM Applicator - dispatchers
    "FEMApplicator",
    "MFGBoundaryHandlerFEM",
    # FEM base class alias
    "BoundaryConditionFEM",
    # FEM Applicator - managers
    "BoundaryManager",
    "GeometricBoundaryCondition",
    # FEM 1D classes
    "BoundaryCondition1D",
    "BoundaryConditionManager1D",
    "DirichletBC1D",
    "NeumannBC1D",
    "RobinBC1D",
    "PeriodicBC1D",
    "MFGBoundaryHandler1D",
    "create_interval_boundary_conditions",
    # FEM 2D classes
    "BoundaryCondition2D",
    "BoundaryConditionManager2D",
    "DirichletBC2D",
    "NeumannBC2D",
    "RobinBC2D",
    "PeriodicBC2D",
    "MFGBoundaryHandler2D",
    "create_rectangle_boundary_conditions",
    "create_circle_boundary_conditions",
    # FEM 3D classes
    "BoundaryCondition3D",
    "BoundaryConditionManager3D",
    "DirichletBC3D",
    "NeumannBC3D",
    "RobinBC3D",
    "PeriodicBC3D",
    "MFGBoundaryHandler3D",
    "create_box_boundary_conditions",
    "create_sphere_boundary_conditions",
    # Helper functions
    "get_bc_class",
    "get_manager_class",
    # Meshfree Applicator (geometry-based)
    "MeshfreeApplicator",
    "ParticleReflector",
    "SDFParticleBCHandler",
    # Particle Applicator (BC-segment-based)
    "ParticleApplicator",
    # Graph Applicator
    "GraphApplicator",
    "GraphBCConfig",
    "GraphBCType",
    "NodeBC",
    "EdgeBC",
    "create_graph_applicator",
    "create_maze_applicator",
    # 1D FDM boundary conditions
    "BoundaryConditions1DFDM",
    "LegacyBoundaryConditions1D",  # Backward compat alias
    # BC Dispatch (unified entry point for solvers - Issue #527)
    "apply_bc",
    "get_applicator_for_geometry",
    "validate_bc_compatibility",
]
