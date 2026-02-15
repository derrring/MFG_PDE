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

    # FDM application (preferred API)
    padded = pad_array_with_ghosts(field, bc)
    # or using class-based API
    applicator = FDMApplicator(dimension=2)
    padded = applicator.apply(field, bc, domain_bounds=bounds)

    # FEM application
    applicator = FEMApplicator(dimension=2)
    applicator.add_dirichlet(region=0, value=0.0)
    matrix, rhs = applicator.apply(matrix, rhs, mesh)
"""

# =============================================================================
# FDM Applicator (ghost cell method for structured grids)
# =============================================================================
# Internal implementation details — kept importable for backward compat
# but not part of public API (__all__). Import directly from submodule:
#   from mfg_pde.geometry.boundary.applicator_base import DirichletCalculator
#   from mfg_pde.geometry.boundary.applicator_fdm import PreallocatedGhostBuffer
from ._compat import get_ghost_values_nd as get_ghost_values_nd
from .applicator_base import (
    BaseBCApplicator as BaseBCApplicator,
)
from .applicator_base import (
    BaseGraphApplicator as BaseGraphApplicator,
)
from .applicator_base import (
    BaseMeshfreeApplicator as BaseMeshfreeApplicator,
)
from .applicator_base import (
    BaseStructuredApplicator as BaseStructuredApplicator,
)
from .applicator_base import (
    BaseUnstructuredApplicator as BaseUnstructuredApplicator,
)
from .applicator_base import (
    BCApplicatorProtocol as BCApplicatorProtocol,
)
from .applicator_base import (
    BoundaryCalculator as BoundaryCalculator,
)
from .applicator_base import (
    BoundaryCapable as BoundaryCapable,
)
from .applicator_base import (
    BoundedTopology as BoundedTopology,
)
from .applicator_base import (
    DirichletCalculator as DirichletCalculator,
)
from .applicator_base import (
    DiscretizationType as DiscretizationType,
)
from .applicator_base import (
    FPNoFluxCalculator as FPNoFluxCalculator,
)
from .applicator_base import (
    GridType as GridType,
)
from .applicator_base import (
    LinearExtrapolationCalculator as LinearExtrapolationCalculator,
)
from .applicator_base import (
    NeumannCalculator as NeumannCalculator,
)
from .applicator_base import (
    NoFluxCalculator as NoFluxCalculator,
)
from .applicator_base import (
    PeriodicTopology as PeriodicTopology,
)
from .applicator_base import (
    QuadraticExtrapolationCalculator as QuadraticExtrapolationCalculator,
)
from .applicator_base import (
    RobinCalculator as RobinCalculator,
)
from .applicator_base import (
    Topology as Topology,
)
from .applicator_base import (
    ZeroFluxCalculator as ZeroFluxCalculator,
)
from .applicator_base import (
    ZeroGradientCalculator as ZeroGradientCalculator,
)
from .applicator_base import (
    ghost_cell_advection_diffusion_no_flux as ghost_cell_advection_diffusion_no_flux,
)
from .applicator_base import (
    ghost_cell_dirichlet as ghost_cell_dirichlet,
)
from .applicator_base import (
    ghost_cell_fp_no_flux as ghost_cell_fp_no_flux,
)
from .applicator_base import (
    ghost_cell_linear_extrapolation as ghost_cell_linear_extrapolation,
)
from .applicator_base import (
    ghost_cell_neumann as ghost_cell_neumann,
)
from .applicator_base import (
    ghost_cell_quadratic_extrapolation as ghost_cell_quadratic_extrapolation,
)
from .applicator_base import (
    ghost_cell_robin as ghost_cell_robin,
)
from .applicator_base import (
    high_order_ghost_dirichlet as high_order_ghost_dirichlet,
)
from .applicator_base import (
    high_order_ghost_neumann as high_order_ghost_neumann,
)
from .applicator_fdm import (
    FDMApplicator,
    # Concrete function API (Issue #577 - preferred)
    pad_array_with_ghosts,
)
from .applicator_fdm import (
    GhostBuffer as GhostBuffer,
)
from .applicator_fdm import (
    GhostCellConfig as GhostCellConfig,
)
from .applicator_fdm import (
    PreallocatedGhostBuffer as PreallocatedGhostBuffer,
)
from .applicator_fdm import (
    bc_to_topology_calculator as bc_to_topology_calculator,
)
from .applicator_fdm import (
    create_ghost_buffer_from_bc as create_ghost_buffer_from_bc,
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
# Implicit Applicator (for dual geometry: structured grid + SDF boundary - Issue #637)
# =============================================================================
from .applicator_implicit import (
    ImplicitApplicator,
    create_implicit_applicator,
)

# =============================================================================
# Interpolation Applicator (for Semi-Lagrangian, particle, RBF methods - Issue #636)
# =============================================================================
from .applicator_interpolation import (
    InterpolationApplicator,
    create_interpolation_applicator,
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

# =============================================================================
# BC Coupling (for coupled MFG systems - Issue #574)
# =============================================================================
from .bc_coupling import (
    compute_adjoint_consistent_bc_values,
    compute_boundary_log_density_gradient_1d,
    create_adjoint_consistent_bc_1d,
)
from .bc_coupling import (
    compute_coupled_hjb_bc_values as compute_coupled_hjb_bc_values,
)

# BC utilities for solver-agnostic BC type detection (Issue #702)
from .bc_utils import (
    bc_type_to_geometric_operation,
    get_bc_type_string,
)

# Unified BoundaryConditions class and factory functions
from .conditions import (
    BoundaryConditions,
    dirichlet_bc,
    mixed_bc,
    mixed_bc_from_regions,
    neumann_bc,
    no_flux_bc,
    periodic_bc,
    robin_bc,
    uniform_bc,
)
from .conditions import (
    MixedBoundaryConditions as MixedBoundaryConditions,
)
from .constraint_protocol import ConstraintProtocol
from .constraints import BilateralConstraint, ObstacleConstraint

# =============================================================================
# Corner Handling (Issue #521 - unified corner handling architecture)
# =============================================================================
from .corner import (
    DEFAULT_CORNER_STRATEGY,
    CornerStrategy,
    CornerStrategyLiteral,
    absorb_positions,
    reflect_positions,
    reflect_velocity,
    reflect_velocity_with_normal,
    validate_corner_strategy,
)
from .corner import (
    wrap_positions as wrap_positions,
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
# Shared BC Enforcement Utilities (Issue #636)
# =============================================================================
from .enforcement import (
    enforce_dirichlet_value_nd,
    enforce_neumann_value_nd,
    enforce_periodic_value_nd,
    enforce_robin_value_nd,
)

# =============================================================================
# Legacy/Deprecated (backward compatibility - will be removed in v1.0.0)
# =============================================================================
# 1D FDM boundary conditions (simple left/right specification)
# DEPRECATED: Use conditions.BoundaryConditions with dimension=1 instead
from .fdm_bc_1d import BoundaryConditions as BoundaryConditions1DFDM

# =============================================================================
# Ghost Point Utilities (reflection-based, for NEUMANN/NO-FLUX BC)
# Parallel to periodic.py which handles PERIODIC BC
# =============================================================================
from .ghost import (
    compute_normal_from_bounds,
    compute_normal_from_sdf,
    create_ghost_points_for_kde,
    create_ghost_stencil,
    create_reflection_ghost_points,
    reflect_point_across_plane,
)
from .handler_protocol import (
    AdvancedBoundaryHandler,
    BoundaryHandler,
    validate_boundary_handler,
)

# =============================================================================
# Periodic BC utilities (parallel to enforcement.py for DIRICHLET/NEUMANN)
# Issue #711: Refactored to be parallel to other BC types
# =============================================================================
from .periodic import (
    create_periodic_ghost_points,
)
from .periodic import (
    wrap_positions as periodic_wrap_positions,  # Canonical location
)

# =============================================================================
# Dynamic BC Value Providers (Issue #625)
# =============================================================================
from .providers import (
    AdjointConsistentProvider,
    BaseBCValueProvider,
    BCValueProvider,
    ConstantProvider,
    is_provider,
    resolve_provider,
)

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

# Backward compatibility alias for bc_coupling (Issue #574)
compute_boundary_log_density_gradient = compute_boundary_log_density_gradient_1d

__all__ = [
    # =========================================================================
    # Core Types (dimension-agnostic BC specification)
    # =========================================================================
    "BCType",
    "BCSegment",
    "create_standard_boundary_names",
    # Unified BC class and factory functions
    "BoundaryConditions",
    "uniform_bc",
    "periodic_bc",
    "dirichlet_bc",
    "neumann_bc",
    "no_flux_bc",
    "robin_bc",
    "mixed_bc",
    "mixed_bc_from_regions",
    # =========================================================================
    # Dynamic BC Value Providers (Issue #625)
    # =========================================================================
    "BCValueProvider",
    "BaseBCValueProvider",
    "AdjointConsistentProvider",
    "ConstantProvider",
    "is_provider",
    "resolve_provider",
    # =========================================================================
    # Solver BC Protocol (Issue #545)
    # =========================================================================
    "BoundaryHandler",
    "AdvancedBoundaryHandler",
    "validate_boundary_handler",
    # =========================================================================
    # Variational Inequality Constraints (Issue #591)
    # =========================================================================
    "ConstraintProtocol",
    "ObstacleConstraint",
    "BilateralConstraint",
    # =========================================================================
    # BC utilities (Issue #702)
    # =========================================================================
    "bc_type_to_geometric_operation",
    "get_bc_type_string",
    # =========================================================================
    # Applicators (public API)
    # =========================================================================
    # FDM (ghost cell method for structured grids)
    "FDMApplicator",
    "pad_array_with_ghosts",
    # FEM (mesh-based method for unstructured grids)
    "FEMApplicator",
    "BoundaryConditionFEM",
    "MFGBoundaryHandlerFEM",
    "BoundaryManager",
    "GeometricBoundaryCondition",
    # FEM dimension-specific classes
    "BoundaryCondition1D",
    "BoundaryConditionManager1D",
    "DirichletBC1D",
    "NeumannBC1D",
    "RobinBC1D",
    "PeriodicBC1D",
    "MFGBoundaryHandler1D",
    "create_interval_boundary_conditions",
    "BoundaryCondition2D",
    "BoundaryConditionManager2D",
    "DirichletBC2D",
    "NeumannBC2D",
    "RobinBC2D",
    "PeriodicBC2D",
    "MFGBoundaryHandler2D",
    "create_rectangle_boundary_conditions",
    "create_circle_boundary_conditions",
    "BoundaryCondition3D",
    "BoundaryConditionManager3D",
    "DirichletBC3D",
    "NeumannBC3D",
    "RobinBC3D",
    "PeriodicBC3D",
    "MFGBoundaryHandler3D",
    "create_box_boundary_conditions",
    "create_sphere_boundary_conditions",
    "get_bc_class",
    "get_manager_class",
    # Graph (graph, network, maze domains)
    "GraphApplicator",
    "GraphBCConfig",
    "GraphBCType",
    "NodeBC",
    "EdgeBC",
    "create_graph_applicator",
    "create_maze_applicator",
    # Implicit (dual geometry: structured grid + SDF - Issue #637)
    "ImplicitApplicator",
    "create_implicit_applicator",
    # Interpolation (Semi-Lagrangian, particle, RBF - Issue #636)
    "InterpolationApplicator",
    "create_interpolation_applicator",
    # Meshfree (geometry-based collocation methods)
    "MeshfreeApplicator",
    "ParticleReflector",
    "SDFParticleBCHandler",
    # Particle (BC-segment-based Lagrangian solvers)
    "ParticleApplicator",
    # =========================================================================
    # BC Dispatch (unified entry point for solvers - Issue #527)
    # =========================================================================
    "apply_bc",
    "get_applicator_for_geometry",
    "validate_bc_compatibility",
    # =========================================================================
    # BC Enforcement Utilities (Issue #636)
    # =========================================================================
    "enforce_dirichlet_value_nd",
    "enforce_neumann_value_nd",
    "enforce_periodic_value_nd",
    "enforce_robin_value_nd",
    # =========================================================================
    # BC Coupling (for coupled MFG systems - Issue #574)
    # =========================================================================
    "compute_adjoint_consistent_bc_values",
    "create_adjoint_consistent_bc_1d",
    "compute_boundary_log_density_gradient_1d",
    # =========================================================================
    # Corner Handling (Issue #521)
    # =========================================================================
    "reflect_positions",
    "absorb_positions",
    "reflect_velocity",
    "reflect_velocity_with_normal",
    "CornerStrategy",
    "CornerStrategyLiteral",
    "DEFAULT_CORNER_STRATEGY",
    "validate_corner_strategy",
    # =========================================================================
    # Periodic / Ghost Point Utilities
    # =========================================================================
    "create_periodic_ghost_points",
    "periodic_wrap_positions",
    "compute_normal_from_bounds",
    "compute_normal_from_sdf",
    "reflect_point_across_plane",
    "create_reflection_ghost_points",
    "create_ghost_stencil",
    "create_ghost_points_for_kde",
]

# =========================================================================
# Internal implementation details (importable but not part of public API)
# =========================================================================
# The following are still importable via:
#   from mfg_pde.geometry.boundary import DirichletCalculator
# but are NOT in __all__ (not shown in autocomplete, not part of public API).
#
# Internal to FDM applicator:
#   Topology, PeriodicTopology, BoundedTopology
#   BoundaryCalculator, DirichletCalculator, NeumannCalculator, RobinCalculator,
#   ZeroGradientCalculator, ZeroFluxCalculator, LinearExtrapolationCalculator,
#   QuadraticExtrapolationCalculator, NoFluxCalculator, FPNoFluxCalculator
#   ghost_cell_*, high_order_ghost_*
#   DiscretizationType, GridType
#   GhostBuffer, PreallocatedGhostBuffer, GhostCellConfig
#   bc_to_topology_calculator, create_ghost_buffer_from_bc
#   BaseBCApplicator, BaseStructuredApplicator, BaseUnstructuredApplicator,
#   BaseMeshfreeApplicator, BaseGraphApplicator, BCApplicatorProtocol, BoundaryCapable
#
# Deprecated aliases (will be removed in v1.0.0):
#   MixedBoundaryConditions -> BoundaryConditions
#   LegacyBoundaryConditions1D, BoundaryConditions1DFDM
#   compute_boundary_log_density_gradient -> compute_boundary_log_density_gradient_1d
#   compute_coupled_hjb_bc_values -> compute_adjoint_consistent_bc_values
#   get_ghost_values_nd -> pad_array_with_ghosts
#   wrap_positions -> periodic_wrap_positions
