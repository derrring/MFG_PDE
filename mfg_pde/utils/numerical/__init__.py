"""
Numerical utilities for MFG computations.

This module provides numerical algorithms and helper functions commonly needed
in MFG research projects, including differential operators, kernel functions,
GFDM operators, particle interpolation, and signed distance functions.

Primary Submodules:
- tensor_calculus: Complete discrete tensor calculus for regular grids
  (gradient, divergence, laplacian, hessian, tensor_diffusion, advection)
- gfdm_strategies: Differential operators for scattered points (GFDM/RBF-FD)
- kernels: Kernel functions (Gaussian, Wendland, B-spline) for GFDM, KDE, SPH
- nonlinear_solvers: Newton, fixed-point, policy iteration solvers
- particle: Particle-based methods (Monte Carlo, MCMC, interpolation)

Deprecated Submodules:
- grid_operators: Use tensor_calculus instead (v0.18.0)
- tensor_operators: Use tensor_calculus instead (v0.18.0)
- gfdm_operators: Use gfdm_strategies instead (v0.17.0)
- differential_utils: Use scipy.optimize or tensor_calculus (v0.18.0)
"""

# Flux diagnostics for mass conservation analysis
from mfg_pde.utils.numerical.flux_diagnostics import (
    BoundaryFluxResult,
    FluxDiagnostics,
    FluxSummary,
    compute_mass_conservation_error,
)
from mfg_pde.utils.numerical.gfdm_operators import GFDMOperator

# GFDM Strategy Pattern (modular operators and BC handlers)
from mfg_pde.utils.numerical.gfdm_strategies import (
    BoundaryHandler,
    DifferentialOperator,
    DirectCollocationHandler,
    GhostNodeHandler,
    LocalRBFOperator,
    TaylorOperator,
    UpwindOperator,
    create_bc_handler,
    create_operator,
)
from mfg_pde.utils.numerical.hjb_policy_iteration import (
    HJBPolicyProblem,
    create_lq_policy_problem,
    policy_iteration_hjb,
)

# Kernels - general numerical functions (not particle-specific)
from mfg_pde.utils.numerical.kernels import (
    CubicSplineKernel,
    GaussianKernel,
    MultiquadricKernel,
    PHSKernel,
    WendlandKernel,
    create_kernel,
)

# Mesh distance metrics for EOC analysis (GFDM)
from mfg_pde.utils.numerical.mesh_distances import (
    MeshDistances,
    compute_distances_for_eoc_study,
    compute_mesh_distances,
)
from mfg_pde.utils.numerical.monotonicity_stats import (
    MonotonicityStats,
    get_m_matrix_diagnostic_string,
    verify_m_matrix_property,
)
from mfg_pde.utils.numerical.nonlinear_solvers import FixedPointSolver, NewtonSolver, PolicyIterationSolver, SolverInfo

# Re-export particle utilities for convenience
from mfg_pde.utils.numerical.particle import (
    HamiltonianMonteCarlo,
    # Monte Carlo
    MCConfig,
    # MCMC
    MCMCConfig,
    MCMCResult,
    MCResult,
    MetropolisHastings,
    monte_carlo_integrate,
)

# Re-export particle interpolation from new location for backward compatibility
from mfg_pde.utils.numerical.particle.interpolation import (
    estimate_kde_bandwidth,
    interpolate_grid_to_particles,
    interpolate_particles_to_grid,
)
from mfg_pde.utils.numerical.sdf_utils import (
    sdf_box,
    sdf_complement,
    sdf_difference,
    sdf_gradient,
    sdf_intersection,
    sdf_smooth_intersection,
    sdf_smooth_union,
    sdf_sphere,
    sdf_union,
)

# Tensor Calculus - Primary differential operators for regular grids
from mfg_pde.utils.numerical.tensor_calculus import (
    advection,
    diffusion,
    divergence,
    gradient,
    gradient_simple,
    hessian,
    laplacian,
    tensor_diffusion,
)

__all__ = [
    # Tensor Calculus (primary - regular grids)
    "gradient",
    "gradient_simple",
    "divergence",
    "laplacian",
    "hessian",
    "diffusion",
    "tensor_diffusion",
    "advection",
    # GFDM operators (legacy, use gfdm_strategies)
    "GFDMOperator",
    # GFDM Strategy Pattern (scattered points)
    "DifferentialOperator",
    "BoundaryHandler",
    "TaylorOperator",
    "UpwindOperator",
    "LocalRBFOperator",
    "DirectCollocationHandler",
    "GhostNodeHandler",
    "create_operator",
    "create_bc_handler",
    # Monotonicity tracking
    "MonotonicityStats",
    "verify_m_matrix_property",
    "get_m_matrix_diagnostic_string",
    # HJB policy iteration
    "HJBPolicyProblem",
    "create_lq_policy_problem",
    "policy_iteration_hjb",
    # Nonlinear solvers
    "FixedPointSolver",
    "NewtonSolver",
    "PolicyIterationSolver",
    "SolverInfo",
    # Particle interpolation (from particle submodule)
    "estimate_kde_bandwidth",
    "interpolate_grid_to_particles",
    "interpolate_particles_to_grid",
    # Signed distance functions
    "sdf_box",
    "sdf_complement",
    "sdf_difference",
    "sdf_gradient",
    "sdf_intersection",
    "sdf_smooth_intersection",
    "sdf_smooth_union",
    "sdf_sphere",
    "sdf_union",
    # Kernels (from particle submodule)
    "GaussianKernel",
    "WendlandKernel",
    "CubicSplineKernel",
    "MultiquadricKernel",
    "PHSKernel",
    "create_kernel",
    # Monte Carlo (from particle submodule)
    "MCConfig",
    "MCResult",
    "monte_carlo_integrate",
    # MCMC (from particle submodule)
    "MCMCConfig",
    "MCMCResult",
    "MetropolisHastings",
    "HamiltonianMonteCarlo",
    # Flux diagnostics (mass conservation)
    "FluxDiagnostics",
    "BoundaryFluxResult",
    "FluxSummary",
    "compute_mass_conservation_error",
    # Mesh distances (EOC analysis)
    "MeshDistances",
    "compute_mesh_distances",
    "compute_distances_for_eoc_study",
]
