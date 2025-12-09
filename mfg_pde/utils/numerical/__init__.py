"""
Numerical utilities for MFG computations.

This module provides numerical algorithms and helper functions commonly needed
in MFG research projects, including kernel functions, GFDM operators, RBF operators,
particle interpolation, signed distance functions, and computational utilities.

Submodules:
- kernels: Kernel functions (Gaussian, Wendland, B-spline) for GFDM, KDE, SPH
- gfdm_operators: GFDM differential operators (GFDMOperator class)
- kernel_rbf_operators: RBF-based differential operators using kernel infrastructure
- particle: Particle-based methods (Monte Carlo, MCMC, interpolation)
"""

# Flux diagnostics for mass conservation analysis
from mfg_pde.utils.numerical.flux_diagnostics import (
    BoundaryFluxResult,
    FluxDiagnostics,
    FluxSummary,
    compute_mass_conservation_error,
)
from mfg_pde.utils.numerical.gfdm_operators import GFDMOperator
from mfg_pde.utils.numerical.hjb_policy_iteration import (
    HJBPolicyProblem,
    create_lq_policy_problem,
    policy_iteration_hjb,
)
from mfg_pde.utils.numerical.kernel_rbf_operators import RBFOperator, create_rbf_operator

# Kernels - general numerical functions (not particle-specific)
from mfg_pde.utils.numerical.kernels import (
    CubicSplineKernel,
    GaussianKernel,
    WendlandKernel,
    create_kernel,
)
from mfg_pde.utils.numerical.monotonicity_stats import MonotonicityStats
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

__all__ = [
    # GFDM operators
    "GFDMOperator",
    # Monotonicity tracking
    "MonotonicityStats",
    # RBF operators
    "RBFOperator",
    "create_rbf_operator",
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
]
