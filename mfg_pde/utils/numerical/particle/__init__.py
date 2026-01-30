"""
Particle-based numerical utilities for MFG computations.

This submodule provides tools for particle-based methods including:
- Monte Carlo sampling and integration
- MCMC and Hamiltonian Monte Carlo sampling
- Particle-grid interpolation
- Boundary condition handling for particles
- Density-based adaptive sampling

Note: Kernel functions (Gaussian, Wendland, etc.) have been moved to
mfg_pde.utils.numerical.kernels as they are general numerical functions.
Re-exported here for backward compatibility.

Organization:
- sampling: Monte Carlo integration, variance reduction, density-based sampling
- mcmc: MCMC samplers (Metropolis-Hastings, HMC, NUTS, Langevin)
- interpolation: Particle-grid conversion utilities
- boundary: Particle boundary condition application
- kde_boundary: Boundary-corrected KDE for density estimation (Issue #709)

Typical Usage:
    from mfg_pde.utils.numerical.particle import (
        # Interpolation
        interpolate_grid_to_particles,
        interpolate_particles_to_grid,
        estimate_kde_bandwidth,
        # Monte Carlo
        monte_carlo_integrate,
        MCConfig,
        MCResult,
        # Boundary conditions
        apply_boundary_conditions_gpu,
        apply_boundary_conditions_numpy,
        # MCMC
        MetropolisHastings,
        HamiltonianMonteCarlo,
        MCMCConfig,
    )

    # For kernels, prefer the new location:
    from mfg_pde.utils.numerical.kernels import GaussianKernel, WendlandKernel
"""

# Boundary conditions
# Kernels - re-export from new location for backward compatibility
from mfg_pde.utils.numerical.kernels import (
    CubicKernel,
    CubicSplineKernel,
    GaussianKernel,
    Kernel,
    QuarticKernel,
    QuinticSplineKernel,
    WendlandKernel,
    create_kernel,
)
from mfg_pde.utils.numerical.particle.boundary import (
    apply_boundary_conditions_gpu,
    apply_boundary_conditions_numpy,
)

# Boundary-corrected KDE (Issue #709)
from mfg_pde.utils.numerical.particle.kde_boundary import (
    # Unified API
    create_ghost_particles,
    reflection_kde,
    renormalization_kde,
    beta_kde,
    # Legacy aliases
    create_ghost_particles_1d,
    create_ghost_particles_nd,
    reflection_kde_1d,
    reflection_kde_nd,
)

# Interpolation
from mfg_pde.utils.numerical.particle.interpolation import (
    estimate_kde_bandwidth,
    interpolate_grid_to_particles,
    interpolate_particles_to_grid,
)

# MCMC sampling (mcmc.py -> mcmc.py)
from mfg_pde.utils.numerical.particle.mcmc import (
    HamiltonianMonteCarlo,
    LangevinDynamics,
    MCMCConfig,
    MCMCResult,
    MCMCSampler,
    MetropolisHastings,
    NoUTurnSampler,
    bayesian_neural_network_sampling,
    compute_rhat,
    effective_sample_size,
    sample_mfg_posterior,
)

# Monte Carlo sampling (monte_carlo.py -> sampling.py)
from mfg_pde.utils.numerical.particle.sampling import (
    ControlVariates,
    ImportanceMCSampler,
    MCConfig,
    MCResult,
    MCSampler,
    PoissonDiskSampler,
    QuasiMCSampler,
    StratifiedMCSampler,
    UniformMCSampler,
    adaptive_monte_carlo,
    estimate_expectation,
    integrate_gaussian_quadrature_mc,
    monte_carlo_integrate,
    sample_from_density,
    sample_from_scattered_density,
)

__all__ = [
    # Kernels
    "Kernel",
    "GaussianKernel",
    "WendlandKernel",
    "CubicSplineKernel",
    "QuinticSplineKernel",
    "CubicKernel",
    "QuarticKernel",
    "create_kernel",
    # Interpolation
    "interpolate_grid_to_particles",
    "interpolate_particles_to_grid",
    "estimate_kde_bandwidth",
    # Boundary conditions
    "apply_boundary_conditions_gpu",
    "apply_boundary_conditions_numpy",
    # Boundary-corrected KDE (Issue #709)
    "create_ghost_particles",
    "reflection_kde",
    "renormalization_kde",
    "beta_kde",
    # Legacy aliases
    "create_ghost_particles_1d",
    "create_ghost_particles_nd",
    "reflection_kde_1d",
    "reflection_kde_nd",
    # Monte Carlo
    "MCConfig",
    "MCResult",
    "MCSampler",
    "UniformMCSampler",
    "StratifiedMCSampler",
    "QuasiMCSampler",
    "ImportanceMCSampler",
    "PoissonDiskSampler",
    "ControlVariates",
    "monte_carlo_integrate",
    "adaptive_monte_carlo",
    "integrate_gaussian_quadrature_mc",
    "estimate_expectation",
    "sample_from_density",
    "sample_from_scattered_density",
    # MCMC
    "MCMCConfig",
    "MCMCResult",
    "MCMCSampler",
    "MetropolisHastings",
    "HamiltonianMonteCarlo",
    "NoUTurnSampler",
    "LangevinDynamics",
    "compute_rhat",
    "effective_sample_size",
    "sample_mfg_posterior",
    "bayesian_neural_network_sampling",
]
