"""
Particle-based numerical utilities for MFG computations.

This submodule provides tools for particle-based methods including:
- Monte Carlo sampling and integration
- MCMC and Hamiltonian Monte Carlo sampling
- Particle-grid interpolation
- Smoothing kernels for GFDM, SPH, KDE

Organization:
- sampling: Monte Carlo integration and variance reduction
- mcmc: MCMC samplers (Metropolis-Hastings, HMC, NUTS, Langevin)
- interpolation: Particle-grid conversion utilities
- kernels: Smoothing kernel functions (Gaussian, Wendland, splines)

Typical Usage:
    from mfg_pde.utils.numerical.particle import (
        # Kernels
        GaussianKernel,
        WendlandKernel,
        CubicSplineKernel,
        create_kernel,
        # Interpolation
        interpolate_grid_to_particles,
        interpolate_particles_to_grid,
        estimate_kde_bandwidth,
        # Monte Carlo
        monte_carlo_integrate,
        MCConfig,
        MCResult,
        # MCMC
        MetropolisHastings,
        HamiltonianMonteCarlo,
        MCMCConfig,
    )
"""

# Kernels (smoothing_kernels.py -> kernels.py)
# Interpolation (particle_interpolation.py -> interpolation.py)
from mfg_pde.utils.numerical.particle.interpolation import (
    estimate_kde_bandwidth,
    interpolate_grid_to_particles,
    interpolate_particles_to_grid,
)
from mfg_pde.utils.numerical.particle.kernels import (
    CubicKernel,
    CubicSplineKernel,
    GaussianKernel,
    Kernel,
    QuarticKernel,
    QuinticSplineKernel,
    WendlandKernel,
    create_kernel,
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
    QuasiMCSampler,
    StratifiedMCSampler,
    UniformMCSampler,
    adaptive_monte_carlo,
    estimate_expectation,
    integrate_gaussian_quadrature_mc,
    monte_carlo_integrate,
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
    # Monte Carlo
    "MCConfig",
    "MCResult",
    "MCSampler",
    "UniformMCSampler",
    "StratifiedMCSampler",
    "QuasiMCSampler",
    "ImportanceMCSampler",
    "ControlVariates",
    "monte_carlo_integrate",
    "adaptive_monte_carlo",
    "integrate_gaussian_quadrature_mc",
    "estimate_expectation",
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
