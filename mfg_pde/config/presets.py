"""
Preset solver configurations for common scenarios.

This module provides ready-to-use solver configurations optimized for
different use cases and problem types. Presets configure HOW to solve
problems (solver methods, tolerances, backends), not WHAT problems to solve.

Usage
-----
>>> from mfg_pde.config import presets
>>> config = presets.fast_solver()  # Speed-optimized
>>> config = presets.accurate_solver()  # Accuracy-optimized
>>> config = presets.crowd_dynamics_solver()  # Domain-specific

Note: Problems are still defined via MFGProblem instances.
"""

from __future__ import annotations

from typing import Literal

from .builder import ConfigBuilder
from .core import SolverConfig  # noqa: TC001

# =============================================================================
# General-Purpose Presets
# =============================================================================


def fast_solver() -> SolverConfig:
    """
    Fast solver configuration (speed-optimized).

    Optimized for rapid prototyping and testing:
    - Low accuracy order (faster computation)
    - No Anderson acceleration (simpler iteration)
    - Minimal logging overhead

    Use when: Quick results needed, accuracy is secondary

    Returns
    -------
    SolverConfig
        Speed-optimized configuration

    Examples
    --------
    >>> config = presets.fast_solver()
    >>> result = solve_mfg(problem, config=config)
    """
    return (
        ConfigBuilder()
        .solver_hjb(method="fdm", accuracy_order=1)
        .solver_fp(method="fdm")
        .picard(max_iterations=30, tolerance=1e-4, anderson_memory=0, verbose=False)
        .backend(backend_type="numpy")
        .logging(level="WARNING", progress_bar=False)
        .build()
    )


def accurate_solver() -> SolverConfig:
    """
    Accurate solver configuration (accuracy-optimized).

    Optimized for high-quality results:
    - High accuracy order (better approximation)
    - Anderson acceleration (faster convergence)
    - Tight tolerance
    - Double precision

    Use when: Publication-quality results needed

    Returns
    -------
    SolverConfig
        Accuracy-optimized configuration

    Examples
    --------
    >>> config = presets.accurate_solver()
    >>> result = solve_mfg(problem, config=config)
    """
    return (
        ConfigBuilder()
        .solver_hjb(method="fdm", accuracy_order=3)
        .solver_fp(method="fdm")
        .picard(max_iterations=100, tolerance=1e-8, anderson_memory=5, verbose=True)
        .backend(backend_type="numpy", precision="float64")
        .logging(level="INFO", progress_bar=True)
        .build()
    )


def research_solver() -> SolverConfig:
    """
    Research solver configuration (comprehensive output).

    Optimized for research and debugging:
    - Balanced accuracy
    - Full logging and diagnostics
    - Save intermediate results
    - Verbose iteration output

    Use when: Analyzing solver behavior, debugging, research

    Returns
    -------
    SolverConfig
        Research-oriented configuration

    Examples
    --------
    >>> config = presets.research_solver(output_dir="results/debug")
    >>> result = solve_mfg(problem, config=config)
    """
    return (
        ConfigBuilder()
        .solver_hjb(method="fdm", accuracy_order=2)
        .solver_fp(method="fdm")
        .picard(max_iterations=100, tolerance=1e-6, anderson_memory=5, verbose=True)
        .backend(backend_type="numpy")
        .logging(level="DEBUG", progress_bar=True, save_intermediate=False)
        .build()
    )


def production_solver() -> SolverConfig:
    """
    Production solver configuration (reliability-optimized).

    Optimized for production deployments:
    - Moderate accuracy (good balance)
    - Robust settings
    - Clean logging
    - No intermediate saves

    Use when: Deploying to production, batch processing

    Returns
    -------
    SolverConfig
        Production-ready configuration
    """
    return (
        ConfigBuilder()
        .solver_hjb(method="fdm", accuracy_order=2)
        .solver_fp(method="fdm")
        .picard(max_iterations=50, tolerance=1e-6, anderson_memory=3, verbose=False)
        .backend(backend_type="numpy", precision="float64")
        .logging(level="INFO", progress_bar=False, save_intermediate=False)
        .build()
    )


# =============================================================================
# Domain-Specific Presets
# =============================================================================


def crowd_dynamics_solver(accuracy: Literal["low", "medium", "high"] = "medium") -> SolverConfig:
    """
    Recommended solver configuration for crowd dynamics problems.

    Crowd dynamics typically involve:
    - Moderate dimensions (2D-3D)
    - Smooth solutions
    - Particle methods for density evolution

    Parameters
    ----------
    accuracy : Literal["low", "medium", "high"]
        Accuracy level (default: medium)

    Returns
    -------
    SolverConfig
        Crowd dynamics-optimized configuration

    Examples
    --------
    >>> problem = MyCrowdProblem(dimension=2, ...)
    >>> config = presets.crowd_dynamics_solver(accuracy="high")
    >>> result = solve_mfg(problem, config=config)
    """
    accuracy_map = {"low": 1, "medium": 2, "high": 3}

    return (
        ConfigBuilder()
        .solver_hjb(method="fdm", accuracy_order=accuracy_map[accuracy])
        .solver_fp_particle(num_particles=5000, normalization="initial_only")
        .picard(max_iterations=50, tolerance=1e-6, anderson_memory=5)
        .backend(backend_type="numpy")
        .logging(level="INFO", progress_bar=True)
        .build()
    )


def traffic_flow_solver(use_semi_lagrangian: bool = True) -> SolverConfig:
    """
    Recommended solver configuration for traffic flow problems.

    Traffic flow problems typically involve:
    - Strong advection (vehicles moving)
    - 1D-2D networks
    - Semi-Lagrangian methods handle advection well

    Parameters
    ----------
    use_semi_lagrangian : bool
        Use Semi-Lagrangian method for HJB (default: True)
        If False, uses FDM

    Returns
    -------
    SolverConfig
        Traffic flow-optimized configuration

    Examples
    --------
    >>> problem = MyTrafficProblem(...)
    >>> config = presets.traffic_flow_solver()
    >>> result = solve_mfg(problem, config=config)
    """
    builder = ConfigBuilder()

    if use_semi_lagrangian:
        builder.solver_hjb_semi_lagrangian(interpolation_method="cubic", rk_order=2)
    else:
        builder.solver_hjb(method="fdm", accuracy_order=2)

    return (
        builder.solver_fp(method="fdm")
        .picard(max_iterations=40, tolerance=1e-6, anderson_memory=3)
        .backend(backend_type="numpy")
        .logging(level="INFO", progress_bar=True)
        .build()
    )


def epidemic_solver() -> SolverConfig:
    """
    Recommended solver configuration for epidemic models.

    Epidemic models typically involve:
    - Network/graph structure
    - Stochastic elements
    - Moderate time horizons

    Returns
    -------
    SolverConfig
        Epidemic model-optimized configuration

    Examples
    --------
    >>> problem = MyEpidemicProblem(...)
    >>> config = presets.epidemic_solver()
    >>> result = solve_mfg(problem, config=config)
    """
    return (
        ConfigBuilder()
        .solver_hjb(method="fdm", accuracy_order=2)
        .solver_fp(method="fdm")
        .picard(max_iterations=60, tolerance=1e-7, anderson_memory=5, verbose=True)
        .backend(backend_type="numpy")
        .logging(level="INFO", progress_bar=True)
        .build()
    )


def financial_solver() -> SolverConfig:
    """
    Recommended solver configuration for financial applications.

    Financial problems typically require:
    - High accuracy (money at stake!)
    - Double precision
    - Robust convergence

    Returns
    -------
    SolverConfig
        Financial applications-optimized configuration

    Examples
    --------
    >>> problem = MyFinancialProblem(...)
    >>> config = presets.financial_solver()
    >>> result = solve_mfg(problem, config=config)
    """
    return (
        ConfigBuilder()
        .solver_hjb(method="fdm", accuracy_order=3)
        .solver_fp(method="fdm")
        .picard(max_iterations=100, tolerance=1e-8, anderson_memory=5, verbose=True)
        .backend(backend_type="numpy", precision="float64")
        .logging(level="INFO", progress_bar=True, save_intermediate=False)
        .build()
    )


# =============================================================================
# Computational Presets
# =============================================================================


def high_dimensional_solver(use_gpu: bool = True) -> SolverConfig:
    """
    Recommended solver configuration for high-dimensional problems (d > 3).

    High-dimensional problems require:
    - Meshfree methods (GFDM for HJB)
    - Many particles (avoid curse of dimensionality)
    - GPU acceleration (if available)

    Parameters
    ----------
    use_gpu : bool
        Use GPU acceleration via JAX (default: True)
        Falls back to CPU if GPU not available

    Returns
    -------
    SolverConfig
        High-dimensional-optimized configuration

    Examples
    --------
    >>> problem = MyHighDimProblem(dimension=10, ...)
    >>> config = presets.high_dimensional_solver()
    >>> result = solve_mfg(problem, config=config)

    Note
    ----
    Uses hybrid mode (sample particles, output to grid via KDE).
    For meshfree collocation output, set external_particles programmatically:
    >>> import numpy as np
    >>> config.fp.particle_config.mode = "collocation"
    >>> config.fp.particle_config.external_particles = np.random.uniform(...)
    """
    device = "gpu" if use_gpu else "cpu"
    backend = "jax" if use_gpu else "numpy"

    return (
        ConfigBuilder()
        .solver_hjb_gfdm(delta=0.1, stencil_size=25, qp_optimization_level="auto")
        .solver_fp_particle(num_particles=10000, mode="hybrid")
        .picard(max_iterations=50, tolerance=1e-6, anderson_memory=5)
        .backend(backend_type=backend, device=device)
        .logging(level="INFO", progress_bar=True)
        .build()
    )


def large_scale_solver() -> SolverConfig:
    """
    Recommended solver configuration for large-scale problems.

    Large-scale problems benefit from:
    - Lower accuracy for faster iteration
    - Anderson acceleration
    - Reduced logging overhead

    Returns
    -------
    SolverConfig
        Large-scale-optimized configuration

    Examples
    --------
    >>> problem = MyLargeScaleProblem(...)
    >>> config = presets.large_scale_solver()
    >>> result = solve_mfg(problem, config=config)
    """
    return (
        ConfigBuilder()
        .solver_hjb(method="fdm", accuracy_order=1)
        .solver_fp_particle(num_particles=3000)
        .picard(max_iterations=30, tolerance=1e-5, anderson_memory=3, verbose=False)
        .backend(backend_type="numpy")
        .logging(level="WARNING", progress_bar=True, save_intermediate=False)
        .build()
    )


def educational_solver() -> SolverConfig:
    """
    Recommended solver configuration for educational/teaching use.

    Educational use benefits from:
    - Clear, verbose output
    - Moderate speed (not too slow)
    - Progress visibility

    Returns
    -------
    SolverConfig
        Education-optimized configuration

    Examples
    --------
    >>> problem = MyTutorialProblem(...)
    >>> config = presets.educational_solver()
    >>> result = solve_mfg(problem, config=config)
    """
    return (
        ConfigBuilder()
        .solver_hjb(method="fdm", accuracy_order=2)
        .solver_fp(method="fdm")
        .picard(max_iterations=50, tolerance=1e-6, anderson_memory=3, verbose=True)
        .backend(backend_type="numpy")
        .logging(level="INFO", progress_bar=True)
        .build()
    )


# =============================================================================
# Convenience Aliases (for backward compatibility)
# =============================================================================


def default_solver() -> SolverConfig:
    """
    Default solver configuration.

    Alias for fast_solver() - reasonable defaults for general use.

    Returns
    -------
    SolverConfig
        Default configuration
    """
    return fast_solver()


# Export all presets
__all__ = [
    # General-purpose
    "fast_solver",
    "accurate_solver",
    "research_solver",
    "production_solver",
    "default_solver",
    # Domain-specific
    "crowd_dynamics_solver",
    "traffic_flow_solver",
    "epidemic_solver",
    "financial_solver",
    # Computational
    "high_dimensional_solver",
    "large_scale_solver",
    "educational_solver",
]
