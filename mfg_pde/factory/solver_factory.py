#!/usr/bin/env python3
"""
MFG Solver Factory

Provides factory patterns for creating optimized solver configurations with
sensible defaults for different use cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from ..alg.amr_enhancement import AMREnhancedSolver
from ..alg.mfg_solvers.adaptive_particle_collocation_solver import AdaptiveParticleCollocationSolver
from ..alg.mfg_solvers.config_aware_fixed_point_iterator import ConfigAwareFixedPointIterator
from ..alg.mfg_solvers.enhanced_particle_collocation_solver import MonitoredParticleCollocationSolver
from ..config.solver_config import (
    FPConfig,
    GFDMConfig,
    HJBConfig,
    MFGSolverConfig,
    NewtonConfig,
    ParticleConfig,
    PicardConfig,
    create_accurate_config,
    create_fast_config,
    create_research_config,
)

if TYPE_CHECKING:
    from mfg_pde.alg.fp_solvers.base_fp import BaseFPSolver
    from mfg_pde.alg.hjb_solvers.base_hjb import BaseHJBSolver
    from mfg_pde.core.mfg_problem import MFGProblem


SolverType = Literal["fixed_point", "particle_collocation", "adaptive_particle", "monitored_particle"]


@dataclass
class SolverFactoryConfig:
    """Configuration for solver factory behavior."""

    solver_type: SolverType = "fixed_point"
    config_preset: str = "balanced"  # fast, accurate, research, balanced
    return_structured: bool = True
    warm_start: bool = False
    custom_config: MFGSolverConfig | None = None
    solver_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.solver_kwargs is None:
            self.solver_kwargs = {}


class SolverFactory:
    """
    Factory for creating MFG solvers with optimized configurations.

    Provides easy creation patterns for different use cases:
    - Fast: Optimized for speed with reasonable accuracy
    - Accurate: High precision configurations
    - Research: Comprehensive monitoring and analysis
    - Custom: User-defined configurations
    """

    @staticmethod
    def create_solver(
        problem: MFGProblem,
        solver_type: SolverType = "fixed_point",
        config_preset: str = "balanced",
        hjb_solver: BaseHJBSolver | None = None,
        fp_solver: BaseFPSolver | None = None,
        collocation_points: np.ndarray | None = None,
        custom_config: MFGSolverConfig | None = None,
        enable_amr: bool = False,
        amr_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> (
        ConfigAwareFixedPointIterator
        | MonitoredParticleCollocationSolver
        | AdaptiveParticleCollocationSolver
        | AMREnhancedSolver
    ):
        """
        Create an MFG solver with optimized configuration.

        Args:
            problem: MFG problem to solve
            solver_type: Type of solver to create
            config_preset: Configuration preset (fast, accurate, research, balanced)
            hjb_solver: HJB solver instance (for fixed_point type)
            fp_solver: FP solver instance (for fixed_point type)
            collocation_points: Spatial points (for particle types)
            custom_config: Custom configuration (overrides preset)
            enable_amr: Enable adaptive mesh refinement enhancement
            amr_config: AMR configuration parameters
            **kwargs: Additional solver-specific parameters

        Returns:
            Configured solver instance (optionally AMR-enhanced)
        """
        # Get configuration
        if custom_config is not None:
            config = custom_config
        else:
            config = SolverFactory._get_config_by_preset(config_preset)

        # Update config with any kwargs
        config = SolverFactory._update_config_with_kwargs(config, **kwargs)

        # Validate problem first
        if problem is None:
            raise ValueError(
                "Problem cannot be None. Please provide a valid MFGProblem instance.\n"
                "Example: problem = ExampleMFGProblem(Nx=50, Nt=100, T=1.0)"
            )

        # Validate solver type with helpful suggestions
        valid_types = [
            "fixed_point",
            "particle_collocation",
            "monitored_particle",
            "adaptive_particle",
        ]
        if solver_type not in valid_types:
            suggestions = "\n".join([f"  • {t}" for t in valid_types])
            raise ValueError(
                f"Unknown solver type: '{solver_type}'\n\n"
                f"Valid solver types are:\n{suggestions}\n\n"
                f"Example: create_fast_solver(problem, solver_type='particle_collocation')"
            )

        # Create base solver based on type
        base_solver: (
            ConfigAwareFixedPointIterator | MonitoredParticleCollocationSolver | AdaptiveParticleCollocationSolver
        )

        if solver_type == "fixed_point":
            base_solver = SolverFactory._create_fixed_point_solver(problem, config, hjb_solver, fp_solver, **kwargs)
        elif solver_type == "particle_collocation":
            base_solver = SolverFactory._create_particle_collocation_solver(
                problem, config, collocation_points, **kwargs
            )
        elif solver_type == "monitored_particle":
            base_solver = SolverFactory._create_monitored_particle_solver(problem, config, collocation_points, **kwargs)
        elif solver_type == "adaptive_particle":
            base_solver = SolverFactory._create_adaptive_particle_solver(problem, config, collocation_points, **kwargs)

        # Enhance with AMR if requested
        if enable_amr:
            from ..alg.amr_enhancement import create_amr_enhanced_solver

            return create_amr_enhanced_solver(
                base_solver=base_solver,
                dimension=getattr(problem, "dimension", None),
                amr_config=amr_config,
            )
        else:
            return base_solver

    @staticmethod
    def _get_config_by_preset(preset: str) -> MFGSolverConfig:
        """Get configuration by preset name."""
        valid_presets = ["fast", "accurate", "research", "balanced"]
        if preset not in valid_presets:
            suggestions = "\n".join([f"  • {p}" for p in valid_presets])
            raise ValueError(
                f"Unknown config preset: '{preset}'\n\n"
                f"Valid presets are:\n{suggestions}\n\n"
                f"Example: create_fast_solver(problem, config_preset='fast')"
            )

        if preset == "fast":
            return create_fast_config()
        elif preset == "accurate":
            return create_accurate_config()
        elif preset == "research":
            return create_research_config()
        elif preset == "balanced":
            # Balanced configuration between speed and accuracy
            return MFGSolverConfig(
                picard=PicardConfig(max_iterations=25, tolerance=1e-4, damping_factor=0.6),
                hjb=HJBConfig(
                    newton=NewtonConfig(max_iterations=25, tolerance=1e-5),
                    gfdm=GFDMConfig(delta=0.3, taylor_order=2, weight_function="wendland"),
                ),
                fp=FPConfig(
                    particle=ParticleConfig(
                        num_particles=3000,
                        kde_bandwidth="scott",
                        normalize_output=True,
                        boundary_handling="absorbing",
                    )
                ),
                return_structured=True,
            )
        else:
            raise ValueError(f"Unknown preset: {preset}. Use 'fast', 'accurate', 'research', or 'balanced'")

    @staticmethod
    def _update_config_with_kwargs(config: MFGSolverConfig, **kwargs: Any) -> MFGSolverConfig:
        """Update configuration with keyword arguments."""
        # Create a copy to avoid modifying original
        import copy

        updated_config = copy.deepcopy(config)

        # Common config updates
        if "max_picard_iterations" in kwargs:
            updated_config.picard.max_iterations = kwargs["max_picard_iterations"]
        if "picard_tolerance" in kwargs:
            updated_config.picard.tolerance = kwargs["picard_tolerance"]
        if "max_newton_iterations" in kwargs:
            updated_config.hjb.newton.max_iterations = kwargs["max_newton_iterations"]
        if "newton_tolerance" in kwargs:
            updated_config.hjb.newton.tolerance = kwargs["newton_tolerance"]
        if "return_structured" in kwargs:
            updated_config.return_structured = kwargs["return_structured"]
        if "warm_start" in kwargs:
            updated_config.warm_start = kwargs["warm_start"]

        # Particle-specific updates
        if "num_particles" in kwargs:
            updated_config.fp.particle.num_particles = kwargs["num_particles"]
        if "delta" in kwargs:
            updated_config.hjb.gfdm.delta = kwargs["delta"]

        return updated_config

    @staticmethod
    def _create_fixed_point_solver(
        problem: MFGProblem,
        config: MFGSolverConfig,
        hjb_solver: BaseHJBSolver | None,
        fp_solver: BaseFPSolver | None,
        **kwargs: Any,
    ) -> ConfigAwareFixedPointIterator:
        """Create a fixed point iterator solver."""
        if hjb_solver is None or fp_solver is None:
            raise ValueError("Fixed point solver requires both hjb_solver and fp_solver")

        # Filter out config-related kwargs that shouldn't be passed to constructor
        config_keys = {
            "max_picard_iterations",
            "picard_tolerance",
            "max_newton_iterations",
            "newton_tolerance",
            "return_structured",
            "warm_start",
            "num_particles",
            "delta",
        }
        constructor_kwargs = {k: v for k, v in kwargs.items() if k not in config_keys}

        return ConfigAwareFixedPointIterator(
            problem=problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            config=config,
            **constructor_kwargs,
        )

    @staticmethod
    def _create_particle_collocation_solver(
        problem: MFGProblem,
        config: MFGSolverConfig,
        collocation_points: np.ndarray | None,
        **kwargs: Any,
    ) -> MonitoredParticleCollocationSolver:
        """Create a particle collocation solver."""
        if collocation_points is None:
            # Create default collocation points
            collocation_points = np.linspace(problem.xmin, problem.xmax, problem.Nx)

        # Extract particle-specific config
        particle_config = config.fp.particle
        gfdm_config = config.hjb.gfdm

        # Filter out config-related kwargs
        config_keys = {
            "max_picard_iterations",
            "picard_tolerance",
            "max_newton_iterations",
            "newton_tolerance",
            "return_structured",
            "warm_start",
            "num_particles",
            "delta",
        }

        solver_kwargs = {
            "num_particles": particle_config.num_particles,
            "delta": gfdm_config.delta,
            "taylor_order": gfdm_config.taylor_order,
            "weight_function": gfdm_config.weight_function,
            "max_newton_iterations": config.hjb.newton.max_iterations,
            "newton_tolerance": config.hjb.newton.tolerance,
            "kde_bandwidth": particle_config.kde_bandwidth,
            "normalize_kde_output": particle_config.normalize_output,
            **{k: v for k, v in kwargs.items() if k not in config_keys},
        }

        return MonitoredParticleCollocationSolver(
            problem=problem, collocation_points=collocation_points, **solver_kwargs
        )

    @staticmethod
    def _create_monitored_particle_solver(
        problem: MFGProblem,
        config: MFGSolverConfig,
        collocation_points: np.ndarray | None,
        **kwargs: Any,
    ) -> MonitoredParticleCollocationSolver:
        """Create a monitored particle collocation solver with enhanced convergence."""
        # Same as particle collocation but with additional monitoring config
        return SolverFactory._create_particle_collocation_solver(problem, config, collocation_points, **kwargs)

    @staticmethod
    def _create_adaptive_particle_solver(
        problem: MFGProblem,
        config: MFGSolverConfig,
        collocation_points: np.ndarray | None,
        **kwargs: Any,
    ) -> AdaptiveParticleCollocationSolver:
        """Create an adaptive particle collocation solver."""
        if collocation_points is None:
            collocation_points = np.linspace(problem.xmin, problem.xmax, problem.Nx)

        particle_config = config.fp.particle
        gfdm_config = config.hjb.gfdm

        # Filter out config-related kwargs
        config_keys = {
            "max_picard_iterations",
            "picard_tolerance",
            "max_newton_iterations",
            "newton_tolerance",
            "return_structured",
            "warm_start",
            "num_particles",
            "delta",
        }

        solver_kwargs = {
            "num_particles": particle_config.num_particles,
            "delta": gfdm_config.delta,
            "taylor_order": gfdm_config.taylor_order,
            "weight_function": gfdm_config.weight_function,
            "max_newton_iterations": config.hjb.newton.max_iterations,
            "newton_tolerance": config.hjb.newton.tolerance,
            "kde_bandwidth": particle_config.kde_bandwidth,
            **{k: v for k, v in kwargs.items() if k not in config_keys},
        }

        return AdaptiveParticleCollocationSolver(
            problem=problem, collocation_points=collocation_points, verbose=False, **solver_kwargs
        )


# Convenience functions for common use cases


def create_solver(
    problem: MFGProblem,
    solver_type: SolverType = "fixed_point",
    preset: str = "balanced",
    **kwargs: Any,
) -> (
    ConfigAwareFixedPointIterator
    | MonitoredParticleCollocationSolver
    | AdaptiveParticleCollocationSolver
    | AMREnhancedSolver
):
    """
    Create an MFG solver with specified type and preset.

    Args:
        problem: MFG problem to solve
        solver_type: Type of solver ("fixed_point", "particle_collocation", "monitored_particle", "adaptive_particle", "amr")
        preset: Configuration preset ("fast", "accurate", "research", "balanced")
        **kwargs: Additional parameters

    Returns:
        Configured solver instance
    """
    return SolverFactory.create_solver(problem=problem, solver_type=solver_type, config_preset=preset, **kwargs)


def create_fast_solver(
    problem: MFGProblem, solver_type: SolverType = "fixed_point", **kwargs: Any
) -> (
    ConfigAwareFixedPointIterator
    | MonitoredParticleCollocationSolver
    | AdaptiveParticleCollocationSolver
    | AMREnhancedSolver
):
    """
    Create a fast MFG solver optimized for speed.

    Args:
        problem: MFG problem to solve
        solver_type: Type of solver
        **kwargs: Additional parameters

    Returns:
        Fast-configured solver instance
    """
    # For fixed_point solvers, create default HJB and FP solvers if not provided
    if solver_type == "fixed_point" and "hjb_solver" not in kwargs and "fp_solver" not in kwargs:
        from mfg_pde.alg.fp_solvers.fp_fdm import FPFDMSolver
        from mfg_pde.alg.hjb_solvers.hjb_fdm import HJBFDMSolver

        # Create stable default solvers using FDM (more stable than particle methods)
        hjb_solver = HJBFDMSolver(problem=problem)
        fp_solver = FPFDMSolver(problem=problem)

        kwargs["hjb_solver"] = hjb_solver
        kwargs["fp_solver"] = fp_solver

    return SolverFactory.create_solver(problem=problem, solver_type=solver_type, config_preset="fast", **kwargs)


def create_semi_lagrangian_solver(
    problem: MFGProblem,
    interpolation_method: str = "linear",
    optimization_method: str = "brent",
    characteristic_solver: str = "explicit_euler",
    use_jax: bool | None = None,
    fp_solver_type: str = "fdm",
    **kwargs: Any,
) -> ConfigAwareFixedPointIterator:
    """
    Create a fixed-point solver with semi-Lagrangian HJB method.

    The semi-Lagrangian method is particularly effective for:
    - Problems with strong convection/transport
    - Discontinuous or non-smooth solutions
    - Large time steps
    - Monotone solution requirements

    Args:
        problem: MFG problem to solve
        interpolation_method: Interpolation for departure points ('linear', 'cubic')
        optimization_method: Hamiltonian optimization ('brent', 'golden')
        characteristic_solver: Characteristic tracing ('explicit_euler', 'rk2')
        use_jax: Enable JAX acceleration (auto-detect if None)
        fp_solver_type: FP solver type ('fdm', 'particle')
        **kwargs: Additional solver configuration

    Returns:
        Fixed-point solver with semi-Lagrangian HJB method

    Example:
        >>> # Create semi-Lagrangian solver for convection-dominated problem
        >>> solver = create_semi_lagrangian_solver(
        ...     problem,
        ...     interpolation_method="cubic",
        ...     optimization_method="brent",
        ...     use_jax=True
        ... )
        >>> result = solver.solve()
    """
    from mfg_pde.alg.hjb_solvers.hjb_semi_lagrangian import HJBSemiLagrangianSolver

    # Create semi-Lagrangian HJB solver
    hjb_solver = HJBSemiLagrangianSolver(
        problem=problem,
        interpolation_method=interpolation_method,
        optimization_method=optimization_method,
        characteristic_solver=characteristic_solver,
        use_jax=use_jax,
        **{k: v for k, v in kwargs.items() if k in ["tolerance", "max_char_iterations"]},
    )

    # Create appropriate FP solver
    from mfg_pde.alg.fp_solvers.fp_fdm import FPFDMSolver
    from mfg_pde.alg.fp_solvers.fp_particle import FPParticleSolver

    fp_solver: FPFDMSolver | FPParticleSolver
    if fp_solver_type == "fdm":
        fp_solver = FPFDMSolver(problem=problem)
    elif fp_solver_type == "particle":
        fp_solver = FPParticleSolver(problem=problem)
    else:
        raise ValueError(f"Unknown FP solver type: {fp_solver_type}")

    # Extract relevant kwargs for fixed-point solver
    fp_kwargs = {k: v for k, v in kwargs.items() if k not in ["tolerance", "max_char_iterations"]}

    solver = create_fast_solver(
        problem=problem,
        solver_type="fixed_point",
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
        **fp_kwargs,
    )
    # Type assertion since we know this returns ConfigAwareFixedPointIterator for fixed_point solver_type
    return cast(ConfigAwareFixedPointIterator, solver)


def create_accurate_solver(
    problem: MFGProblem, solver_type: SolverType = "fixed_point", **kwargs: Any
) -> (
    ConfigAwareFixedPointIterator
    | MonitoredParticleCollocationSolver
    | AdaptiveParticleCollocationSolver
    | AMREnhancedSolver
):
    """
    Create an accurate MFG solver optimized for precision.

    Args:
        problem: MFG problem to solve
        solver_type: Type of solver
        **kwargs: Additional parameters

    Returns:
        Accurate-configured solver instance
    """
    return SolverFactory.create_solver(problem=problem, solver_type=solver_type, config_preset="accurate", **kwargs)


def create_research_solver(
    problem: MFGProblem, solver_type: SolverType = "monitored_particle", **kwargs: Any
) -> (
    ConfigAwareFixedPointIterator
    | MonitoredParticleCollocationSolver
    | AdaptiveParticleCollocationSolver
    | AMREnhancedSolver
):
    """
    Create a research MFG solver with comprehensive monitoring.

    Args:
        problem: MFG problem to solve
        solver_type: Type of solver (defaults to monitored_particle for research)
        **kwargs: Additional parameters

    Returns:
        Research-configured solver instance
    """
    return SolverFactory.create_solver(problem=problem, solver_type=solver_type, config_preset="research", **kwargs)


def create_monitored_solver(
    problem: MFGProblem, collocation_points: np.ndarray | None = None, **kwargs: Any
) -> MonitoredParticleCollocationSolver:
    """
    Create a monitored particle collocation solver with enhanced convergence analysis.

    Args:
        problem: MFG problem to solve
        collocation_points: Spatial collocation points
        **kwargs: Additional parameters

    Returns:
        Monitored particle collocation solver
    """
    solver = SolverFactory.create_solver(
        problem=problem,
        solver_type="monitored_particle",
        config_preset="research",
        collocation_points=collocation_points,
        **kwargs,
    )
    # Type assertion since we know this returns MonitoredParticleCollocationSolver for monitored_particle solver_type
    return cast(MonitoredParticleCollocationSolver, solver)


def create_amr_solver(
    problem: MFGProblem,
    base_solver_type: SolverType = "fixed_point",
    error_threshold: float = 1e-4,
    max_levels: int = 5,
    **kwargs: Any,
) -> AMREnhancedSolver:
    """
    Create an AMR-enhanced MFG solver.

    This function creates any base solver and enhances it with adaptive
    mesh refinement capabilities. AMR is a mesh adaptation technique
    that can improve any underlying solution method.

    Args:
        problem: MFG problem to solve
        base_solver_type: Base solver type to enhance with AMR
        error_threshold: Error threshold for mesh refinement
        max_levels: Maximum refinement levels
        **kwargs: Additional parameters for base solver and AMR

    Returns:
        AMR-enhanced solver wrapping the base solver

    Example:
        >>> # Create FDM solver with AMR enhancement
        >>> amr_solver = create_amr_solver(
        ...     problem,
        ...     base_solver_type="fixed_point",
        ...     error_threshold=1e-5,
        ...     max_levels=6
        ... )
        >>> result = amr_solver.solve()
    """
    # Prepare AMR configuration
    amr_config = {
        "error_threshold": error_threshold,
        "max_levels": max_levels,
    }

    # Extract AMR-specific kwargs
    amr_keys = {"initial_intervals", "adaptation_frequency", "max_adaptations"}
    for key in amr_keys:
        if key in kwargs:
            amr_config[key] = kwargs.pop(key)

    solver = SolverFactory.create_solver(
        problem=problem,
        solver_type=base_solver_type,
        config_preset="accurate",  # AMR typically used for high-accuracy solutions
        enable_amr=True,
        amr_config=amr_config,
        **kwargs,
    )
    # Type assertion since we know this returns AMREnhancedSolver when enable_amr=True
    return cast(AMREnhancedSolver, solver)
