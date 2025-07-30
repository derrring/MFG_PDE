#!/usr/bin/env python3
"""
MFG Solver Factory

Provides factory patterns for creating optimized solver configurations with
sensible defaults for different use cases.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, TYPE_CHECKING, Union

import numpy as np

from ..alg.mfg_solvers.adaptive_particle_collocation_solver import (
    SilentAdaptiveParticleCollocationSolver,
)
from ..alg.mfg_solvers.config_aware_fixed_point_iterator import ConfigAwareFixedPointIterator
from ..alg.mfg_solvers.enhanced_particle_collocation_solver import (
    MonitoredParticleCollocationSolver,
)
from ..config.solver_config import (
    create_accurate_config,
    create_fast_config,
    create_research_config,
    FPConfig,
    GFDMConfig,
    HJBConfig,
    MFGSolverConfig,
    NewtonConfig,
    ParticleConfig,
    PicardConfig,
)

if TYPE_CHECKING:
    from mfg_pde.alg.fp_solvers.base_fp import BaseFPSolver
    from mfg_pde.alg.hjb_solvers.base_hjb import BaseHJBSolver
    from mfg_pde.core.mfg_problem import MFGProblem


SolverType = Literal[
    "fixed_point", "particle_collocation", "adaptive_particle", "monitored_particle"
]


@dataclass
class SolverFactoryConfig:
    """Configuration for solver factory behavior."""

    solver_type: SolverType = "fixed_point"
    config_preset: str = "balanced"  # fast, accurate, research, balanced
    return_structured: bool = True
    warm_start: bool = False
    custom_config: Optional[MFGSolverConfig] = None
    solver_kwargs: Dict[str, Any] = None

    def __post_init__(self):
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
        problem: "MFGProblem",
        solver_type: SolverType = "fixed_point",
        config_preset: str = "balanced",
        hjb_solver: Optional["BaseHJBSolver"] = None,
        fp_solver: Optional["BaseFPSolver"] = None,
        collocation_points: Optional[np.ndarray] = None,
        custom_config: Optional[MFGSolverConfig] = None,
        **kwargs,
    ) -> Union[
        ConfigAwareFixedPointIterator,
        MonitoredParticleCollocationSolver,
        SilentAdaptiveParticleCollocationSolver,
    ]:
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
            **kwargs: Additional solver-specific parameters

        Returns:
            Configured solver instance
        """
        # Get configuration
        if custom_config is not None:
            config = custom_config
        else:
            config = SolverFactory._get_config_by_preset(config_preset)

        # Update config with any kwargs
        config = SolverFactory._update_config_with_kwargs(config, **kwargs)

        # Create solver based on type
        if solver_type == "fixed_point":
            return SolverFactory._create_fixed_point_solver(
                problem, config, hjb_solver, fp_solver, **kwargs
            )
        elif solver_type == "particle_collocation":
            return SolverFactory._create_particle_collocation_solver(
                problem, config, collocation_points, **kwargs
            )
        elif solver_type == "monitored_particle":
            return SolverFactory._create_monitored_particle_solver(
                problem, config, collocation_points, **kwargs
            )
        elif solver_type == "adaptive_particle":
            return SolverFactory._create_adaptive_particle_solver(
                problem, config, collocation_points, **kwargs
            )
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

    @staticmethod
    def _get_config_by_preset(preset: str) -> MFGSolverConfig:
        """Get configuration by preset name."""
        if preset == "fast":
            return create_fast_config()
        elif preset == "accurate":
            return create_accurate_config()
        elif preset == "research":
            return create_research_config()
        elif preset == "balanced":
            # Balanced configuration between speed and accuracy
            return MFGSolverConfig(
                picard=PicardConfig(
                    max_iterations=25, tolerance=1e-4, damping_factor=0.6
                ),
                hjb=HJBConfig(
                    newton=NewtonConfig(max_iterations=25, tolerance=1e-5),
                    gfdm=GFDMConfig(
                        delta=0.3, taylor_order=2, weight_function="wendland"
                    ),
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
            raise ValueError(
                f"Unknown preset: {preset}. Use 'fast', 'accurate', 'research', or 'balanced'"
            )

    @staticmethod
    def _update_config_with_kwargs(
        config: MFGSolverConfig, **kwargs
    ) -> MFGSolverConfig:
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
        problem: "MFGProblem",
        config: MFGSolverConfig,
        hjb_solver: Optional["BaseHJBSolver"],
        fp_solver: Optional["BaseFPSolver"],
        **kwargs,
    ) -> ConfigAwareFixedPointIterator:
        """Create a fixed point iterator solver."""
        if hjb_solver is None or fp_solver is None:
            raise ValueError(
                "Fixed point solver requires both hjb_solver and fp_solver"
            )

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
        problem: "MFGProblem",
        config: MFGSolverConfig,
        collocation_points: Optional[np.ndarray],
        **kwargs,
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
        problem: "MFGProblem",
        config: MFGSolverConfig,
        collocation_points: Optional[np.ndarray],
        **kwargs,
    ) -> MonitoredParticleCollocationSolver:
        """Create a monitored particle collocation solver with enhanced convergence."""
        # Same as particle collocation but with additional monitoring config
        return SolverFactory._create_particle_collocation_solver(
            problem, config, collocation_points, **kwargs
        )

    @staticmethod
    def _create_adaptive_particle_solver(
        problem: "MFGProblem",
        config: MFGSolverConfig,
        collocation_points: Optional[np.ndarray],
        **kwargs,
    ) -> SilentAdaptiveParticleCollocationSolver:
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

        return SilentAdaptiveParticleCollocationSolver(
            problem=problem, collocation_points=collocation_points, **solver_kwargs
        )


# Convenience functions for common use cases


def create_solver(
    problem: "MFGProblem",
    solver_type: SolverType = "fixed_point",
    preset: str = "balanced",
    **kwargs,
) -> Union[
    ConfigAwareFixedPointIterator,
    MonitoredParticleCollocationSolver,
    SilentAdaptiveParticleCollocationSolver,
]:
    """
    Create an MFG solver with specified type and preset.

    Args:
        problem: MFG problem to solve
        solver_type: Type of solver ("fixed_point", "particle_collocation", "monitored_particle", "adaptive_particle")
        preset: Configuration preset ("fast", "accurate", "research", "balanced")
        **kwargs: Additional parameters

    Returns:
        Configured solver instance
    """
    return SolverFactory.create_solver(
        problem=problem, solver_type=solver_type, config_preset=preset, **kwargs
    )


def create_fast_solver(
    problem: "MFGProblem", solver_type: SolverType = "fixed_point", **kwargs
) -> Union[
    ConfigAwareFixedPointIterator,
    MonitoredParticleCollocationSolver,
    SilentAdaptiveParticleCollocationSolver,
]:
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
    if (
        solver_type == "fixed_point"
        and "hjb_solver" not in kwargs
        and "fp_solver" not in kwargs
    ):
        import numpy as np

        from mfg_pde.alg.fp_solvers.fp_fdm import FPFDMSolver
        from mfg_pde.alg.hjb_solvers.hjb_fdm import HJBFDMSolver

        # Create stable default solvers using FDM (more stable than particle methods)
        hjb_solver = HJBFDMSolver(problem=problem)
        fp_solver = FPFDMSolver(problem=problem)

        kwargs["hjb_solver"] = hjb_solver
        kwargs["fp_solver"] = fp_solver

    return SolverFactory.create_solver(
        problem=problem, solver_type=solver_type, config_preset="fast", **kwargs
    )


def create_accurate_solver(
    problem: "MFGProblem", solver_type: SolverType = "fixed_point", **kwargs
) -> Union[
    ConfigAwareFixedPointIterator,
    MonitoredParticleCollocationSolver,
    SilentAdaptiveParticleCollocationSolver,
]:
    """
    Create an accurate MFG solver optimized for precision.

    Args:
        problem: MFG problem to solve
        solver_type: Type of solver
        **kwargs: Additional parameters

    Returns:
        Accurate-configured solver instance
    """
    return SolverFactory.create_solver(
        problem=problem, solver_type=solver_type, config_preset="accurate", **kwargs
    )


def create_research_solver(
    problem: "MFGProblem", solver_type: SolverType = "monitored_particle", **kwargs
) -> Union[
    ConfigAwareFixedPointIterator,
    MonitoredParticleCollocationSolver,
    SilentAdaptiveParticleCollocationSolver,
]:
    """
    Create a research MFG solver with comprehensive monitoring.

    Args:
        problem: MFG problem to solve
        solver_type: Type of solver (defaults to monitored_particle for research)
        **kwargs: Additional parameters

    Returns:
        Research-configured solver instance
    """
    return SolverFactory.create_solver(
        problem=problem, solver_type=solver_type, config_preset="research", **kwargs
    )


def create_monitored_solver(
    problem: "MFGProblem", collocation_points: Optional[np.ndarray] = None, **kwargs
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
    return SolverFactory.create_solver(
        problem=problem,
        solver_type="monitored_particle",
        config_preset="research",
        collocation_points=collocation_points,
        **kwargs,
    )
