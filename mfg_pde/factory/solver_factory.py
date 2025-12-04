#!/usr/bin/env python3
"""
MFG Solver Factory

Provides factory patterns for creating optimized solver configurations with
sensible defaults for different use cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

from mfg_pde.alg.numerical.coupling import FixedPointIterator
from mfg_pde.config.pydantic_config import (
    FPConfig,
    GFDMConfig,
    HJBConfig,
    MFGSolverConfig,
    NewtonConfig,
    ParticleConfig,
    PicardConfig,
)

if TYPE_CHECKING:
    from mfg_pde.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfg_pde.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver

    # AMR Enhancement moved to experimental features
    from mfg_pde.core.mfg_problem import MFGProblem


SolverType = Literal["fixed_point", "monitored_particle", "adaptive_particle"]
# Note: "particle_collocation" has been removed from core package


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
        collocation_points: NDArray[np.floating] | None = None,
        custom_config: MFGSolverConfig | None = None,
        enable_amr: bool = False,
        amr_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> FixedPointIterator:
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
                "Example: problem = MFGProblem(Nx=50, Nt=100, T=1.0)"
            )

        # Validate solver type with helpful suggestions
        valid_types = ["fixed_point"]
        if solver_type not in valid_types:
            suggestions = "\n".join([f"  • {t}" for t in valid_types])
            raise ValueError(
                f"Unknown solver type: '{solver_type}'\n\n"
                f"Valid solver types are:\n{suggestions}\n\n"
                f"Note: 'particle_collocation' has been removed from core package.\n"
                f"Example: create_fast_solver(problem, solver_type='fixed_point')"
            )

        # Create base solver based on type
        if solver_type == "fixed_point":
            base_solver = SolverFactory._create_fixed_point_solver(problem, config, hjb_solver, fp_solver, **kwargs)
        else:
            raise ValueError(
                f"Unsupported solver type: {solver_type}. "
                "Note: 'particle_collocation' has been removed from core package. "
                f"Available types: ['fixed_point']"
            )

        # Enhance with AMR if requested
        if enable_amr:
            # AMR enhancement is currently experimental
            import warnings

            warnings.warn(
                "AMR enhancement is currently experimental and not available in the new paradigm structure. "
                "Using base solver without AMR.",
                UserWarning,
            )

        # Note: AMR enhancement not available in new paradigm
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
            return MFGSolverConfig(
                convergence_tolerance=1e-3,
                strict_convergence_errors=False,
            )
        elif preset == "accurate":
            return MFGSolverConfig(
                convergence_tolerance=1e-7,
                strict_convergence_errors=True,
            )
        elif preset == "research":
            return MFGSolverConfig(
                convergence_tolerance=1e-8,
                enable_warm_start=True,
                strict_convergence_errors=True,
            )
        elif preset == "balanced":
            # Balanced configuration between speed and accuracy
            return MFGSolverConfig(
                picard=PicardConfig(max_iterations=25, tolerance=1e-4, damping_factor=0.6),
                hjb=HJBConfig(
                    newton=NewtonConfig(max_iterations=25, tolerance=1e-5),
                    gfdm=GFDMConfig(delta=0.3, taylor_order=2, weight_function="gaussian"),
                ),
                fp=FPConfig(
                    particle=ParticleConfig(
                        num_particles=3000,
                        kde_bandwidth=0.05,
                        boundary_treatment="absorption",
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
    ) -> FixedPointIterator:
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

        return FixedPointIterator(
            problem=problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            config=config,
            **constructor_kwargs,
        )

    # Note: _create_particle_collocation_solver, _create_monitored_particle_solver,
    # and _create_adaptive_particle_solver methods have been removed.
    # Particle-collocation methods have been removed from core package.


# Convenience functions for common use cases


def create_solver(
    problem: MFGProblem,
    solver_type: SolverType = "fixed_point",
    preset: str = "balanced",
    hjb_solver: BaseHJBSolver | None = None,
    fp_solver: BaseFPSolver | None = None,
    **kwargs: Any,
) -> FixedPointIterator:
    """
    Create an MFG solver with specified type and preset.

    This is the main entry point for creating solvers. For most use cases,
    use problem.solve() directly instead.

    Args:
        problem: MFG problem to solve
        solver_type: Type of solver ("fixed_point")
        preset: Configuration preset ("fast", "accurate", "research", "balanced")
        hjb_solver: Optional HJB solver instance
        fp_solver: Optional FP solver instance
        **kwargs: Additional parameters passed to the solver

    Returns:
        Configured solver instance

    Example:
        >>> from mfg_pde import MFGProblem, create_solver
        >>> problem = MFGProblem(Nx=50, Nt=20, T=1.0)
        >>> solver = create_solver(problem, preset="balanced")
        >>> result = solver.solve()

    Note:
        For simple cases, prefer problem.solve() which handles solver creation internally.
    """
    return SolverFactory.create_solver(
        problem=problem,
        solver_type=solver_type,
        config_preset=preset,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
        **kwargs,
    )


# =============================================================================
# REMOVED CONVENIENCE FUNCTIONS (v0.15.0)
# =============================================================================
# The following functions were removed to simplify the API:
# - create_basic_solver() - Use create_solver(preset="fast") or problem.solve()
# - create_standard_solver() - Use create_solver(preset="balanced") or problem.solve()
# - create_fast_solver() - Use create_solver(preset="fast") or problem.solve()
# - create_accurate_solver() - Use create_solver(preset="accurate") or problem.solve()
# - create_research_solver() - Use create_solver(preset="research") or problem.solve()
# - create_semi_lagrangian_solver() - Instantiate HJBSemiLagrangianSolver directly
# - create_amr_solver() - AMR moved to experimental features
#
# Migration: Use problem.solve() or create_solver() with appropriate preset.
# =============================================================================


# Legacy alias for backward compatibility - will be removed in v1.0.0
def _removed_function_error(name: str) -> None:
    """Raise informative error for removed functions."""
    raise NotImplementedError(
        f"{name}() has been removed. Use create_solver(preset=...) or problem.solve() instead. "
        f"See migration guide: docs/migration/PHASE_3_2_CONFIG_MIGRATION.md"
    )


def create_basic_solver(*args: Any, **kwargs: Any) -> Any:
    """Removed: Use create_solver(preset='fast') or problem.solve()."""
    _removed_function_error("create_basic_solver")


def create_standard_solver(*args: Any, **kwargs: Any) -> Any:
    """Removed: Use create_solver(preset='balanced') or problem.solve()."""
    _removed_function_error("create_standard_solver")


def create_fast_solver(*args: Any, **kwargs: Any) -> Any:
    """Removed: Use create_solver(preset='fast') or problem.solve()."""
    _removed_function_error("create_fast_solver")


def create_accurate_solver(*args: Any, **kwargs: Any) -> Any:
    """Removed: Use create_solver(preset='accurate') or problem.solve()."""
    _removed_function_error("create_accurate_solver")


def create_research_solver(*args: Any, **kwargs: Any) -> Any:
    """Removed: Use create_solver(preset='research') or problem.solve()."""
    _removed_function_error("create_research_solver")


def create_semi_lagrangian_solver(*args: Any, **kwargs: Any) -> Any:
    """Removed: Instantiate HJBSemiLagrangianSolver directly."""
    _removed_function_error("create_semi_lagrangian_solver")


def create_amr_solver(*args: Any, **kwargs: Any) -> Any:
    """Removed: AMR moved to experimental features."""
    _removed_function_error("create_amr_solver")
