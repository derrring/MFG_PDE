#!/usr/bin/env python3
"""
MFG Solver Factory

Provides factory patterns for creating optimized solver configurations with
sensible defaults for different use cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

from mfg_pde.alg.numerical.mfg_solvers import FixedPointIterator
from mfg_pde.config.solver_config import (
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
    from mfg_pde.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfg_pde.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver

    # AMR Enhancement moved to experimental features
    from mfg_pde.core.mfg_problem import MFGProblem


SolverType = Literal["fixed_point", "monitored_particle", "adaptive_particle"]
# Note: "particle_collocation" has been moved to mfg-research repository


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
                "Example: problem = ExampleMFGProblem(Nx=50, Nt=100, T=1.0)"
            )

        # Validate solver type with helpful suggestions
        valid_types = ["fixed_point"]
        if solver_type not in valid_types:
            suggestions = "\n".join([f"  • {t}" for t in valid_types])
            raise ValueError(
                f"Unknown solver type: '{solver_type}'\n\n"
                f"Valid solver types are:\n{suggestions}\n\n"
                f"Note: 'particle_collocation' has been moved to mfg-research repository.\n"
                f"Example: create_fast_solver(problem, solver_type='fixed_point')"
            )

        # Create base solver based on type
        if solver_type == "fixed_point":
            base_solver = SolverFactory._create_fixed_point_solver(problem, config, hjb_solver, fp_solver, **kwargs)
        else:
            raise ValueError(
                f"Unsupported solver type: {solver_type}. "
                "Note: 'particle_collocation' has been moved to mfg-research repository. "
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
    # Particle-collocation methods have been moved to mfg-research repository.


# Convenience functions for common use cases


def create_solver(
    problem: MFGProblem,
    solver_type: SolverType = "fixed_point",
    preset: str = "balanced",
    **kwargs: Any,
) -> FixedPointIterator:
    """
    Create an MFG solver with specified type and preset.

    Args:
        problem: MFG problem to solve
        solver_type: Type of solver ("fixed_point")
        preset: Configuration preset ("fast", "accurate", "research", "balanced")
        **kwargs: Additional parameters

    Returns:
        Configured solver instance

    Note:
        Particle-collocation methods have been moved to mfg-research repository.
    """
    return SolverFactory.create_solver(problem=problem, solver_type=solver_type, config_preset=preset, **kwargs)


def create_basic_solver(
    problem: MFGProblem, damping: float = 0.6, max_iterations: int = 100, tolerance: float = 1e-5, **kwargs: Any
) -> Any:
    """
    Create basic FDM benchmark solver (Tier 1).

    Uses HJB-FDM + FP-FDM with upwind + damped fixed point.
    Fast but approximate (1-10% mass error) - primarily for benchmarking.

    This is the simplest MFG solver, useful for:
    - Benchmarking and validating advanced methods
    - Quick testing and prototyping
    - Educational purposes
    - Comparison baseline

    Args:
        problem: MFG problem to solve
        damping: Damping factor for fixed point (0.5-0.7 recommended, default 0.6)
        max_iterations: Maximum Picard iterations (default 100)
        tolerance: Convergence tolerance (default 1e-5)
        **kwargs: Additional parameters

    Returns:
        Basic FDM solver instance

    Note:
        Mass conservation is approximate (~1-10% error).
        For production, use create_fast_solver() (Tier 2: Hybrid with particles).

    Example:
        >>> solver = create_basic_solver(problem, damping=0.6)
        >>> result = solver.solve()
    """
    from mfg_pde.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
    from mfg_pde.alg.numerical.mfg_solvers import FixedPointIterator

    hjb_solver = HJBFDMSolver(problem=problem)
    fp_solver = FPFDMSolver(problem=problem)

    return FixedPointIterator(problem=problem, hjb_solver=hjb_solver, fp_solver=fp_solver, thetaUM=damping, **kwargs)


def create_standard_solver(
    problem: MFGProblem, solver_type: SolverType = "fixed_point", **kwargs: Any
) -> FixedPointIterator:
    """
    Create standard production MFG solver (Tier 2 - DEFAULT).

    Uses HJB-FDM + FP-Particle hybrid for reliable mass conservation
    and fast convergence. This is the recommended default solver.

    Args:
        problem: MFG problem to solve
        solver_type: Type of solver
        **kwargs: Additional parameters

    Returns:
        Standard-configured solver instance

    Note:
        Default uses Hybrid (HJB-FDM + FP-Particle) for good quality.
        For basic benchmark, use create_basic_solver() (Tier 1: Pure FDM).
        For advanced methods, use create_accurate_solver() (Tier 3).
    """
    # For fixed_point solvers, create default HJB and FP solvers if not provided
    if solver_type == "fixed_point" and "hjb_solver" not in kwargs and "fp_solver" not in kwargs:
        from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
        from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        # Create stable hybrid solver: HJB-FDM + FP-Particle
        # Particle FP naturally conserves mass, FDM HJB is efficient for value function
        hjb_solver = HJBFDMSolver(problem=problem)
        fp_solver = FPParticleSolver(problem=problem, num_particles=5000)

        kwargs["hjb_solver"] = hjb_solver
        kwargs["fp_solver"] = fp_solver

    return SolverFactory.create_solver(problem=problem, solver_type=solver_type, config_preset="fast", **kwargs)


# Backward compatibility alias
def create_fast_solver(
    problem: MFGProblem, solver_type: SolverType = "fixed_point", **kwargs: Any
) -> FixedPointIterator:
    """
    Deprecated: Use create_standard_solver() instead.

    This function is maintained for backward compatibility only.
    """
    import warnings

    warnings.warn(
        "create_fast_solver() is deprecated, use create_standard_solver() instead", DeprecationWarning, stacklevel=2
    )
    return create_standard_solver(problem=problem, solver_type=solver_type, **kwargs)


def create_semi_lagrangian_solver(
    problem: MFGProblem,
    interpolation_method: str = "linear",
    optimization_method: str = "brent",
    characteristic_solver: str = "explicit_euler",
    use_jax: bool | None = None,
    fp_solver_type: str = "fdm",
    **kwargs: Any,
) -> FixedPointIterator:
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
    from mfg_pde.alg.numerical.hjb_solvers.hjb_semi_lagrangian import HJBSemiLagrangianSolver

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
    from mfg_pde.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

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
    # Type assertion since we know this returns FixedPointIterator for fixed_point solver_type
    return cast("FixedPointIterator", solver)


def create_accurate_solver(
    problem: MFGProblem, solver_type: SolverType = "fixed_point", **kwargs: Any
) -> FixedPointIterator:
    """
    Create an accurate MFG solver optimized for precision.

    Args:
        problem: MFG problem to solve
        solver_type: Type of solver
        **kwargs: Additional parameters

    Returns:
        Accurate-configured solver instance
    """
    # For fixed_point solvers, create default HJB and FP solvers if not provided
    if solver_type == "fixed_point" and "hjb_solver" not in kwargs and "fp_solver" not in kwargs:
        from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
        from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        # Create stable hybrid solver: HJB-FDM + FP-Particle
        # Particle FP naturally conserves mass, FDM HJB is efficient for value function
        hjb_solver = HJBFDMSolver(problem=problem)
        fp_solver = FPParticleSolver(problem=problem, num_particles=10000)  # More particles for accuracy

        kwargs["hjb_solver"] = hjb_solver
        kwargs["fp_solver"] = fp_solver

    return SolverFactory.create_solver(problem=problem, solver_type=solver_type, config_preset="accurate", **kwargs)


def create_research_solver(
    problem: MFGProblem, solver_type: SolverType = "fixed_point", **kwargs: Any
) -> FixedPointIterator:
    """
    Create a research MFG solver with comprehensive monitoring.

    Args:
        problem: MFG problem to solve
        solver_type: Type of solver
        **kwargs: Additional parameters

    Returns:
        Research-configured solver instance

    Note:
        Particle-collocation methods have been moved to mfg-research repository.
    """
    return SolverFactory.create_solver(problem=problem, solver_type=solver_type, config_preset="research", **kwargs)


# Note: create_monitored_solver removed - particle-collocation moved to mfg-research repository


def create_amr_solver(
    problem: MFGProblem,
    base_solver_type: SolverType = "fixed_point",
    error_threshold: float = 1e-4,
    max_levels: int = 5,
    **kwargs: Any,
) -> FixedPointIterator:
    """
    Create an AMR-enhanced MFG solver.

    NOTE: AMR enhancement moved to experimental features.
    This function currently returns the base solver type.

    Args:
        problem: MFG problem to solve
        base_solver_type: Base solver type to enhance with AMR
        error_threshold: Error threshold for mesh refinement
        max_levels: Maximum refinement levels
        **kwargs: Additional parameters for base solver and AMR

    Returns:
        Solver (AMR enhancement currently experimental)

    Example:
        >>> # Create solver (AMR enhancement experimental)
        >>> solver = create_amr_solver(
        ...     problem,
        ...     base_solver_type="fixed_point",
        ...     error_threshold=1e-5,
        ...     max_levels=6
        ... )
        >>> result = solver.solve()
    """
    # NOTE: AMR enhancement moved to experimental - returning base solver
    solver = SolverFactory.create_solver(
        problem=problem,
        solver_type=base_solver_type,
        config_preset="accurate",
        **kwargs,
    )
    return solver
