#!/usr/bin/env python3
"""
High-Level MFG Solver Interface

Provides a simple, one-line interface for solving MFG problems with automatic
configuration and method selection.

Example:
    >>> from mfg_pde import MFGProblem, solve_mfg
    >>>
    >>> problem = MFGProblem()
    >>> result = solve_mfg(problem, method="auto", resolution=50)
    >>> U, M = result.U, result.M
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.utils.solver_result import SolverResult

SolverMethod = Literal["auto", "fast", "accurate", "research"]


def solve_mfg(
    problem: MFGProblem,
    method: SolverMethod = "auto",
    resolution: int | None = None,
    max_iterations: int | None = None,
    tolerance: float | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> SolverResult:
    """
    Solve an MFG problem with automatic method selection and configuration.

    This is a high-level convenience function that combines solver creation
    and solving into a single call. For more control, use the factory functions
    (create_fast_solver, create_accurate_solver, create_research_solver) or
    SolverFactory.create_solver() directly.

    Args:
        problem: MFG problem instance to solve
        method: Solution method preset
            - "auto": Automatically select based on problem (recommended)
            - "fast": Optimized for speed with reasonable accuracy
            - "accurate": High precision configuration
            - "research": Comprehensive monitoring and diagnostics
        resolution: Grid resolution or number of particles
            - 1D: Typically 50-200 grid points
            - 2D: Typically 30-100 points per dimension
            - If None, automatic selection based on problem dimension
        max_iterations: Maximum iterations for fixed-point solver
            If None, uses preset default (fast: 100, accurate: 500, research: 1000)
        tolerance: Convergence tolerance
            If None, uses preset default (fast: 1e-4, accurate: 1e-6, research: 1e-8)
        verbose: Print solver progress and diagnostics
        **kwargs: Additional solver-specific parameters
            - damping_factor: Damping parameter for fixed-point (default: 0.5)
            - use_anderson: Enable Anderson acceleration (default: False)
            - backend: Computational backend ("numpy", "jax", "torch", "auto")
                      String values are automatically converted to backend objects
            - hjb_method: Override HJB solver method
            - fp_method: Override FP solver method

    Returns:
        SolverResult with attributes:
            - U: Value function array (Nt+1, Nx+1) or (Nt+1, Nx+1, Ny+1)
            - M: Density array (Nt+1, Nx+1) or (Nt+1, Nx+1, Ny+1)
            - iterations: Number of iterations performed
            - error_history_U: Convergence history for U
            - error_history_M: Convergence history for M
            - converged: Whether convergence was achieved
            - execution_time: Total solve time in seconds
            - metadata: Additional solver-specific information

    Raises:
        ValueError: If problem is invalid or method is unknown
        RuntimeError: If solver fails to converge

    Example:
        >>> from mfg_pde import MFGProblem, solve_mfg
        >>>
        >>> # Simple usage with defaults
        >>> problem = MFGProblem()
        >>> result = solve_mfg(problem)
        >>> U, M = result.U, result.M
        >>> print(f"Converged in {result.iterations} iterations")
        >>>
        >>> # High-accuracy solve with custom resolution
        >>> result = solve_mfg(
        ...     problem,
        ...     method="accurate",
        ...     resolution=100,
        ...     max_iterations=1000,
        ...     tolerance=1e-8
        ... )
        >>>
        >>> # Research mode with diagnostics
        >>> result = solve_mfg(
        ...     problem,
        ...     method="research",
        ...     verbose=True
        ... )
        >>> print(f"Final residual: {result.error_history_U[-1]:.2e}")

    Notes:
        - Automatic method selection (method="auto") chooses:
          * Fast hybrid solver (HJB-FDM + FP-Particle) for 1D/2D
          * Research configuration for complex 3D problems
        - Resolution defaults:
          * 1D: 100 points
          * 2D: 50x50 points
          * 3D: 30x30x30 points
        - For custom solver configurations, use SolverConfig
          and SolverFactory.create_solver() directly
    """
    from mfg_pde.factory import create_accurate_solver, create_research_solver, create_standard_solver

    # Validate method
    valid_methods = ["auto", "fast", "accurate", "research"]
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Choose from: {valid_methods}")

    # Determine automatic resolution if not specified
    if resolution is None:
        dimension = getattr(problem, "dimension", getattr(problem, "d", 1))
        if dimension == 1:
            resolution = 100
        elif dimension == 2:
            resolution = 50
        else:  # 3D
            resolution = 30

    # Create custom config if overrides specified
    custom_config = None
    if max_iterations is not None or tolerance is not None:
        from mfg_pde.config import create_accurate_config, create_fast_config, create_research_config

        # Get base config for method
        if method in ["auto", "fast"]:
            base_config = create_fast_config()
        elif method == "accurate":
            base_config = create_accurate_config()
        else:  # research
            base_config = create_research_config()

        # Apply overrides to base config
        if max_iterations is not None:
            base_config.picard.max_iterations = max_iterations
        if tolerance is not None:
            base_config.picard.tolerance = tolerance
        base_config.picard.verbose = verbose

        custom_config = base_config

    # Select factory function based on method
    if method in ["auto", "fast"]:
        solver_factory = create_standard_solver
    elif method == "accurate":
        solver_factory = create_accurate_solver
    else:  # research
        solver_factory = create_research_solver

    # For research solver, create default hjb_solver and fp_solver if not provided
    if method == "research" and "hjb_solver" not in kwargs and "fp_solver" not in kwargs:
        from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
        from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        kwargs["hjb_solver"] = HJBFDMSolver(problem=problem)
        kwargs["fp_solver"] = FPParticleSolver(problem=problem, num_particles=5000)

    # Convert backend string to backend object if provided
    if "backend" in kwargs and isinstance(kwargs["backend"], str):
        from mfg_pde.backends import create_backend

        kwargs["backend"] = create_backend(kwargs["backend"])

    # Create solver
    solver = solver_factory(problem=problem, custom_config=custom_config, **kwargs)

    # Solve
    result = solver.solve(verbose=verbose)

    return result


# Public API
__all__ = ["solve_mfg"]
