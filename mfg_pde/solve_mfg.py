#!/usr/bin/env python3
"""
High-Level MFG Solver Interface

Provides a simple interface for solving MFG problems.

Example:
    >>> from mfg_pde import MFGProblem, solve_mfg
    >>>
    >>> problem = MFGProblem()
    >>> result = solve_mfg(problem)
    >>> U, M = result.U, result.M
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.utils.solver_result import SolverResult


def solve_mfg(
    problem: MFGProblem,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = True,
    **kwargs: Any,
) -> SolverResult:
    """
    Solve an MFG problem using the standard fixed-point solver.

    This is a convenience function that creates a solver and solves the problem.
    For more control, use create_standard_solver() directly.

    Args:
        problem: MFG problem instance to solve
        max_iterations: Maximum iterations for fixed-point solver (default: 100)
        tolerance: Convergence tolerance (default: 1e-6)
        verbose: Print solver progress (default: True)
        **kwargs: Additional solver-specific parameters passed to create_standard_solver

    Returns:
        SolverResult with attributes:
            - U: Value function array (Nt+1, Nx+1)
            - M: Density array (Nt+1, Nx+1)
            - iterations: Number of iterations performed
            - error_history_U: Convergence history for U
            - error_history_M: Convergence history for M
            - converged: Whether convergence was achieved

    Example:
        >>> from mfg_pde import MFGProblem, solve_mfg
        >>>
        >>> problem = MFGProblem()
        >>> result = solve_mfg(problem)
        >>> print(f"Converged: {result.converged} in {result.iterations} iterations")
        >>>
        >>> # With custom settings
        >>> result = solve_mfg(problem, max_iterations=200, tolerance=1e-8)
    """
    from mfg_pde.config import create_fast_config
    from mfg_pde.factory import create_standard_solver

    # Create config with specified parameters
    config = create_fast_config()
    config.picard.max_iterations = max_iterations
    config.picard.tolerance = tolerance
    config.picard.verbose = verbose

    # Create solver
    solver = create_standard_solver(problem=problem, custom_config=config, **kwargs)

    # Solve
    result = solver.solve(verbose=verbose)

    return result


# Public API
__all__ = ["solve_mfg"]
