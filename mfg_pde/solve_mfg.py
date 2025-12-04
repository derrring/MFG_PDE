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

    .. deprecated::
        Use ``problem.solve()`` instead. This function will be removed in v1.0.0.

    Args:
        problem: MFG problem instance to solve
        max_iterations: Maximum iterations for fixed-point solver (default: 100)
        tolerance: Convergence tolerance (default: 1e-6)
        verbose: Print solver progress (default: True)
        **kwargs: Additional solver-specific parameters (ignored, for compatibility)

    Returns:
        SolverResult with U, M, convergence info

    Example:
        >>> # Old way (deprecated)
        >>> result = solve_mfg(problem)
        >>>
        >>> # New way (recommended)
        >>> result = problem.solve()
    """
    import warnings

    warnings.warn(
        "solve_mfg() is deprecated. Use problem.solve() instead. This function will be removed in v1.0.0.",
        DeprecationWarning,
        stacklevel=2,
    )

    return problem.solve(
        max_iterations=max_iterations,
        tolerance=tolerance,
        verbose=verbose,
    )


# Public API
__all__ = ["solve_mfg"]
