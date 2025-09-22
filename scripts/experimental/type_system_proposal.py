#!/usr/bin/env python3
"""
Type System Design Proposal for MFG_PDE

Goal: Balance type safety for developers with simplicity for users
"""

from typing import Any, Protocol, runtime_checkable

# =============================================================================
# LAYER 1: USER-FACING TYPES (3-4 types maximum)
# =============================================================================


@runtime_checkable
class MFGProblem(Protocol):
    """Simple protocol - anything that acts like an MFG problem."""

    def get_domain_bounds(self) -> tuple[float, float]: ...
    def evaluate_hamiltonian(self, x: float, p: float, m: float, t: float) -> float: ...


@runtime_checkable
class MFGSolver(Protocol):
    """Simple protocol - anything that can solve MFG problems."""

    def solve(self, max_iterations: int = 100) -> Any: ...


@runtime_checkable
class MFGResult(Protocol):
    """Simple protocol - solution results."""

    def get_solution_u(self) -> Any: ...
    def get_solution_m(self) -> Any: ...
    def converged(self) -> bool: ...


# =============================================================================
# LAYER 2: SIMPLIFIED PUBLIC API
# =============================================================================


def create_mfg_problem(
    problem_type: str, domain: tuple[float, float] = (0.0, 1.0), time_horizon: float = 1.0, **kwargs
) -> MFGProblem:
    """
    Create MFG problem with simple string-based configuration.

    Args:
        problem_type: "crowd_dynamics", "portfolio", "traffic", etc.
        domain: Spatial domain (xmin, xmax)
        time_horizon: Time horizon T
        **kwargs: Problem-specific parameters

    Returns:
        MFG problem ready to solve

    Example:
        >>> problem = create_mfg_problem("crowd_dynamics", domain=(0, 5))
        >>> result = solve_mfg(problem)
    """


def create_fast_solver(problem: MFGProblem, method: str = "auto") -> MFGSolver:
    """
    Create optimized solver with automatic method selection.

    Args:
        problem: MFG problem to solve
        method: "auto", "fdm", "particle", "spectral"

    Returns:
        Configured solver
    """


def solve_mfg(problem: MFGProblem, max_iterations: int = 100, tolerance: float = 1e-6) -> MFGResult:
    """
    One-line MFG solver with sensible defaults.

    Args:
        problem: MFG problem
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Solution with u(t,x), m(t,x) and convergence info
    """


# =============================================================================
# LAYER 3: INTERNAL TYPE SYSTEM (for maintainers)
# =============================================================================

# Only import complex types when type checking

# ... all the complex internal types

# =============================================================================
# USAGE PATTERNS
# =============================================================================

"""
BEGINNER USER (90% of users):
    problem = create_mfg_problem("crowd_dynamics")
    result = solve_mfg(problem)

INTERMEDIATE USER (8% of users):
    problem = create_mfg_problem("custom", hamiltonian=my_H, initial_density=my_m0)
    solver = create_fast_solver(problem, method="particle")
    result = solver.solve(max_iterations=200)

EXPERT USER/MAINTAINER (2% of users):
    from mfg_pde.alg.hjb_solvers import HJBSemiLagrangianSolver
    solver = HJBSemiLagrangianSolver(problem, config=advanced_config)
    # Full access to internal complexity when needed
"""
