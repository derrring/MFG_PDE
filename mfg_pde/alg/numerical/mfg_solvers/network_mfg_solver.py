"""
Network MFG Solver Factory.

This module provides factory functions for creating MFG solvers specialized
for network/graph structures, combining NetworkHJBSolver and FPNetworkSolver
with fixed-point iteration.

Mathematical Framework:
- Network HJB: ∂u/∂t + H_i(m, ∇_G u, t) = 0
- Network FP: ∂m/∂t - div_G(m ∇_G H_p) - σ²/2 Δ_G m = 0
- Fixed-point coupling: Iterate between HJB and FP until convergence

Key Features:
- Multiple solver schemes (explicit, implicit, flow-based)
- Graph-specific operators (Laplacian, gradients, divergence)
- Mass conservation on networks
- CFL stability for explicit schemes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mfg_pde.alg.numerical.fp_solvers.fp_network import FPNetworkSolver
from mfg_pde.alg.numerical.hjb_solvers.hjb_network import NetworkHJBSolver
from mfg_pde.alg.numerical.mfg_solvers.fixed_point_iterator import FixedPointIterator

if TYPE_CHECKING:
    from mfg_pde.core.network_mfg_problem import NetworkMFGProblem


def create_network_mfg_solver(
    problem: NetworkMFGProblem,
    solver_type: str = "fixed_point",
    hjb_solver_type: str = "explicit",
    fp_solver_type: str = "explicit",
    damping_factor: float = 0.5,
    hjb_kwargs: dict[str, Any] | None = None,
    fp_kwargs: dict[str, Any] | None = None,
    **solver_kwargs,
) -> FixedPointIterator:
    """
    Create a complete MFG solver for network problems.

    Args:
        problem: NetworkMFGProblem instance
        solver_type: Type of MFG solver ("fixed_point" currently supported)
        hjb_solver_type: HJB scheme ("explicit", "implicit", "semi_implicit")
        fp_solver_type: FP scheme ("explicit", "implicit", "upwind", "flow")
        damping_factor: Fixed-point iteration damping (0.5 = balanced)
        hjb_kwargs: Additional arguments for NetworkHJBSolver
        fp_kwargs: Additional arguments for FPNetworkSolver
        **solver_kwargs: Additional arguments for FixedPointIterator
            - use_anderson: bool = False (Anderson acceleration)
            - anderson_depth: int = 5 (Anderson memory depth)
            - backend: str = None (computational backend)

    Returns:
        Configured MFG solver for network problems

    Example:
        >>> from mfg_pde import NetworkMFGProblem
        >>> from mfg_pde.alg.numerical.mfg_solvers.network_mfg_solver import create_network_mfg_solver
        >>> # Create problem (network geometry, components, etc.)
        >>> problem = NetworkMFGProblem(...)
        >>> # Create solver
        >>> solver = create_network_mfg_solver(
        ...     problem,
        ...     hjb_solver_type="implicit",
        ...     fp_solver_type="implicit",
        ...     damping_factor=0.7,
        ... )
        >>> # Solve
        >>> U, M, info = solver.solve(max_iterations=50, tolerance=1e-6)
    """
    # Default kwargs
    hjb_kwargs = hjb_kwargs or {}
    fp_kwargs = fp_kwargs or {}

    # Create HJB solver
    hjb_solver = NetworkHJBSolver(
        problem=problem,
        scheme=hjb_solver_type,
        **hjb_kwargs,
    )

    # Create FP solver
    fp_solver = FPNetworkSolver(
        problem=problem,
        scheme=fp_solver_type,
        **fp_kwargs,
    )

    # Create fixed-point iterator
    if solver_type == "fixed_point":
        solver = FixedPointIterator(
            problem=problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            damping_factor=damping_factor,
            **solver_kwargs,
        )
    else:
        raise ValueError(f"Unknown solver_type: {solver_type}. Only 'fixed_point' is currently supported.")

    return solver


def create_simple_network_solver(
    problem: NetworkMFGProblem,
    scheme: str = "explicit",
    damping: float = 0.5,
    cfl_factor: float = 0.5,
) -> FixedPointIterator:
    """
    Create a simple network MFG solver with minimal configuration.

    This is a convenience function for quick setup with default parameters.

    Args:
        problem: NetworkMFGProblem instance
        scheme: Both HJB and FP scheme ("explicit" or "implicit")
        damping: Fixed-point damping factor
        cfl_factor: CFL stability factor for explicit schemes

    Returns:
        Configured network MFG solver

    Example:
        >>> solver = create_simple_network_solver(problem, scheme="implicit")
        >>> U, M, info = solver.solve()
    """
    return create_network_mfg_solver(
        problem=problem,
        hjb_solver_type=scheme,
        fp_solver_type=scheme,
        damping_factor=damping,
        hjb_kwargs={"cfl_factor": cfl_factor} if scheme == "explicit" else {},
        fp_kwargs={"cfl_factor": cfl_factor} if scheme == "explicit" else {},
    )
