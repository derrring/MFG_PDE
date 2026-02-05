"""
Fixed Point Iterator Utilities

Shared helper functions for fixed-point iteration solvers to eliminate code duplication.
These utilities are used by both legacy and config-aware fixed-point iterator implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np  # noqa: TC002

if TYPE_CHECKING:
    from mfg_pde.utils.solver_result import SolverResult


def initialize_cold_start(
    U: np.ndarray,
    M: np.ndarray,
    M_initial: np.ndarray,
    U_terminal: np.ndarray,
    Nt: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize U and M arrays for cold start (no warm start provided).

    Cold start initialization sets:
    - U[Nt-1, :] = U_terminal (terminal condition)
    - M[0, :] = M_initial (initial condition)
    - U[0:Nt-1, :] = U_terminal (interior initialized with terminal condition)
    - M[1:Nt, :] = M_initial (interior initialized with initial condition)

    Args:
        U: Value function array to initialize (modified in-place)
        M: Density array to initialize (modified in-place)
        M_initial: Initial density distribution m_0(x)
        U_terminal: Terminal condition g(x) for value function
        Nt: Number of time steps

    Returns:
        Tuple of (initialized U, initialized M)

    Note:
        This is the standard initialization used when no warm start is provided.
        Interior points are set to boundary conditions as a simple initial guess.
    """
    # Set boundary conditions
    U[Nt - 1, :] = U_terminal
    M[0, :] = M_initial

    # Initialize interior with boundary conditions
    for n_time_idx in range(Nt - 1):
        U[n_time_idx, :] = U_terminal
    for n_time_idx in range(1, Nt):
        M[n_time_idx, :] = M_initial

    return U, M


def construct_solver_result(
    U: np.ndarray,
    M: np.ndarray,
    iterations_run: int,
    l2distu_abs: np.ndarray,
    l2distm_abs: np.ndarray,
    l2distu_rel: np.ndarray,
    l2distm_rel: np.ndarray,
    solver_name: str,
    converged: bool,
    convergence_reason: str,
) -> SolverResult:
    """
    Construct a SolverResult object from fixed-point iteration data.

    Args:
        U: Final value function array
        M: Final density array
        iterations_run: Number of iterations executed
        l2distu_abs: Absolute L2 errors for U (full array)
        l2distm_abs: Absolute L2 errors for M (full array)
        l2distu_rel: Relative L2 errors for U (full array)
        l2distm_rel: Relative L2 errors for M (full array)
        solver_name: Name of the solver
        converged: Whether the solver converged
        convergence_reason: Reason for convergence/termination

    Returns:
        SolverResult object with all diagnostic information

    Note:
        Error arrays are truncated to iterations_run length before storage.
    """
    from mfg_pde.utils.solver_result import SolverResult

    # Truncate error arrays to actual iterations run
    l2distu_abs_truncated = l2distu_abs[:iterations_run]
    l2distm_abs_truncated = l2distm_abs[:iterations_run]
    l2distu_rel_truncated = l2distu_rel[:iterations_run]
    l2distm_rel_truncated = l2distm_rel[:iterations_run]

    # Construct result object
    result = SolverResult(
        U=U,
        M=M,
        converged=converged,
        iterations=iterations_run,
        convergence_reason=convergence_reason,
        diagnostics={
            "l2distu_abs": l2distu_abs_truncated,
            "l2distm_abs": l2distm_abs_truncated,
            "l2distu_rel": l2distu_rel_truncated,
            "l2distm_rel": l2distm_rel_truncated,
            "solver_name": solver_name,
        },
    )

    return result


def apply_damping(
    U_new: np.ndarray,
    U_old: np.ndarray,
    M_new: np.ndarray,
    M_old: np.ndarray,
    theta: float,
    theta_M: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply damping to fixed-point iteration updates.

    Damping formula:
        U_damped = theta * U_new + (1 - theta) * U_old
        M_damped = theta_M * M_new + (1 - theta_M) * M_old

    Args:
        U_new: New value function from HJB solve
        U_old: Previous value function
        M_new: New density from FP solve
        M_old: Previous density
        theta: Damping parameter for U in [0, 1] (0=no update, 1=full update)
            Also used for M if theta_M is None (backward compatible).
        theta_M: Optional separate damping parameter for M in [0, 1].
            If None, uses theta for both U and M (backward compatible).
            Issue #719: Per-variable damping support.

    Returns:
        Tuple of (U_damped, M_damped)

    Note:
        theta=1 corresponds to no damping (full update)
        theta=0.5 is a common choice for stability
        Smaller theta increases stability but slows convergence

    Example:
        # Same damping for both (backward compatible)
        U, M = apply_damping(U_new, U_old, M_new, M_old, theta=0.5)

        # Different damping: full update for U, conservative for M
        # Recommended for MFG: U adapts quickly, M filters particle noise
        U, M = apply_damping(U_new, U_old, M_new, M_old, theta=1.0, theta_M=0.2)
    """
    # Issue #719: Support per-variable damping
    theta_U = theta
    if theta_M is None:
        theta_M = theta  # Backward compatible: same damping for both

    U_damped = theta_U * U_new + (1 - theta_U) * U_old
    M_damped = theta_M * M_new + (1 - theta_M) * M_old

    return U_damped, M_damped


def check_convergence_criteria(
    l2distu_rel: float,
    l2distm_rel: float,
    l2distu_abs: float,
    l2distm_abs: float,
    tol_picard: float,
) -> tuple[bool, str]:
    """
    Check if fixed-point iteration has converged.

    Convergence criteria (both must be satisfied):
    1. Relative errors: max(l2distu_rel, l2distm_rel) < tol_picard
    2. Absolute errors: max(l2distu_abs, l2distm_abs) < tol_picard

    Args:
        l2distu_rel: Relative L2 error for U
        l2distm_rel: Relative L2 error for M
        l2distu_abs: Absolute L2 error for U
        l2distm_abs: Absolute L2 error for M
        tol_picard: Convergence tolerance

    Returns:
        Tuple of (converged: bool, reason: str)

    Example:
        >>> converged, reason = check_convergence_criteria(1e-7, 1e-8, 1e-6, 1e-7, 1e-6)
        >>> print(converged)  # True
        >>> print(reason)  # "Converged: Rel err 1.0e-07, Abs err 1.0e-06 < tol 1.0e-06"
    """
    max_rel_err = max(l2distu_rel, l2distm_rel)
    max_abs_err = max(l2distu_abs, l2distm_abs)

    # Both relative and absolute errors must be below tolerance
    if max_rel_err < tol_picard and max_abs_err < tol_picard:
        reason = f"Converged: Rel err {max_rel_err:.1e}, Abs err {max_abs_err:.1e} < tol {tol_picard:.1e}"
        return True, reason
    else:
        return False, ""


def preserve_initial_condition(
    M: np.ndarray,
    M_initial: np.ndarray,
) -> np.ndarray:
    """
    Preserve initial condition for density after updates.

    After damping or Anderson acceleration, M[0, :] may be modified,
    but the initial condition must remain fixed.

    Args:
        M: Density array (modified in-place)
        M_initial: Initial density distribution m_0(x)

    Returns:
        Modified M with preserved initial condition

    Note:
        This is critical for maintaining physical correctness of the solution.
        The initial density distribution is a boundary condition that must not change.
    """
    M[0, :] = M_initial
    return M


def adapt_damping(
    theta_U: float,
    theta_M: float,
    error_history_U: list[float],
    error_history_M: list[float],
    *,
    theta_U_initial: float,
    theta_M_initial: float,
    decay: float = 0.5,
    min_damping: float = 0.05,
    increase_threshold: float = 1.2,
    recovery_rate: float = 1.05,
    stable_window: int = 3,
) -> tuple[float, float, str | None]:
    """
    Adapt Picard damping factors based on error history (Issue #583).

    Detects oscillation (error increasing) and reduces damping to stabilize.
    After sustained convergence, cautiously recovers toward initial damping.

    U and M are adapted independently since U gradient explosion is the
    primary pathology in strongly-coupled MFG systems.

    Args:
        theta_U: Current damping factor for U.
        theta_M: Current damping factor for M.
        error_history_U: Relative L2 error history for U (all iterations so far).
        error_history_M: Relative L2 error history for M (all iterations so far).
        theta_U_initial: Initial damping factor for U (recovery ceiling).
        theta_M_initial: Initial damping factor for M (recovery ceiling).
        decay: Multiplicative decay on oscillation (e.g., 0.5 halves damping).
        min_damping: Minimum damping bound (prevents stalling).
        increase_threshold: Error ratio above which oscillation is detected.
        recovery_rate: Multiplicative increase during stable convergence.
        stable_window: Consecutive decreasing iterations required for recovery.

    Returns:
        (theta_U, theta_M, warning_msg): Updated damping factors and optional
        warning message (None if no oscillation detected).
    """
    warning_msg = None

    # Need at least 2 data points to detect oscillation
    if len(error_history_U) < 2:
        return theta_U, theta_M, None

    # --- Adapt U ---
    ratio_U = error_history_U[-1] / error_history_U[-2] if error_history_U[-2] > 0 else 1.0
    if ratio_U > increase_threshold:
        theta_U = max(theta_U * decay, min_damping)
        warning_msg = f"Adaptive damping: U error increased by {ratio_U:.2f}x. Reduced theta_U to {theta_U:.4f}."
    elif len(error_history_U) >= stable_window and all(
        error_history_U[-(i + 1)] < error_history_U[-(i + 2)] for i in range(stable_window - 1)
    ):
        theta_U = min(theta_U * recovery_rate, theta_U_initial)

    # --- Adapt M ---
    ratio_M = error_history_M[-1] / error_history_M[-2] if error_history_M[-2] > 0 else 1.0
    if ratio_M > increase_threshold:
        theta_M = max(theta_M * decay, min_damping)
        msg_M = f"Adaptive damping: M error increased by {ratio_M:.2f}x. Reduced theta_M to {theta_M:.4f}."
        warning_msg = f"{warning_msg} {msg_M}" if warning_msg else msg_M
    elif len(error_history_M) >= stable_window and all(
        error_history_M[-(i + 1)] < error_history_M[-(i + 2)] for i in range(stable_window - 1)
    ):
        theta_M = min(theta_M * recovery_rate, theta_M_initial)

    return theta_U, theta_M, warning_msg


def preserve_terminal_condition(
    U: np.ndarray,
    U_terminal: np.ndarray,
) -> np.ndarray:
    """
    Preserve terminal condition for value function after updates.

    After damping or Anderson acceleration, U[-1, :] may be modified,
    but the terminal condition must remain fixed.

    Args:
        U: Value function array (modified in-place)
        U_terminal: Terminal condition g(x) at t=T

    Returns:
        Modified U with preserved terminal condition

    Note:
        This is critical for maintaining physical correctness of the solution.
        The terminal condition is a boundary condition that must not change.
        Without this, damping dilutes the terminal condition, causing the
        value function gradient to vanish and agents to not move toward targets.
    """
    U[-1] = U_terminal
    return U
