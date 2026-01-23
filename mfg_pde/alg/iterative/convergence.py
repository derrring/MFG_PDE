"""
Generic Convergence Utilities for Iterative Methods.

This module provides paradigm-agnostic utilities for iterative algorithms:
- Numerical PDE solvers (MFG coupling)
- Neural network training (PINN)
- Reinforcement learning (Policy Iteration)
- Optimization (Alternating minimization)

Architecture (DRY - Issue #630):
    - **This module**: Generic iteration utilities (damping, simple convergence checks)
    - **utils/convergence/**: PDE-aware utilities (grid-scaled errors, Wasserstein, etc.)

For PDE-specific metrics with grid scaling, use:
    from mfg_pde.utils.convergence import calculate_error, RollingConvergenceMonitor

This module provides simpler interfaces for paradigm-agnostic iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np

# =============================================================================
# RE-EXPORTS FROM utils/convergence FOR CONVENIENCE (DRY)
# =============================================================================
# These provide PDE-aware metrics for users who need grid scaling
from mfg_pde.utils.convergence import (
    ConvergenceChecker,
    ConvergenceConfig,
    RollingConvergenceMonitor,
    calculate_error,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

# Generic type for iteration state
T = TypeVar("T")


# =============================================================================
# PROTOCOLS
# =============================================================================


class MetricComputable(Protocol):
    """Protocol for objects that can compute distance metrics."""

    def compute_distance(self, other: MetricComputable) -> float:
        """Compute distance/difference from another state."""
        ...


# =============================================================================
# RESULT TYPES
# =============================================================================


@dataclass
class ConvergenceResult:
    """
    Result of convergence check for iterative patterns.

    Simple result type for paradigm-agnostic iteration.
    For PDE-specific results with more metrics, see utils/convergence.
    """

    converged: bool
    reason: str
    relative_error: float
    absolute_error: float


# =============================================================================
# DAMPING UTILITIES (UNIQUE TO ITERATIVE PATTERNS)
# =============================================================================


def apply_damping_generic(
    new_value: T,
    old_value: T,
    alpha: float,
    combine_fn: Callable[[T, T, float], T] | None = None,
) -> T:
    """
    Apply damping to iteration update (generic version).

    Damping formula:
        result = alpha * new_value + (1 - alpha) * old_value

    Args:
        new_value: New value from forward operator
        old_value: Previous value
        alpha: Damping/learning rate in [0, 1]
        combine_fn: Optional function to combine values
                   (defaults to weighted sum for numpy arrays)

    Returns:
        Damped value

    Note:
        alpha=1 means full update (no damping)
        alpha=0 means no update (keep old value)
        alpha=0.5 is a common choice for stability
    """
    if combine_fn is not None:
        return combine_fn(new_value, old_value, alpha)

    # Default: weighted sum (works for numpy arrays)
    return alpha * new_value + (1 - alpha) * old_value


def apply_damping_arrays(
    new_arrays: tuple[NDArray, ...],
    old_arrays: tuple[NDArray, ...],
    alpha: float,
) -> tuple[NDArray, ...]:
    """
    Apply damping to multiple arrays simultaneously.

    Args:
        new_arrays: Tuple of new arrays from forward operator
        old_arrays: Tuple of previous arrays
        alpha: Damping parameter in [0, 1]

    Returns:
        Tuple of damped arrays

    Example:
        >>> U_new, M_new = compute_step(U_old, M_old)
        >>> U_damped, M_damped = apply_damping_arrays(
        ...     (U_new, M_new), (U_old, M_old), alpha=0.5
        ... )
    """
    return tuple(alpha * new + (1 - alpha) * old for new, old in zip(new_arrays, old_arrays, strict=True))


# =============================================================================
# SIMPLE CONVERGENCE CHECKS (PARADIGM-AGNOSTIC)
# =============================================================================


def check_convergence(
    relative_errors: dict[str, float],
    absolute_errors: dict[str, float],
    tolerance: float,
    mode: str = "all",
) -> ConvergenceResult:
    """
    Check convergence based on relative and absolute errors.

    Args:
        relative_errors: Dict of {name: relative_error}
        absolute_errors: Dict of {name: absolute_error}
        tolerance: Convergence tolerance
        mode: "all" (all must converge) or "any" (any can converge)

    Returns:
        ConvergenceResult with status and diagnostics

    Example:
        >>> result = check_convergence(
        ...     {"U": 1e-7, "M": 1e-8},
        ...     {"U": 1e-6, "M": 1e-7},
        ...     tolerance=1e-6,
        ... )
        >>> print(result.converged)  # True
    """
    max_rel = max(relative_errors.values()) if relative_errors else 0.0
    max_abs = max(absolute_errors.values()) if absolute_errors else 0.0

    if mode == "all":
        converged = max_rel < tolerance and max_abs < tolerance
    else:  # "any"
        converged = max_rel < tolerance or max_abs < tolerance

    if converged:
        reason = f"Converged: Rel err {max_rel:.1e}, Abs err {max_abs:.1e} < tol {tolerance:.1e}"
    else:
        reason = ""

    return ConvergenceResult(
        converged=converged,
        reason=reason,
        relative_error=max_rel,
        absolute_error=max_abs,
    )


def check_convergence_simple(
    rel_error: float,
    abs_error: float,
    tolerance: float,
) -> tuple[bool, str]:
    """
    Simple convergence check (backward compatible interface).

    Both relative AND absolute errors must be below tolerance.

    Args:
        rel_error: Maximum relative error
        abs_error: Maximum absolute error
        tolerance: Convergence tolerance

    Returns:
        Tuple of (converged: bool, reason: str)
    """
    if rel_error < tolerance and abs_error < tolerance:
        reason = f"Converged: Rel err {rel_error:.1e}, Abs err {abs_error:.1e} < tol {tolerance:.1e}"
        return True, reason
    return False, ""


# =============================================================================
# SIMPLE ERROR COMPUTATION (PARADIGM-AGNOSTIC)
# =============================================================================
# For PDE-aware error computation with grid scaling, use:
#   from mfg_pde.utils.convergence import calculate_error


def compute_relative_change(
    new_value: NDArray,
    old_value: NDArray,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute relative change between iterations (simple L2 norm).

    Formula: ||new - old|| / (||old|| + epsilon)

    Args:
        new_value: New array
        old_value: Previous array
        epsilon: Small constant to prevent division by zero

    Returns:
        Relative change (dimensionless)

    Note:
        For PDE-aware computation with grid scaling, use:
        calculate_error(new, old, dx=dx, norm='l2')['relative']
    """
    diff_norm = np.linalg.norm(new_value - old_value)
    old_norm = np.linalg.norm(old_value)
    return float(diff_norm / (old_norm + epsilon))


def compute_absolute_change(
    new_value: NDArray,
    old_value: NDArray,
) -> float:
    """
    Compute absolute change between iterations (L2 norm).

    Formula: ||new - old||

    Args:
        new_value: New array
        old_value: Previous array

    Returns:
        Absolute change (same units as input)

    Note:
        For PDE-aware computation with grid scaling, use:
        calculate_error(new, old, dx=dx, norm='l2')['absolute']
    """
    return float(np.linalg.norm(new_value - old_value))


# =============================================================================
# CONVERGENCE TRACKER (SIMPLE HISTORY FOR ITERATION PATTERNS)
# =============================================================================
# For advanced statistical monitoring, use:
#   from mfg_pde.utils.convergence import RollingConvergenceMonitor


class ConvergenceTracker:
    """
    Track convergence history over iterations.

    Simple history tracker for iteration patterns. Pre-allocates arrays
    for efficient tracking of named quantities (e.g., "U", "M").

    For advanced statistical convergence monitoring (rolling windows,
    quantile-based stopping), use:
        from mfg_pde.utils.convergence import RollingConvergenceMonitor
    """

    def __init__(self, max_iterations: int, tracked_quantities: list[str] | None = None):
        """
        Initialize convergence tracker.

        Args:
            max_iterations: Maximum number of iterations to track
            tracked_quantities: Names of quantities to track (default ["U", "M"])
        """
        self.max_iterations = max_iterations
        self.tracked_quantities = tracked_quantities or ["U", "M"]

        # Initialize history arrays
        self.relative_errors: dict[str, NDArray] = {name: np.ones(max_iterations) for name in self.tracked_quantities}
        self.absolute_errors: dict[str, NDArray] = {name: np.ones(max_iterations) for name in self.tracked_quantities}

        self.iterations_run = 0

    def record(self, iteration: int, relative: dict[str, float], absolute: dict[str, float]) -> None:
        """
        Record errors for an iteration.

        Args:
            iteration: Current iteration number
            relative: Dict of {name: relative_error}
            absolute: Dict of {name: absolute_error}
        """
        for name in self.tracked_quantities:
            if name in relative:
                self.relative_errors[name][iteration] = relative[name]
            if name in absolute:
                self.absolute_errors[name][iteration] = absolute[name]

        self.iterations_run = max(self.iterations_run, iteration + 1)

    def get_history(self) -> dict[str, NDArray]:
        """Get truncated error history."""
        result = {}
        for name in self.tracked_quantities:
            result[f"{name}_rel"] = self.relative_errors[name][: self.iterations_run]
            result[f"{name}_abs"] = self.absolute_errors[name][: self.iterations_run]
        return result

    def get_convergence_rate(self, name: str = "U", window: int = 10) -> float:
        """
        Estimate convergence rate from recent history.

        Uses linear regression on log(error) vs iteration.

        Args:
            name: Quantity to analyze
            window: Number of recent iterations to use

        Returns:
            Estimated convergence rate (negative = converging)
        """
        if self.iterations_run < 2:
            return 0.0

        errors = self.relative_errors[name][: self.iterations_run]
        start = max(0, self.iterations_run - window)
        recent = errors[start:]

        if len(recent) < 2 or np.any(recent <= 0):
            return 0.0

        # Linear regression on log(error)
        x = np.arange(len(recent))
        log_err = np.log(recent)
        slope = np.polyfit(x, log_err, 1)[0]

        return float(slope)


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing convergence utilities...")

    # Test apply_damping_generic
    new = np.array([1.0, 2.0])
    old = np.array([0.0, 0.0])
    damped = apply_damping_generic(new, old, 0.5)
    assert np.allclose(damped, [0.5, 1.0])
    print("  apply_damping_generic() works")

    # Test apply_damping_arrays
    U_new, M_new = np.ones(5), np.ones(5) * 2
    U_old, M_old = np.zeros(5), np.zeros(5)
    U_d, M_d = apply_damping_arrays((U_new, M_new), (U_old, M_old), 0.5)
    assert np.allclose(U_d, 0.5)
    assert np.allclose(M_d, 1.0)
    print("  apply_damping_arrays() works")

    # Test check_convergence
    result = check_convergence({"U": 1e-7, "M": 1e-8}, {"U": 1e-7, "M": 1e-8}, tolerance=1e-6)
    assert result.converged
    print("  check_convergence() works")

    # Test check_convergence_simple
    converged, reason = check_convergence_simple(1e-7, 1e-7, 1e-6)
    assert converged
    print("  check_convergence_simple() works")

    # Test ConvergenceTracker
    tracker = ConvergenceTracker(100, ["U", "M"])
    tracker.record(0, {"U": 0.1, "M": 0.2}, {"U": 1.0, "M": 2.0})
    tracker.record(1, {"U": 0.01, "M": 0.02}, {"U": 0.1, "M": 0.2})
    history = tracker.get_history()
    assert len(history["U_rel"]) == 2
    print("  ConvergenceTracker works")

    # Test re-exports from utils/convergence (DRY verification)
    assert calculate_error is not None
    assert RollingConvergenceMonitor is not None
    assert ConvergenceChecker is not None
    assert ConvergenceConfig is not None
    print("  Re-exports from utils/convergence work")

    print("All smoke tests passed!")
