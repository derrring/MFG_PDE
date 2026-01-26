"""
Runtime validation utilities (NaN/Inf detection, divergence monitoring).

This module provides utilities for validating solver outputs during
and after execution.

Issue #688: Runtime safety (NaN/Inf detection, divergence, bounds checking)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from mfg_pde.utils.validation.protocol import ValidationResult

if TYPE_CHECKING:
    from numpy.typing import NDArray


def check_finite(
    arr: NDArray,
    name: str,
    location: str | None = None,
    *,
    raise_on_error: bool = True,
) -> ValidationResult:
    """
    Check array for NaN/Inf values.

    Args:
        arr: Array to check
        name: Name for error messages (e.g., "u", "m")
        location: Additional location info (e.g., "timestep 10")
        raise_on_error: If True, raise ValueError on non-finite values

    Returns:
        ValidationResult

    Raises:
        ValueError: If raise_on_error=True and non-finite values found

    Issue #688: NaN/Inf detection
    """
    result = ValidationResult()

    if np.all(np.isfinite(arr)):
        return result

    # Find non-finite values
    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)

    n_nan = np.sum(nan_mask)
    n_inf = np.sum(inf_mask)

    # Build message
    parts = [name]
    if location:
        parts.append(f"({location})")

    if n_nan > 0:
        first_nan = np.unravel_index(np.argmax(nan_mask), arr.shape)
        msg = f"{' '.join(parts)} contains {n_nan} NaN values. First at index {first_nan}"
        result.add_error(msg, location=name)

    if n_inf > 0:
        first_inf = np.unravel_index(np.argmax(inf_mask), arr.shape)
        msg = f"{' '.join(parts)} contains {n_inf} Inf values. First at index {first_inf}"
        result.add_error(msg, location=name)

    result.context["n_nan"] = n_nan
    result.context["n_inf"] = n_inf

    if raise_on_error and not result.is_valid:
        raise ValueError(str(result))

    return result


def check_bounds(
    arr: NDArray,
    name: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
    tolerance: float = 0.0,
) -> ValidationResult:
    """
    Check array values are within bounds.

    Args:
        arr: Array to check
        name: Name for error messages
        lower: Lower bound (None = no check)
        upper: Upper bound (None = no check)
        tolerance: Allow values slightly outside bounds

    Returns:
        ValidationResult

    Issue #688: Bounds checking
    """
    result = ValidationResult()

    if lower is not None:
        min_val = np.min(arr)
        if min_val < lower - tolerance:
            n_violations = np.sum(arr < lower - tolerance)
            result.add_error(
                f"{name} has {n_violations} values below lower bound {lower} (min={min_val:.6e})",
                location=name,
            )
            result.context["min_value"] = min_val

    if upper is not None:
        max_val = np.max(arr)
        if max_val > upper + tolerance:
            n_violations = np.sum(arr > upper + tolerance)
            result.add_error(
                f"{name} has {n_violations} values above upper bound {upper} (max={max_val:.6e})",
                location=name,
            )
            result.context["max_value"] = max_val

    return result


@dataclass
class DivergenceMonitor:
    """
    Monitor error history to detect divergence in iterative solvers.

    Tracks error values and detects if the solver is diverging
    (error consistently increasing).

    Args:
        patience: Number of consecutive increases before flagging divergence
        increase_threshold: Relative increase threshold (1.1 = 10% increase)

    Usage:
        monitor = DivergenceMonitor(patience=5)

        for iteration in range(max_iterations):
            error = compute_error()

            if monitor.update(error):
                raise RuntimeError(f"Solver diverging: {monitor.diagnostic}")

            if error < tolerance:
                break

    Issue #688: Divergence detection
    """

    patience: int = 5
    increase_threshold: float = 1.1
    errors: list[float] = field(default_factory=list)
    increasing_count: int = field(default=0, init=False)

    def update(self, error: float) -> bool:
        """
        Update with new error value.

        Args:
            error: Current error value

        Returns:
            True if divergence detected, False otherwise
        """
        if self.errors and error > self.errors[-1] * self.increase_threshold:
            self.increasing_count += 1
        else:
            self.increasing_count = 0

        self.errors.append(error)
        return self.is_diverging

    @property
    def is_diverging(self) -> bool:
        """Check if currently in diverging state."""
        return self.increasing_count >= self.patience

    @property
    def diagnostic(self) -> str:
        """Get diagnostic string for divergence."""
        recent = self.errors[-min(10, len(self.errors)) :]
        return (
            f"Error increased for {self.increasing_count} consecutive iterations. "
            f"Recent errors: {[f'{e:.2e}' for e in recent]}"
        )

    def reset(self) -> None:
        """Reset monitor state."""
        self.errors.clear()
        self.increasing_count = 0


@dataclass
class ConvergenceMonitor:
    """
    Monitor convergence in iterative solvers.

    Tracks error history and provides convergence diagnostics.

    Args:
        tolerance: Convergence tolerance
        stagnation_window: Window for detecting stagnation
        stagnation_threshold: Relative improvement threshold for stagnation

    Usage:
        monitor = ConvergenceMonitor(tolerance=1e-6)

        for iteration in range(max_iterations):
            error = compute_error()
            monitor.update(error)

            if monitor.is_converged:
                print(f"Converged in {iteration} iterations")
                break

            if monitor.is_stagnating:
                print(f"Warning: convergence stagnating")

    Issue #688: Convergence monitoring
    """

    tolerance: float = 1e-6
    stagnation_window: int = 10
    stagnation_threshold: float = 0.01
    errors: list[float] = field(default_factory=list)

    def update(self, error: float) -> None:
        """Update with new error value."""
        self.errors.append(error)

    @property
    def current_error(self) -> float | None:
        """Get current (most recent) error."""
        return self.errors[-1] if self.errors else None

    @property
    def is_converged(self) -> bool:
        """Check if converged (error below tolerance)."""
        return self.current_error is not None and self.current_error < self.tolerance

    @property
    def is_stagnating(self) -> bool:
        """Check if convergence is stagnating."""
        if len(self.errors) < self.stagnation_window:
            return False

        recent = self.errors[-self.stagnation_window :]
        improvement = (recent[0] - recent[-1]) / (recent[0] + 1e-15)
        return improvement < self.stagnation_threshold

    @property
    def convergence_rate(self) -> float | None:
        """Estimate convergence rate from recent errors."""
        if len(self.errors) < 2:
            return None

        # Estimate rate from last few errors
        recent = self.errors[-min(5, len(self.errors)) :]
        if len(recent) < 2:
            return None

        # Linear regression on log(error) vs iteration
        log_errors = np.log(np.array(recent) + 1e-15)
        n = len(log_errors)
        x = np.arange(n)

        slope = (n * np.sum(x * log_errors) - np.sum(x) * np.sum(log_errors)) / (
            n * np.sum(x**2) - np.sum(x) ** 2 + 1e-15
        )

        return float(np.exp(slope))  # Rate per iteration

    def get_summary(self) -> dict:
        """Get convergence summary."""
        return {
            "n_iterations": len(self.errors),
            "initial_error": self.errors[0] if self.errors else None,
            "final_error": self.errors[-1] if self.errors else None,
            "converged": self.is_converged,
            "stagnating": self.is_stagnating,
            "convergence_rate": self.convergence_rate,
        }


def validate_solver_output(
    U: NDArray,
    M: NDArray,
    *,
    check_finite: bool = True,
    check_density_positive: bool = True,
    density_tolerance: float = -1e-10,
) -> ValidationResult:
    """
    Validate complete solver output.

    Args:
        U: Value function array
        M: Density array
        check_finite: Check for NaN/Inf
        check_density_positive: Check density is non-negative
        density_tolerance: Tolerance for density non-negativity

    Returns:
        ValidationResult

    Issue #688: Solver output validation
    """
    from mfg_pde.utils.validation.arrays import validate_finite, validate_non_negative

    result = ValidationResult()

    if check_finite:
        u_result = validate_finite(U, "U (value function)")
        m_result = validate_finite(M, "M (density)")
        result.issues.extend(u_result.issues)
        result.issues.extend(m_result.issues)
        if not u_result.is_valid or not m_result.is_valid:
            result.is_valid = False

    if check_density_positive:
        density_result = validate_non_negative(M, "M (density)", tolerance=-density_tolerance)
        result.issues.extend(density_result.issues)
        if not density_result.is_valid:
            result.is_valid = False

    return result
