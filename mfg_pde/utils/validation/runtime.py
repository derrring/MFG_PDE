"""
Runtime validation utilities (NaN/Inf detection, divergence monitoring).

This module provides utilities for validating solver outputs during
and after execution.

Issue #688: Runtime safety (NaN/Inf detection, divergence, bounds checking)
"""

from __future__ import annotations

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


# Note: DivergenceMonitor and ConvergenceMonitor intentionally NOT included here.
# Use mfg_pde.utils.convergence.convergence_monitors instead, which provides
# comprehensive MFG-specific monitoring: DistributionConvergenceMonitor,
# _ErrorHistoryTracker, ConvergenceWrapper, etc.


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
