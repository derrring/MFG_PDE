"""
Validation for custom MFG functions (Hamiltonian, drift, running_cost).

This module validates user-provided mathematical functions to catch
errors before they propagate into solver iterations.

Issue #686: Custom function validation (Hamiltonian, drift, running_cost)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.utils.validation.protocol import ValidationResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfg_pde.geometry.protocol import GeometryProtocol


def _get_sample_inputs(
    geometry: GeometryProtocol,
    location: str,
) -> tuple[np.ndarray, np.ndarray, float, int, ValidationResult | None]:
    """Extract a sample point (x, p, m) from geometry for validation.

    Returns:
        (x_sample, p_sample, m_sample, dimension, error_result)
        If error_result is not None, the caller should return it immediately.
    """
    try:
        grid = geometry.get_spatial_grid()
        if isinstance(grid, np.ndarray) and grid.ndim == 2:
            # (N, d) array from TensorProductGrid, ImplicitDomain, etc.
            mid = grid.shape[0] // 2
            x_sample = grid[mid]  # shape (d,)
            dimension = grid.shape[1]
        elif isinstance(grid, np.ndarray) and grid.ndim == 1:
            # 1D flat grid
            x_sample = grid[len(grid) // 2]
            dimension = 1
        elif isinstance(grid, (list, tuple)):
            # Legacy: tuple of 1D arrays (meshgrid-style)
            x_sample = np.array([g[len(g) // 2] for g in grid])
            dimension = len(grid)
        else:
            x_sample = np.atleast_1d(grid)[len(np.atleast_1d(grid)) // 2]
            dimension = 1
    except Exception as e:
        result = ValidationResult()
        result.add_error(
            f"Could not get sample point from geometry: {e}",
            location=location,
        )
        return np.array([]), np.array([]), 1.0, 1, result

    x_sample = np.atleast_1d(x_sample).astype(float)
    p_sample = np.zeros(dimension, dtype=float)
    m_sample = 1.0
    return x_sample, p_sample, m_sample, dimension, None


def validate_custom_functions(
    hamiltonian: Any | None,
    dH_dm: Callable | None,
    dH_dp: Callable | None,
    geometry: GeometryProtocol,
    *,
    check_consistency: bool = False,
) -> ValidationResult:
    """
    Validate all custom Hamiltonian-related functions.

    Supports HamiltonianBase instances (preferred) and raw callables.

    Args:
        hamiltonian: HamiltonianBase instance or callable H(x, m, p, t)
        dH_dm: Derivative dH/dm (bound method or callable(x, m, p, t))
        dH_dp: Derivative dH/dp (bound method or callable(x, m, p, t))
        geometry: Geometry for sample point generation
        check_consistency: If True, verify dH_dm/dH_dp are consistent with H

    Returns:
        ValidationResult with any issues found

    Issue #686: Custom function validation
    """
    result = ValidationResult()

    if hamiltonian is not None:
        h_result = validate_hamiltonian(hamiltonian, geometry)
        result.issues.extend(h_result.issues)
        if not h_result.is_valid:
            result.is_valid = False

    if dH_dm is not None:
        dm_result = validate_hamiltonian_derivative(dH_dm, geometry, "dH_dm")
        result.issues.extend(dm_result.issues)
        if not dm_result.is_valid:
            result.is_valid = False

    if dH_dp is not None:
        dp_result = validate_hamiltonian_derivative(dH_dp, geometry, "dH_dp")
        result.issues.extend(dp_result.issues)
        if not dp_result.is_valid:
            result.is_valid = False

    # Check consistency if requested and all functions provided
    if check_consistency and hamiltonian is not None and dH_dm is not None:
        cons_result = validate_hamiltonian_consistency(hamiltonian, dH_dm, geometry, dH_dp=dH_dp)
        result.issues.extend(cons_result.issues)

    return result


def validate_hamiltonian(
    hamiltonian: Any,
    geometry: GeometryProtocol,
) -> ValidationResult:
    """
    Validate Hamiltonian function H(x, m, p, t).

    Supports HamiltonianBase instances (called as H(x, m, p, t))
    and raw callables (tried with same signature).

    Checks:
    - Callable with correct signature
    - Returns float or array
    - No NaN/Inf in output

    Args:
        hamiltonian: HamiltonianBase instance or callable
        geometry: Geometry for sample point

    Returns:
        ValidationResult
    """
    result = ValidationResult()

    x_sample, p_sample, m_sample, _dim, err = _get_sample_inputs(geometry, "hamiltonian")
    if err is not None:
        return err

    # Evaluate: HamiltonianBase.__call__ signature is (x, m, p, t=0.0)
    try:
        value = hamiltonian(x_sample, m_sample, p_sample, 0.0)
    except TypeError as e:
        result.add_error(
            f"Hamiltonian has wrong signature: {e}",
            location="hamiltonian",
            suggestion="Hamiltonian should have signature H(x, m, p, t)",
        )
        return result
    except Exception as e:
        result.add_error(
            f"Hamiltonian raised exception: {e}",
            location="hamiltonian",
        )
        return result

    # Check return type
    if not isinstance(value, (int, float, np.integer, np.floating, np.ndarray)):
        result.add_error(
            f"Hamiltonian must return float or ndarray, got {type(value).__name__}",
            location="hamiltonian",
        )
        return result

    # Check for NaN/Inf
    if np.isscalar(value):
        if not np.isfinite(value):
            result.add_error(
                "Hamiltonian returned NaN or Inf at sample point",
                location="hamiltonian",
            )
    elif isinstance(value, np.ndarray) and not np.all(np.isfinite(value)):
        result.add_error(
            "Hamiltonian returned array with NaN or Inf values",
            location="hamiltonian",
        )

    return result


def validate_hamiltonian_derivative(
    derivative_func: Callable,
    geometry: GeometryProtocol,
    name: str,
) -> ValidationResult:
    """
    Validate a Hamiltonian derivative function (dH_dm or dH_dp).

    The derivative should have signature f(x, m, p, t) matching
    HamiltonianBase.dm() / HamiltonianBase.dp().

    Args:
        derivative_func: Derivative function (bound method or callable)
        geometry: Geometry for sample point
        name: Name for error messages ("dH_dm" or "dH_dp")

    Returns:
        ValidationResult
    """
    result = ValidationResult()

    x_sample, p_sample, m_sample, _dim, err = _get_sample_inputs(geometry, name)
    if err is not None:
        return err

    # Evaluate: derivative signature is (x, m, p, t=0.0)
    try:
        value = derivative_func(x_sample, m_sample, p_sample, 0.0)
    except TypeError as e:
        result.add_error(
            f"{name} has wrong signature: {e}",
            location=name,
            suggestion=f"{name} should have signature {name}(x, m, p, t)",
        )
        return result
    except Exception as e:
        result.add_error(
            f"{name} raised exception: {e}",
            location=name,
        )
        return result

    # Check return type
    if not isinstance(value, (int, float, np.integer, np.floating, np.ndarray)):
        result.add_error(
            f"{name} must return float or ndarray, got {type(value).__name__}",
            location=name,
        )

    return result


def validate_hamiltonian_consistency(
    hamiltonian: Any,
    dH_dm: Callable,
    geometry: GeometryProtocol,
    tolerance: float = 1e-4,
    dH_dp: Callable | None = None,
) -> ValidationResult:
    """
    Check if dH_dm and dH_dp are consistent with H using finite differences.

    Numerical checks:
        dH_dm_numerical = (H(x, m+eps, p, t) - H(x, m-eps, p, t)) / (2*eps)
        dH_dp_numerical[i] = (H(x, m, p+eps*e_i, t) - H(x, m, p-eps*e_i, t)) / (2*eps)

    Args:
        hamiltonian: HamiltonianBase instance or callable H(x, m, p, t)
        dH_dm: Claimed derivative dH/dm with signature (x, m, p, t)
        geometry: Geometry for sample point
        tolerance: Relative tolerance for consistency check
        dH_dp: Claimed gradient dH/dp with signature (x, m, p, t). Optional.

    Returns:
        ValidationResult with warning if inconsistent
    """
    result = ValidationResult()

    try:
        x_sample, p_sample, m_sample, dimension, err = _get_sample_inputs(geometry, "hamiltonian")
        if err is not None:
            return err

        eps = 1e-6

        # --- Check dH_dm consistency ---
        H_plus = hamiltonian(x_sample, m_sample + eps, p_sample, 0.0)
        H_minus = hamiltonian(x_sample, m_sample - eps, p_sample, 0.0)
        dH_dm_numerical = (float(H_plus) - float(H_minus)) / (2 * eps)

        dH_dm_analytical = float(dH_dm(x_sample, m_sample, p_sample, 0.0))

        diff = abs(dH_dm_numerical - dH_dm_analytical)
        scale = max(abs(dH_dm_analytical), 1e-10)

        if diff / scale > tolerance:
            result.add_warning(
                f"dH_dm may be inconsistent with H: numerical={dH_dm_numerical:.6f}, analytical={dH_dm_analytical:.6f}",
                location="dH_dm",
                suggestion="Verify dH_dm is the correct derivative of H with respect to m",
            )
            result.context["dH_dm_numerical"] = dH_dm_numerical
            result.context["dH_dm_analytical"] = dH_dm_analytical

        # --- Check dH_dp consistency (per component) ---
        if dH_dp is not None:
            dp_analytical = np.atleast_1d(dH_dp(x_sample, m_sample, p_sample, 0.0)).astype(float)

            for i in range(dimension):
                p_plus = p_sample.copy()
                p_minus = p_sample.copy()
                p_plus[i] += eps
                p_minus[i] -= eps

                H_p_plus = float(hamiltonian(x_sample, m_sample, p_plus, 0.0))
                H_p_minus = float(hamiltonian(x_sample, m_sample, p_minus, 0.0))
                dp_numerical_i = (H_p_plus - H_p_minus) / (2 * eps)

                dp_diff = abs(dp_numerical_i - dp_analytical[i])
                dp_scale = max(abs(dp_analytical[i]), 1e-10)

                if dp_diff / dp_scale > tolerance:
                    result.add_warning(
                        f"dH_dp[{i}] may be inconsistent with H: numerical={dp_numerical_i:.6f}, analytical={dp_analytical[i]:.6f}",
                        location="dH_dp",
                        suggestion="Verify dH_dp is the correct gradient of H with respect to p",
                    )
                    result.context[f"dH_dp_{i}_numerical"] = dp_numerical_i
                    result.context[f"dH_dp_{i}_analytical"] = float(dp_analytical[i])

    except Exception as e:
        result.add_warning(
            f"Could not verify Hamiltonian consistency: {e}",
            location="hamiltonian",
        )

    return result


def validate_drift(
    drift: Callable,
    geometry: GeometryProtocol,
) -> ValidationResult:
    """
    Validate drift function for FP equation.

    The drift should have signature drift(x, m) or drift(t, x, m)
    and return a vector of the same dimension as x.

    Args:
        drift: Drift function
        geometry: Geometry for sample point

    Returns:
        ValidationResult
    """
    result = ValidationResult()

    x_sample, _p_sample, _m_sample, dimension, err = _get_sample_inputs(geometry, "drift")
    if err is not None:
        return err

    m_sample = np.ones(dimension if dimension > 1 else 1)

    # Try different signatures
    value = None
    for args in [(x_sample, m_sample), (0.0, x_sample, m_sample)]:
        try:
            value = drift(*args)
            break
        except TypeError:
            continue

    if value is None:
        result.add_error(
            "Drift has wrong signature",
            location="drift",
            suggestion="Drift should have signature drift(x, m) or drift(t, x, m)",
        )
        return result

    # Check return shape
    if isinstance(value, np.ndarray):
        if value.shape != (dimension,) and value.shape != ():
            result.add_warning(
                f"Drift returned shape {value.shape}, expected ({dimension},)",
                location="drift",
            )

    return result


def validate_running_cost(
    running_cost: Callable,
    geometry: GeometryProtocol,
) -> ValidationResult:
    """
    Validate running cost function.

    The running cost should have signature f(x, m) or f(t, x, m)
    and return a scalar.

    Args:
        running_cost: Running cost function
        geometry: Geometry for sample point

    Returns:
        ValidationResult
    """
    result = ValidationResult()

    x_sample, _p_sample, _m_sample, _dimension, err = _get_sample_inputs(geometry, "running_cost")
    if err is not None:
        return err

    m_sample = 1.0

    # Try different signatures
    value = None
    for args in [(x_sample, m_sample), (0.0, x_sample, m_sample)]:
        try:
            value = running_cost(*args)
            break
        except TypeError:
            continue

    if value is None:
        result.add_error(
            "Running cost has wrong signature",
            location="running_cost",
            suggestion="Running cost should have signature f(x, m) or f(t, x, m)",
        )
        return result

    # Check return type
    if not np.isscalar(value):
        result.add_warning(
            f"Running cost should return scalar, got {type(value).__name__}",
            location="running_cost",
        )

    return result
