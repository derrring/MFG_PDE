"""
Validation for custom MFG functions (Hamiltonian, drift, running_cost).

This module validates user-provided mathematical functions to catch
errors before they propagate into solver iterations.

Issue #686: Custom function validation (Hamiltonian, drift, running_cost)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfg_pde.utils.validation.protocol import ValidationResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfg_pde.geometry.protocol import GeometryProtocol


def validate_custom_functions(
    hamiltonian: Callable | None,
    dH_dm: Callable | None,
    dH_dp: Callable | None,
    geometry: GeometryProtocol,
    *,
    check_consistency: bool = False,
) -> ValidationResult:
    """
    Validate all custom Hamiltonian-related functions.

    Args:
        hamiltonian: H(x, p, m) function
        dH_dm: Derivative dH/dm
        dH_dp: Derivative dH/dp (optimal control)
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
    if check_consistency and all([hamiltonian, dH_dm]):
        cons_result = validate_hamiltonian_consistency(hamiltonian, dH_dm, geometry)
        result.issues.extend(cons_result.issues)

    return result


def validate_hamiltonian(
    hamiltonian: Callable,
    geometry: GeometryProtocol,
) -> ValidationResult:
    """
    Validate Hamiltonian function H(x, p, m).

    Checks:
    - Callable with correct signature
    - Returns float or array
    - No NaN/Inf in output

    Args:
        hamiltonian: H(x, p, m) function
        geometry: Geometry for sample point

    Returns:
        ValidationResult
    """
    result = ValidationResult()

    # Get sample inputs
    try:
        grid = geometry.get_spatial_grid()
        if isinstance(grid, tuple):
            x_sample = tuple(g[len(g) // 2] for g in grid)
            dimension = len(grid)
        else:
            x_sample = grid[len(grid) // 2]
            dimension = 1
    except Exception as e:
        result.add_error(
            f"Could not get sample point from geometry: {e}",
            location="hamiltonian",
        )
        return result

    p_sample = np.zeros(dimension)
    m_sample = 1.0

    # Try to evaluate
    try:
        value = hamiltonian(x_sample, p_sample, m_sample)
    except TypeError as e:
        result.add_error(
            f"Hamiltonian has wrong signature: {e}",
            location="hamiltonian",
            suggestion="Hamiltonian should have signature H(x, p, m)",
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

    Args:
        derivative_func: Derivative function
        geometry: Geometry for sample point
        name: Name for error messages ("dH_dm" or "dH_dp")

    Returns:
        ValidationResult
    """
    result = ValidationResult()

    # Get sample inputs
    try:
        grid = geometry.get_spatial_grid()
        if isinstance(grid, tuple):
            x_sample = tuple(g[len(g) // 2] for g in grid)
            dimension = len(grid)
        else:
            x_sample = grid[len(grid) // 2]
            dimension = 1
    except Exception as e:
        result.add_error(
            f"Could not get sample point from geometry: {e}",
            location=name,
        )
        return result

    p_sample = np.zeros(dimension)
    m_sample = 1.0

    # Try to evaluate
    try:
        value = derivative_func(x_sample, p_sample, m_sample)
    except TypeError as e:
        result.add_error(
            f"{name} has wrong signature: {e}",
            location=name,
            suggestion=f"{name} should have signature {name}(x, p, m)",
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
    hamiltonian: Callable,
    dH_dm: Callable,
    geometry: GeometryProtocol,
    tolerance: float = 1e-4,
) -> ValidationResult:
    """
    Check if dH_dm is consistent with H using finite differences.

    This is a numerical check that computes:
        dH_dm_numerical = (H(x, p, m+eps) - H(x, p, m-eps)) / (2*eps)

    And compares with the provided dH_dm.

    Args:
        hamiltonian: H(x, p, m) function
        dH_dm: Claimed derivative dH/dm
        geometry: Geometry for sample point
        tolerance: Relative tolerance for consistency check

    Returns:
        ValidationResult with warning if inconsistent
    """
    result = ValidationResult()

    try:
        # Get sample inputs
        grid = geometry.get_spatial_grid()
        if isinstance(grid, tuple):
            x_sample = tuple(g[len(g) // 2] for g in grid)
            dimension = len(grid)
        else:
            x_sample = grid[len(grid) // 2]
            dimension = 1

        p_sample = np.zeros(dimension)
        m_sample = 1.0
        eps = 1e-6

        # Compute numerical derivative
        H_plus = hamiltonian(x_sample, p_sample, m_sample + eps)
        H_minus = hamiltonian(x_sample, p_sample, m_sample - eps)
        dH_dm_numerical = (H_plus - H_minus) / (2 * eps)

        # Compute analytical derivative
        dH_dm_analytical = dH_dm(x_sample, p_sample, m_sample)

        # Compare
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

    try:
        grid = geometry.get_spatial_grid()
        if isinstance(grid, tuple):
            x_sample = tuple(g[len(g) // 2] for g in grid)
            dimension = len(grid)
        else:
            x_sample = grid[len(grid) // 2]
            dimension = 1
    except Exception as e:
        result.add_error(f"Could not get sample point: {e}", location="drift")
        return result

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

    try:
        grid = geometry.get_spatial_grid()
        if isinstance(grid, tuple):
            x_sample = tuple(g[len(g) // 2] for g in grid)
        else:
            x_sample = grid[len(grid) // 2]
    except Exception as e:
        result.add_error(f"Could not get sample point: {e}", location="running_cost")
        return result

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
