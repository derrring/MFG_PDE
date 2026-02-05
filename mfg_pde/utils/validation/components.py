"""
Validation for MFG problem components (IC/BC).

This module validates initial conditions (m_initial) and terminal conditions (u_final)
provided via MFGComponents.

Issues:
- #679: IC/BC Validation
- #681: Core IC/BC array and callable validation
- #682: Geometry-agnostic IC/BC validation
- #683: Mass normalization validation
- #684: Callable signature detection
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfg_pde.utils.validation.protocol import ValidationResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfg_pde.core.mfg_components import MFGComponents
    from mfg_pde.geometry.protocol import GeometryProtocol


def validate_components(
    components: MFGComponents | None,
    geometry: GeometryProtocol,
    *,
    require_m_initial: bool = True,
    require_u_final: bool = True,
    check_mass_normalization: bool = False,
) -> ValidationResult:
    """
    Validate MFGComponents against geometry.

    Args:
        components: MFGComponents to validate
        geometry: Geometry to validate against
        require_m_initial: Whether m_initial is required
        require_u_final: Whether u_final is required
        check_mass_normalization: Whether to verify integral of m_initial = 1

    Returns:
        ValidationResult with any issues found

    Example:
        result = validate_components(components, geometry)
        if not result.is_valid:
            raise ValidationError(result)
    """
    result = ValidationResult()

    if components is None:
        if require_m_initial or require_u_final:
            result.add_error(
                "MFGComponents is required but was None",
                location="components",
                suggestion="Pass components=MFGComponents(m_initial=..., u_final=...)",
            )
        return result

    # Validate m_initial
    if require_m_initial:
        m_result = validate_m_initial(components.m_initial, geometry)
        result.issues.extend(m_result.issues)
        if not m_result.is_valid:
            result.is_valid = False

    # Validate u_final
    if require_u_final:
        u_result = validate_u_final(components.u_final, geometry)
        result.issues.extend(u_result.issues)
        if not u_result.is_valid:
            result.is_valid = False

    # Check mass normalization if requested
    if check_mass_normalization and components.m_initial is not None:
        mass_result = validate_mass_normalization(components.m_initial, geometry)
        result.issues.extend(mass_result.issues)
        # Mass normalization is a warning, not an error
        # (doesn't invalidate the result)

    return result


def validate_m_initial(
    m_initial: Callable | NDArray[np.floating] | None,
    geometry: GeometryProtocol,
) -> ValidationResult:
    """
    Validate initial density m_initial.

    Checks:
    - Not None (required)
    - If callable: correct signature and return type
    - If array: correct shape matching geometry
    - Values are finite (no NaN/Inf)

    Args:
        m_initial: Initial density (callable or array)
        geometry: Geometry for shape validation

    Returns:
        ValidationResult

    Issue #681: Core IC/BC validation
    """
    result = ValidationResult()

    if m_initial is None:
        result.add_error(
            "m_initial (initial density) is required but was None",
            location="m_initial",
            suggestion="Provide m_initial=lambda x: ... or m_initial=np.array(...)",
        )
        return result

    # Validate based on type
    if callable(m_initial):
        return _validate_callable_ic(m_initial, geometry, "m_initial")
    elif isinstance(m_initial, np.ndarray):
        return _validate_array_ic(m_initial, geometry, "m_initial")
    else:
        result.add_error(
            f"m_initial must be callable or ndarray, got {type(m_initial).__name__}",
            location="m_initial",
        )

    return result


def validate_u_final(
    u_final: Callable | NDArray[np.floating] | None,
    geometry: GeometryProtocol,
) -> ValidationResult:
    """
    Validate terminal value function u_final.

    Checks:
    - Not None (required)
    - If callable: correct signature and return type
    - If array: correct shape matching geometry
    - Values are finite (no NaN/Inf)

    Args:
        u_final: Terminal value function (callable or array)
        geometry: Geometry for shape validation

    Returns:
        ValidationResult

    Issue #681: Core IC/BC validation
    """
    result = ValidationResult()

    if u_final is None:
        result.add_error(
            "u_final (terminal condition) is required but was None",
            location="u_final",
            suggestion="Provide u_final=lambda x: ... or u_final=np.array(...)",
        )
        return result

    # Validate based on type
    if callable(u_final):
        return _validate_callable_ic(u_final, geometry, "u_final")
    elif isinstance(u_final, np.ndarray):
        return _validate_array_ic(u_final, geometry, "u_final")
    else:
        result.add_error(
            f"u_final must be callable or ndarray, got {type(u_final).__name__}",
            location="u_final",
        )

    return result


def validate_mass_normalization(
    m_initial: Callable | NDArray[np.floating],
    geometry: GeometryProtocol,
    tolerance: float = 0.1,
) -> ValidationResult:
    """
    Validate that initial density integrates to 1.

    For probability density interpretation, we expect:
        integral of m_initial over domain = 1

    Args:
        m_initial: Initial density (callable or array)
        geometry: Geometry for integration
        tolerance: Relative tolerance for mass check

    Returns:
        ValidationResult with warning if mass != 1

    Issue #683: Mass normalization validation
    """
    result = ValidationResult()

    try:
        # Evaluate m_initial on grid if callable
        if callable(m_initial):
            grid = geometry.get_spatial_grid()
            if isinstance(grid, tuple):
                # Multi-dimensional grid
                from itertools import product

                coords = list(product(*grid))
                values = np.array([m_initial(c) for c in coords])
            else:
                values = np.array([m_initial(x) for x in grid])
        else:
            values = m_initial

        # Compute mass (simple sum * dx for now)
        # TODO: Use geometry.integrate() when available (Issue #682)
        total_mass = np.sum(values)

        # Normalize by grid spacing if available
        if hasattr(geometry, "dx"):
            dx = geometry.dx
            if isinstance(dx, (list, tuple, np.ndarray)):
                total_mass *= np.prod(dx)
            else:
                total_mass *= dx

        # Check if mass is approximately 1
        if abs(total_mass - 1.0) > tolerance:
            result.add_warning(
                f"Initial density mass = {total_mass:.4f}, expected 1.0",
                location="m_initial",
                suggestion="Normalize m_initial so that its integral equals 1",
            )
            result.context["computed_mass"] = total_mass

    except Exception as e:
        result.add_warning(
            f"Could not verify mass normalization: {e}",
            location="m_initial",
        )

    return result


def detect_callable_signature(
    func: Callable,
    name: str = "function",
) -> ValidationResult:
    """
    Detect and validate callable signature.

    Attempts to determine if the callable has signature:
    - f(x) - spatial only
    - f(x, t) or f(t, x) - spatiotemporal
    - f(x, y, ...) - multi-argument spatial

    Args:
        func: Callable to analyze
        name: Name for error messages

    Returns:
        ValidationResult with signature info in context

    Issue #684: Callable signature detection
    """
    import inspect

    result = ValidationResult()

    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        # Filter out *args, **kwargs
        positional_params = [
            p
            for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and p.default is inspect.Parameter.empty
        ]

        n_required = len(positional_params)
        param_names = [p.name for p in positional_params]

        result.context["n_required_args"] = n_required
        result.context["param_names"] = param_names

        # Infer signature type
        if n_required == 1:
            result.context["signature_type"] = "spatial"
        elif n_required == 2:
            # Check if one parameter is 't' or 'time'
            if any(n in ("t", "time") for n in param_names):
                result.context["signature_type"] = "spatiotemporal"
            else:
                result.context["signature_type"] = "multi_spatial"
        else:
            result.context["signature_type"] = "unknown"

    except (ValueError, TypeError) as e:
        result.add_warning(
            f"Could not inspect signature of {name}: {e}",
            location=name,
        )
        result.context["signature_type"] = "unknown"

    return result


# --- Internal helpers (geometry-agnostic, Issue #682) ---


def _get_validation_points(
    geometry: GeometryProtocol,
    n_samples: int = 3,
) -> np.ndarray:
    """Get sample points for callable validation from any geometry type.

    Uses ``get_collocation_points()`` (part of GeometryProtocol) which
    returns ``(N, d)`` for all geometries.  Interior points are preferred
    to avoid boundary edge cases.

    Args:
        geometry: Any geometry implementing GeometryProtocol.
        n_samples: Number of sample points to return.

    Returns:
        Array of shape ``(n_samples, d)`` with sample points.

    Raises:
        ValueError: If the geometry has no collocation points.

    Issue #682: Geometry-agnostic IC/BC validation.
    """
    points = geometry.get_collocation_points()  # (N, d) per protocol

    # Handle 1D case where points may be (N,) instead of (N, 1)
    if points.ndim == 1:
        points = points.reshape(-1, 1)

    n_total = len(points)
    if n_total == 0:
        raise ValueError(f"Geometry {type(geometry).__name__} returned 0 collocation points")

    if n_total <= n_samples:
        return points

    # Sample interior points (skip first/last to avoid boundary)
    indices = np.linspace(0, n_total - 1, n_samples + 2, dtype=int)[1:-1]
    return points[indices]


def _get_expected_ic_shape(
    geometry: GeometryProtocol,
) -> tuple[int, ...]:
    """Get expected IC/BC array shape for any geometry type.

    Uses ``get_grid_shape()`` (part of GeometryProtocol) with a fallback
    to ``(num_spatial_points,)`` if the geometry returns None.

    Returns:
        Expected shape tuple for IC/BC arrays.

    Issue #682: Geometry-agnostic IC/BC validation.
    """
    shape = geometry.get_grid_shape()
    if shape is not None:
        return shape
    # Fallback for geometries whose base class returns None
    return (geometry.num_spatial_points,)


def _validate_callable_ic(
    func: Callable,
    geometry: GeometryProtocol,
    name: str,
) -> ValidationResult:
    """Validate a callable initial/terminal condition.

    Uses ``_get_validation_points()`` to obtain a geometry-agnostic sample
    point, then ``adapt_ic_callable()`` to probe the callable's signature.
    If the callable matches any supported convention the validation passes.
    If no convention works, a detailed error lists every attempted signature
    and what went wrong.

    Issues #682, #684: Geometry-agnostic callable validation.
    """
    from mfg_pde.utils.callable_adapter import adapt_ic_callable

    result = ValidationResult()

    # Get a sample point from any geometry type (Issue #682)
    try:
        points = _get_validation_points(geometry, n_samples=1)
        sample_point = points[0]  # (d,) ndarray
    except Exception as e:
        result.add_error(
            f"Could not get sample point from geometry: {e}",
            location=name,
        )
        return result

    dimension = geometry.dimension

    # For 1D, the callable adapter expects a scalar float
    if dimension == 1:
        adapter_sample: float | np.ndarray = float(sample_point[0])
    else:
        adapter_sample = sample_point

    # Probe signature via adapter
    try:
        sig_type, adapted = adapt_ic_callable(
            func,
            dimension=dimension,
            sample_point=adapter_sample,
        )
    except TypeError as e:
        result.add_error(
            str(e),
            location=name,
            suggestion=f"Ensure {name} accepts the coordinate format from geometry",
        )
        return result

    # Store detected signature and geometry type in context (informational)
    result.context["adapted_signature"] = sig_type.name
    result.context["geometry_type"] = type(geometry).__name__

    # Evaluate the adapted callable at the sample point to check output quality
    try:
        if dimension == 1:
            value = adapted(adapter_sample)
        else:
            value = adapted(sample_point)
    except Exception as e:
        result.add_error(
            f"{name} callable raised exception at sample point {sample_point}: {e}",
            location=name,
            suggestion=f"Ensure {name} accepts the coordinate format from geometry",
        )
        return result

    # Check return type
    if not isinstance(value, (int, float, np.integer, np.floating, np.ndarray)):
        result.add_error(
            f"{name} must return numeric value, got {type(value).__name__}",
            location=name,
        )
        return result

    # Check for NaN/Inf
    if np.isscalar(value):
        if not np.isfinite(value):
            result.add_error(
                f"{name} returned non-finite value at sample point",
                location=name,
            )
    elif isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            result.add_error(
                f"{name} returned array with NaN or Inf values",
                location=name,
            )

    return result


def _validate_array_ic(
    arr: NDArray[np.floating],
    geometry: GeometryProtocol,
    name: str,
) -> ValidationResult:
    """Validate an array initial/terminal condition.

    Uses ``_get_expected_ic_shape()`` for geometry-agnostic shape inference.

    Issue #682: Geometry-agnostic array validation.
    """
    result = ValidationResult()

    # Get expected shape from any geometry type (Issue #682)
    try:
        expected_shape = _get_expected_ic_shape(geometry)
    except Exception as e:
        result.add_error(
            f"Could not get expected IC shape from geometry: {e}",
            location=name,
        )
        return result

    # Check shape
    if arr.shape != expected_shape:
        result.add_error(
            f"{name} has shape {arr.shape}, expected {expected_shape}",
            location=name,
            suggestion=(f"Reshape {name} to match geometry grid shape. Geometry type: {type(geometry).__name__}"),
        )
        return result

    # Check for NaN/Inf
    if not np.all(np.isfinite(arr)):
        n_nan = np.sum(np.isnan(arr))
        n_inf = np.sum(np.isinf(arr))
        result.add_error(
            f"{name} contains {n_nan} NaN and {n_inf} Inf values",
            location=name,
        )

    return result
