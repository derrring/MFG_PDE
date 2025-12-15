"""
Gradient Notation Utilities for Derivative Format Conversion.

.. deprecated:: 0.17.0
    This module is deprecated. Use :class:`mfg_pde.core.DerivativeTensors` instead.

    The new standard for derivative representation is the `DerivativeTensors` class
    which stores derivatives as NumPy tensors:
    - grad: shape (d,) for gradient ∇u
    - hess: shape (d, d) for Hessian ∇²u
    - etc.

    Migration example::

        # Old (deprecated)
        from mfg_pde.compat.gradient_notation import gradient_tuple_to_derivs
        derivs = gradient_tuple_to_derivs((0.5, 0.3))  # dict format
        p_x = derivs[(1, 0)]

        # New (recommended)
        from mfg_pde.core import DerivativeTensors
        derivs = DerivativeTensors.from_gradient(np.array([0.5, 0.3]))
        p_x = derivs.grad[0]  # or derivs[1][0]

    For one-time migration from legacy dict format::

        from mfg_pde.core.derivatives import from_multi_index_dict
        new_derivs = from_multi_index_dict(old_dict_derivs)

This module provides conversion functions between different derivative
representations used across MFG_PDE solvers. All functions in this module
are deprecated in favor of the DerivativeTensors class.

Legacy Standard Notation (tuple multi-index dict) - DEPRECATED:
    derivs = {
        (1, 0): p_x,   # ∂u/∂x
        (0, 1): p_y,   # ∂u/∂y
        (2, 0): p_xx,  # ∂²u/∂x²
    }

New Standard (DerivativeTensors) - RECOMMENDED:
    from mfg_pde.core import DerivativeTensors
    derivs = DerivativeTensors.from_arrays(
        grad=np.array([p_x, p_y]),
        hess=np.array([[p_xx, p_xy], [p_xy, p_yy]])
    )

See Also:
    - mfg_pde.core.DerivativeTensors: The new derivative tensor class
    - mfg_pde.core.derivatives.from_multi_index_dict: Convert legacy format
    - docs/NAMING_CONVENTIONS.md: Derivative Tensor Standard section
"""

from __future__ import annotations

import warnings

import numpy as np

# Module-level deprecation warning
warnings.warn(
    "mfg_pde.compat.gradient_notation is deprecated since v0.17.0. "
    "Use mfg_pde.core.DerivativeTensors instead. "
    "See docs/NAMING_CONVENTIONS.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)


def derivs_to_p_values_1d(derivs: dict[tuple[int], float]) -> dict[str, float]:
    """
    Convert tuple notation to legacy 1D string-key format.

    Args:
        derivs: Dictionary with tuple keys, e.g., {(1,): p}

    Returns:
        Dictionary with string keys {"forward": p, "backward": p}

    Example:
        >>> derivs = {(1,): 0.5}
        >>> p_values = derivs_to_p_values_1d(derivs)
        >>> p_values
        {'forward': 0.5, 'backward': 0.5}
    """
    warnings.warn(
        "Using legacy string-key gradient notation. "
        "Please migrate to tuple notation: derivs[(1,)] instead of p_values['forward']. "
        "See docs/gradient_notation_standard.md for details.",
        DeprecationWarning,
        stacklevel=2,
    )

    p = derivs.get((1,), 0.0)
    return {"forward": p, "backward": p}


def p_values_to_derivs_1d(p_values: dict[str, float], u_value: float = 0.0) -> dict[tuple[int], float]:
    """
    Convert legacy 1D string-key format to tuple notation.

    Args:
        p_values: Dictionary with string keys {"forward": ..., "backward": ...}
        u_value: Function value u (optional)

    Returns:
        Dictionary with tuple keys {(0,): u, (1,): p}

    Example:
        >>> p_values = {"forward": 0.6, "backward": 0.4}
        >>> derivs = p_values_to_derivs_1d(p_values, u_value=1.0)
        >>> derivs
        {(0,): 1.0, (1,): 0.5}
    """
    warnings.warn(
        "Converting from legacy string-key gradient notation. "
        "Please migrate to tuple notation. "
        "See docs/gradient_notation_standard.md for migration guide.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Use central difference approximation
    p_forward = p_values.get("forward", 0.0)
    p_backward = p_values.get("backward", 0.0)
    p_central = (p_forward + p_backward) / 2.0

    return {
        (0,): u_value,  # Function value
        (1,): p_central,  # First derivative (central difference)
    }


def gradient_tuple_to_derivs(grad: tuple[float, ...], u_value: float = 0.0) -> dict[tuple[int, ...], float]:
    """
    Convert gradient tuple to tuple-indexed derivatives dict.

    This bridges the Semi-Lagrangian solver's tuple format to the standard
    multi-index dict format used by FDM, GFDM, and WENO solvers.

    Args:
        grad: Gradient as tuple (p_x,) for 1D, (p_x, p_y) for 2D, etc.
        u_value: Function value u (optional)

    Returns:
        Dictionary with tuple multi-index keys

    Example:
        >>> grad = (0.5, 0.3)
        >>> derivs = gradient_tuple_to_derivs(grad)
        >>> derivs
        {(1, 0): 0.5, (0, 1): 0.3}

        >>> grad = (0.5,)
        >>> derivs = gradient_tuple_to_derivs(grad, u_value=1.0)
        >>> derivs
        {(0,): 1.0, (1,): 0.5}
    """
    d = len(grad)

    if d == 1:
        result = {(1,): float(grad[0])}
        if u_value != 0.0:
            result[(0,)] = u_value
        return result
    elif d == 2:
        result = {
            (1, 0): float(grad[0]),
            (0, 1): float(grad[1]),
        }
        if u_value != 0.0:
            result[(0, 0)] = u_value
        return result
    elif d == 3:
        result = {
            (1, 0, 0): float(grad[0]),
            (0, 1, 0): float(grad[1]),
            (0, 0, 1): float(grad[2]),
        }
        if u_value != 0.0:
            result[(0, 0, 0)] = u_value
        return result
    else:
        # General nD case
        result = {}
        if u_value != 0.0:
            result[tuple([0] * d)] = u_value
        for i in range(d):
            key = tuple(1 if j == i else 0 for j in range(d))
            result[key] = float(grad[i])
        return result


def derivs_to_gradient_tuple(derivs: dict[tuple[int, ...], float]) -> tuple[float, ...]:
    """
    Extract gradient as tuple from tuple-indexed derivatives dict.

    Inverse of gradient_tuple_to_derivs(). Useful for passing derivatives
    to legacy Semi-Lagrangian code that expects tuple format.

    Args:
        derivs: Dictionary with tuple multi-index keys

    Returns:
        Gradient as tuple (p_x,) for 1D, (p_x, p_y) for 2D, etc.

    Example:
        >>> derivs = {(1, 0): 0.5, (0, 1): 0.3}
        >>> grad = derivs_to_gradient_tuple(derivs)
        >>> grad
        (0.5, 0.3)
    """
    # Detect dimension from keys
    if not derivs:
        return ()

    # Find first-order derivative keys to determine dimension
    first_order_keys = [k for k in derivs if sum(k) == 1]
    if not first_order_keys:
        return ()

    d = len(first_order_keys[0])

    if d == 1:
        return (derivs.get((1,), 0.0),)
    elif d == 2:
        return (derivs.get((1, 0), 0.0), derivs.get((0, 1), 0.0))
    elif d == 3:
        return (
            derivs.get((1, 0, 0), 0.0),
            derivs.get((0, 1, 0), 0.0),
            derivs.get((0, 0, 1), 0.0),
        )
    else:
        # General nD case
        result = []
        for i in range(d):
            key = tuple(1 if j == i else 0 for j in range(d))
            result.append(derivs.get(key, 0.0))
        return tuple(result)


def derivs_to_gradient_array(derivs: dict[tuple[int, ...], float], dimension: int) -> np.ndarray:
    """
    Extract gradient as NumPy array from tuple-indexed derivatives.

    Args:
        derivs: Dictionary with tuple keys
        dimension: Spatial dimension (1, 2, or 3)

    Returns:
        Gradient as ndarray of shape (dimension,)

    Example:
        >>> derivs = {(1, 0): 0.5, (0, 1): 0.3}
        >>> p = derivs_to_gradient_array(derivs, dimension=2)
        >>> p
        array([0.5, 0.3])
    """
    if dimension == 1:
        p = derivs.get((1,), 0.0)
        return np.array([p])
    elif dimension == 2:
        p_x = derivs.get((1, 0), 0.0)
        p_y = derivs.get((0, 1), 0.0)
        return np.array([p_x, p_y])
    elif dimension == 3:
        p_x = derivs.get((1, 0, 0), 0.0)
        p_y = derivs.get((0, 1, 0), 0.0)
        p_z = derivs.get((0, 0, 1), 0.0)
        return np.array([p_x, p_y, p_z])
    else:
        raise ValueError(f"Unsupported dimension: {dimension}. Must be 1, 2, or 3.")


def gradient_array_to_derivs(p: np.ndarray, u_value: float = 0.0) -> dict[tuple[int, ...], float]:
    """
    Convert gradient array to tuple-indexed derivatives.

    Args:
        p: Gradient as ndarray of shape (d,) where d is dimension
        u_value: Function value u (optional)

    Returns:
        Dictionary with tuple keys

    Example:
        >>> p = np.array([0.5, 0.3])
        >>> derivs = gradient_array_to_derivs(p, u_value=1.0)
        >>> derivs
        {(0, 0): 1.0, (1, 0): 0.5, (0, 1): 0.3}
    """
    d = len(p)

    if d == 1:
        return {
            (0,): u_value,
            (1,): float(p[0]),
        }
    elif d == 2:
        return {
            (0, 0): u_value,
            (1, 0): float(p[0]),
            (0, 1): float(p[1]),
        }
    elif d == 3:
        return {
            (0, 0, 0): u_value,
            (1, 0, 0): float(p[0]),
            (0, 1, 0): float(p[1]),
            (0, 0, 1): float(p[2]),
        }
    else:
        raise ValueError(f"Unsupported dimension: {d}. Must be 1, 2, or 3.")


def check_derivs_format(derivs: dict | tuple | np.ndarray | None) -> str:
    """
    Check which gradient notation format is being used.

    Args:
        derivs: Derivatives in any supported format

    Returns:
        Format type: "dict_tuple", "dict_string_1d", "dict_string_2d",
                     "plain_tuple", "array", or "unknown"

    Example:
        >>> check_derivs_format({(1, 0): 0.5, (0, 1): 0.3})
        'dict_tuple'
        >>> check_derivs_format((0.5, 0.3))
        'plain_tuple'
        >>> check_derivs_format(np.array([0.5, 0.3]))
        'array'
    """
    if derivs is None:
        return "unknown"

    # Plain tuple of floats (Semi-Lagrangian format)
    if isinstance(derivs, tuple) and len(derivs) > 0 and isinstance(derivs[0], (int, float)):
        return "plain_tuple"

    # NumPy array
    if isinstance(derivs, np.ndarray):
        return "array"

    # Dictionary formats
    if isinstance(derivs, dict):
        if not derivs:
            return "unknown"

        first_key = next(iter(derivs.keys()))

        if isinstance(first_key, tuple):
            return "dict_tuple"
        elif isinstance(first_key, str):
            if first_key in ("forward", "backward"):
                return "dict_string_1d"
            elif first_key in ("x", "y", "z", "dx", "dy", "dz"):
                return "dict_string_2d"

    return "unknown"


def normalize_derivs(
    derivs: dict | tuple | np.ndarray | None,
    dimension: int | None = None,
    u_value: float = 0.0,
) -> dict[tuple[int, ...], float]:
    """
    Normalize any derivative format to standard tuple multi-index dict.

    This is the primary conversion function that handles all formats:
    - dict with tuple keys: returned as-is (already standard)
    - dict with string keys: converted with deprecation warning
    - plain tuple (p_x, p_y): converted via gradient_tuple_to_derivs
    - np.ndarray: converted via gradient_array_to_derivs

    Args:
        derivs: Derivatives in any supported format
        dimension: Spatial dimension (required for some conversions)
        u_value: Function value u (optional)

    Returns:
        Dictionary with tuple multi-index keys

    Example:
        >>> normalize_derivs((0.5, 0.3))
        {(1, 0): 0.5, (0, 1): 0.3}

        >>> normalize_derivs({(1, 0): 0.5})
        {(1, 0): 0.5}

        >>> normalize_derivs(np.array([0.5, 0.3]))
        {(0, 0): 0.0, (1, 0): 0.5, (0, 1): 0.3}
    """
    if derivs is None:
        return {}

    fmt = check_derivs_format(derivs)

    if fmt == "dict_tuple":
        # Already in standard format
        return derivs

    elif fmt == "plain_tuple":
        # Semi-Lagrangian format: (p_x, p_y) → {(1,0): p_x, (0,1): p_y}
        return gradient_tuple_to_derivs(derivs, u_value=u_value)

    elif fmt == "array":
        # NumPy array format
        return gradient_array_to_derivs(derivs, u_value=u_value)

    elif fmt == "dict_string_1d":
        # Legacy 1D string keys
        return p_values_to_derivs_1d(derivs, u_value=u_value)

    elif fmt == "dict_string_2d":
        # Legacy 2D string keys
        if dimension is None:
            raise ValueError("dimension parameter required for string_2d conversion")

        warnings.warn(
            f"Converting from string-key gradient notation (keys: {list(derivs.keys())}). "
            "Please migrate to tuple notation. "
            "See docs/NAMING_CONVENTIONS.md for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )

        if dimension == 2:
            p_x = derivs.get("x", derivs.get("dx", 0.0))
            p_y = derivs.get("y", derivs.get("dy", 0.0))
            result = {(1, 0): p_x, (0, 1): p_y}
            if u_value != 0.0:
                result[(0, 0)] = u_value
            return result
        elif dimension == 3:
            p_x = derivs.get("x", derivs.get("dx", 0.0))
            p_y = derivs.get("y", derivs.get("dy", 0.0))
            p_z = derivs.get("z", derivs.get("dz", 0.0))
            result = {(1, 0, 0): p_x, (0, 1, 0): p_y, (0, 0, 1): p_z}
            if u_value != 0.0:
                result[(0, 0, 0)] = u_value
            return result
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

    else:
        raise ValueError(
            f"Unknown derivative format: {type(derivs)}. Expected dict with tuple keys, plain tuple, or ndarray."
        )


def ensure_tuple_notation(
    derivs: dict | tuple | np.ndarray | None,
    dimension: int | None = None,
    u_value: float = 0.0,
) -> dict[tuple[int, ...], float]:
    """
    Ensure derivatives use tuple notation, converting if necessary.

    Alias for normalize_derivs() for backward compatibility.

    Args:
        derivs: Derivatives in any supported format
        dimension: Spatial dimension (required for some conversions)
        u_value: Function value u (optional)

    Returns:
        Dictionary with tuple multi-index keys

    Example:
        >>> ensure_tuple_notation((0.5, 0.3))
        {(1, 0): 0.5, (0, 1): 0.3}

        >>> ensure_tuple_notation({"forward": 0.6, "backward": 0.4}, u_value=1.0)
        {(0,): 1.0, (1,): 0.5}
    """
    return normalize_derivs(derivs, dimension=dimension, u_value=u_value)
