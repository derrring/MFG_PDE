"""
Gradient Notation Utilities for Backward Compatibility.

This module provides conversion functions between legacy string-key gradient
notation and the standard tuple multi-index notation.

Standard Notation (tuple multi-index):
    derivs = {
        (1, 0): p_x,   # ∂u/∂x
        (0, 1): p_y,   # ∂u/∂y
    }

Legacy Notation (string keys):
    p_values = {
        "forward": p_forward,
        "backward": p_backward,
    }
    # or
    p_values = {"x": p_x, "y": p_y, "dx": p_x, "dy": p_y}

References:
    - Standard: docs/gradient_notation_standard.md
    - Audit: docs/GRADIENT_NOTATION_AUDIT_REPORT.md
    - Theory: docs/theory/foundations/NOTATION_STANDARDS.md

Import:
    from mfg_pde.compat.gradient_notation import ensure_tuple_notation
"""

from __future__ import annotations

import warnings

import numpy as np


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


def check_derivs_format(derivs: dict) -> str:
    """
    Check which gradient notation format is being used.

    Args:
        derivs: Dictionary with either tuple or string keys

    Returns:
        Format type: "tuple", "string_1d", "string_2d", or "unknown"

    Example:
        >>> derivs = {(1, 0): 0.5, (0, 1): 0.3}
        >>> check_derivs_format(derivs)
        'tuple'
    """
    if not derivs:
        return "unknown"

    first_key = next(iter(derivs.keys()))

    if isinstance(first_key, tuple):
        return "tuple"
    elif isinstance(first_key, str):
        if first_key in ("forward", "backward"):
            return "string_1d"
        elif first_key in ("x", "y", "z", "dx", "dy", "dz"):
            return "string_2d"
        else:
            return "unknown"
    else:
        return "unknown"


def ensure_tuple_notation(
    derivs: dict, dimension: int | None = None, u_value: float = 0.0
) -> dict[tuple[int, ...], float]:
    """
    Ensure derivatives use tuple notation, converting if necessary.

    Args:
        derivs: Dictionary with either tuple or string keys
        dimension: Spatial dimension (1, 2, or 3) - required for string_2d conversion
        u_value: Function value u (optional)

    Returns:
        Dictionary with tuple keys

    Raises:
        ValueError: If format is unknown or dimension is required but not provided

    Example:
        >>> derivs = {"forward": 0.6, "backward": 0.4}
        >>> result = ensure_tuple_notation(derivs, u_value=1.0)
        >>> result
        {(0,): 1.0, (1,): 0.5}
    """
    fmt = check_derivs_format(derivs)

    if fmt == "tuple":
        # Already in correct format
        return derivs

    elif fmt == "string_1d":
        # Convert from legacy 1D format
        return p_values_to_derivs_1d(derivs, u_value=u_value)

    elif fmt == "string_2d":
        # Convert from string keys like {"x": ..., "y": ...} or {"dx": ..., "dy": ...}
        if dimension is None:
            raise ValueError("dimension parameter required for string_2d conversion")

        warnings.warn(
            f"Converting from string-key gradient notation (keys: {list(derivs.keys())}). "
            "Please migrate to tuple notation. "
            "See docs/gradient_notation_standard.md for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Try multiple possible key naming conventions
        if dimension == 2:
            p_x = derivs.get("x", derivs.get("dx", 0.0))
            p_y = derivs.get("y", derivs.get("dy", 0.0))
            return {
                (0, 0): u_value,
                (1, 0): p_x,
                (0, 1): p_y,
            }
        elif dimension == 3:
            p_x = derivs.get("x", derivs.get("dx", 0.0))
            p_y = derivs.get("y", derivs.get("dy", 0.0))
            p_z = derivs.get("z", derivs.get("dz", 0.0))
            return {
                (0, 0, 0): u_value,
                (1, 0, 0): p_x,
                (0, 1, 0): p_y,
                (0, 0, 1): p_z,
            }
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

    else:
        raise ValueError(
            f"Unknown derivative format. Keys: {list(derivs.keys())}. "
            "Expected tuple keys like (1, 0) or string keys like 'forward'/'x'."
        )
