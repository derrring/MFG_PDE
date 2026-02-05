"""
Callable Signature Detection and Adaptation for IC/BC functions.

Provides automatic detection and wrapping of user-provided callables (m_initial, u_final)
so that different signature conventions work transparently.

Supported signatures:
    - f(x) where x is scalar float (1D convention)
    - f(x) where x is ndarray of shape (d,)
    - f(x, t) spatiotemporal with time as second argument
    - f(t, x) spatiotemporal with time as first argument
    - f(x, y) expanded 2D coordinates (deprecated)
    - f(x, y, z) expanded 3D coordinates (deprecated)

The probing approach (try/except cascade) is preferred over inspect.signature()
because it handles lambdas, functools.partial, C extensions, and decorated functions
reliably. This follows the same pattern as validate_drift() in validation/functions.py.

Issue #684: Callable signature detection and adaptation.

Example:
    >>> from mfg_pde.utils.callable_adapter import adapt_ic_callable
    >>>
    >>> # User provides array-expecting callable for 1D problem
    >>> m_initial = lambda x: np.exp(-x[0]**2)
    >>>
    >>> sig, adapted = adapt_ic_callable(m_initial, dimension=1, sample_point=0.5)
    >>> # adapted(0.5) now works even though original expected x[0]
"""

from __future__ import annotations

import warnings
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


class CallableSignature(Enum):
    """Detected signature type for an IC/BC callable."""

    SPATIAL_SCALAR = auto()  # f(x) where x is scalar float (1D)
    SPATIAL_ARRAY = auto()  # f(x) where x is ndarray (d,)
    SPATIOTEMPORAL_XT = auto()  # f(x, t)
    SPATIOTEMPORAL_TX = auto()  # f(t, x)
    EXPANDED_2D = auto()  # f(x, y) -- deprecated
    EXPANDED_3D = auto()  # f(x, y, z) -- deprecated
    UNKNOWN = auto()


def adapt_ic_callable(
    func: Callable,
    dimension: int,
    sample_point: float | np.ndarray,
    *,
    time_value: float = 0.0,
) -> tuple[CallableSignature, Callable]:
    """
    Detect a callable's expected signature and return a normalized wrapper.

    The returned wrapper accepts the calling convention used by
    ``_setup_custom_initial_density()`` / ``_setup_custom_final_value()``:
    - 1D: ``wrapper(x_scalar)`` where x_scalar is a Python float
    - nD: ``wrapper(x_array)`` where x_array is ndarray of shape (d,)

    Args:
        func: User-provided IC/BC callable.
        dimension: Spatial dimension of the problem (1, 2, 3, ...).
        sample_point: A representative point to probe the callable.
            In 1D this is a float; in nD an ndarray of shape (d,).
        time_value: Time value for spatiotemporal wrappers (0.0 for m_initial,
            T for u_final).

    Returns:
        Tuple of (detected_signature, wrapped_callable).

    Raises:
        TypeError: If no supported signature convention works.
    """
    attempts: list[tuple[str, str]] = []

    # --- 1D probes ---
    if dimension == 1:
        scalar_sample = float(sample_point) if not isinstance(sample_point, float) else sample_point
        array_sample = np.array([scalar_sample])

        # Probe 1: f(scalar) -- most common 1D convention
        result, err = _try_call(func, scalar_sample)
        if err is None and _is_valid_output(result):
            return CallableSignature.SPATIAL_SCALAR, func
        attempts.append(("f(x) with x=float", _err_str(err, result)))

        # Probe 2: f(ndarray([x])) -- array-expecting 1D
        result, err = _try_call(func, array_sample)
        if err is None and _is_valid_output(result):
            # Wrap: convert scalar -> array for the user's function
            def _array_wrapper_1d(x: float, _fn: Callable = func) -> float:
                return float(_fn(np.array([x])))

            return CallableSignature.SPATIAL_ARRAY, _array_wrapper_1d
        attempts.append(("f(x) with x=ndarray([x])", _err_str(err, result)))

        # Probe 3: f(scalar, t) -- spatiotemporal (x, t)
        result, err = _try_call(func, scalar_sample, time_value)
        if err is None and _is_valid_output(result):
            tv = time_value

            def _xt_wrapper_1d(x: float, _fn: Callable = func, _t: float = tv) -> float:
                return float(_fn(x, _t))

            return CallableSignature.SPATIOTEMPORAL_XT, _xt_wrapper_1d
        attempts.append((f"f(x, t) with x=float, t={time_value}", _err_str(err, result)))

        # Probe 4: f(t, scalar) -- spatiotemporal (t, x)
        result, err = _try_call(func, time_value, scalar_sample)
        if err is None and _is_valid_output(result):
            tv = time_value

            def _tx_wrapper_1d(x: float, _fn: Callable = func, _t: float = tv) -> float:
                return float(_fn(_t, x))

            return CallableSignature.SPATIOTEMPORAL_TX, _tx_wrapper_1d
        attempts.append((f"f(t, x) with t={time_value}, x=float", _err_str(err, result)))

    # --- nD probes ---
    else:
        if isinstance(sample_point, np.ndarray):
            array_sample = sample_point
        else:
            array_sample = np.atleast_1d(sample_point)

        # Probe 1: f(ndarray) -- standard nD convention
        result, err = _try_call(func, array_sample)
        if err is None and _is_valid_output(result):
            return CallableSignature.SPATIAL_ARRAY, func
        attempts.append(("f(x) with x=ndarray", _err_str(err, result)))

        # Probe 2: f(ndarray, t) -- spatiotemporal (x, t)
        result, err = _try_call(func, array_sample, time_value)
        if err is None and _is_valid_output(result):
            tv = time_value

            def _xt_wrapper_nd(x: np.ndarray, _fn: Callable = func, _t: float = tv) -> float:
                return float(_fn(x, _t))

            return CallableSignature.SPATIOTEMPORAL_XT, _xt_wrapper_nd
        attempts.append((f"f(x, t) with x=ndarray, t={time_value}", _err_str(err, result)))

        # Probe 3: f(t, ndarray) -- spatiotemporal (t, x)
        result, err = _try_call(func, time_value, array_sample)
        if err is None and _is_valid_output(result):
            tv = time_value

            def _tx_wrapper_nd(x: np.ndarray, _fn: Callable = func, _t: float = tv) -> float:
                return float(_fn(_t, x))

            return CallableSignature.SPATIOTEMPORAL_TX, _tx_wrapper_nd
        attempts.append((f"f(t, x) with t={time_value}, x=ndarray", _err_str(err, result)))

        # Probe 4: f(*components) -- expanded coordinates (deprecated)
        if dimension == 2 and array_sample.shape == (2,):
            result, err = _try_call(func, float(array_sample[0]), float(array_sample[1]))
            if err is None and _is_valid_output(result):
                warnings.warn(
                    "IC/BC callable uses expanded coordinate signature f(x, y). "
                    "This is deprecated. Use f(x) where x is ndarray of shape (2,) instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )

                def _expanded_2d(x: np.ndarray, _fn: Callable = func) -> float:
                    return float(_fn(float(x[0]), float(x[1])))

                return CallableSignature.EXPANDED_2D, _expanded_2d
            attempts.append(("f(x, y) with expanded coordinates", _err_str(err, result)))

        elif dimension == 3 and array_sample.shape == (3,):
            result, err = _try_call(func, float(array_sample[0]), float(array_sample[1]), float(array_sample[2]))
            if err is None and _is_valid_output(result):
                warnings.warn(
                    "IC/BC callable uses expanded coordinate signature f(x, y, z). "
                    "This is deprecated. Use f(x) where x is ndarray of shape (3,) instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )

                def _expanded_3d(x: np.ndarray, _fn: Callable = func) -> float:
                    return float(_fn(float(x[0]), float(x[1]), float(x[2])))

                return CallableSignature.EXPANDED_3D, _expanded_3d
            attempts.append(("f(x, y, z) with expanded coordinates", _err_str(err, result)))

    # All probes failed
    msg = _format_signature_error(func, dimension, attempts)
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_call(func: Callable, *args: object) -> tuple[object, Exception | None]:
    """Try calling func with given args. Returns (result, None) or (None, exception)."""
    try:
        return func(*args), None
    except Exception as e:
        return None, e


def _is_valid_output(value: object) -> bool:
    """Check that a probe result is a numeric type (signature matched).

    NaN/Inf are accepted here -- they indicate the signature worked but the
    math produced a bad result. The validation layer checks finiteness
    separately (in _validate_callable_ic).
    """
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return True
    if isinstance(value, np.ndarray):
        # Accept scalar-like or 1-element arrays
        return value.ndim == 0 or value.size == 1
    return False


def _err_str(err: Exception | None, result: object) -> str:
    """Format a probe failure for the error message."""
    if err is not None:
        return f"{type(err).__name__}: {err}"
    return f"returned invalid output: {result!r}"


def _format_signature_error(
    func: Callable,
    dimension: int,
    attempts: list[tuple[str, str]],
) -> str:
    """Format a helpful error message listing all attempted calling conventions."""
    func_name = getattr(func, "__name__", repr(func))
    lines = [
        f"Cannot determine signature of IC/BC callable '{func_name}' (dimension={dimension}).",
        "",
        "Attempted calling conventions:",
    ]
    for convention, error in attempts:
        lines.append(f"  - {convention}")
        lines.append(f"    -> {error}")

    lines.append("")
    if dimension == 1:
        lines.append(
            "Accepted signatures for 1D:\n"
            "  f(x)      -- x is a Python float\n"
            "  f(x)      -- x is ndarray of shape (1,)\n"
            "  f(x, t)   -- spatiotemporal\n"
            "  f(t, x)   -- spatiotemporal (reversed)"
        )
    else:
        lines.append(
            f"Accepted signatures for {dimension}D:\n"
            f"  f(x)      -- x is ndarray of shape ({dimension},)\n"
            f"  f(x, t)   -- spatiotemporal\n"
            f"  f(t, x)   -- spatiotemporal (reversed)"
        )
    return "\n".join(lines)
