"""
Integration utilities for MFG_PDE.

Provides consistent integration functions that work across NumPy versions,
with preference for NumPy 2.0+ standard but fallbacks for older versions.
"""

import numpy as np
from typing import Union, Optional


def trapezoid(
    y, x: Optional[Union[np.ndarray, float]] = None, dx: float = 1.0, axis: int = -1
) -> Union[float, np.ndarray]:
    """
    Trapezoidal rule integration.

    Uses trapezoid if available (NumPy 2.0+), otherwise falls back to
    scipy.integrate.trapezoid or np.trapz as appropriate.

    Args:
        y: Values to integrate
        x: Sample points corresponding to y values
        dx: Spacing between samples when x is None
        axis: Axis along which to integrate

    Returns:
        Definite integral approximated by trapezoidal rule
    """
    # Try NumPy 2.0+ first (preferred)
    if hasattr(np, "trapezoid"):
        return trapezoid(y, x=x, dx=dx, axis=axis)

    # Try scipy fallback for compatibility
    try:
        from scipy.integrate import trapezoid as scipy_trapezoid

        return scipy_trapezoid(y, x=x, dx=dx, axis=axis)
    except ImportError:
        pass

    # Final fallback to deprecated np.trapz (NumPy < 2.0)
    if hasattr(np, "trapz"):
        import warnings

        warnings.warn(
            "Using deprecated np.trapz. Consider upgrading to NumPy 2.0+ "
            "or installing scipy for better integration support.",
            DeprecationWarning,
            stacklevel=2,
        )
        return np.trapz(y, x=x, dx=dx, axis=axis)

    # Should never reach here in normal environments
    raise RuntimeError(
        "No trapezoidal integration function available. "
        "Please install NumPy 2.0+ or scipy."
    )


def get_integration_info() -> dict:
    """Get information about available integration methods."""
    info = {
        "numpy_version": np.__version__,
        "numpy_trapezoid_available": hasattr(np, "trapezoid"),
        "numpy_trapz_available": hasattr(np, "trapz"),
        "scipy_trapezoid_available": False,
        "recommended_method": None,
    }

    try:
        from scipy.integrate import trapezoid as scipy_trapezoid

        info["scipy_trapezoid_available"] = True
    except ImportError:
        pass

    # Determine recommended method
    if info["numpy_trapezoid_available"]:
        info["recommended_method"] = "numpy.trapezoid"
    elif info["scipy_trapezoid_available"]:
        info["recommended_method"] = "scipy.integrate.trapezoid"
    elif info["numpy_trapz_available"]:
        info["recommended_method"] = "numpy.trapz (deprecated)"
    else:
        info["recommended_method"] = "None available"

    return info
