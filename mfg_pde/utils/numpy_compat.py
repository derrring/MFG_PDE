#!/usr/bin/env python3
"""
NumPy compatibility utilities for MFG_PDE.

Handles NumPy version differences, particularly the trapz → trapezoid transition
in NumPy 2.0+. Provides consistent interface across NumPy versions.
"""

import warnings
from typing import Optional, Union

import numpy as np

# Check NumPy version and available functions
NUMPY_VERSION = tuple(map(int, np.__version__.split('.')[:2]))
HAS_TRAPEZOID = hasattr(np, 'trapezoid')
HAS_TRAPZ = hasattr(np, 'trapz')

# Import scipy fallback if available
try:
    from scipy.integrate import trapezoid as scipy_trapezoid
    HAS_SCIPY_TRAPEZOID = True
except ImportError:
    HAS_SCIPY_TRAPEZOID = False


def trapezoid(
    y, x: Optional[Union[np.ndarray, float]] = None, dx: float = 1.0, axis: int = -1
) -> Union[float, np.ndarray]:
    """
    NumPy 2.0+ compatible trapezoidal integration.
    
    Automatically uses the best available method:
    1. np.trapezoid (NumPy 2.0+, preferred)
    2. scipy.integrate.trapezoid (if scipy available)
    3. np.trapz (NumPy < 2.0, deprecated in 2.0+)
    
    Args:
        y: Values to integrate
        x: Sample points corresponding to y values
        dx: Spacing between samples when x is None
        axis: Axis along which to integrate
        
    Returns:
        Definite integral approximated by trapezoidal rule
        
    Raises:
        RuntimeError: If no integration method is available
    """
    # Method 1: NumPy 2.0+ trapezoid (preferred)
    if HAS_TRAPEZOID:
        return np.trapezoid(y, x=x, dx=dx, axis=axis)
    
    # Method 2: SciPy fallback (good compatibility)
    if HAS_SCIPY_TRAPEZOID:
        return scipy_trapezoid(y, x=x, dx=dx, axis=axis)
    
    # Method 3: Legacy np.trapz (deprecated in NumPy 2.0+)
    if HAS_TRAPZ:
        if NUMPY_VERSION >= (2, 0):
            warnings.warn(
                "Using deprecated np.trapz with NumPy 2.0+. "
                "Consider updating code to use np.trapezoid directly.",
                DeprecationWarning,
                stacklevel=2
            )
        return np.trapz(y, x=x, dx=dx, axis=axis)
    
    # Should never reach here in normal environments
    raise RuntimeError(
        "No trapezoidal integration function available. "
        "Please install NumPy 2.0+ or scipy."
    )


def get_numpy_info() -> dict:
    """Get information about NumPy version and available functions."""
    return {
        "numpy_version": np.__version__,
        "numpy_version_tuple": NUMPY_VERSION,
        "has_trapezoid": HAS_TRAPEZOID,
        "has_trapz": HAS_TRAPZ,
        "has_scipy_trapezoid": HAS_SCIPY_TRAPEZOID,
        "recommended_method": _get_recommended_method(),
        "is_numpy_2_plus": NUMPY_VERSION >= (2, 0),
    }


def _get_recommended_method() -> str:
    """Determine the recommended integration method."""
    if HAS_TRAPEZOID:
        return "np.trapezoid"
    elif HAS_SCIPY_TRAPEZOID:
        return "scipy.integrate.trapezoid"
    elif HAS_TRAPZ:
        return "np.trapz (deprecated)" if NUMPY_VERSION >= (2, 0) else "np.trapz"
    else:
        return "None available"


def ensure_numpy_compatibility():
    """
    Check NumPy compatibility and emit warnings if needed.
    
    This function should be called during package initialization to
    alert users about potential compatibility issues.
    """
    info = get_numpy_info()
    
    if NUMPY_VERSION >= (2, 0) and not HAS_TRAPEZOID:
        warnings.warn(
            f"NumPy {np.__version__} appears to be 2.0+ but doesn't have trapezoid. "
            "This may indicate a problem with your NumPy installation.",
            UserWarning
        )
    
    if NUMPY_VERSION >= (2, 0) and HAS_TRAPZ and not HAS_TRAPEZOID:
        warnings.warn(
            "NumPy 2.0+ detected but using deprecated trapz. "
            "Consider updating your code to use np.trapezoid.",
            DeprecationWarning
        )
    
    return info


# Convenience aliases for direct usage
if HAS_TRAPEZOID:
    # NumPy 2.0+ - use directly
    numpy_trapezoid = np.trapezoid
elif HAS_SCIPY_TRAPEZOID:
    # Fallback to scipy
    numpy_trapezoid = scipy_trapezoid
elif HAS_TRAPZ:
    # Legacy fallback
    numpy_trapezoid = np.trapz
else:
    # No integration available
    def numpy_trapezoid(*args, **kwargs):
        raise RuntimeError("No trapezoidal integration function available")


# Export the main function for backward compatibility
__all__ = ['trapezoid', 'numpy_trapezoid', 'get_numpy_info', 'ensure_numpy_compatibility']