"""
Compatibility layer for old MFG_PDE API

This module provides deprecation warnings and compatibility wrappers
to help users migrate from the old API to the new progressive disclosure API.

These functions will be removed in v2.0.
"""

import warnings


def deprecated(reason: str, removal_version: str = "2.0"):
    """
    Decorator to mark functions as deprecated.

    Args:
        reason: Explanation of what to use instead
        removal_version: Version when the function will be removed
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated and will be removed in v{removal_version}. {reason}",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = f"DEPRECATED: {func.__doc__ or ''}\n\n{reason}"
        return wrapper

    return decorator


class DeprecatedAPI:
    """Base class for deprecated API components."""

    def __init__(self, replacement: str):
        warnings.warn(
            f"{self.__class__.__name__} is deprecated. Use {replacement} instead. "
            f"See migration guide: https://docs.mfg-pde.org/migration",
            DeprecationWarning,
            stacklevel=2,
        )


__all__ = [
    "DeprecatedAPI",
    "check_derivs_format",
    "deprecated",
    "derivs_to_gradient_array",
    "derivs_to_p_values_1d",
    "ensure_tuple_notation",
    "gradient_array_to_derivs",
    "p_values_to_derivs_1d",
]

# Re-export old API with deprecation warnings
# NOTE: Temporarily commented out due to legacy_config.py having a bug with @deprecated decorator on classes
# from .legacy_config import *
# from .legacy_problems import *
# from .legacy_solvers import *

# Gradient notation backward compatibility
from .gradient_notation import (  # noqa: E402
    check_derivs_format,
    derivs_to_gradient_array,
    derivs_to_p_values_1d,
    ensure_tuple_notation,
    gradient_array_to_derivs,
    p_values_to_derivs_1d,
)
