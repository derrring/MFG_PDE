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
                f"{func.__name__} is deprecated and will be removed in v{removal_version}. " f"{reason}",
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


__all__ = ["DeprecatedAPI", "deprecated"]

# Re-export old API with deprecation warnings
from .legacy_config import *  # noqa: F403, E402
from .legacy_problems import *  # noqa: F403, E402
from .legacy_solvers import *  # noqa: F403, E402

__all__.extend(["LegacyConfig", "LegacyMFGProblem", "LegacyMFGSolver"])
