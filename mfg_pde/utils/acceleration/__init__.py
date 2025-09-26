"""
Acceleration Utilities for MFG_PDE

This module provides acceleration utilities for high-performance computing
across different computational backends (JAX, PyTorch, etc.).

The module is organized to separate framework-specific utilities while
providing common interfaces for acceleration operations.

Components:
- JAX utilities: jax_utils.py - JAX-specific acceleration functions
- Future: PyTorch utilities, Intel GPU utilities, etc.

This replaces the old mfg_pde/accelerated/ directory with better organization.
"""

from __future__ import annotations

import warnings

# Re-export JAX utilities for backward compatibility
try:
    from .jax_utils import *  # noqa: F403

    JAX_UTILS_AVAILABLE = True
except ImportError:
    JAX_UTILS_AVAILABLE = False


def get_acceleration_info():
    """Get information about available acceleration utilities."""
    info = {
        "jax_utils_available": JAX_UTILS_AVAILABLE,
    }

    if JAX_UTILS_AVAILABLE:
        try:
            from .jax_utils import HAS_GPU, HAS_JAX

            info.update(
                {
                    "jax_available": HAS_JAX,
                    "jax_gpu_available": HAS_GPU,
                }
            )
        except ImportError:
            pass

    return info


# Backward compatibility warning for old imports
def _warn_deprecated_import():
    """Warn about deprecated import paths."""
    warnings.warn(
        "Importing from mfg_pde.accelerated is deprecated. " "Use mfg_pde.utils.acceleration instead.",
        DeprecationWarning,
        stacklevel=3,
    )


__all__ = [
    "get_acceleration_info",
    "JAX_UTILS_AVAILABLE",
]

# Export JAX utils if available
if JAX_UTILS_AVAILABLE:
    try:
        from .jax_utils import __all__ as jax_utils_all

        __all__.extend(jax_utils_all)
    except (ImportError, AttributeError):
        pass
