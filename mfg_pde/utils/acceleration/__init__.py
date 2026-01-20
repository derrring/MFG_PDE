"""
Acceleration Utilities for MFG_PDE

This module provides acceleration utilities for high-performance computing
across different computational backends (JAX, PyTorch, etc.).

The module is organized to separate framework-specific utilities while
providing common interfaces for acceleration operations.

Components:
- JAX utilities: jax_utils.py - JAX-specific acceleration functions
- PyTorch utilities: torch_utils.py - PyTorch KDE, tridiagonal solver, device management

This replaces the old mfg_pde/accelerated/ directory with better organization.
"""

from __future__ import annotations

import warnings

from mfg_pde.utils.mfg_logging import get_logger

logger = get_logger(__name__)
# Check JAX availability
try:
    import jax

    HAS_JAX = True
    HAS_GPU = any("gpu" in str(d).lower() for d in jax.devices())
    DEFAULT_DEVICE = jax.devices()[0]
except ImportError:
    HAS_JAX = False
    HAS_GPU = False
    DEFAULT_DEVICE = None

# Re-export JAX utilities
try:
    from .jax_utils import *  # noqa: F403
    from .jax_utils import HAS_JAX

    # Only mark as available if JAX is actually installed
    JAX_UTILS_AVAILABLE = HAS_JAX
except ImportError:
    JAX_UTILS_AVAILABLE = False

# Re-export PyTorch utilities
try:
    from .torch_utils import *  # noqa: F403
    from .torch_utils import HAS_TORCH

    # Only mark as available if torch is actually installed
    TORCH_UTILS_AVAILABLE = HAS_TORCH
except ImportError:
    TORCH_UTILS_AVAILABLE = False


def get_acceleration_info():
    """Get information about available acceleration utilities."""
    info = {
        "jax_utils_available": JAX_UTILS_AVAILABLE,
        "torch_utils_available": TORCH_UTILS_AVAILABLE,
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
        except ImportError as e:
            logger.debug(f"Could not retrieve JAX detailed info: {e}")

    if TORCH_UTILS_AVAILABLE:
        try:
            from .torch_utils import HAS_CUDA, HAS_MPS, HAS_TORCH

            info.update(
                {
                    "torch_available": HAS_TORCH,
                    "torch_cuda_available": HAS_CUDA,
                    "torch_mps_available": HAS_MPS,
                }
            )
        except ImportError as e:
            logger.debug(f"Could not retrieve PyTorch detailed info: {e}")

    return info


# Backward compatibility warning for old imports
def _warn_deprecated_import():
    """Warn about deprecated import paths."""
    warnings.warn(
        "Importing from mfg_pde.accelerated is deprecated. Use mfg_pde.utils.acceleration instead.",
        DeprecationWarning,
        stacklevel=3,
    )


__all__ = [
    "JAX_UTILS_AVAILABLE",
    "TORCH_UTILS_AVAILABLE",
    "get_acceleration_info",
]

# Export JAX utils if available
if JAX_UTILS_AVAILABLE:
    try:
        from .jax_utils import __all__ as jax_utils_all

        __all__.extend(jax_utils_all)
    except (ImportError, AttributeError):
        pass

# Export PyTorch utils if available
if TORCH_UTILS_AVAILABLE:
    try:
        from .torch_utils import __all__ as torch_utils_all

        __all__.extend(torch_utils_all)
    except (ImportError, AttributeError):
        pass
