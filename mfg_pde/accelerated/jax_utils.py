"""
JAX utility functions - DEPRECATED

⚠️  This module has moved to mfg_pde.utils.acceleration.jax_utils

This file provides backward compatibility but will be removed in a future version.
Please update your imports:

  # OLD (deprecated)
  from mfg_pde.accelerated.jax_utils import compute_hamiltonian

  # NEW (recommended)
  from mfg_pde.utils.acceleration.jax_utils import compute_hamiltonian
"""

import warnings

# Re-export everything from new location
from mfg_pde.utils.acceleration.jax_utils import *  # noqa: F403

warnings.warn(
    "mfg_pde.accelerated.jax_utils is deprecated. Use mfg_pde.utils.acceleration.jax_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)
