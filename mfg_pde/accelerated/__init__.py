"""
GPU-accelerated solvers for MFG_PDE using JAX.

⚠️  DEPRECATED: This module has been reorganized for better structure.

NEW LOCATIONS:
- JAX utilities: mfg_pde.utils.acceleration.jax_utils
- JAX MFG solver: mfg_pde.alg.mfg_solvers.JAXMFGSolver

This module provides backward compatibility but will be removed in a future version.
Please update your imports to use the new locations.

MIGRATION GUIDE:
  # OLD (deprecated)
  from mfg_pde.accelerated import JAXMFGSolver
  from mfg_pde.accelerated.jax_utils import compute_hamiltonian

  # NEW (recommended)
  from mfg_pde.alg.mfg_solvers import JAXMFGSolver
  from mfg_pde.utils.acceleration.jax_utils import compute_hamiltonian
"""

from __future__ import annotations

import warnings

# Issue deprecation warning
warnings.warn(
    "mfg_pde.accelerated is deprecated and will be removed in a future version. "
    "Use mfg_pde.utils.acceleration for utilities and mfg_pde.alg.mfg_solvers for solvers.",
    DeprecationWarning,
    stacklevel=2,
)

# Backward compatibility imports
try:
    # Re-export JAX utilities from new location
    # Re-export JAX MFG solver from new location
    from mfg_pde.alg.mfg_solvers import JAXMFGSolver  # noqa: F401
    from mfg_pde.utils.acceleration import *  # noqa: F403
    from mfg_pde.utils.acceleration.jax_utils import *  # noqa: F403

    # Legacy compatibility
    JAX_AVAILABLE = True

    # Re-export key components with deprecation
    def _deprecated_import_wrapper(new_location, old_name):
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"Importing {old_name} from mfg_pde.accelerated is deprecated. Use {new_location} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return new_location(*args, **kwargs)

        return wrapper

except ImportError:
    JAX_AVAILABLE = False

    warnings.warn(
        "JAX components not available. Install JAX for GPU acceleration: pip install jax jaxlib", ImportWarning
    )

# Export availability status
__all__ = ["JAX_AVAILABLE"]

# Add conditionally available exports
if JAX_AVAILABLE:
    try:
        from mfg_pde.utils.acceleration import __all__ as utils_all
        from mfg_pde.utils.acceleration.jax_utils import __all__ as jax_utils_all

        __all__.extend(utils_all)
        __all__.extend(jax_utils_all)
        __all__.append("JAXMFGSolver")

    except (ImportError, AttributeError):
        pass
