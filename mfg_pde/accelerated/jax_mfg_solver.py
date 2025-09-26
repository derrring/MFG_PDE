"""
JAX MFG Solver - DEPRECATED

⚠️  This solver has moved to mfg_pde.alg.mfg_solvers.JAXMFGSolver

This file provides backward compatibility but will be removed in a future version.
Please update your imports:

  # OLD (deprecated)
  from mfg_pde.accelerated.jax_mfg_solver import JAXMFGSolver

  # NEW (recommended)
  from mfg_pde.alg.mfg_solvers import JAXMFGSolver
"""

import warnings

# Re-export from new location
from mfg_pde.alg.mfg_solvers.jax_mfg_solver import JAXMFGSolver

warnings.warn(
    "mfg_pde.accelerated.jax_mfg_solver is deprecated. " "Import JAXMFGSolver from mfg_pde.alg.mfg_solvers instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["JAXMFGSolver"]
