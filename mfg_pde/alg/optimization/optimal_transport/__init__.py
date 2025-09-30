"""
Optimal Transport methods for MFG problems.

This module contains optimal transport approaches for solving Mean Field Games:
- wasserstein_solver: Wasserstein distance-based formulations
- sinkhorn_solvers: Entropic regularized optimal transport
- displacement_interpolation: Geodesic paths in probability space

Optimal transport provides a natural geometric framework for MFG problems
through the Wasserstein space of probability measures.
"""

from mfg_pde.alg.base_solver import BaseOptimizationSolver

try:
    import scipy.optimize
    import scipy.sparse  # noqa: F401

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import ot  # noqa: F401 # Python Optimal Transport library

    POT_AVAILABLE = True
except ImportError:
    POT_AVAILABLE = False

if SCIPY_AVAILABLE and POT_AVAILABLE:
    from .sinkhorn_solver import (
        SinkhornMFGSolver,
        SinkhornSolverConfig,
        SinkhornSolverResult,
    )
    from .wasserstein_solver import (
        WassersteinMFGSolver,
        WassersteinSolverConfig,
        WassersteinSolverResult,
    )

    __all__ = [
        "BaseOptimizationSolver",
        # Sinkhorn Methods
        "SinkhornMFGSolver",
        "SinkhornSolverConfig",
        "SinkhornSolverResult",
        # Wasserstein Methods
        "WassersteinMFGSolver",
        "WassersteinSolverConfig",
        "WassersteinSolverResult",
    ]

    # Solver categories for factory selection
    WASSERSTEIN_SOLVERS = [
        "WassersteinMFGSolver",
    ]

    SINKHORN_SOLVERS = [
        "SinkhornMFGSolver",
    ]

    ALL_OT_SOLVERS = WASSERSTEIN_SOLVERS + SINKHORN_SOLVERS

else:
    import warnings

    missing_deps = []
    if not SCIPY_AVAILABLE:
        missing_deps.append("scipy")
    if not POT_AVAILABLE:
        missing_deps.append("POT (Python Optimal Transport)")

    warnings.warn(
        f"Optimal transport solvers require {', '.join(missing_deps)}. "
        f"Install with: pip install mfg_pde[optimization] or pip install scipy pot",
        ImportWarning,
    )

    __all__ = [
        "BaseOptimizationSolver",
    ]

    WASSERSTEIN_SOLVERS = []
    SINKHORN_SOLVERS = []
    ALL_OT_SOLVERS = []

# Always export availability info
__all__.extend(["POT_AVAILABLE", "SCIPY_AVAILABLE"])
