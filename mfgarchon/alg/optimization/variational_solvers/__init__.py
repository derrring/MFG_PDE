"""
Variational solvers for Lagrangian MFG formulations.

This module provides direct optimization methods for solving Mean Field Games
formulated as variational problems. Instead of solving the HJB-FP system,
these methods directly minimize the cost functional.

Available solvers:
- BaseVariationalSolver: Abstract base class for all variational methods
- VariationalMFGSolver: Direct optimization of the action functional
- PrimalDualMFGSolver: Primal-dual methods for constrained problems
"""

from .base_variational import BaseVariationalSolver, VariationalSolverResult
from .primal_dual_solver import PrimalDualMFGSolver
from .variational_mfg_solver import VariationalMFGSolver

__all__ = [
    "BaseVariationalSolver",
    "PrimalDualMFGSolver",
    "VariationalMFGSolver",
    "VariationalSolverResult",
]

# Solver categories for factory selection
DIRECT_OPTIMIZATION_SOLVERS = [
    "VariationalMFGSolver",  # Direct minimization of Lagrangian
]

CONSTRAINED_OPTIMIZATION_SOLVERS = [
    "PrimalDualMFGSolver",  # Primal-dual methods for constrained problems
]

ALL_VARIATIONAL_SOLVERS = DIRECT_OPTIMIZATION_SOLVERS + CONSTRAINED_OPTIMIZATION_SOLVERS
