"""
Variational Solvers for Lagrangian MFG Problems

This module provides direct optimization methods for solving Mean Field Games
formulated as variational problems. Instead of solving the HJB-FP system,
these methods directly minimize the cost functional.

Available solvers:
- VariationalMFGSolver: Direct optimization of the action functional
- PrimalDualMFGSolver: Primal-dual methods for constrained problems
"""

from .base_variational import BaseVariationalSolver, VariationalSolverResult
from .variational_mfg_solver import VariationalMFGSolver
from .primal_dual_solver import PrimalDualMFGSolver

__all__ = [
    "BaseVariationalSolver",
    "VariationalSolverResult",
    "VariationalMFGSolver",
    "PrimalDualMFGSolver",
]
