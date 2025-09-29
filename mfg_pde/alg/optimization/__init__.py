"""
Optimization paradigm for MFG problems.

This module contains direct optimization approaches for solving Mean Field Games:
- variational_solvers: Lagrangian formulation and variational methods

All methods are based on direct optimization of cost functionals,
providing alternatives to the classical HJB-FP system approach.
"""

from mfg_pde.alg.base_solver import BaseOptimizationSolver

# Import variational solvers
from .variational_solvers import (
    BaseVariationalSolver,
    PrimalDualMFGSolver,
    VariationalMFGSolver,
    VariationalSolverResult,
)

__all__ = [
    "BaseOptimizationSolver",
    # Variational Solvers
    "BaseVariationalSolver",
    "PrimalDualMFGSolver",
    "VariationalMFGSolver",
    "VariationalSolverResult",
]
