"""
Numerical methods paradigm for MFG problems.

This module contains classical numerical analysis approaches:
- hjb_solvers: Individual HJB equation solvers
- fp_solvers: Individual Fokker-Planck solvers
- mfg_solvers: Coupled system numerical methods

All methods are based on discretization and convergence analysis.
"""

from mfg_pde.alg.base_solver import BaseNumericalSolver

# Import HJB solvers
from .hjb_solvers import (
    BaseHJBSolver,
    HJBFDMSolver,
    HJBGFDMSolver,
    HJBSemiLagrangianSolver,
    HJBWenoSolver,
)

__all__ = [
    "BaseNumericalSolver",
    # HJB Solvers
    "BaseHJBSolver",
    "HJBFDMSolver",
    "HJBGFDMSolver",
    "HJBSemiLagrangianSolver",
    "HJBWenoSolver",
]
