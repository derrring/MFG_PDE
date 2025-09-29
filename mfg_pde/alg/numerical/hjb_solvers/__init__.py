"""
HJB Solvers for Numerical Methods.

This module contains Hamilton-Jacobi-Bellman equation solvers using classical
numerical analysis approaches:

- BaseHJBSolver: Abstract base class for all HJB solvers
- HJBFDMSolver: Finite difference method
- HJBGFDMSolver: Generalized finite difference method
- HJBSemiLagrangianSolver: Semi-Lagrangian approach
- HJBWenoSolver: WENO (Weighted Essentially Non-Oscillatory) method

All solvers inherit from BaseNumericalSolver and follow the new paradigm structure.
"""

from .base_hjb import BaseHJBSolver
from .hjb_fdm import HJBFDMSolver
from .hjb_gfdm import HJBGFDMSolver
from .hjb_semi_lagrangian import HJBSemiLagrangianSolver
from .hjb_weno import HJBWenoSolver

__all__ = [
    "BaseHJBSolver",
    "HJBFDMSolver",
    "HJBGFDMSolver",
    "HJBSemiLagrangianSolver",
    "HJBWenoSolver",
]
