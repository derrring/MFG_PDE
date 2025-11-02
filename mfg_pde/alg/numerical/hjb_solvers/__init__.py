"""
HJB Solvers for Numerical Methods.

This module contains Hamilton-Jacobi-Bellman equation solvers using classical
numerical analysis approaches:

- BaseHJBSolver: Abstract base class for all HJB solvers
- HJBFDMSolver: Finite difference method (all dimensions: 1D, 2D, 3D, nD)
- HJBGFDMSolver: Generalized finite difference method (meshfree, nD)
- HJBSemiLagrangianSolver: Semi-Lagrangian approach (characteristic-based, nD)
- HJBWenoSolver: WENO (Weighted Essentially Non-Oscillatory) method (1D/2D/3D)

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
