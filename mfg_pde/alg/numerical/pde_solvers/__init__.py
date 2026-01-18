"""
PDE solvers for physics equations (heat, wave, etc.).

These solvers are designed for coupling with Level Set methods and other
multi-physics problems, not for solving the MFG system itself.
"""

from mfg_pde.alg.numerical.pde_solvers.implicit_heat import ImplicitHeatSolver

__all__ = ["ImplicitHeatSolver"]
