"""
MFG Solvers with Clean Interfaces

This module provides solver classes with clean, simple interfaces
that support the hooks pattern for advanced customization.

Basic Usage:
    from mfg_pde.solvers import FixedPointSolver

    solver = FixedPointSolver()
    result = solver.solve(problem)

Advanced Usage with Hooks:
    from mfg_pde.hooks import DebugHook

    solver = FixedPointSolver()
    result = solver.solve(problem, hooks=DebugHook())
"""

from .base import BaseSolver
from .fixed_point import FixedPointSolver

__all__ = ["BaseSolver", "FixedPointSolver"]
