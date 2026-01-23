"""
Iterative solvers for MFG systems.

This module provides high-level iteration schemes that orchestrate
the coupling between HJB and FP solvers:

- FixedPointSolver: Picard (fixed-point) iteration for MFG systems
- MultiPopulationFixedPointSolver: Extension for multi-population games

These solvers use the numerical solvers from `mfg_pde.alg.numerical`
for the actual HJB/FP equation solving.

Example:
    from mfg_pde.alg.iterative import FixedPointSolver

    solver = FixedPointSolver()
    result = solver.solve(problem)

Example with hooks:
    from mfg_pde.alg.iterative import FixedPointSolver
    from mfg_pde.hooks import DebugHook

    solver = FixedPointSolver()
    result = solver.solve(problem, hooks=DebugHook())
"""

from .base import BaseIterativeSolver, BaseSolver
from .fixed_point import FixedPointResult, FixedPointSolver
from .multi_population import MultiPopulationFixedPointSolver

__all__ = [
    # Base classes
    "BaseIterativeSolver",
    "BaseSolver",  # Backward compatibility alias
    # Solvers
    "FixedPointSolver",
    "FixedPointResult",
    "MultiPopulationFixedPointSolver",
]
