"""
Multi-population MFG solvers.

This module provides specialized solvers for K-population mean field games
where multiple populations interact through coupled HJB-FP systems.

The main solver class is MultiPopulationFixedPointSolver, which orchestrates
K single-population solvers to solve the coupled system iteratively.
"""

from .fixed_point import MultiPopulationFixedPointSolver

__all__ = [
    "MultiPopulationFixedPointSolver",
]
