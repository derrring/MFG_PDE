"""
Solver extensions for specialized MFG problem types.

This module contains specialized solvers for MFG problem extensions
that go beyond the core single-population formulation.

Available extensions:
- multi_population: Solvers for K-population MFG systems
"""

from .multi_population import MultiPopulationFixedPointSolver

__all__ = [
    "MultiPopulationFixedPointSolver",
]
