"""
Numerical methods for stochastic Mean Field Games.

This module provides numerical solvers for stochastic MFG formulations:
- Common Noise MFG: Monte Carlo over noise realizations
- Master Equation: Functional PDE solvers (Week 5-8)
"""

from .common_noise_solver import CommonNoiseMFGResult, CommonNoiseMFGSolver

__all__ = [
    "CommonNoiseMFGSolver",
    "CommonNoiseMFGResult",
]
