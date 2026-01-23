"""
New algorithm structure for MFG_PDE.

This module implements the reorganized algorithm structure with paradigm-based organization:
- numerical: Classical numerical analysis methods
- optimization: Direct optimization approaches
- neural: Neural network-based methods
- reinforcement: Reinforcement learning paradigm
- iterative: High-level iteration schemes (Picard, multi-population)

The structure emphasizes interconnections between paradigms and supports
hybrid methods that combine multiple approaches.
"""

from __future__ import annotations

# Import paradigm modules
from . import iterative, neural, numerical, optimization, reinforcement

# Import base types (Issue #580)
from .base_solver import SchemeFamily

# Import commonly used iterative solvers for convenience
from .iterative import FixedPointSolver, MultiPopulationFixedPointSolver

__all__ = [
    # Paradigm modules
    "iterative",
    "neural",
    "numerical",
    "optimization",
    "reinforcement",
    # Base types
    "SchemeFamily",
    # Commonly used solvers
    "FixedPointSolver",
    "MultiPopulationFixedPointSolver",
]
