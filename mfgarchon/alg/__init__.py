"""
Algorithm structure for MFGarchon.

Paradigm-based organization:
- numerical: Classical numerical analysis methods (FDM, FEM, GFDM, SL)
- optimization: Direct optimization approaches (variational, OT)
- neural: Neural network-based methods (PINN, DGM, operator learning)
- reinforcement: Reinforcement learning paradigm (AC, PPO, multi-population)

Iteration infrastructure (schedules, convergence) lives in utils/convergence/,
not here. See Issue #985.
"""

from __future__ import annotations

# Import paradigm modules
from . import neural, numerical, optimization, reinforcement

# Import base types (Issue #580)
from .base_solver import SchemeFamily

__all__ = [
    # Paradigm modules
    "neural",
    "numerical",
    "optimization",
    "reinforcement",
    # Base types
    "SchemeFamily",
]
