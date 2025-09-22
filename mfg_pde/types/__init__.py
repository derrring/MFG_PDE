"""
Advanced Types for MFG_PDE

This module contains type definitions for advanced users who need
to customize solver behavior or implement custom algorithms.

⚠️  Advanced API Warning:
    These types are stable but considered "advanced". They may change
    between major versions. For simple usage, stick to the main API
    in mfg_pde.__init__.py

Basic Types (safe to use):
    from mfg_pde.types import MFGProblem, MFGResult, SolverConfig

Internal Types (use with caution):
    from mfg_pde.types import SpatialTemporalState, HamiltonianFunction
"""

# Re-export the most commonly needed types
from .protocols import MFGProblem, MFGResult, MFGSolver, SolutionArray, SolverConfig
from .state import ConvergenceInfo, SpatialTemporalState

__all__ = [
    "MFGProblem",
    "MFGResult",
    "MFGSolver",
    "SolverConfig",
    "SolutionArray",
    "SpatialTemporalState",
    "ConvergenceInfo",
]

# Internal types available via explicit import:
# from mfg_pde.types.internal import HamiltonianFunction, ...
