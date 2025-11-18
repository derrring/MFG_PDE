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
    from mfg_pde.types import SolutionArray, SpatialGrid, TimeGrid

Internal Types (use with caution):
    from mfg_pde.types import SpatialTemporalState, HamiltonianFunction
"""

# Re-export the most commonly needed types
from .arrays import SolutionArray, SpatialGrid, TimeGrid
from .pde_coefficients import DiffusionCallable, DiffusionField, DriftCallable, DriftField
from .protocols import MFGProblem, MFGResult, MFGSolver, SolverConfig
from .solver_types import HamiltonianJacobians, SolverReturnTuple
from .state import ConvergenceInfo, SpatialTemporalState

__all__ = [
    "ConvergenceInfo",
    "DiffusionCallable",
    "DiffusionField",
    "DriftCallable",
    "DriftField",
    "HamiltonianJacobians",
    "MFGProblem",
    "MFGResult",
    "MFGSolver",
    "SolutionArray",
    "SolverConfig",
    "SolverReturnTuple",
    "SpatialGrid",
    "SpatialTemporalState",
    "TimeGrid",
]

# Array types available via explicit import:
# from mfg_pde.types.arrays import SpatialArray, TemporalArray, ParticleArray, ...

# Solver types and advanced types available via explicit import:
# from mfg_pde.types.solver_types import JAXSolverReturn, ComplexSolverState, MetadataDict, ...
# from mfg_pde.types.solver_types import HamiltonianFunction, NewtonSolver, SolverError, ...
