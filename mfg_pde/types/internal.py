"""
Internal Types for Advanced Users

⚠️  Advanced API Warning:
    These types are for users who need deep customization of algorithms.
    They are stable but may change between major versions.

    Most users should use the hooks system instead of these types directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Functional type aliases for mathematical objects
type HamiltonianFunction = Callable[[float, float, float, float], float]
"""Hamiltonian function H(x, p, m, t) -> float"""

type LagrangianFunction = Callable[[float, float, float, float], float]
"""Lagrangian function L(x, v, m, t) -> float"""

type DensityFunction = Callable[[float], float]
"""Density function ρ(x) -> float"""

type ValueFunction = Callable[[float], float]
"""Value function g(x) -> float"""

# Note: Array type aliases have been moved to types/arrays.py
# Import from there for consistency:
#   from mfg_pde.types.arrays import SpatialArray, TemporalArray, SolutionArray


# Solver component type aliases
class NewtonSolver(Protocol):
    """Protocol for Newton-type solvers."""

    def solve_step(self, u_current: NDArray, rhs: NDArray) -> NDArray: ...


class LinearSolver(Protocol):
    """Protocol for linear system solvers."""

    def solve(self, A: NDArray, b: NDArray) -> NDArray: ...


# Configuration type aliases
type ParameterDict = dict[str, float | int | str | bool]
"""Dictionary of solver parameters"""

type SolverOptions = dict[str, float | int | str | bool | None]
"""Dictionary of optional solver settings"""

# Note: Solver return types and state types have been moved to types/solver_types.py
# Import from there for consistency:
#   from mfg_pde.types.solver_types import SolverReturnTuple, JAXSolverReturn
#   from mfg_pde.types.solver_types import ComplexSolverState, MetadataDict
#   from mfg_pde.types.solver_types import MultiIndexTuple, DerivativeDict

type FlexibleInput = (
    HamiltonianFunction
    | str  # String identifier for preset Hamiltonians
    | dict[str, float | Callable]  # Configuration dictionary
    | object  # Custom problem objects
)
"""Flexible input type that accepts multiple input formats"""


# Error handling types
class SolverError(Exception):
    """Base exception for solver errors."""


class ConvergenceError(SolverError):
    """Raised when solver fails to converge."""


class ConfigurationError(SolverError):
    """Raised when solver configuration is invalid."""
