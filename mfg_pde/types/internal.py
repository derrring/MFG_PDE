"""
Internal Types for Advanced Users

⚠️  Advanced API Warning:
    These types are for users who need deep customization of algorithms.
    They are stable but may change between major versions.

    Most users should use the hooks system instead of these types directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
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

# Array type aliases for solver internals
type SpatialArray = NDArray
"""1D spatial array, typically shape (Nx+1,)"""

type TemporalArray = NDArray
"""1D temporal array, typically shape (Nt+1,)"""

type SolutionArray = NDArray
"""2D spatio-temporal solution array, typically shape (Nt+1, Nx+1)"""


# Solver component type aliases
class NewtonSolver(Protocol):
    """Protocol for Newton-type solvers."""

    def solve_step(self, u_current: SolutionArray, rhs: SolutionArray) -> SolutionArray: ...


class LinearSolver(Protocol):
    """Protocol for linear system solvers."""

    def solve(self, A: NDArray, b: NDArray) -> NDArray: ...


# Configuration type aliases
type ParameterDict = dict[str, float | int | str | bool]
"""Dictionary of solver parameters"""

type SolverOptions = dict[str, float | int | str | bool | None]
"""Dictionary of optional solver settings"""

# Complex internal types (for maintainers)
type ComplexSolverState = (
    tuple[SolutionArray, SolutionArray]  # Simple (u, m) state
    | dict[str, SolutionArray | float | int]  # Complex state with metadata
    | object  # Completely custom state objects
)
"""Internal solver state - can be simple arrays or complex objects"""

type FlexibleInput = (
    HamiltonianFunction
    | str  # String identifier for preset Hamiltonians
    | dict[str, float | Callable]  # Configuration dictionary
    | object  # Custom problem objects
)
"""Flexible input type that accepts multiple input formats"""

# Complex solver return types that appear frequently
type SolverReturnTuple = tuple[np.ndarray, np.ndarray, dict[str, Any]]
"""Standard solver return type: (U, M, convergence_info)"""

type JAXSolverReturn = tuple[Any, Any, bool, int, float]
"""JAX solver return type: (U_jax, M_jax, converged, iterations, residual)"""

type MultiIndexTuple = tuple[int, ...]
"""Multi-index for GFDM operations"""

type DerivativeDict = dict[tuple[int, ...], float]
"""Dictionary mapping multi-indices to derivative values"""

type GradientDict = dict[str, float]
"""Dictionary for gradient components: {'dx': value, 'dy': value, ...}"""

type StencilResult = list[tuple[np.ndarray, bool]]
"""Stencil computation result: list of (stencil_array, success_flag)"""

type MetadataDict = dict[str, float | int | str | bool | np.ndarray]
"""Flexible metadata dictionary for solver state and results"""

type ErrorCallback = Callable[[Exception], None] | None
"""Optional error handling callback function"""

# JAX-specific type aliases (with fallbacks)
try:
    if TYPE_CHECKING:
        from jax import Array

        type JAXArray = Array | np.ndarray | Any
    else:
        # At runtime, don't import JAX to avoid dependency issues
        type JAXArray = Any
except ImportError:
    # JAX not available - use numpy arrays and Any as fallback
    if not TYPE_CHECKING:
        type JAXArray = np.ndarray | Any


# Error handling types
class SolverError(Exception):
    """Base exception for solver errors."""


class ConvergenceError(SolverError):
    """Raised when solver fails to converge."""


class ConfigurationError(SolverError):
    """Raised when solver configuration is invalid."""
