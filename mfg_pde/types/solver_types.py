"""
Solver Type Definitions

This module consolidates all solver-related type definitions including
return types, state representations, and solver-specific type aliases.

Usage:
    from mfg_pde.types.solver_types import SolverReturnTuple, JAXSolverReturn
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

# === Standard Solver Return Types ===

type SolverReturnTuple = tuple[NDArray, NDArray, dict[str, Any]]
"""
Standard solver return type: (U, M, convergence_info)

This is the legacy return format used by most numerical solvers.

Args:
    U: Value function array, shape (Nt+1, Nx+1)
    M: Density function array, shape (Nt+1, Nx+1)
    convergence_info: Dict with keys like 'converged', 'iterations', 'residual'
"""

type JAXSolverReturn = tuple[Any, Any, bool, int, float]
"""
JAX-accelerated solver return type.

Returns:
    U_jax: JAX array for value function
    M_jax: JAX array for density function
    converged: Boolean convergence flag
    iterations: Number of iterations performed
    residual: Final residual value
"""

# === Solver State Types ===

type SolverState = tuple[NDArray, NDArray] | dict[str, Any]
"""
Simple solver state representation.

Either:
- (u, m): Simple tuple of current solution
- dict: Complex state with metadata
"""

type ComplexSolverState = (
    tuple[NDArray, NDArray]  # Simple (u, m) state
    | dict[str, NDArray | float | int]  # Complex state with metadata
    | object  # Completely custom state objects
)
"""
Internal solver state - can be simple arrays or complex objects.

Used by iterative solvers to maintain state between iterations.
"""

type IntermediateResult = (
    NDArray  # Simple array result
    | tuple[NDArray, ...]  # Multiple arrays
    | dict[str, NDArray]  # Named array collection
    | Any  # Completely flexible
)
"""
Intermediate computation results during solving.

Solvers may return various formats during computation.
"""

# === Solver Configuration Types ===

type ParameterDict = dict[str, float | int | str | bool]
"""Dictionary of solver parameters."""

type SolverOptions = dict[str, float | int | str | bool | None]
"""Dictionary of optional solver settings."""

type ConfigValue = float | int | str | bool | None | list[float | int | str] | dict[str, Any] | Callable | Any
"""Flexible configuration value type."""

# === Callback Types ===

type ErrorCallback = Callable[[Exception], None] | None
"""Optional error handling callback function."""

type ProgressCallback = Callable[[int, float], None] | None
"""
Progress callback: (iteration, residual) -> None

Called after each solver iteration.
"""

type ConvergenceCallback = Callable[[bool, int, float], None] | None
"""
Convergence callback: (converged, iterations, final_residual) -> None

Called when solver terminates.
"""

# === GFDM-Specific Types ===

type MultiIndexTuple = tuple[int, ...]
"""Multi-index for GFDM operations, e.g., (2, 1) for ∂²/∂x²∂y."""

type DerivativeDict = dict[tuple[int, ...], float]
"""Dictionary mapping multi-indices to derivative values."""

type GradientDict = dict[str, float]
"""Dictionary for gradient components: {'dx': value, 'dy': value, ...}"""

type StencilResult = list[tuple[NDArray, bool]]
"""Stencil computation result: list of (stencil_array, success_flag)."""

# === Metadata Types ===

type MetadataDict = dict[str, float | int | str | bool | NDArray | None]
"""Flexible metadata dictionary for solver state and results."""

type ConvergenceMetadata = dict[str, Any]
"""
Convergence information metadata.

Typical keys: 'converged', 'iterations', 'residual', 'reason'
"""

# === JAX-Specific Types ===

try:
    if TYPE_CHECKING:
        from jax import Array as JAXArray
    else:
        JAXArray = Any  # Runtime fallback
except ImportError:
    JAXArray = Any  # JAX not available

type JAXStateArray = JAXArray | NDArray
"""Array type that can be either JAX or NumPy."""

# === Legacy Compatibility ===

# Old names for backward compatibility
LegacySolverReturn = SolverReturnTuple
"""Deprecated: Use SolverReturnTuple instead."""
