"""
Internal Type Definitions - Maintainers Only

⚠️  DO NOT IMPORT THESE TYPES IN USER CODE

This module contains the complex Union types and internal type definitions
that were previously scattered throughout the codebase. These are used
internally by the library but should not be part of the public API.

If you need to customize solver behavior, use the hooks system:
    from mfg_pde.hooks import SolverHooks

If you need type annotations for advanced usage, use:
    from mfg_pde.types import SpatialTemporalState, MFGProblem
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mfg_pde.core.variational_mfg_problem import VariationalMFGProblem
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.core.network_mfg_problem import NetworkMFGProblem
    from mfg_pde.geometry import BoundaryConditions
    from mfg_pde.geometry.base_geometry import BaseGeometry
    from mfg_pde.types.protocols import MFGResult

# === Complex Union Types (previously causing user confusion) ===

# Flexible parameter inputs
type FlexibleFloat = float | int | np.floating | np.integer
type FlexibleArray = NDArray | list[float] | tuple[float, ...]
type FlexibleCallable = Callable | str | None

# Solver input flexibility
type SolverInput = (
    "MFGProblem"
    | dict[str, Any]  # Configuration dict
    | str  # Problem type string
    | object  # Any custom problem object
)

type ConfigValue = float | int | str | bool | None | list[float | int | str] | dict[str, Any] | Callable | NDArray

# Complex function signatures (internal use only)
type HamiltonianLike = (
    Callable[[float, float, float, float], float]  # Standard H(x,p,m,t)
    | Callable[[float, float, float], float]  # H(x,p,m) - time-independent
    | Callable[[float, float], float]  # H(x,p) - no coupling
    | str  # Preset Hamiltonian name
    | None  # Use default
)

type LagrangianLike = (
    Callable[[float, float, float, float], float]  # Standard L(x,v,m,t)
    | Callable[[float, float, float], float]  # L(x,v,m) - time-independent
    | Callable[[float, float], float]  # L(x,v) - no coupling
    | str  # Preset Lagrangian name
    | None  # Use default
)

type BoundaryConditionLike = "BoundaryConditions | str | dict[str, Any] | None"

# Solver return types (internal flexibility)
type SolverReturnType = (
    tuple[NDArray, NDArray, dict[str, Any]]  # Legacy (U, M, info) format
    | dict[str, Any]  # New dict format
    | "MFGResult"  # Protocol-compliant result
    | object  # Custom result object
)

# Network/geometry flexibility
type GeometryLike = "BaseGeometry | str | dict[str, Any] | tuple[float, float] | float"

# Backend/implementation flexibility
type BackendType = (
    str  # Backend name like "numpy", "jax", "scipy"
    | object  # Backend object/module
    | None  # Use default
)

# === Deprecated Type Aliases (for backward compatibility) ===

# These were used in the old codebase and should be gradually phased out
type LegacyMFGProblem = "MFGProblem | VariationalMFGProblem | NetworkMFGProblem"
type LegacySolverType = str | object | Callable
type LegacyConfigType = dict[str, Any] | object | None

# === Internal Solver State Types ===

# Complex state representations that solvers might use internally
type InternalSolverState = (
    # Simple state
    tuple[NDArray, NDArray]  # (u, m)
    |
    # Extended state
    tuple[NDArray, NDArray, dict[str, Any]]  # (u, m, metadata)
    |
    # Complex state
    dict[str, NDArray | float | int | Any]  # Full state dictionary
    |
    # Custom state objects
    object
)

# Intermediate computation results
type IntermediateResult = (
    NDArray  # Simple array result
    | tuple[NDArray, ...]  # Multiple arrays
    | dict[str, NDArray]  # Named array collection
    | Any  # Completely flexible
)

# === Error and Exception Types ===

# Types for error handling and validation
type ValidationResult = bool | str | Exception | None
type ErrorCallback = Callable[[Exception], None] | None

# === Migration Helpers ===


def migrate_legacy_config(old_config: Any) -> dict[str, Any]:
    """
    Convert legacy configuration formats to new standard format.

    This function helps with backward compatibility during the transition.
    """
    if isinstance(old_config, dict):
        return old_config
    elif hasattr(old_config, "__dict__"):
        return vars(old_config)
    else:
        return {"legacy_config": old_config}


def normalize_solver_input(input_obj: Any) -> dict[str, Any]:
    """
    Normalize various input formats to standard dictionary format.

    This is used internally to handle the flexible input types.
    """
    if isinstance(input_obj, dict):
        return input_obj
    elif isinstance(input_obj, str):
        return {"problem_type": input_obj}
    elif hasattr(input_obj, "get_domain_bounds"):  # MFGProblem-like
        return {"problem": input_obj}
    else:
        return {"custom_input": input_obj}
