"""
Array Type Definitions

This module consolidates all array type aliases used throughout MFG_PDE.
These provide semantic meaning and documentation for commonly used array shapes.

Usage:
    from mfg_pde.types import SolutionArray, SpatialGrid, TimeGrid

    # Or for advanced use:
    from mfg_pde.types.arrays import SpatialArray, TemporalArray
"""

from __future__ import annotations

import warnings

from numpy.typing import NDArray

# === Public Array Types (User-Facing) ===

# Core solution arrays
type SolutionArray = NDArray
"""
2D spatio-temporal solution array.

Shape: (Nt+1, Nx+1) for 1D problems
       (Nt+1, Ny+1, Nx+1) for 2D problems

Used for: u(t,x) value function, m(t,x) density function
"""

# Grid coordinate arrays
type SpatialGrid = NDArray
"""
Spatial coordinate array.

Shape: (Nx+1,) for 1D, (Ny+1, Nx+1) for 2D

Used for: x-coordinates, spatial mesh points
"""

type TimeGrid = NDArray
"""
Temporal coordinate array.

Shape: (Nt+1,)

Used for: t-coordinates, time mesh points
"""

# === Advanced Array Types (Internal Use) ===

# Alternative names for consistency with academic literature
type SpatialArray = NDArray
"""
Alias for SpatialGrid - used in some numerical methods.

Prefer SpatialGrid for new code.
"""

type TemporalArray = NDArray
"""
Alias for TimeGrid - used in some numerical methods.

Prefer TimeGrid for new code.
"""

# === Multi-dimensional Array Types ===

type Array1D = NDArray
"""1D array, shape (N,)"""

type Array2D = NDArray
"""2D array, shape (M, N)"""

type Array3D = NDArray
"""3D array, shape (K, M, N)"""

# === Specialized Array Types ===

type StateArray = NDArray
"""
State vector for neural network solvers.

Variable shape depending on architecture.
"""

type ParticleArray = NDArray
"""
Particle position array for particle methods.

Shape: (num_particles, spatial_dim)
"""

type WeightArray = NDArray
"""
Particle weight array for particle methods.

Shape: (num_particles,)
"""

type DensityArray = NDArray
"""
Discretized density function array.

Shape: (Nx+1,) for 1D, matches SpatialGrid shape
"""

# === Legacy Compatibility ===

# Map old names to new standard names for backward compatibility
# These are accessed via __getattr__ to emit deprecation warnings
_DEPRECATED_ALIASES = {
    "SpatialCoordinates": ("SpatialGrid", SpatialGrid),
    "TemporalCoordinates": ("TimeGrid", TimeGrid),
}


def __getattr__(name: str):
    """Emit deprecation warnings for legacy type aliases."""
    if name in _DEPRECATED_ALIASES:
        new_name, value = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"'{name}' is deprecated, use '{new_name}' instead. This alias will be removed in v1.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
