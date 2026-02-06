"""
Level Set Methods for Free Boundary Problems in MFG.

This module provides level set infrastructure for Tier 3 BCs (free boundary problems),
enabling MFG on domains with evolving boundaries.

Key Applications:
- Stefan problems (phase transitions)
- Crowd-driven domain expansion (expanding exits)
- Moving obstacles (dynamic environments)

Design Philosophy:
    Use composition over inheritance - time-dependent geometry as a thin wrapper
    around static ImplicitDomain instances, avoiding changes to existing solvers.

Core Components:
- LevelSetFunction: Container for φ, normals, curvature
- LevelSetEvolver: Solve ∂φ/∂t + V|∇φ| = 0 via Godunov upwind
- TimeDependentDomain: Manage φ(t) time series

Architecture Pattern:
    TimeDependentDomain wraps ImplicitDomain φ(t)
        ↓
    Update geometry → Resolve PDE (existing solvers unchanged)
        ↓
    Leverages existing infrastructure (Issues #595-598)

Mathematical Background:
    Level set evolution:
        ∂φ/∂t + V|∇φ| = 0

    Reinitialization (maintain SDF property):
        ∂φ/∂τ + sign(φ₀)(|∇φ| - 1) = 0

    Mean curvature:
        κ = ∇·(∇φ/|∇φ|) = ∇·n

References:
- Osher & Sethian (1988): Fronts propagating with curvature-dependent speed
- Osher & Fedkiw (2003): Level Set Methods and Dynamic Implicit Surfaces
- Sethian (1999): Level Set Methods and Fast Marching Methods

Created: 2026-01-18 (Issue #592 - Level Set Methods)
Part of: Issue #592 Phase 3.1 - Level Set Infrastructure
"""

from mfg_pde.geometry.level_set.core import LevelSetEvolver, LevelSetFunction
from mfg_pde.geometry.level_set.curvature import compute_curvature
from mfg_pde.geometry.level_set.eikonal import (
    EikonalSolver,
    FastMarchingMethod,
    FastSweepingMethod,
)
from mfg_pde.geometry.level_set.reinitialization import reinitialize
from mfg_pde.geometry.level_set.time_dependent_domain import TimeDependentDomain

__all__ = [
    # Core level set classes
    "LevelSetFunction",
    "LevelSetEvolver",
    "TimeDependentDomain",
    # Geometry computations
    "compute_curvature",
    "reinitialize",
    # Eikonal solvers (Issue #664)
    "EikonalSolver",
    "FastMarchingMethod",
    "FastSweepingMethod",
]
