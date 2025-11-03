"""
Coupled MFG system solvers using numerical methods.

This module provides complete Mean Field Games solvers that combine HJB and FP
solvers to solve the coupled MFG system using classical numerical approaches:
- Fixed point iterators (Picard iteration-based)
- Hybrid methods combining different techniques

Note: Particle-collocation methods have been removed from core package.
"""

from .base_mfg import BaseMFGSolver

# Modern fixed-point iterator (unified)
from .fixed_point_iterator import FixedPointIterator

# Shared utilities
from .fixed_point_utils import (
    apply_damping,
    check_convergence_criteria,
    construct_solver_result,
    initialize_cold_start,
    preserve_boundary_conditions,
)
from .hybrid_fp_particle_hjb_fdm import HybridFPParticleHJBFDM

# Note: ParticleCollocationSolver has been removed from core package

__all__ = [
    "BaseMFGSolver",
    "FixedPointIterator",
    "HybridFPParticleHJBFDM",
    "apply_damping",
    "check_convergence_criteria",
    "construct_solver_result",
    "initialize_cold_start",
    "preserve_boundary_conditions",
]

# Solver categories for factory selection
FIXED_POINT_SOLVERS = [
    "FixedPointIterator",  # Unified fixed point iterator (all features)
]

# Note: PARTICLE_SOLVERS category removed - removed from core package

HYBRID_SOLVERS = [
    "HybridFPParticleHJBFDM",  # FP-Particle + HJB-FDM hybrid solver
]

# JAX-accelerated solvers (optional)
JAX_SOLVERS: list[str] = []

ALL_MFG_SOLVERS = FIXED_POINT_SOLVERS + HYBRID_SOLVERS + JAX_SOLVERS
