"""
Coupled MFG system solvers using numerical methods.

This module provides complete Mean Field Games solvers that combine HJB and FP
solvers to solve the coupled MFG system using classical numerical approaches:
- Fixed point iterators (Picard iteration-based)
- Particle collocation methods (meshfree approaches)
- Hybrid methods combining different techniques
- Adaptive and enhanced variants
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

# Unified particle collocation solver (with optional advanced convergence)
from .particle_collocation_solver import ParticleCollocationSolver

__all__ = [
    "BaseMFGSolver",
    "FixedPointIterator",
    "HybridFPParticleHJBFDM",
    "ParticleCollocationSolver",
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

PARTICLE_SOLVERS = [
    "ParticleCollocationSolver",  # Unified particle-GFDM collocation solver (with optional advanced convergence)
]

HYBRID_SOLVERS = [
    "HybridFPParticleHJBFDM",  # FP-Particle + HJB-FDM hybrid solver
]

# JAX-accelerated solvers (optional)
JAX_SOLVERS: list[str] = []

ALL_MFG_SOLVERS = FIXED_POINT_SOLVERS + PARTICLE_SOLVERS + HYBRID_SOLVERS + JAX_SOLVERS
