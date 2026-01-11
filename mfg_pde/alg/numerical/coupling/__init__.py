"""
Coupled MFG system solvers using numerical methods.

This module provides complete Mean Field Games solvers that combine HJB and FP
solvers to solve the coupled MFG system using classical numerical approaches:
- Fixed point iterators (Picard iteration-based)
- Fictitious Play (decaying learning rate, proven convergence)
- Newton methods (quadratic convergence near solution)
- Hybrid methods combining different techniques

Note: Particle-collocation methods have been removed from core package.
"""

from .base_mfg import BaseMFGSolver
from .fictitious_play import FictitiousPlayIterator

# Coupling iterators
from .fixed_point_iterator import FixedPointIterator

# Shared utilities
from .fixed_point_utils import (
    apply_damping,
    check_convergence_criteria,
    construct_solver_result,
    initialize_cold_start,
    preserve_initial_condition,
    preserve_terminal_condition,
)
from .hybrid_fp_particle_hjb_fdm import HybridFPParticleHJBFDM

# Newton family solvers (Issue #492)
from .mfg_residual import MFGResidual
from .newton_mfg_solver import NewtonMFGSolver

# Note: ParticleCollocationSolver has been removed from core package

__all__ = [
    "BaseMFGSolver",
    "FixedPointIterator",
    "FictitiousPlayIterator",
    "HybridFPParticleHJBFDM",
    # Newton family (Issue #492)
    "MFGResidual",
    "NewtonMFGSolver",
    # Utilities
    "apply_damping",
    "check_convergence_criteria",
    "construct_solver_result",
    "initialize_cold_start",
    "preserve_initial_condition",
    "preserve_terminal_condition",
]

# Solver categories for factory selection
FIXED_POINT_SOLVERS = [
    "FixedPointIterator",  # Picard iteration with fixed damping
    "FictitiousPlayIterator",  # Fictitious play with decaying learning rate
]

# Newton family solvers (Issue #492)
NEWTON_SOLVERS = [
    "NewtonMFGSolver",  # Newton's method for MFG coupling
]

# Note: PARTICLE_SOLVERS category removed - removed from core package

HYBRID_SOLVERS = [
    "HybridFPParticleHJBFDM",  # FP-Particle + HJB-FDM hybrid solver
]

# JAX-accelerated solvers (optional)
JAX_SOLVERS: list[str] = []

ALL_MFG_SOLVERS = FIXED_POINT_SOLVERS + NEWTON_SOLVERS + HYBRID_SOLVERS + JAX_SOLVERS
