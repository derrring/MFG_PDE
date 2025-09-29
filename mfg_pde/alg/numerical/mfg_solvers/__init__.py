"""
Coupled MFG system solvers using numerical methods.

This module provides complete Mean Field Games solvers that combine HJB and FP
solvers to solve the coupled MFG system using classical numerical approaches:
- Fixed point iterators (Picard iteration-based)
- Particle collocation methods (meshfree approaches)
- Hybrid methods combining different techniques
- Adaptive and enhanced variants
"""

from .adaptive_particle_collocation_solver import AdaptiveParticleCollocationSolver
from .base_mfg import BaseMFGSolver
from .config_aware_fixed_point_iterator import ConfigAwareFixedPointIterator

# Additional solvers will be imported as they are migrated
from .fixed_point_iterator import FixedPointIterator
from .hybrid_fp_particle_hjb_fdm import HybridFPParticleHJBFDM
from .monitored_particle_collocation_solver import MonitoredParticleCollocationSolver
from .particle_collocation_solver import ParticleCollocationSolver

__all__ = [
    "BaseMFGSolver",
    # Fixed Point Iterators
    "ConfigAwareFixedPointIterator",
    # Additional solvers will be added as they are migrated
    "FixedPointIterator",
    "ParticleCollocationSolver",
    "AdaptiveParticleCollocationSolver",
    "MonitoredParticleCollocationSolver",
    "HybridFPParticleHJBFDM",
]

# Solver categories for factory selection
FIXED_POINT_SOLVERS = [
    "ConfigAwareFixedPointIterator",
    "FixedPointIterator",  # Classic damped fixed point iterator
]

PARTICLE_SOLVERS = [
    "ParticleCollocationSolver",  # Particle-GFDM collocation solver
    "AdaptiveParticleCollocationSolver",  # Adaptive convergence particle solver
    "MonitoredParticleCollocationSolver",  # Enhanced monitoring particle solver
]

HYBRID_SOLVERS = [
    "HybridFPParticleHJBFDM",  # FP-Particle + HJB-FDM hybrid solver
]

# JAX-accelerated solvers (optional)
JAX_SOLVERS = []

ALL_MFG_SOLVERS = FIXED_POINT_SOLVERS + PARTICLE_SOLVERS + HYBRID_SOLVERS + JAX_SOLVERS
