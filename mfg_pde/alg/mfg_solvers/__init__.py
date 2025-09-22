"""
MFG Solvers Package.

This package contains complete Mean Field Games solvers that combine HJB and FP solvers
to solve the coupled MFG system. All solvers inherit from the base_mfg_solver abstract class.

Solver Types:
- Fixed Point Iterators: Picard iteration-based solvers
- Particle Collocation: Meshfree particle-based solvers
- Enhanced Solvers: Adaptive and monitored variants

Architecture:
- Each MFG solver combines HJB solver + FP solver
- All inherit from BaseMFGSolver in ../base_mfg_solver.py
- Use specialized solvers from ../hjb_solvers/ and ../fp_solvers/
"""

from .adaptive_particle_collocation_solver import AdaptiveParticleCollocationSolver
from .config_aware_fixed_point_iterator import ConfigAwareFixedPointIterator
from .damped_fixed_point_iterator import FixedPointIterator
from .enhanced_particle_collocation_solver import MonitoredParticleCollocationSolver
from .hybrid_fp_particle_hjb_fdm import HybridFPParticleHJBFDM
from .particle_collocation_solver import ParticleCollocationSolver

__all__ = [
    # Particle Collocation Solvers
    "AdaptiveParticleCollocationSolver",
    # Fixed Point Iterators
    "ConfigAwareFixedPointIterator",
    "FixedPointIterator",
    # Hybrid Solvers
    "HybridFPParticleHJBFDM",
    "MonitoredParticleCollocationSolver",
    "ParticleCollocationSolver",
]

# Solver categories for factory selection
FIXED_POINT_SOLVERS = [
    "ConfigAwareFixedPointIterator",
    "FixedPointIterator",
]

PARTICLE_SOLVERS = [
    "ParticleCollocationSolver",
    "AdaptiveParticleCollocationSolver",
    "MonitoredParticleCollocationSolver",
    "EnhancedParticleCollocationSolver",
]

HYBRID_SOLVERS = [
    "HybridFPParticleHJBFDM",
]

ALL_MFG_SOLVERS = FIXED_POINT_SOLVERS + PARTICLE_SOLVERS + HYBRID_SOLVERS
