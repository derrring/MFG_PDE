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
from .enhanced_particle_collocation_solver import EnhancedParticleCollocationSolver, MonitoredParticleCollocationSolver
from .particle_collocation_solver import ParticleCollocationSolver

# Backward compatibility alias
DampedFixedPointIterator = FixedPointIterator

__all__ = [
    # Fixed Point Iterators
    "ConfigAwareFixedPointIterator",
    "FixedPointIterator",
    "DampedFixedPointIterator",  # Alias
    # Particle Collocation Solvers
    "ParticleCollocationSolver",
    "AdaptiveParticleCollocationSolver",
    "MonitoredParticleCollocationSolver",
    "EnhancedParticleCollocationSolver",
]

# Solver categories for factory selection
FIXED_POINT_SOLVERS = [
    "ConfigAwareFixedPointIterator",
    "FixedPointIterator",
    "DampedFixedPointIterator",  # Alias
]

PARTICLE_SOLVERS = [
    "ParticleCollocationSolver",
    "AdaptiveParticleCollocationSolver",
    "MonitoredParticleCollocationSolver",
    "EnhancedParticleCollocationSolver",
]

ALL_MFG_SOLVERS = FIXED_POINT_SOLVERS + PARTICLE_SOLVERS
