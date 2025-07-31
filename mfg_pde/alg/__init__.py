"""
Algorithm Package for MFG_PDE.

This package contains the core algorithms for solving Mean Field Games:

Organization:
- base_mfg_solver.py: Abstract base class for all MFG solvers
- mfg_solvers/: Complete MFG solvers (HJB + FP combined)
- hjb_solvers/: Specialized Hamilton-Jacobi-Bellman solvers
- fp_solvers/: Specialized Fokker-Planck solvers

Architecture:
- MFG solvers inherit from BaseMFGSolver and combine HJB + FP solvers
- Specialized solvers can be used independently or within MFG solvers
- All solvers support the unified problem interface
"""

# Base abstract solver
from .base_mfg_solver import MFGSolver
from .fp_solvers import BaseFPSolver, FPFDMSolver, FPParticleSolver

# Specialized solvers for advanced usage
from .hjb_solvers import (
    BaseHJBSolver,
    HJBFDMSolver,
    HJBGFDMSolver,
    HJBSemiLagrangianSolver,
)

# Complete MFG solvers (combinations of HJB + FP)
from .mfg_solvers import (  # Fixed Point Iterators; Particle Collocation Solvers; Solver categories
    AdaptiveParticleCollocationSolver,
    ALL_MFG_SOLVERS,
    ConfigAwareFixedPointIterator,
    EnhancedParticleCollocationSolver,
    FIXED_POINT_SOLVERS,
    FixedPointIterator,
    MonitoredParticleCollocationSolver,
    PARTICLE_SOLVERS,
    ParticleCollocationSolver,
)

# Backward compatibility aliases
DampedFixedPointIterator = FixedPointIterator

# Additional backward compatibility
QuietAdaptiveParticleCollocationSolver = AdaptiveParticleCollocationSolver
HighPrecisionAdaptiveParticleCollocationSolver = AdaptiveParticleCollocationSolver
SilentAdaptiveParticleCollocationSolver = AdaptiveParticleCollocationSolver


def create_adaptive_particle_solver(**kwargs):
    """Create adaptive particle solver with backward compatibility."""
    return AdaptiveParticleCollocationSolver(**kwargs)


def create_enhanced_solver(**kwargs):
    """Create enhanced particle solver with backward compatibility."""
    return EnhancedParticleCollocationSolver(**kwargs)


__all__ = [
    # Base class
    "MFGSolver",
    # Complete MFG solvers
    "ConfigAwareFixedPointIterator",
    "FixedPointIterator",
    "DampedFixedPointIterator",  # Alias
    "ParticleCollocationSolver",
    "AdaptiveParticleCollocationSolver",
    "MonitoredParticleCollocationSolver",
    "EnhancedParticleCollocationSolver",
    # Specialized solvers
    "BaseHJBSolver",
    "HJBFDMSolver",
    "HJBGFDMSolver",
    "HJBSemiLagrangianSolver",
    "BaseFPSolver",
    "FPFDMSolver",
    "FPParticleSolver",
    # Utilities
    "create_adaptive_particle_solver",
    "create_enhanced_solver",
    # Solver categories
    "FIXED_POINT_SOLVERS",
    "PARTICLE_SOLVERS",
    "ALL_MFG_SOLVERS",
    # Backward compatibility
    "QuietAdaptiveParticleCollocationSolver",
    "HighPrecisionAdaptiveParticleCollocationSolver",
    "SilentAdaptiveParticleCollocationSolver",
]
