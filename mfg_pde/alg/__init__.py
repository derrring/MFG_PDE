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

from typing import Any

# Base abstract solver
from .base_mfg_solver import MFGSolver
from .fp_solvers import BaseFPSolver, FPFDMSolver, FPParticleSolver

# Specialized solvers for advanced usage
from .hjb_solvers import BaseHJBSolver, HJBFDMSolver, HJBGFDMSolver, HJBSemiLagrangianSolver

# Complete MFG solvers (combinations of HJB + FP)
from .mfg_solvers import (  # Fixed Point Iterators; Particle Collocation Solvers; Solver categories
    ALL_MFG_SOLVERS,
    FIXED_POINT_SOLVERS,
    PARTICLE_SOLVERS,
    AdaptiveParticleCollocationSolver,
    ConfigAwareFixedPointIterator,
    FixedPointIterator,
    MonitoredParticleCollocationSolver,
    ParticleCollocationSolver,
)


def create_adaptive_particle_solver(**kwargs: Any) -> AdaptiveParticleCollocationSolver:
    """Create adaptive particle solver with backward compatibility."""
    return AdaptiveParticleCollocationSolver(**kwargs)


def create_enhanced_solver(**kwargs: Any) -> MonitoredParticleCollocationSolver:
    """Create enhanced particle solver with backward compatibility."""
    return MonitoredParticleCollocationSolver(**kwargs)


__all__ = [
    # Solver categories
    "ALL_MFG_SOLVERS",
    "FIXED_POINT_SOLVERS",
    "PARTICLE_SOLVERS",
    # Complete MFG solvers
    "AdaptiveParticleCollocationSolver",
    # Specialized solvers
    "BaseFPSolver",
    "BaseHJBSolver",
    "ConfigAwareFixedPointIterator",
    "FPFDMSolver",
    "FPParticleSolver",
    "FixedPointIterator",
    "HJBFDMSolver",
    "HJBGFDMSolver",
    "HJBSemiLagrangianSolver",
    # Base class
    "MFGSolver",
    "MonitoredParticleCollocationSolver",
    "ParticleCollocationSolver",
    # Utilities
    "create_adaptive_particle_solver",
    "create_enhanced_solver",
]
