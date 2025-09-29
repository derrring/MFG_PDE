"""
Numerical methods paradigm for MFG problems.

This module contains classical numerical analysis approaches:
- hjb_solvers: Individual HJB equation solvers
- fp_solvers: Individual Fokker-Planck solvers
- mfg_solvers: Coupled system numerical methods

All methods are based on discretization and convergence analysis.
Note: FP solvers maintain backward compatibility with original interfaces.
"""

from mfg_pde.alg.base_solver import BaseNumericalSolver

# Import FP solvers
from .fp_solvers import (
    BaseFPSolver,
    FPFDMSolver,
    FPNetworkSolver,
    FPParticleSolver,
)

# Import HJB solvers
from .hjb_solvers import (
    BaseHJBSolver,
    HJBFDMSolver,
    HJBGFDMSolver,
    HJBSemiLagrangianSolver,
    HJBWenoSolver,
)

# Import MFG solvers (coupled system solvers)
from .mfg_solvers import (
    AdaptiveParticleCollocationSolver,
    BaseMFGSolver,
    ConfigAwareFixedPointIterator,
    FixedPointIterator,
    HybridFPParticleHJBFDM,
    MonitoredParticleCollocationSolver,
    ParticleCollocationSolver,
)

__all__ = [
    "BaseNumericalSolver",
    # HJB Solvers
    "BaseHJBSolver",
    "HJBFDMSolver",
    "HJBGFDMSolver",
    "HJBSemiLagrangianSolver",
    "HJBWenoSolver",
    # FP Solvers
    "BaseFPSolver",
    "FPFDMSolver",
    "FPNetworkSolver",
    "FPParticleSolver",
    # MFG Solvers (coupled system solvers)
    "BaseMFGSolver",
    "ConfigAwareFixedPointIterator",
    "FixedPointIterator",
    "ParticleCollocationSolver",
    "AdaptiveParticleCollocationSolver",
    "MonitoredParticleCollocationSolver",
    "HybridFPParticleHJBFDM",
]
