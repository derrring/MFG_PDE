"""
Numerical methods paradigm for MFG problems.

This module contains classical numerical analysis approaches:
- hjb_solvers: Individual HJB equation solvers
- fp_solvers: Individual Fokker-Planck solvers
- coupling: MFG coupling methods (Picard, Policy iteration, etc.)

All methods are based on discretization and convergence analysis.
Note: FP solvers maintain backward compatibility with original interfaces.
"""

from mfg_pde.alg.base_solver import BaseNumericalSolver

# Import MFG coupling methods
# Note: ParticleCollocationSolver has been removed from core package
from .coupling import (
    BaseMFGSolver,
    FixedPointIterator,
    HybridFPParticleHJBFDM,
)

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

__all__ = [
    # FP Solvers
    "BaseFPSolver",
    # HJB Solvers
    "BaseHJBSolver",
    # MFG Solvers (coupled system solvers)
    "BaseMFGSolver",
    "BaseNumericalSolver",
    "FPFDMSolver",
    "FPNetworkSolver",
    "FPParticleSolver",
    "FixedPointIterator",
    "HJBFDMSolver",
    "HJBGFDMSolver",
    "HJBSemiLagrangianSolver",
    "HJBWenoSolver",
    "HybridFPParticleHJBFDM",
    # Note: ParticleCollocationSolver has been removed from core package
]
