"""
Fokker-Planck equation solvers using numerical methods.

This module provides classical numerical analysis approaches for solving
individual Fokker-Planck equations, including:
- Finite difference methods (FPFDMSolver)
- Particle-based methods (FPParticleSolver)
- GFDM meshfree methods (FPGFDMSolver)

Note: Network solvers have been moved to `mfg_pde.alg.numerical.network_solvers`.
FPNetworkSolver is re-exported here for backward compatibility.
"""

# Re-export for backward compatibility (moved to network_solvers)
from mfg_pde.alg.numerical.network_solvers import FPNetworkSolver

from .base_fp import BaseFPSolver
from .fp_fdm import FPFDMSolver
from .fp_gfdm import FPGFDMSolver
from .fp_particle import FPParticleSolver, KDENormalization, ParticleMode

__all__ = [
    "BaseFPSolver",
    "FPFDMSolver",
    "FPGFDMSolver",
    "FPNetworkSolver",  # Backward compat - prefer network_solvers import
    "FPParticleSolver",
    "KDENormalization",
    "ParticleMode",
]
