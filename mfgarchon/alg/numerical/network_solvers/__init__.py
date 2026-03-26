"""
Network MFG Solvers.

This module contains solvers for Mean Field Games on graph/network domains,
which are fundamentally different from grid-based numerical solvers.

Network MFG problems operate on discrete graph structures rather than
continuous spatial domains, requiring specialized algorithms.

Solvers:
- HJBNetworkSolver: HJB solver for network/graph domains
- FPNetworkSolver: Fokker-Planck solver for network/graph domains
"""

from .fp_network import FPNetworkSolver
from .hjb_network import NetworkHJBSolver

# Alias for consistent naming convention
HJBNetworkSolver = NetworkHJBSolver

__all__ = [
    "FPNetworkSolver",
    "HJBNetworkSolver",  # Alias
    "NetworkHJBSolver",  # Original name
]
