"""
Minimal core infrastructure for MFG problems.

This module provides only the essential base classes and protocols.
Framework extensions (network, variational, multi-population) are in mfg_pde.extensions.
"""

from mfg_pde.geometry import BoundaryConditions

from .base_problem import CartesianGridMFGProtocol, MFGProblemProtocol
from .mfg_problem import MFGComponents, MFGProblem

__all__ = [
    # Geometry
    "BoundaryConditions",
    # Core protocols
    "MFGProblemProtocol",
    "CartesianGridMFGProtocol",
    # Base MFG problem
    "MFGProblem",
    "MFGComponents",
    # Capacity-constrained MFG
    "CapacityConstrainedMFGProblem",
    "CongestionModel",
    "QuadraticCongestion",
    "ExponentialCongestion",
    "LogBarrierCongestion",
    "PiecewiseCongestion",
    "create_congestion_model",
]
