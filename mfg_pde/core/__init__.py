"""
Minimal core infrastructure for MFG problems.

This module provides only the essential base classes and protocols.
Framework extensions (network, variational, multi-population) are in mfg_pde.extensions.
"""

from mfg_pde.geometry import BoundaryConditions

from .base_problem import CartesianGridMFGProtocol, MFGProblemProtocol
from .mfg_components import (
    ConditionsMixin,
    HamiltonianMixin,
    MFGComponents,
    MFGComponentsMixin,  # Backward compatibility
)
from .mfg_problem import MFGProblem

__all__ = [
    # Geometry
    "BoundaryConditions",
    # Core protocols
    "MFGProblemProtocol",
    "CartesianGridMFGProtocol",
    # Base MFG problem
    "MFGProblem",
    "MFGComponents",
    # Mixins
    "HamiltonianMixin",
    "ConditionsMixin",
    "MFGComponentsMixin",  # Backward compatibility
    # Capacity-constrained MFG
    "CapacityConstrainedMFGProblem",
    "CongestionModel",
    "QuadraticCongestion",
    "ExponentialCongestion",
    "LogBarrierCongestion",
    "PiecewiseCongestion",
    "create_congestion_model",
]
