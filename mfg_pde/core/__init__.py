"""
Minimal core infrastructure for MFG problems.

This module provides only the essential base classes and protocols.
Framework extensions (network, variational, multi-population) are in mfg_pde.extensions.
"""

from mfg_pde.geometry import BoundaryConditions

from .base_problem import CartesianGridMFGProtocol, MFGProblemProtocol
from .capacity_constrained_problem import CapacityConstrainedMFGProblem
from .congestion import (
    CongestionModel,
    ExponentialCongestion,
    LogBarrierCongestion,
    PiecewiseCongestion,
    QuadraticCongestion,
    create_congestion_model,
)
from .mfg_problem import ExampleMFGProblem, MFGComponents, MFGProblem, MFGProblemBuilder, create_mfg_problem

__all__ = [
    # Geometry
    "BoundaryConditions",
    # Core protocols
    "MFGProblemProtocol",
    "CartesianGridMFGProtocol",
    # Base MFG problem
    "MFGProblem",
    "MFGComponents",
    "MFGProblemBuilder",
    # Legacy alias
    "ExampleMFGProblem",
    # Capacity-constrained MFG
    "CapacityConstrainedMFGProblem",
    "CongestionModel",
    "QuadraticCongestion",
    "ExponentialCongestion",
    "LogBarrierCongestion",
    "PiecewiseCongestion",
    "create_congestion_model",
    # Factory
    "create_mfg_problem",
]
