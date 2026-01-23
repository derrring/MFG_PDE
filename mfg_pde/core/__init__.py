"""
Minimal core infrastructure for MFG problems.

This module provides only the essential base classes and protocols.
Framework extensions (network, variational, multi-population) are in mfg_pde.extensions.
"""

from mfg_pde.geometry import BoundaryConditions

from .base_problem import CartesianGridMFGProtocol, MFGProblemProtocol
from .derivatives import (
    DerivativeTensors,
    from_multi_index_dict,
    to_multi_index_dict,
)
from .hamiltonian import (
    BoundedControlCost,
    BoundedHamiltonian,
    ControlCostBase,
    L1ControlCost,
    L1Hamiltonian,
    OptimizationSense,
    QuadraticControlCost,
    QuadraticHamiltonian,
)
from .mfg_components import (
    ConditionsMixin,
    HamiltonianMixin,
    MFGComponents,
)
from .mfg_problem import MFGProblem

__all__ = [
    # Geometry
    "BoundaryConditions",
    # Core protocols
    "MFGProblemProtocol",
    "CartesianGridMFGProtocol",
    # Derivative tensors
    "DerivativeTensors",
    "from_multi_index_dict",
    "to_multi_index_dict",
    # Base MFG problem
    "MFGProblem",
    "MFGComponents",
    # Mixins
    "HamiltonianMixin",
    "ConditionsMixin",
    # Hamiltonian/Lagrangian (Issue #623)
    "OptimizationSense",
    "ControlCostBase",
    "QuadraticControlCost",
    "QuadraticHamiltonian",
    "L1ControlCost",
    "L1Hamiltonian",
    "BoundedControlCost",
    "BoundedHamiltonian",
    # Capacity-constrained MFG
    "CapacityConstrainedMFGProblem",
    "CongestionModel",
    "QuadraticCongestion",
    "ExponentialCongestion",
    "LogBarrierCongestion",
    "PiecewiseCongestion",
    "create_congestion_model",
]
