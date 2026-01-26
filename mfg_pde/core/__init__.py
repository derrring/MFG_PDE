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
    # Control cost base classes (original)
    BoundedControlCost,
    BoundedHamiltonian,
    ControlCostBase,
    # Backward-compatible aliases (deprecated since v0.17.2)
    DefaultMFGHamiltonian,
    # Dual classes (Legendre transform)
    DualHamiltonian,
    DualLagrangian,
    # Full MFG Hamiltonian classes (Issue #673)
    HamiltonianBase,
    HamiltonianState,
    InverseLegendreeLagrangian,
    L1ControlCost,
    L1Hamiltonian,
    # Lagrangian classes (Issue #651)
    LagrangianBase,
    LegendreHamiltonian,
    # Common base (Issue #651)
    MFGOperatorBase,
    OptimizationSense,
    QuadraticControlCost,
    QuadraticHamiltonian,
    QuadraticMFGHamiltonian,
    SeparableHamiltonian,
    create_hamiltonian,
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
    # Hamiltonian/Lagrangian (Issues #623, #651, #667, #673)
    "OptimizationSense",
    # Common base for H and L (Issue #651)
    "MFGOperatorBase",
    # Control cost classes (original, H(p) only)
    "ControlCostBase",
    "QuadraticControlCost",
    "QuadraticHamiltonian",
    "L1ControlCost",
    "L1Hamiltonian",
    "BoundedControlCost",
    "BoundedHamiltonian",
    # Full MFG Hamiltonian classes (Issue #673)
    "HamiltonianBase",
    "HamiltonianState",
    "SeparableHamiltonian",
    "QuadraticMFGHamiltonian",
    "create_hamiltonian",
    # Lagrangian classes (Issue #651)
    "LagrangianBase",
    # Dual classes (Legendre transform)
    "DualHamiltonian",
    "DualLagrangian",
    # Backward-compatible aliases (deprecated since v0.17.2)
    "DefaultMFGHamiltonian",
    "LegendreHamiltonian",
    "InverseLegendreeLagrangian",
    # Capacity-constrained MFG
    "CapacityConstrainedMFGProblem",
    "CongestionModel",
    "QuadraticCongestion",
    "ExponentialCongestion",
    "LogBarrierCongestion",
    "PiecewiseCongestion",
    "create_congestion_model",
]
