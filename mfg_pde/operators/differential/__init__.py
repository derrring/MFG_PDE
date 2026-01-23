"""
Differential Operators for MFG_PDE.

This module provides differential operators as scipy LinearOperator classes:
    - LaplacianOperator: Laplacian (second derivative)
    - GradientComponentOperator: Gradient component (first derivative in one direction)
    - DivergenceOperator: Divergence of vector fields
    - AdvectionOperator: Advection/transport operator
    - InterfaceJumpOperator: Jump conditions across interfaces

All operators support:
    - scipy.sparse.linalg.LinearOperator interface
    - Callable interface: L(u) preserves field shape
    - Boundary condition handling

Usage:
    >>> from mfg_pde.operators.differential import LaplacianOperator
    >>> L = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(50, 50), bc=bc)
    >>> Lu = L(u)  # Callable interface
    >>> Lu_flat = L @ u.ravel()  # scipy interface
"""

from mfg_pde.operators.differential.advection import AdvectionOperator
from mfg_pde.operators.differential.divergence import DivergenceOperator
from mfg_pde.operators.differential.gradient import (
    GradientComponentOperator,
    create_gradient_operators,
)
from mfg_pde.operators.differential.interface_jump import InterfaceJumpOperator
from mfg_pde.operators.differential.laplacian import LaplacianOperator

__all__ = [
    "LaplacianOperator",
    "GradientComponentOperator",
    "DivergenceOperator",
    "AdvectionOperator",
    "InterfaceJumpOperator",
    "create_gradient_operators",
]
