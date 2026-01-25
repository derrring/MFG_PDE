"""
Differential Operators for MFG_PDE.

This module provides differential operators as scipy LinearOperator classes:
    - LaplacianOperator: Laplacian (second derivative)
    - DiffusionOperator: Unified diffusion operator (scalar or tensor)
    - PartialDerivOperator: Partial derivative (d/dxi)
    - DivergenceOperator: Divergence of vector fields
    - AdvectionOperator: Advection/transport operator
    - InterfaceJumpOperator: Jump conditions across interfaces

All operators support:
    - scipy.sparse.linalg.LinearOperator interface
    - Callable interface: L(u) preserves field shape
    - Boundary condition handling

Usage:
    >>> from mfg_pde.operators.differential import LaplacianOperator, DiffusionOperator
    >>> L = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(50, 50), bc=bc)
    >>> Lu = L(u)  # Callable interface
    >>> Lu_flat = L @ u.ravel()  # scipy interface
    >>>
    >>> # Unified diffusion (scalar or tensor coefficient)
    >>> D = DiffusionOperator(coefficient=sigma, spacings=[dx, dy],
    ...                       field_shape=(Nx, Ny), bc=bc)
    >>> Du = D(u)
    >>>
    >>> # Partial derivatives (gradient components)
    >>> from mfg_pde.operators.differential import PartialDerivOperator, create_gradient_operators
    >>> grad_x, grad_y = create_gradient_operators(spacings=[dx, dy], field_shape=(Nx, Ny))
    >>> du_dx = grad_x(u)
"""

from mfg_pde.operators.differential.advection import AdvectionOperator
from mfg_pde.operators.differential.diffusion import DiffusionOperator
from mfg_pde.operators.differential.divergence import DivergenceOperator
from mfg_pde.operators.differential.gradient import (
    GradientComponentOperator,  # Deprecated alias (v0.18.0)
    PartialDerivOperator,
    create_gradient_operators,
)
from mfg_pde.operators.differential.interface_jump import InterfaceJumpOperator
from mfg_pde.operators.differential.laplacian import LaplacianOperator

__all__ = [
    # Second-order operators
    "LaplacianOperator",
    "DiffusionOperator",
    # First-order operators
    "PartialDerivOperator",
    "GradientComponentOperator",  # Deprecated alias for PartialDerivOperator
    "DivergenceOperator",
    "AdvectionOperator",
    # Interface operators
    "InterfaceJumpOperator",
    # Factory functions
    "create_gradient_operators",
]
