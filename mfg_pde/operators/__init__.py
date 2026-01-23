"""
Operator Framework for MFG_PDE.

This module provides mathematical operators for PDE solving:

Organization:
    differential/     - Differential operators (gradient, laplacian, divergence, advection)
    interpolation/    - Interpolation and projection operators
    stencils/         - Low-level finite difference stencils
    reconstruction/   - High-order reconstruction strategies (WENO, ENO)

Conceptual Hierarchy:
    Stencils (fixed coefficients) -> Reconstruction (adaptive) -> Differential Operators

LinearOperator Classes:
    All differential operators implement scipy.sparse.linalg.LinearOperator interface:
    - Matrix-vector product: L @ u_flat
    - Callable interface: L(u) (preserves field shape)
    - Operator algebra: L1 + L2, alpha * L, L1 @ L2
    - Iterative solvers: gmres(L, b), cg(L, b)

Usage:
    >>> from mfg_pde.operators import LaplacianOperator, GradientComponentOperator
    >>> L = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(50, 50), bc=bc)
    >>> Lu = L(u)

Created: 2026-01-24 (Operator module separation from geometry)
"""

# Differential operators
from mfg_pde.operators.differential import (
    AdvectionOperator,
    DivergenceOperator,
    GradientComponentOperator,
    InterfaceJumpOperator,
    LaplacianOperator,
    create_gradient_operators,
)

# Interpolation operators
from mfg_pde.operators.interpolation import (
    GeometryProjector,
    InterpolationOperator,
    ProjectionRegistry,
)

# Reconstruction strategies
from mfg_pde.operators.reconstruction import (
    compute_weno5_derivative_1d,
    compute_weno5_godunov_upwind_1d,
)

__all__ = [
    # Differential operators
    "LaplacianOperator",
    "GradientComponentOperator",
    "DivergenceOperator",
    "AdvectionOperator",
    "InterfaceJumpOperator",
    "create_gradient_operators",
    # Interpolation
    "InterpolationOperator",
    "GeometryProjector",
    "ProjectionRegistry",
    # Reconstruction
    "compute_weno5_derivative_1d",
    "compute_weno5_godunov_upwind_1d",
]
