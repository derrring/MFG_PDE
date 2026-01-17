"""
Geometric operators for MFG problems.

This module provides:
1. Projection operators for transferring solutions between geometries
2. Differential operators as scipy LinearOperator classes
3. Operator trait implementations for geometry-agnostic solver design

LinearOperator Classes (Issue #595 Phase 2 - COMPLETE):
    - LaplacianOperator: Δu = ∇²u (2nd-order elliptic)
    - GradientComponentOperator: ∂u/∂xᵢ (1st-order)
    - DivergenceOperator: ∇·F (1st-order, vector → scalar)
    - AdvectionOperator: v·∇m or ∇·(vm) (1st-order transport)
    - InterpolationOperator: Grid → query points (non-square)

All operators implement scipy.sparse.linalg.LinearOperator interface:
    - Matrix-vector product: L @ u_flat
    - Callable interface: L(u) (preserves field shape)
    - Operator algebra: L1 + L2, α*L, L1 @ L2
    - Iterative solvers: gmres(L, b), cg(L, b)

Usage:
    >>> from mfg_pde.geometry.operators import LaplacianOperator, AdvectionOperator
    >>> # Create operators
    >>> L = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(50, 50), bc=bc)
    >>> adv = AdvectionOperator(v, spacings=[0.1, 0.1], field_shape=(50, 50))
    >>>
    >>> # Apply operators
    >>> Lu = L(u)  # Callable interface
    >>> adv_u = adv @ u.ravel()  # scipy interface
    >>>
    >>> # Use with iterative solvers
    >>> u_solution, info = gmres(L, b)
"""

from .advection import AdvectionOperator
from .divergence import DivergenceOperator
from .gradient import GradientComponentOperator, create_gradient_operators
from .interpolation import InterpolationOperator
from .laplacian import LaplacianOperator
from .projection import GeometryProjector, ProjectionRegistry

__all__ = [
    # Projection
    "GeometryProjector",
    "ProjectionRegistry",
    # Differential operators
    "LaplacianOperator",
    "GradientComponentOperator",
    "DivergenceOperator",
    "AdvectionOperator",
    "InterpolationOperator",
    # Factory functions
    "create_gradient_operators",
]
