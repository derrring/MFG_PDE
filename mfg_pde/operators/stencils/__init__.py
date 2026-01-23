"""
Finite Difference Stencils for MFG_PDE.

This module provides low-level stencil implementations with fixed coefficients.
Stencils are the building blocks for differential operators.

Conceptual Distinction:
    - **Stencils** (this module): Fixed coefficient formulas
        e.g., central diff = [-1, 0, 1] / (2h)
    - **Reconstruction** (operators/reconstruction/): Adaptive strategies
        e.g., WENO combines multiple stencils based on smoothness

Available Stencils:
    First-order derivatives:
        - gradient_central: 2nd-order, symmetric
        - gradient_forward: 1st-order, positive bias
        - gradient_backward: 1st-order, negative bias
        - gradient_upwind: Godunov selection for stability

    Second-order derivatives:
        - laplacian_stencil_1d: Standard 3-point stencil
        - laplacian_stencil_nd: Sum of 1D stencils

    Utilities:
        - fix_boundaries_one_sided: Correct boundary values
        - get_gradient_stencil_coefficients: For matrix assembly
        - get_laplacian_stencil_coefficients: For matrix assembly

Usage:
    >>> from mfg_pde.operators.stencils import gradient_central, gradient_upwind
    >>> du_dx = gradient_central(u, axis=0, h=0.1)

Note:
    For boundary-aware operators with LinearOperator interface,
    use mfg_pde.operators.differential instead.

Created: 2026-01-24 (Operator module reorganization)
"""

from mfg_pde.operators.stencils.finite_difference import (
    fix_boundaries_one_sided,
    get_gradient_stencil_coefficients,
    get_laplacian_stencil_coefficients,
    gradient_backward,
    gradient_central,
    gradient_forward,
    gradient_upwind,
    laplacian_stencil_1d,
    laplacian_stencil_nd,
)

__all__ = [
    # First-order derivatives
    "gradient_central",
    "gradient_forward",
    "gradient_backward",
    "gradient_upwind",
    # Boundary handling
    "fix_boundaries_one_sided",
    # Second-order derivatives
    "laplacian_stencil_1d",
    "laplacian_stencil_nd",
    # Coefficients for matrix assembly
    "get_gradient_stencil_coefficients",
    "get_laplacian_stencil_coefficients",
]
