"""
High-Order Reconstruction Strategies for MFG_PDE.

This module provides adaptive reconstruction methods for computing derivatives
with higher-order accuracy near discontinuities:

    - WENO (Weighted Essentially Non-Oscillatory): Adaptive stencil weighting
    - ENO (Essentially Non-Oscillatory): Stencil selection (future)

Conceptual Distinction:
    - Stencils: Fixed coefficient formulas (e.g., central diff = [-1, 0, 1]/(2h))
    - Reconstruction: Adaptive strategies that combine stencils based on local smoothness

WENO Philosophy:
    WENO is not a "scheme" in the traditional sense but a reconstruction strategy.
    It adaptively weights multiple candidate stencils to achieve:
    - High-order accuracy in smooth regions
    - Non-oscillatory behavior near discontinuities

Usage:
    >>> from mfg_pde.operators.reconstruction import compute_weno5_derivative_1d
    >>> du_dx = compute_weno5_derivative_1d(u, dx, direction=1)
"""

from mfg_pde.operators.reconstruction.weno import (
    compute_weno5_derivative_1d,
    compute_weno5_godunov_upwind_1d,
)

__all__ = [
    "compute_weno5_derivative_1d",
    "compute_weno5_godunov_upwind_1d",
]
