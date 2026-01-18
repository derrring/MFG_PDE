"""
WENO (Weighted Essentially Non-Oscillatory) Numerical Methods.

This module previously contained WENO5Gradient (Issue #605), which has been
refactored into the operator framework (Issue #606).

For WENO5 gradient computation, use:
    from mfg_pde.geometry.operators import create_gradient_operators
    grad_x, = create_gradient_operators(spacings=[dx], field_shape=(N,), scheme="weno5")

Created: 2026-01-18 (Issue #605 Phase 2.1)
Refactored: 2026-01-18 (Issue #606 - WENO5 Operator Integration)
"""

__all__: list[str] = []
