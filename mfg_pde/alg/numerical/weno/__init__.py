"""
WENO (Weighted Essentially Non-Oscillatory) Numerical Methods.

This module previously contained WENO5Gradient (Issue #605), which has been
refactored into the operator framework (Issue #606).

For WENO5 gradient computation, use:
    from mfg_pde.operators import PartialDerivOperator
    grad_x = PartialDerivOperator(direction=0, spacings=[dx], field_shape=(N,), scheme="weno5")

For direct WENO functions:
    from mfg_pde.operators.reconstruction import compute_weno5_derivative_1d

Created: 2026-01-18 (Issue #605 Phase 2.1)
Refactored: 2026-01-18 (Issue #606 - WENO5 Operator Integration)
Updated: 2026-01-25 (Removed create_gradient_operators, use PartialDerivOperator)
"""

__all__: list[str] = []
