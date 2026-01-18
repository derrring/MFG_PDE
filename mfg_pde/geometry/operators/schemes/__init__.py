"""
Specialized gradient schemes for high-order accuracy.

This module contains advanced numerical differentiation schemes that can be
used with the GradientOperator framework.

Available schemes:
- weno5: 5th-order WENO reconstruction (high-order, shock-capturing)

Created: 2026-01-18 (Issue #606 - WENO5 Operator Refactoring)
"""

from mfg_pde.geometry.operators.schemes.weno5 import compute_weno5_derivative_1d

__all__ = ["compute_weno5_derivative_1d"]
