"""
WENO (Weighted Essentially Non-Oscillatory) Numerical Methods.

Shared high-order spatial discretization utilities for Hamilton-Jacobi equations:
- WENO5 gradient computation for level set evolution
- WENO reconstruction for HJB solvers
- ENO/WENO variants

Created: 2026-01-18 (Issue #605 Phase 2.1)
"""

from mfg_pde.alg.numerical.weno.weno5_gradients import WENO5Gradient

__all__ = ["WENO5Gradient"]
