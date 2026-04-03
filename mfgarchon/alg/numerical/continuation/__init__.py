"""
Continuation methods for tracking MFG equilibrium branches.

- HomotopyContinuation: Predictor-corrector for equilibrium branches
- BifurcationPoint: Detected bifurcation metadata

Issue #926: Part of Phase 3 (Generalized PDE & Institutional MFG Plan).
"""

from .homotopy import BifurcationPoint, ContinuationResult, HomotopyContinuation

__all__ = [
    "HomotopyContinuation",
    "ContinuationResult",
    "BifurcationPoint",
]
