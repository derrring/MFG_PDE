"""
Boundary Condition Validation Tools.

This module provides tools for validating boundary condition discretizations,
including:

- GKS (Gustafsson-Kreiss-Sundström) stability analysis
- Future: L-S (Lopatinskii-Shapiro) well-posedness analysis (Issue #535)

These are **developer tools** for validating BC implementations, not user-facing
runtime checks.

Created: 2026-01-18 (Issue #593 Phase 4.2)
"""

from mfg_pde.geometry.boundary.validation.gks import (
    GKSResult,
    check_gks_stability,
)

__all__ = [
    "GKSResult",
    "check_gks_stability",
]
