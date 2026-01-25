"""
Boundary utilities for MFG_PDE.

This subpackage provides boundary-related utility functions for position
reflection and wrapping at domain boundaries.

Modules:
    boundary_reflection: Position reflection/wrapping at domain boundaries (Issue #521)

Corner Handling:
    At corners, all dimensions are processed simultaneously, producing
    diagonal reflection. This is equivalent to 'average' corner strategy.
"""

from .boundary_reflection import (
    absorb_positions,
    reflect_positions,
    wrap_positions,
)

__all__ = [
    "reflect_positions",
    "wrap_positions",
    "absorb_positions",
]
