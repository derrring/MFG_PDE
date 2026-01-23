"""
Interpolation and Projection Operators for MFG_PDE.

This module provides:
    - InterpolationOperator: Grid-to-particle interpolation
    - GeometryProjector: Solution projection between geometries
    - ProjectionRegistry: Registry for projection methods

Usage:
    >>> from mfg_pde.operators.interpolation import InterpolationOperator
    >>> interp = InterpolationOperator(grid_points, query_points)
    >>> u_at_particles = interp(u_grid)
"""

from mfg_pde.operators.interpolation.interpolation import InterpolationOperator
from mfg_pde.operators.interpolation.projection import (
    GeometryProjector,
    ProjectionRegistry,
)

__all__ = [
    "InterpolationOperator",
    "GeometryProjector",
    "ProjectionRegistry",
]
