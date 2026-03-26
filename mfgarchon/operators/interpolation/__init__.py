"""
Interpolation and Projection Operators for MFGarchon.

This module provides:
    - InterpolationOperator: Grid-to-particle interpolation
    - GeometryProjector: Solution projection between geometries
    - ProjectionRegistry: Registry for projection methods

Usage:
    >>> from mfgarchon.operators.interpolation import InterpolationOperator
    >>> interp = InterpolationOperator(grid_points, query_points)
    >>> u_at_particles = interp(u_grid)
"""

from mfgarchon.operators.interpolation.interpolation import InterpolationOperator
from mfgarchon.operators.interpolation.projection import (
    GeometryProjector,
    ProjectionRegistry,
)

__all__ = [
    "InterpolationOperator",
    "GeometryProjector",
    "ProjectionRegistry",
]
