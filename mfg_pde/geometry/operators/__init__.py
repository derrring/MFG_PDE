"""
Geometric operators for MFG problems.

This module provides:
1. Projection operators for transferring solutions between geometries
2. Differential operators (Laplacian, gradient) as scipy LinearOperator classes
3. Temporary callable wrappers for operators not yet refactored (Issue #595)
4. Operator trait implementations for geometry-agnostic solver design
"""

from .gradient import GradientComponentOperator, create_gradient_operators
from .laplacian import LaplacianOperator
from .projection import GeometryProjector, ProjectionRegistry
from .wrappers import (
    create_advection_operator,
    create_divergence_operator,
    create_interpolation_operator,
)

__all__ = [
    "GeometryProjector",
    "ProjectionRegistry",
    "LaplacianOperator",
    "GradientComponentOperator",
    "create_gradient_operators",
    "create_divergence_operator",
    "create_advection_operator",
    "create_interpolation_operator",
]
