"""
Geometric operators for MFG problems.

This module provides operators for projecting solutions between different
geometry discretizations, enabling hybrid solvers and multi-resolution methods.
"""

from .projection import GeometryProjector, ProjectionRegistry

__all__ = [
    "GeometryProjector",
    "ProjectionRegistry",
]
