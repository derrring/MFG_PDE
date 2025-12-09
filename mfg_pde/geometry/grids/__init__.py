"""
Cartesian grid geometries for MFG problems.

This module provides structured tensor product grids for finite difference methods,
supporting arbitrary-dimensional regular grids through TensorProductGrid.
"""

from .tensor_grid import TensorProductGrid

__all__ = [
    "TensorProductGrid",
]
