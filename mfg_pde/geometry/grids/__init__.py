"""
Cartesian grid geometries for MFG problems.

This module provides structured tensor product grids for finite difference methods,
supporting 1D, 2D, and arbitrary-dimensional regular grids.

Note: SimpleGrid1D/2D/3D have been removed. Use TensorProductGrid instead.
"""

from .tensor_grid import TensorProductGrid

__all__ = [
    "TensorProductGrid",
]
