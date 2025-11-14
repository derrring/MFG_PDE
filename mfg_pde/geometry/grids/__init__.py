"""
Cartesian grid geometries for MFG problems.

This module provides structured tensor product grids for finite difference methods,
supporting 1D, 2D, and arbitrary-dimensional regular grids.
"""

from .grid_1d import SimpleGrid1D
from .grid_2d import SimpleGrid2D
from .grid_3d import SimpleGrid3D
from .tensor_grid import TensorProductGrid

__all__ = [
    "SimpleGrid1D",
    "SimpleGrid2D",
    "SimpleGrid3D",
    "TensorProductGrid",
]
