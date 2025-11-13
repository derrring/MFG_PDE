"""
2D/3D Cartesian Grid Re-export (Backward Compatibility).

This module re-exports SimpleGrid2D and SimpleGrid3D from the subdirectory structure.
The canonical implementation is in mfg_pde.geometry.grids.grid_2d.

This file will be removed in a future version once all imports are updated.
"""

from .grids.grid_2d import SimpleGrid2D, SimpleGrid3D

__all__ = ["SimpleGrid2D", "SimpleGrid3D"]
