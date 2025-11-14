"""
1D Cartesian Grid Re-export (Backward Compatibility).

This module re-exports SimpleGrid1D from the subdirectory structure.
The canonical implementation is in mfg_pde.geometry.grids.grid_1d.

This file will be removed in a future version once all imports are updated.
"""

from .boundary_conditions_1d import BoundaryConditions
from .grids.grid_1d import SimpleGrid1D

__all__ = ["SimpleGrid1D", "BoundaryConditions"]
