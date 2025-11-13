"""
TensorProductGrid Re-export (Backward Compatibility).

This module re-exports TensorProductGrid from the subdirectory structure.
The canonical implementation is in mfg_pde.geometry.grids.tensor_grid.

This file will be removed in a future version once all imports are updated.
"""

from .grids.tensor_grid import TensorProductGrid

__all__ = ["TensorProductGrid"]
