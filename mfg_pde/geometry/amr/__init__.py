"""
Adaptive Mesh Refinement (AMR) - API stub for future library integration.

This module provides a minimal API for AMR that will wrap external libraries
when implemented. The actual AMR functionality is not yet available.

Recommended external libraries for future integration:
- pyAMReX: Block-structured AMR, GPU support (https://github.com/AMReX-Codes/pyamrex)
- Clawpack/AMRClaw: Hyperbolic PDEs, Berger-Oliger-Colella AMR
- pyAMG: Mesh adaptation for complex 2D/3D geometries (Inria)
- p4est: Scalable octree AMR

Status: NOT IMPLEMENTED - This is a placeholder for future development.
"""

from __future__ import annotations

# Re-export protocol definitions from protocol.py
from mfg_pde.geometry.protocol import AdaptiveGeometry, is_adaptive


class AMRNotImplementedError(NotImplementedError):
    """Raised when AMR functionality is called but not yet implemented."""

    def __init__(self, backend: str = ""):
        msg = "AMR is not yet implemented. "
        if backend:
            msg += f"Consider using {backend} directly. "
        msg += "See mfg_pde.geometry.amr module docstring for recommended libraries."
        super().__init__(msg)


def create_amr_grid(*args, **kwargs):
    """
    Factory function for AMR grids - NOT YET IMPLEMENTED.

    Raises:
        AMRNotImplementedError: Always, as AMR is not yet implemented.
    """
    raise AMRNotImplementedError("pyAMReX")


__all__ = [
    "AdaptiveGeometry",
    "AMRNotImplementedError",
    "create_amr_grid",
    "is_adaptive",
]
