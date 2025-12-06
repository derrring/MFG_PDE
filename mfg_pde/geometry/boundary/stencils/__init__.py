"""
BC Stencil Library - Reusable stencil coefficients for matrix construction.

This module provides boundary stencil coefficients that solvers can use when
assembling matrices. It separates BC logic from solver implementation, ensuring
consistent and correct boundary treatment across different solvers.

Architecture
============

The stencil library maps (BC type, operator type) -> coefficients:

- **BoundaryStencil**: Dataclass containing stencil coefficients and metadata
- **FDMBoundaryStencils**: FDM-specific stencils for Laplacian, advection operators
- **FEMBoundaryWeakForms**: FEM weak form contributions (planned)
- **MeshfreeBoundaryConstraints**: Meshfree collocation constraints (planned)

Key Design Principles
=====================

1. **BC module stays lightweight**: Only specifies mathematical intent
2. **Stencil library is reusable**: Same library serves HJB, FP, Poisson solvers
3. **Solvers remain flexible**: Can override for special cases
4. **Conservation properties**: Stencils annotate whether they preserve mass/energy

Usage
=====

FDM diffusion stencil at boundary::

    from mfg_pde.geometry.boundary.stencils import FDMBoundaryStencils, BoundaryStencil

    stencil: BoundaryStencil = FDMBoundaryStencils.diffusion_laplacian(
        bc_type=BCType.NEUMANN,
        position="left",
        dx=0.1,
        diffusion_coeff=0.05,  # sigma^2/2 for Fokker-Planck
    )

    # Use in matrix assembly
    diagonal_value += stencil.diagonal
    data_values.append(stencil.neighbor)

    # Check conservation property
    if stencil.preserves_conservation:
        # Row sum should be 0 for mass conservation

See Also
--------
- GitHub Issue #379: Layered BC Stencil Architecture for Matrix Construction
- mfg_pde.geometry.boundary: BC specification and application modules
"""

from __future__ import annotations

from .base import BoundaryStencil, OperatorType
from .fdm_stencils import FDMBoundaryStencils

__all__ = [
    "BoundaryStencil",
    "OperatorType",
    "FDMBoundaryStencils",
]
