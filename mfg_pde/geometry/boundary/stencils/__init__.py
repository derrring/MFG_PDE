"""
BC Stencil Library - Generic BC transforms for matrix construction.

This module provides generic boundary condition transformations that solvers
can use when assembling matrices. It separates BC logic from solver
implementation, ensuring consistent boundary treatment across different solvers.

Architecture
============

**Key Principle**: Operators belong to solvers, NOT to BC module.

The BC module provides GENERIC TRANSFORMS that can modify ANY operator stencil
at boundaries, avoiding combinatorial explosion of N_operators x N_bcs x N_methods.

- **BCTransforms**: Generic BC transformations (neumann, dirichlet, robin, no_flux)
- **BoundaryStencil**: Dataclass containing stencil coefficients and metadata

Complexity: O(N_bc_types), NOT O(operators x bcs)

Usage
=====

::

    from mfg_pde.geometry.boundary.stencils import BCTransforms

    # Solver builds its interior stencil (solver's responsibility)
    D = sigma**2 / 2
    interior = {"diagonal": -2*D/dx**2, "left": D/dx**2, "right": D/dx**2}

    # Apply Neumann BC transform (BC module's responsibility)
    boundary_stencil = BCTransforms.neumann(interior, "left", dx, bc_value=0.0)

    # Use in matrix assembly
    A[0, 0] = boundary_stencil.diagonal
    A[0, 1] = boundary_stencil.neighbor

See Also
--------
- GitHub Issue #379: Layered BC Stencil Architecture for Matrix Construction
- mfg_pde.geometry.boundary: BC specification and application modules
"""

from __future__ import annotations

from .base import BoundaryStencil, OperatorType
from .transforms import BCTransforms, InteriorStencil

__all__ = [
    # Generic transforms
    "BCTransforms",
    "InteriorStencil",
    # Base types
    "BoundaryStencil",
    "OperatorType",
]
