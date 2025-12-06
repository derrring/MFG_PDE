"""
Base types for BC stencil library.

This module defines the core dataclasses and enums used by stencil implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class OperatorType(Enum):
    """Types of differential operators for which stencils are provided."""

    LAPLACIAN = auto()  # Diffusion: D * nabla^2 u
    GRADIENT = auto()  # First derivative: nabla u
    ADVECTION = auto()  # Transport: v * nabla u
    DIVERGENCE = auto()  # div(v * u)
    FOKKER_PLANCK = auto()  # Combined: -div(alpha*m) + sigma^2/2 * nabla^2 m


@dataclass(frozen=True)
class BoundaryStencil:
    """
    Stencil coefficients for a boundary point in FDM matrix assembly.

    This dataclass encapsulates all information needed to modify matrix entries
    at boundary points, supporting various BC types and operators.

    Attributes
    ----------
    diagonal : float
        Coefficient to add to the diagonal entry A[i,i].

    neighbor : float
        Coefficient for the interior neighbor point. For left boundary this is
        the point to the right; for right boundary, the point to the left.

    neighbor_offset : int
        Index offset from boundary to neighbor point. Default is 1 for left
        boundary (neighbor at i+1) and -1 for right boundary (neighbor at i-1).
        This is set automatically by the factory methods.

    far_neighbor : float | None
        Coefficient for a second interior neighbor (for higher-order stencils).
        None if not used.

    far_neighbor_offset : int | None
        Index offset for far neighbor. None if not used.

    rhs_value : float
        Value to add to right-hand side vector b[i]. Used for non-homogeneous
        BCs (Dirichlet u=g, Neumann du/dn=g).

    rhs_function : Callable[[float], float] | None
        Function to compute RHS value from boundary position/time. Alternative
        to rhs_value for position-dependent BCs.

    eliminates_dof : bool
        If True, this BC eliminates the boundary DOF from the system (strong
        Dirichlet). The diagonal becomes 1 and neighbor becomes 0.

    preserves_conservation : bool
        If True, the stencil row sums to 0, preserving mass conservation for
        Fokker-Planck equations. This is critical for density evolution.

    bc_type : str
        Name of the BC type (e.g., "neumann", "dirichlet", "robin").

    operator_type : OperatorType
        The differential operator this stencil is for.

    Notes
    -----
    For mass-conserving FP solvers, always use stencils with
    preserves_conservation=True for no-flux boundaries.

    The row sum property: For conservation, we need sum of row = 0.
    - Interior: -2D/dx^2 + D/dx^2 + D/dx^2 = 0 (standard 3-point Laplacian)
    - Neumann boundary: D/dx^2 + (-D/dx^2) = 0 (ghost point reflection)

    Examples
    --------
    Using stencil in matrix assembly::

        stencil = FDMBoundaryStencils.diffusion_laplacian(
            bc_type=BCType.NEUMANN,
            position="left",
            dx=0.1,
            diffusion_coeff=0.05,
        )

        # Apply to matrix
        A[i, i] += stencil.diagonal
        A[i, i + stencil.neighbor_offset] = stencil.neighbor
        b[i] += stencil.rhs_value
    """

    diagonal: float
    neighbor: float
    neighbor_offset: int = 1
    far_neighbor: float | None = None
    far_neighbor_offset: int | None = None
    rhs_value: float = 0.0
    rhs_function: Callable[[float], float] | None = field(default=None, compare=False)
    eliminates_dof: bool = False
    preserves_conservation: bool = False
    bc_type: str = "unknown"
    operator_type: OperatorType = OperatorType.LAPLACIAN

    def row_sum(self) -> float:
        """
        Compute the row sum of stencil coefficients.

        For mass-conserving operators, this should be 0 (or very close).

        Returns
        -------
        float
            Sum of diagonal + neighbor + far_neighbor (if present).
        """
        total = self.diagonal + self.neighbor
        if self.far_neighbor is not None:
            total += self.far_neighbor
        return total

    def is_conservative(self, tol: float = 1e-12) -> bool:
        """
        Check if stencil is mass-conserving (row sum = 0).

        Parameters
        ----------
        tol : float
            Tolerance for checking row sum = 0.

        Returns
        -------
        bool
            True if |row_sum| < tol.
        """
        return abs(self.row_sum()) < tol
