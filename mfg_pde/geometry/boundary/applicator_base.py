"""
Abstract base classes and protocols for boundary condition applicators.

This module defines the interface for BC application across different
discretization methods (FDM, FEM, Meshfree, Graph) and dimensions (1D, 2D, 3D, nD).

The key separation is:
- **BC Specification** (conditions.py): What BC to apply (dimension-agnostic)
- **BC Application** (applicator_*.py): How to apply it (discretization-dependent)

Design Principles:
1. Applicators accept unified BoundaryConditions from conditions.py
2. Each discretization type has its own applicator with specialized interface
3. Base classes provide common properties and validation
4. Higher dimensions fall back to generic nD implementation where applicable

Hierarchy:
    BaseBCApplicator (ABC)
    ├── BaseStructuredApplicator (FDM - ghost cells on structured grids)
    ├── BaseUnstructuredApplicator (FEM - matrix modification on meshes)
    ├── BaseMeshfreeApplicator (particles and collocation points)
    └── BaseGraphApplicator (graphs, networks, mazes)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .conditions import BoundaryConditions


class DiscretizationType(Enum):
    """Discretization method for PDE solvers."""

    FDM = auto()  # Finite Difference Method (structured grids)
    FEM = auto()  # Finite Element Method (unstructured meshes)
    FVM = auto()  # Finite Volume Method
    GFDM = auto()  # Generalized Finite Difference Method
    DGM = auto()  # Discontinuous Galerkin Method
    MESHFREE = auto()  # Meshfree/particle methods
    GRAPH = auto()  # Graph-based discretization


class GridType(Enum):
    """Grid/mesh type affecting BC application."""

    CELL_CENTERED = auto()  # Values at cell centers (boundary at faces)
    VERTEX_CENTERED = auto()  # Values at vertices (boundary at vertices)
    STAGGERED = auto()  # Staggered grid (MAC scheme)


@runtime_checkable
class BCApplicatorProtocol(Protocol):
    """Protocol for boundary condition applicators."""

    @property
    def dimension(self) -> int:
        """Spatial dimension this applicator handles."""
        ...

    @property
    def discretization_type(self) -> DiscretizationType:
        """Discretization method this applicator is designed for."""
        ...

    def supports_bc_type(self, bc: BoundaryConditions) -> bool:
        """Check if this applicator supports the given BC specification."""
        ...


class BaseBCApplicator(ABC):
    """
    Abstract base class for all BC applicators.

    Provides common infrastructure:
    - Dimension and discretization type properties
    - BC validation helpers
    - Dimension dispatch logic (optional)

    Subclasses implement discretization-specific application methods.
    """

    def __init__(self, dimension: int | None = None):
        """
        Initialize applicator.

        Args:
            dimension: Spatial dimension (1, 2, 3, or higher). None for dimension-agnostic.
        """
        self._dimension = dimension

    @property
    def dimension(self) -> int | None:
        """Spatial dimension (None if dimension-agnostic)."""
        return self._dimension

    @property
    @abstractmethod
    def discretization_type(self) -> DiscretizationType:
        """Discretization method this applicator handles."""
        ...

    def supports_bc_type(self, bc: BoundaryConditions) -> bool:
        """
        Check if this applicator supports the given BC.

        Default implementation checks dimension match.
        Subclasses can add additional checks.
        """
        if self._dimension is None:
            return True  # Dimension-agnostic
        return bc.dimension == self._dimension

    def _validate_bc(self, bc: BoundaryConditions) -> None:
        """Validate BC specification against this applicator."""
        if self._dimension is not None and bc.dimension != self._dimension:
            raise ValueError(f"BC dimension ({bc.dimension}) does not match applicator dimension ({self._dimension})")

    def __repr__(self) -> str:
        dim_str = f"dimension={self._dimension}" if self._dimension else "dimension-agnostic"
        return f"{self.__class__.__name__}({dim_str}, {self.discretization_type.name})"


# =============================================================================
# Structured Grid (FDM) Base Class
# =============================================================================


class BaseStructuredApplicator(BaseBCApplicator):
    """
    Base class for structured grid (FDM) applicators.

    Uses ghost cell method for BC application on regular grids.
    Supports dimension dispatch for optimized 1D/2D/3D implementations.
    """

    def __init__(self, dimension: int, grid_type: GridType = GridType.CELL_CENTERED):
        """
        Initialize structured grid applicator.

        Args:
            dimension: Spatial dimension (1, 2, 3, or higher)
            grid_type: Grid type (cell-centered, vertex-centered, staggered)
        """
        super().__init__(dimension)
        self._grid_type = grid_type

    @property
    def discretization_type(self) -> DiscretizationType:
        return DiscretizationType.FDM

    @property
    def grid_type(self) -> GridType:
        """Grid type."""
        return self._grid_type


# =============================================================================
# Unstructured Mesh (FEM) Base Class
# =============================================================================


class BaseUnstructuredApplicator(BaseBCApplicator):
    """
    Base class for unstructured mesh (FEM) applicators.

    Applies BCs by modifying system matrices and RHS vectors.
    """

    def __init__(self, dimension: int):
        """
        Initialize FEM applicator.

        Args:
            dimension: Spatial dimension (1, 2, or 3)
        """
        if dimension not in (1, 2, 3):
            raise ValueError(f"FEM applicator supports 1D, 2D, and 3D, got {dimension}D")
        super().__init__(dimension)

    @property
    def discretization_type(self) -> DiscretizationType:
        return DiscretizationType.FEM


# =============================================================================
# Meshfree Base Class
# =============================================================================


class BaseMeshfreeApplicator(BaseBCApplicator):
    """
    Base class for meshfree (particle/collocation) applicators.

    Handles boundary conditions for:
    - Particle methods (Lagrangian FP solver)
    - Collocation methods (RBF, GFDM)
    - High-dimensional implicit domains
    """

    def __init__(self, dimension: int | None = None):
        """
        Initialize meshfree applicator.

        Args:
            dimension: Spatial dimension (None for dimension from geometry)
        """
        super().__init__(dimension)

    @property
    def discretization_type(self) -> DiscretizationType:
        return DiscretizationType.MESHFREE


# =============================================================================
# Graph Base Class
# =============================================================================


class BaseGraphApplicator(BaseBCApplicator):
    """
    Base class for graph-based (network/maze) applicators.

    Handles boundary conditions on discrete graph structures:
    - Network MFG (GridNetwork, RandomNetwork, ScaleFreeNetwork)
    - Maze MFG (MazeGeometry, VoronoiMaze)
    - General graph domains
    """

    def __init__(self, num_nodes: int):
        """
        Initialize graph applicator.

        Args:
            num_nodes: Number of nodes in the graph
        """
        super().__init__(dimension=None)  # Graphs are dimension-agnostic
        self._num_nodes = num_nodes

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self._num_nodes

    @property
    def discretization_type(self) -> DiscretizationType:
        return DiscretizationType.GRAPH

    def supports_bc_type(self, bc: BoundaryConditions) -> bool:
        """Graph applicators are dimension-agnostic."""
        return True


# =============================================================================
# Ghost Cell Formula Helpers (used by FDM applicators)
# =============================================================================


def ghost_cell_dirichlet(
    interior_value: float,
    boundary_value: float,
    grid_type: GridType = GridType.CELL_CENTERED,
) -> float:
    """
    Compute ghost cell value for Dirichlet BC.

    For cell-centered grids (boundary at cell face):
        u_boundary = (u_ghost + u_interior) / 2 = g
        => u_ghost = 2*g - u_interior

    For vertex-centered grids (boundary at vertex):
        u_ghost = g (direct assignment)
    """
    if grid_type == GridType.VERTEX_CENTERED:
        return boundary_value
    else:
        return 2.0 * boundary_value - interior_value


def ghost_cell_neumann(
    interior_value: float,
    flux_value: float,
    dx: float,
    outward_normal_sign: float = 1.0,
    grid_type: GridType = GridType.CELL_CENTERED,
) -> float:
    """
    Compute ghost cell value for Neumann BC.

    For cell-centered grids:
        du/dn = (u_ghost - u_interior) / (2*dx) * sign = g
        => u_ghost = u_interior + 2*dx*g*sign

    Args:
        interior_value: Value at interior point
        flux_value: Prescribed flux (du/dn)
        dx: Grid spacing
        outward_normal_sign: +1 for max boundary, -1 for min boundary
        grid_type: Grid type
    """
    if grid_type == GridType.VERTEX_CENTERED:
        return interior_value + dx * flux_value * outward_normal_sign
    else:
        return interior_value + 2.0 * dx * flux_value * outward_normal_sign


def ghost_cell_robin(
    interior_value: float,
    rhs_value: float,
    alpha: float,
    beta: float,
    dx: float,
    outward_normal_sign: float = 1.0,
    grid_type: GridType = GridType.CELL_CENTERED,
) -> float:
    """
    Compute ghost cell value for Robin BC: alpha*u + beta*du/dn = g.

    For cell-centered grids:
        alpha * (u_ghost + u_interior)/2 + beta * (u_ghost - u_interior)/(2*dx) * sign = g

    Solving for u_ghost:
        u_ghost * (alpha/2 + beta*sign/(2*dx)) = g - u_interior * (alpha/2 - beta*sign/(2*dx))
    """
    if grid_type == GridType.VERTEX_CENTERED:
        # Simplified for vertex-centered
        if abs(alpha) > 1e-12:
            return (rhs_value - beta * outward_normal_sign * interior_value / dx) / alpha
        else:
            return interior_value + dx * rhs_value / beta * outward_normal_sign

    # Cell-centered
    coeff_ghost = alpha / 2.0 + beta * outward_normal_sign / (2.0 * dx)
    coeff_interior = alpha / 2.0 - beta * outward_normal_sign / (2.0 * dx)

    if abs(coeff_ghost) < 1e-12:
        raise ValueError("Robin BC coefficients lead to singular ghost cell formula")

    return (rhs_value - interior_value * coeff_interior) / coeff_ghost


__all__ = [
    # Enums
    "DiscretizationType",
    "GridType",
    # Protocol
    "BCApplicatorProtocol",
    # Base classes
    "BaseBCApplicator",
    "BaseStructuredApplicator",
    "BaseUnstructuredApplicator",
    "BaseMeshfreeApplicator",
    "BaseGraphApplicator",
    # Ghost cell helpers
    "ghost_cell_dirichlet",
    "ghost_cell_neumann",
    "ghost_cell_robin",
]
