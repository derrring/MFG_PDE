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


# =============================================================================
# Topology/Calculator Composition (Semantic Dispatch Pattern)
# =============================================================================
# This is the structural foundation for the 2-layer BC architecture:
#   Layer 1: Topology (Memory/Indexing) - how boundaries connect
#   Layer 2: Calculator (Physics Strategy) - what values to use
#
# See Issue #516 and docs/development/bc_architecture_analysis.md
# =============================================================================


@runtime_checkable
class Topology(Protocol):
    """
    Protocol for boundary topology (Layer 1 of 2-layer BC architecture).

    Topology handles the structural/memory aspects of boundary conditions:
    - Periodic: boundaries wrap around (u[-1] = u[n-1])
    - Bounded: boundaries are physical edges requiring ghost values

    This separation allows the same Calculator (physics) to work with
    different grid connectivities.

    Example:
        >>> # Periodic domain - topology handles wrap-around
        >>> topology = PeriodicTopology(dimension=2, shape=(100, 100))
        >>> if topology.is_periodic:
        ...     # Use np.pad(..., mode='wrap')
        ...     pass

        >>> # Bounded domain - need Calculator for ghost values
        >>> topology = BoundedTopology(dimension=2, shape=(100, 100))
        >>> if not topology.is_periodic:
        ...     # Apply calculator.compute() for each boundary
        ...     pass
    """

    @property
    def is_periodic(self) -> bool:
        """Whether this topology has periodic boundaries."""
        ...

    @property
    def dimension(self) -> int:
        """Spatial dimension."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Grid shape (interior points only)."""
        ...


@runtime_checkable
class BoundaryCalculator(Protocol):
    """
    Protocol for ghost value computation (Layer 2 of 2-layer BC architecture).

    Calculator handles the physics/value aspects of boundary conditions:
    - Dirichlet: u_ghost = 2*g - u_interior
    - Neumann: u_ghost = u_interior + 2*dx*g
    - Robin: combination of above
    - Extrapolation: polynomial continuation

    The Calculator is ONLY used for bounded topologies. Periodic topologies
    compute ghost values directly from wrap-around indices.

    Example:
        >>> # Dirichlet BC with fixed boundary value
        >>> calculator = DirichletCalculator(boundary_value=0.0)
        >>> u_ghost = calculator.compute(interior_value=1.0, dx=0.1, side='min')
        # Returns 2*0 - 1 = -1.0

        >>> # Neumann BC with zero flux
        >>> calculator = NeumannCalculator(flux_value=0.0)
        >>> u_ghost = calculator.compute(interior_value=1.0, dx=0.1, side='min')
        # Returns 1.0 (edge extension)
    """

    def compute(
        self,
        interior_value: float,
        dx: float,
        side: str,
        **kwargs,
    ) -> float:
        """
        Compute ghost cell value from interior value.

        Args:
            interior_value: Value at interior point adjacent to boundary
            dx: Grid spacing
            side: Boundary side ('min' or 'max')
            **kwargs: Additional parameters (e.g., time for time-varying BCs)

        Returns:
            Ghost cell value
        """
        ...


# =============================================================================
# Concrete Topology Implementations
# =============================================================================


class PeriodicTopology:
    """
    Periodic boundary topology.

    In periodic topology, boundaries wrap around: the ghost cell at the
    low boundary equals the interior value at the high boundary, and vice versa.

    This is a MEMORY/INDEXING concept, not a physics concept. The Calculator
    is NOT used for periodic boundaries - values come from wrap-around.
    """

    def __init__(self, dimension: int, shape: tuple[int, ...]):
        """
        Initialize periodic topology.

        Args:
            dimension: Spatial dimension (1, 2, 3, ...)
            shape: Grid shape (interior points)
        """
        if len(shape) != dimension:
            raise ValueError(f"Shape length {len(shape)} must match dimension {dimension}")
        self._dimension = dimension
        self._shape = shape

    @property
    def is_periodic(self) -> bool:
        return True

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __repr__(self) -> str:
        return f"PeriodicTopology(dimension={self._dimension}, shape={self._shape})"


class BoundedTopology:
    """
    Bounded (non-periodic) boundary topology.

    In bounded topology, boundaries are physical edges that require ghost
    values computed by a Calculator. The topology itself just marks that
    boundaries exist - the Calculator provides the values.

    This separation enables:
    - Same Calculator works with any bounded grid
    - Different Calculators can be swapped without changing topology
    """

    def __init__(self, dimension: int, shape: tuple[int, ...]):
        """
        Initialize bounded topology.

        Args:
            dimension: Spatial dimension (1, 2, 3, ...)
            shape: Grid shape (interior points)
        """
        if len(shape) != dimension:
            raise ValueError(f"Shape length {len(shape)} must match dimension {dimension}")
        self._dimension = dimension
        self._shape = shape

    @property
    def is_periodic(self) -> bool:
        return False

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __repr__(self) -> str:
        return f"BoundedTopology(dimension={self._dimension}, shape={self._shape})"


# =============================================================================
# Concrete Calculator Implementations
# =============================================================================


class DirichletCalculator:
    """
    Calculator for Dirichlet (fixed value) boundary conditions.

    Computes ghost cell value such that the boundary value equals the
    prescribed value g:
        u_boundary = (u_ghost + u_interior) / 2 = g  (cell-centered)
        => u_ghost = 2*g - u_interior
    """

    def __init__(
        self,
        boundary_value: float = 0.0,
        grid_type: GridType = GridType.CELL_CENTERED,
    ):
        """
        Initialize Dirichlet calculator.

        Args:
            boundary_value: Prescribed value at boundary
            grid_type: Grid type (cell-centered or vertex-centered)
        """
        self._boundary_value = boundary_value
        self._grid_type = grid_type

    @property
    def boundary_value(self) -> float:
        return self._boundary_value

    @property
    def grid_type(self) -> GridType:
        return self._grid_type

    def compute(
        self,
        interior_value: float,
        dx: float,
        side: str,
        **kwargs,
    ) -> float:
        """Compute ghost value for Dirichlet BC."""
        return ghost_cell_dirichlet(interior_value, self._boundary_value, self._grid_type)

    def __repr__(self) -> str:
        return f"DirichletCalculator(boundary_value={self._boundary_value})"


class NeumannCalculator:
    """
    Calculator for Neumann (fixed flux) boundary conditions.

    Computes ghost cell value such that the normal derivative equals
    the prescribed flux g:
        du/dn = (u_ghost - u_interior) / (2*dx) = g  (cell-centered)
        => u_ghost = u_interior + 2*dx*g
    """

    def __init__(
        self,
        flux_value: float = 0.0,
        grid_type: GridType = GridType.CELL_CENTERED,
    ):
        """
        Initialize Neumann calculator.

        Args:
            flux_value: Prescribed normal flux (du/dn)
            grid_type: Grid type
        """
        self._flux_value = flux_value
        self._grid_type = grid_type

    @property
    def flux_value(self) -> float:
        return self._flux_value

    @property
    def grid_type(self) -> GridType:
        return self._grid_type

    def compute(
        self,
        interior_value: float,
        dx: float,
        side: str,
        **kwargs,
    ) -> float:
        """Compute ghost value for Neumann BC."""
        # Outward normal sign: +1 for max boundary, -1 for min boundary
        outward_sign = 1.0 if side == "max" else -1.0
        return ghost_cell_neumann(interior_value, self._flux_value, dx, outward_sign, self._grid_type)

    def __repr__(self) -> str:
        return f"NeumannCalculator(flux_value={self._flux_value})"


class RobinCalculator:
    """
    Calculator for Robin (mixed) boundary conditions.

    Computes ghost cell value for the Robin condition:
        alpha*u + beta*du/dn = g
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        rhs_value: float = 0.0,
        grid_type: GridType = GridType.CELL_CENTERED,
    ):
        """
        Initialize Robin calculator.

        Args:
            alpha: Coefficient on u (Dirichlet weight)
            beta: Coefficient on du/dn (Neumann weight)
            rhs_value: Right-hand side value g
            grid_type: Grid type
        """
        self._alpha = alpha
        self._beta = beta
        self._rhs_value = rhs_value
        self._grid_type = grid_type

    def compute(
        self,
        interior_value: float,
        dx: float,
        side: str,
        **kwargs,
    ) -> float:
        """Compute ghost value for Robin BC."""
        outward_sign = 1.0 if side == "max" else -1.0
        return ghost_cell_robin(
            interior_value,
            self._rhs_value,
            self._alpha,
            self._beta,
            dx,
            outward_sign,
            self._grid_type,
        )

    def __repr__(self) -> str:
        return f"RobinCalculator(alpha={self._alpha}, beta={self._beta}, rhs={self._rhs_value})"


class NoFluxCalculator:
    """
    Calculator for no-flux (zero Neumann) boundary conditions.

    This is a special case of Neumann with flux = 0, but named explicitly
    for clarity. Ghost cell equals interior value (edge extension).
    """

    def __init__(self, grid_type: GridType = GridType.CELL_CENTERED):
        self._grid_type = grid_type

    def compute(
        self,
        interior_value: float,
        dx: float,
        side: str,
        **kwargs,
    ) -> float:
        """Compute ghost value for no-flux BC (edge extension)."""
        return interior_value

    def __repr__(self) -> str:
        return "NoFluxCalculator()"


class LinearExtrapolationCalculator:
    """
    Calculator for linear extrapolation boundary conditions.

    Uses zero second derivative (d²u/dx² = 0) at boundary.
    Ghost = 2*u_0 - u_1

    Requires access to two interior values.
    """

    def compute(
        self,
        interior_value: float,
        dx: float,
        side: str,
        second_interior_value: float | None = None,
        **kwargs,
    ) -> float:
        """
        Compute ghost value via linear extrapolation.

        Args:
            interior_value: Value at point adjacent to boundary (u_0)
            dx: Grid spacing (not used, but part of protocol)
            side: Boundary side (not used, but part of protocol)
            second_interior_value: Value at second interior point (u_1)

        Returns:
            Ghost value = 2*u_0 - u_1
        """
        if second_interior_value is None:
            # Fall back to edge extension if second value not provided
            return interior_value
        return ghost_cell_linear_extrapolation((interior_value, second_interior_value))

    def __repr__(self) -> str:
        return "LinearExtrapolationCalculator()"


class QuadraticExtrapolationCalculator:
    """
    Calculator for quadratic extrapolation boundary conditions.

    Uses zero third derivative (d³u/dx³ = 0) at boundary.
    Ghost = 3*u_0 - 3*u_1 + u_2

    Requires access to three interior values.
    """

    def compute(
        self,
        interior_value: float,
        dx: float,
        side: str,
        second_interior_value: float | None = None,
        third_interior_value: float | None = None,
        **kwargs,
    ) -> float:
        """
        Compute ghost value via quadratic extrapolation.

        Args:
            interior_value: Value at point adjacent to boundary (u_0)
            dx: Grid spacing (not used)
            side: Boundary side (not used)
            second_interior_value: Value at second interior point (u_1)
            third_interior_value: Value at third interior point (u_2)

        Returns:
            Ghost value = 3*u_0 - 3*u_1 + u_2
        """
        if second_interior_value is None or third_interior_value is None:
            # Fall back to linear if not enough points
            if second_interior_value is not None:
                return ghost_cell_linear_extrapolation((interior_value, second_interior_value))
            return interior_value
        return ghost_cell_quadratic_extrapolation((interior_value, second_interior_value, third_interior_value))

    def __repr__(self) -> str:
        return "QuadraticExtrapolationCalculator()"


class FPNoFluxCalculator:
    """
    Calculator for Fokker-Planck no-flux (zero total flux) boundary conditions.

    For advection-diffusion equations, no-flux means J·n = 0 where
    J = v*ρ - D*∇ρ. This requires a Robin-type formula that accounts
    for both advection and diffusion.

    This is more physically correct than pure Neumann for FP equations.
    """

    def __init__(
        self,
        drift_velocity: float = 0.0,
        diffusion_coeff: float = 1.0,
        grid_type: GridType = GridType.CELL_CENTERED,
    ):
        """
        Initialize FP no-flux calculator.

        Args:
            drift_velocity: Normal component of drift (positive = outward)
            diffusion_coeff: Diffusion coefficient D = σ²/2
            grid_type: Grid type
        """
        self._drift = drift_velocity
        self._diffusion = diffusion_coeff
        self._grid_type = grid_type

    def compute(
        self,
        interior_value: float,
        dx: float,
        side: str,
        drift_velocity: float | None = None,
        **kwargs,
    ) -> float:
        """
        Compute ghost value for FP no-flux BC.

        Args:
            interior_value: Density at interior point
            dx: Grid spacing
            side: Boundary side ('min' or 'max')
            drift_velocity: Override drift velocity (optional)
        """
        outward_sign = 1.0 if side == "max" else -1.0
        v = drift_velocity if drift_velocity is not None else self._drift
        return ghost_cell_fp_no_flux(
            interior_value,
            v,
            self._diffusion,
            dx,
            outward_sign,
            self._grid_type,
        )

    def __repr__(self) -> str:
        return f"FPNoFluxCalculator(drift={self._drift}, D={self._diffusion})"


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


# =============================================================================
# Physics-Aware Ghost Cell Formulas
# =============================================================================
# IMPORTANT LESSON: The discretized BC must match the physics, not just the
# mathematical form. For advection-diffusion equations (like Fokker-Planck),
# a "no-flux" BC means J·n = 0 where J = v*ρ - D*∇ρ.
#
# - Naive approach: Neumann (∂ρ/∂n = 0) only zeroes diffusion flux ❌
# - Correct approach: Robin BC that zeroes TOTAL flux ✅
#
# This distinction is crucial for mass conservation in FP equations.
# =============================================================================


def ghost_cell_fp_no_flux(
    interior_value: float,
    drift_velocity: float,
    diffusion_coeff: float,
    dx: float,
    outward_normal_sign: float = 1.0,
    grid_type: GridType = GridType.CELL_CENTERED,
) -> float:
    """
    Compute ghost cell value for Fokker-Planck no-flux (zero total flux) BC.

    IMPORTANT: For advection-diffusion equations like Fokker-Planck, a "no-flux"
    BC means the TOTAL flux J = v*ρ - D*∇ρ = 0, not just ∂ρ/∂n = 0.

    This requires a Robin-type ghost cell formula that accounts for both
    advection and diffusion contributions to the flux.

    Mathematical derivation:
        Total flux: J = v*ρ - D*∂ρ/∂x
        No-flux BC: J·n = 0 at boundary

        At boundary (cell face for cell-centered):
            v_n * ρ_face - D * (∂ρ/∂n)_face = 0

        Using cell-centered discretization:
            ρ_face ≈ (ρ_ghost + ρ_interior) / 2
            ∂ρ/∂n ≈ (ρ_ghost - ρ_interior) / dx

        Substituting and solving for ρ_ghost:
            v_n * (ρ_ghost + ρ_interior)/2 = D * (ρ_ghost - ρ_interior)/dx
            ρ_ghost = ρ_interior * (2D + v_n*dx) / (2D - v_n*dx)

        Physical interpretation:
            - When v_n > 0 (outflow): ρ_ghost > ρ_interior (diffusion opposes outflow)
            - When v_n < 0 (inflow): ρ_ghost < ρ_interior (diffusion opposes inflow)
            - When v_n = 0: ρ_ghost = ρ_interior (pure Neumann)

    Args:
        interior_value: Density at interior point ρ_interior
        drift_velocity: Normal component of drift velocity v·n (positive = outward)
        diffusion_coeff: Diffusion coefficient D = σ²/2
        dx: Grid spacing
        outward_normal_sign: +1 for max boundary (outward normal points positive),
                            -1 for min boundary (outward normal points negative)
        grid_type: Grid type (cell-centered or vertex-centered)

    Returns:
        Ghost cell value that ensures zero total flux at boundary

    Example:
        >>> # Left boundary with leftward drift (into boundary)
        >>> rho_ghost = ghost_cell_fp_no_flux(
        ...     interior_value=1.0,
        ...     drift_velocity=-0.5,  # v < 0, drift toward left boundary
        ...     diffusion_coeff=0.125,  # D = 0.5²/2
        ...     dx=0.1,
        ...     outward_normal_sign=-1.0  # Left boundary
        ... )

    References:
        - Achdou & Laurière (2020): Mean Field Games and Applications, Section on FP BCs
        - LeVeque (2002): Finite Volume Methods for Hyperbolic Problems
    """
    D = diffusion_coeff
    v_n = drift_velocity * outward_normal_sign  # Normal velocity (positive = outward)

    if grid_type == GridType.VERTEX_CENTERED:
        # Vertex-centered: boundary at grid point
        # ρ_ghost = ρ_interior * (D + v_n*dx) / (D - v_n*dx)
        numerator = D + v_n * dx
        denominator = D - v_n * dx
    else:
        # Cell-centered: boundary at cell face
        # ρ_ghost = ρ_interior * (2*D + v_n*dx) / (2*D - v_n*dx)
        numerator = 2.0 * D + v_n * dx
        denominator = 2.0 * D - v_n * dx

    # Handle edge case where denominator is near zero
    # This happens when diffusion is very small and drift is large
    if abs(denominator) < 1e-12:
        # Fall back to pure advection limit: reflect density
        return interior_value

    return interior_value * (numerator / denominator)


def ghost_cell_advection_diffusion_no_flux(
    interior_value: float,
    velocity_normal: float,
    diffusion_coeff: float,
    dx: float,
    grid_type: GridType = GridType.CELL_CENTERED,
) -> float:
    """
    Alias for ghost_cell_fp_no_flux with clearer parameter naming.

    This is the same as ghost_cell_fp_no_flux but with velocity_normal
    already accounting for the boundary orientation (positive = outward flow).

    Use this for general advection-diffusion equations where the no-flux BC
    means zero total flux J = v*u - D*∇u = 0.
    """
    # velocity_normal is already v·n (positive = outward)
    return ghost_cell_fp_no_flux(
        interior_value=interior_value,
        drift_velocity=velocity_normal,
        diffusion_coeff=diffusion_coeff,
        dx=dx,
        outward_normal_sign=1.0,  # Already accounted for in velocity_normal
        grid_type=grid_type,
    )


# =============================================================================
# Extrapolation Ghost Cell Formulas (for unbounded domains)
# =============================================================================


def ghost_cell_linear_extrapolation(
    interior_values: tuple[float, float],
) -> float:
    """
    Compute ghost cell value using linear extrapolation.

    This is equivalent to the **Zero Second Derivative Condition** (d²u/dx² = 0
    at the boundary). The function is assumed to continue linearly beyond the
    computational domain.

    Mathematical derivation:
        Let u_0 = first interior point, u_1 = second interior point
        Linear extrapolation: u_ghost = 2*u_0 - u_1

        This ensures: (u_ghost - 2*u_0 + u_1) / dx² = 0  (zero second derivative)

    Use cases:
        - HJB value functions on truncated unbounded domains
        - Far-field boundary conditions where solution grows linearly
        - Outflow boundaries in steady-state problems

    Args:
        interior_values: Tuple of (u_0, u_1) where u_0 is adjacent to ghost,
                        u_1 is one cell further into the interior

    Returns:
        Ghost cell value from linear extrapolation

    Example:
        >>> # At right boundary with interior values
        >>> u_ghost = ghost_cell_linear_extrapolation((u[-1], u[-2]))
        >>> # At left boundary with interior values
        >>> u_ghost = ghost_cell_linear_extrapolation((u[0], u[1]))

    Note:
        For problems with quadratic growth (e.g., LQG control), use
        ghost_cell_quadratic_extrapolation() instead.
    """
    u_0, u_1 = interior_values
    return 2.0 * u_0 - u_1


def ghost_cell_quadratic_extrapolation(
    interior_values: tuple[float, float, float],
) -> float:
    """
    Compute ghost cell value using quadratic extrapolation.

    This is equivalent to the **Zero Third Derivative Condition** (d³u/dx³ = 0
    at the boundary). The function is assumed to continue quadratically beyond
    the computational domain.

    Mathematical derivation:
        Let u_0, u_1, u_2 = three interior points (u_0 adjacent to ghost)
        Quadratic extrapolation: u_ghost = 3*u_0 - 3*u_1 + u_2

        This ensures the third derivative vanishes at the boundary.

    Use cases:
        - LQG-type HJB problems with quadratic value functions
        - Problems where linear extrapolation creates artificial "kinks"
        - Higher-accuracy far-field conditions

    Args:
        interior_values: Tuple of (u_0, u_1, u_2) where u_0 is adjacent to ghost,
                        u_1 is one cell in, u_2 is two cells into interior

    Returns:
        Ghost cell value from quadratic extrapolation

    Example:
        >>> # At right boundary
        >>> u_ghost = ghost_cell_quadratic_extrapolation((u[-1], u[-2], u[-3]))
        >>> # At left boundary
        >>> u_ghost = ghost_cell_quadratic_extrapolation((u[0], u[1], u[2]))

    Note:
        Requires at least 3 interior points. For smaller domains, use
        ghost_cell_linear_extrapolation() instead.
    """
    u_0, u_1, u_2 = interior_values
    return 3.0 * u_0 - 3.0 * u_1 + u_2


__all__ = [
    # Enums
    "DiscretizationType",
    "GridType",
    # Protocols
    "BCApplicatorProtocol",
    "Topology",
    "BoundaryCalculator",
    # Topology implementations
    "PeriodicTopology",
    "BoundedTopology",
    # Calculator implementations
    "DirichletCalculator",
    "NeumannCalculator",
    "RobinCalculator",
    "NoFluxCalculator",
    "LinearExtrapolationCalculator",
    "QuadraticExtrapolationCalculator",
    "FPNoFluxCalculator",
    # Base classes
    "BaseBCApplicator",
    "BaseStructuredApplicator",
    "BaseUnstructuredApplicator",
    "BaseMeshfreeApplicator",
    "BaseGraphApplicator",
    # Ghost cell helpers (2nd-order)
    "ghost_cell_dirichlet",
    "ghost_cell_neumann",
    "ghost_cell_robin",
    # Physics-aware ghost cell (for advection-diffusion/FP)
    "ghost_cell_fp_no_flux",
    "ghost_cell_advection_diffusion_no_flux",
    # Extrapolation ghost cell (for unbounded domains)
    "ghost_cell_linear_extrapolation",
    "ghost_cell_quadratic_extrapolation",
]
