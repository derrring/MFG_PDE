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

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable

    from .conditions import BoundaryConditions

# =============================================================================
# Generic Type Alias for Vectorized Operations (PEP 695, Python 3.12+)
# =============================================================================
# FieldData represents any valid field value type for boundary calculations.
# Used as a constraint for generic type parameters in Calculator methods.
#
# The template pattern [T: FieldData] ensures input/output type consistency:
# - If input is float, output is float
# - If input is NDArray, output is NDArray
#
# This is equivalent to C++ template<typename T> T compute(T val)
# =============================================================================
type FieldData = float | NDArray[np.floating]


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


@runtime_checkable
class BoundaryCapable(Protocol):
    """
    Protocol for solvers that support boundary condition handling.

    This protocol standardizes how solvers declare and use boundary conditions,
    enabling:
    1. Capability discovery - what BC types a solver supports
    2. Configuration - setting BCs via a uniform interface
    3. Infrastructure integration - automatic applicator selection

    **Integration Levels** (from BC_CAPABILITY_MATRIX.md):
    - **Low gap**: Solver fully uses geometry/boundary/ infrastructure
    - **Medium gap**: Solver uses some infrastructure (e.g., factory functions)
    - **High gap**: Solver has custom BC handling (integration candidate)

    Example implementation for a solver:
        >>> class MyFDMSolver:
        ...     _SUPPORTED_BC_TYPES = frozenset({
        ...         BCType.DIRICHLET, BCType.NEUMANN, BCType.PERIODIC, BCType.NO_FLUX
        ...     })
        ...
        ...     @property
        ...     def supported_bc_types(self) -> frozenset[BCType]:
        ...         return self._SUPPORTED_BC_TYPES
        ...
        ...     @property
        ...     def boundary_conditions(self) -> BoundaryConditions | None:
        ...         return self._bc
        ...
        ...     @boundary_conditions.setter
        ...     def boundary_conditions(self, bc: BoundaryConditions | None) -> None:
        ...         if bc is not None:
        ...             self._validate_bc_support(bc)
        ...         self._bc = bc
        ...
        ...     @property
        ...     def discretization_type(self) -> DiscretizationType:
        ...         return DiscretizationType.FDM

    Relationship to other protocols:
        - BCApplicatorProtocol: For applicator classes that apply BCs
        - BoundaryCapable: For solver classes that USE applicators (this protocol)
        - BoundaryCalculator: For computing ghost cell values (Layer 2)
    """

    @property
    def supported_bc_types(self) -> frozenset:
        """
        BC types this solver can handle.

        Returns:
            frozenset of BCType values the solver supports.

        Note:
            Use frozenset for immutability. Implementations should define
            this as a class attribute for efficiency:
                _SUPPORTED_BC_TYPES = frozenset({BCType.DIRICHLET, BCType.NEUMANN})
        """
        ...

    @property
    def boundary_conditions(self) -> BoundaryConditions | None:
        """
        Current boundary condition configuration.

        Returns:
            BoundaryConditions object or None if not configured.
        """
        ...

    @property
    def discretization_type(self) -> DiscretizationType:
        """
        Discretization method this solver uses.

        Used to select the appropriate applicator class:
        - FDM -> FDMApplicator (ghost cells)
        - FEM -> FEMApplicator (matrix modification)
        - MESHFREE/GFDM -> MeshfreeApplicator (collocation/particles)
        - GRAPH -> GraphApplicator (network BCs)
        """
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

    **VECTORIZATION**: All calculators support both scalar and array inputs.
    When interior_value is an NDArray, the output is also an NDArray of the
    same shape. This enables efficient vectorized operations without Python loops.

    **TYPE SAFETY**: Uses TypeVar (template pattern) to ensure input/output type
    consistency. If input is float, output is float. If input is NDArray, output
    is NDArray. This is equivalent to C++ template<typename T> T compute(T val).

    Example:
        >>> # Scalar usage
        >>> calculator = DirichletCalculator(boundary_value=0.0)
        >>> u_ghost = calculator.compute(interior_value=1.0, dx=0.1, side='min')
        # Returns -1.0 (float)

        >>> # Vectorized usage (100x-1000x faster for large arrays)
        >>> interior = np.array([1.0, 2.0, 3.0])
        >>> u_ghost = calculator.compute(interior_value=interior, dx=0.1, side='min')
        # Returns array([-1.0, -2.0, -3.0]) (NDArray)
    """

    def compute[T: FieldData](
        self,
        interior_value: T,
        dx: float,
        side: str,
        **kwargs,
    ) -> T:
        """
        Compute ghost cell value from interior value (vectorized).

        Args:
            interior_value: Value(s) at interior point(s) adjacent to boundary.
                           Can be scalar float or NDArray for vectorized operation.
            dx: Grid spacing
            side: Boundary side ('min' or 'max')
            **kwargs: Additional parameters (e.g., time for time-varying BCs)

        Returns:
            Ghost cell value(s). Same type/shape as interior_value (template pattern).
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

    Supports vectorized operations for efficient array processing.
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

    def compute[T: FieldData](
        self,
        interior_value: T,
        dx: float,
        side: str,
        **kwargs,
    ) -> T:
        """Compute ghost value for Dirichlet BC (vectorized)."""
        # NumPy broadcasting handles both scalar and array inputs
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

    Supports vectorized operations for efficient array processing.
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

    def compute[T: FieldData](
        self,
        interior_value: T,
        dx: float,
        side: str,
        **kwargs,
    ) -> T:
        """Compute ghost value for Neumann BC (vectorized)."""
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

    Supports vectorized operations for efficient array processing.
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

    def compute[T: FieldData](
        self,
        interior_value: T,
        dx: float,
        side: str,
        **kwargs,
    ) -> T:
        """Compute ghost value for Robin BC (vectorized)."""
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


class ZeroGradientCalculator:
    """
    Calculator for zero gradient (du/dn = 0) boundary conditions.

    Implements edge extension: ghost = interior, ensuring du/dn = 0.

    Physical meaning: The field has no gradient normal to the boundary.
    Use cases:
    - HJB value functions at reflective walls
    - Any field needing smooth extension at boundaries

    **For mass-conserving boundaries (FP equations), use ZeroFluxCalculator instead.**

    Supports vectorized operations for efficient array processing.
    """

    def __init__(self, grid_type: GridType = GridType.CELL_CENTERED):
        self._grid_type = grid_type

    def compute[T: FieldData](
        self,
        interior_value: T,
        dx: float,
        side: str,
        **kwargs,
    ) -> T:
        """Compute ghost value for zero gradient BC (edge extension, vectorized)."""
        # Simply return interior value - works for both scalar and array
        return interior_value

    def __repr__(self) -> str:
        return "ZeroGradientCalculator()"


# Backward compatibility alias (with deprecation warning)
class NoFluxCalculator(ZeroGradientCalculator):
    """
    Deprecated alias for ZeroGradientCalculator.

    .. deprecated:: 0.16.11
        Use :class:`ZeroGradientCalculator` instead for du/dn = 0.
        For mass-conserving flux BC (J·n = 0), use :class:`ZeroFluxCalculator`.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "NoFluxCalculator is deprecated since v0.16.11. "
            "Use ZeroGradientCalculator for du/dn = 0 (edge extension), "
            "or ZeroFluxCalculator for J·n = 0 (mass conservation).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class LinearExtrapolationCalculator:
    """
    Calculator for linear extrapolation boundary conditions.

    Uses zero second derivative (d²u/dx² = 0) at boundary.
    Ghost = 2*u_0 - u_1

    Suitable for HJB problems with linear value growth at infinity.
    Supports vectorized operations for efficient array processing.
    """

    def compute[T: FieldData](
        self,
        interior_value: T,
        dx: float,
        side: str,
        second_interior_value: T | None = None,
        **kwargs,
    ) -> T:
        """
        Compute ghost value via linear extrapolation (vectorized).

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
        # Vectorized: works for both scalar and array
        return 2.0 * interior_value - second_interior_value

    def __repr__(self) -> str:
        return "LinearExtrapolationCalculator()"


class QuadraticExtrapolationCalculator:
    """
    Calculator for quadratic extrapolation boundary conditions.

    Uses zero third derivative (d³u/dx³ = 0) at boundary.
    Ghost = 3*u_0 - 3*u_1 + u_2

    Suitable for LQG-type problems with quadratic value functions.
    Supports vectorized operations for efficient array processing.
    """

    def compute[T: FieldData](
        self,
        interior_value: T,
        dx: float,
        side: str,
        second_interior_value: T | None = None,
        third_interior_value: T | None = None,
        **kwargs,
    ) -> T:
        """
        Compute ghost value via quadratic extrapolation (vectorized).

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
                return 2.0 * interior_value - second_interior_value
            return interior_value
        # Vectorized: works for both scalar and array
        return 3.0 * interior_value - 3.0 * second_interior_value + third_interior_value

    def __repr__(self) -> str:
        return "QuadraticExtrapolationCalculator()"


class ZeroFluxCalculator:
    """
    Calculator for zero total flux (J·n = 0) boundary conditions.

    For advection-diffusion equations, this ensures the total flux
    J = v*ρ - D*∇ρ vanishes at the boundary, preserving mass conservation.

    Formula: u_ghost = (2D + v*dx) / (2D - v*dx) * u_interior

    Physical meaning: No mass/probability crosses the boundary.
    Use cases:
    - Fokker-Planck density with impermeable walls
    - Any advection-diffusion equation requiring mass conservation

    **For zero gradient (du/dn = 0), use ZeroGradientCalculator instead.**

    Supports vectorized operations for efficient array processing.
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

    def compute[T: FieldData](
        self,
        interior_value: T,
        dx: float,
        side: str,
        drift_velocity: float | None = None,
        **kwargs,
    ) -> T:
        """
        Compute ghost value for FP no-flux BC (vectorized).

        Args:
            interior_value: Density at interior point(s)
            dx: Grid spacing
            side: Boundary side ('min' or 'max')
            drift_velocity: Override drift velocity (optional)

        Returns:
            Ghost value(s) ensuring zero total flux J·n = 0
        """
        outward_sign = 1.0 if side == "max" else -1.0
        v = drift_velocity if drift_velocity is not None else self._drift
        # Vectorized formula: works for both scalar and array
        return ghost_cell_fp_no_flux(
            interior_value,
            v,
            self._diffusion,
            dx,
            outward_sign,
            self._grid_type,
        )

    def __repr__(self) -> str:
        return f"ZeroFluxCalculator(drift={self._drift}, D={self._diffusion})"


# Backward compatibility alias (with deprecation warning)
class FPNoFluxCalculator(ZeroFluxCalculator):
    """
    Deprecated alias for ZeroFluxCalculator.

    .. deprecated:: 0.16.11
        Use :class:`ZeroFluxCalculator` instead for J·n = 0 (mass conservation).
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "FPNoFluxCalculator is deprecated since v0.16.11. "
            "Use ZeroFluxCalculator instead for J·n = 0 (mass conservation).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# =============================================================================
# LinearConstraint: Bridge between Ghost Cells and Matrix Assembly
# =============================================================================
#
# The "Tier-Based Coefficient Folding" pattern from bc_architecture_analysis.md
#
# For EXPLICIT schemes: Ghost cells are filled with computed values
#   u_ghost = Calculator.compute(u_inner, dx)
#
# For IMPLICIT schemes: Ghost node relationships become matrix coefficients
#   u_ghost = sum(weights[k] * u[inner+k]) + bias
#
# This dataclass expresses the linear relationship for matrix folding:
#   Tier 1 (State/Dirichlet): weights={}, bias=value
#   Tier 2 (Gradient/Neumann): weights={0: 1.0}, bias=dx*grad
#   Tier 3 (Flux/Robin): weights={0: alpha}, bias=0
#   Tier 4 (Artificial/Extrapolation): weights={0: 2.0, 1: -1.0}, bias=0
# =============================================================================


@dataclass
class LinearConstraint:
    """
    Linear constraint expressing ghost cell as function of interior values.

    For matrix assembly, when a stencil accesses ghost index j, the assembler:
    1. Adds weight * w to A[i, inner+k] for each (k, w) in weights
    2. Subtracts weight * bias from b[i]

    This is the "Coefficient Folding" pattern from the 2+4 BC architecture.

    Attributes:
        weights: Mapping from relative offset to weight. Offset 0 = boundary cell,
                 offset 1 = one cell inward, etc.
        bias: Constant term (for Dirichlet values or gradient offsets)

    Examples:
        # Tier 1: Dirichlet u=g -> u_ghost = g (constant)
        LinearConstraint(weights={}, bias=g)

        # Tier 2: Neumann du/dn=0 -> u_ghost = u_inner
        LinearConstraint(weights={0: 1.0}, bias=0.0)

        # Tier 2: Neumann du/dn=g -> u_ghost = u_inner + dx*g
        LinearConstraint(weights={0: 1.0}, bias=dx * g)

        # Tier 3: Robin (FP no-flux) -> u_ghost = alpha * u_inner
        LinearConstraint(weights={0: alpha}, bias=0.0)

        # Tier 4: Linear extrapolation -> u_ghost = 2*u[0] - u[1]
        LinearConstraint(weights={0: 2.0, 1: -1.0}, bias=0.0)
    """

    weights: dict[int, float]
    bias: float = 0.0


def calculator_to_constraint(
    calculator: BoundaryCalculator | None,
    dx: float,
    side: str,
    grid_type: GridType = GridType.CELL_CENTERED,
) -> LinearConstraint:
    """
    Convert a BoundaryCalculator to LinearConstraint for matrix assembly.

    This bridges the explicit (ghost cell) and implicit (matrix) worlds,
    ensuring mathematical equivalence as required by GKS stability.

    Args:
        calculator: The physics calculator (None for periodic topology)
        dx: Grid spacing
        side: Boundary side ('min' or 'max')
        grid_type: Grid alignment type

    Returns:
        LinearConstraint describing how ghost depends on interior values

    Note:
        For periodic topology, this function should not be called - the
        Topology layer handles periodic by index wrapping, no physics needed.
    """
    if calculator is None:
        # Periodic topology - should not reach here
        raise ValueError("Periodic boundaries use index wrapping, not LinearConstraint")

    # Tier 1: State constraints (Dirichlet)
    if isinstance(calculator, DirichletCalculator):
        return LinearConstraint(weights={}, bias=calculator._boundary_value)

    # Tier 2: Gradient constraints (Neumann/ZeroGradient)
    if isinstance(calculator, (NeumannCalculator, ZeroGradientCalculator)):
        flux_value = calculator._flux_value if isinstance(calculator, NeumannCalculator) else 0.0
        # For cell-centered: u_ghost = u_inner ± dx * g (sign depends on side)
        sign = 1.0 if side == "max" else -1.0
        return LinearConstraint(weights={0: 1.0}, bias=sign * dx * flux_value)

    # Tier 3: Flux constraints (Robin/ZeroFlux)
    if isinstance(calculator, (RobinCalculator, ZeroFluxCalculator)):
        if isinstance(calculator, ZeroFluxCalculator):
            # FP no-flux: alpha = (2D + v*dx) / (2D - v*dx)
            v = calculator._drift
            D = calculator._diffusion
            outward_sign = 1.0 if side == "max" else -1.0
            v_n = v * outward_sign
            alpha = (2 * D + v_n * dx) / (2 * D - v_n * dx + 1e-14)
            return LinearConstraint(weights={0: alpha}, bias=0.0)
        else:
            # General Robin: α*u + β*(u_ghost - u_inner)/(2*dx) = g
            # (central difference for du/dn, outward sign absorbed by side convention)
            # Solving for u_ghost:
            #   u_ghost = u_inner * (β - 2*α*dx) / (β + 2*α*dx) + 4*g*dx / (β + 2*α*dx)
            # when β = 0 → Dirichlet; when α = 0 → Neumann (degenerate cases handled above)
            alpha = calculator._alpha
            beta = calculator._beta
            g = calculator._rhs_value
            outward_sign = 1.0 if side == "max" else -1.0
            # Effective beta with outward sign
            beta_eff = beta * outward_sign
            denom = beta_eff + 2 * alpha * dx
            if abs(denom) < 1e-14:
                # Degenerate: fall back to copy (Neumann-like)
                return LinearConstraint(weights={0: 1.0}, bias=0.0)
            weight = (beta_eff - 2 * alpha * dx) / denom
            bias = 2 * g * dx / denom
            return LinearConstraint(weights={0: weight}, bias=bias)

    # Tier 4: Artificial constraints (Extrapolation)
    if isinstance(calculator, LinearExtrapolationCalculator):
        # u_ghost = 2*u[0] - u[1]
        return LinearConstraint(weights={0: 2.0, 1: -1.0}, bias=0.0)

    if isinstance(calculator, QuadraticExtrapolationCalculator):
        # u_ghost = 3*u[0] - 3*u[1] + u[2]
        return LinearConstraint(weights={0: 3.0, 1: -3.0, 2: 1.0}, bias=0.0)

    # Default fallback: Neumann-like (copy interior)
    return LinearConstraint(weights={0: 1.0}, bias=0.0)


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

    This class provides shared infrastructure for ghost cell computation
    following the Template Method Pattern (Issue #598):
    - Shared validation logic
    - Shared ghost cell formulas (Dirichlet, Neumann, Robin)
    - Shared buffer creation utilities
    - Dimension-specific slicing in subclasses

    Usage:
        Subclasses override _fill_ghost_cells() for dimension-specific logic
        but use shared formula methods for ghost cell values.
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

    # =========================================================================
    # Shared Ghost Cell Formula Methods (Issue #598 - DRY Principle)
    # =========================================================================

    def _compute_ghost_dirichlet(
        self,
        u_interior: NDArray | float,
        g: float | Callable[[float], float],
        time: float = 0.0,
    ) -> NDArray | float:
        """
        Compute Dirichlet ghost cell value (shared formula).

        For cell-centered grid with boundary at cell face:
            u_boundary = (u_ghost + u_interior) / 2 = g
            => u_ghost = 2*g - u_interior

        For vertex-centered grid with boundary at grid point:
            u_ghost = g

        Args:
            u_interior: Interior values adjacent to boundary
            g: Boundary value (constant or callable)
            time: Current time (for callable g)

        Returns:
            Ghost cell values with same type as u_interior

        Note:
            This formula is shared across 1D/2D/3D/nD to eliminate duplication.
            Previously duplicated at applicator_fdm.py:232, 1255, 1725.
        """
        # Evaluate callable BC values
        if callable(g):
            g_val = g(time)
        else:
            g_val = g

        # Apply formula based on grid type
        if self._grid_type == GridType.CELL_CENTERED:
            return 2.0 * g_val - u_interior
        else:  # VERTEX_CENTERED
            if isinstance(u_interior, np.ndarray):
                return np.full_like(u_interior, g_val)
            else:
                return g_val

    def _compute_ghost_neumann(
        self,
        u_interior: NDArray | float,
        u_next_interior: NDArray | float,
        g: float | Callable[[float], float],
        dx: float,
        side: str,
        time: float = 0.0,
    ) -> NDArray | float:
        """
        Compute Neumann ghost cell value (shared formula).

        For central difference gradient stencil with cell-centered grid:
        - Left boundary (normal points left/inward):
          du/dn = (u_interior - u_ghost) / (2*dx) = -g  (inward normal)
          => u_ghost = u_interior + 2*dx*g
          For zero-flux (g=0): u_ghost = u_next_interior (reflection)

        - Right boundary (normal points right/outward):
          du/dn = (u_ghost - u_interior) / (2*dx) = g  (outward normal)
          => u_ghost = u_interior + 2*dx*g
          For zero-flux (g=0): u_ghost = u_prev_interior (reflection)

        Args:
            u_interior: Interior values adjacent to boundary
            u_next_interior: Interior values one step inward from boundary
            g: Flux value (constant or callable)
            dx: Grid spacing
            side: Boundary side ("left" or "right")
            time: Current time (for callable g)

        Returns:
            Ghost cell values with same type as u_interior

        Note:
            Issue #542 fix: Uses reflection formula (ghost = u_next_interior)
            for central difference stencil, not edge extension (ghost = u_interior).

            This formula is shared across 1D/2D/3D/nD to eliminate duplication.
            Previously duplicated at applicator_fdm.py:1099, 1274, 1748.
        """
        # Evaluate callable BC values
        if callable(g):
            g_val = g(time)
        else:
            g_val = g

        # Zero-flux case: reflection (Issue #542 fix)
        if np.isclose(g_val, 0.0):
            return u_next_interior.copy() if isinstance(u_next_interior, np.ndarray) else u_next_interior

        # General Neumann case
        if side == "left":
            # Left boundary: u_ghost = u_next_interior - 2*dx*g
            return u_next_interior - 2.0 * dx * g_val
        else:  # "right"
            # Right boundary: u_ghost = u_next_interior + 2*dx*g
            return u_next_interior + 2.0 * dx * g_val

    def _compute_ghost_robin(
        self,
        u_interior: NDArray | float,
        alpha: float,
        beta: float,
        g: float | Callable[[float], float],
        dx: float,
        side: str,
        time: float = 0.0,
    ) -> NDArray | float:
        """
        Compute Robin ghost cell value (shared formula).

        Robin BC: alpha*u + beta*du/dn = g at boundary

        For cell-centered grid with boundary at cell face (ghost at -dx/2, interior at +dx/2):
            u_boundary = (u_ghost + u_interior)/2
            du/dn = (u_ghost - u_interior)/dx  (cell centers are dx apart)

            alpha * (u_ghost + u_interior)/2 + beta * (u_ghost - u_interior)/dx = g
            => u_ghost * (alpha/2 + beta/dx) = g - u_interior * (alpha/2 - beta/dx)

        Args:
            u_interior: Interior values adjacent to boundary
            alpha: Robin coefficient for value term
            beta: Robin coefficient for flux term
            g: Robin boundary value (constant or callable)
            dx: Grid spacing
            side: Boundary side ("left" or "right")
            time: Current time (for callable g)

        Returns:
            Ghost cell values with same type as u_interior

        Note:
            This formula is shared across 1D/2D/3D/nD to eliminate duplication.
            Fix (commit 0ae5515a): Changed from /(2*dx) to /dx.
        """
        # Evaluate callable BC values
        if callable(g):
            g_val = g(time)
        else:
            g_val = g

        # Robin formula (valid for both left and right boundaries)
        # Cell centers are dx apart, not 2*dx
        coeff_ghost = alpha / 2.0 + beta / dx
        coeff_interior = alpha / 2.0 - beta / dx

        return (g_val - u_interior * coeff_interior) / coeff_ghost

    # =========================================================================
    # Shared Validation and Utility Methods (Issue #598)
    # =========================================================================

    def _validate_field(self, field: NDArray) -> None:
        """
        Validate field before BC application (shared logic).

        Checks:
        - Field contains finite values (no NaN/Inf)

        Args:
            field: Interior field to validate

        Raises:
            ValueError: If validation fails

        Note:
            This validation is shared across all dimension-specific functions
            to eliminate duplication. Previously duplicated at applicator_fdm.py:1247.
        """
        if not np.isfinite(field).all():
            raise ValueError("Field contains NaN or Inf values. Check solver convergence and boundary conditions.")

    def _create_padded_buffer(
        self,
        field: NDArray,
        ghost_depth: int = 1,
    ) -> NDArray:
        """
        Create zero-initialized padded buffer (shared logic).

        Args:
            field: Interior field
            ghost_depth: Number of ghost cells per side (default: 1)

        Returns:
            Zero-initialized padded array with shape (N1+2*depth, N2+2*depth, ...)

        Note:
            Interior values are copied to padded buffer.
            Ghost cells initialized to zero (will be filled by BC application).
        """
        # Compute padded shape
        padded_shape = tuple(n + 2 * ghost_depth for n in field.shape)

        # Create zero buffer
        padded = np.zeros(padded_shape, dtype=field.dtype)

        # Copy interior values
        interior_slices = tuple(slice(ghost_depth, -ghost_depth) for _ in range(field.ndim))
        padded[interior_slices] = field

        return padded

    def _compute_grid_spacing(
        self,
        field: NDArray,
        domain_bounds: NDArray,
    ) -> tuple[float, ...]:
        """
        Compute grid spacing from domain bounds (shared logic).

        Args:
            field: Interior field (to get shape)
            domain_bounds: Domain bounds shape (ndim, 2)

        Returns:
            Grid spacing for each dimension

        Note:
            This computation is shared across dimension-specific functions
            to eliminate duplication. Previously duplicated at applicator_fdm.py:2664.
        """
        domain_bounds = np.atleast_2d(domain_bounds)
        spacing = []

        for d in range(field.ndim):
            extent = domain_bounds[d, 1] - domain_bounds[d, 0]
            n_points = field.shape[d]
            dx = extent / (n_points - 1) if n_points > 1 else extent
            spacing.append(dx)

        return tuple(spacing)


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

    For cell-centered grids (ghost at -dx/2, interior at +dx/2, boundary at 0):
        u_boundary = (u_ghost + u_interior) / 2
        du/dn = (u_ghost - u_interior) / dx  (distance between cell centers is dx)

        alpha * (u_ghost + u_interior)/2 + beta * (u_ghost - u_interior)/dx = g

    Solving for u_ghost:
        u_ghost * (alpha/2 + beta/dx) = g - u_interior * (alpha/2 - beta/dx)

    IMPORTANT: For cell-centered grids, du/dn = (u_ghost - u_interior)/dx for BOTH
    boundaries because ghost is always "outside" and interior is always "inside"
    regardless of left/right. The outward_normal_sign parameter is kept for backward
    compatibility but is NOT used in the cell-centered formula.

    For vertex-centered grids, the sign convention differs.
    """
    if grid_type == GridType.VERTEX_CENTERED:
        # Vertex-centered: sign matters because derivative direction differs
        if abs(alpha) > 1e-12:
            return (rhs_value - beta * outward_normal_sign * interior_value / dx) / alpha
        else:
            return interior_value + dx * rhs_value / beta * outward_normal_sign

    # Cell-centered: ghost and interior are dx apart
    # CRITICAL: du/dn = (u_ghost - u_interior)/dx for BOTH left and right boundaries
    # The outward_normal_sign is NOT used here because the geometry is symmetric:
    # - At left boundary: ghost at -dx/2, interior at +dx/2
    # - At right boundary: interior at L-dx/2, ghost at L+dx/2
    # In both cases, (u_ghost - u_interior)/dx gives the outward normal derivative.
    coeff_ghost = alpha / 2.0 + beta / dx
    coeff_interior = alpha / 2.0 - beta / dx

    if abs(coeff_ghost) < 1e-12:
        raise ValueError("Robin BC coefficients lead to singular ghost cell formula")

    return (rhs_value - interior_value * coeff_interior) / coeff_ghost


# =============================================================================
# High-Order Ghost Cell Extrapolation (for WENO and other high-order schemes)
# =============================================================================


def high_order_ghost_dirichlet(
    interior_values: list[float],
    boundary_value: float,
    order: int = 4,
    grid_type: GridType = GridType.CELL_CENTERED,
) -> list[float]:
    """
    Compute high-order accurate ghost cell values for Dirichlet BC.

    For WENO5 and other high-order schemes, 2nd-order ghost cells degrade
    boundary accuracy. This function provides 4th or 5th order extrapolation.

    Mathematical derivation (cell-centered, 4th order):
        Given: u_b = g (Dirichlet BC at cell face x = x_0 - dx/2)
        Want: ghost values u_{-1}, u_{-2} that preserve polynomial accuracy

        Using Lagrange interpolation through boundary point and interior points:
        - u_{-1} extrapolated from {g, u_0, u_1, u_2}
        - u_{-2} extrapolated from {g, u_{-1}, u_0, u_1, u_2}

    Args:
        interior_values: Interior point values [u_0, u_1, u_2, ...] from boundary inward
        boundary_value: Dirichlet BC value g
        order: Extrapolation order (4 or 5)
        grid_type: Grid type (cell-centered or vertex-centered)

    Returns:
        Ghost values [u_{-1}, u_{-2}] (first is adjacent to interior)

    References:
        - Fedkiw et al. (1999): "A Non-oscillatory Eulerian Approach..."
        - Shu (1998): "Essentially Non-Oscillatory and WENO Schemes..."
    """
    if order < 4:
        # Fall back to 2nd-order for low orders
        u_int = interior_values[0]
        g = boundary_value
        if grid_type == GridType.VERTEX_CENTERED:
            return [g, 2 * g - u_int]
        else:
            u_ghost_1 = 2.0 * g - u_int
            u_ghost_2 = 2.0 * g - interior_values[1] if len(interior_values) > 1 else u_ghost_1
            return [u_ghost_1, u_ghost_2]

    # High-order extrapolation (4th or 5th order)
    g = boundary_value
    u = interior_values

    if grid_type == GridType.VERTEX_CENTERED:
        # For vertex-centered, boundary is at a grid point
        # u_{-1} = g directly
        # u_{-2} extrapolated using polynomial through g, u_0, u_1, u_2
        if order >= 4 and len(u) >= 3:
            # 4th-order extrapolation for u_{-2}
            # Using Lagrange polynomial through (x=-1, g), (x=0, u0), (x=1, u1), (x=2, u2)
            # evaluated at x=-2
            u_ghost_2 = 4 * g - 6 * u[0] + 4 * u[1] - u[2]
        else:
            u_ghost_2 = 2 * g - u[0]
        return [g, u_ghost_2]

    # Cell-centered: boundary at cell face (x = x_0 - dx/2)
    # Ghost cell centers are at x = x_0 - dx, x_0 - 2*dx, etc.
    # Boundary value g is at x = x_0 - dx/2

    if order >= 5 and len(u) >= 4:
        # 5th-order Lagrange extrapolation
        # Points: (x=-0.5, g), (x=0, u0), (x=1, u1), (x=2, u2), (x=3, u3)
        # Evaluate at x=-1 and x=-2

        # Coefficients derived from Lagrange interpolation formula
        # u_{-1} at x = -1:
        u_ghost_1 = (16 / 5) * g - 3 * u[0] + (8 / 5) * u[1] - (1 / 3) * u[2] + (1 / 30) * u[3]

        # u_{-2} at x = -2:
        u_ghost_2 = (48 / 5) * g - 12 * u[0] + 8 * u[1] - (8 / 3) * u[2] + (2 / 5) * u[3]
        return [u_ghost_1, u_ghost_2]

    elif order >= 4 and len(u) >= 3:
        # 4th-order Lagrange extrapolation
        # Points: (x=-0.5, g), (x=0, u0), (x=1, u1), (x=2, u2)
        # Evaluate at x=-1 and x=-2

        # u_{-1} at x = -1 (using 4-point Lagrange)
        u_ghost_1 = (16 / 5) * g - 3 * u[0] + (8 / 5) * u[1] - (1 / 5) * u[2]

        # u_{-2} at x = -2
        u_ghost_2 = (48 / 5) * g - 12 * u[0] + 8 * u[1] - (8 / 5) * u[2]
        return [u_ghost_1, u_ghost_2]

    else:
        # Fall back to 2nd-order
        u_ghost_1 = 2.0 * g - u[0]
        u_ghost_2 = 2.0 * g - u[1] if len(u) > 1 else u_ghost_1
        return [u_ghost_1, u_ghost_2]


def high_order_ghost_neumann(
    interior_values: list[float],
    flux_value: float,
    dx: float,
    outward_normal_sign: float = 1.0,
    order: int = 4,
    grid_type: GridType = GridType.CELL_CENTERED,
) -> list[float]:
    """
    Compute high-order accurate ghost cell values for Neumann BC.

    Mathematical derivation (cell-centered, 4th order):
        Given: du/dn = g (Neumann BC at cell face)
        Want: ghost values that preserve polynomial accuracy

        The key constraint is that the derivative at the boundary matches g.
        Using polynomial extrapolation with derivative constraint.

    Args:
        interior_values: Interior point values [u_0, u_1, u_2, ...] from boundary inward
        flux_value: Neumann BC value (du/dn = g)
        dx: Grid spacing
        outward_normal_sign: +1 for max boundary, -1 for min boundary
        order: Extrapolation order (4 or 5)
        grid_type: Grid type

    Returns:
        Ghost values [u_{-1}, u_{-2}] (first is adjacent to interior)
    """
    g = flux_value * outward_normal_sign
    u = interior_values

    if order < 4 or len(u) < 3:
        # Fall back to 2nd-order
        u_ghost_1 = u[0] + 2.0 * dx * g
        u_ghost_2 = u[1] + 4.0 * dx * g if len(u) > 1 else u_ghost_1 + 2.0 * dx * g
        return [u_ghost_1, u_ghost_2]

    if grid_type == GridType.VERTEX_CENTERED:
        # Vertex-centered: boundary at grid point
        # du/dn = (u_0 - u_{-1}) / dx = g => u_{-1} = u_0 - dx*g
        u_ghost_1 = u[0] - dx * g

        if order >= 4 and len(u) >= 3:
            # 4th-order: Use polynomial matching derivative at boundary
            # du/dn|_{x=0} = g and smooth extrapolation through interior
            u_ghost_2 = u_ghost_1 - dx * g  # Maintain constant derivative
        else:
            u_ghost_2 = u_ghost_1 - dx * g
        return [u_ghost_1, u_ghost_2]

    # Cell-centered: boundary at cell face (x = x_0 - dx/2)
    # Constraint: du/dn at x = -dx/2 equals g

    if order >= 5 and len(u) >= 4:
        # 5th-order extrapolation with Neumann constraint
        # Construct polynomial through (x=0, u0), (x=1, u1), (x=2, u2), (x=3, u3)
        # and enforce derivative = g at x = -0.5

        # One-sided 4th-order derivative at boundary:
        # du/dx|_{x=-0.5} = (-25*u_{-1} + 48*u_0 - 36*u_1 + 16*u_2 - 3*u_3) / (12*dx)
        # Solve for u_{-1} given du/dx = g

        u_ghost_1 = (48 * u[0] - 36 * u[1] + 16 * u[2] - 3 * u[3] - 12 * dx * g) / 25

        # For u_{-2}, use polynomial continuation
        # du/dx|_{x=-1.5} should match smooth extrapolation
        u_ghost_2 = (48 * u_ghost_1 - 36 * u[0] + 16 * u[1] - 3 * u[2] - 12 * dx * g) / 25

        return [u_ghost_1, u_ghost_2]

    elif order >= 4 and len(u) >= 3:
        # 4th-order extrapolation with Neumann constraint
        # Using 3rd-order one-sided difference:
        # du/dx|_{x=-0.5} = (-11*u_{-1} + 18*u_0 - 9*u_1 + 2*u_2) / (6*dx) = g

        u_ghost_1 = (18 * u[0] - 9 * u[1] + 2 * u[2] - 6 * dx * g) / 11

        # For u_{-2}, maintain the derivative constraint
        u_ghost_2 = (18 * u_ghost_1 - 9 * u[0] + 2 * u[1] - 6 * dx * g) / 11

        return [u_ghost_1, u_ghost_2]

    else:
        # Fall back to 2nd-order
        u_ghost_1 = u[0] + 2.0 * dx * g
        u_ghost_2 = u[1] + 4.0 * dx * g if len(u) > 1 else u_ghost_1 + 2.0 * dx * g
        return [u_ghost_1, u_ghost_2]


# =============================================================================
# Physics-Aware Ghost Cell Formulas
# =============================================================================
# IMPORTANT LESSON: The discretized BC must match the physics, not just the
# mathematical form. For advection-diffusion equations (like Fokker-Planck),
# a "no-flux" BC means J·n = 0 where J = v*ρ - D*∇ρ.
#
# - Naive approach: Neumann (∂ρ/∂n = 0) only zeroes diffusion flux
# - Correct approach: Robin BC that zeroes TOTAL flux
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
    "BoundaryCapable",
    "Topology",
    "BoundaryCalculator",
    # Topology implementations
    "PeriodicTopology",
    "BoundedTopology",
    # Calculator implementations (physics-based naming)
    "DirichletCalculator",
    "NeumannCalculator",
    "RobinCalculator",
    "ZeroGradientCalculator",  # du/dn = 0 (edge extension)
    "ZeroFluxCalculator",  # J·n = 0 (mass conservation)
    "LinearExtrapolationCalculator",
    "QuadraticExtrapolationCalculator",
    # Backward compatibility aliases
    "NoFluxCalculator",  # -> ZeroGradientCalculator
    "FPNoFluxCalculator",  # -> ZeroFluxCalculator
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
    # High-order ghost cell extrapolation (4th/5th order for WENO)
    "high_order_ghost_dirichlet",
    "high_order_ghost_neumann",
    # Physics-aware ghost cell (for advection-diffusion/FP)
    "ghost_cell_fp_no_flux",
    "ghost_cell_advection_diffusion_no_flux",
    # Extrapolation ghost cell (for unbounded domains)
    "ghost_cell_linear_extrapolation",
    "ghost_cell_quadratic_extrapolation",
    # Matrix assembly support (Tier-Based Coefficient Folding)
    "LinearConstraint",
    "calculator_to_constraint",
]
