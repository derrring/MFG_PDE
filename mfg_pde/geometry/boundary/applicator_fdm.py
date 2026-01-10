"""
Boundary Condition Applicator for PDE Solvers.

This module provides utilities for applying mixed boundary conditions to grid-based
fields. Supports:

1. Uniform BCs (same type on all boundaries)
2. Mixed BCs (different types on different boundary segments)
3. Multiple BC types: Dirichlet, Neumann, Robin, Periodic
4. Time-dependent boundary values
5. Cell-centered and vertex-centered grids

The ghost cell formulas are derived for standard 3-point FD stencils.

Ghost Cell Formulas (cell-centered grid, boundary at cell face):
--------------------------------------------------------------
Let u_i = interior point, u_g = ghost point, u_b = boundary value
Boundary is located at midpoint: x_b = (x_g + x_i) / 2

Dirichlet (u = g at boundary):
    u_b = (u_g + u_i) / 2 = g
    => u_g = 2*g - u_i

Neumann (du/dn = g at boundary, outward normal):
    (u_i - u_g) / (2*dx) = g  (for left/bottom boundary, normal points inward)
    => u_g = u_i - 2*dx*g

    (u_g - u_i) / (2*dx) = g  (for right/top boundary, normal points outward)
    => u_g = u_i + 2*dx*g

Robin (alpha*u + beta*du/dn = g at boundary):
    alpha * (u_g + u_i)/2 + beta * (u_g - u_i)/(2*dx) = g
    => u_g * (alpha/2 + beta/(2*dx)) = g - u_i * (alpha/2 - beta/(2*dx))
    => u_g = (g - u_i * (alpha/2 - beta/(2*dx))) / (alpha/2 + beta/(2*dx))

Usage:
    from mfg_pde.geometry.boundary import apply_boundary_conditions_2d

    # With uniform BCs
    m_padded = apply_boundary_conditions_2d(m, uniform_bc, domain_bounds)

    # With mixed BCs and time
    m_padded = apply_boundary_conditions_2d(m, mixed_bc, domain_bounds, time=0.5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mfg_pde.utils.mfg_logging import get_logger

from .applicator_base import (
    BaseStructuredApplicator,
    BoundaryCalculator,
    BoundedTopology,
    DirichletCalculator,
    GridType,
    LinearExtrapolationCalculator,
    NeumannCalculator,
    PeriodicTopology,
    QuadraticExtrapolationCalculator,
    RobinCalculator,
    Topology,
    ZeroFluxCalculator,
    ZeroGradientCalculator,
)
from .conditions import BoundaryConditions
from .fdm_bc_1d import BoundaryConditions as BoundaryConditions1DFDM
from .types import BCType

logger = get_logger(__name__)

# Backward compatibility alias
LegacyBoundaryConditions1D = BoundaryConditions1DFDM

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from .types import BCSegment


@dataclass
class GhostCellConfig:
    """Configuration for ghost cell computation."""

    # Grid type affects ghost cell formula (use GridType enum or string)
    grid_type: GridType | str = GridType.CELL_CENTERED

    # For vertex-centered grids, boundary is at grid point
    # For cell-centered grids, boundary is at cell face (between ghost and interior)

    def __post_init__(self) -> None:
        """Convert string grid_type to enum for backward compatibility."""
        if isinstance(self.grid_type, str):
            self.grid_type = GridType.VERTEX_CENTERED if self.grid_type == "vertex_centered" else GridType.CELL_CENTERED

    @property
    def is_vertex_centered(self) -> bool:
        """Check if grid is vertex-centered."""
        return self.grid_type == GridType.VERTEX_CENTERED

    @property
    def is_cell_centered(self) -> bool:
        """Check if grid is cell-centered."""
        return self.grid_type == GridType.CELL_CENTERED


def apply_boundary_conditions_2d(
    field: NDArray[np.floating],
    boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
    domain_bounds: NDArray[np.floating] | None = None,
    time: float = 0.0,
    config: GhostCellConfig | None = None,
) -> NDArray[np.floating]:
    """
    Apply boundary conditions by padding array with ghost cells.

    Supports both uniform BCs (single BC type on all boundaries) and mixed BCs
    (different BC types on different boundary segments). The unified
    BoundaryConditions class handles both cases via the is_uniform/is_mixed
    properties.

    Args:
        field: Interior field of shape (Ny, Nx)
        boundary_conditions: BC specification (uniform or mixed)
        domain_bounds: Domain bounds array of shape (2, 2) where
                      bounds[i] = [min_i, max_i]. Required for mixed BCs.
        time: Current time for time-dependent BC values
        config: Ghost cell configuration (grid type, etc.)

    Returns:
        Padded field of shape (Ny+2, Nx+2) with ghost cells

    Raises:
        ValueError: If domain_bounds missing for mixed BC, or invalid input
        TypeError: If boundary_conditions is unknown type
    """
    # Input validation
    _validate_field_2d(field)

    if config is None:
        config = GhostCellConfig()

    # Handle legacy BoundaryConditions from bc_1d module
    if isinstance(boundary_conditions, LegacyBoundaryConditions1D):
        return _apply_legacy_uniform_bc_2d(field, boundary_conditions, config)

    # Use unified BoundaryConditions with is_uniform/is_mixed dispatch
    if not isinstance(boundary_conditions, BoundaryConditions):
        raise TypeError(f"Unsupported boundary condition type: {type(boundary_conditions)}")

    if boundary_conditions.is_uniform:
        return _apply_uniform_bc_2d(field, boundary_conditions, config)
    else:
        # Mixed BC - needs domain_bounds
        if domain_bounds is None and boundary_conditions.domain_bounds is None:
            raise ValueError("domain_bounds required for mixed boundary conditions")
        bounds = domain_bounds if domain_bounds is not None else boundary_conditions.domain_bounds
        _validate_domain_bounds_2d(bounds)
        return _apply_mixed_bc_2d(field, boundary_conditions, bounds, time, config)


def _validate_field_2d(field: NDArray[np.floating]) -> None:
    """Validate 2D field input."""
    if field.ndim != 2:
        raise ValueError(f"Expected 2D field, got {field.ndim}D")
    if field.size == 0:
        raise ValueError("Field cannot be empty")
    if not np.isfinite(field).all():
        raise ValueError("Field contains NaN or Inf values")


def _validate_domain_bounds_2d(bounds: NDArray[np.floating]) -> None:
    """Validate 2D domain bounds."""
    bounds = np.asarray(bounds)
    if bounds.shape != (2, 2):
        raise ValueError(f"Expected domain_bounds shape (2, 2), got {bounds.shape}")
    if not (bounds[:, 0] < bounds[:, 1]).all():
        raise ValueError("Domain bounds must have min < max for each axis")


def _apply_uniform_bc_2d(
    field: NDArray[np.floating],
    boundary_conditions: BoundaryConditions,
    config: GhostCellConfig,
) -> NDArray[np.floating]:
    """
    Apply uniform boundary conditions (same type on all boundaries).

    Uses the unified BoundaryConditions class with is_uniform=True.

    Args:
        field: Interior field (Ny, Nx)
        boundary_conditions: Uniform BC specification (unified class)
        config: Ghost cell configuration

    Returns:
        Padded field (Ny+2, Nx+2)
    """
    # Get BC type and value from the single segment
    seg = boundary_conditions.segments[0]
    bc_type = seg.bc_type

    if bc_type == BCType.PERIODIC:
        return np.pad(field, 1, mode="wrap")

    elif bc_type == BCType.DIRICHLET:
        g = seg.value if seg.value is not None else 0.0
        # Handle callable values (use center point as reference)
        if callable(g):
            g = 0.0  # For uniform BC, use constant 0 if callable

        if config.is_vertex_centered:
            return np.pad(field, 1, mode="constant", constant_values=g)
        else:
            padded = np.pad(field, 1, mode="constant", constant_values=0.0)
            padded[0, 1:-1] = 2 * g - field[0, :]
            padded[-1, 1:-1] = 2 * g - field[-1, :]
            padded[1:-1, 0] = 2 * g - field[:, 0]
            padded[1:-1, -1] = 2 * g - field[:, -1]
            padded[0, 0] = 2 * g - field[0, 0]
            padded[0, -1] = 2 * g - field[0, -1]
            padded[-1, 0] = 2 * g - field[-1, 0]
            padded[-1, -1] = 2 * g - field[-1, -1]
            return padded

    elif bc_type in [BCType.NO_FLUX, BCType.NEUMANN, BCType.REFLECTING]:
        return np.pad(field, 1, mode="edge")

    elif bc_type == BCType.ROBIN:
        alpha = seg.alpha if seg.alpha is not None else 1.0
        beta = seg.beta if seg.beta is not None else 0.0
        g = seg.value if seg.value is not None else 0.0
        if callable(g):
            g = 0.0

        if abs(beta) < 1e-14:
            # Pure Dirichlet
            g_eff = g / alpha if abs(alpha) > 1e-14 else 0.0
            padded = np.pad(field, 1, mode="constant", constant_values=0.0)
            padded[0, 1:-1] = 2 * g_eff - field[0, :]
            padded[-1, 1:-1] = 2 * g_eff - field[-1, :]
            padded[1:-1, 0] = 2 * g_eff - field[:, 0]
            padded[1:-1, -1] = 2 * g_eff - field[:, -1]
            padded[0, 0] = 2 * g_eff - field[0, 0]
            padded[0, -1] = 2 * g_eff - field[0, -1]
            padded[-1, 0] = 2 * g_eff - field[-1, 0]
            padded[-1, -1] = 2 * g_eff - field[-1, -1]
            return padded
        elif abs(alpha) < 1e-14:
            return np.pad(field, 1, mode="edge")
        else:
            # General Robin - use edge as approximation
            return np.pad(field, 1, mode="edge")

    else:
        raise ValueError(f"Unsupported boundary condition type: {bc_type}")


def _apply_legacy_uniform_bc_2d(
    field: NDArray[np.floating],
    boundary_conditions: LegacyBoundaryConditions1D,
    config: GhostCellConfig,
) -> NDArray[np.floating]:
    """
    Apply uniform boundary conditions from legacy bc_1d.BoundaryConditions.

    This handles backward compatibility with the old BoundaryConditions class.

    Args:
        field: Interior field (Ny, Nx)
        boundary_conditions: Legacy BC specification from bc_1d module
        config: Ghost cell configuration

    Returns:
        Padded field (Ny+2, Nx+2)
    """
    bc_type = boundary_conditions.type.lower()

    if bc_type == "periodic":
        return np.pad(field, 1, mode="wrap")

    elif bc_type == "dirichlet":
        g = boundary_conditions.left_value if boundary_conditions.left_value is not None else 0.0

        if config.is_vertex_centered:
            return np.pad(field, 1, mode="constant", constant_values=g)
        else:
            padded = np.pad(field, 1, mode="constant", constant_values=0.0)
            padded[0, 1:-1] = 2 * g - field[0, :]
            padded[-1, 1:-1] = 2 * g - field[-1, :]
            padded[1:-1, 0] = 2 * g - field[:, 0]
            padded[1:-1, -1] = 2 * g - field[:, -1]
            padded[0, 0] = 2 * g - field[0, 0]
            padded[0, -1] = 2 * g - field[0, -1]
            padded[-1, 0] = 2 * g - field[-1, 0]
            padded[-1, -1] = 2 * g - field[-1, -1]
            return padded

    elif bc_type in ["no_flux", "neumann"]:
        return np.pad(field, 1, mode="edge")

    elif bc_type == "robin":
        alpha = boundary_conditions.left_alpha if boundary_conditions.left_alpha is not None else 1.0
        beta = boundary_conditions.left_beta if boundary_conditions.left_beta is not None else 0.0
        g = boundary_conditions.left_value if boundary_conditions.left_value is not None else 0.0

        if abs(beta) < 1e-14:
            g_eff = g / alpha if abs(alpha) > 1e-14 else 0.0
            padded = np.pad(field, 1, mode="constant", constant_values=0.0)
            padded[0, 1:-1] = 2 * g_eff - field[0, :]
            padded[-1, 1:-1] = 2 * g_eff - field[-1, :]
            padded[1:-1, 0] = 2 * g_eff - field[:, 0]
            padded[1:-1, -1] = 2 * g_eff - field[:, -1]
            padded[0, 0] = 2 * g_eff - field[0, 0]
            padded[0, -1] = 2 * g_eff - field[0, -1]
            padded[-1, 0] = 2 * g_eff - field[-1, 0]
            padded[-1, -1] = 2 * g_eff - field[-1, -1]
            return padded
        elif abs(alpha) < 1e-14:
            return np.pad(field, 1, mode="edge")
        else:
            return np.pad(field, 1, mode="edge")

    else:
        raise ValueError(f"Unsupported boundary condition type: {bc_type}")


def _apply_mixed_bc_2d(
    field: NDArray[np.floating],
    mixed_bc: BoundaryConditions,
    domain_bounds: NDArray[np.floating],
    time: float,
    config: GhostCellConfig,
) -> NDArray[np.floating]:
    """
    Apply mixed boundary conditions (different types on different segments).

    Uses vectorized operations where possible for performance.

    Args:
        field: Interior field (Ny, Nx)
        mixed_bc: BC specification with is_mixed=True (unified BoundaryConditions)
        domain_bounds: Domain bounds array of shape (2, 2)
        time: Current time for time-dependent values
        config: Ghost cell configuration

    Returns:
        Padded field (Ny+2, Nx+2)
    """
    # Extract number of grid points from field shape (not intervals!)
    num_y_points, num_x_points = field.shape
    padded = np.zeros((num_y_points + 2, num_x_points + 2), dtype=field.dtype)

    # Copy interior
    padded[1:-1, 1:-1] = field

    # Compute grid coordinates and spacing
    x_min, x_max = domain_bounds[0]
    y_min, y_max = domain_bounds[1]
    dx = (x_max - x_min) / (num_x_points - 1) if num_x_points > 1 else 1.0
    dy = (y_max - y_min) / (num_y_points - 1) if num_y_points > 1 else 1.0

    x_coords = np.linspace(x_min, x_max, num_x_points)
    y_coords = np.linspace(y_min, y_max, num_y_points)

    # Ensure domain_bounds is set on mixed_bc
    if mixed_bc.domain_bounds is None:
        mixed_bc.domain_bounds = domain_bounds

    # Apply BCs to each boundary using vectorized operations where possible
    # Left boundary (x = x_min)
    _apply_boundary_ghost_cells(
        padded=padded,
        field=field,
        mixed_bc=mixed_bc,
        coords=y_coords,
        boundary_axis=0,  # x-axis
        boundary_side="min",
        boundary_id="x_min",
        grid_spacing=dx,
        time=time,
        config=config,
    )

    # Right boundary (x = x_max)
    _apply_boundary_ghost_cells(
        padded=padded,
        field=field,
        mixed_bc=mixed_bc,
        coords=y_coords,
        boundary_axis=0,
        boundary_side="max",
        boundary_id="x_max",
        grid_spacing=dx,
        time=time,
        config=config,
    )

    # Bottom boundary (y = y_min)
    _apply_boundary_ghost_cells(
        padded=padded,
        field=field,
        mixed_bc=mixed_bc,
        coords=x_coords,
        boundary_axis=1,  # y-axis
        boundary_side="min",
        boundary_id="y_min",
        grid_spacing=dy,
        time=time,
        config=config,
    )

    # Top boundary (y = y_max)
    _apply_boundary_ghost_cells(
        padded=padded,
        field=field,
        mixed_bc=mixed_bc,
        coords=x_coords,
        boundary_axis=1,
        boundary_side="max",
        boundary_id="y_max",
        grid_spacing=dy,
        time=time,
        config=config,
    )

    # Handle corners (average of adjacent boundary values)
    padded[0, 0] = 0.5 * (padded[0, 1] + padded[1, 0])
    padded[0, -1] = 0.5 * (padded[0, -2] + padded[1, -1])
    padded[-1, 0] = 0.5 * (padded[-1, 1] + padded[-2, 0])
    padded[-1, -1] = 0.5 * (padded[-1, -2] + padded[-2, -1])

    return padded


def _apply_boundary_ghost_cells(
    padded: NDArray[np.floating],
    field: NDArray[np.floating],
    mixed_bc: BoundaryConditions,
    coords: NDArray[np.floating],
    boundary_axis: int,
    boundary_side: str,
    boundary_id: str,
    grid_spacing: float,
    time: float,
    config: GhostCellConfig,
) -> None:
    """
    Apply ghost cells for one boundary.

    Args:
        padded: Padded array to modify (in place)
        field: Original interior field
        mixed_bc: BC specification (unified BoundaryConditions)
        coords: Coordinates along the boundary
        boundary_axis: 0 for x-boundary, 1 for y-boundary
        boundary_side: "min" or "max"
        boundary_id: Boundary identifier string
        grid_spacing: Grid spacing normal to boundary
        time: Current time
        config: Ghost cell configuration
    """
    _ny, _nx = field.shape
    n_points = len(coords)
    domain_bounds = mixed_bc.domain_bounds

    # Validate domain_bounds is available (required for rectangular domain BCs)
    if domain_bounds is None:
        raise ValueError(
            "domain_bounds is required for applying boundary conditions on structured grids. "
            "Ensure BoundaryConditions object was created with domain_bounds specified."
        )

    for i in range(n_points):
        # Construct point on boundary
        if boundary_axis == 0:  # x-boundary (left/right)
            x = domain_bounds[0, 0] if boundary_side == "min" else domain_bounds[0, 1]
            y = coords[i]
            point = np.array([x, y])
            # Interior value
            interior_val = field[i, 0] if boundary_side == "min" else field[i, -1]
            # Ghost cell indices in padded array
            if boundary_side == "min":
                ghost_idx = (i + 1, 0)
            else:
                ghost_idx = (i + 1, -1)
        else:  # y-boundary (bottom/top)
            x = coords[i]
            y = domain_bounds[1, 0] if boundary_side == "min" else domain_bounds[1, 1]
            point = np.array([x, y])
            # Interior value
            interior_val = field[0, i] if boundary_side == "min" else field[-1, i]
            # Ghost cell indices
            if boundary_side == "min":
                ghost_idx = (0, i + 1)
            else:
                ghost_idx = (-1, i + 1)

        # Get BC segment for this point
        bc_segment = mixed_bc.get_bc_at_point(point, boundary_id)

        # Compute ghost cell value
        ghost_val = _compute_ghost_value_enhanced(
            bc_segment=bc_segment,
            interior_val=interior_val,
            grid_spacing=grid_spacing,
            boundary_side=boundary_side,
            point=point,
            time=time,
            config=config,
        )

        padded[ghost_idx] = ghost_val


def _compute_ghost_value_enhanced(
    bc_segment: BCSegment,
    interior_val: float,
    grid_spacing: float,
    boundary_side: str,
    point: NDArray[np.floating],
    time: float,
    config: GhostCellConfig,
) -> float:
    """
    Compute ghost cell value with full BC support.

    Args:
        bc_segment: BC segment with type and value
        interior_val: Value at adjacent interior point
        grid_spacing: Grid spacing in normal direction (dx or dy)
        boundary_side: "min" or "max"
        point: Boundary point coordinates
        time: Current time
        config: Ghost cell configuration

    Returns:
        Ghost cell value
    """
    bc_type = bc_segment.bc_type

    # Evaluate BC value (may be callable)
    g = _evaluate_bc_value(bc_segment.value, point, time)

    # Outward normal sign: -1 for min boundary, +1 for max boundary
    # du/dn with outward normal: (u_exterior - u_interior) / dx for max boundary
    normal_sign = 1.0 if boundary_side == "max" else -1.0

    if bc_type == BCType.DIRICHLET:
        # u = g at boundary
        if config.is_vertex_centered:
            # Vertex-centered: boundary value is at grid point
            # Ghost cell mirrors the boundary value
            return g
        else:
            # Cell-centered: boundary at face, u_b = (u_g + u_i)/2 = g
            # => u_g = 2*g - u_i
            return 2.0 * g - interior_val

    elif bc_type in [BCType.NEUMANN, BCType.NO_FLUX, BCType.REFLECTING]:
        # du/dn = g at boundary (outward normal)
        # For min boundary: normal points inward (negative), so du/dn_outward = -du/dx
        #   (u_i - u_g) / (2*dx) = g  =>  u_g = u_i - 2*dx*g
        # For max boundary: normal points outward (positive), so du/dn_outward = +du/dx
        #   (u_g - u_i) / (2*dx) = g  =>  u_g = u_i + 2*dx*g
        return interior_val + normal_sign * 2.0 * grid_spacing * g

    elif bc_type == BCType.ROBIN:
        # alpha*u + beta*du/dn = g at boundary
        # Get Robin coefficients from segment
        alpha = getattr(bc_segment, "alpha", 1.0)
        beta = getattr(bc_segment, "beta", 0.0)

        # Handle degenerate cases
        if abs(beta) < 1e-14:
            # Pure Dirichlet: alpha*u = g => u = g/alpha
            g_eff = g / alpha if abs(alpha) > 1e-14 else 0.0
            return 2.0 * g_eff - interior_val
        elif abs(alpha) < 1e-14:
            # Pure Neumann: beta*du/dn = g => du/dn = g/beta
            g_eff = g / beta if abs(beta) > 1e-14 else 0.0
            return interior_val + normal_sign * 2.0 * grid_spacing * g_eff
        else:
            # General Robin:
            # alpha * (u_g + u_i)/2 + beta * normal_sign * (u_g - u_i)/(2*dx) = g
            # u_g * (alpha/2 + beta*normal_sign/(2*dx)) = g - u_i*(alpha/2 - beta*normal_sign/(2*dx))
            coeff_g = alpha / 2 + beta * normal_sign / (2 * grid_spacing)
            coeff_i = alpha / 2 - beta * normal_sign / (2 * grid_spacing)

            if abs(coeff_g) < 1e-14:
                # Degenerate case, use Neumann approximation
                return interior_val
            return (g - interior_val * coeff_i) / coeff_g

    elif bc_type == BCType.PERIODIC:
        # Periodic BC should not appear in mixed BC context
        # Return interior value as fallback (will be handled separately)
        return interior_val

    elif bc_type in [BCType.EXTRAPOLATION_LINEAR, BCType.EXTRAPOLATION_QUADRATIC]:
        # Extrapolation BCs require access to multiple interior points
        # This function only receives one interior value, so we fall back to Neumann
        # For proper extrapolation, use the specialized handling in apply_boundary_conditions_1d
        # or call ghost_cell_linear_extrapolation/ghost_cell_quadratic_extrapolation directly
        return interior_val  # Fallback: zero gradient

    else:
        # Unknown type - use Neumann (zero gradient) as safe default
        return interior_val


def _evaluate_bc_value(
    value: float | Callable,
    point: NDArray[np.floating],
    time: float,
) -> float:
    """
    Evaluate BC value, handling both constant and callable values.

    Supports multiple calling conventions for flexible BC specification:
    - value(point, time) - general convention (preferred)
    - value(x, y, t) for 2D / value(x, t) for 1D - coordinate convention
    - value(point) - time-independent convention

    Args:
        value: BC value (constant float or callable)
        point: Boundary point coordinates
        time: Current time

    Returns:
        Evaluated BC value
    """
    if callable(value):
        # Try different calling conventions
        # Convention 1 (preferred): value(point, time)
        try:
            return float(value(point, time))
        except TypeError:
            # Not compatible with (point, time) signature
            pass

        # Convention 2: value(x, y, t) for 2D / value(x, t) for 1D
        try:
            if len(point) == 2:
                return float(value(point[0], point[1], time))
            elif len(point) == 1:
                return float(value(point[0], time))
        except TypeError:
            # Not compatible with expanded coordinate signature
            pass

        # Convention 3: value(point) without time
        try:
            return float(value(point))
        except TypeError:
            # Not compatible with (point,) signature
            pass

        # Last resort: return 0
        logger.warning(f"Could not evaluate BC value function {value} with any known signature. returning 0.0")
        return 0.0
    else:
        return float(value)


# =============================================================================
# nD Mixed Boundary Conditions (generalization of 2D mixed BC)
# =============================================================================


def _get_boundary_name_nd(axis: int, side: str, dimension: int) -> str:
    """
    Get boundary name for axis and side in arbitrary dimension.

    Args:
        axis: Axis index (0, 1, 2, ...)
        side: "min" or "max"
        dimension: Total dimension

    Returns:
        Boundary name string (e.g., "x_min", "y_max", "dim3_min")
    """
    if dimension <= 3:
        axis_names = ["x", "y", "z"]
        return f"{axis_names[axis]}_{side}"
    else:
        # For d > 3, use generic naming
        if axis < 3:
            axis_names = ["x", "y", "z"]
            return f"{axis_names[axis]}_{side}"
        else:
            return f"dim{axis}_{side}"


def _apply_mixed_bc_nd(
    field: NDArray[np.floating],
    mixed_bc: BoundaryConditions,
    domain_bounds: NDArray[np.floating],
    time: float,
    config: GhostCellConfig,
) -> NDArray[np.floating]:
    """
    Apply mixed boundary conditions in n dimensions.

    Generalizes _apply_mixed_bc_2d to arbitrary dimension. Each of the 2d
    boundary faces (d axes × 2 sides) can have different BC types.

    Args:
        field: Interior field of shape (N_0, N_1, ..., N_{d-1})
        mixed_bc: BC specification with segments for different boundaries
        domain_bounds: Domain bounds array of shape (d, 2)
        time: Current time for time-dependent values
        config: Ghost cell configuration

    Returns:
        Padded field with shape (N_0+2, N_1+2, ..., N_{d-1}+2)
    """
    d = field.ndim
    shape = field.shape

    # Create padded array
    padded_shape = tuple(s + 2 for s in shape)
    padded = np.zeros(padded_shape, dtype=field.dtype)

    # Copy interior: padded[1:-1, 1:-1, ..., 1:-1] = field
    interior_slices = tuple(slice(1, -1) for _ in range(d))
    padded[interior_slices] = field

    # Compute grid spacing for each axis
    grid_spacings = []
    for axis in range(d):
        n_points = shape[axis]
        extent = domain_bounds[axis, 1] - domain_bounds[axis, 0]
        dx = extent / (n_points - 1) if n_points > 1 else extent
        grid_spacings.append(dx)

    # Ensure domain_bounds is set on mixed_bc
    if mixed_bc.domain_bounds is None:
        mixed_bc.domain_bounds = domain_bounds

    # Apply BC to each of the 2d boundary faces
    for axis in range(d):
        for side in ["min", "max"]:
            boundary_id = _get_boundary_name_nd(axis, side, d)
            _apply_boundary_face_nd(
                padded=padded,
                field=field,
                mixed_bc=mixed_bc,
                axis=axis,
                side=side,
                boundary_id=boundary_id,
                domain_bounds=domain_bounds,
                grid_spacing=grid_spacings[axis],
                time=time,
                config=config,
            )

    # Handle corners/edges (intersections of multiple boundaries)
    # Use averaging strategy: corner value = average of adjacent face ghost values
    _apply_corner_values_nd(padded, d)

    return padded


def _apply_boundary_face_nd(
    padded: NDArray[np.floating],
    field: NDArray[np.floating],
    mixed_bc: BoundaryConditions,
    axis: int,
    side: str,
    boundary_id: str,
    domain_bounds: NDArray[np.floating],
    grid_spacing: float,
    time: float,
    config: GhostCellConfig,
) -> None:
    """
    Apply ghost cells for one boundary face in nD.

    Args:
        padded: Padded array to modify (in place)
        field: Original interior field
        mixed_bc: BC specification
        axis: Axis normal to this boundary (0, 1, ..., d-1)
        side: "min" or "max"
        boundary_id: Boundary identifier string
        domain_bounds: Domain bounds array
        grid_spacing: Grid spacing normal to boundary
        time: Current time
        config: Ghost cell configuration
    """
    d = field.ndim
    shape = field.shape

    # Iterate over all points on this boundary face
    # Face has shape: (N_0, ..., N_{axis-1}, N_{axis+1}, ..., N_{d-1})
    face_shape = shape[:axis] + shape[axis + 1 :]

    if len(face_shape) == 0:
        # 1D case: single point on each boundary
        face_indices = [()]
    else:
        face_indices = np.ndindex(*face_shape)

    for face_idx in face_indices:
        # Convert face index to full field index
        if axis == 0:
            field_idx_list = list(face_idx)
        else:
            field_idx_list = list(face_idx[:axis]) + list(face_idx[axis:])

        # Insert axis index for interior point adjacent to boundary
        if side == "min":
            field_idx_list.insert(axis, 0)
        else:
            field_idx_list.insert(axis, shape[axis] - 1)

        field_idx = tuple(field_idx_list)

        # Get interior value
        interior_val = field[field_idx]

        # Construct point coordinates on boundary
        point = np.zeros(d)
        for i in range(d):
            if i == axis:
                point[i] = domain_bounds[i, 0] if side == "min" else domain_bounds[i, 1]
            else:
                # Compute coordinate from index
                idx_in_face = face_idx[i] if i < axis else face_idx[i - 1]
                n_i = shape[i]
                point[i] = domain_bounds[i, 0] + idx_in_face * (domain_bounds[i, 1] - domain_bounds[i, 0]) / (
                    n_i - 1 if n_i > 1 else 1
                )

        # Get BC segment for this point
        bc_segment = mixed_bc.get_bc_at_point(point, boundary_id)

        # Compute ghost cell value
        ghost_val = _compute_ghost_value_enhanced(
            bc_segment=bc_segment,
            interior_val=interior_val,
            grid_spacing=grid_spacing,
            boundary_side=side,
            point=point,
            time=time,
            config=config,
        )

        # Compute ghost cell index in padded array
        # padded has +1 offset in each dimension
        ghost_idx_list = [i + 1 for i in field_idx_list]
        if side == "min":
            ghost_idx_list[axis] = 0
        else:
            ghost_idx_list[axis] = padded.shape[axis] - 1

        ghost_idx = tuple(ghost_idx_list)
        padded[ghost_idx] = ghost_val


def _apply_corner_values_nd(padded: NDArray[np.floating], d: int) -> None:
    """
    Apply corner/edge values in nD using averaging.

    Intersections are processed in order of decreasing codimension:
    - First edges (2 boundary dims), using face ghost values
    - Then corners (3+ boundary dims), using edge ghost values

    This ordering ensures that each intersection can average values
    from already-filled lower-codimension neighbors.

    For d=2: 4 corners (0D intersections of 1D edges)
    For d=3: 12 edges (1D lines) + 8 corners (0D points)
    For d>3: generalized intersection handling

    Args:
        padded: Padded array to modify (in place)
        d: Dimension
    """
    padded_shape = padded.shape

    # Process intersections in order of boundary_count (from 2 to d)
    # This ensures edges are filled before corners, etc.
    for target_boundary_count in range(2, d + 1):
        # Iterate over all intersection configurations
        for corner_idx in np.ndindex(*([3] * d)):
            # corner_idx[i] in {0, 1, 2} maps to {low_boundary, interior, high_boundary}
            boundary_count = sum(1 for c in corner_idx if c != 1)

            if boundary_count != target_boundary_count:
                continue

            # Build slices for this intersection
            # - Boundary dims: single index (0 or -1)
            # - Interior dims: slice(1, -1) spanning interior
            intersection_slice = []
            for i, c in enumerate(corner_idx):
                if c == 0:
                    intersection_slice.append(0)
                elif c == 2:
                    intersection_slice.append(padded_shape[i] - 1)
                else:
                    intersection_slice.append(slice(1, -1))

            intersection_slice = tuple(intersection_slice)

            # Find neighbor slices (one boundary dim moved to interior)
            neighbor_values = []
            for i, c in enumerate(corner_idx):
                if c != 1:  # This dimension is at boundary
                    # Create neighbor slice with this dim moved to interior
                    neighbor_slice = list(intersection_slice)
                    if c == 0:
                        neighbor_slice[i] = 1  # Move from 0 to 1
                    else:
                        neighbor_slice[i] = padded_shape[i] - 2  # Move from -1 to -2
                    neighbor_values.append(padded[tuple(neighbor_slice)])

            if neighbor_values:
                # Average all neighbor values
                # For edges: this is a 1D array of values
                # For corners: this is a scalar
                avg_val = np.mean(neighbor_values, axis=0)
                padded[intersection_slice] = avg_val


def apply_boundary_conditions_nd(
    field: NDArray[np.floating],
    boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
    domain_bounds: NDArray[np.floating] | None = None,
    time: float = 0.0,
    config: GhostCellConfig | None = None,
) -> NDArray[np.floating]:
    """
    Apply boundary conditions in n dimensions.

    Args:
        field: Interior field of shape (N_1, N_2, ..., N_d)
        boundary_conditions: BC specification (unified or legacy)
        domain_bounds: Domain bounds array of shape (d, 2)
        time: Current time for time-dependent BCs
        config: Ghost cell configuration

    Returns:
        Padded field with ghost cells
    """
    if config is None:
        config = GhostCellConfig()

    d = field.ndim

    if d == 1:
        return _apply_bc_1d(field, boundary_conditions, domain_bounds, time, config)
    elif d == 2:
        return apply_boundary_conditions_2d(field, boundary_conditions, domain_bounds, time, config)
    else:
        # For d > 2, use uniform BC only
        # Handle legacy BoundaryConditions
        if isinstance(boundary_conditions, LegacyBoundaryConditions1D):
            bc_type = boundary_conditions.type.lower()
            if bc_type == "periodic":
                return np.pad(field, 1, mode="wrap")
            elif bc_type == "dirichlet":
                g = boundary_conditions.left_value if boundary_conditions.left_value is not None else 0.0
                padded = np.pad(field, 1, mode="constant", constant_values=0.0)
                for axis in range(d):
                    slices_low = [slice(1, -1)] * d
                    slices_low[axis] = slice(0, 1)
                    slices_int_low = [slice(1, -1)] * d
                    slices_int_low[axis] = slice(1, 2)
                    slices_high = [slice(1, -1)] * d
                    slices_high[axis] = slice(-1, None)
                    slices_int_high = [slice(1, -1)] * d
                    slices_int_high[axis] = slice(-2, -1)
                    padded[tuple(slices_low)] = 2 * g - padded[tuple(slices_int_low)]
                    padded[tuple(slices_high)] = 2 * g - padded[tuple(slices_int_high)]
                # Apply corner/edge values
                _apply_corner_values_nd(padded, d)
                return padded
            elif bc_type in ["no_flux", "neumann"]:
                return np.pad(field, 1, mode="edge")
            else:
                raise ValueError(f"Unsupported BC type for nD: {bc_type}")

        # Handle unified BoundaryConditions
        if not isinstance(boundary_conditions, BoundaryConditions):
            raise TypeError(f"Unsupported boundary condition type: {type(boundary_conditions)}")

        if boundary_conditions.is_uniform:
            seg = boundary_conditions.segments[0]
            bc_type = seg.bc_type
            if bc_type == BCType.PERIODIC:
                return np.pad(field, 1, mode="wrap")
            elif bc_type == BCType.DIRICHLET:
                g = seg.value if seg.value is not None else 0.0
                if callable(g):
                    g = 0.0
                padded = np.pad(field, 1, mode="constant", constant_values=0.0)
                for axis in range(d):
                    slices_low = [slice(1, -1)] * d
                    slices_low[axis] = slice(0, 1)
                    slices_int_low = [slice(1, -1)] * d
                    slices_int_low[axis] = slice(1, 2)
                    slices_high = [slice(1, -1)] * d
                    slices_high[axis] = slice(-1, None)
                    slices_int_high = [slice(1, -1)] * d
                    slices_int_high[axis] = slice(-2, -1)
                    padded[tuple(slices_low)] = 2 * g - padded[tuple(slices_int_low)]
                    padded[tuple(slices_high)] = 2 * g - padded[tuple(slices_int_high)]
                # Apply corner/edge values (edges first, then corners)
                _apply_corner_values_nd(padded, d)
                return padded
            elif bc_type in [BCType.NO_FLUX, BCType.NEUMANN, BCType.REFLECTING]:
                return np.pad(field, 1, mode="edge")
            elif bc_type == BCType.ROBIN:
                # Robin BC: α*u + β*du/dn = g
                alpha = seg.alpha if seg.alpha is not None else 1.0
                beta = seg.beta if seg.beta is not None else 0.0
                g = seg.value if seg.value is not None else 0.0
                if callable(g):
                    g = 0.0
                if abs(beta) < 1e-14:
                    # Reduces to Dirichlet: u = g/α
                    g_eff = g / alpha if abs(alpha) > 1e-14 else 0.0
                    padded = np.pad(field, 1, mode="constant", constant_values=0.0)
                    for axis in range(d):
                        slices_low = [slice(1, -1)] * d
                        slices_low[axis] = slice(0, 1)
                        slices_int_low = [slice(1, -1)] * d
                        slices_int_low[axis] = slice(1, 2)
                        slices_high = [slice(1, -1)] * d
                        slices_high[axis] = slice(-1, None)
                        slices_int_high = [slice(1, -1)] * d
                        slices_int_high[axis] = slice(-2, -1)
                        padded[tuple(slices_low)] = 2 * g_eff - padded[tuple(slices_int_low)]
                        padded[tuple(slices_high)] = 2 * g_eff - padded[tuple(slices_int_high)]
                    # Apply corner/edge values
                    _apply_corner_values_nd(padded, d)
                    return padded
                else:
                    # General Robin - use Neumann-like ghost cells
                    return np.pad(field, 1, mode="edge")
            else:
                raise ValueError(f"Unsupported BC type for nD: {bc_type}")
        else:
            # Mixed BC for d > 2
            return _apply_mixed_bc_nd(field, boundary_conditions, domain_bounds, time, config)


def _apply_bc_1d(
    field: NDArray[np.floating],
    boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
    domain_bounds: NDArray[np.floating] | None,
    time: float,
    config: GhostCellConfig,
) -> NDArray[np.floating]:
    """
    Apply boundary conditions in 1D.

    Args:
        field: Interior field of shape (Nx,)
        boundary_conditions: BC specification (unified or legacy)
        domain_bounds: Domain bounds array of shape (1, 2) or (2,)
        time: Current time
        config: Ghost cell configuration

    Returns:
        Padded field of shape (Nx+2,)
    """
    if not np.isfinite(field).all():
        raise ValueError("Field contains NaN or Inf values")

    # Handle legacy BoundaryConditions from bc_1d
    if isinstance(boundary_conditions, LegacyBoundaryConditions1D):
        bc_type = boundary_conditions.type.lower()
        if bc_type == "periodic":
            return np.pad(field, 1, mode="wrap")
        elif bc_type == "dirichlet":
            left_val = boundary_conditions.left_value if boundary_conditions.left_value is not None else 0.0
            right_val = boundary_conditions.right_value if boundary_conditions.right_value is not None else 0.0
            padded = np.zeros(len(field) + 2, dtype=field.dtype)
            padded[1:-1] = field
            padded[0] = 2.0 * left_val - field[0]
            padded[-1] = 2.0 * right_val - field[-1]
            return padded
        elif bc_type in ["no_flux", "neumann"]:
            return np.pad(field, 1, mode="edge")
        else:
            raise ValueError(f"Unsupported BC type: {bc_type}")

    # Handle unified BoundaryConditions
    if not isinstance(boundary_conditions, BoundaryConditions):
        raise TypeError(f"Unsupported boundary condition type: {type(boundary_conditions)}")

    if boundary_conditions.is_uniform:
        # Uniform BC - same type on both boundaries
        seg = boundary_conditions.segments[0]
        bc_type = seg.bc_type

        if bc_type == BCType.PERIODIC:
            return np.pad(field, 1, mode="wrap")
        elif bc_type == BCType.DIRICHLET:
            g = seg.value if seg.value is not None else 0.0
            if callable(g):
                g = 0.0
            padded = np.zeros(len(field) + 2, dtype=field.dtype)
            padded[1:-1] = field
            padded[0] = 2.0 * g - field[0]
            padded[-1] = 2.0 * g - field[-1]
            return padded
        elif bc_type in [BCType.NO_FLUX, BCType.NEUMANN, BCType.REFLECTING]:
            return np.pad(field, 1, mode="edge")
        else:
            raise ValueError(f"Unsupported BC type: {bc_type}")
    else:
        # Mixed BC for 1D
        if domain_bounds is None and boundary_conditions.domain_bounds is None:
            raise ValueError("domain_bounds required for mixed BC")

        bounds = domain_bounds if domain_bounds is not None else boundary_conditions.domain_bounds
        bounds = np.atleast_2d(bounds)
        x_min, x_max = bounds[0, 0], bounds[0, 1]

        if boundary_conditions.domain_bounds is None:
            boundary_conditions.domain_bounds = bounds

        padded = np.zeros(len(field) + 2, dtype=field.dtype)
        padded[1:-1] = field

        dx = (x_max - x_min) / (len(field) - 1) if len(field) > 1 else 1.0

        # Left BC
        point_left = np.array([x_min])
        bc_left = boundary_conditions.get_bc_at_point(point_left, "x_min")
        if bc_left.bc_type == BCType.EXTRAPOLATION_LINEAR and len(field) >= 2:
            # Linear extrapolation: ghost = 2*u_0 - u_1
            padded[0] = 2.0 * field[0] - field[1]
        elif bc_left.bc_type == BCType.EXTRAPOLATION_QUADRATIC and len(field) >= 3:
            # Quadratic extrapolation: ghost = 3*u_0 - 3*u_1 + u_2
            padded[0] = 3.0 * field[0] - 3.0 * field[1] + field[2]
        else:
            padded[0] = _compute_ghost_value_enhanced(bc_left, field[0], dx, "min", point_left, time, config)

        # Right BC
        point_right = np.array([x_max])
        bc_right = boundary_conditions.get_bc_at_point(point_right, "x_max")
        if bc_right.bc_type == BCType.EXTRAPOLATION_LINEAR and len(field) >= 2:
            # Linear extrapolation: ghost = 2*u_0 - u_1
            padded[-1] = 2.0 * field[-1] - field[-2]
        elif bc_right.bc_type == BCType.EXTRAPOLATION_QUADRATIC and len(field) >= 3:
            # Quadratic extrapolation: ghost = 3*u_0 - 3*u_1 + u_2
            padded[-1] = 3.0 * field[-1] - 3.0 * field[-2] + field[-3]
        else:
            padded[-1] = _compute_ghost_value_enhanced(bc_right, field[-1], dx, "max", point_right, time, config)

        return padded


def create_boundary_mask_2d(
    mixed_bc: BoundaryConditions,
    grid_shape: tuple[int, int],
    domain_bounds: NDArray[np.floating],
) -> dict[str, dict[str, NDArray[np.bool_]]]:
    """
    Pre-compute boundary masks for each BC segment.

    This optimization allows O(1) BC lookup per segment during solve,
    instead of O(segments) per boundary point.

    Args:
        mixed_bc: BC specification (unified BoundaryConditions)
        grid_shape: Grid shape (Ny, Nx)
        domain_bounds: Domain bounds

    Returns:
        Dictionary mapping segment names to boundary masks:
        {segment_name: {"left": mask, "right": mask, "bottom": mask, "top": mask}}
    """
    Ny, Nx = grid_shape
    x_min, x_max = domain_bounds[0]
    y_min, y_max = domain_bounds[1]

    x_coords = np.linspace(x_min, x_max, Nx)
    y_coords = np.linspace(y_min, y_max, Ny)

    # Ensure domain_bounds is set
    if mixed_bc.domain_bounds is None:
        mixed_bc.domain_bounds = domain_bounds

    masks: dict[str, dict[str, NDArray[np.bool_]]] = {}

    for segment in mixed_bc.segments:
        segment_masks = {
            "left": np.zeros(Ny, dtype=bool),
            "right": np.zeros(Ny, dtype=bool),
            "bottom": np.zeros(Nx, dtype=bool),
            "top": np.zeros(Nx, dtype=bool),
        }

        # Left boundary
        for j in range(Ny):
            point = np.array([x_min, y_coords[j]])
            bc = mixed_bc.get_bc_at_point(point, "x_min")
            segment_masks["left"][j] = bc.name == segment.name

        # Right boundary
        for j in range(Ny):
            point = np.array([x_max, y_coords[j]])
            bc = mixed_bc.get_bc_at_point(point, "x_max")
            segment_masks["right"][j] = bc.name == segment.name

        # Bottom boundary
        for i in range(Nx):
            point = np.array([x_coords[i], y_min])
            bc = mixed_bc.get_bc_at_point(point, "y_min")
            segment_masks["bottom"][i] = bc.name == segment.name

        # Top boundary
        for i in range(Nx):
            point = np.array([x_coords[i], y_max])
            bc = mixed_bc.get_bc_at_point(point, "y_max")
            segment_masks["top"][i] = bc.name == segment.name

        masks[segment.name] = segment_masks

    return masks


# =============================================================================
# Public 1D and 3D Functions
# =============================================================================


def apply_boundary_conditions_1d(
    field: NDArray[np.floating],
    boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
    domain_bounds: NDArray[np.floating] | None = None,
    time: float = 0.0,
    config: GhostCellConfig | None = None,
) -> NDArray[np.floating]:
    """
    Apply boundary conditions in 1D.

    Optimized implementation for 1D problems.

    Args:
        field: Interior field of shape (Nx,)
        boundary_conditions: BC specification (unified or legacy)
        domain_bounds: Domain bounds array of shape (1, 2) or (2,)
        time: Current time for time-dependent BC values
        config: Ghost cell configuration

    Returns:
        Padded field of shape (Nx+2,) with ghost cells
    """
    if config is None:
        config = GhostCellConfig()

    return _apply_bc_1d(field, boundary_conditions, domain_bounds, time, config)


def apply_boundary_conditions_3d(
    field: NDArray[np.floating],
    boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
    domain_bounds: NDArray[np.floating] | None = None,
    time: float = 0.0,
    config: GhostCellConfig | None = None,
) -> NDArray[np.floating]:
    """
    Apply boundary conditions in 3D.

    Optimized implementation for 3D problems.

    Args:
        field: Interior field of shape (Nz, Ny, Nx)
        boundary_conditions: BC specification (unified or legacy)
        domain_bounds: Domain bounds array of shape (3, 2)
        time: Current time for time-dependent BC values
        config: Ghost cell configuration

    Returns:
        Padded field of shape (Nz+2, Ny+2, Nx+2) with ghost cells
    """
    if config is None:
        config = GhostCellConfig()

    if field.ndim != 3:
        raise ValueError(f"Expected 3D field, got {field.ndim}D")

    # Currently delegates to nD implementation
    # TODO: Add optimized 3D implementation with face-specific handling
    return apply_boundary_conditions_nd(field, boundary_conditions, domain_bounds, time, config)


# =============================================================================
# Ghost Value Computation for HJB Upwind Schemes
# =============================================================================


def get_ghost_values_nd(
    field: NDArray[np.floating],
    boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
    spacing: tuple[float, ...] | NDArray[np.floating],
    config: GhostCellConfig | None = None,
    time: float = 0.0,
) -> dict[tuple[int, int], NDArray[np.floating]]:
    """
    Compute ghost values for each boundary without padding the array.

    This function is designed for HJB upwind schemes that need ghost values
    BEFORE computing the Hamiltonian. Unlike apply_boundary_conditions_nd
    which returns a padded array, this returns ghost values separately.

    For upwind schemes at boundary i=0:
    - If drift v > 0 (flow from left): need ghost value u[-1] for backward diff
    - If drift v < 0 (flow from right): use interior u[1] for forward diff

    The ghost values are derived from BC type:
    - Dirichlet: u_ghost = 2*g - u_interior (cell-centered)
    - Neumann/No-flux: u_ghost = u_interior (zero derivative)
    - Periodic: u_ghost = u_opposite_boundary

    Args:
        field: Interior field of shape (N_1, N_2, ..., N_d)
        boundary_conditions: BC specification (unified or legacy)
        spacing: Grid spacing for each dimension, tuple or array of length d
        config: Ghost cell configuration (grid type)
        time: Current time for time-dependent BC values (default: 0.0)

    Returns:
        Dictionary mapping (dimension, side) to ghost value arrays:
        - Key (d, 0): ghost values for left boundary of dimension d
        - Key (d, 1): ghost values for right boundary of dimension d
        Each ghost array has shape matching the boundary slice.

    Example:
        >>> u = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 field
        >>> bc = dirichlet_bc(dimension=2, value=0.0)
        >>> ghosts = get_ghost_values_nd(u, bc, spacing=(0.1, 0.1))
        >>> ghosts[(0, 0)]  # Ghost for left boundary of dim 0 (shape: (3,))
        >>> ghosts[(1, 1)]  # Ghost for right boundary of dim 1 (shape: (2,))
    """
    if config is None:
        config = GhostCellConfig()

    d = field.ndim
    spacing = np.asarray(spacing)
    if len(spacing) != d:
        raise ValueError(f"Spacing length {len(spacing)} != field dimension {d}")

    ghosts: dict[tuple[int, int], NDArray[np.floating]] = {}

    # Determine BC type
    bc_type, bc_value = _get_bc_type_and_value(boundary_conditions)

    for axis in range(d):
        dx = spacing[axis]

        # Get interior values adjacent to boundaries
        # Left boundary: interior value at index 0
        slices_left = [slice(None)] * d
        slices_left[axis] = 0
        u_int_left = field[tuple(slices_left)]

        # Right boundary: interior value at index -1
        slices_right = [slice(None)] * d
        slices_right[axis] = -1
        u_int_right = field[tuple(slices_right)]

        if bc_type == BCType.PERIODIC:
            # Periodic: ghost = value from opposite boundary
            ghosts[(axis, 0)] = u_int_right.copy()  # Left ghost = right interior
            ghosts[(axis, 1)] = u_int_left.copy()  # Right ghost = left interior

        elif bc_type == BCType.DIRICHLET:
            g = bc_value if bc_value is not None else 0.0
            if callable(g):
                # Time-varying BC: call with current time
                g = g(time)

            if config.is_vertex_centered:
                # Vertex-centered: boundary is at grid point
                ghosts[(axis, 0)] = np.full_like(u_int_left, g)
                ghosts[(axis, 1)] = np.full_like(u_int_right, g)
            else:
                # Cell-centered: u_ghost = 2*g - u_interior
                ghosts[(axis, 0)] = 2 * g - u_int_left
                ghosts[(axis, 1)] = 2 * g - u_int_right

        elif bc_type in [BCType.NO_FLUX, BCType.NEUMANN, BCType.REFLECTING]:
            # No-flux/Neumann: du/dn = 0 => u_ghost = u_interior
            # For general Neumann du/dn = g:
            # Left: u_ghost = u_int - 2*dx*g (inward normal)
            # Right: u_ghost = u_int + 2*dx*g (outward normal)
            g = bc_value if bc_value is not None else 0.0
            if callable(g):
                # Time-varying flux: call with current time
                g = g(time)

            ghosts[(axis, 0)] = u_int_left - 2 * dx * g
            ghosts[(axis, 1)] = u_int_right + 2 * dx * g

        elif bc_type == BCType.ROBIN:
            # Robin: alpha*u + beta*du/dn = g
            # For now, treat as Neumann-like
            ghosts[(axis, 0)] = u_int_left.copy()
            ghosts[(axis, 1)] = u_int_right.copy()

        else:
            # Default to Neumann-like (edge extension)
            ghosts[(axis, 0)] = u_int_left.copy()
            ghosts[(axis, 1)] = u_int_right.copy()

    return ghosts


def _get_bc_type_and_value(
    boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
) -> tuple[BCType, float | None]:
    """Extract BC type and value from unified or legacy BC specification."""
    if isinstance(boundary_conditions, LegacyBoundaryConditions1D):
        bc_type_str = boundary_conditions.type.lower()
        if bc_type_str == "periodic":
            return BCType.PERIODIC, None
        elif bc_type_str == "dirichlet":
            return BCType.DIRICHLET, boundary_conditions.left_value
        elif bc_type_str in ["no_flux", "neumann"]:
            return BCType.NO_FLUX, 0.0
        else:
            return BCType.NO_FLUX, 0.0

    if not isinstance(boundary_conditions, BoundaryConditions):
        raise TypeError(f"Unsupported BC type: {type(boundary_conditions)}")

    if boundary_conditions.is_uniform:
        seg = boundary_conditions.segments[0]
        return seg.bc_type, seg.value

    # Mixed BC - return default type (first segment)
    seg = boundary_conditions.segments[0]
    return seg.bc_type, seg.value


# =============================================================================
# Class-Based FDM Applicator
# =============================================================================


class FDMApplicator(BaseStructuredApplicator):
    """
    Finite Difference Method boundary condition applicator.

    Provides a class-based interface for applying ghost cell boundary conditions
    to structured grids. Supports 1D, 2D, 3D, and higher dimensions with
    dimension-specific optimizations.

    Inherits from BaseStructuredApplicator for consistent interface across
    all BC applicators.

    Usage:
        # Create applicator for 2D problems
        applicator = FDMApplicator(dimension=2)

        # Apply BCs to a field
        padded = applicator.apply(field, bc, dx)

        # Or use static methods directly
        padded = FDMApplicator.apply_1d(field, bc)
        padded = FDMApplicator.apply_2d(field, bc)
    """

    def __init__(
        self,
        dimension: int,
        grid_type: GridType | str = GridType.CELL_CENTERED,
    ):
        """
        Initialize FDM applicator.

        Args:
            dimension: Spatial dimension (1, 2, 3, or higher)
            grid_type: Grid type (GridType enum or string for backward compat)
        """
        # Handle both enum and string for backward compatibility
        if isinstance(grid_type, str):
            grid_type_enum = GridType.VERTEX_CENTERED if grid_type == "vertex_centered" else GridType.CELL_CENTERED
        else:
            grid_type_enum = grid_type

        super().__init__(dimension, grid_type_enum)
        self._config = GhostCellConfig(grid_type=grid_type_enum)

    @property
    def grid_type(self) -> GridType:
        """Grid type enum."""
        return self._config.grid_type

    @property
    def grid_type_str(self) -> str:
        """Grid type as string (for backward compatibility)."""
        return "vertex_centered" if self._config.is_vertex_centered else "cell_centered"

    def apply(
        self,
        field: NDArray[np.floating],
        boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
        grid_spacing: float | tuple[float, ...] | None = None,
        domain_bounds: NDArray[np.floating] | None = None,
        time: float = 0.0,
    ) -> NDArray[np.floating]:
        """
        Apply boundary conditions to a field.

        Automatically dispatches to dimension-specific implementation.

        Args:
            field: Interior field values
            boundary_conditions: BC specification
            grid_spacing: Grid spacing (not used for ghost cells, but kept for API consistency)
            domain_bounds: Domain bounds (required for mixed BCs)
            time: Current time for time-dependent BCs

        Returns:
            Padded field with ghost cells
        """
        if self._dimension == 1:
            return apply_boundary_conditions_1d(field, boundary_conditions, domain_bounds, time, self._config)
        elif self._dimension == 2:
            return apply_boundary_conditions_2d(field, boundary_conditions, domain_bounds, time, self._config)
        elif self._dimension == 3:
            return apply_boundary_conditions_3d(field, boundary_conditions, domain_bounds, time, self._config)
        else:
            return apply_boundary_conditions_nd(field, boundary_conditions, domain_bounds, time, self._config)

    @staticmethod
    def apply_1d(
        field: NDArray[np.floating],
        boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
        domain_bounds: NDArray[np.floating] | None = None,
        time: float = 0.0,
        config: GhostCellConfig | None = None,
    ) -> NDArray[np.floating]:
        """Static method for 1D BC application."""
        return apply_boundary_conditions_1d(field, boundary_conditions, domain_bounds, time, config)

    @staticmethod
    def apply_2d(
        field: NDArray[np.floating],
        boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
        domain_bounds: NDArray[np.floating] | None = None,
        time: float = 0.0,
        config: GhostCellConfig | None = None,
    ) -> NDArray[np.floating]:
        """Static method for 2D BC application."""
        return apply_boundary_conditions_2d(field, boundary_conditions, domain_bounds, time, config)

    @staticmethod
    def apply_3d(
        field: NDArray[np.floating],
        boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
        domain_bounds: NDArray[np.floating] | None = None,
        time: float = 0.0,
        config: GhostCellConfig | None = None,
    ) -> NDArray[np.floating]:
        """Static method for 3D BC application."""
        return apply_boundary_conditions_3d(field, boundary_conditions, domain_bounds, time, config)

    @staticmethod
    def apply_nd(
        field: NDArray[np.floating],
        boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
        domain_bounds: NDArray[np.floating] | None = None,
        time: float = 0.0,
        config: GhostCellConfig | None = None,
    ) -> NDArray[np.floating]:
        """Static method for nD BC application."""
        return apply_boundary_conditions_nd(field, boundary_conditions, domain_bounds, time, config)

    def enforce_values(
        self,
        field: NDArray[np.floating],
        boundary_conditions: BoundaryConditions,
        spacing: tuple[float, ...] | NDArray[np.floating],
        time: float = 0.0,
    ) -> NDArray[np.floating]:
        """
        Enforce boundary condition values on solution array (Issue #542).

        This method sets boundary values to satisfy Dirichlet/Neumann BC specifications.
        Complements ghost cell application (which enables derivative computation).

        **Distinction**:
        - `apply()`: Returns padded array with ghost cells for computing ∇u
        - `enforce_values()`: Sets boundary values to satisfy BC (u(boundary) = g)

        Args:
            field: Solution array to enforce BC on (shape: (N1, N2, ..., Nd))
            boundary_conditions: BC specification (must be BoundaryConditions, not legacy)
            spacing: Grid spacing for each dimension (needed for Neumann BC)
            time: Current time for time-dependent BC values

        Returns:
            Solution array with BC enforced at boundaries (modified in-place, also returned)

        Example:
            >>> applicator = FDMApplicator(dimension=2)
            >>> U = solver.solve_timestep(...)
            >>> U = applicator.enforce_values(U, bc, grid.spacing, time=0.5)
        """
        if not isinstance(boundary_conditions, BoundaryConditions):
            raise TypeError(
                f"enforce_values() requires BoundaryConditions, got {type(boundary_conditions).__name__}. "
                f"Legacy BC types should be converted first."
            )

        # Convert spacing to tuple if needed
        if isinstance(spacing, np.ndarray):
            spacing_tuple = tuple(spacing)
        else:
            spacing_tuple = spacing

        # Import BC types
        from mfg_pde.geometry.boundary.types import BCType

        # Enforce BC for each segment
        for segment in boundary_conditions.segments:
            # Parse boundary identifier to get dimension and side
            boundary_id = segment.boundary
            if boundary_id is None:
                continue

            # Map boundary string to (dimension, side)
            dim, side = self._parse_boundary_identifier(boundary_id)
            if dim is None:
                continue  # Unrecognized boundary format

            # Get BC value (time-dependent or constant)
            if callable(segment.value):
                bc_value = segment.value(time)
            else:
                bc_value = segment.value

            # Get grid spacing for this dimension
            h = spacing_tuple[dim] if dim < len(spacing_tuple) else 1.0

            # Enforce BC based on type
            if segment.bc_type == BCType.DIRICHLET:
                # Dirichlet: Set boundary values directly u(boundary) = g
                self._apply_dirichlet_enforcement(field, dim, side, bc_value)

            elif segment.bc_type == BCType.NEUMANN:
                # Neumann: Set boundary value to satisfy gradient constraint ∂u/∂n = g
                self._apply_neumann_enforcement(field, dim, side, bc_value, h)

        return field

    def _parse_boundary_identifier(self, boundary_id: str) -> tuple[int | None, str | None]:
        """
        Parse boundary identifier string to (dimension, side).

        Supports standard tensor-product grid boundary naming:
        - "x_min", "x_max" → (0, 'min'), (0, 'max')
        - "y_min", "y_max" → (1, 'min'), (1, 'max')
        - "z_min", "z_max" → (2, 'min'), (2, 'max')

        Args:
            boundary_id: Boundary identifier string

        Returns:
            (dimension_index, 'min' or 'max') or (None, None) if unrecognized
        """
        if not isinstance(boundary_id, str):
            return None, None

        # Parse format: "{axis}_{side}" e.g., "x_min", "y_max"
        parts = boundary_id.lower().split("_")
        if len(parts) != 2:
            return None, None

        axis, side = parts

        # Map axis to dimension index
        axis_map = {"x": 0, "y": 1, "z": 2}
        if axis not in axis_map:
            return None, None

        dim = axis_map[axis]

        # Validate side
        if side not in ["min", "max"]:
            return None, None

        return dim, side

    def _apply_dirichlet_enforcement(self, field: NDArray[np.floating], dim: int, side: str, value: float) -> None:
        """
        Apply Dirichlet BC enforcement to nD array along specified dimension and side.

        Sets field[boundary_slice] = value where boundary_slice selects the appropriate
        boundary face (e.g., field[0, :, :] for x_min in 3D).

        Args:
            field: Solution array (modified in-place)
            dim: Dimension index (0=x, 1=y, 2=z)
            side: 'min' or 'max'
            value: Dirichlet boundary value
        """
        ndim = field.ndim
        slices = [slice(None)] * ndim

        if side == "min":
            slices[dim] = 0
        else:  # max
            slices[dim] = -1

        field[tuple(slices)] = value

    def _apply_neumann_enforcement(
        self, field: NDArray[np.floating], dim: int, side: str, grad_value: float, h: float
    ) -> None:
        """
        Apply Neumann BC enforcement to nD array along specified dimension and side.

        For Neumann BC ∂u/∂n = g, set boundary value to satisfy gradient constraint:
        - min side: u[0] = u[1] - g*h  (forward difference)
        - max side: u[-1] = u[-2] + g*h  (backward difference)

        Args:
            field: Solution array (modified in-place)
            dim: Dimension index (0=x, 1=y, 2=z)
            side: 'min' or 'max'
            grad_value: Neumann BC gradient value
            h: Grid spacing in this dimension
        """
        ndim = field.ndim
        boundary_slices = [slice(None)] * ndim
        neighbor_slices = [slice(None)] * ndim

        if side == "min":
            # Left boundary: u[0, :, :] = u[1, :, :] - g*h
            boundary_slices[dim] = 0
            neighbor_slices[dim] = 1
            field[tuple(boundary_slices)] = field[tuple(neighbor_slices)] - grad_value * h
        else:  # max
            # Right boundary: u[-1, :, :] = u[-2, :, :] + g*h
            boundary_slices[dim] = -1
            neighbor_slices[dim] = -2
            field[tuple(boundary_slices)] = field[tuple(neighbor_slices)] + grad_value * h


# =============================================================================
# Ghost Buffer with Topology/Calculator Composition (Issue #516)
# =============================================================================


def bc_to_topology_calculator(
    bc: BoundaryConditions | LegacyBoundaryConditions1D,
    shape: tuple[int, ...],
    grid_type: GridType = GridType.CELL_CENTERED,
    use_zero_flux: bool = False,
    drift_velocity: float = 0.0,
    diffusion_coeff: float = 1.0,
) -> tuple[Topology, BoundaryCalculator | None]:
    """
    Convert BoundaryConditions to Topology + Calculator pair.

    This is the bridge between the legacy BoundaryConditions API and the new
    Topology/Calculator composition architecture (Issue #516).

    **For NO_FLUX BC type, two physics-based options:**
    - `use_zero_flux=False` (default): ZeroGradientCalculator (du/dn = 0)
    - `use_zero_flux=True`: ZeroFluxCalculator (J·n = 0, mass conservation)

    Args:
        bc: Boundary condition specification (unified or legacy)
        shape: Grid shape (interior points)
        grid_type: Grid type for ghost cell formulas
        use_zero_flux: If True, NO_FLUX BC uses ZeroFluxCalculator (J·n=0)
                      for mass conservation. Default False uses ZeroGradientCalculator.
        drift_velocity: For ZeroFluxCalculator, the normal drift component
        diffusion_coeff: For ZeroFluxCalculator, the diffusion coefficient D = σ²/2

    Returns:
        Tuple of (Topology, Calculator | None). Calculator is None for periodic.

    Examples:
        >>> from mfg_pde.geometry.boundary import dirichlet_bc, no_flux_bc
        >>> bc = dirichlet_bc(0.0, dimension=2)
        >>> topology, calc = bc_to_topology_calculator(bc, shape=(100, 100))
        >>> type(calc).__name__
        'DirichletCalculator'

        >>> # Zero gradient (default for HJB)
        >>> bc = no_flux_bc(dimension=2)
        >>> topo, calc = bc_to_topology_calculator(bc, (100, 100))
        >>> type(calc).__name__
        'ZeroGradientCalculator'

        >>> # Zero flux (for FP mass conservation)
        >>> topo, calc = bc_to_topology_calculator(bc, (100, 100), use_zero_flux=True)
        >>> type(calc).__name__
        'ZeroFluxCalculator'
    """
    dimension = len(shape)

    def _get_no_flux_calculator() -> BoundaryCalculator:
        """Select calculator based on physics requirement."""
        if use_zero_flux:
            return ZeroFluxCalculator(drift_velocity, diffusion_coeff, grid_type)
        else:
            return ZeroGradientCalculator(grid_type)

    # Handle legacy BoundaryConditions from bc_1d module
    if isinstance(bc, LegacyBoundaryConditions1D):
        bc_type_str = bc.type.lower()
        if bc_type_str == "periodic":
            return PeriodicTopology(dimension, shape), None
        elif bc_type_str == "dirichlet":
            value = bc.left_value if bc.left_value is not None else 0.0
            return BoundedTopology(dimension, shape), DirichletCalculator(value, grid_type)
        elif bc_type_str in ("neumann", "no_flux"):
            return BoundedTopology(dimension, shape), _get_no_flux_calculator()
        else:
            # Default to zero gradient
            return BoundedTopology(dimension, shape), _get_no_flux_calculator()

    # Handle unified BoundaryConditions
    if not isinstance(bc, BoundaryConditions):
        raise TypeError(f"Unsupported BC type: {type(bc)}")

    # Only uniform BCs supported for now (mixed BC requires per-point dispatch)
    if not bc.is_uniform:
        raise NotImplementedError(
            "Mixed BCs not yet supported in Topology/Calculator architecture. "
            "Use PreallocatedGhostBuffer for mixed BCs."
        )

    seg = bc.segments[0]
    bc_type = seg.bc_type

    if bc_type == BCType.PERIODIC:
        return PeriodicTopology(dimension, shape), None
    elif bc_type == BCType.DIRICHLET:
        value = seg.value if not callable(seg.value) else 0.0
        return BoundedTopology(dimension, shape), DirichletCalculator(float(value), grid_type)
    elif bc_type == BCType.NEUMANN:
        flux = seg.value if not callable(seg.value) else 0.0
        return BoundedTopology(dimension, shape), NeumannCalculator(float(flux), grid_type)
    elif bc_type == BCType.NO_FLUX:
        return BoundedTopology(dimension, shape), _get_no_flux_calculator()
    elif bc_type == BCType.ROBIN:
        alpha = seg.alpha if hasattr(seg, "alpha") else 1.0
        beta = seg.beta if hasattr(seg, "beta") else 0.0
        rhs = seg.value if not callable(seg.value) else 0.0
        return BoundedTopology(dimension, shape), RobinCalculator(alpha, beta, float(rhs), grid_type)
    elif bc_type == BCType.EXTRAPOLATION_LINEAR:
        return BoundedTopology(dimension, shape), LinearExtrapolationCalculator()
    elif bc_type == BCType.EXTRAPOLATION_QUADRATIC:
        return BoundedTopology(dimension, shape), QuadraticExtrapolationCalculator()
    elif bc_type == BCType.REFLECTING:
        # Reflecting BC for particles - use zero gradient for field solvers
        return BoundedTopology(dimension, shape), _get_no_flux_calculator()
    else:
        # Unknown type - default to zero gradient
        return BoundedTopology(dimension, shape), _get_no_flux_calculator()


def create_ghost_buffer_from_bc(
    bc: BoundaryConditions | LegacyBoundaryConditions1D,
    shape: tuple[int, ...],
    dx: float | tuple[float, ...],
    ghost_depth: int = 1,
    grid_type: GridType = GridType.CELL_CENTERED,
    dtype: type = np.float64,
    use_zero_flux: bool = False,
    drift_velocity: float = 0.0,
    diffusion_coeff: float = 1.0,
) -> GhostBuffer:
    """
    Factory function to create GhostBuffer from BoundaryConditions.

    This is the primary entry point for using the new Topology/Calculator
    architecture with existing BoundaryConditions specifications.

    Args:
        bc: Boundary condition specification
        shape: Interior grid shape
        dx: Grid spacing (scalar or per-dimension tuple)
        ghost_depth: Number of ghost cell layers
        grid_type: Grid type for ghost cell formulas
        dtype: Data type for buffer
        use_zero_flux: If True, NO_FLUX BC uses ZeroFluxCalculator (J·n=0)
                      for mass conservation. Default False uses ZeroGradientCalculator.
        drift_velocity: For ZeroFluxCalculator, the normal drift component
        diffusion_coeff: For ZeroFluxCalculator, the diffusion coefficient D = σ²/2

    Returns:
        GhostBuffer configured with appropriate Topology and Calculator

    Examples:
        >>> from mfg_pde.geometry.boundary import dirichlet_bc, no_flux_bc
        >>> # Default (zero gradient)
        >>> bc = dirichlet_bc(0.0, dimension=2)
        >>> buffer = create_ghost_buffer_from_bc(bc, shape=(100, 100), dx=0.01)

        >>> # Mass-conserving (zero total flux)
        >>> bc = no_flux_bc(dimension=2)
        >>> buffer = create_ghost_buffer_from_bc(
        ...     bc, shape=(100, 100), dx=0.01,
        ...     use_zero_flux=True, diffusion_coeff=0.5
        ... )
    """
    topology, calculator = bc_to_topology_calculator(
        bc, shape, grid_type, use_zero_flux, drift_velocity, diffusion_coeff
    )
    return GhostBuffer(topology, calculator, dx, ghost_depth, dtype)


class GhostBuffer:
    """
    Ghost cell buffer using Topology/Calculator composition.

    This is the structural foundation for the 2-layer BC architecture:
    - Layer 1: Topology (Memory/Indexing) - periodic vs bounded connectivity
    - Layer 2: Calculator (Physics Strategy) - ghost value computation

    The separation enables:
    - Same Calculator works with different grid topologies
    - Same Topology works with different physics (Dirichlet, Neumann, etc.)
    - Clean interface for FVM/FEM/Graph extensions

    Usage:
        >>> from mfg_pde.geometry.boundary.applicator_base import (
        ...     BoundedTopology, DirichletCalculator, PeriodicTopology
        ... )

        >>> # Bounded domain with Dirichlet BC
        >>> topology = BoundedTopology(dimension=2, shape=(100, 100))
        >>> calculator = DirichletCalculator(boundary_value=0.0)
        >>> buffer = GhostBuffer(topology, calculator, dx=0.1)

        >>> # Set interior values
        >>> buffer.interior[:] = initial_condition

        >>> # Update ghosts (in-place, zero allocation)
        >>> buffer.update()

        >>> # Periodic domain (no calculator needed)
        >>> topology = PeriodicTopology(dimension=2, shape=(100, 100))
        >>> buffer = GhostBuffer(topology)  # Calculator ignored for periodic

    See Also:
        - PreallocatedGhostBuffer: Legacy interface with BoundaryConditions
        - Issue #516: Topology/Calculator composition design
    """

    def __init__(
        self,
        topology: Topology,
        calculator: BoundaryCalculator | None = None,
        dx: float | tuple[float, ...] = 1.0,
        ghost_depth: int = 1,
        dtype: type = np.float64,
    ):
        """
        Initialize ghost buffer with topology and calculator.

        Args:
            topology: Grid topology (PeriodicTopology or BoundedTopology)
            calculator: Ghost value calculator (required for bounded, ignored for periodic)
            dx: Grid spacing (scalar or tuple for each dimension)
            ghost_depth: Number of ghost cells per boundary
            dtype: Data type for the buffer

        Raises:
            ValueError: If bounded topology provided without calculator
        """

        self._topology = topology
        self._calculator = calculator
        self._ghost_depth = ghost_depth
        self._dtype = dtype

        # Validate: bounded topology requires calculator
        if not topology.is_periodic and calculator is None:
            raise ValueError(
                "Bounded topology requires a BoundaryCalculator. "
                "Use PeriodicTopology for periodic boundaries, or provide a calculator."
            )

        # Parse grid spacing
        if isinstance(dx, (int, float)):
            self._dx = tuple([float(dx)] * topology.dimension)
        else:
            self._dx = tuple(float(d) for d in dx)

        if len(self._dx) != topology.dimension:
            raise ValueError(
                f"Grid spacing dimension {len(self._dx)} must match topology dimension {topology.dimension}"
            )

        # Compute shapes
        self._interior_shape = topology.shape
        self._dimension = topology.dimension
        self._padded_shape = tuple(s + 2 * ghost_depth for s in self._interior_shape)

        # Allocate buffer
        self._buffer = np.zeros(self._padded_shape, dtype=dtype)

        # Create interior slice (view, not copy)
        self._interior_slices = tuple(slice(ghost_depth, -ghost_depth) for _ in range(self._dimension))

    @property
    def topology(self) -> Topology:
        """Grid topology."""
        return self._topology

    @property
    def calculator(self) -> BoundaryCalculator | None:
        """Ghost value calculator (None for periodic topology)."""
        return self._calculator

    @property
    def padded(self) -> NDArray[np.floating]:
        """Full padded buffer including ghost cells."""
        return self._buffer

    @property
    def interior(self) -> NDArray[np.floating]:
        """View of interior region (no copy)."""
        return self._buffer[self._interior_slices]

    @property
    def shape(self) -> tuple[int, ...]:
        """Interior shape."""
        return self._interior_shape

    @property
    def padded_shape(self) -> tuple[int, ...]:
        """Padded shape including ghosts."""
        return self._padded_shape

    @property
    def ghost_depth(self) -> int:
        """Number of ghost cells per side."""
        return self._ghost_depth

    @property
    def dx(self) -> tuple[float, ...]:
        """Grid spacing for each dimension."""
        return self._dx

    def update(self, **kwargs) -> None:
        """
        Update ghost cells in-place based on current interior values.

        This is the core zero-allocation operation. The update strategy
        depends on topology type:
        - Periodic: ghost = wrap-around (opposite boundary interior)
        - Bounded: ghost = calculator.compute(interior, dx, side)

        Args:
            **kwargs: Additional arguments passed to calculator.compute()
                     (e.g., time for time-dependent BCs, drift for FP no-flux)
        """
        if self._topology.is_periodic:
            self._update_periodic()
        else:
            self._update_bounded(**kwargs)

    def _update_periodic(self) -> None:
        """Update ghost cells for periodic topology (wrap-around)."""
        d = self._dimension
        g = self._ghost_depth
        buf = self._buffer

        for axis in range(d):
            # Low ghost = high interior
            lo_ghost = [slice(None)] * d
            lo_ghost[axis] = slice(0, g)
            hi_interior = [slice(None)] * d
            hi_interior[axis] = slice(-2 * g, -g)
            buf[tuple(lo_ghost)] = buf[tuple(hi_interior)]

            # High ghost = low interior
            hi_ghost = [slice(None)] * d
            hi_ghost[axis] = slice(-g, None)
            lo_interior = [slice(None)] * d
            lo_interior[axis] = slice(g, 2 * g)
            buf[tuple(hi_ghost)] = buf[tuple(lo_interior)]

    def _update_bounded(self, **kwargs) -> None:
        """Update ghost cells for bounded topology using calculator (vectorized)."""
        if self._calculator is None:
            raise RuntimeError("Calculator is None for bounded topology")

        d = self._dimension
        g = self._ghost_depth
        buf = self._buffer

        for axis in range(d):
            dx = self._dx[axis]

            # Get interior and ghost slices
            lo_ghost = [slice(None)] * d
            lo_ghost[axis] = slice(0, g)
            lo_interior = [slice(None)] * d
            lo_interior[axis] = slice(g, 2 * g)

            hi_ghost = [slice(None)] * d
            hi_ghost[axis] = slice(-g, None)
            hi_interior = [slice(None)] * d
            hi_interior[axis] = slice(-2 * g, -g)

            # Get interior arrays (views, not copies)
            interior_lo = buf[tuple(lo_interior)]
            interior_hi = buf[tuple(hi_interior)]

            # VECTORIZED: Apply calculator to entire boundary array at once
            # All Calculator implementations support NDArray via NumPy broadcasting
            ghost_lo = self._calculator.compute(
                interior_value=interior_lo,
                dx=dx,
                side="min",
                **kwargs,
            )
            ghost_hi = self._calculator.compute(
                interior_value=interior_hi,
                dx=dx,
                side="max",
                **kwargs,
            )

            buf[tuple(lo_ghost)] = ghost_lo
            buf[tuple(hi_ghost)] = ghost_hi

    def reset(self, fill_value: float = 0.0) -> None:
        """Reset buffer to a constant value."""
        self._buffer.fill(fill_value)

    def copy_to_interior(self, data: NDArray[np.floating]) -> None:
        """Copy data to interior region."""
        self.interior[:] = data

    def __repr__(self) -> str:
        calc_str = repr(self._calculator) if self._calculator else "None"
        return (
            f"GhostBuffer(topology={self._topology!r}, calculator={calc_str}, "
            f"shape={self._interior_shape}, ghost_depth={self._ghost_depth})"
        )


# Type hint imports for forward references
if TYPE_CHECKING:
    from .applicator_base import BoundaryCalculator, Topology


# =============================================================================
# Pre-allocated Ghost Cell Buffer (Legacy Interface - Zero-Copy Design)
# =============================================================================


class PreallocatedGhostBuffer:
    """
    Pre-allocated buffer for zero-copy ghost cell boundary conditions.

    This class avoids repeated memory allocation when applying BCs in tight loops
    (e.g., time-stepping). It pre-allocates a padded buffer and provides a view
    into the interior region for solvers to work with directly.

    Memory Layout:
        padded[ghost, interior..., ghost] where interior is a view (no copy)

    Usage:
        # Create buffer once
        buffer = PreallocatedGhostBuffer(
            interior_shape=(100, 100),
            boundary_conditions=bc,
            domain_bounds=bounds,
        )

        # In time loop - zero allocation:
        for t in range(num_steps):
            # Work directly on interior view
            buffer.interior[:] = solve_step(buffer.interior)

            # Update ghost cells in-place
            buffer.update_ghosts(time=t * dt)

            # Access full padded array if needed
            laplacian = compute_laplacian(buffer.padded)

    Performance:
        - Initial allocation: O(N) once
        - update_ghosts(): O(boundary) per call, no allocation
        - Interior access: O(1), view only
    """

    def __init__(
        self,
        interior_shape: tuple[int, ...],
        boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
        domain_bounds: NDArray[np.floating] | None = None,
        dtype: np.dtype = np.float64,
        ghost_depth: int = 1,
        config: GhostCellConfig | None = None,
    ):
        """
        Initialize pre-allocated ghost buffer.

        Args:
            interior_shape: Shape of the interior domain (Ny, Nx) for 2D, etc.
            boundary_conditions: BC specification
            domain_bounds: Domain bounds for mixed BCs
            dtype: Data type for the buffer
            ghost_depth: Number of ghost cells per side (default: 1)
            config: Ghost cell configuration
        """
        self._interior_shape = interior_shape
        self._dimension = len(interior_shape)
        self._ghost_depth = ghost_depth
        self._boundary_conditions = boundary_conditions
        self._domain_bounds = domain_bounds
        self._config = config if config is not None else GhostCellConfig()

        # Compute padded shape
        self._padded_shape = tuple(s + 2 * ghost_depth for s in interior_shape)

        # Allocate the padded buffer
        self._buffer = np.zeros(self._padded_shape, dtype=dtype)

        # Create interior slice (view, not copy)
        self._interior_slices = tuple(slice(ghost_depth, -ghost_depth) for _ in range(self._dimension))

        # Pre-compute grid spacing if domain_bounds provided
        self._grid_spacing: tuple[float, ...] | None = None
        if domain_bounds is not None:
            domain_bounds = np.atleast_2d(domain_bounds)
            spacing = []
            for d in range(self._dimension):
                extent = domain_bounds[d, 1] - domain_bounds[d, 0]
                n_points = interior_shape[d]
                spacing.append(extent / (n_points - 1) if n_points > 1 else extent)
            self._grid_spacing = tuple(spacing)

    @property
    def padded(self) -> NDArray[np.floating]:
        """Full padded buffer including ghost cells."""
        return self._buffer

    @property
    def interior(self) -> NDArray[np.floating]:
        """View of interior region (no copy)."""
        return self._buffer[self._interior_slices]

    @property
    def shape(self) -> tuple[int, ...]:
        """Interior shape."""
        return self._interior_shape

    @property
    def padded_shape(self) -> tuple[int, ...]:
        """Padded shape including ghosts."""
        return self._padded_shape

    @property
    def ghost_depth(self) -> int:
        """Number of ghost cells per side."""
        return self._ghost_depth

    def update_ghosts(self, time: float = 0.0) -> None:
        """
        Update ghost cells in-place based on current interior values.

        This is the core zero-allocation operation. Ghost cells are computed
        from the interior values according to the boundary conditions.

        Args:
            time: Current time for time-dependent BCs
        """
        bc = self._boundary_conditions

        # Get BC type for dispatch
        if isinstance(bc, LegacyBoundaryConditions1D):
            bc_type_str = bc.type.lower()
            self._update_ghosts_legacy(bc_type_str)
            return

        if not isinstance(bc, BoundaryConditions):
            raise TypeError(f"Unsupported BC type: {type(bc)}")

        if bc.is_uniform:
            seg = bc.segments[0]
            self._update_ghosts_uniform(seg.bc_type, seg.value, time)
        else:
            # Mixed BC requires more complex per-point updates
            self._update_ghosts_mixed(bc, time)

    def _update_ghosts_uniform(
        self,
        bc_type: BCType,
        value: float | None,
        time: float,
    ) -> None:
        """Update ghost cells for uniform BC (same type on all boundaries)."""
        d = self._dimension
        g = self._ghost_depth
        buf = self._buffer

        # Evaluate value if callable
        if callable(value):
            v = value(time)
        else:
            v = value if value is not None else 0.0

        if bc_type == BCType.PERIODIC:
            # Periodic: ghost = opposite interior
            for axis in range(d):
                # Low ghost = high interior
                lo_ghost = [slice(None)] * d
                lo_ghost[axis] = slice(0, g)
                hi_interior = [slice(None)] * d
                hi_interior[axis] = slice(-2 * g, -g)
                buf[tuple(lo_ghost)] = buf[tuple(hi_interior)]

                # High ghost = low interior
                hi_ghost = [slice(None)] * d
                hi_ghost[axis] = slice(-g, None)
                lo_interior = [slice(None)] * d
                lo_interior[axis] = slice(g, 2 * g)
                buf[tuple(hi_ghost)] = buf[tuple(lo_interior)]

        elif bc_type == BCType.DIRICHLET:
            # Dirichlet: u_ghost = 2*g - u_interior
            for axis in range(d):
                # Low boundary
                lo_ghost = [slice(None)] * d
                lo_ghost[axis] = slice(0, g)
                lo_interior = [slice(None)] * d
                lo_interior[axis] = slice(g, 2 * g)
                buf[tuple(lo_ghost)] = 2 * v - buf[tuple(lo_interior)]

                # High boundary
                hi_ghost = [slice(None)] * d
                hi_ghost[axis] = slice(-g, None)
                hi_interior = [slice(None)] * d
                hi_interior[axis] = slice(-2 * g, -g)
                buf[tuple(hi_ghost)] = 2 * v - buf[tuple(hi_interior)]

        elif bc_type in [BCType.NO_FLUX, BCType.NEUMANN, BCType.REFLECTING]:
            # No-flux: u_ghost = u_interior (or with flux for general Neumann)
            for axis in range(d):
                # Low boundary
                lo_ghost = [slice(None)] * d
                lo_ghost[axis] = slice(0, g)
                lo_interior = [slice(None)] * d
                lo_interior[axis] = slice(g, 2 * g)
                buf[tuple(lo_ghost)] = buf[tuple(lo_interior)]

                # High boundary
                hi_ghost = [slice(None)] * d
                hi_ghost[axis] = slice(-g, None)
                hi_interior = [slice(None)] * d
                hi_interior[axis] = slice(-2 * g, -g)
                buf[tuple(hi_ghost)] = buf[tuple(hi_interior)]

        elif bc_type == BCType.ROBIN:
            # Robin: treat as Neumann-like for now
            for axis in range(d):
                lo_ghost = [slice(None)] * d
                lo_ghost[axis] = slice(0, g)
                lo_interior = [slice(None)] * d
                lo_interior[axis] = slice(g, 2 * g)
                buf[tuple(lo_ghost)] = buf[tuple(lo_interior)]

                hi_ghost = [slice(None)] * d
                hi_ghost[axis] = slice(-g, None)
                hi_interior = [slice(None)] * d
                hi_interior[axis] = slice(-2 * g, -g)
                buf[tuple(hi_ghost)] = buf[tuple(hi_interior)]

    def _update_ghosts_legacy(self, bc_type_str: str) -> None:
        """Update ghost cells for legacy BC type."""
        d = self._dimension
        g = self._ghost_depth
        buf = self._buffer

        if bc_type_str == "periodic":
            for axis in range(d):
                lo_ghost = [slice(None)] * d
                lo_ghost[axis] = slice(0, g)
                hi_interior = [slice(None)] * d
                hi_interior[axis] = slice(-2 * g, -g)
                buf[tuple(lo_ghost)] = buf[tuple(hi_interior)]

                hi_ghost = [slice(None)] * d
                hi_ghost[axis] = slice(-g, None)
                lo_interior = [slice(None)] * d
                lo_interior[axis] = slice(g, 2 * g)
                buf[tuple(hi_ghost)] = buf[tuple(lo_interior)]

        elif bc_type_str in ["no_flux", "neumann"]:
            for axis in range(d):
                lo_ghost = [slice(None)] * d
                lo_ghost[axis] = slice(0, g)
                lo_interior = [slice(None)] * d
                lo_interior[axis] = slice(g, 2 * g)
                buf[tuple(lo_ghost)] = buf[tuple(lo_interior)]

                hi_ghost = [slice(None)] * d
                hi_ghost[axis] = slice(-g, None)
                hi_interior = [slice(None)] * d
                hi_interior[axis] = slice(-2 * g, -g)
                buf[tuple(hi_ghost)] = buf[tuple(hi_interior)]

        elif bc_type_str == "dirichlet":
            bc = self._boundary_conditions
            v = bc.left_value if hasattr(bc, "left_value") and bc.left_value is not None else 0.0
            for axis in range(d):
                lo_ghost = [slice(None)] * d
                lo_ghost[axis] = slice(0, g)
                lo_interior = [slice(None)] * d
                lo_interior[axis] = slice(g, 2 * g)
                buf[tuple(lo_ghost)] = 2 * v - buf[tuple(lo_interior)]

                hi_ghost = [slice(None)] * d
                hi_ghost[axis] = slice(-g, None)
                hi_interior = [slice(None)] * d
                hi_interior[axis] = slice(-2 * g, -g)
                buf[tuple(hi_ghost)] = 2 * v - buf[tuple(hi_interior)]

    def _update_ghosts_mixed(self, bc: BoundaryConditions, time: float) -> None:
        """Update ghost cells for mixed BCs (different types on different boundaries)."""
        # For mixed BCs, we need to iterate per-point on boundaries
        # This is slower but necessary for accurate BC application
        # Fall back to creating a padded array and copying ghost values
        padded = apply_boundary_conditions_nd(
            self.interior.copy(),
            bc,
            self._domain_bounds,
            time,
            self._config,
        )

        # Copy ghost cells from padded to buffer
        d = self._dimension
        g = self._ghost_depth

        for axis in range(d):
            # Low ghost
            lo_ghost = [slice(None)] * d
            lo_ghost[axis] = slice(0, g)
            self._buffer[tuple(lo_ghost)] = padded[tuple(lo_ghost)]

            # High ghost
            hi_ghost = [slice(None)] * d
            hi_ghost[axis] = slice(-g, None)
            self._buffer[tuple(hi_ghost)] = padded[tuple(hi_ghost)]

    def reset(self, fill_value: float = 0.0) -> None:
        """Reset buffer to a constant value."""
        self._buffer.fill(fill_value)

    def copy_to_interior(self, data: NDArray[np.floating]) -> None:
        """Copy data to interior region."""
        self.interior[:] = data

    def copy_from_padded(self, data: NDArray[np.floating]) -> None:
        """Copy full padded data to buffer."""
        self._buffer[:] = data


__all__ = [
    # Configuration
    "GhostCellConfig",
    # Function-based API
    "apply_boundary_conditions_1d",
    "apply_boundary_conditions_2d",
    "apply_boundary_conditions_3d",
    "apply_boundary_conditions_nd",
    "create_boundary_mask_2d",
    "get_ghost_values_nd",
    # Class-based API (Topology/Calculator composition - Issue #516)
    "GhostBuffer",
    "FDMApplicator",
    "PreallocatedGhostBuffer",
]


if __name__ == "__main__":
    """Smoke tests for applicator_fdm module."""

    print("Testing FDM boundary condition applicators...")

    # Test 1: Basic 2D Neumann BC
    print("\n1. Testing 2D Neumann BC (no-flux)...")
    from mfg_pde.geometry.boundary import neumann_bc

    field_2d = np.ones((5, 5))
    bc = neumann_bc(dimension=2)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    padded = apply_boundary_conditions_2d(field_2d, bc, bounds)
    assert padded.shape == (7, 7), f"Expected (7,7), got {padded.shape}"
    # Neumann with zero flux: ghost = interior
    assert np.allclose(padded[0, 1:-1], padded[1, 1:-1]), "Neumann BC failed"
    print("   PASS: 2D Neumann BC")

    # Test 2: PreallocatedGhostBuffer (2D Dirichlet)
    print("\n2. Testing PreallocatedGhostBuffer (2D Dirichlet)...")
    from mfg_pde.geometry.boundary import dirichlet_bc

    bc_dirichlet = dirichlet_bc(dimension=2, value=0.0)
    buffer = PreallocatedGhostBuffer(
        interior_shape=(5, 5),
        boundary_conditions=bc_dirichlet,
        domain_bounds=bounds,
    )
    # Set interior to constant
    buffer.interior[:] = 1.0
    # Update ghosts
    buffer.update_ghosts()
    # Dirichlet with g=0: ghost = 2*0 - interior = -interior
    padded_buf = buffer.padded
    assert padded_buf.shape == (7, 7), f"Expected (7,7), got {padded_buf.shape}"
    assert np.allclose(padded_buf[0, 1:-1], -1.0), "Dirichlet ghost failed (low)"
    assert np.allclose(buffer.interior, 1.0), "Interior modified unexpectedly"
    print("   PASS: PreallocatedGhostBuffer (2D Dirichlet)")

    # Test 3: PreallocatedGhostBuffer (3D Periodic)
    print("\n3. Testing PreallocatedGhostBuffer (3D Periodic)...")
    from mfg_pde.geometry.boundary import periodic_bc

    bc_periodic = periodic_bc(dimension=3)
    bounds_3d = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    buffer_3d = PreallocatedGhostBuffer(
        interior_shape=(4, 4, 4),
        boundary_conditions=bc_periodic,
        domain_bounds=bounds_3d,
    )
    # Set a gradient in x
    for i in range(4):
        buffer_3d.interior[i, :, :] = float(i)
    buffer_3d.update_ghosts()
    padded_3d = buffer_3d.padded
    # Periodic: low ghost = high interior
    assert np.allclose(padded_3d[0, 1:-1, 1:-1], padded_3d[-2, 1:-1, 1:-1]), "Periodic low ghost failed"
    # Periodic: high ghost = low interior
    assert np.allclose(padded_3d[-1, 1:-1, 1:-1], padded_3d[1, 1:-1, 1:-1]), "Periodic high ghost failed"
    print("   PASS: PreallocatedGhostBuffer (3D Periodic)")

    # Test 4: Zero-copy verification
    print("\n4. Testing zero-copy interior view...")
    bc_neumann = neumann_bc(dimension=2)
    buffer2 = PreallocatedGhostBuffer(
        interior_shape=(3, 3),
        boundary_conditions=bc_neumann,
        domain_bounds=bounds,
    )
    interior_view = buffer2.interior
    interior_view[1, 1] = 42.0
    # Modification should reflect in padded
    assert buffer2.padded[2, 2] == 42.0, "Zero-copy failed"
    print("   PASS: Zero-copy interior view")

    # Test 5: 3D edge/corner handling (Dirichlet)
    print("\n5. Testing 3D edge/corner handling (Dirichlet)...")
    field_3d = np.ones((3, 3, 3))
    bc_3d_dir = dirichlet_bc(dimension=3, value=0.0)
    bounds_3d = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    padded_3d_dir = apply_boundary_conditions_nd(field_3d, bc_3d_dir, bounds_3d)
    # For Dirichlet g=0 with interior=1: ghost = 2*0 - 1 = -1
    # Faces should be -1
    assert np.allclose(padded_3d_dir[0, 2, 2], -1.0), "3D face ghost failed"
    # Edges should be -1 (averaged from 2 faces, both -1)
    assert np.allclose(padded_3d_dir[0, 0, 2], -1.0), "3D edge ghost failed"
    # Corners should be -1 (averaged from 3 edges, all -1)
    assert np.allclose(padded_3d_dir[0, 0, 0], -1.0), "3D corner ghost failed"
    print("   PASS: 3D edge/corner handling (Dirichlet)")

    # Test 6: 4D edge/corner handling (Dirichlet)
    print("\n6. Testing 4D edge/corner handling (Dirichlet)...")
    field_4d = np.ones((2, 2, 2, 2))
    bc_4d = dirichlet_bc(dimension=4, value=0.0)
    bounds_4d = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    padded_4d = apply_boundary_conditions_nd(field_4d, bc_4d, bounds_4d)
    # Corner in 4D should still be -1
    assert np.allclose(padded_4d[0, 0, 0, 0], -1.0), "4D corner ghost failed"
    # Edge (2 dims at boundary)
    assert np.allclose(padded_4d[0, 0, 1, 1], -1.0), "4D edge ghost failed"
    print("   PASS: 4D edge/corner handling (Dirichlet)")

    print("\n" + "=" * 50)
    print("All smoke tests passed!")
    print("=" * 50)
