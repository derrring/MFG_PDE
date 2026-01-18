"""
Demonstration: Using shared ghost cell formulas (Issue #598 Phase 2).

Shows how dimension-specific BC application can be simplified by using
the shared formula methods from BaseStructuredApplicator.

This file demonstrates the migration pattern without breaking existing code.

Run: python mfg_pde/geometry/boundary/_demo_shared_formulas_usage.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfg_pde.geometry.boundary.applicator_base import BaseStructuredApplicator
from mfg_pde.geometry.boundary.applicator_fdm import GhostCellConfig
from mfg_pde.geometry.boundary.conditions import BoundaryConditions  # noqa: TC001 (used at runtime)
from mfg_pde.geometry.boundary.types import BCType

if TYPE_CHECKING:
    from numpy.typing import NDArray


def apply_bc_1d_refactored(
    field: NDArray[np.floating],
    boundary_conditions: BoundaryConditions,
    domain_bounds: NDArray[np.floating] | None = None,
    time: float = 0.0,
    config: GhostCellConfig | None = None,
) -> NDArray[np.floating]:
    """
    Apply 1D boundary conditions using shared ghost cell formulas.

    This is a refactored version of apply_boundary_conditions_1d() that
    demonstrates using the shared formula methods from BaseStructuredApplicator
    instead of duplicating ghost cell logic.

    **Key Changes from Original**:
    - Uses BaseStructuredApplicator._compute_ghost_*() methods
    - Eliminates duplicated Dirichlet/Neumann/Robin formulas
    - Validation extracted to shared _validate_field()
    - Grid spacing computed via shared _compute_grid_spacing()

    Args:
        field: Interior field of shape (N,)
        boundary_conditions: BC specification (uniform or mixed)
        domain_bounds: Domain bounds [[xmin, xmax]] (required for mixed BC)
        time: Current time for time-dependent BC values
        config: Ghost cell configuration (grid type, etc.)

    Returns:
        Padded field of shape (N+2,) with ghost cells

    Example:
        >>> from mfg_pde.geometry.boundary import dirichlet_bc
        >>> bc = dirichlet_bc(dimension=1, value=0.0)
        >>> field = np.linspace(0, 1, 51)
        >>> padded = apply_bc_1d_refactored(field, bc)
        >>> padded.shape
        (53,)
    """
    # Setup
    if config is None:
        config = GhostCellConfig()

    # Create applicator to access shared formula methods
    # (In production, this would be passed in or created once)
    applicator = BaseStructuredApplicator(dimension=1, grid_type=config.grid_type)

    # Shared validation (eliminates duplication)
    applicator._validate_field(field)

    # Handle uniform BC (same type on all boundaries)
    if boundary_conditions.is_uniform:
        seg = boundary_conditions.segments[0]
        bc_type = seg.bc_type
        g = seg.value if seg.value is not None else 0.0

        if bc_type == BCType.PERIODIC:
            return np.pad(field, 1, mode="wrap")

        elif bc_type == BCType.DIRICHLET:
            # OLD (duplicated): padded[0] = 2.0 * g - field[0]
            # NEW (shared): use _compute_ghost_dirichlet()
            padded = np.zeros(len(field) + 2, dtype=field.dtype)
            padded[1:-1] = field
            padded[0] = applicator._compute_ghost_dirichlet(field[0], g, time)
            padded[-1] = applicator._compute_ghost_dirichlet(field[-1], g, time)
            return padded

        elif bc_type in [BCType.NO_FLUX, BCType.NEUMANN]:
            # OLD (duplicated): padded[0] = field[1] (reflection)
            # NEW (shared): use _compute_ghost_neumann()
            if len(field) < 2:
                return np.pad(field, 1, mode="edge")  # Fallback
            padded = np.zeros(len(field) + 2, dtype=field.dtype)
            padded[1:-1] = field
            # For zero-flux Neumann, u_next_interior is field[1] for left
            padded[0] = applicator._compute_ghost_neumann(field[0], field[1], g, dx=0.1, side="left", time=time)
            padded[-1] = applicator._compute_ghost_neumann(field[-1], field[-2], g, dx=0.1, side="right", time=time)
            return padded

        elif bc_type == BCType.ROBIN:
            # Robin BC: alpha*u + beta*du/dn = g
            # Extract alpha, beta from segment (if available)
            alpha = getattr(seg, "alpha", 1.0)
            beta = getattr(seg, "beta", 0.1)
            if domain_bounds is None:
                msg = "Robin BC requires domain_bounds for grid spacing"
                raise ValueError(msg)
            spacing = applicator._compute_grid_spacing(field, domain_bounds)
            dx = spacing[0]

            # OLD (duplicated): Robin formula inline
            # NEW (shared): use _compute_ghost_robin()
            padded = np.zeros(len(field) + 2, dtype=field.dtype)
            padded[1:-1] = field
            padded[0] = applicator._compute_ghost_robin(field[0], alpha, beta, g, dx, side="left", time=time)
            padded[-1] = applicator._compute_ghost_robin(field[-1], alpha, beta, g, dx, side="right", time=time)
            return padded

        else:
            msg = f"Unsupported BC type: {bc_type}"
            raise ValueError(msg)

    # Mixed BC (different types on different boundaries)
    else:
        if domain_bounds is None:
            msg = "Mixed boundary conditions require domain_bounds"
            raise ValueError(msg)

        # Compute grid spacing using shared method
        spacing = applicator._compute_grid_spacing(field, domain_bounds)
        dx = spacing[0]

        # Initialize padded buffer using shared method
        padded = applicator._create_padded_buffer(field, ghost_depth=1)

        # Get BC for left boundary (x_min)
        left_point = np.array([domain_bounds[0, 0]])
        left_segment = boundary_conditions.get_bc_at_point(left_point, "x_min")

        # Get BC for right boundary (x_max)
        right_point = np.array([domain_bounds[0, 1]])
        right_segment = boundary_conditions.get_bc_at_point(right_point, "x_max")

        # Apply left BC using shared formulas
        left_bc = left_segment.bc_type
        left_val = left_segment.value if left_segment.value is not None else 0.0

        if left_bc == BCType.DIRICHLET:
            padded[0] = applicator._compute_ghost_dirichlet(field[0], left_val, time)
        elif left_bc in [BCType.NO_FLUX, BCType.NEUMANN]:
            padded[0] = applicator._compute_ghost_neumann(field[0], field[1], left_val, dx, side="left", time=time)
        elif left_bc == BCType.ROBIN:
            alpha = getattr(left_segment, "alpha", 1.0)
            beta = getattr(left_segment, "beta", 0.1)
            padded[0] = applicator._compute_ghost_robin(field[0], alpha, beta, left_val, dx, side="left", time=time)

        # Apply right BC using shared formulas
        right_bc = right_segment.bc_type
        right_val = right_segment.value if right_segment.value is not None else 0.0

        if right_bc == BCType.DIRICHLET:
            padded[-1] = applicator._compute_ghost_dirichlet(field[-1], right_val, time)
        elif right_bc in [BCType.NO_FLUX, BCType.NEUMANN]:
            padded[-1] = applicator._compute_ghost_neumann(field[-1], field[-2], right_val, dx, side="right", time=time)
        elif right_bc == BCType.ROBIN:
            alpha = getattr(right_segment, "alpha", 1.0)
            beta = getattr(right_segment, "beta", 0.1)
            padded[-1] = applicator._compute_ghost_robin(field[-1], alpha, beta, right_val, dx, side="right", time=time)

        return padded


def demonstrate_refactored_bc():
    """Demonstrate using refactored BC application."""
    print("=" * 70)
    print("Demonstration: Refactored BC Application (Issue #598 Phase 2)")
    print("=" * 70)
    print()

    # Create test field
    field = np.linspace(0.2, 0.8, 11)
    domain_bounds = np.array([[0.0, 1.0]])

    print(f"Interior field: {field}")
    print(f"Domain bounds: [{domain_bounds[0, 0]}, {domain_bounds[0, 1]}]")
    print()

    # Test 1: Uniform Dirichlet BC
    print("Test 1: Uniform Dirichlet BC (g=0.0)")
    print("-" * 70)
    from mfg_pde.geometry.boundary import dirichlet_bc

    bc_dirichlet = dirichlet_bc(dimension=1, value=0.0)
    padded_dirichlet = apply_bc_1d_refactored(field, bc_dirichlet, domain_bounds)
    print(f"Padded field: {padded_dirichlet}")
    print(f"Left ghost: {padded_dirichlet[0]:.4f} (expect 2*0.0 - 0.2 = -0.2)")
    print(f"Right ghost: {padded_dirichlet[-1]:.4f} (expect 2*0.0 - 0.8 = -0.8)")
    print()

    # Test 2: Uniform Neumann BC (zero-flux)
    print("Test 2: Uniform Neumann BC (zero-flux)")
    print("-" * 70)
    from mfg_pde.geometry.boundary import neumann_bc

    bc_neumann = neumann_bc(dimension=1, value=0.0)
    padded_neumann = apply_bc_1d_refactored(field, bc_neumann, domain_bounds)
    print(f"Padded field: {padded_neumann}")
    print(f"Left ghost: {padded_neumann[0]:.4f} (expect reflection = field[1])")
    print(f"Right ghost: {padded_neumann[-1]:.4f} (expect reflection = field[-2])")
    print()

    # Test 3: Mixed BC
    print("Test 3: Mixed BC (Dirichlet left, Neumann right)")
    print("-" * 70)
    from mfg_pde.geometry.boundary import BCSegment, mixed_bc

    bc_left = BCSegment(name="left", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_min")
    bc_right = BCSegment(name="right", bc_type=BCType.NEUMANN, value=0.0, boundary="x_max")
    bc_mixed = mixed_bc([bc_left, bc_right], dimension=1, domain_bounds=domain_bounds)
    padded_mixed = apply_bc_1d_refactored(field, bc_mixed, domain_bounds)
    print(f"Padded field: {padded_mixed}")
    print(f"Left ghost: {padded_mixed[0]:.4f} (Dirichlet: 2*1.0 - 0.2 = 1.8)")
    print(f"Right ghost: {padded_mixed[-1]:.4f} (Neumann: reflection = field[-2])")
    print()

    print("=" * 70)
    print("Demonstration Complete")
    print("=" * 70)
    print()
    print("Key Benefits of Shared Formula Methods:")
    print("1. DRY: Ghost cell formulas defined once in BaseStructuredApplicator")
    print("2. Consistency: All dimensions use same formulas (no drift)")
    print("3. Maintainability: Bug fixes in one place propagate everywhere")
    print("4. Testability: Shared methods have comprehensive unit tests")
    print()
    print("Next: Migrate apply_boundary_conditions_*d() to use this pattern")


if __name__ == "__main__":
    demonstrate_refactored_bc()
