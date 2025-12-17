"""
Unit tests for BC applicator utilities.

Tests the application of uniform and mixed boundary conditions to grid fields.
"""

import pytest

import numpy as np

from mfg_pde.geometry.boundary import (
    BCSegment,
    BCType,
    MixedBoundaryConditions,
    apply_boundary_conditions_2d,
    apply_boundary_conditions_nd,
    create_boundary_mask_2d,
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    # Factory functions for creating BCs
    periodic_bc,
)
from mfg_pde.geometry.boundary.applicator_fdm import GhostCellConfig


class TestUniformBC2D:
    """Tests for uniform BC application in 2D."""

    def test_periodic_bc(self):
        """Test periodic boundary conditions."""
        field = np.arange(25).reshape(5, 5).astype(float)
        bc = periodic_bc(dimension=2)
        padded = apply_boundary_conditions_2d(field, bc)

        assert padded.shape == (7, 7)
        # Check wrap-around
        assert padded[0, 3] == field[-1, 2]  # Top ghost from bottom interior
        assert padded[-1, 3] == field[0, 2]  # Bottom ghost from top interior
        assert padded[3, 0] == field[2, -1]  # Left ghost from right interior
        assert padded[3, -1] == field[2, 0]  # Right ghost from left interior

    def test_dirichlet_bc(self):
        """Test Dirichlet boundary conditions."""
        field = np.ones((5, 5))
        bc = dirichlet_bc(value=0.0, dimension=2)
        padded = apply_boundary_conditions_2d(field, bc)

        assert padded.shape == (7, 7)
        # Ghost cell formula for cell-centered grid: ghost = 2*g - interior
        # With g=0 and interior=1: ghost = 2*0 - 1 = -1
        assert padded[0, 3] == -1.0
        assert padded[-1, 3] == -1.0
        assert padded[3, 0] == -1.0
        assert padded[3, -1] == -1.0

    def test_neumann_bc(self):
        """Test Neumann (no-flux) boundary conditions."""
        field = np.arange(25).reshape(5, 5).astype(float)
        bc = neumann_bc(dimension=2)
        padded = apply_boundary_conditions_2d(field, bc)

        assert padded.shape == (7, 7)
        # Check ghost cells equal adjacent interior (edge padding)
        assert padded[0, 3] == field[0, 2]  # Bottom ghost = first row
        assert padded[-1, 3] == field[-1, 2]  # Top ghost = last row
        assert padded[3, 0] == field[2, 0]  # Left ghost = first column
        assert padded[3, -1] == field[2, -1]  # Right ghost = last column

    def test_no_flux_bc(self):
        """Test no_flux alias for Neumann BC."""
        field = np.ones((5, 5))
        bc = no_flux_bc(dimension=2)
        padded = apply_boundary_conditions_2d(field, bc)

        assert padded.shape == (7, 7)
        # All ghost cells should equal 1 (edge padding of uniform field)
        assert np.allclose(padded, 1.0)


class TestMixedBC2D:
    """Tests for mixed BC application in 2D."""

    def test_basic_mixed_bc(self):
        """Test basic mixed BC with exit and walls."""
        # Exit on right wall (Dirichlet), walls elsewhere (Neumann)
        exit_bc = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            boundary="x_max",
            priority=1,
        )
        wall_bc = BCSegment(
            name="walls",
            bc_type=BCType.NEUMANN,
            value=0.0,
            priority=0,
        )
        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[exit_bc, wall_bc],
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        field = np.ones((5, 5))
        padded = apply_boundary_conditions_2d(field, mixed_bc)

        assert padded.shape == (7, 7)

        # Right boundary (exit, Dirichlet): ghost = 2*g - interior = -1
        # Interior is column -2 in padded = column 4 in field = 1.0
        assert np.allclose(padded[1:-1, -1], -1.0)

        # Left boundary (walls, Neumann): ghost = interior = 1
        assert np.allclose(padded[1:-1, 0], 1.0)

    def test_partial_exit(self):
        """Test mixed BC with exit on part of boundary."""
        # Exit on right wall, y in [0.4, 0.6] (Dirichlet)
        exit_bc = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            boundary="x_max",
            region={"y": (0.4, 0.6)},
            priority=1,
        )
        wall_bc = BCSegment(
            name="walls",
            bc_type=BCType.NEUMANN,
            value=0.0,
            priority=0,
        )
        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[exit_bc, wall_bc],
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        field = np.ones((5, 5))
        padded = apply_boundary_conditions_2d(field, mixed_bc)

        # y coordinates: [0.0, 0.25, 0.5, 0.75, 1.0]
        # Exit region y in [0.4, 0.6] covers only y=0.5 (index 2)

        # Right ghost column: Dirichlet at y=0.5, Neumann elsewhere
        # y=0.0 (padded row 1): Neumann -> ghost = interior = 1
        # y=0.25 (padded row 2): Neumann -> ghost = interior = 1
        # y=0.5 (padded row 3): Dirichlet -> ghost = 2*0 - 1 = -1
        # y=0.75 (padded row 4): Neumann -> ghost = interior = 1
        # y=1.0 (padded row 5): Neumann -> ghost = interior = 1

        assert padded[1, -1] == 1.0  # y=0.0, Neumann
        assert padded[2, -1] == 1.0  # y=0.25, Neumann
        assert padded[3, -1] == -1.0  # y=0.5, Dirichlet
        assert padded[4, -1] == 1.0  # y=0.75, Neumann
        assert padded[5, -1] == 1.0  # y=1.0, Neumann

    def test_domain_bounds_from_mixed_bc(self):
        """Test that domain_bounds can be provided via MixedBoundaryConditions."""
        exit_bc = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0)
        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[exit_bc],
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        field = np.ones((5, 5))
        # Should work without passing domain_bounds explicitly
        padded = apply_boundary_conditions_2d(field, mixed_bc)
        assert padded.shape == (7, 7)


class TestBC1D:
    """Tests for 1D boundary conditions."""

    def test_uniform_bc_1d(self):
        """Test uniform BC in 1D."""
        field = np.ones(5)
        bc = neumann_bc(dimension=1)
        padded = apply_boundary_conditions_nd(field, bc)

        assert padded.shape == (7,)
        assert np.allclose(padded, 1.0)

    def test_dirichlet_bc_1d(self):
        """Test Dirichlet BC in 1D with uniform value."""
        field = np.ones(5)
        # Use uniform Dirichlet with value 0 for simplicity
        bc = dirichlet_bc(value=0.0, dimension=1)
        padded = apply_boundary_conditions_nd(field, bc)

        assert padded.shape == (7,)
        # Ghost cell formula: ghost = 2*g - interior
        # With g=0 and interior=1: ghost = 2*0 - 1 = -1
        assert padded[0] == -1.0
        assert padded[-1] == -1.0

    def test_mixed_bc_1d(self):
        """Test mixed BC in 1D."""
        left_bc = BCSegment(
            name="inlet",
            bc_type=BCType.DIRICHLET,
            value=1.0,
            boundary="x_min",
            priority=1,
        )
        right_bc = BCSegment(
            name="outlet",
            bc_type=BCType.NEUMANN,
            value=0.0,
            boundary="x_max",
            priority=1,
        )
        mixed_bc = MixedBoundaryConditions(
            dimension=1,
            segments=[left_bc, right_bc],
            domain_bounds=np.array([[0.0, 1.0]]),
        )

        field = np.zeros(5)
        padded = apply_boundary_conditions_nd(field, mixed_bc)

        assert padded.shape == (7,)
        # Left: Dirichlet g=1, interior=0 -> ghost = 2*1 - 0 = 2
        assert padded[0] == 2.0
        # Right: Neumann g=0 -> ghost = interior = 0
        assert padded[-1] == 0.0


class TestBoundaryMask:
    """Tests for pre-computed boundary masks."""

    def test_create_boundary_mask(self):
        """Test creation of boundary masks for efficiency."""
        exit_bc = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            boundary="x_max",
            region={"y": (0.4, 0.6)},
            priority=1,
        )
        wall_bc = BCSegment(
            name="walls",
            bc_type=BCType.NEUMANN,
            value=0.0,
            priority=0,
        )
        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[exit_bc, wall_bc],
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        masks = create_boundary_mask_2d(mixed_bc, (5, 5), mixed_bc.domain_bounds)

        # Check exit mask on right boundary
        assert "exit" in masks
        assert "walls" in masks

        # Exit only on right boundary at y=0.5 (index 2)
        assert masks["exit"]["left"].sum() == 0
        assert masks["exit"]["right"].sum() == 1
        assert masks["exit"]["right"][2]  # y=0.5

        # Walls everywhere else
        assert masks["walls"]["left"].sum() == 5  # All left boundary
        assert masks["walls"]["right"].sum() == 4  # Right except exit


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_domain_bounds_error(self):
        """Test that missing domain_bounds raises error for mixed BC."""
        # Create a proper mixed BC with boundary restrictions (not uniform)
        exit_bc = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            boundary="x_max",  # This makes it a true mixed BC
            priority=1,
        )
        wall_bc = BCSegment(
            name="wall",
            bc_type=BCType.NEUMANN,
            value=0.0,
            priority=0,
        )
        mixed_bc_obj = MixedBoundaryConditions(
            dimension=2,
            segments=[exit_bc, wall_bc],
            # No domain_bounds
        )

        field = np.ones((5, 5))
        with pytest.raises(ValueError, match="domain_bounds"):
            apply_boundary_conditions_2d(field, mixed_bc_obj)

    def test_unsupported_bc_type_error(self):
        """Test that invalid BC type raises error during creation."""
        # With the new unified class, invalid types raise error during creation
        from mfg_pde.geometry.boundary import uniform_bc

        with pytest.raises(ValueError):
            uniform_bc(bc_type="unknown_type")

    def test_corner_handling(self):
        """Test that corners are handled correctly (averaged)."""
        exit_bc = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0)
        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[exit_bc],
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        field = np.ones((3, 3))
        padded = apply_boundary_conditions_2d(field, mixed_bc)

        # Corners should be average of adjacent ghost cells
        # Bottom-left corner
        expected_bl = 0.5 * (padded[0, 1] + padded[1, 0])
        assert padded[0, 0] == expected_bl

        # Top-right corner
        expected_tr = 0.5 * (padded[-1, -2] + padded[-2, -1])
        assert padded[-1, -1] == expected_tr


class TestRobinBC:
    """Tests for Robin boundary condition: alpha*u + beta*du/dn = g."""

    def test_robin_bc_basic(self):
        """Test Robin BC with alpha=1, beta=1, g=0.5."""
        # Robin BC: alpha*u + beta*du/dn = g at boundary
        # Ghost formula: ghost = (2*g - alpha*interior + beta*interior/dx) / (alpha + beta/dx)
        robin_bc = BCSegment(
            name="robin",
            bc_type=BCType.ROBIN,
            value=0.5,  # g = 0.5
            alpha=1.0,
            beta=1.0,
            priority=1,
        )
        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[robin_bc],
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        field = np.ones((5, 5))
        padded = apply_boundary_conditions_2d(field, mixed_bc)

        # dx = 1.0/4 = 0.25, interior = 1.0, g = 0.5, alpha = 1, beta = 1
        # For x_min boundary (boundary_side="min"):
        #   ghost = (2*0.5 - 1*1 + 1*1/0.25) / (1 + 1/0.25)
        #        = (1 - 1 + 4) / (1 + 4) = 4/5 = 0.8
        # Note: sign changes for min vs max boundary
        assert padded.shape == (7, 7)
        # Ghost cells should be different from interior due to Robin BC

    def test_robin_reduces_to_dirichlet(self):
        """Test that Robin with beta=0 is equivalent to Dirichlet."""
        # When beta=0: alpha*u = g at boundary -> u = g/alpha
        robin_bc = BCSegment(
            name="dirichlet_like",
            bc_type=BCType.ROBIN,
            value=2.0,  # g = 2
            alpha=1.0,
            beta=0.0,  # No flux term -> pure Dirichlet
            priority=1,
        )
        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[robin_bc],
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        field = np.ones((5, 5))
        padded = apply_boundary_conditions_2d(field, mixed_bc)

        # For Dirichlet-like Robin with g=2, interior=1:
        # ghost = 2*g - interior = 2*2 - 1 = 3
        assert np.isclose(padded[3, 0], 3.0)  # Left boundary
        assert np.isclose(padded[3, -1], 3.0)  # Right boundary

    def test_robin_reduces_to_neumann(self):
        """Test that Robin with alpha=0 is equivalent to Neumann."""
        # When alpha=0: beta*du/dn = g -> du/dn = g/beta
        robin_bc = BCSegment(
            name="neumann_like",
            bc_type=BCType.ROBIN,
            value=0.0,  # g = 0 (no-flux)
            alpha=0.0,
            beta=1.0,  # Only flux term -> pure Neumann
            priority=1,
        )
        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[robin_bc],
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        field = np.arange(25).reshape(5, 5).astype(float)
        padded = apply_boundary_conditions_2d(field, mixed_bc)

        # Neumann with g=0 means ghost = interior (zero gradient)
        assert np.isclose(padded[3, 0], field[2, 0])  # Left: ghost = interior
        assert np.isclose(padded[3, -1], field[2, -1])  # Right: ghost = interior


class TestTimeDependentBC:
    """Tests for time-dependent boundary values."""

    def test_time_dependent_dirichlet(self):
        """Test Dirichlet BC with time-varying value."""

        # Value depends on time: g(t) = 2*t
        def time_varying_value(point, time):
            return 2.0 * time

        exit_bc = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            value=time_varying_value,
            boundary="x_max",
            priority=1,
        )
        wall_bc = BCSegment(
            name="walls",
            bc_type=BCType.NEUMANN,
            value=0.0,
            priority=0,
        )
        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[exit_bc, wall_bc],
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        field = np.ones((5, 5))

        # At t=0: g=0, ghost = 2*0 - 1 = -1
        padded_t0 = apply_boundary_conditions_2d(field, mixed_bc, time=0.0)
        assert np.allclose(padded_t0[1:-1, -1], -1.0)

        # At t=1: g=2, ghost = 2*2 - 1 = 3
        padded_t1 = apply_boundary_conditions_2d(field, mixed_bc, time=1.0)
        assert np.allclose(padded_t1[1:-1, -1], 3.0)

        # At t=0.5: g=1, ghost = 2*1 - 1 = 1
        padded_t05 = apply_boundary_conditions_2d(field, mixed_bc, time=0.5)
        assert np.allclose(padded_t05[1:-1, -1], 1.0)

    def test_space_and_time_dependent_neumann(self):
        """Test Neumann BC with value depending on both space and time."""

        # Flux = x * t (position-dependent flux that varies with time)
        def space_time_flux(point, time):
            return point[0] * time

        flux_bc = BCSegment(
            name="varying_flux",
            bc_type=BCType.NEUMANN,
            value=space_time_flux,
            boundary="y_max",  # Top boundary
            priority=1,
        )
        wall_bc = BCSegment(
            name="walls",
            bc_type=BCType.NEUMANN,
            value=0.0,
            priority=0,
        )
        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[flux_bc, wall_bc],
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        field = np.ones((5, 5))

        # At t=0: flux=0 everywhere, ghost = interior
        padded_t0 = apply_boundary_conditions_2d(field, mixed_bc, time=0.0)
        assert np.isclose(padded_t0[-1, 3], 1.0)  # No change from interior

        # At t=1: flux varies with x position
        # Top boundary points have x coords [0, 0.25, 0.5, 0.75, 1.0]
        # At x=0.5 (center): flux = 0.5*1 = 0.5
        # ghost = interior + 2*dx*flux = 1 + 2*0.25*0.5 = 1.25
        _padded_t1 = apply_boundary_conditions_2d(field, mixed_bc, time=1.0)
        # The ghost values at top should vary with x (stored in _padded_t1 for future assertions)


class TestGridTypeConfiguration:
    """Tests for cell-centered vs vertex-centered grid configuration."""

    def test_cell_centered_dirichlet(self):
        """Test Dirichlet BC with cell-centered grid (default)."""
        field = np.ones((5, 5))
        bc = dirichlet_bc(value=0.0, dimension=2)

        config = GhostCellConfig(grid_type="cell_centered")
        padded = apply_boundary_conditions_2d(field, bc, config=config)

        # Cell-centered: ghost = 2*g - interior = 2*0 - 1 = -1
        assert np.isclose(padded[3, 0], -1.0)
        assert np.isclose(padded[3, -1], -1.0)

    def test_vertex_centered_dirichlet(self):
        """Test Dirichlet BC with vertex-centered grid."""
        field = np.ones((5, 5))
        bc = dirichlet_bc(value=0.0, dimension=2)

        config = GhostCellConfig(grid_type="vertex_centered")
        padded = apply_boundary_conditions_2d(field, bc, config=config)

        # Vertex-centered: ghost = g (boundary value at vertex)
        assert np.isclose(padded[3, 0], 0.0)
        assert np.isclose(padded[3, -1], 0.0)


class TestInputValidation:
    """Tests for input validation."""

    def test_nan_detection(self):
        """Test that NaN values in field are detected."""
        field = np.ones((5, 5))
        field[2, 2] = np.nan

        bc = neumann_bc(dimension=2)
        with pytest.raises(ValueError, match="NaN"):
            apply_boundary_conditions_2d(field, bc)

    def test_inf_detection(self):
        """Test that infinite values in field are detected."""
        field = np.ones((5, 5))
        field[2, 2] = np.inf

        bc = neumann_bc(dimension=2)
        with pytest.raises(ValueError, match="NaN or Inf"):
            apply_boundary_conditions_2d(field, bc)

    def test_invalid_domain_bounds(self):
        """Test that invalid domain bounds are rejected."""
        # Create proper mixed BC with boundary restrictions
        exit_bc = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            boundary="x_max",
            priority=1,
        )
        wall_bc = BCSegment(
            name="wall",
            bc_type=BCType.NEUMANN,
            value=0.0,
            priority=0,
        )
        # min > max is invalid
        mixed_bc_obj = MixedBoundaryConditions(
            dimension=2,
            segments=[exit_bc, wall_bc],
            domain_bounds=np.array([[1.0, 0.0], [0.0, 1.0]]),  # x_min > x_max
        )

        field = np.ones((5, 5))
        with pytest.raises(ValueError, match="min < max"):
            apply_boundary_conditions_2d(field, mixed_bc_obj)


class TestLazyDimensionBinding:
    """Tests for lazy dimension binding (Issue #495)."""

    def test_dirichlet_bc_no_dimension(self):
        """Test creating Dirichlet BC without dimension."""
        bc = dirichlet_bc(value=0.0)  # No dimension specified
        assert bc.dimension is None
        assert not bc.is_bound
        assert str(bc) == "BoundaryConditions(unbound, dirichlet, value=0.0)"

    def test_neumann_bc_no_dimension(self):
        """Test creating Neumann BC without dimension."""
        bc = neumann_bc(value=0.0)  # No dimension specified
        assert bc.dimension is None
        assert not bc.is_bound

    def test_periodic_bc_no_dimension(self):
        """Test creating Periodic BC without dimension."""
        bc = periodic_bc()  # No dimension specified
        assert bc.dimension is None
        assert not bc.is_bound

    def test_bind_dimension_explicit(self):
        """Test explicit dimension binding via bind_dimension()."""
        bc = dirichlet_bc(value=0.0)
        assert bc.dimension is None

        bc_2d = bc.bind_dimension(2)
        assert bc_2d.dimension == 2
        assert bc_2d.is_bound
        assert str(bc_2d) == "BoundaryConditions(2D, dirichlet, value=0.0)"

        # Original BC should be unchanged (immutable via replace)
        assert bc.dimension is None

    def test_bind_dimension_idempotent(self):
        """Test that binding same dimension twice returns same BC."""
        bc = dirichlet_bc(value=0.0, dimension=2)
        bc_bound = bc.bind_dimension(2)

        # Should return same object since dimension already matches
        assert bc_bound is bc

    def test_bind_dimension_mismatch_error(self):
        """Test that binding different dimension raises error."""
        bc = dirichlet_bc(value=0.0, dimension=2)

        with pytest.raises(ValueError, match="BC dimension mismatch"):
            bc.bind_dimension(3)

    def test_grid_binds_dimension_automatically(self):
        """Test that TensorProductGrid automatically binds dimension."""
        from mfg_pde.geometry import TensorProductGrid

        # Create BC without dimension
        bc = dirichlet_bc(value=0.0)
        assert bc.dimension is None

        # Create grid with BC - dimension should be bound automatically
        grid = TensorProductGrid(
            dimension=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx=[10, 10],
            boundary_conditions=bc,
        )

        # Grid's stored BC should have dimension bound
        stored_bc = grid.get_boundary_conditions()
        assert stored_bc.dimension == 2
        assert stored_bc.is_bound

    def test_unbound_bc_cannot_apply(self):
        """Test that unbound BC cannot be applied directly."""
        bc = dirichlet_bc(value=0.0)  # No dimension

        # Applicator requires dimension to compute ghost cells
        with pytest.raises(ValueError, match="BC dimension not set"):
            # This should fail because dimension is needed for ghost cell computation
            bc._require_dimension("apply BC")

    def test_explicit_dimension_still_works(self):
        """Test that explicit dimension specification still works."""
        bc = dirichlet_bc(value=0.0, dimension=2)
        assert bc.dimension == 2
        assert bc.is_bound

        # Should work with apply_boundary_conditions_2d
        field = np.ones((5, 5))
        padded = apply_boundary_conditions_2d(field, bc)
        assert padded.shape == (7, 7)

    def test_validate_unbound_bc_warns(self):
        """Test that validate() warns about unbound dimension."""
        bc = dirichlet_bc(value=0.0)
        _is_valid, warnings = bc.validate()

        # Should have warning about dimension not set
        assert any("Dimension not set" in w for w in warnings)

    def test_backward_compatibility_dimension_default(self):
        """Test backward compatibility with code that didn't specify dimension."""
        # Before this change, dimension defaulted to 2
        # Now dimension defaults to None, but existing code that passes dimension=2
        # should still work identically
        bc_explicit = dirichlet_bc(value=0.0, dimension=2)
        assert bc_explicit.dimension == 2

        # Code that uses the BC with a grid should work
        field = np.ones((5, 5))
        padded = apply_boundary_conditions_2d(field, bc_explicit)
        assert padded.shape == (7, 7)
