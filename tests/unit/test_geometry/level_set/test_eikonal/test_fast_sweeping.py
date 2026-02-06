"""
Unit tests for Fast Sweeping Method (FSM).

These tests verify the FSM solver for the Eikonal equation.
"""

import pytest

import numpy as np

from mfg_pde.geometry.boundary import no_flux_bc
from mfg_pde.geometry.grids.tensor_grid import TensorProductGrid
from mfg_pde.geometry.level_set.eikonal import FastMarchingMethod, FastSweepingMethod


class TestFSMPointSource:
    """Test FSM with point source."""

    def test_1d_point_source(self):
        """Test 1D point source at center."""
        Nx = 101
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0)],
            Nx_points=[Nx],
            boundary_conditions=no_flux_bc(dimension=1),
        )
        x = grid.coordinates[0]
        dx = grid.spacing[0]

        fsm = FastSweepingMethod(grid)

        # Point source at x = 0.5
        i0 = Nx // 2
        frozen_mask = np.zeros(Nx, dtype=bool)
        frozen_mask[i0] = True
        frozen_values = np.zeros(Nx, dtype=np.float64)

        T = fsm.solve(speed=1.0, frozen_mask=frozen_mask, frozen_values=frozen_values)

        # Analytical: T = |x - x0|
        T_exact = np.abs(x - 0.5)
        error = np.max(np.abs(T - T_exact))

        assert error < 2 * dx, f"1D point source error {error} > 2*dx={2 * dx}"

    def test_2d_point_source(self):
        """Test 2D point source at center."""
        N = 51
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[N, N],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        X, Y = grid.meshgrid()
        dx = grid.spacing[0]

        fsm = FastSweepingMethod(grid)

        # Point source at center
        i0, j0 = N // 2, N // 2
        frozen_mask = np.zeros((N, N), dtype=bool)
        frozen_mask[i0, j0] = True
        frozen_values = np.zeros((N, N), dtype=np.float64)

        T = fsm.solve(speed=1.0, frozen_mask=frozen_mask, frozen_values=frozen_values)

        # Analytical: T = sqrt((x-0.5)^2 + (y-0.5)^2)
        T_exact = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
        error = np.max(np.abs(T - T_exact))

        assert error < 3 * dx, f"2D point source error {error} > 3*dx={3 * dx}"


class TestFSMSignedDistance:
    """Test FSM signed distance computation."""

    def test_circle_sdf(self):
        """Test SDF computation for a circle."""
        N = 51
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[N, N],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        X, Y = grid.meshgrid()
        dx = grid.spacing[0]

        fsm = FastSweepingMethod(grid)

        # Circle: phi = sqrt((x-0.5)^2 + (y-0.5)^2) - 0.3
        phi_circle = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - 0.3

        phi_sdf = fsm.compute_signed_distance(phi_circle)

        # Should be close to input (circle is already SDF)
        error = np.max(np.abs(phi_sdf - phi_circle))
        assert error < 2 * dx, f"Circle SDF error {error} > 2*dx={2 * dx}"


class TestFSMConvergence:
    """Test FSM convergence properties."""

    def test_converges_quickly(self):
        """Test that FSM converges in few iterations for simple domain."""
        N = 51
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[N, N],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        # Test with different max_iterations
        fsm_10 = FastSweepingMethod(grid, max_iterations=10, tolerance=1e-14)
        fsm_100 = FastSweepingMethod(grid, max_iterations=100, tolerance=1e-14)

        i0, j0 = N // 2, N // 2
        frozen_mask = np.zeros((N, N), dtype=bool)
        frozen_mask[i0, j0] = True
        frozen_values = np.zeros((N, N), dtype=np.float64)

        T_10 = fsm_10.solve(speed=1.0, frozen_mask=frozen_mask, frozen_values=frozen_values)
        T_100 = fsm_100.solve(speed=1.0, frozen_mask=frozen_mask, frozen_values=frozen_values)

        # Results should be identical (converged before max_iterations)
        assert np.allclose(T_10, T_100, atol=1e-10), "FSM did not converge in 10 iterations"


class TestFSMvsFMM:
    """Test that FSM produces same results as FMM."""

    def test_point_source_equivalence(self):
        """Test that FSM and FMM give same result for point source."""
        N = 51
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[N, N],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        fmm = FastMarchingMethod(grid)
        fsm = FastSweepingMethod(grid)

        # Point source at center
        i0, j0 = N // 2, N // 2
        frozen_mask = np.zeros((N, N), dtype=bool)
        frozen_mask[i0, j0] = True
        frozen_values = np.zeros((N, N), dtype=np.float64)

        T_fmm = fmm.solve(speed=1.0, frozen_mask=frozen_mask, frozen_values=frozen_values)
        T_fsm = fsm.solve(speed=1.0, frozen_mask=frozen_mask, frozen_values=frozen_values)

        diff = np.max(np.abs(T_fmm - T_fsm))
        assert diff < 1e-10, f"FMM and FSM differ by {diff}"

    def test_signed_distance_equivalence(self):
        """Test that FSM and FMM give same signed distance."""
        N = 51
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[N, N],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        X, Y = grid.meshgrid()

        fmm = FastMarchingMethod(grid)
        fsm = FastSweepingMethod(grid)

        phi = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - 0.3

        phi_fmm = fmm.compute_signed_distance(phi)
        phi_fsm = fsm.compute_signed_distance(phi)

        diff = np.max(np.abs(phi_fmm - phi_fsm))
        assert diff < 1e-10, f"FMM and FSM signed distances differ by {diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
