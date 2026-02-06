"""
Unit tests for Fast Marching Method (FMM).

These tests verify the FMM solver for the Eikonal equation.
"""

import pytest

import numpy as np

from mfg_pde.geometry.boundary import no_flux_bc
from mfg_pde.geometry.grids.tensor_grid import TensorProductGrid
from mfg_pde.geometry.level_set.eikonal import FastMarchingMethod


class TestFMMPointSource:
    """Test FMM with point source (T = |x - x0|)."""

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

        fmm = FastMarchingMethod(grid)

        # Point source at x = 0.5
        x0 = 0.5
        i0 = Nx // 2
        frozen_mask = np.zeros(Nx, dtype=bool)
        frozen_mask[i0] = True
        frozen_values = np.zeros(Nx, dtype=np.float64)

        T = fmm.solve(speed=1.0, frozen_mask=frozen_mask, frozen_values=frozen_values)

        # Analytical: T = |x - x0|
        T_exact = np.abs(x - x0)
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

        fmm = FastMarchingMethod(grid)

        # Point source at center
        i0, j0 = N // 2, N // 2
        frozen_mask = np.zeros((N, N), dtype=bool)
        frozen_mask[i0, j0] = True
        frozen_values = np.zeros((N, N), dtype=np.float64)

        T = fmm.solve(speed=1.0, frozen_mask=frozen_mask, frozen_values=frozen_values)

        # Analytical: T = sqrt((x-0.5)^2 + (y-0.5)^2)
        T_exact = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
        error = np.max(np.abs(T - T_exact))

        assert error < 3 * dx, f"2D point source error {error} > 3*dx={3 * dx}"


class TestFMMSignedDistance:
    """Test FMM signed distance computation."""

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

        fmm = FastMarchingMethod(grid)

        # Circle: phi = sqrt((x-0.5)^2 + (y-0.5)^2) - 0.3
        radius = 0.3
        phi_circle = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - radius

        phi_sdf = fmm.compute_signed_distance(phi_circle)

        # Should be close to input (circle is already SDF)
        error = np.max(np.abs(phi_sdf - phi_circle))
        assert error < 2 * dx, f"Circle SDF error {error} > 2*dx={2 * dx}"

    def test_distorted_level_set_recovery(self):
        """Test recovery of SDF from distorted level set."""
        N = 51
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[N, N],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        X, Y = grid.meshgrid()
        dx = grid.spacing[0]

        fmm = FastMarchingMethod(grid)

        # True circle SDF
        phi_circle = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - 0.3

        # Distort: phi^2 * sign(phi)
        phi_distorted = phi_circle**2 * np.sign(phi_circle)

        phi_sdf = fmm.compute_signed_distance(phi_distorted)

        # Should recover true SDF
        error = np.max(np.abs(phi_sdf - phi_circle))
        assert error < 2 * dx, f"Distorted recovery error {error} > 2*dx={2 * dx}"

    def test_preserves_sign(self):
        """Test that signed distance preserves sign of input."""
        N = 51
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[N, N],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        X, Y = grid.meshgrid()

        fmm = FastMarchingMethod(grid)

        phi_circle = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - 0.3
        phi_sdf = fmm.compute_signed_distance(phi_circle)

        # Signs should match (except at interface)
        mask_positive = phi_circle > 0.01
        mask_negative = phi_circle < -0.01

        assert np.all(phi_sdf[mask_positive] > 0), "Sign not preserved in positive region"
        assert np.all(phi_sdf[mask_negative] < 0), "Sign not preserved in negative region"


class TestFMMGridRefinement:
    """Test FMM convergence with grid refinement."""

    def test_convergence_order(self):
        """Test O(dx) convergence rate."""
        errors = []
        grid_sizes = [25, 50, 100]

        for N in grid_sizes:
            grid = TensorProductGrid(
                bounds=[(0.0, 1.0), (0.0, 1.0)],
                Nx_points=[N, N],
                boundary_conditions=no_flux_bc(dimension=2),
            )
            X, Y = grid.meshgrid()

            fmm = FastMarchingMethod(grid)
            phi = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - 0.3
            phi_sdf = fmm.compute_signed_distance(phi)

            error = np.max(np.abs(phi_sdf - phi))
            errors.append(error)

        # Check convergence: error should halve when grid doubles
        rate = np.log(errors[0] / errors[1]) / np.log(2)
        assert rate > 0.5, f"Convergence rate {rate} < 0.5 (expected ~1.0)"


class TestFMMSpeed:
    """Test FMM with non-unit speed functions."""

    def test_constant_speed(self):
        """Test with constant speed F = 2."""
        N = 51
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[N, N],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        X, Y = grid.meshgrid()
        dx = grid.spacing[0]

        fmm = FastMarchingMethod(grid)

        # Point source at center
        i0, j0 = N // 2, N // 2
        frozen_mask = np.zeros((N, N), dtype=bool)
        frozen_mask[i0, j0] = True
        frozen_values = np.zeros((N, N), dtype=np.float64)

        # Speed = 2 means |grad T| = 0.5
        T = fmm.solve(speed=2.0, frozen_mask=frozen_mask, frozen_values=frozen_values)

        # Analytical: T = |x - x0| / F
        T_exact = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 2.0
        error = np.max(np.abs(T - T_exact))

        assert error < 3 * dx, f"Non-unit speed error {error} > 3*dx={3 * dx}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
