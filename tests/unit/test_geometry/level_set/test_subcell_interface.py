"""
Unit tests for subcell interface extraction.

Tests the O(dx^2) subcell precision method vs O(dx) argmin method.
Expected impact: Stefan problem error reduction from 19.58% to < 3%.
"""

import pytest

import numpy as np

from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import no_flux_bc
from mfg_pde.geometry.level_set import LevelSetFunction


class TestSubcellInterface1D:
    """Test 1D subcell interface extraction via linear interpolation."""

    def test_linear_interface_exact(self):
        """Linear level set should give exact interface location."""
        # Grid and linear level set: phi = x - 0.5
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[100], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        phi = x - 0.5  # Interface at x = 0.5

        ls = LevelSetFunction(phi, grid, is_signed_distance=True)

        # Extract interface
        x_interface = ls.get_interface_location_subcell()

        # Should be exact (within floating point tolerance)
        assert np.abs(x_interface - 0.5) < 1e-10, (
            f"Linear interpolation should be exact for linear phi. Got {x_interface:.10f}, expected 0.5"
        )

    def test_accuracy_vs_argmin(self):
        """Subcell method should be more accurate than argmin."""
        # Grid with interface NOT on grid point
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[51], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        dx = x[1] - x[0]

        # Linear interface at x = 0.5 (between grid points)
        phi = x - 0.5

        ls = LevelSetFunction(phi, grid, is_signed_distance=True)

        # Subcell method
        x_subcell = ls.get_interface_location_subcell()

        # Argmin method (O(dx) accuracy)
        x_argmin = x[np.argmin(np.abs(phi))]

        # Errors
        error_subcell = np.abs(x_subcell - 0.5)
        error_argmin = np.abs(x_argmin - 0.5)

        # Subcell should be significantly better
        assert error_subcell < error_argmin, (
            f"Subcell (error={error_subcell:.6f}) should be more accurate than argmin (error={error_argmin:.6f})"
        )

        # Subcell should be near-exact for linear case
        assert error_subcell < 1e-10, f"Subcell should be exact for linear phi, got error={error_subcell:.6e}"

        # Argmin should have O(dx) error
        assert error_argmin < dx, f"Argmin error should be O(dx), got {error_argmin:.6f} vs dx={dx:.6f}"

    def test_quadratic_interface(self):
        """Subcell should improve accuracy for nonlinear level sets."""
        # Quadratic level set: phi = (x - 0.5)^2 - 0.015^2
        # Zero crossing at x = 0.5 - 0.015 = 0.485 (between grid points)
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[100], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        dx = x[1] - x[0]

        phi = (x - 0.5) ** 2 - 0.015**2
        exact_interface = 0.5 - 0.015  # Left zero crossing at x = 0.485

        ls = LevelSetFunction(phi, grid, is_signed_distance=True)

        # Extract interface
        x_subcell = ls.get_interface_location_subcell()

        # Should be close to analytical (within O(dx^2))
        error = np.abs(x_subcell - exact_interface)
        assert error < 10 * dx**2, (
            f"Subcell error should be O(dx^2) for smooth interface. Got error={error:.6e}, 10*dx^2={10 * dx**2:.6e}"
        )

        # Compare to argmin
        x_argmin = x[np.argmin(np.abs(phi))]
        error_argmin = np.abs(x_argmin - exact_interface)

        # Subcell should be better (or at worst equal if argmin happens to be exact)
        assert error <= error_argmin + 1e-12, (
            f"Subcell (error={error:.6e}) should not be worse than argmin (error={error_argmin:.6e})"
        )

    def test_no_zero_crossing_raises(self):
        """Should raise ValueError if no zero crossing exists."""
        # All positive values
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[50], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        phi = x + 1.0  # All positive

        ls = LevelSetFunction(phi, grid, is_signed_distance=True)

        # Should raise ValueError
        with pytest.raises(ValueError, match="No zero crossing found"):
            ls.get_interface_location_subcell()

    def test_multiple_crossings_uses_first(self):
        """If multiple crossings, should use the first one."""
        # Create level set with multiple crossings
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[200], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]

        # phi = sin(4π(x - 0.1)) starts negative, crosses at x ≈ 0.1, 0.35, 0.6, 0.85
        phi = np.sin(4 * np.pi * (x - 0.1))

        ls = LevelSetFunction(phi, grid, is_signed_distance=False)

        # Extract interface
        x_interface = ls.get_interface_location_subcell()

        # Should be near first zero crossing (x ≈ 0.1)
        assert 0.08 < x_interface < 0.12, f"Should use first zero crossing (expect ~0.1), got {x_interface:.4f}"

    def test_interface_at_grid_point(self):
        """If interface is exactly on grid point, should handle correctly."""
        # Interface exactly at x = 0.5 (grid point)
        grid = TensorProductGrid(
            bounds=[(0, 1)], Nx=[100], boundary_conditions=no_flux_bc(dimension=1)
        )  # 101 points, 0.5 is on grid
        x = grid.coordinates[0]
        phi = x - 0.5

        ls = LevelSetFunction(phi, grid, is_signed_distance=True)

        # Should still work and give exact answer
        x_interface = ls.get_interface_location_subcell()
        assert np.abs(x_interface - 0.5) < 1e-10

    def test_fine_grid_convergence(self):
        """Verify O(dx^2) convergence rate for subcell method."""
        # Use quadratic level set for known exact solution
        # phi = (x - 0.5)^2 - 0.0173^2, zero at x = 0.4827 (not on grid for any resolution)

        exact_interface = 0.5 - 0.0173  # x = 0.4827
        resolutions = [50, 100, 200, 400]
        errors = []

        for N in resolutions:
            grid = TensorProductGrid(bounds=[(0, 1)], Nx=[N], boundary_conditions=no_flux_bc(dimension=1))
            x = grid.coordinates[0]
            phi = (x - 0.5) ** 2 - 0.0173**2

            ls = LevelSetFunction(phi, grid, is_signed_distance=False)

            x_subcell = ls.get_interface_location_subcell()
            error = np.abs(x_subcell - exact_interface)
            errors.append(error)

        # Check convergence rate (should be ~2)
        # error(N) ≈ C · (1/N)^2
        # log(error) ≈ log(C) - 2·log(N)
        # Slope of log-log plot should be ~-2

        # Filter out zero errors (machine precision hits)
        valid_errors = [e for e in errors if e > 1e-14]
        if len(valid_errors) < 3:
            # Too many near-zero errors, can't measure convergence rate
            # Just check that errors decrease
            assert errors[-1] < errors[0], "Errors should decrease with refinement"
            return

        log_N = np.log(resolutions[: len(valid_errors)])
        log_err = np.log(valid_errors)

        # Linear fit to log-log data
        coeffs = np.polyfit(log_N, log_err, 1)
        slope = coeffs[0]

        # Slope should be close to -2 (allowing some tolerance)
        assert -2.5 < slope < -1.5, f"Convergence rate should be ~2 (slope ~-2), got slope={slope:.2f}"


def _circle_grid(n: int) -> tuple[TensorProductGrid, np.ndarray, np.ndarray]:
    """Helper: create 2D grid and meshgrid arrays."""
    grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[n, n], boundary_conditions=no_flux_bc(dimension=2))
    X, Y = grid.meshgrid()
    return grid, X, Y


class TestSubcellInterface2D:
    """Test 2D subcell interface extraction via edge interpolation."""

    def test_circle_interface_accuracy(self):
        """Extracted points should lie within O(dx^2) of the true circle."""
        N = 80
        grid, X, Y = _circle_grid(N)
        dx = 1.0 / N

        center, radius = (0.5, 0.5), 0.3
        phi = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius
        ls = LevelSetFunction(phi, grid, is_signed_distance=True)

        points = ls.get_interface_location_subcell()

        # Each point's distance to center should be close to radius
        dist = np.sqrt((points[:, 0] - center[0]) ** 2 + (points[:, 1] - center[1]) ** 2)
        max_error = np.max(np.abs(dist - radius))

        # O(dx^2) accuracy: error should scale with dx^2
        assert max_error < 5 * dx**2, f"Max distance error {max_error:.2e} exceeds 5*dx^2={5 * dx**2:.2e}"

    def test_circle_point_count(self):
        """Number of interface points should scale with circumference / dx."""
        N = 60
        grid, X, Y = _circle_grid(N)
        dx = 1.0 / N

        radius = 0.25
        phi = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - radius
        ls = LevelSetFunction(phi, grid, is_signed_distance=True)

        points = ls.get_interface_location_subcell()

        # Expected: ~2*pi*R / dx crossings per axis, times 2 axes
        # but with deduplication from both axes, the count is roughly
        # 2 * 2*pi*R / dx (each axis finds its own crossings)
        expected_per_axis = 2 * np.pi * radius / dx
        n_points = len(points)

        # Should be in the right ballpark (0.5x to 4x expected per axis)
        assert n_points > 0.5 * expected_per_axis, f"Too few points: {n_points} < {0.5 * expected_per_axis:.0f}"
        assert n_points < 4 * expected_per_axis, f"Too many points: {n_points} > {4 * expected_per_axis:.0f}"

    def test_flat_interface_2d(self):
        """Planar interface phi = x - 0.5 should give all x-coords exactly 0.5."""
        N = 40
        grid, X, _Y = _circle_grid(N)

        phi = X - 0.5
        ls = LevelSetFunction(phi, grid, is_signed_distance=True)

        points = ls.get_interface_location_subcell()

        # All x-coordinates should be exactly 0.5 (linear interpolation is exact)
        assert np.allclose(points[:, 0], 0.5, atol=1e-12), (
            f"x-coords should all be 0.5, got range [{points[:, 0].min()}, {points[:, 0].max()}]"
        )

    def test_no_crossing_raises_2d(self):
        """ValueError for all-positive phi in 2D."""
        N = 20
        grid, X, Y = _circle_grid(N)

        phi = X + Y + 1.0  # All positive
        ls = LevelSetFunction(phi, grid, is_signed_distance=False)

        with pytest.raises(ValueError, match="No zero crossing found"):
            ls.get_interface_location_subcell()

    def test_convergence_2d_circle(self):
        """Verify O(dx^2) convergence rate for 2D circle interface."""
        center, radius = (0.5, 0.5), 0.3
        resolutions = [40, 80, 160]
        max_errors = []

        for N in resolutions:
            grid, X, Y = _circle_grid(N)
            phi = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius
            ls = LevelSetFunction(phi, grid, is_signed_distance=True)

            points = ls.get_interface_location_subcell()
            dist = np.sqrt((points[:, 0] - center[0]) ** 2 + (points[:, 1] - center[1]) ** 2)
            max_errors.append(np.max(np.abs(dist - radius)))

        # Measure convergence rate via log-log slope
        log_dx = np.log([1.0 / N for N in resolutions])
        log_err = np.log(max_errors)
        slope = np.polyfit(log_dx, log_err, 1)[0]

        # Slope should be ~2 (O(dx^2)); allow range [1.5, 3.0]
        assert 1.5 < slope < 3.0, f"Convergence rate should be ~2 (slope ~2 in log(err) vs log(dx)), got {slope:.2f}"


class TestSubcellInterface3D:
    """Test 3D subcell interface extraction."""

    def test_sphere_interface_points(self):
        """Extracted points should lie near the analytical sphere surface."""
        N = 20
        grid = TensorProductGrid(
            bounds=[(0, 1), (0, 1), (0, 1)],
            Nx=[N, N, N],
            boundary_conditions=no_flux_bc(dimension=3),
        )
        X, Y, Z = grid.meshgrid()
        dx = 1.0 / N

        center, radius = (0.5, 0.5, 0.5), 0.3
        phi = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2) - radius
        ls = LevelSetFunction(phi, grid, is_signed_distance=True)

        points = ls.get_interface_location_subcell()

        assert points.shape[1] == 3, f"Expected 3 columns, got {points.shape[1]}"
        assert len(points) > 0, "Expected at least some interface points"

        dist = np.sqrt(
            (points[:, 0] - center[0]) ** 2 + (points[:, 1] - center[1]) ** 2 + (points[:, 2] - center[2]) ** 2
        )
        max_error = np.max(np.abs(dist - radius))

        # Coarse 3D grid: relax to 10*dx^2
        assert max_error < 10 * dx**2, f"Max distance error {max_error:.2e} exceeds 10*dx^2={10 * dx**2:.2e}"


if __name__ == "__main__":
    """Run tests directly for development."""
    print("Testing subcell interface extraction...")

    # 1D tests
    test_1d = TestSubcellInterface1D()
    for name in [
        "test_linear_interface_exact",
        "test_accuracy_vs_argmin",
        "test_quadratic_interface",
        "test_no_zero_crossing_raises",
        "test_multiple_crossings_uses_first",
        "test_interface_at_grid_point",
        "test_fine_grid_convergence",
    ]:
        getattr(test_1d, name)()
        print(f"  PASS {name}")

    # 2D tests
    test_2d = TestSubcellInterface2D()
    for name in [
        "test_circle_interface_accuracy",
        "test_circle_point_count",
        "test_flat_interface_2d",
        "test_no_crossing_raises_2d",
        "test_convergence_2d_circle",
    ]:
        getattr(test_2d, name)()
        print(f"  PASS {name}")

    # 3D test
    test_3d = TestSubcellInterface3D()
    test_3d.test_sphere_interface_points()
    print("  PASS test_sphere_interface_points")

    print("\nAll subcell interface tests passed!")
