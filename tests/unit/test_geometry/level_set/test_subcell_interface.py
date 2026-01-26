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


class TestSubcellInterfaceHigherDimensions:
    """Test error handling for 2D/3D (not yet implemented)."""

    def test_2d_not_implemented(self):
        """2D subcell extraction should raise NotImplementedError."""
        grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[50, 50], boundary_conditions=no_flux_bc(dimension=2))
        X, Y = grid.meshgrid()

        # Circle level set
        phi = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - 0.2

        ls = LevelSetFunction(phi, grid, is_signed_distance=True)

        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="not yet implemented for 2D"):
            ls.get_interface_location_subcell()


if __name__ == "__main__":
    """Run tests directly for development."""
    print("Testing subcell interface extraction...")

    # Test 1: Linear interface
    print("\n[Test 1: Linear Interface Exact]")
    test = TestSubcellInterface1D()
    test.test_linear_interface_exact()
    print("  ✓ Linear interface extraction exact")

    # Test 2: Accuracy vs argmin
    print("\n[Test 2: Accuracy vs Argmin]")
    test.test_accuracy_vs_argmin()
    print("  ✓ Subcell method beats argmin")

    # Test 3: Quadratic
    print("\n[Test 3: Quadratic Interface]")
    test.test_quadratic_interface()
    print("  ✓ O(dx^2) accuracy for smooth interface")

    # Test 4: No crossing
    print("\n[Test 4: No Zero Crossing]")
    test.test_no_zero_crossing_raises()
    print("  ✓ ValueError raised when no crossing")

    # Test 5: Multiple crossings
    print("\n[Test 5: Multiple Crossings]")
    test.test_multiple_crossings_uses_first()
    print("  ✓ Uses first crossing correctly")

    # Test 6: Grid point interface
    print("\n[Test 6: Interface at Grid Point]")
    test.test_interface_at_grid_point()
    print("  ✓ Handles interface on grid point")

    # Test 7: Convergence
    print("\n[Test 7: Convergence Rate]")
    test.test_fine_grid_convergence()
    print("  ✓ Verified O(dx^2) convergence")

    # Test 8: 2D not implemented
    print("\n[Test 8: 2D Not Implemented]")
    test_2d = TestSubcellInterfaceHigherDimensions()
    test_2d.test_2d_not_implemented()
    print("  ✓ 2D raises NotImplementedError")

    print("\n✅ All subcell interface tests passed!")
    print("\nExpected Impact:")
    print("  - Stefan problem error: 19.58% → < 3% (with subcell precision)")
    print("  - Interface location: O(dx) → O(dx^2) accuracy")
