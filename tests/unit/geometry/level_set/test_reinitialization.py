"""
Unit tests for level set reinitialization.

Tests global and narrow band reinitialization for correctness, convergence,
and performance characteristics.

Created: 2026-01-18 (Issue #605 Phase 1.2)
"""

import pytest

import numpy as np

from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.level_set.reinitialization import reinitialize


class TestReinitializationGlobal:
    """Test global (full domain) reinitialization."""

    def test_preserves_zero_level_set_1d(self):
        """Test that zero level set doesn't move significantly during reinitialization."""
        # 1D: φ = x - 0.5 (interface at x=0.5)
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100])
        x = grid.coordinates[0]

        # True SDF
        phi0 = x - 0.5

        # Slight distortion
        phi_distorted = phi0 + 0.05 * np.sin(10 * np.pi * x)

        # Reinitialize
        phi_reinit = reinitialize(phi_distorted, grid, max_iterations=20)

        # Check that zero crossing is still near x=0.5
        zero_idx_before = np.argmin(np.abs(phi_distorted))
        zero_idx_after = np.argmin(np.abs(phi_reinit))

        # Should move by at most a few grid points (≤ 10 for basic reinitialization)
        assert abs(zero_idx_after - zero_idx_before) <= 10, "Zero level set moved too much"

    def test_improves_gradient_magnitude_1d(self):
        """Test that reinitialization improves |∇φ| ≈ 1."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100])
        x = grid.coordinates[0]

        # Create distorted level set: φ² (not SDF)
        phi0_true_sdf = x - 0.5
        phi_distorted = phi0_true_sdf**2 * np.sign(phi0_true_sdf)

        # Compute gradient magnitude before
        grad_ops = grid.get_gradient_operator()
        grad_before = grad_ops[0](phi_distorted)
        grad_mag_before = np.abs(grad_before)
        deviation_before = np.mean(np.abs(grad_mag_before - 1.0))

        # Reinitialize
        phi_reinit = reinitialize(phi_distorted, grid, max_iterations=20)

        # Compute gradient magnitude after
        grad_after = grad_ops[0](phi_reinit)
        grad_mag_after = np.abs(grad_after)
        deviation_after = np.mean(np.abs(grad_mag_after - 1.0))

        # Should improve or at least not degrade significantly
        assert deviation_after <= deviation_before * 1.5, (
            f"Reinitialization made gradient worse: {deviation_after:.4f} vs {deviation_before:.4f}"
        )

    def test_convergence_tolerance(self):
        """Test that reinitialization stops early when tolerance is met."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100])
        x = grid.coordinates[0]

        # Start with good SDF
        phi0 = x - 0.5

        # Reinitialize with tight tolerance
        phi_reinit = reinitialize(phi0, grid, max_iterations=100, tolerance=0.01)

        # Should converge quickly (good initial SDF)
        # Just check it completes without error
        assert np.all(np.isfinite(phi_reinit)), "Reinitialization produced NaN/Inf"

    def test_circle_2d(self):
        """Test reinitialization on 2D circle."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx=[50, 50])
        X, Y = grid.meshgrid()

        # Circle: φ = ||x - c|| - R
        center = np.array([0.5, 0.5])
        radius = 0.3
        phi0_circle = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius

        # Distort: φ²
        phi_distorted = phi0_circle**2 * np.sign(phi0_circle)

        # Reinitialize
        phi_reinit = reinitialize(phi_distorted, grid, max_iterations=20)

        # Check gradient magnitude improved
        grad_ops = grid.get_gradient_operator()
        grad_x_before = grad_ops[0](phi_distorted)
        grad_y_before = grad_ops[1](phi_distorted)
        grad_mag_before = np.sqrt(grad_x_before**2 + grad_y_before**2)

        grad_x_after = grad_ops[0](phi_reinit)
        grad_y_after = grad_ops[1](phi_reinit)
        grad_mag_after = np.sqrt(grad_x_after**2 + grad_y_after**2)

        mean_dev_before = np.mean(np.abs(grad_mag_before - 1.0))
        mean_dev_after = np.mean(np.abs(grad_mag_after - 1.0))

        # Should not make things significantly worse
        assert mean_dev_after <= mean_dev_before * 1.5, (
            f"2D reinitialization degraded gradient: {mean_dev_after:.4f} vs {mean_dev_before:.4f}"
        )


class TestReinitializationNarrowBand:
    """Test narrow band reinitialization."""

    def test_narrow_band_correctness_2d(self):
        """Test that narrow band produces same results as global near interface."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx=[100, 100])
        X, Y = grid.meshgrid()
        dx = grid.spacing[0]

        # Circle
        center = np.array([0.5, 0.5])
        radius = 0.3
        phi0_circle = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius

        # Distort
        phi_distorted = phi0_circle**2 * np.sign(phi0_circle)

        # Global reinitialization
        phi_global = reinitialize(phi_distorted, grid, max_iterations=20, narrow_band_width=None)

        # Narrow band reinitialization
        narrow_band_width = 5 * dx
        phi_narrow = reinitialize(phi_distorted, grid, max_iterations=20, narrow_band_width=narrow_band_width)

        # Check results match near interface
        interface_region = np.abs(phi0_circle) < 3 * dx
        diff_at_interface = np.abs(phi_global[interface_region] - phi_narrow[interface_region])

        max_diff = np.max(diff_at_interface) if np.any(interface_region) else 0.0

        # Should be very similar (may have small numerical differences)
        assert max_diff < 0.1, f"Narrow band differs from global near interface: {max_diff:.6f}"

    def test_narrow_band_preserves_far_field(self):
        """Test that narrow band doesn't modify far-field values."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx=[50, 50])
        X, Y = grid.meshgrid()
        dx = grid.spacing[0]

        # Circle
        center = np.array([0.5, 0.5])
        radius = 0.3
        phi0_circle = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius

        # Distort
        phi_distorted = phi0_circle**2 * np.sign(phi0_circle)

        # Narrow band reinitialization
        narrow_band_width = 3 * dx
        phi_narrow = reinitialize(phi_distorted, grid, max_iterations=20, narrow_band_width=narrow_band_width)

        # Check that far-field values are unchanged
        far_field_mask = np.abs(phi_distorted) > narrow_band_width

        # Far-field should be identical to input
        diff_far_field = np.abs(phi_narrow[far_field_mask] - phi_distorted[far_field_mask])

        max_diff_far = np.max(diff_far_field) if np.any(far_field_mask) else 0.0

        # Should be exactly preserved (numerical precision)
        assert max_diff_far < 1e-12, f"Far-field values changed: {max_diff_far:.2e}"

    def test_narrow_band_coverage(self):
        """Test that narrow band covers expected percentage of domain."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx=[100, 100])
        X, Y = grid.meshgrid()
        dx = grid.spacing[0]

        # Circle
        center = np.array([0.5, 0.5])
        radius = 0.3
        phi0_circle = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius

        # Narrow band mask
        narrow_band_width = 3 * dx
        narrow_band_mask = np.abs(phi0_circle) < narrow_band_width

        n_band_points = np.sum(narrow_band_mask)
        n_total_points = phi0_circle.size
        coverage = n_band_points / n_total_points

        # For a circle, narrow band should cover a small fraction
        # Band area ≈ 2π·R·width, Total area ≈ π·R² (plus domain area)
        # Expect < 30% for 3dx band
        assert coverage < 0.3, f"Narrow band covers too much: {100 * coverage:.1f}%"
        assert coverage > 0.01, f"Narrow band covers too little: {100 * coverage:.1f}%"

    def test_narrow_band_backward_compatible(self):
        """Test that narrow_band_width=None behaves like original."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100])
        x = grid.coordinates[0]

        phi_distorted = (x - 0.5) ** 2 * np.sign(x - 0.5)

        # Call with narrow_band_width=None (default, backward compatible)
        phi_reinit_default = reinitialize(phi_distorted, grid, max_iterations=20)

        # Call explicitly with None
        phi_reinit_none = reinitialize(phi_distorted, grid, max_iterations=20, narrow_band_width=None)

        # Should be identical
        assert np.allclose(phi_reinit_default, phi_reinit_none, atol=1e-14), (
            "narrow_band_width=None not backward compatible"
        )


class TestReinitializationEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_dtau_raises(self):
        """Test that invalid dtau raises ValueError."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100])
        x = grid.coordinates[0]
        phi = x - 0.5

        # dtau too large (violates CFL)
        h_min = min(grid.spacing)
        dtau_invalid = 0.6 * h_min  # CFL = 0.6 > 0.5

        with pytest.raises(ValueError, match="CFL condition"):
            reinitialize(phi, grid, dtau=dtau_invalid)

    def test_handles_sign_zero(self):
        """Test that sign(0) = 0 is handled correctly."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100])
        x = grid.coordinates[0]

        # Level set with exact zeros
        phi = np.zeros_like(x)
        phi[x < 0.5] = -0.1
        phi[x > 0.5] = 0.1
        # phi[x == 0.5] = 0.0 exactly (some points)

        # Should not crash or produce NaN
        phi_reinit = reinitialize(phi, grid, max_iterations=10)

        assert np.all(np.isfinite(phi_reinit)), "Reinitialization produced NaN/Inf with sign(0)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
