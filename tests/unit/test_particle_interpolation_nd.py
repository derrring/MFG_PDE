#!/usr/bin/env python3
"""
Tests for nD particle interpolation utilities.

Validates that particle interpolation functions work for arbitrary dimensions.
"""

import pytest

import numpy as np

from mfg_pde.utils.numerical.particle_interpolation import (
    estimate_kde_bandwidth as adaptive_bandwidth_selection,  # Renamed in main
)
from mfg_pde.utils.numerical.particle_interpolation import (
    interpolate_grid_to_particles,
    interpolate_particles_to_grid,
)


@pytest.mark.skip(reason="4D/5D interpolation not yet implemented - particle_interpolation.py line 172")
class TestGridToParticles4D:
    """Test interpolate_grid_to_particles for 4D grids."""

    def test_4d_linear_interpolation(self):
        """Test 4D grid to particle interpolation with linear method."""
        # Create simple 4D grid
        grid_bounds = ((0, 1), (0, 1), (0, 1), (0, 1))
        n_per_dim = 5
        grid_values = np.random.rand(n_per_dim, n_per_dim, n_per_dim, n_per_dim)

        # Create particle positions
        np.random.seed(42)
        n_particles = 10
        particle_positions = np.random.rand(n_particles, 4)

        # Interpolate
        result = interpolate_grid_to_particles(
            grid_values=grid_values, grid_bounds=grid_bounds, particle_positions=particle_positions, method="linear"
        )

        # Check shape
        assert result.shape == (n_particles,)

        # Check values are reasonable (between min and max of grid)
        assert np.all(result >= grid_values.min())
        assert np.all(result <= grid_values.max())

    def test_4d_nearest_interpolation(self):
        """Test 4D grid to particle interpolation with nearest method."""
        grid_bounds = ((0, 1), (0, 1), (0, 1), (0, 1))
        n_per_dim = 5
        grid_values = np.random.rand(n_per_dim, n_per_dim, n_per_dim, n_per_dim)

        np.random.seed(42)
        n_particles = 10
        particle_positions = np.random.rand(n_particles, 4)

        result = interpolate_grid_to_particles(
            grid_values=grid_values, grid_bounds=grid_bounds, particle_positions=particle_positions, method="nearest"
        )

        assert result.shape == (n_particles,)
        # Nearest should give exact grid values
        assert np.all(np.isin(result, grid_values))

    def test_5d_interpolation(self):
        """Test that 5D interpolation works (verifying true nD support)."""
        grid_bounds = tuple((0, 1) for _ in range(5))
        n_per_dim = 4
        grid_values = np.random.rand(*([n_per_dim] * 5))

        np.random.seed(42)
        n_particles = 10
        particle_positions = np.random.rand(n_particles, 5)

        result = interpolate_grid_to_particles(
            grid_values=grid_values, grid_bounds=grid_bounds, particle_positions=particle_positions, method="linear"
        )

        assert result.shape == (n_particles,)
        assert np.all(result >= grid_values.min())
        assert np.all(result <= grid_values.max())


@pytest.mark.skip(reason="4D/5D interpolation not yet implemented - particle_interpolation.py line 172")
class TestParticlesToGrid4D:
    """Test interpolate_particles_to_grid for 4D grids."""

    def test_4d_kde_density_estimation(self):
        """Test 4D particle to grid interpolation using KDE."""
        # Generate random particles in 4D
        np.random.seed(42)
        n_particles = 100
        particle_positions = np.random.rand(n_particles, 4)

        # Create grid
        n_per_dim = 8
        grid_points = tuple(np.linspace(0, 1, n_per_dim) for _ in range(4))

        # Estimate density
        density = interpolate_particles_to_grid(
            particle_positions=particle_positions, grid_points=grid_points, method="kde", normalize=True
        )

        # Check shape
        assert density.shape == (n_per_dim,) * 4

        # Check normalization (integral should be ≈ 1)
        cell_volume = (1.0 / (n_per_dim - 1)) ** 4
        total_mass = np.sum(density) * cell_volume
        assert np.abs(total_mass - 1.0) < 0.5  # Allow some error due to boundary effects

    def test_4d_histogram_method(self):
        """Test 4D particle to grid with histogram method."""
        np.random.seed(42)
        n_particles = 100
        particle_positions = np.random.rand(n_particles, 4)

        n_per_dim = 8
        grid_points = tuple(np.linspace(0, 1, n_per_dim) for _ in range(4))

        density = interpolate_particles_to_grid(
            particle_positions=particle_positions, grid_points=grid_points, method="histogram", normalize=False
        )

        # Check shape
        assert density.shape == (n_per_dim - 1,) * 4  # Histogram has one less bin than grid points

        # Check total count
        assert np.sum(density) == n_particles

    def test_4d_with_values(self):
        """Test 4D particle to grid interpolation with particle values."""
        np.random.seed(42)
        n_particles = 100
        particle_positions = np.random.rand(n_particles, 4)
        particle_values = np.sin(2 * np.pi * np.sum(particle_positions, axis=1))

        n_per_dim = 8
        grid_points = tuple(np.linspace(0, 1, n_per_dim) for _ in range(4))

        grid_values = interpolate_particles_to_grid(
            particle_positions=particle_positions,
            grid_points=grid_points,
            particle_values=particle_values,
            method="kde",
            normalize=False,
        )

        # Check shape
        assert grid_values.shape == (n_per_dim,) * 4

        # Check values are in reasonable range
        assert np.all(grid_values >= particle_values.min() - 1.0)
        assert np.all(grid_values <= particle_values.max() + 1.0)


@pytest.mark.skip(reason="4D/5D interpolation not yet implemented - particle_interpolation.py line 172")
class TestAdaptiveBandwidth:
    """Test adaptive bandwidth selection for nD data."""

    def test_scott_rule_4d(self):
        """Test Scott's rule for 4D data."""
        np.random.seed(42)
        data = np.random.randn(1000, 4)

        bandwidth = adaptive_bandwidth_selection(data, method="scott")

        # Check bandwidth is positive and reasonable
        assert bandwidth > 0
        assert bandwidth < 1.0  # Should be small for normalized data

    def test_silverman_rule_4d(self):
        """Test Silverman's rule for 4D data."""
        np.random.seed(42)
        data = np.random.randn(1000, 4)

        bandwidth = adaptive_bandwidth_selection(data, method="silverman")

        # Check bandwidth is positive and reasonable
        assert bandwidth > 0
        assert bandwidth < 1.0


@pytest.mark.skip(reason="4D/5D interpolation not yet implemented - particle_interpolation.py line 172")
class TestRoundTrip:
    """Test round-trip: grid → particles → grid."""

    def test_4d_round_trip_kde(self):
        """Test that grid → particles → grid preserves overall structure."""
        # Create smooth 4D function on grid
        n_per_dim = 6
        coords = [np.linspace(0, 1, n_per_dim) for _ in range(4)]
        grids = np.meshgrid(*coords, indexing="ij")
        # Create smooth function: sum of coordinates squared
        grid_values_original = sum(g**2 for g in grids) / 4.0

        # Grid to particles
        grid_bounds = tuple((0, 1) for _ in range(4))
        np.random.seed(42)
        n_particles = 500
        particle_positions = np.random.rand(n_particles, 4)

        particle_values = interpolate_grid_to_particles(
            grid_values=grid_values_original, grid_bounds=grid_bounds, particle_positions=particle_positions
        )

        # Particles back to grid
        grid_values_reconstructed = interpolate_particles_to_grid(
            particle_positions=particle_positions,
            grid_points=tuple(coords),
            particle_values=particle_values,
            method="kde",
            normalize=False,
        )

        # Check correlation (should be high for smooth function)
        correlation = np.corrcoef(grid_values_original.ravel(), grid_values_reconstructed.ravel())[0, 1]
        assert correlation > 0.8  # Allow some loss due to interpolation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
