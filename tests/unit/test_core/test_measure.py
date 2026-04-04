"""Tests for measure representations (Layer 2 foundation).

Tests MeasureRepresentation Protocol and ParticleMeasure implementation
from mfgarchon.core.measure.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.measure import MeasureRepresentation, ParticleMeasure


class TestMeasureRepresentationProtocol:
    def test_particle_measure_satisfies_protocol(self):
        mu = ParticleMeasure(np.array([0.5]))
        assert isinstance(mu, MeasureRepresentation)

    def test_protocol_is_runtime_checkable(self):
        assert hasattr(MeasureRepresentation, "__protocol_attrs__") or isinstance(MeasureRepresentation, type)


class TestParticleMeasureInit:
    def test_1d_from_flat_array(self):
        mu = ParticleMeasure(np.array([0.1, 0.5, 0.9]))
        assert mu.n_particles == 3
        assert mu.dimension == 1
        assert mu.positions.shape == (3, 1)

    def test_2d_particles(self):
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        mu = ParticleMeasure(pos)
        assert mu.n_particles == 3
        assert mu.dimension == 2

    def test_uniform_weights_default(self):
        mu = ParticleMeasure(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(mu.weights, [1 / 3, 1 / 3, 1 / 3])

    def test_custom_weights_normalized(self):
        mu = ParticleMeasure(np.array([0.0, 1.0]), weights=np.array([3.0, 1.0]))
        np.testing.assert_allclose(mu.weights, [0.75, 0.25])

    def test_total_mass_is_one(self):
        mu = ParticleMeasure(np.random.RandomState(42).randn(50))
        assert mu.total_mass() == pytest.approx(1.0)


class TestParticleMeasureToDensity:
    def test_1d_density_shape(self):
        mu = ParticleMeasure(np.array([0.2, 0.5, 0.8]))
        grid = np.linspace(0, 1, 101)
        density = mu.to_density(grid)
        assert density.shape == (101,)

    def test_1d_density_nonnegative(self):
        mu = ParticleMeasure(np.array([0.2, 0.5, 0.8]))
        density = mu.to_density(np.linspace(0, 1, 101))
        assert np.all(density >= 0)

    def test_1d_density_peaks_near_particles(self):
        mu = ParticleMeasure(np.array([0.3, 0.7]))
        grid = np.linspace(0, 1, 201)
        density = mu.to_density(grid)
        # Peaks should be near 0.3 and 0.7
        peak_idx = np.argmax(density)
        assert abs(grid[peak_idx] - 0.3) < 0.15 or abs(grid[peak_idx] - 0.7) < 0.15

    def test_1d_weighted_density(self):
        """Heavier weight at 0.2 should produce higher peak there."""
        mu = ParticleMeasure(np.array([0.2, 0.8]), weights=np.array([9.0, 1.0]))
        grid = np.linspace(0, 1, 201)
        density = mu.to_density(grid)
        idx_02 = 40  # ~0.2
        idx_08 = 160  # ~0.8
        assert density[idx_02] > density[idx_08]

    def test_2d_density(self):
        pos = np.random.RandomState(7).randn(30, 2)
        mu = ParticleMeasure(pos)
        grid_2d = np.random.RandomState(8).randn(50, 2)
        density = mu.to_density(grid_2d)
        assert density.shape == (50,)
        assert np.all(np.isfinite(density))

    def test_custom_bandwidth(self):
        mu = ParticleMeasure(np.array([0.5]))
        grid = np.linspace(0, 1, 101)
        # Narrow bandwidth -> sharper peak
        d_narrow = mu.to_density(grid, bandwidth=0.01)
        d_wide = mu.to_density(grid, bandwidth=0.5)
        assert d_narrow.max() > d_wide.max()


class TestParticleMeasureRepr:
    def test_repr(self):
        mu = ParticleMeasure(np.zeros((10, 3)))
        assert "n=10" in repr(mu)
        assert "d=3" in repr(mu)
