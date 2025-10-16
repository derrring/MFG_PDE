"""
Performance benchmarks for implicit domain geometry infrastructure.

Run with: python -m pytest tests/unit/geometry/test_geometry_benchmarks.py -v --benchmark-only
"""

import pytest

import numpy as np

from mfg_pde.geometry.implicit import (
    DifferenceDomain,
    Hyperrectangle,
    Hypersphere,
)


class TestSamplingPerformance:
    """Benchmark sampling performance across dimensions."""

    def test_hyperrectangle_sampling_2d(self, benchmark):
        """Benchmark 2D hyperrectangle sampling."""
        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        result = benchmark(domain.sample_uniform, 10000, seed=42)
        assert result.shape == (10000, 2)

    def test_hyperrectangle_sampling_5d(self, benchmark):
        """Benchmark 5D hyperrectangle sampling."""
        domain = Hyperrectangle(np.array([[0, 1]] * 5))
        result = benchmark(domain.sample_uniform, 10000, seed=42)
        assert result.shape == (10000, 5)

    def test_hyperrectangle_sampling_10d(self, benchmark):
        """Benchmark 10D hyperrectangle sampling."""
        domain = Hyperrectangle(np.array([[0, 1]] * 10))
        result = benchmark(domain.sample_uniform, 10000, seed=42)
        assert result.shape == (10000, 10)

    def test_hypersphere_sampling_2d(self, benchmark):
        """Benchmark 2D hypersphere sampling (with rejection)."""
        domain = Hypersphere(center=[0, 0], radius=1.0)
        result = benchmark(domain.sample_uniform, 10000, seed=42)
        assert result.shape == (10000, 2)

    def test_hypersphere_sampling_5d(self, benchmark):
        """Benchmark 5D hypersphere sampling (with rejection)."""
        domain = Hypersphere(center=[0] * 5, radius=1.0)
        result = benchmark(domain.sample_uniform, 5000, seed=42)
        assert result.shape == (5000, 5)

    def test_difference_domain_sampling_2d(self, benchmark):
        """Benchmark domain with obstacle sampling."""
        base = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        obstacle = Hypersphere(center=[0.5, 0.5], radius=0.2)
        domain = DifferenceDomain(base, obstacle)
        result = benchmark(domain.sample_uniform, 5000, seed=42)
        assert result.shape[0] <= 5000


class TestSDFPerformance:
    """Benchmark signed distance function evaluation."""

    @pytest.fixture
    def sample_points_2d(self):
        """Generate 10000 random 2D points."""
        np.random.seed(42)
        return np.random.uniform(-2, 2, size=(10000, 2))

    @pytest.fixture
    def sample_points_5d(self):
        """Generate 10000 random 5D points."""
        np.random.seed(42)
        return np.random.uniform(-2, 2, size=(10000, 5))

    @pytest.fixture
    def sample_points_10d(self):
        """Generate 10000 random 10D points."""
        np.random.seed(42)
        return np.random.uniform(-2, 2, size=(10000, 10))

    def test_hyperrectangle_sdf_2d(self, benchmark, sample_points_2d):
        """Benchmark 2D hyperrectangle SDF evaluation."""
        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        result = benchmark(domain.signed_distance, sample_points_2d)
        assert len(result) == 10000

    def test_hyperrectangle_sdf_5d(self, benchmark, sample_points_5d):
        """Benchmark 5D hyperrectangle SDF evaluation."""
        domain = Hyperrectangle(np.array([[0, 1]] * 5))
        result = benchmark(domain.signed_distance, sample_points_5d)
        assert len(result) == 10000

    def test_hyperrectangle_sdf_10d(self, benchmark, sample_points_10d):
        """Benchmark 10D hyperrectangle SDF evaluation."""
        domain = Hyperrectangle(np.array([[0, 1]] * 10))
        result = benchmark(domain.signed_distance, sample_points_10d)
        assert len(result) == 10000

    def test_hypersphere_sdf_2d(self, benchmark, sample_points_2d):
        """Benchmark 2D hypersphere SDF evaluation."""
        domain = Hypersphere(center=[0, 0], radius=1.0)
        result = benchmark(domain.signed_distance, sample_points_2d)
        assert len(result) == 10000

    def test_hypersphere_sdf_5d(self, benchmark, sample_points_5d):
        """Benchmark 5D hypersphere SDF evaluation."""
        domain = Hypersphere(center=[0] * 5, radius=1.0)
        result = benchmark(domain.signed_distance, sample_points_5d)
        assert len(result) == 10000

    def test_hypersphere_sdf_10d(self, benchmark, sample_points_10d):
        """Benchmark 10D hypersphere SDF evaluation."""
        domain = Hypersphere(center=[0] * 10, radius=1.0)
        result = benchmark(domain.signed_distance, sample_points_10d)
        assert len(result) == 10000

    def test_difference_domain_sdf_2d(self, benchmark, sample_points_2d):
        """Benchmark domain with obstacle SDF evaluation."""
        base = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        obstacle = Hypersphere(center=[0.5, 0.5], radius=0.2)
        domain = DifferenceDomain(base, obstacle)
        result = benchmark(domain.signed_distance, sample_points_2d)
        assert len(result) == 10000


class TestContainmentPerformance:
    """Benchmark containment checking."""

    @pytest.fixture
    def sample_points_large(self):
        """Generate 50000 random 2D points."""
        np.random.seed(42)
        return np.random.uniform(-2, 2, size=(50000, 2))

    def test_hyperrectangle_contains(self, benchmark, sample_points_large):
        """Benchmark hyperrectangle containment check."""
        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        result = benchmark(domain.contains, sample_points_large)
        assert len(result) == 50000

    def test_hypersphere_contains(self, benchmark, sample_points_large):
        """Benchmark hypersphere containment check."""
        domain = Hypersphere(center=[0, 0], radius=1.0)
        result = benchmark(domain.contains, sample_points_large)
        assert len(result) == 50000

    def test_difference_domain_contains(self, benchmark, sample_points_large):
        """Benchmark domain with obstacle containment check."""
        base = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        obstacle = Hypersphere(center=[0.5, 0.5], radius=0.2)
        domain = DifferenceDomain(base, obstacle)
        result = benchmark(domain.contains, sample_points_large)
        assert len(result) == 50000


class TestDimensionalScaling:
    """Test how performance scales with dimension."""

    @pytest.mark.parametrize("dim", [2, 3, 4, 5, 6, 8, 10])
    def test_sampling_scaling(self, benchmark, dim):
        """Test how sampling performance scales with dimension."""
        domain = Hyperrectangle(np.array([[0, 1]] * dim))
        result = benchmark(domain.sample_uniform, 1000, seed=42)
        assert result.shape == (1000, dim)

    @pytest.mark.parametrize("dim", [2, 3, 4, 5, 6, 8, 10])
    def test_sdf_scaling(self, benchmark, dim):
        """Test how SDF evaluation scales with dimension."""
        np.random.seed(42)
        points = np.random.uniform(-2, 2, size=(5000, dim))
        domain = Hyperrectangle(np.array([[0, 1]] * dim))
        result = benchmark(domain.signed_distance, points)
        assert len(result) == 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
