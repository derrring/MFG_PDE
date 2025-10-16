"""
Quick tests for implicit domain geometry infrastructure.

Run with: python -m pytest tests/unit/geometry/test_geometry.py -v
"""

import pytest

import numpy as np

from mfg_pde.geometry.implicit import (
    ComplementDomain,
    DifferenceDomain,
    Hyperrectangle,
    Hypersphere,
    IntersectionDomain,
    UnionDomain,
)


class TestHyperrectangle:
    """Test hyperrectangle domain."""

    def test_2d_unit_square(self):
        """Test 2D unit square [0,1]²."""
        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))

        assert domain.dimension == 2
        assert domain.contains(np.array([0.5, 0.5]))  # Interior
        assert domain.contains(np.array([1.0, 1.0]))  # Boundary
        assert not domain.contains(np.array([1.5, 0.5]))  # Exterior

    def test_4d_hypercube(self):
        """Test 4D unit hypercube."""
        domain = Hyperrectangle(np.array([[0, 1]] * 4))

        assert domain.dimension == 4
        particles = domain.sample_uniform(1000, seed=42)
        assert particles.shape == (1000, 4)
        assert np.all(domain.contains(particles))

    def test_signed_distance(self):
        """Test exact signed distance function."""
        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))

        # Interior: should be negative
        assert domain.signed_distance(np.array([0.5, 0.5])) < 0

        # Boundary: should be zero
        assert np.abs(domain.signed_distance(np.array([1.0, 0.5]))) < 1e-10

        # Exterior: should be positive
        assert domain.signed_distance(np.array([1.5, 0.5])) > 0

    def test_volume(self):
        """Test volume calculation."""
        domain = Hyperrectangle(np.array([[0, 2], [0, 3]]))
        assert np.abs(domain.compute_volume() - 6.0) < 1e-10  # 2 × 3


class TestHypersphere:
    """Test hypersphere domain."""

    def test_2d_unit_circle(self):
        """Test 2D unit circle."""
        domain = Hypersphere(center=[0, 0], radius=1.0)

        assert domain.dimension == 2
        assert domain.contains(np.array([0, 0]))  # Center
        assert domain.contains(np.array([0.5, 0]))  # Interior
        assert not domain.contains(np.array([2, 0]))  # Exterior

    def test_4d_hypersphere(self):
        """Test 4D hypersphere."""
        domain = Hypersphere(center=[0] * 4, radius=1.0)

        assert domain.dimension == 4
        particles = domain.sample_uniform(1000, seed=42)
        assert particles.shape == (1000, 4)
        assert np.all(domain.contains(particles))

    def test_signed_distance(self):
        """Test exact signed distance function."""
        domain = Hypersphere(center=[0, 0], radius=1.0)

        # Center: should be -radius
        assert np.abs(domain.signed_distance(np.array([0, 0])) + 1.0) < 1e-10

        # Boundary: should be zero
        assert np.abs(domain.signed_distance(np.array([1, 0]))) < 1e-10

        # Exterior: should be positive
        assert domain.signed_distance(np.array([2, 0])) > 0

    def test_volume_2d(self):
        """Test volume (area) of 2D circle."""
        circle = Hypersphere(center=[0, 0], radius=1.0)
        volume = circle.compute_volume()
        assert np.abs(volume - np.pi) < 1e-6

    def test_volume_3d(self):
        """Test volume of 3D sphere."""
        sphere = Hypersphere(center=[0, 0, 0], radius=1.0)
        volume = sphere.compute_volume()
        expected = 4 * np.pi / 3
        assert np.abs(volume - expected) < 1e-6


class TestCSGOperations:
    """Test CSG operations."""

    def test_union_2_circles(self):
        """Test union of two circles."""
        circle1 = Hypersphere(center=[0, 0], radius=1.0)
        circle2 = Hypersphere(center=[1, 0], radius=1.0)
        union = UnionDomain([circle1, circle2])

        # Point in overlap region
        assert union.contains(np.array([0.5, 0]))

        # Points in individual circles
        assert union.contains(np.array([-0.5, 0]))  # Only in circle1
        assert union.contains(np.array([1.5, 0]))  # Only in circle2

        # Point outside both
        assert not union.contains(np.array([3, 0]))

    def test_intersection_rect_circle(self):
        """Test intersection of rectangle and circle."""
        rect = Hyperrectangle(np.array([[0, 2], [0, 2]]))
        circle = Hypersphere(center=[1, 1], radius=0.8)  # Smaller radius
        intersection = IntersectionDomain([rect, circle])

        # Point in both
        assert intersection.contains(np.array([1, 1]))

        # Point in rect but not circle
        assert not intersection.contains(np.array([0.1, 0.1]))

        # Point in circle but not rect
        assert not intersection.contains(np.array([-0.5, 1]))

    def test_complement(self):
        """Test complement of circle."""
        circle = Hypersphere(center=[0, 0], radius=1.0)
        exterior = ComplementDomain(circle)

        # Inside circle → outside complement
        assert not exterior.contains(np.array([0, 0]))

        # Outside circle → inside complement
        assert exterior.contains(np.array([2, 0]))

    def test_difference_rect_minus_circle(self):
        """Test rectangle with circular hole."""
        rect = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        hole = Hypersphere(center=[0.5, 0.5], radius=0.2)
        domain = DifferenceDomain(rect, hole)

        # Point in rectangle but outside hole
        assert domain.contains(np.array([0.1, 0.1]))

        # Point in hole
        assert not domain.contains(np.array([0.5, 0.5]))

        # Point outside rectangle
        assert not domain.contains(np.array([1.5, 0.5]))

    def test_complex_domain_obstacle_avoidance(self):
        """Test domain with obstacle (key for Stage 2!)."""
        # Base: unit square
        base = Hyperrectangle(np.array([[0, 1], [0, 1]]))

        # Obstacle: circular
        obstacle = Hypersphere(center=[0.5, 0.5], radius=0.2)

        # Navigable domain
        domain = DifferenceDomain(base, obstacle)

        # Sample particles
        particles = domain.sample_uniform(1000, seed=42)

        # Verify all particles avoid obstacle
        dists = np.linalg.norm(particles - np.array([0.5, 0.5]), axis=1)
        assert np.all(dists > 0.2)  # All particles outside obstacle

        # Verify all particles in base domain
        assert np.all((particles >= 0) & (particles <= 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
