"""
Unit tests for ImplicitDomain region marking.

Tests Issue #549 Gap #2: SupportsRegionMarking for implicit domains.
"""

import pytest

import numpy as np

from mfg_pde.geometry.implicit import Hypersphere


class TestImplicitDomainRegionMarking:
    """Test suite for region marking on implicit domains."""

    def test_mark_region_with_predicate(self):
        """Test marking region using predicate function."""
        circle = Hypersphere(center=[0.5, 0.5], radius=0.3)

        # Mark top half
        circle.mark_region("top_half", predicate=lambda x: x[:, 1] > 0.5)

        # Verify region exists
        assert "top_half" in circle.get_region_names()

    def test_mark_region_with_sdf(self):
        """Test marking region using SDF function."""
        circle = Hypersphere(center=[0.5, 0.5], radius=0.3)

        def sector_sdf(x):
            """SDF for top sector."""
            dist = np.linalg.norm(x - np.array([0.5, 0.5]), axis=-1)
            on_boundary = np.abs(dist - 0.3) < 0.05
            in_top = x[:, 1] > 0.55
            return np.where(on_boundary & in_top, -1.0, 1.0)

        circle.mark_region("exit_sector", sdf_region=sector_sdf)

        assert "exit_sector" in circle.get_region_names()

    def test_get_region_mask_with_points(self):
        """Test evaluating region mask at specific points."""
        circle = Hypersphere(center=[0.5, 0.5], radius=0.3)

        circle.mark_region("top_half", predicate=lambda x: x[:, 1] > 0.5)

        # Test points
        test_points = np.array(
            [
                [0.5, 0.6],  # Top (True)
                [0.5, 0.4],  # Bottom (False)
                [0.5, 0.7],  # Top (True)
            ]
        )

        mask = circle.get_region_mask("top_half", points=test_points)

        assert mask.shape == (3,)
        assert np.array_equal(mask, [True, False, True])

    def test_get_region_mask_without_points(self):
        """Test evaluating region mask at collocation points."""
        circle = Hypersphere(center=[0.5, 0.5], radius=0.3)

        circle.mark_region("top_half", predicate=lambda x: x[:, 1] > 0.5)

        # Without points, uses get_collocation_points()
        mask = circle.get_region_mask("top_half")

        # Should return mask for sampled points
        assert mask.shape[0] > 0
        assert mask.dtype == bool

    def test_intersect_regions(self):
        """Test intersection of multiple regions."""
        circle = Hypersphere(center=[0.5, 0.5], radius=0.3)

        circle.mark_region("top", predicate=lambda x: x[:, 1] > 0.5)
        circle.mark_region("right", predicate=lambda x: x[:, 0] > 0.5)

        # Get combined predicate
        combined = circle.intersect_regions("top", "right")

        # Test points
        test_pts = np.array(
            [
                [0.6, 0.6],  # Top-right (True)
                [0.4, 0.6],  # Top-left (False)
                [0.6, 0.4],  # Bottom-right (False)
                [0.4, 0.4],  # Bottom-left (False)
            ]
        )

        mask = combined(test_pts)
        assert np.array_equal(mask, [True, False, False, False])

    def test_get_region_predicate(self):
        """Test retrieving region predicate directly."""
        circle = Hypersphere(center=[0.5, 0.5], radius=0.3)

        circle.mark_region("top", predicate=lambda x: x[:, 1] > 0.5)

        # Get predicate
        top_pred = circle.get_region_predicate("top")

        # Use predicate
        particles = np.array([[0.5, 0.7], [0.5, 0.3]])
        in_top = top_pred(particles)

        assert np.array_equal(in_top, [True, False])

    def test_get_region_names(self):
        """Test listing all marked regions."""
        circle = Hypersphere(center=[0.5, 0.5], radius=0.3)

        # Initially empty
        assert circle.get_region_names() == []

        # Mark regions
        circle.mark_region("inlet", predicate=lambda x: x[:, 0] < 0.3)
        circle.mark_region("outlet", predicate=lambda x: x[:, 0] > 0.7)

        names = circle.get_region_names()
        assert len(names) == 2
        assert "inlet" in names
        assert "outlet" in names

    def test_error_duplicate_name(self):
        """Test error when marking duplicate region name."""
        circle = Hypersphere(center=[0.5, 0.5], radius=0.3)

        circle.mark_region("top", predicate=lambda x: x[:, 1] > 0.5)

        with pytest.raises(ValueError, match="already exists"):
            circle.mark_region("top", predicate=lambda x: x[:, 1] > 0.8)

    def test_error_missing_region(self):
        """Test error when accessing non-existent region."""
        circle = Hypersphere(center=[0.5, 0.5], radius=0.3)

        with pytest.raises(KeyError, match="not found"):
            circle.get_region_mask("nonexistent")

        with pytest.raises(KeyError, match="not found"):
            circle.get_region_predicate("nonexistent")

    def test_error_multiple_specifications(self):
        """Test error when providing both predicate and sdf_region."""
        circle = Hypersphere(center=[0.5, 0.5], radius=0.3)

        def sdf(x):
            return x[:, 0] - 0.5

        with pytest.raises(ValueError, match="Cannot specify both"):
            circle.mark_region("invalid", predicate=lambda x: x[:, 0] > 0.5, sdf_region=sdf)

    def test_error_no_specification(self):
        """Test error when providing neither predicate nor sdf_region."""
        circle = Hypersphere(center=[0.5, 0.5], radius=0.3)

        with pytest.raises(ValueError, match="Must specify one of"):
            circle.mark_region("invalid")

    def test_sdf_to_predicate_conversion(self):
        """Test automatic conversion of SDF to predicate."""
        circle = Hypersphere(center=[0.5, 0.5], radius=0.3)

        # SDF: negative inside region
        def region_sdf(x):
            return x[:, 0] - 0.6  # x < 0.6 is inside

        circle.mark_region("left_zone", sdf_region=region_sdf)

        # Test: points with x < 0.6 should be in region
        test_pts = np.array(
            [
                [0.5, 0.5],  # x=0.5 < 0.6 → in region
                [0.7, 0.5],  # x=0.7 > 0.6 → not in region
            ]
        )

        mask = circle.get_region_mask("left_zone", points=test_pts)
        assert np.array_equal(mask, [True, False])


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing ImplicitDomain region marking...")

    test = TestImplicitDomainRegionMarking()
    test.test_mark_region_with_predicate()
    test.test_mark_region_with_sdf()
    test.test_get_region_mask_with_points()
    test.test_get_region_mask_without_points()
    test.test_intersect_regions()
    test.test_get_region_predicate()
    test.test_get_region_names()
    test.test_error_duplicate_name()
    test.test_error_missing_region()
    test.test_error_multiple_specifications()
    test.test_error_no_specification()
    test.test_sdf_to_predicate_conversion()

    print("✓ All tests passed!")
