"""
Unit tests for TensorProductGrid class.

Tests comprehensive functionality of tensor product grids including:
- Grid initialization and validation
- Coordinate generation (uniform and custom)
- Meshgrid creation and flattening
- Index conversion (multi <-> flat)
- Grid refinement and coarsening
- Volume element computation
- Spacing queries
"""

import pytest

import numpy as np

from mfg_pde.geometry.grids.tensor_grid import TensorProductGrid

# ============================================================================
# Test Initialization and Validation
# ============================================================================


class TestInitialization:
    """Test grid initialization and parameter validation."""

    def test_1d_uniform_grid(self) -> None:
        """Test 1D uniform grid initialization."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 10.0)], num_points=[101])

        assert grid.dimension == 1
        assert grid.bounds == ((0.0, 10.0),)  # Normalized to tuple
        assert grid.num_points == (101,)  # Normalized to tuple
        assert grid.is_uniform is True
        assert len(grid.coordinates) == 1
        assert len(grid.coordinates[0]) == 101

    def test_2d_uniform_grid(self) -> None:
        """Test 2D uniform grid initialization."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 2.0)], num_points=[11, 21])

        assert grid.dimension == 2
        assert grid.bounds == ((0.0, 1.0), (0.0, 2.0))  # Normalized to tuple
        assert grid.num_points == (11, 21)  # Normalized to tuple
        assert grid.is_uniform is True
        assert len(grid.coordinates) == 2

    def test_3d_uniform_grid(self) -> None:
        """Test 3D uniform grid initialization."""
        grid = TensorProductGrid(dimension=3, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], num_points=[11, 11, 11])

        assert grid.dimension == 3
        assert grid.total_points() == 11 * 11 * 11

    def test_invalid_dimension_raises(self) -> None:
        """Test that invalid dimension raises ValueError."""
        # Dimension must be positive (dimension < 1 should raise)
        with pytest.raises(ValueError, match="Dimension must be positive"):
            TensorProductGrid(dimension=0, bounds=[], num_points=[])

        with pytest.raises(ValueError, match="Dimension must be positive"):
            TensorProductGrid(dimension=-1, bounds=[], num_points=[])

    def test_mismatched_bounds_length_raises(self) -> None:
        """Test that mismatched bounds length raises ValueError."""
        with pytest.raises(ValueError, match="bounds and num_points must have length 2"):
            TensorProductGrid(dimension=2, bounds=[(0.0, 1.0)], num_points=[10, 10])

    def test_mismatched_num_points_length_raises(self) -> None:
        """Test that mismatched num_points length raises ValueError."""
        with pytest.raises(ValueError, match="bounds and num_points must have length 2"):
            TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[10])

    def test_custom_coordinates_grid(self) -> None:
        """Test grid with custom (non-uniform) coordinates."""
        custom_x = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
        custom_y = np.array([0.0, 0.5, 1.0])

        grid = TensorProductGrid(
            dimension=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            num_points=[5, 3],
            spacing_type="custom",
            custom_coordinates=[custom_x, custom_y],
        )

        assert grid.dimension == 2
        assert grid.is_uniform is False
        assert len(grid.coordinates[0]) == 5
        assert len(grid.coordinates[1]) == 3
        assert np.allclose(grid.coordinates[0], custom_x)
        assert np.allclose(grid.coordinates[1], custom_y)

    def test_custom_requires_coordinates(self) -> None:
        """Test that custom spacing type requires custom_coordinates."""
        with pytest.raises(ValueError, match="custom_coordinates required"):
            TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[10, 10], spacing_type="custom")

    def test_unknown_spacing_type_raises(self) -> None:
        """Test that unknown spacing type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown spacing_type"):
            TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[10, 10], spacing_type="invalid")


# ============================================================================
# Test Grid Properties and Methods
# ============================================================================


class TestGridProperties:
    """Test grid properties and basic methods."""

    def test_total_points_1d(self) -> None:
        """Test total_points for 1D grid."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], num_points=[101])

        assert grid.total_points() == 101

    def test_total_points_2d(self) -> None:
        """Test total_points for 2D grid."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[11, 21])

        assert grid.total_points() == 11 * 21

    def test_total_points_3d(self) -> None:
        """Test total_points for 3D grid."""
        grid = TensorProductGrid(dimension=3, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], num_points=[10, 10, 10])

        assert grid.total_points() == 10 * 10 * 10

    def test_uniform_spacing_1d(self) -> None:
        """Test uniform spacing computation in 1D."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 10.0)], num_points=[11])

        assert grid.is_uniform is True
        assert grid.spacing[0] == pytest.approx(1.0)

    def test_uniform_spacing_2d(self) -> None:
        """Test uniform spacing computation in 2D."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 2.0)], num_points=[11, 21])

        assert grid.spacing[0] == pytest.approx(0.1)
        assert grid.spacing[1] == pytest.approx(0.1)

    def test_repr_string(self) -> None:
        """Test string representation."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[10, 10])

        repr_str = repr(grid)
        assert "TensorProductGrid" in repr_str
        assert "dimension=2" in repr_str
        assert "total_points=100" in repr_str


# ============================================================================
# Test Meshgrid and Flattening
# ============================================================================


class TestMeshgridAndFlatten:
    """Test meshgrid creation and grid point flattening."""

    def test_meshgrid_1d(self) -> None:
        """Test meshgrid for 1D grid."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], num_points=[11])

        (X,) = grid.meshgrid()

        assert X.shape == (11,)
        assert X[0] == pytest.approx(0.0)
        assert X[-1] == pytest.approx(1.0)

    def test_meshgrid_2d_ij_indexing(self) -> None:
        """Test meshgrid for 2D grid with 'ij' indexing."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 2.0)], num_points=[3, 4])

        X, Y = grid.meshgrid(indexing="ij")

        assert X.shape == (3, 4)
        assert Y.shape == (3, 4)
        assert X[0, 0] == pytest.approx(0.0)
        assert X[2, 0] == pytest.approx(1.0)
        assert Y[0, 0] == pytest.approx(0.0)
        assert Y[0, 3] == pytest.approx(2.0)

    def test_meshgrid_2d_xy_indexing(self) -> None:
        """Test meshgrid for 2D grid with 'xy' indexing."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 2.0)], num_points=[3, 4])

        X, Y = grid.meshgrid(indexing="xy")

        assert X.shape == (4, 3)
        assert Y.shape == (4, 3)

    def test_flatten_1d(self) -> None:
        """Test flattening for 1D grid."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], num_points=[11])

        points = grid.flatten()

        assert points.shape == (11, 1)
        assert points[0, 0] == pytest.approx(0.0)
        assert points[-1, 0] == pytest.approx(1.0)

    def test_flatten_2d(self) -> None:
        """Test flattening for 2D grid."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 2.0)], num_points=[3, 4])

        points = grid.flatten()

        assert points.shape == (12, 2)  # 3 * 4 points
        # Check corners
        assert np.allclose(points[0], [0.0, 0.0])
        assert np.allclose(points[-1], [1.0, 2.0])

    def test_flatten_3d(self) -> None:
        """Test flattening for 3D grid."""
        grid = TensorProductGrid(dimension=3, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], num_points=[2, 2, 2])

        points = grid.flatten()

        assert points.shape == (8, 3)  # 2 * 2 * 2 points


# ============================================================================
# Test Index Conversion
# ============================================================================


class TestIndexConversion:
    """Test conversion between multi-index and flat index."""

    def test_get_index_2d(self) -> None:
        """Test multi-index to flat index conversion in 2D."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[10, 10])

        # Test corner points
        assert grid.get_index((0, 0)) == 0
        assert grid.get_index((9, 9)) == 99
        assert grid.get_index((5, 3)) == 5 * 10 + 3

    def test_get_index_3d(self) -> None:
        """Test multi-index to flat index conversion in 3D."""
        grid = TensorProductGrid(dimension=3, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], num_points=[10, 10, 10])

        assert grid.get_index((0, 0, 0)) == 0
        assert grid.get_index((9, 9, 9)) == 999
        assert grid.get_index((5, 3, 7)) == 5 * 100 + 3 * 10 + 7

    def test_get_multi_index_2d(self) -> None:
        """Test flat index to multi-index conversion in 2D."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[10, 10])

        assert grid.get_multi_index(0) == (0, 0)
        assert grid.get_multi_index(99) == (9, 9)
        assert grid.get_multi_index(53) == (5, 3)

    def test_get_multi_index_3d(self) -> None:
        """Test flat index to multi-index conversion in 3D."""
        grid = TensorProductGrid(dimension=3, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], num_points=[10, 10, 10])

        assert grid.get_multi_index(0) == (0, 0, 0)
        assert grid.get_multi_index(999) == (9, 9, 9)
        assert grid.get_multi_index(537) == (5, 3, 7)

    def test_index_conversion_roundtrip(self) -> None:
        """Test that index conversions are inverses."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[10, 10])

        # Test several points
        for i in range(10):
            for j in range(10):
                multi = (i, j)
                flat = grid.get_index(multi)
                recovered = grid.get_multi_index(flat)
                assert recovered == multi

    def test_get_index_wrong_dimension_raises(self) -> None:
        """Test that wrong multi_index dimension raises ValueError."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[10, 10])

        with pytest.raises(ValueError, match="multi_index must have length 2"):
            grid.get_index((0, 0, 0))

    def test_get_multi_index_out_of_range_raises(self) -> None:
        """Test that out-of-range flat index raises ValueError."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[10, 10])

        with pytest.raises(ValueError, match="out of range"):
            grid.get_multi_index(100)

        with pytest.raises(ValueError, match="out of range"):
            grid.get_multi_index(-1)


# ============================================================================
# Test Grid Refinement and Coarsening
# ============================================================================


class TestRefinementCoarsening:
    """Test grid refinement and coarsening operations."""

    def test_refine_uniform_factor(self) -> None:
        """Test refining grid with uniform factor."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[11, 11])

        fine_grid = grid.refine(2)

        assert fine_grid.num_points == (21, 21)  # Normalized to tuple
        assert fine_grid.bounds == grid.bounds
        assert fine_grid.dimension == grid.dimension

    def test_refine_per_dimension_factors(self) -> None:
        """Test refining grid with per-dimension factors."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[11, 11])

        fine_grid = grid.refine([2, 3])

        assert fine_grid.num_points == (21, 31)  # Normalized to tuple

    def test_refine_preserves_bounds(self) -> None:
        """Test that refinement preserves domain bounds."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 10.0), (0.0, 5.0)], num_points=[11, 11])

        fine_grid = grid.refine(2)

        assert fine_grid.bounds == grid.bounds
        assert fine_grid.coordinates[0][0] == pytest.approx(0.0)
        assert fine_grid.coordinates[0][-1] == pytest.approx(10.0)
        assert fine_grid.coordinates[1][0] == pytest.approx(0.0)
        assert fine_grid.coordinates[1][-1] == pytest.approx(5.0)

    def test_coarsen_uniform_factor(self) -> None:
        """Test coarsening grid with uniform factor."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[21, 21])

        coarse_grid = grid.coarsen(2)

        assert coarse_grid.num_points == (11, 11)  # Normalized to tuple
        assert coarse_grid.bounds == grid.bounds

    def test_coarsen_per_dimension_factors(self) -> None:
        """Test coarsening grid with per-dimension factors."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[21, 31])

        coarse_grid = grid.coarsen([2, 3])

        assert coarse_grid.num_points == (11, 11)  # Normalized to tuple

    def test_refine_coarsen_consistency(self) -> None:
        """Test that refine and coarsen are (approximately) inverse operations."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[11, 11])

        fine_grid = grid.refine(2)
        recovered = fine_grid.coarsen(2)

        assert recovered.num_points == grid.num_points


# ============================================================================
# Test Spacing Queries
# ============================================================================


class TestSpacingQueries:
    """Test grid spacing queries."""

    def test_get_spacing_uniform_1d(self) -> None:
        """Test get_spacing for uniform 1D grid."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 10.0)], num_points=[11])

        spacing = grid.get_spacing(0)

        assert spacing == pytest.approx(1.0)

    def test_get_spacing_uniform_2d(self) -> None:
        """Test get_spacing for uniform 2D grid."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 2.0)], num_points=[11, 21])

        spacing_x = grid.get_spacing(0)
        spacing_y = grid.get_spacing(1)

        assert spacing_x == pytest.approx(0.1)
        assert spacing_y == pytest.approx(0.1)

    def test_get_spacing_custom(self) -> None:
        """Test get_spacing for non-uniform grid."""
        custom_x = np.array([0.0, 0.1, 0.3, 0.6, 1.0])

        grid = TensorProductGrid(
            dimension=1, bounds=[(0.0, 1.0)], num_points=[5], spacing_type="custom", custom_coordinates=[custom_x]
        )

        spacings = grid.get_spacing(0)

        assert isinstance(spacings, np.ndarray)
        assert len(spacings) == 4  # n-1 spacings for n points
        assert spacings[0] == pytest.approx(0.1)
        assert spacings[1] == pytest.approx(0.2)
        assert spacings[2] == pytest.approx(0.3)
        assert spacings[3] == pytest.approx(0.4)

    def test_get_spacing_invalid_dimension_raises(self) -> None:
        """Test that invalid dimension index raises ValueError."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[10, 10])

        with pytest.raises(ValueError, match="dimension_idx 2 >= dimension 2"):
            grid.get_spacing(2)


# ============================================================================
# Test Volume Element Computation
# ============================================================================


class TestVolumeElement:
    """Test volume element computation."""

    def test_volume_element_uniform_1d(self) -> None:
        """Test volume element for uniform 1D grid."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 10.0)], num_points=[11])

        vol = grid.volume_element()

        assert vol == pytest.approx(1.0)

    def test_volume_element_uniform_2d(self) -> None:
        """Test volume element for uniform 2D grid."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 2.0)], num_points=[11, 21])

        vol = grid.volume_element()

        assert vol == pytest.approx(0.1 * 0.1)

    def test_volume_element_uniform_3d(self) -> None:
        """Test volume element for uniform 3D grid."""
        grid = TensorProductGrid(dimension=3, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], num_points=[11, 11, 11])

        vol = grid.volume_element()

        assert vol == pytest.approx(0.1 * 0.1 * 0.1)

    def test_volume_element_custom_requires_index(self) -> None:
        """Test that custom grid requires multi_index for volume element."""
        custom_x = np.array([0.0, 0.1, 0.3, 0.6, 1.0])

        grid = TensorProductGrid(
            dimension=1, bounds=[(0.0, 1.0)], num_points=[5], spacing_type="custom", custom_coordinates=[custom_x]
        )

        with pytest.raises(ValueError, match="multi_index required"):
            grid.volume_element()

    def test_volume_element_custom_with_index(self) -> None:
        """Test volume element for custom grid with multi_index."""
        custom_x = np.array([0.0, 0.1, 0.3, 0.6, 1.0])

        grid = TensorProductGrid(
            dimension=1, bounds=[(0.0, 1.0)], num_points=[5], spacing_type="custom", custom_coordinates=[custom_x]
        )

        # Test at first point
        vol_0 = grid.volume_element(multi_index=(0,))
        assert vol_0 == pytest.approx(0.1)

        # Test at middle point
        vol_2 = grid.volume_element(multi_index=(2,))
        assert vol_2 == pytest.approx(0.25)  # avg of 0.2 and 0.3

        # Test at last point
        vol_4 = grid.volume_element(multi_index=(4,))
        assert vol_4 == pytest.approx(0.4)


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_point_grid(self) -> None:
        """Test grid with single point."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], num_points=[1])

        assert grid.total_points() == 1
        assert grid.spacing[0] == 0.0  # Single point has zero spacing

    def test_two_point_grid(self) -> None:
        """Test grid with two points."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], num_points=[2])

        assert grid.total_points() == 2
        assert grid.spacing[0] == pytest.approx(1.0)

    def test_asymmetric_2d_grid(self) -> None:
        """Test 2D grid with different resolutions in each dimension."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[11, 101])

        assert grid.total_points() == 11 * 101
        assert grid.spacing[0] == pytest.approx(0.1)
        assert grid.spacing[1] == pytest.approx(0.01)

    def test_negative_bounds(self) -> None:
        """Test grid with negative bounds."""
        grid = TensorProductGrid(dimension=2, bounds=[(-1.0, 1.0), (-2.0, 2.0)], num_points=[11, 21])

        assert grid.bounds == ((-1.0, 1.0), (-2.0, 2.0))  # Normalized to tuple
        X, Y = grid.meshgrid()
        assert X[0, 0] == pytest.approx(-1.0)
        assert Y[0, 0] == pytest.approx(-2.0)

    def test_large_grid_total_points(self) -> None:
        """Test total_points calculation for large grid."""
        grid = TensorProductGrid(dimension=3, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], num_points=[100, 100, 100])

        assert grid.total_points() == 1000000
