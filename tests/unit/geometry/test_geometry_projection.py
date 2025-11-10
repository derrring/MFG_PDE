"""
Unit tests for GeometryProjector (Issue #257).

Tests projection operations between different geometry discretizations,
including particle↔grid and grid↔grid projections.
"""

import pytest

import numpy as np

from mfg_pde.geometry import GeometryProjector, SimpleGrid1D, SimpleGrid2D, SimpleGrid3D


class TestGeometryProjectorBasics:
    """Test basic GeometryProjector initialization and detection."""

    def test_same_geometry_no_projection(self):
        """Test that same geometry returns identity projection."""
        grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(10, 10))

        projector = GeometryProjector(hjb_geometry=grid, fp_geometry=grid)

        # Test HJB → FP (should be identity)
        U = np.random.rand(11, 11)
        U_projected = projector.project_hjb_to_fp(U)
        np.testing.assert_array_equal(U, U_projected)

        # Test FP → HJB (should be identity)
        M = np.random.rand(11, 11)
        M_projected = projector.project_fp_to_hjb(M)
        np.testing.assert_array_equal(M, M_projected)

    def test_factory_create(self):
        """Test factory method."""
        grid1 = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(10, 10))
        grid2 = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(20, 20))

        projector = GeometryProjector.create(hjb_geometry=grid1, fp_geometry=grid2)

        assert projector is not None
        assert projector.hjb_geometry is grid1
        assert projector.fp_geometry is grid2


class TestGrid1DProjections:
    """Test projections for 1D grids."""

    def test_grid_to_grid_1d_interpolation(self):
        """Test 1D grid → grid interpolation."""
        from mfg_pde.geometry import BoundaryConditions

        bc = BoundaryConditions(type="periodic")

        # Coarse grid (HJB)
        coarse_grid = SimpleGrid1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)
        coarse_grid.create_grid(num_points=11)  # 11 points

        # Fine grid (FP)
        fine_grid = SimpleGrid1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)
        fine_grid.create_grid(num_points=21)  # 21 points

        projector = GeometryProjector(hjb_geometry=coarse_grid, fp_geometry=fine_grid)

        # Create test function on coarse grid: u(x) = x (linear function for exact interpolation)
        coarse_points = coarse_grid.get_spatial_grid()
        U_coarse = coarse_points

        # Project to fine grid
        U_fine = projector.project_hjb_to_fp(U_coarse)

        # Verify interpolation accuracy
        fine_points = fine_grid.get_spatial_grid()
        U_expected = fine_points

        # Linear interpolation should be exact for linear functions
        np.testing.assert_allclose(U_fine, U_expected, rtol=1e-10, atol=1e-12)

    def test_grid_to_grid_1d_conservation(self):
        """Test that grid→grid projection preserves integral."""
        from mfg_pde.geometry import BoundaryConditions

        bc = BoundaryConditions(type="periodic")

        # Fine grid (FP solver) - 21 points
        fine_grid = SimpleGrid1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)
        fine_grid.create_grid(num_points=21)

        # Coarse grid (HJB solver) - 11 points
        coarse_grid = SimpleGrid1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)
        coarse_grid.create_grid(num_points=11)

        # Setup projector: FP (fine) → HJB (coarse)
        projector = GeometryProjector(hjb_geometry=coarse_grid, fp_geometry=fine_grid)

        # Create density on fine grid (FP)
        points_fine = fine_grid.get_spatial_grid()
        M_fine = np.exp(-((points_fine - 0.5) ** 2) / 0.1)  # Gaussian
        M_fine /= np.trapezoid(M_fine, points_fine)  # Normalize

        # Project from FP (fine) to HJB (coarse)
        M_coarse = projector.project_fp_to_hjb(M_fine)

        # Check integral preservation (approximately)
        points_coarse = coarse_grid.get_spatial_grid()
        integral_fine = np.trapezoid(M_fine, points_fine)
        integral_coarse = np.trapezoid(M_coarse, points_coarse)

        np.testing.assert_allclose(integral_coarse, integral_fine, rtol=0.1)


class TestGrid2DProjections:
    """Test projections for 2D grids."""

    def test_grid_to_grid_2d_shape(self):
        """Test 2D grid → grid projection shape."""
        grid_coarse = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(5, 5))
        grid_fine = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(10, 10))

        projector = GeometryProjector(hjb_geometry=grid_coarse, fp_geometry=grid_fine)

        # Create test field on coarse grid
        U_coarse = np.random.rand(6, 6)  # 5+1 points

        # Project to fine grid
        U_fine = projector.project_hjb_to_fp(U_coarse)

        # Check output shape
        assert U_fine.shape == (11, 11)  # 10+1 points

    def test_grid_to_grid_2d_smooth_function(self):
        """Test 2D grid → grid interpolation accuracy."""
        grid1 = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(10, 10))
        grid2 = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(20, 20))

        projector = GeometryProjector(hjb_geometry=grid1, fp_geometry=grid2)

        # Create smooth test function: u(x,y) = x^2 + y^2
        points1 = grid1.get_spatial_grid()  # (N, 2)
        U_flat = points1[:, 0] ** 2 + points1[:, 1] ** 2
        U1 = U_flat.reshape(11, 11)

        # Project to grid2
        U2 = projector.project_hjb_to_fp(U1)

        # Verify on grid2
        points2 = grid2.get_spatial_grid()
        U_expected_flat = points2[:, 0] ** 2 + points2[:, 1] ** 2
        U_expected = U_expected_flat.reshape(21, 21)

        # Allow interpolation error
        np.testing.assert_allclose(U2, U_expected, rtol=1e-2, atol=1e-2)


class TestGrid3DProjections:
    """Test projections for 3D grids."""

    def test_grid_to_grid_3d_shape(self):
        """Test 3D grid → grid projection shape."""
        grid1 = SimpleGrid3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(3, 3, 3))
        grid2 = SimpleGrid3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(6, 6, 6))

        projector = GeometryProjector(hjb_geometry=grid1, fp_geometry=grid2)

        # Create test field
        U1 = np.random.rand(4, 4, 4)

        # Project
        U2 = projector.project_hjb_to_fp(U1)

        # Check shape
        assert U2.shape == (7, 7, 7)


class TestParticleGridProjections:
    """Test particle ↔ grid projections (simulated with different grids for now)."""

    def test_fine_to_coarse_density_projection(self):
        """Test projecting from fine grid to coarse grid (simulates particle → grid)."""
        # Fine grid simulates particle locations
        fine_grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(50, 50))

        # Coarse grid for HJB
        coarse_grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(10, 10))

        projector = GeometryProjector(hjb_geometry=coarse_grid, fp_geometry=fine_grid)

        # Create density on fine grid (simulates particle density)
        points_fine = fine_grid.get_spatial_grid()
        x, y = points_fine[:, 0], points_fine[:, 1]
        M_fine_flat = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.05)
        M_fine = M_fine_flat.reshape(51, 51)

        # Normalize
        M_fine /= np.sum(M_fine)

        # Project to coarse grid
        M_coarse = projector.project_fp_to_hjb(M_fine)

        # Check shape
        assert M_coarse.shape == (11, 11)

        # Check that density is positive and has reasonable magnitude
        # Note: Simple interpolation doesn't preserve mass - proper restriction operators needed
        assert np.all(M_coarse >= 0.0)  # Non-negative density
        assert np.sum(M_coarse) > 0.0  # Non-zero mass
        assert M_coarse.max() > M_coarse.mean()  # Has a peak

    def test_projection_preserves_peak_location(self):
        """Test that density peak location is preserved during projection."""
        fine_grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(30, 30))
        coarse_grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(10, 10))

        projector = GeometryProjector(hjb_geometry=coarse_grid, fp_geometry=fine_grid)

        # Gaussian centered at (0.7, 0.3)
        center = np.array([0.7, 0.3])
        points_fine = fine_grid.get_spatial_grid()
        dist_sq = np.sum((points_fine - center) ** 2, axis=1)
        M_fine_flat = np.exp(-dist_sq / 0.01)
        M_fine = M_fine_flat.reshape(31, 31)

        # Project to coarse
        M_coarse = projector.project_fp_to_hjb(M_fine)

        # Find peak location in coarse grid
        peak_coarse = np.unravel_index(np.argmax(M_coarse), M_coarse.shape)

        # Get actual coordinates
        points_coarse = coarse_grid.get_spatial_grid().reshape(11, 11, 2)
        coord_coarse = points_coarse[peak_coarse]

        # Peak should be near (0.7, 0.3)
        assert np.linalg.norm(coord_coarse - center) < 0.15  # Within 1.5 grid cells


class TestProjectionMethods:
    """Test different projection methods."""

    def test_auto_detection_grid_to_grid(self):
        """Test auto-detection chooses appropriate method for grid→grid."""
        grid1 = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(5, 5))
        grid2 = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(10, 10))

        projector = GeometryProjector(hjb_geometry=grid1, fp_geometry=grid2, projection_method="auto")

        # Should auto-detect grid→grid methods
        assert projector.hjb_to_fp_method == "grid_interpolation"
        assert projector.fp_to_hjb_method == "grid_restriction"

    def test_manual_method_override(self):
        """Test manual projection method override."""
        grid1 = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(5, 5))
        grid2 = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(10, 10))

        # Force interpolation method
        projector = GeometryProjector(hjb_geometry=grid1, fp_geometry=grid2, projection_method="interpolation")

        assert projector.hjb_to_fp_method == "interpolation"
        assert projector.fp_to_hjb_method == "interpolation"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_values(self):
        """Test projection with zero/empty values."""
        grid1 = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(5, 5))
        grid2 = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(10, 10))

        projector = GeometryProjector(hjb_geometry=grid1, fp_geometry=grid2)

        # Zero field
        U_zero = np.zeros((6, 6))
        U_projected = projector.project_hjb_to_fp(U_zero)

        np.testing.assert_array_equal(U_projected, np.zeros((11, 11)))

    def test_constant_field(self):
        """Test projection of constant field."""
        grid1 = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(5, 5))
        grid2 = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(10, 10))

        projector = GeometryProjector(hjb_geometry=grid1, fp_geometry=grid2)

        # Constant field
        U_const = np.ones((6, 6)) * 5.0
        U_projected = projector.project_hjb_to_fp(U_const)

        # Should remain constant
        np.testing.assert_allclose(U_projected, 5.0 * np.ones((11, 11)), rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
