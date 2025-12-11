"""
Tests for unified Geometry ABC and consolidated geometry base classes.

Part of: Issue #245 Phase 2 - Geometry Architecture Consolidation
"""

import pytest

import numpy as np

from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.base import CartesianGrid, Geometry


class TestGeometryProtocolCompliance:
    """Test that geometries satisfy Geometry ABC."""

    def test_tensorproductgrid_is_geometry(self):
        """Verify TensorProductGrid satisfies Geometry ABC."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10])

        assert isinstance(grid, Geometry)
        assert isinstance(grid, CartesianGrid)

    def test_tensorproductgrid_has_required_properties(self):
        """Verify TensorProductGrid has all required Geometry properties."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10])

        # Data interface
        assert hasattr(grid, "dimension")
        assert hasattr(grid, "geometry_type")
        assert hasattr(grid, "num_spatial_points")
        assert callable(grid.get_spatial_grid)
        assert callable(grid.get_bounds)
        assert callable(grid.get_problem_config)

        # Solver operations
        assert callable(grid.get_laplacian_operator)
        assert callable(grid.get_gradient_operator)
        assert callable(grid.get_interpolator)
        assert callable(grid.get_boundary_handler)

        # CartesianGrid specific
        assert callable(grid.get_grid_spacing)
        assert callable(grid.get_grid_shape)


class TestSolverOperations:
    """Test solver operation methods."""

    def test_laplacian_operator_exists(self):
        """Test that Laplacian operator is callable."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10])

        laplacian = grid.get_laplacian_operator()
        assert callable(laplacian)

    def test_laplacian_on_quadratic_function_2d(self):
        """
        Test finite difference Laplacian on u(x,y) = x² + y².

        Analytical Laplacian: Δu = 2 + 2 = 4 everywhere
        """
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[21, 21])

        laplacian = grid.get_laplacian_operator()

        # Create test function: u(x,y) = x² + y²
        x = np.linspace(0, 1, 21)
        y = np.linspace(0, 1, 21)
        X, Y = np.meshgrid(x, y, indexing="ij")
        u = X**2 + Y**2

        # Test at several interior points (avoid boundaries)
        test_points = [(5, 5), (10, 10), (15, 15), (8, 12)]
        for idx in test_points:
            lap_value = laplacian(u, idx)
            # FD Laplacian should be close to 4.0
            assert np.isclose(lap_value, 4.0, rtol=0.05), f"Laplacian at {idx}: {lap_value}, expected 4.0"

    def test_laplacian_on_linear_function_1d(self):
        """
        Test Laplacian on u(x) = 2*x.

        Analytical Laplacian: Δu = 0 everywhere
        """
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[21])

        laplacian = grid.get_laplacian_operator()

        # Create test function: u(x) = 2*x
        x = np.linspace(0, 1, 21)
        u = 2 * x

        # Test at interior points
        for idx in [(5,), (10,), (15,)]:
            lap_value = laplacian(u, idx)
            # Laplacian of linear function should be 0
            assert np.isclose(lap_value, 0.0, atol=1e-10), f"Laplacian at {idx}: {lap_value}, expected 0.0"

    def test_gradient_operator_exists(self):
        """Test that gradient operator is callable."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10])

        gradient = grid.get_gradient_operator()
        assert callable(gradient)

    def test_gradient_on_linear_function_2d(self):
        """
        Test finite difference gradient on u(x,y) = 2*x + 3*y.

        Analytical gradient: ∇u = [2, 3]
        """
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[21, 21])

        gradient = grid.get_gradient_operator()

        # Create test function: u(x,y) = 2*x + 3*y
        x = np.linspace(0, 1, 21)
        y = np.linspace(0, 1, 21)
        X, Y = np.meshgrid(x, y, indexing="ij")
        u = 2 * X + 3 * Y

        # Test at several interior points
        test_points = [(5, 5), (10, 10), (15, 15), (8, 12)]
        for idx in test_points:
            grad = gradient(u, idx)
            assert grad.shape == (2,)
            # Gradient should be [2, 3]
            assert np.allclose(grad, [2.0, 3.0], rtol=0.05), f"Gradient at {idx}: {grad}, expected [2, 3]"

    def test_gradient_on_quadratic_function_1d(self):
        """
        Test gradient on u(x) = x².

        Analytical gradient: du/dx = 2*x
        """
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[21])

        gradient = grid.get_gradient_operator()

        # Create test function: u(x) = x²
        x = np.linspace(0, 1, 21)
        u = x**2

        # Test at x=0.5 (index 10)
        grad = gradient(u, (10,))
        assert grad.shape == (1,)
        # At x=0.5, du/dx = 2*0.5 = 1.0
        assert np.isclose(grad[0], 1.0, rtol=0.05), f"Gradient at x=0.5: {grad[0]}, expected 1.0"

    def test_interpolator_exists(self):
        """Test that interpolator is callable."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10])

        interpolate = grid.get_interpolator()
        assert callable(interpolate)

    def test_interpolator_on_linear_function(self):
        """
        Test linear interpolation on u(x,y) = 2*x + 3*y.

        Interpolated values should match analytical values at arbitrary points.
        """
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11])

        interpolate = grid.get_interpolator()

        # Create test function: u(x,y) = 2*x + 3*y
        x = np.linspace(0, 1, 11)
        y = np.linspace(0, 1, 11)
        X, Y = np.meshgrid(x, y, indexing="ij")
        u = 2 * X + 3 * Y

        # Test interpolation at several arbitrary points
        test_points = [
            (0.5, 0.5, 2 * 0.5 + 3 * 0.5),  # (x, y, expected)
            (0.3, 0.7, 2 * 0.3 + 3 * 0.7),
            (0.25, 0.25, 2 * 0.25 + 3 * 0.25),
        ]

        for x_val, y_val, expected in test_points:
            interp_value = interpolate(u, np.array([x_val, y_val]))
            assert np.isclose(interp_value, expected, rtol=1e-10), (
                f"Interpolation at ({x_val}, {y_val}): {interp_value}, expected {expected}"
            )

    def test_boundary_handler_exists(self):
        """Test that boundary handler is returned."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10])

        bc_handler = grid.get_boundary_handler()
        assert bc_handler is not None


class TestCartesianGridUtilities:
    """Test CartesianGrid-specific utility methods."""

    def test_get_grid_spacing(self):
        """Test grid spacing calculation."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 2.0)], Nx_points=[11, 21])

        dx = grid.get_grid_spacing()
        assert len(dx) == 2
        # dx1 = (1-0) / (11-1) = 0.1
        assert np.isclose(dx[0], 0.1, rtol=1e-10)
        # dx2 = (2-0) / (21-1) = 0.1
        assert np.isclose(dx[1], 0.1, rtol=1e-10)

    def test_get_grid_shape(self):
        """Test grid shape retrieval."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 20])

        shape = grid.get_grid_shape()
        assert shape == (10, 20)

    def test_get_bounds(self):
        """Test bounding box retrieval."""
        grid = TensorProductGrid(dimension=3, bounds=[(0.0, 1.0), (-1.0, 1.0), (0.0, 2.0)], Nx_points=[10, 10, 10])

        min_coords, max_coords = grid.get_bounds()
        assert np.allclose(min_coords, [0.0, -1.0, 0.0])
        assert np.allclose(max_coords, [1.0, 1.0, 2.0])


class TestDataInterface:
    """Test data interface methods from Geometry ABC."""

    def test_dimension_property(self):
        """Test dimension property."""
        grid_1d = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[10])
        grid_2d = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10])
        grid_3d = TensorProductGrid(dimension=3, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], Nx_points=[5, 5, 5])

        assert grid_1d.dimension == 1
        assert grid_2d.dimension == 2
        assert grid_3d.dimension == 3

    def test_num_spatial_points(self):
        """Test total spatial points calculation."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 20])

        assert grid.num_spatial_points == 200

    def test_get_spatial_grid(self):
        """Test spatial grid retrieval."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[5, 5])

        points = grid.get_spatial_grid()
        assert points.shape == (25, 2)  # 5*5 points, 2D

    def test_get_problem_config(self):
        """Test problem configuration dictionary."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 2.0)], Nx_points=[10, 20])

        config = grid.get_problem_config()
        assert config["num_spatial_points"] == 200
        assert config["spatial_shape"] == (10, 20)
        assert config["spatial_bounds"] == ((0.0, 1.0), (0.0, 2.0))
        assert config["spatial_discretization"] == (10, 20)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_laplacian_wrong_index_dimension(self):
        """Test that Laplacian raises error for wrong index dimension."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10])
        laplacian = grid.get_laplacian_operator()
        u = np.random.rand(10, 10)

        with pytest.raises(ValueError, match="Index must have length 2"):
            laplacian(u, (5,))  # 1D index for 2D grid

    def test_gradient_wrong_index_dimension(self):
        """Test that gradient raises error for wrong index dimension."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10])
        gradient = grid.get_gradient_operator()
        u = np.random.rand(10, 10)

        with pytest.raises(ValueError, match="Index must have length 2"):
            gradient(u, (5, 5, 5))  # 3D index for 2D grid

    def test_interpolator_wrong_point_dimension(self):
        """Test that interpolator raises error for wrong point dimension."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10])
        interpolate = grid.get_interpolator()
        u = np.random.rand(10, 10)

        with pytest.raises(ValueError, match="Point must have length 2"):
            interpolate(u, np.array([0.5]))  # 1D point for 2D grid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
