"""
Tests for unified Geometry ABC and consolidated geometry base classes.

Part of: Issue #245 Phase 2 - Geometry Architecture Consolidation
Updated: Added AdaptiveGeometry protocol tests (Issue #459)
"""

import pytest

import numpy as np

from mfg_pde.geometry import OneDimensionalAMRGrid, TensorProductGrid
from mfg_pde.geometry.base import CartesianGrid, Geometry
from mfg_pde.geometry.protocol import AdaptiveGeometry, is_adaptive


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


class TestAdaptiveGeometryProtocol:
    """Test AdaptiveGeometry protocol for AMR support (Issue #459)."""

    def test_tensorproductgrid_is_not_adaptive(self):
        """Regular TensorProductGrid does not implement AdaptiveGeometry."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10])

        # Regular grids are not adaptive
        assert not isinstance(grid, AdaptiveGeometry)
        assert not is_adaptive(grid)

    def test_is_adaptive_helper_function(self):
        """Test is_adaptive() helper function."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[11])

        # is_adaptive should return False for non-adaptive geometries
        assert is_adaptive(grid) is False
        assert is_adaptive("not a geometry") is False
        assert is_adaptive(None) is False

    def test_mock_adaptive_geometry_compliance(self):
        """Test that a mock adaptive geometry satisfies the protocol."""

        class MockAdaptiveGrid:
            """Mock adaptive geometry for protocol testing."""

            def refine(self, criteria):
                return 4  # Refined 4 cells

            def coarsen(self, criteria):
                return 2  # Coarsened 2 cells

            def adapt(self, solution_data):
                return {"refined": 4, "coarsened": 2, "total_cells": 102}

            @property
            def max_refinement_level(self):
                return 3

            @property
            def num_leaf_cells(self):
                return 100

        mock = MockAdaptiveGrid()

        # Mock should satisfy the protocol
        assert isinstance(mock, AdaptiveGeometry)
        assert is_adaptive(mock)

        # Test method calls
        assert mock.refine({}) == 4
        assert mock.coarsen({}) == 2
        result = mock.adapt({"u": None})
        assert result["refined"] == 4
        assert mock.max_refinement_level == 3
        assert mock.num_leaf_cells == 100

    def test_incomplete_adaptive_geometry_rejected(self):
        """Test that incomplete implementations are rejected."""

        class IncompleteAdaptive:
            """Missing required methods."""

            def refine(self, criteria):
                return 0

            # Missing: coarsen, adapt, max_refinement_level, num_leaf_cells

        incomplete = IncompleteAdaptive()

        # Should NOT satisfy the protocol
        assert not isinstance(incomplete, AdaptiveGeometry)
        assert not is_adaptive(incomplete)

    def test_adaptive_geometry_protocol_is_orthogonal(self):
        """Test that AdaptiveGeometry is orthogonal to base Geometry."""

        class AdaptiveCartesianMock(CartesianGrid):
            """Mock that implements both CartesianGrid and AdaptiveGeometry."""

            def __init__(self):
                # Minimal CartesianGrid initialization
                self._dimension = 2
                self._bounds = [(0.0, 1.0), (0.0, 1.0)]
                self._Nx_points = [10, 10]

            @property
            def dimension(self):
                return self._dimension

            @property
            def geometry_type(self):
                from mfg_pde.geometry.protocol import GeometryType

                return GeometryType.CARTESIAN_GRID

            @property
            def num_spatial_points(self):
                return 100

            def get_spatial_grid(self):
                return np.zeros((100, 2))

            def get_bounds(self):
                return (np.array([0.0, 0.0]), np.array([1.0, 1.0]))

            def get_problem_config(self):
                return {}

            def is_on_boundary(self, points, tolerance=1e-10):
                return np.zeros(len(points), dtype=bool)

            def get_boundary_normal(self, points):
                return np.zeros_like(points)

            def project_to_boundary(self, points):
                return points

            def project_to_interior(self, points):
                return points

            def get_boundary_regions(self):
                return {"all": {}}

            def get_grid_spacing(self):
                return (0.1, 0.1)

            def get_grid_shape(self):
                return (10, 10)

            def get_laplacian_operator(self):
                return lambda u, idx: 0.0

            def get_gradient_operator(self):
                return lambda u, idx: np.zeros(2)

            def get_interpolator(self):
                return lambda u, point: 0.0

            def get_boundary_handler(self, bc_type="dirichlet"):
                return None

            # AdaptiveGeometry methods
            def refine(self, criteria):
                return 4

            def coarsen(self, criteria):
                return 0

            def adapt(self, solution_data):
                return {"refined": 4, "coarsened": 0, "total_cells": 104}

            @property
            def max_refinement_level(self):
                return 2

            @property
            def num_leaf_cells(self):
                return 100

        mock = AdaptiveCartesianMock()

        # Should satisfy both protocols
        assert isinstance(mock, Geometry)
        assert isinstance(mock, CartesianGrid)
        assert isinstance(mock, AdaptiveGeometry)
        assert is_adaptive(mock)


class TestOneDimensionalAMRGrid:
    """Test OneDimensionalAMRGrid protocol compliance (Issue #460)."""

    def test_amr_1d_is_adaptive(self):
        """OneDimensionalAMRGrid implements AdaptiveGeometry and inherits Geometry.

        Note: AMR classes inherit from Geometry directly (not CartesianGrid)
        because they refine existing partitions with dynamic, non-uniform spacing.
        """
        domain = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[11])
        amr = OneDimensionalAMRGrid(domain, initial_num_intervals=10)

        # AdaptiveGeometry protocol
        assert isinstance(amr, AdaptiveGeometry)
        assert is_adaptive(amr)

        # Geometry inheritance (Issue #468 - changed from CartesianGrid to Geometry)
        assert isinstance(amr, Geometry)
        assert not isinstance(amr, CartesianGrid)  # AMR classes don't inherit CartesianGrid

    def test_amr_1d_geometry_properties(self):
        """OneDimensionalAMRGrid has required geometry properties."""
        domain = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[11])
        amr = OneDimensionalAMRGrid(domain, initial_num_intervals=10)

        # Geometry properties
        assert amr.dimension == 1
        assert amr.num_spatial_points == 10
        assert amr.num_leaf_cells == 10
        assert amr.max_refinement_level == 0

        # Grid-like properties
        spacing = amr.get_grid_spacing()
        assert len(spacing) == 1
        assert np.isclose(spacing[0], 0.1, rtol=0.01)

        shape = amr.get_grid_shape()
        assert shape == (10,)

    def test_amr_1d_refinement(self):
        """OneDimensionalAMRGrid refinement works via AdaptiveGeometry protocol."""
        domain = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[11])
        amr = OneDimensionalAMRGrid(domain, initial_num_intervals=10)

        initial_cells = amr.num_leaf_cells

        # Refine using protocol method
        refined = amr.refine({"interval_ids": [0]})
        assert refined == 1
        assert amr.num_leaf_cells == initial_cells + 1
        assert amr.max_refinement_level == 1

        # Test adapt (no-op without error estimator)
        result = amr.adapt({})
        assert result["total_cells"] == amr.num_leaf_cells

    def test_amr_1d_operators(self):
        """OneDimensionalAMRGrid provides working operators."""
        domain = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[11])
        amr = OneDimensionalAMRGrid(domain, initial_num_intervals=10)

        # Get operators
        laplacian = amr.get_laplacian_operator()
        gradient = amr.get_gradient_operator()
        interpolator = amr.get_interpolator()

        # Test on simple function
        u = np.sin(np.linspace(0, np.pi, amr.num_leaf_cells))

        # Laplacian at interior point
        lap_val = laplacian(u, 5)
        assert isinstance(lap_val, float)

        # Gradient at interior point
        grad_val = gradient(u, 5)
        assert grad_val.shape == (1,)

        # Interpolation
        interp_val = interpolator(u, np.array([0.5]))
        assert isinstance(interp_val, float)
        assert 0 < interp_val <= 1  # Should be somewhere on the sin curve


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
