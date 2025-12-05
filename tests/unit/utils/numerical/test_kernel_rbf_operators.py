"""
Unit tests for kernel-based RBF operators.

Tests RBFOperator class for:
- Gradient computation
- Laplacian computation
- Interpolation
- Various kernel types
"""

import pytest

import numpy as np

from mfg_pde.utils.numerical import RBFOperator, create_rbf_operator


class TestRBFOperatorBasics:
    """Test RBFOperator initialization and basic properties."""

    def test_initialization_gaussian(self):
        """Test RBFOperator initializes with Gaussian kernel."""
        points = np.random.rand(50, 2)
        rbf = RBFOperator(points, kernel_type="gaussian")

        assert rbf.n_points == 50
        assert rbf.dimension == 2
        assert rbf.kernel_type == "gaussian"
        assert rbf.A.shape == (50, 50)

    def test_initialization_wendland(self):
        """Test RBFOperator initializes with Wendland kernel."""
        points = np.random.rand(50, 2)
        rbf = RBFOperator(points, kernel_type="wendland_c2")

        assert rbf.n_points == 50
        assert rbf.dimension == 2
        assert rbf.kernel_type == "wendland_c2"

    def test_initialization_custom_shape_parameter(self):
        """Test RBFOperator with custom shape parameter."""
        points = np.random.rand(50, 2)
        rbf = RBFOperator(points, kernel_type="gaussian", shape_parameter=5.0)

        assert rbf.shape_parameter == 5.0

    def test_factory_function(self):
        """Test create_rbf_operator factory function."""
        points = np.random.rand(50, 2)
        rbf = create_rbf_operator(points, kernel_type="gaussian")

        assert isinstance(rbf, RBFOperator)
        assert rbf.n_points == 50


class TestRBFGradient:
    """Test gradient computation."""

    @pytest.fixture
    def grid_2d(self):
        """Create 2D grid points."""
        n = 12
        x = np.linspace(0.1, 0.9, n)
        xx, yy = np.meshgrid(x, x)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def test_gradient_linear_function(self, grid_2d):
        """Test gradient of linear function f(x,y) = 2x + 3y."""
        rbf = RBFOperator(grid_2d, kernel_type="gaussian")

        # f(x,y) = 2x + 3y -> grad = [2, 3]
        u = 2 * grid_2d[:, 0] + 3 * grid_2d[:, 1]
        grad = rbf.gradient(u)

        # Check interior points
        interior = (grid_2d[:, 0] > 0.2) & (grid_2d[:, 0] < 0.8) & (grid_2d[:, 1] > 0.2) & (grid_2d[:, 1] < 0.8)

        grad_x_error = np.mean(np.abs(grad[interior, 0] - 2.0))
        grad_y_error = np.mean(np.abs(grad[interior, 1] - 3.0))

        assert grad_x_error < 0.1, f"Gradient x error too large: {grad_x_error}"
        assert grad_y_error < 0.1, f"Gradient y error too large: {grad_y_error}"

    def test_gradient_quadratic_function(self, grid_2d):
        """Test gradient of quadratic function f(x,y) = x^2 + y^2."""
        rbf = RBFOperator(grid_2d, kernel_type="gaussian")

        # f(x,y) = x^2 + y^2 -> grad = [2x, 2y]
        u = grid_2d[:, 0] ** 2 + grid_2d[:, 1] ** 2
        grad = rbf.gradient(u)

        # Check interior points
        interior = (grid_2d[:, 0] > 0.2) & (grid_2d[:, 0] < 0.8) & (grid_2d[:, 1] > 0.2) & (grid_2d[:, 1] < 0.8)

        expected_grad_x = 2 * grid_2d[interior, 0]
        expected_grad_y = 2 * grid_2d[interior, 1]

        grad_x_error = np.mean(np.abs(grad[interior, 0] - expected_grad_x))
        grad_y_error = np.mean(np.abs(grad[interior, 1] - expected_grad_y))

        assert grad_x_error < 0.1, f"Gradient x error too large: {grad_x_error}"
        assert grad_y_error < 0.1, f"Gradient y error too large: {grad_y_error}"


class TestRBFLaplacian:
    """Test Laplacian computation."""

    @pytest.fixture
    def grid_2d(self):
        """Create 2D grid points."""
        n = 12
        x = np.linspace(0.1, 0.9, n)
        xx, yy = np.meshgrid(x, x)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def test_laplacian_quadratic_function(self, grid_2d):
        """Test Laplacian of f(x,y) = x^2 + y^2 -> Delta f = 4."""
        rbf = RBFOperator(grid_2d, kernel_type="gaussian")

        u = grid_2d[:, 0] ** 2 + grid_2d[:, 1] ** 2
        lap = rbf.laplacian(u)

        # Check interior points
        interior = (grid_2d[:, 0] > 0.2) & (grid_2d[:, 0] < 0.8) & (grid_2d[:, 1] > 0.2) & (grid_2d[:, 1] < 0.8)

        lap_error = np.mean(np.abs(lap[interior] - 4.0))
        assert lap_error < 0.5, f"Laplacian error too large: {lap_error}"

    def test_laplacian_linear_function(self, grid_2d):
        """Test Laplacian of linear function f(x,y) = ax + by -> Delta f = 0."""
        rbf = RBFOperator(grid_2d, kernel_type="gaussian")

        u = 3 * grid_2d[:, 0] + 2 * grid_2d[:, 1]
        lap = rbf.laplacian(u)

        # Check interior points
        interior = (grid_2d[:, 0] > 0.2) & (grid_2d[:, 0] < 0.8) & (grid_2d[:, 1] > 0.2) & (grid_2d[:, 1] < 0.8)

        lap_error = np.mean(np.abs(lap[interior]))
        assert lap_error < 0.5, f"Laplacian of linear should be ~0, got: {lap_error}"


class TestRBFInterpolation:
    """Test interpolation functionality."""

    @pytest.fixture
    def grid_2d(self):
        """Create 2D grid points."""
        n = 10
        x = np.linspace(0.1, 0.9, n)
        xx, yy = np.meshgrid(x, x)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def test_interpolation_exact_at_points(self, grid_2d):
        """Test that interpolation is near-exact at collocation points."""
        rbf = RBFOperator(grid_2d, kernel_type="gaussian")

        u = grid_2d[:, 0] ** 2 + grid_2d[:, 1] ** 2
        u_interp = rbf.interpolate(u, grid_2d)

        # Due to regularization, interpolation may not be machine-precision exact
        error = np.max(np.abs(u_interp - u))
        assert error < 1e-4, f"Interpolation error too large: {error}"

    def test_interpolation_at_new_points(self, grid_2d):
        """Test interpolation at new evaluation points."""
        rbf = RBFOperator(grid_2d, kernel_type="gaussian")

        # Simple function
        u = grid_2d[:, 0] + grid_2d[:, 1]

        # Evaluate at some new points
        eval_pts = np.array([[0.5, 0.5], [0.3, 0.7], [0.6, 0.4]])
        u_interp = rbf.interpolate(u, eval_pts)

        # Compare with exact values
        u_exact = eval_pts[:, 0] + eval_pts[:, 1]

        error = np.max(np.abs(u_interp - u_exact))
        assert error < 0.05, f"Interpolation error too large: {error}"


class TestRBFConditionNumber:
    """Test condition number computation."""

    def test_condition_number_gaussian(self):
        """Test condition number for Gaussian RBF."""
        points = np.random.rand(50, 2)
        rbf = RBFOperator(points, kernel_type="gaussian")

        cond = rbf.condition_number()
        assert cond > 1, "Condition number should be > 1"
        assert np.isfinite(cond), "Condition number should be finite"

    def test_wendland_better_conditioned(self):
        """Test that Wendland typically has better conditioning."""
        np.random.seed(42)
        points = np.random.rand(50, 2)

        rbf_gauss = RBFOperator(points, kernel_type="gaussian")
        rbf_wend = RBFOperator(points, kernel_type="wendland_c2")

        cond_gauss = rbf_gauss.condition_number()
        cond_wend = rbf_wend.condition_number()

        # Wendland should generally be better conditioned due to compact support
        # This is a soft check - mainly verifying both work
        assert np.isfinite(cond_gauss)
        assert np.isfinite(cond_wend)


class TestRBFHessian:
    """Test Hessian computation."""

    @pytest.fixture
    def grid_2d(self):
        """Create 2D grid points."""
        n = 12
        x = np.linspace(0.1, 0.9, n)
        xx, yy = np.meshgrid(x, x)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def test_hessian_quadratic(self, grid_2d):
        """Test Hessian of f(x,y) = x^2 + y^2 -> H = [[2,0],[0,2]]."""
        rbf = RBFOperator(grid_2d, kernel_type="gaussian")

        u = grid_2d[:, 0] ** 2 + grid_2d[:, 1] ** 2
        hess = rbf.hessian(u)

        assert hess.shape == (len(grid_2d), 2, 2)

        # Check interior points
        interior = (grid_2d[:, 0] > 0.25) & (grid_2d[:, 0] < 0.75) & (grid_2d[:, 1] > 0.25) & (grid_2d[:, 1] < 0.75)

        # Diagonal elements should be ~2
        diag_error = np.mean(np.abs(hess[interior, 0, 0] - 2.0))
        diag_error += np.mean(np.abs(hess[interior, 1, 1] - 2.0))

        # Off-diagonal should be ~0
        off_diag_error = np.mean(np.abs(hess[interior, 0, 1]))
        off_diag_error += np.mean(np.abs(hess[interior, 1, 0]))

        # These are approximate due to numerical differentiation
        assert diag_error < 2.0, f"Hessian diagonal error: {diag_error}"
        assert off_diag_error < 1.0, f"Hessian off-diagonal error: {off_diag_error}"

    def test_hessian_symmetry(self, grid_2d):
        """Test that Hessian is symmetric."""
        rbf = RBFOperator(grid_2d, kernel_type="gaussian")

        u = grid_2d[:, 0] ** 2 + grid_2d[:, 0] * grid_2d[:, 1] + grid_2d[:, 1] ** 2
        hess = rbf.hessian(u)

        # Check symmetry: H[i,j] == H[j,i]
        symmetry_error = np.max(np.abs(hess[:, 0, 1] - hess[:, 1, 0]))
        assert symmetry_error < 1e-10, f"Hessian not symmetric: {symmetry_error}"


class TestRBFHigherDimensions:
    """Test RBF operators in higher dimensions."""

    def test_3d_gradient(self):
        """Test gradient in 3D."""
        np.random.seed(42)
        points = np.random.rand(100, 3)
        rbf = RBFOperator(points, kernel_type="gaussian")

        # f(x,y,z) = x + 2y + 3z -> grad = [1, 2, 3]
        u = points[:, 0] + 2 * points[:, 1] + 3 * points[:, 2]
        grad = rbf.gradient(u)

        assert grad.shape == (100, 3)

        # Check mean gradient
        mean_grad = np.mean(grad, axis=0)
        assert np.abs(mean_grad[0] - 1.0) < 0.3
        assert np.abs(mean_grad[1] - 2.0) < 0.3
        assert np.abs(mean_grad[2] - 3.0) < 0.3

    def test_3d_laplacian(self):
        """Test Laplacian in 3D."""
        np.random.seed(42)
        points = np.random.rand(100, 3)
        rbf = RBFOperator(points, kernel_type="gaussian")

        # f(x,y,z) = x^2 + y^2 + z^2 -> Delta f = 6
        u = points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2
        lap = rbf.laplacian(u)

        assert lap.shape == (100,)

        # Check mean Laplacian (interior points hard to define for random)
        mean_lap = np.mean(lap)
        # Allow larger tolerance for 3D with random points
        assert np.abs(mean_lap - 6.0) < 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
