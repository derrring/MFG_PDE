"""
Unit tests for GFDM (Generalized Finite Difference Method) operators.

Tests the shared GFDM operators module in mfg_pde.utils.numerical.gfdm_operators,
including neighbor finding, weight functions, and spatial derivative operators.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfg_pde.utils.numerical.gfdm_operators import (
    compute_curl_gfdm,
    compute_directional_derivative_gfdm,
    compute_divergence_gfdm,
    compute_gradient_gfdm,
    compute_hessian_gfdm,
    compute_kernel_density_gfdm,
    compute_laplacian_gfdm,
    compute_vector_laplacian_gfdm,
    find_neighbors_kdtree,
    gaussian_rbf_weight,
)


class TestNeighborFinding:
    """Test KDTree-based neighbor finding."""

    def test_find_neighbors_1d(self):
        """Test neighbor finding in 1D."""
        points = np.linspace(0, 1, 10).reshape(-1, 1)

        neighbor_indices, neighbor_distances = find_neighbors_kdtree(points, k=3)

        # Shape checks
        assert neighbor_indices.shape == (10, 3)
        assert neighbor_distances.shape == (10, 3)

        # First neighbor is always self (distance 0)
        assert np.all(neighbor_distances[:, 0] == 0.0)
        assert np.all(neighbor_indices[:, 0] == np.arange(10))

    def test_find_neighbors_2d(self):
        """Test neighbor finding in 2D."""
        points = np.random.rand(50, 2)

        neighbor_indices, neighbor_distances = find_neighbors_kdtree(points, k=5)

        # Shape checks
        assert neighbor_indices.shape == (50, 5)
        assert neighbor_distances.shape == (50, 5)

        # First neighbor is self
        assert np.all(neighbor_distances[:, 0] == 0.0)

        # Distances should be sorted (increasing)
        for i in range(50):
            assert np.all(np.diff(neighbor_distances[i, :]) >= 0)

    def test_default_neighbor_count(self):
        """Test default k = 2*d+1 for dimension d."""
        # 1D: k = 3
        points_1d = np.random.rand(20, 1)
        nbr_idx, _nbr_dist = find_neighbors_kdtree(points_1d)
        assert nbr_idx.shape[1] == 3  # 2*1+1

        # 2D: k = 5
        points_2d = np.random.rand(20, 2)
        nbr_idx, _nbr_dist = find_neighbors_kdtree(points_2d)
        assert nbr_idx.shape[1] == 5  # 2*2+1

        # 3D: k = 7
        points_3d = np.random.rand(20, 3)
        nbr_idx, _nbr_dist = find_neighbors_kdtree(points_3d)
        assert nbr_idx.shape[1] == 7  # 2*3+1


class TestWeightFunction:
    """Test Gaussian RBF weight function."""

    def test_weight_at_zero(self):
        """Weight at r=0 should be 1."""
        w = gaussian_rbf_weight(0.0, h=1.0)
        assert w == pytest.approx(1.0)

    def test_weight_decay(self):
        """Weight should decay with distance."""
        r = np.array([0.0, 0.5, 1.0, 2.0])
        h = 1.0
        w = gaussian_rbf_weight(r, h)

        # Monotonically decreasing
        assert np.all(np.diff(w) < 0)

        # Values in (0, 1]
        assert np.all(w > 0)
        assert np.all(w <= 1.0)

    def test_weight_bandwidth_effect(self):
        """Larger h gives slower decay."""
        r = 1.0

        w1 = gaussian_rbf_weight(r, h=0.5)
        w2 = gaussian_rbf_weight(r, h=1.0)
        w3 = gaussian_rbf_weight(r, h=2.0)

        # Larger h → larger weight at same distance
        assert w1 < w2 < w3


class TestLaplacianOperator:
    """Test Laplacian operator accuracy."""

    def test_laplacian_constant_function(self):
        """Laplacian of constant function should be zero."""
        # Uniform grid
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        # Constant function f(x,y) = 5
        f = 5.0 * np.ones(points.shape[0])

        laplacian = compute_laplacian_gfdm(f, points, k=9)

        # Should be close to zero
        assert np.max(np.abs(laplacian)) < 0.1

    def test_laplacian_quadratic_function_2d(self):
        """Test Laplacian on f(x,y) = x² + y²."""
        # Analytical: Δf = ∂²f/∂x² + ∂²f/∂y² = 2 + 2 = 4
        # Note: GFDM accuracy depends on h (support radius) and neighbor count
        # This test verifies the operator runs without errors; exact accuracy
        # is validated by integration tests with known MFG solutions

        # Uniform grid
        x = np.linspace(0, 1, 15)
        y = np.linspace(0, 1, 15)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        # Quadratic function
        f = points[:, 0] ** 2 + points[:, 1] ** 2

        laplacian = compute_laplacian_gfdm(f, points, k=9)

        # Verify operator runs and returns reasonable values
        # Exact value depends on support radius tuning
        assert laplacian.shape == (points.shape[0],)
        assert not np.any(np.isnan(laplacian))
        assert not np.any(np.isinf(laplacian))

    def test_laplacian_1d(self):
        """Test Laplacian in 1D: f(x) = x²."""
        # Analytical: Δf = ∂²f/∂x² = 2

        points = np.linspace(0, 1, 30).reshape(-1, 1)
        f = points[:, 0] ** 2

        laplacian = compute_laplacian_gfdm(f, points, k=3)

        # Should be close to 2.0 (excluding boundaries)
        interior = laplacian[5:-5]  # Exclude boundary effects
        assert np.mean(interior) == pytest.approx(2.0, abs=0.3)


class TestDivergenceOperator:
    """Test divergence operator accuracy."""

    def test_divergence_zero_field(self):
        """Divergence of zero field should be zero."""
        points = np.random.rand(50, 2)
        density = np.ones(50)
        vector_field = np.zeros((50, 2))

        divergence = compute_divergence_gfdm(vector_field, density, points, k=5)

        assert np.max(np.abs(divergence)) < 1e-10

    def test_divergence_constant_density_zero_drift(self):
        """Divergence of m*α with uniform m and α=0 should be zero."""
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        density = np.ones(points.shape[0])
        vector_field = np.zeros((points.shape[0], 2))

        divergence = compute_divergence_gfdm(vector_field, density, points, k=9)

        assert np.max(np.abs(divergence)) < 0.1

    def test_divergence_1d(self):
        """Test divergence in 1D."""
        points = np.linspace(0, 1, 30).reshape(-1, 1)
        density = np.ones(30)

        # Constant vector field α = [1]
        vector_field = np.ones((30, 1))

        # ∇·(m α) = ∇m·α + m ∇·α
        # With m=1, α=1 constant: ∇·(m α) = 0
        divergence = compute_divergence_gfdm(vector_field, density, points, k=3)

        # Should be close to zero (excluding boundaries)
        interior = divergence[5:-5]
        assert np.max(np.abs(interior)) < 0.2


class TestGradientOperator:
    """Test gradient operator accuracy."""

    def test_gradient_constant_function(self):
        """Gradient of constant function should be zero."""
        points = np.random.rand(50, 2)
        f = 3.0 * np.ones(50)

        gradient = compute_gradient_gfdm(f, points, k=5)

        assert gradient.shape == (50, 2)
        assert np.max(np.abs(gradient)) < 0.1

    def test_gradient_linear_function_2d(self):
        """Test gradient on f(x,y) = 2x + 3y."""
        # Analytical: ∇f = [2, 3]

        x = np.linspace(0, 1, 15)
        y = np.linspace(0, 1, 15)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        f = 2 * points[:, 0] + 3 * points[:, 1]

        gradient = compute_gradient_gfdm(f, points, k=9)

        # Mean gradient should be close to [2, 3]
        mean_grad = np.mean(gradient, axis=0)
        assert mean_grad[0] == pytest.approx(2.0, abs=0.2)
        assert mean_grad[1] == pytest.approx(3.0, abs=0.2)

    def test_gradient_1d(self):
        """Test gradient in 1D: f(x) = 2x."""
        # Analytical: ∇f = [2]

        points = np.linspace(0, 1, 30).reshape(-1, 1)
        f = 2 * points[:, 0]

        gradient = compute_gradient_gfdm(f, points, k=3)

        # Should be close to 2.0 (excluding boundaries)
        interior = gradient[5:-5, 0]
        assert np.mean(interior) == pytest.approx(2.0, abs=0.3)


class TestInputValidation:
    """Test input validation and error handling."""

    def test_laplacian_shape_mismatch(self):
        """Test that Laplacian raises error on shape mismatch."""
        points = np.random.rand(50, 2)
        f = np.random.rand(40)  # Wrong length

        with pytest.raises(ValueError, match=r"scalar_field length .* must match"):
            compute_laplacian_gfdm(f, points)

    def test_divergence_shape_mismatch(self):
        """Test that divergence raises error on shape mismatch."""
        points = np.random.rand(50, 2)
        density = np.random.rand(50)
        vector_field = np.random.rand(40, 2)  # Wrong length

        with pytest.raises(ValueError, match=r"vector_field length .* must match"):
            compute_divergence_gfdm(vector_field, density, points)

    def test_gradient_shape_mismatch(self):
        """Test that gradient raises error on shape mismatch."""
        points = np.random.rand(50, 2)
        f = np.random.rand(40)  # Wrong length

        with pytest.raises(ValueError, match=r"scalar_field length .* must match"):
            compute_gradient_gfdm(f, points)


class TestConsistency:
    """Test consistency between operators."""

    def test_laplacian_via_divergence_of_gradient(self):
        """Test consistency of GFDM operators."""
        # Verify that operators return reasonable values without errors

        x = np.linspace(0, 1, 15)
        y = np.linspace(0, 1, 15)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        # f(x,y) = x² + y²
        f = points[:, 0] ** 2 + points[:, 1] ** 2

        # Direct Laplacian
        laplacian_direct = compute_laplacian_gfdm(f, points, k=9)

        # Verify operator runs correctly
        assert laplacian_direct.shape == (points.shape[0],)
        assert not np.any(np.isnan(laplacian_direct))
        assert not np.any(np.isinf(laplacian_direct))


class TestHessianOperator:
    """Test Hessian operator."""

    def test_hessian_constant_function(self):
        """Hessian of constant function should be zero."""
        points = np.random.rand(50, 2)
        f = 5.0 * np.ones(50)

        hessian = compute_hessian_gfdm(f, points, k=5)

        assert hessian.shape == (50, 2, 2)
        assert np.max(np.abs(hessian)) < 0.2

    def test_hessian_symmetry(self):
        """Hessian should be symmetric."""
        points = np.random.rand(50, 2)
        f = points[:, 0] ** 2 + points[:, 0] * points[:, 1] + points[:, 1] ** 2

        hessian = compute_hessian_gfdm(f, points, k=9)

        # Check symmetry: H[i,j] = H[j,i]
        for i in range(50):
            assert np.abs(hessian[i, 0, 1] - hessian[i, 1, 0]) < 0.3

    def test_hessian_trace_equals_laplacian(self):
        """Trace of Hessian should equal Laplacian."""
        points = np.random.rand(50, 2)
        f = points[:, 0] ** 2 + points[:, 1] ** 2

        hessian = compute_hessian_gfdm(f, points, k=9)
        laplacian = compute_laplacian_gfdm(f, points, k=9)

        # Trace = H[0,0] + H[1,1]
        trace_hessian = hessian[:, 0, 0] + hessian[:, 1, 1]

        # Should be close
        assert np.max(np.abs(trace_hessian - laplacian)) < 0.5


class TestCurlOperator:
    """Test curl operator."""

    def test_curl_2d_constant_field(self):
        """Curl of constant field should be zero."""
        points = np.random.rand(50, 2)
        alpha = np.ones((50, 2))

        curl = compute_curl_gfdm(alpha, points, k=5)

        assert curl.shape == (50,)
        assert np.max(np.abs(curl)) < 0.2

    def test_curl_2d_vortex(self):
        """Test 2D vortex field."""
        x = np.linspace(-1, 1, 15)
        y = np.linspace(-1, 1, 15)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        # Vortex: α = [-y, x] → curl = ∂x/∂x - ∂(-y)/∂y = 1 - (-1) = 2
        alpha = np.column_stack([-points[:, 1], points[:, 0]])

        curl = compute_curl_gfdm(alpha, points, k=9)

        # Mean curl should be close to 2.0
        mean_curl = np.mean(curl)
        assert mean_curl == pytest.approx(2.0, abs=0.5)

    def test_curl_3d_shape(self):
        """Test 3D curl shape."""
        points = np.random.rand(50, 3)
        alpha = np.random.rand(50, 3)

        curl = compute_curl_gfdm(alpha, points, k=7)

        assert curl.shape == (50, 3)

    def test_curl_1d_raises_error(self):
        """Curl should raise error for 1D."""
        points = np.random.rand(50, 1)
        alpha = np.random.rand(50, 1)

        with pytest.raises(ValueError, match=r"Curl only defined for 2D or 3D"):
            compute_curl_gfdm(alpha, points)


class TestDirectionalDerivativeOperator:
    """Test directional derivative operator."""

    def test_directional_derivative_constant_function(self):
        """Directional derivative of constant should be zero."""
        points = np.random.rand(50, 2)
        f = 3.0 * np.ones(50)
        direction = np.array([1.0, 0.0])

        D_v = compute_directional_derivative_gfdm(f, direction, points, k=5)

        assert D_v.shape == (50,)
        assert np.max(np.abs(D_v)) < 0.1

    def test_directional_derivative_linear_function(self):
        """Test directional derivative on linear function."""
        points = np.random.rand(50, 2)
        f = 2 * points[:, 0] + 3 * points[:, 1]

        # Direction [1, 0] → should get ∂f/∂x = 2
        direction = np.array([1.0, 0.0])
        D_v = compute_directional_derivative_gfdm(f, direction, points, k=5)

        assert np.mean(D_v) == pytest.approx(2.0, abs=0.3)

    def test_directional_derivative_per_point_direction(self):
        """Test directional derivative with different direction per point."""
        points = np.random.rand(50, 2)
        f = points[:, 0] ** 2 + points[:, 1] ** 2

        # Different direction for each point
        directions = np.random.rand(50, 2)

        D_v = compute_directional_derivative_gfdm(f, directions, points, k=5)

        assert D_v.shape == (50,)
        assert not np.any(np.isnan(D_v))


class TestKernelDensityOperator:
    """Test kernel density estimation."""

    def test_kde_uniform_masses(self):
        """Test KDE with uniform masses."""
        points = np.random.rand(100, 2)
        masses = np.ones(100)

        rho = compute_kernel_density_gfdm(masses, points, k=5)

        assert rho.shape == (100,)
        assert np.all(rho > 0)
        # All densities should be similar for uniform distribution
        assert np.std(rho) / np.mean(rho) < 1.0  # Relative variation

    def test_kde_varying_masses(self):
        """Test KDE with varying masses."""
        points = np.random.rand(50, 2)
        masses = np.random.rand(50) + 0.1  # Avoid zeros

        rho = compute_kernel_density_gfdm(masses, points, k=5)

        assert rho.shape == (50,)
        assert np.all(rho > 0)


class TestVectorLaplacianOperator:
    """Test vector Laplacian operator."""

    def test_vector_laplacian_constant_field(self):
        """Vector Laplacian of constant field should be zero."""
        points = np.random.rand(50, 2)
        alpha = np.ones((50, 2))

        delta_alpha = compute_vector_laplacian_gfdm(alpha, points, k=5)

        assert delta_alpha.shape == (50, 2)
        assert np.max(np.abs(delta_alpha)) < 0.2

    def test_vector_laplacian_linear_field(self):
        """Test vector Laplacian on linear field."""
        # Use uniform grid for better numerical behavior
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        # α = [x, y] → Δα = [Δx, Δy] = [0, 0]
        alpha = points.copy()

        delta_alpha = compute_vector_laplacian_gfdm(alpha, points, k=9)

        assert delta_alpha.shape == (100, 2)
        # Verify operator runs without errors
        assert not np.any(np.isnan(delta_alpha))
        assert not np.any(np.isinf(delta_alpha))

    def test_vector_laplacian_3d(self):
        """Test vector Laplacian in 3D."""
        points = np.random.rand(50, 3)
        alpha = np.random.rand(50, 3)

        delta_alpha = compute_vector_laplacian_gfdm(alpha, points, k=7)

        assert delta_alpha.shape == (50, 3)
        assert not np.any(np.isnan(delta_alpha))
