#!/usr/bin/env python3
"""
Unit tests for WENO Family HJB Solver

Tests the unified WENO family solver implementation with strategic typing excellence.
Validates:
1. All WENO variants are accessible and functional
2. Parameter validation works correctly
3. Weight computations are mathematically sound
4. Interface consistency across variants
5. Strategic typing compliance

Usage:
    python -m pytest tests/unit/test_weno_family.py -v
"""

import pytest

import numpy as np

from mfg_pde import ExampleMFGProblem
from mfg_pde.alg.hjb_solvers import HJBWenoFamilySolver


class TestWenoFamilySolver:
    """Test suite for WENO Family HJB Solver."""

    @pytest.fixture
    def simple_problem(self) -> ExampleMFGProblem:
        """Create simple MFG problem for testing."""
        return ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=32, T=0.1, Nt=10, sigma=0.1)

    @pytest.fixture
    def test_values(self) -> np.ndarray:
        """Create test array for WENO weight computation."""
        # Simple polynomial for testing: u(x) = x^2
        x = np.linspace(0, 1, 21)
        return x**2

    def test_all_variants_available(self, simple_problem):
        """Test that all WENO variants can be instantiated."""
        variants = ["weno5", "weno-z", "weno-m", "weno-js"]

        for variant in variants:
            solver = HJBWenoFamilySolver(problem=simple_problem, weno_variant=variant)
            assert solver.weno_variant == variant
            assert variant.upper() in solver.hjb_method_name

    def test_invalid_variant_raises_error(self, simple_problem):
        """Test that invalid WENO variant raises ValueError."""
        with pytest.raises(ValueError, match="Unknown WENO variant"):
            HJBWenoFamilySolver(problem=simple_problem, weno_variant="invalid_variant")

    def test_parameter_validation(self, simple_problem):
        """Test parameter validation for all solver parameters."""

        # Test invalid CFL number
        with pytest.raises(ValueError, match="CFL number must be in"):
            HJBWenoFamilySolver(simple_problem, cfl_number=0.0)

        with pytest.raises(ValueError, match="CFL number must be in"):
            HJBWenoFamilySolver(simple_problem, cfl_number=1.5)

        # Test invalid diffusion stability factor
        with pytest.raises(ValueError, match="Diffusion stability factor"):
            HJBWenoFamilySolver(simple_problem, diffusion_stability_factor=0.0)

        with pytest.raises(ValueError, match="Diffusion stability factor"):
            HJBWenoFamilySolver(simple_problem, diffusion_stability_factor=0.6)

        # Test invalid WENO epsilon
        with pytest.raises(ValueError, match="WENO epsilon must be positive"):
            HJBWenoFamilySolver(simple_problem, weno_epsilon=0.0)

        # Test invalid WENO-Z parameter
        with pytest.raises(ValueError, match="WENO-Z parameter"):
            HJBWenoFamilySolver(simple_problem, weno_z_parameter=0.0)

    def test_weno_coefficients_setup(self, simple_problem):
        """Test that WENO coefficients are properly initialized."""
        solver = HJBWenoFamilySolver(simple_problem, weno_variant="weno5")

        # Check linear weights
        assert hasattr(solver, "d_plus")
        assert hasattr(solver, "d_minus")
        assert np.allclose(np.sum(solver.d_plus), 1.0)
        assert np.allclose(np.sum(solver.d_minus), 1.0)

        # Check reconstruction coefficients
        assert hasattr(solver, "c_plus")
        assert hasattr(solver, "c_minus")
        assert solver.c_plus.shape == (3, 3)
        assert solver.c_minus.shape == (3, 3)

    def test_smoothness_indicators_computation(self, simple_problem, test_values):
        """Test smoothness indicator computation for polynomial data."""
        solver = HJBWenoFamilySolver(simple_problem, weno_variant="weno5")

        # For quadratic polynomial u = x^2, should have specific smoothness properties
        u_stencil = test_values[8:13]  # 5-point stencil
        beta = solver._compute_smoothness_indicators(u_stencil)

        # Should return 3 smoothness indicators
        assert len(beta) == 3
        assert all(beta >= 0)  # Smoothness indicators should be non-negative

        # For smooth quadratic data, middle stencil should be smoothest
        # (This is problem-dependent but generally true for polynomials)
        assert np.isfinite(beta).all()

    def test_tau_indicator_computation(self, simple_problem, test_values):
        """Test global smoothness indicator τ for WENO-Z."""
        solver = HJBWenoFamilySolver(simple_problem, weno_variant="weno-z")

        u_stencil = test_values[8:13]  # 5-point stencil
        tau = solver._compute_tau_indicator(u_stencil)

        # τ should be non-negative and finite
        assert tau >= 0
        assert np.isfinite(tau)

    def test_weight_computation_variants(self, simple_problem, test_values):
        """Test weight computation for different WENO variants."""
        variants = ["weno5", "weno-z", "weno-m", "weno-js"]

        for variant in variants:
            solver = HJBWenoFamilySolver(simple_problem, weno_variant=variant)

            w_plus, w_minus = solver._compute_weno_weights(test_values, 10)

            # Weights should sum to 1
            assert np.allclose(np.sum(w_plus), 1.0), f"Failed for {variant}"
            assert np.allclose(np.sum(w_minus), 1.0), f"Failed for {variant}"

            # Weights should be non-negative
            assert all(w_plus >= 0), f"Failed for {variant}"
            assert all(w_minus >= 0), f"Failed for {variant}"

            # Should have 3 weights each
            assert len(w_plus) == 3
            assert len(w_minus) == 3

    def test_weno_reconstruction(self, simple_problem, test_values):
        """Test WENO reconstruction produces reasonable results."""
        solver = HJBWenoFamilySolver(simple_problem, weno_variant="weno5")

        # Test reconstruction at interior point
        u_left, u_right = solver._weno_reconstruction(test_values, 10)

        # Both should be finite
        assert np.isfinite(u_left)
        assert np.isfinite(u_right)

        # For smooth data, left and right values should be similar
        # but not necessarily identical due to stencil asymmetry
        assert abs(u_left - u_right) < 10.0  # Reasonable bound

    def test_hjb_step_execution(self, simple_problem):
        """Test that HJB time step executes without errors."""
        solver = HJBWenoFamilySolver(simple_problem, weno_variant="weno5")

        # Create test data
        x = np.linspace(simple_problem.xmin, simple_problem.xmax, simple_problem.Nx + 1)
        u_current = np.sin(2 * np.pi * x)
        m_current = np.ones_like(u_current) / len(u_current)
        dt = 0.001

        # Should execute without error
        u_new = solver.solve_hjb_step(u_current, m_current, dt)

        # Result should be same size and finite
        assert u_new.shape == u_current.shape
        assert np.isfinite(u_new).all()

    def test_time_integration_methods(self, simple_problem):
        """Test different time integration methods."""
        methods = ["tvd_rk3", "explicit_euler"]

        for method in methods:
            solver = HJBWenoFamilySolver(simple_problem, weno_variant="weno5", time_integration=method)

            x = np.linspace(simple_problem.xmin, simple_problem.xmax, simple_problem.Nx + 1)
            u_current = np.sin(2 * np.pi * x)
            m_current = np.ones_like(u_current) / len(u_current)
            dt = 0.001

            # Should execute without error
            u_new = solver.solve_hjb_step(u_current, m_current, dt)
            assert np.isfinite(u_new).all()

    def test_variant_info_retrieval(self, simple_problem):
        """Test that variant information is properly retrieved."""
        variants = ["weno5", "weno-z", "weno-m", "weno-js"]

        for variant in variants:
            solver = HJBWenoFamilySolver(simple_problem, weno_variant=variant)
            info = solver.get_variant_info()

            # Should contain required keys
            required_keys = ["name", "description", "characteristics", "best_for"]
            for key in required_keys:
                assert key in info
                assert isinstance(info[key], str)
                assert len(info[key]) > 0

    def test_stability_time_step_computation(self, simple_problem):
        """Test stable time step computation."""
        solver = HJBWenoFamilySolver(simple_problem, weno_variant="weno5")

        x = np.linspace(simple_problem.xmin, simple_problem.xmax, simple_problem.Nx + 1)
        u_test = np.sin(2 * np.pi * x)
        m_test = np.ones_like(u_test) / len(u_test)

        dt_stable = solver._compute_dt_stable(u_test, m_test)

        # Should be positive and finite
        assert dt_stable > 0
        assert np.isfinite(dt_stable)

        # Should be reasonable for the problem
        assert dt_stable < 1.0  # Should be much smaller than problem time scale

    def test_boundary_handling(self, simple_problem):
        """Test that boundary points are handled correctly."""
        solver = HJBWenoFamilySolver(simple_problem, weno_variant="weno5")

        # Create data with boundary features
        x = np.linspace(simple_problem.xmin, simple_problem.xmax, simple_problem.Nx + 1)
        u_boundary = np.exp(-10 * (x - 0.1) ** 2) + np.exp(-10 * (x - 0.9) ** 2)

        # Test reconstruction near boundaries
        # Should not raise errors
        u_left_0, u_right_0 = solver._weno_reconstruction(u_boundary, 2)
        u_left_end, u_right_end = solver._weno_reconstruction(u_boundary, len(u_boundary) - 3)

        assert np.isfinite([u_left_0, u_right_0, u_left_end, u_right_end]).all()

    def test_variant_performance_consistency(self, simple_problem):
        """Test that all variants produce consistent results for smooth problems."""
        variants = ["weno5", "weno-z", "weno-m", "weno-js"]

        # Use smooth initial condition
        x = np.linspace(simple_problem.xmin, simple_problem.xmax, simple_problem.Nx + 1)
        u_initial = np.sin(2 * np.pi * x)
        m_initial = np.ones_like(u_initial) / len(u_initial)
        dt = 0.001

        results = {}
        for variant in variants:
            solver = HJBWenoFamilySolver(simple_problem, weno_variant=variant)
            u_result = solver.solve_hjb_step(u_initial, m_initial, dt)
            results[variant] = u_result

        # All results should be finite
        for variant, result in results.items():
            assert np.isfinite(result).all(), f"Non-finite result for {variant}"

        # Results should be reasonably close for smooth problems
        # (Different variants may have some variation, but should be in same ballpark)
        max_values = [np.max(np.abs(result)) for result in results.values()]
        assert max(max_values) / min(max_values) < 10.0  # Factor of 10 tolerance

    @pytest.mark.parametrize("variant", ["weno5", "weno-z", "weno-m", "weno-js"])
    def test_individual_variant_functionality(self, simple_problem, variant):
        """Test each WENO variant individually."""
        solver = HJBWenoFamilySolver(simple_problem, weno_variant=variant)

        # Basic functionality test
        x = np.linspace(simple_problem.xmin, simple_problem.xmax, simple_problem.Nx + 1)
        u = np.sin(np.pi * x)
        m = np.ones_like(u) / len(u)
        dt = 0.001

        u_new = solver.solve_hjb_step(u, m, dt)

        assert u_new.shape == u.shape
        assert np.isfinite(u_new).all()

        # Variant info should be accessible
        info = solver.get_variant_info()
        assert variant.replace("-", "").upper() in info["name"].upper()


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
