#!/usr/bin/env python3
"""
Test for Bug #15 fix: Handle callable sigma in QP constraints.

Bug #15: TypeError when using QP constraints with callable sigma(x).
Root cause: Code expected numeric sigma but particle methods use callable sigma(x).

This test verifies that HJBGFDMSolver correctly handles:
1. Numeric sigma (constant diffusion)
2. Callable sigma(x) (spatially-varying diffusion)
3. Legacy nu attribute

Without the fix, using qp_optimization_level with callable sigma would raise TypeError.
"""

import pytest

import numpy as np

from mfg_pde.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfg_pde.geometry import TensorProductGrid


class ProblemWithCallableSigma:
    """Test problem with callable sigma(x)."""

    def __init__(self):
        self.d = 1
        self.xmin = 0.0
        self.xmax = 1.0
        self.T = 1.0
        self.Nt = 10
        self.geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[21])

    def sigma(self, x):
        """Spatially-varying diffusion coefficient."""
        # Simple spatially-varying diffusion: sigma(x) = 0.1 + 0.1*x
        if np.ndim(x) == 0:
            return 0.1 + 0.1 * x
        else:
            return 0.1 + 0.1 * x[0]

    def hamiltonian(self, x, m, p, t):
        """Simple quadratic Hamiltonian."""
        return 0.5 * np.sum(p**2) + 0.5 * np.sum(x**2)

    def terminal_cost(self, x):
        """Terminal cost."""
        return 0.5 * np.sum(x**2)

    def running_cost(self, x, m, t):
        """Running cost."""
        return 0.5 * np.sum(x**2)


class ProblemWithNumericSigma:
    """Test problem with numeric sigma."""

    def __init__(self):
        self.d = 1
        self.xmin = 0.0
        self.xmax = 1.0
        self.T = 1.0
        self.Nt = 10
        self.sigma = 0.1  # Numeric constant
        self.geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[21])

    def hamiltonian(self, x, m, p, t):
        """Simple quadratic Hamiltonian."""
        return 0.5 * np.sum(p**2) + 0.5 * np.sum(x**2)

    def terminal_cost(self, x):
        """Terminal cost."""
        return 0.5 * np.sum(x**2)

    def running_cost(self, x, m, t):
        """Running cost."""
        return 0.5 * np.sum(x**2)


class ProblemWithNu:
    """Test problem with legacy nu attribute."""

    def __init__(self):
        self.d = 1
        self.xmin = 0.0
        self.xmax = 1.0
        self.T = 1.0
        self.Nt = 10
        self.nu = 0.1  # Legacy attribute
        self.geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[21])

    def hamiltonian(self, x, m, p, t):
        """Simple quadratic Hamiltonian."""
        return 0.5 * np.sum(p**2) + 0.5 * np.sum(x**2)

    def terminal_cost(self, x):
        """Terminal cost."""
        return 0.5 * np.sum(x**2)

    def running_cost(self, x, m, t):
        """Running cost."""
        return 0.5 * np.sum(x**2)


class TestBug15CallableSigma:
    """Test Bug #15 fix: callable sigma handling in QP constraints."""

    def test_callable_sigma_with_qp_auto(self):
        """Test that callable sigma works with qp_optimization_level='auto'."""
        problem = ProblemWithCallableSigma()

        # Create collocation points
        collocation_points = np.linspace(0.0, 1.0, 20).reshape(-1, 1)

        # Create solver with QP auto level (Bug #15 would fail here)
        solver = HJBGFDMSolver(
            problem,
            collocation_points,
            delta=0.15,
            taylor_order=2,
            qp_optimization_level="auto",
            qp_solver="scipy",  # Use scipy to avoid OSQP dependency
        )

        # Verify solver initialized correctly
        assert solver.qp_optimization_level == "auto"
        assert callable(problem.sigma)

        # Test _get_sigma_value helper with different point indices
        sigma_0 = solver._get_sigma_value(0)  # x=0.0
        sigma_10 = solver._get_sigma_value(10)  # x≈0.5
        sigma_19 = solver._get_sigma_value(19)  # x=1.0

        # Should return different values for spatially-varying diffusion
        assert sigma_0 == pytest.approx(0.1, abs=0.01)
        assert sigma_10 > sigma_0  # Increasing with x
        assert sigma_19 > sigma_10
        assert sigma_19 == pytest.approx(0.2, abs=0.01)

    def test_callable_sigma_with_qp_always(self):
        """Test that callable sigma works with qp_optimization_level='always'."""
        problem = ProblemWithCallableSigma()
        collocation_points = np.linspace(0.0, 1.0, 15).reshape(-1, 1)

        # Create solver with QP always level (Bug #15 would fail here)
        solver = HJBGFDMSolver(
            problem,
            collocation_points,
            delta=0.15,
            taylor_order=2,
            qp_optimization_level="always",
            qp_solver="scipy",
        )

        assert solver.qp_optimization_level == "always"
        assert callable(problem.sigma)

    def test_numeric_sigma_with_qp(self):
        """Test that numeric sigma still works correctly."""
        problem = ProblemWithNumericSigma()
        collocation_points = np.linspace(0.0, 1.0, 15).reshape(-1, 1)

        solver = HJBGFDMSolver(
            problem,
            collocation_points,
            delta=0.15,
            taylor_order=2,
            qp_optimization_level="auto",
            qp_solver="scipy",
        )

        # Numeric sigma should return same value for all points
        sigma_0 = solver._get_sigma_value(0)
        sigma_10 = solver._get_sigma_value(10)

        assert sigma_0 == pytest.approx(0.1)
        assert sigma_10 == pytest.approx(0.1)

    def test_nu_attribute_with_qp(self):
        """Test that legacy nu attribute is handled correctly."""
        problem = ProblemWithNu()
        collocation_points = np.linspace(0.0, 1.0, 15).reshape(-1, 1)

        solver = HJBGFDMSolver(
            problem,
            collocation_points,
            delta=0.15,
            taylor_order=2,
            qp_optimization_level="auto",
            qp_solver="scipy",
        )

        # Should use nu attribute
        sigma_0 = solver._get_sigma_value(0)
        assert sigma_0 == pytest.approx(0.1)

    def test_no_sigma_fallback(self):
        """Test fallback when no sigma/nu attribute exists."""
        # MFGProblem has sigma attribute, so create one without

        class MinimalProblem:
            d = 1
            xmin = 0.0
            xmax = 1.0
            T = 1.0
            Nt = 10
            geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[21])

            def hamiltonian(self, x, m, p, t):
                return 0.5 * np.sum(p**2)

        problem_minimal = MinimalProblem()
        collocation_points = np.linspace(0.0, 1.0, 15).reshape(-1, 1)

        solver = HJBGFDMSolver(
            problem_minimal,
            collocation_points,
            delta=0.15,
            taylor_order=2,
            qp_optimization_level="none",  # Don't need QP for this test
        )

        # Should fall back to 1.0
        sigma_fallback = solver._get_sigma_value(0)
        assert sigma_fallback == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
