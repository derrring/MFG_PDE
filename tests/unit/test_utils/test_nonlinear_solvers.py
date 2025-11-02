#!/usr/bin/env python3
"""
Unit tests for nonlinear solvers.

Tests the centralized nonlinear system solvers:
- FixedPointSolver: Damped fixed-point iteration (value iteration)
- NewtonSolver: Newton's method with automatic Jacobian
- PolicyIterationSolver: Howard's algorithm for optimal control
"""

import pytest

import numpy as np

from mfg_pde.utils.numerical import (
    FixedPointSolver,
    NewtonSolver,
    PolicyIterationSolver,
    SolverInfo,
)


class TestSolverInfo:
    """Test SolverInfo container."""

    def test_basic_creation(self):
        """Test basic SolverInfo creation."""
        info = SolverInfo(
            converged=True,
            iterations=10,
            residual=1e-7,
            residual_history=[1e-2, 1e-4, 1e-7],
            solver_time=0.5,
        )

        assert info.converged is True
        assert info.iterations == 10
        assert info.residual == 1e-7
        assert len(info.residual_history) == 3
        assert info.solver_time == 0.5

    def test_non_converged(self):
        """Test SolverInfo for non-converged case."""
        info = SolverInfo(
            converged=False,
            iterations=100,
            residual=1e-3,
            residual_history=[1e-2, 1e-2, 1e-3],
            solver_time=2.0,
        )

        assert info.converged is False
        assert info.residual > 1e-6


class TestFixedPointSolver:
    """Test FixedPointSolver for fixed-point iteration."""

    def test_scalar_fixed_point(self):
        """Test fixed-point on scalar problem: x = cos(x)."""

        def G(x):
            return np.cos(x)

        solver = FixedPointSolver(damping_factor=1.0, max_iterations=100, tolerance=1e-8)
        x, info = solver.solve(G, x0=0.5)

        assert info.converged
        assert np.abs(x - np.cos(x)) < 1e-6
        assert 0.7 < x < 0.74  # Known solution ≈ 0.739085

    def test_vector_fixed_point(self):
        """Test fixed-point on vector problem."""

        def G(x):
            # Fixed point at [1, 1]
            return np.array([0.5 * x[0] + 0.5, 0.5 * x[1] + 0.5])

        solver = FixedPointSolver(damping_factor=1.0, max_iterations=100, tolerance=1e-8)
        x0 = np.array([0.0, 0.0])
        x, info = solver.solve(G, x0)

        assert info.converged
        assert np.allclose(x, [1.0, 1.0], atol=1e-6)

    def test_2d_array_fixed_point(self):
        """Test that 2D arrays are preserved in shape."""

        def G(U):
            # Simple fixed point: U = 0.9*U + 0.1
            return 0.9 * U + 0.1

        solver = FixedPointSolver(damping_factor=1.0, max_iterations=200, tolerance=1e-6)
        U0 = np.zeros((10, 20))
        U, info = solver.solve(G, U0)

        assert U.shape == (10, 20)
        assert info.converged
        assert np.allclose(U, 1.0, atol=1e-5)  # Fixed point at U=1

    def test_damping_factor_stability(self):
        """Test that damping improves convergence for oscillating systems."""

        def G(x):
            # Oscillating map (needs damping)
            return -1.5 * x + 2.0

        # Without damping - may not converge
        solver_no_damp = FixedPointSolver(damping_factor=1.0, max_iterations=50, tolerance=1e-6)
        _, _info_no_damp = solver_no_damp.solve(G, x0=0.0)

        # With damping - should converge
        solver_damped = FixedPointSolver(damping_factor=0.5, max_iterations=50, tolerance=1e-6)
        x, info_damped = solver_damped.solve(G, x0=0.0)

        assert info_damped.converged
        assert np.abs(x - 0.8) < 1e-6  # Fixed point at x=0.8

    def test_max_iterations_limit(self):
        """Test that solver stops at max_iterations."""

        def G(x):
            # Very slow convergence
            return 0.999 * x + 0.001

        solver = FixedPointSolver(damping_factor=1.0, max_iterations=10, tolerance=1e-8)
        _x, info = solver.solve(G, x0=0.0)

        assert not info.converged  # Too few iterations
        assert info.iterations == 10

    def test_residual_history(self):
        """Test that residual history is tracked."""

        def G(x):
            return 0.5 * x + 0.5

        solver = FixedPointSolver(damping_factor=1.0, max_iterations=100, tolerance=1e-8)
        _x, info = solver.solve(G, x0=0.0)

        assert len(info.residual_history) > 0
        # Residual should decrease monotonically
        for i in range(len(info.residual_history) - 1):
            assert info.residual_history[i] >= info.residual_history[i + 1]

    def test_invalid_damping_factor(self):
        """Test that invalid damping factors raise errors."""
        with pytest.raises(ValueError, match="damping_factor must be in"):
            FixedPointSolver(damping_factor=0.0)

        with pytest.raises(ValueError, match="damping_factor must be in"):
            FixedPointSolver(damping_factor=1.5)


class TestNewtonSolver:
    """Test NewtonSolver for Newton's method."""

    def test_scalar_newton(self):
        """Test Newton on scalar problem: x^2 - 4 = 0."""

        def F(x):
            return x**2 - 4.0

        solver = NewtonSolver(max_iterations=30, tolerance=1e-8, sparse=False)
        x, info = solver.solve(F, x0=1.0)

        assert info.converged
        assert np.abs(x - 2.0) < 1e-6 or np.abs(x + 2.0) < 1e-6  # Roots at ±2

    def test_vector_newton(self):
        """Test Newton on vector system: F(x) = x^2 - [1, 4]."""

        def F(x):
            return x**2 - np.array([1.0, 4.0])

        solver = NewtonSolver(max_iterations=30, tolerance=1e-8, sparse=False)
        x0 = np.array([0.5, 1.5])
        x, info = solver.solve(F, x0)

        assert info.converged
        assert np.allclose(x, [1.0, 2.0], atol=1e-6)

    def test_2d_array_newton(self):
        """Test that 2D arrays are preserved in shape."""

        def F(U):
            # Simple nonlinear system: U^2 - 1 = 0
            return U**2 - 1.0

        solver = NewtonSolver(max_iterations=30, tolerance=1e-8, sparse=False)
        U0 = 0.5 * np.ones((10, 20))
        U, info = solver.solve(F, U0)

        assert U.shape == (10, 20)
        assert info.converged
        assert np.allclose(np.abs(U), 1.0, atol=1e-6)

    def test_automatic_jacobian(self):
        """Test automatic Jacobian computation via finite differences."""

        def F(x):
            # Nonlinear system: [x1^2 + x2^2 - 1, x1 - x2]
            return np.array([x[0] ** 2 + x[1] ** 2 - 1.0, x[0] - x[1]])

        solver = NewtonSolver(max_iterations=30, tolerance=1e-8, sparse=False, jacobian=None)
        x0 = np.array([0.5, 0.5])
        x, info = solver.solve(F, x0)

        assert info.converged
        # Should find solution on circle x1^2 + x2^2 = 1 with x1 = x2
        assert np.abs(x[0] - x[1]) < 1e-6
        assert np.abs(x[0] ** 2 + x[1] ** 2 - 1.0) < 1e-6

    def test_user_provided_jacobian(self):
        """Test Newton with user-provided Jacobian."""

        def F(x):
            return x**2 - 4.0

        def J(x):
            return np.array([[2 * x]])

        solver = NewtonSolver(max_iterations=30, tolerance=1e-8, sparse=False, jacobian=J)
        x, info = solver.solve(F, x0=1.0)

        assert info.converged
        assert np.abs(x - 2.0) < 1e-6

    def test_sparse_mode(self):
        """Test Newton with sparse Jacobian."""

        def F(x):
            # Tridiagonal-like system
            n = len(x)
            result = np.zeros(n)
            result[0] = x[0] ** 2 - 1.0
            for i in range(1, n - 1):
                result[i] = x[i] ** 2 + 0.1 * x[i - 1] + 0.1 * x[i + 1] - 1.0
            result[n - 1] = x[n - 1] ** 2 - 1.0
            return result

        solver = NewtonSolver(max_iterations=30, tolerance=1e-8, sparse=True)
        x0 = 0.5 * np.ones(50)
        x, info = solver.solve(F, x0)

        assert info.converged
        assert x.shape == (50,)

    def test_quadratic_convergence(self):
        """Test that Newton exhibits quadratic convergence."""

        def F(x):
            return x**2 - 2.0

        solver = NewtonSolver(max_iterations=30, tolerance=1e-12, sparse=False)
        _x, info = solver.solve(F, x0=1.0)

        # Check for quadratic convergence in residual history
        # After initial iterations, each step should square the error
        if len(info.residual_history) >= 5:
            # Last few iterations should show quadratic convergence
            for i in range(-4, -1):
                ratio = info.residual_history[i] / info.residual_history[i - 1] ** 2
                # Ratio should be roughly constant for quadratic convergence
                assert 0.1 < ratio < 10.0  # Loose check for quadratic behavior

    def test_max_iterations_limit(self):
        """Test that Newton stops at max_iterations."""

        def F(x):
            # Difficult problem
            return np.array([x[0] ** 3 + x[1] ** 3 - 2.0, x[0] ** 2 - x[1]])

        solver = NewtonSolver(max_iterations=3, tolerance=1e-8, sparse=False)
        x0 = np.array([10.0, 10.0])  # Bad initial guess
        _x, info = solver.solve(F, x0)

        assert info.iterations <= 3

    def test_residual_history(self):
        """Test that residual history is tracked."""

        def F(x):
            return x**2 - 4.0

        solver = NewtonSolver(max_iterations=30, tolerance=1e-8, sparse=False)
        _x, info = solver.solve(F, x0=1.0)

        assert len(info.residual_history) > 0
        # Final residual should be in history
        assert info.residual == info.residual_history[-1]


class TestPolicyIterationSolver:
    """Test PolicyIterationSolver for optimal control problems."""

    def test_discrete_control_problem(self):
        """Test policy iteration on simple discrete control problem."""

        # Simple 1D problem: minimize ∫(u^2 + α^2)dt
        # State: u, Control: α ∈ {-1, 0, 1}
        # Dynamics: du/dt = α

        def policy_eval(alpha):
            # For fixed policy α, solve: u = u_next - dt*(u^2 + α^2)
            # This is a fixed-point problem
            dt = 0.1
            u_next = np.zeros_like(alpha)  # Terminal condition
            u_prev = np.zeros_like(alpha)  # Initial guess

            def G(u):
                return u_next - dt * (u**2 + alpha**2)

            fp_solver = FixedPointSolver(damping_factor=0.8, max_iterations=50, tolerance=1e-6)
            u, _ = fp_solver.solve(G, u_prev)
            return u

        def policy_improve(u):
            # For given value u, find best α ∈ {-1, 0, 1}
            # Minimize: α^2 + ∂u/∂x * α
            # Optimal: α* = -0.5 * ∂u/∂x (then discretize)
            gradient = np.gradient(u)
            alpha_continuous = -0.5 * gradient

            # Discretize to {-1, 0, 1}
            alpha = np.zeros_like(alpha_continuous)
            alpha[alpha_continuous < -0.5] = -1.0
            alpha[alpha_continuous > 0.5] = 1.0
            return alpha

        solver = PolicyIterationSolver(max_iterations=20, tolerance=1e-6)

        # Initial policy: zero control
        alpha0 = np.zeros(50)
        value, policy, info = solver.solve(policy_eval, policy_improve, alpha0)

        assert info.converged
        assert value.shape == (50,)
        assert policy.shape == (50,)

    def test_continuous_control_problem(self):
        """Test policy iteration with continuous control."""

        # Simple problem: minimize ∫(x^2 + α^2)dt with dx/dt = α

        def policy_eval(alpha):
            # Solve: u = -dt*(x^2 + α^2) where x is just grid positions
            x = np.linspace(0, 1, 50)
            dt = 0.1
            return -dt * (x**2 + alpha**2)

        def policy_improve(u):
            # Optimal control: α* = -0.5 * ∂u/∂x
            gradient = np.gradient(u)
            return -0.5 * gradient

        solver = PolicyIterationSolver(max_iterations=20, tolerance=1e-6)

        # Initial policy: zero control
        alpha0 = np.zeros(50)
        value, policy, info = solver.solve(policy_eval, policy_improve, alpha0)

        assert info.converged
        assert value.shape == (50,)
        assert policy.shape == (50,)

    def test_policy_convergence(self):
        """Test that policy converges to fixed point."""

        # Simple problem with known optimal policy

        def policy_eval(alpha):
            # Dummy evaluation: u = -alpha^2
            return -(alpha**2)

        def policy_improve(u):
            # Improvement: alpha = sqrt(-u) if u < 0, else 0
            alpha = np.zeros_like(u)
            mask = u < 0
            alpha[mask] = np.sqrt(-u[mask])
            return alpha

        solver = PolicyIterationSolver(max_iterations=20, tolerance=1e-6)

        # Initial policy
        alpha0 = np.ones(50)
        _value, _policy, info = solver.solve(policy_eval, policy_improve, alpha0)

        assert info.converged
        # Policy should converge to fixed point
        # alpha = sqrt(-(-alpha^2)) = alpha (fixed point at any positive alpha)

    def test_max_iterations_limit(self):
        """Test that policy iteration stops at max_iterations."""

        def policy_eval(alpha):
            # Never converges (always changes)
            return alpha + np.random.randn(*alpha.shape) * 0.01

        def policy_improve(u):
            # Random improvement
            return u + np.random.randn(*u.shape) * 0.01

        solver = PolicyIterationSolver(max_iterations=5, tolerance=1e-8)

        alpha0 = np.zeros(50)
        _value, _policy, info = solver.solve(policy_eval, policy_improve, alpha0)

        assert info.iterations <= 5

    def test_2d_policy(self):
        """Test policy iteration with 2D state space."""

        def policy_eval(alpha):
            # Simple 2D evaluation
            return -(alpha**2)

        def policy_improve(u):
            # Simple improvement
            alpha = np.sqrt(np.abs(u))
            return alpha

        solver = PolicyIterationSolver(max_iterations=20, tolerance=1e-6)

        # 2D initial policy
        alpha0 = np.ones((10, 20))
        value, policy, _info = solver.solve(policy_eval, policy_improve, alpha0)

        assert value.shape == (10, 20)
        assert policy.shape == (10, 20)


class TestSolverComparison:
    """Compare different solvers on the same problem."""

    def test_fixed_point_vs_newton_same_solution(self):
        """Test that fixed-point and Newton find the same solution."""

        # Problem: x^2 - 4 = 0, reformulated as fixed-point
        def G(x):
            return np.sqrt(4.0) if x > 0 else x  # Fixed point at x=2

        def F(x):
            return x**2 - 4.0

        fp_solver = FixedPointSolver(damping_factor=0.8, max_iterations=100, tolerance=1e-6)
        newton_solver = NewtonSolver(max_iterations=30, tolerance=1e-6, sparse=False)

        x_fp, info_fp = fp_solver.solve(G, x0=1.0)
        x_newton, info_newton = newton_solver.solve(F, x0=1.0)

        assert info_fp.converged
        assert info_newton.converged
        assert np.abs(x_fp - x_newton) < 1e-5

    def test_newton_faster_than_fixed_point(self):
        """Test that Newton converges faster than fixed-point."""

        # Smooth problem where Newton should excel
        def G(x):
            return np.array([0.5 * x[0] + 0.25, 0.5 * x[1] + 0.25])

        def F(x):
            return x - G(x)

        fp_solver = FixedPointSolver(damping_factor=1.0, max_iterations=100, tolerance=1e-8)
        newton_solver = NewtonSolver(max_iterations=30, tolerance=1e-8, sparse=False)

        x0 = np.array([0.0, 0.0])

        _x_fp, info_fp = fp_solver.solve(G, x0)
        _x_newton, info_newton = newton_solver.solve(F, x0)

        assert info_fp.converged
        assert info_newton.converged
        # Newton should take fewer iterations
        assert info_newton.iterations < info_fp.iterations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
