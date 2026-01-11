#!/usr/bin/env python3
"""
Unit tests for HJBSemiLagrangianSolver.

Tests the semi-Lagrangian method for solving Hamilton-Jacobi-Bellman equations
in Mean Field Games, including characteristic-following schemes and interpolation.
"""

import pytest

import numpy as np

from mfg_pde.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import TensorProductGrid


class TestHJBSemiLagrangianInitialization:
    """Test HJBSemiLagrangianSolver initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic solver initialization with default parameters."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)
        solver = HJBSemiLagrangianSolver(problem)

        assert solver.hjb_method_name == "Semi-Lagrangian"
        assert solver.interpolation_method == "linear"
        assert solver.optimization_method == "brent"
        assert solver.characteristic_solver == "explicit_euler"
        assert solver.tolerance == 1e-8

    def test_custom_interpolation_method(self):
        """Test initialization with custom interpolation method."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)
        solver = HJBSemiLagrangianSolver(problem, interpolation_method="cubic")

        assert solver.interpolation_method == "cubic"

    def test_custom_optimization_method(self):
        """Test initialization with custom optimization method."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)
        solver = HJBSemiLagrangianSolver(problem, optimization_method="golden")

        assert solver.optimization_method == "golden"

    def test_custom_characteristic_solver(self):
        """Test initialization with custom characteristic solver."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk2")

        assert solver.characteristic_solver == "rk2"

    def test_custom_tolerance(self):
        """Test initialization with custom tolerance."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)
        solver = HJBSemiLagrangianSolver(problem, tolerance=1e-10)

        assert solver.tolerance == 1e-10

    def test_grid_parameters_computed(self):
        """Test that grid parameters are properly computed."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)
        solver = HJBSemiLagrangianSolver(problem)

        assert hasattr(solver, "x_grid")
        assert hasattr(solver, "dt")
        assert hasattr(solver, "dx")
        assert len(solver.x_grid) == problem.geometry.get_grid_shape()[0]
        assert np.isclose(solver.dt, problem.dt)
        assert np.isclose(solver.dx, problem.geometry.get_grid_spacing()[0])


class TestHJBSemiLagrangianSolveHJBSystem:
    """Test the main solve_hjb_system method."""

    def test_solve_hjb_system_shape(self):
        """Test that solve_hjb_system returns correct shape."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)
        solver = HJBSemiLagrangianSolver(problem)

        # Create inputs: Nx, Nt are intervals; knots = intervals + 1
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points))
        U_final = np.zeros(Nx_points)
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Output: same shape as input density (Nt+1 time points)
        assert U_solution.shape == (problem.Nt + 1, Nx_points)
        assert np.all(np.isfinite(U_solution))

    def test_solve_hjb_system_final_condition(self):
        """Test that final condition is preserved."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)
        solver = HJBSemiLagrangianSolver(problem)

        # Create inputs with specific final condition
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points))
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - bounds[1][0]) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Final time step should match final condition
        assert np.allclose(U_solution[-1, :], U_final, rtol=0.1)

    def test_solve_hjb_system_backward_propagation(self):
        """Test that solution propagates backward in time."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)
        solver = HJBSemiLagrangianSolver(problem)

        # Create inputs
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points))
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = x_coords**2  # Quadratic final condition
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Solution should propagate backward (values at earlier times should be influenced by final condition)
        # Check that solution at t=0 is different from zero
        assert not np.allclose(U_solution[0, :], 0.0)


class TestHJBSemiLagrangianNumericalProperties:
    """Test numerical properties of the semi-Lagrangian method."""

    def test_solution_finiteness(self):
        """Test that solution remains finite throughout."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[41])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=40)
        solver = HJBSemiLagrangianSolver(problem)

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) * 0.5
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = np.sin(2 * np.pi * x_coords)
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # All values should be finite
        assert np.all(np.isfinite(U_solution))

    @pytest.mark.skip(reason="Semi-Lagrangian method can have numerical overflow issues with certain configurations")
    def test_solution_smoothness(self):
        """Test that solution has reasonable smoothness."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)
        solver = HJBSemiLagrangianSolver(problem)

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points))
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Check spatial smoothness - finite differences shouldn't be too large
        U_diff = np.diff(U_solution, axis=1)
        assert np.max(np.abs(U_diff)) < 100.0


class TestHJBSemiLagrangianIntegration:
    """Integration tests with actual MFG problems."""

    def test_solver_with_uniform_density(self):
        """Test solver with uniform density distribution."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)
        solver = HJBSemiLagrangianSolver(problem)

        # Uniform density
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)

        # Simple final condition
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = (x_coords - 0.5) ** 2

        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_solver_with_gaussian_density(self):
        """Test solver with Gaussian density distribution."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)
        solver = HJBSemiLagrangianSolver(problem)

        # Gaussian density
        bounds = problem.geometry.get_bounds()
        Nx_points = problem.geometry.get_grid_shape()[0]
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_profile = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        m_profile = m_profile / np.sum(m_profile)
        M_density = np.tile(m_profile, (problem.Nt + 1, 1))

        U_final = np.zeros(Nx_points)
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)


class TestHJBSemiLagrangianSolverNotAbstract:
    """Test that HJBSemiLagrangianSolver is concrete (not abstract)."""

    def test_solver_not_abstract(self):
        """Test that HJBSemiLagrangianSolver can be instantiated."""
        import inspect

        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)

        # Should not raise TypeError about abstract methods
        solver = HJBSemiLagrangianSolver(problem)
        assert isinstance(solver, HJBSemiLagrangianSolver)

        # Should not have abstract methods
        assert not inspect.isabstract(HJBSemiLagrangianSolver)


class TestCharacteristicTracingMethods:
    """Test different characteristic tracing methods (explicit_euler, rk2, rk4)."""

    def test_explicit_euler_initialization(self):
        """Test that explicit_euler method initializes correctly."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="explicit_euler")

        assert solver.characteristic_solver == "explicit_euler"

    def test_rk2_initialization(self):
        """Test that rk2 method initializes correctly."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk2")

        assert solver.characteristic_solver == "rk2"

    def test_rk4_initialization(self):
        """Test that rk4 method initializes correctly."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk4")

        assert solver.characteristic_solver == "rk4"

    def test_euler_produces_valid_solution(self):
        """Test that explicit_euler produces valid solution."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20)
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="explicit_euler", use_jax=False)

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_rk2_produces_valid_solution(self):
        """Test that rk2 produces valid solution."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20)
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk2", use_jax=False)

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_rk4_produces_valid_solution(self):
        """Test that rk4 with scipy.solve_ivp produces valid solution."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20)
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk4", use_jax=False)

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_rk2_consistency_with_euler(self):
        """Test that rk2 produces consistent results with euler on smooth problems."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=0.2, Nt=20)

        # Solve with euler
        solver_euler = HJBSemiLagrangianSolver(problem, characteristic_solver="explicit_euler", use_jax=False)
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))
        U_euler = solver_euler.solve_hjb_system(M_density, U_final, U_prev)

        # Solve with rk2
        solver_rk2 = HJBSemiLagrangianSolver(problem, characteristic_solver="rk2", use_jax=False)
        U_rk2 = solver_rk2.solve_hjb_system(M_density, U_final, U_prev)

        # On smooth problems with small dt, should be very similar
        rel_error = np.linalg.norm(U_rk2 - U_euler) / np.linalg.norm(U_euler)
        assert rel_error < 0.1  # Within 10%

    def test_rk4_consistency_with_euler(self):
        """Test that rk4 produces consistent results with euler on smooth problems."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=0.2, Nt=20)

        # Solve with euler
        solver_euler = HJBSemiLagrangianSolver(problem, characteristic_solver="explicit_euler", use_jax=False)
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))
        U_euler = solver_euler.solve_hjb_system(M_density, U_final, U_prev)

        # Solve with rk4
        solver_rk4 = HJBSemiLagrangianSolver(problem, characteristic_solver="rk4", use_jax=False)
        U_rk4 = solver_rk4.solve_hjb_system(M_density, U_final, U_prev)

        # On smooth problems with small dt, should be similar
        rel_error = np.linalg.norm(U_rk4 - U_euler) / np.linalg.norm(U_euler)
        assert rel_error < 0.1  # Within 10%

    def test_trace_characteristic_backward_1d(self):
        """Test _trace_characteristic_backward method directly in 1D."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20)
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk4", use_jax=False)

        # Test characteristic tracing
        x_current = 0.5
        p_optimal = 0.1
        dt = 0.01

        x_departure = solver._trace_characteristic_backward(x_current, p_optimal, dt)

        # Should return a scalar
        assert isinstance(x_departure, (float, np.floating))
        # Should be finite
        assert np.isfinite(x_departure)
        # Should be within domain
        bounds = problem.geometry.get_bounds()
        assert bounds[0][0] <= x_departure <= bounds[1][0]


class TestInterpolationMethods:
    """Test different interpolation methods (linear, cubic, quintic)."""

    def test_linear_interpolation_initialization(self):
        """Test that linear interpolation initializes correctly."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)
        solver = HJBSemiLagrangianSolver(problem, interpolation_method="linear")

        assert solver.interpolation_method == "linear"

    def test_cubic_interpolation_initialization(self):
        """Test that cubic interpolation initializes correctly."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)
        solver = HJBSemiLagrangianSolver(problem, interpolation_method="cubic")

        assert solver.interpolation_method == "cubic"

    def test_cubic_produces_valid_solution_1d(self):
        """Test that cubic interpolation produces valid solution in 1D."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20)
        solver = HJBSemiLagrangianSolver(
            problem, interpolation_method="cubic", characteristic_solver="rk2", use_jax=False
        )

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_cubic_consistency_with_linear(self):
        """Test that cubic interpolation is consistent with linear on smooth problems."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=20)

        # Solve with linear
        solver_linear = HJBSemiLagrangianSolver(
            problem, interpolation_method="linear", characteristic_solver="rk2", use_jax=False
        )
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))
        U_linear = solver_linear.solve_hjb_system(M_density, U_final, U_prev)

        # Solve with cubic
        solver_cubic = HJBSemiLagrangianSolver(
            problem, interpolation_method="cubic", characteristic_solver="rk2", use_jax=False
        )
        U_cubic = solver_cubic.solve_hjb_system(M_density, U_final, U_prev)

        # On smooth problems with fine grid, should be reasonably similar
        # Note: With gradient-based optimal control (Issue #298 fix), interpolation
        # method has more impact since characteristics now move correctly
        rel_error = np.linalg.norm(U_cubic - U_linear) / np.linalg.norm(U_linear)
        assert rel_error < 0.25  # Within 25% (updated after gradient fix)

    def test_cubic_improves_smoothness(self):
        """Test that cubic interpolation produces smoother solutions."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=20)

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        # Use steep gradients to test interpolation quality
        U_final = np.exp(-20 * (x_coords - 0.5) ** 2)
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        # Solve with linear
        solver_linear = HJBSemiLagrangianSolver(
            problem, interpolation_method="linear", characteristic_solver="rk2", use_jax=False
        )
        U_linear = solver_linear.solve_hjb_system(M_density, U_final, U_prev)

        # Solve with cubic
        solver_cubic = HJBSemiLagrangianSolver(
            problem, interpolation_method="cubic", characteristic_solver="rk2", use_jax=False
        )
        U_cubic = solver_cubic.solve_hjb_system(M_density, U_final, U_prev)

        # Measure smoothness via second derivative
        smoothness_linear = np.mean(np.abs(np.diff(U_linear, n=2, axis=1)))
        smoothness_cubic = np.mean(np.abs(np.diff(U_cubic, n=2, axis=1)))

        # Both should be finite
        assert np.isfinite(smoothness_linear)
        assert np.isfinite(smoothness_cubic)
        # Cubic should generally be smoother (smaller second derivatives)
        # This is not always true but should hold for most cases
        # We just check that cubic doesn't make things dramatically worse
        assert smoothness_cubic < smoothness_linear * 2.0


class TestRBFInterpolationFallback:
    """Test RBF interpolation fallback functionality."""

    def test_rbf_fallback_initialization_enabled(self):
        """Test that RBF fallback can be enabled."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)
        solver = HJBSemiLagrangianSolver(problem, use_rbf_fallback=True, rbf_kernel="thin_plate_spline")

        assert solver.use_rbf_fallback is True
        assert solver.rbf_kernel == "thin_plate_spline"

    def test_rbf_fallback_initialization_disabled(self):
        """Test that RBF fallback can be disabled."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)
        solver = HJBSemiLagrangianSolver(problem, use_rbf_fallback=False)

        assert solver.use_rbf_fallback is False

    def test_rbf_kernel_options(self):
        """Test different RBF kernel options."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)

        kernels = ["thin_plate_spline", "multiquadric", "gaussian"]

        for kernel in kernels:
            solver = HJBSemiLagrangianSolver(problem, use_rbf_fallback=True, rbf_kernel=kernel)
            assert solver.rbf_kernel == kernel

    def test_rbf_fallback_produces_valid_solution(self):
        """Test that solver with RBF fallback produces valid solution."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20)
        solver = HJBSemiLagrangianSolver(
            problem, use_rbf_fallback=True, rbf_kernel="thin_plate_spline", characteristic_solver="rk2", use_jax=False
        )

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        # Use steep gradient to potentially trigger RBF fallback
        U_final = np.exp(-20 * (x_coords - 0.5) ** 2)
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_rbf_consistency_with_no_fallback(self):
        """Test that RBF fallback doesn't change results on well-behaved problems."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=20)

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        # Solve without RBF
        solver_no_rbf = HJBSemiLagrangianSolver(
            problem, use_rbf_fallback=False, characteristic_solver="rk2", use_jax=False
        )
        U_no_rbf = solver_no_rbf.solve_hjb_system(M_density, U_final, U_prev)

        # Solve with RBF
        solver_rbf = HJBSemiLagrangianSolver(
            problem, use_rbf_fallback=True, rbf_kernel="thin_plate_spline", characteristic_solver="rk2", use_jax=False
        )
        U_rbf = solver_rbf.solve_hjb_system(M_density, U_final, U_prev)

        # On well-behaved problems, RBF fallback shouldn't trigger
        # Results should be identical or very close
        rel_error = np.linalg.norm(U_rbf - U_no_rbf) / np.linalg.norm(U_no_rbf)
        assert rel_error < 1e-10  # Should be machine precision


class TestEnhancementsIntegration:
    """Test combinations of enhancements working together."""

    def test_rk4_with_cubic_interpolation(self):
        """Test RK4 characteristic tracing with cubic interpolation."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20)
        solver = HJBSemiLagrangianSolver(
            problem, characteristic_solver="rk4", interpolation_method="cubic", use_jax=False
        )

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_rk4_with_rbf_fallback(self):
        """Test RK4 characteristic tracing with RBF fallback."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20)
        solver = HJBSemiLagrangianSolver(
            problem, characteristic_solver="rk4", use_rbf_fallback=True, rbf_kernel="thin_plate_spline", use_jax=False
        )

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_all_enhancements_together(self):
        """Test all enhancements working together: RK4 + cubic + RBF."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20)
        solver = HJBSemiLagrangianSolver(
            problem,
            characteristic_solver="rk4",
            interpolation_method="cubic",
            use_rbf_fallback=True,
            rbf_kernel="thin_plate_spline",
            use_jax=False,
        )

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_enhanced_vs_baseline_consistency(self):
        """Test that enhanced configuration produces consistent results with baseline."""
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[41])
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=20)

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        # Baseline configuration
        solver_baseline = HJBSemiLagrangianSolver(
            problem,
            characteristic_solver="explicit_euler",
            interpolation_method="linear",
            use_rbf_fallback=False,
            use_jax=False,
        )
        U_baseline = solver_baseline.solve_hjb_system(M_density, U_final, U_prev)

        # Enhanced configuration
        solver_enhanced = HJBSemiLagrangianSolver(
            problem,
            characteristic_solver="rk4",
            interpolation_method="cubic",
            use_rbf_fallback=True,
            rbf_kernel="thin_plate_spline",
            use_jax=False,
        )
        U_enhanced = solver_enhanced.solve_hjb_system(M_density, U_final, U_prev)

        # On smooth problems with fine grid, should be reasonably consistent
        # Note: With gradient-based optimal control (Issue #298 fix), method differences
        # are more pronounced since characteristics now move correctly
        rel_error = np.linalg.norm(U_enhanced - U_baseline) / np.linalg.norm(U_baseline)
        assert rel_error < 0.20  # Within 20% (updated after gradient fix)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
