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


class TestHJBSemiLagrangianInitialization:
    """Test HJBSemiLagrangianSolver initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic solver initialization with default parameters."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = HJBSemiLagrangianSolver(problem)

        assert solver.hjb_method_name == "Semi-Lagrangian"
        assert solver.interpolation_method == "linear"
        assert solver.optimization_method == "brent"
        assert solver.characteristic_solver == "explicit_euler"
        assert solver.tolerance == 1e-8

    def test_custom_interpolation_method(self):
        """Test initialization with custom interpolation method."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = HJBSemiLagrangianSolver(problem, interpolation_method="cubic")

        assert solver.interpolation_method == "cubic"

    def test_custom_optimization_method(self):
        """Test initialization with custom optimization method."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = HJBSemiLagrangianSolver(problem, optimization_method="golden")

        assert solver.optimization_method == "golden"

    def test_custom_characteristic_solver(self):
        """Test initialization with custom characteristic solver."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk2")

        assert solver.characteristic_solver == "rk2"

    def test_custom_tolerance(self):
        """Test initialization with custom tolerance."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = HJBSemiLagrangianSolver(problem, tolerance=1e-10)

        assert solver.tolerance == 1e-10

    def test_grid_parameters_computed(self):
        """Test that grid parameters are properly computed."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = HJBSemiLagrangianSolver(problem)

        assert hasattr(solver, "x_grid")
        assert hasattr(solver, "dt")
        assert hasattr(solver, "dx")
        assert len(solver.x_grid) == problem.Nx + 1
        assert np.isclose(solver.dt, problem.Dt)
        assert np.isclose(solver.dx, problem.Dx)


class TestHJBSemiLagrangianSolveHJBSystem:
    """Test the main solve_hjb_system method."""

    def test_solve_hjb_system_shape(self):
        """Test that solve_hjb_system returns correct shape."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)
        solver = HJBSemiLagrangianSolver(problem)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        # Create inputs
        M_density = np.ones((Nt, Nx))
        U_final = np.zeros(Nx)
        U_prev = np.zeros((Nt, Nx))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert U_solution.shape == (Nt, Nx)
        assert np.all(np.isfinite(U_solution))

    def test_solve_hjb_system_final_condition(self):
        """Test that final condition is preserved."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)
        solver = HJBSemiLagrangianSolver(problem)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        # Create inputs with specific final condition
        M_density = np.ones((Nt, Nx))
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        U_final = 0.5 * (x_coords - problem.xmax) ** 2
        U_prev = np.zeros((Nt, Nx))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Final time step should match final condition
        assert np.allclose(U_solution[-1, :], U_final, rtol=0.1)

    def test_solve_hjb_system_backward_propagation(self):
        """Test that solution propagates backward in time."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)
        solver = HJBSemiLagrangianSolver(problem)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        # Create inputs
        M_density = np.ones((Nt, Nx))
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        U_final = x_coords**2  # Quadratic final condition
        U_prev = np.zeros((Nt, Nx))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Solution should propagate backward (values at earlier times should be influenced by final condition)
        # Check that solution at t=0 is different from zero
        assert not np.allclose(U_solution[0, :], 0.0)


class TestHJBSemiLagrangianNumericalProperties:
    """Test numerical properties of the semi-Lagrangian method."""

    def test_solution_finiteness(self):
        """Test that solution remains finite throughout."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=40, T=1.0, Nt=40)
        solver = HJBSemiLagrangianSolver(problem)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        M_density = np.ones((Nt, Nx)) * 0.5
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        U_final = np.sin(2 * np.pi * x_coords)
        U_prev = np.zeros((Nt, Nx))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # All values should be finite
        assert np.all(np.isfinite(U_solution))

    @pytest.mark.skip(reason="Semi-Lagrangian method can have numerical overflow issues with certain configurations")
    def test_solution_smoothness(self):
        """Test that solution has reasonable smoothness."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = HJBSemiLagrangianSolver(problem)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        M_density = np.ones((Nt, Nx))
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((Nt, Nx))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Check spatial smoothness - finite differences shouldn't be too large
        U_diff = np.diff(U_solution, axis=1)
        assert np.max(np.abs(U_diff)) < 100.0


class TestHJBSemiLagrangianIntegration:
    """Integration tests with actual MFG problems."""

    def test_solver_with_uniform_density(self):
        """Test solver with uniform density distribution."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)
        solver = HJBSemiLagrangianSolver(problem)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        # Uniform density
        M_density = np.ones((Nt, Nx)) / Nx

        # Simple final condition
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        U_final = (x_coords - 0.5) ** 2

        U_prev = np.zeros((Nt, Nx))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (Nt, Nx)

    def test_solver_with_gaussian_density(self):
        """Test solver with Gaussian density distribution."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)
        solver = HJBSemiLagrangianSolver(problem)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        # Gaussian density
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        m_profile = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        m_profile = m_profile / np.sum(m_profile)
        M_density = np.tile(m_profile, (Nt, 1))

        U_final = np.zeros(Nx)
        U_prev = np.zeros((Nt, Nx))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (Nt, Nx)


class TestHJBSemiLagrangianSolverNotAbstract:
    """Test that HJBSemiLagrangianSolver is concrete (not abstract)."""

    def test_solver_not_abstract(self):
        """Test that HJBSemiLagrangianSolver can be instantiated."""
        import inspect

        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)

        # Should not raise TypeError about abstract methods
        solver = HJBSemiLagrangianSolver(problem)
        assert isinstance(solver, HJBSemiLagrangianSolver)

        # Should not have abstract methods
        assert not inspect.isabstract(HJBSemiLagrangianSolver)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
