#!/usr/bin/env python3
"""
Unit tests for HJBFDMSolver.

Tests the Finite Difference Method (FDM) solver for Hamilton-Jacobi-Bellman equations
using upwind schemes and Newton iteration.
"""

import pytest

import numpy as np

from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde.core.mfg_problem import MFGProblem


class TestHJBFDMSolverInitialization:
    """Test HJBFDMSolver initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic solver initialization with default parameters."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = HJBFDMSolver(problem)

        assert solver.hjb_method_name == "FDM"
        assert solver.max_newton_iterations == 30
        assert solver.newton_tolerance == 1e-6
        assert solver.problem is problem

    def test_custom_newton_parameters(self):
        """Test initialization with custom Newton parameters."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = HJBFDMSolver(
            problem,
            max_newton_iterations=50,
            newton_tolerance=1e-8,
        )

        assert solver.max_newton_iterations == 50
        assert solver.newton_tolerance == 1e-8

    def test_deprecated_parameters_niter(self):
        """Test backward compatibility with deprecated NiterNewton parameter."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)

        with pytest.warns(DeprecationWarning, match="Parameter.*NiterNewton.*deprecated"):
            solver = HJBFDMSolver(problem, NiterNewton=40)

        assert solver.max_newton_iterations == 40

    def test_deprecated_parameters_tolerance(self):
        """Test backward compatibility with deprecated l2errBoundNewton parameter."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)

        with pytest.warns(DeprecationWarning, match="Parameter.*l2errBoundNewton.*deprecated"):
            solver = HJBFDMSolver(problem, l2errBoundNewton=1e-5)

        assert solver.newton_tolerance == 1e-5

    def test_both_deprecated_parameters(self):
        """Test backward compatibility with both deprecated parameters."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)

        with pytest.warns(DeprecationWarning, match="Parameter.*deprecated"):
            solver = HJBFDMSolver(
                problem,
                NiterNewton=25,
                l2errBoundNewton=1e-7,
            )

        assert solver.max_newton_iterations == 25
        assert solver.newton_tolerance == 1e-7

    def test_new_overrides_deprecated(self):
        """Test that new parameters override deprecated ones."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)

        with pytest.warns(DeprecationWarning, match="Parameter.*NiterNewton.*deprecated"):
            solver = HJBFDMSolver(
                problem,
                max_newton_iterations=60,
                NiterNewton=25,  # Should be ignored
            )

        assert solver.max_newton_iterations == 60

    def test_invalid_max_iterations(self):
        """Test that invalid max_newton_iterations raises error."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)

        with pytest.raises(ValueError, match="max_newton_iterations must be >= 1"):
            HJBFDMSolver(problem, max_newton_iterations=0)

        with pytest.raises(ValueError, match="max_newton_iterations must be >= 1"):
            HJBFDMSolver(problem, max_newton_iterations=-5)

    def test_invalid_tolerance(self):
        """Test that invalid newton_tolerance raises error."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)

        with pytest.raises(ValueError, match="newton_tolerance must be > 0"):
            HJBFDMSolver(problem, newton_tolerance=0.0)

        with pytest.raises(ValueError, match="newton_tolerance must be > 0"):
            HJBFDMSolver(problem, newton_tolerance=-1e-6)

    def test_newton_config_storage(self):
        """Test that Newton configuration is properly stored."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = HJBFDMSolver(
            problem,
            max_newton_iterations=35,
            newton_tolerance=1e-7,
        )

        assert hasattr(solver, "_newton_config")
        assert solver._newton_config["max_iterations"] == 35
        assert solver._newton_config["tolerance"] == 1e-7

    def test_backend_initialization(self):
        """Test that backend is properly initialized."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = HJBFDMSolver(problem)

        assert hasattr(solver, "backend")
        assert solver.backend is not None

    def test_custom_backend(self):
        """Test initialization with custom backend."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = HJBFDMSolver(problem, backend="numpy")

        assert solver.backend is not None
        # Backend should be NumPy backend
        assert hasattr(solver.backend, "array")


class TestHJBFDMSolverSolveHJBSystem:
    """Test the main solve_hjb_system method."""

    def test_solve_hjb_system_shape(self):
        """Test that solve_hjb_system returns correct shape."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)
        solver = HJBFDMSolver(problem)

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
        solver = HJBFDMSolver(problem)

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
        solver = HJBFDMSolver(problem)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        # Create inputs
        M_density = np.ones((Nt, Nx))
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        U_final = x_coords**2  # Quadratic final condition
        U_prev = np.zeros((Nt, Nx))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Solution should propagate backward (values at earlier times influenced by final)
        # Check that solution at t=0 is different from zero
        assert not np.allclose(U_solution[0, :], 0.0)

    def test_solve_hjb_system_with_density_variation(self):
        """Test solving with non-uniform density."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)
        solver = HJBFDMSolver(problem)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        # Create Gaussian density
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        m_profile = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        M_density = np.tile(m_profile, (Nt, 1))

        U_final = np.zeros(Nx)
        U_prev = np.zeros((Nt, Nx))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (Nt, Nx)


class TestHJBFDMSolverNumericalProperties:
    """Test numerical properties of the FDM method."""

    def test_solution_finiteness(self):
        """Test that solution remains finite throughout."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=40, T=1.0, Nt=40)
        solver = HJBFDMSolver(problem)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        M_density = np.ones((Nt, Nx)) * 0.5
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        U_final = np.sin(2 * np.pi * x_coords)
        U_prev = np.zeros((Nt, Nx))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # All values should be finite
        assert np.all(np.isfinite(U_solution))

    def test_solution_smoothness(self):
        """Test that solution has reasonable smoothness."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = HJBFDMSolver(problem)

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


class TestHJBFDMSolverParameterSensitivity:
    """Test solver behavior with different parameter configurations."""

    def test_different_newton_iterations(self):
        """Test solver with different Newton iteration counts."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=20, T=0.5, Nt=20)

        for n_iter in [10, 20, 30, 50]:
            solver = HJBFDMSolver(problem, max_newton_iterations=n_iter)

            Nt = problem.Nt + 1
            Nx = problem.Nx + 1

            M_density = np.ones((Nt, Nx))
            U_final = np.zeros(Nx)
            U_prev = np.zeros((Nt, Nx))

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

            assert np.all(np.isfinite(U_solution))

    def test_different_tolerances(self):
        """Test solver with different Newton tolerances."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=20, T=0.5, Nt=20)

        for tol in [1e-4, 1e-6, 1e-8]:
            solver = HJBFDMSolver(problem, newton_tolerance=tol)

            Nt = problem.Nt + 1
            Nx = problem.Nx + 1

            M_density = np.ones((Nt, Nx))
            U_final = np.zeros(Nx)
            U_prev = np.zeros((Nt, Nx))

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

            assert np.all(np.isfinite(U_solution))


class TestHJBFDMSolverIntegration:
    """Integration tests with actual MFG problems."""

    def test_solver_with_uniform_density(self):
        """Test solver with uniform density distribution."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)
        solver = HJBFDMSolver(problem)

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
        solver = HJBFDMSolver(problem)

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


class TestHJBFDMSolverNotAbstract:
    """Test that HJBFDMSolver is concrete (not abstract)."""

    def test_solver_not_abstract(self):
        """Test that HJBFDMSolver can be instantiated."""
        import inspect

        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)

        # Should not raise TypeError about abstract methods
        solver = HJBFDMSolver(problem)
        assert isinstance(solver, HJBFDMSolver)

        # Should not have abstract methods
        assert not inspect.isabstract(HJBFDMSolver)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
