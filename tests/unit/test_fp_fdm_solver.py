#!/usr/bin/env python3
"""
Unit tests for FPFDMSolver - comprehensive coverage.

Tests the Finite Difference Method (FDM) solver for Fokker-Planck equations
with different boundary conditions (periodic, Dirichlet, no-flux).
"""

import pytest

import numpy as np

from mfg_pde import ExampleMFGProblem
from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver
from mfg_pde.geometry import BoundaryConditions


class TestFPFDMSolverInitialization:
    """Test FPFDMSolver initialization and setup."""

    def test_basic_initialization(self):
        """Test basic solver initialization with default boundary conditions."""
        problem = ExampleMFGProblem()
        solver = FPFDMSolver(problem)

        assert solver.fp_method_name == "FDM"
        assert solver.boundary_conditions.type == "no_flux"
        assert solver.problem is problem

    def test_initialization_with_periodic_bc(self):
        """Test initialization with periodic boundary conditions."""
        problem = ExampleMFGProblem()
        bc = BoundaryConditions(type="periodic")
        solver = FPFDMSolver(problem, boundary_conditions=bc)

        assert solver.fp_method_name == "FDM"
        assert solver.boundary_conditions.type == "periodic"

    def test_initialization_with_dirichlet_bc(self):
        """Test initialization with Dirichlet boundary conditions."""
        problem = ExampleMFGProblem()
        bc = BoundaryConditions(type="dirichlet", left_value=0.0, right_value=0.0)
        solver = FPFDMSolver(problem, boundary_conditions=bc)

        assert solver.fp_method_name == "FDM"
        assert solver.boundary_conditions.type == "dirichlet"
        assert solver.boundary_conditions.left_value == 0.0
        assert solver.boundary_conditions.right_value == 0.0

    def test_initialization_with_no_flux_bc(self):
        """Test initialization with no-flux boundary conditions."""
        problem = ExampleMFGProblem()
        bc = BoundaryConditions(type="no_flux")
        solver = FPFDMSolver(problem, boundary_conditions=bc)

        assert solver.fp_method_name == "FDM"
        assert solver.boundary_conditions.type == "no_flux"


class TestFPFDMSolverBasicSolution:
    """Test basic solution functionality."""

    def test_solve_fp_system_shape(self):
        """Test that solve_fp_system returns correct shape."""
        problem = ExampleMFGProblem()
        solver = FPFDMSolver(problem)

        Nx = problem.Nx + 1
        Nt = problem.Nt + 1

        # Create simple inputs
        m_initial = np.ones(Nx) / Nx  # Normalized density
        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        assert m_result.shape == (Nt, Nx)

    def test_solve_fp_system_initial_condition_preserved(self):
        """Test that initial condition is preserved at t=0."""
        problem = ExampleMFGProblem()
        solver = FPFDMSolver(problem)

        Nx = problem.Nx + 1
        Nt = problem.Nt + 1

        # Create Gaussian initial condition
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        m_initial = np.exp(-((x_coords - 0.0) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial)  # Normalize

        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # Initial condition should be preserved (approximately, after non-negativity enforcement)
        assert np.allclose(m_result[0, :], m_initial, rtol=0.1)

    def test_solve_fp_system_zero_timesteps(self):
        """Test behavior with zero time steps (Nt=0)."""
        problem = ExampleMFGProblem()
        problem.Nt = -1  # Results in Nt+1 = 0
        solver = FPFDMSolver(problem)

        Nx = problem.Nx + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((0, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        assert m_result.shape == (0, Nx)

    def test_solve_fp_system_one_timestep(self):
        """Test behavior with single time step (Nt=1)."""
        problem = ExampleMFGProblem()
        problem.Nt = 0  # Results in Nt+1 = 1
        solver = FPFDMSolver(problem)

        Nx = problem.Nx + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((1, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        assert m_result.shape == (1, Nx)
        # Should return initial condition (possibly with non-negativity enforcement)
        assert np.allclose(m_result[0, :], m_initial, rtol=0.1)


class TestFPFDMSolverBoundaryConditions:
    """Test different boundary condition types."""

    def test_periodic_boundary_conditions(self):
        """Test periodic boundary conditions."""
        problem = ExampleMFGProblem()
        bc = BoundaryConditions(type="periodic")
        solver = FPFDMSolver(problem, boundary_conditions=bc)

        Nx = problem.Nx + 1
        Nt = problem.Nt + 1

        # Initial condition with support near boundaries
        m_initial = np.zeros(Nx)
        m_initial[0] = 0.5
        m_initial[-1] = 0.5

        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # With periodic BC, mass should wrap around
        assert m_result.shape == (Nt, Nx)
        # Mass should be preserved
        assert np.all(m_result >= -1e-10)  # Non-negative (with small tolerance)

    def test_dirichlet_boundary_conditions(self):
        """Test Dirichlet boundary conditions."""
        problem = ExampleMFGProblem()
        bc = BoundaryConditions(type="dirichlet", left_value=0.1, right_value=0.2)
        solver = FPFDMSolver(problem, boundary_conditions=bc)

        Nx = problem.Nx + 1
        Nt = problem.Nt + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # Boundary values should be enforced at all time steps
        for t in range(Nt):
            assert np.isclose(m_result[t, 0], 0.1, atol=1e-10)
            assert np.isclose(m_result[t, -1], 0.2, atol=1e-10)

    def test_no_flux_boundary_conditions(self):
        """Test no-flux boundary conditions."""
        problem = ExampleMFGProblem()
        bc = BoundaryConditions(type="no_flux")
        solver = FPFDMSolver(problem, boundary_conditions=bc)

        Nx = problem.Nx + 1
        Nt = problem.Nt + 1

        # Gaussian initial condition
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        m_initial = np.exp(-((x_coords - 0.0) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial)

        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # With no-flux BC, total mass should be approximately conserved
        initial_mass = np.sum(m_initial)
        for t in range(Nt):
            final_mass = np.sum(m_result[t, :])
            # Allow some numerical error
            assert np.isclose(final_mass, initial_mass, rtol=0.1)


class TestFPFDMSolverNonNegativity:
    """Test non-negativity enforcement."""

    def test_non_negativity_enforcement(self):
        """Test that solution remains non-negative."""
        problem = ExampleMFGProblem()
        solver = FPFDMSolver(problem)

        Nx = problem.Nx + 1
        Nt = problem.Nt + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # All values should be non-negative (with small tolerance for numerical errors)
        assert np.all(m_result >= -1e-10)

    def test_initial_condition_non_negativity(self):
        """Test that negative values in initial condition are set to zero."""
        problem = ExampleMFGProblem()
        solver = FPFDMSolver(problem)

        Nx = problem.Nx + 1
        Nt = problem.Nt + 1

        # Initial condition with some negative values
        m_initial = np.random.randn(Nx)
        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # Initial condition should have negative values removed
        assert np.all(m_result[0, :] >= 0)


class TestFPFDMSolverWithDrift:
    """Test solver behavior with non-zero drift (from HJB solution)."""

    def test_solve_with_linear_drift(self):
        """Test solution with linear value function (constant drift)."""
        problem = ExampleMFGProblem()
        solver = FPFDMSolver(problem)

        Nx = problem.Nx + 1
        Nt = problem.Nt + 1

        # Gaussian initial condition at center
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        m_initial = np.exp(-((x_coords - 0.0) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial)

        # Linear value function: U(t,x) = x (constant drift)
        U_solution = np.tile(x_coords, (Nt, 1))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # Solution should evolve (not remain constant)
        assert not np.allclose(m_result[-1, :], m_result[0, :])
        # Should remain non-negative
        assert np.all(m_result >= -1e-10)

    def test_solve_with_quadratic_value_function(self):
        """Test solution with quadratic value function."""
        problem = ExampleMFGProblem()
        solver = FPFDMSolver(problem)

        Nx = problem.Nx + 1
        Nt = problem.Nt + 1

        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        m_initial = np.ones(Nx) / Nx

        # Quadratic value function: U(t,x) = x^2 (linear drift)
        U_solution = np.tile(x_coords**2, (Nt, 1))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # Solution should evolve
        assert not np.allclose(m_result[-1, :], m_result[0, :])
        # Should remain non-negative
        assert np.all(m_result >= -1e-10)


class TestFPFDMSolverEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_diffusion_timestep(self):
        """Test behavior when Dt is extremely small."""
        problem = ExampleMFGProblem()
        problem.Dt = 1e-20  # Very small timestep
        solver = FPFDMSolver(problem)

        Nx = problem.Nx + 1
        Nt = problem.Nt + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # With very small Dt, solution should remain close to initial condition
        assert np.allclose(m_result[1, :], m_result[0, :], rtol=0.1)

    def test_zero_spatial_step(self):
        """Test behavior when Dx is extremely small (but Nx > 1)."""
        problem = ExampleMFGProblem()
        problem.Dx = 1e-20  # Very small spatial step
        solver = FPFDMSolver(problem)

        Nx = problem.Nx + 1
        Nt = problem.Nt + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # With very small Dx, solution should remain close to initial condition
        assert np.allclose(m_result[1, :], m_result[0, :], rtol=0.1)

    def test_single_spatial_point(self):
        """Test behavior with single spatial point (Nx=1)."""
        problem = ExampleMFGProblem()
        problem.Nx = 0  # Results in Nx+1 = 1
        solver = FPFDMSolver(problem)

        Nx = problem.Nx + 1
        Nt = problem.Nt + 1

        m_initial = np.array([1.0])
        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # With single point, solution evolves but should remain positive
        assert m_result.shape == (Nt, 1)
        assert m_result[0, 0] == 1.0  # Initial condition preserved
        assert np.all(m_result >= 0)  # Non-negative throughout


class TestFPFDMSolverMassConservation:
    """Test mass conservation properties."""

    def test_mass_conservation_no_flux(self):
        """Test that mass is conserved with no-flux boundary conditions."""
        problem = ExampleMFGProblem()
        bc = BoundaryConditions(type="no_flux")
        solver = FPFDMSolver(problem, boundary_conditions=bc)

        Nx = problem.Nx + 1
        Nt = problem.Nt + 1

        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        m_initial = np.exp(-((x_coords - 0.0) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial)

        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # Check mass conservation at each time step
        initial_mass = np.sum(m_initial)
        for t in range(Nt):
            current_mass = np.sum(m_result[t, :])
            assert np.isclose(current_mass, initial_mass, rtol=0.1)

    def test_mass_evolution_periodic(self):
        """Test mass evolution with periodic boundary conditions."""
        problem = ExampleMFGProblem()
        bc = BoundaryConditions(type="periodic")
        solver = FPFDMSolver(problem, boundary_conditions=bc)

        Nx = problem.Nx + 1
        Nt = problem.Nt + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # With periodic BC and zero drift, mass should be conserved
        initial_mass = np.sum(m_initial)
        final_mass = np.sum(m_result[-1, :])
        assert np.isclose(final_mass, initial_mass, rtol=0.1)


class TestFPFDMSolverIntegration:
    """Integration tests with actual MFG problems."""

    def test_solver_with_example_problem(self):
        """Test solver works with ExampleMFGProblem."""
        problem = ExampleMFGProblem()
        solver = FPFDMSolver(problem)

        assert solver is not None
        assert hasattr(solver, "solve_fp_system")
        assert callable(solver.solve_fp_system)

    def test_solver_not_abstract(self):
        """Test that FPFDMSolver can be instantiated (is concrete)."""
        import inspect

        problem = ExampleMFGProblem()
        solver = FPFDMSolver(problem)

        assert isinstance(solver, FPFDMSolver)
        assert not inspect.isabstract(FPFDMSolver)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
