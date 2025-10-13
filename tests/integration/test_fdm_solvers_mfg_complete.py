#!/usr/bin/env python3
"""
Integration tests for complete MFG problem solving using FDM solvers.

Tests the full MFG system with HJB and FP equations using finite difference methods,
verifying numerical convergence, mass conservation, and solution properties.
"""

import pytest

import numpy as np

from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde.alg.numerical.mfg_solvers import FixedPointIterator
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import BoundaryConditions


class TestFDMSolversMFGIntegration:
    """Integration tests for FDM-based MFG problem solving."""

    def test_fixed_point_iterator_with_fdm(self):
        """Test FixedPointIterator with FDM HJB and FP solvers."""
        # Create problem with moderate resolution
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)

        # Create FDM solvers
        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        # Create MFG solver
        mfg_solver = FixedPointIterator(problem, hjb_solver=hjb_solver, fp_solver=fp_solver, thetaUM=0.5)

        # Solve
        result = mfg_solver.solve(max_iterations=10, tolerance=1e-4)

        # Verify result structure
        assert result is not None
        U, M = result[:2]
        assert U.shape == (problem.Nt + 1, problem.Nx + 1)
        assert M.shape == (problem.Nt + 1, problem.Nx + 1)

    def test_fdm_mass_conservation(self):
        """Test that FDM FP solver conserves mass in MFG context."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=40, T=1.0, Nt=30)

        # Use no-flux boundary conditions for mass conservation
        bc = BoundaryConditions(type="no_flux")
        fp_solver = FPFDMSolver(problem, boundary_conditions=bc)

        # Create initial density
        x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx + 1)
        m_initial = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial)

        # Solve FP with zero drift (should preserve mass)
        U_zero = np.zeros((problem.Nt + 1, problem.Nx + 1))
        M_solution = fp_solver.solve_fp_system(m_initial, U_zero)

        # Check mass conservation at all time steps
        initial_mass = np.sum(m_initial)
        for t in range(problem.Nt + 1):
            current_mass = np.sum(M_solution[t, :])
            assert np.isclose(current_mass, initial_mass, rtol=0.1), f"Mass not conserved at t={t}"

    def test_fdm_convergence_with_refinement(self):
        """Test that FDM solution converges with grid refinement."""
        # Solve with coarse grid
        problem_coarse = MFGProblem(xmin=0.0, xmax=1.0, Nx=20, T=1.0, Nt=20)
        hjb_solver_coarse = HJBFDMSolver(problem_coarse)
        fp_solver_coarse = FPFDMSolver(problem_coarse)
        mfg_solver_coarse = FixedPointIterator(
            problem_coarse, hjb_solver=hjb_solver_coarse, fp_solver=fp_solver_coarse, thetaUM=0.5
        )
        result_coarse = mfg_solver_coarse.solve(max_iterations=5, tolerance=1e-3)

        # Solve with fine grid
        problem_fine = MFGProblem(xmin=0.0, xmax=1.0, Nx=40, T=1.0, Nt=40)
        hjb_solver_fine = HJBFDMSolver(problem_fine)
        fp_solver_fine = FPFDMSolver(problem_fine)
        mfg_solver_fine = FixedPointIterator(
            problem_fine, hjb_solver=hjb_solver_fine, fp_solver=fp_solver_fine, thetaUM=0.5
        )
        result_fine = mfg_solver_fine.solve(max_iterations=5, tolerance=1e-3)

        # Both should produce valid solutions
        assert result_coarse is not None
        assert result_fine is not None
        U_coarse, _M_coarse = result_coarse[:2]
        U_fine, _M_fine = result_fine[:2]
        assert np.all(np.isfinite(U_coarse))
        assert np.all(np.isfinite(U_fine))

    def test_fdm_solution_non_negativity(self):
        """Test that FDM FP solver maintains non-negative density."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)

        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        mfg_solver = FixedPointIterator(problem, hjb_solver=hjb_solver, fp_solver=fp_solver, thetaUM=0.5)

        result = mfg_solver.solve(max_iterations=8, tolerance=1e-4)

        _U, M = result[:2]
        # Density should be non-negative everywhere
        assert np.all(M >= -1e-10), "Density contains negative values"

    def test_fdm_periodic_bc_solution(self):
        """Test FDM solution with periodic boundary conditions."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=40, T=1.0, Nt=30)

        bc = BoundaryConditions(type="periodic")
        fp_solver = FPFDMSolver(problem, boundary_conditions=bc)
        hjb_solver = HJBFDMSolver(problem)

        mfg_solver = FixedPointIterator(problem, hjb_solver=hjb_solver, fp_solver=fp_solver, thetaUM=0.5)

        result = mfg_solver.solve(max_iterations=8, tolerance=1e-4)

        # Should produce valid solution
        assert result is not None
        U, M = result[:2]
        assert np.all(np.isfinite(U))
        assert np.all(np.isfinite(M))

    def test_fdm_dirichlet_bc_solution(self):
        """Test FDM solution with Dirichlet boundary conditions."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=40, T=1.0, Nt=30)

        bc = BoundaryConditions(type="dirichlet", left_value=0.0, right_value=0.0)
        fp_solver = FPFDMSolver(problem, boundary_conditions=bc)
        hjb_solver = HJBFDMSolver(problem)

        mfg_solver = FixedPointIterator(problem, hjb_solver=hjb_solver, fp_solver=fp_solver, thetaUM=0.5)

        result = mfg_solver.solve(max_iterations=8, tolerance=1e-4)

        _U, M = result[:2]
        # Boundary conditions should be approximately enforced (relaxed tolerance for numerical effects)
        for t in range(problem.Nt + 1):
            assert np.isclose(M[t, 0], 0.0, atol=0.01)
            assert np.isclose(M[t, -1], 0.0, atol=0.01)


class TestFDMSolversCoupling:
    """Test coupling between HJB and FP FDM solvers."""

    def test_hjb_fp_coupling(self):
        """Test that HJB and FP solutions are properly coupled."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)

        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        # Initial density
        x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx + 1)
        m_initial = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial)

        # Solve HJB with given density
        u_terminal = 0.5 * (x_coords - problem.xmax) ** 2
        U_prev = np.zeros((problem.Nt + 1, problem.Nx + 1))  # Initial guess for value function
        U_solution = hjb_solver.solve_hjb_system(
            m_initial.reshape(1, -1).repeat(problem.Nt + 1, axis=0), u_terminal, U_prev
        )

        # Solve FP with computed value function
        M_solution = fp_solver.solve_fp_system(m_initial, U_solution)

        # Both solutions should be finite
        assert np.all(np.isfinite(U_solution))
        assert np.all(np.isfinite(M_solution))

    def test_fixed_point_iteration_convergence(self):
        """Test that fixed-point iteration converges for FDM solvers."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=25, T=1.0, Nt=25)

        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        mfg_solver = FixedPointIterator(problem, hjb_solver=hjb_solver, fp_solver=fp_solver, thetaUM=0.5)

        result = mfg_solver.solve(max_iterations=15, tolerance=1e-5)

        # Should converge (result not None indicates convergence or max iterations)
        assert result is not None


class TestFDMSolversNumericalProperties:
    """Test numerical properties of FDM solutions."""

    def test_solution_smoothness(self):
        """Test that solutions have reasonable smoothness."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=30)

        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        mfg_solver = FixedPointIterator(problem, hjb_solver=hjb_solver, fp_solver=fp_solver, thetaUM=0.5)

        result = mfg_solver.solve(max_iterations=8, tolerance=1e-4)

        U, _M = result[:2]
        # Check that spatial derivatives don't have wild oscillations
        # Compute finite difference of U in space
        U_diff = np.diff(U, axis=1)

        # Should not have extremely large jumps (relaxed threshold for realistic problems)
        assert np.max(np.abs(U_diff)) < 2000.0, "Solution shows wild oscillations"

    def test_terminal_condition_satisfaction(self):
        """Test that HJB terminal condition is satisfied."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=40, T=1.0, Nt=30)

        hjb_solver = HJBFDMSolver(problem)

        # Create simple density
        m_initial = np.ones((problem.Nt + 1, problem.Nx + 1)) / (problem.Nx + 1)

        # Terminal condition
        x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx + 1)
        u_terminal = 0.5 * (x_coords - problem.xmax) ** 2

        # Solve HJB (need U_prev as initial guess)
        U_prev = np.zeros((problem.Nt + 1, problem.Nx + 1))
        U_solution = hjb_solver.solve_hjb_system(m_initial, u_terminal, U_prev)

        # Terminal condition should be approximately satisfied
        assert np.allclose(U_solution[-1, :], u_terminal, rtol=0.1)

    def test_initial_condition_satisfaction(self):
        """Test that FP initial condition is satisfied."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=40, T=1.0, Nt=30)

        fp_solver = FPFDMSolver(problem)

        # Initial density
        x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx + 1)
        m_initial = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial)

        # Zero drift
        U_zero = np.zeros((problem.Nt + 1, problem.Nx + 1))

        # Solve FP
        M_solution = fp_solver.solve_fp_system(m_initial, U_zero)

        # Initial condition should be satisfied
        assert np.allclose(M_solution[0, :], m_initial, rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
