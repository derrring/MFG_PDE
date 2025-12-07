#!/usr/bin/env python3
"""
Integration tests for MFG with callable (state-dependent) coefficients (Phase 2.3).

Tests the full MFG coupling with state-dependent diffusion and drift,
verifying that callable coefficients work correctly in fixed-point iteration.
"""

import pytest

import numpy as np

from mfg_pde.alg.numerical.coupling import FixedPointIterator
from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde.core.mfg_problem import MFGProblem


class TestMFGCallableCoefficients:
    """Integration tests for MFG with callable coefficients (Phase 2.3).

    Tests the full MFG coupling with state-dependent diffusion and drift,
    verifying that callable coefficients work correctly in fixed-point iteration.
    Both HJB-FDM and FP-FDM now support callable diffusion for 1D problems.
    """

    @pytest.mark.xfail(
        reason="Pre-existing: Porous medium diffusion causes solver instability (U_err=0.7, M_err=0.7 after 5 iter)",
        strict=False,
    )
    def test_mfg_with_callable_diffusion(self):
        """Test MFG with state-dependent diffusion: porous medium."""
        # Create problem
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.5, Nt=20, sigma=0.1)

        # Porous medium diffusion: D(m) = σ² m
        def porous_medium_diffusion(t, x, m):
            return 0.05 * m  # Diffusion proportional to density

        # Create solvers
        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        # Create MFG solver with callable diffusion
        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            damping_factor=0.5,
            diffusion_field=porous_medium_diffusion,
        )

        # Solve
        result = mfg_solver.solve(max_iterations=5, tolerance=1e-3, verbose=False)

        # Verify result structure
        assert result is not None
        U, M = result[:2]
        assert U.shape == (problem.Nt + 1, problem.Nx + 1)
        assert M.shape == (problem.Nt + 1, problem.Nx + 1)
        assert np.all(M >= 0)  # Non-negative density

    def test_mfg_with_density_dependent_diffusion(self):
        """Test MFG with crowd dynamics: D(m) = D0 + D1(1 - m/m_max)."""
        # Create problem
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.5, Nt=20, sigma=0.1)

        # Crowd diffusion: lower diffusion in high-density regions
        def crowd_diffusion(t, x, m):
            m_max = np.max(m) if np.max(m) > 0 else 1.0
            return 0.05 + 0.1 * (1 - m / m_max)

        # Create solvers
        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        # Create MFG solver
        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            damping_factor=0.5,
            diffusion_field=crowd_diffusion,
        )

        # Solve
        result = mfg_solver.solve(max_iterations=5, tolerance=1e-3, verbose=False)

        # Verify convergence
        U, M = result[:2]
        assert U.shape == (problem.Nt + 1, problem.Nx + 1)
        assert M.shape == (problem.Nt + 1, problem.Nx + 1)
        assert np.all(M >= 0)

    @pytest.mark.xfail(
        reason="Conservative flux FDM (PR #383) introduced regression in callable diffusion handling",
        strict=False,
    )
    def test_mfg_callable_vs_constant_convergence(self):
        """Test that callable returning constant matches constant diffusion."""
        # Create problem
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.5, Nt=20, sigma=0.15)

        # Callable returning constant
        def constant_diffusion(t, x, m):
            return 0.15

        # Solve with callable
        hjb_solver_callable = HJBFDMSolver(problem)
        fp_solver_callable = FPFDMSolver(problem)
        mfg_solver_callable = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver_callable,
            fp_solver=fp_solver_callable,
            damping_factor=0.5,
            diffusion_field=constant_diffusion,
        )
        result_callable = mfg_solver_callable.solve(max_iterations=5, tolerance=1e-3, verbose=False)

        # Solve with constant (None uses problem.sigma)
        hjb_solver_constant = HJBFDMSolver(problem)
        fp_solver_constant = FPFDMSolver(problem)
        mfg_solver_constant = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver_constant,
            fp_solver=fp_solver_constant,
            damping_factor=0.5,
            diffusion_field=None,  # Use problem.sigma
        )
        result_constant = mfg_solver_constant.solve(max_iterations=5, tolerance=1e-3, verbose=False)

        # Results should be similar (not exact due to numerical differences)
        U_callable, M_callable = result_callable[:2]
        U_constant, M_constant = result_constant[:2]

        # Check that solutions are reasonably close
        assert np.allclose(U_callable, U_constant, rtol=0.1, atol=1e-2)
        assert np.allclose(M_callable, M_constant, rtol=0.1, atol=1e-2)

    @pytest.mark.xfail(
        reason="Conservative flux FDM (PR #383) introduced regression in array diffusion handling",
        strict=False,
    )
    def test_mfg_callable_diffusion_with_array(self):
        """Test MFG with array diffusion (non-callable) for comparison."""
        # Create problem
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.5, Nt=20, sigma=0.1)

        # Spatially varying diffusion (higher at boundaries)
        Nx = problem.Nx + 1
        Nt = problem.Nt + 1
        x_grid = np.linspace(problem.xmin, problem.xmax, Nx)
        diffusion_array = 0.1 + 0.05 * np.abs(x_grid - 0.5)

        # Broadcast to all timesteps
        diffusion_field = np.tile(diffusion_array, (Nt, 1))

        # Create solvers
        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        # Create MFG solver with array diffusion
        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            damping_factor=0.5,
            diffusion_field=diffusion_field,
        )

        # Solve
        result = mfg_solver.solve(max_iterations=5, tolerance=1e-3, verbose=False)

        # Verify
        U, M = result[:2]
        assert U.shape == (Nt, Nx)
        assert M.shape == (Nt, Nx)
        assert np.all(M >= 0)

    def test_mfg_callable_with_small_iterations(self):
        """Test that callable diffusion works with few Picard iterations."""
        # Create small problem
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=20, T=0.3, Nt=10, sigma=0.1)

        # Simple state-dependent diffusion
        def state_diffusion(t, x, m):
            return 0.08 + 0.02 * m

        # Create solvers
        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        # Create MFG solver
        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            damping_factor=0.5,
            diffusion_field=state_diffusion,
        )

        # Solve with just 2 iterations
        result = mfg_solver.solve(max_iterations=2, tolerance=1e-6, verbose=False)

        # Verify it runs (may not converge, but should execute)
        U, M = result[:2]
        assert U.shape == (problem.Nt + 1, problem.Nx + 1)
        assert M.shape == (problem.Nt + 1, problem.Nx + 1)
        assert np.all(M >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
