"""
Integration test for particle-collocation MFG workflow.

Tests the complete MFG system using:
- HJBGFDMSolver (particle-collocation HJB)
- FPParticleSolver in collocation mode (particle-based FP)

Both solvers share the same particle discretization - true meshfree MFG.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfg_pde.alg.numerical.fp_solvers import FPParticleSolver, ParticleMode
from mfg_pde.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry.implicit import Hyperrectangle


class SimpleLQMFG2D(MFGProblem):
    """Simple 2D LQ-MFG problem for integration testing."""

    def __init__(self):
        super().__init__(
            T=1.0,
            Nt=20,
            Nx=30,
            Lx=1.0,
            xmin=0.0,
            sigma=0.2,
            coefCT=0.5,
            dimension=2,
        )
        # GFDM solver expects problem.d for spatial dimension
        self.d = 2


class TestParticleCollocationIntegration:
    """Integration tests for particle-collocation MFG workflow."""

    def test_shared_discretization(self):
        """Test that HJB and FP solvers can share same particle discretization."""
        problem = SimpleLQMFG2D()
        N_points = 200

        # Sample collocation points once
        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        points = domain.sample_uniform(N_points, seed=42)

        # Create HJB solver on particles
        hjb_solver = HJBGFDMSolver(problem, collocation_points=points, delta=0.15)

        # Create FP solver in collocation mode on same particles
        fp_solver = FPParticleSolver(problem, mode=ParticleMode.COLLOCATION, external_particles=points)

        # Verify both solvers use the same particles
        assert hjb_solver.collocation_points.shape == (N_points, 2)
        assert fp_solver.collocation_points.shape == (N_points, 2)
        assert fp_solver.num_particles == N_points

    def test_single_picard_iteration(self):
        """Test a single Picard iteration workflow compatibility."""
        problem = SimpleLQMFG2D()
        N_points = 150

        # Sample collocation points
        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        points = domain.sample_uniform(N_points, seed=42)

        # Create solvers
        _hjb_solver = HJBGFDMSolver(problem, collocation_points=points, delta=0.15)
        fp_solver = FPParticleSolver(problem, mode="collocation", external_particles=points)

        # Initial density (uniform)
        m0 = np.ones(N_points) / N_points

        # Create dummy value function (in real workflow, would come from HJB solve)
        # This tests that FP solver accepts the right shape from GFDM output
        U_dummy = np.zeros((problem.Nt + 1, N_points))

        # FP solve with dummy U
        M = fp_solver.solve_fp_system(m0, U_dummy, show_progress=False)

        # Verify output shapes match (same spatial discretization)
        assert U_dummy.shape == (problem.Nt + 1, N_points)
        assert M.shape == (problem.Nt + 1, N_points)

        # Verify mass conservation
        for t_idx in range(problem.Nt + 1):
            mass = np.sum(M[t_idx, :])
            assert np.abs(mass - 1.0) < 1e-10

    def test_shape_compatibility(self):
        """Test that output shapes are compatible for Picard iteration."""
        problem = SimpleLQMFG2D()
        N_points = 100

        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        points = domain.sample_uniform(N_points, seed=42)

        _hjb_solver = HJBGFDMSolver(problem, collocation_points=points, delta=0.15)
        fp_solver = FPParticleSolver(problem, mode="collocation", external_particles=points)

        # Initial conditions
        m0 = np.ones(N_points) / N_points

        # Dummy value functions (in real workflow, would come from HJB solve)
        U1 = np.zeros((problem.Nt + 1, N_points))
        U2 = np.zeros((problem.Nt + 1, N_points))

        # First iteration
        M1 = fp_solver.solve_fp_system(m0, U1, show_progress=False)

        # Second iteration should accept outputs from first
        M2 = fp_solver.solve_fp_system(M1[0, :], U2, show_progress=False)

        # Verify iterations are compatible
        assert U1.shape == U2.shape
        assert M1.shape == M2.shape

    def test_collocation_vs_hybrid_mode(self):
        """Test that collocation mode outputs on particles, hybrid on grid."""
        problem = SimpleLQMFG2D()
        N_points = 100

        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        points = domain.sample_uniform(N_points, seed=42)

        # Collocation mode
        fp_coll = FPParticleSolver(problem, mode="collocation", external_particles=points)

        # Hybrid mode (default)
        fp_hybrid = FPParticleSolver(problem, num_particles=1000)

        # Dummy inputs
        m0_coll = np.ones(N_points) / N_points
        U_coll = np.zeros((problem.Nt + 1, N_points))

        m0_grid = np.ones(problem.Nx + 1) / (problem.Nx + 1)
        U_grid = np.zeros((problem.Nt + 1, problem.Nx + 1))

        # Solve
        M_coll = fp_coll.solve_fp_system(m0_coll, U_coll, show_progress=False)
        M_hybrid = fp_hybrid.solve_fp_system(m0_grid, U_grid, show_progress=False)

        # Verify different output shapes
        assert M_coll.shape == (problem.Nt + 1, N_points)  # Particle output
        assert M_hybrid.shape == (problem.Nt + 1, problem.Nx + 1)  # Grid output

    def test_mass_conservation_through_picard(self):
        """Test mass conservation through multiple Picard iterations."""
        problem = SimpleLQMFG2D()
        N_points = 100

        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        points = domain.sample_uniform(N_points, seed=42)

        fp_solver = FPParticleSolver(problem, mode="collocation", external_particles=points)

        # Initial density
        m0 = np.ones(N_points) / N_points
        M_current = m0.copy()

        # Run 3 Picard iterations (with dummy U)
        for _ in range(3):
            U_dummy = np.zeros((problem.Nt + 1, N_points))

            M_new = fp_solver.solve_fp_system(M_current, U_dummy, show_progress=False)

            # Check mass conservation at each time step
            for t_idx in range(problem.Nt + 1):
                mass = np.sum(M_new[t_idx, :])
                assert np.abs(mass - 1.0) < 1e-10, f"Mass not conserved at t={t_idx}"

            M_current = M_new[-1, :]  # Use final density for next iteration


class TestCollocationModeValidation:
    """Test validation and error handling in collocation mode."""

    def test_requires_external_particles(self):
        """Test that collocation mode requires external_particles."""
        problem = SimpleLQMFG2D()

        with pytest.raises(ValueError, match="requires external_particles"):
            FPParticleSolver(problem, mode="collocation")

    def test_validates_particle_dimension(self):
        """Test that external_particles must be 2D array."""
        problem = SimpleLQMFG2D()

        points_1d = np.random.uniform(0, 1, 100)
        with pytest.raises(ValueError, match="must be 2D array"):
            FPParticleSolver(problem, mode="collocation", external_particles=points_1d)

    def test_validates_input_shapes_in_solve(self):
        """Test that solve_fp_system validates input shapes."""
        problem = SimpleLQMFG2D()
        N_points = 100

        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        points = domain.sample_uniform(N_points, seed=42)

        solver = FPParticleSolver(problem, mode="collocation", external_particles=points)

        # Wrong m0 shape
        m0_wrong = np.ones(50)
        U_correct = np.zeros((problem.Nt + 1, N_points))

        with pytest.raises(ValueError, match="must match collocation_points count"):
            solver.solve_fp_system(m0_wrong, U_correct, show_progress=False)

        # Wrong U shape
        m0_correct = np.ones(N_points) / N_points
        U_wrong = np.zeros((problem.Nt + 1, 50))

        with pytest.raises(ValueError, match="must be"):
            solver.solve_fp_system(m0_correct, U_wrong, show_progress=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
