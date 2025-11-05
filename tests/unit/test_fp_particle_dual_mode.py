"""
Tests for dual-mode FP particle solver.

Tests the new collocation mode capability that enables true meshfree MFG
when combined with particle-collocation HJB solvers (GFDM).
"""

from __future__ import annotations

import pytest

import numpy as np

from mfg_pde.alg.numerical.fp_solvers import FPParticleSolver, ParticleMode
from mfg_pde.core.mfg_problem import MFGProblem


class Simple2DMFGProblem(MFGProblem):
    """Simple 2D test problem for dual-mode testing."""

    def __init__(self):
        super().__init__(
            T=1.0,
            Nt=10,
            Nx=20,
            Lx=1.0,
            xmin=0.0,
            sigma=0.1,
            coupling_coefficient=0.5,
            dimension=2,
        )


class TestParticleMode:
    """Test ParticleMode enum and mode selection."""

    def test_particle_mode_enum_values(self):
        """Test that ParticleMode enum has correct values."""
        assert ParticleMode.HYBRID == "hybrid"
        assert ParticleMode.COLLOCATION == "collocation"

    def test_hybrid_mode_is_default(self):
        """Test that hybrid mode is default (backward compatibility)."""
        problem = Simple2DMFGProblem()
        solver = FPParticleSolver(problem, num_particles=100)

        assert solver.mode == ParticleMode.HYBRID
        assert solver.fp_method_name == "Particle"
        assert solver.num_particles == 100
        assert solver.collocation_points is None

    def test_mode_string_conversion(self):
        """Test that mode parameter accepts strings."""
        problem = Simple2DMFGProblem()

        # Test hybrid as string
        solver_hybrid = FPParticleSolver(problem, mode="hybrid", num_particles=100)
        assert solver_hybrid.mode == ParticleMode.HYBRID

        # Test collocation as string (with external_particles)
        points = np.random.uniform(0, 1, (50, 2))
        solver_coll = FPParticleSolver(problem, mode="collocation", external_particles=points)
        assert solver_coll.mode == ParticleMode.COLLOCATION


class TestCollocationMode:
    """Test collocation mode functionality."""

    def test_collocation_mode_requires_external_particles(self):
        """Test that collocation mode validates external_particles."""
        problem = Simple2DMFGProblem()

        with pytest.raises(ValueError, match="requires external_particles"):
            FPParticleSolver(problem, mode=ParticleMode.COLLOCATION)

    def test_collocation_mode_validates_particle_shape(self):
        """Test that collocation mode validates external_particles shape."""
        problem = Simple2DMFGProblem()

        # 1D array should fail
        points_1d = np.random.uniform(0, 1, 100)
        with pytest.raises(ValueError, match="must be 2D array"):
            FPParticleSolver(problem, mode=ParticleMode.COLLOCATION, external_particles=points_1d)

    def test_collocation_mode_configuration(self):
        """Test that collocation mode configures solver correctly."""
        problem = Simple2DMFGProblem()
        points = np.random.uniform(0, 1, (50, 2))

        solver = FPParticleSolver(problem, mode=ParticleMode.COLLOCATION, external_particles=points)

        assert solver.mode == ParticleMode.COLLOCATION
        assert solver.fp_method_name == "Particle-Collocation"
        assert solver.num_particles == 50
        assert solver.collocation_points is not None
        assert solver.collocation_points.shape == (50, 2)
        # Check that collocation_points is a copy
        assert solver.collocation_points is not points

    def test_collocation_mode_output_shape(self):
        """Test that collocation mode outputs on particles."""
        problem = Simple2DMFGProblem()
        N_points = 100
        points = np.random.uniform(0, 1, (N_points, 2))

        solver = FPParticleSolver(problem, mode=ParticleMode.COLLOCATION, external_particles=points)

        # Create dummy initial condition and value function
        m0 = np.ones(N_points) / N_points
        U = np.zeros((problem.Nt + 1, N_points))

        # Solve FP system
        M = solver.solve_fp_system(m0, U, show_progress=False)

        # Check output shape: should be (Nt, N_particles) not (Nt, Nx)
        assert M.shape == (problem.Nt + 1, N_points)

    def test_collocation_mode_mass_conservation(self):
        """Test that collocation mode conserves mass."""
        problem = Simple2DMFGProblem()
        N_points = 100
        points = np.random.uniform(0, 1, (N_points, 2))

        solver = FPParticleSolver(problem, mode=ParticleMode.COLLOCATION, external_particles=points)

        # Initial condition with unit mass
        m0 = np.ones(N_points) / N_points
        U = np.zeros((problem.Nt + 1, N_points))

        M = solver.solve_fp_system(m0, U, show_progress=False)

        # Check mass conservation at each time step
        for t in range(problem.Nt + 1):
            mass = np.sum(M[t, :])
            assert np.abs(mass - 1.0) < 1e-10  # Mass should be preserved

    def test_collocation_mode_validates_input_shapes(self):
        """Test that collocation mode validates input shapes."""
        problem = Simple2DMFGProblem()
        N_points = 100
        points = np.random.uniform(0, 1, (N_points, 2))

        solver = FPParticleSolver(problem, mode=ParticleMode.COLLOCATION, external_particles=points)

        # Wrong m0 shape
        m0_wrong = np.ones(50)  # Should be N_points
        U = np.zeros((problem.Nt + 1, N_points))

        with pytest.raises(ValueError, match="must match collocation_points count"):
            solver.solve_fp_system(m0_wrong, U, show_progress=False)

        # Wrong U shape
        m0 = np.ones(N_points) / N_points
        U_wrong = np.zeros((problem.Nt + 1, 50))  # Should be N_points

        with pytest.raises(ValueError, match="must be"):
            solver.solve_fp_system(m0, U_wrong, show_progress=False)


class TestHybridMode:
    """Test that hybrid mode still works (backward compatibility)."""

    def test_hybrid_mode_output_shape(self):
        """Test that hybrid mode outputs on grid (existing behavior)."""
        problem = Simple2DMFGProblem()
        solver = FPParticleSolver(problem, mode=ParticleMode.HYBRID, num_particles=1000)

        m0 = np.ones(problem.Nx + 1) / (problem.Nx + 1)
        U = np.zeros((problem.Nt + 1, problem.Nx + 1))

        M = solver.solve_fp_system(m0, U, show_progress=False)

        # Check output shape: should be (Nt, Nx) - grid output
        assert M.shape == (problem.Nt + 1, problem.Nx + 1)

    def test_existing_code_unchanged(self):
        """Test that existing code without mode parameter works."""
        problem = Simple2DMFGProblem()

        # Old-style usage (no mode parameter)
        solver = FPParticleSolver(problem, num_particles=1000)

        m0 = np.ones(problem.Nx + 1) / (problem.Nx + 1)
        U = np.zeros((problem.Nt + 1, problem.Nx + 1))

        M = solver.solve_fp_system(m0, U, show_progress=False)

        # Should work exactly as before
        assert M.shape == (problem.Nt + 1, problem.Nx + 1)
        assert solver.mode == ParticleMode.HYBRID


class TestModeSwitching:
    """Test switching between modes."""

    def test_different_solvers_different_modes(self):
        """Test that different solver instances can use different modes."""
        problem = Simple2DMFGProblem()
        points = np.random.uniform(0, 1, (100, 2))

        # Create two solvers with different modes
        solver_hybrid = FPParticleSolver(problem, mode=ParticleMode.HYBRID, num_particles=1000)
        solver_coll = FPParticleSolver(problem, mode=ParticleMode.COLLOCATION, external_particles=points)

        assert solver_hybrid.mode == ParticleMode.HYBRID
        assert solver_coll.mode == ParticleMode.COLLOCATION

        # They should produce different output shapes
        m0_grid = np.ones(problem.Nx + 1) / (problem.Nx + 1)
        U_grid = np.zeros((problem.Nt + 1, problem.Nx + 1))
        M_hybrid = solver_hybrid.solve_fp_system(m0_grid, U_grid, show_progress=False)

        m0_particles = np.ones(100) / 100
        U_particles = np.zeros((problem.Nt + 1, 100))
        M_coll = solver_coll.solve_fp_system(m0_particles, U_particles, show_progress=False)

        assert M_hybrid.shape == (problem.Nt + 1, problem.Nx + 1)
        assert M_coll.shape == (problem.Nt + 1, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
