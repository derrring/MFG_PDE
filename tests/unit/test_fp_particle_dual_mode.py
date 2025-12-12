"""
Tests for FP particle solver modes.

Tests the hybrid mode (particle sampling with KDE output to grid).
Note: Collocation mode was removed in v0.17.0. Use FPGFDMSolver for
meshfree density evolution on collocation points.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfg_pde.alg.numerical.fp_solvers import FPParticleSolver, ParticleMode
from mfg_pde.core.mfg_problem import MFGProblem


class Simple2DMFGProblem(MFGProblem):
    """Simple 2D test problem for particle solver testing."""

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

    def test_hybrid_mode_is_default(self):
        """Test that hybrid mode is default."""
        problem = Simple2DMFGProblem()
        solver = FPParticleSolver(problem, num_particles=100)

        assert solver.mode == ParticleMode.HYBRID
        assert solver.fp_method_name == "Particle"
        assert solver.num_particles == 100

    def test_mode_string_conversion(self):
        """Test that mode parameter accepts strings."""
        problem = Simple2DMFGProblem()

        # Test hybrid as string
        solver_hybrid = FPParticleSolver(problem, mode="hybrid", num_particles=100)
        assert solver_hybrid.mode == ParticleMode.HYBRID


class TestCollocationModeDeprecation:
    """Test that deprecated collocation mode raises helpful error."""

    def test_collocation_mode_raises_error(self):
        """Test that collocation mode raises ValueError."""
        problem = Simple2DMFGProblem()

        with pytest.raises(ValueError, match="Collocation mode has been removed"):
            FPParticleSolver(problem, mode="collocation")

    def test_error_message_suggests_fpgfdmsolver(self):
        """Test that error message suggests FPGFDMSolver."""
        problem = Simple2DMFGProblem()

        with pytest.raises(ValueError, match="FPGFDMSolver"):
            FPParticleSolver(problem, mode="collocation")


class TestHybridMode:
    """Test hybrid mode (particle sampling with KDE output to grid)."""

    def test_hybrid_mode_output_shape(self):
        """Test that hybrid mode outputs on grid."""
        problem = Simple2DMFGProblem()
        solver = FPParticleSolver(problem, mode=ParticleMode.HYBRID, num_particles=1000)

        m0 = np.ones(problem.Nx + 1) / (problem.Nx + 1)
        U = np.zeros((problem.Nt + 1, problem.Nx + 1))

        M = solver.solve_fp_system(m0, U, show_progress=False)

        # Check output shape: should be (Nt+1, Nx+1) - grid output
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

    def test_hybrid_mode_non_negative_density(self):
        """Test that hybrid mode produces non-negative density."""
        problem = Simple2DMFGProblem()
        solver = FPParticleSolver(problem, num_particles=1000)

        m0 = np.ones(problem.Nx + 1) / (problem.Nx + 1)
        U = np.zeros((problem.Nt + 1, problem.Nx + 1))

        M = solver.solve_fp_system(m0, U, show_progress=False)

        # Density should be non-negative
        assert np.all(M >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
