"""
Tests for FP particle solver.

Tests the particle-based FP solver using Monte Carlo sampling and KDE.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfg_pde.alg.numerical.fp_solvers import FPParticleSolver, KDENormalization
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


class TestFPParticleSolverBasic:
    """Test basic FPParticleSolver functionality."""

    def test_default_initialization(self):
        """Test default initialization."""
        problem = Simple2DMFGProblem()
        solver = FPParticleSolver(problem, num_particles=100)

        assert solver.fp_method_name == "Particle"
        assert solver.num_particles == 100

    def test_output_shape(self):
        """Test that solver outputs density on grid."""
        problem = Simple2DMFGProblem()
        solver = FPParticleSolver(problem, num_particles=1000)

        m0 = np.ones(problem.Nx + 1) / (problem.Nx + 1)
        U = np.zeros((problem.Nt + 1, problem.Nx + 1))

        M = solver.solve_fp_system(m0, U, show_progress=False)

        # Output shape should be (Nt+1, Nx+1) - grid output via KDE
        assert M.shape == (problem.Nt + 1, problem.Nx + 1)

    def test_non_negative_density(self):
        """Test that solver produces non-negative density."""
        problem = Simple2DMFGProblem()
        solver = FPParticleSolver(problem, num_particles=1000)

        m0 = np.ones(problem.Nx + 1) / (problem.Nx + 1)
        U = np.zeros((problem.Nt + 1, problem.Nx + 1))

        M = solver.solve_fp_system(m0, U, show_progress=False)

        assert np.all(M >= 0)
        assert np.all(np.isfinite(M))


class TestKDENormalization:
    """Test KDE normalization options."""

    def test_kde_normalization_enum(self):
        """Test KDENormalization enum values."""
        assert KDENormalization.NONE == "none"
        assert KDENormalization.INITIAL_ONLY == "initial_only"
        assert KDENormalization.ALL == "all"

    def test_kde_normalization_string_conversion(self):
        """Test that kde_normalization accepts strings."""
        problem = Simple2DMFGProblem()

        solver = FPParticleSolver(problem, num_particles=100, kde_normalization="all")
        assert solver.kde_normalization == KDENormalization.ALL

        solver = FPParticleSolver(problem, num_particles=100, kde_normalization="none")
        assert solver.kde_normalization == KDENormalization.NONE


class TestDeprecatedParameters:
    """Test deprecated parameters raise appropriate warnings/errors."""

    def test_collocation_mode_raises_error(self):
        """Test that mode='collocation' raises ValueError."""
        problem = Simple2DMFGProblem()

        with pytest.raises(ValueError, match="Collocation mode has been removed"):
            FPParticleSolver(problem, mode="collocation")

    def test_error_suggests_fpgfdmsolver(self):
        """Test that error message suggests FPGFDMSolver."""
        problem = Simple2DMFGProblem()

        with pytest.raises(ValueError, match="FPGFDMSolver"):
            FPParticleSolver(problem, mode="collocation")

    def test_hybrid_mode_accepted(self):
        """Test that mode='hybrid' is accepted for backward compatibility."""
        problem = Simple2DMFGProblem()

        # Should not raise
        solver = FPParticleSolver(problem, mode="hybrid", num_particles=100)
        assert solver.num_particles == 100

    def test_external_particles_warns(self):
        """Test that external_particles raises deprecation warning."""
        problem = Simple2DMFGProblem()
        points = np.random.rand(50, 2)

        with pytest.warns(DeprecationWarning, match="external_particles"):
            FPParticleSolver(problem, num_particles=100, external_particles=points)

    def test_unknown_mode_raises_error(self):
        """Test that unknown mode raises ValueError."""
        problem = Simple2DMFGProblem()

        with pytest.raises(ValueError, match="Unknown mode"):
            FPParticleSolver(problem, mode="unknown")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
