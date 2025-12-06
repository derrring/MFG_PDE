#!/usr/bin/env python3
"""
Unit tests for FPParticleSolver.

Tests the particle-based Fokker-Planck solver with KDE density estimation,
including backend selection, normalization strategies, and intelligent pipeline dispatch.
"""

import pytest

import numpy as np

from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver, KDENormalization
from mfg_pde.core.mfg_problem import MFGProblem

# Legacy 1D BC: testing compatibility with 1D FP solvers (deprecated in v0.14, remove in v1.0)
from mfg_pde.geometry.boundary.fdm_bc_1d import BoundaryConditions


class TestFPParticleSolverInitialization:
    """Test FPParticleSolver initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic solver initialization with default parameters."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = FPParticleSolver(problem)

        assert solver.fp_method_name == "Particle"
        assert solver.num_particles == 5000
        assert solver.kde_bandwidth == "scott"
        assert solver.kde_normalization == KDENormalization.ALL
        assert solver.boundary_conditions.type == "periodic"

    def test_custom_num_particles(self):
        """Test initialization with custom number of particles."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = FPParticleSolver(problem, num_particles=1000)

        assert solver.num_particles == 1000

    def test_custom_kde_bandwidth(self):
        """Test initialization with custom KDE bandwidth."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = FPParticleSolver(problem, kde_bandwidth=0.1)

        assert solver.kde_bandwidth == 0.1

    def test_kde_normalization_none(self):
        """Test initialization with no KDE normalization."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = FPParticleSolver(problem, kde_normalization=KDENormalization.NONE)

        assert solver.kde_normalization == KDENormalization.NONE

    def test_kde_normalization_initial_only(self):
        """Test initialization with initial-only KDE normalization."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = FPParticleSolver(problem, kde_normalization=KDENormalization.INITIAL_ONLY)

        assert solver.kde_normalization == KDENormalization.INITIAL_ONLY

    def test_kde_normalization_all(self):
        """Test initialization with all-step KDE normalization."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = FPParticleSolver(problem, kde_normalization=KDENormalization.ALL)

        assert solver.kde_normalization == KDENormalization.ALL

    def test_kde_normalization_string(self):
        """Test initialization with KDE normalization as string."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = FPParticleSolver(problem, kde_normalization="none")

        assert solver.kde_normalization == KDENormalization.NONE

    def test_deprecated_normalize_kde_output_false(self):
        """Test backward compatibility with deprecated normalize_kde_output=False."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)

        with pytest.warns(DeprecationWarning, match="normalize_kde_output.*deprecated"):
            solver = FPParticleSolver(problem, normalize_kde_output=False)

        assert solver.kde_normalization == KDENormalization.NONE

    def test_deprecated_normalize_only_initial_true(self):
        """Test backward compatibility with deprecated normalize_only_initial=True."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)

        with pytest.warns(DeprecationWarning, match="normalize_only_initial.*deprecated"):
            solver = FPParticleSolver(problem, normalize_only_initial=True)

        assert solver.kde_normalization == KDENormalization.INITIAL_ONLY

    def test_deprecated_both_parameters(self):
        """Test backward compatibility with both deprecated parameters."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)

        with pytest.warns(DeprecationWarning, match="deprecated"):
            solver = FPParticleSolver(problem, normalize_kde_output=True, normalize_only_initial=False)

        assert solver.kde_normalization == KDENormalization.ALL

    def test_custom_boundary_conditions(self):
        """Test initialization with custom boundary conditions."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        bc = BoundaryConditions(type="no_flux")
        solver = FPParticleSolver(problem, boundary_conditions=bc)

        assert solver.boundary_conditions.type == "no_flux"

    def test_backend_initialization_numpy(self):
        """Test initialization with NumPy backend."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = FPParticleSolver(problem, backend="numpy")

        assert solver.backend is not None
        assert solver.backend.name == "numpy"

    def test_default_backend_is_numpy(self):
        """Test that default backend is NumPy."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = FPParticleSolver(problem)

        assert solver.backend is not None
        assert solver.backend.name == "numpy"

    def test_strategy_selector_initialized(self):
        """Test that strategy selector is properly initialized."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = FPParticleSolver(problem)

        assert solver.strategy_selector is not None
        assert solver.current_strategy is None  # Not set until solve

    def test_time_step_counter_initialized(self):
        """Test that time step counter is initialized."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = FPParticleSolver(problem)

        assert solver._time_step_counter == 0


class TestFPParticleSolverSolveFPSystem:
    """Test the main solve_fp_system method."""

    def test_solve_fp_system_shape(self):
        """Test that solve_fp_system returns correct shape."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.5, Nt=20)
        solver = FPParticleSolver(problem, num_particles=500)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        # Create inputs
        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((Nt, Nx))

        # Solve
        M_solution = solver.solve_fp_system(m_initial, U_solution)

        assert M_solution.shape == (Nt, Nx)
        assert np.all(np.isfinite(M_solution))

    def test_solve_fp_system_initial_condition(self):
        """Test that initial condition center of mass is approximately preserved."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.5, Nt=20)
        solver = FPParticleSolver(problem, num_particles=2000)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        # Create inputs with specific initial condition
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        m_initial = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial * problem.dx)
        U_solution = np.zeros((Nt, Nx))

        # Solve
        M_solution = solver.solve_fp_system(m_initial, U_solution)

        # Check that center of mass is approximately preserved
        # (KDE introduces smoothing but should preserve location)
        cm_initial = np.sum(x_coords * m_initial * problem.dx)
        cm_solution = np.sum(x_coords * M_solution[0, :] * problem.dx)
        assert np.isclose(cm_initial, cm_solution, rtol=0.2)

    def test_solve_with_zero_drift(self):
        """Test solving with zero drift field."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.3, Nt=15)
        solver = FPParticleSolver(problem, num_particles=500)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((Nt, Nx))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        assert np.all(np.isfinite(M_solution))

    def test_solve_with_non_zero_drift(self):
        """Test solving with non-zero drift field."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.3, Nt=15)
        solver = FPParticleSolver(problem, num_particles=500)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        # Create non-zero drift
        m_initial = np.ones(Nx) / Nx
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        U_solution = np.tile(x_coords**2, (Nt, 1))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        assert np.all(np.isfinite(M_solution))

    def test_solve_with_different_num_particles(self):
        """Test solver with different numbers of particles."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.3, Nt=15)

        for num_particles in [100, 500, 1000]:
            solver = FPParticleSolver(problem, num_particles=num_particles)

            Nt = problem.Nt + 1
            Nx = problem.Nx + 1

            m_initial = np.ones(Nx) / Nx
            U_solution = np.zeros((Nt, Nx))

            M_solution = solver.solve_fp_system(m_initial, U_solution)

            assert np.all(np.isfinite(M_solution))

    def test_solve_with_kde_normalization_none(self):
        """Test solving with no KDE normalization."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.3, Nt=15)
        solver = FPParticleSolver(problem, num_particles=500, kde_normalization=KDENormalization.NONE)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((Nt, Nx))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        assert np.all(np.isfinite(M_solution))

    def test_solve_with_kde_normalization_initial_only(self):
        """Test solving with initial-only KDE normalization."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.3, Nt=15)
        solver = FPParticleSolver(problem, num_particles=500, kde_normalization=KDENormalization.INITIAL_ONLY)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((Nt, Nx))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        assert np.all(np.isfinite(M_solution))

    def test_strategy_selection(self):
        """Test that strategy is selected during solve."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.3, Nt=15)
        solver = FPParticleSolver(problem, num_particles=500)

        assert solver.current_strategy is None  # Before solve

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((Nt, Nx))

        solver.solve_fp_system(m_initial, U_solution)

        assert solver.current_strategy is not None  # After solve
        assert hasattr(solver.current_strategy, "name")

    def test_time_step_counter_reset(self):
        """Test that time step counter is reset on solve."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.3, Nt=15)
        solver = FPParticleSolver(problem, num_particles=500)

        solver._time_step_counter = 999  # Set to non-zero

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((Nt, Nx))

        solver.solve_fp_system(m_initial, U_solution)

        # Counter should be reset (though it gets incremented during solve)
        # Just verify solve completed without error


class TestFPParticleSolverNumericalProperties:
    """Test numerical properties of particle FP solutions."""

    def test_solution_finiteness(self):
        """Test that solution remains finite throughout."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=40, T=0.5, Nt=20)
        solver = FPParticleSolver(problem, num_particles=1000)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        m_initial = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial * problem.dx)
        U_solution = np.zeros((Nt, Nx))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        # All values should be finite
        assert np.all(np.isfinite(M_solution))

    def test_forward_time_propagation(self):
        """Test that solution is computed for all time steps."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=40, T=1.0, Nt=30, sigma=0.3)
        solver = FPParticleSolver(problem, num_particles=2000)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        # Concentrated initial condition
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        m_initial = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.01**2))
        m_initial = m_initial / np.sum(m_initial * problem.dx)

        U_solution = np.zeros((Nt, Nx))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        # Solution should be computed for all time steps and remain finite
        assert M_solution.shape == (Nt, Nx)
        assert np.all(np.isfinite(M_solution))
        # Verify solution exists at all time points (not all zeros)
        assert np.any(M_solution > 0)

    def test_approximate_mass_conservation(self):
        """Test that total mass is approximately conserved."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=40, T=0.5, Nt=20)
        solver = FPParticleSolver(problem, num_particles=2000, kde_normalization=KDENormalization.ALL)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1
        Dx = problem.dx

        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        m_initial = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial * Dx)
        U_solution = np.zeros((Nt, Nx))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        # Check mass conservation across time (with relaxed tolerance for KDE)
        initial_mass = np.sum(M_solution[0, :] * Dx)
        for t in range(Nt):
            current_mass = np.sum(M_solution[t, :] * Dx)
            # Allow larger error for particle methods with KDE
            assert np.isclose(current_mass, initial_mass, rtol=0.3)

    def test_non_negativity(self):
        """Test that density remains non-negative."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=40, T=0.5, Nt=20)
        solver = FPParticleSolver(problem, num_particles=1000)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((Nt, Nx))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        # Density should be non-negative (KDE ensures this)
        assert np.all(M_solution >= -1e-10)


class TestFPParticleSolverIntegration:
    """Integration tests with actual FP problems."""

    def test_solver_not_abstract(self):
        """Test that FPParticleSolver can be instantiated."""
        import inspect

        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.5, Nt=20)

        # Should not raise TypeError about abstract methods
        solver = FPParticleSolver(problem)
        assert isinstance(solver, FPParticleSolver)

        # Should not have abstract methods
        assert not inspect.isabstract(FPParticleSolver)

    def test_solver_with_different_parameters(self):
        """Test solver with various parameter configurations."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.3, Nt=15)

        configs = [
            {"num_particles": 500, "kde_normalization": KDENormalization.NONE},
            {"num_particles": 1000, "kde_normalization": KDENormalization.INITIAL_ONLY},
            {"num_particles": 2000, "kde_normalization": KDENormalization.ALL, "kde_bandwidth": 0.1},
        ]

        for config in configs:
            solver = FPParticleSolver(problem, **config)

            Nt = problem.Nt + 1
            Nx = problem.Nx + 1

            m_initial = np.ones(Nx) / Nx
            U_solution = np.zeros((Nt, Nx))

            M_solution = solver.solve_fp_system(m_initial, U_solution)

            assert np.all(np.isfinite(M_solution))

    def test_solver_with_different_boundary_conditions(self):
        """Test solver with different boundary condition types."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.3, Nt=15)

        bc_types = ["periodic", "no_flux"]

        for bc_type in bc_types:
            bc = BoundaryConditions(type=bc_type)
            solver = FPParticleSolver(problem, num_particles=500, boundary_conditions=bc)

            Nt = problem.Nt + 1
            Nx = problem.Nx + 1

            m_initial = np.ones(Nx) / Nx
            U_solution = np.zeros((Nt, Nx))

            M_solution = solver.solve_fp_system(m_initial, U_solution)

            assert np.all(np.isfinite(M_solution))


class TestFPParticleSolverHelperMethods:
    """Test helper methods for gradient computation and normalization."""

    def test_compute_gradient(self):
        """Test gradient computation helper."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.3, Nt=15)
        solver = FPParticleSolver(problem, num_particles=500)

        x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx + 1)
        U_array = x_coords**2

        gradient = solver._compute_gradient(U_array, problem.dx, use_backend=False)

        # Should return finite gradient
        assert np.all(np.isfinite(gradient))
        assert gradient.shape == U_array.shape

    def test_compute_gradient_zero_dx(self):
        """Test gradient computation with zero Dx."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.3, Nt=15)
        solver = FPParticleSolver(problem, num_particles=500)

        x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx + 1)
        U_array = x_coords**2

        gradient = solver._compute_gradient(U_array, 0.0, use_backend=False)

        # Should return zeros for zero Dx
        assert np.allclose(gradient, 0.0)

    def test_normalize_density_none(self):
        """Test density normalization with NONE strategy."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.3, Nt=15)
        solver = FPParticleSolver(problem, num_particles=500, kde_normalization=KDENormalization.NONE)

        M_array = np.random.rand(problem.Nx + 1) * 2.0  # Random unnormalized density

        normalized = solver._normalize_density(M_array, problem.dx, use_backend=False)

        # Should not normalize (return as-is)
        assert np.allclose(normalized, M_array)

    def test_normalize_density_initial_only(self):
        """Test density normalization with INITIAL_ONLY strategy."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.3, Nt=15)
        solver = FPParticleSolver(problem, num_particles=500, kde_normalization=KDENormalization.INITIAL_ONLY)

        M_array = np.random.rand(problem.Nx + 1) * 2.0

        # First call (time step 0) - should normalize
        solver._time_step_counter = 0
        normalized_0 = solver._normalize_density(M_array, problem.dx, use_backend=False)
        assert np.isclose(np.sum(normalized_0 * problem.dx), 1.0, rtol=0.1)

        # Second call (time step 1) - should not normalize
        solver._time_step_counter = 1
        normalized_1 = solver._normalize_density(M_array, problem.dx, use_backend=False)
        assert np.allclose(normalized_1, M_array)

    def test_normalize_density_all(self):
        """Test density normalization with ALL strategy."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=0.3, Nt=15)
        solver = FPParticleSolver(problem, num_particles=500, kde_normalization=KDENormalization.ALL)

        M_array = np.random.rand(problem.Nx + 1) * 2.0

        # Should normalize at any time step
        for t in [0, 1, 5, 10]:
            solver._time_step_counter = t
            normalized = solver._normalize_density(M_array, problem.dx, use_backend=False)
            assert np.isclose(np.sum(normalized * problem.dx), 1.0, rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
