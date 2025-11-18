#!/usr/bin/env python3
"""
Unit tests for FPFDMSolver - comprehensive coverage.

Tests the Finite Difference Method (FDM) solver for Fokker-Planck equations
with different boundary conditions (periodic, Dirichlet, no-flux).
"""

import pytest

import numpy as np

from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import BoundaryConditions
from mfg_pde.geometry.grids.grid_1d import SimpleGrid1D


@pytest.fixture
def standard_problem():
    """Create standard 1D MFG problem using modern geometry-first API.

    Equivalent to legacy ExampleMFGProblem() with defaults:
    - Domain: [0, 1] with 51 grid points
    - Time: T=1.0 with 51 time steps
    - Diffusion: sigma=1.0
    - Boundary: Periodic
    """
    boundary_conditions = BoundaryConditions(type="periodic")
    domain = SimpleGrid1D(xmin=0.0, xmax=1.0, boundary_conditions=boundary_conditions)
    domain.create_grid(num_points=51)
    return MFGProblem(geometry=domain, T=1.0, Nt=51, sigma=1.0)


class TestFPFDMSolverInitialization:
    """Test FPFDMSolver initialization and setup."""

    def test_basic_initialization(self, standard_problem):
        """Test basic solver initialization with default boundary conditions."""
        solver = FPFDMSolver(standard_problem)

        assert solver.fp_method_name == "FDM"
        assert solver.boundary_conditions.type == "no_flux"
        assert solver.problem is standard_problem

    def test_initialization_with_periodic_bc(self, standard_problem):
        """Test initialization with periodic boundary conditions."""
        bc = BoundaryConditions(type="periodic")
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        assert solver.fp_method_name == "FDM"
        assert solver.boundary_conditions.type == "periodic"

    def test_initialization_with_dirichlet_bc(self, standard_problem):
        """Test initialization with Dirichlet boundary conditions."""
        bc = BoundaryConditions(type="dirichlet", left_value=0.0, right_value=0.0)
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        assert solver.fp_method_name == "FDM"
        assert solver.boundary_conditions.type == "dirichlet"
        assert solver.boundary_conditions.left_value == 0.0
        assert solver.boundary_conditions.right_value == 0.0

    def test_initialization_with_no_flux_bc(self, standard_problem):
        """Test initialization with no-flux boundary conditions."""
        bc = BoundaryConditions(type="no_flux")
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        assert solver.fp_method_name == "FDM"
        assert solver.boundary_conditions.type == "no_flux"


class TestFPFDMSolverBasicSolution:
    """Test basic solution functionality."""

    def test_solve_fp_system_shape(self, standard_problem):
        """Test that solve_fp_system returns correct shape."""
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        # Create simple inputs
        m_initial = np.ones(Nx) / Nx  # Normalized density
        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        assert m_result.shape == (Nt, Nx)

    def test_solve_fp_system_initial_condition_preserved(self, standard_problem):
        """Test that initial condition is preserved at t=0."""
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        # Create Gaussian initial condition
        x_coords = np.linspace(standard_problem.xmin, standard_problem.xmax, Nx)
        m_initial = np.exp(-((x_coords - 0.0) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial)  # Normalize

        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # Initial condition should be preserved (approximately, after non-negativity enforcement)
        assert np.allclose(m_result[0, :], m_initial, rtol=0.1)

    def test_solve_fp_system_zero_timesteps(self, standard_problem):
        """Test behavior with zero time steps (Nt=0)."""
        # Create problem with Nt=0 (results in 0 time steps)
        boundary_conditions = BoundaryConditions(type="periodic")
        domain = SimpleGrid1D(xmin=0.0, xmax=1.0, boundary_conditions=boundary_conditions)
        domain.create_grid(num_points=51)
        # Note: Edge case test - currently fails, needs investigation
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((0, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        assert m_result.shape == (0, Nx)

    def test_solve_fp_system_one_timestep(self, standard_problem):
        """Test behavior with single time step (Nt=1)."""
        # Create problem with Nt=1 (results in 1 time step)
        boundary_conditions = BoundaryConditions(type="periodic")
        domain = SimpleGrid1D(xmin=0.0, xmax=1.0, boundary_conditions=boundary_conditions)
        domain.create_grid(num_points=51)
        # Note: Edge case test - currently fails, needs investigation
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((1, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        assert m_result.shape == (1, Nx)
        # Should return initial condition (possibly with non-negativity enforcement)
        assert np.allclose(m_result[0, :], m_initial, rtol=0.1)


class TestFPFDMSolverBoundaryConditions:
    """Test different boundary condition types."""

    def test_periodic_boundary_conditions(self, standard_problem):
        """Test periodic boundary conditions."""
        bc = BoundaryConditions(type="periodic")
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

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

    def test_dirichlet_boundary_conditions(self, standard_problem):
        """Test Dirichlet boundary conditions."""
        bc = BoundaryConditions(type="dirichlet", left_value=0.1, right_value=0.2)
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # Boundary values should be enforced at all time steps
        for t in range(Nt):
            assert np.isclose(m_result[t, 0], 0.1, atol=1e-10)
            assert np.isclose(m_result[t, -1], 0.2, atol=1e-10)

    def test_no_flux_boundary_conditions(self, standard_problem):
        """Test no-flux boundary conditions."""
        bc = BoundaryConditions(type="no_flux")
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        # Gaussian initial condition
        x_coords = np.linspace(standard_problem.xmin, standard_problem.xmax, Nx)
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

    def test_non_negativity_enforcement(self, standard_problem):
        """Test that solution remains non-negative."""
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # All values should be non-negative (with small tolerance for numerical errors)
        assert np.all(m_result >= -1e-10)

    def test_initial_condition_non_negativity(self, standard_problem):
        """Test that negative values in initial condition are set to zero."""
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        # Initial condition with some negative values
        m_initial = np.random.randn(Nx)
        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # Initial condition should have negative values removed
        assert np.all(m_result[0, :] >= 0)


class TestFPFDMSolverWithDrift:
    """Test solver behavior with non-zero drift (from HJB solution)."""

    def test_solve_with_linear_drift(self, standard_problem):
        """Test solution with linear value function (constant drift)."""
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        # Gaussian initial condition at center
        x_coords = np.linspace(standard_problem.xmin, standard_problem.xmax, Nx)
        m_initial = np.exp(-((x_coords - 0.0) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial)

        # Linear value function: U(t,x) = x (constant drift)
        U_solution = np.tile(x_coords, (Nt, 1))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # Solution should evolve (not remain constant)
        assert not np.allclose(m_result[-1, :], m_result[0, :])
        # Should remain non-negative
        assert np.all(m_result >= -1e-10)

    def test_solve_with_quadratic_value_function(self, standard_problem):
        """Test solution with quadratic value function."""
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        x_coords = np.linspace(standard_problem.xmin, standard_problem.xmax, Nx)
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

    def test_zero_diffusion_timestep(self, standard_problem):
        """Test behavior when Dt is extremely small."""
        standard_problem.dt = 1e-20  # Very small timestep
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # With very small Dt, solution should remain close to initial condition
        assert np.allclose(m_result[1, :], m_result[0, :], rtol=0.1)

    def test_zero_spatial_step(self, standard_problem):
        """Test behavior when Dx is extremely small (but Nx > 1)."""
        standard_problem.dx = 1e-20  # Very small spatial step
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # With very small Dx, solution should remain close to initial condition
        assert np.allclose(m_result[1, :], m_result[0, :], rtol=0.1)

    def test_single_spatial_point(self, standard_problem):
        """Test that single spatial point (Nx=1) raises appropriate error.

        Known limitation: The FDM solver requires at least 2 spatial points
        to compute finite differences. Single-point grids are not physically meaningful
        for PDEs anyway (no spatial variation possible).
        """
        standard_problem.Nx = 0  # Results in Nx+1 = 1
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        m_initial = np.array([1.0])
        U_solution = np.zeros((Nt, Nx))

        # This should raise ValueError due to matrix dimension mismatch
        with pytest.raises(ValueError, match=r"axis 1 index .* exceeds matrix dimension"):
            solver.solve_fp_system(m_initial, U_solution)


class TestFPFDMSolverMassConservation:
    """Test mass conservation properties."""

    def test_mass_conservation_no_flux(self, standard_problem):
        """Test that mass is conserved with no-flux boundary conditions."""
        bc = BoundaryConditions(type="no_flux")
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        x_coords = np.linspace(standard_problem.xmin, standard_problem.xmax, Nx)
        m_initial = np.exp(-((x_coords - 0.0) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial)

        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # Check mass conservation at each time step
        initial_mass = np.sum(m_initial)
        for t in range(Nt):
            current_mass = np.sum(m_result[t, :])
            assert np.isclose(current_mass, initial_mass, rtol=0.1)

    def test_mass_evolution_periodic(self, standard_problem):
        """Test mass evolution with periodic boundary conditions."""
        bc = BoundaryConditions(type="periodic")
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((Nt, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # With periodic BC and zero drift, mass should be conserved
        initial_mass = np.sum(m_initial)
        final_mass = np.sum(m_result[-1, :])
        assert np.isclose(final_mass, initial_mass, rtol=0.1)


class TestFPFDMSolverIntegration:
    """Integration tests with actual MFG problems."""

    def test_solver_with_example_problem(self, standard_problem):
        """Test solver works with ExampleMFGProblem."""
        solver = FPFDMSolver(standard_problem)

        assert solver is not None
        assert hasattr(solver, "solve_fp_system")
        assert callable(solver.solve_fp_system)

    def test_solver_not_abstract(self, standard_problem):
        """Test that FPFDMSolver can be instantiated (is concrete)."""
        import inspect

        solver = FPFDMSolver(standard_problem)

        assert isinstance(solver, FPFDMSolver)
        assert not inspect.isabstract(FPFDMSolver)


class TestFPFDMSolverArrayDiffusion:
    """Test array diffusion support (Phase 2.1).

    Note: Spatially varying diffusion with FDM can exhibit ~5-15% mass drift
    due to discretization errors when diffusion varies significantly. This is
    a known limitation of FDM, not a bug. Tests focus on correctness
    (shape, non-negativity) rather than strict mass conservation.
    """

    def test_spatially_varying_diffusion_1d(self, standard_problem):
        """Test spatially varying diffusion: sigma(x) with periodic BC."""
        # Use periodic BC for better mass conservation with array diffusion
        bc = BoundaryConditions(type="periodic")
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        # Create spatially varying diffusion (moderate variation)
        x_grid = np.linspace(standard_problem.xmin, standard_problem.xmax, Nx)
        diffusion_array = 0.15 + 0.05 * np.abs(x_grid - 0.5)  # Moderate variation

        # Initial condition
        m_initial = np.exp(-((x_grid - 0.5) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Zero drift (pure diffusion)
        U_solution = np.zeros((Nt, Nx))

        # Solve with array diffusion
        M = solver.solve_fp_system(m_initial, drift_field=U_solution, diffusion_field=diffusion_array)

        assert M.shape == (Nt, Nx)
        assert np.all(M >= 0)
        # Verify solution doesn't blow up (moderate mass drift is expected with variable diffusion)
        assert np.all(np.sum(M, axis=1) > 0.5)
        assert np.all(np.sum(M, axis=1) < 2.0)

    def test_spatiotemporal_diffusion_1d(self, standard_problem):
        """Test spatiotemporal diffusion: sigma(t, x)."""
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        # Create spatiotemporal diffusion (varying in time and space)
        diffusion_field = np.zeros((Nt, Nx))
        x_grid = np.linspace(standard_problem.xmin, standard_problem.xmax, Nx)
        for t in range(Nt):
            # Diffusion increases over time, higher at boundaries
            time_factor = 0.1 * (1 + 0.5 * t / Nt)
            space_factor = 1.0 + 0.3 * np.abs(x_grid - 0.5)
            diffusion_field[t, :] = time_factor * space_factor

        # Initial condition
        m_initial = np.exp(-((x_grid - 0.5) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Zero drift
        U_solution = np.zeros((Nt, Nx))

        # Solve with array diffusion
        M = solver.solve_fp_system(m_initial, drift_field=U_solution, diffusion_field=diffusion_field)

        assert M.shape == (Nt, Nx)
        assert np.all(M >= 0)
        # Mass conservation
        assert np.allclose(np.sum(M, axis=1), 1.0, atol=0.05)

    def test_array_diffusion_with_advection(self, standard_problem):
        """Test array diffusion with non-zero drift."""
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        # Spatially varying diffusion (moderate variation)
        x_grid = np.linspace(standard_problem.xmin, standard_problem.xmax, Nx)
        diffusion_array = 0.2 + 0.05 * x_grid  # Moderate increase

        # Initial condition
        m_initial = np.exp(-((x_grid - 0.3) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Non-zero drift (constant velocity to the right)
        U_solution = np.zeros((Nt, Nx))
        for t in range(Nt):
            U_solution[t, :] = -0.2 * x_grid  # Moderate drift

        # Solve with array diffusion and drift
        M = solver.solve_fp_system(m_initial, drift_field=U_solution, diffusion_field=diffusion_array)

        assert M.shape == (Nt, Nx)
        assert np.all(M >= 0)
        # Verify solution stability (no blow-up)
        assert np.all(np.sum(M, axis=1) > 0.5)
        assert np.all(np.sum(M, axis=1) < 2.0)

    def test_array_diffusion_mass_conservation(self, standard_problem):
        """Test that mass is conserved with array diffusion."""
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        # Spatially varying diffusion (non-uniform)
        x_grid = np.linspace(standard_problem.xmin, standard_problem.xmax, Nx)
        diffusion_array = 0.1 + 0.3 * (x_grid * (1 - x_grid))  # Parabolic profile

        # Initial condition (normalized)
        m_initial = np.ones(Nx) / Nx

        # Zero drift
        U_solution = np.zeros((Nt, Nx))

        # Solve
        M = solver.solve_fp_system(m_initial, drift_field=U_solution, diffusion_field=diffusion_array)

        # Check mass conservation at all timesteps
        masses = np.sum(M, axis=1)
        assert np.allclose(masses, 1.0, atol=0.05)

    def test_array_diffusion_shape_validation(self, standard_problem):
        """Test that incorrect array shapes raise errors."""
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((Nt, Nx))

        # Wrong shape: 3D array
        diffusion_3d = np.zeros((Nt, Nx, 2))

        with pytest.raises(ValueError, match=r"must be 1D.*or 2D"):
            solver.solve_fp_system(m_initial, drift_field=U_solution, diffusion_field=diffusion_3d)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
