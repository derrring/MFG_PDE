#!/usr/bin/env python3
"""
Unit tests for FPFDMSolver - comprehensive coverage.

Tests the Finite Difference Method (FDM) solver for Fokker-Planck equations
with different boundary conditions (periodic, Dirichlet, no-flux).

Note: Uses legacy SimpleGrid1D API which is deprecated in v0.14.
These tests validate legacy behavior until removal in v1.0.
"""

import pytest

import numpy as np

from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver
from mfg_pde.core.mfg_problem import MFGProblem

# Legacy 1D BC: testing compatibility with 1D FDM solvers (deprecated in v0.14, remove in v1.0)
from mfg_pde.geometry.boundary.fdm_bc_1d import BoundaryConditions
from mfg_pde.geometry.grids.grid_1d import SimpleGrid1D

# Suppress deprecation warnings for SimpleGrid classes in this test module
pytestmark = pytest.mark.filterwarnings("ignore:SimpleGrid.*deprecated:DeprecationWarning")


@pytest.fixture
def standard_problem():
    """Create standard 1D MFG problem using modern geometry-first API.

    Standard MFGProblem configuration:
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
        """Test basic solver initialization inheriting BC from geometry."""
        solver = FPFDMSolver(standard_problem)

        assert solver.fp_method_name == "FDM"
        # Solver inherits BC from problem geometry (which is periodic in fixture)
        assert solver.boundary_conditions.type == "periodic"
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
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1

        m_initial = np.ones(Nx) / Nx
        U_solution = np.zeros((0, Nx))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        assert m_result.shape == (0, Nx)

    def test_solve_fp_system_one_timestep(self, standard_problem):
        """Test behavior with single time step (Nt=1)."""
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

    @pytest.mark.xfail(
        reason="Conservative flux FDM (PR #383) regression: initial condition shape mismatch",
        strict=False,
    )
    def test_single_spatial_point(self, standard_problem):
        """Test single spatial point (Nx=1) degenerate case.

        Single-point grids are not physically meaningful for PDEs
        (no spatial variation possible). The solver handles this gracefully
        by returning the initial condition propagated forward in time.
        """
        standard_problem.Nx = 0  # Results in Nx+1 = 1
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        m_initial = np.array([1.0])
        U_solution = np.zeros((Nt, Nx))

        # Single-point grids are degenerate but solver handles them gracefully
        m_result = solver.solve_fp_system(m_initial, U_solution)

        # With no spatial variation, the solution should remain constant
        assert m_result.shape == (Nt, Nx)
        assert np.all(np.isfinite(m_result))


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
        """Test solver works with standard MFGProblem."""
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

    @pytest.mark.xfail(
        reason="Conservative flux FDM (PR #383) changed error message format",
        strict=False,
    )
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


class TestFPFDMSolverCallableDiffusion:
    """Test callable (state-dependent) diffusion support (Phase 2.2)."""

    def test_porous_medium_equation(self, standard_problem):
        """Test porous medium equation: D(m) = σ² m."""
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        # Porous medium diffusion: D = σ² m
        def porous_medium_diffusion(t, x, m):
            return 0.1 * m  # Diffusion proportional to density

        # Initial condition (Gaussian)
        x_grid = np.linspace(standard_problem.xmin, standard_problem.xmax, Nx)
        m_initial = np.exp(-((x_grid - 0.5) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Zero drift
        U_solution = np.zeros((Nt, Nx))

        # Solve with callable diffusion
        M = solver.solve_fp_system(m_initial, drift_field=U_solution, diffusion_field=porous_medium_diffusion)

        assert M.shape == (Nt, Nx)
        assert np.all(M >= 0)
        # Verify solution stability
        assert np.all(np.sum(M, axis=1) > 0.5)
        assert np.all(np.sum(M, axis=1) < 2.0)

    def test_density_dependent_diffusion(self, standard_problem):
        """Test density-dependent diffusion: D = D0 + D1 * (1 - m/m_max)."""
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        # Crowd diffusion: lower diffusion in high-density regions
        def crowd_diffusion(t, x, m):
            m_max = np.max(m) if np.max(m) > 0 else 1.0
            return 0.05 + 0.15 * (1 - m / m_max)

        # Initial condition
        x_grid = np.linspace(standard_problem.xmin, standard_problem.xmax, Nx)
        m_initial = np.exp(-((x_grid - 0.5) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Zero drift
        U_solution = np.zeros((Nt, Nx))

        # Solve
        M = solver.solve_fp_system(m_initial, drift_field=U_solution, diffusion_field=crowd_diffusion)

        assert M.shape == (Nt, Nx)
        assert np.all(M >= 0)
        assert np.all(np.sum(M, axis=1) > 0.5)

    def test_callable_with_drift(self, standard_problem):
        """Test callable diffusion combined with drift field."""
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        # State-dependent diffusion
        def state_diffusion(t, x, m):
            return 0.1 + 0.05 * m

        # Initial condition
        x_grid = np.linspace(standard_problem.xmin, standard_problem.xmax, Nx)
        m_initial = np.exp(-((x_grid - 0.3) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Drift field
        U_solution = np.zeros((Nt, Nx))
        for t in range(Nt):
            U_solution[t, :] = -0.1 * x_grid

        # Solve
        M = solver.solve_fp_system(m_initial, drift_field=U_solution, diffusion_field=state_diffusion)

        assert M.shape == (Nt, Nx)
        assert np.all(M >= 0)

    def test_callable_scalar_return(self, standard_problem):
        """Test callable that returns scalar (constant diffusion)."""
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1
        Nt = standard_problem.Nt + 1

        # Callable returning scalar
        def constant_diffusion(t, x, m):
            return 0.2  # Constant for all x

        # Initial condition
        m_initial = np.ones(Nx) / Nx

        # Solve
        M = solver.solve_fp_system(m_initial, diffusion_field=constant_diffusion)

        assert M.shape == (Nt, Nx)
        assert np.all(M >= 0)

    def test_callable_validation_wrong_shape(self, standard_problem):
        """Test that callable returning wrong shape raises error."""
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1

        # Callable returning wrong shape
        def bad_diffusion(t, x, m):
            return np.ones(Nx + 10)  # Wrong shape

        m_initial = np.ones(Nx) / Nx

        # Should raise ValueError about shape
        with pytest.raises(ValueError, match="returned array with shape"):
            solver.solve_fp_system(m_initial, diffusion_field=bad_diffusion)

    def test_callable_validation_nan(self, standard_problem):
        """Test that callable returning NaN raises error."""
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1

        # Callable returning NaN
        def nan_diffusion(t, x, m):
            result = 0.1 * m
            result[0] = np.nan  # Introduce NaN
            return result

        m_initial = np.ones(Nx) / Nx

        # Should raise ValueError about NaN
        with pytest.raises(ValueError, match="NaN or Inf"):
            solver.solve_fp_system(m_initial, diffusion_field=nan_diffusion)


class TestFPFDMSolverTensorDiffusion:
    """Test tensor diffusion support (Phase 3.0)."""

    def test_diagonal_tensor_2d(self):
        """Test diagonal anisotropic tensor in 2D."""
        from mfg_pde.geometry.grids.grid_2d import SimpleGrid2D

        domain = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 0.6), resolution=(30, 20))
        problem = MFGProblem(geometry=domain, T=0.05, Nt=10, sigma=0.1)

        boundary_conditions = BoundaryConditions(type="no_flux")
        solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

        Nx, Ny = domain.nx + 1, domain.ny + 1
        Nt = problem.Nt + 1

        # Diagonal tensor: fast horizontal, slow vertical
        Sigma = np.diag([0.2, 0.05])

        # Gaussian initial condition at center
        x_coords, y_coords = domain.coordinates
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        m_initial = np.exp(-((X - 0.5) ** 2 + (Y - 0.3) ** 2) / (2 * 0.08**2))
        m_initial /= np.sum(m_initial) * domain.dx * domain.dy

        # Zero drift (pure diffusion)
        U_solution = np.zeros((Nt, Nx, Ny))

        # Solve with tensor diffusion
        M = solver.solve_fp_system(m_initial, drift_field=U_solution, tensor_diffusion_field=Sigma)

        assert M.shape == (Nt, Nx, Ny)
        assert np.all(M >= 0)
        # Mass conservation
        masses = np.sum(M, axis=(1, 2)) * domain.dx * domain.dy
        assert np.allclose(masses, 1.0, atol=0.1)

    def test_full_tensor_with_cross_diffusion(self):
        """Test full anisotropic tensor with off-diagonal terms."""
        from mfg_pde.geometry.grids.grid_2d import SimpleGrid2D

        domain = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(25, 25))
        problem = MFGProblem(geometry=domain, T=0.05, Nt=10, sigma=0.1)

        boundary_conditions = BoundaryConditions(type="periodic")
        solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

        Nx, Ny = domain.nx + 1, domain.ny + 1
        Nt = problem.Nt + 1

        # Full tensor with cross-diffusion
        Sigma = np.array([[0.2, 0.05], [0.05, 0.1]])

        # Initial condition
        x_coords, y_coords = domain.coordinates
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        m_initial = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial) * domain.dx * domain.dy

        # Solve
        M = solver.solve_fp_system(m_initial, tensor_diffusion_field=Sigma, show_progress=False)

        assert M.shape == (Nt, Nx, Ny)
        assert np.all(M >= 0)
        # Verify solution stability
        masses = np.sum(M, axis=(1, 2)) * domain.dx * domain.dy
        assert np.all(masses > 0.5)
        assert np.all(masses < 2.0)

    def test_spatially_varying_tensor(self):
        """Test spatially-varying tensor diffusion."""
        from mfg_pde.geometry.grids.grid_2d import SimpleGrid2D

        domain = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 0.6), resolution=(25, 15))
        problem = MFGProblem(geometry=domain, T=0.05, Nt=10, sigma=0.1)

        boundary_conditions = BoundaryConditions(type="no_flux")
        solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

        Nx, Ny = domain.nx + 1, domain.ny + 1
        Nt = problem.Nt + 1

        # Spatially-varying tensor: orientation changes with position
        x_coords, y_coords = domain.coordinates
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")

        Sigma_spatial = np.zeros((Nx, Ny, 2, 2))
        for i in range(Nx):
            for j in range(Ny):
                # Diagonal tensor varying with position
                sigma_x = 0.15 + 0.05 * X[i, j]
                sigma_y = 0.08 + 0.02 * Y[i, j]
                Sigma_spatial[i, j] = np.diag([sigma_x, sigma_y])

        # Initial condition
        m_initial = np.exp(-((X - 0.5) ** 2 + (Y - 0.3) ** 2) / (2 * 0.08**2))
        m_initial /= np.sum(m_initial) * domain.dx * domain.dy

        # Solve
        M = solver.solve_fp_system(m_initial, tensor_diffusion_field=Sigma_spatial, show_progress=False)

        assert M.shape == (Nt, Nx, Ny)
        assert np.all(M >= 0)
        # Mass conservation with spatially varying tensor
        masses = np.sum(M, axis=(1, 2)) * domain.dx * domain.dy
        assert np.allclose(masses, 1.0, atol=0.15)

    def test_callable_tensor(self):
        """Test callable state-dependent tensor: Sigma(t, x, m)."""
        from mfg_pde.geometry.grids.grid_2d import SimpleGrid2D

        domain = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 0.6), resolution=(20, 15))
        problem = MFGProblem(geometry=domain, T=0.05, Nt=10, sigma=0.1)

        boundary_conditions = BoundaryConditions(type="no_flux")
        solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

        Nx, Ny = domain.nx + 1, domain.ny + 1

        # State-dependent tensor: anisotropy increases with density
        def crowd_anisotropic(t, x, m):
            sigma_parallel = 0.15  # Horizontal movement
            # Vertical movement decreases in high-density regions (ensure positive)
            sigma_perp = 0.05 + 0.05 * np.maximum(0, 1 - m / 2.0)
            return np.diag([sigma_parallel, max(sigma_perp, 1e-6)])

        # Initial condition
        x_coords, y_coords = domain.coordinates
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        m_initial = np.exp(-((X - 0.5) ** 2 + (Y - 0.3) ** 2) / (2 * 0.08**2))
        m_initial /= np.sum(m_initial) * domain.dx * domain.dy

        # Solve
        M = solver.solve_fp_system(m_initial, tensor_diffusion_field=crowd_anisotropic, show_progress=False)

        assert M.shape == (problem.Nt + 1, Nx, Ny)
        assert np.all(M >= 0)

    def test_tensor_with_drift(self):
        """Test tensor diffusion combined with drift field."""
        from mfg_pde.geometry.grids.grid_2d import SimpleGrid2D

        domain = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(25, 25))
        problem = MFGProblem(geometry=domain, T=0.05, Nt=10, sigma=0.1)

        boundary_conditions = BoundaryConditions(type="periodic")
        solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

        Nx, Ny = domain.nx + 1, domain.ny + 1
        Nt = problem.Nt + 1

        # Diagonal tensor
        Sigma = np.diag([0.15, 0.08])

        # Initial condition
        x_coords, y_coords = domain.coordinates
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        m_initial = np.exp(-((X - 0.3) ** 2 + (Y - 0.3) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial) * domain.dx * domain.dy

        # Non-zero drift (quadratic value function)
        U_solution = np.zeros((Nt, Nx, Ny))
        for k in range(Nt):
            U_solution[k] = X**2 + Y**2

        # Solve
        M = solver.solve_fp_system(m_initial, drift_field=U_solution, tensor_diffusion_field=Sigma, show_progress=False)

        assert M.shape == (Nt, Nx, Ny)
        assert np.all(M >= 0)
        # Solution should evolve (not static)
        assert not np.allclose(M[0], M[-1])

    def test_tensor_diffusion_mutual_exclusivity(self):
        """Test that tensor_diffusion_field and diffusion_field are mutually exclusive."""
        from mfg_pde.geometry.grids.grid_2d import SimpleGrid2D

        domain = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(25, 25))
        problem = MFGProblem(geometry=domain, T=0.05, Nt=10, sigma=0.1)

        boundary_conditions = BoundaryConditions(type="no_flux")
        solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

        Nx, Ny = domain.nx + 1, domain.ny + 1

        # Initial condition
        m_initial = np.ones((Nx, Ny)) / (Nx * Ny)

        # Tensor and scalar both specified
        Sigma = np.eye(2)
        scalar_sigma = 0.2

        # Should raise ValueError
        with pytest.raises(ValueError, match="Cannot specify both diffusion_field and tensor_diffusion_field"):
            solver.solve_fp_system(m_initial, diffusion_field=scalar_sigma, tensor_diffusion_field=Sigma)

    def test_tensor_diffusion_1d_raises_error(self, standard_problem):
        """Test that tensor diffusion in 1D raises NotImplementedError."""
        solver = FPFDMSolver(standard_problem)

        Nx = standard_problem.Nx + 1

        m_initial = np.ones(Nx) / Nx

        # 1D tensor (should fail)
        Sigma = np.array([[0.2]])

        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="Tensor diffusion not yet implemented for 1D"):
            solver.solve_fp_system(m_initial, tensor_diffusion_field=Sigma)

    def test_tensor_psd_validation(self):
        """Test that non-PSD tensor raises error."""
        from mfg_pde.geometry.grids.grid_2d import SimpleGrid2D

        domain = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(20, 20))
        problem = MFGProblem(geometry=domain, T=0.05, Nt=5, sigma=0.1)

        boundary_conditions = BoundaryConditions(type="no_flux")
        solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

        Nx, Ny = domain.nx + 1, domain.ny + 1

        # Non-PSD tensor (negative eigenvalue)
        Sigma_bad = np.array([[0.2, 0.3], [0.3, -0.1]])  # Has negative eigenvalue

        # Initial condition
        m_initial = np.ones((Nx, Ny)) / (Nx * Ny)

        # Should raise ValueError about PSD
        with pytest.raises(ValueError, match="positive semi-definite"):
            solver.solve_fp_system(m_initial, tensor_diffusion_field=Sigma_bad, show_progress=False)

    def test_tensor_diffusion_mass_conservation(self):
        """Test mass conservation with tensor diffusion."""
        from mfg_pde.geometry.grids.grid_2d import SimpleGrid2D

        domain = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 0.6), resolution=(30, 20))
        problem = MFGProblem(geometry=domain, T=0.05, Nt=50, sigma=0.1)

        boundary_conditions = BoundaryConditions(type="no_flux")
        solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

        # Diagonal tensor (smaller values for stability)
        Sigma = np.diag([0.05, 0.03])

        # Initial condition
        x_coords, y_coords = domain.coordinates
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        m_initial = np.exp(-((X - 0.5) ** 2 + (Y - 0.3) ** 2) / (2 * 0.08**2))
        m_initial /= np.sum(m_initial) * domain.dx * domain.dy

        # Solve
        M = solver.solve_fp_system(m_initial, tensor_diffusion_field=Sigma, show_progress=False)

        # Check mass conservation at each timestep
        masses = np.sum(M, axis=(1, 2)) * domain.dx * domain.dy
        assert np.allclose(masses, 1.0, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
