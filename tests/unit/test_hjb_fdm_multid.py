#!/usr/bin/env python3
"""
Unit tests for nD HJB FDM solver (hjb_fdm_multid.py).

Tests the dimension-agnostic FDM solver using dimensional splitting for 2D, 3D,
and higher-dimensional problems.
"""

import pytest

import numpy as np

from mfg_pde import MFGComponents
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem


class SimpleGridMFGProblem(GridBasedMFGProblem):
    """
    Simple concrete implementation of GridBasedMFGProblem for testing.

    Implements minimal isotropic MFG problem with H = (1/2)|p|^2.
    """

    def __init__(
        self,
        domain_bounds: tuple,
        grid_resolution: int | tuple,
        time_domain: tuple[float, int] = (1.0, 100),
        diffusion_coeff: float = 0.1,
    ):
        super().__init__(domain_bounds, grid_resolution, time_domain, diffusion_coeff)

    def initial_density(self, x: np.ndarray) -> np.ndarray:
        """Uniform initial density."""
        return np.ones(x.shape[0])

    def terminal_cost(self, x: np.ndarray) -> np.ndarray:
        """Zero terminal cost."""
        return np.zeros(x.shape[0])

    def running_cost(self, x: np.ndarray, t: float) -> np.ndarray:
        """Zero running cost."""
        return np.zeros(x.shape[0])

    def hamiltonian(self, x: np.ndarray, m: np.ndarray, p: np.ndarray, t: float) -> np.ndarray:
        """Simple isotropic Hamiltonian: H = (1/2)|p|^2."""
        # p is gradient, compute |p|^2
        p_squared = np.sum(p**2, axis=1) if p.ndim > 1 else p**2
        return 0.5 * p_squared

    def setup_components(self) -> MFGComponents:
        """Setup minimal MFG components for testing."""

        def hamiltonian_func(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, derivs=None, **kwargs):
            """Simple isotropic Hamiltonian: H = (1/2)|p|^2."""
            if derivs is not None:
                # Use tuple-indexed derivatives (new notation)
                grad_u = []
                ndim = problem.geometry.grid.dimension
                for d in range(ndim):
                    idx_tuple = tuple([0] * d + [1] + [0] * (ndim - d - 1))
                    grad_u.append(derivs.get(idx_tuple, 0.0))

                # |∇u|^2
                p_squared = sum(g**2 for g in grad_u)
                return 0.5 * p_squared
            else:
                # Fallback: use p_values (legacy notation)
                p_forward = p_values.get("forward", 0.0)
                p_backward = p_values.get("backward", 0.0)
                p_magnitude = abs(p_forward - p_backward)
                return 0.5 * p_magnitude**2

        def hamiltonian_dm(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, **kwargs):
            """Derivative of Hamiltonian with respect to density (zero for isotropic case)."""
            return 0.0

        def initial_density_func(x_position):
            """Uniform initial density."""
            return 1.0

        def terminal_cost_func(x_position):
            """Zero terminal cost."""
            return 0.0

        def running_cost_func(x_position, m_at_x, t, problem):
            """Zero running cost."""
            return 0.0

        return MFGComponents(
            hamiltonian_func=hamiltonian_func,
            hamiltonian_dm_func=hamiltonian_dm,
            initial_density_func=initial_density_func,
            final_value_func=terminal_cost_func,
            # Note: running_cost is not a standard MFGComponents parameter
        )


class TestHJBFDMDimensionDetection:
    """Test dimension detection for routing to appropriate solver."""

    def test_detect_2d_problem(self):
        """Test that 2D GridBasedMFGProblem is detected correctly."""
        problem = SimpleGridMFGProblem(
            domain_bounds=(0, 1, 0, 1),
            grid_resolution=10,
            time_domain=(0.5, 10),
            diffusion_coeff=0.1,
        )
        solver = HJBFDMSolver(problem)

        assert solver.dimension == 2

    def test_detect_3d_problem(self):
        """Test that 3D GridBasedMFGProblem is detected correctly."""
        problem = SimpleGridMFGProblem(
            domain_bounds=(0, 1, 0, 1, 0, 1),
            grid_resolution=8,
            time_domain=(0.5, 10),
            diffusion_coeff=0.1,
        )
        solver = HJBFDMSolver(problem)

        assert solver.dimension == 3

    def test_detect_4d_problem(self):
        """Test that 4D GridBasedMFGProblem is detected correctly."""
        problem = SimpleGridMFGProblem(
            domain_bounds=(0, 1, 0, 1, 0, 1, 0, 1),
            grid_resolution=5,
            time_domain=(0.5, 10),
            diffusion_coeff=0.1,
        )
        solver = HJBFDMSolver(problem)

        assert solver.dimension == 4


class TestHJBFDM2DBasicFunctionality:
    """Test basic functionality of 2D HJB FDM solver."""

    def test_2d_solve_shape(self):
        """Test that 2D solve returns correct shape."""
        problem = SimpleGridMFGProblem(
            domain_bounds=(0, 1, 0, 1),
            grid_resolution=10,
            time_domain=(0.5, 10),
            diffusion_coeff=0.1,
        )
        solver = HJBFDMSolver(problem, max_newton_iterations=20, newton_tolerance=1e-5)

        Nt = problem.Nt + 1
        Nx = Ny = problem.geometry.grid.num_points[0] - 1

        # Create inputs
        M_density = np.ones((Nt, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt, Nx, Ny))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Check shape
        assert U_solution.shape == (Nt, Nx, Ny)
        assert np.all(np.isfinite(U_solution))

    def test_2d_final_condition_preserved(self):
        """Test that final condition is preserved in 2D."""
        problem = SimpleGridMFGProblem(
            domain_bounds=(0, 1, 0, 1),
            grid_resolution=10,
            time_domain=(0.5, 10),
            diffusion_coeff=0.1,
        )
        solver = HJBFDMSolver(problem, max_newton_iterations=20, newton_tolerance=1e-5)

        Nt = problem.Nt + 1
        Nx = Ny = problem.geometry.grid.num_points[0] - 1

        # Create specific final condition (quadratic bowl)
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        U_final = 0.5 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2)

        M_density = np.ones((Nt, Nx, Ny)) * 0.5
        U_prev = np.zeros((Nt, Nx, Ny))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Final time step should match final condition
        assert np.allclose(U_solution[-1, :, :], U_final, rtol=0.1)

    def test_2d_backward_propagation(self):
        """Test that solution propagates backward in time for 2D."""
        problem = SimpleGridMFGProblem(
            domain_bounds=(0, 1, 0, 1),
            grid_resolution=10,
            time_domain=(0.5, 10),
            diffusion_coeff=0.1,
        )
        solver = HJBFDMSolver(problem, max_newton_iterations=20, newton_tolerance=1e-5)

        Nt = problem.Nt + 1
        Nx = Ny = problem.geometry.grid.num_points[0] - 1

        # Create Gaussian final condition
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        U_final = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.1**2))

        M_density = np.ones((Nt, Nx, Ny)) * 0.5
        U_prev = np.zeros((Nt, Nx, Ny))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Solution should propagate backward (t=0 should be influenced by final condition)
        assert not np.allclose(U_solution[0, :, :], 0.0)
        assert np.all(np.isfinite(U_solution))

    def test_2d_with_non_uniform_density(self):
        """Test 2D solver with non-uniform density."""
        problem = SimpleGridMFGProblem(
            domain_bounds=(0, 1, 0, 1),
            grid_resolution=10,
            time_domain=(0.5, 10),
            diffusion_coeff=0.1,
        )
        solver = HJBFDMSolver(problem, max_newton_iterations=20, newton_tolerance=1e-5)

        Nt = problem.Nt + 1
        Nx = Ny = problem.geometry.grid.num_points[0] - 1

        # Create Gaussian density
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        m_profile = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.1**2))
        M_density = np.tile(m_profile[np.newaxis, :, :], (Nt, 1, 1))

        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt, Nx, Ny))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (Nt, Nx, Ny)


class TestHJBFDM3DBasicFunctionality:
    """Test basic functionality of 3D HJB FDM solver."""

    def test_3d_solve_shape(self):
        """Test that 3D solve returns correct shape."""
        problem = SimpleGridMFGProblem(
            domain_bounds=(0, 1, 0, 1, 0, 1),
            grid_resolution=6,  # Small for fast test
            time_domain=(0.5, 8),
            diffusion_coeff=0.1,
        )
        solver = HJBFDMSolver(problem, max_newton_iterations=15, newton_tolerance=1e-5)

        Nt = problem.Nt + 1
        Nx = Ny = Nz = problem.geometry.grid.num_points[0] - 1

        # Create inputs
        M_density = np.ones((Nt, Nx, Ny, Nz)) * 0.5
        U_final = np.zeros((Nx, Ny, Nz))
        U_prev = np.zeros((Nt, Nx, Ny, Nz))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Check shape
        assert U_solution.shape == (Nt, Nx, Ny, Nz)
        assert np.all(np.isfinite(U_solution))

    def test_3d_final_condition_preserved(self):
        """Test that final condition is preserved in 3D."""
        problem = SimpleGridMFGProblem(
            domain_bounds=(0, 1, 0, 1, 0, 1),
            grid_resolution=6,
            time_domain=(0.5, 8),
            diffusion_coeff=0.1,
        )
        solver = HJBFDMSolver(problem, max_newton_iterations=15, newton_tolerance=1e-5)

        Nt = problem.Nt + 1
        Nx = Ny = Nz = problem.geometry.grid.num_points[0] - 1

        # Create specific final condition (3D quadratic bowl)
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        z = np.linspace(0, 1, Nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        U_final = 0.5 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)

        M_density = np.ones((Nt, Nx, Ny, Nz)) * 0.5
        U_prev = np.zeros((Nt, Nx, Ny, Nz))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Final time step should match final condition
        assert np.allclose(U_solution[-1, :, :, :], U_final, rtol=0.1)

    def test_3d_backward_propagation(self):
        """Test that solution propagates backward in time for 3D."""
        problem = SimpleGridMFGProblem(
            domain_bounds=(0, 1, 0, 1, 0, 1),
            grid_resolution=6,
            time_domain=(0.5, 8),
            diffusion_coeff=0.1,
        )
        solver = HJBFDMSolver(problem, max_newton_iterations=15, newton_tolerance=1e-5)

        Nt = problem.Nt + 1
        Nx = Ny = Nz = problem.geometry.grid.num_points[0] - 1

        # Create 3D Gaussian final condition
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        z = np.linspace(0, 1, Nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        U_final = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) / (2 * 0.1**2))

        M_density = np.ones((Nt, Nx, Ny, Nz)) * 0.5
        U_prev = np.zeros((Nt, Nx, Ny, Nz))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Solution should propagate backward
        assert not np.allclose(U_solution[0, :, :, :], 0.0)
        assert np.all(np.isfinite(U_solution))


class TestHJBFDMNumericalProperties:
    """Test numerical properties specific to dimensional splitting."""

    def test_2d_solution_symmetry(self):
        """Test that symmetric inputs produce symmetric outputs in 2D."""
        problem = SimpleGridMFGProblem(
            domain_bounds=(0, 1, 0, 1),
            grid_resolution=10,
            time_domain=(0.5, 10),
            diffusion_coeff=0.1,
        )
        solver = HJBFDMSolver(problem, max_newton_iterations=20, newton_tolerance=1e-5)

        Nt = problem.Nt + 1
        Nx = Ny = problem.geometry.grid.num_points[0] - 1

        # Create symmetric inputs (radially symmetric)
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        U_final = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.1**2))
        m_profile = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.15**2))
        M_density = np.tile(m_profile[np.newaxis, :, :], (Nt, 1, 1))

        U_prev = np.zeros((Nt, Nx, Ny))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Check approximate radial symmetry at t=0
        # Compare values at (0.3, 0.5) and (0.5, 0.3) - should be similar
        if Nx > 5 and Ny > 5:
            i1, j1 = int(0.3 * Nx), int(0.5 * Ny)
            i2, j2 = int(0.5 * Nx), int(0.3 * Ny)
            # Allow some tolerance due to numerical splitting
            assert np.abs(U_solution[0, i1, j1] - U_solution[0, i2, j2]) < 0.1

    def test_2d_solution_smoothness(self):
        """Test that 2D solution has reasonable spatial smoothness."""
        problem = SimpleGridMFGProblem(
            domain_bounds=(0, 1, 0, 1),
            grid_resolution=12,
            time_domain=(0.5, 10),
            diffusion_coeff=0.1,
        )
        solver = HJBFDMSolver(problem, max_newton_iterations=20, newton_tolerance=1e-5)

        Nt = problem.Nt + 1
        Nx = Ny = problem.geometry.grid.num_points[0] - 1

        # Create smooth final condition
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        U_final = 0.5 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2)

        M_density = np.ones((Nt, Nx, Ny)) * 0.5
        U_prev = np.zeros((Nt, Nx, Ny))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Check spatial smoothness - gradients shouldn't be too large
        U_diff_x = np.diff(U_solution, axis=1)
        U_diff_y = np.diff(U_solution, axis=2)
        assert np.max(np.abs(U_diff_x)) < 100.0
        assert np.max(np.abs(U_diff_y)) < 100.0


class TestHJBFDMNonUniformResolution:
    """Test solver with non-uniform grid resolutions."""

    def test_2d_non_uniform_resolution(self):
        """Test 2D solver with different resolutions in x and y."""
        problem = SimpleGridMFGProblem(
            domain_bounds=(0, 1, 0, 1),
            grid_resolution=(12, 8),  # Different x and y resolution
            time_domain=(0.5, 10),
            diffusion_coeff=0.1,
        )
        solver = HJBFDMSolver(problem, max_newton_iterations=20, newton_tolerance=1e-5)

        Nt = problem.Nt + 1
        Nx = problem.geometry.grid.num_points[0] - 1
        Ny = problem.geometry.grid.num_points[1] - 1

        # Create inputs
        M_density = np.ones((Nt, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt, Nx, Ny))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Check shape matches non-uniform resolution
        assert U_solution.shape == (Nt, Nx, Ny)
        assert np.all(np.isfinite(U_solution))

    def test_3d_non_uniform_resolution(self):
        """Test 3D solver with different resolutions per dimension."""
        problem = SimpleGridMFGProblem(
            domain_bounds=(0, 1, 0, 1, 0, 1),
            grid_resolution=(8, 6, 5),  # Different resolution per dimension
            time_domain=(0.5, 8),
            diffusion_coeff=0.1,
        )
        solver = HJBFDMSolver(problem, max_newton_iterations=15, newton_tolerance=1e-5)

        Nt = problem.Nt + 1
        Nx = problem.geometry.grid.num_points[0] - 1
        Ny = problem.geometry.grid.num_points[1] - 1
        Nz = problem.geometry.grid.num_points[2] - 1

        # Create inputs
        M_density = np.ones((Nt, Nx, Ny, Nz)) * 0.5
        U_final = np.zeros((Nx, Ny, Nz))
        U_prev = np.zeros((Nt, Nx, Ny, Nz))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Check shape matches non-uniform resolution
        assert U_solution.shape == (Nt, Nx, Ny, Nz)
        assert np.all(np.isfinite(U_solution))


class TestHJBFDMParameterSensitivity:
    """Test solver behavior with different parameters for nD problems."""

    def test_2d_different_newton_iterations(self):
        """Test 2D solver with different Newton iteration counts."""
        problem = SimpleGridMFGProblem(
            domain_bounds=(0, 1, 0, 1),
            grid_resolution=8,
            time_domain=(0.5, 8),
            diffusion_coeff=0.1,
        )

        for n_iter in [10, 20, 30]:
            solver = HJBFDMSolver(problem, max_newton_iterations=n_iter, newton_tolerance=1e-5)

            Nt = problem.Nt + 1
            Nx = Ny = problem.geometry.grid.num_points[0] - 1

            M_density = np.ones((Nt, Nx, Ny)) * 0.5
            U_final = np.zeros((Nx, Ny))
            U_prev = np.zeros((Nt, Nx, Ny))

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

            assert np.all(np.isfinite(U_solution))

    def test_2d_different_diffusion_coefficients(self):
        """Test 2D solver with different diffusion coefficients."""
        for sigma in [0.05, 0.1, 0.2]:
            problem = SimpleGridMFGProblem(
                domain_bounds=(0, 1, 0, 1),
                grid_resolution=8,
                time_domain=(0.5, 8),
                diffusion_coeff=sigma,
            )
            solver = HJBFDMSolver(problem, max_newton_iterations=20, newton_tolerance=1e-5)

            Nt = problem.Nt + 1
            Nx = Ny = problem.geometry.grid.num_points[0] - 1

            M_density = np.ones((Nt, Nx, Ny)) * 0.5
            U_final = np.zeros((Nx, Ny))
            U_prev = np.zeros((Nt, Nx, Ny))

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

            assert np.all(np.isfinite(U_solution))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
