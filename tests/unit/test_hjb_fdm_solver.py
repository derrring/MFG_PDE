#!/usr/bin/env python3
"""
Unit tests for HJBFDMSolver.

Tests the Finite Difference Method (FDM) solver for Hamilton-Jacobi-Bellman equations
using upwind schemes and Newton iteration.
"""

import pytest

import numpy as np

from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import TensorProductGrid


class TestHJBFDMSolverInitialization:
    """Test HJBFDMSolver initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic solver initialization with default parameters."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)
        solver = HJBFDMSolver(problem)

        assert solver.hjb_method_name == "FDM"
        assert solver.max_newton_iterations == 30
        assert solver.newton_tolerance == 1e-6
        assert solver.problem is problem

    def test_custom_newton_parameters(self):
        """Test initialization with custom Newton parameters."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)
        solver = HJBFDMSolver(
            problem,
            max_newton_iterations=50,
            newton_tolerance=1e-8,
        )

        assert solver.max_newton_iterations == 50
        assert solver.newton_tolerance == 1e-8

    def test_deprecated_parameters_niter(self):
        """Test backward compatibility with deprecated NiterNewton parameter."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)

        with pytest.warns(DeprecationWarning, match="Parameter.*NiterNewton.*deprecated"):
            solver = HJBFDMSolver(problem, NiterNewton=40)

        assert solver.max_newton_iterations == 40

    def test_deprecated_parameters_tolerance(self):
        """Test backward compatibility with deprecated l2errBoundNewton parameter."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)

        with pytest.warns(DeprecationWarning, match="Parameter.*l2errBoundNewton.*deprecated"):
            solver = HJBFDMSolver(problem, l2errBoundNewton=1e-5)

        assert solver.newton_tolerance == 1e-5

    def test_both_deprecated_parameters(self):
        """Test backward compatibility with both deprecated parameters."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)

        with pytest.warns(DeprecationWarning, match="Parameter.*deprecated"):
            solver = HJBFDMSolver(
                problem,
                NiterNewton=25,
                l2errBoundNewton=1e-7,
            )

        assert solver.max_newton_iterations == 25
        assert solver.newton_tolerance == 1e-7

    def test_new_overrides_deprecated(self):
        """Test that new parameters override deprecated ones."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)

        with pytest.warns(DeprecationWarning, match="Parameter.*NiterNewton.*deprecated"):
            solver = HJBFDMSolver(
                problem,
                max_newton_iterations=60,
                NiterNewton=25,  # Should be ignored
            )

        assert solver.max_newton_iterations == 60

    def test_invalid_max_iterations(self):
        """Test that invalid max_newton_iterations raises error."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)

        with pytest.raises(ValueError, match="max_newton_iterations must be >= 1"):
            HJBFDMSolver(problem, max_newton_iterations=0)

        with pytest.raises(ValueError, match="max_newton_iterations must be >= 1"):
            HJBFDMSolver(problem, max_newton_iterations=-5)

    def test_invalid_tolerance(self):
        """Test that invalid newton_tolerance raises error."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)

        with pytest.raises(ValueError, match="newton_tolerance must be > 0"):
            HJBFDMSolver(problem, newton_tolerance=0.0)

        with pytest.raises(ValueError, match="newton_tolerance must be > 0"):
            HJBFDMSolver(problem, newton_tolerance=-1e-6)

    def test_newton_config_storage(self):
        """Test that Newton configuration is properly stored."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)
        solver = HJBFDMSolver(
            problem,
            max_newton_iterations=35,
            newton_tolerance=1e-7,
        )

        assert hasattr(solver, "_newton_config")
        assert solver._newton_config["max_iterations"] == 35
        assert solver._newton_config["tolerance"] == 1e-7

    def test_backend_initialization(self):
        """Test that backend is properly initialized."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)
        solver = HJBFDMSolver(problem)

        assert hasattr(solver, "backend")
        assert solver.backend is not None

    def test_custom_backend(self):
        """Test initialization with custom backend."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)
        solver = HJBFDMSolver(problem, backend="numpy")

        assert solver.backend is not None
        # Backend should be NumPy backend
        assert hasattr(solver.backend, "array")

    def test_advection_scheme_default(self):
        """Test default advection_scheme is gradient_upwind."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)
        solver = HJBFDMSolver(problem)

        assert solver.advection_scheme == "gradient_upwind"
        assert solver.use_upwind is True

    def test_advection_scheme_gradient_upwind(self):
        """Test explicit gradient_upwind advection scheme."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)
        solver = HJBFDMSolver(problem, advection_scheme="gradient_upwind")

        assert solver.advection_scheme == "gradient_upwind"
        assert solver.use_upwind is True

    def test_advection_scheme_gradient_centered(self):
        """Test gradient_centered advection scheme."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)
        solver = HJBFDMSolver(problem, advection_scheme="gradient_centered")

        assert solver.advection_scheme == "gradient_centered"
        assert solver.use_upwind is False

    def test_advection_scheme_invalid(self):
        """Test that invalid advection_scheme raises ValueError."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)

        with pytest.raises(ValueError, match="Invalid advection_scheme"):
            HJBFDMSolver(problem, advection_scheme="invalid_scheme")

        with pytest.raises(ValueError, match="Invalid advection_scheme"):
            HJBFDMSolver(problem, advection_scheme="divergence_upwind")  # FP scheme


class TestHJBFDMSolverSolveHJBSystem:
    """Test the main solve_hjb_system method."""

    def test_solve_hjb_system_shape(self):
        """Test that solve_hjb_system returns correct shape."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)
        solver = HJBFDMSolver(problem)

        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]

        # Create inputs
        M_density = np.ones((Nt_points, Nx_points))
        U_final = np.zeros(Nx_points)
        U_prev = np.zeros((Nt_points, Nx_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert U_solution.shape == (Nt_points, Nx_points)
        assert np.all(np.isfinite(U_solution))

    def test_solve_hjb_system_final_condition(self):
        """Test that final condition is preserved."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)
        solver = HJBFDMSolver(problem)

        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        # Create inputs with specific final condition
        M_density = np.ones((Nt_points, Nx_points))
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - bounds[1][0]) ** 2
        U_prev = np.zeros((Nt_points, Nx_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Final time step should match final condition
        assert np.allclose(U_solution[-1, :], U_final, rtol=0.1)

    def test_solve_hjb_system_backward_propagation(self):
        """Test that solution propagates backward in time."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)
        solver = HJBFDMSolver(problem)

        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        # Create inputs
        M_density = np.ones((Nt_points, Nx_points))
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = x_coords**2  # Quadratic final condition
        U_prev = np.zeros((Nt_points, Nx_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Solution should propagate backward (values at earlier times influenced by final)
        # Check that solution at t=0 is different from zero
        assert not np.allclose(U_solution[0, :], 0.0)

    def test_solve_hjb_system_with_density_variation(self):
        """Test solving with non-uniform density."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)
        solver = HJBFDMSolver(problem)

        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        # Create Gaussian density
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_profile = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        M_density = np.tile(m_profile, (Nt_points, 1))

        U_final = np.zeros(Nx_points)
        U_prev = np.zeros((Nt_points, Nx_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (Nt_points, Nx_points)


class TestHJBFDMSolverNumericalProperties:
    """Test numerical properties of the FDM method."""

    def test_solution_finiteness(self):
        """Test that solution remains finite throughout."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=40)
        solver = HJBFDMSolver(problem)

        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        M_density = np.ones((Nt_points, Nx_points)) * 0.5
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = np.sin(2 * np.pi * x_coords)
        U_prev = np.zeros((Nt_points, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # All values should be finite
        assert np.all(np.isfinite(U_solution))

    def test_solution_smoothness(self):
        """Test that solution has reasonable smoothness."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)
        solver = HJBFDMSolver(problem)

        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        M_density = np.ones((Nt_points, Nx_points))
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((Nt_points, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Check spatial smoothness - finite differences shouldn't be too large
        # With dx ≈ 0.02, max gradient differences can be significant for HJB solutions
        U_diff = np.diff(U_solution, axis=1)
        assert np.max(np.abs(U_diff)) < 1000.0


class TestHJBFDMSolverParameterSensitivity:
    """Test solver behavior with different parameter configurations."""

    def test_different_newton_iterations(self):
        """Test solver with different Newton iteration counts."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21])
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20)
        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]

        for n_iter in [10, 20, 30, 50]:
            solver = HJBFDMSolver(problem, max_newton_iterations=n_iter)

            M_density = np.ones((Nt_points, Nx_points))
            U_final = np.zeros(Nx_points)
            U_prev = np.zeros((Nt_points, Nx_points))

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

            assert np.all(np.isfinite(U_solution))

    def test_different_tolerances(self):
        """Test solver with different Newton tolerances."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21])
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20)
        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]

        for tol in [1e-4, 1e-6, 1e-8]:
            solver = HJBFDMSolver(problem, newton_tolerance=tol)

            M_density = np.ones((Nt_points, Nx_points))
            U_final = np.zeros(Nx_points)
            U_prev = np.zeros((Nt_points, Nx_points))

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

            assert np.all(np.isfinite(U_solution))


class TestHJBFDMSolverIntegration:
    """Integration tests with actual MFG problems."""

    def test_solver_with_uniform_density(self):
        """Test solver with uniform density distribution."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)
        solver = HJBFDMSolver(problem)

        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        # Uniform density
        M_density = np.ones((Nt_points, Nx_points)) / Nx_points

        # Simple final condition
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = (x_coords - 0.5) ** 2

        U_prev = np.zeros((Nt_points, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (Nt_points, Nx_points)

    def test_solver_with_gaussian_density(self):
        """Test solver with Gaussian density distribution."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)
        solver = HJBFDMSolver(problem)

        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        # Gaussian density
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_profile = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        m_profile = m_profile / np.sum(m_profile)
        M_density = np.tile(m_profile, (Nt_points, 1))

        U_final = np.zeros(Nx_points)
        U_prev = np.zeros((Nt_points, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (Nt_points, Nx_points)


class TestHJBFDMSolverNotAbstract:
    """Test that HJBFDMSolver is concrete (not abstract)."""

    def test_solver_not_abstract(self):
        """Test that HJBFDMSolver can be instantiated."""
        import inspect

        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)

        # Should not raise TypeError about abstract methods
        solver = HJBFDMSolver(problem)
        assert isinstance(solver, HJBFDMSolver)

        # Should not have abstract methods
        assert not inspect.isabstract(HJBFDMSolver)


class TestHJBFDMSolverDiagonalTensor:
    """Test HJBFDMSolver with diagonal tensor diffusion (Phase 3.1)."""

    def test_diagonal_tensor_2d(self):
        """Test HJB solver with constant diagonal tensor in 2D."""
        domain = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 0.6)], Nx_points=[16, 11])
        problem = MFGProblem(geometry=domain, T=0.05, Nt=5, diffusion=0.1)

        solver = HJBFDMSolver(problem, solver_type="newton")

        # Get grid shape
        Nx, Ny = domain.get_grid_shape()
        Nt_points = problem.Nt_points

        # Create dummy density and initial conditions
        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        # Diagonal tensor: fast horizontal, slow vertical
        Sigma = np.diag([0.15, 0.05])

        # Should not warn for diagonal tensor
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev, tensor_diffusion_field=Sigma)

            # Check that no warning was raised for diagonal tensor
            tensor_warnings = [warning for warning in w if "tensor_diffusion_field" in str(warning.message)]
            assert len(tensor_warnings) == 0, "Should not warn for diagonal tensor"

        # Verify solution shape and validity
        assert U_solution.shape == (Nt_points, Nx, Ny)
        assert not np.any(np.isnan(U_solution))
        assert not np.any(np.isinf(U_solution))

    def test_non_diagonal_tensor_warning(self):
        """Test that HJB solver warns for non-diagonal tensor."""
        domain = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 0.6)], Nx_points=[16, 11])
        problem = MFGProblem(geometry=domain, T=0.05, Nt=5, diffusion=0.1)

        solver = HJBFDMSolver(problem, solver_type="newton")

        # Get grid shape
        Nx, Ny = domain.get_grid_shape()
        Nt_points = problem.Nt_points

        # Create dummy density and initial conditions
        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        # Non-diagonal tensor with off-diagonal elements
        Sigma = np.array([[0.15, 0.02], [0.02, 0.05]])

        # Should warn for non-diagonal tensor
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev, tensor_diffusion_field=Sigma)

            # Check that warning was raised
            tensor_warnings = [warning for warning in w if "non-diagonal" in str(warning.message).lower()]
            assert len(tensor_warnings) > 0, "Should warn for non-diagonal tensor"

        # Solution should still be computed (fallback to diagonal)
        assert U_solution.shape == (Nt_points, Nx, Ny)
        assert not np.any(np.isnan(U_solution))

    def test_diagonal_tensor_is_diagonal_helper(self):
        """Test is_diagonal_tensor helper function."""
        from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import is_diagonal_tensor

        # Test diagonal tensor
        Sigma_diag = np.diag([0.15, 0.05])
        assert is_diagonal_tensor(Sigma_diag), "Should detect diagonal tensor"

        # Test non-diagonal tensor
        Sigma_full = np.array([[0.15, 0.02], [0.02, 0.05]])
        assert not is_diagonal_tensor(Sigma_full), "Should detect non-diagonal tensor"

        # Test nearly-diagonal tensor (within tolerance)
        Sigma_nearly = np.diag([0.15, 0.05])
        Sigma_nearly[0, 1] = 1e-12  # Very small off-diagonal
        Sigma_nearly[1, 0] = 1e-12
        assert is_diagonal_tensor(Sigma_nearly), "Should treat nearly-diagonal as diagonal"

        # Test spatially-varying diagonal tensor
        Sigma_spatial = np.zeros((10, 10, 2, 2))
        for i in range(10):
            for j in range(10):
                Sigma_spatial[i, j] = np.diag([0.15, 0.05])
        assert is_diagonal_tensor(Sigma_spatial), "Should detect spatially-varying diagonal"

    def test_diagonal_tensor_spatially_varying(self):
        """Test HJB solver with spatially-varying diagonal tensor."""
        domain = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 0.6)], Nx_points=[16, 11])
        problem = MFGProblem(geometry=domain, T=0.05, Nt=3, diffusion=0.1)

        solver = HJBFDMSolver(problem, solver_type="newton")

        # Get grid shape
        Nx, Ny = domain.get_grid_shape()
        Nt_points = problem.Nt_points

        # Create dummy density and initial conditions
        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        # Spatially-varying diagonal tensor
        Sigma_spatial = np.zeros((Nx, Ny, 2, 2))
        for i in range(Nx):
            for j in range(Ny):
                # Vary diffusion based on position
                sigma_x = 0.1 + 0.1 * (i / Nx)
                sigma_y = 0.05
                Sigma_spatial[i, j] = np.diag([sigma_x**2, sigma_y**2])

        # Should not warn for spatially-varying diagonal tensor
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev, tensor_diffusion_field=Sigma_spatial)

            # No warnings for spatially-varying diagonal
            tensor_warnings = [warning for warning in w if "tensor_diffusion_field" in str(warning.message)]
            assert len(tensor_warnings) == 0, "Should not warn for spatially-varying diagonal tensor"

        # Verify solution
        assert U_solution.shape == (Nt_points, Nx, Ny)
        assert not np.any(np.isnan(U_solution))

    def test_diagonal_tensor_callable(self):
        """Test HJB solver with callable diagonal tensor Σ(t, x, m)."""
        domain = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 0.6)], Nx_points=[11, 9])
        problem = MFGProblem(geometry=domain, T=0.05, Nt=3, diffusion=0.1)

        solver = HJBFDMSolver(problem, solver_type="newton")

        # Get grid shape
        Nx, Ny = domain.get_grid_shape()
        Nt_points = problem.Nt_points

        # Create dummy density and initial conditions
        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        # Callable diagonal tensor that varies with density
        def tensor_func(t, x, m):
            sigma_x = 0.15
            sigma_y = 0.05 * (1.0 + 0.5 * m)  # Increases with density
            return np.diag([sigma_x**2, sigma_y**2])

        # Should warn for callable (can't check if diagonal without evaluation)
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev, tensor_diffusion_field=tensor_func)

            # Callable tensors should produce a warning
            tensor_warnings = [warning for warning in w if "callable" in str(warning.message).lower()]
            assert len(tensor_warnings) > 0, "Should warn for callable tensor"

        # Verify solution
        assert U_solution.shape == (Nt_points, Nx, Ny)
        assert not np.any(np.isnan(U_solution))

    def test_diagonal_tensor_mutual_exclusivity(self):
        """Test that tensor_diffusion_field and diffusion_field are mutually exclusive."""
        domain = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 0.6)], Nx_points=[11, 9])
        problem = MFGProblem(geometry=domain, T=0.05, Nt=3, diffusion=0.1)

        solver = HJBFDMSolver(problem, solver_type="newton")

        # Get grid shape
        Nx, Ny = domain.get_grid_shape()
        Nt_points = problem.Nt_points

        # Create dummy density and initial conditions
        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        # Both tensor and scalar diffusion
        Sigma = np.diag([0.15, 0.05])
        sigma_scalar = 0.2

        # Should raise ValueError
        with pytest.raises(ValueError, match="Cannot specify both"):
            solver.solve_hjb_system(
                M_density, U_final, U_prev, diffusion_field=sigma_scalar, tensor_diffusion_field=Sigma
            )


class TestHJBFDMSolverGhostValueBC:
    """Test HJB FDM solver with ghost value boundary conditions (Issue #494)."""

    def test_get_ghost_values_nd_dirichlet(self):
        """Test get_ghost_values_nd for Dirichlet BC."""
        from mfg_pde.geometry.boundary import dirichlet_bc, get_ghost_values_nd

        # Create 2D field
        field = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        bc = dirichlet_bc(dimension=2, value=0.0)
        spacing = (0.1, 0.1)

        ghosts = get_ghost_values_nd(field, bc, spacing)

        # Check all boundaries have ghost values
        assert (0, 0) in ghosts  # x_min
        assert (0, 1) in ghosts  # x_max
        assert (1, 0) in ghosts  # y_min
        assert (1, 1) in ghosts  # y_max

        # For Dirichlet g=0, cell-centered: u_ghost = 2*0 - u_interior = -u_interior
        # Left boundary (axis 0): interior values are [1, 2, 3]
        np.testing.assert_allclose(ghosts[(0, 0)], -np.array([1.0, 2.0, 3.0]))
        # Right boundary (axis 0): interior values are [4, 5, 6]
        np.testing.assert_allclose(ghosts[(0, 1)], -np.array([4.0, 5.0, 6.0]))

    def test_get_ghost_values_nd_neumann(self):
        """Test get_ghost_values_nd for Neumann/no-flux BC."""
        from mfg_pde.geometry.boundary import get_ghost_values_nd, no_flux_bc

        # Create 2D field
        field = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        bc = no_flux_bc(dimension=2)
        spacing = (0.1, 0.1)

        ghosts = get_ghost_values_nd(field, bc, spacing)

        # For no-flux (Neumann g=0): use reflection formula (Issue #542 fix)
        # ghost[k] = interior[g-1-k] where g=1
        # Left boundary (axis 0): ghost = field[1, :] (reflection of next interior)
        np.testing.assert_allclose(ghosts[(0, 0)], np.array([4.0, 5.0, 6.0]))
        # Right boundary (axis 0): ghost = field[0, :] (reflection of next interior)
        np.testing.assert_allclose(ghosts[(0, 1)], np.array([1.0, 2.0, 3.0]))

    def test_get_ghost_values_nd_periodic(self):
        """Test get_ghost_values_nd for periodic BC."""
        from mfg_pde.geometry.boundary import get_ghost_values_nd, periodic_bc

        # Create 2D field
        field = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        bc = periodic_bc(dimension=2)
        spacing = (0.1, 0.1)

        ghosts = get_ghost_values_nd(field, bc, spacing)

        # For periodic: ghost = opposite boundary value
        # Left ghost (axis 0) = right interior [4, 5, 6]
        np.testing.assert_allclose(ghosts[(0, 0)], np.array([4.0, 5.0, 6.0]))
        # Right ghost (axis 0) = left interior [1, 2, 3]
        np.testing.assert_allclose(ghosts[(0, 1)], np.array([1.0, 2.0, 3.0]))

    def test_hjb_solver_uses_ghost_values_with_geometry(self):
        """Test that HJB solver uses ghost values when geometry has BCs."""
        from mfg_pde.geometry.boundary import no_flux_bc

        # Create problem with geometry that has BC
        domain = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10])
        # Set boundary conditions on geometry
        domain.boundary_conditions = no_flux_bc(dimension=2)

        problem = MFGProblem(geometry=domain, T=0.1, Nt=5, diffusion=0.1)
        solver = HJBFDMSolver(problem, solver_type="fixed_point", advection_scheme="gradient_upwind")

        # Get grid shape
        Nx, Ny = domain.get_grid_shape()
        Nt_points = problem.Nt_points

        # Create density and initial conditions
        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        # Solve - should use ghost values for gradient computation at boundaries
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Verify solution is valid
        assert U_solution.shape == (Nt_points, Nx, Ny)
        assert np.all(np.isfinite(U_solution))

    def test_hjb_gradient_computation_with_ghost_values(self):
        """Test that _get_ghost_values integrates correctly in gradient computation."""
        from mfg_pde.geometry.boundary import dirichlet_bc

        # Create 2D problem with Dirichlet BC
        domain = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[8, 8])
        domain.boundary_conditions = dirichlet_bc(dimension=2, value=0.0)

        problem = MFGProblem(geometry=domain, T=0.1, Nt=3, diffusion=0.1)
        solver = HJBFDMSolver(problem, solver_type="fixed_point", advection_scheme="gradient_upwind")

        # Test gradient computation directly
        Nx, Ny = domain.get_grid_shape()
        U_test = np.random.rand(Nx, Ny)

        # Call _compute_gradients_nd which should use ghost values
        gradients = solver._compute_gradients_nd(U_test)

        # Check gradient structure
        assert 0 in gradients  # x gradient
        assert 1 in gradients  # y gradient
        assert -1 in gradients  # function value

        # Verify gradient shapes
        assert gradients[0].shape == U_test.shape
        assert gradients[1].shape == U_test.shape

        # Verify gradients are finite
        assert np.all(np.isfinite(gradients[0]))
        assert np.all(np.isfinite(gradients[1]))

    def test_hjb_solver_centered_scheme_with_ghost_values(self):
        """Test centered scheme gradient computation with ghost values."""
        from mfg_pde.geometry.boundary import no_flux_bc

        # Create 2D problem
        domain = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10])
        domain.boundary_conditions = no_flux_bc(dimension=2)

        problem = MFGProblem(geometry=domain, T=0.1, Nt=3, diffusion=0.1)
        # Use centered scheme (non-upwind)
        solver = HJBFDMSolver(problem, solver_type="fixed_point", advection_scheme="gradient_centered")

        Nx, Ny = domain.get_grid_shape()
        Nt_points = problem.Nt_points

        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        # Should work without issues
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert U_solution.shape == (Nt_points, Nx, Ny)
        assert np.all(np.isfinite(U_solution))

    def test_time_varying_dirichlet_bc(self):
        """Test that time-varying Dirichlet BC receives correct time (Issue #496)."""
        from mfg_pde.geometry.boundary import dirichlet_bc, get_ghost_values_nd

        # Track times when BC is called
        call_times = []

        def time_varying_value(t):
            call_times.append(t)
            return np.sin(2 * np.pi * t)

        # Create BC with time-varying value
        bc = dirichlet_bc(dimension=2, value=time_varying_value)

        # Create test field
        field = np.array([[1.0, 2.0], [3.0, 4.0]])
        spacing = (0.1, 0.1)

        # Call at different times
        for t in [0.0, 0.25, 0.5, 0.75]:
            call_times.clear()
            _ = get_ghost_values_nd(field, bc, spacing, time=t)

            # Verify the function was called with correct time
            # (Called once per dimension for Dirichlet)
            assert len(call_times) == 2, f"Expected 2 calls at t={t}, got {len(call_times)}"
            for call_t in call_times:
                assert call_t == t, f"Expected time {t}, got {call_t}"

    def test_hjb_solver_time_varying_bc(self):
        """Test HJB solver with time-varying Dirichlet BC (Issue #496)."""
        from mfg_pde.geometry.boundary import dirichlet_bc

        # Track times when BC is called
        call_times = []

        def time_varying_value(t):
            call_times.append(t)
            return 0.0  # Simple value, we just want to track times

        # Create 2D problem with time-varying BC
        domain = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[8, 8])
        domain.boundary_conditions = dirichlet_bc(dimension=2, value=time_varying_value)

        T = 0.4
        Nt = 4
        problem = MFGProblem(geometry=domain, T=T, Nt=Nt, diffusion=0.1)
        solver = HJBFDMSolver(problem, solver_type="fixed_point", advection_scheme="gradient_upwind")

        Nx, Ny = domain.get_grid_shape()
        n_time_points = Nt + 1

        M_density = np.ones((n_time_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((n_time_points, Nx, Ny))

        # Clear call times and solve
        call_times.clear()
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Verify solution is valid
        assert U_solution.shape == (n_time_points, Nx, Ny)
        assert np.all(np.isfinite(U_solution))

        # Verify that BC was called at different times (not always t=0)
        # HJB solves backward, so times should be n*dt for n in [Nt-1, ..., 0]
        # Check that we have calls at different times
        unique_times = sorted(set(call_times))
        assert len(unique_times) >= 2, (
            f"Expected BC calls at multiple times, got times: {unique_times}. "
            "This suggests time is not being passed correctly to BC."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
