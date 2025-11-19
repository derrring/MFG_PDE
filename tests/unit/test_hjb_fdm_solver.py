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


class TestHJBFDMSolverInitialization:
    """Test HJBFDMSolver initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic solver initialization with default parameters."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = HJBFDMSolver(problem)

        assert solver.hjb_method_name == "FDM"
        assert solver.max_newton_iterations == 30
        assert solver.newton_tolerance == 1e-6
        assert solver.problem is problem

    def test_custom_newton_parameters(self):
        """Test initialization with custom Newton parameters."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = HJBFDMSolver(
            problem,
            max_newton_iterations=50,
            newton_tolerance=1e-8,
        )

        assert solver.max_newton_iterations == 50
        assert solver.newton_tolerance == 1e-8

    def test_deprecated_parameters_niter(self):
        """Test backward compatibility with deprecated NiterNewton parameter."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)

        with pytest.warns(DeprecationWarning, match="Parameter.*NiterNewton.*deprecated"):
            solver = HJBFDMSolver(problem, NiterNewton=40)

        assert solver.max_newton_iterations == 40

    def test_deprecated_parameters_tolerance(self):
        """Test backward compatibility with deprecated l2errBoundNewton parameter."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)

        with pytest.warns(DeprecationWarning, match="Parameter.*l2errBoundNewton.*deprecated"):
            solver = HJBFDMSolver(problem, l2errBoundNewton=1e-5)

        assert solver.newton_tolerance == 1e-5

    def test_both_deprecated_parameters(self):
        """Test backward compatibility with both deprecated parameters."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)

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
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)

        with pytest.warns(DeprecationWarning, match="Parameter.*NiterNewton.*deprecated"):
            solver = HJBFDMSolver(
                problem,
                max_newton_iterations=60,
                NiterNewton=25,  # Should be ignored
            )

        assert solver.max_newton_iterations == 60

    def test_invalid_max_iterations(self):
        """Test that invalid max_newton_iterations raises error."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)

        with pytest.raises(ValueError, match="max_newton_iterations must be >= 1"):
            HJBFDMSolver(problem, max_newton_iterations=0)

        with pytest.raises(ValueError, match="max_newton_iterations must be >= 1"):
            HJBFDMSolver(problem, max_newton_iterations=-5)

    def test_invalid_tolerance(self):
        """Test that invalid newton_tolerance raises error."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)

        with pytest.raises(ValueError, match="newton_tolerance must be > 0"):
            HJBFDMSolver(problem, newton_tolerance=0.0)

        with pytest.raises(ValueError, match="newton_tolerance must be > 0"):
            HJBFDMSolver(problem, newton_tolerance=-1e-6)

    def test_newton_config_storage(self):
        """Test that Newton configuration is properly stored."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
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
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = HJBFDMSolver(problem)

        assert hasattr(solver, "backend")
        assert solver.backend is not None

    def test_custom_backend(self):
        """Test initialization with custom backend."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = HJBFDMSolver(problem, backend="numpy")

        assert solver.backend is not None
        # Backend should be NumPy backend
        assert hasattr(solver.backend, "array")


class TestHJBFDMSolverSolveHJBSystem:
    """Test the main solve_hjb_system method."""

    def test_solve_hjb_system_shape(self):
        """Test that solve_hjb_system returns correct shape."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)
        solver = HJBFDMSolver(problem)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        # Create inputs
        M_density = np.ones((Nt, Nx))
        U_final = np.zeros(Nx)
        U_prev = np.zeros((Nt, Nx))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert U_solution.shape == (Nt, Nx)
        assert np.all(np.isfinite(U_solution))

    def test_solve_hjb_system_final_condition(self):
        """Test that final condition is preserved."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)
        solver = HJBFDMSolver(problem)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        # Create inputs with specific final condition
        M_density = np.ones((Nt, Nx))
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        U_final = 0.5 * (x_coords - problem.xmax) ** 2
        U_prev = np.zeros((Nt, Nx))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Final time step should match final condition
        assert np.allclose(U_solution[-1, :], U_final, rtol=0.1)

    def test_solve_hjb_system_backward_propagation(self):
        """Test that solution propagates backward in time."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)
        solver = HJBFDMSolver(problem)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        # Create inputs
        M_density = np.ones((Nt, Nx))
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        U_final = x_coords**2  # Quadratic final condition
        U_prev = np.zeros((Nt, Nx))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Solution should propagate backward (values at earlier times influenced by final)
        # Check that solution at t=0 is different from zero
        assert not np.allclose(U_solution[0, :], 0.0)

    def test_solve_hjb_system_with_density_variation(self):
        """Test solving with non-uniform density."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)
        solver = HJBFDMSolver(problem)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        # Create Gaussian density
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        m_profile = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        M_density = np.tile(m_profile, (Nt, 1))

        U_final = np.zeros(Nx)
        U_prev = np.zeros((Nt, Nx))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (Nt, Nx)


class TestHJBFDMSolverNumericalProperties:
    """Test numerical properties of the FDM method."""

    def test_solution_finiteness(self):
        """Test that solution remains finite throughout."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=40, T=1.0, Nt=40)
        solver = HJBFDMSolver(problem)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        M_density = np.ones((Nt, Nx)) * 0.5
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        U_final = np.sin(2 * np.pi * x_coords)
        U_prev = np.zeros((Nt, Nx))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # All values should be finite
        assert np.all(np.isfinite(U_solution))

    def test_solution_smoothness(self):
        """Test that solution has reasonable smoothness."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50)
        solver = HJBFDMSolver(problem)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        M_density = np.ones((Nt, Nx))
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((Nt, Nx))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Check spatial smoothness - finite differences shouldn't be too large
        U_diff = np.diff(U_solution, axis=1)
        assert np.max(np.abs(U_diff)) < 100.0


class TestHJBFDMSolverParameterSensitivity:
    """Test solver behavior with different parameter configurations."""

    def test_different_newton_iterations(self):
        """Test solver with different Newton iteration counts."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=20, T=0.5, Nt=20)

        for n_iter in [10, 20, 30, 50]:
            solver = HJBFDMSolver(problem, max_newton_iterations=n_iter)

            Nt = problem.Nt + 1
            Nx = problem.Nx + 1

            M_density = np.ones((Nt, Nx))
            U_final = np.zeros(Nx)
            U_prev = np.zeros((Nt, Nx))

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

            assert np.all(np.isfinite(U_solution))

    def test_different_tolerances(self):
        """Test solver with different Newton tolerances."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=20, T=0.5, Nt=20)

        for tol in [1e-4, 1e-6, 1e-8]:
            solver = HJBFDMSolver(problem, newton_tolerance=tol)

            Nt = problem.Nt + 1
            Nx = problem.Nx + 1

            M_density = np.ones((Nt, Nx))
            U_final = np.zeros(Nx)
            U_prev = np.zeros((Nt, Nx))

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

            assert np.all(np.isfinite(U_solution))


class TestHJBFDMSolverIntegration:
    """Integration tests with actual MFG problems."""

    def test_solver_with_uniform_density(self):
        """Test solver with uniform density distribution."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)
        solver = HJBFDMSolver(problem)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        # Uniform density
        M_density = np.ones((Nt, Nx)) / Nx

        # Simple final condition
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        U_final = (x_coords - 0.5) ** 2

        U_prev = np.zeros((Nt, Nx))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (Nt, Nx)

    def test_solver_with_gaussian_density(self):
        """Test solver with Gaussian density distribution."""
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)
        solver = HJBFDMSolver(problem)

        Nt = problem.Nt + 1
        Nx = problem.Nx + 1

        # Gaussian density
        x_coords = np.linspace(problem.xmin, problem.xmax, Nx)
        m_profile = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        m_profile = m_profile / np.sum(m_profile)
        M_density = np.tile(m_profile, (Nt, 1))

        U_final = np.zeros(Nx)
        U_prev = np.zeros((Nt, Nx))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (Nt, Nx)


class TestHJBFDMSolverNotAbstract:
    """Test that HJBFDMSolver is concrete (not abstract)."""

    def test_solver_not_abstract(self):
        """Test that HJBFDMSolver can be instantiated."""
        import inspect

        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)

        # Should not raise TypeError about abstract methods
        solver = HJBFDMSolver(problem)
        assert isinstance(solver, HJBFDMSolver)

        # Should not have abstract methods
        assert not inspect.isabstract(HJBFDMSolver)


class TestHJBFDMSolverDiagonalTensor:
    """Test HJBFDMSolver with diagonal tensor diffusion (Phase 3.1)."""

    def test_diagonal_tensor_2d(self):
        """Test HJB solver with constant diagonal tensor in 2D."""
        from mfg_pde.geometry.grids.grid_2d import SimpleGrid2D

        domain = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 0.6), resolution=(15, 10))
        problem = MFGProblem(geometry=domain, T=0.05, Nt=5, sigma=0.1)

        solver = HJBFDMSolver(problem, solver_type="newton")

        # Get grid shape
        Nx, Ny = domain.get_grid_shape()
        Nt = problem.Nt + 1

        # Create dummy density and initial conditions
        M_density = np.ones((Nt, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt, Nx, Ny))

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
        assert U_solution.shape == (Nt, Nx, Ny)
        assert not np.any(np.isnan(U_solution))
        assert not np.any(np.isinf(U_solution))

    def test_non_diagonal_tensor_warning(self):
        """Test that HJB solver warns for non-diagonal tensor."""
        from mfg_pde.geometry.grids.grid_2d import SimpleGrid2D

        domain = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 0.6), resolution=(15, 10))
        problem = MFGProblem(geometry=domain, T=0.05, Nt=5, sigma=0.1)

        solver = HJBFDMSolver(problem, solver_type="newton")

        # Get grid shape
        Nx, Ny = domain.get_grid_shape()
        Nt = problem.Nt + 1

        # Create dummy density and initial conditions
        M_density = np.ones((Nt, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt, Nx, Ny))

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
        assert U_solution.shape == (Nt, Nx, Ny)
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
        from mfg_pde.geometry.grids.grid_2d import SimpleGrid2D

        domain = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 0.6), resolution=(15, 10))
        problem = MFGProblem(geometry=domain, T=0.05, Nt=3, sigma=0.1)

        solver = HJBFDMSolver(problem, solver_type="newton")

        # Get grid shape
        Nx, Ny = domain.get_grid_shape()
        Nt = problem.Nt + 1

        # Create dummy density and initial conditions
        M_density = np.ones((Nt, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt, Nx, Ny))

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
        assert U_solution.shape == (Nt, Nx, Ny)
        assert not np.any(np.isnan(U_solution))

    def test_diagonal_tensor_callable(self):
        """Test HJB solver with callable diagonal tensor Σ(t, x, m)."""
        from mfg_pde.geometry.grids.grid_2d import SimpleGrid2D

        domain = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 0.6), resolution=(10, 8))
        problem = MFGProblem(geometry=domain, T=0.05, Nt=3, sigma=0.1)

        solver = HJBFDMSolver(problem, solver_type="newton")

        # Get grid shape
        Nx, Ny = domain.get_grid_shape()
        Nt = problem.Nt + 1

        # Create dummy density and initial conditions
        M_density = np.ones((Nt, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt, Nx, Ny))

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
        assert U_solution.shape == (Nt, Nx, Ny)
        assert not np.any(np.isnan(U_solution))

    def test_diagonal_tensor_mutual_exclusivity(self):
        """Test that tensor_diffusion_field and diffusion_field are mutually exclusive."""
        from mfg_pde.geometry.grids.grid_2d import SimpleGrid2D

        domain = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 0.6), resolution=(10, 8))
        problem = MFGProblem(geometry=domain, T=0.05, Nt=3, sigma=0.1)

        solver = HJBFDMSolver(problem, solver_type="newton")

        # Get grid shape
        Nx, Ny = domain.get_grid_shape()
        Nt = problem.Nt + 1

        # Create dummy density and initial conditions
        M_density = np.ones((Nt, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt, Nx, Ny))

        # Both tensor and scalar diffusion
        Sigma = np.diag([0.15, 0.05])
        sigma_scalar = 0.2

        # Should raise ValueError
        with pytest.raises(ValueError, match="Cannot specify both"):
            solver.solve_hjb_system(
                M_density, U_final, U_prev, diffusion_field=sigma_scalar, tensor_diffusion_field=Sigma
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
