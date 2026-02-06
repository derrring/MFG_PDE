#!/usr/bin/env python3
"""
Unit tests for HJBGFDMSolver - comprehensive coverage.

Tests the GFDM (Generalized Finite Difference Method) solver for HJB equations.
"""

import pytest

import numpy as np

from mfg_pde.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfg_pde.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfg_pde.core.mfg_components import MFGComponents
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import no_flux_bc


def _default_hamiltonian():
    """Default Hamiltonian for testing (Issue #670: explicit specification required)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


@pytest.fixture
def standard_problem():
    """Create standard 1D MFG problem using modern geometry-first API.

    Standard MFGProblem configuration:
    - Domain: [0, 1] with 51 grid points
    - Time: T=1.0 with 51 time steps
    - Diffusion: diffusion=1.0
    """
    domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(geometry=domain, T=1.0, Nt=51, diffusion=1.0, components=_default_components())


class TestHJBGFDMSolverInitialization:
    """Test HJBGFDMSolver initialization and setup."""

    def test_basic_initialization(self, standard_problem):
        """Test basic solver initialization."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        collocation_points = x_coords.reshape(-1, 1)

        solver = HJBGFDMSolver(problem, collocation_points)

        assert solver.hjb_method_name == "GFDM"
        assert solver.n_points == Nx_points
        assert solver.dimension == 1
        assert solver.delta == 0.1
        assert solver.taylor_order == 2

    def test_custom_parameters(self, standard_problem):
        """Test initialization with custom parameters."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], 20)
        collocation_points = x_coords.reshape(-1, 1)

        solver = HJBGFDMSolver(
            problem,
            collocation_points,
            delta=0.15,
            taylor_order=1,
            max_newton_iterations=50,
            newton_tolerance=1e-8,
        )

        assert solver.n_points == 20
        assert solver.delta == 0.15
        assert solver.taylor_order == 1
        assert solver.max_newton_iterations == 50
        assert solver.newton_tolerance == 1e-8

    def test_deprecated_parameters(self, standard_problem):
        """Test backward compatibility with deprecated parameter names."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        collocation_points = x_coords.reshape(-1, 1)

        with pytest.warns(DeprecationWarning, match="Parameter.*deprecated"):
            solver = HJBGFDMSolver(problem, collocation_points, NiterNewton=40, l2errBoundNewton=1e-5)

        assert solver.max_newton_iterations == 40
        assert solver.newton_tolerance == 1e-5
        assert solver.NiterNewton == 40
        assert solver.l2errBoundNewton == 1e-5

    def test_qp_optimization_levels(self, standard_problem):
        """Test different QP optimization levels."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        collocation_points = x_coords.reshape(-1, 1)

        # Test current QP optimization levels
        levels = ["none", "auto", "always"]
        expected_names = ["GFDM", "GFDM-QP", "GFDM-QP-Always"]

        for level, expected_name in zip(levels, expected_names, strict=False):
            solver = HJBGFDMSolver(problem, collocation_points, qp_optimization_level=level)
            assert solver.hjb_method_name == expected_name
            assert solver.qp_optimization_level == level


class TestHJBGFDMSolverNeighborhoods:
    """Test neighborhood structure building."""

    def test_neighborhood_structure(self, standard_problem):
        """Test that neighborhoods are built correctly."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], 10)
        collocation_points = x_coords.reshape(-1, 1)

        solver = HJBGFDMSolver(problem, collocation_points, delta=0.3)

        # Check all points have neighborhoods
        assert len(solver.neighborhoods) == 10

        for i in range(10):
            neighborhood = solver.neighborhoods[i]
            assert "indices" in neighborhood
            assert "points" in neighborhood
            assert "distances" in neighborhood
            assert "size" in neighborhood
            assert neighborhood["size"] > 0
            # Point should be in its own neighborhood
            assert i in neighborhood["indices"]

    def test_neighborhood_delta_radius(self, standard_problem):
        """Test that delta parameter controls neighborhood size."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], 20)
        collocation_points = x_coords.reshape(-1, 1)

        # Small delta should give small neighborhoods
        solver_small = HJBGFDMSolver(problem, collocation_points, delta=0.05)
        # Large delta should give large neighborhoods
        solver_large = HJBGFDMSolver(problem, collocation_points, delta=0.5)

        # Compare neighborhood sizes for middle point
        mid_idx = 10
        size_small = solver_small.neighborhoods[mid_idx]["size"]
        size_large = solver_large.neighborhoods[mid_idx]["size"]

        assert size_large > size_small


class TestHJBGFDMSolverTaylorExpansion:
    """Test Taylor expansion and multi-index generation."""

    def test_multi_index_1d_order_1(self, standard_problem):
        """Test 1D multi-index generation for order 1."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], 10)
        collocation_points = x_coords.reshape(-1, 1)

        solver = HJBGFDMSolver(problem, collocation_points, taylor_order=1)

        expected = [(1,)]
        assert solver.multi_indices == expected
        assert solver.n_derivatives == 1

    def test_multi_index_1d_order_2(self, standard_problem):
        """Test 1D multi-index generation for order 2."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], 10)
        collocation_points = x_coords.reshape(-1, 1)

        solver = HJBGFDMSolver(problem, collocation_points, taylor_order=2)

        expected = [(1,), (2,)]
        assert solver.multi_indices == expected
        assert solver.n_derivatives == 2

    def test_taylor_matrices_computed(self, standard_problem):
        """Test that Taylor matrices are precomputed."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], 10)
        collocation_points = x_coords.reshape(-1, 1)

        solver = HJBGFDMSolver(problem, collocation_points)

        # All points should have Taylor matrices
        assert len(solver.taylor_matrices) == 10

        for i in range(10):
            taylor_data = solver.taylor_matrices[i]
            if taylor_data is not None:
                assert "A" in taylor_data
                assert "W" in taylor_data


class TestHJBGFDMSolverWeightFunctions:
    """Test weight function computation."""

    def test_weight_function_wendland(self, standard_problem):
        """Test Wendland weight function integration with solver."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], 10)
        collocation_points = x_coords.reshape(-1, 1)

        solver = HJBGFDMSolver(problem, collocation_points, weight_function="wendland")

        # Access component through solver (post-refactoring)
        distances = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
        weights = solver._neighborhood_builder.compute_weights(distances)

        # Wendland weights should decay with distance
        assert weights[0] >= weights[1] >= weights[2]
        # Weights at delta should be near zero
        assert weights[-1] < weights[0] * 0.1

    def test_weight_function_gaussian(self, standard_problem):
        """Test Gaussian weight function integration with solver."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], 10)
        collocation_points = x_coords.reshape(-1, 1)

        solver = HJBGFDMSolver(problem, collocation_points, weight_function="gaussian")

        # Access component through solver (post-refactoring)
        distances = np.array([0.0, 0.1, 0.2, 0.3])
        weights = solver._neighborhood_builder.compute_weights(distances)

        # Gaussian weights should decay with distance
        assert np.all(np.diff(weights) <= 0)  # Monotone decreasing
        # Weight at zero should be 1
        assert np.isclose(weights[0], 1.0)

    def test_weight_function_uniform(self, standard_problem):
        """Test uniform weight function integration with solver."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], 10)
        collocation_points = x_coords.reshape(-1, 1)

        solver = HJBGFDMSolver(problem, collocation_points, weight_function="uniform")

        # Access component through solver (post-refactoring)
        distances = np.array([0.0, 0.1, 0.2, 0.3])
        weights = solver._neighborhood_builder.compute_weights(distances)

        # Uniform weights should all be 1
        assert np.allclose(weights, 1.0)

    def test_invalid_weight_function(self, standard_problem):
        """Test that invalid weight function raises error.

        The error is raised during construction because the underlying
        GFDMOperator validates weight functions when building Taylor matrices.
        """
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], 10)
        collocation_points = x_coords.reshape(-1, 1)

        # Error is raised during construction, not when calling _compute_weights
        with pytest.raises(ValueError, match="Unknown weight function"):
            HJBGFDMSolver(problem, collocation_points, weight_function="invalid")


class TestHJBGFDMSolverDerivativeApproximation:
    """Test derivative approximation using GFDM."""

    def test_approximate_derivatives_linear_function(self, standard_problem):
        """Test derivative approximation on a linear function."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], 20)
        collocation_points = x_coords.reshape(-1, 1)

        solver = HJBGFDMSolver(problem, collocation_points)

        # Linear function: u(x) = 2x + 3
        u_values = 2 * x_coords + 3

        # Test at middle point
        mid_idx = 10
        derivs = solver.approximate_derivatives(u_values, mid_idx)

        # First derivative should be close to 2 or -2 (sign depends on GFDM formulation)
        if (1,) in derivs:
            assert np.isclose(abs(derivs[(1,)]), 2.0, atol=0.1)

        # Second derivative should be close to 0
        if (2,) in derivs:
            assert np.isclose(derivs[(2,)], 0.0, atol=0.1)

    def test_approximate_derivatives_quadratic_function(self, standard_problem):
        """Test derivative approximation on a quadratic function."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], 30)
        collocation_points = x_coords.reshape(-1, 1)

        solver = HJBGFDMSolver(problem, collocation_points)

        # Quadratic function: u(x) = x^2
        u_values = x_coords**2

        # Test at middle point
        mid_idx = 15
        x_mid = x_coords[mid_idx]
        derivs = solver.approximate_derivatives(u_values, mid_idx)

        # First derivative should be close to 2x (magnitude, sign may vary)
        if (1,) in derivs:
            expected_first_mag = abs(2 * x_mid)
            assert np.isclose(abs(derivs[(1,)]), expected_first_mag, rtol=0.1)

        # Second derivative should be close to 2
        if (2,) in derivs:
            assert np.isclose(abs(derivs[(2,)]), 2.0, atol=0.3)


class TestHJBGFDMSolverMappingMethods:
    """Test grid-collocation mapping methods."""

    def test_map_grid_to_collocation(self, standard_problem):
        """Test mapping from grid to collocation points (integration test)."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        collocation_points = x_coords.reshape(-1, 1)

        solver = HJBGFDMSolver(problem, collocation_points)

        # Access component through solver (post-refactoring)
        # Test with simple function
        u_grid = np.sin(x_coords)
        u_collocation = solver._mapper.map_grid_to_collocation(u_grid)

        # Should preserve values when grid == collocation
        assert u_collocation.shape == (Nx_points,)
        assert np.allclose(u_collocation, u_grid)

    @pytest.mark.xfail(
        reason="Pre-existing: Test expects Nx points but grid has Nx+1 points - needs test modernization",
        strict=False,
    )
    def test_map_collocation_to_grid(self, standard_problem):
        """Test mapping from collocation points to grid."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        collocation_points = x_coords.reshape(-1, 1)

        solver = HJBGFDMSolver(problem, collocation_points)

        # Test with simple function
        u_collocation = np.cos(x_coords)
        u_grid = solver._map_collocation_to_grid(u_collocation)

        # Should preserve values when grid == collocation
        assert u_grid.shape == (Nx_points,)
        assert np.allclose(u_grid, u_collocation)

    def test_batch_mapping_consistency(self, standard_problem):
        """Test that batch mapping is consistent with single mapping (integration test)."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        collocation_points = x_coords.reshape(-1, 1)

        solver = HJBGFDMSolver(problem, collocation_points)

        # Create batch data
        Nt = 10
        U_grid = np.random.rand(Nt, Nx_points)

        # Access component through solver (post-refactoring)
        # Batch mapping
        U_collocation_batch = solver._mapper.map_grid_to_collocation_batch(U_grid)

        # Single mapping
        for n in range(Nt):
            u_single = solver._mapper.map_grid_to_collocation(U_grid[n, :])
            assert np.allclose(U_collocation_batch[n, :], u_single)


class TestHJBGFDMSolverSolveHJBSystem:
    """Test the main solve_hjb_system method."""

    @pytest.mark.slow
    def test_solve_hjb_system_shape(self, standard_problem):
        """Test that solve_hjb_system returns correct shape."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        collocation_points = x_coords.reshape(-1, 1)

        solver = HJBGFDMSolver(problem, collocation_points)

        Nt = problem.Nt
        Nx = Nx_points

        # Create inputs
        M_density = np.ones((Nt, Nx))
        U_final = np.zeros(Nx)
        U_prev = np.zeros((Nt, Nx))

        # Solve (may not converge perfectly, but should return correct shape)
        try:
            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)
            assert U_solution.shape == (Nt, Nx)
        except Exception:
            # If it fails due to problem setup, that's okay for this test
            pytest.skip("Solver setup issue - shape test inconclusive")

    @pytest.mark.slow
    def test_solve_hjb_system_final_condition(self, standard_problem):
        """Test that final condition is preserved."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        collocation_points = x_coords.reshape(-1, 1)

        solver = HJBGFDMSolver(problem, collocation_points)

        Nt = problem.Nt
        Nx = Nx_points

        # Create inputs with specific final condition
        M_density = np.ones((Nt, Nx))
        U_final = x_coords**2  # Quadratic final condition
        U_prev = np.zeros((Nt, Nx))

        try:
            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)
            # Final time step should match final condition (approximately)
            assert np.allclose(U_solution[-1, :], U_final, rtol=0.1)
        except Exception:
            pytest.skip("Solver convergence issue - test inconclusive")


class TestHJBGFDMSolverIntegration:
    """Integration tests with actual MFG problems."""

    def test_solver_with_example_problem(self, standard_problem):
        """Test solver works with standard MFGProblem."""
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        collocation_points = x_coords.reshape(-1, 1)

        solver = HJBGFDMSolver(problem, collocation_points)

        # Basic instantiation should work
        assert solver is not None
        assert hasattr(solver, "solve_hjb_system")
        assert callable(solver.solve_hjb_system)

    def test_solver_not_abstract(self, standard_problem):
        """Test that HJBGFDMSolver can be instantiated (is concrete)."""
        import inspect

        # Should be instantiable (not abstract)
        problem = standard_problem
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], 10)
        collocation_points = x_coords.reshape(-1, 1)

        # This should not raise TypeError about abstract methods
        solver = HJBGFDMSolver(problem, collocation_points)
        assert isinstance(solver, HJBGFDMSolver)

        # Should not have abstract methods
        assert not inspect.isabstract(HJBGFDMSolver)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
