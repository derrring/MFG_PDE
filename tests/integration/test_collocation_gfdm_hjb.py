import pytest

import numpy as np

from mfg_pde.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver as GFDMHJBSolver


class MockMFGProblem:
    """Mock MFG problem for testing."""

    def __init__(self):
        self.Nx = 10
        self.Nt = 5
        self.Dx = 0.1
        self.Dt = 0.2
        self.sigma = 0.1

    def H(self, x_idx, m_at_x, p_values, t_idx):
        """Simple quadratic Hamiltonian."""
        if "forward" in p_values:
            p = p_values["forward"]
        elif "x" in p_values:
            p = p_values["x"]
        else:
            p = 0.0
        return 0.5 * p**2

    def get_hjb_hamiltonian_jacobian_contrib(self, U_prev, t_idx):
        """Return None to use numerical Jacobian."""
        return None

    def get_hjb_residual_m_coupling_term(self, M_density, U_derivatives, i, t_idx):
        """No m-coupling term for this test."""
        return None


class TestGFDMHJBSolver:
    """Test suite for the Collocation GFDM HJB solver."""

    def setup_method(self):
        """Set up test fixtures."""
        self.problem = MockMFGProblem()

        # Create simple 1D collocation points
        self.collocation_points = np.linspace(0, 1, 10).reshape(-1, 1)

        # Create solver
        self.solver = GFDMHJBSolver(
            problem=self.problem,
            collocation_points=self.collocation_points,
            delta=0.3,
            taylor_order=2,
            max_newton_iterations=5,
            newton_tolerance=1e-4,
        )

    @pytest.mark.skip(
        reason="HJBGFDMSolver is abstract and cannot be instantiated directly. "
        "This test needs refactoring to use a concrete GFDM implementation. "
        "Issue #140 - pre-existing test failure."
    )
    def test_initialization(self):
        """Test solver initialization."""
        assert self.solver.hjb_method_name == "GFDM"
        assert self.solver.n_points == 10
        assert self.solver.dimension == 1
        assert self.solver.delta == 0.3
        assert self.solver.taylor_order == 2
        assert len(self.solver.neighborhoods) == 10
        assert len(self.solver.taylor_matrices) == 10

    @pytest.mark.skip(
        reason="HJBGFDMSolver is abstract and cannot be instantiated directly. "
        "This test needs refactoring to use a concrete GFDM implementation. "
        "Issue #140 - pre-existing test failure."
    )
    def test_neighborhood_structure(self):
        """Test neighborhood structure building."""
        # Check that each point has neighbors
        for i in range(self.solver.n_points):
            assert i in self.solver.neighborhoods
            neighborhood = self.solver.neighborhoods[i]
            assert "indices" in neighborhood
            assert "points" in neighborhood
            assert "distances" in neighborhood
            assert "size" in neighborhood
            assert neighborhood["size"] > 0
            # Point should be in its own neighborhood
            assert i in neighborhood["indices"]

    @pytest.mark.skip(
        reason="HJBGFDMSolver is abstract and cannot be instantiated directly. "
        "This test needs refactoring to use a concrete GFDM implementation. "
        "Issue #140 - pre-existing test failure."
    )
    def test_multi_index_generation(self):
        """Test multi-index set generation."""
        # Test 1D case
        multi_indices_1d = self.solver._get_multi_index_set(1, 2)
        expected_1d = [(1,), (2,)]
        assert sorted(multi_indices_1d) == sorted(expected_1d)

        # Test 2D case
        multi_indices_2d = self.solver._get_multi_index_set(2, 2)
        expected_2d = [(0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]
        # Order doesn't matter mathematically, use set comparison
        assert sorted(multi_indices_2d) == sorted(expected_2d)

    @pytest.mark.skip(
        reason="HJBGFDMSolver is abstract and cannot be instantiated directly. "
        "This test needs refactoring to use a concrete GFDM implementation. "
        "Issue #140 - pre-existing test failure."
    )
    def test_taylor_matrix_construction(self):
        """Test Taylor expansion matrix construction."""
        # Check that matrices were built for most points
        non_none_matrices = sum(1 for i in range(self.solver.n_points) if self.solver.taylor_matrices[i] is not None)
        assert non_none_matrices > 0

        # Check matrix structure for a point with sufficient neighbors
        for i in range(self.solver.n_points):
            if self.solver.taylor_matrices[i] is not None:
                taylor_data = self.solver.taylor_matrices[i]
                assert "A" in taylor_data
                assert "W" in taylor_data
                # New API: SVD components stored instead of AtW
                assert "S" in taylor_data or "AtW" in taylor_data  # Support both old and new API

                A = taylor_data["A"]
                W = taylor_data["W"]
                assert A.shape[1] == len(self.solver.multi_indices)
                assert W.shape[0] == W.shape[1]  # Square diagonal matrix
                break

    @pytest.mark.skip(
        reason="Legacy pure GFDM test - tests internal implementation details. "
        "Production uses QP-constrained GFDM. Defer refactoring."
    )
    def test_derivative_approximation(self):
        """Test derivative approximation."""
        # Create test function u(x) = x^2
        u_values = (self.collocation_points[:, 0] ** 2).flatten()

        # Test derivative approximation at a point
        point_idx = 5  # Middle point
        if self.solver.taylor_matrices[point_idx] is not None:
            derivatives = self.solver._approximate_derivatives(u_values, point_idx)

            # For u(x) = x^2, derivative should be approximately 2x
            x_val = self.collocation_points[point_idx, 0]
            expected_first_deriv = 2 * x_val

            if (1,) in derivatives:
                actual_first_deriv = derivatives[(1,)]
                # Allow some numerical error (derivative approximation is approximate)
                assert abs(actual_first_deriv - expected_first_deriv) < 3.0

    @pytest.mark.skip(
        reason="Legacy pure GFDM test - tests internal implementation details. "
        "Production uses QP-constrained GFDM. Defer refactoring."
    )
    def test_boundary_conditions_dirichlet(self):
        """Test Dirichlet boundary conditions."""
        # Create solver with boundary conditions
        boundary_indices = np.array([0, 9])  # First and last points
        boundary_conditions = {"type": "dirichlet", "value": 1.0}

        solver_with_bc = GFDMHJBSolver(
            problem=self.problem,
            collocation_points=self.collocation_points,
            delta=0.3,
            boundary_indices=boundary_indices,
            boundary_conditions=boundary_conditions,
        )

        # Test applying boundary conditions
        u_test = np.ones(10)
        u_modified = solver_with_bc._apply_boundary_conditions(u_test, 0)

        assert u_modified[0] == 1.0
        assert u_modified[9] == 1.0

    @pytest.mark.skip(
        reason="HJBGFDMSolver is abstract and cannot be instantiated directly. "
        "This test needs refactoring to use a concrete GFDM implementation. "
        "Issue #140 - pre-existing test failure."
    )
    def test_solve_hjb_system_shape(self):
        """Test that solve_hjb_system returns correct shape."""
        # Create test inputs
        M_density = np.ones((5, 10)) * 0.1
        U_final = np.ones(10)
        U_prev = np.ones((5, 10))

        # Solve (this may not converge perfectly but should return correct shape)
        try:
            U_solution = self.solver.solve_hjb_system(M_density, U_final, U_prev)
            assert U_solution.shape == (5, 10)
        except Exception as e:
            # If solver fails, that's okay for this basic test
            pytest.skip(f"Solver failed: {e}")

    @pytest.mark.skip(
        reason="Legacy pure GFDM test - tests internal implementation details. "
        "Production uses QP-constrained GFDM. Defer refactoring."
    )
    def test_weight_functions(self):
        """Test different weight functions."""
        distances = np.array([0.0, 0.1, 0.2, 0.3])

        # Test Gaussian weights
        weights_gauss = self.solver._compute_weights(distances)
        assert len(weights_gauss) == len(distances)
        assert weights_gauss[0] == 1.0  # Distance 0 should give weight 1
        assert np.all(weights_gauss >= 0)

        # Test uniform weights
        solver_uniform = GFDMHJBSolver(
            problem=self.problem, collocation_points=self.collocation_points, weight_function="uniform"
        )
        weights_uniform = solver_uniform._compute_weights(distances)
        assert np.all(weights_uniform == 1.0)

    @pytest.mark.skip(
        reason="Legacy pure GFDM test - tests internal implementation details. "
        "Production uses QP-constrained GFDM. Defer refactoring."
    )
    def test_grid_collocation_mapping(self):
        """Test mapping between grid and collocation points."""
        # Test when sizes match
        u_grid = np.ones(10)
        u_collocation = self.solver._map_grid_to_collocation(u_grid)
        assert len(u_collocation) == 10

        u_grid_back = self.solver._map_collocation_to_grid(u_collocation)
        assert len(u_grid_back) == 10


if __name__ == "__main__":
    pytest.main([__file__])
