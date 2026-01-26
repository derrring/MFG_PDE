"""
Unit tests for HJB GFDM monotonicity violation detection.

Tests the _check_monotonicity_violation() method for basic M-matrix checking.
"""

import pytest

import numpy as np

from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfg_pde.core.mfg_components import MFGComponents
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import neumann_bc


def _default_components_2d():
    """Provide default components for 2D test problems."""

    def m_initial_2d(x):
        x_arr = np.asarray(x)
        if x_arr.ndim == 1:
            return np.exp(-10 * np.sum((x_arr - 0.5) ** 2))
        return np.exp(-10 * np.sum((x_arr - 0.5) ** 2, axis=-1))

    return MFGComponents(
        m_initial=m_initial_2d,
        u_final=lambda x: 0.0,
    )


class SimpleMFGProblem(MFGProblem):
    """Minimal MFG problem for testing GFDM solver."""

    def __init__(self):
        # Create a 2D geometry
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[21, 21],
            boundary_conditions=neumann_bc(dimension=2),
        )
        super().__init__(
            geometry=geometry,
            T=1.0,
            Nt=10,
            diffusion=0.1,
            components=_default_components_2d(),
        )
        self.domain = [0.0, 1.0, 0.0, 1.0]  # 2D domain (legacy attribute)

    def hamiltonian(self, p, m, x, t):
        """Simple quadratic Hamiltonian."""
        return 0.5 * np.sum(p**2)

    def terminal_condition(self, x):
        """Simple terminal condition."""
        return 0.0


@pytest.mark.skip(reason="Methods moved to MonotonicityMixin - HJBGFDMSolver needs refactoring. See issue #TBD")
def test_check_monotonicity_violation_basic_mode():
    """Test basic mode (strict enforcement) of violation check."""
    problem = SimpleMFGProblem()
    points = np.random.rand(50, 2)

    # Create solver with always level (strict enforcement)
    solver = HJBGFDMSolver(problem=problem, collocation_points=points, qp_optimization_level="always")

    # Build multi-indices for 2D, order 2
    # Should include: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2)
    solver.multi_indices = [
        (0, 0),  # Constant
        (1, 0),  # ∂/∂x
        (0, 1),  # ∂/∂y
        (2, 0),  # ∂²/∂x²
        (1, 1),  # ∂²/∂x∂y
        (0, 2),  # ∂²/∂y²
    ]

    # Test case 1: Valid coefficients (no violation)
    # Laplacian negative, gradient bounded, no higher-order
    D_valid = np.array([1.0, 0.1, 0.1, -0.5, 0.0, -0.5])
    result = solver._check_monotonicity_violation(D_valid, 0, use_adaptive=False)
    assert not result, "Valid coefficients should not trigger violation"

    # Test case 2: Positive Laplacian (violation of criterion 1)
    D_positive_laplacian = np.array([1.0, 0.1, 0.1, 0.5, 0.0, 0.5])
    result = solver._check_monotonicity_violation(D_positive_laplacian, 0, use_adaptive=False)
    assert result, "Positive Laplacian should trigger violation"

    # Test case 3: Large gradient (violation of criterion 2)
    D_large_gradient = np.array([1.0, 10.0, 0.1, -0.1, 0.0, -0.1])
    result = solver._check_monotonicity_violation(D_large_gradient, 0, use_adaptive=False)
    assert result, "Large gradient should trigger violation"


@pytest.mark.skip(reason="Methods moved to MonotonicityMixin - HJBGFDMSolver needs refactoring. See issue #TBD")
def test_no_laplacian_returns_false():
    """Test that missing Laplacian term returns False."""
    problem = SimpleMFGProblem()
    points = np.random.rand(50, 2)

    solver = HJBGFDMSolver(problem=problem, collocation_points=points, qp_optimization_level="always")

    # Multi-indices without Laplacian (order < 2)
    solver.multi_indices = [(0, 0), (1, 0), (0, 1)]
    D_no_laplacian = np.array([1.0, 0.5, 0.5])

    result = solver._check_monotonicity_violation(D_no_laplacian, 0)
    assert not result, "Should return False when Laplacian term missing"


class TestHamiltonianGradientConstraints:
    """Tests for _build_hamiltonian_gradient_constraints method."""

    def test_hamiltonian_constraint_mode_parameter(self):
        """Test that qp_constraint_mode parameter is properly stored."""
        problem = SimpleMFGProblem()
        points = np.random.rand(50, 2)

        # Default should be "indirect"
        solver_default = HJBGFDMSolver(problem=problem, collocation_points=points, qp_optimization_level="auto")
        assert solver_default.qp_constraint_mode == "indirect"

        # Can set to "hamiltonian"
        solver_hamiltonian = HJBGFDMSolver(
            problem=problem,
            collocation_points=points,
            qp_optimization_level="auto",
            qp_constraint_mode="hamiltonian",
        )
        assert solver_hamiltonian.qp_constraint_mode == "hamiltonian"

    @pytest.mark.skip(reason="Methods moved to MonotonicityMixin - HJBGFDMSolver needs refactoring. See issue #TBD")
    def test_build_hamiltonian_constraints_1d(self):
        """Test Hamiltonian gradient constraints in 1D."""
        problem = SimpleMFGProblem()
        # 1D points
        points = np.linspace(0.1, 0.9, 20).reshape(-1, 1)

        solver = HJBGFDMSolver(
            problem=problem,
            collocation_points=points,
            qp_optimization_level="auto",
            qp_constraint_mode="hamiltonian",
            delta=0.15,
        )

        # Multi-indices for 1D, order 2
        solver.multi_indices = [(0,), (1,), (2,)]

        # Get a neighborhood (need to setup neighborhoods first)
        center_idx = 10
        center_point = points[center_idx]

        # Create mock neighbor data
        neighbor_indices = np.array([9, 10, 11])  # Left, center, right
        neighbor_points = points[neighbor_indices]

        # Mock Taylor matrix (not used in constraint building)
        A = np.eye(3)

        constraints = solver._build_hamiltonian_gradient_constraints(
            A,
            neighbor_indices,
            neighbor_points,
            center_point,
            center_idx,
            u_values=None,
            m_density=0.0,
            gamma=0.0,
        )

        # Should have constraints for each non-center neighbor
        assert len(constraints) == 2, f"Expected 2 constraints, got {len(constraints)}"

        # All constraints should be inequality type
        for c in constraints:
            assert c["type"] == "ineq"

    @pytest.mark.skip(reason="Methods moved to MonotonicityMixin - HJBGFDMSolver needs refactoring. See issue #TBD")
    def test_build_hamiltonian_constraints_2d(self):
        """Test Hamiltonian gradient constraints in 2D."""
        problem = SimpleMFGProblem()
        # 2D points
        n = 5
        x = np.linspace(0.1, 0.9, n)
        xx, yy = np.meshgrid(x, x)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        solver = HJBGFDMSolver(
            problem=problem,
            collocation_points=points,
            qp_optimization_level="auto",
            qp_constraint_mode="hamiltonian",
            delta=0.3,
        )

        # Multi-indices for 2D, order 2
        solver.multi_indices = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]

        # Get center point (middle of grid)
        center_idx = 12  # Middle of 5x5 grid
        center_point = points[center_idx]

        # Create mock neighbor data (5 neighbors including center)
        neighbor_indices = np.array([7, 11, 12, 13, 17])  # Cross pattern
        neighbor_points = points[neighbor_indices]

        # Mock Taylor matrix
        A = np.eye(len(solver.multi_indices))

        constraints = solver._build_hamiltonian_gradient_constraints(
            A,
            neighbor_indices,
            neighbor_points,
            center_point,
            center_idx,
            u_values=None,
            m_density=0.0,
            gamma=0.0,
        )

        # Should have constraints for each non-center neighbor (4 constraints)
        assert len(constraints) == 4, f"Expected 4 constraints, got {len(constraints)}"

    @pytest.mark.skip(reason="Methods moved to MonotonicityMixin - HJBGFDMSolver needs refactoring. See issue #TBD")
    def test_hamiltonian_constraints_with_coupling(self):
        """Test that gamma and m_density affect constraints."""
        problem = SimpleMFGProblem()
        problem.gamma = 1.0  # Add coupling
        points = np.linspace(0.1, 0.9, 20).reshape(-1, 1)

        solver = HJBGFDMSolver(
            problem=problem,
            collocation_points=points,
            qp_optimization_level="auto",
            qp_constraint_mode="hamiltonian",
            delta=0.15,
        )

        solver.multi_indices = [(0,), (1,), (2,)]

        center_idx = 10
        center_point = points[center_idx]
        neighbor_indices = np.array([9, 10, 11])
        neighbor_points = points[neighbor_indices]
        A = np.eye(3)

        # With no density, coupling factor is 1 + 2*gamma*m = 1 + 0 = 1
        constraints_no_density = solver._build_hamiltonian_gradient_constraints(
            A,
            neighbor_indices,
            neighbor_points,
            center_point,
            center_idx,
            u_values=None,
            m_density=0.0,
            gamma=1.0,
        )

        # With density m=0.5 and gamma=1, coupling factor is 1 + 2*1*0.5 = 2
        constraints_with_density = solver._build_hamiltonian_gradient_constraints(
            A,
            neighbor_indices,
            neighbor_points,
            center_point,
            center_idx,
            u_values=None,
            m_density=0.5,
            gamma=1.0,
        )

        # Both should have same structure (2 constraints each)
        assert len(constraints_no_density) == len(constraints_with_density) == 2

        # Test constraint values: with larger coupling factor, constraint values should scale
        test_x = np.array([1.0, 1.0, -1.0])  # Test Taylor coefficients
        val_no_density = constraints_no_density[0]["fun"](test_x)
        val_with_density = constraints_with_density[0]["fun"](test_x)

        # coupling_with_density / coupling_no_density = 2/1 = 2
        assert abs(val_with_density / val_no_density - 2.0) < 1e-10

    @pytest.mark.skip(reason="Methods moved to MonotonicityMixin - HJBGFDMSolver needs refactoring. See issue #TBD")
    def test_hamiltonian_constraints_for_high_dim(self):
        """Test that Hamiltonian constraints work for d > 3 (nD support)."""
        problem = SimpleMFGProblem()
        # 4D points - nD generalization should handle this
        points = np.random.rand(50, 4)

        solver = HJBGFDMSolver(
            problem=problem,
            collocation_points=points,
            qp_optimization_level="auto",
            qp_constraint_mode="hamiltonian",
            delta=0.3,
        )

        # Multi-indices for 4D, order 2 (need gradients)
        solver.multi_indices = [
            (0, 0, 0, 0),  # Constant
            (1, 0, 0, 0),  # d/dx1
            (0, 1, 0, 0),  # d/dx2
            (0, 0, 1, 0),  # d/dx3
            (0, 0, 0, 1),  # d/dx4
        ]

        center_idx = 25
        center_point = points[center_idx]
        neighbor_indices = np.array([20, 21, 22, 23, 24, 25])
        neighbor_points = points[neighbor_indices]
        A = np.eye(5)

        constraints = solver._build_hamiltonian_gradient_constraints(
            A,
            neighbor_indices,
            neighbor_points,
            center_point,
            center_idx,
            u_values=None,
            m_density=0.0,
            gamma=0.0,
        )

        # With nD support, should generate constraints for neighbors (excluding center)
        # 6 neighbors minus 1 center = 5 constraints expected
        assert len(constraints) == 5
        # Each constraint should be a dict with 'type' and 'fun'
        for c in constraints:
            assert "type" in c
            assert "fun" in c
            assert c["type"] == "ineq"


if __name__ == "__main__":
    # Run tests manually
    import pytest

    pytest.main([__file__, "-v"])
