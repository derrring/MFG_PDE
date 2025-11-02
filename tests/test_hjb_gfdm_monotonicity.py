"""
Unit tests for HJB GFDM monotonicity violation detection.

Tests the _check_monotonicity_violation() method for basic M-matrix checking.
"""

import numpy as np

from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.hjb_solvers import HJBGFDMSolver


class SimpleMFGProblem(MFGProblem):
    """Minimal MFG problem for testing GFDM solver."""

    def __init__(self):
        super().__init__()
        self.sigma = 0.1  # Diffusion coefficient
        self.domain = [0.0, 1.0, 0.0, 1.0]  # 2D domain
        self.T = 1.0

    def hamiltonian(self, p, m, x, t):
        """Simple quadratic Hamiltonian."""
        return 0.5 * np.sum(p**2)

    def terminal_condition(self, x):
        """Simple terminal condition."""
        return 0.0


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


if __name__ == "__main__":
    # Run tests manually
    import pytest

    pytest.main([__file__, "-v"])
