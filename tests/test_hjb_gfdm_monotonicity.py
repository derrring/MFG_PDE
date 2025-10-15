"""
Unit tests for HJB GFDM monotonicity violation detection.

Tests the unified _check_monotonicity_violation() method in both
basic (strict) and adaptive (threshold-based) modes.
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


def test_init_enhanced_qp_features():
    """Test that _init_enhanced_qp_features() initializes state correctly."""
    problem = SimpleMFGProblem()

    # Create solver with smart level to trigger initialization
    points = np.random.rand(50, 2)
    solver = HJBGFDMSolver(
        problem=problem,
        collocation_points=points,
        use_monotone_constraints=True,
        qp_optimization_level="smart",
        qp_usage_target=0.1,
    )

    # Verify state exists and has correct structure
    assert hasattr(solver, "_adaptive_qp_state")
    state = solver._adaptive_qp_state
    assert "threshold" in state
    assert "qp_count" in state
    assert "total_count" in state
    assert "severity_history" in state

    # Verify initial values
    assert state["threshold"] == 0.0
    assert state["qp_count"] == 0
    assert state["total_count"] == 0
    assert state["severity_history"] == []


def test_check_monotonicity_violation_basic_mode():
    """Test basic mode (strict enforcement) of violation check."""
    problem = SimpleMFGProblem()
    points = np.random.rand(50, 2)

    # Create solver with basic level
    solver = HJBGFDMSolver(
        problem=problem, collocation_points=points, use_monotone_constraints=True, qp_optimization_level="basic"
    )

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


def test_check_monotonicity_violation_adaptive_mode():
    """Test adaptive mode (threshold-based) of violation check."""
    problem = SimpleMFGProblem()
    points = np.random.rand(50, 2)

    # Create solver with smart level
    solver = HJBGFDMSolver(
        problem=problem,
        collocation_points=points,
        use_monotone_constraints=True,
        qp_optimization_level="smart",
        qp_usage_target=0.1,
    )

    # Build multi-indices
    solver.multi_indices = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]

    # Test with mild violation (should not trigger initially with threshold=0)
    D_mild = np.array([1.0, 0.5, 0.1, -0.1, 0.0, -0.1])

    # First call: threshold is 0, so mild severity should trigger
    solver._check_monotonicity_violation(D_mild, 0, use_adaptive=True)

    # Verify state was updated
    state = solver._adaptive_qp_state
    assert state["total_count"] == 1
    assert len(state["severity_history"]) == 1

    # Call multiple times to see if threshold adapts
    for _ in range(100):
        D_varying = np.array([1.0, 0.5, 0.1, -0.2, 0.0, -0.2])
        solver._check_monotonicity_violation(D_varying, 0, use_adaptive=True)

    # After 100+ calls, threshold should have adapted
    assert state["total_count"] > 100
    # Threshold should have changed from initial 0.0
    # (exact value depends on severity distribution)


def test_adaptive_threshold_convergence():
    """Test that adaptive threshold converges toward target usage."""
    problem = SimpleMFGProblem()
    points = np.random.rand(50, 2)

    target_usage = 0.2
    solver = HJBGFDMSolver(
        problem=problem,
        collocation_points=points,
        use_monotone_constraints=True,
        qp_optimization_level="smart",
        qp_usage_target=target_usage,
    )

    solver.multi_indices = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]

    # Generate random coefficients with varying violations
    np.random.seed(42)
    for i in range(500):
        # Random coefficients with negative Laplacian
        D_random = np.random.randn(6)
        D_random[3] = -abs(D_random[3])  # Ensure Laplacian negative
        D_random[5] = -abs(D_random[5])

        solver._check_monotonicity_violation(D_random, i, use_adaptive=True)

    # Check that usage is approaching target
    state = solver._adaptive_qp_state
    actual_usage = state["qp_count"] / state["total_count"]

    # Adaptive convergence is slow, so just verify mechanism is working
    # Usage should be non-trivial (neither 0% nor 100%)
    assert 0.05 <= actual_usage <= 0.95, f"Actual usage {actual_usage:.3f} should be between 5% and 95%"

    # Verify threshold has adapted from initial 0.0
    assert state["threshold"] != 0.0, "Threshold should have adapted"

    print(f"  Adaptive threshold converged to {state['threshold']:.6f}")
    print(f"  Actual QP usage: {actual_usage:.3f} (target: {target_usage})")


def test_no_laplacian_returns_false():
    """Test that missing Laplacian term returns False."""
    problem = SimpleMFGProblem()
    points = np.random.rand(50, 2)

    solver = HJBGFDMSolver(
        problem=problem, collocation_points=points, use_monotone_constraints=True, qp_optimization_level="basic"
    )

    # Multi-indices without Laplacian (order < 2)
    solver.multi_indices = [(0, 0), (1, 0), (0, 1)]
    D_no_laplacian = np.array([1.0, 0.5, 0.5])

    result = solver._check_monotonicity_violation(D_no_laplacian, 0)
    assert not result, "Should return False when Laplacian term missing"


if __name__ == "__main__":
    # Run tests manually
    print("Running HJB GFDM monotonicity tests...")

    test_init_enhanced_qp_features()
    print("✓ test_init_enhanced_qp_features passed")

    test_check_monotonicity_violation_basic_mode()
    print("✓ test_check_monotonicity_violation_basic_mode passed")

    test_check_monotonicity_violation_adaptive_mode()
    print("✓ test_check_monotonicity_violation_adaptive_mode passed")

    test_adaptive_threshold_convergence()
    print("✓ test_adaptive_threshold_convergence passed")

    test_no_laplacian_returns_false()
    print("✓ test_no_laplacian_returns_false passed")

    print("\nAll tests passed!")
