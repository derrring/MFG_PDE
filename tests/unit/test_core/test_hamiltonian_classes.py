"""
Unit tests for class-based Hamiltonian and Lagrangian (Issues #651, #667, #673).

Tests the new class hierarchy:
- MFGOperatorBase: Common base for H and L
- HamiltonianBase: Full MFG Hamiltonian H(x, m, p, t)
- LagrangianBase: Full MFG Lagrangian L(x, α, m, t)
- Legendre transform duality between H and L
"""

import pytest

import numpy as np

from mfg_pde.core.hamiltonian import (
    # Control cost classes
    BoundedControlCost,
    # Dual classes (Legendre transform)
    DualHamiltonian,
    DualLagrangian,
    L1ControlCost,
    LagrangianBase,
    OptimizationSense,
    QuadraticControlCost,
    QuadraticMFGHamiltonian,
    SeparableHamiltonian,
    create_hamiltonian,
)


class TestControlCostBase:
    """Tests for original ControlCostBase hierarchy."""

    def test_quadratic_control_cost_minimize(self):
        """Test quadratic control cost with MINIMIZE sense."""
        cost = QuadraticControlCost(sense=OptimizationSense.MINIMIZE, control_cost=2.0)
        p = np.array([1.0, 2.0, -3.0])

        alpha = cost.optimal_control(p)

        # α* = -p/λ for MINIMIZE
        expected = np.array([-0.5, -1.0, 1.5])
        np.testing.assert_allclose(alpha, expected)

    def test_quadratic_control_cost_maximize(self):
        """Test quadratic control cost with MAXIMIZE sense."""
        cost = QuadraticControlCost(sense=OptimizationSense.MAXIMIZE, control_cost=2.0)
        p = np.array([1.0, 2.0, -3.0])

        alpha = cost.optimal_control(p)

        # α* = +p/λ for MAXIMIZE
        expected = np.array([0.5, 1.0, -1.5])
        np.testing.assert_allclose(alpha, expected)

    def test_l1_control_cost_bang_bang(self):
        """Test L1 control cost gives bang-bang control."""
        cost = L1ControlCost(control_cost=1.5)
        p = np.array([0.5, 2.0, -3.0])

        alpha = cost.optimal_control(p)

        # α* = -sign(p) where |p| > λ, else 0
        expected = np.array([0.0, -1.0, 1.0])
        np.testing.assert_allclose(alpha, expected)

    def test_bounded_control_cost_clipping(self):
        """Test bounded control cost clips at max_control."""
        cost = BoundedControlCost(control_cost=1.0, max_control=1.5)
        p = np.array([1.0, 2.0, 3.0])

        alpha = cost.optimal_control(p)

        # Unconstrained would be -p, but clipped at ±1.5
        expected = np.array([-1.0, -1.5, -1.5])
        np.testing.assert_allclose(alpha, expected)


class TestMFGOperatorBase:
    """Tests for MFGOperatorBase common interface."""

    def test_hamiltonian_is_hamiltonian(self):
        """Test HamiltonianBase.is_hamiltonian returns True."""
        H = SeparableHamiltonian(control_cost=QuadraticControlCost())
        assert H.is_hamiltonian is True
        assert H.is_lagrangian is False

    def test_lagrangian_is_lagrangian(self):
        """Test LagrangianBase.is_lagrangian returns True."""

        class TestLagrangian(LagrangianBase):
            def __call__(self, x, alpha, m, t=0.0):
                return 0.5 * np.sum(alpha**2)

        L = TestLagrangian()
        assert L.is_lagrangian is True
        assert L.is_hamiltonian is False


class TestHamiltonianBase:
    """Tests for HamiltonianBase interface."""

    @pytest.fixture
    def simple_hamiltonian(self):
        """Create a simple separable Hamiltonian for testing."""
        return SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=2.0),
            coupling=lambda m: -(m**2),
            coupling_dm=lambda m: -2 * m,
        )

    def test_hamiltonian_evaluation(self, simple_hamiltonian):
        """Test Hamiltonian __call__ evaluation."""
        x = np.array([0.5])
        m = 0.3
        p = np.array([1.0])
        t = 0.0

        H_val = simple_hamiltonian(x, m, p, t)

        # H = ½|p|²/λ + f(m) = 0.5 * 1.0 / 2.0 + (-0.09) = 0.16
        expected = 0.5 * 1.0 / 2.0 - 0.3**2
        assert abs(H_val - expected) < 1e-10

    def test_hamiltonian_dp_analytic(self, simple_hamiltonian):
        """Test ∂H/∂p computation (analytic for quadratic)."""
        x = np.array([0.5])
        m = 0.3
        p = np.array([1.0])
        t = 0.0

        dp = simple_hamiltonian.dp(x, m, p, t)

        # For quadratic H_control = ½|p|²/λ, ∂H/∂p = p/λ
        expected = np.array([0.5])  # p/λ = 1/2
        np.testing.assert_allclose(dp, expected)

    def test_hamiltonian_dm_analytic(self, simple_hamiltonian):
        """Test ∂H/∂m computation (analytic when coupling_dm provided)."""
        x = np.array([0.5])
        m = 0.3
        p = np.array([1.0])
        t = 0.0

        dm = simple_hamiltonian.dm(x, m, p, t)

        # df/dm = -2m = -0.6
        assert abs(dm - (-0.6)) < 1e-10

    def test_hamiltonian_optimal_control(self, simple_hamiltonian):
        """Test optimal control computation."""
        x = np.array([0.5])
        m = 0.3
        p = np.array([1.0])
        t = 0.0

        alpha = simple_hamiltonian.optimal_control(x, m, p, t)

        # For quadratic: α* = -p/λ (MINIMIZE)
        expected = np.array([-0.5])
        np.testing.assert_allclose(alpha, expected)

    def test_hamiltonian_finite_diff_dm(self):
        """Test finite difference fallback for dm when coupling_dm not provided."""
        H = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: -(m**2),
            # No coupling_dm provided - should use finite diff
        )
        x = np.array([0.5])
        m = 0.3
        p = np.array([1.0])
        t = 0.0

        dm = H.dm(x, m, p, t)

        # Should approximate -2m = -0.6
        assert abs(dm - (-0.6)) < 0.01  # Allow some FD error


class TestLagrangianBase:
    """Tests for LagrangianBase interface."""

    @pytest.fixture
    def quadratic_lagrangian(self):
        """Create a quadratic Lagrangian for testing."""

        class QuadraticLagrangian(LagrangianBase):
            def __init__(self, lam=1.0):
                super().__init__()
                self.lam = lam

            def __call__(self, x, alpha, m, t=0.0):
                return 0.5 * self.lam * np.sum(alpha**2)

        return QuadraticLagrangian(lam=2.0)

    def test_lagrangian_evaluation(self, quadratic_lagrangian):
        """Test Lagrangian __call__ evaluation."""
        x = np.array([0.5])
        alpha = np.array([1.0])
        m = 0.3
        t = 0.0

        L_val = quadratic_lagrangian(x, alpha, m, t)

        # L = ½λ|α|² = 0.5 * 2 * 1 = 1.0
        assert abs(L_val - 1.0) < 1e-10

    def test_lagrangian_to_hamiltonian(self, quadratic_lagrangian):
        """Test Legendre transform L -> H."""
        H = quadratic_lagrangian.legendre_transform()

        assert isinstance(H, DualHamiltonian)

        # For L = ½λ|α|², H = ½|p|²/λ
        # With λ=2, p=1: H = 0.5 * 1 / 2 = 0.25
        x = np.array([0.5])
        m = 0.3
        p = np.array([1.0])
        t = 0.0

        H_val = H(x, m, p, t)
        assert abs(H_val - 0.25) < 0.05  # Allow numerical tolerance


class TestLegendreDuality:
    """Tests for Legendre transform duality (Issue #651)."""

    def test_hamiltonian_to_lagrangian(self):
        """Test inverse Legendre transform H -> L."""
        H = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=2.0),
        )
        L = H.legendre_transform()

        assert isinstance(L, DualLagrangian)

        # For H = ½|p|²/λ, L = ½λ|α|²
        # With λ=2, α=1: L = 0.5 * 2 * 1 = 1.0
        x = np.array([0.5])
        alpha = np.array([1.0])
        m = 0.3
        t = 0.0

        L_val = L(x, alpha, m, t)
        assert abs(L_val - 1.0) < 0.1  # Allow numerical tolerance

    def test_duality_cycle_l_h_l(self):
        """Test L -> H -> L recovers original."""

        class QuadraticLagrangian(LagrangianBase):
            def __init__(self, lam=1.0):
                super().__init__()
                self.lam = lam

            def __call__(self, x, alpha, m, t=0.0):
                return 0.5 * self.lam * np.sum(alpha**2)

        L_orig = QuadraticLagrangian(lam=2.0)
        H_from_L = L_orig.legendre_transform()
        L_recovered = H_from_L.legendre_transform()

        x = np.array([0.5])
        alpha = np.array([1.0])
        m = 0.3
        t = 0.0

        L_orig_val = L_orig(x, alpha, m, t)
        L_recovered_val = L_recovered(x, alpha, m, t)

        # Should recover original within numerical tolerance
        assert abs(L_recovered_val - L_orig_val) < 0.2


class TestQuadraticMFGHamiltonian:
    """Tests for QuadraticMFGHamiltonian."""

    def test_default_hamiltonian_matches_mfg_problem_default(self):
        """Test QuadraticMFGHamiltonian gives same result as MFGProblem default."""
        H = QuadraticMFGHamiltonian(coupling_coefficient=1.0)

        x = np.array([0.5])
        m = 0.3
        p = np.array([1.0])
        t = 0.0

        H_val = H(x, m, p, t)

        # H = ½c|p|² - m² = 0.5 * 1.0 * 1.0 - 0.09 = 0.41
        expected = 0.5 * 1.0 * 1.0**2 - 0.3**2
        assert abs(H_val - expected) < 1e-10

    def test_default_hamiltonian_dm(self):
        """Test QuadraticMFGHamiltonian.dm() = -2m."""
        H = QuadraticMFGHamiltonian(coupling_coefficient=1.0)

        x = np.array([0.5])
        m = 0.3
        p = np.array([1.0])
        t = 0.0

        dm = H.dm(x, m, p, t)

        assert abs(dm - (-0.6)) < 1e-10


class TestCreateHamiltonian:
    """Tests for create_hamiltonian factory function."""

    def test_create_quadratic(self):
        """Test creating quadratic Hamiltonian via factory."""
        H = create_hamiltonian("quadratic", control_cost=2.0)

        assert isinstance(H, SeparableHamiltonian)

        x = np.array([0.5])
        m = 0.0
        p = np.array([1.0])
        t = 0.0

        H_val = H(x, m, p, t)
        # H = ½|p|²/λ = 0.5 * 1 / 2 = 0.25
        assert abs(H_val - 0.25) < 1e-10

    def test_create_default(self):
        """Test creating default MFG Hamiltonian via factory."""
        H = create_hamiltonian("default", coupling_coefficient=2.0)

        assert isinstance(H, QuadraticMFGHamiltonian)
        assert H.coupling_coefficient == 2.0

    def test_create_invalid_raises(self):
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown hamiltonian_type"):
            create_hamiltonian("invalid_type")


# Issue #673: TestToLegacyFunc removed - to_legacy_func() method deleted
# Class-based Hamiltonian API is now called directly (no legacy function wrappers)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
