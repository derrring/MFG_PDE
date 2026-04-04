"""Tests for regime switching configuration and validation.

Tests RegimeSwitchingConfig from mfgarchon.core.regime_switching.

Covers:
- Generator matrix validation (row sums, non-negative off-diags)
- Stationary distribution computation
- Transition rate access
- Edge cases (single regime, large K)
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.regime_switching import RegimeSwitchingConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_state_config():
    Q = np.array([[-0.1, 0.1], [0.2, -0.2]])
    return RegimeSwitchingConfig(
        transition_matrix=Q,
        regime_names=["high", "low"],
    )


@pytest.fixture
def three_state_config():
    Q = np.array([[-0.3, 0.2, 0.1], [0.1, -0.2, 0.1], [0.05, 0.15, -0.2]])
    return RegimeSwitchingConfig(transition_matrix=Q)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_two_state(self, two_state_config):
        two_state_config.validate()  # Should not raise

    def test_valid_three_state(self, three_state_config):
        three_state_config.validate()  # Should not raise

    def test_invalid_not_square(self):
        Q = np.array([[1, 2, 3], [4, 5, 6]])
        config = RegimeSwitchingConfig(transition_matrix=Q)
        with pytest.raises(ValueError, match="square"):
            config.validate()

    def test_invalid_negative_offdiag(self):
        Q = np.array([[-0.1, 0.1], [-0.2, 0.2]])  # Q[1,0] < 0
        config = RegimeSwitchingConfig(transition_matrix=Q)
        with pytest.raises(ValueError, match="non-negative"):
            config.validate()

    def test_invalid_rows_not_zero(self):
        Q = np.array([[-0.1, 0.2], [0.2, -0.2]])  # Row 0 sums to 0.1
        config = RegimeSwitchingConfig(transition_matrix=Q)
        with pytest.raises(ValueError, match="sum to zero"):
            config.validate()

    def test_invalid_regime_names_length(self):
        Q = np.array([[-0.1, 0.1], [0.2, -0.2]])
        config = RegimeSwitchingConfig(transition_matrix=Q, regime_names=["a", "b", "c"])
        with pytest.raises(ValueError, match="regime_names length"):
            config.validate()


# ---------------------------------------------------------------------------
# Properties and accessors
# ---------------------------------------------------------------------------


class TestProperties:
    def test_n_regimes(self, two_state_config, three_state_config):
        assert two_state_config.n_regimes == 2
        assert three_state_config.n_regimes == 3

    def test_transition_rate(self, two_state_config):
        assert two_state_config.transition_rate(0, 1) == pytest.approx(0.1)
        assert two_state_config.transition_rate(1, 0) == pytest.approx(0.2)
        assert two_state_config.transition_rate(0, 0) == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# Stationary distribution
# ---------------------------------------------------------------------------


class TestStationaryDistribution:
    def test_two_state_analytic(self, two_state_config):
        """For Q = [[-a, a], [b, -b]], pi = [b/(a+b), a/(a+b)]."""
        pi = two_state_config.stationary_distribution()
        assert pi[0] == pytest.approx(2 / 3, abs=1e-10)
        assert pi[1] == pytest.approx(1 / 3, abs=1e-10)

    def test_sums_to_one(self, three_state_config):
        pi = three_state_config.stationary_distribution()
        assert abs(pi.sum() - 1.0) < 1e-10

    def test_non_negative(self, three_state_config):
        pi = three_state_config.stationary_distribution()
        assert np.all(pi >= 0)

    def test_is_stationary(self, three_state_config):
        """pi @ Q should be zero vector."""
        pi = three_state_config.stationary_distribution()
        Q = three_state_config.transition_matrix
        residual = pi @ Q
        np.testing.assert_allclose(residual, 0.0, atol=1e-10)

    def test_symmetric_two_state(self):
        """Symmetric rates -> uniform distribution."""
        Q = np.array([[-0.5, 0.5], [0.5, -0.5]])
        config = RegimeSwitchingConfig(transition_matrix=Q)
        pi = config.stationary_distribution()
        np.testing.assert_allclose(pi, [0.5, 0.5], atol=1e-10)

    def test_single_regime(self):
        """One regime: trivially pi = [1]."""
        Q = np.array([[0.0]])
        config = RegimeSwitchingConfig(transition_matrix=Q)
        config.validate()
        pi = config.stationary_distribution()
        assert pi[0] == pytest.approx(1.0)
