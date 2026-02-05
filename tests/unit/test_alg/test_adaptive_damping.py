"""Tests for adaptive Picard damping (Issue #583) and damping schedules (Issue #719)."""

from __future__ import annotations

import pytest

from mfg_pde.alg.numerical.coupling.fixed_point_utils import adapt_damping, compute_scheduled_damping


@pytest.mark.unit
class TestAdaptDamping:
    """Tests for the adapt_damping() function."""

    def test_no_change_on_decreasing_errors(self):
        """Steadily decreasing errors should not reduce damping."""
        theta_U, theta_M, msg = adapt_damping(
            theta_U=0.5,
            theta_M=0.5,
            error_history_U=[1.0, 0.8, 0.6],
            error_history_M=[1.0, 0.9, 0.7],
            theta_U_initial=0.5,
            theta_M_initial=0.5,
        )
        # Recovery kicks in (3 decreasing = stable_window), so theta can increase
        # But it must not exceed initial
        assert theta_U <= 0.5
        assert theta_M <= 0.5
        assert msg is None

    def test_reduces_on_oscillation(self):
        """Error increase above threshold should halve damping."""
        theta_U, theta_M, msg = adapt_damping(
            theta_U=0.5,
            theta_M=0.5,
            error_history_U=[1.0, 2.0],  # 2x increase > 1.2 threshold
            error_history_M=[1.0, 0.8],  # M converging
            theta_U_initial=0.5,
            theta_M_initial=0.5,
        )
        assert theta_U == pytest.approx(0.25)  # 0.5 * 0.5 decay
        assert theta_M == 0.5  # M unchanged (no oscillation)
        assert msg is not None
        assert "theta_U" in msg

    def test_min_damping_enforced(self):
        """Repeated oscillation should clamp at min_damping."""
        theta_U = 0.1
        for _ in range(5):
            theta_U, _, _ = adapt_damping(
                theta_U=theta_U,
                theta_M=0.5,
                error_history_U=[1.0, 5.0],  # Strong oscillation
                error_history_M=[1.0, 0.9],
                theta_U_initial=0.5,
                theta_M_initial=0.5,
                min_damping=0.05,
            )
        assert theta_U == pytest.approx(0.05)

    def test_independent_uv_adaptation(self):
        """U oscillating and M converging should only reduce U damping."""
        theta_U, theta_M, msg = adapt_damping(
            theta_U=0.5,
            theta_M=0.5,
            error_history_U=[1.0, 3.0],  # U oscillating
            error_history_M=[1.0, 0.5],  # M converging
            theta_U_initial=0.5,
            theta_M_initial=0.5,
        )
        assert theta_U < 0.5  # U reduced
        assert theta_M == 0.5  # M unchanged
        assert msg is not None
        assert "theta_U" in msg
        assert "theta_M" not in msg

    def test_recovery_after_stable_convergence(self):
        """3 stable decreasing iterations should trigger cautious recovery."""
        theta_U, theta_M, msg = adapt_damping(
            theta_U=0.3,  # Previously reduced
            theta_M=0.3,
            error_history_U=[1.0, 0.9, 0.8, 0.7],  # 3 consecutive decreases
            error_history_M=[1.0, 0.9, 0.8, 0.7],
            theta_U_initial=0.5,
            theta_M_initial=0.5,
            recovery_rate=1.05,
        )
        assert theta_U == pytest.approx(0.3 * 1.05)
        assert theta_M == pytest.approx(0.3 * 1.05)
        assert msg is None

    def test_recovery_never_exceeds_initial(self):
        """Recovered damping must not exceed initial value."""
        theta_U, theta_M, msg = adapt_damping(
            theta_U=0.49,  # Close to initial
            theta_M=0.5,
            error_history_U=[1.0, 0.9, 0.8, 0.7],
            error_history_M=[1.0, 0.9, 0.8, 0.7],
            theta_U_initial=0.5,
            theta_M_initial=0.5,
            recovery_rate=1.1,  # Would push to 0.539 > 0.5
        )
        assert theta_U == pytest.approx(0.5)  # Clamped at initial
        assert theta_M == pytest.approx(0.5)
        assert msg is None

    def test_warning_message_on_oscillation(self):
        """Oscillation should return a non-None warning string."""
        _, _, msg = adapt_damping(
            theta_U=0.5,
            theta_M=0.5,
            error_history_U=[1.0, 5.0],
            error_history_M=[1.0, 5.0],
            theta_U_initial=0.5,
            theta_M_initial=0.5,
        )
        assert msg is not None
        assert "theta_U" in msg
        assert "theta_M" in msg

    def test_no_change_with_single_error(self):
        """Only 1 data point should result in no action."""
        theta_U, theta_M, msg = adapt_damping(
            theta_U=0.5,
            theta_M=0.5,
            error_history_U=[1.0],
            error_history_M=[1.0],
            theta_U_initial=0.5,
            theta_M_initial=0.5,
        )
        assert theta_U == 0.5
        assert theta_M == 0.5
        assert msg is None


@pytest.mark.unit
class TestComputeScheduledDamping:
    """Tests for compute_scheduled_damping() (Issue #719 Phase 2)."""

    def test_constant_returns_base(self):
        """Constant schedule always returns the base damping value."""
        for k in [0, 1, 5, 99]:
            assert compute_scheduled_damping(k, 0.5, "constant") == 0.5

    def test_harmonic_decay(self):
        """Harmonic: theta(k) = base / (k+1)."""
        base = 1.0
        assert compute_scheduled_damping(0, base, "harmonic") == pytest.approx(1.0)
        assert compute_scheduled_damping(1, base, "harmonic") == pytest.approx(0.5)
        assert compute_scheduled_damping(9, base, "harmonic") == pytest.approx(0.1)

    def test_sqrt_decay(self):
        """Sqrt: theta(k) = base / sqrt(k+1)."""
        base = 1.0
        assert compute_scheduled_damping(0, base, "sqrt") == pytest.approx(1.0)
        assert compute_scheduled_damping(3, base, "sqrt") == pytest.approx(0.5)

    def test_exponential_decay(self):
        """Exponential: theta(k) = base^(k+1)."""
        assert compute_scheduled_damping(0, 0.5, "exponential") == pytest.approx(0.5)
        assert compute_scheduled_damping(1, 0.5, "exponential") == pytest.approx(0.25)
        assert compute_scheduled_damping(2, 0.5, "exponential") == pytest.approx(0.125)

    def test_min_damping_floor(self):
        """Decayed values are clamped to min_damping."""
        # Harmonic at k=999: base/(1000) = 0.001, should clamp to 0.01
        result = compute_scheduled_damping(999, 1.0, "harmonic", min_damping=0.01)
        assert result == pytest.approx(0.01)

    def test_unknown_schedule_raises(self):
        """Unknown schedule name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown damping schedule"):
            compute_scheduled_damping(0, 0.5, "cosine")

    def test_base_damping_scales_harmonic(self):
        """Non-unity base_damping correctly scales the schedule."""
        assert compute_scheduled_damping(0, 0.6, "harmonic") == pytest.approx(0.6)
        assert compute_scheduled_damping(1, 0.6, "harmonic") == pytest.approx(0.3)
