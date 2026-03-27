"""Tests for advanced control flow hooks."""

import pytest

import numpy as np

from mfgarchon.hooks.control_flow import (
    AdaptiveControlHook,
    ControlState,
    PerformanceControlHook,
)
from mfgarchon.types.state import SpatialTemporalState


def _make_state(iteration=0, residual=1.0):
    return SpatialTemporalState(
        u=np.zeros((5, 10)),
        m=np.ones((5, 10)),
        iteration=iteration,
        residual=residual,
        metadata={},
    )


@pytest.mark.unit
class TestControlState:
    def test_basic_creation(self):
        cs = ControlState(action="stop", reason="test", metadata={})
        assert cs.action == "stop"
        assert cs.reason == "test"

    def test_metadata_default(self):
        cs = ControlState(action="continue", reason="ok", metadata=None)
        assert cs.metadata == {}


@pytest.mark.unit
class TestAdaptiveControlHook:
    def test_empty_hook_returns_none(self):
        """Hook with no rules should not intervene."""
        hook = AdaptiveControlHook()
        assert hook.on_iteration_end(_make_state()) is None

    def test_stop_rule_triggers(self):
        """Stop rule should return 'stop' when condition is met."""
        hook = AdaptiveControlHook()
        hook.add_stop_rule(
            condition=lambda state: state.residual < 0.01,
            reason="Converged enough",
        )

        # Not triggered
        assert hook.on_iteration_end(_make_state(residual=1.0)) is None

        # Triggered
        assert hook.on_iteration_end(_make_state(residual=0.001)) == "stop"

    def test_convergence_rule_restart(self):
        """Convergence rule with restart action should return 'restart'."""
        hook = AdaptiveControlHook()
        hook.add_convergence_rule(
            condition=lambda state: state.iteration > 10 and state.residual > 0.5,
            action="restart",
            reason="Stagnated",
        )

        # Not triggered (low iteration)
        assert hook.on_iteration_end(_make_state(iteration=5, residual=0.8)) is None

        # Triggered
        assert hook.on_iteration_end(_make_state(iteration=15, residual=0.8)) == "restart"
        assert hook.restart_count == 1

    def test_max_restarts_respected(self):
        """Should stop restarting after max_restarts."""
        hook = AdaptiveControlHook()
        hook.max_restarts = 2
        hook.add_convergence_rule(
            condition=lambda state: True,
            action="restart",
        )

        state = _make_state()
        assert hook.on_iteration_end(state) == "restart"  # restart 1
        assert hook.on_iteration_end(state) == "restart"  # restart 2
        assert hook.on_iteration_end(state) is None  # max reached, no more restarts

    def test_stop_rules_checked_before_convergence(self):
        """Stop rules have priority over convergence rules."""
        hook = AdaptiveControlHook()
        hook.add_stop_rule(condition=lambda s: True, reason="Always stop")
        hook.add_convergence_rule(condition=lambda s: True, action="restart")

        assert hook.on_iteration_end(_make_state()) == "stop"

    def test_force_converge(self):
        """Convergence check can force convergence."""
        hook = AdaptiveControlHook()
        hook.add_convergence_rule(
            condition=lambda s: s.residual < 0.1,
            action="force_converge",
        )

        assert hook.on_convergence_check(_make_state(residual=0.01)) is True
        assert hook.on_convergence_check(_make_state(residual=1.0)) is None

    def test_force_continue(self):
        """Convergence check can prevent early convergence."""
        hook = AdaptiveControlHook()
        hook.add_convergence_rule(
            condition=lambda s: s.iteration < 5,
            action="force_continue",
        )

        assert hook.on_convergence_check(_make_state(iteration=2)) is False
        assert hook.on_convergence_check(_make_state(iteration=10)) is None


@pytest.mark.unit
class TestPerformanceControlHook:
    def test_initial_state(self):
        hook = PerformanceControlHook()
        assert hook.stagnation_limit == 20
        assert hook.slow_progress_threshold == 0.01
        assert len(hook.residual_history) == 0

    def test_set_stagnation_limit(self):
        hook = PerformanceControlHook()
        hook.set_stagnation_limit(5)
        assert hook.stagnation_limit == 5

    def test_set_slow_progress_threshold(self):
        hook = PerformanceControlHook()
        hook.set_slow_progress_threshold(0.05)
        assert hook.slow_progress_threshold == 0.05
