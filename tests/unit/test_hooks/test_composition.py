"""Tests for hook composition system."""

import pytest

import numpy as np

from mfgarchon.hooks.base import SolverHooks
from mfgarchon.hooks.composition import MultiHook
from mfgarchon.types.state import SpatialTemporalState


def _make_state(iteration=0, residual=1.0):
    """Create a minimal SpatialTemporalState for testing."""
    return SpatialTemporalState(
        u=np.zeros((5, 10)),
        m=np.ones((5, 10)),
        iteration=iteration,
        residual=residual,
        metadata={},
    )


class TrackingHook(SolverHooks):
    """Hook that records which methods were called."""

    def __init__(self, name=""):
        self.name = name
        self.calls = []

    def on_solve_start(self, initial_state):
        self.calls.append(f"{self.name}:solve_start")

    def on_iteration_start(self, state):
        self.calls.append(f"{self.name}:iter_start")

    def on_iteration_end(self, state):
        self.calls.append(f"{self.name}:iter_end")
        return None

    def on_solve_end(self, result):
        self.calls.append(f"{self.name}:solve_end")


class StopHook(SolverHooks):
    """Hook that requests stop on iteration_end."""

    def on_iteration_end(self, state):
        return "stop"


@pytest.mark.unit
class TestMultiHook:
    def test_empty_multihook(self):
        """MultiHook with no hooks should work silently."""
        multi = MultiHook()
        state = _make_state()
        multi.on_solve_start(state)
        multi.on_iteration_start(state)
        assert multi.on_iteration_end(state) is None

    def test_single_hook(self):
        """MultiHook with one hook delegates correctly."""
        h = TrackingHook("A")
        multi = MultiHook(h)
        state = _make_state()

        multi.on_solve_start(state)
        multi.on_iteration_start(state)
        multi.on_iteration_end(state)

        assert h.calls == ["A:solve_start", "A:iter_start", "A:iter_end"]

    def test_multiple_hooks_execution_order(self):
        """Hooks execute in the order they were added."""
        h1 = TrackingHook("A")
        h2 = TrackingHook("B")
        multi = MultiHook(h1, h2)
        state = _make_state()

        multi.on_solve_start(state)
        multi.on_iteration_start(state)

        assert h1.calls == ["A:solve_start", "A:iter_start"]
        assert h2.calls == ["B:solve_start", "B:iter_start"]

    def test_add_hook(self):
        """Hooks can be added dynamically."""
        multi = MultiHook()
        h = TrackingHook("late")
        multi.add_hook(h)
        state = _make_state()

        multi.on_solve_start(state)
        assert h.calls == ["late:solve_start"]

    def test_remove_hook(self):
        """Hooks can be removed."""
        h1 = TrackingHook("A")
        h2 = TrackingHook("B")
        multi = MultiHook(h1, h2)

        assert multi.remove_hook(h1) is True
        assert len(multi.hooks) == 1

    def test_remove_nonexistent_hook(self):
        """Removing a hook not in the list returns False."""
        multi = MultiHook()
        h = TrackingHook("X")
        assert multi.remove_hook(h) is False

    def test_first_control_flow_wins(self):
        """First hook returning a control signal wins."""
        h1 = StopHook()
        h2 = TrackingHook("B")
        multi = MultiHook(h1, h2)
        state = _make_state()

        result = multi.on_iteration_end(state)
        assert result == "stop"
        # h2 should NOT have been called for on_iteration_end
        assert "B:iter_end" not in h2.calls

    def test_no_control_flow_returns_none(self):
        """If no hook requests control, returns None."""
        h1 = TrackingHook("A")
        h2 = TrackingHook("B")
        multi = MultiHook(h1, h2)
        state = _make_state()

        assert multi.on_iteration_end(state) is None


@pytest.mark.unit
class TestMultiHookConvergenceCheck:
    def test_convergence_override_true(self):
        """Hook can force convergence."""

        class ForceConverge(SolverHooks):
            def on_convergence_check(self, state):
                return True

        multi = MultiHook(ForceConverge())
        assert multi.on_convergence_check(_make_state()) is True

    def test_convergence_override_false(self):
        """Hook can prevent convergence."""

        class PreventConverge(SolverHooks):
            def on_convergence_check(self, state):
                return False

        multi = MultiHook(PreventConverge())
        assert multi.on_convergence_check(_make_state()) is False

    def test_no_override_returns_none(self):
        """Without override, returns None (use default check)."""
        multi = MultiHook(TrackingHook("A"))
        assert multi.on_convergence_check(_make_state()) is None
