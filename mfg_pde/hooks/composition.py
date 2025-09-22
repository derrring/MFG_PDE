"""
Hook Composition System

Utilities for combining multiple hooks into complex behaviors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .base import SolverHooks

if TYPE_CHECKING:
    from ..types import MFGResult, SpatialTemporalState


class MultiHook(SolverHooks):
    """
    Compose multiple hooks into one.

    Executes hooks in order, with control flow handled intelligently.

    Example:
        combined = MultiHook(
            DebugHook(),
            PlottingHook(plot_every=10),
            PerformanceHook()
        )

        result = solver.solve(problem, hooks=combined)
    """

    def __init__(self, *hooks: SolverHooks):
        """
        Initialize with multiple hooks.

        Args:
            *hooks: Variable number of hook instances
        """
        self.hooks = list(hooks)

    def add_hook(self, hook: SolverHooks) -> None:
        """Add a hook to the composition."""
        self.hooks.append(hook)

    def remove_hook(self, hook: SolverHooks) -> bool:
        """Remove a hook from the composition. Returns True if found and removed."""
        try:
            self.hooks.remove(hook)
            return True
        except ValueError:
            return False

    def on_solve_start(self, initial_state: SpatialTemporalState) -> None:
        """Execute all hooks' on_solve_start methods."""
        for hook in self.hooks:
            hook.on_solve_start(initial_state)

    def on_iteration_start(self, state: SpatialTemporalState) -> None:
        """Execute all hooks' on_iteration_start methods."""
        for hook in self.hooks:
            hook.on_iteration_start(state)

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """
        Execute all hooks' on_iteration_end methods.

        First hook to request control flow wins.
        """
        for hook in self.hooks:
            result = hook.on_iteration_end(state)
            if result:  # First hook to request control flow wins
                return result
        return None

    def on_convergence_check(self, state: SpatialTemporalState) -> bool | None:
        """
        Execute all hooks' on_convergence_check methods.

        First definitive answer (True/False) wins.
        """
        for hook in self.hooks:
            result = hook.on_convergence_check(state)
            if result is not None:  # First definitive answer wins
                return result
        return None

    def on_solve_end(self, result: MFGResult) -> MFGResult:
        """Execute all hooks' on_solve_end methods in sequence."""
        for hook in self.hooks:
            result = hook.on_solve_end(result)
        return result


class ConditionalHook(SolverHooks):
    """
    Execute hook only when condition is met.

    Example:
        # Only plot every 10th iteration
        every_10th = ConditionalHook(
            PlottingHook(),
            lambda state: state.iteration % 10 == 0
        )

        # Only debug when residual is high
        debug_when_struggling = ConditionalHook(
            DebugHook(),
            lambda state: state.residual > 1e-3
        )
    """

    def __init__(self, hook: SolverHooks, condition: Callable[[SpatialTemporalState], bool]):
        """
        Initialize conditional hook.

        Args:
            hook: The hook to execute conditionally
            condition: Function that takes state and returns bool
        """
        self.hook = hook
        self.condition = condition

    def on_solve_start(self, initial_state: SpatialTemporalState) -> None:
        if self.condition(initial_state):
            self.hook.on_solve_start(initial_state)

    def on_iteration_start(self, state: SpatialTemporalState) -> None:
        if self.condition(state):
            self.hook.on_iteration_start(state)

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        if self.condition(state):
            return self.hook.on_iteration_end(state)
        return None

    def on_convergence_check(self, state: SpatialTemporalState) -> bool | None:
        if self.condition(state):
            return self.hook.on_convergence_check(state)
        return None

    def on_solve_end(self, result: MFGResult) -> MFGResult:
        # Note: For solve_end, we check the final state if available
        # Otherwise, we always execute (conservative approach)
        try:
            # Try to get final state information for condition checking
            if hasattr(result, "final_state"):
                final_state = getattr(result, "final_state", None)
                if final_state is not None and hasattr(final_state, "iteration"):
                    if self.condition(final_state):
                        return self.hook.on_solve_end(result)
                    else:
                        return result
            # If no final_state available, execute hook unconditionally
            return self.hook.on_solve_end(result)
        except (AttributeError, TypeError):
            # If any attribute access fails, execute hook unconditionally (safe default)
            return self.hook.on_solve_end(result)


class PriorityHook(SolverHooks):
    """
    Execute hooks in priority order.

    Higher priority numbers execute first.

    Example:
        priority_hook = PriorityHook()
        priority_hook.add_hook(DebugHook(), priority=10)
        priority_hook.add_hook(PlottingHook(), priority=5)
        priority_hook.add_hook(CleanupHook(), priority=1)
        # Execution order: Debug -> Plotting -> Cleanup
    """

    def __init__(self):
        self._hooks_by_priority: list[tuple[int, SolverHooks]] = []

    def add_hook(self, hook: SolverHooks, priority: int = 0) -> None:
        """Add hook with specified priority."""
        self._hooks_by_priority.append((priority, hook))
        # Keep sorted by priority (highest first)
        self._hooks_by_priority.sort(key=lambda x: x[0], reverse=True)

    def _get_ordered_hooks(self) -> list[SolverHooks]:
        """Get hooks in priority order."""
        return [hook for _, hook in self._hooks_by_priority]

    def on_solve_start(self, initial_state: SpatialTemporalState) -> None:
        for hook in self._get_ordered_hooks():
            hook.on_solve_start(initial_state)

    def on_iteration_start(self, state: SpatialTemporalState) -> None:
        for hook in self._get_ordered_hooks():
            hook.on_iteration_start(state)

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        for hook in self._get_ordered_hooks():
            result = hook.on_iteration_end(state)
            if result:
                return result
        return None

    def on_convergence_check(self, state: SpatialTemporalState) -> bool | None:
        for hook in self._get_ordered_hooks():
            result = hook.on_convergence_check(state)
            if result is not None:
                return result
        return None

    def on_solve_end(self, result: MFGResult) -> MFGResult:
        for hook in self._get_ordered_hooks():
            result = hook.on_solve_end(result)
        return result


class FilterHook(SolverHooks):
    """
    Filter which hook methods are called based on dynamic conditions.

    Unlike ConditionalHook which evaluates the same condition for all methods,
    FilterHook allows different conditions for different hook methods.

    Example:
        filter_hook = FilterHook(DebugHook())
        filter_hook.filter_solve_start(lambda state: True)  # Always call
        filter_hook.filter_iteration_end(lambda state: state.iteration % 5 == 0)  # Every 5th
        filter_hook.filter_convergence_check(lambda state: state.residual < 1e-4)  # Only when close
    """

    def __init__(self, hook: SolverHooks):
        self.hook = hook
        self._solve_start_filter: Callable | None = None
        self._iteration_start_filter: Callable | None = None
        self._iteration_end_filter: Callable | None = None
        self._convergence_check_filter: Callable | None = None
        self._solve_end_filter: Callable | None = None

    def filter_solve_start(self, condition: Callable[[SpatialTemporalState], bool]) -> FilterHook:
        """Set condition for on_solve_start calls."""
        self._solve_start_filter = condition
        return self

    def filter_iteration_start(self, condition: Callable[[SpatialTemporalState], bool]) -> FilterHook:
        """Set condition for on_iteration_start calls."""
        self._iteration_start_filter = condition
        return self

    def filter_iteration_end(self, condition: Callable[[SpatialTemporalState], bool]) -> FilterHook:
        """Set condition for on_iteration_end calls."""
        self._iteration_end_filter = condition
        return self

    def filter_convergence_check(self, condition: Callable[[SpatialTemporalState], bool]) -> FilterHook:
        """Set condition for on_convergence_check calls."""
        self._convergence_check_filter = condition
        return self

    def filter_solve_end(self, condition: Callable[[MFGResult], bool]) -> FilterHook:
        """Set condition for on_solve_end calls."""
        self._solve_end_filter = condition
        return self

    def on_solve_start(self, initial_state: SpatialTemporalState) -> None:
        if self._solve_start_filter is None or self._solve_start_filter(initial_state):
            self.hook.on_solve_start(initial_state)

    def on_iteration_start(self, state: SpatialTemporalState) -> None:
        if self._iteration_start_filter is None or self._iteration_start_filter(state):
            self.hook.on_iteration_start(state)

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        if self._iteration_end_filter is None or self._iteration_end_filter(state):
            return self.hook.on_iteration_end(state)
        return None

    def on_convergence_check(self, state: SpatialTemporalState) -> bool | None:
        if self._convergence_check_filter is None or self._convergence_check_filter(state):
            return self.hook.on_convergence_check(state)
        return None

    def on_solve_end(self, result: MFGResult) -> MFGResult:
        if self._solve_end_filter is None or self._solve_end_filter(result):
            return self.hook.on_solve_end(result)
        return result


class TransformHook(SolverHooks):
    """
    Transform state or result data before passing to wrapped hook.

    This allows for data preprocessing, unit conversion, or state manipulation
    before hook execution.

    Example:
        # Convert residual to log scale for visualization
        log_debug = TransformHook(
            DebugHook(),
            state_transform=lambda state: state.copy_with_updates(
                residual=math.log10(state.residual + 1e-16)
            )
        )
    """

    def __init__(
        self,
        hook: SolverHooks,
        state_transform: Callable[[SpatialTemporalState], SpatialTemporalState] | None = None,
        result_transform: Callable[[MFGResult], MFGResult] | None = None,
    ):
        self.hook = hook
        self.state_transform = state_transform
        self.result_transform = result_transform

    def _transform_state(self, state: SpatialTemporalState) -> SpatialTemporalState:
        """Apply state transformation if provided."""
        if self.state_transform:
            return self.state_transform(state)
        return state

    def _transform_result(self, result: MFGResult) -> MFGResult:
        """Apply result transformation if provided."""
        if self.result_transform:
            return self.result_transform(result)
        return result

    def on_solve_start(self, initial_state: SpatialTemporalState) -> None:
        transformed_state = self._transform_state(initial_state)
        self.hook.on_solve_start(transformed_state)

    def on_iteration_start(self, state: SpatialTemporalState) -> None:
        transformed_state = self._transform_state(state)
        self.hook.on_iteration_start(transformed_state)

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        transformed_state = self._transform_state(state)
        return self.hook.on_iteration_end(transformed_state)

    def on_convergence_check(self, state: SpatialTemporalState) -> bool | None:
        transformed_state = self._transform_state(state)
        return self.hook.on_convergence_check(transformed_state)

    def on_solve_end(self, result: MFGResult) -> MFGResult:
        transformed_result = self._transform_result(result)
        final_result = self.hook.on_solve_end(transformed_result)
        return self._transform_result(final_result)


class ChainHook(SolverHooks):
    """
    Chain hooks together so that output of one becomes input of the next.

    Unlike MultiHook which executes hooks in parallel, ChainHook creates
    a pipeline where each hook can modify the state/result for the next.

    Example:
        # Data processing pipeline
        pipeline = ChainHook([
            NormalizationHook(),  # Normalize state data
            FilteringHook(),      # Apply noise filtering
            AnalysisHook()        # Perform analysis on clean data
        ])
    """

    def __init__(self, hooks: list[SolverHooks]):
        self.hooks = hooks

    def on_solve_start(self, initial_state: SpatialTemporalState) -> None:
        """Execute hooks in sequence, each potentially modifying state."""
        current_state = initial_state
        for hook in self.hooks:
            # For on_solve_start, hooks don't return modified state,
            # so we just call them in sequence
            hook.on_solve_start(current_state)

    def on_iteration_start(self, state: SpatialTemporalState) -> None:
        """Execute hooks in sequence."""
        current_state = state
        for hook in self.hooks:
            hook.on_iteration_start(current_state)

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """Execute hooks in sequence, first control signal wins."""
        current_state = state
        for hook in self.hooks:
            result = hook.on_iteration_end(current_state)
            if result:  # First hook to request control flow wins
                return result
        return None

    def on_convergence_check(self, state: SpatialTemporalState) -> bool | None:
        """Execute hooks in sequence, first definitive answer wins."""
        current_state = state
        for hook in self.hooks:
            result = hook.on_convergence_check(current_state)
            if result is not None:
                return result
        return None

    def on_solve_end(self, result: MFGResult) -> MFGResult:
        """Execute hooks in sequence, chaining result modifications."""
        current_result = result
        for hook in self.hooks:
            current_result = hook.on_solve_end(current_result)
        return current_result
