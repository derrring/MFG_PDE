"""
Averaging (Fictitious Play) Iteration Pattern.

Abstract pattern for averaging iteration with decaying learning rate:
    x_{n+1} = (1 - alpha_n) * x_n + alpha_n * T(x_n)

where alpha_n = 1/(n+1) (harmonic) or other decaying schedule.

This is equivalent to Cesaro averaging:
    x_n = (1/n) * sum_{k=1}^{n} T(x_{k-1})

Key advantages over fixed damping:
- Proven convergence for potential MFGs (even when Picard fails)
- Noise suppression through averaging
- Robust for long time horizons

This pattern applies to:
- MFG: Fictitious Play for potential games
- RL: Fictitious Play for multi-agent learning
- Game Theory: Learning in games
- Optimization: Averaged gradient methods

References:
    Cardaliaguet & Hadikhanloo (2017). Learning in mean field games:
    the fictitious play. ESAIM: COCV.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, runtime_checkable

from ..convergence import ConvergenceResult, ConvergenceTracker, check_convergence
from ..schedules import get_schedule, harmonic_schedule
from .picard import FixedPointIteratorBase

if TYPE_CHECKING:
    from collections.abc import Callable

# Generic type for iteration state
StateT = TypeVar("StateT")


@runtime_checkable
class AveragingPattern(Protocol[StateT]):
    """
    Protocol for averaging (Fictitious Play) iteration.

    Extends PicardPattern with decaying learning rate support.
    """

    def forward_step(self, state: StateT) -> StateT:
        """Compute forward step T(x_n)."""
        ...

    def get_learning_rate(self, iteration: int) -> float:
        """
        Get learning rate for iteration.

        Standard Fictitious Play: alpha(k) = 1/(k+1)
        """
        ...

    def apply_averaging(self, old_state: StateT, new_state: StateT, alpha: float) -> StateT:
        """
        Apply averaging update.

        Formula: result = (1 - alpha) * old + alpha * new
        """
        ...


class AveragingIterator(FixedPointIteratorBase[StateT], Generic[StateT]):
    """
    Base class for averaging (Fictitious Play) iterators.

    Extends FixedPointIteratorBase with:
    - Decaying learning rate schedules
    - Learning rate history tracking
    - Optional selective averaging (e.g., average M but not U)

    The key difference from standard Picard iteration is that the
    "damping" parameter decays over iterations following a schedule.

    Example:
        >>> class MyFictitiousPlay(AveragingIterator[tuple[NDArray, NDArray]]):
        ...     def forward_step(self, state):
        ...         U, M = state
        ...         U_new = solve_hjb(M)  # Full best response
        ...         M_new = solve_fp(U_new)
        ...         return (U_new, M_new)
        ...
        ...     def apply_averaging(self, old, new, alpha):
        ...         U_old, M_old = old
        ...         U_new, M_new = new
        ...         # Only average M (standard fictitious play)
        ...         return (U_new, (1-alpha)*M_old + alpha*M_new)
        ...
        >>> fp = MyFictitiousPlay(schedule="harmonic")
        >>> result = fp.solve(max_iterations=100)
    """

    def __init__(
        self,
        schedule: str | Callable[[int], float] = "harmonic",
        initial_learning_rate: float = 1.0,
        min_learning_rate: float = 0.01,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
    ):
        """
        Initialize averaging iterator.

        Args:
            schedule: Learning rate schedule name or callable
                - "harmonic": 1/(k+1) - standard fictitious play
                - "sqrt": 1/sqrt(k+1) - faster initial progress
                - "polynomial": 1/(k+1)^0.6 - balanced
                - Callable[[int], float]: Custom schedule
            initial_learning_rate: Scale factor for learning rate
            min_learning_rate: Floor for learning rate (prevents stalling)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Show progress
        """
        # Initialize base with damping=1.0 (will be overridden per-iteration)
        super().__init__(
            damping=1.0,
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=verbose,
        )

        # Learning rate schedule
        self._schedule_fn = get_schedule(schedule)
        self._schedule_name = schedule if isinstance(schedule, str) else "custom"
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate

        # Learning rate history
        self.learning_rate_history: list[float] = []

    def get_learning_rate(self, iteration: int) -> float:
        """
        Compute learning rate for given iteration.

        Applies schedule, scales by initial rate, and clamps to minimum.

        Args:
            iteration: Current iteration (0-indexed)

        Returns:
            Learning rate in (0, 1]
        """
        alpha = self._schedule_fn(iteration)
        alpha = self.initial_learning_rate * alpha
        alpha = max(alpha, self.min_learning_rate)
        return alpha

    @abstractmethod
    def apply_averaging(self, old_state: StateT, new_state: StateT, alpha: float) -> StateT:
        """
        Apply averaging update with learning rate alpha.

        This may differ from standard damping. For example, in MFG
        Fictitious Play, we typically:
        - Use full best response for U (no averaging)
        - Average only M with decaying rate

        Args:
            old_state: Previous state
            new_state: State from forward_step
            alpha: Learning rate from schedule

        Returns:
            Updated state
        """
        ...

    def apply_damping(self, old_state: StateT, new_state: StateT, alpha: float) -> StateT:
        """
        Override base class damping to use averaging.

        Note: alpha parameter from base class is ignored; we compute
        learning rate based on iteration number.
        """
        # Get current iteration from tracker
        iteration = self.iterations_run
        lr = self.get_learning_rate(iteration)
        self.learning_rate_history.append(lr)
        return self.apply_averaging(old_state, new_state, lr)

    def solve(
        self,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        **kwargs,
    ) -> ConvergenceResult:
        """
        Run averaging iteration until convergence.

        Args:
            max_iterations: Override max iterations
            tolerance: Override tolerance

        Returns:
            ConvergenceResult with final status
        """
        # Reset learning rate history
        self.learning_rate_history = []

        # Call base class solve (damping parameter is ignored)
        return super().solve(max_iterations=max_iterations, tolerance=tolerance, damping=1.0)

    def get_convergence_data(self) -> dict:
        """Get convergence data including learning rate history."""
        data = super().get_convergence_history()
        data["learning_rate_history"] = self.learning_rate_history
        data["schedule"] = self._schedule_name
        return data


if __name__ == "__main__":
    """Quick smoke test for development."""
    import numpy as np

    print("Testing AveragingPattern and AveragingIterator...")

    # Test that Protocol is properly defined
    assert AveragingPattern is not None
    print("  AveragingPattern protocol defined")

    # Test AveragingIterator with a simple example
    class SimpleAveraging(AveragingIterator[np.ndarray]):
        """Simple averaging: converge to mean of operator outputs."""

        def __init__(self):
            super().__init__(
                schedule="harmonic",
                max_iterations=100,
                tolerance=1e-6,
            )
            self.target = 1.0
            self._call_count = 0

        def forward_step(self, state: np.ndarray) -> np.ndarray:
            # Operator that outputs target + noise
            self._call_count += 1
            noise = 0.1 * np.sin(self._call_count)  # Deterministic "noise"
            return np.array([self.target + noise])

        def apply_averaging(self, old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
            return (1 - alpha) * old + alpha * new

        def compute_error(self, old: np.ndarray, new: np.ndarray) -> dict[str, float]:
            return {"x_rel": float(np.abs(new - old) / (np.abs(old) + 1e-10))}

        def initialize_state(self) -> np.ndarray:
            return np.array([0.0])

    iterator = SimpleAveraging()
    result = iterator.solve(max_iterations=50)
    final_state = iterator.get_state()

    # Should converge close to target despite noise
    error_from_target = abs(final_state[0] - 1.0)
    print(f"  Final state: {final_state[0]:.4f}, error from target: {error_from_target:.4f}")
    print(f"  Learning rates: {iterator.learning_rate_history[:5]} ... {iterator.learning_rate_history[-3:]}")

    # Verify learning rate schedule
    assert abs(iterator.learning_rate_history[0] - 1.0) < 1e-10, "First LR should be 1.0"
    assert abs(iterator.learning_rate_history[1] - 0.5) < 1e-10, "Second LR should be 0.5"
    print("  Learning rate schedule verified")

    print("All smoke tests passed!")
