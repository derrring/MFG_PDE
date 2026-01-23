"""
Picard (Fixed-Point) Iteration Pattern.

Abstract pattern for fixed-point iteration:
    x_{n+1} = (1 - alpha) * x_n + alpha * T(x_n)

where T is a forward operator and alpha is a constant damping factor.

This pattern applies to:
- MFG: HJB-FP coupling with Picard iteration
- RL: Policy Iteration
- Optimization: Proximal point methods
- Numerical: Nonlinear equation solving

Implementations inherit from FixedPointIteratorBase and implement
the forward_step() method for their specific problem type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, runtime_checkable

from ..convergence import ConvergenceResult, ConvergenceTracker, check_convergence

if TYPE_CHECKING:
    from collections.abc import Callable

# Generic type for iteration state
StateT = TypeVar("StateT")


@runtime_checkable
class PicardPattern(Protocol[StateT]):
    """
    Protocol for Picard (fixed-point) iteration.

    Defines the interface that all Picard-style iterators must implement.
    This is the minimal contract for fixed-point iteration.

    Type parameter StateT represents the state being iterated
    (e.g., tuple[NDArray, NDArray] for MFG U, M).
    """

    def forward_step(self, state: StateT) -> StateT:
        """
        Compute one forward step: T(x_n).

        This is the core operator being iterated. For MFG, this would be:
        1. Solve HJB backward given M -> U_new
        2. Solve FP forward given U_new -> M_new
        3. Return (U_new, M_new)

        Args:
            state: Current iteration state

        Returns:
            New state from forward operator
        """
        ...

    def apply_damping(self, old_state: StateT, new_state: StateT, alpha: float) -> StateT:
        """
        Apply damping to update.

        Formula: result = (1 - alpha) * old + alpha * new

        Args:
            old_state: Previous state
            new_state: State from forward_step
            alpha: Damping parameter in (0, 1]

        Returns:
            Damped state
        """
        ...

    def compute_error(self, old_state: StateT, new_state: StateT) -> dict[str, float]:
        """
        Compute error metrics between states.

        Args:
            old_state: Previous state
            new_state: Current state

        Returns:
            Dict of error metrics (e.g., {"U_rel": 0.01, "M_rel": 0.02})
        """
        ...


class FixedPointIteratorBase(ABC, Generic[StateT]):
    """
    Base class for fixed-point (Picard) iterators.

    Provides the iteration loop structure with:
    - Configurable damping
    - Convergence tracking
    - Progress reporting hooks
    - Warm start support

    Subclasses implement:
    - forward_step(): The specific forward operator
    - apply_damping(): How to combine old and new states
    - compute_error(): How to measure convergence
    - initialize_state(): Create initial state

    Example:
        >>> class MyIterator(FixedPointIteratorBase[tuple[NDArray, NDArray]]):
        ...     def forward_step(self, state):
        ...         U, M = state
        ...         U_new = solve_hjb(M)
        ...         M_new = solve_fp(U_new)
        ...         return (U_new, M_new)
        ...
        ...     # ... implement other abstract methods
        ...
        >>> iterator = MyIterator(damping=0.5)
        >>> result = iterator.solve(max_iterations=100, tolerance=1e-6)
    """

    def __init__(
        self,
        damping: float = 0.5,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
    ):
        """
        Initialize fixed-point iterator.

        Args:
            damping: Damping parameter alpha in (0, 1]
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            verbose: Whether to show progress
        """
        self.damping = damping
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose

        # State management
        self._state: StateT | None = None
        self._warm_start: StateT | None = None

        # Convergence tracking
        self._tracker: ConvergenceTracker | None = None
        self.iterations_run = 0

    @abstractmethod
    def forward_step(self, state: StateT) -> StateT:
        """
        Compute one forward step: T(x_n).

        Must be implemented by subclasses for their specific problem.
        """
        ...

    @abstractmethod
    def apply_damping(self, old_state: StateT, new_state: StateT, alpha: float) -> StateT:
        """
        Apply damping to combine old and new states.

        Default implementation assumes StateT supports arithmetic operations.
        Override for custom state types.
        """
        ...

    @abstractmethod
    def compute_error(self, old_state: StateT, new_state: StateT) -> dict[str, float]:
        """
        Compute error metrics for convergence checking.

        Returns dict with keys like "U_rel", "M_rel", "U_abs", "M_abs".
        """
        ...

    @abstractmethod
    def initialize_state(self) -> StateT:
        """
        Create initial state for iteration.

        Called when no warm start is provided.
        """
        ...

    def preserve_constraints(self, state: StateT) -> StateT:
        """
        Preserve any constraints after damping.

        Override to enforce constraints like:
        - Boundary conditions (PDE)
        - Positivity (probability distributions)
        - Normalization (mass conservation)

        Default: no-op (return state unchanged)
        """
        return state

    def on_iteration_start(self, iteration: int, state: StateT) -> None:
        """Hook called at start of each iteration."""
        pass

    def on_iteration_end(
        self,
        iteration: int,
        old_state: StateT,
        new_state: StateT,
        errors: dict[str, float],
    ) -> None:
        """Hook called at end of each iteration."""
        pass

    def solve(
        self,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        damping: float | None = None,
    ) -> ConvergenceResult:
        """
        Run fixed-point iteration until convergence.

        Args:
            max_iterations: Override max iterations
            tolerance: Override tolerance
            damping: Override damping parameter

        Returns:
            ConvergenceResult with final status
        """
        max_iter = max_iterations or self.max_iterations
        tol = tolerance or self.tolerance
        alpha = damping or self.damping

        # Initialize state
        if self._warm_start is not None:
            state = self._warm_start
        else:
            state = self.initialize_state()

        self._state = state

        # Initialize tracker
        self._tracker = ConvergenceTracker(max_iter)

        # Main iteration loop
        converged = False
        final_result = ConvergenceResult(
            converged=False,
            reason="Maximum iterations reached",
            relative_error=1.0,
            absolute_error=1.0,
        )

        for iteration in range(max_iter):
            self.on_iteration_start(iteration, state)

            old_state = state

            # Forward step
            new_state = self.forward_step(state)

            # Apply damping
            state = self.apply_damping(old_state, new_state, alpha)

            # Preserve constraints
            state = self.preserve_constraints(state)

            self._state = state

            # Compute errors
            errors = self.compute_error(old_state, state)

            # Separate relative and absolute errors
            rel_errors = {k: v for k, v in errors.items() if "_rel" in k or not ("_abs" in k)}
            abs_errors = {k: v for k, v in errors.items() if "_abs" in k}

            # If no explicit separation, treat all as relative
            if not abs_errors:
                abs_errors = rel_errors

            # Record in tracker
            self._tracker.record(iteration, rel_errors, abs_errors)

            self.iterations_run = iteration + 1

            self.on_iteration_end(iteration, old_state, state, errors)

            # Check convergence
            result = check_convergence(rel_errors, abs_errors, tol)
            if result.converged:
                final_result = result
                converged = True
                break

            final_result = result

        return final_result

    def get_state(self) -> StateT:
        """Get current state."""
        if self._state is None:
            raise RuntimeError("No state available. Call solve() first.")
        return self._state

    def set_warm_start(self, state: StateT) -> None:
        """Set warm start state."""
        self._warm_start = state

    def clear_warm_start(self) -> None:
        """Clear warm start state."""
        self._warm_start = None

    def get_convergence_history(self) -> dict[str, list[float]]:
        """Get convergence history."""
        if self._tracker is None:
            return {}
        return self._tracker.get_history()


if __name__ == "__main__":
    """Quick smoke test for development."""
    import numpy as np

    print("Testing PicardPattern and FixedPointIteratorBase...")

    # Test that Protocol is properly defined
    assert PicardPattern is not None
    print("  PicardPattern protocol defined")

    # Test FixedPointIteratorBase with a simple example
    class SimpleIterator(FixedPointIteratorBase[np.ndarray]):
        """Simple contraction mapping: T(x) = 0.5 * x + 0.25"""

        def __init__(self):
            super().__init__(damping=1.0, max_iterations=100, tolerance=1e-8)
            self.target = 0.5  # Fixed point of T(x) = 0.5x + 0.25

        def forward_step(self, state: np.ndarray) -> np.ndarray:
            return 0.5 * state + 0.25

        def apply_damping(self, old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
            return alpha * new + (1 - alpha) * old

        def compute_error(self, old: np.ndarray, new: np.ndarray) -> dict[str, float]:
            rel = float(np.abs(new - old) / (np.abs(old) + 1e-10))
            return {"x_rel": rel, "x_abs": float(np.abs(new - old))}

        def initialize_state(self) -> np.ndarray:
            return np.array([0.0])

    iterator = SimpleIterator()
    result = iterator.solve()
    final_state = iterator.get_state()

    assert result.converged, f"Should converge, got: {result.reason}"
    assert np.abs(final_state[0] - 0.5) < 1e-6, f"Should converge to 0.5, got {final_state[0]}"
    print(f"  SimpleIterator converged to {final_state[0]:.6f} in {iterator.iterations_run} iterations")

    print("All smoke tests passed!")
