"""
Base Solver with Hooks Support

This module provides the base class for all MFG solvers using the new
clean interface design with hooks pattern support.
"""

from __future__ import annotations


from abc import ABC, abstractmethod
from typing import Any
import time

from ..types import MFGProblem, MFGResult, SpatialTemporalState, ConvergenceInfo
from ..hooks import SolverHooks


class BaseSolver(ABC):
    """
    Base class for all MFG solvers with hooks support.

    This class provides the common infrastructure for solver iteration,
    convergence checking, and hooks integration. Subclasses only need
    to implement the core algorithmic steps.

    Example:
        class MyCustomSolver(BaseSolver):
            def _initialize_state(self, problem):
                # Initialize solver state
                return SpatialTemporalState(...)

            def _iteration_step(self, state, problem):
                # Perform one iteration
                return updated_state
    """

    def __init__(self,
                 max_iterations: int = 100,
                 tolerance: float = 1e-6,
                 **config):
        """
        Initialize base solver.

        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            **config: Additional solver-specific configuration
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.config = config

        # Statistics tracking
        self._start_time = None
        self._iteration_times = []

    def solve(self,
              problem: MFGProblem,
              hooks: SolverHooks | None = None,
              max_iterations: int | None = None,
              tolerance: float | None = None) -> MFGResult:
        """
        Solve the MFG problem with optional hooks.

        Args:
            problem: MFG problem instance
            hooks: Optional hooks for customization
            max_iterations: Override default max iterations
            tolerance: Override default tolerance

        Returns:
            Solution result
        """
        # Use provided parameters or defaults
        max_iter = max_iterations or self.max_iterations
        tol = tolerance or self.tolerance

        # Initialize timing and statistics
        self._start_time = time.time()
        self._iteration_times = []
        residual_history = []

        # Initialize solver state
        state = self._initialize_state(problem)
        residual_history.append(state.residual)

        # Call solve start hook
        if hooks:
            hooks.on_solve_start(state)

        # Main iteration loop
        converged = False
        convergence_reason = "maximum_iterations_reached"

        for iteration in range(max_iter):
            iteration_start_time = time.time()

            # Update iteration counter
            state = state.copy_with_updates(iteration=iteration)

            # Pre-iteration hook
            if hooks:
                hooks.on_iteration_start(state)

            # Perform one iteration step
            try:
                new_state = self._iteration_step(state, problem)
            except Exception as e:
                convergence_reason = f"iteration_failed: {e}"
                break

            # Record timing
            iteration_time = time.time() - iteration_start_time
            self._iteration_times.append(iteration_time)

            # Update residual history
            residual_history.append(new_state.residual)

            # Post-iteration hook with control flow
            control_signal = None
            if hooks:
                control_signal = hooks.on_iteration_end(new_state)

            # Handle control signals
            if control_signal == "stop":
                convergence_reason = "stopped_by_hook"
                state = new_state
                break
            elif control_signal == "restart":
                # Restart with fresh initial conditions
                state = self._initialize_state(problem)
                residual_history = [state.residual]
                convergence_reason = "restarted_by_hook"
                continue

            # Check convergence
            state = new_state
            converged = self._check_convergence(state, tol)

            # Allow hooks to override convergence decision
            if hooks:
                hook_convergence = hooks.on_convergence_check(state)
                if hook_convergence is not None:
                    converged = hook_convergence

            if converged:
                convergence_reason = "tolerance_achieved"
                break

        # Create convergence info
        total_time = time.time() - self._start_time
        avg_iteration_time = (total_time / len(self._iteration_times)
                             if self._iteration_times else 0.0)

        convergence_info = ConvergenceInfo(
            converged=converged,
            iterations=state.iteration + 1,
            final_residual=state.residual,
            residual_history=residual_history,
            convergence_reason=convergence_reason
        )

        # Create result
        result = self._create_result(state, problem, convergence_info, total_time, avg_iteration_time)

        # Final hook
        if hooks:
            result = hooks.on_solve_end(result)

        return result

    @abstractmethod
    def _initialize_state(self, problem: MFGProblem) -> SpatialTemporalState:
        """
        Initialize solver state from problem.

        Args:
            problem: MFG problem instance

        Returns:
            Initial solver state
        """
        pass

    @abstractmethod
    def _iteration_step(self, state: SpatialTemporalState, problem: MFGProblem) -> SpatialTemporalState:
        """
        Perform one iteration step.

        Args:
            state: Current solver state
            problem: MFG problem instance

        Returns:
            Updated solver state
        """
        pass

    def _check_convergence(self, state: SpatialTemporalState, tolerance: float) -> bool:
        """
        Check if solver has converged.

        Default implementation checks residual against tolerance.
        Subclasses can override for custom convergence criteria.

        Args:
            state: Current solver state
            tolerance: Convergence tolerance

        Returns:
            True if converged, False otherwise
        """
        return state.residual < tolerance

    @abstractmethod
    def _create_result(self,
                      state: SpatialTemporalState,
                      problem: MFGProblem,
                      convergence_info: ConvergenceInfo,
                      total_time: float,
                      avg_iteration_time: float) -> MFGResult:
        """
        Create final result object.

        Args:
            state: Final solver state
            problem: MFG problem instance
            convergence_info: Convergence information
            total_time: Total solving time
            avg_iteration_time: Average iteration time

        Returns:
            Final result object
        """
        pass

    def get_solver_info(self) -> dict[str, Any]:
        """Get information about this solver."""
        return {
            'solver_type': self.__class__.__name__,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance,
            'config': self.config.copy()
        }