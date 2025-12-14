"""
Base Solver with Hooks Support

This module provides the base class for all MFG solvers using the new
clean interface design with hooks pattern support.

Convergence Integration (Issue #456):
    Solvers can now accept optional `convergence` parameter with a
    `ConvergenceConfig` object for fine-grained control over convergence
    criteria. The base class provides default convergence checking, but
    subclasses can override `_create_convergence_checker()` to use
    specialized checkers (HJB, FP, MFG).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from mfg_pde.types import ConvergenceInfo, MFGProblem, MFGResult, SpatialTemporalState
from mfg_pde.utils.convergence import ConvergenceConfig, MFGConvergenceChecker

if TYPE_CHECKING:
    import numpy as np

    from mfg_pde.hooks import SolverHooks
    from mfg_pde.utils.convergence import ConvergenceChecker


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

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        convergence: ConvergenceConfig | None = None,
        **config,
    ):
        """
        Initialize base solver.

        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            convergence: Optional ConvergenceConfig for fine-grained control.
                If provided, max_iterations and tolerance are overridden.
            **config: Additional solver-specific configuration
        """
        # Handle ConvergenceConfig if provided
        if convergence is not None:
            self.convergence_config = convergence
            self.max_iterations = convergence.max_iterations
            self.tolerance = convergence.tolerance
        else:
            self.convergence_config = ConvergenceConfig(
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
            self.max_iterations = max_iterations
            self.tolerance = tolerance

        self.config = config

        # Statistics tracking
        self._start_time: float | None = None
        self._iteration_times: list[float] = []
        self._convergence_checker: ConvergenceChecker | None = None

    def solve(
        self,
        problem: MFGProblem,
        hooks: SolverHooks | None = None,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        convergence: ConvergenceConfig | None = None,
    ) -> MFGResult:
        """
        Solve the MFG problem with optional hooks.

        Args:
            problem: MFG problem instance
            hooks: Optional hooks for customization
            max_iterations: Override default max iterations
            tolerance: Override default tolerance
            convergence: Override convergence configuration

        Returns:
            Solution result
        """
        # Determine convergence config to use
        if convergence is not None:
            conv_config = convergence
        else:
            conv_config = self.convergence_config

        # Override with explicit parameters if provided
        max_iter = max_iterations or conv_config.max_iterations
        tol = tolerance or conv_config.tolerance
        min_iter = conv_config.min_iterations

        # Create convergence checker
        self._convergence_checker = self._create_convergence_checker(conv_config)

        # Initialize timing and statistics
        self._start_time = time.time()
        self._iteration_times = []
        residual_history: list[float] = []
        convergence_metrics_history: list[dict[str, float]] = []

        # Initialize solver state
        state = self._initialize_state(problem)
        prev_state = state  # Track previous state for checker
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
                prev_state = state
                residual_history = [state.residual]
                convergence_metrics_history = []
                convergence_reason = "restarted_by_hook"
                continue

            # Check convergence using Protocol-based checker
            prev_state = state
            state = new_state

            # Skip convergence check during min_iterations
            if iteration < min_iter:
                continue

            # Use the checker if available, otherwise fall back to simple check
            if self._convergence_checker is not None:
                old_state_dict = self._map_state_to_dict(prev_state)
                new_state_dict = self._map_state_to_dict(state)
                converged, metrics = self._convergence_checker.check(old_state_dict, new_state_dict)
                convergence_metrics_history.append(metrics)

                # Check for divergence
                if metrics.get("status", "OK") != "OK":
                    convergence_reason = f"diverged: {metrics['status']}"
                    break
            else:
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
        avg_iteration_time = total_time / len(self._iteration_times) if self._iteration_times else 0.0

        convergence_info = ConvergenceInfo(
            converged=converged,
            iterations=state.iteration + 1,
            final_residual=state.residual,
            residual_history=residual_history,
            convergence_reason=convergence_reason,
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

    def _create_convergence_checker(self, config: ConvergenceConfig) -> ConvergenceChecker:
        """
        Create a convergence checker for this solver.

        Default implementation creates an MFGConvergenceChecker.
        Subclasses can override to use specialized checkers:
        - HJBConvergenceChecker for standalone HJB solvers
        - FPConvergenceChecker for standalone FP solvers
        - MFGConvergenceChecker for coupled MFG solvers

        Args:
            config: Convergence configuration

        Returns:
            Appropriate ConvergenceChecker instance
        """
        return MFGConvergenceChecker(config)

    def _map_state_to_dict(self, state: SpatialTemporalState) -> dict[str, np.ndarray]:
        """
        Map solver state to dictionary format for convergence checker.

        The convergence checker protocol expects states as dictionaries
        with 'U' and 'M' keys. Subclasses can override to customize
        the mapping (e.g., for different state representations).

        Args:
            state: Solver state

        Returns:
            Dictionary with 'U' and 'M' arrays
        """
        return {"U": state.u, "M": state.m}

    @abstractmethod
    def _create_result(
        self,
        state: SpatialTemporalState,
        problem: MFGProblem,
        convergence_info: ConvergenceInfo,
        total_time: float,
        avg_iteration_time: float,
    ) -> MFGResult:
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

    def get_solver_info(self) -> dict[str, Any]:
        """Get information about this solver."""
        return {
            "solver_type": self.__class__.__name__,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "convergence_config": {
                "tolerance": self.convergence_config.tolerance,
                "tolerance_U": self.convergence_config.get_tolerance_U(),
                "tolerance_M": self.convergence_config.get_tolerance_M(),
                "norm": self.convergence_config.norm,
                "relative": self.convergence_config.relative,
                "min_iterations": self.convergence_config.min_iterations,
                "require_both": self.convergence_config.require_both,
            },
            "config": self.config.copy(),
        }
