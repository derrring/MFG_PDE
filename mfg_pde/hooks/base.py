"""
Base Hooks System for MFG Solvers

This module provides the core hooks architecture that allows users
to customize solver behavior without complex inheritance.
"""

from __future__ import annotations


from typing import TYPE_CHECKING
from abc import ABC

if TYPE_CHECKING:
    # Avoid circular imports - only import types when type checking
    from ..types import SpatialTemporalState, MFGResult


class SolverHooks(ABC):
    """
    Base class for solver customization hooks.

    Override any method to customize solver behavior. All methods
    are optional - only implement what you need.

    Example:
        class MyHook(SolverHooks):
            def on_iteration_end(self, state):
                print(f"Iteration {state.iteration}: residual={state.residual}")

        solver.solve(problem, hooks=MyHook())
    """

    def on_solve_start(self, initial_state: SpatialTemporalState) -> None:
        """
        Called once at the beginning of the solve process.

        Use this for:
        - Initialization of custom data structures
        - Starting timers or logging
        - Saving initial conditions

        Args:
            initial_state: Initial solution state with u, m, and metadata
        """
        pass

    def on_iteration_start(self, state: SpatialTemporalState) -> None:
        """
        Called at the start of each solver iteration.

        Use this for:
        - Pre-iteration logging
        - Adaptive parameter updates
        - Custom preprocessing

        Args:
            state: Current solution state before iteration
        """
        pass

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """
        Called after each solver iteration completes.

        Use this for:
        - Progress monitoring and visualization
        - Custom convergence checks
        - Intermediate result saving
        - Algorithm state inspection

        Args:
            state: Updated solution state after iteration

        Returns:
            Control string to influence solver behavior:
            - None: Continue normally
            - "stop": Stop iteration early
            - "restart": Restart with modified conditions
        """
        pass

    def on_convergence_check(self, state: SpatialTemporalState) -> bool | None:
        """
        Called during convergence checking.

        Use this for:
        - Custom convergence criteria
        - Override default convergence logic

        Args:
            state: Current solution state

        Returns:
            Convergence decision:
            - None: Use default convergence check
            - True: Force convergence (stop iteration)
            - False: Force non-convergence (continue iteration)
        """
        pass

    def on_solve_end(self, result: MFGResult) -> MFGResult:
        """
        Called just before returning the final result.

        Use this for:
        - Post-processing of results
        - Adding custom metadata
        - Final visualization
        - Result validation

        Args:
            result: Final solver result

        Returns:
            Potentially modified result object
        """
        return result