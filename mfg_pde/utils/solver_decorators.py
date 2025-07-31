#!/usr/bin/env python3
"""
Solver Enhancement Decorators

Provides decorators for adding progress monitoring, timing, and other
modern features to MFG solver classes.
"""

import functools
import time
from typing import Any, Dict, Optional, Union

from .progress import IterationProgress, SolverTimer, time_solver_operation


def with_progress_monitoring(
    show_progress: bool = True,
    show_timing: bool = True,
    update_frequency: Optional[int] = None,
):
    """
    Decorator to add progress monitoring to solver methods.

    Args:
        show_progress: Whether to show progress bars
        show_timing: Whether to show timing information
        update_frequency: How often to update progress (None for auto)
    """

    def decorator(solve_method):
        @functools.wraps(solve_method)
        def wrapper(self, *args, **kwargs):
            # Extract parameters for progress monitoring
            max_iterations = None
            verbose = kwargs.get("verbose", True)

            # Try to determine max iterations from various parameter names
            for param_name in [
                "max_iterations",
                "Niter",
                "max_picard_iterations",
                "Niter_max",
            ]:
                if param_name in kwargs:
                    max_iterations = kwargs[param_name]
                    break
                elif hasattr(self, param_name):
                    max_iterations = getattr(self, param_name)
                    break
                elif hasattr(self, "config") and hasattr(self.config, "picard"):
                    max_iterations = self.config.picard.max_iterations
                    break

            # Default to reasonable fallback
            if max_iterations is None:
                max_iterations = 50

            # Disable progress if verbose is False or explicitly disabled
            disable_progress = not (show_progress and verbose)

            # Determine solver name for progress description
            solver_name = getattr(self, "__class__").__name__
            description = f"{solver_name} Progress"

            # Time the entire operation if requested
            timer_context = (
                SolverTimer(f"{solver_name} Solve", verbose=show_timing and verbose)
                if show_timing
                else None
            )

            try:
                if timer_context:
                    timer_context.__enter__()

                # Add progress tracking to kwargs
                if not disable_progress and max_iterations > 1:
                    freq = update_frequency or max(1, max_iterations // 20)
                    kwargs["_progress_tracker"] = IterationProgress(
                        max_iterations=max_iterations,
                        description=description,
                        update_frequency=freq,
                        disable=disable_progress,
                    )

                # Call the original solve method
                result = solve_method(self, *args, **kwargs)

                # Add timing information to result if it's structured
                if (
                    timer_context
                    and isinstance(result, (dict, tuple))
                    and hasattr(result, "_asdict")
                ):
                    # For named tuples or dataclasses
                    if hasattr(result, "metadata"):
                        result.metadata["execution_time"] = timer_context.duration
                elif timer_context and isinstance(result, dict):
                    result["execution_time"] = timer_context.duration

                return result

            except Exception as e:
                if timer_context:
                    timer_context.__exit__(type(e), e, e.__traceback__)
                raise
            finally:
                if timer_context:
                    timer_context.__exit__(None, None, None)

        return wrapper

    return decorator


def enhanced_solver_method(
    monitor_convergence: bool = True, auto_progress: bool = True, timing: bool = True
):
    """
    Comprehensive decorator for enhancing solver methods with modern features.

    Args:
        monitor_convergence: Add convergence monitoring
        auto_progress: Automatically add progress bars
        timing: Add timing information
    """

    def decorator(solve_method):
        @functools.wraps(solve_method)
        def wrapper(self, *args, **kwargs):
            # Get solver configuration
            verbose = kwargs.get("verbose", True)

            # Apply progress monitoring if enabled
            if auto_progress:
                enhanced_method = with_progress_monitoring(
                    show_progress=True, show_timing=timing
                )(solve_method)
                return enhanced_method(self, *args, **kwargs)
            elif timing:
                # Just add timing without progress bars
                enhanced_method = time_solver_operation(solve_method)
                return enhanced_method(self, *args, **kwargs)
            else:
                # No enhancements
                return solve_method(self, *args, **kwargs)

        return wrapper

    return decorator


class SolverProgressMixin:
    """
    Mixin class that adds progress monitoring capabilities to any solver.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress_enabled = True
        self._timing_enabled = True

    def enable_progress(self, enabled: bool = True):
        """Enable or disable progress monitoring."""
        self._progress_enabled = enabled

    def enable_timing(self, enabled: bool = True):
        """Enable or disable timing information."""
        self._timing_enabled = enabled

    def _should_show_progress(self, verbose: bool = True) -> bool:
        """Determine if progress should be shown."""
        return self._progress_enabled and verbose

    def _create_progress_tracker(
        self, max_iterations: int, description: str = None
    ) -> Optional[IterationProgress]:
        """Create a progress tracker if progress is enabled."""
        if not self._progress_enabled:
            return None

        desc = description or f"{self.__class__.__name__} Progress"
        return IterationProgress(
            max_iterations=max_iterations,
            description=desc,
            update_frequency=max(1, max_iterations // 20),
            disable=not self._progress_enabled,
        )


# Example of how to apply to existing solver classes
def upgrade_solver_with_progress(solver_class):
    """
    Class decorator to upgrade existing solver classes with progress monitoring.

    Args:
        solver_class: Solver class to upgrade

    Returns:
        Enhanced solver class with progress monitoring
    """

    # Add progress mixin
    class EnhancedSolver(SolverProgressMixin, solver_class):
        pass

    # Enhance the solve method if it exists
    if hasattr(EnhancedSolver, "solve"):
        original_solve = EnhancedSolver.solve
        EnhancedSolver.solve = enhanced_solver_method()(original_solve)

    # Preserve class name and documentation
    EnhancedSolver.__name__ = f"Enhanced{solver_class.__name__}"
    EnhancedSolver.__qualname__ = EnhancedSolver.__name__

    return EnhancedSolver


# Utility functions for manual progress integration


def update_solver_progress(
    progress_tracker: Optional[IterationProgress],
    iteration: int,
    error: Optional[float] = None,
    **additional_info,
):
    """
    Safely update progress tracker with solver information.

    Args:
        progress_tracker: Progress tracker instance (may be None)
        iteration: Current iteration number
        error: Current error/residual
        **additional_info: Additional metrics to display
    """
    if progress_tracker is not None:
        info = additional_info.copy()
        if error is not None:
            info["error"] = f"{error:.2e}"
        progress_tracker.update(1, error=error, additional_info=info)


def format_solver_summary(
    solver_name: str,
    iterations: int,
    final_error: Optional[float] = None,
    execution_time: Optional[float] = None,
    converged: bool = True,
) -> str:
    """
    Format a nice summary of solver results.

    Args:
        solver_name: Name of the solver
        iterations: Number of iterations completed
        final_error: Final error/residual
        execution_time: Total execution time
        converged: Whether the solver converged

    Returns:
        Formatted summary string
    """
    status = "SUCCESS: Converged" if converged else "WARNING:  Max iterations reached"

    summary = f"\n{'='*60}\n"
    summary += f"{solver_name} Summary\n"
    summary += f"{'='*60}\n"
    summary += f"{status} in {iterations} iterations\n"

    if final_error is not None:
        summary += f"Final error: {final_error:.2e}\n"

    if execution_time is not None:
        if execution_time < 1:
            summary += f"Execution time: {execution_time*1000:.1f}ms\n"
        elif execution_time < 60:
            summary += f"Execution time: {execution_time:.2f}s\n"
        else:
            minutes = int(execution_time // 60)
            seconds = execution_time % 60
            summary += f"Execution time: {minutes}m {seconds:.1f}s\n"

    summary += f"{'='*60}\n"

    return summary
