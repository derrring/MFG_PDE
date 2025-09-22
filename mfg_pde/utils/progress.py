#!/usr/bin/env python3
"""
Progress Monitoring Utilities for MFG_PDE

Provides elegant progress bars, timing, and performance monitoring for long-running
solver operations using modern tools like tqdm.
"""

from __future__ import annotations

import functools
import time
import warnings
from contextlib import contextmanager
from typing import Any

try:
    from tqdm import tqdm as _tqdm_real
    from tqdm import trange as _trange_real

    TQDM_AVAILABLE = True
    tqdm = _tqdm_real  # type: ignore
    trange = _trange_real  # type: ignore
except ImportError:
    TQDM_AVAILABLE = False

    # Fallback simple progress implementation
    class tqdm:  # noqa: N801
        def __init__(self, iterable=None, total=None, desc=None, disable=False, **kwargs):
            self.iterable = iterable or range(total or 0)
            self.total = total or (len(iterable) if iterable else 0)
            self.desc = desc or ""
            self.disable = disable
            self.n = 0

        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)

        def __enter__(self):
            if not self.disable:
                print(f"Starting {self.desc}..." if self.desc else "Starting...")
            return self

        def __exit__(self, *args: Any) -> None:
            if not self.disable:
                print(f"Completed {self.desc}!" if self.desc else "Completed!")

        def update(self, n=1):
            self.n += n
            if not self.disable and self.total > 0:
                progress = (self.n / self.total) * 100
                if self.n % max(1, self.total // 10) == 0:  # Update every 10%
                    print(f"Progress: {progress:.1f}% ({self.n}/{self.total})")

        def set_postfix(self, **kwargs: Any) -> None:
            pass  # Simple fallback ignores postfix

        def close(self) -> None:
            pass  # Simple fallback ignores close

        def set_description(self, desc: str) -> None:
            self.desc = desc

    def trange(n: int, **kwargs: Any) -> tqdm:
        return tqdm(range(n), **kwargs)


class SolverTimer:
    """
    Context manager for timing solver operations with detailed statistics.
    """

    def __init__(self, description: str = "Operation", verbose: bool = True):
        self.description = description
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        if self.verbose:
            print(f"Starting {self.description}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - (self.start_time or 0)

        if self.verbose:
            if exc_type is None:
                print(f"SUCCESS: {self.description} completed in {self.format_duration()}")
            else:
                print(f"ERROR: {self.description} failed after {self.format_duration()}")

    def format_duration(self) -> str:
        """Format duration in human-readable form."""
        if self.duration is None:
            return "Unknown"

        if self.duration < 1:
            return f"{self.duration * 1000:.1f}ms"
        elif self.duration < 60:
            return f"{self.duration:.2f}s"
        elif self.duration < 3600:
            minutes = int(self.duration // 60)
            seconds = self.duration % 60
            return f"{minutes}m {seconds:.1f}s"
        else:
            hours = int(self.duration // 3600)
            minutes = int((self.duration % 3600) // 60)
            seconds = self.duration % 60
            return f"{hours}h {minutes}m {seconds:.1f}s"


class IterationProgress:
    """
    Advanced progress tracking for iterative solvers with convergence monitoring.
    """

    def __init__(
        self,
        max_iterations: int,
        description: str = "Solver Iterations",
        show_rate: bool = True,
        show_eta: bool = True,
        update_frequency: int = 1,
        disable: bool = False,
    ):
        self.max_iterations = max_iterations
        self.description = description
        self.show_rate = show_rate
        self.show_eta = show_eta
        self.update_frequency = update_frequency
        self.disable = disable

        # Progress bar configuration
        self.pbar = None
        self.start_time = None
        self.current_iteration = 0

    def __enter__(self):
        if self.disable:
            return self

        self.start_time = time.perf_counter()

        # Configure progress bar
        bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"
        if self.show_rate:
            bar_format += " [{rate_fmt}"
        if self.show_eta:
            bar_format += ", {remaining}"
        if self.show_rate or self.show_eta:
            bar_format += "]"

        self.pbar = tqdm(
            total=self.max_iterations,
            desc=self.description,
            bar_format=bar_format,
            disable=self.disable,
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar and hasattr(self.pbar, "close"):
            self.pbar.close()  # type: ignore

    def update(
        self,
        n: int = 1,
        error: float | None = None,
        additional_info: dict[str, Any] | None = None,
    ):
        """
        Update progress with optional convergence information.

        Args:
            n: Number of iterations to advance
            error: Current error/residual value
            additional_info: Additional metrics to display
        """
        if self.disable or not self.pbar:
            return

        self.current_iteration += n

        # Update every update_frequency iterations or on final iteration
        if self.current_iteration % self.update_frequency == 0 or self.current_iteration >= self.max_iterations:
            # Prepare postfix information
            postfix = {}
            if error is not None:
                postfix["error"] = f"{error:.2e}"
            if additional_info:
                postfix.update(additional_info)

            self.pbar.update(n)
            if postfix and hasattr(self.pbar, "set_postfix"):
                self.pbar.set_postfix(postfix)  # type: ignore

    def set_description(self, desc: str):
        """Update the progress bar description."""
        if self.pbar and hasattr(self.pbar, "set_description"):
            self.pbar.set_description(desc)  # type: ignore


def timed_operation(description: str | None = None, verbose: bool = True):
    """
    Decorator for timing function calls.

    Args:
        description: Custom description for the operation
        verbose: Whether to print timing information
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            desc = description or f"Function '{func.__name__}'"

            with SolverTimer(desc, verbose=verbose) as timer:
                result = func(*args, **kwargs)

            # Add timing info to result if it's a dictionary
            if isinstance(result, dict) and "execution_time" not in result:
                result["execution_time"] = timer.duration

            return result

        return wrapper

    return decorator


@contextmanager
def progress_context(
    iterable: Any,
    description: str = "Processing",
    show_rate: bool = True,
    disable: bool = False,
):
    """
    Context manager for easy progress tracking of iterables.

    Args:
        iterable: Iterable to track progress for
        description: Description to show
        show_rate: Whether to show processing rate
        disable: Whether to disable progress bar

    Yields:
        tqdm progress bar wrapped iterable
    """
    try:
        # Determine total if possible
        try:
            total = len(iterable) if hasattr(iterable, "__len__") else None
        except (TypeError, AttributeError):
            total = None

        with tqdm(
            iterable,
            desc=description,
            total=total,
            disable=disable,
            unit="it",
            unit_scale=show_rate,
        ) as pbar:
            yield pbar

    except Exception as e:
        if not disable:
            print(f"Progress tracking error: {e}")
        # Fallback to plain iteration
        yield iterable


def check_tqdm_availability() -> bool:
    """Check if tqdm is available and recommend installation if not."""
    if not TQDM_AVAILABLE:
        warnings.warn(
            "tqdm is not available. For enhanced progress bars, install with: "
            "pip install tqdm\n"
            "Falling back to basic progress indication.",
            UserWarning,
            stacklevel=2,
        )
    return TQDM_AVAILABLE


# Convenience functions for common patterns


def solver_progress(max_iterations: int, description: str = "Solver Progress", **kwargs) -> IterationProgress:
    """
    Create progress tracker optimized for solver iterations.

    Args:
        max_iterations: Maximum number of iterations
        description: Progress description
        **kwargs: Additional arguments for IterationProgress

    Returns:
        Configured IterationProgress instance
    """
    return IterationProgress(
        max_iterations=max_iterations,
        description=description,
        show_rate=True,
        show_eta=True,
        update_frequency=max(1, max_iterations // 100),  # Update every 1%
        **kwargs,
    )


def time_solver_operation(func):
    """
    Decorator specifically for timing solver operations.

    Automatically adds execution time to solver results.
    """
    return timed_operation(description=f"Solver operation '{func.__name__}'", verbose=True)(func)


# Example usage demonstrations
if __name__ == "__main__":
    import numpy as np

    print("🧪 Testing MFG_PDE Progress Utilities")
    print("=" * 50)

    # Test basic timing
    with SolverTimer("Matrix multiplication", verbose=True):
        time.sleep(0.1)  # Simulate work
        result = np.random.rand(100, 100) @ np.random.rand(100, 100)

    # Test iteration progress
    print("\nTesting iteration progress:")
    with solver_progress(20, "Sample Solver") as progress:
        for i in range(20):
            time.sleep(0.02)  # Simulate solver iteration
            error = 1.0 / (i + 1)  # Decreasing error
            progress.update(1, error=error, additional_info={"iteration": i + 1})

    # Test timed decorator
    @time_solver_operation
    def sample_computation():
        time.sleep(0.05)
        return {"result": 42, "converged": True}

    print("\nTesting timed decorator:")
    result = sample_computation()
    print(f"Result: {result}")

    print("\nProgress utilities test completed!")
