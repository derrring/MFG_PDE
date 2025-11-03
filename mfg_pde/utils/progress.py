#!/usr/bin/env python3
"""
Progress Monitoring Utilities for MFG_PDE

Provides elegant progress bars, timing, and performance monitoring for long-running
solver operations using modern tools like rich (preferred) or tqdm (fallback).
"""

from __future__ import annotations

import functools
import time
import warnings
from contextlib import contextmanager
from typing import Any

# Try rich first (preferred), then tqdm, then fallback
PROGRESS_BACKEND = None

try:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

    PROGRESS_BACKEND = "rich"
    RICH_AVAILABLE = True
    TQDM_AVAILABLE = False  # Rich is preferred, so tqdm not used
    console = Console()

except ImportError:
    RICH_AVAILABLE = False
    console = None

    try:
        from tqdm import tqdm as _tqdm_real
        from tqdm import trange as _trange_real

        PROGRESS_BACKEND = "tqdm"
        TQDM_AVAILABLE = True
        tqdm = _tqdm_real
        trange = _trange_real
    except ImportError:
        PROGRESS_BACKEND = "fallback"
        TQDM_AVAILABLE = False


# Rich-based progress bar wrapper
class RichProgressBar:
    """Rich progress bar that mimics tqdm interface."""

    def __init__(self, iterable=None, total=None, desc=None, disable=False, **kwargs):
        self.iterable = iterable
        self.total = total or (len(iterable) if iterable and hasattr(iterable, "__len__") else None)
        self.desc = desc or ""
        self.disable = disable
        self.n = 0
        self.progress = None
        self.task_id = None
        self.postfix_data = {}

    def __iter__(self):
        if self.iterable is None:
            return iter([])
        for item in self.iterable:
            yield item
            self.update(1)

    def __enter__(self):
        if not self.disable and RICH_AVAILABLE:
            self.progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console,
            )
            self.progress.__enter__()
            self.task_id = self.progress.add_task(self.desc, total=self.total or 100)
        return self

    def __exit__(self, *args):
        if self.progress:
            self.progress.__exit__(*args)

    def update(self, n=1):
        self.n += n
        if not self.disable and self.progress and self.task_id is not None:
            # Update with postfix if available
            desc = self.desc
            if self.postfix_data:
                postfix_str = ", ".join(f"{k}={v}" for k, v in self.postfix_data.items())
                desc = f"{self.desc} [{postfix_str}]"
            self.progress.update(self.task_id, advance=n, description=desc)

    def set_postfix(self, **kwargs):
        self.postfix_data.update(kwargs)
        if not self.disable and self.progress and self.task_id is not None:
            postfix_str = ", ".join(f"{k}={v}" for k, v in self.postfix_data.items())
            self.progress.update(self.task_id, description=f"{self.desc} [{postfix_str}]")

    def set_description(self, desc: str):
        self.desc = desc
        if not self.disable and self.progress and self.task_id is not None:
            self.progress.update(self.task_id, description=desc)

    def close(self):
        if self.progress:
            self.progress.stop()


# Unified interface
if PROGRESS_BACKEND == "rich":
    tqdm = RichProgressBar

    def trange(n: int, **kwargs) -> RichProgressBar:
        return RichProgressBar(range(n), total=n, **kwargs)

elif PROGRESS_BACKEND == "tqdm":
    # tqdm already imported above
    pass

else:
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

    def trange(n: int, **kwargs: Any) -> tqdm:  # type: ignore[name-defined]
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
            self.pbar.close()

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
                self.pbar.set_postfix(**postfix)

    def set_description(self, desc: str):
        """Update the progress bar description."""
        if self.pbar and hasattr(self.pbar, "set_description"):
            self.pbar.set_description(desc)


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


def check_progress_backend() -> str:
    """
    Check which progress backend is being used.

    Returns:
        Backend name: "rich", "tqdm", or "fallback"
    """
    if PROGRESS_BACKEND == "fallback":
        warnings.warn(
            "Neither rich nor tqdm is available. For enhanced progress bars, install with:\n"
            "  pip install rich  (recommended, modern)\n"
            "  pip install tqdm  (alternative)\n"
            "Falling back to basic progress indication.",
            UserWarning,
            stacklevel=2,
        )
    return PROGRESS_BACKEND


def check_tqdm_availability() -> bool:
    """
    Check if tqdm is available (deprecated: use check_progress_backend instead).

    Returns:
        True if any progress backend is available
    """
    warnings.warn(
        "check_tqdm_availability() is deprecated. Use check_progress_backend() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return PROGRESS_BACKEND in ("rich", "tqdm")


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
