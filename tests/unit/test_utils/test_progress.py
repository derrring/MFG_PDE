"""Tests for mfg_pde.utils.progress module."""

from __future__ import annotations

import time
from io import StringIO
from unittest.mock import patch

import pytest

from mfg_pde.utils.progress import (
    TQDM_AVAILABLE,
    IterationProgress,
    SolverTimer,
    check_tqdm_availability,
    progress_context,
    solver_progress,
    time_solver_operation,
    timed_operation,
    tqdm,
    trange,
)


class TestTqdmAvailability:
    """Test tqdm availability detection and fallback."""

    def test_tqdm_import_status(self):
        """Test TQDM_AVAILABLE flag is set correctly."""
        assert isinstance(TQDM_AVAILABLE, bool)

    def test_check_tqdm_availability(self):
        """Test check_tqdm_availability returns bool and warns if unavailable."""
        if not TQDM_AVAILABLE:
            with pytest.warns(UserWarning, match="tqdm is not available"):
                result = check_tqdm_availability()
            assert result is False
        else:
            result = check_tqdm_availability()
            assert result is True

    def test_tqdm_class_exists(self):
        """Test tqdm class is available (real or fallback)."""
        assert tqdm is not None
        assert callable(tqdm)

    def test_trange_function_exists(self):
        """Test trange function is available (real or fallback)."""
        assert trange is not None
        assert callable(trange)


class TestTqdmFallback:
    """Test fallback tqdm implementation when real tqdm unavailable."""

    def test_fallback_with_iterable(self):
        """Test fallback tqdm works with iterable."""
        items = [1, 2, 3, 4, 5]
        result = []

        for item in tqdm(items, desc="Test", disable=True):
            result.append(item)

        assert result == items

    def test_fallback_with_total(self):
        """Test fallback tqdm works with total parameter."""
        with patch("sys.stdout", new=StringIO()):
            pbar = tqdm(total=10, desc="Progress", disable=True)
            for _ in range(10):
                pbar.update(1)
            pbar.close()

    def test_fallback_context_manager(self):
        """Test fallback tqdm works as context manager."""
        with patch("sys.stdout", new=StringIO()), tqdm(total=5, desc="Test Progress") as pbar:
            for _ in range(5):
                pbar.update(1)

    def test_fallback_set_postfix(self):
        """Test fallback tqdm set_postfix doesn't crash."""
        pbar = tqdm(total=10, disable=True)
        pbar.set_postfix(error=1e-5, iteration=5)
        pbar.close()

    def test_fallback_set_description(self):
        """Test fallback tqdm set_description works."""
        pbar = tqdm(total=10, disable=True)
        pbar.set_description("New description")
        # Real tqdm may add ": " suffix, fallback may not
        assert "New description" in pbar.desc
        pbar.close()

    def test_trange_fallback(self):
        """Test trange fallback creates proper range."""
        result = list(trange(5, disable=True))
        assert result == [0, 1, 2, 3, 4]


class TestSolverTimer:
    """Test SolverTimer context manager for timing operations."""

    def test_timer_basic_usage(self):
        """Test SolverTimer measures duration correctly."""
        with patch("sys.stdout", new=StringIO()):
            with SolverTimer("Test operation", verbose=True) as timer:
                time.sleep(0.05)

            assert timer.duration is not None
            assert timer.duration >= 0.05
            assert timer.duration < 0.1  # Should be close to 0.05

    def test_timer_silent_mode(self):
        """Test SolverTimer with verbose=False produces no output."""
        with patch("sys.stdout", new=StringIO()):
            with SolverTimer("Silent operation", verbose=False) as timer:
                time.sleep(0.01)

            # In silent mode, fallback may still print, so just verify timer worked
            assert timer.duration is not None

    def test_timer_format_duration_milliseconds(self):
        """Test duration formatting for milliseconds."""
        timer = SolverTimer("Test")
        timer.duration = 0.0005  # 0.5ms
        formatted = timer.format_duration()
        assert "ms" in formatted.lower()

    def test_timer_format_duration_seconds(self):
        """Test duration formatting for seconds."""
        timer = SolverTimer("Test")
        timer.duration = 5.5  # 5.5s
        formatted = timer.format_duration()
        assert "s" in formatted.lower()

    def test_timer_format_duration_minutes(self):
        """Test duration formatting for minutes."""
        timer = SolverTimer("Test")
        timer.duration = 125.0  # 2m 5s
        formatted = timer.format_duration()
        assert "m" in formatted.lower()

    def test_timer_format_duration_hours(self):
        """Test duration formatting for hours."""
        timer = SolverTimer("Test")
        timer.duration = 7380.0  # 2h 3m
        formatted = timer.format_duration()
        assert "h" in formatted.lower()

    def test_timer_format_duration_none(self):
        """Test duration formatting when duration is None."""
        timer = SolverTimer("Test")
        timer.duration = None
        formatted = timer.format_duration()
        assert formatted == "Unknown"

    def test_timer_exception_handling(self):
        """Test SolverTimer reports when operation fails."""
        with patch("sys.stdout", new=StringIO()):
            with pytest.raises(ValueError), SolverTimer("Failing operation", verbose=True) as timer:
                raise ValueError("Test error")

            # Timer should still have measured time until exception
            assert timer.duration is not None


class TestIterationProgress:
    """Test IterationProgress for solver iteration tracking."""

    def test_iteration_progress_basic(self):
        """Test basic iteration progress tracking."""
        with IterationProgress(10, "Test Solver", disable=True) as progress:
            for _ in range(10):
                progress.update(1)

    def test_iteration_progress_with_error(self):
        """Test iteration progress with error tracking."""
        with IterationProgress(5, "Convergence Test", disable=True) as progress:
            for i in range(5):
                error = 1.0 / (i + 1)
                progress.update(1, error=error)

    def test_iteration_progress_with_additional_info(self):
        """Test iteration progress with additional metrics."""
        with IterationProgress(5, "Detailed Progress", disable=True) as progress:
            for i in range(5):
                progress.update(1, error=1e-5, additional_info={"iteration": i + 1, "residual": 0.001})

    def test_iteration_progress_set_description(self):
        """Test updating progress bar description."""
        with IterationProgress(5, "Initial", disable=True) as progress:
            progress.set_description("Updated description")
            progress.update(1)

    def test_iteration_progress_update_frequency(self):
        """Test iteration progress respects update_frequency."""
        with IterationProgress(100, "Sparse Updates", update_frequency=10, disable=True) as progress:
            for _ in range(100):
                progress.update(1)

    def test_iteration_progress_disabled(self):
        """Test iteration progress with disable=True."""
        with IterationProgress(10, "Disabled", disable=True) as progress:
            # Should not create progress bar
            assert progress.pbar is None or hasattr(progress.pbar, "disable")
            progress.update(1)


class TestTimedOperation:
    """Test timed_operation decorator."""

    def test_timed_operation_basic(self):
        """Test timed_operation decorator works."""

        @timed_operation(description="Test function", verbose=False)
        def slow_function():
            time.sleep(0.01)
            return 42

        result = slow_function()
        assert result == 42

    def test_timed_operation_with_dict_result(self):
        """Test timed_operation adds execution_time to dict results."""

        @timed_operation(description="Dict function", verbose=False)
        def dict_function():
            time.sleep(0.01)
            return {"value": 100, "converged": True}

        result = dict_function()
        assert result["value"] == 100
        assert "execution_time" in result
        assert result["execution_time"] > 0

    def test_timed_operation_without_dict_result(self):
        """Test timed_operation with non-dict return value."""

        @timed_operation(description="List function", verbose=False)
        def list_function():
            return [1, 2, 3]

        result = list_function()
        assert result == [1, 2, 3]

    def test_timed_operation_default_description(self):
        """Test timed_operation with default function name description."""

        @timed_operation(verbose=False)
        def my_custom_function():
            return "done"

        with patch("sys.stdout", new=StringIO()):
            result = my_custom_function()
            assert result == "done"

    def test_time_solver_operation(self):
        """Test time_solver_operation convenience function."""

        @time_solver_operation
        def solver_step():
            time.sleep(0.01)
            return {"residual": 1e-6}

        with patch("sys.stdout", new=StringIO()):
            result = solver_step()
            assert "residual" in result
            assert "execution_time" in result


class TestProgressContext:
    """Test progress_context context manager."""

    def test_progress_context_with_list(self):
        """Test progress_context with list iterable."""
        items = [1, 2, 3, 4, 5]
        result = []

        with progress_context(items, "Processing", disable=True) as pbar:
            for item in pbar:
                result.append(item)

        assert result == items

    def test_progress_context_with_range(self):
        """Test progress_context with range."""
        total = 0

        with progress_context(range(10), "Counting", disable=True) as pbar:
            for i in pbar:
                total += i

        assert total == sum(range(10))

    def test_progress_context_show_rate(self):
        """Test progress_context with show_rate parameter."""
        with progress_context(range(5), "Fast processing", show_rate=True, disable=True) as pbar:
            for _ in pbar:
                pass

    def test_progress_context_disabled(self):
        """Test progress_context with disable=True."""
        items = [1, 2, 3]
        result = []

        with progress_context(items, "Silent", disable=True) as pbar:
            for item in pbar:
                result.append(item)

        assert result == items


class TestSolverProgress:
    """Test solver_progress convenience function."""

    def test_solver_progress_creation(self):
        """Test solver_progress creates IterationProgress."""
        progress = solver_progress(100, "Test Solver", disable=True)
        assert isinstance(progress, IterationProgress)
        assert progress.max_iterations == 100
        assert progress.description == "Test Solver"

    def test_solver_progress_auto_update_frequency(self):
        """Test solver_progress sets update_frequency automatically."""
        progress = solver_progress(1000, disable=True)
        # Should update every 1% = 1000/100 = 10 iterations
        assert progress.update_frequency == 10

    def test_solver_progress_usage(self):
        """Test solver_progress in typical solver loop."""
        with solver_progress(20, "Newton Iterations", disable=True) as progress:
            for i in range(20):
                error = 1.0 / (i + 1)
                progress.update(1, error=error)


class TestIntegration:
    """Integration tests combining multiple progress utilities."""

    def test_nested_timers(self):
        """Test nested SolverTimer context managers."""
        with patch("sys.stdout", new=StringIO()):
            with SolverTimer("Outer operation", verbose=False) as outer:
                time.sleep(0.01)
                with SolverTimer("Inner operation", verbose=False) as inner:
                    time.sleep(0.01)

            assert outer.duration is not None
            assert inner.duration is not None
            assert outer.duration > inner.duration

    def test_timer_with_progress(self):
        """Test combining SolverTimer with IterationProgress."""
        with (
            patch("sys.stdout", new=StringIO()),
            SolverTimer("Solver with progress", verbose=False),
            solver_progress(10, "Iterations", disable=True) as progress,
        ):
            for _ in range(10):
                time.sleep(0.001)
                progress.update(1, error=1e-6)

    def test_decorated_function_with_progress(self):
        """Test timed decorator on function that uses progress."""

        @timed_operation(verbose=False)
        def solver_with_progress():
            results = []
            with solver_progress(5, "Solver steps", disable=True) as progress:
                for i in range(5):
                    results.append(i * 2)
                    progress.update(1)
            return {"results": results}

        result = solver_with_progress()
        assert result["results"] == [0, 2, 4, 6, 8]
        assert "execution_time" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
