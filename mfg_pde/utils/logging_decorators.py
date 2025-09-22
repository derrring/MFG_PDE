#!/usr/bin/env python3
"""
Logging Decorators for MFG_PDE

Provides decorators to easily add professional logging to solver methods
and other computational functions without modifying their core logic.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable

from .logging import LoggedOperation, get_logger, log_solver_completion, log_solver_start


def logged_solver_method(
    logger_name: str | None = None,
    log_level: str = "INFO",
    log_config: bool = True,
    log_performance: bool = True,
    log_result: bool = True,
):
    """
    Decorator to add comprehensive logging to solver methods.

    Args:
        logger_name: Custom logger name (if None, uses class name)
        log_level: Logging level for the operation
        log_config: Whether to log solver configuration
        log_performance: Whether to log performance metrics
        log_result: Whether to log result summary
    """

    def decorator(solve_method: Callable) -> Callable:
        @functools.wraps(solve_method)
        def wrapper(self, *args, **kwargs):
            # Get logger
            if logger_name:
                logger = get_logger(logger_name)
            else:
                class_name = self.__class__.__name__
                logger = get_logger(f"mfg_pde.solvers.{class_name}")

            # Extract solver configuration for logging
            solver_config = {}
            if log_config:
                # Try to extract common solver parameters
                for attr in ["max_iterations", "tolerance", "num_particles", "preset"]:
                    if hasattr(self, attr):
                        solver_config[attr] = getattr(self, attr)

                # Add any explicit kwargs
                for key in ["max_iterations", "tolerance", "verbose"]:
                    if key in kwargs:
                        solver_config[key] = kwargs[key]

            # Log solver start
            solver_name = self.__class__.__name__
            if log_config and solver_config:
                log_solver_start(logger, solver_name, solver_config)
            else:
                logger.info(f"Starting {solver_name}")

            # Execute with performance monitoring
            start_time = time.time()
            try:
                result = solve_method(self, *args, **kwargs)
                execution_time = time.time() - start_time

                # Log completion
                if log_result:
                    # Try to extract result information
                    iterations_raw = getattr(result, "iterations", 0)
                    iterations = int(iterations_raw) if isinstance(iterations_raw, (int, float)) else 0
                    final_error = getattr(result, "final_error", 0.0)
                    converged = getattr(result, "converged", True)

                    log_solver_completion(
                        logger,
                        solver_name,
                        iterations,
                        final_error,
                        execution_time,
                        converged,
                    )
                elif log_performance:
                    logger.info(f"{solver_name} completed in {execution_time:.3f}s")

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{solver_name} failed after {execution_time:.3f}s: {e!s}")
                raise

        return wrapper

    return decorator


def logged_operation(
    operation_name: str | None = None,
    logger_name: str | None = None,
    log_level: str = "INFO",
    log_args: bool = False,
    log_result: bool = False,
):
    """
    Decorator to add logging to any operation.

    Args:
        operation_name: Name of the operation (if None, uses function name)
        logger_name: Custom logger name (if None, uses module name)
        log_level: Logging level
        log_args: Whether to log function arguments
        log_result: Whether to log return value
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            logger = get_logger(logger_name) if logger_name else get_logger(func.__module__)

            # Determine operation name
            op_name = operation_name or func.__name__

            # Log arguments if requested
            if log_args:
                args_str = f"args={args}" if args else ""
                kwargs_str = f"kwargs={kwargs}" if kwargs else ""
                params_str = ", ".join(filter(None, [args_str, kwargs_str]))
                logger.debug(f"Calling {op_name} with {params_str}")

            # Execute with logging
            with LoggedOperation(logger, op_name, getattr(logger, log_level.lower())):
                result = func(*args, **kwargs)

                # Log result if requested
                if log_result:
                    logger.debug(f"{op_name} returned: {result}")

                return result

        return wrapper

    return decorator


def logged_validation(component_name: str | None = None, logger_name: str | None = None):
    """
    Decorator to add logging to validation functions.

    Args:
        component_name: Name of the component being validated
        logger_name: Custom logger name
    """

    def decorator(validation_func: Callable) -> Callable:
        @functools.wraps(validation_func)
        def wrapper(*args, **kwargs):
            # Get logger
            logger = get_logger(logger_name) if logger_name else get_logger("mfg_pde.validation")

            component = component_name or validation_func.__name__

            try:
                result = validation_func(*args, **kwargs)
                logger.debug(f"Validation passed: {component}")
                return result
            except Exception as e:
                from .logging import log_validation_error

                log_validation_error(logger, component, str(e))
                raise

        return wrapper

    return decorator


def performance_logged(
    operation_name: str | None = None,
    logger_name: str | None = None,
    include_memory: bool = False,
):
    """
    Decorator to log performance metrics for functions.

    Args:
        operation_name: Name of the operation
        logger_name: Custom logger name
        include_memory: Whether to include memory usage metrics
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            logger = get_logger(logger_name) if logger_name else get_logger("mfg_pde.performance")

            op_name = operation_name or func.__name__

            # Memory tracking (if requested)
            memory_enabled = include_memory  # Capture closure variable
            memory_before = 0.0
            process = None
            if memory_enabled:
                try:
                    import psutil

                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                except ImportError:
                    memory_enabled = False
                    logger.warning("psutil not available for memory tracking")

            # Execute with timing
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # Collect metrics
            metrics = {}
            if memory_enabled and process is not None:
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                metrics["memory_delta"] = f"{memory_after - memory_before:.1f}MB"
                metrics["memory_peak"] = f"{memory_after:.1f}MB"

            # Log performance
            from .logging import log_performance_metric

            log_performance_metric(logger, op_name, duration, metrics)

            return result

        return wrapper

    return decorator


class LoggingMixin:
    """
    Mixin class to add logging capabilities to any class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._enable_logging = True

    def enable_logging(self, enabled: bool = True):
        """Enable or disable logging for this instance."""
        self._enable_logging = enabled

    def log_info(self, message: str):
        """Log info message."""
        if self._enable_logging:
            self._logger.info(message)

    def log_debug(self, message: str):
        """Log debug message."""
        if self._enable_logging:
            self._logger.debug(message)

    def log_warning(self, message: str):
        """Log warning message."""
        if self._enable_logging:
            self._logger.warning(message)

    def log_error(self, message: str):
        """Log error message."""
        if self._enable_logging:
            self._logger.error(message)

    def log_operation(self, operation_name: str):
        """Return a LoggedOperation context manager."""
        if self._enable_logging:
            return LoggedOperation(self._logger, operation_name)
        else:
            # No-op context manager
            class NoOpContext:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

            return NoOpContext()


# Convenience function to add logging to existing classes
def add_logging_to_class(cls, logger_name: str | None = None):
    """
    Add logging capabilities to an existing class.

    Args:
        cls: Class to enhance with logging
        logger_name: Custom logger name

    Returns:
        Enhanced class with logging capabilities
    """

    class LoggedClass(LoggingMixin, cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if logger_name:
                self._logger = get_logger(logger_name)

    # Preserve class metadata
    LoggedClass.__name__ = f"Logged{cls.__name__}"
    LoggedClass.__qualname__ = LoggedClass.__name__
    LoggedClass.__module__ = cls.__module__

    return LoggedClass


# Example usage functions
def demonstrate_logging_decorators():
    """Demonstrate the logging decorators."""
    from .logging import configure_logging

    # Configure logging
    configure_logging(level="DEBUG", use_colors=True)

    # Example solver class
    class ExampleSolver:
        def __init__(self, max_iterations=50, tolerance=1e-6):
            self.max_iterations = max_iterations
            self.tolerance = tolerance

        @logged_solver_method(log_config=True, log_performance=True)
        def solve(self, verbose=True):
            """Example solve method with logging."""
            import time

            time.sleep(0.5)  # Simulate computation

            # Mock result object
            class MockResult:
                def __init__(self):
                    self.iterations = 25
                    self.final_error = 1.2e-7
                    self.converged = True

            return MockResult()

    # Example utility function
    @logged_operation("Matrix computation", log_args=True)
    def compute_matrix(size: int, method: str = "random"):
        """Example function with operation logging."""
        import numpy as np

        time.sleep(0.1)
        return np.random.rand(size, size)

    # Example validation function
    @logged_validation("Array dimensions")
    def validate_array(arr, expected_shape):
        """Example validation with logging."""
        if arr.shape != expected_shape:
            raise ValueError(f"Expected {expected_shape}, got {arr.shape}")
        return True

    # Demonstrate usage
    print("Demonstrating logging decorators:")

    # Solver with logging
    solver = ExampleSolver()
    _ = solver.solve()

    # Operation with logging
    _ = compute_matrix(10, "random")

    # Validation with logging
    import numpy as np

    test_array = np.ones((5, 5))
    validate_array(test_array, (5, 5))

    print("Logging decorators demonstration complete!")


if __name__ == "__main__":
    demonstrate_logging_decorators()
