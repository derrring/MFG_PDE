"""
Workflow decorators for MFG_PDE workflow system.

This module provides convenient decorators for creating workflows, experiments,
and parameter studies with minimal boilerplate code.
"""

from __future__ import annotations

import functools
import inspect
import time
from typing import TYPE_CHECKING, Any

from .experiment_tracker import ExperimentTracker
from .parameter_sweep import ParameterSweep

if TYPE_CHECKING:
    from collections.abc import Callable

    from .workflow_manager import Workflow


def workflow_step(
    workflow: Workflow | None = None,
    name: str | None = None,
    dependencies: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """
    Decorator to mark a function as a workflow step.

    Args:
        workflow: Workflow to add step to (if None, uses global workflow)
        name: Name of the step (if None, uses function name)
        dependencies: List of step names this step depends on
        metadata: Additional metadata for the step

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        step_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # If workflow is provided, add as step and return step ID
            if workflow is not None:
                step_id = workflow.add_step(
                    name=step_name,
                    function=func,
                    dependencies=dependencies or [],
                    metadata=metadata or {},
                )
                return step_id
            else:
                # Execute function directly
                return func(*args, **kwargs)

        # Store workflow metadata on function
        wrapper._workflow_step = True  # type: ignore[attr-defined]
        wrapper._step_name = step_name  # type: ignore[attr-defined]
        wrapper._dependencies = dependencies or []  # type: ignore[attr-defined]
        wrapper._metadata = metadata or {}  # type: ignore[attr-defined]

        return wrapper

    return decorator


def experiment(
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    auto_save: bool = True,
    workspace_path: str | None = None,
):
    """
    Decorator to mark a function as an experiment.

    Args:
        name: Experiment name (if None, uses function name)
        description: Experiment description
        tags: List of tags for categorization
        auto_save: Whether to automatically save experiment results
        workspace_path: Path to experiment workspace

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        experiment_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create experiment tracker
            from pathlib import Path

            workspace_path_obj = Path(workspace_path) if workspace_path else None
            tracker = ExperimentTracker(workspace_path=workspace_path_obj)

            # Create experiment
            exp = tracker.create_experiment(
                name=experiment_name,
                description=description or func.__doc__ or "",
                tags=tags or [],
            )

            # Start experiment
            exp.start()

            try:
                # Execute function with experiment context
                kwargs["experiment"] = exp
                result = func(*args, **kwargs)

                # Record result
                exp.add_result("final_result", result)
                exp.complete()

                # Auto-save if requested
                if auto_save:
                    exp.save()

                return result

            except Exception as e:
                exp.fail(str(e))
                if auto_save:
                    exp.save()
                raise

        # Store experiment metadata
        wrapper._experiment = True  # type: ignore[attr-defined]
        wrapper._experiment_name = experiment_name  # type: ignore[attr-defined]
        wrapper._description = description  # type: ignore[attr-defined]
        wrapper._tags = tags or []  # type: ignore[attr-defined]

        return wrapper

    return decorator


def parameter_study(
    parameters: dict[str, list[Any]],
    execution_mode: str = "sequential",
    max_workers: int | None = None,
    save_results: bool = True,
    output_dir: str | None = None,
):
    """
    Decorator to convert a function into a parameter study.

    Args:
        parameters: Dictionary of parameter names to lists of values
        execution_mode: Execution mode ("sequential", "parallel_threads", "parallel_processes")
        max_workers: Maximum number of parallel workers
        save_results: Whether to save results automatically
        output_dir: Output directory for results

    Returns:
        Decorated function that runs parameter study
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create parameter sweep
            from pathlib import Path

            from .parameter_sweep import SweepConfiguration

            output_dir_obj = Path(output_dir) if output_dir else None
            config = SweepConfiguration(
                parameters=parameters,
                execution_mode=execution_mode,
                max_workers=max_workers,
                save_intermediate=save_results,
                output_dir=output_dir_obj,
            )

            sweep = ParameterSweep(parameters, config)

            # Execute parameter study
            def study_function(**params):
                # Merge with any additional kwargs
                combined_params = {**kwargs, **params}
                return func(*args, **combined_params)

            results = sweep.execute(study_function)

            # Return both sweep object and results for analysis
            return {
                "sweep": sweep,
                "results": results,
                "analysis": sweep.analyze_results(),
                "dataframe": sweep.to_dataframe(),
            }

        # Store parameter study metadata
        wrapper._parameter_study = True  # type: ignore[attr-defined]
        wrapper._parameters = parameters  # type: ignore[attr-defined]
        wrapper._execution_mode = execution_mode  # type: ignore[attr-defined]

        return wrapper

    return decorator


def timed(func: Callable) -> Callable:
    """
    Decorator to time function execution and log results.

    Args:
        func: Function to time

    Returns:
        Decorated function with timing
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            print(f"🕐 {func.__name__} completed in {execution_time:.3f}s")

            # Add timing to result if it's a dictionary
            if isinstance(result, dict):
                result["_execution_time"] = execution_time

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"WARNING: {func.__name__} failed after {execution_time:.3f}s: {e}")
            raise

    return wrapper


def cached(
    cache_dir: str | None = None,
    cache_key: Callable | None = None,
    ttl_seconds: int | None = None,
):
    """
    Decorator to cache function results.

    Args:
        cache_dir: Directory to store cache files
        cache_key: Function to generate cache key from arguments
        ttl_seconds: Time-to-live for cache entries

    Returns:
        Decorated function with caching
    """

    def decorator(func: Callable) -> Callable:
        import hashlib
        import os
        import pickle
        from pathlib import Path

        # Set up cache directory
        cache_path = Path.cwd() / ".mfg_cache" / func.__name__ if cache_dir is None else Path(cache_dir) / func.__name__

        cache_path.mkdir(parents=True, exist_ok=True)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key is not None:
                key = cache_key(*args, **kwargs)
            else:
                # Default: hash arguments
                key_data = str(args) + str(sorted(kwargs.items()))
                key = hashlib.md5(key_data.encode()).hexdigest()

            cache_file = cache_path / f"{key}.pkl"

            # Check if cache exists and is valid
            if cache_file.exists():
                if ttl_seconds is None:
                    # No TTL - use cached result
                    with open(cache_file, "rb") as f:
                        cached_result = pickle.load(f)
                    print(f"Using cached result for {func.__name__}")
                    return cached_result["result"]
                else:
                    # Check TTL
                    cache_time = os.path.getmtime(cache_file)
                    if time.time() - cache_time < ttl_seconds:
                        with open(cache_file, "rb") as f:
                            cached_result = pickle.load(f)
                        print(f"Using cached result for {func.__name__}")
                        return cached_result["result"]

            # Execute function and cache result
            print(f"Computing {func.__name__}...")
            result = func(*args, **kwargs)

            # Save to cache
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(
                        {
                            "result": result,
                            "timestamp": time.time(),
                            "function": func.__name__,
                        },
                        f,
                    )
                print(f"💾 Cached result for {func.__name__}")
            except Exception as e:
                print(f"WARNING: Failed to cache result: {e}")

            return result

        # Add cache management methods
        def clear_cache():
            """Clear all cached results for this function."""
            import shutil

            if cache_path.exists():
                shutil.rmtree(cache_path)
                cache_path.mkdir(parents=True, exist_ok=True)
            print(f"Cleared cache for {func.__name__}")

        def cache_info():
            """Get information about cached results."""
            if not cache_path.exists():
                return {"cache_files": 0, "total_size": 0}

            cache_files = list(cache_path.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)

            return {
                "cache_files": len(cache_files),
                "total_size": total_size,
                "cache_dir": str(cache_path),
            }

        wrapper.clear_cache = clear_cache  # type: ignore[attr-defined]
        wrapper.cache_info = cache_info  # type: ignore[attr-defined]

        return wrapper

    return decorator


def retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator to retry function execution on failure.

    Args:
        max_attempts: Maximum number of retry attempts
        delay_seconds: Initial delay between retries
        backoff_factor: Factor to multiply delay by after each failure
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = delay_seconds

            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        print(f"SUCCESS: {func.__name__} succeeded on attempt {attempt + 1}")
                    return result

                except exceptions as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        print(f"WARNING: {func.__name__} failed on attempt {attempt + 1}, retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        print(f"ERROR: {func.__name__} failed after {max_attempts} attempts")

            # Re-raise the last exception
            if last_exception is not None:
                raise last_exception
            else:
                raise RuntimeError(f"Function {func.__name__} failed but no exception was captured")

        return wrapper

    return decorator


def validate_inputs(validation_rules: dict[str, Callable]):
    """
    Decorator to validate function inputs.

    Args:
        validation_rules: Dictionary of parameter_name -> validation_function

    Returns:
        Decorated function with input validation
    """

    def decorator(func: Callable) -> Callable:
        # Get function signature
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Bind arguments to parameters
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate inputs
            for param_name, validator in validation_rules.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        if not validator(value):
                            raise ValueError(f"Validation failed for parameter '{param_name}': {value}")
                    except Exception as e:
                        raise ValueError(f"Validation error for parameter '{param_name}': {e}") from e

            return func(*args, **kwargs)

        return wrapper

    return decorator


def log_execution(
    logger_name: str | None = None,
    log_inputs: bool = False,
    log_outputs: bool = False,
    log_performance: bool = True,
):
    """
    Decorator to log function execution details.

    Args:
        logger_name: Name of logger to use
        log_inputs: Whether to log input parameters
        log_outputs: Whether to log output values
        log_performance: Whether to log performance metrics

    Returns:
        Decorated function with logging
    """

    def decorator(func: Callable) -> Callable:
        import logging

        logger = logging.getLogger(logger_name or func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            logger.info(f"Starting execution of {func.__name__}")

            if log_inputs:
                logger.debug(f"Inputs - args: {args}, kwargs: {kwargs}")

            try:
                result = func(*args, **kwargs)

                execution_time = time.time() - start_time

                if log_performance:
                    logger.info(f"Completed {func.__name__} in {execution_time:.3f}s")

                if log_outputs:
                    logger.debug(f"Output: {result}")

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Failed {func.__name__} after {execution_time:.3f}s: {e}")
                raise

        return wrapper

    return decorator


# Convenience decorators for common MFG use cases
def mfg_solver(
    problem_type: str = "general",
    cache_results: bool = True,
    log_performance: bool = True,
    retry_on_failure: bool = True,
):
    """
    Composite decorator for MFG solver functions.

    Args:
        problem_type: Type of MFG problem
        cache_results: Whether to cache solver results
        log_performance: Whether to log performance metrics
        retry_on_failure: Whether to retry on convergence failures

    Returns:
        Composite decorated function
    """

    def decorator(func: Callable) -> Callable:
        # Apply decorators in order
        decorated_func = func

        if log_performance:
            decorated_func = timed(decorated_func)
            decorated_func = log_execution(logger_name=f"mfg_solver.{problem_type}", log_performance=True)(
                decorated_func
            )

        if retry_on_failure:
            decorated_func = retry(max_attempts=3, delay_seconds=0.1, exceptions=(RuntimeError, ValueError))(
                decorated_func
            )

        if cache_results:
            decorated_func = cached(cache_dir=f".mfg_cache/{problem_type}", ttl_seconds=3600)(  # 1 hour TTL
                decorated_func
            )

        return decorated_func

    return decorator


def convergence_study(
    tolerances: list[float],
    max_iterations: list[int],
    execution_mode: str = "parallel_threads",
):
    """
    Decorator for convergence analysis studies.

    Args:
        tolerances: List of tolerance values to test
        max_iterations: List of maximum iteration values to test
        execution_mode: How to execute the parameter study

    Returns:
        Decorated function for convergence study
    """
    parameters: dict[str, list[Any]] = {"tolerance": tolerances, "max_iterations": max_iterations}

    return parameter_study(
        parameters=parameters,
        execution_mode=execution_mode,
        save_results=True,
        output_dir=".mfg_convergence_studies",
    )


# Export all decorators
__all__ = [
    "cached",
    "convergence_study",
    "experiment",
    "log_execution",
    "mfg_solver",
    "parameter_study",
    "retry",
    "timed",
    "validate_inputs",
    "workflow_step",
]
