"""
Parameter sweep functionality for MFG_PDE workflow system.

This module provides automated parameter space exploration capabilities,
enabling researchers to efficiently study the behavior of MFG systems
across different parameter configurations.
"""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

import numpy as np

from .common import setup_workflow_logging

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable


@dataclass
class SweepConfiguration:
    """Configuration for parameter sweep execution."""

    parameters: dict[str, list[Any]]
    execution_mode: str = "sequential"  # "sequential", "parallel_threads", "parallel_processes"
    max_workers: int | None = None
    batch_size: int | None = None
    save_intermediate: bool = True
    output_dir: Path | None = None
    timeout_per_run: float | None = None
    retry_failed: bool = True
    max_retries: int = 3

    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = mp.cpu_count()

        if self.output_dir is None:
            self.output_dir = Path.cwd() / ".mfg_sweeps"

        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class ParameterSweep:
    """
    Parameter sweep executor for systematic exploration of parameter spaces.

    This class provides comprehensive parameter space exploration capabilities,
    including grid search, random sampling, and adaptive methods.
    """

    def __init__(self, parameters: dict[str, Any], config: SweepConfiguration | None = None):
        """
        Initialize parameter sweep.

        Args:
            parameters: Dictionary of parameter names to values/ranges
            config: Sweep configuration options
        """
        self.parameters = parameters
        self.config = config or SweepConfiguration(parameters)

        # Generate parameter combinations
        self.parameter_combinations = self._generate_combinations()
        self.total_combinations = len(self.parameter_combinations)

        # Results storage
        self.results: list[dict[str, Any]] = []
        self.failed_runs: list[dict[str, Any]] = []

        # Execution tracking
        self.completed_runs = 0
        self.start_time: float | None = None
        self.end_time: float | None = None

        # Setup logging
        self.logger = self._setup_logging()

        self.logger.info(f"Initialized parameter sweep with {self.total_combinations} combinations")

    def _generate_combinations(self) -> list[dict[str, Any]]:
        """Generate all parameter combinations."""
        param_names = []
        param_values = []

        for name, values in self.parameters.items():
            param_names.append(name)

            if not isinstance(values, (list, tuple, np.ndarray)):
                # Single value - convert to list
                param_values.append([values])
            else:
                param_values.append(list(values))

        # Generate Cartesian product
        combinations = []
        for combo in itertools.product(*param_values):
            combination = dict(zip(param_names, combo, strict=True))
            combinations.append(combination)

        return combinations

    def execute(self, function: Callable, **kwargs) -> list[dict[str, Any]]:
        """
        Execute parameter sweep with given function.

        Args:
            function: Function to execute for each parameter combination
            **kwargs: Additional arguments for execution

        Returns:
            List of results for each parameter combination
        """
        self.logger.info(f"Starting parameter sweep execution with {self.total_combinations} combinations")

        self.start_time = time.time()

        try:
            if self.config.execution_mode == "sequential":
                self._execute_sequential(function, **kwargs)
            elif self.config.execution_mode == "parallel_threads":
                self._execute_parallel_threads(function, **kwargs)
            elif self.config.execution_mode == "parallel_processes":
                self._execute_parallel_processes(function, **kwargs)
            else:
                raise ValueError(f"Unknown execution mode: {self.config.execution_mode}")

            self.end_time = time.time()

            # Save final results
            if self.config.save_intermediate:
                self._save_results()

            self.logger.info(f"Parameter sweep completed in {self.end_time - self.start_time:.2f}s")
            self.logger.info(f"Successful runs: {len(self.results)}/{self.total_combinations}")

            return self.results

        except Exception as e:
            self.logger.error(f"Parameter sweep failed: {e}")
            raise

    def _execute_sequential(self, function: Callable, **kwargs) -> list[dict[str, Any]]:
        """Execute parameter sweep sequentially."""
        for i, params in enumerate(self.parameter_combinations):
            self.logger.info(f"Executing combination {i + 1}/{self.total_combinations}: {params}")

            result = self._execute_single(function, params, run_id=i, **kwargs)

            if result["status"] == "success":
                self.results.append(result)
            else:
                self.failed_runs.append(result)

            self.completed_runs += 1

            # Save intermediate results
            if self.config.save_intermediate and (i + 1) % 10 == 0:
                self._save_intermediate_results()

        return self.results

    def _execute_parallel_threads(self, function: Callable, **kwargs) -> list[dict[str, Any]]:
        """Execute parameter sweep using thread pool."""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_params = {}
            for i, params in enumerate(self.parameter_combinations):
                future = executor.submit(self._execute_single, function, params, run_id=i, **kwargs)
                future_to_params[future] = (i, params)

            # Collect results as they complete
            for future in as_completed(future_to_params):
                i, params = future_to_params[future]

                try:
                    result = future.result(timeout=self.config.timeout_per_run)

                    if result["status"] == "success":
                        self.results.append(result)
                    else:
                        self.failed_runs.append(result)

                    self.completed_runs += 1

                    self.logger.info(f"Completed {self.completed_runs}/{self.total_combinations}: {params}")

                    # Save intermediate results
                    if self.config.save_intermediate and self.completed_runs % 10 == 0:
                        self._save_intermediate_results()

                except Exception as e:
                    self.logger.error(f"Thread execution failed for {params}: {e}")
                    self.failed_runs.append(
                        {
                            "run_id": i,
                            "parameters": params,
                            "status": "failed",
                            "error": str(e),
                            "execution_time": 0,
                        }
                    )

        return self.results

    def _execute_parallel_processes(self, function: Callable, **kwargs) -> list[dict[str, Any]]:
        """Execute parameter sweep using process pool."""

        # Create worker function that can be pickled
        def worker_function(args):
            func, params, run_id, extra_kwargs = args
            return self._execute_single(func, params, run_id, **extra_kwargs)

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Prepare arguments for each worker
            worker_args = [(function, params, i, kwargs) for i, params in enumerate(self.parameter_combinations)]

            # Submit all tasks
            futures = [executor.submit(worker_function, args) for args in worker_args]

            # Collect results
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result(timeout=self.config.timeout_per_run)

                    if result["status"] == "success":
                        self.results.append(result)
                    else:
                        self.failed_runs.append(result)

                    self.completed_runs += 1

                    params = self.parameter_combinations[result["run_id"]]
                    self.logger.info(f"Completed {self.completed_runs}/{self.total_combinations}: {params}")

                    # Save intermediate results
                    if self.config.save_intermediate and self.completed_runs % 10 == 0:
                        self._save_intermediate_results()

                except Exception as e:
                    self.logger.error(f"Process execution failed: {e}")
                    self.failed_runs.append(
                        {
                            "run_id": i,
                            "parameters": self.parameter_combinations[i],
                            "status": "failed",
                            "error": str(e),
                            "execution_time": 0,
                        }
                    )

        return self.results

    def _execute_single(self, function: Callable, params: dict[str, Any], run_id: int, **kwargs) -> dict[str, Any]:
        """Execute function with single parameter combination."""
        start_time = time.time()

        try:
            # Execute function with parameters
            result = function(**params, **kwargs)

            execution_time = time.time() - start_time

            # Prepare result record
            result_record = {
                "run_id": run_id,
                "parameters": params,
                "result": result,
                "status": "success",
                "execution_time": execution_time,
                "timestamp": time.time(),
            }

            # Add metadata if result is a dict
            if isinstance(result, dict):
                result_record.update(result)

            return result_record

        except Exception as e:
            execution_time = time.time() - start_time

            error_record = {
                "run_id": run_id,
                "parameters": params,
                "status": "failed",
                "error": str(e),
                "execution_time": execution_time,
                "timestamp": time.time(),
            }

            # Retry if configured
            if self.config.retry_failed:
                if not hasattr(self, "_retry_count"):
                    self._retry_count: dict[int, int] = {}
                if self._retry_count.get(run_id, 0) < self.config.max_retries:
                    self._retry_count[run_id] = self._retry_count.get(run_id, 0) + 1
                    self.logger.warning(f"Retrying run {run_id} (attempt {self._retry_count[run_id]})")
                    return self._execute_single(function, params, run_id, **kwargs)

            return error_record

    def analyze_results(self) -> dict[str, Any]:
        """Analyze parameter sweep results."""
        if not self.results:
            return {"error": "No successful results to analyze"}

        analysis = {
            "summary": {
                "total_runs": self.total_combinations,
                "successful_runs": len(self.results),
                "failed_runs": len(self.failed_runs),
                "success_rate": len(self.results) / self.total_combinations if self.total_combinations > 0 else 0,
                "total_time": (self.end_time - self.start_time if self.end_time and self.start_time else None),
                "average_time_per_run": np.mean([r["execution_time"] for r in self.results]),
            },
            "parameter_analysis": {},
            "performance_analysis": {},
        }

        # Analyze parameter effects
        df = self.to_dataframe()
        if df is not None:
            analysis["parameter_analysis"] = self._analyze_parameter_effects(df)
            analysis["performance_analysis"] = self._analyze_performance(df)

        return analysis

    def _analyze_parameter_effects(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze effects of different parameters on results."""
        effects = {}

        # Get numeric result columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        result_columns = [
            col for col in numeric_columns if col not in [*self.parameters.keys(), "run_id", "execution_time"]
        ]

        for param_name in self.parameters:
            if param_name in df.columns:
                param_effects = {}

                for result_col in result_columns:
                    if result_col in df.columns:
                        # Group by parameter value and compute statistics
                        grouped = df.groupby(param_name)[result_col].agg(["mean", "std", "min", "max"])
                        param_effects[result_col] = grouped.to_dict()

                effects[param_name] = param_effects

        return effects

    def _analyze_performance(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze performance characteristics."""
        performance = {}

        if "execution_time" in df.columns:
            performance["execution_time"] = {
                "mean": df["execution_time"].mean(),
                "std": df["execution_time"].std(),
                "min": df["execution_time"].min(),
                "max": df["execution_time"].max(),
                "median": df["execution_time"].median(),
            }

        # Analyze convergence if available
        if "converged" in df.columns:
            performance["convergence_rate"] = df["converged"].mean()

        if "iterations" in df.columns:
            performance["iterations"] = {
                "mean": df["iterations"].mean(),
                "std": df["iterations"].std(),
                "min": df["iterations"].min(),
                "max": df["iterations"].max(),
            }

        return performance

    def to_dataframe(self) -> pd.DataFrame | None:
        """Convert results to pandas DataFrame for analysis."""
        if not self.results:
            return None

        try:
            # Flatten results
            flattened_results = []

            for result in self.results:
                flat_result = {}

                # Add parameters
                flat_result.update(result["parameters"])

                # Add metadata
                flat_result["run_id"] = result["run_id"]
                flat_result["execution_time"] = result["execution_time"]
                flat_result["status"] = result["status"]

                # Add result data
                if isinstance(result.get("result"), dict):
                    flat_result.update(result["result"])
                else:
                    flat_result["result_value"] = result.get("result")

                flattened_results.append(flat_result)

            return pd.DataFrame(flattened_results)

        except Exception as e:
            self.logger.error(f"Failed to create DataFrame: {e}")
            return None

    def save_to_csv(self, filename: str | None = None) -> str:
        """Save results to CSV file."""
        df = self.to_dataframe()
        if df is None:
            raise ValueError("No results to save")

        if filename is None:
            timestamp = int(time.time())
            filename = f"parameter_sweep_{timestamp}.csv"

        filepath = self.config.output_dir / filename if self.config.output_dir is not None else Path(filename)
        df.to_csv(filepath, index=False)

        self.logger.info(f"Saved results to {filepath}")
        return str(filepath)

    def _save_results(self):
        """Save complete results to disk."""
        timestamp = int(time.time())

        # Save results as pickle
        if self.config.output_dir is not None:
            results_file = self.config.output_dir / f"sweep_results_{timestamp}.pkl"
        else:
            results_file = Path(f"sweep_results_{timestamp}.pkl")
        with open(results_file, "wb") as f:
            pickle.dump(
                {
                    "parameters": self.parameters,
                    "config": self.config,
                    "results": self.results,
                    "failed_runs": self.failed_runs,
                    "metadata": {
                        "total_combinations": self.total_combinations,
                        "completed_runs": self.completed_runs,
                        "start_time": self.start_time,
                        "end_time": self.end_time,
                    },
                },
                f,
            )

        # Save as JSON (excluding non-serializable data)
        if self.config.output_dir is not None:
            json_file = self.config.output_dir / f"sweep_summary_{timestamp}.json"
        else:
            json_file = Path(f"sweep_summary_{timestamp}.json")
        summary = {
            "parameters": self.parameters,
            "total_combinations": self.total_combinations,
            "successful_runs": len(self.results),
            "failed_runs": len(self.failed_runs),
            "execution_time": (self.end_time - self.start_time if self.end_time and self.start_time else None),
        }

        with open(json_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Saved complete results to {results_file}")

    def _save_intermediate_results(self):
        """Save intermediate results during execution."""
        timestamp = int(time.time())

        if self.config.output_dir is not None:
            intermediate_file = self.config.output_dir / f"intermediate_{timestamp}.json"
        else:
            intermediate_file = Path(f"intermediate_{timestamp}.json")

        summary = {
            "completed_runs": self.completed_runs,
            "total_combinations": self.total_combinations,
            "successful_runs": len(self.results),
            "failed_runs": len(self.failed_runs),
            "progress": self.completed_runs / self.total_combinations if self.total_combinations > 0 else 0,
            "latest_results": (self.results[-5:] if len(self.results) >= 5 else self.results),
        }

        with open(intermediate_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for parameter sweep."""
        if self.config.output_dir is not None:
            log_file = self.config.output_dir / "parameter_sweep.log"
        else:
            log_file = Path("parameter_sweep.log")
        return setup_workflow_logging(
            "mfg_parameter_sweep",
            log_file,
            console=True,
        )


def create_grid_sweep(parameters: dict[str, list]) -> ParameterSweep:
    """Create a grid-based parameter sweep."""
    return ParameterSweep(parameters)


def create_random_sweep(parameters: dict[str, tuple], n_samples: int = 100) -> ParameterSweep:
    """
    Create a random parameter sweep.

    Args:
        parameters: Dict of parameter_name -> (min_value, max_value)
        n_samples: Number of random samples to generate
    """
    random_params = {}

    for param_name, (min_val, max_val) in parameters.items():
        if isinstance(min_val, int) and isinstance(max_val, int):
            # Integer parameters
            random_params[param_name] = np.random.randint(min_val, max_val + 1, size=n_samples).tolist()
        else:
            # Float parameters
            random_params[param_name] = np.random.uniform(min_val, max_val, size=n_samples).tolist()

    return ParameterSweep(random_params)


def create_adaptive_sweep(
    parameters: dict[str, list],
    objective_function: Callable,
    n_initial: int = 10,
    n_iterations: int = 50,
) -> ParameterSweep:
    """
    Create an adaptive parameter sweep using Bayesian optimization.

    Args:
        parameters: Parameter space definition
        objective_function: Function to optimize
        n_initial: Number of initial random samples
        n_iterations: Number of optimization iterations
    """
    # This would require scikit-optimize or similar library
    # For now, return a grid sweep as fallback
    return ParameterSweep(parameters)


# Export main classes and functions
__all__ = [
    "ParameterSweep",
    "SweepConfiguration",
    "create_adaptive_sweep",
    "create_grid_sweep",
    "create_random_sweep",
]
