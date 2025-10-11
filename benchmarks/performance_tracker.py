"""
Performance Tracking System for MFG_PDE
========================================

Automated performance monitoring to track solver performance over time,
detect regressions, and generate historical trend reports.

This module provides:
- PerformanceTracker: Core tracking and storage
- JSON-based time series storage
- Git commit tracking
- Regression detection
- Performance metrics collection

Part of Issue #128 implementation (Phase 1).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """
    Single benchmark measurement result.

    Stores performance metrics for a specific solver run, including
    timing, memory usage, convergence statistics, and metadata.
    """

    # Identification
    timestamp: str  # ISO 8601 format
    commit_hash: str  # Git commit SHA
    branch: str  # Git branch name
    solver_name: str  # e.g., "HJB-FDM", "Particle-Collocation"
    problem_name: str  # e.g., "LQ-MFG-Small", "Congestion-Medium"
    problem_size: dict[str, int]  # e.g., {"Nx": 50, "Nt": 50}

    # Performance metrics
    execution_time: float  # seconds
    peak_memory_mb: float | None  # MB (None if psutil unavailable)
    converged: bool
    iterations: int
    final_error: float  # ||U - U_prev||

    # System info
    python_version: str
    numpy_version: str
    platform: str  # e.g., "darwin", "linux"

    # Optional metadata
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        """Create from dictionary (JSON deserialization)."""
        return cls(**data)


class PerformanceTracker:
    """
    Track and store solver performance metrics over time.

    Manages a time-series database of benchmark results stored as JSON files.
    Provides regression detection and historical comparison capabilities.

    Usage:
        tracker = PerformanceTracker(history_dir="benchmarks/history")

        # Record a benchmark result
        result = tracker.track_solver(
            solver_name="HJB-FDM",
            problem_name="LQ-MFG-Small",
            problem_size={"Nx": 50, "Nt": 50},
            execution_time=1.23,
            converged=True,
            iterations=50,
            final_error=1.2e-6
        )

        # Check for regression
        is_regressed, pct_change = tracker.check_regression(
            result, threshold=0.2
        )
    """

    def __init__(self, history_dir: str | Path = "benchmarks/history"):
        """
        Initialize performance tracker.

        Args:
            history_dir: Directory to store benchmark history JSON files
        """
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def track_solver(
        self,
        solver_name: str,
        problem_name: str,
        problem_size: dict[str, int],
        execution_time: float,
        converged: bool,
        iterations: int,
        final_error: float,
        peak_memory_mb: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """
        Record a benchmark result and save to history.

        Args:
            solver_name: Name of the solver
            problem_name: Name of the benchmark problem
            problem_size: Dictionary with grid sizes (e.g., {"Nx": 50, "Nt": 50})
            execution_time: Execution time in seconds
            converged: Whether solver converged
            iterations: Number of iterations performed
            final_error: Final convergence error
            peak_memory_mb: Peak memory usage in MB (optional)
            metadata: Additional metadata (optional)

        Returns:
            BenchmarkResult object
        """
        # Get system information
        timestamp = datetime.now().isoformat()
        commit_hash = self._get_git_commit()
        branch = self._get_git_branch()
        python_version = self._get_python_version()
        numpy_version = self._get_numpy_version()
        platform = self._get_platform()

        # Auto-detect memory if not provided and psutil available
        if peak_memory_mb is None and PSUTIL_AVAILABLE:
            process = psutil.Process()
            peak_memory_mb = process.memory_info().rss / (1024 * 1024)  # Convert to MB

        # Create result
        result = BenchmarkResult(
            timestamp=timestamp,
            commit_hash=commit_hash,
            branch=branch,
            solver_name=solver_name,
            problem_name=problem_name,
            problem_size=problem_size,
            execution_time=execution_time,
            peak_memory_mb=peak_memory_mb,
            converged=converged,
            iterations=iterations,
            final_error=final_error,
            python_version=python_version,
            numpy_version=numpy_version,
            platform=platform,
            metadata=metadata,
        )

        # Save to history
        self._save_result(result)

        return result

    def _save_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to JSON history file."""
        # Create filename based on solver and problem
        filename = f"{result.solver_name}_{result.problem_name}.json"
        filepath = self.history_dir / filename

        # Load existing results or create new list
        if filepath.exists():
            with open(filepath) as f:
                results = json.load(f)
        else:
            results = []

        # Append new result
        results.append(result.to_dict())

        # Save back to file
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

    def load_history(self, solver_name: str, problem_name: str) -> list[BenchmarkResult]:
        """
        Load benchmark history for a specific solver and problem.

        Args:
            solver_name: Name of the solver
            problem_name: Name of the problem

        Returns:
            List of BenchmarkResult objects, sorted by timestamp
        """
        filename = f"{solver_name}_{problem_name}.json"
        filepath = self.history_dir / filename

        if not filepath.exists():
            return []

        with open(filepath) as f:
            data = json.load(f)

        results = [BenchmarkResult.from_dict(item) for item in data]

        # Sort by timestamp
        results.sort(key=lambda r: r.timestamp)

        return results

    def check_regression(
        self, result: BenchmarkResult, threshold: float = 0.2, baseline_count: int = 5
    ) -> tuple[bool, float]:
        """
        Check if a benchmark result represents a performance regression.

        Compares the result against recent historical baselines.

        Args:
            result: The benchmark result to check
            threshold: Regression threshold (e.g., 0.2 = 20% slower)
            baseline_count: Number of recent results to use as baseline

        Returns:
            Tuple of (is_regression, percent_change)
            - is_regression: True if execution time exceeds threshold
            - percent_change: Percentage change (positive = slower)
        """
        # Load historical results
        history = self.load_history(result.solver_name, result.problem_name)

        if len(history) < baseline_count:
            # Not enough history to determine regression
            return False, 0.0

        # Get recent baselines (excluding current result)
        recent = history[-baseline_count:]
        baseline_times = [r.execution_time for r in recent]
        baseline_avg = np.mean(baseline_times)

        # Calculate percentage change
        pct_change = (result.execution_time - baseline_avg) / baseline_avg

        # Check if regression (convert to Python bool to avoid numpy bool)
        is_regression = bool(pct_change > threshold)

        return is_regression, float(pct_change)

    def get_statistics(self, solver_name: str, problem_name: str) -> dict[str, Any]:
        """
        Get statistical summary of benchmark history.

        Args:
            solver_name: Name of the solver
            problem_name: Name of the problem

        Returns:
            Dictionary with statistics (mean, std, min, max, count)
        """
        history = self.load_history(solver_name, problem_name)

        if not history:
            return {"count": 0}

        times = [r.execution_time for r in history]
        errors = [r.final_error for r in history]
        iterations = [r.iterations for r in history]

        return {
            "count": len(history),
            "time_mean": float(np.mean(times)),
            "time_std": float(np.std(times)),
            "time_min": float(np.min(times)),
            "time_max": float(np.max(times)),
            "error_mean": float(np.mean(errors)),
            "error_std": float(np.std(errors)),
            "iterations_mean": float(np.mean(iterations)),
            "convergence_rate": sum(r.converged for r in history) / len(history),
        }

    # Helper methods for system information

    @staticmethod
    def _get_git_commit() -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5)
            return result.stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return "unknown"

    @staticmethod
    def _get_git_branch() -> str:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return "unknown"

    @staticmethod
    def _get_python_version() -> str:
        """Get Python version."""
        import sys

        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    @staticmethod
    def _get_numpy_version() -> str:
        """Get NumPy version."""
        return np.__version__

    @staticmethod
    def _get_platform() -> str:
        """Get platform name."""
        import sys

        return sys.platform
