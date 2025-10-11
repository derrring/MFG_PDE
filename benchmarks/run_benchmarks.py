"""
Benchmark Runner for MFG_PDE

Executes standard benchmark problems and tracks performance metrics
using PerformanceTracker. Supports selective execution by category,
memory profiling, and automated regression detection.

Usage:
    # Run all benchmarks
    python benchmarks/run_benchmarks.py

    # Run specific category
    python benchmarks/run_benchmarks.py --category small

    # Run specific problem
    python benchmarks/run_benchmarks.py --problem LQ-MFG-Small

    # Enable memory profiling
    python benchmarks/run_benchmarks.py --profile-memory

    # Check for regressions
    python benchmarks/run_benchmarks.py --check-regression
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.performance_tracker import PerformanceTracker
from benchmarks.standard_problems import (
    BenchmarkProblem,
    get_problem_by_name,
    get_problems_by_category,
)


class BenchmarkRunner:
    """
    Execute benchmark problems and track performance.

    Handles:
    - Problem setup and solver execution
    - Timing and memory profiling
    - Result validation against success criteria
    - Performance tracking via PerformanceTracker
    """

    def __init__(
        self,
        history_dir: str | Path = "benchmarks/history",
        profile_memory: bool = False,
    ):
        """
        Initialize benchmark runner.

        Args:
            history_dir: Directory for performance history
            profile_memory: Enable memory profiling (requires psutil)
        """
        self.tracker = PerformanceTracker(history_dir=history_dir)
        self.profile_memory = profile_memory and PSUTIL_AVAILABLE

        if profile_memory and not PSUTIL_AVAILABLE:
            print("Warning: psutil not available, memory profiling disabled")
            print("Install with: pip install psutil")
            self.profile_memory = False

    def run_benchmark(self, problem: BenchmarkProblem, verbose: bool = True) -> dict:
        """
        Execute a single benchmark problem.

        Args:
            problem: BenchmarkProblem specification
            verbose: Print progress information

        Returns:
            Dictionary with benchmark results including:
            - success: bool (whether benchmark passed validation)
            - execution_time: float (seconds)
            - converged: bool
            - iterations: int
            - final_error: float
            - validation_errors: list of validation failures
        """
        if verbose:
            print(f"\nRunning benchmark: {problem.name}")
            print(f"  Category: {problem.category}")
            print(f"  Description: {problem.description}")

        # Memory tracking setup
        if self.profile_memory:
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024 * 1024)  # MB

        # Execute solver (placeholder - will integrate with actual solvers)
        start_time = time.perf_counter()

        try:
            result = self._execute_solver(problem)
            execution_time = time.perf_counter() - start_time

            # Memory tracking
            peak_memory_mb = None
            if self.profile_memory:
                memory_after = process.memory_info().rss / (1024 * 1024)  # MB
                peak_memory_mb = max(memory_before, memory_after)

            # Validate results
            validation_errors = self._validate_results(problem, result)
            success = len(validation_errors) == 0

            if verbose:
                self._print_results(
                    problem=problem,
                    execution_time=execution_time,
                    result=result,
                    peak_memory_mb=peak_memory_mb,
                    validation_errors=validation_errors,
                    success=success,
                )

            # Track performance
            benchmark_result = self.tracker.track_solver(
                solver_name=problem.solver_type,
                problem_name=problem.name,
                problem_size={k: v for k, v in problem.problem_config.items() if k in ["Nx", "Ny", "Nt"]},
                execution_time=execution_time,
                converged=result["converged"],
                iterations=result["iterations"],
                final_error=result["final_error"],
                peak_memory_mb=peak_memory_mb,
                metadata={
                    "category": problem.category,
                    "success": success,
                    "validation_errors": validation_errors,
                },
            )

            return {
                "success": success,
                "execution_time": execution_time,
                "converged": result["converged"],
                "iterations": result["iterations"],
                "final_error": result["final_error"],
                "peak_memory_mb": peak_memory_mb,
                "validation_errors": validation_errors,
                "benchmark_result": benchmark_result,
            }

        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")

            return {
                "success": False,
                "execution_time": 0.0,
                "converged": False,
                "iterations": 0,
                "final_error": float("inf"),
                "peak_memory_mb": None,
                "validation_errors": [f"Execution failed: {e}"],
                "benchmark_result": None,
            }

    def _execute_solver(self, problem: BenchmarkProblem) -> dict:
        """
        Execute solver for benchmark problem.

        This is a placeholder that simulates solver execution.
        In production, this would call actual MFG solvers.

        Args:
            problem: BenchmarkProblem specification

        Returns:
            Dictionary with solver results (U, M, convergence info)
        """
        # PLACEHOLDER: Simulate solver execution
        # In production, this would use actual solvers from mfg_pde
        np.random.seed(42)  # Reproducible results

        # Simulate solver iterations
        max_iterations = problem.solver_config["max_iterations"]
        tolerance = problem.solver_config["tolerance"]

        # Simulate convergence
        converged = True
        iterations = int(np.random.uniform(20, max_iterations * 0.6))
        final_error = tolerance * np.random.uniform(0.1, 1.0)

        return {
            "converged": converged,
            "iterations": iterations,
            "final_error": final_error,
            "U": None,  # Would contain actual solution
            "M": None,  # Would contain actual density
        }

    def _validate_results(self, problem: BenchmarkProblem, result: dict) -> list[str]:
        """
        Validate benchmark results against success criteria.

        Args:
            problem: BenchmarkProblem specification
            result: Solver execution results

        Returns:
            List of validation error messages (empty if all pass)
        """
        errors = []
        criteria = problem.success_criteria

        # Check convergence requirement
        if criteria.get("must_converge", False) and not result["converged"]:
            errors.append("Solver failed to converge")

        # Check iteration limit
        max_iter = criteria.get("max_iterations_allowed")
        if max_iter and result["iterations"] > max_iter:
            errors.append(f"Exceeded iteration limit: {result['iterations']} > {max_iter}")

        # Check error threshold
        if result["final_error"] > problem.convergence_threshold * 10:
            errors.append(
                f"Final error too large: {result['final_error']:.2e} > {problem.convergence_threshold * 10:.2e}"
            )

        return errors

    def _print_results(
        self,
        problem: BenchmarkProblem,
        execution_time: float,
        result: dict,
        peak_memory_mb: float | None,
        validation_errors: list[str],
        success: bool,
    ):
        """Print benchmark results to console."""
        print(f"  Execution time: {execution_time:.3f}s")

        # Compare to expected range
        min_time, max_time = problem.expected_time_range
        if execution_time < min_time:
            print(f"    (faster than expected: {min_time:.1f}s)")
        elif execution_time > max_time:
            print(f"    (slower than expected: {max_time:.1f}s)")

        print(f"  Converged: {result['converged']}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Final error: {result['final_error']:.2e}")

        if peak_memory_mb:
            print(f"  Peak memory: {peak_memory_mb:.1f} MB")

        if success:
            print("  Status: PASS")
        else:
            print("  Status: FAIL")
            for error in validation_errors:
                print(f"    - {error}")

    def run_suite(
        self,
        problems: list[BenchmarkProblem],
        check_regression: bool = False,
        verbose: bool = True,
    ) -> dict:
        """
        Run a suite of benchmark problems.

        Args:
            problems: List of BenchmarkProblem instances
            check_regression: Check for performance regressions
            verbose: Print progress information

        Returns:
            Dictionary with summary statistics
        """
        if verbose:
            print("=" * 70)
            print(f"Running {len(problems)} benchmark problems")
            print("=" * 70)

        results = []
        regressions = []

        for problem in problems:
            result = self.run_benchmark(problem, verbose=verbose)
            results.append(result)

            # Check for regression
            if check_regression and result["benchmark_result"]:
                is_regression, pct_change = self.tracker.check_regression(result["benchmark_result"], threshold=0.2)
                if is_regression:
                    regressions.append(
                        {
                            "problem": problem.name,
                            "percent_change": pct_change,
                        }
                    )
                    if verbose:
                        print(f"  WARNING: Performance regression detected ({pct_change * 100:.1f}% slower)")

        # Summary statistics
        total = len(results)
        passed = sum(1 for r in results if r["success"])
        failed = total - passed
        total_time = sum(r["execution_time"] for r in results)

        if verbose:
            print("\n" + "=" * 70)
            print("Benchmark Summary")
            print("=" * 70)
            print(f"Total problems: {total}")
            print(f"Passed: {passed}")
            print(f"Failed: {failed}")
            print(f"Total time: {total_time:.2f}s")

            if regressions:
                print(f"\nPerformance Regressions: {len(regressions)}")
                for reg in regressions:
                    print(f"  - {reg['problem']}: {reg['percent_change'] * 100:.1f}% slower")

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "total_time": total_time,
            "regressions": regressions,
            "results": results,
        }


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run MFG_PDE benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--category",
        choices=["small", "medium", "large", "all"],
        default="all",
        help="Run benchmarks from specific category (default: all)",
    )

    parser.add_argument(
        "--problem",
        type=str,
        help="Run specific problem by name (e.g., 'LQ-MFG-Small')",
    )

    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Enable memory profiling (requires psutil)",
    )

    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Check for performance regressions",
    )

    parser.add_argument(
        "--history-dir",
        type=str,
        default="benchmarks/history",
        help="Directory for performance history (default: benchmarks/history)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (only summary)",
    )

    args = parser.parse_args()

    # Create runner
    runner = BenchmarkRunner(
        history_dir=args.history_dir,
        profile_memory=args.profile_memory,
    )

    # Select problems
    if args.problem:
        try:
            problems = [get_problem_by_name(args.problem)]
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    else:
        problems = get_problems_by_category(args.category)

    # Run benchmarks
    summary = runner.run_suite(
        problems=problems,
        check_regression=args.check_regression,
        verbose=not args.quiet,
    )

    # Exit with error code if any failed
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
