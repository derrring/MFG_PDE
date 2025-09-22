#!/usr/bin/env python3
"""
Progress Monitoring and Timing Example

Demonstrates how to add modern progress bars, timing, and monitoring
to MFG_PDE solvers using the new progress utilities.
"""

import os
import sys
import time

import numpy as np

# Add the parent directory to the path so we can import mfg_pde
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mfg_pde import MFGProblem, create_fast_solver
from mfg_pde.utils.progress import (
    IterationProgress,
    SolverTimer,
    check_tqdm_availability,
    progress_context,
    solver_progress,
    timed_operation,
)
from mfg_pde.utils.solver_decorators import (
    enhanced_solver_method,
    format_solver_summary,
    update_solver_progress,
)


def create_test_problem():
    """Create a simple test MFG problem."""

    class SimpleMFGProblem(MFGProblem):
        def __init__(self):
            super().__init__(T=1.0, Nt=20, xmin=0.0, xmax=1.0, Nx=50)

        def g(self, x):
            return 0.5 * (x - 0.5) ** 2

        def rho0(self, x):
            return np.exp(-10 * (x - 0.3) ** 2) + np.exp(-10 * (x - 0.7) ** 2)

        def f(self, x, u, m):
            return 0.1 * u**2 + 0.05 * m

        def sigma(self, x):
            return 0.1

        def H(self, x, p, m):
            return 0.5 * p**2

        def dH_dm(self, x, p, m):
            return 0.0

    return SimpleMFGProblem()


def demo_basic_timing():
    """Demonstrate basic timing utilities."""
    print("=" * 60)
    print("BASIC TIMING DEMONSTRATION")
    print("=" * 60)

    # Simple timing context
    with SolverTimer("Matrix Operation") as timer:
        time.sleep(0.1)  # Simulate work
        result = np.random.rand(100, 100) @ np.random.rand(100, 100)

    print(f"Operation completed in: {timer.format_duration()}")
    print()

    # Timed decorator
    @timed_operation("Custom computation", verbose=True)
    def expensive_computation():
        time.sleep(0.05)
        return {"result": 42, "status": "success"}

    result = expensive_computation()
    print(f"Decorated function result: {result}")
    print()


def demo_progress_bars():
    """Demonstrate progress bar utilities."""
    print("=" * 60)
    print("PROGRESS BAR DEMONSTRATION")
    print("=" * 60)

    # Check if tqdm is available
    has_tqdm = check_tqdm_availability()
    print(f"TQDM available: {has_tqdm}")
    print()

    # Basic iteration progress
    print("1. Basic iteration progress:")
    with solver_progress(30, "Sample Solver") as progress:
        for i in range(30):
            time.sleep(0.01)  # Simulate solver work
            error = 1.0 / (i + 1)  # Decreasing error
            progress.update(1, error=error)
    print()

    # Advanced progress with additional metrics
    print("2. Advanced progress with metrics:")
    with IterationProgress(20, "Advanced Solver", update_frequency=2) as progress:
        for i in range(20):
            time.sleep(0.02)
            error = np.exp(-i * 0.2)  # Exponential decay
            residual = error * 0.1
            additional_info = {"residual": f"{residual:.2e}", "step": f"{i+1}/20"}
            progress.update(1, error=error, additional_info=additional_info)
    print()

    # Progress context for iterables
    print("3. Progress context for data processing:")
    data = range(15)
    with progress_context(data, "Processing Data", show_rate=True) as pbar:
        for item in pbar:
            time.sleep(0.03)
            pbar.set_postfix(item=item, status="OK")
    print()


def demo_solver_integration():
    """Demonstrate integration with actual MFG solvers."""
    print("=" * 60)
    print("SOLVER INTEGRATION DEMONSTRATION")
    print("=" * 60)

    problem = create_test_problem()
    x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx)
    collocation_points = x_coords.reshape(-1, 1)

    # Create solver with factory
    print("Creating solver with factory patterns...")
    solver = create_fast_solver(
        problem=problem, solver_type="monitored_particle", collocation_points=collocation_points, num_particles=1000
    )

    print(f"Solver created: {type(solver).__name__}")
    print(f"Particles: {solver.num_particles}")
    print()

    # Demonstrate manual progress integration
    print("Manual progress integration example:")
    max_iter = 10

    with solver_progress(max_iter, "Manual Integration Demo") as progress:
        for i in range(max_iter):
            # Simulate solver iteration
            time.sleep(0.05)

            # Simulate convergence
            error = 1e-2 * np.exp(-i * 0.5)
            converged = error < 1e-4

            # Update progress
            progress.update(
                1, error=error, additional_info={"iteration": i + 1, "converged": "Yes" if converged else "No"}
            )

            if converged:
                print(f"\n Converged early at iteration {i+1}!")
                break
    print()


def demo_enhanced_solver_wrapper():
    """Demonstrate enhanced solver wrapper functionality."""
    print("=" * 60)
    print("ENHANCED SOLVER WRAPPER DEMONSTRATION")
    print("=" * 60)

    class MockSolver:
        """Mock solver class for demonstration."""

        def __init__(self, max_iterations=15):
            self.max_iterations = max_iterations
            self.name = "MockSolver"

        @enhanced_solver_method(monitor_convergence=True, auto_progress=True, timing=True)
        def solve(self, verbose=True, **kwargs):
            """Mock solve method with progress enhancement."""
            # Extract progress tracker if provided
            progress_tracker = kwargs.get("_progress_tracker")

            results = []
            converged = False

            # Simulate solver iterations
            for i in range(self.max_iterations):
                # Simulate computation
                time.sleep(0.03)

                # Simulate convergence
                error = 1e-1 * np.exp(-i * 0.3)
                results.append(error)

                # Update progress if tracker is available
                if progress_tracker:
                    update_solver_progress(progress_tracker, i + 1, error=error, phase="Picard")

                # Check convergence
                if error < 1e-3:
                    converged = True
                    if verbose:
                        print(f"\nSUCCESS: Converged at iteration {i+1}")
                    break

            return {
                "converged": converged,
                "iterations": i + 1 if converged else self.max_iterations,
                "final_error": error,
                "error_history": results,
            }

    # Test enhanced solver
    solver = MockSolver(max_iterations=20)
    print("Running enhanced mock solver...")

    with SolverTimer("Enhanced Solver Demo"):
        result = solver.solve(verbose=True)

    print("\nSolver Results:")
    print(f"  Converged: {result['converged']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Final error: {result['final_error']:.2e}")
    if "execution_time" in result:
        print(f"  Execution time: {result['execution_time']:.3f}s")
    print()


def demo_performance_comparison():
    """Demonstrate performance comparison with and without progress monitoring."""
    print("=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    def benchmark_operation(use_progress=True, n_iterations=50):
        """Benchmark operation with optional progress."""
        if use_progress:
            with solver_progress(n_iterations, "Benchmark") as progress:
                for i in range(n_iterations):
                    time.sleep(0.002)  # Minimal work
                    progress.update(1)
        else:
            for i in range(n_iterations):
                time.sleep(0.002)  # Same minimal work

    # Benchmark with progress
    with SolverTimer("With Progress Monitoring") as timer1:
        benchmark_operation(use_progress=True)

    # Benchmark without progress
    with SolverTimer("Without Progress Monitoring") as timer2:
        benchmark_operation(use_progress=False)

    overhead = ((timer1.duration - timer2.duration) / timer2.duration) * 100
    print(f"\nProgress monitoring overhead: {overhead:.1f}%")
    print("(Overhead is typically minimal for real solver operations)")
    print()


def demo_summary_formatting():
    """Demonstrate summary formatting utilities."""
    print("=" * 60)
    print("SUMMARY FORMATTING DEMONSTRATION")
    print("=" * 60)

    # Example solver results
    results = [
        {
            "solver_name": "FastParticleCollocationSolver",
            "iterations": 15,
            "final_error": 1.23e-6,
            "execution_time": 2.5,
            "converged": True,
        },
        {
            "solver_name": "AccurateMonitoredSolver",
            "iterations": 50,
            "final_error": 3.45e-8,
            "execution_time": 125.7,
            "converged": True,
        },
        {
            "solver_name": "TimeoutSolver",
            "iterations": 100,
            "final_error": 1.2e-3,
            "execution_time": 0.15,
            "converged": False,
        },
    ]

    for result in results:
        summary = format_solver_summary(**result)
        print(summary)


def run_progress_demo():
    """Run complete progress monitoring demonstration."""
    print(" MFG_PDE PROGRESS MONITORING & TIMING DEMONSTRATION")
    print("=" * 80)
    print("This example demonstrates modern progress bars, timing, and")
    print("performance monitoring capabilities for MFG_PDE solvers.")
    print("=" * 80)
    print()

    try:
        demo_basic_timing()
        demo_progress_bars()
        demo_solver_integration()
        demo_enhanced_solver_wrapper()
        demo_performance_comparison()
        demo_summary_formatting()

        print("=" * 80)
        print("SUCCESS: PROGRESS MONITORING DEMONSTRATION COMPLETED")
        print("=" * 80)
        print()
        print("Key Features Demonstrated:")
        print("• ⏱️  Precise timing with SolverTimer context manager")
        print("•  Beautiful progress bars with tqdm integration")
        print("•  Easy decorator-based solver enhancement")
        print("•  Real-time convergence monitoring")
        print("•  Minimal performance overhead")
        print("•  Professional result summaries")
        print()
        print("Installation tip: For best experience, install tqdm:")
        print("  pip install tqdm")
        print()

    except Exception as e:
        print(f"ERROR: Error in demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_progress_demo()
