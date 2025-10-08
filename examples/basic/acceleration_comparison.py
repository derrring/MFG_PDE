"""
MFG Solver Acceleration Comparison.

This example demonstrates different acceleration techniques for MFG solvers:
- Backend acceleration: JAX (CPU/GPU), PyTorch (MPS/CUDA), Numba
- Anderson acceleration: Convergence speedup via extrapolation
- Solver configuration: Damping, tolerance, iteration strategies

Demonstrates performance improvements across different hardware and algorithmic approaches.

Example Usage:
    python examples/basic/acceleration_comparison.py
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt

from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_standard_solver
from mfg_pde.utils.logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("acceleration_comparison", level="INFO")
logger = get_logger(__name__)

# Output directory
EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR.parent / "outputs" / "basic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_test_problem():
    """Create a simple LQ-MFG problem for benchmarking."""

    def hamiltonian(x, p, m):
        """Simple LQ Hamiltonian: H = (1/2)p² + (1/2)x² + m"""
        return 0.5 * p**2 + 0.5 * x**2 + m

    problem = ExampleMFGProblem(
        xmin=-5.0,
        xmax=5.0,
        Nx=100,
        T=1.0,
        Nt=50,
        sigma=0.3,
        hamiltonian=hamiltonian,
    )

    return problem


def benchmark_backend(backend_name: str, problem):
    """
    Benchmark a specific backend.

    Args:
        backend_name: "numpy", "jax", "torch", "numba"
        problem: MFG problem instance

    Returns:
        Tuple of (solve_time, iterations, converged)
    """
    logger.info(f"\n  Testing backend: {backend_name}")

    try:
        # Create solver with specified backend
        solver = create_standard_solver(problem)

        # Override backend if supported
        if hasattr(solver, "backend") and backend_name != "numpy":
            try:
                from mfg_pde.backends import create_backend

                solver.backend = create_backend(backend_name)
                if hasattr(solver.hjb_solver, "backend"):
                    solver.hjb_solver.backend = solver.backend
                if hasattr(solver.fp_solver, "backend"):
                    solver.fp_solver.backend = solver.backend
            except ImportError as e:
                logger.warning(f"    Backend {backend_name} not available: {e}")
                return None, None, False

        # Solve and time
        start = time.time()
        result = solver.solve(max_iterations=30, tolerance=1e-4)
        solve_time = time.time() - start

        logger.info(
            f"    Time: {solve_time:.3f}s, Iterations: {result.iterations}, Converged: {result.convergence_achieved}"
        )

        return solve_time, result.iterations, result.convergence_achieved

    except Exception as e:
        logger.error(f"    Error with {backend_name}: {e}")
        return None, None, False


def benchmark_anderson(problem, use_anderson: bool, depth: int = 5):
    """
    Benchmark with/without Anderson acceleration.

    Args:
        problem: MFG problem instance
        use_anderson: Whether to use Anderson acceleration
        depth: Anderson memory depth

    Returns:
        Tuple of (solve_time, iterations, converged)
    """
    label = f"Anderson (m={depth})" if use_anderson else "No Anderson"
    logger.info(f"\n  Testing: {label}")

    from mfg_pde.alg.numerical.fp_solvers.fp_upwind import FPUpwindSolver
    from mfg_pde.alg.numerical.hjb_solvers.hjb_upwind import HJBUpwindSolver
    from mfg_pde.alg.numerical.mfg_solvers.fixed_point_iterator import FixedPointIterator

    hjb_solver = HJBUpwindSolver(problem)
    fp_solver = FPUpwindSolver(problem)

    solver = FixedPointIterator(
        problem,
        hjb_solver,
        fp_solver,
        thetaUM=0.7,
        use_anderson=use_anderson,
        anderson_depth=depth if use_anderson else 5,
    )

    start = time.time()
    _U, _M, iterations, err_U, err_M = solver.solve(max_iterations=50, tolerance=1e-4)
    solve_time = time.time() - start

    converged = err_U[iterations - 1] < 1e-4 and err_M[iterations - 1] < 1e-4

    logger.info(f"    Time: {solve_time:.3f}s, Iterations: {iterations}, Converged: {converged}")

    return solve_time, iterations, converged


def plot_results(backend_results, anderson_results):
    """Visualize benchmark results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Backend comparison
    ax = axes[0]
    backends = [name for name, (t, _, _) in backend_results.items() if t is not None]
    times = [t for _, (t, _, _) in backend_results.items() if t is not None]

    if backends:
        colors = ["blue", "green", "orange", "red"][: len(backends)]
        bars = ax.bar(backends, times, color=colors, alpha=0.7, edgecolor="black")

        # Add value labels on bars
        for bar, time_val in zip(bars, times, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{time_val:.2f}s",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_ylabel("Solve Time (seconds)", fontsize=12)
        ax.set_title("Backend Performance Comparison", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Anderson acceleration comparison
    ax = axes[1]
    anderson_labels = list(anderson_results.keys())
    anderson_iters = [iters for _, iters, _ in anderson_results.values()]

    if anderson_labels:
        colors_anderson = ["lightcoral", "lightgreen"]
        bars_anderson = ax.bar(anderson_labels, anderson_iters, color=colors_anderson, alpha=0.7, edgecolor="black")

        for bar, iter_val in zip(bars_anderson, anderson_iters, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{iter_val}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_ylabel("Iterations to Convergence", fontsize=12)
        ax.set_title("Anderson Acceleration Impact", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def main():
    """Run acceleration comparison benchmarks."""
    logger.info("=" * 70)
    logger.info("MFG Solver Acceleration Comparison")
    logger.info("=" * 70)

    # Create test problem
    logger.info("\nCreating test problem (LQ-MFG, Nx=100, Nt=50)...")
    problem = create_test_problem()

    # Benchmark backends
    logger.info("\n[1/2] Backend Acceleration Comparison:")
    backend_results = {}

    for backend in ["numpy", "jax", "torch", "numba"]:
        time_val, iters, converged = benchmark_backend(backend, problem)
        backend_results[backend] = (time_val, iters, converged)

    # Benchmark Anderson acceleration
    logger.info("\n[2/2] Anderson Acceleration Comparison:")
    anderson_results = {}

    time_no, iters_no, conv_no = benchmark_anderson(problem, use_anderson=False)
    anderson_results["No Anderson"] = (time_no, iters_no, conv_no)

    time_yes, iters_yes, conv_yes = benchmark_anderson(problem, use_anderson=True, depth=5)
    anderson_results["Anderson (m=5)"] = (time_yes, iters_yes, conv_yes)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Summary:")
    logger.info("=" * 70)

    logger.info("\nBackend Performance:")
    for backend, (t, iters, conv) in backend_results.items():
        if t is not None:
            logger.info(f"  {backend:10s}: {t:6.3f}s, {iters:3d} iterations, converged={conv}")

    logger.info("\nAnderson Acceleration:")
    for label, (t, iters, conv) in anderson_results.items():
        logger.info(f"  {label:15s}: {t:6.3f}s, {iters:3d} iterations, converged={conv}")

    # Speedup analysis
    logger.info("\nSpeedup Analysis:")
    numpy_time = backend_results.get("numpy", (None, None, None))[0]
    if numpy_time:
        for backend, (t, _, _) in backend_results.items():
            if t and backend != "numpy":
                speedup = numpy_time / t
                logger.info(f"  {backend} vs numpy: {speedup:.2f}x")

    no_anderson_iters = anderson_results.get("No Anderson", (None, None, None))[1]
    yes_anderson_iters = anderson_results.get("Anderson (m=5)", (None, None, None))[1]
    if no_anderson_iters and yes_anderson_iters:
        reduction = (no_anderson_iters - yes_anderson_iters) / no_anderson_iters * 100
        logger.info(f"  Anderson iteration reduction: {reduction:.1f}%")

    # Visualize
    logger.info("\nGenerating visualization...")
    fig = plot_results(backend_results, anderson_results)

    output_path = OUTPUT_DIR / "acceleration_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved: {output_path}")

    plt.show()

    logger.info("\n" + "=" * 70)
    logger.info("Acceleration Comparison Complete!")
    logger.info("=" * 70)
    logger.info("\nKey Insights:")
    logger.info("  - JAX: Best for GPU acceleration (CUDA support)")
    logger.info("  - PyTorch: Good for MPS (Apple Silicon) and CUDA")
    logger.info("  - Numba: Moderate CPU speedup with JIT compilation")
    logger.info("  - Anderson: Reduces iterations, especially for stiff problems")


if __name__ == "__main__":
    main()
