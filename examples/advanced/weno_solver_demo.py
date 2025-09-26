#!/usr/bin/env python3
"""
WENO Solver Comprehensive Demo

This example demonstrates the unified WENO solver (HJBWenoSolver) which replaces
the previous separate WENO5 solver. Shows both legacy compatibility and new
advanced features.

Key Features Demonstrated:
1. Legacy compatibility - WENO5 equivalent usage
2. New WENO variants - WENO-Z, WENO-M, WENO-JS
3. Performance comparison across variants
4. Academic-quality benchmarking
5. Easy variant switching interface

Usage:
    python examples/advanced/weno_solver_demo.py

This example supersedes:
- weno5_simple_demo.py (legacy - use weno_variant="weno5")
- weno5_hjb_benchmarking_demo.py (legacy - use weno_family_comparison_demo.py)
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import ExampleMFGProblem
from mfg_pde.alg.hjb_solvers import HJBFDMSolver, HJBWenoSolver
from mfg_pde.utils.logging import configure_research_logging, get_logger

# Configure logging for research session
configure_research_logging("weno_solver_demo", level="INFO")
logger = get_logger(__name__)


def create_test_problem(Nx: int = 64, Nt: int = 32) -> ExampleMFGProblem:
    """Create a test MFG problem for benchmarking."""
    return ExampleMFGProblem(xmin=0.0, xmax=2.0 * np.pi, Nx=Nx, T=1.0, Nt=Nt, sigma=0.3)


def demonstrate_legacy_compatibility():
    """
    Demonstrate that new HJBWenoSolver provides full compatibility with old HJBWeno5Solver.
    """
    logger.info("🔄 Demonstrating Legacy Compatibility")
    logger.info("New HJBWenoSolver(weno_variant='weno5') = Old HJBWeno5Solver")

    problem = create_test_problem()

    # New unified approach (recommended)
    weno_solver = HJBWenoSolver(
        problem=problem,
        weno_variant="weno5",  # Equivalent to old HJBWeno5Solver
        cfl_number=0.3,
        time_integration="tvd_rk3",
    )

    # Test basic functionality
    x = np.linspace(problem.xmin, problem.xmax, problem.Nx + 1)
    u_test = np.sin(2 * np.pi * x / (problem.xmax - problem.xmin))
    m_test = np.ones_like(u_test) / len(u_test)

    start_time = time.time()
    u_result = weno_solver.solve_hjb_step(u_test, m_test, 0.01)
    execution_time = time.time() - start_time

    logger.info(f"✅ WENO5 variant test completed in {execution_time:.4f}s")
    logger.info(f"   Max |u_result| = {np.max(np.abs(u_result)):.4f}")
    logger.info("   Legacy users: simply change HJBWeno5Solver → HJBWenoSolver(weno_variant='weno5')")

    return weno_solver


def demonstrate_advanced_variants():
    """
    Demonstrate new WENO variants not available in legacy solver.
    """
    logger.info("\n🚀 Demonstrating Advanced WENO Variants")

    problem = create_test_problem()
    variants = ["weno5", "weno-z", "weno-m", "weno-js"]

    # Create challenging test case
    x = np.linspace(problem.xmin, problem.xmax, problem.Nx + 1)
    u_test = np.exp(-10 * (x - np.pi) ** 2) + 0.5 * np.exp(-20 * (x - 2 * np.pi) ** 2)
    m_test = np.ones_like(u_test) / len(u_test)

    results = {}

    for variant in variants:
        logger.info(f"Testing {variant.upper()}...")

        solver = HJBWenoSolver(problem=problem, weno_variant=variant, cfl_number=0.3, time_integration="tvd_rk3")

        # Get variant info
        info = solver.get_variant_info()
        logger.info(f"  Description: {info['description']}")

        # Performance test
        start_time = time.time()
        u_result = solver.solve_hjb_step(u_test, m_test, 0.01)
        execution_time = time.time() - start_time

        results[variant] = {"solver": solver, "result": u_result, "time": execution_time, "info": info}

        logger.info(f"  ✅ Completed in {execution_time:.4f}s")
        logger.info(f"  Max |u| = {np.max(np.abs(u_result)):.4f}")
        logger.info(f"  Best for: {info['best_for']}")

    return results


def create_comparison_visualization(results: dict):
    """
    Create visualization comparing different WENO variants.
    """
    logger.info("\n📊 Creating WENO Variants Comparison Plot")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("WENO Solver Variants Comparison\n(Unified HJBWenoSolver Interface)", fontsize=16, fontweight="bold")

    # Plot 1: Solution profiles
    ax = axes[0, 0]
    x = np.linspace(0, 2 * np.pi, len(results["weno5"]["result"]))
    colors = ["blue", "red", "green", "orange"]

    for i, (variant, data) in enumerate(results.items()):
        ax.plot(x, data["result"], color=colors[i], linewidth=2, label=f"{variant.upper()}", alpha=0.8)

    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title("Solution Profiles Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Execution times
    ax = axes[0, 1]
    variants = list(results.keys())
    times = [results[v]["time"] for v in variants]

    bars = ax.bar(range(len(variants)), times, color=colors[: len(variants)], alpha=0.7)
    ax.set_xlabel("WENO Variant")
    ax.set_ylabel("Execution Time (s)")
    ax.set_title("Performance Comparison")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v.upper() for v in variants])

    # Add time labels on bars
    for bar, time_val in zip(bars, times, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height + height * 0.01, f"{time_val:.4f}s", ha="center", va="bottom"
        )

    # Plot 3: Variant characteristics table
    ax = axes[1, 0]
    ax.axis("off")

    table_data = []
    headers = ["Variant", "Description", "Best For"]

    for variant in variants:
        info = results[variant]["info"]
        table_data.append(
            [
                variant.upper(),
                info["description"][:30] + "..." if len(info["description"]) > 30 else info["description"],
                info["best_for"][:35] + "..." if len(info["best_for"]) > 35 else info["best_for"],
            ]
        )

    table = ax.table(cellText=table_data, colLabels=headers, cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    ax.set_title("WENO Variants Overview", pad=20)

    # Plot 4: Solution quality metrics
    ax = axes[1, 1]
    total_variations = [np.sum(np.abs(np.diff(results[v]["result"]))) for v in variants]

    bars = ax.bar(range(len(variants)), total_variations, color=colors[: len(variants)], alpha=0.7)
    ax.set_xlabel("WENO Variant")
    ax.set_ylabel("Total Variation")
    ax.set_title("Solution Quality\n(Lower = Less Oscillatory)")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v.upper() for v in variants])

    plt.tight_layout()
    plt.savefig("weno_solver_comparison.png", dpi=300, bbox_inches="tight")
    logger.info("📁 Saved comparison plot: weno_solver_comparison.png")


def demonstrate_vs_fdm():
    """
    Compare WENO solver against traditional FDM for academic validation.
    """
    logger.info("\n📈 WENO vs Traditional FDM Comparison")

    problem = create_test_problem(Nx=128)  # Higher resolution for accuracy test

    # Create solvers
    weno_solver = HJBWenoSolver(problem, weno_variant="weno5")
    fdm_solver = HJBFDMSolver(problem)

    # Test data with sharp features
    x = np.linspace(problem.xmin, problem.xmax, problem.Nx + 1)
    u_test = np.exp(-5 * (x - np.pi) ** 2) * np.sin(3 * x)  # Sharp + oscillatory
    m_test = np.ones_like(u_test) / len(u_test)

    # WENO test
    logger.info("Testing WENO5 solver...")
    start_time = time.time()
    u_weno = weno_solver.solve_hjb_step(u_test, m_test, 0.005)
    weno_time = time.time() - start_time

    # FDM test
    logger.info("Testing traditional FDM solver...")
    start_time = time.time()
    u_fdm = fdm_solver.solve_hjb_step(u_test, m_test, 0.005)
    fdm_time = time.time() - start_time

    # Analysis
    weno_variation = np.sum(np.abs(np.diff(u_weno)))
    fdm_variation = np.sum(np.abs(np.diff(u_fdm)))

    logger.info("📊 Comparison Results:")
    logger.info(f"  WENO5 execution time: {weno_time:.4f}s")
    logger.info(f"  FDM execution time: {fdm_time:.4f}s")
    logger.info(f"  WENO5 solution quality (total variation): {weno_variation:.4f}")
    logger.info(f"  FDM solution quality (total variation): {fdm_variation:.4f}")
    logger.info(f"  WENO quality advantage: {fdm_variation/weno_variation:.2f}x less oscillatory")


def main():
    """
    Main demonstration function showing unified WENO solver capabilities.
    """
    logger.info("🚀 WENO Solver Comprehensive Demo Starting")
    logger.info("Demonstrates unified HJBWenoSolver replacing legacy HJBWeno5Solver")

    try:
        # 1. Show legacy compatibility
        demonstrate_legacy_compatibility()

        # 2. Show advanced variants
        results = demonstrate_advanced_variants()

        # 3. Create visualization
        create_comparison_visualization(results)

        # 4. Compare against traditional methods
        demonstrate_vs_fdm()

        logger.info("\n🎊 WENO Solver Demo Completed Successfully!")
        logger.info("Key takeaways:")
        logger.info("• HJBWenoSolver provides unified interface for all WENO variants")
        logger.info("• Legacy HJBWeno5Solver users: use HJBWenoSolver(weno_variant='weno5')")
        logger.info("• New variants offer specialized advantages for different problem types")
        logger.info("• Easy switching between variants enables optimal performance")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
