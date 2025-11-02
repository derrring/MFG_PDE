#!/usr/bin/env python3
"""
High-Level solve_mfg() Interface Demonstration

Demonstrates the simplified one-line interface for solving MFG problems.

This example shows how solve_mfg() reduces boilerplate from ~30 lines of
solver creation and configuration to a single function call.

Run:
    python examples/basic/solve_mfg_demo.py
"""

from mfg_pde import ExampleMFGProblem, solve_mfg


def demo_simple_usage():
    """Simplest usage with all defaults."""
    print("\n" + "=" * 60)
    print("Demo 1: Simplest Usage (Auto Configuration)")
    print("=" * 60)

    problem = ExampleMFGProblem()

    # One-line solve!
    result = solve_mfg(problem)

    print("\n✓ Solved successfully")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final U error: {result.error_history_U[-1]:.2e}")
    print(f"  Final M error: {result.error_history_M[-1]:.2e}")
    if result.execution_time is not None:
        print(f"  Execution time: {result.execution_time:.3f}s")
    print(f"  Solution shapes: U={result.U.shape}, M={result.M.shape}")


def demo_method_presets():
    """Demonstrate different method presets."""
    print("\n" + "=" * 60)
    print("Demo 2: Method Presets")
    print("=" * 60)

    problem = ExampleMFGProblem()

    # Fast solve (optimized for speed)
    print("\n1. Fast method (optimized for speed)")
    result_fast = solve_mfg(problem, method="fast", verbose=False)
    print(f"   Iterations: {result_fast.iterations}")
    if result_fast.execution_time:
        print(f"   Time: {result_fast.execution_time:.3f}s")
    print(f"   Final error: {result_fast.error_history_U[-1]:.2e}")

    # Accurate solve (high precision)
    print("\n2. Accurate method (high precision)")
    result_accurate = solve_mfg(problem, method="accurate", verbose=False)
    print(f"   Iterations: {result_accurate.iterations}")
    if result_accurate.execution_time:
        print(f"   Time: {result_accurate.execution_time:.3f}s")
    print(f"   Final error: {result_accurate.error_history_U[-1]:.2e}")

    print(f"\n   Accuracy improvement: {result_fast.error_history_U[-1] / result_accurate.error_history_U[-1]:.1f}×")


def demo_custom_parameters():
    """Demonstrate custom parameter overrides."""
    print("\n" + "=" * 60)
    print("Demo 3: Custom Parameters")
    print("=" * 60)

    problem = ExampleMFGProblem()

    # Custom resolution and tolerance
    result = solve_mfg(
        problem,
        method="auto",
        resolution=150,  # Higher resolution
        max_iterations=200,
        tolerance=1e-6,  # Tighter tolerance
        damping_factor=0.3,  # Custom damping
        verbose=False,
    )

    print("\n✓ Solved with custom parameters")
    print("  Resolution: 150 points")
    print("  Max iterations: 200")
    print("  Tolerance: 1e-6")
    print("  Damping: 0.3")
    print("\nResults:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final error: {result.error_history_U[-1]:.2e}")


def demo_comparison_with_factory():
    """Show how solve_mfg() simplifies the workflow."""
    print("\n" + "=" * 60)
    print("Demo 4: Code Simplification")
    print("=" * 60)

    problem = ExampleMFGProblem()

    print("\nTraditional factory approach (~30 lines):")
    print("  from mfg_pde import create_standard_solver")
    print("  from mfg_pde.config import create_fast_config")
    print("  ")
    print("  config = create_fast_config()")
    print("  config.max_iterations = 100")
    print("  config.tolerance_U = 1e-5")
    print("  ")
    print("  solver = create_standard_solver(")
    print("      problem=problem,")
    print("      custom_config=config")
    print("  )")
    print("  result = solver.solve(verbose=True)")

    print("\nNew solve_mfg() approach (1 line):")
    print("  result = solve_mfg(problem, method='fast')")

    # Both produce the same result
    result = solve_mfg(problem, method="fast", verbose=False)
    print("\n✓ Same result in 1 line instead of 30")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("solve_mfg() High-Level Interface Demonstration")
    print("Phase 2.3 Quick Wins")
    print("=" * 60)

    demo_simple_usage()
    demo_method_presets()
    demo_custom_parameters()
    demo_comparison_with_factory()

    print("\n" + "=" * 60)
    print("All demonstrations complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
