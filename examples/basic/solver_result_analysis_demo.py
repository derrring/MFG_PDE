"""
Demonstration of SolverResult analysis and visualization features.

This example shows how to use the new analysis methods added to SolverResult:
- analyze_convergence() for detailed convergence diagnostics
- plot_convergence() for publication-quality plots
- compare_to() for benchmarking solvers
- export_summary() for reports and documentation

Part of Issue #127 implementation.
"""

from pathlib import Path

import numpy as np

from mfg_pde.utils.solver_result import SolverResult

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "basic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_sample_result(
    name: str,
    converged: bool = True,
    iterations: int = 50,
    Nx: int = 50,
    Nt: int = 30,
) -> SolverResult:
    """Create a sample solver result for demonstration."""
    # Create sample solution arrays
    U = np.random.rand(Nt + 1, Nx + 1)
    M = np.random.rand(Nt + 1, Nx + 1)

    # Create realistic error histories (exponential decay)
    if converged:
        # Converging error history
        error_U = 1.0 * np.exp(-0.1 * np.arange(iterations))
        error_M = 0.8 * np.exp(-0.12 * np.arange(iterations))
    else:
        # Stagnating error history
        error_U = 0.1 * np.ones(iterations)
        error_M = 0.08 * np.ones(iterations)

    return SolverResult(
        U=U,
        M=M,
        iterations=iterations,
        error_history_U=error_U,
        error_history_M=error_M,
        solver_name=name,
        converged=converged,
        execution_time=np.random.uniform(0.5, 2.0),
        metadata={"Nx": Nx, "Nt": Nt},
    )


def demo_analyze_convergence():
    """Demonstrate convergence analysis."""
    print("=" * 60)
    print("Demo 1: Convergence Analysis")
    print("=" * 60)

    # Create converged result
    result = create_sample_result("Fast Solver", converged=True)

    # Analyze convergence
    analysis = result.analyze_convergence()

    print(f"\nSolver: {result.solver_name}")
    print(f"Status: {'✅ Converged' if analysis.converged else '❌ Not converged'}")
    print(f"Iterations: {analysis.iterations}")
    print(
        f"Convergence Rate: {analysis.convergence_rate:.4f}" if analysis.convergence_rate else "Convergence Rate: N/A"
    )
    print(f"Final Error (U): {analysis.final_error_U:.6e}")
    print(f"Final Error (M): {analysis.final_error_M:.6e}")
    print(f"Error Reduction (U): {analysis.error_reduction_ratio_U:.1f}x")
    print(f"Error Reduction (M): {analysis.error_reduction_ratio_M:.1f}x")

    if analysis.stagnation_detected:
        print("⚠️  Warning: Stagnation detected")
    if analysis.oscillation_detected:
        print("⚠️  Warning: Oscillation detected")

    print(f"\nAnalysis object: {analysis}")


def demo_plot_convergence():
    """Demonstrate convergence plotting."""
    print("\n" + "=" * 60)
    print("Demo 2: Convergence Plotting")
    print("=" * 60)

    # Create result
    result = create_sample_result("Accurate Solver", converged=True, iterations=100)

    # Generate plot
    save_path = OUTPUT_DIR / "convergence_plot.png"
    print(f"\nGenerating convergence plot: {save_path}")

    result.plot_convergence(
        save_path=save_path,
        show=False,  # Don't show interactively
        figsize=(10, 6),
        dpi=150,
        log_scale=True,
    )

    print(f"✅ Plot saved to: {save_path}")


def demo_compare_results():
    """Demonstrate result comparison."""
    print("\n" + "=" * 60)
    print("Demo 3: Result Comparison")
    print("=" * 60)

    # Create two different solver results
    result1 = create_sample_result("Fast Solver", converged=True, iterations=30)
    result2 = create_sample_result("Accurate Solver", converged=True, iterations=100)

    # Compare them
    comparison = result1.compare_to(result2)

    print(f"\nComparing: {comparison.result1_name} vs {comparison.result2_name}")
    print(f"Solution Difference (L2): {comparison.solution_diff_l2:.6e}")
    print(f"Solution Difference (L∞): {comparison.solution_diff_linf:.6e}")
    print(f"Iteration Difference: {comparison.iterations_diff}")
    print(f"Time Difference: {abs(comparison.time_diff):.3f}s" if comparison.time_diff else "Time Difference: N/A")
    print(f"Both Converged: {'✅ Yes' if comparison.converged_both else '❌ No'}")
    print(f"Faster Solver: {comparison.faster_solver}")
    print(f"More Accurate Solver: {comparison.more_accurate_solver}")

    print(f"\nComparison object: {comparison}")


def demo_export_summary():
    """Demonstrate summary export."""
    print("\n" + "=" * 60)
    print("Demo 4: Summary Export")
    print("=" * 60)

    # Create result
    result = create_sample_result("Demo Solver", converged=True)

    # Export markdown summary
    md_path = OUTPUT_DIR / "solver_summary.md"
    print(f"\nExporting markdown summary: {md_path}")
    result.export_summary(output_format="markdown", filename=md_path)
    print(f"✅ Markdown summary saved to: {md_path}")

    # Export LaTeX summary
    latex_path = OUTPUT_DIR / "solver_summary.tex"
    print(f"\nExporting LaTeX summary: {latex_path}")
    result.export_summary(output_format="latex", filename=latex_path)
    print(f"✅ LaTeX summary saved to: {latex_path}")

    # Print markdown to console
    print("\nMarkdown Summary Preview:")
    print("-" * 60)
    md_content = result.export_summary(output_format="markdown")
    # Print first 20 lines
    print("\n".join(md_content.split("\n")[:20]))
    print("...")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("Solver Result Analysis Tools Demonstration")
    print("=" * 60)

    demo_analyze_convergence()
    demo_plot_convergence()
    demo_compare_results()
    demo_export_summary()

    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
