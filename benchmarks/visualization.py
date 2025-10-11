"""
Performance Visualization Tools

Generate plots and reports for benchmark performance trends over time.
Supports matplotlib-based static plots for performance analysis.

Usage:
    from benchmarks.visualization import PerformanceVisualizer

    viz = PerformanceVisualizer(history_dir="benchmarks/history")
    viz.plot_time_trend("hjb_fdm", "LQ-MFG-Small", save_path="trend.png")
    viz.plot_solver_comparison(["LQ-MFG-Small", "Congestion-Small"])
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.performance_tracker import PerformanceTracker


class PerformanceVisualizer:
    """
    Visualize performance trends and comparisons.

    Creates matplotlib plots for:
    - Time trends for individual problems
    - Solver comparisons across problems
    - Error convergence trends
    - Memory usage trends
    """

    def __init__(self, history_dir: str | Path = "benchmarks/history"):
        """
        Initialize visualizer.

        Args:
            history_dir: Directory containing performance history JSON files
        """
        self.tracker = PerformanceTracker(history_dir=history_dir)

    def plot_time_trend(
        self,
        solver_name: str,
        problem_name: str,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> Path | None:
        """
        Plot execution time trend over commits.

        Args:
            solver_name: Name of the solver
            problem_name: Name of the problem
            save_path: Path to save plot (if None, auto-generate)
            show: Display plot interactively

        Returns:
            Path to saved plot, or None if show=True
        """
        history = self.tracker.load_history(solver_name, problem_name)

        if not history:
            print(f"No history found for {solver_name} on {problem_name}")
            return None

        # Extract data
        times = [r.execution_time for r in history]
        iterations = [r.iterations for r in history]
        converged = [r.converged for r in history]
        commits = [r.commit_hash[:7] for r in history]

        # Create figure
        _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot execution time
        x = np.arange(len(times))
        colors = ["green" if c else "red" for c in converged]
        ax1.scatter(x, times, c=colors, alpha=0.6, s=50)
        ax1.plot(x, times, "b-", alpha=0.3, linewidth=1)

        # Add trend line
        if len(times) > 1:
            z = np.polyfit(x, times, 1)
            p = np.poly1d(z)
            ax1.plot(x, p(x), "r--", alpha=0.5, label=f"Trend: {z[0]:.3f}s/commit")
            ax1.legend()

        ax1.set_ylabel("Execution Time (s)")
        ax1.set_title(f"Performance Trend: {solver_name} on {problem_name}")
        ax1.grid(True, alpha=0.3)

        # Plot iterations
        ax2.scatter(x, iterations, c=colors, alpha=0.6, s=50)
        ax2.plot(x, iterations, "b-", alpha=0.3, linewidth=1)
        ax2.set_xlabel("Commit")
        ax2.set_ylabel("Iterations")
        ax2.set_xticks(x[:: max(1, len(x) // 10)])
        ax2.set_xticklabels(commits[:: max(1, len(commits) // 10)], rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None and not show:
            save_path = f"performance_trend_{solver_name}_{problem_name}.png"

        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            return save_path
        elif show:
            plt.show()
            return None
        else:
            plt.close()
            return None

    def plot_solver_comparison(
        self,
        problem_names: list[str],
        solver_name: str = "hjb_fdm",
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> Path | None:
        """
        Compare solver performance across multiple problems.

        Args:
            problem_names: List of problem names to compare
            solver_name: Solver to analyze
            save_path: Path to save plot
            show: Display plot interactively

        Returns:
            Path to saved plot, or None if show=True
        """
        stats_list = []
        valid_problems = []

        for problem_name in problem_names:
            stats = self.tracker.get_statistics(solver_name, problem_name)
            if stats.get("count", 0) > 0:
                stats_list.append(stats)
                valid_problems.append(problem_name)

        if not stats_list:
            print(f"No data found for solver {solver_name}")
            return None

        # Extract data
        mean_times = [s["time_mean"] for s in stats_list]
        std_times = [s["time_std"] for s in stats_list]
        error_means = [s["error_mean"] for s in stats_list]
        convergence_rates = [s["convergence_rate"] * 100 for s in stats_list]

        # Create figure
        _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        x = np.arange(len(valid_problems))
        width = 0.6

        # Execution time comparison
        ax1.bar(x, mean_times, width, yerr=std_times, capsize=5, alpha=0.7)
        ax1.set_ylabel("Execution Time (s)")
        ax1.set_title(f"Average Execution Time - {solver_name}")
        ax1.set_xticks(x)
        ax1.set_xticklabels(valid_problems, rotation=45, ha="right")
        ax1.grid(True, alpha=0.3, axis="y")

        # Convergence error
        ax2.bar(x, error_means, width, alpha=0.7, color="orange")
        ax2.set_ylabel("Final Error")
        ax2.set_title("Average Convergence Error")
        ax2.set_xticks(x)
        ax2.set_xticklabels(valid_problems, rotation=45, ha="right")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3, axis="y")

        # Convergence rate
        ax3.bar(x, convergence_rates, width, alpha=0.7, color="green")
        ax3.set_ylabel("Success Rate (%)")
        ax3.set_title("Convergence Success Rate")
        ax3.set_xticks(x)
        ax3.set_xticklabels(valid_problems, rotation=45, ha="right")
        ax3.set_ylim(0, 105)
        ax3.grid(True, alpha=0.3, axis="y")

        # Statistics table
        ax4.axis("off")
        table_data = []
        for i, problem in enumerate(valid_problems):
            table_data.append(
                [
                    problem,
                    f"{mean_times[i]:.2f}s",
                    f"{error_means[i]:.2e}",
                    f"{convergence_rates[i]:.0f}%",
                ]
            )

        table = ax4.table(
            cellText=table_data,
            colLabels=["Problem", "Avg Time", "Avg Error", "Success"],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        plt.suptitle(f"Performance Comparison: {solver_name}", fontsize=14, y=0.995)
        plt.tight_layout()

        if save_path is None and not show:
            save_path = f"solver_comparison_{solver_name}.png"

        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            return save_path
        elif show:
            plt.show()
            return None
        else:
            plt.close()
            return None

    def generate_summary_report(
        self,
        output_path: str | Path = "benchmark_summary.txt",
    ) -> Path:
        """
        Generate text summary report of all benchmark results.

        Args:
            output_path: Path to save report

        Returns:
            Path to saved report
        """
        from benchmarks.standard_problems import get_all_problems

        output_path = Path(output_path)
        lines = []

        lines.append("=" * 70)
        lines.append("BENCHMARK PERFORMANCE SUMMARY")
        lines.append("=" * 70)
        lines.append("")

        for problem in get_all_problems():
            stats = self.tracker.get_statistics(problem.solver_type, problem.name)

            if stats.get("count", 0) == 0:
                continue

            lines.append(f"\n{problem.name} ({problem.category.upper()})")
            lines.append("-" * 70)
            lines.append(f"  Solver: {problem.solver_type}")
            lines.append(f"  Runs: {stats['count']}")
            lines.append("  Execution Time:")
            lines.append(f"    Mean: {stats['time_mean']:.3f}s")
            lines.append(f"    Std:  {stats['time_std']:.3f}s")
            lines.append(f"    Min:  {stats['time_min']:.3f}s")
            lines.append(f"    Max:  {stats['time_max']:.3f}s")
            lines.append("  Convergence Error:")
            lines.append(f"    Mean: {stats['error_mean']:.2e}")
            lines.append(f"    Std:  {stats['error_std']:.2e}")
            lines.append(f"  Iterations (avg): {stats['iterations_mean']:.1f}")
            lines.append(f"  Success Rate: {stats['convergence_rate'] * 100:.1f}%")

            # Check if within expected range
            expected_min, expected_max = problem.expected_time_range
            if stats["time_mean"] > expected_max:
                lines.append(f"  ⚠️  SLOWER than expected ({expected_max:.1f}s)")
            elif stats["time_mean"] < expected_min:
                lines.append(f"  ✓ Faster than expected ({expected_min:.1f}s)")
            else:
                lines.append("  ✓ Within expected range")

        report_text = "\n".join(lines)
        output_path.write_text(report_text)

        return output_path


if __name__ == "__main__":
    # Example usage

    viz = PerformanceVisualizer()

    # Generate summary report
    report_path = viz.generate_summary_report("benchmark_summary.txt")
    print(f"Summary report: {report_path}")

    # Plot available trends
    from benchmarks.standard_problems import get_all_problems

    for problem in get_all_problems()[:2]:  # First 2 problems as demo
        history = viz.tracker.load_history(problem.solver_type, problem.name)
        if history:
            viz_path = viz.plot_time_trend(
                problem.solver_type,
                problem.name,
                save_path=f"trend_{problem.name}.png",
            )
            if viz_path:
                print(f"Trend plot: {viz_path}")

    # Comparison plot
    small_problems = [p.name for p in get_all_problems() if p.category == "small"]
    if small_problems:
        comp_path = viz.plot_solver_comparison(
            small_problems,
            solver_name="hjb_fdm",
            save_path="solver_comparison.png",
        )
        if comp_path:
            print(f"Comparison plot: {comp_path}")
