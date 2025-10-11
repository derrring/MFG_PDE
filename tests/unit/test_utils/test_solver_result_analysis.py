"""
Tests for SolverResult analysis methods (Issue #127).

Tests the new analysis and visualization features:
- analyze_convergence()
- plot_convergence()
- compare_to()
- export_summary()
"""

from pathlib import Path

import pytest

import numpy as np

from mfg_pde.utils.solver_result import (
    ComparisonReport,
    ConvergenceAnalysis,
    SolverResult,
)


@pytest.fixture
def converged_result():
    """Create a sample converged result."""
    iterations = 50
    Nx, Nt = 30, 20

    U = np.random.rand(Nt + 1, Nx + 1)
    M = np.random.rand(Nt + 1, Nx + 1)

    # Exponentially decaying errors (converging)
    error_U = 1.0 * np.exp(-0.1 * np.arange(iterations))
    error_M = 0.8 * np.exp(-0.12 * np.arange(iterations))

    return SolverResult(
        U=U,
        M=M,
        iterations=iterations,
        error_history_U=error_U,
        error_history_M=error_M,
        solver_name="Test Solver",
        converged=True,
        execution_time=1.5,
        metadata={"Nx": Nx, "Nt": Nt},
    )


@pytest.fixture
def stagnating_result():
    """Create a result with stagnating convergence."""
    iterations = 50
    Nx, Nt = 30, 20

    U = np.random.rand(Nt + 1, Nx + 1)
    M = np.random.rand(Nt + 1, Nx + 1)

    # Constant errors (stagnating)
    error_U = 0.1 * np.ones(iterations)
    error_M = 0.08 * np.ones(iterations)

    return SolverResult(
        U=U,
        M=M,
        iterations=iterations,
        error_history_U=error_U,
        error_history_M=error_M,
        solver_name="Stagnating Solver",
        converged=False,
        execution_time=2.0,
    )


@pytest.fixture
def oscillating_result():
    """Create a result with oscillating errors."""
    iterations = 50
    Nx, Nt = 30, 20

    U = np.random.rand(Nt + 1, Nx + 1)
    M = np.random.rand(Nt + 1, Nx + 1)

    # Strongly oscillating errors (alternating up/down pattern)
    error_U = 0.1 * np.ones(iterations)
    error_M = 0.08 * np.ones(iterations)
    # Add strong oscillations: +20%, -20%, +20%, -20%, ...
    for i in range(iterations):
        if i % 2 == 0:
            error_U[i] *= 1.2
            error_M[i] *= 1.2
        else:
            error_U[i] *= 0.8
            error_M[i] *= 0.8

    return SolverResult(
        U=U,
        M=M,
        iterations=iterations,
        error_history_U=error_U,
        error_history_M=error_M,
        solver_name="Oscillating Solver",
        converged=False,
        execution_time=1.8,
    )


# ===== analyze_convergence() Tests =====


class TestAnalyzeConvergence:
    """Tests for analyze_convergence() method."""

    def test_analyze_convergence_returns_correct_type(self, converged_result):
        """Test that analyze_convergence returns ConvergenceAnalysis."""
        analysis = converged_result.analyze_convergence()
        assert isinstance(analysis, ConvergenceAnalysis)

    def test_convergence_status_matches_result(self, converged_result):
        """Test that analysis converged status matches result."""
        analysis = converged_result.analyze_convergence()
        assert analysis.converged == converged_result.converged

    def test_iterations_match(self, converged_result):
        """Test that analysis iterations match result."""
        analysis = converged_result.analyze_convergence()
        assert analysis.iterations == converged_result.iterations

    def test_convergence_rate_estimated(self, converged_result):
        """Test that convergence rate is estimated for converging results."""
        analysis = converged_result.analyze_convergence()
        assert analysis.convergence_rate is not None
        assert analysis.convergence_rate > 0  # Should be positive for decreasing errors

    def test_convergence_rate_reasonable_range(self, converged_result):
        """Test that estimated rate is in reasonable range."""
        analysis = converged_result.analyze_convergence()
        # For exponential decay with rate 0.1, should estimate close to 0.1
        assert 0.05 < analysis.convergence_rate < 0.15

    def test_stagnation_detected(self, stagnating_result):
        """Test that stagnation is detected in flat error history."""
        analysis = stagnating_result.analyze_convergence()
        assert analysis.stagnation_detected  # Use truthy check instead of 'is True'

    def test_oscillation_detected(self, oscillating_result):
        """Test that oscillation is detected in oscillating errors."""
        analysis = oscillating_result.analyze_convergence()
        assert analysis.oscillation_detected  # Use truthy check instead of 'is True'

    def test_no_false_stagnation_on_converging(self, converged_result):
        """Test that converging results don't trigger stagnation."""
        analysis = converged_result.analyze_convergence()
        assert not analysis.stagnation_detected  # Use falsy check instead of 'is False'

    def test_error_reduction_ratio_calculated(self, converged_result):
        """Test that error reduction ratios are calculated."""
        analysis = converged_result.analyze_convergence()
        assert analysis.error_reduction_ratio_U > 1.0  # Should have reduced
        assert analysis.error_reduction_ratio_M > 1.0

    def test_error_reduction_ratio_large_for_exponential(self, converged_result):
        """Test that exponential decay gives large reduction ratio."""
        analysis = converged_result.analyze_convergence()
        # For 50 iterations with rate 0.1, reduction should be ~exp(5) ≈ 148
        assert analysis.error_reduction_ratio_U > 100

    def test_final_errors_match_result(self, converged_result):
        """Test that final errors match the result."""
        analysis = converged_result.analyze_convergence()
        assert analysis.final_error_U == converged_result.final_error_U
        assert analysis.final_error_M == converged_result.final_error_M

    def test_analysis_repr_contains_status(self, converged_result):
        """Test that __repr__ contains convergence status."""
        analysis = converged_result.analyze_convergence()
        repr_str = repr(analysis)
        assert "CONVERGED" in repr_str or "NOT CONVERGED" in repr_str

    def test_analysis_repr_contains_rate(self, converged_result):
        """Test that __repr__ contains convergence rate."""
        analysis = converged_result.analyze_convergence()
        repr_str = repr(analysis)
        assert "rate=" in repr_str

    def test_insufficient_data_returns_none_rate(self):
        """Test that insufficient data returns None for convergence rate."""
        # Create result with only 2 iterations
        U = np.random.rand(10, 10)
        M = np.random.rand(10, 10)
        error_U = np.array([1.0, 0.5])
        error_M = np.array([0.8, 0.4])

        result = SolverResult(
            U=U,
            M=M,
            iterations=2,
            error_history_U=error_U,
            error_history_M=error_M,
            solver_name="Insufficient Data",
            converged=False,
        )

        analysis = result.analyze_convergence()
        assert analysis.convergence_rate is None


# ===== plot_convergence() Tests =====


class TestPlotConvergence:
    """Tests for plot_convergence() method."""

    def test_plot_convergence_returns_figure(self, converged_result):
        """Test that plot_convergence returns a matplotlib figure."""
        import matplotlib.pyplot as plt

        fig = converged_result.plot_convergence(show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_saves_to_file(self, converged_result, tmp_path):
        """Test that plot is saved when save_path is provided."""
        import matplotlib.pyplot as plt

        save_path = tmp_path / "test_convergence.png"
        converged_result.plot_convergence(save_path=save_path, show=False)

        assert save_path.exists()
        plt.close("all")

    def test_plot_with_log_scale(self, converged_result):
        """Test plotting with log scale."""
        import matplotlib.pyplot as plt

        fig = converged_result.plot_convergence(show=False, log_scale=True)
        ax = fig.axes[0]
        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_plot_with_linear_scale(self, converged_result):
        """Test plotting with linear scale."""
        import matplotlib.pyplot as plt

        fig = converged_result.plot_convergence(show=False, log_scale=False)
        ax = fig.axes[0]
        assert ax.get_yscale() == "linear"
        plt.close(fig)

    def test_plot_custom_figsize(self, converged_result):
        """Test plotting with custom figure size."""
        import matplotlib.pyplot as plt

        figsize = (12, 8)
        fig = converged_result.plot_convergence(show=False, figsize=figsize)
        assert fig.get_size_inches()[0] == pytest.approx(figsize[0], rel=0.1)
        assert fig.get_size_inches()[1] == pytest.approx(figsize[1], rel=0.1)
        plt.close(fig)

    def test_plot_has_title(self, converged_result):
        """Test that plot has a title."""
        import matplotlib.pyplot as plt

        fig = converged_result.plot_convergence(show=False)
        ax = fig.axes[0]
        assert ax.get_title() != ""
        assert converged_result.solver_name in ax.get_title()
        plt.close(fig)

    def test_plot_has_labels(self, converged_result):
        """Test that plot has axis labels."""
        import matplotlib.pyplot as plt

        fig = converged_result.plot_convergence(show=False)
        ax = fig.axes[0]
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)

    def test_plot_has_legend(self, converged_result):
        """Test that plot has a legend."""
        import matplotlib.pyplot as plt

        fig = converged_result.plot_convergence(show=False)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)

    def test_plot_multiple_calls_dont_interfere(self, converged_result):
        """Test that multiple plot calls don't interfere with each other."""
        import matplotlib.pyplot as plt

        fig1 = converged_result.plot_convergence(show=False)
        fig2 = converged_result.plot_convergence(show=False)

        assert fig1 is not fig2
        plt.close("all")


# ===== compare_to() Tests =====


class TestCompareTo:
    """Tests for compare_to() method."""

    def test_compare_to_returns_comparison_report(self, converged_result, stagnating_result):
        """Test that compare_to returns ComparisonReport."""
        comparison = converged_result.compare_to(stagnating_result)
        assert isinstance(comparison, ComparisonReport)

    def test_comparison_has_all_fields(self, converged_result, stagnating_result):
        """Test that comparison report has all required fields."""
        comparison = converged_result.compare_to(stagnating_result)

        assert hasattr(comparison, "solution_diff_l2")
        assert hasattr(comparison, "solution_diff_linf")
        assert hasattr(comparison, "iterations_diff")
        assert hasattr(comparison, "time_diff")
        assert hasattr(comparison, "converged_both")
        assert hasattr(comparison, "faster_solver")
        assert hasattr(comparison, "more_accurate_solver")

    def test_solution_diff_is_positive(self, converged_result, stagnating_result):
        """Test that solution differences are positive."""
        comparison = converged_result.compare_to(stagnating_result)
        assert comparison.solution_diff_l2 >= 0
        assert comparison.solution_diff_linf >= 0

    def test_iterations_diff_calculated(self, converged_result, stagnating_result):
        """Test that iteration difference is calculated correctly."""
        comparison = converged_result.compare_to(stagnating_result)
        expected_diff = converged_result.iterations - stagnating_result.iterations
        assert comparison.iterations_diff == expected_diff

    def test_time_diff_calculated(self, converged_result, stagnating_result):
        """Test that time difference is calculated."""
        comparison = converged_result.compare_to(stagnating_result)
        assert comparison.time_diff is not None

    def test_converged_both_true_when_both_converged(self, converged_result):
        """Test converged_both is True when both results converged."""
        # Create another converged result
        result2 = SolverResult(
            U=converged_result.U,
            M=converged_result.M,
            iterations=converged_result.iterations,
            error_history_U=converged_result.error_history_U,
            error_history_M=converged_result.error_history_M,
            solver_name="Solver 2",
            converged=True,
        )

        comparison = converged_result.compare_to(result2)
        assert comparison.converged_both is True

    def test_converged_both_false_when_one_failed(self, converged_result, stagnating_result):
        """Test converged_both is False when one didn't converge."""
        comparison = converged_result.compare_to(stagnating_result)
        assert comparison.converged_both is False

    def test_faster_solver_identified(self, converged_result, stagnating_result):
        """Test that faster solver is identified."""
        comparison = converged_result.compare_to(stagnating_result)
        assert comparison.faster_solver in [converged_result.solver_name, stagnating_result.solver_name, "tie"]

    def test_more_accurate_solver_identified(self, converged_result, stagnating_result):
        """Test that more accurate solver is identified."""
        comparison = converged_result.compare_to(stagnating_result)
        assert comparison.more_accurate_solver in [
            converged_result.solver_name,
            stagnating_result.solver_name,
            "tie",
        ]

    def test_comparison_repr_contains_solver_names(self, converged_result, stagnating_result):
        """Test that __repr__ contains both solver names."""
        comparison = converged_result.compare_to(stagnating_result)
        repr_str = repr(comparison)
        assert converged_result.solver_name in repr_str
        assert stagnating_result.solver_name in repr_str

    def test_comparison_repr_contains_metrics(self, converged_result, stagnating_result):
        """Test that __repr__ contains key metrics."""
        comparison = converged_result.compare_to(stagnating_result)
        repr_str = repr(comparison)
        assert "L2=" in repr_str
        assert "L∞=" in repr_str or "Linf=" in repr_str

    def test_compare_raises_on_shape_mismatch(self, converged_result):
        """Test that comparison raises error on shape mismatch."""
        # Create result with different shape
        different_shape = SolverResult(
            U=np.random.rand(10, 10),
            M=np.random.rand(10, 10),
            iterations=10,
            error_history_U=np.ones(10),
            error_history_M=np.ones(10),
            solver_name="Different Shape",
            converged=False,
        )

        with pytest.raises(ValueError, match="shapes don't match"):
            converged_result.compare_to(different_shape)

    def test_self_comparison_gives_zero_diff(self, converged_result):
        """Test that comparing result to itself gives zero difference."""
        comparison = converged_result.compare_to(converged_result)
        assert comparison.solution_diff_l2 == pytest.approx(0.0)
        assert comparison.solution_diff_linf == pytest.approx(0.0)
        assert comparison.iterations_diff == 0


# ===== export_summary() Tests =====


class TestExportSummary:
    """Tests for export_summary() method."""

    def test_export_markdown_returns_string(self, converged_result):
        """Test that export_summary returns a string for markdown."""
        summary = converged_result.export_summary(output_format="markdown")
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_export_latex_returns_string(self, converged_result):
        """Test that export_summary returns a string for LaTeX."""
        summary = converged_result.export_summary(output_format="latex")
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_markdown_contains_solver_name(self, converged_result):
        """Test that markdown summary contains solver name."""
        summary = converged_result.export_summary(output_format="markdown")
        assert converged_result.solver_name in summary

    def test_markdown_contains_convergence_status(self, converged_result):
        """Test that markdown summary contains convergence status."""
        summary = converged_result.export_summary(output_format="markdown")
        assert "Converged" in summary or "converged" in summary

    def test_markdown_contains_iterations(self, converged_result):
        """Test that markdown summary contains iteration count."""
        summary = converged_result.export_summary(output_format="markdown")
        assert str(converged_result.iterations) in summary

    def test_markdown_contains_errors(self, converged_result):
        """Test that markdown summary contains error values."""
        summary = converged_result.export_summary(output_format="markdown")
        # Should contain scientific notation errors
        assert "e-" in summary or "e+" in summary

    def test_latex_contains_begin_table(self, converged_result):
        """Test that LaTeX summary contains table environment."""
        summary = converged_result.export_summary(output_format="latex")
        assert "\\begin{table}" in summary
        assert "\\end{table}" in summary

    def test_latex_contains_tabular(self, converged_result):
        """Test that LaTeX summary contains tabular environment."""
        summary = converged_result.export_summary(output_format="latex")
        assert "\\begin{tabular}" in summary
        assert "\\end{tabular}" in summary

    def test_latex_contains_toprule(self, converged_result):
        """Test that LaTeX summary uses booktabs rules."""
        summary = converged_result.export_summary(output_format="latex")
        assert "\\toprule" in summary
        assert "\\midrule" in summary
        assert "\\bottomrule" in summary

    def test_export_saves_to_file_markdown(self, converged_result, tmp_path):
        """Test that markdown export saves to file."""
        filepath = tmp_path / "summary.md"
        converged_result.export_summary(output_format="markdown", filename=filepath)
        assert filepath.exists()
        content = filepath.read_text()
        assert len(content) > 0

    def test_export_saves_to_file_latex(self, converged_result, tmp_path):
        """Test that LaTeX export saves to file."""
        filepath = tmp_path / "summary.tex"
        converged_result.export_summary(output_format="latex", filename=filepath)
        assert filepath.exists()
        content = filepath.read_text()
        assert len(content) > 0

    def test_export_raises_on_invalid_format(self, converged_result):
        """Test that export raises error on unsupported format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            converged_result.export_summary(output_format="invalid")

    def test_export_handles_pathlib_path(self, converged_result, tmp_path):
        """Test that export works with pathlib Path objects."""
        filepath = Path(tmp_path) / "summary.md"
        converged_result.export_summary(output_format="markdown", filename=filepath)
        assert filepath.exists()

    def test_export_includes_execution_time_if_present(self, converged_result):
        """Test that execution time is included when present."""
        summary = converged_result.export_summary(output_format="markdown")
        assert "Execution Time" in summary or "execution" in summary.lower()

    def test_export_includes_metadata_if_present(self, converged_result):
        """Test that metadata is included in export."""
        summary = converged_result.export_summary(output_format="markdown")
        # Check for metadata keys
        assert "Nx" in summary or "metadata" in summary.lower()
