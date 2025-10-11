"""
Standardized result objects for MFG_PDE solvers.

This module provides structured result objects that replace tuple returns,
improving code readability, IDE support, and API maintainability.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


@dataclass(init=False)
class SolverResult:
    """
    Standardized result object for MFG solver outputs.

    This class provides structured access to solver results while maintaining
    backward compatibility with tuple unpacking for existing code.

    Attributes:
        U: Control/value function solution array (Nt+1, Nx+1)
        M: Density/distribution function solution array (Nt+1, Nx+1)
        iterations: Number of iterations performed
        error_history_U: History of U convergence errors
        error_history_M: History of M convergence errors
        solver_name: Name/type of solver used
        converged: Whether convergence was achieved
        execution_time: Total solve time in seconds
        metadata: Additional solver-specific information
    """

    U: NDArray[np.floating]
    M: NDArray[np.floating]
    iterations: int
    error_history_U: NDArray[np.floating]
    error_history_M: NDArray[np.floating]
    solver_name: str = "Unknown Solver"
    converged: bool = False
    execution_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        U: NDArray[np.floating],
        M: NDArray[np.floating],
        iterations: int,
        error_history_U: NDArray[np.floating],
        error_history_M: NDArray[np.floating],
        solver_name: str = "Unknown Solver",
        converged: bool = False,
        execution_time: float | None = None,
        metadata: dict[str, Any] | None = None,
        # Deprecated parameters
        convergence_achieved: bool | None = None,
        convergence_reason: str | None = None,
        diagnostics: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize SolverResult with support for deprecated parameters.

        Args:
            U: Control/value function solution array
            M: Density/distribution function solution array
            iterations: Number of iterations performed
            error_history_U: History of U convergence errors
            error_history_M: History of M convergence errors
            solver_name: Name/type of solver used
            converged: Whether convergence was achieved
            execution_time: Total solve time in seconds
            metadata: Additional solver-specific information
            convergence_achieved: DEPRECATED - Use converged instead
            convergence_reason: DEPRECATED - Add to metadata instead
            diagnostics: DEPRECATED - Add to metadata instead
        """
        # Handle deprecated 'convergence_achieved' parameter
        if convergence_achieved is not None:
            warnings.warn(
                "Parameter 'convergence_achieved' is deprecated, use 'converged' instead",
                DeprecationWarning,
                stacklevel=2,
            )
            converged = convergence_achieved

        # Handle deprecated 'convergence_reason' parameter
        if convergence_reason is not None:
            warnings.warn(
                "Parameter 'convergence_reason' is deprecated, add to metadata instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if metadata is None:
                metadata = {}
            metadata["convergence_reason"] = convergence_reason

        # Handle deprecated 'diagnostics' parameter
        if diagnostics is not None:
            warnings.warn(
                "Parameter 'diagnostics' is deprecated, add to metadata instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if metadata is None:
                metadata = {}
            metadata.update(diagnostics)

        # Handle any other unexpected kwargs
        if kwargs:
            warnings.warn(
                f"Unknown parameters: {list(kwargs.keys())}",
                DeprecationWarning,
                stacklevel=2,
            )

        # Initialize dataclass fields
        self.U = U
        self.M = M
        self.iterations = iterations
        self.error_history_U = error_history_U
        self.error_history_M = error_history_M
        self.solver_name = solver_name
        self.converged = converged
        self.execution_time = execution_time
        self.metadata = metadata if metadata is not None else {}

        # Validate
        self.__post_init__()

    def __post_init__(self):
        """Validate result data after initialization."""
        if self.U.shape != self.M.shape:
            raise ValueError(f"U and M shapes must match: U{self.U.shape} vs M{self.M.shape}")

        if len(self.error_history_U) != len(self.error_history_M):
            raise ValueError("Error history arrays must have same length")

        if self.iterations != len(self.error_history_U):
            # Allow for cases where arrays are pre-allocated but not fully used
            if self.iterations <= len(self.error_history_U):
                # Trim arrays to actual iterations used
                self.error_history_U = self.error_history_U[: self.iterations]
                self.error_history_M = self.error_history_M[: self.iterations]
            else:
                raise ValueError(
                    f"Iterations ({self.iterations}) exceeds error history length ({len(self.error_history_U)})"
                )

    # Backward compatibility: allow tuple-like unpacking
    def __iter__(self):
        """Enable tuple-like unpacking: U, M, iterations, err_u, err_m = result"""
        yield self.U
        yield self.M
        yield self.iterations
        yield self.error_history_U
        yield self.error_history_M

    def __len__(self):
        """Return standard tuple length for compatibility."""
        return 5

    def __getitem__(self, index):
        """Enable indexing like a tuple for backward compatibility."""
        tuple_representation = (
            self.U,
            self.M,
            self.iterations,
            self.error_history_U,
            self.error_history_M,
        )
        return tuple_representation[index]

    # Deprecated property for backward compatibility
    @property
    def convergence_achieved(self) -> bool:
        """DEPRECATED: Use converged instead."""
        warnings.warn(
            "Property 'convergence_achieved' is deprecated, use 'converged' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.converged

    # Convenience properties
    @property
    def final_error_U(self) -> float:
        """Get the final convergence error for U."""
        return float(self.error_history_U[-1]) if len(self.error_history_U) > 0 else float("inf")

    @property
    def final_error_M(self) -> float:
        """Get the final convergence error for M."""
        return float(self.error_history_M[-1]) if len(self.error_history_M) > 0 else float("inf")

    @property
    def max_error(self) -> float:
        """Get the maximum of the final errors."""
        return max(self.final_error_U, self.final_error_M)

    @property
    def solution_shape(self) -> tuple[int, ...]:
        """Get the shape of the solution arrays."""
        return self.U.shape

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "U": self.U,
            "M": self.M,
            "iterations": self.iterations,
            "error_history_U": self.error_history_U,
            "error_history_M": self.error_history_M,
            "solver_name": self.solver_name,
            "converged": self.converged,
            "execution_time": self.execution_time,
            "final_error_U": self.final_error_U,
            "final_error_M": self.final_error_M,
            "max_error": self.max_error,
            "solution_shape": self.solution_shape,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        """String representation of the result."""
        convergence_status = "SUCCESS:" if self.converged else "WARNING:"
        time_str = f", {self.execution_time:.3f}s" if self.execution_time else ""

        return (
            f"SolverResult({self.solver_name}: {convergence_status} "
            f"{self.iterations} iters, errors U={self.final_error_U:.2e} "
            f"M={self.final_error_M:.2e}{time_str})"
        )

    def save_hdf5(
        self,
        filename: str | Path,
        *,
        compression: str = "gzip",
        compression_opts: int = 4,
        x_grid: NDArray | None = None,
        t_grid: NDArray | None = None,
    ) -> None:
        """
        Save solver result to HDF5 file.

        Args:
            filename: Output HDF5 file path
            compression: Compression algorithm ('gzip', 'lzf', None)
            compression_opts: Compression level (1-9 for gzip)
            x_grid: Optional spatial grid coordinates
            t_grid: Optional temporal grid coordinates

        Raises:
            ImportError: If h5py not installed

        Example:
            >>> result = solver.solve()
            >>> result.save_hdf5('solution.h5')
        """
        from mfg_pde.utils.io.hdf5_utils import save_solution

        # Prepare metadata from result fields
        metadata = {
            "solver_name": self.solver_name,
            "converged": self.converged,
            "iterations": self.iterations,
            "error_history_U": self.error_history_U,
            "error_history_M": self.error_history_M,
            "final_error_U": self.final_error_U,
            "final_error_M": self.final_error_M,
        }

        if self.execution_time is not None:
            metadata["execution_time"] = self.execution_time

        # Add user metadata
        metadata.update(self.metadata)

        save_solution(
            self.U,
            self.M,
            metadata,
            filename,
            compression=compression,
            compression_opts=compression_opts,
            x_grid=x_grid,
            t_grid=t_grid,
        )

    @classmethod
    def load_hdf5(cls, filename: str | Path) -> SolverResult:
        """
        Load solver result from HDF5 file.

        Args:
            filename: HDF5 file path to load

        Returns:
            SolverResult object reconstructed from file

        Raises:
            ImportError: If h5py not installed
            FileNotFoundError: If file doesn't exist
            KeyError: If file missing required datasets

        Example:
            >>> result = SolverResult.load_hdf5('solution.h5')
            >>> print(f"Converged: {result.converged}")
        """
        from mfg_pde.utils.io.hdf5_utils import load_solution

        U, M, metadata = load_solution(filename)

        # Extract standard fields
        iterations = metadata.pop("iterations", 0)
        error_history_U = metadata.pop("error_history_U", np.array([]))
        error_history_M = metadata.pop("error_history_M", np.array([]))
        solver_name = metadata.pop("solver_name", "Unknown Solver")
        converged = metadata.pop("converged", False)
        execution_time = metadata.pop("execution_time", None)

        # Remove derived fields that will be recomputed
        metadata.pop("final_error_U", None)
        metadata.pop("final_error_M", None)

        return cls(
            U=U,
            M=M,
            iterations=iterations,
            error_history_U=error_history_U,
            error_history_M=error_history_M,
            solver_name=solver_name,
            converged=converged,
            execution_time=execution_time,
            metadata=metadata,
        )

    def analyze_convergence(self) -> ConvergenceAnalysis:
        """
        Perform detailed convergence analysis on the solver result.

        Analyzes convergence history to detect convergence rate, stagnation,
        oscillation, and other patterns. Useful for understanding solver behavior
        and diagnosing convergence issues.

        Returns:
            ConvergenceAnalysis object with detailed diagnostics

        Example:
            >>> result = solver.solve()
            >>> analysis = result.analyze_convergence()
            >>> print(f"Convergence rate: {analysis.convergence_rate}")
            >>> if analysis.stagnation_detected:
            >>>     print("Warning: Stagnation detected")
        """
        return ConvergenceAnalysis.from_solver_result(self)

    def plot_convergence(
        self,
        save_path: str | Path | None = None,
        show: bool = True,
        figsize: tuple[float, float] = (10, 6),
        dpi: int = 150,
        log_scale: bool = True,
    ) -> Any:
        """
        Plot convergence history with automatic formatting.

        Creates a publication-quality convergence plot showing error evolution
        for both U and M, with optional rate estimation and pattern detection.

        Args:
            save_path: Path to save figure (if None, not saved)
            show: Whether to display the plot interactively
            figsize: Figure size in inches (width, height)
            dpi: Resolution for saved figure
            log_scale: Whether to use log scale for y-axis

        Returns:
            Matplotlib figure object

        Example:
            >>> result = solver.solve()
            >>> result.plot_convergence(save_path='convergence.png')
        """
        import matplotlib.pyplot as plt

        # Set backend to Agg if not showing (for non-interactive environments)
        if not show:
            import matplotlib

            matplotlib.use("Agg")

        fig, ax = plt.subplots(figsize=figsize)

        iterations = np.arange(1, self.iterations + 1)

        # Plot error histories
        ax.plot(iterations, self.error_history_U, "o-", label="U error", linewidth=2, markersize=4)
        ax.plot(iterations, self.error_history_M, "s-", label="M error", linewidth=2, markersize=4)

        # Set scale
        if log_scale:
            ax.set_yscale("log")

        # Labels and title
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Error (L∞ norm)", fontsize=12)
        ax.set_title(f"Convergence History - {self.solver_name}", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add convergence status annotation
        status_color = "green" if self.converged else "orange"
        status_text = "CONVERGED" if self.converged else "DID NOT CONVERGE"
        ax.text(
            0.98,
            0.98,
            status_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": status_color, "alpha": 0.3},
        )

        # Add execution time if available
        if self.execution_time is not None:
            time_text = f"Time: {self.execution_time:.3f}s"
            ax.text(
                0.02,
                0.02,
                time_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="bottom",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
            )

        plt.tight_layout()

        # Save if requested
        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        # Show if requested
        if show:
            plt.show()

        return fig

    def compare_to(self, other: SolverResult) -> ComparisonReport:
        """
        Compare this result to another solver result.

        Performs detailed comparison of solution accuracy, convergence behavior,
        and computational performance. Useful for benchmarking solvers and
        parameter studies.

        Args:
            other: Another SolverResult to compare against

        Returns:
            ComparisonReport with detailed comparison metrics

        Raises:
            ValueError: If solution shapes don't match

        Example:
            >>> result1 = fast_solver.solve(problem)
            >>> result2 = accurate_solver.solve(problem)
            >>> comparison = result1.compare_to(result2)
            >>> print(f"Solution difference: {comparison.solution_diff_l2}")
        """
        return ComparisonReport.from_solver_results(self, other)

    def export_summary(
        self,
        output_format: str = "markdown",
        filename: str | Path | None = None,
    ) -> str:
        """
        Generate publication-ready summary of solver results.

        Creates formatted summary tables suitable for reports, papers, or
        documentation. Supports markdown and LaTeX formats.

        Args:
            output_format: Output format ('markdown' or 'latex')
            filename: Optional path to save summary (if None, returns string)

        Returns:
            Formatted summary string

        Raises:
            ValueError: If output_format is not supported

        Example:
            >>> result = solver.solve()
            >>> result.export_summary(output_format='markdown', filename='results.md')
            >>> latex_summary = result.export_summary(output_format='latex')
        """
        if output_format not in ("markdown", "latex"):
            raise ValueError(f"Unsupported format: {output_format}. Use 'markdown' or 'latex'")

        if output_format == "markdown":
            summary = self._export_markdown()
        else:
            summary = self._export_latex()

        # Save if filename provided
        if filename is not None:
            from pathlib import Path

            Path(filename).write_text(summary)

        return summary

    def _export_markdown(self) -> str:
        """Generate markdown-formatted summary."""
        lines = [
            f"# Solver Results: {self.solver_name}",
            "",
            "## Summary",
            "",
            f"- **Status**: {'✅ Converged' if self.converged else '❌ Did not converge'}",
            f"- **Iterations**: {self.iterations}",
            f"- **Final Error (U)**: {self.final_error_U:.6e}",
            f"- **Final Error (M)**: {self.final_error_M:.6e}",
            f"- **Max Error**: {self.max_error:.6e}",
        ]

        if self.execution_time is not None:
            lines.append(f"- **Execution Time**: {self.execution_time:.3f} seconds")

        lines.extend(
            [
                "",
                "## Solution Details",
                "",
                f"- **Solution Shape**: {self.solution_shape}",
                f"- **Convergence History Length**: {len(self.error_history_U)}",
            ]
        )

        # Add metadata if present
        if self.metadata:
            lines.extend(
                [
                    "",
                    "## Additional Metadata",
                    "",
                ]
            )
            for key, value in self.metadata.items():
                if not isinstance(value, (np.ndarray, list, dict)):
                    lines.append(f"- **{key}**: {value}")

        return "\n".join(lines)

    def _export_latex(self) -> str:
        """Generate LaTeX-formatted summary."""
        status = "Converged" if self.converged else "Did not converge"
        exec_time = f"{self.execution_time:.3f}" if self.execution_time else "N/A"

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Solver Results: {self.solver_name}}}",
            "\\begin{tabular}{ll}",
            "\\toprule",
            "Property & Value \\\\",
            "\\midrule",
            f"Status & {status} \\\\",
            f"Iterations & {self.iterations} \\\\",
            f"Final Error ($u$) & ${self.final_error_U:.6e}$ \\\\",
            f"Final Error ($m$) & ${self.final_error_M:.6e}$ \\\\",
            f"Max Error & ${self.max_error:.6e}$ \\\\",
            f"Execution Time (s) & {exec_time} \\\\",
            f"Solution Shape & {self.solution_shape} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]

        return "\n".join(lines)


@dataclass
class ConvergenceAnalysis:
    """
    Detailed convergence analysis with pattern detection.

    Provides comprehensive analysis of solver convergence behavior including
    rate estimation, stagnation detection, and oscillation patterns.

    Attributes:
        converged: Whether convergence was achieved
        iterations: Number of iterations performed
        convergence_rate: Estimated linear convergence rate
        stagnation_detected: Whether stagnation was detected
        oscillation_detected: Whether oscillation was detected
        final_error_U: Final error for U
        final_error_M: Final error for M
        error_reduction_ratio_U: Total error reduction for U
        error_reduction_ratio_M: Total error reduction for M
    """

    converged: bool
    iterations: int
    convergence_rate: float | None
    stagnation_detected: bool
    oscillation_detected: bool
    final_error_U: float
    final_error_M: float
    error_reduction_ratio_U: float
    error_reduction_ratio_M: float

    @classmethod
    def from_solver_result(cls, result: SolverResult) -> ConvergenceAnalysis:
        """Create convergence analysis from solver result."""
        # Estimate convergence rate
        rate = cls._estimate_convergence_rate(
            result.error_history_U,
            result.error_history_M,
        )

        # Detect patterns
        stagnation = cls._detect_stagnation(
            result.error_history_U,
            result.error_history_M,
        )
        oscillation = cls._detect_oscillation(
            result.error_history_U,
            result.error_history_M,
        )

        # Calculate error reduction
        initial_error_U = result.error_history_U[0] if len(result.error_history_U) > 0 else 1.0
        initial_error_M = result.error_history_M[0] if len(result.error_history_M) > 0 else 1.0

        error_reduction_U = initial_error_U / result.final_error_U if result.final_error_U > 0 else float("inf")
        error_reduction_M = initial_error_M / result.final_error_M if result.final_error_M > 0 else float("inf")

        return cls(
            converged=result.converged,
            iterations=result.iterations,
            convergence_rate=rate,
            stagnation_detected=stagnation,
            oscillation_detected=oscillation,
            final_error_U=result.final_error_U,
            final_error_M=result.final_error_M,
            error_reduction_ratio_U=error_reduction_U,
            error_reduction_ratio_M=error_reduction_M,
        )

    @staticmethod
    def _estimate_convergence_rate(
        error_history_U: NDArray[np.floating],
        error_history_M: NDArray[np.floating],
    ) -> float | None:
        """Estimate linear convergence rate from error history."""
        if len(error_history_U) < 3:
            return None

        try:
            # Use maximum of U and M errors
            errors = np.maximum(error_history_U, error_history_M)

            # Filter out zeros and take log
            errors_nonzero = errors[errors > 0]
            if len(errors_nonzero) < 3:
                return None

            log_errors = np.log(errors_nonzero)
            iterations = np.arange(len(log_errors))

            # Fit linear trend: log(error) = a * iteration + b
            coeffs = np.polyfit(iterations, log_errors, 1)
            slope = coeffs[0]

            # Convergence rate is -slope (positive rate means decreasing error)
            return float(-slope) if slope < 0 else None

        except (ValueError, TypeError, FloatingPointError):
            return None

    @staticmethod
    def _detect_stagnation(
        error_history_U: NDArray[np.floating],
        error_history_M: NDArray[np.floating],
        window: int = 5,
        threshold: float = 0.01,
    ) -> bool:
        """Detect if convergence has stagnated."""
        if len(error_history_U) < window:
            return False

        # Check last 'window' iterations
        recent_U = error_history_U[-window:]
        recent_M = error_history_M[-window:]

        # Calculate relative change
        range_U = np.max(recent_U) - np.min(recent_U)
        range_M = np.max(recent_M) - np.min(recent_M)

        mean_U = np.mean(recent_U)
        mean_M = np.mean(recent_M)

        # Stagnation if relative change is very small
        rel_change_U = range_U / mean_U if mean_U > 0 else 0
        rel_change_M = range_M / mean_M if mean_M > 0 else 0

        return rel_change_U < threshold and rel_change_M < threshold

    @staticmethod
    def _detect_oscillation(
        error_history_U: NDArray[np.floating],
        error_history_M: NDArray[np.floating],
        window: int = 6,
    ) -> bool:
        """Detect if errors are oscillating."""
        if len(error_history_U) < window:
            return False

        # Check last 'window' iterations for sign changes in differences
        recent_U = error_history_U[-window:]
        recent_M = error_history_M[-window:]

        diff_U = np.diff(recent_U)
        diff_M = np.diff(recent_M)

        # Count sign changes
        sign_changes_U = np.sum(np.diff(np.sign(diff_U)) != 0)
        sign_changes_M = np.sum(np.diff(np.sign(diff_M)) != 0)

        # Oscillation if multiple sign changes
        return sign_changes_U >= 3 or sign_changes_M >= 3

    def __repr__(self) -> str:
        """String representation of analysis."""
        status = "CONVERGED" if self.converged else "NOT CONVERGED"
        rate_str = f"{self.convergence_rate:.4f}" if self.convergence_rate else "N/A"

        warnings = []
        if self.stagnation_detected:
            warnings.append("STAGNATION")
        if self.oscillation_detected:
            warnings.append("OSCILLATION")

        warning_str = f" [{', '.join(warnings)}]" if warnings else ""

        return (
            f"ConvergenceAnalysis({status} in {self.iterations} iters, "
            f"rate={rate_str}, errors U={self.final_error_U:.2e} "
            f"M={self.final_error_M:.2e}{warning_str})"
        )


@dataclass
class ComparisonReport:
    """
    Detailed comparison of two solver results.

    Provides metrics for comparing solution accuracy, convergence behavior,
    and computational performance between two solver results.

    Attributes:
        solution_diff_l2: L2 norm of solution difference
        solution_diff_linf: L∞ norm of solution difference
        iterations_diff: Difference in iteration counts
        time_diff: Difference in execution time (seconds)
        converged_both: Whether both results converged
        faster_solver: Name of faster solver (by time)
        more_accurate_solver: Name of more accurate solver (by error)
    """

    solution_diff_l2: float
    solution_diff_linf: float
    iterations_diff: int
    time_diff: float | None
    converged_both: bool
    faster_solver: str
    more_accurate_solver: str
    result1_name: str
    result2_name: str

    @classmethod
    def from_solver_results(
        cls,
        result1: SolverResult,
        result2: SolverResult,
    ) -> ComparisonReport:
        """Create comparison report from two solver results."""
        # Validate shapes match
        if result1.solution_shape != result2.solution_shape:
            raise ValueError(f"Solution shapes don't match: {result1.solution_shape} vs {result2.solution_shape}")

        # Calculate solution differences
        diff_U = result1.U - result2.U
        diff_M = result1.M - result2.M

        # L2 norms
        l2_U = float(np.linalg.norm(diff_U))
        l2_M = float(np.linalg.norm(diff_M))
        l2_total = float(np.sqrt(l2_U**2 + l2_M**2))

        # L∞ norms
        linf_U = float(np.max(np.abs(diff_U)))
        linf_M = float(np.max(np.abs(diff_M)))
        linf_total = float(max(linf_U, linf_M))

        # Iteration and time comparison
        iter_diff = result1.iterations - result2.iterations

        time_diff = None
        faster = "tie"
        if result1.execution_time is not None and result2.execution_time is not None:
            time_diff = result1.execution_time - result2.execution_time
            if abs(time_diff) > 0.001:  # 1ms threshold
                faster = result1.solver_name if time_diff < 0 else result2.solver_name

        # Accuracy comparison (lower error is better)
        more_accurate = "tie"
        if result1.max_error < result2.max_error:
            more_accurate = result1.solver_name
        elif result2.max_error < result1.max_error:
            more_accurate = result2.solver_name

        return cls(
            solution_diff_l2=l2_total,
            solution_diff_linf=linf_total,
            iterations_diff=iter_diff,
            time_diff=time_diff,
            converged_both=result1.converged and result2.converged,
            faster_solver=faster,
            more_accurate_solver=more_accurate,
            result1_name=result1.solver_name,
            result2_name=result2.solver_name,
        )

    def __repr__(self) -> str:
        """String representation of comparison."""
        time_str = f"{abs(self.time_diff):.3f}s" if self.time_diff is not None else "N/A"
        time_winner = f" (faster: {self.faster_solver})" if self.time_diff is not None else ""

        return (
            f"ComparisonReport({self.result1_name} vs {self.result2_name}: "
            f"L2={self.solution_diff_l2:.2e}, L∞={self.solution_diff_linf:.2e}, "
            f"Δiters={self.iterations_diff}, Δtime={time_str}{time_winner})"
        )


@dataclass
class ConvergenceResult:
    """
    Detailed convergence information for analysis and debugging.

    This class provides extended convergence analysis beyond basic error tracking.
    """

    error_history_U: NDArray[np.floating]
    error_history_M: NDArray[np.floating]
    iterations_performed: int
    convergence_achieved: bool
    final_tolerance: float
    convergence_criteria: str = "L2_relative"
    stagnation_detected: bool = False
    oscillation_detected: bool = False
    divergence_detected: bool = False
    convergence_rate_estimate: float | None = None

    @property
    def convergence_trend(self) -> str:
        """Analyze overall convergence trend."""
        if len(self.error_history_U) < 3:
            return "insufficient_data"

        # Combined error trend analysis
        recent_errors = np.maximum(self.error_history_U[-3:], self.error_history_M[-3:])

        if recent_errors[-1] < recent_errors[-2] < recent_errors[-3]:
            return "converging"
        elif recent_errors[-1] > recent_errors[-2] * 1.2:
            return "diverging"
        elif np.max(recent_errors) / np.min(recent_errors) < 1.1:
            return "stagnating"
        else:
            return "oscillating"

    def estimate_convergence_rate(self) -> float | None:
        """Estimate linear convergence rate if possible."""
        if len(self.error_history_U) < 3:
            return None

        try:
            errors = np.maximum(self.error_history_U, self.error_history_M)
            # Take log of errors to estimate rate
            log_errors = np.log(errors[errors > 0])
            if len(log_errors) < 3:
                return None

            # Fit linear trend to log(error) vs iteration
            iterations = np.arange(len(log_errors))
            slope = np.polyfit(iterations, log_errors, 1)[0]

            # Convergence rate is the exponential decay rate
            return float(-slope) if slope < 0 else None

        except (ValueError, TypeError):
            return None


def create_solver_result(
    U: NDArray[np.floating],
    M: NDArray[np.floating],
    iterations: int,
    error_history_U: NDArray[np.floating],
    error_history_M: NDArray[np.floating],
    solver_name: str = "Unknown Solver",
    converged: bool = False,
    tolerance: float | None = None,
    execution_time: float | None = None,
    **metadata,
) -> SolverResult:
    """
    Factory function to create SolverResult with automatic convergence detection.

    Args:
        U: Control/value function solution
        M: Density/distribution function solution
        iterations: Number of iterations performed
        error_history_U: History of U convergence errors
        error_history_M: History of M convergence errors
        solver_name: Name of the solver
        converged: Whether convergence was achieved (auto-detected if None)
        tolerance: Convergence tolerance used (for auto-detection)
        execution_time: Solve time in seconds
        **metadata: Additional solver-specific data

    Returns:
        SolverResult object with all fields populated
    """

    # Auto-detect convergence if not explicitly provided
    if not converged and tolerance is not None and len(error_history_U) > 0:
        final_error_U = error_history_U[-1] if len(error_history_U) > 0 else float("inf")
        final_error_M = error_history_M[-1] if len(error_history_M) > 0 else float("inf")
        converged = max(final_error_U, final_error_M) < tolerance

    # Create convergence analysis
    convergence_info = ConvergenceResult(
        error_history_U=error_history_U,
        error_history_M=error_history_M,
        iterations_performed=iterations,
        convergence_achieved=converged,  # ConvergenceResult keeps its own naming
        final_tolerance=tolerance or 0.0,
        convergence_rate_estimate=None,  # Will be computed on access
    )
    convergence_info.convergence_rate_estimate = convergence_info.estimate_convergence_rate()

    # Add convergence analysis to metadata
    metadata["convergence_analysis"] = convergence_info

    return SolverResult(
        U=U,
        M=M,
        iterations=iterations,
        error_history_U=error_history_U,
        error_history_M=error_history_M,
        solver_name=solver_name,
        converged=converged,
        execution_time=execution_time,
        metadata=metadata,
    )


# Backward compatibility type alias
MFGSolverResult = SolverResult
