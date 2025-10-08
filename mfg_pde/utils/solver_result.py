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


@dataclass
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
        convergence_achieved: Whether convergence was achieved
        execution_time: Total solve time in seconds
        metadata: Additional solver-specific information
    """

    U: NDArray[np.floating]
    M: NDArray[np.floating]
    iterations: int
    error_history_U: NDArray[np.floating]
    error_history_M: NDArray[np.floating]
    solver_name: str = "Unknown Solver"
    convergence_achieved: bool = False
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
        convergence_achieved: bool = False,
        execution_time: float | None = None,
        metadata: dict[str, Any] | None = None,
        # Deprecated parameters
        converged: bool | None = None,
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
            convergence_achieved: Whether convergence was achieved
            execution_time: Total solve time in seconds
            metadata: Additional solver-specific information
            converged: DEPRECATED - Use convergence_achieved instead
            convergence_reason: DEPRECATED - Add to metadata instead
            diagnostics: DEPRECATED - Add to metadata instead
        """
        # Handle deprecated 'converged' parameter
        if converged is not None:
            warnings.warn(
                "Parameter 'converged' is deprecated, use 'convergence_achieved' instead",
                DeprecationWarning,
                stacklevel=2,
            )
            convergence_achieved = converged

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
        self.convergence_achieved = convergence_achieved
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
    def converged(self) -> bool:
        """DEPRECATED: Use convergence_achieved instead."""
        warnings.warn(
            "Property 'converged' is deprecated, use 'convergence_achieved' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.convergence_achieved

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
            "convergence_achieved": self.convergence_achieved,
            "execution_time": self.execution_time,
            "final_error_U": self.final_error_U,
            "final_error_M": self.final_error_M,
            "max_error": self.max_error,
            "solution_shape": self.solution_shape,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        """String representation of the result."""
        convergence_status = "SUCCESS:" if self.convergence_achieved else "WARNING:"
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
            "convergence_achieved": self.convergence_achieved,
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
            >>> print(f"Converged: {result.convergence_achieved}")
        """
        from mfg_pde.utils.io.hdf5_utils import load_solution

        U, M, metadata = load_solution(filename)

        # Extract standard fields
        iterations = metadata.pop("iterations", 0)
        error_history_U = metadata.pop("error_history_U", np.array([]))
        error_history_M = metadata.pop("error_history_M", np.array([]))
        solver_name = metadata.pop("solver_name", "Unknown Solver")
        convergence_achieved = metadata.pop("convergence_achieved", False)
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
            convergence_achieved=convergence_achieved,
            execution_time=execution_time,
            metadata=metadata,
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
    convergence_achieved: bool = False,
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
        convergence_achieved: Whether convergence was achieved (auto-detected if None)
        tolerance: Convergence tolerance used (for auto-detection)
        execution_time: Solve time in seconds
        **metadata: Additional solver-specific data

    Returns:
        SolverResult object with all fields populated
    """

    # Auto-detect convergence if not explicitly provided
    if not convergence_achieved and tolerance is not None and len(error_history_U) > 0:
        final_error_U = error_history_U[-1] if len(error_history_U) > 0 else float("inf")
        final_error_M = error_history_M[-1] if len(error_history_M) > 0 else float("inf")
        convergence_achieved = max(final_error_U, final_error_M) < tolerance

    # Create convergence analysis
    convergence_info = ConvergenceResult(
        error_history_U=error_history_U,
        error_history_M=error_history_M,
        iterations_performed=iterations,
        convergence_achieved=convergence_achieved,
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
        convergence_achieved=convergence_achieved,
        execution_time=execution_time,
        metadata=metadata,
    )


# Backward compatibility type alias
MFGSolverResult = SolverResult
