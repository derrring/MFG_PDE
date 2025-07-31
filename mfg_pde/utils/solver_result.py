"""
Standardized result objects for MFG_PDE solvers.

This module provides structured result objects that replace tuple returns,
improving code readability, IDE support, and API maintainability.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


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

    U: np.ndarray
    M: np.ndarray
    iterations: int
    error_history_U: np.ndarray
    error_history_M: np.ndarray
    solver_name: str = "Unknown Solver"
    convergence_achieved: bool = False
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate result data after initialization."""
        if self.U.shape != self.M.shape:
            raise ValueError(
                f"U and M shapes must match: U{self.U.shape} vs M{self.M.shape}"
            )

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

    # Convenience properties
    @property
    def final_error_U(self) -> float:
        """Get the final convergence error for U."""
        return (
            self.error_history_U[-1] if len(self.error_history_U) > 0 else float("inf")
        )

    @property
    def final_error_M(self) -> float:
        """Get the final convergence error for M."""
        return (
            self.error_history_M[-1] if len(self.error_history_M) > 0 else float("inf")
        )

    @property
    def max_error(self) -> float:
        """Get the maximum of the final errors."""
        return max(self.final_error_U, self.final_error_M)

    @property
    def solution_shape(self) -> Tuple[int, int]:
        """Get the shape of the solution arrays."""
        return self.U.shape

    def to_dict(self) -> Dict[str, Any]:
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


@dataclass
class ConvergenceResult:
    """
    Detailed convergence information for analysis and debugging.

    This class provides extended convergence analysis beyond basic error tracking.
    """

    error_history_U: np.ndarray
    error_history_M: np.ndarray
    iterations_performed: int
    convergence_achieved: bool
    final_tolerance: float
    convergence_criteria: str = "L2_relative"
    stagnation_detected: bool = False
    oscillation_detected: bool = False
    divergence_detected: bool = False
    convergence_rate_estimate: Optional[float] = None

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

    def estimate_convergence_rate(self) -> Optional[float]:
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
            return -slope if slope < 0 else None

        except (ValueError, TypeError):
            return None


def create_solver_result(
    U: np.ndarray,
    M: np.ndarray,
    iterations: int,
    error_history_U: np.ndarray,
    error_history_M: np.ndarray,
    solver_name: str = "Unknown Solver",
    convergence_achieved: bool = False,
    tolerance: Optional[float] = None,
    execution_time: Optional[float] = None,
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
        final_error_U = (
            error_history_U[-1] if len(error_history_U) > 0 else float("inf")
        )
        final_error_M = (
            error_history_M[-1] if len(error_history_M) > 0 else float("inf")
        )
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
    convergence_info.convergence_rate_estimate = (
        convergence_info.estimate_convergence_rate()
    )

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
