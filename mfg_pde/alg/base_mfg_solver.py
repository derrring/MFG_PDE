from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.mfg_problem import MFGProblem


class MFGSolver(ABC):
    def __init__(self, problem: "MFGProblem") -> None:
        self.problem = problem
        self.warm_start_data: Optional[Dict[str, Any]] = None
        self._solution_computed: bool = False

    @abstractmethod
    def solve(
        self, max_iterations: int, tolerance: float = 1e-5, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Solves the MFG system and returns U, M, and convergence info.

        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            **kwargs: Additional solver-specific parameters

        Returns:
            Tuple of (U, M, convergence_info) where:
            - U: Hamilton-Jacobi-Bellman solution array
            - M: Fokker-Planck density array
            - convergence_info: Dictionary with convergence details
        """
        pass

    @abstractmethod
    def get_results(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the computed U and M.

        Returns:
            Tuple of (U, M) solution arrays
        """
        pass

    def set_warm_start_data(
        self,
        previous_solution: Tuple[np.ndarray, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set warm start data from a previous solution.

        Args:
            previous_solution: Tuple of (U, M) arrays from previous solve
            metadata: Optional metadata about the previous solution (parameters, convergence info, etc.)
        """
        U_prev, M_prev = previous_solution

        # Validate dimensions with enhanced error messages
        from ..utils.exceptions import validate_array_dimensions

        expected_shape = (self.problem.Nt + 1, self.problem.Nx + 1)

        validate_array_dimensions(U_prev, expected_shape, "warm_start_U", "MFGSolver")
        validate_array_dimensions(M_prev, expected_shape, "warm_start_M", "MFGSolver")

        self.warm_start_data = {
            "U": U_prev.copy(),
            "M": M_prev.copy(),
            "metadata": metadata or {},
            "timestamp": np.datetime64("now"),
        }

    def has_warm_start_data(self) -> bool:
        """Check if warm start data is available."""
        return self.warm_start_data is not None

    def clear_warm_start_data(self):
        """Clear warm start data."""
        self.warm_start_data = None

    def _extrapolate_solution(
        self,
        previous_solution: np.ndarray,
        parameter_change_info: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Extrapolate previous solution for warm start initialization.

        Args:
            previous_solution: Previous U or M solution array
            parameter_change_info: Information about parameter changes (for future enhancements)

        Returns:
            Extrapolated solution for initialization
        """
        # For now, use simple copy (can be enhanced with linear extrapolation, etc.)
        return previous_solution.copy()

    def _get_warm_start_initialization(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get warm start initialization if available.

        Returns:
            Tuple of (U_init, M_init) if warm start data available, None otherwise
        """
        if not self.has_warm_start_data():
            return None

        U_init = self._extrapolate_solution(self.warm_start_data["U"])
        M_init = self._extrapolate_solution(self.warm_start_data["M"])

        return U_init, M_init
