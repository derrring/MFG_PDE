from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.types.solver_types import SolverReturnTuple
else:
    pass


class BaseCouplingIterator(ABC):
    """
    Abstract base class for iterative coupling solvers (Picard, block, fictitious play).

    Provides the interface for MFG solvers that iterate between HJB and FP
    sub-solvers to solve the coupled system. Distinguished from
    ``alg.base_solver.BaseMFGSolver`` which is the cross-paradigm base.
    """

    def __init__(self, problem: MFGProblem) -> None:
        """
        Initialize the MFG solver with a problem definition.

        Args:
            problem: The MFG problem to solve
        """
        self.problem = problem
        self.warm_start_data: dict[str, Any] | None = None
        self._solution_computed: bool = False

    @abstractmethod
    def solve(self, max_iterations: int, tolerance: float = 1e-5, **kwargs: Any) -> SolverReturnTuple:
        """
        Solve the coupled MFG system.

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

    @abstractmethod
    def get_results(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the computed solution arrays.

        Returns:
            Tuple of (U, M) solution arrays
        """

    def set_warm_start_data(
        self,
        previous_solution: tuple[np.ndarray, np.ndarray],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Set warm start data from a previous solution.

        Args:
            previous_solution: Tuple of (U, M) arrays from previous solve
            metadata: Optional metadata about the previous solution
        """
        U_prev, M_prev = previous_solution

        # Validate dimensions with enhanced error messages
        from mfgarchon.utils.exceptions import validate_array_dimensions

        # Get expected shape from geometry
        spatial_shape = tuple(self.problem.geometry.get_grid_shape())
        expected_shape = (self.problem.Nt + 1, *spatial_shape)

        try:
            validate_array_dimensions(
                U_prev,
                expected_shape=expected_shape,
                array_name="warm_start_U",
            )
            validate_array_dimensions(
                M_prev,
                expected_shape=expected_shape,
                array_name="warm_start_M",
            )
        except Exception as e:
            raise ValueError(f"Invalid warm start data dimensions: {e}") from e

        self.warm_start_data = {
            "U_prev": U_prev.copy(),
            "M_prev": M_prev.copy(),
            "metadata": metadata or {},
        }

    def get_warm_start_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Get warm start data if available.

        Returns:
            Tuple of (U, M) arrays if warm start data exists, None otherwise
        """
        if self.warm_start_data is None:
            return None
        return self.warm_start_data["U_prev"], self.warm_start_data["M_prev"]

    def clear_warm_start_data(self) -> None:
        """Clear any stored warm start data."""
        self.warm_start_data = None

    @property
    def has_warm_start_data(self) -> bool:
        """Check if warm start data is available."""
        return self.warm_start_data is not None

    @property
    def is_solved(self) -> bool:
        """Check if the solver has computed a solution."""
        return self._solution_computed


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing BaseCouplingIterator...")

    # Test base class availability
    assert BaseCouplingIterator is not None
    print("  BaseCouplingIterator class available")

    # Note: BaseCouplingIterator is abstract and requires implementation
    # See FixedPointMFGSolver for concrete implementation

    print("Smoke tests passed!")
