from __future__ import annotations

from typing import TYPE_CHECKING

# Assuming base_hjb is in the same directory or correctly pathed
from . import base_hjb
from .base_hjb import BaseHJBSolver

if TYPE_CHECKING:
    import numpy as np

    from mfg_pde.core.mfg_problem import MFGProblem


class HJBFDMSolver(BaseHJBSolver):
    def __init__(
        self,
        problem: MFGProblem,
        max_newton_iterations: int | None = None,
        newton_tolerance: float | None = None,
        # Deprecated parameters for backward compatibility
        NiterNewton: int | None = None,
        l2errBoundNewton: float | None = None,
        backend: str | None = None,
    ):
        import warnings

        super().__init__(problem)
        self.hjb_method_name = "FDM"

        # Initialize backend (defaults to NumPy)
        from mfg_pde.backends import create_backend

        if backend is not None:
            self.backend = create_backend(backend)
        else:
            self.backend = create_backend("numpy")  # NumPy fallback

        # Handle backward compatibility
        if NiterNewton is not None:
            warnings.warn(
                "Parameter 'NiterNewton' is deprecated. Use 'max_newton_iterations' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if max_newton_iterations is None:
                max_newton_iterations = NiterNewton

        if l2errBoundNewton is not None:
            warnings.warn(
                "Parameter 'l2errBoundNewton' is deprecated. Use 'newton_tolerance' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if newton_tolerance is None:
                newton_tolerance = l2errBoundNewton

        # Set defaults if still None
        if max_newton_iterations is None:
            max_newton_iterations = 30
        if newton_tolerance is None:
            newton_tolerance = 1e-6

        # Store with new names
        self.max_newton_iterations = max_newton_iterations
        self.newton_tolerance = newton_tolerance

        # Validate parameter ranges
        if self.max_newton_iterations < 1:
            raise ValueError(f"max_newton_iterations must be >= 1, got {self.max_newton_iterations}")
        if self.newton_tolerance <= 0:
            raise ValueError(f"newton_tolerance must be > 0, got {self.newton_tolerance}")

        # Store parameters for solver access
        self._newton_config = {
            "max_iterations": self.max_newton_iterations,
            "tolerance": self.newton_tolerance,
        }

        # Detect problem dimension
        self.dimension = self._detect_dimension(problem)

    def _detect_dimension(self, problem) -> int:
        """
        Detect the dimension of the problem.

        Args:
            problem: MFGProblem or GridBasedMFGProblem instance

        Returns:
            dimension: 1 for 1D problems, 2 for 2D, 3 for 3D, etc.

        Raises:
            ValueError: If dimension cannot be determined
        """
        # Check if it's a GridBasedMFGProblem with explicit dimension
        if hasattr(problem, "geometry") and hasattr(problem.geometry, "grid"):
            if hasattr(problem.geometry.grid, "ndim"):
                return problem.geometry.grid.ndim

        # Check for 1D MFGProblem (has Nx but not Ny)
        if hasattr(problem, "Nx") and not hasattr(problem, "Ny"):
            return 1

        # Check for explicit dimension attribute
        if hasattr(problem, "dimension"):
            return problem.dimension

        # If we can't determine dimension, raise error
        raise ValueError(
            "Cannot determine problem dimension. Problem must be either 1D MFGProblem or GridBasedMFGProblem."
        )

    def solve_hjb_system(
        self,
        M_density_evolution: np.ndarray,
        U_final_condition: np.ndarray,
        U_from_prev_picard: np.ndarray,  # Added this argument
    ) -> np.ndarray:
        """
        Solves the full HJB system backward in time using FDM.

        Routes to appropriate solver based on problem dimension:
        - 1D: Uses existing base_hjb.solve_hjb_system_backward()
        - 2D, 3D, ...: Uses dimensional splitting (hjb_fdm_multid module)

        Args:
            M_density_evolution: Density evolution from previous Picard iteration
                - 1D: (Nt, Nx) array
                - nD: (Nt, N1, N2, ..., Nd) array
            U_final_condition: Terminal condition for value function
                - 1D: (Nx,) array
                - nD: (N1, N2, ..., Nd) array
            U_from_prev_picard: Value function from previous Picard iteration
                - 1D: (Nt, Nx) array
                - nD: (Nt, N1, N2, ..., Nd) array

        Returns:
            U_solution: Value function evolution
                - 1D: (Nt, Nx) array
                - nD: (Nt, N1, N2, ..., Nd) array

        Notes:
            - 1D solver: Direct finite difference method
            - nD solver: Dimensional splitting (Strang splitting)
            - Both methods use Newton iteration for nonlinear HJB
        """
        if self.dimension == 1:
            # Use existing 1D FDM solver
            U_new_solution = base_hjb.solve_hjb_system_backward(
                M_density_from_prev_picard=M_density_evolution,
                U_final_condition_at_T=U_final_condition,
                U_from_prev_picard=U_from_prev_picard,
                problem=self.problem,
                max_newton_iterations=self.max_newton_iterations,
                newton_tolerance=self.newton_tolerance,
                backend=self.backend,
            )
        else:
            # Use dimension-agnostic nD solver (works for 2D, 3D, 4D, ...)
            from . import hjb_fdm_multid

            U_new_solution = hjb_fdm_multid.solve_hjb_nd_dimensional_splitting(
                M_density=M_density_evolution,
                U_final=U_final_condition,
                U_prev=U_from_prev_picard,
                problem=self.problem,
                max_newton_iterations=self.max_newton_iterations,
                newton_tolerance=self.newton_tolerance,
                backend=self.backend,
            )

        return U_new_solution
