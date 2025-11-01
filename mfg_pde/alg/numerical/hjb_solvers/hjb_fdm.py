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
        U_from_prev_picard: np.ndarray,
    ) -> np.ndarray:
        """
        Solves the HJB system backward in time using FDM.

        Note: This solver only supports 1D problems. For multi-dimensional problems,
        use HJBGFDMSolver or HJBWENOSolver instead.

        Args:
            M_density_evolution: Density evolution from previous Picard iteration
                Shape: (Nt, Nx) for 1D problems
            U_final_condition: Terminal condition for value function
                Shape: (Nx,) for 1D problems
            U_from_prev_picard: Value function from previous Picard iteration
                Shape: (Nt, Nx) for 1D problems

        Returns:
            U_solution: Value function evolution
                Shape: (Nt, Nx) for 1D problems

        Raises:
            NotImplementedError: If problem dimension is not 1D

        Notes:
            - Uses Newton iteration for nonlinear HJB equation
            - For nD problems, use HJBGFDMSolver (meshfree) or HJBWENOSolver (high-order)
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
            # HJB FDM only supports 1D problems
            # For multi-dimensional problems, use production nD methods:
            raise NotImplementedError(
                f"HJB FDM solver only supports 1D problems (got {self.dimension}D). "
                f"For multi-dimensional HJB problems, use:\n"
                f"  - HJBGFDMSolver (meshfree, flexible geometry, arbitrary dimensions)\n"
                f"  - HJBWENOSolver (high-order, structured grids, 2D/3D)\n"
                f"Example: from mfg_pde.alg.numerical.hjb_solvers import HJBGFDMSolver"
            )

        return U_new_solution
