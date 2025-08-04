from typing import TYPE_CHECKING

import numpy as np

# Assuming base_hjb is in the same directory or correctly pathed
from . import base_hjb
from .base_hjb import BaseHJBSolver

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem


class HJBFDMSolver(BaseHJBSolver):
    def __init__(
        self,
        problem: "MFGProblem",
        max_newton_iterations: int = None,
        newton_tolerance: float = None,
        # Deprecated parameters for backward compatibility
        NiterNewton: int = None,
        l2errBoundNewton: float = None,
    ):
        import warnings

        super().__init__(problem)
        self.hjb_method_name = "FDM"

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

    def solve_hjb_system(
        self,
        M_density_evolution: np.ndarray,
        U_final_condition: np.ndarray,
        U_from_prev_picard: np.ndarray,  # Added this argument
    ) -> np.ndarray:
        """
        Solves the full HJB system backward in time using FDM (via base_hjb utilities).
        Args:
            M_density_evolution (np.ndarray): (Nt, Nx) array of density m(t,x) from prev. Picard.
            U_final_condition (np.ndarray): (Nx,) array for U(T,x).
            U_from_prev_picard (np.ndarray): (Nt, Nx) array of U(t,x) from prev. Picard iter.
                                             Used for specific Jacobian forms (like original notebook).
        Returns:
            np.ndarray: U_solution (Nt, Nx) for the current Picard iteration.
        """
        # print(f"****** Solving HJB ({self.hjb_method_name} via base_hjb utilities)")
        U_new_solution = base_hjb.solve_hjb_system_backward(
            M_density_from_prev_picard=M_density_evolution,  # Renamed for clarity in base_hjb
            U_final_condition_at_T=U_final_condition,
            U_from_prev_picard=U_from_prev_picard,  # Pass this through
            problem=self.problem,
            max_newton_iterations=self.max_newton_iterations,
            newton_tolerance=self.newton_tolerance,
        )
        return U_new_solution
