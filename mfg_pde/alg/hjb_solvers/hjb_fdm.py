import numpy as np

# Assuming base_hjb is in the same directory or correctly pathed
from . import base_hjb
from .base_hjb import BaseHJBSolver
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem


class HJBFDMSolver(BaseHJBSolver):
    def __init__(
        self,
        problem: "MFGProblem",
        NiterNewton: int = 30,
        l2errBoundNewton: float = 1e-6,
    ):
        super().__init__(problem)
        self.hjb_method_name = "FDM"
        self.NiterNewton = NiterNewton
        self.l2errBoundNewton = l2errBoundNewton

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
            NiterNewton=self.NiterNewton,
            l2errBoundNewton=self.l2errBoundNewton,
        )
        return U_new_solution
