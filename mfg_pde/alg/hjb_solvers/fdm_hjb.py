import numpy as np
from . import base_hjb
from .base_hjb import BaseHJBSolver


class FdmHJBSolver(BaseHJBSolver):
    def __init__(self, problem, NiterNewton=30, l2errBoundNewton=1e-6):
        super().__init__(problem)
        self.hjb_method_name = "FDM"
        self.NiterNewton = NiterNewton
        self.l2errBoundNewton = l2errBoundNewton

    def solve_hjb_system(self, M_density_evolution, U_final_condition):
        """
        Solves the full HJB system backward in time using FDM (via hjb_utils).
        Args:
            M_density_evolution (np.array): (Nt+1, Nx) array of density m(t,x).
            U_final_condition (np.array): (Nx,) array for U(T,x).
        Returns:
            np.array: U_solution (Nt+1, Nx)
        """
        print(f"****** Solving HJB ({self.hjb_method_name} via hjb_utils)")
        U_new_solution = base_hjb.solve_hjb_system_backward(
            M_density_evolution_from_FP=M_density_evolution,
            U_final_condition_at_T=U_final_condition,
            problem=self.problem,
            NiterNewton=self.NiterNewton,
            l2errBoundNewton=self.l2errBoundNewton,
        )
        return U_new_solution