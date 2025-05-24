import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
# from .base_fp_solver import BaseFPSolver # Assuming BaseFPSolver is in the same directory
# If BaseFPSolver is in mfg_pde.alg.fp_solvers.base_fp_solver:
from .base_fp import BaseFPSolver 
# Or adjust path: from ..base_fp_solver import BaseFPSolver
# Or: from mfg_pde.alg.fp_solvers.base_fp_solver import BaseFPSolver

class FdmFPSolver(BaseFPSolver):
    def __init__(self, problem):
        super().__init__(problem)
        self.fp_method_name = "FDM"

    def solve_fp_system(self, m_initial_condition, U_solution_for_drift):
        """
        Solves the full FP system forward in time using FDM.
        Args:
            m_initial_condition (np.array): Initial density M(0,x).
            U_solution_for_drift (np.array): (Nt+1, Nx) array of value function U(t,x) used for drift.
        Returns:
            np.array: M_solution (Nt+1, Nx)
        """
        print(f"****** Solving FP ({self.fp_method_name})")
        Nx = self.problem.Nx
        Nt = self.problem.Nt
        Dx = self.problem.Dx
        Dt = self.problem.Dt
        sigma = self.problem.sigma
        coefCT = self.problem.coefCT

        m = np.zeros((Nt + 1, Nx))
        m[0] = m_initial_condition

        for k_idx_fp in range(Nt):
            u_at_tk = U_solution_for_drift[k_idx_fp]

            A_L = np.zeros(Nx)
            A_D = np.zeros(Nx)
            A_U = np.zeros(Nx)
            A_upcorner = np.zeros(Nx)
            A_lowcorner = np.zeros(Nx)

            A_D += 1.0 / Dt
            A_D += sigma**2 / Dx**2
            A_D[1 : Nx - 1] += (
                coefCT
                * (
                    self.problem._npart(u_at_tk[2:Nx] - u_at_tk[1 : Nx - 1])
                    + self.problem._ppart(u_at_tk[1 : Nx - 1] - u_at_tk[0 : Nx - 2])
                )
                / Dx**2
            )
            A_D[0] += (
                coefCT
                * (
                    self.problem._npart(u_at_tk[1] - u_at_tk[0])
                    + self.problem._ppart(u_at_tk[0] - u_at_tk[Nx - 1])
                )
                / Dx**2
            )
            A_D[Nx - 1] += (
                coefCT
                * (
                    self.problem._npart(u_at_tk[0] - u_at_tk[Nx - 1])
                    + self.problem._ppart(u_at_tk[Nx - 1] - u_at_tk[Nx - 2])
                )
                / Dx**2
            )
            A_L_val_diff = -(sigma**2) / (2 * Dx**2)
            A_U_val_diff = -(sigma**2) / (2 * Dx**2)
            A_L[1:Nx] += A_L_val_diff
            A_L[1:Nx] += (
                -coefCT
                * self.problem._npart(u_at_tk[1:Nx] - u_at_tk[0 : Nx - 1])
                / Dx**2
            )
            A_U[0 : Nx - 1] += A_U_val_diff
            A_U[0 : Nx - 1] += (
                -coefCT
                * self.problem._ppart(u_at_tk[1:Nx] - u_at_tk[0 : Nx - 1])
                / Dx**2
            )
            A_upcorner[0] = (
                A_U_val_diff
                - coefCT * self.problem._ppart(u_at_tk[1] - u_at_tk[0]) / Dx**2
            )
            A_lowcorner[Nx - 1] = (
                A_L_val_diff
                - coefCT * self.problem._npart(u_at_tk[0] - u_at_tk[Nx - 1]) / Dx**2
            )
            A_L_sp = np.roll(A_L, -1)
            A_U_sp = np.roll(A_U, 1)
            A_matrix = sparse.spdiags(
                [A_lowcorner, A_L_sp, A_D, A_U_sp, A_upcorner],
                [-(Nx - 1), -1, 0, 1, Nx - 1],
                Nx, Nx, format="csr",
            )
            b_rhs = m[k_idx_fp] / Dt
            try:
                m[k_idx_fp + 1] = sparse.linalg.spsolve(A_matrix, b_rhs)
            except Exception as e:
                print(f"Error solving linear system for FP at time step {k_idx_fp + 1}: {e}")
                m[k_idx_fp + 1] = m[k_idx_fp]
        return m