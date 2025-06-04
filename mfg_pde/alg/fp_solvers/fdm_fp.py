import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from .base_fp import BaseFPSolver
from mfg_pde.utils.aux_func import ppart, npart  # Assuming this path is correct


class FdmFPSolver(BaseFPSolver):
    def __init__(self, problem):
        super().__init__(problem)
        self.fp_method_name = "FDM"

    def solve_fp_system(self, m_initial_condition, U_solution_for_drift):
        """
        Solves the full FP system forward in time using FDM.
        Args:
            m_initial_condition (np.array): Initial density M(0,x).
            U_solution_for_drift (np.array): (Nt, Nx) array of value function U(t,x) used for drift.
                                             Note: U_solution_for_drift[k] is U at time t_k.
        Returns:
            np.array: M_solution (Nt, Nx)
        """
        # print(f"****** Solving FP ({self.fp_method_name})")
        Nx = self.problem.Nx
        Nt = self.problem.Nt  # Number of time points
        Dx = self.problem.Dx
        Dt = self.problem.Dt
        sigma = self.problem.sigma
        coefCT = self.problem.coefCT

        if Nt == 0:  # Handle empty time dimension
            return np.zeros((0, Nx))
        if Nt == 1:  # Only initial condition
            m_sol = np.zeros((1, Nx))
            m_sol[0, :] = m_initial_condition
            return m_sol

        m = np.zeros((Nt, Nx))
        m[0, :] = m_initial_condition  # m[0] is density at t_0

        # Loop Nt-1 times to compute m[1] through m[Nt-1]
        # k_idx_fp goes from 0 to Nt-2
        # We compute m[k_idx_fp + 1] (density at t_{k+1}) using m[k_idx_fp] (density at t_k)
        # and U_solution_for_drift[k_idx_fp] (value function at t_k for drift)
        for k_idx_fp in range(Nt - 1):
            if Dt < 1e-14:  # Avoid division by zero if Dt is effectively zero
                print(
                    f"Warning: Dt is very small ({Dt}) in FdmFPSolver. Setting m[k+1]=m[k]."
                )
                m[k_idx_fp + 1, :] = m[k_idx_fp, :]
                continue
            if (
                Dx < 1e-14 and Nx > 1
            ):  # Avoid division by zero if Dx is effectively zero
                print(
                    f"Warning: Dx is very small ({Dx}) with Nx={Nx} in FdmFPSolver. Setting m[k+1]=m[k]."
                )
                m[k_idx_fp + 1, :] = m[k_idx_fp, :]
                continue

            u_at_tk = U_solution_for_drift[
                k_idx_fp, :
            ]  # U at current time step k_idx_fp

            A_L = np.zeros(Nx)
            A_D = np.zeros(Nx)
            A_U = np.zeros(Nx)
            A_upcorner = np.zeros(Nx)
            A_lowcorner = np.zeros(Nx)

            A_D += 1.0 / Dt
            if Nx > 1:  # Diffusion term only if more than one spatial point
                A_D += sigma**2 / Dx**2
                A_L_val_diff = -(sigma**2) / (2 * Dx**2)
                A_U_val_diff = -(sigma**2) / (2 * Dx**2)
            else:  # Nx == 1, no spatial derivatives for diffusion
                A_L_val_diff = 0.0
                A_U_val_diff = 0.0

            if Nx > 1:
                A_D[1 : Nx - 1] += (
                    coefCT
                    * (
                        npart(u_at_tk[2:Nx] - u_at_tk[1 : Nx - 1])
                        + ppart(u_at_tk[1 : Nx - 1] - u_at_tk[0 : Nx - 2])
                    )
                    / Dx**2
                )
                A_D[0] += (  # Periodic BC for drift
                    coefCT
                    * (
                        npart(u_at_tk[1] - u_at_tk[0])
                        + ppart(u_at_tk[0] - u_at_tk[Nx - 1])
                    )
                    / Dx**2
                )
                A_D[Nx - 1] += (  # Periodic BC for drift
                    coefCT
                    * (
                        npart(u_at_tk[0] - u_at_tk[Nx - 1])
                        + ppart(u_at_tk[Nx - 1] - u_at_tk[Nx - 2])
                    )
                    / Dx**2
                )

                A_L[1:Nx] += A_L_val_diff
                A_L[1:Nx] += (
                    -coefCT * npart(u_at_tk[1:Nx] - u_at_tk[0 : Nx - 1]) / Dx**2
                )
                A_U[0 : Nx - 1] += A_U_val_diff
                A_U[0 : Nx - 1] += (
                    -coefCT * ppart(u_at_tk[1:Nx] - u_at_tk[0 : Nx - 1]) / Dx**2
                )
                A_upcorner[0] = (  # Periodic BC for drift and diffusion
                    A_U_val_diff - coefCT * ppart(u_at_tk[1] - u_at_tk[0]) / Dx**2
                )
                A_lowcorner[Nx - 1] = (  # Periodic BC for drift and diffusion
                    A_L_val_diff - coefCT * npart(u_at_tk[0] - u_at_tk[Nx - 1]) / Dx**2
                )
            # else Nx == 1, A_D only has 1/Dt, other arrays remain zero.

            A_L_sp = np.roll(A_L, -1)
            A_U_sp = np.roll(A_U, 1)

            if Nx > 1:
                A_matrix = sparse.spdiags(
                    [A_lowcorner, A_L_sp, A_D, A_U_sp, A_upcorner],
                    [-(Nx - 1), -1, 0, 1, Nx - 1],  # Corrected offsets for spdiags
                    Nx,
                    Nx,
                    format="csr",
                )
            else:  # Nx == 1
                A_matrix = sparse.diags([A_D], [0], shape=(Nx, Nx), format="csr")

            b_rhs = m[k_idx_fp, :] / Dt  # m at current time step k_idx_fp

            try:
                if not A_matrix.nnz > 0 and Nx > 0:  # Check if matrix is all zeros
                    # print(f"Warning: FP matrix is all zeros at k_idx_fp={k_idx_fp}. Setting m[k+1]=m[k].")
                    m[k_idx_fp + 1, :] = m[k_idx_fp, :]
                else:
                    m[k_idx_fp + 1, :] = sparse.linalg.spsolve(A_matrix, b_rhs)
            except Exception as e:
                print(
                    f"Error solving linear system for FP at time step {k_idx_fp + 1} (index {k_idx_fp}): {e}"
                )
                # Fallback: copy previous time step's density
                m[k_idx_fp + 1, :] = m[k_idx_fp, :]

        # print(f"FDM_FP_DEBUG: Returning m with shape {m.shape}. problem.Nt={self.problem.Nt}, problem.Nx={self.problem.Nx}")
        return m
