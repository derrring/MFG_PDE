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
        Nx = self.problem.Nx
        Nt = self.problem.Nt
        Dx = self.problem.Dx
        Dt = self.problem.Dt
        sigma = self.problem.sigma
        coefCT = self.problem.coefCT

        if Nt == 0:
            return np.zeros((0, Nx))
        if Nt == 1:
            m_sol = np.zeros((1, Nx))
            m_sol[0, :] = m_initial_condition
            return m_sol

        m = np.zeros((Nt, Nx))
        m[0, :] = m_initial_condition

        for k_idx_fp in range(Nt - 1):
            if Dt < 1e-14:
                m[k_idx_fp + 1, :] = m[k_idx_fp, :]
                continue
            if Dx < 1e-14 and Nx > 1:
                m[k_idx_fp + 1, :] = m[k_idx_fp, :]
                continue

            u_at_tk = U_solution_for_drift[k_idx_fp, :]

            A_L = np.zeros(Nx)
            A_D = np.zeros(Nx)
            A_U = np.zeros(Nx)

            # These vectors are for constructing the periodic connections
            A_corner_0_Nm1 = np.zeros(Nx)  # For element A[0, Nx-1]
            A_corner_Nm1_0 = np.zeros(Nx)  # For element A[Nx-1, 0]

            A_D += 1.0 / Dt
            if Nx > 1:
                A_D += sigma**2 / Dx**2
                A_L_val_diff = -(sigma**2) / (2 * Dx**2)
                A_U_val_diff = -(sigma**2) / (2 * Dx**2)
            else:
                A_L_val_diff = 0.0
                A_U_val_diff = 0.0

            if Nx > 1:
                # Diagonal advection terms (same as original notebook)
                A_D[1 : Nx - 1] += (
                    coefCT
                    * (
                        npart(u_at_tk[2:Nx] - u_at_tk[1 : Nx - 1])
                        + ppart(u_at_tk[1 : Nx - 1] - u_at_tk[0 : Nx - 2])
                    )
                    / Dx**2
                )
                A_D[0] += (
                    coefCT
                    * (
                        npart(u_at_tk[1] - u_at_tk[0])
                        + ppart(u_at_tk[0] - u_at_tk[Nx - 1])  # Periodic for ppart
                    )
                    / Dx**2
                )
                A_D[Nx - 1] += (
                    coefCT
                    * (
                        npart(u_at_tk[0] - u_at_tk[Nx - 1])  # Periodic for npart
                        + ppart(u_at_tk[Nx - 1] - u_at_tk[Nx - 2])
                    )
                    / Dx**2
                )

                # Off-diagonal advection & diffusion (same as original notebook for A_L_orig, A_U_orig)
                A_L[1:Nx] += A_L_val_diff  # Diffusion part for A[i, i-1]
                A_L[1:Nx] += (  # Advection part for A[i, i-1]
                    -coefCT * npart(u_at_tk[1:Nx] - u_at_tk[0 : Nx - 1]) / Dx**2
                )

                A_U[0 : Nx - 1] += A_U_val_diff  # Diffusion part for A[i, i+1]
                A_U[0 : Nx - 1] += (  # Advection part for A[i, i+1]
                    -coefCT * ppart(u_at_tk[1:Nx] - u_at_tk[0 : Nx - 1]) / Dx**2
                )

                # Corner terms to match original notebook's Mupcorner[0] and Mlowcorner[Nx-1] logic
                # Mupcorner[0] in original was coeff of m[Nx-1] in eq for m[0] (A[0, Nx-1])
                # Original: - (sigma**2)/(2 * Dx**2) -coefCT * npart(ukm1[0]-ukm1[Nx-1]) / (Dx**2)
                A_corner_0_Nm1[0] = (
                    A_L_val_diff  # Diffusion part for A[0, Nx-1] (from left)
                )
                A_corner_0_Nm1[0] += (
                    -coefCT * npart(u_at_tk[0] - u_at_tk[Nx - 1]) / Dx**2
                )  # Advection

                # Mlowcorner[Nx-1] in original was coeff of m[0] in eq for m[Nx-1] (A[Nx-1, 0])
                # Original: - (sigma**2)/(2 * Dx**2) -coefCT * ppart(ukm1[0]-ukm1[Nx-1]) / (Dx**2)
                A_corner_Nm1_0[Nx - 1] = (
                    A_U_val_diff  # Diffusion part for A[Nx-1, 0] (from right)
                )
                A_corner_Nm1_0[Nx - 1] += (
                    -coefCT * ppart(u_at_tk[0] - u_at_tk[Nx - 1]) / Dx**2
                )  # Advection

            # np.roll is not needed if A_L and A_U are defined directly as the diagonals
            # A_L[i] is A[i,i-1], A_U[i] is A[i,i+1]
            # For spdiags:
            #   offset -1 uses A_L[1:] (A_L[0] is for periodic corner)
            #   offset  1 uses A_U[:-1] (A_U[Nx-1] is for periodic corner)

            if Nx > 1:
                diagonals = [A_D, A_L[1:], A_U[:-1], A_corner_Nm1_0, A_corner_0_Nm1]
                offsets = [0, -1, 1, -(Nx - 1), Nx - 1]
                A_matrix = sparse.diags(
                    diagonals, offsets, shape=(Nx, Nx), format="csr"
                )
                # Explicitly set/add corners if diags doesn't handle it perfectly for csr
                # A_matrix[0, Nx-1] += A_corner_0_Nm1[0] # This is now done by spdiags
                # A_matrix[Nx-1, 0] += A_corner_Nm1_0[Nx-1] # This is now done by spdiags
            else:  # Nx == 1
                A_matrix = sparse.diags([A_D], [0], shape=(Nx, Nx), format="csr")

            b_rhs = m[k_idx_fp, :] / Dt

            try:
                if not A_matrix.nnz > 0 and Nx > 0:
                    m[k_idx_fp + 1, :] = m[k_idx_fp, :]
                else:
                    m_next_step = sparse.linalg.spsolve(A_matrix, b_rhs)
                    if np.any(np.isnan(m_next_step)) or np.any(np.isinf(m_next_step)):
                        print(
                            f"Warning: spsolve for FP at t_step {k_idx_fp+1} resulted in NaN/Inf. Using previous m."
                        )
                        m[k_idx_fp + 1, :] = m[k_idx_fp, :]
                    else:
                        m[k_idx_fp + 1, :] = m_next_step
            except Exception as e:
                print(
                    f"Error solving linear system for FP at time step {k_idx_fp + 1} (index {k_idx_fp}): {e}"
                )
                m[k_idx_fp + 1, :] = m[k_idx_fp, :]

        return m
