import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from .base_fp import BaseFPSolver
from mfg_pde.utils.aux_func import ppart, npart


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
            m_sol[0, :] = np.maximum(m_sol[0, :], 0)
            return m_sol

        m = np.zeros((Nt, Nx))
        m[0, :] = m_initial_condition
        m[0, :] = np.maximum(m[0, :], 0)

        # Pre-allocate lists for COO format, then convert to CSR
        row_indices = []
        col_indices = []
        data_values = []

        for k_idx_fp in range(Nt - 1):
            if Dt < 1e-14:
                m[k_idx_fp + 1, :] = m[k_idx_fp, :]
                continue
            if Dx < 1e-14 and Nx > 1:
                m[k_idx_fp + 1, :] = m[k_idx_fp, :]
                continue

            u_at_tk = U_solution_for_drift[k_idx_fp, :]

            row_indices.clear()
            col_indices.clear()
            data_values.clear()

            for i in range(Nx):  # Iterate over each row i (equation for m_i^{k+1})
                # Diagonal term for m_i^{k+1}
                val_A_ii = 1.0 / Dt
                if Nx > 1:
                    val_A_ii += sigma**2 / Dx**2
                    # Advection part of diagonal (outflow from cell i)
                    # Original: MD[i] += coefCT * (npart(ukm1[i+1]-ukm1[i]) + ppart(ukm1[i]-ukm1[i-1])) / (Dx**2)
                    ip1 = (i + 1) % Nx
                    im1 = (i - 1 + Nx) % Nx
                    val_A_ii += (
                        coefCT
                        * (
                            npart(u_at_tk[ip1] - u_at_tk[i])
                            + ppart(u_at_tk[i] - u_at_tk[im1])
                        )
                        / Dx**2
                    )

                row_indices.append(i)
                col_indices.append(i)
                data_values.append(val_A_ii)

                if Nx > 1:
                    # Lower diagonal term for m_{i-1}^{k+1} (flux from left)
                    # Original ML_orig[i] (coeff of m[i-1] in eq for m[i]):
                    #   - (sigma**2)/(2 * Dx**2) - coefCT * npart(u_at_tk[i]-u_at_tk[i-1]) / Dx**2
                    im1 = (i - 1 + Nx) % Nx  # Previous cell index (periodic)
                    val_A_i_im1 = -(sigma**2) / (2 * Dx**2)
                    val_A_i_im1 += -coefCT * npart(u_at_tk[i] - u_at_tk[im1]) / Dx**2
                    row_indices.append(i)
                    col_indices.append(im1)
                    data_values.append(val_A_i_im1)

                    # Upper diagonal term for m_{i+1}^{k+1} (flux from right)
                    # Original MU_orig[i] (coeff of m[i+1] in eq for m[i]):
                    #   - (sigma**2)/(2 * Dx**2) - coefCT * ppart(u_at_tk[i+1]-u_at_tk[i]) / Dx**2
                    ip1 = (i + 1) % Nx  # Next cell index (periodic)
                    val_A_i_ip1 = -(sigma**2) / (2 * Dx**2)
                    val_A_i_ip1 += -coefCT * ppart(u_at_tk[ip1] - u_at_tk[i]) / Dx**2
                    row_indices.append(i)
                    col_indices.append(ip1)
                    data_values.append(val_A_i_ip1)

            A_matrix = sparse.coo_matrix(
                (data_values, (row_indices, col_indices)), shape=(Nx, Nx)
            ).tocsr()
            b_rhs = m[k_idx_fp, :] / Dt

            m_next_step_raw = np.zeros(Nx)
            try:
                if not A_matrix.nnz > 0 and Nx > 0:
                    m_next_step_raw = m[k_idx_fp, :]
                else:
                    m_next_step_raw = sparse.linalg.spsolve(A_matrix, b_rhs)

                if np.any(np.isnan(m_next_step_raw)) or np.any(
                    np.isinf(m_next_step_raw)
                ):
                    m_next_step_raw = m[k_idx_fp, :]
            except Exception as e:
                m_next_step_raw = m[k_idx_fp, :]

            m[k_idx_fp + 1, :] = m_next_step_raw

        return m
