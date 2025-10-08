from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sparse

from mfg_pde.backends.compat import has_nan_or_inf
from mfg_pde.geometry import BoundaryConditions
from mfg_pde.utils.aux_func import npart, ppart

from .base_fp import BaseFPSolver


class FPFDMSolver(BaseFPSolver):
    def __init__(self, problem: Any, boundary_conditions: BoundaryConditions | None = None) -> None:
        super().__init__(problem)
        self.fp_method_name = "FDM"
        # Default to no-flux boundaries for mass conservation testing
        if boundary_conditions is None:
            self.boundary_conditions = BoundaryConditions(type="no_flux")
        else:
            self.boundary_conditions = boundary_conditions

    def solve_fp_system(self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray) -> np.ndarray:
        Nx = self.problem.Nx + 1
        Nt = self.problem.Nt + 1
        Dx = self.problem.Dx
        Dt = self.problem.Dt
        sigma = self.problem.sigma
        coefCT = self.problem.coefCT

        if Nt == 0:
            if self.backend is not None:
                return self.backend.zeros((0, Nx))
            return np.zeros((0, Nx))
        if Nt == 1:
            if self.backend is not None:
                m_sol = self.backend.zeros((1, Nx))
            else:
                m_sol = np.zeros((1, Nx))
            m_sol[0, :] = m_initial_condition
            m_sol[0, :] = np.maximum(m_sol[0, :], 0)
            # Apply boundary conditions
            if self.boundary_conditions.type == "dirichlet":
                m_sol[0, 0] = self.boundary_conditions.left_value
                m_sol[0, -1] = self.boundary_conditions.right_value
            return m_sol

        if self.backend is not None:
            m = self.backend.zeros((Nt, Nx))
        else:
            m = np.zeros((Nt, Nx))
        m[0, :] = m_initial_condition
        m[0, :] = np.maximum(m[0, :], 0)
        # Apply boundary conditions to initial condition
        if self.boundary_conditions.type == "dirichlet":
            m[0, 0] = self.boundary_conditions.left_value
            m[0, -1] = self.boundary_conditions.right_value

        # Pre-allocate lists for COO format, then convert to CSR
        row_indices: list[int] = []
        col_indices: list[int] = []
        data_values: list[float] = []

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

            # Handle different boundary conditions
            if self.boundary_conditions.type == "periodic":
                # Original periodic boundary implementation
                for i in range(Nx):
                    # Diagonal term for m_i^{k+1}
                    val_A_ii = 1.0 / Dt
                    if Nx > 1:
                        val_A_ii += sigma**2 / Dx**2
                        # Advection part of diagonal (outflow from cell i)
                        ip1 = (i + 1) % Nx
                        im1 = (i - 1 + Nx) % Nx
                        val_A_ii += float(
                            coefCT * (npart(u_at_tk[ip1] - u_at_tk[i]) + ppart(u_at_tk[i] - u_at_tk[im1])) / Dx**2
                        )

                    row_indices.append(i)
                    col_indices.append(i)
                    data_values.append(val_A_ii)

                    if Nx > 1:
                        # Lower diagonal term
                        im1 = (i - 1 + Nx) % Nx  # Previous cell index (periodic)
                        val_A_i_im1 = -(sigma**2) / (2 * Dx**2)
                        val_A_i_im1 += float(-coefCT * npart(u_at_tk[i] - u_at_tk[im1]) / Dx**2)
                        row_indices.append(i)
                        col_indices.append(im1)
                        data_values.append(val_A_i_im1)

                        # Upper diagonal term
                        ip1 = (i + 1) % Nx  # Next cell index (periodic)
                        val_A_i_ip1 = -(sigma**2) / (2 * Dx**2)
                        val_A_i_ip1 += float(-coefCT * ppart(u_at_tk[ip1] - u_at_tk[i]) / Dx**2)
                        row_indices.append(i)
                        col_indices.append(ip1)
                        data_values.append(val_A_i_ip1)

            elif self.boundary_conditions.type == "dirichlet":
                # Dirichlet boundary conditions: m[0] = left_value, m[Nx-1] = right_value
                for i in range(Nx):
                    if i == 0 or i == Nx - 1:
                        # Boundary points: identity equation m[i] = boundary_value
                        row_indices.append(i)
                        col_indices.append(i)
                        data_values.append(1.0)
                    else:
                        # Interior points: standard FDM discretization
                        val_A_ii = 1.0 / Dt
                        if Nx > 1:
                            val_A_ii += sigma**2 / Dx**2
                            # Advection part (no wrapping for interior points)
                            if i > 0 and i < Nx - 1:
                                val_A_ii += float(
                                    coefCT
                                    * (npart(u_at_tk[i + 1] - u_at_tk[i]) + ppart(u_at_tk[i] - u_at_tk[i - 1]))
                                    / Dx**2
                                )

                        row_indices.append(i)
                        col_indices.append(i)
                        data_values.append(val_A_ii)

                        if Nx > 1 and i > 0:
                            # Lower diagonal term (flux from left)
                            val_A_i_im1 = -(sigma**2) / (2 * Dx**2)
                            val_A_i_im1 += float(-coefCT * npart(u_at_tk[i] - u_at_tk[i - 1]) / Dx**2)
                            row_indices.append(i)
                            col_indices.append(i - 1)
                            data_values.append(val_A_i_im1)

                        if Nx > 1 and i < Nx - 1:
                            # Upper diagonal term (flux from right)
                            val_A_i_ip1 = -(sigma**2) / (2 * Dx**2)
                            val_A_i_ip1 += float(-coefCT * ppart(u_at_tk[i + 1] - u_at_tk[i]) / Dx**2)
                            row_indices.append(i)
                            col_indices.append(i + 1)
                            data_values.append(val_A_i_ip1)

            elif self.boundary_conditions.type == "no_flux":
                # No-flux boundary conditions using one-sided differences
                # dm/dx = 0 at boundaries (homogeneous Neumann conditions)
                for i in range(Nx):
                    if i == 0:
                        # Left boundary: dm/dx = 0 using forward difference
                        # (m[1] - m[0])/Dx = 0 => m[1] = m[0]
                        val_A_ii = 1.0 / Dt + sigma**2 / Dx**2

                        # Add advection terms at boundary using one-sided derivative for velocity
                        if Nx > 1:
                            # Velocity: v_0 = -coefCT * du/dx
                            # Upwind: ppart(v_0) / Dx for outflow
                            val_A_ii += float(coefCT * ppart((u_at_tk[1] - u_at_tk[0]) / Dx) / Dx)

                        row_indices.append(i)
                        col_indices.append(i)
                        data_values.append(val_A_ii)

                        if Nx > 1:
                            # Coupling to next point (modified for no-flux)
                            val_A_i_ip1 = -(sigma**2) / Dx**2
                            # Add advection flux: npart(v_0) / Dx for inflow from neighbor
                            val_A_i_ip1 += float(-coefCT * npart((u_at_tk[1] - u_at_tk[0]) / Dx) / Dx)
                            row_indices.append(i)
                            col_indices.append(i + 1)
                            data_values.append(val_A_i_ip1)

                    elif i == Nx - 1:
                        # Right boundary: dm/dx = 0 using backward difference
                        # (m[Nx-1] - m[Nx-2])/Dx = 0 => m[Nx-1] = m[Nx-2]
                        val_A_ii = 1.0 / Dt + sigma**2 / Dx**2

                        # Add advection terms at boundary using one-sided derivative for velocity
                        if Nx > 1:
                            # Velocity: v_N = -coefCT * du/dx
                            # Upwind: ppart(v_N) / Dx for outflow (rightward)
                            val_A_ii += float(coefCT * ppart((u_at_tk[Nx - 1] - u_at_tk[Nx - 2]) / Dx) / Dx)

                        row_indices.append(i)
                        col_indices.append(i)
                        data_values.append(val_A_ii)

                        if Nx > 1:
                            # Coupling to previous point (modified for no-flux)
                            val_A_i_im1 = -(sigma**2) / Dx**2
                            # Add advection flux: npart(v_N) / Dx for inflow from neighbor (leftward)
                            val_A_i_im1 += float(-coefCT * npart((u_at_tk[Nx - 1] - u_at_tk[Nx - 2]) / Dx) / Dx)
                            row_indices.append(i)
                            col_indices.append(i - 1)
                            data_values.append(val_A_i_im1)

                    else:
                        # Interior points: standard FDM discretization
                        val_A_ii = 1.0 / Dt + sigma**2 / Dx**2

                        val_A_ii += float(
                            coefCT * (npart(u_at_tk[i + 1] - u_at_tk[i]) + ppart(u_at_tk[i] - u_at_tk[i - 1])) / Dx**2
                        )

                        row_indices.append(i)
                        col_indices.append(i)
                        data_values.append(val_A_ii)

                        # Lower diagonal term
                        val_A_i_im1 = -(sigma**2) / (2 * Dx**2)
                        val_A_i_im1 += float(-coefCT * npart(u_at_tk[i] - u_at_tk[i - 1]) / Dx**2)
                        row_indices.append(i)
                        col_indices.append(i - 1)
                        data_values.append(val_A_i_im1)

                        # Upper diagonal term
                        val_A_i_ip1 = -(sigma**2) / (2 * Dx**2)
                        val_A_i_ip1 += float(-coefCT * ppart(u_at_tk[i + 1] - u_at_tk[i]) / Dx**2)
                        row_indices.append(i)
                        col_indices.append(i + 1)
                        data_values.append(val_A_i_ip1)

            A_matrix = sparse.coo_matrix((data_values, (row_indices, col_indices)), shape=(Nx, Nx)).tocsr()

            # Set up right-hand side
            b_rhs = m[k_idx_fp, :] / Dt

            # Apply boundary conditions to RHS
            if self.boundary_conditions.type == "dirichlet":
                b_rhs[0] = self.boundary_conditions.left_value  # Left boundary
                b_rhs[-1] = self.boundary_conditions.right_value  # Right boundary
            elif self.boundary_conditions.type == "no_flux":
                # For no-flux boundaries, RHS remains as m[k]/Dt
                # The no-flux condition is enforced through the matrix coefficients
                pass

            if self.backend is not None:
                m_next_step_raw = self.backend.zeros((Nx,))
            else:
                m_next_step_raw = np.zeros(Nx, dtype=np.float64)
            try:
                if not A_matrix.nnz > 0 and Nx > 0:
                    m_next_step_raw[:] = m[k_idx_fp, :]
                else:
                    solution = sparse.linalg.spsolve(A_matrix, b_rhs)
                    m_next_step_raw[:] = solution

                if has_nan_or_inf(m_next_step_raw, self.backend):
                    m_next_step_raw[:] = m[k_idx_fp, :]
            except Exception:
                m_next_step_raw[:] = m[k_idx_fp, :]

            m[k_idx_fp + 1, :] = m_next_step_raw

            # Ensure boundary conditions are satisfied
            if self.boundary_conditions.type == "dirichlet":
                m[k_idx_fp + 1, 0] = self.boundary_conditions.left_value
                m[k_idx_fp + 1, -1] = self.boundary_conditions.right_value
            elif self.boundary_conditions.type == "no_flux":
                # No additional enforcement needed - no-flux is built into the discretization
                # Ensure non-negativity
                m[k_idx_fp + 1, :] = np.maximum(m[k_idx_fp + 1, :], 0)

        return m
