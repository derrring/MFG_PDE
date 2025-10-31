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

        # Detect problem dimension
        self.dimension = self._detect_dimension(problem)

    def _detect_dimension(self, problem: Any) -> int:
        """
        Detect the dimension of the problem.

        Returns
        -------
        int
            Problem dimension (1, 2, 3, ...)
        """
        # Check for GridBasedMFGProblem with TensorProductGrid
        if hasattr(problem, "geometry") and hasattr(problem.geometry, "grid"):
            if hasattr(problem.geometry.grid, "dimension"):
                return problem.geometry.grid.dimension
            if hasattr(problem.geometry.grid, "ndim"):
                return problem.geometry.grid.ndim

        # Check for explicit dimension attribute
        if hasattr(problem, "dimension"):
            return problem.dimension

        # Check for 1D attributes (Nx but no Ny)
        if hasattr(problem, "Nx") and not hasattr(problem, "Ny"):
            return 1

        # Default to 1D for backward compatibility
        return 1

    def solve_fp_system(
        self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray, show_progress: bool = True
    ) -> np.ndarray:
        """
        Solve FP system forward in time.

        Automatically routes to 1D or nD solver based on problem dimension.

        Parameters
        ----------
        m_initial_condition : np.ndarray
            Initial density. Shape: (Nx+1,) for 1D or (N1-1, N2-1, ...) for nD
        U_solution_for_drift : np.ndarray
            Value function for drift term. Shape: (Nt+1, Nx+1) for 1D or
            (Nt+1, N1-1, N2-1, ...) for nD
        show_progress : bool
            Whether to show progress bar

        Returns
        -------
        np.ndarray
            Density evolution. Shape: (Nt+1, Nx+1) for 1D or
            (Nt+1, N1-1, N2-1, ...) for nD
        """
        # Route to appropriate solver based on dimension
        if self.dimension == 1:
            return self._solve_fp_1d(m_initial_condition, U_solution_for_drift, show_progress)
        else:
            # Multi-dimensional solver via dimensional splitting
            from . import fp_fdm_multid

            return fp_fdm_multid.solve_fp_nd_dimensional_splitting(
                m_initial_condition=m_initial_condition,
                U_solution_for_drift=U_solution_for_drift,
                problem=self.problem,
                boundary_conditions=self.boundary_conditions,
                show_progress=show_progress,
                backend=self.backend,
            )

    def _solve_fp_1d(
        self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray, show_progress: bool = True
    ) -> np.ndarray:
        """Original 1D FP solver implementation."""
        # Handle both old 1D interface and new GridBasedMFGProblem interface
        if hasattr(self.problem, "Nx"):
            # Old 1D interface
            Nx = self.problem.Nx + 1
            Dx = self.problem.Dx
            Dt = self.problem.Dt
        else:
            # GridBasedMFGProblem interface
            Nx = self.problem.geometry.grid.num_points[0]
            Dx = self.problem.geometry.grid.spacing[0]
            Dt = self.problem.dt

        Nt = self.problem.Nt + 1
        sigma = self.problem.sigma
        coefCT = getattr(self.problem, "coefCT", 1.0)

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

        # Progress bar for forward timesteps
        from mfg_pde.utils.progress import tqdm

        timestep_range = range(Nt - 1)
        if show_progress:
            timestep_range = tqdm(
                timestep_range,
                desc="FP (forward)",
                unit="step",
                disable=False,
            )

        for k_idx_fp in timestep_range:
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
                # Bug #8 Fix: Conservative no-flux boundary conditions
                # Strategy: Use diffusion-only at boundaries to ensure row sum = 1/Dt
                # This sacrifices some accuracy at boundaries for strict mass conservation

                for i in range(Nx):
                    if i == 0:
                        # Left boundary: diffusion-only with one-sided stencil
                        # No-flux: dm/dx = 0 => use forward difference approximation
                        # Discretize: (m[1] - m[0])/Dx = 0 as diffusion boundary condition

                        # Diagonal term (time + diffusion)
                        val_A_ii = 1.0 / Dt + sigma**2 / Dx**2

                        row_indices.append(i)
                        col_indices.append(i)
                        data_values.append(val_A_ii)

                        # Coupling to m[1] (one-sided, diffusion-only)
                        val_A_i_ip1 = -(sigma**2) / Dx**2

                        row_indices.append(i)
                        col_indices.append(i + 1)
                        data_values.append(val_A_i_ip1)

                        # Row sum: (1/Dt + σ²/Δx²) - σ²/Δx² = 1/Dt ✓ Conservative

                    elif i == Nx - 1:
                        # Right boundary: diffusion-only with one-sided stencil

                        # Diagonal term (time + diffusion)
                        val_A_ii = 1.0 / Dt + sigma**2 / Dx**2

                        row_indices.append(i)
                        col_indices.append(i)
                        data_values.append(val_A_ii)

                        # Coupling to m[N-2] (one-sided, diffusion-only)
                        val_A_i_im1 = -(sigma**2) / Dx**2

                        row_indices.append(i)
                        col_indices.append(i - 1)
                        data_values.append(val_A_i_im1)

                        # Row sum: (1/Dt + σ²/Δx²) - σ²/Δx² = 1/Dt ✓ Conservative

                    else:
                        # Interior points: standard conservative FDM discretization
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
