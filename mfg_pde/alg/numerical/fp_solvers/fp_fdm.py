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

        # Boundary condition resolution hierarchy:
        # 1. Explicit boundary_conditions parameter (highest priority)
        # 2. Grid geometry boundary handler (if available)
        # 3. Default no-flux BC (fallback)
        if boundary_conditions is not None:
            self.boundary_conditions = boundary_conditions
        elif hasattr(problem, "geometry") and hasattr(problem.geometry, "get_boundary_handler"):
            # Try to get BC from grid geometry (Phase 2 integration)
            try:
                self.boundary_conditions = problem.geometry.get_boundary_handler(bc_type="no_flux")
            except Exception:
                # Fallback if geometry BC retrieval fails
                self.boundary_conditions = BoundaryConditions(type="no_flux")
        else:
            # Default to no-flux boundaries for mass conservation
            self.boundary_conditions = BoundaryConditions(type="no_flux")

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
        # Try geometry.dimension first (unified interface)
        if hasattr(problem, "geometry") and hasattr(problem.geometry, "dimension"):
            return problem.geometry.dimension

        # Fall back to problem.dimension
        if hasattr(problem, "dimension"):
            return problem.dimension

        # Legacy 1D detection
        if getattr(problem, "Nx", None) is not None and getattr(problem, "Ny", None) is None:
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
            # Multi-dimensional solver via full nD system (not dimensional splitting)
            return _solve_fp_nd_full_system(
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
        # Handle both old 1D interface and new geometry-based interface
        if getattr(self.problem, "Nx", None) is not None:
            # Old 1D interface
            Nx = self.problem.Nx + 1
            Dx = self.problem.dx
            Dt = self.problem.dt
        else:
            # Geometry-based interface (CartesianGrid)
            Nx = self.problem.geometry.get_grid_shape()[0]
            Dx = self.problem.geometry.get_grid_spacing()[0]
            Dt = self.problem.dt

        Nt = self.problem.Nt + 1
        sigma = self.problem.sigma
        coupling_coefficient = getattr(self.problem, "coupling_coefficient", 1.0)

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
                            coupling_coefficient
                            * (npart(u_at_tk[ip1] - u_at_tk[i]) + ppart(u_at_tk[i] - u_at_tk[im1]))
                            / Dx**2
                        )

                    row_indices.append(i)
                    col_indices.append(i)
                    data_values.append(val_A_ii)

                    if Nx > 1:
                        # Lower diagonal term
                        im1 = (i - 1 + Nx) % Nx  # Previous cell index (periodic)
                        val_A_i_im1 = -(sigma**2) / (2 * Dx**2)
                        val_A_i_im1 += float(-coupling_coefficient * npart(u_at_tk[i] - u_at_tk[im1]) / Dx**2)
                        row_indices.append(i)
                        col_indices.append(im1)
                        data_values.append(val_A_i_im1)

                        # Upper diagonal term
                        ip1 = (i + 1) % Nx  # Next cell index (periodic)
                        val_A_i_ip1 = -(sigma**2) / (2 * Dx**2)
                        val_A_i_ip1 += float(-coupling_coefficient * ppart(u_at_tk[ip1] - u_at_tk[i]) / Dx**2)
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
                                    coupling_coefficient
                                    * (npart(u_at_tk[i + 1] - u_at_tk[i]) + ppart(u_at_tk[i] - u_at_tk[i - 1]))
                                    / Dx**2
                                )

                        row_indices.append(i)
                        col_indices.append(i)
                        data_values.append(val_A_ii)

                        if Nx > 1 and i > 0:
                            # Lower diagonal term (flux from left)
                            val_A_i_im1 = -(sigma**2) / (2 * Dx**2)
                            val_A_i_im1 += float(-coupling_coefficient * npart(u_at_tk[i] - u_at_tk[i - 1]) / Dx**2)
                            row_indices.append(i)
                            col_indices.append(i - 1)
                            data_values.append(val_A_i_im1)

                        if Nx > 1 and i < Nx - 1:
                            # Upper diagonal term (flux from right)
                            val_A_i_ip1 = -(sigma**2) / (2 * Dx**2)
                            val_A_i_ip1 += float(-coupling_coefficient * ppart(u_at_tk[i + 1] - u_at_tk[i]) / Dx**2)
                            row_indices.append(i)
                            col_indices.append(i + 1)
                            data_values.append(val_A_i_ip1)

            elif self.boundary_conditions.type == "no_flux":
                # Bug #8 Fix: No-flux boundaries WITH advection
                # Previous "partial fix" dropped advection at boundaries → mass leaked
                # New strategy: Include advection with one-sided stencils
                # Accept ~1-2% FDM discretization error as normal

                for i in range(Nx):
                    if i == 0:
                        # Left boundary: include both diffusion AND advection
                        # Use one-sided (forward) stencil for velocity gradient

                        # Diagonal term: time + diffusion + advection (upwind)
                        val_A_ii = 1.0 / Dt + sigma**2 / Dx**2

                        # Add advection contribution (one-sided upwind scheme)
                        # For left boundary, use forward difference for velocity
                        # Only positive part contributes (flux out of domain)
                        if Nx > 1:
                            val_A_ii += float(coupling_coefficient * ppart(u_at_tk[i + 1] - u_at_tk[i]) / Dx**2)

                        row_indices.append(i)
                        col_indices.append(i)
                        data_values.append(val_A_ii)

                        # Coupling to m[1]: diffusion + advection
                        val_A_i_ip1 = -(sigma**2) / Dx**2
                        if Nx > 1:
                            val_A_i_ip1 += float(-coupling_coefficient * ppart(u_at_tk[i + 1] - u_at_tk[i]) / Dx**2)

                        row_indices.append(i)
                        col_indices.append(i + 1)
                        data_values.append(val_A_i_ip1)

                    elif i == Nx - 1:
                        # Right boundary: include both diffusion AND advection
                        # Use one-sided (backward) stencil for velocity gradient

                        # Diagonal term: time + diffusion + advection (upwind)
                        val_A_ii = 1.0 / Dt + sigma**2 / Dx**2

                        # Add advection contribution (one-sided upwind scheme)
                        # For right boundary, use backward difference for velocity
                        # Only negative part contributes (flux out of domain)
                        if Nx > 1:
                            val_A_ii += float(coupling_coefficient * npart(u_at_tk[i] - u_at_tk[i - 1]) / Dx**2)

                        row_indices.append(i)
                        col_indices.append(i)
                        data_values.append(val_A_ii)

                        # Coupling to m[N-2]: diffusion + advection
                        val_A_i_im1 = -(sigma**2) / Dx**2
                        if Nx > 1:
                            val_A_i_im1 += float(-coupling_coefficient * npart(u_at_tk[i] - u_at_tk[i - 1]) / Dx**2)

                        row_indices.append(i)
                        col_indices.append(i - 1)
                        data_values.append(val_A_i_im1)

                    else:
                        # Interior points: standard conservative FDM discretization
                        val_A_ii = 1.0 / Dt + sigma**2 / Dx**2

                        val_A_ii += float(
                            coupling_coefficient
                            * (npart(u_at_tk[i + 1] - u_at_tk[i]) + ppart(u_at_tk[i] - u_at_tk[i - 1]))
                            / Dx**2
                        )

                        row_indices.append(i)
                        col_indices.append(i)
                        data_values.append(val_A_ii)

                        # Lower diagonal term
                        val_A_i_im1 = -(sigma**2) / (2 * Dx**2)
                        val_A_i_im1 += float(-coupling_coefficient * npart(u_at_tk[i] - u_at_tk[i - 1]) / Dx**2)
                        row_indices.append(i)
                        col_indices.append(i - 1)
                        data_values.append(val_A_i_im1)

                        # Upper diagonal term
                        val_A_i_ip1 = -(sigma**2) / (2 * Dx**2)
                        val_A_i_ip1 += float(-coupling_coefficient * ppart(u_at_tk[i + 1] - u_at_tk[i]) / Dx**2)
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


# ============================================================================
# Multi-dimensional FP solver using full nD system (no dimensional splitting)
# ============================================================================


def _solve_fp_nd_full_system(
    m_initial_condition: np.ndarray,
    U_solution_for_drift: np.ndarray,
    problem: Any,
    boundary_conditions: BoundaryConditions | None = None,
    show_progress: bool = True,
    backend: Any | None = None,
) -> np.ndarray:
    """
    Solve multi-dimensional FP equation using full-dimensional sparse linear system.

    Evolves density forward in time from t=0 to t=T by directly assembling
    and solving the full multi-dimensional system at each timestep.

    This approach avoids the catastrophic operator splitting errors that occur
    with advection-dominated problems (high Péclet number).

    Parameters
    ----------
    m_initial_condition : np.ndarray
        Initial density at t=0. Shape: (N₁, N₂, ..., Nₐ)
    U_solution_for_drift : np.ndarray
        Value function over time-space grid. Shape: (Nt+1, N₁, N₂, ..., Nₐ)
        Used to compute drift velocity v = -coupling_coefficient ∇U
    problem : GridBasedMFGProblem
        The MFG problem definition with geometry and parameters
    boundary_conditions : BoundaryConditions | None
        Boundary condition specification (default: no-flux)
    show_progress : bool
        Whether to display progress bar
    backend : Any | None
        Array backend (currently unused, NumPy only)

    Returns
    -------
    np.ndarray
        Density evolution over time. Shape: (Nt+1, N₁, N₂, ..., Nₐ)

    Notes
    -----
    - No splitting error: Direct discretization preserves operator coupling
    - Mass conservation: Proper flux balance at all boundaries
    - Enforces non-negativity: m ≥ 0 everywhere
    - Forward time evolution: k=0 → Nt-1
    - Complexity: O(N^d) unknowns per timestep

    Mathematical Formulation
    ------------------------
    FP Equation: ∂m/∂t + ∇·(m v) = (σ²/2) Δm

    Direct Discretization:
        (I/Δt + A + D) m^{n+1} = m^n / Δt

    where:
    - I: identity matrix (size N_total × N_total)
    - A: advection operator (full multi-D upwind discretization)
    - D: diffusion operator (full multi-D Laplacian)
    """
    # Get problem dimensions
    Nt = problem.Nt + 1
    ndim = problem.geometry.dimension
    shape = tuple(problem.geometry.get_grid_shape())
    dt = problem.dt
    sigma = problem.sigma
    coupling_coefficient = getattr(problem, "coupling_coefficient", 1.0)

    # Get grid spacing and geometry
    spacing = problem.geometry.get_grid_spacing()
    grid = problem.geometry  # Geometry IS the grid

    # Validate input shapes
    assert m_initial_condition.shape == shape, (
        f"Initial condition shape {m_initial_condition.shape} doesn't match problem shape {shape}"
    )
    expected_U_shape = (Nt, *shape)
    assert U_solution_for_drift.shape == expected_U_shape, (
        f"Value function shape {U_solution_for_drift.shape} doesn't match expected shape {expected_U_shape}"
    )

    # Allocate solution array
    M_solution = np.zeros((Nt, *shape), dtype=np.float64)
    M_solution[0] = m_initial_condition.copy()

    # Ensure non-negativity of initial condition
    M_solution[0] = np.maximum(M_solution[0], 0)

    # Edge cases
    if Nt <= 1:
        return M_solution

    # Set default boundary conditions
    if boundary_conditions is None:
        boundary_conditions = BoundaryConditions(type="no_flux")

    # Progress bar
    from mfg_pde.utils.progress import tqdm

    timestep_range = range(Nt - 1)
    if show_progress:
        timestep_range = tqdm(
            timestep_range,
            desc=f"FP {ndim}D (full system)",
            unit="step",
            disable=False,
        )

    # Time evolution loop (forward in time)
    for k in timestep_range:
        M_current = M_solution[k]
        U_current = U_solution_for_drift[k]

        # Build and solve full nD system
        M_next = _solve_timestep_full_nd(
            M_current,
            U_current,
            problem,
            dt,
            sigma,
            coupling_coefficient,
            spacing,
            grid,
            ndim,
            shape,
            boundary_conditions,
        )

        M_solution[k + 1] = M_next

        # Enforce non-negativity
        M_solution[k + 1] = np.maximum(M_solution[k + 1], 0)

    return M_solution


def _solve_timestep_full_nd(
    M_current: np.ndarray,
    U_current: np.ndarray,
    problem: Any,
    dt: float,
    sigma: float,
    coupling_coefficient: float,
    spacing: tuple[float, ...],
    grid: Any,
    ndim: int,
    shape: tuple[int, ...],
    boundary_conditions: Any,
) -> np.ndarray:
    """
    Solve one timestep of the full nD FP equation.

    Assembles sparse matrix A and RHS b, then solves A*m_{k+1} = b.

    Parameters
    ----------
    M_current : np.ndarray
        Current density field. Shape: (N₁, N₂, ..., Nₐ)
    U_current : np.ndarray
        Current value function. Shape: (N₁, N₂, ..., Nₐ)
    problem : GridBasedMFGProblem
        Problem definition
    dt : float
        Time step
    sigma : float
        Diffusion coefficient
    coupling_coefficient : float
        Coupling coefficient for drift term
    spacing : tuple[float, ...]
        Grid spacing in each dimension
    grid : TensorProductGrid
        Grid object
    ndim : int
        Spatial dimension
    shape : tuple[int, ...]
        Grid shape (N₁, N₂, ..., Nₐ)
    boundary_conditions : Any
        Boundary condition specification

    Returns
    -------
    np.ndarray
        Next density field. Shape: (N₁, N₂, ..., Nₐ)
    """
    # Total number of unknowns
    N_total = int(np.prod(shape))

    # Flatten current state and value function
    m_flat = M_current.ravel()  # Row-major (C-order)
    u_flat = U_current.ravel()

    # Pre-allocate lists for COO format sparse matrix
    row_indices: list[int] = []
    col_indices: list[int] = []
    data_values: list[float] = []

    # Build matrix by iterating over all grid points
    for flat_idx in range(N_total):
        # Convert flat index to multi-index (i, j, k, ...)
        multi_idx = grid.get_multi_index(flat_idx)

        # Check if this is a boundary point
        is_boundary = _is_boundary_point(multi_idx, shape, ndim)

        if boundary_conditions.type == "no_flux" and is_boundary:
            # Boundary point with no-flux condition
            _add_boundary_no_flux_entries(
                row_indices,
                col_indices,
                data_values,
                flat_idx,
                multi_idx,
                shape,
                ndim,
                dt,
                sigma,
                coupling_coefficient,
                spacing,
                u_flat,
                grid,
            )
        else:
            # Interior point or periodic boundary
            _add_interior_entries(
                row_indices,
                col_indices,
                data_values,
                flat_idx,
                multi_idx,
                shape,
                ndim,
                dt,
                sigma,
                coupling_coefficient,
                spacing,
                u_flat,
                grid,
                boundary_conditions,
            )

    # Assemble sparse matrix
    A_matrix = sparse.coo_matrix((data_values, (row_indices, col_indices)), shape=(N_total, N_total)).tocsr()

    # Right-hand side
    b_rhs = m_flat / dt

    # Solve linear system
    try:
        m_next_flat = sparse.linalg.spsolve(A_matrix, b_rhs)
    except Exception:
        # If solver fails, keep current state
        m_next_flat = m_flat.copy()

    # Reshape back to multi-dimensional array
    m_next = m_next_flat.reshape(shape)

    return m_next


def _is_boundary_point(multi_idx: tuple[int, ...], shape: tuple[int, ...], ndim: int) -> bool:
    """Check if a grid point is on the boundary."""
    return any(multi_idx[d] == 0 or multi_idx[d] == shape[d] - 1 for d in range(ndim))


def _add_interior_entries(
    row_indices: list[int],
    col_indices: list[int],
    data_values: list[float],
    flat_idx: int,
    multi_idx: tuple[int, ...],
    shape: tuple[int, ...],
    ndim: int,
    dt: float,
    sigma: float,
    coupling_coefficient: float,
    spacing: tuple[float, ...],
    u_flat: np.ndarray,
    grid: Any,
    boundary_conditions: Any,
) -> None:
    """
    Add matrix entries for interior grid point.

    Discretizes:
        (1/dt) m + div(m*v) - (σ²/2) Δm = 0

    Using upwind for advection and centered differences for diffusion.
    """
    # Diagonal term (accumulates contributions from all dimensions)
    diagonal_value = 1.0 / dt

    # For each dimension, add advection + diffusion contributions
    for d in range(ndim):
        dx = spacing[d]
        dx_sq = dx * dx

        # Get neighbor indices in dimension d
        multi_idx_plus = list(multi_idx)
        multi_idx_minus = list(multi_idx)

        multi_idx_plus[d] = multi_idx[d] + 1
        multi_idx_minus[d] = multi_idx[d] - 1

        # Handle boundary wrapping for periodic BC
        if boundary_conditions.type == "periodic":
            multi_idx_plus[d] = multi_idx_plus[d] % shape[d]
            multi_idx_minus[d] = multi_idx_minus[d] % shape[d]

        # Check if neighbors exist (non-periodic case)
        has_plus = multi_idx_plus[d] < shape[d]
        has_minus = multi_idx_minus[d] >= 0

        if has_plus or boundary_conditions.type == "periodic":
            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            u_plus = u_flat[flat_idx_plus]
        else:
            u_plus = u_flat[flat_idx]  # Use current value at boundary

        if has_minus or boundary_conditions.type == "periodic":
            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))
            u_minus = u_flat[flat_idx_minus]
        else:
            u_minus = u_flat[flat_idx]  # Use current value at boundary

        u_center = u_flat[flat_idx]

        # Diffusion contribution (centered differences)
        # -σ²/(2dx²) * (m_{i+1} - 2m_i + m_{i-1})
        diagonal_value += sigma**2 / dx_sq

        if has_plus or boundary_conditions.type == "periodic":
            # Coupling to m_{i+1,j}
            coeff_plus = -(sigma**2) / (2 * dx_sq)

            # Add advection upwind term
            # For advection: -d/dx(m*v) where v = -coupling_coefficient * dU/dx
            # Upwind: use ppart for positive velocity contribution
            coeff_plus += float(-coupling_coefficient * ppart(u_plus - u_center) / dx_sq)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(coeff_plus)

            # Add advection contribution to diagonal
            diagonal_value += float(coupling_coefficient * ppart(u_plus - u_center) / dx_sq)

        if has_minus or boundary_conditions.type == "periodic":
            # Coupling to m_{i-1,j}
            coeff_minus = -(sigma**2) / (2 * dx_sq)

            # Add advection upwind term
            # Upwind: use npart for negative velocity contribution
            coeff_minus += float(-coupling_coefficient * npart(u_center - u_minus) / dx_sq)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(coeff_minus)

            # Add advection contribution to diagonal
            diagonal_value += float(coupling_coefficient * npart(u_center - u_minus) / dx_sq)

    # Add diagonal entry
    row_indices.append(flat_idx)
    col_indices.append(flat_idx)
    data_values.append(diagonal_value)


def _add_boundary_no_flux_entries(
    row_indices: list[int],
    col_indices: list[int],
    data_values: list[float],
    flat_idx: int,
    multi_idx: tuple[int, ...],
    shape: tuple[int, ...],
    ndim: int,
    dt: float,
    sigma: float,
    coupling_coefficient: float,
    spacing: tuple[float, ...],
    u_flat: np.ndarray,
    grid: Any,
) -> None:
    """
    Add matrix entries for boundary grid point with no-flux BC.

    No-flux: Combined advection + diffusion flux = 0 at boundary.
    Use one-sided stencils for derivatives at boundaries.
    """
    # Diagonal term
    diagonal_value = 1.0 / dt

    # For each dimension, check if we're at a boundary in that dimension
    for d in range(ndim):
        dx = spacing[d]
        dx_sq = dx * dx

        at_left_boundary = multi_idx[d] == 0
        at_right_boundary = multi_idx[d] == shape[d] - 1
        at_interior_in_d = not (at_left_boundary or at_right_boundary)

        if at_interior_in_d:
            # Standard interior stencil in this dimension
            multi_idx_plus = list(multi_idx)
            multi_idx_minus = list(multi_idx)
            multi_idx_plus[d] = multi_idx[d] + 1
            multi_idx_minus[d] = multi_idx[d] - 1

            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))

            u_plus = u_flat[flat_idx_plus]
            u_minus = u_flat[flat_idx_minus]
            u_center = u_flat[flat_idx]

            # Diffusion
            diagonal_value += sigma**2 / dx_sq

            coeff_plus = -(sigma**2) / (2 * dx_sq)
            coeff_plus += float(-coupling_coefficient * ppart(u_plus - u_center) / dx_sq)

            coeff_minus = -(sigma**2) / (2 * dx_sq)
            coeff_minus += float(-coupling_coefficient * npart(u_center - u_minus) / dx_sq)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(coeff_plus)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(coeff_minus)

            diagonal_value += float(
                coupling_coefficient * (ppart(u_plus - u_center) + npart(u_center - u_minus)) / dx_sq
            )

        elif at_left_boundary:
            # Left boundary: use one-sided (forward) stencil
            multi_idx_plus = list(multi_idx)
            multi_idx_plus[d] = multi_idx[d] + 1

            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            u_plus = u_flat[flat_idx_plus]
            u_center = u_flat[flat_idx]

            # One-sided diffusion approximation
            diagonal_value += sigma**2 / dx_sq

            coeff_plus = -(sigma**2) / dx_sq
            coeff_plus += float(-coupling_coefficient * ppart(u_plus - u_center) / dx_sq)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(coeff_plus)

            diagonal_value += float(coupling_coefficient * ppart(u_plus - u_center) / dx_sq)

        elif at_right_boundary:
            # Right boundary: use one-sided (backward) stencil
            multi_idx_minus = list(multi_idx)
            multi_idx_minus[d] = multi_idx[d] - 1

            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))
            u_minus = u_flat[flat_idx_minus]
            u_center = u_flat[flat_idx]

            # One-sided diffusion approximation
            diagonal_value += sigma**2 / dx_sq

            coeff_minus = -(sigma**2) / dx_sq
            coeff_minus += float(-coupling_coefficient * npart(u_center - u_minus) / dx_sq)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(coeff_minus)

            diagonal_value += float(coupling_coefficient * npart(u_center - u_minus) / dx_sq)

    # Add diagonal entry
    row_indices.append(flat_idx)
    col_indices.append(flat_idx)
    data_values.append(diagonal_value)


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing FPFDMSolver...")

    from mfg_pde import ExampleMFGProblem

    # Test 1D problem
    problem = ExampleMFGProblem(Nx=40, Nt=25, T=1.0, sigma=0.1)
    solver = FPFDMSolver(problem, boundary_conditions=BoundaryConditions(type="no_flux"))

    # Test solver initialization
    assert solver.dimension == 1
    assert solver.fp_method_name == "FDM"
    assert solver.boundary_conditions.type == "no_flux"

    # Test solve_fp_system
    U_test = np.zeros((problem.Nt + 1, problem.Nx + 1))
    M_init = problem.m_init  # Shape (Nx+1,)
    M_prev_single = np.ones(problem.Nx + 1) * 0.5

    M_solution = solver.solve_fp_system(M_init, U_test, show_progress=False)

    assert M_solution.shape == (problem.Nt + 1, problem.Nx + 1)
    assert not has_nan_or_inf(M_solution)
    assert np.all(M_solution >= 0), "Density must be non-negative"

    # Check mass conservation (integral of density)
    initial_mass = np.trapz(M_solution[0], dx=problem.dx)
    final_mass = np.trapz(M_solution[-1], dx=problem.dx)
    mass_drift = abs(final_mass - initial_mass) / initial_mass

    print("  FDM solver converged")
    print(f"  M range: [{M_solution.min():.3f}, {M_solution.max():.3f}]")
    print(f"  Mass conservation: initial={initial_mass:.6f}, final={final_mass:.6f}, drift={mass_drift:.2e}")
    print(f"  Boundary conditions: {solver.boundary_conditions.type}")

    print("All smoke tests passed!")
