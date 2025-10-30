"""
Multi-dimensional HJB FDM solver using dimensional splitting.

This module implements dimension-agnostic HJB solvers that work for 2D, 3D, 4D, ...
using Strang splitting (operator splitting in space).

Design Principle:
- Dimension is a parameter, not a constraint
- Single implementation works for any dimension
- Matches MFG_PDE architecture (HighDimMFGProblem, GridBasedMFGProblem)

Algorithm:
- Strang splitting: sequentially solve 1D problems along each dimension
- Second-order accurate in time (symmetric splitting)
- Exact for isotropic Hamiltonians, approximate for anisotropic (with cross-derivatives)

See:
    docs/architecture/proposals/DIMENSION_AGNOSTIC_FDM_ANALYSIS.md
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from . import base_hjb

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.backends.base_backend import BaseBackend
    from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem


def solve_hjb_nd_dimensional_splitting(
    M_density: NDArray,  # (Nt, N1, N2, ..., Nd)
    U_final: NDArray,  # (N1, N2, ..., Nd)
    U_prev: NDArray,  # (Nt, N1, N2, ..., Nd)
    problem: GridBasedMFGProblem,
    max_newton_iterations: int,
    newton_tolerance: float,
    backend: BaseBackend | None = None,
) -> NDArray:  # Returns (Nt, N1, N2, ..., Nd)
    """
    Solve nD HJB using dimensional splitting (Strang splitting).

    Works for any dimension: 2D, 3D, 4D, etc.

    Algorithm:
        Strang splitting for 2nd-order accuracy:
        1. Forward sweeps: dimensions 0, 1, 2, ..., d-1 (half timesteps)
        2. Backward sweeps: dimensions d-1, ..., 2, 1, 0 (half timesteps)

    Args:
        M_density: Density evolution from previous Picard iteration (Nt, N1, N2, ..., Nd)
        U_final: Terminal condition for value function (N1, N2, ..., Nd)
        U_prev: Value function from previous Picard iteration (Nt, N1, N2, ..., Nd)
        problem: GridBasedMFGProblem instance (nD problem)
        max_newton_iterations: Maximum Newton iterations per 1D solve
        newton_tolerance: Newton convergence tolerance
        backend: Backend for array operations (optional)

    Returns:
        U_solution: Value function evolution (Nt, N1, N2, ..., Nd)

    Notes:
        - Exact for isotropic Hamiltonians H = (1/2)|p|^2
        - Approximate for anisotropic Hamiltonians with cross-derivatives (p1*p2 terms)
        - Splitting error is O(dt^2) (Strang splitting)

    Example:
        >>> # 2D problem
        >>> problem_2d = GridBasedMFGProblem(grid=TensorProductGrid(nx=50, ny=50))
        >>> U_2d = solve_hjb_nd_dimensional_splitting(M_2d, U_final_2d, U_prev_2d, problem_2d, ...)
        >>>
        >>> # 3D problem - same function, no code changes!
        >>> problem_3d = GridBasedMFGProblem(grid=TensorProductGrid(nx=30, ny=30, nz=30))
        >>> U_3d = solve_hjb_nd_dimensional_splitting(M_3d, U_final_3d, U_prev_3d, problem_3d, ...)
    """
    Nt = problem.Nt + 1
    ndim = problem.geometry.grid.ndim
    shape = tuple(problem.geometry.grid.num_points[d] - 1 for d in range(ndim))
    dt = problem.dt

    # Initialize solution array
    if backend is not None:
        U_solution = backend.zeros((Nt, *shape))
    else:
        U_solution = np.zeros((Nt, *shape))

    # Set terminal condition
    U_solution[Nt - 1] = U_final

    # Backward time integration with dimensional splitting
    for n in range(Nt - 2, -1, -1):
        U_current = U_solution[n + 1]
        M_np1 = M_density[n + 1]

        # Strang splitting: forward half-steps (dimensions 0 → d-1)
        U = U_current
        for dim in range(ndim):
            U = _sweep_dimension(
                U,
                M_np1,
                problem,
                dt / (2 * ndim),
                dim,
                max_newton_iterations,
                newton_tolerance,
                backend,
            )

        # Backward half-steps (dimensions d-1 → 0)
        for dim in range(ndim - 1, -1, -1):
            U = _sweep_dimension(
                U,
                M_np1,
                problem,
                dt / (2 * ndim),
                dim,
                max_newton_iterations,
                newton_tolerance,
                backend,
            )

        U_solution[n] = U

    return U_solution


def _sweep_dimension(
    U: NDArray,  # Shape: (N1, N2, ..., Nd)
    M: NDArray,  # Shape: (N1, N2, ..., Nd)
    problem: GridBasedMFGProblem,
    dt: float,
    dim: int,  # Which dimension to sweep (0 to d-1)
    max_newton_iterations: int,
    newton_tolerance: float,
    backend: BaseBackend | None = None,
) -> NDArray:  # Returns same shape as U
    """
    Sweep along dimension `dim`, treating all other dimensions as independent slices.

    This is the core of the dimension-agnostic design. For each perpendicular hyperplane,
    solve a 1D HJB problem along the sweep dimension.

    Examples:
        2D with dim=0:
            Iterate over y, solve 1D HJB in x for each y-slice
            U[:, j] = solve_1d_hjb(U[:, j], M[:, j], ...)

        2D with dim=1:
            Iterate over x, solve 1D HJB in y for each x-slice
            U[i, :] = solve_1d_hjb(U[i, :], M[i, :], ...)

        3D with dim=0:
            Iterate over (y, z) pairs, solve 1D HJB in x for each (y, z)
            U[:, j, k] = solve_1d_hjb(U[:, j, k], M[:, j, k], ...)

        3D with dim=1:
            Iterate over (x, z) pairs, solve 1D HJB in y for each (x, z)
            U[i, :, k] = solve_1d_hjb(U[i, :, k], M[i, :, k], ...)

    Args:
        U: Current value function (N1, N2, ..., Nd)
        M: Current density (N1, N2, ..., Nd)
        problem: GridBasedMFGProblem instance
        dt: Timestep for this sweep
        dim: Dimension to sweep along (0 to ndim-1)
        max_newton_iterations: Maximum Newton iterations per 1D solve
        newton_tolerance: Newton tolerance per 1D solve
        backend: Backend for array operations

    Returns:
        U_new: Updated value function (same shape as U)

    Implementation Notes:
        - Uses itertools.product() to iterate over perpendicular dimensions
        - Dynamic slicing: insert slice(None) at position `dim`
        - Each 1D solve is independent (embarrassingly parallel, future optimization)
    """
    shape = U.shape
    ndim = len(shape)

    # Get indices for all dimensions except `dim`
    other_dims = [d for d in range(ndim) if d != dim]
    other_ranges = [range(shape[d]) for d in other_dims]

    # Copy U for modification
    if backend is not None:
        U_new = backend.copy(U)
    else:
        U_new = U.copy()

    # Iterate over all slices perpendicular to dimension `dim`
    for indices in itertools.product(*other_ranges):
        # Build indexing tuple: insert slice(None) at position `dim`
        # Example for 3D, dim=1, indices=(2, 5):
        #   full_idx = [2, slice(None), 5] → (2, :, 5)
        full_idx = list(indices)
        full_idx.insert(dim, slice(None))
        full_idx = tuple(full_idx)

        # Extract 1D slice along dimension `dim`
        U_slice = U[full_idx]  # Shape: (N_dim,)
        M_slice = M[full_idx]  # Shape: (N_dim,)

        # Solve 1D HJB problem along this dimension
        U_new_slice = _solve_1d_hjb_slice(
            U_slice,
            M_slice,
            problem,
            dt,
            dim,
            indices,
            max_newton_iterations,
            newton_tolerance,
            backend,
        )

        # Update solution
        U_new[full_idx] = U_new_slice

    return U_new


def _solve_1d_hjb_slice(
    U_slice: NDArray,  # (N,) for dimension `dim`
    M_slice: NDArray,  # (N,)
    problem: GridBasedMFGProblem,
    dt: float,
    dim: int,  # Which dimension this slice is along
    slice_indices: tuple,  # Indices in other dimensions
    max_newton_iterations: int,
    newton_tolerance: float,
    backend: BaseBackend | None = None,
) -> NDArray:  # Returns (N,)
    """
    Solve 1D HJB problem along one dimension using existing 1D FDM solver.

    This function adapts the nD problem to look like a 1D problem, then calls
    the existing base_hjb.solve_hjb_system_backward() function.

    Strategy:
        1. Create _Problem1DAdapter to make nD slice look like 1D MFGProblem
        2. Call existing 1D FDM solver (reuses all existing code)
        3. Extract solution for this slice

    Args:
        U_slice: Value function slice along dimension `dim` (N,)
        M_slice: Density slice along dimension `dim` (N,)
        problem: Full nD GridBasedMFGProblem
        dt: Timestep for this solve
        dim: Dimension this slice is along
        slice_indices: Indices in perpendicular dimensions
        max_newton_iterations: Maximum Newton iterations
        newton_tolerance: Newton tolerance
        backend: Backend for array operations

    Returns:
        U_new_slice: Updated value function slice (N,)

    Notes:
        - This is where the nD→1D adaptation happens
        - Reuses existing, well-tested 1D HJB solver
        - No new numerical methods needed
    """
    # Create 1D adapter for this slice
    problem_1d = _Problem1DAdapter(problem, dim, slice_indices)

    # Prepare 1D arrays for solve_hjb_system_backward
    # It expects (Nt, Nx) arrays, but we only have one timeslice
    # So create (2, N) arrays: [dummy, actual_data]
    N = len(U_slice)

    if backend is not None:
        M_1d = backend.zeros((2, N))
        U_final_1d = backend.copy(U_slice)
        U_prev_1d = backend.zeros((2, N))
    else:
        M_1d = np.zeros((2, N))
        U_final_1d = U_slice.copy()
        U_prev_1d = np.zeros((2, N))

    # Set data for time index 1 (solve for time index 0)
    M_1d[1, :] = M_slice
    U_prev_1d[1, :] = U_slice

    # Temporarily override problem's Nt and dt for single timestep solve
    original_Nt = problem_1d.Nt
    original_dt = problem_1d.dt
    problem_1d.Nt = 1
    problem_1d.dt = dt

    # Solve 1D HJB problem
    U_solution_1d = base_hjb.solve_hjb_system_backward(
        M_density_from_prev_picard=M_1d,
        U_final_condition_at_T=U_final_1d,
        U_from_prev_picard=U_prev_1d,
        problem=problem_1d,
        max_newton_iterations=max_newton_iterations,
        newton_tolerance=newton_tolerance,
        backend=backend,
    )

    # Restore original parameters
    problem_1d.Nt = original_Nt
    problem_1d.dt = original_dt

    # Extract solution at time index 0
    U_new_slice = U_solution_1d[0, :]

    return U_new_slice


class _Problem1DAdapter:
    """
    Adapter to make 1D slice of GridBasedMFGProblem look like 1D MFGProblem.

    This class bridges the interface mismatch between:
    - GridBasedMFGProblem: nD problem with hamiltonian(x: ndarray, m: ndarray, p: ndarray, t: float)
    - MFGProblem: 1D problem with H(x_idx: int, m_at_x: float, derivs: dict, ...)

    The adapter:
    1. Converts 1D slice index `i` to multi-dimensional grid index
    2. Looks up coordinates if needed
    3. Converts 1D derivative notation to multi-dimensional format
    4. Calls the full problem's Hamiltonian

    Attributes:
        full_problem: The original nD GridBasedMFGProblem
        sweep_dim: Which dimension we're sweeping along (0 to ndim-1)
        fixed_indices: Indices in perpendicular dimensions
        Nx: Number of points in sweep dimension (for 1D interface)
        Dx: Grid spacing in sweep dimension (for 1D interface)
        Nt, dt, T, sigma: Pass through from full problem
    """

    def __init__(
        self,
        full_problem: GridBasedMFGProblem,
        sweep_dim: int,
        fixed_indices: tuple,
    ):
        """
        Initialize adapter for 1D slice.

        Args:
            full_problem: Original nD GridBasedMFGProblem
            sweep_dim: Dimension to sweep along (0 to ndim-1)
            fixed_indices: Fixed indices in perpendicular dimensions

        Example:
            For 3D problem sweeping in y-direction (dim=1) at x=10, z=5:
                adapter = _Problem1DAdapter(problem_3d, sweep_dim=1, fixed_indices=(10, 5))
                # Now adapter.H(i, ...) evaluates Hamiltonian at (10, i, 5)
        """
        self.full_problem = full_problem
        self.sweep_dim = sweep_dim
        self.fixed_indices = fixed_indices

        # Extract 1D parameters from grid
        grid = full_problem.geometry.grid
        self.Nx = grid.num_points[sweep_dim] - 1  # Number of interior points
        self.Dx = grid.spacing[sweep_dim]

        # Pass through time parameters
        self.Nt = full_problem.Nt
        self.dt = full_problem.dt
        self.T = full_problem.T

        # Pass through diffusion coefficient
        # Handle both callable and numeric sigma
        if hasattr(full_problem, "sigma"):
            self.sigma = full_problem.sigma
        elif hasattr(full_problem, "nu"):  # Legacy name
            self.sigma = full_problem.nu
        else:
            self.sigma = 0.1  # Default fallback

    def H(
        self,
        x_idx: int,
        m_at_x: float,
        derivs: dict[tuple, float] | None = None,
        p_values: dict[str, float] | None = None,
    ) -> float:
        """
        Evaluate Hamiltonian for 1D slice.

        Converts 1D interface (x_idx, derivs) to nD interface (coordinates, multi-dim derivs).

        Args:
            x_idx: 1D index along sweep dimension
            m_at_x: Density at this point
            derivs: 1D derivatives {(0,): u, (1,): ∂u/∂x_dim}
            p_values: Legacy gradient notation (deprecated)

        Returns:
            H: Hamiltonian value

        Implementation:
            1. Build multi-dimensional index from 1D index and fixed indices
            2. Look up coordinates from grid
            3. Convert 1D derivatives to multi-dim format
            4. Call full problem's Hamiltonian

        Example:
            3D problem, sweep_dim=1 (y-direction), fixed_indices=(10, 5):
                x_idx=3 → multi_idx=(10, 3, 5) → coordinates at (x=0.2, y=0.06, z=0.1)
                derivs={(1,): 0.5} → multi_derivs={(0,1): 0.5} (∂u/∂y)
        """
        # Build multi-dimensional index
        # Insert x_idx at position sweep_dim
        multi_idx_list = list(self.fixed_indices)
        multi_idx_list.insert(self.sweep_dim, x_idx)
        multi_idx = tuple(multi_idx_list)

        # Get coordinates (if Hamiltonian needs them)
        # For now, pass the multi-dimensional index
        # TODO: May need to look up actual coordinates from grid.vertices

        # Convert 1D derivatives to multi-dimensional format
        if derivs is not None:
            derivs_multidim = self._convert_derivs_to_multidim(derivs)
        elif p_values is not None:
            # Handle legacy p_values format
            # Convert to derivs first, then to multidim
            derivs_1d = {(0,): 0.0, (1,): p_values.get("forward", 0.0)}
            derivs_multidim = self._convert_derivs_to_multidim(derivs_1d)
        else:
            derivs_multidim = {}

        # Call full problem's Hamiltonian
        # Note: This assumes full_problem.H() accepts scalar index x_idx
        # If it needs coordinates, we'll need to look them up from grid
        return self.full_problem.H(
            x_idx=multi_idx,  # Pass as tuple for now
            m_at_x=m_at_x,
            derivs=derivs_multidim,
        )

    def _convert_derivs_to_multidim(self, derivs_1d: dict[tuple, float]) -> dict[tuple, float]:
        """
        Convert 1D derivative notation to multi-dimensional.

        Mapping:
            1D format:  {(0,): u, (1,): ∂u/∂x_dim}
            nD format:  {(0,): u, (..., 1, ...): ∂u/∂x_dim}
                                      ↑
                                   sweep_dim

        Args:
            derivs_1d: 1D derivatives

        Returns:
            derivs_multidim: Multi-dimensional derivatives

        Example (3D, sweep_dim=1):
            Input:  {(0,): 1.5, (1,): 0.3}
            Output: {(0,): 1.5, (0, 1, 0): 0.3}  # ∂u/∂y in 3D
        """
        ndim = self.full_problem.geometry.grid.ndim
        derivs_out = {}

        for key, val in derivs_1d.items():
            if key == (0,):
                # Function value - stays the same
                derivs_out[(0,)] = val
            elif key == (1,):
                # First derivative in sweep direction
                # Build multi-index: (0, ..., 0, 1, 0, ..., 0)
                #                                ↑
                #                            sweep_dim
                multi_key = tuple(1 if d == self.sweep_dim else 0 for d in range(ndim))
                derivs_out[multi_key] = val

        return derivs_out
