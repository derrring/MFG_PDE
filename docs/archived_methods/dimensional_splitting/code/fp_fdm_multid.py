"""
Multi-dimensional Fokker-Planck FDM solver using dimensional splitting.

This module extends the 1D FP FDM solver to arbitrary dimensions (2D, 3D, 4D, ...)
using operator splitting in space (Strang splitting).

Mathematical Formulation:
-----------------------
FP Equation: ∂m/∂t + ∇·(m v) = (σ²/2) Δm

where v = -coefCT ∇U is the optimal drift computed from the value function U.

Dimensional Splitting:
---------------------
Split the spatial operator into contributions from each dimension:
    L = L₁ + L₂ + ... + Lₐ

where Lᵢ(m) = -∂/∂xᵢ(m vᵢ) + (σ²/2) ∂²m/∂xᵢ²

Strang Splitting (2nd-order accurate):
1. Forward sweep: solve L₁, L₂, ..., Lₐ for Δt/(2d)
2. Backward sweep: solve Lₐ, ..., L₂, L₁ for Δt/(2d)

This gives O(Δt²) + O(Δx²) overall accuracy.

Interface Conventions:
---------------------
GridBasedMFGProblem convention: arrays have shape (N₁-1, N₂-1, ..., Nₐ-1)
- Excludes right boundary in each dimension
- Example: 10 grid points → array shape 9

1D MFGProblem convention: arrays have shape (Nx+1,)
- Includes both boundaries
- Example: Nx=9 intervals → array shape 10

This module handles the conversion via padding/unpadding.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.backends.base import BaseBackend
    from mfg_pde.problems.grid_based_mfg_problem import GridBasedMFGProblem


def solve_fp_nd_dimensional_splitting(
    m_initial_condition: NDArray,
    U_solution_for_drift: NDArray,
    problem: GridBasedMFGProblem,
    boundary_conditions: Any | None = None,
    show_progress: bool = True,
    backend: BaseBackend | None = None,
    enforce_mass_conservation: bool = False,
    mass_conservation_tolerance: float = 0.05,
) -> NDArray:
    """
    Solve multi-dimensional FP equation using dimensional splitting.

    Evolves density forward in time from t=0 to t=T using Strang splitting
    in space. Each sweep solves a 1D FP problem along one coordinate direction.

    Parameters
    ----------
    m_initial_condition : NDArray
        Initial density at t=0. Shape: (N₁-1, N₂-1, ..., Nₐ-1)
    U_solution_for_drift : NDArray
        Value function over time-space grid. Shape: (Nt+1, N₁-1, N₂-1, ..., Nₐ-1)
        Used to compute drift velocity v = -coefCT ∇U
    problem : GridBasedMFGProblem
        The MFG problem definition with geometry and parameters
    boundary_conditions : Any | None
        Boundary condition specification (default: no-flux)
    show_progress : bool
        Whether to display progress bar
    backend : BaseBackend | None
        Array backend (NumPy, PyTorch, JAX)
    enforce_mass_conservation : bool, default=False
        Whether to apply explicit mass renormalization to enforce conservation.
        When False (default), dimensional splitting errors are left uncorrected,
        preserving the natural dynamics at the cost of ~1-5% mass drift.
        When True, mass is renormalized when error exceeds tolerance.
    mass_conservation_tolerance : float, default=0.05
        Relative mass error threshold (0.05 = 5%) for triggering renormalization.
        Only used when enforce_mass_conservation=True.
        Errors below this threshold are considered acceptable numerical artifacts.

    Returns
    -------
    NDArray
        Density evolution over time. Shape: (Nt+1, N₁-1, N₂-1, ..., Nₐ-1)

    Notes
    -----
    - Uses Strang splitting for 2nd-order accuracy: O(Δt²) + O(Δx²)
    - Mass conservation depends on enforce_mass_conservation flag
    - Enforces non-negativity: m ≥ 0 everywhere
    - Forward time evolution: k=0 → Nt-1

    Examples
    --------
    Default behavior (natural dynamics, may drift ~1-5%):
    >>> M = solve_fp_nd_dimensional_splitting(m0, U, problem)

    Strict conservation (renormalize when |error| > 5%):
    >>> M = solve_fp_nd_dimensional_splitting(
    ...     m0, U, problem,
    ...     enforce_mass_conservation=True,
    ...     mass_conservation_tolerance=0.05
    ... )
    """
    # Get problem dimensions
    Nt = problem.Nt + 1
    ndim = problem.geometry.grid.dimension
    shape = tuple(problem.geometry.grid.num_points)
    dt = problem.dt

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

    # Progress bar
    from mfg_pde.utils.progress import tqdm

    timestep_range = range(Nt - 1)
    if show_progress:
        timestep_range = tqdm(
            timestep_range,
            desc="FP nD (forward)",
            unit="step",
            disable=False,
        )

    # Time evolution loop (forward in time)
    for k in timestep_range:
        M_current = M_solution[k]
        U_current = U_solution_for_drift[k]

        # Strang splitting: forward sweep then backward sweep
        # Each dimension gets dt/(2*ndim) timestep

        # Forward sweep: dimensions 0, 1, ..., ndim-1
        M = M_current.copy()
        for dim in range(ndim):
            M = _sweep_dimension(
                M,
                U_current,
                problem,
                dt / (2 * ndim),
                dim,
                boundary_conditions,
                backend,
            )

        # Backward sweep: dimensions ndim-1, ..., 1, 0
        for dim in range(ndim - 1, -1, -1):
            M = _sweep_dimension(
                M,
                U_current,
                problem,
                dt / (2 * ndim),
                dim,
                boundary_conditions,
                backend,
            )

        M_solution[k + 1] = M

        # Enforce non-negativity
        M_solution[k + 1] = np.maximum(M_solution[k + 1], 0)

        # Optional mass conservation enforcement
        if enforce_mass_conservation:
            # Compute initial and current mass
            dV = float(np.prod(problem.geometry.grid.spacing))
            mass_initial = np.sum(M_solution[0]) * dV
            mass_current = np.sum(M_solution[k + 1]) * dV

            # Only renormalize if error exceeds tolerance
            if mass_current > 1e-16:  # Avoid division by zero
                relative_error = abs(mass_current - mass_initial) / (mass_initial + 1e-16)
                if relative_error > mass_conservation_tolerance:
                    M_solution[k + 1] = M_solution[k + 1] * (mass_initial / mass_current)

    return M_solution


def _sweep_dimension(
    M_in: NDArray,
    U_current: NDArray,
    problem: GridBasedMFGProblem,
    dt: float,
    sweep_dim: int,
    boundary_conditions: Any | None,
    backend: BaseBackend | None,
) -> NDArray:
    """
    Perform one dimensional sweep of the FP equation.

    Solves 1D FP problems along hyperplanes perpendicular to sweep_dim.

    Parameters
    ----------
    M_in : NDArray
        Input density field. Shape: (N₁-1, N₂-1, ..., Nₐ-1)
    U_current : NDArray
        Value function at current time. Shape: (N₁-1, N₂-1, ..., Nₐ-1)
    problem : GridBasedMFGProblem
        The MFG problem definition
    dt : float
        Time step for this sweep (typically dt/(2*ndim))
    sweep_dim : int
        Dimension along which to sweep (0, 1, ..., ndim-1)
    boundary_conditions : Any | None
        Boundary condition specification
    backend : BaseBackend | None
        Array backend

    Returns
    -------
    NDArray
        Updated density field after sweep. Shape: (N₁-1, N₂-1, ..., Nₐ-1)
    """
    ndim = problem.geometry.grid.dimension
    shape = M_in.shape

    # Get dimensions perpendicular to sweep dimension
    perp_dims = [d for d in range(ndim) if d != sweep_dim]

    # Allocate output
    M_out = M_in.copy()

    # Iterate over all hyperplanes perpendicular to sweep_dim
    if len(perp_dims) == 0:
        # 1D case: only one hyperplane
        ranges = [range(1)]
    else:
        ranges = [range(shape[d]) for d in perp_dims]

    for perp_indices in itertools.product(*ranges):
        # Build full indices for slicing
        full_indices = [slice(None)] * ndim
        for i, d in enumerate(perp_dims):
            full_indices[d] = perp_indices[i]
        full_indices[sweep_dim] = slice(None)
        full_indices = tuple(full_indices)

        # Extract 1D slices
        M_slice = M_in[full_indices]  # Shape: (N_sweep_dim,)
        U_slice = U_current[full_indices]  # Shape: (N_sweep_dim,)

        # No padding needed - GridBasedMFGProblem now includes all boundaries
        # Both conventions match after the fix!
        M_slice_padded = M_slice
        U_slice_padded = U_slice

        # Create 1D problem adapter with sweep timestep
        problem_1d = _FPProblem1DAdapter(
            full_problem=problem,
            sweep_dim=sweep_dim,
            fixed_indices=perp_indices,
            sweep_dt=dt,  # Pass the sweep timestep (dt/(2*ndim))
        )

        # Solve 1D FP problem
        # Import here to avoid circular dependency
        from . import fp_fdm

        # Create temporary 1D solver
        solver_1d = fp_fdm.FPFDMSolver(
            problem=problem_1d,
            boundary_conditions=boundary_conditions,
        )
        solver_1d.backend = backend

        # Prepare inputs for 1D solver
        # 1D solver expects:
        # - m_initial_condition: (Nx+1,)
        # - U_solution_for_drift: (2, Nx+1)  # Two time levels: current and next
        # But we only have one time level (current), so we duplicate it
        U_for_1d = np.stack([U_slice_padded, U_slice_padded], axis=0)

        # Solve 1D FP for one timestep
        # Note: 1D solver's dt is already set in problem_1d
        M_solution_1d = solver_1d.solve_fp_system(
            m_initial_condition=M_slice_padded,
            U_solution_for_drift=U_for_1d,
            show_progress=False,  # Suppress nested progress bars
        )

        # Extract result (1D solver returns shape (2, Nx+1), we want second timestep)
        # No boundary removal needed - conventions now match!
        M_new_slice = M_solution_1d[1, :]

        # Store back into output array
        M_out[full_indices] = M_new_slice

    return M_out


class _FPProblem1DAdapter:
    """
    Adapter to make 1D slice of GridBasedMFGProblem look like 1D MFGProblem for FP solver.

    This adapter bridges the interface between:
    - GridBasedMFGProblem: nD problem with TensorProductGrid
    - 1D FP FDM solver: expects 1D problem with Nx, Dx, Dt, sigma, coefCT

    The adapter extracts a 1D slice along one dimension while holding other dimensions fixed.
    """

    def __init__(
        self,
        full_problem: GridBasedMFGProblem,
        sweep_dim: int,
        fixed_indices: tuple[int, ...],
        sweep_dt: float | None = None,
    ):
        """
        Create 1D problem adapter for FP solver.

        Parameters
        ----------
        full_problem : GridBasedMFGProblem
            The full nD MFG problem
        sweep_dim : int
            Dimension along which to sweep (0, 1, ..., ndim-1)
        fixed_indices : tuple[int, ...]
            Fixed indices in perpendicular dimensions
        sweep_dt : float | None
            Timestep for this sweep (typically dt/(2*ndim) for Strang splitting)
            If None, uses full_problem.dt
        """
        self.full_problem = full_problem
        self.sweep_dim = sweep_dim
        self.fixed_indices = fixed_indices

        # Extract 1D grid parameters
        grid = full_problem.geometry.grid
        # BUG FIX: grid.num_points now stores actual points, not intervals
        # For consistency with GridBasedMFGProblem which includes all boundaries:
        # - grid.num_points[dim] = N (actual points)
        # - 1D solver needs Nx = N-1 (intervals) to get Nx+1 = N points
        # BUT: Due to how 1D solver is called with already-extracted slices,
        # we need Nx = N-1 where N is the slice length
        self.num_grid_points_x = grid.num_points[sweep_dim] - 1
        self.grid_spacing_x = grid.spacing[sweep_dim]
        self.Nt = 1  # Single timestep for sweep
        self.dt = sweep_dt if sweep_dt is not None else full_problem.dt  # Use sweep dt

        # Backward compatibility aliases for old naming convention
        self.Nx = self.num_grid_points_x
        self.Dx = self.grid_spacing_x
        self.Dt = self.dt
        self.sigma = full_problem.sigma
        self.coefCT = getattr(full_problem, "coefCT", 1.0)

    def __repr__(self):
        return f"_FPProblem1DAdapter(sweep_dim={self.sweep_dim}, Nx={self.num_grid_points_x}, Dx={self.grid_spacing_x:.4f})"
