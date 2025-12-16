"""Time-stepping routines for multi-dimensional Fokker-Planck FDM solver.

This module contains the forward time evolution logic for solving the FP equation
using finite differences on multi-dimensional grids.

Module structure per issue #388:
    fp_fdm_time_stepping.py - When - time integration (forward Euler, implicit schemes)

Functions:
    solve_fp_nd_full_system: Main time evolution loop for nD FP equation
    solve_timestep_tensor_explicit: Explicit timestep with tensor diffusion
    solve_timestep_full_nd: Implicit timestep solver for scalar diffusion

Mathematical Background:
    FP Equation: dm/dt + div(alpha * m) = (sigma^2/2) * Laplacian(m)

    The implicit scheme solves:
        (I/dt + A + D) m^{n+1} = m^n / dt

    where A is advection, D is diffusion operators.

Note:
    Functions are exported without leading underscore but should be imported
    with underscore alias in fp_fdm.py for internal use:

        from .fp_fdm_time_stepping import (
            solve_fp_nd_full_system as _solve_fp_nd_full_system,
        )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sparse

from mfg_pde.geometry import BoundaryConditions
from mfg_pde.utils.pde_coefficients import CoefficientField

# Import from responsibility-based modules per issue #388
from .fp_fdm_advection import compute_advection_term_nd
from .fp_fdm_alg_divergence_centered import (
    add_boundary_no_flux_entries_divergence_centered,
    add_interior_entries_divergence_centered,
)
from .fp_fdm_alg_divergence_upwind import add_interior_entries_divergence_upwind
from .fp_fdm_alg_gradient_centered import (
    add_boundary_no_flux_entries_gradient_centered,
    add_interior_entries_gradient_centered,
)
from .fp_fdm_alg_gradient_upwind import (
    add_boundary_no_flux_entries_gradient_upwind,
    add_interior_entries_gradient_upwind,
)
from .fp_fdm_bc import add_boundary_no_flux_entries_divergence_upwind
from .fp_fdm_operators import is_boundary_point

if TYPE_CHECKING:
    from collections.abc import Callable


def solve_fp_nd_full_system(
    m_initial_condition: np.ndarray,
    U_solution_for_drift: np.ndarray,
    problem: Any,
    boundary_conditions: BoundaryConditions | None = None,
    show_progress: bool = True,
    backend: Any | None = None,
    diffusion_field: float | np.ndarray | Any | None = None,
    tensor_diffusion_field: np.ndarray | Callable | None = None,
    advection_scheme: str = "upwind",
    # Deprecated parameter for backward compatibility
    conservative: bool | None = None,
) -> np.ndarray:
    """
    Solve multi-dimensional FP equation using full-dimensional sparse linear system.

    Evolves density forward in time from t=0 to t=T by directly assembling
    and solving the full multi-dimensional system at each timestep.

    This approach avoids the catastrophic operator splitting errors that occur
    with advection-dominated problems (high Peclet number).

    Parameters
    ----------
    m_initial_condition : np.ndarray
        Initial density at t=0. Shape: (N1, N2, ..., Nd)
    U_solution_for_drift : np.ndarray
        Value function over time-space grid. Shape: (Nt+1, N1, N2, ..., Nd)
        Used to compute drift velocity v = -coupling_coefficient * grad(U)
    problem : MFGProblem
        The MFG problem definition with geometry and parameters
    boundary_conditions : BoundaryConditions | None
        Boundary condition specification (default: no-flux)
    show_progress : bool
        Whether to display progress bar
    backend : Any | None
        Array backend (currently unused, NumPy only)
    diffusion_field : float | np.ndarray | Callable | None
        Optional diffusion override (Phase 2.4):
        - None: Use problem.sigma
        - float: Constant diffusion
        - ndarray: Spatially/temporally varying diffusion
        - Callable: State-dependent diffusion D(t, x, m) -> float | ndarray
    tensor_diffusion_field : np.ndarray | Callable | None
        Tensor diffusion coefficient (Phase 3.0)
    advection_scheme : str
        Advection term discretization scheme. Options:
        - "gradient_centered": Gradient form + central differences
          (NOT conservative, oscillates for Peclet > 2)
        - "gradient_upwind": Gradient form + upwind differences
          (conservative via row sums, stable, O(dx))
        - "divergence_centered": Divergence form + centered fluxes
          (conservative via telescoping, oscillates for Peclet > 2)
        - "divergence_upwind": Divergence form + upwind fluxes
          (conservative via telescoping, stable, O(dx))
        Legacy aliases: "centered"->"gradient_centered",
        "upwind"->"gradient_upwind", "flux"->"divergence_upwind"
    conservative : bool | None
        DEPRECATED. Use advection_scheme instead.

    Returns
    -------
    np.ndarray
        Density evolution over time. Shape: (Nt+1, N1, N2, ..., Nd)

    Notes
    -----
    - No splitting error: Direct discretization preserves operator coupling
    - Mass conservation: Proper flux balance at all boundaries
    - Enforces non-negativity: m >= 0 everywhere
    - Forward time evolution: k=0 -> Nt-1
    - Complexity: O(N^d) unknowns per timestep

    Mathematical Formulation
    ------------------------
    FP Equation: dm/dt + div(m * v) = (sigma^2/2) * Laplacian(m)

    Direct Discretization:
        (I/dt + A + D) m^{n+1} = m^n / dt

    where:
    - I: identity matrix (size N_total x N_total)
    - A: advection operator (full multi-D upwind discretization)
    - D: diffusion operator (full multi-D Laplacian)
    """
    # Get problem dimensions
    # Use U_solution shape for timestep count (allows flexible input sizes)
    Nt = U_solution_for_drift.shape[0]
    ndim = problem.geometry.dimension
    shape = tuple(problem.geometry.get_grid_shape())
    dt = problem.dt
    coupling_coefficient = getattr(problem, "coupling_coefficient", 1.0)

    # Get grid spacing and geometry
    spacing = problem.geometry.get_grid_spacing()
    grid = problem.geometry  # Geometry IS the grid

    # Handle tensor diffusion (Phase 3.0) vs scalar diffusion
    use_tensor_diffusion = tensor_diffusion_field is not None

    if use_tensor_diffusion:
        # Tensor diffusion path
        tensor_base = tensor_diffusion_field
        sigma_base = None  # Not used for tensor diffusion
    else:
        # Scalar diffusion path (Phase 2.4)
        if diffusion_field is None:
            sigma_base = problem.sigma
        elif isinstance(diffusion_field, (int, float)):
            sigma_base = float(diffusion_field)
        elif callable(diffusion_field):
            # Callable: will be evaluated per timestep
            sigma_base = diffusion_field
        elif isinstance(diffusion_field, np.ndarray):
            # Array: spatially or spatiotemporally varying
            sigma_base = diffusion_field
        else:
            sigma_base = problem.sigma
        tensor_base = None

    # Validate input shapes (spatial dimensions must match)
    assert m_initial_condition.shape == shape, (
        f"Initial condition shape {m_initial_condition.shape} doesn't match problem shape {shape}"
    )
    # Only validate spatial dimensions of U_solution (timestep count is flexible)
    if Nt > 0:
        U_spatial_shape = U_solution_for_drift.shape[1:]
        assert U_spatial_shape == shape, (
            f"Value function spatial shape {U_spatial_shape} doesn't match problem shape {shape}"
        )

    # Edge case: zero timesteps - return empty array
    if Nt == 0:
        return np.zeros((0, *shape), dtype=np.float64)

    # Allocate solution array
    M_solution = np.zeros((Nt, *shape), dtype=np.float64)
    M_solution[0] = m_initial_condition.copy()

    # Ensure non-negativity of initial condition
    M_solution[0] = np.maximum(M_solution[0], 0)

    # Enforce Dirichlet BC on initial condition (for 1D problems)
    if boundary_conditions is not None and boundary_conditions.type == "dirichlet" and ndim == 1:
        M_solution[0, 0] = boundary_conditions.left_value
        M_solution[0, -1] = boundary_conditions.right_value

    # Edge cases
    if Nt <= 1:
        return M_solution

    # Set default boundary conditions
    if boundary_conditions is None:
        boundary_conditions = BoundaryConditions(type="no_flux")

    # Progress bar for forward timesteps
    # n_time_points - 1 steps to go from t=0 to t=T
    from mfg_pde.utils.progress import RichProgressBar

    timestep_range = range(Nt - 1)
    if show_progress:
        timestep_range = RichProgressBar(
            timestep_range,
            desc=f"FP {ndim}D (full system)",
            unit="step",
            disable=False,
        )

    # Time evolution loop (forward in time)
    for k in timestep_range:
        M_current = M_solution[k]
        U_current = U_solution_for_drift[k]

        if use_tensor_diffusion:
            # Tensor diffusion path (Phase 3.0) - explicit timestepping
            M_next = solve_timestep_tensor_explicit(
                M_current,
                U_current,
                problem,
                dt,
                tensor_base,
                coupling_coefficient,
                spacing,
                grid,
                ndim,
                shape,
                boundary_conditions,
                k,
            )
        else:
            # Scalar diffusion path - implicit solver
            # Determine diffusion at current timestep using CoefficientField abstraction
            diffusion = CoefficientField(sigma_base, problem.sigma, "diffusion_field", dimension=ndim)
            sigma_at_k = diffusion.evaluate_at(timestep_idx=k, grid=grid.coordinates, density=M_current, dt=dt)

            # Build and solve full nD system
            M_next = solve_timestep_full_nd(
                M_current,
                U_current,
                problem,
                dt,
                sigma_at_k,
                coupling_coefficient,
                spacing,
                grid,
                ndim,
                shape,
                boundary_conditions,
                advection_scheme=advection_scheme,
                conservative=conservative,
            )

        M_solution[k + 1] = M_next

        # Enforce non-negativity
        M_solution[k + 1] = np.maximum(M_solution[k + 1], 0)

        # Enforce Dirichlet boundary conditions (for 1D problems)
        if boundary_conditions.type == "dirichlet" and ndim == 1:
            M_solution[k + 1, 0] = boundary_conditions.left_value
            M_solution[k + 1, -1] = boundary_conditions.right_value

    return M_solution


def solve_timestep_tensor_explicit(
    M_current: np.ndarray,
    U_current: np.ndarray,
    problem: Any,
    dt: float,
    tensor_field: np.ndarray | Callable,
    coupling_coefficient: float,
    spacing: tuple[float, ...],
    grid: Any,
    ndim: int,
    shape: tuple[int, ...],
    boundary_conditions: Any,
    timestep_idx: int,
) -> np.ndarray:
    """
    Solve one timestep with tensor diffusion using explicit Forward Euler.

    Implements: m^{k+1} = m^k + dt * (div(Sigma * grad(m)) - div(alpha * m))

    Parameters
    ----------
    M_current : np.ndarray
        Current density
    U_current : np.ndarray
        Current value function
    problem : Any
        MFG problem
    dt : float
        Time step
    tensor_field : np.ndarray or callable
        Tensor diffusion coefficient Sigma
    coupling_coefficient : float
        Drift coupling coefficient
    spacing : tuple
        Grid spacing (dx, dy, ...)
    grid : Any
        Geometry/grid object
    ndim : int
        Spatial dimension
    shape : tuple
        Grid shape
    boundary_conditions : BoundaryConditions
        Boundary condition specification
    timestep_idx : int
        Current timestep index

    Returns
    -------
    np.ndarray
        Updated density at next timestep
    """
    from mfg_pde.utils.numerical.tensor_operators import divergence_tensor_diffusion_nd

    # Evaluate tensor at current state
    if callable(tensor_field):
        # Callable tensor: Sigma(t, x, m) -> (d, d) array
        t = timestep_idx * dt
        Sigma = np.zeros((*shape, ndim, ndim))
        for idx in np.ndindex(shape):
            x_coords = np.array([grid.coordinates[d][idx[d]] for d in range(ndim)])
            m_at_point = M_current[idx]
            Sigma[idx] = tensor_field(t, x_coords, m_at_point)

        # Validate PSD
        coeff = CoefficientField(tensor_field, None, "tensor_diffusion_field", dimension=ndim)
        coeff.validate_tensor_psd(Sigma)
    elif isinstance(tensor_field, np.ndarray):
        # Array tensor: constant or spatially varying
        if tensor_field.ndim == 2:
            # Constant tensor (d, d)
            Sigma = tensor_field
        elif tensor_field.ndim == ndim + 2:
            # Spatially varying (*shape, d, d)
            Sigma = tensor_field
        else:
            raise ValueError(
                f"tensor_field array must have shape (d, d) or (*shape, d, d), "
                f"got shape {tensor_field.shape} for ndim={ndim}"
            )

        # Validate PSD
        coeff = CoefficientField(tensor_field, None, "tensor_diffusion_field", dimension=ndim)
        coeff.validate_tensor_psd(Sigma)
    else:
        raise TypeError(f"tensor_field must be np.ndarray or callable, got {type(tensor_field)}")

    # Compute tensor diffusion term: div(Sigma * grad(m))
    diffusion_term = divergence_tensor_diffusion_nd(M_current, Sigma, spacing, boundary_conditions)

    # Compute advection term: div(alpha * m)
    # Use upwind scheme from fp_fdm_advection module
    advection_term = compute_advection_term_nd(
        M_current, U_current, coupling_coefficient, spacing, ndim, boundary_conditions
    )

    # Explicit Forward Euler update
    M_next = M_current + dt * (diffusion_term - advection_term)

    return M_next


def solve_timestep_full_nd(
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
    advection_scheme: str = "upwind",
    conservative: bool | None = None,
) -> np.ndarray:
    """
    Solve one timestep of the full nD FP equation.

    Assembles sparse matrix A and RHS b, then solves A*m_{k+1} = b.

    Parameters
    ----------
    M_current : np.ndarray
        Current density field. Shape: (N1, N2, ..., Nd)
    U_current : np.ndarray
        Current value function. Shape: (N1, N2, ..., Nd)
    problem : MFGProblem
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
        Grid shape (N1, N2, ..., Nd)
    boundary_conditions : Any
        Boundary condition specification
    advection_scheme : str
        Advection term discretization scheme. Options:
        - "gradient_centered": Gradient form + central differences
          (NOT conservative, oscillates for Peclet > 2)
        - "gradient_upwind": Gradient form + upwind differences
          (conservative via row sums, stable, O(dx))
        - "divergence_centered": Divergence form + centered fluxes
          (conservative via telescoping, oscillates for Peclet > 2)
        - "divergence_upwind": Divergence form + upwind fluxes
          (conservative via telescoping, stable, O(dx))
        Legacy aliases: "centered"->"gradient_centered",
        "upwind"->"gradient_upwind", "flux"->"divergence_upwind"
    conservative : bool | None
        DEPRECATED. Use advection_scheme instead.
        If True, maps to advection_scheme="flux".
        If False, maps to advection_scheme="upwind".

    Returns
    -------
    np.ndarray
        Next density field. Shape: (N1, N2, ..., Nd)
    """
    # Handle backward compatibility: conservative parameter overrides advection_scheme
    if conservative is not None:
        import warnings

        warnings.warn(
            "The 'conservative' parameter is deprecated. "
            "Use advection_scheme='divergence_upwind' for conservative or 'gradient_upwind' for non-conservative.",
            DeprecationWarning,
            stacklevel=2,
        )
        advection_scheme = "divergence_upwind" if conservative else "gradient_upwind"

    # Map legacy scheme names to new names
    scheme_aliases = {
        "centered": "gradient_centered",
        "upwind": "gradient_upwind",
        "flux": "divergence_upwind",
    }
    advection_scheme = scheme_aliases.get(advection_scheme, advection_scheme)

    # Validate scheme name
    valid_schemes = {"gradient_centered", "gradient_upwind", "divergence_centered", "divergence_upwind"}
    if advection_scheme not in valid_schemes:
        raise ValueError(f"Unknown advection_scheme '{advection_scheme}'. Valid options: {sorted(valid_schemes)}")
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

        # Extract local diffusion coefficient (scalar or from spatially varying array)
        if isinstance(sigma, np.ndarray):
            # For spatially varying diffusion, extract value at this grid point
            sigma_local = float(sigma[multi_idx])
        else:
            sigma_local = sigma

        # Check if this is a boundary point
        is_boundary = is_boundary_point(multi_idx, shape, ndim)

        # Determine BC type - for mixed BC, default to no-flux behavior
        # Handle both legacy BC interface and new BoundaryConditionManager2D
        if hasattr(boundary_conditions, "is_uniform") and hasattr(boundary_conditions, "type"):
            is_no_flux = boundary_conditions.is_uniform and boundary_conditions.type == "no_flux"
            is_uniform = boundary_conditions.is_uniform
        else:
            # For BoundaryConditionManager2D or unknown types, default to no-flux
            is_no_flux = True
            is_uniform = False

        # For mixed BC (not uniform), treat boundaries with default no-flux behavior
        # The actual BC application will be handled by tensor_operators
        if (is_no_flux or not is_uniform) and is_boundary:
            # Boundary point with no-flux condition
            # Select boundary function based on advection scheme
            if advection_scheme == "gradient_centered":
                add_boundary_no_flux_entries_gradient_centered(
                    row_indices,
                    col_indices,
                    data_values,
                    flat_idx,
                    multi_idx,
                    shape,
                    ndim,
                    dt,
                    sigma_local,
                    coupling_coefficient,
                    spacing,
                    u_flat,
                    grid,
                )
            elif advection_scheme == "gradient_upwind":
                add_boundary_no_flux_entries_gradient_upwind(
                    row_indices,
                    col_indices,
                    data_values,
                    flat_idx,
                    multi_idx,
                    shape,
                    ndim,
                    dt,
                    sigma_local,
                    coupling_coefficient,
                    spacing,
                    u_flat,
                    grid,
                )
            elif advection_scheme == "divergence_centered":
                add_boundary_no_flux_entries_divergence_centered(
                    row_indices,
                    col_indices,
                    data_values,
                    flat_idx,
                    multi_idx,
                    shape,
                    ndim,
                    dt,
                    sigma_local,
                    coupling_coefficient,
                    spacing,
                    u_flat,
                    grid,
                )
            else:  # "divergence_upwind"
                add_boundary_no_flux_entries_divergence_upwind(
                    row_indices,
                    col_indices,
                    data_values,
                    flat_idx,
                    multi_idx,
                    shape,
                    ndim,
                    dt,
                    sigma_local,
                    coupling_coefficient,
                    spacing,
                    u_flat,
                    grid,
                )
        else:
            # Interior point or periodic boundary
            # Select interior function based on advection scheme
            if advection_scheme == "gradient_centered":
                add_interior_entries_gradient_centered(
                    row_indices,
                    col_indices,
                    data_values,
                    flat_idx,
                    multi_idx,
                    shape,
                    ndim,
                    dt,
                    sigma_local,
                    coupling_coefficient,
                    spacing,
                    u_flat,
                    grid,
                    boundary_conditions,
                )
            elif advection_scheme == "gradient_upwind":
                add_interior_entries_gradient_upwind(
                    row_indices,
                    col_indices,
                    data_values,
                    flat_idx,
                    multi_idx,
                    shape,
                    ndim,
                    dt,
                    sigma_local,
                    coupling_coefficient,
                    spacing,
                    u_flat,
                    grid,
                    boundary_conditions,
                )
            elif advection_scheme == "divergence_centered":
                add_interior_entries_divergence_centered(
                    row_indices,
                    col_indices,
                    data_values,
                    flat_idx,
                    multi_idx,
                    shape,
                    ndim,
                    dt,
                    sigma_local,
                    coupling_coefficient,
                    spacing,
                    u_flat,
                    grid,
                    boundary_conditions,
                )
            else:  # "divergence_upwind"
                add_interior_entries_divergence_upwind(
                    row_indices,
                    col_indices,
                    data_values,
                    flat_idx,
                    multi_idx,
                    shape,
                    ndim,
                    dt,
                    sigma_local,
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
