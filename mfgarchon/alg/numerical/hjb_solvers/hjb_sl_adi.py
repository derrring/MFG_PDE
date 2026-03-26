"""
ADI Diffusion Methods for Semi-Lagrangian HJB Solver.

This module provides Alternating Direction Implicit (ADI) methods for solving
the diffusion step in the operator-splitting semi-Lagrangian scheme.

The diffusion equation solved is:
    ∂u/∂t = σ²/2 Δu

Methods:
- 1D: Crank-Nicolson (unconditionally stable)
- nD: Peaceman-Rachford ADI splitting with optional full tensor support

For full tensor diffusion (non-diagonal σ_ij):
    ADI handles diagonal terms implicitly, cross terms explicitly:
    u^{n+1} = ADI_solve(u^n + dt * Σ_{i≠j} σ_ij ∂²u/∂x_i∂x_j)

Module structure per issue #392:
    hjb_sl_adi.py - ADI diffusion methods for semi-Lagrangian solver

Functions:
    solve_crank_nicolson_diffusion_1d: 1D Crank-Nicolson diffusion solve
    adi_diffusion_step: nD ADI diffusion solve (Peaceman-Rachford)
    apply_cross_diffusion_explicit: Explicit cross-derivative terms for full tensor
    compute_mixed_derivative: Mixed partial derivative computation
    solve_1d_diffusion_along_axis: Single-axis implicit diffusion solve
    thomas_solve_batched: Vectorized Thomas algorithm for batched tridiagonal systems
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded


def solve_crank_nicolson_diffusion_1d(
    U_star: np.ndarray,
    dt: float,
    sigma: float,
    x_grid: np.ndarray,
) -> np.ndarray:
    """
    Solve 1D diffusion step using Crank-Nicolson (unconditionally stable).

    Solves: (I - θ*dt*σ²/2*L) u^n = (I + θ*dt*σ²/2*L) u*
    where θ = 0.5 for Crank-Nicolson, L is the 1D Laplacian operator.

    Args:
        U_star: Intermediate solution after advection step, shape (Nx,)
        dt: Time step size
        sigma: Diffusion coefficient
        x_grid: 1D spatial grid points

    Returns:
        Solution after implicit diffusion step, shape (Nx,)
    """
    Nx = len(U_star)
    dx = x_grid[1] - x_grid[0]

    # Diffusion coefficient: α = σ²/2 * dt/dx²
    alpha = 0.5 * sigma**2 * dt / dx**2
    theta = 0.5  # Crank-Nicolson parameter

    # Build tridiagonal system: A * u^n = b
    # where A = (I - θ*α*L), b = (I + (1-θ)*α*L) * u*

    # Tridiagonal matrix coefficients
    # Main diagonal: 1 + 2*θ*α
    # Off-diagonals: -θ*α
    main_diag = np.ones(Nx) * (1.0 + 2.0 * theta * alpha)
    off_diag = np.ones(Nx - 1) * (-theta * alpha)

    # Build banded matrix format for solve_banded (upper_diag, main_diag, lower_diag)
    ab = np.zeros((3, Nx))
    ab[0, 1:] = off_diag  # Upper diagonal
    ab[1, :] = main_diag  # Main diagonal
    ab[2, :-1] = off_diag  # Lower diagonal

    # Build right-hand side: b = (I + (1-θ)*α*L) * u*
    b = np.zeros(Nx)
    for i in range(1, Nx - 1):
        b[i] = (1.0 - 2.0 * (1.0 - theta) * alpha) * U_star[i] + (1.0 - theta) * alpha * (U_star[i - 1] + U_star[i + 1])

    # Boundary conditions (Neumann: du/dx = 0)
    # Left boundary (i=0)
    b[0] = (1.0 - (1.0 - theta) * alpha) * U_star[0] + (1.0 - theta) * alpha * U_star[1]
    ab[1, 0] = 1.0 + theta * alpha  # Modified main diagonal
    ab[0, 1] = -theta * alpha  # Upper diagonal

    # Right boundary (i=Nx-1)
    b[Nx - 1] = (1.0 - (1.0 - theta) * alpha) * U_star[Nx - 1] + (1.0 - theta) * alpha * U_star[Nx - 2]
    ab[1, Nx - 1] = 1.0 + theta * alpha  # Modified main diagonal
    ab[2, Nx - 2] = -theta * alpha  # Lower diagonal

    # Solve tridiagonal system
    u_new = solve_banded((1, 1), ab, b)

    return u_new


def adi_diffusion_step(
    U_star: np.ndarray,
    dt: float,
    sigma: float | np.ndarray,
    spacing: np.ndarray,
    grid_shape: tuple[int, ...],
) -> np.ndarray:
    """
    Apply ADI (Alternating Direction Implicit) diffusion for nD grids.

    Uses Peaceman-Rachford splitting for unconditional stability:
    - For each dimension d, solve implicit diffusion in that direction
    - Each direction is a set of independent 1D tridiagonal solves
    - For full tensor diffusion, cross-derivative terms are added explicitly

    For 2D with isotropic σ²:
        Half-step x: (I - θα L_x) u^{1/2} = (I + θα L_y) u*
        Half-step y: (I - θα L_y) u^{n}   = (I + θα L_x) u^{1/2}

    For full tensor diffusion (σ_ij):
        ADI handles diagonal terms implicitly, cross terms explicitly:
        u^{n+1} = ADI_solve(u^n + dt * Σ_{i≠j} σ_ij ∂²u/∂x_i∂x_j)

    Args:
        U_star: Intermediate solution after advection step, shape (N1, N2, ..., Nd)
        dt: Time step size
        sigma: Diffusion coefficient - scalar, 1D array (diagonal), or 2D array (full tensor)
        spacing: Grid spacing in each dimension, shape (d,)
        grid_shape: Shape of the grid (N1, N2, ..., Nd)

    Returns:
        Solution after ADI diffusion step, same shape as U_star
    """
    dimension = len(grid_shape)

    # Parse sigma into tensor form
    sigma_tensor = None  # Full tensor if non-diagonal

    # Backend compatibility - NumPy array detection (Issue #543 acceptable)
    # hasattr used to detect array vs scalar sigma parameter
    if isinstance(sigma, (int, float)):
        # Isotropic: same sigma in all directions
        sigma_vec = np.full(dimension, float(sigma))
    elif hasattr(sigma, "ndim") and sigma.ndim == 1 and len(sigma) == dimension:  # Issue #543 acceptable
        # Diagonal anisotropic: different sigma per direction
        sigma_vec = np.asarray(sigma)
    elif hasattr(sigma, "ndim") and sigma.ndim == 2:  # Issue #543 acceptable
        # Full tensor diffusion
        sigma_tensor = np.asarray(sigma)
        sigma_vec = np.sqrt(np.diag(sigma_tensor))  # Diagonal part for ADI
    else:
        # Fallback to scalar
        sigma_vec = np.full(dimension, 0.1)

    # Ensure U_star is shaped correctly
    if U_star.ndim == 1:
        U_shaped = U_star.reshape(grid_shape)
    else:
        U_shaped = U_star.copy()

    # Handle cross-derivative terms explicitly (for full tensor)
    if sigma_tensor is not None:
        U_shaped = apply_cross_diffusion_explicit(U_shaped, sigma_tensor, dt, spacing)

    # Peaceman-Rachford ADI: cycle through dimensions
    # Split dt evenly across dimensions for symmetric treatment
    dt_per_dim = dt / dimension

    U_current = U_shaped

    for d in range(dimension):
        # Solve implicit diffusion in dimension d
        # α_d = (σ_d²/2) * dt / dx_d²
        dx_d = spacing[d]
        alpha_d = 0.5 * sigma_vec[d] ** 2 * dt_per_dim / dx_d**2
        theta = 0.5  # Crank-Nicolson parameter

        # Apply implicit solve along dimension d
        U_current = solve_1d_diffusion_along_axis(U_current, d, alpha_d, theta)

    # Return in original shape
    if U_star.ndim == 1:
        return U_current.ravel()
    else:
        return U_current


def apply_cross_diffusion_explicit(
    U: np.ndarray,
    sigma_tensor: np.ndarray,
    dt: float,
    spacing: np.ndarray,
) -> np.ndarray:
    """
    Apply cross-derivative diffusion terms explicitly.

    For full tensor diffusion with off-diagonal terms σ_ij (i != j),
    we need to add: dt * Σ_{i<j} 2*σ_ij * ∂²u/∂x_i∂x_j

    Uses central differences for mixed partial derivatives:
        ∂²u/∂x_i∂x_j ≈ (u_{i+1,j+1} - u_{i+1,j-1} - u_{i-1,j+1} + u_{i-1,j-1}) / (4*dx_i*dx_j)

    Args:
        U: Solution array, shape (N1, N2, ..., Nd)
        sigma_tensor: Full diffusion tensor, shape (d, d)
        dt: Time step size
        spacing: Grid spacing in each dimension

    Returns:
        Updated solution with cross-diffusion terms added
    """
    U_new = U.copy()
    d = U.ndim

    for i in range(d):
        for j in range(i + 1, d):
            # Off-diagonal coefficient
            sigma_ij = sigma_tensor[i, j]
            if abs(sigma_ij) < 1e-14:
                continue

            dx_i = spacing[i]
            dx_j = spacing[j]

            # Compute mixed partial derivative using central differences
            # Need to handle boundary carefully
            mixed_deriv = compute_mixed_derivative(U, i, j, dx_i, dx_j)

            # Add explicit contribution: dt * σ_ij * ∂²u/∂x_i∂x_j
            # Factor of 2 because we only loop over i < j but tensor has both (i,j) and (j,i)
            U_new += dt * 2.0 * sigma_ij * mixed_deriv

    return U_new


def compute_mixed_derivative(
    U: np.ndarray,
    axis_i: int,
    axis_j: int,
    dx_i: float,
    dx_j: float,
) -> np.ndarray:
    """
    Compute mixed partial derivative ∂²u/∂x_i∂x_j using vectorized central differences.

    Uses the formula:
        ∂²u/∂x_i∂x_j ≈ (u_{+i,+j} - u_{+i,-j} - u_{-i,+j} + u_{-i,-j}) / (4*dx_i*dx_j)

    Boundaries are set to zero (Neumann-compatible).

    Args:
        U: Solution array
        axis_i, axis_j: Axes for the mixed derivative
        dx_i, dx_j: Grid spacings

    Returns:
        Mixed derivative array, same shape as U
    """
    ndim = U.ndim
    result = np.zeros_like(U)

    # Build slicers for the four corner points
    # Interior region: 1:-1 in both axis_i and axis_j directions
    interior_slice = [slice(None)] * ndim
    interior_slice[axis_i] = slice(1, -1)
    interior_slice[axis_j] = slice(1, -1)

    # u_{+i, +j}: shift +1 in axis_i, +1 in axis_j
    slice_pp = [slice(None)] * ndim
    slice_pp[axis_i] = slice(2, None)
    slice_pp[axis_j] = slice(2, None)

    # u_{+i, -j}: shift +1 in axis_i, -1 in axis_j
    slice_pm = [slice(None)] * ndim
    slice_pm[axis_i] = slice(2, None)
    slice_pm[axis_j] = slice(None, -2)

    # u_{-i, +j}: shift -1 in axis_i, +1 in axis_j
    slice_mp = [slice(None)] * ndim
    slice_mp[axis_i] = slice(None, -2)
    slice_mp[axis_j] = slice(2, None)

    # u_{-i, -j}: shift -1 in axis_i, -1 in axis_j
    slice_mm = [slice(None)] * ndim
    slice_mm[axis_i] = slice(None, -2)
    slice_mm[axis_j] = slice(None, -2)

    # Compute mixed derivative for interior points (vectorized)
    result[tuple(interior_slice)] = (
        U[tuple(slice_pp)] - U[tuple(slice_pm)] - U[tuple(slice_mp)] + U[tuple(slice_mm)]
    ) / (4.0 * dx_i * dx_j)

    return result


def solve_1d_diffusion_along_axis(
    U: np.ndarray,
    axis: int,
    alpha: float,
    theta: float = 0.5,
) -> np.ndarray:
    """
    Solve 1D implicit diffusion along a specific axis using vectorized Thomas algorithm.

    For each line along the axis, solve:
        (I - θ*α*L) u_new = (I + (1-θ)*α*L) u_old

    where L is the 1D Laplacian operator.

    This implementation is fully vectorized - all lines are solved simultaneously
    using batched tridiagonal solves, avoiding Python loops.

    Args:
        U: Solution array, shape (N1, N2, ..., Nd)
        axis: Dimension along which to apply diffusion (0, 1, ..., d-1)
        alpha: Diffusion parameter α = (σ²/2) * dt / dx²
        theta: Implicitness parameter (0.5 = Crank-Nicolson)

    Returns:
        Updated solution array (in-place modification avoided)
    """
    N_axis = U.shape[axis]

    # Move target axis to last position for easier vectorization
    U_transposed = np.moveaxis(U, axis, -1)
    original_shape = U_transposed.shape
    n_lines = int(np.prod(original_shape[:-1]))

    # Reshape to (n_lines, N_axis) for batched solve
    U_2d = U_transposed.reshape(n_lines, N_axis)

    # Tridiagonal coefficients (same for all lines)
    main_coef = 1.0 + 2.0 * theta * alpha
    off_coef = -theta * alpha

    # RHS coefficients
    main_rhs = 1.0 - 2.0 * (1.0 - theta) * alpha
    off_rhs = (1.0 - theta) * alpha

    # Build RHS vectorized: b = (I + (1-θ)*α*L) * u
    # Shape: (n_lines, N_axis)
    b = np.zeros_like(U_2d)

    # Interior points (vectorized over all lines)
    b[:, 1:-1] = main_rhs * U_2d[:, 1:-1] + off_rhs * (U_2d[:, :-2] + U_2d[:, 2:])

    # Boundary conditions (Neumann: du/dx = 0)
    b[:, 0] = (1.0 - (1.0 - theta) * alpha) * U_2d[:, 0] + off_rhs * U_2d[:, 1]
    b[:, -1] = (1.0 - (1.0 - theta) * alpha) * U_2d[:, -1] + off_rhs * U_2d[:, -2]

    # Solve tridiagonal systems using vectorized Thomas algorithm
    U_new_2d = thomas_solve_batched(main_coef, off_coef, b, theta, alpha, N_axis)

    # Reshape back and move axis to original position
    U_new_transposed = U_new_2d.reshape(original_shape)
    return np.moveaxis(U_new_transposed, -1, axis)


def thomas_solve_batched(
    main_coef: float,
    off_coef: float,
    b: np.ndarray,
    theta: float,
    alpha: float,
    N: int,
) -> np.ndarray:
    """
    Vectorized Thomas algorithm for batched tridiagonal solves.

    Solves multiple independent tridiagonal systems simultaneously:
        A @ x = b for each row of b

    where A is the same tridiagonal matrix for all systems.

    Args:
        main_coef: Main diagonal coefficient (interior points)
        off_coef: Off-diagonal coefficient
        b: Right-hand side, shape (n_lines, N)
        theta: Implicitness parameter
        alpha: Diffusion parameter
        N: Size of each system

    Returns:
        Solution x, shape (n_lines, N)
    """
    n_lines = b.shape[0]

    # Build full diagonal arrays with boundary modifications
    main_diag = np.full(N, main_coef)
    main_diag[0] = 1.0 + theta * alpha  # Neumann BC
    main_diag[-1] = 1.0 + theta * alpha  # Neumann BC

    upper_diag = np.full(N - 1, off_coef)
    lower_diag = np.full(N - 1, off_coef)

    # Forward elimination (vectorized over all lines)
    # c' and d' arrays
    c_prime = np.zeros(N - 1)
    d_prime = np.zeros((n_lines, N))

    # First element
    c_prime[0] = upper_diag[0] / main_diag[0]
    d_prime[:, 0] = b[:, 0] / main_diag[0]

    # Forward sweep
    for i in range(1, N - 1):
        denom = main_diag[i] - lower_diag[i - 1] * c_prime[i - 1]
        c_prime[i] = upper_diag[i] / denom
        d_prime[:, i] = (b[:, i] - lower_diag[i - 1] * d_prime[:, i - 1]) / denom

    # Last element
    denom = main_diag[N - 1] - lower_diag[N - 2] * c_prime[N - 2]
    d_prime[:, N - 1] = (b[:, N - 1] - lower_diag[N - 2] * d_prime[:, N - 2]) / denom

    # Back substitution (vectorized)
    x = np.zeros_like(b)
    x[:, N - 1] = d_prime[:, N - 1]

    for i in range(N - 2, -1, -1):
        x[:, i] = d_prime[:, i] - c_prime[i] * x[:, i + 1]

    return x


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    """Smoke test for ADI diffusion methods."""
    print("Testing ADI diffusion methods...")

    # Test 1: 1D Crank-Nicolson
    print("\n1. Testing 1D Crank-Nicolson diffusion...")
    x_grid = np.linspace(0, 1, 51)
    U_test = np.exp(-50 * (x_grid - 0.5) ** 2)

    U_diffused = solve_crank_nicolson_diffusion_1d(U_test, dt=0.01, sigma=0.1, x_grid=x_grid)

    assert U_diffused.shape == U_test.shape
    assert not np.any(np.isnan(U_diffused))
    assert U_diffused.max() < U_test.max()  # Diffusion smooths peak
    print(f"   Peak: {U_test.max():.4f} -> {U_diffused.max():.4f}")
    print("   1D Crank-Nicolson: OK")

    # Test 2: 2D ADI diffusion (isotropic)
    print("\n2. Testing 2D ADI diffusion (isotropic)...")
    grid_shape = (21, 21)
    spacing = np.array([0.05, 0.05])
    x = np.linspace(0, 1, grid_shape[0])
    y = np.linspace(0, 1, grid_shape[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    U_2d = np.exp(-50 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))

    U_2d_diffused = adi_diffusion_step(U_2d, dt=0.01, sigma=0.1, spacing=spacing, grid_shape=grid_shape)

    assert U_2d_diffused.shape == U_2d.shape
    assert not np.any(np.isnan(U_2d_diffused))
    assert U_2d_diffused.max() < U_2d.max()
    print(f"   Peak: {U_2d.max():.4f} -> {U_2d_diffused.max():.4f}")
    print("   2D ADI (isotropic): OK")

    # Test 3: 2D ADI with anisotropic sigma
    print("\n3. Testing 2D ADI (anisotropic)...")
    sigma_aniso = np.array([0.15, 0.05])
    U_2d_aniso = adi_diffusion_step(U_2d.copy(), dt=0.01, sigma=sigma_aniso, spacing=spacing, grid_shape=grid_shape)

    assert U_2d_aniso.shape == U_2d.shape
    assert not np.any(np.isnan(U_2d_aniso))
    print(f"   Peak: {U_2d.max():.4f} -> {U_2d_aniso.max():.4f}")
    print("   2D ADI (anisotropic): OK")

    # Test 4: Full tensor diffusion
    print("\n4. Testing 2D ADI (full tensor)...")
    sigma_tensor = np.array([[0.01, 0.005], [0.005, 0.01]])
    U_2d_tensor = adi_diffusion_step(U_2d.copy(), dt=0.01, sigma=sigma_tensor, spacing=spacing, grid_shape=grid_shape)

    assert U_2d_tensor.shape == U_2d.shape
    assert not np.any(np.isnan(U_2d_tensor))
    print(f"   Peak: {U_2d.max():.4f} -> {U_2d_tensor.max():.4f}")
    print("   2D ADI (full tensor): OK")

    # Test 5: 3D ADI diffusion
    print("\n5. Testing 3D ADI diffusion...")
    grid_shape_3d = (11, 11, 11)
    spacing_3d = np.array([0.1, 0.1, 0.1])
    x3 = np.linspace(0, 1, grid_shape_3d[0])
    y3 = np.linspace(0, 1, grid_shape_3d[1])
    z3 = np.linspace(0, 1, grid_shape_3d[2])
    X3, Y3, Z3 = np.meshgrid(x3, y3, z3, indexing="ij")
    U_3d = np.exp(-30 * ((X3 - 0.5) ** 2 + (Y3 - 0.5) ** 2 + (Z3 - 0.5) ** 2))

    U_3d_diffused = adi_diffusion_step(U_3d, dt=0.01, sigma=0.1, spacing=spacing_3d, grid_shape=grid_shape_3d)

    assert U_3d_diffused.shape == U_3d.shape
    assert not np.any(np.isnan(U_3d_diffused))
    assert U_3d_diffused.max() < U_3d.max()
    print(f"   Peak: {U_3d.max():.4f} -> {U_3d_diffused.max():.4f}")
    print("   3D ADI: OK")

    # Test 6: Mass conservation
    print("\n6. Testing mass conservation...")
    mass_before = np.sum(U_2d) * spacing[0] * spacing[1]
    mass_after = np.sum(U_2d_diffused) * spacing[0] * spacing[1]
    mass_error = abs(mass_after - mass_before) / mass_before
    print(f"   Mass: {mass_before:.6f} -> {mass_after:.6f}")
    print(f"   Relative error: {mass_error:.2e}")
    assert mass_error < 0.05
    print("   Mass conservation: OK")

    print("\nAll ADI diffusion smoke tests passed!")
