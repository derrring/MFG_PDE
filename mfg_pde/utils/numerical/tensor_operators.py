"""
Tensor Diffusion Operators for Anisotropic PDEs.

.. deprecated:: 0.18.0
    This module is deprecated. Use ``tensor_calculus`` instead:

    Migration Guide::

        # Old
        from mfg_pde.utils.numerical.tensor_operators import (
            divergence_tensor_diffusion_2d,
            divergence_tensor_diffusion_nd,
        )

        # New
        from mfg_pde.utils.numerical.tensor_calculus import tensor_diffusion

        # tensor_diffusion() auto-dispatches to 1D/2D/nD implementations

    The new ``tensor_calculus`` module provides:
    - Unified API for all differential operators
    - Consistent BC handling across operators
    - Complete tensor calculus: gradient, divergence, laplacian, hessian, advection

This module is kept for backward compatibility and will be removed in v1.0.

Original Features (still available via tensor_calculus.tensor_diffusion):
- Supports constant and spatially-varying tensors
- Handles periodic, Dirichlet, and no-flux boundary conditions
- Optimized diagonal case for efficiency
- Full anisotropic and cross-diffusion support
- Optional Numba JIT compilation for 10-50x speedup
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

# Try to import Numba
try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Dummy decorator if Numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# Control JIT compilation via environment variable
USE_NUMBA = os.environ.get("MFG_USE_NUMBA", "auto")
if USE_NUMBA == "auto":
    USE_NUMBA = NUMBA_AVAILABLE
elif USE_NUMBA.lower() in ("true", "1", "yes"):
    USE_NUMBA = True
else:
    USE_NUMBA = False

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.geometry import BoundaryConditions
    from mfg_pde.geometry.boundary import MixedBoundaryConditions


# ============================================================================
# Numba JIT-compiled kernels for performance
# ============================================================================


@njit(cache=True)
def _compute_full_tensor_kernel(
    m_padded: np.ndarray,
    Sigma: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    JIT-compiled kernel for full tensor diffusion computation.

    This is the performance-critical inner loop that benefits most from JIT.
    Implements the staggered grid finite difference scheme.

    Args:
        m_padded: Density with ghost cells (Ny+2, Nx+2)
        Sigma: Spatially-varying tensor (Ny, Nx, 2, 2)
        dx, dy: Grid spacing

    Returns:
        Diffusion term (Ny, Nx)
    """
    Ny, Nx = Sigma.shape[0], Sigma.shape[1]

    # Allocate output
    result = np.zeros((Ny, Nx))

    # Compute gradients and fluxes manually (Numba-compatible)
    for i in range(Ny):
        for j in range(Nx):
            # Extract tensor at cell center
            s11 = Sigma[i, j, 0, 0]
            s12 = Sigma[i, j, 0, 1]
            s21 = Sigma[i, j, 1, 0]
            s22 = Sigma[i, j, 1, 1]

            # Neighboring tensors for face averaging
            # x-faces at j+1/2 and j-1/2
            if j < Nx - 1:
                s11_xp = 0.5 * (s11 + Sigma[i, j + 1, 0, 0])
                s12_xp = 0.5 * (s12 + Sigma[i, j + 1, 0, 1])
            else:
                s11_xp = s11
                s12_xp = s12

            if j > 0:
                s11_xm = 0.5 * (s11 + Sigma[i, j - 1, 0, 0])
                s12_xm = 0.5 * (s12 + Sigma[i, j - 1, 0, 1])
            else:
                s11_xm = s11
                s12_xm = s12

            # y-faces at i+1/2 and i-1/2
            if i < Ny - 1:
                s21_yp = 0.5 * (s21 + Sigma[i + 1, j, 1, 0])
                s22_yp = 0.5 * (s22 + Sigma[i + 1, j, 1, 1])
            else:
                s21_yp = s21
                s22_yp = s22

            if i > 0:
                s21_ym = 0.5 * (s21 + Sigma[i - 1, j, 1, 0])
                s22_ym = 0.5 * (s22 + Sigma[i - 1, j, 1, 1])
            else:
                s21_ym = s21
                s22_ym = s22

            # Padded indices (shift by 1 due to ghost cells)
            ip = i + 1
            jp = j + 1

            # Gradients at faces
            # x-faces (j+1/2): dm_dx and dm_dy
            dm_dx_xp = (m_padded[ip, jp + 1] - m_padded[ip, jp]) / dx
            dm_dy_xp = (
                0.25
                * (
                    (m_padded[ip + 1, jp + 1] - m_padded[ip - 1, jp + 1])
                    + (m_padded[ip + 1, jp] - m_padded[ip - 1, jp])
                )
                / dy
            )

            dm_dx_xm = (m_padded[ip, jp] - m_padded[ip, jp - 1]) / dx
            dm_dy_xm = (
                0.25
                * (
                    (m_padded[ip + 1, jp] - m_padded[ip - 1, jp])
                    + (m_padded[ip + 1, jp - 1] - m_padded[ip - 1, jp - 1])
                )
                / dy
            )

            # y-faces (i+1/2): dm_dx and dm_dy
            dm_dy_yp = (m_padded[ip + 1, jp] - m_padded[ip, jp]) / dy
            dm_dx_yp = (
                0.25
                * (
                    (m_padded[ip + 1, jp + 1] - m_padded[ip + 1, jp - 1])
                    + (m_padded[ip, jp + 1] - m_padded[ip, jp - 1])
                )
                / dx
            )

            dm_dy_ym = (m_padded[ip, jp] - m_padded[ip - 1, jp]) / dy
            dm_dx_ym = (
                0.25
                * (
                    (m_padded[ip, jp + 1] - m_padded[ip, jp - 1])
                    + (m_padded[ip - 1, jp + 1] - m_padded[ip - 1, jp - 1])
                )
                / dx
            )

            # Fluxes at faces
            Fx_xp = s11_xp * dm_dx_xp + s12_xp * dm_dy_xp
            Fx_xm = s11_xm * dm_dx_xm + s12_xm * dm_dy_xm

            Fy_yp = s21_yp * dm_dx_yp + s22_yp * dm_dy_yp
            Fy_ym = s21_ym * dm_dx_ym + s22_ym * dm_dy_ym

            # Divergence
            div_x = (Fx_xp - Fx_xm) / dx
            div_y = (Fy_yp - Fy_ym) / dy

            result[i, j] = div_x + div_y

    return result


@njit(cache=True)
def _compute_diagonal_kernel(
    m_padded: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    JIT-compiled kernel for diagonal tensor diffusion.

    Optimized for diagonal tensors Σ = diag([σₓ², σᵧ²]).
    Simpler than full tensor case (no cross-terms).

    Args:
        m_padded: Density with ghost cells (Ny+2, Nx+2)
        sigma_x: x-diffusion coefficient (Ny, Nx)
        sigma_y: y-diffusion coefficient (Ny, Nx)
        dx, dy: Grid spacing

    Returns:
        Diffusion term (Ny, Nx)
    """
    Ny, Nx = sigma_x.shape

    result = np.zeros((Ny, Nx))

    for i in range(Ny):
        for j in range(Nx):
            # Padded indices
            ip = i + 1
            jp = j + 1

            # x-direction: ∂/∂x(σₓ² ∂m/∂x)
            # Average σₓ² to x-faces
            if j < Nx - 1:
                sigma_x_xp = 0.5 * (sigma_x[i, j] + sigma_x[i, j + 1])
            else:
                sigma_x_xp = sigma_x[i, j]

            if j > 0:
                sigma_x_xm = 0.5 * (sigma_x[i, j] + sigma_x[i, j - 1])
            else:
                sigma_x_xm = sigma_x[i, j]

            # Gradients at x-faces
            dm_dx_xp = (m_padded[ip, jp + 1] - m_padded[ip, jp]) / dx
            dm_dx_xm = (m_padded[ip, jp] - m_padded[ip, jp - 1]) / dx

            # Fluxes
            Fx_xp = sigma_x_xp * dm_dx_xp
            Fx_xm = sigma_x_xm * dm_dx_xm

            # Divergence in x
            div_x = (Fx_xp - Fx_xm) / dx

            # y-direction: ∂/∂y(σᵧ² ∂m/∂y)
            # Average σᵧ² to y-faces
            if i < Ny - 1:
                sigma_y_yp = 0.5 * (sigma_y[i, j] + sigma_y[i + 1, j])
            else:
                sigma_y_yp = sigma_y[i, j]

            if i > 0:
                sigma_y_ym = 0.5 * (sigma_y[i, j] + sigma_y[i - 1, j])
            else:
                sigma_y_ym = sigma_y[i, j]

            # Gradients at y-faces
            dm_dy_yp = (m_padded[ip + 1, jp] - m_padded[ip, jp]) / dy
            dm_dy_ym = (m_padded[ip, jp] - m_padded[ip - 1, jp]) / dy

            # Fluxes
            Fy_yp = sigma_y_yp * dm_dy_yp
            Fy_ym = sigma_y_ym * dm_dy_ym

            # Divergence in y
            div_y = (Fy_yp - Fy_ym) / dy

            result[i, j] = div_x + div_y

    return result


# ============================================================================
# Public API (with JIT dispatch)
# ============================================================================


def divergence_tensor_diffusion_2d(
    m: NDArray[np.floating],
    sigma_tensor: NDArray[np.floating],
    dx: float,
    dy: float,
    boundary_conditions: BoundaryConditions | MixedBoundaryConditions,
    domain_bounds: NDArray[np.floating] | None = None,
    time: float = 0.0,
) -> NDArray[np.floating]:
    """
    Compute ∇ · (Σ ∇m) in 2D with tensor diffusion.

    Discretization:
        ∇ · (Σ ∇m) = ∂/∂x(σ₁₁ ∂m/∂x + σ₁₂ ∂m/∂y)
                    + ∂/∂y(σ₂₁ ∂m/∂x + σ₂₂ ∂m/∂y)

    Uses centered finite differences with ghost cells for boundary conditions.

    Args:
        m: Density field (Ny, Nx)
        sigma_tensor: Diffusion tensor Σ
            - Constant: (2, 2) array
            - Spatially varying: (Ny, Nx, 2, 2) array
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction
        boundary_conditions: Boundary condition specification (uniform or mixed)
        domain_bounds: Domain bounds array (2, 2) for mixed BCs
        time: Current time for time-dependent BCs

    Returns:
        Diffusion term ∇ · (Σ ∇m) with same shape as m

    Mathematical Note:
        For symmetric Σ: σ₁₂ = σ₂₁, so cross-diffusion terms appear twice.
        The discretization preserves the self-adjoint structure of the operator.
    """
    Ny, Nx = m.shape

    # Expand sigma_tensor to spatially-varying if constant
    if sigma_tensor.ndim == 2:
        # Constant tensor: broadcast to (Ny, Nx, 2, 2)
        Sigma = np.tile(sigma_tensor, (Ny, Nx, 1, 1))
    else:
        # Already spatially varying
        Sigma = sigma_tensor

    # Apply boundary conditions (add ghost cells)
    m_padded = _apply_bc_2d(m, boundary_conditions, domain_bounds, time)

    # Use JIT kernel if available and enabled
    if USE_NUMBA and NUMBA_AVAILABLE:
        return _compute_full_tensor_kernel(m_padded, Sigma, dx, dy)

    # Fallback: Pure NumPy implementation
    # Compute gradients using central differences
    # Note: gradients are at cell faces (staggered grid)
    # m_padded shape: (Ny+2, Nx+2)
    # For Nx cells, we have Nx+1 x-faces (including boundaries)

    # ∂m/∂x at x-faces (i, j+1/2) - Nx+1 faces in x-direction
    dm_dx_x = (m_padded[1:-1, 1:] - m_padded[1:-1, :-1]) / dx  # Shape: (Ny, Nx+1)

    # ∂m/∂y at y-faces (i+1/2, j) - Ny+1 faces in y-direction
    dm_dy_y = (m_padded[1:, 1:-1] - m_padded[:-1, 1:-1]) / dy  # Shape: (Ny+1, Nx)

    # ∂m/∂x at y-faces (i+1/2, j) - need to average in y
    # Average of central differences at (i+1/2, j)
    dm_dx_y = (
        0.5 * ((m_padded[1:, 2:] - m_padded[1:, :-2]) + (m_padded[:-1, 2:] - m_padded[:-1, :-2])) / (2 * dx)
    )  # Shape: (Ny+1, Nx)

    # ∂m/∂y at x-faces (i, j+1/2) - need to average in x
    # Average of central differences at (i, j+1/2)
    dm_dy_x = (
        0.5 * ((m_padded[2:, 1:] - m_padded[:-2, 1:]) + (m_padded[2:, :-1] - m_padded[:-2, :-1])) / (2 * dy)
    )  # Shape: (Ny, Nx+1)

    # Compute fluxes F = Σ ∇m at cell faces
    # Fx = (σ₁₁ ∂m/∂x + σ₁₂ ∂m/∂y) at x-faces
    # Fy = (σ₂₁ ∂m/∂x + σ₂₂ ∂m/∂y) at y-faces

    # Average Σ to faces
    # Σ at x-faces (i, j+1/2): average in x-direction (Nx+1 faces)
    Sigma_x_faces = np.zeros((Ny, Nx + 1, 2, 2))
    Sigma_x_faces[:, 1:-1, :, :] = 0.5 * (Sigma[:, 1:, :, :] + Sigma[:, :-1, :, :])
    Sigma_x_faces[:, 0, :, :] = Sigma[:, 0, :, :]  # Boundary face
    Sigma_x_faces[:, -1, :, :] = Sigma[:, -1, :, :]  # Boundary face

    # Σ at y-faces (i+1/2, j): average in y-direction (Ny+1 faces)
    Sigma_y_faces = np.zeros((Ny + 1, Nx, 2, 2))
    Sigma_y_faces[1:-1, :, :, :] = 0.5 * (Sigma[1:, :, :, :] + Sigma[:-1, :, :, :])
    Sigma_y_faces[0, :, :, :] = Sigma[0, :, :, :]  # Boundary face
    Sigma_y_faces[-1, :, :, :] = Sigma[-1, :, :, :]  # Boundary face

    # Compute fluxes
    Fx = Sigma_x_faces[:, :, 0, 0] * dm_dx_x + Sigma_x_faces[:, :, 0, 1] * dm_dy_x  # Shape: (Ny, Nx+1)
    Fy = Sigma_y_faces[:, :, 1, 0] * dm_dx_y + Sigma_y_faces[:, :, 1, 1] * dm_dy_y  # Shape: (Ny+1, Nx)

    # Compute divergence: ∇ · F = ∂Fx/∂x + ∂Fy/∂y
    div_x = (Fx[:, 1:] - Fx[:, :-1]) / dx  # Shape: (Ny, Nx)
    div_y = (Fy[1:, :] - Fy[:-1, :]) / dy  # Shape: (Ny, Nx)

    return div_x + div_y


def divergence_diagonal_diffusion_2d(
    m: NDArray[np.floating],
    sigma_diag: NDArray[np.floating],
    dx: float,
    dy: float,
    boundary_conditions: BoundaryConditions | MixedBoundaryConditions,
    domain_bounds: NDArray[np.floating] | None = None,
    time: float = 0.0,
) -> NDArray[np.floating]:
    """
    Optimized divergence for diagonal diffusion tensor.

    For Σ = diag([σₓ², σᵧ²]), the operator simplifies to:
        ∇ · (Σ ∇m) = ∂/∂x(σₓ² ∂m/∂x) + ∂/∂y(σᵧ² ∂m/∂y)

    This is faster than the full tensor case (no cross-terms).

    Args:
        m: Density field (Ny, Nx)
        sigma_diag: Diagonal diffusion coefficients
            - Constant: (2,) array [σₓ², σᵧ²]
            - Spatially varying: (Ny, Nx, 2) array
        dx, dy: Grid spacing
        boundary_conditions: BC specification (uniform or mixed)
        domain_bounds: Domain bounds array (2, 2) for mixed BCs
        time: Current time for time-dependent BCs

    Returns:
        Diffusion term with shape matching m
    """
    Ny, Nx = m.shape

    # Expand to spatially-varying if constant
    if sigma_diag.ndim == 1:
        sigma_x = np.full((Ny, Nx), sigma_diag[0])
        sigma_y = np.full((Ny, Nx), sigma_diag[1])
    else:
        sigma_x = sigma_diag[:, :, 0]
        sigma_y = sigma_diag[:, :, 1]

    # Apply boundary conditions
    m_padded = _apply_bc_2d(m, boundary_conditions, domain_bounds, time)

    # Use JIT kernel if available and enabled
    if USE_NUMBA and NUMBA_AVAILABLE:
        return _compute_diagonal_kernel(m_padded, sigma_x, sigma_y, dx, dy)

    # Fallback: Pure NumPy implementation
    # Compute ∂/∂x(σₓ² ∂m/∂x)
    # Gradient at x-faces (Nx+1 faces)
    dm_dx = (m_padded[1:-1, 1:] - m_padded[1:-1, :-1]) / dx  # Shape: (Ny, Nx+1)

    # Average σₓ² to x-faces
    sigma_x_face = np.zeros((Ny, Nx + 1))
    sigma_x_face[:, 1:-1] = 0.5 * (sigma_x[:, 1:] + sigma_x[:, :-1])
    sigma_x_face[:, 0] = sigma_x[:, 0]
    sigma_x_face[:, -1] = sigma_x[:, -1]

    # Flux at x-faces
    Fx = sigma_x_face * dm_dx  # Shape: (Ny, Nx+1)

    # Divergence in x
    div_x = (Fx[:, 1:] - Fx[:, :-1]) / dx  # Shape: (Ny, Nx)

    # Compute ∂/∂y(σᵧ² ∂m/∂y) similarly
    # Gradient at y-faces (Ny+1 faces)
    dm_dy = (m_padded[1:, 1:-1] - m_padded[:-1, 1:-1]) / dy  # Shape: (Ny+1, Nx)

    # Average σᵧ² to y-faces
    sigma_y_face = np.zeros((Ny + 1, Nx))
    sigma_y_face[1:-1, :] = 0.5 * (sigma_y[1:, :] + sigma_y[:-1, :])
    sigma_y_face[0, :] = sigma_y[0, :]
    sigma_y_face[-1, :] = sigma_y[-1, :]

    # Flux at y-faces
    Fy = sigma_y_face * dm_dy  # Shape: (Ny+1, Nx)

    # Divergence in y
    div_y = (Fy[1:, :] - Fy[:-1, :]) / dy  # Shape: (Ny, Nx)

    return div_x + div_y


def _apply_bc_2d(
    m: NDArray[np.floating],
    boundary_conditions: BoundaryConditions | MixedBoundaryConditions,
    domain_bounds: NDArray[np.floating] | None = None,
    time: float = 0.0,
) -> NDArray[np.floating]:
    """
    Apply boundary conditions by padding array with ghost cells.

    Supports both uniform BCs (BoundaryConditions) and mixed BCs
    (MixedBoundaryConditions) with different BC types on different
    boundary segments.

    Args:
        m: Interior field (Ny, Nx)
        boundary_conditions: BC specification (uniform or mixed)
        domain_bounds: Domain bounds array (2, 2) for mixed BCs
                      Can be omitted if provided in MixedBoundaryConditions
        time: Current time for time-dependent BCs

    Returns:
        Padded field (Ny+2, Nx+2) with ghost cells

    Supported BC types:
        - periodic: Wrap-around boundaries
        - dirichlet: Specified boundary values
        - no_flux/neumann: Zero normal derivative
        - robin: alpha*u + beta*du/dn = g
    """
    # Import here to avoid circular imports
    from mfg_pde.geometry.boundary import MixedBoundaryConditions
    from mfg_pde.geometry.boundary.applicator_fdm import apply_boundary_conditions_2d

    # Check if using new mixed BC system
    if isinstance(boundary_conditions, MixedBoundaryConditions):
        return apply_boundary_conditions_2d(
            field=m,
            boundary_conditions=boundary_conditions,
            domain_bounds=domain_bounds,
            time=time,
        )

    # Legacy uniform BC handling (fast path for simple cases)
    bc_type = boundary_conditions.type.lower()

    if bc_type == "periodic":
        # Periodic: wrap around
        return np.pad(m, 1, mode="wrap")

    elif bc_type == "dirichlet":
        # Use the new BC applicator for correct ghost cell formula
        return apply_boundary_conditions_2d(
            field=m,
            boundary_conditions=boundary_conditions,
            time=time,
        )

    elif bc_type in ["no_flux", "neumann"]:
        # No-flux: zero normal derivative (reflective)
        return np.pad(m, 1, mode="edge")

    else:
        raise ValueError(f"Unsupported boundary condition type: {bc_type}")


# ============================================================================
# Higher-dimensional operators (nD generalization)
# ============================================================================


def divergence_tensor_diffusion_nd(
    m: NDArray[np.floating],
    sigma_tensor: NDArray[np.floating],
    dx: tuple[float, ...],
    boundary_conditions: BoundaryConditions,
) -> NDArray[np.floating]:
    """
    Compute ∇ · (Σ ∇m) in arbitrary dimensions.

    Generalization of 2D formula:
        ∇ · (Σ ∇m) = Σᵢ ∂/∂xᵢ (Σⱼ Σᵢⱼ ∂m/∂xⱼ)

    Args:
        m: Density field with shape (N₁, N₂, ..., Nₐ)
        sigma_tensor: Diffusion tensor
            - Constant: (d, d) array
            - Spatially varying: (N₁, ..., Nₐ, d, d) array
        dx: Grid spacing tuple (dx₁, dx₂, ..., dxₐ)
        boundary_conditions: BC specification

    Returns:
        Diffusion term with same shape as m

    Note:
        For d=1: Falls back to scalar laplacian
        For d=2: Calls optimized 2D implementation
        For d>2: Uses general nD stencil
    """
    d = len(m.shape)  # Spatial dimension

    # Special case: 1D (no tensor structure)
    if d == 1:
        return _divergence_tensor_1d(m, sigma_tensor, dx[0], boundary_conditions)

    # Special case: 2D (optimized implementation)
    if d == 2:
        return divergence_tensor_diffusion_2d(m, sigma_tensor, dx[0], dx[1], boundary_conditions)

    # General nD case
    return _divergence_tensor_general_nd(m, sigma_tensor, dx, boundary_conditions)


def _divergence_tensor_1d(
    m: NDArray[np.floating],
    sigma_tensor: NDArray[np.floating] | float,
    dx: float,
    boundary_conditions: BoundaryConditions,
) -> NDArray[np.floating]:
    """
    1D tensor diffusion (reduces to scalar diffusion).

    Implements ∂/∂x(σ² ∂m/∂x) in 1D using finite differences.
    """
    # In 1D, Σ is just a scalar (1×1 matrix)
    if isinstance(sigma_tensor, np.ndarray):
        if sigma_tensor.ndim == 2:
            # Extract scalar from (1, 1) matrix
            sigma_sq = sigma_tensor[0, 0]
        elif sigma_tensor.ndim == 1:
            # Already a scalar array
            sigma_sq = sigma_tensor
        else:
            # Spatially varying (Nx, 1, 1)
            sigma_sq = sigma_tensor[:, 0, 0]
    else:
        sigma_sq = sigma_tensor

    # Apply boundary conditions
    bc_type = boundary_conditions.type.lower()
    if bc_type == "periodic":
        m_padded = np.pad(m, 1, mode="wrap")
    elif bc_type == "dirichlet":
        m_padded = np.pad(m, 1, mode="constant", constant_values=0.0)
    elif bc_type in ["no_flux", "neumann"]:
        m_padded = np.pad(m, 1, mode="edge")
    else:
        raise ValueError(f"Unsupported boundary condition: {bc_type}")

    Nx = len(m)

    # Handle constant vs spatially-varying diffusion
    if np.isscalar(sigma_sq) or (isinstance(sigma_sq, np.ndarray) and sigma_sq.ndim == 0):
        # Constant diffusion: σ² ∂²m/∂x²
        laplacian = (m_padded[2:] - 2 * m_padded[1:-1] + m_padded[:-2]) / dx**2
        return sigma_sq * laplacian
    else:
        # Spatially-varying: ∂/∂x(σ²(x) ∂m/∂x)
        # Gradient at faces
        dm_dx = (m_padded[1:] - m_padded[:-1]) / dx  # Nx+1 faces

        # Average σ² to faces
        sigma_face = np.zeros(Nx + 1)
        sigma_face[1:-1] = 0.5 * (sigma_sq[1:] + sigma_sq[:-1])
        sigma_face[0] = sigma_sq[0]
        sigma_face[-1] = sigma_sq[-1]

        # Flux
        flux = sigma_face * dm_dx

        # Divergence
        return (flux[1:] - flux[:-1]) / dx


def _divergence_tensor_general_nd(
    m: NDArray[np.floating],
    sigma_tensor: NDArray[np.floating],
    dx: tuple[float, ...],
    boundary_conditions: BoundaryConditions,
) -> NDArray[np.floating]:
    """
    General nD tensor diffusion using component-wise stencils.

    Computes ∇·(Σ∇m) = Σᵢ ∂/∂xᵢ (Σⱼ Σᵢⱼ ∂m/∂xⱼ) in arbitrary dimensions.

    Algorithm:
        1. Pad array with ghost cells for boundary conditions
        2. For each flux direction i:
           a. Compute gradients ∂m/∂xⱼ at i-faces for all j
           b. Compute flux Fᵢ = Σⱼ Σᵢⱼ (∂m/∂xⱼ)
        3. Sum divergences: Σᵢ (∂Fᵢ/∂xᵢ)

    Args:
        m: Density field with shape (N₁, N₂, ..., Nₐ)
        sigma_tensor: Diffusion tensor
            - Constant: (d, d) array
            - Spatially varying: (N₁, ..., Nₐ, d, d) array
        dx: Grid spacing tuple (dx₁, dx₂, ..., dxₐ)
        boundary_conditions: BC specification

    Returns:
        Diffusion term with same shape as m
    """
    d = len(m.shape)
    shape = m.shape

    # Expand sigma_tensor to spatially-varying if constant
    if sigma_tensor.ndim == 2:
        # Constant tensor: broadcast to (*shape, d, d)
        Sigma = np.broadcast_to(sigma_tensor, (*shape, d, d)).copy()
    else:
        Sigma = sigma_tensor

    # Apply boundary conditions (pad with ghost cells)
    bc_type = boundary_conditions.type.lower()
    if bc_type == "periodic":
        m_padded = np.pad(m, 1, mode="wrap")
    elif bc_type == "dirichlet":
        m_padded = np.pad(m, 1, mode="constant", constant_values=0.0)
    elif bc_type in ["no_flux", "neumann"]:
        m_padded = np.pad(m, 1, mode="edge")
    else:
        raise ValueError(f"Unsupported boundary condition: {bc_type}")

    # Initialize result
    result = np.zeros(shape, dtype=m.dtype)

    # For each flux direction i, compute divergence contribution
    for i in range(d):
        # Build slices for computing gradient at i-faces
        # i-faces have shape: (..., N_i + 1, ...) in dimension i

        # Compute flux F_i = Σ_j Σ_ij * (∂m/∂x_j at i-faces)
        flux_shape = list(shape)
        flux_shape[i] += 1
        F_i = np.zeros(flux_shape, dtype=m.dtype)

        for j in range(d):
            # Compute ∂m/∂x_j at i-faces

            if i == j:
                # Direct gradient in direction i at i-faces
                # dm/dx_i = (m[..., k+1, ...] - m[..., k, ...]) / dx_i
                # Using padded array: indices 0 to N_i+1 -> N_i+1 faces
                slice_plus = [slice(1, -1)] * d
                slice_minus = [slice(1, -1)] * d
                slice_plus[i] = slice(1, None)  # indices 1 to N_i+2
                slice_minus[i] = slice(None, -1)  # indices 0 to N_i+1
                dm_dxj = (m_padded[tuple(slice_plus)] - m_padded[tuple(slice_minus)]) / dx[j]
            else:
                # Cross-derivative: ∂m/∂x_j at i-faces
                # Compute gradient in j-direction and average across i-direction to get i-faces
                #
                # For face at position (k+1/2) in direction i, average gradients at cells k and k+1
                # Gradient in j uses central diff: (m[j+1] - m[j-1]) / (2*dx_j)

                # Build slices for cross-derivative at i-faces
                # We need shape: flux_shape (with N_i+1 in dimension i)

                # Gradient at i-face (k+1/2): average of gradient at cell k and k+1
                # Cell k gradient: (m[j+1,k] - m[j-1,k]) / (2*dx_j)
                # Cell k+1 gradient: (m[j+1,k+1] - m[j-1,k+1]) / (2*dx_j)

                # In padded coords (all shifted by 1):
                # Cell k uses i-index k+1, Cell k+1 uses i-index k+2
                # j+1 uses j-index j+2, j-1 uses j-index j

                # For i-faces from 0 to N_i (N_i+1 total), we need padded i-indices 1:N_i+2
                # which is slice(1, -1) but including one more: slice(1, None) for right, slice(None, -1) for left

                # Simpler approach: compute dm/dxj at all interior+ghost points, then interpolate to faces
                slice_j_plus = [slice(1, -1)] * d
                slice_j_minus = [slice(1, -1)] * d
                slice_j_plus[j] = slice(2, None)
                slice_j_minus[j] = slice(None, -2)

                # Extend i-direction to include ghost cells for averaging
                slice_j_plus[i] = slice(None)  # All i-indices including ghosts
                slice_j_minus[i] = slice(None)

                dm_dxj_extended = (m_padded[tuple(slice_j_plus)] - m_padded[tuple(slice_j_minus)]) / (2 * dx[j])
                # Shape: (N_i+2, ...) with N_i+2 in dimension i

                # Average to i-faces
                slice_k = [slice(None)] * d
                slice_k1 = [slice(None)] * d
                slice_k[i] = slice(None, -1)  # indices 0 to N_i+1
                slice_k1[i] = slice(1, None)  # indices 1 to N_i+2
                dm_dxj = 0.5 * (dm_dxj_extended[tuple(slice_k)] + dm_dxj_extended[tuple(slice_k1)])

            # Average Sigma_ij to i-faces
            # Need N_i+1 faces in dimension i
            Sigma_ij = Sigma[..., i, j]  # Shape: (*shape)

            # Create face-valued array
            face_shape = list(shape)
            face_shape[i] += 1
            Sigma_ij_faces = np.zeros(face_shape, dtype=Sigma.dtype)

            # Interior faces (N_i-1 of them): average of neighboring cells
            # Face k (for k=1..N_i-1) is average of cell k-1 and cell k
            slice_interior_dest = [slice(None)] * d
            slice_interior_dest[i] = slice(1, -1)  # N_i-1 interior faces

            slice_cell_left = [slice(None)] * d
            slice_cell_left[i] = slice(None, -1)  # cells 0..N_i-2

            slice_cell_right = [slice(None)] * d
            slice_cell_right[i] = slice(1, None)  # cells 1..N_i-1

            # Average gives N_i-1 values for interior faces
            Sigma_ij_faces[tuple(slice_interior_dest)] = 0.5 * (
                Sigma_ij[tuple(slice_cell_left)] + Sigma_ij[tuple(slice_cell_right)]
            )

            # Boundary faces: use boundary cell values
            slice_first_face = [slice(None)] * d
            slice_first_face[i] = 0
            slice_first_cell = [slice(None)] * d
            slice_first_cell[i] = 0

            slice_last_face = [slice(None)] * d
            slice_last_face[i] = -1
            slice_last_cell = [slice(None)] * d
            slice_last_cell[i] = -1

            Sigma_ij_faces[tuple(slice_first_face)] = Sigma_ij[tuple(slice_first_cell)]
            Sigma_ij_faces[tuple(slice_last_face)] = Sigma_ij[tuple(slice_last_cell)]

            # Add contribution to flux
            F_i += Sigma_ij_faces * dm_dxj

        # Compute divergence of F_i in direction i: (F_i[k+1] - F_i[k]) / dx_i
        slice_plus_i = [slice(None)] * d
        slice_minus_i = [slice(None)] * d
        slice_plus_i[i] = slice(1, None)
        slice_minus_i[i] = slice(None, -1)

        div_F_i = (F_i[tuple(slice_plus_i)] - F_i[tuple(slice_minus_i)]) / dx[i]

        result += div_F_i

    return result


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing tensor diffusion operators...")

    import numpy as np

    from mfg_pde.geometry.boundary.conditions import periodic_bc

    # Test 2D diagonal diffusion (isotropic case)
    Nx, Ny = 20, 15
    dx, dy = 0.1, 0.1

    # Create a Gaussian density
    x = np.linspace(0, (Nx - 1) * dx, Nx)
    y = np.linspace(0, (Ny - 1) * dy, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    m = np.exp(-((X - 1.0) ** 2 + (Y - 0.75) ** 2) / 0.1)

    # Diagonal tensor (isotropic diffusion)
    sigma_diag = np.array([0.1, 0.1])

    # Periodic boundary conditions
    bc = periodic_bc(dimension=2)

    # Compute divergence
    div_m = divergence_diagonal_diffusion_2d(m, sigma_diag, dx, dy, bc)

    assert div_m.shape == m.shape, f"Shape mismatch: {div_m.shape} vs {m.shape}"
    assert not np.any(np.isnan(div_m)), "NaN values in divergence"
    assert not np.any(np.isinf(div_m)), "Inf values in divergence"

    print(f"  2D diagonal diffusion: shape {div_m.shape}, range [{div_m.min():.3e}, {div_m.max():.3e}]")

    # Test that Laplacian of Gaussian is negative (diffusion smooths peaks)
    assert div_m.sum() < 0, "Laplacian of Gaussian should be negative at peak"

    # Test 3D tensor diffusion (general nD implementation)
    print("\n  Testing 3D tensor diffusion...")
    Nx, Ny, Nz = 10, 10, 10
    dx3, dy3, dz3 = 0.1, 0.1, 0.1

    # Create 3D Gaussian
    x3 = np.linspace(0, (Nx - 1) * dx3, Nx)
    y3 = np.linspace(0, (Ny - 1) * dy3, Ny)
    z3 = np.linspace(0, (Nz - 1) * dz3, Nz)
    X3, Y3, Z3 = np.meshgrid(x3, y3, z3, indexing="ij")
    m3 = np.exp(-((X3 - 0.5) ** 2 + (Y3 - 0.5) ** 2 + (Z3 - 0.5) ** 2) / 0.1)

    # 3x3 diffusion tensor (isotropic for simplicity)
    sigma_3d = 0.1 * np.eye(3)

    # Test with nD function
    bc3 = periodic_bc(dimension=3)
    div_m3 = divergence_tensor_diffusion_nd(m3, sigma_3d, (dx3, dy3, dz3), bc3)

    assert div_m3.shape == m3.shape, f"3D shape mismatch: {div_m3.shape} vs {m3.shape}"
    assert not np.any(np.isnan(div_m3)), "NaN values in 3D divergence"
    assert not np.any(np.isinf(div_m3)), "Inf values in 3D divergence"

    print(f"  3D tensor diffusion: shape {div_m3.shape}, range [{div_m3.min():.3e}, {div_m3.max():.3e}]")

    # Test 4D (to verify true nD capability)
    print("\n  Testing 4D tensor diffusion...")
    shape_4d = (5, 5, 5, 5)
    dx_4d = (0.2, 0.2, 0.2, 0.2)
    m4 = np.random.rand(*shape_4d)
    sigma_4d = 0.1 * np.eye(4)

    bc4 = periodic_bc(dimension=4)
    div_m4 = divergence_tensor_diffusion_nd(m4, sigma_4d, dx_4d, bc4)

    assert div_m4.shape == m4.shape, f"4D shape mismatch: {div_m4.shape} vs {m4.shape}"
    assert not np.any(np.isnan(div_m4)), "NaN values in 4D divergence"
    print(f"  4D tensor diffusion: shape {div_m4.shape}, range [{div_m4.min():.3e}, {div_m4.max():.3e}]")

    print("\nAll smoke tests passed!")
