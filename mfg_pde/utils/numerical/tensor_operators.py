"""
Tensor Diffusion Operators for Anisotropic PDEs.

This module implements finite difference operators for tensor diffusion:
    ∇ · (Σ ∇m)
where Σ is a d×d symmetric positive semi-definite diffusion tensor.

Key Features:
- Supports constant and spatially-varying tensors
- Handles periodic, Dirichlet, and no-flux boundary conditions
- Optimized diagonal case for efficiency
- Full anisotropic and cross-diffusion support
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.geometry import BoundaryConditions


def divergence_tensor_diffusion_2d(
    m: NDArray[np.floating],
    sigma_tensor: NDArray[np.floating],
    dx: float,
    dy: float,
    boundary_conditions: BoundaryConditions,
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
        boundary_conditions: Boundary condition specification

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
    m_padded = _apply_bc_2d(m, boundary_conditions)

    # Compute gradients using central differences
    # Note: gradients are at cell faces (staggered grid)

    # ∂m/∂x at faces (i+1/2, j)
    dm_dx_x = (m_padded[1:-1, 2:] - m_padded[1:-1, 1:-1]) / dx  # Shape: (Ny, Nx+1)

    # ∂m/∂y at faces (i, j+1/2)
    dm_dy_y = (m_padded[2:, 1:-1] - m_padded[1:-1, 1:-1]) / dy  # Shape: (Ny+1, Nx)

    # ∂m/∂x at y-faces (i, j+1/2) - need to average in y
    dm_dx_y = (m_padded[1:, 1:-1] + m_padded[:-1, 1:-1] - m_padded[1:, :-2] - m_padded[:-1, :-2]) / (
        2 * dx
    )  # Shape: (Ny+1, Nx)

    # ∂m/∂y at x-faces (i+1/2, j) - need to average in x
    dm_dy_x = (m_padded[1:-1, 1:] + m_padded[1:-1, :-1] - m_padded[:-2, 1:] - m_padded[:-2, :-1]) / (
        2 * dy
    )  # Shape: (Ny, Nx+1)

    # Compute fluxes F = Σ ∇m at cell faces
    # Fx = (σ₁₁ ∂m/∂x + σ₁₂ ∂m/∂y) at x-faces
    # Fy = (σ₂₁ ∂m/∂x + σ₂₂ ∂m/∂y) at y-faces

    # Average Σ to faces
    # Σ at x-faces (i+1/2, j): average in x
    Sigma_x = 0.5 * (Sigma[:, 1:, :, :] + Sigma[:, :-1, :, :])  # Shape: (Ny, Nx+1, 2, 2)

    # Σ at y-faces (i, j+1/2): average in y
    Sigma_y = 0.5 * (Sigma[1:, :, :, :] + Sigma[:-1, :, :, :])  # Shape: (Ny+1, Nx, 2, 2)

    # Compute fluxes
    Fx = Sigma_x[:, :, 0, 0] * dm_dx_x + Sigma_x[:, :, 0, 1] * dm_dy_x  # Shape: (Ny, Nx+1)
    Fy = Sigma_y[:, :, 1, 0] * dm_dx_y + Sigma_y[:, :, 1, 1] * dm_dy_y  # Shape: (Ny+1, Nx)

    # Compute divergence: ∇ · F = ∂Fx/∂x + ∂Fy/∂y
    div_x = (Fx[:, 1:] - Fx[:, :-1]) / dx  # Shape: (Ny, Nx)
    div_y = (Fy[1:, :] - Fy[:-1, :]) / dy  # Shape: (Ny, Nx)

    return div_x + div_y


def divergence_diagonal_diffusion_2d(
    m: NDArray[np.floating],
    sigma_diag: NDArray[np.floating],
    dx: float,
    dy: float,
    boundary_conditions: BoundaryConditions,
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
        boundary_conditions: BC specification

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
    m_padded = _apply_bc_2d(m, boundary_conditions)

    # Compute ∂/∂x(σₓ² ∂m/∂x)
    # Gradient at faces
    dm_dx = (m_padded[1:-1, 2:] - m_padded[1:-1, 1:-1]) / dx  # At (i+1/2, j)

    # Average σₓ² to faces
    sigma_x_face = 0.5 * (sigma_x[:, 1:] + sigma_x[:, :-1])

    # Flux
    Fx = sigma_x_face * dm_dx

    # Divergence in x
    div_x = (Fx[:, 1:] - Fx[:, :-1]) / dx

    # Compute ∂/∂y(σᵧ² ∂m/∂y) similarly
    dm_dy = (m_padded[2:, 1:-1] - m_padded[1:-1, 1:-1]) / dy
    sigma_y_face = 0.5 * (sigma_y[1:, :] + sigma_y[:-1, :])
    Fy = sigma_y_face * dm_dy
    div_y = (Fy[1:, :] - Fy[:-1, :]) / dy

    return div_x + div_y


def _apply_bc_2d(
    m: NDArray[np.floating],
    boundary_conditions: BoundaryConditions,
) -> NDArray[np.floating]:
    """
    Apply boundary conditions by padding array with ghost cells.

    Args:
        m: Interior field (Ny, Nx)
        boundary_conditions: BC specification

    Returns:
        Padded field (Ny+2, Nx+2) with ghost cells

    Supported BC types:
        - periodic: Wrap-around boundaries
        - dirichlet: Zero boundary values
        - no_flux (Neumann): Zero normal derivative
    """
    bc_type = boundary_conditions.type.lower()

    if bc_type == "periodic":
        # Periodic: wrap around
        m_padded = np.pad(m, 1, mode="wrap")

    elif bc_type == "dirichlet":
        # Dirichlet: zero at boundaries
        m_padded = np.pad(m, 1, mode="constant", constant_values=0.0)

    elif bc_type in ["no_flux", "neumann"]:
        # No-flux: zero normal derivative (reflective)
        m_padded = np.pad(m, 1, mode="edge")

    else:
        raise ValueError(f"Unsupported boundary condition type: {bc_type}")

    return m_padded


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
    """1D tensor diffusion (reduces to scalar diffusion)."""
    # In 1D, Σ is just a scalar (1×1 matrix)
    if isinstance(sigma_tensor, np.ndarray):
        if sigma_tensor.ndim == 2:
            # Extract scalar from (1, 1) matrix
            sigma = sigma_tensor[0, 0]
        elif sigma_tensor.ndim == 1:
            # Already a scalar array
            sigma = sigma_tensor
        else:
            # Spatially varying (Nx, 1, 1)
            sigma = sigma_tensor[:, 0, 0]
    else:
        sigma = sigma_tensor

    # Use standard 1D laplacian
    from mfg_pde.alg.numerical.fp_solvers.fp_fdm import _apply_diffusion_1d

    return _apply_diffusion_1d(m, sigma, dx, boundary_conditions)


def _divergence_tensor_general_nd(
    m: NDArray[np.floating],
    sigma_tensor: NDArray[np.floating],
    dx: tuple[float, ...],
    boundary_conditions: BoundaryConditions,
) -> NDArray[np.floating]:
    """
    General nD tensor diffusion using component-wise stencils.

    Implementation note:
        This is a placeholder for full nD support.
        For production use, consider optimized implementations or
        using existing PDE libraries (e.g., FEniCS, Firedrake).
    """
    raise NotImplementedError(
        "General nD tensor diffusion (d>2) not yet implemented. "
        "Supported dimensions: 1D (scalar) and 2D (full tensor). "
        "For higher dimensions, consider using diagonal diffusion or "
        "implementing problem-specific operators."
    )
