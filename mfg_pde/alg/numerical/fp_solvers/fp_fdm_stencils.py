"""
FDM stencil building functions for Fokker-Planck equation discretization.

This module contains functions for building sparse matrix entries for both
gradient-based and conservative (flux-based) FDM discretizations.

Functions:
    - is_boundary_point: Check if a grid point is on the domain boundary
    - add_interior_entries: Gradient-based interior stencil
    - add_boundary_no_flux_entries: Gradient-based no-flux BC stencil
    - add_interior_entries_conservative: Conservative flux interior stencil
    - add_boundary_no_flux_entries_conservative: Conservative flux no-flux BC stencil
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mfg_pde.utils.aux_func import npart, ppart

if TYPE_CHECKING:
    import numpy as np


def is_boundary_point(multi_idx: tuple[int, ...], shape: tuple[int, ...], ndim: int) -> bool:
    """Check if a grid point is on the boundary."""
    return any(multi_idx[d] == 0 or multi_idx[d] == shape[d] - 1 for d in range(ndim))


def add_interior_entries(
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
        (1/dt) m + div(m*v) - (sigma^2/2) Delta m = 0

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
        # Handle both legacy BC interface and new BoundaryConditionManager2D
        if hasattr(boundary_conditions, "is_uniform") and hasattr(boundary_conditions, "type"):
            is_periodic = boundary_conditions.is_uniform and boundary_conditions.type == "periodic"
        else:
            # For BoundaryConditionManager2D or unknown types, default to non-periodic
            is_periodic = False

        if is_periodic:
            multi_idx_plus[d] = multi_idx_plus[d] % shape[d]
            multi_idx_minus[d] = multi_idx_minus[d] % shape[d]

        # Check if neighbors exist (non-periodic case)
        has_plus = multi_idx_plus[d] < shape[d]
        has_minus = multi_idx_minus[d] >= 0

        if has_plus or is_periodic:
            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            u_plus = u_flat[flat_idx_plus]
        else:
            u_plus = u_flat[flat_idx]  # Use current value at boundary

        if has_minus or is_periodic:
            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))
            u_minus = u_flat[flat_idx_minus]
        else:
            u_minus = u_flat[flat_idx]  # Use current value at boundary

        u_center = u_flat[flat_idx]

        # Diffusion contribution (centered differences)
        # -sigma^2/(2dx^2) * (m_{i+1} - 2m_i + m_{i-1})
        diagonal_value += sigma**2 / dx_sq

        if has_plus or is_periodic:
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

        if has_minus or is_periodic:
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


def add_boundary_no_flux_entries(
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

    No-flux BC for Fokker-Planck: J.n = 0 where J = alpha*m - D*grad(m)

    Uses flux-form (finite volume) discretization at boundaries:
    - Boundary cells treated as half-cells (volume dx/2)
    - Flux at boundary face = 0 (no-flux condition)
    - Interior face flux uses upwind for advection

    Mathematical formulation:
    For FP equation: dm/dt = D*Delta(m) - div(alpha*m) where alpha = -lambda*grad(U)

    At right boundary (half-cell):
        (dx/2)*dm/dt = J_interior - J_boundary = J_interior - 0
        dm/dt = 2*J_interior / dx

    This ensures mass conservation by explicitly zeroing boundary flux.
    """
    # Diagonal term (time derivative)
    diagonal_value = 1.0 / dt

    # Diffusion coefficient D = sigma^2/2
    D = sigma**2 / 2.0

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

            # Diffusion: D*(m_{i+1} - 2m_i + m_{i-1}) / dx^2
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
            # Left boundary: ghost point approach for no-flux BC
            #
            # For no-flux: J.n = 0 where J = alpha*m - D*grad(m)
            # This gives: dm/dn = (alpha.n)*m / D at boundary
            #
            # Using ghost point: m_ghost = m_0 (reflection for pure diffusion)
            # But for advection-diffusion, we need to adjust.
            #
            # Simplified approach: Use one-sided stencil that preserves
            # row sum = 1/dt for mass conservation.
            #
            # At left boundary, use forward stencil for diffusion:
            # d^2m/dx^2 approx (m_1 - m_0) / dx^2 (first-order one-sided)
            #
            # For advection with upwind:
            # If alpha > 0 (flow right), mass leaves: d(alpha*m)/dx approx alpha*(m_0 - m_ghost)/dx
            #   With no-flux, mass that "wants to leave" stays, so we use zero BC
            # If alpha < 0 (flow left, into boundary), upwind from right: d(alpha*m)/dx approx alpha*(m_1 - m_0)/dx

            multi_idx_plus = list(multi_idx)
            multi_idx_plus[d] = multi_idx[d] + 1

            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            u_plus = u_flat[flat_idx_plus]
            u_center = u_flat[flat_idx]

            # Diffusion: one-sided gives (m_1 - m_0)/dx^2 for no-flux (ghost = m_0)
            # This adds: +D/dx^2 to diagonal, -D/dx^2 to coeff_plus
            diagonal_value += D / dx_sq
            coeff_plus = -D / dx_sq

            # Advection: use upwind with proper no-flux handling
            # alpha = -lambda*grad(U)
            grad_U = (u_plus - u_center) / dx
            alpha = -coupling_coefficient * grad_U

            if alpha >= 0:
                # Flow to right (away from left boundary)
                # No-flux means this flux is zero - mass stays in place
                # Don't add advection terms (they would cause mass loss)
                pass
            else:
                # Flow to left (into left boundary from right)
                # Upwind from m_1: advection term = alpha*(m_1 - m_0)/dx (with alpha < 0)
                # In implicit form: -alpha/dx * (m_1 - m_0)
                # = -alpha/dx * m_1 + alpha/dx * m_0
                diagonal_value += -alpha / dx  # Note: alpha < 0, so -alpha > 0
                coeff_plus += alpha / dx  # alpha < 0, so this is negative

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(coeff_plus)

        elif at_right_boundary:
            # Right boundary: ghost point approach for no-flux BC
            #
            # Similar to left boundary but mirrored.

            multi_idx_minus = list(multi_idx)
            multi_idx_minus[d] = multi_idx[d] - 1

            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))
            u_minus = u_flat[flat_idx_minus]
            u_center = u_flat[flat_idx]

            # Diffusion: one-sided gives (m_{-1} - m_N)/dx^2 for no-flux
            # This adds: +D/dx^2 to diagonal, -D/dx^2 to coeff_minus
            diagonal_value += D / dx_sq
            coeff_minus = -D / dx_sq

            # Advection: use upwind with proper no-flux handling
            grad_U = (u_center - u_minus) / dx
            alpha = -coupling_coefficient * grad_U

            if alpha >= 0:
                # Flow to right (into right boundary from left)
                # Upwind from m_{-1}: advection term = alpha*(m_N - m_{-1})/dx
                # In implicit form: -alpha/dx * (m_N - m_{-1})
                # = -alpha/dx * m_N + alpha/dx * m_{-1}
                diagonal_value += alpha / dx  # alpha > 0
                coeff_minus += -alpha / dx  # negative
            else:
                # Flow to left (away from right boundary)
                # No-flux means this flux is zero - mass stays in place
                pass

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(coeff_minus)

    # Add diagonal entry
    row_indices.append(flat_idx)
    col_indices.append(flat_idx)
    data_values.append(diagonal_value)


# ============================================================================
# Conservative Flux FDM discretization (mass-preserving)
# ============================================================================


def add_interior_entries_conservative(
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
    Add matrix entries for interior grid point using CONSERVATIVE Flux FDM.

    Conservative discretization: div(J) where J = alpha*m - D*grad(m)

    Flux at cell interface i+1/2:
        F_{i+1/2} = alpha_{i+1/2} * m_upwind - D * (m_{i+1} - m_i) / dx

    where alpha_{i+1/2} = -lambda * (U_{i+1} - U_i) / dx

    Key property: Column sums = 1/dt (exact mass conservation)
    """
    # Diagonal term (time derivative)
    diagonal_value = 1.0 / dt

    # Diffusion coefficient D = sigma^2/2
    D = sigma**2 / 2.0

    # For each dimension, add advection + diffusion contributions
    for d in range(ndim):
        dx = spacing[d]

        # Get neighbor indices in dimension d
        multi_idx_plus = list(multi_idx)
        multi_idx_minus = list(multi_idx)

        multi_idx_plus[d] = multi_idx[d] + 1
        multi_idx_minus[d] = multi_idx[d] - 1

        # Handle boundary wrapping for periodic BC
        if hasattr(boundary_conditions, "is_uniform") and hasattr(boundary_conditions, "type"):
            is_periodic = boundary_conditions.is_uniform and boundary_conditions.type == "periodic"
        else:
            is_periodic = False

        if is_periodic:
            multi_idx_plus[d] = multi_idx_plus[d] % shape[d]
            multi_idx_minus[d] = multi_idx_minus[d] % shape[d]

        # Check if neighbors exist (non-periodic case)
        has_plus = multi_idx_plus[d] < shape[d]
        has_minus = multi_idx_minus[d] >= 0

        if (has_plus or is_periodic) and (has_minus or is_periodic):
            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))

            u_plus = u_flat[flat_idx_plus]
            u_minus = u_flat[flat_idx_minus]
            u_center = u_flat[flat_idx]

            # Diffusion (conservative centered differences)
            # d^2m/dx^2 approx (m_{i+1} - 2m_i + m_{i-1}) / dx^2
            # Contributes: +2D/dx^2 to diagonal, -D/dx^2 to each neighbor
            diagonal_value += 2 * D / (dx * dx)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(-D / (dx * dx))

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(-D / (dx * dx))

            # Conservative flux advection
            # Right flux F_{i+1/2} uses velocity alpha_{i+1/2} = -lambda * (U_{i+1} - U_i) / dx
            alpha_right = -coupling_coefficient * (u_plus - u_center) / dx
            # Left flux F_{i-1/2} uses velocity alpha_{i-1/2} = -lambda * (U_i - U_{i-1}) / dx
            alpha_left = -coupling_coefficient * (u_center - u_minus) / dx

            # Right flux F_{i+1/2}: upwind selection
            if alpha_right >= 0:
                # Flow to right: upwind from m_i (diagonal contribution)
                # F_{i+1/2} = alpha_right * m_i
                # In matrix: -div(F) = -(F_{i+1/2} - F_{i-1/2})/dx
                # Contribution to diagonal from F_{i+1/2}: +alpha_right/dx
                diagonal_value += alpha_right / dx
            else:
                # Flow to left: upwind from m_{i+1}
                # F_{i+1/2} = alpha_right * m_{i+1}
                # Contribution to col_{i+1} from F_{i+1/2}: +alpha_right/dx (negative)
                row_indices.append(flat_idx)
                col_indices.append(flat_idx_plus)
                data_values.append(alpha_right / dx)

            # Left flux -F_{i-1/2}: upwind selection (note the negative sign)
            if alpha_left >= 0:
                # Flow to right: upwind from m_{i-1}
                # F_{i-1/2} = alpha_left * m_{i-1}
                # Contribution to col_{i-1} from -F_{i-1/2}: -alpha_left/dx
                row_indices.append(flat_idx)
                col_indices.append(flat_idx_minus)
                data_values.append(-alpha_left / dx)
            else:
                # Flow to left: upwind from m_i (diagonal contribution)
                # F_{i-1/2} = alpha_left * m_i
                # Contribution to diagonal from -F_{i-1/2}: -alpha_left/dx (positive since alpha_left < 0)
                diagonal_value += -alpha_left / dx

    # Add diagonal entry
    row_indices.append(flat_idx)
    col_indices.append(flat_idx)
    data_values.append(diagonal_value)


def add_boundary_no_flux_entries_conservative(
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
    Add matrix entries for boundary grid point with no-flux BC using CONSERVATIVE Flux FDM.

    No-flux BC: Total flux J = alpha*m - D*grad(m) = 0 at boundary.

    For conservative scheme:
    - Boundary flux F_{boundary} = 0 (enforced exactly)
    - Interior flux uses standard upwind selection
    - This maintains column sum = 1/dt even at boundaries
    """
    # Diagonal term (time derivative)
    diagonal_value = 1.0 / dt

    # Diffusion coefficient D = sigma^2/2
    D = sigma**2 / 2.0

    # For each dimension, check if we're at a boundary in that dimension
    for d in range(ndim):
        dx = spacing[d]

        at_left_boundary = multi_idx[d] == 0
        at_right_boundary = multi_idx[d] == shape[d] - 1
        at_interior_in_d = not (at_left_boundary or at_right_boundary)

        if at_interior_in_d:
            # Standard interior flux stencil in this dimension
            multi_idx_plus = list(multi_idx)
            multi_idx_minus = list(multi_idx)
            multi_idx_plus[d] = multi_idx[d] + 1
            multi_idx_minus[d] = multi_idx[d] - 1

            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))

            u_plus = u_flat[flat_idx_plus]
            u_minus = u_flat[flat_idx_minus]
            u_center = u_flat[flat_idx]

            # Diffusion (conservative centered differences)
            diagonal_value += 2 * D / (dx * dx)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(-D / (dx * dx))

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(-D / (dx * dx))

            # Conservative flux advection
            alpha_right = -coupling_coefficient * (u_plus - u_center) / dx
            alpha_left = -coupling_coefficient * (u_center - u_minus) / dx

            # Right flux F_{i+1/2}
            if alpha_right >= 0:
                diagonal_value += alpha_right / dx
            else:
                row_indices.append(flat_idx)
                col_indices.append(flat_idx_plus)
                data_values.append(alpha_right / dx)

            # Left flux -F_{i-1/2}
            if alpha_left >= 0:
                row_indices.append(flat_idx)
                col_indices.append(flat_idx_minus)
                data_values.append(-alpha_left / dx)
            else:
                diagonal_value += -alpha_left / dx

        elif at_left_boundary:
            # Left boundary: F_{-1/2} = 0 (no-flux), only interior flux F_{1/2}
            multi_idx_plus = list(multi_idx)
            multi_idx_plus[d] = multi_idx[d] + 1

            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            u_plus = u_flat[flat_idx_plus]
            u_center = u_flat[flat_idx]

            # Diffusion: one-sided for no-flux (ghost = m_0)
            # d^2m/dx^2 approx (m_1 - m_0)/dx^2 when ghost = m_0
            diagonal_value += D / (dx * dx)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(-D / (dx * dx))

            # Conservative flux advection: only F_{1/2}, no F_{-1/2} (zero flux at boundary)
            alpha_right = -coupling_coefficient * (u_plus - u_center) / dx

            if alpha_right >= 0:
                # Outflow to right
                diagonal_value += alpha_right / dx
            else:
                # Inflow from right
                row_indices.append(flat_idx)
                col_indices.append(flat_idx_plus)
                data_values.append(alpha_right / dx)

            # Note: NO contribution from left flux - it's exactly zero (no-flux BC)
            # This is key for mass conservation at boundaries

        elif at_right_boundary:
            # Right boundary: F_{N+1/2} = 0 (no-flux), only interior flux F_{N-1/2}
            multi_idx_minus = list(multi_idx)
            multi_idx_minus[d] = multi_idx[d] - 1

            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))
            u_minus = u_flat[flat_idx_minus]
            u_center = u_flat[flat_idx]

            # Diffusion: one-sided for no-flux (ghost = m_N)
            diagonal_value += D / (dx * dx)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(-D / (dx * dx))

            # Conservative flux advection: only -F_{N-1/2}, no F_{N+1/2} (zero flux at boundary)
            alpha_left = -coupling_coefficient * (u_center - u_minus) / dx

            if alpha_left >= 0:
                # Inflow from left
                row_indices.append(flat_idx)
                col_indices.append(flat_idx_minus)
                data_values.append(-alpha_left / dx)
            else:
                # Outflow to left
                diagonal_value += -alpha_left / dx

            # Note: NO contribution from right flux - it's exactly zero (no-flux BC)

    # Add diagonal entry
    row_indices.append(flat_idx)
    col_indices.append(flat_idx)
    data_values.append(diagonal_value)
