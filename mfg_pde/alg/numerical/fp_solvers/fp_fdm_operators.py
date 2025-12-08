"""Finite difference operators for FP equation discretization.

This module provides the mathematical operations used in non-conservative
(gradient-based) FDM discretization: interior stencils for diffusion and advection.

Module structure per issue #388:
    fp_fdm_operators.py - What mathematical operations (differential operators, stencils)

Functions:
    is_boundary_point: Utility to check if a point is on the boundary
    add_interior_entries: Interior stencil for gradient FDM (non-conservative)

Note:
    - Boundary condition enforcement is in fp_fdm_bc.py
    - Conservative flux discretization is in fp_fdm_alg_flux.py
    - Advection term computation is in fp_fdm_advection.py
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
    Add matrix entries for interior grid point using non-conservative gradient FDM.

    Discretizes:
        (1/dt) m + div(m*v) - (sigma^2/2) Laplacian(m) = 0

    Using upwind for advection and centered differences for diffusion.

    Note: This is non-conservative discretization. For mass-preserving
    discretization, use add_interior_entries_conservative from fp_fdm_alg_flux.py.
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
