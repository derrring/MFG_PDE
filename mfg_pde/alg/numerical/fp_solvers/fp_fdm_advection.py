"""Advection term discretization for FP FDM solver.

This module provides advection term computation using upwind schemes
for the Fokker-Planck finite difference solver.

Module structure per issue #388:
    fp_fdm_advection.py - How transport is discretized (upwinding, centered schemes)

Functions:
    compute_advection_term_nd: Compute div(alpha * m) using upwind scheme
    compute_advection_from_drift_nd: Compute div(alpha * m) with drift provided directly

Mathematical Background:
    Advection term: div(alpha * m)

    Two modes:
    1. MFG coupled: alpha = -coupling_coefficient * grad(U) (derived from HJB)
    2. Standalone FP: alpha provided directly by user

    Uses upwind scheme for stability in advection-dominated problems:
    - Positive velocity: backward difference (upwind from left)
    - Negative velocity: forward difference (upwind from right)
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _compute_upwind_advection(
    M: np.ndarray,
    drift_per_dim: list[np.ndarray],
    spacing: tuple[float, ...],
    ndim: int,
) -> np.ndarray:
    """
    Core upwind advection computation given drift per dimension.

    This is the shared implementation used by both U-based and drift-based
    advection functions.

    Parameters
    ----------
    M : np.ndarray
        Density field
    drift_per_dim : list[np.ndarray]
        List of drift arrays, one per dimension. Each has same shape as M.
    spacing : tuple[float, ...]
        Grid spacing (dx, dy, ...)
    ndim : int
        Spatial dimension

    Returns
    -------
    np.ndarray
        Advection term div(alpha * m), same shape as M
    """
    advection = np.zeros_like(M)

    for d in range(ndim):
        dx = spacing[d]
        alpha_d = drift_per_dim[d]

        # Compute flux: flux_d = alpha_d * M
        flux_d = alpha_d * M

        # Upwind scheme for advection
        slice_all = slice(None)
        n_d = M.shape[d]

        # Compute differences using np.diff along axis d
        flux_diff = np.diff(flux_d, axis=d) / dx

        # Forward difference: result at positions 0 to n-2
        d_flux_forward = np.zeros_like(M)
        slices_forward_dst = [slice_all] * ndim
        slices_forward_dst[d] = slice(0, n_d - 1)
        d_flux_forward[tuple(slices_forward_dst)] = flux_diff

        # Backward difference: result at positions 1 to n-1
        d_flux_backward = np.zeros_like(M)
        slices_backward_dst = [slice_all] * ndim
        slices_backward_dst[d] = slice(1, n_d)
        d_flux_backward[tuple(slices_backward_dst)] = flux_diff

        # Select based on velocity direction (upwind)
        advection += np.where(alpha_d >= 0, d_flux_backward, d_flux_forward)

    return advection


def compute_advection_term_nd(
    M: np.ndarray,
    U: np.ndarray,
    coupling_coefficient: float,
    spacing: tuple[float, ...],
    ndim: int,
    boundary_conditions: Any,
) -> np.ndarray:
    """
    Compute advection term div(alpha * m) where alpha = -coupling_coefficient * grad(U).

    This is the MFG-coupled mode where drift is derived from HJB value function U
    via Pontryagin's maximum principle.

    Parameters
    ----------
    M : np.ndarray
        Density field
    U : np.ndarray
        Value function from HJB equation
    coupling_coefficient : float
        Coupling coefficient for drift term (λ in v = -λ∇U)
    spacing : tuple
        Grid spacing (dx, dy, ...)
    ndim : int
        Spatial dimension
    boundary_conditions : BoundaryConditions
        Boundary conditions specification

    Returns
    -------
    np.ndarray
        Advection term div(alpha * m), same shape as M

    Notes
    -----
    The upwind scheme selects the direction based on flow direction:
    - For alpha >= 0 (flow to right): use backward difference
    - For alpha < 0 (flow to left): use forward difference
    """
    # Compute drift from U: alpha_d = -coupling_coefficient * grad_U_d
    drift_per_dim = []
    for d in range(ndim):
        dx = spacing[d]
        grad_U_d = np.gradient(U, dx, axis=d)
        alpha_d = -coupling_coefficient * grad_U_d
        drift_per_dim.append(alpha_d)

    return _compute_upwind_advection(M, drift_per_dim, spacing, ndim)


def compute_advection_from_drift_nd(
    M: np.ndarray,
    drift: np.ndarray,
    spacing: tuple[float, ...],
    ndim: int,
) -> np.ndarray:
    """
    Compute advection term div(alpha * m) with drift alpha provided directly.

    This is the standalone FP mode where user provides drift field directly,
    without going through HJB value function.

    Parameters
    ----------
    M : np.ndarray
        Density field with shape (N1, N2, ..., Nd)
    drift : np.ndarray
        Drift field. For 1D: shape (N,) scalar drift.
        For nD: shape (ndim, N1, N2, ..., Nd) vector drift.
    spacing : tuple[float, ...]
        Grid spacing (dx, dy, ...)
    ndim : int
        Spatial dimension

    Returns
    -------
    np.ndarray
        Advection term div(alpha * m), same shape as M

    Notes
    -----
    This function directly uses the provided drift without any conversion.
    For MFG systems where drift comes from HJB via v = -λ∇U, use
    compute_advection_term_nd instead.
    """
    # Parse drift into per-dimension arrays
    if ndim == 1:
        # 1D: drift is scalar field
        if drift.ndim == 1:
            drift_per_dim = [drift]
        elif drift.ndim == 2 and drift.shape[0] == 1:
            drift_per_dim = [drift[0]]
        else:
            drift_per_dim = [drift.ravel()]
    else:
        # nD: drift should be vector field (ndim, N1, N2, ...)
        if drift.ndim == ndim + 1 and drift.shape[0] == ndim:
            drift_per_dim = [drift[d] for d in range(ndim)]
        elif drift.ndim == ndim:
            # Scalar drift applied to first dimension only (simplified case)
            drift_per_dim = [drift] + [np.zeros_like(drift) for _ in range(ndim - 1)]
        else:
            raise ValueError(
                f"Drift shape {drift.shape} incompatible with {ndim}D grid. "
                f"Expected ({ndim}, ...) for vector drift or (...) for scalar drift."
            )

    return _compute_upwind_advection(M, drift_per_dim, spacing, ndim)
