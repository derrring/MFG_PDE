"""Advection term discretization for FP FDM solver.

This module provides advection term computation using upwind schemes
for the Fokker-Planck finite difference solver.

Module structure per issue #388:
    fp_fdm_advection.py - How transport is discretized (upwinding, centered schemes)

Functions:
    compute_advection_term_nd: Compute div(alpha * m) using upwind scheme

Mathematical Background:
    Advection term: div(alpha * m) where alpha = -coupling_coefficient * grad(U)

    Uses upwind scheme for stability in advection-dominated problems:
    - Positive velocity: backward difference (upwind from left)
    - Negative velocity: forward difference (upwind from right)
"""

from __future__ import annotations

from typing import Any

import numpy as np


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

    Uses upwind scheme for stability.

    Parameters
    ----------
    M : np.ndarray
        Density field
    U : np.ndarray
        Value function
    coupling_coefficient : float
        Coupling coefficient for drift term
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

    This ensures numerical stability for advection-dominated problems.
    """
    # Compute drift: alpha = -coupling_coefficient * grad(U)
    # Use upwind scheme for advection-dominated stability

    advection = np.zeros_like(M)

    # General nD implementation using dimension loop
    for d in range(ndim):
        dx = spacing[d]

        # Compute grad(U) along dimension d with central differences
        # Use np.gradient for general nD gradient computation
        grad_U_d = np.gradient(U, dx, axis=d)

        # Drift velocity: alpha_d = -coupling_coefficient * grad_U_d
        alpha_d = -coupling_coefficient * grad_U_d

        # Compute flux: flux_d = alpha_d * M
        flux_d = alpha_d * M

        # Upwind scheme for advection: compute forward and backward differences
        # Forward difference: (flux[i+1] - flux[i]) / dx at position i
        # Backward difference: (flux[i] - flux[i-1]) / dx at position i

        # Create slice objects for axis-agnostic indexing
        # For dimension d: we need flux[..., i+1, ...] - flux[..., i, ...]
        slice_all = slice(None)
        n_d = M.shape[d]

        # Compute differences using np.diff along axis d
        flux_diff = np.diff(flux_d, axis=d) / dx  # shape reduced by 1 along axis d

        # Forward difference: result at positions 0 to n-2 (size n-1)
        # We pad with zeros at the end (position n-1)
        d_flux_forward = np.zeros_like(M)
        slices_forward_dst = [slice_all] * ndim
        slices_forward_dst[d] = slice(0, n_d - 1)
        d_flux_forward[tuple(slices_forward_dst)] = flux_diff

        # Backward difference: result at positions 1 to n-1 (size n-1)
        # We pad with zeros at the beginning (position 0)
        d_flux_backward = np.zeros_like(M)
        slices_backward_dst = [slice_all] * ndim
        slices_backward_dst[d] = slice(1, n_d)
        d_flux_backward[tuple(slices_backward_dst)] = flux_diff

        # Select based on velocity direction (upwind)
        # Positive velocity -> backward difference (upwind from left)
        # Negative velocity -> forward difference (upwind from right)
        advection += np.where(alpha_d >= 0, d_flux_backward, d_flux_forward)

    return advection
