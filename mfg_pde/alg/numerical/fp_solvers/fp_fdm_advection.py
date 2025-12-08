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

    if ndim == 2:
        dx, dy = spacing

        # Compute grad(U) with central differences for velocity field
        grad_U_x = np.zeros_like(U)
        grad_U_y = np.zeros_like(U)

        # Central differences with boundary handling
        grad_U_x[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / (2 * dx)
        grad_U_y[1:-1, :] = (U[2:, :] - U[:-2, :]) / (2 * dy)

        # Drift velocity: alpha = -lambda * grad(U)
        alpha_x = -coupling_coefficient * grad_U_x
        alpha_y = -coupling_coefficient * grad_U_y

        # Upwind scheme for advection: div(alpha * m)
        # Use one-sided differences based on flow direction
        # Positive velocity -> backward difference (upwind from left)
        # Negative velocity -> forward difference (upwind from right)

        # X-direction upwind flux divergence
        # Forward difference: (flux[i+1] - flux[i]) / dx
        # Backward difference: (flux[i] - flux[i-1]) / dx
        flux_x = alpha_x * M

        # Compute both forward and backward differences
        d_flux_x_forward = np.zeros_like(M)
        d_flux_x_backward = np.zeros_like(M)
        d_flux_x_forward[:, :-1] = (flux_x[:, 1:] - flux_x[:, :-1]) / dx
        d_flux_x_backward[:, 1:] = (flux_x[:, 1:] - flux_x[:, :-1]) / dx

        # Select based on velocity direction (upwind)
        advection += np.where(alpha_x >= 0, d_flux_x_backward, d_flux_x_forward)

        # Y-direction upwind flux divergence
        flux_y = alpha_y * M

        d_flux_y_forward = np.zeros_like(M)
        d_flux_y_backward = np.zeros_like(M)
        d_flux_y_forward[:-1, :] = (flux_y[1:, :] - flux_y[:-1, :]) / dy
        d_flux_y_backward[1:, :] = (flux_y[1:, :] - flux_y[:-1, :]) / dy

        # Select based on velocity direction (upwind)
        advection += np.where(alpha_y >= 0, d_flux_y_backward, d_flux_y_forward)

    else:
        # General nD (placeholder)
        raise NotImplementedError(f"Advection term not yet implemented for {ndim}D")

    return advection
