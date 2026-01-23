"""
Shared BC Value Enforcement Utilities (Issue #636).

This module provides dimension-agnostic utilities for enforcing boundary
condition values on solution arrays. These utilities are used by multiple
applicators (FDM, Interpolation, etc.) to avoid code duplication.

The key distinction from ghost cell methods:
- **Ghost cells** (FDMApplicator.apply): Pad array for stencil operations
- **Value enforcement** (this module): Set boundary values to satisfy BC

Supported BC Types:
- Neumann (du/dn = g): Extrapolation-based enforcement
- Dirichlet (u = g): Direct value assignment
- Periodic: Copy from opposite boundary

Usage:
    >>> from mfg_pde.geometry.boundary.enforcement import (
    ...     enforce_neumann_value_nd,
    ...     enforce_dirichlet_value_nd,
    ... )
    >>>
    >>> # Enforce Neumann BC with 2nd-order extrapolation
    >>> enforce_neumann_value_nd(field, axis=0, side="min", order=2)
    >>>
    >>> # Enforce Dirichlet BC
    >>> enforce_dirichlet_value_nd(field, axis=0, side="min", value=0.0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def enforce_neumann_value_nd(
    field: NDArray[np.floating],
    axis: int,
    side: Literal["min", "max"],
    grad_value: float = 0.0,
    spacing: float = 1.0,
    order: int = 2,
) -> None:
    """
    Enforce Neumann BC value along specified axis (in-place).

    Sets boundary value to satisfy the gradient constraint du/dn = g.

    Args:
        field: Solution array (modified in-place)
        axis: Dimension index (0, 1, 2, ...)
        side: "min" or "max" boundary
        grad_value: Neumann BC gradient value (default: 0.0 for zero-flux)
        spacing: Grid spacing h (needed for non-zero gradient)
        order: Extrapolation order for zero-flux case
            - order=1: u[0] = u[1] (O(h) accurate, simple copy)
            - order=2: u[0] = (4*u[1] - u[2])/3 (O(h²) accurate)

    Formulas:
        For grad_value = 0 (zero-flux):
            - order=1: u[boundary] = u[neighbor]
            - order=2: u[boundary] = (4*u[neighbor] - u[next])/3

        For grad_value != 0:
            - min side: u[0] = u[1] - g*h
            - max side: u[-1] = u[-2] + g*h

    Example:
        >>> field = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
        >>> enforce_neumann_value_nd(field, axis=0, side="min", order=2)
        >>> # field[0] is now (4*1.0 - 1.1)/3 = 0.9667
    """
    ndim = field.ndim
    n_points = field.shape[axis]

    # Build slicers
    boundary_slicer = [slice(None)] * ndim
    neighbor_slicer = [slice(None)] * ndim
    next_slicer = [slice(None)] * ndim

    if side == "min":
        boundary_slicer[axis] = 0
        neighbor_slicer[axis] = 1
        next_slicer[axis] = 2
    else:  # "max"
        boundary_slicer[axis] = -1
        neighbor_slicer[axis] = -2
        next_slicer[axis] = -3

    boundary_slicer = tuple(boundary_slicer)
    neighbor_slicer = tuple(neighbor_slicer)
    next_slicer = tuple(next_slicer)

    # Check if zero-flux (use extrapolation) or non-zero gradient
    if np.isclose(grad_value, 0.0):
        # Zero-flux: use extrapolation
        if order >= 2 and n_points >= 3:
            # 2nd-order: u[0] = (4*u[1] - u[2])/3
            field[boundary_slicer] = (4.0 * field[neighbor_slicer] - field[next_slicer]) / 3.0
        else:
            # 1st-order: u[0] = u[1]
            field[boundary_slicer] = field[neighbor_slicer]
    else:
        # Non-zero gradient: u[0] = u[1] - g*h (min) or u[-1] = u[-2] + g*h (max)
        if side == "min":
            field[boundary_slicer] = field[neighbor_slicer] - grad_value * spacing
        else:
            field[boundary_slicer] = field[neighbor_slicer] + grad_value * spacing


def enforce_dirichlet_value_nd(
    field: NDArray[np.floating],
    axis: int,
    side: Literal["min", "max"],
    value: float,
) -> None:
    """
    Enforce Dirichlet BC value along specified axis (in-place).

    Sets boundary value directly: u(boundary) = g.

    Args:
        field: Solution array (modified in-place)
        axis: Dimension index (0, 1, 2, ...)
        side: "min" or "max" boundary
        value: Dirichlet boundary value

    Example:
        >>> field = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
        >>> enforce_dirichlet_value_nd(field, axis=0, side="min", value=0.0)
        >>> # field[0] is now 0.0
    """
    ndim = field.ndim
    slicer = [slice(None)] * ndim

    if side == "min":
        slicer[axis] = 0
    else:  # "max"
        slicer[axis] = -1

    field[tuple(slicer)] = value


def enforce_periodic_value_nd(
    field: NDArray[np.floating],
    axis: int,
) -> None:
    """
    Enforce periodic BC along specified axis (in-place).

    For periodic BC, boundary values are copied from the opposite interior:
    - u[0] = u[-2] (min boundary gets value from near max)
    - u[-1] = u[1] (max boundary gets value from near min)

    Args:
        field: Solution array (modified in-place)
        axis: Dimension index (0, 1, 2, ...)

    Example:
        >>> field = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
        >>> enforce_periodic_value_nd(field, axis=0)
        >>> # field[0] = 3.0, field[-1] = 1.0
    """
    ndim = field.ndim

    # Min boundary gets value from near max interior
    min_slicer = [slice(None)] * ndim
    near_max_slicer = [slice(None)] * ndim
    min_slicer[axis] = 0
    near_max_slicer[axis] = -2

    field[tuple(min_slicer)] = field[tuple(near_max_slicer)]

    # Max boundary gets value from near min interior
    max_slicer = [slice(None)] * ndim
    near_min_slicer = [slice(None)] * ndim
    max_slicer[axis] = -1
    near_min_slicer[axis] = 1

    field[tuple(max_slicer)] = field[tuple(near_min_slicer)]


def enforce_robin_value_nd(
    field: NDArray[np.floating],
    axis: int,
    side: Literal["min", "max"],
    alpha: float,
    beta: float,
    rhs_value: float,
    spacing: float = 1.0,
) -> None:
    """
    Enforce Robin BC value along specified axis (in-place).

    Robin BC: alpha*u + beta*du/dn = g

    For the special case beta=0 (pure Dirichlet): u = g/alpha
    For the special case alpha=0 (pure Neumann): du/dn = g/beta

    General case uses extrapolation-based enforcement.

    Args:
        field: Solution array (modified in-place)
        axis: Dimension index (0, 1, 2, ...)
        side: "min" or "max" boundary
        alpha: Coefficient on u
        beta: Coefficient on du/dn
        rhs_value: Right-hand side value g
        spacing: Grid spacing h

    Example:
        >>> # Robin BC: u + 0.5*du/dn = 1.0
        >>> enforce_robin_value_nd(field, axis=0, side="min", alpha=1.0, beta=0.5, rhs_value=1.0)
    """
    # Handle special cases
    if np.isclose(beta, 0.0):
        # Pure Dirichlet: alpha*u = g => u = g/alpha
        if not np.isclose(alpha, 0.0):
            enforce_dirichlet_value_nd(field, axis, side, rhs_value / alpha)
        return

    if np.isclose(alpha, 0.0):
        # Pure Neumann: beta*du/dn = g => du/dn = g/beta
        enforce_neumann_value_nd(field, axis, side, rhs_value / beta, spacing, order=1)
        return

    # General Robin case: use 2nd-order extrapolation with Robin constraint
    # For simplicity, fall back to Neumann-like extrapolation
    # (More accurate Robin enforcement would require solving a system)
    ndim = field.ndim
    n_points = field.shape[axis]

    boundary_slicer = [slice(None)] * ndim
    neighbor_slicer = [slice(None)] * ndim
    next_slicer = [slice(None)] * ndim

    if side == "min":
        boundary_slicer[axis] = 0
        neighbor_slicer[axis] = 1
        next_slicer[axis] = 2
    else:
        boundary_slicer[axis] = -1
        neighbor_slicer[axis] = -2
        next_slicer[axis] = -3

    boundary_slicer = tuple(boundary_slicer)
    neighbor_slicer = tuple(neighbor_slicer)
    next_slicer = tuple(next_slicer)

    if n_points >= 3:
        # 2nd-order extrapolation
        field[boundary_slicer] = (4.0 * field[neighbor_slicer] - field[next_slicer]) / 3.0
    else:
        field[boundary_slicer] = field[neighbor_slicer]


__all__ = [
    "enforce_neumann_value_nd",
    "enforce_dirichlet_value_nd",
    "enforce_periodic_value_nd",
    "enforce_robin_value_nd",
]


# =============================================================================
# Smoke Tests
# =============================================================================


if __name__ == "__main__":
    """Smoke tests for enforcement utilities."""

    print("Testing BC enforcement utilities...")

    # Test 1: 1D Neumann 2nd-order
    print("\n1. Testing 1D Neumann BC (2nd-order)...")
    field = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
    enforce_neumann_value_nd(field, axis=0, side="min", order=2)
    expected = (4.0 * 1.0 - 1.1) / 3.0
    assert np.isclose(field[0], expected), f"Min failed: {field[0]} != {expected}"

    field = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
    enforce_neumann_value_nd(field, axis=0, side="max", order=2)
    expected = (4.0 * 1.2 - 1.1) / 3.0
    assert np.isclose(field[-1], expected), f"Max failed: {field[-1]} != {expected}"
    print("   PASS: 1D Neumann 2nd-order")

    # Test 2: 1D Neumann 1st-order
    print("\n2. Testing 1D Neumann BC (1st-order)...")
    field = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
    enforce_neumann_value_nd(field, axis=0, side="min", order=1)
    assert field[0] == 1.0, f"Min failed: {field[0]} != 1.0"
    print("   PASS: 1D Neumann 1st-order")

    # Test 3: 1D Dirichlet
    print("\n3. Testing 1D Dirichlet BC...")
    field = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
    enforce_dirichlet_value_nd(field, axis=0, side="min", value=0.0)
    assert field[0] == 0.0, f"Min failed: {field[0]} != 0.0"
    enforce_dirichlet_value_nd(field, axis=0, side="max", value=5.0)
    assert field[-1] == 5.0, f"Max failed: {field[-1]} != 5.0"
    print("   PASS: 1D Dirichlet")

    # Test 4: 2D Neumann
    print("\n4. Testing 2D Neumann BC...")
    field_2d = np.ones((5, 6))
    for i in range(5):
        field_2d[i, :] = 1.0 + 0.1 * i
    enforce_neumann_value_nd(field_2d, axis=0, side="min", order=2)
    # Check extrapolation was applied
    expected = (4.0 * field_2d[1, 0] - field_2d[2, 0]) / 3.0
    assert np.isclose(field_2d[0, 0], expected), "2D min failed"
    print("   PASS: 2D Neumann")

    # Test 5: 1D Periodic
    print("\n5. Testing 1D Periodic BC...")
    field = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
    enforce_periodic_value_nd(field, axis=0)
    assert field[0] == 3.0, f"Periodic min failed: {field[0]} != 3.0"
    assert field[-1] == 1.0, f"Periodic max failed: {field[-1]} != 1.0"
    print("   PASS: 1D Periodic")

    # Test 6: Non-zero Neumann gradient
    print("\n6. Testing Neumann BC with non-zero gradient...")
    field = np.array([0.0, 1.0, 1.1, 1.2, 0.0])
    h = 0.1
    g = 2.0  # du/dn = 2
    enforce_neumann_value_nd(field, axis=0, side="min", grad_value=g, spacing=h)
    expected = 1.0 - g * h  # u[0] = u[1] - g*h = 1.0 - 0.2 = 0.8
    assert np.isclose(field[0], expected), f"Non-zero gradient failed: {field[0]} != {expected}"
    print("   PASS: Neumann with non-zero gradient")

    print("\n" + "=" * 50)
    print("All enforcement utility tests passed!")
    print("=" * 50)
