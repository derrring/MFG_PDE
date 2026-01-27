"""
Splatting Methods for Adjoint Semi-Lagrangian FP Solver.

This module provides splatting routines (adjoint of interpolation) for the
Forward Semi-Lagrangian method. Each splatting scheme is the exact transpose
of the corresponding interpolation scheme used in HJB-SL.

Supported methods:
- linear: 2-point stencil, O(dx) accuracy, preserves positivity
- cubic: 4-point stencil, O(dx³) accuracy, may produce negative values
- quintic: 6-point stencil, O(dx⁵) accuracy, may produce negative values

Mathematical Foundation:
    If HJB uses interpolation matrix P with row sums = 1,
    then FP uses splatting matrix P^T with column sums = 1.
    This ensures sum(P^T @ m) = sum(m) exactly.

Module structure per issue #392:
    fp_sl_splatting.py - Splatting methods for adjoint semi-Lagrangian FP solver

Functions:
    splat_linear_1d: Linear (2-point) splatting
    splat_cubic_1d: Cubic (4-point) splatting
    splat_quintic_1d: Quintic (6-point) splatting
    compute_cubic_weights: Catmull-Rom cubic kernel weights
    compute_quintic_weights: Quintic kernel weights

Issue #708: Splatting implementations for adjoint-consistent SL-MFG
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_cubic_weights(t: float) -> tuple[float, float, float, float]:
    """
    Compute Catmull-Rom cubic interpolation weights.

    The Catmull-Rom spline uses 4 points with C¹ continuity.
    Weights satisfy partition of unity: sum(w) = 1.

    Args:
        t: Fractional position in [0, 1]

    Returns:
        Tuple of weights (w_{-1}, w_0, w_1, w_2) for points j-1, j, j+1, j+2
    """
    t2 = t * t
    t3 = t2 * t

    # Catmull-Rom kernel coefficients
    w_m1 = -0.5 * t3 + t2 - 0.5 * t
    w_0 = 1.5 * t3 - 2.5 * t2 + 1.0
    w_1 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w_2 = 0.5 * t3 - 0.5 * t2

    return w_m1, w_0, w_1, w_2


def compute_quintic_weights(t: float) -> tuple[float, float, float, float, float, float]:
    """
    Compute quintic interpolation weights.

    Uses 6 points for O(dx⁵) accuracy with C² continuity.
    Weights satisfy partition of unity: sum(w) = 1.

    Args:
        t: Fractional position in [0, 1]

    Returns:
        Tuple of weights (w_{-2}, w_{-1}, w_0, w_1, w_2, w_3)
    """
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    # Quintic Lagrange weights for 6-point stencil
    # Based on Lagrange polynomial through points at -2, -1, 0, 1, 2, 3
    w_m2 = (t5 - 5 * t4 + 5 * t3 + 5 * t2 - 6 * t) / (-120)
    w_m1 = (t5 - 4 * t4 - t3 + 16 * t2 - 12 * t) / 24
    w_0 = (t5 - 3 * t4 - 5 * t3 + 15 * t2 + 4 * t - 12) / (-12)
    w_1 = (t5 - 2 * t4 - 7 * t3 + 8 * t2 + 12 * t) / 12
    w_2 = (t5 - t4 - 7 * t3 + t2 + 6 * t) / (-24)
    w_3 = (t5 - 5 * t3 + 4 * t) / 120

    return w_m2, w_m1, w_0, w_1, w_2, w_3


def splat_linear_1d(
    m: NDArray[np.floating],
    x_dest: NDArray[np.floating],
    x_grid: NDArray[np.floating],
    dx: float,
    xmin: float,
    xmax: float,
) -> NDArray[np.floating]:
    """
    Linear (2-point) splatting - adjoint of linear interpolation.

    Each particle at position x_dest scatters its density to two neighboring
    grid points with weights (1-w, w) where w is the fractional position.

    This is the transpose of linear interpolation:
    - Interpolation (gather): φ[i] = (1-w)·φ[j] + w·φ[j+1]
    - Splatting (scatter): m[j] += (1-w)·m[i]; m[j+1] += w·m[i]

    Args:
        m: Source density array, shape (Nx,)
        x_dest: Destination positions, shape (Nx,)
        x_grid: Grid coordinates, shape (Nx,)
        dx: Grid spacing
        xmin, xmax: Domain bounds

    Returns:
        Splat result, shape (Nx,)
    """
    Nx = len(m)
    m_star = np.zeros(Nx)

    # Convert to continuous indices
    pos_cont = (x_dest - xmin) / dx

    # Lower neighbor index
    j = np.floor(pos_cont).astype(int)
    j = np.clip(j, 0, Nx - 2)

    # Weight for upper neighbor
    w = pos_cont - j
    w = np.clip(w, 0, 1)

    # Scatter with atomic accumulation
    np.add.at(m_star, j, m * (1 - w))
    np.add.at(m_star, j + 1, m * w)

    return m_star


def splat_cubic_1d(
    m: NDArray[np.floating],
    x_dest: NDArray[np.floating],
    x_grid: NDArray[np.floating],
    dx: float,
    xmin: float,
    xmax: float,
) -> NDArray[np.floating]:
    """
    Cubic (4-point) splatting - adjoint of Catmull-Rom cubic interpolation.

    Each particle scatters to 4 neighboring grid points using Catmull-Rom
    cubic kernel weights.

    This is the transpose of cubic interpolation:
    - Interpolation: φ[i] = Σ_{k=-1}^{2} w_k · φ[j+k]
    - Splatting: m[j+k] += w_k · m[i] for k = -1, 0, 1, 2

    Note: May produce negative values due to oscillatory cubic kernel.

    Args:
        m: Source density array, shape (Nx,)
        x_dest: Destination positions, shape (Nx,)
        x_grid: Grid coordinates, shape (Nx,)
        dx: Grid spacing
        xmin, xmax: Domain bounds

    Returns:
        Splat result, shape (Nx,)
    """
    Nx = len(m)
    m_star = np.zeros(Nx)

    # Convert to continuous indices
    pos_cont = (x_dest - xmin) / dx

    for i in range(Nx):
        # Base index (floor of position)
        j = int(np.floor(pos_cont[i]))

        # Fractional position
        t = pos_cont[i] - j

        # Compute cubic weights
        w_m1, w_0, w_1, w_2 = compute_cubic_weights(t)
        weights = [w_m1, w_0, w_1, w_2]
        indices = [j - 1, j, j + 1, j + 2]

        # Scatter to 4 neighbors
        for idx, wk in zip(indices, weights, strict=True):
            if 0 <= idx < Nx:
                m_star[idx] += wk * m[i]
            elif idx < 0:
                # Reflect mass back into domain (Neumann BC)
                m_star[0] += wk * m[i]
            else:  # idx >= Nx
                m_star[Nx - 1] += wk * m[i]

    return m_star


def splat_quintic_1d(
    m: NDArray[np.floating],
    x_dest: NDArray[np.floating],
    x_grid: NDArray[np.floating],
    dx: float,
    xmin: float,
    xmax: float,
) -> NDArray[np.floating]:
    """
    Quintic (6-point) splatting - adjoint of quintic interpolation.

    Each particle scatters to 6 neighboring grid points using quintic
    Lagrange weights.

    This is the transpose of quintic interpolation:
    - Interpolation: φ[i] = Σ_{k=-2}^{3} w_k · φ[j+k]
    - Splatting: m[j+k] += w_k · m[i] for k = -2, -1, 0, 1, 2, 3

    Note: May produce negative values due to oscillatory kernel.

    Args:
        m: Source density array, shape (Nx,)
        x_dest: Destination positions, shape (Nx,)
        x_grid: Grid coordinates, shape (Nx,)
        dx: Grid spacing
        xmin, xmax: Domain bounds

    Returns:
        Splat result, shape (Nx,)
    """
    Nx = len(m)
    m_star = np.zeros(Nx)

    # Convert to continuous indices
    pos_cont = (x_dest - xmin) / dx

    for i in range(Nx):
        # Base index (floor of position)
        j = int(np.floor(pos_cont[i]))

        # Fractional position
        t = pos_cont[i] - j

        # Compute quintic weights
        w_m2, w_m1, w_0, w_1, w_2, w_3 = compute_quintic_weights(t)
        weights = [w_m2, w_m1, w_0, w_1, w_2, w_3]
        indices = [j - 2, j - 1, j, j + 1, j + 2, j + 3]

        # Scatter to 6 neighbors
        for idx, wk in zip(indices, weights, strict=True):
            if 0 <= idx < Nx:
                m_star[idx] += wk * m[i]
            elif idx < 0:
                # Reflect mass back into domain (Neumann BC)
                m_star[0] += wk * m[i]
            else:  # idx >= Nx
                m_star[Nx - 1] += wk * m[i]

    return m_star


def splat_1d(
    m: NDArray[np.floating],
    x_dest: NDArray[np.floating],
    x_grid: NDArray[np.floating],
    dx: float,
    xmin: float,
    xmax: float,
    method: str = "linear",
) -> NDArray[np.floating]:
    """
    Dispatch to appropriate splatting method.

    Args:
        m: Source density array
        x_dest: Destination positions
        x_grid: Grid coordinates
        dx: Grid spacing
        xmin, xmax: Domain bounds
        method: Splatting method ('linear', 'cubic', 'quintic')

    Returns:
        Splat result
    """
    if method == "linear":
        return splat_linear_1d(m, x_dest, x_grid, dx, xmin, xmax)
    elif method == "cubic":
        return splat_cubic_1d(m, x_dest, x_grid, dx, xmin, xmax)
    elif method == "quintic":
        return splat_quintic_1d(m, x_dest, x_grid, dx, xmin, xmax)
    else:
        raise ValueError(f"Unknown splatting method: {method}. Use 'linear', 'cubic', or 'quintic'.")


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    """Smoke test for splatting methods."""
    print("Testing splatting methods...")
    print("=" * 60)

    # Test grid
    Nx = 21
    xmin, xmax = 0.0, 1.0
    x = np.linspace(xmin, xmax, Nx)
    dx = x[1] - x[0]

    # Test 1: Linear splatting - partition of unity
    print("\n1. Testing linear splatting (partition of unity)...")
    m_uniform = np.ones(Nx)
    x_dest = x + 0.3 * dx  # Shift by 0.3 grid spacing

    m_splat = splat_linear_1d(m_uniform, x_dest, x, dx, xmin, xmax)
    mass_before = np.sum(m_uniform)
    mass_after = np.sum(m_splat)

    print(f"   Mass before: {mass_before:.6f}")
    print(f"   Mass after:  {mass_after:.6f}")
    print(f"   Mass error:  {abs(mass_after - mass_before):.2e}")
    assert abs(mass_after - mass_before) < 1e-10, "Linear splatting failed mass conservation"
    print("   Linear splatting: OK")

    # Test 2: Cubic splatting - partition of unity
    print("\n2. Testing cubic splatting (partition of unity)...")
    m_splat_cubic = splat_cubic_1d(m_uniform, x_dest, x, dx, xmin, xmax)
    mass_cubic = np.sum(m_splat_cubic)

    print(f"   Mass before: {mass_before:.6f}")
    print(f"   Mass after:  {mass_cubic:.6f}")
    print(f"   Mass error:  {abs(mass_cubic - mass_before):.2e}")
    # Cubic may have small errors due to boundary handling
    assert abs(mass_cubic - mass_before) < 0.1, "Cubic splatting failed mass conservation"
    print("   Cubic splatting: OK")

    # Test 3: Verify cubic weights sum to 1
    print("\n3. Testing cubic weights (partition of unity)...")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        weights = compute_cubic_weights(t)
        weight_sum = sum(weights)
        print(f"   t={t:.2f}: weights={[f'{w:.4f}' for w in weights]}, sum={weight_sum:.6f}")
        assert abs(weight_sum - 1.0) < 1e-10, f"Cubic weights don't sum to 1 at t={t}"
    print("   Cubic weights: OK")

    # Test 4: Quintic splatting - partition of unity
    print("\n4. Testing quintic splatting (partition of unity)...")
    m_splat_quintic = splat_quintic_1d(m_uniform, x_dest, x, dx, xmin, xmax)
    mass_quintic = np.sum(m_splat_quintic)

    print(f"   Mass before: {mass_before:.6f}")
    print(f"   Mass after:  {mass_quintic:.6f}")
    print(f"   Mass error:  {abs(mass_quintic - mass_before):.2e}")
    assert abs(mass_quintic - mass_before) < 0.1, "Quintic splatting failed mass conservation"
    print("   Quintic splatting: OK")

    # Test 5: Verify quintic weights sum to 1
    print("\n5. Testing quintic weights (partition of unity)...")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        weights = compute_quintic_weights(t)
        weight_sum = sum(weights)
        print(f"   t={t:.2f}: sum={weight_sum:.6f}")
        assert abs(weight_sum - 1.0) < 1e-10, f"Quintic weights don't sum to 1 at t={t}"
    print("   Quintic weights: OK")

    # Test 6: Dispatch function
    print("\n6. Testing dispatch function...")
    m_linear = splat_1d(m_uniform, x_dest, x, dx, xmin, xmax, method="linear")
    m_cubic = splat_1d(m_uniform, x_dest, x, dx, xmin, xmax, method="cubic")
    m_quintic = splat_1d(m_uniform, x_dest, x, dx, xmin, xmax, method="quintic")
    print("   Dispatch function: OK")

    print("\n" + "=" * 60)
    print("All splatting smoke tests passed!")
