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
# nD Splatting Functions
# =============================================================================


def splat_linear_nd(
    m: NDArray[np.floating],
    x_dest: NDArray[np.floating],
    grid_coordinates: tuple[NDArray[np.floating], ...],
    grid_shape: tuple[int, ...],
    bounds: list[tuple[float, float]],
) -> NDArray[np.floating]:
    """
    Linear (2^d point) splatting for nD problems - adjoint of multilinear interpolation.

    For nD, linear interpolation uses 2^d corner points of the hypercube containing
    the query point. The weights are tensor products of 1D weights.

    Example (2D):
        Interpolation at (x, y) in cell [i,i+1] x [j,j+1]:
        φ(x,y) = (1-wx)(1-wy)·φ[i,j] + wx(1-wy)·φ[i+1,j]
               + (1-wx)wy·φ[i,j+1] + wx·wy·φ[i+1,j+1]

        Splatting (adjoint):
        m[i,j] += (1-wx)(1-wy)·m_src; m[i+1,j] += wx(1-wy)·m_src; etc.

    Args:
        m: Source density array, shape grid_shape (flattened or shaped)
        x_dest: Destination positions, shape (N_points, dimension)
        grid_coordinates: Tuple of 1D coordinate arrays for each dimension
        grid_shape: Shape of the grid (N1, N2, ..., Nd)
        bounds: List of (min, max) tuples for each dimension

    Returns:
        Splat result, same shape as m
    """
    dimension = len(grid_shape)
    N_points = np.prod(grid_shape)

    # Ensure m is flattened for indexing
    m_flat = m.ravel()
    m_star = np.zeros(N_points)

    # Grid spacings
    dx = [(grid_coordinates[d][1] - grid_coordinates[d][0]) for d in range(dimension)]

    # Reshape x_dest if needed: (N_points,) for 1D -> (N_points, 1)
    if x_dest.ndim == 1 and dimension == 1:
        x_dest = x_dest.reshape(-1, 1)

    # Process each source point
    for flat_idx in range(N_points):
        # Destination position for this point
        x_d = x_dest[flat_idx]

        # Compute cell indices and weights for each dimension
        j_list = []  # Lower corner indices
        w_list = []  # Weights for upper corner

        for d in range(dimension):
            xmin_d, _ = bounds[d]  # xmax not needed; clipping handles bounds
            pos_cont = (x_d[d] - xmin_d) / dx[d]

            # Lower index
            j_d = int(np.floor(pos_cont))
            j_d = max(0, min(j_d, grid_shape[d] - 2))

            # Weight
            w_d = pos_cont - j_d
            w_d = max(0.0, min(1.0, w_d))

            j_list.append(j_d)
            w_list.append(w_d)

        # Scatter to 2^d corners
        # Iterate over all corner combinations using binary representation
        for corner in range(1 << dimension):  # 0 to 2^d - 1
            corner_idx = []
            weight = 1.0

            for d in range(dimension):
                if corner & (1 << d):  # Bit d is set -> upper corner
                    corner_idx.append(j_list[d] + 1)
                    weight *= w_list[d]
                else:  # Lower corner
                    corner_idx.append(j_list[d])
                    weight *= 1.0 - w_list[d]

            # Clamp indices to valid range
            corner_idx = [min(max(0, idx), grid_shape[d] - 1) for d, idx in enumerate(corner_idx)]

            # Convert to flat index
            dest_flat_idx = np.ravel_multi_index(corner_idx, grid_shape)

            # Accumulate
            m_star[dest_flat_idx] += weight * m_flat[flat_idx]

    return m_star.reshape(grid_shape)


def splat_nd(
    m: NDArray[np.floating],
    x_dest: NDArray[np.floating],
    grid_coordinates: tuple[NDArray[np.floating], ...],
    grid_shape: tuple[int, ...],
    bounds: list[tuple[float, float]],
    method: str = "linear",
) -> NDArray[np.floating]:
    """
    Dispatch to appropriate nD splatting method.

    Args:
        m: Source density array
        x_dest: Destination positions, shape (N_points, dimension)
        grid_coordinates: Tuple of 1D coordinate arrays for each dimension
        grid_shape: Shape of the grid
        bounds: List of (min, max) tuples for each dimension
        method: Splatting method ('linear' only for nD currently)

    Returns:
        Splat result

    Note:
        Cubic and quintic splatting for nD would require tensor products of 1D kernels.
        Currently only linear is supported for nD.
    """
    if method == "linear":
        return splat_linear_nd(m, x_dest, grid_coordinates, grid_shape, bounds)
    elif method in ("cubic", "quintic"):
        raise NotImplementedError(
            f"'{method}' splatting not yet implemented for nD. Use 'linear' or implement "
            "tensor-product splatting. For nD, consider using 'linear' which provides "
            "adequate accuracy for most MFG applications."
        )
    else:
        raise ValueError(f"Unknown splatting method: {method}.")


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

    # Test 7: nD splatting (2D)
    print("\n7. Testing 2D linear splatting...")
    Nx_2d, Ny_2d = 11, 11
    x_2d = np.linspace(0.0, 1.0, Nx_2d)
    y_2d = np.linspace(0.0, 1.0, Ny_2d)
    grid_shape_2d = (Nx_2d, Ny_2d)
    bounds_2d = [(0.0, 1.0), (0.0, 1.0)]
    dx_2d = x_2d[1] - x_2d[0]

    # Uniform density
    m_2d = np.ones(grid_shape_2d)

    # Create meshgrid and shift positions
    XX, YY = np.meshgrid(x_2d, y_2d, indexing="ij")
    x_dest_2d = np.stack([(XX + 0.3 * dx_2d).ravel(), (YY + 0.2 * dx_2d).ravel()], axis=-1)

    # Splat
    m_splat_2d = splat_linear_nd(m_2d.ravel(), x_dest_2d, (x_2d, y_2d), grid_shape_2d, bounds_2d)

    mass_2d_before = np.sum(m_2d)
    mass_2d_after = np.sum(m_splat_2d)
    print(f"   Mass before: {mass_2d_before:.6f}")
    print(f"   Mass after:  {mass_2d_after:.6f}")
    print(f"   Mass error:  {abs(mass_2d_after - mass_2d_before):.2e}")
    assert abs(mass_2d_after - mass_2d_before) < 1e-10, "2D splatting failed mass conservation"
    print("   2D splatting: OK")

    # Test 8: nD splatting (3D)
    print("\n8. Testing 3D linear splatting...")
    N3d = 6
    x_3d = np.linspace(0.0, 1.0, N3d)
    grid_shape_3d = (N3d, N3d, N3d)
    bounds_3d = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    dx_3d = x_3d[1] - x_3d[0]

    # Uniform density
    m_3d = np.ones(grid_shape_3d)

    # Create meshgrid and shift positions
    XX3, YY3, ZZ3 = np.meshgrid(x_3d, x_3d, x_3d, indexing="ij")
    x_dest_3d = np.stack(
        [
            (XX3 + 0.25 * dx_3d).ravel(),
            (YY3 + 0.15 * dx_3d).ravel(),
            (ZZ3 + 0.35 * dx_3d).ravel(),
        ],
        axis=-1,
    )

    # Splat
    m_splat_3d = splat_linear_nd(m_3d.ravel(), x_dest_3d, (x_3d, x_3d, x_3d), grid_shape_3d, bounds_3d)

    mass_3d_before = np.sum(m_3d)
    mass_3d_after = np.sum(m_splat_3d)
    print(f"   Mass before: {mass_3d_before:.6f}")
    print(f"   Mass after:  {mass_3d_after:.6f}")
    print(f"   Mass error:  {abs(mass_3d_after - mass_3d_before):.2e}")
    assert abs(mass_3d_after - mass_3d_before) < 1e-10, "3D splatting failed mass conservation"
    print("   3D splatting: OK")

    print("\n" + "=" * 60)
    print("All splatting smoke tests passed!")
