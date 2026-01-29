"""
Periodic boundary condition utilities (Issue #711).

This module provides utilities for PERIODIC boundary conditions, parallel to
enforcement.py which handles DIRICHLET/NEUMANN/ROBIN.

Architecture:
    BCType.DIRICHLET/NEUMANN/ROBIN → enforcement.py (value-based)
    BCType.PERIODIC                → periodic.py (topology-based)

The key difference is that periodic BCs work at the TOPOLOGY level
(index wrapping, ghost point augmentation) rather than computing
ghost cell VALUES like other BC types.

Functions:
    wrap_positions: Wrap particle positions to fundamental domain
    create_periodic_ghost_points: Augment point cloud for KD-tree search

Reference:
    - Issue #711: Periodic BC support for GFDM
    - geometry/protocols/topology.py: SupportsPeriodic protocol

Created: 2026-01-29 (Issue #711 refactor)
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def wrap_positions(
    positions: NDArray[np.floating],
    bounds: list[tuple[float, float]] | NDArray[np.floating],
    periodic_dims: tuple[int, ...] | None = None,
) -> NDArray[np.floating]:
    """
    Wrap positions around domain boundaries (periodic BC, n-D).

    For periodic dimension i with period L_i = xmax_i - xmin_i:
        x_wrapped = xmin + (x - xmin) mod L

    Args:
        positions: Particle positions, shape (N, d) or (d,) for single point
        bounds: Domain bounds as [(xmin, xmax), (ymin, ymax), ...] or (d, 2) array
        periodic_dims: Dimensions to wrap. If None, all dimensions are wrapped.

    Returns:
        Wrapped positions in [xmin, xmax), same shape as input

    Examples:
        >>> positions = np.array([[1.5, 0.5], [-0.3, 0.5]])
        >>> bounds = [(0, 1), (0, 1)]
        >>> wrapped = wrap_positions(positions, bounds)
        >>> # (1.5, 0.5) -> (0.5, 0.5), (-0.3, 0.5) -> (0.7, 0.5)

        >>> # Cylinder: periodic in x only
        >>> wrapped = wrap_positions(positions, bounds, periodic_dims=(0,))

    Note:
        This is the canonical utility for particle position wrapping.
        Used by MeshfreeApplicator._apply_periodic_bc() and others.
    """
    positions = np.atleast_2d(positions)
    was_1d = positions.shape[0] == 1 and len(positions.shape) == 2
    result = positions.copy()

    bounds = np.asarray(bounds)
    if bounds.ndim == 1:
        bounds = bounds.reshape(1, 2)

    ndim = result.shape[1]
    if bounds.shape[0] != ndim:
        raise ValueError(f"Bounds dimension {bounds.shape[0]} != positions dimension {ndim}")

    # Default: all dimensions periodic
    if periodic_dims is None:
        periodic_dims = tuple(range(ndim))

    for d in periodic_dims:
        xmin, xmax = bounds[d, 0], bounds[d, 1]
        Lx = xmax - xmin
        if Lx > 1e-14:
            result[:, d] = xmin + ((result[:, d] - xmin) % Lx)

    if was_1d:
        return result[0]
    return result


def create_periodic_ghost_points(
    points: NDArray[np.floating],
    bounds: list[tuple[float, float]] | NDArray[np.floating],
    periodic_dims: tuple[int, ...] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.int64]]:
    """
    Create augmented point cloud with ghost copies for periodic neighbor search.

    For meshfree methods (GFDM, RBF) on periodic domains, KD-tree neighbor
    search requires ghost points near domain boundaries to find periodic
    neighbors correctly.

    For d-dimensional domain with |P| periodic dimensions, creates 3^|P|
    copies of the point cloud shifted by +/- L in each periodic direction.

    Args:
        points: Original points, shape (n_points, dimension)
        bounds: Domain bounds as [(xmin, xmax), ...] or (d, 2) array
        periodic_dims: Dimensions with periodic topology. If None, all dims periodic.

    Returns:
        Tuple of (augmented_points, original_indices):
            - augmented_points: Shape (n_augmented, dimension)
            - original_indices: Maps augmented index -> original point index

    Examples:
        >>> # 2D torus - creates 9 copies (3^2)
        >>> points = np.random.rand(100, 2)
        >>> bounds = [(0, 1), (0, 1)]
        >>> aug_pts, orig_idx = create_periodic_ghost_points(points, bounds)
        >>> # aug_pts.shape == (900, 2), orig_idx.shape == (900,)

        >>> # Cylinder - periodic in x only (3 copies)
        >>> aug_pts, orig_idx = create_periodic_ghost_points(
        ...     points, bounds, periodic_dims=(0,)
        ... )
        >>> # aug_pts.shape == (300, 2)

    Note:
        Issue #711: Canonical utility for periodic meshfree methods.
        Used by TaylorOperator for GFDM on periodic domains.
    """
    points = np.atleast_2d(points)
    n_points, ndim = points.shape

    bounds = np.asarray(bounds)
    if bounds.ndim == 1:
        bounds = bounds.reshape(1, 2)

    # Default: all dimensions periodic
    if periodic_dims is None:
        periodic_dims = tuple(range(ndim))

    if not periodic_dims:
        # No periodicity
        return points, np.arange(n_points, dtype=np.int64)

    # Generate shift combinations for periodic dimensions
    shift_options = []
    for d in range(ndim):
        if d in periodic_dims:
            shift_options.append([-1, 0, 1])
        else:
            shift_options.append([0])

    shifts = list(itertools.product(*shift_options))

    # Period lengths
    period_lengths = np.array([bounds[d, 1] - bounds[d, 0] for d in range(ndim)])

    augmented_list = []
    index_list = []

    for shift in shifts:
        shift_vec = np.array(shift) * period_lengths
        shifted_points = points + shift_vec
        augmented_list.append(shifted_points)
        index_list.append(np.arange(n_points, dtype=np.int64))

    return np.vstack(augmented_list), np.concatenate(index_list)


__all__ = [
    "wrap_positions",
    "create_periodic_ghost_points",
]


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    """Quick smoke test for periodic boundary utilities."""
    print("Testing periodic boundary utilities...")

    bounds_2d = [(0, 1), (0, 1)]

    # Test wrap_positions
    print("\n1. wrap_positions:")
    positions_wrap = np.array([[1.5, 0.5], [-0.3, 0.5]])
    wrapped = wrap_positions(positions_wrap, bounds_2d)
    expected_wrap = np.array([[0.5, 0.5], [0.7, 0.5]])
    assert np.allclose(wrapped, expected_wrap), f"wrap failed: {wrapped}"
    print("   Full wrap: passed")

    # Test partial periodic (cylinder)
    wrapped_partial = wrap_positions(positions_wrap, bounds_2d, periodic_dims=(0,))
    expected_partial = np.array([[0.5, 0.5], [0.7, 0.5]])  # y unchanged
    assert np.allclose(wrapped_partial, expected_partial)
    print("   Partial wrap (cylinder): passed")

    # Test single point
    single = np.array([1.5, 0.5])
    single_wrapped = wrap_positions(single, bounds_2d)
    assert single_wrapped.shape == (2,)
    assert np.allclose(single_wrapped, [0.5, 0.5])
    print("   Single point: passed")

    # Test create_periodic_ghost_points
    print("\n2. create_periodic_ghost_points:")
    points = np.random.rand(10, 2)

    # Full torus (3^2 = 9 copies)
    aug, idx = create_periodic_ghost_points(points, bounds_2d)
    assert aug.shape == (90, 2), f"Expected (90, 2), got {aug.shape}"
    assert idx.shape == (90,)
    print(f"   2D torus: {len(points)} -> {len(aug)} points (9x)")

    # Cylinder (3^1 = 3 copies)
    aug_cyl, idx_cyl = create_periodic_ghost_points(points, bounds_2d, periodic_dims=(0,))
    assert aug_cyl.shape == (30, 2), f"Expected (30, 2), got {aug_cyl.shape}"
    print(f"   Cylinder: {len(points)} -> {len(aug_cyl)} points (3x)")

    # Non-periodic (1 copy)
    aug_none, idx_none = create_periodic_ghost_points(points, bounds_2d, periodic_dims=())
    assert aug_none.shape == (10, 2)
    print(f"   Non-periodic: {len(points)} -> {len(aug_none)} points (1x)")

    print("\nAll periodic utilities tests passed!")
