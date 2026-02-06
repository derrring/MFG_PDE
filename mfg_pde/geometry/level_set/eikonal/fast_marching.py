"""
Fast Marching Method (FMM) for the Eikonal Equation.

The Fast Marching Method solves the Eikonal equation |grad T| = 1/F using a
heap-based algorithm with O(N log N) complexity. It processes grid points
in order of increasing arrival time, ensuring causality (information flows
from smaller to larger T).

Algorithm Overview:
    1. Initialize: Frozen points (known T), Trial points (adjacent to Frozen)
    2. Loop until all points processed:
       a. Extract minimum T from Trial heap
       b. Move point from Trial to Frozen
       c. Update neighbors: recompute T, add to Trial if not Frozen
    3. Return T array

Point States:
    - FROZEN: T is finalized (will not change)
    - TRIAL: T is tentative (in the heap, may be updated)
    - FAR: T is unknown (not yet reached by the front)

Complexity Analysis:
    - Each point enters/exits heap once: O(N) heap operations
    - Each heap operation: O(log N)
    - Total: O(N log N)

Compared to PDE-based reinitialization O(N * iterations), FMM is dramatically
faster for large grids.

References:
- Sethian (1996): A fast marching level set method, PNAS
- Sethian (1999): Level Set Methods and Fast Marching Methods (book)
- Adalsteinsson & Sethian (1995): A fast level set method for propagating interfaces

Created: 2026-02-06 (Issue #664)
"""

from __future__ import annotations

import heapq
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from mfg_pde.geometry.level_set.eikonal.godunov_update import godunov_update_nd
from mfg_pde.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.geometry.grids.tensor_grid import TensorProductGrid

logger = get_logger(__name__)


class PointState(IntEnum):
    """State of a grid point in FMM."""

    FAR = 0  # Unknown T
    TRIAL = 1  # Tentative T (in heap)
    FROZEN = 2  # Final T


class FastMarchingMethod:
    """
    Fast Marching Method for solving the Eikonal equation.

    Solves |grad T| = 1/F where T is arrival time (or distance) and F is speed.

    Attributes:
        geometry: TensorProductGrid providing grid structure and spacing.

    Example:
        >>> from mfg_pde.geometry.grids import TensorProductGrid
        >>> from mfg_pde.geometry.boundary import no_flux_bc
        >>> grid = TensorProductGrid(
        ...     bounds=[(0, 1), (0, 1)], Nx=[100, 100],
        ...     boundary_conditions=no_flux_bc(dimension=2)
        ... )
        >>> fmm = FastMarchingMethod(grid)
        >>> # Point source: T(x) = |x - x0|
        >>> phi = fmm.compute_signed_distance(phi_initial)
    """

    def __init__(self, geometry: TensorProductGrid) -> None:
        """
        Initialize FMM solver.

        Args:
            geometry: Grid providing spacing and structure.
        """
        self.geometry = geometry
        self._spacing = tuple(geometry.spacing)
        self._shape = tuple(geometry.Nx_points)
        self._ndim = len(self._shape)

    def solve(
        self,
        speed: NDArray[np.float64] | float,
        frozen_mask: NDArray[np.bool_],
        frozen_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Solve the Eikonal equation |grad T| = 1/F.

        Args:
            speed: Speed function F(x) > 0. Scalar or array matching grid shape.
            frozen_mask: Boolean mask of points with known values (boundary conditions).
            frozen_values: Values at frozen points.

        Returns:
            Solution T with |grad T| = 1/F.
        """
        # Convert speed to array if scalar
        if np.isscalar(speed):
            speed_array = np.full(self._shape, float(speed), dtype=np.float64)
        else:
            speed_array = np.asarray(speed, dtype=np.float64)

        # Initialize arrays
        T = np.full(self._shape, np.inf, dtype=np.float64)
        T[frozen_mask] = frozen_values[frozen_mask]

        state = np.full(self._shape, PointState.FAR, dtype=np.int8)
        state[frozen_mask] = PointState.FROZEN

        # Initialize trial heap with neighbors of frozen points
        # Heap entries: (T_value, flat_index) for O(1) comparison
        trial_heap: list[tuple[float, int]] = []

        # Find all neighbors of frozen points
        frozen_indices = np.argwhere(frozen_mask)
        for idx_tuple in frozen_indices:
            idx = tuple(idx_tuple)
            for neighbor in self._get_neighbors(idx):
                if state[neighbor] == PointState.FAR:
                    # Compute initial T estimate
                    T_new = self._compute_update(T, neighbor, speed_array[neighbor])
                    T[neighbor] = T_new
                    state[neighbor] = PointState.TRIAL
                    flat_idx = np.ravel_multi_index(neighbor, self._shape)
                    heapq.heappush(trial_heap, (T_new, flat_idx))

        # Main loop: process trial points in order of increasing T
        while trial_heap:
            T_current, flat_idx = heapq.heappop(trial_heap)
            idx = np.unravel_index(flat_idx, self._shape)

            # Skip if already frozen (heap may contain stale entries)
            if state[idx] == PointState.FROZEN:
                continue

            # Freeze this point
            state[idx] = PointState.FROZEN
            T[idx] = T_current

            # Update neighbors
            for neighbor in self._get_neighbors(idx):
                if state[neighbor] != PointState.FROZEN:
                    T_new = self._compute_update(T, neighbor, speed_array[neighbor])
                    if T_new < T[neighbor]:
                        T[neighbor] = T_new
                        flat_neighbor = np.ravel_multi_index(neighbor, self._shape)
                        heapq.heappush(trial_heap, (T_new, flat_neighbor))
                        state[neighbor] = PointState.TRIAL

        return T

    def compute_signed_distance(
        self,
        phi_initial: NDArray[np.float64],
        subcell_accuracy: bool = True,
    ) -> NDArray[np.float64]:
        """
        Compute signed distance function from initial level set.

        The zero level set of phi_initial is preserved, and the result
        satisfies |grad phi| = 1.

        Args:
            phi_initial: Initial level set. Zero level set is the interface.
            subcell_accuracy: If True, use linear interpolation to locate
                             interface with subcell precision.

        Returns:
            Signed distance function with |grad phi| = 1.
        """
        phi = phi_initial.copy()

        # Identify interface cells (sign change between neighbors)
        frozen_mask, frozen_values = self._initialize_interface(phi, subcell_accuracy=subcell_accuracy)

        # Handle exact zeros: points with phi = 0 exactly should be frozen at 0
        exact_zeros = np.abs(phi_initial) < 1e-14
        frozen_mask = frozen_mask | exact_zeros
        frozen_values[exact_zeros] = 0.0

        # Solve for positive region (phi >= 0, includes zeros)
        positive_mask = phi_initial >= 0
        T_positive = self._solve_one_sided(frozen_mask, frozen_values, positive_mask)

        # Solve for negative region (phi <= 0, includes zeros)
        negative_mask = phi_initial <= 0
        T_negative = self._solve_one_sided(frozen_mask, frozen_values, negative_mask)

        # Combine: positive distance where phi >= 0, negative where phi < 0
        result = np.where(phi_initial >= 0, T_positive, -T_negative)

        return result

    def _solve_one_sided(
        self,
        frozen_mask: NDArray[np.bool_],
        frozen_values: NDArray[np.float64],
        region_mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Solve FMM for one side of the interface."""
        # Speed = 1 for SDF computation
        speed = 1.0

        T = np.full(self._shape, np.inf, dtype=np.float64)
        state = np.full(self._shape, PointState.FAR, dtype=np.int8)

        # Set frozen points
        T[frozen_mask] = np.abs(frozen_values[frozen_mask])
        state[frozen_mask] = PointState.FROZEN

        # Initialize trial heap with neighbors of frozen points that are in the region
        trial_heap: list[tuple[float, int]] = []

        frozen_indices = np.argwhere(frozen_mask)
        for idx_tuple in frozen_indices:
            idx = tuple(idx_tuple)
            for neighbor in self._get_neighbors(idx):
                # Include neighbor if in original region (not just expanded)
                if state[neighbor] == PointState.FAR and region_mask[neighbor]:
                    T_new = self._compute_update(T, neighbor, speed)
                    T[neighbor] = T_new
                    state[neighbor] = PointState.TRIAL
                    flat_idx = np.ravel_multi_index(neighbor, self._shape)
                    heapq.heappush(trial_heap, (T_new, flat_idx))

        # Main loop
        while trial_heap:
            T_current, flat_idx = heapq.heappop(trial_heap)
            idx = np.unravel_index(flat_idx, self._shape)

            if state[idx] == PointState.FROZEN:
                continue

            state[idx] = PointState.FROZEN
            T[idx] = T_current

            for neighbor in self._get_neighbors(idx):
                if state[neighbor] != PointState.FROZEN and region_mask[neighbor]:
                    T_new = self._compute_update(T, neighbor, speed)
                    if T_new < T[neighbor]:
                        T[neighbor] = T_new
                        flat_neighbor = np.ravel_multi_index(neighbor, self._shape)
                        heapq.heappush(trial_heap, (T_new, flat_neighbor))
                        state[neighbor] = PointState.TRIAL

        return T

    def _initialize_interface(
        self,
        phi: NDArray[np.float64],
        subcell_accuracy: bool = True,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """
        Initialize frozen points at the interface.

        Args:
            phi: Level set function.
            subcell_accuracy: Use subcell interface location.

        Returns:
            (frozen_mask, frozen_values) tuple.
        """
        frozen_mask = np.zeros(self._shape, dtype=bool)
        frozen_values = np.zeros(self._shape, dtype=np.float64)

        # Find interface cells: where phi changes sign with a neighbor
        it = np.nditer(phi, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            phi_i = phi[idx]

            for dim in range(self._ndim):
                for direction in [-1, 1]:
                    neighbor_idx = list(idx)
                    neighbor_idx[dim] += direction
                    neighbor = tuple(neighbor_idx)

                    # Check bounds
                    if not (0 <= neighbor[dim] < self._shape[dim]):
                        continue

                    phi_j = phi[neighbor]

                    # Sign change indicates interface
                    if phi_i * phi_j < 0:
                        # Interface between idx and neighbor
                        if subcell_accuracy:
                            # Linear interpolation: interface at theta fraction from idx
                            # phi_i + theta * (phi_j - phi_i) = 0
                            # theta = -phi_i / (phi_j - phi_i) = phi_i / (phi_i - phi_j)
                            theta = phi_i / (phi_i - phi_j)
                            dist_to_interface = theta * self._spacing[dim]
                        else:
                            dist_to_interface = 0.5 * self._spacing[dim]

                        # Update if this is a better (smaller) distance
                        if not frozen_mask[idx] or dist_to_interface < frozen_values[idx]:
                            frozen_mask[idx] = True
                            frozen_values[idx] = dist_to_interface

            it.iternext()

        return frozen_mask, frozen_values

    def _get_neighbors(self, idx: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Get valid neighbor indices (face-connected, not diagonal)."""
        neighbors = []
        for dim in range(self._ndim):
            for direction in [-1, 1]:
                neighbor_idx = list(idx)
                neighbor_idx[dim] += direction
                neighbor = tuple(neighbor_idx)

                # Check bounds
                if 0 <= neighbor[dim] < self._shape[dim]:
                    neighbors.append(neighbor)

        return neighbors

    def _compute_update(
        self,
        T: NDArray[np.float64],
        idx: tuple[int, ...],
        speed: float,
    ) -> float:
        """Compute Godunov update at a grid point."""
        # Gather neighbor values in each dimension
        T_neighbors = []
        for dim in range(self._ndim):
            T_minus = np.inf
            T_plus = np.inf

            # Minus direction
            if idx[dim] > 0:
                neighbor_idx = list(idx)
                neighbor_idx[dim] -= 1
                T_minus = T[tuple(neighbor_idx)]

            # Plus direction
            if idx[dim] < self._shape[dim] - 1:
                neighbor_idx = list(idx)
                neighbor_idx[dim] += 1
                T_plus = T[tuple(neighbor_idx)]

            T_neighbors.append((T_minus, T_plus))

        return godunov_update_nd(T_neighbors, self._spacing, speed)


if __name__ == "__main__":
    """Smoke tests for Fast Marching Method."""
    print("Testing Fast Marching Method...")

    from mfg_pde.geometry.boundary import no_flux_bc
    from mfg_pde.geometry.grids.tensor_grid import TensorProductGrid

    # Test 1: 1D point source
    print("\n[Test 1: 1D Point Source]")
    Nx_pts = 101
    grid_1d = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx_pts], boundary_conditions=no_flux_bc(dimension=1))
    x = grid_1d.coordinates[0]
    dx = grid_1d.spacing[0]
    print(f"  Grid: {Nx_pts} points, dx = {dx:.4f}")

    fmm_1d = FastMarchingMethod(grid_1d)

    # Point source at x = 0.5
    x0 = 0.5
    i0 = int(x0 / dx)
    frozen_mask_1d = np.zeros(Nx_pts, dtype=bool)
    frozen_mask_1d[i0] = True
    frozen_values_1d = np.zeros(Nx_pts, dtype=np.float64)
    frozen_values_1d[i0] = 0.0

    T_1d = fmm_1d.solve(speed=1.0, frozen_mask=frozen_mask_1d, frozen_values=frozen_values_1d)

    # Analytical solution: T = |x - x0|
    T_exact_1d = np.abs(x - x0)
    error_1d = np.max(np.abs(T_1d - T_exact_1d))
    print(f"  Point source at x = {x0}")
    print(f"  Max error: {error_1d:.6f} (should be ~ dx = {dx:.4f})")
    assert error_1d < 2 * dx, f"1D point source error too large: {error_1d}"
    print("  [OK] 1D point source")

    # Test 2: 2D point source
    print("\n[Test 2: 2D Point Source]")
    Nx_pts, Ny_pts = 51, 51
    grid_2d = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[Nx_pts, Ny_pts],
        boundary_conditions=no_flux_bc(dimension=2),
    )
    X, Y = grid_2d.meshgrid()
    dx_2d = grid_2d.spacing[0]
    print(f"  Grid: {Nx_pts}x{Ny_pts}, dx = {dx_2d:.4f}")

    fmm_2d = FastMarchingMethod(grid_2d)

    # Point source at center
    center = (0.5, 0.5)
    i0, j0 = Nx_pts // 2, Ny_pts // 2
    frozen_mask_2d = np.zeros((Nx_pts, Ny_pts), dtype=bool)
    frozen_mask_2d[i0, j0] = True
    frozen_values_2d = np.zeros((Nx_pts, Ny_pts), dtype=np.float64)

    T_2d = fmm_2d.solve(speed=1.0, frozen_mask=frozen_mask_2d, frozen_values=frozen_values_2d)

    # Analytical solution: T = sqrt((x-x0)^2 + (y-y0)^2)
    T_exact_2d = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    error_2d = np.max(np.abs(T_2d - T_exact_2d))
    print(f"  Point source at center ({center[0]}, {center[1]})")
    print(f"  Max error: {error_2d:.6f} (expected O(dx) = {dx_2d:.4f})")
    assert error_2d < 3 * dx_2d, f"2D point source error too large: {error_2d}"
    print("  [OK] 2D point source")

    # Test 3: Signed distance from circle
    print("\n[Test 3: Signed Distance - Circle]")
    radius = 0.3
    phi_circle = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - radius
    print(f"  Circle: center (0.5, 0.5), radius {radius}")

    phi_sdf = fmm_2d.compute_signed_distance(phi_circle, subcell_accuracy=True)

    # Analytical SDF is the initial phi (already SDF for circle)
    error_sdf = np.max(np.abs(phi_sdf - phi_circle))
    print(f"  Max error vs analytical: {error_sdf:.6f}")

    # Check |grad phi| = 1
    grad_x = np.gradient(phi_sdf, dx_2d, axis=0)
    grad_y = np.gradient(phi_sdf, dx_2d, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_error = np.abs(grad_mag - 1.0)
    # Exclude boundaries where gradient is less accurate
    interior = (slice(2, -2), slice(2, -2))
    max_grad_error = np.max(grad_error[interior])
    mean_grad_error = np.mean(grad_error[interior])
    print(f"  |grad phi| - 1: max = {max_grad_error:.4f}, mean = {mean_grad_error:.4f}")

    assert error_sdf < 2 * dx_2d, f"SDF error too large: {error_sdf}"
    print("  [OK] Circle SDF")

    # Test 4: Grid refinement convergence
    print("\n[Test 4: Grid Refinement Convergence]")
    errors = []
    grid_sizes = [25, 50, 100]
    for N in grid_sizes:
        grid_test = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[N, N],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        fmm_test = FastMarchingMethod(grid_test)
        X_test, Y_test = grid_test.meshgrid()
        dx_test = grid_test.spacing[0]

        # Circle SDF
        phi_test = np.sqrt((X_test - 0.5) ** 2 + (Y_test - 0.5) ** 2) - 0.3
        phi_sdf_test = fmm_test.compute_signed_distance(phi_test)
        error = np.max(np.abs(phi_sdf_test - phi_test))
        errors.append(error)
        print(f"  N = {N:3d}, dx = {dx_test:.4f}, error = {error:.6f}")

    # Check convergence rate (should be ~O(dx))
    rate_1 = np.log(errors[0] / errors[1]) / np.log(2)
    rate_2 = np.log(errors[1] / errors[2]) / np.log(2)
    print(f"  Convergence rates: {rate_1:.2f}, {rate_2:.2f} (expected ~1.0)")
    assert rate_1 > 0.5, f"Poor convergence rate: {rate_1}"
    print("  [OK] Grid refinement shows convergence")

    # Test 5: Non-unit speed
    print("\n[Test 5: Non-unit Speed Function]")
    # Speed F = 2 means |grad T| = 0.5, so T grows half as fast
    T_fast = fmm_2d.solve(speed=2.0, frozen_mask=frozen_mask_2d, frozen_values=frozen_values_2d)
    T_exact_fast = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 2.0
    error_fast = np.max(np.abs(T_fast - T_exact_fast))
    print(f"  Speed = 2.0, max error: {error_fast:.6f}")
    assert error_fast < 3 * dx_2d, "Non-unit speed error too large"
    print("  [OK] Non-unit speed")

    print("\n[OK] All Fast Marching Method tests passed!")
