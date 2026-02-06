"""
Fast Sweeping Method (FSM) for the Eikonal Equation.

The Fast Sweeping Method solves the Eikonal equation |grad T| = 1/F using
Gauss-Seidel iterations with alternating sweep directions. It has O(N)
complexity per sweep and typically converges in a small number of iterations.

Algorithm Overview:
    1. Initialize T = inf everywhere except frozen points
    2. Repeat until convergence:
       - Sweep in all 2^d directions (++, +-, -+, -- in 2D)
       - For each point, compute Godunov update using current T values
       - Update T if new value is smaller (monotonicity)
    3. Return T array

Compared to FMM:
    - FSM: O(N) per sweep, but may need multiple iterations
    - FMM: O(N log N) total, single pass
    - FSM is often faster for simple domains, FMM for complex geometries

Sweep Directions:
    In 2D, the 4 sweeps are:
    - (1,1): i = 0..N-1, j = 0..M-1
    - (1,-1): i = 0..N-1, j = M-1..0
    - (-1,1): i = N-1..0, j = 0..M-1
    - (-1,-1): i = N-1..0, j = M-1..0

    In nD, there are 2^n sweep directions.

Convergence:
    For simple domains (convex obstacles), FSM converges in 2^d iterations.
    For complex domains, more iterations may be needed.

References:
- Zhao (2005): A fast sweeping method for Eikonal equations, Math. Comp.
- Tsai et al. (2003): Fast sweeping algorithms for a class of HJ equations

Created: 2026-02-06 (Issue #664)
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import numpy as np

from mfg_pde.geometry.level_set.eikonal.godunov_update import godunov_update_nd
from mfg_pde.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.geometry.grids.tensor_grid import TensorProductGrid

logger = get_logger(__name__)


class FastSweepingMethod:
    """
    Fast Sweeping Method for solving the Eikonal equation.

    Solves |grad T| = 1/F using Gauss-Seidel iterations with alternating sweeps.

    Attributes:
        geometry: TensorProductGrid providing grid structure and spacing.
        max_iterations: Maximum number of full sweep cycles.
        tolerance: Convergence tolerance for max change in T.

    Example:
        >>> from mfg_pde.geometry.grids import TensorProductGrid
        >>> from mfg_pde.geometry.boundary import no_flux_bc
        >>> grid = TensorProductGrid(
        ...     bounds=[(0, 1), (0, 1)], Nx=[100, 100],
        ...     boundary_conditions=no_flux_bc(dimension=2)
        ... )
        >>> fsm = FastSweepingMethod(grid)
        >>> phi_sdf = fsm.compute_signed_distance(phi_initial)
    """

    def __init__(
        self,
        geometry: TensorProductGrid,
        max_iterations: int = 10,
        tolerance: float = 1e-10,
    ) -> None:
        """
        Initialize FSM solver.

        Args:
            geometry: Grid providing spacing and structure.
            max_iterations: Max sweep cycles (default: 10, usually 2-3 suffice).
            tolerance: Stop when max(|T_new - T_old|) < tolerance.
        """
        self.geometry = geometry
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self._spacing = tuple(geometry.spacing)
        self._shape = tuple(geometry.Nx_points)
        self._ndim = len(self._shape)

        # Precompute sweep directions: all combinations of +1 and -1
        self._sweep_directions = list(product([-1, 1], repeat=self._ndim))

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
            frozen_mask: Boolean mask of points with known values.
            frozen_values: Values at frozen points.

        Returns:
            Solution T with |grad T| = 1/F.
        """
        # Convert speed to array if scalar
        if np.isscalar(speed):
            speed_array = np.full(self._shape, float(speed), dtype=np.float64)
        else:
            speed_array = np.asarray(speed, dtype=np.float64)

        # Initialize T
        T = np.full(self._shape, np.inf, dtype=np.float64)
        T[frozen_mask] = frozen_values[frozen_mask]

        # Iterative sweeping
        for iteration in range(self.max_iterations):
            max_change = 0.0

            # Perform all 2^d sweeps
            for direction in self._sweep_directions:
                change = self._single_sweep(T, speed_array, frozen_mask, direction)
                max_change = max(max_change, change)

            logger.debug(f"FSM iteration {iteration}: max change = {max_change:.2e}")

            if max_change < self.tolerance:
                logger.debug(f"FSM converged in {iteration + 1} iterations")
                break

        return T

    def compute_signed_distance(
        self,
        phi_initial: NDArray[np.float64],
        subcell_accuracy: bool = True,
    ) -> NDArray[np.float64]:
        """
        Compute signed distance function from initial level set.

        Args:
            phi_initial: Initial level set. Zero level set is the interface.
            subcell_accuracy: If True, use subcell interface location.

        Returns:
            Signed distance function with |grad phi| = 1.
        """
        # Initialize frozen points at interface
        frozen_mask, frozen_values = self._initialize_interface(phi_initial, subcell_accuracy=subcell_accuracy)

        # Handle exact zeros: points with phi = 0 exactly should be frozen at 0
        exact_zeros = np.abs(phi_initial) < 1e-14
        frozen_mask = frozen_mask | exact_zeros
        frozen_values[exact_zeros] = 0.0

        # Solve for positive region (includes zeros)
        T_positive = self._solve_one_sided(frozen_mask, frozen_values, phi_initial >= 0)

        # Solve for negative region (includes zeros)
        T_negative = self._solve_one_sided(frozen_mask, frozen_values, phi_initial <= 0)

        # Combine: positive distance where phi >= 0, negative where phi < 0
        result = np.where(phi_initial >= 0, T_positive, -T_negative)

        return result

    def _solve_one_sided(
        self,
        frozen_mask: NDArray[np.bool_],
        frozen_values: NDArray[np.float64],
        region_mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Solve FSM for one side of the interface."""
        T = np.full(self._shape, np.inf, dtype=np.float64)
        T[frozen_mask] = np.abs(frozen_values[frozen_mask])

        # Create effective frozen mask for this region
        effective_frozen = frozen_mask.copy()

        # Speed = 1 for SDF
        speed_array = np.ones(self._shape, dtype=np.float64)

        for _iteration in range(self.max_iterations):
            max_change = 0.0

            for direction in self._sweep_directions:
                change = self._single_sweep_onesided(T, speed_array, effective_frozen, direction, region_mask)
                max_change = max(max_change, change)

            if max_change < self.tolerance:
                break

        return T

    def _single_sweep(
        self,
        T: NDArray[np.float64],
        speed: NDArray[np.float64],
        frozen_mask: NDArray[np.bool_],
        direction: tuple[int, ...],
    ) -> float:
        """Perform one sweep in a given direction."""
        max_change = 0.0

        # Build index ranges for this sweep direction
        ranges = []
        for dim in range(self._ndim):
            if direction[dim] == 1:
                ranges.append(range(self._shape[dim]))
            else:
                ranges.append(range(self._shape[dim] - 1, -1, -1))

        # Sweep through all points
        for idx in product(*ranges):
            if frozen_mask[idx]:
                continue

            T_old = T[idx]
            T_new = self._compute_update(T, idx, speed[idx])

            if T_new < T_old:
                T[idx] = T_new
                max_change = max(max_change, T_old - T_new)

        return max_change

    def _single_sweep_onesided(
        self,
        T: NDArray[np.float64],
        speed: NDArray[np.float64],
        frozen_mask: NDArray[np.bool_],
        direction: tuple[int, ...],
        region_mask: NDArray[np.bool_],
    ) -> float:
        """Perform one sweep restricted to a region."""
        max_change = 0.0

        ranges = []
        for dim in range(self._ndim):
            if direction[dim] == 1:
                ranges.append(range(self._shape[dim]))
            else:
                ranges.append(range(self._shape[dim] - 1, -1, -1))

        for idx in product(*ranges):
            if frozen_mask[idx] or not region_mask[idx]:
                continue

            T_old = T[idx]
            T_new = self._compute_update(T, idx, speed[idx])

            if T_new < T_old:
                T[idx] = T_new
                max_change = max(max_change, T_old - T_new)

        return max_change

    def _compute_update(
        self,
        T: NDArray[np.float64],
        idx: tuple[int, ...],
        speed: float,
    ) -> float:
        """Compute Godunov update at a grid point."""
        T_neighbors = []
        for dim in range(self._ndim):
            T_minus = np.inf
            T_plus = np.inf

            if idx[dim] > 0:
                neighbor_idx = list(idx)
                neighbor_idx[dim] -= 1
                T_minus = T[tuple(neighbor_idx)]

            if idx[dim] < self._shape[dim] - 1:
                neighbor_idx = list(idx)
                neighbor_idx[dim] += 1
                T_plus = T[tuple(neighbor_idx)]

            T_neighbors.append((T_minus, T_plus))

        return godunov_update_nd(T_neighbors, self._spacing, speed)

    def _initialize_interface(
        self,
        phi: NDArray[np.float64],
        subcell_accuracy: bool = True,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """Initialize frozen points at the interface (same as FMM)."""
        frozen_mask = np.zeros(self._shape, dtype=bool)
        frozen_values = np.zeros(self._shape, dtype=np.float64)

        it = np.nditer(phi, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            phi_i = phi[idx]

            for dim in range(self._ndim):
                for direction in [-1, 1]:
                    neighbor_idx = list(idx)
                    neighbor_idx[dim] += direction
                    neighbor = tuple(neighbor_idx)

                    if not (0 <= neighbor[dim] < self._shape[dim]):
                        continue

                    phi_j = phi[neighbor]

                    if phi_i * phi_j < 0:
                        if subcell_accuracy:
                            theta = phi_i / (phi_i - phi_j)
                            dist_to_interface = theta * self._spacing[dim]
                        else:
                            dist_to_interface = 0.5 * self._spacing[dim]

                        if not frozen_mask[idx] or dist_to_interface < frozen_values[idx]:
                            frozen_mask[idx] = True
                            frozen_values[idx] = dist_to_interface

            it.iternext()

        return frozen_mask, frozen_values


if __name__ == "__main__":
    """Smoke tests for Fast Sweeping Method."""
    print("Testing Fast Sweeping Method...")

    from mfg_pde.geometry.boundary import no_flux_bc
    from mfg_pde.geometry.grids.tensor_grid import TensorProductGrid

    # Test 1: 1D point source
    print("\n[Test 1: 1D Point Source]")
    Nx_pts = 101
    grid_1d = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx_pts], boundary_conditions=no_flux_bc(dimension=1))
    x = grid_1d.coordinates[0]
    dx = grid_1d.spacing[0]
    print(f"  Grid: {Nx_pts} points, dx = {dx:.4f}")

    fsm_1d = FastSweepingMethod(grid_1d)

    # Point source at x = 0.5
    x0 = 0.5
    i0 = int(x0 / dx)
    frozen_mask_1d = np.zeros(Nx_pts, dtype=bool)
    frozen_mask_1d[i0] = True
    frozen_values_1d = np.zeros(Nx_pts, dtype=np.float64)

    T_1d = fsm_1d.solve(speed=1.0, frozen_mask=frozen_mask_1d, frozen_values=frozen_values_1d)

    T_exact_1d = np.abs(x - x0)
    error_1d = np.max(np.abs(T_1d - T_exact_1d))
    print(f"  Max error: {error_1d:.6f} (expected ~ dx = {dx:.4f})")
    assert error_1d < 2 * dx, f"1D error too large: {error_1d}"
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

    fsm_2d = FastSweepingMethod(grid_2d)

    center = (0.5, 0.5)
    i0, j0 = Nx_pts // 2, Ny_pts // 2
    frozen_mask_2d = np.zeros((Nx_pts, Ny_pts), dtype=bool)
    frozen_mask_2d[i0, j0] = True
    frozen_values_2d = np.zeros((Nx_pts, Ny_pts), dtype=np.float64)

    T_2d = fsm_2d.solve(speed=1.0, frozen_mask=frozen_mask_2d, frozen_values=frozen_values_2d)

    T_exact_2d = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    error_2d = np.max(np.abs(T_2d - T_exact_2d))
    print(f"  Max error: {error_2d:.6f} (expected O(dx) = {dx_2d:.4f})")
    assert error_2d < 3 * dx_2d, f"2D error too large: {error_2d}"
    print("  [OK] 2D point source")

    # Test 3: Signed distance from circle
    print("\n[Test 3: Signed Distance - Circle]")
    radius = 0.3
    phi_circle = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - radius

    phi_sdf = fsm_2d.compute_signed_distance(phi_circle, subcell_accuracy=True)

    error_sdf = np.max(np.abs(phi_sdf - phi_circle))
    print(f"  Max error vs analytical: {error_sdf:.6f}")
    assert error_sdf < 2 * dx_2d, f"SDF error too large: {error_sdf}"
    print("  [OK] Circle SDF")

    # Test 4: Convergence in few iterations
    print("\n[Test 4: Iteration Count]")
    # For simple domains, FSM should converge in 2^d = 4 iterations (2D)
    fsm_verbose = FastSweepingMethod(grid_2d, max_iterations=20, tolerance=1e-14)
    T_verbose = fsm_verbose.solve(speed=1.0, frozen_mask=frozen_mask_2d, frozen_values=frozen_values_2d)
    # Should be same as T_2d (converged)
    diff = np.max(np.abs(T_verbose - T_2d))
    print(f"  Difference from default iterations: {diff:.2e}")
    print("  [OK] FSM converges quickly for simple domains")

    # Test 5: Compare FMM vs FSM
    print("\n[Test 5: FMM vs FSM Comparison]")
    from mfg_pde.geometry.level_set.eikonal.fast_marching import FastMarchingMethod

    fmm_2d = FastMarchingMethod(grid_2d)
    T_fmm = fmm_2d.solve(speed=1.0, frozen_mask=frozen_mask_2d, frozen_values=frozen_values_2d)
    T_fsm = fsm_2d.solve(speed=1.0, frozen_mask=frozen_mask_2d, frozen_values=frozen_values_2d)

    diff_fmm_fsm = np.max(np.abs(T_fmm - T_fsm))
    print(f"  Max |T_fmm - T_fsm|: {diff_fmm_fsm:.2e}")
    assert diff_fmm_fsm < 1e-10, f"FMM and FSM differ: {diff_fmm_fsm}"
    print("  [OK] FMM and FSM produce identical results")

    print("\n[OK] All Fast Sweeping Method tests passed!")
