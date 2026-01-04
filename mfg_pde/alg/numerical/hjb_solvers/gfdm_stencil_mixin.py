"""
GFDM Stencil and Taylor Expansion Mixin for HJB Solvers.

This module provides stencil construction and Taylor expansion functionality:
- Neighborhood structure building with adaptive delta enlargement
- Reverse neighborhood mapping for sparse Jacobian
- Taylor matrix construction with SVD/QR decomposition
- Weight function computation (Wendland, Gaussian, cubic spline)
- Derivative weight extraction from Taylor coefficients

These methods are extracted from HJBGFDMSolver to improve maintainability.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist


class GFDMStencilMixin:
    """
    Mixin providing stencil construction methods for GFDM solvers.

    This mixin expects the following attributes from the host class:
    - collocation_points: np.ndarray of shape (N, dim)
    - dimension: int
    - n_points: int
    - delta: float (neighborhood radius)
    - taylor_order: int
    - weight_function: str
    - weight_scale: float
    - k_min: int (minimum neighbors)
    - adaptive_neighborhoods: bool
    - max_delta_multiplier: float
    - boundary_indices: np.ndarray
    - n_derivatives: int
    - multi_indices: list of tuples
    - _gfdm_operator: GFDMOperator instance
    - _use_local_coordinate_rotation: bool
    - adaptive_stats: dict
    """

    # Type hints for attributes expected from host class
    collocation_points: np.ndarray
    dimension: int
    n_points: int
    delta: float
    taylor_order: int
    weight_function: str
    weight_scale: float
    k_min: int
    adaptive_neighborhoods: bool
    max_delta_multiplier: float
    boundary_indices: np.ndarray
    n_derivatives: int
    multi_indices: list
    _gfdm_operator: Any
    _use_local_coordinate_rotation: bool
    adaptive_stats: dict
    neighborhoods: dict
    taylor_matrices: dict

    def _build_neighborhood_structure(self) -> None:
        """
        Build delta-neighborhood structure for all collocation points.

        Uses GFDMOperator's neighborhoods (without ghost particles - pure one-sided stencils)
        and only extends for points needing adaptive delta enlargement.

        For boundary points, also ensures sufficient "deep" neighbors exist
        (neighbors with adequate depth in the normal direction) to prevent
        degenerate "pancake" stencils that cannot capture normal derivatives.
        """
        self.neighborhoods = {}

        # For adaptive delta, we need pairwise distances
        if self.adaptive_neighborhoods:
            distances = cdist(self.collocation_points, self.collocation_points)
        else:
            distances = None

        # Parameters for deep neighbor check (Issue #531: GFDM boundary stencil degeneracy)
        # Minimum depth: fraction of delta that neighbors must extend into interior
        min_depth_fraction = 0.3  # Neighbors must be at least 0.3*delta into interior
        # Minimum number of deep neighbors required for boundary points
        k_deep = max(2, self.dimension)  # At least 2 (or dimension) deep neighbors

        # Pre-compute boundary set for fast lookup
        boundary_set = set(self.boundary_indices) if len(self.boundary_indices) > 0 else set()

        for i in range(self.n_points):
            # Start with GFDMOperator's neighborhood (pure one-sided stencils, no ghost particles)
            base_neighborhood = self._gfdm_operator.get_neighborhood(i)
            n_neighbors = base_neighborhood["size"]

            # Check if this is a boundary point
            is_boundary = i in boundary_set

            # For boundary points, also check depth quality
            needs_depth_adaptation = False
            if is_boundary and self.adaptive_neighborhoods and distances is not None:
                min_depth = min_depth_fraction * self.delta
                deep_count = self._count_deep_neighbors(i, base_neighborhood["points"], min_depth)
                needs_depth_adaptation = deep_count < k_deep

            # Check if adaptive delta enlargement is needed (count OR depth insufficient)
            needs_adaptation = (
                self.adaptive_neighborhoods
                and distances is not None
                and (n_neighbors < self.k_min or needs_depth_adaptation)
            )

            if needs_adaptation:
                # Adaptive delta enlargement for insufficient neighbors or depth
                neighbor_indices = base_neighborhood["indices"].copy()
                neighbor_points = base_neighborhood["points"].copy()
                neighbor_distances = base_neighborhood["distances"].copy()

                # Only count real neighbors (not ghost particles) for k_min check
                real_neighbor_count = np.sum(neighbor_indices >= 0)

                delta_current = self.delta
                delta_multiplier = 1.0
                was_adapted = False
                max_delta = self.delta * self.max_delta_multiplier

                # Track deep neighbor count for boundary points
                deep_count = 0
                if is_boundary:
                    min_depth = min_depth_fraction * delta_current
                    deep_count = self._count_deep_neighbors(i, neighbor_points, min_depth)

                # Continue expanding if count OR depth is insufficient
                count_ok = real_neighbor_count >= self.k_min
                depth_ok = (not is_boundary) or (deep_count >= k_deep)

                while (not count_ok or not depth_ok) and delta_current < max_delta:
                    # Enlarge delta by 20% increments
                    delta_multiplier *= 1.2
                    delta_current = self.delta * delta_multiplier

                    # Recompute neighborhood with enlarged delta
                    neighbor_mask = distances[i, :] < delta_current
                    neighbor_indices = np.where(neighbor_mask)[0]
                    real_neighbor_count = len(neighbor_indices)
                    was_adapted = True

                    # Recompute deep neighbors for boundary points
                    if is_boundary:
                        neighbor_points_temp = self.collocation_points[neighbor_indices]
                        min_depth = min_depth_fraction * delta_current
                        deep_count = self._count_deep_neighbors(i, neighbor_points_temp, min_depth)

                    # Update loop conditions
                    count_ok = real_neighbor_count >= self.k_min
                    depth_ok = (not is_boundary) or (deep_count >= k_deep)

                # Update neighborhood data for enlarged delta
                if was_adapted:
                    neighbor_points = self.collocation_points[neighbor_indices]
                    neighbor_distances = distances[i, neighbor_indices]

                    # Re-add ghost particles if this is a boundary point
                    if base_neighborhood.get("has_ghost", False):
                        ghost_particles, ghost_distances = self._gfdm_operator._create_ghost_particles(i)
                        if ghost_particles:
                            neighbor_points = np.vstack(
                                [neighbor_points] + [gp.reshape(1, -1) for gp in ghost_particles]
                            )
                            neighbor_distances = np.concatenate([neighbor_distances, np.array(ghost_distances)])
                            neighbor_indices = np.concatenate(
                                [neighbor_indices, np.array([-1 - j for j in range(len(ghost_particles))])]
                            )

                # Track maximum delta used
                if delta_current > self.adaptive_stats["max_delta_used"]:
                    self.adaptive_stats["max_delta_used"] = delta_current

                # Record adaptive enlargement
                if was_adapted:
                    self.adaptive_stats["n_adapted"] += 1
                    self.adaptive_stats["adaptive_enlargements"].append(
                        {
                            "point_idx": i,
                            "base_delta": self.delta,
                            "adapted_delta": delta_current,
                            "delta_multiplier": delta_multiplier,
                            "n_neighbors": len(neighbor_indices),
                            "is_boundary": is_boundary,
                            "deep_neighbors": deep_count if is_boundary else None,
                        }
                    )

                # Warn if still insufficient neighbors or depth
                import warnings as _warnings

                if real_neighbor_count < self.k_min:
                    _warnings.warn(
                        f"Point {i}: Could not find {self.k_min} neighbors even with "
                        f"delta={delta_current:.4f} ({delta_multiplier:.2f}x base). "
                        f"Only found {real_neighbor_count} neighbors. GFDM approximation may be poor.",
                        UserWarning,
                        stacklevel=3,
                    )
                elif is_boundary and deep_count < k_deep:
                    _warnings.warn(
                        f"Boundary point {i}: Only {deep_count}/{k_deep} deep neighbors "
                        f"(depth >= {min_depth_fraction}*delta) even with "
                        f"delta={delta_current:.4f} ({delta_multiplier:.2f}x base). "
                        f"Normal derivative approximation may be inaccurate.",
                        UserWarning,
                        stacklevel=3,
                    )

                # Store adapted neighborhood
                self.neighborhoods[i] = {
                    "indices": np.array(neighbor_indices) if isinstance(neighbor_indices, list) else neighbor_indices,
                    "points": np.array(neighbor_points) if isinstance(neighbor_points, list) else neighbor_points,
                    "distances": np.array(neighbor_distances)
                    if isinstance(neighbor_distances, list)
                    else neighbor_distances,
                    "size": len(neighbor_indices),
                    "has_ghost": base_neighborhood.get("has_ghost", False),
                    "ghost_count": base_neighborhood.get("ghost_count", 0),
                    "adapted": True,
                }
            else:
                # Use GFDMOperator's neighborhood directly (no adaptation needed)
                self.neighborhoods[i] = {
                    "indices": base_neighborhood["indices"],
                    "points": base_neighborhood["points"],
                    "distances": base_neighborhood["distances"],
                    "size": base_neighborhood["size"],
                    "has_ghost": base_neighborhood.get("has_ghost", False),
                    "ghost_count": base_neighborhood.get("ghost_count", 0),
                    "adapted": False,
                }

        # Report adaptive neighborhood statistics if enabled
        if self.adaptive_neighborhoods:
            n_adapted = self.adaptive_stats["n_adapted"]
            if n_adapted > 0:
                pct_adapted = 100.0 * n_adapted / self.n_points
                avg_multiplier = np.mean([e["delta_multiplier"] for e in self.adaptive_stats["adaptive_enlargements"]])
                max_multiplier = np.max([e["delta_multiplier"] for e in self.adaptive_stats["adaptive_enlargements"]])

                import warnings

                warnings.warn(
                    f"Adaptive neighborhoods: {n_adapted}/{self.n_points} points ({pct_adapted:.1f}%) "
                    f"required delta enlargement. Base delta: {self.delta:.4f}, "
                    f"Max delta used: {self.adaptive_stats['max_delta_used']:.4f} "
                    f"({max_multiplier:.2f}x base), Avg multiplier: {avg_multiplier:.2f}x. "
                    f"Consider increasing base delta for better theoretical accuracy.",
                    UserWarning,
                    stacklevel=2,
                )

    def _build_reverse_neighborhoods(self) -> None:
        """
        Build reverse neighborhood map: for each point j, find all points i that have j in their neighborhood.

        This enables sparse Jacobian computation - when perturbing u[j], only residuals at
        points in reverse_neighborhoods[j] are affected.

        Complexity: O(n * k) where k is average neighborhood size.
        """
        self._reverse_neighborhoods: dict[int, list[int]] = {j: [] for j in range(self.n_points)}

        for i in range(self.n_points):
            neighborhood = self.neighborhoods[i]
            neighbor_indices = neighborhood["indices"]

            # For each neighbor j of point i, add i to j's reverse neighborhood
            for j in neighbor_indices:
                if 0 <= j < self.n_points:  # Exclude ghost particles (negative indices)
                    self._reverse_neighborhoods[j].append(i)

        # Convert to arrays for faster access
        self._reverse_neighborhoods = {j: np.array(rows, dtype=int) for j, rows in self._reverse_neighborhoods.items()}

    def _get_affected_rows(self, j: int) -> np.ndarray:
        """
        Get rows affected by perturbing u[j] for sparse Jacobian.

        Perturbing u[j] affects:
        1. Row j (the point itself - its residual depends on its own value)
        2. All rows i where j is in the neighborhood of i (j affects i's derivative approximation)

        Returns:
            Array of row indices affected by u[j]
        """
        # Get points that have j in their neighborhood
        affected = self._reverse_neighborhoods.get(j, np.array([], dtype=int))

        # Always include j itself (residual at j depends on u[j])
        if j not in affected:
            affected = np.append(affected, j)

        return affected

    def _build_taylor_matrices(self) -> None:
        """
        Pre-compute Taylor expansion matrices A for all collocation points.

        Uses GFDMOperator's Taylor matrices when the neighborhood wasn't adapted,
        only rebuilds for points that needed adaptive delta enlargement.
        """
        self.taylor_matrices = {}

        for i in range(self.n_points):
            neighborhood = self.neighborhoods[i]
            n_neighbors_raw = neighborhood["size"]
            n_neighbors = int(n_neighbors_raw) if isinstance(n_neighbors_raw, int | float) else 0

            if n_neighbors < self.n_derivatives:
                self.taylor_matrices[i] = None
                continue

            # Check if we can reuse GFDMOperator's Taylor matrices
            # (only if neighborhood wasn't adapted AND no LCR rotation applied)
            # For LCR boundary points, we need to rebuild with rotated offsets
            has_lcr_rotation = "rotated_offsets" in neighborhood and self._use_local_coordinate_rotation
            if not neighborhood.get("adapted", False) and not has_lcr_rotation:
                base_taylor = self._gfdm_operator.get_taylor_data(i)
                if base_taylor is not None:
                    # Compute condition number safely
                    S = base_taylor["S"]
                    if len(S) > 0 and S[-1] > 1e-15:
                        cond_num = S[0] / S[-1]
                    else:
                        cond_num = np.inf

                    # Wrap GFDMOperator's format to HJBGFDMSolver's expected format
                    self.taylor_matrices[i] = {
                        "A": base_taylor["A"],
                        "W": base_taylor["W"],
                        "sqrt_W": base_taylor["sqrt_W"],
                        "U": base_taylor["U"],
                        "S": S,
                        "Vt": base_taylor["Vt"],
                        "rank": base_taylor["rank"],
                        "condition_number": cond_num,
                        "use_svd": True,
                        "use_qr": False,
                    }
                    continue

            # Need to build Taylor matrices for adapted neighborhoods
            # Build Taylor expansion matrix A
            A = np.zeros((n_neighbors, self.n_derivatives))
            center_point = self.collocation_points[i]
            neighbor_points = neighborhood["points"]

            # For LCR-enabled boundary points, use rotated offsets for better conditioning
            # of normal derivative computation (Issue #531)
            use_rotated = "rotated_offsets" in neighborhood and self._use_local_coordinate_rotation
            rotated_offsets = neighborhood.get("rotated_offsets") if use_rotated else None

            for j, neighbor_point in enumerate(neighbor_points):
                if rotated_offsets is not None:
                    delta_x = rotated_offsets[j]
                else:
                    delta_x = neighbor_point - center_point

                for k, beta in enumerate(self.multi_indices):
                    # Compute (x - x_center)^beta / beta!
                    term = 1.0
                    factorial = 1.0

                    for dim in range(self.dimension):
                        if beta[dim] > 0:
                            term *= delta_x[dim] ** beta[dim]
                            factorial *= math.factorial(beta[dim])

                    A[j, k] = term / factorial

            # Compute weights and store matrices
            weights = self._compute_weights(np.asarray(neighborhood["distances"]))
            W = np.diag(weights)

            # Use SVD or QR decomposition to avoid condition number amplification
            sqrt_W = np.sqrt(W)
            WA = sqrt_W @ A

            # Try SVD first (most robust)
            try:
                # SVD decomposition: WA = U @ S @ V^T
                U, S, Vt = np.linalg.svd(WA, full_matrices=False)

                # Condition number check and regularization
                condition_number = S[0] / S[-1] if S[-1] > 1e-15 else np.inf

                # Truncate small singular values for regularization
                tolerance = 1e-12
                rank = np.sum(tolerance < S)

                self.taylor_matrices[i] = {
                    "A": A,
                    "W": W,
                    "sqrt_W": sqrt_W,
                    "U": U[:, :rank],
                    "S": S[:rank],
                    "Vt": Vt[:rank, :],
                    "rank": rank,
                    "condition_number": condition_number,
                    "use_svd": True,
                    "use_qr": False,
                }

            except np.linalg.LinAlgError:
                # Fallback to QR decomposition if SVD fails
                try:
                    # QR decomposition: WA = Q @ R
                    Q, R = np.linalg.qr(WA)
                    self.taylor_matrices[i] = {
                        "A": A,
                        "W": W,
                        "sqrt_W": sqrt_W,
                        "Q": Q,
                        "R": R,
                        "use_qr": True,
                        "use_svd": False,
                    }
                except np.linalg.LinAlgError:
                    # Final fallback to normal equations if QR also fails
                    self.taylor_matrices[i] = {
                        "A": A,
                        "W": W,
                        "AtW": A.T @ W,
                        "AtWA_inv": None,
                        "use_qr": False,
                        "use_svd": False,
                    }

                    try:
                        AtWA = A.T @ W @ A
                        if np.linalg.det(AtWA) > 1e-12:
                            self.taylor_matrices[i]["AtWA_inv"] = np.linalg.inv(AtWA)
                    except (np.linalg.LinAlgError, FloatingPointError):
                        # Cannot compute inverse - leave AtWA_inv as None
                        pass

    def _compute_weights(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute weights based on distance and weight function using smoothing kernels.

        Uses the unified kernel API from mfg_pde.utils.numerical.kernels.

        For QP optimization (monotonicity constraints), prefer:
        - 'cubic_spline': Compact support with good weight distribution
        - 'wendland': Compact support with C^4 smoothness
        Avoid 'gaussian' for QP - weights decay too fast, causing distant neighbors
        to have near-zero weight which can re-introduce singularity issues.
        """
        from mfg_pde.utils.numerical.kernels import (
            CubicSplineKernel,
            GaussianKernel,
            WendlandKernel,
        )

        if self.weight_function == "gaussian":
            # Use GaussianKernel with smoothing length = weight_scale
            kernel = GaussianKernel()
            return kernel(distances, h=self.weight_scale)

        elif self.weight_function == "inverse_distance":
            # Keep legacy inverse distance weights (not a standard kernel)
            return 1.0 / (distances + 1e-12)

        elif self.weight_function == "uniform":
            # Keep legacy uniform weights (trivial case)
            return np.ones_like(distances)

        elif self.weight_function == "wendland":
            # Use Wendland C^4 kernel: (1 - r/h)_+^6 (35q^2 + 18q + 3)
            kernel = WendlandKernel(k=2)  # k=2 -> C^4 continuity
            # Support radius = delta (neighborhood size)
            return kernel(distances, h=self.delta)

        elif self.weight_function == "cubic_spline":
            # Cubic B-spline (M4) kernel - good for QP optimization
            # Compact support (2h) with smoother profile than Wendland
            # Better weight distribution for distant neighbors vs gaussian
            kernel = CubicSplineKernel(dimension=self.dimension)
            return kernel(distances, h=self.delta / 2)  # Support radius = 2h, so h = delta/2

        else:
            raise ValueError(
                f"Unknown weight function: {self.weight_function}. "
                f"Available options: 'gaussian', 'inverse_distance', 'uniform', 'wendland', 'cubic_spline'"
            )

    def _compute_derivative_weights_from_taylor(self, point_idx: int) -> dict | None:
        """
        Compute derivative weights from our Taylor matrices for a single point.

        Used for LCR boundary points where we need weights computed from
        LCR-rotated Taylor matrices rather than from GFDMOperator.

        Args:
            point_idx: Index of collocation point

        Returns:
            Dictionary with neighbor_indices, grad_weights, lap_weights,
            or None if Taylor matrix is unavailable.
        """
        taylor_data = self.taylor_matrices.get(point_idx)
        if taylor_data is None:
            return None

        neighborhood = self.neighborhoods[point_idx]
        neighbor_indices = neighborhood["indices"]
        n_neighbors = len(neighbor_indices)

        # Get SVD components for pseudoinverse
        if not taylor_data.get("use_svd", False):
            return None

        sqrt_W = taylor_data["sqrt_W"]
        U = taylor_data["U"]
        S = taylor_data["S"]
        Vt = taylor_data["Vt"]

        # Compute pseudoinverse: (A^T W A)^{-1} A^T W = V S^{-1} U^T sqrt(W)
        # Each column of this matrix gives weights for one derivative coefficient
        # We need the weights that multiply (u_neighbor - u_center) to get derivatives

        # The derivative operator: coeffs = pinv(sqrt_W @ A) @ sqrt_W @ b
        # where b = u_neighbors - u_center
        # So weights_matrix = V @ diag(1/S) @ U^T @ sqrt_W
        S_inv = 1.0 / S
        weights_matrix = Vt.T @ np.diag(S_inv) @ U.T @ sqrt_W  # shape: (n_derivs, n_neighbors)

        # Extract gradient weights (first-order derivatives)
        grad_weights = np.zeros((self.dimension, n_neighbors))
        for k, beta in enumerate(self.multi_indices):
            if sum(beta) == 1:  # First-order derivative
                for d in range(self.dimension):
                    if beta[d] == 1:
                        grad_weights[d, :] = weights_matrix[k, :]

        # Extract Laplacian weights (sum of second-order pure derivatives)
        lap_weights = np.zeros(n_neighbors)
        for k, beta in enumerate(self.multi_indices):
            if sum(beta) == 2:  # Second-order derivative
                for d in range(self.dimension):
                    if beta[d] == 2:  # Pure second derivative d^2/dx_d^2
                        lap_weights += weights_matrix[k, :]

        # For LCR boundary points, rotate gradient weights back to original frame
        if (
            self._use_local_coordinate_rotation
            and hasattr(self, "_boundary_rotations")
            and point_idx in self._boundary_rotations
        ):
            R = self._boundary_rotations[point_idx]
            # grad_weights_orig = R^T @ grad_weights_rotated
            grad_weights = R.T @ grad_weights

        return {
            "neighbor_indices": neighbor_indices,
            "grad_weights": grad_weights,
            "lap_weights": lap_weights,
        }
