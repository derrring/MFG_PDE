"""
GFDM (Generalized Finite Difference Method) Operators for Meshfree Computations.

This module provides spatial differential operators on scattered particles using
the Generalized Finite Difference Method (GFDM). GFDM approximates derivatives
via weighted least squares fitting of local Taylor expansions.

NOTE: Strategy Pattern Refactoring (Issue #524)
-----------------------------------------------
For NEW code, consider using the Strategy Pattern implementation in gfdm_strategies.py:

    from mfg_pde.utils.numerical.gfdm_strategies import (
        TaylorOperator,           # Clean GFDM without ghost particles
        UpwindOperator,           # Flow-biased stencils for advection
        DirectCollocationHandler, # Row Replacement for BC (RECOMMENDED)
        create_operator,          # Factory function
        create_bc_handler,        # Factory function
    )

The new module separates:
- DifferentialOperator: Geometric derivative computation
- BoundaryHandler: BC enforcement (Row Replacement vs Ghost Particles)

GFDMOperator in this file is maintained for backward compatibility.

Primary Interface: GFDMOperator class
-------------------------------------
The recommended way to use GFDM is through the GFDMOperator class, which
precomputes neighbor structure and Taylor matrices for efficient repeated use:

    from mfg_pde.utils.numerical.gfdm_operators import GFDMOperator

    # Create operator (precomputes structure)
    gfdm = GFDMOperator(points, delta=0.1, taylor_order=2)

    # Compute derivatives (reuses precomputed structure)
    grad = gfdm.gradient(u)      # Shape: (n_points, dimension)
    lap = gfdm.laplacian(u)      # Shape: (n_points,)
    hess = gfdm.hessian(u)       # Shape: (n_points, dim, dim)

Mathematical Background:
-----------------------
GFDM approximates spatial derivatives at point x_i using neighbors within radius δ.

Taylor expansion around x_i:
    f(x_j) ≈ f(x_i) + ∇f|_i · (x_j - x_i) + (1/2)(x_j - x_i)^T H|_i (x_j - x_i) + ...

Weighted least squares problem:
    min_coeffs Σ_j w(r_ij) [f(x_j) - Σ_α c_α φ_α(x_j - x_i)]²

Result: Derivative as linear combination of neighbor values
    ∂f/∂x|_i ≈ Σ_j w_ij f(x_j)

Legacy Functions (Deprecated):
-----------------------------
The standalone functions (compute_gradient_gfdm, etc.) are deprecated.
Use GFDMOperator instead for better performance and cleaner API.

References:
-----------
- Benito et al., "Influence of several factors in the generalized finite
  difference method", Applied Mathematical Modelling, 2001.
- Gavete et al., "Improvements of generalized finite difference method and
  comparison with other meshless method", Applied Mathematical Modelling, 2003.

Author: MFG_PDE Development Team
Created: 2025-11-04
Updated: 2025-12-03 (Added GFDMOperator class)
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.spatial import cKDTree

# =============================================================================
# GFDMOperator Class - Primary Interface
# =============================================================================


class GFDMOperator:
    """
    GFDM differential operator with precomputed structure.

    .. deprecated:: 0.17.0
        Use ``TaylorOperator`` from ``gfdm_strategies`` module instead:

            from mfg_pde.utils.numerical.gfdm_strategies import TaylorOperator
            op = TaylorOperator(points, delta=0.1, taylor_order=2)

        For boundary conditions, use ``DirectCollocationHandler`` instead of
        ghost particles.

    This class provides efficient GFDM derivative computation by precomputing
    neighbor structure and Taylor expansion matrices once during initialization.

    Supports optional boundary condition handling with ghost particles for
    no-flux boundary conditions.

    Attributes:
        points: Collocation points, shape (n_points, dimension)
        n_points: Number of collocation points
        dimension: Spatial dimension
        delta: Neighborhood radius
        taylor_order: Order of Taylor expansion (1 or 2)
        multi_indices: List of derivative multi-indices
        boundary_indices: Set of point indices on the boundary
        domain_bounds: Domain bounds for ghost particle placement

    Example:
        >>> import numpy as np
        >>> from mfg_pde.utils.numerical.gfdm_operators import GFDMOperator
        >>>
        >>> # Create 2D grid points
        >>> x = np.linspace(0, 1, 10)
        >>> xx, yy = np.meshgrid(x, x)
        >>> points = np.column_stack([xx.ravel(), yy.ravel()])
        >>>
        >>> # Create operator (basic, no BC handling)
        >>> gfdm = GFDMOperator(points, delta=0.2, taylor_order=2)
        >>>
        >>> # Create operator with no-flux BC handling
        >>> boundary_idx = {0, 9, 90, 99}  # Corner points
        >>> gfdm_bc = GFDMOperator(
        ...     points, delta=0.2, taylor_order=2,
        ...     boundary_indices=boundary_idx,
        ...     domain_bounds=[(0, 1), (0, 1)],
        ...     boundary_type="no_flux"
        ... )
        >>>
        >>> # Compute derivatives of f(x,y) = x² + y²
        >>> u = points[:, 0]**2 + points[:, 1]**2
        >>> grad = gfdm.gradient(u)      # Shape: (100, 2), ≈ [2x, 2y]
        >>> lap = gfdm.laplacian(u)      # Shape: (100,), ≈ 4.0
    """

    def __init__(
        self,
        points: np.ndarray,
        delta: float = 0.1,
        taylor_order: int = 2,
        weight_function: str = "wendland",
        weight_scale: float = 1.0,
        boundary_indices: set[int] | None = None,
        domain_bounds: list[tuple[float, float]] | None = None,
        boundary_type: str | None = None,
        k_neighbors: int | None = None,
        neighborhood_mode: str = "hybrid",
    ):
        """
        Initialize GFDM operator with precomputed structure.

        Args:
            points: Collocation points, shape (n_points, dimension)
            delta: Neighborhood radius for finding neighbors
            taylor_order: Order of Taylor expansion (1 or 2)
            weight_function: Weight function type ("wendland", "gaussian", "uniform")
            weight_scale: Scale parameter for weight function
            boundary_indices: Set of indices of boundary points (optional)
            domain_bounds: List of (min, max) tuples for each dimension (optional)
            boundary_type: Type of boundary condition ("no_flux" or None)
            k_neighbors: Number of neighbors for k-NN mode (auto-computed if None)
            neighborhood_mode: Neighborhood selection strategy:
                - "radius": Use all points within delta (original behavior)
                - "knn": Use exactly k nearest neighbors
                - "hybrid": Use delta, but ensure at least k neighbors (default, most robust)
        """
        warnings.warn(
            "GFDMOperator is deprecated since v0.17.0 and will be removed in v1.0.0. "
            "Use TaylorOperator from gfdm_strategies instead:\n"
            "  from mfg_pde.utils.numerical.gfdm_strategies import TaylorOperator\n"
            "  op = TaylorOperator(points, delta=0.1, taylor_order=2)",
            DeprecationWarning,
            stacklevel=2,
        )
        self.points = np.asarray(points)
        self.n_points, self.dimension = self.points.shape
        self.delta = delta
        self.taylor_order = taylor_order
        self.weight_function = weight_function
        self.weight_scale = weight_scale
        self.neighborhood_mode = neighborhood_mode

        # Compute k_neighbors from Taylor order if not provided
        n_derivatives = self._count_derivatives(self.dimension, taylor_order)
        if k_neighbors is None:
            # Need at least as many neighbors as derivatives for well-posed system
            self.k_neighbors = n_derivatives + 1  # +1 for safety margin
        else:
            self.k_neighbors = max(k_neighbors, n_derivatives)

        # Boundary condition support
        # Handle various input types for boundary_indices
        if boundary_indices is None:
            self.boundary_indices: set[int] = set()
        elif isinstance(boundary_indices, set):
            self.boundary_indices = boundary_indices
        elif isinstance(boundary_indices, np.ndarray):
            self.boundary_indices = {int(x) for x in boundary_indices} if boundary_indices.size > 0 else set()
        else:
            self.boundary_indices = set(boundary_indices)
        self.domain_bounds = domain_bounds
        self.boundary_type = boundary_type

        # Build multi-index set for Taylor expansion
        self.multi_indices = self._build_multi_indices()
        self.n_derivatives = len(self.multi_indices)

        # Precompute neighbor structure (with optional ghost particles)
        self._build_neighborhoods()

        # Precompute Taylor matrices
        self._build_taylor_matrices()

    def _count_derivatives(self, dimension: int, order: int) -> int:
        """Count number of derivatives for given dimension and order."""
        # |B(d,p)| = sum_{k=1}^{p} C(d+k-1, k) = C(d+p, p) - 1
        from math import comb

        return comb(dimension + order, order) - 1

    def _build_multi_indices(self) -> list[tuple[int, ...]]:
        """Generate multi-index set B(d,p) = {β ∈ N^d : 0 < |β| ≤ p}."""
        d, p = self.dimension, self.taylor_order
        multi_indices: list[tuple[int, ...]] = []

        def generate(current: list[int], remaining_dims: int, remaining_order: int):
            if remaining_dims == 0:
                if 0 < sum(current) <= p:
                    multi_indices.append(tuple(current))
                return
            for i in range(remaining_order + 1):
                generate([*current, i], remaining_dims - 1, remaining_order - i)

        generate([], d, p)
        # Sort: first by order |β|, then lexicographically
        multi_indices.sort(key=lambda beta: (sum(beta), beta))
        return multi_indices

    def _build_neighborhoods(self):
        """
        Build neighborhood structure for all points, with optional ghost particles.

        Supports three modes:
        - "radius": All points within delta (original behavior)
        - "knn": Exactly k nearest neighbors
        - "hybrid": Points within delta, but at least k neighbors (most robust)
        """
        # Build KDTree for efficient neighbor queries
        tree = cKDTree(self.points)

        self.neighborhoods: list[dict] = []
        self._hybrid_expanded_count = 0  # Track how many points needed k-NN fallback

        for i in range(self.n_points):
            # Get neighbors based on mode
            if self.neighborhood_mode == "knn":
                # Pure k-NN mode: exactly k nearest
                distances, neighbor_indices = tree.query(self.points[i], k=self.k_neighbors)
                neighbor_indices = np.array(neighbor_indices)
                distances = np.array(distances)
            elif self.neighborhood_mode == "radius":
                # Pure radius mode: all within delta
                neighbor_indices = tree.query_ball_point(self.points[i], self.delta)
                neighbor_indices = np.array(neighbor_indices)
                neighbor_points = self.points[neighbor_indices]
                distances = np.linalg.norm(neighbor_points - self.points[i], axis=1)
            else:
                # Hybrid mode (default): radius, but ensure at least k neighbors
                neighbor_indices = tree.query_ball_point(self.points[i], self.delta)
                neighbor_indices = np.array(neighbor_indices)

                if len(neighbor_indices) < self.k_neighbors:
                    # Not enough neighbors within delta, use k-NN as fallback
                    distances, neighbor_indices = tree.query(self.points[i], k=self.k_neighbors)
                    neighbor_indices = np.array(neighbor_indices)
                    distances = np.array(distances)
                    self._hybrid_expanded_count += 1
                else:
                    neighbor_points = self.points[neighbor_indices]
                    distances = np.linalg.norm(neighbor_points - self.points[i], axis=1)

            # Get neighbor points (for all modes)
            neighbor_points = self.points[neighbor_indices]

            # Check if ghost particles are needed for boundary points
            ghost_particles = []
            ghost_distances = []
            is_boundary = i in self.boundary_indices

            if is_boundary and self.boundary_type == "no_flux" and self.domain_bounds is not None:
                ghost_particles, ghost_distances = self._create_ghost_particles(i)

            # Combine regular neighbors and ghost particles
            if ghost_particles:
                all_points = list(neighbor_points) + ghost_particles
                all_distances = list(distances) + ghost_distances
                all_indices = list(neighbor_indices) + [-1 - j for j in range(len(ghost_particles))]

                self.neighborhoods.append(
                    {
                        "indices": np.array(all_indices),
                        "points": np.array(all_points),
                        "distances": np.array(all_distances),
                        "size": len(all_points),
                        "has_ghost": True,
                        "ghost_count": len(ghost_particles),
                    }
                )
            else:
                self.neighborhoods.append(
                    {
                        "indices": neighbor_indices,
                        "points": neighbor_points,
                        "distances": distances,
                        "size": len(neighbor_indices),
                        "has_ghost": False,
                        "ghost_count": 0,
                    }
                )

        # Report hybrid mode statistics
        if self.neighborhood_mode == "hybrid" and self._hybrid_expanded_count > 0:
            import warnings

            pct = 100.0 * self._hybrid_expanded_count / self.n_points
            warnings.warn(
                f"Hybrid neighborhood: {self._hybrid_expanded_count}/{self.n_points} points ({pct:.1f}%) "
                f"had fewer than k={self.k_neighbors} neighbors within delta={self.delta:.4f}. "
                f"Used k-NN fallback for these points.",
                UserWarning,
                stacklevel=2,
            )

    def _create_ghost_particles(self, point_idx: int) -> tuple[list[np.ndarray], list[float]]:
        """
        Create ghost particles for no-flux boundary conditions.

        For a point on the boundary, creates ghost particles placed slightly
        outside the domain to enforce zero normal derivative.

        Args:
            point_idx: Index of the boundary point

        Returns:
            Tuple of (ghost_particles, ghost_distances)
        """
        ghost_particles = []
        ghost_distances = []

        current_point = self.points[point_idx]
        bounds_array = np.array(self.domain_bounds)  # shape: (d, 2)
        h = 0.1 * self.delta  # Ghost particle distance from boundary

        # Detect which boundaries this point is on
        at_left = np.abs(current_point - bounds_array[:, 0]) < 1e-10
        at_right = np.abs(current_point - bounds_array[:, 1]) < 1e-10

        # Only create ghost particles if within delta range
        if h < self.delta:
            # Create ghost particles for left boundaries
            for d in np.where(at_left)[0]:
                ghost_point = current_point.copy()
                ghost_point[d] = bounds_array[d, 0] - h
                ghost_particles.append(ghost_point)
                ghost_distances.append(h)

            # Create ghost particles for right boundaries
            for d in np.where(at_right)[0]:
                ghost_point = current_point.copy()
                ghost_point[d] = bounds_array[d, 1] + h
                ghost_particles.append(ghost_point)
                ghost_distances.append(h)

        return ghost_particles, ghost_distances

    def _build_taylor_matrices(self):
        """Precompute Taylor expansion matrices for all points."""
        self.taylor_matrices: list[dict | None] = []

        for i in range(self.n_points):
            neighborhood = self.neighborhoods[i]
            n_neighbors = neighborhood["size"]

            # Need at least n_derivatives neighbors for well-posed system
            if n_neighbors < self.n_derivatives:
                self.taylor_matrices.append(None)
                continue

            # Build Taylor expansion matrix A
            A = np.zeros((n_neighbors, self.n_derivatives))
            center_point = self.points[i]
            neighbor_points = neighborhood["points"]

            for j, neighbor_point in enumerate(neighbor_points):
                delta_x = neighbor_point - center_point

                for k, beta in enumerate(self.multi_indices):
                    # Compute (x - x_center)^β / β!
                    term = 1.0
                    factorial = 1.0
                    for dim in range(self.dimension):
                        if beta[dim] > 0:
                            term *= delta_x[dim] ** beta[dim]
                            factorial *= math.factorial(beta[dim])
                    A[j, k] = term / factorial

            # Compute weights
            weights = self._compute_weights(neighborhood["distances"])
            W = np.diag(weights)
            sqrt_W = np.sqrt(W)

            # SVD decomposition for numerical stability
            WA = sqrt_W @ A
            try:
                U, S, Vt = np.linalg.svd(WA, full_matrices=False)
                # Truncate small singular values
                tol = 1e-12
                rank = np.sum(tol < S)

                self.taylor_matrices.append(
                    {
                        "A": A,
                        "W": W,
                        "sqrt_W": sqrt_W,
                        "U": U[:, :rank],
                        "S": S[:rank],
                        "Vt": Vt[:rank, :],
                        "rank": rank,
                    }
                )
            except np.linalg.LinAlgError:
                self.taylor_matrices.append(None)

    def _compute_weights(self, distances: np.ndarray) -> np.ndarray:
        """Compute weights based on distance."""
        if self.weight_function == "wendland":
            # Wendland C^4 kernel
            q = distances / self.delta
            q = np.clip(q, 0, 1)
            return (1 - q) ** 6 * (35 * q**2 + 18 * q + 3) / 3

        elif self.weight_function == "gaussian":
            return np.exp(-((distances / self.weight_scale) ** 2))

        elif self.weight_function == "uniform":
            return np.ones_like(distances)

        else:
            raise ValueError(f"Unknown weight function: {self.weight_function}")

    def _approximate_derivatives_at_point(self, u: np.ndarray, point_idx: int) -> dict[tuple[int, ...], float]:
        """Compute all Taylor derivatives at a single point."""
        taylor_data = self.taylor_matrices[point_idx]
        if taylor_data is None:
            return {}

        neighborhood = self.neighborhoods[point_idx]
        neighbor_indices = neighborhood["indices"]

        # Function values - handle ghost particles for no-flux BC
        u_center = u[point_idx]
        u_neighbors = self._get_neighbor_values(u, point_idx, neighbor_indices)
        b = u_neighbors - u_center

        # Solve using SVD
        sqrt_W = taylor_data["sqrt_W"]
        U = taylor_data["U"]
        S = taylor_data["S"]
        Vt = taylor_data["Vt"]

        Wb = sqrt_W @ b
        UT_Wb = U.T @ Wb
        S_inv_UT_Wb = UT_Wb / S
        coeffs = Vt.T @ S_inv_UT_Wb

        # Map to multi-indices
        return {beta: coeffs[k] for k, beta in enumerate(self.multi_indices)}

    def _get_neighbor_values(self, u: np.ndarray, point_idx: int, neighbor_indices: np.ndarray) -> np.ndarray:
        """
        Get function values at neighbors, handling ghost particles.

        For no-flux BC, ghost particles use mirrored values (same as center point).
        This enforces zero normal derivative at the boundary.
        """
        u_neighbors = np.zeros(len(neighbor_indices))

        for j, idx in enumerate(neighbor_indices):
            if idx >= 0:
                # Regular neighbor
                u_neighbors[j] = u[idx]
            else:
                # Ghost particle - use center point value for no-flux BC
                # This makes the function appear symmetric across the boundary,
                # resulting in zero normal derivative
                u_neighbors[j] = u[point_idx]

        return u_neighbors

    def gradient(self, u: np.ndarray) -> np.ndarray:
        """
        Compute gradient ∇u at all points.

        Args:
            u: Function values at points, shape (n_points,)

        Returns:
            Gradient array, shape (n_points, dimension).
        """
        grad = np.zeros((self.n_points, self.dimension))

        for i in range(self.n_points):
            derivs = self._approximate_derivatives_at_point(u, i)
            for d in range(self.dimension):
                # Multi-index for ∂/∂x_d: e.g., (1, 0) for ∂/∂x in 2D
                multi_idx = tuple(1 if j == d else 0 for j in range(self.dimension))
                grad[i, d] = derivs.get(multi_idx, 0.0)

        return grad

    def laplacian(self, u: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian Δu at all points.

        Args:
            u: Function values at points, shape (n_points,)

        Returns:
            Laplacian array, shape (n_points,).
        """
        lap = np.zeros(self.n_points)

        for i in range(self.n_points):
            derivs = self._approximate_derivatives_at_point(u, i)
            for d in range(self.dimension):
                # Multi-index for ∂²/∂x_d²: e.g., (2, 0) for ∂²/∂x² in 2D
                multi_idx = tuple(2 if j == d else 0 for j in range(self.dimension))
                lap[i] += derivs.get(multi_idx, 0.0)

        return lap

    def hessian(self, u: np.ndarray) -> np.ndarray:
        """
        Compute Hessian matrix at all points.

        Args:
            u: Function values at points, shape (n_points,)

        Returns:
            Hessian array, shape (n_points, dimension, dimension).
        """
        hess = np.zeros((self.n_points, self.dimension, self.dimension))

        for i in range(self.n_points):
            derivs = self._approximate_derivatives_at_point(u, i)
            for d1 in range(self.dimension):
                for d2 in range(self.dimension):
                    # Build multi-index for ∂²/∂x_d1∂x_d2
                    multi_idx_list = [0] * self.dimension
                    multi_idx_list[d1] += 1
                    multi_idx_list[d2] += 1
                    multi_idx = tuple(multi_idx_list)
                    hess[i, d1, d2] = derivs.get(multi_idx, 0.0)

        return hess

    def divergence(self, v: np.ndarray, m: np.ndarray | None = None) -> np.ndarray:
        """
        Compute divergence ∇·v or ∇·(m·v) at all points.

        Args:
            v: Vector field at points, shape (n_points, dimension)
            m: Optional density field, shape (n_points,). If provided,
               computes ∇·(m·v) instead of ∇·v.

        Returns:
            Divergence array, shape (n_points,).
        """
        div = np.zeros(self.n_points)

        for d in range(self.dimension):
            if m is not None:
                # ∇·(m·v) = m·∇·v + v·∇m (product rule)
                flux = m * v[:, d]
                div += self.gradient(flux)[:, d]
            else:
                # ∇·v = Σ_d ∂v_d/∂x_d
                div += self.gradient(v[:, d])[:, d]

        return div

    def all_derivatives(self, u: np.ndarray) -> dict[int, dict[tuple[int, ...], float]]:
        """
        Compute all Taylor derivatives at all points.

        Args:
            u: Function values at points, shape (n_points,)

        Returns:
            Dictionary mapping point index to derivative dictionary.
        """
        return {i: self._approximate_derivatives_at_point(u, i) for i in range(self.n_points)}

    # =========================================================================
    # Accessors for composition by HJBGFDMSolver
    # =========================================================================

    def get_neighborhood(self, point_idx: int) -> dict:
        """Get neighborhood data for a point (for use by HJBGFDMSolver)."""
        return self.neighborhoods[point_idx]

    def get_taylor_data(self, point_idx: int) -> dict | None:
        """Get Taylor matrix data for a point (for use by HJBGFDMSolver)."""
        return self.taylor_matrices[point_idx]

    def get_derivative_weights(self, point_idx: int) -> dict | None:
        """
        Get derivative sensitivity weights for analytic Jacobian computation.

        Returns the weights that map perturbations in neighbor u values to
        perturbations in derivatives at point i:
            δ(∂u/∂x_d)|_i = Σ_j w^grad_{d,j} δu_j
            δ(Δu)|_i = Σ_j w^lap_j δu_j

        Args:
            point_idx: Index of the center point

        Returns:
            Dictionary with:
            - "neighbor_indices": indices of neighbors (including center)
            - "grad_weights": shape (dimension, n_neighbors), gradient sensitivity
            - "lap_weights": shape (n_neighbors,), Laplacian sensitivity
            - "center_idx_in_neighbors": position of center in neighbor list
            Or None if Taylor data not available.
        """
        taylor_data = self.taylor_matrices[point_idx]
        if taylor_data is None:
            return None

        neighborhood = self.neighborhoods[point_idx]
        neighbor_indices = neighborhood["indices"]
        n_neighbors = len(neighbor_indices)

        # Compute weight matrix M = Vt.T @ diag(1/S) @ U.T @ sqrt_W
        # Such that coeffs = M @ b, where b = u_neighbors - u_center
        sqrt_W = taylor_data["sqrt_W"]
        U = taylor_data["U"]
        S = taylor_data["S"]
        Vt = taylor_data["Vt"]

        # M maps b (neighbor deviations) to Taylor coefficients
        M = Vt.T @ np.diag(1.0 / S) @ U.T @ sqrt_W  # shape: (n_derivatives, n_neighbors)

        # Gradient weights: for each dimension d, find the multi-index (0,...,1,...,0)
        grad_weights = np.zeros((self.dimension, n_neighbors))
        for d in range(self.dimension):
            multi_idx = tuple(1 if j == d else 0 for j in range(self.dimension))
            if multi_idx in self.multi_indices:
                k = self.multi_indices.index(multi_idx)
                # Weights for neighbors: M[k, :]
                # Weight for center: -sum(M[k, :]) (since b = u_neighbors - u_center)
                grad_weights[d, :] = M[k, :]

        # Laplacian weights: sum of second derivatives
        lap_weights = np.zeros(n_neighbors)
        for d in range(self.dimension):
            multi_idx = tuple(2 if j == d else 0 for j in range(self.dimension))
            if multi_idx in self.multi_indices:
                k = self.multi_indices.index(multi_idx)
                lap_weights += M[k, :]

        # Find center index in neighbors (if present, usually is)
        center_in_neighbors = -1
        for idx, n_idx in enumerate(neighbor_indices):
            if n_idx == point_idx:
                center_in_neighbors = idx
                break

        return {
            "neighbor_indices": neighbor_indices,
            "grad_weights": grad_weights,
            "lap_weights": lap_weights,
            "center_idx_in_neighbors": center_in_neighbors,
            "weight_matrix": M,  # Full weight matrix for advanced use
        }

    def get_multi_indices(self) -> list[tuple[int, ...]]:
        """Get the multi-index set used for derivatives."""
        return self.multi_indices

    def approximate_derivatives_at_point(self, u: np.ndarray, point_idx: int) -> dict[tuple[int, ...], float]:
        """
        Public accessor for derivative computation at a single point.

        This is the core GFDM computation without QP constraints.
        HJBGFDMSolver can use this and add QP logic on top.
        """
        return self._approximate_derivatives_at_point(u, point_idx)


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    """Smoke test for GFDMOperator."""
    print("Testing GFDMOperator...")

    # 1D test
    print("\n[1D] Testing 1D operator...")
    x_1d = np.linspace(0, 1, 21).reshape(-1, 1)
    gfdm_1d = GFDMOperator(x_1d, delta=0.15, taylor_order=2)

    # f(x) = x^2 -> df/dx = 2x, d^2f/dx^2 = 2
    u_1d = x_1d[:, 0] ** 2
    grad_1d = gfdm_1d.gradient(u_1d)
    lap_1d = gfdm_1d.laplacian(u_1d)

    # Check interior points (boundary points may have fewer neighbors)
    interior = slice(3, -3)
    grad_error = np.mean(np.abs(grad_1d[interior, 0] - 2 * x_1d[interior, 0]))
    lap_error = np.mean(np.abs(lap_1d[interior] - 2.0))
    print(f"     Gradient error: {grad_error:.2e}")
    print(f"     Laplacian error: {lap_error:.2e}")
    assert grad_error < 0.1, f"1D gradient error too large: {grad_error}"
    assert lap_error < 0.5, f"1D Laplacian error too large: {lap_error}"

    # 2D test
    print("\n[2D] Testing 2D operator...")
    x_2d = np.linspace(0, 1, 10)
    xx, yy = np.meshgrid(x_2d, x_2d)
    points_2d = np.column_stack([xx.ravel(), yy.ravel()])

    gfdm_2d = GFDMOperator(points_2d, delta=0.2, taylor_order=2)
    print(f"     Points: {gfdm_2d.n_points}, Delta: {gfdm_2d.delta}")
    print(f"     Multi-indices: {gfdm_2d.multi_indices}")

    # f(x,y) = x^2 + y^2 -> gradient = [2x, 2y], laplacian = 4
    u_2d = points_2d[:, 0] ** 2 + points_2d[:, 1] ** 2
    grad_2d = gfdm_2d.gradient(u_2d)
    lap_2d = gfdm_2d.laplacian(u_2d)
    hess_2d = gfdm_2d.hessian(u_2d)

    # Check interior points (find points not near boundary)
    interior_mask = (
        (points_2d[:, 0] > 0.15) & (points_2d[:, 0] < 0.85) & (points_2d[:, 1] > 0.15) & (points_2d[:, 1] < 0.85)
    )

    expected_grad_x = 2 * points_2d[interior_mask, 0]
    expected_grad_y = 2 * points_2d[interior_mask, 1]

    grad_error_x = np.mean(np.abs(grad_2d[interior_mask, 0] - expected_grad_x))
    grad_error_y = np.mean(np.abs(grad_2d[interior_mask, 1] - expected_grad_y))
    lap_error_2d = np.mean(np.abs(lap_2d[interior_mask] - 4.0))

    print(f"     Gradient errors: dx={grad_error_x:.2e}, dy={grad_error_y:.2e}")
    print(f"     Laplacian error: {lap_error_2d:.2e}")

    # Hessian check: H = [[2, 0], [0, 2]]
    hess_diag_error = np.mean(np.abs(hess_2d[interior_mask, 0, 0] - 2.0))
    hess_diag_error += np.mean(np.abs(hess_2d[interior_mask, 1, 1] - 2.0))
    print(f"     Hessian diagonal error: {hess_diag_error:.2e}")

    # Test boundary condition support
    print("\n[BC] Testing no-flux boundary conditions...")
    x_bc = np.linspace(0, 1, 10)
    xx_bc, yy_bc = np.meshgrid(x_bc, x_bc)
    points_bc = np.column_stack([xx_bc.ravel(), yy_bc.ravel()])
    n_side = 10

    # Identify boundary points (on edges of unit square)
    boundary_idx = set()
    for i in range(len(points_bc)):
        x, y = points_bc[i]
        if abs(x) < 1e-10 or abs(x - 1) < 1e-10 or abs(y) < 1e-10 or abs(y - 1) < 1e-10:
            boundary_idx.add(i)

    print(f"     Boundary points: {len(boundary_idx)}/{len(points_bc)}")

    # Create operator with no-flux BC
    gfdm_bc = GFDMOperator(
        points_bc,
        delta=0.25,
        taylor_order=2,
        boundary_indices=boundary_idx,
        domain_bounds=[(0, 1), (0, 1)],
        boundary_type="no_flux",
    )

    # Check that boundary points have ghost particles
    n_with_ghost = sum(1 for i in range(gfdm_bc.n_points) if gfdm_bc.neighborhoods[i].get("has_ghost", False))
    print(f"     Points with ghost particles: {n_with_ghost}")
    assert n_with_ghost > 0, "Expected some boundary points to have ghost particles"

    # Test derivative computation still works
    u_bc = points_bc[:, 0] ** 2 + points_bc[:, 1] ** 2
    grad_bc = gfdm_bc.gradient(u_bc)
    lap_bc = gfdm_bc.laplacian(u_bc)

    # Check that we get finite results everywhere
    assert np.all(np.isfinite(grad_bc)), "Gradient contains non-finite values"
    assert np.all(np.isfinite(lap_bc)), "Laplacian contains non-finite values"
    print("     All derivatives finite: OK")

    print("\nGFDMOperator smoke tests passed!")


# NOTE: Legacy standalone functions have been removed (v0.17.0).
# Use GFDMOperator class instead:
#
#   from mfg_pde.utils.numerical.gfdm_operators import GFDMOperator
#
#   gfdm = GFDMOperator(points, delta=0.1, taylor_order=2)
#   grad = gfdm.gradient(u)       # replaces compute_gradient_gfdm
#   lap = gfdm.laplacian(u)       # replaces compute_laplacian_gfdm
#   hess = gfdm.hessian(u)        # replaces compute_hessian_gfdm
#   div = gfdm.divergence(v, m)   # replaces compute_divergence_gfdm
#
# Removed functions:
#   - find_neighbors_kdtree
#   - gaussian_rbf_weight
#   - compute_laplacian_gfdm
#   - compute_divergence_gfdm
#   - compute_gradient_gfdm
#   - compute_hessian_gfdm
#   - compute_curl_gfdm
#   - compute_directional_derivative_gfdm
#   - compute_kernel_density_gfdm
#   - compute_vector_laplacian_gfdm
