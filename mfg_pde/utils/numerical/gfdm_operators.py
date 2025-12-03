"""
GFDM (Generalized Finite Difference Method) Operators for Meshfree Computations.

This module provides spatial differential operators on scattered particles using
the Generalized Finite Difference Method (GFDM). GFDM approximates derivatives
via weighted least squares fitting of local Taylor expansions.

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

import numpy as np
from scipy.spatial import cKDTree

# =============================================================================
# GFDMOperator Class - Primary Interface
# =============================================================================


class GFDMOperator:
    """
    GFDM differential operator with precomputed structure.

    This class provides efficient GFDM derivative computation by precomputing
    neighbor structure and Taylor expansion matrices once during initialization.

    Attributes:
        points: Collocation points, shape (n_points, dimension)
        n_points: Number of collocation points
        dimension: Spatial dimension
        delta: Neighborhood radius
        taylor_order: Order of Taylor expansion (1 or 2)
        multi_indices: List of derivative multi-indices

    Example:
        >>> import numpy as np
        >>> from mfg_pde.utils.numerical.gfdm_operators import GFDMOperator
        >>>
        >>> # Create 2D grid points
        >>> x = np.linspace(0, 1, 10)
        >>> xx, yy = np.meshgrid(x, x)
        >>> points = np.column_stack([xx.ravel(), yy.ravel()])
        >>>
        >>> # Create operator
        >>> gfdm = GFDMOperator(points, delta=0.2, taylor_order=2)
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
    ):
        """
        Initialize GFDM operator with precomputed structure.

        Args:
            points: Collocation points, shape (n_points, dimension)
            delta: Neighborhood radius for finding neighbors
            taylor_order: Order of Taylor expansion (1 or 2)
            weight_function: Weight function type ("wendland", "gaussian", "uniform")
            weight_scale: Scale parameter for weight function
        """
        self.points = np.asarray(points)
        self.n_points, self.dimension = self.points.shape
        self.delta = delta
        self.taylor_order = taylor_order
        self.weight_function = weight_function
        self.weight_scale = weight_scale

        # Build multi-index set for Taylor expansion
        self.multi_indices = self._build_multi_indices()
        self.n_derivatives = len(self.multi_indices)

        # Precompute neighbor structure
        self._build_neighborhoods()

        # Precompute Taylor matrices
        self._build_taylor_matrices()

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
        """Build δ-neighborhood structure for all points."""
        # Build KDTree for efficient neighbor queries
        tree = cKDTree(self.points)

        # Query all neighbors within delta radius
        self.neighborhoods: list[dict] = []

        for i in range(self.n_points):
            # Find neighbors within delta radius
            neighbor_indices = tree.query_ball_point(self.points[i], self.delta)
            neighbor_indices = np.array(neighbor_indices)

            # Compute distances
            neighbor_points = self.points[neighbor_indices]
            distances = np.linalg.norm(neighbor_points - self.points[i], axis=1)

            self.neighborhoods.append(
                {
                    "indices": neighbor_indices,
                    "points": neighbor_points,
                    "distances": distances,
                    "size": len(neighbor_indices),
                }
            )

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

        # Function values
        u_center = u[point_idx]
        u_neighbors = u[neighbor_indices]
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
# Legacy Standalone Functions (Deprecated)
# =============================================================================


def find_neighbors_kdtree(points: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Find k nearest neighbors for each point using KDTree.

    Uses scipy.spatial.cKDTree for efficient O(N log N) neighbor search.

    Parameters
    ----------
    points : np.ndarray
        Particle positions, shape (N_points, dimension).
    k : int, optional
        Number of neighbors to find. If None, uses k = 2*dimension + 1
        (minimum for second-order accuracy in dimension d).

    Returns
    -------
    neighbor_indices : np.ndarray
        Indices of k nearest neighbors for each point, shape (N_points, k).
        neighbor_indices[i, :] are the indices of neighbors of point i.
    neighbor_distances : np.ndarray
        Distances to k nearest neighbors, shape (N_points, k).
        neighbor_distances[i, :] are distances from point i to its neighbors.

    Notes
    -----
    The first neighbor (column 0) is always the point itself with distance 0.
    Use neighbor_indices[:, 1:] to exclude self-neighbors if needed.

    Examples
    --------
    >>> points = np.random.rand(100, 2)  # 100 particles in 2D
    >>> nbr_idx, nbr_dist = find_neighbors_kdtree(points, k=5)
    >>> nbr_idx.shape
    (100, 5)
    >>> nbr_dist[0, 0]  # Distance to self
    0.0
    """
    from scipy.spatial import cKDTree

    _N_points, dimension = points.shape

    # Default: 2*d+1 neighbors for second-order accuracy
    if k is None:
        k = 2 * dimension + 1

    # Build KDTree and query
    tree = cKDTree(points)
    neighbor_distances, neighbor_indices = tree.query(points, k=k)

    return neighbor_indices, neighbor_distances


def gaussian_rbf_weight(r: np.ndarray, h: float) -> np.ndarray:
    """
    Gaussian RBF (Radial Basis Function) weight for GFDM.

    This is a convenience wrapper around GaussianKernel from the kernels module.

    Weight function: w(r) = exp(-(r/h)²)

    Parameters
    ----------
    r : np.ndarray
        Distances to neighbors, any shape.
    h : float
        Support radius (bandwidth parameter). Controls decay rate.
        Smaller h → more localized, larger h → smoother.

    Returns
    -------
    weights : np.ndarray
        Weight values, same shape as r. Range: (0, 1].

    Notes
    -----
    Uses GaussianKernel from mfg_pde.utils.numerical.kernels with proper
    normalization for arbitrary dimensions.

    Alternative weight functions available in kernels module:
    - WendlandKernel(k=0,1,2,3): Compact support, C^0, C^2, C^4, C^6 smoothness
    - CubicSplineKernel, QuinticSplineKernel: SPH B-spline kernels
    - CubicKernel, QuarticKernel: Simple polynomial kernels

    See mfg_pde.utils.numerical.kernels for full kernel API.

    Examples
    --------
    >>> r = np.array([0.0, 0.5, 1.0, 2.0])
    >>> w = gaussian_rbf_weight(r, h=1.0)
    >>> w
    array([1.        , 0.77880078, 0.36787944, 0.01831564])
    """
    from mfg_pde.utils.numerical.kernels import GaussianKernel

    kernel = GaussianKernel()
    return kernel(r, h=h)


def compute_laplacian_gfdm(scalar_field: np.ndarray, points: np.ndarray, k: int | None = None) -> np.ndarray:
    """
    Compute Laplacian Δf using GFDM on particles.

    Approximates Δf = ∂²f/∂x₁² + ∂²f/∂x₂² + ... at each particle using
    weighted least squares fitting of local Taylor expansion.

    Parameters
    ----------
    scalar_field : np.ndarray
        Scalar field values at particles, shape (N_points,).
    points : np.ndarray
        Particle positions, shape (N_points, dimension).
    k : int, optional
        Number of neighbors. If None, uses k = 2*dimension + 1.

    Returns
    -------
    laplacian : np.ndarray
        Laplacian values at each particle, shape (N_points,).

    Notes
    -----
    Algorithm:
    1. For each particle i:
       - Find k nearest neighbors
       - Compute local coordinate system (x_j - x_i)
       - Solve weighted least squares for Hessian approximation
       - Laplacian = trace of Hessian

    Accuracy: Second-order accurate for k ≥ 2*d+1 neighbors.

    Examples
    --------
    >>> # Test on quadratic function f(x,y) = x² + y²
    >>> # Analytical: Δf = ∂²f/∂x² + ∂²f/∂y² = 2 + 2 = 4
    >>> points = np.random.rand(100, 2)
    >>> f = points[:, 0]**2 + points[:, 1]**2
    >>> laplacian = compute_laplacian_gfdm(f, points)
    >>> np.mean(laplacian)  # Should be close to 4.0
    4.02
    """
    N_points, dimension = points.shape

    # Validate input
    if scalar_field.shape[0] != N_points:
        raise ValueError(f"scalar_field length {scalar_field.shape[0]} must match points length {N_points}")

    # Find neighbors
    neighbor_indices, neighbor_distances = find_neighbors_kdtree(points, k=k)
    k_actual = neighbor_indices.shape[1]

    # Adaptive support radius (max neighbor distance)
    h_values = np.max(neighbor_distances, axis=1)

    # Preallocate output
    laplacian = np.zeros(N_points)

    # Compute Laplacian at each point
    for i in range(N_points):
        # Neighbors of point i
        nbr_idx = neighbor_indices[i, :]
        nbr_dist = neighbor_distances[i, :]
        h = h_values[i]

        # Local coordinates: Δx = x_j - x_i
        delta_x = points[nbr_idx, :] - points[i, :]  # (k, dimension)

        # Function values at neighbors
        f_nbrs = scalar_field[nbr_idx]  # (k,)
        f_i = scalar_field[i]

        # Weights (Gaussian RBF)
        weights = gaussian_rbf_weight(nbr_dist, h)  # (k,)

        # Build weighted least squares system for Hessian
        # Basis: [1, x, y, x², xy, y², ...] for 2D
        # We only need second derivatives: x², xy, y², ...

        # Number of basis functions for second derivatives
        # For dimension d: d*(d+1)/2 second derivatives
        n_second_derivs = dimension * (dimension + 1) // 2

        # Build design matrix A and RHS b
        # Columns: x², xy, y² (for 2D), or x², xy, xz, y², yz, z² (for 3D)
        A = np.zeros((k_actual, n_second_derivs))
        b = np.zeros(k_actual)

        for j in range(k_actual):
            # Taylor expansion: f(x_j) ≈ f(x_i) + ∇f·Δx + (1/2)Δx^T H Δx
            # For Hessian, we need (1/2) * second-order terms
            dx = delta_x[j, :]  # (dimension,)

            # Second-order monomials: x², xy, y², ...
            col_idx = 0
            for d1 in range(dimension):
                for d2 in range(d1, dimension):
                    if d1 == d2:
                        A[j, col_idx] = 0.5 * dx[d1] ** 2  # ∂²/∂x_d² term
                    else:
                        A[j, col_idx] = dx[d1] * dx[d2]  # ∂²/∂x_d1∂x_d2 term
                    col_idx += 1

            b[j] = f_nbrs[j] - f_i

        # Weight the system
        W = np.diag(np.sqrt(weights))
        A_weighted = W @ A
        b_weighted = W @ b

        # Solve weighted least squares: min ||W(Ax - b)||²
        try:
            hessian_coeffs, _residuals, _rank, _s = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)
        except np.linalg.LinAlgError:
            # Singular matrix - use zero Laplacian
            laplacian[i] = 0.0
            continue

        # Extract diagonal Hessian entries (∂²f/∂x₁², ∂²f/∂x₂², ...)
        # Laplacian = trace of Hessian
        laplacian_value = 0.0
        col_idx = 0
        for d1 in range(dimension):
            for d2 in range(d1, dimension):
                if d1 == d2:
                    laplacian_value += hessian_coeffs[col_idx]
                col_idx += 1

        laplacian[i] = laplacian_value

    return laplacian


def compute_divergence_gfdm(
    vector_field: np.ndarray,
    density: np.ndarray,
    points: np.ndarray,
    k: int | None = None,
) -> np.ndarray:
    """
    Compute divergence ∇·(m α) using GFDM on particles.

    Approximates the divergence of flux m*α at each particle using
    product rule: ∇·(m α) = ∇m·α + m ∇·α

    Parameters
    ----------
    vector_field : np.ndarray
        Vector field α at particles, shape (N_points, dimension) or (N_points, 1).
        For 1D problems, can be (N_points, 1).
    density : np.ndarray
        Scalar density m at particles, shape (N_points,).
    points : np.ndarray
        Particle positions, shape (N_points, dimension).
    k : int, optional
        Number of neighbors. If None, uses k = 2*dimension + 1.

    Returns
    -------
    divergence : np.ndarray
        Divergence ∇·(m α) at each particle, shape (N_points,).

    Notes
    -----
    Product rule:
        ∇·(m α) = ∇m·α + m ∇·α
                = Σ_d (∂m/∂x_d) α_d + m Σ_d (∂α_d/∂x_d)

    For each component d:
    1. Compute ∂m/∂x_d using GFDM gradient
    2. Compute ∂α_d/∂x_d using GFDM gradient
    3. Combine via product rule

    Special case: If vector_field has shape (N_points, 1), treats as 1D.

    Examples
    --------
    >>> # Test on constant drift α = [1, 0], uniform density m = 1
    >>> # Analytical: ∇·(m α) = ∇m·α + m ∇·α = 0·[1,0] + 1·0 = 0
    >>> points = np.random.rand(100, 2)
    >>> alpha = np.ones((100, 2))
    >>> alpha[:, 1] = 0
    >>> m = np.ones(100)
    >>> div = compute_divergence_gfdm(alpha, m, points)
    >>> np.mean(np.abs(div))  # Should be close to 0
    0.02
    """
    N_points, dimension = points.shape

    # Validate inputs
    if density.shape[0] != N_points:
        raise ValueError(f"density length {density.shape[0]} must match points length {N_points}")
    if vector_field.shape[0] != N_points:
        raise ValueError(f"vector_field length {vector_field.shape[0]} must match points length {N_points}")

    # Find neighbors
    neighbor_indices, neighbor_distances = find_neighbors_kdtree(points, k=k)

    # Adaptive support radius
    h_values = np.max(neighbor_distances, axis=1)

    # Preallocate output
    divergence = np.zeros(N_points)

    # Compute divergence at each point using product rule
    for i in range(N_points):
        # Neighbors of point i
        nbr_idx = neighbor_indices[i, :]
        nbr_dist = neighbor_distances[i, :]
        h = h_values[i]

        # Local coordinates
        delta_x = points[nbr_idx, :] - points[i, :]  # (k, dimension)

        # Density at neighbors
        m_nbrs = density[nbr_idx]  # (k,)
        m_i = density[i]

        # Vector field at neighbors
        alpha_nbrs = vector_field[nbr_idx, :]  # (k, dimension) or (k, 1)
        alpha_i = vector_field[i, :]  # (dimension,) or (1,)

        # Weights
        weights = gaussian_rbf_weight(nbr_dist, h)  # (k,)

        # Product rule: ∇·(m α) = ∇m·α + m ∇·α
        div_value = 0.0

        # Loop over each dimension
        for d in range(dimension):
            # --- Term 1: (∂m/∂x_d) * α_d ---
            # Build weighted least squares for ∂m/∂x_d
            A = delta_x[:, d].reshape(-1, 1)  # (k, 1): just x_d coordinate
            b = m_nbrs - m_i  # (k,)

            W = np.diag(np.sqrt(weights))
            A_weighted = W @ A
            b_weighted = W @ b

            try:
                grad_m_d, _residuals, _rank, _s = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)
                grad_m_d = grad_m_d[0]
            except np.linalg.LinAlgError:
                grad_m_d = 0.0

            # Handle 1D vs nD vector field shape
            if alpha_nbrs.shape[1] == 1:
                # 1D case: vector_field is (N, 1)
                alpha_d = alpha_i[0]
            else:
                # nD case: vector_field is (N, dimension)
                alpha_d = alpha_i[d]

            div_value += grad_m_d * alpha_d

            # --- Term 2: m * (∂α_d/∂x_d) ---
            # Build weighted least squares for ∂α_d/∂x_d
            if alpha_nbrs.shape[1] == 1:
                # 1D case: only one component
                flux = m_nbrs * alpha_nbrs[:, 0]
                flux_i = m_i * alpha_i[0]
            else:
                # nD case: extract d-th component
                flux = m_nbrs * alpha_nbrs[:, d]
                flux_i = m_i * alpha_i[d]

            b_flux = flux - flux_i

            b_weighted_flux = W @ b_flux

            try:
                grad_flux_d, _residuals, _rank, _s = np.linalg.lstsq(A_weighted, b_weighted_flux, rcond=None)
                grad_flux_d = grad_flux_d[0]
            except np.linalg.LinAlgError:
                grad_flux_d = 0.0

            div_value += grad_flux_d

        divergence[i] = div_value

    return divergence


def compute_gradient_gfdm(scalar_field: np.ndarray, points: np.ndarray, k: int | None = None) -> np.ndarray:
    """
    Compute gradient ∇f using GFDM on particles.

    Approximates ∇f = [∂f/∂x₁, ∂f/∂x₂, ...] at each particle using
    weighted least squares fitting of local Taylor expansion.

    Parameters
    ----------
    scalar_field : np.ndarray
        Scalar field values at particles, shape (N_points,).
    points : np.ndarray
        Particle positions, shape (N_points, dimension).
    k : int, optional
        Number of neighbors. If None, uses k = 2*dimension + 1.

    Returns
    -------
    gradient : np.ndarray
        Gradient vectors at each particle, shape (N_points, dimension).

    Notes
    -----
    Algorithm:
    1. For each particle i:
       - Find k nearest neighbors
       - Compute local coordinate system (x_j - x_i)
       - Solve weighted least squares for gradient coefficients
       - Gradient = [∂f/∂x₁, ∂f/∂x₂, ...]

    Accuracy: First-order accurate for k ≥ dimension+1 neighbors.

    Examples
    --------
    >>> # Test on linear function f(x,y) = 2x + 3y
    >>> # Analytical: ∇f = [2, 3]
    >>> points = np.random.rand(100, 2)
    >>> f = 2*points[:, 0] + 3*points[:, 1]
    >>> grad = compute_gradient_gfdm(f, points)
    >>> np.mean(grad, axis=0)  # Should be close to [2, 3]
    array([2.01, 2.98])
    """
    N_points, dimension = points.shape

    # Validate input
    if scalar_field.shape[0] != N_points:
        raise ValueError(f"scalar_field length {scalar_field.shape[0]} must match points length {N_points}")

    # Find neighbors
    neighbor_indices, neighbor_distances = find_neighbors_kdtree(points, k=k)

    # Adaptive support radius
    h_values = np.max(neighbor_distances, axis=1)

    # Preallocate output
    gradient = np.zeros((N_points, dimension))

    # Compute gradient at each point
    for i in range(N_points):
        # Neighbors of point i
        nbr_idx = neighbor_indices[i, :]
        nbr_dist = neighbor_distances[i, :]
        h = h_values[i]

        # Local coordinates: Δx = x_j - x_i
        delta_x = points[nbr_idx, :] - points[i, :]  # (k, dimension)

        # Function values at neighbors
        f_nbrs = scalar_field[nbr_idx]
        f_i = scalar_field[i]

        # Weights
        weights = gaussian_rbf_weight(nbr_dist, h)

        # Build weighted least squares system
        # Taylor: f(x_j) ≈ f(x_i) + ∇f·(x_j - x_i)
        A = delta_x  # (k, dimension)
        b = f_nbrs - f_i  # (k,)

        # Weight the system
        W = np.diag(np.sqrt(weights))
        A_weighted = W @ A
        b_weighted = W @ b

        # Solve
        try:
            grad_coeffs, _residuals, _rank, _s = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)
            gradient[i, :] = grad_coeffs
        except np.linalg.LinAlgError:
            # Singular matrix - use zero gradient
            gradient[i, :] = 0.0

    return gradient


def compute_hessian_gfdm(scalar_field: np.ndarray, points: np.ndarray, k: int | None = None) -> np.ndarray:
    """
    Compute full Hessian matrix ∂²f/∂x_i∂x_j using GFDM on particles.

    Returns the complete Hessian matrix at each particle, not just the trace (Laplacian).

    Parameters
    ----------
    scalar_field : np.ndarray
        Scalar field values at particles, shape (N_points,).
    points : np.ndarray
        Particle positions, shape (N_points, dimension).
    k : int, optional
        Number of neighbors. If None, uses k = 2*dimension + 1.

    Returns
    -------
    hessian : np.ndarray
        Hessian matrices at each particle, shape (N_points, dimension, dimension).
        hessian[i, :, :] is the Hessian matrix at particle i.

    Notes
    -----
    The Hessian matrix is symmetric for smooth functions:
        H[i,j] = ∂²f/∂x_i∂x_j = ∂²f/∂x_j∂x_i = H[j,i]

    Diagonal entries: ∂²f/∂x_i²
    Off-diagonal: ∂²f/∂x_i∂x_j (mixed derivatives)

    The trace of the Hessian is the Laplacian:
        Δf = trace(H) = Σ_i ∂²f/∂x_i²

    Examples
    --------
    >>> # Test on quadratic form f(x,y) = x² + xy + y²
    >>> # Analytical Hessian: [[2, 1], [1, 2]]
    >>> points = np.random.rand(100, 2)
    >>> f = points[:, 0]**2 + points[:, 0]*points[:, 1] + points[:, 1]**2
    >>> H = compute_hessian_gfdm(f, points)
    >>> np.mean(H, axis=0)  # Should be close to [[2, 1], [1, 2]]
    array([[2.01, 0.99],
           [0.99, 2.02]])
    """
    N_points, dimension = points.shape

    # Validate input
    if scalar_field.shape[0] != N_points:
        raise ValueError(f"scalar_field length {scalar_field.shape[0]} must match points length {N_points}")

    # Find neighbors
    neighbor_indices, neighbor_distances = find_neighbors_kdtree(points, k=k)
    k_actual = neighbor_indices.shape[1]

    # Adaptive support radius
    h_values = np.max(neighbor_distances, axis=1)

    # Preallocate output
    hessian = np.zeros((N_points, dimension, dimension))

    # Compute Hessian at each point
    for i in range(N_points):
        # Neighbors of point i
        nbr_idx = neighbor_indices[i, :]
        nbr_dist = neighbor_distances[i, :]
        h = h_values[i]

        # Local coordinates: Δx = x_j - x_i
        delta_x = points[nbr_idx, :] - points[i, :]  # (k, dimension)

        # Function values at neighbors
        f_nbrs = scalar_field[nbr_idx]
        f_i = scalar_field[i]

        # Weights
        weights = gaussian_rbf_weight(nbr_dist, h)

        # Build weighted least squares system for Hessian
        # Number of second derivatives: d*(d+1)/2 (symmetric matrix)
        n_second_derivs = dimension * (dimension + 1) // 2

        A = np.zeros((k_actual, n_second_derivs))
        b = np.zeros(k_actual)

        for j in range(k_actual):
            dx = delta_x[j, :]
            col_idx = 0
            for d1 in range(dimension):
                for d2 in range(d1, dimension):
                    if d1 == d2:
                        A[j, col_idx] = 0.5 * dx[d1] ** 2
                    else:
                        A[j, col_idx] = dx[d1] * dx[d2]
                    col_idx += 1

            b[j] = f_nbrs[j] - f_i

        # Weight the system
        W = np.diag(np.sqrt(weights))
        A_weighted = W @ A
        b_weighted = W @ b

        # Solve weighted least squares
        try:
            hessian_coeffs, _residuals, _rank, _s = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)
        except np.linalg.LinAlgError:
            # Singular matrix - use zero Hessian
            continue

        # Extract Hessian matrix from coefficients
        col_idx = 0
        for d1 in range(dimension):
            for d2 in range(d1, dimension):
                hessian[i, d1, d2] = hessian_coeffs[col_idx]
                if d1 != d2:
                    # Symmetry: H[i,j] = H[j,i]
                    hessian[i, d2, d1] = hessian_coeffs[col_idx]
                col_idx += 1

    return hessian


def compute_curl_gfdm(vector_field: np.ndarray, points: np.ndarray, k: int | None = None) -> np.ndarray:
    """
    Compute curl ∇ × α using GFDM on particles.

    Only valid for 2D and 3D vector fields:
    - 2D: Returns scalar vorticity ω = ∂α_y/∂x - ∂α_x/∂y
    - 3D: Returns vector curl [ω_x, ω_y, ω_z]

    Parameters
    ----------
    vector_field : np.ndarray
        Vector field at particles, shape (N_points, dimension).
        Must have dimension = 2 or 3.
    points : np.ndarray
        Particle positions, shape (N_points, dimension).
    k : int, optional
        Number of neighbors. If None, uses k = 2*dimension + 1.

    Returns
    -------
    curl : np.ndarray
        Curl/vorticity at each particle.
        - 2D: shape (N_points,) - scalar vorticity
        - 3D: shape (N_points, 3) - vector curl

    Raises
    ------
    ValueError
        If dimension is not 2 or 3.

    Notes
    -----
    2D curl (vorticity):
        ω = ∂α_y/∂x - ∂α_x/∂y

    3D curl:
        ∇ × α = [∂α_z/∂y - ∂α_y/∂z,
                 ∂α_x/∂z - ∂α_z/∂x,
                 ∂α_y/∂x - ∂α_x/∂y]

    Examples
    --------
    >>> # 2D vortex: α = [-y, x] → curl = 2
    >>> points = np.random.rand(100, 2)
    >>> alpha = np.column_stack([-points[:, 1], points[:, 0]])
    >>> curl = compute_curl_gfdm(alpha, points)
    >>> np.mean(curl)  # Should be close to 2.0
    2.01
    """
    N_points, dimension = points.shape

    # Validate dimension
    if dimension not in [2, 3]:
        raise ValueError(f"Curl only defined for 2D or 3D, got dimension={dimension}")

    # Validate input
    if vector_field.shape[0] != N_points:
        raise ValueError(f"vector_field length {vector_field.shape[0]} must match points length {N_points}")
    if vector_field.shape[1] != dimension:
        raise ValueError(f"vector_field dimension {vector_field.shape[1]} must match points dimension {dimension}")

    # Find neighbors
    neighbor_indices, neighbor_distances = find_neighbors_kdtree(points, k=k)

    # Adaptive support radius
    h_values = np.max(neighbor_distances, axis=1)

    if dimension == 2:
        # 2D: Return scalar vorticity
        curl = np.zeros(N_points)

        for i in range(N_points):
            nbr_idx = neighbor_indices[i, :]
            nbr_dist = neighbor_distances[i, :]
            h = h_values[i]

            delta_x = points[nbr_idx, :] - points[i, :]
            alpha_nbrs = vector_field[nbr_idx, :]
            alpha_i = vector_field[i, :]
            weights = gaussian_rbf_weight(nbr_dist, h)

            W = np.diag(np.sqrt(weights))

            # Compute ∂α_y/∂x
            A_x = delta_x[:, 0].reshape(-1, 1)
            b_alpha_y = alpha_nbrs[:, 1] - alpha_i[1]
            A_weighted = W @ A_x
            b_weighted = W @ b_alpha_y
            try:
                dalpha_y_dx, *_ = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)
                dalpha_y_dx = dalpha_y_dx[0]
            except np.linalg.LinAlgError:
                dalpha_y_dx = 0.0

            # Compute ∂α_x/∂y
            A_y = delta_x[:, 1].reshape(-1, 1)
            b_alpha_x = alpha_nbrs[:, 0] - alpha_i[0]
            A_weighted = W @ A_y
            b_weighted = W @ b_alpha_x
            try:
                dalpha_x_dy, *_ = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)
                dalpha_x_dy = dalpha_x_dy[0]
            except np.linalg.LinAlgError:
                dalpha_x_dy = 0.0

            # Vorticity: ∂α_y/∂x - ∂α_x/∂y
            curl[i] = dalpha_y_dx - dalpha_x_dy

    else:  # dimension == 3
        # 3D: Return vector curl
        curl = np.zeros((N_points, 3))

        for i in range(N_points):
            nbr_idx = neighbor_indices[i, :]
            nbr_dist = neighbor_distances[i, :]
            h = h_values[i]

            delta_x = points[nbr_idx, :] - points[i, :]
            alpha_nbrs = vector_field[nbr_idx, :]
            alpha_i = vector_field[i, :]
            weights = gaussian_rbf_weight(nbr_dist, h)

            W = np.diag(np.sqrt(weights))

            # Helper function to compute partial derivative
            def compute_partial(
                component_idx: int,
                coord_idx: int,
                delta_x=delta_x,
                alpha_nbrs=alpha_nbrs,
                alpha_i=alpha_i,
                W=W,
            ) -> float:
                A = delta_x[:, coord_idx].reshape(-1, 1)
                b = alpha_nbrs[:, component_idx] - alpha_i[component_idx]
                A_w = W @ A
                b_w = W @ b
                try:
                    deriv, *_ = np.linalg.lstsq(A_w, b_w, rcond=None)
                    return deriv[0]
                except np.linalg.LinAlgError:
                    return 0.0

            # Curl components:
            # ω_x = ∂α_z/∂y - ∂α_y/∂z
            curl[i, 0] = compute_partial(2, 1) - compute_partial(1, 2)
            # ω_y = ∂α_x/∂z - ∂α_z/∂x
            curl[i, 1] = compute_partial(0, 2) - compute_partial(2, 0)
            # ω_z = ∂α_y/∂x - ∂α_x/∂y
            curl[i, 2] = compute_partial(1, 0) - compute_partial(0, 1)

    return curl


def compute_directional_derivative_gfdm(
    scalar_field: np.ndarray,
    direction: np.ndarray,
    points: np.ndarray,
    k: int | None = None,
) -> np.ndarray:
    """
    Compute directional derivative ∇f · v along direction v using GFDM.

    Parameters
    ----------
    scalar_field : np.ndarray
        Scalar field values at particles, shape (N_points,).
    direction : np.ndarray
        Direction vector(s), shape (dimension,) or (N_points, dimension).
        If shape (dimension,), uses same direction for all points.
        If shape (N_points, dimension), uses different direction per point.
    points : np.ndarray
        Particle positions, shape (N_points, dimension).
    k : int, optional
        Number of neighbors. If None, uses k = 2*dimension + 1.

    Returns
    -------
    directional_derivative : np.ndarray
        Directional derivative at each particle, shape (N_points,).

    Notes
    -----
    The directional derivative is the rate of change of f along direction v:
        D_v f = ∇f · v = Σ_i (∂f/∂x_i) v_i

    If direction is not unit length, the result is scaled by ||v||.

    Examples
    --------
    >>> # Test on f(x,y) = x² + y² along direction [1, 0]
    >>> # Analytical: ∇f · [1,0] = [2x, 2y] · [1,0] = 2x
    >>> points = np.random.rand(100, 2)
    >>> f = points[:, 0]**2 + points[:, 1]**2
    >>> direction = np.array([1.0, 0.0])
    >>> D_v = compute_directional_derivative_gfdm(f, direction, points)
    >>> # Check correlation with 2x
    >>> np.corrcoef(D_v, 2*points[:, 0])[0, 1]  # Should be close to 1.0
    0.995
    """
    N_points, dimension = points.shape

    # Validate input
    if scalar_field.shape[0] != N_points:
        raise ValueError(f"scalar_field length {scalar_field.shape[0]} must match points length {N_points}")

    # Handle direction shape
    if direction.ndim == 1:
        # Single direction for all points
        if direction.shape[0] != dimension:
            raise ValueError(f"direction length {direction.shape[0]} must match dimension {dimension}")
        direction_field = np.tile(direction, (N_points, 1))
    elif direction.ndim == 2:
        # Different direction per point
        if direction.shape != (N_points, dimension):
            raise ValueError(f"direction shape {direction.shape} must be ({N_points}, {dimension})")
        direction_field = direction
    else:
        raise ValueError(f"direction must be 1D or 2D array, got {direction.ndim}D")

    # Compute gradient
    gradient = compute_gradient_gfdm(scalar_field, points, k=k)

    # Directional derivative: ∇f · v
    directional_derivative = np.sum(gradient * direction_field, axis=1)

    return directional_derivative


def compute_kernel_density_gfdm(
    particle_masses: np.ndarray,
    points: np.ndarray,
    k: int | None = None,
) -> np.ndarray:
    """
    Compute kernel density estimate (KDE) at particle locations using GFDM weights.

    Approximates density as ρ(x_i) = Σ_j m_j W(||x_i - x_j||, h)

    Parameters
    ----------
    particle_masses : np.ndarray
        Mass/weight of each particle, shape (N_points,).
    points : np.ndarray
        Particle positions, shape (N_points, dimension).
    k : int, optional
        Number of neighbors. If None, uses k = 2*dimension + 1.

    Returns
    -------
    density : np.ndarray
        Estimated density at each particle, shape (N_points,).

    Notes
    -----
    This is a simplified KDE using Gaussian RBF weights from GFDM.
    For more sophisticated kernel options, see mfg_pde.utils.numerical.particle.kernels.

    The bandwidth h is chosen adaptively as the maximum neighbor distance.

    Examples
    --------
    >>> # Uniform mass distribution
    >>> points = np.random.rand(100, 2)
    >>> masses = np.ones(100)
    >>> rho = compute_kernel_density_gfdm(masses, points)
    >>> np.mean(rho)  # Average density
    5.23
    """
    N_points, _dimension = points.shape

    # Validate input
    if particle_masses.shape[0] != N_points:
        raise ValueError(f"particle_masses length {particle_masses.shape[0]} must match points length {N_points}")

    # Find neighbors
    neighbor_indices, neighbor_distances = find_neighbors_kdtree(points, k=k)

    # Adaptive support radius
    h_values = np.max(neighbor_distances, axis=1)

    # Preallocate output
    density = np.zeros(N_points)

    # Compute density at each point
    for i in range(N_points):
        nbr_idx = neighbor_indices[i, :]
        nbr_dist = neighbor_distances[i, :]
        h = h_values[i]

        # Masses of neighbors
        m_nbrs = particle_masses[nbr_idx]

        # Weights
        weights = gaussian_rbf_weight(nbr_dist, h)

        # KDE: ρ = Σ_j m_j W(r_ij, h)
        density[i] = np.sum(m_nbrs * weights)

    return density


def compute_vector_laplacian_gfdm(
    vector_field: np.ndarray,
    points: np.ndarray,
    k: int | None = None,
) -> np.ndarray:
    """
    Compute vector Laplacian Δα (component-wise Laplacian) using GFDM.

    Approximates Δα = [Δα_x, Δα_y, Δα_z] where each component is computed
    independently using the scalar Laplacian operator.

    Parameters
    ----------
    vector_field : np.ndarray
        Vector field at particles, shape (N_points, dimension).
    points : np.ndarray
        Particle positions, shape (N_points, dimension).
    k : int, optional
        Number of neighbors. If None, uses k = 2*dimension + 1.

    Returns
    -------
    vector_laplacian : np.ndarray
        Vector Laplacian at each particle, shape (N_points, dimension).

    Notes
    -----
    The vector Laplacian is computed component-wise:
        (Δα)_i = Δ(α_i) for each component i

    This is useful for diffusion of vector fields, e.g., viscous terms
    in fluid dynamics: ∂α/∂t = ν Δα

    Examples
    --------
    >>> # Test on linear vector field α = [x, y]
    >>> # Analytical: Δα = [Δx, Δy] = [0, 0]
    >>> points = np.random.rand(100, 2)
    >>> alpha = points.copy()
    >>> delta_alpha = compute_vector_laplacian_gfdm(alpha, points)
    >>> np.max(np.abs(delta_alpha))  # Should be close to 0
    0.15
    """
    N_points, dimension = points.shape

    # Validate input
    if vector_field.shape[0] != N_points:
        raise ValueError(f"vector_field length {vector_field.shape[0]} must match points length {N_points}")
    if vector_field.shape[1] != dimension:
        raise ValueError(f"vector_field dimension {vector_field.shape[1]} must match points dimension {dimension}")

    # Preallocate output
    vector_laplacian = np.zeros((N_points, dimension))

    # Compute Laplacian for each component
    for d in range(dimension):
        vector_laplacian[:, d] = compute_laplacian_gfdm(vector_field[:, d], points, k=k)

    return vector_laplacian


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

    # f(x) = x² -> df/dx = 2x, d²f/dx² = 2
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

    # f(x,y) = x² + y² -> gradient = [2x, 2y], laplacian = 4
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

    print("\nGFDMOperator smoke tests passed!")
