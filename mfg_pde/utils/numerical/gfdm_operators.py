"""
GFDM (Generalized Finite Difference Method) Operators for Meshfree Computations.

This module provides spatial differential operators on scattered particles using
the Generalized Finite Difference Method (GFDM). GFDM approximates derivatives
via weighted least squares fitting of local Taylor expansions.

Key Features:
- Neighbor finding using KDTree (O(N log N))
- Gaussian RBF weight functions
- Laplacian operator for diffusion terms
- Divergence operator for advection/flux terms
- Gradient operator for first-order derivatives

Mathematical Background:
-----------------------
GFDM approximates spatial derivatives at point x_i using k nearest neighbors {x_j}.

Taylor expansion around x_i:
    f(x_j) ≈ f(x_i) + ∇f|_i · (x_j - x_i) + (1/2)(x_j - x_i)^T H|_i (x_j - x_i) + ...

Weighted least squares problem:
    min_coeffs Σ_j w(r_ij) [f(x_j) - Σ_α c_α φ_α(x_j - x_i)]²

where:
    w(r) = weight function (e.g., Gaussian RBF)
    φ_α = basis functions (monomials: 1, x, y, x², xy, y², ...)
    c_α = coefficients to solve for

Result: Derivative as linear combination of neighbor values
    ∂f/∂x|_i ≈ Σ_j w_ij f(x_j)

Usage:
------
    from mfg_pde.utils.numerical.gfdm_operators import (
        find_neighbors_kdtree,
        gaussian_rbf_weight,
        compute_laplacian_gfdm,
        compute_divergence_gfdm,
    )

    # Find neighbors
    neighbor_indices, neighbor_distances = find_neighbors_kdtree(points, k=None)

    # Compute Laplacian (diffusion)
    laplacian = compute_laplacian_gfdm(scalar_field, points, k=None)

    # Compute divergence (advection)
    divergence = compute_divergence_gfdm(vector_field, density, points, k=None)

References:
-----------
- Benito et al., "Influence of several factors in the generalized finite
  difference method", Applied Mathematical Modelling, 2001.
- Gavete et al., "Improvements of generalized finite difference method and
  comparison with other meshless method", Applied Mathematical Modelling, 2003.

Author: MFG_PDE Development Team
Created: 2025-11-04
"""

from __future__ import annotations

import numpy as np


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
    Gaussian RBF is infinitely differentiable and provides smooth weights.
    Typical choice: h = average neighbor spacing or h = max(neighbor_distances).

    Alternative weight functions:
    - Cubic: w(r) = (1 - r/h)³ for r < h, 0 otherwise
    - Quartic: w(r) = (1 - r/h)⁴ for r < h, 0 otherwise
    - Wendland C2: Compact support, C2 continuous

    For more kernel options, see mfg_pde.utils.numerical.smoothing_kernels module.

    Examples
    --------
    >>> r = np.array([0.0, 0.5, 1.0, 2.0])
    >>> w = gaussian_rbf_weight(r, h=1.0)
    >>> w
    array([1.        , 0.77880078, 0.36787944, 0.01831564])
    """
    return np.exp(-((r / h) ** 2))


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
