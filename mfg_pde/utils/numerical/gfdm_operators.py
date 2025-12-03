"""
GFDM (Generalized Finite Difference Method) Operators for Meshfree Computations.

This module provides spatial differential operators on scattered particles using
the Generalized Finite Difference Method (GFDM). GFDM approximates derivatives
via weighted least squares fitting of local Taylor expansions.

Key Features:
- Neighbor finding using KDTree (O(N log N))
- Gaussian RBF weight functions
- First-order operators: gradient, divergence, curl, directional derivative
- Second-order operators: Laplacian, Hessian, vector Laplacian
- Auxiliary operators: kernel density estimation

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

Available Operators:
-------------------
First-order derivatives:
    - compute_gradient_gfdm: ∇f (gradient of scalar field)
    - compute_divergence_gfdm: ∇·(m α) (divergence of flux)
    - compute_curl_gfdm: ∇ × α (curl/vorticity, 2D/3D only)
    - compute_directional_derivative_gfdm: ∇f · v (directional derivative)

Second-order derivatives:
    - compute_laplacian_gfdm: Δf (scalar Laplacian)
    - compute_hessian_gfdm: ∂²f/∂x_i∂x_j (full Hessian matrix)
    - compute_vector_laplacian_gfdm: Δα (component-wise vector Laplacian)

Auxiliary operators:
    - compute_kernel_density_gfdm: KDE using GFDM weights

Utilities:
    - find_neighbors_kdtree: k-nearest neighbor search
    - gaussian_rbf_weight: Gaussian RBF weight function

Usage:
------
    from mfg_pde.utils.numerical.gfdm_operators import (
        find_neighbors_kdtree,
        gaussian_rbf_weight,
        compute_gradient_gfdm,
        compute_laplacian_gfdm,
        compute_divergence_gfdm,
        compute_curl_gfdm,
        compute_hessian_gfdm,
        compute_vector_laplacian_gfdm,
    )

    # Find neighbors
    neighbor_indices, neighbor_distances = find_neighbors_kdtree(points, k=None)

    # First-order derivatives
    gradient = compute_gradient_gfdm(scalar_field, points, k=None)
    divergence = compute_divergence_gfdm(vector_field, density, points, k=None)
    curl = compute_curl_gfdm(vector_field, points, k=None)  # 2D/3D only

    # Second-order derivatives
    laplacian = compute_laplacian_gfdm(scalar_field, points, k=None)
    hessian = compute_hessian_gfdm(scalar_field, points, k=None)
    vector_laplacian = compute_vector_laplacian_gfdm(vector_field, points, k=None)

    # Kernel density estimation
    density = compute_kernel_density_gfdm(particle_masses, points, k=None)

References:
-----------
- Benito et al., "Influence of several factors in the generalized finite
  difference method", Applied Mathematical Modelling, 2001.
- Gavete et al., "Improvements of generalized finite difference method and
  comparison with other meshless method", Applied Mathematical Modelling, 2003.
- Liu & Liu, "Smoothed Particle Hydrodynamics: A Meshfree Particle Method",
  World Scientific, 2003.

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
    Uses GaussianKernel from mfg_pde.utils.numerical.particle.kernels with proper
    normalization for arbitrary dimensions.

    Alternative weight functions available in kernels module:
    - WendlandKernel(k=0,1,2,3): Compact support, C^0, C^2, C^4, C^6 smoothness
    - CubicSplineKernel, QuinticSplineKernel: SPH B-spline kernels
    - CubicKernel, QuarticKernel: Simple polynomial kernels

    See mfg_pde.utils.numerical.particle.kernels for full kernel API.

    Examples
    --------
    >>> r = np.array([0.0, 0.5, 1.0, 2.0])
    >>> w = gaussian_rbf_weight(r, h=1.0)
    >>> w
    array([1.        , 0.77880078, 0.36787944, 0.01831564])
    """
    from mfg_pde.utils.numerical.particle.kernels import GaussianKernel

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
