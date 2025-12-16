"""
Differential Utilities for Numerical PDE Solvers.

Provides generic finite difference and Jacobian building tools that work
with any PDE discretization providing derivative sensitivity weights.

Architecture:
    Layer 1 - CORE: Generic FD gradient/Jacobian (no MFG dependencies)
    Layer 2 - PDE: Jacobian row builder from derivative weights
    Layer 3 - HJB: Thin wrappers with equation-specific coefficients

Usage Examples:
    # Generic FD gradient of scalar function
    grad = gradient_fd(f, x)

    # Generic FD Jacobian of vector function
    J = jacobian_fd(F, x)

    # PDE Jacobian row from discretization weights
    row = build_jacobian_row(
        coeffs={"grad": dR_dp, "lap": dR_dlap},
        weights={"grad": grad_weights, "lap": lap_weights},
        ...
    )

    # HJB-specific convenience wrapper
    row = build_hjb_jacobian_row(dH_dp, grad_w, lap_w, ...)

References:
    - Nesterov & Spokoiny (2017): Random gradient-free minimization
    - Burke et al. (2005): Robust gradient sampling for nonsmooth optimization
    - Nocedal & Wright (2006): Numerical Optimization, Ch. 7
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


# =============================================================================
# LAYER 1: CORE - Generic Finite Difference (no MFG dependencies)
# =============================================================================


def gradient_fd(
    f: Callable[[NDArray], float],
    x: NDArray,
    eps: float = 1e-7,
    method: Literal["forward", "central"] = "forward",
) -> NDArray:
    """
    Finite difference gradient of scalar function f: R^n -> R.

    Args:
        f: Scalar function taking array x and returning float
        x: Point at which to compute gradient, shape (n,)
        eps: Perturbation size
        method: "forward" O(eps) or "central" O(eps^2) accuracy

    Returns:
        Gradient array, shape (n,)

    Examples:
        >>> f = lambda x: x[0]**2 + x[1]**2  # f(x) = |x|^2
        >>> x = np.array([1.0, 2.0])
        >>> grad = gradient_fd(f, x)  # Should be [2, 4]
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    grad = np.zeros(n)

    if method == "forward":
        f_base = f(x)
        for j in range(n):
            x_plus = x.copy()
            x_plus[j] += eps
            grad[j] = (f(x_plus) - f_base) / eps

    elif method == "central":
        for j in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += eps
            x_minus[j] -= eps
            grad[j] = (f(x_plus) - f(x_minus)) / (2 * eps)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'forward' or 'central'.")

    return grad


def jacobian_fd(
    F: Callable[[NDArray], NDArray],
    x: NDArray,
    F_x: NDArray | None = None,
    eps: float = 1e-7,
    sparse: bool = False,
) -> NDArray:
    """
    Finite difference Jacobian of vector function F: R^n -> R^m.

    Computes J[i,j] = dF_i/dx_j using forward differences.

    Args:
        F: Vector function taking array x and returning array
        x: Point at which to compute Jacobian, shape (n,)
        F_x: F(x) if already computed (avoids redundant evaluation)
        eps: Perturbation size
        sparse: If True, return scipy.sparse.csr_matrix

    Returns:
        Jacobian matrix, shape (m, n)

    Examples:
        >>> F = lambda x: np.array([x[0]**2, x[0]*x[1]])
        >>> x = np.array([1.0, 2.0])
        >>> J = jacobian_fd(F, x)  # [[2, 0], [2, 1]]
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)

    if F_x is None:
        F_x = F(x)
    F_x = np.asarray(F_x).flatten()
    m = len(F_x)

    if sparse:
        import scipy.sparse as sp

        J = sp.lil_matrix((m, n), dtype=np.float64)
    else:
        J = np.zeros((m, n), dtype=np.float64)

    for j in range(n):
        x_pert = x.copy()
        x_pert[j] += eps
        F_pert = np.asarray(F(x_pert)).flatten()
        J[:, j] = (F_pert - F_x) / eps

    if sparse:
        return J.tocsr()
    return J


def hessian_fd(
    f: Callable[[NDArray], float],
    x: NDArray,
    eps: float = 1e-5,
) -> NDArray:
    """
    Finite difference Hessian of scalar function f: R^n -> R.

    Uses central differences for second derivatives:
        H[i,j] = (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4*eps^2)

    Args:
        f: Scalar function
        x: Point, shape (n,)
        eps: Perturbation size (use larger eps than gradient for stability)

    Returns:
        Hessian matrix, shape (n, n)
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    H = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += eps
            x_pp[j] += eps
            x_pm[i] += eps
            x_pm[j] -= eps
            x_mp[i] -= eps
            x_mp[j] += eps
            x_mm[i] -= eps
            x_mm[j] -= eps

            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps * eps)
            H[j, i] = H[i, j]  # Symmetric

    return H


def partial_derivative_fd(
    f: Callable[[NDArray], float],
    x: NDArray,
    dim: int,
    eps: float = 1e-7,
    method: Literal["forward", "central"] = "central",
) -> float:
    """
    Finite difference partial derivative df/dx_dim.

    Args:
        f: Scalar function
        x: Point, shape (n,)
        dim: Dimension index for partial derivative
        eps: Perturbation size
        method: "forward" or "central"

    Returns:
        Partial derivative value (scalar)
    """
    x = np.asarray(x, dtype=np.float64)

    if method == "forward":
        x_plus = x.copy()
        x_plus[dim] += eps
        return (f(x_plus) - f(x)) / eps

    elif method == "central":
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[dim] += eps
        x_minus[dim] -= eps
        return (f(x_plus) - f(x_minus)) / (2 * eps)

    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# LAYER 2: PDE - Jacobian from Derivative Weights (generic PDE structure)
# =============================================================================


def build_jacobian_row(
    residual_coeffs: dict[str, NDArray],
    derivative_weights: dict[str, NDArray],
    neighbor_indices: NDArray,
    center_idx: int,
    n_points: int,
    diagonal_term: float = 0.0,
    center_weight_mode: Literal["explicit", "negative_sum"] = "negative_sum",
) -> NDArray:
    """
    Build Jacobian row for PDE residual depending on spatial derivatives.

    Generic formula for residual R depending on derivatives:
        dR_i/du_j = sum_k (dR/d(deriv_k)) . (d(deriv_k)/du_j) + diagonal_term * delta_ij

    This works for any PDE where the residual depends on gradient, Laplacian,
    or other spatial derivatives that are linear combinations of neighbor values.

    Args:
        residual_coeffs: Coefficients {deriv_name: dR/d(deriv)}
            - "grad": shape (dim,) - coefficient for gradient terms
            - "lap": shape (1,) or scalar - coefficient for Laplacian
            - Can include other derivative types
        derivative_weights: Sensitivity weights from discretization
            - "grad": shape (dim, n_neighbors) - d(grad)/du_neighbor
            - "lap": shape (n_neighbors,) - d(lap)/du_neighbor
        neighbor_indices: Indices of neighbors, shape (n_neighbors,)
        center_idx: Index of center point (row index i)
        n_points: Total number of points
        diagonal_term: Added to diagonal (e.g., 1/dt for time stepping)
        center_weight_mode: How to compute center point weight
            - "explicit": Center is included in neighbor_indices
            - "negative_sum": Center weight = -sum(neighbor weights)
              (for GFDM where b = u_neighbors - u_center)

    Returns:
        Jacobian row, shape (n_points,)

    Examples:
        # HJB equation: R = u_t + H(p) - diffusion * Lap(u)
        # dR/du = 1/dt + dH/dp . dp/du - diffusion * d(Lap)/du
        row = build_jacobian_row(
            residual_coeffs={"grad": dH_dp, "lap": np.array([-diffusion])},
            derivative_weights={"grad": grad_weights, "lap": lap_weights},
            neighbor_indices=neighbors, center_idx=i, n_points=n,
            diagonal_term=1.0/dt
        )

        # FP equation: R = m_t - div(m*v) + diffusion * Lap(m)
        row = build_jacobian_row(
            residual_coeffs={"grad": -velocity, "lap": np.array([diffusion])},
            ...
        )
    """
    row = np.zeros(n_points)

    # Process each derivative type
    for deriv_name, coeff in residual_coeffs.items():
        if deriv_name not in derivative_weights:
            continue

        weights = derivative_weights[deriv_name]
        coeff = np.atleast_1d(coeff)

        # Handle different weight shapes
        if weights.ndim == 1:
            # Scalar derivative (e.g., Laplacian): weights shape (n_neighbors,)
            # coeff should be scalar or (1,)
            weight_contrib = float(coeff[0]) * weights
        else:
            # Vector derivative (e.g., gradient): weights shape (dim, n_neighbors)
            # coeff shape (dim,), compute dot product
            weight_contrib = coeff @ weights  # shape (n_neighbors,)

        # Add contribution from neighbors
        for k, j in enumerate(neighbor_indices):
            if j < 0:
                continue  # Skip ghost particles
            row[j] += weight_contrib[k]

        # Handle center point contribution
        if center_weight_mode == "negative_sum":
            # For methods where b = u_neighbors - u_center
            # d(deriv)/du_center = -sum(d(deriv)/du_neighbors)
            if weights.ndim == 1:
                center_weight = float(coeff[0]) * (-np.sum(weights))
            else:
                center_weight = np.dot(coeff, -np.sum(weights, axis=1))
            row[center_idx] += center_weight

    # Add diagonal term (e.g., 1/dt from time derivative)
    row[center_idx] += diagonal_term

    return row


def build_sparse_jacobian_pattern(
    neighbor_indices_list: list[NDArray],
    n_points: int,
) -> tuple[NDArray, NDArray]:
    """
    Build sparse pattern (row, col indices) from neighborhood structure.

    Useful for scipy.sparse construction when only non-zero pattern is needed.

    Args:
        neighbor_indices_list: List of neighbor indices for each point
        n_points: Total number of points

    Returns:
        Tuple of (row_indices, col_indices) for sparse matrix construction
    """
    rows = []
    cols = []

    for i in range(n_points):
        neighbors = neighbor_indices_list[i]
        for j in neighbors:
            if j >= 0:  # Skip ghost particles
                rows.append(i)
                cols.append(j)
        # Center point is always in pattern (diagonal)
        if i not in neighbors:
            rows.append(i)
            cols.append(i)

    return np.array(rows), np.array(cols)


# =============================================================================
# LAYER 3: HJB-Specific Wrappers (thin convenience functions)
# =============================================================================


def compute_dH_dp(
    H_func: Callable,
    x_idx: int,
    m: float,
    p: NDArray,
    hess: NDArray | None = None,
    eps: float = 1e-7,
    method: Literal["forward", "central"] = "forward",
) -> NDArray:
    """
    Compute dH/dp for Hamiltonian via finite difference.

    Wrapper around gradient_fd that handles DerivativeTensors construction.

    Args:
        H_func: Hamiltonian function H(x_idx, m, derivs=...) -> float
        x_idx: Point index
        m: Density value at point
        p: Current momentum (gradient of u), shape (dim,)
        hess: Hessian of u (optional), shape (dim, dim)
        eps: FD perturbation size
        method: "forward" or "central"

    Returns:
        dH/dp array, shape (dim,)
    """
    from mfg_pde.core.derivatives import DerivativeTensors

    p = np.asarray(p)
    dim = len(p)

    if hess is None:
        hess = np.zeros((dim, dim))

    def H_of_p(p_vec: NDArray) -> float:
        derivs = DerivativeTensors.from_arrays(grad=p_vec, hess=hess)
        return H_func(x_idx, m, derivs=derivs)

    return gradient_fd(H_of_p, p, eps=eps, method=method)


def build_hjb_jacobian_row(
    dH_dp: NDArray,
    grad_weights: NDArray,
    lap_weights: NDArray,
    neighbor_indices: NDArray,
    center_idx: int,
    n_points: int,
    dt: float,
    diffusion_coeff: float,
) -> NDArray:
    """
    Build HJB Jacobian row using analytic formula.

    HJB residual: R = (u - u^{n+1})/dt + H(x, p, m) - (sigma^2/2) * Lap(u)
    Jacobian: dR/du_j = 1/dt * delta_ij + dH/dp . dp/du_j - diffusion * d(Lap)/du_j

    Args:
        dH_dp: Hamiltonian momentum derivative, shape (dim,)
        grad_weights: Gradient sensitivity, shape (dim, n_neighbors)
        lap_weights: Laplacian sensitivity, shape (n_neighbors,)
        neighbor_indices: Neighbor indices, shape (n_neighbors,)
        center_idx: Center point index
        n_points: Total points
        dt: Time step
        diffusion_coeff: Diffusion coefficient (sigma^2/2)

    Returns:
        Jacobian row, shape (n_points,)
    """
    return build_jacobian_row(
        residual_coeffs={
            "grad": dH_dp,
            "lap": np.array([-diffusion_coeff]),
        },
        derivative_weights={
            "grad": grad_weights,
            "lap": lap_weights,
        },
        neighbor_indices=neighbor_indices,
        center_idx=center_idx,
        n_points=n_points,
        diagonal_term=1.0 / dt,
        center_weight_mode="negative_sum",
    )


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing differential_utils...")

    # ----- Layer 1: Core FD tests -----
    print("\n[Layer 1] Testing generic FD...")

    # Test gradient_fd with f(x) = |x|^2
    def f_quadratic(x):
        return np.sum(x**2)

    x = np.array([1.0, 2.0, 3.0])
    grad_fwd = gradient_fd(f_quadratic, x, method="forward")
    grad_ctr = gradient_fd(f_quadratic, x, method="central")
    expected_grad = 2 * x

    print(f"  gradient_fd forward error: {np.linalg.norm(grad_fwd - expected_grad):.2e}")
    print(f"  gradient_fd central error: {np.linalg.norm(grad_ctr - expected_grad):.2e}")
    assert np.linalg.norm(grad_fwd - expected_grad) < 1e-5
    assert np.linalg.norm(grad_ctr - expected_grad) < 1e-7  # Central is O(eps^2)

    # Test jacobian_fd with F(x) = [x0^2, x0*x1]
    def F_vec(x):
        return np.array([x[0] ** 2, x[0] * x[1]])

    x2 = np.array([2.0, 3.0])
    J = jacobian_fd(F_vec, x2)
    expected_J = np.array([[4.0, 0.0], [3.0, 2.0]])  # [[2*x0, 0], [x1, x0]]

    print(f"  jacobian_fd error: {np.linalg.norm(J - expected_J):.2e}")
    assert np.linalg.norm(J - expected_J) < 1e-5

    # Test hessian_fd with f(x) = x0^2 + x1^2 + x0*x1
    def f_with_cross(x):
        return x[0] ** 2 + x[1] ** 2 + x[0] * x[1]

    x3 = np.array([1.0, 1.0])
    H = hessian_fd(f_with_cross, x3)
    expected_H = np.array([[2.0, 1.0], [1.0, 2.0]])

    print(f"  hessian_fd error: {np.linalg.norm(H - expected_H):.2e}")
    assert np.linalg.norm(H - expected_H) < 1e-4

    # ----- Layer 2: PDE Jacobian tests -----
    print("\n[Layer 2] Testing PDE Jacobian builder...")

    # Simple test: 1D case with 2 neighbors
    grad_weights = np.array([[0.5, -0.3]])  # 1D, 2 neighbors
    lap_weights = np.array([1.0, -0.5])
    neighbor_indices = np.array([1, 2])
    dR_dp = np.array([2.0])  # dR/d(grad)
    dR_dlap = np.array([-0.5])  # dR/d(lap)

    row = build_jacobian_row(
        residual_coeffs={"grad": dR_dp, "lap": dR_dlap},
        derivative_weights={"grad": grad_weights, "lap": lap_weights},
        neighbor_indices=neighbor_indices,
        center_idx=0,
        n_points=5,
        diagonal_term=10.0,
    )

    print(f"  Jacobian row: {row}")
    assert row[0] != 0, "Center should have contribution"
    assert row[1] != 0, "Neighbor 1 should have contribution"
    assert row[2] != 0, "Neighbor 2 should have contribution"
    assert row[3] == 0, "Non-neighbor should be zero"
    assert row[4] == 0, "Non-neighbor should be zero"

    # ----- Layer 3: HJB wrapper tests -----
    print("\n[Layer 3] Testing HJB wrappers...")

    # Test compute_dH_dp with H = |p|^2 / 2

    def H_lq(x_idx, m, derivs):
        return 0.5 * derivs.grad_norm_squared  # Property, not method

    p = np.array([1.0, 2.0])
    dH_dp = compute_dH_dp(H_lq, 0, 1.0, p)
    expected_dH_dp = p  # For H = |p|^2/2, dH/dp = p

    print(f"  compute_dH_dp error: {np.linalg.norm(dH_dp - expected_dH_dp):.2e}")
    assert np.linalg.norm(dH_dp - expected_dH_dp) < 1e-5

    # Test build_hjb_jacobian_row
    grad_w = np.array([[0.5, -0.3], [0.2, 0.1]])  # 2D, 2 neighbors
    lap_w = np.array([1.0, -0.5])
    neighbors = np.array([1, 2])

    hjb_row = build_hjb_jacobian_row(
        dH_dp=dH_dp,
        grad_weights=grad_w,
        lap_weights=lap_w,
        neighbor_indices=neighbors,
        center_idx=0,
        n_points=5,
        dt=0.1,
        diffusion_coeff=0.5,
    )

    print(f"  HJB Jacobian row: {hjb_row}")
    assert hjb_row[0] != 0, "Center entry should be non-zero"
    assert abs(hjb_row[0]) > 9, "Should include 1/dt = 10"

    print("\nAll tests passed!")
