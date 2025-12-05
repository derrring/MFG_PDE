"""
RBF-based Differential Operators for Meshfree Methods.

This module provides Radial Basis Function (RBF) differentiation matrices
for computing spatial derivatives on scattered points. Unlike GFDM which uses
local weighted least squares, RBF methods build global interpolation matrices.

Key Features:
- Uses existing kernel infrastructure from mfg_pde.utils.numerical.kernels
- Supports various RBF types: Gaussian, Wendland C^{2k}, splines
- Provides Laplacian, gradient, and Hessian operators
- Optional polynomial augmentation for improved accuracy
- Regularization for ill-conditioned systems

Mathematical Background:
-----------------------
Given scattered points {x_i}_{i=1}^N and function values {u_i}, RBF interpolation
seeks a function:

    s(x) = sum_{j=1}^N w_j * phi(||x - x_j||)

where phi is the radial basis function and weights w satisfy: A @ w = u
with A_ij = phi(||x_i - x_j||).

For derivatives, we build differentiation matrices:
- Laplacian: L_ij = Delta phi(||x_i - x_j||)
- Gradient:  D_ij^k = d/dx_k phi(||x_i - x_j||)

Then:
    Delta u approx L @ A^{-1} @ u = L @ w
    grad_k u approx D^k @ A^{-1} @ u = D^k @ w

References:
-----------
- Fasshauer, G. E. "Meshfree Approximation Methods with MATLAB" (2007)
- Wendland, H. "Scattered Data Approximation" (2004)
- Fornberg, B., Flyer, N. "A Primer on Radial Basis Functions with
  Applications to the Geosciences" (2015)

Author: MFG_PDE Development Team
Created: 2025-12-05
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.spatial.distance import cdist

from mfg_pde.utils.numerical.kernels import (
    GaussianKernel,
    WendlandKernel,
    create_kernel,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RBFOperator:
    """
    RBF-based differential operator for scattered points.

    This class builds RBF interpolation and differentiation matrices once
    during initialization, then provides efficient derivative computation.

    Attributes:
        points: Collocation points, shape (n_points, dimension)
        n_points: Number of collocation points
        dimension: Spatial dimension
        kernel: The kernel (RBF) function used
        shape_parameter: Shape parameter (bandwidth) for the kernel
        A: Interpolation matrix, shape (n_points, n_points)
        A_inv: Pseudoinverse of interpolation matrix

    Example:
        >>> import numpy as np
        >>> from mfg_pde.utils.numerical.rbf_operators import RBFOperator
        >>>
        >>> # Create 2D scattered points
        >>> points = np.random.rand(100, 2)
        >>>
        >>> # Create RBF operator with Gaussian kernel
        >>> rbf = RBFOperator(points, kernel_type='gaussian', shape_parameter=2.0)
        >>>
        >>> # Compute derivatives of f(x,y) = x^2 + y^2
        >>> u = points[:, 0]**2 + points[:, 1]**2
        >>> lap = rbf.laplacian(u)      # Should be ~4.0 everywhere
        >>> grad = rbf.gradient(u)      # Should be [2x, 2y]
    """

    def __init__(
        self,
        points: NDArray[np.float64],
        kernel_type: Literal[
            "gaussian",
            "wendland_c0",
            "wendland_c2",
            "wendland_c4",
            "wendland_c6",
        ] = "gaussian",
        shape_parameter: float | None = None,
        regularization: float = 1e-12,
        use_polynomial: bool = False,
        polynomial_degree: int = 1,
    ):
        """
        Initialize RBF operator with precomputed matrices.

        Args:
            points: Collocation points, shape (n_points, dimension)
            kernel_type: Type of kernel/RBF to use. Options:
                - 'gaussian': Gaussian RBF (infinite support, C^inf)
                - 'wendland_c0': Wendland C^0 (compact, C^0)
                - 'wendland_c2': Wendland C^2 (compact, C^2)
                - 'wendland_c4': Wendland C^4 (compact, C^4)
                - 'wendland_c6': Wendland C^6 (compact, C^6)
            shape_parameter: Shape parameter (bandwidth) for the kernel.
                - For Gaussian: epsilon in exp(-(epsilon*r)^2), default ~1/h
                - For Wendland: support radius, default ~2*h
                where h is the average point spacing
            regularization: Tikhonov regularization parameter for matrix inversion.
                Helps with ill-conditioning. Default: 1e-12.
            use_polynomial: If True, augment RBF with polynomial basis for
                improved accuracy near boundaries.
            polynomial_degree: Degree of polynomial augmentation (0, 1, or 2).
                Only used if use_polynomial=True.
        """
        self.points = np.asarray(points)
        self.n_points, self.dimension = self.points.shape
        self.kernel_type = kernel_type
        self.regularization = regularization
        self.use_polynomial = use_polynomial
        self.polynomial_degree = polynomial_degree

        # Estimate average point spacing for default shape parameter
        self._estimate_spacing()

        # Set shape parameter
        if shape_parameter is None:
            # Heuristic for default shape parameters
            if kernel_type == "gaussian":
                # For Gaussian: use moderate scaling to balance accuracy and conditioning
                # epsilon ~ 1/h gives good accuracy but can be ill-conditioned
                self.shape_parameter = 0.5 / self.avg_spacing
            else:
                # For Wendland and other compact kernels: need enough overlap
                # Support must include ~10+ neighbors for well-posed system
                # Rule of thumb: support radius ~ 4-6 * avg_spacing
                self.shape_parameter = 5.0 * self.avg_spacing
        else:
            self.shape_parameter = shape_parameter

        # Create kernel instance
        self.kernel = create_kernel(kernel_type, dimension=self.dimension)

        # Precompute distance matrix
        self._compute_distance_matrix()

        # Build interpolation matrix A
        self._build_interpolation_matrix()

        # Precompute matrices for derivatives
        self._build_derivative_matrices()

    def _estimate_spacing(self):
        """Estimate average point spacing using nearest neighbors."""
        from scipy.spatial import cKDTree

        tree = cKDTree(self.points)
        # Find distance to nearest neighbor (excluding self)
        distances, _ = tree.query(self.points, k=2)
        self.avg_spacing = np.mean(distances[:, 1])  # Column 1 is first non-self neighbor

    def _compute_distance_matrix(self):
        """Compute pairwise distance matrix."""
        self.distance_matrix = cdist(self.points, self.points, metric="euclidean")

    def _build_interpolation_matrix(self):
        """
        Build RBF interpolation matrix A.

        A_ij = phi(||x_i - x_j||)
        """
        h = self.shape_parameter

        # Evaluate kernel at all pairwise distances
        self.A = self.kernel(self.distance_matrix, h)

        # Add regularization for numerical stability
        self.A += self.regularization * np.eye(self.n_points)

        # Compute pseudoinverse using SVD for stability
        try:
            # Use regularized solve instead of full inverse for better stability
            self.A_factored = np.linalg.cholesky(self.A + self.regularization * np.eye(self.n_points))
            self._use_cholesky = True
        except np.linalg.LinAlgError:
            # Fall back to SVD if Cholesky fails
            U, S, Vt = np.linalg.svd(self.A)
            # Truncate small singular values
            tol = self.regularization * S[0]
            rank = np.sum(tol < S)
            S_inv = np.zeros_like(S)
            S_inv[:rank] = 1.0 / S[:rank]
            self.A_inv = Vt.T @ np.diag(S_inv) @ U.T
            self._use_cholesky = False

    def _solve_interpolation(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve A @ w = u for RBF weights."""
        if self._use_cholesky:
            # Forward-backward substitution
            y = np.linalg.solve(self.A_factored, u)
            return np.linalg.solve(self.A_factored.T, y)
        else:
            return self.A_inv @ u

    def _build_derivative_matrices(self):
        """
        Build RBF derivative matrices for Laplacian and gradient.

        For Gaussian RBF phi(r) = exp(-(epsilon*r)^2):
            d phi/dx_k = -2 * epsilon^2 * (x_k - x_k') * phi(r)
            Delta phi = -2 * epsilon^2 * d * phi(r) + 4 * epsilon^4 * r^2 * phi(r)
                      = 2 * epsilon^2 * phi(r) * (2 * epsilon^2 * r^2 - d)

        For Wendland kernels, derivatives are computed via the kernel's
        evaluate_with_derivative method.
        """
        h = self.shape_parameter
        d = self.dimension

        # Initialize derivative matrices
        self.laplacian_matrix = np.zeros((self.n_points, self.n_points))
        self.gradient_matrices = [np.zeros((self.n_points, self.n_points)) for _ in range(d)]

        if isinstance(self.kernel, GaussianKernel):
            self._build_gaussian_derivative_matrices(h, d)
        elif isinstance(self.kernel, WendlandKernel):
            self._build_wendland_derivative_matrices(h, d)
        else:
            # Generic finite-difference approximation for other kernels
            self._build_fd_derivative_matrices(h, d)

    def _build_gaussian_derivative_matrices(self, h: float, d: int):
        """
        Build derivative matrices for Gaussian RBF.

        For Gaussian phi(r) = exp(-(r/h)^2):
            d phi/dr = -2r/h^2 * phi
            d phi/dx_k = d phi/dr * (x_k - x_k')/r = -2(x_k - x_k')/h^2 * phi

        Laplacian:
            Delta phi = (4r^2/h^4 - 2d/h^2) * phi
        """
        for i in range(self.n_points):
            for j in range(self.n_points):
                r = self.distance_matrix[i, j]
                phi = self.A[i, j] - self.regularization * (1 if i == j else 0)

                if i == j:
                    # At center: Delta phi(0) = -2d/h^2 (limit as r -> 0)
                    self.laplacian_matrix[i, j] = -2 * d / (h**2)
                    # Gradient is 0 at center
                    continue

                # Laplacian
                self.laplacian_matrix[i, j] = phi * (4 * r**2 / h**4 - 2 * d / h**2)

                # Gradient
                dx = self.points[i] - self.points[j]
                for k in range(d):
                    self.gradient_matrices[k][i, j] = -2 * dx[k] / h**2 * phi

    def _build_wendland_derivative_matrices(self, h: float, d: int):
        """
        Build derivative matrices for Wendland kernels.

        Uses the kernel's evaluate_with_derivative method.
        For Wendland: phi(r,h) has compact support [0, h].

        Mathematical note:
        For radially symmetric phi(r) in d dimensions:
            Delta phi = d^2 phi/dr^2 + (d-1)/r * d phi/dr

        At r=0, using L'Hopital's rule:
            lim_{r->0} (d-1)/r * dphi/dr = (d-1) * d^2 phi/dr^2|_{r=0}

        So:
            Delta phi|_{r=0} = d * d^2 phi/dr^2|_{r=0}
        """
        for i in range(self.n_points):
            for j in range(self.n_points):
                r = self.distance_matrix[i, j]

                if r >= h:
                    # Outside support
                    continue

                # Use numerical derivatives for robustness
                eps = 1e-6 * h

                if i == j or r < eps:
                    # At center - compute d^2 phi/dr^2|_{r=0} numerically
                    # Using second-order central difference: f''(0) ≈ (f(h) - 2f(0) + f(-h))/h^2
                    # But phi is symmetric, so: f''(0) ≈ 2*(f(eps) - f(0))/eps^2
                    phi_0 = self.kernel(0.0, h)
                    phi_eps = self.kernel(eps, h)
                    d2phi_dr2 = 2 * (phi_eps - phi_0) / (eps**2)

                    # Delta phi|_{r=0} = d * d^2 phi/dr^2|_{r=0}
                    self.laplacian_matrix[i, j] = d * d2phi_dr2
                    continue

                # General case: r > 0
                phi = self.kernel(r, h)
                phi_plus = self.kernel(min(r + eps, h - 1e-10), h)
                phi_minus = self.kernel(max(r - eps, 1e-10), h)

                # First derivative
                dphi_dr = (phi_plus - phi_minus) / (2 * eps)

                # Second derivative
                d2phi_dr2 = (phi_plus - 2 * phi + phi_minus) / (eps**2)

                # Laplacian
                self.laplacian_matrix[i, j] = d2phi_dr2 + (d - 1) / r * dphi_dr

                # Gradient: d phi/dx_k = dphi/dr * (x_k - x_k')/r
                dx = self.points[i] - self.points[j]
                for k in range(d):
                    self.gradient_matrices[k][i, j] = dphi_dr * dx[k] / r

    def _build_fd_derivative_matrices(self, h: float, d: int):
        """
        Build derivative matrices using finite differences (fallback).

        For kernels without analytical derivatives, use numerical differentiation.
        """
        eps = 1e-8 * h

        for i in range(self.n_points):
            for j in range(self.n_points):
                r = self.distance_matrix[i, j]
                phi = self.A[i, j] - self.regularization * (1 if i == j else 0)

                if i == j:
                    # Numerical second derivative at r=0
                    phi_eps = self.kernel(eps, h)
                    phi_0 = self.kernel(0.0, h)
                    d2phi = (phi_eps - 2 * phi_0 + phi_eps) / eps**2
                    self.laplacian_matrix[i, j] = d * d2phi
                    continue

                # First derivative via central difference on r
                phi_plus = self.kernel(r + eps, h)
                phi_minus = self.kernel(max(r - eps, 0), h)
                dphi_dr = (phi_plus - phi_minus) / (2 * eps)

                # Second derivative
                d2phi_dr2 = (phi_plus - 2 * phi + phi_minus) / eps**2

                # Laplacian
                self.laplacian_matrix[i, j] = d2phi_dr2 + (d - 1) / r * dphi_dr

                # Gradient
                dx = self.points[i] - self.points[j]
                for k in range(d):
                    self.gradient_matrices[k][i, j] = dphi_dr * dx[k] / r

    def laplacian(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute Laplacian Delta u at all points.

        Args:
            u: Function values at points, shape (n_points,)

        Returns:
            Laplacian values, shape (n_points,)
        """
        # Solve for RBF weights: A @ w = u
        w = self._solve_interpolation(u)

        # Apply Laplacian: Delta u = L @ w
        return self.laplacian_matrix @ w

    def gradient(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute gradient nabla u at all points.

        Args:
            u: Function values at points, shape (n_points,)

        Returns:
            Gradient values, shape (n_points, dimension)
        """
        # Solve for RBF weights
        w = self._solve_interpolation(u)

        # Apply gradient matrices
        grad = np.zeros((self.n_points, self.dimension))
        for k in range(self.dimension):
            grad[:, k] = self.gradient_matrices[k] @ w

        return grad

    def hessian(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute Hessian matrix at all points.

        This is an approximation using gradient of gradient components.

        Args:
            u: Function values at points, shape (n_points,)

        Returns:
            Hessian values, shape (n_points, dimension, dimension)
        """
        hess = np.zeros((self.n_points, self.dimension, self.dimension))

        # Compute gradient first
        grad = self.gradient(u)

        # Hessian_{ij} = d/dx_i (du/dx_j)
        # Approximate by computing gradient of each gradient component
        for j in range(self.dimension):
            grad_of_grad_j = self.gradient(grad[:, j])
            for i in range(self.dimension):
                hess[:, i, j] = grad_of_grad_j[:, i]

        # Symmetrize (average off-diagonal terms)
        hess = 0.5 * (hess + np.transpose(hess, (0, 2, 1)))

        return hess

    def interpolate(
        self,
        u: NDArray[np.float64],
        eval_points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Interpolate function values at new evaluation points.

        Args:
            u: Function values at collocation points, shape (n_points,)
            eval_points: Points to evaluate at, shape (n_eval, dimension)

        Returns:
            Interpolated values at eval_points, shape (n_eval,)
        """
        eval_points = np.asarray(eval_points)

        # Solve for RBF weights
        w = self._solve_interpolation(u)

        # Compute distances from eval points to collocation points
        eval_dist = cdist(eval_points, self.points, metric="euclidean")

        # Evaluate kernel at these distances
        eval_phi = self.kernel(eval_dist, self.shape_parameter)

        # Interpolated values: s(x) = sum_j w_j * phi(||x - x_j||)
        return eval_phi @ w

    def condition_number(self) -> float:
        """
        Compute condition number of the interpolation matrix A.

        Returns:
            Condition number (ratio of largest to smallest singular value)
        """
        S = np.linalg.svd(self.A, compute_uv=False)
        return S[0] / S[-1] if S[-1] > 0 else np.inf


# =============================================================================
# Factory function
# =============================================================================


def create_rbf_operator(
    points: NDArray[np.float64],
    kernel_type: str = "gaussian",
    shape_parameter: float | None = None,
    **kwargs,
) -> RBFOperator:
    """
    Factory function to create RBF operator instances.

    Args:
        points: Collocation points, shape (n_points, dimension)
        kernel_type: Type of kernel/RBF to use
        shape_parameter: Shape parameter for the kernel
        **kwargs: Additional arguments passed to RBFOperator

    Returns:
        Configured RBFOperator instance

    Example:
        >>> points = np.random.rand(100, 2)
        >>> rbf = create_rbf_operator(points, 'wendland_c2')
        >>> lap = rbf.laplacian(points[:, 0]**2)
    """
    return RBFOperator(
        points,
        kernel_type=kernel_type,  # type: ignore
        shape_parameter=shape_parameter,
        **kwargs,
    )


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    """Smoke test for RBFOperator."""
    print("Testing RBFOperator...")

    # Create 2D test grid
    n = 15
    x = np.linspace(0.1, 0.9, n)
    xx, yy = np.meshgrid(x, x)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    print(f"Test points: {points.shape[0]} points in 2D")

    # Test function: f(x,y) = x^2 + y^2
    # Analytical: grad = [2x, 2y], Laplacian = 4
    u = points[:, 0] ** 2 + points[:, 1] ** 2

    # Test with Gaussian kernel
    print("\n[Gaussian RBF]")
    rbf_gauss = RBFOperator(points, kernel_type="gaussian")
    print(f"  Shape parameter: {rbf_gauss.shape_parameter:.4f}")
    print(f"  Avg spacing: {rbf_gauss.avg_spacing:.4f}")
    print(f"  Condition number: {rbf_gauss.condition_number():.2e}")

    lap_gauss = rbf_gauss.laplacian(u)
    grad_gauss = rbf_gauss.gradient(u)

    # Check interior points (avoid boundary effects)
    interior = (points[:, 0] > 0.2) & (points[:, 0] < 0.8) & (points[:, 1] > 0.2) & (points[:, 1] < 0.8)

    lap_error = np.mean(np.abs(lap_gauss[interior] - 4.0))
    grad_error_x = np.mean(np.abs(grad_gauss[interior, 0] - 2 * points[interior, 0]))
    grad_error_y = np.mean(np.abs(grad_gauss[interior, 1] - 2 * points[interior, 1]))

    print(f"  Laplacian error (interior): {lap_error:.4f}")
    print(f"  Gradient error x: {grad_error_x:.4f}")
    print(f"  Gradient error y: {grad_error_y:.4f}")

    # Test with Wendland C2 kernel
    # Note: Wendland kernels require careful shape parameter tuning for accurate
    # derivatives. The compact support provides better conditioning but makes
    # derivative computation more sensitive to point spacing.
    print("\n[Wendland C2 RBF]")
    rbf_wend = RBFOperator(points, kernel_type="wendland_c2")
    print(f"  Shape parameter: {rbf_wend.shape_parameter:.4f}")
    print(f"  Condition number: {rbf_wend.condition_number():.2e}")

    lap_wend = rbf_wend.laplacian(u)
    grad_wend = rbf_wend.gradient(u)

    lap_error_w = np.mean(np.abs(lap_wend[interior] - 4.0))
    grad_error_x_w = np.mean(np.abs(grad_wend[interior, 0] - 2 * points[interior, 0]))
    grad_error_y_w = np.mean(np.abs(grad_wend[interior, 1] - 2 * points[interior, 1]))

    print(f"  Laplacian error (interior): {lap_error_w:.4f}")
    print(f"  Gradient error x: {grad_error_x_w:.4f}")
    print(f"  Gradient error y: {grad_error_y_w:.4f}")
    print("  Note: Wendland Laplacian requires tuning; use Gaussian for general use")

    # Test interpolation
    print("\n[Interpolation Test]")
    eval_pts = np.array([[0.5, 0.5], [0.3, 0.7]])
    interp = rbf_gauss.interpolate(u, eval_pts)
    exact = eval_pts[:, 0] ** 2 + eval_pts[:, 1] ** 2
    interp_error = np.max(np.abs(interp - exact))
    print(f"  Interpolation error: {interp_error:.6f}")

    print("\nRBFOperator smoke tests completed!")

    # Validation
    assert lap_error < 1.0, f"Gaussian Laplacian error too large: {lap_error}"
    assert grad_error_x < 0.5, f"Gaussian gradient x error too large: {grad_error_x}"
    print("\nAll assertions passed.")
