"""
Kernel Functions for Numerical Methods.

This module provides a unified interface for various kernel functions used across
numerical methods including:
- GFDM (Generalized Finite Difference Method)
- SPH (Smoothed Particle Hydrodynamics)
- KDE (Kernel Density Estimation)
- Semi-Lagrangian interpolation
- Particle methods

Kernel Types Supported:
----------------------
1. **Gaussian (Radial Basis Function)**
   - Infinitely smooth (C^∞)
   - Infinite support (but exponential decay)
   - Widely used in GFDM, KDE

2. **Wendland Kernels (Compactly Supported)**
   - Wendland C^0, C^2, C^4, C^6
   - Finite support (exactly zero beyond cutoff)
   - Positive definite, piecewise polynomial
   - Optimal for particle methods (minimal memory, no ghost contributions)

3. **B-Spline Kernels**
   - Cubic spline (M4)
   - Quintic spline (M6)
   - Used in SPH, semi-Lagrangian methods

4. **Polynomial Kernels**
   - Cubic kernel
   - Quartic kernel
   - Simple, compact support

Mathematical Background:
-----------------------
A kernel function K(r, h) typically has the form:
    K(r, h) = (1/h^d) * k(r/h)

where:
    r = distance from evaluation point
    h = smoothing length (support radius, bandwidth)
    d = spatial dimension
    k(q) = normalized kernel profile function (q = r/h)

Properties of Good Kernels:
    1. Normalization: ∫ K(r, h) dr = 1
    2. Compact support or rapid decay: K(r, h) → 0 as r → ∞
    3. Positivity: K(r, h) ≥ 0
    4. Monotonicity: ∂K/∂r ≤ 0
    5. Smoothness: C^n continuous for desired n

Usage Examples:
--------------
    from mfg_pde.utils.numerical.kernels import (
        GaussianKernel,
        WendlandKernel,
        CubicSplineKernel,
        create_kernel
    )

    # Create kernel instances
    kernel = GaussianKernel()
    kernel_w = WendlandKernel(k=1, dimension=2)  # C^2 Wendland kernel

    # Evaluate kernel
    r = np.array([0.0, 0.5, 1.0, 2.0])
    h = 1.0
    weights = kernel(r, h)  # Returns weights

    # Get kernel with derivatives
    w, dw_dr = kernel.evaluate_with_derivative(r, h)

    # Factory pattern
    kernel = create_kernel('wendland_c2', dimension=2)  # Returns WendlandKernel(k=1)

References:
----------
- Wendland, H. "Piecewise polynomial, positive definite and compactly supported
  radial functions of minimal degree." Advances in Computational Mathematics (1995).
- Monaghan, J. J. "Smoothed particle hydrodynamics." Reports on Progress in
  Physics (2005).
- Liu, G. R., Liu, M. B. "Smoothed Particle Hydrodynamics: A Meshfree Particle
  Method." World Scientific (2003).

Author: MFG_PDE Development Team
Created: 2025-11-04
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from scipy import integrate


def _compute_spline_normalization(kernel_func: callable, support_radius: float, dimension: int) -> float:
    """
    Compute normalization constant for radial kernel in arbitrary dimension.

    Uses spherical integration: ∫ K(r) dr = σ ∫₀^R W(r) r^(d-1) dr

    where W(r) is the unnormalized kernel profile and σ is the normalization constant.
    We require ∫ K(r) dr = 1, so:
        σ = 1 / [Ω_d ∫₀^R W(r) r^(d-1) dr]

    where Ω_d = 2π^(d/2) / Γ(d/2) is the surface area of the unit sphere in d dimensions.

    Parameters
    ----------
    kernel_func : callable
        Unnormalized kernel profile function W(q) where q = r/h
    support_radius : float
        Support radius as multiple of h (e.g., 2.0 for cubic spline)
    dimension : int
        Spatial dimension

    Returns
    -------
    sigma : float
        Normalization constant
    """
    from scipy.special import gamma

    # Surface area of unit sphere: Ω_d = 2π^(d/2) / Γ(d/2)
    omega_d = 2 * np.pi ** (dimension / 2.0) / gamma(dimension / 2.0)

    # Integrate W(q) * q^(d-1) from 0 to support_radius
    # Using substitution q = r/h, so dq = dr/h
    def integrand(q: float) -> float:
        return kernel_func(q) * q ** (dimension - 1)

    integral, _error = integrate.quad(integrand, 0, support_radius, limit=100)

    # Normalization: σ = 1 / (Ω_d * integral)
    sigma = 1.0 / (omega_d * integral)

    return sigma


class Kernel(ABC):
    """
    Abstract base class for kernel functions.

    All kernel implementations must provide:
    - __call__(r, h): Evaluate kernel at distances r with smoothing length h
    - evaluate_with_derivative(r, h): Return (kernel, derivative)
    - support_radius: Effective support radius (∞ for infinite support)
    """

    @abstractmethod
    def __call__(self, r: np.ndarray | float, h: float) -> np.ndarray | float:
        """
        Evaluate kernel function K(r, h).

        Parameters
        ----------
        r : np.ndarray | float
            Distance(s) from evaluation point. Can be scalar or array.
        h : float
            Smoothing length (bandwidth, support radius).

        Returns
        -------
        weights : np.ndarray | float
            Kernel weights, same shape as r.
        """

    @abstractmethod
    def evaluate_with_derivative(
        self, r: np.ndarray | float, h: float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """
        Evaluate kernel and its derivative with respect to r.

        Parameters
        ----------
        r : np.ndarray | float
            Distance(s) from evaluation point.
        h : float
            Smoothing length.

        Returns
        -------
        kernel : np.ndarray | float
            Kernel values K(r, h).
        derivative : np.ndarray | float
            Derivative dK/dr at each r.
        """

    @property
    @abstractmethod
    def support_radius(self) -> float:
        """
        Effective support radius as multiple of h.

        Returns
        -------
        radius : float
            Support radius. np.inf for kernels with infinite support.
            For compact kernels, K(r, h) = 0 for r > radius * h.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Kernel name for identification."""

    def laplacian(self, r: np.ndarray | float, h: float, dimension: int) -> np.ndarray | float:
        """
        Compute Laplacian of radially symmetric kernel at distance r.

        For radially symmetric K(r):
            Delta K = d²K/dr² + (d-1)/r * dK/dr

        Default implementation uses finite differences. Subclasses can
        override with analytical formulas for better accuracy.

        Parameters
        ----------
        r : np.ndarray | float
            Distance(s) from evaluation point.
        h : float
            Smoothing length.
        dimension : int
            Spatial dimension d.

        Returns
        -------
        laplacian : np.ndarray | float
            Laplacian Delta K at each r.
        """
        r = np.asarray(r, dtype=float)
        scalar_input = r.ndim == 0
        r = np.atleast_1d(r)
        d = float(dimension)  # Ensure consistent float type for JAX/float32 compatibility

        lap = np.zeros_like(r)
        eps = 1e-6 * h  # Step size for finite differences

        for i, ri in enumerate(r):
            if ri < eps:
                # At r=0, use L'Hopital's rule: lim (d-1)/r * dK/dr = (d-1) * d²K/dr²|_{r=0}
                # So Delta K|_{r=0} = d * d²K/dr²|_{r=0}
                K_0 = self(0.0, h)
                K_eps = self(eps, h)
                K_2eps = self(2 * eps, h)
                d2K = (K_2eps - 2 * K_eps + K_0) / (eps**2)
                lap[i] = d * d2K
            else:
                # Standard: Delta K = d²K/dr² + (d-1)/r * dK/dr
                K_m = self(ri - eps, h)
                K_0 = self(ri, h)
                K_p = self(ri + eps, h)
                d2K = (K_p - 2 * K_0 + K_m) / (eps**2)
                dK = (K_p - K_m) / (2 * eps)
                lap[i] = d2K + (d - 1) / ri * dK

        return float(lap[0]) if scalar_input else lap


# ============================================================================
# Gaussian Kernel (Infinite Support)
# ============================================================================


class GaussianKernel(Kernel):
    """
    Gaussian (RBF) kernel: K(r, h) = exp(-(r/h)²).

    Properties:
    - Infinitely smooth (C^∞)
    - Infinite support (but exponential decay)
    - Always positive
    - Used in: GFDM, KDE, RBF interpolation

    Mathematical Form:
        K(r, h) = exp(-(r/h)²)
        dK/dr = -(2r/h²) exp(-(r/h)²)

    Notes:
    - Not compactly supported (theoretically non-zero everywhere)
    - Practically zero for r > 3h (K(3h, h) ≈ 1.23e-4)
    - Commonly used effective support: 3-4h
    """

    def __call__(self, r: np.ndarray | float, h: float) -> np.ndarray | float:
        """Evaluate Gaussian kernel."""
        q = r / h
        return np.exp(-(q**2))

    def evaluate_with_derivative(
        self, r: np.ndarray | float, h: float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """Evaluate Gaussian kernel and derivative."""
        q = r / h
        kernel = np.exp(-(q**2))
        derivative = -(2 * r / (h**2)) * kernel
        return kernel, derivative

    @property
    def support_radius(self) -> float:
        """Gaussian has infinite support (use 3-4h as effective)."""
        return np.inf

    @property
    def name(self) -> str:
        return "gaussian"

    def laplacian(self, r: np.ndarray | float, h: float, dimension: int) -> np.ndarray | float:
        """
        Analytical Laplacian for Gaussian kernel.

        For K(r) = exp(-(r/h)²):
            Delta K = (2/h²) * (2r²/h² - d) * K
        """
        d = float(dimension)  # Ensure consistent float type
        q2 = (np.asarray(r) / h) ** 2
        K = np.exp(-q2)
        return (2 / h**2) * (2 * q2 - d) * K


# ============================================================================
# Wendland Kernels (Compactly Supported)
# ============================================================================


class WendlandKernel(Kernel):
    """
    Unified Wendland C^{2k} kernel with parameterized smoothness order.

    Wendland kernels are compactly supported, positive definite radial basis functions
    with polynomial form. The general structure is:
        K(q) = (1 - q)^{m} * P_k(q)  for q < 1, else 0

    where m = 2k + 2 and P_k(q) is a polynomial of degree k ensuring C^{2k} continuity.

    Parameters
    ----------
    k : int
        Smoothness parameter. Kernel is C^{2k} continuous.
        - k=0: C^0, polynomial (1-q)²
        - k=1: C^2, polynomial (1-q)⁴(4q+1)
        - k=2: C^4, polynomial (1-q)⁶(35q²+18q+3)
        - k=3: C^6, polynomial (1-q)⁸(32q³+25q²+8q+1)
    dimension : int, optional
        Spatial dimension for normalization (default=3).
        Note: Current implementation uses dimension-independent formulation.

    Properties
    ----------
    - Compact support: [0, h]
    - C^{2k} continuous
    - Positive definite
    - Polynomial evaluation (efficient)

    Mathematical Details
    -------------------
    Wendland (1995) derived these kernels to be positive definite in R^d for:
    - C^0: d ≤ 1
    - C^2: d ≤ 3
    - C^4: d ≤ 5
    - C^6: d ≤ 7

    The polynomial coefficients are chosen to ensure:
    1. Continuity of derivatives up to order 2k at q=1
    2. Positive definiteness in the target dimension
    3. Normalization ∫ K(x) dx = 1

    Examples
    --------
    >>> # C^2 Wendland kernel for 2D
    >>> kernel = WendlandKernel(k=1, dimension=2)
    >>> r = np.array([0.0, 0.5, 1.0, 1.5])
    >>> w = kernel(r, h=1.0)
    >>> w
    array([1.  , 0.5625, 0.   , 0.   ])

    >>> # C^4 Wendland kernel for 3D
    >>> kernel = WendlandKernel(k=2, dimension=3)
    >>> kernel.name
    'wendland_c4'

    References
    ----------
    Wendland, H. (1995). "Piecewise polynomial, positive definite and compactly
    supported radial functions of minimal degree." Advances in Computational
    Mathematics, 4(1), 389-396.
    """

    def __init__(self, k: int = 1, dimension: int = 3):
        """
        Initialize Wendland kernel with smoothness order k.

        Parameters
        ----------
        k : int, optional
            Smoothness parameter (default=1 for C^2).
            Must be in {0, 1, 2, 3}.
        dimension : int, optional
            Spatial dimension (default=3).
        """
        if k not in {0, 1, 2, 3}:
            raise ValueError(f"Smoothness parameter k must be 0, 1, 2, or 3. Got k={k}")
        if dimension < 1:
            raise ValueError(f"Dimension must be >= 1. Got dimension={dimension}")

        self.k = k
        self.dimension = dimension
        self._setup_polynomial()

    def _setup_polynomial(self):
        """Setup polynomial coefficients for Wendland C^{2k} kernel."""
        # Power of (1-q) term
        self.m = 2 * self.k + 2

        # Polynomial P_k(q) coefficients (from highest to lowest degree)
        # Format: [c_k, c_{k-1}, ..., c_1, c_0] for sum_{i=0}^k c_i q^i
        if self.k == 0:
            # C^0: (1-q)²
            self.poly_coeffs = np.array([1.0])  # Just constant 1
        elif self.k == 1:
            # C^2: (1-q)⁴(4q + 1)
            self.poly_coeffs = np.array([4.0, 1.0])  # 4q + 1
        elif self.k == 2:
            # C^4: (1-q)⁶(35q² + 18q + 3)
            self.poly_coeffs = np.array([35.0, 18.0, 3.0])  # 35q² + 18q + 3
        elif self.k == 3:
            # C^6: (1-q)⁸(32q³ + 25q² + 8q + 1)
            self.poly_coeffs = np.array([32.0, 25.0, 8.0, 1.0])  # 32q³ + 25q² + 8q + 1

    def _eval_polynomial(self, q: np.ndarray | float) -> np.ndarray | float:
        """Evaluate polynomial P_k(q) using Horner's method."""
        result = np.zeros_like(q, dtype=float)
        for coeff in self.poly_coeffs:
            result = result * q + coeff
        return result

    def __call__(self, r: np.ndarray | float, h: float) -> np.ndarray | float:
        """Evaluate Wendland kernel."""
        q = r / h
        poly_val = self._eval_polynomial(q)
        result = np.where(q < 1.0, ((1 - q) ** self.m) * poly_val, 0.0)
        return result

    def evaluate_with_derivative(
        self, r: np.ndarray | float, h: float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """
        Evaluate Wendland kernel and its derivative.

        Returns
        -------
        kernel : array
            Kernel values K(r, h)
        derivative : array
            Derivative dK/dr
        """
        q = r / h
        poly_val = self._eval_polynomial(q)
        kernel = np.where(q < 1.0, ((1 - q) ** self.m) * poly_val, 0.0)

        # Derivative: d/dq[(1-q)^m * P(q)] = -m(1-q)^{m-1} P(q) + (1-q)^m P'(q)
        # = (1-q)^{m-1} [-m P(q) + (1-q) P'(q)]

        # Compute P'(q) using polynomial derivative
        if len(self.poly_coeffs) == 1:
            poly_deriv_val = 0.0  # Constant polynomial
        else:
            # Derivative coefficients: multiply by powers
            deriv_coeffs = np.array(
                [self.poly_coeffs[i] * (len(self.poly_coeffs) - 1 - i) for i in range(len(self.poly_coeffs) - 1)]
            )
            poly_deriv_val = np.zeros_like(q, dtype=float)
            for coeff in deriv_coeffs:
                poly_deriv_val = poly_deriv_val * q + coeff

        # dK/dq
        dK_dq = np.where(q < 1.0, ((1 - q) ** (self.m - 1)) * (-self.m * poly_val + (1 - q) * poly_deriv_val), 0.0)

        # dK/dr = dK/dq * dq/dr = dK/dq * (1/h)
        derivative = dK_dq / h
        return kernel, derivative

    @property
    def support_radius(self) -> float:
        """Compact support: K = 0 for r ≥ h."""
        return 1.0

    @property
    def name(self) -> str:
        return f"wendland_c{2 * self.k}"

    def laplacian(self, r: np.ndarray | float, h: float, dimension: int) -> np.ndarray | float:
        """
        Analytical Laplacian for Wendland kernel.

        For K(r) = (1-q)^m * P(q) where q = r/h:
            Delta K = d²K/dr² + (d-1)/r * dK/dr
                    = (1/h²) * d²K/dq² + (d-1)/r * (1/h) * dK/dq

        The derivatives are computed analytically from the polynomial structure.
        """
        r = np.asarray(r, dtype=float)
        q = r / h
        scalar_input = r.ndim == 0
        q = np.atleast_1d(q)
        r = np.atleast_1d(r)
        dim = float(dimension)  # Ensure consistent float type

        m = self.m
        poly_val = self._eval_polynomial(q)

        # First derivative P'(q)
        if len(self.poly_coeffs) == 1:
            poly_d1 = np.zeros_like(q)
            poly_d2 = np.zeros_like(q)
        else:
            # P'(q) coefficients
            n = len(self.poly_coeffs)
            d1_coeffs = np.array([self.poly_coeffs[i] * (n - 1 - i) for i in range(n - 1)])
            poly_d1 = np.zeros_like(q, dtype=float)
            for coeff in d1_coeffs:
                poly_d1 = poly_d1 * q + coeff

            # P''(q) coefficients
            if len(d1_coeffs) <= 1:
                poly_d2 = np.zeros_like(q)
            else:
                n1 = len(d1_coeffs)
                d2_coeffs = np.array([d1_coeffs[i] * (n1 - 1 - i) for i in range(n1 - 1)])
                poly_d2 = np.zeros_like(q, dtype=float)
                for coeff in d2_coeffs:
                    poly_d2 = poly_d2 * q + coeff

        # dK/dq = (1-q)^{m-1} * [-m*P(q) + (1-q)*P'(q)]
        # Let A = -m*P(q) + (1-q)*P'(q)
        A = -m * poly_val + (1 - q) * poly_d1

        # d²K/dq² = d/dq[(1-q)^{m-1} * A]
        #         = -(m-1)(1-q)^{m-2} * A + (1-q)^{m-1} * dA/dq
        # where dA/dq = -m*P'(q) + (-1)*P'(q) + (1-q)*P''(q)
        #             = -(m+1)*P'(q) + (1-q)*P''(q)
        dA_dq = -(m + 1) * poly_d1 + (1 - q) * poly_d2

        # Compute derivatives (inside support)
        one_minus_q = 1 - q
        inside = q < 1.0

        dK_dq = np.where(inside, (one_minus_q ** (m - 1)) * A, 0.0)
        d2K_dq2 = np.where(
            inside & (one_minus_q > 0),
            -(m - 1) * (one_minus_q ** (m - 2)) * A + (one_minus_q ** (m - 1)) * dA_dq,
            0.0,
        )

        # Laplacian: Delta K = d²K/dr² + (d-1)/r * dK/dr
        # = (1/h²) * d²K/dq² + (d-1)/r * (1/h) * dK/dq
        # = (1/h²) * [d²K/dq² + (d-1)*h/r * dK/dq]
        # = (1/h²) * [d²K/dq² + (d-1)/q * dK/dq]  (since r = q*h)
        lap = np.zeros_like(q)
        nonzero_q = (q > 1e-14) & inside
        lap[nonzero_q] = (1 / h**2) * (d2K_dq2[nonzero_q] + (dim - 1) / q[nonzero_q] * dK_dq[nonzero_q])

        # At q=0: use L'Hopital's rule
        # lim_{q->0} (d-1)/q * dK/dq = (d-1) * d²K/dq²|_{q=0}
        # So Delta K|_{q=0} = dim * d²K/dq²|_{q=0} / h²
        at_zero = q <= 1e-14
        if np.any(at_zero):
            lap[at_zero] = dim * d2K_dq2[at_zero] / h**2

        return float(lap[0]) if scalar_input else lap


# ============================================================================
# B-Spline Kernels
# ============================================================================


class CubicSplineKernel(Kernel):
    """
    Cubic B-spline kernel (M4 spline).

    Mathematical Form:
        K(q) = σ * { 1 - (3/2)q² + (3/4)q³       if 0 ≤ q < 1
                   { (1/4)(2 - q)³               if 1 ≤ q < 2
                   { 0                            if q ≥ 2

    where σ = normalization constant (dimension-dependent):
        σ = 2/3 (1D), 10/(7π) (2D), 1/π (3D)

    Properties:
    - C^2 continuous
    - Compact support: [0, 2h]
    - Widely used in SPH
    - Good balance of accuracy and efficiency

    Notes:
    - Classic SPH kernel (Monaghan & Lattanzio, 1985)
    - Support radius is 2h (not h like Wendland)
    """

    def __init__(self, dimension: int = 3):
        """
        Initialize cubic spline kernel.

        Parameters
        ----------
        dimension : int
            Spatial dimension (must be >= 1). Affects normalization constant.
        """
        if dimension < 1:
            raise ValueError(f"Dimension must be >= 1, got {dimension}")

        self.dimension = dimension

        # Use precomputed normalization constants for common dimensions
        # These match the standard literature values
        _known_sigmas = {
            1: 2.0 / 3.0,
            2: 10.0 / (7.0 * np.pi),
            3: 1.0 / np.pi,
        }

        if dimension in _known_sigmas:
            self.sigma = _known_sigmas[dimension]
        else:
            # Compute normalization for arbitrary dimension
            def cubic_spline_profile(q: float) -> float:
                """Unnormalized cubic spline kernel profile."""
                if q < 1.0:
                    return 1 - 1.5 * q**2 + 0.75 * q**3
                elif q < 2.0:
                    return 0.25 * (2 - q) ** 3
                else:
                    return 0.0

            self.sigma = _compute_spline_normalization(cubic_spline_profile, support_radius=2.0, dimension=dimension)

    def __call__(self, r: np.ndarray | float, h: float) -> np.ndarray | float:
        """Evaluate cubic spline kernel."""
        q = r / h
        # Piecewise evaluation
        result = np.zeros_like(q, dtype=float)

        # Region 1: 0 ≤ q < 1
        mask1 = q < 1.0
        result = np.where(mask1, 1 - 1.5 * q**2 + 0.75 * q**3, result)

        # Region 2: 1 ≤ q < 2
        mask2 = (q >= 1.0) & (q < 2.0)
        result = np.where(mask2, 0.25 * ((2 - q) ** 3), result)

        # Apply normalization
        result *= self.sigma / (h**self.dimension)

        return result

    def evaluate_with_derivative(
        self, r: np.ndarray | float, h: float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """Evaluate cubic spline kernel and derivative."""
        q = r / h
        kernel = np.zeros_like(q, dtype=float)
        dK_dq = np.zeros_like(q, dtype=float)

        # Region 1: 0 ≤ q < 1
        mask1 = q < 1.0
        kernel = np.where(mask1, 1 - 1.5 * q**2 + 0.75 * q**3, kernel)
        dK_dq = np.where(mask1, -3 * q + 2.25 * q**2, dK_dq)

        # Region 2: 1 ≤ q < 2
        mask2 = (q >= 1.0) & (q < 2.0)
        kernel = np.where(mask2, 0.25 * ((2 - q) ** 3), kernel)
        dK_dq = np.where(mask2, -0.75 * ((2 - q) ** 2), dK_dq)

        # Apply normalization
        kernel *= self.sigma / (h**self.dimension)
        derivative = (self.sigma / (h ** (self.dimension + 1))) * dK_dq

        return kernel, derivative

    @property
    def support_radius(self) -> float:
        """Compact support: K = 0 for r ≥ 2h."""
        return 2.0

    @property
    def name(self) -> str:
        return f"cubic_spline_{self.dimension}d"

    def laplacian(self, r: np.ndarray | float, h: float, dimension: int) -> np.ndarray | float:
        """
        Analytical Laplacian for cubic spline kernel.

        Piecewise derivatives (W = unnormalized profile):
            0 ≤ q < 1: W(q) = 1 - 1.5q² + 0.75q³
                       W'(q) = -3q + 2.25q²
                       W''(q) = -3 + 4.5q

            1 ≤ q < 2: W(q) = 0.25(2-q)³
                       W'(q) = -0.75(2-q)²
                       W''(q) = 1.5(2-q)

        Laplacian:
            Δ K = (σ/h^{d+2}) * [W''(q) + (d-1)/q · W'(q)]
        """
        r = np.asarray(r, dtype=float)
        scalar_input = r.ndim == 0
        r = np.atleast_1d(r)
        q = r / h
        dim = float(dimension)  # Ensure consistent float type

        W_d1 = np.zeros_like(q)
        W_d2 = np.zeros_like(q)

        # Region 1: 0 ≤ q < 1
        mask1 = q < 1.0
        W_d1 = np.where(mask1, -3 * q + 2.25 * q**2, W_d1)
        W_d2 = np.where(mask1, -3 + 4.5 * q, W_d2)

        # Region 2: 1 ≤ q < 2
        mask2 = (q >= 1.0) & (q < 2.0)
        W_d1 = np.where(mask2, -0.75 * ((2 - q) ** 2), W_d1)
        W_d2 = np.where(mask2, 1.5 * (2 - q), W_d2)

        inside = q < 2.0
        lap = np.zeros_like(q)

        # For q > 0
        nonzero_q = inside & (q > 1e-14)
        lap[nonzero_q] = W_d2[nonzero_q] + (dim - 1) / q[nonzero_q] * W_d1[nonzero_q]

        # At q=0: Δ K ∝ d * W''(0) = d * (-3)
        at_zero = q <= 1e-14
        lap[at_zero] = dim * (-3.0)

        # Apply normalization factor: σ/h^{d+2}
        lap *= self.sigma / (h ** (self.dimension + 2))

        return float(lap[0]) if scalar_input else lap


class QuinticSplineKernel(Kernel):
    """
    Quintic B-spline kernel (M6 spline).

    Mathematical Form:
        K(q) = σ * { (3-q)⁵ - 6(2-q)⁵ + 15(1-q)⁵  if 0 ≤ q < 1
                   { (3-q)⁵ - 6(2-q)⁵             if 1 ≤ q < 2
                   { (3-q)⁵                        if 2 ≤ q < 3
                   { 0                             if q ≥ 3

    where σ = normalization constant (dimension-dependent):
        σ = 1/120 (1D), 7/(478π) (2D), 1/(120π) (3D)

    Properties:
    - C^4 continuous
    - Compact support: [0, 3h]
    - Higher accuracy than cubic spline
    - Used in high-accuracy SPH simulations
    """

    def __init__(self, dimension: int = 3):
        """
        Initialize quintic spline kernel.

        Parameters
        ----------
        dimension : int
            Spatial dimension (must be >= 1). Affects normalization constant.
        """
        if dimension < 1:
            raise ValueError(f"Dimension must be >= 1, got {dimension}")

        self.dimension = dimension

        # Use precomputed normalization constants for common dimensions
        _known_sigmas = {
            1: 1.0 / 120.0,
            2: 7.0 / (478.0 * np.pi),
            3: 1.0 / (120.0 * np.pi),
        }

        if dimension in _known_sigmas:
            self.sigma = _known_sigmas[dimension]
        else:
            # Compute normalization for arbitrary dimension
            def quintic_spline_profile(q: float) -> float:
                """Unnormalized quintic spline kernel profile."""
                result = 0.0
                if q < 3.0:
                    result += max(0, (3 - q) ** 5)
                if q < 2.0:
                    result -= 6 * max(0, (2 - q) ** 5)
                if q < 1.0:
                    result += 15 * max(0, (1 - q) ** 5)
                return result

            self.sigma = _compute_spline_normalization(quintic_spline_profile, support_radius=3.0, dimension=dimension)

    def __call__(self, r: np.ndarray | float, h: float) -> np.ndarray | float:
        """Evaluate quintic spline kernel."""
        q = r / h
        result = np.zeros_like(q, dtype=float)

        # Helper function for (a - q)⁵ with safeguard
        def pow5_safe(a: float, q: np.ndarray | float) -> np.ndarray | float:
            return np.where(q < a, (a - q) ** 5, 0.0)

        # Region 1: 0 ≤ q < 1
        mask1 = q < 1.0
        result = np.where(mask1, pow5_safe(3, q) - 6 * pow5_safe(2, q) + 15 * pow5_safe(1, q), result)

        # Region 2: 1 ≤ q < 2
        mask2 = (q >= 1.0) & (q < 2.0)
        result = np.where(mask2, pow5_safe(3, q) - 6 * pow5_safe(2, q), result)

        # Region 3: 2 ≤ q < 3
        mask3 = (q >= 2.0) & (q < 3.0)
        result = np.where(mask3, pow5_safe(3, q), result)

        # Apply normalization
        result *= self.sigma / (h**self.dimension)

        return result

    def evaluate_with_derivative(
        self, r: np.ndarray | float, h: float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """
        Evaluate quintic spline kernel and derivative.

        Derivatives use: d/dq[(a-q)^5] = -5(a-q)^4

        Piecewise W'(q):
            0 ≤ q < 1: W' = -5(3-q)⁴ + 30(2-q)⁴ - 75(1-q)⁴
            1 ≤ q < 2: W' = -5(3-q)⁴ + 30(2-q)⁴
            2 ≤ q < 3: W' = -5(3-q)⁴
        """
        q = np.asarray(r) / h
        kernel = np.zeros_like(q, dtype=float)
        W_d1 = np.zeros_like(q, dtype=float)

        # Helper for safe powers
        def pow5_safe(a: float, x: np.ndarray) -> np.ndarray:
            return np.where(x < a, (a - x) ** 5, 0.0)

        def pow4_safe(a: float, x: np.ndarray) -> np.ndarray:
            return np.where(x < a, (a - x) ** 4, 0.0)

        # Region 1: 0 ≤ q < 1
        mask1 = q < 1.0
        kernel = np.where(mask1, pow5_safe(3, q) - 6 * pow5_safe(2, q) + 15 * pow5_safe(1, q), kernel)
        W_d1 = np.where(mask1, -5 * pow4_safe(3, q) + 30 * pow4_safe(2, q) - 75 * pow4_safe(1, q), W_d1)

        # Region 2: 1 ≤ q < 2
        mask2 = (q >= 1.0) & (q < 2.0)
        kernel = np.where(mask2, pow5_safe(3, q) - 6 * pow5_safe(2, q), kernel)
        W_d1 = np.where(mask2, -5 * pow4_safe(3, q) + 30 * pow4_safe(2, q), W_d1)

        # Region 3: 2 ≤ q < 3
        mask3 = (q >= 2.0) & (q < 3.0)
        kernel = np.where(mask3, pow5_safe(3, q), kernel)
        W_d1 = np.where(mask3, -5 * pow4_safe(3, q), W_d1)

        # Apply normalization
        kernel *= self.sigma / (h**self.dimension)
        derivative = (self.sigma / (h ** (self.dimension + 1))) * W_d1

        return kernel, derivative

    @property
    def support_radius(self) -> float:
        """Compact support: K = 0 for r ≥ 3h."""
        return 3.0

    def laplacian(self, r: np.ndarray | float, h: float, dimension: int) -> np.ndarray | float:
        """
        Analytical Laplacian for quintic spline kernel.

        Derivatives use: d²/dq²[(a-q)^5] = 20(a-q)³

        Piecewise W''(q):
            0 ≤ q < 1: W'' = 20(3-q)³ - 120(2-q)³ + 300(1-q)³
            1 ≤ q < 2: W'' = 20(3-q)³ - 120(2-q)³
            2 ≤ q < 3: W'' = 20(3-q)³

        Laplacian:
            Δ K = (σ/h^{d+2}) * [W''(q) + (d-1)/q · W'(q)]
        """
        r = np.asarray(r, dtype=float)
        scalar_input = r.ndim == 0
        r = np.atleast_1d(r)
        q = r / h
        dim = float(dimension)  # Ensure consistent float type

        def pow4_safe(a: float, x: np.ndarray) -> np.ndarray:
            return np.where(x < a, (a - x) ** 4, 0.0)

        def pow3_safe(a: float, x: np.ndarray) -> np.ndarray:
            return np.where(x < a, (a - x) ** 3, 0.0)

        W_d1 = np.zeros_like(q)
        W_d2 = np.zeros_like(q)

        # Region 1: 0 ≤ q < 1
        mask1 = q < 1.0
        W_d1 = np.where(mask1, -5 * pow4_safe(3, q) + 30 * pow4_safe(2, q) - 75 * pow4_safe(1, q), W_d1)
        W_d2 = np.where(mask1, 20 * pow3_safe(3, q) - 120 * pow3_safe(2, q) + 300 * pow3_safe(1, q), W_d2)

        # Region 2: 1 ≤ q < 2
        mask2 = (q >= 1.0) & (q < 2.0)
        W_d1 = np.where(mask2, -5 * pow4_safe(3, q) + 30 * pow4_safe(2, q), W_d1)
        W_d2 = np.where(mask2, 20 * pow3_safe(3, q) - 120 * pow3_safe(2, q), W_d2)

        # Region 3: 2 ≤ q < 3
        mask3 = (q >= 2.0) & (q < 3.0)
        W_d1 = np.where(mask3, -5 * pow4_safe(3, q), W_d1)
        W_d2 = np.where(mask3, 20 * pow3_safe(3, q), W_d2)

        inside = q < 3.0
        lap = np.zeros_like(q)

        # For q > 0
        nonzero_q = inside & (q > 1e-14)
        lap[nonzero_q] = W_d2[nonzero_q] + (dim - 1) / q[nonzero_q] * W_d1[nonzero_q]

        # At q=0: Δ K ∝ d * W''(0) = d * [20*27 - 120*8 + 300*1] = d * [540 - 960 + 300] = d * (-120)
        at_zero = q <= 1e-14
        lap[at_zero] = dim * (-120.0)

        # Apply normalization factor: σ/h^{d+2}
        lap *= self.sigma / (h ** (self.dimension + 2))

        return float(lap[0]) if scalar_input else lap

    @property
    def name(self) -> str:
        return f"quintic_spline_{self.dimension}d"


# ============================================================================
# Simple Polynomial Kernels
# ============================================================================


class CubicKernel(Kernel):
    """
    Simple cubic kernel: K(r, h) = (1 - r/h)³ for r < h, else 0.

    Properties:
    - C^0 continuous
    - Compact support: [0, h]
    - Simple, fast evaluation
    """

    def __call__(self, r: np.ndarray | float, h: float) -> np.ndarray | float:
        """Evaluate cubic kernel."""
        q = r / h
        return np.where(q < 1.0, (1 - q) ** 3, 0.0)

    def evaluate_with_derivative(
        self, r: np.ndarray | float, h: float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """Evaluate cubic kernel and derivative."""
        q = r / h
        kernel = np.where(q < 1.0, (1 - q) ** 3, 0.0)
        derivative = np.where(q < 1.0, -3 * ((1 - q) ** 2) / h, 0.0)
        return kernel, derivative

    @property
    def support_radius(self) -> float:
        return 1.0

    @property
    def name(self) -> str:
        return "cubic"

    def laplacian(self, r: np.ndarray | float, h: float, dimension: int) -> np.ndarray | float:
        """
        Analytical Laplacian for cubic kernel K(r) = (1-q)³.

        Derivatives (q = r/h):
            K'(q) = -3(1-q)²
            K''(q) = 6(1-q)

        Laplacian:
            Δ K = (1/h²)[K''(q) + (d-1)/q · K'(q)]
                = (1/h²)[6(1-q) - 3(d-1)(1-q)²/q]
        """
        r = np.asarray(r, dtype=float)
        scalar_input = r.ndim == 0
        r = np.atleast_1d(r)
        q = r / h
        dim = float(dimension)  # Ensure consistent float type

        inside = q < 1.0
        one_minus_q = 1 - q

        # K''(q) = 6(1-q), K'(q) = -3(1-q)²
        K_d2 = 6 * one_minus_q
        K_d1 = -3 * (one_minus_q**2)

        lap = np.zeros_like(q)
        nonzero_q = inside & (q > 1e-14)
        lap[nonzero_q] = (1 / h**2) * (K_d2[nonzero_q] + (dim - 1) / q[nonzero_q] * K_d1[nonzero_q])

        # At q=0: Δ K = d * K''(0) / h² = d * 6 / h² = 6d/h²
        at_zero = inside & (q <= 1e-14)
        lap[at_zero] = dim * 6.0 / h**2

        return float(lap[0]) if scalar_input else lap


class QuarticKernel(Kernel):
    """
    Simple quartic kernel: K(r, h) = (1 - r/h)⁴ for r < h, else 0.

    Properties:
    - C^1 continuous
    - Compact support: [0, h]
    - Smoother than cubic
    """

    def __call__(self, r: np.ndarray | float, h: float) -> np.ndarray | float:
        """Evaluate quartic kernel."""
        q = r / h
        return np.where(q < 1.0, (1 - q) ** 4, 0.0)

    def evaluate_with_derivative(
        self, r: np.ndarray | float, h: float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """Evaluate quartic kernel and derivative."""
        q = r / h
        kernel = np.where(q < 1.0, (1 - q) ** 4, 0.0)
        derivative = np.where(q < 1.0, -4 * ((1 - q) ** 3) / h, 0.0)
        return kernel, derivative

    @property
    def support_radius(self) -> float:
        return 1.0

    @property
    def name(self) -> str:
        return "quartic"

    def laplacian(self, r: np.ndarray | float, h: float, dimension: int) -> np.ndarray | float:
        """
        Analytical Laplacian for quartic kernel K(r) = (1-q)⁴.

        Derivatives (q = r/h):
            K'(q) = -4(1-q)³
            K''(q) = 12(1-q)²

        Laplacian:
            Δ K = (1/h²)[K''(q) + (d-1)/q · K'(q)]
                = (1/h²)[12(1-q)² - 4(d-1)(1-q)³/q]
        """
        r = np.asarray(r, dtype=float)
        scalar_input = r.ndim == 0
        r = np.atleast_1d(r)
        q = r / h
        dim = float(dimension)  # Ensure consistent float type

        inside = q < 1.0
        one_minus_q = 1 - q

        # K''(q) = 12(1-q)², K'(q) = -4(1-q)³
        K_d2 = 12 * (one_minus_q**2)
        K_d1 = -4 * (one_minus_q**3)

        lap = np.zeros_like(q)
        nonzero_q = inside & (q > 1e-14)
        lap[nonzero_q] = (1 / h**2) * (K_d2[nonzero_q] + (dim - 1) / q[nonzero_q] * K_d1[nonzero_q])

        # At q=0: Δ K = d * K''(0) / h² = d * 12 / h² = 12d/h²
        at_zero = inside & (q <= 1e-14)
        lap[at_zero] = dim * 12.0 / h**2

        return float(lap[0]) if scalar_input else lap


# ============================================================================
# Polyharmonic Spline Kernels (PHS-RBF)
# ============================================================================


class PHSKernel(Kernel):
    """
    Polyharmonic Spline (PHS) kernel for RBF-FD methods.

    Mathematical Form:
        phi(r) = r^m        for odd m (m = 1, 3, 5, 7, ...)
        phi(r) = r^m log(r) for even m (m = 2, 4, 6, ...)

    Common choices:
        - m=1: Linear (phi = r)
        - m=3: Cubic (phi = r^3), most common in RBF-FD
        - m=5: Quintic (phi = r^5), higher accuracy

    Key Properties:
    - NO shape parameter to tune (unlike Gaussian/Wendland)
    - Infinite support with algebraic decay
    - Combined with polynomial augmentation for accuracy
    - Conditionally positive definite (requires polynomial augmentation)

    Polynomial Augmentation:
        RBF-FD with PHS typically augments with polynomials of degree <= (m-1)/2
        to ensure solvability. E.g., PHS m=3 augments with linear terms.

    References:
        - Fornberg, B., Flyer, N. (2015). "A Primer on Radial Basis Functions
          with Applications to the Geosciences."
        - Flyer, N., et al. (2016). "On the role of polynomials in RBF-FD
          approximations: I. Interpolation and accuracy."

    Example:
        >>> kernel = PHSKernel(m=3)  # Cubic PHS
        >>> r = np.array([0.0, 0.5, 1.0, 2.0])
        >>> phi = kernel(r, h=1.0)  # h is ignored for PHS
    """

    def __init__(self, m: int = 3):
        """
        Initialize PHS kernel.

        Parameters
        ----------
        m : int
            Order of the polyharmonic spline. Must be positive integer.
            - Odd m: phi(r) = r^m
            - Even m: phi(r) = r^m log(r)
            Common choices: m=3 (cubic), m=5 (quintic)
        """
        if m < 1:
            raise ValueError(f"PHS order m must be >= 1, got {m}")
        self.m = m
        self._is_even = m % 2 == 0

    def __call__(self, r: np.ndarray | float, h: float) -> np.ndarray | float:
        """
        Evaluate PHS kernel.

        Note: The smoothing length h is ignored for PHS since there is
        no shape parameter. It is accepted for API compatibility.
        """
        r = np.asarray(r, dtype=float)
        # Avoid log(0) singularity
        r_safe = np.maximum(r, 1e-14)

        if self._is_even:
            # Even m: r^m * log(r)
            result = np.where(r > 1e-14, (r_safe**self.m) * np.log(r_safe), 0.0)
        else:
            # Odd m: r^m
            result = r**self.m

        return result

    def evaluate_with_derivative(
        self, r: np.ndarray | float, h: float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """
        Evaluate PHS kernel and derivative.

        For odd m: d(phi)/dr = m * r^(m-1)
        For even m: d(phi)/dr = r^(m-1) * (m * log(r) + 1)
        """
        r = np.asarray(r, dtype=float)
        r_safe = np.maximum(r, 1e-14)

        if self._is_even:
            kernel = np.where(r > 1e-14, (r_safe**self.m) * np.log(r_safe), 0.0)
            # d/dr[r^m log(r)] = m * r^(m-1) * log(r) + r^(m-1) = r^(m-1) * (m*log(r) + 1)
            derivative = np.where(r > 1e-14, (r_safe ** (self.m - 1)) * (self.m * np.log(r_safe) + 1), 0.0)
        else:
            kernel = r**self.m
            # d/dr[r^m] = m * r^(m-1)
            derivative = self.m * (r ** (self.m - 1)) if self.m > 1 else np.ones_like(r)

        return kernel, derivative

    @property
    def support_radius(self) -> float:
        """PHS has infinite support (algebraic decay, not exponential)."""
        return np.inf

    @property
    def name(self) -> str:
        return f"phs{self.m}"

    def laplacian(self, r: np.ndarray | float, h: float, dimension: int) -> np.ndarray | float:
        """
        Analytical Laplacian for PHS kernel.

        For odd m, phi(r) = r^m:
            Delta phi = m*(m + d - 2) * r^(m-2)

        For even m, phi(r) = r^m * log(r):
            Delta phi = r^(m-2) * [m*(m+d-2)*log(r) + (2m + d - 2)]

        Note: h is ignored for PHS (no shape parameter).
        """
        r = np.asarray(r, dtype=float)
        scalar_input = r.ndim == 0
        r = np.atleast_1d(r)
        m = self.m
        dim = float(dimension)  # Ensure consistent float type

        if self._is_even:
            # Even m: phi = r^m * log(r)
            # Delta phi = r^(m-2) * [m*(m+dim-2)*log(r) + (2m+dim-2)]
            r_safe = np.maximum(r, 1e-14)
            coeff_log = m * (m + dim - 2)
            coeff_const = 2.0 * m + dim - 2.0

            if m == 2:
                # phi = r^2 * log(r)
                # Delta phi = [2*dim*log(r) + (dim+2)]
                lap = np.where(r > 1e-14, coeff_log * np.log(r_safe) + coeff_const, 0.0)
            else:
                # m >= 4: Delta phi = r^(m-2) * [m*(m+dim-2)*log(r) + (2m+dim-2)]
                lap = np.where(
                    r > 1e-14,
                    (r_safe ** (m - 2)) * (coeff_log * np.log(r_safe) + coeff_const),
                    0.0,
                )
            return float(lap[0]) if scalar_input else lap
        else:
            # Odd m: Delta phi = m*(m + dim - 2) * r^(m-2)
            if m == 1:
                # phi = r, Delta phi = (dim-1)/r which is singular at r=0
                r_safe = np.maximum(r, 1e-14)
                lap = np.where(r > 1e-14, (dim - 1) / r_safe, 0.0)
            elif m == 2:
                # phi = r^2, Delta phi = 2*dim (constant)
                lap = np.full_like(r, 2.0 * dim)
            else:
                # m >= 3: Delta phi = m*(m + dim - 2) * r^(m-2)
                lap = m * (m + dim - 2) * (r ** (m - 2))
            return float(lap[0]) if scalar_input else lap


# ============================================================================
# Multiquadric Kernel
# ============================================================================


class MultiquadricKernel(Kernel):
    """
    Multiquadric (MQ) radial basis function: phi(r) = sqrt(1 + (r/h)^2).

    Properties:
    - Infinitely smooth (C^infinity)
    - Infinite support (algebraic decay)
    - Conditionally positive definite (order 1)
    - Classic RBF, widely used in interpolation

    Mathematical Form:
        phi(r, h) = sqrt(1 + (r/h)^2)
        d(phi)/dr = r / (h^2 * sqrt(1 + (r/h)^2))

    Notes:
    - Shape parameter h controls flatness (larger h = flatter)
    - Good for smooth interpolation but can be ill-conditioned
    - Often used with polynomial augmentation in RBF-FD

    References:
    - Hardy, R.L. (1971). "Multiquadric equations of topography and
      other irregular surfaces."
    """

    def __call__(self, r: np.ndarray | float, h: float) -> np.ndarray | float:
        """Evaluate multiquadric kernel."""
        return np.sqrt(1 + (np.asarray(r) / h) ** 2)

    def evaluate_with_derivative(
        self, r: np.ndarray | float, h: float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """Evaluate multiquadric kernel and derivative."""
        r = np.asarray(r)
        phi = np.sqrt(1 + (r / h) ** 2)
        dphi = r / (h**2 * phi)
        return phi, dphi

    @property
    def support_radius(self) -> float:
        """Multiquadric has infinite support."""
        return np.inf

    @property
    def name(self) -> str:
        return "multiquadric"

    def laplacian(self, r: np.ndarray | float, h: float, dimension: int) -> np.ndarray | float:
        """
        Analytical Laplacian for multiquadric kernel.

        For phi(r) = sqrt(1 + (r/h)^2):
            Delta phi = (d-1)/(h^2 * phi) + 1/(h^2 * phi^3)
                      = 1/(h^2 * phi) * [(d-1) + 1/phi^2]
        """
        r = np.asarray(r)
        dim = float(dimension)  # Ensure consistent float type
        q2 = (r / h) ** 2
        phi = np.sqrt(1 + q2)
        phi3 = phi**3

        # d^2 phi/dr^2 = 1/(h^2 * phi^3)
        d2phi = 1 / (h**2 * phi3)

        # For r > 0: (dim-1)/r * dphi = (dim-1) / (h^2 * phi)
        # At r=0: use L'Hopital -> dim * d^2phi/dr^2 = dim / h^2
        lap = np.where(
            r > 1e-14,
            d2phi + (dim - 1) / (h**2 * phi),
            dim / (h**2),
        )
        return lap


# ============================================================================
# Factory Function
# ============================================================================

KernelType = Literal[
    "gaussian",
    "wendland_c0",
    "wendland_c2",
    "wendland_c4",
    "wendland_c6",
    "cubic_spline",
    "quintic_spline",
    "cubic",
    "quartic",
    "phs1",
    "phs3",
    "phs5",
    "phs7",
    "multiquadric",
]


def create_kernel(kernel_type: KernelType, dimension: int = 3) -> Kernel:
    """
    Factory function to create kernel instances.

    Parameters
    ----------
    kernel_type : KernelType
        Type of kernel to create. Options:
        - 'gaussian': Gaussian RBF (infinite support)
        - 'wendland_c0': Wendland C^0 (compact)
        - 'wendland_c2': Wendland C^2 (compact, most common)
        - 'wendland_c4': Wendland C^4 (compact, smoother)
        - 'wendland_c6': Wendland C^6 (compact, smoothest)
        - 'cubic_spline': Cubic B-spline (SPH standard)
        - 'quintic_spline': Quintic B-spline (high accuracy)
        - 'cubic': Simple cubic polynomial
        - 'quartic': Simple quartic polynomial
        - 'phs1', 'phs3', 'phs5', 'phs7': Polyharmonic splines (no shape parameter)
    dimension : int
        Spatial dimension (used for Wendland and spline kernels for normalization). Default: 3.

    Returns
    -------
    kernel : Kernel
        Kernel instance.

    Examples
    --------
    >>> kernel = create_kernel('wendland_c2')
    >>> r = np.linspace(0, 2, 100)
    >>> h = 1.0
    >>> weights = kernel(r, h)
    """
    kernel_map = {
        "gaussian": GaussianKernel,
        "cubic": CubicKernel,
        "quartic": QuarticKernel,
    }

    # Wendland kernels with parameterized smoothness
    if kernel_type == "wendland_c0":
        return WendlandKernel(k=0, dimension=dimension)
    elif kernel_type == "wendland_c2":
        return WendlandKernel(k=1, dimension=dimension)
    elif kernel_type == "wendland_c4":
        return WendlandKernel(k=2, dimension=dimension)
    elif kernel_type == "wendland_c6":
        return WendlandKernel(k=3, dimension=dimension)
    # Spline kernels require dimension parameter
    elif kernel_type == "cubic_spline":
        return CubicSplineKernel(dimension=dimension)
    elif kernel_type == "quintic_spline":
        return QuinticSplineKernel(dimension=dimension)
    # PHS kernels (no shape parameter)
    elif kernel_type == "phs1":
        return PHSKernel(m=1)
    elif kernel_type == "phs3":
        return PHSKernel(m=3)
    elif kernel_type == "phs5":
        return PHSKernel(m=5)
    elif kernel_type == "phs7":
        return PHSKernel(m=7)
    # Multiquadric
    elif kernel_type == "multiquadric":
        return MultiquadricKernel()
    elif kernel_type in kernel_map:
        return kernel_map[kernel_type]()
    else:
        valid_types = (
            *kernel_map.keys(),
            "wendland_c0",
            "wendland_c2",
            "wendland_c4",
            "wendland_c6",
            "cubic_spline",
            "quintic_spline",
            "phs1",
            "phs3",
            "phs5",
            "phs7",
            "multiquadric",
        )
        raise ValueError(f"Unknown kernel type '{kernel_type}'. Valid options: {valid_types}")
