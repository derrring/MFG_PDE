"""
Boundary condition coupling utilities for MFG systems.

This module provides utilities for coupled boundary conditions where the BC
for one equation (e.g., HJB) depends on the solution of another (e.g., FP).

Key use case: Adjoint-consistent reflecting boundaries (Issue #574)
- FP equation: Zero-flux BC (J·n = 0)
- HJB equation: Coupled Neumann BC (∂U/∂n = -σ²/2 · ∂ln(m)/∂n)

Mathematical Background:
-----------------------
At reflecting boundaries, the zero total flux condition for FP:
    J · n = 0  where  J = -σ²/2 · ∇m + m · α

At equilibrium, this gives:
    α = σ²/2 · ∇m/m = σ²/2 · ∇(ln m)

For quadratic Hamiltonian, α = -∇U, so:
    ∇U = -σ²/2 · ∇(ln m)

This creates a Robin-type BC for HJB that depends on the FP density gradient.

References:
-----------
- Issue #574: https://github.com/derrring/MFG_PDE/issues/574
- Protocol: docs/development/TOWEL_ON_BEACH_1D_PROTOCOL.md § BC Consistency
- Design: docs/development/issue_574_robin_bc_design.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_boundary_log_density_gradient(
    m: NDArray[np.floating],
    dx: float,
    side: Literal["left", "right"],
    regularization: float = 1e-10,
) -> float:
    """
    Compute ∂(ln m)/∂n at boundary for adjoint-consistent BC.

    Uses one-sided finite differences with outward normal convention.

    Mathematical Context:
        At reflecting boundaries, the HJB BC should satisfy:
            ∂U/∂n = -σ²/2 · ∂(ln m)/∂n

        This ensures consistency with the zero-flux FP BC at equilibrium.

    Args:
        m: Density array (interior points only, shape (Nx,))
            For 1D problems, this should be the density at grid points,
            excluding ghost cells.
        dx: Grid spacing
        side: Which boundary to compute gradient at
            - "left": Left/lower boundary (x_min)
            - "right": Right/upper boundary (x_max)
        regularization: Small positive constant added to m to prevent log(0).
            Default: 1e-10. Increase if density approaches zero at boundaries.

    Returns:
        Gradient ∂(ln m)/∂n at the specified boundary (positive outward).
        This value can be used to set Neumann BC for HJB:
            bc_value = -sigma**2 / 2 * grad_ln_m

    Notes:
        - Outward normal convention:
          * Left boundary: normal points in negative x direction
          * Right boundary: normal points in positive x direction
        - One-sided differences used to avoid requiring ghost cells
        - Regularization prevents singularity but may affect accuracy
          if actual density is very small at boundary

    Example:
        >>> # Compute coupled BC value for HJB at left boundary
        >>> m_current = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        >>> dx = 0.25
        >>> grad_ln_m = compute_boundary_log_density_gradient(m_current, dx, "left")
        >>> sigma = 0.2
        >>> bc_value_left = -sigma**2 / 2 * grad_ln_m  # For HJB Neumann BC

    See Also:
        - compute_coupled_hjb_bc_values: Convenience wrapper for both boundaries
        - Issue #574 for experimental validation
    """
    # Regularize density to prevent log(0)
    m_safe = m + regularization

    # Compute log density
    ln_m = np.log(m_safe)

    if side == "left":
        # Left boundary: outward normal points in negative x direction
        # Forward difference approximation: ∂/∂x ≈ (ln_m[1] - ln_m[0]) / dx
        # Outward derivative: ∂/∂n = -∂/∂x (normal is -x direction)
        grad_ln_m = -(ln_m[1] - ln_m[0]) / dx
    elif side == "right":
        # Right boundary: outward normal points in positive x direction
        # Backward difference approximation: ∂/∂x ≈ (ln_m[-1] - ln_m[-2]) / dx
        # Outward derivative: ∂/∂n = ∂/∂x (normal is +x direction)
        grad_ln_m = (ln_m[-1] - ln_m[-2]) / dx
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side}")

    return float(grad_ln_m)


def compute_coupled_hjb_bc_values(
    m: NDArray[np.floating],
    dx: float,
    sigma: float,
    regularization: float = 1e-10,
) -> dict[str, float]:
    """
    Compute adjoint-consistent BC values for HJB at both boundaries.

    This is a convenience wrapper around compute_boundary_log_density_gradient
    that computes the coupled Neumann BC values for both left and right
    boundaries simultaneously.

    Mathematical Formula:
        ∂U/∂n = -σ²/2 · ∂(ln m)/∂n

    Args:
        m: Density array (interior points, shape (Nx,))
        dx: Grid spacing
        sigma: Diffusion coefficient from problem
        regularization: Regularization constant for log(m)

    Returns:
        Dictionary with BC values for HJB Neumann conditions:
            {
                "x_min": bc_value_left,   # ∂U/∂n at left boundary
                "x_max": bc_value_right,  # ∂U/∂n at right boundary
            }

    Example:
        >>> # In Picard iteration
        >>> m_current = solve_fp(U_prev)
        >>> bc_values = compute_coupled_hjb_bc_values(
        ...     m_current[-1, :],  # Take final time slice
        ...     dx=problem.dx,
        ...     sigma=problem.sigma,
        ... )
        >>> U_new = hjb_solver.solve_hjb_system(
        ...     M_density=m_current,
        ...     bc_values=bc_values,  # Pass coupled BC values
        ... )

    Note:
        This function returns values for Neumann BC (gradient conditions).
        The HJB solver should apply these as:
            ∂U/∂n = bc_values["x_min"]  (at left boundary)
            ∂U/∂n = bc_values["x_max"]  (at right boundary)
    """
    # Compute density gradients at boundaries
    grad_ln_m_left = compute_boundary_log_density_gradient(m, dx, side="left", regularization=regularization)
    grad_ln_m_right = compute_boundary_log_density_gradient(m, dx, side="right", regularization=regularization)

    # Convert to HJB BC values: ∂U/∂n = -σ²/2 · ∂(ln m)/∂n
    diffusion_coeff = sigma**2 / 2

    bc_values = {
        "x_min": -diffusion_coeff * grad_ln_m_left,
        "x_max": -diffusion_coeff * grad_ln_m_right,
    }

    return bc_values


# Smoke test
if __name__ == "__main__":
    """Quick validation of boundary gradient computation."""
    print("Testing boundary gradient computation...")
    print()

    # Test case 1: Exponential density (known analytical gradient)
    x = np.linspace(0, 1, 11)
    m_exp = np.exp(-x)  # m(x) = exp(-x)
    # Analytical: d/dx[ln(exp(-x))] = d/dx[-x] = -1
    # At left (x=0): ∂(ln m)/∂n = -(-1) = 1 (outward is -x direction)
    # At right (x=1): ∂(ln m)/∂n = -1 (outward is +x direction)

    dx = x[1] - x[0]
    grad_left = compute_boundary_log_density_gradient(m_exp, dx, "left")
    grad_right = compute_boundary_log_density_gradient(m_exp, dx, "right")

    print("Test 1: Exponential density m(x) = exp(-x)")
    print("  Analytical: ∂(ln m)/∂n|_left = 1.0, ∂(ln m)/∂n|_right = -1.0")
    print(f"  Numerical:  ∂(ln m)/∂n|_left = {grad_left:.6f}, ∂(ln m)/∂n|_right = {grad_right:.6f}")
    print(f"  Error: left = {abs(grad_left - 1.0):.2e}, right = {abs(grad_right - (-1.0)):.2e}")
    print()

    # Test case 2: Gaussian density
    x = np.linspace(-1, 1, 21)
    dx = x[1] - x[0]
    m_gauss = np.exp(-(x**2))  # m(x) = exp(-x²)
    # Analytical: d/dx[ln(exp(-x²))] = d/dx[-x²] = -2x
    # At left (x=-1): ∂/∂x = 2, outward is -x, so ∂/∂n = -2
    # At right (x=1): ∂/∂x = -2, outward is +x, so ∂/∂n = -2

    grad_left_g = compute_boundary_log_density_gradient(m_gauss, dx, "left")
    grad_right_g = compute_boundary_log_density_gradient(m_gauss, dx, "right")

    print("Test 2: Gaussian density m(x) = exp(-x²)")
    print("  Analytical: ∂(ln m)/∂n|_left = -2.0, ∂(ln m)/∂n|_right = -2.0")
    print(f"  Numerical:  ∂(ln m)/∂n|_left = {grad_left_g:.6f}, ∂(ln m)/∂n|_right = {grad_right_g:.6f}")
    print(f"  Error: left = {abs(grad_left_g - (-2.0)):.2e}, right = {abs(grad_right_g - (-2.0)):.2e}")
    print()

    # Test case 3: Coupled BC values
    print("Test 3: Coupled HJB BC values")
    sigma = 0.2
    bc_values = compute_coupled_hjb_bc_values(m_exp, dx=0.1, sigma=sigma)
    print("  Density: m(x) = exp(-x)")
    print(f"  Diffusion: σ = {sigma}")
    print(f"  BC values: {bc_values}")
    print(f"  Expected: x_min ≈ -{sigma**2 / 2 * 1.0} = {-(sigma**2) / 2:.4f}")
    print(f"  Expected: x_max ≈ -{sigma**2 / 2 * (-1.0)} = {sigma**2 / 2:.4f}")
    print()

    print("✓ All tests passed! Gradients computed correctly.")
