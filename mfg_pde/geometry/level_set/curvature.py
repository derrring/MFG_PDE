"""
Curvature Computation for Level Set Methods.

Mean curvature of the level set interface is computed via:
    κ = ∇·(∇φ / |∇φ|) = ∇·n

where n = ∇φ/|∇φ| is the unit normal field.

For geometric interpretation:
- κ > 0: Convex interface (bulging outward)
- κ < 0: Concave interface (bulging inward)
- κ = 1/R: Sphere of radius R

Applications:
- Curvature-dependent velocity: V = V₀ + ε·κ (surface tension effects)
- Mean curvature flow: ∂φ/∂t = κ|∇φ|
- Stefan problems with surface tension

Mathematical Background:
    For a curve in 2D or surface in 3D defined by φ = 0:
        κ = ∇·(∇φ/|∇φ|)
          = (φₓₓφᵧ² - 2φₓφᵧφₓᵧ + φᵧᵧφₓ²) / (φₓ² + φᵧ²)^{3/2}  (2D)

    But the divergence formula is dimension-agnostic and numerically stable.

References:
- Osher & Fedkiw (2003): Level Set Methods, Chapter 3
- Sethian (1999): Level Set Methods and Fast Marching Methods, Section 2.4

Created: 2026-01-18 (Issue #592 Milestone 3.1.3)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfg_pde.geometry.boundary import no_flux_bc
from mfg_pde.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.geometry.grids.tensor_grid import TensorProductGrid

# Module logger
logger = get_logger(__name__)


def compute_curvature(
    phi: NDArray[np.float64],
    geometry: TensorProductGrid,
    epsilon: float = 1e-10,
) -> NDArray[np.float64]:
    """
    Compute mean curvature: κ = ∇·(∇φ/|∇φ|) = ∇·n.

    Uses geometry's gradient and divergence operators for dimension-agnostic
    computation. Works for 1D, 2D, 3D, and higher dimensions.

    Args:
        phi: Level set function, shape (Nx, Ny, ...) matching geometry
        geometry: Grid providing gradient and divergence operators
        epsilon: Regularization to avoid division by zero (default: 1e-10)
            Added to |∇φ| to prevent singularities at critical points

    Returns:
        Curvature field κ, same shape as phi

    Raises:
        AttributeError: If geometry doesn't support required operators

    Example:
        >>> # 2D circle: φ = ||x - c|| - R
        >>> phi_circle = np.linalg.norm(X - center, axis=0) - radius
        >>> kappa = compute_curvature(phi_circle, grid)
        >>> # On interface (φ ≈ 0): κ ≈ 1/R

    Note:
        Curvature is most accurate near the interface (|φ| ≈ 0) for signed
        distance functions. For best results, reinitialize φ before computing
        curvature to ensure |∇φ| = 1.
    """
    # Get operators from geometry (Issue #595 infrastructure)
    grad_ops = geometry.get_gradient_operator()
    div_op = geometry.get_divergence_operator()

    # Compute gradient components: ∇φ = (φₓ, φᵧ, ...)
    grad_phi = np.array([grad_op(phi) for grad_op in grad_ops])

    # Compute gradient magnitude: |∇φ|
    grad_mag = np.linalg.norm(grad_phi, axis=0) + epsilon

    # Compute unit normal field: n = ∇φ / |∇φ|
    # Shape: (dimension, Nx, Ny, ...)
    normal_field = grad_phi / grad_mag

    # Compute divergence of normal: κ = ∇·n
    curvature = div_op(normal_field)

    return curvature


if __name__ == "__main__":
    """Smoke test for curvature computation."""
    print("Testing Curvature Computation...")

    from mfg_pde.geometry.grids.tensor_grid import TensorProductGrid

    # Test 1: 2D Circle - Analytical curvature κ = 1/R
    print("\n[Test 1: 2D Circle Curvature]")
    print("Problem: Verify κ = 1/R for circle of radius R")

    # Create 2D grid
    Nx, Ny = 100, 100
    grid_2d = TensorProductGrid(
        dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx=[Nx, Ny], boundary_conditions=no_flux_bc(dimension=2)
    )
    X, Y = grid_2d.meshgrid()
    dx = grid_2d.spacing[0]

    print(f"  Grid: {Nx}×{Ny}, dx = {dx:.4f}")

    # Circle: φ = ||x - c|| - R (true SDF)
    center = np.array([0.5, 0.5])
    radius = 0.3
    phi_circle = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius

    print(f"  Circle: center={center}, radius={radius}")
    print(f"  Analytical curvature: κ = 1/R = {1.0 / radius:.4f}")

    # Compute curvature
    kappa = compute_curvature(phi_circle, grid_2d)

    print(f"  Curvature field shape: {kappa.shape}")

    # Check on interface (|φ| < 2dx)
    interface_mask = np.abs(phi_circle) < 2 * dx
    kappa_interface = kappa[interface_mask]

    kappa_analytical = 1.0 / radius
    kappa_mean = np.mean(kappa_interface)
    kappa_std = np.std(kappa_interface)
    error = np.abs(kappa_mean - kappa_analytical)

    print(f"  Curvature on interface: mean = {kappa_mean:.4f}, std = {kappa_std:.4f}")
    print(f"  Error: |κ_computed - κ_analytical| = {error:.4f} ({100 * error / kappa_analytical:.2f}%)")

    # Tolerance: Within 10% for coarse grid
    assert error < 0.1 * kappa_analytical, f"Curvature error too large: {error:.4f}"
    print("  ✓ Circle curvature test passed!")

    # Test 2: 2D Flat interface - κ ≈ 0
    print("\n[Test 2: 2D Flat Interface]")
    print("Problem: Verify κ ≈ 0 for flat interface")

    # Flat interface: φ = x - 0.5 (vertical line)
    phi_flat = X - 0.5

    kappa_flat = compute_curvature(phi_flat, grid_2d)

    interface_flat = np.abs(phi_flat) < 2 * dx
    kappa_flat_interface = kappa_flat[interface_flat]

    kappa_flat_mean = np.mean(np.abs(kappa_flat_interface))
    print(f"  Flat interface curvature: mean(|κ|) = {kappa_flat_mean:.4e}")

    # Should be very close to zero
    assert kappa_flat_mean < 0.1, f"Flat interface has non-zero curvature: {kappa_flat_mean:.4e}"
    print("  ✓ Flat interface test passed!")

    # Test 3: 1D "Curvature" (second derivative)
    print("\n[Test 3: 1D Second Derivative]")
    print("Problem: In 1D, κ = ∇·(∇φ/|∇φ|) reduces to sign-preserving second derivative")

    # Create 1D grid
    Nx_1d = 200
    grid_1d = TensorProductGrid(
        dimension=1, bounds=[(0.0, 1.0)], Nx=[Nx_1d], boundary_conditions=no_flux_bc(dimension=1)
    )
    x = grid_1d.coordinates[0]
    dx_1d = grid_1d.spacing[0]

    print(f"  Grid: {Nx_1d} points, dx = {dx_1d:.4f}")

    # Parabola: φ = (x - 0.5)² - 0.1 (interface at x = 0.5 ± √0.1)
    phi_parabola = (x - 0.5) ** 2 - 0.1

    kappa_1d = compute_curvature(phi_parabola, grid_1d)

    # For φ = (x - c)² - h: ∇φ = 2(x - c), |∇φ| = 2|x - c|
    # κ = d/dx[2(x-c)/|2(x-c)|] = d/dx[sign(x-c)] = 0 (except at x=c)
    # But for the curvature of the 0-level set:
    # The level set φ = 0 has two points in 1D, so "curvature" is not standard

    print(f"  1D curvature range: [{kappa_1d.min():.4f}, {kappa_1d.max():.4f}]")
    print(f"  1D curvature mean(|κ|): {np.mean(np.abs(kappa_1d)):.4f}")

    # Check it doesn't explode to infinity (expect singularities at critical points)
    # In 1D, κ = d/dx[∇φ/|∇φ|] has discontinuities where ∇φ changes sign
    assert np.max(np.abs(kappa_1d)) < 1000, "1D curvature exploded"
    assert np.isfinite(kappa_1d).all(), "1D curvature contains NaN/Inf"
    print("  ✓ 1D curvature computation stable (expected singularities at critical points)!")

    # Test 4: 3D Sphere
    print("\n[Test 4: 3D Sphere Curvature]")
    print("Problem: Verify κ = 2/R for sphere of radius R in 3D")

    # Create 3D grid (coarse for speed)
    Nx_3d = 30
    grid_3d = TensorProductGrid(
        dimension=3,
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        Nx=[Nx_3d, Nx_3d, Nx_3d],
        boundary_conditions=no_flux_bc(dimension=3),
    )
    X_3d, Y_3d, Z_3d = grid_3d.meshgrid()
    dx_3d = grid_3d.spacing[0]

    print(f"  Grid: {Nx_3d}×{Nx_3d}×{Nx_3d}, dx = {dx_3d:.4f}")

    # Sphere: φ = ||x - c|| - R
    center_3d = np.array([0.5, 0.5, 0.5])
    radius_3d = 0.3
    phi_sphere = (
        np.sqrt((X_3d - center_3d[0]) ** 2 + (Y_3d - center_3d[1]) ** 2 + (Z_3d - center_3d[2]) ** 2) - radius_3d
    )

    print(f"  Sphere: center={center_3d}, radius={radius_3d}")
    print(f"  Analytical mean curvature: H = 2/R = {2.0 / radius_3d:.4f}")

    kappa_3d = compute_curvature(phi_sphere, grid_3d)

    # Check on interface
    interface_3d = np.abs(phi_sphere) < 2 * dx_3d
    kappa_3d_interface = kappa_3d[interface_3d]

    kappa_3d_analytical = 2.0 / radius_3d  # Mean curvature H = 2/R in 3D
    kappa_3d_mean = np.mean(kappa_3d_interface)
    error_3d = np.abs(kappa_3d_mean - kappa_3d_analytical)

    print(f"  Curvature on interface: mean = {kappa_3d_mean:.4f}")
    print(f"  Error: |H_computed - H_analytical| = {error_3d:.4f} ({100 * error_3d / kappa_3d_analytical:.2f}%)")

    # Coarse grid → larger tolerance (20%)
    assert error_3d < 0.2 * kappa_3d_analytical, f"3D curvature error too large: {error_3d:.4f}"
    print("  ✓ 3D sphere curvature test passed!")

    print("\n✅ All Curvature Computation tests passed!")
