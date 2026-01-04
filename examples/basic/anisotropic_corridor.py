"""
Anisotropic Diffusion in Corridor Navigation.

Demonstrates tensor diffusion Σ(x) in MFG with spatially-varying anisotropy.

Physical Setup
--------------
Agents navigate a 2D corridor with:
- Preferred direction: Along x-axis (easier to move horizontally than vertically)
- Terminal cost: Reach target at (x=1, y=0.5)
- Running cost: Effort to move against preferred direction

Diffusion Tensor
----------------
Σ(x, y) = R(y) D R(y)ᵀ

where:
- D = diag([σ_parallel², σ_perp²]) with σ_parallel > σ_perp
- R(y) = rotation matrix aligning with corridor direction
- Near walls: Reduce diffusion perpendicular to wall

This example focuses on demonstrating the tensor diffusion operators.
Full MFG coupling will be added in future work.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.geometry.boundary.conditions import no_flux_bc
from mfg_pde.utils.numerical.tensor_calculus import diffusion


def create_corridor_diffusion_tensor(
    X: np.ndarray,
    Y: np.ndarray,
    sigma_parallel: float = 0.2,
    sigma_perp: float = 0.05,
    rotation_angle: float = 0.0,
) -> np.ndarray:
    """
    Create spatially-varying anisotropic diffusion tensor for corridor.

    Parameters
    ----------
    X, Y : ndarray
        Meshgrid coordinates
    sigma_parallel : float
        Diffusion coefficient along preferred direction
    sigma_perp : float
        Diffusion coefficient perpendicular to preferred direction
    rotation_angle : float
        Rotation angle for preferred direction (radians)

    Returns
    -------
    ndarray
        Diffusion tensor Σ(x,y) with shape (Nx, Ny, 2, 2)
    """
    Nx, Ny = X.shape

    # Diagonal tensor in rotated frame
    D = np.diag([sigma_parallel, sigma_perp])

    # Rotation matrix (constant for now)
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    R = np.array([[c, -s], [s, c]])

    # Transform to original frame: Σ = R D Rᵀ
    Sigma_base = R @ D @ R.T

    # Spatially varying: reduce perpendicular diffusion near walls
    Sigma = np.zeros((Nx, Ny, 2, 2))
    for i in range(Nx):
        for j in range(Ny):
            # Modulation factor (reduce diffusion near walls y=0, y=1)
            y_dist_from_center = abs(Y[i, j] - 0.5)
            wall_factor = 1.0 - 0.5 * (y_dist_from_center / 0.5) ** 2

            Sigma[i, j] = wall_factor * Sigma_base

    return Sigma


def visualize_diffusion_tensor(
    X: np.ndarray,
    Y: np.ndarray,
    Sigma: np.ndarray,
    title: str = "Anisotropic Diffusion Tensor",
) -> None:
    """
    Visualize diffusion tensor field.

    Shows eigenvalues and eigenvectors at grid points.
    """
    Nx, Ny = X.shape

    _fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: σ₁₁ component
    im1 = axes[0].contourf(X, Y, Sigma[:, :, 0, 0], levels=20, cmap="viridis")
    axes[0].set_title("σ₁₁ (x-x diffusion)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im1, ax=axes[0])

    # Plot 2: σ₁₂ component (cross-diffusion)
    im2 = axes[1].contourf(X, Y, Sigma[:, :, 0, 1], levels=20, cmap="RdBu_r")
    axes[1].set_title("σ₁₂ (cross-diffusion)")
    axes[1].set_xlabel("x")
    plt.colorbar(im2, ax=axes[1])

    # Plot 3: Eigenvalues and eigenvectors
    axes[2].set_title("Principal Directions")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    # Sample points for quiver plot (every 4th point)
    skip = 4

    for i in range(0, Nx, skip):
        for j in range(0, Ny, skip):
            eigenvalues, eigenvectors = np.linalg.eigh(Sigma[i, j])

            # Draw eigenvectors scaled by eigenvalues
            for k in range(2):
                v = eigenvectors[:, k]
                lambda_k = eigenvalues[k]
                scale = 0.8 * skip * lambda_k

                axes[2].arrow(
                    X[i, j],
                    Y[i, j],
                    scale * v[0],
                    scale * v[1],
                    head_width=0.02,
                    head_length=0.02,
                    fc="blue" if k == 1 else "red",
                    ec="blue" if k == 1 else "red",
                    alpha=0.6,
                    length_includes_head=True,
                )

    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_aspect("equal")

    plt.suptitle(title)
    plt.tight_layout()


def test_tensor_diffusion_evolution():
    """
    Test tensor diffusion evolution (without full MFG coupling).

    Evolves density forward using ∂m/∂t = ∇·(Σ ∇m).
    """
    print("Testing Anisotropic Diffusion in Corridor")
    print("=" * 60)

    # Grid setup
    Nx, Ny = 50, 30
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Initial condition: Gaussian blob at (0.2, 0.5)
    x0, y0 = 0.2, 0.5
    sigma_init = 0.05
    m0 = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma_init**2))
    m0 /= np.sum(m0) * dx * dy  # Normalize

    # Create anisotropic diffusion tensor (prefer horizontal movement)
    Sigma = create_corridor_diffusion_tensor(X, Y, sigma_parallel=0.15, sigma_perp=0.03, rotation_angle=0.0)

    # Boundary conditions
    bc = no_flux_bc(dimension=2)

    # Time evolution parameters
    dt = 0.001
    T_final = 0.1
    Nt = int(T_final / dt)

    print(f"\nGrid: {Nx}×{Ny}")
    print(f"Time steps: {Nt} (dt={dt})")
    print(f"Diffusion: σ_parallel={0.15}, σ_perp={0.03}")

    # Visualize diffusion tensor
    visualize_diffusion_tensor(X, Y, Sigma)

    # Evolve density
    m = m0.copy()
    times_to_plot = [0, int(Nt * 0.25), int(Nt * 0.5), Nt - 1]
    snapshots = []
    time_values = []

    for k in range(Nt):
        if k in times_to_plot:
            snapshots.append(m.copy())
            time_values.append(k * dt)

        # Compute diffusion term: ∇·(Σ ∇m)
        diffusion_term = diffusion(m, Sigma, [dx, dy], bc=bc)

        # Forward Euler timestep
        m = m + dt * diffusion_term

        # Enforce non-negativity and mass conservation
        m = np.maximum(m, 0)
        m /= np.sum(m) * dx * dy

    # Final snapshot
    if (Nt - 1) not in times_to_plot:
        snapshots.append(m.copy())
        time_values.append((Nt - 1) * dt)

    # Visualize evolution
    _fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, (snapshot, t) in enumerate(zip(snapshots[:4], time_values[:4], strict=True)):
        im = axes[idx].contourf(X, Y, snapshot, levels=20, cmap="hot")
        axes[idx].set_title(f"Density at t = {t:.3f}")
        axes[idx].set_xlabel("x")
        axes[idx].set_ylabel("y")
        axes[idx].set_aspect("equal")
        plt.colorbar(im, ax=axes[idx])

        # Add ellipse showing principal axes at blob center
        cy, cx = np.unravel_index(np.argmax(snapshot), snapshot.shape)
        Sigma_center = Sigma[cx, cy]
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma_center)

        for k in range(2):
            v = eigenvectors[:, k]
            lambda_k = eigenvalues[k]
            scale = 15 * lambda_k

            axes[idx].arrow(
                X[cx, cy],
                Y[cx, cy],
                scale * v[0],
                scale * v[1],
                head_width=0.02,
                color="cyan",
                linewidth=2,
                alpha=0.8,
            )

    plt.suptitle("Anisotropic Diffusion Evolution (Preferred Horizontal Direction)")
    plt.tight_layout()

    print("\n✓ Tensor diffusion test complete")
    print(f"  Final mass: {np.sum(m) * dx * dy:.6f} (should be ≈ 1.0)")

    # Compare with isotropic diffusion
    print("\nComparing with isotropic diffusion...")
    m_iso = m0.copy()
    sigma_iso = np.sqrt(0.15 * 0.03) * np.eye(2)  # Geometric mean

    for k in range(Nt):
        diffusion_term_iso = diffusion(m_iso, sigma_iso, [dx, dy], bc=bc)
        m_iso = m_iso + dt * diffusion_term_iso
        m_iso = np.maximum(m_iso, 0)
        m_iso /= np.sum(m_iso) * dx * dy

    # Plot comparison
    _fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].contourf(X, Y, m, levels=20, cmap="hot")
    axes[0].set_title(f"Anisotropic (t={T_final})")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(X, Y, m_iso, levels=20, cmap="hot")
    axes[1].set_title(f"Isotropic (t={T_final})")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].contourf(X, Y, m - m_iso, levels=20, cmap="RdBu_r")
    axes[2].set_title("Difference (Anisotropic - Isotropic)")
    axes[2].set_aspect("equal")
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle("Anisotropic vs Isotropic Diffusion")
    plt.tight_layout()

    print("✓ Comparison complete")
    print("\nNote: Anisotropic diffusion spreads more in horizontal direction")
    print("      as expected from larger σ_parallel vs σ_perp")

    plt.show()


if __name__ == "__main__":
    test_tensor_diffusion_evolution()
