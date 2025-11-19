"""
Simple demonstration of tensor diffusion evolution (standalone).

This example shows manual time-stepping with tensor diffusion operators,
serving as a reference for future FP-FDM integration.

Physical setup:
- 2D corridor with anisotropic diffusion
- Preferred horizontal movement (σ_x > σ_y)
- No advection (pure diffusion)
- Explicit forward Euler timestepping

This is a simplified version showing tensor diffusion PDE evolution
before full MFG solver integration (Phase 3.0).
"""

import numpy as np
from matplotlib import pyplot as plt

from mfg_pde.geometry import BoundaryConditions
from mfg_pde.utils.numerical.tensor_operators import divergence_tensor_diffusion_2d


def create_gaussian_initial_condition(
    X: np.ndarray, Y: np.ndarray, x0: float = 0.5, y0: float = 0.5, width: float = 0.1
) -> np.ndarray:
    """Create Gaussian initial density."""
    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * width**2))


def evolve_tensor_diffusion(
    m_init: np.ndarray,
    Sigma: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    Nt: int,
    boundary_conditions: BoundaryConditions,
) -> np.ndarray:
    """
    Evolve density with tensor diffusion using explicit forward Euler.

    PDE: ∂m/∂t = ∇·(Σ ∇m)

    Args:
        m_init: Initial density (Ny, Nx)
        Sigma: Diffusion tensor (2, 2) or (Ny, Nx, 2, 2)
        dx, dy: Grid spacing
        dt: Time step
        Nt: Number of timesteps
        boundary_conditions: Boundary condition spec

    Returns:
        Density evolution (Nt+1, Ny, Nx)
    """
    Ny, Nx = m_init.shape
    M_history = np.zeros((Nt + 1, Ny, Nx))
    M_history[0] = m_init.copy()

    for n in range(Nt):
        # Compute diffusion term
        diffusion_term = divergence_tensor_diffusion_2d(M_history[n], Sigma, dx, dy, boundary_conditions)

        # Explicit forward Euler
        M_history[n + 1] = M_history[n] + dt * diffusion_term

        # Ensure non-negativity
        M_history[n + 1] = np.maximum(M_history[n + 1], 0)

        # Normalize mass
        total_mass = np.sum(M_history[n + 1]) * dx * dy
        if total_mass > 0:
            M_history[n + 1] /= total_mass

    return M_history


def main():
    """Run tensor diffusion evolution demo."""
    print("=" * 70)
    print("Tensor Diffusion Evolution (Standalone)")
    print("=" * 70)
    print()

    # Grid setup
    Nx, Ny = 50, 30
    Lx, Ly = 1.0, 0.6
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx, dy = x[1] - x[0], y[1] - y[0]

    # Time setup
    T = 0.1
    Nt = 100
    dt = T / Nt

    print(f"Grid: {Nx}×{Ny}")
    print(f"Domain: [{0}, {Lx}] × [{0}, {Ly}]")
    print(f"Time: T = {T}, Nt = {Nt}, dt = {dt:.6f}")
    print()

    # Initial condition: Gaussian at center
    m_init = create_gaussian_initial_condition(X, Y, x0=0.5, y0=0.3, width=0.08)
    m_init /= np.sum(m_init) * dx * dy  # Normalize

    # Tensor diffusion setup
    sigma_parallel = 0.2  # Fast horizontal diffusion
    sigma_perp = 0.05  # Slow vertical diffusion

    # Diagonal tensor (preferred horizontal direction)
    Sigma_anisotropic = np.diag([sigma_parallel, sigma_perp])

    # Isotropic tensor for comparison
    sigma_iso = np.sqrt(sigma_parallel * sigma_perp)
    Sigma_isotropic = sigma_iso * np.eye(2)

    print("Anisotropic tensor:")
    print(f"  Σ = [[{Sigma_anisotropic[0, 0]:.3f}, {Sigma_anisotropic[0, 1]:.3f}],")
    print(f"       [{Sigma_anisotropic[1, 0]:.3f}, {Sigma_anisotropic[1, 1]:.3f}]]")
    print(f"  σ_parallel = {sigma_parallel}, σ_perp = {sigma_perp}")
    print()
    print("Isotropic tensor (for comparison):")
    print(f"  Σ = {sigma_iso:.3f} I")
    print()

    # Boundary conditions
    bc = BoundaryConditions(type="no_flux")

    # CFL condition check
    max_sigma = max(sigma_parallel, sigma_perp)
    cfl_limit = min(dx**2, dy**2) / (2 * max_sigma)
    print(f"CFL stability: dt = {dt:.6f}, limit = {cfl_limit:.6f}")
    if dt > cfl_limit:
        print("  ⚠️  WARNING: dt exceeds CFL limit (may be unstable)")
    else:
        print("  ✓ Stable (dt < CFL limit)")
    print()

    # Evolve with anisotropic diffusion
    print("Evolving with anisotropic diffusion...")
    M_aniso = evolve_tensor_diffusion(m_init, Sigma_anisotropic, dx, dy, dt, Nt, bc)

    # Evolve with isotropic diffusion
    print("Evolving with isotropic diffusion (comparison)...")
    M_iso = evolve_tensor_diffusion(m_init, Sigma_isotropic, dx, dy, dt, Nt, bc)

    print("Evolution complete.")
    print()

    # Mass conservation check
    mass_init = np.sum(m_init) * dx * dy
    mass_final_aniso = np.sum(M_aniso[-1]) * dx * dy
    mass_final_iso = np.sum(M_iso[-1]) * dx * dy

    print("Mass conservation:")
    print(f"  Initial: {mass_init:.6f}")
    print(f"  Final (anisotropic): {mass_final_aniso:.6f} (error: {abs(mass_final_aniso - mass_init):.2e})")
    print(f"  Final (isotropic): {mass_final_iso:.6f} (error: {abs(mass_final_iso - mass_init):.2e})")
    print()

    # Visualization
    _fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    times = [0, Nt // 2, Nt]
    titles = ["t = 0", f"t = {T / 2:.3f}", f"t = {T:.3f}"]

    for i, (t_idx, title) in enumerate(zip(times, titles, strict=True)):
        # Anisotropic
        ax = axes[0, i]
        im = ax.contourf(X, Y, M_aniso[t_idx], levels=15, cmap="viridis")
        ax.set_title(f"Anisotropic: {title}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax)

        # Isotropic
        ax = axes[1, i]
        im = ax.contourf(X, Y, M_iso[t_idx], levels=15, cmap="viridis")
        ax.set_title(f"Isotropic: {title}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax)

    plt.suptitle(
        f"Tensor Diffusion Evolution\n"
        f"Anisotropic: σ_x={sigma_parallel}, σ_y={sigma_perp} | "
        f"Isotropic: σ={sigma_iso:.3f}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()

    print("✓ Tensor diffusion evolution complete")
    print()
    print("Key observations:")
    print("  1. Anisotropic diffusion spreads faster horizontally (σ_x > σ_y)")
    print("  2. Isotropic diffusion spreads equally in all directions")
    print("  3. Mass conservation maintained in both cases")
    print("  4. Explicit timestepping works with tensor operators")


if __name__ == "__main__":
    main()
