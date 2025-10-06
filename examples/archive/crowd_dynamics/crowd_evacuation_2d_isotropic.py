"""
2D Isotropic Crowd Evacuation using Hybrid FP-Particle + HJB-FDM Solver

Demonstrates the HybridFPParticleHJBFDM solver on a 2D crowd evacuation problem
using grid flattening to convert 2D problem to 1D solver format.

Physical Setup:
- Room: 10m × 10m square domain
- Initial crowd: Gaussian distribution centered at (5.0, 7.0)
- Exit: Located at bottom boundary (y=0)
- Goal: Minimize evacuation time while avoiding congestion

Mathematical Model:
- HJB (value function): -∂u/∂t + H(∇u, m) = 0
  where H = (1/2)|∇u|² + λm (congestion cost)

- FP (density evolution): ∂m/∂t - div(m∇u) - σ²Δm = 0
  solved using particle method for natural mass conservation

Grid Flattening:
- 2D grid (nx1+1) × (nx2+1) → 1D array of size Nx = (nx1+1) * (nx2+1)
- Index mapping: idx_1d = i1 * (nx2+1) + i2
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mfg_pde.alg.numerical.mfg_solvers.hybrid_fp_particle_hjb_fdm import HybridFPParticleHJBFDM
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import BoundaryConditions


class Grid2DAdapter:
    """
    Adapter to convert 2D grid indices to 1D for solver compatibility.

    The hybrid solver expects 1D arrays, but we have a 2D problem.
    This adapter handles the index mapping and field conversion.
    """

    def __init__(self, nx1: int, nx2: int, bounds: tuple[float, float, float, float]):
        """
        Initialize 2D grid adapter.

        Args:
            nx1: Number of grid cells in x1 direction
            nx2: Number of grid cells in x2 direction
            bounds: (x1_min, x1_max, x2_min, x2_max)
        """
        self.nx1, self.nx2 = nx1, nx2
        self.x1_min, self.x1_max, self.x2_min, self.x2_max = bounds

        # Grid spacing
        self.dx1 = (self.x1_max - self.x1_min) / nx1
        self.dx2 = (self.x2_max - self.x2_min) / nx2

        # Total 1D size
        self.n_total = (nx1 + 1) * (nx2 + 1)

    def idx_2d_to_1d(self, i1: int, i2: int) -> int:
        """Convert 2D grid indices to 1D array index."""
        return i1 * (self.nx2 + 1) + i2

    def idx_1d_to_2d(self, idx_1d: int) -> tuple[int, int]:
        """Convert 1D array index to 2D grid indices."""
        i1 = idx_1d // (self.nx2 + 1)
        i2 = idx_1d % (self.nx2 + 1)
        return i1, i2

    def field_1d_to_2d(self, field_1d: np.ndarray) -> np.ndarray:
        """
        Convert 1D field array to 2D grid for visualization.

        Args:
            field_1d: 1D array of shape (Nt, Nx) or (Nx,)

        Returns:
            2D array of shape (Nt, nx1+1, nx2+1) or (nx1+1, nx2+1)
        """
        if field_1d.ndim == 1:
            # Single time step
            field_2d = np.zeros((self.nx1 + 1, self.nx2 + 1))
            for i1 in range(self.nx1 + 1):
                for i2 in range(self.nx2 + 1):
                    idx = self.idx_2d_to_1d(i1, i2)
                    if idx < len(field_1d):
                        field_2d[i1, i2] = field_1d[idx]
            return field_2d
        else:
            # Multiple time steps (Nt, Nx)
            Nt = field_1d.shape[0]
            field_2d = np.zeros((Nt, self.nx1 + 1, self.nx2 + 1))
            for t in range(Nt):
                for i1 in range(self.nx1 + 1):
                    for i2 in range(self.nx2 + 1):
                        idx = self.idx_2d_to_1d(i1, i2)
                        if idx < field_1d.shape[1]:
                            field_2d[t, i1, i2] = field_1d[t, idx]
            return field_2d

    def field_2d_to_1d(self, field_2d: np.ndarray) -> np.ndarray:
        """
        Convert 2D grid array to 1D field for solver.

        Args:
            field_2d: 2D array of shape (nx1+1, nx2+1)

        Returns:
            1D array of shape (Nx,)
        """
        field_1d = np.zeros(self.n_total)
        for i1 in range(self.nx1 + 1):
            for i2 in range(self.nx2 + 1):
                idx = self.idx_2d_to_1d(i1, i2)
                field_1d[idx] = field_2d[i1, i2]
        return field_1d


def create_2d_crowd_evacuation_problem(adapter: Grid2DAdapter) -> MFGProblem:
    """
    Create 2D crowd evacuation MFG problem.

    Setup:
    - Domain: [0, 10] × [0, 10] meters
    - Exit at y=0 (bottom boundary)
    - Crowd initially concentrated at (5.0, 7.0)
    - Isotropic movement (no directional preferences)

    Args:
        adapter: Grid2DAdapter instance for index mapping

    Returns:
        MFGProblem instance with flattened 2D grid
    """
    # Time parameters
    T = 5.0  # Evacuation time horizon (seconds)
    Nt = 30  # Temporal resolution (reduced for speed)

    # Physical parameters
    sigma = 0.3  # Diffusion coefficient (m²/s)
    lambda_crowd = 2.0  # Congestion cost coefficient

    # Create grids
    x1 = np.linspace(adapter.x1_min, adapter.x1_max, adapter.nx1 + 1)
    x2 = np.linspace(adapter.x2_min, adapter.x2_max, adapter.nx2 + 1)
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")

    # Initial crowd distribution: Gaussian centered at (5.0, 7.0)
    x_center, y_center = 5.0, 7.0
    sigma_x, sigma_y = 1.5, 1.5

    initial_density_2d = np.exp(-((X1 - x_center) ** 2 / (2 * sigma_x**2) + (X2 - y_center) ** 2 / (2 * sigma_y**2)))

    # Normalize
    total_mass = np.sum(initial_density_2d) * adapter.dx1 * adapter.dx2
    initial_density_2d = initial_density_2d / total_mass

    # Terminal cost: distance to exit (y=0)
    terminal_cost_2d = X2.copy()

    # Convert to 1D
    initial_density_1d = adapter.field_2d_to_1d(initial_density_2d)
    terminal_cost_1d = adapter.field_2d_to_1d(terminal_cost_2d)

    # Create 1D problem with flattened 2D grid
    problem = MFGProblem(
        xmin=0.0,
        xmax=float(adapter.n_total - 1),  # Flattened grid indices
        Nx=adapter.n_total - 1,
        T=T,
        Nt=Nt,
        sigma=sigma,
        coefCT=lambda_crowd,
    )

    # Set boundary conditions (no-flux)
    bc = BoundaryConditions(type="neumann", left_value=0.0, right_value=0.0)
    problem.boundary_conditions = bc

    # Set initial condition and terminal cost
    problem.rho0 = initial_density_1d
    problem.g = terminal_cost_1d

    return problem


def solve_2d_evacuation(problem: MFGProblem, verbose: bool = True):
    """
    Solve 2D crowd evacuation using Hybrid FP-Particle + HJB-FDM solver.

    Args:
        problem: MFG problem instance (flattened 2D)
        verbose: Print progress information

    Returns:
        Tuple of (U, M, info)
    """
    if verbose:
        print("=" * 70)
        print("2D Isotropic Crowd Evacuation - Hybrid Solver")
        print("=" * 70)
        print(f"Grid size: {problem.Nx+1} points (flattened 2D)")
        print(f"Time: [0, {problem.T}]s with {problem.Nt+1} steps")
        print(f"Diffusion: σ = {problem.sigma}")
        print(f"Congestion cost: λ = {problem.coefCT}")
        print()

    # Create hybrid solver
    solver = HybridFPParticleHJBFDM(
        problem=problem,
        num_particles=5000,  # Particles for smooth 2D density
        kde_bandwidth="scott",
        max_newton_iterations=20,
        newton_tolerance=1e-6,
        damping_parameter=0.6,  # Higher damping for 2D stability
    )

    if verbose:
        print("Solver Configuration:")
        print(f"  FP Method: Particle (N={solver.num_particles})")
        print("  HJB Method: Finite Difference (Newton)")
        print(f"  Coupling: Picard iteration (damping={solver.damping_parameter})")
        print()

    # Solve
    U, M, info = solver.solve(max_iterations=30, tolerance=1e-2, verbose=verbose)

    if verbose:
        print()
        print("=" * 70)
        print("SOLUTION SUMMARY")
        print("=" * 70)
        print(f"Converged: {info.get('converged', False)}")
        print(f"Iterations: {info.get('iterations', 'N/A')}")
        print(f"Final error: {info.get('final_error', 'N/A'):.6e}")
        print(f"Execution time: {info.get('execution_time', 'N/A'):.2f}s")
        print("=" * 70)

    return U, M, info


def visualize_2d_results(
    adapter: Grid2DAdapter,
    problem: MFGProblem,
    U: np.ndarray,
    M: np.ndarray,
    info: dict,
    save_path: str | None = None,
):
    """
    Create comprehensive visualization of 2D crowd evacuation results.

    Args:
        adapter: Grid2DAdapter for converting 1D → 2D
        problem: MFG problem instance
        U: Value function solution (1D)
        M: Density solution (1D)
        info: Solver information
        save_path: Optional path to save figure
    """
    # Convert 1D solutions to 2D
    M_2d = adapter.field_1d_to_2d(M)
    U_2d = adapter.field_1d_to_2d(U)

    # Create coordinate grids
    x1 = np.linspace(adapter.x1_min, adapter.x1_max, adapter.nx1 + 1)
    x2 = np.linspace(adapter.x2_min, adapter.x2_max, adapter.nx2 + 1)
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")

    # Time snapshots
    Nt = problem.Nt + 1
    time_indices = [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt - 1]
    times = [problem.T * t / (Nt - 1) for t in time_indices]

    # Create figure
    plt.figure(figsize=(18, 12))

    # Plot 1-5: Density evolution
    for idx, (t_idx, t) in enumerate(zip(time_indices, times, strict=False), start=1):
        ax = plt.subplot(3, 5, idx)
        im = ax.contourf(X1, X2, M_2d[t_idx], levels=20, cmap="YlOrRd")
        ax.set_title(f"Density at t={t:.1f}s", fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        ax.axhline(y=0, color="green", linestyle="--", linewidth=2, label="Exit" if idx == 1 else "")
        if idx == 1:
            ax.legend()
        plt.colorbar(im, ax=ax, label="Density")

    # Plot 6-10: Value function evolution
    for idx, (t_idx, t) in enumerate(zip(time_indices, times, strict=False), start=6):
        ax = plt.subplot(3, 5, idx)
        im = ax.contourf(X1, X2, U_2d[t_idx], levels=20, cmap="viridis")
        ax.set_title(f"Value u at t={t:.1f}s", fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="Cost")

    # Plot 11: Mass conservation
    ax11 = plt.subplot(3, 5, 11)
    time_vec = np.linspace(0, problem.T, Nt)
    dx_vol = adapter.dx1 * adapter.dx2
    masses = np.array([np.sum(M[t]) * dx_vol for t in range(Nt)])
    ax11.plot(time_vec, masses, "b-", linewidth=2, label="Total mass")
    ax11.axhline(y=masses[0], color="r", linestyle="--", label="Initial mass")
    ax11.set_xlabel("Time (s)")
    ax11.set_ylabel("Total Mass")
    ax11.set_title("Mass Conservation")
    ax11.legend()
    ax11.grid(True, alpha=0.3)

    # Plot 12: Convergence history
    ax12 = plt.subplot(3, 5, 12)
    if info.get("convergence_history"):
        history = info["convergence_history"]
        iterations = [h["iteration"] for h in history]
        errors = [h.get("total_error", 0) for h in history]
        ax12.semilogy(iterations, errors, "b-o", markersize=3)
        ax12.axhline(y=1e-2, color="r", linestyle="--", label="Tolerance")
        ax12.set_xlabel("Picard Iteration")
        ax12.set_ylabel("Relative Error")
        ax12.set_title("Convergence History")
        ax12.legend()
        ax12.grid(True, alpha=0.3)

    # Plot 13: 3D surface plot of final density
    ax13 = plt.subplot(3, 5, 13, projection="3d")
    ax13.plot_surface(X1, X2, M_2d[-1], cmap="YlOrRd", alpha=0.8)
    ax13.set_xlabel("x (m)")
    ax13.set_ylabel("y (m)")
    ax13.set_zlabel("Density")
    ax13.set_title("Final Density (3D)")

    # Plot 14: Exit flux
    ax14 = plt.subplot(3, 5, 14)
    # Flux at bottom boundary (y=0)
    exit_flux = [np.sum(M_2d[t, :, 0]) * adapter.dx1 for t in range(Nt)]
    ax14.plot(time_vec, exit_flux, "g-", linewidth=2)
    ax14.set_xlabel("Time (s)")
    ax14.set_ylabel("Exit Flux")
    ax14.set_title("Evacuation Rate")
    ax14.grid(True, alpha=0.3)

    # Plot 15: Statistics
    ax15 = plt.subplot(3, 5, 15)
    ax15.axis("off")
    stats_text = f"""
SIMULATION STATISTICS

Domain:
  Size: {adapter.x1_max-adapter.x1_min}m × {adapter.x2_max-adapter.x2_min}m
  Grid: {adapter.nx1+1} × {adapter.nx2+1}

Time:
  Duration: {problem.T}s
  Steps: {problem.Nt+1}

Solver:
  Method: Hybrid FP-Particle + HJB-FDM
  Particles: {5000}
  Converged: {info.get('converged', False)}
  Iterations: {info.get('iterations', 'N/A')}

Mass Conservation:
  Initial: {masses[0]:.4f}
  Final: {masses[-1]:.4f}
  Loss: {100*(masses[0]-masses[-1])/masses[0]:.2f}%

Performance:
  Solve time: {info.get('execution_time', 'N/A'):.2f}s
    """
    ax15.text(0.1, 0.5, stats_text, fontsize=9, family="monospace", verticalalignment="center")

    plt.suptitle("2D Isotropic Crowd Evacuation - Hybrid Solver", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()


def main():
    """Run 2D isotropic crowd evacuation example."""
    print("Creating 2D crowd evacuation problem...")

    # Create 2D grid adapter
    # Using smaller grid for faster demo
    adapter = Grid2DAdapter(
        nx1=20,  # 21 points in x direction
        nx2=20,  # 21 points in y direction
        bounds=(0.0, 10.0, 0.0, 10.0),  # [0,10] × [0,10]
    )

    print(f"2D Grid: {adapter.nx1+1} × {adapter.nx2+1} = {adapter.n_total} points")

    # Create problem
    problem = create_2d_crowd_evacuation_problem(adapter)

    # Solve
    U, M, info = solve_2d_evacuation(problem, verbose=True)

    # Visualize
    print("\nCreating visualization...")
    save_path = "crowd_evacuation_2d_isotropic_results.png"
    visualize_2d_results(adapter, problem, U, M, info, save_path=save_path)

    print("\n✓ Example completed successfully!")
    print(f"  Results saved to: {save_path}")


if __name__ == "__main__":
    main()
