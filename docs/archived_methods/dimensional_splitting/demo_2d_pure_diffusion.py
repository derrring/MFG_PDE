"""
Demonstrate 2D Pure Diffusion with Dimensional Splitting

Test case to validate that dimensional splitting works correctly for
diffusion-dominated problems (no advection). This isolates the dimensional
splitting method from the advection coupling issues seen in full MFG.

Problem:
- Initial Gaussian blob at center
- Pure diffusion (σ² Δm), no advection
- Should show smooth spreading with ~1% mass conservation error
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import MFGComponents
from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem
from mfg_pde.geometry import BoundaryConditions


class PureDiffusion2D(GridBasedMFGProblem):
    """
    2D pure diffusion problem.

    No advection, no potential, just diffusion of a Gaussian blob.
    """

    def __init__(
        self,
        grid_resolution=12,
        time_horizon=0.5,
        num_timesteps=15,
        diffusion=0.05,
        center=(0.5, 0.5),
    ):
        super().__init__(
            domain_bounds=(0.0, 1.0, 0.0, 1.0),
            grid_resolution=grid_resolution,
            time_domain=(time_horizon, num_timesteps),
            diffusion_coeff=diffusion,
        )

        self.center = np.array(center)

    def initial_density(self, x):
        """Gaussian blob centered at domain center."""
        dist_sq = np.sum((x - self.center) ** 2, axis=1)
        density = np.exp(-100 * dist_sq)
        # Normalize by integral: ∫m dx = 1, not just ∑m = 1
        dV = float(np.prod(self.geometry.grid.spacing))
        return density / (np.sum(density) * dV + 1e-10)

    def terminal_cost(self, x):
        """Zero terminal cost."""
        return np.zeros(x.shape[0])

    def running_cost(self, x, t):
        """Zero running cost."""
        return np.zeros(x.shape[0])

    def hamiltonian(self, x, m, p, t):
        """Zero Hamiltonian (no advection)."""
        return np.zeros(x.shape[0])

    def setup_components(self):
        """Setup MFG components (all zeros since no dynamics)."""

        def hamiltonian_func(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, derivs=None, **kwargs):
            """H = 0 (no dynamics)."""
            return 0.0

        def hamiltonian_dm(x_idx, x_position, m_at_x, **kwargs):
            """dH/dm = 0 (no coupling)."""
            return 0.0

        def initial_density_func(x_idx):
            """Gaussian initial density."""
            coords = []
            for d in range(2):
                idx_d = x_idx[d] if isinstance(x_idx, tuple) else x_idx
                x_d = self.geometry.grid.bounds[d][0] + idx_d * self.geometry.grid.spacing[d]
                coords.append(x_d)
            coords_array = np.array(coords).reshape(1, -1)
            return float(self.initial_density(coords_array)[0])

        def terminal_cost_func(x_idx):
            """Zero terminal cost."""
            return 0.0

        return MFGComponents(
            hamiltonian_func=hamiltonian_func,
            hamiltonian_dm_func=hamiltonian_dm,
            initial_density_func=initial_density_func,
            final_value_func=terminal_cost_func,
        )


def solve_pure_diffusion():
    """
    Solve pure diffusion problem directly using FP solver.

    No HJB solver needed - just evolve density with zero velocity field.
    """
    print("\n" + "=" * 70)
    print("  2D Pure Diffusion - Dimensional Splitting Validation")
    print("=" * 70 + "\n")

    # Create problem
    print("Creating 2D pure diffusion problem...")
    problem = PureDiffusion2D(
        grid_resolution=12,  # 12x12 grid
        time_horizon=0.5,
        num_timesteps=15,
        diffusion=0.05,
        center=(0.5, 0.5),
    )

    print("  Domain: [0,1] × [0,1]")
    print(f"  Grid: {problem.geometry.grid.num_points[0]} × {problem.geometry.grid.num_points[1]}")
    print(f"  Timesteps: {problem.Nt + 1}")
    print(f"  dt: {problem.dt:.4f}")
    print(f"  Diffusion sigma: {problem.sigma}")
    print()

    # Get initial density
    grid_points = problem.geometry.grid.flatten()
    shape = tuple(problem.geometry.grid.num_points)
    m0 = problem.initial_density(grid_points).reshape(shape)

    # Zero velocity field (no advection)
    U_zero = np.zeros((problem.Nt + 1, *shape))

    print("Solving FP equation (pure diffusion, zero velocity field)...")

    # Import dimensional splitting FP solver
    from mfg_pde.alg.numerical.fp_solvers.fp_fdm_multid import solve_fp_nd_dimensional_splitting

    boundary_conditions = BoundaryConditions(type="no_flux")
    M_solution = solve_fp_nd_dimensional_splitting(
        m_initial_condition=m0,
        U_solution_for_drift=U_zero,
        problem=problem,
        boundary_conditions=boundary_conditions,
        show_progress=True,
    )

    # Results
    print("\n" + "-" * 70)
    print("Results:")
    print("-" * 70)

    # Mass conservation analysis
    dx, dy = problem.geometry.grid.spacing
    dV = dx * dy
    initial_mass = np.sum(M_solution[0]) * dV
    final_mass = np.sum(M_solution[-1]) * dV
    mass_error_percent = (final_mass - initial_mass) / initial_mass * 100

    print("\nMass conservation:")
    print(f"  Initial mass: {initial_mass:.10f}")
    print(f"  Final mass: {final_mass:.10f}")
    print(f"  Error: {mass_error_percent:+.4f}%")

    if abs(mass_error_percent) < 2.0:
        print("  ✓ PASS: Mass conservation within acceptable range (~1-2%)")
    else:
        print("  ✗ FAIL: Mass error exceeds acceptable range")

    # Density statistics
    print("\nDensity statistics:")
    print(f"  Initial max: {np.max(M_solution[0]):.6f}")
    print(f"  Final max: {np.max(M_solution[-1]):.6f}")
    print(f"  Initial sum*dV: {np.sum(M_solution[0]) * dV:.10f}")
    print(f"  Final sum*dV: {np.sum(M_solution[-1]) * dV:.10f}")

    return M_solution, problem


def save_solution_data(problem, M_solution, output_dir):
    """Save solution data to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as numpy arrays
    np.savez(
        output_dir / "pure_diffusion_solution.npz",
        M=M_solution,
        time_grid=problem.time_grid,
        x_grid=problem.geometry.grid.bounds[0],
        y_grid=problem.geometry.grid.bounds[1],
        dx=problem.geometry.grid.spacing[0],
        dy=problem.geometry.grid.spacing[1],
    )

    print(f"\nSolution data saved to {output_dir / 'pure_diffusion_solution.npz'}")


def plot_density_evolution(problem, M, output_dir):
    """Create visualization of density evolution."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get grid information
    Nx, Ny = problem.geometry.grid.num_points
    dx, dy = problem.geometry.grid.spacing
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Select time snapshots (initial, mid, final)
    num_snapshots = min(6, len(M))
    time_indices = np.linspace(0, len(M) - 1, num_snapshots, dtype=int)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Find global vmin/vmax for consistent colormap
    vmin = 0
    vmax = np.max([np.max(M[i]) for i in time_indices])

    for idx, t_idx in enumerate(time_indices):
        ax = axes[idx]

        # Plot density as heatmap
        im = ax.contourf(X, Y, M[t_idx], levels=20, cmap="viridis", vmin=vmin, vmax=vmax)

        # Mark center
        ax.plot(problem.center[0], problem.center[1], "r*", markersize=15, label="Center")

        # Compute mass at this time
        mass = np.sum(M[t_idx]) * dx * dy

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"t = {problem.time_grid[t_idx]:.3f} (mass = {mass:.6f})")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)

    # Add colorbar
    fig.colorbar(im, ax=axes, label="Density", orientation="horizontal", pad=0.05, aspect=40)

    plt.suptitle("2D Pure Diffusion: Density Evolution", fontsize=16, y=0.98)
    plt.tight_layout()

    # Save figure
    fig_path = output_dir / "density_evolution_pure_diffusion.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {fig_path}")

    plt.close()


def plot_mass_conservation(problem, M, output_dir):
    """Plot mass conservation over time."""
    output_dir = Path(output_dir)

    # Compute mass at each timestep
    dx, dy = problem.geometry.grid.spacing
    dV = dx * dy

    masses = [np.sum(M[k]) * dV for k in range(len(M))]
    initial_mass = masses[0]
    mass_errors = [(m - initial_mass) / initial_mass * 100 for m in masses]

    # Create plot
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot absolute mass
    ax1.plot(problem.time_grid, masses, "b-", linewidth=2)
    ax1.axhline(initial_mass, color="k", linestyle="--", label=f"Initial mass = {initial_mass:.6f}")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Total Mass")
    ax1.set_title("Mass Conservation: Absolute")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot relative error
    ax2.plot(problem.time_grid, mass_errors, "r-", linewidth=2)
    ax2.axhline(0, color="k", linestyle="--")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Mass Error (%)")
    ax2.set_title("Mass Conservation: Relative Error")
    ax2.grid(True, alpha=0.3)

    # Add statistics
    final_error = mass_errors[-1]
    ax2.text(
        0.05,
        0.95,
        f"Final error: {final_error:+.3f}%",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / "mass_conservation_pure_diffusion.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {fig_path}")

    plt.close()


def main():
    """Run 2D pure diffusion demonstration."""

    # Solve
    M_solution, problem = solve_pure_diffusion()

    # Save data and create visualizations
    print("\n" + "-" * 70)
    print("Saving data and creating visualizations...")
    print("-" * 70)

    output_dir = Path("benchmarks/validation/results/pure_diffusion_2d")

    save_solution_data(problem, M_solution, output_dir)
    plot_density_evolution(problem, M_solution, output_dir)
    plot_mass_conservation(problem, M_solution, output_dir)

    # Compute final statistics
    dx, dy = problem.geometry.grid.spacing
    dV = dx * dy
    initial_mass = np.sum(M_solution[0]) * dV
    final_mass = np.sum(M_solution[-1]) * dV
    mass_error_percent = (final_mass - initial_mass) / initial_mass * 100

    print("\n" + "=" * 70)
    print("  Demonstration Complete!")
    print("=" * 70)
    print(f"\nOutput saved to: {output_dir}")
    print("Files created:")
    print("  - pure_diffusion_solution.npz (solution data)")
    print("  - density_evolution_pure_diffusion.png (6 time snapshots)")
    print("  - mass_conservation_pure_diffusion.png (mass tracking)")
    print()
    print("Key observations:")
    print(f"  • Mass error: {mass_error_percent:+.4f}% (dimensional splitting, pure diffusion)")
    print("  • No advection - just smooth diffusion from center")
    print("  • Density spreads symmetrically (Gaussian → flatter Gaussian)")
    print("  • Validates dimensional splitting works for diffusion-dominated cases")
    print()

    return M_solution, problem


if __name__ == "__main__":
    main()
