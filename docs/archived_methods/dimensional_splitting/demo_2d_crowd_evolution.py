"""
Demonstrate 2D Crowd Motion with Bug #8 Fix

Shows density evolution over time for a meaningful 2D MFG problem.
Saves data and creates visualizations to demonstrate the fixed FP solver.

Problem:
- Agents start concentrated at bottom-left
- Goal is top-right corner
- Minimize travel cost + congestion cost
- Demonstrate mass conservation with Bug #8 fix
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import MFGComponents
from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem
from mfg_pde.factory import create_basic_solver


class CrowdMotion2D(GridBasedMFGProblem):
    """
    2D crowd motion MFG problem.

    Agents navigate from start to goal while avoiding congestion.
    """

    def __init__(
        self,
        grid_resolution=16,
        time_horizon=0.5,
        num_timesteps=20,
        diffusion=0.05,
        congestion_weight=0.5,
        goal=(0.8, 0.8),
        start=(0.2, 0.2),
    ):
        super().__init__(
            domain_bounds=(0.0, 1.0, 0.0, 1.0),
            grid_resolution=grid_resolution,
            time_domain=(time_horizon, num_timesteps),
            diffusion_coeff=diffusion,
        )

        self.congestion_weight = congestion_weight
        self.goal = np.array(goal)
        self.start = np.array(start)

    def initial_density(self, x):
        """Gaussian blob centered at start position."""
        dist_sq = np.sum((x - self.start) ** 2, axis=1)
        density = np.exp(-100 * dist_sq)
        # Normalize by integral: ∫m dx = 1, not just ∑m = 1
        dV = float(np.prod(self.geometry.grid.spacing))
        return density / (np.sum(density) * dV + 1e-10)

    def terminal_cost(self, x):
        """Quadratic cost: distance to goal."""
        dist_sq = np.sum((x - self.goal) ** 2, axis=1)
        return 5.0 * dist_sq

    def running_cost(self, x, t):
        """Small running cost encourages fast movement."""
        return 0.1 * np.ones(x.shape[0])

    def hamiltonian(self, x, m, p, t):
        """H = (1/2)|p|^2 + kappa*m (isotropic control + congestion)."""
        p_squared = np.sum(p**2, axis=1) if p.ndim > 1 else p**2
        h = 0.5 * p_squared
        h += self.congestion_weight * m
        return h

    def setup_components(self):
        """Setup MFG components for numerical solvers."""
        kappa = self.congestion_weight

        def hamiltonian_func(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, derivs=None, **kwargs):
            """H = (1/2)|grad u|^2 + kappa*m"""
            h_value = 0.0

            if derivs is not None:
                # Extract gradient grad u
                grad_u = []
                for d in range(2):  # 2D
                    idx_tuple = tuple([0] * d + [1] + [0] * (2 - d - 1))
                    grad_u.append(derivs.get(idx_tuple, 0.0))
                h_value += 0.5 * float(np.sum(np.array(grad_u) ** 2))

            # Congestion cost
            h_value += kappa * float(m_at_x)

            return h_value

        def hamiltonian_dm(x_idx, x_position, m_at_x, **kwargs):
            """dH/dm = kappa"""
            return kappa

        def initial_density_func(x_idx):
            """Gaussian initial density."""
            # Convert grid index to physical position
            coords = []
            for d in range(2):
                idx_d = x_idx[d] if isinstance(x_idx, tuple) else x_idx
                x_d = self.geometry.grid.bounds[d][0] + idx_d * self.geometry.grid.spacing[d]
                coords.append(x_d)
            coords_array = np.array(coords).reshape(1, -1)
            return float(self.initial_density(coords_array)[0])

        def terminal_cost_func(x_idx):
            """Quadratic terminal cost."""
            coords = []
            for d in range(2):
                idx_d = x_idx[d] if isinstance(x_idx, tuple) else x_idx
                x_d = self.geometry.grid.bounds[d][0] + idx_d * self.geometry.grid.spacing[d]
                coords.append(x_d)
            coords_array = np.array(coords).reshape(1, -1)
            return float(self.terminal_cost(coords_array)[0])

        return MFGComponents(
            hamiltonian_func=hamiltonian_func,
            hamiltonian_dm_func=hamiltonian_dm,
            initial_density_func=initial_density_func,
            final_value_func=terminal_cost_func,
        )


def save_solution_data(problem, result, output_dir):
    """Save solution data to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as numpy arrays
    np.savez(
        output_dir / "crowd_motion_solution.npz",
        U=result.U,
        M=result.M,
        time_grid=problem.time_grid,
        x_grid=problem.geometry.grid.bounds[0],
        y_grid=problem.geometry.grid.bounds[1],
        dx=problem.geometry.grid.spacing[0],
        dy=problem.geometry.grid.spacing[1],
        converged=result.converged,
        iterations=result.iterations,
    )

    print(f"Solution data saved to {output_dir / 'crowd_motion_solution.npz'}")


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

        # Mark start and goal
        ax.plot(problem.start[0], problem.start[1], "wo", markersize=10, label="Start")
        ax.plot(problem.goal[0], problem.goal[1], "r*", markersize=15, label="Goal")

        # Compute mass at this time
        mass = np.sum(M[t_idx]) * dx * dy

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"t = {problem.time_grid[t_idx]:.3f} (mass = {mass:.4f})")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)

    # Add colorbar
    fig.colorbar(im, ax=axes, label="Density", orientation="horizontal", pad=0.05, aspect=40)

    plt.suptitle("2D Crowd Motion: Density Evolution", fontsize=16, y=0.98)
    plt.tight_layout()

    # Save figure
    fig_path = output_dir / "density_evolution.png"
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
        f"Final error: {final_error:+.2f}%",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / "mass_conservation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {fig_path}")

    plt.close()


def main():
    """Run 2D crowd motion demonstration with visualization."""
    print("\n" + "=" * 70)
    print("  2D Crowd Motion MFG Problem - Bug #8 Fix Demonstration")
    print("=" * 70 + "\n")

    # Create problem
    print("Creating 2D MFG problem...")
    problem = CrowdMotion2D(
        grid_resolution=16,  # 16x16 grid
        time_horizon=0.5,
        num_timesteps=20,
        diffusion=0.05,
        congestion_weight=0.3,
        goal=(0.8, 0.8),
        start=(0.2, 0.2),
    )

    print("  Domain: [0,1] × [0,1]")
    print(f"  Grid: {problem.geometry.grid.num_points[0]} × {problem.geometry.grid.num_points[1]}")
    print(f"  Timesteps: {problem.Nt + 1}")
    print(f"  dt: {problem.dt:.4f}")
    print(f"  Start: {problem.start}")
    print(f"  Goal: {problem.goal}")
    print(f"  Diffusion sigma: {problem.sigma}")
    print(f"  Congestion weight kappa: {problem.congestion_weight}")

    # Create solver
    print("\nCreating solver...")
    solver = create_basic_solver(
        problem,
        damping=0.6,
        max_iterations=30,
        tolerance=1e-4,
    )

    print(f"  HJB solver: {solver.hjb_solver.__class__.__name__}")
    print(f"  FP solver: {solver.fp_solver.__class__.__name__}")
    print(f"  Dimension: {solver.hjb_solver.dimension}D")

    # Solve
    print("\nSolving MFG system...")
    result = solver.solve()

    # Results
    print("\n" + "-" * 70)
    print("Results:")
    print("-" * 70)
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final error: {result.residual:.6e}")

    # Mass conservation analysis
    U, M = result.U, result.M
    dx, dy = problem.geometry.grid.spacing
    dV = dx * dy
    initial_mass = np.sum(M[0]) * dV
    final_mass = np.sum(M[-1]) * dV
    mass_error_percent = (final_mass - initial_mass) / initial_mass * 100

    print("\nMass conservation:")
    print(f"  Initial mass: {initial_mass:.6f}")
    print(f"  Final mass: {final_mass:.6f}")
    print(f"  Error: {mass_error_percent:+.2f}%")

    # Density statistics
    print("\nDensity statistics:")
    print(f"  Initial max: {np.max(M[0]):.6f}")
    print(f"  Final max: {np.max(M[-1]):.6f}")
    print(f"  Initial sum*dV: {np.sum(M[0]) * dV:.6f}")
    print(f"  Final sum*dV: {np.sum(M[-1]) * dV:.6f}")

    # Value function statistics
    print("\nValue function statistics:")
    print(f"  At t=0: min={np.min(U[0]):.4f}, max={np.max(U[0]):.4f}")
    print(f"  At t=T: min={np.min(U[-1]):.4f}, max={np.max(U[-1]):.4f}")

    # Save data and create visualizations
    print("\n" + "-" * 70)
    print("Saving data and creating visualizations...")
    print("-" * 70)

    output_dir = Path("benchmarks/validation/results/crowd_motion_2d")

    save_solution_data(problem, result, output_dir)
    plot_density_evolution(problem, M, output_dir)
    plot_mass_conservation(problem, M, output_dir)

    print("\n" + "=" * 70)
    print("  Demonstration Complete!")
    print("=" * 70)
    print(f"\nOutput saved to: {output_dir}")
    print("Files created:")
    print("  - crowd_motion_solution.npz (solution data)")
    print("  - density_evolution.png (6 time snapshots)")
    print("  - mass_conservation.png (mass tracking)")
    print()
    print("Key observations:")
    print(f"  • Mass error: {mass_error_percent:+.2f}% (Bug #8 fix applied)")
    print(f"  • Converged in {result.iterations} iterations")
    print("  • Density evolves from start (0.2, 0.2) toward goal (0.8, 0.8)")
    print("  • Dimensional splitting used for 2D FP solver")
    print()

    return result, problem


if __name__ == "__main__":
    main()
