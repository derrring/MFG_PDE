"""
2D Crowd Evacuation - Proper nD Solver Implementation

Uses GridBasedMFGProblem with standard Hamiltonian signature for true 2D solving.
This implementation follows the Phase 3 nD pattern with:
- GridBasedMFGProblem base class
- Standard hamiltonian(x, m, p, t) signature
- Proper gradient vector p computation
- True 2D HJB and FP solving (no dimension splitting)

Physical Setup:
- Room: 10m × 10m square domain
- Initial crowd: Gaussian at (5.0, 7.0)
- Exit: Located at y=0 (bottom boundary)
- Congestion-aware dynamics
"""

from __future__ import annotations

import time
from pathlib import Path

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem


class CrowdEvacuation2D(GridBasedMFGProblem):
    """
    2D Crowd Evacuation Problem using proper nD infrastructure.

    Implements GridBasedMFGProblem with standard Hamiltonian signature.
    """

    def __init__(
        self,
        grid_resolution: int = 30,
        time_horizon: float = 8.0,
        num_timesteps: int = 50,
        diffusion_coeff: float = 0.1,
        congestion_weight: float = 2.0,
    ):
        """
        Initialize 2D crowd evacuation problem.

        Args:
            grid_resolution: Grid points per dimension (N×N grid)
            time_horizon: Final time T
            num_timesteps: Number of time steps
            diffusion_coeff: Diffusion coefficient σ
            congestion_weight: Congestion cost weight λ
        """
        # Initialize base class with 2D domain
        super().__init__(
            domain_bounds=(0.0, 10.0, 0.0, 10.0),  # 2D: (xmin, xmax, ymin, ymax)
            grid_resolution=grid_resolution,  # N×N grid
            time_domain=(time_horizon, num_timesteps),
            diffusion_coeff=diffusion_coeff,
        )

        # Problem-specific parameters
        self.congestion_weight = congestion_weight
        self.start_location = np.array([5.0, 7.0])  # Initial crowd center
        self.exit_location = np.array([5.0, 0.0])  # Exit at bottom

        print("Created 2D Crowd Evacuation Problem:")
        print(f"  Grid: {grid_resolution}×{grid_resolution} = {self.num_spatial_points} points")
        print("  Domain: [0, 10] × [0, 10] m")
        print(f"  Time: T={time_horizon}s, Nt={num_timesteps}")
        print(f"  Diffusion: σ={diffusion_coeff}")
        print(f"  Congestion: λ={congestion_weight}")

    def hamiltonian(self, x, m, p, t):
        """
        Standard Hamiltonian: H(x, m, p, t)

        H = (1/2)|p|² + λ·m·|p|²

        Where:
        - (1/2)|p|² is kinetic cost (control effort)
        - λ·m·|p|² is congestion cost (density-dependent friction)

        Args:
            x: ndarray shape (2,) or (N, 2) - spatial coordinates
            m: scalar or ndarray shape (N,) - density
            p: ndarray shape (2,) or (N, 2) - gradient vector ∇u
            t: scalar - time (unused in this problem)

        Returns:
            scalar or ndarray shape (N,) - Hamiltonian value
        """
        # Handle both single point and vectorized input
        if p.ndim == 1:
            # Single point: p is (2,)
            p_squared = np.sum(p**2)
        else:
            # Vectorized: p is (N, 2)
            p_squared = np.sum(p**2, axis=1)

        # Kinetic cost: (1/2)|p|²
        kinetic = 0.5 * p_squared

        # Congestion cost: λ·m·|p|²
        congestion = self.congestion_weight * m * p_squared

        return kinetic + congestion

    def initial_density(self, x):
        """
        Initial density: Gaussian blob at start location.

        m₀(x) = exp(-||x - x₀||²/(2σ²))

        Args:
            x: ndarray shape (N, 2) - spatial points

        Returns:
            ndarray shape (N,) - density values
        """
        # Distance squared to start location
        dist_sq = np.sum((x - self.start_location) ** 2, axis=1)

        # Gaussian with width 1.5m
        density = np.exp(-dist_sq / (2 * 1.5**2))

        # Normalize to unit mass
        total_mass = np.sum(density)
        if total_mass > 0:
            density = density / total_mass

        return density

    def terminal_cost(self, x):
        """
        Terminal cost: Distance to exit.

        g(x) = y (vertical distance to y=0)

        Args:
            x: ndarray shape (N, 2) - spatial points

        Returns:
            ndarray shape (N,) - terminal cost values
        """
        # Cost is simply the y-coordinate (distance to y=0)
        return x[:, 1]

    def running_cost(self, x, t):
        """
        Running cost: Constant time penalty.

        f(x, t) = 1 (encourages reaching exit quickly)

        Args:
            x: ndarray shape (N, 2) - spatial points
            t: scalar - time

        Returns:
            ndarray shape (N,) - running cost values
        """
        # Constant time penalty to encourage fast evacuation
        return np.ones(x.shape[0])

    def setup_components(self):
        """
        Setup MFGComponents (required abstract method).

        Modern FDM solvers don't use this - they call hamiltonian() directly.
        Return None to indicate we're using the standard interface.
        """
        return None


def solve_2d_evacuation(
    problem: CrowdEvacuation2D, damping: float = 0.5, max_iterations: int = 30, verbose: bool = True
):
    """
    Solve 2D evacuation using nD FDM solvers via factory.

    Args:
        problem: CrowdEvacuation2D instance
        damping: Damping factor for Picard iteration
        max_iterations: Maximum Picard iterations
        verbose: Print progress

    Returns:
        Tuple of (solver_result, solve_time)
    """
    from mfg_pde.factory import create_basic_solver

    if verbose:
        print("\n" + "=" * 70)
        print("SOLVING 2D CROWD EVACUATION")
        print("=" * 70)
        print("  Solver: HJB-FDM + FP-FDM (auto-detected 2D)")
        print(f"  Picard damping: {damping}")
        print(f"  Max iterations: {max_iterations}")
        print()

    # Create solver using factory (auto-detects 2D)
    solver = create_basic_solver(
        problem,
        damping=damping,
        max_iterations=max_iterations,
    )

    # Solve
    start_time = time.time()
    result = solver.solve()
    solve_time = time.time() - start_time

    if verbose:
        print(f"\n{'=' * 70}")
        if result is not None:
            print(f"✓ Solution completed in {solve_time:.2f}s")
            if hasattr(result, "converged"):
                print(f"  Converged: {result.converged}")
            if hasattr(result, "iterations"):
                print(f"  Iterations: {result.iterations}")
        else:
            print("✗ Solution failed")
        print(f"{'=' * 70}\n")

    return result, solve_time


def visualize_2d_evolution(
    problem: CrowdEvacuation2D,
    result,
    num_snapshots: int = 9,
    save_path: str | None = None,
):
    """
    Visualize 2D density evolution with proper grid structure.

    Args:
        problem: CrowdEvacuation2D instance
        result: Solver result object
        num_snapshots: Number of time snapshots
        save_path: Optional path to save figure
    """
    if result is None:
        print("Cannot visualize - solution failed")
        return

    # Extract density history from result
    # Result has M attribute with shape (Nt, N1, N2) or flattened
    if hasattr(result, "M"):
        density_flat = result.M  # Shape: (Nt, N1*N2) or (Nt, N1, N2)
    else:
        print("Cannot visualize - no density data in result")
        return

    # Reshape if needed
    Nt = density_flat.shape[0]
    if density_flat.ndim == 2:
        # Flatten format: (Nt, N1*N2) -> (Nt, N1, N2)
        N1 = problem.geometry.grid.num_points[0]
        N2 = problem.geometry.grid.num_points[1]
        density_history = density_flat.reshape(Nt, N1, N2)
    else:
        # Already correct shape: (Nt, N1, N2)
        density_history = density_flat
    Nt = density_history.shape[0]

    # Get grid coordinates
    grid = problem.geometry.grid
    x1_coords = grid.coordinates[0]
    x2_coords = grid.coordinates[1]
    X1, X2 = np.meshgrid(x1_coords, x2_coords, indexing="ij")

    # Select time indices
    time_indices = np.linspace(0, Nt - 1, num_snapshots, dtype=int)
    times = [problem.time_grid[t] for t in time_indices]

    # Create figure
    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15))
    fig.suptitle("2D Crowd Evacuation - Proper nD Solver", fontsize=16, fontweight="bold")

    # Global colorbar range
    vmin = 0
    vmax = np.max(density_history) * 1.1

    # Plot each snapshot
    for idx, (ax, t_idx, t) in enumerate(zip(axes.flat, time_indices, times)):
        # Density heatmap
        im = ax.contourf(X1, X2, density_history[t_idx], levels=20, cmap="YlOrRd", vmin=vmin, vmax=vmax)

        # Exit line at y=0
        ax.axhline(y=0, color="green", linewidth=3, linestyle="--", alpha=0.7, label="Exit" if idx == 0 else "")

        # Formatting
        ax.set_xlabel("x (m)", fontsize=10)
        ax.set_ylabel("y (m)", fontsize=10)
        ax.set_title(f"t = {t:.2f}s", fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle=":")

        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Density", fontsize=9)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Figure saved to: {save_path}")

    plt.show()


def compute_evacuation_metrics(problem: CrowdEvacuation2D, result):
    """
    Compute evacuation metrics from solution.

    Args:
        problem: CrowdEvacuation2D instance
        result: Solver result object

    Returns:
        Dictionary of metrics
    """
    if result is None or not hasattr(result, "M"):
        return {"success": False}

    # Extract and reshape density
    density_flat = result.M
    Nt = density_flat.shape[0]
    if density_flat.ndim == 2:
        N1 = problem.geometry.grid.num_points[0]
        N2 = problem.geometry.grid.num_points[1]
        density_history = density_flat.reshape(Nt, N1, N2)
    else:
        density_history = density_flat  # Already (Nt, N1, N2)
    Nt = density_history.shape[0]

    # Grid info
    grid = problem.geometry.grid
    dx1, dx2 = grid.spacing
    area_element = dx1 * dx2

    # Get coordinates
    x1_coords = grid.coordinates[0]
    x2_coords = grid.coordinates[1]
    X1, X2 = np.meshgrid(x1_coords, x2_coords, indexing="ij")

    # Compute metrics over time
    masses = []
    com_x = []
    com_y = []
    exit_flux = []

    for t in range(Nt):
        density = density_history[t]

        # Total mass
        mass = np.sum(density) * area_element
        masses.append(mass)

        # Center of mass
        if mass > 0:
            cx = np.sum(X1 * density) / np.sum(density)
            cy = np.sum(X2 * density) / np.sum(density)
        else:
            cx, cy = 0, 0
        com_x.append(cx)
        com_y.append(cy)

        # Exit flux (mass in bottom 10% of domain)
        bottom_mask = X2 < 1.0  # y < 1.0m
        exit_mass = np.sum(density[bottom_mask]) * area_element
        exit_flux.append(exit_mass / mass if mass > 0 else 0)

    # Mass conservation
    mass_loss_percent = 100 * (masses[0] - masses[-1]) / masses[0] if masses[0] > 0 else 0

    metrics = {
        "success": True,
        "time": problem.time_grid,
        "masses": np.array(masses),
        "mass_loss_percent": mass_loss_percent,
        "com_x": np.array(com_x),
        "com_y": np.array(com_y),
        "exit_flux": np.array(exit_flux),
        "initial_com": (com_x[0], com_y[0]),
        "final_com": (com_x[-1], com_y[-1]),
        "vertical_displacement": com_y[0] - com_y[-1],
        "evacuation_progress": exit_flux[-1] * 100,
    }

    return metrics


def visualize_metrics(metrics: dict, save_path: str | None = None):
    """
    Visualize evacuation metrics.

    Args:
        metrics: Metrics dictionary
        save_path: Optional save path
    """
    if not metrics["success"]:
        print("Cannot visualize - no metrics available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("2D Crowd Evacuation Metrics", fontsize=14, fontweight="bold")

    time = metrics["time"]

    # Mass conservation
    ax = axes[0, 0]
    ax.plot(time, metrics["masses"], "b-", linewidth=2, label="Total mass")
    ax.axhline(y=metrics["masses"][0], color="r", linestyle="--", linewidth=1.5, label="Initial mass")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Total Mass", fontsize=11)
    ax.set_title(f"Mass Conservation (loss: {metrics['mass_loss_percent']:.4f}%)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Center of mass trajectory
    ax = axes[0, 1]
    ax.plot(metrics["com_x"], metrics["com_y"], "b-", linewidth=2, alpha=0.7)
    ax.scatter(metrics["com_x"][0], metrics["com_y"][0], c="green", s=100, marker="o", label="Start", zorder=5)
    ax.scatter(metrics["com_x"][-1], metrics["com_y"][-1], c="red", s=100, marker="s", label="End", zorder=5)
    ax.axhline(y=0, color="green", linestyle="--", linewidth=2, alpha=0.5, label="Exit")
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)
    ax.set_title(
        f"Center of Mass Trajectory (Δy={metrics['vertical_displacement']:.2f}m)", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Vertical position over time
    ax = axes[1, 0]
    ax.plot(time, metrics["com_y"], "b-", linewidth=2)
    ax.axhline(y=0, color="green", linestyle="--", linewidth=1.5, label="Exit level")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Y-coordinate (m)", fontsize=11)
    ax.set_title("Vertical Position of Center of Mass", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Exit flux over time
    ax = axes[1, 1]
    ax.plot(time, metrics["exit_flux"] * 100, "g-", linewidth=2)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Exit Flux (%)", fontsize=11)
    ax.set_title(f"Evacuation Progress (final: {metrics['evacuation_progress']:.1f}%)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Metrics figure saved to: {save_path}")

    plt.show()


def main():
    """Run 2D crowd evacuation with proper nD solver."""
    print("\n" + "=" * 70)
    print("2D CROWD EVACUATION - PROPER nD IMPLEMENTATION")
    print("=" * 70)
    print("\nUsing GridBasedMFGProblem with standard Hamiltonian signature")
    print("True 2D solving with HJB-FDM and FP-FDM (no dimensional splitting)")
    print()

    # Create problem
    problem = CrowdEvacuation2D(
        grid_resolution=20,  # 20×20 = 400 points
        time_horizon=8.0,  # 8 seconds
        num_timesteps=50,  # 50 timesteps
        diffusion_coeff=0.2,  # σ = 0.2
        congestion_weight=2.0,  # λ = 2.0
    )

    # Solve
    result, solve_time = solve_2d_evacuation(problem, damping=0.5, max_iterations=30, verbose=True)

    if result is not None:
        # Compute metrics
        print("Computing evacuation metrics...")
        metrics = compute_evacuation_metrics(problem, result)

        print(f"\n{'=' * 70}")
        print("RESULTS SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Solve time: {solve_time:.2f}s")
        print(f"  Mass loss: {metrics['mass_loss_percent']:.4f}%")
        print(f"  Initial CoM: ({metrics['initial_com'][0]:.2f}, {metrics['initial_com'][1]:.2f})")
        print(f"  Final CoM: ({metrics['final_com'][0]:.2f}, {metrics['final_com'][1]:.2f})")
        print(f"  Vertical displacement: {metrics['vertical_displacement']:.2f}m")
        print(f"  Evacuation progress: {metrics['evacuation_progress']:.1f}%")
        print(f"{'=' * 70}\n")

        # Visualize
        print("Creating visualizations...")

        density_path = "examples/outputs/particle_methods/2d_proper_nd_density.png"
        visualize_2d_evolution(problem, result, num_snapshots=9, save_path=density_path)

        metrics_path = "examples/outputs/particle_methods/2d_proper_nd_metrics.png"
        visualize_metrics(metrics, save_path=metrics_path)

        print(f"\n{'=' * 70}")
        print("✓ 2D CROWD EVACUATION COMPLETE")
        print(f"{'=' * 70}")
        print("\nGenerated files:")
        print(f"  1. {density_path}")
        print(f"  2. {metrics_path}")
        print()
    else:
        print(f"\n✗ Solve failed: {result.get('error', 'Unknown error')}\n")


if __name__ == "__main__":
    main()
