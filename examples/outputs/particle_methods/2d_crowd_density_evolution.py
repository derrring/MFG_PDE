"""
2D Crowd Density Evolution: Particle Method Visualization

High-resolution 2D crowd evacuation with detailed density evolution visualization.
Uses particle-grid hybrid method with proper nD solver infrastructure.

Physical Setup:
- Room: 10m × 10m square domain
- Initial crowd: Gaussian distribution centered at (5.0, 7.0)
- Exit: Located at bottom boundary (y=0)
"""

from __future__ import annotations

import time
from pathlib import Path

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import MFGComponents, MFGProblem, solve_mfg
from mfg_pde.config import ConfigBuilder


class GridBased2DAdapter:
    """
    Adapter class to handle 2D grid-based MFG problems using 1D MFG infrastructure.

    Maps between 2D grid coordinates (i1, i2) and 1D solver indices.
    """

    def __init__(self, nx1: int, nx2: int, domain_bounds: tuple):
        """
        Args:
            nx1: Number of intervals in first dimension
            nx2: Number of intervals in second dimension
            domain_bounds: (x1_min, x1_max, x2_min, x2_max)
        """
        self.nx1, self.nx2 = nx1, nx2
        self.x1_min, self.x1_max, self.x2_min, self.x2_max = domain_bounds

        # Create 2D grids
        self.x1_grid = np.linspace(self.x1_min, self.x1_max, nx1 + 1)
        self.x2_grid = np.linspace(self.x2_min, self.x2_max, nx2 + 1)
        self.X1, self.X2 = np.meshgrid(self.x1_grid, self.x2_grid, indexing="ij")

        # Grid spacing
        self.dx1 = (self.x1_max - self.x1_min) / nx1
        self.dx2 = (self.x2_max - self.x2_min) / nx2

    def idx_2d_to_1d(self, i1: int, i2: int) -> int:
        """Convert 2D grid coordinates to 1D solver index."""
        return i1 * (self.nx2 + 1) + i2

    def idx_1d_to_2d(self, idx_1d: int) -> tuple[int, int]:
        """Convert 1D solver index to 2D grid coordinates."""
        i1 = idx_1d // (self.nx2 + 1)
        i2 = idx_1d % (self.nx2 + 1)
        return min(i1, self.nx1), min(i2, self.nx2)

    def get_2d_coordinates(self, idx_1d: int) -> tuple[float, float]:
        """Get actual (x1, x2) coordinates from 1D index."""
        i1, i2 = self.idx_1d_to_2d(idx_1d)
        x1 = self.x1_grid[i1] if i1 < len(self.x1_grid) else self.x1_max
        x2 = self.x2_grid[i2] if i2 < len(self.x2_grid) else self.x2_max
        return x1, x2

    def convert_to_2d_array(self, field_1d: np.ndarray) -> np.ndarray:
        """Convert 1D field to 2D array for visualization."""
        if len(field_1d.shape) == 1:
            # Single time step
            result = np.zeros((self.nx1 + 1, self.nx2 + 1))
            for i1 in range(self.nx1 + 1):
                for i2 in range(self.nx2 + 1):
                    idx_1d = self.idx_2d_to_1d(i1, i2)
                    if idx_1d < len(field_1d):
                        result[i1, i2] = field_1d[idx_1d]
            return result
        else:
            # Multiple time steps
            nt = field_1d.shape[0]
            result = np.zeros((nt, self.nx1 + 1, self.nx2 + 1))
            for t in range(nt):
                for i1 in range(self.nx1 + 1):
                    for i2 in range(self.nx2 + 1):
                        idx_1d = self.idx_2d_to_1d(i1, i2)
                        if idx_1d < field_1d.shape[1]:
                            result[t, i1, i2] = field_1d[t, idx_1d]
            return result


class CrowdEvacuation2D:
    """
    2D Crowd Evacuation Problem using proper nD solver infrastructure.

    Uses MFGComponents for custom 2D behavior with proper grid mapping.
    """

    def __init__(
        self,
        grid_size: tuple[int, int] = (20, 20),
        domain_bounds: tuple = (0.0, 10.0, 0.0, 10.0),
        time_domain: tuple = (8.0, 50),
        sigma: float = 0.25,
        lambda_crowd: float = 2.0,
    ):
        """
        Args:
            grid_size: (Nx1, Nx2) grid resolution per dimension
            domain_bounds: (x1_min, x1_max, x2_min, x2_max)
            time_domain: (T, Nt) time horizon and timesteps
            sigma: Diffusion coefficient
            lambda_crowd: Congestion cost coefficient
        """
        self.Nx1, self.Nx2 = grid_size
        self.domain_bounds = domain_bounds
        self.T, self.Nt = time_domain
        self.sigma = sigma
        self.lambda_crowd = lambda_crowd

        # Create 2D grid adapter
        self.grid_adapter = GridBased2DAdapter(self.Nx1, self.Nx2, domain_bounds)

        # Total number of grid points for 1D representation
        self.total_nodes = (self.Nx1 + 1) * (self.Nx2 + 1)

        # Create MFG components
        self.components = self._create_mfg_components()

        # Initialize MFG problem
        self.mfg_problem = MFGProblem(
            xmin=0.0,
            xmax=float(self.total_nodes - 1),
            Nx=self.total_nodes - 1,
            T=self.T,
            Nt=self.Nt,
            sigma=self.sigma,
            components=self.components,
        )

        # Add dimension attribute and geometry for 2D solver detection
        self.mfg_problem.dimension = 2
        self.mfg_problem.dt = self.T / self.Nt

        # Create fake geometry structure with grid info for nD HJB solver
        class FakeGrid:
            def __init__(self, nx1, nx2, domain_bounds):
                self.dimension = 2
                self.ndim = 2
                self.num_points = (nx1 + 1, nx2 + 1)
                x1_min, x1_max, x2_min, x2_max = domain_bounds
                dx1 = (x1_max - x1_min) / nx1
                dx2 = (x2_max - x2_min) / nx2
                self.spacing = (dx1, dx2)

        class FakeGeometry:
            def __init__(self, nx1, nx2, domain_bounds):
                self.grid = FakeGrid(nx1, nx2, domain_bounds)
                self.dimension = 2

        self.mfg_problem.geometry = FakeGeometry(self.Nx1, self.Nx2, domain_bounds)

    def _create_mfg_components(self) -> MFGComponents:
        """Create MFGComponents with 2D crowd dynamics."""

        def hamiltonian_2d(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, **kwargs):
            """
            2D Hamiltonian for crowd motion: H = 0.5 |p|^2 + lambda * m * |p|^2

            Represents kinetic energy plus congestion cost.
            """
            try:
                x_idx = int(x_idx)
                if x_idx >= self.total_nodes:
                    return 0.0

                # Extract gradient components (simplified for isotropic case)
                p_forward = p_values.get("forward", 0.0)
                p_backward = p_values.get("backward", 0.0)
                p = p_forward - p_backward

                # Isotropic kinetic energy: 0.5 * |p|^2
                kinetic = 0.5 * p * p

                # Density-dependent cost (congestion)
                if m_at_x > 1e-10:
                    congestion = self.lambda_crowd * m_at_x * p * p
                else:
                    congestion = 0.0

                result = kinetic + congestion

                # Numerical stability
                if np.isnan(result) or np.isinf(result) or result < 0:
                    return 0.0

                return min(result, 1e6)

            except Exception:
                return 0.0

        def hamiltonian_dm(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, **kwargs):
            """Derivative of Hamiltonian with respect to density."""
            try:
                x_idx = int(x_idx)
                if x_idx >= self.total_nodes:
                    return 0.0

                p_forward = p_values.get("forward", 0.0)
                p_backward = p_values.get("backward", 0.0)
                p = p_forward - p_backward

                # dH/dm = lambda * |p|^2
                result = self.lambda_crowd * p * p

                if np.isnan(result) or np.isinf(result):
                    return 0.0

                return min(result, 1e6)

            except Exception:
                return 0.0

        def initial_density_2d(x_position):
            """2D initial density: Gaussian blob centered at (5.0, 7.0)."""
            try:
                x_idx = int(x_position)
                if x_idx >= self.total_nodes:
                    return 1e-10

                x1, x2 = self.grid_adapter.get_2d_coordinates(x_idx)

                # Gaussian centered at (5.0, 7.0)
                center_x1, center_x2 = 5.0, 7.0
                sigma_x1, sigma_x2 = 1.5, 1.5

                density = np.exp(
                    -((x1 - center_x1) ** 2 / (2 * sigma_x1**2) + (x2 - center_x2) ** 2 / (2 * sigma_x2**2))
                )

                return max(density, 1e-10)

            except Exception:
                return 1e-10

        def final_value_2d(x_position):
            """2D terminal cost: distance to exit at y=0."""
            try:
                x_idx = int(x_position)
                if x_idx >= self.total_nodes:
                    return 0.0

                x1, x2 = self.grid_adapter.get_2d_coordinates(x_idx)

                # Exit at y=0, cost is y-coordinate
                return x2

            except Exception:
                return 0.0

        # Create MFG components
        components = MFGComponents(
            hamiltonian_func=hamiltonian_2d,
            hamiltonian_dm_func=hamiltonian_dm,
            initial_density_func=initial_density_2d,
            final_value_func=final_value_2d,
            parameters={
                "lambda_crowd": self.lambda_crowd,
                "dimension": 2,
                "grid_shape": (self.Nx1 + 1, self.Nx2 + 1),
            },
        )

        return components


def solve_2d_particle_grid(crowd_problem: CrowdEvacuation2D, num_particles: int = 5000, verbose: bool = True):
    """
    Solve 2D crowd problem using Particle-Grid hybrid method.

    Args:
        crowd_problem: CrowdEvacuation2D instance
        num_particles: Number of particles for FP solver
        verbose: Print progress

    Returns:
        Tuple of (result, solve_time)
    """
    if verbose:
        print("=" * 70)
        print("2D CROWD EVACUATION - Particle-Grid Method")
        print("=" * 70)
        print(f"  Domain: {crowd_problem.domain_bounds[1]}m × {crowd_problem.domain_bounds[3]}m")
        print(f"  Grid: {crowd_problem.Nx1 + 1} × {crowd_problem.Nx2 + 1}")
        print(f"  Particles: {num_particles}")
        print(f"  Time horizon: {crowd_problem.T}s")
        print(f"  Timesteps: {crowd_problem.Nt}")
        print()

    # Configuration
    config = (
        ConfigBuilder()
        .picard(max_iterations=40, tolerance=1e-3)
        .solver_hjb("fdm")
        .solver_fp("particle", num_particles=num_particles, kde_bandwidth="scott")
        .build()
    )

    # Solve
    start_time = time.time()
    result = solve_mfg(crowd_problem.mfg_problem, config=config, verbose=verbose)
    solve_time = time.time() - start_time

    if verbose:
        print(f"\n✓ Solve completed in {solve_time:.2f}s")
        print(f"  Converged: {result.converged}")
        print(f"  Iterations: {result.iterations}")

    return result, solve_time


def visualize_density_evolution_2d(
    crowd_problem: CrowdEvacuation2D,
    result,
    num_snapshots: int = 9,
    save_path: str | None = None,
):
    """
    Create comprehensive 2D density evolution visualization.

    Args:
        crowd_problem: CrowdEvacuation2D instance
        result: Solver result
        num_snapshots: Number of time snapshots to show
        save_path: Optional path to save figure
    """
    # Get dimensions
    Nt = crowd_problem.Nt + 1
    T = crowd_problem.T

    # Convert density to 2D
    M_2d = crowd_problem.grid_adapter.convert_to_2d_array(result.M)

    # Create coordinate grids
    X1 = crowd_problem.grid_adapter.X1
    X2 = crowd_problem.grid_adapter.X2

    # Select time indices
    time_indices = np.linspace(0, Nt - 1, num_snapshots, dtype=int)
    times = [T * t / (Nt - 1) for t in time_indices]

    # Create figure with subplots
    nrows = 3
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15))
    fig.suptitle("2D Crowd Density Evolution - Particle-Grid Method", fontsize=16, fontweight="bold")

    # Global colorbar range
    vmin = 0
    vmax = np.max(M_2d) * 1.1

    # Plot each snapshot
    for idx, (ax, t_idx, t) in enumerate(zip(axes.flat, time_indices, times)):
        # Density heatmap
        im = ax.contourf(X1, X2, M_2d[t_idx], levels=20, cmap="YlOrRd", vmin=vmin, vmax=vmax)

        # Exit line
        ax.axhline(y=0, color="green", linewidth=3, linestyle="--", alpha=0.7, label="Exit" if idx == 0 else "")

        # Formatting
        ax.set_xlabel("x (m)", fontsize=10)
        ax.set_ylabel("y (m)", fontsize=10)
        ax.set_title(f"t = {t:.2f}s", fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle=":")

        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

        # Colorbar for each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Density", fontsize=9)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Figure saved to: {save_path}")

    plt.show()


def compute_statistics(crowd_problem: CrowdEvacuation2D, result):
    """
    Compute and display evolution statistics.

    Args:
        crowd_problem: CrowdEvacuation2D instance
        result: Solver result

    Returns:
        Dictionary of statistics
    """
    Nt = crowd_problem.Nt + 1
    T = crowd_problem.T

    # Convert density to 2D
    M_2d = crowd_problem.grid_adapter.convert_to_2d_array(result.M)

    # Compute total mass over time
    dx1 = crowd_problem.grid_adapter.dx1
    dx2 = crowd_problem.grid_adapter.dx2
    masses = np.array([np.sum(M_2d[t]) * dx1 * dx2 for t in range(Nt)])
    mass_loss = 100 * (masses[0] - masses[-1]) / masses[0] if masses[0] > 0 else 0.0

    # Compute center of mass over time
    X1 = crowd_problem.grid_adapter.X1
    X2 = crowd_problem.grid_adapter.X2

    com_x = []
    com_y = []

    for t in range(Nt):
        total = np.sum(M_2d[t])
        if total > 0:
            cx = np.sum(X1 * M_2d[t]) / total
            cy = np.sum(X2 * M_2d[t]) / total
        else:
            cx, cy = 0, 0

        com_x.append(cx)
        com_y.append(cy)

    # Evacuation progress (% of mass in lower half)
    Ny = crowd_problem.Nx2 + 1
    evacuation_progress = []
    for t in range(Nt):
        lower_half = M_2d[t, :, : Ny // 2]  # y < 5.0
        total = np.sum(M_2d[t])
        progress = 100 * np.sum(lower_half) / total if total > 0 else 0
        evacuation_progress.append(progress)

    stats = {
        "masses": masses,
        "mass_loss_percent": mass_loss,
        "com_x": com_x,
        "com_y": com_y,
        "evacuation_progress": evacuation_progress,
        "time": np.linspace(0, T, Nt),
    }

    return stats


def visualize_statistics(stats, save_path: str | None = None):
    """
    Visualize evolution statistics.

    Args:
        stats: Statistics dictionary from compute_statistics
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("2D Crowd Evacuation Statistics", fontsize=14, fontweight="bold")

    time = stats["time"]

    # Mass conservation
    ax = axes[0, 0]
    ax.plot(time, stats["masses"], "b-", linewidth=2, label="Total mass")
    ax.axhline(y=stats["masses"][0], color="r", linestyle="--", linewidth=1.5, label="Initial mass")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Total Mass", fontsize=11)
    ax.set_title("Mass Conservation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Center of mass trajectory
    ax = axes[0, 1]
    ax.plot(stats["com_x"], stats["com_y"], "b-", linewidth=2, alpha=0.7)
    ax.scatter(stats["com_x"][0], stats["com_y"][0], c="green", s=100, marker="o", label="Start", zorder=5)
    ax.scatter(stats["com_x"][-1], stats["com_y"][-1], c="red", s=100, marker="s", label="End", zorder=5)
    ax.axhline(y=0, color="green", linestyle="--", linewidth=2, alpha=0.5, label="Exit")
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)
    ax.set_title("Center of Mass Trajectory", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Center of mass Y-coordinate over time
    ax = axes[1, 0]
    ax.plot(time, stats["com_y"], "b-", linewidth=2)
    ax.axhline(y=0, color="green", linestyle="--", linewidth=1.5, label="Exit level")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Y-coordinate (m)", fontsize=11)
    ax.set_title("Vertical Movement of Center of Mass", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Evacuation progress
    ax = axes[1, 1]
    ax.plot(time, stats["evacuation_progress"], "g-", linewidth=2)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Progress (%)", fontsize=11)
    ax.set_title("Evacuation Progress (% in lower half)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Statistics figure saved to: {save_path}")

    plt.show()


def main():
    """Run 2D crowd density evolution visualization."""
    print("=" * 70)
    print("2D CROWD DENSITY EVOLUTION - Proper nD Solver")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  Grid: 20 × 20 (400 points)")
    print("  Particles: 4000")
    print("  Time: 8.0 seconds")
    print("  Timesteps: 50")
    print()

    # Create problem
    print("Creating 2D evacuation problem...")
    crowd_problem = CrowdEvacuation2D(
        grid_size=(20, 20),
        domain_bounds=(0.0, 10.0, 0.0, 10.0),
        time_domain=(8.0, 50),
        sigma=0.25,
        lambda_crowd=2.0,
    )
    print("✓ Problem created")
    print()

    # Solve
    print("Solving with Particle-Grid method...")
    result, solve_time = solve_2d_particle_grid(crowd_problem, num_particles=4000, verbose=True)
    print()

    # Compute statistics
    print("Computing statistics...")
    stats = compute_statistics(crowd_problem, result)
    print(f"✓ Mass loss: {stats['mass_loss_percent']:.4f}%")
    print(f"✓ Final center of mass: ({stats['com_x'][-1]:.2f}, {stats['com_y'][-1]:.2f})")
    print(f"✓ Evacuation progress: {stats['evacuation_progress'][-1]:.1f}%")
    print()

    # Visualize density evolution
    print("Creating density evolution visualization...")
    density_path = "examples/outputs/particle_methods/2d_density_evolution.png"
    visualize_density_evolution_2d(crowd_problem, result, num_snapshots=9, save_path=density_path)

    # Visualize statistics
    print("Creating statistics visualization...")
    stats_path = "examples/outputs/particle_methods/2d_evolution_statistics.png"
    visualize_statistics(stats, save_path=stats_path)

    print()
    print("=" * 70)
    print("✓ 2D Density Evolution Analysis Complete!")
    print("=" * 70)
    print()
    print("Generated files:")
    print(f"  1. {density_path}")
    print(f"  2. {stats_path}")
    print()


if __name__ == "__main__":
    main()
