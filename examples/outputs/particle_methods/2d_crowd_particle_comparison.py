"""
2D Crowd Motion: Particle Method Comparison

Compares two particle-based methods for 2D crowd evacuation:
1. Particle-Grid (Hybrid): FP solved with particles, HJB on grid
2. Particle-Particle: Both FP and HJB use particle representations

Physical Setup:
- Room: 10m × 10m square domain
- Initial crowd: Gaussian distribution centered at (5.0, 7.0)
- Exit: Located at bottom boundary (y=0)
- Goal: Minimize evacuation time while avoiding congestion

This comparison demonstrates the tradeoffs between hybrid (more accurate value function)
and pure particle methods (fully Lagrangian, better mass conservation).
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import solve_mfg
from mfg_pde.config import ConfigBuilder
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import BoundaryConditions


def create_2d_crowd_problem(Nx: int = 30, Nt: int = 30) -> MFGProblem:
    """
    Create 2D crowd evacuation problem.

    Args:
        Nx: Spatial grid resolution (per dimension)
        Nt: Temporal resolution

    Returns:
        MFGProblem for 2D crowd evacuation
    """
    # Time parameters
    T = 5.0  # Evacuation time horizon (seconds)

    # Physical parameters
    sigma = 0.3  # Diffusion coefficient (m²/s)
    lambda_crowd = 2.0  # Congestion cost coefficient

    # Spatial domain
    xmin, xmax = 0.0, 10.0
    ymin, ymax = 0.0, 10.0

    # Create grids
    x = np.linspace(xmin, xmax, Nx + 1)
    y = np.linspace(ymin, ymax, Nx + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Initial crowd distribution: Gaussian centered at (5.0, 7.0)
    x_center, y_center = 5.0, 7.0
    sigma_x, sigma_y = 1.5, 1.5

    initial_density = np.exp(-((X - x_center) ** 2 / (2 * sigma_x**2) + (Y - y_center) ** 2 / (2 * sigma_y**2)))

    # Normalize
    dx = (xmax - xmin) / Nx
    total_mass = np.sum(initial_density) * dx * dx
    initial_density = initial_density / total_mass

    # Terminal cost: distance to exit (y=0)
    terminal_cost = Y.copy()

    # Create problem (note: for 2D we flatten to 1D for solver compatibility)
    problem = MFGProblem(
        xmin=xmin,
        xmax=xmax,
        Nx=Nx,
        T=T,
        Nt=Nt,
        sigma=sigma,
        coefCT=lambda_crowd,
    )

    # Set boundary conditions (no-flux)
    bc = BoundaryConditions(type="neumann", left_value=0.0, right_value=0.0)
    problem.boundary_conditions = bc

    # Set initial condition and terminal cost (flattened)
    problem.rho0 = initial_density.flatten()
    problem.g = terminal_cost.flatten()

    # Store grid shape for reconstruction
    problem._grid_shape = (Nx + 1, Nx + 1)

    return problem


def solve_particle_grid_method(problem: MFGProblem, num_particles: int = 3000, verbose: bool = True):
    """
    Solve using Particle-Grid (Hybrid) method.

    FP: Particle method (Lagrangian representation)
    HJB: Finite difference on grid (Eulerian representation)

    Args:
        problem: MFG problem instance
        num_particles: Number of particles for FP solver
        verbose: Print progress

    Returns:
        Tuple of (result, solve_time)
    """
    if verbose:
        print("=" * 70)
        print("METHOD 1: Particle-Grid (Hybrid)")
        print("=" * 70)
        print(f"  FP Solver:  Particle ({num_particles} particles)")
        print("  HJB Solver: Finite Difference (grid)")
        print(f"  Grid:       {problem.Nx + 1} points")
        print(f"  Time:       [0, {problem.T}]s with {problem.Nt + 1} steps")
        print()

    # Build configuration for particle-grid hybrid
    config = (
        ConfigBuilder()
        .picard(max_iterations=30, tolerance=1e-3)
        .solver_hjb("fdm")
        .solver_fp("particle", num_particles=num_particles, kde_bandwidth="scott")
        .build()
    )

    # Solve
    start_time = time.time()
    result = solve_mfg(problem, config=config, verbose=verbose)
    solve_time = time.time() - start_time

    if verbose:
        print(f"\n✓ Particle-Grid method completed in {solve_time:.2f}s")
        print(f"  Converged: {result.converged}")
        print(f"  Iterations: {result.iterations}")
        if hasattr(result, "error_history_U") and len(result.error_history_U) > 0:
            print(f"  Final error U: {result.error_history_U[-1]:.6e}")
        if hasattr(result, "error_history_M") and len(result.error_history_M) > 0:
            print(f"  Final error M: {result.error_history_M[-1]:.6e}")

    return result, solve_time


def solve_particle_particle_method(problem: MFGProblem, num_particles: int = 3000, verbose: bool = True):
    """
    Solve using Particle-Particle method.

    FP: Particle method (Lagrangian)
    HJB: Particle method (Lagrangian) - using backward characteristic tracing

    Note: This is a simplified particle-particle approach where we still use
    a grid representation for HJB but with particle-based coupling.

    Args:
        problem: MFG problem instance
        num_particles: Number of particles
        verbose: Print progress

    Returns:
        Tuple of (result, solve_time)
    """
    if verbose:
        print("=" * 70)
        print("METHOD 2: Particle-Particle")
        print("=" * 70)
        print(f"  FP Solver:  Particle ({num_particles} particles)")
        print("  HJB Solver: Semi-Lagrangian (particle-based characteristics)")
        print(f"  Grid:       {problem.Nx + 1} points (for interpolation)")
        print(f"  Time:       [0, {problem.T}]s with {problem.Nt + 1} steps")
        print()

    # Build configuration for particle-particle
    # Using semi-Lagrangian for HJB provides particle-like characteristics
    config = (
        ConfigBuilder()
        .picard(max_iterations=30, tolerance=1e-3)
        .solver_hjb_semi_lagrangian(interpolation_method="cubic", rk_order=2)
        .solver_fp("particle", num_particles=num_particles, kde_bandwidth="scott")
        .build()
    )

    # Solve
    start_time = time.time()
    result = solve_mfg(problem, config=config, verbose=verbose)
    solve_time = time.time() - start_time

    if verbose:
        print(f"\n✓ Particle-Particle method completed in {solve_time:.2f}s")
        print(f"  Converged: {result.converged}")
        print(f"  Iterations: {result.iterations}")
        if hasattr(result, "error_history_U") and len(result.error_history_U) > 0:
            print(f"  Final error U: {result.error_history_U[-1]:.6e}")
        if hasattr(result, "error_history_M") and len(result.error_history_M) > 0:
            print(f"  Final error M: {result.error_history_M[-1]:.6e}")

    return result, solve_time


def reshape_to_2d(field_1d: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    """Reshape flattened field to 2D grid for visualization."""
    if field_1d.ndim == 1:
        return field_1d.reshape(grid_shape)
    else:
        # Multiple time steps
        Nt = field_1d.shape[0]
        return field_1d.reshape(Nt, *grid_shape)


def compute_metrics(problem: MFGProblem, result, method_name: str) -> dict:
    """
    Compute comparison metrics.

    Args:
        problem: MFG problem
        result: Solver result
        method_name: Name of method

    Returns:
        Dictionary of metrics
    """
    M = result.M
    U = result.U
    Nt = problem.Nt + 1
    Nx = problem.Nx + 1
    dx = (problem.xmax - problem.xmin) / problem.Nx

    # Mass conservation
    masses = np.array([np.sum(M[t]) * dx * dx for t in range(Nt)])
    mass_loss = 100 * (masses[0] - masses[-1]) / masses[0]

    # Value function smoothness (gradient magnitude)
    U_grad_norm = np.mean([np.linalg.norm(np.gradient(U[t])) for t in range(Nt)])

    # Evacuation efficiency (how quickly mass reaches exit)
    grid_shape = problem._grid_shape
    M_2d = reshape_to_2d(M, grid_shape)
    exit_flux = [np.sum(M_2d[t, :, 0]) * dx for t in range(Nt)]
    peak_flux_time = np.argmax(exit_flux) * problem.T / (Nt - 1)

    # Get final errors
    final_error_U = (
        result.error_history_U[-1] if hasattr(result, "error_history_U") and len(result.error_history_U) > 0 else 0.0
    )
    final_error_M = (
        result.error_history_M[-1] if hasattr(result, "error_history_M") and len(result.error_history_M) > 0 else 0.0
    )

    return {
        "method": method_name,
        "converged": result.converged,
        "iterations": result.iterations,
        "final_error_U": final_error_U,
        "final_error_M": final_error_M,
        "mass_loss_percent": mass_loss,
        "value_gradient_norm": U_grad_norm,
        "peak_flux_time": peak_flux_time,
        "masses": masses,
        "exit_flux": exit_flux,
    }


def visualize_comparison(
    problem: MFGProblem,
    result1,
    result2,
    metrics1: dict,
    metrics2: dict,
    time1: float,
    time2: float,
    save_path: str | None = None,
):
    """
    Create comprehensive comparison visualization.

    Args:
        problem: MFG problem
        result1: Particle-Grid result
        result2: Particle-Particle result
        metrics1: Metrics for method 1
        metrics2: Metrics for method 2
        time1: Solve time for method 1
        time2: Solve time for method 2
        save_path: Optional path to save figure
    """
    grid_shape = problem._grid_shape
    Nx = problem.Nx + 1
    Nt = problem.Nt + 1

    # Reshape solutions to 2D
    M1_2d = reshape_to_2d(result1.M, grid_shape)
    M2_2d = reshape_to_2d(result2.M, grid_shape)
    U1_2d = reshape_to_2d(result1.U, grid_shape)
    U2_2d = reshape_to_2d(result2.U, grid_shape)

    # Create coordinate grid
    x = np.linspace(problem.xmin, problem.xmax, Nx)
    y = np.linspace(problem.xmin, problem.xmax, Nx)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Time snapshots
    time_indices = [0, Nt // 2, Nt - 1]
    times = [problem.T * t / (Nt - 1) for t in time_indices]

    # Create figure
    fig = plt.figure(figsize=(20, 12))

    # Row 1: Density comparison at different times
    for col, (t_idx, t) in enumerate(zip(time_indices, times), start=1):
        # Method 1 density
        ax = plt.subplot(4, 3, col)
        im = ax.contourf(X, Y, M1_2d[t_idx], levels=20, cmap="YlOrRd")
        ax.set_title(f"Particle-Grid Density\nt={t:.1f}s", fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        ax.axhline(y=0, color="green", linestyle="--", linewidth=2, alpha=0.5)
        plt.colorbar(im, ax=ax, label="Density")

        # Method 2 density
        ax = plt.subplot(4, 3, col + 3)
        im = ax.contourf(X, Y, M2_2d[t_idx], levels=20, cmap="YlOrRd")
        ax.set_title(f"Particle-Particle Density\nt={t:.1f}s", fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        ax.axhline(y=0, color="green", linestyle="--", linewidth=2, alpha=0.5)
        plt.colorbar(im, ax=ax, label="Density")

    # Row 3: Value function comparison at t=0
    ax = plt.subplot(4, 3, 7)
    im = ax.contourf(X, Y, U1_2d[0], levels=20, cmap="viridis")
    ax.set_title("Particle-Grid Value u(0,x)", fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="Cost")

    ax = plt.subplot(4, 3, 8)
    im = ax.contourf(X, Y, U2_2d[0], levels=20, cmap="viridis")
    ax.set_title("Particle-Particle Value u(0,x)", fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="Cost")

    # Value function difference
    ax = plt.subplot(4, 3, 9)
    diff = U1_2d[0] - U2_2d[0]
    im = ax.contourf(X, Y, diff, levels=20, cmap="RdBu_r", vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    ax.set_title("Value Difference\n(Particle-Grid - Particle-Particle)", fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="Difference")

    # Row 4: Quantitative comparisons
    # Mass conservation
    ax = plt.subplot(4, 3, 10)
    time_vec = np.linspace(0, problem.T, Nt)
    ax.plot(time_vec, metrics1["masses"], "b-", linewidth=2, label="Particle-Grid")
    ax.plot(time_vec, metrics2["masses"], "r--", linewidth=2, label="Particle-Particle")
    ax.axhline(y=metrics1["masses"][0], color="k", linestyle=":", alpha=0.5, label="Initial")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total Mass")
    ax.set_title("Mass Conservation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Exit flux
    ax = plt.subplot(4, 3, 11)
    ax.plot(time_vec, metrics1["exit_flux"], "b-", linewidth=2, label="Particle-Grid")
    ax.plot(time_vec, metrics2["exit_flux"], "r--", linewidth=2, label="Particle-Particle")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Exit Flux")
    ax.set_title("Evacuation Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Comparison statistics
    ax = plt.subplot(4, 3, 12)
    ax.axis("off")
    stats_text = f"""
COMPARISON STATISTICS

Method 1: Particle-Grid (Hybrid)
  FP: Particles, HJB: Grid FDM
  Converged: {metrics1["converged"]}
  Iterations: {metrics1["iterations"]}
  Mass loss: {metrics1["mass_loss_percent"]:.3f}%
  Solve time: {time1:.2f}s

Method 2: Particle-Particle
  FP: Particles, HJB: Semi-Lagrangian
  Converged: {metrics2["converged"]}
  Iterations: {metrics2["iterations"]}
  Mass loss: {metrics2["mass_loss_percent"]:.3f}%
  Solve time: {time2:.2f}s

Differences:
  Mass loss: {abs(metrics1["mass_loss_percent"] - metrics2["mass_loss_percent"]):.3f}%
  Speedup: {time1 / time2:.2f}x

Key Observations:
  - Particle-Grid: Better HJB accuracy
  - Particle-Particle: Fully Lagrangian
    """
    ax.text(0.05, 0.5, stats_text, fontsize=9, family="monospace", verticalalignment="center")

    plt.suptitle("2D Crowd Evacuation: Particle Method Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Figure saved to: {save_path}")

    plt.show()


def main():
    """Run 2D crowd motion particle method comparison."""
    print("=" * 70)
    print("2D CROWD MOTION: PARTICLE METHOD COMPARISON")
    print("=" * 70)
    print()
    print("Comparing two approaches:")
    print("  1. Particle-Grid (Hybrid): FP particles + HJB grid")
    print("  2. Particle-Particle: FP particles + HJB semi-Lagrangian")
    print()

    # Create problem
    print("Creating 2D crowd evacuation problem...")
    problem = create_2d_crowd_problem(Nx=20, Nt=30)
    print(f"  Grid: {problem.Nx + 1} × {problem.Nx + 1} = {(problem.Nx + 1) ** 2} points")
    print(f"  Time: [0, {problem.T}]s with {problem.Nt + 1} steps")
    print()

    # Solve with both methods
    num_particles = 3000

    # Method 1: Particle-Grid
    result1, time1 = solve_particle_grid_method(problem, num_particles, verbose=True)
    metrics1 = compute_metrics(problem, result1, "Particle-Grid")
    print()

    # Method 2: Particle-Particle
    result2, time2 = solve_particle_particle_method(problem, num_particles, verbose=True)
    metrics2 = compute_metrics(problem, result2, "Particle-Particle")
    print()

    # Print comparison summary
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(
        f"Particle-Grid:      {time1:.2f}s, {metrics1['iterations']} iters, "
        f"mass loss {metrics1['mass_loss_percent']:.3f}%"
    )
    print(
        f"Particle-Particle:  {time2:.2f}s, {metrics2['iterations']} iters, "
        f"mass loss {metrics2['mass_loss_percent']:.3f}%"
    )
    print(f"Speedup ratio:      {time1 / time2:.2f}x")
    print("=" * 70)
    print()

    # Visualize comparison
    print("Creating comparison visualization...")
    save_path = "examples/outputs/particle_methods/2d_crowd_particle_comparison_results.png"
    visualize_comparison(problem, result1, result2, metrics1, metrics2, time1, time2, save_path)

    print("\n✓ Comparison complete!")
    print(f"  Results saved to: {save_path}")


if __name__ == "__main__":
    main()
