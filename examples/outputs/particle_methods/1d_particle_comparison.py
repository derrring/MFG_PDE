"""
1D Crowd Motion: Particle Method Comparison

Compares two particle-based methods for 1D crowd evacuation:
1. Particle-Grid (Hybrid): FP solved with particles, HJB on grid
2. Particle-FDM: Both FP (particles) and HJB (FDM) on grid

Physical Setup:
- Domain: [0, 10] meters (1D corridor)
- Initial crowd: Gaussian distribution centered at x=7.0
- Exit: Located at x=0 (left boundary)
- Goal: Minimize evacuation time while avoiding congestion
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


def create_1d_crowd_problem(Nx: int = 50, Nt: int = 40) -> MFGProblem:
    """
    Create 1D crowd evacuation problem.

    Args:
        Nx: Spatial grid resolution
        Nt: Temporal resolution

    Returns:
        MFGProblem for 1D crowd evacuation
    """
    # Time parameters
    T = 5.0  # Evacuation time horizon (seconds)

    # Physical parameters
    sigma = 0.2  # Diffusion coefficient (m²/s)
    lambda_crowd = 1.5  # Congestion cost coefficient

    # Spatial domain
    xmin, xmax = 0.0, 10.0

    # Create grid
    x = np.linspace(xmin, xmax, Nx + 1)

    # Initial crowd distribution: Gaussian centered at x=7.0
    x_center = 7.0
    sigma_x = 1.0

    initial_density = np.exp(-((x - x_center) ** 2) / (2 * sigma_x**2))

    # Normalize
    dx = (xmax - xmin) / Nx
    total_mass = np.sum(initial_density) * dx
    initial_density = initial_density / total_mass

    # Terminal cost: distance to exit (x=0)
    terminal_cost = x.copy()

    # Create problem
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

    # Set initial condition and terminal cost
    problem.rho0 = initial_density
    problem.g = terminal_cost

    return problem


def solve_particle_grid_method(problem: MFGProblem, num_particles: int = 5000, verbose: bool = True):
    """
    Solve using Particle-Grid (Hybrid) method.

    FP: Particle method (Lagrangian representation)
    HJB: Finite difference on grid (Eulerian representation)
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
        .picard(max_iterations=50, tolerance=1e-3)
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

    return result, solve_time


def solve_fdm_fdm_method(problem: MFGProblem, verbose: bool = True):
    """
    Solve using FDM-FDM method (both solvers on grid).

    FP: Finite difference (Eulerian)
    HJB: Finite difference (Eulerian)
    """
    if verbose:
        print("=" * 70)
        print("METHOD 2: FDM-FDM (Both on Grid)")
        print("=" * 70)
        print("  FP Solver:  Finite Difference (grid)")
        print("  HJB Solver: Finite Difference (grid)")
        print(f"  Grid:       {problem.Nx + 1} points")
        print(f"  Time:       [0, {problem.T}]s with {problem.Nt + 1} steps")
        print()

    # Build configuration for FDM-FDM
    config = ConfigBuilder().picard(max_iterations=50, tolerance=1e-3).solver_hjb("fdm").solver_fp("fdm").build()

    # Solve
    start_time = time.time()
    result = solve_mfg(problem, config=config, verbose=verbose)
    solve_time = time.time() - start_time

    if verbose:
        print(f"\n✓ FDM-FDM method completed in {solve_time:.2f}s")
        print(f"  Converged: {result.converged}")
        print(f"  Iterations: {result.iterations}")

    return result, solve_time


def compute_metrics(problem: MFGProblem, result, method_name: str) -> dict:
    """Compute comparison metrics."""
    M = result.M
    U = result.U
    Nt = problem.Nt + 1
    Nx = problem.Nx + 1
    dx = (problem.xmax - problem.xmin) / problem.Nx

    # Mass conservation
    masses = np.array([np.sum(M[t]) * dx for t in range(Nt)])
    mass_loss = 100 * (masses[0] - masses[-1]) / masses[0]

    # Value function smoothness (gradient magnitude)
    U_grad_norm = np.mean([np.linalg.norm(np.gradient(U[t])) for t in range(Nt)])

    # Evacuation efficiency (exit flux at x=0)
    exit_flux = [M[t, 0] for t in range(Nt)]
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
    """Create comprehensive comparison visualization."""
    Nx = problem.Nx + 1
    Nt = problem.Nt + 1
    x = np.linspace(problem.xmin, problem.xmax, Nx)
    time_vec = np.linspace(0, problem.T, Nt)

    # Time snapshots
    time_indices = [0, Nt // 2, Nt - 1]
    times = [problem.T * t / (Nt - 1) for t in time_indices]

    # Create figure
    fig = plt.figure(figsize=(16, 10))

    # Row 1: Density snapshots
    for col, (t_idx, t) in enumerate(zip(time_indices, times), start=1):
        ax = plt.subplot(3, 3, col)
        ax.plot(x, result1.M[t_idx], "b-", linewidth=2, label="Particle-Grid")
        ax.plot(x, result2.M[t_idx], "r--", linewidth=2, label="FDM-FDM")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("Density m(t,x)")
        ax.set_title(f"Density at t={t:.1f}s", fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color="green", linestyle=":", alpha=0.5)

    # Row 2: Value function snapshots
    for col, (t_idx, t) in enumerate(zip(time_indices, times), start=4):
        ax = plt.subplot(3, 3, col)
        ax.plot(x, result1.U[t_idx], "b-", linewidth=2, label="Particle-Grid")
        ax.plot(x, result2.U[t_idx], "r--", linewidth=2, label="FDM-FDM")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("Value u(t,x)")
        ax.set_title(f"Value at t={t:.1f}s", fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Row 3: Quantitative comparisons
    # Mass conservation
    ax = plt.subplot(3, 3, 7)
    ax.plot(time_vec, metrics1["masses"], "b-", linewidth=2, label="Particle-Grid")
    ax.plot(time_vec, metrics2["masses"], "r--", linewidth=2, label="FDM-FDM")
    ax.axhline(y=metrics1["masses"][0], color="k", linestyle=":", alpha=0.5, label="Initial")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total Mass")
    ax.set_title("Mass Conservation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Exit flux
    ax = plt.subplot(3, 3, 8)
    ax.plot(time_vec, metrics1["exit_flux"], "b-", linewidth=2, label="Particle-Grid")
    ax.plot(time_vec, metrics2["exit_flux"], "r--", linewidth=2, label="FDM-FDM")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Exit Flux")
    ax.set_title("Evacuation Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Comparison statistics
    ax = plt.subplot(3, 3, 9)
    ax.axis("off")
    stats_text = f"""
COMPARISON STATISTICS

Method 1: Particle-Grid (Hybrid)
  FP: Particles, HJB: Grid FDM
  Converged: {metrics1["converged"]}
  Iterations: {metrics1["iterations"]}
  Mass loss: {metrics1["mass_loss_percent"]:.3f}%
  Solve time: {time1:.2f}s

Method 2: FDM-FDM
  FP: Grid FDM, HJB: Grid FDM
  Converged: {metrics2["converged"]}
  Iterations: {metrics2["iterations"]}
  Mass loss: {metrics2["mass_loss_percent"]:.3f}%
  Solve time: {time2:.2f}s

Differences:
  Mass loss: {abs(metrics1["mass_loss_percent"] - metrics2["mass_loss_percent"]):.3f}%
  Speedup: {time2 / time1:.2f}x (Particle vs FDM)
    """
    ax.text(0.05, 0.5, stats_text, fontsize=9, family="monospace", verticalalignment="center")

    plt.suptitle("1D Crowd Evacuation: Particle vs FDM Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Figure saved to: {save_path}")

    plt.show()


def main():
    """Run 1D crowd motion particle method comparison."""
    print("=" * 70)
    print("1D CROWD MOTION: PARTICLE METHOD COMPARISON")
    print("=" * 70)
    print()
    print("Comparing two approaches:")
    print("  1. Particle-Grid: FP particles + HJB grid")
    print("  2. FDM-FDM: FP grid + HJB grid")
    print()

    # Create problem
    print("Creating 1D crowd evacuation problem...")
    problem = create_1d_crowd_problem(Nx=50, Nt=40)
    print(f"  Grid: {problem.Nx + 1} points")
    print(f"  Time: [0, {problem.T}]s with {problem.Nt + 1} steps")
    print()

    # Solve with both methods
    num_particles = 5000

    # Method 1: Particle-Grid
    result1, time1 = solve_particle_grid_method(problem, num_particles, verbose=True)
    metrics1 = compute_metrics(problem, result1, "Particle-Grid")
    print()

    # Method 2: FDM-FDM
    result2, time2 = solve_fdm_fdm_method(problem, verbose=True)
    metrics2 = compute_metrics(problem, result2, "FDM-FDM")
    print()

    # Print comparison summary
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(
        f"Particle-Grid:  {time1:.2f}s, {metrics1['iterations']} iters, mass loss {metrics1['mass_loss_percent']:.3f}%"
    )
    print(
        f"FDM-FDM:        {time2:.2f}s, {metrics2['iterations']} iters, mass loss {metrics2['mass_loss_percent']:.3f}%"
    )
    print(f"Speedup ratio:  {time2 / time1:.2f}x")
    print("=" * 70)
    print()

    # Visualize comparison
    print("Creating comparison visualization...")
    save_path = "examples/outputs/particle_methods/1d_crowd_particle_comparison_results.png"
    visualize_comparison(problem, result1, result2, metrics1, metrics2, time1, time2, save_path)

    print("\n✓ Comparison complete!")
    print(f"  Results saved to: {save_path}")


if __name__ == "__main__":
    main()
