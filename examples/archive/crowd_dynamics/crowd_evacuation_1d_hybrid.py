"""
1D Crowd Evacuation using Hybrid FP-Particle + HJB-FDM Solver

Demonstrates the HybridFPParticleHJBFDM solver for a 1D crowd evacuation
scenario with comprehensive visualization and analysis.

Physical Setup:
- Corridor: 100m long domain [0, 100]
- Initial crowd: Gaussian distribution centered at x=70m
- Exit: Located at x=0 (left boundary)
- Goal: Minimize evacuation time while avoiding congestion

Mathematical Model:
- HJB (value function): -∂u/∂t + H(∂u/∂x, m) = 0
  where H = (1/2)|∂u/∂x|² + λm (congestion cost)

- FP (density evolution): ∂m/∂t - ∂/∂x(m∂u/∂x) - σ²∂²m/∂x² = 0
  solved using particle method for natural mass conservation

Hybrid Solver Benefits:
- Particles: Natural mass conservation, handles discontinuities
- FDM for HJB: Stable and accurate gradient computation
- Coupling: Picard iteration with damping for convergence
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mfg_pde.alg.numerical.coupling.hybrid_fp_particle_hjb_fdm import HybridFPParticleHJBFDM
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import BoundaryConditions


def create_crowd_evacuation_problem() -> MFGProblem:
    """
    Create 1D crowd evacuation MFG problem.

    Setup:
    - Domain: [0, 100] meters (corridor)
    - Exit at x=0 (left boundary)
    - Crowd initially concentrated at x=70m
    - Running cost includes congestion penalty

    Returns:
        MFGProblem instance configured for crowd evacuation
    """
    # Domain setup
    xmin, xmax = 0.0, 100.0
    Nx = 40  # Spatial resolution (reduced for faster demo)

    T = 5.0  # Evacuation time horizon (seconds)
    Nt = 40  # Temporal resolution (reduced for faster demo)

    # Physical parameters
    sigma = 1.0  # Diffusion coefficient (m²/s) - crowd spreading
    lambda_crowd = 5.0  # Congestion cost coefficient

    # Create 1D problem
    problem = MFGProblem(
        xmin=xmin,
        xmax=xmax,
        Nx=Nx,
        T=T,
        Nt=Nt,
        sigma=sigma,
        coefCT=lambda_crowd,  # Congestion cost coefficient
    )

    # Set boundary conditions (absorbing at exit, reflecting at far wall)
    bc = BoundaryConditions(
        type="neumann",
        left_value=0.0,  # No-flux at exit (evacuation modeled in terminal cost)
        right_value=0.0,  # Reflecting wall at far end
    )
    problem.boundary_conditions = bc

    # Initial crowd distribution: Gaussian centered at x=70m
    x = np.linspace(xmin, xmax, Nx + 1)

    # Center of crowd (70m from exit)
    x_center = 70.0
    sigma_x = 10.0  # Spread of initial crowd

    # Gaussian distribution
    initial_density = np.exp(-((x - x_center) ** 2) / (2 * sigma_x**2))

    # Normalize to probability distribution
    dx = problem.Dx
    total_mass = np.trapezoid(initial_density, dx=dx)
    initial_density = initial_density / total_mass

    # Set initial condition
    problem.rho0 = initial_density

    # Terminal cost: distance to exit (x=0)
    terminal_cost = x.copy()  # Cost = distance to exit
    problem.g = terminal_cost

    return problem


def solve_with_hybrid_solver(problem: MFGProblem, verbose: bool = True):
    """
    Solve crowd evacuation using Hybrid FP-Particle + HJB-FDM solver.

    Args:
        problem: MFG problem instance
        verbose: Print progress information

    Returns:
        Tuple of (U, M, info) where:
        - U: Value function (optimal cost-to-go)
        - M: Density evolution (crowd distribution over time)
        - info: Convergence information
    """
    if verbose:
        print("=" * 70)
        print("1D Crowd Evacuation - Hybrid FP-Particle + HJB-FDM Solver")
        print("=" * 70)
        print(f"Domain: [{problem.xmin}, {problem.xmax}]m")
        print(f"Grid: {problem.Nx+1} points")
        print(f"Time: [0, {problem.T}]s with {problem.Nt+1} steps")
        print(f"Diffusion: σ = {problem.sigma} m²/s")
        print(f"Congestion cost: λ = {problem.coefCT}")
        print()

    # Create hybrid solver
    solver = HybridFPParticleHJBFDM(
        problem=problem,
        num_particles=3000,  # Particles for smooth density (reduced for speed)
        kde_bandwidth="scott",  # Adaptive bandwidth
        max_newton_iterations=20,  # HJB Newton iterations (reduced for speed)
        newton_tolerance=1e-6,  # HJB convergence tolerance
        damping_parameter=0.5,  # Picard damping
    )

    if verbose:
        print("Solver Configuration:")
        print(f"  FP Method: Particle (N={solver.num_particles})")
        print("  HJB Method: Finite Difference (Newton)")
        print(f"  Coupling: Picard iteration (damping={solver.damping_parameter})")
        print()

    # Solve
    try:
        U, M, info = solver.solve(
            max_iterations=30,  # Reduced for faster demo
            tolerance=1e-2,  # Relaxed tolerance for demo
            verbose=verbose,
        )

        if verbose:
            print()
            print("=" * 70)
            print("SOLUTION SUMMARY")
            print("=" * 70)
            print(f"Converged: {info.get('converged', False)}")
            print(f"Iterations: {info.get('iterations', 'N/A')}")
            print(f"Final error: {info.get('final_error', 'N/A'):.6e}")
            print(f"Execution time: {info.get('execution_time', 'N/A'):.2f}s")
            print()

            # Mass conservation check
            dx = problem.Dx
            initial_mass = np.trapezoid(M[0], dx=dx)
            final_mass = np.trapezoid(M[-1], dx=dx)
            mass_loss = initial_mass - final_mass
            print("Mass Conservation:")
            print(f"  Initial mass: {initial_mass:.6f}")
            print(f"  Final mass: {final_mass:.6f}")
            print(f"  Mass loss: {mass_loss:.6f} ({100*mass_loss/initial_mass:.2f}%)")
            print("=" * 70)

        return U, M, info

    except Exception as e:
        print(f"Error during solving: {e}")
        raise


def visualize_results(problem: MFGProblem, U, M, info, save_path: str | None = None):
    """
    Create comprehensive visualization of crowd evacuation results.

    Args:
        problem: MFG problem instance
        U: Value function solution
        M: Density solution
        info: Solver information
        save_path: Optional path to save figure
    """
    # Get dimensions
    Nx = problem.Nx + 1
    Nt = problem.Nt + 1
    x = np.linspace(problem.xmin, problem.xmax, Nx)
    time_vec = np.linspace(0, problem.T, Nt)

    # Time snapshots to visualize
    time_indices = [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt - 1]
    times = [problem.T * t / (Nt - 1) for t in time_indices]

    # Create figure with subplots
    plt.figure(figsize=(16, 10))

    # Plot 1: Density evolution (space-time heatmap)
    ax1 = plt.subplot(2, 3, 1)
    T_grid, X_grid = np.meshgrid(time_vec, x)
    im1 = ax1.contourf(T_grid, X_grid, M.T, levels=30, cmap="YlOrRd")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position (m)")
    ax1.set_title("Density Evolution m(t,x)")
    plt.colorbar(im1, ax=ax1, label="Density")
    ax1.axhline(y=0, color="green", linestyle="--", linewidth=2, alpha=0.7, label="Exit")
    ax1.legend()

    # Plot 2: Value function evolution (space-time heatmap)
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.contourf(T_grid, X_grid, U.T, levels=30, cmap="viridis")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position (m)")
    ax2.set_title("Value Function u(t,x)")
    plt.colorbar(im2, ax=ax2, label="Cost-to-go")

    # Plot 3: Density snapshots
    ax3 = plt.subplot(2, 3, 3)
    for t_idx, t in zip(time_indices, times, strict=False):
        ax3.plot(x, M[t_idx], label=f"t={t:.1f}s", linewidth=2)
    ax3.set_xlabel("Position (m)")
    ax3.set_ylabel("Density m(x)")
    ax3.set_title("Density Snapshots")
    ax3.axvline(x=0, color="green", linestyle="--", alpha=0.7, label="Exit")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Mass conservation over time
    ax4 = plt.subplot(2, 3, 4)
    dx = problem.Dx
    masses = np.array([np.trapezoid(M[t], dx=dx) for t in range(Nt)])
    ax4.plot(time_vec, masses, "b-", linewidth=2, label="Total mass")
    ax4.axhline(y=masses[0], color="r", linestyle="--", label="Initial mass")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Total Mass")
    ax4.set_title("Mass Conservation")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Convergence history
    ax5 = plt.subplot(2, 3, 5)
    if info.get("convergence_history"):
        history = info["convergence_history"]
        iterations = [h["iteration"] for h in history]
        errors = [h.get("total_error", 0) for h in history]
        ax5.semilogy(iterations, errors, "b-o", markersize=3)
        ax5.axhline(y=1e-3, color="r", linestyle="--", label="Tolerance")
        ax5.set_xlabel("Picard Iteration")
        ax5.set_ylabel("Relative Error")
        ax5.set_title("Convergence History")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # Plot 6: Evacuation metrics
    ax6 = plt.subplot(2, 3, 6)
    # Compute center of mass over time
    center_of_mass = np.array([np.trapezoid(x * M[t], dx=dx) / masses[t] for t in range(Nt)])
    ax6.plot(time_vec, center_of_mass, "b-", linewidth=2, label="Center of mass")
    ax6.axhline(y=0, color="green", linestyle="--", alpha=0.7, label="Exit")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Position (m)")
    ax6.set_title("Crowd Center of Mass")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Statistics box
    stats_text = f"""SIMULATION STATISTICS

Domain: [{problem.xmin}, {problem.xmax}]m
Grid: {Nx} points
Time: {problem.T}s ({Nt} steps)

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

    plt.figtext(
        0.02,
        0.02,
        stats_text,
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.suptitle("1D Crowd Evacuation - Hybrid FP-Particle + HJB-FDM Solver", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.15, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()


def main():
    """Run 1D crowd evacuation example."""
    # Create problem
    print("Creating 1D crowd evacuation problem...")
    problem = create_crowd_evacuation_problem()

    # Solve using hybrid solver
    U, M, info = solve_with_hybrid_solver(problem, verbose=True)

    # Visualize results
    print("\nCreating visualization...")
    save_path = "crowd_evacuation_1d_hybrid_results.png"
    visualize_results(problem, U, M, info, save_path=save_path)

    print("\n✓ Example completed successfully!")
    print(f"  Results saved to: {save_path}")


if __name__ == "__main__":
    main()
