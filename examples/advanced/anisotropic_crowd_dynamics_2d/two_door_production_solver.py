#!/usr/bin/env python3
"""
Two-Door Evacuation with Production MFG_PDE Solver

Uses the hybrid HJB-FDM + FP-Particle method for perfect mass conservation.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add MFG_PDE to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from mfg_pde import MFGComponents, MFGProblem
from mfg_pde.factory import create_standard_solver
from mfg_pde.utils.logging import get_logger

logger = get_logger(__name__)


def create_two_door_evacuation_problem(Nx=100, Nt=100, T=1.0):
    """
    Create a two-door evacuation problem using modern MFG_PDE API.

    Strategy: Map 2D room to 1D interval representing "distance along evacuation path"
    - x=0: Back of room (far from exits)
    - x=0.5: Middle (equidistant from doors)
    - x=1.0: Front (near exits/doors)

    This is a simplified 1D projection of the 2D problem that captures
    the essential evacuation dynamics.

    Args:
        Nx: Spatial grid points
        Nt: Time steps
        T: Final evacuation time

    Returns:
        MFGProblem instance with custom components
    """
    gamma = 0.1  # Congestion parameter
    door_left_proj = 0.3
    door_right_proj = 0.7

    def initial_density_func(x):
        """
        Initial crowd: concentrated in middle/back of room.

        In 1D projection, crowd starts around x=0.3-0.5 (middle of room).

        Args:
            x: Scalar spatial coordinate

        Returns:
            Unnormalized density (MFGProblem normalizes automatically)
        """
        center = 0.4  # Crowd center
        sigma_crowd = 0.15  # Spread

        # Return unnormalized Gaussian - MFGProblem will normalize
        return 100.0 * np.exp(-((x - center) ** 2) / sigma_crowd**2)

    def terminal_cost_func(x):
        """
        Terminal cost: distance to nearest exit.

        Two doors at x=0.3 and x=0.7 → dual-well structure
        """
        # Distance to left door
        dist_left = (x - door_left_proj) ** 2

        # Distance to right door
        dist_right = (x - door_right_proj) ** 2

        # Cost: distance to nearest door
        return np.minimum(dist_left, dist_right)

    def hamiltonian_func(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, **kwargs):
        """
        Hamiltonian with congestion.

        H = 0.5 * p^2 + gamma * m * p^2

        The gamma*m*p^2 term creates density-dependent friction.
        """
        # Extract momentum (for 1D: use forward/backward average)
        p_fwd = p_values.get("forward", 0.0)
        p_bwd = p_values.get("backward", 0.0)
        p = (abs(p_fwd) + abs(p_bwd)) / 2.0  # Average magnitude

        # Standard kinetic energy
        kinetic = 0.5 * p**2

        # Congestion term (movement slows in crowds)
        congestion = gamma * m_at_x * p**2

        return kinetic + congestion

    def hamiltonian_dm_func(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, **kwargs):
        """
        Derivative of Hamiltonian with respect to density.

        ∂H/∂m = gamma * p^2

        This is needed for the Fokker-Planck equation solver.
        """
        # Extract momentum
        p_fwd = p_values.get("forward", 0.0)
        p_bwd = p_values.get("backward", 0.0)
        p = (abs(p_fwd) + abs(p_bwd)) / 2.0

        # Derivative of congestion term
        return gamma * p**2

    # Create custom components
    components = MFGComponents(
        hamiltonian_func=hamiltonian_func,
        hamiltonian_dm_func=hamiltonian_dm_func,  # NOW PROVIDED!
        initial_density_func=initial_density_func,
        final_value_func=terminal_cost_func,
        description="Two-door evacuation (1D projection)",
        parameters={
            "gamma": gamma,
            "door_left": door_left_proj,
            "door_right": door_right_proj,
        },
    )

    # Create problem
    problem = MFGProblem(
        xmin=0.0,
        xmax=1.0,
        Nx=Nx,
        T=T,
        Nt=Nt,
        sigma=0.02,  # Diffusion
        coefCT=0.5,  # Control cost coefficient
        components=components,
    )

    logger.info(f"Two-door evacuation problem: {Nx} points, {Nt} timesteps, T={T}")
    logger.info(f"Door projections: {door_left_proj}, {door_right_proj}")

    return problem


def run_production_evacuation():
    """Run evacuation with production hybrid solver."""

    print("=" * 70)
    print("TWO-DOOR EVACUATION - PRODUCTION SOLVER")
    print("HJB-FDM + FP-Particle Hybrid (Mass-Conserving)")
    print("=" * 70)

    # Create problem
    print("\n1️⃣ Creating evacuation problem...")
    problem = create_two_door_evacuation_problem(
        Nx=100,  # Spatial resolution
        Nt=100,  # Time steps
        T=1.0,  # Evacuation time
    )

    # Create production solver (Tier 2: Hybrid)
    print("\n2️⃣ Creating hybrid solver (HJB-FDM + FP-Particle)...")
    solver = create_standard_solver(problem, solver_type="fixed_point")

    print(f"   Solver type: {type(solver).__name__}")
    print(f"   HJB solver: {type(solver.hjb_solver).__name__}")
    print(f"   FP solver: {type(solver.fp_solver).__name__}")

    # Solve
    print("\n3️⃣ Solving MFG system...")
    result = solver.solve()

    print("\n✅ Solution complete!")

    # Extract solution
    u = result.U  # Value function [Nt+1, Nx+1]
    m = result.M  # Density [Nt+1, Nx+1]

    # Verify mass conservation
    print("\n4️⃣ Checking mass conservation...")
    x = problem.xSpace
    mass_history = []

    for t_idx in range(problem.Nt + 1):
        total_mass = np.trapz(m[t_idx, :], x)
        mass_history.append(total_mass)

    initial_mass = mass_history[0]
    final_mass = mass_history[-1]
    max_mass_error = max(abs(m - initial_mass) for m in mass_history)

    print(f"   Initial mass: {initial_mass:.6f}")
    print(f"   Final mass: {final_mass:.6f}")
    print(f"   Max deviation: {max_mass_error:.2e}")
    print(f"   Evacuation: {(1 - final_mass / initial_mass) * 100:.2f}%")

    # Visualize
    print("\n5️⃣ Creating visualizations...")
    visualize_production_results(problem, u, m, mass_history)

    return problem, result, mass_history


def visualize_production_results(problem, u, m, mass_history):
    """Create comprehensive visualization."""

    x = problem.xSpace
    t = problem.tSpace
    X, T = np.meshgrid(x, t)

    # Create figure
    plt.figure(figsize=(16, 10))

    # 1. Density evolution (heatmap)
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.contourf(X, T, m, levels=20, cmap="YlOrRd")
    ax1.set_xlabel("Position x", fontsize=11)
    ax1.set_ylabel("Time t", fontsize=11)
    ax1.set_title("Density Evolution m(x,t)\n(Production Solver)", fontsize=12, weight="bold")
    plt.colorbar(im1, ax=ax1, label="Density")

    # Mark door locations
    ax1.axvline(0.3, color="green", linestyle="--", linewidth=2, alpha=0.7, label="Left Door")
    ax1.axvline(0.7, color="blue", linestyle="--", linewidth=2, alpha=0.7, label="Right Door")
    ax1.legend(loc="upper right", fontsize=9)

    # 2. Value function evolution
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.contourf(X, T, u, levels=20, cmap="viridis")
    ax2.set_xlabel("Position x", fontsize=11)
    ax2.set_ylabel("Time t", fontsize=11)
    ax2.set_title("Value Function u(x,t)\n(Cost-to-go)", fontsize=12, weight="bold")
    plt.colorbar(im2, ax=ax2, label="Value")

    # Mark doors
    ax2.axvline(0.3, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax2.axvline(0.7, color="red", linestyle="--", linewidth=2, alpha=0.7)

    # 3. Mass conservation check
    ax3 = plt.subplot(2, 3, 3)
    t_points = np.linspace(0, problem.T, len(mass_history))
    ax3.plot(t_points, mass_history, "b-", linewidth=2, label="Total Mass")
    ax3.axhline(mass_history[0], color="r", linestyle="--", alpha=0.5, label="Initial Mass")
    ax3.set_xlabel("Time t", fontsize=11)
    ax3.set_ylabel("Total Mass", fontsize=11)
    ax3.set_title("Mass Conservation Check\n(Should be flat)", fontsize=12, weight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Density snapshots
    ax4 = plt.subplot(2, 3, 4)
    snapshot_times = [0, problem.Nt // 3, 2 * problem.Nt // 3, problem.Nt]
    for t_idx in snapshot_times:
        t_val = t_idx * problem.T / problem.Nt
        ax4.plot(x, m[t_idx, :], label=f"t={t_val:.2f}s", linewidth=2)

    ax4.axvline(0.3, color="green", linestyle="--", alpha=0.5)
    ax4.axvline(0.7, color="blue", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Position x", fontsize=11)
    ax4.set_ylabel("Density m(x)", fontsize=11)
    ax4.set_title("Density Snapshots\n(Time Evolution)", fontsize=12, weight="bold")
    ax4.legend(loc="upper right", fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Terminal value function (dual-well)
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(x, u[0, :], "r-", linewidth=2, label="Terminal u(x,T)")
    ax5.axvline(0.3, color="green", linestyle="--", alpha=0.5, label="Left Door")
    ax5.axvline(0.7, color="blue", linestyle="--", alpha=0.5, label="Right Door")
    ax5.set_xlabel("Position x", fontsize=11)
    ax5.set_ylabel("Value u(x,T)", fontsize=11)
    ax5.set_title("Terminal Cost Function\n(Dual-well structure)", fontsize=12, weight="bold")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Mass conservation error
    ax6 = plt.subplot(2, 3, 6)
    mass_errors = [(m - mass_history[0]) / mass_history[0] * 100 for m in mass_history]
    ax6.plot(t_points, mass_errors, "g-", linewidth=2)
    ax6.set_xlabel("Time t", fontsize=11)
    ax6.set_ylabel("Mass Error (%)", fontsize=11)
    ax6.set_title("Mass Conservation Error\n(Should be ~0)", fontsize=12, weight="bold")
    ax6.grid(True, alpha=0.3)
    ax6.axhline(0, color="k", linestyle="-", alpha=0.3)

    plt.tight_layout()

    # Save
    output_file = "room_evacuation_output/production_solver_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"   ✅ Saved: {output_file}")


if __name__ == "__main__":
    import time

    start_time = time.time()

    # Run simulation
    problem, result, mass_history = run_production_evacuation()

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("✅ PRODUCTION SIMULATION COMPLETE")
    print("=" * 70)
    print(f"\n⏱️  Total time: {elapsed:.2f}s")
    print(f"📊 Mass conservation error: {result.mass_conservation_error:.2e}")
    print(f"🎯 Converged in {result.iterations} iterations")
    print("\n✨ Using HJB-FDM + FP-Particle hybrid ensures perfect mass conservation!")
    print("\nOutput: room_evacuation_output/production_solver_results.png")
