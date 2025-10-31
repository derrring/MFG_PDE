"""
Trace mass evolution at each timestep to identify when/where mass loss occurs.

This is Step 1 of the FP mass loss investigation (see FP_MASS_LOSS_INVESTIGATION.md).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.validation.test_2d_crowd_motion import CrowdMotion2D
from mfg_pde.factory import create_basic_solver


def compute_total_mass_2d(M: np.ndarray, dx: float, dy: float) -> float:
    """Compute total mass by integrating over 2D grid."""
    return float(np.sum(M) * dx * dy)


def trace_mass_evolution():
    """Track mass at each timestep during MFG solve."""
    print("\n" + "=" * 70)
    print("  MASS EVOLUTION TRACE: FP Solver Investigation")
    print("=" * 70 + "\n")

    # Use same parameters as verification
    problem = CrowdMotion2D(
        grid_resolution=8,
        time_horizon=0.4,
        num_timesteps=10,
    )

    dx, dy = problem.geometry.grid.spacing
    dV = dx * dy

    nx, ny = problem.geometry.grid.num_points
    print(f"Grid: {nx}×{ny}")
    print(f"Spacing: dx = {dx:.6f}, dy = {dy:.6f}")
    print(f"Volume element: dV = {dV:.6f}")
    print(f"Time steps: {problem.Nt}")
    print(f"Time horizon: {problem.T}")
    print(f"dt = {problem.T / (problem.Nt - 1):.6f}\n")

    # Verify initial density
    points = problem.geometry.grid.flatten()
    initial_m = problem.initial_density(points)
    m_grid = initial_m.reshape((nx, ny))
    initial_mass = compute_total_mass_2d(m_grid, dx, dy)

    print("Initial density check:")
    print(f"  sum(m) = {np.sum(m_grid):.6f}")
    print(f"  ∫∫ m dx dy = {initial_mass:.6f}")
    print(f"  Error: {abs(initial_mass - 1.0) * 100:.4f}%")

    if abs(initial_mass - 1.0) < 0.01:
        print("  ✅ Initial mass OK\n")
    else:
        print(f"  ❌ Initial mass error: {abs(initial_mass - 1.0) * 100:.2f}%\n")

    # Create solver
    print("Creating solver...")
    solver = create_basic_solver(
        problem,
        damping=0.6,
        max_iterations=20,  # Fewer iterations for faster testing
        tolerance=1e-4,
    )

    # Solve
    print("Running solve...\n")
    print("=" * 70)
    result = solver.solve()
    print("=" * 70 + "\n")

    print("Solver completed:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final error: {result.max_error:.6e}\n")

    # Trace mass at each timestep
    print("=" * 70)
    print("  MASS AT EACH TIMESTEP")
    print("=" * 70)

    masses = []
    mass_loss = []
    relative_loss = []

    for t_idx in range(result.M.shape[0]):
        M_t = result.M[t_idx]
        mass_t = compute_total_mass_2d(M_t, dx, dy)
        masses.append(mass_t)

        loss = initial_mass - mass_t
        rel_loss = loss / initial_mass * 100

        mass_loss.append(loss)
        relative_loss.append(rel_loss)

        # Print detailed info for each timestep
        status = "✅" if rel_loss < 2.0 else "⚠️" if rel_loss < 10.0 else "❌"
        dt = problem.T / (problem.Nt - 1)
        print(f"t={t_idx:2d} (t={t_idx * dt:.4f}s): mass = {mass_t:.6e}  loss = {rel_loss:6.2f}%  {status}")

    masses = np.array(masses)
    mass_loss = np.array(mass_loss)
    relative_loss = np.array(relative_loss)

    print("\n" + "=" * 70)
    print("  ANALYSIS")
    print("=" * 70)

    print(f"\nInitial mass (t=0):     {masses[0]:.6e}")
    print(f"Final mass (t={len(masses) - 1}):      {masses[-1]:.6e}")
    print(f"Total mass loss:        {mass_loss[-1]:.6e}")
    print(f"Relative mass loss:     {relative_loss[-1]:.2f}%")

    # Analyze loss pattern
    print("\nLoss pattern:")
    print(f"  Loss t=0→1:           {relative_loss[1]:.2f}%")
    if len(relative_loss) > 1:
        print(f"  Loss t=1→2:           {relative_loss[2] - relative_loss[1]:.2f}%")
    if len(relative_loss) > 2:
        print(f"  Loss per timestep:    {relative_loss[-1] / len(relative_loss):.2f}% (avg)")

    # Identify when most mass is lost
    if len(mass_loss) > 1:
        timestep_losses = np.diff(mass_loss)
        max_loss_idx = np.argmax(timestep_losses)
        print(f"  Largest loss at:      t={max_loss_idx} → t={max_loss_idx + 1}")

    # Create visualization
    print("\nCreating visualization...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot mass vs time
    timesteps = np.arange(len(masses))
    ax1.plot(timesteps, masses, "o-", linewidth=2, markersize=6, label="Mass")
    ax1.axhline(y=1.0, color="g", linestyle="--", linewidth=1, label="Target (1.0)")
    ax1.set_xlabel("Timestep", fontsize=11)
    ax1.set_ylabel("Total Mass", fontsize=11)
    ax1.set_title("Mass Evolution During FP Solve", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot relative mass loss vs time
    ax2.plot(timesteps, relative_loss, "ro-", linewidth=2, markersize=6)
    ax2.axhline(y=2.0, color="orange", linestyle="--", linewidth=1, label="Expected FDM error (2%)")
    ax2.set_xlabel("Timestep", fontsize=11)
    ax2.set_ylabel("Mass Loss (%)", fontsize=11)
    ax2.set_title("Cumulative Mass Loss", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    fig_path = output_dir / "mass_evolution_trace.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fig_path}")

    print("\n" + "=" * 70)
    print("  TRACE COMPLETE")
    print("=" * 70)

    # Summary verdict
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70 + "\n")

    if relative_loss[-1] < 2.0:
        print("✅ PASS: Mass conservation within expected FDM error (< 2%)")
    elif relative_loss[-1] < 10.0:
        print("⚠️  WARNING: Mass loss higher than expected (2-10%)")
        print("   This may indicate a solver configuration issue")
    else:
        print("❌ FAIL: Catastrophic mass loss (> 10%)")
        print("   This indicates a critical bug in the FP solver")

    print(f"\nFinal mass conservation: {100 - relative_loss[-1]:.2f}%")
    print(f"Mass loss: {relative_loss[-1]:.2f}%\n")

    return {
        "masses": masses,
        "mass_loss": mass_loss,
        "relative_loss": relative_loss,
        "result": result,
    }


if __name__ == "__main__":
    trace_mass_evolution()
