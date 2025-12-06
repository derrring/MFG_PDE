"""
Tutorial 02: Custom Hamiltonian

Learn how to define your own MFG problem with a custom Hamiltonian.

What you'll learn:
- How to create a custom MFGProblem subclass
- How to define your own Hamiltonian H(x, p, m)
- How to specify terminal costs and initial density
- How to use the modern Components API (optional)

Mathematical Problem:
    A crowd evacuation problem with congestion:
    - Hamiltonian: H(x, p, m) = (1/2)|p|^2 + λ·m·|p|^2 (congestion slows movement)
    - Terminal cost: Distance to exit at x=1
    - Initial density: Uniform distribution (crowd at rest)
"""

import numpy as np

from mfg_pde import MFGProblem

# ==============================================================================
# Step 1: Define Custom Problem
# ==============================================================================

print("=" * 70)
print("TUTORIAL 02: Custom Hamiltonian")
print("=" * 70)
print()


class CrowdEvacuationMFG(MFGProblem):
    """
    Custom MFG problem: Crowd evacuation with congestion.

    Agents want to reach the exit (x=1) but congestion slows them down.
    The Hamiltonian captures this trade-off:
    - |p|^2 term: Movement cost (agents want to minimize effort)
    - λ·m·|p|^2 term: Congestion penalty (movement is harder in crowded areas)
    """

    def __init__(self, congestion_strength=2.0):
        """
        Initialize crowd evacuation problem.

        Args:
            congestion_strength: λ parameter controlling congestion effect
        """
        # Call parent constructor with domain and discretization
        super().__init__(
            xmin=0.0,  # Left boundary
            xmax=1.0,  # Exit location
            Nx=60,  # Spatial resolution
            T=1.0,  # Time horizon
            Nt=50,  # Time discretization
            sigma=0.1,  # Diffusion (agent randomness)
        )

        self.congestion_strength = congestion_strength

    def hamiltonian(self, x, p, m, t):
        """
        Hamiltonian H(x, p, m, t).

        Args:
            x: Spatial position(s)
            p: Momentum (gradient of value function)
            m: Density
            t: Time

        Returns:
            Hamiltonian value
        """
        # Standard LQ term: (1/2)|p|^2
        standard_cost = 0.5 * p**2

        # Congestion term: λ·m·|p|^2
        # Movement is harder in crowded areas (high m)
        congestion_cost = self.congestion_strength * m * p**2

        return standard_cost + congestion_cost

    def hamiltonian_dm(self, x, p, m, t):
        """
        Derivative of Hamiltonian w.r.t. density: ∂H/∂m.

        This appears in the Fokker-Planck equation drift term.

        Returns:
            λ·|p|^2 (congestion creates drift away from crowds)
        """
        return self.congestion_strength * p**2

    def terminal_cost(self, x):
        """
        Terminal cost g(x): Distance to exit.

        Agents want to be close to the exit (x=1) at final time.
        """
        return (x - 1.0) ** 2

    def initial_density(self, x):
        """
        Initial density m₀(x): Uniform crowd distribution.
        """
        # Uniform distribution on [0.1, 0.9] (avoid boundaries)
        density = np.where((x >= 0.1) & (x <= 0.9), 1.0, 0.0)
        # Normalize to integrate to 1
        return density / np.trapz(density, x)


# ==============================================================================
# Step 2: Create and Solve Problem
# ==============================================================================

print("Creating custom problem...")
problem = CrowdEvacuationMFG(congestion_strength=2.0)

print(f"  Domain: [{problem.xmin}, {problem.xmax}]")
print(f"  Congestion strength: λ = {problem.congestion_strength}")
print()

print("Solving MFG system...")
result = problem.solve(verbose=True)

print()
print(f"Converged: {result.converged} (iterations: {result.iterations})")
print()

# ==============================================================================
# Step 3: Analyze Congestion Effect
# ==============================================================================

print("=" * 70)
print("CONGESTION ANALYSIS")
print("=" * 70)
print()

# Compare with no-congestion case
print("Solving baseline (no congestion)...")
baseline = CrowdEvacuationMFG(congestion_strength=0.0)
result_baseline = baseline.solve(verbose=False)

# Measure evacuation efficiency
# How much mass reaches the exit (x > 0.9) by final time?
exit_mass_congested = np.sum(result.M[-1, problem.xSpace > 0.9]) * problem.dx
exit_mass_baseline = np.sum(result_baseline.M[-1, baseline.xSpace > 0.9]) * baseline.dx

print(f"Mass at exit (with congestion):    {exit_mass_congested:.3f}")
print(f"Mass at exit (without congestion): {exit_mass_baseline:.3f}")
print(f"Evacuation slowdown: {(1 - exit_mass_congested / exit_mass_baseline) * 100:.1f}%")
print()

# ==============================================================================
# Step 4: Visualize
# ==============================================================================

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Density evolution (with congestion)
    X, T = np.meshgrid(problem.xSpace, problem.tSpace)
    contour = axes[0, 0].contourf(X, T, result.M, levels=20, cmap="viridis")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("t")
    axes[0, 0].set_title(f"Density Evolution (λ={problem.congestion_strength})")
    plt.colorbar(contour, ax=axes[0, 0])

    # Plot 2: Density evolution (no congestion)
    contour2 = axes[0, 1].contourf(X, T, result_baseline.M, levels=20, cmap="viridis")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("t")
    axes[0, 1].set_title("Density Evolution (λ=0, baseline)")
    plt.colorbar(contour2, ax=axes[0, 1])

    # Plot 3: Snapshots at different times
    times_to_plot = [0, len(problem.tSpace) // 2, -1]
    for t_idx in times_to_plot:
        axes[1, 0].plot(problem.xSpace, result.M[t_idx, :], label=f"t={problem.tSpace[t_idx]:.2f}", linewidth=2)
    axes[1, 0].axvline(x=0.9, color="red", linestyle="--", alpha=0.5, label="Exit zone")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("m(t, x)")
    axes[1, 0].set_title("Density Snapshots (with congestion)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Value function at t=0
    axes[1, 1].plot(problem.xSpace, result.U[0, :], label="With congestion", linewidth=2)
    axes[1, 1].plot(baseline.xSpace, result_baseline.U[0, :], label="Baseline", linewidth=2, linestyle="--")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("u(0, x)")
    axes[1, 1].set_title("Value Function at t=0")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("examples/outputs/tutorials/02_custom_hamiltonian.png", dpi=150, bbox_inches="tight")
    print("Saved plot to: examples/outputs/tutorials/02_custom_hamiltonian.png")
    print()

except ImportError:
    print("Matplotlib not available - skipping visualization")
    print()

# ==============================================================================
# Summary
# ==============================================================================

print("=" * 70)
print("TUTORIAL COMPLETE")
print("=" * 70)
print()
print("What you learned:")
print("  1. How to create a custom MFGProblem subclass")
print("  2. How to define hamiltonian(), hamiltonian_dm(), terminal_cost(), initial_density()")
print("  3. How to model congestion effects in the Hamiltonian")
print("  4. How to compare different parameter settings")
print()
print("Next: Tutorial 03 - 2D Geometry")
print("=" * 70)
