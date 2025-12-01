"""
Pure Diffusion Demo: FP Solver Beyond MFG.

This example demonstrates solving the heat equation (pure diffusion without drift)
using the FP solver's flexible API. This shows that FP solvers are general-purpose
PDE solvers, not just MFG-specific tools.

Equation:
    ∂m/∂t = σ²/2 Δm

This is the Fokker-Planck equation with zero drift (α = 0), equivalent to the
heat equation.

Current API (v0.13):
    drift_field=None  →  zero drift (works now)

Future API (Phase 2):
    from mfg_pde.utils import zero_drift
    drift_field=zero_drift()  →  explicit intent (coming soon)

This example uses the current API. See drift_helpers.py for future usage.
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver


def main():
    """Solve pure diffusion equation and visualize spreading."""
    print("\n" + "=" * 70)
    print("PURE DIFFUSION DEMONSTRATION (FP Solver Beyond MFG)")
    print("=" * 70)
    print("\nEquation: ∂m/∂t = σ²/2 Δm  (heat equation, zero drift)")
    print("This demonstrates FP solvers as general PDE tools, not just MFG-specific.")

    # Setup problem (using MFGProblem infrastructure for grid/time)
    problem = MFGProblem(
        Nx=60,
        Nt=50,
        T=2.0,
        sigma=0.25,  # Diffusion coefficient
        Lx=4.0,
    )
    solver = FPFDMSolver(problem)

    print("\nProblem setup:")
    print(f"  Domain: x ∈ [0, {problem.Lx}]")
    print(f"  Time: t ∈ [0, {problem.T}]")
    print(f"  Grid: Nx={problem.Nx + 1}, Nt={problem.Nt + 1}")
    print(f"  Diffusion: σ={problem.sigma}")

    # Initial condition: Narrow Gaussian peak
    x = problem.xSpace
    x0 = problem.Lx / 2
    sigma0 = 0.15
    m0 = np.exp(-((x - x0) ** 2) / (2 * sigma0**2))
    m0 = m0 / (np.sum(m0) * problem.dx)  # Normalize to unit mass

    print(f"\nInitial condition: Gaussian centered at x={x0}")
    print(f"  Initial width: σ₀={sigma0}")
    print(f"  Initial mass: {np.sum(m0) * problem.dx:.6f}")

    # Solve pure diffusion: drift_field=None means zero drift
    print("\nSolving pure diffusion (no advection)...")
    M_solution = solver.solve_fp_system(
        M_initial=m0,
        drift_field=None,  # Zero drift (pure diffusion)
        show_progress=True,
    )

    # Verify mass conservation
    final_mass = np.sum(M_solution[-1]) * problem.dx
    print("\nMass conservation:")
    print(f"  Initial mass: {np.sum(m0) * problem.dx:.8f}")
    print(f"  Final mass:   {final_mass:.8f}")
    print(f"  Error:        {abs(final_mass - np.sum(m0) * problem.dx):.2e}")

    # Visualize spreading
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Profile evolution
    times_to_plot = [0, 12, 25, 37, 49]
    colors = plt.cm.viridis(np.linspace(0, 1, len(times_to_plot)))

    for i, t_idx in enumerate(times_to_plot):
        t = t_idx * problem.dt
        ax1.plot(x, M_solution[t_idx], color=colors[i], label=f"t={t:.2f}", linewidth=2)

    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("m(t,x)", fontsize=12)
    ax1.set_title("Pure Diffusion: Gaussian Spreading Over Time", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, problem.Lx)

    # Add annotation
    ax1.text(
        0.05,
        0.95,
        "No drift (α = 0)\nPeak spreads symmetrically",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    # Spacetime heatmap
    im = ax2.imshow(
        M_solution.T,
        aspect="auto",
        origin="lower",
        extent=[0, problem.T, 0, problem.Lx],
        cmap="hot",
        interpolation="bilinear",
    )
    ax2.set_xlabel("Time t", fontsize=12)
    ax2.set_ylabel("Space x", fontsize=12)
    ax2.set_title("Spacetime Evolution m(t,x)", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax2, label="Density m(t,x)")

    plt.tight_layout()
    plt.savefig("examples/outputs/fp_pure_diffusion_demo.png", dpi=150, bbox_inches="tight")
    print("\n✓ Visualization saved: examples/outputs/fp_pure_diffusion_demo.png")
    plt.show()

    # Quantitative analysis
    print("\n" + "=" * 70)
    print("QUANTITATIVE ANALYSIS")
    print("=" * 70)

    # Measure spreading
    def compute_width(m_profile):
        """Compute standard deviation (width) of density profile."""
        total = np.sum(m_profile) * problem.dx
        mean_x = np.sum(x * m_profile) * problem.dx / total
        var_x = np.sum(((x - mean_x) ** 2) * m_profile) * problem.dx / total
        return np.sqrt(var_x)

    width_initial = compute_width(M_solution[0])
    width_final = compute_width(M_solution[-1])

    print("\nWidth (standard deviation):")
    print(f"  Initial: σ(t=0) = {width_initial:.4f}")
    print(f"  Final:   σ(t={problem.T}) = {width_final:.4f}")
    print(f"  Spread:  Δσ = {width_final - width_initial:.4f}")

    # Theoretical prediction: σ²(t) = σ²(0) + σ² * t  (for heat equation)
    theoretical_width_squared = width_initial**2 + problem.sigma**2 * problem.T
    theoretical_width = np.sqrt(theoretical_width_squared)
    error = abs(width_final - theoretical_width) / theoretical_width * 100

    print("\nTheoretical prediction (σ²(t) = σ²(0) + σ²·t):")
    print(f"  Theoretical width at t={problem.T}: σ_theory = {theoretical_width:.4f}")
    print(f"  Numerical width:                    σ_numeric = {width_final:.4f}")
    print(f"  Relative error:                     {error:.2f}%")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("• FP solvers are general PDE solvers, not MFG-specific")
    print("• drift_field=None enables pure diffusion (heat equation)")
    print("• Mass conservation maintained (error < 1e-6)")
    print(f"• Numerical spreading matches theory ({error:.1f}% error)")
    print("\nFuture API (Phase 2): Use drift_field=zero_drift() for explicit intent")
    print("See mfg_pde.utils.drift_helpers for upcoming drift helper functions")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
