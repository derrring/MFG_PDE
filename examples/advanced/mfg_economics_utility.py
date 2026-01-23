#!/usr/bin/env python3
"""
Demonstration: Economics Utility Maximization in Mean Field Games (Issue #623 Phase 3)

This example demonstrates the OptimizationSense abstraction, contrasting:
- Control theory: Agents MINIMIZE cost (standard MFG)
- Economics: Agents MAXIMIZE utility

Mathematical Background
-----------------------
In control theory (MINIMIZE):
    J[α] = E[∫₀ᵀ L(x, α, m) dt + g(x(T))]  (cost to minimize)
    α* = -∇U/λ  (move downhill on value function)

In economics (MAXIMIZE):
    J[α] = E[∫₀ᵀ U(x, α, m) dt + V(x(T))]  (utility to maximize)
    α* = +∇U/λ  (move uphill on value function)

The sign difference in optimal control is captured by OptimizationSense.

Physical Interpretation
-----------------------
Consider agents seeking a "desirable location" (e.g., economic opportunity):

MINIMIZE (cost perspective):
    - Value function U(x) = cost-to-go (high U = bad location)
    - Agents move toward LOW U (away from costly regions)
    - α* = -∇U (gradient descent)

MAXIMIZE (utility perspective):
    - Value function U(x) = utility (high U = good location)
    - Agents move toward HIGH U (toward valuable regions)
    - α* = +∇U (gradient ascent)

Both describe the same physical behavior when U_cost = -U_utility.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver
from mfg_pde.core.hamiltonian import (
    OptimizationSense,
    QuadraticControlCost,
)
from mfg_pde.geometry import TensorProductGrid, no_flux_bc


def create_value_function(Nt: int, Nx: int, x: np.ndarray, perspective: str) -> np.ndarray:
    """
    Create value function representing spatial preferences.

    For MINIMIZE: U(x) = (x - 0.7)² → minimum at x=0.7 (low cost location)
    For MAXIMIZE: U(x) = -(x - 0.7)² → maximum at x=0.7 (high utility location)

    Both perspectives drive agents toward x=0.7, but with opposite signs.
    """
    U = np.zeros((Nt + 1, Nx))

    for t_idx in range(Nt + 1):
        if perspective == "minimize":
            # Cost perspective: quadratic cost, minimum at x=0.7
            U[t_idx, :] = (x - 0.7) ** 2
        else:
            # Utility perspective: quadratic utility, maximum at x=0.7
            U[t_idx, :] = -((x - 0.7) ** 2)

    return U


def compute_gradient(U: np.ndarray, dx: float) -> np.ndarray:
    """Compute spatial gradient using central differences."""
    grad_U = np.zeros_like(U)
    Nt = U.shape[0]

    for t_idx in range(Nt):
        # Central differences in interior
        grad_U[t_idx, 1:-1] = (U[t_idx, 2:] - U[t_idx, :-2]) / (2 * dx)
        # One-sided at boundaries
        grad_U[t_idx, 0] = (U[t_idx, 1] - U[t_idx, 0]) / dx
        grad_U[t_idx, -1] = (U[t_idx, -1] - U[t_idx, -2]) / dx

    return grad_U


def solve_fp_with_drift(
    problem: MFGProblem,
    m0: np.ndarray,
    drift_field: np.ndarray,
    label: str,
) -> np.ndarray:
    """Solve FP equation with given drift field."""
    bc = no_flux_bc(dimension=1)
    fp_solver = FPFDMSolver(problem, boundary_conditions=bc)

    print(f"  Solving FP equation ({label})...")
    M = fp_solver.solve_fp_system(M_initial=m0, drift_field=drift_field, show_progress=False)

    return M


def analyze_results(
    x: np.ndarray,
    dx: float,
    M_minimize: np.ndarray,
    M_maximize: np.ndarray,
    target_location: float,
):
    """Analyze and compare results from both perspectives."""
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Mass conservation check
    for name, M in [("MINIMIZE", M_minimize), ("MAXIMIZE", M_maximize)]:
        mass_initial = np.trapezoid(M[0, :], dx=dx)
        mass_final = np.trapezoid(M[-1, :], dx=dx)
        print(f"\n{name} - Mass Conservation:")
        print(f"  Initial: {mass_initial:.6f}, Final: {mass_final:.6f}")
        print(f"  Deviation: {abs(mass_final - mass_initial):.2e}")

    # Center of mass evolution
    print("\nCenter of Mass Evolution:")
    for name, M in [("MINIMIZE", M_minimize), ("MAXIMIZE", M_maximize)]:
        com_initial = np.trapezoid(x * M[0, :], dx=dx) / np.trapezoid(M[0, :], dx=dx)
        com_final = np.trapezoid(x * M[-1, :], dx=dx) / np.trapezoid(M[-1, :], dx=dx)
        print(f"  {name}: {com_initial:.3f} -> {com_final:.3f} (target: {target_location})")

    # Final distribution comparison
    print("\nFinal Distribution Similarity:")
    diff = np.abs(M_minimize[-1, :] - M_maximize[-1, :])
    print(f"  Max difference: {diff.max():.6f}")
    print(f"  L2 difference: {np.sqrt(np.trapezoid(diff**2, dx=dx)):.6f}")


def plot_comparison(
    x: np.ndarray,
    m0: np.ndarray,
    U_minimize: np.ndarray,
    U_maximize: np.ndarray,
    grad_minimize: np.ndarray,
    grad_maximize: np.ndarray,
    alpha_minimize: np.ndarray,
    alpha_maximize: np.ndarray,
    M_minimize: np.ndarray,
    M_maximize: np.ndarray,
    target_location: float,
):
    """Create comprehensive comparison visualization."""
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3)

    # Row 1: Value functions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, U_minimize[0, :], "b-", linewidth=2, label="U(x) = (x-0.7)²")
    ax1.axvline(x=target_location, color="k", linestyle="--", alpha=0.5, label="Target x=0.7")
    ax1.set_xlabel("x")
    ax1.set_ylabel("Value Function U")
    ax1.set_title("MINIMIZE: Cost Function\n(Low U = desirable)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, U_maximize[0, :], "r-", linewidth=2, label="U(x) = -(x-0.7)²")
    ax2.axvline(x=target_location, color="k", linestyle="--", alpha=0.5, label="Target x=0.7")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Value Function U")
    ax2.set_title("MAXIMIZE: Utility Function\n(High U = desirable)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Row 2: Gradients
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x, grad_minimize[0, :], "b-", linewidth=2, label="∇U")
    ax3.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax3.axvline(x=target_location, color="k", linestyle="--", alpha=0.5)
    ax3.set_xlabel("x")
    ax3.set_ylabel("Gradient ∇U")
    ax3.set_title("MINIMIZE: Gradient Field")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(x, grad_maximize[0, :], "r-", linewidth=2, label="∇U")
    ax4.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax4.axvline(x=target_location, color="k", linestyle="--", alpha=0.5)
    ax4.set_xlabel("x")
    ax4.set_ylabel("Gradient ∇U")
    ax4.set_title("MAXIMIZE: Gradient Field")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Row 3: Optimal controls (THE KEY DIFFERENCE)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(x, alpha_minimize[0, :], "b-", linewidth=2.5, label="MINIMIZE: α* = -∇U/λ", alpha=0.8)
    ax5.plot(x, alpha_maximize[0, :], "r--", linewidth=2.5, label="MAXIMIZE: α* = +∇U/λ", alpha=0.8)
    ax5.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax5.axvline(x=target_location, color="k", linestyle="--", alpha=0.5, label="Target x=0.7")
    ax5.set_xlabel("x")
    ax5.set_ylabel("Optimal Control α*(x)")
    ax5.set_title(
        "Optimal Control Comparison\n"
        "Note: SAME physical direction (toward x=0.7) despite opposite formulas!\n"
        "This is because U_cost = -U_utility",
        fontsize=11,
    )
    ax5.legend(loc="upper right")
    ax5.grid(True, alpha=0.3)

    # Row 4: Density evolution heatmaps
    t_grid = np.linspace(0, 1, M_minimize.shape[0])
    extent = [x.min(), x.max(), t_grid.min(), t_grid.max()]

    ax6 = fig.add_subplot(gs[3, 0])
    im1 = ax6.imshow(M_minimize, aspect="auto", origin="lower", extent=extent, cmap="viridis", interpolation="bilinear")
    ax6.axvline(x=target_location, color="white", linestyle="--", alpha=0.7)
    ax6.set_xlabel("Space x")
    ax6.set_ylabel("Time t")
    ax6.set_title("MINIMIZE: Density Evolution m(t,x)")
    plt.colorbar(im1, ax=ax6, label="Density")

    ax7 = fig.add_subplot(gs[3, 1])
    im2 = ax7.imshow(M_maximize, aspect="auto", origin="lower", extent=extent, cmap="viridis", interpolation="bilinear")
    ax7.axvline(x=target_location, color="white", linestyle="--", alpha=0.7)
    ax7.set_xlabel("Space x")
    ax7.set_ylabel("Time t")
    ax7.set_title("MAXIMIZE: Density Evolution m(t,x)")
    plt.colorbar(im2, ax=ax7, label="Density")

    # Row 5: Final density comparison
    ax8 = fig.add_subplot(gs[4, :])
    ax8.plot(x, m0, "k--", linewidth=1.5, label="Initial m₀(x)", alpha=0.5)
    ax8.plot(x, M_minimize[-1, :], "b-", linewidth=2, label="MINIMIZE final", alpha=0.8)
    ax8.plot(x, M_maximize[-1, :], "r--", linewidth=2, label="MAXIMIZE final", alpha=0.8)
    ax8.axvline(x=target_location, color="k", linestyle="--", alpha=0.5, label="Target x=0.7")
    ax8.set_xlabel("x")
    ax8.set_ylabel("Density m(x)")
    ax8.set_title("Final Density Comparison (t = T)\nBoth perspectives yield identical physical evolution!")
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    plt.suptitle(
        "OptimizationSense Demonstration: MINIMIZE (Control Theory) vs MAXIMIZE (Economics)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    return fig


def main():
    """Run economics utility maximization demonstration."""
    print("=" * 70)
    print("ECONOMICS UTILITY MAXIMIZATION DEMONSTRATION (Issue #623 Phase 3)")
    print("=" * 70)
    print("\nComparing OptimizationSense.MINIMIZE vs OptimizationSense.MAXIMIZE")
    print("Both should produce identical physical behavior with opposite value signs.\n")

    # Setup
    Nx = 101
    Nt = 50
    T = 1.0
    sigma = 0.1
    control_cost_lambda = 1.0
    target_location = 0.7

    geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[Nx])
    problem = MFGProblem(
        geometry=geometry,
        T=T,
        Nt=Nt,
        diffusion=sigma,
        coupling_coefficient=0.0,  # No density coupling for clarity
    )

    x = np.linspace(0, 1, Nx)
    dx = x[1] - x[0]

    # Initial density: Gaussian at x=0.2 (away from target)
    print("Setting up initial density (Gaussian at x=0.2)...")
    m0 = np.exp(-100 * (x - 0.2) ** 2)
    m0 = m0 / np.trapezoid(m0, dx=dx)

    # Create Hamiltonian abstractions
    print("\nCreating Hamiltonian abstractions:")
    cost_minimize = QuadraticControlCost(
        sense=OptimizationSense.MINIMIZE,
        control_cost=control_cost_lambda,
    )
    cost_maximize = QuadraticControlCost(
        sense=OptimizationSense.MAXIMIZE,
        control_cost=control_cost_lambda,
    )
    print("  MINIMIZE: α* = -∇U/λ (gradient descent on cost)")
    print("  MAXIMIZE: α* = +∇U/λ (gradient ascent on utility)")

    # Create value functions (with appropriate signs for each perspective)
    print("\nCreating value functions...")
    U_minimize = create_value_function(Nt, Nx, x, "minimize")
    U_maximize = create_value_function(Nt, Nx, x, "maximize")
    print(f"  MINIMIZE: U(x) = (x-{target_location})² (cost, minimum at target)")
    print(f"  MAXIMIZE: U(x) = -(x-{target_location})² (utility, maximum at target)")

    # Compute gradients
    grad_minimize = compute_gradient(U_minimize, dx)
    grad_maximize = compute_gradient(U_maximize, dx)

    # Compute optimal controls using Hamiltonian abstraction
    print("\nComputing optimal controls via ControlCostBase.optimal_control():")
    alpha_minimize = cost_minimize.optimal_control(grad_minimize)
    alpha_maximize = cost_maximize.optimal_control(grad_maximize)

    # Verify they produce same physical direction
    print(f"  MINIMIZE α* at x=0.3: {alpha_minimize[0, 30]:.4f} (should be positive, toward target)")
    print(f"  MAXIMIZE α* at x=0.3: {alpha_maximize[0, 30]:.4f} (should be positive, toward target)")

    # Solve FP equations
    print("\nSolving Fokker-Planck equations:")
    M_minimize = solve_fp_with_drift(problem, m0, alpha_minimize, "MINIMIZE")
    M_maximize = solve_fp_with_drift(problem, m0, alpha_maximize, "MAXIMIZE")

    # Analysis
    analyze_results(x, dx, M_minimize, M_maximize, target_location)

    # Visualization
    print("\nGenerating comparison plot...")
    fig = plot_comparison(
        x,
        m0,
        U_minimize,
        U_maximize,
        grad_minimize,
        grad_maximize,
        alpha_minimize,
        alpha_maximize,
        M_minimize,
        M_maximize,
        target_location,
    )

    output_file = "mfg_economics_utility_comparison.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_file}")

    plt.show()

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. OptimizationSense encodes the fundamental perspective difference:
   - MINIMIZE: Control theory - agents minimize cost (α* = -∇U/λ)
   - MAXIMIZE: Economics - agents maximize utility (α* = +∇U/λ)

2. Physical equivalence: When U_cost = -U_utility, both produce
   identical physical behavior (agents move toward the same location).

3. The Hamiltonian abstraction (ControlCostBase) handles the sign
   convention automatically via the 'sense' parameter.

4. Use MINIMIZE for:
   - Crowd evacuation (minimize travel cost)
   - Traffic flow (minimize congestion cost)
   - Robot swarms (minimize energy expenditure)

5. Use MAXIMIZE for:
   - Economic models (maximize profit/utility)
   - Resource seeking (maximize resource collection)
   - Opinion dynamics (maximize social utility)

6. The abstraction in mfg_pde/core/hamiltonian.py provides:
   - QuadraticControlCost: Standard L = ½λ|α|²
   - L1ControlCost: Bang-bang control L = λ|α|
   - BoundedControlCost: Constrained control |α| ≤ α_max
    """)


if __name__ == "__main__":
    main()
