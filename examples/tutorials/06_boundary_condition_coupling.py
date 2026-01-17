"""
Tutorial: Adjoint-Consistent Boundary Conditions for MFG Systems

This tutorial demonstrates Issue #574 solution: using adjoint-consistent
boundary conditions to improve equilibrium consistency when stall points
occur at domain boundaries.

Topics covered:
- Boundary condition consistency in coupled HJB-FP systems
- The bc_mode parameter in HJBFDMSolver
- When to use adjoint-consistent vs standard BC
- Convergence improvement demonstration

Mathematical Background:
-----------------------
At reflecting boundaries with Neumann BC, the standard condition ∂U/∂n = 0
may be inconsistent with the equilibrium (Boltzmann-Gibbs) solution when the
stall point occurs at the boundary.

The adjoint-consistent BC couples HJB to the FP density gradient:
    ∂U/∂n = -σ²/2 · ∂ln(m)/∂n

This maintains the zero-flux condition J·n = 0 at equilibrium.

Implementation:
--------------
Uses the existing Robin BC framework (BCType.ROBIN with alpha=0, beta=1) for
dimension-agnostic support. The solver automatically creates proper
BoundaryConditions objects when bc_mode="adjoint_consistent".

Reference: docs/development/TOWEL_ON_BEACH_1D_PROTOCOL.md
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfg_pde.geometry.boundary import neumann_bc


def create_boundary_stall_problem():
    """
    Create 1D MFG problem with stall point at boundary.

    This configuration maximizes the BC consistency issue:
    - Stall point at x=0 (left boundary)
    - Target at x=1 (right boundary)
    - Reflecting boundaries (Neumann BC)

    Returns:
        Configured MFGProblem instance
    """
    # Problem parameters
    Nx = 50  # Spatial grid points
    Nt = 50  # Temporal grid points
    T = 1.0  # Time horizon
    sigma = 0.2  # Diffusion coefficient

    # Create problem with reflecting boundaries
    problem = MFGProblem(
        domain=[0.0, 1.0],
        Nx=Nx,
        Nt=Nt,
        T=T,
        sigma=sigma,
        boundary_conditions=neumann_bc(dimension=1),
    )

    return problem


def solve_with_bc_mode(problem, bc_mode: str, max_iterations: int = 30):
    """
    Solve MFG problem with specified boundary condition mode.

    Args:
        problem: MFG problem instance
        bc_mode: "standard" or "adjoint_consistent"
        max_iterations: Maximum Picard iterations

    Returns:
        SolverResult with solution and convergence info
    """
    print(f"\nSolving with bc_mode='{bc_mode}'...")

    # Create HJB solver with specified BC mode
    hjb_solver = HJBFDMSolver(problem, bc_mode=bc_mode)

    # Create FP solver (unchanged)
    fp_solver = FPFDMSolver(problem)

    # Create Picard iterator for coupling
    iterator = FixedPointIterator(
        problem=problem,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
    )

    # Solve coupled system
    result = iterator.solve(
        max_iterations=max_iterations,
        tolerance=1e-6,
        verbose=True,
    )

    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final error (U): {result.final_error_U:.6e}")
    print(f"  Final error (M): {result.final_error_M:.6e}")

    return result


def compare_bc_modes():
    """
    Main tutorial: Compare standard vs adjoint-consistent BC modes.

    Demonstrates:
    1. Problem setup with boundary stall
    2. Solving with both BC modes
    3. Convergence comparison
    4. Visualization of differences
    """
    print("=" * 70)
    print("Tutorial: Adjoint-Consistent Boundary Conditions")
    print("=" * 70)

    # Step 1: Create problem
    print("\n[Step 1] Creating boundary stall problem...")
    problem = create_boundary_stall_problem()

    print("  Domain: [0, 1]")
    print(f"  Grid points: {problem.geometry.get_grid_shape()[0]}")
    print(f"  Time steps: {problem.Nt}")
    print(f"  Diffusion: σ = {problem.sigma}")
    print("  BC type: Neumann (reflecting)")

    # Step 2: Solve with standard BC
    print("\n[Step 2] Solving with STANDARD Neumann BC...")
    print("  Using: bc_mode='standard' (classical ∂U/∂n = 0)")
    result_std = solve_with_bc_mode(problem, "standard", max_iterations=30)

    # Step 3: Solve with adjoint-consistent BC
    print("\n[Step 3] Solving with ADJOINT-CONSISTENT BC...")
    print("  Using: bc_mode='adjoint_consistent' (couples to ∂ln(m)/∂n)")
    result_ac = solve_with_bc_mode(problem, "adjoint_consistent", max_iterations=30)

    # Step 4: Compare results
    print("\n[Step 4] Comparison of BC modes")
    print("=" * 70)

    print("\nConvergence Comparison:")
    print("  Standard BC:")
    print(f"    Iterations: {result_std.iterations}")
    print(f"    Max error: {result_std.max_error:.6e}")

    print("  Adjoint-Consistent BC:")
    print(f"    Iterations: {result_ac.iterations}")
    print(f"    Max error: {result_ac.max_error:.6e}")

    if result_ac.max_error > 0:
        improvement = result_std.max_error / result_ac.max_error
        print(f"\n  Convergence improvement: {improvement:.2f}x")

    # Solution differences
    U_diff = np.linalg.norm(result_std.U - result_ac.U)
    M_diff = np.linalg.norm(result_std.M - result_ac.M)

    print("\nSolution Differences:")
    print(f"  ||U_standard - U_ac||: {U_diff:.6e}")
    print(f"  ||M_standard - M_ac||: {M_diff:.6e}")

    # Step 5: Visualize
    print("\n[Step 5] Generating comparison plots...")
    visualize_comparison(problem, result_std, result_ac)

    # Summary
    print("\n" + "=" * 70)
    print("TUTORIAL SUMMARY")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Standard Neumann BC (∂U/∂n = 0) may be inconsistent at boundaries")
    print("  2. Adjoint-consistent BC couples to density gradient: ∂U/∂n = -σ²/2·∂ln(m)/∂n")
    print("  3. Use bc_mode='adjoint_consistent' for boundary stall configurations")
    print("  4. Negligible computational overhead with potential convergence improvement")

    print("\nWhen to use adjoint-consistent BC:")
    print("  ✓ Stall point at domain boundary")
    print("  ✓ Reflecting boundaries (Neumann/no-flux)")
    print("  ✓ High accuracy requirements")
    print("  ✗ Not needed for interior stall or periodic BC")

    plt.show()


def visualize_comparison(problem, result_std, result_ac):
    """
    Create comparison plots for standard vs adjoint-consistent BC.

    Args:
        problem: MFG problem instance
        result_std: Result with standard BC
        result_ac: Result with adjoint-consistent BC
    """
    # Get grid
    x = np.linspace(0, 1, problem.geometry.get_grid_shape()[0])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Comparison: Standard vs Adjoint-Consistent BC", fontsize=14, fontweight="bold")

    # Plot 1: Final density comparison
    ax = axes[0, 0]
    ax.plot(x, result_std.M[-1, :], "b-", label="Standard BC", linewidth=2)
    ax.plot(x, result_ac.M[-1, :], "r--", label="Adjoint-Consistent BC", linewidth=2)
    ax.set_xlabel("Position x")
    ax.set_ylabel("Density m(T, x)")
    ax.set_title("Final Density Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Final value function comparison
    ax = axes[0, 1]
    ax.plot(x, result_std.U[-1, :], "b-", label="Standard BC", linewidth=2)
    ax.plot(x, result_ac.U[-1, :], "r--", label="Adjoint-Consistent BC", linewidth=2)
    ax.set_xlabel("Position x")
    ax.set_ylabel("Value U(T, x)")
    ax.set_title("Final Value Function")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Density difference
    ax = axes[1, 0]
    m_diff = result_std.M[-1, :] - result_ac.M[-1, :]
    ax.plot(x, m_diff, "g-", linewidth=2)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Position x")
    ax.set_ylabel("Δm = m_std - m_ac")
    ax.set_title("Density Difference")
    ax.grid(True, alpha=0.3)

    # Plot 4: Convergence history
    ax = axes[1, 1]
    if hasattr(result_std, "error_history_U") and len(result_std.error_history_U) > 0:
        ax.semilogy(result_std.error_history_U, "b-o", label="Standard BC", markersize=4)
        ax.semilogy(result_ac.error_history_U, "r-s", label="Adjoint-Consistent BC", markersize=4)
        ax.set_xlabel("Picard Iteration")
        ax.set_ylabel("Error ||U - U_prev||")
        ax.set_title("Convergence History")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "Convergence history not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    plt.tight_layout()


if __name__ == "__main__":
    """Run the tutorial."""
    compare_bc_modes()
