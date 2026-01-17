"""
Three-Mode Solving API Demo (Issue #580)

This example demonstrates the three ways to solve MFG problems with guaranteed
adjoint duality between HJB and FP solvers.

References:
    - Issue #580: Adjoint-aware solver pairing
    - docs/theory/adjoint_operators_mfg.md: Mathematical theory
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde.types import NumericalScheme


def demo_safe_mode():
    """
    Safe Mode: Specify numerical scheme for automatic dual pairing.

    Benefits:
    - Guaranteed adjoint duality by construction
    - No way to accidentally mix incompatible solvers
    - Recommended for most users
    """
    print("\n" + "=" * 70)
    print("SAFE MODE: Automatic Dual Pairing")
    print("=" * 70)

    # Create a simple 1D problem
    problem = MFGProblem(
        Nx=[40],
        Nt=20,
        T=1.0,
        diffusion=0.1,
    )

    # Safe Mode: Just specify the scheme
    result = problem.solve(
        scheme=NumericalScheme.FDM_UPWIND,
        max_iterations=20,
        tolerance=1e-6,
        verbose=True,
    )

    print(f"\nConverged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final error (max): {result.max_error:.3e}")

    return result


def demo_expert_mode():
    """
    Expert Mode: Manual solver injection with duality validation.

    Benefits:
    - Full control over solver configuration
    - Duality validation with educational warnings
    - Useful for advanced customization
    """
    print("\n" + "=" * 70)
    print("EXPERT MODE: Manual Solver Injection")
    print("=" * 70)

    problem = MFGProblem(
        Nx=[40],
        Nt=20,
        T=1.0,
        diffusion=0.1,
    )

    # Expert Mode: Create and configure solvers manually
    hjb = HJBFDMSolver(problem)
    fp = FPFDMSolver(problem, advection_scheme="gradient_upwind")

    # Solve with custom solvers (duality automatically validated)
    result = problem.solve(
        hjb_solver=hjb,
        fp_solver=fp,
        max_iterations=20,
        tolerance=1e-6,
        verbose=True,
    )

    print(f"\nConverged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final error (max): {result.max_error:.3e}")

    return result


def demo_auto_mode():
    """
    Auto Mode: Intelligent automatic scheme selection.

    Benefits:
    - Zero configuration required
    - Analyzes problem geometry and selects appropriate scheme
    - Good default for quick experiments
    """
    print("\n" + "=" * 70)
    print("AUTO MODE: Intelligent Automatic Selection")
    print("=" * 70)

    problem = MFGProblem(
        Nx=[40],
        Nt=20,
        T=1.0,
        diffusion=0.1,
    )

    # Auto Mode: No scheme or solvers specified
    # System automatically selects FDM_UPWIND (safe default)
    result = problem.solve(
        max_iterations=20,
        tolerance=1e-6,
        verbose=True,
    )

    print(f"\nConverged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final error (max): {result.max_error:.3e}")

    return result


def compare_schemes():
    """
    Compare different numerical schemes in Safe Mode.

    Shows how easy it is to test different discretizations while
    maintaining adjoint duality guarantees.
    """
    print("\n" + "=" * 70)
    print("SCHEME COMPARISON: Testing Different Discretizations")
    print("=" * 70)

    problem = MFGProblem(
        Nx=[40],
        Nt=20,
        T=1.0,
        diffusion=0.1,
    )

    schemes = [
        NumericalScheme.FDM_UPWIND,
        NumericalScheme.FDM_CENTERED,
    ]

    results = {}

    for scheme in schemes:
        print(f"\n--- Testing {scheme.value} ---")
        result = problem.solve(
            scheme=scheme,
            max_iterations=20,
            tolerance=1e-6,
            verbose=False,
        )
        results[scheme.value] = result
        print(f"Converged: {result.converged} in {result.iterations} iterations")
        print(f"Final error (max): {result.max_error:.3e}")

    return results


def plot_results(result):
    """Visualize the solution from any mode."""
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)

    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot value function
    ax = axes[0]
    x = np.linspace(0, 1, result.U.shape[1])
    for t_idx in [0, result.U.shape[0] // 2, -1]:
        ax.plot(x, result.U[t_idx, :], label=f"t={t_idx * result.problem.T / result.U.shape[0]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("U(t, x)")
    ax.set_title("Value Function")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot density
    ax = axes[1]
    for t_idx in [0, result.M.shape[0] // 2, -1]:
        ax.plot(x, result.M[t_idx, :], label=f"t={t_idx * result.problem.T / result.M.shape[0]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("m(t, x)")
    ax.set_title("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Plots displayed.")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "THREE-MODE SOLVING API DEMO" + " " * 25 + "║")
    print("║" + " " * 20 + "(Issue #580)" + " " * 35 + "║")
    print("╚" + "=" * 68 + "╝")

    print("\nThis demo shows three ways to solve MFG problems:")
    print("  1. Safe Mode: Specify scheme for automatic dual pairing")
    print("  2. Expert Mode: Manual solver injection with validation")
    print("  3. Auto Mode: Intelligent automatic selection")
    print("\nAll three modes guarantee adjoint duality between HJB and FP solvers.")

    # Run demos
    result_safe = demo_safe_mode()
    result_expert = demo_expert_mode()
    result_auto = demo_auto_mode()

    # Compare schemes
    compare_schemes()

    # Visualize one result
    plot_results(result_safe)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nAll three modes produced solutions:")
    print(f"  Safe Mode:   {result_safe.iterations} iterations, error={result_safe.max_error:.3e}")
    print(f"  Expert Mode: {result_expert.iterations} iterations, error={result_expert.max_error:.3e}")
    print(f"  Auto Mode:   {result_auto.iterations} iterations, error={result_auto.max_error:.3e}")

    print("\nRecommendation: Use Safe Mode for most applications.")
    print("                Use Expert Mode only when you need custom solver config.")
    print("                Use Auto Mode for quick experiments with default settings.")


if __name__ == "__main__":
    main()
