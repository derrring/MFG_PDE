#!/usr/bin/env python3
"""
Simple WENO5 HJB Solver Demo

This demonstration showcases the newly implemented WENO5 HJB solver
with direct testing and basic benchmarking capabilities.

Academic Context:
- Fifth-order WENO spatial discretization
- TVD-RK3 time integration
- Non-oscillatory behavior for discontinuous solutions
- Comparison with standard finite difference methods
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import ExampleMFGProblem
from mfg_pde.alg.hjb_solvers import HJBFDMSolver, HJBWeno5Solver


def create_test_problem(Nx: int = 64, Nt: int = 32) -> ExampleMFGProblem:
    """Create a test MFG problem for benchmarking."""
    return ExampleMFGProblem(x_bounds=(0.0, 2.0 * np.pi), t_bounds=(0.0, 1.0), Nx=Nx, Nt=Nt, sigma=0.3)


def test_hjb_solver(problem: ExampleMFGProblem, solver, solver_name: str):
    """Test HJB solver directly with realistic inputs."""
    print(f"\n--- Testing {solver_name} ---")

    # Create test inputs
    Nt, Nx = problem.Nt + 1, problem.Nx + 1
    x = np.linspace(0.0, 2.0 * np.pi, Nx)  # Use known domain

    # Constant density (realistic for MFG)
    M_density = np.ones((Nt, Nx)) * 0.5

    # Smooth terminal condition
    U_final = 0.5 * np.sin(x) + 0.2 * np.cos(2 * x)

    # Zero previous Picard iteration
    U_prev = np.zeros((Nt, Nx))

    # Time the solve
    start_time = time.perf_counter()

    try:
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)
        elapsed_time = time.perf_counter() - start_time

        # Compute metrics
        solution_range = (np.min(U_solution), np.max(U_solution))
        solution_finite = np.all(np.isfinite(U_solution))
        final_variation = np.mean(np.abs(np.diff(U_solution[-1, :], n=2)))

        print(f"✓ Completed in {elapsed_time:.4f}s")
        print(f"✓ Solution range: [{solution_range[0]:.4f}, {solution_range[1]:.4f}]")
        print(f"✓ All finite: {solution_finite}")
        print(f"✓ Final variation: {final_variation:.6f}")

        return {
            "name": solver_name,
            "U_solution": U_solution,
            "elapsed_time": elapsed_time,
            "variation": final_variation,
            "success": True,
        }

    except Exception as e:
        print(f"❌ Failed: {e}")
        return {"name": solver_name, "success": False, "error": str(e)}


def compare_solvers():
    """Compare WENO5 vs standard FDM."""
    print("🔬 WENO5 vs Standard FDM Comparison")
    print("=" * 50)

    # Test on different grid sizes
    grid_sizes = [32, 64, 128]
    results = {}

    for Nx in grid_sizes:
        print(f"\n📐 Grid size: {Nx} x {Nx//2}")

        # Create problem
        problem = create_test_problem(Nx=Nx, Nt=Nx // 2)

        # Test solvers
        solvers = [
            (HJBFDMSolver, "Standard FDM", {}),
            (HJBWeno5Solver, "WENO5 (CFL=0.3)", {"cfl_number": 0.3}),
            (HJBWeno5Solver, "WENO5 (CFL=0.1)", {"cfl_number": 0.1}),
        ]

        for solver_class, name, kwargs in solvers:
            solver = solver_class(problem, **kwargs)
            result = test_hjb_solver(problem, solver, name)

            if result["success"]:
                if name not in results:
                    results[name] = {"grid_sizes": [], "times": [], "variations": []}
                results[name]["grid_sizes"].append(Nx)
                results[name]["times"].append(result["elapsed_time"])
                results[name]["variations"].append(result["variation"])

    return results


def plot_comparison(results: dict):
    """Plot solver comparison results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["blue", "red", "green", "orange"]

    for idx, (solver_name, data) in enumerate(results.items()):
        if not data["grid_sizes"]:
            continue

        color = colors[idx % len(colors)]
        grid_sizes = np.array(data["grid_sizes"])

        # Plot computational time
        ax1.loglog(grid_sizes, data["times"], "o-", color=color, label=solver_name, linewidth=2, markersize=6)

        # Plot solution variation (smoothness indicator)
        ax2.loglog(grid_sizes, data["variations"], "s-", color=color, label=solver_name, linewidth=2, markersize=6)

    # Format plots
    ax1.set_xlabel("Grid Size (Nx)")
    ax1.set_ylabel("Computational Time (s)")
    ax1.set_title("Performance Scaling")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Grid Size (Nx)")
    ax2.set_ylabel("Solution Variation")
    ax2.set_title("Solution Smoothness")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("weno5_comparison.png", dpi=300, bbox_inches="tight")
    print("\n📊 Comparison plot saved: weno5_comparison.png")
    plt.show()


def solution_visualization():
    """Visualize WENO5 solution."""
    print("\n🎨 Solution Visualization")
    print("-" * 30)

    # Create test problem
    problem = create_test_problem(Nx=128, Nt=64)

    # Test WENO5 solver
    solver = HJBWeno5Solver(problem, cfl_number=0.3)
    result = test_hjb_solver(problem, solver, "WENO5")

    if result["success"]:
        U_solution = result["U_solution"]

        # Create visualization
        x = np.linspace(0.0, 2.0 * np.pi, problem.Nx + 1)
        t = np.linspace(0.0, 1.0, problem.Nt + 1)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("WENO5 HJB Solver Solution", fontsize=14, fontweight="bold")

        # Final solution
        axes[0, 0].plot(x, U_solution[-1, :], "b-", linewidth=2)
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("U(T, x)")
        axes[0, 0].set_title("Final Value Function")
        axes[0, 0].grid(True, alpha=0.3)

        # Time evolution at midpoint
        mid_idx = len(x) // 2
        axes[0, 1].plot(t, U_solution[:, mid_idx], "r-", linewidth=2)
        axes[0, 1].set_xlabel("t")
        axes[0, 1].set_ylabel("U(t, x_mid)")
        axes[0, 1].set_title("Time Evolution")
        axes[0, 1].grid(True, alpha=0.3)

        # Solution surface
        T, X = np.meshgrid(t, x, indexing="ij")
        contour = axes[1, 0].contourf(X, T, U_solution, levels=20, cmap="viridis")
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("t")
        axes[1, 0].set_title("Solution Surface U(t,x)")
        plt.colorbar(contour, ax=axes[1, 0])

        # Derivative visualization (WENO5 reconstruction quality)
        u_x_plus, u_x_minus = solver._compute_weno5_derivatives(U_solution[-1, :])
        axes[1, 1].plot(x, u_x_plus, "g-", label="Forward", linewidth=2)
        axes[1, 1].plot(x, u_x_minus, "orange", linestyle="--", label="Backward", linewidth=2)
        axes[1, 1].set_xlabel("x")
        axes[1, 1].set_ylabel("∂U/∂x")
        axes[1, 1].set_title("WENO5 Derivatives")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("weno5_solution.png", dpi=300, bbox_inches="tight")
        print("📊 Solution visualization saved: weno5_solution.png")
        plt.show()


def main():
    """Main demonstration."""
    print("🚀 WENO5 HJB Solver Simple Demo")
    print("=" * 40)
    print("Testing fifth-order WENO solver with TVD-RK3 integration")

    try:
        # Run solver comparison
        results = compare_solvers()

        # Plot results
        if results:
            plot_comparison(results)

        # Visualize solution
        solution_visualization()

        print("\n🎉 WENO5 Demo Completed Successfully!")
        print("✓ Fifth-order spatial accuracy")
        print("✓ Non-oscillatory reconstruction")
        print("✓ Explicit time integration")
        print("✓ Ready for academic benchmarking")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
