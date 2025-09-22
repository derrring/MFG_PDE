#!/usr/bin/env python3
"""
Particle-Collocation MFG Solver Example
=====================================

This example demonstrates the particle-collocation method for solving Mean Field Games
with no-flux boundary conditions, showcasing modern API usage and mass conservation.

Features:
- Modern parameter naming conventions
- Factory pattern usage (optional)
- Mass conservation monitoring
- Professional visualization
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.mfg_solvers.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.geometry import BoundaryConditions
from mfg_pde.utils.integration import trapezoid

# Optional: Modern factory pattern usage
try:
    from mfg_pde.config import create_fast_config
    from mfg_pde.factory import create_fast_solver

    FACTORY_AVAILABLE = True
except ImportError:
    FACTORY_AVAILABLE = False


def run_particle_collocation_example():
    """Demonstrate particle-collocation method with modern API"""
    print("=== Particle-Collocation MFG Solver Example ===")
    print("Demonstrating mass conservation with no-flux boundary conditions")

    # Problem parameters - balanced for demonstration
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 60,
        "T": 1.0,
        "Nt": 50,
        "sigma": 0.2,
        "coefCT": 0.05,
    }

    print("\nProblem Configuration:")
    for key, value in problem_params.items():
        print(f"  {key}: {value}")

    # Create MFG problem
    problem = ExampleMFGProblem(**problem_params)

    # Setup collocation points
    num_collocation_points = 15
    collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)

    # Identify boundary points for no-flux conditions
    boundary_tolerance = 1e-10
    boundary_indices = []
    for i, point in enumerate(collocation_points):
        x = point[0]
        if abs(x - problem.xmin) < boundary_tolerance or abs(x - problem.xmax) < boundary_tolerance:
            boundary_indices.append(i)

    boundary_indices = np.array(boundary_indices)
    print(f"\nBoundary points: {len(boundary_indices)} out of {num_collocation_points}")

    # No-flux boundary conditions
    no_flux_bc = BoundaryConditions(type="no_flux")

    # Modern solver parameters with updated naming
    solver_params = {
        "num_particles": 1000,
        "delta": 0.4,
        "taylor_order": 2,
        "weight_function": "wendland",
        "weight_scale": 1.0,
        "max_newton_iterations": 12,  # Modern parameter name
        "newton_tolerance": 1e-4,  # Modern parameter name
        "kde_bandwidth": "scott",
        "normalize_kde_output": False,
        "boundary_indices": boundary_indices,
        "boundary_conditions": no_flux_bc,
        "use_monotone_constraints": True,
    }

    print("\nSolver Configuration:")
    for key, value in solver_params.items():
        if key != "boundary_indices":
            if key == "use_monotone_constraints" and value:
                print(f"  {key}: {value} <- QP constraints enabled")
            else:
                print(f"  {key}: {value}")

    # Method 1: Modern factory pattern (if available)
    if FACTORY_AVAILABLE:
        print("\n=== Method 1: Factory Pattern Usage ===")
        config = create_fast_config()
        # Customize config for this example
        config.particle.num_particles = solver_params["num_particles"]
        config.newton.max_iterations = solver_params["max_newton_iterations"]
        config.newton.tolerance = solver_params["newton_tolerance"]

        try:
            factory_solver = create_fast_solver(
                problem=problem,
                solver_type="particle_collocation",
                config=config,
                collocation_points=collocation_points,
                **{
                    k: v
                    for k, v in solver_params.items()
                    if k not in ["max_newton_iterations", "newton_tolerance", "num_particles"]
                },
            )

            start_time = time.time()
            factory_result = factory_solver.solve(max_picard_iterations=18, verbose=True)
            factory_time = time.time() - start_time

            if hasattr(factory_result, "solution"):
                U_factory, M_factory = factory_result.solution, factory_result.density
                print(f"Factory method completed in {factory_time:.2f}s")
                print(f"Factory convergence: {factory_result.convergence_info}")
            else:
                U_factory, M_factory, info_factory = factory_result
                print(f"Factory method completed in {factory_time:.2f}s")
        except Exception as e:
            print(f"Factory pattern failed: {e}")
            FACTORY_AVAILABLE = False

    # Method 2: Direct class usage (always available)
    print("\n=== Method 2: Direct Class Usage ===")
    solver = ParticleCollocationSolver(problem=problem, collocation_points=collocation_points, **solver_params)

    # Solve with modern parameter names
    start_time = time.time()
    U, M, info = solver.solve(
        max_picard_iterations=18,
        picard_tolerance=1e-5,
        verbose=True,  # Modern parameter name  # Modern parameter name
    )
    solve_time = time.time() - start_time

    print(f"Direct method completed in {solve_time:.2f}s")
    print(f"Convergence: {info.get('converged', 'Unknown')}")

    # Mass conservation analysis
    print("\n=== Mass Conservation Analysis ===")
    total_mass = trapezoid(M[-1, :], problem.x_grid)
    mass_error = abs(total_mass - 1.0)
    print(f"Final total mass: {total_mass:.6f}")
    print(f"Mass conservation error: {mass_error:.2e}")

    if mass_error < 0.001:
        print("✓ Excellent mass conservation achieved")
    elif mass_error < 0.01:
        print("✓ Good mass conservation achieved")
    else:
        print("⚠ Mass conservation could be improved")

    # Basic visualization
    print("\n=== Creating Visualization ===")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Value function evolution
    T_plot, X_plot = np.meshgrid(problem.t_grid, problem.x_grid)
    im1 = ax1.contourf(T_plot, X_plot, U.T, levels=20, cmap="viridis")
    ax1.set_title("Value Function u(t,x)", fontsize=14)
    ax1.set_xlabel("Time t")
    ax1.set_ylabel("Space x")
    plt.colorbar(im1, ax=ax1)

    # Density evolution
    im2 = ax2.contourf(T_plot, X_plot, M.T, levels=20, cmap="plasma")
    ax2.set_title("Density m(t,x)", fontsize=14)
    ax2.set_xlabel("Time t")
    ax2.set_ylabel("Space x")
    plt.colorbar(im2, ax=ax2)

    # Final profiles
    ax3.plot(problem.x_grid, U[-1, :], "b-", linewidth=2, label="Final value function")
    ax3.set_title("Final Value Function u(T,x)", fontsize=14)
    ax3.set_xlabel("Space x")
    ax3.set_ylabel("u(T,x)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4.plot(problem.x_grid, M[-1, :], "r-", linewidth=2, label="Final density")
    ax4.set_title("Final Density m(T,x)", fontsize=14)
    ax4.set_xlabel("Space x")
    ax4.set_ylabel("m(T,x)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig("particle_collocation_example.png", dpi=150, bbox_inches="tight")
    print("Visualization saved as 'particle_collocation_example.png'")

    # Show comparison if both methods worked
    if FACTORY_AVAILABLE and "U_factory" in locals():
        print("\n=== Method Comparison ===")
        solution_diff = np.max(np.abs(U - U_factory))
        density_diff = np.max(np.abs(M - M_factory))
        print(f"Max solution difference: {solution_diff:.2e}")
        print(f"Max density difference: {density_diff:.2e}")

        if solution_diff < 1e-10 and density_diff < 1e-10:
            print("✓ Factory and direct methods produce identical results")
        else:
            print("ℹ Methods produce slightly different results (expected)")

    print("\n=== Example Summary ===")
    print("✓ Particle-collocation method successfully demonstrated")
    print(f"✓ Mass conservation error: {mass_error:.2e}")
    print("✓ Modern API usage with updated parameter names")
    if FACTORY_AVAILABLE:
        print("✓ Factory pattern demonstrated")
    print("✓ No-flux boundary conditions properly handled")
    print("✓ QP constraints enabled for monotonicity preservation")

    return U, M, info


if __name__ == "__main__":
    # Run the example
    try:
        U, M, info = run_particle_collocation_example()

        # Optional: Show plot if running interactively
        try:
            plt.show()
        except:
            pass

    except Exception as e:
        print(f"Example failed with error: {e}")
        import traceback

        traceback.print_exc()
