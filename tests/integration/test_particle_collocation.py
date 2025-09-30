#!/usr/bin/env python3
"""
Example usage of the new particle-collocation framework.

This example demonstrates how to use the split architecture:
- GFDMHJBSolver for HJB equations (collocation method)
- ParticleCollocationSolver for combined MFG problems
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mfg_pde.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver as GFDMHJBSolver
from mfg_pde.alg.numerical.mfg_solvers.particle_collocation_solver import ParticleCollocationSolver


class ExampleMFGProblem:
    """Example MFG problem with quadratic costs."""

    def __init__(self, Nx=20, Nt=10, T=1.0, L=1.0, sigma=0.1):
        self.Nx = Nx
        self.Nt = Nt
        self.T = T
        self.Lx = L
        self.xmin = 0.0
        self.xmax = L
        self.Dx = L / (Nx - 1) if Nx > 1 else 0.0
        self.Dt = T / (Nt - 1) if Nt > 1 else 0.0
        self.sigma = sigma
        self.coefCT = 1.0
        self.xSpace = np.linspace(self.xmin, self.xmax, self.Nx)

    def H(self, x_idx, m_at_x, p_values, t_idx):
        """
        Hamiltonian: H(x, m, p) = 0.5 * |p|^2 + F(x, m)
        where F(x, m) represents congestion cost.
        """
        # Extract gradient component
        if isinstance(p_values, dict):
            if "forward" in p_values:
                p = p_values["forward"]
            elif "x" in p_values:
                p = p_values["x"]
            else:
                p = 0.0
        else:
            p = p_values

        # Quadratic control cost
        control_cost = 0.5 * self.coefCT * p**2

        # Congestion cost (quadratic in density)
        congestion_cost = 0.5 * m_at_x**2

        return control_cost + congestion_cost

    def get_initial_density(self):
        """Initial density: Gaussian centered at x=0.2."""
        x = self.xSpace
        center = 0.2
        width = 0.1
        density = np.exp(-0.5 * ((x - center) / width) ** 2)
        return density / (np.sum(density) * self.Dx)

    def get_terminal_condition(self):
        """Terminal condition: quadratic potential favoring x=0.8."""
        x = self.xSpace
        target = 0.8
        return -0.5 * (x - target) ** 2


def example_particle_collocation_solver():
    """Example usage of ParticleCollocationSolver."""
    print("Example: Particle-Collocation MFG Solver")
    print("=" * 50)

    # Create problem
    problem = ExampleMFGProblem(Nx=15, Nt=8, T=1.0, sigma=0.1)

    # Create collocation points (slightly irregular for demonstration)
    base_points = np.linspace(0, 1, 12)
    noise = np.random.normal(0, 0.02, 12)
    collocation_points = np.clip(base_points + noise, 0, 1).reshape(-1, 1)

    # Initialize solver
    solver = ParticleCollocationSolver(
        problem=problem,
        collocation_points=collocation_points,
        num_particles=1000,
        delta=0.2,
        taylor_order=2,
        weight_function="gaussian",
        NiterNewton=10,
        l2errBoundNewton=1e-4,
    )

    # Print solver information
    print("Solver Configuration:")
    info = solver.get_solver_info()
    print(f"  Method: {info['method']}")
    print(f"  FP Solver: {info['fp_solver']['method']} with {info['fp_solver']['num_particles']} particles")
    print(f"  HJB Solver: {info['hjb_solver']['method']} with {info['hjb_solver']['n_collocation_points']} points")

    # Print collocation information
    coll_info = solver.get_collocation_info()
    print(f"  Collocation delta: {coll_info['delta']}")
    print(f"  Taylor order: {coll_info['taylor_order']}")
    print(
        f"  Neighborhood sizes: {coll_info['min_neighborhood_size']}-{coll_info['max_neighborhood_size']} (avg: {coll_info['avg_neighborhood_size']:.1f})"
    )

    print("\nSolving MFG system...")

    # Solve the system
    try:
        U_solution, M_solution, convergence_info = solver.solve(Niter=15, l2errBound=1e-3, verbose=True)

        print("\nSolution Results:")
        print(f"  Converged: {convergence_info['converged']}")
        print(f"  Final error: {convergence_info['final_error']:.2e}")
        print(f"  Iterations: {convergence_info['iterations']}")

        # Get particle trajectory
        particles = solver.get_particles_trajectory()
        if particles is not None:
            print(f"  Particle trajectory shape: {particles.shape}")

        # Simple visualization
        print("\nCreating visualization...")

        _fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Plot value function evolution
        axes[0, 0].imshow(U_solution, aspect="auto", origin="lower", cmap="viridis")
        axes[0, 0].set_title("Value Function U(t,x)")
        axes[0, 0].set_xlabel("Space")
        axes[0, 0].set_ylabel("Time")

        # Plot density evolution
        axes[0, 1].imshow(M_solution, aspect="auto", origin="lower", cmap="plasma")
        axes[0, 1].set_title("Density M(t,x)")
        axes[0, 1].set_xlabel("Space")
        axes[0, 1].set_ylabel("Time")

        # Plot final profiles
        axes[1, 0].plot(problem.xSpace, U_solution[-1, :], "b-", label="Final U")
        axes[1, 0].plot(problem.xSpace, U_solution[0, :], "b--", alpha=0.5, label="Initial U")
        axes[1, 0].set_title("Value Function Profiles")
        axes[1, 0].set_xlabel("x")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(problem.xSpace, M_solution[0, :], "r-", label="Initial M")
        axes[1, 1].plot(problem.xSpace, M_solution[-1, :], "r--", alpha=0.5, label="Final M")
        axes[1, 1].set_title("Density Profiles")
        axes[1, 1].set_xlabel("x")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("../output/particle_collocation_example.png", dpi=150, bbox_inches="tight")
        print("  Saved visualization as 'particle_collocation_example.png'")

        # Plot convergence history
        if convergence_info["history"]:
            plt.figure(figsize=(10, 6))
            history = convergence_info["history"]
            iterations = [h["iteration"] for h in history]
            u_errors = [h["U_error"] for h in history]
            m_errors = [h["M_error"] for h in history]

            plt.semilogy(iterations, u_errors, "b-o", label="U error", markersize=4)
            plt.semilogy(iterations, m_errors, "r-s", label="M error", markersize=4)
            plt.xlabel("Picard Iteration")
            plt.ylabel("Relative Error")
            plt.title("Convergence History")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("../output/convergence_history.png", dpi=150, bbox_inches="tight")
            print("  Saved convergence history as 'convergence_history.png'")

        return True

    except Exception as e:
        print(f"Error during solving: {e}")
        return False


def example_gfdm_hjb_solver():
    """Example usage of standalone GFDMHJBSolver."""
    print("\nExample: Standalone GFDM HJB Solver")
    print("=" * 50)

    # Create problem
    problem = ExampleMFGProblem(Nx=12, Nt=6, T=0.5, sigma=0.05)

    # Create collocation points
    collocation_points = np.linspace(0, 1, 10).reshape(-1, 1)

    # Initialize solver
    solver = GFDMHJBSolver(
        problem=problem,
        collocation_points=collocation_points,
        delta=0.25,
        taylor_order=2,
        weight_function="gaussian",
        NiterNewton=15,
        l2errBoundNewton=1e-5,
    )

    print("GFDM HJB Solver initialized:")
    print(f"  Method: {solver.hjb_method_name}")
    print(f"  Collocation points: {solver.n_points}")
    print(f"  Dimension: {solver.dimension}")
    print(f"  Delta: {solver.delta}")
    print(f"  Taylor order: {solver.taylor_order}")

    # Create mock density evolution (for demonstration)
    M_density = np.ones((problem.Nt, problem.Nx)) * 0.5
    # Add some variation
    for t in range(problem.Nt):
        M_density[t, :] = 0.5 + 0.3 * np.sin(2 * np.pi * problem.xSpace + t * 0.5)

    # Create terminal condition
    U_final = problem.get_terminal_condition()

    # Create initial guess
    U_prev = np.zeros((problem.Nt, problem.Nx))

    print("\nSolving HJB system...")

    try:
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        print(f"  Solution shape: {U_solution.shape}")
        print(f"  Solution range: [{U_solution.min():.3f}, {U_solution.max():.3f}]")

        # Simple visualization
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(U_solution, aspect="auto", origin="lower", cmap="viridis")
        plt.title("HJB Solution U(t,x)")
        plt.xlabel("Space")
        plt.ylabel("Time")
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.imshow(M_density, aspect="auto", origin="lower", cmap="plasma")
        plt.title("Given Density M(t,x)")
        plt.xlabel("Space")
        plt.ylabel("Time")
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.plot(problem.xSpace, U_solution[0, :], "b-", label="Initial U")
        plt.plot(problem.xSpace, U_solution[-1, :], "r-", label="Final U")
        plt.title("Value Function Profiles")
        plt.xlabel("x")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("../output/gfdm_hjb_example.png", dpi=150, bbox_inches="tight")
        print("  Saved visualization as 'gfdm_hjb_example.png'")

        return True

    except Exception as e:
        print(f"Error during solving: {e}")
        return False


if __name__ == "__main__":
    print("Particle-Collocation Framework Usage Examples")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    success = True

    # Example 1: Combined particle-collocation solver
    success &= example_particle_collocation_solver()

    # Example 2: Standalone GFDM HJB solver
    success &= example_gfdm_hjb_solver()

    print("\n" + "=" * 60)
    if success:
        print("✓ All examples completed successfully!")
        print("Check the generated PNG files for visualizations.")
    else:
        print("✗ Some examples failed.")

    print("\nUsage Summary:")
    print("- Use ParticleCollocationSolver for complete MFG problems")
    print("- Use GFDMHJBSolver for HJB-only problems with given density")
    print("- Both solvers support irregular collocation points")
    print("- GFDM provides meshfree, flexible spatial discretization")
    print("- Particle method naturally handles density evolution")
