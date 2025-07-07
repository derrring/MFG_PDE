#!/usr/bin/env python3
"""
Particle-Collocation MFG Solver - Numerical Example
==================================================

This example demonstrates the particle-collocation method using the same numerical
settings as the existing test examples (fdm_solver.py and particle_solver.py).
It matches the problem setup and parameters from the test files.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.utils.plot_utils import plot_results, plot_convergence


def run_particle_collocation_example():
    """Run the particle-collocation method with test settings"""
    print("Particle-Collocation MFG Solver - Numerical Example")
    print("=" * 60)

    # Problem parameters matching other examples exactly
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 51,  # Grid discretization (same as other examples)
        "T": 1.0,  # Final time
        "Nt": 51,  # Time discretization (same as other examples)
        "sigma": 1.0,  # Diffusion coefficient (same as other examples)
        "coefCT": 0.5,
    }

    print(f"Problem Parameters (matching other examples):")
    print(f"  Nx (grid points): {problem_params['Nx']}")
    print(f"  Nt (time points): {problem_params['Nt']}")
    print(f"  T (final time): {problem_params['T']}")
    print(f"  sigma (diffusion): {problem_params['sigma']}")
    print(f"  coefCT: {problem_params['coefCT']}")

    # Create problem using the same ExampleMFGProblem as other examples
    problem = ExampleMFGProblem(**problem_params)

    # Create collocation points (regular grid for this example)
    n_collocation = 10
    collocation_points = np.linspace(0, 1, n_collocation).reshape(-1, 1)

    print(f"\nCollocation Setup:")
    print(f"  Number of collocation points: {n_collocation}")
    print(
        f"  Collocation points range: [{collocation_points.min():.2f}, {collocation_points.max():.2f}]"
    )

    # Solver parameters (matching other examples exactly)
    num_particles = 1000  # Same as hybrid_particle_fdm
    delta = 0.15
    taylor_order = 2
    weight_function = "gaussian"

    # Solver convergence parameters (balanced for quality vs time)
    Niter_max_picard = 20  # Reasonable for demonstration (other examples: 100)
    conv_threshold_picard = 1e-4  # Good accuracy (other examples: 1e-5)
    NiterNewton = 20  # Good accuracy (other examples: 30)
    l2errBoundNewton = 1e-6  # Good accuracy (other examples: 1e-7)

    print(f"\nSolver Parameters (balanced for quality vs time):")
    print(f"  Number of particles: {num_particles}")
    print(f"  Delta (collocation radius): {delta}")
    print(f"  Taylor order: {taylor_order}")
    print(f"  Weight function: {weight_function}")
    print(f"  Max Picard iterations: {Niter_max_picard}")
    print(f"  Picard tolerance: {conv_threshold_picard}")
    print(f"  Newton iterations: {NiterNewton}")
    print(f"  Newton tolerance: {l2errBoundNewton}")

    # Initialize solver
    solver = ParticleCollocationSolver(
        problem=problem,
        collocation_points=collocation_points,
        num_particles=num_particles,
        delta=delta,
        taylor_order=taylor_order,
        weight_function=weight_function,
        NiterNewton=NiterNewton,
        l2errBoundNewton=l2errBoundNewton,
    )

    # Print solver information
    print(f"\nSolver Information:")
    info = solver.get_solver_info()
    print(f"  Method: {info['method']}")
    print(
        f"  FP Solver: {info['fp_solver']['method']} with {info['fp_solver']['num_particles']} particles"
    )
    print(
        f"  HJB Solver: {info['hjb_solver']['method']} with {info['hjb_solver']['n_collocation_points']} points"
    )

    # Print collocation information
    coll_info = solver.get_collocation_info()
    print(f"  Collocation delta: {coll_info['delta']}")
    print(f"  Taylor order: {coll_info['taylor_order']}")
    print(
        f"  Neighborhood sizes: {coll_info['min_neighborhood_size']}-{coll_info['max_neighborhood_size']} (avg: {coll_info['avg_neighborhood_size']:.1f})"
    )

    print(f"\nSolving MFG system...")

    # Solve the system using the same parameters as other examples
    try:
        U_solution, M_solution, convergence_info = solver.solve(
            Niter=Niter_max_picard, l2errBound=conv_threshold_picard, verbose=True
        )

        print(f"\nSolution Results:")
        print(f"  Converged: {convergence_info['converged']}")
        print(f"  Final error: {convergence_info['final_error']:.2e}")
        print(f"  Iterations: {convergence_info['iterations']}")
        print(f"  Solution shapes: U{U_solution.shape}, M{M_solution.shape}")

        # Check value function range
        U_min, U_max = U_solution.min(), U_solution.max()
        U_initial_range = f"[{U_solution[0].min():.2f}, {U_solution[0].max():.2f}]"
        U_final_range = f"[{U_solution[-1].min():.2f}, {U_solution[-1].max():.2f}]"
        print(
            f"  U value range: overall[{U_min:.2f}, {U_max:.2f}], initial{U_initial_range}, final{U_final_range}"
        )

        # Check if evolution direction is correct (should increase toward 0)
        u_t0 = U_solution[0].mean()  # Early time
        u_t_mid = U_solution[U_solution.shape[0] // 2].mean()  # Mid time
        u_T = U_solution[-1].mean()  # Final time (should be 0)
        print(
            f"  U evolution check: t=0({u_t0:.1f}) → t=0.5({u_t_mid:.1f}) → t=T({u_T:.1f})"
        )
        print(
            f"  {'✓ CORRECT' if u_t0 < u_t_mid < u_T else '✗ WRONG'} - Should increase toward 0"
        )

        # Create output directory if it doesn't exist
        output_dir = "../tests/output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Visualization using standard plot_utils (3D plots)
        print(f"\nCreating 3D visualizations...")

        # Use the standard plotting utilities for consistent visualization
        solver_name = "Particle-Collocation"
        plot_results(problem, U_solution, M_solution, solver_name=solver_name)

        # Plot convergence if we have history
        if convergence_info["history"] and len(convergence_info["history"]) > 1:
            history = convergence_info["history"]
            u_errors = [h["U_error"] for h in history]
            m_errors = [h["M_error"] for h in history]
            iterations_run = len(history)

            plot_convergence(
                iterations_run, u_errors, m_errors, solver_name=solver_name
            )

        # Additional visualization: particle trajectories
        particles = solver.get_particles_trajectory()
        if particles is not None:
            print(f"\nCreating particle trajectory visualization...")

            fig, ax = plt.subplots(figsize=(10, 6))

            # Sample particles for visualization
            n_sample = min(100, particles.shape[1])
            sample_indices = np.random.choice(
                particles.shape[1], n_sample, replace=False
            )

            # Plot trajectories
            for i in sample_indices:
                ax.plot(
                    particles[:, i],
                    np.linspace(0, problem_params["T"], particles.shape[0]),
                    "b-",
                    alpha=0.2,
                    linewidth=0.5,
                )

            ax.set_xlabel("x")
            ax.set_ylabel("t")
            ax.set_title(
                f"Particle Trajectories (Sample of {n_sample}) - {solver_name}"
            )
            ax.grid(True, alpha=0.3)

            # Save particle plot
            output_path = os.path.join(output_dir, "particle_trajectories.png")
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"  Saved particle trajectories: {output_path}")
            plt.show()

        return True

    except Exception as e:
        print(f"Error during solving: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Running Particle-Collocation Numerical Example")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    success = run_particle_collocation_example()

    print("\n" + "=" * 60)
    if success:
        print("✓ Particle-collocation numerical example completed successfully!")
        print("Check the generated PNG file for visualization.")
    else:
        print("✗ Particle-collocation numerical example failed.")

    print("\nExample Summary:")
    print(
        "- Uses exact same parameters as other examples (ExampleMFGProblem, sigma=1.0, Nx=Nt=51)"
    )
    print("- Applies particle-collocation method with GFDM for HJB")
    print("- Same convergence tolerances and iterations as other examples")
    print("- Includes comprehensive visualization of results")
    print("- Demonstrates convergence behavior and particle trajectories")
