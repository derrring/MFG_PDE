#!/usr/bin/env python3
"""
Particle-Collocation MFG Solver - No-Flux Boundary Conditions Test
================================================================

This example tests the particle-collocation method with no-flux boundary conditions
to evaluate mass conservation properties.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.utils.plot_utils import plot_convergence, plot_results


def run_particle_collocation_no_flux_test():
    """Test particle-collocation method with no-flux boundary conditions"""
    print("--- Testing Particle-Collocation with No-Flux Boundary Conditions ---")
    print("--- Particles should reflect, collocation should handle boundaries ---")
    print("--- FEATURING: Constrained QP for monotonicity preservation ---")

    # Problem parameters (balanced real-world case - T=1)
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 60,  # Higher resolution as requested
        "T": 1.0,  # Full time horizon as requested
        "Nt": 50,  # More time steps for T=1 stability
        "sigma": 0.2,  # Balanced diffusion coefficient
        "coefCT": 0.05,  # Balanced coupling strength
    }

    print(f"Problem Parameters:")
    for key, value in problem_params.items():
        print(f"  {key}: {value}")

    # Create problem
    problem = ExampleMFGProblem(**problem_params)

    # Create collocation points (proven stable scaled up)
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
    print(f"Boundary collocation points: {len(boundary_indices)} out of {num_collocation_points}")

    # No-flux boundary conditions
    no_flux_bc = BoundaryConditions(type="no_flux")

    # Solver parameters (realistic configuration with QP debugging)
    solver_params = {
        "num_particles": 1000,  # Realistic particle count
        "delta": 0.4,  # Standard neighborhood size
        "taylor_order": 2,  # Second-order Taylor for stability
        "weight_function": "wendland",  # Wendland kernel from mathematical framework
        "weight_scale": 1.0,  # Weight scale parameter
        "NiterNewton": 12,  # Adequate Newton iterations
        "l2errBoundNewton": 1e-4,  # Standard tolerance
        "kde_bandwidth": "scott",  # KDE bandwidth
        "normalize_kde_output": False,  # As requested - no KDE renormalization
        "boundary_indices": boundary_indices,
        "boundary_conditions": no_flux_bc,
        "use_monotone_constraints": True,  # ENABLE QP CONSTRAINTS FOR MONOTONICITY
    }

    print(f"\nSolver Parameters:")
    for key, value in solver_params.items():
        if key != "boundary_indices":  # Skip array printing
            if key == "use_monotone_constraints" and value:
                print(f"  {key}: {value} ← QP CONSTRAINTS ENABLED")
            else:
                print(f"  {key}: {value}")

    # Create solver
    print(f"\n--- Creating Particle-Collocation Solver ---")
    solver = ParticleCollocationSolver(problem=problem, collocation_points=collocation_points, **solver_params)

    # Solve (balanced settings for T=1 with QP constraints)
    max_iterations = 18
    convergence_tolerance = 1e-3

    print(f"\n--- Running Particle-Collocation Solver ---")
    start_time = time.time()

    try:
        U_solution, M_solution, solve_info = solver.solve(
            Niter=max_iterations, l2errBound=convergence_tolerance, verbose=True
        )

        solve_time = time.time() - start_time
        iterations_run = solve_info.get("iterations", max_iterations)
        converged = solve_info.get("converged", False)

        print(f"\n--- Solver finished in {solve_time:.2f} seconds ({iterations_run} iterations) ---")
        print(f"Converged: {converged}")

        if U_solution is not None and M_solution is not None:
            # Mass conservation analysis
            print(f"\n--- Mass Conservation Analysis ---")
            total_mass = np.sum(M_solution * problem.Dx, axis=1)

            print(f"Initial mass: {total_mass[0]:.10f}")
            print(f"Final mass: {total_mass[-1]:.10f}")
            print(f"Mass change: {(total_mass[-1] - total_mass[0]):.2e}")
            print(f"Relative mass change: {(total_mass[-1] - total_mass[0])/total_mass[0]*100:.6f}%")

            max_mass = np.max(total_mass)
            min_mass = np.min(total_mass)
            mass_variation = max_mass - min_mass
            print(f"Max mass variation: {mass_variation:.2e}")
            print(f"Relative mass variation: {mass_variation/total_mass[0]*100:.6f}%")

            # Particle boundary analysis
            particles_trajectory = solver.fp_solver.M_particles_trajectory
            if particles_trajectory is not None:
                print(f"\n--- Particle Boundary Analysis ---")
                xmin, xmax = problem.xmin, problem.xmax

                final_particles = particles_trajectory[-1, :]
                particles_in_bounds = np.all((final_particles >= xmin) & (final_particles <= xmax))

                print(f"All particles within bounds [{xmin:.2f}, {xmax:.2f}]: {particles_in_bounds}")
                print(f"Final particle range: [{np.min(final_particles):.4f}, {np.max(final_particles):.4f}]")
                print(f"Number of particles: {len(final_particles)}")

                # Count boundary violations
                total_violations = 0
                for t_step in range(particles_trajectory.shape[0]):
                    step_particles = particles_trajectory[t_step, :]
                    violations = np.sum((step_particles < xmin) | (step_particles > xmax))
                    total_violations += violations

                print(f"Total boundary violations: {total_violations}")
                if total_violations == 0:
                    print("✓ EXCELLENT: No particles escaped boundaries!")
                else:
                    print(f"⚠ WARNING: {total_violations} boundary violations detected")

            # Mass conservation assessment
            if mass_variation < 1e-6:
                print("✓ EXCELLENT: Mass is well conserved")
            elif mass_variation < 1e-3:
                print("? FAIR: Some mass variation present")
            else:
                print("✗ POOR: Significant mass variation")

            # Plotting
            print(f"\n--- Plotting Results ---")
            solver_name = "Particle-Collocation"
            plot_results(problem, U_solution, M_solution, solver_name=f"{solver_name}_NoFlux")

            # Mass conservation plot
            plt.figure()
            plt.plot(
                problem.tSpace,
                total_mass,
                "g-",
                linewidth=2,
                label="Particle-Collocation",
            )
            plt.xlabel("Time t")
            plt.ylabel("Total Mass ∫m(t,x)dx")
            plt.title(f"Mass Conservation - No-Flux Boundaries ({solver_name})")
            plt.grid(True)
            plt.legend()

            # Adaptive y-axis
            y_margin = 0.1 * abs(max_mass - min_mass) if mass_variation > 1e-10 else 0.01
            plt.ylim([min_mass - y_margin, max_mass + y_margin])
            plt.show()

        else:
            print("❌ Solver failed to produce valid results")

    except Exception as e:
        print(f"❌ Solver failed with error: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n--- Particle-Collocation No-Flux Test Finished ---")


if __name__ == "__main__":
    run_particle_collocation_no_flux_test()
