#!/usr/bin/env python3
"""
Debug test for no-flux boundary conditions - focusing on initial conditions
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def test_debug_noflux():
    print("=== Debug No-Flux Test ===")

    # Very simple problem
    problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=10, T=0.05, Nt=3, sigma=0.3, coefCT=0.1)  # Very short time

    # Minimal collocation setup
    num_colloc = 5
    collocation_points = np.linspace(0.0, 1.0, num_colloc).reshape(-1, 1)
    boundary_indices = np.array([0, num_colloc - 1])

    print(f"Problem: Nx={problem.Nx}, T={problem.T}, Nt={problem.Nt}")
    print(f"Collocation: {num_colloc} points")
    print(f"Boundary indices: {boundary_indices}")

    no_flux_bc = BoundaryConditions(type='no_flux')

    try:
        solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=50,  # Fewer particles
            delta=0.8,
            taylor_order=1,
            weight_function="wendland",
            NiterNewton=5,  # Fewer Newton iterations
            l2errBoundNewton=1e-3,  # Relaxed tolerance
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=boundary_indices,
            boundary_conditions=no_flux_bc,
        )

        print("✓ Solver created")

        # Check initial conditions
        print("\n--- Initial Condition Check ---")

        # Check FP solver initial conditions
        fp_solver = solver.fp_solver
        initial_particles = fp_solver.M_particles_trajectory[0, :]
        print(f"Initial particles range: [{np.min(initial_particles):.4f}, {np.max(initial_particles):.4f}]")
        print(f"Particles in bounds: {np.all((initial_particles >= 0) & (initial_particles <= 1))}")

        # Check initial density
        M_initial = fp_solver.M_density_evolution[0, :]
        print(f"Initial density range: [{np.min(M_initial):.4f}, {np.max(M_initial):.4f}]")
        print(f"Initial mass: {np.sum(M_initial * problem.Dx):.6f}")

        # Check HJB initial condition
        hjb_solver = solver.hjb_solver
        U_final = hjb_solver.U_final_condition
        print(f"Final condition U range: [{np.min(U_final):.4f}, {np.max(U_final):.4f}]")

        # Run one Picard iteration to debug
        print(f"\n--- Single Picard Iteration Debug ---")

        # Run FP step
        print("Running FP solver...")
        M_evolution = fp_solver.solve_fp_system(
            U_evolution_from_HJB=np.zeros((problem.Nt, problem.Nx + 1)), verbose=True
        )

        if M_evolution is not None:
            print(f"✓ FP solver successful")
            mass_evolution = np.sum(M_evolution * problem.Dx, axis=1)
            print(f"FP mass evolution: {mass_evolution}")

            # Run HJB step
            print("Running HJB solver...")
            U_evolution = hjb_solver.solve_hjb_system(
                M_density_evolution_from_FP=M_evolution,
                U_final_condition_at_T=U_final,
                U_from_prev_picard=np.zeros((problem.Nt, problem.Nx + 1)),
            )

            if U_evolution is not None:
                print(f"✓ HJB solver successful")
                print(f"U range: [{np.min(U_evolution):.4f}, {np.max(U_evolution):.4f}]")

                # Check boundary conditions
                print(f"\n--- Boundary Condition Check ---")
                for t in range(problem.Nt):
                    u_left = U_evolution[t, 0]
                    u_right = U_evolution[t, -1]
                    print(f"t={t}: U(left)={u_left:.4f}, U(right)={u_right:.4f}")

                # Quick plot
                plt.figure(figsize=(10, 4))

                plt.subplot(1, 2, 1)
                plt.contourf(problem.xSpace, problem.tSpace, M_evolution, levels=10)
                plt.colorbar(label='Density')
                plt.xlabel('Position x')
                plt.ylabel('Time t')
                plt.title('Density Evolution')

                plt.subplot(1, 2, 2)
                plt.contourf(problem.xSpace, problem.tSpace, U_evolution, levels=10)
                plt.colorbar(label='Value Function')
                plt.xlabel('Position x')
                plt.ylabel('Time t')
                plt.title('Value Function')

                plt.tight_layout()
                plt.show()

            else:
                print("❌ HJB solver failed")
        else:
            print("❌ FP solver failed")

    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_debug_noflux()
