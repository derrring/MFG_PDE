#!/usr/bin/env python3
"""
Debug script to understand mass loss in particle-collocation method
"""

import numpy as np

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def debug_mass_loss():
    print("=== Debugging Particle-Collocation Mass Loss ===")

    # Simple problem setup
    problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=20, T=0.1, Nt=10, sigma=1.0, coefCT=0.5)

    # Small number of collocation points and particles for debugging
    num_collocation_points = 10
    collocation_points = np.linspace(0.0, 1.0, num_collocation_points).reshape(-1, 1)
    boundary_indices = np.array([0, num_collocation_points - 1])
    no_flux_bc = BoundaryConditions(type='no_flux')

    print(f"Problem: Nx={problem.Nx}, T={problem.T}, Nt={problem.Nt}")
    print(f"Collocation points: {num_collocation_points}")
    print(f"Boundary indices: {boundary_indices}")

    # Create solver
    solver = ParticleCollocationSolver(
        problem=problem,
        collocation_points=collocation_points,
        num_particles=100,  # Small number for debugging
        delta=0.3,
        taylor_order=2,
        kde_bandwidth="scott",
        normalize_kde_output=True,
        boundary_indices=boundary_indices,
        boundary_conditions=no_flux_bc,
    )

    print("\n=== Initial Conditions ===")
    # Check initial mass distribution
    initial_density = problem.m_init
    initial_mass = np.sum(initial_density * problem.Dx)
    print(f"Initial density sum: {np.sum(initial_density):.6f}")
    print(f"Initial mass (with Dx): {initial_mass:.6f}")
    print(f"Initial density range: [{np.min(initial_density):.6f}, {np.max(initial_density):.6f}]")

    # Test just the FP solver step
    print("\n=== Testing FP Solver Step ===")
    try:
        # Test with zero velocity field (U=0 everywhere)
        U_zero = np.zeros((problem.Nt + 1, problem.Nx + 1))
        M_fp_result = solver.fp_solver.solve_fp_system(m_initial_condition=initial_density, U_solution_for_drift=U_zero)

        if M_fp_result is not None:
            print(f"FP result shape: {M_fp_result.shape}")
            mass_evolution = np.sum(M_fp_result * problem.Dx, axis=1)
            print(f"Mass evolution: initial={mass_evolution[0]:.6f}, final={mass_evolution[-1]:.6f}")
            print(f"Mass change in FP step: {mass_evolution[-1] - mass_evolution[0]:.6f}")

            # Check if particles are being lost
            if hasattr(solver.fp_solver, 'M_particles_trajectory'):
                particles = solver.fp_solver.M_particles_trajectory
                if particles is not None:
                    print(f"Particle trajectory shape: {particles.shape}")
                    print(f"Particles at t=0: min={np.min(particles[0,:]):.4f}, max={np.max(particles[0,:]):.4f}")
                    print(f"Particles at t=T: min={np.min(particles[-1,:]):.4f}, max={np.max(particles[-1,:]):.4f}")

                    # Check for particles outside domain
                    outside_count = np.sum((particles[-1, :] < 0) | (particles[-1, :] > 1))
                    print(f"Particles outside [0,1] at final time: {outside_count}")
        else:
            print("FP solver returned None!")

    except Exception as e:
        print(f"FP solver failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_mass_loss()
