#!/usr/bin/env python3
"""
Test to demonstrate the initialization improvement with stable parameters
"""

import numpy as np

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def test_initialization_improvement():
    print("=== Testing Initialization Improvement ===")

    # Use stable parameters adapted for longer simulation T=1
    problem = ExampleMFGProblem(
        xmin=0.0,
        xmax=1.0,
        Nx=50,
        T=1.0,
        Nt=25,  # Longer time with moderate temporal resolution
        sigma=0.25,
        coefCT=0.08,  # More conservative for longer simulation
    )

    # Stable collocation setup
    num_colloc = 15
    collocation_points = np.linspace(0.0, 1.0, num_colloc).reshape(-1, 1)
    boundary_indices = np.array([0, num_colloc - 1])
    no_flux_bc = BoundaryConditions(type='no_flux')

    print(f"Problem: Nx={problem.Nx}, T={problem.T}, Nt={problem.Nt}")
    print(f"Collocation: {num_colloc} points")

    # Create solver with parameters adapted for longer simulation
    solver = ParticleCollocationSolver(
        problem=problem,
        collocation_points=collocation_points,
        num_particles=350,  # Slightly fewer particles for longer simulation
        delta=0.4,  # Larger neighborhood for stability over longer time
        taylor_order=1,  # First-order for guaranteed stability
        weight_function="wendland",
        NiterNewton=10,  # Conservative Newton iterations
        l2errBoundNewton=2e-4,  # Slightly relaxed tolerance
        kde_bandwidth="scott",
        normalize_kde_output=False,
        boundary_indices=boundary_indices,
        boundary_conditions=no_flux_bc,
    )

    print("\n--- Running Longer Simulation (T=1) with Better Initialization ---")
    U, M, info = solver.solve(
        Niter=8, l2errBound=5e-3, verbose=True
    )  # More iterations and relaxed tolerance for longer sim

    if M is not None:
        mass_evolution = np.sum(M * problem.Dx, axis=1)
        mass_change = abs(mass_evolution[-1] - mass_evolution[0])

        print(f"\n--- Results with Longer Simulation (T=1) ---")
        print(f"Mass change: {mass_change:.2e}")
        print(f"Initial mass: {mass_evolution[0]:.6f}")
        print(f"Final mass: {mass_evolution[-1]:.6f}")

        # Check particle boundaries
        particles_trajectory = solver.get_particles_trajectory()
        if particles_trajectory is not None:
            final_particles = particles_trajectory[-1, :]
            violations = np.sum((final_particles < 0) | (final_particles > 1))
            print(f"Particle violations: {violations}")

            if violations == 0:
                print("✓ EXCELLENT: Longer simulation (T=1) successful with no-flux boundaries!")
            elif violations < 50:
                print(f"✓ GOOD: Longer simulation mostly stable with {violations} minor violations")
            else:
                print(f"⚠ Issues with longer simulation: {violations} violations")

    else:
        print("❌ Solver failed")


if __name__ == "__main__":
    test_initialization_improvement()
