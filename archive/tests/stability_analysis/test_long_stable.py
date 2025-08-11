#!/usr/bin/env python3
"""
Conservative longer simulation with T=1 optimized for stability
"""

import numpy as np

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def test_long_stable():
    print("=== Conservative Long Simulation (T=1) ===")

    # Very conservative parameters for T=1
    problem = ExampleMFGProblem(
        xmin=0.0, xmax=1.0, Nx=40, T=1.0, Nt=20, sigma=0.2, coefCT=0.05  # Coarser for stability  # Very conservative
    )

    # Conservative collocation setup
    num_colloc = 12
    collocation_points = np.linspace(0.0, 1.0, num_colloc).reshape(-1, 1)
    boundary_indices = np.array([0, num_colloc - 1])
    no_flux_bc = BoundaryConditions(type='no_flux')

    print(f"Problem: Nx={problem.Nx}, T={problem.T}, Nt={problem.Nt}")
    print(f"Collocation: {num_colloc} points")
    print(f"Conservative parameters: σ={problem.sigma}, coefCT={problem.coefCT}")

    # Very conservative solver parameters
    solver = ParticleCollocationSolver(
        problem=problem,
        collocation_points=collocation_points,
        num_particles=250,  # Moderate particle count
        delta=0.5,  # Large neighborhood for stability
        taylor_order=1,  # First-order only
        weight_function="wendland",
        NiterNewton=8,  # Fewer Newton iterations
        l2errBoundNewton=5e-4,  # Relaxed tolerance
        kde_bandwidth="scott",
        normalize_kde_output=False,
        boundary_indices=boundary_indices,
        boundary_conditions=no_flux_bc,
    )

    print("\n--- Running Conservative Long Simulation (T=1) ---")
    U, M, info = solver.solve(Niter=6, l2errBound=1e-2, verbose=True)  # Very relaxed

    if M is not None:
        mass_evolution = np.sum(M * problem.Dx, axis=1)
        mass_change = abs(mass_evolution[-1] - mass_evolution[0])
        relative_change = mass_change / mass_evolution[0] * 100

        print(f"\n--- Conservative Long Simulation Results ---")
        print(f"Initial mass: {mass_evolution[0]:.6f}")
        print(f"Final mass: {mass_evolution[-1]:.6f}")
        print(f"Absolute mass change: {mass_change:.3e}")
        print(f"Relative mass change: {relative_change:.2f}%")

        # Check particle boundaries
        particles_trajectory = solver.get_particles_trajectory()
        if particles_trajectory is not None:
            final_particles = particles_trajectory[-1, :]
            violations = np.sum((final_particles < 0) | (final_particles > 1))
            print(f"Particle violations: {violations}")

            if violations == 0:
                print("✓ EXCELLENT: Long simulation (T=1) fully stable!")
            elif violations < 20:
                print(f"✓ VERY GOOD: Long simulation mostly stable ({violations} minor violations)")
            elif violations < 100:
                print(f"✓ ACCEPTABLE: Long simulation reasonably stable ({violations} violations)")
            else:
                print(f"⚠ POOR: Long simulation unstable ({violations} violations)")

            # Check solution quality
            max_U = np.max(np.abs(U)) if U is not None else np.inf
            print(f"Max |U|: {max_U:.2e}")

            if max_U < 1e2:
                print("✓ Solution magnitude is reasonable")
            elif max_U < 1e6:
                print("⚠ Solution magnitude is large but manageable")
            else:
                print("❌ Solution magnitude indicates instability")

    else:
        print("❌ Solver failed")


if __name__ == "__main__":
    test_long_stable()
