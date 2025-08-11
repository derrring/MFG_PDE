#!/usr/bin/env python3
"""
Final attempt at stable T=1 simulation with ultra-conservative parameters
"""

import numpy as np

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def test_final_t1():
    print("=== Final Ultra-Conservative T=1 Attempt ===")

    # Ultra-conservative parameters
    problem = ExampleMFGProblem(
        xmin=0.0, xmax=1.0, Nx=20, T=1.0, Nt=10, sigma=0.1, coefCT=0.02  # Very coarse  # Ultra-conservative
    )

    # Minimal collocation setup
    num_colloc = 8
    collocation_points = np.linspace(0.0, 1.0, num_colloc).reshape(-1, 1)
    boundary_indices = np.array([0, num_colloc - 1])
    no_flux_bc = BoundaryConditions(type='no_flux')

    print(f"Ultra-conservative setup:")
    print(f"  Nx={problem.Nx}, T={problem.T}, Nt={problem.Nt}")
    print(f"  Collocation points: {num_colloc}")
    print(f"  σ={problem.sigma}, coefCT={problem.coefCT}")

    solver = ParticleCollocationSolver(
        problem=problem,
        collocation_points=collocation_points,
        num_particles=100,  # Minimal particles
        delta=0.8,  # Very large neighborhood
        taylor_order=1,  # First-order only
        weight_function="wendland",
        NiterNewton=5,  # Minimal Newton iterations
        l2errBoundNewton=1e-3,  # Relaxed tolerance
        kde_bandwidth="scott",
        normalize_kde_output=False,
        boundary_indices=boundary_indices,
        boundary_conditions=no_flux_bc,
    )

    print("\n--- Running Ultra-Conservative T=1 Simulation ---")
    U, M, info = solver.solve(Niter=3, l2errBound=5e-2, verbose=True)  # Very relaxed

    if M is not None:
        mass_evolution = np.sum(M * problem.Dx, axis=1)
        mass_change = abs(mass_evolution[-1] - mass_evolution[0])
        relative_change = mass_change / mass_evolution[0] * 100

        print(f"\n--- Ultra-Conservative T=1 Results ---")
        print(f"Initial mass: {mass_evolution[0]:.6f}")
        print(f"Final mass: {mass_evolution[-1]:.6f}")
        print(f"Relative mass change: {relative_change:.2f}%")

        particles_trajectory = solver.get_particles_trajectory()
        if particles_trajectory is not None:
            final_particles = particles_trajectory[-1, :]
            violations = np.sum((final_particles < 0) | (final_particles > 1))
            print(f"Particle violations: {violations}")

            max_U = np.max(np.abs(U)) if U is not None else np.inf
            print(f"Max |U|: {max_U:.2e}")

            # Assess results
            if violations == 0 and relative_change < 20:
                print("✓ EXCELLENT: T=1 simulation successful!")
                print("  Perfect particle boundary compliance")
                print("  Reasonable mass conservation")
            elif violations < 10 and relative_change < 50:
                print("✓ GOOD: T=1 simulation mostly successful")
                print(f"  Minor issues: {violations} violations, {relative_change:.1f}% mass change")
            elif violations < 50 and max_U < 1e3:
                print("✓ ACCEPTABLE: T=1 simulation completed with issues")
                print(f"  Manageable problems: {violations} violations")
            else:
                print("❌ POOR: T=1 simulation still unstable")
                print("  Consider shorter time horizons or different methods")

            print(f"\nFinal T=1 Configuration:")
            print(f"  Time horizon: T={problem.T}")
            print(f"  Grid: {problem.Nx}×{problem.Nt}")
            print(f"  Particles: 100")
            print(f"  Collocation: {num_colloc} points")
            print(f"  Physics: σ={problem.sigma}, coefCT={problem.coefCT}")

    else:
        print("❌ Ultra-conservative attempt failed")
        print("T=1 may require different algorithms or problem formulation")


if __name__ == "__main__":
    test_final_t1()
