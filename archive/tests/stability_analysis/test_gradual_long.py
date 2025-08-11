#!/usr/bin/env python3
"""
Gradual approach to T=1 with very conservative parameters
"""

import numpy as np

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def test_gradual_long():
    print("=== Gradual Approach to Long Simulation ===")

    # Test different time horizons to find the stability limit
    time_horizons = [0.5, 0.7, 1.0]

    for T in time_horizons:
        print(f"\n{'='*50}")
        print(f"Testing T = {T}")
        print(f"{'='*50}")

        # Scale parameters with time horizon
        Nt = max(10, int(15 * T))  # Scale time steps with T
        sigma = max(0.15, 0.3 / T)  # Smaller sigma for longer time
        coefCT = max(0.03, 0.05 / T)  # Smaller coefCT for longer time

        problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=T, Nt=Nt, sigma=sigma, coefCT=coefCT)

        # Conservative setup
        num_colloc = 10
        collocation_points = np.linspace(0.0, 1.0, num_colloc).reshape(-1, 1)
        boundary_indices = np.array([0, num_colloc - 1])
        no_flux_bc = BoundaryConditions(type='no_flux')

        print(f"Parameters: Nt={Nt}, σ={sigma:.3f}, coefCT={coefCT:.3f}")

        try:
            solver = ParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                num_particles=200,  # Small particle count
                delta=0.6,  # Large neighborhood
                taylor_order=1,  # First-order only
                weight_function="wendland",
                NiterNewton=6,  # Few Newton iterations
                l2errBoundNewton=1e-3,  # Relaxed tolerance
                kde_bandwidth="scott",
                normalize_kde_output=False,
                boundary_indices=boundary_indices,
                boundary_conditions=no_flux_bc,
            )

            U, M, info = solver.solve(Niter=4, l2errBound=2e-2, verbose=False)

            if M is not None:
                mass_evolution = np.sum(M * problem.Dx, axis=1)
                mass_change = abs(mass_evolution[-1] - mass_evolution[0])
                relative_change = mass_change / mass_evolution[0] * 100

                particles_trajectory = solver.get_particles_trajectory()
                violations = 0
                if particles_trajectory is not None:
                    final_particles = particles_trajectory[-1, :]
                    violations = np.sum((final_particles < 0) | (final_particles > 1))

                max_U = np.max(np.abs(U)) if U is not None else np.inf

                print(f"Results for T={T}:")
                print(f"  Mass change: {relative_change:.1f}%")
                print(f"  Particle violations: {violations}")
                print(f"  Max |U|: {max_U:.1e}")

                # Success criteria
                mass_ok = relative_change < 50.0
                particles_ok = violations < 50
                solution_ok = max_U < 1e4

                if mass_ok and particles_ok and solution_ok:
                    print(f"  ✓ SUCCESS: T={T} is stable!")
                    if T == 1.0:
                        print(f"  🎉 ACHIEVED T=1 STABILITY!")

                        print(f"\n--- Final T=1 Results ---")
                        print(f"Initial mass: {mass_evolution[0]:.6f}")
                        print(f"Final mass: {mass_evolution[-1]:.6f}")
                        print(f"Simulation time: T={T}")
                        print(f"Time steps: {Nt}")
                        print(f"Particles: 200")
                        print(f"Collocation points: {num_colloc}")
                        break
                else:
                    print(f"  ❌ UNSTABLE: T={T} failed")
                    if not mass_ok:
                        print(f"    - Mass loss: {relative_change:.1f}%")
                    if not particles_ok:
                        print(f"    - Particle violations: {violations}")
                    if not solution_ok:
                        print(f"    - Large solution: {max_U:.1e}")
            else:
                print(f"  ❌ SOLVER FAILED for T={T}")

        except Exception as e:
            print(f"  ❌ ERROR for T={T}: {e}")

    print(f"\n{'='*50}")
    print("Gradual testing complete")


if __name__ == "__main__":
    test_gradual_long()
