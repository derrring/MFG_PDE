#!/usr/bin/env python3
"""
Test original failing parameters with improved SVD implementation
"""

import numpy as np

from mfg_pde.alg.mfg_solvers.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.geometry import BoundaryConditions


def test_original_params_with_svd():
    print("=== Testing Original Parameters with SVD + No-Flux BC ===")

    # Original parameters that were failing
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 30,  # Original
        "T": 0.5,  # Original
        "Nt": 25,  # Original
        "sigma": 1.0,
        "coefCT": 0.5,
    }

    # Try with conservative collocation settings first
    conservative_settings = [
        {"n_colloc": 10, "particles": 200, "delta": 0.8, "taylor_order": 1, "name": "Conservative"},
        {"n_colloc": 12, "particles": 250, "delta": 0.8, "taylor_order": 1, "name": "Moderate"},
        {"n_colloc": 15, "particles": 300, "delta": 0.8, "taylor_order": 1, "name": "Original_1st"},
        {"n_colloc": 15, "particles": 300, "delta": 0.8, "taylor_order": 2, "name": "Original_2nd"},
    ]

    problem = ExampleMFGProblem(**problem_params)
    no_flux_bc = BoundaryConditions(type="no_flux")

    print(f"Problem: Nx={problem_params['Nx']}, T={problem_params['T']}, Nt={problem_params['Nt']}")
    print(f"Initial mass: {np.sum(problem.m_init * problem.Dx):.6f}")

    for i, settings in enumerate(conservative_settings):
        print(f"\n{'=' * 60}")
        print(f"TEST {i + 1}: {settings['name']}")
        print(f"Collocation: {settings['n_colloc']}, Particles: {settings['particles']}")
        print(f"Delta: {settings['delta']}, Taylor Order: {settings['taylor_order']}")
        print(f"{'=' * 60}")

        # Create collocation points
        num_collocation_points = settings["n_colloc"]
        collocation_points = np.linspace(0.0, 1.0, num_collocation_points).reshape(-1, 1)
        boundary_indices = np.array([0, num_collocation_points - 1])

        try:
            solver = ParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                num_particles=settings["particles"],
                delta=settings["delta"],
                taylor_order=settings["taylor_order"],
                weight_function="wendland",
                NiterNewton=10,
                l2errBoundNewton=1e-5,
                kde_bandwidth="scott",
                normalize_kde_output=True,
                boundary_indices=boundary_indices,
                boundary_conditions=no_flux_bc,
            )

            # Get detailed SVD diagnostics
            decomp_info = solver.hjb_solver.get_decomposition_info()
            print("SVD Diagnostics:")
            print(f"  SVD coverage: {decomp_info['svd_percentage']:.1f}%")
            print(f"  Condition number: avg={decomp_info['avg_condition_number']:.1e}")
            if decomp_info["condition_numbers"]:
                cond_min = min(decomp_info["condition_numbers"])
                cond_max = max(decomp_info["condition_numbers"])
                print(f"  Condition range: [{cond_min:.1e}, {cond_max:.1e}]")
            print(f"  Rank: [{decomp_info['min_rank']}, {decomp_info['max_rank']}]")

            # Run with limited iterations first
            print("\nRunning 5 Picard iterations...")
            time.time() if "time" in dir() else 0

            U, M, info = solver.solve(Niter=5, l2errBound=1e-4, verbose=False)

            if M is not None:
                mass_evolution = np.sum(M * problem.Dx, axis=1)
                mass_initial = mass_evolution[0]
                mass_final = mass_evolution[-1]
                mass_change = mass_final - mass_initial
                mass_variation = np.max(mass_evolution) - np.min(mass_evolution)

                max_U = np.max(np.abs(U)) if U is not None else np.inf

                print("Results:")
                print(f"  Initial mass: {mass_initial:.6f}")
                print(f"  Final mass: {mass_final:.6f}")
                print(f"  Mass change: {mass_change:.2e}")
                print(f"  Mass variation: {mass_variation:.2e}")
                print(f"  Max |U|: {max_U:.1e}")
                print(f"  Converged: {info.get('converged', False)}")
                print(f"  Iterations: {info.get('iterations', 5)}")

                # Comprehensive stability assessment
                if mass_final < 0.01:
                    status = "❌ COLLAPSED"
                elif abs(mass_change) < 1e-10 and max_U < 1e3:
                    status = "✓ EXCELLENT"
                elif abs(mass_change) < 1e-6 and max_U < 1e6:
                    status = "✓ GOOD"
                elif abs(mass_change) < 0.01 and max_U < 1e8:
                    status = "⚠ ACCEPTABLE"
                elif abs(mass_change) < 0.1 and max_U < 1e10:
                    status = "⚠ POOR"
                else:
                    status = "❌ UNSTABLE"

                print(f"  STATUS: {status}")

                # Check particle boundary behavior
                particles_trajectory = solver.fp_solver.M_particles_trajectory
                if particles_trajectory is not None:
                    final_particles = particles_trajectory[-1, :]
                    outside_bounds = np.sum((final_particles < 0) | (final_particles > 1))
                    print(f"  Particle violations: {outside_bounds}")

            else:
                print("  ❌ FAILED: Solution is None")

        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("ORIGINAL PARAMETERS TEST COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import time

    test_original_params_with_svd()
