#!/usr/bin/env python3
"""
Test scaling up no-flux BC with SVD to larger problems
"""

import numpy as np

from mfg_pde.alg.mfg_solvers.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.geometry import BoundaryConditions


def test_scaling_noflux():
    print("=== Testing Scaling of No-Flux BC with SVD ===")

    # Known stable case: Medium from previous test
    base_case = {"Nx": 20, "T": 0.1, "Nt": 10, "n_colloc": 8, "particles": 200}

    # Scaling test cases - gradually increase complexity
    scaling_tests = [
        {"Nx": 20, "T": 0.2, "Nt": 15, "n_colloc": 8, "particles": 200, "name": "Longer_Time"},
        {"Nx": 30, "T": 0.1, "Nt": 10, "n_colloc": 10, "particles": 300, "name": "More_Grid"},
        {"Nx": 20, "T": 0.1, "Nt": 10, "n_colloc": 12, "particles": 200, "name": "More_Colloc"},
        {"Nx": 30, "T": 0.2, "Nt": 15, "n_colloc": 12, "particles": 300, "name": "Combined"},
    ]

    for i, case in enumerate(scaling_tests):
        print(f"\n{'='*50}")
        print(f"SCALING TEST {i+1}: {case['name']}")
        print(f"Nx={case['Nx']}, T={case['T']}, Nt={case['Nt']}, Colloc={case['n_colloc']}")
        print(f"{'='*50}")

        # Create problem
        problem = ExampleMFGProblem(
            xmin=0.0, xmax=1.0, Nx=case['Nx'], T=case['T'], Nt=case['Nt'], sigma=0.5, coefCT=0.1
        )

        # Create collocation points
        num_collocation_points = case['n_colloc']
        collocation_points = np.linspace(0.0, 1.0, num_collocation_points).reshape(-1, 1)
        boundary_indices = np.array([0, num_collocation_points - 1])
        no_flux_bc = BoundaryConditions(type='no_flux')

        try:
            # Create solver with conservative settings
            solver = ParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                num_particles=case['particles'],
                delta=0.8,  # Keep large delta for stability
                taylor_order=1,  # Keep first order for stability
                weight_function="wendland",
                NiterNewton=8,
                l2errBoundNewton=1e-4,
                kde_bandwidth="scott",
                normalize_kde_output=True,
                boundary_indices=boundary_indices,
                boundary_conditions=no_flux_bc,
            )

            # Get SVD diagnostics
            decomp_info = solver.hjb_solver.get_decomposition_info()
            print(f"SVD: {decomp_info['svd_percentage']:.0f}%, κ_avg={decomp_info['avg_condition_number']:.1e}")

            # Run 3 Picard iterations
            U, M, info = solver.solve(Niter=3, l2errBound=1e-3, verbose=False)

            if M is not None:
                mass_evolution = np.sum(M * problem.Dx, axis=1)
                mass_change = mass_evolution[-1] - mass_evolution[0]
                max_U = np.max(np.abs(U)) if U is not None else np.inf

                print(f"Results:")
                print(f"  Mass change: {mass_change:.2e}")
                print(f"  Max |U|: {max_U:.1e}")
                print(f"  Convergence: {info.get('converged', False)}")

                # Stability assessment
                if abs(mass_change) < 1e-10 and max_U < 1e3:
                    status = "✓ EXCELLENT"
                elif abs(mass_change) < 1e-6 and max_U < 1e6:
                    status = "✓ GOOD"
                elif abs(mass_change) < 0.1 and max_U < 1e10:
                    status = "⚠ ACCEPTABLE"
                else:
                    status = "❌ UNSTABLE"

                print(f"  Status: {status}")

            else:
                print(f"  ❌ FAILED: M is None")

        except Exception as e:
            print(f"❌ ERROR: {e}")

    print(f"\n{'='*50}")
    print(f"SCALING ANALYSIS COMPLETE")
    print(f"{'='*50}")


if __name__ == "__main__":
    test_scaling_noflux()
