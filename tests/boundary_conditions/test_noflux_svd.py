#!/usr/bin/env python3
"""
Test no-flux boundary conditions with SVD diagnostics
"""

import numpy as np

from mfg_pde.alg.numerical.mfg_solvers.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.geometry import BoundaryConditions


def test_noflux_with_svd():
    print("=== Testing No-Flux BC with SVD Diagnostics ===")

    # Progressive testing: start very simple and scale up
    test_cases = [
        {"Nx": 5, "T": 0.01, "Nt": 2, "n_colloc": 3, "particles": 50, "name": "Minimal"},
        {"Nx": 10, "T": 0.05, "Nt": 5, "n_colloc": 5, "particles": 100, "name": "Small"},
        {"Nx": 20, "T": 0.1, "Nt": 10, "n_colloc": 8, "particles": 200, "name": "Medium"},
    ]

    results = {}

    for i, case in enumerate(test_cases):
        print(f"\n{'=' * 60}")
        print(f"TEST CASE {i + 1}: {case['name']}")
        print(
            f"Nx={case['Nx']}, T={case['T']}, Nt={case['Nt']}, Colloc={case['n_colloc']}, Particles={case['particles']}"
        )
        print(f"{'=' * 60}")

        # Create problem
        problem = ExampleMFGProblem(
            xmin=0.0, xmax=1.0, Nx=case["Nx"], T=case["T"], Nt=case["Nt"], sigma=0.5, coefCT=0.1
        )

        # Create collocation points
        num_collocation_points = case["n_colloc"]
        collocation_points = np.linspace(0.0, 1.0, num_collocation_points).reshape(-1, 1)
        boundary_indices = np.array([0, num_collocation_points - 1])
        no_flux_bc = BoundaryConditions(type="no_flux")

        try:
            # Create solver
            solver = ParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                num_particles=case["particles"],
                delta=0.8,  # Large delta for stability
                taylor_order=1,  # First order for stability
                weight_function="wendland",
                NiterNewton=10,
                l2errBoundNewton=1e-4,
                kde_bandwidth="scott",
                normalize_kde_output=True,
                boundary_indices=boundary_indices,
                boundary_conditions=no_flux_bc,
            )

            # Get SVD diagnostics from HJB solver
            hjb_solver = solver.hjb_solver
            decomp_info = hjb_solver.get_decomposition_info()

            print("SVD Diagnostics:")
            print(
                f"  SVD points: {decomp_info['svd_points']}/{decomp_info['total_points']} ({decomp_info['svd_percentage']:.1f}%)"
            )
            print(f"  Avg condition number: {decomp_info['avg_condition_number']:.2e}")
            print(f"  Rank range: [{decomp_info['min_rank']}, {decomp_info['max_rank']}]")

            # Run very limited Picard iterations
            print("\nRunning 2 Picard iterations...")
            U, M, _info = solver.solve(Niter=2, l2errBound=1e-3, verbose=True)

            if M is not None:
                mass_evolution = np.sum(M * problem.Dx, axis=1)
                print("\nMass Analysis:")
                print(f"  Initial mass: {mass_evolution[0]:.6f}")
                print(f"  Final mass: {mass_evolution[-1]:.6f}")
                print(f"  Mass change: {mass_evolution[-1] - mass_evolution[0]:.6f}")

                # Check if solution collapsed
                if mass_evolution[-1] < 0.01:
                    status = "COLLAPSED"
                    print(f"  ❌ STATUS: {status}")
                elif abs(mass_evolution[-1] - mass_evolution[0]) < 0.1:
                    status = "STABLE"
                    print(f"  ✓ STATUS: {status}")
                else:
                    status = "UNSTABLE"
                    print(f"  ⚠ STATUS: {status}")

                # Check U values
                max_U = np.max(np.abs(U)) if U is not None else np.inf
                print(f"  Max |U|: {max_U:.2e}")

                if max_U > 1e10:
                    print("  ❌ HJB: Extreme values")
                elif max_U > 1e3:
                    print("  ⚠ HJB: Large values")
                else:
                    print("  ✓ HJB: Reasonable values")

                results[case["name"]] = {
                    "status": status,
                    "mass_change": mass_evolution[-1] - mass_evolution[0],
                    "max_U": max_U,
                    "svd_percentage": decomp_info["svd_percentage"],
                    "condition_number": decomp_info["avg_condition_number"],
                }

            else:
                print("  ❌ STATUS: FAILED (M is None)")
                results[case["name"]] = {"status": "FAILED"}

        except Exception as e:
            print(f"❌ ERROR: {e}")
            results[case["name"]] = {"status": "ERROR", "error": str(e)}

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Case':<10} {'Status':<10} {'Mass Change':<12} {'Max |U|':<10} {'SVD %':<8}")
    print(f"{'-' * 10} {'-' * 10} {'-' * 12} {'-' * 10} {'-' * 8}")

    for name, result in results.items():
        if result["status"] not in ["FAILED", "ERROR"]:
            mass_change = f"{result['mass_change']:.2e}"
            max_U = f"{result['max_U']:.1e}"
            svd_pct = f"{result['svd_percentage']:.0f}%"
        else:
            mass_change = "-"
            max_U = "-"
            svd_pct = "-"

        print(f"{name:<10} {result['status']:<10} {mass_change:<12} {max_U:<10} {svd_pct:<8}")

    # Find the largest stable case
    stable_cases = [name for name, result in results.items() if result.get("status") == "STABLE"]
    if stable_cases:
        print(f"\n✓ Largest stable case: {stable_cases[-1]}")
    else:
        print("\n❌ No stable cases found")


if __name__ == "__main__":
    test_noflux_with_svd()
