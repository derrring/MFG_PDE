#!/usr/bin/env python3
"""
Comprehensive test of ghost particle no-flux implementation
"""

import numpy as np

from mfg_pde.alg.numerical.mfg_solvers.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.geometry import BoundaryConditions


def test_comprehensive_noflux():
    print("=== Comprehensive Ghost Particle No-Flux Test ===")

    # Test cases with increasing complexity
    test_cases = [
        {"Nx": 8, "T": 0.05, "Nt": 3, "n_colloc": 5, "particles": 100, "name": "Simple"},
        {"Nx": 15, "T": 0.1, "Nt": 8, "n_colloc": 7, "particles": 150, "name": "Medium"},
        {"Nx": 20, "T": 0.15, "Nt": 12, "n_colloc": 8, "particles": 200, "name": "Large"},
        {"Nx": 25, "T": 0.2, "Nt": 15, "n_colloc": 10, "particles": 250, "name": "XLarge"},
    ]

    no_flux_bc = BoundaryConditions(type="no_flux")
    results = {}

    for i, case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"TEST {i+1}: {case['name']}")
        print(f"Nx={case['Nx']}, T={case['T']}, Nt={case['Nt']}, Colloc={case['n_colloc']}")
        print(f"{'='*60}")

        # Create problem
        problem = ExampleMFGProblem(
            xmin=0.0, xmax=1.0, Nx=case["Nx"], T=case["T"], Nt=case["Nt"], sigma=0.3, coefCT=0.1
        )

        # Create collocation points
        num_collocation_points = case["n_colloc"]
        collocation_points = np.linspace(0.0, 1.0, num_collocation_points).reshape(-1, 1)
        boundary_indices = np.array([0, num_collocation_points - 1])

        try:
            solver = ParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                num_particles=case["particles"],
                delta=0.8,
                taylor_order=1,  # Keep first order for stability
                weight_function="wendland",
                NiterNewton=8,
                l2errBoundNewton=1e-4,
                kde_bandwidth="scott",
                normalize_kde_output=True,
                boundary_indices=boundary_indices,
                boundary_conditions=no_flux_bc,
            )

            # Get diagnostics
            decomp_info = solver.hjb_solver.get_decomposition_info()

            # Check ghost particle structure
            boundary_info = []
            for b_idx in [0, num_collocation_points - 1]:
                neighborhood = solver.hjb_solver.neighborhoods[b_idx]
                boundary_info.append(
                    {"index": b_idx, "ghosts": neighborhood["ghost_count"], "total_neighbors": neighborhood["size"]}
                )

            print(f"SVD: {decomp_info['svd_percentage']:.0f}%, κ={decomp_info['avg_condition_number']:.1e}")
            print(
                f"Ghosts: [{boundary_info[0]['ghosts']}, {boundary_info[1]['ghosts']}], Neighbors: [{boundary_info[0]['total_neighbors']}, {boundary_info[1]['total_neighbors']}]"
            )

            # Run solver with limited iterations
            U, M, info = solver.solve(Niter=3, l2errBound=1e-3, verbose=False)

            if M is not None:
                mass_evolution = np.sum(M * problem.Dx, axis=1)
                mass_initial = mass_evolution[0]
                mass_final = mass_evolution[-1]
                mass_change = mass_final - mass_initial
                mass_variation = np.max(mass_evolution) - np.min(mass_evolution)

                max_U = np.max(np.abs(U)) if U is not None else np.inf
                converged = info.get("converged", False)

                print("Results:")
                print(f"  Initial mass: {mass_initial:.6f}")
                print(f"  Final mass: {mass_final:.6f}")
                print(f"  Mass change: {mass_change:.2e}")
                print(f"  Mass variation: {mass_variation:.2e}")
                print(f"  Max |U|: {max_U:.1e}")
                print(f"  Converged: {converged}")

                # Stability assessment
                if mass_final < 0.01:
                    status = "❌ COLLAPSED"
                    score = 0
                elif abs(mass_change) < 1e-10 and max_U < 1e2:
                    status = "✓ EXCELLENT"
                    score = 5
                elif abs(mass_change) < 1e-6 and max_U < 1e3:
                    status = "✓ VERY_GOOD"
                    score = 4
                elif abs(mass_change) < 1e-3 and max_U < 1e4:
                    status = "✓ GOOD"
                    score = 3
                elif abs(mass_change) < 0.01 and max_U < 1e6:
                    status = "⚠ ACCEPTABLE"
                    score = 2
                elif mass_final > 0.5:
                    status = "⚠ POOR"
                    score = 1
                else:
                    status = "❌ UNSTABLE"
                    score = 0

                print(f"  STATUS: {status}")

                # Check particle boundary violations
                particles_trajectory = solver.fp_solver.M_particles_trajectory
                if particles_trajectory is not None:
                    final_particles = particles_trajectory[-1, :]
                    violations = np.sum((final_particles < 0) | (final_particles > 1))
                    print(f"  Particle violations: {violations}")
                else:
                    violations = "N/A"

                results[case["name"]] = {
                    "status": status,
                    "score": score,
                    "mass_change": mass_change,
                    "max_U": max_U,
                    "violations": violations,
                    "svd_percentage": decomp_info["svd_percentage"],
                    "condition_number": decomp_info["avg_condition_number"],
                }

            else:
                print("  ❌ FAILED: M is None")
                results[case["name"]] = {"status": "❌ FAILED", "score": 0}

        except Exception as e:
            print(f"❌ ERROR: {e}")
            results[case["name"]] = {"status": "❌ ERROR", "score": 0}

    # Summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Case':<10} {'Status':<12} {'Mass Δ':<10} {'Max |U|':<8} {'SVD%':<6} {'Score'}")
    print(f"{'-'*10} {'-'*12} {'-'*10} {'-'*8} {'-'*6} {'-'*5}")

    total_score = 0
    max_score = 0

    for name, result in results.items():
        if result.get("score", 0) > 0:
            mass_change = f"{result['mass_change']:.1e}"
            max_U = f"{result['max_U']:.1e}"
            svd_pct = f"{result['svd_percentage']:.0f}%"
            score = result["score"]
        else:
            mass_change = "-"
            max_U = "-"
            svd_pct = "-"
            score = 0

        total_score += score
        max_score += 5

        print(f"{name:<10} {result['status']:<12} {mass_change:<10} {max_U:<8} {svd_pct:<6} {score}/5")

    # Overall assessment
    success_rate = total_score / max_score * 100 if max_score > 0 else 0

    print(f"\n{'='*80}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*80}")
    print(f"Success Rate: {success_rate:.1f}% ({total_score}/{max_score})")

    if success_rate >= 80:
        print("✓ EXCELLENT: Ghost particle no-flux implementation is highly successful!")
    elif success_rate >= 60:
        print("✓ GOOD: Ghost particle implementation works well for most cases")
    elif success_rate >= 40:
        print("⚠ FAIR: Ghost particle implementation has mixed results")
    else:
        print("❌ POOR: Ghost particle implementation needs significant improvement")


if __name__ == "__main__":
    test_comprehensive_noflux()
