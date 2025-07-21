#!/usr/bin/env python3
"""
Test second-order Taylor expansion with SVD and no-flux BC
"""

import numpy as np
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.core.boundaries import BoundaryConditions

def test_second_order_svd():
    print("=== Testing Second-Order Taylor with SVD + No-Flux BC ===")
    
    # Start with known stable case and add second-order
    problem = ExampleMFGProblem(
        xmin=0.0, xmax=1.0, Nx=20, T=0.1, Nt=10, sigma=0.5, coefCT=0.1
    )
    
    num_collocation_points = 8
    collocation_points = np.linspace(0.0, 1.0, num_collocation_points).reshape(-1, 1)
    boundary_indices = np.array([0, num_collocation_points-1])
    no_flux_bc = BoundaryConditions(type='no_flux')
    
    # Test with different delta values for second-order
    delta_values = [0.9, 0.8, 0.7, 0.6]
    
    for delta in delta_values:
        print(f"\n{'='*50}")
        print(f"Testing Second-Order with delta = {delta}")
        print(f"{'='*50}")
        
        try:
            solver = ParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                num_particles=200,
                delta=delta,
                taylor_order=2,  # Second-order!
                weight_function="wendland",
                NiterNewton=8,
                l2errBoundNewton=1e-4,
                kde_bandwidth="scott",
                normalize_kde_output=True,
                boundary_indices=boundary_indices,
                boundary_conditions=no_flux_bc
            )
            
            # Get SVD diagnostics
            decomp_info = solver.hjb_solver.get_decomposition_info()
            print(f"SVD Diagnostics:")
            print(f"  SVD points: {decomp_info['svd_points']}/{decomp_info['total_points']} ({decomp_info['svd_percentage']:.1f}%)")
            print(f"  Avg condition number: {decomp_info['avg_condition_number']:.1e}")
            print(f"  Rank range: [{decomp_info['min_rank']}, {decomp_info['max_rank']}]")
            
            if decomp_info['condition_numbers']:
                print(f"  Condition number range: [{min(decomp_info['condition_numbers']):.1e}, {max(decomp_info['condition_numbers']):.1e}]")
            
            # Run solver
            U, M, info = solver.solve(Niter=3, l2errBound=1e-3, verbose=False)
            
            if M is not None:
                mass_evolution = np.sum(M * problem.Dx, axis=1)
                mass_change = mass_evolution[-1] - mass_evolution[0]
                max_U = np.max(np.abs(U)) if U is not None else np.inf
                
                print(f"Results:")
                print(f"  Mass change: {mass_change:.2e}")
                print(f"  Max |U|: {max_U:.1e}")
                print(f"  Converged: {info.get('converged', False)}")
                
                # Stability assessment
                if abs(mass_change) < 1e-10 and max_U < 1e3:
                    status = "✓ EXCELLENT"
                    stability_score = 5
                elif abs(mass_change) < 1e-6 and max_U < 1e6:
                    status = "✓ GOOD"
                    stability_score = 4
                elif abs(mass_change) < 0.01 and max_U < 1e8:
                    status = "⚠ ACCEPTABLE"
                    stability_score = 3
                elif abs(mass_change) < 0.1 and max_U < 1e10:
                    status = "⚠ POOR"
                    stability_score = 2
                else:
                    status = "❌ UNSTABLE"
                    stability_score = 1
                
                print(f"  Status: {status}")
                
                # Test derivative accuracy with second-order
                print(f"Testing derivative accuracy:")
                u_test = np.sin(np.pi * collocation_points.flatten())
                mid_idx = num_collocation_points // 2
                derivatives = solver.hjb_solver.approximate_derivatives(u_test, mid_idx)
                
                x_mid = collocation_points[mid_idx, 0]
                analytical_second = -np.pi**2 * np.sin(np.pi * x_mid)
                
                if (2,) in derivatives:
                    numerical_second = derivatives[(2,)]
                    error_second = abs(numerical_second - analytical_second)
                    print(f"  Second derivative error: {error_second:.2e}")
                
            else:
                print(f"  ❌ FAILED: M is None")
                status = "❌ FAILED"
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            status = "❌ ERROR"
    
    print(f"\n{'='*50}")
    print(f"SECOND-ORDER ANALYSIS COMPLETE")
    print(f"{'='*50}")
    print(f"Recommendation: Use delta >= 0.8 for second-order Taylor with SVD")

if __name__ == "__main__":
    test_second_order_svd()