#!/usr/bin/env python3
"""
Test different weight functions in GFDM solver
"""

import numpy as np
from mfg_pde.alg.hjb_solvers.hjb_gfdm import HJBGFDMSolver as GFDMHJBSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.geometry import BoundaryConditions

def test_weight_functions():
    print("=== Testing Different Weight Functions in GFDM ===")
    
    # Simple problem
    problem = ExampleMFGProblem(
        xmin=0.0, xmax=1.0, Nx=10, T=0.02, Nt=2, sigma=0.1, coefCT=0.1
    )
    
    num_collocation_points = 5
    collocation_points = np.linspace(0.0, 1.0, num_collocation_points).reshape(-1, 1)
    boundary_indices = np.array([0, num_collocation_points-1])
    no_flux_bc = BoundaryConditions(type='no_flux')
    
    # Test data
    M_simple = np.ones((problem.Nt + 1, problem.Nx + 1)) * 0.5
    U_terminal = np.zeros(problem.Nx + 1)
    U_initial = np.zeros((problem.Nt + 1, problem.Nx + 1))
    
    weight_functions = ["uniform", "gaussian", "wendland", "inverse_distance"]
    results = {}
    
    for weight_func in weight_functions:
        print(f"\n=== Testing {weight_func} weight function ===")
        
        try:
            hjb_solver = GFDMHJBSolver(
                problem=problem,
                collocation_points=collocation_points,
                delta=0.8,
                taylor_order=1,
                weight_function=weight_func,
                NiterNewton=5,
                l2errBoundNewton=1e-3,
                boundary_indices=boundary_indices,
                boundary_conditions=no_flux_bc
            )
            
            U_solution = hjb_solver.solve_hjb_system(
                M_density_evolution_from_FP=M_simple,
                U_final_condition_at_T=U_terminal,
                U_from_prev_picard=U_initial
            )
            
            max_val = np.max(np.abs(U_solution))
            results[weight_func] = max_val
            
            print(f"  Success: Max |U| = {max_val:.3f}")
            
            if np.any(np.isnan(U_solution)) or np.any(np.isinf(U_solution)):
                print(f"  ERROR: Contains NaN or Inf")
            elif max_val > 1e6:
                print(f"  WARNING: Extreme values")
            else:
                print(f"  OK: Reasonable values")
                
        except Exception as e:
            print(f"  FAILED: {e}")
            results[weight_func] = None
    
    print(f"\n=== Weight Function Comparison ===")
    print(f"{'Function':<15} {'Max |U|':<10} {'Status'}")
    print(f"{'-'*15} {'-'*10} {'-'*10}")
    
    for func, max_val in results.items():
        if max_val is not None:
            status = "OK" if max_val < 100 else "High" if max_val < 1e6 else "Extreme"
            print(f"{func:<15} {max_val:<10.3f} {status}")
        else:
            print(f"{func:<15} {'FAILED':<10} {'ERROR'}")
    
    # Find the best weight function (lowest max value)
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_func = min(valid_results.keys(), key=lambda k: valid_results[k])
        print(f"\nBest weight function: {best_func} (Max |U| = {valid_results[best_func]:.3f})")

if __name__ == "__main__":
    test_weight_functions()