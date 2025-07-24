#!/usr/bin/env python3
"""
Quick QP Validation Test
Quick test to validate the tuned QP optimization approaches.
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.alg.hjb_solvers.smart_qp_gfdm_hjb import SmartQPGFDMHJBSolver
from mfg_pde.alg.hjb_solvers.tuned_smart_qp_gfdm_hjb import TunedSmartQPGFDMHJBSolver
from mfg_pde.core.boundaries import BoundaryConditions

def quick_qp_validation():
    """Quick validation of Smart and Tuned Smart QP approaches"""
    print("="*60)
    print("QUICK QP OPTIMIZATION VALIDATION")
    print("="*60)
    
    # Smaller problem for quick testing
    problem_params = {
        'xmin': 0.0, 'xmax': 1.0, 'Nx': 15, 'T': 0.5, 'Nt': 20,
        'sigma': 0.1, 'coefCT': 0.015
    }
    
    print(f"Problem: Nx={problem_params['Nx']}, T={problem_params['T']}, Nt={problem_params['Nt']}")
    
    results = {}
    
    # Test 1: Smart QP
    print(f"\n{'-'*40}")
    print("TESTING SMART QP")
    print(f"{'-'*40}")
    
    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")
        
        num_collocation_points = 8
        collocation_points = np.linspace(
            problem.xmin, problem.xmax, num_collocation_points
        ).reshape(-1, 1)
        boundary_indices = [0, num_collocation_points-1]
        
        smart_hjb_solver = SmartQPGFDMHJBSolver(
            problem=problem,
            collocation_points=collocation_points,
            delta=0.4,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=3,
            l2errBoundNewton=1e-3,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
            qp_usage_target=0.1
        )
        
        smart_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=100,
            delta=0.4,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=3,
            l2errBoundNewton=1e-3,
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True
        )
        smart_solver.hjb_solver = smart_hjb_solver
        
        print("Running Smart QP solver...")
        start_time = time.time()
        U_smart, M_smart, info_smart = smart_solver.solve(Niter=4, l2errBound=1e-3, verbose=True)
        smart_time = time.time() - start_time
        
        # Get statistics
        smart_stats = smart_hjb_solver.get_smart_qp_report()
        
        results['smart'] = {
            'success': True,
            'time': smart_time,
            'qp_usage_rate': smart_stats.get('qp_usage_rate', 1.0),
            'stats': smart_stats
        }
        
        print(f"✓ Smart QP: {smart_time:.1f}s, QP usage: {smart_stats.get('qp_usage_rate', 0):.1%}")
        
    except Exception as e:
        print(f"✗ Smart QP failed: {e}")
        results['smart'] = {'success': False, 'error': str(e)}
    
    # Test 2: Tuned Smart QP
    print(f"\n{'-'*40}")
    print("TESTING TUNED SMART QP")
    print(f"{'-'*40}")
    
    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")
        
        collocation_points = np.linspace(
            problem.xmin, problem.xmax, num_collocation_points
        ).reshape(-1, 1)
        boundary_indices = [0, num_collocation_points-1]
        
        tuned_hjb_solver = TunedSmartQPGFDMHJBSolver(
            problem=problem,
            collocation_points=collocation_points,
            delta=0.4,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=3,
            l2errBoundNewton=1e-3,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
            qp_usage_target=0.1
        )
        
        tuned_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=100,
            delta=0.4,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=3,
            l2errBoundNewton=1e-3,
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True
        )
        tuned_solver.hjb_solver = tuned_hjb_solver
        
        print("Running Tuned Smart QP solver...")
        start_time = time.time()
        U_tuned, M_tuned, info_tuned = tuned_solver.solve(Niter=4, l2errBound=1e-3, verbose=True)
        tuned_time = time.time() - start_time
        
        # Get statistics
        tuned_stats = tuned_hjb_solver.get_tuned_qp_report()
        
        results['tuned'] = {
            'success': True,
            'time': tuned_time,
            'qp_usage_rate': tuned_stats.get('qp_usage_rate', 1.0),
            'stats': tuned_stats
        }
        
        print(f"✓ Tuned Smart QP: {tuned_time:.1f}s, QP usage: {tuned_stats.get('qp_usage_rate', 0):.1%}")
        
        # Print detailed summary
        if hasattr(tuned_hjb_solver, 'print_tuned_qp_summary'):
            tuned_hjb_solver.print_tuned_qp_summary()
        
    except Exception as e:
        print(f"✗ Tuned Smart QP failed: {e}")
        import traceback
        traceback.print_exc()
        results['tuned'] = {'success': False, 'error': str(e)}
    
    # Summary
    print(f"\n{'='*60}")
    print("QUICK VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    print(f"{'Method':<20} {'Success':<8} {'Time(s)':<10} {'QP Usage':<12} {'Status':<15}")
    print("-" * 70)
    
    for key, result in results.items():
        if result['success']:
            success_str = "✓"
            time_str = f"{result['time']:.1f}"
            qp_usage_str = f"{result['qp_usage_rate']:.1%}"
            
            # Status assessment
            qp_rate = result['qp_usage_rate']
            if qp_rate <= 0.12:  # Within 20% of 10% target
                status = "EXCELLENT"
            elif qp_rate <= 0.2:  # Within 100% of target
                status = "GOOD"
            elif qp_rate <= 0.4:  # Some improvement
                status = "FAIR"
            else:
                status = "POOR"
        else:
            success_str = "✗"
            time_str = "FAILED"
            qp_usage_str = "N/A"
            status = "FAILED"
        
        method_name = key.title().replace('_', ' ') + " QP"
        print(f"{method_name:<20} {success_str:<8} {time_str:<10} {qp_usage_str:<12} {status:<15}")
    
    # Assessment
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if len(successful_results) > 0:
        print(f"\nOVERALL ASSESSMENT:")
        print("-" * 30)
        
        best_qp_rate = 1.0
        best_method = None
        
        for key, result in successful_results.items():
            if result['qp_usage_rate'] < best_qp_rate:
                best_qp_rate = result['qp_usage_rate']
                best_method = key
        
        if best_qp_rate <= 0.12:
            print("✓ OPTIMIZATION TARGET ACHIEVED")
            print(f"  Best method: {best_method} with {best_qp_rate:.1%} QP usage")
        elif best_qp_rate <= 0.2:
            print("⚠️ CLOSE TO OPTIMIZATION TARGET")
            print(f"  Best method: {best_method} with {best_qp_rate:.1%} QP usage")
        else:
            print("❌ OPTIMIZATION TARGET NOT REACHED")
            print(f"  Best method: {best_method} with {best_qp_rate:.1%} QP usage")
            print("  Further tuning needed")
    
    return results

if __name__ == "__main__":
    results = quick_qp_validation()