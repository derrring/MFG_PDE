#!/usr/bin/env python3
"""
Comprehensive Comparison: All Methods with No-Flux Boundary Conditions
====================================================================

This script compares mass conservation across all implemented methods:
1. Pure FDM with no-flux boundaries
2. Hybrid Particle-FDM with no-flux boundaries  
3. Particle-Collocation with no-flux boundaries

Focus: Mass conservation behavior with no-flux boundaries
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.alg.hjb_solvers import HJBFDMSolver
from mfg_pde.alg.fp_solvers.fdm_fp import FdmFPSolver
from mfg_pde.alg.fp_solvers.particle_fp import ParticleFPSolver
from mfg_pde.alg.damped_fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions


def run_method_comparison():
    """Compare all methods with no-flux boundary conditions"""
    print("=" * 80)
    print("    COMPREHENSIVE NO-FLUX BOUNDARY CONDITIONS COMPARISON")
    print("=" * 80)
    
    # Common problem parameters
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 30,  # Smaller grid for faster comparison
        "T": 0.5,   # Shorter time for stability
        "Nt": 25,   # Fewer time steps
        "sigma": 1.0,
        "coefCT": 0.5,
    }
    
    # Common solver parameters
    max_iterations = 15  # Reduced for faster comparison
    convergence_tolerance = 1e-4  # Relaxed tolerance
    damping_factor = 0.5
    num_particles = 300  # Smaller for faster computation

    print(f"Problem Parameters: Nx={problem_params['Nx']}, T={problem_params['T']}, Nt={problem_params['Nt']}")
    print(f"Solver Parameters: max_iter={max_iterations}, tol={convergence_tolerance}")
    
    # Create common problem
    mfg_problem = ExampleMFGProblem(**problem_params)
    no_flux_bc = BoundaryConditions(type='no_flux')
    
    results = {}
    
    # ========================================================================
    # METHOD 1: Pure FDM with No-Flux Boundaries
    # ========================================================================
    print(f"\n" + "="*60)
    print(f"METHOD 1: Pure FDM with No-Flux Boundaries")
    print(f"="*60)
    
    try:
        start_time = time.time()
        
        hjb_solver = FdmHJBSolver(mfg_problem, NiterNewton=15, l2errBoundNewton=1e-6)
        fp_solver = FdmFPSolver(mfg_problem, boundary_conditions=no_flux_bc)
        
        fdm_iterator = FixedPointIterator(
            mfg_problem, hjb_solver=hjb_solver, fp_solver=fp_solver, thetaUM=damping_factor
        )
        
        U_fdm, M_fdm, iters_fdm, _, _ = fdm_iterator.solve(max_iterations, convergence_tolerance)
        solve_time_fdm = time.time() - start_time
        
        if U_fdm is not None and M_fdm is not None:
            total_mass_fdm = np.sum(M_fdm * mfg_problem.Dx, axis=1)
            results['FDM'] = {
                'mass_initial': total_mass_fdm[0],
                'mass_final': total_mass_fdm[-1],
                'mass_change': total_mass_fdm[-1] - total_mass_fdm[0],
                'mass_variation': np.max(total_mass_fdm) - np.min(total_mass_fdm),
                'iterations': iters_fdm,
                'time': solve_time_fdm,
                'total_mass': total_mass_fdm,
                'status': 'SUCCESS'
            }
            print(f"✓ FDM: {iters_fdm} iters, {solve_time_fdm:.2f}s")
            print(f"  Mass change: {results['FDM']['mass_change']:.2e}")
        else:
            results['FDM'] = {'status': 'FAILED'}
            print(f"❌ FDM: Failed to converge")
            
    except Exception as e:
        results['FDM'] = {'status': 'ERROR', 'error': str(e)}
        print(f"❌ FDM: Error - {e}")
    
    # ========================================================================
    # METHOD 2: Hybrid Particle-FDM with No-Flux Boundaries
    # ========================================================================
    print(f"\n" + "="*60)
    print(f"METHOD 2: Hybrid Particle-FDM with No-Flux Boundaries")
    print(f"="*60)
    
    try:
        start_time = time.time()
        
        hjb_solver = FdmHJBSolver(mfg_problem, NiterNewton=15, l2errBoundNewton=1e-6)
        fp_solver = ParticleFPSolver(
            mfg_problem, 
            num_particles=num_particles,
            kde_bandwidth="scott",
            normalize_kde_output=True,
            boundary_conditions=no_flux_bc
        )
        
        hybrid_iterator = FixedPointIterator(
            mfg_problem, hjb_solver=hjb_solver, fp_solver=fp_solver, thetaUM=damping_factor
        )
        
        U_hybrid, M_hybrid, iters_hybrid, _, _ = hybrid_iterator.solve(max_iterations, convergence_tolerance)
        solve_time_hybrid = time.time() - start_time
        
        if U_hybrid is not None and M_hybrid is not None:
            total_mass_hybrid = np.sum(M_hybrid * mfg_problem.Dx, axis=1)
            
            # Check particle boundaries
            particles_trajectory = fp_solver.M_particles_trajectory
            boundary_violations = 0
            if particles_trajectory is not None:
                for t_step in range(particles_trajectory.shape[0]):
                    step_particles = particles_trajectory[t_step, :]
                    violations = np.sum((step_particles < 0.0) | (step_particles > 1.0))
                    boundary_violations += violations
            
            results['Hybrid'] = {
                'mass_initial': total_mass_hybrid[0],
                'mass_final': total_mass_hybrid[-1],
                'mass_change': total_mass_hybrid[-1] - total_mass_hybrid[0],
                'mass_variation': np.max(total_mass_hybrid) - np.min(total_mass_hybrid),
                'iterations': iters_hybrid,
                'time': solve_time_hybrid,
                'total_mass': total_mass_hybrid,
                'boundary_violations': boundary_violations,
                'status': 'SUCCESS'
            }
            print(f"✓ Hybrid: {iters_hybrid} iters, {solve_time_hybrid:.2f}s")
            print(f"  Mass change: {results['Hybrid']['mass_change']:.2e}")
            print(f"  Boundary violations: {boundary_violations}")
        else:
            results['Hybrid'] = {'status': 'FAILED'}
            print(f"❌ Hybrid: Failed to converge")
            
    except Exception as e:
        results['Hybrid'] = {'status': 'ERROR', 'error': str(e)}
        print(f"❌ Hybrid: Error - {e}")
    
    # ========================================================================
    # METHOD 3: Particle-Collocation with No-Flux Boundaries
    # ========================================================================
    print(f"\n" + "="*60)
    print(f"METHOD 3: Particle-Collocation with No-Flux Boundaries")
    print(f"="*60)
    
    try:
        start_time = time.time()
        
        # Create collocation points
        num_collocation_points = 15
        collocation_points = np.linspace(0.0, 1.0, num_collocation_points).reshape(-1, 1)
        
        # Identify boundary points
        boundary_indices = [0, num_collocation_points-1]  # First and last points
        
        collocation_solver = ParticleCollocationSolver(
            problem=mfg_problem,
            collocation_points=collocation_points,
            num_particles=num_particles,
            delta=0.2,
            taylor_order=2,
            kde_bandwidth="scott",
            normalize_kde_output=True,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc
        )
        
        U_collocation, M_collocation, solve_info = collocation_solver.solve(
            Niter=max_iterations, l2errBound=convergence_tolerance, verbose=False
        )
        solve_time_collocation = time.time() - start_time
        
        if U_collocation is not None and M_collocation is not None:
            total_mass_collocation = np.sum(M_collocation * mfg_problem.Dx, axis=1)
            
            # Check particle boundaries
            particles_trajectory = collocation_solver.fp_solver.M_particles_trajectory
            boundary_violations = 0
            if particles_trajectory is not None:
                for t_step in range(particles_trajectory.shape[0]):
                    step_particles = particles_trajectory[t_step, :]
                    violations = np.sum((step_particles < 0.0) | (step_particles > 1.0))
                    boundary_violations += violations
            
            results['Collocation'] = {
                'mass_initial': total_mass_collocation[0],
                'mass_final': total_mass_collocation[-1],
                'mass_change': total_mass_collocation[-1] - total_mass_collocation[0],
                'mass_variation': np.max(total_mass_collocation) - np.min(total_mass_collocation),
                'iterations': solve_info.get('iterations', max_iterations),
                'time': solve_time_collocation,
                'total_mass': total_mass_collocation,
                'boundary_violations': boundary_violations,
                'status': 'SUCCESS'
            }
            print(f"✓ Collocation: {results['Collocation']['iterations']} iters, {solve_time_collocation:.2f}s")
            print(f"  Mass change: {results['Collocation']['mass_change']:.2e}")
            print(f"  Boundary violations: {boundary_violations}")
        else:
            results['Collocation'] = {'status': 'FAILED'}
            print(f"❌ Collocation: Failed to converge")
            
    except Exception as e:
        results['Collocation'] = {'status': 'ERROR', 'error': str(e)}
        print(f"❌ Collocation: Error - {e}")
    
    # ========================================================================
    # COMPARISON SUMMARY
    # ========================================================================
    print(f"\n" + "="*80)
    print(f"                        COMPARISON SUMMARY")
    print(f"="*80)
    
    print(f"{'Method':<15} {'Status':<10} {'Mass Change':<12} {'Mass Var':<10} {'Time':<8} {'Boundary'}")
    print(f"{'-'*15} {'-'*10} {'-'*12} {'-'*10} {'-'*8} {'-'*10}")
    
    for method_name, result in results.items():
        if result['status'] == 'SUCCESS':
            mass_change = f"{result['mass_change']:.2e}"
            mass_var = f"{result['mass_variation']:.2e}"
            time_str = f"{result['time']:.2f}s"
            boundary_str = f"{result.get('boundary_violations', 'N/A')}"
        else:
            mass_change = result['status']
            mass_var = "-"
            time_str = "-"
            boundary_str = "-"
        
        print(f"{method_name:<15} {result['status']:<10} {mass_change:<12} {mass_var:<10} {time_str:<8} {boundary_str}")
    
    # ========================================================================
    # MASS CONSERVATION PLOT
    # ========================================================================
    print(f"\n--- Plotting Mass Conservation Comparison ---")
    
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green']
    method_names = ['FDM', 'Hybrid', 'Collocation']
    
    for i, method in enumerate(method_names):
        if method in results and results[method]['status'] == 'SUCCESS':
            plt.plot(
                mfg_problem.tSpace, 
                results[method]['total_mass'], 
                color=colors[i], 
                linewidth=2, 
                label=f"{method} (Δm={results[method]['mass_change']:.2e})",
                marker='o' if len(mfg_problem.tSpace) < 30 else None,
                markersize=4
            )
    
    plt.xlabel('Time t', fontsize=12)
    plt.ylabel('Total Mass ∫m(t,x)dx', fontsize=12)
    plt.title('Mass Conservation Comparison - No-Flux Boundaries', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Set reasonable y-axis limits
    if any(results[m]['status'] == 'SUCCESS' for m in method_names if m in results):
        all_masses = []
        for method in method_names:
            if method in results and results[method]['status'] == 'SUCCESS':
                all_masses.extend(results[method]['total_mass'])
        
        if all_masses:
            y_min, y_max = min(all_masses), max(all_masses)
            y_margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.01
            plt.ylim([y_min - y_margin, y_max + y_margin])
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n" + "="*80)
    print(f"                    ANALYSIS COMPLETE")
    print(f"="*80)


if __name__ == "__main__":
    run_method_comparison()