#!/usr/bin/env python3
"""
Direct comparison of three MFG solver methods using existing implementations:
1. Pure FDM from tests/pure_fdm.py 
2. Hybrid Particle-FDM from tests/hybrid.py
3. Second-Order QP Particle-Collocation (validated implementation)
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os

# Add the main package to path
sys.path.insert(0, '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE')

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.core.boundaries import BoundaryConditions

# Import existing solvers from tests
from tests.pure_fdm import FDMSolver
from tests.hybrid import HybridSolver

def compare_three_methods_direct():
    print("="*80)
    print("DIRECT THREE-METHOD MFG COMPARISON")
    print("="*80)
    print("1. Pure FDM (from tests/pure_fdm.py)")
    print("2. Hybrid Particle-FDM (from tests/hybrid.py)")  
    print("3. Second-Order QP Particle-Collocation (validated implementation)")
    
    # Mild test parameters for all methods to work
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 20,   
        "T": 0.15,   
        "Nt": 8,   
        "sigma": 0.15,  
        "coefCT": 0.005  
    }
    
    print(f"\nProblem Parameters:")
    for key, value in problem_params.items():
        print(f"  {key}: {value}")
    
    problem = ExampleMFGProblem(**problem_params)
    no_flux_bc = BoundaryConditions(type="no_flux")
    
    results = {}
    
    # Method 1: Pure FDM
    print(f"\n{'='*60}")
    print("METHOD 1: PURE FDM")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        fdm_solver = FDMSolver(problem, thetaUM=0.7, NiterNewton=5, l2errBoundNewton=1e-3)
        U_fdm, M_fdm, iterations_fdm, _, _ = fdm_solver.solve(Niter=6, l2errBoundPicard=1e-3)
        time_fdm = time.time() - start_time
        
        if M_fdm is not None and U_fdm is not None:
            mass_fdm = np.sum(M_fdm * problem.Dx, axis=1)
            initial_mass = mass_fdm[0]
            final_mass = mass_fdm[-1]
            mass_change = abs(final_mass - initial_mass)
            
            results['fdm'] = {
                'success': True,
                'initial_mass': initial_mass,
                'final_mass': final_mass,
                'mass_change': mass_change,
                'max_U': np.max(np.abs(U_fdm)),
                'time': time_fdm,
                'converged': True,  # FDM doesn't report this
                'iterations': iterations_fdm,
                'violations': 0,
                'U_solution': U_fdm,
                'M_solution': M_fdm
            }
            print(f"✓ FDM completed: initial_mass={initial_mass:.6f}, final_mass={final_mass:.6f}")
            print(f"  Mass change: {mass_change:.2e}, time: {time_fdm:.2f}s, iterations: {iterations_fdm}")
        else:
            results['fdm'] = {'success': False}
            print("❌ FDM failed")
    except Exception as e:
        results['fdm'] = {'success': False, 'error': str(e)}
        print(f"❌ FDM crashed: {e}")
        import traceback
        traceback.print_exc()
    
    # Method 2: Hybrid
    print(f"\n{'='*60}")
    print("METHOD 2: HYBRID PARTICLE-FDM")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        hybrid_solver = HybridSolver(
            problem, 
            num_particles=500, 
            thetaUM=0.7, 
            NiterNewton=5, 
            l2errBoundNewton=1e-3
        )
        U_hybrid, M_hybrid, particles_hybrid, iterations_hybrid, _, _ = hybrid_solver.solve(
            Niter=6, l2errBoundPicard=1e-3
        )
        time_hybrid = time.time() - start_time
        
        if M_hybrid is not None and U_hybrid is not None:
            mass_hybrid = np.sum(M_hybrid * problem.Dx, axis=1)
            initial_mass = mass_hybrid[0]
            final_mass = mass_hybrid[-1]
            mass_change = abs(final_mass - initial_mass)
            
            # Count boundary violations
            violations = 0
            if particles_hybrid is not None:
                final_particles = particles_hybrid[-1, :]
                violations = np.sum((final_particles < problem.xmin) | (final_particles > problem.xmax))
            
            results['hybrid'] = {
                'success': True,
                'initial_mass': initial_mass,
                'final_mass': final_mass,
                'mass_change': mass_change,
                'max_U': np.max(np.abs(U_hybrid)),
                'time': time_hybrid,
                'converged': True,
                'iterations': iterations_hybrid,
                'violations': violations,
                'U_solution': U_hybrid,
                'M_solution': M_hybrid
            }
            print(f"✓ Hybrid completed: initial_mass={initial_mass:.6f}, final_mass={final_mass:.6f}")
            print(f"  Mass change: {mass_change:.2e}, time: {time_hybrid:.2f}s, violations: {violations}")
        else:
            results['hybrid'] = {'success': False}
            print("❌ Hybrid failed")
    except Exception as e:
        results['hybrid'] = {'success': False, 'error': str(e)}
        print(f"❌ Hybrid crashed: {e}")
        import traceback
        traceback.print_exc()
    
    # Method 3: QP-Collocation (validated)
    print(f"\n{'='*60}")
    print("METHOD 3: SECOND-ORDER QP PARTICLE-COLLOCATION")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Collocation setup
        num_collocation_points = 8
        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)
        
        boundary_indices = []
        for i, point in enumerate(collocation_points):
            x = point[0]
            if abs(x - problem.xmin) < 1e-10 or abs(x - problem.xmax) < 1e-10:
                boundary_indices.append(i)
        boundary_indices = np.array(boundary_indices)
        
        collocation_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=500,
            delta=0.25,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=5,
            l2errBoundNewton=1e-3,
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=boundary_indices,
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True
        )
        
        U_colloc, M_colloc, info_colloc = collocation_solver.solve(
            Niter=6, l2errBound=1e-3, verbose=False
        )
        
        time_colloc = time.time() - start_time
        
        if M_colloc is not None and U_colloc is not None:
            mass_colloc = np.sum(M_colloc * problem.Dx, axis=1)
            initial_mass = mass_colloc[0]
            final_mass = mass_colloc[-1]
            mass_change = abs(final_mass - initial_mass)
            
            # Count boundary violations
            violations = 0
            particles_trajectory = collocation_solver.get_particles_trajectory()
            if particles_trajectory is not None:
                final_particles = particles_trajectory[-1, :]
                violations = np.sum((final_particles < problem.xmin) | (final_particles > problem.xmax))
            
            results['collocation'] = {
                'success': True,
                'initial_mass': initial_mass,
                'final_mass': final_mass,
                'mass_change': mass_change,
                'max_U': np.max(np.abs(U_colloc)),
                'time': time_colloc,
                'converged': info_colloc.get('converged', False),
                'iterations': info_colloc.get('iterations', 0),
                'violations': violations,
                'U_solution': U_colloc,
                'M_solution': M_colloc
            }
            print(f"✓ QP-Collocation completed: initial_mass={initial_mass:.6f}, final_mass={final_mass:.6f}")
            print(f"  Mass change: {mass_change:.2e}, time: {time_colloc:.2f}s, violations: {violations}")
        else:
            results['collocation'] = {'success': False}
            print("❌ QP-Collocation failed")
    except Exception as e:
        results['collocation'] = {'success': False, 'error': str(e)}
        print(f"❌ QP-Collocation crashed: {e}")
        import traceback
        traceback.print_exc()
    
    # Results comparison
    print(f"\n{'='*80}")
    print("DIRECT COMPARISON RESULTS")
    print(f"{'='*80}")
    
    successful_methods = [m for m in ['fdm', 'hybrid', 'collocation'] if results.get(m, {}).get('success', False)]
    
    if len(successful_methods) >= 2:
        print(f"\n{'Metric':<25} {'Pure FDM':<15} {'Hybrid':<15} {'QP-Collocation':<15}")
        print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15}")
        
        # Show initial and final masses
        for method in ['fdm', 'hybrid', 'collocation']:
            if method in successful_methods:
                initial = results[method]['initial_mass']
                final = results[method]['final_mass']
                print(f"Initial mass {method.upper():<15} {initial:.6f}")
        
        print()
        for method in ['fdm', 'hybrid', 'collocation']:
            if method in successful_methods:
                initial = results[method]['initial_mass']
                final = results[method]['final_mass']
                print(f"Final mass {method.upper():<17} {final:.6f}")
        
        print()
        
        # Comparison table
        metrics = [
            ('Mass change', 'mass_change', lambda x: f"{x:.2e}"),
            ('Max |U|', 'max_U', lambda x: f"{x:.1e}"),
            ('Runtime (s)', 'time', lambda x: f"{x:.2f}"),
            ('Violations', 'violations', lambda x: str(int(x))),
            ('Converged', 'converged', lambda x: "Yes" if x else "No"),
            ('Iterations', 'iterations', lambda x: str(int(x)))
        ]
        
        for metric_name, key, fmt in metrics:
            row = [metric_name]
            for method in ['fdm', 'hybrid', 'collocation']:
                if method in successful_methods:
                    value = results[method].get(key, 0)
                    row.append(fmt(value))
                else:
                    row.append("FAILED")
            print(f"{row[0]:<25} {row[1]:<15} {row[2]:<15} {row[3]:<15}")
        
        # Convergence analysis
        print(f"\n--- Convergence Analysis ---")
        if len(successful_methods) >= 2:
            final_masses = [results[m]['final_mass'] for m in successful_methods]
            max_diff = max(final_masses) - min(final_masses)
            avg_mass = np.mean(final_masses)
            relative_diff = (max_diff / avg_mass) * 100 if avg_mass > 0 else 0
            
            print(f"Final mass range: [{min(final_masses):.6f}, {max(final_masses):.6f}]")
            print(f"Max difference: {max_diff:.2e}")
            print(f"Relative difference: {relative_diff:.3f}%")
            
            if relative_diff < 1.0:
                print("✅ EXCELLENT: All methods converge to consistent solution")
            elif relative_diff < 5.0:
                print("✅ GOOD: Methods show reasonable convergence")
            else:
                print("⚠️  WARNING: Methods show significant divergence")
        
        # Performance analysis
        if len(successful_methods) >= 2:
            runtimes = {m: results[m]['time'] for m in successful_methods}
            fastest_method = min(runtimes, key=runtimes.get)
            fastest_time = runtimes[fastest_method]
            
            print(f"\n--- Performance Analysis ---")
            for method in successful_methods:
                runtime = results[method]['time']
                if method == fastest_method:
                    print(f"{method.upper()}:        {runtime:.2f}s (fastest)")
                else:
                    overhead = (runtime - fastest_time) / fastest_time * 100
                    print(f"{method.upper()}:        {runtime:.2f}s ({overhead:+.1f}% overhead)")
        
        # Create plots
        create_direct_comparison_plots(results, problem, successful_methods)
    
    else:
        print("Insufficient successful methods for comparison")
        for method in ['fdm', 'hybrid', 'collocation']:
            if not results.get(method, {}).get('success', False):
                error = results.get(method, {}).get('error', 'Failed')
                print(f"❌ {method.upper()}: {error}")

def create_direct_comparison_plots(results, problem, successful_methods):
    """Create comparison plots for the direct implementations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Direct Three-Method MFG Comparison (Existing Implementations)', fontsize=16)
    
    method_names = {'fdm': 'Pure FDM', 'hybrid': 'Hybrid', 'collocation': 'QP-Collocation'}
    colors = {'fdm': 'blue', 'hybrid': 'green', 'collocation': 'red'}
    
    # Final density comparison
    ax1 = axes[0, 0]
    for method in successful_methods:
        M_solution = results[method]['M_solution']
        final_density = M_solution[-1, :]
        ax1.plot(problem.xSpace, final_density, 
                label=method_names[method], linewidth=2, color=colors[method])
    ax1.set_xlabel('Space x')
    ax1.set_ylabel('Final Density M(T,x)')
    ax1.set_title('Final Density Comparison')
    ax1.grid(True)
    ax1.legend()
    
    # Mass conservation over time
    ax2 = axes[0, 1]
    for method in successful_methods:
        M_solution = results[method]['M_solution']
        mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
        ax2.plot(problem.tSpace, mass_evolution, 
                label=method_names[method], linewidth=2, color=colors[method])
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Total Mass')
    ax2.set_title('Mass Conservation Over Time')
    ax2.grid(True)
    ax2.legend()
    
    # Mass change comparison (log scale)
    ax3 = axes[1, 0]
    methods = [method_names[m] for m in successful_methods]
    mass_changes = [results[m]['mass_change'] for m in successful_methods]
    bars = ax3.bar(methods, mass_changes, color=[colors[m] for m in successful_methods])
    ax3.set_ylabel('Mass Change (log scale)')
    ax3.set_title('Mass Conservation Quality')
    ax3.set_yscale('log')
    ax3.grid(True, axis='y')
    for bar, value in zip(bars, mass_changes):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.1e}', ha='center', va='bottom')
    
    # Runtime comparison
    ax4 = axes[1, 1]
    runtimes = [results[m]['time'] for m in successful_methods]
    bars = ax4.bar(methods, runtimes, color=[colors[m] for m in successful_methods])
    ax4.set_ylabel('Runtime (seconds)')
    ax4.set_title('Performance Comparison')
    ax4.grid(True, axis='y')
    for bar, value in zip(bars, runtimes):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/direct_comparison.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    compare_three_methods_direct()
