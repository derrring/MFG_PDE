#!/usr/bin/env python3
"""
Compare three core MFG solver methods using the same equation and no-flux BC:
1. Pure FDM (Finite Difference Method)
2. Hybrid Particle-FDM (Particle-FP + FDM-HJB)
3. Second-order Taylor + QP Particle-Collocation (our best method)
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os

# Add paths for the solvers
sys.path.append(os.path.join(os.path.dirname(__file__), 'mfg_pde', 'alg'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tests'))

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.core.boundaries import BoundaryConditions

def compare_three_core_methods():
    print("="*80)
    print("COMPREHENSIVE MFG SOLVER COMPARISON")
    print("="*80)
    print("Methods: Pure FDM vs Hybrid Particle-FDM vs Second-Order QP Particle-Collocation")
    print("Same MFG equation, same no-flux boundary conditions, same parameters")
    
    # Use challenging but manageable parameters
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 60,   # High resolution
        "T": 1.0,   # Full time horizon  
        "Nt": 50,   # Adequate time steps
        "sigma": 0.2,   # Realistic diffusion
        "coefCT": 0.05  # Realistic coupling
    }
    
    print(f"\nProblem Parameters:")
    for key, value in problem_params.items():
        print(f"  {key}: {value}")
    
    # Create common MFG problem
    problem = ExampleMFGProblem(**problem_params)
    
    # No-flux boundary conditions for all methods
    no_flux_bc = BoundaryConditions(type="no_flux")
    
    # Storage for results
    results = {}
    
    print(f"\n{'='*80}")
    print("METHOD 1: PURE FDM (Finite Difference Method)")
    print(f"{'='*80}")
    print("Classical finite difference for both HJB and Fokker-Planck equations")
    
    try:
        # Import and use pure FDM solver
        from pure_fdm import FDMSolver
        
        start_time = time.time()
        
        fdm_solver = FDMSolver(
            problem=problem,
            thetaUM=0.5,  # Damping parameter
            NiterNewton=30,
            l2errBoundNewton=1e-6
        )
        
        U_fdm, M_fdm = fdm_solver.solve(
            Niter=12,  # Conservative iteration count
            l2errBoundPicard=2e-3
        )
        
        time_fdm = time.time() - start_time
        
        if M_fdm is not None and U_fdm is not None:
            # Mass conservation analysis
            mass_fdm = np.sum(M_fdm * problem.Dx, axis=1)
            mass_change_fdm = abs(mass_fdm[-1] - mass_fdm[0])
            mass_variation_fdm = np.max(mass_fdm) - np.min(mass_fdm)
            
            # Solution metrics
            max_U_fdm = np.max(np.abs(U_fdm))
            max_M_fdm = np.max(M_fdm)
            
            # Check convergence
            converged_fdm = (fdm_solver.l2distu is not None and 
                           len(fdm_solver.l2distu) > 0 and 
                           fdm_solver.l2distu[-1] < 2e-3)
            
            results['fdm'] = {
                'success': True,
                'mass_change': mass_change_fdm,
                'mass_variation': mass_variation_fdm,
                'max_U': max_U_fdm,
                'max_M': max_M_fdm,
                'time': time_fdm,
                'converged': converged_fdm,
                'iterations': fdm_solver.iterations_run,
                'U_solution': U_fdm,
                'M_solution': M_fdm,
                'violations': 0  # FDM doesn't have particle violations
            }
            
            print(f"\nPure FDM Results:")
            print(f"  Mass change: {mass_change_fdm:.3e}")
            print(f"  Mass variation: {mass_variation_fdm:.3e}")
            print(f"  Max |U|: {max_U_fdm:.2e}")
            print(f"  Max M: {max_M_fdm:.3f}")
            print(f"  Runtime: {time_fdm:.2f}s")
            print(f"  Converged: {converged_fdm}")
            print(f"  Iterations: {fdm_solver.iterations_run}")
            
        else:
            print("❌ Pure FDM failed to produce valid solutions")
            results['fdm'] = {'success': False}
            
    except Exception as e:
        print(f"❌ Pure FDM crashed: {e}")
        results['fdm'] = {'success': False, 'error': str(e)}
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("METHOD 2: HYBRID PARTICLE-FDM")
    print(f"{'='*80}")
    print("Particle method for Fokker-Planck + FDM for HJB")
    
    try:
        # Import and use hybrid solver
        from hybrid_fdm import ParticleSolver as HybridSolver
        
        start_time = time.time()
        
        hybrid_solver = HybridSolver(
            problem=problem,
            num_particles=400,  # Same as particle-collocation
            particle_thetaUM=0.5,
            kde_bandwidth="scott",
            NiterNewton=30,
            l2errBoundNewton=1e-6
        )
        
        U_hybrid, M_hybrid = hybrid_solver.solve(
            Niter=12,
            l2errBoundPicard=2e-3
        )
        
        time_hybrid = time.time() - start_time
        
        if M_hybrid is not None and U_hybrid is not None:
            # Mass conservation analysis
            mass_hybrid = np.sum(M_hybrid * problem.Dx, axis=1)
            mass_change_hybrid = abs(mass_hybrid[-1] - mass_hybrid[0])
            mass_variation_hybrid = np.max(mass_hybrid) - np.min(mass_hybrid)
            
            # Solution metrics
            max_U_hybrid = np.max(np.abs(U_hybrid))
            max_M_hybrid = np.max(M_hybrid)
            
            # Particle boundary violations
            violations_hybrid = 0
            if hasattr(hybrid_solver, 'M_particles') and hybrid_solver.M_particles is not None:
                final_particles = hybrid_solver.M_particles[-1, :]
                violations_hybrid = np.sum((final_particles < problem.xmin) | (final_particles > problem.xmax))
            
            # Check convergence
            converged_hybrid = (hybrid_solver.l2distu is not None and 
                              len(hybrid_solver.l2distu) > 0 and 
                              hybrid_solver.l2distu[-1] < 2e-3)
            
            results['hybrid'] = {
                'success': True,
                'mass_change': mass_change_hybrid,
                'mass_variation': mass_variation_hybrid,
                'max_U': max_U_hybrid,
                'max_M': max_M_hybrid,
                'violations': violations_hybrid,
                'time': time_hybrid,
                'converged': converged_hybrid,
                'iterations': hybrid_solver.iterations_run,
                'U_solution': U_hybrid,
                'M_solution': M_hybrid
            }
            
            print(f"\nHybrid Particle-FDM Results:")
            print(f"  Mass change: {mass_change_hybrid:.3e}")
            print(f"  Mass variation: {mass_variation_hybrid:.3e}")
            print(f"  Max |U|: {max_U_hybrid:.2e}")
            print(f"  Max M: {max_M_hybrid:.3f}")
            print(f"  Boundary violations: {violations_hybrid}")
            print(f"  Runtime: {time_hybrid:.2f}s")
            print(f"  Converged: {converged_hybrid}")
            print(f"  Iterations: {hybrid_solver.iterations_run}")
            
        else:
            print("❌ Hybrid method failed to produce valid solutions")
            results['hybrid'] = {'success': False}
            
    except Exception as e:
        print(f"❌ Hybrid method crashed: {e}")
        results['hybrid'] = {'success': False, 'error': str(e)}
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("METHOD 3: SECOND-ORDER QP PARTICLE-COLLOCATION")
    print(f"{'='*80}")
    print("Our best method: Second-order Taylor + QP constraints + Particle-FP")
    
    try:
        start_time = time.time()
        
        # Collocation setup
        num_collocation_points = 15
        collocation_points = np.linspace(
            problem.xmin, problem.xmax, num_collocation_points
        ).reshape(-1, 1)
        
        # Boundary indices
        boundary_tolerance = 1e-10
        boundary_indices = []
        for i, point in enumerate(collocation_points):
            x = point[0]
            if (abs(x - problem.xmin) < boundary_tolerance or 
                abs(x - problem.xmax) < boundary_tolerance):
                boundary_indices.append(i)
        boundary_indices = np.array(boundary_indices)
        
        collocation_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=400,  # Same as hybrid
            delta=0.4,
            taylor_order=2,  # SECOND-ORDER Taylor expansion
            weight_function="wendland",
            NiterNewton=12,
            l2errBoundNewton=1e-4,
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=boundary_indices,
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True  # QP constraints enabled
        )
        
        U_colloc, M_colloc, info_colloc = collocation_solver.solve(
            Niter=12,
            l2errBound=2e-3,
            verbose=True
        )
        
        time_colloc = time.time() - start_time
        
        if M_colloc is not None and U_colloc is not None:
            # Mass conservation analysis
            mass_colloc = np.sum(M_colloc * problem.Dx, axis=1)
            mass_change_colloc = abs(mass_colloc[-1] - mass_colloc[0])
            mass_variation_colloc = np.max(mass_colloc) - np.min(mass_colloc)
            
            # Solution metrics
            max_U_colloc = np.max(np.abs(U_colloc))
            max_M_colloc = np.max(M_colloc)
            
            # Particle boundary violations
            violations_colloc = 0
            particles_trajectory = collocation_solver.get_particles_trajectory()
            if particles_trajectory is not None:
                final_particles = particles_trajectory[-1, :]
                violations_colloc = np.sum((final_particles < problem.xmin) | (final_particles > problem.xmax))
            
            results['collocation'] = {
                'success': True,
                'mass_change': mass_change_colloc,
                'mass_variation': mass_variation_colloc,
                'max_U': max_U_colloc,
                'max_M': max_M_colloc,
                'violations': violations_colloc,
                'time': time_colloc,
                'converged': info_colloc.get('converged', False),
                'iterations': info_colloc.get('iterations', 0),
                'U_solution': U_colloc,
                'M_solution': M_colloc
            }
            
            print(f"\nSecond-Order QP Particle-Collocation Results:")
            print(f"  Mass change: {mass_change_colloc:.3e}")
            print(f"  Mass variation: {mass_variation_colloc:.3e}")
            print(f"  Max |U|: {max_U_colloc:.2e}")
            print(f"  Max M: {max_M_colloc:.3f}")
            print(f"  Boundary violations: {violations_colloc}")
            print(f"  Runtime: {time_colloc:.2f}s")
            print(f"  Converged: {info_colloc.get('converged', False)}")
            print(f"  Iterations: {info_colloc.get('iterations', 0)}")
            
        else:
            print("❌ Second-Order QP Particle-Collocation failed")
            results['collocation'] = {'success': False}
            
    except Exception as e:
        print(f"❌ Second-Order QP Particle-Collocation crashed: {e}")
        results['collocation'] = {'success': False, 'error': str(e)}
        import traceback
        traceback.print_exc()
    
    # Comprehensive comparison
    print(f"\n{'='*90}")
    print("COMPREHENSIVE COMPARISON ANALYSIS")
    print(f"{'='*90}")
    
    successful_methods = [method for method in ['fdm', 'hybrid', 'collocation'] 
                         if results.get(method, {}).get('success', False)]
    
    if len(successful_methods) >= 2:
        print(f"\nSuccessful methods: {', '.join([m.upper() for m in successful_methods])}")
        
        # Create comprehensive comparison table
        print(f"\n{'Metric':<25} {'Pure FDM':<15} {'Hybrid':<15} {'QP-Collocation':<15}")
        print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15}")
        
        # Mass conservation metrics
        metrics = [
            ('Mass change', 'mass_change', ':.2e'),
            ('Mass variation', 'mass_variation', ':.2e'),
            ('Max |U|', 'max_U', ':.1e'),
            ('Max M', 'max_M', ':.3f'),
            ('Runtime (s)', 'time', ':.2f'),
            ('Boundary violations', 'violations', ':d'),
            ('Iterations', 'iterations', ':d')
        ]
        
        for metric_name, metric_key, fmt in metrics:
            row = [metric_name]
            for method in ['fdm', 'hybrid', 'collocation']:
                if method in successful_methods:
                    value = results[method].get(metric_key, 0)
                    if fmt == ':d':
                        row.append(str(int(value)))
                    elif fmt == ':.2e':
                        row.append(f"{value:.2e}")
                    elif fmt == ':.1e':
                        row.append(f"{value:.1e}")
                    elif fmt == ':.3f':
                        row.append(f"{value:.3f}")
                    elif fmt == ':.2f':
                        row.append(f"{value:.2f}")
                    else:
                        row.append(str(value))
                else:
                    row.append("FAILED")
            
            print(f"{row[0]:<25} {row[1]:<15} {row[2]:<15} {row[3]:<15}")
        
        # Convergence status
        conv_row = ['Converged']
        for method in ['fdm', 'hybrid', 'collocation']:
            if method in successful_methods:
                converged = results[method]['converged']
                conv_row.append("Yes" if converged else "No")
            else:
                conv_row.append("FAILED")
        
        print(f"{conv_row[0]:<25} {conv_row[1]:<15} {conv_row[2]:<15} {conv_row[3]:<15}")
        
        print(f"\n{'='*90}")
        print("METHOD ASSESSMENT")
        print(f"{'='*90}")
        
        # Find best performer in each category
        best_mass = min(successful_methods, key=lambda m: results[m]['mass_change'])
        best_stability = min(successful_methods, key=lambda m: results[m]['max_U'])
        fastest = min(successful_methods, key=lambda m: results[m]['time'])
        best_boundary = min(successful_methods, key=lambda m: results[m]['violations'])
        
        print(f"\n🏆 Best mass conservation: {best_mass.upper()}")
        print(f"🏆 Best solution stability: {best_stability.upper()}")
        print(f"🏆 Fastest runtime: {fastest.upper()}")
        print(f"🏆 Best boundary compliance: {best_boundary.upper()}")
        
        print(f"\n--- Method Characteristics ---")
        if 'fdm' in successful_methods:
            print(f"✓ Pure FDM: Classical approach, grid-based, proven stability")
        if 'hybrid' in successful_methods:
            print(f"✓ Hybrid: Combines FDM reliability with particle flexibility")
        if 'collocation' in successful_methods:
            print(f"✓ QP-Collocation: Meshfree, second-order accuracy, monotonicity preservation")
        
        # Performance comparison
        if len(successful_methods) == 3:
            fdm_time = results['fdm']['time']
            hybrid_overhead = (results['hybrid']['time'] - fdm_time) / fdm_time * 100
            colloc_overhead = (results['collocation']['time'] - fdm_time) / fdm_time * 100
            
            print(f"\n--- Performance Overhead (vs Pure FDM) ---")
            print(f"Hybrid: {hybrid_overhead:+.1f}%")
            print(f"QP-Collocation: {colloc_overhead:+.1f}%")
        
        # Overall recommendation
        print(f"\n--- Overall Assessment ---")
        
        # Count wins for each method
        wins = {'fdm': 0, 'hybrid': 0, 'collocation': 0}
        categories = [best_mass, best_stability, fastest, best_boundary]
        
        for winner in categories:
            if winner in wins:
                wins[winner] += 1
        
        winner = max(wins.keys(), key=lambda k: wins[k])
        
        print(f"🎉 OVERALL WINNER: {winner.upper()}")
        print(f"   Wins: {wins[winner]}/4 categories")
        
        if winner == 'fdm':
            print("   → Recommended for: Classical applications, proven stability requirements")
        elif winner == 'hybrid':
            print("   → Recommended for: Balance of flexibility and reliability")
        elif winner == 'collocation':
            print("   → Recommended for: High-accuracy applications, complex geometries")
        
        # Create plots if solutions available
        if len(successful_methods) >= 2:
            print(f"\n--- Creating Comparison Plots ---")
            create_comparison_plots(results, problem, successful_methods)
        
    else:
        print(f"\nInsufficient successful methods for comparison.")
        for method in ['fdm', 'hybrid', 'collocation']:
            if not results.get(method, {}).get('success', False):
                error = results.get(method, {}).get('error', 'Unknown error')
                print(f"❌ {method.upper()}: {error}")

def create_comparison_plots(results, problem, successful_methods):
    """Create comprehensive comparison plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Three-Method MFG Solver Comparison', fontsize=16)
    
    method_names = {'fdm': 'Pure FDM', 'hybrid': 'Hybrid', 'collocation': 'QP-Collocation'}
    colors = {'fdm': 'blue', 'hybrid': 'green', 'collocation': 'red'}
    
    # Plot 1: Final density profiles
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
    
    # Plot 2: Final value function profiles  
    ax2 = axes[0, 1]
    for method in successful_methods:
        U_solution = results[method]['U_solution']
        final_value = U_solution[-1, :]
        ax2.plot(problem.xSpace, final_value, 
                label=method_names[method], linewidth=2, color=colors[method])
    ax2.set_xlabel('Space x')
    ax2.set_ylabel('Final Value U(T,x)')
    ax2.set_title('Final Value Function Comparison')
    ax2.grid(True)
    ax2.legend()
    
    # Plot 3: Mass conservation over time
    ax3 = axes[0, 2]
    for method in successful_methods:
        M_solution = results[method]['M_solution']
        mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
        ax3.plot(problem.tSpace, mass_evolution, 
                label=method_names[method], linewidth=2, color=colors[method])
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Total Mass ∫M(t,x)dx')
    ax3.set_title('Mass Conservation')
    ax3.grid(True)
    ax3.legend()
    
    # Plot 4: Performance metrics bar chart
    ax4 = axes[1, 0]
    methods = [method_names[m] for m in successful_methods]
    mass_changes = [results[m]['mass_change'] for m in successful_methods]
    
    bars = ax4.bar(methods, mass_changes, color=[colors[m] for m in successful_methods])
    ax4.set_ylabel('Mass Change')
    ax4.set_title('Mass Conservation Comparison')
    ax4.set_yscale('log')
    ax4.grid(True, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, mass_changes):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1e}', ha='center', va='bottom')
    
    # Plot 5: Runtime comparison
    ax5 = axes[1, 1]
    runtimes = [results[m]['time'] for m in successful_methods]
    
    bars = ax5.bar(methods, runtimes, color=[colors[m] for m in successful_methods])
    ax5.set_ylabel('Runtime (seconds)')
    ax5.set_title('Performance Comparison')
    ax5.grid(True, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, runtimes):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}s', ha='center', va='bottom')
    
    # Plot 6: Boundary violations (for particle methods)
    ax6 = axes[1, 2]
    particle_methods = [m for m in successful_methods if m in ['hybrid', 'collocation']]
    if particle_methods:
        methods_6 = [method_names[m] for m in particle_methods]
        violations = [results[m]['violations'] for m in particle_methods]
        
        bars = ax6.bar(methods_6, violations, color=[colors[m] for m in particle_methods])
        ax6.set_ylabel('Boundary Violations')
        ax6.set_title('Boundary Compliance')
        ax6.grid(True, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, violations):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value}', ha='center', va='bottom')
    else:
        ax6.text(0.5, 0.5, 'No particle methods\navailable', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Boundary Compliance (N/A)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_three_core_methods()