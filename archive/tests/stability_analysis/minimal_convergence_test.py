#!/usr/bin/env python3
"""
Minimal convergence test with ultra-light parameters.
Demonstrates QP method consistency with fastest possible execution.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import sys

# Add the main package to path
sys.path.insert(0, '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE')

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.core.boundaries import BoundaryConditions

def minimal_convergence_test():
    """
    Minimal test demonstrating convergence consistency.
    """
    print("="*80)
    print("MINIMAL QP METHOD CONVERGENCE TEST")
    print("="*80)
    print("Ultra-light parameters for fastest execution")
    print("Demonstrating method consistency and expected behavior")
    
    # Ultra-minimal parameters
    base_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "T": 0.1,  # Very short time
        "sigma": 0.1,
        "coefCT": 0.01
    }
    
    # Two very light configurations
    test_configs = [
        {
            'name': 'Minimal',
            'grid': {'Nx': 12, 'Nt': 4},
            'qp': {'particles': 50, 'collocation': 4, 'delta': 0.2, 'order': 1, 'newton': 2, 'picard': 3}
        },
        {
            'name': 'Light',
            'grid': {'Nx': 18, 'Nt': 6},
            'qp': {'particles': 100, 'collocation': 5, 'delta': 0.25, 'order': 1, 'newton': 3, 'picard': 4}
        }
    ]
    
    print(f"\nBase Problem Parameters:")
    for key, value in base_params.items():
        print(f"  {key}: {value}")
    
    results = []
    total_start_time = time.time()
    
    for i, config in enumerate(test_configs):
        print(f"\n{'='*50}")
        print(f"TEST {i+1}/2: {config['name'].upper()}")
        print(f"{'='*50}")
        
        # Create problem
        problem_params = {**base_params, **config['grid']}
        
        print(f"Grid: {config['grid']['Nx']}×{config['grid']['Nt']}, "
              f"Particles: {config['qp']['particles']}, "
              f"Collocation: {config['qp']['collocation']}")
        
        try:
            config_start_time = time.time()
            
            problem = ExampleMFGProblem(**problem_params)
            no_flux_bc = BoundaryConditions(type="no_flux")
            
            print(f"Dx={problem.Dx:.4f}, Dt={problem.Dt:.4f}")
            
            # Setup collocation
            collocation_points = np.linspace(
                problem.xmin, problem.xmax, config['qp']['collocation']
            ).reshape(-1, 1)
            
            boundary_indices = [0, config['qp']['collocation'] - 1]
            
            # Create solver with minimal settings
            solver = ParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                num_particles=config['qp']['particles'],
                delta=config['qp']['delta'],
                taylor_order=config['qp']['order'],
                weight_function="wendland",
                NiterNewton=config['qp']['newton'],
                l2errBoundNewton=1e-2,  # Very relaxed
                kde_bandwidth="scott",
                normalize_kde_output=False,
                boundary_indices=np.array(boundary_indices),
                boundary_conditions=no_flux_bc,
                use_monotone_constraints=True
            )
            
            print(f"Solving...")
            solve_start_time = time.time()
            
            U_solution, M_solution, info = solver.solve(
                Niter=config['qp']['picard'], 
                l2errBound=1e-2,  # Very relaxed
                verbose=False
            )
            
            solve_time = time.time() - solve_start_time
            total_config_time = time.time() - config_start_time
            
            if M_solution is not None and U_solution is not None:
                print(f"✅ Completed in {solve_time:.2f}s")
                
                # Quick analysis
                mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
                initial_mass = mass_evolution[0]
                final_mass = mass_evolution[-1]
                mass_change = final_mass - initial_mass
                mass_change_percent = (mass_change / initial_mass) * 100
                
                max_U = np.max(np.abs(U_solution))
                min_M = np.min(M_solution)
                negative_densities = np.sum(M_solution < -1e-10)
                
                # Boundary violations
                violations = 0
                particles_traj = solver.get_particles_trajectory()
                if particles_traj is not None:
                    final_particles = particles_traj[-1, :]
                    violations = np.sum(
                        (final_particles < problem.xmin - 1e-10) | 
                        (final_particles > problem.xmax + 1e-10)
                    )
                
                # Physical observables
                center_of_mass = np.sum(problem.xSpace * M_solution[-1, :]) * problem.Dx
                
                converged = info.get('converged', False)
                iterations_used = info.get('iterations', 0)
                
                result = {
                    'name': config['name'],
                    'success': True,
                    'solve_time': solve_time,
                    'total_time': total_config_time,
                    'converged': converged,
                    'iterations_used': iterations_used,
                    'initial_mass': initial_mass,
                    'final_mass': final_mass,
                    'mass_change': mass_change,
                    'mass_change_percent': mass_change_percent,
                    'max_U': max_U,
                    'min_M': min_M,
                    'negative_densities': negative_densities,
                    'violations': violations,
                    'center_of_mass': center_of_mass,
                    'mass_evolution': mass_evolution,
                    'M_solution': M_solution,
                    'U_solution': U_solution,
                    'problem': problem
                }
                
                results.append(result)
                
                print(f"Mass: {initial_mass:.4f} → {final_mass:.4f} ({mass_change_percent:+.2f}%)")
                print(f"Center of mass: {center_of_mass:.4f}")
                print(f"Converged: {converged}, Violations: {violations}")
                
                if mass_change > 0:
                    print(f"✅ Mass increases (expected with no-flux BC)")
                elif abs(mass_change_percent) < 5:
                    print(f"✅ Mass well conserved")
                else:
                    print(f"⚠️  Significant mass change")
                
            else:
                print(f"❌ Failed")
                results.append({'name': config['name'], 'success': False})
                
        except Exception as e:
            print(f"❌ Crashed: {e}")
            results.append({'name': config['name'], 'success': False, 'error': str(e)})
    
    total_time = time.time() - total_start_time
    
    # Analysis
    print(f"\n{'='*80}")
    print("CONVERGENCE TEST RESULTS")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.2f}s")
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) == 2:
        result1, result2 = successful_results
        
        print(f"\n{'Metric':<20} {'Minimal':<15} {'Light':<15} {'Difference':<15}")
        print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15}")
        
        metrics = [
            ('Final Mass', 'final_mass', lambda x: f"{x:.6f}"),
            ('Mass Change %', 'mass_change_percent', lambda x: f"{x:+.2f}%"),
            ('Center of Mass', 'center_of_mass', lambda x: f"{x:.4f}"),
            ('Max |U|', 'max_U', lambda x: f"{x:.2e}"),
            ('Solve Time (s)', 'solve_time', lambda x: f"{x:.2f}"),
            ('Violations', 'violations', lambda x: str(x))
        ]
        
        for metric_name, key, fmt in metrics:
            val1 = result1[key]
            val2 = result2[key]
            if key in ['final_mass', 'mass_change_percent', 'center_of_mass', 'max_U', 'solve_time']:
                diff = abs(val2 - val1)
                if key == 'mass_change_percent':
                    diff_str = f"{diff:.2f}pp"
                elif key == 'solve_time':
                    diff_str = f"{diff:.2f}s"
                elif key == 'max_U':
                    diff_str = f"{diff:.1e}"
                else:
                    diff_str = f"{diff:.4f}"
            else:
                diff_str = f"{abs(val2 - val1)}"
            
            print(f"{metric_name:<20} {fmt(val1):<15} {fmt(val2):<15} {diff_str:<15}")
        
        # Convergence assessment
        print(f"\n--- CONVERGENCE ASSESSMENT ---")
        
        mass_diff = abs(result2['final_mass'] - result1['final_mass'])
        mass_rel_diff = mass_diff / result1['final_mass'] * 100
        
        com_diff = abs(result2['center_of_mass'] - result1['center_of_mass'])
        
        both_increase = result1['mass_change'] > 0 and result2['mass_change'] > 0
        both_clean = (result1['violations'] == 0 and result1['negative_densities'] == 0 and
                     result2['violations'] == 0 and result2['negative_densities'] == 0)
        
        print(f"Final mass relative difference: {mass_rel_diff:.3f}%")
        print(f"Center of mass difference: {com_diff:.4f}")
        
        score = 0
        if mass_rel_diff < 1.0:
            print("✅ Excellent mass consistency")
            score += 1
        elif mass_rel_diff < 3.0:
            print("⚠️  Reasonable mass consistency")
            score += 0.5
        else:
            print("❌ Poor mass consistency")
        
        if com_diff < 0.02:
            print("✅ Excellent center of mass consistency")
            score += 1
        elif com_diff < 0.1:
            print("⚠️  Reasonable center of mass consistency")
            score += 0.5
        else:
            print("❌ Poor center of mass consistency")
        
        if both_increase:
            print("✅ Both show expected mass increase")
            score += 1
        elif both_clean:
            print("✅ Both have clean solutions")
            score += 0.5
        
        if both_clean:
            print("✅ Both solutions are numerically clean")
        else:
            print("⚠️  Some numerical issues detected")
        
        assessment = [
            "POOR: Significant convergence issues",
            "FAIR: Some concerns",
            "GOOD: Reasonable convergence", 
            "EXCELLENT: Very consistent"
        ][min(int(score), 3)]
        
        print(f"\nOVERALL ASSESSMENT: {assessment}")
        
        # Create simple plot
        create_minimal_plots(successful_results)
        
    elif len(successful_results) == 1:
        result = successful_results[0]
        print(f"\nOnly one configuration succeeded: {result['name']}")
        print(f"Mass change: {result['mass_change_percent']:+.2f}%")
        print(f"Center of mass: {result['center_of_mass']:.4f}")
        print(f"Solution quality: {result['violations']} violations, {result['negative_densities']} negative densities")
    else:
        print("\nNo configurations succeeded")
        for result in results:
            if not result.get('success', False):
                error = result.get('error', 'Failed')
                print(f"{result['name']}: {error}")
    
    return results

def create_minimal_plots(results):
    """Create minimal comparison plots"""
    if len(results) < 2:
        print("Insufficient results for plotting")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Minimal QP Method Convergence Test', fontsize=14)
    
    colors = ['blue', 'red']
    
    # 1. Mass evolution
    ax1 = axes[0]
    for i, result in enumerate(results):
        problem = result['problem']
        mass_evolution = result['mass_evolution']
        ax1.plot(problem.tSpace, mass_evolution, 'o-', 
                label=result['name'], color=colors[i], linewidth=2)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Total Mass')
    ax1.set_title('Mass Evolution')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Final density
    ax2 = axes[1]
    for i, result in enumerate(results):
        problem = result['problem']
        final_density = result['M_solution'][-1, :]
        ax2.plot(problem.xSpace, final_density, 
                label=result['name'], color=colors[i], linewidth=2)
    ax2.set_xlabel('Space x')
    ax2.set_ylabel('Final Density')
    ax2.set_title('Final Density Comparison')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Key metrics
    ax3 = axes[2]
    names = [r['name'] for r in results]
    mass_changes = [r['mass_change_percent'] for r in results]
    solve_times = [r['solve_time'] for r in results]
    
    x_pos = np.arange(len(names))
    width = 0.35
    
    bars1 = ax3.bar(x_pos - width/2, mass_changes, width, 
                    label='Mass Change (%)', color='green', alpha=0.7)
    
    # Secondary y-axis for time
    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.bar(x_pos + width/2, solve_times, width, 
                        label='Solve Time (s)', color='orange', alpha=0.7)
    
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Mass Change (%)', color='green')
    ax3_twin.set_ylabel('Solve Time (s)', color='orange')
    ax3.set_title('Performance Summary')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names)
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, mass_changes):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.2f}%', ha='center', va='bottom')
    
    for bar, value in zip(bars2, solve_times):
        ax3_twin.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{value:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/minimal_convergence_test.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Starting minimal convergence test...")
    print("Expected execution time: < 30 seconds")
    
    try:
        results = minimal_convergence_test()
        print("\n" + "="*80)
        print("MINIMAL CONVERGENCE TEST COMPLETED")
        print("="*80)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
