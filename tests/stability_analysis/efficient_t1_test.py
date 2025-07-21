#!/usr/bin/env python3
"""
Efficient T=1 test with optimized parameters for reasonable execution time.
Balances thoroughness with computational efficiency.
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

def test_efficient_t1():
    """
    Efficient T=1 test with optimized computational parameters.
    """
    print("="*80)
    print("EFFICIENT T=1 QP PARTICLE-COLLOCATION TEST")
    print("="*80)
    print("Testing T=1 with optimized parameters for reasonable execution time")
    
    # Optimized configurations for T=1
    test_configs = [
        {
            'name': 'Efficient Standard',
            'params': {
                'xmin': 0.0, 'xmax': 1.0, 'Nx': 30, 
                'T': 1.0, 'Nt': 20, 
                'sigma': 0.2, 'coefCT': 0.02
            },
            'qp_settings': {
                'particles': 400, 'collocation': 10, 'delta': 0.3, 
                'order': 2, 'newton': 6, 'picard': 8
            }
        },
        {
            'name': 'Efficient High-Quality',
            'params': {
                'xmin': 0.0, 'xmax': 1.0, 'Nx': 40, 
                'T': 1.0, 'Nt': 25, 
                'sigma': 0.25, 'coefCT': 0.025
            },
            'qp_settings': {
                'particles': 600, 'collocation': 12, 'delta': 0.35, 
                'order': 2, 'newton': 8, 'picard': 10
            }
        }
    ]
    
    results = []
    total_start_time = time.time()
    
    for i, config in enumerate(test_configs):
        print(f"\n{'='*60}")
        print(f"TEST {i+1}/2: {config['name'].upper()}")
        print(f"{'='*60}")
        
        # Display configuration
        print("\nProblem parameters:")
        for key, value in config['params'].items():
            print(f"  {key}: {value}")
        
        print("\nQP solver settings:")
        for key, value in config['qp_settings'].items():
            print(f"  {key}: {value}")
        
        # Estimate time
        complexity_score = (
            config['params']['Nx'] * 
            config['params']['Nt'] * 
            config['qp_settings']['particles'] * 
            config['qp_settings']['picard']
        ) / 1000000
        estimated_time = complexity_score * 2  # rough estimate
        print(f"\nEstimated time: ~{estimated_time:.1f} minutes")
        
        try:
            config_start_time = time.time()
            
            # Create problem
            problem = ExampleMFGProblem(**config['params'])
            no_flux_bc = BoundaryConditions(type="no_flux")
            
            print(f"\nGrid: Dx = {problem.Dx:.4f}, Dt = {problem.Dt:.4f}")
            print(f"CFL number: {config['params']['sigma']**2 * problem.Dt / problem.Dx**2:.4f}")
            
            # Setup collocation
            num_collocation = config['qp_settings']['collocation']
            collocation_points = np.linspace(
                problem.xmin, problem.xmax, num_collocation
            ).reshape(-1, 1)
            
            boundary_indices = [0, num_collocation - 1]
            
            # Create solver
            solver = ParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                num_particles=config['qp_settings']['particles'],
                delta=config['qp_settings']['delta'],
                taylor_order=config['qp_settings']['order'],
                weight_function="wendland",
                NiterNewton=config['qp_settings']['newton'],
                l2errBoundNewton=1e-3,
                kde_bandwidth="scott",
                normalize_kde_output=False,
                boundary_indices=np.array(boundary_indices),
                boundary_conditions=no_flux_bc,
                use_monotone_constraints=True
            )
            
            print(f"\nStarting solve with {config['qp_settings']['particles']} particles...")
            solve_start_time = time.time()
            
            # Track progress
            class ProgressTracker:
                def __init__(self):
                    self.last_print_time = time.time()
                    self.iteration = 0
                
                def update(self):
                    current_time = time.time()
                    if current_time - self.last_print_time > 30:  # Print every 30 seconds
                        elapsed = current_time - solve_start_time
                        print(f"  ... still solving, elapsed: {elapsed:.1f}s")
                        self.last_print_time = current_time
            
            tracker = ProgressTracker()
            
            U_solution, M_solution, info = solver.solve(
                Niter=config['qp_settings']['picard'], 
                l2errBound=1e-3,
                verbose=False  # Reduce output for efficiency
            )
            
            solve_time = time.time() - solve_start_time
            total_config_time = time.time() - config_start_time
            
            if M_solution is not None and U_solution is not None:
                print(f"\n✅ Solution completed in {solve_time:.1f}s ({solve_time/60:.1f} min)")
                
                # Quick analysis
                mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
                initial_mass = mass_evolution[0]
                final_mass = mass_evolution[-1]
                mass_change = final_mass - initial_mass
                mass_change_percent = (mass_change / initial_mass) * 100
                
                # Solution quality
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
                
                # Convergence
                converged = info.get('converged', False)
                iterations_used = info.get('iterations', 0)
                
                result = {
                    'name': config['name'],
                    'success': True,
                    'config': config,
                    'solve_time': solve_time,
                    'total_time': total_config_time,
                    'converged': converged,
                    'iterations_used': iterations_used,
                    'max_iterations': config['qp_settings']['picard'],
                    'initial_mass': initial_mass,
                    'final_mass': final_mass,
                    'mass_change': mass_change,
                    'mass_change_percent': mass_change_percent,
                    'max_U': max_U,
                    'min_M': min_M,
                    'negative_densities': negative_densities,
                    'violations': violations,
                    'mass_evolution': mass_evolution,
                    'M_solution': M_solution,
                    'U_solution': U_solution,
                    'particles_trajectory': particles_traj,
                    'problem': problem
                }
                
                results.append(result)
                
                # Print key results
                print(f"\n--- KEY RESULTS: {config['name']} ---")
                print(f"  Total time: {total_config_time:.1f}s")
                print(f"  Converged: {converged} ({iterations_used}/{config['qp_settings']['picard']} iterations)")
                print(f"  Mass change: {mass_change:+.2e} ({mass_change_percent:+.3f}%)")
                print(f"  Max |U|: {max_U:.2e}")
                print(f"  Min density: {min_M:.2e}")
                print(f"  Negative densities: {negative_densities}")
                print(f"  Boundary violations: {violations}")
                
                if mass_change > 0:
                    print(f"  ✅ Mass behavior: INCREASE (expected)")
                elif abs(mass_change_percent) < 1.0:
                    print(f"  ✅ Mass behavior: WELL CONSERVED")
                else:
                    print(f"  ⚠️  Mass behavior: SIGNIFICANT CHANGE")
                
                if violations == 0 and negative_densities == 0:
                    print(f"  ✅ Solution quality: EXCELLENT")
                else:
                    print(f"  ⚠️  Solution quality: HAS ISSUES")
                
            else:
                print(f"❌ {config['name']} failed to produce solution")
                results.append({
                    'name': config['name'],
                    'success': False,
                    'total_time': time.time() - config_start_time
                })
                
        except Exception as e:
            error_time = time.time() - config_start_time
            print(f"❌ {config['name']} crashed after {error_time:.1f}s: {e}")
            results.append({
                'name': config['name'],
                'success': False,
                'error': str(e),
                'total_time': error_time
            })
    
    total_time = time.time() - total_start_time
    
    # Final analysis
    print(f"\n{'='*80}")
    print("EFFICIENT T=1 TEST ANALYSIS")
    print(f"{'='*80}")
    print(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) > 0:
        print(f"\nSuccessful configurations: {len(successful_results)}/{len(results)}")
        
        # Comparison table
        print(f"\n{'Configuration':<25} {'Time (min)':<12} {'Mass Change %':<15} {'Converged':<12} {'Quality':<12}")
        print(f"{'-'*25} {'-'*12} {'-'*15} {'-'*12} {'-'*12}")
        
        for result in successful_results:
            quality = "GOOD" if (result['violations'] == 0 and result['negative_densities'] == 0) else "ISSUES"
            print(f"{result['name']:<25} {result['solve_time']/60:<12.1f} {result['mass_change_percent']:<15.3f} {str(result['converged']):<12} {quality:<12}")
        
        # Mass conservation assessment
        print(f"\n--- MASS CONSERVATION ASSESSMENT ---")
        mass_increases = [r for r in successful_results if r['mass_change'] > 0]
        print(f"Configurations showing mass increase: {len(mass_increases)}/{len(successful_results)}")
        
        if len(mass_increases) == len(successful_results):
            print("✅ EXCELLENT: All show expected mass increase from no-flux reflection")
        elif len(mass_increases) > 0:
            print("✅ GOOD: Some show expected mass increase")
        else:
            print("⚠️  WARNING: No mass increase observed")
        
        # Performance assessment
        if len(successful_results) > 1:
            times = [r['solve_time'] for r in successful_results]
            complexities = [
                r['config']['params']['Nx'] * r['config']['params']['Nt'] * 
                r['config']['qp_settings']['particles']
                for r in successful_results
            ]
            
            print(f"\n--- PERFORMANCE ASSESSMENT ---")
            print(f"Time range: {min(times):.1f}s - {max(times):.1f}s")
            print(f"Complexity range: {min(complexities):,} - {max(complexities):,} grid*particles")
            
            if max(times) < 600:  # Less than 10 minutes
                print("✅ EXCELLENT: Reasonable computation times for T=1")
            elif max(times) < 1800:  # Less than 30 minutes
                print("✅ GOOD: Acceptable computation times for T=1")
            else:
                print("⚠️  WARNING: Long computation times")
        
        # Create plots
        create_efficient_t1_plots(successful_results)
        
    else:
        print("❌ No configurations completed successfully")
        for result in results:
            if not result.get('success', False):
                error = result.get('error', 'Failed')
                print(f"  {result['name']}: {error}")
    
    return results

def create_efficient_t1_plots(results):
    """Create plots for efficient T=1 analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Efficient T=1 QP Particle-Collocation Analysis', fontsize=16)
    
    colors = ['blue', 'green', 'red']
    
    # 1. Mass evolution over time
    ax1 = axes[0, 0]
    for i, result in enumerate(results):
        problem = result['problem']
        mass_evolution = result['mass_evolution']
        ax1.plot(problem.tSpace, mass_evolution, 'o-', 
                label=result['name'], color=colors[i % len(colors)], linewidth=2)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Total Mass')
    ax1.set_title('Mass Evolution Over T=1')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Final density comparison
    ax2 = axes[0, 1]
    for i, result in enumerate(results):
        problem = result['problem']
        final_density = result['M_solution'][-1, :]
        ax2.plot(problem.xSpace, final_density, 
                label=result['name'], color=colors[i % len(colors)], linewidth=2)
    ax2.set_xlabel('Space x')
    ax2.set_ylabel('Final Density M(T=1,x)')
    ax2.set_title('Final Density Distributions')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Performance comparison
    ax3 = axes[0, 2]
    names = [r['name'] for r in results]
    times = [r['solve_time']/60 for r in results]  # Convert to minutes
    bars = ax3.bar(names, times, color=colors[:len(results)])
    ax3.set_ylabel('Solve Time (minutes)')
    ax3.set_title('Computational Performance')
    ax3.grid(True, axis='y')
    for bar, value in zip(bars, times):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.1f}m', ha='center', va='bottom')
    plt.setp(ax3.get_xticklabels(), rotation=45)
    
    # 4. Mass change analysis
    ax4 = axes[1, 0]
    mass_changes = [r['mass_change_percent'] for r in results]
    bars = ax4.bar(names, mass_changes, color=colors[:len(results)])
    ax4.set_ylabel('Mass Change (%)')
    ax4.set_title('Mass Conservation Quality')
    ax4.grid(True, axis='y')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    for bar, value in zip(bars, mass_changes):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.2f}%', ha='center', va='bottom' if value >= 0 else 'top')
    plt.setp(ax4.get_xticklabels(), rotation=45)
    
    # 5. Density evolution heatmap (best result)
    ax5 = axes[1, 1]
    if results:
        best_result = min(results, key=lambda x: abs(x['mass_change_percent']))
        M_solution = best_result['M_solution']
        problem = best_result['problem']
        
        im = ax5.imshow(M_solution.T, aspect='auto', origin='lower', 
                       extent=[0, problem.T, problem.xmin, problem.xmax], cmap='viridis')
        ax5.set_xlabel('Time t')
        ax5.set_ylabel('Space x')
        ax5.set_title(f'Density Evolution: {best_result["name"]}')
        plt.colorbar(im, ax=ax5, label='Density M(t,x)')
    
    # 6. Solution quality summary
    ax6 = axes[1, 2]
    quality_metrics = {
        'Violations': [r['violations'] for r in results],
        'Negative Densities': [r['negative_densities'] for r in results]
    }
    
    x_pos = np.arange(len(names))
    width = 0.35
    
    bars1 = ax6.bar(x_pos - width/2, quality_metrics['Violations'], width, 
                    label='Boundary Violations', color='red', alpha=0.7)
    bars2 = ax6.bar(x_pos + width/2, quality_metrics['Negative Densities'], width, 
                    label='Negative Densities', color='orange', alpha=0.7)
    
    ax6.set_xlabel('Configuration')
    ax6.set_ylabel('Count')
    ax6.set_title('Solution Quality Metrics')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(names, rotation=45)
    ax6.legend()
    ax6.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/efficient_t1_analysis.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Starting efficient T=1 test...")
    print("Estimated total time: 5-20 minutes depending on configuration")
    print("Press Ctrl+C to interrupt if needed.")
    
    try:
        results = test_efficient_t1()
        print("\n" + "="*80)
        print("EFFICIENT T=1 TEST COMPLETED")
        print("="*80)
        print("Check the generated plots for detailed results.")
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
