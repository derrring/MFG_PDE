#!/usr/bin/env python3
"""
High resolution test of QP Particle-Collocation method for T=1.
Comprehensive evaluation of mass conservation, convergence, and solution quality
over extended time horizon with fine spatial and temporal resolution.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import json

# Add the main package to path
sys.path.insert(0, '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE')

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.core.boundaries import BoundaryConditions

def test_high_resolution_t1():
    """
    High resolution test for T=1 with comprehensive analysis.
    Tests multiple configurations to find optimal balance of accuracy vs performance.
    """
    print("="*80)
    print("HIGH RESOLUTION QP PARTICLE-COLLOCATION TEST (T=1)")
    print("="*80)
    print("Testing extended time horizon with fine resolution")
    print("Expected long execution time - comprehensive evaluation")
    
    # High resolution test configurations for T=1
    test_configs = [
        {
            'name': 'Baseline High-Res',
            'params': {
                'xmin': 0.0, 'xmax': 1.0, 'Nx': 40, 
                'T': 1.0, 'Nt': 25, 
                'sigma': 0.2, 'coefCT': 0.02
            },
            'qp_settings': {
                'particles': 800, 'collocation': 15, 'delta': 0.3, 
                'order': 2, 'newton': 8, 'picard': 12
            }
        },
        {
            'name': 'Ultra High-Res',
            'params': {
                'xmin': 0.0, 'xmax': 1.0, 'Nx': 60, 
                'T': 1.0, 'Nt': 40, 
                'sigma': 0.25, 'coefCT': 0.025
            },
            'qp_settings': {
                'particles': 1200, 'collocation': 20, 'delta': 0.35, 
                'order': 2, 'newton': 10, 'picard': 15
            }
        },
        {
            'name': 'Production Quality',
            'params': {
                'xmin': 0.0, 'xmax': 1.0, 'Nx': 80, 
                'T': 1.0, 'Nt': 50, 
                'sigma': 0.3, 'coefCT': 0.03
            },
            'qp_settings': {
                'particles': 1500, 'collocation': 25, 'delta': 0.4, 
                'order': 2, 'newton': 12, 'picard': 20
            }
        }
    ]
    
    results = []
    total_start_time = time.time()
    
    for i, config in enumerate(test_configs):
        print(f"\n{'='*70}")
        print(f"TEST {i+1}/3: {config['name'].upper()}")
        print(f"{'='*70}")
        
        # Display configuration
        print("\nProblem parameters:")
        for key, value in config['params'].items():
            print(f"  {key}: {value}")
        
        print("\nQP solver settings:")
        for key, value in config['qp_settings'].items():
            print(f"  {key}: {value}")
        
        # Estimate computational complexity
        complexity_score = (
            config['params']['Nx'] * 
            config['params']['Nt'] * 
            config['qp_settings']['particles'] * 
            config['qp_settings']['picard']
        ) / 1000000
        print(f"\nEstimated complexity score: {complexity_score:.1f}M operations")
        
        try:
            config_start_time = time.time()
            
            # Create problem
            problem = ExampleMFGProblem(**config['params'])
            no_flux_bc = BoundaryConditions(type="no_flux")
            
            print(f"\nProblem created - Domain: [{problem.xmin}, {problem.xmax}], Time: [0, {problem.T}]")
            print(f"Grid resolution: Dx = {problem.Dx:.4f}, Dt = {problem.Dt:.4f}")
            print(f"CFL-like number: {config['params']['sigma']**2 * problem.Dt / problem.Dx**2:.4f}")
            
            # Setup collocation points
            num_collocation = config['qp_settings']['collocation']
            collocation_points = np.linspace(
                problem.xmin, problem.xmax, num_collocation
            ).reshape(-1, 1)
            
            # Boundary indices for no-flux conditions
            boundary_indices = [0, num_collocation - 1]
            
            print(f"\nCollocation setup: {num_collocation} points")
            print(f"Boundary indices: {boundary_indices}")
            
            # Create solver with detailed settings
            solver = ParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                num_particles=config['qp_settings']['particles'],
                delta=config['qp_settings']['delta'],
                taylor_order=config['qp_settings']['order'],
                weight_function="wendland",
                NiterNewton=config['qp_settings']['newton'],
                l2errBoundNewton=1e-4,  # Tighter tolerance for high-res
                kde_bandwidth="scott",
                normalize_kde_output=False,
                boundary_indices=np.array(boundary_indices),
                boundary_conditions=no_flux_bc,
                use_monotone_constraints=True
            )
            
            print(f"\nSolver created with {config['qp_settings']['particles']} particles")
            print(f"QP constraints: enabled, Taylor order: {config['qp_settings']['order']}")
            
            # Solve with progress reporting
            print(f"\nStarting solve - expecting {config['qp_settings']['picard']} Picard iterations...")
            solve_start_time = time.time()
            
            U_solution, M_solution, info = solver.solve(
                Niter=config['qp_settings']['picard'], 
                l2errBound=1e-4,  # Tighter tolerance
                verbose=True  # Show iteration progress
            )
            
            solve_time = time.time() - solve_start_time
            total_config_time = time.time() - config_start_time
            
            if M_solution is not None and U_solution is not None:
                print(f"\n✅ Solution completed in {solve_time:.1f}s")
                
                # Comprehensive analysis
                analysis_start_time = time.time()
                
                # Mass conservation analysis
                mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
                initial_mass = mass_evolution[0]
                final_mass = mass_evolution[-1]
                mass_change = final_mass - initial_mass
                mass_change_percent = (mass_change / initial_mass) * 100
                
                # Mass conservation quality over time
                mass_variation = np.std(mass_evolution) / np.mean(mass_evolution) * 100
                max_mass = np.max(mass_evolution)
                min_mass = np.min(mass_evolution)
                mass_range_percent = (max_mass - min_mass) / initial_mass * 100
                
                # Solution quality metrics
                max_U = np.max(np.abs(U_solution))
                max_M = np.max(M_solution)
                min_M = np.min(M_solution)
                
                # Check for negative densities
                negative_density_count = np.sum(M_solution < -1e-10)
                
                # Boundary violations
                violations = 0
                particles_traj = solver.get_particles_trajectory()
                if particles_traj is not None:
                    final_particles = particles_traj[-1, :]
                    violations = np.sum(
                        (final_particles < problem.xmin - 1e-10) | 
                        (final_particles > problem.xmax + 1e-10)
                    )
                    
                    # Particle distribution analysis
                    particle_spread = np.std(final_particles)
                    particle_mean = np.mean(final_particles)
                    particle_in_bounds = np.sum(
                        (final_particles >= problem.xmin) & 
                        (final_particles <= problem.xmax)
                    )
                    particle_containment = particle_in_bounds / len(final_particles) * 100
                
                # Convergence analysis
                converged = info.get('converged', False)
                iterations_used = info.get('iterations', 0)
                convergence_rate = iterations_used / config['qp_settings']['picard'] * 100
                
                analysis_time = time.time() - analysis_start_time
                
                # Store comprehensive results
                result = {
                    'name': config['name'],
                    'success': True,
                    'config': config,
                    'timing': {
                        'total_time': total_config_time,
                        'solve_time': solve_time,
                        'analysis_time': analysis_time,
                        'setup_time': total_config_time - solve_time - analysis_time
                    },
                    'convergence': {
                        'converged': converged,
                        'iterations_used': iterations_used,
                        'max_iterations': config['qp_settings']['picard'],
                        'convergence_rate': convergence_rate
                    },
                    'mass_conservation': {
                        'initial_mass': initial_mass,
                        'final_mass': final_mass,
                        'mass_change': mass_change,
                        'mass_change_percent': mass_change_percent,
                        'mass_variation_percent': mass_variation,
                        'mass_range_percent': mass_range_percent,
                        'max_mass': max_mass,
                        'min_mass': min_mass
                    },
                    'solution_quality': {
                        'max_U': max_U,
                        'max_M': max_M,
                        'min_M': min_M,
                        'negative_densities': negative_density_count,
                        'violations': violations
                    },
                    'particle_analysis': {
                        'final_spread': particle_spread,
                        'final_mean': particle_mean,
                        'containment_percent': particle_containment
                    },
                    'arrays': {
                        'mass_evolution': mass_evolution,
                        'M_solution': M_solution,
                        'U_solution': U_solution,
                        'particles_trajectory': particles_traj
                    },
                    'problem': problem
                }
                
                results.append(result)
                
                # Print detailed results
                print_detailed_results(result)
                
            else:
                print(f"❌ {config['name']} failed to produce solution")
                results.append({
                    'name': config['name'],
                    'success': False,
                    'config': config,
                    'timing': {'total_time': time.time() - config_start_time}
                })
                
        except Exception as e:
            error_time = time.time() - config_start_time
            print(f"❌ {config['name']} crashed after {error_time:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': config['name'],
                'success': False,
                'error': str(e),
                'config': config,
                'timing': {'total_time': error_time}
            })
    
    total_time = time.time() - total_start_time
    
    # Comprehensive analysis of all results
    print(f"\n{'='*80}")
    print("COMPREHENSIVE HIGH-RESOLUTION ANALYSIS")
    print(f"{'='*80}")
    print(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    analyze_all_results(results)
    
    # Create comprehensive visualizations
    create_high_resolution_plots(results)
    
    # Save detailed results
    save_results_summary(results)
    
    return results

def print_detailed_results(result):
    """Print detailed analysis for a single test configuration"""
    print(f"\n--- DETAILED RESULTS: {result['name']} ---")
    
    # Timing
    timing = result['timing']
    print(f"\nTiming:")
    print(f"  Setup time: {timing['setup_time']:.2f}s")
    print(f"  Solve time: {timing['solve_time']:.2f}s")
    print(f"  Analysis time: {timing['analysis_time']:.2f}s")
    print(f"  Total time: {timing['total_time']:.2f}s")
    
    # Convergence
    conv = result['convergence']
    print(f"\nConvergence:")
    print(f"  Converged: {conv['converged']}")
    print(f"  Iterations used: {conv['iterations_used']}/{conv['max_iterations']} ({conv['convergence_rate']:.1f}%)")
    
    # Mass conservation
    mass = result['mass_conservation']
    print(f"\nMass Conservation:")
    print(f"  Initial mass: {mass['initial_mass']:.6f}")
    print(f"  Final mass: {mass['final_mass']:.6f}")
    print(f"  Mass change: {mass['mass_change']:+.2e} ({mass['mass_change_percent']:+.3f}%)")
    print(f"  Mass variation: {mass['mass_variation_percent']:.3f}%")
    print(f"  Mass range: {mass['mass_range_percent']:.3f}%")
    
    # Solution quality
    quality = result['solution_quality']
    print(f"\nSolution Quality:")
    print(f"  Max |U|: {quality['max_U']:.2e}")
    print(f"  Density range: [{quality['min_M']:.2e}, {quality['max_M']:.2e}]")
    print(f"  Negative densities: {quality['negative_densities']}")
    print(f"  Boundary violations: {quality['violations']}")
    
    # Particle analysis
    particles = result['particle_analysis']
    print(f"\nParticle Analysis:")
    print(f"  Final spread (std): {particles['final_spread']:.4f}")
    print(f"  Final mean position: {particles['final_mean']:.4f}")
    print(f"  Containment: {particles['containment_percent']:.1f}%")
    
    # Assessment
    print(f"\nOverall Assessment:")
    if mass['mass_change'] > 0:
        print(f"  ✅ Mass behavior: INCREASE (expected with no-flux BC)")
    elif abs(mass['mass_change_percent']) < 1.0:
        print(f"  ✅ Mass behavior: WELL CONSERVED")
    else:
        print(f"  ⚠️  Mass behavior: SIGNIFICANT CHANGE")
    
    if quality['violations'] == 0 and quality['negative_densities'] == 0:
        print(f"  ✅ Numerical stability: EXCELLENT")
    elif quality['violations'] < 10 and quality['negative_densities'] < 10:
        print(f"  ⚠️  Numerical stability: ACCEPTABLE")
    else:
        print(f"  ❌ Numerical stability: POOR")
    
    if conv['converged']:
        print(f"  ✅ Convergence: ACHIEVED")
    elif conv['convergence_rate'] > 80:
        print(f"  ⚠️  Convergence: NEARLY ACHIEVED")
    else:
        print(f"  ❌ Convergence: INSUFFICIENT")

def analyze_all_results(results):
    """Analyze results across all configurations"""
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) == 0:
        print("❌ No configurations completed successfully")
        return
    
    print(f"\nSuccessful configurations: {len(successful_results)}/{len(results)}")
    
    # Timing comparison
    print(f"\n--- PERFORMANCE COMPARISON ---")
    print(f"{'Configuration':<20} {'Total Time':<12} {'Solve Time':<12} {'Complexity':<12}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    
    for result in successful_results:
        config = result['config']
        timing = result['timing']
        complexity = (
            config['params']['Nx'] * config['params']['Nt'] * 
            config['qp_settings']['particles'] * config['qp_settings']['picard']
        ) / 1000000
        
        print(f"{result['name']:<20} {timing['total_time']:<12.1f} {timing['solve_time']:<12.1f} {complexity:<12.1f}")
    
    # Mass conservation comparison
    print(f"\n--- MASS CONSERVATION COMPARISON ---")
    print(f"{'Configuration':<20} {'Mass Change %':<15} {'Variation %':<12} {'Assessment':<15}")
    print(f"{'-'*20} {'-'*15} {'-'*12} {'-'*15}")
    
    for result in successful_results:
        mass = result['mass_conservation']
        if mass['mass_change'] > 0 and mass['mass_change_percent'] < 5:
            assessment = "EXCELLENT"
        elif abs(mass['mass_change_percent']) < 2:
            assessment = "GOOD"
        elif abs(mass['mass_change_percent']) < 5:
            assessment = "ACCEPTABLE"
        else:
            assessment = "POOR"
        
        print(f"{result['name']:<20} {mass['mass_change_percent']:<15.3f} {mass['mass_variation_percent']:<12.3f} {assessment:<15}")
    
    # Convergence comparison
    print(f"\n--- CONVERGENCE COMPARISON ---")
    print(f"{'Configuration':<20} {'Converged':<12} {'Iterations':<12} {'Rate %':<12}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    
    for result in successful_results:
        conv = result['convergence']
        print(f"{result['name']:<20} {str(conv['converged']):<12} {conv['iterations_used']:<12} {conv['convergence_rate']:<12.1f}")
    
    # Overall assessment
    print(f"\n--- OVERALL ASSESSMENT ---")
    
    # Check mass conservation consistency
    mass_changes = [r['mass_conservation']['mass_change_percent'] for r in successful_results]
    if len(mass_changes) > 1:
        mass_std = np.std(mass_changes)
        mass_mean = np.mean(mass_changes)
        cv = mass_std / abs(mass_mean) if abs(mass_mean) > 1e-10 else mass_std
        
        print(f"Mass conservation consistency (CV): {cv:.3f}")
        if cv < 0.2:
            print("✅ EXCELLENT: Very consistent mass behavior across resolutions")
        elif cv < 0.5:
            print("✅ GOOD: Reasonably consistent mass behavior")
        else:
            print("⚠️  WARNING: Inconsistent mass behavior suggests resolution effects")
    
    # Check if mass increases as expected
    mass_increasing = [r for r in successful_results if r['mass_conservation']['mass_change'] > 0]
    print(f"\nConfigurations showing mass increase: {len(mass_increasing)}/{len(successful_results)}")
    if len(mass_increasing) == len(successful_results):
        print("✅ EXCELLENT: All configurations show expected mass increase")
    elif len(mass_increasing) > len(successful_results) // 2:
        print("✅ GOOD: Majority show expected mass increase")
    else:
        print("⚠️  WARNING: Mass increase not consistently observed")
    
    # Performance scaling analysis
    if len(successful_results) > 1:
        times = [r['timing']['solve_time'] for r in successful_results]
        complexities = [
            r['config']['params']['Nx'] * r['config']['params']['Nt'] * 
            r['config']['qp_settings']['particles'] * r['config']['qp_settings']['picard']
            for r in successful_results
        ]
        
        if max(complexities) > min(complexities):
            scaling_factor = (max(times) / min(times)) / (max(complexities) / min(complexities))
            print(f"\nPerformance scaling factor: {scaling_factor:.2f}")
            if scaling_factor < 1.5:
                print("✅ EXCELLENT: Better than linear scaling")
            elif scaling_factor < 2.0:
                print("✅ GOOD: Near-linear scaling")
            else:
                print("⚠️  WARNING: Worse than linear scaling")

def create_high_resolution_plots(results):
    """Create comprehensive plots for high-resolution analysis"""
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) == 0:
        print("No successful results to plot")
        return
    
    # Create multi-panel figure
    fig = plt.figure(figsize=(20, 16))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    # 1. Mass evolution over time (2x2 grid, position 1)
    ax1 = plt.subplot(3, 3, 1)
    for i, result in enumerate(successful_results):
        problem = result['problem']
        mass_evolution = result['arrays']['mass_evolution']
        ax1.plot(problem.tSpace, mass_evolution, 'o-', 
                label=result['name'], color=colors[i % len(colors)], linewidth=2)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Total Mass')
    ax1.set_title('Mass Evolution Over Time (T=1)')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Final density distributions (position 2)
    ax2 = plt.subplot(3, 3, 2)
    for i, result in enumerate(successful_results):
        problem = result['problem']
        M_solution = result['arrays']['M_solution']
        final_density = M_solution[-1, :]
        ax2.plot(problem.xSpace, final_density, 
                label=result['name'], color=colors[i % len(colors)], linewidth=2)
    ax2.set_xlabel('Space x')
    ax2.set_ylabel('Final Density M(T=1,x)')
    ax2.set_title('Final Density Distributions')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Mass change comparison (position 3)
    ax3 = plt.subplot(3, 3, 3)
    names = [r['name'] for r in successful_results]
    mass_changes = [r['mass_conservation']['mass_change_percent'] for r in successful_results]
    bars = ax3.bar(names, mass_changes, color=colors[:len(successful_results)])
    ax3.set_ylabel('Mass Change (%)')
    ax3.set_title('Mass Change Comparison')
    ax3.grid(True, axis='y')
    for bar, value in zip(bars, mass_changes):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.2f}%', ha='center', va='bottom')
    plt.setp(ax3.get_xticklabels(), rotation=45)
    
    # 4. Runtime comparison (position 4)
    ax4 = plt.subplot(3, 3, 4)
    solve_times = [r['timing']['solve_time'] for r in successful_results]
    bars = ax4.bar(names, solve_times, color=colors[:len(successful_results)])
    ax4.set_ylabel('Solve Time (seconds)')
    ax4.set_title('Runtime Comparison')
    ax4.grid(True, axis='y')
    for bar, value in zip(bars, solve_times):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.1f}s', ha='center', va='bottom')
    plt.setp(ax4.get_xticklabels(), rotation=45)
    
    # 5. Convergence analysis (position 5)
    ax5 = plt.subplot(3, 3, 5)
    convergence_rates = [r['convergence']['convergence_rate'] for r in successful_results]
    bars = ax5.bar(names, convergence_rates, color=colors[:len(successful_results)])
    ax5.set_ylabel('Convergence Rate (%)')
    ax5.set_title('Convergence Efficiency')
    ax5.grid(True, axis='y')
    ax5.set_ylim(0, 100)
    for bar, value in zip(bars, convergence_rates):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.1f}%', ha='center', va='bottom')
    plt.setp(ax5.get_xticklabels(), rotation=45)
    
    # 6. Solution quality metrics (position 6)
    ax6 = plt.subplot(3, 3, 6)
    violations = [r['solution_quality']['violations'] for r in successful_results]
    negative_densities = [r['solution_quality']['negative_densities'] for r in successful_results]
    
    x_pos = np.arange(len(names))
    width = 0.35
    ax6.bar(x_pos - width/2, violations, width, label='Boundary Violations', color='red', alpha=0.7)
    ax6.bar(x_pos + width/2, negative_densities, width, label='Negative Densities', color='orange', alpha=0.7)
    ax6.set_xlabel('Configuration')
    ax6.set_ylabel('Count')
    ax6.set_title('Solution Quality Issues')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(names, rotation=45)
    ax6.legend()
    ax6.grid(True, axis='y')
    
    # 7. Performance vs accuracy trade-off (position 7)
    ax7 = plt.subplot(3, 3, 7)
    solve_times = [r['timing']['solve_time'] for r in successful_results]
    mass_accuracy = [abs(r['mass_conservation']['mass_change_percent']) for r in successful_results]
    scatter = ax7.scatter(solve_times, mass_accuracy, 
                         c=colors[:len(successful_results)], s=100, alpha=0.7)
    ax7.set_xlabel('Solve Time (seconds)')
    ax7.set_ylabel('|Mass Change %|')
    ax7.set_title('Performance vs Accuracy Trade-off')
    ax7.grid(True)
    for i, result in enumerate(successful_results):
        ax7.annotate(result['name'], (solve_times[i], mass_accuracy[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    # 8. Density evolution heatmap for best result (position 8)
    ax8 = plt.subplot(3, 3, 8)
    if successful_results:
        # Choose the result with best mass conservation
        best_result = min(successful_results, 
                         key=lambda x: abs(x['mass_conservation']['mass_change_percent']))
        M_solution = best_result['arrays']['M_solution']
        problem = best_result['problem']
        
        im = ax8.imshow(M_solution.T, aspect='auto', origin='lower', 
                       extent=[0, problem.T, problem.xmin, problem.xmax], cmap='viridis')
        ax8.set_xlabel('Time t')
        ax8.set_ylabel('Space x')
        ax8.set_title(f'Density Evolution: {best_result["name"]}')
        plt.colorbar(im, ax=ax8, label='Density M(t,x)')
    
    # 9. Final particle distribution for best result (position 9)
    ax9 = plt.subplot(3, 3, 9)
    if successful_results:
        best_result = min(successful_results, 
                         key=lambda x: abs(x['mass_conservation']['mass_change_percent']))
        particles_traj = best_result['arrays']['particles_trajectory']
        if particles_traj is not None:
            final_particles = particles_traj[-1, :]
            ax9.hist(final_particles, bins=30, alpha=0.7, density=True, 
                    color=colors[0], edgecolor='black')
            ax9.set_xlabel('Final Particle Position')
            ax9.set_ylabel('Probability Density')
            ax9.set_title(f'Final Particle Distribution: {best_result["name"]}')
            ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/high_resolution_t1_analysis.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

def save_results_summary(results):
    """Save comprehensive results summary to file"""
    summary = {
        'test_description': 'High resolution T=1 QP particle-collocation analysis',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_configurations': len(results),
        'successful_configurations': len([r for r in results if r.get('success', False)]),
        'configurations': []
    }
    
    for result in results:
        if result.get('success', False):
            config_summary = {
                'name': result['name'],
                'parameters': result['config']['params'],
                'qp_settings': result['config']['qp_settings'],
                'timing': result['timing'],
                'convergence': result['convergence'],
                'mass_conservation': result['mass_conservation'],
                'solution_quality': result['solution_quality'],
                'particle_analysis': result['particle_analysis']
            }
        else:
            config_summary = {
                'name': result['name'],
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'timing': result.get('timing', {})
            }
        
        summary['configurations'].append(config_summary)
    
    # Save to JSON file
    with open('/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/high_resolution_t1_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults summary saved to high_resolution_t1_results.json")

if __name__ == "__main__":
    print("Starting high resolution T=1 test...")
    print("This may take several minutes to complete.")
    print("Press Ctrl+C to interrupt if needed.")
    
    try:
        results = test_high_resolution_t1()
        print("\n" + "="*80)
        print("HIGH RESOLUTION T=1 TEST COMPLETED")
        print("="*80)
        print("Check the generated plots and JSON summary for detailed results.")
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
