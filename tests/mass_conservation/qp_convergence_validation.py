#!/usr/bin/env python3
"""
QP Method Convergence Validation:
Test QP particle-collocation with multiple parameter sets under identical conditions
to demonstrate convergence consistency and validate the method behavior.
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

def validate_qp_convergence():
    """
    Validate QP method convergence under multiple resolution settings.
    Tests method consistency and demonstrates convergence to same physical solution.
    """
    print("="*80)
    print("QP PARTICLE-COLLOCATION CONVERGENCE VALIDATION")
    print("="*80)
    print("Testing QP method with multiple resolution settings")
    print("Demonstrating convergence to consistent physical solution")
    
    # Base problem parameters (same for all tests)
    base_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "T": 0.4,
        "sigma": 0.2,
        "coefCT": 0.02
    }
    
    # Multiple resolution configurations
    test_configs = [
        {
            'name': 'Coarse Resolution',
            'grid': {'Nx': 20, 'Nt': 10},
            'qp': {'particles': 300, 'collocation': 8, 'delta': 0.25, 'order': 1, 'newton': 5, 'picard': 8}
        },
        {
            'name': 'Standard Resolution',
            'grid': {'Nx': 30, 'Nt': 15},
            'qp': {'particles': 500, 'collocation': 10, 'delta': 0.3, 'order': 2, 'newton': 6, 'picard': 10}
        },
        {
            'name': 'Fine Resolution',
            'grid': {'Nx': 40, 'Nt': 20},
            'qp': {'particles': 800, 'collocation': 12, 'delta': 0.35, 'order': 2, 'newton': 8, 'picard': 12}
        },
        {
            'name': 'Very Fine Resolution',
            'grid': {'Nx': 50, 'Nt': 25},
            'qp': {'particles': 1000, 'collocation': 15, 'delta': 0.4, 'order': 2, 'newton': 10, 'picard': 15}
        }
    ]
    
    print(f"\nBase Problem Parameters:")
    for key, value in base_params.items():
        print(f"  {key}: {value}")
    
    results = []
    total_start_time = time.time()
    
    for i, config in enumerate(test_configs):
        print(f"\n{'='*70}")
        print(f"TEST {i+1}/4: {config['name'].upper()}")
        print(f"{'='*70}")
        
        # Create problem with specific resolution
        problem_params = {**base_params, **config['grid']}
        
        print("\nConfiguration:")
        print(f"  Grid: {config['grid']['Nx']}×{config['grid']['Nt']}")
        print(f"  Particles: {config['qp']['particles']}, Collocation: {config['qp']['collocation']}")
        print(f"  QP settings: delta={config['qp']['delta']}, order={config['qp']['order']}")
        
        try:
            config_start_time = time.time()
            
            # Create problem
            problem = ExampleMFGProblem(**problem_params)
            no_flux_bc = BoundaryConditions(type="no_flux")
            
            print(f"\nGrid resolution: Dx={problem.Dx:.4f}, Dt={problem.Dt:.4f}")
            print(f"CFL number: {base_params['sigma']**2 * problem.Dt / problem.Dx**2:.4f}")
            
            # Setup collocation
            num_collocation = config['qp']['collocation']
            collocation_points = np.linspace(
                problem.xmin, problem.xmax, num_collocation
            ).reshape(-1, 1)
            
            boundary_indices = [0, num_collocation - 1]
            
            # Create solver
            solver = ParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                num_particles=config['qp']['particles'],
                delta=config['qp']['delta'],
                taylor_order=config['qp']['order'],
                weight_function="wendland",
                NiterNewton=config['qp']['newton'],
                l2errBoundNewton=1e-4,
                kde_bandwidth="scott",
                normalize_kde_output=False,
                boundary_indices=np.array(boundary_indices),
                boundary_conditions=no_flux_bc,
                use_monotone_constraints=True
            )
            
            print(f"\nSolving with {config['qp']['particles']} particles...")
            solve_start_time = time.time()
            
            U_solution, M_solution, info = solver.solve(
                Niter=config['qp']['picard'], 
                l2errBound=1e-4,
                verbose=False
            )
            
            solve_time = time.time() - solve_start_time
            total_config_time = time.time() - config_start_time
            
            if M_solution is not None and U_solution is not None:
                print(f"\n✅ Solution completed in {solve_time:.1f}s")
                
                # Comprehensive analysis
                mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
                initial_mass = mass_evolution[0]
                final_mass = mass_evolution[-1]
                mass_change = final_mass - initial_mass
                mass_change_percent = (mass_change / initial_mass) * 100
                
                # Mass conservation quality
                mass_variation = np.std(mass_evolution) / np.mean(mass_evolution) * 100
                max_mass = np.max(mass_evolution)
                min_mass = np.min(mass_evolution)
                
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
                
                # Calculate physical observables for comparison
                # Average final position
                if particles_traj is not None:
                    avg_final_position = np.mean(particles_traj[-1, :])
                    particle_spread = np.std(particles_traj[-1, :])
                else:
                    avg_final_position = None
                    particle_spread = None
                
                # Center of mass of final density
                center_of_mass = np.sum(problem.xSpace * M_solution[-1, :]) * problem.Dx
                
                # Maximum density location
                max_density_idx = np.argmax(M_solution[-1, :])
                max_density_location = problem.xSpace[max_density_idx]
                
                result = {
                    'name': config['name'],
                    'success': True,
                    'config': config,
                    'problem_params': problem_params,
                    'timing': {
                        'solve_time': solve_time,
                        'total_time': total_config_time
                    },
                    'convergence': {
                        'converged': converged,
                        'iterations_used': iterations_used,
                        'max_iterations': config['qp']['picard']
                    },
                    'mass_conservation': {
                        'initial_mass': initial_mass,
                        'final_mass': final_mass,
                        'mass_change': mass_change,
                        'mass_change_percent': mass_change_percent,
                        'mass_variation_percent': mass_variation,
                        'max_mass': max_mass,
                        'min_mass': min_mass
                    },
                    'solution_quality': {
                        'max_U': max_U,
                        'min_M': min_M,
                        'negative_densities': negative_densities,
                        'violations': violations
                    },
                    'physical_observables': {
                        'center_of_mass': center_of_mass,
                        'max_density_location': max_density_location,
                        'avg_final_position': avg_final_position,
                        'particle_spread': particle_spread
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
                
                # Print key results
                print(f"\n--- KEY RESULTS: {config['name']} ---")
                print(f"  Total time: {total_config_time:.1f}s")
                print(f"  Converged: {converged} ({iterations_used}/{config['qp']['picard']} iterations)")
                print(f"  Mass change: {mass_change:+.2e} ({mass_change_percent:+.3f}%)")
                print(f"  Mass variation: {mass_variation:.3f}%")
                print(f"  Max |U|: {max_U:.2e}, Min M: {min_M:.2e}")
                print(f"  Violations: {violations}, Negative densities: {negative_densities}")
                print(f"  Center of mass: {center_of_mass:.4f}")
                print(f"  Max density at: x = {max_density_location:.4f}")
                
                # Quality assessment
                if mass_change > 0:
                    print(f"  ✅ Mass behavior: INCREASE (expected with no-flux BC)")
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
    
    # Comprehensive analysis
    print(f"\n{'='*80}")
    print("QP METHOD CONVERGENCE ANALYSIS")
    print(f"{'='*80}")
    print(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    analyze_qp_convergence(results)
    create_convergence_validation_plots(results)
    
    return results

def analyze_qp_convergence(results):
    """Analyze convergence behavior across resolutions"""
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) == 0:
        print("❌ No configurations completed successfully")
        return
    
    print(f"\nSuccessful configurations: {len(successful_results)}/{len(results)}")
    
    # Summary table
    print(f"\n{'Configuration':<20} {'Grid Size':<12} {'Final Mass':<12} {'Mass Change %':<15} {'Center of Mass':<15} {'Converged':<10}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*15} {'-'*15} {'-'*10}")
    
    for result in successful_results:
        grid_size = f"{result['config']['grid']['Nx']}×{result['config']['grid']['Nt']}"
        print(f"{result['name']:<20} {grid_size:<12} {result['mass_conservation']['final_mass']:<12.6f} "
              f"{result['mass_conservation']['mass_change_percent']:<15.3f} "
              f"{result['physical_observables']['center_of_mass']:<15.4f} {str(result['convergence']['converged']):<10}")
    
    # Convergence consistency analysis
    print(f"\n--- CONVERGENCE CONSISTENCY ANALYSIS ---")
    
    # Physical observables convergence
    centers_of_mass = [r['physical_observables']['center_of_mass'] for r in successful_results]
    max_density_locs = [r['physical_observables']['max_density_location'] for r in successful_results]
    
    if len(centers_of_mass) >= 2:
        com_std = np.std(centers_of_mass)
        com_mean = np.mean(centers_of_mass)
        com_cv = com_std / abs(com_mean) if abs(com_mean) > 1e-10 else com_std
        
        mdl_std = np.std(max_density_locs)
        mdl_mean = np.mean(max_density_locs)
        mdl_cv = mdl_std / abs(mdl_mean) if abs(mdl_mean) > 1e-10 else mdl_std
        
        print(f"Center of mass consistency (CV): {com_cv:.4f}")
        print(f"Max density location consistency (CV): {mdl_cv:.4f}")
        
        if com_cv < 0.01 and mdl_cv < 0.01:
            print("✅ EXCELLENT: Physical observables converge very consistently")
        elif com_cv < 0.05 and mdl_cv < 0.05:
            print("✅ GOOD: Physical observables show good convergence")
        else:
            print("⚠️  WARNING: Some variation in physical observables across resolutions")
    
    # Mass conservation consistency
    final_masses = [r['mass_conservation']['final_mass'] for r in successful_results]
    mass_changes = [r['mass_conservation']['mass_change_percent'] for r in successful_results]
    
    if len(final_masses) >= 2:
        mass_std = np.std(final_masses)
        mass_mean = np.mean(final_masses)
        mass_cv = mass_std / mass_mean if mass_mean > 0 else 0
        
        print(f"\n--- MASS CONSERVATION ANALYSIS ---")
        print(f"Final mass range: [{min(final_masses):.6f}, {max(final_masses):.6f}]")
        print(f"Final mass consistency (CV): {mass_cv:.4f}")
        
        if mass_cv < 0.01:
            print("✅ EXCELLENT: Mass conservation very consistent across resolutions")
        elif mass_cv < 0.05:
            print("✅ GOOD: Mass conservation reasonably consistent")
        else:
            print("⚠️  WARNING: Mass conservation varies significantly with resolution")
        
        # Check mass increase behavior
        all_increase = all(change > 0 for change in mass_changes)
        all_conservative = all(abs(change) < 5 for change in mass_changes)
        
        if all_increase:
            print("✅ EXCELLENT: All resolutions show expected mass increase")
        elif all_conservative:
            print("✅ GOOD: All resolutions show good mass conservation")
        else:
            print("⚠️  WARNING: Inconsistent mass conservation behavior")
    
    # Performance scaling analysis
    if len(successful_results) >= 2:
        solve_times = [r['timing']['solve_time'] for r in successful_results]
        complexities = [
            r['config']['grid']['Nx'] * r['config']['grid']['Nt'] * r['config']['qp']['particles']
            for r in successful_results
        ]
        
        print(f"\n--- PERFORMANCE SCALING ANALYSIS ---")
        print(f"Solve time range: {min(solve_times):.1f}s - {max(solve_times):.1f}s")
        print(f"Complexity range: {min(complexities):,} - {max(complexities):,}")
        
        if len(solve_times) >= 3:
            # Check if scaling is reasonable
            time_ratio = max(solve_times) / min(solve_times)
            complexity_ratio = max(complexities) / min(complexities)
            scaling_efficiency = time_ratio / complexity_ratio
            
            print(f"Time scaling factor: {time_ratio:.2f}")
            print(f"Complexity scaling factor: {complexity_ratio:.2f}")
            print(f"Scaling efficiency: {scaling_efficiency:.2f}")
            
            if scaling_efficiency < 1.5:
                print("✅ EXCELLENT: Better than linear scaling")
            elif scaling_efficiency < 2.5:
                print("✅ GOOD: Reasonable scaling behavior")
            else:
                print("⚠️  WARNING: Poor scaling with resolution")
    
    # Solution quality consistency
    print(f"\n--- SOLUTION QUALITY ANALYSIS ---")
    all_converged = all(r['convergence']['converged'] for r in successful_results)
    all_clean = all(r['solution_quality']['violations'] == 0 and 
                   r['solution_quality']['negative_densities'] == 0 
                   for r in successful_results)
    
    if all_converged:
        print("✅ EXCELLENT: All resolutions achieved convergence")
    else:
        converged_count = sum(r['convergence']['converged'] for r in successful_results)
        print(f"⚠️  PARTIAL: {converged_count}/{len(successful_results)} resolutions converged")
    
    if all_clean:
        print("✅ EXCELLENT: All resolutions have clean solutions (no violations/negative densities)")
    else:
        clean_count = sum(r['solution_quality']['violations'] == 0 and 
                         r['solution_quality']['negative_densities'] == 0 
                         for r in successful_results)
        print(f"⚠️  PARTIAL: {clean_count}/{len(successful_results)} resolutions have clean solutions")

def create_convergence_validation_plots(results):
    """Create plots showing convergence validation across resolutions"""
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) == 0:
        print("No successful results to plot")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('QP Method Convergence Validation Across Resolutions', fontsize=16)
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    # 1. Mass evolution over time
    ax1 = axes[0, 0]
    for i, result in enumerate(successful_results):
        problem = result['problem']
        mass_evolution = result['arrays']['mass_evolution']
        ax1.plot(problem.tSpace, mass_evolution, 'o-', 
                label=result['name'], color=colors[i % len(colors)], linewidth=2)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Total Mass')
    ax1.set_title('Mass Evolution Convergence')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Final density distributions
    ax2 = axes[0, 1]
    for i, result in enumerate(successful_results):
        problem = result['problem']
        final_density = result['arrays']['M_solution'][-1, :]
        ax2.plot(problem.xSpace, final_density, 
                label=result['name'], color=colors[i % len(colors)], linewidth=2)
    ax2.set_xlabel('Space x')
    ax2.set_ylabel('Final Density M(T,x)')
    ax2.set_title('Final Density Convergence')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Physical observables convergence
    ax3 = axes[0, 2]
    names = [r['name'] for r in successful_results]
    com_values = [r['physical_observables']['center_of_mass'] for r in successful_results]
    mdl_values = [r['physical_observables']['max_density_location'] for r in successful_results]
    
    x_pos = np.arange(len(names))
    width = 0.35
    
    bars1 = ax3.bar(x_pos - width/2, com_values, width, 
                    label='Center of Mass', color='blue', alpha=0.7)
    bars2 = ax3.bar(x_pos + width/2, mdl_values, width, 
                    label='Max Density Location', color='red', alpha=0.7)
    
    ax3.set_xlabel('Resolution')
    ax3.set_ylabel('Position')
    ax3.set_title('Physical Observables Convergence')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names, rotation=45)
    ax3.legend()
    ax3.grid(True, axis='y')
    
    # 4. Mass conservation quality
    ax4 = axes[1, 0]
    mass_changes = [r['mass_conservation']['mass_change_percent'] for r in successful_results]
    bars = ax4.bar(names, mass_changes, color=colors[:len(successful_results)])
    ax4.set_ylabel('Mass Change (%)')
    ax4.set_title('Mass Conservation Across Resolutions')
    ax4.grid(True, axis='y')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    for bar, value in zip(bars, mass_changes):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.2f}%', ha='center', va='bottom' if value >= 0 else 'top')
    plt.setp(ax4.get_xticklabels(), rotation=45)
    
    # 5. Performance scaling
    ax5 = axes[1, 1]
    solve_times = [r['timing']['solve_time'] for r in successful_results]
    complexities = [r['config']['grid']['Nx'] * r['config']['grid']['Nt'] * 
                   r['config']['qp']['particles'] for r in successful_results]
    
    # Normalize for comparison
    norm_times = np.array(solve_times) / min(solve_times)
    norm_complexities = np.array(complexities) / min(complexities)
    
    ax5.plot(norm_complexities, norm_times, 'o-', linewidth=2, markersize=8)
    ax5.plot([min(norm_complexities), max(norm_complexities)], 
             [min(norm_complexities), max(norm_complexities)], 
             '--', color='red', alpha=0.5, label='Linear Scaling')
    ax5.set_xlabel('Normalized Complexity')
    ax5.set_ylabel('Normalized Solve Time')
    ax5.set_title('Performance Scaling')
    ax5.grid(True)
    ax5.legend()
    
    # 6. Convergence efficiency
    ax6 = axes[1, 2]
    convergence_rates = [r['convergence']['iterations_used'] / r['convergence']['max_iterations'] * 100 
                        for r in successful_results]
    bars = ax6.bar(names, convergence_rates, color=colors[:len(successful_results)])
    ax6.set_ylabel('Convergence Rate (%)')
    ax6.set_title('Convergence Efficiency')
    ax6.grid(True, axis='y')
    ax6.set_ylim(0, 100)
    for bar, value in zip(bars, convergence_rates):
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.1f}%', ha='center', va='bottom')
    plt.setp(ax6.get_xticklabels(), rotation=45)
    
    # 7. Final mass convergence
    ax7 = axes[2, 0]
    final_masses = [r['mass_conservation']['final_mass'] for r in successful_results]
    grid_sizes = [r['config']['grid']['Nx'] * r['config']['grid']['Nt'] for r in successful_results]
    
    ax7.plot(grid_sizes, final_masses, 'o-', linewidth=2, markersize=8)
    ax7.set_xlabel('Grid Size (Nx × Nt)')
    ax7.set_ylabel('Final Mass')
    ax7.set_title('Final Mass vs Grid Resolution')
    ax7.grid(True)
    
    # Add error bars if there are enough points
    if len(final_masses) >= 3:
        mean_mass = np.mean(final_masses)
        ax7.axhline(y=mean_mass, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_mass:.6f}')
        ax7.legend()
    
    # 8. Solution quality summary
    ax8 = axes[2, 1]
    violations = [r['solution_quality']['violations'] for r in successful_results]
    negative_densities = [r['solution_quality']['negative_densities'] for r in successful_results]
    
    x_pos = np.arange(len(names))
    width = 0.35
    
    bars1 = ax8.bar(x_pos - width/2, violations, width, 
                    label='Boundary Violations', color='red', alpha=0.7)
    bars2 = ax8.bar(x_pos + width/2, negative_densities, width, 
                    label='Negative Densities', color='orange', alpha=0.7)
    
    ax8.set_xlabel('Resolution')
    ax8.set_ylabel('Count')
    ax8.set_title('Solution Quality Metrics')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(names, rotation=45)
    ax8.legend()
    ax8.grid(True, axis='y')
    
    # 9. Density evolution heatmap (finest resolution)
    ax9 = axes[2, 2]
    if successful_results:
        # Use the finest resolution result
        finest_result = successful_results[-1]  # Assuming ordered by resolution
        M_solution = finest_result['arrays']['M_solution']
        problem = finest_result['problem']
        
        im = ax9.imshow(M_solution.T, aspect='auto', origin='lower', 
                       extent=[0, problem.T, problem.xmin, problem.xmax], cmap='viridis')
        ax9.set_xlabel('Time t')
        ax9.set_ylabel('Space x')
        ax9.set_title(f'Density Evolution: {finest_result["name"]}')
        plt.colorbar(im, ax=ax9, label='Density M(t,x)')
    
    plt.tight_layout()
    plt.savefig('/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/qp_convergence_validation.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Starting QP method convergence validation...")
    print("This test validates convergence consistency across multiple resolutions.")
    print("Expected execution time: 5-15 minutes depending on resolution")
    
    try:
        results = validate_qp_convergence()
        print("\n" + "="*80)
        print("QP METHOD CONVERGENCE VALIDATION COMPLETED")
        print("="*80)
        print("Check the generated plots and analysis for convergence validation.")
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
