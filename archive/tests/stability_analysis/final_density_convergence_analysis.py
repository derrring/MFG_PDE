#!/usr/bin/env python3
"""
Final Density Convergence Analysis

Address the key question: Why do different experiments show "always a little bit different" final densities?
Is this due to:
1. Different numerical methods converging to different solutions?
2. Parameter sensitivity and discretization effects?
3. Insufficient convergence within each method?
4. Different initial conditions or boundary treatments?

Using the proven-stable QP method to isolate and understand these differences.
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

def analyze_final_density_convergence():
    """
    Systematic analysis of why final densities differ between experiments.
    """
    print("="*80)
    print("FINAL DENSITY CONVERGENCE ANALYSIS")
    print("="*80)
    print("Investigating why final densities are 'always a little bit different'")
    print("Using the stable QP method to isolate sources of variation")
    
    # Base parameters for all tests
    base_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "T": 0.3,
        "sigma": 0.2,
        "coefCT": 0.02
    }
    
    print(f"\nBase Physical Parameters (fixed across all tests):")
    for key, value in base_params.items():
        print(f"  {key}: {value}")
    
    # Test different sources of variation
    test_scenarios = [
        {
            'name': 'Reference Standard',
            'description': 'Baseline configuration',
            'params': {'Nx': 30, 'Nt': 15},
            'qp_settings': {'particles': 400, 'collocation': 10, 'delta': 0.3, 'order': 2, 'newton': 6, 'picard': 8, 'seed': 42}
        },
        {
            'name': 'Same Config Repeat',
            'description': 'Identical parameters, different random seed',
            'params': {'Nx': 30, 'Nt': 15},
            'qp_settings': {'particles': 400, 'collocation': 10, 'delta': 0.3, 'order': 2, 'newton': 6, 'picard': 8, 'seed': 123}
        },
        {
            'name': 'Higher Resolution',
            'description': 'Finer grid discretization',
            'params': {'Nx': 40, 'Nt': 20},
            'qp_settings': {'particles': 400, 'collocation': 10, 'delta': 0.3, 'order': 2, 'newton': 6, 'picard': 8, 'seed': 42}
        },
        {
            'name': 'More Particles',
            'description': 'Higher particle resolution',
            'params': {'Nx': 30, 'Nt': 15},
            'qp_settings': {'particles': 600, 'collocation': 10, 'delta': 0.3, 'order': 2, 'newton': 6, 'picard': 8, 'seed': 42}
        },
        {
            'name': 'More Collocation',
            'description': 'Higher collocation point resolution',
            'params': {'Nx': 30, 'Nt': 15},
            'qp_settings': {'particles': 400, 'collocation': 12, 'delta': 0.3, 'order': 2, 'newton': 6, 'picard': 8, 'seed': 42}
        },
        {
            'name': 'More Convergence',
            'description': 'Higher convergence requirements',
            'params': {'Nx': 30, 'Nt': 15},
            'qp_settings': {'particles': 400, 'collocation': 10, 'delta': 0.3, 'order': 2, 'newton': 8, 'picard': 12, 'seed': 42}
        }
    ]
    
    results = []
    total_start_time = time.time()
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n{'='*60}")
        print(f"SCENARIO {i+1}/{len(test_scenarios)}: {scenario['name'].upper()}")
        print(f"{'='*60}")
        print(f"Description: {scenario['description']}")
        
        # Create problem with scenario-specific parameters
        problem_params = {**base_params, **scenario['params']}
        
        print(f"\nConfiguration:")
        print(f"  Grid: {scenario['params']['Nx']}×{scenario['params']['Nt']}")
        print(f"  Particles: {scenario['qp_settings']['particles']}, Collocation: {scenario['qp_settings']['collocation']}")
        print(f"  Random seed: {scenario['qp_settings']['seed']}")
        
        try:
            scenario_start_time = time.time()
            
            # Set random seed for reproducibility
            np.random.seed(scenario['qp_settings']['seed'])
            
            # Create problem
            problem = ExampleMFGProblem(**problem_params)
            no_flux_bc = BoundaryConditions(type="no_flux")
            
            print(f"Grid resolution: Dx={problem.Dx:.4f}, Dt={problem.Dt:.4f}")
            
            # Setup collocation
            num_collocation = scenario['qp_settings']['collocation']
            collocation_points = np.linspace(
                problem.xmin, problem.xmax, num_collocation
            ).reshape(-1, 1)
            
            boundary_indices = [0, num_collocation - 1]
            
            # Create solver
            solver = ParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                num_particles=scenario['qp_settings']['particles'],
                delta=scenario['qp_settings']['delta'],
                taylor_order=scenario['qp_settings']['order'],
                weight_function="wendland",
                NiterNewton=scenario['qp_settings']['newton'],
                l2errBoundNewton=1e-4,
                kde_bandwidth="scott",
                normalize_kde_output=False,
                boundary_indices=np.array(boundary_indices),
                boundary_conditions=no_flux_bc,
                use_monotone_constraints=True
            )
            
            print(f"\nSolving...")
            solve_start_time = time.time()
            
            U_solution, M_solution, info = solver.solve(
                Niter=scenario['qp_settings']['picard'], 
                l2errBound=1e-4,
                verbose=False
            )
            
            solve_time = time.time() - solve_start_time
            total_scenario_time = time.time() - scenario_start_time
            
            if M_solution is not None and U_solution is not None:
                print(f"✓ Completed in {solve_time:.2f}s")
                
                # Comprehensive analysis
                mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
                initial_mass = mass_evolution[0]
                final_mass = mass_evolution[-1]
                mass_change = final_mass - initial_mass
                
                # Physical observables
                center_of_mass = np.sum(problem.xSpace * M_solution[-1, :]) * problem.Dx
                max_density_idx = np.argmax(M_solution[-1, :])
                max_density_location = problem.xSpace[max_density_idx]
                final_density_peak = M_solution[-1, max_density_idx]
                
                # Density shape characterization
                # Calculate moments
                final_density = M_solution[-1, :]
                # Second moment (spread)
                second_moment = np.sum(((problem.xSpace - center_of_mass)**2) * final_density) * problem.Dx
                density_spread = np.sqrt(second_moment)
                
                # Calculate density at specific points for comparison
                density_at_quarter = np.interp(0.25, problem.xSpace, final_density)
                density_at_half = np.interp(0.5, problem.xSpace, final_density)
                density_at_three_quarter = np.interp(0.75, problem.xSpace, final_density)
                
                # Solution quality
                max_U = np.max(np.abs(U_solution))
                min_M = np.min(M_solution)
                negative_densities = np.sum(M_solution < -1e-10)
                
                # Convergence quality
                converged = info.get('converged', False)
                iterations_used = info.get('iterations', 0)
                
                # Boundary violations
                violations = 0
                particles_traj = solver.get_particles_trajectory()
                if particles_traj is not None:
                    final_particles = particles_traj[-1, :]
                    violations = np.sum(
                        (final_particles < problem.xmin - 1e-10) | 
                        (final_particles > problem.xmax + 1e-10)
                    )
                
                result = {
                    'name': scenario['name'],
                    'description': scenario['description'],
                    'success': True,
                    'scenario': scenario,
                    'timing': {
                        'solve_time': solve_time,
                        'total_time': total_scenario_time
                    },
                    'convergence': {
                        'converged': converged,
                        'iterations_used': iterations_used,
                        'max_iterations': scenario['qp_settings']['picard']
                    },
                    'mass_conservation': {
                        'initial_mass': initial_mass,
                        'final_mass': final_mass,
                        'mass_change': mass_change,
                        'mass_change_percent': (mass_change / initial_mass) * 100
                    },
                    'physical_observables': {
                        'center_of_mass': center_of_mass,
                        'max_density_location': max_density_location,
                        'final_density_peak': final_density_peak,
                        'density_spread': density_spread
                    },
                    'density_profile': {
                        'at_quarter': density_at_quarter,
                        'at_half': density_at_half,
                        'at_three_quarter': density_at_three_quarter
                    },
                    'solution_quality': {
                        'max_U': max_U,
                        'min_M': min_M,
                        'negative_densities': negative_densities,
                        'violations': violations
                    },
                    'arrays': {
                        'final_density': final_density,
                        'mass_evolution': mass_evolution,
                        'M_solution': M_solution,
                        'U_solution': U_solution
                    },
                    'problem': problem
                }
                
                results.append(result)
                
                # Print key results
                print(f"\n--- RESULTS: {scenario['name']} ---")
                print(f"  Converged: {converged} ({iterations_used}/{scenario['qp_settings']['picard']} iterations)")
                print(f"  Mass: {initial_mass:.6f} → {final_mass:.6f} ({(mass_change/initial_mass)*100:+.3f}%)")
                print(f"  Center of mass: {center_of_mass:.6f}")
                print(f"  Max density: {final_density_peak:.6f} at x = {max_density_location:.6f}")
                print(f"  Density spread: {density_spread:.6f}")
                print(f"  Density profile: [0.25: {density_at_quarter:.3f}, 0.5: {density_at_half:.3f}, 0.75: {density_at_three_quarter:.3f}]")
                print(f"  Quality: {violations} violations, {negative_densities} negative densities")
                
            else:
                print(f"❌ Failed to produce solution")
                results.append({
                    'name': scenario['name'],
                    'success': False,
                    'total_time': time.time() - scenario_start_time
                })
                
        except Exception as e:
            error_time = time.time() - scenario_start_time
            print(f"❌ Crashed after {error_time:.2f}s: {e}")
            results.append({
                'name': scenario['name'],
                'success': False,
                'error': str(e),
                'total_time': error_time
            })
    
    total_time = time.time() - total_start_time
    
    # Comprehensive analysis
    print(f"\n{'='*80}")
    print("FINAL DENSITY VARIATION ANALYSIS")
    print(f"{'='*80}")
    print(f"Total execution time: {total_time:.2f}s")
    
    analyze_density_variations(results)
    create_density_variation_plots(results)
    
    return results

def analyze_density_variations(results):
    """Analyze sources of variation in final densities"""
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) < 2:
        print(f"\nInsufficient successful results for variation analysis: {len(successful_results)}")
        return
    
    print(f"\nSuccessful scenarios: {len(successful_results)}/{len(results)}")
    
    # Reference result (first one)
    reference = successful_results[0]
    print(f"\nUsing '{reference['name']}' as reference for comparison")
    
    # Summary table
    print(f"\n{'Scenario':<20} {'Center of Mass':<15} {'Max Density Loc':<15} {'Peak Value':<12} {'Spread':<10}")
    print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*12} {'-'*10}")
    
    for result in successful_results:
        phys = result['physical_observables']
        print(f"{result['name']:<20} {phys['center_of_mass']:<15.6f} "
              f"{phys['max_density_location']:<15.6f} {phys['final_density_peak']:<12.6f} "
              f"{phys['density_spread']:<10.6f}")
    
    # Detailed variation analysis
    print(f"\n--- DETAILED VARIATION ANALYSIS ---")
    
    # Extract observables
    observables = {
        'Center of Mass': [r['physical_observables']['center_of_mass'] for r in successful_results],
        'Max Density Location': [r['physical_observables']['max_density_location'] for r in successful_results],
        'Peak Density Value': [r['physical_observables']['final_density_peak'] for r in successful_results],
        'Density Spread': [r['physical_observables']['density_spread'] for r in successful_results],
        'Final Mass': [r['mass_conservation']['final_mass'] for r in successful_results]
    }
    
    ref_values = {
        'Center of Mass': reference['physical_observables']['center_of_mass'],
        'Max Density Location': reference['physical_observables']['max_density_location'],
        'Peak Density Value': reference['physical_observables']['final_density_peak'],
        'Density Spread': reference['physical_observables']['density_spread'],
        'Final Mass': reference['mass_conservation']['final_mass']
    }
    
    for obs_name, values in observables.items():
        ref_val = ref_values[obs_name]
        
        print(f"\n{obs_name}:")
        print(f"  Reference value: {ref_val:.8f}")
        
        for i, (result, value) in enumerate(zip(successful_results, values)):
            if i == 0:  # Skip reference
                continue
            
            abs_diff = abs(value - ref_val)
            rel_diff = (abs_diff / abs(ref_val)) * 100 if abs(ref_val) > 1e-10 else 0
            
            print(f"  {result['name']:<20}: {value:.8f} (diff: {abs_diff:+.2e}, {rel_diff:.4f}%)")
        
        # Statistical summary
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = std_val / abs(mean_val) * 100 if abs(mean_val) > 1e-10 else 0
        max_diff = max(values) - min(values)
        
        print(f"  Range: [{min(values):.8f}, {max(values):.8f}]")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Coefficient of variation: {cv:.4f}%")
        
        # Categorize variation level
        if cv < 0.01:
            variation_level = "✅ NEGLIGIBLE"
        elif cv < 0.1:
            variation_level = "✅ VERY LOW"
        elif cv < 1.0:
            variation_level = "✅ LOW"
        elif cv < 5.0:
            variation_level = "⚠️  MODERATE"
        else:
            variation_level = "❌ HIGH"
        
        print(f"  Variation level: {variation_level}")
    
    # Source of variation analysis
    print(f"\n--- SOURCE OF VARIATION ANALYSIS ---")
    
    # Compare specific scenario pairs
    scenario_comparisons = [
        (0, 1, "Random Seed Effect", "Same parameters, different random seed"),
        (0, 2, "Grid Resolution Effect", "Finer spatial/temporal discretization"),
        (0, 3, "Particle Count Effect", "More particles for density estimation"),
        (0, 4, "Collocation Points Effect", "More collocation points"),
        (0, 5, "Convergence Tolerance Effect", "Higher convergence requirements")
    ]
    
    for ref_idx, comp_idx, effect_name, description in scenario_comparisons:
        if comp_idx < len(successful_results):
            ref_result = successful_results[ref_idx]
            comp_result = successful_results[comp_idx]
            
            print(f"\n{effect_name}:")
            print(f"  {description}")
            
            # Compare key observables
            ref_com = ref_result['physical_observables']['center_of_mass']
            comp_com = comp_result['physical_observables']['center_of_mass']
            com_diff = abs(comp_com - ref_com)
            com_rel_diff = (com_diff / abs(ref_com)) * 100 if abs(ref_com) > 1e-10 else 0
            
            ref_peak = ref_result['physical_observables']['final_density_peak']
            comp_peak = comp_result['physical_observables']['final_density_peak']
            peak_diff = abs(comp_peak - ref_peak)
            peak_rel_diff = (peak_diff / abs(ref_peak)) * 100 if abs(ref_peak) > 1e-10 else 0
            
            print(f"  Center of mass change: {com_diff:.2e} ({com_rel_diff:.4f}%)")
            print(f"  Peak density change: {peak_diff:.2e} ({peak_rel_diff:.4f}%)")
            
            # Overall impact assessment
            max_rel_change = max(com_rel_diff, peak_rel_diff)
            if max_rel_change < 0.01:
                impact = "✅ NEGLIGIBLE IMPACT"
            elif max_rel_change < 0.1:
                impact = "✅ MINIMAL IMPACT"
            elif max_rel_change < 1.0:
                impact = "⚠️  SMALL IMPACT"
            elif max_rel_change < 5.0:
                impact = "⚠️  MODERATE IMPACT"
            else:
                impact = "❌ SIGNIFICANT IMPACT"
            
            print(f"  Overall impact: {impact}")
    
    # Final assessment
    print(f"\n--- FINAL ASSESSMENT ---")
    
    # Calculate overall variation across all observables
    all_cvs = []
    for obs_name, values in observables.items():
        if len(values) > 1:
            cv = np.std(values) / abs(np.mean(values)) * 100 if abs(np.mean(values)) > 1e-10 else 0
            all_cvs.append(cv)
    
    overall_cv = np.mean(all_cvs) if all_cvs else 0
    
    print(f"Average coefficient of variation across all observables: {overall_cv:.4f}%")
    
    if overall_cv < 0.1:
        assessment = "✅ EXCELLENT: Final densities are highly consistent across different configurations"
        conclusion = "The 'little bit different' densities are due to numerical precision effects only."
    elif overall_cv < 1.0:
        assessment = "✅ VERY GOOD: Final densities show excellent consistency with minor variation"
        conclusion = "The density differences are primarily due to discretization and convergence tolerances."
    elif overall_cv < 5.0:
        assessment = "✅ GOOD: Final densities are reasonably consistent with some expected variation"
        conclusion = "The density differences reflect normal parameter sensitivity and method limitations."
    else:
        assessment = "⚠️  CONCERNING: Final densities show significant variation between configurations"
        conclusion = "The density differences may indicate convergence issues or parameter sensitivity problems."
    
    print(f"\n{assessment}")
    print(f"\nConclusion: {conclusion}")
    
    # Recommendations
    print(f"\n--- RECOMMENDATIONS ---")
    
    if overall_cv < 1.0:
        print("✅ The observed density differences are within expected numerical variation.")
        print("✅ Different numerical methods should converge to similar solutions.")
        print("• Use consistent convergence tolerances across methods")
        print("• Ensure adequate spatial/temporal resolution")
        print("• Compare methods using the same random seeds when applicable")
    else:
        print("⚠️  The density differences warrant further investigation.")
        print("• Check convergence criteria - methods may not be fully converged")
        print("• Verify boundary condition implementations are consistent")
        print("• Consider parameter sensitivity analysis")
        print("• Compare with analytical solutions if available")

def create_density_variation_plots(results):
    """Create plots showing density variations and their sources"""
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) < 2:
        print("Insufficient results for plotting")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Final Density Variation Analysis: Understanding the "Little Bit Different" Densities', fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(successful_results)))
    
    # 1. Final density overlay
    ax1 = axes[0, 0]
    reference_density = None
    for i, result in enumerate(successful_results):
        problem = result['problem']
        final_density = result['arrays']['final_density']
        
        if i == 0:
            reference_density = final_density
            ax1.plot(problem.xSpace, final_density, 
                    label=f"{result['name']} (Reference)", color=colors[i], linewidth=3, alpha=0.9)
        else:
            ax1.plot(problem.xSpace, final_density, 
                    label=result['name'], color=colors[i], linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Space x')
    ax1.set_ylabel('Final Density M(T,x)')
    ax1.set_title('Final Density Overlay\n(Visual Difference Assessment)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Density differences from reference
    ax2 = axes[0, 1]
    if reference_density is not None:
        for i, result in enumerate(successful_results[1:], 1):  # Skip reference
            problem = result['problem']
            final_density = result['arrays']['final_density']
            
            # Interpolate to common grid if needed
            if len(final_density) == len(reference_density):
                density_diff = final_density - reference_density
            else:
                # Interpolate to reference grid
                ref_problem = successful_results[0]['problem']
                density_interp = np.interp(ref_problem.xSpace, problem.xSpace, final_density)
                density_diff = density_interp - reference_density
            
            ax2.plot(ref_problem.xSpace, density_diff, 
                    label=result['name'], color=colors[i], linewidth=2)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Space x')
        ax2.set_ylabel('Density Difference from Reference')
        ax2.set_title('Density Differences\n(Quantifying Variations)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # 3. Physical observables comparison
    ax3 = axes[0, 2]
    scenario_names = [r['name'] for r in successful_results]
    centers_of_mass = [r['physical_observables']['center_of_mass'] for r in successful_results]
    max_density_locs = [r['physical_observables']['max_density_location'] for r in successful_results]
    
    x_pos = np.arange(len(scenario_names))
    width = 0.35
    
    bars1 = ax3.bar(x_pos - width/2, centers_of_mass, width, 
                    label='Center of Mass', alpha=0.8, color='blue')
    bars2 = ax3.bar(x_pos + width/2, max_density_locs, width, 
                    label='Max Density Location', alpha=0.8, color='red')
    
    ax3.set_xlabel('Scenario')
    ax3.set_ylabel('Position')
    ax3.set_title('Physical Observables\n(Key Convergence Metrics)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, centers_of_mass):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.4f}', ha='center', va='bottom', fontsize=8, rotation=0)
    for bar, value in zip(bars2, max_density_locs):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.4f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    # 4. Variation sources analysis
    ax4 = axes[1, 0]
    
    # Calculate coefficient of variation for different observables
    observables_data = {
        'Center\nof Mass': [r['physical_observables']['center_of_mass'] for r in successful_results],
        'Max Density\nLocation': [r['physical_observables']['max_density_location'] for r in successful_results],
        'Peak\nValue': [r['physical_observables']['final_density_peak'] for r in successful_results],
        'Final\nMass': [r['mass_conservation']['final_mass'] for r in successful_results]
    }
    
    obs_names = list(observables_data.keys())
    cvs = []
    for values in observables_data.values():
        cv = np.std(values) / abs(np.mean(values)) * 100 if abs(np.mean(values)) > 1e-10 else 0
        cvs.append(cv)
    
    bars = ax4.bar(obs_names, cvs, color=['blue', 'red', 'green', 'orange'], alpha=0.7)
    ax4.set_ylabel('Coefficient of Variation (%)')
    ax4.set_title('Observable Variation Levels\n(Lower = More Consistent)')
    ax4.grid(True, axis='y', alpha=0.3)
    
    # Color code bars and add labels
    for bar, value in zip(bars, cvs):
        if value < 0.1:
            color_label = 'Excellent'
            text_color = 'green'
        elif value < 1.0:
            color_label = 'Very Good'
            text_color = 'blue'
        elif value < 5.0:
            color_label = 'Good'
            text_color = 'orange'
        else:
            color_label = 'Poor'
            text_color = 'red'
        
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.3f}%\n({color_label})', ha='center', va='bottom', 
                fontsize=9, color=text_color, weight='bold')
    
    # 5. Mass evolution comparison
    ax5 = axes[1, 1]
    for i, result in enumerate(successful_results):
        problem = result['problem']
        mass_evolution = result['arrays']['mass_evolution']
        ax5.plot(problem.tSpace, mass_evolution, 'o-', 
                label=result['name'], color=colors[i], linewidth=2, alpha=0.8)
    
    ax5.set_xlabel('Time t')
    ax5.set_ylabel('Total Mass')
    ax5.set_title('Mass Evolution Comparison\n(Conservation Consistency)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Execution time vs variation
    ax6 = axes[1, 2]
    
    # Reference values for calculating relative differences
    ref_result = successful_results[0]
    ref_com = ref_result['physical_observables']['center_of_mass']
    ref_peak = ref_result['physical_observables']['final_density_peak']
    
    execution_times = [r['timing']['solve_time'] for r in successful_results]
    
    # Calculate relative differences from reference
    com_rel_diffs = []
    peak_rel_diffs = []
    
    for result in successful_results:
        com = result['physical_observables']['center_of_mass']
        peak = result['physical_observables']['final_density_peak']
        
        com_rel_diff = abs(com - ref_com) / abs(ref_com) * 100 if abs(ref_com) > 1e-10 else 0
        peak_rel_diff = abs(peak - ref_peak) / abs(ref_peak) * 100 if abs(ref_peak) > 1e-10 else 0
        
        com_rel_diffs.append(com_rel_diff)
        peak_rel_diffs.append(peak_rel_diff)
    
    # Scatter plot
    scatter1 = ax6.scatter(execution_times, com_rel_diffs, 
                          c='blue', alpha=0.7, s=100, label='Center of Mass')
    scatter2 = ax6.scatter(execution_times, peak_rel_diffs, 
                          c='red', alpha=0.7, s=100, label='Peak Density')
    
    # Annotate points
    for i, result in enumerate(successful_results):
        if i > 0:  # Skip reference point
            ax6.annotate(result['name'], (execution_times[i], max(com_rel_diffs[i], peak_rel_diffs[i])), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax6.set_xlabel('Execution Time (seconds)')
    ax6.set_ylabel('Relative Difference from Reference (%)')
    ax6.set_title('Execution Time vs Accuracy\n(Computational Trade-off)')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/final_density_convergence_analysis.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Starting final density convergence analysis...")
    print("This analysis will help understand why final densities are 'always a little bit different'")
    print("Expected execution time: 2-8 minutes depending on number of scenarios")
    
    try:
        results = analyze_final_density_convergence()
        print("\n" + "="*80)
        print("FINAL DENSITY CONVERGENCE ANALYSIS COMPLETED")
        print("="*80)
        print("Check the analysis above and generated plots for insights into density variations.")
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nAnalysis failed with error: {e}")
        import traceback
        traceback.print_exc()
