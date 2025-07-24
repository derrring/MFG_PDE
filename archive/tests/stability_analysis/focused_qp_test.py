#!/usr/bin/env python3
"""
Focused test of QP Particle-Collocation method to address user feedback.
This test specifically examines:
1. Mass conservation behavior with no-flux boundary conditions
2. Convergence properties and final mass consistency
3. Comparison with expected theoretical behavior
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

def test_qp_mass_conservation():
    """
    Test QP method behavior focusing on mass conservation properties.
    Addresses user feedback about expected mass increase with no-flux BC.
    """
    print("="*80)
    print("FOCUSED QP PARTICLE-COLLOCATION MASS CONSERVATION TEST")
    print("="*80)
    print("Testing mass conservation behavior with no-flux boundary conditions")
    print("Expected: slight mass increase due to particle reflection")
    
    # Test with multiple parameter sets
    test_cases = [
        {
            'name': 'Ultra Conservative',
            'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 12, 'T': 0.05, 'Nt': 3, 'sigma': 0.05, 'coefCT': 0.001},
            'qp_settings': {'particles': 50, 'collocation': 4, 'delta': 0.15, 'order': 1, 'newton': 2, 'picard': 2}
        },
        {
            'name': 'Conservative',
            'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 15, 'T': 0.1, 'Nt': 5, 'sigma': 0.1, 'coefCT': 0.005},
            'qp_settings': {'particles': 100, 'collocation': 5, 'delta': 0.2, 'order': 1, 'newton': 3, 'picard': 3}
        },
        {
            'name': 'Moderate',
            'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 20, 'T': 0.15, 'Nt': 8, 'sigma': 0.15, 'coefCT': 0.01},
            'qp_settings': {'particles': 200, 'collocation': 6, 'delta': 0.25, 'order': 2, 'newton': 4, 'picard': 4}
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i+1}: {test_case['name'].upper()}")
        print(f"{'='*60}")
        
        # Display parameters
        print("Problem parameters:")
        for key, value in test_case['params'].items():
            print(f"  {key}: {value}")
        
        print("QP settings:")
        for key, value in test_case['qp_settings'].items():
            print(f"  {key}: {value}")
        
        try:
            # Create problem
            problem = ExampleMFGProblem(**test_case['params'])
            no_flux_bc = BoundaryConditions(type="no_flux")
            
            # Setup collocation
            num_collocation = test_case['qp_settings']['collocation']
            collocation_points = np.linspace(
                problem.xmin, problem.xmax, num_collocation
            ).reshape(-1, 1)
            
            boundary_indices = [0, num_collocation - 1]
            
            # Create solver
            start_time = time.time()
            
            solver = ParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                num_particles=test_case['qp_settings']['particles'],
                delta=test_case['qp_settings']['delta'],
                taylor_order=test_case['qp_settings']['order'],
                weight_function="wendland",
                NiterNewton=test_case['qp_settings']['newton'],
                l2errBoundNewton=1e-2,
                kde_bandwidth="scott",
                normalize_kde_output=False,
                boundary_indices=np.array(boundary_indices),
                boundary_conditions=no_flux_bc,
                use_monotone_constraints=True
            )
            
            # Solve
            U_solution, M_solution, info = solver.solve(
                Niter=test_case['qp_settings']['picard'], 
                l2errBound=1e-2, 
                verbose=False
            )
            
            elapsed_time = time.time() - start_time
            
            if M_solution is not None and U_solution is not None:
                # Analyze mass conservation
                mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
                initial_mass = mass_evolution[0]
                final_mass = mass_evolution[-1]
                mass_change = final_mass - initial_mass
                mass_change_percent = (mass_change / initial_mass) * 100
                
                # Check particle boundary violations
                violations = 0
                particles_traj = solver.get_particles_trajectory()
                if particles_traj is not None:
                    final_particles = particles_traj[-1, :]
                    violations = np.sum(
                        (final_particles < problem.xmin) | (final_particles > problem.xmax)
                    )
                
                # Assess mass conservation quality
                if mass_change >= 0:
                    mass_behavior = "INCREASE" if mass_change > 1e-6 else "CONSERVED"
                    mass_status = "✅ EXPECTED"
                else:
                    mass_behavior = "DECREASE"
                    mass_status = "⚠️  UNEXPECTED"
                
                result = {
                    'name': test_case['name'],
                    'success': True,
                    'runtime': elapsed_time,
                    'converged': info.get('converged', False),
                    'iterations': info.get('iterations', 0),
                    'initial_mass': initial_mass,
                    'final_mass': final_mass,
                    'mass_change': mass_change,
                    'mass_change_percent': mass_change_percent,
                    'mass_behavior': mass_behavior,
                    'mass_status': mass_status,
                    'violations': violations,
                    'mass_evolution': mass_evolution,
                    'M_solution': M_solution,
                    'U_solution': U_solution,
                    'problem': problem
                }
                
                results.append(result)
                
                print(f"\n✅ {test_case['name']} completed successfully")
                print(f"  Runtime: {elapsed_time:.2f}s")
                print(f"  Converged: {info.get('converged', False)} in {info.get('iterations', 0)} iterations")
                print(f"  Initial mass: {initial_mass:.6f}")
                print(f"  Final mass: {final_mass:.6f}")
                print(f"  Mass change: {mass_change:+.2e} ({mass_change_percent:+.3f}%)")
                print(f"  Mass behavior: {mass_behavior} - {mass_status}")
                print(f"  Boundary violations: {violations}")
                
            else:
                print(f"❌ {test_case['name']} failed to produce solution")
                results.append({'name': test_case['name'], 'success': False})
                
        except Exception as e:
            print(f"❌ {test_case['name']} crashed: {e}")
            results.append({'name': test_case['name'], 'success': False, 'error': str(e)})
    
    # Analysis of results
    print(f"\n{'='*80}")
    print("MASS CONSERVATION ANALYSIS")
    print(f"{'='*80}")
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) > 0:
        print(f"\n{'Test Case':<20} {'Mass Change':<15} {'Behavior':<12} {'Status':<15} {'Violations':<12}")
        print(f"{'-'*20} {'-'*15} {'-'*12} {'-'*15} {'-'*12}")
        
        for result in successful_results:
            print(f"{result['name']:<20} {result['mass_change']:+.2e} {result['mass_behavior']:<12} {result['mass_status']:<15} {result['violations']:<12}")
        
        # Summary assessment
        print(f"\n--- Summary Assessment ---")
        
        mass_increasing = [r for r in successful_results if r['mass_change'] > 0]
        mass_conserving = [r for r in successful_results if abs(r['mass_change_percent']) < 1.0]
        mass_decreasing = [r for r in successful_results if r['mass_change'] < -1e-6]
        
        print(f"Tests showing mass increase: {len(mass_increasing)}/{len(successful_results)}")
        print(f"Tests with good conservation (< 1% change): {len(mass_conserving)}/{len(successful_results)}")
        print(f"Tests showing mass decrease: {len(mass_decreasing)}/{len(successful_results)}")
        
        if len(mass_increasing) > 0:
            print("✅ POSITIVE: Some tests show expected mass increase from no-flux reflection")
        elif len(mass_conserving) == len(successful_results):
            print("✅ ACCEPTABLE: All tests show good mass conservation")
        else:
            print("⚠️  CONCERN: Mass behavior not as expected")
        
        # Convergence consistency check
        if len(successful_results) >= 2:
            final_masses = [r['final_mass'] for r in successful_results]
            # Scale by problem size for fair comparison
            normalized_masses = [fm / (r['problem'].xmax - r['problem'].xmin) for fm, r in zip(final_masses, successful_results)]
            
            mass_std = np.std(normalized_masses)
            mass_mean = np.mean(normalized_masses)
            cv = mass_std / mass_mean if mass_mean > 0 else 0
            
            print(f"\n--- Convergence Consistency ---")
            print(f"Coefficient of variation in final masses: {cv:.3f}")
            if cv < 0.1:
                print("✅ EXCELLENT: Very consistent final masses across parameter sets")
            elif cv < 0.3:
                print("✅ GOOD: Reasonably consistent final masses")
            else:
                print("⚠️  WARNING: Inconsistent final masses suggest numerical issues")
        
        # Create visualization
        create_mass_conservation_plots(successful_results)
        
    else:
        print("❌ No test cases completed successfully")
        for result in results:
            if not result.get('success', False):
                error = result.get('error', 'Failed')
                print(f"  {result['name']}: {error}")

def create_mass_conservation_plots(results):
    """Create plots focused on mass conservation analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('QP Method Mass Conservation Analysis', fontsize=16)
    
    colors = ['blue', 'green', 'red', 'orange']
    
    # Mass evolution over time for each test case
    ax1 = axes[0, 0]
    for i, result in enumerate(results):
        if 'mass_evolution' in result:
            problem = result['problem']
            ax1.plot(problem.tSpace, result['mass_evolution'], 
                    'o-', label=result['name'], color=colors[i % len(colors)], linewidth=2)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Total Mass')
    ax1.set_title('Mass Evolution Over Time')
    ax1.grid(True)
    ax1.legend()
    
    # Mass change comparison
    ax2 = axes[0, 1]
    names = [r['name'] for r in results]
    changes = [r['mass_change'] for r in results]
    colors_bars = [colors[i % len(colors)] for i in range(len(results))]
    bars = ax2.bar(names, changes, color=colors_bars)
    ax2.set_ylabel('Mass Change')
    ax2.set_title('Mass Change by Test Case')
    ax2.grid(True, axis='y')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    for bar, value in zip(bars, changes):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:+.1e}', ha='center', va='bottom' if value >= 0 else 'top')
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    # Final density comparison
    ax3 = axes[1, 0]
    for i, result in enumerate(results):
        if 'M_solution' in result:
            problem = result['problem']
            final_density = result['M_solution'][-1, :]
            ax3.plot(problem.xSpace, final_density, 
                    label=result['name'], color=colors[i % len(colors)], linewidth=2)
    ax3.set_xlabel('Space x')
    ax3.set_ylabel('Final Density M(T,x)')
    ax3.set_title('Final Density Distributions')
    ax3.grid(True)
    ax3.legend()
    
    # Runtime vs mass conservation quality
    ax4 = axes[1, 1]
    runtimes = [r['runtime'] for r in results]
    mass_changes_abs = [abs(r['mass_change_percent']) for r in results]
    scatter = ax4.scatter(runtimes, mass_changes_abs, c=colors_bars, s=100, alpha=0.7)
    ax4.set_xlabel('Runtime (seconds)')
    ax4.set_ylabel('|Mass Change %|')
    ax4.set_title('Runtime vs Mass Conservation Quality')
    ax4.grid(True)
    for i, result in enumerate(results):
        ax4.annotate(result['name'], (runtimes[i], mass_changes_abs[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/mass_conservation_analysis.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    test_qp_mass_conservation()
