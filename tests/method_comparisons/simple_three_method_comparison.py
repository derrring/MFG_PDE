#!/usr/bin/env python3
"""
Simple comparison focusing on QP-Collocation vs existing working examples.
Addresses user feedback on mass conservation and method convergence.
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

def compare_qp_method_analysis():
    print("="*80)
    print("QP PARTICLE-COLLOCATION METHOD ANALYSIS")
    print("="*80)
    print("Addressing user feedback on mass conservation and convergence issues")
    
    # Conservative test parameters
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 25,   
        "T": 0.2,   
        "Nt": 10,   
        "sigma": 0.2,  
        "coefCT": 0.01  
    }
    
    print(f"\nProblem Parameters:")
    for key, value in problem_params.items():
        print(f"  {key}: {value}")
    
    problem = ExampleMFGProblem(**problem_params)
    no_flux_bc = BoundaryConditions(type="no_flux")
    
    results = {}
    
    # Test multiple configurations of QP-Collocation
    configs = [
        {
            'name': 'Conservative QP',
            'num_particles': 300,
            'num_collocation': 8,
            'delta': 0.25,
            'taylor_order': 1,
            'newton_iters': 4,
            'picard_iters': 6
        },
        {
            'name': 'Standard QP',
            'num_particles': 500,
            'num_collocation': 10,
            'delta': 0.3,
            'taylor_order': 2,
            'newton_iters': 6,
            'picard_iters': 8
        },
        {
            'name': 'High-Quality QP',
            'num_particles': 800,
            'num_collocation': 12,
            'delta': 0.35,
            'taylor_order': 2,
            'newton_iters': 8,
            'picard_iters': 10
        }
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"TESTING: {config['name'].upper()}")
        print(f"{'='*60}")
        print(f"Particles: {config['num_particles']}, Collocation points: {config['num_collocation']}")
        print(f"Delta: {config['delta']}, Taylor order: {config['taylor_order']}")
        
        try:
            start_time = time.time()
            
            # Collocation setup
            collocation_points = np.linspace(
                problem.xmin, problem.xmax, config['num_collocation']
            ).reshape(-1, 1)
            
            boundary_indices = []
            for i, point in enumerate(collocation_points):
                x = point[0]
                if abs(x - problem.xmin) < 1e-10 or abs(x - problem.xmax) < 1e-10:
                    boundary_indices.append(i)
            boundary_indices = np.array(boundary_indices)
            
            collocation_solver = ParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                num_particles=config['num_particles'],
                delta=config['delta'],
                taylor_order=config['taylor_order'],
                weight_function="wendland",
                NiterNewton=config['newton_iters'],
                l2errBoundNewton=1e-3,
                kde_bandwidth="scott",
                normalize_kde_output=False,
                boundary_indices=boundary_indices,
                boundary_conditions=no_flux_bc,
                use_monotone_constraints=True
            )
            
            U_solution, M_solution, info = collocation_solver.solve(
                Niter=config['picard_iters'], l2errBound=1e-3, verbose=False
            )
            
            elapsed_time = time.time() - start_time
            
            if M_solution is not None and U_solution is not None:
                mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
                initial_mass = mass_evolution[0]
                final_mass = mass_evolution[-1]
                mass_change = abs(final_mass - initial_mass)
                mass_loss_percent = (mass_change / initial_mass) * 100
                
                # Count boundary violations
                violations = 0
                particles_trajectory = collocation_solver.get_particles_trajectory()
                if particles_trajectory is not None:
                    final_particles = particles_trajectory[-1, :]
                    violations = np.sum((final_particles < problem.xmin) | (final_particles > problem.xmax))
                
                # Mass conservation quality assessment
                if mass_loss_percent < 0.5:
                    mass_quality = "EXCELLENT"
                elif mass_loss_percent < 2.0:
                    mass_quality = "GOOD"
                elif mass_loss_percent < 5.0:
                    mass_quality = "ACCEPTABLE"
                else:
                    mass_quality = "POOR"
                
                results[config['name']] = {
                    'success': True,
                    'initial_mass': initial_mass,
                    'final_mass': final_mass,
                    'mass_change': mass_change,
                    'mass_loss_percent': mass_loss_percent,
                    'mass_quality': mass_quality,
                    'max_U': np.max(np.abs(U_solution)),
                    'time': elapsed_time,
                    'converged': info.get('converged', False),
                    'iterations': info.get('iterations', 0),
                    'violations': violations,
                    'U_solution': U_solution,
                    'M_solution': M_solution,
                    'mass_evolution': mass_evolution
                }
                
                print(f"✓ {config['name']} completed successfully")
                print(f"  Initial mass: {initial_mass:.6f}")
                print(f"  Final mass: {final_mass:.6f}")
                print(f"  Mass change: {mass_change:.2e} ({mass_loss_percent:.3f}%) - {mass_quality}")
                print(f"  Runtime: {elapsed_time:.2f}s, Violations: {violations}")
                print(f"  Converged: {info.get('converged', False)}, Iterations: {info.get('iterations', 0)}")
            else:
                results[config['name']] = {'success': False}
                print(f"❌ {config['name']} failed")
        except Exception as e:
            results[config['name']] = {'success': False, 'error': str(e)}
            print(f"❌ {config['name']} crashed: {e}")
    
    # Analysis and comparison
    print(f"\n{'='*80}")
    print("ANALYSIS RESULTS")
    print(f"{'='*80}")
    
    successful_configs = [name for name in results if results[name].get('success', False)]
    
    if len(successful_configs) > 0:
        print(f"\n{'Configuration':<20} {'Mass Loss %':<12} {'Quality':<12} {'Runtime (s)':<12} {'Violations':<12}")
        print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        
        for config_name in successful_configs:
            result = results[config_name]
            print(f"{config_name:<20} {result['mass_loss_percent']:<12.3f} {result['mass_quality']:<12} {result['time']:<12.2f} {result['violations']:<12}")
        
        # Find best configuration
        best_mass_conservation = min(successful_configs, key=lambda x: results[x]['mass_loss_percent'])
        fastest = min(successful_configs, key=lambda x: results[x]['time'])
        
        print(f"\n--- Performance Winners ---")
        print(f"🏆 Best mass conservation: {best_mass_conservation} ({results[best_mass_conservation]['mass_loss_percent']:.3f}% loss)")
        print(f"🏆 Fastest execution: {fastest} ({results[fastest]['time']:.2f}s)")
        
        # Convergence analysis
        print(f"\n--- Convergence Analysis ---")
        if len(successful_configs) >= 2:
            final_masses = [results[config]['final_mass'] for config in successful_configs]
            max_diff = max(final_masses) - min(final_masses)
            avg_mass = np.mean(final_masses)
            relative_diff = (max_diff / avg_mass) * 100 if avg_mass > 0 else 0
            
            print(f"Final mass range: [{min(final_masses):.6f}, {max(final_masses):.6f}]")
            print(f"Max difference: {max_diff:.2e}")
            print(f"Relative difference: {relative_diff:.3f}%")
            
            if relative_diff < 1.0:
                print("✅ EXCELLENT: All configurations converge consistently")
            elif relative_diff < 5.0:
                print("✅ GOOD: Configurations show reasonable convergence")
            else:
                print("⚠️  WARNING: Configurations show significant divergence")
        
        # User feedback assessment
        print(f"\n--- Addressing User Feedback ---")
        
        # Check for proper mass conservation (user noted FDM should increase mass slightly)
        mass_increasing_configs = [name for name in successful_configs 
                                 if results[name]['final_mass'] > results[name]['initial_mass']]
        
        if mass_increasing_configs:
            print("✅ POSITIVE: Some configurations show mass increase (expected with no-flux BC reflection)")
            for config in mass_increasing_configs:
                increase = results[config]['final_mass'] - results[config]['initial_mass']
                print(f"  {config}: +{increase:.2e} mass increase")
        else:
            print("⚠️  WARNING: No configurations show expected mass increase from reflection")
        
        # Check convergence consistency
        mass_changes = [results[config]['mass_change'] for config in successful_configs]
        if len(mass_changes) > 1:
            mass_std = np.std(mass_changes)
            mass_mean = np.mean(mass_changes)
            cv = mass_std / mass_mean if mass_mean > 0 else 0
            
            if cv < 0.2:
                print("✅ GOOD: Consistent mass conservation across configurations")
            else:
                print("⚠️  WARNING: Inconsistent mass conservation suggests implementation issues")
        
        # Create visualization
        create_qp_analysis_plots(results, problem, successful_configs)
    
    else:
        print("No configurations completed successfully")
        for config_name in results:
            if not results[config_name].get('success', False):
                error = results[config_name].get('error', 'Failed')
                print(f"❌ {config_name}: {error}")

def create_qp_analysis_plots(results, problem, successful_configs):
    """Create analysis plots for QP method configurations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('QP Particle-Collocation Method Analysis', fontsize=16)
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    # Mass evolution over time
    ax1 = axes[0, 0]
    for i, config_name in enumerate(successful_configs):
        result = results[config_name]
        mass_evolution = result['mass_evolution']
        ax1.plot(problem.tSpace, mass_evolution, 
                label=config_name, linewidth=2, color=colors[i % len(colors)])
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Total Mass')
    ax1.set_title('Mass Conservation Over Time')
    ax1.grid(True)
    ax1.legend()
    
    # Final density comparison
    ax2 = axes[0, 1]
    for i, config_name in enumerate(successful_configs):
        result = results[config_name]
        M_solution = result['M_solution']
        final_density = M_solution[-1, :]
        ax2.plot(problem.xSpace, final_density, 
                label=config_name, linewidth=2, color=colors[i % len(colors)])
    ax2.set_xlabel('Space x')
    ax2.set_ylabel('Final Density M(T,x)')
    ax2.set_title('Final Density Comparison')
    ax2.grid(True)
    ax2.legend()
    
    # Mass loss percentage comparison
    ax3 = axes[1, 0]
    config_names = list(successful_configs)
    mass_losses = [results[config]['mass_loss_percent'] for config in config_names]
    bars = ax3.bar(config_names, mass_losses, color=colors[:len(config_names)])
    ax3.set_ylabel('Mass Loss (%)')
    ax3.set_title('Mass Conservation Quality')
    ax3.grid(True, axis='y')
    for bar, value in zip(bars, mass_losses):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.2f}%', ha='center', va='bottom')
    plt.setp(ax3.get_xticklabels(), rotation=45)
    
    # Runtime comparison
    ax4 = axes[1, 1]
    runtimes = [results[config]['time'] for config in config_names]
    bars = ax4.bar(config_names, runtimes, color=colors[:len(config_names)])
    ax4.set_ylabel('Runtime (seconds)')
    ax4.set_title('Performance Comparison')
    ax4.grid(True, axis='y')
    for bar, value in zip(bars, runtimes):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.2f}s', ha='center', va='bottom')
    plt.setp(ax4.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/qp_analysis.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    compare_qp_method_analysis()
