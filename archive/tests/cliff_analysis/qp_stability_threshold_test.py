#!/usr/bin/env python3
"""
QP-Collocation Stability Threshold Test
Systematic study of where the "cliff" behavior begins using proven stable parameters.
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.core.boundaries import BoundaryConditions


def run_stability_threshold_test():
    """Test systematic progression to find stability threshold"""
    print("=" * 80)
    print("QP-COLLOCATION STABILITY THRESHOLD TEST")
    print("=" * 80)
    print("Systematic study to identify when 'cliff' behavior begins")

    # Proven stable parameters from T=2 success
    base_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 50,        # Proven stable
        "sigma": 0.2,    # Proven stable  
        "coefCT": 0.03,  # Proven stable
    }

    stable_solver_params = {
        "num_particles": 800,
        "delta": 0.3,
        "taylor_order": 2,
        "weight_function": "wendland",
        "NiterNewton": 8,
        "l2errBoundNewton": 1e-4,
        "kde_bandwidth": "scott",
        "normalize_kde_output": False,
        "use_monotone_constraints": True
    }

    # Systematic test progression - find where cliff begins
    test_sequence = [
        {"T": 2.5, "Nt": 125, "name": "T=2.5"},
        {"T": 3.0, "Nt": 150, "name": "T=3.0"},
        {"T": 3.5, "Nt": 175, "name": "T=3.5"},
        {"T": 4.0, "Nt": 200, "name": "T=4.0"},
    ]

    results = {}
    no_flux_bc = BoundaryConditions(type="no_flux")
    num_collocation_points = 12

    for test_config in test_sequence:
        T = test_config["T"]
        Nt = test_config["Nt"]
        name = test_config["name"]
        
        print(f"\n{'='*50}")
        print(f"TESTING: {name}")
        print(f"{'='*50}")

        problem_params = base_params.copy()
        problem_params.update({"T": T, "Nt": Nt})

        try:
            # Create problem
            problem = ExampleMFGProblem(**problem_params)
            
            print(f"Domain: [0, 1] × [0, {T}], Dt = {problem.Dt:.4f}")
            print(f"CFL = {problem_params['sigma']**2 * problem.Dt / problem.Dx**2:.3f}")

            # Setup solver
            collocation_points = np.linspace(
                problem.xmin, problem.xmax, num_collocation_points
            ).reshape(-1, 1)
            boundary_indices = [0, num_collocation_points - 1]

            solver = ParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                boundary_indices=np.array(boundary_indices),
                boundary_conditions=no_flux_bc,
                **stable_solver_params,
            )

            print(f"Starting {name} simulation...")
            start_time = time.time()

            U_solution, M_solution, solve_info = solver.solve(
                Niter=12,  # Reduced for speed
                l2errBound=1e-3,
                verbose=False
            )

            total_time = time.time() - start_time
            
            if U_solution is not None and M_solution is not None:
                # Detailed mass analysis
                mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
                initial_mass = mass_evolution[0]
                final_mass = mass_evolution[-1]
                
                # Cliff detection - look for sudden drops
                mass_diff = np.diff(mass_evolution)
                mass_diff_percent = (mass_diff / initial_mass) * 100
                
                # Multiple cliff criteria
                cliff_20 = np.any(mass_diff_percent < -20)  # 20% single-step loss
                cliff_10 = np.any(mass_diff_percent < -10)  # 10% single-step loss
                cliff_5 = np.any(mass_diff_percent < -5)    # 5% single-step loss
                
                max_single_loss = np.min(mass_diff_percent)
                cliff_step = np.argmin(mass_diff_percent) if max_single_loss < -1 else -1
                
                # Total mass change
                total_change_percent = (final_mass - initial_mass) / initial_mass * 100
                
                # Check if mass goes to near zero
                mass_collapse = final_mass < 0.01 * initial_mass
                
                # Particle violations
                particles_trajectory = solver.fp_solver.M_particles_trajectory
                total_violations = 0
                if particles_trajectory is not None:
                    for t_step in range(particles_trajectory.shape[0]):
                        step_particles = particles_trajectory[t_step, :]
                        violations = np.sum(
                            (step_particles < problem.xmin - 1e-10) |
                            (step_particles > problem.xmax + 1e-10)
                        )
                        total_violations += violations

                # Store comprehensive results
                results[name] = {
                    'success': True,
                    'T': T,
                    'time': total_time,
                    'mass_initial': initial_mass,
                    'mass_final': final_mass,
                    'total_change_percent': total_change_percent,
                    'max_single_loss_percent': max_single_loss,
                    'cliff_step': cliff_step,
                    'cliff_time': problem.tSpace[cliff_step] if cliff_step >= 0 else -1,
                    'cliff_5': cliff_5,
                    'cliff_10': cliff_10,
                    'cliff_20': cliff_20,
                    'mass_collapse': mass_collapse,
                    'total_violations': total_violations,
                    'arrays': {
                        'mass_evolution': mass_evolution,
                        'tSpace': problem.tSpace,
                        'M_solution': M_solution
                    }
                }

                # Report results
                print(f"✓ {name} completed in {total_time:.1f}s")
                print(f"  Mass: {initial_mass:.6f} → {final_mass:.6f} ({total_change_percent:+.2f}%)")
                print(f"  Max single loss: {max_single_loss:.2f}%")
                print(f"  Violations: {total_violations}")
                
                if mass_collapse:
                    print(f"  ❌ MASS COLLAPSE: Final mass < 1% of initial")
                elif cliff_20:
                    print(f"  ⚠️  SEVERE CLIFF: >20% single-step loss at t={problem.tSpace[cliff_step]:.3f}")
                elif cliff_10:
                    print(f"  ⚠️  MODERATE CLIFF: >10% single-step loss at t={problem.tSpace[cliff_step]:.3f}")
                elif cliff_5:
                    print(f"  ⚠️  MILD CLIFF: >5% single-step loss at t={problem.tSpace[cliff_step]:.3f}")
                else:
                    print(f"  ✅ STABLE: No significant cliff behavior detected")

            else:
                results[name] = {'success': False, 'T': T}
                print(f"❌ {name} failed to produce solution")

        except Exception as e:
            results[name] = {'success': False, 'T': T, 'error': str(e)}
            print(f"❌ {name} crashed: {e}")

    # Analysis
    print(f"\n{'='*80}")
    print("STABILITY THRESHOLD ANALYSIS")
    print(f"{'='*80}")
    
    analyze_stability_threshold(results)
    create_stability_threshold_plots(results)
    
    return results


def analyze_stability_threshold(results):
    """Analyze where the stability threshold occurs"""
    successful = [(name, result) for name, result in results.items() if result.get('success', False)]
    
    if not successful:
        print("No successful simulations to analyze")
        return

    print(f"{'Simulation':<10} {'Mass Change':<12} {'Max Loss':<10} {'Cliff?':<15} {'Collapse?':<10}")
    print(f"{'-'*10} {'-'*12} {'-'*10} {'-'*15} {'-'*10}")
    
    stable_threshold = None
    for name, result in successful:
        mass_change = result['total_change_percent']
        max_loss = result['max_single_loss_percent']
        
        cliff_status = "None"
        if result['cliff_20']:
            cliff_status = "Severe (>20%)"
        elif result['cliff_10']:
            cliff_status = "Moderate (>10%)"
        elif result['cliff_5']:
            cliff_status = "Mild (>5%)"
            
        collapse = "YES" if result['mass_collapse'] else "NO"
        
        print(f"{name:<10} {mass_change:<+12.2f} {max_loss:<10.2f} {cliff_status:<15} {collapse:<10}")
        
        # Identify stability threshold
        if not result['cliff_5'] and stable_threshold is None:
            stable_threshold = result['T']

    print(f"\n--- THRESHOLD ANALYSIS ---")
    if stable_threshold:
        print(f"Last stable simulation: T = {stable_threshold}")
        
        # Find first unstable
        first_unstable = None
        for name, result in successful:
            if result['cliff_5']:
                first_unstable = result['T']
                break
        
        if first_unstable:
            print(f"First unstable simulation: T = {first_unstable}")
            print(f"Stability threshold: Between T = {stable_threshold} and T = {first_unstable}")
        else:
            print(f"All simulations up to T = {stable_threshold} are stable")
    else:
        print("No stable simulations found in test range")

    # Cliff timing analysis
    print(f"\n--- CLIFF TIMING ANALYSIS ---")
    for name, result in successful:
        if result['cliff_step'] >= 0:
            cliff_time = result['cliff_time']
            cliff_fraction = cliff_time / result['T']
            print(f"{name}: Cliff at t = {cliff_time:.3f} ({cliff_fraction:.1%} through simulation)")


def create_stability_threshold_plots(results):
    """Create plots showing stability threshold behavior"""
    successful = [(name, result) for name, result in results.items() if result.get('success', False)]
    
    if len(successful) < 2:
        print("Insufficient data for plotting")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('QP-Collocation Stability Threshold Analysis', fontsize=16, fontweight='bold')

    # 1. Mass evolution for all simulations
    ax1 = axes[0, 0]
    colors = ['green', 'blue', 'orange', 'red', 'purple']
    
    for i, (name, result) in enumerate(successful):
        arrays = result['arrays']
        color = colors[i % len(colors)]
        
        # Line style based on stability
        if result['mass_collapse']:
            linestyle = ':'
            linewidth = 3
            alpha = 0.9
        elif result['cliff_20']:
            linestyle = '--'
            linewidth = 2.5
            alpha = 0.8
        elif result['cliff_10']:
            linestyle = '-.'
            linewidth = 2
            alpha = 0.7
        else:
            linestyle = '-'
            linewidth = 2
            alpha = 0.8
            
        ax1.plot(arrays['tSpace'], arrays['mass_evolution'], 
                color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha,
                label=f"{name} ({result['total_change_percent']:+.1f}%)")
    
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Total Mass')
    ax1.set_title('Mass Evolution: Stability Progression')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Maximum single-step loss vs T
    ax2 = axes[0, 1]
    T_vals = [result['T'] for name, result in successful]
    max_losses = [result['max_single_loss_percent'] for name, result in successful]
    
    # Color code by severity
    colors_scatter = []
    for name, result in successful:
        if result['mass_collapse']:
            colors_scatter.append('red')
        elif result['cliff_20']:
            colors_scatter.append('orange')
        elif result['cliff_10']:
            colors_scatter.append('yellow')
        elif result['cliff_5']:
            colors_scatter.append('lightblue')
        else:
            colors_scatter.append('green')
    
    scatter = ax2.scatter(T_vals, max_losses, c=colors_scatter, s=100, alpha=0.7, edgecolors='black')
    
    # Add threshold lines
    ax2.axhline(y=-5, color='lightblue', linestyle='--', alpha=0.7, label='5% cliff threshold')
    ax2.axhline(y=-10, color='orange', linestyle='--', alpha=0.7, label='10% cliff threshold')
    ax2.axhline(y=-20, color='red', linestyle='--', alpha=0.7, label='20% cliff threshold')
    
    ax2.set_xlabel('Time Horizon T')
    ax2.set_ylabel('Max Single-Step Loss (%)')
    ax2.set_title('Cliff Severity vs Time Horizon')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Final mass vs T
    ax3 = axes[1, 0]
    final_masses = [result['mass_final'] for name, result in successful]
    
    ax3.scatter(T_vals, final_masses, c=colors_scatter, s=100, alpha=0.7, edgecolors='black')
    
    for i, (T, mass, (name, result)) in enumerate(zip(T_vals, final_masses, successful)):
        ax3.text(T, mass + 0.02, name, ha='center', fontsize=9)
    
    ax3.set_xlabel('Time Horizon T')
    ax3.set_ylabel('Final Mass')
    ax3.set_title('Final Mass vs Time Horizon')
    ax3.grid(True, alpha=0.3)

    # 4. Stability summary
    ax4 = axes[1, 1]
    
    # Count stability categories
    stable_count = sum(1 for name, result in successful if not result['cliff_5'])
    mild_cliff_count = sum(1 for name, result in successful if result['cliff_5'] and not result['cliff_10'])
    moderate_cliff_count = sum(1 for name, result in successful if result['cliff_10'] and not result['cliff_20'])
    severe_cliff_count = sum(1 for name, result in successful if result['cliff_20'] and not result['mass_collapse'])
    collapse_count = sum(1 for name, result in successful if result['mass_collapse'])
    
    categories = ['Stable', 'Mild Cliff\n(5-10%)', 'Moderate Cliff\n(10-20%)', 'Severe Cliff\n(>20%)', 'Mass Collapse']
    counts = [stable_count, mild_cliff_count, moderate_cliff_count, severe_cliff_count, collapse_count]
    colors_bar = ['green', 'lightblue', 'orange', 'red', 'darkred']
    
    bars = ax4.bar(categories, counts, color=colors_bar, alpha=0.7, edgecolor='black')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        if count > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    ax4.set_ylabel('Number of Simulations')
    ax4.set_title('Stability Classification Summary')
    ax4.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/qp_stability_threshold.png', 
                dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Starting QP-Collocation Stability Threshold Test...")
    print("Testing T=2.5, 3.0, 3.5, 4.0 to find stability limits")
    print("Expected execution time: 8-15 minutes")

    try:
        results = run_stability_threshold_test()
        print("\n" + "=" * 80)
        print("STABILITY THRESHOLD TEST COMPLETED")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()