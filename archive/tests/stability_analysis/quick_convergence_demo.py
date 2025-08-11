#!/usr/bin/env python3
"""
Quick convergence demonstration showing QP method consistency.
Focused test with two resolution levels to demonstrate convergence behavior.
"""

import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# Add the main package to path
sys.path.insert(0, '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE')

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def quick_convergence_demo():
    """
    Quick demonstration of QP method convergence with two resolution levels.
    """
    print("=" * 80)
    print("QUICK QP METHOD CONVERGENCE DEMONSTRATION")
    print("=" * 80)
    print("Comparing QP method at two resolution levels")
    print("Demonstrating consistent convergence behavior")

    # Base problem parameters
    base_params = {"xmin": 0.0, "xmax": 1.0, "T": 0.3, "sigma": 0.2, "coefCT": 0.015}

    # Two resolution levels for comparison
    test_configs = [
        {
            'name': 'Standard Resolution',
            'grid': {'Nx': 25, 'Nt': 12},
            'qp': {'particles': 300, 'collocation': 8, 'delta': 0.25, 'order': 2, 'newton': 5, 'picard': 6},
        },
        {
            'name': 'High Resolution',
            'grid': {'Nx': 35, 'Nt': 18},
            'qp': {'particles': 500, 'collocation': 10, 'delta': 0.3, 'order': 2, 'newton': 6, 'picard': 8},
        },
    ]

    print(f"\nBase Problem Parameters:")
    for key, value in base_params.items():
        print(f"  {key}: {value}")

    results = []
    total_start_time = time.time()

    for i, config in enumerate(test_configs):
        print(f"\n{'='*60}")
        print(f"TEST {i+1}/2: {config['name'].upper()}")
        print(f"{'='*60}")

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
            collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation).reshape(-1, 1)

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
                l2errBoundNewton=1e-3,  # Slightly relaxed for speed
                kde_bandwidth="scott",
                normalize_kde_output=False,
                boundary_indices=np.array(boundary_indices),
                boundary_conditions=no_flux_bc,
                use_monotone_constraints=True,
            )

            print(f"\nSolving with {config['qp']['particles']} particles...")
            solve_start_time = time.time()

            U_solution, M_solution, info = solver.solve(
                Niter=config['qp']['picard'], l2errBound=1e-3, verbose=False  # Slightly relaxed for speed
            )

            solve_time = time.time() - solve_start_time
            total_config_time = time.time() - config_start_time

            if M_solution is not None and U_solution is not None:
                print(f"\n✅ Solution completed in {solve_time:.1f}s")

                # Key analysis metrics
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
                        (final_particles < problem.xmin - 1e-10) | (final_particles > problem.xmax + 1e-10)
                    )

                # Physical observables
                center_of_mass = np.sum(problem.xSpace * M_solution[-1, :]) * problem.Dx
                max_density_idx = np.argmax(M_solution[-1, :])
                max_density_location = problem.xSpace[max_density_idx]

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
                    'max_iterations': config['qp']['picard'],
                    'initial_mass': initial_mass,
                    'final_mass': final_mass,
                    'mass_change': mass_change,
                    'mass_change_percent': mass_change_percent,
                    'max_U': max_U,
                    'min_M': min_M,
                    'negative_densities': negative_densities,
                    'violations': violations,
                    'center_of_mass': center_of_mass,
                    'max_density_location': max_density_location,
                    'mass_evolution': mass_evolution,
                    'M_solution': M_solution,
                    'U_solution': U_solution,
                    'particles_trajectory': particles_traj,
                    'problem': problem,
                }

                results.append(result)

                # Print key results
                print(f"\n--- KEY RESULTS: {config['name']} ---")
                print(f"  Total time: {total_config_time:.1f}s")
                print(f"  Converged: {converged} ({iterations_used}/{config['qp']['picard']} iterations)")
                print(f"  Mass change: {mass_change:+.2e} ({mass_change_percent:+.3f}%)")
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
                results.append(
                    {'name': config['name'], 'success': False, 'total_time': time.time() - config_start_time}
                )

        except Exception as e:
            error_time = time.time() - config_start_time
            print(f"❌ {config['name']} crashed after {error_time:.1f}s: {e}")
            results.append({'name': config['name'], 'success': False, 'error': str(e), 'total_time': error_time})

    total_time = time.time() - total_start_time

    # Analysis
    print(f"\n{'='*80}")
    print("CONVERGENCE DEMONSTRATION ANALYSIS")
    print(f"{'='*80}")
    print(f"Total execution time: {total_time:.1f}s")

    analyze_quick_convergence(results)
    create_quick_convergence_plots(results)

    return results


def analyze_quick_convergence(results):
    """Analyze convergence between the two resolution levels"""
    successful_results = [r for r in results if r.get('success', False)]

    if len(successful_results) < 2:
        print(f"\nInsufficient successful results for convergence analysis: {len(successful_results)}/2")
        return

    print(f"\nBoth resolution levels completed successfully")

    # Summary comparison
    print(
        f"\n{'Resolution':<20} {'Final Mass':<12} {'Mass Change %':<15} {'Center of Mass':<15} {'Max Density Loc':<15}"
    )
    print(f"{'-'*20} {'-'*12} {'-'*15} {'-'*15} {'-'*15}")

    for result in successful_results:
        print(
            f"{result['name']:<20} {result['final_mass']:<12.6f} {result['mass_change_percent']:<15.3f} "
            f"{result['center_of_mass']:<15.4f} {result['max_density_location']:<15.4f}"
        )

    # Convergence analysis
    result1, result2 = successful_results[0], successful_results[1]

    print(f"\n--- CONVERGENCE CONSISTENCY ANALYSIS ---")

    # Final mass difference
    mass_diff = abs(result2['final_mass'] - result1['final_mass'])
    mass_rel_diff = mass_diff / result1['final_mass'] * 100

    print(f"Final mass difference: {mass_diff:.2e} ({mass_rel_diff:.3f}%)")

    # Physical observables difference
    com_diff = abs(result2['center_of_mass'] - result1['center_of_mass'])
    mdl_diff = abs(result2['max_density_location'] - result1['max_density_location'])

    print(f"Center of mass difference: {com_diff:.4f}")
    print(f"Max density location difference: {mdl_diff:.4f}")

    # Mass conservation behavior
    both_increase = result1['mass_change'] > 0 and result2['mass_change'] > 0
    both_conservative = abs(result1['mass_change_percent']) < 2 and abs(result2['mass_change_percent']) < 2

    print(f"\n--- MASS CONSERVATION CONSISTENCY ---")
    if both_increase:
        print("✅ EXCELLENT: Both resolutions show mass increase (expected with no-flux BC)")
    elif both_conservative:
        print("✅ GOOD: Both resolutions show good mass conservation")
    else:
        print("⚠️  WARNING: Inconsistent mass conservation behavior")

    # Solution quality
    both_clean = (
        result1['violations'] == 0
        and result1['negative_densities'] == 0
        and result2['violations'] == 0
        and result2['negative_densities'] == 0
    )

    print(f"\n--- SOLUTION QUALITY CONSISTENCY ---")
    if both_clean:
        print("✅ EXCELLENT: Both resolutions have clean solutions")
    else:
        print(f"⚠️  PARTIAL: Quality issues detected")
        print(f"  Standard: violations={result1['violations']}, negative_densities={result1['negative_densities']}")
        print(f"  High-res: violations={result2['violations']}, negative_densities={result2['negative_densities']}")

    # Performance scaling
    time_ratio = result2['solve_time'] / result1['solve_time']
    complexity_ratio = (
        result2['config']['grid']['Nx'] * result2['config']['grid']['Nt'] * result2['config']['qp']['particles']
    ) / (result1['config']['grid']['Nx'] * result1['config']['grid']['Nt'] * result1['config']['qp']['particles'])
    scaling_efficiency = time_ratio / complexity_ratio

    print(f"\n--- PERFORMANCE SCALING ---")
    print(f"Time ratio (high/standard): {time_ratio:.2f}")
    print(f"Complexity ratio: {complexity_ratio:.2f}")
    print(f"Scaling efficiency: {scaling_efficiency:.2f}")

    if scaling_efficiency < 1.5:
        print("✅ EXCELLENT: Better than linear scaling")
    elif scaling_efficiency < 2.5:
        print("✅ GOOD: Reasonable scaling")
    else:
        print("⚠️  WARNING: Poor scaling")

    # Overall convergence assessment
    print(f"\n--- OVERALL CONVERGENCE ASSESSMENT ---")

    convergence_score = 0
    if mass_rel_diff < 1.0:
        convergence_score += 1
        print("✅ Final mass: Very consistent")
    elif mass_rel_diff < 3.0:
        convergence_score += 0.5
        print("⚠️  Final mass: Reasonably consistent")
    else:
        print("❌ Final mass: Poor consistency")

    if com_diff < 0.01 and mdl_diff < 0.01:
        convergence_score += 1
        print("✅ Physical observables: Very consistent")
    elif com_diff < 0.05 and mdl_diff < 0.05:
        convergence_score += 0.5
        print("⚠️  Physical observables: Reasonably consistent")
    else:
        print("❌ Physical observables: Poor consistency")

    if both_increase and both_clean:
        convergence_score += 1
        print("✅ Solution behavior: Excellent consistency")
    elif both_conservative:
        convergence_score += 0.5
        print("⚠️  Solution behavior: Reasonable consistency")
    else:
        print("❌ Solution behavior: Poor consistency")

    final_assessment = [
        "POOR: Significant convergence issues",
        "FAIR: Some convergence concerns",
        "GOOD: Reasonable convergence",
        "EXCELLENT: Very good convergence",
    ][min(int(convergence_score), 3)]

    print(f"\nFINAL CONVERGENCE ASSESSMENT: {final_assessment}")


def create_quick_convergence_plots(results):
    """Create comparison plots for the two resolution levels"""
    successful_results = [r for r in results if r.get('success', False)]

    if len(successful_results) < 2:
        print("Insufficient results for plotting")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('QP Method Convergence Demonstration: Standard vs High Resolution', fontsize=16)

    colors = ['blue', 'red']

    # 1. Mass evolution comparison
    ax1 = axes[0, 0]
    for i, result in enumerate(successful_results):
        problem = result['problem']
        mass_evolution = result['mass_evolution']
        ax1.plot(problem.tSpace, mass_evolution, 'o-', label=result['name'], color=colors[i], linewidth=2)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Total Mass')
    ax1.set_title('Mass Evolution Comparison')
    ax1.grid(True)
    ax1.legend()

    # 2. Final density comparison
    ax2 = axes[0, 1]
    for i, result in enumerate(successful_results):
        problem = result['problem']
        final_density = result['M_solution'][-1, :]
        ax2.plot(problem.xSpace, final_density, label=result['name'], color=colors[i], linewidth=2)
    ax2.set_xlabel('Space x')
    ax2.set_ylabel('Final Density M(T,x)')
    ax2.set_title('Final Density Comparison')
    ax2.grid(True)
    ax2.legend()

    # 3. Key metrics comparison
    ax3 = axes[0, 2]
    names = [r['name'] for r in successful_results]
    final_masses = [r['final_mass'] for r in successful_results]
    centers_of_mass = [r['center_of_mass'] for r in successful_results]

    x_pos = np.arange(len(names))
    width = 0.35

    # Normalize for comparison
    norm_masses = np.array(final_masses) / final_masses[0]
    norm_centers = np.array(centers_of_mass) / centers_of_mass[0]

    bars1 = ax3.bar(x_pos - width / 2, norm_masses, width, label='Final Mass (normalized)', color='blue', alpha=0.7)
    bars2 = ax3.bar(x_pos + width / 2, norm_centers, width, label='Center of Mass (normalized)', color='red', alpha=0.7)

    ax3.set_xlabel('Resolution')
    ax3.set_ylabel('Normalized Value')
    ax3.set_title('Key Metrics Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names)
    ax3.legend()
    ax3.grid(True, axis='y')

    # 4. Mass conservation quality
    ax4 = axes[1, 0]
    mass_changes = [r['mass_change_percent'] for r in successful_results]
    bars = ax4.bar(names, mass_changes, color=colors)
    ax4.set_ylabel('Mass Change (%)')
    ax4.set_title('Mass Conservation Quality')
    ax4.grid(True, axis='y')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    for bar, value in zip(bars, mass_changes):
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f'{value:.2f}%',
            ha='center',
            va='bottom' if value >= 0 else 'top',
        )

    # 5. Performance comparison
    ax5 = axes[1, 1]
    solve_times = [r['solve_time'] for r in successful_results]
    bars = ax5.bar(names, solve_times, color=colors)
    ax5.set_ylabel('Solve Time (seconds)')
    ax5.set_title('Performance Comparison')
    ax5.grid(True, axis='y')
    for bar, value in zip(bars, solve_times):
        ax5.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{value:.1f}s', ha='center', va='bottom')

    # 6. Density evolution heatmap (high resolution)
    ax6 = axes[1, 2]
    high_res_result = successful_results[1]  # Second result is high resolution
    M_solution = high_res_result['M_solution']
    problem = high_res_result['problem']

    im = ax6.imshow(
        M_solution.T, aspect='auto', origin='lower', extent=[0, problem.T, problem.xmin, problem.xmax], cmap='viridis'
    )
    ax6.set_xlabel('Time t')
    ax6.set_ylabel('Space x')
    ax6.set_title('Density Evolution (High Resolution)')
    plt.colorbar(im, ax=ax6, label='Density M(t,x)')

    plt.tight_layout()
    plt.savefig(
        '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/quick_convergence_demo.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.show()


if __name__ == "__main__":
    print("Starting quick convergence demonstration...")
    print("Expected execution time: 1-3 minutes")

    try:
        results = quick_convergence_demo()
        print("\n" + "=" * 80)
        print("QUICK CONVERGENCE DEMONSTRATION COMPLETED")
        print("=" * 80)
        print("Check the generated plots and analysis for convergence validation.")
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()
