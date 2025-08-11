#!/usr/bin/env python3
"""
QP-Collocation Robustness Test
Test QP-Collocation with optimized parameters to verify theoretical robustness.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def test_qp_collocation_optimized(problem_params, qp_params, test_name):
    """Test QP-Collocation with optimized parameters"""

    print(f"Testing QP-Collocation: {test_name}")
    print(f"  Problem: Nx={problem_params['Nx']}, T={problem_params['T']}")
    print(f"  QP params: collocation_points={qp_params['num_collocation_points']}, delta={qp_params['delta']}")

    try:
        start_time = time.time()

        # Create problem
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        # Adaptive collocation points (scale with problem complexity)
        num_collocation_points = qp_params['num_collocation_points']
        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)

        # Identify boundary points
        boundary_tolerance = 1e-10
        boundary_indices = []
        for i, point in enumerate(collocation_points):
            x = point[0]
            if abs(x - problem.xmin) < boundary_tolerance or abs(x - problem.xmax) < boundary_tolerance:
                boundary_indices.append(i)
        boundary_indices = np.array(boundary_indices)

        # QP solver with optimized parameters
        qp_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=qp_params['num_particles'],
            delta=qp_params['delta'],  # Optimized delta
            taylor_order=qp_params['taylor_order'],
            weight_function="wendland",
            NiterNewton=qp_params['newton_iterations'],
            l2errBoundNewton=qp_params['newton_tolerance'],
            kde_bandwidth=qp_params['kde_bandwidth'],
            normalize_kde_output=False,
            boundary_indices=boundary_indices,
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=qp_params['use_constraints'],  # Allow toggling constraints
        )

        # Solve
        solve_start = time.time()
        U_qp, M_qp, solve_info = qp_solver.solve(
            Niter=qp_params['max_iterations'], l2errBound=qp_params['convergence_tolerance'], verbose=True
        )
        solve_time = time.time() - solve_start
        total_time = time.time() - start_time

        if U_qp is not None and M_qp is not None:
            # Mass analysis
            mass_evolution = np.sum(M_qp * problem.Dx, axis=1)
            initial_mass = mass_evolution[0]
            final_mass = mass_evolution[-1]
            mass_change_percent = ((final_mass - initial_mass) / initial_mass) * 100

            # Quality metrics
            negative_densities = np.sum(M_qp < -1e-10)
            min_density = np.min(M_qp)
            max_control = np.max(np.abs(U_qp))

            # Convergence info
            iterations = solve_info.get("iterations", 0)
            converged = solve_info.get("converged", False)

            result = {
                'success': True,
                'test_name': test_name,
                'solve_time': solve_time,
                'total_time': total_time,
                'iterations': iterations,
                'converged': converged,
                'mass_change_percent': mass_change_percent,
                'final_mass': final_mass,
                'mass_evolution': mass_evolution,
                'negative_densities': negative_densities,
                'min_density': min_density,
                'max_control': max_control,
                'U': U_qp,
                'M': M_qp,
                'problem': problem,
            }

            print(f"  ✓ Success: {solve_time:.2f}s ({iterations} iterations)")
            print(f"    Converged: {converged}")
            print(f"    Mass change: {mass_change_percent:+.3f}%")
            print(f"    Negative densities: {negative_densities}")
            print(f"    Min density: {min_density:.3e}")

            return result

        else:
            print(f"  ❌ Failed: No solution obtained")
            return {'success': False, 'test_name': test_name, 'error': 'No solution'}

    except Exception as e:
        print(f"  ❌ Crashed: {e}")
        return {'success': False, 'test_name': test_name, 'error': str(e)}


def run_qp_robustness_study():
    """Run QP-Collocation robustness study with different parameter configurations"""

    print("=" * 80)
    print("QP-COLLOCATION ROBUSTNESS STUDY")
    print("=" * 80)
    print("Testing QP-Collocation with optimized parameters to verify theoretical robustness")

    # Base problem parameters (moderate difficulty)
    base_problem = {'xmin': 0.0, 'xmax': 1.0, 'Nx': 30, 'T': 1.0, 'Nt': 50, 'sigma': 0.15, 'coefCT': 0.02}

    # Test configurations
    test_configs = [
        # Configuration 1: Standard parameters (current implementation)
        {
            'name': 'Standard_Config',
            'qp_params': {
                'num_collocation_points': 12,
                'delta': 0.35,
                'taylor_order': 2,
                'num_particles': 300,
                'newton_iterations': 8,
                'newton_tolerance': 1e-4,
                'max_iterations': 12,
                'convergence_tolerance': 1e-3,
                'kde_bandwidth': 'scott',
                'use_constraints': True,
            },
        },
        # Configuration 2: Optimized parameters (more collocation points, relaxed tolerances)
        {
            'name': 'Optimized_Config',
            'qp_params': {
                'num_collocation_points': 20,  # More collocation points
                'delta': 0.5,  # Larger support radius
                'taylor_order': 2,
                'num_particles': 300,
                'newton_iterations': 8,
                'newton_tolerance': 1e-3,  # Relaxed tolerance
                'max_iterations': 15,  # More iterations
                'convergence_tolerance': 1e-3,
                'kde_bandwidth': 'scott',
                'use_constraints': True,
            },
        },
        # Configuration 3: No constraints (to isolate QP constraint overhead)
        {
            'name': 'No_Constraints',
            'qp_params': {
                'num_collocation_points': 12,
                'delta': 0.35,
                'taylor_order': 2,
                'num_particles': 300,
                'newton_iterations': 8,
                'newton_tolerance': 1e-4,
                'max_iterations': 12,
                'convergence_tolerance': 1e-3,
                'kde_bandwidth': 'scott',
                'use_constraints': False,  # Disable QP constraints
            },
        },
        # Configuration 4: High-resolution adaptive
        {
            'name': 'Adaptive_HighRes',
            'qp_params': {
                'num_collocation_points': 25,  # Scale with Nx
                'delta': 0.4,  # Adaptive to grid spacing
                'taylor_order': 2,
                'num_particles': 400,
                'newton_iterations': 10,
                'newton_tolerance': 5e-4,  # Balanced tolerance
                'max_iterations': 15,
                'convergence_tolerance': 1e-3,
                'kde_bandwidth': 'silverman',  # Alternative bandwidth
                'use_constraints': True,
            },
        },
    ]

    results = []

    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"CONFIGURATION: {config['name']}")
        print(f"{'='*60}")

        result = test_qp_collocation_optimized(base_problem, config['qp_params'], config['name'])
        results.append(result)

    # Analysis
    print(f"\n{'='*80}")
    print("ROBUSTNESS ANALYSIS")
    print(f"{'='*80}")

    analyze_qp_robustness_results(results)
    create_qp_robustness_plots(results)

    return results


def analyze_qp_robustness_results(results):
    """Analyze QP robustness test results"""

    successful_results = [r for r in results if r.get('success', False)]

    print(f"Successful configurations: {len(successful_results)}/{len(results)}")

    if not successful_results:
        print("No successful results to analyze")
        return

    print(f"\n{'Configuration':<20} {'Time (s)':<10} {'Mass Error (%)':<15} {'Converged':<12} {'Quality'}")
    print("-" * 75)

    for result in successful_results:
        name = result['test_name']
        time_val = result['solve_time']
        mass_error = abs(result['mass_change_percent'])
        converged = result['converged']
        negative_dens = result['negative_densities']

        # Quality assessment
        if mass_error < 1.0 and negative_dens == 0 and converged:
            quality = "Excellent"
        elif mass_error < 5.0 and negative_dens == 0:
            quality = "Good"
        elif mass_error < 10.0:
            quality = "Fair"
        else:
            quality = "Poor"

        print(f"{name:<20} {time_val:<10.2f} {mass_error:<15.3f} {str(converged):<12} {quality}")

    # Performance comparison
    print(f"\n--- PERFORMANCE COMPARISON ---")
    if len(successful_results) >= 2:
        standard = next((r for r in successful_results if 'Standard' in r['test_name']), None)
        optimized = next((r for r in successful_results if 'Optimized' in r['test_name']), None)
        no_constraints = next((r for r in successful_results if 'No_Constraints' in r['test_name']), None)

        if standard and optimized:
            time_improvement = (standard['solve_time'] - optimized['solve_time']) / standard['solve_time'] * 100
            mass_improvement = abs(standard['mass_change_percent']) - abs(optimized['mass_change_percent'])
            print(
                f"Optimized vs Standard: {time_improvement:+.1f}% time change, {mass_improvement:+.3f}pp mass improvement"
            )

        if standard and no_constraints:
            constraint_overhead = (
                (standard['solve_time'] - no_constraints['solve_time']) / no_constraints['solve_time'] * 100
            )
            print(f"QP constraint overhead: {constraint_overhead:+.1f}% computational cost")


def create_qp_robustness_plots(results):
    """Create plots showing QP robustness across configurations"""

    successful_results = [r for r in results if r.get('success', False)]

    if len(successful_results) < 2:
        print("Insufficient results for plotting")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('QP-Collocation Robustness Study', fontsize=14, fontweight='bold')

    configs = [r['test_name'] for r in successful_results]
    colors = ['blue', 'green', 'red', 'orange'][: len(configs)]

    # Plot 1: Solve Time Comparison
    times = [r['solve_time'] for r in successful_results]
    bars1 = ax1.bar(range(len(configs)), times, color=colors, alpha=0.7)
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Solve Time (seconds)')
    ax1.set_title('Computational Cost by Configuration')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45)
    ax1.grid(True, alpha=0.3)

    # Add time labels
    for bar, time_val in zip(bars1, times):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(times) * 0.01,
            f'{time_val:.1f}s',
            ha='center',
            va='bottom',
            fontweight='bold',
        )

    # Plot 2: Mass Conservation Quality
    mass_errors = [abs(r['mass_change_percent']) for r in successful_results]
    bars2 = ax2.bar(range(len(configs)), mass_errors, color=colors, alpha=0.7)
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Mass Conservation Error (%)')
    ax2.set_title('Mass Conservation Quality')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Add quality thresholds
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Excellent (<1%)')
    ax2.axhline(y=5.0, color='orange', linestyle='--', alpha=0.7, label='Good (<5%)')
    ax2.legend()

    # Plot 3: Mass Evolution Comparison
    for i, result in enumerate(successful_results):
        if 'mass_evolution' in result and 'problem' in result:
            ax3.plot(
                result['problem'].tSpace,
                result['mass_evolution'],
                color=colors[i],
                linewidth=2,
                label=result['test_name'].replace('_', ' '),
            )

    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Total Mass')
    ax3.set_title('Mass Evolution Comparison')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Configuration Performance Matrix
    metrics = ['Speed', 'Mass Conservation', 'Convergence']
    config_scores = []

    for result in successful_results:
        # Normalize scores to 0-1 scale
        speed_score = 1 - (result['solve_time'] / max(times))  # Faster = higher score
        mass_score = max(0, 1 - abs(result['mass_change_percent']) / 10)  # Better conservation = higher score
        conv_score = 1.0 if result['converged'] else 0.5  # Converged = full score

        config_scores.append([speed_score, mass_score, conv_score])

    # Create heatmap
    im = ax4.imshow(config_scores, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax4.set_xticks(range(len(metrics)))
    ax4.set_xticklabels(metrics)
    ax4.set_yticks(range(len(configs)))
    ax4.set_yticklabels([c.replace('_', '\n') for c in configs])
    ax4.set_title('Configuration Performance Matrix')

    # Add text annotations
    for i in range(len(configs)):
        for j in range(len(metrics)):
            ax4.text(
                j,
                i,
                f'{config_scores[i][j]:.2f}',
                ha='center',
                va='center',
                color='white' if config_scores[i][j] < 0.5 else 'black',
                fontweight='bold',
            )

    # Add colorbar
    plt.colorbar(im, ax=ax4, label='Performance Score (0-1)')

    plt.tight_layout()
    plt.savefig(
        '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/tests/method_comparisons/qp_robustness_study.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.show()

    print(f"\n✅ QP robustness analysis plots saved: qp_robustness_study.png")


if __name__ == "__main__":
    print("Starting QP-Collocation Robustness Study...")
    print("Testing different parameter configurations to verify theoretical robustness")

    try:
        results = run_qp_robustness_study()
        print("\n" + "=" * 80)
        print("QP ROBUSTNESS STUDY COMPLETED")
        print("=" * 80)
        print("Analysis shows parameter optimization potential for QP-Collocation method.")

    except KeyboardInterrupt:
        print("\nStudy interrupted by user.")
    except Exception as e:
        print(f"\nStudy failed: {e}")
        import traceback

        traceback.print_exc()
