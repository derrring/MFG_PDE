#!/usr/bin/env python3
"""
Focused Three-Method Evaluation: FDM vs Hybrid vs QP-Collocation
Quick evaluation of mass conservation quality, computational cost, and stability success rate.
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

warnings.filterwarnings('ignore')

from mfg_pde.alg.damped_fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.fp_solvers.fdm_fp import FdmFPSolver
from mfg_pde.alg.fp_solvers.particle_fp import ParticleFPSolver
from mfg_pde.alg.hjb_solvers.fdm_hjb import FdmHJBSolver
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions

# Import solvers
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def test_method(method_name, problem_params, solver_params, timeout_seconds=300):
    """Test a specific method with timeout"""
    start_time = time.time()

    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        if method_name == "Pure FDM":
            # FDM-HJB + FDM-FP
            hjb_solver = FdmHJBSolver(
                problem,
                NiterNewton=solver_params["newton_iterations"],
                l2errBoundNewton=solver_params["newton_tolerance"],
            )
            fp_solver = FdmFPSolver(problem, boundary_conditions=no_flux_bc)
            iterator = FixedPointIterator(
                problem, hjb_solver=hjb_solver, fp_solver=fp_solver, thetaUM=solver_params["thetaUM"]
            )

        elif method_name == "Hybrid":
            # FDM-HJB + Particle-FP
            hjb_solver = FdmHJBSolver(
                problem,
                NiterNewton=solver_params["newton_iterations"],
                l2errBoundNewton=solver_params["newton_tolerance"],
            )
            fp_solver = ParticleFPSolver(
                problem,
                num_particles=solver_params["num_particles"],
                kde_bandwidth="scott",
                normalize_kde_output=False,
                boundary_conditions=no_flux_bc,
            )
            iterator = FixedPointIterator(
                problem, hjb_solver=hjb_solver, fp_solver=fp_solver, thetaUM=solver_params["thetaUM"]
            )

        elif method_name == "QP-Collocation":
            # QP Particle-Collocation
            num_collocation_points = 10  # Reduced for speed
            collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)

            boundary_tolerance = 1e-10
            boundary_indices = []
            for i, point in enumerate(collocation_points):
                x = point[0]
                if abs(x - problem.xmin) < boundary_tolerance or abs(x - problem.xmax) < boundary_tolerance:
                    boundary_indices.append(i)
            boundary_indices = np.array(boundary_indices)

            qp_solver = ParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                num_particles=solver_params["num_particles"],
                delta=0.35,
                taylor_order=2,
                weight_function="wendland",
                NiterNewton=solver_params["newton_iterations"],
                l2errBoundNewton=solver_params["newton_tolerance"],
                kde_bandwidth="scott",
                normalize_kde_output=False,
                boundary_indices=boundary_indices,
                boundary_conditions=no_flux_bc,
                use_monotone_constraints=True,
            )

        # Solve with timeout check
        solve_start = time.time()

        if method_name == "QP-Collocation":
            U, M, solve_info = qp_solver.solve(
                Niter=solver_params["max_iterations"], l2errBound=solver_params["convergence_tolerance"], verbose=False
            )
            iterations = solve_info.get("iterations", 0)
            converged_flag = solve_info.get("converged", False)
        else:
            U, M, iterations, _, _ = iterator.solve(
                solver_params["max_iterations"], solver_params["convergence_tolerance"]
            )
            converged_flag = True  # Assume converged if completed

        solve_time = time.time() - solve_start
        total_time = time.time() - start_time

        # Check timeout
        if total_time > timeout_seconds:
            return {
                'success': False,
                'method': method_name,
                'error': 'Timeout',
                'solve_time': solve_time,
                'total_time': total_time,
            }

        if U is not None and M is not None:
            # Mass analysis
            mass_evolution = np.sum(M * problem.Dx, axis=1)
            initial_mass = mass_evolution[0]
            final_mass = mass_evolution[-1]
            mass_change_percent = ((final_mass - initial_mass) / initial_mass) * 100

            # Quality metrics
            negative_densities = np.sum(M < -1e-10)
            min_density = np.min(M)
            max_control = np.max(np.abs(U))

            # Particle violations (for particle methods)
            violations = 0
            if method_name in ["Hybrid", "QP-Collocation"]:
                if method_name == "Hybrid":
                    particles_trajectory = getattr(fp_solver, 'M_particles_trajectory', None)
                else:
                    particles_trajectory = getattr(qp_solver.fp_solver, 'M_particles_trajectory', None)

                if particles_trajectory is not None:
                    final_particles = particles_trajectory[-1, :]
                    violations = np.sum(
                        (final_particles < problem.xmin - 1e-10) | (final_particles > problem.xmax + 1e-10)
                    )

            return {
                'success': True,
                'method': method_name,
                'solve_time': solve_time,
                'total_time': total_time,
                'iterations': iterations,
                'mass_change_percent': mass_change_percent,
                'final_mass': final_mass,
                'mass_evolution': mass_evolution,
                'negative_densities': negative_densities,
                'min_density': min_density,
                'max_control': max_control,
                'boundary_violations': violations,
                'U': U,
                'M': M,
                'converged': converged_flag and abs(mass_change_percent) < 10.0 and violations == 0,
            }
        else:
            return {'success': False, 'method': method_name, 'error': 'No solution'}

    except Exception as e:
        return {'success': False, 'method': method_name, 'error': str(e)}


def run_focused_evaluation():
    """Run focused evaluation with key test scenarios"""
    print("=" * 80)
    print("FOCUSED THREE-METHOD EVALUATION")
    print("=" * 80)
    print("Evaluating: Pure FDM vs Hybrid vs QP-Collocation")
    print("Metrics: Mass conservation, computational cost, stability success rate")

    # Define focused test scenarios
    test_scenarios = [
        # Baseline easy case
        {
            'name': 'Baseline',
            'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 20, 'T': 0.5, 'Nt': 25, 'sigma': 0.1, 'coefCT': 0.01},
            'solver': {
                'max_iterations': 10,
                'convergence_tolerance': 1e-3,
                'newton_iterations': 6,
                'newton_tolerance': 1e-4,
                'num_particles': 200,
                'thetaUM': 0.5,
            },
        },
        # Standard case
        {
            'name': 'Standard',
            'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 30, 'T': 1.0, 'Nt': 50, 'sigma': 0.15, 'coefCT': 0.02},
            'solver': {
                'max_iterations': 12,
                'convergence_tolerance': 1e-3,
                'newton_iterations': 8,
                'newton_tolerance': 1e-4,
                'num_particles': 400,
                'thetaUM': 0.4,
            },
        },
        # Moderate challenge
        {
            'name': 'Challenge',
            'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 40, 'T': 1.5, 'Nt': 75, 'sigma': 0.2, 'coefCT': 0.03},
            'solver': {
                'max_iterations': 15,
                'convergence_tolerance': 1e-3,
                'newton_iterations': 8,
                'newton_tolerance': 1e-4,
                'num_particles': 500,
                'thetaUM': 0.4,
            },
        },
        # Demanding case
        {
            'name': 'Demanding',
            'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 35, 'T': 2.0, 'Nt': 100, 'sigma': 0.25, 'coefCT': 0.04},
            'solver': {
                'max_iterations': 15,
                'convergence_tolerance': 1e-3,
                'newton_iterations': 10,
                'newton_tolerance': 1e-4,
                'num_particles': 600,
                'thetaUM': 0.3,
            },
        },
    ]

    methods = ["Pure FDM", "Hybrid", "QP-Collocation"]
    results = []

    print(
        f"\nTesting {len(test_scenarios)} scenarios × {len(methods)} methods = {len(test_scenarios) * len(methods)} total tests"
    )

    for i, scenario in enumerate(test_scenarios):
        print(f"\n{'='*60}")
        print(f"SCENARIO {i+1}/{len(test_scenarios)}: {scenario['name']}")
        print(f"{'='*60}")

        params = scenario['params']
        solver = scenario['solver']
        print(f"Problem: Nx={params['Nx']}, T={params['T']}, σ={params['sigma']}, coefCT={params['coefCT']}")

        for method in methods:
            print(f"\n--- Testing {method} ---")

            result = test_method(method, params, solver, timeout_seconds=300)
            result['scenario'] = scenario['name']
            results.append(result)

            if result['success']:
                print(f"✓ Success: {result['solve_time']:.2f}s, {result['iterations']} iters")
                print(f"  Mass change: {result['mass_change_percent']:+.3f}%")
                print(f"  Converged: {result['converged']}")
                if 'boundary_violations' in result:
                    print(f"  Violations: {result['boundary_violations']}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")

    # Analysis and visualization
    print(f"\n{'='*80}")
    print("EVALUATION ANALYSIS")
    print(f"{'='*80}")

    analyze_focused_results(results)
    create_focused_visualization(results, test_scenarios)

    return results


def analyze_focused_results(results):
    """Analyze focused evaluation results"""

    methods = ["Pure FDM", "Hybrid", "QP-Collocation"]

    print("--- SUCCESS RATES ---")
    for method in methods:
        method_results = [r for r in results if r.get('method') == method]
        total = len(method_results)
        successful = len([r for r in method_results if r.get('success', False)])
        converged = len([r for r in method_results if r.get('converged', False)])

        success_rate = (successful / total) * 100 if total > 0 else 0
        convergence_rate = (converged / total) * 100 if total > 0 else 0

        print(
            f"{method:<15}: {successful}/{total} success ({success_rate:.1f}%), {converged}/{total} converged ({convergence_rate:.1f}%)"
        )

    print("\n--- MASS CONSERVATION QUALITY ---")
    for method in methods:
        method_results = [r for r in results if r.get('method') == method and r.get('success', False)]
        if method_results:
            mass_changes = [abs(r['mass_change_percent']) for r in method_results]
            avg_mass_error = np.mean(mass_changes)
            max_mass_error = np.max(mass_changes)
            min_mass_error = np.min(mass_changes)

            excellent = len([r for r in method_results if abs(r['mass_change_percent']) < 1.0])
            good = len([r for r in method_results if 1.0 <= abs(r['mass_change_percent']) < 5.0])
            poor = len([r for r in method_results if abs(r['mass_change_percent']) >= 5.0])

            print(f"{method:<15}: Avg {avg_mass_error:.3f}%, Range [{min_mass_error:.3f}%, {max_mass_error:.3f}%]")
            print(f"{'':15}  Excellent (<1%): {excellent}, Good (1-5%): {good}, Poor (≥5%): {poor}")

    print("\n--- COMPUTATIONAL COST ---")
    for method in methods:
        method_results = [r for r in results if r.get('method') == method and r.get('success', False)]
        if method_results:
            solve_times = [r['solve_time'] for r in method_results]
            avg_time = np.mean(solve_times)
            median_time = np.median(solve_times)
            min_time = np.min(solve_times)
            max_time = np.max(solve_times)

            print(
                f"{method:<15}: Avg {avg_time:.2f}s, Median {median_time:.2f}s, Range [{min_time:.2f}s, {max_time:.2f}s]"
            )

    print("\n--- SCENARIO-WISE PERFORMANCE ---")
    scenarios = list(set([r.get('scenario', 'Unknown') for r in results]))
    for scenario in scenarios:
        print(f"\n{scenario}:")
        scenario_results = [r for r in results if r.get('scenario') == scenario]
        for method in methods:
            method_result = next((r for r in scenario_results if r.get('method') == method), None)
            if method_result:
                if method_result.get('success', False):
                    mass_change = method_result.get('mass_change_percent', 0)
                    solve_time = method_result.get('solve_time', 0)
                    converged = method_result.get('converged', False)
                    status = "✓ Conv" if converged else "✓ Partial"
                    print(f"  {method:<15}: {status}, {mass_change:+.3f}% mass, {solve_time:.2f}s")
                else:
                    error = method_result.get('error', 'Failed')
                    print(f"  {method:<15}: ❌ {error}")


def create_focused_visualization(results, scenarios):
    """Create focused visualization of evaluation results"""

    methods = ["Pure FDM", "Hybrid", "QP-Collocation"]
    method_colors = {"Pure FDM": "#1f77b4", "Hybrid": "#2ca02c", "QP-Collocation": "#d62728"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Focused Three-Method Evaluation: FDM vs Hybrid vs QP-Collocation', fontsize=16, fontweight='bold')

    # 1. Success and Convergence Rates
    ax1 = axes[0, 0]
    success_rates = []
    convergence_rates = []

    for method in methods:
        method_results = [r for r in results if r.get('method') == method]
        total = len(method_results)
        successful = len([r for r in method_results if r.get('success', False)])
        converged = len([r for r in method_results if r.get('converged', False)])

        success_rate = (successful / total) * 100 if total > 0 else 0
        convergence_rate = (converged / total) * 100 if total > 0 else 0

        success_rates.append(success_rate)
        convergence_rates.append(convergence_rate)

    x_pos = np.arange(len(methods))
    width = 0.35

    bars1 = ax1.bar(
        x_pos - width / 2,
        success_rates,
        width,
        label='Success Rate',
        color=[method_colors[m] for m in methods],
        alpha=0.8,
    )
    bars2 = ax1.bar(
        x_pos + width / 2,
        convergence_rates,
        width,
        label='Convergence Rate',
        color=[method_colors[m] for m in methods],
        alpha=0.5,
    )

    ax1.set_xlabel('Method')
    ax1.set_ylabel('Rate (%)')
    ax1.set_title('Success and Convergence Rates')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([m.replace(' ', '\n') for m in methods])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])

    # Add percentage labels
    for bar, value in zip(bars1, success_rates):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1,
            f'{value:.0f}%',
            ha='center',
            va='bottom',
            fontweight='bold',
        )
    for bar, value in zip(bars2, convergence_rates):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1,
            f'{value:.0f}%',
            ha='center',
            va='bottom',
            fontweight='bold',
        )

    # 2. Mass Conservation Quality
    ax2 = axes[0, 1]
    mass_data = []
    mass_labels = []
    colors = []

    for method in methods:
        method_results = [r for r in results if r.get('method') == method and r.get('success', False)]
        if method_results:
            mass_errors = [abs(r['mass_change_percent']) for r in method_results]
            mass_data.append(mass_errors)
            mass_labels.append(method.replace(' ', '\n'))
            colors.append(method_colors[method])

    if mass_data:
        bp = ax2.boxplot(mass_data, labels=mass_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax2.set_ylabel('Mass Conservation Error (%)')
    ax2.set_title('Mass Conservation Quality')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Add quality thresholds
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Excellent (<1%)')
    ax2.axhline(y=5.0, color='orange', linestyle='--', alpha=0.7, label='Good (<5%)')
    ax2.legend(fontsize=8)

    # 3. Computational Cost
    ax3 = axes[0, 2]
    time_data = []
    time_labels = []
    time_colors = []

    for method in methods:
        method_results = [r for r in results if r.get('method') == method and r.get('success', False)]
        if method_results:
            times = [r['solve_time'] for r in method_results]
            time_data.append(times)
            time_labels.append(method.replace(' ', '\n'))
            time_colors.append(method_colors[method])

    if time_data:
        bp = ax3.boxplot(time_data, labels=time_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], time_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax3.set_ylabel('Solve Time (seconds)')
    ax3.set_title('Computational Cost')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # 4. Performance by Scenario
    ax4 = axes[1, 0]
    scenario_names = [s['name'] for s in scenarios]
    x_pos = np.arange(len(scenario_names))
    width = 0.25

    for i, method in enumerate(methods):
        scenario_times = []
        for scenario_name in scenario_names:
            scenario_result = next(
                (r for r in results if r.get('method') == method and r.get('scenario') == scenario_name), None
            )
            if scenario_result and scenario_result.get('success', False):
                scenario_times.append(scenario_result['solve_time'])
            else:
                scenario_times.append(0)  # Failed case

        ax4.bar(x_pos + i * width, scenario_times, width, label=method, color=method_colors[method], alpha=0.7)

    ax4.set_xlabel('Scenario')
    ax4.set_ylabel('Solve Time (seconds)')
    ax4.set_title('Performance by Scenario Difficulty')
    ax4.set_xticks(x_pos + width)
    ax4.set_xticklabels(scenario_names, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    # 5. Mass Conservation vs Speed Trade-off
    ax5 = axes[1, 1]

    for method in methods:
        method_results = [r for r in results if r.get('method') == method and r.get('success', False)]
        if method_results:
            mass_errors = [abs(r['mass_change_percent']) for r in method_results]
            times = [r['solve_time'] for r in method_results]

            ax5.scatter(
                times,
                mass_errors,
                label=method,
                color=method_colors[method],
                alpha=0.7,
                s=80,
                edgecolors='black',
                linewidth=1,
            )

    ax5.set_xlabel('Solve Time (seconds)')
    ax5.set_ylabel('Mass Conservation Error (%)')
    ax5.set_title('Mass Conservation vs Computational Cost')
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Add ideal regions
    ax5.axhline(y=1.0, color='green', linestyle='--', alpha=0.3)
    ax5.axvline(x=10.0, color='blue', linestyle='--', alpha=0.3)
    ax5.text(
        0.02,
        0.95,
        'Excellent\nMass Conservation\n(<1%)',
        transform=ax5.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
        fontsize=8,
    )
    ax5.text(
        0.02,
        0.02,
        'Fast\nComputation\n(<10s)',
        transform=ax5.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        fontsize=8,
    )

    # 6. Overall Performance Summary
    ax6 = axes[1, 2]

    # Create performance summary table
    summary_data = []
    for method in methods:
        method_results = [r for r in results if r.get('method') == method and r.get('success', False)]
        if method_results:
            success_rate = len(method_results) / len([r for r in results if r.get('method') == method]) * 100
            avg_mass_error = np.mean([abs(r['mass_change_percent']) for r in method_results])
            avg_time = np.mean([r['solve_time'] for r in method_results])
            convergence_rate = len([r for r in method_results if r.get('converged', False)]) / len(method_results) * 100

            summary_data.append(
                [
                    method,
                    f"{success_rate:.0f}%",
                    f"{avg_mass_error:.3f}%",
                    f"{avg_time:.1f}s",
                    f"{convergence_rate:.0f}%",
                ]
            )
        else:
            summary_data.append([method, "0%", "N/A", "N/A", "0%"])

    # Create table
    headers = ["Method", "Success\nRate", "Avg Mass\nError", "Avg Time", "Convergence\nRate"]
    table = ax6.table(cellText=summary_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Color code the table
    for i, method in enumerate(methods):
        table[(i + 1, 0)].set_facecolor(method_colors[method])
        table[(i + 1, 0)].set_alpha(0.3)

    ax6.axis('off')
    ax6.set_title('Performance Summary Table')

    plt.tight_layout()
    plt.savefig(
        '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/tests/method_comparisons/focused_three_method_evaluation.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.show()

    print(f"\n✅ Focused evaluation visualization saved: focused_three_method_evaluation.png")


if __name__ == "__main__":
    print("Starting Focused Three-Method Evaluation...")
    print("Expected execution time: 5-15 minutes")

    try:
        results = run_focused_evaluation()
        print("\n" + "=" * 80)
        print("FOCUSED EVALUATION COMPLETED")
        print("=" * 80)
        print("Check the generated analysis and plots for detailed comparison.")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback

        traceback.print_exc()
