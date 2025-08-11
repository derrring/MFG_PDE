#!/usr/bin/env python3
"""
Final Optimization Test
Complete validation of all optimization approaches with realistic parameters.
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

from mfg_pde.alg.hjb_solvers.smart_qp_gfdm_hjb import SmartQPGFDMHJBSolver
from mfg_pde.alg.hjb_solvers.tuned_smart_qp_gfdm_hjb import TunedSmartQPGFDMHJBSolver
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def final_optimization_test():
    """Final comprehensive test of QP optimization approaches"""
    print("=" * 80)
    print("FINAL QP OPTIMIZATION TEST")
    print("=" * 80)
    print("Testing Smart QP and Tuned Smart QP with realistic parameters")

    # Realistic problem parameters
    problem_params = {'xmin': 0.0, 'xmax': 1.0, 'Nx': 20, 'T': 1.0, 'Nt': 30, 'sigma': 0.12, 'coefCT': 0.02}

    print(f"Problem: Nx={problem_params['Nx']}, T={problem_params['T']}, Nt={problem_params['Nt']}")
    print(f"Parameters: σ={problem_params['sigma']}, coefCT={problem_params['coefCT']}")

    results = {}

    # Common solver parameters
    num_collocation_points = 10
    solver_params = {
        'num_particles': 120,
        'delta': 0.4,
        'taylor_order': 2,
        'newton_iterations': 4,
        'newton_tolerance': 1e-3,
        'max_picard_iterations': 6,
        'convergence_tolerance': 1e-3,
    }

    # Test 1: Smart QP-Collocation
    print(f"\n{'-'*60}")
    print("TESTING SMART QP-COLLOCATION")
    print(f"{'-'*60}")

    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)
        boundary_indices = [0, num_collocation_points - 1]

        smart_hjb_solver = SmartQPGFDMHJBSolver(
            problem=problem,
            collocation_points=collocation_points,
            delta=solver_params['delta'],
            taylor_order=solver_params['taylor_order'],
            weight_function="wendland",
            NiterNewton=solver_params['newton_iterations'],
            l2errBoundNewton=solver_params['newton_tolerance'],
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
            qp_usage_target=0.1,
        )

        smart_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=solver_params['num_particles'],
            delta=solver_params['delta'],
            taylor_order=solver_params['taylor_order'],
            weight_function="wendland",
            NiterNewton=solver_params['newton_iterations'],
            l2errBoundNewton=solver_params['newton_tolerance'],
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
        )
        smart_solver.hjb_solver = smart_hjb_solver

        print("Running Smart QP-Collocation solver...")
        start_time = time.time()
        U_smart, M_smart, info_smart = smart_solver.solve(
            Niter=solver_params['max_picard_iterations'],
            l2errBound=solver_params['convergence_tolerance'],
            verbose=True,
        )
        smart_time = time.time() - start_time

        # Calculate mass conservation
        Dx = (problem_params['xmax'] - problem_params['xmin']) / problem_params['Nx']
        initial_mass = np.sum(M_smart[0, :]) * Dx
        final_mass = np.sum(M_smart[-1, :]) * Dx
        mass_error = abs(final_mass - initial_mass) / initial_mass * 100

        # Get smart optimization statistics
        smart_stats = smart_hjb_solver.get_smart_qp_report()

        results['smart'] = {
            'method': 'Smart QP-Collocation',
            'success': True,
            'time': smart_time,
            'mass_error': mass_error,
            'converged': info_smart.get('converged', False),
            'iterations': info_smart.get('iterations', 0),
            'qp_usage_rate': smart_stats.get('qp_usage_rate', 1.0),
            'smart_stats': smart_stats,
            'U': U_smart,
            'M': M_smart,
        }

        print(f"✓ Smart QP completed: {smart_time:.1f}s, mass error: {mass_error:.2f}%")
        print(f"  QP Usage Rate: {smart_stats.get('qp_usage_rate', 0):.1%}")
        print(f"  Optimization Effectiveness: {smart_stats.get('optimization_effectiveness', 0):.1%}")

    except Exception as e:
        print(f"✗ Smart QP failed: {e}")
        import traceback

        traceback.print_exc()
        results['smart'] = {'method': 'Smart QP-Collocation', 'success': False, 'error': str(e)}

    # Test 2: Tuned Smart QP-Collocation
    print(f"\n{'-'*60}")
    print("TESTING TUNED SMART QP-COLLOCATION")
    print(f"{'-'*60}")

    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)
        boundary_indices = [0, num_collocation_points - 1]

        tuned_hjb_solver = TunedSmartQPGFDMHJBSolver(
            problem=problem,
            collocation_points=collocation_points,
            delta=solver_params['delta'],
            taylor_order=solver_params['taylor_order'],
            weight_function="wendland",
            NiterNewton=solver_params['newton_iterations'],
            l2errBoundNewton=solver_params['newton_tolerance'],
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
            qp_usage_target=0.1,
        )

        tuned_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=solver_params['num_particles'],
            delta=solver_params['delta'],
            taylor_order=solver_params['taylor_order'],
            weight_function="wendland",
            NiterNewton=solver_params['newton_iterations'],
            l2errBoundNewton=solver_params['newton_tolerance'],
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
        )
        tuned_solver.hjb_solver = tuned_hjb_solver

        print("Running Tuned Smart QP-Collocation solver...")
        start_time = time.time()
        U_tuned, M_tuned, info_tuned = tuned_solver.solve(
            Niter=solver_params['max_picard_iterations'],
            l2errBound=solver_params['convergence_tolerance'],
            verbose=True,
        )
        tuned_time = time.time() - start_time

        # Calculate mass conservation
        initial_mass = np.sum(M_tuned[0, :]) * Dx
        final_mass = np.sum(M_tuned[-1, :]) * Dx
        mass_error = abs(final_mass - initial_mass) / initial_mass * 100

        # Get tuned optimization statistics
        tuned_stats = tuned_hjb_solver.get_tuned_qp_report()

        results['tuned'] = {
            'method': 'Tuned Smart QP-Collocation',
            'success': True,
            'time': tuned_time,
            'mass_error': mass_error,
            'converged': info_tuned.get('converged', False),
            'iterations': info_tuned.get('iterations', 0),
            'qp_usage_rate': tuned_stats.get('qp_usage_rate', 1.0),
            'tuned_stats': tuned_stats,
            'U': U_tuned,
            'M': M_tuned,
        }

        print(f"✓ Tuned Smart QP completed: {tuned_time:.1f}s, mass error: {mass_error:.2f}%")
        print(f"  QP Usage Rate: {tuned_stats.get('qp_usage_rate', 0):.1%}")
        print(f"  Optimization Quality: {tuned_stats.get('optimization_quality', 'N/A')}")

        # Print detailed tuned summary
        if hasattr(tuned_hjb_solver, 'print_tuned_qp_summary'):
            tuned_hjb_solver.print_tuned_qp_summary()

    except Exception as e:
        print(f"✗ Tuned Smart QP failed: {e}")
        import traceback

        traceback.print_exc()
        results['tuned'] = {'method': 'Tuned Smart QP-Collocation', 'success': False, 'error': str(e)}

    # Print final summary
    print_final_summary(results)

    # Create comparison plots
    create_final_plots(results, problem_params)

    return results


def print_final_summary(results):
    """Print final optimization summary"""
    print(f"\n{'='*80}")
    print("FINAL QP OPTIMIZATION SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Method':<30} {'Success':<8} {'Time(s)':<10} {'Mass Err %':<12} {'QP Usage':<12} {'Status':<15}")
    print("-" * 95)

    for key, result in results.items():
        if result['success']:
            success_str = "✓"
            time_str = f"{result['time']:.1f}"
            mass_str = f"{result['mass_error']:.2f}"
            qp_usage_str = f"{result['qp_usage_rate']:.1%}"

            # Status assessment
            qp_rate = result['qp_usage_rate']
            if qp_rate <= 0.12:  # Within 20% of 10% target
                status = "EXCELLENT"
            elif qp_rate <= 0.2:  # Within 100% of target
                status = "GOOD"
            elif qp_rate <= 0.4:  # Some improvement
                status = "FAIR"
            else:
                status = "POOR"
        else:
            success_str = "✗"
            time_str = "FAILED"
            mass_str = "N/A"
            qp_usage_str = "N/A"
            status = "FAILED"

        print(f"{result['method']:<30} {success_str:<8} {time_str:<10} {mass_str:<12} {qp_usage_str:<12} {status:<15}")

    # Performance analysis
    successful_results = {k: v for k, v in results.items() if v['success']}

    if len(successful_results) > 1:
        print(f"\nPERFORMANCE COMPARISON:")
        print("-" * 50)

        # Find best performers
        fastest = min(successful_results.items(), key=lambda x: x[1]['time'])
        lowest_qp_usage = min(successful_results.items(), key=lambda x: x[1]['qp_usage_rate'])

        print(f"Fastest Method: {fastest[1]['method']} ({fastest[1]['time']:.1f}s)")
        print(f"Lowest QP Usage: {lowest_qp_usage[1]['method']} ({lowest_qp_usage[1]['qp_usage_rate']:.1%})")

        # Speed comparison
        if 'smart' in successful_results and 'tuned' in successful_results:
            smart_time = successful_results['smart']['time']
            tuned_time = successful_results['tuned']['time']
            speedup = smart_time / tuned_time
            print(f"Tuned vs Smart Speedup: {speedup:.2f}x")

    # Overall assessment
    print(f"\nOVERALL ASSESSMENT:")
    print("-" * 30)

    best_result = None
    best_qp_rate = 1.0

    for key, result in successful_results.items():
        if result['qp_usage_rate'] < best_qp_rate:
            best_qp_rate = result['qp_usage_rate']
            best_result = result

    if best_result:
        if best_qp_rate <= 0.12:
            print("✓ OPTIMIZATION TARGET ACHIEVED")
            print(f"  Best optimization: {best_result['method']} with {best_qp_rate:.1%} QP usage")
            print("  QP-Collocation is now production-ready with significant speedup")
        elif best_qp_rate <= 0.2:
            print("⚠️ CLOSE TO OPTIMIZATION TARGET")
            print(f"  Best optimization: {best_result['method']} with {best_qp_rate:.1%} QP usage")
            print("  Good performance, minor tuning could improve further")
        else:
            print("❌ OPTIMIZATION TARGET NOT REACHED")
            print(f"  Best optimization: {best_result['method']} with {best_qp_rate:.1%} QP usage")
            print("  Further optimization needed")


def create_final_plots(results, problem_params):
    """Create comprehensive final plots"""
    successful_results = {k: v for k, v in results.items() if v['success']}

    if len(successful_results) == 0:
        print("No successful results to plot")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: QP Usage Rate Comparison
    methods = [r['method'] for r in successful_results.values()]
    qp_rates = [r['qp_usage_rate'] * 100 for r in successful_results.values()]
    colors = ['blue', 'green'][: len(methods)]

    bars1 = ax1.bar(methods, qp_rates, color=colors, alpha=0.7)
    ax1.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Target: 10%')
    ax1.set_ylabel('QP Usage Rate (%)')
    ax1.set_title('Final QP Usage Rate Comparison')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars1, qp_rates):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0, height + 1, f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold'
        )

    # Plot 2: Solve Time Comparison
    times = [r['time'] for r in successful_results.values()]

    bars2 = ax2.bar(methods, times, color=colors, alpha=0.7)
    ax2.set_ylabel('Solve Time (seconds)')
    ax2.set_title('Computational Time Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Add value labels and speedup
    for i, (bar, time_val) in enumerate(zip(bars2, times)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f'{time_val:.1f}s',
            ha='center',
            va='bottom',
            fontweight='bold',
        )

    if len(times) > 1:
        speedup = times[0] / times[1] if times[1] > 0 else 1.0
        ax2.text(
            0.5,
            0.9,
            f'Speedup: {speedup:.1f}x',
            transform=ax2.transAxes,
            ha='center',
            va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            fontsize=12,
            fontweight='bold',
        )

    # Plot 3: Solution Quality (Final Density Profiles)
    x_grid = np.linspace(
        problem_params['xmin'],
        problem_params['xmax'],
        successful_results[list(successful_results.keys())[0]]['M'].shape[1],
    )

    for i, (key, result) in enumerate(successful_results.items()):
        M = result['M']
        ax3.plot(x_grid, M[-1, :], label=f"{result['method']} (final)", color=colors[i], linewidth=2)
        ax3.plot(
            x_grid,
            M[0, :],
            label=f"{result['method']} (initial)",
            color=colors[i],
            linewidth=1,
            linestyle='--',
            alpha=0.7,
        )

    ax3.set_xlabel('x')
    ax3.set_ylabel('Density')
    ax3.set_title('Solution Quality Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Optimization Summary
    ax4.axis('off')

    # Create summary text
    summary_text = "FINAL OPTIMIZATION RESULTS\n\n"

    for key, result in successful_results.items():
        summary_text += f"{result['method']}:\n"
        summary_text += f"  Time: {result['time']:.1f}s\n"
        summary_text += f"  Mass Error: {result['mass_error']:.2f}%\n"
        summary_text += f"  QP Usage: {result['qp_usage_rate']:.1%}\n"

        # Estimated speedup calculation
        qp_skip_rate = 1.0 - result['qp_usage_rate']
        estimated_speedup = 1 / (1 - qp_skip_rate * 0.9)
        summary_text += f"  Est. Speedup: {estimated_speedup:.1f}x\n\n"

    # Overall conclusion
    best_qp_rate = min([r['qp_usage_rate'] for r in successful_results.values()])

    if best_qp_rate <= 0.12:
        summary_text += "CONCLUSION:\n"
        summary_text += "✓ QP OPTIMIZATION SUCCESSFUL\n"
        summary_text += "Target achieved! QP-Collocation\n"
        summary_text += "is now production-ready with\n"
        summary_text += "significant performance gains."
    elif best_qp_rate <= 0.2:
        summary_text += "CONCLUSION:\n"
        summary_text += "⚠️ NEAR TARGET ACHIEVEMENT\n"
        summary_text += "Close to optimal performance.\n"
        summary_text += "Minor tuning recommended."
    else:
        summary_text += "CONCLUSION:\n"
        summary_text += "❌ FURTHER OPTIMIZATION NEEDED\n"
        summary_text += "Continue with Hybrid method\n"
        summary_text += "until QP optimization improves."

    ax4.text(
        0.05,
        0.95,
        summary_text,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/tests/method_comparisons/final_optimization_results.png',
        dpi=300,
        bbox_inches='tight',
    )
    print(f"\nFinal optimization results saved to: final_optimization_results.png")
    plt.show()


def main():
    """Run the final optimization test"""
    print("Starting Final QP Optimization Test...")
    print("This validates the complete optimization framework")
    print("Expected execution time: 10-20 minutes")

    try:
        results = final_optimization_test()

        print(f"\n{'='*80}")
        print("FINAL QP OPTIMIZATION TEST COMPLETED")
        print(f"{'='*80}")
        print("Check the summary above and generated plots for final results.")

        return results

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        return None
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
