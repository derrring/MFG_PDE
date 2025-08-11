#!/usr/bin/env python3
"""
Smart QP Validation Test
Test the Smart QP GFDM HJB Solver to validate it achieves ~10% QP usage rate
while maintaining solution quality.
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

from mfg_pde.alg.hjb_solvers.optimized_gfdm_hjb_v2 import OptimizedGFDMHJBSolver
from mfg_pde.alg.hjb_solvers.smart_qp_gfdm_hjb import SmartQPGFDMHJBSolver
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def compare_qp_optimizations():
    """Compare the old vs new QP optimization approaches"""
    print("=" * 80)
    print("SMART QP OPTIMIZATION VALIDATION TEST")
    print("=" * 80)
    print("Comparing Basic vs Smart QP optimization approaches")

    # Problem parameters - same as previous tests
    problem_params = {'xmin': 0.0, 'xmax': 1.0, 'Nx': 20, 'T': 1.0, 'Nt': 40, 'sigma': 0.15, 'coefCT': 0.02}

    print(f"Problem: Nx={problem_params['Nx']}, T={problem_params['T']}, Nt={problem_params['Nt']}")
    print(f"Parameters: σ={problem_params['sigma']}, coefCT={problem_params['coefCT']}")

    results = {}

    # Test 1: Basic Optimized QP-Collocation (previous version)
    print(f"\n{'-'*60}")
    print("TESTING BASIC OPTIMIZED QP-COLLOCATION")
    print(f"{'-'*60}")

    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        # Setup collocation points
        num_collocation_points = 10
        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)

        boundary_indices = [0, num_collocation_points - 1]

        # Create basic optimized HJB solver
        basic_optimized_hjb_solver = OptimizedGFDMHJBSolver(
            problem=problem,
            collocation_points=collocation_points,
            delta=0.4,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=4,
            l2errBoundNewton=1e-3,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
            qp_activation_tolerance=1e-3,
        )

        # Create particle collocation solver
        basic_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=150,
            delta=0.4,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=4,
            l2errBoundNewton=1e-3,
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
        )

        # Replace HJB solver with basic optimized version
        basic_solver.hjb_solver = basic_optimized_hjb_solver

        print("Running Basic Optimized QP-Collocation solver...")
        start_time = time.time()
        U_basic, M_basic, info_basic = basic_solver.solve(Niter=6, l2errBound=1e-3, verbose=True)
        basic_time = time.time() - start_time

        # Calculate mass conservation
        Dx = (problem_params['xmax'] - problem_params['xmin']) / problem_params['Nx']
        initial_mass_basic = np.sum(M_basic[0, :]) * Dx
        final_mass_basic = np.sum(M_basic[-1, :]) * Dx
        mass_error_basic = abs(final_mass_basic - initial_mass_basic) / initial_mass_basic * 100

        # Get basic optimization statistics
        basic_stats = {}
        if hasattr(basic_optimized_hjb_solver, 'get_performance_report'):
            basic_stats = basic_optimized_hjb_solver.get_performance_report()

        results['basic'] = {
            'method': 'Basic Optimized QP-Collocation',
            'success': True,
            'time': basic_time,
            'mass_error': mass_error_basic,
            'converged': info_basic.get('converged', False),
            'iterations': info_basic.get('iterations', 0),
            'U': U_basic,
            'M': M_basic,
            'optimization_stats': basic_stats,
        }

        print(f"✓ Basic Optimized completed: {basic_time:.1f}s, mass error: {mass_error_basic:.2f}%")
        if basic_stats:
            print(f"  QP Activation Rate: {basic_stats.get('qp_activation_rate', 0):.1%}")
            print(f"  QP Skip Rate: {basic_stats.get('qp_skip_rate', 0):.1%}")

    except Exception as e:
        print(f"✗ Basic Optimized failed: {e}")
        results['basic'] = {'method': 'Basic Optimized QP-Collocation', 'success': False, 'error': str(e)}

    # Test 2: Smart QP-Collocation (new version)
    print(f"\n{'-'*60}")
    print("TESTING SMART QP-COLLOCATION")
    print(f"{'-'*60}")

    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        # Setup collocation points (same as basic)
        num_collocation_points = 10
        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)

        boundary_indices = [0, num_collocation_points - 1]

        # Create smart QP HJB solver
        smart_qp_hjb_solver = SmartQPGFDMHJBSolver(
            problem=problem,
            collocation_points=collocation_points,
            delta=0.4,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=4,
            l2errBoundNewton=1e-3,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
            qp_usage_target=0.1,  # Target 10% QP usage
        )

        # Create particle collocation solver
        smart_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=150,
            delta=0.4,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=4,
            l2errBoundNewton=1e-3,
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
        )

        # Replace HJB solver with smart version
        smart_solver.hjb_solver = smart_qp_hjb_solver

        print("Running Smart QP-Collocation solver...")
        start_time = time.time()
        U_smart, M_smart, info_smart = smart_solver.solve(Niter=6, l2errBound=1e-3, verbose=True)
        smart_time = time.time() - start_time

        # Calculate mass conservation
        initial_mass_smart = np.sum(M_smart[0, :]) * Dx
        final_mass_smart = np.sum(M_smart[-1, :]) * Dx
        mass_error_smart = abs(final_mass_smart - initial_mass_smart) / initial_mass_smart * 100

        # Get smart optimization statistics
        smart_stats = {}
        if hasattr(smart_qp_hjb_solver, 'get_smart_qp_report'):
            smart_stats = smart_qp_hjb_solver.get_smart_qp_report()

        results['smart'] = {
            'method': 'Smart QP-Collocation',
            'success': True,
            'time': smart_time,
            'mass_error': mass_error_smart,
            'converged': info_smart.get('converged', False),
            'iterations': info_smart.get('iterations', 0),
            'U': U_smart,
            'M': M_smart,
            'smart_stats': smart_stats,
        }

        print(f"✓ Smart QP completed: {smart_time:.1f}s, mass error: {mass_error_smart:.2f}%")
        if smart_stats:
            print(f"  QP Usage Rate: {smart_stats.get('qp_usage_rate', 0):.1%}")
            print(f"  QP Skip Rate: {smart_stats.get('qp_skip_rate', 0):.1%}")
            print(f"  Optimization Effectiveness: {smart_stats.get('optimization_effectiveness', 0):.1%}")

        # Print detailed smart QP summary
        if hasattr(smart_qp_hjb_solver, 'print_smart_qp_summary'):
            smart_qp_hjb_solver.print_smart_qp_summary()

    except Exception as e:
        print(f"✗ Smart QP failed: {e}")
        import traceback

        traceback.print_exc()
        results['smart'] = {'method': 'Smart QP-Collocation', 'success': False, 'error': str(e)}

    # Comparison Summary
    print(f"\n{'='*80}")
    print("QP OPTIMIZATION COMPARISON SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Method':<30} {'Success':<8} {'Time(s)':<10} {'Mass Error %':<12} {'QP Usage':<12} {'Status':<15}")
    print("-" * 95)

    for key, result in results.items():
        if result['success']:
            success_str = "✓"
            time_str = f"{result['time']:.1f}"
            mass_str = f"{result['mass_error']:.2f}"

            # Extract QP usage rate
            if key == 'basic' and 'optimization_stats' in result:
                qp_usage = f"{result['optimization_stats'].get('qp_activation_rate', 0):.1%}"
            elif key == 'smart' and 'smart_stats' in result:
                qp_usage = f"{result['smart_stats'].get('qp_usage_rate', 0):.1%}"
            else:
                qp_usage = "N/A"

            # Status assessment
            if key == 'smart' and 'smart_stats' in result:
                qp_rate = result['smart_stats'].get('qp_usage_rate', 1.0)
                if qp_rate <= 0.15:  # Within 50% of 10% target
                    status = "EXCELLENT"
                elif qp_rate <= 0.25:  # Within 150% of target
                    status = "GOOD"
                elif qp_rate <= 0.5:  # Some improvement
                    status = "FAIR"
                else:
                    status = "POOR"
            else:
                status = "BASELINE"
        else:
            success_str = "✗"
            time_str = "N/A"
            mass_str = "N/A"
            qp_usage = "N/A"
            status = "FAILED"

        print(f"{result['method']:<30} {success_str:<8} {time_str:<10} {mass_str:<12} {qp_usage:<12} {status:<15}")

    # Performance Analysis
    successful_results = {k: v for k, v in results.items() if v['success']}

    if len(successful_results) > 1:
        print(f"\nPERFORMANCE ANALYSIS:")
        print("-" * 50)

        # Speedup comparison
        if 'basic' in successful_results and 'smart' in successful_results:
            basic_time = successful_results['basic']['time']
            smart_time = successful_results['smart']['time']
            speedup = basic_time / smart_time
            print(f"Smart QP Speedup vs Basic: {speedup:.2f}x")

        # QP usage comparison
        if 'basic' in results and results['basic']['success'] and 'optimization_stats' in results['basic']:
            basic_qp_rate = results['basic']['optimization_stats'].get('qp_activation_rate', 1.0)
            print(f"Basic QP Usage Rate: {basic_qp_rate:.1%}")

        if 'smart' in results and results['smart']['success'] and 'smart_stats' in results['smart']:
            smart_qp_rate = results['smart']['smart_stats'].get('qp_usage_rate', 1.0)
            print(f"Smart QP Usage Rate: {smart_qp_rate:.1%}")

            if 'basic' in results and results['basic']['success']:
                qp_reduction = (1.0 - smart_qp_rate / basic_qp_rate) * 100
                print(f"QP Usage Reduction: {qp_reduction:.1f}%")

        # Solution quality comparison
        mass_errors = [r['mass_error'] for r in successful_results.values()]
        if len(mass_errors) > 1:
            max_mass_diff = max(mass_errors) - min(mass_errors)
            print(f"Mass Conservation Difference: {max_mass_diff:.2f}% (should be small)")

    # Create comparison plots
    create_optimization_comparison_plots(results, problem_params)

    return results


def create_optimization_comparison_plots(results, problem_params):
    """Create plots comparing the optimization approaches"""
    successful_results = {k: v for k, v in results.items() if v['success']}

    if len(successful_results) == 0:
        print("No successful results to plot")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: QP Usage Rate Comparison
    methods = [r['method'] for r in successful_results.values()]
    qp_rates = []

    for key, result in successful_results.items():
        if key == 'basic' and 'optimization_stats' in result:
            qp_rates.append(result['optimization_stats'].get('qp_activation_rate', 0) * 100)
        elif key == 'smart' and 'smart_stats' in result:
            qp_rates.append(result['smart_stats'].get('qp_usage_rate', 0) * 100)
        else:
            qp_rates.append(100)  # Assume 100% if no stats

    colors = ['orange', 'green'][: len(methods)]
    bars1 = ax1.bar(methods, qp_rates, color=colors, alpha=0.7)

    # Add target line
    ax1.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Target: 10%')

    ax1.set_ylabel('QP Usage Rate (%)')
    ax1.set_title('QP Usage Rate Comparison')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars1, qp_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 2, f'{rate:.1f}%', ha='center', va='bottom')

    # Plot 2: Solve Time Comparison
    times = [r['time'] for r in successful_results.values()]

    bars2 = ax2.bar(methods, times, color=colors, alpha=0.7)
    ax2.set_ylabel('Solve Time (seconds)')
    ax2.set_title('Computational Time Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for bar, time_val in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f'{time_val:.1f}s', ha='center', va='bottom')

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

    # Plot 4: Optimization Effectiveness Summary
    ax4.axis('off')

    # Create effectiveness summary text
    summary_text = "OPTIMIZATION EFFECTIVENESS SUMMARY\n\n"

    for key, result in successful_results.items():
        summary_text += f"{result['method']}:\n"
        summary_text += f"  Time: {result['time']:.1f}s\n"
        summary_text += f"  Mass Error: {result['mass_error']:.2f}%\n"

        if key == 'basic' and 'optimization_stats' in result:
            stats = result['optimization_stats']
            summary_text += f"  QP Rate: {stats.get('qp_activation_rate', 0):.1%}\n"
        elif key == 'smart' and 'smart_stats' in result:
            stats = result['smart_stats']
            summary_text += f"  QP Rate: {stats.get('qp_usage_rate', 0):.1%}\n"
            summary_text += f"  Effectiveness: {stats.get('optimization_effectiveness', 0):.1%}\n"

        summary_text += "\n"

    # Calculate overall assessment
    if len(successful_results) > 1 and 'smart' in successful_results:
        smart_stats = successful_results['smart'].get('smart_stats', {})
        qp_rate = smart_stats.get('qp_usage_rate', 1.0)

        if qp_rate <= 0.15:
            summary_text += "OVERALL ASSESSMENT: ✓ EXCELLENT\n"
            summary_text += "Smart QP optimization working as intended.\n"
        elif qp_rate <= 0.25:
            summary_text += "OVERALL ASSESSMENT: ✓ GOOD\n"
            summary_text += "Smart QP showing significant improvement.\n"
        elif qp_rate <= 0.5:
            summary_text += "OVERALL ASSESSMENT: ⚠️ FAIR\n"
            summary_text += "Some optimization achieved, needs tuning.\n"
        else:
            summary_text += "OVERALL ASSESSMENT: ❌ POOR\n"
            summary_text += "Optimization not effective, needs debugging.\n"

    ax4.text(
        0.05,
        0.95,
        summary_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/tests/method_comparisons/smart_qp_validation.png',
        dpi=300,
        bbox_inches='tight',
    )
    print(f"\nSmart QP validation plots saved to: smart_qp_validation.png")
    plt.show()


def main():
    """Run the smart QP validation test"""
    print("Starting Smart QP Validation Test...")
    print("This validates that Smart QP optimization achieves ~10% QP usage")
    print("Expected execution time: 5-15 minutes")

    try:
        results = compare_qp_optimizations()

        print(f"\n{'='*80}")
        print("SMART QP VALIDATION TEST COMPLETED")
        print(f"{'='*80}")
        print("Check the summary above and generated plots for validation results.")

        return results

    except KeyboardInterrupt:
        print("\nValidation interrupted by user.")
        return None
    except Exception as e:
        print(f"\nValidation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
