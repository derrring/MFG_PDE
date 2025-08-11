#!/usr/bin/env python3
"""
Robust Three-Method Comparison
Comprehensive statistical analysis of FDM, Hybrid, and Optimized QP-Collocation
across various test cases to assess success rate and robustness.
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

from mfg_pde.alg.damped_fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.fp_solvers.fdm_fp import FdmFPSolver
from mfg_pde.alg.fp_solvers.particle_fp import ParticleFPSolver
from mfg_pde.alg.hjb_solvers.fdm_hjb import FdmHJBSolver
from mfg_pde.alg.hjb_solvers.tuned_smart_qp_gfdm_hjb import TunedSmartQPGFDMHJBSolver
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def generate_test_cases():
    """Generate diverse test cases for robustness analysis"""
    test_cases = []

    # Case 1: Small scale problems
    for i, (sigma, coefCT) in enumerate([(0.08, 0.01), (0.1, 0.015), (0.12, 0.02)]):
        test_cases.append(
            {
                'name': f'Small-{i+1}',
                'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 15, 'T': 0.5, 'Nt': 20, 'sigma': sigma, 'coefCT': coefCT},
                'category': 'Small Scale',
                'expected_difficulty': 'Easy',
            }
        )

    # Case 2: Medium scale problems
    for i, (sigma, coefCT, T) in enumerate([(0.1, 0.02, 0.8), (0.12, 0.025, 1.0), (0.15, 0.03, 1.2)]):
        test_cases.append(
            {
                'name': f'Medium-{i+1}',
                'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 18, 'T': T, 'Nt': 25, 'sigma': sigma, 'coefCT': coefCT},
                'category': 'Medium Scale',
                'expected_difficulty': 'Moderate',
            }
        )

    # Case 3: Large scale problems
    for i, (sigma, coefCT, Nx) in enumerate([(0.12, 0.02, 22), (0.15, 0.025, 25), (0.18, 0.03, 28)]):
        test_cases.append(
            {
                'name': f'Large-{i+1}',
                'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': Nx, 'T': 1.0, 'Nt': 30, 'sigma': sigma, 'coefCT': coefCT},
                'category': 'Large Scale',
                'expected_difficulty': 'Hard',
            }
        )

    # Case 4: High volatility problems
    for i, sigma in enumerate([0.2, 0.25, 0.3]):
        test_cases.append(
            {
                'name': f'HighVol-{i+1}',
                'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 20, 'T': 0.8, 'Nt': 25, 'sigma': sigma, 'coefCT': 0.02},
                'category': 'High Volatility',
                'expected_difficulty': 'Hard',
            }
        )

    # Case 5: Long time horizon problems
    for i, T in enumerate([1.5, 2.0, 2.5]):
        test_cases.append(
            {
                'name': f'LongTime-{i+1}',
                'params': {
                    'xmin': 0.0,
                    'xmax': 1.0,
                    'Nx': 18,
                    'T': T,
                    'Nt': int(T * 20),
                    'sigma': 0.12,
                    'coefCT': 0.02,
                },
                'category': 'Long Time Horizon',
                'expected_difficulty': 'Hard',
            }
        )

    return test_cases


def test_pure_fdm(problem_params, test_name, timeout=300):
    """Test Pure FDM method with timeout"""
    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        fdm_hjb_solver = FdmHJBSolver(problem=problem)
        fdm_fp_solver = FdmFPSolver(problem=problem, boundary_conditions=no_flux_bc)
        fdm_solver = FixedPointIterator(
            problem=problem, hjb_solver=fdm_hjb_solver, fp_solver=fdm_fp_solver, thetaUM=0.5
        )

        start_time = time.time()
        U_fdm, M_fdm, iterations_run, l2distu_rel, l2distm_rel = fdm_solver.solve(Niter_max=6, l2errBoundPicard=1e-3)
        solve_time = time.time() - start_time

        # Create info dict for compatibility
        info_fdm = {
            'converged': l2distu_rel[-1] < 1e-3 if len(l2distu_rel) > 0 else False,
            'iterations': iterations_run,
        }

        if solve_time > timeout:
            return {'success': False, 'error': 'Timeout', 'time': solve_time}

        # Calculate mass conservation
        Dx = (problem_params['xmax'] - problem_params['xmin']) / problem_params['Nx']
        initial_mass = np.sum(M_fdm[0, :]) * Dx
        final_mass = np.sum(M_fdm[-1, :]) * Dx
        mass_error = abs(final_mass - initial_mass) / initial_mass * 100

        return {
            'success': True,
            'time': solve_time,
            'mass_error': mass_error,
            'converged': info_fdm.get('converged', False),
            'iterations': info_fdm.get('iterations', 0),
            'method_specific': {'type': 'Grid-based', 'complexity': 'Low'},
        }

    except Exception as e:
        return {'success': False, 'error': str(e), 'time': 0}


def test_hybrid_particle_fdm(problem_params, test_name, timeout=300):
    """Test Hybrid Particle-FDM method with timeout"""
    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        num_particles = min(100, max(50, problem_params['Nx'] * 4))

        hybrid_hjb_solver = FdmHJBSolver(problem=problem)
        hybrid_fp_solver = ParticleFPSolver(
            problem=problem, num_particles=num_particles, boundary_conditions=no_flux_bc
        )
        hybrid_solver = FixedPointIterator(
            problem=problem, hjb_solver=hybrid_hjb_solver, fp_solver=hybrid_fp_solver, thetaUM=0.5
        )

        start_time = time.time()
        U_hybrid, M_hybrid, iterations_run, l2distu_rel, l2distm_rel = hybrid_solver.solve(
            Niter_max=6, l2errBoundPicard=1e-3
        )
        solve_time = time.time() - start_time

        # Create info dict for compatibility
        info_hybrid = {
            'converged': l2distu_rel[-1] < 1e-3 if len(l2distu_rel) > 0 else False,
            'iterations': iterations_run,
        }

        if solve_time > timeout:
            return {'success': False, 'error': 'Timeout', 'time': solve_time}

        # Calculate mass conservation
        Dx = (problem_params['xmax'] - problem_params['xmin']) / problem_params['Nx']
        initial_mass = np.sum(M_hybrid[0, :]) * Dx
        final_mass = np.sum(M_hybrid[-1, :]) * Dx
        mass_error = abs(final_mass - initial_mass) / initial_mass * 100

        return {
            'success': True,
            'time': solve_time,
            'mass_error': mass_error,
            'converged': info_hybrid.get('converged', False),
            'iterations': info_hybrid.get('iterations', 0),
            'method_specific': {'type': 'Hybrid', 'particles': num_particles, 'complexity': 'Medium'},
        }

    except Exception as e:
        return {'success': False, 'error': str(e), 'time': 0}


def test_optimized_qp_collocation(problem_params, test_name, timeout=600):
    """Test Optimized QP-Collocation method with timeout"""
    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        # Adaptive parameters based on problem size
        num_collocation_points = min(12, max(6, problem_params['Nx'] // 2))
        num_particles = min(150, max(80, problem_params['Nx'] * 5))

        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)
        boundary_indices = [0, num_collocation_points - 1]

        tuned_hjb_solver = TunedSmartQPGFDMHJBSolver(
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
            qp_usage_target=0.1,
        )

        qp_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=num_particles,
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
        qp_solver.hjb_solver = tuned_hjb_solver

        start_time = time.time()
        U_qp, M_qp, info_qp = qp_solver.solve(Niter=6, l2errBound=1e-3, verbose=False)
        solve_time = time.time() - start_time

        if solve_time > timeout:
            return {'success': False, 'error': 'Timeout', 'time': solve_time}

        # Calculate mass conservation
        Dx = (problem_params['xmax'] - problem_params['xmin']) / problem_params['Nx']
        initial_mass = np.sum(M_qp[0, :]) * Dx
        final_mass = np.sum(M_qp[-1, :]) * Dx
        mass_error = abs(final_mass - initial_mass) / initial_mass * 100

        # Get QP optimization statistics
        qp_stats = tuned_hjb_solver.get_tuned_qp_report()
        qp_usage_rate = qp_stats.get('qp_usage_rate', 1.0)

        return {
            'success': True,
            'time': solve_time,
            'mass_error': mass_error,
            'converged': info_qp.get('converged', False),
            'iterations': info_qp.get('iterations', 0),
            'method_specific': {
                'type': 'Advanced Collocation',
                'particles': num_particles,
                'collocation_points': num_collocation_points,
                'qp_usage_rate': qp_usage_rate,
                'optimization_quality': qp_stats.get('optimization_quality', 'N/A'),
                'complexity': 'High',
            },
        }

    except Exception as e:
        return {'success': False, 'error': str(e), 'time': 0}


def run_comprehensive_robustness_test():
    """Run comprehensive robustness test across all methods and test cases"""
    print("=" * 100)
    print("COMPREHENSIVE THREE-METHOD ROBUSTNESS ANALYSIS")
    print("=" * 100)
    print("Testing Pure FDM, Hybrid Particle-FDM, and Optimized QP-Collocation")
    print("Across diverse problem configurations for statistical analysis")

    test_cases = generate_test_cases()
    print(f"\nGenerated {len(test_cases)} test cases across 5 categories:")

    categories = {}
    for case in test_cases:
        cat = case['category']
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in categories.items():
        print(f"  • {cat}: {count} cases")

    results = {'test_cases': test_cases, 'fdm_results': [], 'hybrid_results': [], 'qp_results': []}

    total_tests = len(test_cases) * 3
    current_test = 0

    print(f"\n{'='*80}")
    print("RUNNING ROBUSTNESS TESTS")
    print(f"{'='*80}")

    for i, test_case in enumerate(test_cases):
        case_name = test_case['name']
        params = test_case['params']
        category = test_case['category']

        print(f"\n{'-'*60}")
        print(f"TEST CASE {i+1}/{len(test_cases)}: {case_name} ({category})")
        print(f"Parameters: Nx={params['Nx']}, T={params['T']}, σ={params['sigma']}, coefCT={params['coefCT']}")
        print(f"{'-'*60}")

        # Test 1: Pure FDM
        print(f"[{current_test+1:2d}/{total_tests}] Testing Pure FDM...")
        fdm_result = test_pure_fdm(params, case_name)
        fdm_result['test_case'] = case_name
        fdm_result['category'] = category
        results['fdm_results'].append(fdm_result)
        current_test += 1

        if fdm_result['success']:
            print(f"    ✓ Pure FDM: {fdm_result['time']:.1f}s, mass error: {fdm_result['mass_error']:.3f}%")
        else:
            print(f"    ✗ Pure FDM failed: {fdm_result['error']}")

        # Test 2: Hybrid Particle-FDM
        print(f"[{current_test+1:2d}/{total_tests}] Testing Hybrid Particle-FDM...")
        hybrid_result = test_hybrid_particle_fdm(params, case_name)
        hybrid_result['test_case'] = case_name
        hybrid_result['category'] = category
        results['hybrid_results'].append(hybrid_result)
        current_test += 1

        if hybrid_result['success']:
            particles = hybrid_result['method_specific']['particles']
            print(
                f"    ✓ Hybrid P-FDM: {hybrid_result['time']:.1f}s, mass error: {hybrid_result['mass_error']:.3f}%, particles: {particles}"
            )
        else:
            print(f"    ✗ Hybrid P-FDM failed: {hybrid_result['error']}")

        # Test 3: Optimized QP-Collocation
        print(f"[{current_test+1:2d}/{total_tests}] Testing Optimized QP-Collocation...")
        qp_result = test_optimized_qp_collocation(params, case_name)
        qp_result['test_case'] = case_name
        qp_result['category'] = category
        results['qp_results'].append(qp_result)
        current_test += 1

        if qp_result['success']:
            qp_usage = qp_result['method_specific']['qp_usage_rate']
            print(
                f"    ✓ QP-Collocation: {qp_result['time']:.1f}s, mass error: {qp_result['mass_error']:.3f}%, QP usage: {qp_usage:.1%}"
            )
        else:
            print(f"    ✗ QP-Collocation failed: {qp_result['error']}")

    # Generate comprehensive statistical analysis
    print(f"\n{'='*100}")
    print("STATISTICAL ANALYSIS COMPLETED")
    print(f"{'='*100}")

    analyze_results_and_create_plots(results)
    return results


def analyze_results_and_create_plots(results):
    """Analyze results and create comprehensive statistical plots"""

    # Extract success rates by method and category
    methods = ['Pure FDM', 'Hybrid P-FDM', 'QP-Collocation']
    method_results = [results['fdm_results'], results['hybrid_results'], results['qp_results']]

    # Calculate overall statistics
    print("\nOVERALL ROBUSTNESS STATISTICS:")
    print("=" * 50)

    overall_stats = {}
    for method_name, method_result in zip(methods, method_results):
        total_tests = len(method_result)
        successful_tests = sum(1 for r in method_result if r['success'])
        success_rate = successful_tests / total_tests * 100

        successful_results = [r for r in method_result if r['success']]
        if successful_results:
            avg_time = np.mean([r['time'] for r in successful_results])
            avg_mass_error = np.mean([r['mass_error'] for r in successful_results])
            std_time = np.std([r['time'] for r in successful_results])
            std_mass_error = np.std([r['mass_error'] for r in successful_results])
        else:
            avg_time = avg_mass_error = std_time = std_mass_error = 0

        overall_stats[method_name] = {
            'success_rate': success_rate,
            'successful_tests': successful_tests,
            'total_tests': total_tests,
            'avg_time': avg_time,
            'std_time': std_time,
            'avg_mass_error': avg_mass_error,
            'std_mass_error': std_mass_error,
        }

        print(f"\n{method_name}:")
        print(f"  Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        if successful_results:
            print(f"  Avg Time: {avg_time:.1f} ± {std_time:.1f} seconds")
            print(f"  Avg Mass Error: {avg_mass_error:.3f} ± {std_mass_error:.3f}%")

    # Category-wise analysis
    categories = list(set(r['category'] for r in results['test_cases']))
    print(f"\nCATEGORY-WISE ANALYSIS:")
    print("=" * 50)

    category_stats = {}
    for category in categories:
        category_stats[category] = {}

        print(f"\n{category}:")
        for method_name, method_result in zip(methods, method_results):
            cat_results = [r for r in method_result if r['category'] == category]
            successful_cat = [r for r in cat_results if r['success']]

            success_rate = len(successful_cat) / len(cat_results) * 100 if cat_results else 0
            avg_time = np.mean([r['time'] for r in successful_cat]) if successful_cat else 0

            category_stats[category][method_name] = {
                'success_rate': success_rate,
                'avg_time': avg_time,
                'successful': len(successful_cat),
                'total': len(cat_results),
            }

            print(f"  {method_name}: {success_rate:.1f}% ({len(successful_cat)}/{len(cat_results)})")
            if successful_cat:
                print(f"    Avg Time: {avg_time:.1f}s")

    # Create comprehensive statistical plots
    create_comprehensive_statistical_plots(overall_stats, category_stats, results)


def create_comprehensive_statistical_plots(overall_stats, category_stats, results):
    """Create comprehensive statistical visualization"""

    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))

    methods = list(overall_stats.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue

    # Plot 1: Overall Success Rates
    ax1 = plt.subplot(3, 4, 1)
    success_rates = [overall_stats[method]['success_rate'] for method in methods]
    bars1 = ax1.bar(methods, success_rates, color=colors, alpha=0.8)
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Overall Success Rate Comparison')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars1, success_rates):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0, height + 2, f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold'
        )

    # Plot 2: Average Solution Time (for successful cases)
    ax2 = plt.subplot(3, 4, 2)
    avg_times = [overall_stats[method]['avg_time'] for method in methods]
    std_times = [overall_stats[method]['std_time'] for method in methods]

    bars2 = ax2.bar(methods, avg_times, yerr=std_times, color=colors, alpha=0.8, capsize=5)
    ax2.set_ylabel('Average Time (seconds)')
    ax2.set_title('Average Solution Time\n(Successful Cases Only)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Average Mass Conservation Error
    ax3 = plt.subplot(3, 4, 3)
    avg_mass_errors = [overall_stats[method]['avg_mass_error'] for method in methods]
    std_mass_errors = [overall_stats[method]['std_mass_error'] for method in methods]

    bars3 = ax3.bar(methods, avg_mass_errors, yerr=std_mass_errors, color=colors, alpha=0.8, capsize=5)
    ax3.set_ylabel('Mass Conservation Error (%)')
    ax3.set_title('Average Mass Conservation Error\n(Successful Cases Only)')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Robustness Score
    ax4 = plt.subplot(3, 4, 4)

    # Calculate robustness score (success_rate * accuracy_factor * speed_factor)
    robustness_scores = []
    for method in methods:
        stats = overall_stats[method]
        success_factor = stats['success_rate'] / 100

        if stats['avg_time'] > 0:
            speed_factor = min(1.0, 30 / stats['avg_time'])  # Normalize to 30s baseline
        else:
            speed_factor = 0

        if stats['avg_mass_error'] > 0:
            accuracy_factor = min(1.0, 1.0 / stats['avg_mass_error'])  # Better accuracy = higher score
        else:
            accuracy_factor = 1.0

        robustness_score = success_factor * 0.6 + speed_factor * 0.2 + accuracy_factor * 0.2
        robustness_scores.append(robustness_score * 100)

    bars4 = ax4.bar(methods, robustness_scores, color=colors, alpha=0.8)
    ax4.set_ylabel('Robustness Score')
    ax4.set_title('Overall Robustness Score\n(Success×60% + Speed×20% + Accuracy×20%)')
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3)

    for bar, score in zip(bars4, robustness_scores):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0, height + 2, f'{score:.1f}', ha='center', va='bottom', fontweight='bold'
        )

    # Plot 5-8: Category-wise Success Rates
    categories = list(category_stats.keys())
    for i, category in enumerate(categories[:4]):  # Show first 4 categories
        ax = plt.subplot(3, 4, 5 + i)

        cat_success_rates = [category_stats[category][method]['success_rate'] for method in methods]
        bars = ax.bar(methods, cat_success_rates, color=colors, alpha=0.8)
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(f'{category}\nSuccess Rates')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        for bar, rate in zip(bars, cat_success_rates):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 2,
                    f'{rate:.0f}%',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )

    # Plot 9: Time vs Accuracy Scatter
    ax9 = plt.subplot(3, 4, 9)

    for i, (method_name, method_results) in enumerate(
        zip(methods, [results['fdm_results'], results['hybrid_results'], results['qp_results']])
    ):
        successful_results = [r for r in method_results if r['success']]
        if successful_results:
            times = [r['time'] for r in successful_results]
            mass_errors = [r['mass_error'] for r in successful_results]
            ax9.scatter(times, mass_errors, color=colors[i], alpha=0.7, s=60, label=method_name)

    ax9.set_xlabel('Solution Time (seconds)')
    ax9.set_ylabel('Mass Conservation Error (%)')
    ax9.set_title('Time vs Accuracy Trade-off')
    ax9.set_yscale('log')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # Plot 10: QP Optimization Analysis
    ax10 = plt.subplot(3, 4, 10)

    qp_successful = [r for r in results['qp_results'] if r['success']]
    if qp_successful:
        qp_usage_rates = [
            r['method_specific']['qp_usage_rate'] * 100
            for r in qp_successful
            if 'qp_usage_rate' in r['method_specific']
        ]

        if qp_usage_rates:
            ax10.hist(qp_usage_rates, bins=10, color='#45B7D1', alpha=0.7, edgecolor='black')
            ax10.axvline(x=10, color='red', linestyle='--', linewidth=2, label='Target: 10%')
            ax10.set_xlabel('QP Usage Rate (%)')
            ax10.set_ylabel('Frequency')
            ax10.set_title('QP Usage Rate Distribution\n(Successful QP-Collocation Cases)')
            ax10.legend()
            ax10.grid(True, alpha=0.3)
        else:
            ax10.text(0.5, 0.5, 'No QP usage data\navailable', ha='center', va='center', transform=ax10.transAxes)
            ax10.set_title('QP Usage Analysis')

    # Plot 11: Failure Analysis
    ax11 = plt.subplot(3, 4, 11)

    failure_reasons = {}
    for method_name, method_results in zip(
        methods, [results['fdm_results'], results['hybrid_results'], results['qp_results']]
    ):
        failed_results = [r for r in method_results if not r['success']]
        method_failures = {}
        for failed in failed_results:
            error = failed.get('error', 'Unknown')
            # Simplify error messages
            if 'timeout' in error.lower() or 'Timeout' in error:
                error = 'Timeout'
            elif 'convergence' in error.lower():
                error = 'Convergence'
            elif 'memory' in error.lower():
                error = 'Memory'
            else:
                error = 'Other'
            method_failures[error] = method_failures.get(error, 0) + 1
        failure_reasons[method_name] = method_failures

    # Create stacked bar chart for failure reasons
    failure_types = list(
        set(reason for method_failures in failure_reasons.values() for reason in method_failures.keys())
    )

    if failure_types:
        bottom = np.zeros(len(methods))
        failure_colors = plt.cm.Set3(np.linspace(0, 1, len(failure_types)))

        for i, failure_type in enumerate(failure_types):
            counts = [failure_reasons[method].get(failure_type, 0) for method in methods]
            ax11.bar(methods, counts, bottom=bottom, label=failure_type, color=failure_colors[i], alpha=0.8)
            bottom += counts

        ax11.set_ylabel('Number of Failures')
        ax11.set_title('Failure Analysis by Type')
        ax11.legend()
    else:
        ax11.text(0.5, 0.5, 'No failures\nto analyze', ha='center', va='center', transform=ax11.transAxes)
        ax11.set_title('Failure Analysis')

    ax11.grid(True, alpha=0.3)

    # Plot 12: Summary Statistics Table
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')

    # Create summary table
    summary_text = "ROBUSTNESS SUMMARY\n\n"

    # Find best performing method
    best_success_rate = max(overall_stats[method]['success_rate'] for method in methods)
    best_method = [method for method in methods if overall_stats[method]['success_rate'] == best_success_rate][0]

    summary_text += f"🏆 MOST ROBUST: {best_method}\n"
    summary_text += f"   Success Rate: {best_success_rate:.1f}%\n\n"

    # QP optimization assessment
    qp_stats = overall_stats.get('QP-Collocation', {})
    if qp_stats.get('success_rate', 0) > 0:
        qp_successful = [r for r in results['qp_results'] if r['success']]
        if qp_successful:
            avg_qp_usage = np.mean(
                [
                    r['method_specific']['qp_usage_rate'] * 100
                    for r in qp_successful
                    if 'qp_usage_rate' in r['method_specific']
                ]
            )

            summary_text += f"🎯 QP OPTIMIZATION:\n"
            if avg_qp_usage <= 15:
                summary_text += f"   ✅ SUCCESS ({avg_qp_usage:.1f}% avg usage)\n"
            else:
                summary_text += f"   ⚠️ PARTIAL ({avg_qp_usage:.1f}% avg usage)\n"
        summary_text += "\n"

    # Overall assessment
    if best_success_rate >= 80:
        summary_text += "✅ HIGH ROBUSTNESS\n"
        summary_text += "   Methods are production-ready\n"
    elif best_success_rate >= 60:
        summary_text += "⚠️ MODERATE ROBUSTNESS\n"
        summary_text += "   Suitable for research use\n"
    else:
        summary_text += "❌ LOW ROBUSTNESS\n"
        summary_text += "   Needs improvement\n"

    ax12.text(
        0.05,
        0.95,
        summary_text,
        transform=ax12.transAxes,
        fontsize=11,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
    )

    plt.tight_layout()

    # Save comprehensive results
    filename = '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/tests/method_comparisons/comprehensive_three_method_evaluation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comprehensive statistical analysis saved to: {filename}")

    # Also save high-resolution version
    filename_hires = filename.replace('.png', '_high_res.png')
    plt.savefig(filename_hires, dpi=600, bbox_inches='tight')
    print(f"📊 High-resolution version saved to: {filename_hires}")

    plt.show()


def main():
    """Run comprehensive robustness analysis"""
    print("Starting Comprehensive Three-Method Robustness Analysis...")
    print("This will test all methods across diverse problem configurations")
    print("Expected execution time: 15-30 minutes depending on problem complexity")

    try:
        results = run_comprehensive_robustness_test()

        print(f"\n{'='*100}")
        print("COMPREHENSIVE ROBUSTNESS ANALYSIS COMPLETED")
        print(f"{'='*100}")
        print("📊 Statistical analysis and visualizations have been generated")
        print("🔍 Check the plots for detailed robustness assessment")

        return results

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        return None
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
