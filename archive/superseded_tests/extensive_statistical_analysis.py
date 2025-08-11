#!/usr/bin/env python3
"""
Extensive Statistical Analysis
Comprehensive statistical evaluation of all three methods with extensive test cases
to demonstrate robustness, stability, computational cost, and mass conservation.
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy import ndimage

warnings.filterwarnings('ignore')

from mfg_pde.alg.damped_fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.fp_solvers.fdm_fp import FdmFPSolver
from mfg_pde.alg.fp_solvers.particle_fp import ParticleFPSolver
from mfg_pde.alg.hjb_solvers.fdm_hjb import FdmHJBSolver
from mfg_pde.alg.hjb_solvers.tuned_smart_qp_gfdm_hjb import TunedSmartQPGFDMHJBSolver
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def generate_extensive_test_cases():
    """Generate extensive test cases for comprehensive statistical analysis"""
    test_cases = []

    # 1. Scale variation tests (15 cases)
    print("Generating scale variation tests...")
    for Nx in [12, 15, 18, 20, 22]:
        for T in [0.4, 0.8, 1.2]:
            test_cases.append(
                {
                    'name': f'Scale_Nx{Nx}_T{T}',
                    'params': {
                        'xmin': 0.0,
                        'xmax': 1.0,
                        'Nx': Nx,
                        'T': T,
                        'Nt': int(T * 25),
                        'sigma': 0.12,
                        'coefCT': 0.02,
                    },
                    'category': 'Scale Variation',
                    'difficulty': 'Easy' if Nx <= 15 and T <= 0.8 else 'Medium',
                }
            )

    # 2. Volatility sweep tests (12 cases)
    print("Generating volatility sweep tests...")
    for sigma in [0.05, 0.08, 0.12, 0.16, 0.20, 0.25]:
        for coefCT in [0.01, 0.03]:
            test_cases.append(
                {
                    'name': f'Vol_s{sigma:.2f}_c{coefCT:.2f}',
                    'params': {
                        'xmin': 0.0,
                        'xmax': 1.0,
                        'Nx': 18,
                        'T': 0.8,
                        'Nt': 20,
                        'sigma': sigma,
                        'coefCT': coefCT,
                    },
                    'category': 'Volatility Sweep',
                    'difficulty': 'Easy' if sigma <= 0.12 else 'Hard',
                }
            )

    # 3. Time horizon tests (9 cases)
    print("Generating time horizon tests...")
    for T in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for Nt_factor in [15, 25]:
            if len([case for case in test_cases if case['params']['T'] == T]) < 3:
                test_cases.append(
                    {
                        'name': f'Time_T{T}_Nt{int(T*Nt_factor)}',
                        'params': {
                            'xmin': 0.0,
                            'xmax': 1.0,
                            'Nx': 16,
                            'T': T,
                            'Nt': int(T * Nt_factor),
                            'sigma': 0.10,
                            'coefCT': 0.02,
                        },
                        'category': 'Time Horizon',
                        'difficulty': 'Easy' if T <= 1.0 else 'Hard',
                    }
                )
            if len([case for case in test_cases if case['category'] == 'Time Horizon']) >= 9:
                break

    # 4. Control cost variation (8 cases)
    print("Generating control cost tests...")
    for coefCT in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]:
        if len([case for case in test_cases if case['category'] == 'Control Cost']) < 8:
            test_cases.append(
                {
                    'name': f'Control_c{coefCT:.3f}',
                    'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 16, 'T': 1.0, 'Nt': 25, 'sigma': 0.12, 'coefCT': coefCT},
                    'category': 'Control Cost',
                    'difficulty': 'Easy' if coefCT >= 0.02 else 'Medium',
                }
            )

    # 5. Mixed challenging tests (6 cases)
    print("Generating challenging combination tests...")
    challenging_params = [
        {'Nx': 24, 'T': 1.5, 'sigma': 0.18, 'coefCT': 0.01, 'Nt': 30},
        {'Nx': 20, 'T': 2.0, 'sigma': 0.22, 'coefCT': 0.015, 'Nt': 40},
        {'Nx': 26, 'T': 1.0, 'sigma': 0.25, 'coefCT': 0.025, 'Nt': 25},
        {'Nx': 18, 'T': 2.5, 'sigma': 0.15, 'coefCT': 0.01, 'Nt': 50},
        {'Nx': 22, 'T': 1.8, 'sigma': 0.20, 'coefCT': 0.02, 'Nt': 36},
        {'Nx': 16, 'T': 3.0, 'sigma': 0.12, 'coefCT': 0.005, 'Nt': 60},
    ]

    for i, params in enumerate(challenging_params):
        test_cases.append(
            {
                'name': f'Challenge_{i+1}',
                'params': {
                    'xmin': 0.0,
                    'xmax': 1.0,
                    'Nx': params['Nx'],
                    'T': params['T'],
                    'Nt': params['Nt'],
                    'sigma': params['sigma'],
                    'coefCT': params['coefCT'],
                },
                'category': 'Challenging',
                'difficulty': 'Hard',
            }
        )

    print(f"Generated {len(test_cases)} test cases")
    return test_cases


def test_method_with_stats(method_name, solver_func, params, test_name, timeout=600):
    """Test a method and return comprehensive statistics"""
    try:
        start_time = time.time()
        result = solver_func(params, test_name, timeout)
        total_time = time.time() - start_time

        if not result['success']:
            return {
                'success': False,
                'method': method_name,
                'test_name': test_name,
                'error': result.get('error', 'Unknown'),
                'time': total_time,
            }

        # Calculate additional statistics
        U, M = result.get('U'), result.get('M')

        # Mass conservation analysis
        Dx = (params['xmax'] - params['xmin']) / params['Nx']
        initial_mass = np.sum(M[0, :]) * Dx
        final_mass = np.sum(M[-1, :]) * Dx
        mass_error = abs(final_mass - initial_mass) / initial_mass * 100

        # Solution stability metrics
        if M.shape[0] > 2:
            # Temporal stability - how much solution changes over time
            temporal_changes = []
            for t in range(1, M.shape[0]):
                change = np.linalg.norm(M[t, :] - M[t - 1, :]) / np.linalg.norm(M[t - 1, :])
                temporal_changes.append(change)
            temporal_stability = np.mean(temporal_changes)
            temporal_variance = np.var(temporal_changes)
        else:
            temporal_stability = 0.0
            temporal_variance = 0.0

        # Spatial smoothness
        if M.shape[1] > 2:
            final_density = M[-1, :]
            spatial_gradient = np.gradient(final_density)
            spatial_smoothness = np.mean(np.abs(spatial_gradient))
            spatial_roughness = np.var(spatial_gradient)
        else:
            spatial_smoothness = 0.0
            spatial_roughness = 0.0

        # Numerical stability indicators
        u_final = U[-1, :]
        u_max = np.max(np.abs(u_final))
        u_std = np.std(u_final)

        return {
            'success': True,
            'method': method_name,
            'test_name': test_name,
            'time': result['time'],
            'total_time': total_time,
            'mass_error': mass_error,
            'converged': result.get('converged', False),
            'iterations': result.get('iterations', 0),
            'temporal_stability': temporal_stability,
            'temporal_variance': temporal_variance,
            'spatial_smoothness': spatial_smoothness,
            'spatial_roughness': spatial_roughness,
            'u_max': u_max,
            'u_std': u_std,
            'method_specific': result.get('method_specific', {}),
            'solution_quality': 'Good' if mass_error < 10 and u_max < 100 else 'Poor',
        }

    except Exception as e:
        return {'success': False, 'method': method_name, 'test_name': test_name, 'error': str(e), 'time': 0}


def test_pure_fdm(problem_params, test_name, timeout=300):
    """Test Pure FDM method"""
    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        fdm_hjb_solver = FdmHJBSolver(problem=problem)
        fdm_fp_solver = FdmFPSolver(problem=problem, boundary_conditions=no_flux_bc)
        fdm_solver = FixedPointIterator(
            problem=problem, hjb_solver=fdm_hjb_solver, fp_solver=fdm_fp_solver, thetaUM=0.5
        )

        start_time = time.time()
        U_fdm, M_fdm, iterations_run, l2distu_rel, l2distm_rel = fdm_solver.solve(Niter_max=8, l2errBoundPicard=1e-3)
        solve_time = time.time() - start_time

        if solve_time > timeout:
            return {'success': False, 'error': 'Timeout', 'time': solve_time}

        return {
            'success': True,
            'time': solve_time,
            'converged': l2distu_rel[-1] < 1e-3 if len(l2distu_rel) > 0 else False,
            'iterations': iterations_run,
            'U': U_fdm,
            'M': M_fdm,
            'method_specific': {
                'type': 'Grid-based',
                'complexity': 'Low',
                'final_error_u': l2distu_rel[-1] if len(l2distu_rel) > 0 else 0,
                'final_error_m': l2distm_rel[-1] if len(l2distm_rel) > 0 else 0,
            },
        }

    except Exception as e:
        return {'success': False, 'error': str(e), 'time': 0}


def test_hybrid_particle_fdm(problem_params, test_name, timeout=300):
    """Test Hybrid Particle-FDM method"""
    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        num_particles = min(120, max(60, problem_params['Nx'] * 4))

        hybrid_hjb_solver = FdmHJBSolver(problem=problem)
        hybrid_fp_solver = ParticleFPSolver(
            problem=problem, num_particles=num_particles, boundary_conditions=no_flux_bc
        )
        hybrid_solver = FixedPointIterator(
            problem=problem, hjb_solver=hybrid_hjb_solver, fp_solver=hybrid_fp_solver, thetaUM=0.5
        )

        start_time = time.time()
        U_hybrid, M_hybrid, iterations_run, l2distu_rel, l2distm_rel = hybrid_solver.solve(
            Niter_max=8, l2errBoundPicard=1e-3
        )
        solve_time = time.time() - start_time

        if solve_time > timeout:
            return {'success': False, 'error': 'Timeout', 'time': solve_time}

        return {
            'success': True,
            'time': solve_time,
            'converged': l2distu_rel[-1] < 1e-3 if len(l2distu_rel) > 0 else False,
            'iterations': iterations_run,
            'U': U_hybrid,
            'M': M_hybrid,
            'method_specific': {
                'type': 'Hybrid',
                'particles': num_particles,
                'complexity': 'Medium',
                'final_error_u': l2distu_rel[-1] if len(l2distu_rel) > 0 else 0,
                'final_error_m': l2distm_rel[-1] if len(l2distm_rel) > 0 else 0,
            },
        }

    except Exception as e:
        return {'success': False, 'error': str(e), 'time': 0}


def test_optimized_qp_collocation(problem_params, test_name, timeout=600):
    """Test Optimized QP-Collocation method"""
    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        # Adaptive parameters
        num_collocation_points = min(12, max(8, problem_params['Nx'] // 2))
        num_particles = min(150, max(100, problem_params['Nx'] * 5))

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
        U_qp, M_qp, info_qp = qp_solver.solve(Niter=8, l2errBound=1e-3, verbose=False)
        solve_time = time.time() - start_time

        if solve_time > timeout:
            return {'success': False, 'error': 'Timeout', 'time': solve_time}

        # Get QP optimization statistics
        qp_stats = tuned_hjb_solver.get_tuned_qp_report()

        return {
            'success': True,
            'time': solve_time,
            'converged': info_qp.get('converged', False),
            'iterations': info_qp.get('iterations', 0),
            'U': U_qp,
            'M': M_qp,
            'method_specific': {
                'type': 'Advanced Collocation',
                'particles': num_particles,
                'collocation_points': num_collocation_points,
                'qp_usage_rate': qp_stats.get('qp_usage_rate', 1.0),
                'optimization_quality': qp_stats.get('optimization_quality', 'N/A'),
                'complexity': 'High',
                'qp_stats': qp_stats,
            },
        }

    except Exception as e:
        return {'success': False, 'error': str(e), 'time': 0}


def run_extensive_statistical_analysis():
    """Run extensive statistical analysis"""
    print("=" * 100)
    print("EXTENSIVE STATISTICAL ANALYSIS - THREE METHOD COMPARISON")
    print("=" * 100)
    print("Comprehensive evaluation with extensive test cases for robustness and stability")

    test_cases = generate_extensive_test_cases()
    total_cases = len(test_cases)

    print(f"\n📊 Analysis Overview:")
    print(f"   • Total test cases: {total_cases}")
    print(f"   • Methods: Pure FDM, Hybrid P-FDM, Optimized QP-Collocation")
    print(f"   • Metrics: Robustness, Stability, Cost, Mass Conservation")
    print(f"   • Expected time: ~30 minutes")

    # Group test cases by category
    categories = {}
    for case in test_cases:
        cat = case['category']
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n📋 Test Case Distribution:")
    for cat, count in categories.items():
        print(f"   • {cat}: {count} cases")

    # Initialize results storage
    results = {'test_cases': test_cases, 'fdm_results': [], 'hybrid_results': [], 'qp_results': []}

    # Method configurations
    methods = [
        ('Pure FDM', test_pure_fdm),
        ('Hybrid P-FDM', test_hybrid_particle_fdm),
        ('QP-Collocation', test_optimized_qp_collocation),
    ]

    total_tests = total_cases * len(methods)
    current_test = 0

    print(f"\n{'='*80}")
    print("RUNNING EXTENSIVE STATISTICAL TESTS")
    print(f"{'='*80}")

    # Run tests with progress tracking
    for i, test_case in enumerate(test_cases):
        case_name = test_case['name']
        params = test_case['params']
        category = test_case['category']
        difficulty = test_case['difficulty']

        print(f"\n{'-'*70}")
        print(f"TEST CASE {i+1}/{total_cases}: {case_name} ({category} - {difficulty})")
        print(f"Params: Nx={params['Nx']}, T={params['T']}, σ={params['sigma']:.3f}, coefCT={params['coefCT']:.3f}")
        print(f"{'-'*70}")

        case_start_time = time.time()

        # Test each method
        for method_name, test_func in methods:
            current_test += 1
            print(f"[{current_test:3d}/{total_tests}] Testing {method_name}...", end=" ", flush=True)

            # Set timeout based on problem difficulty
            timeout = 300 if difficulty == 'Easy' else 600 if difficulty == 'Medium' else 900

            result = test_method_with_stats(method_name, test_func, params, case_name, timeout)
            result['category'] = category
            result['difficulty'] = difficulty

            # Store results
            if method_name == 'Pure FDM':
                results['fdm_results'].append(result)
            elif method_name == 'Hybrid P-FDM':
                results['hybrid_results'].append(result)
            else:
                results['qp_results'].append(result)

            # Print result summary
            if result['success']:
                time_str = f"{result['time']:.1f}s"
                if 'mass_error' in result:
                    mass_str = f", mass: {result['mass_error']:.2f}%"
                else:
                    mass_str = ""

                if method_name == 'QP-Collocation' and 'method_specific' in result:
                    qp_rate = result['method_specific'].get('qp_usage_rate', 0)
                    qp_str = f", QP: {qp_rate:.1%}"
                else:
                    qp_str = ""

                print(f"✓ {time_str}{mass_str}{qp_str}")
            else:
                error_str = result.get('error', 'Unknown')[:30]
                print(f"✗ {error_str}")

        case_time = time.time() - case_start_time
        print(f"Case completed in {case_time:.1f}s")

    print(f"\n{'='*100}")
    print("GENERATING COMPREHENSIVE STATISTICAL ANALYSIS")
    print(f"{'='*100}")

    # Generate comprehensive analysis
    analyze_and_visualize_results(results)

    return results


def analyze_and_visualize_results(results):
    """Generate comprehensive statistical analysis and visualization"""

    # Calculate comprehensive statistics
    methods = ['Pure FDM', 'Hybrid P-FDM', 'QP-Collocation']
    method_results = [results['fdm_results'], results['hybrid_results'], results['qp_results']]

    # Overall statistics
    overall_stats = {}
    for method_name, method_result in zip(methods, method_results):
        successful = [r for r in method_result if r['success']]
        total = len(method_result)

        if successful:
            times = [r['time'] for r in successful]
            mass_errors = [r.get('mass_error', 0) for r in successful if 'mass_error' in r]
            temporal_stabilities = [r.get('temporal_stability', 0) for r in successful if 'temporal_stability' in r]
            spatial_smoothness = [r.get('spatial_smoothness', 0) for r in successful if 'spatial_smoothness' in r]

            overall_stats[method_name] = {
                'success_rate': len(successful) / total * 100,
                'successful_count': len(successful),
                'total_count': total,
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'median_time': np.median(times),
                'avg_mass_error': np.mean(mass_errors) if mass_errors else 0,
                'std_mass_error': np.std(mass_errors) if mass_errors else 0,
                'avg_temporal_stability': np.mean(temporal_stabilities) if temporal_stabilities else 0,
                'avg_spatial_smoothness': np.mean(spatial_smoothness) if spatial_smoothness else 0,
                'robust_score': len(successful) / total * 100,
                'times': times,
                'mass_errors': mass_errors,
            }
        else:
            overall_stats[method_name] = {
                'success_rate': 0,
                'successful_count': 0,
                'total_count': total,
                'avg_time': 0,
                'std_time': 0,
                'median_time': 0,
                'avg_mass_error': 0,
                'std_mass_error': 0,
                'avg_temporal_stability': 0,
                'avg_spatial_smoothness': 0,
                'robust_score': 0,
                'times': [],
                'mass_errors': [],
            }

    # Print comprehensive statistics
    print("\n📈 COMPREHENSIVE STATISTICAL SUMMARY:")
    print("=" * 80)

    for method_name, stats in overall_stats.items():
        print(f"\n{method_name}:")
        print(f"  Success Rate: {stats['success_rate']:.1f}% ({stats['successful_count']}/{stats['total_count']})")
        if stats['successful_count'] > 0:
            print(f"  Computational Cost: {stats['avg_time']:.2f} ± {stats['std_time']:.2f} seconds")
            print(f"  Mass Conservation: {stats['avg_mass_error']:.3f} ± {stats['std_mass_error']:.3f}% error")
            print(f"  Temporal Stability: {stats['avg_temporal_stability']:.4f}")
            print(f"  Spatial Smoothness: {stats['avg_spatial_smoothness']:.4f}")

    # QP-specific analysis
    qp_successful = [r for r in results['qp_results'] if r['success']]
    if qp_successful:
        qp_usage_rates = []
        for r in qp_successful:
            if 'method_specific' in r and 'qp_usage_rate' in r['method_specific']:
                qp_usage_rates.append(r['method_specific']['qp_usage_rate'] * 100)

        if qp_usage_rates:
            print(f"\n🎯 QP OPTIMIZATION ANALYSIS:")
            print(f"  Average QP Usage: {np.mean(qp_usage_rates):.1f} ± {np.std(qp_usage_rates):.1f}%")
            print(f"  QP Usage Range: {np.min(qp_usage_rates):.1f}% - {np.max(qp_usage_rates):.1f}%")
            print(
                f"  Target Achievement: {np.sum(np.array(qp_usage_rates) <= 15) / len(qp_usage_rates) * 100:.1f}% within 15%"
            )

    # Create comprehensive 1x3 figure
    create_comprehensive_1x3_figure(overall_stats, results)


def create_comprehensive_1x3_figure(overall_stats, results):
    """Create comprehensive 1x3 statistical figure"""

    # Set up the figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Comprehensive Three-Method Statistical Analysis', fontsize=16, fontweight='bold')

    methods = list(overall_stats.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue

    # SUBPLOT 1: Robustness & Stability Analysis
    ax1_twin = ax1.twinx()

    # Success rates (bars)
    success_rates = [overall_stats[method]['success_rate'] for method in methods]
    bars1 = ax1.bar(
        [i - 0.2 for i in range(len(methods))], success_rates, width=0.4, color=colors, alpha=0.7, label='Success Rate'
    )

    # Temporal stability (line)
    stability_scores = [
        overall_stats[method]['avg_temporal_stability'] * 1000 for method in methods
    ]  # Scale up for visibility
    line1 = ax1_twin.plot(
        range(len(methods)), stability_scores, 'ko-', linewidth=2, markersize=8, label='Temporal Stability (×1000)'
    )

    ax1.set_ylabel('Success Rate (%)', fontweight='bold')
    ax1.set_xlabel('Methods', fontweight='bold')
    ax1.set_title('Robustness & Stability', fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax1_twin.set_ylabel('Temporal Stability (×1000)', fontweight='bold')
    ax1_twin.set_ylim(0, max(stability_scores) * 1.2 if stability_scores else 1)

    # Add value labels
    for i, (bar, rate, stability) in enumerate(zip(bars1, success_rates, stability_scores)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f'{rate:.0f}%',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=9,
        )

        if stability > 0:
            ax1_twin.text(
                i, stability + max(stability_scores) * 0.05, f'{stability:.1f}', ha='center', va='bottom', fontsize=8
            )

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    # SUBPLOT 2: Computational Cost Analysis
    successful_methods = [method for method in methods if overall_stats[method]['successful_count'] > 0]

    if successful_methods:
        # Box plots for computational time
        time_data = []
        positions = []
        colors_box = []

        for i, method in enumerate(methods):
            if overall_stats[method]['successful_count'] > 0:
                times = overall_stats[method]['times']
                if times:
                    time_data.append(times)
                    positions.append(i)
                    colors_box.append(colors[i])

        if time_data:
            bp = ax2.boxplot(time_data, positions=positions, patch_artist=True, widths=0.6, showfliers=True)

            for patch, color in zip(bp['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add mean markers
            for i, (pos, method) in enumerate(zip(positions, [methods[p] for p in positions])):
                mean_time = overall_stats[method]['avg_time']
                ax2.scatter(pos, mean_time, color='red', s=50, zorder=5, marker='D')
                ax2.text(
                    pos,
                    mean_time + max([max(td) for td in time_data]) * 0.05,
                    f'{mean_time:.1f}s',
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    fontsize=9,
                )

    ax2.set_ylabel('Computational Time (seconds)', fontweight='bold')
    ax2.set_xlabel('Methods', fontweight='bold')
    ax2.set_title('Computational Cost Distribution', fontweight='bold')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Add statistics text
    stats_text = "Statistics:\n"
    for method in methods:
        if overall_stats[method]['successful_count'] > 0:
            avg_t = overall_stats[method]['avg_time']
            std_t = overall_stats[method]['std_time']
            stats_text += f"{method.split()[0]}: {avg_t:.1f}±{std_t:.1f}s\n"

    ax2.text(
        0.02,
        0.98,
        stats_text,
        transform=ax2.transAxes,
        fontsize=8,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # SUBPLOT 3: Mass Conservation & QP Optimization
    ax3_twin = ax3.twinx()

    # Mass conservation errors (bars)
    mass_errors_avg = [overall_stats[method]['avg_mass_error'] for method in methods]
    mass_errors_std = [overall_stats[method]['std_mass_error'] for method in methods]

    bars3 = ax3.bar(
        [i - 0.2 for i in range(len(methods))],
        mass_errors_avg,
        yerr=mass_errors_std,
        width=0.4,
        color=colors,
        alpha=0.7,
        capsize=5,
        label='Mass Conservation Error',
    )

    # QP usage rates (line) - only for QP-Collocation
    qp_successful = [r for r in results['qp_results'] if r['success']]
    if qp_successful:
        qp_usage_rates = []
        for r in qp_successful:
            if 'method_specific' in r and 'qp_usage_rate' in r['method_specific']:
                qp_usage_rates.append(r['method_specific']['qp_usage_rate'] * 100)

        if qp_usage_rates:
            avg_qp_usage = np.mean(qp_usage_rates)
            std_qp_usage = np.std(qp_usage_rates)

            # Plot QP usage for QP-Collocation method only
            qp_method_idx = methods.index('QP-Collocation')
            ax3_twin.errorbar(
                [qp_method_idx],
                [avg_qp_usage],
                yerr=[std_qp_usage],
                fmt='ro-',
                linewidth=2,
                markersize=8,
                capsize=5,
                label=f'QP Usage Rate',
            )

            # Add target line
            ax3_twin.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Target: 10%')

            # Add QP usage text
            ax3_twin.text(
                qp_method_idx,
                avg_qp_usage + std_qp_usage + 2,
                f'{avg_qp_usage:.1f}±{std_qp_usage:.1f}%',
                ha='center',
                va='bottom',
                fontweight='bold',
                fontsize=9,
            )

    ax3.set_ylabel('Mass Conservation Error (%)', fontweight='bold')
    ax3.set_xlabel('Methods', fontweight='bold')
    ax3.set_title('Mass Conservation & QP Optimization', fontweight='bold')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    ax3_twin.set_ylabel('QP Usage Rate (%)', fontweight='bold', color='red')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    ax3_twin.set_ylim(0, 25)

    # Add value labels for mass conservation
    for i, (bar, error, std_err) in enumerate(zip(bars3, mass_errors_avg, mass_errors_std)):
        if error > 0:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std_err + height * 0.1,
                f'{error:.2f}%',
                ha='center',
                va='bottom',
                fontweight='bold',
                fontsize=9,
            )

    # Legend
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left', fontsize=8)

    # Adjust layout and save
    plt.tight_layout()

    # Save comprehensive results
    filename = '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/tests/method_comparisons/extensive_statistical_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Extensive statistical analysis saved to: {filename}")

    # Also save high-resolution version
    filename_hires = filename.replace('.png', '_high_res.png')
    plt.savefig(filename_hires, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"📊 High-resolution version saved to: {filename_hires}")

    plt.show()

    # Print final summary
    print(f"\n{'='*100}")
    print("EXTENSIVE STATISTICAL ANALYSIS COMPLETED")
    print(f"{'='*100}")

    print("🎯 KEY FINDINGS:")
    best_robust = max(overall_stats.items(), key=lambda x: x[1]['success_rate'])
    fastest_avg = min(
        [(k, v) for k, v in overall_stats.items() if v['successful_count'] > 0],
        key=lambda x: x[1]['avg_time'],
        default=(None, None),
    )
    best_conservation = min(
        [(k, v) for k, v in overall_stats.items() if v['successful_count'] > 0],
        key=lambda x: x[1]['avg_mass_error'],
        default=(None, None),
    )

    print(f"   🏆 Most Robust: {best_robust[0]} ({best_robust[1]['success_rate']:.1f}% success)")
    if fastest_avg[0]:
        print(f"   ⚡ Fastest Average: {fastest_avg[0]} ({fastest_avg[1]['avg_time']:.1f}s)")
    if best_conservation[0]:
        print(
            f"   🎯 Best Mass Conservation: {best_conservation[0]} ({best_conservation[1]['avg_mass_error']:.3f}% error)"
        )

    if qp_successful and qp_usage_rates:
        target_achieved = np.sum(np.array(qp_usage_rates) <= 15) / len(qp_usage_rates) * 100
        print(f"   🔧 QP Optimization: {np.mean(qp_usage_rates):.1f}% avg usage ({target_achieved:.0f}% within target)")


def main():
    """Run extensive statistical analysis"""
    print("Starting Extensive Statistical Analysis...")
    print("This will run comprehensive tests across all three methods")
    print("Expected execution time: ~30 minutes")
    print("Generating 1×3 statistical figure with robustness, cost, and conservation analysis")

    try:
        results = run_extensive_statistical_analysis()

        print(f"\n{'='*100}")
        print("🎉 EXTENSIVE STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"{'='*100}")
        print("📊 Comprehensive 1×3 statistical figure generated")
        print("🔍 All metrics analyzed: robustness, stability, cost, mass conservation")

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
