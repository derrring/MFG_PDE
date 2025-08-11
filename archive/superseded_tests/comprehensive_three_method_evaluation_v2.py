#!/usr/bin/env python3
"""
Comprehensive Three Method Evaluation v2
Comparing FDM, Hybrid, and Improved QP-Collocation Methods

This version includes the optimized QP-collocation solver and tests more challenging scenarios
to properly evaluate the improvements and relative performance.
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

from mfg_pde.alg.damped_fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.hjb_solvers.optimized_gfdm_hjb_v2 import OptimizedGFDMHJBSolver
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


class ComprehensiveThreeMethodEvaluator:
    """Evaluates FDM, Hybrid, and Improved QP-Collocation methods"""

    def __init__(self, base_params):
        self.base_params = base_params
        self.results = {}

    def create_fdm_solver(self, problem_params, solver_params):
        """Create pure FDM solver"""
        problem = ExampleMFGProblem(**problem_params)

        solver = FixedPointIterator(
            problem=problem,
            hjb_method="FDM",
            fp_method="FDM",
            damping_factor=solver_params.get('damping_factor', 0.5),
            hjb_solver_params={},
            fp_solver_params={},
        )

        return solver, "Pure FDM"

    def create_hybrid_solver(self, problem_params, solver_params):
        """Create hybrid particle-FDM solver"""
        problem = ExampleMFGProblem(**problem_params)

        solver = FixedPointIterator(
            problem=problem,
            hjb_method="FDM",
            fp_method="Particle",
            damping_factor=solver_params.get('damping_factor', 0.3),
            hjb_solver_params={},
            fp_solver_params={
                'num_particles': solver_params.get('num_particles', 2000),
                'boundary_conditions': BoundaryConditions(type='no_flux'),
            },
        )

        return solver, "Hybrid Particle-FDM"

    def create_improved_qp_solver(self, problem_params, solver_params):
        """Create improved QP-collocation solver"""
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        # Setup collocation points
        num_collocation_points = solver_params.get('num_collocation_points', 12)
        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)

        # Identify boundary points
        boundary_tolerance = 1e-10
        boundary_indices = []
        for i, point in enumerate(collocation_points):
            x = point[0]
            if abs(x - problem.xmin) < boundary_tolerance or abs(x - problem.xmax) < boundary_tolerance:
                boundary_indices.append(i)
        boundary_indices = np.array(boundary_indices)

        # Create optimized HJB solver
        optimized_hjb_solver = OptimizedGFDMHJBSolver(
            problem=problem,
            collocation_points=collocation_points,
            delta=solver_params.get('delta', 0.35),
            taylor_order=solver_params.get('taylor_order', 2),
            weight_function="wendland",
            NiterNewton=solver_params.get('newton_iterations', 6),
            l2errBoundNewton=solver_params.get('newton_tolerance', 1e-3),
            boundary_indices=boundary_indices,
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
            qp_activation_tolerance=solver_params.get('qp_tolerance', 1e-3),
        )

        # Create particle collocation solver with optimized HJB solver
        solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=solver_params.get('num_particles', 200),
            delta=solver_params.get('delta', 0.35),
            taylor_order=solver_params.get('taylor_order', 2),
            weight_function="wendland",
            NiterNewton=solver_params.get('newton_iterations', 6),
            l2errBoundNewton=solver_params.get('newton_tolerance', 1e-3),
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=boundary_indices,
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
        )

        # Replace HJB solver with optimized version
        solver.hjb_solver = optimized_hjb_solver

        return solver, "Improved QP-Collocation", optimized_hjb_solver

    def run_single_test(self, scenario_name, problem_params, solver_params, max_time_minutes=30):
        """Run a single scenario test for all three methods"""
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*80}")
        print(f"Problem: Nx={problem_params['Nx']}, T={problem_params['T']}, σ={problem_params['sigma']}")
        print(f"Max simulation time: {max_time_minutes} minutes")

        scenario_results = {}

        # Test each method
        methods = [
            ('fdm', self.create_fdm_solver),
            ('hybrid', self.create_hybrid_solver),
            ('improved_qp', self.create_improved_qp_solver),
        ]

        for method_name, create_method in methods:
            print(f"\n{'-'*60}")
            print(f"Testing {method_name.upper()} method...")
            print(f"{'-'*60}")

            try:
                # Create solver
                if method_name == 'improved_qp':
                    solver, method_display_name, hjb_solver = create_method(problem_params, solver_params)
                else:
                    solver, method_display_name = create_method(problem_params, solver_params)
                    hjb_solver = None

                print(f"Solver created: {method_display_name}")

                # Run simulation with timeout
                start_time = time.time()
                max_time_seconds = max_time_minutes * 60

                try:
                    U, M, solve_info = solver.solve(
                        Niter=solver_params.get('max_iterations', 15),
                        l2errBound=solver_params.get('convergence_tolerance', 1e-3),
                        verbose=True,
                    )

                    solve_time = time.time() - start_time

                    # Check if we exceeded time limit
                    if solve_time > max_time_seconds:
                        print(f"⚠️  Simulation exceeded time limit ({solve_time/60:.1f} min)")
                        success = False
                        timeout = True
                    else:
                        success = U is not None and M is not None
                        timeout = False

                except Exception as e:
                    solve_time = time.time() - start_time
                    success = False
                    timeout = False
                    U, M, solve_info = None, None, {'iterations': 0, 'converged': False}
                    print(f"❌ Simulation failed: {e}")

                # Calculate quality metrics
                if success and M is not None:
                    # Mass conservation
                    Dx = (
                        problem_params.get('Dx')
                        or (problem_params['xmax'] - problem_params['xmin']) / problem_params['Nx']
                    )
                    initial_mass = np.sum(M[0, :]) * Dx
                    final_mass = np.sum(M[-1, :]) * Dx
                    if initial_mass > 1e-9:
                        mass_error = abs(final_mass - initial_mass) / initial_mass * 100
                    else:
                        mass_error = float('inf')

                    # Solution stability (variance in final solution)
                    solution_variance = np.var(U[-1, :]) if U is not None else float('inf')

                    # Convergence
                    converged = solve_info.get('converged', False)
                    iterations = solve_info.get('iterations', 0)

                else:
                    mass_error = float('inf')
                    solution_variance = float('inf')
                    converged = False
                    iterations = 0

                # Get optimization statistics for improved QP method
                optimization_stats = {}
                if method_name == 'improved_qp' and hjb_solver is not None:
                    if hasattr(hjb_solver, 'get_performance_report'):
                        optimization_stats = hjb_solver.get_performance_report()
                        print(f"QP Optimization Stats:")
                        print(f"  QP Activation Rate: {optimization_stats.get('qp_activation_rate', 0):.1%}")
                        print(f"  QP Skip Rate: {optimization_stats.get('qp_skip_rate', 0):.1%}")
                        print(f"  Total QP Calls: {optimization_stats.get('total_qp_calls', 0)}")

                # Store results
                scenario_results[method_name] = {
                    'method_name': method_display_name,
                    'success': success,
                    'timeout': timeout,
                    'solve_time': solve_time,
                    'solve_time_minutes': solve_time / 60,
                    'mass_error_percent': mass_error,
                    'solution_variance': solution_variance,
                    'converged': converged,
                    'iterations': iterations,
                    'U': U,
                    'M': M,
                    'solve_info': solve_info,
                    'optimization_stats': optimization_stats,
                }

                print(f"Results:")
                print(f"  Success: {success}")
                print(f"  Time: {solve_time:.1f}s ({solve_time/60:.1f} min)")
                print(f"  Mass Error: {mass_error:.2f}%")
                print(f"  Converged: {converged} ({iterations} iterations)")

            except Exception as e:
                print(f"❌ Method setup failed: {e}")
                scenario_results[method_name] = {
                    'method_name': method_name,
                    'success': False,
                    'timeout': False,
                    'solve_time': 0,
                    'solve_time_minutes': 0,
                    'mass_error_percent': float('inf'),
                    'solution_variance': float('inf'),
                    'converged': False,
                    'iterations': 0,
                    'U': None,
                    'M': None,
                    'solve_info': None,
                    'optimization_stats': {},
                    'error': str(e),
                }

        return scenario_results

    def run_comprehensive_evaluation(self, test_scenarios, max_time_minutes=30):
        """Run comprehensive evaluation across all scenarios"""
        print("=" * 100)
        print("COMPREHENSIVE THREE METHOD EVALUATION v2")
        print("=" * 100)
        print("Methods: Pure FDM, Hybrid Particle-FDM, Improved QP-Collocation")
        print(f"Maximum simulation time per method: {max_time_minutes} minutes")
        print("=" * 100)

        all_results = {}

        for scenario_name, scenario_config in test_scenarios.items():
            problem_params = {**self.base_params, **scenario_config['problem']}
            solver_params = scenario_config['solver']

            scenario_results = self.run_single_test(scenario_name, problem_params, solver_params, max_time_minutes)
            all_results[scenario_name] = scenario_results

        self.results = all_results
        self.print_summary()
        self.create_comparison_plots()

        return all_results

    def print_summary(self):
        """Print comprehensive results summary"""
        print(f"\n{'='*100}")
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print(f"{'='*100}")

        # Summary table
        print(
            f"\n{'Scenario':<20} {'Method':<20} {'Success':<8} {'Time(min)':<10} {'Mass Err %':<12} {'Converged':<10} {'Status':<15}"
        )
        print("-" * 100)

        for scenario_name, scenario_results in self.results.items():
            for method_name, result in scenario_results.items():
                success_str = "✓" if result['success'] else "✗"
                timeout_str = "(timeout)" if result.get('timeout', False) else ""
                converged_str = "✓" if result['converged'] else "✗"

                mass_err_str = (
                    f"{result['mass_error_percent']:.1f}" if result['mass_error_percent'] != float('inf') else "∞"
                )

                status = "GOOD" if result['success'] and not result.get('timeout', False) else "POOR"
                if result.get('timeout', False):
                    status = "TIMEOUT"
                elif not result['success']:
                    status = "FAILED"

                print(
                    f"{scenario_name:<20} {result['method_name']:<20} {success_str:<8} {result['solve_time_minutes']:<10.1f} {mass_err_str:<12} {converged_str:<10} {status:<15}"
                )

        print("-" * 100)

        # Performance analysis by method
        print(f"\nPERFORMANCE ANALYSIS BY METHOD:")
        print("-" * 50)

        methods = ['fdm', 'hybrid', 'improved_qp']
        method_stats = {}

        for method in methods:
            method_results = []
            for scenario_results in self.results.values():
                if method in scenario_results:
                    method_results.append(scenario_results[method])

            if method_results:
                success_rate = sum(1 for r in method_results if r['success']) / len(method_results) * 100
                avg_time = np.mean([r['solve_time_minutes'] for r in method_results if r['success']])

                mass_errors = [
                    r['mass_error_percent']
                    for r in method_results
                    if r['success'] and r['mass_error_percent'] != float('inf')
                ]
                avg_mass_error = np.mean(mass_errors) if mass_errors else float('inf')

                convergence_rate = sum(1 for r in method_results if r['converged']) / len(method_results) * 100

                method_stats[method] = {
                    'success_rate': success_rate,
                    'avg_time': avg_time,
                    'avg_mass_error': avg_mass_error,
                    'convergence_rate': convergence_rate,
                }

                print(f"\n{method_results[0]['method_name']}:")
                print(f"  Success Rate: {success_rate:.0f}%")
                print(f"  Average Time: {avg_time:.1f} min")
                print(f"  Average Mass Error: {avg_mass_error:.2f}%")
                print(f"  Convergence Rate: {convergence_rate:.0f}%")

        # QP Optimization Analysis
        print(f"\nQP OPTIMIZATION ANALYSIS:")
        print("-" * 50)

        qp_results = []
        for scenario_results in self.results.values():
            if 'improved_qp' in scenario_results:
                qp_result = scenario_results['improved_qp']
                if qp_result['optimization_stats']:
                    qp_results.append(qp_result['optimization_stats'])

        if qp_results:
            avg_activation_rate = np.mean([stats.get('qp_activation_rate', 1.0) for stats in qp_results])
            avg_skip_rate = np.mean([stats.get('qp_skip_rate', 0.0) for stats in qp_results])
            total_qp_calls = sum([stats.get('total_qp_calls', 0) for stats in qp_results])

            print(f"Average QP Activation Rate: {avg_activation_rate:.1%}")
            print(f"Average QP Skip Rate: {avg_skip_rate:.1%}")
            print(f"Total QP Calls Across All Tests: {total_qp_calls}")

            if avg_skip_rate > 0:
                estimated_speedup = 1 / (1 - avg_skip_rate * 0.9)
                print(f"Estimated Optimization Speedup: {estimated_speedup:.1f}x")

        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        print("-" * 50)

        if method_stats:
            # Find best method for each criterion
            best_success = max(method_stats.items(), key=lambda x: x[1]['success_rate'])
            best_speed = min(
                method_stats.items(), key=lambda x: x[1]['avg_time'] if not np.isnan(x[1]['avg_time']) else float('inf')
            )
            best_mass = min(
                method_stats.items(),
                key=lambda x: x[1]['avg_mass_error'] if not np.isinf(x[1]['avg_mass_error']) else float('inf'),
            )
            best_convergence = max(method_stats.items(), key=lambda x: x[1]['convergence_rate'])

            print(f"Most Reliable: {best_success[0]} ({best_success[1]['success_rate']:.0f}% success)")
            print(f"Fastest: {best_speed[0]} ({best_speed[1]['avg_time']:.1f} min avg)")
            print(f"Best Mass Conservation: {best_mass[0]} ({best_mass[1]['avg_mass_error']:.2f}% error)")
            print(f"Best Convergence: {best_convergence[0]} ({best_convergence[1]['convergence_rate']:.0f}% rate)")

    def create_comparison_plots(self):
        """Create comprehensive comparison plots"""
        if not self.results:
            print("No results to plot")
            return

        # Setup plot
        fig = plt.figure(figsize=(20, 16))

        # Extract data for plotting
        scenarios = list(self.results.keys())
        methods = ['fdm', 'hybrid', 'improved_qp']
        method_labels = ['Pure FDM', 'Hybrid Particle-FDM', 'Improved QP-Collocation']
        colors = ['red', 'blue', 'green']

        # Plot 1: Solve Time Comparison
        ax1 = plt.subplot(3, 3, 1)
        x_pos = np.arange(len(scenarios))
        width = 0.25

        for i, method in enumerate(methods):
            times = []
            for scenario in scenarios:
                if method in self.results[scenario] and self.results[scenario][method]['success']:
                    times.append(self.results[scenario][method]['solve_time_minutes'])
                else:
                    times.append(0)

            ax1.bar(x_pos + i * width, times, width, label=method_labels[i], color=colors[i], alpha=0.7)

        ax1.set_xlabel('Test Scenarios')
        ax1.set_ylabel('Solve Time (minutes)')
        ax1.set_title('Solve Time Comparison')
        ax1.set_xticks(x_pos + width)
        ax1.set_xticklabels(scenarios, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Mass Conservation Error
        ax2 = plt.subplot(3, 3, 2)

        for i, method in enumerate(methods):
            mass_errors = []
            for scenario in scenarios:
                if (
                    method in self.results[scenario]
                    and self.results[scenario][method]['success']
                    and self.results[scenario][method]['mass_error_percent'] != float('inf')
                ):
                    mass_errors.append(self.results[scenario][method]['mass_error_percent'])
                else:
                    mass_errors.append(100)  # High error for failed cases

            ax2.bar(x_pos + i * width, mass_errors, width, label=method_labels[i], color=colors[i], alpha=0.7)

        ax2.set_xlabel('Test Scenarios')
        ax2.set_ylabel('Mass Conservation Error (%)')
        ax2.set_title('Mass Conservation Quality')
        ax2.set_xticks(x_pos + width)
        ax2.set_xticklabels(scenarios, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # Plot 3: Success Rate by Method
        ax3 = plt.subplot(3, 3, 3)

        success_rates = []
        for method in methods:
            successes = 0
            total = 0
            for scenario_results in self.results.values():
                if method in scenario_results:
                    total += 1
                    if scenario_results[method]['success']:
                        successes += 1
            success_rates.append(successes / total * 100 if total > 0 else 0)

        bars = ax3.bar(method_labels, success_rates, color=colors, alpha=0.7)
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_title('Overall Success Rate')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f'{rate:.0f}%', ha='center', va='bottom')

        # Plot 4: QP Optimization Statistics (if available)
        ax4 = plt.subplot(3, 3, 4)

        qp_scenarios = []
        qp_activation_rates = []
        qp_skip_rates = []

        for scenario, results in self.results.items():
            if 'improved_qp' in results and results['improved_qp']['optimization_stats']:
                stats = results['improved_qp']['optimization_stats']
                qp_scenarios.append(scenario)
                qp_activation_rates.append(stats.get('qp_activation_rate', 1.0) * 100)
                qp_skip_rates.append(stats.get('qp_skip_rate', 0.0) * 100)

        if qp_scenarios:
            x_qp = np.arange(len(qp_scenarios))
            width_qp = 0.35

            ax4.bar(
                x_qp - width_qp / 2,
                qp_activation_rates,
                width_qp,
                label='QP Activation Rate',
                color='orange',
                alpha=0.7,
            )
            ax4.bar(x_qp + width_qp / 2, qp_skip_rates, width_qp, label='QP Skip Rate', color='green', alpha=0.7)

            ax4.set_xlabel('Test Scenarios')
            ax4.set_ylabel('Rate (%)')
            ax4.set_title('QP Optimization Statistics')
            ax4.set_xticks(x_qp)
            ax4.set_xticklabels(qp_scenarios, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # Plot 5-6: Solution Visualization (for first successful scenario)
        first_successful_scenario = None
        for scenario, results in self.results.items():
            if any(r['success'] and r['U'] is not None for r in results.values()):
                first_successful_scenario = scenario
                break

        if first_successful_scenario:
            results = self.results[first_successful_scenario]

            # Plot final density profiles
            ax5 = plt.subplot(3, 3, 5)

            for method, method_label, color in zip(methods, method_labels, colors):
                if method in results and results[method]['success'] and results[method]['M'] is not None:
                    M = results[method]['M']
                    problem_params = {**self.base_params, **test_scenarios[first_successful_scenario]['problem']}
                    x_grid = np.linspace(problem_params['xmin'], problem_params['xmax'], M.shape[1])
                    ax5.plot(x_grid, M[-1, :], label=f'{method_label} (t=T)', color=color, linewidth=2)

            ax5.set_xlabel('x')
            ax5.set_ylabel('Density')
            ax5.set_title(f'Final Density Profiles - {first_successful_scenario}')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

            # Plot value function profiles
            ax6 = plt.subplot(3, 3, 6)

            for method, method_label, color in zip(methods, method_labels, colors):
                if method in results and results[method]['success'] and results[method]['U'] is not None:
                    U = results[method]['U']
                    problem_params = {**self.base_params, **test_scenarios[first_successful_scenario]['problem']}
                    x_grid = np.linspace(problem_params['xmin'], problem_params['xmax'], U.shape[1])
                    ax6.plot(x_grid, U[0, :], label=f'{method_label} (t=0)', color=color, linewidth=2)

            ax6.set_xlabel('x')
            ax6.set_ylabel('Value Function')
            ax6.set_title(f'Initial Value Function - {first_successful_scenario}')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # Plot 7: Computational Efficiency (Time vs Quality)
        ax7 = plt.subplot(3, 3, 7)

        for method, method_label, color in zip(methods, method_labels, colors):
            times = []
            mass_errors = []

            for scenario_results in self.results.values():
                if method in scenario_results and scenario_results[method]['success']:
                    result = scenario_results[method]
                    if result['mass_error_percent'] != float('inf'):
                        times.append(result['solve_time_minutes'])
                        mass_errors.append(result['mass_error_percent'])

            if times and mass_errors:
                ax7.scatter(times, mass_errors, label=method_label, color=color, s=100, alpha=0.7)

        ax7.set_xlabel('Solve Time (minutes)')
        ax7.set_ylabel('Mass Conservation Error (%)')
        ax7.set_title('Computational Efficiency')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.set_yscale('log')

        # Plot 8: Convergence Analysis
        ax8 = plt.subplot(3, 3, 8)

        convergence_rates = []
        for method in methods:
            converged = 0
            total = 0
            for scenario_results in self.results.values():
                if method in scenario_results:
                    total += 1
                    if scenario_results[method]['converged']:
                        converged += 1
            convergence_rates.append(converged / total * 100 if total > 0 else 0)

        bars = ax8.bar(method_labels, convergence_rates, color=colors, alpha=0.7)
        ax8.set_ylabel('Convergence Rate (%)')
        ax8.set_title('Convergence Success Rate')
        ax8.set_ylim(0, 100)
        ax8.grid(True, alpha=0.3)

        # Add value labels
        for bar, rate in zip(bars, convergence_rates):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f'{rate:.0f}%', ha='center', va='bottom')

        # Plot 9: Method Ranking Summary
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        # Create ranking text
        ranking_text = "METHOD RANKING SUMMARY\n\n"

        # Calculate overall scores (lower is better for time and mass error)
        method_scores = {}
        for method, label in zip(methods, method_labels):
            # Success rate (higher is better)
            success_score = sum(
                1
                for scenario_results in self.results.values()
                if method in scenario_results and scenario_results[method]['success']
            )

            # Average time (lower is better)
            times = [
                r[method]['solve_time_minutes'] for r in self.results.values() if method in r and r[method]['success']
            ]
            avg_time = np.mean(times) if times else float('inf')

            # Average mass error (lower is better)
            mass_errors = [
                r[method]['mass_error_percent']
                for r in self.results.values()
                if method in r and r[method]['success'] and r[method]['mass_error_percent'] != float('inf')
            ]
            avg_mass_error = np.mean(mass_errors) if mass_errors else float('inf')

            method_scores[method] = {
                'label': label,
                'success_score': success_score,
                'avg_time': avg_time,
                'avg_mass_error': avg_mass_error,
            }

        # Rank by different criteria
        success_ranking = sorted(method_scores.items(), key=lambda x: x[1]['success_score'], reverse=True)
        speed_ranking = sorted(method_scores.items(), key=lambda x: x[1]['avg_time'])
        quality_ranking = sorted(method_scores.items(), key=lambda x: x[1]['avg_mass_error'])

        ranking_text += "RELIABILITY (Success Rate):\n"
        for i, (method, scores) in enumerate(success_ranking):
            ranking_text += f"{i+1}. {scores['label']} ({scores['success_score']} successes)\n"

        ranking_text += "\nSPEED:\n"
        for i, (method, scores) in enumerate(speed_ranking):
            if not np.isinf(scores['avg_time']):
                ranking_text += f"{i+1}. {scores['label']} ({scores['avg_time']:.1f} min)\n"

        ranking_text += "\nQUALITY (Mass Conservation):\n"
        for i, (method, scores) in enumerate(quality_ranking):
            if not np.isinf(scores['avg_mass_error']):
                ranking_text += f"{i+1}. {scores['label']} ({scores['avg_mass_error']:.2f}% error)\n"

        ax9.text(
            0.05,
            0.95,
            ranking_text,
            transform=ax9.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
        )

        plt.tight_layout()
        plt.savefig(
            '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/tests/method_comparisons/comprehensive_three_method_evaluation_v2.png',
            dpi=300,
            bbox_inches='tight',
        )
        print(f"\nComprehensive comparison plots saved to: comprehensive_three_method_evaluation_v2.png")
        plt.show()


def main():
    """Run comprehensive three method evaluation"""

    # Base problem parameters
    base_params = {'xmin': 0.0, 'xmax': 1.0, 'sigma': 0.15, 'coefCT': 0.02}

    # Test scenarios with increasing difficulty
    test_scenarios = {
        'moderate_standard': {
            'problem': {'Nx': 25, 'T': 1.0, 'Nt': 50},
            'solver': {
                'max_iterations': 12,
                'convergence_tolerance': 1e-3,
                'damping_factor': 0.3,
                'num_particles': 2000,
                'num_collocation_points': 12,
                'delta': 0.35,
                'taylor_order': 2,
                'newton_iterations': 6,
                'newton_tolerance': 1e-3,
                'qp_tolerance': 1e-3,
            },
        },
        'challenging_long': {
            'problem': {'Nx': 30, 'T': 2.0, 'Nt': 80, 'sigma': 0.2},
            'solver': {
                'max_iterations': 15,
                'convergence_tolerance': 1e-3,
                'damping_factor': 0.25,
                'num_particles': 2500,
                'num_collocation_points': 14,
                'delta': 0.3,
                'taylor_order': 2,
                'newton_iterations': 8,
                'newton_tolerance': 1e-4,
                'qp_tolerance': 1e-3,
            },
        },
        'extreme_test': {
            'problem': {'Nx': 35, 'T': 2.5, 'Nt': 100, 'sigma': 0.25, 'coefCT': 0.03},
            'solver': {
                'max_iterations': 20,
                'convergence_tolerance': 1e-3,
                'damping_factor': 0.2,
                'num_particles': 3000,
                'num_collocation_points': 16,
                'delta': 0.25,
                'taylor_order': 2,
                'newton_iterations': 10,
                'newton_tolerance': 1e-4,
                'qp_tolerance': 1e-3,
            },
        },
    }

    # Create evaluator and run tests
    evaluator = ComprehensiveThreeMethodEvaluator(base_params)

    try:
        print("Starting Comprehensive Three Method Evaluation v2...")
        print("This will test FDM, Hybrid, and Improved QP-Collocation methods")
        print("Expected total execution time: 30-120 minutes depending on scenarios")

        results = evaluator.run_comprehensive_evaluation(test_scenarios, max_time_minutes=45)

        print("\n" + "=" * 100)
        print("COMPREHENSIVE THREE METHOD EVALUATION v2 COMPLETED")
        print("=" * 100)
        print("Check the summary above and generated plots for detailed results.")

        return results

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        return None
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
