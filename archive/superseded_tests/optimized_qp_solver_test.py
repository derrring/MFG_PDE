#!/usr/bin/env python3
"""
Optimized QP-Collocation Solver Test
Comprehensive validation of the optimized solver against baseline implementation.
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

from mfg_pde.alg.hjb_solvers.optimized_gfdm_hjb_v2 import OptimizedGFDMHJBSolver
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


class OptimizedQPSolverTester:
    """Comprehensive tester for optimized QP-Collocation solver"""

    def __init__(self, problem_params, test_scenarios):
        self.problem_params = problem_params
        self.test_scenarios = test_scenarios
        self.results = {}

    def setup_problem(self, scenario_name):
        """Setup MFG problem for testing"""
        scenario = self.test_scenarios[scenario_name]
        params = {**self.problem_params, **scenario.get('problem_overrides', {})}

        problem = ExampleMFGProblem(**params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        # Setup collocation points
        num_collocation_points = scenario['num_collocation_points']
        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)

        # Identify boundary points
        boundary_tolerance = 1e-10
        boundary_indices = []
        for i, point in enumerate(collocation_points):
            x = point[0]
            if abs(x - problem.xmin) < boundary_tolerance or abs(x - problem.xmax) < boundary_tolerance:
                boundary_indices.append(i)
        boundary_indices = np.array(boundary_indices)

        return problem, collocation_points, boundary_indices, no_flux_bc

    def create_baseline_solver(self, problem, collocation_points, boundary_indices, boundary_conditions, scenario):
        """Create baseline QP-collocation solver"""
        return ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=scenario['num_particles'],
            delta=scenario['delta'],
            taylor_order=scenario['taylor_order'],
            weight_function="wendland",
            NiterNewton=scenario['newton_iterations'],
            l2errBoundNewton=scenario['newton_tolerance'],
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=boundary_indices,
            boundary_conditions=boundary_conditions,
            use_monotone_constraints=True,  # Always use constraints for fair comparison
        )

    def create_optimized_solver(self, problem, collocation_points, boundary_indices, boundary_conditions, scenario):
        """Create optimized QP-collocation solver"""
        # Create optimized HJB solver directly
        optimized_hjb_solver = OptimizedGFDMHJBSolver(
            problem=problem,
            collocation_points=collocation_points,
            delta=scenario['delta'],
            taylor_order=scenario['taylor_order'],
            weight_function="wendland",
            NiterNewton=scenario['newton_iterations'],
            l2errBoundNewton=scenario['newton_tolerance'],
            boundary_indices=boundary_indices,
            boundary_conditions=boundary_conditions,
            use_monotone_constraints=True,
            qp_activation_tolerance=1e-3,
        )

        # Create particle collocation solver with optimized HJB solver
        optimized_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=scenario['num_particles'],
            delta=scenario['delta'],
            taylor_order=scenario['taylor_order'],
            weight_function="wendland",
            NiterNewton=scenario['newton_iterations'],
            l2errBoundNewton=scenario['newton_tolerance'],
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=boundary_indices,
            boundary_conditions=boundary_conditions,
            use_monotone_constraints=True,
        )

        # Replace the HJB solver with our optimized version
        optimized_solver.hjb_solver = optimized_hjb_solver

        return optimized_solver, optimized_hjb_solver

    def run_solver_test(self, solver, scenario_name, solver_type):
        """Run a single solver test"""
        scenario = self.test_scenarios[scenario_name]

        print(f"    Running {solver_type} solver...")

        start_time = time.time()
        try:
            U, M, solve_info = solver.solve(
                Niter=scenario['max_iterations'], l2errBound=scenario['convergence_tolerance'], verbose=False
            )

            solve_time = time.time() - start_time

            # Calculate mass conservation error
            if M is not None and len(M) > 0:
                initial_mass = np.sum(M[0, :]) * self.problem_params.get(
                    'Dx', 1.0 / (self.problem_params.get('Nx', 25))
                )
                final_mass = np.sum(M[-1, :]) * self.problem_params.get('Dx', 1.0 / (self.problem_params.get('Nx', 25)))
                mass_error = abs(final_mass - initial_mass) / initial_mass * 100
            else:
                mass_error = float('inf')

            success = U is not None and M is not None
            converged = solve_info.get('converged', False)
            iterations = solve_info.get('iterations', 0)

            result = {
                'success': success,
                'converged': converged,
                'solve_time': solve_time,
                'iterations': iterations,
                'mass_error_percent': mass_error,
                'U': U,
                'M': M,
                'solve_info': solve_info,
            }

            print(f"      {solver_type}: {solve_time:.2f}s, success={success}, mass_error={mass_error:.2f}%")

            return result

        except Exception as e:
            solve_time = time.time() - start_time
            print(f"      {solver_type} FAILED: {e}")
            return {
                'success': False,
                'converged': False,
                'solve_time': solve_time,
                'iterations': 0,
                'mass_error_percent': float('inf'),
                'U': None,
                'M': None,
                'error': str(e),
            }

    def run_comparative_test(self, scenario_name):
        """Run comparative test between baseline and optimized solvers"""
        print(f"\n{'='*60}")
        print(f"TESTING SCENARIO: {scenario_name}")
        print(f"{'='*60}")

        scenario = self.test_scenarios[scenario_name]
        print(f"Parameters: {scenario}")

        # Setup problem
        problem, collocation_points, boundary_indices, boundary_conditions = self.setup_problem(scenario_name)

        # Test baseline solver
        print(f"  Setting up baseline solver...")
        try:
            baseline_solver = self.create_baseline_solver(
                problem, collocation_points, boundary_indices, boundary_conditions, scenario
            )
            baseline_result = self.run_solver_test(baseline_solver, scenario_name, "BASELINE")
        except Exception as e:
            print(f"  Baseline solver setup failed: {e}")
            baseline_result = {'success': False, 'error': str(e), 'solve_time': 0}

        # Test optimized solver
        print(f"  Setting up optimized solver...")
        try:
            optimized_solver, optimized_hjb_solver = self.create_optimized_solver(
                problem, collocation_points, boundary_indices, boundary_conditions, scenario
            )
            optimized_result = self.run_solver_test(optimized_solver, scenario_name, "OPTIMIZED")

            # Get performance statistics from optimized solver
            if hasattr(optimized_hjb_solver, 'get_performance_report'):
                performance_stats = optimized_hjb_solver.get_performance_report()
                optimized_result['performance_stats'] = performance_stats

                print(f"      QP Activation Rate: {performance_stats.get('qp_activation_rate', 0):.1%}")
                print(f"      QP Skip Rate: {performance_stats.get('qp_skip_rate', 0):.1%}")

        except Exception as e:
            print(f"  Optimized solver setup failed: {e}")
            optimized_result = {'success': False, 'error': str(e), 'solve_time': 0}

        # Calculate speedup
        if baseline_result['success'] and optimized_result['success']:
            speedup = baseline_result['solve_time'] / optimized_result['solve_time']
            print(f"  SPEEDUP: {speedup:.2f}x faster")

            # Solution quality comparison
            if baseline_result.get('U') is not None and optimized_result.get('U') is not None:
                u_diff = np.mean(np.abs(baseline_result['U'] - optimized_result['U']))
                m_diff = np.mean(np.abs(baseline_result['M'] - optimized_result['M']))
                print(f"  Solution Quality: U_diff={u_diff:.2e}, M_diff={m_diff:.2e}")
        else:
            speedup = None
            print(f"  SPEEDUP: Cannot calculate (one or both solvers failed)")

        # Store results
        self.results[scenario_name] = {
            'scenario': scenario,
            'baseline': baseline_result,
            'optimized': optimized_result,
            'speedup': speedup,
        }

        return self.results[scenario_name]

    def run_all_tests(self):
        """Run all test scenarios"""
        print("=" * 80)
        print("OPTIMIZED QP-COLLOCATION SOLVER COMPREHENSIVE TEST")
        print("=" * 80)
        print("Testing optimized solver against baseline implementation")

        for scenario_name in self.test_scenarios:
            self.run_comparative_test(scenario_name)

        # Summary
        self.print_summary()
        return self.results

    def print_summary(self):
        """Print comprehensive test summary"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*80}")

        successful_tests = 0
        total_speedups = []
        quality_preserved = 0

        print(
            "\n{:<20} {:<15} {:<15} {:<12} {:<12} {:<10}".format(
                "Scenario", "Baseline", "Optimized", "Speedup", "Quality", "Status"
            )
        )
        print("-" * 85)

        for scenario_name, result in self.results.items():
            baseline = result['baseline']
            optimized = result['optimized']
            speedup = result.get('speedup')

            # Status indicators
            baseline_status = "✓" if baseline['success'] else "✗"
            optimized_status = "✓" if optimized['success'] else "✗"

            speedup_str = f"{speedup:.1f}x" if speedup else "N/A"

            # Quality check
            quality_status = "Good"
            if baseline['success'] and optimized['success']:
                mass_diff = abs(baseline['mass_error_percent'] - optimized['mass_error_percent'])
                if mass_diff < 5.0:  # Within 5% mass conservation difference
                    quality_preserved += 1
                    quality_status = "Good"
                else:
                    quality_status = "Poor"
            else:
                quality_status = "N/A"

            overall_status = (
                "PASS" if (baseline['success'] and optimized['success'] and speedup and speedup > 1.0) else "FAIL"
            )

            print(
                "{:<20} {:<15} {:<15} {:<12} {:<12} {:<10}".format(
                    scenario_name[:19],
                    f"{baseline['solve_time']:.1f}s {baseline_status}",
                    f"{optimized['solve_time']:.1f}s {optimized_status}",
                    speedup_str,
                    quality_status,
                    overall_status,
                )
            )

            if speedup:
                total_speedups.append(speedup)
            if baseline['success'] and optimized['success']:
                successful_tests += 1

        print("-" * 85)

        # Overall statistics
        print(f"\nOVERALL RESULTS:")
        print(f"  Successful Tests: {successful_tests}/{len(self.results)}")
        print(f"  Quality Preserved: {quality_preserved}/{successful_tests if successful_tests > 0 else 1}")

        if total_speedups:
            avg_speedup = np.mean(total_speedups)
            max_speedup = np.max(total_speedups)
            min_speedup = np.min(total_speedups)
            print(f"  Average Speedup: {avg_speedup:.1f}x")
            print(f"  Maximum Speedup: {max_speedup:.1f}x")
            print(f"  Minimum Speedup: {min_speedup:.1f}x")

        # Detailed performance analysis
        print(f"\nDETAILED PERFORMANCE ANALYSIS:")
        for scenario_name, result in self.results.items():
            if 'performance_stats' in result['optimized']:
                stats = result['optimized']['performance_stats']
                print(f"\n{scenario_name}:")
                print(f"  QP Activation Rate: {stats.get('qp_activation_rate', 0):.1%}")
                print(f"  QP Calls Skipped: {stats.get('qp_calls_skipped', 0)}/{stats.get('total_qp_calls', 0)}")
                print(f"  Batch QP Calls: {stats.get('batch_qp_calls', 0)}")
                print(f"  Warm Start Successes: {stats.get('warm_start_successes', 0)}")

    def create_performance_plots(self):
        """Create performance comparison plots"""
        if not self.results:
            print("No results to plot")
            return

        # Extract data for plotting
        scenarios = list(self.results.keys())
        baseline_times = []
        optimized_times = []
        speedups = []

        for scenario_name in scenarios:
            result = self.results[scenario_name]
            if result['baseline']['success'] and result['optimized']['success']:
                baseline_times.append(result['baseline']['solve_time'])
                optimized_times.append(result['optimized']['solve_time'])
                speedups.append(result['speedup'])
            else:
                baseline_times.append(None)
                optimized_times.append(None)
                speedups.append(None)

        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Solve Time Comparison
        valid_indices = [
            i for i, (b, o) in enumerate(zip(baseline_times, optimized_times)) if b is not None and o is not None
        ]

        if valid_indices:
            x_pos = np.arange(len(valid_indices))
            valid_scenarios = [scenarios[i] for i in valid_indices]
            valid_baseline = [baseline_times[i] for i in valid_indices]
            valid_optimized = [optimized_times[i] for i in valid_indices]

            width = 0.35
            ax1.bar(x_pos - width / 2, valid_baseline, width, label='Baseline', alpha=0.8, color='red')
            ax1.bar(x_pos + width / 2, valid_optimized, width, label='Optimized', alpha=0.8, color='green')
            ax1.set_xlabel('Test Scenarios')
            ax1.set_ylabel('Solve Time (seconds)')
            ax1.set_title('Solve Time Comparison')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(valid_scenarios, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Plot 2: Speedup Chart
        valid_speedups = [s for s in speedups if s is not None]
        valid_speedup_scenarios = [scenarios[i] for i, s in enumerate(speedups) if s is not None]

        if valid_speedups:
            ax2.bar(range(len(valid_speedups)), valid_speedups, color='blue', alpha=0.7)
            ax2.set_xlabel('Test Scenarios')
            ax2.set_ylabel('Speedup Factor')
            ax2.set_title('Optimization Speedup by Scenario')
            ax2.set_xticks(range(len(valid_speedup_scenarios)))
            ax2.set_xticklabels(valid_speedup_scenarios, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)

            # Add speedup target line
            target_speedup = 10.0  # Minimum target from analysis
            ax2.axhline(y=target_speedup, color='red', linestyle='--', label=f'Target: {target_speedup}x')
            ax2.legend()

        # Plot 3: Mass Conservation Comparison
        baseline_mass_errors = []
        optimized_mass_errors = []

        for result in self.results.values():
            if result['baseline']['success'] and result['optimized']['success']:
                baseline_mass_errors.append(result['baseline']['mass_error_percent'])
                optimized_mass_errors.append(result['optimized']['mass_error_percent'])

        if baseline_mass_errors and optimized_mass_errors:
            ax3.scatter(baseline_mass_errors, optimized_mass_errors, alpha=0.7, s=50)
            ax3.plot(
                [0, max(max(baseline_mass_errors), max(optimized_mass_errors))],
                [0, max(max(baseline_mass_errors), max(optimized_mass_errors))],
                'r--',
                alpha=0.5,
                label='Perfect Match',
            )
            ax3.set_xlabel('Baseline Mass Error (%)')
            ax3.set_ylabel('Optimized Mass Error (%)')
            ax3.set_title('Mass Conservation Quality Comparison')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Plot 4: QP Activation Statistics
        qp_activation_rates = []
        scenario_labels = []

        for scenario_name, result in self.results.items():
            if 'performance_stats' in result['optimized']:
                stats = result['optimized']['performance_stats']
                qp_activation_rates.append(stats.get('qp_activation_rate', 0) * 100)
                scenario_labels.append(scenario_name)

        if qp_activation_rates:
            ax4.bar(range(len(qp_activation_rates)), qp_activation_rates, color='orange', alpha=0.7)
            ax4.set_xlabel('Test Scenarios')
            ax4.set_ylabel('QP Activation Rate (%)')
            ax4.set_title('QP Constraint Activation Rate')
            ax4.set_xticks(range(len(scenario_labels)))
            ax4.set_xticklabels(scenario_labels, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)

            # Add expected activation rate line
            expected_rate = 10.0  # From experimental analysis
            ax4.axhline(y=expected_rate, color='red', linestyle='--', label=f'Expected: {expected_rate}%')
            ax4.legend()

        plt.tight_layout()
        plt.savefig(
            '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/tests/method_comparisons/optimized_qp_solver_performance.png',
            dpi=300,
            bbox_inches='tight',
        )
        print(f"\nPerformance plots saved to: optimized_qp_solver_performance.png")
        plt.show()


def main():
    """Main test execution"""

    # Base problem parameters
    problem_params = {'xmin': 0.0, 'xmax': 1.0, 'T': 1.0, 'sigma': 0.15, 'coefCT': 0.02}

    # Test scenarios focusing on optimization validation
    test_scenarios = {
        'validation_test': {
            'Nx': 20,
            'Nt': 25,
            'num_collocation_points': 8,
            'num_particles': 100,
            'delta': 0.35,
            'taylor_order': 2,
            'newton_iterations': 3,
            'newton_tolerance': 1e-3,
            'max_iterations': 4,
            'convergence_tolerance': 1e-2,
        }
    }

    # Create tester and run tests
    tester = OptimizedQPSolverTester(problem_params, test_scenarios)

    try:
        results = tester.run_all_tests()

        # Create performance plots
        tester.create_performance_plots()

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
    print("Starting Optimized QP-Collocation Solver Test...")
    print("Expected execution time: 5-15 minutes")
    print("This will test the optimized solver against baseline implementation")

    results = main()

    if results:
        print("\n" + "=" * 80)
        print("OPTIMIZED QP-COLLOCATION SOLVER TEST COMPLETED")
        print("=" * 80)
        print("Check the performance plots and summary above for results.")
        print("The optimized solver should show significant speedup while maintaining solution quality.")
    else:
        print("\nTest did not complete successfully.")
