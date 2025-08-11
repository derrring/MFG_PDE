#!/usr/bin/env python3
"""
QP-Collocation Stable Long-Time Series
Using proven T=2 parameters to extend to T=5, T=10 for long-time behavior study.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def run_stable_long_time_series():
    """Run a series of long-time simulations using stable T=2 parameters"""
    print("=" * 80)
    print("QP-COLLOCATION STABLE LONG-TIME SERIES")
    print("=" * 80)
    print("Using proven T=2 parameters to study T=5, T=10 behavior")
    print("Investigating why some simulations suddenly fail vs. remain stable")

    # Base stable parameters from successful T=2 simulation
    base_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 50,  # Proven stable spatial resolution
        "sigma": 0.2,  # Proven stable diffusion
        "coefCT": 0.03,  # Proven stable coupling
    }

    # QP solver parameters that worked for T=2
    stable_solver_params = {
        "num_particles": 800,  # Proven particle count
        "delta": 0.3,  # Proven neighborhood size
        "taylor_order": 2,  # Second-order accuracy
        "weight_function": "wendland",  # Proven stable weight
        "NiterNewton": 8,  # Newton iterations
        "l2errBoundNewton": 1e-4,  # Newton tolerance
        "kde_bandwidth": "scott",  # Adaptive bandwidth
        "normalize_kde_output": False,  # No artificial normalization
        "use_monotone_constraints": True,  # QP constraints
    }

    # Collocation setup
    num_collocation_points = 12
    no_flux_bc = BoundaryConditions(type="no_flux")

    # Test series: T=3, T=5, T=10
    test_series = [
        {"T": 3.0, "Nt": 150, "name": "T=3 (Extended)"},
        {"T": 5.0, "Nt": 250, "name": "T=5 (Long-Time)"},
        {"T": 10.0, "Nt": 500, "name": "T=10 (Very Long-Time)"},
    ]

    results = {}

    for test_config in test_series:
        T = test_config["T"]
        Nt = test_config["Nt"]
        name = test_config["name"]

        print(f"\n{'='*60}")
        print(f"TESTING: {name}")
        print(f"{'='*60}")

        # Create problem parameters
        problem_params = base_params.copy()
        problem_params.update({"T": T, "Nt": Nt})

        print(f"Problem Parameters for {name}:")
        for key, value in problem_params.items():
            print(f"  {key}: {value}")

        try:
            # Create problem
            problem = ExampleMFGProblem(**problem_params)

            print(f"\nProblem setup:")
            print(f"  Domain: [{problem.xmin}, {problem.xmax}] × [0, {problem.T}]")
            print(f"  Grid: Dx = {problem.Dx:.4f}, Dt = {problem.Dt:.4f}")
            print(f"  CFL number: {problem_params['sigma']**2 * problem.Dt / problem.Dx**2:.4f}")
            print(f"  Total time steps: {problem.Nt}")

            # Setup collocation
            collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)
            boundary_indices = [0, num_collocation_points - 1]

            # Create solver
            print(f"  Creating QP solver for {name}...")
            solver = ParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                boundary_indices=np.array(boundary_indices),
                boundary_conditions=no_flux_bc,
                **stable_solver_params,
            )

            # Solve with monitoring
            max_iterations = 15
            convergence_tolerance = 1e-3

            print(f"  Starting {name} simulation...")
            print(f"  Expected time: {T*2:.1f}-{T*4:.1f} minutes")

            start_time = time.time()

            U_solution, M_solution, solve_info = solver.solve(
                Niter=max_iterations, l2errBound=convergence_tolerance, verbose=True
            )

            total_time = time.time() - start_time
            iterations_run = solve_info.get("iterations", max_iterations)
            converged = solve_info.get("converged", False)

            print(f"\n  {name} COMPLETED")
            print(f"  Execution time: {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"  Iterations: {iterations_run}/{max_iterations}")
            print(f"  Converged: {converged}")

            if U_solution is not None and M_solution is not None:
                # Analyze results
                mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
                initial_mass = mass_evolution[0]
                final_mass = mass_evolution[-1]
                mass_change = final_mass - initial_mass
                mass_change_percent = (mass_change / initial_mass) * 100

                # Check for sudden mass loss (cliff detection)
                mass_diff = np.diff(mass_evolution)
                max_single_loss = np.min(mass_diff)
                max_single_loss_percent = (max_single_loss / initial_mass) * 100
                cliff_detected = max_single_loss_percent < -20  # More than 20% loss in one step

                # Particle violations
                particles_trajectory = solver.fp_solver.M_particles_trajectory
                total_violations = 0
                if particles_trajectory is not None:
                    for t_step in range(particles_trajectory.shape[0]):
                        step_particles = particles_trajectory[t_step, :]
                        violations = np.sum(
                            (step_particles < problem.xmin - 1e-10) | (step_particles > problem.xmax + 1e-10)
                        )
                        total_violations += violations

                # Solution quality
                negative_densities = np.sum(M_solution < -1e-10)
                max_U = np.max(np.abs(U_solution))
                min_M = np.min(M_solution)

                # Store results
                results[name] = {
                    'success': True,
                    'T': T,
                    'time': total_time,
                    'iterations': iterations_run,
                    'converged': converged,
                    'mass_initial': initial_mass,
                    'mass_final': final_mass,
                    'mass_change_percent': mass_change_percent,
                    'cliff_detected': cliff_detected,
                    'max_single_loss_percent': max_single_loss_percent,
                    'total_violations': total_violations,
                    'negative_densities': negative_densities,
                    'max_U': max_U,
                    'min_M': min_M,
                    'arrays': {
                        'mass_evolution': mass_evolution,
                        'M_solution': M_solution,
                        'U_solution': U_solution,
                        'tSpace': problem.tSpace,
                        'xSpace': problem.xSpace,
                    },
                }

                print(f"  Mass: {initial_mass:.6f} → {final_mass:.6f} ({mass_change_percent:+.3f}%)")
                print(f"  Max single-step loss: {max_single_loss_percent:+.3f}%")
                print(f"  Cliff detected: {cliff_detected}")
                print(f"  Boundary violations: {total_violations}")
                print(f"  Negative densities: {negative_densities}")

                if cliff_detected:
                    print(f"  ⚠️  CLIFF DETECTED: Sudden mass loss > 20% in one step!")
                    # Find when the cliff occurred
                    cliff_step = np.argmin(mass_diff)
                    cliff_time = problem.tSpace[cliff_step]
                    print(f"      Cliff occurred at t = {cliff_time:.3f} (step {cliff_step})")
                    print(f"      Mass before cliff: {mass_evolution[cliff_step]:.6f}")
                    print(f"      Mass after cliff: {mass_evolution[cliff_step+1]:.6f}")
                else:
                    print(f"  ✅ NO CLIFF: Stable mass evolution throughout simulation")

            else:
                results[name] = {'success': False, 'T': T}
                print(f"  ❌ {name} FAILED - No solution produced")

        except Exception as e:
            results[name] = {'success': False, 'T': T, 'error': str(e)}
            print(f"  ❌ {name} CRASHED: {e}")

    # Comprehensive Analysis
    print(f"\n{'='*80}")
    print("STABLE LONG-TIME SERIES ANALYSIS")
    print(f"{'='*80}")

    analyze_long_time_series(results)
    create_long_time_series_plots(results)

    return results


def analyze_long_time_series(results):
    """Analyze the long-time series results"""
    successful_tests = [name for name, result in results.items() if result.get('success', False)]

    print(f"Successful simulations: {len(successful_tests)}/{len(results)}")

    if not successful_tests:
        print("No successful simulations to analyze")
        return

    # Summary table
    print(f"\n{'Simulation':<20} {'T':<5} {'Mass Change %':<12} {'Cliff?':<8} {'Violations':<12} {'Time (min)':<10}")
    print(f"{'-'*20} {'-'*5} {'-'*12} {'-'*8} {'-'*12} {'-'*10}")

    for name in successful_tests:
        result = results[name]
        T = result['T']
        mass_change = result['mass_change_percent']
        cliff = "YES" if result['cliff_detected'] else "NO"
        violations = result['total_violations']
        time_min = result['time'] / 60

        print(f"{name:<20} {T:<5.1f} {mass_change:<+12.3f} {cliff:<8} {violations:<12} {time_min:<10.1f}")

    # Stability analysis
    print(f"\n--- STABILITY ANALYSIS ---")

    stable_sims = [name for name in successful_tests if not results[name]['cliff_detected']]
    unstable_sims = [name for name in successful_tests if results[name]['cliff_detected']]

    print(f"Stable simulations: {len(stable_sims)}")
    print(f"Unstable simulations (cliff detected): {len(unstable_sims)}")

    if stable_sims:
        print(f"\nStable simulations show:")
        for name in stable_sims:
            result = results[name]
            print(
                f"  {name}: {result['mass_change_percent']:+.3f}% mass change, {result['total_violations']} violations"
            )

    if unstable_sims:
        print(f"\nUnstable simulations show:")
        for name in unstable_sims:
            result = results[name]
            print(f"  {name}: {result['max_single_loss_percent']:+.3f}% max single-step loss")

    # Mass conservation trends
    print(f"\n--- MASS CONSERVATION TRENDS ---")
    T_values = [results[name]['T'] for name in successful_tests]
    mass_changes = [results[name]['mass_change_percent'] for name in successful_tests]

    if len(T_values) > 1:
        correlation = np.corrcoef(T_values, mass_changes)[0, 1]
        print(f"Correlation between T and mass change: {correlation:.3f}")

        if correlation > 0.5:
            print("  → Longer simulations tend to accumulate more mass (expected with no-flux BC)")
        elif correlation < -0.5:
            print("  → Longer simulations tend to lose mass (potential numerical issues)")
        else:
            print("  → No clear trend between T and mass change")

    # Performance trends
    print(f"\n--- PERFORMANCE TRENDS ---")
    times = [results[name]['time'] for name in successful_tests]
    avg_time_per_T = [results[name]['time'] / results[name]['T'] for name in successful_tests]

    print(f"Average execution time per unit T: {np.mean(avg_time_per_T):.1f} seconds")
    print(f"Time scaling with T: {np.mean(times) / np.mean(T_values) if T_values else 0:.1f}s per T unit")


def create_long_time_series_plots(results):
    """Create plots comparing the long-time series"""
    successful_tests = [name for name, result in results.items() if result.get('success', False)]

    if len(successful_tests) < 2:
        print("Insufficient successful results for plotting")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('QP-Collocation Stable Long-Time Series Analysis', fontsize=16, fontweight='bold')

    colors = ['blue', 'green', 'red', 'purple', 'orange']

    # 1. Mass evolution comparison
    ax1 = axes[0, 0]
    for i, name in enumerate(successful_tests):
        result = results[name]
        arrays = result['arrays']
        color = colors[i % len(colors)]
        linestyle = '--' if result['cliff_detected'] else '-'
        linewidth = 3 if result['cliff_detected'] else 2

        ax1.plot(
            arrays['tSpace'],
            arrays['mass_evolution'],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=f"{name} ({result['mass_change_percent']:+.1f}%)",
            alpha=0.8,
        )

    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Total Mass')
    ax1.set_title('Mass Evolution Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Mass change percentage vs. T
    ax2 = axes[0, 1]
    T_vals = [results[name]['T'] for name in successful_tests]
    mass_changes = [results[name]['mass_change_percent'] for name in successful_tests]
    cliff_status = [results[name]['cliff_detected'] for name in successful_tests]

    for i, (T, mc, cliff) in enumerate(zip(T_vals, mass_changes, cliff_status)):
        color = 'red' if cliff else 'green'
        marker = 'x' if cliff else 'o'
        size = 100 if cliff else 80
        ax2.scatter(T, mc, color=color, marker=marker, s=size, alpha=0.7)
        ax2.text(T, mc + 0.1, successful_tests[i].split()[0], ha='center', fontsize=9)

    ax2.set_xlabel('Time Horizon T')
    ax2.set_ylabel('Mass Change (%)')
    ax2.set_title('Mass Change vs. Time Horizon')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # 3. Final density profiles
    ax3 = axes[0, 2]
    for i, name in enumerate(successful_tests):
        result = results[name]
        arrays = result['arrays']
        color = colors[i % len(colors)]
        linestyle = '--' if result['cliff_detected'] else '-'
        alpha = 0.6 if result['cliff_detected'] else 0.8

        ax3.plot(
            arrays['xSpace'],
            arrays['M_solution'][-1, :],
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=f"{name}",
            alpha=alpha,
        )

    ax3.set_xlabel('Space x')
    ax3.set_ylabel('Final Density')
    ax3.set_title('Final Density Profiles')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Execution time vs. T
    ax4 = axes[1, 0]
    times = [results[name]['time'] / 60 for name in successful_tests]  # Convert to minutes

    for i, (T, time, cliff) in enumerate(zip(T_vals, times, cliff_status)):
        color = 'red' if cliff else 'blue'
        marker = 'x' if cliff else 'o'
        ax4.scatter(T, time, color=color, marker=marker, s=80, alpha=0.7)
        ax4.text(T, time + 0.2, f"{time:.1f}min", ha='center', fontsize=9)

    ax4.set_xlabel('Time Horizon T')
    ax4.set_ylabel('Execution Time (minutes)')
    ax4.set_title('Performance vs. Time Horizon')
    ax4.grid(True, alpha=0.3)

    # 5. Solution quality metrics
    ax5 = axes[1, 1]
    violations = [results[name]['total_violations'] for name in successful_tests]
    max_Us = [results[name]['max_U'] for name in successful_tests]

    # Dual y-axis plot
    ax5_twin = ax5.twinx()

    bars1 = ax5.bar([f"T={T}" for T in T_vals], violations, alpha=0.7, color='orange', label='Violations')
    line1 = ax5_twin.plot(range(len(T_vals)), max_Us, 'ro-', linewidth=2, label='Max |U|')

    ax5.set_xlabel('Simulation')
    ax5.set_ylabel('Boundary Violations', color='orange')
    ax5_twin.set_ylabel('Max |U|', color='red')
    ax5.set_title('Solution Quality Metrics')
    ax5.grid(True, alpha=0.3)

    # 6. Stability summary
    ax6 = axes[1, 2]

    stable_count = len([name for name in successful_tests if not results[name]['cliff_detected']])
    unstable_count = len(successful_tests) - stable_count

    labels = ['Stable', 'Unstable (Cliff)']
    sizes = [stable_count, unstable_count]
    colors_pie = ['green', 'red']

    wedges, texts, autotexts = ax6.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax6.set_title('Stability Assessment')

    plt.tight_layout()
    plt.savefig(
        '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/qp_stable_long_time_series.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.show()


if __name__ == "__main__":
    print("Starting QP-Collocation Stable Long-Time Series...")
    print("Testing T=3, T=5, T=10 using proven stable T=2 parameters")
    print("Expected total execution time: 15-45 minutes")

    try:
        results = run_stable_long_time_series()
        print("\n" + "=" * 80)
        print("STABLE LONG-TIME SERIES COMPLETED")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\nSeries interrupted by user.")
    except Exception as e:
        print(f"\nSeries failed: {e}")
        import traceback

        traceback.print_exc()
