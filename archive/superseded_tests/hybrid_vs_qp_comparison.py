#!/usr/bin/env python3
"""
Hybrid vs QP-Collocation Method Comparison
Focused comparison of the two particle-based methods using identical parameters.
"""

import importlib.util
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# Add the main package to path
sys.path.insert(0, '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE')

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def load_hybrid_solver():
    """Load existing Hybrid solver from tests directory"""
    test_dir = '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/tests'

    # Load hybrid_fdm module
    hybrid_spec = importlib.util.spec_from_file_location("hybrid_fdm", os.path.join(test_dir, "hybrid_fdm.py"))
    hybrid_module = importlib.util.module_from_spec(hybrid_spec)

    # Add required imports to module
    sys.modules['hybrid_fdm'] = hybrid_module

    # Add the base solver to module before executing
    from mfg_pde.alg.base_mfg_solver import MFGSolver

    hybrid_module.MFGSolver = MFGSolver

    try:
        hybrid_spec.loader.exec_module(hybrid_module)
        print("✓ Successfully loaded Hybrid solver")
        return hybrid_module.ParticleSolver  # Note: it's called ParticleSolver in the file
    except Exception as e:
        print(f"❌ Failed to load Hybrid solver: {e}")
        return None


def compare_hybrid_vs_qp():
    """Compare Hybrid and QP-Collocation methods with identical parameters"""
    print("=" * 80)
    print("HYBRID vs QP-COLLOCATION COMPARISON")
    print("=" * 80)
    print("Focused comparison of two particle-based methods")
    print("Using identical problem parameters and particle counts")

    # Load Hybrid solver
    print("\nLoading Hybrid solver...")
    HybridSolver = load_hybrid_solver()

    if HybridSolver is None:
        print("❌ Cannot proceed without Hybrid solver")
        return None

    # Unified problem parameters
    problem_params = {"xmin": 0.0, "xmax": 1.0, "Nx": 25, "T": 0.25, "Nt": 12, "sigma": 0.15, "coefCT": 0.015}

    # Shared settings for both methods
    shared_settings = {
        "num_particles": 800,
        "newton_iterations": 8,
        "picard_iterations": 12,
        "newton_tolerance": 1e-4,
        "picard_tolerance": 1e-4,
    }

    print(f"\nProblem Parameters:")
    for key, value in problem_params.items():
        print(f"  {key}: {value}")

    print(f"\nShared Settings:")
    for key, value in shared_settings.items():
        print(f"  {key}: {value}")

    problem = ExampleMFGProblem(**problem_params)
    no_flux_bc = BoundaryConditions(type="no_flux")

    print(f"\nProblem setup:")
    print(f"  Domain: [{problem.xmin}, {problem.xmax}] × [0, {problem.T}]")
    print(f"  Grid: Dx = {problem.Dx:.4f}, Dt = {problem.Dt:.4f}")
    print(f"  CFL: {problem_params['sigma']**2 * problem.Dt / problem.Dx**2:.4f}")

    results = {}

    # Method 1: Hybrid Particle-FDM
    print(f"\n{'='*60}")
    print("METHOD 1: HYBRID PARTICLE-FDM")
    print(f"{'='*60}")

    try:
        start_time = time.time()

        hybrid_solver = HybridSolver(
            problem=problem,
            num_particles=shared_settings["num_particles"],
            particle_thetaUM=0.5,
            kde_bandwidth="scott",
            NiterNewton=shared_settings["newton_iterations"],
            l2errBoundNewton=shared_settings["newton_tolerance"],
        )

        print(f"  Starting Hybrid solve with {shared_settings['num_particles']} particles...")
        U_hybrid, M_hybrid, iterations_hybrid, l2dist_u_h, l2dist_m_h = hybrid_solver.solve(
            Niter=shared_settings["picard_iterations"], l2errBoundPicard=shared_settings["picard_tolerance"]
        )

        hybrid_time = time.time() - start_time

        if M_hybrid is not None and U_hybrid is not None:
            # Analysis
            mass_evolution_hybrid = np.sum(M_hybrid * problem.Dx, axis=1)
            initial_mass_hybrid = mass_evolution_hybrid[0]
            final_mass_hybrid = mass_evolution_hybrid[-1]
            mass_change_hybrid = final_mass_hybrid - initial_mass_hybrid

            center_of_mass_hybrid = np.sum(problem.xSpace * M_hybrid[-1, :]) * problem.Dx
            max_density_idx_hybrid = np.argmax(M_hybrid[-1, :])
            max_density_loc_hybrid = problem.xSpace[max_density_idx_hybrid]
            final_density_peak_hybrid = M_hybrid[-1, max_density_idx_hybrid]

            # Boundary violations
            violations_hybrid = 0
            if hasattr(hybrid_solver, 'M_particles') and hybrid_solver.M_particles is not None:
                final_particles = hybrid_solver.M_particles[-1, :]
                violations_hybrid = np.sum(
                    (final_particles < problem.xmin - 1e-10) | (final_particles > problem.xmax + 1e-10)
                )

            results['hybrid'] = {
                'success': True,
                'method_name': 'Hybrid Particle-FDM',
                'time': hybrid_time,
                'iterations': iterations_hybrid,
                'mass_conservation': {
                    'initial_mass': initial_mass_hybrid,
                    'final_mass': final_mass_hybrid,
                    'mass_change': mass_change_hybrid,
                    'mass_change_percent': (mass_change_hybrid / initial_mass_hybrid) * 100,
                },
                'physical_observables': {
                    'center_of_mass': center_of_mass_hybrid,
                    'max_density_location': max_density_loc_hybrid,
                    'final_density_peak': final_density_peak_hybrid,
                },
                'solution_quality': {
                    'max_U': np.max(np.abs(U_hybrid)),
                    'min_M': np.min(M_hybrid),
                    'negative_densities': np.sum(M_hybrid < -1e-10),
                    'violations': violations_hybrid,
                },
                'arrays': {'U_solution': U_hybrid, 'M_solution': M_hybrid, 'mass_evolution': mass_evolution_hybrid},
            }

            print(f"  ✓ Hybrid completed in {hybrid_time:.2f}s")
            print(f"    Iterations: {iterations_hybrid}")
            print(
                f"    Mass: {initial_mass_hybrid:.6f} → {final_mass_hybrid:.6f} ({(mass_change_hybrid/initial_mass_hybrid)*100:+.3f}%)"
            )
            print(f"    Center of mass: {center_of_mass_hybrid:.4f}")
            print(f"    Max density: {final_density_peak_hybrid:.3f} at x = {max_density_loc_hybrid:.4f}")
            print(f"    Boundary violations: {violations_hybrid}")

        else:
            results['hybrid'] = {'success': False, 'method_name': 'Hybrid Particle-FDM'}
            print(f"  ❌ Hybrid failed to produce solution")

    except Exception as e:
        results['hybrid'] = {'success': False, 'method_name': 'Hybrid Particle-FDM', 'error': str(e)}
        print(f"  ❌ Hybrid crashed: {e}")
        import traceback

        traceback.print_exc()

    # Method 2: QP Particle-Collocation
    print(f"\n{'='*60}")
    print("METHOD 2: QP PARTICLE-COLLOCATION")
    print(f"{'='*60}")

    try:
        start_time = time.time()

        # Setup collocation points
        num_collocation_points = 10
        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)
        boundary_indices = [0, num_collocation_points - 1]

        qp_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=shared_settings["num_particles"],
            delta=0.3,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=shared_settings["newton_iterations"],
            l2errBoundNewton=shared_settings["newton_tolerance"],
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
        )

        print(
            f"  Starting QP solve with {shared_settings['num_particles']} particles, {num_collocation_points} collocation points..."
        )
        U_qp, M_qp, info_qp = qp_solver.solve(
            Niter=shared_settings["picard_iterations"], l2errBound=shared_settings["picard_tolerance"], verbose=False
        )

        qp_time = time.time() - start_time

        if M_qp is not None and U_qp is not None:
            # Analysis
            mass_evolution_qp = np.sum(M_qp * problem.Dx, axis=1)
            initial_mass_qp = mass_evolution_qp[0]
            final_mass_qp = mass_evolution_qp[-1]
            mass_change_qp = final_mass_qp - initial_mass_qp

            center_of_mass_qp = np.sum(problem.xSpace * M_qp[-1, :]) * problem.Dx
            max_density_idx_qp = np.argmax(M_qp[-1, :])
            max_density_loc_qp = problem.xSpace[max_density_idx_qp]
            final_density_peak_qp = M_qp[-1, max_density_idx_qp]

            # Boundary violations
            violations_qp = 0
            particles_traj = qp_solver.get_particles_trajectory()
            if particles_traj is not None:
                final_particles = particles_traj[-1, :]
                violations_qp = np.sum(
                    (final_particles < problem.xmin - 1e-10) | (final_particles > problem.xmax + 1e-10)
                )

            results['qp'] = {
                'success': True,
                'method_name': 'QP Particle-Collocation',
                'time': qp_time,
                'iterations': info_qp.get('iterations', 0),
                'converged': info_qp.get('converged', False),
                'mass_conservation': {
                    'initial_mass': initial_mass_qp,
                    'final_mass': final_mass_qp,
                    'mass_change': mass_change_qp,
                    'mass_change_percent': (mass_change_qp / initial_mass_qp) * 100,
                },
                'physical_observables': {
                    'center_of_mass': center_of_mass_qp,
                    'max_density_location': max_density_loc_qp,
                    'final_density_peak': final_density_peak_qp,
                },
                'solution_quality': {
                    'max_U': np.max(np.abs(U_qp)),
                    'min_M': np.min(M_qp),
                    'negative_densities': np.sum(M_qp < -1e-10),
                    'violations': violations_qp,
                },
                'arrays': {'U_solution': U_qp, 'M_solution': M_qp, 'mass_evolution': mass_evolution_qp},
            }

            print(f"  ✓ QP completed in {qp_time:.2f}s")
            print(f"    Iterations: {info_qp.get('iterations', 0)}")
            print(f"    Converged: {info_qp.get('converged', False)}")
            print(
                f"    Mass: {initial_mass_qp:.6f} → {final_mass_qp:.6f} ({(mass_change_qp/initial_mass_qp)*100:+.3f}%)"
            )
            print(f"    Center of mass: {center_of_mass_qp:.4f}")
            print(f"    Max density: {final_density_peak_qp:.3f} at x = {max_density_loc_qp:.4f}")
            print(f"    Boundary violations: {violations_qp}")

        else:
            results['qp'] = {'success': False, 'method_name': 'QP Particle-Collocation'}
            print(f"  ❌ QP failed to produce solution")

    except Exception as e:
        results['qp'] = {'success': False, 'method_name': 'QP Particle-Collocation', 'error': str(e)}
        print(f"  ❌ QP crashed: {e}")
        import traceback

        traceback.print_exc()

    # Analysis
    print(f"\n{'='*80}")
    print("HYBRID vs QP-COLLOCATION ANALYSIS")
    print(f"{'='*80}")

    analyze_hybrid_qp_comparison(results, problem)
    create_hybrid_qp_plots(results, problem)

    return results


def analyze_hybrid_qp_comparison(results, problem):
    """Analyze differences between Hybrid and QP-Collocation methods"""
    successful_methods = [method for method in ['hybrid', 'qp'] if results.get(method, {}).get('success', False)]

    print(f"Successful methods: {len(successful_methods)}/2")

    if len(successful_methods) != 2:
        print("Cannot perform full comparison - not all methods succeeded")
        for method in ['hybrid', 'qp']:
            if method not in successful_methods:
                error = results.get(method, {}).get('error', 'Failed')
                print(f"  Failed: {results.get(method, {}).get('method_name', method)} - {error}")
        return

    hybrid = results['hybrid']
    qp = results['qp']

    # Summary table
    print(f"\n{'Metric':<25} {'Hybrid':<15} {'QP-Collocation':<15} {'Difference':<15}")
    print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15}")

    # Key metrics comparison
    metrics = [
        ('Final Mass', 'mass_conservation', 'final_mass', lambda x: f"{x:.6f}"),
        ('Mass Change %', 'mass_conservation', 'mass_change_percent', lambda x: f"{x:+.2f}%"),
        ('Center of Mass', 'physical_observables', 'center_of_mass', lambda x: f"{x:.4f}"),
        ('Max Density Location', 'physical_observables', 'max_density_location', lambda x: f"{x:.4f}"),
        ('Peak Density Value', 'physical_observables', 'final_density_peak', lambda x: f"{x:.3f}"),
        ('Max |U|', 'solution_quality', 'max_U', lambda x: f"{x:.2e}"),
        ('Min M', 'solution_quality', 'min_M', lambda x: f"{x:.2e}"),
        ('Boundary Violations', 'solution_quality', 'violations', lambda x: str(x)),
        ('Negative Densities', 'solution_quality', 'negative_densities', lambda x: str(x)),
        ('Execution Time (s)', None, 'time', lambda x: f"{x:.2f}"),
        ('Iterations', None, 'iterations', lambda x: str(x)),
    ]

    for metric_name, category, key, fmt in metrics:
        if category is None:
            hybrid_val = hybrid[key]
            qp_val = qp[key]
        else:
            hybrid_val = hybrid[category][key]
            qp_val = qp[category][key]

        # Calculate difference
        if isinstance(hybrid_val, (int, float)) and isinstance(qp_val, (int, float)):
            if 'percent' in key.lower() or 'change' in key.lower():
                diff = abs(qp_val - hybrid_val)
                diff_str = f"{diff:.2f}pp"
            elif 'time' in key.lower():
                diff = abs(qp_val - hybrid_val)
                diff_str = f"{diff:.2f}s"
            elif 'max_u' in key.lower() or 'min_m' in key.lower():
                diff = abs(qp_val - hybrid_val)
                diff_str = f"{diff:.1e}"
            else:
                diff = abs(qp_val - hybrid_val)
                diff_str = f"{diff:.4f}"
        else:
            diff_str = (
                f"{abs(int(qp_val) - int(hybrid_val))}"
                if str(hybrid_val).isdigit() and str(qp_val).isdigit()
                else "N/A"
            )

        print(f"{metric_name:<25} {fmt(hybrid_val):<15} {fmt(qp_val):<15} {diff_str:<15}")

    # Detailed convergence analysis
    print(f"\n--- CONVERGENCE ANALYSIS ---")

    # Mass conservation consistency
    hybrid_mass_change = hybrid['mass_conservation']['mass_change_percent']
    qp_mass_change = qp['mass_conservation']['mass_change_percent']
    mass_diff = abs(qp_mass_change - hybrid_mass_change)

    print(f"Mass change difference: {mass_diff:.2f} percentage points")

    if mass_diff < 0.5:
        print("✅ Excellent mass conservation consistency")
    elif mass_diff < 2.0:
        print("✅ Good mass conservation consistency")
    elif mass_diff < 5.0:
        print("⚠️  Moderate mass conservation differences")
    else:
        print("❌ Significant mass conservation differences")

    # Physical observables consistency
    hybrid_com = hybrid['physical_observables']['center_of_mass']
    qp_com = qp['physical_observables']['center_of_mass']
    com_diff = abs(qp_com - hybrid_com)

    hybrid_peak_loc = hybrid['physical_observables']['max_density_location']
    qp_peak_loc = qp['physical_observables']['max_density_location']
    peak_loc_diff = abs(qp_peak_loc - hybrid_peak_loc)

    print(f"Center of mass difference: {com_diff:.4f}")
    print(f"Peak location difference: {peak_loc_diff:.4f}")

    if com_diff < 0.01 and peak_loc_diff < 0.01:
        print("✅ Excellent physical observables consistency")
    elif com_diff < 0.05 and peak_loc_diff < 0.05:
        print("✅ Good physical observables consistency")
    elif com_diff < 0.1 and peak_loc_diff < 0.1:
        print("⚠️  Moderate physical observables differences")
    else:
        print("❌ Significant physical observables differences")

    # Solution quality comparison
    hybrid_violations = hybrid['solution_quality']['violations']
    qp_violations = qp['solution_quality']['violations']
    hybrid_negatives = hybrid['solution_quality']['negative_densities']
    qp_negatives = qp['solution_quality']['negative_densities']

    print(f"\nSolution Quality:")
    print(f"  Hybrid: {hybrid_violations} violations, {hybrid_negatives} negative densities")
    print(f"  QP: {qp_violations} violations, {qp_negatives} negative densities")

    if hybrid_violations == 0 and qp_violations == 0 and hybrid_negatives == 0 and qp_negatives == 0:
        print("✅ Both methods produce numerically clean solutions")
    else:
        print("⚠️  Some numerical quality issues detected")

    # Performance comparison
    print(f"\nPerformance:")
    print(f"  Hybrid time: {hybrid['time']:.2f}s ({hybrid['iterations']} iterations)")
    print(f"  QP time: {qp['time']:.2f}s ({qp['iterations']} iterations)")

    time_ratio = max(hybrid['time'], qp['time']) / min(hybrid['time'], qp['time'])
    faster_method = "Hybrid" if hybrid['time'] < qp['time'] else "QP-Collocation"
    print(f"  Faster method: {faster_method} ({time_ratio:.2f}x speedup)")

    # Overall assessment
    print(f"\n--- OVERALL ASSESSMENT ---")

    # Calculate overall similarity score
    score = 0
    max_score = 5

    # Mass conservation score
    if mass_diff < 1.0:
        score += 1
    elif mass_diff < 3.0:
        score += 0.5

    # Physical observables score
    if com_diff < 0.02 and peak_loc_diff < 0.02:
        score += 1
    elif com_diff < 0.1 and peak_loc_diff < 0.1:
        score += 0.5

    # Solution quality score
    if (hybrid_violations + qp_violations + hybrid_negatives + qp_negatives) == 0:
        score += 1
    elif (hybrid_violations + qp_violations + hybrid_negatives + qp_negatives) <= 2:
        score += 0.5

    # Convergence behavior score
    both_converge = hybrid.get('converged', True) and qp.get('converged', True)
    if both_converge:
        score += 1

    # Performance reasonableness score
    if time_ratio < 3.0:  # Neither method is dramatically slower
        score += 1
    elif time_ratio < 10.0:
        score += 0.5

    percentage = (score / max_score) * 100

    if percentage >= 80:
        assessment = "✅ EXCELLENT: Methods converge to very similar solutions"
    elif percentage >= 60:
        assessment = "✅ GOOD: Methods show good agreement with minor differences"
    elif percentage >= 40:
        assessment = "⚠️  ACCEPTABLE: Methods show reasonable agreement but with notable differences"
    else:
        assessment = "❌ POOR: Methods show significant differences"

    print(f"Similarity score: {score:.1f}/{max_score} ({percentage:.1f}%)")
    print(f"{assessment}")


def create_hybrid_qp_plots(results, problem):
    """Create plots comparing Hybrid and QP-Collocation methods"""
    successful_methods = [method for method in ['hybrid', 'qp'] if results.get(method, {}).get('success', False)]

    if len(successful_methods) < 2:
        print("Insufficient successful results for plotting")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Hybrid vs QP-Collocation Method Comparison', fontsize=16)

    colors = {'hybrid': 'green', 'qp': 'red'}
    method_names = {'hybrid': 'Hybrid Particle-FDM', 'qp': 'QP Particle-Collocation'}

    # 1. Final density comparison
    ax1 = axes[0, 0]
    for method in successful_methods:
        result = results[method]
        final_density = result['arrays']['M_solution'][-1, :]
        ax1.plot(problem.xSpace, final_density, label=method_names[method], color=colors[method], linewidth=3)
    ax1.set_xlabel('Space x')
    ax1.set_ylabel('Final Density M(T,x)')
    ax1.set_title('Final Density Comparison', fontweight='bold')
    ax1.grid(True)
    ax1.legend()

    # 2. Final density difference
    ax2 = axes[0, 1]
    if len(successful_methods) == 2:
        hybrid_density = results['hybrid']['arrays']['M_solution'][-1, :]
        qp_density = results['qp']['arrays']['M_solution'][-1, :]
        density_diff = qp_density - hybrid_density
        ax2.plot(problem.xSpace, density_diff, 'purple', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Space x')
        ax2.set_ylabel('Density Difference (QP - Hybrid)')
        ax2.set_title('Final Density Difference')
        ax2.grid(True)

    # 3. Mass evolution comparison
    ax3 = axes[0, 2]
    for method in successful_methods:
        result = results[method]
        mass_evolution = result['arrays']['mass_evolution']
        ax3.plot(problem.tSpace, mass_evolution, 'o-', label=method_names[method], color=colors[method], linewidth=2)
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Total Mass')
    ax3.set_title('Mass Evolution Comparison')
    ax3.grid(True)
    ax3.legend()

    # 4. Physical observables comparison
    ax4 = axes[1, 0]
    if len(successful_methods) == 2:
        observables = ['Center of Mass', 'Max Density Loc', 'Peak Value']
        hybrid_vals = [
            results['hybrid']['physical_observables']['center_of_mass'],
            results['hybrid']['physical_observables']['max_density_location'],
            results['hybrid']['physical_observables']['final_density_peak'],
        ]
        qp_vals = [
            results['qp']['physical_observables']['center_of_mass'],
            results['qp']['physical_observables']['max_density_location'],
            results['qp']['physical_observables']['final_density_peak'],
        ]

        x_pos = np.arange(len(observables))
        width = 0.35

        bars1 = ax4.bar(x_pos - width / 2, hybrid_vals, width, label='Hybrid', color='green', alpha=0.8)
        bars2 = ax4.bar(x_pos + width / 2, qp_vals, width, label='QP-Collocation', color='red', alpha=0.8)

        ax4.set_xlabel('Observable')
        ax4.set_ylabel('Value')
        ax4.set_title('Physical Observables Comparison')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(observables, rotation=15)
        ax4.legend()
        ax4.grid(True, axis='y')

    # 5. Performance comparison
    ax5 = axes[1, 1]
    if len(successful_methods) == 2:
        methods = ['Hybrid', 'QP-Collocation']
        times = [results['hybrid']['time'], results['qp']['time']]
        iterations = [results['hybrid']['iterations'], results['qp']['iterations']]

        x_pos = np.arange(len(methods))
        width = 0.35

        bars1 = ax5.bar(x_pos - width / 2, times, width, label='Time (s)', color='blue', alpha=0.8)

        # Secondary y-axis for iterations
        ax5_twin = ax5.twinx()
        bars2 = ax5_twin.bar(x_pos + width / 2, iterations, width, label='Iterations', color='orange', alpha=0.8)

        ax5.set_xlabel('Method')
        ax5.set_ylabel('Time (s)', color='blue')
        ax5_twin.set_ylabel('Iterations', color='orange')
        ax5.set_title('Performance Comparison')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(methods)
        ax5.grid(True, axis='y')

        # Add value labels
        for bar, value in zip(bars1, times):
            ax5.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{value:.2f}s', ha='center', va='bottom')
        for bar, value in zip(bars2, iterations):
            ax5_twin.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{value}', ha='center', va='bottom')

    # 6. Control field comparison
    ax6 = axes[1, 2]
    for method in successful_methods:
        result = results[method]
        final_U = result['arrays']['U_solution'][-1, :]
        ax6.plot(problem.xSpace, final_U, label=method_names[method], color=colors[method], linewidth=2)
    ax6.set_xlabel('Space x')
    ax6.set_ylabel('Final Control U(T,x)')
    ax6.set_title('Final Control Field Comparison')
    ax6.grid(True)
    ax6.legend()

    plt.tight_layout()
    plt.savefig(
        '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/hybrid_vs_qp_comparison.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.show()


if __name__ == "__main__":
    print("Starting Hybrid vs QP-Collocation comparison...")
    print("Expected execution time: 2-5 minutes")

    try:
        results = compare_hybrid_vs_qp()

        if results is not None:
            print("\n" + "=" * 80)
            print("HYBRID vs QP-COLLOCATION COMPARISON COMPLETED")
            print("=" * 80)
            print("Check the analysis above and generated plots for detailed comparison.")
        else:
            print("\nComparison could not be completed.")

    except KeyboardInterrupt:
        print("\nComparison interrupted by user.")
    except Exception as e:
        print(f"\nComparison failed with error: {e}")
        import traceback

        traceback.print_exc()
