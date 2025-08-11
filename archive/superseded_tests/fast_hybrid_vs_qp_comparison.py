#!/usr/bin/env python3
"""
Fast Hybrid vs QP-Collocation Comparison
Using proven implementations with optimized parameters for quick execution.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.damped_fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.fp_solvers.particle_fp import ParticleFPSolver
from mfg_pde.alg.hjb_solvers.fdm_hjb import FdmHJBSolver

# Import QP-collocation solver
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions

# Import hybrid solver components
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def run_fast_comparison():
    """Fast comparison with optimized parameters"""
    print("=" * 80)
    print("FAST HYBRID vs QP-COLLOCATION COMPARISON")
    print("=" * 80)
    print("Using proven implementations with optimized parameters for speed")

    # Problem parameters for longer simulation
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 25,  # Reduced resolution for speed
        "T": 2.0,  # Longer T=2 simulation
        "Nt": 100,  # More time steps for longer simulation
        "sigma": 0.2,  # Moderate diffusion
        "coefCT": 0.05,  # Light coupling
    }

    print(f"\nOptimized Problem Parameters:")
    for key, value in problem_params.items():
        print(f"  {key}: {value}")

    # Create problem
    mfg_problem = ExampleMFGProblem(**problem_params)

    print(f"\nProblem setup:")
    print(f"  Domain: [{mfg_problem.xmin}, {mfg_problem.xmax}] × [0, {mfg_problem.T}]")
    print(f"  Grid: Dx = {mfg_problem.Dx:.4f}, Dt = {mfg_problem.Dt:.4f}")

    # Optimized shared settings for T=1 stability
    shared_settings = {
        "num_particles": 400,  # Particles count
        "max_iterations": 15,  # More iterations for T=1
        "convergence_tolerance": 1e-3,  # Tolerance
        "newton_iterations": 8,  # More Newton steps for stability
        "newton_tolerance": 1e-4,  # Newton tolerance
    }

    print(f"\nOptimized Shared Settings:")
    for key, value in shared_settings.items():
        print(f"  {key}: {value}")

    no_flux_bc = BoundaryConditions(type="no_flux")
    results = {}

    # METHOD 1: HYBRID PARTICLE-FDM
    print(f"\n{'='*60}")
    print("METHOD 1: HYBRID PARTICLE-FDM (Fast Configuration)")
    print(f"{'='*60}")

    try:
        start_time = time.time()

        print("  Setting up Hybrid solver...")

        # HJB solver (FDM)
        hjb_solver_component = FdmHJBSolver(
            mfg_problem,
            NiterNewton=shared_settings["newton_iterations"],
            l2errBoundNewton=shared_settings["newton_tolerance"],
        )

        # FP solver (Particle)
        fp_solver_component = ParticleFPSolver(
            mfg_problem,
            num_particles=shared_settings["num_particles"],
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_conditions=no_flux_bc,
        )

        # Fixed point iterator
        hybrid_iterator = FixedPointIterator(
            mfg_problem,
            hjb_solver=hjb_solver_component,
            fp_solver=fp_solver_component,
            thetaUM=0.7,  # More aggressive damping for speed
        )

        print(f"  Solving with {shared_settings['num_particles']} particles...")
        solve_start_time = time.time()

        U_hybrid, M_hybrid, iters_hybrid, rel_distu_h, rel_distm_h = hybrid_iterator.solve(
            shared_settings["max_iterations"], shared_settings["convergence_tolerance"]
        )

        solve_time = time.time() - solve_start_time
        total_time = time.time() - start_time

        if U_hybrid is not None and M_hybrid is not None and iters_hybrid > 0:
            print(f"  ✓ Hybrid completed in {solve_time:.2f}s ({iters_hybrid} iterations)")

            # Quick analysis
            mass_evolution = np.sum(M_hybrid * mfg_problem.Dx, axis=1)
            initial_mass = mass_evolution[0]
            final_mass = mass_evolution[-1]
            mass_change = final_mass - initial_mass

            center_of_mass = np.sum(mfg_problem.xSpace * M_hybrid[-1, :]) * mfg_problem.Dx
            max_idx = np.argmax(M_hybrid[-1, :])
            max_density_loc = mfg_problem.xSpace[max_idx]
            peak_density = M_hybrid[-1, max_idx]

            # Particle violations
            violations = 0
            particles_trajectory = fp_solver_component.M_particles_trajectory
            if particles_trajectory is not None:
                final_particles = particles_trajectory[-1, :]
                violations = np.sum(
                    (final_particles < mfg_problem.xmin - 1e-10) | (final_particles > mfg_problem.xmax + 1e-10)
                )

            results['hybrid'] = {
                'success': True,
                'time': solve_time,
                'iterations': iters_hybrid,
                'mass_initial': initial_mass,
                'mass_final': final_mass,
                'mass_change_percent': (mass_change / initial_mass) * 100,
                'center_of_mass': center_of_mass,
                'max_density_location': max_density_loc,
                'peak_density': peak_density,
                'violations': violations,
                'negative_densities': np.sum(M_hybrid < -1e-10),
                'max_U': np.max(np.abs(U_hybrid)),
                'arrays': {'U': U_hybrid, 'M': M_hybrid, 'mass_evolution': mass_evolution},
            }

            print(f"    Mass: {initial_mass:.6f} → {final_mass:.6f} ({(mass_change/initial_mass)*100:+.3f}%)")
            print(f"    Center of mass: {center_of_mass:.4f}")
            print(f"    Max density: {peak_density:.3f} at x = {max_density_loc:.4f}")
            print(f"    Violations: {violations}")

        else:
            results['hybrid'] = {'success': False}
            print(f"  ❌ Hybrid failed")

    except Exception as e:
        results['hybrid'] = {'success': False, 'error': str(e)}
        print(f"  ❌ Hybrid crashed: {e}")

    # METHOD 2: QP PARTICLE-COLLOCATION
    print(f"\n{'='*60}")
    print("METHOD 2: QP PARTICLE-COLLOCATION (Fast Configuration)")
    print(f"{'='*60}")

    try:
        start_time = time.time()

        print("  Setting up QP-Collocation solver...")

        # Minimal collocation setup
        num_collocation_points = 6  # Reduced for speed
        collocation_points = np.linspace(mfg_problem.xmin, mfg_problem.xmax, num_collocation_points).reshape(-1, 1)

        boundary_indices = [0, num_collocation_points - 1]

        qp_solver = ParticleCollocationSolver(
            problem=mfg_problem,
            collocation_points=collocation_points,
            num_particles=shared_settings["num_particles"],
            delta=0.25,  # Smaller delta for T=1 stability
            taylor_order=2,  # Second-order for better accuracy
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
            f"  Solving with {shared_settings['num_particles']} particles, {num_collocation_points} collocation points..."
        )
        solve_start_time = time.time()

        U_qp, M_qp, solve_info = qp_solver.solve(
            Niter=shared_settings["max_iterations"], l2errBound=shared_settings["convergence_tolerance"], verbose=False
        )

        solve_time = time.time() - solve_start_time
        total_time = time.time() - start_time

        if U_qp is not None and M_qp is not None:
            iterations = solve_info.get("iterations", 0)
            converged = solve_info.get("converged", False)

            print(f"  ✓ QP completed in {solve_time:.2f}s ({iterations} iterations)")
            print(f"    Converged: {converged}")

            # Quick analysis
            mass_evolution = np.sum(M_qp * mfg_problem.Dx, axis=1)
            initial_mass = mass_evolution[0]
            final_mass = mass_evolution[-1]
            mass_change = final_mass - initial_mass

            center_of_mass = np.sum(mfg_problem.xSpace * M_qp[-1, :]) * mfg_problem.Dx
            max_idx = np.argmax(M_qp[-1, :])
            max_density_loc = mfg_problem.xSpace[max_idx]
            peak_density = M_qp[-1, max_idx]

            # Particle violations
            violations = 0
            particles_trajectory = qp_solver.fp_solver.M_particles_trajectory
            if particles_trajectory is not None:
                final_particles = particles_trajectory[-1, :]
                violations = np.sum(
                    (final_particles < mfg_problem.xmin - 1e-10) | (final_particles > mfg_problem.xmax + 1e-10)
                )

            results['qp'] = {
                'success': True,
                'time': solve_time,
                'iterations': iterations,
                'converged': converged,
                'mass_initial': initial_mass,
                'mass_final': final_mass,
                'mass_change_percent': (mass_change / initial_mass) * 100,
                'center_of_mass': center_of_mass,
                'max_density_location': max_density_loc,
                'peak_density': peak_density,
                'violations': violations,
                'negative_densities': np.sum(M_qp < -1e-10),
                'max_U': np.max(np.abs(U_qp)),
                'arrays': {'U': U_qp, 'M': M_qp, 'mass_evolution': mass_evolution},
            }

            print(f"    Mass: {initial_mass:.6f} → {final_mass:.6f} ({(mass_change/initial_mass)*100:+.3f}%)")
            print(f"    Center of mass: {center_of_mass:.4f}")
            print(f"    Max density: {peak_density:.3f} at x = {max_density_loc:.4f}")
            print(f"    Violations: {violations}")

        else:
            results['qp'] = {'success': False}
            print(f"  ❌ QP failed")

    except Exception as e:
        results['qp'] = {'success': False, 'error': str(e)}
        print(f"  ❌ QP crashed: {e}")

    # ANALYSIS
    print(f"\n{'='*80}")
    print("FAST COMPARISON ANALYSIS")
    print(f"{'='*80}")

    analyze_fast_results(results, mfg_problem)
    create_fast_plots(results, mfg_problem)

    return results


def analyze_fast_results(results, problem):
    """Analyze the fast comparison results"""
    successful_methods = [method for method in ['hybrid', 'qp'] if results.get(method, {}).get('success', False)]

    print(f"Successful methods: {len(successful_methods)}/2")

    if len(successful_methods) != 2:
        print("Cannot perform full comparison")
        for method in ['hybrid', 'qp']:
            if method not in successful_methods:
                error = results.get(method, {}).get('error', 'Failed')
                print(f"  Failed: {method} - {error}")
        return

    hybrid = results['hybrid']
    qp = results['qp']

    # Quick comparison table
    print(f"\n{'Metric':<25} {'Hybrid':<15} {'QP-Collocation':<15} {'Difference':<15}")
    print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15}")

    metrics = [
        ('Final Mass', 'mass_final', '.6f'),
        ('Mass Change %', 'mass_change_percent', '+.2f'),
        ('Center of Mass', 'center_of_mass', '.4f'),
        ('Max Density Loc', 'max_density_location', '.4f'),
        ('Peak Density', 'peak_density', '.3f'),
        ('Solve Time (s)', 'time', '.2f'),
        ('Iterations', 'iterations', 'd'),
        ('Violations', 'violations', 'd'),
    ]

    for metric_name, key, fmt in metrics:
        hybrid_val = hybrid[key]
        qp_val = qp[key]

        if fmt == 'd':
            diff_str = str(abs(qp_val - hybrid_val))
            hybrid_str = str(hybrid_val)
            qp_str = str(qp_val)
        else:
            if '+' in fmt:
                hybrid_str = f"{hybrid_val:{fmt}}%"
                qp_str = f"{qp_val:{fmt}}%"
                diff_str = f"{abs(qp_val - hybrid_val):.2f}pp"
            else:
                hybrid_str = f"{hybrid_val:{fmt}}"
                qp_str = f"{qp_val:{fmt}}"
                if 'time' in key:
                    diff_str = f"{abs(qp_val - hybrid_val):.2f}s"
                else:
                    diff_str = f"{abs(qp_val - hybrid_val):.4f}"

        print(f"{metric_name:<25} {hybrid_str:<15} {qp_str:<15} {diff_str:<15}")

    # Key differences analysis
    print(f"\n--- KEY DIFFERENCES ANALYSIS ---")

    mass_diff = abs(qp['mass_change_percent'] - hybrid['mass_change_percent'])
    com_diff = abs(qp['center_of_mass'] - hybrid['center_of_mass'])
    peak_loc_diff = abs(qp['max_density_location'] - hybrid['max_density_location'])

    print(f"Mass change difference: {mass_diff:.2f} percentage points")
    print(f"Center of mass difference: {com_diff:.4f}")
    print(f"Peak location difference: {peak_loc_diff:.4f}")

    # Assessment
    print(f"\n--- ASSESSMENT ---")

    # Mass conservation
    both_reasonable_mass = abs(hybrid['mass_change_percent']) < 10 and abs(qp['mass_change_percent']) < 10
    if both_reasonable_mass:
        print("✅ Both methods show reasonable mass behavior")
    else:
        print("⚠️  Mass conservation issues detected")

    # Solution quality
    both_clean = (
        hybrid['violations'] == 0
        and qp['violations'] == 0
        and hybrid['negative_densities'] == 0
        and qp['negative_densities'] == 0
    )
    if both_clean:
        print("✅ Both methods produce clean solutions")
    else:
        print("⚠️  Some numerical quality issues")

    # Convergence agreement
    if com_diff < 0.1 and peak_loc_diff < 0.1:
        print("✅ Good agreement on physical observables")
    elif com_diff < 0.2 and peak_loc_diff < 0.2:
        print("⚠️  Moderate agreement on physical observables")
    else:
        print("❌ Poor agreement on physical observables")

    # Performance
    faster_method = "Hybrid" if hybrid['time'] < qp['time'] else "QP-Collocation"
    time_ratio = max(hybrid['time'], qp['time']) / min(hybrid['time'], qp['time'])
    print(f"Faster method: {faster_method} ({time_ratio:.1f}x speedup)")


def create_fast_plots(results, problem):
    """Create comparison plot showing both methods on same axis"""
    if not (results.get('hybrid', {}).get('success') and results.get('qp', {}).get('success')):
        print("Cannot create plots - insufficient results")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f'Hybrid vs QP-Collocation Mass Evolution Comparison (T={problem.T})', fontsize=14)

    hybrid = results['hybrid']
    qp = results['qp']

    # Plot both methods on same axis
    ax.plot(
        problem.tSpace,
        hybrid['arrays']['mass_evolution'],
        'g-o',
        linewidth=2,
        markersize=4,
        label=f'Hybrid (Change: {hybrid["mass_change_percent"]:+.2f}%)',
    )
    ax.plot(
        problem.tSpace,
        qp['arrays']['mass_evolution'],
        'r-s',
        linewidth=2,
        markersize=4,
        label=f'QP-Collocation (Change: {qp["mass_change_percent"]:+.2f}%)',
    )

    # Add reference lines for initial mass
    initial_mass_hybrid = hybrid['arrays']['mass_evolution'][0]
    initial_mass_qp = qp['arrays']['mass_evolution'][0]
    ax.axhline(
        y=initial_mass_hybrid, color='g', linestyle='--', alpha=0.5, label=f'Hybrid Initial: {initial_mass_hybrid:.6f}'
    )
    ax.axhline(y=initial_mass_qp, color='r', linestyle='--', alpha=0.5, label=f'QP Initial: {initial_mass_qp:.6f}')

    ax.set_xlabel('Time t')
    ax.set_ylabel('Total Mass')
    ax.set_title('Mass Evolution Comparison')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add performance comparison annotation
    time_comparison = f"Hybrid: {hybrid['time']:.2f}s | QP: {qp['time']:.2f}s"
    faster_method = "Hybrid" if hybrid['time'] < qp['time'] else "QP-Collocation"
    time_ratio = max(hybrid['time'], qp['time']) / min(hybrid['time'], qp['time'])

    ax.text(
        0.05,
        0.95,
        f'{time_comparison}\nFaster: {faster_method} ({time_ratio:.1f}x)',
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    # # 1. Final density comparison
    # ax1 = axes[0, 0]
    # ax1.plot(problem.xSpace, hybrid['arrays']['M'][-1, :],
    #          'g-', linewidth=2, label='Hybrid')
    # ax1.plot(problem.xSpace, qp['arrays']['M'][-1, :],
    #          'r--', linewidth=2, label='QP-Collocation')
    # ax1.set_xlabel('Space x')
    # ax1.set_ylabel('Final Density')
    # ax1.set_title('Final Density Comparison')
    # ax1.grid(True)
    # ax1.legend()

    # # 3. Key metrics comparison
    # ax3 = axes[1, 0]
    # metrics = ['Center of Mass', 'Max Density Loc']
    # hybrid_vals = [hybrid['center_of_mass'], hybrid['max_density_location']]
    # qp_vals = [qp['center_of_mass'], qp['max_density_location']]

    # x_pos = np.arange(len(metrics))
    # width = 0.35

    # ax3.bar(x_pos - width/2, hybrid_vals, width, label='Hybrid', color='green', alpha=0.7)
    # ax3.bar(x_pos + width/2, qp_vals, width, label='QP-Collocation', color='red', alpha=0.7)

    # ax3.set_xlabel('Observable')
    # ax3.set_ylabel('Value')
    # ax3.set_title('Physical Observables')
    # ax3.set_xticks(x_pos)
    # ax3.set_xticklabels(metrics)
    # ax3.legend()
    # ax3.grid(True, axis='y')

    # # 4. Performance comparison
    # ax4 = axes[1, 1]
    # methods = ['Hybrid', 'QP-Collocation']
    # times = [hybrid['time'], qp['time']]
    # iterations = [hybrid['iterations'], qp['iterations']]

    # x_pos = np.arange(len(methods))
    # width = 0.35

    # bars1 = ax4.bar(x_pos - width/2, times, width, label='Time (s)', color='blue', alpha=0.7)
    # ax4_twin = ax4.twinx()
    # bars2 = ax4_twin.bar(x_pos + width/2, iterations, width, label='Iterations', color='orange', alpha=0.7)

    # ax4.set_xlabel('Method')
    # ax4.set_ylabel('Time (s)', color='blue')
    # ax4_twin.set_ylabel('Iterations', color='orange')
    # ax4.set_title('Performance Comparison')
    # ax4.set_xticks(x_pos)
    # ax4.set_xticklabels(methods)
    # ax4.grid(True, axis='y')

    # # Add value labels
    # for bar, value in zip(bars1, times):
    #     ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
    #             f'{value:.2f}s', ha='center', va='bottom')
    # for bar, value in zip(bars2, iterations):
    #     ax4_twin.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
    #                  f'{value}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(
        '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/fast_hybrid_vs_qp_comparison.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.show()


if __name__ == "__main__":
    print("Starting Fast Hybrid vs QP-Collocation comparison...")
    print("Optimized parameters for quick execution and reliable results")
    print("Expected execution time: < 1 minute")

    try:
        results = run_fast_comparison()
        print("\n" + "=" * 80)
        print("FAST COMPARISON COMPLETED")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\nComparison interrupted by user.")
    except Exception as e:
        print(f"\nComparison failed: {e}")
        import traceback

        traceback.print_exc()
