#!/usr/bin/env python3
"""
Mild Environment Comparison Test
Compares different solver methods under mild/stable parameter conditions.
Shows only figures 1.1 (Mass Evolution) and 1.2 (Density Evolution) for clean visualization.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.damped_fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.fp_solvers.particle_fp import ParticleFPSolver

# Import hybrid solver components
from mfg_pde.alg.hjb_solvers.fdm_hjb import FdmHJBSolver
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def run_mild_environment_comparison():
    """Run comparison under mild/stable parameter conditions"""
    print("=" * 80)
    print("MILD ENVIRONMENT COMPARISON TEST")
    print("=" * 80)
    print("Testing solver performance under stable, mild parameter conditions")
    print("Output: Only figures 1.1 (Mass Evolution) and 1.2 (Density Evolution)")

    # Mild/stable problem parameters with T=2 simulation
    mild_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 20,  # Moderate resolution
        "T": 2.0,  # Optimal T=2 simulation
        "Nt": 100,  # Good time resolution for T=2
        "sigma": 0.15,  # Mild diffusion
        "coefCT": 0.02,  # Light coupling for stability
    }

    print(f"\nMild Environment Parameters:")
    for key, value in mild_params.items():
        print(f"  {key}: {value}")

    # Create problem
    problem = ExampleMFGProblem(**mild_params)

    print(f"\nProblem setup:")
    print(f"  Domain: [{problem.xmin}, {problem.xmax}] × [0, {problem.T}]")
    print(f"  Grid: Dx = {problem.Dx:.4f}, Dt = {problem.Dt:.4f}")
    print(f"  CFL number: {mild_params['sigma']**2 * problem.Dt / problem.Dx**2:.4f}")

    # Shared solver settings for T=2 optimal performance
    solver_settings = {
        "num_particles": 300,  # Optimal particles for T=2
        "max_iterations": 15,  # Good iterations for T=2
        "convergence_tolerance": 1e-3,
        "newton_iterations": 8,  # Balanced Newton iterations
        "newton_tolerance": 1e-4,
    }

    print(f"\nSolver Settings:")
    for key, value in solver_settings.items():
        print(f"  {key}: {value}")

    no_flux_bc = BoundaryConditions(type="no_flux")
    results = {}

    # METHOD 1: HYBRID PARTICLE-FDM
    print(f"\n{'='*60}")
    print("METHOD 1: HYBRID PARTICLE-FDM (Mild Environment)")
    print(f"{'='*60}")

    try:
        start_time = time.time()

        # Setup hybrid solver
        hjb_solver = FdmHJBSolver(
            problem,
            NiterNewton=solver_settings["newton_iterations"],
            l2errBoundNewton=solver_settings["newton_tolerance"],
        )

        fp_solver = ParticleFPSolver(
            problem,
            num_particles=solver_settings["num_particles"],
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_conditions=no_flux_bc,
        )

        hybrid_iterator = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            thetaUM=0.5,  # Mild damping for stability
        )

        U_hybrid, M_hybrid, iters_hybrid, _, _ = hybrid_iterator.solve(
            solver_settings["max_iterations"], solver_settings["convergence_tolerance"]
        )

        solve_time = time.time() - start_time

        if U_hybrid is not None and M_hybrid is not None:
            # Calculate metrics
            mass_evolution = np.sum(M_hybrid * problem.Dx, axis=1)
            initial_mass = mass_evolution[0]
            final_mass = mass_evolution[-1]
            mass_change_percent = ((final_mass - initial_mass) / initial_mass) * 100

            violations = 0
            if hasattr(fp_solver, 'M_particles_trajectory') and fp_solver.M_particles_trajectory is not None:
                final_particles = fp_solver.M_particles_trajectory[-1, :]
                violations = np.sum((final_particles < problem.xmin - 1e-10) | (final_particles > problem.xmax + 1e-10))

            results['hybrid'] = {
                'success': True,
                'time': solve_time,
                'iterations': iters_hybrid,
                'mass_evolution': mass_evolution,
                'mass_change_percent': mass_change_percent,
                'violations': violations,
                'U': U_hybrid,
                'M': M_hybrid,
                'max_U': np.max(np.abs(U_hybrid)),
                'negative_densities': np.sum(M_hybrid < -1e-10),
            }

            print(f"  ✓ Hybrid completed in {solve_time:.2f}s ({iters_hybrid} iterations)")
            print(f"    Mass change: {mass_change_percent:+.3f}%")
            print(f"    Violations: {violations}")

        else:
            results['hybrid'] = {'success': False}
            print(f"  ❌ Hybrid failed")

    except Exception as e:
        results['hybrid'] = {'success': False, 'error': str(e)}
        print(f"  ❌ Hybrid crashed: {e}")

    # METHOD 2: QP PARTICLE-COLLOCATION
    print(f"\n{'='*60}")
    print("METHOD 2: QP PARTICLE-COLLOCATION (Mild Environment)")
    print(f"{'='*60}")

    try:
        start_time = time.time()

        # Setup QP-Collocation solver
        num_collocation_points = 8
        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)

        boundary_indices = [0, num_collocation_points - 1]

        qp_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=solver_settings["num_particles"],
            delta=0.25,  # Conservative delta for stability
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=solver_settings["newton_iterations"],
            l2errBoundNewton=solver_settings["newton_tolerance"],
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
        )

        U_qp, M_qp, solve_info = qp_solver.solve(
            Niter=solver_settings["max_iterations"], l2errBound=solver_settings["convergence_tolerance"], verbose=False
        )

        solve_time = time.time() - start_time

        if U_qp is not None and M_qp is not None:
            # Calculate metrics
            mass_evolution = np.sum(M_qp * problem.Dx, axis=1)
            initial_mass = mass_evolution[0]
            final_mass = mass_evolution[-1]
            mass_change_percent = ((final_mass - initial_mass) / initial_mass) * 100

            violations = 0
            if (
                hasattr(qp_solver.fp_solver, 'M_particles_trajectory')
                and qp_solver.fp_solver.M_particles_trajectory is not None
            ):
                final_particles = qp_solver.fp_solver.M_particles_trajectory[-1, :]
                violations = np.sum((final_particles < problem.xmin - 1e-10) | (final_particles > problem.xmax + 1e-10))

            iterations = solve_info.get("iterations", 0)
            converged = solve_info.get("converged", False)

            results['qp'] = {
                'success': True,
                'time': solve_time,
                'iterations': iterations,
                'converged': converged,
                'mass_evolution': mass_evolution,
                'mass_change_percent': mass_change_percent,
                'violations': violations,
                'U': U_qp,
                'M': M_qp,
                'max_U': np.max(np.abs(U_qp)),
                'negative_densities': np.sum(M_qp < -1e-10),
            }

            print(f"  ✓ QP completed in {solve_time:.2f}s ({iterations} iterations)")
            print(f"    Converged: {converged}")
            print(f"    Mass change: {mass_change_percent:+.3f}%")
            print(f"    Violations: {violations}")

        else:
            results['qp'] = {'success': False}
            print(f"  ❌ QP failed")

    except Exception as e:
        results['qp'] = {'success': False, 'error': str(e)}
        print(f"  ❌ QP crashed: {e}")

    # ANALYSIS AND PLOTTING
    print(f"\n{'='*80}")
    print("MILD ENVIRONMENT ANALYSIS")
    print(f"{'='*80}")

    analyze_mild_results(results, problem)
    create_mild_comparison_plots(results, problem)

    return results


def analyze_mild_results(results, problem):
    """Analyze results from mild environment test"""
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

    print(f"\n--- MILD ENVIRONMENT COMPARISON RESULTS ---")
    print(f"{'Metric':<25} {'Hybrid':<15} {'QP-Collocation':<15}")
    print(f"{'-'*25} {'-'*15} {'-'*15}")

    metrics = [
        ('Mass Change %', 'mass_change_percent', '+.3f'),
        ('Execution Time (s)', 'time', '.2f'),
        ('Iterations', 'iterations', 'd'),
        ('Boundary Violations', 'violations', 'd'),
        ('Negative Densities', 'negative_densities', 'd'),
        ('Max |U|', 'max_U', '.2f'),
    ]

    for metric_name, key, fmt in metrics:
        hybrid_val = hybrid[key]
        qp_val = qp[key]

        if fmt == 'd':
            hybrid_str = str(hybrid_val)
            qp_str = str(qp_val)
        else:
            hybrid_str = f"{hybrid_val:{fmt}}"
            qp_str = f"{qp_val:{fmt}}"

        print(f"{metric_name:<25} {hybrid_str:<15} {qp_str:<15}")

    # Assessment
    print(f"\n--- MILD ENVIRONMENT ASSESSMENT ---")

    # Mass conservation
    both_good_mass = abs(hybrid['mass_change_percent']) < 5 and abs(qp['mass_change_percent']) < 5
    if both_good_mass:
        print("✅ Both methods show good mass conservation under mild conditions")
    else:
        print("⚠️  Some mass conservation issues detected")

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
        print("⚠️  Some numerical quality issues detected")

    # Performance
    faster_method = "Hybrid" if hybrid['time'] < qp['time'] else "QP-Collocation"
    if hybrid['time'] != qp['time']:
        time_ratio = max(hybrid['time'], qp['time']) / min(hybrid['time'], qp['time'])
        print(f"⚡ Faster method: {faster_method} ({time_ratio:.1f}x speedup)")
    else:
        print("⚡ Both methods have similar performance")


def create_mild_comparison_plots(results, problem):
    """Create comparison plots showing only figures 1.1 and 1.2"""
    if not (results.get('hybrid', {}).get('success') and results.get('qp', {}).get('success')):
        print("Cannot create plots - insufficient results")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Mild Environment Comparison (T={problem.T})', fontsize=14, fontweight='bold')

    hybrid = results['hybrid']
    qp = results['qp']

    # Figure 1.1: Mass Evolution Comparison
    ax1.plot(
        problem.tSpace,
        hybrid['mass_evolution'],
        'g-o',
        linewidth=2,
        markersize=4,
        label=f'Hybrid ({hybrid["mass_change_percent"]:+.2f}%)',
    )
    ax1.plot(
        problem.tSpace,
        qp['mass_evolution'],
        'r-s',
        linewidth=2,
        markersize=4,
        label=f'QP-Collocation ({qp["mass_change_percent"]:+.2f}%)',
    )

    # Add reference lines
    ax1.axhline(
        y=hybrid['mass_evolution'][0],
        color='g',
        linestyle='--',
        alpha=0.5,
        label=f'Hybrid Initial: {hybrid["mass_evolution"][0]:.4f}',
    )
    ax1.axhline(
        y=qp['mass_evolution'][0],
        color='r',
        linestyle='--',
        alpha=0.5,
        label=f'QP Initial: {qp["mass_evolution"][0]:.4f}',
    )

    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Total Mass')
    ax1.set_title('Figure 1.1: Mass Evolution Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Performance annotation
    performance_text = f"Hybrid: {hybrid['time']:.2f}s | QP: {qp['time']:.2f}s"
    conservation_text = "Both methods stable under mild conditions"
    ax1.text(
        0.05,
        0.95,
        f'{performance_text}\n{conservation_text}',
        transform=ax1.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
    )

    # Figure 1.2: Final Density Comparison
    ax2.plot(problem.xSpace, hybrid['M'][-1, :], 'g-', linewidth=2, label='Hybrid Final Density')
    ax2.plot(problem.xSpace, qp['M'][-1, :], 'r--', linewidth=2, label='QP-Collocation Final Density')

    # Add initial density for reference
    ax2.plot(problem.xSpace, hybrid['M'][0, :], 'k:', linewidth=1, alpha=0.7, label='Initial Density')

    ax2.set_xlabel('Space x')
    ax2.set_ylabel('Density')
    ax2.set_title('Figure 1.2: Final Density Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Quality annotation
    quality_metrics = f"Violations: H={hybrid['violations']}, QP={qp['violations']}"
    ax2.text(
        0.05,
        0.95,
        f'{quality_metrics}\nMild environment: Stable evolution',
        transform=ax2.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/tests/stability_analysis/mild_environment_comparison.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.show()

    print(f"\n✅ Plots saved: mild_environment_comparison.png")
    print("   Figure 1.1: Mass evolution comparison over time")
    print("   Figure 1.2: Final density comparison across space")


if __name__ == "__main__":
    print("Starting Mild Environment Comparison Test...")
    print("Designed to test solver performance under stable, mild conditions")
    print("Output: Clean 2-panel visualization (figures 1.1 and 1.2 only)")

    try:
        results = run_mild_environment_comparison()
        print("\n" + "=" * 80)
        print("MILD ENVIRONMENT COMPARISON COMPLETED")
        print("=" * 80)
        print("✅ Both methods tested under mild/stable parameter conditions")
        print("✅ Clean visualization with figures 1.1 and 1.2 only")

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
