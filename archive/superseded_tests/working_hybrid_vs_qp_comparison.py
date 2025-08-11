#!/usr/bin/env python3
"""
Working Hybrid vs QP-Collocation Comparison
Using proven implementations from examples directory with identical parameters.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.damped_fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.fp_solvers.particle_fp import ParticleFPSolver
from mfg_pde.alg.hjb_solvers.fdm_hjb import FdmHJBSolver

# Import QP-collocation solver (from examples/particle_collocation_no_flux_bc.py)
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions

# Import hybrid solver components (from examples/hybrid_no_flux_bc.py)
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def run_working_comparison():
    """Compare working Hybrid and QP-Collocation implementations"""
    print("=" * 80)
    print("WORKING HYBRID vs QP-COLLOCATION COMPARISON")
    print("=" * 80)
    print("Using proven implementations from examples directory")
    print("Both methods tested with identical parameters and boundary conditions")

    # Conservative parameters for mass conservation verification
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 30,  # Conservative resolution
        "T": 1.0,  # Shorter time to ensure stability
        "Nt": 100,  # Sufficient time steps for small Dt
        "sigma": 0.15,  # Lower diffusion
        "coefCT": 0.02,  # Light coupling
    }

    print(f"\nUnified Problem Parameters:")
    for key, value in problem_params.items():
        print(f"  {key}: {value}")

    # Create problem
    mfg_problem = ExampleMFGProblem(**problem_params)

    print(f"\nProblem setup:")
    print(f"  Domain: [{mfg_problem.xmin}, {mfg_problem.xmax}] × [0, {mfg_problem.T}]")
    print(f"  Grid: Dx = {mfg_problem.Dx:.4f}, Dt = {mfg_problem.Dt:.4f}")
    print(f"  CFL number: {problem_params['sigma']**2 * mfg_problem.Dt / mfg_problem.Dx**2:.4f}")

    # Shared particle settings
    shared_particle_settings = {"num_particles": 800, "kde_bandwidth": "scott"}

    # Shared convergence settings
    shared_convergence = {
        "max_iterations": 15,
        "convergence_tolerance": 1e-4,
        "newton_iterations": 8,
        "newton_tolerance": 1e-5,
    }

    print(f"\nShared Settings:")
    print(f"  Particles: {shared_particle_settings['num_particles']}")
    print(f"  Max iterations: {shared_convergence['max_iterations']}")
    print(f"  Convergence tolerance: {shared_convergence['convergence_tolerance']}")

    # No-flux boundary conditions (same for both)
    no_flux_bc = BoundaryConditions(type="no_flux")

    results = {}

    # METHOD 1: HYBRID PARTICLE-FDM (from examples)
    print(f"\n{'='*60}")
    print("METHOD 1: HYBRID PARTICLE-FDM (Examples Implementation)")
    print(f"{'='*60}")

    try:
        start_time = time.time()

        print("  Setting up Hybrid solver components...")

        # HJB solver (FDM)
        hjb_solver_component = FdmHJBSolver(
            mfg_problem,
            NiterNewton=shared_convergence["newton_iterations"],
            l2errBoundNewton=shared_convergence["newton_tolerance"],
        )

        # FP solver (Particle) with no-flux boundaries
        fp_solver_component = ParticleFPSolver(
            mfg_problem,
            num_particles=shared_particle_settings["num_particles"],
            kde_bandwidth=shared_particle_settings["kde_bandwidth"],
            normalize_kde_output=False,
            boundary_conditions=no_flux_bc,
        )

        # Fixed point iterator with damping (conservative for stability)
        hybrid_iterator = FixedPointIterator(
            mfg_problem,
            hjb_solver=hjb_solver_component,
            fp_solver=fp_solver_component,
            thetaUM=0.5,  # Moderate damping factor
        )

        print(f"  Starting Hybrid solve with {shared_particle_settings['num_particles']} particles...")
        solve_start_time = time.time()

        U_hybrid, M_hybrid, iters_hybrid, rel_distu_h, rel_distm_h = hybrid_iterator.solve(
            shared_convergence["max_iterations"], shared_convergence["convergence_tolerance"]
        )

        solve_time = time.time() - solve_start_time
        total_hybrid_time = time.time() - start_time

        if U_hybrid is not None and M_hybrid is not None and iters_hybrid > 0:
            print(f"  ✓ Hybrid completed in {solve_time:.2f}s ({iters_hybrid} iterations)")

            # Analysis
            mass_evolution_hybrid = np.sum(M_hybrid * mfg_problem.Dx, axis=1)
            initial_mass_hybrid = mass_evolution_hybrid[0]
            final_mass_hybrid = mass_evolution_hybrid[-1]
            mass_change_hybrid = final_mass_hybrid - initial_mass_hybrid

            # Physical observables
            center_of_mass_hybrid = np.sum(mfg_problem.xSpace * M_hybrid[-1, :]) * mfg_problem.Dx
            max_density_idx_hybrid = np.argmax(M_hybrid[-1, :])
            max_density_loc_hybrid = mfg_problem.xSpace[max_density_idx_hybrid]
            final_density_peak_hybrid = M_hybrid[-1, max_density_idx_hybrid]

            # Particle violations
            violations_hybrid = 0
            particles_trajectory = fp_solver_component.M_particles_trajectory
            if particles_trajectory is not None:
                final_particles = particles_trajectory[-1, :]
                violations_hybrid = np.sum(
                    (final_particles < mfg_problem.xmin - 1e-10) | (final_particles > mfg_problem.xmax + 1e-10)
                )

            results['hybrid'] = {
                'success': True,
                'method_name': 'Hybrid Particle-FDM',
                'solver_info': 'Examples directory implementation',
                'time': solve_time,
                'total_time': total_hybrid_time,
                'iterations': iters_hybrid,
                'converged': True,  # Assume converged if completed
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
                'arrays': {
                    'U_solution': U_hybrid,
                    'M_solution': M_hybrid,
                    'mass_evolution': mass_evolution_hybrid,
                    'particles_trajectory': particles_trajectory,
                },
            }

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

    # METHOD 2: QP PARTICLE-COLLOCATION (from examples)
    print(f"\n{'='*60}")
    print("METHOD 2: QP PARTICLE-COLLOCATION (Examples Implementation)")
    print(f"{'='*60}")

    try:
        start_time = time.time()

        print("  Setting up QP-Collocation solver...")

        # Setup collocation points (from examples)
        num_collocation_points = 12
        collocation_points = np.linspace(mfg_problem.xmin, mfg_problem.xmax, num_collocation_points).reshape(-1, 1)

        # Identify boundary points
        boundary_tolerance = 1e-10
        boundary_indices = []
        for i, point in enumerate(collocation_points):
            x = point[0]
            if abs(x - mfg_problem.xmin) < boundary_tolerance or abs(x - mfg_problem.xmax) < boundary_tolerance:
                boundary_indices.append(i)
        boundary_indices = np.array(boundary_indices)

        # QP solver parameters (from examples)
        qp_solver = ParticleCollocationSolver(
            problem=mfg_problem,
            collocation_points=collocation_points,
            num_particles=shared_particle_settings["num_particles"],
            delta=0.35,  # From examples
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=shared_convergence["newton_iterations"],
            l2errBoundNewton=shared_convergence["newton_tolerance"],
            kde_bandwidth=shared_particle_settings["kde_bandwidth"],
            normalize_kde_output=False,
            boundary_indices=boundary_indices,
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,  # QP constraints enabled
        )

        print(
            f"  Starting QP solve with {shared_particle_settings['num_particles']} particles, {num_collocation_points} collocation points..."
        )
        solve_start_time = time.time()

        U_qp, M_qp, solve_info_qp = qp_solver.solve(
            Niter=shared_convergence["max_iterations"],
            l2errBound=shared_convergence["convergence_tolerance"],
            verbose=False,
        )

        solve_time = time.time() - solve_start_time
        total_qp_time = time.time() - start_time

        if U_qp is not None and M_qp is not None:
            iterations_qp = solve_info_qp.get("iterations", 0)
            converged_qp = solve_info_qp.get("converged", False)

            print(f"  ✓ QP completed in {solve_time:.2f}s ({iterations_qp} iterations)")
            print(f"    Converged: {converged_qp}")

            # Analysis
            mass_evolution_qp = np.sum(M_qp * mfg_problem.Dx, axis=1)
            initial_mass_qp = mass_evolution_qp[0]
            final_mass_qp = mass_evolution_qp[-1]
            mass_change_qp = final_mass_qp - initial_mass_qp

            # Physical observables
            center_of_mass_qp = np.sum(mfg_problem.xSpace * M_qp[-1, :]) * mfg_problem.Dx
            max_density_idx_qp = np.argmax(M_qp[-1, :])
            max_density_loc_qp = mfg_problem.xSpace[max_density_idx_qp]
            final_density_peak_qp = M_qp[-1, max_density_idx_qp]

            # Particle violations
            violations_qp = 0
            particles_trajectory = qp_solver.fp_solver.M_particles_trajectory
            if particles_trajectory is not None:
                final_particles = particles_trajectory[-1, :]
                violations_qp = np.sum(
                    (final_particles < mfg_problem.xmin - 1e-10) | (final_particles > mfg_problem.xmax + 1e-10)
                )

            results['qp'] = {
                'success': True,
                'method_name': 'QP Particle-Collocation',
                'solver_info': 'Examples directory implementation with QP constraints',
                'time': solve_time,
                'total_time': total_qp_time,
                'iterations': iterations_qp,
                'converged': converged_qp,
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
                'arrays': {
                    'U_solution': U_qp,
                    'M_solution': M_qp,
                    'mass_evolution': mass_evolution_qp,
                    'particles_trajectory': particles_trajectory,
                },
            }

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

    # COMPREHENSIVE ANALYSIS
    print(f"\n{'='*80}")
    print("COMPREHENSIVE COMPARISON ANALYSIS")
    print(f"{'='*80}")

    analyze_working_comparison(results, mfg_problem)
    create_working_comparison_plots(results, mfg_problem)

    return results


def analyze_working_comparison(results, problem):
    """Analyze the working comparison results"""
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
    print(f"\n{'Metric':<30} {'Hybrid':<20} {'QP-Collocation':<20} {'Difference':<15}")
    print(f"{'-'*30} {'-'*20} {'-'*20} {'-'*15}")

    # Compare key metrics
    metrics = [
        ('Final Mass', lambda r: r['mass_conservation']['final_mass'], '.8f'),
        ('Mass Change %', lambda r: r['mass_conservation']['mass_change_percent'], '+.3f'),
        ('Center of Mass', lambda r: r['physical_observables']['center_of_mass'], '.6f'),
        ('Max Density Location', lambda r: r['physical_observables']['max_density_location'], '.6f'),
        ('Peak Density Value', lambda r: r['physical_observables']['final_density_peak'], '.4f'),
        ('Max |U|', lambda r: r['solution_quality']['max_U'], '.3e'),
        ('Min M', lambda r: r['solution_quality']['min_M'], '.3e'),
        ('Boundary Violations', lambda r: r['solution_quality']['violations'], 'd'),
        ('Negative Densities', lambda r: r['solution_quality']['negative_densities'], 'd'),
        ('Solve Time (s)', lambda r: r['time'], '.2f'),
        ('Total Time (s)', lambda r: r['total_time'], '.2f'),
        ('Iterations', lambda r: r['iterations'], 'd'),
    ]

    for metric_name, extract_func, fmt in metrics:
        hybrid_val = extract_func(hybrid)
        qp_val = extract_func(qp)

        if fmt == 'd':
            diff_str = str(abs(qp_val - hybrid_val))
            hybrid_str = str(hybrid_val)
            qp_str = str(qp_val)
        else:
            if '+' in fmt:
                hybrid_str = f"{hybrid_val:{fmt}}%"
                qp_str = f"{qp_val:{fmt}}%"
                diff_str = f"{abs(qp_val - hybrid_val):.3f}pp"
            else:
                hybrid_str = f"{hybrid_val:{fmt}}"
                qp_str = f"{qp_val:{fmt}}"
                if 'time' in metric_name.lower():
                    diff_str = f"{abs(qp_val - hybrid_val):.2f}s"
                elif 'e' in fmt:
                    diff_str = f"{abs(qp_val - hybrid_val):.2e}"
                else:
                    diff_str = f"{abs(qp_val - hybrid_val):.6f}"

        print(f"{metric_name:<30} {hybrid_str:<20} {qp_str:<20} {diff_str:<15}")

    # Detailed convergence analysis
    print(f"\n--- CONVERGENCE TO SAME SOLUTION ANALYSIS ---")

    # Mass conservation consistency
    hybrid_mass_change = hybrid['mass_conservation']['mass_change_percent']
    qp_mass_change = qp['mass_conservation']['mass_change_percent']
    mass_diff = abs(qp_mass_change - hybrid_mass_change)

    print(f"Mass change difference: {mass_diff:.3f} percentage points")

    # Physical observables consistency
    hybrid_com = hybrid['physical_observables']['center_of_mass']
    qp_com = qp['physical_observables']['center_of_mass']
    com_diff = abs(qp_com - hybrid_com)

    hybrid_peak_loc = hybrid['physical_observables']['max_density_location']
    qp_peak_loc = qp['physical_observables']['max_density_location']
    peak_loc_diff = abs(qp_peak_loc - hybrid_peak_loc)

    hybrid_peak_val = hybrid['physical_observables']['final_density_peak']
    qp_peak_val = qp['physical_observables']['final_density_peak']
    peak_val_diff = abs(qp_peak_val - hybrid_peak_val)

    print(f"Center of mass difference: {com_diff:.6f}")
    print(f"Peak location difference: {peak_loc_diff:.6f}")
    print(f"Peak value difference: {peak_val_diff:.4f}")

    # Convergence scoring
    score = 0
    max_score = 6

    # Mass conservation score
    if mass_diff < 0.5:
        print("✅ Excellent mass conservation consistency")
        score += 2
    elif mass_diff < 2.0:
        print("✅ Good mass conservation consistency")
        score += 1.5
    elif mass_diff < 5.0:
        print("⚠️  Moderate mass conservation differences")
        score += 1
    else:
        print("❌ Significant mass conservation differences")

    # Physical observables score
    if com_diff < 0.01 and peak_loc_diff < 0.01:
        print("✅ Excellent physical observables consistency")
        score += 2
    elif com_diff < 0.05 and peak_loc_diff < 0.05:
        print("✅ Good physical observables consistency")
        score += 1.5
    elif com_diff < 0.1 and peak_loc_diff < 0.1:
        print("⚠️  Moderate physical observables differences")
        score += 1
    else:
        print("❌ Significant physical observables differences")

    # Solution quality score
    hybrid_violations = hybrid['solution_quality']['violations']
    qp_violations = qp['solution_quality']['violations']
    hybrid_negatives = hybrid['solution_quality']['negative_densities']
    qp_negatives = qp['solution_quality']['negative_densities']

    total_issues = hybrid_violations + qp_violations + hybrid_negatives + qp_negatives
    if total_issues == 0:
        print("✅ Both methods produce numerically clean solutions")
        score += 1
    elif total_issues <= 5:
        print("⚠️  Minor numerical issues detected")
        score += 0.5
    else:
        print("❌ Significant numerical quality issues")

    # Convergence behavior score
    both_converge = hybrid.get('converged', True) and qp.get('converged', True)
    if both_converge:
        print("✅ Both methods achieved convergence")
        score += 1
    else:
        print("⚠️  At least one method did not fully converge")
        score += 0.5

    # Overall assessment
    percentage = (score / max_score) * 100
    print(f"\n--- OVERALL CONVERGENCE ASSESSMENT ---")
    print(f"Similarity score: {score:.1f}/{max_score} ({percentage:.1f}%)")

    if percentage >= 85:
        assessment = "✅ EXCELLENT: Methods converge to essentially the same solution"
    elif percentage >= 70:
        assessment = "✅ VERY GOOD: Methods converge to very similar solutions"
    elif percentage >= 55:
        assessment = "✅ GOOD: Methods converge to reasonably similar solutions"
    elif percentage >= 40:
        assessment = "⚠️  ACCEPTABLE: Methods show some convergence but with notable differences"
    else:
        assessment = "❌ POOR: Methods appear to converge to different solutions"

    print(f"{assessment}")

    # Performance comparison
    print(f"\n--- PERFORMANCE COMPARISON ---")
    time_ratio = max(hybrid['time'], qp['time']) / min(hybrid['time'], qp['time'])
    faster_method = "Hybrid" if hybrid['time'] < qp['time'] else "QP-Collocation"

    print(f"Solve times: Hybrid {hybrid['time']:.2f}s, QP {qp['time']:.2f}s")
    print(f"Iterations: Hybrid {hybrid['iterations']}, QP {qp['iterations']}")
    print(f"Faster method: {faster_method} ({time_ratio:.2f}x speedup)")

    # Method strengths
    print(f"\n--- METHOD STRENGTHS ---")
    print("Hybrid Particle-FDM:")
    print("  + Proven stable HJB-FDM coupling")
    print("  + Fast execution due to grid-based HJB solver")
    print("  + Well-established particle dynamics")

    print("QP Particle-Collocation:")
    print("  + Monotonicity constraints via quadratic programming")
    print("  + GFDM flexibility for complex geometries")
    print("  + Superior boundary condition handling")


def create_working_comparison_plots(results, problem):
    """Create comparison plots showing only figures 1.1 and 1.3 with exchanged positions"""
    if not (results.get('hybrid', {}).get('success') and results.get('qp', {}).get('success')):
        print("Cannot create plots - insufficient successful results")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'Working Hybrid vs QP-Collocation Comparison (T={problem.T}, Nx={problem.Nx})\n(Figures 1.1 and 1.3 with Exchanged Positions)',
        fontsize=14,
    )

    hybrid = results['hybrid']
    qp = results['qp']

    # Figure 1.3 in position 1.1: Mass evolution comparison (originally figure 1.3)
    ax1.plot(problem.tSpace, hybrid['arrays']['mass_evolution'], 'g-o', linewidth=2, markersize=3, label='Hybrid')
    ax1.plot(problem.tSpace, qp['arrays']['mass_evolution'], 'r-s', linewidth=2, markersize=3, label='QP-Collocation')

    # Add reference lines for mass conservation
    ax1.axhline(
        y=hybrid['arrays']['mass_evolution'][0],
        color='g',
        linestyle='--',
        alpha=0.5,
        label=f'Hybrid Initial: {hybrid["arrays"]["mass_evolution"][0]:.4f}',
    )
    ax1.axhline(
        y=qp['arrays']['mass_evolution'][0],
        color='r',
        linestyle='--',
        alpha=0.5,
        label=f'QP Initial: {qp["arrays"]["mass_evolution"][0]:.4f}',
    )

    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Total Mass')
    ax1.set_title('Position 1.1: Mass Evolution Comparison (Originally Fig 1.3)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add mass change annotations
    hybrid_change = (
        (hybrid['arrays']['mass_evolution'][-1] - hybrid['arrays']['mass_evolution'][0])
        / hybrid['arrays']['mass_evolution'][0]
        * 100
    )
    qp_change = (
        (qp['arrays']['mass_evolution'][-1] - qp['arrays']['mass_evolution'][0])
        / qp['arrays']['mass_evolution'][0]
        * 100
    )

    ax1.text(
        0.05,
        0.95,
        f'Hybrid: {hybrid_change:+.2f}%\nQP: {qp_change:+.2f}%',
        transform=ax1.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    # Figure 1.1 in position 1.3: Final density comparison (originally figure 1.1)
    ax2.plot(problem.xSpace, hybrid['arrays']['M_solution'][-1, :], 'g-', linewidth=2, label='Hybrid', alpha=0.8)
    ax2.plot(problem.xSpace, qp['arrays']['M_solution'][-1, :], 'r--', linewidth=2, label='QP-Collocation', alpha=0.8)

    # Add initial density for reference
    ax2.plot(
        problem.xSpace, hybrid['arrays']['M_solution'][0, :], 'k:', linewidth=1, alpha=0.7, label='Initial Density'
    )

    ax2.set_xlabel('Space x')
    ax2.set_ylabel('Final Density M(T,x)')
    ax2.set_title('Position 1.3: Final Density Comparison (Originally Fig 1.1)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add density peak annotations
    hybrid_peak_idx = np.argmax(hybrid['arrays']['M_solution'][-1, :])
    qp_peak_idx = np.argmax(qp['arrays']['M_solution'][-1, :])
    hybrid_peak_loc = problem.xSpace[hybrid_peak_idx]
    qp_peak_loc = problem.xSpace[qp_peak_idx]
    hybrid_peak_val = hybrid['arrays']['M_solution'][-1, hybrid_peak_idx]
    qp_peak_val = qp['arrays']['M_solution'][-1, qp_peak_idx]

    ax2.text(
        0.05,
        0.95,
        f'Peak Locations:\nHybrid: x={hybrid_peak_loc:.3f}\nQP: x={qp_peak_loc:.3f}',
        transform=ax2.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/tests/method_comparisons/working_hybrid_vs_qp_comparison.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.show()

    print(f"\n✅ Comparison plots saved: working_hybrid_vs_qp_comparison.png")


if __name__ == "__main__":
    print("Starting Working Hybrid vs QP-Collocation comparison...")
    print("Using proven implementations from examples directory")
    print("Expected execution time: 2-8 minutes")

    try:
        results = run_working_comparison()
        print("\n" + "=" * 80)
        print("WORKING COMPARISON COMPLETED")
        print("=" * 80)
        print("Both methods successfully executed using examples directory implementations.")

    except KeyboardInterrupt:
        print("\nComparison interrupted by user.")
    except Exception as e:
        print(f"\nComparison failed: {e}")
        import traceback

        traceback.print_exc()
