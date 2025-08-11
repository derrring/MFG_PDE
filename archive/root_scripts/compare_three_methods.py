#!/usr/bin/env python3
"""
Compare three MFG solver methods using the same equation:
1. FDM (Finite Difference Method)
2. Hybrid (FDM-HJB + Particle-FP)
3. Particle-Collocation (GFDM-HJB + Particle-FP)
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem

# Import existing solvers and create simple implementations for comparison
# We'll focus on particle-collocation vs simplified reference methods


def compare_three_methods():
    print("=" * 70)
    print("PARTICLE-COLLOCATION METHOD COMPARISON")
    print("=" * 70)
    print("Comparing: Standard vs QP-Constrained vs High-Order Collocation")
    print("Same MFG equation, same boundary conditions, same parameters")

    # Use balanced parameters that work well for all methods
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 60,  # High resolution
        "T": 1.0,  # Full time horizon
        "Nt": 50,  # Adequate time steps
        "sigma": 0.2,  # Balanced diffusion
        "coefCT": 0.05,  # Balanced coupling
    }

    print(f"\nProblem Parameters:")
    for key, value in problem_params.items():
        print(f"  {key}: {value}")

    # Create common MFG problem
    problem = ExampleMFGProblem(**problem_params)

    # No-flux boundary conditions for all methods
    no_flux_bc = BoundaryConditions(type="no_flux")

    # Storage for results
    results = {}

    print(f"\n{'='*70}")
    print("METHOD 1: STANDARD PARTICLE-COLLOCATION")
    print(f"{'='*70}")
    print("Standard weighted least squares for derivative approximation")

    try:
        start_time = time.time()

        # Collocation setup
        num_collocation_points = 15
        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)

        # Boundary indices
        boundary_tolerance = 1e-10
        boundary_indices = []
        for i, point in enumerate(collocation_points):
            x = point[0]
            if abs(x - problem.xmin) < boundary_tolerance or abs(x - problem.xmax) < boundary_tolerance:
                boundary_indices.append(i)
        boundary_indices = np.array(boundary_indices)

        standard_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=400,
            delta=0.4,
            taylor_order=1,  # First-order
            weight_function="wendland",
            NiterNewton=12,
            l2errBoundNewton=1e-4,
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=boundary_indices,
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=False,  # NO QP constraints
        )

        U_std, M_std, info_std = standard_solver.solve(Niter=8, l2errBound=2e-3, verbose=False)

        time_std = time.time() - start_time

        if M_std is not None and U_std is not None:
            # Mass conservation analysis
            mass_std = np.sum(M_std * problem.Dx, axis=1)
            mass_change_std = abs(mass_std[-1] - mass_std[0])
            mass_variation_std = np.max(mass_std) - np.min(mass_std)

            # Solution metrics
            max_U_std = np.max(np.abs(U_std))
            max_M_std = np.max(M_std)

            # Particle trajectory analysis
            particles_trajectory = standard_solver.get_particles_trajectory()
            violations_std = 0
            if particles_trajectory is not None:
                final_particles = particles_trajectory[-1, :]
                violations_std = np.sum((final_particles < problem.xmin) | (final_particles > problem.xmax))

            results['standard'] = {
                'success': True,
                'mass_change': mass_change_std,
                'mass_variation': mass_variation_std,
                'max_U': max_U_std,
                'max_M': max_M_std,
                'violations': violations_std,
                'time': time_std,
                'converged': info_std.get('converged', False),
                'iterations': info_std.get('iterations', 0),
                'U_solution': U_std,
                'M_solution': M_std,
            }

            print(f"\nStandard Particle-Collocation Results:")
            print(f"  Mass change: {mass_change_std:.3e}")
            print(f"  Mass variation: {mass_variation_std:.3e}")
            print(f"  Max |U|: {max_U_std:.2e}")
            print(f"  Max M: {max_M_std:.3f}")
            print(f"  Boundary violations: {violations_std}")
            print(f"  Runtime: {time_std:.2f}s")
            print(f"  Converged: {info_std.get('converged', False)}")
            print(f"  Iterations: {info_std.get('iterations', 0)}")

        else:
            print("❌ Standard method failed to produce valid solutions")
            results['standard'] = {'success': False}

    except Exception as e:
        print(f"❌ Standard method crashed: {e}")
        results['standard'] = {'success': False, 'error': str(e)}

    print(f"\n{'='*70}")
    print("METHOD 2: QP-CONSTRAINED PARTICLE-COLLOCATION")
    print(f"{'='*70}")
    print("Constrained QP for monotonicity preservation")

    try:
        start_time = time.time()

        qp_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=400,  # Same as standard
            delta=0.4,
            taylor_order=1,  # First-order for fair comparison
            weight_function="wendland",
            NiterNewton=12,
            l2errBoundNewton=1e-4,
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=boundary_indices,
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,  # ENABLE QP constraints
        )

        U_qp, M_qp, info_qp = qp_solver.solve(Niter=8, l2errBound=2e-3, verbose=False)

        time_qp = time.time() - start_time

        if M_qp is not None and U_qp is not None:
            # Mass conservation analysis
            mass_qp = np.sum(M_qp * problem.Dx, axis=1)
            mass_change_qp = abs(mass_qp[-1] - mass_qp[0])
            mass_variation_qp = np.max(mass_qp) - np.min(mass_qp)

            # Solution metrics
            max_U_qp = np.max(np.abs(U_qp))
            max_M_qp = np.max(M_qp)

            # Particle trajectory analysis
            violations_qp = 0
            particles_trajectory = qp_solver.get_particles_trajectory()
            if particles_trajectory is not None:
                final_particles = particles_trajectory[-1, :]
                violations_qp = np.sum((final_particles < problem.xmin) | (final_particles > problem.xmax))

            results['qp'] = {
                'success': True,
                'mass_change': mass_change_qp,
                'mass_variation': mass_variation_qp,
                'max_U': max_U_qp,
                'max_M': max_M_qp,
                'violations': violations_qp,
                'time': time_qp,
                'converged': info_qp.get('converged', False),
                'iterations': info_qp.get('iterations', 0),
                'U_solution': U_qp,
                'M_solution': M_qp,
            }

            print(f"\nQP-Constrained Results:")
            print(f"  Mass change: {mass_change_qp:.3e}")
            print(f"  Mass variation: {mass_variation_qp:.3e}")
            print(f"  Max |U|: {max_U_qp:.2e}")
            print(f"  Max M: {max_M_qp:.3f}")
            print(f"  Boundary violations: {violations_qp}")
            print(f"  Runtime: {time_qp:.2f}s")
            print(f"  Converged: {info_qp.get('converged', False)}")
            print(f"  Iterations: {info_qp.get('iterations', 0)}")

        else:
            print("❌ QP method failed to produce valid solutions")
            results['qp'] = {'success': False}

    except Exception as e:
        print(f"❌ QP method crashed: {e}")
        results['qp'] = {'success': False, 'error': str(e)}

    print(f"\n{'='*70}")
    print("METHOD 3: HIGH-ORDER PARTICLE-COLLOCATION")
    print(f"{'='*70}")
    print("Second-order Taylor expansion with QP constraints")

    try:
        start_time = time.time()

        # Collocation points for HJB solver
        num_collocation_points = 15
        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)

        # Boundary indices
        boundary_tolerance = 1e-10
        boundary_indices = []
        for i, point in enumerate(collocation_points):
            x = point[0]
            if abs(x - problem.xmin) < boundary_tolerance or abs(x - problem.xmax) < boundary_tolerance:
                boundary_indices.append(i)
        boundary_indices = np.array(boundary_indices)

        high_order_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=400,  # Same as others for fair comparison
            delta=0.4,
            taylor_order=2,  # SECOND-ORDER for higher accuracy
            weight_function="wendland",
            NiterNewton=12,
            l2errBoundNewton=1e-4,
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=boundary_indices,
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,  # QP constraints + high order
        )

        U_ho, M_ho, info_ho = high_order_solver.solve(Niter=8, l2errBound=2e-3, verbose=False)

        time_ho = time.time() - start_time

        if M_ho is not None and U_ho is not None:
            # Mass conservation analysis
            mass_ho = np.sum(M_ho * problem.Dx, axis=1)
            mass_change_ho = abs(mass_ho[-1] - mass_ho[0])
            mass_variation_ho = np.max(mass_ho) - np.min(mass_ho)

            # Solution metrics
            max_U_ho = np.max(np.abs(U_ho))
            max_M_ho = np.max(M_ho)

            # Particle trajectory analysis
            particles_trajectory = high_order_solver.get_particles_trajectory()
            violations_ho = 0
            if particles_trajectory is not None:
                final_particles = particles_trajectory[-1, :]
                violations_ho = np.sum((final_particles < problem.xmin) | (final_particles > problem.xmax))

            results['high_order'] = {
                'success': True,
                'mass_change': mass_change_ho,
                'mass_variation': mass_variation_ho,
                'max_U': max_U_ho,
                'max_M': max_M_ho,
                'violations': violations_ho,
                'time': time_ho,
                'converged': info_ho.get('converged', False),
                'iterations': info_ho.get('iterations', 0),
                'U_solution': U_ho,
                'M_solution': M_ho,
            }

            print(f"\nHigh-Order Results:")
            print(f"  Mass change: {mass_change_ho:.3e}")
            print(f"  Mass variation: {mass_variation_ho:.3e}")
            print(f"  Max |U|: {max_U_ho:.2e}")
            print(f"  Max M: {max_M_ho:.3f}")
            print(f"  Boundary violations: {violations_ho}")
            print(f"  Runtime: {time_ho:.2f}s")
            print(f"  Converged: {info_ho.get('converged', False)}")
            print(f"  Iterations: {info_ho.get('iterations', 0)}")

        else:
            print("❌ High-Order method failed to produce valid solutions")
            results['high_order'] = {'success': False}

    except Exception as e:
        print(f"❌ High-Order method crashed: {e}")
        results['high_order'] = {'success': False, 'error': str(e)}

    # Comprehensive comparison
    print(f"\n{'='*80}")
    print("COMPREHENSIVE COMPARISON ANALYSIS")
    print(f"{'='*80}")

    successful_methods = [
        method for method in ['standard', 'qp', 'high_order'] if results.get(method, {}).get('success', False)
    ]

    if len(successful_methods) >= 2:
        print(f"\nSuccessful methods: {', '.join(successful_methods).upper()}")

        # Create simplified comparison table
        print(f"\n{'Metric':<20} {'Standard':<15} {'QP':<15} {'High-Order':<15}")
        print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15}")

        # Mass conservation
        metrics = ['mass_change', 'max_U', 'time', 'violations']
        metric_names = ['Mass change', 'Max |U|', 'Runtime (s)', 'Boundary violations']

        for i, metric in enumerate(metrics):
            row = [metric_names[i]]
            for method in ['standard', 'qp', 'high_order']:
                if method in successful_methods:
                    value = results[method].get(metric, 0)
                    if metric == 'mass_change' or metric == 'max_U':
                        row.append(f"{value:.2e}")
                    elif metric == 'time':
                        row.append(f"{value:.2f}")
                    else:
                        row.append(str(value))
                else:
                    row.append("FAILED")

            print(f"{row[0]:<20} {row[1]:<15} {row[2]:<15} {row[3]:<15}")

        # Convergence status
        conv_row = ['Converged']
        for method in ['standard', 'qp', 'high_order']:
            if method in successful_methods:
                converged = results[method]['converged']
                conv_row.append("Yes" if converged else "No")
            else:
                conv_row.append("FAILED")

        print(f"{conv_row[0]:<20} {conv_row[1]:<15} {conv_row[2]:<15} {conv_row[3]:<15}")

        print(f"\n{'='*80}")
        print("METHOD ASSESSMENT")
        print(f"{'='*80}")

        # Find best performer in each category
        best_mass = min(successful_methods, key=lambda m: results[m]['mass_change'])
        best_stability = min(successful_methods, key=lambda m: results[m]['max_U'])
        fastest = min(successful_methods, key=lambda m: results[m]['time'])

        print(f"\n🏆 Best mass conservation: {best_mass.upper()}")
        print(f"🏆 Best solution stability: {best_stability.upper()}")
        print(f"🏆 Fastest runtime: {fastest.upper()}")

        # Overall recommendation
        print(f"\n--- Method Characteristics ---")
        if 'standard' in successful_methods:
            print(f"✓ Standard: Basic weighted least squares, fast, unconstrained")
        if 'qp' in successful_methods:
            print(f"✓ QP: Constrained optimization, monotonicity preservation, boundary compliance")
        if 'high_order' in successful_methods:
            print(f"✓ High-Order: Second-order accuracy, QP constraints, higher computational cost")

        # Create plots if solutions available
        if len(successful_methods) >= 2:
            print(f"\n--- Plotting Comparison ---")
            create_comparison_plots(results, problem, successful_methods)

    else:
        print(f"\nInsufficient successful methods for comparison.")
        for method in ['standard', 'qp', 'high_order']:
            if not results.get(method, {}).get('success', False):
                error = results.get(method, {}).get('error', 'Unknown error')
                print(f"❌ {method.upper()}: {error}")


def create_comparison_plots(results, problem, successful_methods):
    """Create comparison plots for successful methods"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('MFG Solver Method Comparison', fontsize=16)

    # Plot 1: Final density profiles
    ax1 = axes[0, 0]
    for method in successful_methods:
        M_solution = results[method]['M_solution']
        final_density = M_solution[-1, :]
        ax1.plot(problem.xSpace, final_density, label=method.upper(), linewidth=2)
    ax1.set_xlabel('Space x')
    ax1.set_ylabel('Final Density M(T,x)')
    ax1.set_title('Final Density Comparison')
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Final value function profiles
    ax2 = axes[0, 1]
    for method in successful_methods:
        U_solution = results[method]['U_solution']
        final_value = U_solution[-1, :]
        ax2.plot(problem.xSpace, final_value, label=method.upper(), linewidth=2)
    ax2.set_xlabel('Space x')
    ax2.set_ylabel('Final Value U(T,x)')
    ax2.set_title('Final Value Function Comparison')
    ax2.grid(True)
    ax2.legend()

    # Plot 3: Mass conservation over time
    ax3 = axes[1, 0]
    for method in successful_methods:
        M_solution = results[method]['M_solution']
        mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
        ax3.plot(problem.tSpace, mass_evolution, label=method.upper(), linewidth=2)
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Total Mass ∫M(t,x)dx')
    ax3.set_title('Mass Conservation')
    ax3.grid(True)
    ax3.legend()

    # Plot 4: Performance metrics
    ax4 = axes[1, 1]
    methods = []
    mass_changes = []
    runtimes = []

    for method in successful_methods:
        methods.append(method.upper())
        mass_changes.append(results[method]['mass_change'])
        runtimes.append(results[method]['time'])

    x_pos = np.arange(len(methods))
    ax4_twin = ax4.twinx()

    bars1 = ax4.bar(x_pos - 0.2, mass_changes, 0.4, label='Mass Change', alpha=0.7)
    bars2 = ax4_twin.bar(x_pos + 0.2, runtimes, 0.4, label='Runtime (s)', alpha=0.7, color='orange')

    ax4.set_xlabel('Method')
    ax4.set_ylabel('Mass Change', color='blue')
    ax4_twin.set_ylabel('Runtime (seconds)', color='orange')
    ax4.set_title('Performance Metrics')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(methods)
    ax4.set_yscale('log')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_three_methods()
