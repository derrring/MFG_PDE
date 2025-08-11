#!/usr/bin/env python3
"""
Damping Factor Mass Instability Study
Systematic analysis of how thetaUM damping parameter affects mass evolution in hybrid methods.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.damped_fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.fp_solvers.particle_fp import ParticleFPSolver

# Import hybrid solver components
from mfg_pde.alg.hjb_solvers.fdm_hjb import FdmHJBSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def run_damping_mass_study():
    """Study how different thetaUM values affect mass evolution"""
    print("=" * 80)
    print("DAMPING FACTOR MASS INSTABILITY STUDY")
    print("=" * 80)
    print("Systematic analysis of thetaUM parameter effects on mass evolution")

    # Fixed problem parameters
    base_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 20,
        "T": 2.0,  # Use T=2 to see effects
        "Nt": 100,
        "sigma": 0.2,  # Moderate diffusion
        "coefCT": 0.03,  # Light coupling
    }

    # Fixed solver parameters
    solver_params = {
        "num_particles": 300,
        "max_iterations": 15,
        "convergence_tolerance": 1e-3,
        "newton_iterations": 8,
        "newton_tolerance": 1e-4,
    }

    # Test different damping factors
    damping_values = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]

    print(f"\nBase Parameters:")
    for key, value in base_params.items():
        print(f"  {key}: {value}")

    print(f"\nSolver Parameters:")
    for key, value in solver_params.items():
        print(f"  {key}: {value}")

    print(f"\nTesting thetaUM values: {damping_values}")

    results = {}
    no_flux_bc = BoundaryConditions(type="no_flux")
    problem = ExampleMFGProblem(**base_params)

    for theta in damping_values:
        print(f"\n{'='*60}")
        print(f"TESTING thetaUM = {theta}")
        print(f"{'='*60}")

        try:
            start_time = time.time()

            # Setup hybrid solver
            hjb_solver = FdmHJBSolver(
                problem,
                NiterNewton=solver_params["newton_iterations"],
                l2errBoundNewton=solver_params["newton_tolerance"],
            )

            fp_solver = ParticleFPSolver(
                problem,
                num_particles=solver_params["num_particles"],
                kde_bandwidth="scott",
                normalize_kde_output=False,
                boundary_conditions=no_flux_bc,
            )

            hybrid_iterator = FixedPointIterator(
                problem,
                hjb_solver=hjb_solver,
                fp_solver=fp_solver,
                thetaUM=theta,  # Variable damping parameter
            )

            U_hybrid, M_hybrid, iters, _, _ = hybrid_iterator.solve(
                solver_params["max_iterations"], solver_params["convergence_tolerance"]
            )

            solve_time = time.time() - start_time

            if U_hybrid is not None and M_hybrid is not None:
                # Calculate mass evolution metrics
                mass_evolution = np.sum(M_hybrid * problem.Dx, axis=1)
                initial_mass = mass_evolution[0]
                final_mass = mass_evolution[-1]

                # Calculate early-time mass loss (first 20% of simulation)
                early_idx = int(0.2 * len(mass_evolution))
                early_mass = mass_evolution[early_idx]
                early_loss = (initial_mass - early_mass) / initial_mass * 100

                # Calculate maximum mass deviation
                min_mass = np.min(mass_evolution)
                max_mass = np.max(mass_evolution)
                max_deviation = max(abs(min_mass - initial_mass), abs(max_mass - initial_mass)) / initial_mass * 100

                # Calculate mass oscillation (standard deviation of changes)
                mass_changes = np.diff(mass_evolution)
                mass_oscillation = np.std(mass_changes) / initial_mass * 100

                # Final mass conservation
                final_change = (final_mass - initial_mass) / initial_mass * 100

                results[theta] = {
                    'success': True,
                    'time': solve_time,
                    'iterations': iters,
                    'mass_evolution': mass_evolution,
                    'initial_mass': initial_mass,
                    'final_mass': final_mass,
                    'early_loss_percent': early_loss,
                    'max_deviation_percent': max_deviation,
                    'mass_oscillation_percent': mass_oscillation,
                    'final_change_percent': final_change,
                    'U': U_hybrid,
                    'M': M_hybrid,
                }

                print(f"  ✓ Completed in {solve_time:.2f}s ({iters} iterations)")
                print(f"    Early mass loss (first 20%): {early_loss:+.3f}%")
                print(f"    Maximum deviation: {max_deviation:.3f}%")
                print(f"    Mass oscillation: {mass_oscillation:.3f}%")
                print(f"    Final mass change: {final_change:+.3f}%")

            else:
                results[theta] = {'success': False, 'error': 'Solver failed'}
                print(f"  ❌ Failed to solve")

        except Exception as e:
            results[theta] = {'success': False, 'error': str(e)}
            print(f"  ❌ Exception: {e}")

    # Analysis
    analyze_damping_effects(results, problem)
    create_damping_plots(results, problem)

    return results


def analyze_damping_effects(results, problem):
    """Analyze how damping factor affects mass stability"""
    print(f"\n{'='*80}")
    print("DAMPING FACTOR ANALYSIS")
    print(f"{'='*80}")

    successful_results = {k: v for k, v in results.items() if v.get('success', False)}

    if not successful_results:
        print("No successful results to analyze")
        return

    print(f"\n{'thetaUM':<8} {'Early Loss %':<12} {'Max Dev %':<12} {'Oscillation %':<12} {'Final %':<10} {'Status'}")
    print("-" * 75)

    for theta in sorted(successful_results.keys()):
        r = successful_results[theta]
        early_loss = r['early_loss_percent']
        max_dev = r['max_deviation_percent']
        oscillation = r['mass_oscillation_percent']
        final_change = r['final_change_percent']

        # Classify stability
        if max_dev < 1.0 and oscillation < 0.5:
            status = "Stable"
        elif max_dev < 3.0 and oscillation < 1.0:
            status = "Moderate"
        else:
            status = "Unstable"

        print(
            f"{theta:<8.2f} {early_loss:<+12.3f} {max_dev:<12.3f} {oscillation:<12.3f} {final_change:<+10.3f} {status}"
        )

    # Find stability threshold
    print(f"\n--- STABILITY ANALYSIS ---")

    stable_thetas = []
    unstable_thetas = []

    for theta in sorted(successful_results.keys()):
        r = successful_results[theta]
        if r['max_deviation_percent'] < 2.0 and r['mass_oscillation_percent'] < 1.0:
            stable_thetas.append(theta)
        else:
            unstable_thetas.append(theta)

    if stable_thetas and unstable_thetas:
        stability_threshold = (max(stable_thetas) + min(unstable_thetas)) / 2
        print(f"Approximate stability threshold: thetaUM ≈ {stability_threshold:.2f}")

    print(f"Stable damping values: {stable_thetas}")
    print(f"Unstable damping values: {unstable_thetas}")


def create_damping_plots(results, problem):
    """Create plots showing damping factor effects"""
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}

    if len(successful_results) < 2:
        print("Insufficient results for plotting")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Damping Factor Effects on Mass Evolution', fontsize=14, fontweight='bold')

    colors = plt.cm.viridis(np.linspace(0, 1, len(successful_results)))

    # Plot 1: Mass evolution time series
    for (theta, result), color in zip(sorted(successful_results.items()), colors):
        ax1.plot(problem.tSpace, result['mass_evolution'], color=color, linewidth=2, label=f'θ={theta:.2f}')

    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Total Mass')
    ax1.set_title('Mass Evolution vs Damping Factor')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Early mass loss vs damping
    thetas = sorted(successful_results.keys())
    early_losses = [successful_results[theta]['early_loss_percent'] for theta in thetas]

    ax2.plot(thetas, early_losses, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Damping Factor (thetaUM)')
    ax2.set_ylabel('Early Mass Loss (%)')
    ax2.set_title('Early Mass Loss vs Damping')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Plot 3: Maximum deviation vs damping
    max_devs = [successful_results[theta]['max_deviation_percent'] for theta in thetas]

    ax3.plot(thetas, max_devs, 'bo-', linewidth=2, markersize=6)
    ax3.set_xlabel('Damping Factor (thetaUM)')
    ax3.set_ylabel('Maximum Mass Deviation (%)')
    ax3.set_title('Mass Stability vs Damping')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Mass oscillation vs damping
    oscillations = [successful_results[theta]['mass_oscillation_percent'] for theta in thetas]

    ax4.plot(thetas, oscillations, 'go-', linewidth=2, markersize=6)
    ax4.set_xlabel('Damping Factor (thetaUM)')
    ax4.set_ylabel('Mass Oscillation (%)')
    ax4.set_title('Mass Oscillation vs Damping')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/tests/stability_analysis/damping_factor_mass_study.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.show()

    print(f"\n✅ Analysis plots saved: damping_factor_mass_study.png")


if __name__ == "__main__":
    print("Starting Damping Factor Mass Instability Study...")
    print("Systematic analysis of thetaUM effects on hybrid method mass conservation")

    try:
        results = run_damping_mass_study()
        print("\n" + "=" * 80)
        print("DAMPING FACTOR STUDY COMPLETED")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\nStudy interrupted by user.")
    except Exception as e:
        print(f"\nStudy failed: {e}")
        import traceback

        traceback.print_exc()
