#!/usr/bin/env python3
"""
Mass Loss Analysis for Hybrid Method
Systematic analysis of when hybrid method shows initial mass loss vs smooth evolution.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.damped_fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.fp_solvers.particle_fp import ParticleFPSolver
from mfg_pde.alg.hjb_solvers.fdm_hjb import FdmHJBSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def test_hybrid_mass_evolution(T, thetaUM, sigma, coefCT, num_particles, newton_iterations, label):
    """Test hybrid method with specific parameters and return mass evolution"""

    # Problem parameters
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 25,
        "T": T,
        "Nt": max(25, int(T * 50)),  # Adaptive time steps
        "sigma": sigma,
        "coefCT": coefCT,
    }

    problem = ExampleMFGProblem(**problem_params)
    no_flux_bc = BoundaryConditions(type="no_flux")

    print(f"\n=== Testing {label} ===")
    print(f"T={T}, thetaUM={thetaUM}, sigma={sigma}, coefCT={coefCT}")
    print(f"particles={num_particles}, newton_iter={newton_iterations}")

    try:
        # Setup hybrid solver
        hjb_solver = FdmHJBSolver(problem, NiterNewton=newton_iterations, l2errBoundNewton=1e-4)

        fp_solver = ParticleFPSolver(
            problem,
            num_particles=num_particles,
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_conditions=no_flux_bc,
        )

        hybrid_iterator = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            thetaUM=thetaUM,
        )

        start_time = time.time()
        U_hybrid, M_hybrid, iters_hybrid, _, _ = hybrid_iterator.solve(15, 1e-3)  # max iterations  # tolerance
        solve_time = time.time() - start_time

        if U_hybrid is not None and M_hybrid is not None:
            # Calculate mass evolution
            mass_evolution = np.sum(M_hybrid * problem.Dx, axis=1)
            initial_mass = mass_evolution[0]
            final_mass = mass_evolution[-1]

            # Check for early mass loss (first 20% of time)
            early_steps = max(1, len(mass_evolution) // 5)
            early_mass_loss = initial_mass - np.min(mass_evolution[:early_steps])
            early_mass_loss_percent = (early_mass_loss / initial_mass) * 100

            # Overall mass change
            total_mass_change_percent = ((final_mass - initial_mass) / initial_mass) * 100

            # Violations
            violations = 0
            if hasattr(fp_solver, 'M_particles_trajectory') and fp_solver.M_particles_trajectory is not None:
                final_particles = fp_solver.M_particles_trajectory[-1, :]
                violations = np.sum((final_particles < problem.xmin - 1e-10) | (final_particles > problem.xmax + 1e-10))

            result = {
                'success': True,
                'label': label,
                'params': {
                    'T': T,
                    'thetaUM': thetaUM,
                    'sigma': sigma,
                    'coefCT': coefCT,
                    'particles': num_particles,
                    'newton_iter': newton_iterations,
                },
                'time': solve_time,
                'iterations': iters_hybrid,
                'mass_evolution': mass_evolution,
                'time_grid': problem.tSpace,
                'initial_mass': initial_mass,
                'final_mass': final_mass,
                'early_mass_loss_percent': early_mass_loss_percent,
                'total_mass_change_percent': total_mass_change_percent,
                'violations': violations,
                'early_loss_detected': early_mass_loss_percent > 1.0,  # Threshold for "initial mass loss"
                'smooth_evolution': early_mass_loss_percent < 0.5,  # Threshold for "smooth evolution"
            }

            print(f"✓ Success: {solve_time:.2f}s ({iters_hybrid} iterations)")
            print(f"  Early mass loss: {early_mass_loss_percent:.3f}%")
            print(f"  Total mass change: {total_mass_change_percent:.3f}%")
            print(f"  Violations: {violations}")
            print(f"  Classification: {'INITIAL MASS LOSS' if result['early_loss_detected'] else 'SMOOTH EVOLUTION'}")

            return result

        else:
            print("❌ Failed to solve")
            return {'success': False, 'label': label}

    except Exception as e:
        print(f"❌ Crashed: {e}")
        return {'success': False, 'label': label, 'error': str(e)}


def run_mass_loss_analysis():
    """Run systematic analysis of mass loss patterns"""
    print("=" * 80)
    print("HYBRID METHOD MASS LOSS ANALYSIS")
    print("=" * 80)
    print("Systematic analysis of initial mass loss vs smooth evolution patterns")

    test_cases = [
        # Case 1: Short time horizons (expected smooth evolution)
        (0.5, 0.5, 0.15, 0.02, 200, 6, "Short T, Mild damping"),
        (0.5, 0.7, 0.15, 0.02, 200, 6, "Short T, Aggressive damping"),
        # Case 2: Medium time horizons
        (1.0, 0.5, 0.15, 0.02, 250, 6, "Medium T, Mild damping"),
        (1.0, 0.7, 0.15, 0.02, 250, 8, "Medium T, Aggressive damping"),
        # Case 3: Longer time horizons (potential mass loss)
        (2.0, 0.5, 0.15, 0.05, 300, 8, "Long T, Mild damping"),
        (2.0, 0.7, 0.2, 0.05, 400, 8, "Long T, Aggressive damping"),
        # Case 4: Very aggressive parameters (likely mass loss)
        (2.0, 0.8, 0.3, 0.1, 500, 10, "Aggressive params"),
        # Case 5: Conservative parameters (should be smooth)
        (1.0, 0.3, 0.1, 0.01, 200, 4, "Conservative params"),
        # Case 6: High particle count effect
        (1.5, 0.5, 0.15, 0.03, 100, 6, "Low particles"),
        (1.5, 0.5, 0.15, 0.03, 600, 6, "High particles"),
        # Case 7: Newton iteration effect
        (1.5, 0.6, 0.2, 0.04, 300, 4, "Few Newton iter"),
        (1.5, 0.6, 0.2, 0.04, 300, 12, "Many Newton iter"),
    ]

    results = []

    for T, thetaUM, sigma, coefCT, num_particles, newton_iterations, label in test_cases:
        result = test_hybrid_mass_evolution(T, thetaUM, sigma, coefCT, num_particles, newton_iterations, label)
        results.append(result)

    # Analysis
    print("\n" + "=" * 80)
    print("MASS LOSS PATTERN ANALYSIS")
    print("=" * 80)

    successful_results = [r for r in results if r.get('success', False)]

    if len(successful_results) == 0:
        print("No successful tests to analyze")
        return results

    # Categorize results
    smooth_evolution = [r for r in successful_results if r['smooth_evolution']]
    initial_mass_loss = [r for r in successful_results if r['early_loss_detected']]
    intermediate = [r for r in successful_results if not r['smooth_evolution'] and not r['early_loss_detected']]

    print(f"\nCATEGORIZATION:")
    print(f"  Smooth Evolution: {len(smooth_evolution)}/{len(successful_results)}")
    print(f"  Initial Mass Loss: {len(initial_mass_loss)}/{len(successful_results)}")
    print(f"  Intermediate: {len(intermediate)}/{len(successful_results)}")

    # Pattern analysis
    print(f"\n--- SMOOTH EVOLUTION CASES ---")
    for r in smooth_evolution:
        p = r['params']
        print(f"  {r['label']}: T={p['T']}, thetaUM={p['thetaUM']}, particles={p['particles']}")
        print(f"    Early loss: {r['early_mass_loss_percent']:.3f}%, Total: {r['total_mass_change_percent']:.3f}%")

    print(f"\n--- INITIAL MASS LOSS CASES ---")
    for r in initial_mass_loss:
        p = r['params']
        print(f"  {r['label']}: T={p['T']}, thetaUM={p['thetaUM']}, particles={p['particles']}")
        print(f"    Early loss: {r['early_mass_loss_percent']:.3f}%, Total: {r['total_mass_change_percent']:.3f}%")

    # Key parameter analysis
    print(f"\n--- PARAMETER PATTERNS ---")

    # Time horizon analysis
    short_T = [r for r in successful_results if r['params']['T'] <= 1.0]
    long_T = [r for r in successful_results if r['params']['T'] > 1.0]

    short_T_loss = [r for r in short_T if r['early_loss_detected']]
    long_T_loss = [r for r in long_T if r['early_loss_detected']]

    print(f"Time Horizon Effect:")
    print(f"  Short T (≤1.0): {len(short_T_loss)}/{len(short_T)} show initial mass loss")
    print(f"  Long T (>1.0): {len(long_T_loss)}/{len(long_T)} show initial mass loss")

    # Damping analysis
    mild_damping = [r for r in successful_results if r['params']['thetaUM'] <= 0.5]
    aggressive_damping = [r for r in successful_results if r['params']['thetaUM'] > 0.5]

    mild_damping_loss = [r for r in mild_damping if r['early_loss_detected']]
    aggressive_damping_loss = [r for r in aggressive_damping if r['early_loss_detected']]

    print(f"Damping Effect:")
    print(f"  Mild damping (≤0.5): {len(mild_damping_loss)}/{len(mild_damping)} show initial mass loss")
    print(
        f"  Aggressive damping (>0.5): {len(aggressive_damping_loss)}/{len(aggressive_damping)} show initial mass loss"
    )

    # Particle count analysis
    low_particles = [r for r in successful_results if r['params']['particles'] <= 300]
    high_particles = [r for r in successful_results if r['params']['particles'] > 300]

    low_particles_loss = [r for r in low_particles if r['early_loss_detected']]
    high_particles_loss = [r for r in high_particles if r['early_loss_detected']]

    print(f"Particle Count Effect:")
    print(f"  Low particles (≤300): {len(low_particles_loss)}/{len(low_particles)} show initial mass loss")
    print(f"  High particles (>300): {len(high_particles_loss)}/{len(high_particles)} show initial mass loss")

    # Create visualization
    create_mass_loss_plots(successful_results)

    return results


def create_mass_loss_plots(results):
    """Create plots showing mass evolution patterns"""

    # Separate smooth vs mass loss cases
    smooth_cases = [r for r in results if r['smooth_evolution']]
    mass_loss_cases = [r for r in results if r['early_loss_detected']]

    if len(smooth_cases) == 0 and len(mass_loss_cases) == 0:
        print("No cases to plot")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hybrid Method Mass Evolution Analysis', fontsize=16)

    # Plot 1: Smooth evolution cases
    ax1.set_title('Smooth Evolution Cases')
    for r in smooth_cases[:5]:  # Limit to first 5
        ax1.plot(
            r['time_grid'],
            r['mass_evolution'],
            'g-',
            alpha=0.7,
            label=f"{r['label'][:15]}... ({r['early_mass_loss_percent']:.2f}%)",
        )
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Total Mass')
    ax1.grid(True)
    if smooth_cases:
        ax1.legend(fontsize=8)

    # Plot 2: Mass loss cases
    ax2.set_title('Initial Mass Loss Cases')
    for r in mass_loss_cases[:5]:  # Limit to first 5
        ax2.plot(
            r['time_grid'],
            r['mass_evolution'],
            'r-',
            alpha=0.7,
            label=f"{r['label'][:15]}... ({r['early_mass_loss_percent']:.2f}%)",
        )
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Total Mass')
    ax2.grid(True)
    if mass_loss_cases:
        ax2.legend(fontsize=8)

    # Plot 3: Parameter space analysis - T vs thetaUM
    ax3.set_title('Parameter Space: Time Horizon vs Damping')

    # Plot all points
    mass_loss_plotted = False
    smooth_plotted = False
    intermediate_plotted = False

    for r in results:
        p = r['params']
        if r['early_loss_detected']:
            label = 'Mass Loss' if not mass_loss_plotted else None
            ax3.scatter(p['T'], p['thetaUM'], c='red', s=60, alpha=0.7, label=label)
            mass_loss_plotted = True
        elif r['smooth_evolution']:
            label = 'Smooth' if not smooth_plotted else None
            ax3.scatter(p['T'], p['thetaUM'], c='green', s=60, alpha=0.7, label=label)
            smooth_plotted = True
        else:
            label = 'Intermediate' if not intermediate_plotted else None
            ax3.scatter(p['T'], p['thetaUM'], c='orange', s=60, alpha=0.7, label=label)
            intermediate_plotted = True

    ax3.set_xlabel('Time Horizon T')
    ax3.set_ylabel('Damping Parameter thetaUM')
    ax3.grid(True)
    ax3.legend()

    # Plot 4: Early mass loss vs parameters
    ax4.set_title('Early Mass Loss vs Parameters')

    # Create scatter plot with multiple parameters encoded
    for r in results:
        p = r['params']
        # Size = particles/10, color = early mass loss
        size = max(20, p['particles'] / 10)
        color = r['early_mass_loss_percent']

        scatter = ax4.scatter(p['T'], p['thetaUM'], c=color, s=size, alpha=0.7, cmap='Reds', vmin=0, vmax=5)

    ax4.set_xlabel('Time Horizon T')
    ax4.set_ylabel('Damping Parameter thetaUM')
    ax4.grid(True)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Early Mass Loss (%)')

    plt.tight_layout()
    plt.savefig(
        '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/mass_loss_analysis.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.show()

    print(f"\n✅ Mass loss analysis plot saved: mass_loss_analysis.png")


if __name__ == "__main__":
    print("Starting Hybrid Method Mass Loss Analysis...")
    print("Systematic testing of mass evolution patterns")
    print("Expected execution time: 10-20 minutes")

    try:
        results = run_mass_loss_analysis()
        print("\n" + "=" * 80)
        print("MASS LOSS ANALYSIS COMPLETED")
        print("=" * 80)
        print("Check the analysis above and generated plots for mass evolution patterns.")

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        import traceback

        traceback.print_exc()
