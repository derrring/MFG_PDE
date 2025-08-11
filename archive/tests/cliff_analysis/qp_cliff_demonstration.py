#!/usr/bin/env python3
"""
QP-Collocation Cliff Demonstration
Quick demonstration of the sudden mass loss "cliff" phenomenon.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def demonstrate_cliff_behavior():
    """Demonstrate the sudden mass loss cliff with a controlled example"""
    print("=" * 80)
    print("QP-COLLOCATION CLIFF BEHAVIOR DEMONSTRATION")
    print("=" * 80)
    print("Demonstrating the sudden 90-degree mass loss phenomenon")

    # Test two scenarios: one stable, one that triggers cliff
    scenarios = [
        {
            "name": "Stable Case",
            "params": {
                "T": 1.0,
                "Nt": 50,
                "Nx": 30,
                "sigma": 0.2,
                "coefCT": 0.05,
                "num_particles": 500,
                "delta": 0.4,
                "max_iter": 10,
            },
            "description": "Conservative parameters - should remain stable",
        },
        {
            "name": "Cliff Case",
            "params": {
                "T": 2.0,
                "Nt": 40,
                "Nx": 60,
                "sigma": 0.1,
                "coefCT": 0.02,
                "num_particles": 1500,
                "delta": 0.2,
                "max_iter": 20,
            },
            "description": "Aggressive parameters - likely to trigger cliff",
        },
    ]

    results = {}
    no_flux_bc = BoundaryConditions(type="no_flux")

    for scenario in scenarios:
        name = scenario["name"]
        params = scenario["params"]
        description = scenario["description"]

        print(f"\n{'='*50}")
        print(f"SCENARIO: {name}")
        print(f"{'='*50}")
        print(f"Description: {description}")

        # Create problem
        problem_params = {
            "xmin": 0.0,
            "xmax": 1.0,
            "T": params["T"],
            "Nt": params["Nt"],
            "Nx": params["Nx"],
            "sigma": params["sigma"],
            "coefCT": params["coefCT"],
        }

        try:
            problem = ExampleMFGProblem(**problem_params)

            print(f"Problem: T={params['T']}, Dt={problem.Dt:.4f}, Dx={problem.Dx:.4f}")
            print(f"CFL = {params['sigma']**2 * problem.Dt / problem.Dx**2:.3f}")

            # Setup solver
            num_collocation_points = 8  # Minimal for speed
            collocation_points = np.linspace(0, 1, num_collocation_points).reshape(-1, 1)
            boundary_indices = [0, num_collocation_points - 1]

            solver_params = {
                "num_particles": params["num_particles"],
                "delta": params["delta"],
                "taylor_order": 1,  # First order for speed
                "weight_function": "wendland",
                "NiterNewton": 5,
                "l2errBoundNewton": 1e-4,
                "kde_bandwidth": "scott",
                "normalize_kde_output": False,
                "use_monotone_constraints": True,
            }

            solver = ParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                boundary_indices=np.array(boundary_indices),
                boundary_conditions=no_flux_bc,
                **solver_params,
            )

            print(f"Starting {name} simulation...")
            start_time = time.time()

            U_solution, M_solution, solve_info = solver.solve(Niter=params["max_iter"], l2errBound=1e-3, verbose=True)

            total_time = time.time() - start_time

            if U_solution is not None and M_solution is not None:
                # Analyze mass evolution for cliff detection
                mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
                initial_mass = mass_evolution[0]
                final_mass = mass_evolution[-1]

                # Detect cliff: sudden large drops
                mass_diff = np.diff(mass_evolution)
                mass_diff_percent = (mass_diff / initial_mass) * 100

                # Find the biggest single-step loss
                biggest_loss_step = np.argmin(mass_diff_percent)
                biggest_loss_percent = mass_diff_percent[biggest_loss_step]
                biggest_loss_time = problem.tSpace[biggest_loss_step]

                # Cliff criteria
                cliff_detected = biggest_loss_percent < -10  # >10% loss in one step
                mass_collapse = final_mass < 0.1 * initial_mass

                # Check particle violations
                particles_trajectory = solver.fp_solver.M_particles_trajectory
                total_violations = 0
                if particles_trajectory is not None:
                    for t_step in range(particles_trajectory.shape[0]):
                        step_particles = particles_trajectory[t_step, :]
                        violations = np.sum((step_particles < -0.01) | (step_particles > 1.01))
                        total_violations += violations

                results[name] = {
                    'success': True,
                    'time': total_time,
                    'mass_initial': initial_mass,
                    'mass_final': final_mass,
                    'total_change_percent': (final_mass - initial_mass) / initial_mass * 100,
                    'biggest_loss_percent': biggest_loss_percent,
                    'biggest_loss_time': biggest_loss_time,
                    'cliff_detected': cliff_detected,
                    'mass_collapse': mass_collapse,
                    'total_violations': total_violations,
                    'arrays': {
                        'mass_evolution': mass_evolution,
                        'tSpace': problem.tSpace,
                        'M_solution': M_solution,
                        'xSpace': problem.xSpace,
                    },
                }

                print(f"✓ {name} completed in {total_time:.1f}s")
                print(f"  Mass: {initial_mass:.6f} → {final_mass:.6f}")
                print(f"  Total change: {(final_mass-initial_mass)/initial_mass*100:+.2f}%")
                print(f"  Biggest single loss: {biggest_loss_percent:.2f}% at t={biggest_loss_time:.3f}")
                print(f"  Particle violations: {total_violations}")

                if mass_collapse:
                    print(f"  ❌ MASS COLLAPSE: Final mass < 10% of initial")
                elif cliff_detected:
                    print(f"  ⚠️  CLIFF DETECTED: >10% single-step loss")
                    print(f"      The 'cliff' occurred at t = {biggest_loss_time:.3f}")
                    print(f"      Mass before: {mass_evolution[biggest_loss_step]:.6f}")
                    print(f"      Mass after: {mass_evolution[biggest_loss_step+1]:.6f}")
                    print(f"      This demonstrates the sudden '90-degree' mass loss you observed!")
                else:
                    print(f"  ✅ STABLE: No cliff behavior detected")

            else:
                results[name] = {'success': False}
                print(f"❌ {name} failed to produce solution")

        except Exception as e:
            results[name] = {'success': False, 'error': str(e)}
            print(f"❌ {name} crashed: {e}")

    # Create comparison plot
    create_cliff_demonstration_plot(results)

    # Explain the cliff phenomenon
    explain_cliff_phenomenon(results)

    return results


def create_cliff_demonstration_plot(results):
    """Create a plot clearly showing the cliff behavior"""
    successful = [(name, result) for name, result in results.items() if result.get('success', False)]

    if len(successful) < 1:
        print("No successful results to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('QP-Collocation Cliff Behavior Demonstration', fontsize=16, fontweight='bold')

    colors = ['green', 'red']

    # 1. Mass evolution comparison
    ax1 = axes[0, 0]
    for i, (name, result) in enumerate(successful):
        arrays = result['arrays']
        color = colors[i % len(colors)]

        if result['cliff_detected']:
            linestyle = '--'
            linewidth = 3
            alpha = 0.9
        else:
            linestyle = '-'
            linewidth = 2
            alpha = 0.8

        ax1.plot(
            arrays['tSpace'],
            arrays['mass_evolution'],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=f"{name}",
        )

        # Mark cliff point if detected
        if result['cliff_detected']:
            cliff_time = result['biggest_loss_time']
            cliff_idx = np.argmin(np.abs(arrays['tSpace'] - cliff_time))
            cliff_mass = arrays['mass_evolution'][cliff_idx]
            ax1.plot(cliff_time, cliff_mass, 'ro', markersize=10, label=f"Cliff at t={cliff_time:.3f}")

    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Total Mass')
    ax1.set_title('Mass Evolution: Cliff vs Stable')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Mass change rate (derivative)
    ax2 = axes[0, 1]
    for i, (name, result) in enumerate(successful):
        arrays = result['arrays']
        color = colors[i % len(colors)]

        # Calculate mass change rate
        mass_diff = np.diff(arrays['mass_evolution'])
        time_centers = (arrays['tSpace'][:-1] + arrays['tSpace'][1:]) / 2

        ax2.plot(time_centers, mass_diff, color=color, linewidth=2, alpha=0.8, label=f"{name} rate")

        # Mark cliff point
        if result['cliff_detected']:
            cliff_time = result['biggest_loss_time']
            cliff_idx = np.argmin(np.abs(time_centers - cliff_time))
            cliff_rate = mass_diff[cliff_idx]
            ax2.plot(cliff_time, cliff_rate, 'ro', markersize=10)

    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Mass Change Rate')
    ax2.set_title('Mass Change Rate (Shows Cliff as Spike)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Final density profiles
    ax3 = axes[1, 0]
    for i, (name, result) in enumerate(successful):
        arrays = result['arrays']
        color = colors[i % len(colors)]

        final_density = arrays['M_solution'][-1, :]

        ax3.plot(arrays['xSpace'], final_density, color=color, linewidth=2, alpha=0.8, label=f"{name} final")

    ax3.set_xlabel('Space x')
    ax3.set_ylabel('Final Density')
    ax3.set_title('Final Density Profiles')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Summary statistics
    ax4 = axes[1, 1]

    if len(successful) >= 2:
        names = [name for name, _ in successful]
        mass_changes = [result['total_change_percent'] for _, result in successful]
        cliff_losses = [result['biggest_loss_percent'] for _, result in successful]

        x_pos = np.arange(len(names))
        width = 0.35

        bars1 = ax4.bar(x_pos - width / 2, mass_changes, width, label='Total Mass Change (%)', alpha=0.7)
        bars2 = ax4.bar(x_pos + width / 2, cliff_losses, width, label='Biggest Single Loss (%)', alpha=0.7)

        ax4.set_xlabel('Scenario')
        ax4.set_ylabel('Percentage (%)')
        ax4.set_title('Mass Change Summary')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(names)
        ax4.legend()
        ax4.grid(True, axis='y', alpha=0.3)

        # Add value labels
        for bar, value in zip(bars1, mass_changes):
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f'{value:.1f}%',
                ha='center',
                va='bottom',
                fontsize=9,
            )
        for bar, value in zip(bars2, cliff_losses):
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f'{value:.1f}%',
                ha='center',
                va='bottom',
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(
        '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/qp_cliff_demonstration.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.show()


def explain_cliff_phenomenon(results):
    """Explain the cliff phenomenon based on results"""
    print(f"\n{'='*80}")
    print("CLIFF PHENOMENON EXPLANATION")
    print(f"{'='*80}")

    cliff_cases = [
        name for name, result in results.items() if result.get('success', False) and result.get('cliff_detected', False)
    ]
    stable_cases = [
        name
        for name, result in results.items()
        if result.get('success', False) and not result.get('cliff_detected', False)
    ]

    print(f"🔬 ANALYSIS OF THE SUDDEN MASS LOSS 'CLIFF':")
    print(f"")

    if cliff_cases:
        print(f"✅ CLIFF DEMONSTRATED in: {', '.join(cliff_cases)}")
        for name in cliff_cases:
            result = results[name]
            print(f"   {name}: {result['biggest_loss_percent']:.1f}% loss at t={result['biggest_loss_time']:.3f}")
    else:
        print(f"❌ NO CLIFF DETECTED in current test cases")

    if stable_cases:
        print(f"✅ STABLE BEHAVIOR in: {', '.join(stable_cases)}")

    print(f"\n🧠 WHY THE CLIFF OCCURS:")
    print(f"The sudden 'vertical drop' in mass happens due to:")
    print(f"")
    print(f"1. PARTICLE EXPLOSION:")
    print(f"   - Numerical instability causes particles to suddenly escape domain")
    print(f"   - When particles leave [0,1], they no longer contribute to mass")
    print(f"   - This creates the sharp 90-degree drop you observed")
    print(f"")
    print(f"2. QP SOLVER BREAKDOWN:")
    print(f"   - Quadratic programming constraints become infeasible")
    print(f"   - Solver falls back to unconstrained solution")
    print(f"   - Loss of monotonicity control leads to numerical blow-up")
    print(f"")
    print(f"3. KERNEL BANDWIDTH COLLAPSE:")
    print(f"   - Adaptive KDE bandwidth becomes too small")
    print(f"   - Particle contributions vanish from density estimation")
    print(f"   - Effective 'loss' of mass from numerical representation")
    print(f"")
    print(f"4. ACCUMULATING ERRORS:")
    print(f"   - Small numerical errors compound over time")
    print(f"   - Eventually trigger catastrophic failure")
    print(f"   - Results in sudden, not gradual, mass loss")
    print(f"")
    print(f"🎯 KEY INSIGHT:")
    print(f"The cliff is NOT a gradual numerical diffusion but a sudden")
    print(f"catastrophic failure when numerical stability is lost.")
    print(f"This is why you see a sharp 'cliff' rather than smooth decay.")


if __name__ == "__main__":
    print("Starting QP-Collocation Cliff Demonstration...")
    print("Quick test to demonstrate sudden mass loss phenomenon")
    print("Expected execution time: 2-5 minutes")

    try:
        results = demonstrate_cliff_behavior()
        print("\n" + "=" * 80)
        print("CLIFF DEMONSTRATION COMPLETED")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback

        traceback.print_exc()
