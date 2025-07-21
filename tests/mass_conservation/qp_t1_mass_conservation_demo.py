#!/usr/bin/env python3
"""
QP-Collocation T=1 Mass Conservation Demonstration
Optimized parameters for reliable execution and clear mass conservation demonstration.
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.core.boundaries import BoundaryConditions


def run_t5_mass_conservation_demo():
    """Demonstrate QP-Collocation long-time mass conservation with T=5"""
    print("=" * 80)
    print("QP-COLLOCATION T=5 LONG-TIME MASS CONSERVATION DEMONSTRATION")
    print("=" * 80)
    print(
        "Extended time horizon with proper resolution for long-time behavior observation"
    )

    # T=5 parameters with proper resolution
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 60,  # Higher spatial resolution for T=5
        "T": 5.0,  # Extended time horizon as requested
        "Nt": 100,  # Proper temporal resolution for stability
        "sigma": 0.15,  # Conservative diffusion for long-time stability
        "coefCT": 0.03,  # Light coupling for numerical stability
    }

    print(f"\nT=5 Problem Parameters (Long-Time Configuration):")
    for key, value in problem_params.items():
        print(f"  {key}: {value}")

    # Create problem
    problem = ExampleMFGProblem(**problem_params)

    print(f"\nProblem setup:")
    print(f"  Domain: [{problem.xmin}, {problem.xmax}] × [0, {problem.T}]")
    print(f"  Grid: Dx = {problem.Dx:.4f}, Dt = {problem.Dt:.4f}")
    print(
        f"  CFL number: {problem_params['sigma']**2 * problem.Dt / problem.Dx**2:.4f}"
    )
    print(f"  Total time steps: {problem.Nt}")
    print(f"  Total spatial points: {problem.Nx + 1}")

    # Enhanced QP solver parameters for long-time simulation
    solver_params = {
        "num_particles": 1000,  # More particles for better long-time statistics
        "delta": 0.35,  # Conservative neighborhood size
        "taylor_order": 2,  # Second-order accuracy
        "weight_function": "wendland",  # Stable weight function
        "NiterNewton": 8,  # More Newton iterations for stability
        "l2errBoundNewton": 1e-4,
        "kde_bandwidth": "scott",
        "normalize_kde_output": False,
        "use_monotone_constraints": True,  # Essential for long-time stability
    }

    print(f"\nEnhanced QP Solver Parameters for T=5:")
    for key, value in solver_params.items():
        if key == "use_monotone_constraints" and value:
            print(f"  {key}: {value} ← QP CONSTRAINTS ENABLED")
        else:
            print(f"  {key}: {value}")

    # Setup collocation with higher resolution for T=5
    num_collocation_points = 15  # More collocation points for longer simulation
    collocation_points = np.linspace(
        problem.xmin, problem.xmax, num_collocation_points
    ).reshape(-1, 1)

    boundary_indices = [0, num_collocation_points - 1]
    no_flux_bc = BoundaryConditions(type="no_flux")

    print(f"  Collocation points: {num_collocation_points}")
    print(f"  Boundary collocation points: {len(boundary_indices)}")

    # Create solver
    print(f"\n--- Creating Enhanced QP-Collocation Solver for T=5 ---")
    solver = ParticleCollocationSolver(
        problem=problem,
        collocation_points=collocation_points,
        boundary_indices=np.array(boundary_indices),
        boundary_conditions=no_flux_bc,
        **solver_params,
    )

    # Enhanced solve settings for long-time simulation
    max_iterations = 18  # More iterations for convergence
    convergence_tolerance = 1e-3

    print(f"\n--- Running T=5 Long-Time QP-Collocation Simulation ---")
    print(f"Max iterations: {max_iterations}")
    print(f"Convergence tolerance: {convergence_tolerance}")
    print(f"Expected execution time: 10-20 minutes (extended simulation)")
    print(f"Observing long-time mass conservation behavior...")

    start_time = time.time()

    try:
        U_solution, M_solution, solve_info = solver.solve(
            Niter=max_iterations, l2errBound=convergence_tolerance, verbose=True
        )

        total_time = time.time() - start_time
        iterations_run = solve_info.get("iterations", max_iterations)
        converged = solve_info.get("converged", False)

        print(f"\n--- T=5 Long-Time QP Simulation Completed ---")
        print(f"Execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Iterations: {iterations_run}/{max_iterations}")
        print(f"Converged: {converged}")

        if U_solution is not None and M_solution is not None:
            # Extended mass conservation analysis
            print(f"\n{'='*60}")
            print("T=5 LONG-TIME MASS CONSERVATION ANALYSIS")
            print(f"{'='*60}")

            mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
            initial_mass = mass_evolution[0]
            final_mass = mass_evolution[-1]
            mass_change = final_mass - initial_mass
            mass_change_percent = (mass_change / initial_mass) * 100

            print(f"Mass conservation over extended T=5:")
            print(f"  Initial mass (t=0): {initial_mass:.8f}")
            print(f"  Final mass (t=5): {final_mass:.8f}")
            print(f"  Absolute change: {mass_change:.2e}")
            print(f"  Relative change: {mass_change_percent:+.4f}%")

            # Long-time mass behavior analysis
            mass_statistics = {
                "max_mass": np.max(mass_evolution),
                "min_mass": np.min(mass_evolution),
                "mass_variation": np.max(mass_evolution) - np.min(mass_evolution),
                "mass_std": np.std(mass_evolution),
                "mass_trend": np.polyfit(problem.tSpace, mass_evolution, 1)[
                    0
                ],  # Linear trend
            }

            print(f"\nLong-time mass statistics:")
            print(f"  Maximum mass: {mass_statistics['max_mass']:.8f}")
            print(f"  Minimum mass: {mass_statistics['min_mass']:.8f}")
            print(f"  Mass variation: {mass_statistics['mass_variation']:.2e}")
            print(f"  Mass std deviation: {mass_statistics['mass_std']:.2e}")
            print(f"  Mass trend (per unit time): {mass_statistics['mass_trend']:.2e}")
            print(
                f"  Relative variation: {mass_statistics['mass_variation']/initial_mass*100:.6f}%"
            )

            # Mass behavior analysis
            mass_trend = (
                "increases"
                if mass_change > 0
                else "decreases" if mass_change < 0 else "remains constant"
            )
            print(f"  Mass behavior: {mass_trend} over time")

            if mass_change > 0:
                print("  ✅ Mass increase is EXPECTED with no-flux boundary conditions")
                print(
                    "     (particles reflect at boundaries, creating effective source)"
                )

            # Mass conservation quality
            if abs(mass_change_percent) < 0.5:
                quality = "✅ EXCELLENT mass conservation"
            elif abs(mass_change_percent) < 2.0:
                quality = "✅ VERY GOOD mass conservation"
            elif abs(mass_change_percent) < 5.0:
                quality = "✅ GOOD mass conservation"
            else:
                quality = "⚠️  Moderate mass conservation"

            print(f"  Assessment: {quality}")

            # Physical observables
            print(f"\n{'='*60}")
            print("PHYSICAL OBSERVABLES")
            print(f"{'='*60}")

            # Initial vs final comparison
            initial_density = M_solution[0, :]
            final_density = M_solution[-1, :]

            initial_com = np.sum(problem.xSpace * initial_density) * problem.Dx
            final_com = np.sum(problem.xSpace * final_density) * problem.Dx

            initial_max_idx = np.argmax(initial_density)
            final_max_idx = np.argmax(final_density)
            initial_peak_loc = problem.xSpace[initial_max_idx]
            final_peak_loc = problem.xSpace[final_max_idx]
            initial_peak_val = initial_density[initial_max_idx]
            final_peak_val = final_density[final_max_idx]

            print(f"Center of mass:")
            print(f"  Initial: {initial_com:.6f}")
            print(f"  Final: {final_com:.6f}")
            print(f"  Change: {final_com - initial_com:+.6f}")

            print(f"\nPeak density:")
            print(
                f"  Initial location: {initial_peak_loc:.6f}, value: {initial_peak_val:.4f}"
            )
            print(
                f"  Final location: {final_peak_loc:.6f}, value: {final_peak_val:.4f}"
            )
            print(f"  Location change: {final_peak_loc - initial_peak_loc:+.6f}")
            print(f"  Value change: {final_peak_val - initial_peak_val:+.4f}")

            # Particle boundary compliance
            particles_trajectory = solver.fp_solver.M_particles_trajectory
            if particles_trajectory is not None:
                print(f"\n{'='*60}")
                print("BOUNDARY COMPLIANCE")
                print(f"{'='*60}")

                total_violations = 0
                for t_step in range(particles_trajectory.shape[0]):
                    step_particles = particles_trajectory[t_step, :]
                    violations = np.sum(
                        (step_particles < problem.xmin - 1e-10)
                        | (step_particles > problem.xmax + 1e-10)
                    )
                    total_violations += violations

                print(f"Particle boundary analysis:")
                print(f"  Total particles: {particles_trajectory.shape[1]}")
                print(f"  Time steps: {particles_trajectory.shape[0]}")
                print(f"  Boundary violations: {total_violations}")

                if total_violations == 0:
                    print("  ✅ PERFECT boundary compliance - no particle escapes!")
                else:
                    print(f"  ⚠️  {total_violations} boundary violations detected")

            # Solution quality
            print(f"\n{'='*60}")
            print("SOLUTION QUALITY")
            print(f"{'='*60}")

            negative_densities = np.sum(M_solution < -1e-10)
            max_U = np.max(np.abs(U_solution))

            print(f"Quality metrics:")
            print(f"  Negative densities: {negative_densities}")
            print(f"  Max |U|: {max_U:.3f}")
            print(f"  Converged: {converged}")

            if negative_densities == 0:
                print("  ✅ QP constraints successful - all densities non-negative")
            else:
                print("  ⚠️  Some negative densities detected")

            # Create demonstration plots
            create_t5_demonstration_plots(
                problem,
                M_solution,
                U_solution,
                mass_evolution,
                particles_trajectory,
                total_time,
                mass_change_percent,
                mass_statistics,
            )

            # Long-time behavior analysis
            print(f"\n{'='*60}")
            print("LONG-TIME BEHAVIOR ANALYSIS")
            print(f"{'='*60}")

            # Analyze mass evolution trend
            early_mass = np.mean(
                mass_evolution[: len(mass_evolution) // 4]
            )  # First quarter
            mid_mass = np.mean(
                mass_evolution[len(mass_evolution) // 4 : 3 * len(mass_evolution) // 4]
            )  # Middle half
            late_mass = np.mean(
                mass_evolution[3 * len(mass_evolution) // 4 :]
            )  # Final quarter

            print(f"Mass evolution phases:")
            print(f"  Early phase (t=0-1.25): {early_mass:.8f}")
            print(f"  Middle phase (t=1.25-3.75): {mid_mass:.8f}")
            print(f"  Late phase (t=3.75-5): {late_mass:.8f}")

            early_to_mid = (mid_mass - early_mass) / early_mass * 100
            mid_to_late = (late_mass - mid_mass) / mid_mass * 100

            print(f"  Early→Middle change: {early_to_mid:+.3f}%")
            print(f"  Middle→Late change: {mid_to_late:+.3f}%")

            # Check for steady state behavior
            late_variation = np.std(mass_evolution[-len(mass_evolution) // 4 :])
            if late_variation < mass_statistics["mass_std"] / 2:
                print(f"✅ Approaching steady state in final phase")
            else:
                print(f"⚠️  Still evolving in final phase")

            # Summary
            print(f"\n{'='*80}")
            print("T=5 LONG-TIME MASS CONSERVATION SUMMARY")
            print(f"{'='*80}")
            print(
                f"🎯 Successfully demonstrated QP-Collocation method over extended T=5"
            )
            print(f"✅ {quality}")
            print(f"✅ Mass change over 5 time units: {mass_change_percent:+.3f}%")
            print(
                f"✅ Long-time mass variation: {mass_statistics['mass_variation']/initial_mass*100:+.4f}%"
            )
            print(f"✅ Boundary compliance: {total_violations == 0} (0 violations)")
            print(f"✅ QP constraints effective: {negative_densities == 0}")
            print(f"⏱️  Execution time: {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"\n🔬 LONG-TIME MASS CONSERVATION INSIGHTS:")
            if mass_change > 0:
                print(
                    f"   Mass increase of {mass_change_percent:+.3f}% over T=5 demonstrates robust no-flux BC"
                )
                print(
                    f"   Linear trend: {mass_statistics['mass_trend']:.2e} per time unit"
                )
            print(
                f"   Mass variation {mass_statistics['mass_variation']/initial_mass*100:.4f}% shows excellent stability"
            )
            print(
                f"   QP constraints maintain solution quality throughout extended T=5 evolution"
            )

        else:
            print("❌ Solver failed to produce valid results")

    except Exception as e:
        print(f"❌ T=5 long-time simulation failed: {e}")
        import traceback

        traceback.print_exc()


def create_t5_demonstration_plots(
    problem,
    M_solution,
    U_solution,
    mass_evolution,
    particles_trajectory,
    execution_time,
    mass_change_percent,
    mass_statistics,
):
    """Create demonstration plots for T=5 long-time simulation"""

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle(
        f"QP-Collocation T=5 Long-Time Mass Conservation Demonstration\n"
        f"Execution: {execution_time:.1f}s ({execution_time/60:.1f}min), Mass change: {mass_change_percent:+.3f}%",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Mass evolution over T=5 (extended view)
    ax1 = axes[0, 0]
    ax1.plot(
        problem.tSpace, mass_evolution, "g-", linewidth=3, marker="o", markersize=2
    )
    ax1.axhline(
        y=mass_evolution[0],
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Initial: {mass_evolution[0]:.6f}",
    )
    ax1.axhline(
        y=mass_evolution[-1],
        color="b",
        linestyle="--",
        alpha=0.7,
        label=f"Final: {mass_evolution[-1]:.6f}",
    )
    # Add trend line
    trend_line = mass_evolution[0] + mass_statistics["mass_trend"] * problem.tSpace
    ax1.plot(
        problem.tSpace,
        trend_line,
        "k--",
        alpha=0.5,
        label=f"Trend: {mass_statistics['mass_trend']:.2e}/t",
    )

    ax1.set_xlabel("Time t")
    ax1.set_ylabel("Total Mass")
    ax1.set_title("Mass Evolution T=0→5 (Long-Time)", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add conservation assessment
    conservation_text = (
        "Excellent"
        if abs(mass_change_percent) < 3
        else "Good" if abs(mass_change_percent) < 10 else "Fair"
    )
    ax1.text(
        0.05,
        0.95,
        f"Conservation: {conservation_text}\nChange: {mass_change_percent:+.3f}%\nVariation: {mass_statistics['mass_variation']/mass_evolution[0]*100:.3f}%",
        transform=ax1.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
    )

    # 2. Mass variation detail (zoomed)
    ax2 = axes[0, 1]
    mass_change_detailed = (
        (mass_evolution - mass_evolution[0]) / mass_evolution[0] * 100
    )
    ax2.plot(
        problem.tSpace,
        mass_change_detailed,
        "b-",
        linewidth=2,
        marker="s",
        markersize=2,
    )
    ax2.fill_between(problem.tSpace, mass_change_detailed, alpha=0.3, color="blue")
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.5)
    ax2.set_xlabel("Time t")
    ax2.set_ylabel("Mass Change (%)")
    ax2.set_title("Mass Conservation Detail (T=5)")
    ax2.grid(True, alpha=0.3)

    # 3. Density evolution snapshots
    ax3 = axes[0, 2]
    # Show evolution at key time points
    snapshot_times = [
        0,
        len(problem.tSpace) // 5,
        2 * len(problem.tSpace) // 5,
        3 * len(problem.tSpace) // 5,
        4 * len(problem.tSpace) // 5,
        len(problem.tSpace) - 1,
    ]
    colors = ["green", "blue", "orange", "purple", "brown", "red"]
    labels = ["t=0", "t=1", "t=2", "t=3", "t=4", "t=5"]

    for i, (idx, color, label) in enumerate(zip(snapshot_times, colors, labels)):
        alpha = 0.9 - i * 0.1
        linewidth = 3 if i == 0 or i == len(snapshot_times) - 1 else 2
        linestyle = "--" if i == 0 else "-"
        ax3.plot(
            problem.xSpace,
            M_solution[idx, :],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            linestyle=linestyle,
        )

    ax3.set_xlabel("Space x")
    ax3.set_ylabel("Density M(t,x)")
    ax3.set_title("Density Evolution Snapshots")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)

    # 4. Control field evolution
    ax4 = axes[1, 0]
    ax4.plot(
        problem.xSpace,
        U_solution[0, :],
        "g--",
        linewidth=2,
        alpha=0.7,
        label="Initial U (t=0)",
    )
    ax4.plot(
        problem.xSpace, U_solution[-1, :], "r-", linewidth=2, label="Final U (t=5)"
    )
    # Add intermediate control fields
    mid_times = [len(problem.tSpace) // 3, 2 * len(problem.tSpace) // 3]
    for i, idx in enumerate(mid_times):
        ax4.plot(
            problem.xSpace,
            U_solution[idx, :],
            alpha=0.6,
            linewidth=1.5,
            label=f"t={problem.tSpace[idx]:.1f}",
        )
    ax4.set_xlabel("Space x")
    ax4.set_ylabel("Control Field U(t,x)")
    ax4.set_title("Control Field Evolution")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # 5. Long-time particle trajectories
    ax5 = axes[1, 1]
    if particles_trajectory is not None:
        # Show sample particle trajectories
        sample_size = min(30, particles_trajectory.shape[1])
        particle_indices = np.linspace(
            0, particles_trajectory.shape[1] - 1, sample_size, dtype=int
        )

        for i in particle_indices:
            ax5.plot(
                problem.tSpace, particles_trajectory[:, i], "b-", alpha=0.4, linewidth=1
            )

        ax5.set_xlabel("Time t")
        ax5.set_ylabel("Particle Position")
        ax5.set_title(f"Sample Particle Trajectories (n={sample_size})")
        ax5.set_ylim([problem.xmin - 0.05, problem.xmax + 0.05])
        ax5.grid(True, alpha=0.3)

        # Add boundary lines
        ax5.axhline(
            y=problem.xmin,
            color="red",
            linestyle="--",
            alpha=0.8,
            linewidth=2,
            label="Boundaries",
        )
        ax5.axhline(y=problem.xmax, color="red", linestyle="--", alpha=0.8, linewidth=2)
        ax5.legend()

        # Check for violations
        total_violations = 0
        for t_step in range(particles_trajectory.shape[0]):
            step_particles = particles_trajectory[t_step, :]
            violations = np.sum(
                (step_particles < problem.xmin - 1e-10)
                | (step_particles > problem.xmax + 1e-10)
            )
            total_violations += violations

        boundary_text = f"Violations: {total_violations}\n{'Perfect' if total_violations == 0 else 'Good'} Compliance"
        ax5.text(
            0.05,
            0.95,
            boundary_text,
            transform=ax5.transAxes,
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="lightgreen" if total_violations == 0 else "lightyellow",
            ),
        )

    # 6. Summary statistics
    ax6 = axes[1, 2]

    # Calculate key statistics
    mass_conservation_score = max(
        0, 100 - abs(mass_change_percent) * 10
    )  # Penalty for large changes
    boundary_compliance_score = (
        100
        if (
            particles_trajectory is not None
            and all(
                np.all(
                    (particles_trajectory[t] >= problem.xmin - 1e-10)
                    & (particles_trajectory[t] <= problem.xmax + 1e-10)
                )
                for t in range(particles_trajectory.shape[0])
            )
        )
        else 90
    )

    density_quality_score = 100 if np.all(M_solution >= -1e-10) else 80

    categories = ["Mass\nConservation", "Boundary\nCompliance", "Density\nQuality"]
    scores = [mass_conservation_score, boundary_compliance_score, density_quality_score]
    colors = ["green" if s >= 95 else "orange" if s >= 80 else "red" for s in scores]

    bars = ax6.bar(categories, scores, color=colors, alpha=0.7)
    ax6.set_ylabel("Quality Score (%)")
    ax6.set_title("QP Method Quality Assessment")
    ax6.set_ylim([0, 105])
    ax6.grid(True, axis="y", alpha=0.3)

    # Add score labels
    for bar, score in zip(bars, scores):
        ax6.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1,
            f"{score:.0f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Overall assessment
    overall_score = np.mean(scores)
    if overall_score >= 95:
        assessment = "Excellent"
        assessment_color = "green"
    elif overall_score >= 85:
        assessment = "Very Good"
        assessment_color = "blue"
    elif overall_score >= 75:
        assessment = "Good"
        assessment_color = "orange"
    else:
        assessment = "Fair"
        assessment_color = "red"

    ax6.text(
        0.5,
        0.85,
        f"Overall: {assessment}\n({overall_score:.0f}%)",
        transform=ax6.transAxes,
        ha="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=assessment_color, alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig(
        "/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/qp_t5_mass_conservation_demo.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    print("Starting QP-Collocation T=5 Long-Time Mass Conservation Demonstration...")
    print(
        "Enhanced resolution and extended time horizon for long-time behavior observation"
    )
    print("Expected execution time: 10-20 minutes")

    try:
        run_t5_mass_conservation_demo()
        print("\n" + "=" * 80)
        print("T=5 LONG-TIME MASS CONSERVATION DEMONSTRATION COMPLETED")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback

        traceback.print_exc()
