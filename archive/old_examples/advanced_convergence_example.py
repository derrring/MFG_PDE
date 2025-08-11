#!/usr/bin/env python3
"""
Advanced Convergence Criteria Example

This example demonstrates the enhanced convergence monitoring capabilities
for particle-based MFG methods, showcasing:

1. Wasserstein distance tracking for probability distributions
2. Oscillation stabilization detection for value functions
3. Multi-criteria convergence assessment
4. Comparison with traditional L2 error-based convergence

The example shows how robust convergence criteria handle the statistical
noise and oscillatory behavior inherent in particle-based coupled systems.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import BoundaryConditions, ExampleMFGProblem
from mfg_pde.alg.enhanced_particle_collocation_solver import EnhancedParticleCollocationSolver
from mfg_pde.utils.convergence import AdvancedConvergenceMonitor


def demonstrate_convergence_criteria():
    """
    Main demonstration of advanced convergence criteria.
    """
    print("=" * 80)
    print("DEMONSTRATION: ADVANCED CONVERGENCE CRITERIA FOR PARTICLE-BASED MFG")
    print("=" * 80)
    print()

    # Problem setup
    print("Setting up MFG problem...")
    problem = ExampleMFGProblem(
        xmin=0.0, xmax=1.0, Nx=25, T=1.0, Nt=20, sigma=0.15, coefCT=0.03  # Higher volatility to induce more noise
    )

    boundary_conditions = BoundaryConditions(type="no_flux")

    # Collocation points
    num_collocation_points = 8
    collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)

    print(f"Problem: {problem.Nx}×{problem.Nt} grid, σ={problem.sigma}")
    print(f"Collocation points: {num_collocation_points}")
    print()

    # Create enhanced solver with custom convergence settings
    print("Initializing enhanced solver with advanced convergence monitoring...")

    # Customize convergence tolerances for demonstration
    convergence_settings = {
        'wasserstein_tol': 2e-4,  # Slightly relaxed for noisy particles
        'u_magnitude_tol': 1e-3,  # Oscillation magnitude tolerance
        'u_stability_tol': 5e-4,  # Oscillation stability tolerance
        'history_length': 8,  # Track recent history
    }

    solver = EnhancedParticleCollocationSolver(
        problem=problem,
        collocation_points=collocation_points,
        num_particles=3000,  # Moderate particle count for demonstration
        delta=0.4,
        boundary_conditions=boundary_conditions,
        use_monotone_constraints=True,
        **convergence_settings,
    )

    print("Convergence settings:")
    for key, value in convergence_settings.items():
        print(f"  {key}: {value}")
    print()

    # Solve with enhanced monitoring
    print("Solving with advanced convergence criteria...")
    print("─" * 80)

    start_time = time.time()

    U, M, info = solver.solve(
        Niter=15,
        l2errBound=1e-3,  # Legacy tolerance for comparison
        verbose=True,
        plot_convergence=False,  # We'll create custom plots
    )

    solve_time = time.time() - start_time

    print("─" * 80)
    print(f"Solve completed in {solve_time:.2f} seconds")
    print()

    # Detailed convergence analysis
    print("=" * 80)
    print("DETAILED CONVERGENCE ANALYSIS")
    print("=" * 80)

    diagnostics = solver.get_convergence_diagnostics()
    summary = diagnostics['summary']

    print(f"Convergence Status: {'✅ SUCCESS' if summary['converged'] else '❌ NOT CONVERGED'}")
    print(f"Total Iterations: {summary['total_iterations']}")
    if summary['converged']:
        print(f"Convergence Iteration: {summary['convergence_iteration']}")

    print()
    print("Final Metrics:")
    print(f"  Final L2 Error: {summary['final_u_error']:.2e}")
    if summary['final_wasserstein'] is not None:
        print(f"  Final Wasserstein Distance: {summary['final_wasserstein']:.2e}")

    print()
    print("L2 Error Statistics:")
    trend = summary['u_error_trend']
    print(f"  Minimum: {trend['min']:.2e}")
    print(f"  Maximum: {trend['max']:.2e}")
    print(f"  Final Mean (last 5): {trend['final_mean']:.2e}")
    print(f"  Final Std (last 5): {trend['final_std']:.2e}")

    # Create comprehensive convergence visualization
    create_convergence_analysis_plots(solver, diagnostics, save_prefix="advanced_convergence")

    # Compare with traditional convergence
    demonstrate_traditional_vs_advanced_convergence()

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("Key Insights:")
    print("1. Advanced criteria handle statistical noise in particle methods")
    print("2. Oscillation detection prevents premature convergence declaration")
    print("3. Wasserstein distance provides robust distribution comparison")
    print("4. Multi-criteria approach ensures system-wide stability")


def create_convergence_analysis_plots(solver, diagnostics, save_prefix="convergence"):
    """
    Create detailed convergence analysis plots.
    """
    print("Creating detailed convergence analysis plots...")

    # Extract data
    u_errors = diagnostics['u_error_history']
    wasserstein_history = diagnostics['wasserstein_history']
    detailed_history = solver.detailed_convergence_history

    iterations = list(range(1, len(u_errors) + 1))

    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Advanced Convergence Criteria Analysis', fontsize=16)

    # Plot 1: L2 Error Evolution
    ax1 = axes[0, 0]
    ax1.semilogy(iterations, u_errors, 'b-', linewidth=2, label='L2 Error')
    ax1.axhline(
        y=solver.convergence_monitor.u_magnitude_tol, color='r', linestyle='--', alpha=0.7, label='Magnitude Tolerance'
    )
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('L2 Error')
    ax1.set_title('Value Function L2 Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Wasserstein Distance
    ax2 = axes[0, 1]
    if wasserstein_history:
        w_iterations = list(range(2, len(wasserstein_history) + 2))  # Start from iteration 2
        ax2.semilogy(w_iterations, wasserstein_history, 'g-', linewidth=2, label='Wasserstein Distance')
        ax2.axhline(
            y=solver.convergence_monitor.wasserstein_tol,
            color='r',
            linestyle='--',
            alpha=0.7,
            label='Wasserstein Tolerance',
        )
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Wasserstein Distance')
    ax2.set_title('Distribution Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Oscillation Analysis
    ax3 = axes[1, 0]
    oscillation_means = []
    oscillation_stds = []

    for d in detailed_history:
        if 'u_oscillation' in d and 'mean_error' in d['u_oscillation']:
            oscillation_means.append(d['u_oscillation']['mean_error'])
            oscillation_stds.append(d['u_oscillation']['std_error'])
        else:
            oscillation_means.append(np.nan)
            oscillation_stds.append(np.nan)

    valid_means = [m for m in oscillation_means if not np.isnan(m)]
    valid_stds = [s for s in oscillation_stds if not np.isnan(s)]
    valid_iter = list(range(len(valid_means)))

    if valid_means:
        ax3.semilogy(valid_iter, valid_means, 'orange', linewidth=2, label='Mean Error (History)')
        ax3.semilogy(valid_iter, valid_stds, 'purple', linewidth=2, label='Std Error (History)')
        ax3.axhline(
            y=solver.convergence_monitor.u_magnitude_tol, color='r', linestyle='--', alpha=0.7, label='Magnitude Tol'
        )
        ax3.axhline(
            y=solver.convergence_monitor.u_stability_tol,
            color='orange',
            linestyle='--',
            alpha=0.7,
            label='Stability Tol',
        )

    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Error Statistics')
    ax3.set_title('Oscillation Stabilization Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Convergence Criteria Status
    ax4 = axes[1, 1]

    # Extract convergence criteria over time
    wasserstein_ok = []
    u_stabilized_ok = []
    overall_converged = []

    for d in detailed_history:
        criteria = d['convergence_criteria']
        wasserstein_ok.append(criteria.get('wasserstein', False))
        u_stabilized_ok.append(criteria.get('u_stabilized', False))
        overall_converged.append(d['converged'])

    criteria_iterations = list(range(1, len(wasserstein_ok) + 1))

    # Plot as step functions
    ax4.step(
        criteria_iterations,
        [1 if x else 0 for x in wasserstein_ok],
        'g-',
        linewidth=2,
        where='post',
        label='Wasserstein OK',
    )
    ax4.step(
        criteria_iterations,
        [1 if x else 0 for x in u_stabilized_ok],
        'b-',
        linewidth=2,
        where='post',
        label='U Stabilized OK',
    )
    ax4.step(
        criteria_iterations,
        [1 if x else 0 for x in overall_converged],
        'r-',
        linewidth=3,
        where='post',
        label='Overall Converged',
    )

    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Criterion Satisfied')
    ax4.set_title('Convergence Criteria Status')
    ax4.set_ylim(-0.1, 1.1)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['No', 'Yes'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = f"{save_prefix}_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Convergence analysis plot saved to: {filename}")

    plt.show()


def demonstrate_traditional_vs_advanced_convergence():
    """
    Compare traditional L2 error convergence with advanced criteria.
    """
    print("\n" + "─" * 80)
    print("COMPARISON: TRADITIONAL vs ADVANCED CONVERGENCE")
    print("─" * 80)

    print("Creating synthetic example to show convergence differences...")

    # Create synthetic convergence data that illustrates the problem
    # with traditional L2 error-based convergence

    iterations = np.arange(1, 21)

    # Simulate L2 error that oscillates due to particle noise
    base_decay = 1e-2 * np.exp(-0.3 * iterations)  # Exponential decay
    noise = 5e-4 * np.sin(2 * iterations) * np.exp(-0.1 * iterations)  # Oscillatory noise
    l2_errors = base_decay + noise + 1e-4  # Add small baseline

    # Simulate Wasserstein distance with smoother convergence
    wasserstein_dist = 2e-3 * np.exp(-0.4 * iterations) + 1e-5

    # Traditional convergence (would fail due to oscillations)
    l2_threshold = 1e-3
    traditional_converged = l2_errors < l2_threshold

    # Advanced convergence (detects stabilization)
    # Simulate the advanced criteria
    magnitude_ok = np.zeros_like(iterations, dtype=bool)
    stability_ok = np.zeros_like(iterations, dtype=bool)
    wasserstein_ok = wasserstein_dist < 5e-4

    # Advanced criteria kick in after some history is built
    for i in range(7, len(iterations)):  # After 7 iterations
        recent_errors = l2_errors[max(0, i - 5) : i + 1]
        magnitude_ok[i] = np.mean(recent_errors) < 2e-3
        stability_ok[i] = np.std(recent_errors) < 5e-4

    advanced_converged = magnitude_ok & stability_ok & wasserstein_ok

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Error evolution
    ax1.semilogy(iterations, l2_errors, 'b-', linewidth=2, label='L2 Error (Oscillatory)')
    ax1.semilogy(iterations, wasserstein_dist, 'g-', linewidth=2, label='Wasserstein Distance')
    ax1.axhline(y=l2_threshold, color='r', linestyle='--', alpha=0.7, label='L2 Threshold')
    ax1.axhline(y=5e-4, color='orange', linestyle='--', alpha=0.7, label='Wasserstein Threshold')

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Error/Distance')
    ax1.set_title('Error Evolution: Traditional vs Advanced Metrics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Convergence decisions
    ax2.step(
        iterations,
        traditional_converged.astype(int),
        'r-',
        linewidth=3,
        where='post',
        label='Traditional (L2 only)',
        alpha=0.7,
    )
    ax2.step(
        iterations, advanced_converged.astype(int), 'g-', linewidth=3, where='post', label='Advanced (Multi-criteria)'
    )

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Converged')
    ax2.set_title('Convergence Decisions Comparison')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['No', 'Yes'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = "traditional_vs_advanced_convergence.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {filename}")

    plt.show()

    # Summary
    traditional_first_converged = np.where(traditional_converged)[0]
    advanced_first_converged = np.where(advanced_converged)[0]

    print("\nComparison Results:")
    if len(traditional_first_converged) > 0:
        print(f"Traditional: First 'converged' at iteration {traditional_first_converged[0] + 1}")
        print("  Problem: May declare convergence prematurely due to noise")
    else:
        print("Traditional: Never converged (good - avoids false positive)")

    if len(advanced_first_converged) > 0:
        print(f"Advanced: Converged at iteration {advanced_first_converged[0] + 1}")
        print("  Benefit: Waits for true stabilization, avoids noise-induced decisions")
    else:
        print("Advanced: Did not converge in this example")

    print("\nKey Advantages of Advanced Criteria:")
    print("• Handles statistical noise from particle methods")
    print("• Detects oscillation stabilization rather than momentary drops")
    print("• Uses distribution-aware metrics (Wasserstein distance)")
    print("• Multi-criteria validation prevents false convergence")


if __name__ == "__main__":
    demonstrate_convergence_criteria()
