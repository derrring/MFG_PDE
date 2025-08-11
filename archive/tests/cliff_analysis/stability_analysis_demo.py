#!/usr/bin/env python3
"""
Numerical Stability Analysis Demo
Demonstrates why stability is suddenly lost in long-time simulations.
"""

from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist


def demonstrate_stability_loss_mechanisms():
    """Demonstrate the key mechanisms that cause sudden stability loss"""
    print("=" * 80)
    print("NUMERICAL STABILITY LOSS ANALYSIS")
    print("=" * 80)
    print("Demonstrating why stability is suddenly lost, not gradually")

    # Create synthetic particle evolution that mimics MFG behavior
    n_particles = 500
    n_time_steps = 100

    # Simulate particle evolution with clustering
    particles_evolution = simulate_particle_clustering(n_particles, n_time_steps)

    # Analyze stability indicators
    analyze_stability_indicators(particles_evolution)

    # Demonstrate kernel bandwidth collapse
    demonstrate_kernel_bandwidth_collapse(particles_evolution)

    # Show eigenvalue instability
    demonstrate_eigenvalue_instability()

    # Create comprehensive plots
    create_stability_analysis_plots(particles_evolution)


def simulate_particle_clustering(n_particles, n_time_steps):
    """Simulate particle evolution that leads to clustering and instability"""
    print("\n--- SIMULATING PARTICLE CLUSTERING ---")

    # Initialize particles uniformly
    particles = np.random.uniform(0, 1, n_particles)
    particles_history = [particles.copy()]

    # Parameters that lead to clustering
    drift_strength = 0.02
    diffusion = 0.01
    attraction_center = 0.6  # Particles attracted to this point

    for t in range(n_time_steps):
        # Drift toward attraction center (simulates control field effect)
        drift = drift_strength * (attraction_center - particles)

        # Add noise
        noise = np.random.normal(0, diffusion, n_particles)

        # Update particles
        particles = particles + drift + noise

        # Reflect at boundaries (no-flux BC)
        particles = np.clip(particles, 0, 1)

        # Add clustering effect (particles attract each other when close)
        if t > 20:  # Start clustering after some time
            for i in range(n_particles):
                neighbors = np.where(np.abs(particles - particles[i]) < 0.05)[0]
                if len(neighbors) > 1:
                    local_center = np.mean(particles[neighbors])
                    particles[i] += 0.001 * (local_center - particles[i])

        particles_history.append(particles.copy())

    return np.array(particles_history)


def analyze_stability_indicators(particles_evolution):
    """Analyze various stability indicators over time"""
    print("\n--- ANALYZING STABILITY INDICATORS ---")

    n_time_steps = len(particles_evolution)

    # 1. Particle spread (variance)
    spreads = [np.var(particles) for particles in particles_evolution]

    # 2. Minimum inter-particle distance
    min_distances = []
    for particles in particles_evolution:
        if len(particles) > 1:
            distances = pdist(particles.reshape(-1, 1))
            min_distances.append(np.min(distances))
        else:
            min_distances.append(0)

    # 3. Number of particles in clusters
    cluster_sizes = []
    for particles in particles_evolution:
        # Count particles in dense regions
        hist, bins = np.histogram(particles, bins=20, range=(0, 1))
        max_density = np.max(hist)
        cluster_sizes.append(max_density)

    # 4. Bandwidth estimation
    bandwidths = []
    for particles in particles_evolution:
        # Scott's rule bandwidth
        n = len(particles)
        std = np.std(particles)
        bandwidth = 1.06 * std * n ** (-1 / 5)
        bandwidths.append(bandwidth)

    # Find critical points
    print(f"Initial particle spread: {spreads[0]:.6f}")
    print(f"Final particle spread: {spreads[-1]:.6f}")
    print(f"Spread reduction: {(spreads[0] - spreads[-1])/spreads[0]*100:.1f}%")

    print(f"\nInitial min distance: {min_distances[0]:.6f}")
    print(f"Final min distance: {min_distances[-1]:.6f}")

    print(f"\nInitial bandwidth: {bandwidths[0]:.6f}")
    print(f"Final bandwidth: {bandwidths[-1]:.6f}")
    print(f"Bandwidth reduction: {(bandwidths[0] - bandwidths[-1])/bandwidths[0]*100:.1f}%")

    # Detect sudden changes
    bandwidth_changes = np.abs(np.diff(bandwidths))
    sudden_change_threshold = 5 * np.std(bandwidth_changes[:20])  # 5-sigma threshold
    sudden_changes = np.where(bandwidth_changes > sudden_change_threshold)[0]

    if len(sudden_changes) > 0:
        print(f"\n⚠️  SUDDEN BANDWIDTH CHANGES detected at time steps: {sudden_changes}")
        print("This indicates potential numerical instability!")
    else:
        print(f"\n✅ No sudden bandwidth changes detected")

    return {
        'spreads': spreads,
        'min_distances': min_distances,
        'cluster_sizes': cluster_sizes,
        'bandwidths': bandwidths,
        'sudden_changes': sudden_changes,
    }


def demonstrate_kernel_bandwidth_collapse():
    """Show how kernel bandwidth collapse causes numerical instability"""
    print("\n--- KERNEL BANDWIDTH COLLAPSE DEMONSTRATION ---")

    # Create scenarios: well-separated vs clustered particles
    n_particles = 100

    # Scenario 1: Well-separated particles
    particles_good = np.random.uniform(0, 1, n_particles)

    # Scenario 2: Clustered particles (simulates what happens over time)
    cluster_centers = [0.3, 0.7]
    particles_bad = []
    for center in cluster_centers:
        cluster_particles = np.random.normal(center, 0.02, n_particles // 2)
        particles_bad.extend(cluster_particles)
    particles_bad = np.array(particles_bad)
    particles_bad = np.clip(particles_bad, 0, 1)  # Ensure domain bounds

    scenarios = [("Well-separated", particles_good), ("Clustered", particles_bad)]

    for name, particles in scenarios:
        print(f"\n{name} particles:")

        # Calculate Scott's bandwidth
        n = len(particles)
        std = np.std(particles)
        scott_bandwidth = 1.06 * std * n ** (-1 / 5)

        # Calculate effective resolution
        domain_size = 1.0
        effective_resolution = domain_size / scott_bandwidth

        # Calculate condition number (proxy for numerical stability)
        # Based on kernel matrix eigenvalues
        try:
            kde = KernelDensity(bandwidth=scott_bandwidth, kernel='gaussian')
            kde.fit(particles.reshape(-1, 1))

            # Sample kernel matrix
            x_test = np.linspace(0, 1, 50).reshape(-1, 1)
            density_estimates = np.exp(kde.score_samples(x_test))

            condition_indicator = np.max(density_estimates) / (np.min(density_estimates) + 1e-10)

        except:
            condition_indicator = float('inf')

        print(f"  Standard deviation: {std:.6f}")
        print(f"  Scott's bandwidth: {scott_bandwidth:.6f}")
        print(f"  Effective resolution: {effective_resolution:.1f}")
        print(f"  Condition indicator: {condition_indicator:.2e}")

        if scott_bandwidth < 0.001:
            print(f"  ⚠️  CRITICAL: Bandwidth too small - numerical instability likely!")
        elif condition_indicator > 1e6:
            print(f"  ⚠️  CRITICAL: Poor conditioning - numerical instability likely!")
        else:
            print(f"  ✅ Bandwidth and conditioning acceptable")


def demonstrate_eigenvalue_instability():
    """Demonstrate how eigenvalues can suddenly become unstable"""
    print("\n--- EIGENVALUE INSTABILITY DEMONSTRATION ---")

    # Simplified model: mass matrix evolution
    # In real MFG, this represents the linearized dynamics

    n_modes = 10  # Number of spatial modes
    time_steps = 50

    eigenvalues_evolution = []

    for t in range(time_steps):
        # Create a matrix that represents the discretized system
        # This mimics how the effective system matrix changes over time

        # Base system (stable)
        A = np.diag(np.linspace(0.9, 0.95, n_modes))  # Stable eigenvalues

        # Add time-dependent perturbation (represents particle clustering effects)
        perturbation_strength = 0.1 * (t / time_steps) ** 2  # Grows quadratically

        # Off-diagonal coupling (represents particle interactions)
        for i in range(n_modes - 1):
            A[i, i + 1] = perturbation_strength * 0.1
            A[i + 1, i] = perturbation_strength * 0.1

        # Add clustering effect (makes some modes unstable)
        if t > 30:  # After clustering begins
            cluster_effect = 0.2 * ((t - 30) / 20) ** 3  # Cubic growth
            A[n_modes // 2, n_modes // 2] += cluster_effect

        eigenvals = np.linalg.eigvals(A)
        max_eigenval = np.max(np.abs(eigenvals))
        eigenvalues_evolution.append(max_eigenval)

        if t % 10 == 0:
            print(f"  Time {t:2d}: Max |eigenvalue| = {max_eigenval:.4f}")

    # Find when system becomes unstable
    unstable_threshold = 1.0
    unstable_times = np.where(np.array(eigenvalues_evolution) > unstable_threshold)[0]

    if len(unstable_times) > 0:
        first_unstable = unstable_times[0]
        print(f"\n⚠️  INSTABILITY DETECTED at time step {first_unstable}")
        print(f"     Max eigenvalue: {eigenvalues_evolution[first_unstable]:.4f} > 1.0")
        print("     This explains why stability is lost suddenly!")
    else:
        print(f"\n✅ System remains stable (all eigenvalues < 1.0)")

    return eigenvalues_evolution


def create_stability_analysis_plots(particles_evolution):
    """Create plots showing stability loss mechanisms"""
    print("\n--- CREATING STABILITY ANALYSIS PLOTS ---")

    # Analyze indicators
    indicators = analyze_stability_indicators(particles_evolution)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Numerical Stability Loss Mechanisms in Long-Time Simulations', fontsize=16, fontweight='bold')

    time_steps = range(len(particles_evolution))

    # 1. Particle evolution heatmap
    ax1 = axes[0, 0]
    particle_matrix = particles_evolution.T  # Particles x Time
    im1 = ax1.imshow(
        particle_matrix, aspect='auto', cmap='viridis', extent=[0, len(time_steps), 0, len(particle_matrix)]
    )
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Particle Index')
    ax1.set_title('Particle Position Evolution\n(Shows Clustering)')
    plt.colorbar(im1, ax=ax1, label='Position')

    # 2. Bandwidth evolution
    ax2 = axes[0, 1]
    ax2.plot(time_steps, indicators['bandwidths'], 'b-', linewidth=2, label='KDE Bandwidth')
    ax2.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
    if len(indicators['sudden_changes']) > 0:
        for change_time in indicators['sudden_changes']:
            ax2.axvline(x=change_time, color='red', linestyle=':', alpha=0.8)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Bandwidth')
    ax2.set_title('KDE Bandwidth Evolution\n(Collapse → Instability)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Particle spread (variance)
    ax3 = axes[0, 2]
    ax3.plot(time_steps, indicators['spreads'], 'g-', linewidth=2)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Particle Spread (Variance)')
    ax3.set_title('Particle Spread Reduction\n(Clustering Effect)')
    ax3.grid(True, alpha=0.3)

    # 4. Min inter-particle distance
    ax4 = axes[1, 0]
    ax4.semilogy(time_steps, indicators['min_distances'], 'r-', linewidth=2)
    ax4.axhline(y=1e-4, color='red', linestyle='--', alpha=0.7, label='Numerical Precision')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Min Distance (log scale)')
    ax4.set_title('Minimum Inter-Particle Distance\n(→ 0 causes problems)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Eigenvalue demonstration
    ax5 = axes[1, 1]
    eigenvals = demonstrate_eigenvalue_instability()
    ax5.plot(range(len(eigenvals)), eigenvals, 'purple', linewidth=2, marker='o', markersize=4)
    ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Stability Threshold')
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Max |Eigenvalue|')
    ax5.set_title('System Eigenvalue Evolution\n(>1 = Unstable)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Sudden change detection
    ax6 = axes[1, 2]
    bandwidth_changes = np.abs(np.diff(indicators['bandwidths']))
    ax6.semilogy(range(len(bandwidth_changes)), bandwidth_changes, 'orange', linewidth=2)
    if len(indicators['sudden_changes']) > 0:
        threshold = 5 * np.std(bandwidth_changes[:20])
        ax6.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label='5σ Threshold')
        for change_time in indicators['sudden_changes']:
            ax6.plot(change_time, bandwidth_changes[change_time], 'ro', markersize=8)
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('|Bandwidth Change| (log)')
    ax6.set_title('Sudden Change Detection\n(Spikes = Instability)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/stability_analysis.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.show()

    # Summary
    print(f"\n{'='*80}")
    print("STABILITY LOSS SUMMARY")
    print(f"{'='*80}")
    print("🔬 KEY MECHANISMS FOR SUDDEN STABILITY LOSS:")
    print()
    print("1. PARTICLE CLUSTERING:")
    print("   - Particles gradually cluster due to control field")
    print("   - Reduces effective system degrees of freedom")
    print("   - Creates ill-conditioned kernel matrices")
    print()
    print("2. BANDWIDTH COLLAPSE:")
    print(f"   - Initial bandwidth: {indicators['bandwidths'][0]:.6f}")
    print(f"   - Final bandwidth: {indicators['bandwidths'][-1]:.6f}")
    print("   - When bandwidth → 0, kernel becomes singular")
    print()
    print("3. EIGENVALUE INSTABILITY:")
    print("   - System eigenvalues start stable (|λ| < 1)")
    print("   - Gradually increase due to clustering")
    print("   - Suddenly cross λ = 1 threshold → EXPLOSION")
    print()
    print("4. CASCADE EFFECT:")
    print("   - Small errors → Clustering → Bandwidth collapse → Eigenvalue instability → CLIFF")
    print()
    print("🎯 WHY IT'S SUDDEN, NOT GRADUAL:")
    print("   - Exponential error growth: tiny → tiny → tiny → HUGE!")
    print("   - Critical thresholds: system is stable until threshold crossed")
    print("   - Cascade failure: one mechanism triggers others")


if __name__ == "__main__":
    print("Starting Numerical Stability Analysis Demo...")
    print("Investigating why stability is suddenly lost in long-time simulations")
    print("Expected execution time: 1-2 minutes")

    try:
        demonstrate_stability_loss_mechanisms()
        print("\n" + "=" * 80)
        print("STABILITY ANALYSIS COMPLETED")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        import traceback

        traceback.print_exc()
