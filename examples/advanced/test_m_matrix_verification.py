"""
Test M-matrix verification infrastructure for QP particle-collocation.

This script validates the newly implemented verification methods on a simple
1D LQ-MFG problem to measure empirical M-matrix satisfaction rate.

Expected outcomes:
- Success rate > 95%: Current constraints are effective
- Success rate 70-95%: Constraints need tuning
- Success rate < 70%: Need different approach (reformulation)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver, MonotonicityStats
from mfg_pde.core.mfg_problem import MFGProblem

# Configuration
SEED = 42
np.random.seed(SEED)

# Output directory
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "advanced"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_test_problem():
    """Create simple 1D MFG problem for testing."""
    # Simple 1D MFG with moderate diffusion
    problem = MFGProblem(
        xmin=0.0,
        xmax=1.0,
        Nx=51,
        T=1.0,
        Nt=51,
        sigma=0.2,  # Moderate diffusion
        coefCT=0.5,  # Control cost coefficient
    )

    return problem


def test_m_matrix_verification(problem, n_points):
    """
    Test M-matrix verification on collocation solver.

    Args:
        problem: MFG problem instance
        n_points: Number of collocation points

    Returns:
        stats: MonotonicityStats object with verification results
    """
    # Create collocation points
    x = np.linspace(problem.xmin, problem.xmax, n_points)
    collocation_points = x.reshape(-1, 1)

    # Create a minimal solver instance for testing verification methods
    # We don't need full solver functionality, just the GFDM infrastructure
    class TestableGFDMSolver(HJBGFDMSolver):
        """Minimal concrete implementation for testing."""

        def solve_hjb_system(self, M_density_evolution_from_FP, U_final_condition_at_T, U_from_prev_picard):
            """Minimal implementation (not used in this test)."""
            return np.zeros_like(U_from_prev_picard)

    solver = TestableGFDMSolver(
        problem=problem,
        collocation_points=collocation_points,
        delta=0.15,  # Neighborhood radius
        taylor_order=2,
        weight_function="wendland",
        use_monotone_constraints=True,
        qp_optimization_level="basic",
    )

    print(f"Solver initialized: {solver.hjb_method_name}")
    print(f"Collocation points: {n_points}")
    print(f"Delta (neighborhood): {solver.delta}")
    print(f"Taylor order: {solver.taylor_order}")
    print()

    # Verify M-matrix property for each point
    stats = MonotonicityStats()

    print("Verifying M-matrix property at each collocation point...")

    for point_idx in range(n_points):
        taylor_data = solver.taylor_matrices[point_idx]

        if taylor_data is None:
            print(f"  Point {point_idx}: Insufficient neighbors")
            continue

        # Find Laplacian derivative index
        laplacian_idx = None
        for k, beta in enumerate(solver.multi_indices):
            if beta == (2,):  # Second derivative in 1D
                laplacian_idx = k
                break

        if laplacian_idx is None:
            print(f"  Point {point_idx}: No Laplacian in multi-indices")
            continue

        # Extract FD weights for Laplacian
        weights = solver._compute_fd_weights_from_taylor(taylor_data, laplacian_idx)

        if weights is None:
            print(f"  Point {point_idx}: FD weight extraction failed")
            continue

        # Verify M-matrix property
        is_monotone, diagnostics = solver._check_m_matrix_property(weights, point_idx)

        # Record result
        stats.record_point(point_idx, is_monotone, diagnostics)

        # Print status for first few points
        if point_idx < 5 or not is_monotone:
            status = "✓" if is_monotone else "✗"
            print(
                f"  Point {point_idx}: {status} "
                f"(w_center={diagnostics['w_center']:.6f}, "
                f"min_neighbor={diagnostics['min_neighbor_weight']:.6f})"
            )

    print()
    return stats


def analyze_results(stats):
    """Analyze and display M-matrix verification results."""
    summary = stats.get_summary()

    print("=" * 70)
    print("M-MATRIX VERIFICATION RESULTS")
    print("=" * 70)
    print()
    print(f"Total points tested: {summary['total_points']}")
    print(f"Monotone points: {summary['monotone_points']}")
    print(f"Violating points: {summary['num_violating_points']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print(f"Max violation severity: {summary['max_violation_severity']:.6f}")
    print()

    # Interpretation
    success_rate = summary["success_rate"]
    if success_rate > 95:
        assessment = "EXCELLENT - Current constraints are effective ✓"
    elif success_rate > 70:
        assessment = "GOOD - Constraints may need tuning"
    else:
        assessment = "NEEDS WORK - Consider reformulation"

    print(f"Assessment: {assessment}")
    print("=" * 70)
    print()

    return summary


def plot_results(stats, output_path):
    """Create visualization of M-matrix verification results."""
    summary = stats.get_summary()

    _fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Success rate pie chart
    ax1 = axes[0]
    monotone = summary["monotone_points"]
    violations = summary["num_violating_points"]

    colors = ["#28a745", "#d93f0b"]
    labels = [f"Monotone ({monotone})", f"Violations ({violations})"]
    ax1.pie([monotone, violations], labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax1.set_title(f"M-Matrix Satisfaction\n(Success Rate: {summary['success_rate']:.1f}%)")

    # Right: Violation severity distribution
    ax2 = axes[1]
    if stats.worst_violations and len(stats.worst_violations) > 2:
        # Only show histogram if we have enough violations
        severities = [v["severity"] for v in stats.worst_violations]
        # Use adaptive binning based on number of violations
        n_bins = min(20, max(5, len(severities) // 5))
        ax2.hist(severities, bins=n_bins, color="#d93f0b", alpha=0.7, edgecolor="black")
        ax2.set_xlabel("Violation Severity")
        ax2.set_ylabel("Count")
        ax2.set_title("Violation Severity Distribution")
        ax2.grid(True, alpha=0.3)
    elif stats.worst_violations:
        # Show violation details as text
        severities = [v["severity"] for v in stats.worst_violations]
        text = f"Violations: {len(severities)}\n"
        for i, sev in enumerate(severities):
            text += f"Point {stats.worst_violations[i]['point_idx']}: {sev:.2f}\n"
        ax2.text(0.5, 0.5, text, ha="center", va="center", fontsize=12, transform=ax2.transAxes)
        ax2.set_title("Violation Details")
    else:
        ax2.text(
            0.5,
            0.5,
            "No violations detected\n✓",
            ha="center",
            va="center",
            fontsize=16,
            transform=ax2.transAxes,
        )
        ax2.set_title("Violation Severity Distribution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    """Run M-matrix verification test."""
    print("=" * 70)
    print("M-MATRIX VERIFICATION TEST")
    print("QP Particle-Collocation Implementation")
    print("=" * 70)
    print()

    # Create test problem
    problem = create_test_problem()
    print("Test problem: 1D LQ-MFG")
    print(f"Domain: [{problem.xmin}, {problem.xmax}]")
    print(f"Time horizon: T = {problem.T}")
    print(f"Diffusion: σ = {problem.sigma}")
    print()

    # Test with different grid resolutions
    for n_points in [30, 50, 100]:
        print(f"\n{'=' * 70}")
        print(f"Testing with {n_points} collocation points")
        print(f"{'=' * 70}\n")

        stats = test_m_matrix_verification(problem, n_points=n_points)
        analyze_results(stats)

        # Save plot
        output_path = OUTPUT_DIR / f"m_matrix_verification_n{n_points}.png"
        plot_results(stats, output_path)
        print()


if __name__ == "__main__":
    main()
