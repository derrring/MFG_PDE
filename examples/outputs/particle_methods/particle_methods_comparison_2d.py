#!/usr/bin/env python3
"""
2D Crowd Evacuation: Particle Methods Comparison

Compare different solver combinations for 2D MFG:
1. Grid-Grid (FDM-FDM): Both HJB and FP on grid
2. Grid-Particle (FDM-Particle): HJB on grid, FP with particles
3. Particle-Particle: Both HJB and FP with particles (if available)

This tests the framework's particle method capabilities for true 2D problems.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem
from mfg_pde.factory import create_basic_solver


class CrowdEvacuation2D(GridBasedMFGProblem):
    """
    2D crowd evacuation problem using proper nD infrastructure.

    Physical Setup:
    - Domain: 10m × 10m room
    - Initial: Gaussian crowd at (5.0, 7.0)
    - Goal: Evacuate to y=0 (bottom exit)
    - Dynamics: Diffusion + congestion effects
    """

    def __init__(
        self,
        grid_resolution: int = 20,
        time_horizon: float = 8.0,
        num_timesteps: int = 50,
        diffusion_coeff: float = 0.2,
        congestion_weight: float = 2.0,
    ):
        # Store problem parameters
        self.congestion_weight = congestion_weight
        self.start_location = np.array([5.0, 7.0])
        self.exit_location = np.array([5.0, 0.0])

        # Initialize GridBasedMFGProblem with 2D domain
        super().__init__(
            domain_bounds=(0.0, 10.0, 0.0, 10.0),  # (xmin, xmax, ymin, ymax)
            grid_resolution=grid_resolution,
            time_domain=(time_horizon, num_timesteps),
            diffusion_coeff=diffusion_coeff,
        )

    def hamiltonian(self, x, m, p, t):
        """
        Hamiltonian: H = (1/2)|p|² + λ·m·|p|²

        Args:
            x: (N, 2) spatial coordinates
            m: (N,) density values
            p: (N, 2) gradient vector ∇u
            t: scalar time

        Returns:
            (N,) Hamiltonian values
        """
        # Handle both single point and vectorized
        if p.ndim == 1:
            p_squared = np.sum(p**2)
        else:
            p_squared = np.sum(p**2, axis=1)

        # Kinetic energy + congestion penalty
        return 0.5 * p_squared + self.congestion_weight * m * p_squared

    def initial_density(self, x):
        """
        Initial crowd distribution: Gaussian centered at start location.

        Args:
            x: (N, 2) spatial coordinates

        Returns:
            (N,) density values
        """
        # Distance from start location
        dist_sq = np.sum((x - self.start_location) ** 2, axis=1)

        # Gaussian distribution
        sigma = 1.5
        density = np.exp(-dist_sq / (2 * sigma**2))

        # Normalize to integrate to 1
        total_mass = np.sum(density) * (10.0 / x.shape[0]) ** 2
        return density / total_mass if total_mass > 0 else density

    def terminal_cost(self, x):
        """
        Terminal cost: Distance to exit at y=0.

        Args:
            x: (N, 2) spatial coordinates

        Returns:
            (N,) cost values
        """
        # Cost is y-coordinate (distance to y=0)
        return x[:, 1]

    def running_cost(self, x, t):
        """
        Running cost: Time penalty to encourage immediate evacuation.

        Args:
            x: (N, 2) spatial coordinates
            t: scalar time

        Returns:
            (N,) cost values
        """
        # Constant time penalty
        return np.ones(x.shape[0])

    def setup_components(self):
        """Modern FDM solvers don't use MFGComponents."""
        return None


def run_experiment(
    method_name: str,
    problem: CrowdEvacuation2D,
    hjb_method: str = "fdm",
    fp_method: str = "fdm",
    num_particles: int = 4000,
    max_iterations: int = 25,
    damping: float = 0.5,
):
    """
    Run single experiment with specified solver configuration.

    Args:
        method_name: Display name for method
        problem: MFG problem instance
        hjb_method: "fdm" or "particle"
        fp_method: "fdm" or "particle"
        num_particles: Number of particles (if using particle method)
        max_iterations: Maximum Picard iterations
        damping: Picard damping factor

    Returns:
        Dictionary with results and timing information
    """
    print(f"\n{'=' * 70}")
    print(f"METHOD: {method_name}")
    print(f"{'=' * 70}")
    print(f"  HJB solver: {hjb_method.upper()}")
    print(f"  FP solver: {fp_method.upper()}")
    if fp_method == "particle":
        print(f"  Particles: {num_particles}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Damping: {damping}")
    print()

    start_time = time.time()

    try:
        if hjb_method == "fdm" and fp_method == "fdm":
            # Grid-Grid: Standard factory
            solver = create_basic_solver(
                problem,
                damping=damping,
                max_iterations=max_iterations,
            )

        elif hjb_method == "fdm" and fp_method == "particle":
            # Grid-Particle: Custom config using correct API
            from mfg_pde import solve_mfg
            from mfg_pde.config import ConfigBuilder

            config = (
                ConfigBuilder()
                .solver_hjb(method="fdm", accuracy_order=2)
                .solver_fp_particle(num_particles=num_particles)
                .picard(max_iterations=max_iterations, tolerance=1e-4, damping_factor=damping)
                .build()
            )

            # Use solve_mfg function
            class SolverWrapper:
                """Wrapper to match expected solver interface."""

                def __init__(self, problem, config):
                    self.problem = problem
                    self.config = config

                def solve(self):
                    return solve_mfg(self.problem, config=self.config)

            solver = SolverWrapper(problem, config)

        else:
            raise NotImplementedError(f"Method combination not yet implemented: HJB={hjb_method}, FP={fp_method}")

        # Solve
        print("Solving...")
        result = solver.solve()

        solve_time = time.time() - start_time

        # Extract results
        converged = result.converged if hasattr(result, "converged") else False
        num_iters = result.num_iterations if hasattr(result, "num_iterations") else 0

        # Get density at final time
        if hasattr(result, "M"):
            M_final = result.M[-1] if result.M.ndim > 1 else result.M
        else:
            M_final = None

        # Compute mass conservation
        if M_final is not None:
            initial_mass = 1.0  # Normalized
            final_mass = np.sum(M_final) * (10.0 / problem.geometry.grid.num_points[0]) ** 2
            mass_loss_pct = abs(1.0 - final_mass) * 100
        else:
            mass_loss_pct = None

        print(f"\n✓ Solve completed in {solve_time:.2f}s")
        print(f"  Converged: {converged}")
        print(f"  Iterations: {num_iters}")
        if mass_loss_pct is not None:
            print(f"  Mass loss: {mass_loss_pct:.4f}%")

        return {
            "method": method_name,
            "hjb": hjb_method,
            "fp": fp_method,
            "converged": converged,
            "iterations": num_iters,
            "solve_time": solve_time,
            "mass_loss_pct": mass_loss_pct,
            "result": result,
            "success": True,
        }

    except Exception as e:
        solve_time = time.time() - start_time
        print(f"\n✗ Method failed after {solve_time:.2f}s")
        print(f"  Error: {e!s}")

        return {
            "method": method_name,
            "hjb": hjb_method,
            "fp": fp_method,
            "converged": False,
            "iterations": 0,
            "solve_time": solve_time,
            "mass_loss_pct": None,
            "result": None,
            "success": False,
            "error": str(e),
        }


def visualize_comparison(results: list, output_dir: Path):
    """Create comparison visualizations."""

    # Filter successful results
    successful = [r for r in results if r["success"]]

    if not successful:
        print("No successful results to visualize")
        return

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Particle Methods Comparison: 2D Crowd Evacuation", fontsize=16, fontweight="bold")

    # Extract data
    methods = [r["method"] for r in successful]
    solve_times = [r["solve_time"] for r in successful]
    iterations = [r["iterations"] for r in successful]
    mass_losses = [r["mass_loss_pct"] for r in successful if r["mass_loss_pct"] is not None]

    # Plot 1: Solve time comparison
    ax = axes[0, 0]
    colors = ["#2ecc71" if r["converged"] else "#e74c3c" for r in successful]
    bars = ax.bar(range(len(methods)), solve_times, color=colors, alpha=0.7, edgecolor="black")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Solve Time (s)", fontsize=12)
    ax.set_title("Computation Time", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, (bar, time_val) in enumerate(zip(bars, solve_times)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{time_val:.1f}s", ha="center", va="bottom", fontsize=10)

    # Plot 2: Iterations comparison
    ax = axes[0, 1]
    bars = ax.bar(range(len(methods)), iterations, color=colors, alpha=0.7, edgecolor="black")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Iterations", fontsize=12)
    ax.set_title("Convergence Iterations", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, (bar, iter_val) in enumerate(zip(bars, iterations)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{iter_val}", ha="center", va="bottom", fontsize=10)

    # Plot 3: Mass conservation
    ax = axes[1, 0]
    if mass_losses:
        methods_with_mass = [r["method"] for r in successful if r["mass_loss_pct"] is not None]
        colors_mass = ["#2ecc71" if loss < 1.0 else "#e74c3c" for loss in mass_losses]
        bars = ax.bar(range(len(methods_with_mass)), mass_losses, color=colors_mass, alpha=0.7, edgecolor="black")
        ax.set_xticks(range(len(methods_with_mass)))
        ax.set_xticklabels(methods_with_mass, rotation=45, ha="right")
        ax.set_ylabel("Mass Loss (%)", fontsize=12)
        ax.set_title("Mass Conservation", fontsize=13, fontweight="bold")
        ax.axhline(y=1.0, color="orange", linestyle="--", label="1% threshold", linewidth=2)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()

        # Add value labels
        for i, (bar, loss_val) in enumerate(zip(bars, mass_losses)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, height, f"{loss_val:.2f}%", ha="center", va="bottom", fontsize=9
            )
    else:
        ax.text(0.5, 0.5, "No mass data available", ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis("off")

    # Create summary data
    table_data = []
    table_data.append(["Method", "Converged", "Time (s)", "Iters", "Mass Loss"])
    for r in successful:
        conv_str = "✓" if r["converged"] else "✗"
        time_str = f"{r['solve_time']:.1f}"
        iter_str = str(r["iterations"])
        mass_str = f"{r['mass_loss_pct']:.2f}%" if r["mass_loss_pct"] is not None else "N/A"
        table_data.append([r["method"], conv_str, time_str, iter_str, mass_str])

    # Create table
    table = ax.table(cellText=table_data, cellLoc="center", loc="center", colWidths=[0.35, 0.15, 0.15, 0.15, 0.20])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor("#3498db")
        cell.set_text_props(weight="bold", color="white")

    # Color convergence cells
    for i in range(1, len(table_data)):
        conv_cell = table[(i, 1)]
        if table_data[i][1] == "✓":
            conv_cell.set_facecolor("#d5f4e6")
        else:
            conv_cell.set_facecolor("#fadbd8")

    ax.set_title("Results Summary", fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / "particle_methods_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n✓ Comparison figure saved to: {output_file}")
    plt.close()


def main():
    """Run particle methods comparison experiments."""

    print("=" * 70)
    print("2D CROWD EVACUATION: PARTICLE METHODS COMPARISON")
    print("=" * 70)
    print()
    print("Testing different solver combinations:")
    print("  1. Grid-Grid (FDM-FDM): Both on grid")
    print("  2. Grid-Particle (FDM-Particle): HJB on grid, FP with particles")
    print("  3. Particle-Particle: Both with particles (if available)")
    print()

    # Create output directory
    output_dir = Path("examples/outputs/particle_methods")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Common problem setup (smaller for faster experiments)
    print("Creating 2D crowd evacuation problem...")
    problem = CrowdEvacuation2D(
        grid_resolution=15,  # 15×15 grid for faster testing
        time_horizon=8.0,
        num_timesteps=40,
        diffusion_coeff=0.2,
        congestion_weight=2.0,
    )
    print(f"✓ Problem created: {problem.geometry.grid.num_points[0]}×{problem.geometry.grid.num_points[1]} grid\n")

    # Experiment configurations
    experiments = [
        {
            "name": "Grid-Grid\n(FDM-FDM)",
            "hjb": "fdm",
            "fp": "fdm",
            "particles": 0,
            "max_iters": 20,
            "damping": 0.5,
        },
        {
            "name": "Grid-Particle\n(FDM-Particle 3k)",
            "hjb": "fdm",
            "fp": "particle",
            "particles": 3000,
            "max_iters": 20,
            "damping": 0.5,
        },
        {
            "name": "Grid-Particle\n(FDM-Particle 5k)",
            "hjb": "fdm",
            "fp": "particle",
            "particles": 5000,
            "max_iters": 20,
            "damping": 0.5,
        },
    ]

    # Run experiments
    results = []
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'=' * 70}")
        print(f"EXPERIMENT {i}/{len(experiments)}")
        print(f"{'=' * 70}")

        result = run_experiment(
            method_name=exp["name"],
            problem=problem,
            hjb_method=exp["hjb"],
            fp_method=exp["fp"],
            num_particles=exp["particles"],
            max_iterations=exp["max_iters"],
            damping=exp["damping"],
        )
        results.append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for r in results:
        status = "✓ SUCCESS" if r["success"] else "✗ FAILED"
        print(f"\n{r['method']}: {status}")
        if r["success"]:
            print(f"  Converged: {r['converged']}")
            print(f"  Time: {r['solve_time']:.2f}s")
            print(f"  Iterations: {r['iterations']}")
            if r["mass_loss_pct"] is not None:
                print(f"  Mass loss: {r['mass_loss_pct']:.4f}%")
        else:
            print(f"  Error: {r.get('error', 'Unknown error')}")

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    visualize_comparison(results, output_dir)

    print("\n" + "=" * 70)
    print("✓ PARTICLE METHODS COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\nGenerated files in: {output_dir}")
    print("  - particle_methods_comparison.png")


if __name__ == "__main__":
    main()
