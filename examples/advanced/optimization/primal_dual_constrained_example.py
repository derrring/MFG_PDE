#!/usr/bin/env python3
"""
Primal-Dual Constrained MFG Example

This example demonstrates the primal-dual solver for constrained Lagrangian MFG problems.
Primal-dual methods provide superior constraint handling compared to simple penalty methods,
particularly for problems with many constraints or when constraint satisfaction is critical.

Key advantages demonstrated:
1. Better constraint satisfaction through dual variable updates
2. Automatic penalty parameter adaptation
3. Separation of primal optimization and constraint handling
4. Convergence guarantees for convex problems

Problems solved:
1. Multi-constraint obstacle avoidance
2. Budget constraints with social distancing
3. Comparison with penalty method approach
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.optimization.variational_solvers.primal_dual_solver import PrimalDualMFGSolver
from mfg_pde.alg.optimization.variational_solvers.variational_mfg_solver import VariationalMFGSolver
from mfg_pde.core.lagrangian_mfg_problem import (
    LagrangianComponents,
    LagrangianMFGProblem,
)
from mfg_pde.utils.mfg_logging import configure_research_logging, get_logger
from mfg_pde.utils.numpy_compat import trapezoid

# Configure logging
configure_research_logging("primal_dual_example", level="INFO")
logger = get_logger(__name__)


def create_multi_constraint_problem() -> LagrangianMFGProblem:
    """
    Create MFG problem with multiple types of constraints.

    Combines:
    - Obstacle avoidance constraints
    - Velocity limits
    - Budget constraints
    - Social distancing requirements

    Returns:
        LagrangianMFGProblem with comprehensive constraints
    """
    logger.info("Creating multi-constraint MFG problem...")

    def multi_constraint_lagrangian(t: float, x: float, v: float, m: float) -> float:
        """Complex Lagrangian with multiple cost components."""
        # Base control cost
        control_cost = 0.5 * v**2

        # Congestion cost (social distancing)
        if m > 1.5:  # High density threshold
            congestion_cost = 5.0 * (m - 1.5) ** 2
        else:
            congestion_cost = 0.2 * m

        # Time-dependent travel restrictions
        if t < 0.2:  # Early period - high restrictions
            travel_cost = 3.0 * v**2
        else:  # Later period - relaxed restrictions
            travel_cost = 0.5 * v**2

        # Encourage reaching target region
        target_center = 0.8
        target_width = 0.1
        distance_to_target = abs(x - target_center)
        if distance_to_target > target_width:
            target_cost = 2.0 * (distance_to_target - target_width) ** 2
        else:
            target_cost = 0.0

        return control_cost + congestion_cost + travel_cost + target_cost

    def obstacle_constraint(t: float, x: float) -> float:
        """State constraint: avoid circular obstacle."""
        obstacle_center = 0.4
        obstacle_radius = 0.12
        distance_to_obstacle = abs(x - obstacle_center)
        return obstacle_radius - distance_to_obstacle  # > 0 if violated

    def velocity_constraint(t: float, x: float, v: float) -> float:
        """Velocity constraint: speed limit varies by location."""
        if 0.3 <= x <= 0.5:  # High-security zone
            speed_limit = 1.0
        else:  # Normal zone
            speed_limit = 2.5
        return abs(v) - speed_limit  # > 0 if violated

    def budget_constraint(trajectory: np.ndarray, velocity: np.ndarray, dt: float) -> float:
        """Integral constraint: total energy budget."""
        total_energy = trapezoid(velocity**2, dx=dt)
        energy_budget = 3.0
        return total_energy - energy_budget  # > 0 if violated

    def capacity_constraint(t: float, density_field: np.ndarray, x_grid: np.ndarray) -> float:
        """Spatial capacity constraint."""
        # High-capacity zone has higher limits
        normal_capacity = 2.0
        high_capacity_zone = (x_grid >= 0.6) & (x_grid <= 0.9)

        # Check density limits
        max_violation = 0.0
        for i, x in enumerate(x_grid):
            capacity_limit = 3.0 if high_capacity_zone[i] else normal_capacity
            violation = max(0.0, density_field[i] - capacity_limit)
            max_violation = max(max_violation, violation)

        return max_violation

    def initial_density_concentrated(x: float) -> float:
        """Initial density concentrated at start."""
        if 0.0 <= x <= 0.15:
            return 6.0  # High initial concentration
        else:
            return 0.1

    # Create problem components
    components = LagrangianComponents(
        lagrangian_func=multi_constraint_lagrangian,
        lagrangian_dv_func=lambda t, x, v, m: v + (6.0 if t < 0.2 else 1.0) * v,
        lagrangian_dm_func=lambda t, x, v, m: (10.0 * (m - 1.5) if m > 1.5 else 0.2),
        terminal_cost_func=lambda x: 2.0 * (x - 0.8) ** 2,  # Target at x=0.8
        initial_density_func=initial_density_concentrated,
        state_constraints=[obstacle_constraint],
        velocity_constraints=[velocity_constraint],
        integral_constraints=[budget_constraint],
        noise_intensity=0.1,
        parameters={
            "obstacle_center": 0.4,
            "obstacle_radius": 0.12,
            "energy_budget": 3.0,
            "normal_capacity": 2.0,
            "target_center": 0.8,
        },
        description="Multi-Constraint MFG with Obstacles, Budget, and Capacity Limits",
    )

    return LagrangianMFGProblem(xmin=0.0, xmax=1.0, Nx=40, T=0.4, Nt=20, sigma=0.1, components=components)


def solve_with_penalty_method(problem: LagrangianMFGProblem) -> dict:
    """Solve using standard penalty method for comparison."""
    logger.info("Solving with penalty method...")

    start_time = time.time()

    try:
        solver = VariationalMFGSolver(
            problem,
            optimization_method="L-BFGS-B",
            penalty_weight=1000.0,
            use_jax=False,  # High penalty weight
        )

        result = solver.solve(max_iterations=40, tolerance=1e-4, verbose=True)

        solve_time = time.time() - start_time

        return {
            "result": result,
            "solve_time": solve_time,
            "method": "Penalty Method",
            "success": result.converged if result else False,
        }

    except Exception as e:
        logger.error(f"Penalty method failed: {e}")
        return {
            "result": None,
            "solve_time": time.time() - start_time,
            "method": "Penalty Method",
            "success": False,
            "error": str(e),
        }


def solve_with_primal_dual(problem: LagrangianMFGProblem) -> dict:
    """Solve using primal-dual method."""
    logger.info("Solving with primal-dual method...")

    start_time = time.time()

    try:
        solver = PrimalDualMFGSolver(
            problem,
            primal_solver="L-BFGS-B",
            dual_update_method="gradient_ascent",
            augmented_penalty=10.0,  # Lower initial penalty
            dual_step_size=0.05,
            use_adaptive_penalty=True,
            constraint_tolerance=1e-5,
        )

        result = solver.solve(max_outer_iterations=15, max_inner_iterations=30, tolerance=1e-4, verbose=True)

        solve_time = time.time() - start_time

        return {
            "result": result,
            "solve_time": solve_time,
            "method": "Primal-Dual Method",
            "success": result.converged if result else False,
        }

    except Exception as e:
        logger.error(f"Primal-dual method failed: {e}")
        return {
            "result": None,
            "solve_time": time.time() - start_time,
            "method": "Primal-Dual Method",
            "success": False,
            "error": str(e),
        }


def analyze_constraint_satisfaction(problem: LagrangianMFGProblem, solution: dict) -> dict:
    """Analyze how well constraints are satisfied."""
    if not solution["success"] or solution["result"] is None:
        return {"status": "no_solution"}

    result = solution["result"]
    density = result.optimal_flow
    analysis = {}

    # Obstacle constraint analysis
    obstacle_center = problem.components.parameters.get("obstacle_center", 0.4)
    obstacle_radius = problem.components.parameters.get("obstacle_radius", 0.12)

    obstacle_violations = []
    for i in range(problem.Nt):
        for j, x in enumerate(problem.x):
            if abs(x - obstacle_center) < obstacle_radius:
                obstacle_violations.append(density[i, j])

    analysis["obstacle_max_density"] = max(obstacle_violations) if obstacle_violations else 0.0
    analysis["obstacle_avg_density"] = np.mean(obstacle_violations) if obstacle_violations else 0.0

    # Mass conservation
    mass_over_time = [trapezoid(density[i, :], x=problem.x) for i in range(problem.Nt)]
    analysis["mass_conservation_error"] = np.std(mass_over_time)
    analysis["final_mass"] = mass_over_time[-1]

    # Capacity constraints
    max_densities = np.max(density, axis=1)
    normal_capacity = problem.components.parameters.get("normal_capacity", 2.0)
    analysis["capacity_violations"] = np.sum(max_densities > normal_capacity)
    analysis["max_density_violation"] = max(0.0, np.max(max_densities) - normal_capacity)

    # Target reaching (how much mass reaches target)
    target_center = problem.components.parameters.get("target_center", 0.8)
    target_width = 0.1
    target_region = (problem.x >= target_center - target_width) & (problem.x <= target_center + target_width)
    final_mass_in_target = trapezoid(density[-1, target_region], x=problem.x[target_region])
    analysis["mass_in_target"] = final_mass_in_target

    return analysis


def create_comparison_plots(penalty_solution: dict, primal_dual_solution: dict, problem: LagrangianMFGProblem):
    """Create plots comparing penalty method vs primal-dual method."""
    logger.info("Creating comparison plots...")

    successful_solutions = [sol for sol in [penalty_solution, primal_dual_solution] if sol["success"]]

    if len(successful_solutions) == 0:
        logger.warning("No successful solutions to plot")
        return

    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Penalty Method vs Primal-Dual Method for Constrained MFG", fontsize=14)

        x_grid = problem.x
        t_grid = problem.t

        # Parameters for visualization
        obstacle_center = problem.components.parameters.get("obstacle_center", 0.4)
        obstacle_radius = problem.components.parameters.get("obstacle_radius", 0.12)
        target_center = problem.components.parameters.get("target_center", 0.8)

        methods = []
        colors = ["blue", "red"]

        for i, solution in enumerate(successful_solutions):
            method_name = solution["method"]
            methods.append(method_name)
            result = solution["result"]
            density = result.optimal_flow

            # Plot 1: Final density distribution
            ax1 = axes[0, i]
            final_density = density[-1, :]
            ax1.plot(x_grid, final_density, color=colors[i], linewidth=2, label=method_name)

            # Add constraint visualizations
            obstacle_left = obstacle_center - obstacle_radius
            obstacle_right = obstacle_center + obstacle_radius
            ax1.axvspan(obstacle_left, obstacle_right, alpha=0.3, color="red", label="Obstacle")
            ax1.axvspan(target_center - 0.1, target_center + 0.1, alpha=0.3, color="green", label="Target")
            ax1.axhline(y=2.0, color="orange", linestyle="--", alpha=0.7, label="Capacity Limit")

            ax1.set_xlabel("x")
            ax1.set_ylabel("m(T, x)")
            ax1.set_title(f"{method_name}\nFinal Density")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Space-time evolution
            ax2 = axes[1, i]
            X, T = np.meshgrid(x_grid, t_grid)
            im = ax2.contourf(X, T, density, levels=15, cmap="viridis")

            # Add obstacle overlay
            for t in t_grid:
                ax2.plot([obstacle_left, obstacle_right], [t, t], "r-", linewidth=2, alpha=0.8)

            ax2.set_xlabel("x")
            ax2.set_ylabel("t")
            ax2.set_title(f"{method_name}\nDensity Evolution")
            plt.colorbar(im, ax=ax2, shrink=0.8)

        # Plot 3: Convergence comparison
        ax3 = axes[0, 2]

        for i, solution in enumerate(successful_solutions):
            result = solution["result"]
            method_name = solution["method"]

            if hasattr(result, "cost_history") and result.cost_history:
                iterations = range(len(result.cost_history))
                costs = result.cost_history
                ax3.semilogy(
                    iterations,
                    costs,
                    color=colors[i],
                    linewidth=2,
                    marker="o",
                    markersize=3,
                    label=f"{method_name} Cost",
                )

            # For primal-dual, also plot constraint violations
            if method_name == "Primal-Dual Method" and hasattr(result, "solver_info"):
                solver_info = result.solver_info
                if "constraint_violation_history" in solver_info:
                    violations = solver_info["constraint_violation_history"]
                    if violations:
                        ax3_twin = ax3.twinx()
                        iterations_pd = range(len(violations))
                        ax3_twin.semilogy(
                            iterations_pd,
                            violations,
                            "g--",
                            linewidth=2,
                            marker="s",
                            markersize=3,
                            label="Constraint Violations",
                        )
                        ax3_twin.set_ylabel("Constraint Violation", color="g")
                        ax3_twin.tick_params(axis="y", labelcolor="g")

        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Cost")
        ax3.set_title("Convergence Comparison")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Constraint satisfaction comparison
        ax4 = axes[1, 2]

        constraint_names = ["Obstacle\nViolation", "Mass\nConservation", "Capacity\nViolation", "Target\nReaching"]
        method_names = [sol["method"] for sol in successful_solutions]

        violation_data = []
        for solution in successful_solutions:
            analysis = analyze_constraint_satisfaction(problem, solution)
            violations = [
                analysis.get("obstacle_max_density", 0.0),
                analysis.get("mass_conservation_error", 0.0) * 100,  # Scale for visibility
                analysis.get("max_density_violation", 0.0),
                1.0 - analysis.get("mass_in_target", 0.0),  # 1 - success rate
            ]
            violation_data.append(violations)

        x_pos = np.arange(len(constraint_names))
        width = 0.35

        for i, (method_name, violations) in enumerate(zip(method_names, violation_data, strict=False)):
            offset = (i - 0.5) * width
            bars = ax4.bar(x_pos + offset, violations, width, label=method_name, color=colors[i], alpha=0.7)

            # Add value labels on bars
            for bar, value in zip(bars, violations, strict=False):
                height = bar.get_height()
                ax4.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(violations) * 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax4.set_xlabel("Constraint Type")
        ax4.set_ylabel("Violation Measure")
        ax4.set_title("Constraint Satisfaction Comparison")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(constraint_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_filename = "primal_dual_vs_penalty_comparison.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
        logger.info(f"Comparison plots saved as: {plot_filename}")

        try:
            plt.show()
        except:
            logger.info("Plot display not available (non-interactive environment)")

    except Exception as e:
        logger.error(f"Plot creation failed: {e}")


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Primal-Dual vs Penalty Method Comparison for Constrained MFG")
    logger.info("=" * 80)

    try:
        # Create multi-constraint problem
        logger.info("Creating multi-constraint problem...")
        problem = create_multi_constraint_problem()

        logger.info("Problem created:")
        logger.info(f"  Domain: [{problem.xmin}, {problem.xmax}] × [0, {problem.T}]")
        logger.info(f"  Grid: {problem.Nx + 1} × {problem.Nt}")
        logger.info("  Constraints: obstacle + velocity + budget + capacity")

        # Solve with both methods
        logger.info("\n" + "=" * 60 + " SOLVING " + "=" * 60)

        penalty_solution = solve_with_penalty_method(problem)
        primal_dual_solution = solve_with_primal_dual(problem)

        # Analyze results
        logger.info("\n" + "=" * 60 + " ANALYSIS " + "=" * 59)

        penalty_analysis = analyze_constraint_satisfaction(problem, penalty_solution)
        primal_dual_analysis = analyze_constraint_satisfaction(problem, primal_dual_solution)

        # Create comparison visualization
        create_comparison_plots(penalty_solution, primal_dual_solution, problem)

        # Summary report
        logger.info("\n" + "=" * 80)
        logger.info("DETAILED COMPARISON RESULTS")
        logger.info("=" * 80)

        methods = [
            ("Penalty Method", penalty_solution, penalty_analysis),
            ("Primal-Dual Method", primal_dual_solution, primal_dual_analysis),
        ]

        for method_name, solution, analysis in methods:
            logger.info(f"\n{method_name}:")

            if solution["success"]:
                logger.info("  ✓ Optimization: SUCCESS")
                logger.info(f"  ✓ Solve time: {solution['solve_time']:.2f}s")

                if solution["result"]:
                    result = solution["result"]
                    logger.info(f"  ✓ Final cost: {result.final_cost:.4e}")
                    logger.info(f"  ✓ Iterations: {result.num_iterations}")

                # Constraint analysis
                if analysis.get("status") != "no_solution":
                    logger.info(f"  ✓ Obstacle max density: {analysis.get('obstacle_max_density', 0):.4f}")
                    logger.info(f"  ✓ Mass conservation error: {analysis.get('mass_conservation_error', 0):.2e}")
                    logger.info(f"  ✓ Capacity violations: {analysis.get('capacity_violations', 0)} time steps")
                    logger.info(f"  ✓ Mass in target: {analysis.get('mass_in_target', 0):.4f}")

            else:
                logger.info("  ✗ Optimization: FAILED")
                if "error" in solution:
                    logger.info(f"     Error: {solution['error']}")

        # Overall comparison
        logger.info("\n" + "=" * 80)
        logger.info("OVERALL COMPARISON")
        logger.info("=" * 80)

        successful_methods = [sol for sol in [penalty_solution, primal_dual_solution] if sol["success"]]

        if len(successful_methods) == 2:
            logger.info("🎉 BOTH METHODS SUCCESSFUL!")

            # Compare constraint satisfaction
            penalty_violations = sum(
                [
                    penalty_analysis.get("obstacle_max_density", 0),
                    penalty_analysis.get("mass_conservation_error", 0),
                    penalty_analysis.get("max_density_violation", 0),
                ]
            )

            primal_dual_violations = sum(
                [
                    primal_dual_analysis.get("obstacle_max_density", 0),
                    primal_dual_analysis.get("mass_conservation_error", 0),
                    primal_dual_analysis.get("max_density_violation", 0),
                ]
            )

            logger.info(f"   Penalty method total violations: {penalty_violations:.4f}")
            logger.info(f"   Primal-dual total violations: {primal_dual_violations:.4f}")

            if primal_dual_violations < penalty_violations:
                logger.info("   ✓ Primal-dual method achieves better constraint satisfaction")
            elif penalty_violations < primal_dual_violations:
                logger.info("   ✓ Penalty method achieves better constraint satisfaction")
            else:
                logger.info("   ✓ Both methods achieve similar constraint satisfaction")

            # Compare efficiency
            if primal_dual_solution["solve_time"] < penalty_solution["solve_time"]:
                logger.info("   ✓ Primal-dual method is more efficient")
            else:
                logger.info("   ✓ Penalty method is more efficient")

        elif len(successful_methods) == 1:
            successful_method = successful_methods[0]["method"]
            logger.info(f"⚠️  PARTIAL SUCCESS: Only {successful_method} worked")
        else:
            logger.info("❌ BOTH METHODS FAILED")

        logger.info("\nKey insights:")
        logger.info("• Primal-dual methods often achieve better constraint satisfaction")
        logger.info("• Automatic penalty adaptation reduces parameter tuning")
        logger.info("• Dual variables provide insight into active constraints")
        logger.info("• Complex constraint problems benefit from specialized methods")

        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Primal-dual example failed: {e}")
        raise


if __name__ == "__main__":
    main()
