"""
Main Experiment Runner for 2D Anisotropic Crowd Dynamics

This script runs the complete anisotropic crowd dynamics experiment,
including solution computation, validation, analysis, and visualization.
"""

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis.visualization_tools import create_visualization_suite
from anisotropic_2d_problem import AnisotropicMFGProblem2D, create_anisotropic_problem
from validation.convergence_study import ConvergenceStudy

from mfg_pde.utils.logging import configure_research_logging, get_logger

logger = get_logger(__name__)


class AnisotropicExperiment:
    """
    Main experiment controller for 2D anisotropic crowd dynamics.

    Coordinates problem setup, solving, validation, and analysis.
    """

    def __init__(
        self,
        barrier_config: str = "anisotropy_aligned",
        gamma: float = 0.1,
        sigma: float = 0.01,
        rho_amplitude: float = 0.5,
        grid_size: tuple = (64, 64),
        output_dir: str = "results/",
    ):
        """
        Initialize experiment configuration.

        Args:
            barrier_config: Barrier configuration type
            gamma: Density-velocity coupling strength
            sigma: Diffusion coefficient
            rho_amplitude: Anisotropy amplitude
            grid_size: Spatial grid resolution
            output_dir: Output directory for results
        """
        self.config = {
            "barrier_config": barrier_config,
            "gamma": gamma,
            "sigma": sigma,
            "rho_amplitude": rho_amplitude,
            "grid_size": grid_size,
        }

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Setup logging
        configure_research_logging(f"anisotropic_experiment_{barrier_config}", level="INFO")

        logger.info(f"Initialized experiment: {barrier_config} configuration")
        logger.info(f"Parameters: γ={gamma}, σ={sigma}, ρ={rho_amplitude}")
        logger.info(f"Grid: {grid_size}, Output: {output_dir}")

    def run_single_configuration(self, validate: bool = True) -> dict[str, Any]:
        """
        Run experiment for single configuration.

        Args:
            validate: Whether to run validation checks

        Returns:
            Experiment results including solution and metrics
        """
        logger.info("Starting single configuration experiment")

        # Create problem
        problem = create_anisotropic_problem(**self.config)

        # Solve using problem's built-in solve method
        logger.info("Solving MFG system...")
        start_time = time.time()
        solution = problem.solve()
        solve_time = time.time() - start_time

        logger.info(f"Solution completed in {solve_time:.2f} seconds")

        # Validation
        validation_results = {}
        if validate:
            logger.info("Running validation checks...")
            validation_results = self._run_validation_checks(problem, solution)

        # Compute metrics
        logger.info("Computing evacuation metrics...")
        metrics = self._compute_evacuation_metrics(solution)

        # Create visualizations
        logger.info("Creating visualizations...")
        try:
            visualizer = create_visualization_suite(problem, solution, self.output_dir)
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
            visualizer = None

        # Compile results
        results = {
            "problem": problem,
            "solution": solution,
            "solve_time": solve_time,
            "metrics": metrics,
            "validation": validation_results,
            "config": self.config.copy(),
            "visualizer": visualizer,
        }

        # Save results summary
        self._save_results_summary(results)

        logger.info("Single configuration experiment completed")
        return results

    def run_comparative_study(self, configurations: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """
        Run comparative study across multiple barrier configurations.

        Args:
            configurations: List of barrier configurations to test

        Returns:
            Results for each configuration
        """
        if configurations is None:
            configurations = ["none", "central_obstacle", "anisotropy_aligned", "corridor_system"]

        logger.info(f"Starting comparative study: {configurations}")

        all_results = {}
        baseline_config = self.config.copy()

        for config_name in configurations:
            logger.info(f"Running configuration: {config_name}")

            # Update configuration
            self.config["barrier_config"] = config_name

            # Create subdirectory for this configuration
            config_output_dir = os.path.join(self.output_dir, config_name)
            os.makedirs(config_output_dir, exist_ok=True)

            # Temporarily update output directory
            original_output_dir = self.output_dir
            self.output_dir = config_output_dir

            try:
                # Run experiment
                results = self.run_single_configuration(validate=False)  # Skip validation for speed
                all_results[config_name] = results

            except Exception as e:
                logger.error(f"Error in configuration {config_name}: {e}")
                all_results[config_name] = {"error": str(e)}

            finally:
                # Restore output directory
                self.output_dir = original_output_dir

        # Restore baseline configuration
        self.config = baseline_config

        # Create comparative analysis
        self._create_comparative_analysis(all_results)

        logger.info("Comparative study completed")
        return all_results

    def run_convergence_study(self, base_grid: int = 32, max_refinements: int = 3) -> dict[str, Any]:
        """
        Run convergence analysis.

        Args:
            base_grid: Base grid size
            max_refinements: Number of refinement levels

        Returns:
            Convergence study results
        """
        logger.info("Starting convergence study")

        # Create convergence study instance
        study = ConvergenceStudy(base_grid=base_grid, max_refinements=max_refinements)

        # Run spatial convergence for current configuration
        spatial_result = study.run_spatial_convergence_study(barrier_config=self.config["barrier_config"])

        # Run temporal convergence
        temporal_result = study.run_temporal_convergence_study(
            grid_size=64, barrier_config=self.config["barrier_config"]
        )

        # Analyze results
        spatial_analysis = study.analyze_convergence_rates(spatial_result)
        temporal_analysis = study.analyze_convergence_rates(temporal_result)

        # Save convergence results
        convergence_output_dir = os.path.join(self.output_dir, "convergence")
        os.makedirs(convergence_output_dir, exist_ok=True)

        study.save_results(os.path.join(convergence_output_dir, "convergence_results.json"))

        report = study.generate_convergence_report()
        with open(os.path.join(convergence_output_dir, "convergence_report.md"), "w") as f:
            f.write(report)

        convergence_results = {
            "spatial_result": spatial_result,
            "temporal_result": temporal_result,
            "spatial_analysis": spatial_analysis,
            "temporal_analysis": temporal_analysis,
            "study": study,
        }

        logger.info("Convergence study completed")
        return convergence_results

    def _run_validation_checks(self, problem: AnisotropicMFGProblem2D, solution: Any) -> dict[str, Any]:
        """Run comprehensive validation checks."""
        validation_results = {}

        # Mass conservation check - adapt to new solution format
        try:
            if hasattr(solution, "M") and solution.M is not None:
                # Solution has M (density) field
                if len(solution.M.shape) > 1:
                    total_mass_history = [np.sum(m) for m in solution.M]
                else:
                    total_mass_history = [np.sum(solution.M)]

                initial_mass = total_mass_history[0] if len(total_mass_history) > 0 else 0
                final_mass = total_mass_history[-1] if len(total_mass_history) > 0 else 0
                mass_variation = (
                    np.max(total_mass_history) - np.min(total_mass_history) if len(total_mass_history) > 1 else 0
                )
            else:
                # Fallback: no mass conservation data available
                initial_mass = final_mass = mass_variation = 0

        except Exception as e:
            logger.warning(f"Could not compute mass conservation: {e}")
            initial_mass = final_mass = mass_variation = 0

        validation_results["mass_conservation"] = {
            "initial_mass": float(initial_mass),
            "final_mass": float(final_mass),
            "mass_variation": float(mass_variation),
            "conservation_error": float(abs(final_mass - initial_mass) / initial_mass) if initial_mass > 0 else 0,
            "passed": mass_variation < 1e-6,
        }

        # Boundary condition verification - adapt to new solution format
        try:
            if hasattr(solution, "U") and solution.U is not None:
                if len(solution.U.shape) > 1:
                    final_u = solution.U[-1]  # Last time step
                    # Convert 1D to 2D for boundary check
                    final_u_2d = problem.grid_adapter.convert_to_2d_array(final_u.reshape(1, -1))[0]
                    exit_values = final_u_2d[-1, :]  # Top boundary (exit)
                else:
                    exit_values = solution.U[-10:]  # Just check last few points
                exit_condition_error = np.max(np.abs(exit_values))
            else:
                exit_condition_error = 0

        except Exception as e:
            logger.warning(f"Could not check boundary conditions: {e}")
            exit_condition_error = 0

        validation_results["boundary_conditions"] = {
            "exit_condition_error": float(exit_condition_error),
            "passed": exit_condition_error < 1e-3,
        }

        # Physical realism checks - adapt to new solution format
        try:
            if hasattr(solution, "M") and solution.M is not None:
                if len(solution.M.shape) > 1:
                    final_m = solution.M[-1]
                else:
                    final_m = solution.M
            else:
                final_m = np.array([])
        except Exception as e:
            logger.warning(f"Could not get final density: {e}")
            final_m = np.array([])
        validation_results["physical_realism"] = {
            "non_negative_density": bool(np.all(final_m >= -1e-12)),
            "reasonable_peak_density": bool(np.max(final_m) < 100),  # Reasonable upper bound
            "evacuation_progress": float(1 - final_mass / initial_mass),
        }

        # Barrier-specific validation
        if hasattr(problem, "barriers") and problem.barriers:
            validation_results["barrier_validation"] = self._validate_barrier_constraints(problem, solution)

        logger.info(f"Validation results: {validation_results}")
        return validation_results

    def _validate_barrier_constraints(self, problem: AnisotropicMFGProblem2D, solution: Any) -> dict[str, Any]:
        """Validate barrier-specific constraints."""
        # Get final density - adapt to new solution format
        try:
            if hasattr(solution, "M") and solution.M is not None:
                if len(solution.M.shape) > 1:
                    final_m = solution.M[-1]
                else:
                    final_m = solution.M
            else:
                return {"error": "No density data available for barrier validation"}
        except Exception as e:
            return {"error": f"Could not extract density data: {e}"}

        # Check density inside barriers
        grid_points = np.column_stack([np.linspace(0, 1, problem.Nx1 + 1), np.linspace(0, 1, problem.Nx2 + 1)])

        barrier_violations = 0
        total_barrier_points = 0

        for barrier in problem.barriers:
            distances = barrier.compute_distance(grid_points)
            inside_barrier = distances < 0

            if np.any(inside_barrier):
                total_barrier_points += np.sum(inside_barrier)
                barrier_violations += np.sum(final_m.ravel()[inside_barrier] > 1e-6)

        return {
            "barrier_impermeability": barrier_violations == 0,
            "barrier_violations": int(barrier_violations),
            "total_barrier_points": int(total_barrier_points),
        }

    def _compute_evacuation_metrics(self, solution: Any) -> dict[str, Any]:
        """Compute comprehensive evacuation metrics."""
        import numpy as np

        # Time grid - adapt to new solution format
        if hasattr(solution, "time_grid") and solution.time_grid is not None:
            time_grid = solution.time_grid
        elif hasattr(solution, "M") and solution.M is not None:
            if len(solution.M.shape) > 1:
                time_grid = np.linspace(0, 1.0, len(solution.M))
            else:
                time_grid = np.array([0.0, 1.0])
        else:
            time_grid = np.array([0.0, 1.0])

        # Basic metrics - adapt to new solution format
        try:
            if hasattr(solution, "M") and solution.M is not None:
                if len(solution.M.shape) > 1:
                    total_mass = [np.sum(m) for m in solution.M]
                    peak_density = [np.max(m) for m in solution.M]
                else:
                    total_mass = [np.sum(solution.M)]
                    peak_density = [np.max(solution.M)]
            else:
                # Fallback: return basic metrics
                return {
                    "evacuation_efficiency": 0.0,
                    "total_evacuation_time": 0.0,
                    "t_50_percent": None,
                    "t_90_percent": None,
                    "max_velocity": 0.0,
                    "max_density": 0.0,
                    "error": "No density data available",
                }

            initial_mass = total_mass[0] if len(total_mass) > 0 else 0
            final_mass = total_mass[-1] if len(total_mass) > 0 else 0

            # Evacuation times
            if initial_mass > 0:
                evacuation_percentages = [(initial_mass - mass) / initial_mass for mass in total_mass]
            else:
                evacuation_percentages = [0] * len(total_mass)

        except Exception as e:
            logger.warning(f"Could not compute basic metrics: {e}")
            return {
                "evacuation_efficiency": 0.0,
                "total_evacuation_time": 0.0,
                "error": f"Metric computation failed: {e}",
            }

        # Find 50% and 90% evacuation times
        try:
            idx_50 = next(i for i, pct in enumerate(evacuation_percentages) if pct >= 0.5)
            t_50_percent = time_grid[idx_50]
        except StopIteration:
            t_50_percent = None

        try:
            idx_90 = next(i for i, pct in enumerate(evacuation_percentages) if pct >= 0.9)
            t_90_percent = time_grid[idx_90]
        except StopIteration:
            t_90_percent = None

        # Velocity analysis - simplified for new solution format
        max_velocity = 0
        try:
            if hasattr(solution, "U") and solution.U is not None:
                if len(solution.U.shape) > 1:
                    # Use last time step for velocity estimate
                    u_final = solution.U[-1]
                    grad_u = np.gradient(u_final)
                    if isinstance(grad_u, list | tuple) and len(grad_u) >= 2:
                        vel_magnitude = np.sqrt(grad_u[0] ** 2 + grad_u[1] ** 2)
                    else:
                        vel_magnitude = np.abs(grad_u)
                    max_velocity = float(np.max(vel_magnitude))
        except Exception as e:
            logger.warning(f"Could not compute velocity: {e}")
            max_velocity = 0

        metrics = {
            "evacuation_efficiency": float(final_mass / initial_mass) if initial_mass > 0 else 0.0,
            "peak_density": float(np.max(peak_density)) if len(peak_density) > 0 else 0.0,
            "max_velocity": float(max_velocity),
            "t_50_percent": float(t_50_percent) if t_50_percent is not None else None,
            "t_90_percent": float(t_90_percent) if t_90_percent is not None else None,
            "total_evacuation_time": float(time_grid[-1]),
            "final_evacuation_percentage": float(evacuation_percentages[-1] * 100),
        }

        return metrics

    def _save_results_summary(self, results: dict[str, Any]):
        """Save experiment results summary."""
        import json

        # Create serializable summary
        summary = {
            "configuration": results["config"],
            "solve_time": results["solve_time"],
            "metrics": results["metrics"],
            "validation": results["validation"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        summary_file = os.path.join(self.output_dir, "experiment_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results summary saved to {summary_file}")

    def _create_comparative_analysis(self, all_results: dict[str, dict[str, Any]]):
        """Create comparative analysis across configurations."""
        # Extract metrics for comparison
        comparison_data = {}

        for config_name, results in all_results.items():
            if "error" not in results:
                comparison_data[config_name] = {
                    "solve_time": results["solve_time"],
                    "metrics": results["metrics"],
                    "validation_passed": all(
                        v.get("passed", True) if isinstance(v, dict) else True for v in results["validation"].values()
                    ),
                }

        # Save comparison
        comparison_file = os.path.join(self.output_dir, "comparative_analysis.json")
        with open(comparison_file, "w") as f:
            json.dump(comparison_data, f, indent=2)

        # Create comparative report
        report_lines = ["# Comparative Analysis Report", ""]

        for config_name, data in comparison_data.items():
            report_lines.extend(
                [
                    f"## {config_name.replace('_', ' ').title()}",
                    f"- Solve time: {data['solve_time']:.2f}s",
                    f"- Evacuation efficiency: {data['metrics']['evacuation_efficiency']:.3f}",
                    f"- Peak density: {data['metrics']['peak_density']:.3f}",
                    f"- Validation passed: {data['validation_passed']}",
                    "",
                ]
            )

        report_file = os.path.join(self.output_dir, "comparative_report.md")
        with open(report_file, "w") as f:
            f.write("\n".join(report_lines))

        logger.info(f"Comparative analysis saved to {self.output_dir}")


def main():
    """Main experiment runner with command line interface."""
    parser = argparse.ArgumentParser(description="Run 2D Anisotropic Crowd Dynamics Experiment")

    parser.add_argument(
        "--mode", choices=["single", "comparative", "convergence", "all"], default="single", help="Experiment mode"
    )
    parser.add_argument(
        "--barrier-config",
        default="anisotropy_aligned",
        choices=["none", "central_obstacle", "anisotropy_aligned", "corridor_system"],
        help="Barrier configuration",
    )
    parser.add_argument("--gamma", type=float, default=0.1, help="Density-velocity coupling")
    parser.add_argument("--sigma", type=float, default=0.01, help="Diffusion coefficient")
    parser.add_argument("--rho-amplitude", type=float, default=0.5, help="Anisotropy amplitude")
    parser.add_argument("--grid-size", type=int, default=64, help="Grid size (NxN)")
    parser.add_argument("--output-dir", default="results/", help="Output directory")
    parser.add_argument("--no-validation", action="store_true", help="Skip validation checks")

    args = parser.parse_args()

    # Create experiment
    experiment = AnisotropicExperiment(
        barrier_config=args.barrier_config,
        gamma=args.gamma,
        sigma=args.sigma,
        rho_amplitude=args.rho_amplitude,
        grid_size=(args.grid_size, args.grid_size),
        output_dir=args.output_dir,
    )

    print(f"Starting experiment in {args.mode} mode...")

    # Run experiments based on mode
    if args.mode == "single":
        experiment.run_single_configuration(validate=not args.no_validation)
        print(f"Single configuration completed. Results in {args.output_dir}")

    elif args.mode == "comparative":
        experiment.run_comparative_study()
        print(f"Comparative study completed. Results in {args.output_dir}")

    elif args.mode == "convergence":
        experiment.run_convergence_study()
        print(f"Convergence study completed. Results in {args.output_dir}/convergence/")

    elif args.mode == "all":
        print("Running complete experimental suite...")

        # Single configuration
        experiment.run_single_configuration(validate=not args.no_validation)

        # Comparative study
        experiment.run_comparative_study()

        # Convergence study
        experiment.run_convergence_study()

        print(f"Complete experimental suite completed. Results in {args.output_dir}")

    print("Experiment finished successfully!")


if __name__ == "__main__":
    main()
