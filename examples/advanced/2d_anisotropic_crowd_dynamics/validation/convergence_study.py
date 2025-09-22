"""
Convergence Study for 2D Anisotropic Crowd Dynamics

This module implements comprehensive convergence analysis including
grid refinement studies, temporal convergence, and barrier-specific validation.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json

from mfg_pde.utils.logging import get_logger, configure_research_logging

# Local imports
import sys
sys.path.append('..')
from anisotropic_2d_problem import AnisotropicMFGProblem2D, create_anisotropic_problem
from solver_config import create_experiment_solver, get_convergence_study_configs

logger = get_logger(__name__)


@dataclass
class ConvergenceResult:
    """Container for convergence analysis results."""
    grid_sizes: List[int]
    errors: List[float]
    convergence_rates: List[float]
    solver_times: List[float]
    memory_usage: List[float]
    problem_config: Dict[str, Any]


class ConvergenceStudy:
    """
    Comprehensive convergence analysis for anisotropic MFG problems.

    Performs systematic grid refinement studies to validate
    numerical accuracy and stability.
    """

    def __init__(self, base_grid: int = 32, max_refinements: int = 3):
        """
        Initialize convergence study.

        Args:
            base_grid: Starting grid size
            max_refinements: Number of refinement levels
        """
        self.base_grid = base_grid
        self.max_refinements = max_refinements
        self.results = {}

        configure_research_logging("convergence_study", level="INFO")
        logger.info(f"Initialized convergence study: base_grid={base_grid}, refinements={max_refinements}")

    def run_spatial_convergence_study(
        self,
        barrier_config: str = 'anisotropy_aligned',
        reference_grid: Optional[int] = None
    ) -> ConvergenceResult:
        """
        Run spatial convergence analysis with grid refinement.

        Args:
            barrier_config: Barrier configuration to test
            reference_grid: Grid size for reference solution (None for finest grid)

        Returns:
            Convergence analysis results
        """
        logger.info(f"Starting spatial convergence study for {barrier_config}")

        # Generate grid sizes
        grid_sizes = [self.base_grid * (2**i) for i in range(self.max_refinements)]

        if reference_grid is None:
            reference_grid = grid_sizes[-1] * 2  # One level finer than finest test grid

        logger.info(f"Grid sizes: {grid_sizes}, reference: {reference_grid}")

        # Compute reference solution
        logger.info("Computing reference solution...")
        reference_solution = self._solve_problem(barrier_config, reference_grid)

        # Compute solutions on test grids
        solutions = []
        solver_times = []
        memory_usage = []

        for grid_size in grid_sizes:
            logger.info(f"Solving on {grid_size}x{grid_size} grid...")

            start_time = time.time()
            solution = self._solve_problem(barrier_config, grid_size)
            solve_time = time.time() - start_time

            solutions.append(solution)
            solver_times.append(solve_time)
            memory_usage.append(self._estimate_memory_usage(grid_size))

            logger.info(f"Grid {grid_size}: solved in {solve_time:.2f}s")

        # Compute errors and convergence rates
        errors = []
        for i, (grid_size, solution) in enumerate(zip(grid_sizes, solutions)):
            error = self._compute_solution_error(solution, reference_solution, grid_size, reference_grid)
            errors.append(error)
            logger.info(f"Grid {grid_size}: L2 error = {error:.6e}")

        # Estimate convergence rates
        convergence_rates = []
        for i in range(1, len(errors)):
            h_ratio = grid_sizes[i-1] / grid_sizes[i]  # Ratio of grid spacings
            error_ratio = errors[i-1] / errors[i]
            rate = np.log(error_ratio) / np.log(h_ratio)
            convergence_rates.append(rate)
            logger.info(f"Convergence rate {grid_sizes[i-1]} -> {grid_sizes[i]}: {rate:.2f}")

        # Create result object
        result = ConvergenceResult(
            grid_sizes=grid_sizes,
            errors=errors,
            convergence_rates=convergence_rates,
            solver_times=solver_times,
            memory_usage=memory_usage,
            problem_config={
                'barrier_config': barrier_config,
                'reference_grid': reference_grid,
                'study_type': 'spatial_convergence'
            }
        )

        self.results[f'spatial_{barrier_config}'] = result
        return result

    def run_temporal_convergence_study(
        self,
        grid_size: int = 64,
        barrier_config: str = 'anisotropy_aligned'
    ) -> ConvergenceResult:
        """
        Run temporal convergence analysis with time step refinement.

        Args:
            grid_size: Fixed spatial grid size
            barrier_config: Barrier configuration to test

        Returns:
            Temporal convergence results
        """
        logger.info(f"Starting temporal convergence study on {grid_size}x{grid_size} grid")

        # Generate time step sizes
        base_dt = 0.01
        dt_values = [base_dt / (2**i) for i in range(self.max_refinements)]
        reference_dt = dt_values[-1] / 4  # Much finer reference

        logger.info(f"Time steps: {dt_values}, reference: {reference_dt}")

        # Compute reference solution
        logger.info("Computing temporal reference solution...")
        reference_solution = self._solve_problem(barrier_config, grid_size, dt=reference_dt)

        # Compute solutions with different time steps
        solutions = []
        solver_times = []

        for dt in dt_values:
            logger.info(f"Solving with dt = {dt:.6f}...")

            start_time = time.time()
            solution = self._solve_problem(barrier_config, grid_size, dt=dt)
            solve_time = time.time() - start_time

            solutions.append(solution)
            solver_times.append(solve_time)

            logger.info(f"dt = {dt:.6f}: solved in {solve_time:.2f}s")

        # Compute temporal errors
        errors = []
        for i, (dt, solution) in enumerate(zip(dt_values, solutions)):
            error = self._compute_temporal_error(solution, reference_solution)
            errors.append(error)
            logger.info(f"dt = {dt:.6f}: temporal error = {error:.6e}")

        # Estimate temporal convergence rates
        convergence_rates = []
        for i in range(1, len(errors)):
            dt_ratio = dt_values[i-1] / dt_values[i]
            error_ratio = errors[i-1] / errors[i]
            rate = np.log(error_ratio) / np.log(dt_ratio)
            convergence_rates.append(rate)
            logger.info(f"Temporal rate {dt_values[i-1]:.6f} -> {dt_values[i]:.6f}: {rate:.2f}")

        # Create result object
        result = ConvergenceResult(
            grid_sizes=[1/dt for dt in dt_values],  # Use 1/dt as "grid size" for temporal
            errors=errors,
            convergence_rates=convergence_rates,
            solver_times=solver_times,
            memory_usage=[self._estimate_memory_usage(grid_size)] * len(dt_values),
            problem_config={
                'barrier_config': barrier_config,
                'grid_size': grid_size,
                'dt_values': dt_values,
                'study_type': 'temporal_convergence'
            }
        )

        self.results[f'temporal_{barrier_config}'] = result
        return result

    def run_barrier_convergence_study(self) -> Dict[str, ConvergenceResult]:
        """
        Compare convergence across different barrier configurations.

        Returns:
            Dictionary of convergence results for each barrier type
        """
        logger.info("Starting barrier convergence comparison study")

        barrier_configs = ['none', 'central_obstacle', 'anisotropy_aligned', 'corridor_system']
        results = {}

        for config in barrier_configs:
            logger.info(f"Running convergence study for barrier config: {config}")
            result = self.run_spatial_convergence_study(barrier_config=config)
            results[config] = result

        self.results.update({f'barrier_study_{k}': v for k, v in results.items()})
        return results

    def _solve_problem(
        self,
        barrier_config: str,
        grid_size: int,
        dt: Optional[float] = None
    ) -> Any:
        """
        Solve MFG problem with specified configuration.

        Args:
            barrier_config: Barrier configuration
            grid_size: Spatial grid resolution
            dt: Time step size (optional)

        Returns:
            Solution object
        """
        # Create problem
        problem = create_anisotropic_problem(
            barrier_config=barrier_config,
            gamma=0.1,
            sigma=0.01,
            rho_amplitude=0.5
        )
        problem.grid_size = (grid_size, grid_size)

        # Create solver with appropriate configuration
        solver = create_experiment_solver(problem)

        # Override time step if specified
        if dt is not None:
            solver.config['fixed_time_step'] = dt
            solver.config['adaptive_time_stepping'] = False

        # Solve
        solution = solver.solve(problem)

        return solution

    def _compute_solution_error(
        self,
        solution: Any,
        reference_solution: Any,
        grid_size: int,
        reference_grid: int
    ) -> float:
        """
        Compute L2 error between solution and reference.

        Args:
            solution: Test solution
            reference_solution: Reference solution
            grid_size: Test grid size
            reference_grid: Reference grid size

        Returns:
            L2 error norm
        """
        # Extract final density and value function
        m_test = solution.density_history[-1]
        u_test = solution.value_history[-1]

        m_ref = reference_solution.density_history[-1]
        u_ref = reference_solution.value_history[-1]

        # Interpolate reference to test grid if needed
        if reference_grid != grid_size:
            m_ref_interp = self._interpolate_solution(m_ref, reference_grid, grid_size)
            u_ref_interp = self._interpolate_solution(u_ref, reference_grid, grid_size)
        else:
            m_ref_interp = m_ref
            u_ref_interp = u_ref

        # Compute L2 errors
        h = 1.0 / grid_size  # Grid spacing
        error_m = np.sqrt(h**2 * np.sum((m_test - m_ref_interp)**2))
        error_u = np.sqrt(h**2 * np.sum((u_test - u_ref_interp)**2))

        # Combined error
        total_error = np.sqrt(error_m**2 + error_u**2)

        return total_error

    def _compute_temporal_error(self, solution: Any, reference_solution: Any) -> float:
        """Compute temporal error between solutions."""
        # Compare final states
        m_test = solution.density_history[-1]
        u_test = solution.value_history[-1]

        m_ref = reference_solution.density_history[-1]
        u_ref = reference_solution.value_history[-1]

        # L2 errors
        error_m = np.linalg.norm(m_test - m_ref)
        error_u = np.linalg.norm(u_test - u_ref)

        return np.sqrt(error_m**2 + error_u**2)

    def _interpolate_solution(
        self,
        solution: np.ndarray,
        from_grid: int,
        to_grid: int
    ) -> np.ndarray:
        """
        Interpolate solution between different grid sizes.

        Args:
            solution: Solution array
            from_grid: Source grid size
            to_grid: Target grid size

        Returns:
            Interpolated solution
        """
        from scipy.interpolate import RegularGridInterpolator

        # Create coordinate grids
        x_from = np.linspace(0, 1, from_grid)
        y_from = np.linspace(0, 1, from_grid)

        x_to = np.linspace(0, 1, to_grid)
        y_to = np.linspace(0, 1, to_grid)

        # Create interpolator
        interpolator = RegularGridInterpolator(
            (x_from, y_from),
            solution,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        # Create target grid
        X_to, Y_to = np.meshgrid(x_to, y_to)
        points = np.column_stack([X_to.ravel(), Y_to.ravel()])

        # Interpolate
        solution_interp = interpolator(points).reshape(to_grid, to_grid)

        return solution_interp

    def _estimate_memory_usage(self, grid_size: int) -> float:
        """
        Estimate memory usage for given grid size.

        Args:
            grid_size: Grid resolution

        Returns:
            Estimated memory in MB
        """
        # Rough estimate: 2 arrays (u, m) × 8 bytes × grid points × history length
        arrays_per_timestep = 2
        bytes_per_float = 8
        typical_timesteps = 100

        memory_bytes = arrays_per_timestep * bytes_per_float * grid_size**2 * typical_timesteps
        memory_mb = memory_bytes / (1024**2)

        return memory_mb

    def analyze_convergence_rates(self, result: ConvergenceResult) -> Dict[str, Any]:
        """
        Analyze convergence rates and provide theoretical comparison.

        Args:
            result: Convergence result to analyze

        Returns:
            Analysis summary
        """
        rates = result.convergence_rates

        analysis = {
            'mean_rate': np.mean(rates),
            'std_rate': np.std(rates),
            'min_rate': np.min(rates),
            'max_rate': np.max(rates),
            'theoretical_rate': 2.0,  # Expected for second-order methods
            'rate_efficiency': np.mean(rates) / 2.0,  # Efficiency relative to theoretical
            'convergence_achieved': np.mean(rates) > 1.5  # Reasonable convergence threshold
        }

        logger.info(f"Convergence analysis: mean_rate={analysis['mean_rate']:.2f}, "
                   f"efficiency={analysis['rate_efficiency']:.2f}")

        return analysis

    def save_results(self, filename: str = "convergence_results.json"):
        """Save convergence study results to file."""
        # Convert results to serializable format
        serializable_results = {}

        for key, result in self.results.items():
            serializable_results[key] = {
                'grid_sizes': result.grid_sizes,
                'errors': result.errors,
                'convergence_rates': result.convergence_rates,
                'solver_times': result.solver_times,
                'memory_usage': result.memory_usage,
                'problem_config': result.problem_config
            }

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Convergence results saved to {filename}")

    def generate_convergence_report(self) -> str:
        """Generate summary report of convergence studies."""
        report_lines = ["# Convergence Study Report", ""]

        for key, result in self.results.items():
            report_lines.extend([
                f"## {key.replace('_', ' ').title()}",
                f"- Grid sizes: {result.grid_sizes}",
                f"- Final error: {result.errors[-1]:.2e}",
                f"- Mean convergence rate: {np.mean(result.convergence_rates):.2f}",
                f"- Total solver time: {sum(result.solver_times):.1f}s",
                ""
            ])

        return "\n".join(report_lines)


def run_full_convergence_study():
    """Run comprehensive convergence analysis."""
    logger.info("Starting full convergence study")

    study = ConvergenceStudy(base_grid=32, max_refinements=3)

    # Spatial convergence for different barrier configurations
    spatial_results = study.run_barrier_convergence_study()

    # Temporal convergence
    temporal_result = study.run_temporal_convergence_study()

    # Analyze results
    for config, result in spatial_results.items():
        analysis = study.analyze_convergence_rates(result)
        logger.info(f"Analysis for {config}: {analysis}")

    # Save results
    study.save_results("../results/convergence_results.json")

    # Generate report
    report = study.generate_convergence_report()
    with open("../results/convergence_report.md", 'w') as f:
        f.write(report)

    logger.info("Full convergence study completed")
    return study


if __name__ == "__main__":
    study = run_full_convergence_study()
    print("Convergence study completed. Check results/ directory for output.")