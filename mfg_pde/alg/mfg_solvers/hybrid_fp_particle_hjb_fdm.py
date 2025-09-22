#!/usr/bin/env python3
"""
Hybrid FP-Particle + HJB-FDM Solver for Mean Field Games

This module implements a hybrid solver that combines:
- Fokker-Planck equation: Particle-based solution
- Hamilton-Jacobi-Bellman equation: Finite Difference Method (FDM)

This hybrid approach leverages the strengths of both methods:
- Particle methods handle complex geometries and mass conservation naturally
- FDM provides stable and accurate solution of the HJB equation
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ...config.solver_config import MFGSolverConfig
from ...core.mfg_problem import MFGProblem
from ...utils.logging import get_logger
from ..fp_solvers.fp_particle import FPParticleSolver
from ..hjb_solvers.hjb_fdm import HJBFDMSolver
from .damped_fixed_point_iterator import FixedPointIterator

logger = get_logger(__name__)


class HybridFPParticleHJBFDM:
    """
    Hybrid solver combining Particle FP and FDM HJB methods.

    This solver implements the specific combination requested:
    - Fokker-Planck equation solved using particle methods
    - HJB equation solved using finite difference methods
    - Coupled through fixed point iteration with damping
    """

    def __init__(
        self,
        mfg_problem: MFGProblem,
        num_particles: int = 5000,
        kde_bandwidth: str | float = "scott",
        hjb_newton_iterations: int = 30,
        hjb_newton_tolerance: float = 1e-7,
        **kwargs
    ):
        """
        Initialize the hybrid FP-Particle + HJB-FDM solver.

        Args:
            mfg_problem: MFG problem to solve
            num_particles: Number of particles for FP solver
            kde_bandwidth: Bandwidth for kernel density estimation
            hjb_newton_iterations: Newton iterations for HJB solver
            hjb_newton_tolerance: Newton tolerance for HJB solver
            **kwargs: Additional parameters
        """
        self.mfg_problem = mfg_problem
        self.num_particles = num_particles
        self.kde_bandwidth = kde_bandwidth
        self.hjb_newton_iterations = hjb_newton_iterations
        self.hjb_newton_tolerance = hjb_newton_tolerance

        # Initialize component solvers
        self._setup_component_solvers()

        # Solver state
        self.convergence_history = []
        self.timing_info = {}

        logger.info("Initialized Hybrid FP-Particle + HJB-FDM solver")
        logger.info(f"  Particles: {num_particles}")
        logger.info(f"  KDE bandwidth: {kde_bandwidth}")
        logger.info(f"  HJB Newton iterations: {hjb_newton_iterations}")

    def _setup_component_solvers(self):
        """Initialize the component solvers for FP and HJB."""
        # HJB solver using finite differences
        self.hjb_solver = HJBFDMSolver(
            self.mfg_problem,
            NiterNewton=self.hjb_newton_iterations,
            l2errBoundNewton=self.hjb_newton_tolerance
        )

        # FP solver using particles
        self.fp_solver = FPParticleSolver(
            self.mfg_problem,
            num_particles=self.num_particles,
            kde_bandwidth=self.kde_bandwidth
        )

        logger.debug("Component solvers initialized successfully")

    def solve(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-4,
        damping_factor: float = 0.5,
        config: MFGSolverConfig | None = None,
        return_structured: bool = True,
        **kwargs
    ) -> dict[str, Any]:
        """
        Solve the MFG problem using hybrid FP-Particle + HJB-FDM approach.

        Args:
            max_iterations: Maximum Picard iterations
            tolerance: Convergence tolerance
            damping_factor: Damping factor for fixed point iteration
            config: Optional solver configuration
            return_structured: Whether to return structured results
            **kwargs: Additional solver parameters

        Returns:
            Dictionary containing solution and convergence information
        """
        logger.info("Starting Hybrid FP-Particle + HJB-FDM solve")
        logger.info(f"Parameters: max_iter={max_iterations}, tol={tolerance:.2e}, damping={damping_factor}")

        start_time = time.time()

        # Create fixed point iterator with hybrid solvers
        fixed_point_solver = FixedPointIterator(
            self.mfg_problem,
            hjb_solver=self.hjb_solver,
            fp_solver=self.fp_solver,
            thetaUM=damping_factor
        )

        # Solve using fixed point iteration
        try:
            result = fixed_point_solver.solve(
                max_iterations=max_iterations,
                tolerance=tolerance,
                return_structured=return_structured,
                **kwargs
            )
            solve_time = time.time() - start_time

            # Extract convergence information based on result type
            if isinstance(result, dict):
                converged = result.get('success', False)
                iterations = result.get('iterations', 0)
                final_residual = result.get('final_residual', float('inf'))
            elif isinstance(result, tuple) and len(result) >= 3:
                # Traditional tuple format: (U, M, iterations, l2distu_abs, l2distm_abs)
                U, M, iterations = result[:3]
                converged = True  # Assume convergence if no error
                final_residual = result[3][-1] if len(result) > 3 and len(result[3]) > 0 else 0.0
                # Convert to dict format for consistency
                result = {
                    'U': U, 'M': M, 'iterations': iterations,
                    'success': True, 'converged': True,
                    'final_residual': final_residual
                }
            else:
                # Unknown format
                converged = False
                iterations = 0
                final_residual = float('inf')

            # Store timing and convergence info
            self.timing_info = {
                'total_solve_time': solve_time,
                'iterations': iterations,
                'time_per_iteration': solve_time / max(iterations, 1)
            }

            # Enhanced result dictionary
            enhanced_result = {
                'success': converged,
                'converged': converged,
                'iterations': iterations,
                'final_residual': final_residual,
                'solve_time': solve_time,
                'solver_type': 'hybrid_fp_particle_hjb_fdm',
                'num_particles': self.num_particles,
                'kde_bandwidth': self.kde_bandwidth,
                'hjb_method': 'fdm',
                'fp_method': 'particle',
                'damping_factor': damping_factor,
                'timing_info': self.timing_info
            }

            # Include original result data
            enhanced_result.update(result)

            if converged:
                logger.info(f"✅ Hybrid solver converged in {iterations} iterations ({solve_time:.2f}s)")
                logger.info(f"   Final residual: {final_residual:.2e}")
                logger.info(f"   Average time per iteration: {solve_time/max(iterations,1):.2f}s")
            else:
                logger.warning(f"⚠️ Hybrid solver did not converge after {iterations} iterations")
                logger.warning(f"   Final residual: {final_residual:.2e}")

            return enhanced_result

        except Exception as e:
            solve_time = time.time() - start_time
            logger.error(f"❌ Hybrid solver failed after {solve_time:.2f}s: {e}")

            return {
                'success': False,
                'converged': False,
                'error': str(e),
                'solver_type': 'hybrid_fp_particle_hjb_fdm',
                'solve_time': solve_time,
                'iterations': 0,
                'final_residual': float('inf')
            }

    def get_solver_info(self) -> dict[str, Any]:
        """Get information about the hybrid solver configuration."""
        return {
            'solver_name': 'Hybrid FP-Particle + HJB-FDM',
            'solver_type': 'hybrid_fp_particle_hjb_fdm',
            'fp_method': 'particle',
            'hjb_method': 'fdm',
            'num_particles': self.num_particles,
            'kde_bandwidth': self.kde_bandwidth,
            'hjb_newton_iterations': self.hjb_newton_iterations,
            'hjb_newton_tolerance': self.hjb_newton_tolerance,
            'description': 'Hybrid solver combining particle-based FP with FDM-based HJB',
            'advantages': [
                'Natural mass conservation from particles',
                'Stable HJB solution with FDM',
                'Good performance on complex geometries',
                'Flexible particle distribution'
            ],
            'use_cases': [
                'Complex domain geometries',
                'Problems requiring mass conservation',
                'Medium to large scale problems',
                'Research applications'
            ]
        }

    def update_particle_count(self, num_particles: int):
        """
        Update the number of particles and reinitialize FP solver.

        Args:
            num_particles: New number of particles
        """
        logger.info(f"Updating particle count from {self.num_particles} to {num_particles}")
        self.num_particles = num_particles

        # Reinitialize FP solver with new particle count
        self.fp_solver = FPParticleSolver(
            self.mfg_problem,
            num_particles=self.num_particles,
            kde_bandwidth=self.kde_bandwidth
        )

    def update_hjb_newton_config(self, max_iterations: int, tolerance: float):
        """
        Update HJB Newton solver configuration.

        Args:
            max_iterations: New maximum Newton iterations
            tolerance: New Newton tolerance
        """
        logger.info(f"Updating HJB Newton config: iter={max_iterations}, tol={tolerance:.2e}")
        self.hjb_newton_iterations = max_iterations
        self.hjb_newton_tolerance = tolerance

        # Reinitialize HJB solver with new configuration
        self.hjb_solver = HJBFDMSolver(
            self.mfg_problem,
            NiterNewton=self.hjb_newton_iterations,
            l2errBoundNewton=self.hjb_newton_tolerance
        )

    def benchmark_particle_scaling(
        self,
        particle_counts: list[int] | None = None,
        max_iterations: int = 20,
        tolerance: float = 1e-3
    ) -> dict[str, Any]:
        """
        Benchmark solver performance across different particle counts.

        Args:
            particle_counts: List of particle counts to test
            max_iterations: Maximum iterations for each test
            tolerance: Convergence tolerance

        Returns:
            Dictionary with benchmarking results
        """
        if particle_counts is None:
            particle_counts = [1000, 2000, 5000, 10000]

        logger.info("Starting particle scaling benchmark")

        original_particles = self.num_particles
        results = []

        for count in particle_counts:
            logger.info(f"Testing with {count} particles...")

            try:
                # Update particle count
                self.update_particle_count(count)

                # Solve with current configuration
                start_time = time.time()
                result = self.solve(
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    return_structured=False  # Faster for benchmarking
                )
                solve_time = time.time() - start_time

                # Store results
                benchmark_result = {
                    'num_particles': count,
                    'converged': result.get('converged', False),
                    'iterations': result.get('iterations', 0),
                    'solve_time': solve_time,
                    'final_residual': result.get('final_residual', float('inf')),
                    'particles_per_second': count / solve_time if solve_time > 0 else 0
                }

                results.append(benchmark_result)
                logger.info(f"  Result: {'✅' if benchmark_result['converged'] else '❌'} "
                          f"{benchmark_result['iterations']} iter, {solve_time:.2f}s")

            except Exception as e:
                logger.error(f"  Benchmark failed for {count} particles: {e}")
                results.append({
                    'num_particles': count,
                    'converged': False,
                    'error': str(e),
                    'solve_time': float('inf')
                })

        # Restore original configuration
        self.update_particle_count(original_particles)

        benchmark_summary = {
            'results': results,
            'optimal_particle_count': self._find_optimal_particle_count(results),
            'scaling_analysis': self._analyze_particle_scaling(results)
        }

        logger.info("Particle scaling benchmark completed")
        return benchmark_summary

    def _find_optimal_particle_count(self, results: list) -> int | None:
        """Find optimal particle count based on convergence and performance."""
        converged_results = [r for r in results if r.get('converged', False)]

        if not converged_results:
            return None

        # Find best trade-off between speed and particle count
        # Prefer fewer particles if convergence is similar
        best_result = min(converged_results,
                         key=lambda r: (r['solve_time'], r['num_particles']))

        return best_result['num_particles']

    def _analyze_particle_scaling(self, results: list) -> dict[str, Any]:
        """Analyze scaling behavior of particle count vs performance."""
        converged_results = [r for r in results if r.get('converged', False)]

        if len(converged_results) < 2:
            return {'scaling_coefficient': None, 'efficiency_trend': 'insufficient_data'}

        # Analyze time scaling with particle count
        particle_counts = [r['num_particles'] for r in converged_results]
        solve_times = [r['solve_time'] for r in converged_results]

        if len(particle_counts) > 2:
            # Fit power law: time ~ particles^alpha
            log_particles = np.log(particle_counts)
            log_times = np.log(solve_times)
            scaling_coeff = np.polyfit(log_particles, log_times, 1)[0]
        else:
            scaling_coeff = None

        return {
            'scaling_coefficient': scaling_coeff,
            'efficiency_trend': 'linear' if scaling_coeff and scaling_coeff < 1.2 else 'super_linear',
            'min_particles_converged': min(particle_counts),
            'max_particles_tested': max(particle_counts),
            'convergence_rate': len(converged_results) / len(results)
        }


# Factory function for easy creation
def create_hybrid_fp_particle_hjb_fdm_solver(
    mfg_problem: MFGProblem,
    num_particles: int = 5000,
    kde_bandwidth: str | float = "scott",
    hjb_newton_iterations: int = 30,
    hjb_newton_tolerance: float = 1e-7,
    **kwargs
) -> HybridFPParticleHJBFDM:
    """
    Factory function to create Hybrid FP-Particle + HJB-FDM solver.

    Args:
        mfg_problem: MFG problem to solve
        num_particles: Number of particles for FP solver
        kde_bandwidth: Bandwidth for kernel density estimation
        hjb_newton_iterations: Newton iterations for HJB solver
        hjb_newton_tolerance: Newton tolerance for HJB solver
        **kwargs: Additional parameters

    Returns:
        Configured HybridFPParticleHJBFDM solver instance
    """
    return HybridFPParticleHJBFDM(
        mfg_problem=mfg_problem,
        num_particles=num_particles,
        kde_bandwidth=kde_bandwidth,
        hjb_newton_iterations=hjb_newton_iterations,
        hjb_newton_tolerance=hjb_newton_tolerance,
        **kwargs
    )


# Configuration presets
class HybridSolverPresets:
    """Predefined configurations for the hybrid solver."""

    @staticmethod
    def fast_hybrid(mfg_problem: MFGProblem) -> HybridFPParticleHJBFDM:
        """Fast configuration optimized for speed."""
        return create_hybrid_fp_particle_hjb_fdm_solver(
            mfg_problem,
            num_particles=2000,
            kde_bandwidth=0.05,
            hjb_newton_iterations=15,
            hjb_newton_tolerance=1e-5
        )

    @staticmethod
    def accurate_hybrid(mfg_problem: MFGProblem) -> HybridFPParticleHJBFDM:
        """Accurate configuration optimized for precision."""
        return create_hybrid_fp_particle_hjb_fdm_solver(
            mfg_problem,
            num_particles=10000,
            kde_bandwidth="scott",
            hjb_newton_iterations=50,
            hjb_newton_tolerance=1e-8
        )

    @staticmethod
    def research_hybrid(mfg_problem: MFGProblem) -> HybridFPParticleHJBFDM:
        """Research configuration with comprehensive monitoring."""
        return create_hybrid_fp_particle_hjb_fdm_solver(
            mfg_problem,
            num_particles=15000,
            kde_bandwidth="scott",
            hjb_newton_iterations=100,
            hjb_newton_tolerance=1e-10
        )


# Usage examples in docstring
"""
Usage Examples:

# Basic usage
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.alg.mfg_solvers.hybrid_fp_particle_hjb_fdm import create_hybrid_fp_particle_hjb_fdm_solver

problem = ExampleMFGProblem(Nx=50, Nt=50)
hybrid_solver = create_hybrid_fp_particle_hjb_fdm_solver(problem, num_particles=5000)

result = hybrid_solver.solve(max_iterations=50, tolerance=1e-4)
print(f"Converged: {result['converged']} in {result['iterations']} iterations")

# Using presets
from mfg_pde.alg.mfg_solvers.hybrid_fp_particle_hjb_fdm import HybridSolverPresets

fast_solver = HybridSolverPresets.fast_hybrid(problem)
accurate_solver = HybridSolverPresets.accurate_hybrid(problem)

# Benchmarking particle scaling
benchmark_results = hybrid_solver.benchmark_particle_scaling([1000, 5000, 10000])
optimal_particles = benchmark_results['optimal_particle_count']
"""
