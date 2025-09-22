"""
Solver Configuration for 2D Anisotropic Crowd Dynamics

This module provides optimized solver configurations for the anisotropic
crowd dynamics experiment, including special handling for barriers and
cross-coupling terms.
"""

import numpy as np
from typing import Dict, Any, Optional

from mfg_pde.factory import create_fast_solver
from mfg_pde.config import create_fast_config
from mfg_pde.utils.logging import get_logger

logger = get_logger(__name__)


def create_anisotropic_solver_config(
    has_barriers: bool = False,
    grid_size: tuple = (64, 64),
    cfl_safety_factor: Optional[float] = None,
    solver_type: str = 'semi_lagrangian'
) -> Dict[str, Any]:
    """
    Create optimized solver configuration for anisotropic experiment.

    Args:
        has_barriers: Whether the problem includes barriers
        grid_size: Spatial grid resolution
        cfl_safety_factor: Safety factor for CFL condition
        solver_type: Type of solver to use

    Returns:
        Solver configuration dictionary
    """
    # Base configuration for non-separable 2D problem
    config_obj = create_fast_config()

    # Convert to dictionary for backwards compatibility
    config = {
        'solver_type': solver_type,
        'spatial_scheme': 'central_difference',
        'temporal_scheme': 'semi_implicit',
        'grid_refinement': 'adaptive' if has_barriers else 'uniform'
    }

    # Adjust CFL safety factor for cross-coupling stability
    if cfl_safety_factor is None:
        cfl_safety_factor = 0.2 if has_barriers else 0.3

    # Enhanced configuration for cross-coupling and barriers
    enhanced_config = {
        'cfl_safety_factor': cfl_safety_factor,
        'nonlinear_tolerance': 1e-6,
        'max_iterations': 1000,
        'diffusion_implicit': True,  # Implicit diffusion for stability
        'cross_coupling_treatment': 'explicit',  # Explicit for ρ(x) ∂u/∂x₁ ∂u/∂x₂
        'barrier_method': 'penalty' if has_barriers else None,
        'barrier_penalty_strength': 1e6 if has_barriers else None,
        'barrier_smoothing_parameter': 0.02 if has_barriers else None,
    }

    # Grid-specific adjustments
    N = max(grid_size)
    if N >= 128:
        enhanced_config.update({
            'matrix_solver': 'sparse_direct',
            'preconditioner': 'ilu',
            'memory_optimization': True
        })
    elif N >= 64:
        enhanced_config.update({
            'matrix_solver': 'iterative',
            'preconditioner': 'jacobi'
        })

    # Merge configurations
    config.update(enhanced_config)

    logger.info(f"Created solver config: {solver_type}, grid={grid_size}, barriers={has_barriers}")
    logger.info(f"CFL safety factor: {cfl_safety_factor}")

    return config


def create_experiment_solver(
    problem,
    solver_config: Optional[Dict[str, Any]] = None
):
    """
    Create optimized solver for anisotropic experiment.

    Args:
        problem: AnisotropicCrowdDynamics problem instance
        solver_config: Optional custom configuration

    Returns:
        Configured solver instance
    """
    if solver_config is None:
        has_barriers = len(problem.barriers) > 0
        solver_config = create_anisotropic_solver_config(
            has_barriers=has_barriers,
            grid_size=problem.grid_size
        )

    # Create solver with enhanced configuration
    solver = create_fast_solver(problem, solver_type='fixed_point')

    # Additional setup for barrier handling
    if hasattr(problem, 'barriers') and problem.barriers:
        _setup_barrier_handling(solver, problem)

    return solver


def _setup_barrier_handling(solver, problem):
    """Setup specialized barrier handling in solver."""

    def enhanced_cfl_condition(u, m, dx, dy, dt_base):
        """Enhanced CFL condition accounting for barriers and cross-coupling."""

        # Standard velocity-based CFL
        velocity = problem.compute_hamiltonian_gradient(
            problem._get_grid_points(),
            np.gradient(u),
            m.flatten()
        )
        max_velocity = np.max(np.linalg.norm(velocity, axis=1))

        # Transport CFL
        dt_transport = 0.5 * min(dx, dy) / (max_velocity + 1e-12)

        # Diffusion CFL
        dt_diffusion = 0.5 * min(dx**2, dy**2) / problem.sigma

        # Cross-coupling CFL (conservative estimate)
        rho_max = abs(problem.rho_amplitude)
        dt_cross = 0.1 * min(dx, dy) / (rho_max * max_velocity + 1e-12)

        # Barrier resolution CFL
        dt_barrier = dt_base
        if problem.barriers:
            # Estimate minimum barrier distance
            grid_points = problem._get_grid_points()
            min_barrier_dist = np.inf
            for barrier in problem.barriers:
                distances = barrier.compute_distance(grid_points)
                min_barrier_dist = min(min_barrier_dist, np.min(np.abs(distances)))

            dt_barrier = 0.1 * min_barrier_dist / (max_velocity + 1e-12)

        # Nonlinear coupling CFL
        max_density = np.max(m)
        penalty_strength = solver.config.get('barrier_penalty_strength', 0)
        dt_nonlinear = 0.05 / (problem.gamma * max_density + penalty_strength + 1e-12)

        # Take most restrictive
        dt_final = min(dt_transport, dt_diffusion, dt_cross, dt_barrier, dt_nonlinear)

        logger.debug(f"CFL components: transport={dt_transport:.2e}, diffusion={dt_diffusion:.2e}, "
                    f"cross={dt_cross:.2e}, barrier={dt_barrier:.2e}, nonlinear={dt_nonlinear:.2e}")

        return dt_final

    # Attach enhanced CFL to solver
    solver.enhanced_cfl_condition = enhanced_cfl_condition


def get_convergence_study_configs(base_grid: int = 32) -> Dict[str, Dict[str, Any]]:
    """
    Generate solver configurations for convergence study.

    Args:
        base_grid: Base grid size for scaling

    Returns:
        Dictionary of configurations for different grid sizes
    """
    configs = {}

    grid_sizes = [base_grid, 2*base_grid, 4*base_grid]

    for i, N in enumerate(grid_sizes):
        config_name = f"grid_{N}x{N}"

        # Adjust parameters based on grid size
        if N <= 64:
            solver_type = 'semi_lagrangian'
            cfl_factor = 0.3
        elif N <= 128:
            solver_type = 'semi_lagrangian'
            cfl_factor = 0.2
        else:
            solver_type = 'finite_difference'  # More stable for large grids
            cfl_factor = 0.1

        configs[config_name] = create_anisotropic_solver_config(
            has_barriers=True,  # Assume barriers for convergence study
            grid_size=(N, N),
            cfl_safety_factor=cfl_factor,
            solver_type=solver_type
        )

        # Additional refinements for larger grids
        if N >= 128:
            configs[config_name].update({
                'adaptive_time_stepping': True,
                'error_tolerance': 1e-8,
                'max_time_step_refinement': 3
            })

    return configs


def create_performance_optimized_config(
    grid_size: tuple,
    target_accuracy: str = 'standard'
) -> Dict[str, Any]:
    """
    Create performance-optimized configuration.

    Args:
        grid_size: Spatial resolution
        target_accuracy: 'fast', 'standard', or 'high_accuracy'

    Returns:
        Performance-optimized configuration
    """
    base_config = create_anisotropic_solver_config(
        has_barriers=True,
        grid_size=grid_size
    )

    if target_accuracy == 'fast':
        # Optimized for speed
        performance_config = {
            'cfl_safety_factor': 0.4,
            'nonlinear_tolerance': 1e-4,
            'max_iterations': 500,
            'matrix_solver': 'iterative',
            'preconditioner': 'jacobi'
        }
    elif target_accuracy == 'high_accuracy':
        # Optimized for accuracy
        performance_config = {
            'cfl_safety_factor': 0.1,
            'nonlinear_tolerance': 1e-8,
            'max_iterations': 2000,
            'matrix_solver': 'sparse_direct',
            'temporal_order': 2,
            'spatial_order': 4
        }
    else:  # standard
        # Balanced configuration
        performance_config = {
            'cfl_safety_factor': 0.2,
            'nonlinear_tolerance': 1e-6,
            'max_iterations': 1000,
            'matrix_solver': 'iterative',
            'preconditioner': 'ilu'
        }

    base_config.update(performance_config)
    return base_config


def validate_solver_config(config: Dict[str, Any]) -> bool:
    """
    Validate solver configuration for anisotropic problems.

    Args:
        config: Solver configuration dictionary

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ['cfl_safety_factor', 'nonlinear_tolerance', 'max_iterations']

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    # Validate CFL safety factor
    if not 0.01 <= config['cfl_safety_factor'] <= 1.0:
        raise ValueError(f"CFL safety factor {config['cfl_safety_factor']} must be in [0.01, 1.0]")

    # Validate tolerance
    if not 1e-12 <= config['nonlinear_tolerance'] <= 1e-2:
        raise ValueError(f"Tolerance {config['nonlinear_tolerance']} must be in [1e-12, 1e-2]")

    # Validate max iterations
    if config['max_iterations'] < 10:
        raise ValueError(f"Max iterations {config['max_iterations']} must be >= 10")

    # Cross-coupling specific validations
    if config.get('cross_coupling_treatment') not in [None, 'explicit', 'implicit']:
        raise ValueError("cross_coupling_treatment must be 'explicit' or 'implicit'")

    logger.info("Solver configuration validation passed")
    return True


if __name__ == "__main__":
    # Example usage
    print("=== Anisotropic Solver Configuration Examples ===")

    # Basic configuration
    config1 = create_anisotropic_solver_config(has_barriers=False)
    print(f"Basic config CFL factor: {config1['cfl_safety_factor']}")

    # Barrier configuration
    config2 = create_anisotropic_solver_config(has_barriers=True, grid_size=(128, 128))
    print(f"Barrier config with large grid: {config2['matrix_solver']}")

    # Performance configurations
    fast_config = create_performance_optimized_config((64, 64), 'fast')
    accurate_config = create_performance_optimized_config((64, 64), 'high_accuracy')

    print(f"Fast config tolerance: {fast_config['nonlinear_tolerance']}")
    print(f"Accurate config tolerance: {accurate_config['nonlinear_tolerance']}")

    # Convergence study
    convergence_configs = get_convergence_study_configs()
    print(f"Convergence study grids: {list(convergence_configs.keys())}")

    # Validation
    try:
        validate_solver_config(config1)
        print("Configuration validation: PASSED")
    except ValueError as e:
        print(f"Configuration validation: FAILED - {e}")