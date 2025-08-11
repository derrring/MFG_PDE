#!/usr/bin/env python3
"""
MFG Solver Factory Patterns Example

Demonstrates how to use the new factory patterns for easy solver creation
with optimized configurations for different use cases.
"""

import os
import sys

import numpy as np

# Add the parent directory to the path so we can import mfg_pde
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mfg_pde import (
    MFGProblem,
    SolverFactory,
    create_accurate_solver,
    create_fast_solver,
    create_monitored_solver,
    create_research_solver,
    create_solver,
)
from mfg_pde.alg.fp_solvers import FPParticleSolver
from mfg_pde.alg.hjb_solvers import HJBGFDMSolver
from mfg_pde.config import create_research_config_dataclass as create_research_config


def create_test_problem():
    """Create a simple test MFG problem."""

    class SimpleMFGProblem(MFGProblem):
        def __init__(self):
            super().__init__(T=1.0, Nt=20, xmin=0.0, xmax=1.0, Nx=50)

        def g(self, x):
            """Terminal cost function."""
            return 0.5 * (x - 0.5) ** 2

        def rho0(self, x):
            """Initial distribution."""
            return np.exp(-10 * (x - 0.3) ** 2) + np.exp(-10 * (x - 0.7) ** 2)

        def f(self, x, u, m):
            """Running cost function."""
            return 0.1 * u**2 + 0.05 * m

        def sigma(self, x):
            """Diffusion coefficient."""
            return 0.1

        def H(self, x, p, m):
            """Hamiltonian function."""
            return 0.5 * p**2

        def dH_dm(self, x, p, m):
            """Derivative of Hamiltonian with respect to density."""
            return 0.0

    return SimpleMFGProblem()


def demonstrate_simple_factory_usage():
    """Demonstrate simple factory usage with convenience functions."""
    print("=" * 80)
    print("SIMPLE FACTORY USAGE DEMONSTRATION")
    print("=" * 80)

    problem = create_test_problem()

    # Create collocation points for particle methods (needs to be 2D for GFDM)
    x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx)
    collocation_points = x_coords.reshape(-1, 1)  # Make it 2D

    print("1. Creating a FAST particle collocation solver:")
    fast_solver = create_fast_solver(
        problem=problem, solver_type="particle_collocation", collocation_points=collocation_points
    )
    print(f"   Solver type: {type(fast_solver).__name__}")
    print(f"   Particles: {fast_solver.num_particles}")
    print(f"   HJB solver newton tolerance: {fast_solver.hjb_solver.newton_tolerance}")
    print()

    print("2. Creating an ACCURATE monitored particle solver:")
    accurate_solver = create_accurate_solver(
        problem=problem, solver_type="monitored_particle", collocation_points=collocation_points
    )
    print(f"   Solver type: {type(accurate_solver).__name__}")
    print(f"   Particles: {accurate_solver.num_particles}")
    print(f"   HJB solver newton tolerance: {accurate_solver.hjb_solver.newton_tolerance}")
    print()

    print("3. Creating a RESEARCH solver with comprehensive monitoring:")
    research_solver = create_research_solver(problem=problem, collocation_points=collocation_points)
    print(f"   Solver type: {type(research_solver).__name__}")
    print(f"   Particles: {research_solver.num_particles}")
    print(f"   Has convergence monitor: {hasattr(research_solver, 'convergence_monitor')}")
    print()

    print("4. Creating a custom MONITORED solver:")
    monitored_solver = create_monitored_solver(
        problem=problem,
        collocation_points=collocation_points,
        num_particles=8000,  # Custom parameter
        newton_tolerance=1e-6,  # Custom parameter
    )
    print(f"   Solver type: {type(monitored_solver).__name__}")
    print(f"   Particles: {monitored_solver.num_particles}")
    print(f"   HJB solver newton tolerance: {monitored_solver.hjb_solver.newton_tolerance}")
    print()


def demonstrate_advanced_factory_usage():
    """Demonstrate advanced factory usage with custom configurations."""
    print("=" * 80)
    print("ADVANCED FACTORY USAGE DEMONSTRATION")
    print("=" * 80)

    problem = create_test_problem()
    x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx)
    collocation_points = x_coords.reshape(-1, 1)

    print("1. Using SolverFactory directly with custom config:")

    # Create a custom configuration
    custom_config = create_research_config()
    custom_config.picard.max_iterations = 15
    custom_config.fp.particle.num_particles = 7500
    custom_config.hjb.gfdm.delta = 0.25

    custom_solver = SolverFactory.create_solver(
        problem=problem,
        solver_type="monitored_particle",
        collocation_points=collocation_points,
        custom_config=custom_config,
    )

    print(f"   Solver type: {type(custom_solver).__name__}")
    print(f"   Particles: {custom_solver.num_particles}")
    print(f"   HJB solver newton tolerance: {custom_solver.hjb_solver.newton_tolerance}")
    print()

    print("2. Creating solver with mixed preset and custom parameters:")

    mixed_solver = SolverFactory.create_solver(
        problem=problem,
        solver_type="adaptive_particle",
        config_preset="fast",  # Start with fast preset
        collocation_points=collocation_points,
        # Override specific parameters
        num_particles=4000,
        max_newton_iterations=40,
        newton_tolerance=1e-6,
    )

    print(f"   Solver type: {type(mixed_solver).__name__}")
    print(f"   Particles: {mixed_solver.num_particles}")
    print(f"   HJB newton tolerance: {mixed_solver.hjb_solver.newton_tolerance}")
    print()

    print("3. Creating fixed point solver with factory:")

    # Create component solvers
    hjb_solver = HJBGFDMSolver(problem, collocation_points)
    fp_solver = FPParticleSolver(problem, collocation_points)

    fixed_point_solver = create_accurate_solver(
        problem=problem,
        solver_type="fixed_point",
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
        max_picard_iterations=30,
        warm_start=True,
    )

    print(f"   Solver type: {type(fixed_point_solver).__name__}")
    print(f"   Has warm start: {hasattr(fixed_point_solver, 'set_warm_start')}")
    print(f"   Config picard iterations: {fixed_point_solver.config.picard.max_iterations}")
    print()


def demonstrate_solver_comparison():
    """Demonstrate how different presets create different solver behaviors."""
    print("=" * 80)
    print("SOLVER PRESET COMPARISON")
    print("=" * 80)

    problem = create_test_problem()
    x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx)
    collocation_points = x_coords.reshape(-1, 1)

    presets = ["fast", "balanced", "accurate", "research"]

    for preset in presets:
        print(f"{preset.upper()} Preset Configuration:")

        solver = create_solver(
            problem=problem, solver_type="monitored_particle", preset=preset, collocation_points=collocation_points
        )

        print(f"   Particles: {solver.num_particles}")
        print(f"   HJB newton tolerance: {solver.hjb_solver.newton_tolerance}")
        print(f"   Has convergence monitor: {hasattr(solver, 'convergence_monitor')}")

        if hasattr(solver, 'convergence_monitor'):
            monitor = solver.convergence_monitor
            print(f"   Wasserstein tolerance: {getattr(monitor, 'wasserstein_tol', 'N/A')}")

        print()


def demonstrate_factory_benefits():
    """Demonstrate the benefits of using factory patterns."""
    print("=" * 80)
    print("FACTORY PATTERN BENEFITS")
    print("=" * 80)

    problem = create_test_problem()
    x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx)
    collocation_points = x_coords.reshape(-1, 1)

    print("1. SIMPLICITY - One-line solver creation:")
    print("   solver = create_fast_solver(problem, 'particle_collocation', collocation_points=points)")
    solver1 = create_fast_solver(problem, "particle_collocation", collocation_points=collocation_points)
    print(f"   Created: {type(solver1).__name__}")
    print()

    print("2. CONSISTENCY - Optimized defaults for use cases:")
    print("   Fast: Optimized for speed")
    print("   Accurate: Optimized for precision")
    print("   Research: Comprehensive monitoring and analysis")
    print()

    print("3. FLEXIBILITY - Easy customization:")
    print("   create_solver(problem, 'monitored_particle', preset='fast', num_particles=10000)")
    solver2 = create_solver(
        problem, "monitored_particle", preset="fast", collocation_points=collocation_points, num_particles=10000
    )
    print(f"   Created: {type(solver2).__name__} with {solver2.num_particles} particles")
    print()

    print("4. MAINTAINABILITY - Centralized configuration management:")
    print("   All solver parameters managed through configuration system")
    print("   Easy to update defaults and add new presets")
    print()

    print("5. EXTENSIBILITY - Easy to add new solver types:")
    print("   Factory pattern makes it simple to add new solver variants")
    print("   Consistent interface across all solver types")
    print()


def run_factory_examples():
    """Run all factory pattern examples."""
    print("MFG SOLVER FACTORY PATTERNS DEMONSTRATION")
    print("=" * 80)
    print("This example demonstrates the new factory patterns for creating")
    print("MFG solvers with optimized configurations for different use cases.")
    print()

    try:
        demonstrate_simple_factory_usage()
        demonstrate_advanced_factory_usage()
        demonstrate_solver_comparison()
        demonstrate_factory_benefits()

        print("=" * 80)
        print("FACTORY PATTERNS DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()
        print("Key Benefits of Factory Patterns:")
        print("• One-line solver creation with optimized defaults")
        print("• Consistent configurations for different use cases")
        print("• Easy customization through parameter overrides")
        print("• Centralized configuration management")
        print("• Extensible design for new solver types")
        print()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_factory_examples()
