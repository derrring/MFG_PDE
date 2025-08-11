#!/usr/bin/env python3
"""
Adaptive Convergence Decorator Example

This example demonstrates the adaptive convergence decorator that automatically
detects particle methods and applies appropriate convergence criteria:

1. Classical L2 error convergence for grid-based methods
2. Advanced multi-criteria convergence for particle-based methods

The decorator automatically adapts based on solver components, eliminating
the need for separate enhanced solver classes.
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import BoundaryConditions, ExampleMFGProblem
from mfg_pde.alg.damped_fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.fp_solvers import FPFDMSolver
from mfg_pde.alg.hjb_solvers import HJBFDMSolver
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.utils.convergence import (
    adaptive_convergence,
    test_particle_detection,
    wrap_solver_with_adaptive_convergence,
)


def demonstrate_adaptive_convergence():
    """
    Main demonstration of adaptive convergence decorator.
    """
    print("=" * 80)
    print("ADAPTIVE CONVERGENCE DECORATOR DEMONSTRATION")
    print("=" * 80)
    print("Automatically detects particle methods and adapts convergence criteria")
    print()

    # Setup common problem
    problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=20, T=0.8, Nt=15, sigma=0.12, coefCT=0.025)
    boundary_conditions = BoundaryConditions(type="no_flux")

    print(f"Test Problem: {problem.Nx}×{problem.Nt} grid, σ={problem.sigma}")
    print()

    # Test 1: Pure FDM Solver (should use classical convergence)
    print("=" * 60)
    print("TEST 1: Pure FDM Solver (Grid-based)")
    print("=" * 60)

    fdm_hjb_solver = HJBFDMSolver(problem=problem)
    fdm_fp_solver = FPFDMSolver(problem=problem, boundary_conditions=boundary_conditions)

    pure_fdm_solver = FixedPointIterator(
        problem=problem, hjb_solver=fdm_hjb_solver, fp_solver=fdm_fp_solver, thetaUM=0.5
    )

    # Test particle detection
    detection_result = test_particle_detection(pure_fdm_solver)
    print("Particle Detection Results:")
    print(f"   Has particles: {detection_result['has_particles']}")
    print(f"   Recommended: {detection_result['recommended_convergence']} convergence")
    print(f"   Evidence: {detection_result['detection_info']['particle_components']}")
    print()

    # Wrap with adaptive convergence
    adaptive_fdm = wrap_solver_with_adaptive_convergence(pure_fdm_solver, classical_tol=1e-3, verbose=True)

    print("Solving with adaptive convergence wrapper...")
    result = adaptive_fdm.solve(Niter_max=8, l2errBoundPicard=1e-3)

    # Handle variable return values from FixedPointIterator
    if len(result) == 5:
        U_fdm, M_fdm, iters_fdm, rel_distu, rel_distm = result
        info_fdm = {
            'iterations': iters_fdm,
            'final_rel_distu': rel_distu,
            'final_rel_distm': rel_distm,
            'convergence_mode': 'classical',
        }
    else:
        U_fdm, M_fdm = result[:2]
        info_fdm = {'convergence_mode': 'classical'}

    print(f"Solved successfully!")
    print(f"   Convergence mode: {info_fdm.get('convergence_mode', 'classical')}")
    print(f"   Iterations: {info_fdm.get('iterations', 'unknown')}")
    print()

    # Test 2: Particle Collocation Solver (should use advanced convergence)
    print("=" * 60)
    print("TEST 2: Particle Collocation Solver")
    print("=" * 60)

    collocation_points = np.linspace(problem.xmin, problem.xmax, 8).reshape(-1, 1)

    particle_solver = ParticleCollocationSolver(
        problem=problem,
        collocation_points=collocation_points,
        num_particles=2000,
        delta=0.4,
        boundary_conditions=boundary_conditions,
        use_monotone_constraints=True,
    )

    # Test particle detection
    detection_result = test_particle_detection(particle_solver)
    print("Particle Detection Results:")
    print(f"   Has particles: {detection_result['has_particles']}")
    print(f"   Recommended: {detection_result['recommended_convergence']} convergence")
    print(f"   Evidence: {detection_result['detection_info']['particle_components'][:3]}")
    print(f"   Confidence: {detection_result['detection_info']['confidence']:.1%}")
    print()

    # Wrap with adaptive convergence
    adaptive_particle = wrap_solver_with_adaptive_convergence(
        particle_solver,
        classical_tol=1e-3,
        wasserstein_tol=2e-4,
        u_magnitude_tol=1e-3,
        u_stability_tol=5e-4,
        verbose=True,
    )

    print("Solving with adaptive convergence wrapper...")
    result = adaptive_particle.solve(Niter=10, l2errBound=1e-3)

    # Handle variable return values from ParticleCollocationSolver
    if len(result) == 3:
        U_particle, M_particle, info_particle = result
    else:
        U_particle, M_particle = result[:2]
        info_particle = {'convergence_mode': 'particle_aware'}

    print(f"Solved successfully!")
    print(f"   Convergence mode: {info_particle.get('convergence_mode', 'unknown')}")
    if 'mass_conservation_error' in info_particle:
        print(f"   Mass conservation error: {info_particle['mass_conservation_error']:.3f}%")
    print()

    # Test 3: Using decorator on custom solver class
    print("=" * 60)
    print("TEST 3: Decorator Pattern on Custom Solver")
    print("=" * 60)

    demonstrate_decorator_pattern(problem, boundary_conditions)

    # Create comparison visualization
    create_adaptive_convergence_comparison(
        [U_fdm, U_particle], [M_fdm, M_particle], ['Pure FDM (Classical)', 'Particle Collocation (Advanced)'], problem
    )

    print("\n" + "=" * 80)
    print("ADAPTIVE CONVERGENCE DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("Key Benefits:")
    print("- Automatic detection of particle methods")
    print("- Seamless adaptation of convergence criteria")
    print("- No need for separate enhanced solver classes")
    print("- Works with any existing solver via wrapper or decorator")
    print("- Maintains backward compatibility")


def demonstrate_decorator_pattern(problem, boundary_conditions):
    """
    Demonstrate using the decorator pattern directly on solver classes.
    """

    # Create a custom solver class with adaptive convergence decorator
    @adaptive_convergence(classical_tol=1e-3, wasserstein_tol=1e-4, verbose=True)
    class AdaptiveParticleCollocationSolver(ParticleCollocationSolver):
        """
        Particle collocation solver with automatic adaptive convergence.
        The decorator automatically detects particle usage and applies
        advanced convergence criteria.
        """

        pass

    print("Creating solver with @adaptive_convergence decorator...")

    collocation_points = np.linspace(problem.xmin, problem.xmax, 6).reshape(-1, 1)

    # The solver automatically gets adaptive convergence behavior
    decorated_solver = AdaptiveParticleCollocationSolver(
        problem=problem,
        collocation_points=collocation_points,
        num_particles=1500,
        boundary_conditions=boundary_conditions,
    )

    print("Solving with decorated solver class...")
    result = decorated_solver.solve(Niter=8, l2errBound=1e-3)

    # Handle variable return values
    if len(result) == 3:
        U, M, info = result
    else:
        U, M = result[:2]
        info = {'convergence_mode': 'particle_aware'}

    print(f"Decorator pattern successful!")
    print(f"   Convergence mode: {info.get('convergence_mode', 'unknown')}")
    if 'particle_detection' in info:
        print(f"   Automatic detection worked: {info['particle_detection'].get('confidence', 0):.1%} confidence")

    # Show the wrapper reference for debugging
    if hasattr(decorated_solver, '_adaptive_convergence_wrapper'):
        wrapper = decorated_solver._adaptive_convergence_wrapper
        print(f"   Wrapper mode: {wrapper.get_convergence_mode()}")


def create_adaptive_convergence_comparison(U_list, M_list, labels, problem):
    """
    Create visualization comparing solutions from different convergence approaches.
    """
    print("\nCreating adaptive convergence comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Adaptive Convergence: Classical vs Advanced Criteria', fontsize=16)

    x_grid = np.linspace(problem.xmin, problem.xmax, problem.Nx)
    time_grid = np.linspace(0, problem.T, problem.Nt + 1)

    colors = ['#FF6B6B', '#4ECDC4']

    # Plot 1: Initial Value Function
    ax1 = axes[0, 0]
    for i, (U, label) in enumerate(zip(U_list, labels)):
        ax1.plot(x_grid, U[0, :], color=colors[i], linewidth=2, label=f'{label}')
    ax1.set_xlabel('Space (x)')
    ax1.set_ylabel('Value Function U(x, 0)')
    ax1.set_title('Initial Value Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Terminal Value Function
    ax2 = axes[0, 1]
    for i, (U, label) in enumerate(zip(U_list, labels)):
        ax2.plot(x_grid, U[-1, :], color=colors[i], linewidth=2, label=f'{label}')
    ax2.set_xlabel('Space (x)')
    ax2.set_ylabel('Value Function U(x, T)')
    ax2.set_title('Terminal Value Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Initial Distribution
    ax3 = axes[1, 0]
    for i, (M, label) in enumerate(zip(M_list, labels)):
        ax3.plot(x_grid, M[0, :], color=colors[i], linewidth=2, label=f'{label}')
    ax3.set_xlabel('Space (x)')
    ax3.set_ylabel('Distribution M(x, 0)')
    ax3.set_title('Initial Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Final Distribution
    ax4 = axes[1, 1]
    for i, (M, label) in enumerate(zip(M_list, labels)):
        ax4.plot(x_grid, M[-1, :], color=colors[i], linewidth=2, label=f'{label}')
    ax4.set_xlabel('Space (x)')
    ax4.set_ylabel('Distribution M(x, T)')
    ax4.set_title('Final Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = "adaptive_convergence_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {filename}")

    plt.show()


def demonstrate_detection_capabilities():
    """
    Demonstrate the particle detection capabilities on various solver types.
    """
    print("\n" + "=" * 60)
    print("PARTICLE DETECTION CAPABILITIES DEMO")
    print("=" * 60)

    # Create test problem
    problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=15, T=0.5, Nt=10)
    boundary_conditions = BoundaryConditions(type="no_flux")

    # Test various solver types
    test_cases = []

    # Case 1: Pure FDM
    fdm_hjb = HJBFDMSolver(problem=problem)
    fdm_fp = FPFDMSolver(problem=problem, boundary_conditions=boundary_conditions)
    pure_fdm = FixedPointIterator(problem=problem, hjb_solver=fdm_hjb, fp_solver=fdm_fp)
    test_cases.append(("Pure FDM Iterator", pure_fdm))

    # Case 2: Particle Collocation
    points = np.linspace(0, 1, 5).reshape(-1, 1)
    particle_collocation = ParticleCollocationSolver(problem=problem, collocation_points=points, num_particles=1000)
    test_cases.append(("Particle Collocation", particle_collocation))

    # Run detection tests
    for name, solver in test_cases:
        print(f"\nTesting: {name}")
        result = test_particle_detection(solver)

        print(f"   Particle methods detected: {'YES' if result['has_particles'] else 'NO'}")
        print(f"   Confidence: {result['detection_info']['confidence']:.1%}")
        print(f"   Detection methods: {result['detection_info']['detection_methods']}")
        print(f"   Evidence found: {len(result['detection_info']['particle_components'])} items")
        print(f"   Recommended convergence: {result['recommended_convergence'].upper()}")


if __name__ == "__main__":
    demonstrate_adaptive_convergence()
    demonstrate_detection_capabilities()
