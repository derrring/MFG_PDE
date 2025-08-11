#!/usr/bin/env python3
"""
Simplified Adaptive Convergence Decorator Example

This example demonstrates the adaptive convergence decorator that automatically
detects particle methods and applies appropriate convergence criteria.
"""

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
    print("SIMPLIFIED ADAPTIVE CONVERGENCE DEMONSTRATION")
    print("=" * 80)
    print("Automatically detects particle methods and adapts convergence criteria")
    print()

    # Create test problem (smaller for faster execution)
    problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=12, T=0.5, Nt=10, sigma=0.12, coefCT=0.02)
    boundary_conditions = BoundaryConditions(type="no_flux")

    print(f"Test Problem: {problem.Nx}x{problem.Nt} grid, sigma={problem.sigma}")
    print()

    # Test 1: Pure FDM Solver (Grid-based)
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
    print()

    # Wrap with adaptive convergence
    adaptive_fdm = wrap_solver_with_adaptive_convergence(pure_fdm_solver, classical_tol=1e-3, verbose=True)

    print("Solving with adaptive convergence wrapper...")
    result = adaptive_fdm.solve(Niter_max=3, l2errBoundPicard=1e-3)

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

    collocation_points = np.linspace(0.1, 0.9, 5).reshape(-1, 1)

    particle_solver = ParticleCollocationSolver(
        problem=problem,
        collocation_points=collocation_points,
        num_particles=500,  # Smaller number for faster execution
        boundary_conditions=boundary_conditions,
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
    result = adaptive_particle.solve(Niter=5, l2errBound=1e-3)

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

    @adaptive_convergence(classical_tol=1e-3, wasserstein_tol=2e-4, verbose=True)
    class CustomAdaptiveParticleCollocationSolver(ParticleCollocationSolver):
        """Example of using decorator pattern on solver class definition."""

        pass

    decorated_solver = CustomAdaptiveParticleCollocationSolver(
        problem=problem,
        collocation_points=collocation_points,
        num_particles=500,
        boundary_conditions=boundary_conditions,
    )

    print("Solving with decorated solver class...")
    result = decorated_solver.solve(Niter=5, l2errBound=1e-3)

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
    print()

    # Summary
    print()
    print("=" * 80)
    print("ADAPTIVE CONVERGENCE DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("Key Benefits:")
    print("- Automatic detection of particle methods")
    print("- Seamless adaptation of convergence criteria")
    print("- No need for separate enhanced solver classes")
    print("- Works with any existing solver via wrapper or decorator")
    print("- Maintains backward compatibility")


def demonstrate_detection_system():
    """
    Demonstrate the particle detection system on various solver types.
    """
    print("\n" + "=" * 80)
    print("PARTICLE DETECTION SYSTEM DEMONSTRATION")
    print("=" * 80)

    problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=10, T=0.5, Nt=8)
    boundary_conditions = BoundaryConditions(type="no_flux")
    collocation_points = np.linspace(0.1, 0.9, 3).reshape(-1, 1)

    # Test cases
    test_cases = []

    # Pure FDM
    fdm_hjb = HJBFDMSolver(problem=problem)
    fdm_fp = FPFDMSolver(problem=problem, boundary_conditions=boundary_conditions)
    fdm_solver = FixedPointIterator(problem, fdm_hjb, fdm_fp, thetaUM=0.5)
    test_cases.append(("Pure FDM", fdm_solver))

    # Particle Collocation
    particle_collocation = ParticleCollocationSolver(
        problem, collocation_points, num_particles=300, boundary_conditions=boundary_conditions
    )
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
    demonstrate_detection_system()
