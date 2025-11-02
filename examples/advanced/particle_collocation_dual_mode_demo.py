"""
Demonstration of Dual-Mode FP Solver with Particle-Collocation MFG.

This example demonstrates the new collocation mode of FPParticleSolver,
which enables true meshfree MFG workflows when combined with particle-collocation
HJB solvers (GFDM).

Key Features:
- HJB and FP solvers share the SAME particle discretization
- No grid interpolation (fully meshfree)
- Output on particles (not grid)
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from mfg_pde.alg.numerical.fp_solvers import FPParticleSolver, ParticleMode
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry.implicit import Hyperrectangle


class SimpleLQMFG2D(MFGProblem):
    """Simple 2D LQ-MFG problem for demonstration."""

    def __init__(self):
        super().__init__(
            T=1.0,
            Nt=20,
            Nx=30,  # Not used in collocation mode, but required by base class
            Lx=1.0,
            xmin=0.0,
            sigma=0.2,
            coefCT=0.5,
            dimension=2,
        )


def run_hybrid_mode_demo():
    """Demo 1: Hybrid mode (default) - particles → grid output."""
    print("\n" + "=" * 70)
    print("DEMO 1: Hybrid Mode (Default Behavior)")
    print("=" * 70)
    print("Particles sample their own distribution, output to grid via KDE")

    problem = SimpleLQMFG2D()

    # Hybrid mode: default behavior (backward compatible)
    solver = FPParticleSolver(problem, num_particles=5000)

    print("\nSolver Configuration:")
    print(f"  Mode: {solver.mode.value}")
    print(f"  Method Name: {solver.fp_method_name}")
    print(f"  Num Particles: {solver.num_particles}")
    print(f"  Collocation Points: {solver.collocation_points}")

    # Initial condition and value function on GRID
    m0 = np.ones(problem.Nx + 1) / (problem.Nx + 1)
    U = np.zeros((problem.Nt + 1, problem.Nx + 1))

    # Solve FP system
    M = solver.solve_fp_system(m0, U, show_progress=False)

    print("\nOutput:")
    print(f"  M shape: {M.shape}")
    print(f"  Expected: ({problem.Nt + 1}, {problem.Nx + 1}) [grid output]")
    print(f"  Mass conservation: {np.sum(M[0, :]):.6f} → {np.sum(M[-1, :]):.6f}")


def run_collocation_mode_demo():
    """Demo 2: Collocation mode - particles → particles output."""
    print("\n" + "=" * 70)
    print("DEMO 2: Collocation Mode (New Capability)")
    print("=" * 70)
    print("External particles used as collocation points, output on particles")

    problem = SimpleLQMFG2D()

    # Sample collocation points from domain
    domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
    N_points = 1000
    points = domain.sample_uniform(N_points, seed=42)

    print("\nCollocation Points:")
    print(f"  Shape: {points.shape}")
    print("  Domain: [0, 1] × [0, 1]")

    # Collocation mode: output on particles (no KDE)
    solver = FPParticleSolver(problem, mode=ParticleMode.COLLOCATION, external_particles=points)

    print("\nSolver Configuration:")
    print(f"  Mode: {solver.mode.value}")
    print(f"  Method Name: {solver.fp_method_name}")
    print(f"  Num Particles: {solver.num_particles}")
    print(f"  Collocation Points Shape: {solver.collocation_points.shape}")

    # Initial condition and value function on PARTICLES
    m0 = np.ones(N_points) / N_points
    U = np.zeros((problem.Nt + 1, N_points))

    # Solve FP system
    M = solver.solve_fp_system(m0, U, show_progress=False)

    print("\nOutput:")
    print(f"  M shape: {M.shape}")
    print(f"  Expected: ({problem.Nt + 1}, {N_points}) [particle output]")
    print(f"  Mass conservation: {np.sum(M[0, :]):.6f} → {np.sum(M[-1, :]):.6f}")

    return points, M


def run_particle_collocation_mfg_demo():
    """Demo 3: Conceptual particle-collocation MFG workflow."""
    print("\n" + "=" * 70)
    print("DEMO 3: Particle-Collocation MFG Workflow (Conceptual)")
    print("=" * 70)
    print("HJB (GFDM) and FP (Collocation) can use SAME particles - true meshfree MFG")

    problem = SimpleLQMFG2D()

    # Sample collocation points ONCE
    domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
    N_points = 500
    points = domain.sample_uniform(N_points, seed=42)

    print("\nShared Discretization:")
    print(f"  Collocation Points: {N_points}")
    print("  Domain: [0, 1] × [0, 1]")

    # Create FP solver in collocation mode
    fp_solver = FPParticleSolver(problem, mode="collocation", external_particles=points)

    print("\nFP Solver (Collocation Mode):")
    print(f"  Mode: {fp_solver.mode.value}")
    print(f"  Points: {fp_solver.num_particles}")
    print("  Output: Density on particles (not grid)")

    print("\nConceptual MFG Workflow:")
    print("  1. Sample N collocation points from domain")
    print("  2. Create HJBGFDMSolver(collocation_points=points)")
    print("  3. Create FPParticleSolver(mode='collocation', external_particles=points)")
    print("  4. Both solvers use SAME N points - no grid!")
    print("  5. In Picard iteration:")
    print("     - HJB outputs U: shape (Nt, N)")
    print("     - FP outputs M: shape (Nt, N)")
    print("     - Perfect compatibility - same discretization")

    # Demonstrate FP collocation solve with dummy U
    M_current = np.ones(N_points) / N_points
    U_dummy = np.zeros((problem.Nt + 1, N_points))

    print("\nDemonstration with Dummy Value Function:")
    M_new = fp_solver.solve_fp_system(M_current, U_dummy, show_progress=False)
    print(f"  Input M shape: {M_current.shape}")
    print(f"  Output M shape: {M_new.shape}")
    print(f"  Mass conservation: {np.sum(M_current):.6f} → {np.sum(M_new[-1, :]):.6f}")

    print("\nKey Achievement:")
    print("  True meshfree MFG: Both HJB and FP on particles")
    print("  No grid interpolation required")
    print("  Enables high-dimensional MFG (d > 3)")

    return points, U_dummy, M_new


def visualize_collocation_mode(points, M):
    """Visualize density evolution on particles."""
    print("\n" + "=" * 70)
    print("Visualization: Density on Particles")
    print("=" * 70)

    _fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Time snapshots
    time_indices = [0, M.shape[0] // 2, M.shape[0] - 1]
    time_labels = ["Initial", "Middle", "Final"]

    for ax, t_idx, label in zip(axes, time_indices, time_labels, strict=False):
        scatter = ax.scatter(points[:, 0], points[:, 1], c=M[t_idx, :], s=20, cmap="viridis")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"Density at {label} Time")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect("equal")
        plt.colorbar(scatter, ax=ax, label="m(t,x)")

    plt.tight_layout()
    plt.savefig("examples/outputs/dual_mode_fp_solver_demo.png", dpi=150)
    print("\nSaved visualization: examples/outputs/dual_mode_fp_solver_demo.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DUAL-MODE FP SOLVER DEMONSTRATION")
    print("=" * 70)
    print("Demonstrating hybrid mode (default) and collocation mode (new capability)")

    # Demo 1: Hybrid mode (backward compatibility)
    run_hybrid_mode_demo()

    # Demo 2: Collocation mode (new capability)
    points, M = run_collocation_mode_demo()

    # Demo 3: Full particle-collocation MFG workflow
    points_mfg, U, M_mfg = run_particle_collocation_mfg_demo()

    # Visualize collocation mode results
    visualize_collocation_mode(points, M)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✓ Hybrid mode: Particles → Grid (default, backward compatible)")
    print("✓ Collocation mode: Particles → Particles (new, meshfree)")
    print("✓ Particle-Collocation MFG: HJB + FP on same particles")
    print("\nCore Achievement:")
    print("  True meshfree MFG workflow enabled by dual-mode FP solver")
    print("=" * 70)
