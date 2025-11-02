#!/usr/bin/env python3
"""
Utility Module Demonstration

Demonstrates new utility features from Phase 2.2 improvements:
1. Particle interpolation utilities
2. Geometry utilities (SDF, obstacles)
3. QP solver with caching

Run:
    python examples/basic/utility_demo.py
"""

import numpy as np

from mfg_pde.utils import (
    CircleObstacle,
    QPCache,
    QPSolver,
    RectangleObstacle,
    Union,
    create_circle_obstacle,
    interpolate_grid_to_particles,
    interpolate_particles_to_grid,
)


def demo_particle_interpolation():
    """Demonstrate particle-grid interpolation utilities."""
    print("\n" + "=" * 60)
    print("Particle Interpolation Demo")
    print("=" * 60)

    # 1D example: Grid to particles
    print("\n1. Grid → Particles (1D)")
    x_grid = np.linspace(0, 1, 50)
    u_grid = np.sin(2 * np.pi * x_grid)  # Smooth function on grid

    particle_positions = np.array([[0.25], [0.5], [0.75]])
    u_particles = interpolate_grid_to_particles(u_grid, ((0, 1),), particle_positions, method="cubic")

    print(f"   Grid points: {len(x_grid)}")
    print(f"   Particles: {len(particle_positions)}")
    print("   Interpolated values at particles:")
    for pos, val in zip(particle_positions, u_particles, strict=True):
        print(f"     x={pos[0]:.2f}: u={val:.3f} (exact: {np.sin(2 * np.pi * pos[0]):.3f})")

    # 2D example: Particles to grid (KDE density estimation)
    print("\n2. Particles → Grid (2D density estimation)")
    particles_2d = np.random.randn(500, 2) * 0.1 + 0.5  # Gaussian blob at (0.5, 0.5)

    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 1, 30)
    density = interpolate_particles_to_grid(particles_2d, (x, y), method="kde")

    print(f"   Particles: {len(particles_2d)}")
    print(f"   Grid: {len(x)} × {len(y)} = {density.size} points")
    print(f"   Density integral: {np.sum(density) * (x[1] - x[0]) * (y[1] - y[0]):.3f} (should be ≈ 1.0)")
    print(
        f"   Max density at: ({x[np.argmax(np.max(density, axis=1))]:.2f}, {y[np.argmax(np.max(density, axis=0))]:.2f})"
    )


def demo_geometry_utilities():
    """Demonstrate geometry utility aliases and CSG operations."""
    print("\n" + "=" * 60)
    print("Geometry Utilities Demo")
    print("=" * 60)

    # Create obstacles using convenient aliases
    print("\n1. Creating obstacles")
    wall = RectangleObstacle(np.array([[0.4, 0.6], [0.0, 0.5]]))
    pillar = CircleObstacle(center=np.array([0.7, 0.7]), radius=0.1)

    print("   Wall (rectangle): bounds = [[0.4, 0.6], [0.0, 0.5]]")
    print("   Pillar (circle): center = [0.7, 0.7], radius = 0.1")

    # Alternative: Factory functions
    obstacle3 = create_circle_obstacle(0.3, 0.3, 0.15)
    print("   Obstacle 3 (via factory): center = [0.3, 0.3], radius = 0.15")

    # Combine using CSG operations
    print("\n2. CSG operations (Union)")
    obstacles = Union([wall, pillar, obstacle3])

    # Query signed distance
    test_points = np.array([[0.5, 0.25], [0.1, 0.1], [0.7, 0.7]])

    print("\n3. Signed distance queries")
    for point in test_points:
        sdf = obstacles.signed_distance(point.reshape(1, -1))[0]
        inside = obstacles.contains(point.reshape(1, -1))[0]
        print(f"   Point {point}: SDF = {sdf:+.3f}, inside = {inside}")


def demo_qp_solver():
    """Demonstrate QP solver with caching."""
    print("\n" + "=" * 60)
    print("QP Solver Demo")
    print("=" * 60)

    # Create solver with caching
    cache = QPCache(max_size=100)
    solver = QPSolver(backend="auto", enable_warm_start=True, cache=cache)

    print("\n1. Solver configuration")
    print(f"   Backend: {solver.backend}")
    print(f"   Warm-start: {solver.enable_warm_start}")
    print(f"   Cache size: {cache.max_size}")

    # Solve a simple constrained least-squares problem
    print("\n2. Solving constrained least-squares")
    np.random.seed(42)
    A = np.random.randn(20, 5)
    b = np.random.randn(20)
    W = np.eye(20)
    bounds = [(-2, 2)] * 5  # Constrain all variables to [-2, 2]

    print("   Problem: min ||W^(1/2) (Ax - b)||^2  s.t. -2 ≤ x ≤ 2")
    print(f"   Matrix A: {A.shape}, vector b: {b.shape}")

    # First solve
    x1 = solver.solve_weighted_least_squares(A, b, W, bounds=bounds, point_id=0)
    print(f"   Solution: {x1}")
    print(f"   Bounds satisfied: {np.all(x1 >= -2) and np.all(x1 <= 2)}")

    # Solve again with same problem (should hit cache)
    print("\n3. Solving again (cache test)")
    x2 = solver.solve_weighted_least_squares(A, b, W, bounds=bounds, point_id=0)
    print(f"   Cache hits: {solver.stats['cache_hits']}")
    print(f"   Cache misses: {solver.stats['cache_misses']}")
    print(f"   Solutions match: {np.allclose(x1, x2)}")

    # Solve with modified problem (warm-start test)
    print("\n4. Solving modified problem (warm-start test)")
    b_mod = b + 0.1 * np.random.randn(20)
    solver.solve_weighted_least_squares(A, b_mod, W, bounds=bounds, point_id=0)
    print(f"   Warm starts: {solver.stats['warm_starts']}")
    print(f"   Cold starts: {solver.stats['cold_starts']}")

    # Print statistics
    print("\n5. Solver statistics")
    print(f"   Total solves: {solver.stats['total_solves']}")
    print(f"   Successes: {solver.stats['successes']}")
    print(f"   Cache hit rate: {cache.hit_rate:.1%}")
    print(
        f"   Warm-start rate: {solver.stats['warm_starts'] / (solver.stats['warm_starts'] + solver.stats['cold_starts']):.1%}"
    )


def main():
    """Run all utility demonstrations."""
    print("\n" + "=" * 60)
    print("MFG_PDE Utility Module Demonstration")
    print("Phase 2.2 Improvements")
    print("=" * 60)

    demo_particle_interpolation()
    demo_geometry_utilities()
    demo_qp_solver()

    print("\n" + "=" * 60)
    print("Demo complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
