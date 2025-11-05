#!/usr/bin/env python
"""
Geometry-First API Demo

Demonstrates the new recommended pattern for creating MFG problems using
the geometry-first API introduced in v0.10.0.

This example shows:
1. TensorProductGrid (structured grid)
2. Domain1D (1D with boundary conditions)
3. Hyperrectangle (implicit box domain)
4. Hypersphere (implicit sphere domain)
5. MazeGeometry (maze-based grid)
"""

import warnings

import numpy as np

# Suppress deprecation warnings (we're using new API)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def demo_tensor_product_grid():
    """Demonstrate TensorProductGrid for structured grids."""
    print("\n" + "=" * 70)
    print("1. TensorProductGrid (Structured Grid)")
    print("=" * 70)

    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry import TensorProductGrid

    # Create 2D grid: [0,1] × [0,1] with 51×51 points
    geometry = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[51, 51])

    print(f"\nGeometry: {geometry.geometry_type}")
    print(f"Dimension: {geometry.dimension}")
    print(f"Total points: {geometry.num_spatial_points}")

    # Create problem
    problem = MFGProblem(geometry=geometry, T=1.0, Nt=10, sigma=0.1)

    print(f"\nProblem dimension: {problem.dimension}")
    print(f"Problem domain type: {problem.domain_type}")
    print(f"Spatial shape: {problem.spatial_shape}")
    print(f"Time steps: {problem.Nt}")

    print("\n✓ TensorProductGrid working correctly")


def demo_domain_1d():
    """Demonstrate Domain1D with boundary conditions."""
    print("\n" + "=" * 70)
    print("2. Domain1D (1D with Boundary Conditions)")
    print("=" * 70)

    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry import Domain1D

    # Create 1D periodic domain
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions="periodic")
    domain.create_grid(Nx=101)

    print(f"\nGeometry: {domain.geometry_type}")
    print(f"Dimension: {domain.dimension}")
    print(f"Boundary conditions: {domain.boundary_conditions}")
    print(f"Total points: {domain.num_spatial_points}")

    # Create problem
    problem = MFGProblem(geometry=domain, T=1.0, Nt=10, sigma=0.1)

    print(f"\nProblem dimension: {problem.dimension}")
    print(f"Time steps: {problem.Nt}")

    print("\n✓ Domain1D working correctly")


def demo_hyperrectangle():
    """Demonstrate Hyperrectangle (implicit box domain)."""
    print("\n" + "=" * 70)
    print("3. Hyperrectangle (Implicit Box Domain)")
    print("=" * 70)

    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry.implicit import Hyperrectangle

    # Create [0,1]² box
    geometry = Hyperrectangle(bounds=np.array([[0, 1], [0, 1]]))

    print(f"\nGeometry: {geometry.geometry_type}")
    print(f"Dimension: {geometry.dimension}")
    print(f"Estimated points: {geometry.num_spatial_points}")

    # Test signed distance function
    print("\nSigned Distance Function:")
    print(f"  Interior point [0.5, 0.5]: φ = {geometry.signed_distance(np.array([0.5, 0.5])):.4f} (< 0)")
    print(f"  Exterior point [1.5, 0.5]: φ = {geometry.signed_distance(np.array([1.5, 0.5])):.4f} (> 0)")

    # Create problem
    problem = MFGProblem(geometry=geometry, T=1.0, Nt=10, sigma=0.1)

    print(f"\nProblem dimension: {problem.dimension}")
    print(f"Problem domain type: {problem.domain_type}")

    print("\n✓ Hyperrectangle working correctly")


def demo_hypersphere():
    """Demonstrate Hypersphere (implicit sphere domain)."""
    print("\n" + "=" * 70)
    print("4. Hypersphere (Implicit Sphere Domain)")
    print("=" * 70)

    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry.implicit import Hypersphere

    # Create unit sphere centered at origin
    geometry = Hypersphere(center=[0, 0], radius=1.0)

    print(f"\nGeometry: {geometry.geometry_type}")
    print(f"Dimension: {geometry.dimension}")
    print(f"Estimated points: {geometry.num_spatial_points}")

    # Test signed distance function
    print("\nSigned Distance Function:")
    print(f"  Center [0, 0]: φ = {geometry.signed_distance(np.array([0, 0])):.4f} (< 0)")
    print(f"  Exterior [2, 0]: φ = {geometry.signed_distance(np.array([2, 0])):.4f} (> 0)")
    print(f"  Boundary [1, 0]: φ = {geometry.signed_distance(np.array([1, 0])):.4f} (≈ 0)")

    # Create problem
    problem = MFGProblem(geometry=geometry, T=1.0, Nt=10, sigma=0.1)

    print(f"\nProblem dimension: {problem.dimension}")
    print(f"Problem domain type: {problem.domain_type}")

    print("\n✓ Hypersphere working correctly")


def demo_maze_geometry():
    """Demonstrate MazeGeometry."""
    print("\n" + "=" * 70)
    print("5. MazeGeometry (Maze-Based Grid)")
    print("=" * 70)

    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry.mazes import PerfectMazeGenerator

    # Generate 5×5 maze
    maze_gen = PerfectMazeGenerator(rows=5, cols=5)
    geometry = maze_gen.generate()

    print(f"\nGeometry: {geometry.geometry_type}")
    print(f"Dimension: {geometry.dimension}")
    print(f"Maze size: {geometry.rows}×{geometry.cols}")
    print(f"Total cells: {geometry.num_spatial_points}")

    # Create problem
    problem = MFGProblem(geometry=geometry, T=1.0, Nt=10, sigma=0.1)

    print(f"\nProblem dimension: {problem.dimension}")
    print(f"Problem domain type: {problem.domain_type}")

    print("\n✓ MazeGeometry working correctly")


def demo_high_dimensional():
    """Demonstrate high-dimensional grid (4D)."""
    print("\n" + "=" * 70)
    print("6. High-Dimensional Grid (4D TensorProductGrid)")
    print("=" * 70)

    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry import TensorProductGrid

    # Create 4D grid (will emit performance warning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geometry = TensorProductGrid(
            dimension=4,
            bounds=[(0.0, 1.0)] * 4,
            num_points=[5, 5, 5, 5],  # 625 points
        )

    print(f"\nGeometry: {geometry.geometry_type}")
    print(f"Dimension: {geometry.dimension}")
    print(f"Total points: {geometry.num_spatial_points:,}")
    print("\nNote: High-dimensional grids (d > 3) have O(N^d) complexity.")
    print("Consider meshfree methods or implicit domains for d > 3.")

    # Create problem
    problem = MFGProblem(geometry=geometry, T=1.0, Nt=10, sigma=0.1)

    print(f"\nProblem dimension: {problem.dimension}")

    print("\n✓ High-dimensional grid working correctly")


def demo_geometry_reuse():
    """Demonstrate reusing geometry for multiple problems."""
    print("\n" + "=" * 70)
    print("7. Geometry Reuse Pattern")
    print("=" * 70)

    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry import TensorProductGrid

    # Create geometry once
    geometry = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[51, 51])

    print(f"\nCreated geometry: {geometry.num_spatial_points} points")

    # Use for multiple problems with different parameters
    problem1 = MFGProblem(geometry=geometry, T=1.0, Nt=10, sigma=0.1)
    problem2 = MFGProblem(geometry=geometry, T=2.0, Nt=20, sigma=0.2)
    problem3 = MFGProblem(geometry=geometry, T=0.5, Nt=5, sigma=0.05)

    print("\nCreated 3 problems with same geometry:")
    print(f"  Problem 1: T={problem1.T}, Nt={problem1.Nt}, σ={problem1.sigma}")
    print(f"  Problem 2: T={problem2.T}, Nt={problem2.Nt}, σ={problem2.sigma}")
    print(f"  Problem 3: T={problem3.T}, Nt={problem3.Nt}, σ={problem3.sigma}")

    print("\n✓ Geometry reuse working correctly")


def demo_geometry_refinement():
    """Demonstrate geometry refinement."""
    print("\n" + "=" * 70)
    print("8. Geometry Refinement Pattern")
    print("=" * 70)

    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry import TensorProductGrid

    # Create coarse grid
    coarse = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[11, 11])

    print(f"\nCoarse grid: {coarse.num_spatial_points} points")

    # Refine by factor of 2
    medium = coarse.refine(factor=2)  # 21×21
    fine = medium.refine(factor=2)  # 41×41

    print(f"Medium grid (2× refined): {medium.num_spatial_points} points")
    print(f"Fine grid (4× refined): {fine.num_spatial_points} points")

    # Create problems at different resolutions
    problem_coarse = MFGProblem(geometry=coarse, T=1.0, Nt=10, sigma=0.1)
    problem_fine = MFGProblem(geometry=fine, T=1.0, Nt=10, sigma=0.1)

    print(f"\nCoarse problem: {problem_coarse.num_spatial_points} spatial points")
    print(f"Fine problem: {problem_fine.num_spatial_points} spatial points")

    print("\n✓ Geometry refinement working correctly")


def main():
    """Run all demos."""
    print("=" * 70)
    print("GEOMETRY-FIRST API DEMONSTRATION (v0.10.0)")
    print("=" * 70)
    print("\nThis demo shows the new recommended pattern for creating MFG problems.")
    print("The geometry-first API provides:")
    print("  - Better type safety and validation")
    print("  - Geometry reusability")
    print("  - Clearer separation of concerns")
    print("  - Support for diverse geometry types")

    try:
        demo_tensor_product_grid()
        demo_domain_1d()
        demo_hyperrectangle()
        demo_hypersphere()
        demo_maze_geometry()
        demo_high_dimensional()
        demo_geometry_reuse()
        demo_geometry_refinement()

        print("\n" + "=" * 70)
        print("✓ ALL GEOMETRY-FIRST API PATTERNS DEMONSTRATED SUCCESSFULLY!")
        print("=" * 70)
        print("\nSee docs/migration/GEOMETRY_FIRST_API_GUIDE.md for more details.")

    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
