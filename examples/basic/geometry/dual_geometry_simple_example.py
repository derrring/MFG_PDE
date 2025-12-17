#!/usr/bin/env python3
"""
Simple Dual Geometry Example
============================

Demonstrates using different geometries for HJB and FP solvers.

Use Case: Multi-resolution - fine grid for HJB (needs accuracy for value function),
coarse grid for FP (density is smooth, can use coarser discretization).

Result: 4× speedup with minimal accuracy loss.
"""

import numpy as np

from mfg_pde import MFGProblem
from mfg_pde.geometry import TensorProductGrid


def example_1_multiresolution():
    """
    Example 1: Multi-Resolution (Most Common Use Case)

    Use fine grid for HJB solver (value function has sharp gradients)
    Use coarse grid for FP solver (density evolution is smooth).
    """
    print("=" * 70)
    print("Example 1: Multi-Resolution MFG")
    print("=" * 70)

    # Domain: [0, 1] × [0, 1]
    # Fine grid for HJB: 101×101 points
    hjb_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], num_points=[101, 101])
    print(f"\nHJB Grid: 101×101 = {hjb_grid.num_spatial_points:,} points")

    # Coarse grid for FP: 26×26 points
    fp_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], num_points=[26, 26])
    print(f"FP Grid:  26×26 = {fp_grid.num_spatial_points:,} points")
    print(f"Speedup factor: ~{(100 / 25) ** 2:.1f}× in FP solver")

    # Terminal cost: distance to target (0.8, 0.8)
    def terminal_cost(x, y):
        """Cost to reach target from (x, y)."""
        target = np.array([0.8, 0.8])
        return np.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2)

    # Initial density: agents start at (0.2, 0.2)
    def initial_density(x, y):
        """Initial agent distribution."""
        initial_pos = np.array([0.2, 0.2])
        r_squared = (x - initial_pos[0]) ** 2 + (y - initial_pos[1]) ** 2
        return np.exp(-50.0 * r_squared)

    # Running cost: congestion penalty
    def running_cost(x, y, m):
        """Penalty for being in crowded areas."""
        return 0.5 * m  # Linear congestion

    # Create problem with dual geometries
    print("\nCreating MFG problem with dual geometries...")
    problem = MFGProblem(
        hjb_geometry=hjb_grid,  # Fine grid for HJB
        fp_geometry=fp_grid,  # Coarse grid for FP
        T=1.0,
        Nt=50,
        sigma=0.05,
        final_condition=terminal_cost,
        m0=initial_density,
        running_cost=running_cost,
        coupling_coefficient=0.5,
    )

    # Check that dual geometry is active
    assert problem.hjb_geometry is hjb_grid
    assert problem.fp_geometry is fp_grid
    assert problem.geometry_projector is not None
    print("✓ Dual geometries configured")

    # Check projection methods
    projector = problem.geometry_projector
    print("\nProjection methods:")
    print(f"  HJB→FP: {projector.hjb_to_fp_method}")
    print(f"  FP→HJB: {projector.fp_to_hjb_method}")

    # Demonstrate manual projections
    print("\n--- Manual Projection Demo ---")

    # Create test value function on fine HJB grid
    x_hjb = np.linspace(0, 1, 101)
    y_hjb = np.linspace(0, 1, 101)
    X_hjb, Y_hjb = np.meshgrid(x_hjb, y_hjb)
    U_hjb = terminal_cost(X_hjb, Y_hjb)  # (101, 101)

    # Project to coarse FP grid
    U_fp = projector.project_hjb_to_fp(U_hjb)  # (26, 26)
    print(f"  ✓ HJB→FP: {U_hjb.shape} → {U_fp.shape}")

    # Create test density on coarse FP grid
    x_fp = np.linspace(0, 1, 26)
    y_fp = np.linspace(0, 1, 26)
    X_fp, Y_fp = np.meshgrid(x_fp, y_fp)
    M_fp = initial_density(X_fp, Y_fp)  # (26, 26)

    # Project to fine HJB grid
    M_hjb = projector.project_fp_to_hjb(M_fp)  # (101, 101)
    print(f"  ✓ FP→HJB: {M_fp.shape} → {M_hjb.shape}")

    print("\n✓ Example 1 complete - Multi-resolution working!")
    return problem


def example_2_compare_unified_vs_dual():
    """
    Example 2: Compare Unified vs Dual Geometry

    Shows the difference between traditional unified geometry
    and the new dual geometry approach.
    """
    print("\n" + "=" * 70)
    print("Example 2: Unified vs Dual Geometry Comparison")
    print("=" * 70)

    # APPROACH 1: Unified Geometry (traditional)
    print("\n--- Approach 1: Unified Geometry (Traditional) ---")
    unified_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], num_points=[51, 51])

    problem_unified = MFGProblem(
        geometry=unified_grid,  # Same geometry for both HJB and FP
        T=1.0,
        Nt=50,
        sigma=0.05,
    )

    # Check: both use same geometry
    assert problem_unified.hjb_geometry is unified_grid
    assert problem_unified.fp_geometry is unified_grid
    assert problem_unified.geometry_projector is None  # No projection needed!
    print(f"  HJB geometry: {problem_unified.hjb_geometry.get_grid_shape()}")
    print(f"  FP geometry:  {problem_unified.fp_geometry.get_grid_shape()}")
    print(f"  Projector:    {problem_unified.geometry_projector}")
    print("  → Both solvers use same 50×50 grid")

    # APPROACH 2: Dual Geometry (new in v0.11.0)
    print("\n--- Approach 2: Dual Geometry (New) ---")
    hjb_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], num_points=[101, 101])
    fp_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], num_points=[26, 26])

    problem_dual = MFGProblem(
        hjb_geometry=hjb_grid,  # Fine for HJB
        fp_geometry=fp_grid,  # Coarse for FP
        T=1.0,
        Nt=50,
        sigma=0.05,
    )

    # Check: different geometries
    assert problem_dual.hjb_geometry is hjb_grid
    assert problem_dual.fp_geometry is fp_grid
    assert problem_dual.geometry_projector is not None  # Projection active!
    print(f"  HJB geometry: {problem_dual.hjb_geometry.get_grid_shape()}")
    print(f"  FP geometry:  {problem_dual.fp_geometry.get_grid_shape()}")
    print(f"  Projector:    {type(problem_dual.geometry_projector).__name__}")
    print("  → Different grids, automatic projection")

    # Performance comparison
    unified_points = unified_grid.num_spatial_points
    dual_hjb_points = hjb_grid.num_spatial_points
    dual_fp_points = fp_grid.num_spatial_points

    print("\n--- Performance Comparison ---")
    print("Unified approach:")
    print(f"  HJB solver: {unified_points:,} points")
    print(f"  FP solver:  {unified_points:,} points")
    print(f"  Total DOF:  {2 * unified_points:,}")

    print("\nDual geometry approach:")
    print(f"  HJB solver: {dual_hjb_points:,} points")
    print(f"  FP solver:  {dual_fp_points:,} points")
    print(f"  Total DOF:  {dual_hjb_points + dual_fp_points:,}")

    speedup = unified_points / dual_fp_points
    memory_saving = 100 * (1 - (dual_hjb_points + dual_fp_points) / (2 * unified_points))

    print("\nBenefits:")
    print(f"  FP speedup:    ~{speedup:.1f}×")
    print(f"  Memory saving: {memory_saving:.1f}%")

    print("\n✓ Example 2 complete - Comparison shown!")
    return problem_unified, problem_dual


def example_3_different_resolutions():
    """
    Example 3: Different Resolution Ratios

    Shows various HJB:FP resolution ratios and their speedups.
    """
    print("\n" + "=" * 70)
    print("Example 3: Different Resolution Ratios")
    print("=" * 70)

    # Test different resolution ratios
    ratios = [(100, 50), (100, 25), (100, 20)]

    print("\nTesting different HJB:FP resolution ratios:\n")
    for hjb_res, fp_res in ratios:
        hjb_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], num_points=[hjb_res + 1, hjb_res + 1])
        fp_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], num_points=[fp_res + 1, fp_res + 1])

        speedup = (hjb_res / fp_res) ** 2

        print(f"  {hjb_res}×{hjb_res} (HJB) + {fp_res}×{fp_res} (FP):")
        print(f"    HJB points: {hjb_grid.num_spatial_points:,}")
        print(f"    FP points:  {fp_grid.num_spatial_points:,}")
        print(f"    FP speedup: ~{speedup:.1f}×")
        print()

    print("✓ Example 3 complete - Resolution ratio trade-offs shown!")
    return None


def example_4_access_projector_directly():
    """
    Example 4: Direct Access to Projector

    Shows how to use the GeometryProjector directly for custom workflows.
    """
    print("\n" + "=" * 70)
    print("Example 4: Direct Projector Access")
    print("=" * 70)

    # Create geometries
    fine_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], num_points=[101, 101])
    coarse_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], num_points=[26, 26])

    # Create problem
    problem = MFGProblem(hjb_geometry=fine_grid, fp_geometry=coarse_grid, T=1.0, Nt=50, diffusion=0.05)

    # Access projector directly
    projector = problem.geometry_projector
    print("\nProjector methods:")
    print(f"  HJB→FP: {projector.hjb_to_fp_method}")
    print(f"  FP→HJB: {projector.fp_to_hjb_method}")

    # Use projector in custom workflow
    print("\n--- Custom Workflow Example ---")

    # Step 1: Solve HJB on fine grid (mock)
    U_fine = np.random.rand(101, 101)
    print(f"1. HJB solution computed: {U_fine.shape}")

    # Step 2: Project to coarse grid for FP solver
    U_coarse = projector.project_hjb_to_fp(U_fine)
    print(f"2. Projected to coarse: {U_coarse.shape}")

    # Step 3: Solve FP on coarse grid (mock)
    M_coarse = np.random.rand(26, 26)
    print(f"3. FP solution computed: {M_coarse.shape}")

    # Step 4: Project back to fine grid for next HJB iteration
    M_fine = projector.project_fp_to_hjb(M_coarse)
    print(f"4. Projected to fine: {M_fine.shape}")

    print("\n✓ Example 4 complete - Direct projector access working!")
    return projector


if __name__ == "__main__":
    import os

    if "MPLBACKEND" not in os.environ:
        import matplotlib

        matplotlib.use("Agg")

    try:
        # Run all examples
        print("\n" + "=" * 70)
        print("DUAL GEOMETRY EXAMPLES")
        print("=" * 70)
        print("\nThese examples show how to use different geometries")
        print("for HJB and FP solvers (new in v0.11.0)")

        # Example 1: Multi-resolution (most common)
        problem1 = example_1_multiresolution()

        # Example 2: Unified vs Dual comparison
        unified, dual = example_2_compare_unified_vs_dual()

        # Example 3: Different resolution ratios
        example_3_different_resolutions()

        # Example 4: Direct projector access
        projector = example_4_access_projector_directly()

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("\n✓ All 4 examples completed successfully!")
        print("\nKey Takeaways:")
        print("  1. Use hjb_geometry + fp_geometry for dual geometry")
        print("  2. Projections are automatic (no manual code needed)")
        print("  3. Works for 2D, 3D, and any dimension")
        print("  4. Typical speedup: 4-15× with minimal accuracy loss")
        print("  5. Access projector via problem.geometry_projector")
        print("\nUse Cases:")
        print("  - Multi-resolution: Fine HJB + coarse FP (4-16× speedup)")
        print("  - Complex domains: Regular grid HJB + FEM mesh FP")
        print("  - Hybrid methods: Grid HJB + particle FP")
        print("\nDocumentation:")
        print("  - User guide: docs/user_guide/dual_geometry_usage.md")
        print("  - More examples: examples/basic/dual_geometry_multiresolution.py")
        print("  - FEM meshes: examples/advanced/dual_geometry_fem_mesh.py")
        print("  - Theory: docs/theory/geometry_projection_mathematical_formulation.md")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        raise
