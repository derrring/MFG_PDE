"""
SDF-Based Mixed Boundary Conditions Demo.

Demonstrates the new SDF-based boundary condition specification for general domains
(non-rectangular). Shows how to specify BCs using:

1. SDF regions: Define BC segments using signed distance functions
2. Normal directions: Match boundary segments by outward normal direction
3. Combined matching: Use both SDF regions and normal directions

Examples include:
- Circular domain with exit at top
- L-shaped domain (Lipschitz boundary)
- Domain with obstacle

Usage:
    python examples/basic/sdf_mixed_bc_demo.py
"""

import numpy as np

from mfg_pde.geometry.boundary import (
    BCSegment,
    BCType,
    MixedBoundaryConditions,
)


def demo_circular_domain():
    """
    Circular domain with exit at top.

    Domain: Circle of radius 5 centered at origin
    BC: Dirichlet (u=0) at top (exit), Neumann elsewhere (walls)
    """
    print("=" * 70)
    print("Demo 1: Circular Domain with Exit at Top")
    print("=" * 70)

    # Define circular domain SDF
    # phi(x) = ||x|| - r  (negative inside, positive outside)
    def circle_sdf(x):
        return np.linalg.norm(np.asarray(x) - np.array([0.0, 0.0])) - 5.0

    # Exit segment: top of circle (normal pointing up)
    exit_bc = BCSegment(
        name="exit",
        bc_type=BCType.DIRICHLET,
        value=0.0,
        normal_direction=np.array([0.0, 1.0]),  # Upward normal
        normal_tolerance=0.7,  # cos(45 deg) - matches top 90 deg arc
        priority=1,
    )

    # Walls: everywhere else on boundary
    wall_bc = BCSegment(
        name="walls",
        bc_type=BCType.NEUMANN,
        value=0.0,
        priority=0,
    )

    mixed_bc = MixedBoundaryConditions(
        dimension=2,
        segments=[exit_bc, wall_bc],
        domain_sdf=circle_sdf,
    )

    print(f"\n{mixed_bc}\n")

    # Test points on boundary
    test_points = [
        (np.array([0.0, 5.0]), "top (y=5)"),
        (np.array([5.0, 0.0]), "right (x=5)"),
        (np.array([0.0, -5.0]), "bottom (y=-5)"),
        (np.array([-5.0, 0.0]), "left (x=-5)"),
        (np.array([3.54, 3.54]), "top-right (~45 deg)"),
        (np.array([-3.54, 3.54]), "top-left (~135 deg)"),
    ]

    print("Boundary point BC assignment:")
    print("-" * 60)
    for point, desc in test_points:
        bc = mixed_bc.get_bc_at_point(point)
        normal = mixed_bc.get_outward_normal(point)
        normal_str = f"({normal[0]:.2f}, {normal[1]:.2f})" if normal is not None else "None"
        print(f"  {desc:25s} -> {bc.name:10s} (normal: {normal_str})")

    print()


def demo_l_shaped_domain():
    """
    L-shaped domain (Lipschitz boundary with re-entrant corner).

    Domain: [0,2]^2 with [1,2]x[1,2] cut out
    BC: Different conditions on different faces
    """
    print("=" * 70)
    print("Demo 2: L-Shaped Domain (Lipschitz Boundary)")
    print("=" * 70)

    # L-shaped domain SDF using CSG difference
    def l_shape_sdf(x):
        x = np.asarray(x)
        # Large box [0,2]^2
        d_large = np.maximum(
            np.maximum(-x[0], x[0] - 2.0),
            np.maximum(-x[1], x[1] - 2.0),
        )
        # Cutout box [1,2]^2
        d_cutout = np.maximum(
            np.maximum(x[0] - 1.0, 2.0 - x[0]),
            np.maximum(x[1] - 1.0, 2.0 - x[1]),
        )
        d_cutout = -d_cutout  # Inside cutout
        # L-shape = large box minus cutout = intersection(large, complement(cutout))
        return np.maximum(d_large, d_cutout)

    # Different BC on different faces
    left_bc = BCSegment(
        name="left_inlet",
        bc_type=BCType.DIRICHLET,
        value=1.0,
        normal_direction=np.array([-1.0, 0.0]),  # Pointing left
        normal_tolerance=0.9,
        priority=1,
    )

    bottom_bc = BCSegment(
        name="bottom_wall",
        bc_type=BCType.NEUMANN,
        value=0.0,
        normal_direction=np.array([0.0, -1.0]),  # Pointing down
        normal_tolerance=0.9,
        priority=1,
    )

    # Special handling for re-entrant corner region
    corner_bc = BCSegment(
        name="corner",
        bc_type=BCType.NEUMANN,
        value=0.0,
        sdf_region=lambda x: np.linalg.norm(np.asarray(x) - np.array([1.0, 1.0])) - 0.2,
        priority=2,  # Highest priority
    )

    default_bc = BCSegment(
        name="other_walls",
        bc_type=BCType.NEUMANN,
        value=0.0,
        priority=0,
    )

    mixed_bc = MixedBoundaryConditions(
        dimension=2,
        segments=[corner_bc, left_bc, bottom_bc, default_bc],
        domain_sdf=l_shape_sdf,
        corner_strategy="priority",
    )

    print(f"\n{mixed_bc}\n")

    # Test points
    test_points = [
        (np.array([0.0, 0.5]), "left edge (x=0)"),
        (np.array([0.5, 0.0]), "bottom edge (y=0)"),
        (np.array([1.0, 1.0]), "re-entrant corner"),
        (np.array([2.0, 0.5]), "right edge (x=2)"),
        (np.array([0.5, 2.0]), "top edge (y=2)"),
    ]

    print("Boundary point BC assignment:")
    print("-" * 60)
    for point, desc in test_points:
        # Check if point is on boundary
        if mixed_bc.is_on_boundary(point, tolerance=0.1):
            bc = mixed_bc.get_bc_at_point(point)
            print(f"  {desc:25s} -> {bc.name}")
        else:
            print(f"  {desc:25s} -> (not on boundary)")

    print()


def demo_domain_with_obstacle():
    """
    Rectangular room with circular obstacle.

    Domain: [0,10]^2 with circular obstacle at center
    BC: Exit on right wall, reflecting elsewhere, including obstacle boundary
    """
    print("=" * 70)
    print("Demo 3: Room with Circular Obstacle")
    print("=" * 70)

    # Room with obstacle SDF
    def room_with_obstacle_sdf(x):
        x = np.asarray(x)
        # Room [0,10]^2
        d_room = np.maximum(
            np.maximum(-x[0], x[0] - 10.0),
            np.maximum(-x[1], x[1] - 10.0),
        )
        # Circular obstacle at center, radius 2
        d_obstacle = np.linalg.norm(x - np.array([5.0, 5.0])) - 2.0
        # Room minus obstacle
        return np.maximum(d_room, -d_obstacle)

    # Exit on right wall (y in [4, 6])
    exit_bc = BCSegment(
        name="exit",
        bc_type=BCType.DIRICHLET,
        value=0.0,
        normal_direction=np.array([1.0, 0.0]),  # Right-pointing normal
        normal_tolerance=0.9,
        sdf_region=lambda x: np.maximum(4.0 - x[1], x[1] - 6.0),  # y in [4, 6]
        priority=2,
    )

    # Obstacle boundary
    obstacle_bc = BCSegment(
        name="obstacle",
        bc_type=BCType.NEUMANN,
        value=0.0,
        # Match points near the obstacle (within 0.5 of obstacle center region)
        sdf_region=lambda x: np.linalg.norm(np.asarray(x) - np.array([5.0, 5.0])) - 2.5,
        priority=1,
    )

    # All other walls
    wall_bc = BCSegment(
        name="walls",
        bc_type=BCType.NEUMANN,
        value=0.0,
        priority=0,
    )

    mixed_bc = MixedBoundaryConditions(
        dimension=2,
        segments=[exit_bc, obstacle_bc, wall_bc],
        domain_sdf=room_with_obstacle_sdf,
    )

    print(f"\n{mixed_bc}\n")

    # Test points
    test_points = [
        (np.array([10.0, 5.0]), "exit center (10, 5)"),
        (np.array([10.0, 3.0]), "right wall below exit (10, 3)"),
        (np.array([0.0, 5.0]), "left wall (0, 5)"),
        (np.array([5.0, 0.0]), "bottom wall (5, 0)"),
        (np.array([7.0, 5.0]), "near obstacle, right side"),
        (np.array([5.0, 7.0]), "near obstacle, top side"),
    ]

    print("Boundary point BC assignment:")
    print("-" * 60)
    for point, desc in test_points:
        if mixed_bc.is_on_boundary(point, tolerance=0.1):
            bc = mixed_bc.get_bc_at_point(point)
            print(f"  {desc:35s} -> {bc.name}")
        else:
            print(f"  {desc:35s} -> (not on boundary)")

    print()


def demo_validation():
    """Demonstrate validation of SDF-based configurations."""
    print("=" * 70)
    print("Demo 4: Configuration Validation")
    print("=" * 70)

    # Valid configuration
    print("\n1. Valid configuration:")
    valid_segment = BCSegment(
        name="exit",
        bc_type=BCType.DIRICHLET,
        normal_direction=np.array([0.0, 1.0]),
    )
    mixed_bc = MixedBoundaryConditions(
        dimension=2,
        segments=[valid_segment],
        domain_sdf=lambda x: np.linalg.norm(x) - 1.0,
    )
    is_valid, warnings = mixed_bc.validate()
    print(f"   Valid: {is_valid}, Warnings: {warnings}")

    # Invalid: wrong normal_direction dimension
    print("\n2. Invalid: wrong normal_direction dimension (3D in 2D problem):")
    bad_segment = BCSegment(
        name="bad",
        bc_type=BCType.DIRICHLET,
        normal_direction=np.array([1.0, 0.0, 0.0]),  # 3D vector in 2D problem
    )
    mixed_bc_bad = MixedBoundaryConditions(
        dimension=2,
        segments=[bad_segment],
        domain_sdf=lambda x: np.linalg.norm(x) - 1.0,
    )
    is_valid, warnings = mixed_bc_bad.validate()
    print(f"   Valid: {is_valid}, Warnings: {warnings}")

    # Invalid: no domain specification
    print("\n3. Invalid: neither domain_bounds nor domain_sdf set:")
    mixed_bc_no_domain = MixedBoundaryConditions(
        dimension=2,
        segments=[valid_segment],
    )
    is_valid, warnings = mixed_bc_no_domain.validate()
    print(f"   Valid: {is_valid}, Warnings: {warnings}")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SDF-Based Mixed Boundary Conditions Demonstration")
    print("=" * 70 + "\n")

    demo_circular_domain()
    demo_l_shaped_domain()
    demo_domain_with_obstacle()
    demo_validation()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Key features demonstrated:
1. SDF-based domain definition (circular, L-shaped, room with obstacle)
2. Normal-direction based segment matching (exit at top of circle)
3. SDF region constraints (corner handling, obstacle boundary)
4. Combined matching (normal + region for exit on specific wall segment)
5. Priority-based segment resolution
6. Configuration validation

Next steps:
1. Integrate with HJB and FP solvers
2. Add visualization of BC assignments on domain boundary
3. Implement corner mollification for smoother gradient computation
""")
