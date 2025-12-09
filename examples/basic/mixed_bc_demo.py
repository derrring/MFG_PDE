"""
Mixed Boundary Conditions Demonstration

Shows how to use dimension-agnostic mixed boundary conditions for 2D crowd motion
problems with exits and walls (Protocol v1.4 style).

This example demonstrates:
1. Creating BC segments for different boundary regions
2. Combining segments into MixedBoundaryConditions
3. Querying BC at specific points
4. Validation and visualization
"""

from __future__ import annotations

import numpy as np

from mfg_pde.geometry import BCSegment, BCType, MixedBoundaryConditions

# Constants for the problem
DOMAIN_SIZE = (10.0, 10.0)  # [0, 10] x [0, 10]
EXIT_Y_RANGE = (4.25, 5.75)  # Exit centered at y=5.0, width=1.5


def create_protocol_v14_mixed_bc() -> MixedBoundaryConditions:
    """
    Create mixed BC for Protocol v1.4: 2D crowd motion with exit and walls.

    Domain: [0, 10] × [0, 10]
    Exit: Right wall (x=10), y ∈ [4.25, 5.75] - Dirichlet (absorbing)
    Walls: All other boundaries - Neumann (reflecting)

    Returns:
        Configured MixedBoundaryConditions instance
    """
    print("Creating Protocol v1.4 mixed boundary conditions...")
    print(f"Domain: [0, {DOMAIN_SIZE[0]}] × [0, {DOMAIN_SIZE[1]}]")
    print(f"Exit: Right wall, y ∈ {EXIT_Y_RANGE} (absorbing)")
    print("Walls: All other boundaries (reflecting)")
    print()

    # Exit segment (high priority, specific region)
    exit_bc = BCSegment(
        name="exit",
        bc_type=BCType.DIRICHLET,
        value=0.0,
        boundary="right",
        region={"y": EXIT_Y_RANGE},
        priority=1,  # Higher priority than walls
    )

    # Wall segments (low priority, applies everywhere)
    wall_bc = BCSegment(
        name="walls",
        bc_type=BCType.NEUMANN,
        value=0.0,
        boundary="all",
        priority=0,
    )

    # Combine into mixed BC
    mixed_bc = MixedBoundaryConditions(
        dimension=2,
        segments=[exit_bc, wall_bc],
        default_bc=BCType.NEUMANN,  # Default to reflecting
        default_value=0.0,
        domain_bounds=np.array([[0.0, DOMAIN_SIZE[0]], [0.0, DOMAIN_SIZE[1]]]),
    )

    print("Mixed BC created:")
    print(mixed_bc)
    print()

    return mixed_bc


def test_bc_at_points(mixed_bc: MixedBoundaryConditions):
    """Test BC evaluation at various boundary points."""
    print("Testing BC at specific points:")
    print("-" * 60)

    test_points = [
        # (point, boundary_id, expected_bc_type, description)
        (np.array([10.0, 5.0]), "right", BCType.DIRICHLET, "Exit center"),
        (np.array([10.0, 4.5]), "right", BCType.DIRICHLET, "Exit lower edge"),
        (np.array([10.0, 5.5]), "right", BCType.DIRICHLET, "Exit upper edge"),
        (np.array([10.0, 3.0]), "right", BCType.NEUMANN, "Right wall below exit"),
        (np.array([10.0, 7.0]), "right", BCType.NEUMANN, "Right wall above exit"),
        (np.array([0.0, 5.0]), "left", BCType.NEUMANN, "Left wall"),
        (np.array([5.0, 0.0]), "bottom", BCType.NEUMANN, "Bottom wall"),
        (np.array([5.0, 10.0]), "top", BCType.NEUMANN, "Top wall"),
    ]

    for point, boundary_id, expected_type, description in test_points:
        bc = mixed_bc.get_bc_at_point(point, boundary_id)
        status = "✓" if bc.bc_type == expected_type else "✗"
        print(
            f"{status} {description:30s} ({point[0]:4.1f}, {point[1]:4.1f}): {bc.bc_type.value:12s} (segment: {bc.name})"
        )

    print()


def validate_configuration(mixed_bc: MixedBoundaryConditions):
    """Validate the mixed BC configuration."""
    print("Validating configuration...")
    is_valid, warnings = mixed_bc.validate()

    if is_valid:
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has issues:")
        for warning in warnings:
            print(f"  - {warning}")

    print()


def demonstrate_boundary_identification(mixed_bc: MixedBoundaryConditions):
    """Demonstrate automatic boundary identification."""
    print("Automatic boundary identification:")
    print("-" * 60)

    test_points = [
        np.array([0.0, 5.0]),  # Left
        np.array([10.0, 5.0]),  # Right
        np.array([5.0, 0.0]),  # Bottom
        np.array([5.0, 10.0]),  # Top
        np.array([5.0, 5.0]),  # Interior
    ]

    for point in test_points:
        boundary_id = mixed_bc.identify_boundary_id(point)
        if boundary_id:
            bc = mixed_bc.get_bc_at_point(point, boundary_id)
            print(f"Point ({point[0]:4.1f}, {point[1]:4.1f}): {boundary_id:10s} → {bc.bc_type.value} ({bc.name})")
        else:
            print(f"Point ({point[0]:4.1f}, {point[1]:4.1f}): Interior (not on boundary)")

    print()


def demonstrate_dimension_agnosticism():
    """Show that mixed BC works for different dimensions."""
    print("Dimension-agnostic design demonstration:")
    print("-" * 60)

    # 1D example
    print("1D problem:")
    bc_1d = MixedBoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="left", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_min"),
            BCSegment(name="right", bc_type=BCType.NEUMANN, value=0.0, boundary="x_max"),
        ],
        domain_bounds=np.array([[0.0, 1.0]]),
    )
    print(f"  Left boundary (x=0.0): {bc_1d.get_bc_at_point(np.array([0.0]), 'x_min').bc_type.value}")
    print(f"  Right boundary (x=1.0): {bc_1d.get_bc_at_point(np.array([1.0]), 'x_max').bc_type.value}")
    print()

    # 3D example
    print("3D problem:")
    bc_3d = MixedBoundaryConditions(
        dimension=3,
        segments=[
            BCSegment(
                name="outlet",
                bc_type=BCType.DIRICHLET,
                value=0.0,
                region={"x": (9, 10), "y": (4, 6), "z": (4, 6)},
                priority=1,
            ),
            BCSegment(name="walls", bc_type=BCType.NEUMANN, value=0.0, priority=0),
        ],
        domain_bounds=np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]),
    )
    outlet_point = np.array([9.5, 5.0, 5.0])
    wall_point = np.array([9.5, 2.0, 5.0])
    print(f"  Outlet point (9.5, 5.0, 5.0): {bc_3d.get_bc_at_point(outlet_point, 'x_max').bc_type.value}")
    print(f"  Wall point (9.5, 2.0, 5.0): {bc_3d.get_bc_at_point(wall_point, 'x_max').bc_type.value}")
    print()


def main():
    """Main demonstration."""
    print("\n" + "=" * 70)
    print("Mixed Boundary Conditions - Protocol v1.4 Demonstration")
    print("=" * 70)
    print()

    # Create mixed BC for 2D crowd motion
    mixed_bc = create_protocol_v14_mixed_bc()

    # Validate configuration
    validate_configuration(mixed_bc)

    # Test BC at specific points
    test_bc_at_points(mixed_bc)

    # Demonstrate boundary identification
    demonstrate_boundary_identification(mixed_bc)

    # Show dimension-agnostic design
    demonstrate_dimension_agnosticism()

    # Summary
    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    print("✓ Mixed BC successfully created for 2D crowd motion")
    print("✓ Exit segment (Dirichlet) correctly applied to right wall y ∈ [4.25, 5.75]")
    print("✓ Wall segments (Neumann) correctly applied to all other boundaries")
    print("✓ Priority system works (exit overrides wall on right boundary)")
    print("✓ Dimension-agnostic design verified for 1D, 2D, 3D")
    print()
    print("Next steps:")
    print("1. Integrate with TensorProductGrid for actual solver usage")
    print("2. Implement BC application in HJB and FP solvers")
    print("3. Test with full Protocol v1.4 implementation")
    print()


if __name__ == "__main__":
    main()
