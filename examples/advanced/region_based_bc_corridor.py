"""
Region-Based Boundary Conditions: 2D Corridor Flow

Demonstrates region-based BC specification for complex geometries with:
1. Inlet region (predicate-based, partial left boundary)
2. Outlet region (predicate-based, partial right boundary)
3. Obstacle region (interior circular obstacle with no-flux BC)
4. Wall regions (top and bottom boundaries)
5. Priority resolution for overlapping regions

This example shows the modern approach for complex BC specifications
where boundary identifiers alone are insufficient.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import (
    BCSegment,
    BCType,
    FDMApplicator,
    mixed_bc_from_regions,
)

# Corridor geometry
CORRIDOR_LENGTH = 4.0  # [0, 4] in x
CORRIDOR_HEIGHT = 1.0  # [0, 1] in y
INLET_Y_RANGE = (0.3, 0.7)  # Inlet centered, 40% of height
OUTLET_Y_RANGE = (0.2, 0.8)  # Outlet wider, 60% of height
OBSTACLE_CENTER = (2.0, 0.5)  # Circular obstacle in middle
OBSTACLE_RADIUS = 0.2


def create_corridor_geometry() -> TensorProductGrid:
    """Create 2D corridor geometry with marked regions."""
    print("Creating 2D corridor geometry...")
    print(f"  Domain: [0, {CORRIDOR_LENGTH}] × [0, {CORRIDOR_HEIGHT}]")

    # Create grid
    geometry = TensorProductGrid(
        bounds=[(0, CORRIDOR_LENGTH), (0, CORRIDOR_HEIGHT)],
        Nx_points=[81, 21],  # Higher resolution in x
    )

    print(f"  Grid: {geometry.Nx_points} points")
    print()

    return geometry


def mark_corridor_regions(geometry: TensorProductGrid):
    """Mark all regions on the corridor geometry."""
    print("Marking regions...")

    # Inlet: Left boundary, partial (centered vertically)
    def inlet_predicate(x):
        """Points at left boundary within y-range."""
        return (x[:, 0] < 0.01) & (x[:, 1] >= INLET_Y_RANGE[0]) & (x[:, 1] <= INLET_Y_RANGE[1])

    geometry.mark_region("inlet", predicate=inlet_predicate)
    print(f"  ✓ Inlet region marked: x=0, y ∈ [{INLET_Y_RANGE[0]}, {INLET_Y_RANGE[1]}]")

    # Outlet: Right boundary, partial (wider than inlet)
    def outlet_predicate(x):
        """Points at right boundary within y-range."""
        return (x[:, 0] > CORRIDOR_LENGTH - 0.01) & (x[:, 1] >= OUTLET_Y_RANGE[0]) & (x[:, 1] <= OUTLET_Y_RANGE[1])

    geometry.mark_region("outlet", predicate=outlet_predicate)
    print(f"  ✓ Outlet region marked: x={CORRIDOR_LENGTH}, y ∈ [{OUTLET_Y_RANGE[0]}, {OUTLET_Y_RANGE[1]}]")

    # Obstacle: Circular region in interior (for demonstration)
    def obstacle_predicate(x):
        """Points within circular obstacle."""
        dx = x[:, 0] - OBSTACLE_CENTER[0]
        dy = x[:, 1] - OBSTACLE_CENTER[1]
        return (dx**2 + dy**2) <= OBSTACLE_RADIUS**2

    geometry.mark_region("obstacle", predicate=obstacle_predicate)
    print(f"  ✓ Obstacle region marked: circle at {OBSTACLE_CENTER}, radius={OBSTACLE_RADIUS}")

    # Walls: Top and bottom boundaries (combined using predicate)
    def walls_predicate(x):
        """Points at top or bottom boundaries."""
        return (x[:, 1] < 0.01) | (x[:, 1] > CORRIDOR_HEIGHT - 0.01)

    geometry.mark_region("walls", predicate=walls_predicate)
    print("  ✓ Wall regions marked: y=0 and y=1")

    print()


def create_region_based_bc(geometry: TensorProductGrid):
    """Create boundary conditions using region references."""
    print("Creating region-based boundary conditions...")

    bc_config = {
        # Inlet: Dirichlet BC (prescribed inflow value)
        "inlet": BCSegment(
            name="inlet_bc",
            bc_type=BCType.DIRICHLET,
            value=1.0,  # Inflow value
        ),
        # Outlet: Neumann BC (zero gradient, natural outflow)
        "outlet": BCSegment(
            name="outlet_bc",
            bc_type=BCType.NEUMANN,
            value=0.0,  # Zero gradient
        ),
        # Obstacle: No-flux BC (reflecting boundary)
        "obstacle": BCSegment(
            name="obstacle_bc",
            bc_type=BCType.NO_FLUX,
        ),
        # Walls: No-flux BC (reflecting boundaries)
        "walls": BCSegment(
            name="wall_bc",
            bc_type=BCType.NO_FLUX,
        ),
        # Default: Periodic (for any remaining boundaries)
        "default": BCSegment(
            name="default_bc",
            bc_type=BCType.PERIODIC,
        ),
    }

    # Create BC from regions
    bc = mixed_bc_from_regions(geometry, bc_config)

    print(f"  ✓ Created BC with {len(bc.segments)} segments:")
    for segment in bc.segments:
        print(f"    - {segment.name}: {segment.bc_type.value} (region: {segment.region_name})")
    print(f"  ✓ Default BC: {bc.default_bc.value}")
    print()

    return bc


def apply_and_visualize(geometry: TensorProductGrid, bc):
    """Apply BCs to a test field and visualize results."""
    print("Applying boundary conditions to test field...")

    # Create test field (uniform interior value)
    # Note: For 2D, array indexing is (y, x) so shape is (Ny, Nx)
    Nx, Ny = geometry.Nx_points  # Nx_points = [Nx, Ny]
    field = np.ones((Ny, Nx)) * 0.5

    # Apply BCs using FDMApplicator
    applicator = FDMApplicator(dimension=2)
    padded = applicator.apply(field, bc, domain_bounds=np.array(geometry.bounds), geometry=geometry)

    print("  ✓ Applied BCs")
    print(f"    Interior shape: {field.shape}")
    print(f"    Padded shape: {padded.shape} (includes ghost cells)")
    print()

    # Visualize
    visualize_results(geometry, field, padded)


def visualize_results(geometry: TensorProductGrid, field, padded):
    """Create visualization of field and BC application."""
    print("Creating visualization...")

    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get grid coordinates
    # Nx_points = [Nx, Ny], so bounds[0] is x-range, bounds[1] is y-range
    Nx, Ny = geometry.Nx_points
    x = np.linspace(geometry.bounds[0][0], geometry.bounds[0][1], Nx)
    y = np.linspace(geometry.bounds[1][0], geometry.bounds[1][1], Ny)
    X, Y = np.meshgrid(x, y)

    # Plot 1: Interior field
    ax1 = axes[0]
    im1 = ax1.contourf(X, Y, field, levels=20, cmap="viridis")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Interior Field (Before BC Application)")
    ax1.set_aspect("equal")
    plt.colorbar(im1, ax=ax1)

    # Mark regions on plot
    inlet_y = np.linspace(INLET_Y_RANGE[0], INLET_Y_RANGE[1], 20)
    ax1.plot([0] * len(inlet_y), inlet_y, "r-", linewidth=3, label="Inlet (Dirichlet)")

    outlet_y = np.linspace(OUTLET_Y_RANGE[0], OUTLET_Y_RANGE[1], 20)
    ax1.plot([CORRIDOR_LENGTH] * len(outlet_y), outlet_y, "b-", linewidth=3, label="Outlet (Neumann)")

    circle = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color="white", fill=False, linewidth=2)
    ax1.add_patch(circle)
    ax1.plot([], [], "w-", linewidth=2, label="Obstacle (No-flux)")

    ax1.legend(loc="upper right")

    # Plot 2: Padded field (with ghosts)
    ax2 = axes[1]
    im2 = ax2.contourf(padded, levels=20, cmap="viridis")
    ax2.set_xlabel("Grid index (x)")
    ax2.set_ylabel("Grid index (y)")
    ax2.set_title("Padded Field (With Ghost Cells)")
    ax2.set_aspect("equal")
    plt.colorbar(im2, ax=ax2)

    # Mark ghost cell boundaries
    Ny, Nx = field.shape
    ax2.axvline(0, color="red", linestyle="--", alpha=0.5, label="Ghost cells")
    ax2.axvline(Nx + 1, color="red", linestyle="--", alpha=0.5)
    ax2.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax2.axhline(Ny + 1, color="red", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    print("  ✓ Visualization complete")
    print()


def demonstrate_priority_resolution(geometry: TensorProductGrid):
    """Demonstrate overlapping region priority resolution."""
    print("Demonstrating priority resolution for overlapping regions...")

    # Mark overlapping regions
    geometry.mark_region("left_half", predicate=lambda x: x[:, 0] < CORRIDOR_LENGTH / 2)
    geometry.mark_region("bottom_half", predicate=lambda x: x[:, 1] < CORRIDOR_HEIGHT / 2)

    # Create segments with different priorities
    from mfg_pde.geometry.boundary import BoundaryConditions

    bc_left = BCSegment(
        name="left_bc",
        bc_type=BCType.DIRICHLET,
        value=1.0,
        region_name="left_half",
        priority=2,  # Higher precedence
    )

    bc_bottom = BCSegment(
        name="bottom_bc",
        bc_type=BCType.NEUMANN,
        value=0.0,
        region_name="bottom_half",
        priority=1,  # Lower precedence
    )

    bc = BoundaryConditions(
        dimension=2,
        segments=[bc_left, bc_bottom],
        default_bc=BCType.PERIODIC,
        domain_bounds=np.array(geometry.bounds),
    )

    # Test points in different regions
    test_points = [
        (np.array([1.0, 0.3]), "left_bc", "Bottom-left (both regions, left wins)"),
        (np.array([3.0, 0.3]), "bottom_bc", "Bottom-right (bottom only)"),
        (np.array([1.0, 0.7]), "left_bc", "Top-left (left only)"),
    ]

    print("  Testing BC resolution at points:")
    for point, expected_name, description in test_points:
        segment = bc.get_bc_at_point(point, boundary_id=None, geometry=geometry)
        status = "✓" if segment.name == expected_name else "✗"
        print(f"    {status} {description:45s} → {segment.name} ({segment.bc_type.value})")

    print()


def main():
    """Main demonstration."""
    print("\n" + "=" * 80)
    print("Region-Based Boundary Conditions: 2D Corridor Flow Demonstration")
    print("=" * 80)
    print()

    # Create geometry
    geometry = create_corridor_geometry()

    # Mark regions
    mark_corridor_regions(geometry)

    # Create BCs from regions
    bc = create_region_based_bc(geometry)

    # Apply and visualize
    apply_and_visualize(geometry, bc)

    # Demonstrate priority resolution
    demonstrate_priority_resolution(geometry)

    # Summary
    print("=" * 80)
    print("Summary:")
    print("=" * 80)
    print("✓ Region-based BC successfully created for 2D corridor")
    print("✓ Inlet, outlet, obstacle, and wall regions marked using predicates and boundaries")
    print("✓ BCs applied using FDMApplicator with geometry parameter")
    print("✓ Priority resolution demonstrated for overlapping regions")
    print()
    print("Key advantages over boundary-based approach:")
    print("  - Flexible region definition via predicates")
    print("  - Partial boundary BCs (inlet/outlet cover only part of boundary)")
    print("  - Interior region BCs (obstacle)")
    print("  - Automatic priority resolution for overlaps")
    print("  - Composable with boundary-based segments")
    print()


if __name__ == "__main__":
    main()
