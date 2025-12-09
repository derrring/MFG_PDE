#!/usr/bin/env python3
"""
AMR Geometry Protocol Demo - v0.10.1 GeometryProtocol Support
==============================================================

This example demonstrates the new GeometryProtocol support for AMR geometries
introduced in v0.10.1. Shows how all geometry types (uniform, AMR, network, etc.)
now share a common interface.

Key Features:
    - GeometryProtocol-compliant AMR meshes
    - Uniform interface across all geometry types
    - Polymorphic get_problem_config() method
    - Runtime type checking and validation

Created: 2025-11-05
Part of: v0.10.1 AMR GeometryProtocol support
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.amr.amr_1d import AMRRefinementCriteria, OneDimensionalAMRMesh
from mfg_pde.geometry.amr.amr_quadtree_2d import AdaptiveMesh
from mfg_pde.geometry.protocol import (
    is_geometry_compatible,
    validate_geometry,
)


def demonstrate_protocol_compliance():
    """Demonstrate that all geometry types satisfy GeometryProtocol."""
    print("=" * 70)
    print("GeometryProtocol Compliance Demonstration")
    print("=" * 70)

    # Create different geometry types
    geometries = []

    # 1. Uniform 1D tensor product grid
    print("\n1. Creating uniform 1D tensor product grid...")
    grid_1d = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], num_points=[50])
    geometries.append(("Uniform 1D Grid", grid_1d))

    # 2. 1D AMR mesh
    print("2. Creating 1D AMR mesh...")
    base_domain = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], num_points=[21])
    refinement_criteria = AMRRefinementCriteria(
        max_refinement_levels=2, gradient_threshold=0.5, coarsening_threshold=0.25, min_cell_size=0.001
    )
    amr_1d = OneDimensionalAMRMesh(
        domain_1d=base_domain, initial_num_intervals=20, refinement_criteria=refinement_criteria
    )
    geometries.append(("1D AMR Mesh", amr_1d))

    # 3. 2D AMR quadtree mesh
    print("3. Creating 2D AMR quadtree mesh...")
    amr_2d = AdaptiveMesh(
        domain_bounds=(0.0, 1.0, 0.0, 1.0),
        initial_resolution=(8, 8),
        refinement_criteria=AMRRefinementCriteria(max_refinement_levels=4),
    )
    geometries.append(("2D AMR Quadtree", amr_2d))

    print("\n" + "=" * 70)
    print("GeometryProtocol Validation")
    print("=" * 70)

    # Validate each geometry
    for name, geom in geometries:
        print(f"\n{name}:")
        print(f"  is_geometry_compatible: {is_geometry_compatible(geom)}")

        try:
            validate_geometry(geom)
            print("  validate_geometry: PASS")
        except (TypeError, ValueError) as e:
            print(f"  validate_geometry: FAIL - {e}")

        # Check protocol properties
        print(f"  dimension: {geom.dimension}")
        print(f"  geometry_type: {geom.geometry_type.value}")
        print(f"  num_spatial_points: {geom.num_spatial_points}")

        # Get spatial grid
        grid = geom.get_spatial_grid()
        if isinstance(grid, np.ndarray):
            print(f"  spatial_grid shape: {grid.shape}")
        else:
            print(f"  spatial_grid type: {type(grid)}")

        # Get problem config
        config = geom.get_problem_config()
        print(f"  problem_config keys: {list(config.keys())}")


def demonstrate_polymorphic_config():
    """Demonstrate polymorphic get_problem_config() method."""
    print("\n" + "=" * 70)
    print("Polymorphic get_problem_config() Demonstration")
    print("=" * 70)

    # Create different geometries
    grid_1d = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], num_points=[50])

    base_domain = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], num_points=[21])
    refinement_criteria = AMRRefinementCriteria(max_refinement_levels=2)
    amr_1d = OneDimensionalAMRMesh(
        domain_1d=base_domain, initial_num_intervals=20, refinement_criteria=refinement_criteria
    )

    geometries = [
        ("Uniform 1D Grid", grid_1d),
        ("AMR 1D", amr_1d),
    ]

    for name, geom in geometries:
        print(f"\n{name}:")
        config = geom.get_problem_config()

        print(f"  num_spatial_points: {config['num_spatial_points']}")
        print(f"  spatial_shape: {config['spatial_shape']}")
        print(f"  spatial_bounds: {config['spatial_bounds']}")
        print(f"  spatial_discretization: {config['spatial_discretization']}")

        if config["legacy_1d_attrs"] is not None:
            print(f"  legacy_1d_attrs: {list(config['legacy_1d_attrs'].keys())}")
        else:
            print("  legacy_1d_attrs: None (AMR mesh)")


def visualize_amr_structure():
    """Visualize AMR mesh structure."""
    print("\n" + "=" * 70)
    print("AMR Mesh Structure Visualization")
    print("=" * 70)

    # Create 1D AMR mesh
    base_domain = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], num_points=[11])
    refinement_criteria = AMRRefinementCriteria(
        max_refinement_levels=3, gradient_threshold=0.3, coarsening_threshold=0.15, min_cell_size=0.001
    )
    amr_mesh = OneDimensionalAMRMesh(
        domain_1d=base_domain, initial_num_intervals=10, refinement_criteria=refinement_criteria
    )

    print("\n1D AMR Mesh:")
    print(f"  Total intervals: {amr_mesh.num_spatial_points}")
    print(f"  Dimension: {amr_mesh.dimension}")
    print(f"  Geometry type: {amr_mesh.geometry_type.value}")

    # Get spatial grid
    grid = amr_mesh.get_spatial_grid()
    print(f"  Grid points shape: {grid.shape}")

    # Create 2D AMR quadtree
    amr_2d = AdaptiveMesh(
        domain_bounds=(0.0, 1.0, 0.0, 1.0),
        initial_resolution=(8, 8),
        refinement_criteria=AMRRefinementCriteria(max_refinement_levels=4),
    )

    print("\n2D AMR Quadtree:")
    print(f"  Total nodes: {amr_2d.num_spatial_points}")
    print(f"  Dimension: {amr_2d.dimension}")
    print(f"  Geometry type: {amr_2d.geometry_type.value}")

    grid_2d = amr_2d.get_spatial_grid()
    print(f"  Grid points shape: {grid_2d.shape}")

    # Visualize
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1D AMR structure
    ax1.set_title("1D AMR Mesh Structure", fontweight="bold")
    intervals = amr_mesh.intervals  # dict of {id: Interval1D}
    for interval in intervals.values():
        if interval.is_leaf:
            # Color by refinement level
            color = plt.cm.viridis(interval.level / refinement_criteria.max_refinement_levels)
            ax1.axvspan(interval.x_min, interval.x_max, alpha=0.5, color=color)
            ax1.axvline(interval.x_min, color="k", linewidth=0.5, alpha=0.5)

    ax1.set_xlabel("x")
    ax1.set_ylabel("Refinement Level")
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Plot 2D AMR structure
    ax2.set_title("2D AMR Quadtree Structure", fontweight="bold")
    nodes = amr_2d.leaf_nodes
    for node in nodes:
        # Color by refinement level
        color = plt.cm.viridis(node.level / 4)
        rect = plt.Rectangle(
            (node.center_x - node.dx / 2, node.center_y - node.dy / 2),
            node.dx,
            node.dy,
            facecolor=color,
            edgecolor="k",
            linewidth=0.5,
            alpha=0.5,
        )
        ax2.add_patch(rect)

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)


def main():
    """Main execution function."""
    print("\nAMR Geometry Protocol Demo - v0.10.1")
    print("=" * 70)

    # Demonstrate protocol compliance
    demonstrate_protocol_compliance()

    # Demonstrate polymorphic config
    demonstrate_polymorphic_config()

    # Visualize AMR structure
    visualize_amr_structure()

    print("\nDemo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
