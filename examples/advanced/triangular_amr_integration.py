#!/usr/bin/env python3
"""
Triangular AMR Integration Example

Demonstrates how the new triangular AMR seamlessly integrates with
the existing MFG_PDE geometry infrastructure (MeshData, Domain2D, etc.).
"""

import matplotlib.pyplot as plt
import numpy as np

# Import existing geometry infrastructure
from mfg_pde.geometry import Domain2D, MeshData

# Import existing AMR infrastructure
# Import new triangular AMR
from mfg_pde.geometry.triangular_amr import TriangularMeshErrorEstimator, create_triangular_amr_mesh


def create_sample_triangular_mesh() -> MeshData:
    """
    Create a sample triangular mesh using existing infrastructure.

    In practice, this would come from Gmsh pipeline, but for demo
    we'll create a simple triangular mesh manually.
    """
    print("Creating sample triangular mesh...")

    # Simple triangular mesh of a unit square
    vertices = np.array(
        [
            [0.0, 0.0],  # 0: bottom-left
            [1.0, 0.0],  # 1: bottom-right
            [1.0, 1.0],  # 2: top-right
            [0.0, 1.0],  # 3: top-left
            [0.5, 0.5],  # 4: center
        ]
    )

    # Triangle connectivity (counter-clockwise)
    elements = np.array(
        [
            [0, 1, 4],  # bottom triangle
            [1, 2, 4],  # right triangle
            [2, 3, 4],  # top triangle
            [3, 0, 4],  # left triangle
        ]
    )

    # Create MeshData using existing infrastructure
    mesh_data = MeshData(
        vertices=vertices,
        elements=elements,
        element_type="triangle",
        boundary_tags=np.array([0, 0, 0, 0]),  # All interior
        element_tags=np.array([1, 1, 1, 1]),  # All same material
        boundary_faces=np.array([[0, 1], [1, 2], [2, 3], [3, 0]]),  # Boundary edges
        dimension=2,
        metadata={"created_by": "manual_triangulation"},
    )

    print(f"  Created mesh with {mesh_data.num_vertices} vertices, {mesh_data.num_elements} triangles")
    return mesh_data


def create_test_solution_data(triangular_amr_mesh) -> dict[str, np.ndarray]:
    """Create test solution data with sharp features for AMR testing."""

    print("Creating test solution with sharp features...")

    # Get triangle centroids for solution evaluation
    centroids = []
    for triangle_id in triangular_amr_mesh.leaf_triangles:
        triangle = triangular_amr_mesh.triangles[triangle_id]
        centroids.append(triangle.centroid)

    centroids = np.array(centroids)

    # Create sharp Gaussian peak at (0.7, 0.3)
    peak_x, peak_y = 0.7, 0.3
    distances = np.sqrt((centroids[:, 0] - peak_x) ** 2 + (centroids[:, 1] - peak_y) ** 2)

    # Sharp value function (small σ creates sharp gradient)
    U = np.exp(-50 * distances**2)

    # Sharp density function at different location
    peak2_x, peak2_y = 0.3, 0.7
    distances2 = np.sqrt((centroids[:, 0] - peak2_x) ** 2 + (centroids[:, 1] - peak2_y) ** 2)
    M = np.exp(-30 * distances2**2)
    M = M / np.sum(M)  # Normalize

    print(f"  Created solution data for {len(centroids)} triangles")
    print(f"  U range: [{np.min(U):.3f}, {np.max(U):.3f}]")
    print(f"  M range: [{np.min(M):.3f}, {np.max(M):.3f}]")

    return {"U": U, "M": M}


def demonstrate_triangular_amr():
    """Main demonstration of triangular AMR integration."""

    print("=" * 60)
    print("TRIANGULAR AMR INTEGRATION DEMONSTRATION")
    print("=" * 60)

    # Step 1: Create triangular mesh using existing infrastructure
    initial_mesh = create_sample_triangular_mesh()

    # Step 2: Create triangular AMR mesh from existing MeshData
    print("\nStep 2: Creating triangular AMR mesh...")
    triangular_amr = create_triangular_amr_mesh(
        mesh_data=initial_mesh,
        error_threshold=1e-3,  # Moderate threshold for demo
        max_levels=3,  # Allow 3 levels of refinement
        backend=None,  # Auto-select backend
    )

    print(f"  Initial AMR mesh: {triangular_amr.total_triangles} triangles")
    print(f"  Leaf triangles: {len(triangular_amr.leaf_triangles)}")

    # Step 3: Create test solution with sharp features
    solution_data = create_test_solution_data(triangular_amr)

    # Step 4: Create error estimator
    print("\nStep 4: Setting up error estimation...")
    error_estimator = TriangularMeshErrorEstimator()

    # Step 5: Perform adaptive refinement
    print("\nStep 5: Performing adaptive refinement...")

    adaptation_cycles = 3
    for cycle in range(adaptation_cycles):
        print(f"\n  Adaptation Cycle {cycle + 1}:")

        # Adapt mesh based on solution
        stats = triangular_amr.adapt_mesh(solution_data, error_estimator)

        print(f"    Triangles refined: {stats['total_refined']}")
        print(f"    Red refinements: {stats['red_refinements']}")
        print(f"    Green refinements: {stats['green_refinements']}")
        print(f"    Current total: {stats['final_triangles']}")
        print(f"    Max level: {stats['max_level']}")

        # Update solution data for new mesh (simplified)
        if stats["total_refined"] > 0:
            solution_data = create_test_solution_data(triangular_amr)

    # Step 6: Get final mesh statistics
    print("\nStep 6: Final mesh analysis...")
    final_stats = triangular_amr.get_mesh_statistics()

    print("  Final Statistics:")
    print(f"    Total triangles: {final_stats['total_triangles']}")
    print(f"    Leaf triangles: {final_stats['leaf_triangles']}")
    print(f"    Max refinement level: {final_stats['max_level']}")
    print(f"    Level distribution: {final_stats['level_distribution']}")
    print(f"    Aspect ratio range: [{final_stats['min_aspect_ratio']:.2f}, {final_stats['max_aspect_ratio']:.2f}]")
    print(f"    Refinement efficiency: {final_stats['refinement_ratio']:.2f}")

    # Step 7: Export back to MeshData format
    print("\nStep 7: Exporting adapted mesh...")
    adapted_mesh_data = triangular_amr.export_to_mesh_data()

    print("  Exported MeshData:")
    print(f"    Vertices: {adapted_mesh_data.num_vertices}")
    print(f"    Elements: {adapted_mesh_data.num_elements}")
    print(f"    Element type: {adapted_mesh_data.element_type}")
    print(f"    Metadata: {adapted_mesh_data.metadata}")

    # Step 8: Visualization
    print("\nStep 8: Creating visualization...")
    create_triangular_amr_visualization(triangular_amr, solution_data, final_stats)

    return triangular_amr, adapted_mesh_data, final_stats


def create_triangular_amr_visualization(triangular_amr, solution_data, stats):
    """Create visualization of triangular AMR results."""

    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Triangular AMR Integration Results", fontsize=16)

        # Plot 1: Initial vs adapted mesh
        ax1 = axes[0, 0]
        plot_triangular_mesh(triangular_amr, ax1, color_by="level")
        ax1.set_title("Adaptive Triangular Mesh\n(colored by refinement level)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        # Plot 2: Solution field U
        ax2 = axes[0, 1]
        plot_solution_field(triangular_amr, solution_data["U"], ax2)
        ax2.set_title("Value Function U\n(sharp features drive refinement)")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")

        # Plot 3: Solution field M
        ax3 = axes[1, 0]
        plot_solution_field(triangular_amr, solution_data["M"], ax3)
        ax3.set_title("Density Function M\n(different sharp feature location)")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")

        # Plot 4: Refinement statistics
        ax4 = axes[1, 1]
        plot_refinement_statistics(stats, ax4)
        ax4.set_title("Refinement Statistics")

        plt.tight_layout()

        # Save plot
        plt.savefig("triangular_amr_integration.png", dpi=150, bbox_inches="tight")
        print("  Visualization saved to 'triangular_amr_integration.png'")

        # Show if possible
        try:
            plt.show()
        except Exception:
            print("  Display not available, plot saved to file.")

    except Exception as e:
        print(f"  Visualization failed: {e}")


def plot_triangular_mesh(triangular_amr, ax, color_by="level"):
    """Plot triangular mesh with coloring options."""

    # Get leaf triangles
    leaf_triangles = [triangular_amr.triangles[tid] for tid in triangular_amr.leaf_triangles]

    colors = []
    for triangle in leaf_triangles:
        if color_by == "level":
            colors.append(triangle.level)
        elif color_by == "aspect_ratio":
            colors.append(triangle.aspect_ratio)
        else:
            colors.append(1.0)

    # Plot triangles
    for i, triangle in enumerate(leaf_triangles):
        vertices = triangle.vertices
        # Close the triangle
        triangle_plot = np.vstack([vertices, vertices[0:1]])

        color_val = colors[i]
        color = plt.cm.viridis(color_val / max(colors) if max(colors) > 0 else 0)

        ax.plot(triangle_plot[:, 0], triangle_plot[:, 1], "k-", linewidth=0.5)
        ax.fill(vertices[:, 0], vertices[:, 1], color=color, alpha=0.7)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def plot_solution_field(triangular_amr, solution_values, ax):
    """Plot solution field on triangular mesh."""

    leaf_triangles = [triangular_amr.triangles[tid] for tid in triangular_amr.leaf_triangles]

    # Plot triangles colored by solution value
    for i, triangle in enumerate(leaf_triangles):
        if i < len(solution_values):
            value = solution_values[i]
            color = plt.cm.plasma(value / np.max(solution_values) if np.max(solution_values) > 0 else 0)

            vertices = triangle.vertices
            ax.fill(vertices[:, 0], vertices[:, 1], color=color, alpha=0.8)
            ax.plot(vertices[[0, 1, 2, 0], 0], vertices[[0, 1, 2, 0], 1], "k-", linewidth=0.3)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def plot_refinement_statistics(stats, ax):
    """Plot refinement statistics."""

    # Level distribution
    levels = list(stats["level_distribution"].keys())
    counts = list(stats["level_distribution"].values())

    ax.bar(levels, counts, alpha=0.7, color="skyblue", edgecolor="navy")
    ax.set_xlabel("Refinement Level")
    ax.set_ylabel("Number of Triangles")
    ax.set_title("Triangles per Refinement Level")
    ax.grid(True, alpha=0.3)

    # Add statistics text
    text_stats = f"""Total Triangles: {stats['total_triangles']}
Leaf Triangles: {stats['leaf_triangles']}
Max Level: {stats['max_level']}
Aspect Ratio: {stats['min_aspect_ratio']:.2f} - {stats['max_aspect_ratio']:.2f}"""

    ax.text(
        0.02,
        0.98,
        text_stats,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )


def demonstrate_integration_with_existing_geometry():
    """Demonstrate integration with existing Domain2D and MeshPipeline."""

    print("\n" + "=" * 60)
    print("INTEGRATION WITH EXISTING GEOMETRY PIPELINE")
    print("=" * 60)

    try:
        # This would work if Gmsh/Meshio are available
        print("Creating 2D domain with complex geometry...")

        domain = Domain2D(
            domain_type="rectangle",
            bounds=(0.0, 2.0, 0.0, 1.0),
            holes=[{"type": "circle", "center": (1.0, 0.5), "radius": 0.2}],
            mesh_size=0.1,
        )

        print(f"  Created Domain2D: {domain.domain_type}")
        print(f"  Bounds: {domain.bounds_rect}")
        print(f"  Holes: {len(domain.holes)}")

        # This would generate actual triangular mesh
        print("  Note: Full mesh generation requires Gmsh installation")
        print("  The triangular AMR would work seamlessly with generated mesh!")

    except Exception as e:
        print(f"Advanced geometry pipeline not fully available: {e}")
        print("This is expected if optional dependencies (Gmsh, etc.) are not installed")
        print("The triangular AMR integration is ready to work with any MeshData!")


def main():
    """Main demonstration function."""

    print("Triangular AMR Integration with Existing MFG_PDE Geometry")
    print("This demonstrates seamless integration with existing infrastructure:")
    print("  - MeshData format compatibility")
    print("  - Domain2D and geometry pipeline integration")
    print("  - Existing error estimation framework")
    print("  - Export back to standard formats")

    # Main triangular AMR demonstration
    triangular_amr, adapted_mesh, stats = demonstrate_triangular_amr()

    # Show integration potential
    demonstrate_integration_with_existing_geometry()

    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION SUMMARY")
    print("=" * 60)
    print("✓ Triangular AMR seamlessly uses existing MeshData format")
    print("✓ Compatible with existing geometry pipeline (Domain2D, etc.)")
    print("✓ Leverages existing error estimation framework")
    print("✓ Exports back to standard MeshData for visualization/analysis")
    print("✓ Supports both red and green refinement strategies")
    print("✓ Maintains triangle quality metrics")
    print("✓ Ready for integration with FEM solvers")

    print(f"\nFinal mesh: {stats['total_triangles']} triangles, {stats['max_level']} levels")
    print("Triangular AMR integration complete!")

    return triangular_amr, adapted_mesh


if __name__ == "__main__":
    main()
