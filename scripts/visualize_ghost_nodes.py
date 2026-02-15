#!/usr/bin/env python3
"""
Visualize Ghost Nodes Method for GFDM.

Shows:
1. All collocation points (interior + boundary)
2. Ghost points created for a selected boundary point
3. Mirror relationships between ghosts and interior neighbors
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import BoundaryConditions, MFGProblem
from mfg_pde.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import BCType
from mfg_pde.geometry.collocation import CollocationSampler


def visualize_ghost_nodes():
    """Create visualization of ghost nodes for a boundary point."""
    print("=" * 70)
    print("Ghost Nodes Visualization")
    print("=" * 70)
    print()

    # Create 2D problem
    Lx, Ly = 10.0, 5.0
    geometry = TensorProductGrid(
        dimension=2,
        bounds=[(0.0, Lx), (0.0, Ly)],
        Nx=[21, 11],
    )

    bc = BoundaryConditions(default_bc=BCType.NEUMANN, dimension=2)

    problem = MFGProblem(
        geometry=geometry,
        T=1.0,
        Nt=10,
        sigma=0.1,
        lambda_=1.0,
        gamma=0.5,
        boundary_conditions=bc,
    )

    # Generate collocation points
    sampler = CollocationSampler(geometry)
    coll = sampler.generate_collocation(
        n_interior=64,  # Use power of 2 for Sobol
        n_boundary=40,
        interior_method="sobol",
        seed=42,
    )

    print(f"Collocation points: {len(coll.points)}")
    print(f"  Interior: {coll.n_interior}")
    print(f"  Boundary: {coll.n_boundary}")
    print()

    # Create GFDM solver with ghost nodes
    avg_spacing = np.sqrt(Lx * Ly / coll.n_interior)
    delta = 2.5 * avg_spacing

    print("GFDM parameters:")
    print(f"  Delta (neighborhood radius): {delta:.3f}")
    print(f"  Average spacing: {avg_spacing:.3f}")
    print()

    solver = HJBGFDMSolver(
        problem,
        collocation_points=coll.points,
        boundary_indices=coll.boundary_indices,
        delta=delta,
        use_ghost_nodes=True,
    )

    # Select a boundary point to visualize (preferably on a wall, not corner)
    # Find a point on the left wall (x ≈ 0)
    boundary_points = coll.points[coll.boundary_indices]
    left_wall_mask = boundary_points[:, 0] < 0.5
    left_wall_indices = coll.boundary_indices[left_wall_mask]

    if len(left_wall_indices) > 0:
        # Pick the middle point on left wall
        selected_idx = left_wall_indices[len(left_wall_indices) // 2]
    else:
        # Fallback to first boundary point
        selected_idx = coll.boundary_indices[0]

    selected_point = coll.points[selected_idx]
    print(f"Selected boundary point: index={selected_idx}, position=({selected_point[0]:.2f}, {selected_point[1]:.2f})")

    # Get ghost node information
    if selected_idx in solver._ghost_node_map:
        ghost_info = solver._ghost_node_map[selected_idx]
        n_ghosts = ghost_info["n_ghosts"]
        ghost_to_mirror = ghost_info["ghost_to_mirror"]
        print(f"  Number of ghosts: {n_ghosts}")
    else:
        print("  No ghosts found for this point")
        return

    # Get neighborhood information
    neighborhood = solver.neighborhoods[selected_idx]
    neighbor_points = neighborhood["points"]
    neighbor_indices = neighborhood["indices"]

    # Separate real and ghost neighbors
    real_mask = neighbor_indices >= 0
    ghost_mask = neighbor_indices < 0

    neighbor_indices[real_mask]
    real_neighbor_points = neighbor_points[real_mask]

    ghost_indices = neighbor_indices[ghost_mask]
    ghost_points = neighbor_points[ghost_mask]

    # Get mirror points for each ghost
    mirror_indices = []
    mirror_points = []
    for ghost_idx in ghost_indices:
        mirror_idx = ghost_to_mirror[int(ghost_idx)]
        mirror_indices.append(mirror_idx)
        mirror_points.append(coll.points[mirror_idx])
    mirror_points = np.array(mirror_points)

    # Get normal vector
    normal = solver._compute_outward_normal(selected_idx)

    print(f"  Real neighbors: {len(real_neighbor_points)}")
    print(f"  Ghost neighbors: {len(ghost_points)}")
    print(f"  Outward normal: ({normal[0]:.3f}, {normal[1]:.3f})")
    print()

    # Create visualization
    _fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # === Left plot: Full domain overview ===
    ax = axes[0]

    # Plot all interior points
    interior_points = coll.points[np.setdiff1d(np.arange(len(coll.points)), coll.boundary_indices)]
    ax.scatter(
        interior_points[:, 0], interior_points[:, 1], c="lightblue", s=20, alpha=0.6, label="Interior points", zorder=1
    )

    # Plot all boundary points
    ax.scatter(
        boundary_points[:, 0], boundary_points[:, 1], c="orange", s=30, alpha=0.8, label="Boundary points", zorder=2
    )

    # Highlight selected boundary point
    ax.scatter(
        [selected_point[0]],
        [selected_point[1]],
        c="red",
        s=200,
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        label="Selected point",
        zorder=5,
    )

    # Plot domain boundary
    ax.plot([0, Lx, Lx, 0, 0], [0, 0, Ly, Ly, 0], "k-", linewidth=2, alpha=0.5)

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Full Domain: Collocation Points", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # === Right plot: Zoomed view showing ghost nodes ===
    ax = axes[1]

    # Plot selected boundary point
    ax.scatter(
        [selected_point[0]],
        [selected_point[1]],
        c="red",
        s=300,
        marker="*",
        edgecolors="black",
        linewidths=2,
        label="Boundary point",
        zorder=10,
    )

    # Plot real neighbors
    ax.scatter(
        real_neighbor_points[:, 0],
        real_neighbor_points[:, 1],
        c="blue",
        s=80,
        alpha=0.7,
        label="Real neighbors",
        zorder=3,
    )

    # Plot ghost neighbors
    ax.scatter(
        ghost_points[:, 0],
        ghost_points[:, 1],
        c="purple",
        s=120,
        marker="s",
        alpha=0.7,
        edgecolors="black",
        linewidths=1,
        label="Ghost neighbors",
        zorder=4,
    )

    # Plot mirror points
    ax.scatter(
        mirror_points[:, 0],
        mirror_points[:, 1],
        c="green",
        s=100,
        marker="D",
        alpha=0.6,
        edgecolors="darkgreen",
        linewidths=1.5,
        label="Mirror points",
        zorder=5,
    )

    # Draw mirror relationships (dashed lines)
    for _i, (ghost_pt, mirror_pt) in enumerate(zip(ghost_points, mirror_points, strict=False)):
        ax.plot([ghost_pt[0], mirror_pt[0]], [ghost_pt[1], mirror_pt[1]], "g--", alpha=0.4, linewidth=1, zorder=1)

    # Draw neighborhood circle
    circle = plt.Circle(
        selected_point,
        delta,
        fill=False,
        edgecolor="gray",
        linestyle=":",
        linewidth=2,
        alpha=0.5,
        label=f"δ-neighborhood (r={delta:.2f})",
    )
    ax.add_patch(circle)

    # Draw normal vector
    arrow_length = delta * 0.6
    ax.arrow(
        selected_point[0],
        selected_point[1],
        normal[0] * arrow_length,
        normal[1] * arrow_length,
        head_width=0.15,
        head_length=0.1,
        fc="red",
        ec="darkred",
        linewidth=2,
        zorder=8,
        label="Outward normal",
    )

    # Draw boundary wall (local segment)
    if abs(normal[0]) > 0.9:  # Vertical wall (left/right)
        wall_x = selected_point[0]
        wall_y_range = [max(0, selected_point[1] - 1.5), min(Ly, selected_point[1] + 1.5)]
        ax.plot([wall_x, wall_x], wall_y_range, "k-", linewidth=3, alpha=0.7, label="Domain boundary")
    elif abs(normal[1]) > 0.9:  # Horizontal wall (top/bottom)
        wall_y = selected_point[1]
        wall_x_range = [max(0, selected_point[0] - 1.5), min(Lx, selected_point[0] + 1.5)]
        ax.plot(wall_x_range, [wall_y, wall_y], "k-", linewidth=3, alpha=0.7, label="Domain boundary")

    # Set zoom limits
    zoom = delta * 1.8
    ax.set_xlim(selected_point[0] - zoom, selected_point[0] + zoom)
    ax.set_ylim(selected_point[1] - zoom, selected_point[1] + zoom)

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(f"Ghost Nodes Detail (Point {selected_idx})", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.text(
        0.02,
        0.98,
        f"{n_ghosts} ghost nodes created",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()
    plt.savefig("ghost_nodes_visualization.png", dpi=150, bbox_inches="tight")
    print("Saved: ghost_nodes_visualization.png")
    plt.show()

    print("=" * 70)
    print("Visualization complete!")
    print("=" * 70)


if __name__ == "__main__":
    visualize_ghost_nodes()
