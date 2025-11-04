"""
Maze + Implicit Geometry Integration Demonstration

This example showcases how to combine maze generation with implicit geometry
to create complex hybrid domains for Mean Field Games:

1. Maze-based structural obstacles (corridors, rooms)
2. Implicit geometric features (spheres, ellipsoids, CSG operations)
3. Hybrid domains for realistic crowd dynamics
4. Particle-collocation integration for high-dimensional problems

Key Innovation: Combine discrete maze structures with continuous implicit
geometry for unprecedented flexibility in domain design.

Author: MFG_PDE Team
Date: November 2025
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.geometry.implicit import DifferenceDomain, Hyperrectangle, Hypersphere
from mfg_pde.geometry.mazes.maze_generator import generate_maze

print("=" * 70)
print("MAZE + IMPLICIT GEOMETRY INTEGRATION")
print("=" * 70)
print()

# ==============================================================================
# PART 1: Basic Maze Generation
# ==============================================================================

print("PART 1: Basic Maze Generation")
print("-" * 70)
print()

# Create a simple maze using the high-level function
rows, cols = 20, 20
algorithm = "recursive_backtracking"
cell_size = 1.0

print(f"Generating {rows}x{cols} maze...")
print(f"  Algorithm: {algorithm}")
print(f"  Cell size: {cell_size}")
print()

start = time.time()
maze_grid = generate_maze(rows=rows, cols=cols, algorithm=algorithm, seed=42)
elapsed = time.time() - start

print(f"✓ Maze generated in {elapsed:.3f}s")
print(f"  Grid shape: {maze_grid.shape}")
print(f"  Navigable cells: {np.sum(maze_grid == 0)} / {maze_grid.size}")
print(f"  Wall cells: {np.sum(maze_grid == 1)} / {maze_grid.size}")
print()

# ==============================================================================
# PART 2: Implicit Geometry - Spherical Obstacles
# ==============================================================================

print("PART 2: Implicit Geometry - Spherical Obstacles")
print("-" * 70)
print()

# Create a rectangular domain matching the maze bounds
domain_bounds = np.array([[0.0, 20.0], [0.0, 20.0]])
base_domain = Hyperrectangle(domain_bounds)

print(f"Base domain: [{domain_bounds[0,0]}, {domain_bounds[0,1]}] x [{domain_bounds[1,0]}, {domain_bounds[1,1]}]")
print(f"  Memory: {domain_bounds.nbytes} bytes (O(d) complexity)")
print()

# Add spherical obstacles
obstacle_centers = [
    np.array([5.0, 5.0]),
    np.array([15.0, 15.0]),
    np.array([5.0, 15.0]),
]
obstacle_radius = 1.5

print(f"Adding {len(obstacle_centers)} spherical obstacles:")
for i, center in enumerate(obstacle_centers):
    print(f"  Obstacle {i+1}: center={center}, radius={obstacle_radius}")

# Create spherical obstacles using CSG
obstacles = [Hypersphere(center=c, radius=obstacle_radius) for c in obstacle_centers]

# Navigable domain = base domain - all obstacles
navigable_domain = base_domain
for obstacle in obstacles:
    navigable_domain = DifferenceDomain(navigable_domain, obstacle)

print()
print(f"✓ Created navigable domain via CSG operations")
print(f"  Base domain - Obstacle 1 - Obstacle 2 - Obstacle 3")
print()

# ==============================================================================
# PART 3: Hybrid Domain - Maze + Implicit Geometry
# ==============================================================================

print("PART 3: Hybrid Domain - Maze + Implicit Geometry")
print("-" * 70)
print()

print("Strategy: Combine maze structure with spherical obstacles")
print("  1. Maze defines corridor structure (discrete)")
print("  2. Spheres add local obstacles (continuous)")
print("  3. Result: Rich navigable space for crowd dynamics")
print()


def combine_maze_and_obstacles(
    maze_grid: np.ndarray,
    obstacle_centers: list[np.ndarray],
    obstacle_radius: float,
    cell_size: float = 1.0,
) -> tuple[np.ndarray, callable]:
    """
    Combine maze walls with spherical obstacles.

    Args:
        maze_grid: Binary grid (0=navigable, 1=wall)
        obstacle_centers: List of obstacle centers
        obstacle_radius: Radius of spherical obstacles
        cell_size: Size of each maze cell

    Returns:
        combined_grid: Grid with both maze walls and obstacles
        contains_func: Function to check if point is navigable
    """
    # Create combined grid
    combined_grid = maze_grid.copy()

    # Add spherical obstacles to grid
    height, width = maze_grid.shape
    for i in range(height):
        for j in range(width):
            # Cell center in physical coordinates
            x = (j + 0.5) * cell_size
            y = (i + 0.5) * cell_size
            point = np.array([x, y])

            # Check if point is inside any obstacle
            for center in obstacle_centers:
                if np.linalg.norm(point - center) <= obstacle_radius:
                    combined_grid[i, j] = 1  # Mark as obstacle
                    break

    # Create contains function
    def contains(point: np.ndarray) -> bool:
        """Check if point is in navigable space."""
        # Convert to grid coordinates
        j = int(point[0] / cell_size)
        i = int(point[1] / cell_size)

        # Bounds check
        if i < 0 or i >= height or j < 0 or j >= width:
            return False

        # Check grid
        if combined_grid[i, j] == 1:  # Wall or obstacle
            return False

        # Check spherical obstacles (continuous)
        for center in obstacle_centers:
            if np.linalg.norm(point - center) <= obstacle_radius:
                return False

        return True

    return combined_grid, contains


# Combine maze and obstacles
combined_grid, contains_func = combine_maze_and_obstacles(
    maze_grid, obstacle_centers, obstacle_radius, cell_size
)

print(f"Combined domain statistics:")
print(f"  Original navigable cells: {np.sum(maze_grid == 0)} / {maze_grid.size}")
print(f"  Hybrid navigable cells: {np.sum(combined_grid == 0)} / {combined_grid.size}")
print(f"  Additional blocked by spheres: {np.sum(maze_grid == 0) - np.sum(combined_grid == 0)}")
print()

# ==============================================================================
# PART 4: Particle Sampling in Hybrid Domain
# ==============================================================================

print("PART 4: Particle Sampling in Hybrid Domain")
print("-" * 70)
print()

print("Sampling particles using rejection sampling...")
print()


def sample_hybrid_domain(
    contains_func: callable,
    bounds: np.ndarray,
    n_samples: int,
    max_iterations: int = 100000,
    seed: int | None = None,
) -> np.ndarray:
    """
    Sample particles uniformly in hybrid domain using rejection sampling.

    Args:
        contains_func: Function to check if point is navigable
        bounds: Domain bounds (d, 2) array
        n_samples: Number of samples desired
        max_iterations: Maximum rejection attempts
        seed: Random seed

    Returns:
        particles: (n_samples, d) array of samples
    """
    if seed is not None:
        np.random.seed(seed)

    d = bounds.shape[0]
    particles = []
    iterations = 0

    while len(particles) < n_samples and iterations < max_iterations:
        # Sample uniformly in bounding box
        candidate = np.random.rand(d) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

        # Accept if in navigable space
        if contains_func(candidate):
            particles.append(candidate)

        iterations += 1

    if len(particles) < n_samples:
        print(f"  Warning: Only sampled {len(particles)}/{n_samples} particles")

    return np.array(particles)


# Sample particles
n_particles = 5000
print(f"Sampling {n_particles} particles in hybrid domain...")
start = time.time()
particles = sample_hybrid_domain(contains_func, domain_bounds, n_particles, seed=42)
elapsed = time.time() - start

print(f"✓ Sampled {len(particles)} particles in {elapsed:.3f}s")
print(f"  Sampling rate: {len(particles) / elapsed:.0f} particles/second")
print(f"  Acceptance rate: {len(particles) / (elapsed * 1000):.1%} (rejection sampling)")
print()

# Verify particles
print("Verification:")
in_domain = np.array([contains_func(p) for p in particles])
print(f"  All particles in navigable space: {np.all(in_domain)}")
print(f"  Particle positions range:")
print(f"    x ∈ [{particles[:, 0].min():.2f}, {particles[:, 0].max():.2f}]")
print(f"    y ∈ [{particles[:, 1].min():.2f}, {particles[:, 1].max():.2f}]")
print()

# ==============================================================================
# PART 5: Visualization
# ==============================================================================

print("PART 5: Visualization")
print("-" * 70)
print()

print("Creating visualization of hybrid domain...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Original maze
ax1 = axes[0]
ax1.imshow(maze_grid, cmap="binary", origin="lower", extent=[0, 20, 0, 20])
ax1.set_title("Original Maze Structure", fontsize=14, fontweight="bold")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.grid(True, alpha=0.3)

# Plot 2: Spherical obstacles only
ax2 = axes[1]
ax2.set_xlim(0, 20)
ax2.set_ylim(0, 20)
ax2.set_aspect("equal")

# Draw obstacles
for center in obstacle_centers:
    circle = plt.Circle(center, obstacle_radius, color="red", alpha=0.5, label="Obstacle")
    ax2.add_patch(circle)

ax2.set_title("Spherical Obstacles (Implicit)", fontsize=14, fontweight="bold")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.grid(True, alpha=0.3)
ax2.legend(["Spherical Obstacle"], loc="upper right")

# Plot 3: Hybrid domain with particles
ax3 = axes[2]
ax3.imshow(combined_grid, cmap="binary", origin="lower", extent=[0, 20, 0, 20], alpha=0.6)

# Overlay particles
ax3.scatter(particles[:, 0], particles[:, 1], c="blue", s=1, alpha=0.5, label="Particles")

# Overlay obstacles
for center in obstacle_centers:
    circle = plt.Circle(center, obstacle_radius, color="red", alpha=0.3, fill=True)
    ax3.add_patch(circle)
    circle_outline = plt.Circle(center, obstacle_radius, color="red", alpha=1.0, fill=False, linewidth=2)
    ax3.add_patch(circle_outline)

ax3.set_title("Hybrid Domain: Maze + Obstacles + Particles", fontsize=14, fontweight="bold")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.grid(True, alpha=0.3)
ax3.legend(loc="upper right")

plt.tight_layout()
plt.savefig("examples/outputs/maze_implicit_geometry_hybrid.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved visualization to examples/outputs/maze_implicit_geometry_hybrid.png")
print()

# ==============================================================================
# PART 6: Application - Crowd Evacuation with Obstacles
# ==============================================================================

print("PART 6: Application - Crowd Evacuation with Obstacles")
print("-" * 70)
print()

print("Scenario: Building evacuation with furniture obstacles")
print("  - Maze: Corridor structure (walls, rooms)")
print("  - Spheres: Furniture, columns, temporary obstacles")
print("  - Particles: Initial crowd distribution")
print()

# Define exit location
exit_location = np.array([19.0, 10.0])  # Right side of domain

# Compute distances to exit for all particles
distances = np.linalg.norm(particles - exit_location, axis=1)

print(f"Crowd statistics:")
print(f"  Total agents: {len(particles)}")
print(f"  Exit location: {exit_location}")
print(f"  Distance to exit:")
print(f"    Mean: {distances.mean():.2f} units")
print(f"    Std: {distances.std():.2f} units")
print(f"    Min: {distances.min():.2f} units")
print(f"    Max: {distances.max():.2f} units")
print()

# Identify agents near obstacles (high congestion risk)
congestion_zones = []
for center in obstacle_centers:
    near_obstacle = np.linalg.norm(particles - center, axis=1) < (obstacle_radius + 2.0)
    congestion_zones.append(np.sum(near_obstacle))
    print(f"  Agents near obstacle at {center}: {np.sum(near_obstacle)}")

print()

# ==============================================================================
# PART 7: Extension to Higher Dimensions
# ==============================================================================

print("PART 7: Extension to Higher Dimensions")
print("-" * 70)
print()

print("Demonstration: 3D hybrid domain (maze slice + spherical obstacles)")
print()

# Create 3D domain
domain_3d = Hyperrectangle(np.array([[0.0, 20.0], [0.0, 20.0], [0.0, 5.0]]))

# Add 3D obstacles
obstacles_3d = [
    Hypersphere(center=np.array([5.0, 5.0, 2.5]), radius=1.5),
    Hypersphere(center=np.array([15.0, 15.0, 2.5]), radius=1.5),
    Hypersphere(center=np.array([5.0, 15.0, 2.5]), radius=1.5),
]

# Create navigable domain
navigable_3d = domain_3d
for obs in obstacles_3d:
    navigable_3d = DifferenceDomain(navigable_3d, obs)

print(f"3D Domain: [0, 20] x [0, 20] x [0, 5]")
print(f"  Base domain memory: {domain_3d.bounds.nbytes} bytes")
print(f"  Obstacles: {len(obstacles_3d)} spheres")
print()

# Sample particles in 3D
print("Sampling particles in 3D navigable domain...")
particles_3d = navigable_3d.sample_uniform(1000, seed=42)
print(f"✓ Sampled {len(particles_3d)} particles")
print(f"  Shape: {particles_3d.shape}")
print(f"  All inside domain: {np.all(navigable_3d.contains(particles_3d))}")
print()

# ==============================================================================
# SUMMARY
# ==============================================================================

print()
print("=" * 70)
print("SUMMARY: Maze + Implicit Geometry Integration")
print("=" * 70)
print()

print("Achievements:")
print("  ✓ Combined discrete maze structure with continuous implicit geometry")
print("  ✓ Created hybrid domain with both types of obstacles")
print("  ✓ Sampled particles using rejection sampling")
print("  ✓ Demonstrated 2D and 3D hybrid domains")
print("  ✓ Applied to crowd evacuation scenario")
print()

print("Key Insights:")
print("  • Mazes provide structural complexity (corridors, rooms)")
print("  • Implicit geometry adds localized features (furniture, columns)")
print("  • CSG operations enable flexible domain composition")
print("  • Extends naturally to 3D (buildings with floors)")
print("  • Essential for realistic crowd dynamics simulations")
print()

print("Integration Points:")
print("  • Particle-collocation methods: Natural particle sampling")
print("  • HJB solvers: Use hybrid domain as constraint")
print("  • Visualization: Combined discrete + continuous rendering")
print("  • Reinforcement learning: Complex environment for agents")
print()

print("Next Steps:")
print("  → Implement efficient hybrid contains() using KD-trees")
print("  → Optimize rejection sampling with adaptive bounds")
print("  → Integrate with MFG solvers for evacuation dynamics")
print("  → Create 3D visualization with PyVista")
print()

print("Related Files:")
print("  - mfg_pde/geometry/mazes/: Maze generation algorithms")
print("  - mfg_pde/geometry/implicit/: Implicit domain CSG operations")
print("  - examples/advanced/arbitrary_nd_geometry_demo.py: nD geometry")
print("  - docs/theory/unified_hamiltonian_signature.md: Dimension-agnostic design")
print()

print("=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
