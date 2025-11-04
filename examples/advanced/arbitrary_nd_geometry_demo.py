"""
Arbitrary nD Geometry Demonstration for MFG_PDE

This example showcases the power of dimension-agnostic implicit geometry:
1. Same code works in 2D, 4D, 10D, 100D
2. O(d) memory complexity (no mesh curse of dimensionality)
3. CSG operations for complex domains
4. Integration with particle methods
5. Unified Hamiltonian signature H(t, x, p, m)

Key Innovation: Implicit domains via signed distance functions (SDFs) enable
true arbitrary nD support without mesh generation.

Author: MFG_PDE Team
Date: November 2025
"""

from __future__ import annotations

import time

import numpy as np

from mfg_pde.geometry.implicit import DifferenceDomain, Hyperrectangle, Hypersphere

print("=" * 70)
print("ARBITRARY nD GEOMETRY DEMONSTRATION")
print("=" * 70)
print()

# ==============================================================================
# PART 1: Dimension Scaling - Same Code, Any Dimension
# ==============================================================================

print("PART 1: Dimension Scaling")
print("-" * 70)
print()
print("Demonstrating O(d) memory complexity vs O(N^d) for meshes")
print()

dimensions = [2, 4, 10, 50, 100]
n_samples = 10000

results = []

for d in dimensions:
    print(f"Dimension d={d}:")
    print(f"  Creating domain [0,1]^{d}...")

    # Same code for all dimensions!
    bounds = np.array([[0.0, 1.0]] * d)
    domain = Hyperrectangle(bounds)

    # Memory footprint
    memory_bytes = bounds.nbytes
    print(f"  Memory: {memory_bytes} bytes (O(d) = {2 * d * 8} bytes)")

    # Sampling performance
    start = time.time()
    particles = domain.sample_uniform(n_samples, seed=42)
    elapsed = time.time() - start

    # Verification
    assert particles.shape == (n_samples, d)
    assert np.all(domain.contains(particles))

    print(f"  Sampled {n_samples} particles in {elapsed:.3f}s")
    print(f"  Sampling rate: {n_samples / elapsed:.0f} particles/second")

    # Mesh comparison (hypothetical)
    if d <= 3:
        N = 50  # Grid points per dimension
        mesh_points = N**d
        mesh_memory = mesh_points * d * 8  # bytes for coordinates
        print(f"  [Compare] Mesh with N={N}: {mesh_points:,} points, {mesh_memory / 1e6:.1f} MB")
    else:
        print(f"  [Compare] Mesh infeasible for d={d} (curse of dimensionality)")

    print()
    results.append({"d": d, "memory": memory_bytes, "time": elapsed})

print()
print("Key Insight: Memory scales as O(d), not O(N^d)")
print("            Sampling time essentially constant!")
print()

# ==============================================================================
# PART 2: CSG Operations - Complex Domains via Set Operations
# ==============================================================================

print()
print("PART 2: CSG Operations (Constructive Solid Geometry)")
print("-" * 70)
print()

print("Creating complex domain: Room with spherical obstacle")
print()

# Works in any dimension - let's demonstrate in 2D and 4D
for d in [2, 4]:
    print(f"{d}D Domain:")

    # Base room
    room = Hyperrectangle(np.array([[0.0, 10.0]] * d))
    print(f"  Room: [0,10]^{d}")

    # Obstacle at center
    center = np.array([5.0] * d)
    radius = 2.0
    obstacle = Hypersphere(center=center, radius=radius)
    print(f"  Obstacle: Sphere at {center[:2]}..., radius={radius}")

    # Navigable space = room \\ obstacle
    navigable = DifferenceDomain(room, obstacle)
    print("  Navigable: Room - Obstacle (CSG difference)")

    # Sample particles
    particles = navigable.sample_uniform(5000, seed=42)
    print(f"  Sampled {len(particles)} particles")

    # Verify all particles:
    # 1. Inside room
    # 2. Outside obstacle
    # 3. Inside navigable domain
    in_room = room.contains(particles)
    in_obstacle = obstacle.contains(particles)
    in_navigable = navigable.contains(particles)

    print("  Verification:")
    print(f"    In room: {np.sum(in_room)} / {len(particles)}")
    print(f"    In obstacle: {np.sum(in_obstacle)} / {len(particles)}")
    print(f"    In navigable: {np.sum(in_navigable)} / {len(particles)}")

    assert np.sum(in_navigable) == len(particles), "All particles should be in navigable space"
    assert np.sum(in_obstacle) == 0, "No particles should be in obstacle"

    print()

print("Key Insight: CSG operations work identically in any dimension")
print("            Build complex domains from simple primitives!")
print()

# ==============================================================================
# PART 3: Unified Hamiltonian Signature H(t, x, p, m)
# ==============================================================================

print()
print("PART 3: Unified Hamiltonian Signature")
print("-" * 70)
print()

print("Demonstrating dimension-agnostic Hamiltonian specification")
print()


def hamiltonian_nd(t: float, x: np.ndarray, derivs: tuple[np.ndarray, ...], m: float) -> float:
    """
    Dimension-agnostic Hamiltonian using tuple notation.

    Works for any d:
    - 2D: derivs = (p_x, p_y)
    - 4D: derivs = (p_x1, p_x2, p_x3, p_x4)
    - 10D: derivs = (p_x1, ..., p_x10)

    Args:
        t: Time
        x: Position (d,) array
        derivs: Tuple of gradients (each is (d,) array)
        m: Density

    Returns:
        Hamiltonian value
    """
    p = derivs[0]  # First gradient component

    # Standard quadratic Hamiltonian with crowd aversion
    # H = (1/2)||p||² + (ν/2)m
    kinetic = 0.5 * np.sum(p**2)
    interaction = 0.5 * m  # ν = 1

    return kinetic + interaction


print("Hamiltonian signature: H(t, x, derivs, m)")
print("  t: scalar time")
print("  x: (d,) position vector")
print("  derivs: tuple of (d,) gradient arrays")
print("  m: scalar density")
print()

# Test in different dimensions
for d in [2, 4, 10]:
    print(f"Testing in {d}D:")

    t = 0.5
    x = np.random.rand(d)
    p = np.random.rand(d)
    m = 1.0

    H = hamiltonian_nd(t, x, (p,), m)

    expected = 0.5 * np.sum(p**2) + 0.5 * m

    print(f"  H(t={t:.1f}, x∈ℝ^{d}, p∈ℝ^{d}, m={m}) = {H:.4f}")
    print(f"  Verified: {np.abs(H - expected) < 1e-10}")
    print()

print("Key Insight: Tuple notation `derivs=(p,)` enables dimension-agnostic code")
print("            Same Hamiltonian works in 2D, 4D, 10D, 100D!")
print()

# ==============================================================================
# PART 4: Memory Complexity Comparison
# ==============================================================================

print()
print("PART 4: Memory Complexity Analysis")
print("-" * 70)
print()

print("Domain Storage Requirements:")
print()
print(f"{'Dimension':<12} {'Implicit (bytes)':<20} {'Mesh (N=50, MB)':<20}")
print("-" * 52)

for d in [2, 3, 4, 5, 10]:
    # Implicit: just stores bounds (d, 2) float64
    implicit_bytes = d * 2 * 8

    # Mesh: N^d grid points, each storing d coordinates
    N = 50
    if d <= 5:
        mesh_points = N**d
        mesh_bytes = mesh_points * d * 8
        mesh_mb = mesh_bytes / 1e6
        mesh_str = f"{mesh_mb:.1f}"
    else:
        mesh_str = "Infeasible"

    print(f"{d:<12} {implicit_bytes:<20} {mesh_str:<20}")

print()
print("Key Insight: Implicit geometry breaks the curse of dimensionality")
print("            Mesh methods become infeasible beyond d=3 or 4")
print()

# ==============================================================================
# PART 5: Practical Application - High-D Crowd Dynamics
# ==============================================================================

print()
print("PART 5: Practical Application")
print("-" * 70)
print()

print("Scenario: Crowd dynamics in high-dimensional configuration space")
print("  State space: position + velocity + orientation")
print("  Example: 2D space → 6D state (x, y, vx, vy, θ, ω)")
print()

# 6D configuration space: (x, y, vx, vy, theta, omega)
d = 6
bounds_6d = np.array(
    [
        [0.0, 10.0],  # x position
        [0.0, 10.0],  # y position
        [-2.0, 2.0],  # vx velocity
        [-2.0, 2.0],  # vy velocity
        [0.0, 2 * np.pi],  # theta angle
        [-1.0, 1.0],  # omega angular velocity
    ]
)

domain_6d = Hyperrectangle(bounds_6d)

print("Created 6D domain: position × velocity × orientation")
print(f"  Memory: {bounds_6d.nbytes} bytes")
print()

# Sample initial crowd configuration
n_agents = 1000
print(f"Sampling {n_agents} agents in 6D configuration space...")
start = time.time()
agents = domain_6d.sample_uniform(n_agents, seed=42)
elapsed = time.time() - start

print(f"  Sampled in {elapsed:.3f}s")
print(f"  Agent 0 state: x={agents[0, 0]:.2f}, y={agents[0, 1]:.2f}, vx={agents[0, 2]:.2f}, ...")
print()

# Extract 2D positions for visualization (if needed)
positions_2d = agents[:, :2]
velocities = agents[:, 2:4]
print(
    f"  2D positions range: x∈[{positions_2d[:, 0].min():.1f}, {positions_2d[:, 0].max():.1f}], "
    f"y∈[{positions_2d[:, 1].min():.1f}, {positions_2d[:, 1].max():.1f}]"
)
print(
    f"  Velocity magnitude: {np.linalg.norm(velocities, axis=1).mean():.2f} ± "
    f"{np.linalg.norm(velocities, axis=1).std():.2f}"
)
print()

print("Key Insight: Implicit geometry makes high-D state spaces tractable")
print("            Essential for realistic crowd dynamics with velocity/orientation")
print()

# ==============================================================================
# SUMMARY
# ==============================================================================

print()
print("=" * 70)
print("SUMMARY: Arbitrary nD Geometry Capabilities")
print("=" * 70)
print()

print("Achievements:")
print("  ✓ Same code works in 2D, 4D, 10D, 50D, 100D")
print("  ✓ O(d) memory complexity (no mesh curse)")
print("  ✓ CSG operations for complex domains")
print("  ✓ Unified Hamiltonian H(t, x, derivs, m)")
print("  ✓ High-D applications (6D crowd dynamics)")
print()

print("Next Steps:")
print("  → Integrate with particle-collocation solvers")
print("  → Combine with maze generation for hybrid domains")
print("  → Demonstrate on high-D MFG problems (d ≥ 4)")
print()

print("Related Files:")
print("  - mfg_pde/geometry/implicit/: Implicit domain implementation")
print("  - mfg_pde/geometry/mazes/: Maze generation (newly migrated!)")
print("  - mfg_pde/alg/numerical/fp_solvers/fp_particle.py: Particle-collocation")
print()

print("=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
