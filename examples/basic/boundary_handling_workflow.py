"""
Unified Boundary Handling Workflow Example

This example demonstrates the new unified workflow for boundary condition handling
introduced in v0.16.17 (Issue #545).

Shows:
1. Using geometry.get_boundary_indices() for detection
2. Using geometry.get_boundary_info() for combined detection + normals
3. Composition pattern for solver BC handling

Related:
- Issue #545 (Mixin Refactoring)
- docs/development/BOUNDARY_HANDLING.md
"""

import numpy as np

from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import no_flux_bc


def main():
    """Demonstrate unified boundary handling workflow."""

    # ==========================================================================
    # Setup: Create 2D geometry
    # ==========================================================================

    print("=" * 70)
    print("Unified Boundary Handling Workflow Example")
    print("=" * 70)

    geometry = TensorProductGrid(
        dimension=2,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[21, 21],  # 21x21 grid → 441 total points
        boundary_conditions=no_flux_bc(dimension=2),
    )

    print(f"\\nGeometry: {geometry.dimension}D grid")
    print(f"Grid shape: {geometry.get_grid_shape()}")
    print(f"Total points: {geometry.num_spatial_points}")

    # ==========================================================================
    # Example 1: Basic Boundary Detection
    # ==========================================================================

    print("\\n" + "-" * 70)
    print("Example 1: Basic Boundary Detection")
    print("-" * 70)

    # Get collocation points
    points = geometry.get_collocation_points()
    print(f"Collocation points shape: {points.shape}")

    # OLD WAY: Manual boolean mask
    print("\\n[OLD WAY] Manual detection:")
    on_boundary_mask = geometry.is_on_boundary(points)
    boundary_indices_manual = np.where(on_boundary_mask)[0]
    print(f"  on_boundary_mask shape: {on_boundary_mask.shape}")
    print(f"  boundary_indices (manual): {len(boundary_indices_manual)} points")

    # NEW WAY: Use geometry helper
    print("\\n[NEW WAY] Using get_boundary_indices():")
    boundary_indices = geometry.get_boundary_indices(points)
    print(f"  boundary_indices shape: {boundary_indices.shape}")
    print(f"  boundary_indices: {len(boundary_indices)} points")

    # Verify equivalence
    assert np.array_equal(boundary_indices, boundary_indices_manual)
    print("  ✓ Results match!")

    # For 21x21 grid: 21 + 21 + 19 + 19 = 80 boundary points
    expected_boundary_points = 2 * 21 + 2 * 19
    assert len(boundary_indices) == expected_boundary_points
    print(f"  ✓ Expected {expected_boundary_points} boundary points")

    # ==========================================================================
    # Example 2: Boundary Info (Indices + Normals)
    # ==========================================================================

    print("\\n" + "-" * 70)
    print("Example 2: Boundary Info (Indices + Normals)")
    print("-" * 70)

    # OLD WAY: Two separate calls
    print("\\n[OLD WAY] Two separate calls:")
    boundary_indices_old = geometry.get_boundary_indices(points)
    boundary_points_old = points[boundary_indices_old]
    normals_old = geometry.get_boundary_normal(boundary_points_old)
    print(f"  boundary_indices: {len(boundary_indices_old)}")
    print(f"  normals shape: {normals_old.shape}")

    # NEW WAY: Combined call
    print("\\n[NEW WAY] Using get_boundary_info():")
    boundary_indices_new, normals_new = geometry.get_boundary_info(points)
    print(f"  boundary_indices: {len(boundary_indices_new)}")
    print(f"  normals shape: {normals_new.shape}")

    # Verify equivalence
    assert np.array_equal(boundary_indices_new, boundary_indices_old)
    assert np.allclose(normals_new, normals_old)
    print("  ✓ Results match!")

    # Verify normals are unit vectors
    norms = np.linalg.norm(normals_new, axis=1)
    assert np.allclose(norms, 1.0)
    print("  ✓ All normals are unit vectors (‖n‖ = 1)")

    # Show some boundary points and normals
    print("\\n  Sample boundary points and normals:")
    for i in range(min(5, len(boundary_indices_new))):
        idx = boundary_indices_new[i]
        point = points[idx]
        normal = normals_new[i]
        print(f"    Point {idx}: {point} → Normal: {normal}")

    # ==========================================================================
    # Example 3: Particle Solver Pattern (Composition)
    # ==========================================================================

    print("\\n" + "-" * 70)
    print("Example 3: Particle Solver Pattern (Composition)")
    print("-" * 70)

    class SimpleParticleSolver:
        """
        Example particle solver using composition (not mixins).

        Demonstrates unified workflow:
        - Use geometry for boundary detection
        - Use geometry for normals
        - Explicit composition (no implicit state)
        """

        def __init__(self, geometry, n_particles=100):
            # Explicit dependencies
            self.geometry = geometry
            self.n_particles = n_particles

            # Initialize particles randomly in domain
            bounds_result = geometry.get_bounds()
            if bounds_result is not None:
                min_coords, max_coords = bounds_result
                self.particles = np.random.uniform(min_coords, max_coords, size=(n_particles, geometry.dimension))
            else:
                # Fallback for unbounded domains
                self.particles = np.random.randn(n_particles, geometry.dimension)

            # Initialize random velocities
            self.velocities = 0.1 * np.random.randn(n_particles, geometry.dimension)

        def apply_boundary_reflection(self):
            """
            Apply reflection BC using geometry methods.

            OLD WAY (mixin):
                - Custom boundary detection in solver
                - Custom normal computation
                - Implicit state from mixin

            NEW WAY (composition):
                - Use geometry.get_boundary_info()
                - Clear data flow
            """
            # Step 0: Project any outside particles to boundary first
            # (project_to_interior brings outside points to nearest boundary)
            projected_particles = self.geometry.project_to_interior(self.particles)

            # Step 1: Detect which particles were outside (now on boundary)
            was_outside = np.linalg.norm(projected_particles - self.particles, axis=1) > 1e-10

            if not np.any(was_outside):
                return 0  # No particles needed correction

            # Update positions to projected (on boundary)
            self.particles = projected_particles

            # Step 2: Get normals for particles that were outside
            outside_indices = np.where(was_outside)[0]
            boundary_points = self.particles[outside_indices]
            normals = self.geometry.get_boundary_normal(boundary_points)

            # Step 3: Apply reflection formula: v_new = v - 2(v·n)n
            for i, idx in enumerate(outside_indices):
                velocity = self.velocities[idx]
                normal = normals[i]
                v_normal = np.dot(velocity, normal)  # Normal component
                self.velocities[idx] = velocity - 2.0 * v_normal * normal

            return len(outside_indices)

    # Create solver instance
    print("\\nCreating particle solver with 200 particles...")
    solver = SimpleParticleSolver(geometry, n_particles=200)

    # Move some particles outside to trigger reflection
    print("Moving some particles outside domain...")
    solver.particles[:20, 0] = 1.05  # Push 20 particles past right boundary
    solver.particles[20:40, 1] = -0.05  # Push 20 particles past bottom boundary

    # Apply reflection
    n_reflected = solver.apply_boundary_reflection()
    print(f"Reflected {n_reflected} particles")

    # Verify all particles are inside
    on_boundary = geometry.is_on_boundary(solver.particles, tolerance=1e-8)
    outside = np.any((solver.particles < 0) | (solver.particles > 1), axis=1)
    n_outside = np.sum(outside & ~on_boundary)
    assert n_outside == 0, f"Found {n_outside} particles outside domain!"
    print("✓ All particles inside domain (or on boundary)")

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\\n" + "=" * 70)
    print("Summary: Unified Boundary Handling Workflow")
    print("=" * 70)
    print("""
Key Principles:
1. Use geometry.get_boundary_indices() for detection
2. Use geometry.get_boundary_info() for combined detection + normals
3. Use composition over mixins (explicit dependencies)
4. Delegate to geometry for boundary operations

Benefits:
- No duplicate BC detection logic
- Clear data flow
- Easier testing
- Better maintainability

See docs/development/BOUNDARY_HANDLING.md for full documentation.
""")


if __name__ == "__main__":
    main()
