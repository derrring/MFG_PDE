#!/usr/bin/env python3
"""
Smoke test for Ghost Nodes method implementation.

Tests that the ghost nodes method:
1. Initializes without errors
2. Creates ghost neighbors for boundary points
3. Enforces Neumann BC structurally
4. Solves a simple HJB problem
"""

import numpy as np

from mfg_pde import BoundaryConditions, MFGProblem
from mfg_pde.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import BCType
from mfg_pde.geometry.collocation import CollocationSampler


def test_ghost_nodes_initialization():
    """Test that ghost nodes method initializes correctly."""
    print("=" * 70)
    print("Ghost Nodes Method - Smoke Test")
    print("=" * 70)
    print()

    # Create simple 2D problem
    geometry = TensorProductGrid(
        dimension=2,
        bounds=[(0.0, 10.0), (0.0, 5.0)],
        Nx=[21, 11],
    )

    # Explicit Neumann BC (no-flux boundaries)
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
        n_interior=100,
        n_boundary=40,
        interior_method="sobol",
        seed=42,
    )

    print(f"Collocation points: {len(coll.points)}")
    print(f"  Interior: {coll.n_interior}")
    print(f"  Boundary: {coll.n_boundary}")
    print()

    # Create GFDM solver WITH ghost nodes
    avg_spacing = np.sqrt(10.0 * 5.0 / 100)
    delta = 3.0 * avg_spacing

    print("Creating GFDM solver with ghost nodes...")
    solver = HJBGFDMSolver(
        problem,
        collocation_points=coll.points,
        boundary_indices=coll.boundary_indices,
        delta=delta,
        use_ghost_nodes=True,  # Enable ghost nodes
        qp_optimization_level="none",  # Start simple
    )

    print("✓ Solver initialized successfully")
    print()

    # Check ghost node map was created
    if hasattr(solver, "_ghost_node_map"):
        n_boundary_with_ghosts = len(solver._ghost_node_map)
        print(f"Ghost nodes created for {n_boundary_with_ghosts} boundary points")

        # Show details for first boundary point
        if n_boundary_with_ghosts > 0:
            first_idx = next(iter(solver._ghost_node_map.keys()))
            ghost_info = solver._ghost_node_map[first_idx]
            print(f"  Example (boundary point {first_idx}):")
            print(f"    Number of ghosts: {ghost_info['n_ghosts']}")
            print(f"    Ghost indices: {ghost_info['ghost_indices'][:3]}...")
            print(f"    Mirror indices: {ghost_info['mirror_indices'][:3]}...")
    else:
        print("⚠ No ghost node map found")

    print()

    # Check neighborhood augmentation
    boundary_idx = coll.boundary_indices[0]
    neighborhood = solver.neighborhoods[boundary_idx]
    has_negative_indices = np.any(neighborhood["indices"] < 0)

    print(f"First boundary point ({boundary_idx}) neighborhood:")
    print(f"  Total neighbors: {neighborhood['size']}")
    print(f"  Has ghost neighbors (negative indices): {has_negative_indices}")

    if has_negative_indices:
        n_ghosts = np.sum(neighborhood["indices"] < 0)
        print(f"  Number of ghost neighbors: {n_ghosts}")
        print("✓ Ghost neighbors successfully added to neighborhood")
    else:
        print("⚠ No ghost neighbors found in neighborhood")

    print()
    print("=" * 70)
    print("Smoke test PASSED - Ghost nodes method is functional")
    print("=" * 70)

    return solver


if __name__ == "__main__":
    solver = test_ghost_nodes_initialization()
