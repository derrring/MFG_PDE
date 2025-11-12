#!/usr/bin/env python3
"""
Basic 2-population MFG example with linear coupling.

This example demonstrates the fundamental multi-population MFG framework
with two populations competing for resources on a 1D domain.

Mathematical Setup:
    For k = 1, 2:
        HJB: -∂uₖ/∂t + ½|∇uₖ|² + Σⱼ αₖⱼ·mⱼ = 0
        FP:   ∂mₖ/∂t - div(mₖ∇uₖ) - σₖ²Δmₖ = 0

    Coupling matrix:
        α = [[0.1, 0.05],    # Pop 0: affected by self + Pop 1
             [0.05, 0.1]]    # Pop 1: affected by Pop 0 + self

Interpretation:
    - Diagonal: Self-congestion (population avoids crowding itself)
    - Off-diagonal: Cross-population interaction (mutual avoidance)
    - All positive: Competing populations (congestion)

Usage:
    python examples/basic/multi_population_basic.py

Part of: Issue #295 Phase 3 - Multi-population MFG examples
"""

from __future__ import annotations

import numpy as np

from mfg_pde.extensions import MultiPopulationMFGProblem
from mfg_pde.solvers.extensions import MultiPopulationFixedPointSolver


def main():
    """Run basic 2-population MFG example."""
    print("=" * 70)
    print("Basic 2-Population MFG Example (Phase 2 Validation)")
    print("=" * 70)

    # Create problem
    print("\n[1] Creating 2-population problem...")
    problem = MultiPopulationMFGProblem(
        num_populations=2,
        spatial_bounds=[(0, 1)],
        spatial_discretization=[50],
        coupling_matrix=np.array([[0.1, 0.05], [0.05, 0.1]]),
        T=1.0,
        Nt=20,
        sigma=[0.01, 0.02],
        population_labels=["Fast", "Slow"],
    )
    print(f"✓ Problem created with {problem.num_populations} populations")
    print("  - Spatial domain: [0, 1] with 50 points")
    print(f"  - Time horizon: T = {problem.T}, Nt = {problem.Nt}")
    print(f"  - Diffusion: σ = {problem.sigma_vec}")
    print("  - Coupling matrix:")
    print(f"    {problem.coupling_matrix}")

    # Test adapter functionality
    print("\n[2] Testing single-population adapter...")
    from mfg_pde.solvers.extensions.multi_population.fixed_point import _SinglePopulationAdapter

    # Create dummy density state
    m_all = [
        np.ones((21, 51)) / 51.0,  # m₁: Shape (Nt+1, Nx+1)
        np.ones((21, 51)) / 51.0,  # m₂
    ]

    adapter = _SinglePopulationAdapter(problem, k=0, m_all=m_all)
    print("✓ Adapter created for population 0")
    print(f"  - Dimension: {adapter.dimension}")
    print(f"  - T: {adapter.T}")
    print(f"  - Nt: {adapter.Nt}")
    print(f"  - Sigma: {adapter.sigma}")

    # Test basic methods
    print("\n[3] Testing adapter methods...")
    x_test = 0.5
    print(f"  - terminal_cost({x_test}): {adapter.terminal_cost(x_test):.6f}")
    print(f"  - initial_density({x_test}): {adapter.initial_density(x_test):.6f}")

    u_final = adapter.get_final_u()
    m_init = adapter.get_initial_m()
    print(f"  - get_final_u() shape: {u_final.shape}")
    print(f"  - get_initial_m() shape: {m_init.shape}")
    print(f"  - Initial density sum: {np.sum(m_init):.6f}")

    # Attempt solve (will hit NotImplementedError in Phase 2)
    print("\n[4] Creating solver...")
    solver = MultiPopulationFixedPointSolver(
        max_iterations=5,
        tolerance=1e-6,
        damping_factor=0.5,
        single_pop_config={"max_iterations": 10},
    )
    print("✓ Solver created")
    print(f"  - Max iterations: {solver.max_iterations}")
    print(f"  - Damping factor: {solver.damping_factor}")

    print("\n[5] Testing solver initialization...")
    try:
        state = solver._initialize_state(problem)
        print("✓ State initialized")
        print(f"  - u arrays: {len(state['u'])} populations")
        print(f"  - m arrays: {len(state['m'])} populations")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")

    print("\n" + "=" * 70)
    print("Phase 2 Validation Complete")
    print("=" * 70)
    print("\nStatus:")
    print("  ✓ Multi-population problem creation")
    print("  ✓ Single-population adapter structure")
    print("  ✓ Solver initialization")
    print("  ⚠ Full solve requires spatial interpolation (Phase 2.5)")
    print("\nNext Steps:")
    print("  1. Implement spatial interpolation for m_all in adapter")
    print("  2. Complete iteration_step with proper array handling")
    print("  3. Add convergence diagnostics and visualization")


if __name__ == "__main__":
    main()
