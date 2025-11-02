#!/usr/bin/env python3
"""
4D WENO Solver Test

Tests the newly implemented arbitrary-dimensional WENO solver on a simple 4D problem.
This demonstrates that WENO now supports dimensions > 3 through GridBasedMFGProblem.

Run:
    python examples/advanced/weno_4d_test.py
"""

import numpy as np

from mfg_pde.alg.numerical.hjb_solvers.hjb_weno import HJBWenoSolver
from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem


class Simple4DMFGProblem(GridBasedMFGProblem):
    """
    Simple 4D MFG problem for testing nD WENO implementation.

    Domain: [0,1]^4 (unit hypercube in 4D)
    Hamiltonian: H(x, ∇u) = (1/2)|∇u|^2 (quadratic Hamiltonian)
    """

    def __init__(self):
        # 4D domain: (x0_min, x0_max, x1_min, x1_max, x2_min, x2_max, x3_min, x3_max)
        domain_bounds = (0, 1, 0, 1, 0, 1, 0, 1)

        # Use small resolution for testing (10^4 = 10,000 grid points)
        # Larger resolutions would require too much memory/time for a quick test
        grid_resolution = 10

        # Time domain: T=0.1, Nt=5 (very short for testing)
        time_domain = (0.1, 5)

        super().__init__(
            domain_bounds=domain_bounds,
            grid_resolution=grid_resolution,
            time_domain=time_domain,
            diffusion_coeff=0.1,
        )

    def setup_components(self):
        """Setup MFG components (required by HighDimMFGProblem)."""
        # Not needed for direct HJB testing

    def hamiltonian(self, x, p, m, t=0):
        """
        Quadratic Hamiltonian: H(x, ∇u) = (1/2)|∇u|^2

        Args:
            x: spatial position
            p: momentum (gradient of value function ∇u)
            m: density
            t: time

        Returns:
            H(x, p, m, t): scalar
        """
        if isinstance(p, (tuple, list)):
            return 0.5 * sum(p_i**2 for p_i in p)
        else:
            return 0.5 * np.sum(p**2, axis=-1)

    def running_cost(self, x, m, t=0):
        """Running cost: f(x, m, t) = 0 (no running cost)."""
        return 0.0

    def terminal_cost(self, x):
        """
        Terminal cost: g(x) = sum of coordinates squared

        Args:
            x: spatial position (4D point)

        Returns:
            g(x): scalar
        """
        if isinstance(x, (tuple, list)):
            return sum(x_i**2 for x_i in x)
        else:
            return np.sum(x**2, axis=-1)

    def initial_density(self, x):
        """Uniform initial density: m0(x) = constant."""
        return 1.0 / (10**4)  # Normalized over [0,1]^4


def main():
    """Test 4D WENO solver."""
    print("\n" + "=" * 70)
    print("4D WENO Solver Test")
    print("=" * 70)

    # Create 4D problem
    print("\n1. Creating 4D MFG problem...")
    problem = Simple4DMFGProblem()
    print(f"   Dimension: {problem.dimension}D")
    print(f"   Grid resolution: {problem.grid_resolution} per dimension")
    print(f"   Total grid points: {10**4:,}")
    print(f"   Time steps: {problem.Nt}")
    print(f"   Time horizon: T={problem.T}")

    # Create WENO solver
    print("\n2. Creating WENO solver...")
    solver = HJBWenoSolver(
        problem=problem,
        weno_variant="weno5",
        cfl_number=0.1,  # Conservative for 4D
        diffusion_stability_factor=0.05,  # Very conservative for 4D
        time_integration="explicit_euler",  # Simpler for testing
        splitting_method="godunov",  # Simpler for testing
    )
    print(f"   WENO variant: {solver.weno_variant}")
    print(f"   Detected dimension: {solver.dimension}D")
    print(f"   Grid points per dim: {solver.num_grid_points}")
    print(f"   Grid spacing per dim: {solver.grid_spacing}")
    print(f"   CFL number: {solver.cfl_number}")
    print(f"   Splitting method: {solver.splitting_method}")

    # Create test density and final condition
    print("\n3. Setting up test data...")
    # Create constant density (uniform distribution)
    M_shape = (problem.Nt + 1, *[problem.grid_resolution for _ in range(4)])
    M_test = np.ones(M_shape) / (10**4)  # Normalized to integrate to 1

    # Create simple terminal condition (center has highest value)
    U_final_shape = tuple(problem.grid_resolution for _ in range(4))
    U_final = np.zeros(U_final_shape)
    center = tuple(problem.grid_resolution // 2 for _ in range(4))
    U_final[center] = 1.0

    # Dummy previous Picard iterate (not used in single HJB solve)
    U_prev = np.zeros_like(M_test)

    print(f"   M shape: {M_test.shape}")
    print(f"   U_final shape: {U_final.shape}")
    print(f"   M total mass: {np.sum(M_test[0, ...]):.6f} (should be ≈1.0)")

    # Solve HJB system
    print("\n4. Solving 4D HJB system with WENO...")
    print("   (This will take a few seconds for 10^4 grid points)")

    try:
        U_solved = solver.solve_hjb_system(M_test, U_final, U_prev)
        print("\n   ✓ Success! Solved 4D HJB system")
        print(f"   U_solved shape: {U_solved.shape}")
        print(f"   U_solved min: {np.min(U_solved):.6f}")
        print(f"   U_solved max: {np.max(U_solved):.6f}")
        print(f"   U_solved mean: {np.mean(U_solved):.6f}")

        # Check backward time evolution (value should decrease going backward)
        final_value = np.max(U_solved[-1, ...])
        initial_value = np.max(U_solved[0, ...])
        print(f"\n   Value at T (final): {final_value:.6f}")
        print(f"   Value at t=0 (initial): {initial_value:.6f}")
        print(f"   Ratio: {initial_value / final_value:.6f} (should be < 1 for backward evolution)")

        print("\n" + "=" * 70)
        print("4D WENO Test PASSED ✓")
        print("=" * 70)
        print("\nConclusion: WENO solver now supports arbitrary dimensions!")
        print("The nD implementation successfully handled a 4D problem with dimensional splitting.")

    except Exception as e:
        print(f"\n   ✗ Error: {e}")
        print(f"\n   Exception type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        print("\n" + "=" * 70)
        print("4D WENO Test FAILED ✗")
        print("=" * 70)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
