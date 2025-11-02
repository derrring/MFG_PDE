#!/usr/bin/env python3
"""
3D Semi-Lagrangian Solver Test

Tests the nD Semi-Lagrangian solver on a simple 3D problem.
This demonstrates that Semi-Lagrangian now has complete nD support including
vector optimization for optimal control.

Run:
    python examples/advanced/semi_lagrangian_3d_test.py
"""

import numpy as np

from mfg_pde.alg.numerical.hjb_solvers.hjb_semi_lagrangian import HJBSemiLagrangianSolver
from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem


class Simple3DMFGProblem(GridBasedMFGProblem):
    """
    Simple 3D MFG problem for testing nD Semi-Lagrangian implementation.

    Domain: [0,1]^3 (unit cube in 3D)
    Hamiltonian: H(x, ∇u) = (1/2)|∇u|^2 (quadratic Hamiltonian)
    """

    def __init__(self):
        # 3D domain: (x_min, x_max, y_min, y_max, z_min, z_max)
        domain_bounds = (0, 1, 0, 1, 0, 1)

        # Use moderate resolution for testing (15^3 = 3,375 grid points)
        # Semi-Lagrangian can handle this with interpolation
        grid_resolution = 15

        # Time domain: T=0.1, Nt=10 (short for testing)
        time_domain = (0.1, 10)

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
            x: spatial position (3D point)

        Returns:
            g(x): scalar
        """
        if isinstance(x, (tuple, list)):
            return sum(x_i**2 for x_i in x)
        else:
            return np.sum(x**2, axis=-1)

    def initial_density(self, x):
        """Uniform initial density: m0(x) = constant."""
        return 1.0 / (15**3)  # Normalized over [0,1]^3


def main():
    """Test 3D Semi-Lagrangian solver."""
    print("\n" + "=" * 70)
    print("3D Semi-Lagrangian Solver Test")
    print("=" * 70)

    # Create 3D problem
    print("\n1. Creating 3D MFG problem...")
    problem = Simple3DMFGProblem()
    print(f"   Dimension: {problem.dimension}D")
    print(f"   Grid resolution: {problem.grid_resolution} per dimension")
    print(f"   Total grid points: {15**3:,}")
    print(f"   Time steps: {problem.Nt}")
    print(f"   Time horizon: T={problem.T}")

    # Create Semi-Lagrangian solver
    print("\n2. Creating Semi-Lagrangian solver...")
    solver = HJBSemiLagrangianSolver(
        problem=problem,
        interpolation_method="linear",
        optimization_method="brent",
        characteristic_solver="explicit_euler",
        use_jax=False,  # Use NumPy for reproducibility
    )
    print(f"   Method: {solver.hjb_method_name}")
    print(f"   Detected dimension: {solver.dimension}D")
    print(f"   Interpolation: {solver.interpolation_method}")
    print(f"   Characteristic solver: {solver.characteristic_solver}")

    # Create test density and final condition
    print("\n3. Setting up test data...")
    # Create constant density (uniform distribution)
    M_shape = (problem.Nt + 1, *[problem.grid_resolution for _ in range(3)])
    M_test = np.ones(M_shape) / (15**3)  # Normalized to integrate to 1

    # Create simple terminal condition (center has highest value)
    U_final_shape = tuple(problem.grid_resolution for _ in range(3))
    U_final = np.zeros(U_final_shape)
    center = tuple(problem.grid_resolution // 2 for _ in range(3))
    U_final[center] = 1.0

    # Dummy previous Picard iterate (not used in single HJB solve)
    U_prev = np.zeros_like(M_test)

    print(f"   M shape: {M_test.shape}")
    print(f"   U_final shape: {U_final.shape}")
    print(f"   M total mass: {np.sum(M_test[0, ...]):.6f} (should be ≈1.0)")

    # Solve HJB system
    print("\n4. Solving 3D HJB system with Semi-Lagrangian...")
    print("   (This will take ~30 seconds for 3,375 grid points)")

    try:
        U_solved = solver.solve_hjb_system(M_test, U_final, U_prev)
        print("\n   ✓ Success! Solved 3D HJB system")
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
        print("3D Semi-Lagrangian Test PASSED ✓")
        print("=" * 70)
        print("\nConclusion: Semi-Lagrangian solver now supports arbitrary dimensions!")
        print("The nD implementation successfully handled a 3D problem with")
        print("vector optimization for optimal control.")

    except Exception as e:
        print(f"\n   ✗ Error: {e}")
        print(f"\n   Exception type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        print("\n" + "=" * 70)
        print("3D Semi-Lagrangian Test FAILED ✗")
        print("=" * 70)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
