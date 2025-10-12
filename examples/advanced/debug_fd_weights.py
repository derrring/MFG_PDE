"""
Debug FD weight extraction to understand what's failing.
"""

import numpy as np

from mfg_pde.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfg_pde.core.mfg_problem import MFGProblem


class TestableGFDMSolver(HJBGFDMSolver):
    """Minimal concrete implementation for testing."""

    def solve_hjb_system(self, M_density_evolution_from_FP, U_final_condition_at_T, U_from_prev_picard):
        """Minimal implementation (not used in this test)."""
        return np.zeros_like(U_from_prev_picard)


def main():
    # Create simple problem
    problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=51, T=1.0, Nt=51, sigma=0.2, coefCT=0.5)

    # Create collocation points
    n_points = 10
    x = np.linspace(problem.xmin, problem.xmax, n_points)
    collocation_points = x.reshape(-1, 1)

    # Create solver
    solver = TestableGFDMSolver(
        problem=problem,
        collocation_points=collocation_points,
        delta=0.15,
        taylor_order=2,
        weight_function="wendland",
    )

    print("Testing FD weight extraction...")
    print(f"Number of collocation points: {n_points}")
    print(f"Multi-indices: {solver.multi_indices}")
    print()

    # Test first interior point (not boundary)
    point_idx = 5
    print(f"Testing point {point_idx} at x={collocation_points[point_idx, 0]:.3f}")
    print()

    taylor_data = solver.taylor_matrices[point_idx]

    if taylor_data is None:
        print("ERROR: Taylor data is None")
        return

    print(f"Taylor data keys: {taylor_data.keys()}")
    print(f"use_svd: {taylor_data.get('use_svd')}")
    print(f"use_qr: {taylor_data.get('use_qr')}")
    print()

    # Find Laplacian index
    laplacian_idx = None
    for k, beta in enumerate(solver.multi_indices):
        if beta == (2,):
            laplacian_idx = k
            print(f"Laplacian index: {laplacian_idx} (beta={beta})")
            break

    if laplacian_idx is None:
        print("ERROR: No Laplacian in multi-indices")
        return

    print()

    # Try to extract weights
    print("Attempting FD weight extraction...")
    weights = solver._compute_fd_weights_from_taylor(taylor_data, laplacian_idx)

    if weights is None:
        print("ERROR: Weight extraction returned None")

        # Debug why
        if taylor_data.get("use_svd"):
            print("\nDEBUG SVD path:")
            U = taylor_data["U"]
            S = taylor_data["S"]
            Vt = taylor_data["Vt"]
            sqrt_W = taylor_data["sqrt_W"]

            print(f"  U shape: {U.shape}")
            print(f"  S shape: {S.shape}")
            print(f"  S values: {S}")
            print(f"  Vt shape: {Vt.shape}")
            print(f"  sqrt_W shape: {sqrt_W.shape}")

            n_derivs = len(solver.multi_indices)
            print(f"  n_derivs: {n_derivs}")

            e_beta = np.zeros(n_derivs)
            e_beta[laplacian_idx] = 1.0
            print(f"  e_beta shape: {e_beta.shape}")
            print(f"  e_beta: {e_beta}")

            # Try computation step by step
            try:
                print("\n  Step 1: sqrt_W @ e_beta")
                result1 = sqrt_W @ e_beta
                print(f"    Result shape: {result1.shape}")
            except Exception as e:
                print(f"    ERROR: {e}")

    else:
        print(f"SUCCESS: Extracted weights with shape {weights.shape}")
        print(f"Weights: {weights}")


if __name__ == "__main__":
    main()
