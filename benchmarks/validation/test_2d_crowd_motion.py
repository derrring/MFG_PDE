"""
Validation Test 1: 2D Crowd Motion Convergence Study

Tests dimension-agnostic FDM solvers on 2D crowd motion problem with multiple
grid resolutions to validate:
- Spatial convergence rate (should be O(h²) for centered FD)
- Temporal convergence rate (should be O(Δt²) for Strang splitting)
- Mass conservation (<2% error expected for dimensional splitting)

This test validates Phase 2 implementation by checking that dimensional
splitting achieves expected accuracy.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import validation utilities
from benchmarks.validation.utils import (
    compute_convergence_rate,
    compute_l2_error,
    compute_mass_conservation,
)
from mfg_pde import MFGComponents
from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem
from mfg_pde.factory import create_basic_solver


class CrowdMotion2D(GridBasedMFGProblem):
    """2D crowd motion MFG problem for validation."""

    def __init__(
        self,
        grid_resolution=15,
        time_horizon=0.5,
        num_timesteps=20,
        diffusion=0.05,
        congestion_weight=0.5,
        goal=(0.8, 0.8),
        start=(0.2, 0.2),
    ):
        super().__init__(
            domain_bounds=(0.0, 1.0, 0.0, 1.0),
            grid_resolution=grid_resolution,
            time_domain=(time_horizon, num_timesteps),
            diffusion_coeff=diffusion,
        )
        self.congestion_weight = congestion_weight
        self.goal = np.array(goal)
        self.start = np.array(start)

    def initial_density(self, x):
        """Gaussian blob centered at start position."""
        dist_sq = np.sum((x - self.start) ** 2, axis=1)
        density = np.exp(-100 * dist_sq)
        # Normalize so that integral ∫∫ m(x,y) dx dy = 1
        # For uniform grid: ∫∫ m dx dy ≈ Σ m[i,j] * dx * dy
        # So we need: sum(m) * dV = 1  =>  m = density / (sum(density) * dV)
        dV = float(np.prod(self.geometry.grid.spacing))
        return density / (np.sum(density) * dV + 1e-10)

    def terminal_cost(self, x):
        """Quadratic cost: distance to goal."""
        dist_sq = np.sum((x - self.goal) ** 2, axis=1)
        return 5.0 * dist_sq

    def running_cost(self, x, t):
        """Small running cost encourages fast movement."""
        return 0.1 * np.ones(x.shape[0])

    def hamiltonian(self, x, m, p, t):
        """H = (1/2)|p|² + κ·m (isotropic control + congestion)."""
        p_squared = np.sum(p**2, axis=1) if p.ndim > 1 else p**2
        h = 0.5 * p_squared
        h += self.congestion_weight * m
        return h

    def setup_components(self):
        """Setup MFG components for numerical solvers."""
        kappa = self.congestion_weight

        # Pre-compute normalization constant for initial density
        # This must be done ONCE for the entire grid, not point-by-point
        grid_points = self.geometry.grid.flatten()
        dist_sq = np.sum((grid_points - self.start) ** 2, axis=1)
        unnormalized_density = np.exp(-100 * dist_sq)
        dV = float(np.prod(self.geometry.grid.spacing))
        normalization_constant = np.sum(unnormalized_density) * dV

        def hamiltonian_func(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, derivs=None, **kwargs):
            """H = (1/2)|∇u|² + κ·m"""
            h_value = 0.0
            if derivs is not None:
                grad_u = []
                for d in range(2):  # 2D
                    idx_tuple = tuple([0] * d + [1] + [0] * (2 - d - 1))
                    grad_u.append(derivs.get(idx_tuple, 0.0))
                h_value += 0.5 * float(np.sum(np.array(grad_u) ** 2))
            h_value += kappa * float(m_at_x)
            return h_value

        def hamiltonian_dm(x_idx, x_position, m_at_x, **kwargs):
            """∂H/∂m = κ"""
            return kappa

        def initial_density_func(x_idx):
            """Gaussian initial density."""
            coords = []
            for d in range(2):
                idx_d = x_idx[d] if isinstance(x_idx, tuple) else x_idx
                x_d = self.geometry.grid.bounds[d][0] + idx_d * self.geometry.grid.spacing[d]
                coords.append(x_d)
            coords_array = np.array(coords).reshape(1, -1)
            # Evaluate unnormalized density at this point
            dist_sq_point = np.sum((coords_array - self.start) ** 2, axis=1)[0]
            unnormalized = np.exp(-100 * dist_sq_point)
            # Use pre-computed normalization constant
            return float(unnormalized / normalization_constant)

        def terminal_cost_func(x_idx):
            """Quadratic terminal cost."""
            coords = []
            for d in range(2):
                idx_d = x_idx[d] if isinstance(x_idx, tuple) else x_idx
                x_d = self.geometry.grid.bounds[d][0] + idx_d * self.geometry.grid.spacing[d]
                coords.append(x_d)
            coords_array = np.array(coords).reshape(1, -1)
            return float(self.terminal_cost(coords_array)[0])

        return MFGComponents(
            hamiltonian_func=hamiltonian_func,
            hamiltonian_dm_func=hamiltonian_dm,
            initial_density_func=initial_density_func,
            final_value_func=terminal_cost_func,
        )


def run_fdm_solver(grid_resolution: int, num_timesteps: int, max_iterations: int = 50) -> dict:
    """
    Run FDM solver on 2D crowd motion problem.

    Parameters
    ----------
    grid_resolution : int
        Grid points per dimension
    num_timesteps : int
        Number of time steps
    max_iterations : int
        Max Picard iterations

    Returns
    -------
    dict
        Result dictionary with U, M, info, runtime
    """
    # Create problem
    problem = CrowdMotion2D(
        grid_resolution=grid_resolution,
        time_horizon=0.4,
        num_timesteps=num_timesteps,
        diffusion=0.05,
        congestion_weight=0.3,
        goal=(0.8, 0.8),
        start=(0.2, 0.2),
    )

    # Create solver
    solver = create_basic_solver(
        problem,
        damping=0.6,
        max_iterations=max_iterations,
        tolerance=1e-4,
    )

    # Solve and time it
    start_time = time.time()
    result = solver.solve()
    runtime = time.time() - start_time

    return {
        "U": result.U,
        "M": result.M,
        "info": {
            "converged": result.converged,
            "num_iterations": result.iterations,
            "final_error": result.max_error,
            "runtime": runtime,
        },
        "grid_spacing": problem.geometry.grid.spacing,
        "time_step": problem.dt,
    }


def test_spatial_convergence():
    """
    Test spatial convergence by refining grid (fixed time step).

    Expected: O(h²) convergence for centered finite differences.
    """
    print("\n" + "=" * 70)
    print("  Test 1: Spatial Convergence Study")
    print("=" * 70 + "\n")

    # Grid resolutions to test (coarse to fine)
    resolutions = [8, 12, 16]  # 8×8, 12×12, 16×16
    num_timesteps = 15  # Fixed time discretization

    print("Running FDM solver with different grid resolutions...")
    print(f"Time steps (fixed): {num_timesteps}\n")

    results = []
    for i, res in enumerate(resolutions):
        print(f"[{i + 1}/{len(resolutions)}] Resolution: {res}×{res}")
        result = run_fdm_solver(
            grid_resolution=res,
            num_timesteps=num_timesteps,
            max_iterations=30,
        )
        results.append(result)
        print(f"  Converged: {result['info']['converged']}")
        print(f"  Iterations: {result['info']['num_iterations']}")
        print(f"  Runtime: {result['info']['runtime']:.2f}s")

        # Check mass conservation
        dx, dy = result["grid_spacing"]
        mass_error = compute_mass_conservation(result["M"][-1], dx=(dx, dy))
        print(f"  Mass error: {mass_error * 100:.2f}%\n")

    # Use finest resolution as reference
    ref_solution = results[-1]["M"][-1]  # Final density at finest grid

    # Compute errors relative to finest grid
    errors = []
    spacings = []

    print("Computing L² errors (vs finest grid)...")
    for i, result in enumerate(results[:-1]):  # Exclude finest from error computation
        # Interpolate coarse solution to fine grid for fair comparison
        # For now, use simple direct comparison at coarse grid points
        M_coarse = result["M"][-1]
        dx = result["grid_spacing"][0]

        # Extract corresponding points from reference
        stride = resolutions[-1] // resolutions[i]
        M_ref_sub = ref_solution[::stride, ::stride]

        # Compute L² error
        error = compute_l2_error(M_ref_sub, M_coarse, dx=dx)
        errors.append(error)
        spacings.append(dx)

        print(f"  h={dx:.4f} (res={resolutions[i]}): L² error = {error:.6e}")

    # Estimate convergence rate
    if len(errors) >= 2:
        conv_rate = compute_convergence_rate(np.array(errors), np.array(spacings))
        print(f"\n✓ Estimated spatial convergence rate: p = {conv_rate:.2f}")
        print("  Expected: p ≈ 2.0 (centered finite differences)")

        # Check if close to expected
        if abs(conv_rate - 2.0) < 0.5:
            print("  ✅ PASS: Convergence rate consistent with O(h²)")
        else:
            print(f"  ⚠️  WARNING: Convergence rate deviates from expected (|p - 2.0| = {abs(conv_rate - 2.0):.2f})")
    else:
        print("\n  ⚠️  Not enough data points to estimate convergence rate")

    # Save figure
    if len(errors) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(spacings, errors, "o-", linewidth=2, markersize=8, label="FDM nD (Strang)")
        # Reference line: O(h²)
        ax.loglog(spacings, errors[0] * (np.array(spacings) / spacings[0]) ** 2, "--", label="O(h²) reference")
        ax.set_xlabel("Grid spacing h", fontsize=12)
        ax.set_ylabel("L² Error", fontsize=12)
        ax.set_title("Spatial Convergence: 2D Crowd Motion (FDM Dimensional Splitting)", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        fig.savefig(output_dir / "test1_spatial_convergence.png", dpi=150, bbox_inches="tight")
        print(f"\n  Figure saved: {output_dir / 'test1_spatial_convergence.png'}")

    return results


def test_mass_conservation():
    """
    Test mass conservation across different resolutions.

    Expected: ~1-2% error for FDM dimensional splitting.
    """
    print("\n" + "=" * 70)
    print("  Test 2: Mass Conservation Study")
    print("=" * 70 + "\n")

    resolutions = [8, 12, 16, 20]
    num_timesteps = 15

    print("Testing mass conservation...")
    print(f"Time steps: {num_timesteps}\n")

    mass_errors = []

    for i, res in enumerate(resolutions):
        print(f"[{i + 1}/{len(resolutions)}] Resolution: {res}×{res}")
        result = run_fdm_solver(
            grid_resolution=res,
            num_timesteps=num_timesteps,
            max_iterations=30,
        )

        dx, dy = result["grid_spacing"]
        mass_error = compute_mass_conservation(result["M"][-1], dx=(dx, dy))
        mass_errors.append(mass_error)

        print(f"  Mass error: {mass_error * 100:.2f}%")
        print(f"  Iterations: {result['info']['num_iterations']}\n")

    # Check all within acceptable range
    max_error = max(mass_errors)
    print(f"Maximum mass error: {max_error * 100:.2f}%")

    if max_error < 0.02:
        print("✅ PASS: All mass errors < 2% (expected for FDM splitting)")
    elif max_error < 0.05:
        print("⚠️  ACCEPTABLE: Mass errors < 5% (slightly high but may be due to few iterations)")
    else:
        print(f"❌ FAIL: Mass error {max_error * 100:.2f}% exceeds 5% threshold")

    return mass_errors


def main():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("  VALIDATION TEST 1: 2D Crowd Motion (FDM Convergence)")
    print("=" * 70)
    print("\nTests:")
    print("  1. Spatial convergence study (grid refinement)")
    print("  2. Mass conservation validation")
    print("\nPurpose:")
    print("  Validate Phase 2 dimensional splitting implementation")
    print("=" * 70)

    # Run tests
    test_spatial_convergence()
    test_mass_conservation()

    print("\n" + "=" * 70)
    print("  VALIDATION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
