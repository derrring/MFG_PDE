"""
Diagnostic script to trace mass conservation failure in 2D crowd motion test.

Checks:
1. Initial density normalization
2. Mass at each timestep during solve
3. Mass conservation computation correctness
"""

from __future__ import annotations

import numpy as np

from mfg_pde import MFGComponents, MFGProblem
from mfg_pde.factory import create_basic_solver


class CrowdMotion2D(MFGProblem):
    """2D crowd motion MFG problem for diagnosis."""

    def __init__(
        self,
        grid_resolution=8,
        time_horizon=0.4,
        num_timesteps=15,
        diffusion=0.05,
        congestion_weight=0.3,
        goal=(0.8, 0.8),
        start=(0.2, 0.2),
    ):
        super().__init__(
            spatial_bounds=[(0.0, 1.0), (0.0, 1.0)],
            spatial_discretization=[grid_resolution, grid_resolution],
            T=time_horizon,
            Nt=num_timesteps,
            sigma=diffusion,
        )
        self.grid_resolution = grid_resolution  # Store for convenience
        self.congestion_weight = congestion_weight
        self.goal = np.array(goal)
        self.start = np.array(start)

    def initial_density(self, x):
        """Gaussian blob centered at start position."""
        dist_sq = np.sum((x - self.start) ** 2, axis=1)
        density = np.exp(-100 * dist_sq)
        # Normalization: sum(density * dV) should equal 1
        return density / (np.sum(density) + 1e-10)

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
            return float(self.initial_density(coords_array)[0])

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


def compute_total_mass_2d(M: np.ndarray, dx: float, dy: float) -> float:
    """
    Compute total mass by integrating over 2D grid.

    For 2D: ∫∫ m(x,y) dx dy ≈ Σ m[i,j] * dx * dy
    """
    return float(np.sum(M) * dx * dy)


def diagnose_initial_density():
    """Check if initial density is properly normalized."""
    print("\n" + "=" * 70)
    print("  DIAGNOSIS 1: Initial Density Normalization")
    print("=" * 70 + "\n")

    problem = CrowdMotion2D(grid_resolution=8, time_horizon=0.4, num_timesteps=15)

    # Get grid spacing
    dx, dy = problem.geometry.grid.spacing
    print(f"Grid spacing: dx = {dx:.6f}, dy = {dy:.6f}")
    print(f"Volume element: dV = dx * dy = {dx * dy:.6f}")

    # Get all grid points
    grid_points = problem.geometry.grid.get_all_points()
    print(f"Total grid points: {len(grid_points)}")

    # Compute initial density at all points
    initial_m = problem.initial_density(grid_points)
    print("\nInitial density stats:")
    print(f"  Shape: {initial_m.shape}")
    print(f"  Min: {np.min(initial_m):.6e}")
    print(f"  Max: {np.max(initial_m):.6e}")
    print(f"  Sum: {np.sum(initial_m):.6e}")

    # Check normalization: should sum to 1
    # BUT: This is wrong! We normalized by sum, not by sum * dV
    print(f"\n  Σ m[i,j] = {np.sum(initial_m):.6e}")
    print(f"  Σ m[i,j] * dV = {np.sum(initial_m) * dx * dy:.6e}")

    # ISSUE IDENTIFIED: initial_density() normalizes by sum(density)
    # but should normalize by sum(density * dV) to get unit mass
    print("\n❌ ISSUE FOUND: initial_density() normalizes incorrectly!")
    print("   Current: density / sum(density) → sum = 1.0")
    print("   Correct: density / sum(density * dV) → integral = 1.0")
    print(f"   Result: Actual mass = {np.sum(initial_m) * dx * dy:.6e} (should be 1.0)")

    return problem, initial_m


def diagnose_mass_during_solve():
    """Track mass at each Picard iteration during solve."""
    print("\n" + "=" * 70)
    print("  DIAGNOSIS 2: Mass During Solve")
    print("=" * 70 + "\n")

    problem = CrowdMotion2D(grid_resolution=8, time_horizon=0.4, num_timesteps=5)  # Shorter for speed
    dx, dy = problem.geometry.grid.spacing

    print(f"Running solver for {problem.nt} timesteps...")
    print(f"Grid: {problem.geometry.grid.resolution[0]}×{problem.geometry.grid.resolution[1]}")

    # Create solver
    solver = create_basic_solver(
        problem,
        damping=0.6,
        max_iterations=10,  # Just a few iterations to trace
        tolerance=1e-4,
    )

    # Solve
    result = solver.solve()

    print(f"\nSolver completed: {result.iterations} iterations")
    print(f"Converged: {result.converged}")

    # Check mass at final timestep
    M_final = result.M[-1]  # Last timestep
    print("\nFinal density M(T, x, y):")
    print(f"  Shape: {M_final.shape}")
    print(f"  Min: {np.min(M_final):.6e}")
    print(f"  Max: {np.max(M_final):.6e}")
    print(f"  Sum: {np.sum(M_final):.6e}")

    mass_final = compute_total_mass_2d(M_final, dx, dy)
    print(f"\nFinal mass: ∫∫ M(T,x,y) dx dy = {mass_final:.6e}")
    print(f"Mass error: {abs(mass_final - 1.0) / 1.0 * 100:.2f}%")

    # Check mass at each timestep
    print("\nMass at each timestep:")
    masses = []
    for t_idx in range(result.M.shape[0]):
        mass_t = compute_total_mass_2d(result.M[t_idx], dx, dy)
        masses.append(mass_t)
        print(f"  t={t_idx}: mass = {mass_t:.6e}")

    masses = np.array(masses)
    mass_loss = abs(masses[-1] - masses[0])
    print(f"\nTotal mass loss: {mass_loss:.6e}")
    print(f"Relative mass loss: {mass_loss / masses[0] * 100:.2f}%")

    return result, masses


def diagnose_mass_computation():
    """Check if mass computation itself is correct."""
    print("\n" + "=" * 70)
    print("  DIAGNOSIS 3: Mass Computation Correctness")
    print("=" * 70 + "\n")

    # Create a simple test case: uniform density = 1 / area
    # For domain [0,1] x [0,1], if M = 1/area = 1.0, then ∫∫ M dx dy = 1.0

    nx, ny = 8, 8
    dx, dy = 1.0 / nx, 1.0 / ny

    # Uniform density
    M_uniform = np.ones((nx, ny)) / (1.0 * 1.0)  # 1 / area
    mass_uniform = compute_total_mass_2d(M_uniform, dx, dy)

    print("Test case: Uniform density on [0,1] x [0,1]")
    print(f"  Grid: {nx} x {ny}")
    print(f"  dx = {dx:.6f}, dy = {dy:.6f}")
    print(f"  M[i,j] = {M_uniform[0, 0]:.6f} (constant)")
    print("  Expected mass: 1.0")
    print(f"  Computed mass: {mass_uniform:.6e}")
    print(f"  Error: {abs(mass_uniform - 1.0):.6e}")

    if abs(mass_uniform - 1.0) < 1e-10:
        print("\n✅ Mass computation is correct!")
    else:
        print("\n❌ Mass computation has error!")

    # Test with Gaussian
    print("\n--- Testing with Gaussian density ---")
    problem = CrowdMotion2D(grid_resolution=8)
    grid_points = problem.geometry.grid.get_all_points()
    m_gaussian = problem.initial_density(grid_points)

    # Reshape to 2D grid
    m_grid = m_gaussian.reshape(problem.geometry.grid.resolution)
    mass_gaussian = compute_total_mass_2d(m_grid, *problem.geometry.grid.spacing)

    print("Gaussian initial density:")
    print(f"  Σ m[i,j] = {np.sum(m_grid):.6e}")
    print(f"  ∫∫ m dx dy = {mass_gaussian:.6e}")
    print("  Expected: 1.0")
    print(f"  Error: {abs(mass_gaussian - 1.0) * 100:.2f}%")


def main():
    """Run all diagnostics."""
    print("\n" + "=" * 70)
    print("  MASS CONSERVATION DIAGNOSTIC SUITE")
    print("=" * 70)

    # Diagnosis 1: Initial density normalization
    _, _initial_m = diagnose_initial_density()

    # Diagnosis 2: Mass during solve
    _result, _masses = diagnose_mass_during_solve()

    # Diagnosis 3: Mass computation correctness
    diagnose_mass_computation()

    print("\n" + "=" * 70)
    print("  DIAGNOSIS COMPLETE")
    print("=" * 70 + "\n")

    print("SUMMARY:")
    print("--------")
    print("The issue is in CrowdMotion2D.initial_density():")
    print("  Current normalization: density / sum(density)")
    print("  Should be: density / (sum(density) * dV)")
    print("")
    print("Fix: Change line in initial_density():")
    print("  FROM: return density / (np.sum(density) + 1e-10)")
    print("  TO:   dV = problem.geometry.grid.spacing")
    print("        return density / (np.sum(density) * np.prod(dV) + 1e-10)")


if __name__ == "__main__":
    main()
