"""
1D Simple MFG Validation Test - Mass Conservation Check

Simplest possible test to isolate the mass conservation bug:
- 1D domain [0,1] with N=32 points
- Gaussian initial density
- Quadratic terminal cost
- Track mass at every timestep

This test isolates Bug #2 in a simpler 1D setting.
"""

from __future__ import annotations

import time

import numpy as np

from mfg_pde import MFGComponents, MFGProblem
from mfg_pde.factory import create_basic_solver


class Simple1DMFG(MFGProblem):
    """Simplest possible 1D MFG for mass conservation testing."""

    def __init__(
        self,
        grid_resolution=32,
        time_horizon=0.5,
        num_timesteps=15,
        diffusion=0.05,
    ):
        super().__init__(
            spatial_bounds=[(0.0, 1.0)],
            spatial_discretization=[grid_resolution],
            T=time_horizon,
            Nt=num_timesteps,
            sigma=diffusion,
        )
        self.grid_resolution = grid_resolution  # Store for convenience
        self.center = 0.3  # Initial density center
        self.goal = 0.7  # Terminal cost goal

    def initial_density(self, x):
        """Gaussian blob centered at 0.3."""
        # x is 2D array: (N, 1) for 1D problem
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        dist_sq = (x[:, 0] - self.center) ** 2
        density = np.exp(-100 * dist_sq)

        # Normalize: ∫ m(x) dx = 1
        # For discrete grid: sum(m) * dx = 1
        dx = float(self.geometry.grid.spacing[0])
        return density / (np.sum(density) * dx + 1e-10)

    def terminal_cost(self, x):
        """Quadratic cost: distance to goal."""
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        dist_sq = (x[:, 0] - self.goal) ** 2
        return 5.0 * dist_sq

    def running_cost(self, x, t):
        """Small running cost."""
        return 0.1 * np.ones(x.shape[0])

    def hamiltonian(self, x, m, p, t):
        """H = (1/2)|p|² + 0.3·m (isotropic control + congestion)."""
        p_squared = p**2 if p.ndim == 1 else np.sum(p**2, axis=1)
        h = 0.5 * p_squared
        h += 0.3 * m
        return h

    def setup_components(self):
        """Setup MFG components for numerical solvers."""
        kappa = 0.3  # Congestion weight

        # Pre-compute normalization constant
        grid_points = self.geometry.grid.flatten()
        if grid_points.ndim == 1:
            grid_points = grid_points.reshape(-1, 1)

        dist_sq = (grid_points[:, 0] - self.center) ** 2
        unnormalized_density = np.exp(-100 * dist_sq)
        dx = float(self.geometry.grid.spacing[0])
        normalization_constant = np.sum(unnormalized_density) * dx

        def hamiltonian_func(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, derivs=None, **kwargs):
            """H = (1/2)|∇u|² + κ·m"""
            h_value = 0.0
            if derivs is not None:
                grad_u = derivs.get((1,), 0.0)  # 1D: derivative in x direction
                h_value += 0.5 * float(grad_u**2)
            h_value += kappa * float(m_at_x)
            return h_value

        def hamiltonian_dm(x_idx, x_position, m_at_x, **kwargs):
            """∂H/∂m = κ"""
            return kappa

        def initial_density_func(x_idx):
            """Gaussian initial density."""
            # Get x coordinate from index
            idx = x_idx[0] if isinstance(x_idx, tuple) else x_idx
            x_coord = self.geometry.grid.bounds[0][0] + idx * self.geometry.grid.spacing[0]

            # Evaluate unnormalized density
            dist_sq_point = (x_coord - self.center) ** 2
            unnormalized = np.exp(-100 * dist_sq_point)

            # Use pre-computed normalization constant
            return float(unnormalized / normalization_constant)

        def terminal_cost_func(x_idx):
            """Quadratic terminal cost."""
            idx = x_idx[0] if isinstance(x_idx, tuple) else x_idx
            x_coord = self.geometry.grid.bounds[0][0] + idx * self.geometry.grid.spacing[0]
            dist_sq = (x_coord - self.goal) ** 2
            return 5.0 * float(dist_sq)

        return MFGComponents(
            hamiltonian_func=hamiltonian_func,
            hamiltonian_dm_func=hamiltonian_dm,
            initial_density_func=initial_density_func,
            final_value_func=terminal_cost_func,
        )


def compute_total_mass_1d(M: np.ndarray, dx: float) -> float:
    """Compute total mass by integrating over 1D grid."""
    return float(np.sum(M) * dx)


def test_1d_mass_conservation():
    """Test mass conservation in 1D problem."""
    print("\n" + "=" * 70)
    print("  1D Mass Conservation Test")
    print("=" * 70 + "\n")

    # Create problem with finer resolution than 2D test
    grid_resolution = 32
    num_timesteps = 15

    print("Creating 1D problem:")
    print(f"  Grid resolution: {grid_resolution} points")
    print(f"  Time steps: {num_timesteps}")
    print("  Domain: [0, 1]")
    print("  Time horizon: T = 0.5\n")

    problem = Simple1DMFG(
        grid_resolution=grid_resolution,
        time_horizon=0.5,
        num_timesteps=num_timesteps,
        diffusion=0.05,
    )

    # Check grid spacing
    dx = problem.geometry.grid.spacing[0]
    print(f"Grid spacing: dx = {dx:.6f}")
    print(f"Volume element: dV = {dx:.6f}\n")

    # Verify initial density
    print("=" * 70)
    print("  Step 1: Verify initial_density() method")
    print("=" * 70 + "\n")

    points = problem.geometry.grid.flatten()
    initial_m = problem.initial_density(points)

    if initial_m.ndim > 1:
        initial_m = initial_m.flatten()

    initial_mass = compute_total_mass_1d(initial_m, dx)

    print("Initial density check:")
    print(f"  sum(m) = {np.sum(initial_m):.6f}")
    print(f"  ∫ m dx = {initial_mass:.6f}")
    print("  Expected: 1.0")
    print(f"  Error: {abs(initial_mass - 1.0) * 100:.4f}%")

    if abs(initial_mass - 1.0) < 0.01:
        print("  ✅ initial_density() method PASS\n")
    else:
        print("  ❌ initial_density() method FAIL\n")

    # Create solver
    print("=" * 70)
    print("  Step 2: Run solver and track mass evolution")
    print("=" * 70 + "\n")

    solver = create_basic_solver(
        problem,
        damping=0.6,
        max_iterations=30,
        tolerance=1e-4,
    )

    # Solve
    start_time = time.time()
    result = solver.solve()
    runtime = time.time() - start_time

    print("\nSolver completed:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Runtime: {runtime:.2f}s")
    print(f"  Final error: {result.max_error:.6e}\n")

    # Trace mass at each timestep
    print("=" * 70)
    print("  Step 3: Mass at each timestep")
    print("=" * 70 + "\n")

    masses = []
    mass_loss = []
    relative_loss = []

    for t_idx in range(result.M.shape[0]):
        M_t = result.M[t_idx]
        if M_t.ndim > 1:
            M_t = M_t.flatten()

        mass_t = compute_total_mass_1d(M_t, dx)
        masses.append(mass_t)

        loss = initial_mass - mass_t
        rel_loss = loss / initial_mass * 100

        mass_loss.append(loss)
        relative_loss.append(rel_loss)

        # Print detailed info
        status = "✅" if rel_loss < 2.0 else "⚠️" if rel_loss < 10.0 else "❌"
        dt = problem.T / (problem.Nt - 1)
        print(f"t={t_idx:2d} (t={t_idx * dt:.4f}s): mass = {mass_t:.6e}  loss = {rel_loss:6.2f}%  {status}")

    masses = np.array(masses)
    mass_loss = np.array(mass_loss)
    relative_loss = np.array(relative_loss)

    # Analysis
    print("\n" + "=" * 70)
    print("  ANALYSIS")
    print("=" * 70 + "\n")

    print(f"Initial mass (t=0):     {masses[0]:.6e}")
    print(f"Final mass (t={len(masses) - 1}):      {masses[-1]:.6e}")
    print(f"Total mass loss:        {mass_loss[-1]:.6e}")
    print(f"Relative mass loss:     {relative_loss[-1]:.2f}%")

    # Check if bug is same as 2D
    print("\nDiagnostic:")
    print(f"  Initial mass = {masses[0]:.6f}")
    print(f"  Expected dV = {dx:.6f}")
    print(f"  Ratio: {masses[0] / dx:.4f}")

    if abs(masses[0] - dx) < 0.001:
        print("  ⚠️  SAME BUG AS 2D: Initial mass ≈ dx (missing 1/dx factor)")
    elif abs(masses[0] - 1.0) < 0.01:
        print("  ✅ Initial mass correct - bug is elsewhere")
    else:
        print("  ⚠️  Different bug pattern than 2D")

    # Final verdict
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70 + "\n")

    if relative_loss[-1] < 2.0:
        print("✅ PASS: Mass conservation within expected FDM error (< 2%)")
    elif relative_loss[-1] < 10.0:
        print("⚠️  WARNING: Mass loss higher than expected (2-10%)")
    else:
        print("❌ FAIL: Catastrophic mass loss (> 10%)")
        print("   Same bug as 2D case")

    print(f"\nFinal mass conservation: {100 - relative_loss[-1]:.2f}%")
    print(f"Mass loss: {relative_loss[-1]:.2f}%\n")

    return {
        "masses": masses,
        "mass_loss": mass_loss,
        "relative_loss": relative_loss,
        "result": result,
        "problem": problem,
    }


if __name__ == "__main__":
    test_1d_mass_conservation()
