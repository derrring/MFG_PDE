"""
Test convergence with lower damping (0.4) as requested.

This test uses:
- Finer grid: 16×16 (up from 8×8)
- Lower damping: 0.4 (down from 0.6)
- Longer time horizon for better observation
"""

from __future__ import annotations

import time

import numpy as np

from mfg_pde import MFGComponents
from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver
from mfg_pde.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfg_pde.alg.numerical.mfg_solvers import FixedPointIterator
from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem


class CrowdMotion2D(GridBasedMFGProblem):
    """Simple 2D crowd motion for testing convergence with lower damping."""

    def __init__(
        self,
        grid_resolution=16,
        time_horizon=0.4,
        num_timesteps=20,
        diffusion=0.05,
    ):
        super().__init__(
            domain_bounds=(0.0, 1.0, 0.0, 1.0),
            grid_resolution=grid_resolution,
            time_domain=(time_horizon, num_timesteps),
            diffusion_coeff=diffusion,
        )
        self.center = (0.3, 0.3)  # Initial density center
        self.goal = (0.7, 0.7)  # Terminal cost goal

    def initial_density(self, x):
        """Gaussian blob centered at (0.3, 0.3)."""
        if x.ndim == 1:
            x = x.reshape(-1, 2)

        dist_sq = (x[:, 0] - self.center[0]) ** 2 + (x[:, 1] - self.center[1]) ** 2
        density = np.exp(-100 * dist_sq)

        # Normalize: ∫ m(x) dx = 1
        dx = float(self.geometry.grid.spacing[0])
        dy = float(self.geometry.grid.spacing[1])
        dV = dx * dy
        return density / (np.sum(density) * dV + 1e-10)

    def terminal_cost(self, x):
        """Quadratic cost: distance to goal."""
        if x.ndim == 1:
            x = x.reshape(-1, 2)
        dist_sq = (x[:, 0] - self.goal[0]) ** 2 + (x[:, 1] - self.goal[1]) ** 2
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
            grid_points = grid_points.reshape(-1, 2)

        dist_sq = (grid_points[:, 0] - self.center[0]) ** 2 + (grid_points[:, 1] - self.center[1]) ** 2
        unnormalized_density = np.exp(-100 * dist_sq)
        dx = float(self.geometry.grid.spacing[0])
        dy = float(self.geometry.grid.spacing[1])
        dV = dx * dy
        normalization_constant = np.sum(unnormalized_density) * dV

        def hamiltonian_func(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, derivs=None, **kwargs):
            """H = (1/2)|∇u|² + κ·m"""
            h_value = 0.0
            if derivs is not None:
                grad_u_x = derivs.get((1, 0), 0.0)  # ∂u/∂x
                grad_u_y = derivs.get((0, 1), 0.0)  # ∂u/∂y
                h_value += 0.5 * float(grad_u_x**2 + grad_u_y**2)
            h_value += kappa * float(m_at_x)
            return h_value

        def hamiltonian_dm(x_idx, x_position, m_at_x, **kwargs):
            """∂H/∂m = κ"""
            return kappa

        def initial_density_func(x_idx):
            """Gaussian initial density."""
            idx_x, idx_y = x_idx
            x_coord = self.geometry.grid.bounds[0][0] + idx_x * self.geometry.grid.spacing[0]
            y_coord = self.geometry.grid.bounds[1][0] + idx_y * self.geometry.grid.spacing[1]

            dist_sq_point = (x_coord - self.center[0]) ** 2 + (y_coord - self.center[1]) ** 2
            unnormalized = np.exp(-100 * dist_sq_point)

            return float(unnormalized / normalization_constant)

        def terminal_cost_func(x_idx):
            """Quadratic terminal cost."""
            idx_x, idx_y = x_idx
            x_coord = self.geometry.grid.bounds[0][0] + idx_x * self.geometry.grid.spacing[0]
            y_coord = self.geometry.grid.bounds[1][0] + idx_y * self.geometry.grid.spacing[1]
            dist_sq = (x_coord - self.goal[0]) ** 2 + (y_coord - self.goal[1]) ** 2
            return 5.0 * float(dist_sq)

        return MFGComponents(
            hamiltonian_func=hamiltonian_func,
            hamiltonian_dm_func=hamiltonian_dm,
            initial_density_func=initial_density_func,
            final_value_func=terminal_cost_func,
        )


def compute_total_mass(M: np.ndarray, dV: float) -> float:
    """Compute total mass by integrating over grid."""
    return float(np.sum(M) * dV)


def test_lower_damping():
    """Test convergence with damping_factor=0.4."""
    print("\n" + "=" * 70)
    print("  Deep Debugging: Lower Damping Test (damping_factor=0.4)")
    print("=" * 70 + "\n")

    # Problem setup
    grid_resolution = 16
    num_timesteps = 20

    print("Problem configuration:")
    print(f"  Grid resolution: {grid_resolution}×{grid_resolution}")
    print(f"  Time steps: {num_timesteps}")
    print("  Domain: [0,1] × [0,1]")
    print("  Time horizon: T = 0.4")
    print("  Damping factor: 0.4 (LOWER than default 0.6)\n")

    problem = CrowdMotion2D(
        grid_resolution=grid_resolution,
        time_horizon=0.4,
        num_timesteps=num_timesteps,
        diffusion=0.05,
    )

    # Check grid spacing and CFL
    dx = problem.geometry.grid.spacing[0]
    dy = problem.geometry.grid.spacing[1]
    dt = problem.dt
    sigma = problem.sigma

    dV = dx * dy
    cfl_number = sigma**2 * dt / dx**2

    print("Grid parameters:")
    print(f"  dx = {dx:.6f}")
    print(f"  dy = {dy:.6f}")
    print(f"  dt = {dt:.6f}")
    print(f"  dV = dx·dy = {dV:.6e}")
    print(f"  σ²·dt/dx² = {cfl_number:.6f} (should be < 0.5)")

    if cfl_number >= 0.5:
        print("  ⚠️  CFL condition violated!\n")
    else:
        print("  ✓ CFL condition satisfied\n")

    # Verify initial density
    print("=" * 70)
    print("  Initial density check")
    print("=" * 70 + "\n")

    points = problem.geometry.grid.flatten()
    initial_m = problem.initial_density(points)

    if initial_m.ndim > 1:
        initial_m = initial_m.flatten()
    initial_m = initial_m.reshape(grid_resolution, grid_resolution)

    initial_mass = compute_total_mass(initial_m, dV)

    print("Initial density:")
    print(f"  sum(m) = {np.sum(initial_m):.6f}")
    print(f"  ∫ m dV = {initial_mass:.6f}")
    print("  Expected: 1.0")
    print(f"  Error: {abs(initial_mass - 1.0) * 100:.4f}%\n")

    # Create solver with LOWER DAMPING
    print("=" * 70)
    print("  Running solver with damping_factor=0.4")
    print("=" * 70 + "\n")

    # Use 2D-capable solvers
    hjb_solver = HJBSemiLagrangianSolver(problem)
    fp_solver = FPFDMSolver(problem)

    solver = FixedPointIterator(
        problem=problem,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
        damping=0.4,  # LOWER DAMPING as requested
        max_iterations=30,
        tolerance=1e-4,
    )

    # Solve
    start_time = time.time()
    result = solver.solve()
    runtime = time.time() - start_time

    print("\n" + "=" * 70)
    print("  Solver Results")
    print("=" * 70 + "\n")

    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Runtime: {runtime:.2f}s")
    print(f"Final U error: {result.error_history_U[-1]:.6e}")
    print(f"Final M error: {result.error_history_M[-1]:.6e}\n")

    # Check mass conservation
    print("=" * 70)
    print("  Mass Conservation Check")
    print("=" * 70 + "\n")

    masses = []
    for t_idx in range(result.M.shape[0]):
        M_t = result.M[t_idx]
        mass_t = compute_total_mass(M_t, dV)
        masses.append(mass_t)

        loss = initial_mass - mass_t
        rel_loss = loss / initial_mass * 100

        status = "✓" if rel_loss < 2.0 else "⚠️" if rel_loss < 10.0 else "✗"
        print(f"t={t_idx:2d}: mass = {mass_t:.6e}  loss = {rel_loss:6.2f}%  {status}")

    masses = np.array(masses)

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70 + "\n")

    print(f"Initial mass:    {masses[0]:.6e}")
    print(f"Final mass:      {masses[-1]:.6e}")
    print(f"Mass loss:       {(masses[0] - masses[-1]):.6e}")
    print(f"Relative loss:   {((masses[0] - masses[-1]) / masses[0] * 100):.2f}%\n")

    # Convergence analysis
    print("=" * 70)
    print("  Convergence Analysis")
    print("=" * 70 + "\n")

    if result.converged:
        print("✓ Solver converged successfully!")
    else:
        print("✗ Solver did NOT converge")

        # Check if errors are decreasing
        U_errors = result.metadata.get("l2distu_rel", result.error_history_U)
        M_errors = result.metadata.get("l2distm_rel", result.error_history_M)

        if len(U_errors) >= 3:
            u_trend = "decreasing" if U_errors[-1] < U_errors[-3] else "increasing/oscillating"
            m_trend = "decreasing" if M_errors[-1] < M_errors[-3] else "increasing/oscillating"

            print(f"  U error trend: {u_trend}")
            print(f"  M error trend: {m_trend}")

            print("\n  Last 5 iterations:")
            for i in range(max(0, len(U_errors) - 5), len(U_errors)):
                print(f"    Iter {i + 1}: U_err={U_errors[i]:.3e}, M_err={M_errors[i]:.3e}")

    return {
        "problem": problem,
        "result": result,
        "masses": masses,
        "initial_mass": initial_mass,
    }


if __name__ == "__main__":
    test_lower_damping()
