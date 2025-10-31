"""
Trace mass conservation through individual sweeps in dimensional splitting.

This diagnostic tool instruments the dimensional splitting to see:
1. How much mass is lost in each forward/backward sweep
2. Whether mass loss is uniform or concentrated in specific dimensions
3. Where in the algorithm the mass leakage occurs
"""

import numpy as np

from mfg_pde import MFGComponents
from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem
from mfg_pde.geometry import BoundaryConditions


class Simple2DProblem(GridBasedMFGProblem):
    """Minimal 2D problem for tracing mass through sweeps."""

    def __init__(self, N=8, T=0.4, Nt=10, sigma=0.1):
        super().__init__(
            domain_bounds=(0.0, 1.0, 0.0, 1.0),
            grid_resolution=N,
            time_domain=(T, Nt),
            diffusion_coeff=sigma,
        )

    def initial_density(self, x):
        """Gaussian blob centered at (0.5, 0.5)."""
        dist_sq = np.sum((x - 0.5) ** 2, axis=1)
        density = np.exp(-10.0 * dist_sq)
        dV = float(np.prod(self.geometry.grid.spacing))
        return density / (np.sum(density) * dV + 1e-10)

    def terminal_cost(self, x):
        return np.zeros(x.shape[0])

    def running_cost(self, x, t):
        return np.zeros(x.shape[0])

    def hamiltonian(self, x, m, p, t):
        p_squared = np.sum(p**2, axis=1) if p.ndim > 1 else p**2
        return 0.5 * p_squared

    def setup_components(self):
        return MFGComponents(
            hamiltonian_func=lambda *args, **kwargs: 0.0,
            hamiltonian_dm_func=lambda *args, **kwargs: 0.0,
            initial_density_func=lambda *args, **kwargs: 0.0,
            final_value_func=lambda *args, **kwargs: 0.0,
        )


# Monkey-patch _sweep_dimension to add mass tracking
from mfg_pde.alg.numerical.fp_solvers import fp_fdm_multid

original_sweep = fp_fdm_multid._sweep_dimension

# Global storage for mass tracking
mass_trace = []


def instrumented_sweep(M_in, U_current, problem, dt, sweep_dim, boundary_conditions, backend):
    """Instrumented version that tracks mass before and after."""
    dV = float(np.prod(problem.geometry.grid.spacing))
    mass_before = np.sum(M_in) * dV

    # Call original sweep
    M_out = original_sweep(M_in, U_current, problem, dt, sweep_dim, boundary_conditions, backend)

    mass_after = np.sum(M_out) * dV
    mass_change = mass_after - mass_before
    mass_change_percent = (mass_change / mass_before) * 100

    mass_trace.append(
        {
            "dim": sweep_dim,
            "mass_before": mass_before,
            "mass_after": mass_after,
            "change_abs": mass_change,
            "change_percent": mass_change_percent,
        }
    )

    return M_out


# Apply patch
fp_fdm_multid._sweep_dimension = instrumented_sweep


def test_pure_diffusion():
    """Trace mass through sweeps with pure diffusion."""
    global mass_trace
    mass_trace = []

    print("=" * 70)
    print("  Trace: Pure Diffusion (Zero Velocity)")
    print("=" * 70)
    print()

    N = 8
    T = 0.4
    Nt = 10
    problem = Simple2DProblem(N=N, T=T, Nt=Nt, sigma=0.1)

    grid_points = problem.geometry.grid.flatten()
    shape = tuple(problem.geometry.grid.num_points)
    m0 = problem.initial_density(grid_points).reshape(shape)

    # Zero velocity field
    U_zero = np.zeros((Nt + 1, *shape))

    print(f"Problem: {N}×{N} grid, T={T}, {Nt} timesteps")
    print(f"Timestep: dt={T / Nt:.4f}")
    print()

    from mfg_pde.alg.numerical.fp_solvers.fp_fdm_multid import solve_fp_nd_dimensional_splitting

    boundary_conditions = BoundaryConditions(type="no_flux")
    M_solution = solve_fp_nd_dimensional_splitting(
        m_initial_condition=m0,
        U_solution_for_drift=U_zero,
        problem=problem,
        boundary_conditions=boundary_conditions,
        show_progress=False,
        enforce_mass_conservation=False,
    )

    print()
    print("Mass Changes Per Sweep:")
    print("-" * 70)
    print(f"{'Timestep':<10} {'Sweep':<10} {'Dimension':<10} {'Mass Change (%)':<15}")
    print("-" * 70)

    # Each timestep has 4 sweeps (forward: dim 0, 1; backward: dim 1, 0)
    ndim = 2
    sweeps_per_timestep = 2 * ndim

    for k in range(Nt):
        for sweep_idx in range(sweeps_per_timestep):
            idx = k * sweeps_per_timestep + sweep_idx
            if idx < len(mass_trace):
                trace = mass_trace[idx]
                sweep_type = "Forward" if sweep_idx < ndim else "Backward"
                print(f"{k:<10} {sweep_type:<10} {trace['dim']:<10} {trace['change_percent']:+14.6f}")

    print("-" * 70)

    dV = float(np.prod(problem.geometry.grid.spacing))
    mass_initial = np.sum(M_solution[0]) * dV
    mass_final = np.sum(M_solution[-1]) * dV
    total_loss_percent = (mass_final - mass_initial) / mass_initial * 100

    print()
    print(f"Initial mass:  {mass_initial:.10f}")
    print(f"Final mass:    {mass_final:.10f}")
    print(f"Total change:  {total_loss_percent:+.6f}%")
    print()

    return M_solution


def test_with_advection():
    """Trace mass through sweeps with advection."""
    global mass_trace
    mass_trace = []

    print("=" * 70)
    print("  Trace: With Advection (Constant Velocity)")
    print("=" * 70)
    print()

    N = 8
    T = 0.4
    Nt = 10
    problem = Simple2DProblem(N=N, T=T, Nt=Nt, sigma=0.1)

    grid_points = problem.geometry.grid.flatten()
    shape = tuple(problem.geometry.grid.num_points)
    m0 = problem.initial_density(grid_points).reshape(shape)

    # Linear potential: U(x, y) = x + y -> velocity v = (-1, -1)
    x = np.linspace(0, 1, shape[0])
    y = np.linspace(0, 1, shape[1])
    X, Y = np.meshgrid(x, y, indexing="ij")

    U_linear = np.zeros((Nt + 1, *shape))
    for k in range(Nt + 1):
        U_linear[k] = X + Y

    print(f"Problem: {N}×{N} grid, T={T}, {Nt} timesteps")
    print(f"Timestep: dt={T / Nt:.4f}")
    print("Velocity: v = -∇U = (-1, -1) everywhere")
    print()

    from mfg_pde.alg.numerical.fp_solvers.fp_fdm_multid import solve_fp_nd_dimensional_splitting

    boundary_conditions = BoundaryConditions(type="no_flux")
    M_solution = solve_fp_nd_dimensional_splitting(
        m_initial_condition=m0,
        U_solution_for_drift=U_linear,
        problem=problem,
        boundary_conditions=boundary_conditions,
        show_progress=False,
        enforce_mass_conservation=False,
    )

    print()
    print("Mass Changes Per Sweep:")
    print("-" * 70)
    print(f"{'Timestep':<10} {'Sweep':<10} {'Dimension':<10} {'Mass Change (%)':<15}")
    print("-" * 70)

    # Each timestep has 4 sweeps (forward: dim 0, 1; backward: dim 1, 0)
    ndim = 2
    sweeps_per_timestep = 2 * ndim

    for k in range(Nt):
        for sweep_idx in range(sweeps_per_timestep):
            idx = k * sweeps_per_timestep + sweep_idx
            if idx < len(mass_trace):
                trace = mass_trace[idx]
                sweep_type = "Forward" if sweep_idx < ndim else "Backward"
                print(f"{k:<10} {sweep_type:<10} {trace['dim']:<10} {trace['change_percent']:+14.6f}")

    print("-" * 70)

    dV = float(np.prod(problem.geometry.grid.spacing))
    mass_initial = np.sum(M_solution[0]) * dV
    mass_final = np.sum(M_solution[-1]) * dV
    total_loss_percent = (mass_final - mass_initial) / mass_initial * 100

    print()
    print(f"Initial mass:  {mass_initial:.10f}")
    print(f"Final mass:    {mass_final:.10f}")
    print(f"Total change:  {total_loss_percent:+.6f}%")
    print()

    return M_solution


if __name__ == "__main__":
    # Test 1: Pure diffusion
    print("\n")
    test_pure_diffusion()

    print("\n")

    # Test 2: With advection
    test_with_advection()

    print()
    print("=" * 70)
    print("  Analysis")
    print("=" * 70)
    print()
    print("This trace shows which sweeps are losing mass.")
    print("Look for patterns:")
    print("  - Is loss uniform across dimensions?")
    print("  - Does it worsen over time?")
    print("  - Is forward sweep worse than backward?")
    print()
