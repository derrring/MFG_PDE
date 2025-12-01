"""
Bug #8 Phase 1: Test 2D FP solver in isolation (without MFG coupling).

This test determines if mass loss occurs in the pure FP solver or in the MFG iteration coupling.
"""

import numpy as np

from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfg_pde.geometry.boundary import no_flux_bc


class Simple2DFPProblem(MFGProblem):
    """Minimal 2D problem for testing FP solver in isolation."""

    def __init__(self, N=8, T=0.4, Nt=10, sigma=0.1):
        super().__init__(
            spatial_bounds=[(0.0, 1.0), (0.0, 1.0)],
            spatial_discretization=[N, N],
            T=T,
            Nt=Nt,
            sigma=sigma,
        )

    def initial_density(self, x):
        """Gaussian blob centered at (0.5, 0.5)."""
        dist_sq = np.sum((x - 0.5) ** 2, axis=1)
        density = np.exp(-10.0 * dist_sq)
        # Normalize to unit mass
        dV = float(np.prod(self.geometry.grid.spacing))
        return density / (np.sum(density) * dV + 1e-10)

    def terminal_cost(self, x):
        return np.zeros(x.shape[0])

    def running_cost(self, x, t):
        return np.zeros(x.shape[0])

    def hamiltonian(self, x, m, p, t):
        """H = (1/2)|p|²"""
        p_squared = np.sum(p**2, axis=1) if p.ndim > 1 else p**2
        return 0.5 * p_squared

    def setup_components(self):
        """Stub implementation - not needed for FP isolation test."""
        from mfg_pde import MFGComponents

        return MFGComponents(
            hamiltonian_func=lambda *args, **kwargs: 0.0,
            hamiltonian_dm_func=lambda *args, **kwargs: 0.0,
            initial_density_func=lambda *args, **kwargs: 0.0,
            final_value_func=lambda *args, **kwargs: 0.0,
        )


def test_pure_diffusion():
    """Test 2D FP with pure diffusion (zero velocity field)."""
    print("=" * 70)
    print("  Bug #8 Phase 1: 2D FP Isolation Test - Pure Diffusion")
    print("=" * 70)
    print()

    # Create problem
    N = 8
    T = 0.4
    Nt = 10
    problem = Simple2DFPProblem(N=N, T=T, Nt=Nt, sigma=0.1)

    print(f"Problem: {N}×{N} grid, T={T}, {Nt} timesteps")
    print(f"Timestep: dt={T / Nt:.4f}")
    print()

    # Get initial density
    grid_points = problem.geometry.grid.flatten()
    shape = tuple(problem.geometry.grid.num_points)
    m0 = problem.initial_density(grid_points).reshape(shape)

    dx, dy = problem.geometry.grid.spacing
    dV = dx * dy
    print(f"Grid spacing: dx={dx:.4f}, dy={dy:.4f}")
    print(f"Initial mass: {np.sum(m0) * dV:.10f}")
    print()

    # Zero velocity field (U = constant everywhere -> drift v = -∇U = 0)
    U_zero = np.zeros((Nt + 1, *shape))

    print("Running 2D FP solver (pure diffusion, zero velocity)...")
    print()

    # Call the FP FDM solver directly
    boundary_conditions = no_flux_bc(dimension=2)
    fp_solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)
    M_solution = fp_solver.solve_fp_system(
        M_initial=m0,
        drift_field=U_zero,
        show_progress=True,
    )

    print()
    print("Mass Conservation Analysis:")
    print("-" * 70)
    print(f"{'Time':<10} {'Mass':<15} {'Loss (%)':<12} {'Loss per step (%)':<15}")
    print("-" * 70)

    mass_initial = np.sum(M_solution[0]) * dV
    prev_mass = mass_initial

    for k in range(Nt + 1):
        mass_k = np.sum(M_solution[k]) * dV
        loss_total = (mass_k - mass_initial) / mass_initial * 100
        if k > 0:
            loss_step = (mass_k - prev_mass) / prev_mass * 100
        else:
            loss_step = 0.0

        print(f"{k * T / Nt:<10.3f} {mass_k:<15.10f} {loss_total:+11.6f}  {loss_step:+14.6f}")
        prev_mass = mass_k

    print("-" * 70)

    mass_final = np.sum(M_solution[-1]) * dV
    total_loss_percent = abs(mass_final - mass_initial) / mass_initial * 100

    print()
    print(f"Initial mass:  {mass_initial:.10f}")
    print(f"Final mass:    {mass_final:.10f}")
    print(f"Total loss:    {total_loss_percent:.6f}%")
    print()

    if total_loss_percent < 1e-6:
        print("✓ PASS: Mass conserved to machine precision")
        result = "PASS"
    elif total_loss_percent < 0.1:
        print("⚠ PARTIAL: Small mass loss (< 0.1%)")
        result = "PARTIAL"
    else:
        print("✗ FAIL: Significant mass loss")
        print(f"  → Loss rate: ~{total_loss_percent / Nt:.6f}% per timestep")
        result = "FAIL"

    print()
    print("=" * 70)
    print()

    return M_solution, total_loss_percent, result


def test_with_constant_velocity():
    """Test 2D FP with a constant velocity field."""
    print("=" * 70)
    print("  Bug #8 Phase 1: 2D FP Isolation Test - With Advection")
    print("=" * 70)
    print()

    # Create problem
    N = 8
    T = 0.4
    Nt = 10
    problem = Simple2DFPProblem(N=N, T=T, Nt=Nt, sigma=0.1)

    print(f"Problem: {N}×{N} grid, T={T}, {Nt} timesteps")
    print(f"Timestep: dt={T / Nt:.4f}")
    print()

    # Get initial density
    grid_points = problem.geometry.grid.flatten()
    shape = tuple(problem.geometry.grid.num_points)
    m0 = problem.initial_density(grid_points).reshape(shape)

    dx, dy = problem.geometry.grid.spacing
    dV = dx * dy
    print(f"Grid spacing: dx={dx:.4f}, dy={dy:.4f}")
    print(f"Initial mass: {np.sum(m0) * dV:.10f}")
    print()

    # Linear potential: U(x, y) = x + y
    # This gives constant velocity: v = -∇U = (-1, -1)
    x = np.linspace(0, 1, shape[0])
    y = np.linspace(0, 1, shape[1])
    X, Y = np.meshgrid(x, y, indexing="ij")

    U_linear = np.zeros((Nt + 1, *shape))
    for k in range(Nt + 1):
        U_linear[k] = X + Y  # Linear potential

    print("Running 2D FP solver (with advection, constant velocity field)...")
    print()

    # Call the FP FDM solver directly
    boundary_conditions = no_flux_bc(dimension=2)
    fp_solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)
    M_solution = fp_solver.solve_fp_system(
        M_initial=m0,
        drift_field=U_linear,
        show_progress=True,
    )

    print()
    print("Mass Conservation Analysis:")
    print("-" * 70)
    print(f"{'Time':<10} {'Mass':<15} {'Loss (%)':<12} {'Loss per step (%)':<15}")
    print("-" * 70)

    mass_initial = np.sum(M_solution[0]) * dV
    prev_mass = mass_initial

    for k in range(Nt + 1):
        mass_k = np.sum(M_solution[k]) * dV
        loss_total = (mass_k - mass_initial) / mass_initial * 100
        if k > 0:
            loss_step = (mass_k - prev_mass) / prev_mass * 100
        else:
            loss_step = 0.0

        print(f"{k * T / Nt:<10.3f} {mass_k:<15.10f} {loss_total:+11.6f}  {loss_step:+14.6f}")
        prev_mass = mass_k

    print("-" * 70)

    mass_final = np.sum(M_solution[-1]) * dV
    total_loss_percent = abs(mass_final - mass_initial) / mass_initial * 100

    print()
    print(f"Initial mass:  {mass_initial:.10f}")
    print(f"Final mass:    {mass_final:.10f}")
    print(f"Total loss:    {total_loss_percent:.6f}%")
    print()

    if total_loss_percent < 1e-6:
        print("✓ PASS: Mass conserved to machine precision")
        result = "PASS"
    elif total_loss_percent < 0.1:
        print("⚠ PARTIAL: Small mass loss (< 0.1%)")
        result = "PARTIAL"
    else:
        print("✗ FAIL: Significant mass loss")
        print(f"  → Loss rate: ~{total_loss_percent / Nt:.6f}% per timestep")
        result = "FAIL"

    print()
    print("=" * 70)
    print()

    return M_solution, total_loss_percent, result


if __name__ == "__main__":
    # Test 1: Pure diffusion (should be exact)
    _, loss1, result1 = test_pure_diffusion()

    print("\n")

    # Test 2: With advection
    _, loss2, result2 = test_with_constant_velocity()

    # Summary
    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"Test 1 (pure diffusion):  {loss1:.6f}% mass loss - {result1}")
    print(f"Test 2 (with advection):  {loss2:.6f}% mass loss - {result2}")
    print()

    if loss1 < 1e-6 and loss2 < 1e-6:
        print("CONCLUSION: 2D FP solver conserves mass in isolation")
        print("  -> Mass loss in full MFG solver must be from coupling/normalization")
    elif loss1 < 1e-6 and loss2 > 0.1:
        print("CONCLUSION: 2D FP advection has mass loss issue")
        print("  -> Check boundary discretization with advection terms")
    else:
        print("CONCLUSION: 2D FP FDM solver loses mass")
        print("  -> Check numerical scheme and boundary handling")

    print()
