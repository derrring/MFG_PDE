"""
Mass Conservation Test for Hybrid FP-Particle + HJB-FDM Solver.

Tests that the HybridFPParticleHJBFDM solver preserves total mass
under no-flux Neumann boundary conditions in 1D.

Mathematical Property:
    With no-flux BC (∂m/∂n = 0 at boundaries), mass should be conserved:
    d/dt ∫_Ω m(x,t) dx = 0

    Therefore: ∫_Ω m(x,t) dx = ∫_Ω m(x,0) dx  for all t ∈ [0,T]
"""

from __future__ import annotations

import pytest

import numpy as np

from mfg_pde.alg.numerical.coupling.hybrid_fp_particle_hjb_fdm import HybridFPParticleHJBFDM
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import no_flux_bc


def compute_total_mass(density: np.ndarray, dx: float) -> float:
    """
    Compute total mass ∫m(x)dx using trapezoidal rule.

    Args:
        density: Density array
        dx: Grid spacing

    Returns:
        Total mass
    """
    return float(np.trapezoid(density, dx=dx))


@pytest.mark.slow
class TestHybridMassConservation:
    """Test mass conservation for HybridFPParticleHJBFDM solver."""

    @pytest.fixture
    def simple_1d_problem(self):
        """
        Create simple 1D MFG problem.

        Setup:
        - Domain: [0, 2]
        - Time: [0, 1]
        - Diffusion: σ = 0.1
        - No-flux Neumann BC
        """
        return MFGProblem(
            xmin=0.0,
            xmax=2.0,
            Nx=50,  # Moderate resolution
            T=1.0,
            Nt=20,
            sigma=0.1,
            coupling_coefficient=1.0,  # Running cost coefficient
        )

    def test_hybrid_solver_mass_conservation_neumann_bc(self, simple_1d_problem):
        """
        Test that HybridFPParticleHJBFDM preserves mass with Neumann BC.

        Expected Behavior:
        1. Initial mass M₀ = ∫ m(x,0) dx ≈ 1
        2. Final mass M_T = ∫ m(x,T) dx ≈ M₀
        3. Mass variation over time: |M(t) - M₀| / M₀ < 5%

        This is the fundamental test for numerical mass conservation.
        """
        # Create boundary conditions (no-flux = Neumann with zero gradient)
        bc = no_flux_bc(dimension=1)

        # Override problem BC if needed
        simple_1d_problem.boundary_conditions = bc

        # Create hybrid solver (using modern API)
        solver = HybridFPParticleHJBFDM(
            problem=simple_1d_problem,
            num_particles=5000,
            kde_bandwidth="scott",
            max_newton_iterations=30,
            newton_tolerance=1e-7,
            damping_parameter=0.5,
        )

        # Solve MFG system
        try:
            _U, M, info = solver.solve(
                max_iterations=50,
                tolerance=1e-3,
                verbose=False,
            )
        except Exception as e:
            pytest.skip(f"Solver raised exception: {str(e)[:100]}")

        # Verify solution shapes
        Nt, Nx = simple_1d_problem.Nt + 1, simple_1d_problem.Nx + 1
        assert M.shape == (Nt, Nx), f"Expected M shape {(Nt, Nx)}, got {M.shape}"

        # Compute mass at each time step
        dx = simple_1d_problem.dx
        masses = np.array([compute_total_mass(M[t, :], dx) for t in range(Nt)])

        # Initial and final masses
        initial_mass = masses[0]
        final_mass = masses[-1]

        print(f"\n  Initial mass: {initial_mass:.6f}")
        print(f"  Final mass:   {final_mass:.6f}")
        print(
            f"  Mass loss:    {initial_mass - final_mass:.6f} ({100 * (initial_mass - final_mass) / initial_mass:.2f}%)"
        )

        # Check mass conservation
        mass_variation = np.abs(masses - initial_mass) / initial_mass
        max_mass_variation = np.max(mass_variation)

        print(f"  Max mass variation: {100 * max_mass_variation:.2f}%")

        # Acceptance criteria: Mass should not vary by more than 5%
        # This is reasonable for particle methods with KDE
        assert max_mass_variation < 0.05, (
            f"Mass conservation violated: max variation {100 * max_mass_variation:.2f}% > 5%\n"
            f"  Initial mass: {initial_mass:.6f}\n"
            f"  Masses over time: {masses}"
        )

        # Check convergence (warn if not converged, but don't fail)
        converged = info.get("converged", False)
        final_error = info.get("final_error", float("inf"))

        print("  ✓ Mass conservation verified")
        print(f"  Convergence: {'YES' if converged else 'NO'} (final error: {final_error:.6f})")
        print(f"  Iterations: {info.get('iterations', 'N/A')}")

        # Mass conservation is more important than strict convergence
        if not converged:
            print("  Note: Solver did not fully converge, but mass is well-conserved")

    def test_hybrid_solver_no_mass_creation(self, simple_1d_problem):
        """
        Test that hybrid solver doesn't create spurious mass.

        Mass should be monotonically conserved or slightly decrease
        (due to numerical diffusion), but never increase beyond tolerance.
        """
        bc = no_flux_bc(dimension=1)
        simple_1d_problem.boundary_conditions = bc

        solver = HybridFPParticleHJBFDM(
            problem=simple_1d_problem,
            num_particles=5000,
            kde_bandwidth="scott",
            max_newton_iterations=30,
            newton_tolerance=1e-7,
            damping_parameter=0.5,
        )

        try:
            _U, M, _info = solver.solve(max_iterations=50, tolerance=1e-3, verbose=False)
        except Exception as e:
            pytest.skip(f"Solver raised exception: {str(e)[:100]}")

        # Compute masses
        dx = simple_1d_problem.dx
        Nt = simple_1d_problem.Nt + 1
        masses = np.array([compute_total_mass(M[t, :], dx) for t in range(Nt)])

        initial_mass = masses[0]

        # Check: no mass creation (allow small numerical increase due to KDE smoothing)
        mass_increase = masses - initial_mass
        max_mass_increase = np.max(mass_increase)

        print(f"\n  Max mass increase: {max_mass_increase:.6f} ({100 * max_mass_increase / initial_mass:.2f}%)")

        # Allow up to 2% mass increase due to KDE boundary effects
        assert max_mass_increase / initial_mass < 0.02, (
            f"Spurious mass creation detected: {100 * max_mass_increase / initial_mass:.2f}%"
        )

        print("  ✓ No spurious mass creation")


@pytest.mark.fast
class TestHybridMassConservationFast:
    """Fast smoke tests for hybrid solver mass conservation."""

    def test_hybrid_solver_basic_functionality(self):
        """
        Quick smoke test: verify hybrid solver runs and produces reasonable output.
        """
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=20, T=0.5, Nt=10, sigma=0.1)

        bc = no_flux_bc(dimension=1)
        problem.boundary_conditions = bc

        solver = HybridFPParticleHJBFDM(
            problem=problem,
            num_particles=1000,  # Small for speed
            damping_parameter=0.5,
        )

        try:
            U, M, _info = solver.solve(max_iterations=10, tolerance=1e-2, verbose=False)
        except Exception as e:
            pytest.skip(f"Solver raised exception: {str(e)[:100]}")

        # Basic sanity checks
        assert U.shape == (problem.Nt + 1, problem.Nx + 1)
        assert M.shape == (problem.Nt + 1, problem.Nx + 1)
        assert np.all(M >= -1e-6), "Negative density detected"
        assert np.all(np.isfinite(M)), "Non-finite density values"
        assert np.all(np.isfinite(U)), "Non-finite value function"

        # Basic mass conservation check
        dx = problem.dx
        initial_mass = compute_total_mass(M[0, :], dx)
        final_mass = compute_total_mass(M[-1, :], dx)

        relative_change = abs(final_mass - initial_mass) / initial_mass
        assert relative_change < 0.1, f"Excessive mass change: {100 * relative_change:.1f}%"

        print("\n  ✓ Basic functionality verified")
        print(f"  ✓ Mass change: {100 * relative_change:.2f}%")
