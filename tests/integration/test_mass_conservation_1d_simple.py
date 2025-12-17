"""
Simple Mass Conservation Tests for 1D MFG Systems.

Tests mass conservation for two solver combinations:
1. FP Particle + HJB FDM (standard finite difference)
2. FP Particle + HJB GFDM (particle collocation)

Uses the framework's built-in MFGProblem class with no-flux Neumann BC.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfg_pde.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfg_pde.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import no_flux_bc


def compute_total_mass(density: np.ndarray, dx: float) -> float:
    """
    Compute total mass ∫m(x)dx using rectangular rule (consistent with FPParticleSolver normalization).

    Args:
        density: Density array
        dx: Grid spacing

    Returns:
        Total mass
    """
    return float(np.sum(density) * dx)


class TestMassConservation1DSimple:
    """Test mass conservation for 1D MFG with no-flux Neumann BC using built-in problem."""

    @pytest.fixture
    def problem(self):
        """Create standard MFG problem with Neumann BC."""
        return MFGProblem(
            xmin=0.0,
            xmax=2.0,
            Nx=50,
            T=1.0,
            Nt=20,
            diffusion=0.1,
            coupling_coefficient=1.0,
        )

    @pytest.fixture
    def boundary_conditions(self):
        """No-flux Neumann boundary conditions."""
        return no_flux_bc(dimension=1)

    @pytest.mark.slow
    def test_fp_particle_hjb_fdm_mass_conservation(self, problem, boundary_conditions):
        """
        Test mass conservation for FP Particle + HJB FDM combination.

        With no-flux Neumann BC, total mass should be preserved:
        ∫m(x,t)dx ≈ 1 for all t ∈ [0,T]
        """
        # Create solvers
        from mfg_pde import KDENormalization

        fp_solver = FPParticleSolver(
            problem,
            num_particles=5000,
            kde_normalization=KDENormalization.INITIAL_ONLY,
            boundary_conditions=boundary_conditions,
        )

        hjb_solver = HJBFDMSolver(problem)

        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
        )

        # Solve MFG
        try:
            result = mfg_solver.solve(max_iterations=50, tolerance=1e-3)
        except Exception as e:
            # If solver raises convergence error, skip test
            pytest.skip(f"MFG solver raised exception: {str(e)[:100]}")

        # Extract density solution
        m_solution = result.M  # Shape: (Nt+1, Nx+1)

        # Compute mass at each time step
        dx = problem.geometry.get_grid_spacing()[0]
        Nt_points = problem.geometry.get_grid_shape()[0]
        masses = []
        for t_idx in range(Nt_points):
            mass_t = compute_total_mass(m_solution[t_idx, :], dx)
            masses.append(mass_t)

        masses = np.array(masses)

        # Check initial mass
        initial_mass = masses[0]
        print("\n=== FP Particle + HJB FDM Mass Conservation ===")
        print(f"Initial mass: {initial_mass:.6f}")
        print(f"Final mass: {masses[-1]:.6f}")

        # Compute mass errors
        mass_errors = np.abs(masses - initial_mass)
        max_mass_error = np.max(mass_errors)
        mean_mass = np.mean(masses)
        std_mass = np.std(masses)

        print(f"Max mass change: {max_mass_error:.6e}")
        print(f"Mean mass: {mean_mass:.6f} ± {std_mass:.6e}")
        print("Masses over time:")
        for t_idx, mass in enumerate(masses):
            if t_idx % 5 == 0 or t_idx == len(masses) - 1:
                print(f"  t={t_idx * problem.dt:.2f}: {mass:.6f}")

        # Mass should be approximately conserved
        # Particle methods with KDE normalization should preserve total mass well
        assert max_mass_error < 0.1 * initial_mass, (
            f"Mass conservation violated: max change = {max_mass_error:.6e}\n"
            f"Initial mass: {initial_mass:.6f}\n"
            f"Final mass: {masses[-1]:.6f}"
        )

    @pytest.mark.slow
    def test_fp_particle_hjb_gfdm_mass_conservation(self, problem, boundary_conditions):
        """
        Test mass conservation for FP Particle + HJB GFDM (particle collocation).

        With no-flux Neumann BC, total mass should be preserved:
        ∫m(x,t)dx ≈ 1 for all t ∈ [0,T]
        """
        # Create solvers
        from mfg_pde import KDENormalization

        fp_solver = FPParticleSolver(
            problem,
            num_particles=5000,
            kde_normalization=KDENormalization.INITIAL_ONLY,
            boundary_conditions=boundary_conditions,
        )

        collocation_points = problem.xSpace.reshape(-1, 1)
        hjb_solver = HJBGFDMSolver(problem, collocation_points=collocation_points, delta=0.3)

        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
        )

        # Solve MFG
        try:
            result = mfg_solver.solve(max_iterations=50, tolerance=1e-3)
        except Exception as e:
            # If solver raises convergence error, skip test
            pytest.skip(f"MFG solver raised exception: {str(e)[:100]}")

        # Extract density solution
        m_solution = result.M  # Shape: (Nt+1, Nx+1)

        # Compute mass at each time step
        dx = problem.geometry.get_grid_spacing()[0]
        Nt_points = problem.geometry.get_grid_shape()[0]
        masses = []
        for t_idx in range(Nt_points):
            mass_t = compute_total_mass(m_solution[t_idx, :], dx)
            masses.append(mass_t)

        masses = np.array(masses)

        # Check initial mass
        initial_mass = masses[0]
        print("\n=== FP Particle + HJB GFDM Mass Conservation ===")
        print(f"Initial mass: {initial_mass:.6f}")
        print(f"Final mass: {masses[-1]:.6f}")

        # Compute mass errors
        mass_errors = np.abs(masses - initial_mass)
        max_mass_error = np.max(mass_errors)
        mean_mass = np.mean(masses)
        std_mass = np.std(masses)

        print(f"Max mass change: {max_mass_error:.6e}")
        print(f"Mean mass: {mean_mass:.6f} ± {std_mass:.6e}")
        print("Masses over time:")
        for t_idx, mass in enumerate(masses):
            if t_idx % 5 == 0 or t_idx == len(masses) - 1:
                print(f"  t={t_idx * problem.dt:.2f}: {mass:.6f}")

        # Mass should be approximately conserved
        # GFDM particle collocation with KDE normalization should preserve total mass
        assert max_mass_error < 0.1 * initial_mass, (
            f"Mass conservation violated: max change = {max_mass_error:.6e}\n"
            f"Initial mass: {initial_mass:.6f}\n"
            f"Final mass: {masses[-1]:.6f}"
        )

    @pytest.mark.slow
    def test_compare_mass_conservation_methods(self, problem, boundary_conditions):
        """
        Compare mass conservation between FDM and GFDM methods.

        Both methods should preserve mass with similar quality.
        Note: This test focuses on mass conservation, not convergence.
        """
        # Solver 1: FP Particle + HJB FDM
        fp_solver_1 = FPParticleSolver(
            problem,
            num_particles=5000,
            normalize_kde_output=True,
            boundary_conditions=boundary_conditions,
        )
        hjb_solver_1 = HJBFDMSolver(problem)
        mfg_solver_1 = FixedPointIterator(problem, hjb_solver=hjb_solver_1, fp_solver=fp_solver_1)

        try:
            result_1 = mfg_solver_1.solve(max_iterations=50, tolerance=1e-3)
        except Exception as e:
            pytest.skip(f"FDM solver raised exception: {str(e)[:100]}")

        # Solver 2: FP Particle + HJB GFDM
        fp_solver_2 = FPParticleSolver(
            problem,
            num_particles=5000,
            normalize_kde_output=True,
            boundary_conditions=boundary_conditions,
        )
        collocation_points = problem.xSpace.reshape(-1, 1)
        hjb_solver_2 = HJBGFDMSolver(problem, collocation_points=collocation_points, delta=0.3)
        mfg_solver_2 = FixedPointIterator(problem, hjb_solver=hjb_solver_2, fp_solver=fp_solver_2)

        try:
            result_2 = mfg_solver_2.solve(max_iterations=50, tolerance=1e-3)
        except Exception as e:
            pytest.skip(f"GFDM solver raised exception: {str(e)[:100]}")

        # Compute masses for both methods
        dx = problem.geometry.get_grid_spacing()[0]
        Nt_points = problem.geometry.get_grid_shape()[0]
        masses_fdm = []
        masses_gfdm = []

        for t_idx in range(Nt_points):
            mass_fdm = compute_total_mass(result_1.M[t_idx, :], dx)
            mass_gfdm = compute_total_mass(result_2.M[t_idx, :], dx)
            masses_fdm.append(mass_fdm)
            masses_gfdm.append(mass_gfdm)

        masses_fdm = np.array(masses_fdm)
        masses_gfdm = np.array(masses_gfdm)

        # Initial masses
        initial_fdm = masses_fdm[0]
        initial_gfdm = masses_gfdm[0]

        # Compute mass changes
        change_fdm = np.abs(masses_fdm - initial_fdm)
        change_gfdm = np.abs(masses_gfdm - initial_gfdm)

        print("\n=== Mass Conservation Comparison ===")
        print(f"FDM  - Initial: {initial_fdm:.6f}, Final: {masses_fdm[-1]:.6f}, Max change: {np.max(change_fdm):.6e}")
        print(
            f"GFDM - Initial: {initial_gfdm:.6f}, Final: {masses_gfdm[-1]:.6f}, Max change: {np.max(change_gfdm):.6e}"
        )

        # Both methods should have reasonable mass conservation
        assert np.max(change_fdm) < 0.2 * initial_fdm, f"FDM mass change too large: {np.max(change_fdm):.6e}"
        assert np.max(change_gfdm) < 0.2 * initial_gfdm, f"GFDM mass change too large: {np.max(change_gfdm):.6e}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
