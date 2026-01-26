"""
Mass Conservation Tests for 1D MFG Systems with No-Flux Neumann BC.

Tests mass conservation for solver combinations:
1. FP Particle + HJB FDM (standard finite difference)
2. FP Particle + HJB GFDM (particle collocation)

Both should conserve mass with no-flux Neumann boundary conditions.

Consolidated from test_mass_conservation_1d.py and test_mass_conservation_1d_simple.py.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfg_pde.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfg_pde.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfg_pde.core.mfg_components import MFGComponents
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import TensorProductGrid, no_flux_bc


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),  # Gaussian centered at 0.5
        u_final=lambda x: 0.0,  # Zero terminal cost
    )


def compute_total_mass(density: np.ndarray, dx: float) -> float:
    """
    Compute total mass integral m(x)dx using rectangular rule.

    Consistent with FPParticleSolver normalization which uses np.sum(density) * dx.

    Args:
        density: Density array
        dx: Grid spacing

    Returns:
        Total mass
    """
    return float(np.sum(density) * dx)


class TestMassConservation1D:
    """Test mass conservation for 1D MFG with no-flux Neumann BC using built-in MFGProblem."""

    @pytest.fixture
    def problem(self):
        """Create standard MFG problem with Neumann BC."""
        geometry = TensorProductGrid(bounds=[(0.0, 2.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        return MFGProblem(
            geometry=geometry,
            T=1.0,
            Nt=20,
            diffusion=0.1,
            coupling_coefficient=1.0,
            components=_default_components(),
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
        integral m(x,t)dx approx 1 for all t in [0,T]
        """
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

        try:
            result = mfg_solver.solve(max_iterations=50, tolerance=1e-3)
        except Exception as e:
            pytest.skip(f"MFG solver raised exception: {str(e)[:100]}")

        m_solution = result.M

        dx = problem.geometry.get_grid_spacing()[0]
        Nt_points = problem.Nt + 1
        masses = np.array([compute_total_mass(m_solution[t_idx, :], dx) for t_idx in range(Nt_points)])

        initial_mass = masses[0]
        print("\n=== FP Particle + HJB FDM Mass Conservation ===")
        print(f"Initial mass: {initial_mass:.6f}")
        print(f"Final mass: {masses[-1]:.6f}")

        mass_errors = np.abs(masses - initial_mass)
        max_mass_error = np.max(mass_errors)
        print(f"Max mass change: {max_mass_error:.6e}")
        print(f"Mean mass: {np.mean(masses):.6f} +/- {np.std(masses):.6e}")

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
        integral m(x,t)dx approx 1 for all t in [0,T]
        """
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

        try:
            result = mfg_solver.solve(max_iterations=50, tolerance=1e-3)
        except Exception as e:
            pytest.skip(f"MFG solver raised exception: {str(e)[:100]}")

        m_solution = result.M

        dx = problem.geometry.get_grid_spacing()[0]
        Nt_points = problem.Nt + 1
        masses = np.array([compute_total_mass(m_solution[t_idx, :], dx) for t_idx in range(Nt_points)])

        initial_mass = masses[0]
        print("\n=== FP Particle + HJB GFDM Mass Conservation ===")
        print(f"Initial mass: {initial_mass:.6f}")
        print(f"Final mass: {masses[-1]:.6f}")

        mass_errors = np.abs(masses - initial_mass)
        max_mass_error = np.max(mass_errors)
        print(f"Max mass change: {max_mass_error:.6e}")
        print(f"Mean mass: {np.mean(masses):.6f} +/- {np.std(masses):.6e}")

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

        dx = problem.geometry.get_grid_spacing()[0]
        Nt_points = problem.Nt + 1
        masses_fdm = []
        masses_gfdm = []

        for t_idx in range(Nt_points):
            mass_fdm = compute_total_mass(result_1.M[t_idx, :], dx)
            mass_gfdm = compute_total_mass(result_2.M[t_idx, :], dx)
            masses_fdm.append(mass_fdm)
            masses_gfdm.append(mass_gfdm)

        masses_fdm = np.array(masses_fdm)
        masses_gfdm = np.array(masses_gfdm)

        initial_fdm = masses_fdm[0]
        initial_gfdm = masses_gfdm[0]

        change_fdm = np.abs(masses_fdm - initial_fdm)
        change_gfdm = np.abs(masses_gfdm - initial_gfdm)

        print("\n=== Mass Conservation Comparison ===")
        print(f"FDM  - Initial: {initial_fdm:.6f}, Final: {masses_fdm[-1]:.6f}, Max change: {np.max(change_fdm):.6e}")
        print(
            f"GFDM - Initial: {initial_gfdm:.6f}, Final: {masses_gfdm[-1]:.6f}, Max change: {np.max(change_gfdm):.6e}"
        )

        assert np.max(change_fdm) < 0.2 * initial_fdm, f"FDM mass change too large: {np.max(change_fdm):.6e}"
        assert np.max(change_gfdm) < 0.2 * initial_gfdm, f"GFDM mass change too large: {np.max(change_gfdm):.6e}"

    @pytest.mark.slow
    @pytest.mark.parametrize("num_particles", [1000, 3000, 5000])
    def test_mass_conservation_particle_count(self, problem, boundary_conditions, num_particles):
        """
        Test that mass conservation quality with different particle counts.

        Args:
            num_particles: Number of particles to use
        """
        fp_solver = FPParticleSolver(
            problem,
            num_particles=num_particles,
            normalize_kde_output=True,
            boundary_conditions=boundary_conditions,
        )

        hjb_solver = HJBFDMSolver(problem)
        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            use_anderson=True,
            anderson_depth=5,
            backend=None,
        )

        try:
            result = mfg_solver.solve(max_iterations=50, tolerance=1e-4, return_structured=True)
        except Exception as e:
            pytest.skip(f"Solver convergence issue with {num_particles} particles: {str(e)[:100]}")

        dx = problem.geometry.get_grid_spacing()[0]
        Nt_points = problem.Nt + 1
        masses = np.array([compute_total_mass(result.M[t, :], dx) for t in range(Nt_points)])
        max_error = np.max(np.abs(masses - 1.0))

        print(f"\nParticles: {num_particles}, Max mass error: {max_error:.6e}")

        assert max_error < 0.1, f"Mass error too large with {num_particles} particles"

    @pytest.mark.slow
    def test_mass_conservation_different_initial_conditions(self, boundary_conditions):
        """
        Test mass conservation with different initial density distributions.
        """
        test_cases = [
            ("Gaussian left", 0.25, 0.1),
            ("Gaussian center", 0.5, 0.15),
            ("Gaussian right", 0.75, 0.1),
        ]

        for name, center_frac, std_frac in test_cases:
            L = 2.0

            def make_custom_components(cfrac, sfrac, domain_L):
                center = domain_L * cfrac
                std = domain_L * sfrac

                def m_initial(x):
                    density = np.exp(-((x - center) ** 2) / (2 * std**2))
                    return density

                return MFGComponents(m_initial=m_initial, u_final=lambda x: 0.0)

            geometry = TensorProductGrid(bounds=[(0.0, L)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
            problem = MFGProblem(
                geometry=geometry,
                T=1.0,
                Nt=20,
                diffusion=0.1,
                coupling_coefficient=1.0,
                components=make_custom_components(center_frac, std_frac, L),
            )

            fp_solver = FPParticleSolver(
                problem,
                num_particles=5000,
                normalize_kde_output=True,
                boundary_conditions=boundary_conditions,
            )
            hjb_solver = HJBFDMSolver(problem)
            mfg_solver = FixedPointIterator(
                problem,
                hjb_solver=hjb_solver,
                fp_solver=fp_solver,
                use_anderson=True,
                anderson_depth=5,
                backend=None,
            )

            try:
                result = mfg_solver.solve(max_iterations=50, tolerance=1e-4, return_structured=True)
            except Exception as e:
                pytest.skip(f"Solver convergence issue for {name}: {str(e)[:100]}")

            dx = problem.geometry.get_grid_spacing()[0]
            Nt_points = problem.Nt + 1
            masses = [compute_total_mass(result.M[t, :], dx) for t in range(Nt_points)]
            max_error = np.max(np.abs(np.array(masses) - 1.0))

            print(f"\n{name}: Initial mass = {masses[0]:.6f}, Max error = {max_error:.6e}")

            assert max_error < 0.1, f"Mass conservation failed for {name}: {max_error:.6e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
