"""
Mathematical validation tests for mass conservation in MFG solvers.

This module tests the fundamental mathematical property that mass (probability)
should be conserved throughout the evolution of the MFG system.
"""

import pytest

import numpy as np

from mfg_pde import ExampleMFGProblem, create_accurate_solver, create_standard_solver


class TestMassConservation:
    """Test mass conservation properties of MFG solvers."""

    @pytest.mark.mathematical
    def test_mass_conservation_small_problem(self, small_problem):
        """Test mass conservation for a small problem."""
        solver = create_standard_solver(small_problem, "fixed_point")
        result = solver.solve()

        # Calculate initial mass
        initial_mass = np.sum(small_problem.m_init) * small_problem.Dx

        # Check mass conservation at each time step
        for t_idx in range(small_problem.Nt + 1):
            current_mass = np.sum(result.M[t_idx, :]) * small_problem.Dx
            mass_error = abs(current_mass - initial_mass)

            assert mass_error < 1e-2, (
                f"Mass not conserved at t_idx={t_idx}: "
                f"initial={initial_mass:.6f}, current={current_mass:.6f}, "
                f"error={mass_error:.6e}"
            )

    @pytest.mark.mathematical
    @pytest.mark.parametrize("solver_type", ["fixed_point"])  # Add more solver types as available
    def test_mass_conservation_across_solvers(self, small_problem, solver_type):
        """Test mass conservation across different solver types."""
        solver = create_standard_solver(small_problem, solver_type)
        result = solver.solve()

        initial_mass = np.sum(small_problem.m_init) * small_problem.Dx
        final_mass = np.sum(result.M[-1, :]) * small_problem.Dx

        mass_error = abs(final_mass - initial_mass)
        assert mass_error < 1e-2, (
            f"Mass not conserved for {solver_type}: "
            f"initial={initial_mass:.6f}, final={final_mass:.6f}, "
            f"error={mass_error:.6e}"
        )

    @pytest.mark.mathematical
    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1.0, 2.0])
    def test_mass_conservation_diffusion_coefficients(self, sigma):
        """Test mass conservation for different diffusion coefficients."""
        problem = ExampleMFGProblem(Nx=20, Nt=8, T=0.5, sigma=sigma)
        solver = create_standard_solver(problem, "fixed_point")

        result = solver.solve()

        initial_mass = np.sum(problem.m_init) * problem.Dx
        final_mass = np.sum(result.M[-1, :]) * problem.Dx

        mass_error = abs(final_mass - initial_mass)
        assert mass_error < 1e-2, (
            f"Mass not conserved for sigma={sigma}: "
            f"initial={initial_mass:.6f}, final={final_mass:.6f}, "
            f"error={mass_error:.6e}"
        )

    @pytest.mark.mathematical
    def test_non_negativity_property(self, small_problem):
        """Test that density remains non-negative throughout evolution."""
        solver = create_standard_solver(small_problem, "fixed_point")
        result = solver.solve()

        # Density should be non-negative everywhere (allowing small numerical errors)
        min_density = np.min(result.M)
        assert min_density >= -1e-10, f"Negative density found: min={min_density:.6e}"

    @pytest.mark.mathematical
    def test_mass_conservation_accurate_solver(self, small_problem):
        """Test mass conservation with more accurate solver settings."""
        solver = create_accurate_solver(small_problem, "fixed_point")
        result = solver.solve()

        initial_mass = np.sum(small_problem.m_init) * small_problem.Dx
        final_mass = np.sum(result.M[-1, :]) * small_problem.Dx

        # Should have better conservation with accurate solver
        mass_error = abs(final_mass - initial_mass)
        assert mass_error < 5e-3, (  # Tighter tolerance for accurate solver
            f"Mass not well conserved with accurate solver: "
            f"initial={initial_mass:.6f}, final={final_mass:.6f}, "
            f"error={mass_error:.6e}"
        )

    @pytest.mark.mathematical
    @pytest.mark.parametrize(
        "problem_size",
        [
            {"Nx": 15, "Nt": 6},
            {"Nx": 25, "Nt": 10},
            {"Nx": 35, "Nt": 14},
        ],
    )
    def test_mass_conservation_problem_scaling(self, problem_size):
        """Test mass conservation across different problem sizes."""
        problem = ExampleMFGProblem(**problem_size, T=0.5)
        solver = create_standard_solver(problem, "fixed_point")

        result = solver.solve()

        initial_mass = np.sum(problem.m_init) * problem.Dx
        final_mass = np.sum(result.M[-1, :]) * problem.Dx

        mass_error = abs(final_mass - initial_mass)
        relative_error = mass_error / initial_mass if initial_mass > 0 else mass_error

        assert relative_error < 0.02, (  # 2% relative error
            f"Mass conservation poor for problem size {problem_size}: relative_error={relative_error:.4f}"
        )

    @pytest.mark.mathematical
    def test_mass_conservation_time_evolution(self, small_problem):
        """Test that mass conservation holds throughout time evolution."""
        solver = create_standard_solver(small_problem, "fixed_point")
        result = solver.solve()

        initial_mass = np.sum(small_problem.m_init) * small_problem.Dx
        mass_errors = []

        # Check mass at each time step
        for t_idx in range(small_problem.Nt + 1):
            current_mass = np.sum(result.M[t_idx, :]) * small_problem.Dx
            mass_error = abs(current_mass - initial_mass)
            mass_errors.append(mass_error)

            # Individual time step check
            assert mass_error < 1e-2, f"Mass not conserved at time step {t_idx}: error={mass_error:.6e}"

        # Check that mass errors don't grow unboundedly
        max_error = max(mass_errors)
        assert max_error < 1e-2, f"Maximum mass error too large: {max_error:.6e}"

        # Check that errors don't consistently increase (indicating instability)
        if len(mass_errors) > 3:
            # Simple trend check: last 3 errors shouldn't all be increasing
            last_three = mass_errors[-3:]
            is_consistently_increasing = last_three[1] > last_three[0] and last_three[2] > last_three[1]
            assert (
                not is_consistently_increasing
            ), "Mass conservation errors are consistently increasing, indicating potential numerical instability"


class TestPhysicalProperties:
    """Test additional physical properties of MFG solutions."""

    @pytest.mark.mathematical
    def test_probability_density_normalization(self, small_problem):
        """Test that the density integrates to 1 at all times."""
        solver = create_standard_solver(small_problem, "fixed_point")
        result = solver.solve()

        for t_idx in range(small_problem.Nt + 1):
            total_probability = np.sum(result.M[t_idx, :]) * small_problem.Dx

            # Should integrate to approximately 1
            assert (
                abs(total_probability - 1.0) < 0.02
            ), f"Density not normalized at t_idx={t_idx}: integral={total_probability:.6f}"

    @pytest.mark.mathematical
    def test_energy_bounds(self, small_problem):
        """Test that the value function has reasonable bounds."""
        solver = create_standard_solver(small_problem, "fixed_point")
        result = solver.solve()

        # Value function should be finite
        assert np.all(np.isfinite(result.U)), "Value function contains non-finite values"

        # Should have reasonable magnitude (problem-dependent, but shouldn't be extreme)
        max_abs_value = np.max(np.abs(result.U))
        assert max_abs_value < 1000, f"Value function has extreme values: max_abs={max_abs_value:.2f}"

    @pytest.mark.mathematical
    def test_initial_condition_preservation(self, small_problem):
        """Test that initial conditions are properly preserved."""
        solver = create_standard_solver(small_problem, "fixed_point")
        result = solver.solve()

        # Initial density should match the problem's initial condition
        computed_initial = result.M[0, :]
        expected_initial = small_problem.m_init

        # Allow for small numerical differences
        max_diff = np.max(np.abs(computed_initial - expected_initial))
        assert max_diff < 1e-10, f"Initial condition not preserved: max_diff={max_diff:.6e}"

    @pytest.mark.mathematical
    def test_boundary_condition_consistency(self, small_problem):
        """Test that boundary conditions are consistently applied."""
        solver = create_standard_solver(small_problem, "fixed_point")
        result = solver.solve()

        # For this test, we assume periodic or zero-flux boundary conditions
        # Check that densities at boundaries are reasonable
        left_boundary = result.M[:, 0]
        right_boundary = result.M[:, -1]

        # Boundary values should be non-negative and finite
        assert np.all(left_boundary >= -1e-10), "Left boundary has negative values"
        assert np.all(right_boundary >= -1e-10), "Right boundary has negative values"
        assert np.all(np.isfinite(left_boundary)), "Left boundary has non-finite values"
        assert np.all(np.isfinite(right_boundary)), "Right boundary has non-finite values"
