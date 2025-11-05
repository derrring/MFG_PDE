"""
Property-Based Tests for MFG_PDE Solvers

This module implements property-based testing using Hypothesis to validate
mathematical properties of Mean Field Game solutions across a wide range
of parameter combinations.
"""

import pytest

import numpy as np

from mfg_pde import ExampleMFGProblem, create_fast_solver
from mfg_pde.utils.exceptions import ConfigurationError, ConvergenceError

# Skip all tests if hypothesis is not installed
pytest.importorskip("hypothesis")

from hypothesis import assume, given, note, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

# === Test Strategies ===

# Valid parameter ranges for MFG problems
spatial_points = st.integers(min_value=10, max_value=50)
temporal_points = st.integers(min_value=5, max_value=30)
diffusion_coeff = st.floats(min_value=0.1, max_value=2.0)
coupling_coeff = st.floats(min_value=0.1, max_value=1.0)
final_time = st.floats(min_value=0.5, max_value=2.0)
tolerance = st.floats(min_value=1e-6, max_value=1e-3)

# Solver types available for testing
solver_types = st.sampled_from(["fixed_point", "particle_collocation"])


# === Mathematical Property Tests ===


class TestMFGMathematicalProperties:
    """Test fundamental mathematical properties of MFG solutions."""

    @given(
        Nx=spatial_points, Nt=temporal_points, sigma=diffusion_coeff, coupling_coefficient=coupling_coeff, T=final_time
    )
    @settings(max_examples=20, deadline=30000)  # 30 second timeout per test
    def test_mass_conservation_property(self, Nx, Nt, sigma, coupling_coefficient, T):
        """
        Property: Total mass should be conserved in MFG solutions.

        The integral of the density function m(t,x) over space should remain
        approximately constant throughout time evolution.
        """
        # Skip computationally expensive combinations
        assume(Nx * Nt < 1000)

        note(f"Testing mass conservation with Nx={Nx}, Nt={Nt}, σ={sigma:.3f}, λ={coupling_coefficient:.3f}")

        try:
            problem = ExampleMFGProblem(Nx=Nx, Nt=Nt, T=T, sigma=sigma, coupling_coefficient=coupling_coefficient)
            solver = create_fast_solver(problem, "fixed_point")

            # Solve with moderate tolerance to ensure convergence
            _U, M, info = solver.solve(max_iterations=50, tolerance=1e-4)

            # Check that we got a valid solution
            assume(info.get("converged", False))

            # Calculate total mass at each time step
            Dx = problem.Dx
            masses = [np.sum(M[t, :]) * Dx for t in range(Nt + 1)]

            initial_mass = masses[0]
            final_mass = masses[-1]
            max_mass_variation = max(masses) - min(masses)

            note(f"Initial mass: {initial_mass:.6f}")
            note(f"Final mass: {final_mass:.6f}")
            note(f"Max variation: {max_mass_variation:.6f}")

            # Property: Mass should be conserved within tolerance
            relative_mass_change = abs(final_mass - initial_mass) / initial_mass
            assert relative_mass_change < 0.05, f"Mass not conserved: {relative_mass_change:.6f} > 0.05"

            # Property: Mass should remain positive
            assert all(mass > 0 for mass in masses), "Mass became non-positive"

        except ConvergenceError:
            # Skip if solver doesn't converge for this parameter combination
            assume(False)

    @given(Nx=spatial_points, Nt=temporal_points, sigma=diffusion_coeff, T=final_time)
    @settings(max_examples=15, deadline=25000)
    def test_solution_boundedness_property(self, Nx, Nt, sigma, T):
        """
        Property: MFG solutions should remain bounded.

        The value function U and density function M should not exhibit
        unbounded growth or infinite values.
        """
        assume(Nx * Nt < 800)

        note(f"Testing boundedness with Nx={Nx}, Nt={Nt}, σ={sigma:.3f}")

        try:
            problem = ExampleMFGProblem(Nx=Nx, Nt=Nt, T=T, sigma=sigma)
            solver = create_fast_solver(problem, "fixed_point")

            U, M, info = solver.solve(max_iterations=30, tolerance=1e-4)
            assume(info.get("converged", False))

            # Property: Solutions should be finite
            assert np.all(np.isfinite(U)), "Value function contains infinite values"
            assert np.all(np.isfinite(M)), "Density function contains infinite values"

            # Property: Density should be non-negative
            assert np.all(M >= 0), "Density function has negative values"

            # Property: Solutions should be reasonably bounded
            U_range = np.max(U) - np.min(U)
            M_max = np.max(M)

            note(f"U range: {U_range:.3f}")
            note(f"M max: {M_max:.3f}")

            assert U_range < 1000, f"Value function range too large: {U_range}"
            assert M_max < 100, f"Density function too large: {M_max}"

        except ConvergenceError:
            assume(False)

    @given(Nx=spatial_points, sigma1=diffusion_coeff, sigma2=diffusion_coeff)
    @settings(max_examples=10, deadline=20000)
    def test_monotonicity_in_diffusion_property(self, Nx, sigma1, sigma2):
        """
        Property: Solutions should vary monotonically with diffusion coefficient.

        Higher diffusion should lead to smoother (more spread out) density functions.
        """
        assume(abs(sigma1 - sigma2) > 0.2)  # Ensure significant difference
        assume(Nx <= 30)  # Keep computational cost reasonable

        Nt = 10  # Fixed temporal resolution

        try:
            # Solve for both diffusion coefficients
            problem1 = ExampleMFGProblem(Nx=Nx, Nt=Nt, sigma=sigma1)
            problem2 = ExampleMFGProblem(Nx=Nx, Nt=Nt, sigma=sigma2)

            solver1 = create_fast_solver(problem1, "fixed_point")
            solver2 = create_fast_solver(problem2, "fixed_point")

            _U1, M1, info1 = solver1.solve(max_iterations=30, tolerance=1e-4)
            _U2, M2, info2 = solver2.solve(max_iterations=30, tolerance=1e-4)

            assume(info1.get("converged", False) and info2.get("converged", False))

            # Property: Higher diffusion should lead to smoother densities
            # Measure smoothness by total variation
            def total_variation(arr):
                return np.sum(np.abs(np.diff(arr, axis=1)))

            tv1 = total_variation(M1)
            tv2 = total_variation(M2)

            if sigma1 > sigma2:
                _smoother_M, _rougher_M = M1, M2
                smoother_sigma, rougher_sigma = sigma1, sigma2
                smoother_tv, rougher_tv = tv1, tv2
            else:
                _smoother_M, _rougher_M = M2, M1
                smoother_sigma, rougher_sigma = sigma2, sigma1
                smoother_tv, rougher_tv = tv2, tv1

            note(f"σ_smooth={smoother_sigma:.3f}, TV_smooth={smoother_tv:.3f}")
            note(f"σ_rough={rougher_sigma:.3f}, TV_rough={rougher_tv:.3f}")

            # Property: Higher diffusion → lower total variation (smoother)
            # Allow some tolerance for numerical effects
            relative_smoothing = (rougher_tv - smoother_tv) / rougher_tv
            assert relative_smoothing > -0.2, f"Diffusion monotonicity violated: {relative_smoothing:.3f}"

        except ConvergenceError:
            assume(False)


class TestMFGNumericalStability:
    """Test numerical stability properties of MFG solvers."""

    @given(
        Nx=st.integers(min_value=20, max_value=40),
        tolerance=st.floats(min_value=1e-6, max_value=1e-3),
        max_iterations=st.integers(min_value=20, max_value=100),
    )
    @settings(max_examples=15, deadline=25000)
    def test_convergence_stability_property(self, Nx, tolerance, max_iterations):
        """
        Property: Convergence should be stable and deterministic.

        Running the same problem multiple times should produce consistent results.
        """
        note(f"Testing convergence stability: Nx={Nx}, tol={tolerance:.1e}, max_iter={max_iterations}")

        problem = ExampleMFGProblem(Nx=Nx, Nt=15, T=1.0, sigma=0.5)

        try:
            # Run solver multiple times
            results = []
            for _ in range(3):
                solver = create_fast_solver(problem, "fixed_point")
                U, M, info = solver.solve(max_iterations=max_iterations, tolerance=tolerance)

                if info.get("converged", False):
                    results.append((U.copy(), M.copy(), info["iterations"]))

            assume(len(results) >= 2)  # Need at least 2 successful runs

            # Property: Results should be consistent
            U1, M1, iter1 = results[0]
            U2, M2, iter2 = results[1]

            U_diff = np.max(np.abs(U1 - U2))
            M_diff = np.max(np.abs(M1 - M2))

            note(f"Max U difference: {U_diff:.1e}")
            note(f"Max M difference: {M_diff:.1e}")
            note(f"Iteration counts: {iter1}, {iter2}")

            # Property: Solutions should be nearly identical
            assert U_diff < tolerance * 10, f"U solutions not reproducible: {U_diff:.1e}"
            assert M_diff < tolerance * 10, f"M solutions not reproducible: {M_diff:.1e}"

            # Property: Iteration counts should be similar
            iter_diff = abs(iter1 - iter2)
            assert iter_diff <= 3, f"Iteration counts too different: {iter1} vs {iter2}"

        except ConvergenceError:
            assume(False)

    @given(base_Nx=st.integers(min_value=15, max_value=25), refinement_factor=st.integers(min_value=2, max_value=3))
    @settings(max_examples=8, deadline=30000)
    def test_grid_refinement_convergence_property(self, base_Nx, refinement_factor):
        """
        Property: Solutions should converge as grid is refined.

        Finer grids should produce more accurate solutions that converge
        to the true solution.
        """
        coarse_Nx = base_Nx
        fine_Nx = base_Nx * refinement_factor

        note(f"Testing grid convergence: {coarse_Nx} → {fine_Nx}")

        try:
            # Solve on coarse grid
            problem_coarse = ExampleMFGProblem(Nx=coarse_Nx, Nt=10, T=1.0, sigma=0.5)
            solver_coarse = create_fast_solver(problem_coarse, "fixed_point")
            U_coarse, M_coarse, info_coarse = solver_coarse.solve(max_iterations=50, tolerance=1e-5)

            # Solve on fine grid
            problem_fine = ExampleMFGProblem(Nx=fine_Nx, Nt=10, T=1.0, sigma=0.5)
            solver_fine = create_fast_solver(problem_fine, "fixed_point")
            U_fine, M_fine, info_fine = solver_fine.solve(max_iterations=50, tolerance=1e-5)

            assume(info_coarse.get("converged", False) and info_fine.get("converged", False))

            # Interpolate coarse solution to fine grid for comparison
            x_coarse = problem_coarse.xSpace
            x_fine = problem_fine.xSpace

            U_coarse_interp = np.zeros_like(U_fine)
            M_coarse_interp = np.zeros_like(M_fine)

            for t in range(U_fine.shape[0]):
                U_coarse_interp[t, :] = np.interp(x_fine, x_coarse, U_coarse[t, :])
                M_coarse_interp[t, :] = np.interp(x_fine, x_coarse, M_coarse[t, :])

            # Property: Fine grid solution should be more accurate
            # (measured by smoother convergence history)
            coarse_error = info_coarse.get("final_error", 1.0)
            fine_error = info_fine.get("final_error", 1.0)

            note(f"Coarse final error: {coarse_error:.1e}")
            note(f"Fine final error: {fine_error:.1e}")

            # Property: Finer grid should achieve better or similar convergence
            assert (
                fine_error <= coarse_error * 2
            ), f"Fine grid worse convergence: {fine_error:.1e} vs {coarse_error:.1e}"

            # Property: Solutions should be reasonably close
            U_diff = np.mean(np.abs(U_fine - U_coarse_interp))
            M_diff = np.mean(np.abs(M_fine - M_coarse_interp))

            note(f"Mean U difference: {U_diff:.3f}")
            note(f"Mean M difference: {M_diff:.3f}")

            # Allow reasonable difference due to discretization
            assert U_diff < 1.0, f"U solutions too different: {U_diff:.3f}"
            assert M_diff < 0.5, f"M solutions too different: {M_diff:.3f}"

        except ConvergenceError:
            assume(False)


class TestMFGParameterValidation:
    """Test parameter validation and error handling properties."""

    @given(Nx=st.integers(min_value=1, max_value=5), Nt=st.integers(min_value=1, max_value=5))  # Too small grids
    @settings(max_examples=10)
    def test_small_grid_handling_property(self, Nx, Nt):
        """
        Property: Solvers should handle or reject very small grids gracefully.

        Either solve successfully or raise appropriate exceptions.
        """
        note(f"Testing small grid handling: Nx={Nx}, Nt={Nt}")

        try:
            problem = ExampleMFGProblem(Nx=Nx, Nt=Nt, T=1.0, sigma=1.0)
            solver = create_fast_solver(problem, "fixed_point")

            # Should either work or raise a clear error
            U, M, info = solver.solve(max_iterations=20, tolerance=1e-3)

            # If it works, check basic properties
            if info.get("converged", False):
                assert U.shape == (Nt + 1, Nx + 1), "Incorrect solution shape"
                assert M.shape == (Nt + 1, Nx + 1), "Incorrect density shape"
                assert np.all(np.isfinite(U)), "Solution contains invalid values"
                assert np.all(np.isfinite(M)), "Density contains invalid values"
                assert np.all(M >= 0), "Density has negative values"

        except (ConfigurationError, ValueError, ConvergenceError):
            # These are acceptable exceptions for small grids
            pass

    @given(
        sigma=st.one_of(
            st.floats(min_value=-1.0, max_value=0.0),  # Negative diffusion
            st.floats(min_value=5.0, max_value=10.0),  # Very large diffusion
            st.just(0.0),  # Zero diffusion
        )
    )
    @settings(max_examples=15)
    def test_extreme_diffusion_parameter_property(self, sigma):
        """
        Property: Solvers should handle extreme diffusion parameters appropriately.

        Invalid parameters should be rejected, extreme valid parameters
        should either work or fail gracefully.
        """
        note(f"Testing extreme diffusion: σ={sigma}")

        try:
            problem = ExampleMFGProblem(Nx=20, Nt=10, T=1.0, sigma=sigma)
            solver = create_fast_solver(problem, "fixed_point")

            # Should either work with extreme parameters or raise clear errors
            U, M, info = solver.solve(max_iterations=30, tolerance=1e-3)

            # If it works, solutions should still be valid
            if info.get("converged", False):
                assert np.all(np.isfinite(U)), "Value function not finite"
                assert np.all(np.isfinite(M)), "Density function not finite"
                assert np.all(M >= 0), "Density has negative values"

                # Check that extreme diffusion affects solution as expected
                if sigma > 2.0:
                    # Very high diffusion should lead to smooth solutions
                    M_variation = np.sum(np.abs(np.diff(M, axis=1)))
                    note(f"High diffusion variation: {M_variation:.3f}")
                    assert M_variation < 10.0, "Solution not smooth enough for high diffusion"

        except (ConfigurationError, ValueError, ConvergenceError):
            # These are acceptable for extreme parameters
            pass

    @given(
        tolerance=st.one_of(
            st.floats(min_value=1e-12, max_value=1e-10),  # Very tight tolerance
            st.floats(min_value=0.1, max_value=1.0),  # Very loose tolerance
            st.just(0.0),  # Zero tolerance
        )
    )
    @settings(max_examples=10)
    def test_extreme_tolerance_property(self, tolerance):
        """
        Property: Solvers should handle extreme tolerance values appropriately.
        """
        note(f"Testing extreme tolerance: {tolerance:.1e}")

        try:
            problem = ExampleMFGProblem(Nx=15, Nt=8, T=1.0, sigma=0.5)
            solver = create_fast_solver(problem, "fixed_point")

            if tolerance <= 0:
                # Zero or negative tolerance should raise an error
                with pytest.raises((ValueError, ConfigurationError)):
                    solver.solve(max_iterations=20, tolerance=tolerance)
            else:
                # Extreme but positive tolerances should work or timeout gracefully
                _U, _M, info = solver.solve(max_iterations=100, tolerance=tolerance)

                # Very tight tolerances might not converge within iteration limit
                if tolerance < 1e-10:
                    # Should either converge or report non-convergence clearly
                    pass  # This is expected behavior
                else:
                    # Loose tolerances should generally converge quickly
                    if info.get("converged", False):
                        final_error = info.get("final_error", tolerance)
                        assert (
                            final_error <= tolerance * 10
                        ), f"Final error {final_error:.1e} too large for tolerance {tolerance:.1e}"

        except (ConfigurationError, ValueError, ConvergenceError):
            # These are acceptable for extreme tolerances
            pass


# === Test Configuration ===


def pytest_configure(config):
    """Configure pytest for property-based testing."""
    config.addinivalue_line("markers", "property: property-based tests using Hypothesis")


if __name__ == "__main__":
    # Run property-based tests directly
    pytest.main([__file__, "-v", "--tb=short"])
