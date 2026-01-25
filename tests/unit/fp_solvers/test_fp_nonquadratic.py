#!/usr/bin/env python3
"""
Unit tests for non-quadratic Hamiltonian support in FP solvers (Issue #573).

Tests the new drift_field parameter for L1, quartic, and custom Hamiltonians.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver
from mfg_pde.geometry import TensorProductGrid, no_flux_bc


class TestNonQuadraticHamiltonians:
    """Test FP solvers with non-quadratic Hamiltonians."""

    @pytest.fixture
    def problem_1d(self):
        """Create 1D test problem."""
        geometry = TensorProductGrid(
            dimension=1, bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1)
        )
        return MFGProblem(
            geometry=geometry,
            T=1.0,
            Nt=20,
            sigma=0.1,
            coupling_coefficient=0.0,  # No coupling for unit tests
        )

    @pytest.fixture
    def solver_1d(self, problem_1d):
        """Create 1D FDM solver."""
        bc = no_flux_bc(dimension=1)
        return FPFDMSolver(problem_1d, boundary_conditions=bc)

    @pytest.fixture
    def initial_density_1d(self, problem_1d):
        """Gaussian initial density."""
        Nx = problem_1d.geometry.get_grid_shape()[0]  # 1D: (Nx,)
        x = np.linspace(0, 1, Nx)
        m0 = np.exp(-50 * (x - 0.5) ** 2)
        m0 = m0 / np.trapezoid(m0, dx=1.0 / (Nx - 1))
        return m0

    def test_drift_shape_validation(self, solver_1d, initial_density_1d, problem_1d):
        """Test that drift_field shape is validated."""
        Nt, Nx = problem_1d.Nt + 1, problem_1d.geometry.get_grid_shape()[0]

        # Wrong shape should raise error
        wrong_shape_drift = np.zeros((Nt, Nx + 10))  # Wrong Nx

        with pytest.raises((ValueError, IndexError, AssertionError)):
            solver_1d.solve_fp_system(
                M_initial=initial_density_1d,
                drift_field=wrong_shape_drift,
            )

    def test_backward_compatibility_quadratic(self, solver_1d, initial_density_1d, problem_1d):
        """Test that drift_field still works (quadratic H: α* = -∇U)."""
        Nt, Nx = problem_1d.Nt + 1, problem_1d.geometry.get_grid_shape()[0]
        x = np.linspace(0, 1, Nx)

        # Create synthetic value function U(t,x) = x²
        U_hjb = np.zeros((Nt, Nx))
        for t_idx in range(Nt):
            U_hjb[t_idx, :] = x**2

        # Solve with legacy drift_field API
        M_legacy = solver_1d.solve_fp_system(
            M_initial=initial_density_1d,
            drift_field=U_hjb,
        )

        # Should produce valid density
        assert M_legacy.shape == (Nt, Nx)
        assert np.all(M_legacy >= -1e-10), "Density should be non-negative"
        assert not np.any(np.isnan(M_legacy)), "No NaN values"
        assert not np.any(np.isinf(M_legacy)), "No Inf values"

    def test_equivalence_quadratic_explicit(self, solver_1d, initial_density_1d, problem_1d):
        """Test that drift_field and drift_field give same result for quadratic H."""
        Nt, Nx = problem_1d.Nt + 1, problem_1d.geometry.get_grid_shape()[0]
        x = np.linspace(0, 1, Nx)
        dx = x[1] - x[0]

        # Create synthetic value function U(t,x) = x²
        U_hjb = np.zeros((Nt, Nx))
        for t_idx in range(Nt):
            U_hjb[t_idx, :] = x**2

        # Compute gradient manually: ∇U = 2x (central differences)
        grad_U = np.zeros((Nt, Nx))
        for t_idx in range(Nt):
            grad_U[t_idx, 1:-1] = (U_hjb[t_idx, 2:] - U_hjb[t_idx, :-2]) / (2 * dx)
            # Forward/backward at boundaries
            grad_U[t_idx, 0] = (U_hjb[t_idx, 1] - U_hjb[t_idx, 0]) / dx
            grad_U[t_idx, -1] = (U_hjb[t_idx, -1] - U_hjb[t_idx, -2]) / dx

        # Quadratic H: α* = -∇U
        alpha_quadratic = -grad_U

        # Solve with drift_field (legacy)
        M_legacy = solver_1d.solve_fp_system(
            M_initial=initial_density_1d,
            drift_field=U_hjb,
        )

        # Solve with drift_field (new API)
        M_new = solver_1d.solve_fp_system(
            M_initial=initial_density_1d,
            drift_field=alpha_quadratic,
        )

        # Should produce nearly identical results
        np.testing.assert_allclose(
            M_new,
            M_legacy,
            rtol=1e-10,
            atol=1e-12,
            err_msg="drift_field and drift_field should match for quadratic H",
        )

    def test_l1_control_drift(self, solver_1d, initial_density_1d, problem_1d):
        """Test L1 Hamiltonian: H = |p|, α* = -sign(∇U)."""
        Nt, Nx = problem_1d.Nt + 1, problem_1d.geometry.get_grid_shape()[0]
        x = np.linspace(0, 1, Nx)
        dx = x[1] - x[0]

        # Create synthetic value function U(t,x) = (x - 0.5)²
        U_hjb = np.zeros((Nt, Nx))
        for t_idx in range(Nt):
            U_hjb[t_idx, :] = (x - 0.5) ** 2

        # Compute gradient: ∇U = 2(x - 0.5)
        grad_U = np.zeros((Nt, Nx))
        for t_idx in range(Nt):
            grad_U[t_idx, 1:-1] = (U_hjb[t_idx, 2:] - U_hjb[t_idx, :-2]) / (2 * dx)
            grad_U[t_idx, 0] = (U_hjb[t_idx, 1] - U_hjb[t_idx, 0]) / dx
            grad_U[t_idx, -1] = (U_hjb[t_idx, -1] - U_hjb[t_idx, -2]) / dx

        # L1 Hamiltonian: α* = -sign(∇U)
        alpha_L1 = -np.sign(grad_U)

        # Solve with L1 drift
        M_L1 = solver_1d.solve_fp_system(
            M_initial=initial_density_1d,
            drift_field=alpha_L1,
        )

        # Should produce valid density
        assert M_L1.shape == (Nt, Nx)
        assert np.all(M_L1 >= -1e-10), "Density should be non-negative"
        assert not np.any(np.isnan(M_L1)), "No NaN values"

        # L1 drift behaves differently from quadratic - verify density evolution occurs
        # Check that solution changes over time (not static)
        density_change = np.linalg.norm(M_L1[-1, :] - M_L1[0, :])
        assert density_change > 1e-6, "L1 drift should evolve density over time"

    def test_quartic_control_drift(self, solver_1d, initial_density_1d, problem_1d):
        """Test quartic Hamiltonian: H = (1/4)|p|⁴, α* = -(∇U)^(1/3)."""
        Nt, Nx = problem_1d.Nt + 1, problem_1d.geometry.get_grid_shape()[0]
        x = np.linspace(0, 1, Nx)
        dx = x[1] - x[0]

        # Create synthetic value function U(t,x) = (x - 0.5)²
        U_hjb = np.zeros((Nt, Nx))
        for t_idx in range(Nt):
            U_hjb[t_idx, :] = (x - 0.5) ** 2

        # Compute gradient: ∇U = 2(x - 0.5)
        grad_U = np.zeros((Nt, Nx))
        for t_idx in range(Nt):
            grad_U[t_idx, 1:-1] = (U_hjb[t_idx, 2:] - U_hjb[t_idx, :-2]) / (2 * dx)
            grad_U[t_idx, 0] = (U_hjb[t_idx, 1] - U_hjb[t_idx, 0]) / dx
            grad_U[t_idx, -1] = (U_hjb[t_idx, -1] - U_hjb[t_idx, -2]) / dx

        # Quartic Hamiltonian: α* = -(∇U)^(1/3)
        # Need to preserve sign
        alpha_quartic = -np.sign(grad_U) * np.abs(grad_U) ** (1.0 / 3.0)

        # Solve with quartic drift
        M_quartic = solver_1d.solve_fp_system(
            M_initial=initial_density_1d,
            drift_field=alpha_quartic,
        )

        # Should produce valid density
        assert M_quartic.shape == (Nt, Nx)
        assert np.all(M_quartic >= -1e-10), "Density should be non-negative"
        assert not np.any(np.isnan(M_quartic)), "No NaN values"

        # Quartic drift behaves differently - verify density evolution occurs
        # Check that solution changes over time (not static)
        density_change = np.linalg.norm(M_quartic[-1, :] - M_quartic[0, :])
        assert density_change > 1e-6, "Quartic drift should evolve density over time"

    def test_zero_drift_pure_diffusion(self, solver_1d, initial_density_1d, problem_1d):
        """Test zero drift (pure diffusion) with drift_field=0."""
        Nt, Nx = problem_1d.Nt + 1, problem_1d.geometry.get_grid_shape()[0]

        # Zero drift velocity
        alpha_zero = np.zeros((Nt, Nx))

        # Solve with zero drift
        M_diffusion = solver_1d.solve_fp_system(
            M_initial=initial_density_1d,
            drift_field=alpha_zero,
        )

        # Should produce valid density that spreads out (pure diffusion)
        assert M_diffusion.shape == (Nt, Nx)
        assert np.all(M_diffusion >= -1e-10), "Density should be non-negative"

        # Diffusion should reduce peak height and spread mass
        assert M_diffusion[-1, :].max() < initial_density_1d.max(), "Diffusion should reduce peak"

    def test_custom_drift_callable_like(self, solver_1d, initial_density_1d, problem_1d):
        """Test custom drift velocity (e.g., state-constrained control)."""
        Nt, Nx = problem_1d.Nt + 1, problem_1d.geometry.get_grid_shape()[0]
        x = np.linspace(0, 1, Nx)

        # Custom drift: push left in left half, push right in right half
        # Simulates a "barrier" at x=0.5
        alpha_custom = np.zeros((Nt, Nx))
        for t_idx in range(Nt):
            alpha_custom[t_idx, x < 0.5] = -0.5  # Push left
            alpha_custom[t_idx, x >= 0.5] = 0.5  # Push right

        # Solve with custom drift
        M_custom = solver_1d.solve_fp_system(
            M_initial=initial_density_1d,
            drift_field=alpha_custom,
        )

        # Should produce valid density
        assert M_custom.shape == (Nt, Nx)
        assert np.all(M_custom >= -1e-10), "Density should be non-negative"
        assert not np.any(np.isnan(M_custom)), "No NaN values"

    def test_mass_conservation_with_drift_field(self, solver_1d, initial_density_1d, problem_1d):
        """Test that mass is conserved with drift_field."""
        Nt, Nx = problem_1d.Nt + 1, problem_1d.geometry.get_grid_shape()[0]
        x = np.linspace(0, 1, Nx)
        dx = x[1] - x[0]

        # Create synthetic drift
        alpha = np.zeros((Nt, Nx))
        for t_idx in range(Nt):
            alpha[t_idx, :] = 0.5 * np.sin(2 * np.pi * x)

        # Solve
        M = solver_1d.solve_fp_system(
            M_initial=initial_density_1d,
            drift_field=alpha,
        )

        # Check mass conservation at each timestep
        initial_mass = np.trapezoid(initial_density_1d, dx=dx)
        for t_idx in range(Nt):
            mass_t = np.trapezoid(M[t_idx, :], dx=dx)
            np.testing.assert_allclose(
                mass_t,
                initial_mass,
                rtol=1e-2,
                err_msg=f"Mass not conserved at t={t_idx * problem_1d.dt}",
            )


if __name__ == "__main__":
    """Run smoke tests."""
    import sys

    print("Running non-quadratic Hamiltonian tests...")
    pytest.main([__file__, "-v", "--tb=short", *sys.argv[1:]])
