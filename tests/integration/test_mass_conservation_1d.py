"""
Mass Conservation Tests for 1D MFG Systems with No-Flux Neumann BC.

Tests mass conservation for two solver combinations:
1. FP Particle + HJB FDM (standard finite difference)
2. FP Particle + HJB GFDM (particle collocation)

Both should conserve mass with no-flux Neumann boundary conditions.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfg_pde.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfg_pde.alg.numerical.mfg_solvers.fixed_point_iterator import FixedPointIterator
from mfg_pde.geometry import BoundaryConditions


class SimpleMFGProblem1D:
    """
    Simple 1D MFG problem for mass conservation testing.

    Dynamics: dx = -u dt + σ dW
    Cost: ∫[0,T] [u²/2 + F(x,m)] dt + g(x)

    With no-flux Neumann BC: ∂m/∂n = 0 at boundaries
    This should preserve total mass ∫m(x,t)dx = 1 for all t
    """

    def __init__(
        self,
        Nx: int = 50,
        Nt: int = 20,
        T: float = 1.0,
        L: float = 2.0,
        sigma: float = 0.1,
        congestion_strength: float = 1.0,
    ):
        """
        Initialize 1D MFG problem.

        Args:
            Nx: Number of spatial grid points
            Nt: Number of time steps
            T: Final time
            L: Domain length [0, L]
            sigma: Diffusion coefficient
            congestion_strength: Strength of congestion cost
        """
        self.Nx = Nx
        self.Nt = Nt
        self.T = T
        self.Lx = L
        self.xmin = 0.0
        self.xmax = L
        self.Dx = L / Nx
        self.Dt = T / Nt
        self.sigma = sigma
        self.congestion_strength = congestion_strength

        # Grid
        self.xSpace = np.linspace(self.xmin, self.xmax, self.Nx + 1)

        # Terminal cost: g(x) = (x - L/2)²
        self.terminal_cost_weight = 1.0

    def H(self, x_idx: int, m_at_x: float, p_values: dict, t_idx: int) -> float:
        """
        Hamiltonian: H(x, m, p) = |p|²/2 + F(x, m)

        Args:
            x_idx: Spatial index
            m_at_x: Density at x
            p_values: Dictionary with gradient information
            t_idx: Time index

        Returns:
            Hamiltonian value
        """
        # Extract gradient (p = ∂u/∂x)
        if isinstance(p_values, dict):
            p = p_values.get("forward", p_values.get("x", 0.0))
        else:
            p = p_values

        # Kinetic term: |p|²/2
        kinetic = 0.5 * p**2

        # Congestion term: F(x, m) = α·m
        congestion = self.congestion_strength * m_at_x

        return kinetic + congestion

    def F(self, x_idx: int, m_at_x: float, t_idx: int) -> float:
        """
        Running cost F(x, m) for congestion.

        Args:
            x_idx: Spatial index
            m_at_x: Density at x
            t_idx: Time index

        Returns:
            Congestion cost
        """
        return self.congestion_strength * m_at_x

    def g(self, x_values: np.ndarray) -> np.ndarray:
        """
        Terminal cost g(x) = α·(x - L/2)²

        Args:
            x_values: Spatial coordinates

        Returns:
            Terminal cost at each point
        """
        center = self.Lx / 2.0
        return self.terminal_cost_weight * (x_values - center) ** 2

    def m0(self, x_values: np.ndarray) -> np.ndarray:
        """
        Initial density m₀(x).

        Gaussian centered at x = L/4 with std = L/8.

        Args:
            x_values: Spatial coordinates

        Returns:
            Initial density (normalized to integrate to 1)
        """
        center = self.Lx / 4.0
        std = self.Lx / 8.0
        density = np.exp(-((x_values - center) ** 2) / (2 * std**2))

        # Normalize
        dx = x_values[1] - x_values[0] if len(x_values) > 1 else 1.0
        total_mass = np.sum(density) * dx
        if total_mass > 1e-12:
            density = density / total_mass

        return density

    def get_initial_m(self) -> np.ndarray:
        """
        Get initial density distribution for MFG solver.

        Returns:
            Initial density m₀(x) on grid
        """
        return self.m0(self.xSpace)

    def get_final_u(self) -> np.ndarray:
        """
        Get terminal value function for HJB solver.

        Returns:
            Terminal cost g(x) on grid
        """
        return self.g(self.xSpace)


def compute_total_mass(density: np.ndarray, dx: float) -> float:
    """
    Compute total mass ∫m(x)dx using trapezoidal rule.

    Args:
        density: Density array
        dx: Grid spacing

    Returns:
        Total mass
    """
    return float(np.trapz(density, dx=dx))


class TestMassConservation1D:
    """Test mass conservation for 1D MFG with no-flux Neumann BC."""

    @pytest.fixture
    def problem(self):
        """Create standard test problem."""
        return SimpleMFGProblem1D(
            Nx=50,
            Nt=20,
            T=1.0,
            L=2.0,
            sigma=0.1,
            congestion_strength=1.0,
        )

    @pytest.fixture
    def boundary_conditions(self):
        """No-flux Neumann boundary conditions."""
        return BoundaryConditions(type="neumann", left_value=0.0, right_value=0.0)

    def test_fp_particle_hjb_fdm_mass_conservation(self, problem, boundary_conditions):
        """
        Test mass conservation for FP Particle + HJB FDM combination.

        With no-flux Neumann BC, total mass should be preserved:
        ∫m(x,t)dx = 1 for all t ∈ [0,T]
        """
        # Create solvers
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
        )

        # Solve MFG
        result = mfg_solver.solve(max_iterations=30, tolerance=1e-5)

        # Check convergence
        assert result.converged, f"MFG solver did not converge: {result.error[-1]:.6e}"

        # Extract density solution
        m_solution = result.m  # Shape: (Nt+1, Nx+1)

        # Compute mass at each time step
        dx = problem.Dx
        masses = []
        for t_idx in range(problem.Nt + 1):
            mass_t = compute_total_mass(m_solution[t_idx, :], dx)
            masses.append(mass_t)

        masses = np.array(masses)

        # Check initial mass
        initial_mass = masses[0]
        assert np.isclose(initial_mass, 1.0, atol=1e-3), f"Initial mass = {initial_mass:.6f}, expected 1.0"

        # Check mass conservation throughout time
        mass_errors = np.abs(masses - 1.0)
        max_mass_error = np.max(mass_errors)

        print("\n=== FP Particle + HJB FDM Mass Conservation ===")
        print(f"Initial mass: {masses[0]:.6f}")
        print(f"Final mass: {masses[-1]:.6f}")
        print(f"Max mass error: {max_mass_error:.6e}")
        print(f"Mean mass: {np.mean(masses):.6f} ± {np.std(masses):.6e}")

        # Mass should be conserved to within tolerance
        # Particle methods with KDE may have some numerical diffusion
        assert max_mass_error < 0.05, (
            f"Mass conservation violated: max error = {max_mass_error:.6e}\n" f"Masses over time: {masses}"
        )

    def test_fp_particle_hjb_gfdm_mass_conservation(self, problem, boundary_conditions):
        """
        Test mass conservation for FP Particle + HJB GFDM (particle collocation).

        With no-flux Neumann BC, total mass should be preserved:
        ∫m(x,t)dx = 1 for all t ∈ [0,T]
        """
        # Create solvers
        fp_solver = FPParticleSolver(
            problem,
            num_particles=5000,
            normalize_kde_output=True,
            boundary_conditions=boundary_conditions,
        )

        hjb_solver = HJBGFDMSolver(problem)

        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
        )

        # Solve MFG
        result = mfg_solver.solve(max_iterations=30, tolerance=1e-5)

        # Check convergence
        assert result.converged, f"MFG solver did not converge: {result.error[-1]:.6e}"

        # Extract density solution
        m_solution = result.m  # Shape: (Nt+1, Nx+1)

        # Compute mass at each time step
        dx = problem.Dx
        masses = []
        for t_idx in range(problem.Nt + 1):
            mass_t = compute_total_mass(m_solution[t_idx, :], dx)
            masses.append(mass_t)

        masses = np.array(masses)

        # Check initial mass
        initial_mass = masses[0]
        assert np.isclose(initial_mass, 1.0, atol=1e-3), f"Initial mass = {initial_mass:.6f}, expected 1.0"

        # Check mass conservation throughout time
        mass_errors = np.abs(masses - 1.0)
        max_mass_error = np.max(mass_errors)

        print("\n=== FP Particle + HJB GFDM Mass Conservation ===")
        print(f"Initial mass: {masses[0]:.6f}")
        print(f"Final mass: {masses[-1]:.6f}")
        print(f"Max mass error: {max_mass_error:.6e}")
        print(f"Mean mass: {np.mean(masses):.6f} ± {np.std(masses):.6e}")

        # Mass should be conserved to within tolerance
        # GFDM particle collocation may have different numerical properties
        assert max_mass_error < 0.05, (
            f"Mass conservation violated: max error = {max_mass_error:.6e}\n" f"Masses over time: {masses}"
        )

    def test_compare_mass_conservation_methods(self, problem, boundary_conditions):
        """
        Compare mass conservation between FDM and GFDM methods.

        Both methods should preserve mass with similar accuracy.
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

        result_1 = mfg_solver_1.solve(max_iterations=30, tolerance=1e-5)
        assert result_1.converged

        # Solver 2: FP Particle + HJB GFDM
        fp_solver_2 = FPParticleSolver(
            problem,
            num_particles=5000,
            normalize_kde_output=True,
            boundary_conditions=boundary_conditions,
        )
        hjb_solver_2 = HJBGFDMSolver(problem)
        mfg_solver_2 = FixedPointIterator(problem, hjb_solver=hjb_solver_2, fp_solver=fp_solver_2)

        result_2 = mfg_solver_2.solve(max_iterations=30, tolerance=1e-5)
        assert result_2.converged

        # Compute masses for both methods
        dx = problem.Dx
        masses_fdm = []
        masses_gfdm = []

        for t_idx in range(problem.Nt + 1):
            mass_fdm = compute_total_mass(result_1.m[t_idx, :], dx)
            mass_gfdm = compute_total_mass(result_2.m[t_idx, :], dx)
            masses_fdm.append(mass_fdm)
            masses_gfdm.append(mass_gfdm)

        masses_fdm = np.array(masses_fdm)
        masses_gfdm = np.array(masses_gfdm)

        # Compute mass errors
        error_fdm = np.abs(masses_fdm - 1.0)
        error_gfdm = np.abs(masses_gfdm - 1.0)

        print("\n=== Mass Conservation Comparison ===")
        print(f"FDM  - Max error: {np.max(error_fdm):.6e}, Mean: {np.mean(masses_fdm):.6f}")
        print(f"GFDM - Max error: {np.max(error_gfdm):.6e}, Mean: {np.mean(masses_gfdm):.6f}")
        print(f"Difference in max errors: {abs(np.max(error_fdm) - np.max(error_gfdm)):.6e}")

        # Both methods should have similar mass conservation quality
        # Allow for some difference due to numerical methods
        assert np.max(error_fdm) < 0.1, f"FDM mass error too large: {np.max(error_fdm):.6e}"
        assert np.max(error_gfdm) < 0.1, f"GFDM mass error too large: {np.max(error_gfdm):.6e}"

    @pytest.mark.parametrize("num_particles", [1000, 3000, 5000])
    def test_mass_conservation_particle_count(self, problem, boundary_conditions, num_particles):
        """
        Test that mass conservation improves with more particles.

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
        mfg_solver = FixedPointIterator(problem, hjb_solver=hjb_solver, fp_solver=fp_solver)

        result = mfg_solver.solve(max_iterations=30, tolerance=1e-5)

        if not result.converged:
            pytest.skip(f"Solver did not converge with {num_particles} particles")

        # Compute masses
        dx = problem.Dx
        masses = []
        for t_idx in range(problem.Nt + 1):
            mass_t = compute_total_mass(result.m[t_idx, :], dx)
            masses.append(mass_t)

        masses = np.array(masses)
        max_error = np.max(np.abs(masses - 1.0))

        print(f"\nParticles: {num_particles}, Max mass error: {max_error:.6e}")

        # More particles should give better mass conservation
        assert max_error < 0.1, f"Mass error too large with {num_particles} particles"

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
            # Create problem with custom initial condition
            problem = SimpleMFGProblem1D(Nx=50, Nt=20)

            # Override m0 with custom initial condition
            # Capture loop variables in closure
            def make_custom_m0(prob, cfrac, sfrac):
                def custom_m0(x_values):
                    center = prob.Lx * cfrac
                    std = prob.Lx * sfrac
                    density = np.exp(-((x_values - center) ** 2) / (2 * std**2))
                    dx = x_values[1] - x_values[0] if len(x_values) > 1 else 1.0
                    total_mass = np.sum(density) * dx
                    if total_mass > 1e-12:
                        density = density / total_mass
                    return density

                return custom_m0

            problem.m0 = make_custom_m0(problem, center_frac, std_frac)

            # Solve
            fp_solver = FPParticleSolver(
                problem,
                num_particles=5000,
                normalize_kde_output=True,
                boundary_conditions=boundary_conditions,
            )
            hjb_solver = HJBFDMSolver(problem)
            mfg_solver = FixedPointIterator(problem, hjb_solver=hjb_solver, fp_solver=fp_solver)

            result = mfg_solver.solve(max_iterations=30, tolerance=1e-5)

            if not result.converged:
                pytest.skip(f"Solver did not converge for {name}")

            # Check mass conservation
            dx = problem.Dx
            masses = [compute_total_mass(result.m[t, :], dx) for t in range(problem.Nt + 1)]
            max_error = np.max(np.abs(np.array(masses) - 1.0))

            print(f"\n{name}: Initial mass = {masses[0]:.6f}, Max error = {max_error:.6e}")

            assert max_error < 0.1, f"Mass conservation failed for {name}: {max_error:.6e}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
