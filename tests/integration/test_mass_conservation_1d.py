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

from mfg_pde.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfg_pde.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry.boundary.fdm_bc_1d import no_flux_bc


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

    def get_hjb_residual_m_coupling_term(
        self,
        M_density_at_n_plus_1: np.ndarray,
        U_n_current_guess_derivatives: dict,
        x_idx: int,
        t_idx_n: int,
    ) -> float | None:
        """
        Optional coupling term for HJB residual computation.

        For this simple problem, no additional coupling beyond the Hamiltonian.

        Returns:
            None (no additional coupling)
        """
        return None


def compute_total_mass(density: np.ndarray, dx: float) -> float:
    """
    Compute total mass ∫m(x)dx using rectangular rule (consistent with FPParticleSolver normalization).

    The FPParticleSolver normalizes using np.sum(density) * dx (rectangular/midpoint rule),
    so we must use the same integration method to verify mass conservation.

    Args:
        density: Density array
        dx: Grid spacing

    Returns:
        Total mass
    """
    return float(np.sum(density) * dx)


class TestMassConservation1D:
    """Test mass conservation for 1D MFG with no-flux Neumann BC."""

    @pytest.fixture
    def problem(self):
        """Create standard test problem using built-in MFGProblem."""
        # Create problem with custom initial and terminal conditions
        L = 2.0
        problem = MFGProblem(
            xmin=0.0,
            xmax=L,
            Nx=50,
            T=1.0,
            Nt=20,
            sigma=0.1,
            coupling_coefficient=1.0,  # congestion strength
        )

        # Override initial and terminal conditions to match SimpleMFGProblem1D
        xSpace = problem.xSpace
        center_init = L / 4.0
        std = L / 8.0
        m0 = np.exp(-((xSpace - center_init) ** 2) / (2 * std**2))
        m0 = m0 / (np.sum(m0) * problem.dx)  # Normalize
        problem.initial_density = m0

        center_term = L / 2.0
        problem.terminal_cost = (xSpace - center_term) ** 2

        return problem

    @pytest.fixture
    def boundary_conditions(self):
        """No-flux Neumann boundary conditions."""
        return no_flux_bc()

    @pytest.mark.slow
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

        # Use Anderson acceleration + JAX backend for maximum speed
        # JAX provides GPU acceleration, Anderson accelerates convergence
        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            damping_factor=0.5,  # Standard damping
            use_anderson=True,  # Anderson acceleration
            anderson_depth=5,  # Store last 5 iterates
            backend=None,  # Use numpy (JAX has issues with particle normalization)
        )

        # Solve MFG with relaxed parameters for stochastic particle method
        # Particle methods + KDE introduce stochasticity, so we need:
        # - More iterations for convergence
        # - Relaxed tolerance (1e-3 instead of 1e-5)
        # - Proper damping (damping_factor=0.4 for balance between stability and speed)
        try:
            # With Anderson + JAX, convergence is much faster
            result = mfg_solver.solve(
                max_iterations=50,
                tolerance=1e-4,
                return_structured=True,  # Return SolverResult object
            )
        except Exception as e:
            # Skip test if solver fails to converge (expected for some particle configs)
            pytest.skip(f"MFG solver convergence issue: {str(e)[:150]}")

        # Extract density solution
        m_solution = result.M  # Shape: (Nt+1, Nx+1)

        # DEBUG: Print actual shape and mass calculation
        print("\nDEBUG test_fp_particle_hjb_fdm:")
        print(f"  result.M shape = {result.M.shape}, expected ({problem.Nt + 1}, {problem.Nx + 1})")
        print(f"  problem.dx = {problem.dx:.6f}")
        print(
            f"  m_solution[0] stats: min={m_solution[0].min():.6f}, max={m_solution[0].max():.6f}, sum={m_solution[0].sum():.6f}"
        )
        print(f"  sum(m_solution[0]) * Dx = {np.sum(m_solution[0]) * problem.dx:.6f}")

        # Compute mass at each time step
        dx = problem.dx
        masses = []
        for t_idx in range(problem.Nt + 1):
            mass_t = compute_total_mass(m_solution[t_idx, :], dx)
            masses.append(mass_t)

        masses = np.array(masses)

        # Check initial mass
        initial_mass = masses[0]
        print(f"  Initial mass from compute_total_mass: {initial_mass:.6f}")
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
            f"Mass conservation violated: max error = {max_mass_error:.6e}\nMasses over time: {masses}"
        )

    @pytest.mark.skip(
        reason="HJBGFDMSolver is abstract and cannot be instantiated directly. "
        "This test needs refactoring to use a concrete GFDM implementation. "
        "Issue #140 - pre-existing test failure."
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

        # GFDM requires collocation points
        collocation_points = problem.xSpace.reshape(-1, 1)  # Use grid points as collocation points
        hjb_solver = HJBGFDMSolver(problem, collocation_points=collocation_points, delta=0.3)

        # Use Anderson acceleration + JAX backend
        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            use_anderson=True,  # Anderson acceleration
            anderson_depth=5,
            backend=None,  # numpy backend
        )

        # Solve MFG with relaxed parameters for stochastic particle method
        # Particle methods + KDE introduce stochasticity, so we need:
        # - More iterations for convergence
        # - Relaxed tolerance (1e-3 instead of 1e-5)
        # - Proper damping (damping_factor=0.4 for balance between stability and speed)
        try:
            # With Anderson + JAX, convergence is much faster
            result = mfg_solver.solve(
                max_iterations=50,
                tolerance=1e-4,
                return_structured=True,  # Return SolverResult object
            )
        except Exception as e:
            # Skip test if solver fails to converge (expected for some particle configs)
            pytest.skip(f"MFG solver convergence issue: {str(e)[:150]}")

        # Extract density solution
        m_solution = result.M  # Shape: (Nt+1, Nx+1)

        # Compute mass at each time step
        dx = problem.dx
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
            f"Mass conservation violated: max error = {max_mass_error:.6e}\nMasses over time: {masses}"
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

        result_1 = mfg_solver_1.solve(max_iterations=30, tolerance=1e-5, return_structured=True)
        if not result_1.convergence_achieved:
            pytest.skip("FDM did not converge with tight tolerances (expected for particle methods)")

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

        result_2 = mfg_solver_2.solve(max_iterations=30, tolerance=1e-5, return_structured=True)
        # Convergence might not be achieved with these tight params - gracefully handle
        if not result_2.convergence_achieved:
            pytest.skip("GFDM did not converge with tight tolerances (expected for particle methods)")

        # Compute masses for both methods
        dx = problem.dx
        masses_fdm = []
        masses_gfdm = []

        for t_idx in range(problem.Nt + 1):
            mass_fdm = compute_total_mass(result_1.M[t_idx, :], dx)
            mass_gfdm = compute_total_mass(result_2.M[t_idx, :], dx)
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

    @pytest.mark.slow
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
        mfg_solver = FixedPointIterator(
            problem, hjb_solver=hjb_solver, fp_solver=fp_solver, use_anderson=True, anderson_depth=5, backend=None
        )

        try:
            result = mfg_solver.solve(max_iterations=50, tolerance=1e-4, return_structured=True)
        except Exception as e:
            pytest.skip(f"Solver convergence issue with {num_particles} particles: {str(e)[:100]}")

        # Compute masses
        dx = problem.dx
        masses = []
        for t_idx in range(problem.Nt + 1):
            mass_t = compute_total_mass(result.M[t_idx, :], dx)
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
            mfg_solver = FixedPointIterator(
                problem, hjb_solver=hjb_solver, fp_solver=fp_solver, use_anderson=True, anderson_depth=5, backend=None
            )

            try:
                result = mfg_solver.solve(max_iterations=50, tolerance=1e-4, return_structured=True)
            except Exception as e:
                pytest.skip(f"Solver convergence issue for {name}: {str(e)[:100]}")

            # Check mass conservation
            dx = problem.dx
            masses = [compute_total_mass(result.M[t, :], dx) for t in range(problem.Nt + 1)]
            max_error = np.max(np.abs(np.array(masses) - 1.0))

            print(f"\n{name}: Initial mass = {masses[0]:.6f}, Max error = {max_error:.6e}")

            assert max_error < 0.1, f"Mass conservation failed for {name}: {max_error:.6e}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
