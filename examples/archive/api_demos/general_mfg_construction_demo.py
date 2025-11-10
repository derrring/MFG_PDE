"""
General MFG Problem Construction Demo.

This example demonstrates how to construct any MFG problem by defining
your own Hamiltonians, boundary conditions, and coupling terms using the
general MFG problem builder system.
"""

import sys
from pathlib import Path

import numpy as np

# Add MFG_PDE to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mfg_pde.core.mfg_problem import VALUE_BEFORE_SQUARE_LIMIT, MFGProblemBuilder
from mfg_pde.geometry import BoundaryConditions
from mfg_pde.utils.aux_func import npart, ppart
from mfg_pde.utils.logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("general_mfg_demo", level="INFO")
logger = get_logger(__name__)


def example_1_quadratic_hamiltonian():
    """Example 1: Custom quadratic Hamiltonian with congestion."""
    logger.info(" Example 1: Custom Quadratic Hamiltonian")

    def my_hamiltonian(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        """Custom Hamiltonian: H = (1/2)γ|p|² - V(x) - α*m²"""

        p_forward = p_values.get("forward")
        p_backward = p_values.get("backward")

        if p_forward is None or p_backward is None:
            return np.nan
        if (
            np.isnan(p_forward)
            or np.isinf(p_forward)
            or np.isnan(p_backward)
            or np.isinf(p_backward)
            or np.isnan(m_at_x)
            or np.isinf(m_at_x)
        ):
            return np.nan

        # Use standard npart/ppart processing
        npart_val_fwd = npart(p_forward)
        ppart_val_bwd = ppart(p_backward)

        if abs(npart_val_fwd) > VALUE_BEFORE_SQUARE_LIMIT or abs(ppart_val_bwd) > VALUE_BEFORE_SQUARE_LIMIT:
            return np.nan

        try:
            # Kinetic energy term: (1/2)γ|p|²
            gamma = problem.coupling_coefficient
            kinetic_energy = 0.5 * gamma * (npart_val_fwd**2 + ppart_val_bwd**2)

            # Potential term: V(x)
            potential = problem.f_potential[x_idx]

            # Congestion term: α*m²
            alpha = problem.components.parameters.get("congestion_strength", 1.0)
            congestion = alpha * m_at_x**2

            return kinetic_energy - potential - congestion

        except (OverflowError, IndexError):
            return np.nan

    def my_hamiltonian_dm(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        """Derivative: dH/dm = -2*α*m"""
        if np.isnan(m_at_x) or np.isinf(m_at_x):
            return np.nan

        try:
            alpha = problem.components.parameters.get("congestion_strength", 1.0)
            return -2.0 * alpha * m_at_x
        except OverflowError:
            return np.nan

    def my_potential(x, t=None):
        """Custom potential: V(x) = 0.5*(x-0.5)² + 0.1*sin(4πx)"""
        return 0.5 * (x - 0.5) ** 2 + 0.1 * np.sin(4 * np.pi * x)

    def my_initial_density(x):
        """Custom initial density: Gaussian centered at x=0.2"""
        return np.exp(-50 * (x - 0.2) ** 2)

    def my_final_value(x):
        """Custom final value: u_T(x) = (x-1)²"""
        return (x - 1.0) ** 2

    # Build the problem using the builder pattern
    problem = (
        MFGProblemBuilder()
        .hamiltonian(my_hamiltonian, my_hamiltonian_dm)
        .potential(my_potential)
        .initial_density(my_initial_density)
        .final_value(my_final_value)
        .domain(xmin=0.0, xmax=1.0, Nx=101)
        .time(T=1.0, Nt=101)
        .coefficients(sigma=0.5, coupling_coefficient=1.0)
        .parameters(congestion_strength=2.0)
        .boundary_conditions(BoundaryConditions(type="dirichlet", left_value=0.0, right_value=0.0))
        .description("Quadratic Hamiltonian with congestion", "congestion_control")
        .build()
    )

    logger.info(f"SUCCESS: Created problem: {problem.components.description}")
    logger.info(f"   Parameters: {problem.components.parameters}")

    # Test Hamiltonian evaluation
    test_p_values = {"forward": 0.1, "backward": -0.1}
    test_h = problem.H(x_idx=50, m_at_x=0.5, p_values=test_p_values, t_idx=0)
    logger.info(f"   Test H(x=0.5, m=0.5, p=±0.1) = {test_h:.6f}")

    return problem


def example_2_time_dependent_potential():
    """Example 2: Time-dependent potential and Hamiltonian."""
    logger.info(" Example 2: Time-Dependent Potential")

    def time_dependent_hamiltonian(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        """Hamiltonian with time-dependent control cost."""

        p_forward = p_values.get("forward")
        p_backward = p_values.get("backward")

        if p_forward is None or p_backward is None:
            return np.nan
        if (
            np.isnan(p_forward)
            or np.isinf(p_forward)
            or np.isnan(p_backward)
            or np.isinf(p_backward)
            or np.isnan(m_at_x)
            or np.isinf(m_at_x)
        ):
            return np.nan

        npart_val_fwd = npart(p_forward)
        ppart_val_bwd = ppart(p_backward)

        if abs(npart_val_fwd) > VALUE_BEFORE_SQUARE_LIMIT or abs(ppart_val_bwd) > VALUE_BEFORE_SQUARE_LIMIT:
            return np.nan

        try:
            # Time-dependent control cost: γ(t) = γ₀(1 + 0.5*sin(2πt))
            gamma_0 = problem.coupling_coefficient
            gamma_t = gamma_0 * (1.0 + 0.5 * np.sin(2 * np.pi * current_time))

            kinetic_energy = 0.5 * gamma_t * (npart_val_fwd**2 + ppart_val_bwd**2)
            potential = problem.get_potential_at_time(t_idx)[x_idx]

            # Linear coupling
            coupling = problem.components.parameters.get("coupling_strength", 0.5) * m_at_x

            return kinetic_energy - potential - coupling

        except (OverflowError, IndexError):
            return np.nan

    def time_dependent_hamiltonian_dm(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        """dH/dm for linear coupling."""
        if np.isnan(m_at_x) or np.isinf(m_at_x):
            return np.nan

        coupling_strength = problem.components.parameters.get("coupling_strength", 0.5)
        return -coupling_strength

    def moving_potential(x, t):
        """Moving potential: V(x,t) = 10*exp(-50*(x-0.5-0.3*sin(πt))²)"""
        center = 0.5 + 0.3 * np.sin(np.pi * t)
        return 10.0 * np.exp(-50 * (x - center) ** 2)

    def dispersed_initial(x):
        """Dispersed initial condition."""
        return 1.0 + 0.5 * np.cos(2 * np.pi * x)

    # Build time-dependent problem
    problem = (
        MFGProblemBuilder()
        .hamiltonian(time_dependent_hamiltonian, time_dependent_hamiltonian_dm)
        .potential(moving_potential)
        .initial_density(dispersed_initial)
        .final_value(lambda x: 0.0)  # Zero final cost
        .domain(xmin=0.0, xmax=1.0, Nx=81)
        .time(T=2.0, Nt=81)
        .coefficients(sigma=0.3, coupling_coefficient=0.8)
        .parameters(coupling_strength=0.5)
        .boundary_conditions(BoundaryConditions(type="periodic"))
        .description("Time-dependent potential and control cost", "time_dependent")
        .build()
    )

    logger.info(f"SUCCESS: Created time-dependent problem: {problem.components.description}")

    # Test time-dependent potential
    potential_t0 = problem.get_potential_at_time(0)
    potential_t_half = problem.get_potential_at_time(problem.Nt // 2)
    logger.info(f"   Potential at t=0: max={np.max(potential_t0):.3f}")
    logger.info(f"   Potential at t=T/2: max={np.max(potential_t_half):.3f}")

    return problem


def example_3_nonlocal_coupling():
    """Example 3: Nonlocal coupling with custom jacobian."""
    logger.info(" Example 3: Nonlocal Coupling")

    def nonlocal_hamiltonian(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        """Hamiltonian with nonlocal coupling term."""

        p_forward = p_values.get("forward")
        p_backward = p_values.get("backward")

        if p_forward is None or p_backward is None:
            return np.nan
        if (
            np.isnan(p_forward)
            or np.isinf(p_forward)
            or np.isnan(p_backward)
            or np.isinf(p_backward)
            or np.isnan(m_at_x)
            or np.isinf(m_at_x)
        ):
            return np.nan

        npart_val_fwd = npart(p_forward)
        ppart_val_bwd = ppart(p_backward)

        if abs(npart_val_fwd) > VALUE_BEFORE_SQUARE_LIMIT or abs(ppart_val_bwd) > VALUE_BEFORE_SQUARE_LIMIT:
            return np.nan

        try:
            # Standard kinetic energy
            kinetic_energy = 0.5 * problem.coupling_coefficient * (npart_val_fwd**2 + ppart_val_bwd**2)

            # Potential
            potential = problem.f_potential[x_idx]

            # Nonlocal coupling: ∫ K(x,y) m(y) dy ≈ Σ K(x,y) m(y) Δy
            # For demo, use K(x,y) = exp(-|x-y|/σ)
            sigma_kernel = problem.components.parameters.get("kernel_width", 0.1)
            nonlocal_term = 0.0

            # Access current density (this is a simplified approach)
            if hasattr(problem, "_current_density") and problem._current_density is not None:
                for j in range(len(problem.xSpace)):
                    y_j = problem.xSpace[j]
                    kernel_value = np.exp(-abs(x_position - y_j) / sigma_kernel)
                    nonlocal_term += kernel_value * problem._current_density[j] * problem.dx
            else:
                # Fallback to local term
                nonlocal_term = m_at_x

            coupling_strength = problem.components.parameters.get("nonlocal_strength", 1.0)

            return kinetic_energy - potential - coupling_strength * nonlocal_term

        except (OverflowError, IndexError):
            return np.nan

    def nonlocal_hamiltonian_dm(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        """dH/dm for nonlocal coupling (simplified)."""
        if np.isnan(m_at_x) or np.isinf(m_at_x):
            return np.nan

        # Simplified: just return the local derivative
        coupling_strength = problem.components.parameters.get("nonlocal_strength", 1.0)
        return -coupling_strength

    def custom_jacobian(U_for_jacobian_terms, t_idx_n, problem):
        """Custom Jacobian for nonlocal terms."""
        # This is a placeholder - real implementation would compute
        # the Jacobian of the nonlocal integral term
        Nx = problem.Nx + 1
        J_D = np.ones(Nx) * 0.1
        J_L = np.ones(Nx) * 0.05
        J_U = np.ones(Nx) * 0.05
        return J_D, J_L, J_U

    def custom_coupling(M_density_at_n_plus_1, U_n_current_guess_derivatives, x_idx, t_idx_n, problem):
        """Custom coupling term for residual."""
        # Store current density for nonlocal computation
        problem._current_density = M_density_at_n_plus_1
        return 0.0  # Additional residual term

    # Build nonlocal problem
    problem = (
        MFGProblemBuilder()
        .hamiltonian(nonlocal_hamiltonian, nonlocal_hamiltonian_dm)
        .potential(lambda x: 0.5 * x * (1 - x))  # Simple quadratic well
        .initial_density(lambda x: np.exp(-20 * (x - 0.3) ** 2) + np.exp(-20 * (x - 0.7) ** 2))
        .final_value(lambda x: x**2)
        .jacobian(custom_jacobian)
        .coupling(custom_coupling)
        .domain(xmin=0.0, xmax=1.0, Nx=51)
        .time(T=1.0, Nt=51)
        .coefficients(sigma=0.2, coupling_coefficient=1.0)
        .parameters(nonlocal_strength=0.5, kernel_width=0.1)
        .boundary_conditions(BoundaryConditions(type="neumann", left_value=0.0, right_value=0.0))
        .description("Nonlocal coupling MFG", "nonlocal")
        .build()
    )

    logger.info(f"SUCCESS: Created nonlocal problem: {problem.components.description}")
    logger.info(f"   Has custom Jacobian: {problem.components.hamiltonian_jacobian_func is not None}")
    logger.info(f"   Has custom coupling: {problem.components.coupling_func is not None}")

    return problem


def example_4_portfolio_optimization():
    """Example 4: Portfolio optimization with realistic financial Hamiltonian."""
    logger.info(" Example 4: Portfolio Optimization")

    def portfolio_hamiltonian(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        """Portfolio optimization Hamiltonian with risk aversion."""

        p_forward = p_values.get("forward")
        p_backward = p_values.get("backward")

        if p_forward is None or p_backward is None:
            return np.nan
        if (
            np.isnan(p_forward)
            or np.isinf(p_forward)
            or np.isnan(p_backward)
            or np.isinf(p_backward)
            or np.isnan(m_at_x)
            or np.isinf(m_at_x)
        ):
            return np.nan

        npart_val_fwd = npart(p_forward)
        ppart_val_bwd = ppart(p_backward)

        if abs(npart_val_fwd) > VALUE_BEFORE_SQUARE_LIMIT or abs(ppart_val_bwd) > VALUE_BEFORE_SQUARE_LIMIT:
            return np.nan

        try:
            # Risk aversion parameter
            risk_aversion = problem.components.parameters.get("risk_aversion", 1.0)

            # Transaction cost (quadratic in control)
            transaction_cost = problem.components.parameters.get("transaction_cost", 0.01)
            control_cost = (problem.coupling_coefficient + transaction_cost) * risk_aversion

            kinetic_energy = 0.5 * control_cost * (npart_val_fwd**2 + ppart_val_bwd**2)

            # Market drift and volatility
            drift = problem.components.parameters.get("drift", 0.05)
            problem.components.parameters.get("volatility", 0.2)

            # Position-dependent cost (away from optimal)
            optimal_position = problem.components.parameters.get("optimal_position", 0.0)
            position_cost = 0.5 * (x_position - optimal_position) ** 2

            # Price impact (proportional to density)
            price_impact = problem.components.parameters.get("price_impact", 0.1)
            impact_cost = price_impact * m_at_x**2

            # Market timing term
            time_factor = 1.0 + 0.1 * np.cos(2 * np.pi * current_time / problem.T)

            return kinetic_energy * time_factor - drift * x_position - position_cost - impact_cost

        except (OverflowError, IndexError):
            return np.nan

    def portfolio_hamiltonian_dm(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        """dH/dm for price impact."""
        if np.isnan(m_at_x) or np.isinf(m_at_x):
            return np.nan

        price_impact = problem.components.parameters.get("price_impact", 0.1)
        return -2.0 * price_impact * m_at_x

    def initial_portfolio_distribution(x):
        """Initial portfolio distribution around zero position."""
        return np.exp(-10 * x**2)

    def final_portfolio_value(x):
        """Final portfolio value - liquidation cost."""
        liquidation_cost = 0.01
        return -liquidation_cost * abs(x)

    # Build portfolio problem
    problem = (
        MFGProblemBuilder()
        .hamiltonian(portfolio_hamiltonian, portfolio_hamiltonian_dm)
        .potential(lambda x: 0.0)  # No external potential
        .initial_density(initial_portfolio_distribution)
        .final_value(final_portfolio_value)
        .domain(xmin=-2.0, xmax=2.0, Nx=81)
        .time(T=1.0, Nt=101)
        .coefficients(sigma=0.2, coupling_coefficient=1.0)
        .parameters(
            risk_aversion=1.5,
            transaction_cost=0.02,
            drift=0.05,
            volatility=0.25,
            optimal_position=0.0,
            price_impact=0.15,
        )
        .boundary_conditions(BoundaryConditions(type="dirichlet", left_value=0.0, right_value=0.0))
        .description("Portfolio optimization with price impact", "finance")
        .build()
    )

    logger.info(f"SUCCESS: Created portfolio problem: {problem.components.description}")
    logger.info(f"   Risk aversion: {problem.components.parameters['risk_aversion']}")
    logger.info(f"   Price impact: {problem.components.parameters['price_impact']}")

    return problem


def main():
    """Run all general MFG construction examples."""
    logger.info(" Starting General MFG Construction Demo")

    try:
        # Run all examples
        logger.info("\n" + "=" * 60)
        problem1 = example_1_quadratic_hamiltonian()

        logger.info("\n" + "=" * 60)
        problem2 = example_2_time_dependent_potential()

        logger.info("\n" + "=" * 60)
        problem3 = example_3_nonlocal_coupling()

        logger.info("\n" + "=" * 60)
        problem4 = example_4_portfolio_optimization()

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info(" All examples completed successfully!")

        problems = [problem1, problem2, problem3, problem4]
        for i, prob in enumerate(problems, 1):
            info = prob.get_problem_info()
            logger.info(f"\n Problem {i}: {info['description']}")
            logger.info(f"   Type: {info['problem_type']}")
            logger.info(
                f"   Domain: [{info['domain']['xmin']}, {info['domain']['xmax']}] with {info['domain']['Nx']} points"
            )
            logger.info(f"   Time: [0, {info['time']['T']}] with {info['time']['Nt']} steps")
            logger.info(
                f"   Custom components: H={info['has_custom_hamiltonian']}, V={info['has_custom_potential']}, "
                f"m0={info['has_custom_initial']}, uT={info['has_custom_final']}"
            )
            logger.info(f"   Advanced features: Jacobian={info['has_jacobian']}, Coupling={info['has_coupling']}")

        logger.info("\n🔬 All problems are ready for solving with any MFG_PDE solver!")
        logger.info("   Usage: solver = create_fast_solver(); result = solver.solve(problem)")

    except Exception as e:
        logger.error(f"ERROR: Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
