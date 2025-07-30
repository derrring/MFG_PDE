"""
Replacing Specific Models with GeneralMFGProblem.

This example shows how the GeneralMFGProblem can replace all specific model
implementations (CrowdDynamicsMFG, FinancialMarketMFG, etc.) with custom
Hamiltonian definitions. This approach is more flexible and mathematically precise.
"""

import numpy as np
import sys
from pathlib import Path

# Add MFG_PDE to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mfg_pde import MFGProblem, MFGProblemBuilder
from mfg_pde.geometry import BoundaryConditions
from mfg_pde.utils.logging import get_logger, configure_research_logging
from mfg_pde.utils.aux_func import npart, ppart
from mfg_pde.core.mfg_problem import VALUE_BEFORE_SQUARE_LIMIT

# Configure logging
configure_research_logging("general_models_demo", level="INFO")
logger = get_logger(__name__)


def crowd_dynamics_example():
    """Example: Crowd dynamics with congestion (replaces CrowdDynamicsMFG)."""
    logger.info("🚶 Building crowd dynamics model with GeneralMFGProblem")
    
    def crowd_hamiltonian(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        """Hamiltonian for crowd dynamics with congestion effects."""
        
        p_forward = p_values.get("forward")
        p_backward = p_values.get("backward")
        
        if p_forward is None or p_backward is None:
            return np.nan
        if (np.isnan(p_forward) or np.isinf(p_forward) or
            np.isnan(p_backward) or np.isinf(p_backward) or
            np.isnan(m_at_x) or np.isinf(m_at_x)):
            return np.nan
        
        # Use standard processing
        npart_val_fwd = npart(p_forward)
        ppart_val_bwd = ppart(p_backward)
        
        if (abs(npart_val_fwd) > VALUE_BEFORE_SQUARE_LIMIT or 
            abs(ppart_val_bwd) > VALUE_BEFORE_SQUARE_LIMIT):
            return np.nan
        
        try:
            # Control cost with panic factor
            panic_factor = problem.components.parameters.get("panic_factor", 0.0)
            effective_coefCT = problem.coefCT * (1.0 - 0.5 * panic_factor)
            kinetic_energy = 0.5 * effective_coefCT * (npart_val_fwd**2 + ppart_val_bwd**2)
            
            # Potential (attraction to exits, repulsion from obstacles)
            potential = problem.f_potential[x_idx]
            
            # Congestion penalty (higher power = more avoidance)
            congestion_power = problem.components.parameters.get("congestion_power", 2.0)
            congestion_penalty = m_at_x**congestion_power
            
            return kinetic_energy - potential - congestion_penalty
            
        except (OverflowError, IndexError):
            return np.nan
    
    def crowd_hamiltonian_dm(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        """dH/dm for crowd dynamics."""
        if np.isnan(m_at_x) or np.isinf(m_at_x):
            return np.nan
        
        try:
            congestion_power = problem.components.parameters.get("congestion_power", 2.0)
            return congestion_power * (m_at_x**(congestion_power - 1))
        except OverflowError:
            return np.nan
    
    def exit_attraction_potential(x):
        """Potential with attraction to exit at x=1."""
        exit_position = 0.9
        attraction_strength = 2.0
        distance_to_exit = abs(x - exit_position)
        return -attraction_strength * np.exp(-distance_to_exit / 0.2)
    
    def entrance_initial_density(x):
        """Initial crowd density concentrated at entrance."""
        entrance_center = 0.1
        entrance_width = 0.05
        distance_to_entrance = abs(x - entrance_center)
        if distance_to_entrance < entrance_width:
            return np.exp(-0.5 * (distance_to_entrance / (entrance_width/3))**2)
        return 0.01
    
    # Build crowd dynamics problem
    problem = (MFGProblemBuilder()
               .hamiltonian(crowd_hamiltonian, crowd_hamiltonian_dm)
               .potential(exit_attraction_potential)
               .initial_density(entrance_initial_density)
               .final_value(lambda x: 0.0)  # No final cost
               .domain(xmin=0.0, xmax=1.0, Nx=51)
               .time(T=2.0, Nt=51)
               .coefficients(sigma=0.3, coefCT=1.0)
               .parameters(
                   congestion_power=2.5,
                   panic_factor=0.2,
                   target_velocity=1.0
               )
               .boundary_conditions(BoundaryConditions(type="neumann", left_value=0.0, right_value=0.0))
               .description("Crowd dynamics with congestion", "crowd_dynamics")
               .build())
    
    logger.info(f"✅ Built crowd dynamics: {problem.components.description}")
    return problem


def financial_market_example():
    """Example: Financial market model (replaces FinancialMarketMFG)."""
    logger.info("💰 Building financial market model with GeneralMFGProblem")
    
    def financial_hamiltonian(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        """Hamiltonian for financial market with price impact."""
        
        p_forward = p_values.get("forward")
        p_backward = p_values.get("backward")
        
        if p_forward is None or p_backward is None:
            return np.nan
        if (np.isnan(p_forward) or np.isinf(p_forward) or
            np.isnan(p_backward) or np.isinf(p_backward) or
            np.isnan(m_at_x) or np.isinf(m_at_x)):
            return np.nan
        
        npart_val_fwd = npart(p_forward)
        ppart_val_bwd = ppart(p_backward)
        
        if (abs(npart_val_fwd) > VALUE_BEFORE_SQUARE_LIMIT or 
            abs(ppart_val_bwd) > VALUE_BEFORE_SQUARE_LIMIT):
            return np.nan
        
        try:
            # Risk aversion and transaction costs
            risk_aversion = problem.components.parameters.get("risk_aversion", 1.0)
            transaction_cost = problem.components.parameters.get("transaction_cost", 0.01)
            control_cost = (problem.coefCT + transaction_cost) * risk_aversion
            
            kinetic_energy = 0.5 * control_cost * (npart_val_fwd**2 + ppart_val_bwd**2)
            
            # Market drift
            drift = problem.components.parameters.get("drift", 0.05)
            drift_term = drift * x_position
            
            # Price impact (market impact of collective trading)
            price_impact = problem.components.parameters.get("price_impact", 0.1)
            impact_cost = price_impact * m_at_x**2
            
            # Market volatility affects optimal position
            volatility = problem.components.parameters.get("volatility", 0.2)
            optimal_position = problem.components.parameters.get("optimal_position", 0.0)
            position_cost = 0.5 * volatility * (x_position - optimal_position)**2
            
            return kinetic_energy - drift_term - position_cost - impact_cost
            
        except (OverflowError, IndexError):
            return np.nan
    
    def financial_hamiltonian_dm(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        """dH/dm for financial market (price impact)."""
        if np.isnan(m_at_x) or np.isinf(m_at_x):
            return np.nan
        
        price_impact = problem.components.parameters.get("price_impact", 0.1)
        return -2.0 * price_impact * m_at_x
    
    def initial_portfolio_distribution(x):
        """Initial portfolio distribution around neutral position."""
        return np.exp(-5 * x**2)
    
    def liquidation_cost(x):
        """Terminal liquidation cost."""
        liquidation_penalty = 0.02
        return -liquidation_penalty * abs(x)
    
    # Build financial market problem
    problem = (MFGProblemBuilder()
               .hamiltonian(financial_hamiltonian, financial_hamiltonian_dm)
               .potential(lambda x: 0.0)  # No external potential
               .initial_density(initial_portfolio_distribution)
               .final_value(liquidation_cost)
               .domain(xmin=-2.0, xmax=2.0, Nx=81)
               .time(T=1.0, Nt=101)
               .coefficients(sigma=0.25, coefCT=1.0)
               .parameters(
                   risk_aversion=1.5,
                   transaction_cost=0.02,
                   drift=0.05,
                   volatility=0.25,
                   optimal_position=0.0,
                   price_impact=0.15
               )
               .boundary_conditions(BoundaryConditions(type="dirichlet", left_value=0.0, right_value=0.0))
               .description("Financial market with price impact", "finance")
               .build())
    
    logger.info(f"✅ Built financial market: {problem.components.description}")
    return problem


def epidemic_spread_example():
    """Example: Epidemic spread model (replaces EpidemicSpreadMFG)."""
    logger.info("🦠 Building epidemic spread model with GeneralMFGProblem")
    
    def epidemic_hamiltonian(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        """Hamiltonian for epidemic spread with social distancing."""
        
        p_forward = p_values.get("forward")
        p_backward = p_values.get("backward")
        
        if p_forward is None or p_backward is None:
            return np.nan
        if (np.isnan(p_forward) or np.isinf(p_forward) or
            np.isnan(p_backward) or np.isinf(p_backward) or
            np.isnan(m_at_x) or np.isinf(m_at_x)):
            return np.nan
        
        npart_val_fwd = npart(p_forward)
        ppart_val_bwd = ppart(p_backward)
        
        if (abs(npart_val_fwd) > VALUE_BEFORE_SQUARE_LIMIT or 
            abs(ppart_val_bwd) > VALUE_BEFORE_SQUARE_LIMIT):
            return np.nan
        
        try:
            # Movement cost (higher during lockdown)
            lockdown_factor = problem.components.parameters.get("lockdown_factor", 1.0)
            effective_coefCT = problem.coefCT * lockdown_factor
            kinetic_energy = 0.5 * effective_coefCT * (npart_val_fwd**2 + ppart_val_bwd**2)
            
            # Essential services attraction
            potential = problem.f_potential[x_idx]
            
            # Infection risk proportional to local density
            infection_rate = problem.components.parameters.get("infection_rate", 0.1)
            social_distancing = problem.components.parameters.get("social_distancing", 1.0)
            infection_risk = infection_rate * m_at_x / social_distancing
            
            # Economic cost of staying home
            economic_cost = problem.components.parameters.get("economic_cost", 0.05)
            home_position = problem.components.parameters.get("home_position", 0.5)
            distance_from_home = abs(x_position - home_position)
            economic_penalty = economic_cost * distance_from_home
            
            return kinetic_energy - potential - infection_risk - economic_penalty
            
        except (OverflowError, IndexError):
            return np.nan
    
    def epidemic_hamiltonian_dm(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        """dH/dm for epidemic spread (infection risk)."""
        if np.isnan(m_at_x) or np.isinf(m_at_x):
            return np.nan
        
        infection_rate = problem.components.parameters.get("infection_rate", 0.1)
        social_distancing = problem.components.parameters.get("social_distancing", 1.0)
        return -infection_rate / social_distancing
    
    def essential_services_potential(x):
        """Potential representing essential services (grocery, pharmacy, etc.)."""
        # Multiple essential service locations
        services = [0.2, 0.8]  # Locations of essential services
        service_strength = 1.0
        
        potential = 0.0
        for service_pos in services:
            distance = abs(x - service_pos)
            potential -= service_strength * np.exp(-distance / 0.1)
        
        return potential
    
    def home_initial_distribution(x):
        """Initial distribution concentrated at home."""
        home_center = 0.5
        return np.exp(-20 * (x - home_center)**2)
    
    # Build epidemic spread problem
    problem = (MFGProblemBuilder()
               .hamiltonian(epidemic_hamiltonian, epidemic_hamiltonian_dm)
               .potential(essential_services_potential)
               .initial_density(home_initial_distribution)
               .final_value(lambda x: 0.0)  # No final cost
               .domain(xmin=0.0, xmax=1.0, Nx=71)
               .time(T=3.0, Nt=71)
               .coefficients(sigma=0.2, coefCT=0.5)
               .parameters(
                   infection_rate=0.15,
                   social_distancing=2.0,
                   lockdown_factor=2.0,
                   economic_cost=0.1,
                   home_position=0.5
               )
               .boundary_conditions(BoundaryConditions(type="periodic"))
               .description("Epidemic spread with social distancing", "epidemic")
               .build())
    
    logger.info(f"✅ Built epidemic model: {problem.components.description}")
    return problem


def main():
    """Demonstrate replacing specific models with GeneralMFGProblem."""
    logger.info("🚀 Demonstrating GeneralMFGProblem replacing specific models")
    
    # Build all example problems
    logger.info("\n" + "="*60)
    crowd_problem = crowd_dynamics_example()
    
    logger.info("\n" + "="*60)
    financial_problem = financial_market_example()
    
    logger.info("\n" + "="*60)
    epidemic_problem = epidemic_spread_example()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("🎉 All specific models successfully replaced with GeneralMFGProblem!")
    
    problems = [crowd_problem, financial_problem, epidemic_problem]
    
    for i, prob in enumerate(problems, 1):
        info = prob.get_problem_info()
        logger.info(f"\n📋 Problem {i}: {info['description']}")
        logger.info(f"   Type: {info['problem_type']}")
        logger.info(f"   Domain: [{info['domain']['xmin']}, {info['domain']['xmax']}] with {info['domain']['Nx']} points")
        logger.info(f"   Time: [0, {info['time']['T']}] with {info['time']['Nt']} steps")
        logger.info(f"   Parameters: {list(info['parameters'].keys())}")
    
    logger.info(f"\n✨ Benefits of GeneralMFGProblem approach:")
    logger.info("   • Complete mathematical control over Hamiltonian")
    logger.info("   • No hidden assumptions or predefined behaviors")
    logger.info("   • Easy to modify and extend problem formulations")
    logger.info("   • Single unified interface for all MFG problems")
    logger.info("   • Research-grade flexibility and precision")


if __name__ == "__main__":
    main()