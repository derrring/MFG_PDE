#!/usr/bin/env python3
"""
El Farol Bar Problem - Mean Field Games Implementation

The El Farol Bar Problem (Santa Fe Bar Problem) is a classic example in game theory
where agents must decide whether to attend a bar based on their prediction of 
attendance by others. This creates a self-referential problem perfect for MFG analysis.

Mathematical Formulation:

State and Population:
- State space: x ∈ [0,1] (tendency to attend bar)  
- Population density: m(t,x) with ∫₀¹ m(t,x) dx = 1
- Expected attendance: A(t) = ∫₀¹ x·m(t,x) dx

Dynamics:
- State dynamics: dx_t = u_t dt + σ dW_t
- Population evolution: ∂m/∂t = -∂/∂x[u(t,x)m(t,x)] + (σ²/2)∂²m/∂x²

Cost Structure:
- Running cost: L(t,x,u,m) = α·max(0, A(t) - C̄)² + ½u² + β(x - x_hist)²
- Terminal cost: Φ(x) = (γ/2)(x - C̄)²

HJB Equation:
- Value function: U(t,x) satisfies
  -∂U/∂t = min_u [L(t,x,u,m) + u·∂U/∂x + (σ²/2)∂²U/∂x²]
- Optimal control: u*(t,x) = -∂U/∂x
- Hamiltonian: H(t,x,p,m) = L(t,x,u*,m) + p·u*

Parameters:
- α > 0: crowd aversion parameter
- C̄ ∈ (0,1): normalized bar capacity
- β > 0: historical memory weight  
- σ > 0: decision volatility
- γ > 0: terminal penalty weight
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
import time

from mfg_pde import MFGProblem, BoundaryConditions
from mfg_pde.factory import create_fast_solver
from mfg_pde.utils.logging import get_logger, configure_research_logging
from mfg_pde.utils.integration import trapezoid


class ElFarolBarMFG(MFGProblem):
    """
    Mean Field Games formulation of the El Farol Bar Problem.
    
    This models the decision-making process of agents choosing whether to attend
    a bar based on their expectation of crowd levels, formulated as an MFG.
    """
    
    def __init__(
        self,
        xmin: float = 0.0,
        xmax: float = 1.0,
        Nx: int = 50,
        T: float = 1.0,
        Nt: int = 50,
        bar_capacity: float = 0.6,  # Normalized capacity (60% attendance optimal)
        crowd_aversion: float = 2.0,  # How much agents dislike crowding
        decision_volatility: float = 0.1,  # Noise in decision-making
        memory_weight: float = 0.5,  # Weight on historical information
        **kwargs
    ):
        """
        Initialize El Farol Bar MFG Problem.
        
        Parameters:
        -----------
        bar_capacity : float
            Normalized bar capacity (0.6 = 60% optimal attendance)
        crowd_aversion : float  
            Coefficient for crowd aversion in utility function
        decision_volatility : float
            Diffusion coefficient for decision-making uncertainty
        memory_weight : float
            Weight given to historical attendance patterns
        """
        
        self.bar_capacity = bar_capacity
        self.crowd_aversion = crowd_aversion
        self.decision_volatility = decision_volatility
        self.memory_weight = memory_weight
        
        # Initialize base MFG problem
        super().__init__(
            xmin=xmin, xmax=xmax, Nx=Nx,
            T=T, Nt=Nt,
            sigma=decision_volatility,  # Decision uncertainty
            **kwargs
        )
        
        # Store grid for use in H and dH_dm methods
        self.x_grid = np.linspace(self.xmin, self.xmax, self.Nx)
        self.current_m_density = None  # Will be set during solving
        
        self.logger = get_logger(__name__)
        self.logger.info(f"Initialized El Farol Bar MFG with capacity={bar_capacity}, "
                        f"aversion={crowd_aversion}, volatility={decision_volatility}")
    
    def running_cost(self, t: float, x: np.ndarray, u: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        Running cost function for El Farol Bar Problem.
        
        The cost represents the disutility from:
        1. Crowding effects (quadratic in total attendance)
        2. Decision effort (quadratic in control)
        3. Deviation from historical patterns
        
        Parameters:
        -----------
        t : float
            Current time
        x : np.ndarray
            State (decision tendency)
        u : np.ndarray  
            Control (rate of deciding to go)
        m : np.ndarray
            Population density
            
        Returns:
        --------
        np.ndarray
            Running cost at each point
        """
        
        # Expected total attendance (integral of m weighted by going probability)
        # In state x, probability of going is x, so expected attendance is ∫ x*m(x) dx
        x_grid = np.linspace(self.xmin, self.xmax, len(m))
        expected_attendance = trapezoid(x_grid * m, x_grid)
        
        # Crowding cost: quadratic penalty when attendance exceeds capacity
        crowding_penalty = self.crowd_aversion * np.maximum(0, expected_attendance - self.bar_capacity)**2
        
        # Individual decision cost: effort of changing decision state
        decision_cost = 0.5 * u**2
        
        # Memory-based cost: deviation from historical average
        # (simplified as deviation from capacity)
        historical_target = self.bar_capacity
        memory_cost = self.memory_weight * (x - historical_target)**2
        
        # Total cost
        total_cost = crowding_penalty + decision_cost + memory_cost
        
        return total_cost
    
    def terminal_cost(self, x: np.ndarray) -> np.ndarray:
        """
        Terminal cost function.
        
        Penalizes final states that deviate from optimal attendance patterns.
        
        Parameters:
        -----------
        x : np.ndarray
            Final state
            
        Returns:
        --------
        np.ndarray
            Terminal cost
        """
        # Penalty for being far from optimal attendance probability
        return 0.5 * (x - self.bar_capacity)**2
    
    def initial_condition_m(self, x: np.ndarray) -> np.ndarray:
        """
        Initial population distribution.
        
        Start with agents uniformly distributed over decision tendencies,
        representing initial uncertainty about others' behavior.
        
        Parameters:
        -----------
        x : np.ndarray
            State space grid
            
        Returns:
        --------
        np.ndarray
            Initial density
        """
        # Uniform distribution initially (complete uncertainty)
        uniform_density = np.ones_like(x) / (self.xmax - self.xmin)
        
        # Add small perturbation for numerical stability
        perturbation = 0.1 * np.sin(2 * np.pi * x / (self.xmax - self.xmin))
        
        density = uniform_density + perturbation
        
        # Ensure non-negative and normalized
        density = np.maximum(density, 0.01)
        density = density / trapezoid(density, x)
        
        return density
    
    def boundary_conditions_m(self) -> BoundaryConditions:
        """Boundary conditions for population density (no-flux)."""
        return BoundaryConditions(type="no_flux")
    
    def boundary_conditions_u(self) -> BoundaryConditions:
        """Boundary conditions for value function (Dirichlet)."""
        return BoundaryConditions(type="dirichlet", value=0.0)
    
    def H(
        self,
        x_idx: int,
        m_at_x: float,
        p_values: Dict[str, float],
        t_idx: Optional[int] = None,
    ) -> float:
        """
        Hamiltonian function for El Farol Bar Problem.
        
        Parameters:
        -----------
        x_idx : int
            Spatial grid index
        m_at_x : float
            Population density at this spatial point
        p_values : Dict[str, float]
            Dictionary containing spatial derivatives of value function
        t_idx : Optional[int]
            Time index (optional)
            
        Returns:
        --------
        float
            Hamiltonian value at this point
        """
        
        # Get spatial position
        x_val = self.x_grid[x_idx]
        
        # Get derivative of value function (costate)
        p_x = p_values.get('p_x', 0.0)
        
        # Optimal control: u* = -p_x (minimizing quadratic cost + linear coupling)
        u_optimal = -p_x
        
        # Expected attendance across all agents
        if self.current_m_density is not None:
            total_attendance = trapezoid(self.x_grid * self.current_m_density, self.x_grid)
        else:
            # Fallback: assume uniform distribution
            total_attendance = 0.5  # Middle of [0,1] range
        
        # Running cost components
        # 1. Crowding penalty
        crowding_cost = self.crowd_aversion * max(0, total_attendance - self.bar_capacity)**2
        
        # 2. Decision effort cost
        decision_cost = 0.5 * u_optimal**2
        
        # 3. Memory/historical cost
        memory_cost = self.memory_weight * (x_val - self.bar_capacity)**2
        
        # Total Hamiltonian
        hamiltonian = crowding_cost + decision_cost + memory_cost + p_x * u_optimal
        
        return hamiltonian
    
    def dH_dm(
        self,
        x_idx: int,
        m_at_x: float,
        p_values: Dict[str, float],
        t_idx: Optional[int] = None,
    ) -> float:
        """
        Derivative of Hamiltonian with respect to population density.
        
        Parameters:
        -----------
        x_idx : int
            Spatial grid index
        m_at_x : float
            Population density at this spatial point
        p_values : Dict[str, float]
            Dictionary containing spatial derivatives of value function
        t_idx : Optional[int]
            Time index (optional)
            
        Returns:
        --------
        float
            Derivative dH/dm at this point
        """
        
        # Get spatial position
        x_val = self.x_grid[x_idx]
        
        # Expected total attendance
        if self.current_m_density is not None:
            total_attendance = trapezoid(self.x_grid * self.current_m_density, self.x_grid)
        else:
            # Fallback: assume uniform distribution
            total_attendance = 0.5  # Middle of [0,1] range
        
        # Derivative of crowding cost w.r.t. density
        if total_attendance > self.bar_capacity:
            # d/dm[crowding_cost] = crowd_aversion * 2 * (attendance - capacity) * x_val
            crowd_derivative = 2 * self.crowd_aversion * (total_attendance - self.bar_capacity) * x_val
        else:
            crowd_derivative = 0.0
        
        return crowd_derivative
    
    def analyze_equilibrium(self, U: np.ndarray, M: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the equilibrium solution for economic insights.
        
        Parameters:
        -----------
        U : np.ndarray
            Value function solution
        M : np.ndarray
            Population density solution
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results
        """
        
        x_grid = np.linspace(self.xmin, self.xmax, self.Nx)
        t_grid = np.linspace(0, self.T, self.Nt)
        
        analysis = {}
        
        # Time-evolved attendance rates
        attendance_rates = []
        for t_idx in range(self.Nt):
            # Expected attendance at time t
            attendance = trapezoid(x_grid * M[t_idx, :], x_grid)
            attendance_rates.append(attendance)
        
        analysis['attendance_evolution'] = attendance_rates
        analysis['final_attendance'] = attendance_rates[-1]
        analysis['capacity_utilization'] = attendance_rates[-1] / self.bar_capacity
        
        # Convergence to equilibrium
        attendance_variance = np.var(attendance_rates[-10:])  # Last 10 time steps
        analysis['equilibrium_stability'] = attendance_variance
        analysis['converged'] = attendance_variance < 1e-4
        
        # Spatial distribution analysis
        final_density = M[-1, :]
        peak_location = x_grid[np.argmax(final_density)]
        analysis['peak_decision_tendency'] = peak_location
        analysis['density_spread'] = np.std(final_density)
        
        # Economic efficiency
        optimal_attendance = self.bar_capacity
        efficiency = 1.0 - abs(attendance_rates[-1] - optimal_attendance) / optimal_attendance
        analysis['efficiency'] = efficiency
        
        # Herding behavior indicator
        final_mass_center = trapezoid(x_grid * final_density, x_grid)
        analysis['herding_indicator'] = abs(final_mass_center - 0.5)  # Deviation from uniform
        
        self.logger.info(f"Equilibrium Analysis: Attendance={attendance_rates[-1]:.3f}, "
                        f"Efficiency={efficiency:.3f}, Converged={analysis['converged']}")
        
        return analysis


def solve_el_farol_bar_problem(
    bar_capacity: float = 0.6,
    crowd_aversion: float = 2.0,
    solver_type: str = "fixed_point",
    visualize: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Solve the El Farol Bar Problem using Mean Field Games.
    
    Parameters:
    -----------
    bar_capacity : float
        Normalized bar capacity (optimal attendance level)
    crowd_aversion : float
        How much agents dislike crowding
    solver_type : str
        MFG solver type ("fixed_point", "adaptive_particle")
    visualize : bool
        Whether to create visualizations
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
        Value function, population density, and analysis results
    """
    
    logger = get_logger(__name__)
    logger.info(f"🍺 Solving El Farol Bar Problem with capacity={bar_capacity}")
    
    # Create the MFG problem
    problem = ElFarolBarMFG(
        xmin=0.0, xmax=1.0, Nx=50,
        T=1.0, Nt=50,
        bar_capacity=bar_capacity,
        crowd_aversion=crowd_aversion,
        decision_volatility=0.1,
        memory_weight=0.5
    )
    
    # Create solver
    logger.info(f"Creating {solver_type} solver...")
    solver = create_fast_solver(problem, solver_type=solver_type)
    
    # Solve the problem
    start_time = time.time()
    result = solver.solve()
    solve_time = time.time() - start_time
    
    U, M = result.U, result.M
    
    logger.info(f"✅ Solution completed in {solve_time:.2f} seconds")
    logger.info(f"Convergence: {result.convergence_achieved}, "
                f"Iterations: {getattr(result, 'iterations', 'N/A')}")
    
    # Analyze the solution
    analysis = problem.analyze_equilibrium(U, M)
    analysis['solve_time'] = solve_time
    analysis['convergence_achieved'] = result.convergence_achieved
    
    # Create visualizations
    if visualize:
        create_el_farol_visualizations(problem, U, M, analysis)
    
    return U, M, analysis


def create_el_farol_visualizations(
    problem: ElFarolBarMFG,
    U: np.ndarray,
    M: np.ndarray,
    analysis: Dict[str, Any]
) -> None:
    """
    Create comprehensive visualizations for El Farol Bar Problem.
    
    Parameters:
    -----------
    problem : ElFarolBarMFG
        The MFG problem instance
    U : np.ndarray
        Value function solution
    M : np.ndarray
        Population density solution
    analysis : Dict[str, Any]
        Analysis results
    """
    
    logger = get_logger(__name__)
    logger.info("📊 Creating El Farol Bar visualizations...")
    
    # Setup grids
    x_grid = np.linspace(problem.xmin, problem.xmax, problem.Nx)
    t_grid = np.linspace(0, problem.T, problem.Nt)
    X, T = np.meshgrid(x_grid, t_grid)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('El Farol Bar Problem - Mean Field Games Analysis', fontsize=16, fontweight='bold')
    
    # 1. Population Density Evolution
    ax1 = plt.subplot(2, 3, 1)
    contour = ax1.contourf(X, T, M, levels=20, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Decision Tendency x')
    ax1.set_ylabel('Time t')
    ax1.set_title('Population Density m(t,x)')
    plt.colorbar(contour, ax=ax1, label='Density')
    
    # Add capacity line
    ax1.axvline(x=problem.bar_capacity, color='red', linestyle='--', 
                label=f'Capacity ({problem.bar_capacity})')
    ax1.legend()
    
    # 2. Value Function Evolution  
    ax2 = plt.subplot(2, 3, 2)
    contour2 = ax2.contourf(X, T, U, levels=20, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('Decision Tendency x')
    ax2.set_ylabel('Time t')
    ax2.set_title('Value Function U(t,x)')
    plt.colorbar(contour2, ax=ax2, label='Value')
    
    # 3. Attendance Evolution
    ax3 = plt.subplot(2, 3, 3)
    attendance_rates = analysis['attendance_evolution']
    ax3.plot(t_grid, attendance_rates, 'b-', linewidth=2, label='Actual Attendance')
    ax3.axhline(y=problem.bar_capacity, color='red', linestyle='--', 
                label=f'Optimal Capacity ({problem.bar_capacity})')
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Expected Attendance')
    ax3.set_title('Attendance Rate Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add efficiency annotation
    efficiency = analysis['efficiency']
    ax3.text(0.7 * problem.T, 0.9 * max(attendance_rates), 
             f'Efficiency: {efficiency:.1%}', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 4. Final Population Distribution
    ax4 = plt.subplot(2, 3, 4)
    final_density = M[-1, :]
    ax4.plot(x_grid, final_density, 'g-', linewidth=2, label='Final Density')
    ax4.axvline(x=problem.bar_capacity, color='red', linestyle='--', 
                label=f'Capacity ({problem.bar_capacity})')
    ax4.axvline(x=analysis['peak_decision_tendency'], color='orange', linestyle=':', 
                label=f'Peak at {analysis["peak_decision_tendency"]:.2f}')
    ax4.set_xlabel('Decision Tendency x')
    ax4.set_ylabel('Density')
    ax4.set_title('Final Population Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Phase Portrait (Density over Time)
    ax5 = plt.subplot(2, 3, 5)
    # Show density evolution at key decision points
    key_points = [0.2, 0.5, 0.8]  # Low, medium, high tendency to go
    for point in key_points:
        idx = np.argmin(np.abs(x_grid - point))
        density_evolution = M[:, idx]
        ax5.plot(t_grid, density_evolution, label=f'x = {point}')
    
    ax5.set_xlabel('Time t')
    ax5.set_ylabel('Density')
    ax5.set_title('Density Evolution at Key Points')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Economic Analysis Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary statistics
    summary_text = f"""
El Farol Bar Problem Analysis

Problem Parameters:
• Bar Capacity: {problem.bar_capacity:.2f}
• Crowd Aversion: {problem.crowd_aversion:.2f}
• Decision Volatility: {problem.decision_volatility:.2f}

Equilibrium Results:
• Final Attendance: {analysis['final_attendance']:.3f}
• Capacity Utilization: {analysis['capacity_utilization']:.1%}
• Economic Efficiency: {analysis['efficiency']:.1%}
• Equilibrium Stability: {analysis['equilibrium_stability']:.2e}

Behavioral Insights:
• Peak Decision Tendency: {analysis['peak_decision_tendency']:.3f}
• Herding Indicator: {analysis['herding_indicator']:.3f}
• Converged: {'Yes' if analysis['converged'] else 'No'}

Solution Quality:
• Solve Time: {analysis['solve_time']:.2f}s
• Convergence: {'Yes' if analysis['convergence_achieved'] else 'No'}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the figure
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"el_farol_bar_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"📁 Visualization saved as {filename}")
    
    plt.show()


def compare_crowd_aversion_scenarios() -> None:
    """
    Compare different crowd aversion scenarios to understand behavioral impact.
    """
    
    logger = get_logger(__name__)
    logger.info("🔬 Comparing different crowd aversion scenarios...")
    
    crowd_aversions = [0.5, 1.0, 2.0, 4.0]  # Low to high crowd aversion
    results = {}
    
    plt.figure(figsize=(15, 10))
    
    for i, aversion in enumerate(crowd_aversions):
        logger.info(f"Solving for crowd aversion = {aversion}")
        
        U, M, analysis = solve_el_farol_bar_problem(
            crowd_aversion=aversion,
            visualize=False
        )
        
        results[aversion] = analysis
        
        # Plot attendance evolution
        plt.subplot(2, 2, 1)
        t_grid = np.linspace(0, 1.0, 50)
        plt.plot(t_grid, analysis['attendance_evolution'], 
                label=f'Aversion = {aversion}', linewidth=2)
    
    # Finish attendance evolution plot
    plt.subplot(2, 2, 1)
    plt.axhline(y=0.6, color='red', linestyle='--', label='Optimal Capacity')
    plt.xlabel('Time')
    plt.ylabel('Attendance Rate')
    plt.title('Attendance Evolution by Crowd Aversion')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Efficiency comparison
    plt.subplot(2, 2, 2)
    efficiencies = [results[a]['efficiency'] for a in crowd_aversions]
    plt.bar(range(len(crowd_aversions)), efficiencies, 
            color=['lightblue', 'lightgreen', 'orange', 'lightcoral'])
    plt.xlabel('Crowd Aversion Level')
    plt.ylabel('Economic Efficiency')
    plt.title('Efficiency vs Crowd Aversion')
    plt.xticks(range(len(crowd_aversions)), [f'{a}' for a in crowd_aversions])
    plt.grid(True, alpha=0.3)
    
    # Final attendance comparison
    plt.subplot(2, 2, 3)
    final_attendances = [results[a]['final_attendance'] for a in crowd_aversions]
    plt.plot(crowd_aversions, final_attendances, 'o-', markersize=8, linewidth=2)
    plt.axhline(y=0.6, color='red', linestyle='--', label='Optimal')
    plt.xlabel('Crowd Aversion')
    plt.ylabel('Final Attendance')
    plt.title('Final Attendance vs Crowd Aversion')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Convergence comparison
    plt.subplot(2, 2, 4)
    stabilities = [results[a]['equilibrium_stability'] for a in crowd_aversions]
    plt.semilogy(crowd_aversions, stabilities, 's-', markersize=8, linewidth=2)
    plt.xlabel('Crowd Aversion')
    plt.ylabel('Equilibrium Stability (log scale)')
    plt.title('Stability vs Crowd Aversion')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"el_farol_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"📁 Comparison saved as {filename}")
    
    plt.show()
    
    # Print summary
    logger.info("📊 Crowd Aversion Analysis Summary:")
    for aversion in crowd_aversions:
        analysis = results[aversion]
        logger.info(f"  Aversion {aversion}: Attendance={analysis['final_attendance']:.3f}, "
                   f"Efficiency={analysis['efficiency']:.3f}")


def main():
    """
    Main function to demonstrate El Farol Bar Problem analysis.
    """
    
    # Configure logging for research session
    configure_research_logging("el_farol_bar_analysis", level="INFO")
    logger = get_logger(__name__)
    
    logger.info("🍺 Starting El Farol Bar Problem Analysis with Mean Field Games")
    logger.info("=" * 70)
    
    try:
        # Basic problem solution
        logger.info("1. Solving basic El Farol Bar Problem...")
        U, M, analysis = solve_el_farol_bar_problem(
            bar_capacity=0.6,
            crowd_aversion=2.0,
            solver_type="fixed_point",
            visualize=True
        )
        
        logger.info("2. Running crowd aversion comparison...")
        compare_crowd_aversion_scenarios()
        
        logger.info("✅ El Farol Bar analysis completed successfully!")
        
        # Print key insights
        logger.info("\n📊 Key Economic Insights:")
        logger.info(f"  • Final attendance rate: {analysis['final_attendance']:.1%}")
        logger.info(f"  • Economic efficiency: {analysis['efficiency']:.1%}")
        logger.info(f"  • System converged: {'Yes' if analysis['converged'] else 'No'}")
        logger.info(f"  • Herding behavior: {analysis['herding_indicator']:.3f}")
        
    except Exception as e:
        logger.error(f"❌ Error in El Farol Bar analysis: {e}")
        raise


if __name__ == "__main__":
    main()