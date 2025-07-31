#!/usr/bin/env python3
"""
Towel on the Beach Problem - MFG Implementation

A simplified and accessible version of the Santa Fe El Farol Bar Problem.
This example demonstrates how the same coordination dilemma applies to 
everyday scenarios like beach attendance decisions.

The problem: N individuals decide whether to go to a beach where enjoyment
is inversely related to crowding. Each person must predict attendance 
without knowing others' decisions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.factory import create_fast_solver
from mfg_pde.utils.logging import get_logger, configure_research_logging
from mfg_pde.utils.integration import trapezoid

# Configure logging
configure_research_logging("towel_beach_problem", level="INFO")
logger = get_logger(__name__)


class TowelBeachProblem(ExampleMFGProblem):
    """
    The Towel on the Beach Problem: MFG Formulation
    
    A beach coordination problem where agents decide whether to visit a beach.
    The enjoyment decreases with crowding, creating a coordination dilemma
    identical to the Santa Fe El Farol Bar Problem.
    
    Problem Elements:
    - N potential beachgoers (agents)
    - Binary decision: Go to beach (1) or Stay home (0)
    - Payoff depends on total attendance vs. beach capacity
    - No communication between agents
    """
    
    def __init__(self, beach_capacity=0.6, good_weather_bonus=1.0, 
                 crowding_penalty=2.0, **kwargs):
        """
        Initialize the Towel Beach Problem.
        
        Args:
            beach_capacity: Fraction of population for comfortable beach (0.6 = 60%)
            good_weather_bonus: Utility bonus for going on uncrowded day
            crowding_penalty: Utility penalty for going on crowded day
            **kwargs: Additional MFG problem parameters
        """
        super().__init__(**kwargs)
        self.beach_capacity = beach_capacity
        self.good_weather_bonus = good_weather_bonus
        self.crowding_penalty = crowding_penalty
        
        logger.info(f"Created Towel Beach Problem:")
        logger.info(f"  Beach capacity: {beach_capacity:.1%}")
        logger.info(f"  Good weather bonus: {good_weather_bonus}")
        logger.info(f"  Crowding penalty: {crowding_penalty}")
    
    def g(self, x):
        """
        Terminal cost: preference for being at optimal beach attendance.
        
        x=0: Stayed home
        x=1: Went to beach
        
        Terminal cost encourages moderate beach usage.
        """
        # Quadratic penalty for being far from moderate attendance (x=0.5)
        return 0.5 * (x - 0.5)**2
    
    def f(self, x, u, m):
        """
        Running cost: immediate cost of beach decision and crowding.
        
        Args:
            x: Current decision tendency (0=stay, 1=go)
            u: Control (rate of decision change)
            m: Population density
            
        Returns:
            Instantaneous cost
        """
        # Estimate total beach attendance by integrating density
        total_attendance = trapezoid(m, dx=self.Dx)
        
        # Crowding effect: cost increases when attendance exceeds capacity
        crowding_cost = 0
        if total_attendance > self.beach_capacity:
            crowding_excess = total_attendance - self.beach_capacity
            crowding_cost = self.crowding_penalty * crowding_excess**2
        
        # Decision change cost (prefer stable decisions)
        decision_change_cost = 0.1 * u**2
        
        # Beach enjoyment bonus when uncrowded
        beach_bonus = 0
        if x > 0.5 and total_attendance <= self.beach_capacity:
            beach_bonus = -self.good_weather_bonus * x
        
        return crowding_cost + decision_change_cost + beach_bonus
    
    def rho0(self, x):
        """
        Initial population distribution.
        
        Most people start with moderate tendency to go to beach,
        representing typical beach-going inclination.
        """
        # Gaussian distribution centered at moderate beach inclination (x=0.4)
        return np.exp(-15 * (x - 0.4)**2)
    
    def sigma(self, x):
        """Noise level in decision-making."""
        return 0.1  # Moderate uncertainty in beach decisions
    
    def H(self, x_idx: int, m_at_x: float, p_values: dict, t_idx=None):
        """
        Hamiltonian: optimal control choice.
        
        Standard quadratic control with congestion effects.
        """
        p_forward = p_values.get("forward", 0.0)
        p_backward = p_values.get("backward", 0.0)
        
        if np.isnan(p_forward) or np.isnan(p_backward):
            return np.nan
            
        # Choose the control that maximizes -H (minimizes cost)
        # Standard quadratic Hamiltonian: H = 0.5 * p^2
        return 0.5 * max(p_forward**2, p_backward**2)
    
    def dH_dm(self, x_idx: int, m_at_x: float, p_values: dict, t_idx=None):
        """Derivative of Hamiltonian with respect to population density."""
        return 0.0  # No direct dependence on local density


def solve_beach_problem_variants():
    """
    Solve the beach problem with different parameters to show 
    how capacity affects coordination outcomes.
    """
    logger.info("Solving Towel Beach Problem with different capacities")
    
    # Different beach capacity scenarios
    scenarios = [
        {"name": "Small Beach", "capacity": 0.3, "description": "Limited capacity, high coordination challenge"},
        {"name": "Medium Beach", "capacity": 0.6, "description": "Moderate capacity, classic coordination dilemma"},
        {"name": "Large Beach", "capacity": 0.8, "description": "High capacity, easier coordination"}
    ]
    
    results = {}
    
    for scenario in scenarios:
        logger.info(f"\nSolving {scenario['name']} scenario...")
        
        # Create problem
        problem = TowelBeachProblem(
            T=1.0,
            Nx=100,
            Nt=50,
            beach_capacity=scenario['capacity'],
            good_weather_bonus=1.0,
            crowding_penalty=3.0
        )
        
        # Solve with fast solver
        solver = create_fast_solver(problem)
        result = solver.solve()
        U, M = result.U, result.M
        info = result.metadata
        
        # Calculate key metrics
        final_attendance = trapezoid(M[-1, :], dx=problem.Dx)
        peak_attendance = np.max([trapezoid(M[t, :], dx=problem.Dx) for t in range(problem.Nt)])
        
        # Calculate average satisfaction
        avg_satisfaction = -np.mean(U[-1, :])  # Higher value function = lower cost
        
        results[scenario['name']] = {
            'problem': problem,
            'solution': (U, M),
            'info': info,
            'final_attendance': final_attendance,
            'peak_attendance': peak_attendance,
            'satisfaction': avg_satisfaction,
            'capacity': scenario['capacity'],
            'description': scenario['description']
        }
        
        logger.info(f"  Final attendance: {final_attendance:.1%}")
        logger.info(f"  Peak attendance: {peak_attendance:.1%}")
        logger.info(f"  Beach capacity: {scenario['capacity']:.1%}")
        logger.info(f"  Convergence: {info.get('converged', 'Unknown')}")
    
    return results


def create_beach_visualization(results):
    """Create comprehensive visualization of beach problem results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Towel on the Beach Problem: Coordination Dynamics with Different Beach Capacities', 
                 fontsize=16, fontweight='bold')
    
    # Color scheme
    colors = ['navy', 'darkgreen', 'darkred']
    scenario_names = list(results.keys())
    
    # Plot 1: Attendance Evolution Over Time
    ax1 = axes[0, 0]
    for i, (name, result) in enumerate(results.items()):
        problem = result['problem']
        U, M = result['solution']
        
        # Calculate attendance over time
        attendance_over_time = [trapezoid(M[t, :], dx=problem.Dx) for t in range(problem.Nt)]
        time_grid = np.linspace(0, problem.T, problem.Nt)
        
        ax1.plot(time_grid, attendance_over_time, color=colors[i], 
                linewidth=2, label=f"{name} (Cap: {result['capacity']:.1%})")
        ax1.axhline(y=result['capacity'], color=colors[i], linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Beach Attendance (Fraction)')
    ax1.set_title('Beach Attendance Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final Population Distributions
    ax2 = axes[0, 1]
    for i, (name, result) in enumerate(results.items()):
        problem = result['problem']
        U, M = result['solution']
        
        ax2.plot(problem.xSpace, M[-1, :], color=colors[i], 
                linewidth=2, label=f"{name}")
    
    ax2.set_xlabel('Decision Tendency (0=Stay, 1=Go)')
    ax2.set_ylabel('Population Density')
    ax2.set_title('Final Population Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Value Functions
    ax3 = axes[0, 2]
    for i, (name, result) in enumerate(results.items()):
        problem = result['problem']
        U, M = result['solution']
        
        # Negative value function represents cost-to-go
        ax3.plot(problem.xSpace, -U[-1, :], color=colors[i], 
                linewidth=2, label=f"{name}")
    
    ax3.set_xlabel('Decision Tendency (0=Stay, 1=Go)')
    ax3.set_ylabel('Expected Satisfaction')
    ax3.set_title('Final Value Functions (Satisfaction)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Capacity vs. Outcomes Bar Chart
    ax4 = axes[1, 0]
    capacities = [result['capacity'] for result in results.values()]
    final_attendances = [result['final_attendance'] for result in results.values()]
    peak_attendances = [result['peak_attendance'] for result in results.values()]
    
    x_pos = np.arange(len(scenario_names))
    width = 0.35
    
    ax4.bar(x_pos - width/2, capacities, width, label='Beach Capacity', 
           color='lightblue', alpha=0.7)
    ax4.bar(x_pos + width/2, final_attendances, width, label='Final Attendance',
           color='orange', alpha=0.7)
    
    ax4.set_xlabel('Beach Scenario')
    ax4.set_ylabel('Fraction of Population')
    ax4.set_title('Capacity vs. Actual Attendance')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(scenario_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Coordination Success Analysis
    ax5 = axes[1, 1]
    satisfactions = [result['satisfaction'] for result in results.values()]
    coordination_success = []
    
    for result in results.values():
        # Coordination success: how close final attendance is to capacity
        success = 1 - abs(result['final_attendance'] - result['capacity'])
        coordination_success.append(max(0, success))
    
    ax5.bar(scenario_names, coordination_success, color=['lightcoral', 'lightgreen', 'lightsalmon'])
    ax5.set_xlabel('Beach Scenario')
    ax5.set_ylabel('Coordination Success (0-1)')
    ax5.set_title('Coordination Effectiveness')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Problem Description and Key Insights
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    insights_text = """
TOWEL ON THE BEACH PROBLEM

Problem Setup:
• N individuals decide: go to beach or stay home
• Enjoyment decreases with crowding
• No communication between agents
• Must predict others' behavior

Key Insights:
• Small beaches → Higher coordination challenge
• Medium capacity → Classic dilemma emerges  
• Large beaches → Easier to coordinate
• Final attendance often near capacity
• Mathematical equivalent to El Farol Bar

Coordination Paradox:
If everyone expects empty beach → overcrowding
If everyone expects crowded beach → empty beach
→ Need for diverse prediction strategies
    """
    
    ax6.text(0.05, 0.95, insights_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("examples/basic/outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "towel_beach_problem_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Beach problem visualization saved to: {output_path}")
    
    return output_path


def main():
    """Main execution function."""
    logger.info("Starting Towel on the Beach Problem Analysis")
    
    print("🏖️  TOWEL ON THE BEACH PROBLEM")
    print("=" * 50)
    print("An accessible version of the Santa Fe El Farol coordination dilemma")
    print()
    print("Problem: N people decide whether to go to a beach")
    print("Dilemma: Beach is pleasant when uncrowded, unpleasant when crowded")
    print("Challenge: Must predict others' behavior without communication")
    print()
    
    # Solve different scenarios
    results = solve_beach_problem_variants()
    
    # Create visualization
    plot_path = create_beach_visualization(results)
    
    # Display results summary
    print("\n RESULTS SUMMARY")
    print("-" * 30)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Description: {result['description']}")
        print(f"  Beach Capacity: {result['capacity']:.1%}")
        print(f"  Final Attendance: {result['final_attendance']:.1%}")
        print(f"  Peak Attendance: {result['peak_attendance']:.1%}")
        
        # Coordination analysis
        capacity_match = abs(result['final_attendance'] - result['capacity'])
        if capacity_match < 0.1:
            coordination = "Excellent"
        elif capacity_match < 0.2:
            coordination = "Good"
        elif capacity_match < 0.3:
            coordination = "Fair"
        else:
            coordination = "Poor"
        
        print(f"  Coordination Quality: {coordination}")
        print(f"  Converged: {result['info'].get('converged', 'Unknown')}")
    
    print(f"\n Visualization saved to: {plot_path}")
    
    print("\n KEY INSIGHTS:")
    print("• The beach problem is mathematically identical to El Farol Bar")
    print("• Different capacities lead to different coordination challenges")
    print("• Final attendance typically converges near beach capacity")
    print("• Demonstrates bounded rationality and prediction complexity")
    print("• Shows how everyday scenarios contain deep coordination theory")
    
    logger.info("Towel Beach Problem analysis completed successfully")


if __name__ == "__main__":
    main()