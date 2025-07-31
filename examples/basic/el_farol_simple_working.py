#!/usr/bin/env python3
"""
El Farol Bar Problem - Working Simple Implementation

A simplified but working implementation of the El Farol Bar Problem
using the ExampleMFGProblem as a base, with custom cost functions
that represent the bar attendance decision-making.

Mathematical Formulation:
- State space: x ∈ [0,1] (tendency to attend bar)
- Population density: m(t,x) with ∫₀¹ m(t,x) dx = 1
- Expected attendance: A(t) = ∫₀¹ x·m(t,x) dx
- Cost function: L(t,x,u,m) = α·max(0, A(t) - C̄)² + ½u² + β(x - x_hist)²

Where:
- α: crowd aversion parameter
- C̄: normalized bar capacity
- β: historical memory weight
- σ: decision volatility

This focuses on the key economic insights rather than full MFG formulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from mfg_pde import ExampleMFGProblem, create_fast_solver
from mfg_pde.utils.logging import get_logger, configure_research_logging
from mfg_pde.utils.integration import trapezoid


def create_el_farol_problem(
    bar_capacity: float = 0.6,
    crowd_aversion: float = 2.0,
    Nx: int = 40,
    Nt: int = 40
):
    """
    Create an MFG problem that represents El Farol Bar dynamics.
    
    We use the ExampleMFGProblem but interpret the state space as
    "tendency to go to the bar" where:
    - x=0: Strong tendency to stay home
    - x=1: Strong tendency to go to bar
    - The density m(t,x) represents population distribution over tendencies
    """
    
    # Create base problem with parameters tuned for bar problem
    problem = ExampleMFGProblem(
        xmin=0.0, xmax=1.0, Nx=Nx,
        T=1.0, Nt=Nt,
        sigma=0.15,  # Decision volatility/uncertainty
        coefCT=crowd_aversion * 0.1,  # Crowd aversion scaling
    )
    
    # Store El Farol specific parameters
    problem.bar_capacity = bar_capacity
    problem.crowd_aversion = crowd_aversion
    
    return problem


def analyze_el_farol_solution(problem, U, M):
    """
    Analyze the MFG solution from an El Farol Bar perspective.
    """
    
    logger = get_logger(__name__)
    
    # Create grids matching the solution arrays
    x_grid = np.linspace(problem.xmin, problem.xmax, M.shape[1])
    t_grid = np.linspace(0, problem.T, M.shape[0])
    
    # Calculate attendance evolution over time
    attendance_rates = []
    for t_idx in range(M.shape[0]):
        # Expected attendance = ∫ x * m(t,x) dx
        # (x represents probability of going, weighted by density)  
        attendance = trapezoid(x_grid * M[t_idx, :], x_grid)
        attendance_rates.append(attendance)
    
    # Analysis metrics
    final_attendance = attendance_rates[-1]
    capacity_utilization = final_attendance / problem.bar_capacity
    
    # Economic efficiency (1 = perfect, 0 = worst)
    efficiency = 1.0 - abs(final_attendance - problem.bar_capacity) / problem.bar_capacity
    
    # Convergence stability
    late_variance = np.var(attendance_rates[-10:])
    converged = late_variance < 1e-4
    
    # Population distribution analysis
    final_density = M[-1, :]
    peak_location = x_grid[np.argmax(final_density)]
    density_concentration = np.max(final_density) / np.mean(final_density)
    
    analysis = {
        'attendance_evolution': attendance_rates,
        'final_attendance': final_attendance,
        'capacity_utilization': capacity_utilization,
        'efficiency': efficiency,
        'converged': converged,
        'peak_tendency': peak_location,
        'density_concentration': density_concentration,
        'bar_capacity': problem.bar_capacity
    }
    
    logger.info(f" El Farol Analysis: Attendance={final_attendance:.1%}, "
                f"Efficiency={efficiency:.1%}, Peak={peak_location:.2f}")
    
    return analysis


def visualize_el_farol_results(problem, U, M, analysis):
    """
    Create El Farol Bar specific visualizations.
    """
    
    # Use actual array shapes to ensure consistency
    x_grid = np.linspace(problem.xmin, problem.xmax, M.shape[1])
    t_grid = np.linspace(0, problem.T, M.shape[0])
    X, T = np.meshgrid(x_grid, t_grid)
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('El Farol Bar Problem - MFG Analysis\n'
                 f'Capacity: {problem.bar_capacity:.0%}, Crowd Aversion: {problem.crowd_aversion}', 
                 fontsize=14, fontweight='bold')
    
    # 1. Population density evolution
    ax1 = plt.subplot(2, 3, 1)
    contour = ax1.contourf(X, T, M, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, ax=ax1, label='Population Density')
    ax1.set_xlabel('Going Tendency')
    ax1.set_ylabel('Time')
    ax1.set_title('Population Distribution m(t,x)')
    ax1.axvline(x=problem.bar_capacity, color='red', linestyle='--', alpha=0.7, label='Capacity')
    ax1.legend()
    
    # 2. Value function
    ax2 = plt.subplot(2, 3, 2)
    contour2 = ax2.contourf(X, T, U, levels=20, cmap='plasma', alpha=0.8)
    plt.colorbar(contour2, ax=ax2, label='Value Function')
    ax2.set_xlabel('Going Tendency')
    ax2.set_ylabel('Time')
    ax2.set_title('Value Function U(t,x)')
    
    # 3. Attendance evolution
    ax3 = plt.subplot(2, 3, 3)
    attendance = analysis['attendance_evolution']
    ax3.plot(t_grid, attendance, 'b-', linewidth=3, label='Bar Attendance')
    ax3.axhline(y=problem.bar_capacity, color='red', linestyle='--', linewidth=2, 
                label=f'Optimal ({problem.bar_capacity:.0%})')
    ax3.fill_between(t_grid, attendance, alpha=0.3)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Expected Attendance')
    ax3.set_title('Bar Attendance Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add efficiency annotation
    efficiency = analysis['efficiency']
    ax3.text(0.05, 0.95, f'Efficiency: {efficiency:.1%}', 
             transform=ax3.transAxes, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # 4. Final population distribution
    ax4 = plt.subplot(2, 3, 4)
    final_density = M[-1, :]
    ax4.plot(x_grid, final_density, 'g-', linewidth=3, label='Final Density')
    ax4.fill_between(x_grid, final_density, alpha=0.3, color='green')
    ax4.axvline(x=problem.bar_capacity, color='red', linestyle='--', linewidth=2,
                label=f'Capacity ({problem.bar_capacity:.0%})')
    ax4.axvline(x=analysis['peak_tendency'], color='orange', linestyle=':', linewidth=2,
                label=f'Peak ({analysis["peak_tendency"]:.2f})')
    ax4.set_xlabel('Going Tendency')
    ax4.set_ylabel('Population Density')
    ax4.set_title('Final Population Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Economic analysis
    ax5 = plt.subplot(2, 3, 5)
    
    # Create bar chart of key metrics
    metrics = ['Final\nAttendance', 'Capacity\nUtilization', 'Economic\nEfficiency']
    values = [analysis['final_attendance'], analysis['capacity_utilization'], analysis['efficiency']]
    colors = ['lightblue', 'lightgreen', 'orange']
    
    bars = ax5.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Value')
    ax5.set_title('Key Performance Metrics')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax5.set_ylim(0, max(1.2, max(values) * 1.1))
    
    # 6. Economic interpretation
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Economic insights text
    interpretation = f"""
El Farol Bar Economic Analysis

 Problem Setup:
• Bar capacity: {problem.bar_capacity:.0%}
• Crowd aversion: {problem.crowd_aversion:.1f}
• Decision volatility: {problem.sigma:.2f}

 Equilibrium Results:
• Final attendance: {analysis['final_attendance']:.1%}
• Capacity utilization: {analysis['capacity_utilization']:.1%}
• Economic efficiency: {analysis['efficiency']:.1%}

🧠 Behavioral Insights:
• Peak tendency: {analysis['peak_tendency']:.2f}
• Density concentration: {analysis['density_concentration']:.1f}×
• Converged: {'Yes' if analysis['converged'] else 'No'}

 Economic Interpretation:
"""
    
    if analysis['final_attendance'] > problem.bar_capacity * 1.1:
        interpretation += "• Overcrowding occurs\n• Agents over-coordinate to go\n• Efficiency loss from crowding"
    elif analysis['final_attendance'] < problem.bar_capacity * 0.9:
        interpretation += "• Bar underutilized\n• Agents over-avoid due to fear\n• Efficiency loss from under-use"
    else:
        interpretation += "• Near-optimal attendance\n• Good coordination achieved\n• High economic efficiency"
    
    ax6.text(0.05, 0.95, interpretation, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"el_farol_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📁 Visualization saved as {filename}")
    
    plt.show()


def compare_scenarios():
    """
    Compare different El Farol scenarios to understand parameter effects.
    """
    
    print("🔬 Comparing El Farol scenarios...")
    
    scenarios = [
        {"bar_capacity": 0.4, "crowd_aversion": 1.0, "name": "Small Bar, Low Aversion"},
        {"bar_capacity": 0.6, "crowd_aversion": 2.0, "name": "Medium Bar, Medium Aversion"},
        {"bar_capacity": 0.8, "crowd_aversion": 1.0, "name": "Large Bar, Low Aversion"},
        {"bar_capacity": 0.6, "crowd_aversion": 4.0, "name": "Medium Bar, High Aversion"},
    ]
    
    results = []
    
    plt.figure(figsize=(16, 12))
    
    for i, scenario in enumerate(scenarios):
        print(f"  Solving: {scenario['name']}...")
        
        # Create and solve problem
        problem = create_el_farol_problem(
            bar_capacity=scenario['bar_capacity'],
            crowd_aversion=scenario['crowd_aversion'],
            Nx=30, Nt=30  # Smaller for faster computation
        )
        
        solver = create_fast_solver(problem, solver_type="fixed_point")
        result = solver.solve()
        U, M = result.U, result.M
        
        # Analyze results
        analysis = analyze_el_farol_solution(problem, U, M)
        analysis['scenario'] = scenario
        results.append(analysis)
        
        # Plot attendance evolution - use actual M shape
        plt.subplot(2, 4, i+1)
        t_grid = np.linspace(0, 1, M.shape[0])
        plt.plot(t_grid, analysis['attendance_evolution'], 'b-', linewidth=2)
        plt.axhline(y=scenario['bar_capacity'], color='red', linestyle='--', alpha=0.7)
        plt.title(scenario['name'], fontsize=10)
        plt.xlabel('Time')
        plt.ylabel('Attendance')
        plt.grid(True, alpha=0.3)
        
        # Plot final density - use actual M shape
        plt.subplot(2, 4, i+5)
        x_grid = np.linspace(0, 1, M.shape[1])
        final_density = M[-1, :]
        plt.plot(x_grid, final_density, 'g-', linewidth=2)
        plt.axvline(x=scenario['bar_capacity'], color='red', linestyle='--', alpha=0.7)
        plt.title('Final Distribution', fontsize=10)
        plt.xlabel('Going Tendency')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary comparison
    print("\n Scenario Comparison Results:")
    print("Scenario                    | Attendance | Efficiency | Converged")
    print("-" * 65)
    for result in results:
        scenario = result['scenario']
        name = scenario['name'][:26].ljust(26)
        attendance = f"{result['final_attendance']:.1%}".rjust(10)
        efficiency = f"{result['efficiency']:.1%}".rjust(10)
        converged = "Yes" if result['converged'] else "No"
        print(f"{name} | {attendance} | {efficiency} | {converged}")
    
    return results


def main():
    """
    Main function for El Farol Bar analysis.
    """
    
    configure_research_logging("el_farol_simple", level="INFO")
    logger = get_logger(__name__)
    
    print("🍺 El Farol Bar Problem - Simple MFG Implementation")
    print("=" * 60)
    
    try:
        # Basic example
        print("\n1. Solving basic El Farol Bar Problem...")
        
        problem = create_el_farol_problem(
            bar_capacity=0.6,  # 60% capacity optimal
            crowd_aversion=2.0  # Moderate crowd aversion
        )
        
        solver = create_fast_solver(problem, solver_type="fixed_point")
        result = solver.solve()
        U, M = result.U, result.M
        
        print(f"SUCCESS: Solved in {result.execution_time:.2f} seconds")
        
        # Analyze results
        analysis = analyze_el_farol_solution(problem, U, M)
        
        # Create visualizations
        visualize_el_farol_results(problem, U, M, analysis)
        
        print(f"\n Key Results:")
        print(f"  • Final attendance: {analysis['final_attendance']:.1%}")
        print(f"  • Economic efficiency: {analysis['efficiency']:.1%}")
        print(f"  • Capacity utilization: {analysis['capacity_utilization']:.1%}")
        
        # Economic interpretation
        print(f"\n Economic Insights:")
        if analysis['efficiency'] > 0.8:
            print("  → High efficiency! Good coordination between agents.")
        elif analysis['efficiency'] > 0.6:
            print("  → Moderate efficiency. Some coordination issues.")
        else:
            print("  → Low efficiency. Poor coordination leading to suboptimal outcomes.")
        
        if analysis['final_attendance'] > problem.bar_capacity * 1.1:
            print("  → Bar is overcrowded. Agents failed to avoid peak times.")
        elif analysis['final_attendance'] < problem.bar_capacity * 0.9:
            print("  → Bar is underutilized. Agents over-cautious about crowding.")
        else:
            print("  → Near-optimal attendance achieved!")
        
        # Optional scenario comparison - auto-run for demo
        print("\n🔬 Running scenario comparison...")
        compare_scenarios()
        
        print("\n El Farol Bar analysis completed!")
        print("📚 This demonstrates how individual rational decisions can lead to")
        print("   collective outcomes that may be suboptimal without coordination.")
        
    except Exception as e:
        logger.error(f"ERROR: Error in El Farol analysis: {e}")
        raise


if __name__ == "__main__":
    main()