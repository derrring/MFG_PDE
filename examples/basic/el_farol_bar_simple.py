#!/usr/bin/env python3
"""
Simple El Farol Bar Problem Example

A minimal example showing how to solve the classic El Farol Bar Problem
using Mean Field Games with the MFG_PDE framework.

Quick start for the Santa Fe Bar Problem!
"""

import numpy as np
import matplotlib.pyplot as plt
from mfg_pde import ExampleMFGProblem, create_fast_solver
import sys
import os

# Add the advanced example to path so we can import our custom class
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'advanced'))

try:
    from el_farol_bar_mfg import ElFarolBarMFG, solve_el_farol_bar_problem
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False


def simple_el_farol_example():
    """
    Simple example of El Farol Bar Problem using MFG.
    """
    
    print("🍺 El Farol Bar Problem - Simple Example")
    print("=" * 50)
    
    if not ADVANCED_AVAILABLE:
        print("⚠️ Advanced El Farol implementation not available.")
        print("Using basic MFG formulation as approximation...")
        
        # Fallback to basic MFG problem
        problem = ExampleMFGProblem(
            xmin=0.0, xmax=1.0, Nx=30,
            T=1.0, Nt=30,
            sigma=0.1,  # Decision uncertainty
            coefCT=0.02  # Cost scaling
        )
        
        solver = create_fast_solver(problem, solver_type="fixed_point")
        result = solver.solve()
        
        U, M = result.U, result.M
        
        # Basic visualization
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        x = np.linspace(0, 1, 30)
        t = np.linspace(0, 1, 30)
        X, T = np.meshgrid(x, t)
        plt.contourf(X, T, M, levels=15, cmap='viridis')
        plt.colorbar(label='Population Density')
        plt.xlabel('Decision State x')
        plt.ylabel('Time t')
        plt.title('Population Evolution')
        
        plt.subplot(1, 3, 2)
        plt.contourf(X, T, U, levels=15, cmap='plasma')
        plt.colorbar(label='Value Function')
        plt.xlabel('Decision State x')
        plt.ylabel('Time t')
        plt.title('Value Function')
        
        plt.subplot(1, 3, 3)
        final_density = M[-1, :]
        plt.plot(x, final_density, 'b-', linewidth=2)
        plt.xlabel('Decision State x')
        plt.ylabel('Final Density')
        plt.title('Final Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Basic example completed!")
        return
    
    # Use the advanced implementation
    print("🚀 Using advanced El Farol Bar implementation...")
    
    # Solve with default parameters
    U, M, analysis = solve_el_farol_bar_problem(
        bar_capacity=0.6,      # 60% capacity is optimal
        crowd_aversion=2.0,    # Moderate crowd aversion
        solver_type="fixed_point",
        visualize=True
    )
    
    print("\n📊 Results Summary:")
    print(f"  • Final attendance: {analysis['final_attendance']:.1%}")
    print(f"  • Economic efficiency: {analysis['efficiency']:.1%}")
    print(f"  • Converged: {'Yes' if analysis['converged'] else 'No'}")
    print(f"  • Solve time: {analysis['solve_time']:.2f} seconds")
    
    print("\n🎯 Economic Interpretation:")
    if analysis['final_attendance'] > 0.6:
        print("  → Bar is overcrowded! Agents avoid due to crowding.")
    elif analysis['final_attendance'] < 0.5:
        print("  → Bar is underutilized. Agents could coordinate better.")
    else:
        print("  → Near-optimal attendance achieved!")
    
    if analysis['efficiency'] > 0.8:
        print("  → High economic efficiency - good coordination!")
    else:
        print("  → Room for improvement in coordination.")
    
    print("\n✅ El Farol Bar analysis completed!")
    

def interactive_parameter_exploration():
    """
    Interactive exploration of El Farol Bar parameters.
    """
    
    if not ADVANCED_AVAILABLE:
        print("⚠️ Advanced implementation needed for parameter exploration.")
        return
    
    print("\n🔬 Parameter Exploration")
    print("Testing different crowd aversion levels...")
    
    aversion_levels = [0.5, 1.0, 2.0, 4.0]
    results = []
    
    plt.figure(figsize=(10, 6))
    
    for i, aversion in enumerate(aversion_levels):
        print(f"  Testing crowd aversion = {aversion}...")
        
        U, M, analysis = solve_el_farol_bar_problem(
            crowd_aversion=aversion,
            visualize=False
        )
        
        results.append(analysis)
        
        # Plot attendance evolution
        plt.subplot(2, 2, i+1)
        t_grid = np.linspace(0, 1, len(analysis['attendance_evolution']))
        plt.plot(t_grid, analysis['attendance_evolution'], 'b-', linewidth=2)
        plt.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Optimal')
        plt.title(f'Crowd Aversion = {aversion}')
        plt.xlabel('Time')
        plt.ylabel('Attendance')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Summary table
    print("\n📊 Parameter Exploration Results:")
    print("Aversion | Final Attend. | Efficiency | Converged")
    print("-" * 48)
    for i, aversion in enumerate(aversion_levels):
        analysis = results[i]
        converged = "Yes" if analysis['converged'] else "No"
        print(f"{aversion:8.1f} | {analysis['final_attendance']:12.3f} | "
              f"{analysis['efficiency']:9.3f} | {converged:9s}")
    
    print("\n💡 Insights:")
    print("  • Higher crowd aversion → Lower attendance")
    print("  • Moderate aversion often gives best efficiency")
    print("  • Too low aversion → Overcrowding")
    print("  • Too high aversion → Underutilization")


def main():
    """Main function to run simple El Farol Bar examples."""
    
    try:
        # Basic example
        simple_el_farol_example()
        
        # Interactive exploration
        response = input("\n❓ Would you like to explore different parameters? (y/n): ")
        if response.lower().startswith('y'):
            interactive_parameter_exploration()
        
        print("\n🎉 Thanks for exploring the El Farol Bar Problem with MFG!")
        print("📚 The El Farol Bar Problem demonstrates how individual rational")
        print("   decisions can lead to suboptimal collective outcomes.")
        print("🔬 Mean Field Games help us understand and optimize such systems!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the MFG_PDE package is properly installed.")


if __name__ == "__main__":
    main()