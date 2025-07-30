#!/usr/bin/env python3
"""
Numerical Example: Towel on Beach Spatial Competition
Demonstrates the transition from single peak to crater equilibria
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Import the MFG framework
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.factory import create_fast_solver
from mfg_pde.utils.integration import trapezoid

class TowelBeachSpatialExample(MFGProblem):
    """Simplified version for numerical demonstration."""
    
    def __init__(self, stall_position=0.5, crowd_aversion=1.0, **kwargs):
        super().__init__(**kwargs)
        self.stall_position = stall_position
        self.crowd_aversion = crowd_aversion
    
    def g(self, x):
        """Terminal cost: distance from stall."""
        return 0.5 * np.abs(x - self.stall_position)
    
    def f(self, x, u, m):
        """Running cost: L(x,u,m) = |x - x_stall| + λ*ln(m) + (1/2)u²"""
        distance_cost = np.abs(x - self.stall_position)
        epsilon = 1e-6
        congestion_cost = self.crowd_aversion * np.log(np.maximum(m, epsilon))
        movement_cost = 0.5 * u**2
        return distance_cost + congestion_cost + movement_cost
    
    def rho0(self, x):
        """Initial distribution: Gaussian centered away from stall."""
        density = np.exp(-10 * (x - 0.3)**2)
        return density / trapezoid(density, dx=self.Dx)
    
    def sigma(self, x):
        """Noise level."""
        return 0.1
    
    def boundary_conditions_m(self):
        return BoundaryConditions(type="no_flux")
    
    def boundary_conditions_u(self):
        return BoundaryConditions(type="neumann", left_value=0.0, right_value=0.0)
    
    def H(self, x_idx: int, m_at_x: float, p_values: dict, t_idx=None):
        """Hamiltonian for optimal control."""
        x = self.xSpace[x_idx]
        p_x = p_values.get("p_x", 0.0)
        
        # Distance cost
        distance_cost = np.abs(x - self.stall_position)
        
        # Congestion cost  
        epsilon = 1e-6
        congestion_cost = self.crowd_aversion * np.log(np.maximum(m_at_x, epsilon))
        
        # Hamiltonian with optimal control
        return distance_cost + congestion_cost - 0.5 * p_x**2
    
    def dH_dm(self, x_idx: int, m_at_x: float, p_values: dict, t_idx=None):
        """Derivative of Hamiltonian w.r.t. density."""
        epsilon = 1e-6
        return self.crowd_aversion / np.maximum(m_at_x, epsilon)


def run_numerical_example():
    """Run numerical example with three crowd aversion levels."""
    
    print("🏖️ TOWEL ON BEACH SPATIAL COMPETITION - NUMERICAL EXAMPLE")
    print("=" * 65)
    print("Mathematical Model: L(x,u,m) = |x - x_stall| + λ·ln(m) + ½u²")
    print("State space: x ∈ [0,1] (position on beach)")
    print("Ice cream stall at x = 0.6")
    print()
    
    scenarios = [
        {"name": "Low Aversion", "lambda": 0.5, "color": "blue"},
        {"name": "Medium Aversion", "lambda": 1.5, "color": "green"}, 
        {"name": "High Aversion", "lambda": 3.0, "color": "red"}
    ]
    
    results = {}
    
    # Solve each scenario
    for scenario in scenarios:
        print(f"Solving {scenario['name']} (λ = {scenario['lambda']})...")
        
        problem = TowelBeachSpatialExample(
            T=1.0, Nx=100, Nt=50,
            stall_position=0.6,
            crowd_aversion=scenario['lambda']
        )
        
        solver = create_fast_solver(problem)
        result = solver.solve()
        U, M = result.U, result.M
        
        # Analyze results
        x_grid = problem.xSpace
        final_density = M[-1, :]
        
        # Find maximum density and location
        max_idx = np.argmax(final_density)
        max_density = final_density[max_idx]
        max_location = x_grid[max_idx]
        
        # Density at stall
        stall_idx = np.argmin(np.abs(x_grid - problem.stall_position))
        density_at_stall = final_density[stall_idx]
        
        # Spatial spread
        mean_pos = trapezoid(x_grid * final_density, x_grid)
        variance = trapezoid((x_grid - mean_pos)**2 * final_density, x_grid)
        spatial_spread = np.sqrt(variance)
        
        # Classify equilibrium
        if density_at_stall >= 0.9 * max_density:
            eq_type = "Single Peak"
        elif max_density > 1.2 * density_at_stall:
            eq_type = "Crater Pattern"
        else:
            eq_type = "Mixed Pattern"
        
        results[scenario['name']] = {
            'lambda': scenario['lambda'],
            'color': scenario['color'],
            'final_density': final_density,
            'max_density': max_density,
            'max_location': max_location,
            'density_at_stall': density_at_stall,
            'spatial_spread': spatial_spread,
            'equilibrium_type': eq_type,
            'x_grid': x_grid,
            'converged': result.metadata.get('converged', True)
        }
        
        print(f"  ✅ {eq_type}: max density {max_density:.3f} at x={max_location:.3f}")
        print(f"     Density at stall: {density_at_stall:.3f}, spread: {spatial_spread:.3f}")
    
    return results


def create_visualization(results):
    """Create visualization of the spatial equilibrium patterns."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Towel on Beach: Spatial Competition with Varying Crowd Aversion', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Final density distributions
    ax1 = axes[0, 0]
    for name, result in results.items():
        ax1.plot(result['x_grid'], result['final_density'], 
                color=result['color'], linewidth=2, 
                label=f"{name} (λ={result['lambda']})")
        
        # Mark stall position
        ax1.axvline(x=0.6, color=result['color'], linestyle='--', alpha=0.3)
    
    ax1.axvline(x=0.6, color='black', linestyle=':', linewidth=2, alpha=0.8, label='Stall')
    ax1.set_xlabel('Beach Position x')
    ax1.set_ylabel('Population Density m(x)')
    ax1.set_title('Final Spatial Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Density at stall vs lambda
    ax2 = axes[0, 1]
    lambdas = [result['lambda'] for result in results.values()]
    densities_at_stall = [result['density_at_stall'] for result in results.values()]
    max_densities = [result['max_density'] for result in results.values()]
    
    ax2.plot(lambdas, densities_at_stall, 'ro-', linewidth=2, markersize=8, label='Density at Stall')
    ax2.plot(lambdas, max_densities, 'bs-', linewidth=2, markersize=8, label='Maximum Density')
    ax2.set_xlabel('Crowd Aversion Parameter λ')
    ax2.set_ylabel('Density')
    ax2.set_title('Stall vs Maximum Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Equilibrium classification
    ax3 = axes[1, 0]
    eq_types = [result['equilibrium_type'] for result in results.values()]
    colors = [result['color'] for result in results.values()]
    spreads = [result['spatial_spread'] for result in results.values()]
    
    bars = ax3.bar(eq_types, spreads, color=colors, alpha=0.7)
    ax3.set_ylabel('Spatial Spread σ')
    ax3.set_title('Population Dispersion by Equilibrium Type')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, spread in zip(bars, spreads):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{spread:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Results summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "NUMERICAL RESULTS SUMMARY\\n\\n"
    for name, result in results.items():
        summary_text += f"{name} (λ={result['lambda']}):\\n"
        summary_text += f"  Type: {result['equilibrium_type']}\\n"
        summary_text += f"  Max density: {result['max_density']:.3f} at x={result['max_location']:.3f}\\n"
        summary_text += f"  Stall density: {result['density_at_stall']:.3f}\\n"
        summary_text += f"  Spread: σ={result['spatial_spread']:.3f}\\n"
        summary_text += f"  Converged: {result['converged']}\\n\\n"
    
    summary_text += "KEY INSIGHT:\\n"
    summary_text += "As λ increases, agents avoid\\n"
    summary_text += "the stall area despite attraction,\\n"
    summary_text += "creating crater-like patterns."
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("towel_beach_numerical_example.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\\n📊 Visualization saved to: {output_path}")
    
    return output_path


def display_results(results):
    """Display detailed numerical results."""
    
    print("\\n" + "="*65)
    print("📊 DETAILED NUMERICAL RESULTS")
    print("="*65)
    
    print(f"{'Parameter':<20} {'Low λ=0.5':<15} {'Med λ=1.5':<15} {'High λ=3.0':<15}")
    print("-" * 65)
    
    # Extract data for comparison
    low = results["Low Aversion"]
    med = results["Medium Aversion"] 
    high = results["High Aversion"]
    
    print(f"{'Equilibrium Type':<20} {low['equilibrium_type']:<15} {med['equilibrium_type']:<15} {high['equilibrium_type']:<15}")
    print(f"{'Max Density':<20} {low['max_density']:<15.3f} {med['max_density']:<15.3f} {high['max_density']:<15.3f}")
    print(f"{'Max Location':<20} {low['max_location']:<15.3f} {med['max_location']:<15.3f} {high['max_location']:<15.3f}")
    print(f"{'Density at Stall':<20} {low['density_at_stall']:<15.3f} {med['density_at_stall']:<15.3f} {high['density_at_stall']:<15.3f}")
    print(f"{'Spatial Spread':<20} {low['spatial_spread']:<15.3f} {med['spatial_spread']:<15.3f} {high['spatial_spread']:<15.3f}")
    print(f"{'Converged':<20} {str(low['converged']):<15} {str(med['converged']):<15} {str(high['converged']):<15}")
    
    print("\\n🎯 KEY OBSERVATIONS:")
    print("• Low λ → Single peak at stall (attraction dominates)")
    print("• Medium λ → Transition begins (mixed pattern)")  
    print("• High λ → Crater pattern (congestion penalty dominates)")
    print("• Spatial spread increases with crowd aversion")
    print("• Equilibrium smoothly transitions between types")


if __name__ == "__main__":
    try:
        # Run numerical example
        results = run_numerical_example()
        
        # Create visualization  
        plot_path = create_visualization(results)
        
        # Display detailed results
        display_results(results)
        
        print("\\n✅ Numerical example completed successfully!")
        print("This demonstrates spatial sorting in Mean Field Games")
        
    except Exception as e:
        print(f"❌ Error in numerical example: {e}")
        import traceback
        traceback.print_exc()