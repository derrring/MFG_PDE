#!/usr/bin/env python3
"""
Simple MFG Visualization Demo - Show Solutions with Plots

This creates a visual demonstration of the MFG solution using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Core MFG framework
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.config.array_validation import MFGGridConfig, MFGArrays

def create_and_solve_mfg():
    """Create and solve a simple MFG problem."""
    print("🎯 Creating and solving MFG problem...")
    
    # Create problem with stable parameters
    grid_config = MFGGridConfig(
        Nx=30, Nt=15, xmin=0.0, xmax=1.0, T=0.3, sigma=0.08
    )
    
    problem = ExampleMFGProblem(
        xmin=grid_config.xmin, xmax=grid_config.xmax, 
        T=grid_config.T, Nx=grid_config.Nx, Nt=grid_config.Nt, 
        sigma=grid_config.sigma
    )
    
    print(f"   Grid: {grid_config.Nx}×{grid_config.Nt}, CFL={grid_config.cfl_number:.3f}")
    
    # Create collocation points
    np.random.seed(123)
    collocation_points = np.random.uniform(0, 1, (20, 1))
    
    # Solve with particle method
    solver = ParticleCollocationSolver(
        problem=problem, 
        collocation_points=collocation_points,
        num_particles=800, 
        kde_bandwidth=0.06
    )
    
    print("   Solving...")
    result = solver.solve(max_iterations=10, tolerance=1e-3, verbose=False)
    U, M, info = result
    
    print("   ✅ Solution completed")
    
    return U, M, problem, grid_config

def validate_and_visualize(U, M, problem, grid_config):
    """Validate solution and create visualizations."""
    
    # Validate with Pydantic
    print("🔍 Validating solution...")
    arrays = MFGArrays(U_solution=U, M_solution=M, grid_config=grid_config)
    stats = arrays.get_solution_statistics()
    
    print(f"   Mass conservation: {stats['mass_conservation']['mass_drift']:.2e}")
    print(f"   Final mass: {stats['mass_conservation']['final_mass']:.6f}")
    
    # Create visualizations
    print("📊 Creating visualizations...")
    
    # Grid for plotting
    x_grid = np.linspace(problem.xmin, problem.xmax, problem.Nx + 1)
    t_grid = np.linspace(0, problem.T, problem.Nt + 1)
    X, T = np.meshgrid(x_grid, t_grid)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Value function u(t,x) - 3D surface
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(T, X, U, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Space x') 
    ax1.set_zlabel('u(t,x)')
    ax1.set_title('Value Function u(t,x)')
    
    # Density m(t,x) - 3D surface
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(T, X, M, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Space x')
    ax2.set_zlabel('m(t,x)')
    ax2.set_title('Density m(t,x)')
    
    # Value function - contour plot
    ax3 = fig.add_subplot(2, 3, 3)
    contour1 = ax3.contourf(T, X, U, levels=20, cmap='viridis')
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Space x')
    ax3.set_title('Value Function u(t,x) - Contours')
    plt.colorbar(contour1, ax=ax3)
    
    # Density - contour plot
    ax4 = fig.add_subplot(2, 3, 4)
    contour2 = ax4.contourf(T, X, M, levels=20, cmap='plasma')
    ax4.set_xlabel('Time t')
    ax4.set_ylabel('Space x')
    ax4.set_title('Density m(t,x) - Contours')
    plt.colorbar(contour2, ax=ax4)
    
    # Time evolution of value function
    ax5 = fig.add_subplot(2, 3, 5)
    time_indices = [0, len(t_grid)//3, 2*len(t_grid)//3, -1]
    for i, t_idx in enumerate(time_indices):
        ax5.plot(x_grid, U[t_idx], label=f't={t_grid[t_idx]:.2f}', 
                linewidth=2, alpha=0.8)
    ax5.set_xlabel('Space x')
    ax5.set_ylabel('u(t,x)')
    ax5.set_title('Value Function Evolution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Time evolution of density
    ax6 = fig.add_subplot(2, 3, 6)
    for i, t_idx in enumerate(time_indices):
        ax6.plot(x_grid, M[t_idx], label=f't={t_grid[t_idx]:.2f}', 
                linewidth=2, alpha=0.8)
    ax6.set_xlabel('Space x')
    ax6.set_ylabel('m(t,x)')
    ax6.set_title('Density Evolution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("./results/visualization_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_dir / "mfg_solution_visualization.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"   📄 Visualization saved: {plot_path}")
    
    return stats

def main():
    """Main demonstration."""
    print("=" * 60)
    print("🎨 MFG Solution Visualization Demo")
    print("=" * 60)
    
    # Solve MFG
    U, M, problem, grid_config = create_and_solve_mfg()
    
    # Validate and visualize
    stats = validate_and_visualize(U, M, problem, grid_config)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VISUALIZATION SUMMARY")
    print("=" * 60)
    print(f"✅ Solution shape: {U.shape}")
    print(f"🔍 Mass conservation: {stats['mass_conservation']['mass_drift']:.2e}")
    print(f"📊 Plots generated and displayed")
    print("=" * 60)

if __name__ == "__main__":
    main()