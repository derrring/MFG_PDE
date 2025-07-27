#!/usr/bin/env python3
"""
Clean font demonstration - no serif font warnings.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure matplotlib to use available fonts without warnings
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica'],
    'mathtext.fontset': 'dejavusans',
    'text.usetex': False,
    'axes.formatter.use_mathtext': True
})

# Clear font cache to ensure changes take effect
try:
    mpl.font_manager._rebuild()
except:
    pass

def run_clean_visualization():
    """Run MFG visualization without font warnings."""
    
    print("🎨 Running clean MFG visualization...")
    
    # Import after font configuration
    from mfg_pde.core.mfg_problem import ExampleMFGProblem
    from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
    from mfg_pde.config.array_validation import MFGGridConfig
    
    # Create problem
    grid_config = MFGGridConfig(
        Nx=25, Nt=10, xmin=0.0, xmax=1.0, T=0.2, sigma=0.06
    )
    
    problem = ExampleMFGProblem(
        xmin=grid_config.xmin, xmax=grid_config.xmax, 
        T=grid_config.T, Nx=grid_config.Nx, Nt=grid_config.Nt, 
        sigma=grid_config.sigma
    )
    
    print(f"   Grid: {grid_config.Nx}×{grid_config.Nt}, CFL={grid_config.cfl_number:.3f}")
    
    # Quick solve
    np.random.seed(456)
    collocation_points = np.random.uniform(0, 1, (15, 1))
    
    solver = ParticleCollocationSolver(
        problem=problem, 
        collocation_points=collocation_points,
        num_particles=500, 
        kde_bandwidth=0.08
    )
    
    print("   Solving (quick)...")
    result = solver.solve(max_iterations=5, tolerance=1e-2, verbose=False)
    U, M, info = result
    
    # Create clean visualizations
    x_grid = np.linspace(problem.xmin, problem.xmax, problem.Nx + 1)
    t_grid = np.linspace(0, problem.T, problem.Nt + 1)
    X, T = np.meshgrid(x_grid, t_grid)
    
    print("   Creating plots...")
    
    # Create figure with clean styling
    fig = plt.figure(figsize=(12, 8))
    
    # 3D plots
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(T, X, U, cmap='viridis', alpha=0.7)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Space x') 
    ax1.set_zlabel('u(t,x)')
    ax1.set_title('Value Function u(t,x)')
    
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(T, X, M, cmap='plasma', alpha=0.7)
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Space x')
    ax2.set_zlabel('m(t,x)')
    ax2.set_title('Density m(t,x)')
    
    # 2D plots
    ax3 = fig.add_subplot(2, 2, 3)
    contour1 = ax3.contourf(T, X, U, levels=15, cmap='viridis')
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Space x')
    ax3.set_title('Value Function Contours')
    
    ax4 = fig.add_subplot(2, 2, 4)
    # Plot final states
    ax4.plot(x_grid, U[-1], 'b-', linewidth=2, label='Final u(t,x)')
    ax4.plot(x_grid, M[-1], 'r-', linewidth=2, label='Final m(t,x)')
    ax4.set_xlabel('Space x')
    ax4.set_ylabel('Value')
    ax4.set_title('Final State')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = "results/clean_mfg_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Clean visualization saved: {output_path}")
    print(f"   📊 Solution shape: {U.shape}")
    
    # Validate solution
    from mfg_pde.config.array_validation import MFGArrays
    
    try:
        arrays = MFGArrays(U_solution=U, M_solution=M, grid_config=grid_config)
        stats = arrays.get_solution_statistics()
        mass_drift = stats['mass_conservation']['mass_drift']
        print(f"   🔍 Mass conservation: {mass_drift:.2e}")
    except Exception as e:
        print(f"   ⚠️ Validation: {e}")
    
    return U, M

if __name__ == "__main__":
    run_clean_visualization()