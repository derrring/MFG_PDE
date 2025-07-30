#!/usr/bin/env python3
"""
Advanced Visualization Demonstration for MFG_PDE.

This example demonstrates state-of-the-art interactive visualizations using:
- Plotly for 3D surfaces and interactive dashboards
- Bokeh for real-time monitoring and custom layouts
- Combined 2D/3D visualization capabilities
- High-quality export options
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging
import time

# Add MFG_PDE to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mfg_pde.visualization.interactive_plots import (
        create_visualization_manager,
        create_plotly_visualizer,
        create_bokeh_visualizer,
        quick_2d_plot,
        quick_3d_plot,
        PLOTLY_AVAILABLE,
        BOKEH_AVAILABLE
    )
    from mfg_pde.utils.polars_integration import (
        create_parameter_sweep_analyzer,
        POLARS_AVAILABLE
    )
except ImportError as e:
    print(f"Import error: {e}")
    PLOTLY_AVAILABLE = BOKEH_AVAILABLE = POLARS_AVAILABLE = False


def generate_synthetic_mfg_data(nx: int = 100, nt: int = 50) -> dict:
    """Generate synthetic MFG simulation data for visualization."""
    
    # Spatial and temporal grids
    x_grid = np.linspace(0, 1, nx)
    t_grid = np.linspace(0, 2, nt)
    X, T = np.meshgrid(x_grid, t_grid)
    
    # Synthetic density evolution (Towel on Beach scenario)
    # Initial Gaussian distribution evolving into crater pattern
    density_history = np.zeros((nt, nx))
    value_history = np.zeros((nt, nx))
    
    for i, t in enumerate(t_grid):
        # Density: evolving from Gaussian to crater pattern
        initial_gaussian = np.exp(-20 * (x_grid - 0.2)**2)
        
        # Crowd aversion creates crater at stall position (0.6)
        stall_effect = -0.8 * np.exp(-50 * (x_grid - 0.6)**2) * (t / 2.0)
        beach_attraction = 0.3 * np.exp(-10 * (x_grid - 0.9)**2) * (t / 2.0)
        
        density = initial_gaussian + stall_effect + beach_attraction + 0.1
        density = np.maximum(density, 0.05)  # Ensure positive
        density = density / np.trapz(density, x_grid)  # Normalize
        
        density_history[i, :] = density
        
        # Value function: optimal control problem solution
        # Higher values where density is low (less crowded)
        value_history[i, :] = 2.0 - 1.5 * density + 0.5 * np.sin(3 * np.pi * x_grid) * np.exp(-t)
    
    # Generate convergence data
    max_iter = 80
    convergence_history = []
    
    for iter_num in range(max_iter):  
        # Simulate convergence with noise
        error = 1.0 * np.exp(-0.1 * iter_num) * (1 + 0.1 * np.random.randn())
        error = max(1e-12, error)
        
        # Current iteration solutions
        current_density = density_history[min(iter_num // 2, nt-1), :]
        current_value = value_history[min(iter_num // 2, nt-1), :]
        
        convergence_history.append({
            'iteration': iter_num,
            'error': error,
            'x_grid': x_grid,
            'density': current_density,
            'value_function': current_value,
            'residual': error
        })
    
    return {
        'x_grid': x_grid,
        't_grid': t_grid,
        'density_history': density_history,
        'value_history': value_history,
        'convergence_history': convergence_history
    }


def generate_parameter_sweep_data(n_sweeps: int = 30) -> list:
    """Generate synthetic parameter sweep results."""
    
    np.random.seed(42)
    lambda_values = np.linspace(0.5, 4.0, n_sweeps)
    sweep_results = []
    
    for lambda_val in lambda_values:
        # Simulate different equilibrium patterns based on lambda
        if lambda_val < 1.0:
            crater_depth = 0.01 + 0.1 * np.random.rand()
            spatial_spread = 0.15 + 0.05 * np.random.rand()
            eq_type = "Single Peak"
        elif lambda_val < 2.5:
            crater_depth = 0.1 + 0.3 * lambda_val + 0.1 * np.random.rand()
            spatial_spread = 0.2 + 0.1 * lambda_val + 0.05 * np.random.rand()
            eq_type = "Mixed"
        else:
            crater_depth = 0.5 + 0.2 * lambda_val + 0.1 * np.random.rand()
            spatial_spread = 0.25 + 0.05 * lambda_val + 0.05 * np.random.rand()
            eq_type = "Crater"
        
        # Additional metrics
        max_density = 2.0 + np.random.rand()
        convergence_iters = max(10, int(100 - lambda_val * 15 + 20 * np.random.rand()))
        
        result = {
            'lambda': lambda_val,
            'crater_depth': crater_depth,
            'spatial_spread': spatial_spread,
            'equilibrium_type': eq_type,
            'max_density': max_density,
            'convergence_iterations': convergence_iters,
            'final_error': 1e-8 * (1 + np.random.rand())
        }
        
        sweep_results.append(result)
    
    return sweep_results


def demo_plotly_visualizations():
    """Demonstrate advanced Plotly visualizations."""
    print("🎨 PLOTLY ADVANCED VISUALIZATIONS")
    print("="*35)
    
    if not PLOTLY_AVAILABLE:
        print("❌ Plotly not available")
        return
    
    # Generate synthetic data
    print("1. Generating synthetic MFG data...")
    mfg_data = generate_synthetic_mfg_data(nx=80, nt=40)
    sweep_data = generate_parameter_sweep_data(25)
    
    # Create Plotly visualizer
    plotly_viz = create_plotly_visualizer()
    print("✓ Plotly visualizer initialized")
    
    # Create output directory
    output_dir = Path("visualization_demo")
    output_dir.mkdir(exist_ok=True)
    
    # 1. 2D Density Evolution Heatmap
    print("\n2. Creating 2D density evolution heatmap...")
    fig_2d = plotly_viz.plot_density_evolution_2d(
        mfg_data['x_grid'], 
        mfg_data['t_grid'],
        mfg_data['density_history'],
        "Towel on Beach: Density Evolution m(t,x)"
    )
    
    plotly_viz.save_figure(fig_2d, output_dir / "density_evolution_2d.html")
    print("   ✓ 2D heatmap saved")
    
    # 2. 3D Density Surface
    print("\n3. Creating 3D density surface...")
    fig_3d_density = plotly_viz.plot_density_surface_3d(
        mfg_data['x_grid'],
        mfg_data['t_grid'], 
        mfg_data['density_history'],
        "3D Density Surface: m(t,x)"
    )
    
    plotly_viz.save_figure(fig_3d_density, output_dir / "density_surface_3d.html")
    print("   ✓ 3D density surface saved")
    
    # 3. 3D Value Function Surface
    print("\n4. Creating 3D value function surface...")
    fig_3d_value = plotly_viz.plot_value_function_3d(
        mfg_data['x_grid'],
        mfg_data['t_grid'],
        mfg_data['value_history'],
        "3D Value Function: u(t,x)"
    )
    
    plotly_viz.save_figure(fig_3d_value, output_dir / "value_function_3d.html")
    print("   ✓ 3D value function surface saved")
    
    # 4. Parameter Sweep Dashboard
    print("\n5. Creating parameter sweep dashboard...")
    fig_dashboard = plotly_viz.create_parameter_sweep_dashboard(
        sweep_data,
        parameter_name="lambda",
        title="Lambda Parameter Sweep Analysis"
    )
    
    plotly_viz.save_figure(fig_dashboard, output_dir / "parameter_sweep_dashboard.html")
    print("   ✓ Parameter sweep dashboard saved")
    
    # 5. Convergence Animation
    print("\n6. Creating convergence animation...")
    convergence_subset = mfg_data['convergence_history'][::2]  # Every 2nd iteration
    fig_animation = plotly_viz.create_convergence_animation(
        convergence_subset,
        "MFG Solver Convergence Animation"
    )
    
    plotly_viz.save_figure(fig_animation, output_dir / "convergence_animation.html")
    print("   ✓ Convergence animation saved")
    
    print(f"\n✅ Plotly visualizations completed")
    print(f"   Files saved in: {output_dir}")
    
    return output_dir


def demo_bokeh_visualizations():
    """Demonstrate advanced Bokeh visualizations."""
    print("\n🎯 BOKEH INTERACTIVE VISUALIZATIONS")
    print("="*36)
    
    if not BOKEH_AVAILABLE:
        print("❌ Bokeh not available")
        return
    
    # Generate data
    print("1. Generating data for Bokeh visualizations...")
    mfg_data = generate_synthetic_mfg_data(nx=60, nt=30)
    
    # Create Bokeh visualizer
    bokeh_viz = create_bokeh_visualizer()
    print("✓ Bokeh visualizer initialized")
    
    # Create output directory
    output_dir = Path("visualization_demo")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Density Heatmap
    print("\n2. Creating Bokeh density heatmap...")
    heatmap = bokeh_viz.plot_density_heatmap(
        mfg_data['x_grid'],
        mfg_data['t_grid'],
        mfg_data['density_history'],
        "Bokeh: MFG Density Heatmap"
    )
    
    bokeh_viz.save_plot(heatmap, output_dir / "bokeh_density_heatmap.html")
    print("   ✓ Bokeh heatmap saved")
    
    # 2. Convergence Monitoring
    print("\n3. Creating convergence monitoring plot...")
    conv_data = mfg_data['convergence_history']
    iterations = np.array([d['iteration'] for d in conv_data])
    errors = np.array([d['error'] for d in conv_data])
    
    conv_plot = bokeh_viz.plot_convergence_monitoring(
        iterations, errors, "Real-time Convergence Monitoring"
    )
    
    bokeh_viz.save_plot(conv_plot, output_dir / "bokeh_convergence.html")
    print("   ✓ Convergence monitoring saved")
    
    # 3. Comprehensive Dashboard
    print("\n4. Creating MFG solution dashboard...")
    final_density = mfg_data['density_history'][-1, :]
    final_value = mfg_data['value_history'][-1, :]
    
    convergence_data = {
        'iterations': iterations,
        'errors': errors,
        'phase_x': np.cumsum(np.diff(errors, prepend=errors[0])),
        'phase_y': errors[:-1] if len(errors) > 1 else errors
    }
    
    dashboard = bokeh_viz.create_mfg_dashboard(
        mfg_data['x_grid'],
        final_density,
        final_value,
        convergence_data,
        "Comprehensive MFG Solution Dashboard"
    )
    
    bokeh_viz.save_plot(dashboard, output_dir / "bokeh_dashboard.html")
    print("   ✓ MFG dashboard saved")
    
    print(f"\n✅ Bokeh visualizations completed")
    
    return output_dir


def demo_unified_visualization_manager():
    """Demonstrate unified visualization manager."""
    print("\n🔄 UNIFIED VISUALIZATION MANAGER")
    print("="*33)
    
    # Create visualization manager
    viz_manager = create_visualization_manager(prefer_plotly=True)
    print("✓ Visualization manager initialized")
    print(f"  Available backends: {viz_manager.get_available_backends()}")
    
    # Generate test data
    mfg_data = generate_synthetic_mfg_data(nx=50, nt=25)
    sweep_data = generate_parameter_sweep_data(20)
    
    output_dir = Path("visualization_demo")
    output_dir.mkdir(exist_ok=True)
    
    # Test automatic backend selection
    print("\n1. Testing automatic backend selection...")
    
    # 2D plot with auto backend
    plot_2d = viz_manager.create_2d_density_plot(
        mfg_data['x_grid'],
        mfg_data['t_grid'],
        mfg_data['density_history'],
        backend="auto",
        title="Auto Backend: 2D Density Plot"
    )
    
    viz_manager.save_plot(plot_2d, output_dir / "unified_2d_plot.html")
    print("   ✓ 2D plot created with auto backend")
    
    # 3D plot (Plotly only)
    if PLOTLY_AVAILABLE:
        plot_3d = viz_manager.create_3d_surface_plot(
            mfg_data['x_grid'],
            mfg_data['t_grid'],
            mfg_data['density_history'],
            data_type="density",
            title="Unified Manager: 3D Surface"
        )
        
        viz_manager.save_plot(plot_3d, output_dir / "unified_3d_plot.html")
        print("   ✓ 3D plot created")
    
    # Parameter sweep dashboard
    dashboard = viz_manager.create_parameter_sweep_dashboard(
        sweep_data,
        parameter_name="lambda",
        backend="auto"
    )
    
    viz_manager.save_plot(dashboard, output_dir / "unified_dashboard.html")
    print("   ✓ Parameter sweep dashboard created")
    
    print(f"\n✅ Unified visualization manager demo completed")


def demo_quick_visualization_functions():
    """Demonstrate quick visualization functions."""
    print("\n⚡ QUICK VISUALIZATION FUNCTIONS")
    print("="*33)
    
    # Generate simple test data
    x_grid = np.linspace(0, 1, 40)
    t_grid = np.linspace(0, 1, 20)
    
    # Simple density pattern
    X, T = np.meshgrid(x_grid, t_grid)
    density_data = np.exp(-5 * ((X - 0.5)**2 + (T - 0.5)**2)) + 0.1
    
    output_dir = Path("visualization_demo")
    output_dir.mkdir(exist_ok=True)
    
    # Quick 2D plot
    print("1. Creating quick 2D plot...")
    quick_2d = quick_2d_plot(x_grid, t_grid, density_data, 
                            title="Quick 2D Visualization", backend="auto")
    
    if hasattr(quick_2d, 'write_html'):  # Plotly
        quick_2d.write_html(str(output_dir / "quick_2d_plot.html"))
    print("   ✓ Quick 2D plot created")
    
    # Quick 3D plot
    if PLOTLY_AVAILABLE:
        print("2. Creating quick 3D plot...")
        quick_3d = quick_3d_plot(x_grid, t_grid, density_data,
                                data_type="density", title="Quick 3D Surface")
        
        quick_3d.write_html(str(output_dir / "quick_3d_plot.html"))
        print("   ✓ Quick 3D plot created")
    
    print(f"\n✅ Quick visualization functions demo completed")


def demo_performance_comparison():
    """Demonstrate performance comparison between backends."""
    print("\n📊 PERFORMANCE COMPARISON")
    print("="*26)
    
    if not (PLOTLY_AVAILABLE and BOKEH_AVAILABLE):
        print("⚠ Both Plotly and Bokeh required for performance comparison")
        return
    
    # Test different data sizes
    sizes = [25, 50, 100]
    results = {"plotly": [], "bokeh": []}
    
    for size in sizes:
        print(f"\nTesting with {size}x{size//2} grid...")
        
        # Generate test data
        x_grid = np.linspace(0, 1, size)
        t_grid = np.linspace(0, 1, size//2)
        X, T = np.meshgrid(x_grid, t_grid)
        density_data = np.sin(2*np.pi*X) * np.cos(2*np.pi*T) + 1
        
        # Test Plotly performance
        if PLOTLY_AVAILABLE:
            start_time = time.time()
            plotly_viz = create_plotly_visualizer()
            fig = plotly_viz.plot_density_evolution_2d(x_grid, t_grid, density_data)
            plotly_time = time.time() - start_time
            results["plotly"].append(plotly_time)
            print(f"  Plotly: {plotly_time:.4f}s")
        
        # Test Bokeh performance  
        if BOKEH_AVAILABLE:
            start_time = time.time()
            bokeh_viz = create_bokeh_visualizer()
            plot = bokeh_viz.plot_density_heatmap(x_grid, t_grid, density_data)
            bokeh_time = time.time() - start_time
            results["bokeh"].append(bokeh_time)
            print(f"  Bokeh: {bokeh_time:.4f}s")
    
    # Summary
    print(f"\n📈 Performance Summary:")
    if results["plotly"] and results["bokeh"]:
        avg_plotly = np.mean(results["plotly"])
        avg_bokeh = np.mean(results["bokeh"])
        print(f"  Average Plotly time: {avg_plotly:.4f}s")
        print(f"  Average Bokeh time: {avg_bokeh:.4f}s")
        
        if avg_plotly < avg_bokeh:
            print(f"  Plotly is {avg_bokeh/avg_plotly:.1f}x faster")
        else:
            print(f"  Bokeh is {avg_plotly/avg_bokeh:.1f}x faster")


def create_visualization_summary(output_dir: Path):
    """Create comprehensive summary of visualization capabilities."""
    print("\n📋 VISUALIZATION SYSTEM SUMMARY")
    print("="*34)
    
    print("🎨 Available Libraries:")
    print(f"  • Plotly: {'✅ Available' if PLOTLY_AVAILABLE else '❌ Not available'}")
    print(f"  • Bokeh: {'✅ Available' if BOKEH_AVAILABLE else '❌ Not available'}")
    print(f"  • Polars Integration: {'✅ Available' if POLARS_AVAILABLE else '❌ Not available'}")
    
    print("\n🚀 Visualization Capabilities:")
    print("  • Interactive 2D density heatmaps")
    print("  • 3D surface plots for density and value functions")
    print("  • Parameter sweep analysis dashboards")
    print("  • Real-time convergence monitoring")
    print("  • Animated convergence visualizations")
    print("  • Multi-panel solution dashboards")
    print("  • High-quality export (HTML, PNG, PDF, SVG)")
    
    print("\n🔧 Technical Features:")
    print("  • Automatic backend selection")
    print("  • Fallback mechanism between libraries")
    print("  • Unified visualization manager interface")
    print("  • Quick visualization functions")
    print("  • Custom hover tooltips and interactions")
    print("  • Professional publication-quality output")
    
    if output_dir.exists():
        files = list(output_dir.glob("*.html"))
        print(f"\n📁 Generated Files ({len(files)} total):")
        for file in sorted(files):
            print(f"  • {file.name}")
    
    print("\n💡 Usage Examples:")
    print("  # Quick 2D plot:")
    print("  plot = quick_2d_plot(x_grid, t_grid, density_data)")
    print("  ")
    print("  # Unified manager:")
    print("  viz = create_visualization_manager()")
    print("  plot = viz.create_3d_surface_plot(x, t, data)")
    print("  ")
    print("  # Advanced dashboard:")
    print("  dashboard = viz.create_parameter_sweep_dashboard(results)")
    
    print(f"\n✨ Advanced Features:")
    print("  • WebGL-accelerated 3D rendering")
    print("  • Interactive zoom, pan, and rotation")
    print("  • Customizable color schemes and palettes")
    print("  • Animation controls and time sliders")
    print("  • Multi-format export capabilities")
    print("  • Responsive layouts for different screen sizes")


def main():
    """Run complete advanced visualization demonstration."""
    print("🎨 MFG_PDE ADVANCED VISUALIZATION SYSTEM")
    print("="*42)
    
    if not (PLOTLY_AVAILABLE or BOKEH_AVAILABLE):
        print("\n❌ No visualization libraries available. Install with:")
        print("pip install plotly bokeh")
        return
    
    try:
        print("Demonstrating state-of-the-art MFG visualizations...")
        
        # Plotly demonstrations
        output_dir = demo_plotly_visualizations()
        
        # Bokeh demonstrations
        demo_bokeh_visualizations()
        
        # Unified manager
        demo_unified_visualization_manager()
        
        # Quick functions
        demo_quick_visualization_functions()
        
        # Performance comparison
        demo_performance_comparison()
        
        # Summary
        create_visualization_summary(output_dir)
        
        print("\n✅ ADVANCED VISUALIZATION DEMONSTRATION COMPLETED")
        print("="*50)
        print(f"\nAdvanced visualization system is ready!")
        print("Key benefits:")
        print("• Professional-quality interactive plots")
        print("• 2D and 3D visualization capabilities")
        print("• Real-time monitoring and animation")
        print("• Multiple export formats")
        print("• Unified interface with automatic fallbacks")
        print("• Optimized performance for large datasets")
        
        if output_dir and output_dir.exists():
            print(f"\nDemo visualizations saved to: {output_dir}")
            print("Open the HTML files in your browser to explore!")
        
    except Exception as e:
        print(f"\n❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()