#!/usr/bin/env python3
"""
Advanced Visualization Example

Demonstrates the comprehensive visualization capabilities for MFG_PDE including
interactive Plotly plots, publication-quality matplotlib figures, monitoring
dashboards, and animation features.
"""

import numpy as np
import sys
import os
import time
from pathlib import Path

# Add the parent directory to the path so we can import mfg_pde
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mfg_pde import MFGProblem, create_fast_solver
from mfg_pde.utils import configure_logging, get_logger
from mfg_pde.visualization import (
    MFGPlotlyVisualizer, MFGBokehVisualizer, create_visualization_manager,
    quick_2d_plot, quick_3d_plot
)


def create_test_problem():
    """Create a test MFG problem with interesting dynamics."""
    
    class VisualizationTestProblem(MFGProblem):
        def __init__(self):
            super().__init__(T=1.0, Nt=30, xmin=0.0, xmax=1.0, Nx=60)
        
        def g(self, x):
            # Multiple peaks terminal condition
            return 0.5 * (np.sin(3 * np.pi * x) + 1) * np.exp(-5 * (x - 0.5)**2)
        
        def rho0(self, x):
            # Bimodal initial distribution
            return (np.exp(-20 * (x - 0.2)**2) + 
                   np.exp(-20 * (x - 0.8)**2) + 
                   0.1 * np.exp(-2 * (x - 0.5)**2))
        
        def f(self, x, u, m):
            # Nonlinear running cost
            return 0.05 * u**2 + 0.02 * m * u + 0.01 * m**2
        
        def sigma(self, x):
            # Variable diffusion
            return 0.1 * (1 + 0.5 * np.sin(2 * np.pi * x))
        
        def H(self, x, p, m):
            return 0.5 * p**2
        
        def dH_dm(self, x, p, m):
            return 0.0
    
    return VisualizationTestProblem()


def generate_sample_data():
    """Generate sample MFG solution data for visualization."""
    logger = get_logger(__name__)
    logger.info("Generating sample MFG solution data")
    
    # Create grids
    x_grid = np.linspace(0, 1, 60)
    t_grid = np.linspace(0, 1, 30)
    X, T = np.meshgrid(x_grid, t_grid)
    
    # Generate synthetic MFG solution with interesting features
    # Value function: wave-like structure evolving in time
    U = (np.sin(2 * np.pi * X) * np.cos(np.pi * X) * 
         np.exp(-2 * T) * (1 + 0.3 * np.sin(4 * np.pi * T)))
    
    # Density: evolving from bimodal to unimodal
    M = (np.exp(-10 * (X - 0.3 + 0.2 * T)**2) + 
         np.exp(-10 * (X - 0.7 - 0.2 * T)**2) * (1 - T) +
         0.5 * np.exp(-5 * (X - 0.5)**2) * T)
    
    # Normalize density at each time step
    for i in range(len(t_grid)):
        M[i, :] = M[i, :] / trapezoid(M[i, :], x_grid)
    
    return U.T, M.T, x_grid, t_grid


def demo_basic_visualization():
    """Demonstrate basic visualization capabilities."""
    print("=" * 60)
    print("BASIC VISUALIZATION DEMONSTRATION")
    print("=" * 60)
    
    logger = get_logger(__name__)
    U, M, x_grid, t_grid = generate_sample_data()
    
    # Create output directory
    output_dir = Path("visualization_output")
    output_dir.mkdir(exist_ok=True)
    
    # Test both backends
    backends = []
    try:
        from mfg_pde.utils.advanced_visualization import PLOTLY_AVAILABLE, MATPLOTLIB_AVAILABLE
        if PLOTLY_AVAILABLE:
            backends.append("plotly")
        if MATPLOTLIB_AVAILABLE:
            backends.append("matplotlib")
    except ImportError:
        pass
    
    if not backends:
        logger.error("No visualization backends available")
        return
    
    for backend in backends:
        logger.info(f"Creating visualizations with {backend} backend")
        
        try:
            visualizer = MFGVisualizer(backend=backend, theme="default")
            
            # Plot complete MFG solution
            fig1 = visualizer.plot_mfg_solution(
                U, M, x_grid, t_grid,
                title=f"MFG Solution Visualization ({backend})",
                save_path=output_dir / f"mfg_solution_{backend}.{'html' if backend == 'plotly' else 'png'}",
                show=False
            )
            
            # Plot solution snapshots at specific times
            t_indices = [0, len(t_grid)//4, len(t_grid)//2, 3*len(t_grid)//4, -1]
            fig2 = visualizer.plot_solution_snapshots(
                U, M, x_grid, t_indices, t_grid,
                title=f"Solution Evolution ({backend})",
                save_path=output_dir / f"solution_snapshots_{backend}.{'html' if backend == 'plotly' else 'png'}",
                show=False
            )
            
            logger.info(f"Successfully created {backend} visualizations")
            
        except Exception as e:
            logger.error(f"Error with {backend} backend: {e}")
    
    print()


def demo_convergence_visualization():
    """Demonstrate convergence history visualization."""
    print("=" * 60)
    print("CONVERGENCE VISUALIZATION DEMONSTRATION")
    print("=" * 60)
    
    logger = get_logger(__name__)
    
    # Generate realistic convergence data
    iterations = 50
    convergence_data = {
        'HJB_residual': [1e-1 * (0.85 ** i) * (1 + 0.1 * np.sin(i * 0.5)) for i in range(iterations)],
        'FP_residual': [8e-2 * (0.82 ** i) * (1 + 0.15 * np.cos(i * 0.3)) for i in range(iterations)],
        'L2_error': [5e-2 * (0.88 ** i) for i in range(iterations)],
        'mass_conservation': [1e-3 * (0.9 ** i) * (1 + 0.05 * np.random.random()) for i in range(iterations)]
    }
    
    output_dir = Path("visualization_output")
    
    # Test both backends for convergence plots
    try:
        from mfg_pde.utils.advanced_visualization import PLOTLY_AVAILABLE, MATPLOTLIB_AVAILABLE
        
        if PLOTLY_AVAILABLE:
            visualizer_plotly = MFGVisualizer(backend="plotly")
            fig_plotly = visualizer_plotly.plot_convergence_history(
                convergence_data,
                title="Convergence History (Interactive)",
                log_scale=True,
                save_path=output_dir / "convergence_interactive.html",
                show=False
            )
            logger.info("Created interactive convergence plot")
        
        if MATPLOTLIB_AVAILABLE:
            visualizer_mpl = MFGVisualizer(backend="matplotlib")
            fig_mpl = visualizer_mpl.plot_convergence_history(
                convergence_data,
                title="Convergence History (Publication Quality)",
                log_scale=True,
                save_path=output_dir / "convergence_publication.png",
                show=False
            )
            logger.info("Created publication-quality convergence plot")
        
        # Quick plot function
        quick_fig = quick_plot_convergence(
            convergence_data,
            backend="auto",
            save_path=output_dir / "convergence_quick.html"
        )
        logger.info("Created quick convergence plot")
        
    except Exception as e:
        logger.error(f"Error in convergence visualization: {e}")
    
    print()


def demo_monitoring_dashboard():
    """Demonstrate real-time monitoring dashboard."""
    print("=" * 60)
    print("MONITORING DASHBOARD DEMONSTRATION")
    print("=" * 60)
    
    logger = get_logger(__name__)
    
    try:
        from mfg_pde.utils.advanced_visualization import PLOTLY_AVAILABLE
        if not PLOTLY_AVAILABLE:
            logger.warning("Dashboard requires Plotly - skipping demonstration")
            return
        
        # Create monitoring dashboard
        dashboard = SolverMonitoringDashboard(update_interval=0.5)
        
        # Simulate solver execution with monitoring
        logger.info("Simulating solver execution with monitoring")
        
        max_iterations = 40
        for i in range(max_iterations):
            # Simulate solver iteration
            time.sleep(0.02)  # Brief delay to simulate computation
            
            # Generate realistic metrics
            hjb_error = 1e-1 * (0.85 ** i) * (1 + 0.1 * np.random.random())
            fp_error = 8e-2 * (0.82 ** i) * (1 + 0.1 * np.random.random())  
            mass_error = 1e-4 * (0.9 ** i) * (1 + 0.05 * np.random.random())
            
            # Add metrics to dashboard
            dashboard.add_metric("HJB_Error", hjb_error, i + 1)
            dashboard.add_metric("FP_Error", fp_error, i + 1)
            dashboard.add_metric("Mass_Conservation", mass_error, i + 1)
            
            # Add performance data
            iteration_time = 0.1 + 0.02 * np.random.random()
            dashboard.add_performance_data("Iteration", iteration_time, 
                                         {"memory_mb": 150 + 10 * np.random.random()})
            
            if i % 10 == 0:
                matrix_time = 0.05 + 0.01 * np.random.random()
                dashboard.add_performance_data("Matrix_Assembly", matrix_time)
        
        # Create dashboard
        output_dir = Path("visualization_output")
        dashboard_fig = dashboard.create_dashboard(
            save_path=output_dir / "monitoring_dashboard.html",
            show=False
        )
        
        logger.info("Monitoring dashboard created successfully")
        
    except Exception as e:
        logger.error(f"Error creating monitoring dashboard: {e}")
    
    print()


def demo_animation_features():
    """Demonstrate animation capabilities."""
    print("=" * 60)
    print("ANIMATION FEATURES DEMONSTRATION")
    print("=" * 60)
    
    logger = get_logger(__name__)
    
    try:
        # Generate time series data for animation
        x_grid = np.linspace(0, 1, 50)
        n_frames = 20
        
        # Create evolving solution
        U_sequence = []
        for t in np.linspace(0, 1, n_frames):
            # Wave that propagates and changes shape
            U_t = (np.sin(2 * np.pi * (x_grid - 0.5 * t)) * 
                   np.exp(-2 * t) * 
                   np.exp(-5 * (x_grid - 0.5)**2 * (1 - t)))
            U_sequence.append(U_t)
        
        output_dir = Path("visualization_output")
        
        # Create animation
        animation = VisualizationUtils.create_animation(
            U_sequence,
            x_grid,
            title="MFG Solution Evolution Animation",
            save_path=output_dir / "solution_animation.html"
        )
        
        logger.info("Solution evolution animation created")
        
    except Exception as e:
        logger.error(f"Error creating animation: {e}")
    
    print()


def demo_interactive_report():
    """Demonstrate interactive report generation."""
    print("=" * 60)
    print("INTERACTIVE REPORT DEMONSTRATION")
    print("=" * 60)
    
    logger = get_logger(__name__)
    
    try:
        from mfg_pde.utils.advanced_visualization import PLOTLY_AVAILABLE
        if not PLOTLY_AVAILABLE:
            logger.warning("Interactive reports require Plotly - skipping demonstration")
            return
        
        # Generate sample analysis results
        U, M, x_grid, t_grid = generate_sample_data()
        
        # Create visualizations for the report
        visualizer = MFGVisualizer(backend="plotly", theme="publication")
        
        # Solution plot
        solution_fig = visualizer.plot_mfg_solution(
            U, M, x_grid, t_grid,
            title="Complete MFG Solution",
            show=False
        )
        
        # Convergence plot
        convergence_data = {
            'Total_Error': [1e-1 * (0.85 ** i) for i in range(30)],
            'HJB_Residual': [5e-2 * (0.88 ** i) for i in range(30)]
        }
        
        convergence_fig = visualizer.plot_convergence_history(
            convergence_data,
            title="Convergence Analysis",
            show=False
        )
        
        # Package results
        results = {
            "MFG Solution": {"figure": solution_fig, "description": "Complete solution visualization"},
            "Convergence Analysis": {"figure": convergence_fig, "description": "Solver convergence metrics"}
        }
        
        # Generate interactive report
        output_dir = Path("visualization_output")
        report_path = VisualizationUtils.save_interactive_report(
            results,
            output_dir / "mfg_analysis_report.html",
            title="MFG Analysis Report - Advanced Visualization Demo"
        )
        
        logger.info(f"Interactive report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"Error creating interactive report: {e}")
    
    print()


def demo_solver_integration():
    """Demonstrate integration with actual MFG solvers."""
    print("=" * 60)
    print("SOLVER INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    logger = get_logger(__name__)
    
    try:
        # Create test problem
        problem = create_test_problem()
        x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx)
        collocation_points = x_coords.reshape(-1, 1)
        
        logger.info("Creating MFG solver for visualization integration")
        
        # Create solver
        solver = create_fast_solver(
            problem=problem,
            solver_type="monitored_particle",
            collocation_points=collocation_points,
            num_particles=500
        )
        
        logger.info(f"Solving MFG problem with {type(solver).__name__}")
        
        # Solve the problem
        result = solver.solve(verbose=False)
        
        # Extract solution data
        if hasattr(result, 'U') and hasattr(result, 'M'):
            U, M = result.U, result.M
        else:
            # Handle tuple result
            U, M = result[0], result[1]
        
        # Create time grid
        t_grid = np.linspace(0, problem.T, problem.Nt)
        
        # Visualize the actual solver result
        output_dir = Path("visualization_output")
        
        visualizer = MFGVisualizer(backend="auto", theme="default")
        
        # Plot the real solution
        fig = visualizer.plot_mfg_solution(
            U, M, x_coords, t_grid,
            title="Real MFG Solver Result",
            save_path=output_dir / "real_solver_result.html",
            show=False
        )
        
        # Plot snapshots at different times
        t_indices = [0, problem.Nt//3, 2*problem.Nt//3, -1]
        snapshots_fig = visualizer.plot_solution_snapshots(
            U, M, x_coords, t_indices, t_grid,
            title="Real Solver Solution Evolution",
            save_path=output_dir / "real_solver_snapshots.html",
            show=False
        )
        
        logger.info("Successfully visualized real solver results")
        
        # Generate convergence data if available
        if hasattr(result, 'convergence_history'):
            convergence_fig = visualizer.plot_convergence_history(
                result.convergence_history,
                title="Real Solver Convergence",
                save_path=output_dir / "real_solver_convergence.html",
                show=False
            )
            logger.info("Visualized real solver convergence history")
        
    except Exception as e:
        logger.error(f"Error in solver integration: {e}")
        logger.debug("This may be expected if solver dependencies are missing")
    
    print()


def run_comprehensive_visualization_demo():
    """Run complete advanced visualization demonstration."""
    print("[DEMO] MFG_PDE ADVANCED VISUALIZATION MODULE")
    print("=" * 80)
    print("This example demonstrates comprehensive visualization capabilities")
    print("including interactive Plotly plots, publication-quality matplotlib")
    print("figures, monitoring dashboards, and animation features.")
    print("=" * 80)
    print()
    
    # Configure logging
    configure_logging(level="INFO", use_colors=True)
    logger = get_logger(__name__)
    
    # Create output directory
    output_dir = Path("visualization_output")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    try:
        demo_basic_visualization()
        demo_convergence_visualization()
        demo_monitoring_dashboard()
        demo_animation_features()
        demo_interactive_report()
        demo_solver_integration()
        
        print("=" * 80)
        print("[COMPLETED] ADVANCED VISUALIZATION DEMONSTRATION")
        print("=" * 80)
        print()
        print("Features Demonstrated:")
        print("• Interactive 3D surface plots with Plotly")
        print("• Publication-quality static plots with matplotlib")
        print("• Real-time monitoring dashboards")
        print("• Solution evolution animations")
        print("• Comprehensive interactive HTML reports")
        print("• Integration with actual MFG solvers")
        print()
        print("Output Files:")
        if output_dir.exists():
            output_files = list(output_dir.glob("*"))
            for file_path in sorted(output_files):
                print(f"  {file_path}")
        print()
        print("Installation tips:")
        print("  For interactive features: pip install plotly")
        print("  For static plots: pip install matplotlib")
        print("  For animations: pip install pillow (with matplotlib)")
        print()
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_comprehensive_visualization_demo()