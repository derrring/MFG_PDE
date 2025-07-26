#!/usr/bin/env python3
"""
Advanced Jupyter Notebook Demo with Graphics
===========================================

Demonstrates the notebook reporting system with sophisticated mathematical
visualizations, interactive plots, and comprehensive analysis.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mfg_pde.utils.notebook_reporting import (
    create_mfg_research_report,
    create_comparative_analysis
)
from mfg_pde.utils.logging import configure_research_logging, get_logger


def generate_synthetic_mfg_data():
    """Generate realistic synthetic MFG solution data for demonstration."""
    # Spatial and temporal grids
    nx, nt = 50, 30
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    
    # Synthetic value function u(t,x) - more sophisticated pattern
    u = np.zeros((nt, nx))
    for i in range(nt):
        for j in range(nx):
            # Complex pattern with time evolution
            u[i, j] = (0.5 * np.sin(2*np.pi*x[j]) * np.exp(-t[i]) + 
                      0.3 * np.cos(3*np.pi*x[j]) * (1-t[i]) +
                      0.2 * x[j]**2 * (1-t[i])**2)
    
    # Synthetic density m(t,x) - evolving population distribution
    m = np.zeros((nt, nx))
    for i in range(nt):
        # Population concentration that spreads over time
        center = 0.5 + 0.2 * np.sin(2*np.pi*t[i])
        width = 0.1 + 0.05 * t[i]
        m[i, :] = np.exp(-0.5 * ((x - center) / width)**2) / (width * np.sqrt(2*np.pi))
        m[i, :] /= np.trapz(m[i, :], x)  # Normalize
    
    # Convergence history with realistic pattern
    iterations = np.arange(1, 26)
    errors = 1e-2 * np.exp(-0.3 * iterations) + 1e-6 * np.random.exponential(0.5, len(iterations))
    
    # Mass conservation tracking
    mass_history = 1.0 + 1e-5 * np.cumsum(np.random.normal(0, 1, len(iterations)))
    
    # Performance metrics
    timing_data = {
        'total_time': 45.67,
        'assembly_time': 12.34,
        'solve_time': 28.92,
        'postprocess_time': 4.41,
        'memory_peak_mb': 256.7
    }
    
    return {
        'x_grid': x,
        't_grid': t,
        'value_function': u,
        'density_function': m,
        'convergence_errors': errors,
        'mass_conservation': mass_history,
        'timing': timing_data,
        'final_error': errors[-1],
        'converged': True,
        'iterations': len(errors)
    }


def create_single_solver_demo():
    """Create a comprehensive single solver analysis notebook."""
    print("🔬 Creating Single Solver Analysis Notebook...")
    
    # Generate synthetic data
    solver_results = generate_synthetic_mfg_data()
    
    # Problem configuration
    problem_config = {
        'problem_type': 'Mean Field Game - Crowd Motion',
        'domain': '[0,1] × [0,1]',
        'time_horizon': 1.0,
        'num_agents': 10000,
        'running_cost': 'Quadratic congestion cost',
        'terminal_cost': 'Gaussian target distribution',
        'solver_method': 'HJB-GFDM with Tuned QP',
        'mesh_size': (50, 30),
        'tolerance': 1e-6,
        'max_iterations': 100
    }
    
    # Create notebook
    paths = create_mfg_research_report(
        title="Advanced MFG Solver Analysis: Crowd Motion Dynamics",
        solver_results=solver_results,
        problem_config=problem_config,
        output_dir="notebook_demos",
        export_html=True
    )
    
    return paths


def create_comparative_demo():
    """Create a comparative analysis notebook with multiple solvers."""
    print("🔄 Creating Comparative Analysis Notebook...")
    
    # Generate data for multiple "solvers"
    solver_results = {}
    
    # Solver 1: HJB-GFDM
    data1 = generate_synthetic_mfg_data()
    data1['timing']['total_time'] = 45.67
    solver_results['HJB-GFDM Tuned QP'] = data1
    
    # Solver 2: Particle Method (slightly different characteristics)
    data2 = generate_synthetic_mfg_data()
    data2['convergence_errors'] = data2['convergence_errors'] * 1.2  # Slower convergence
    data2['timing']['total_time'] = 38.92
    data2['final_error'] = data2['convergence_errors'][-1]
    solver_results['Particle Collocation'] = data2
    
    # Solver 3: Semi-Lagrangian (different pattern)
    data3 = generate_synthetic_mfg_data()
    data3['convergence_errors'] = data3['convergence_errors'] * 0.8  # Faster convergence
    data3['timing']['total_time'] = 52.31
    data3['final_error'] = data3['convergence_errors'][-1]
    solver_results['Semi-Lagrangian HJB'] = data3
    
    # Problem configuration
    problem_config = {
        'problem_type': 'Mean Field Game - Financial Markets',
        'domain': '[0,1] × [0,2]',
        'time_horizon': 2.0,
        'num_agents': 50000,
        'scenario': 'Portfolio optimization with market impact',
        'volatility': 0.2,
        'risk_aversion': 1.5
    }
    
    # Create comparative notebook
    paths = create_comparative_analysis(
        title="MFG Solver Comparison: Portfolio Optimization Strategies",
        solver_results=solver_results,
        problem_config=problem_config,
        output_dir="notebook_demos",
        export_html=True
    )
    
    return paths


def create_advanced_custom_demo():
    """Create advanced notebook with sophisticated mathematical analysis."""
    print("🎯 Creating Advanced Mathematical Analysis...")
    
    # Generate complex synthetic data with additional sophisticated metrics
    solver_results = generate_synthetic_mfg_data()
    
    # Add advanced mathematical analysis data
    solver_results.update({
        'stability_metrics': {
            'eigenvalues': np.random.exponential(0.1, 10),
            'condition_number': 1247.3,
            'spectral_radius': 0.876
        },
        'advanced_timing': {
            'operations_count': 1.47e6,
            'memory_peak_mb': 512.3,
            'parallelization_efficiency': 0.834
        }
    })
    
    # Enhanced problem configuration with mathematical details
    problem_config = {
        'problem_type': 'Mean Field Game - Advanced Mathematical Analysis',
        'domain': '[0,1] × [0,2]',
        'mathematical_formulation': 'Coupled HJB-FP system with nonlocal terms',
        'numerical_method': 'High-order finite differences + Newton-Krylov',
        'boundary_conditions': 'Periodic with Robin conditions',
        'research_focus': 'Stability analysis and computational efficiency',
        'special_features': 'Eigenvalue analysis, mass conservation tracking'
    }
    
    # Use the single solver report but with enhanced configuration
    paths = create_mfg_research_report(
        title="Advanced MFG Mathematical Analysis: Stability & Performance",
        solver_results=solver_results,
        problem_config=problem_config,
        output_dir="notebook_demos",
        export_html=True
    )
    
    return paths


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("ADVANCED JUPYTER NOTEBOOK DEMONSTRATION")
    print("=" * 80)
    
    # Configure research logging
    configure_research_logging("notebook_demo_session", level="INFO")
    logger = get_logger("notebook_demo")
    
    logger.info("Starting advanced notebook demonstration")
    
    # Create output directory
    Path("notebook_demos").mkdir(exist_ok=True)
    
    # Generate different types of notebooks
    all_paths = {}
    
    try:
        # Single solver analysis
        paths1 = create_single_solver_demo()
        all_paths['single_solver'] = paths1
        if isinstance(paths1, dict):
            logger.info(f"Created single solver notebook: {paths1.get('notebook_path', 'N/A')}")
        else:
            logger.info(f"Created single solver notebook: {paths1}")
        
        # Comparative analysis
        paths2 = create_comparative_demo()
        all_paths['comparative'] = paths2
        if isinstance(paths2, dict):
            logger.info(f"Created comparative notebook: {paths2.get('notebook_path', 'N/A')}")
        else:
            logger.info(f"Created comparative notebook: {paths2}")
        
        # Advanced customized analysis
        paths3 = create_advanced_custom_demo()
        all_paths['advanced'] = paths3
        if isinstance(paths3, dict):
            logger.info(f"Created advanced notebook: {paths3.get('notebook_path', 'N/A')}")
        else:
            logger.info(f"Created advanced notebook: {paths3}")
        
    except Exception as e:
        logger.error(f"Error creating notebooks: {e}")
        # Continue to show summary even if some notebooks failed
        pass
    
    # Summary
    print("\n" + "=" * 80)
    print("NOTEBOOK GENERATION COMPLETE")
    print("=" * 80)
    
    print("\n📋 GENERATED NOTEBOOKS:")
    for notebook_type, paths in all_paths.items():
        print(f"\n{notebook_type.upper().replace('_', ' ')}:")
        if isinstance(paths, dict):
            nb_path = paths.get('notebook_path', 'N/A')
            html_path = paths.get('html_path', 'N/A')
            print(f"  📓 Notebook: {nb_path}")
            print(f"  🌐 HTML:     {html_path}")
            
            # Show file sizes if files exist
            try:
                if nb_path != 'N/A' and Path(nb_path).exists():
                    nb_size = Path(nb_path).stat().st_size / 1024
                    print(f"  📊 Notebook size: {nb_size:.1f}KB")
                if html_path != 'N/A' and Path(html_path).exists():
                    html_size = Path(html_path).stat().st_size / 1024
                    print(f"  📊 HTML size: {html_size:.1f}KB")
            except Exception:
                pass
        else:
            print(f"  📓 Path: {paths}")
            # Try to show file size
            try:
                if Path(str(paths)).exists():
                    size = Path(str(paths)).stat().st_size / 1024
                    print(f"  📊 Size: {size:.1f}KB")
            except Exception:
                pass
    
    print("\n🎯 FEATURES DEMONSTRATED:")
    print("  ✓ Interactive Plotly visualizations")
    print("  ✓ Mathematical LaTeX expressions")
    print("  ✓ Comprehensive solver analysis")
    print("  ✓ Performance metrics and timing")
    print("  ✓ Convergence and stability analysis")
    print("  ✓ Mass conservation tracking")
    print("  ✓ HTML export for browser viewing")
    print("  ✓ Professional research formatting")
    
    print("\n🚀 USAGE:")
    print("  • Open .ipynb files in Jupyter Lab/Notebook")
    print("  • View .html files directly in web browser")
    print("  • All plots are interactive with zoom/pan/hover")
    print("  • Mathematical expressions render with MathJax")
    
    print("\n✨ Ready for advanced mathematical research workflows!")
    
    return all_paths


if __name__ == "__main__":
    results = main()