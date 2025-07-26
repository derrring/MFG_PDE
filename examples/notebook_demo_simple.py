#!/usr/bin/env python3
"""
Simple Notebook Demo - Quick Generation
=====================================

Creates a single comprehensive notebook with advanced graphics.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mfg_pde.utils.notebook_reporting import create_mfg_research_report
from mfg_pde.utils.logging import configure_research_logging, get_logger


def generate_demonstration_data():
    """Generate sophisticated demonstration data for the notebook."""
    # Create realistic spatial-temporal grids
    nx, nt = 60, 40
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 2, nt)
    
    # Generate sophisticated value function with multiple features
    u = np.zeros((nt, nx))
    for i, time in enumerate(t):
        for j, space in enumerate(x):
            # Complex evolving pattern
            wave1 = 0.4 * np.sin(3*np.pi*space) * np.exp(-0.5*time)
            wave2 = 0.3 * np.cos(2*np.pi*space) * (2-time)/2
            trend = 0.2 * space**2 * (2-time)**2
            noise = 0.05 * np.sin(10*np.pi*space) * np.sin(5*np.pi*time)
            u[i, j] = wave1 + wave2 + trend + noise
    
    # Generate population density with crowd motion behavior
    m = np.zeros((nt, nx))
    for i, time in enumerate(t):
        # Moving crowd with dispersion
        center1 = 0.3 + 0.2 * time/2  # Moving right
        center2 = 0.8 - 0.1 * time/2  # Moving left
        width = 0.08 + 0.02 * time/2  # Increasing dispersion
        
        # Two-peak distribution
        peak1 = np.exp(-0.5 * ((x - center1) / width)**2)
        peak2 = 0.6 * np.exp(-0.5 * ((x - center2) / (width*0.8))**2)
        
        m[i, :] = peak1 + peak2
        m[i, :] /= np.trapz(m[i, :], x)  # Normalize to ensure mass conservation
    
    # Realistic convergence pattern
    iterations = np.arange(1, 31)
    base_decay = 1e-2 * np.exp(-0.25 * iterations)
    oscillations = 0.3 * base_decay * np.sin(0.8 * iterations)
    noise = 1e-6 * np.random.exponential(0.5, len(iterations))
    convergence_errors = base_decay + oscillations + noise
    
    # Mass conservation with realistic drift
    mass_history = 1.0 + 1e-4 * np.cumsum(np.random.normal(0, 0.5, len(iterations)))
    
    return {
        'x_grid': x,
        't_grid': t,
        'value_function': u,
        'density_function': m,
        'convergence_errors': convergence_errors,
        'mass_conservation': mass_history,
        'timing': {
            'total_time': 67.89,
            'assembly_time': 23.45,
            'solve_time': 38.76,
            'postprocess_time': 5.68,
            'memory_peak_mb': 384.2
        },
        'final_error': convergence_errors[-1],
        'converged': True,
        'iterations': len(convergence_errors)
    }


def main():
    """Create advanced demonstration notebook."""
    print("🚀 Creating Advanced Notebook with Graphics Demo")
    print("=" * 60)
    
    # Configure logging
    configure_research_logging("notebook_graphics_demo", level="INFO")
    logger = get_logger("notebook_graphics_demo")
    
    # Generate sophisticated demo data
    logger.info("Generating sophisticated demonstration data")
    solver_results = generate_demonstration_data()
    
    # Enhanced problem configuration
    problem_config = {
        'title': 'Advanced MFG Analysis: Interactive Graphics Demo',
        'problem_type': 'Mean Field Game - Sophisticated Crowd Dynamics',
        'domain': '[0,1] × [0,2]',
        'time_horizon': 2.0,
        'num_agents': 25000,
        'application': 'Smart city pedestrian flow optimization',
        'mathematical_model': 'Coupled HJB-FP system with obstacle avoidance',
        'numerical_method': 'High-order GFDM + Particle collocation',
        'mesh_details': f"{solver_results['value_function'].shape} grid points",
        'boundary_conditions': 'Periodic with Neumann flux constraints',
        'special_features': [
            'Interactive 3D surface plots',
            'Real-time convergence monitoring', 
            'Mass conservation analysis',
            'Mathematical LaTeX expressions',
            'Professional research formatting'
        ]
    }
    
    # Create the notebook
    logger.info("Creating advanced interactive notebook")
    paths = create_mfg_research_report(
        title="Advanced MFG Interactive Analysis: Graphics Demo",
        solver_results=solver_results,
        problem_config=problem_config,
        output_dir="demo_output",
        export_html=False  # Skip HTML export to avoid the path issue
    )
    
    # Show results
    print(f"\n✅ NOTEBOOK CREATED SUCCESSFULLY!")
    print(f"📓 Location: {paths}")
    
    if Path(str(paths)).exists():
        size_kb = Path(str(paths)).stat().st_size / 1024
        print(f"📊 Size: {size_kb:.1f} KB")
        
        print(f"\n🎯 FEATURES INCLUDED:")
        print(f"  ✓ Interactive 3D Plotly visualizations")
        print(f"  ✓ Mathematical LaTeX expressions") 
        print(f"  ✓ {solver_results['value_function'].shape[0]}×{solver_results['value_function'].shape[1]} solution grid")
        print(f"  ✓ {len(solver_results['convergence_errors'])} convergence iterations")
        print(f"  ✓ Mass conservation analysis")
        print(f"  ✓ Professional research formatting")
        print(f"  ✓ Export capabilities for HTML/PDF/slides")
        
        print(f"\n🚀 USAGE:")
        print(f"  • Open with: jupyter lab {paths}")
        print(f"  • Or: jupyter notebook {paths}")
        print(f"  • All plots are interactive (zoom, pan, hover)")
        print(f"  • Mathematical expressions render with MathJax")
        print(f"  • Export options available in Jupyter interface")
        
        print(f"\n✨ Ready for advanced mathematical research workflows!")
        
        return str(paths)
    else:
        print("❌ Notebook creation failed")
        return None


if __name__ == "__main__":
    result = main()