#!/usr/bin/env python3
"""
Interactive Research Notebook Generation Example
==============================================

This example demonstrates how to generate comprehensive research reports
as interactive Jupyter notebooks with Plotly visualizations and LaTeX
mathematical expressions for Mean Field Games analysis.

Features demonstrated:
- Automatic notebook generation with professional formatting
- Interactive Plotly visualizations embedded in notebooks
- LaTeX mathematical notation for research-quality documentation
- HTML export for easy sharing and presentation
- Comparative analysis across multiple methods
"""

import os
import sys
from pathlib import Path

import numpy as np

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.utils.integration import trapezoid
from mfg_pde.utils.logging import configure_logging, get_logger
from mfg_pde.utils.notebook_reporting import (
    MFGNotebookReporter,
    create_comparative_analysis,
    create_mfg_research_report,
)


def generate_sample_data():
    """Generate sample MFG solution data for demonstration."""
    print("Generating sample MFG solution data...")

    # Create a simple MFG problem
    problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=30, sigma=0.2, coefCT=0.05)

    # Simple solver setup for demonstration
    collocation_points = np.linspace(0, 1, 12).reshape(-1, 1)
    boundary_conditions = BoundaryConditions(type="no_flux")

    solver = ParticleCollocationSolver(
        problem=problem,
        collocation_points=collocation_points,
        boundary_conditions=boundary_conditions,
        num_particles=500,
        max_newton_iterations=15,
        newton_tolerance=1e-4,
    )

    # Solve the problem
    try:
        U, M, info = solver.solve(max_picard_iterations=10, verbose=True)

        # Package results for notebook generation
        solver_results = {
            'U': U,
            'M': M,
            'convergence_info': info,
            'x_grid': problem.x_grid,
            't_grid': problem.t_grid,
            'solver_type': 'Particle-Collocation',
            'num_particles': 500,
        }

        problem_config = {
            'xmin': problem.xmin,
            'xmax': problem.xmax,
            'Nx': problem.Nx,
            'T': problem.T,
            'Nt': problem.Nt,
            'sigma': problem.sigma,
            'coefCT': problem.coefCT,
            'boundary_conditions': 'no-flux',
        }

        return solver_results, problem_config

    except Exception as e:
        print(f"Solver failed: {e}")
        # Generate synthetic data for demonstration
        return generate_synthetic_data(problem)


def generate_synthetic_data(problem):
    """Generate synthetic data for demonstration when solver fails."""
    print("Generating synthetic demonstration data...")

    # Create synthetic solution data
    X, T = np.meshgrid(problem.x_grid, problem.t_grid)

    # Synthetic value function - parabolic profile evolving in time
    U = np.sin(np.pi * X) * np.exp(-0.5 * T) + 0.1 * X * (1 - X) * T

    # Synthetic density - Gaussian that spreads over time
    M = np.exp(-5 * (X - 0.5 - 0.1 * np.sin(2 * np.pi * T)) ** 2) / np.sqrt(np.pi / 5)
    # Normalize each time slice
    for t in range(M.shape[0]):
        M[t, :] = M[t, :] / trapezoid(M[t, :], problem.x_grid)

    # Synthetic convergence info
    convergence_info = {
        'converged': True,
        'iterations': 8,
        'final_error': 2.3e-6,
        'error_history': [1e-1, 5e-2, 2e-2, 8e-3, 3e-3, 1.2e-3, 4e-4, 1.5e-4, 2.3e-6],
        'solver_info': 'Synthetic data for demonstration',
    }

    solver_results = {
        'U': U,
        'M': M,
        'convergence_info': convergence_info,
        'x_grid': problem.x_grid,
        't_grid': problem.t_grid,
        'solver_type': 'Synthetic Demo Data',
        'synthetic': True,
    }

    problem_config = {
        'xmin': problem.xmin,
        'xmax': problem.xmax,
        'Nx': problem.Nx,
        'T': problem.T,
        'Nt': problem.Nt,
        'sigma': problem.sigma,
        'coefCT': problem.coefCT,
        'boundary_conditions': 'no-flux',
        'note': 'Synthetic data for demonstration',
    }

    return solver_results, problem_config


def demonstrate_single_report():
    """Demonstrate creating a single research report notebook."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION 1: Single Research Report Generation")
    print("=" * 60)

    # Generate sample data
    solver_results, problem_config = generate_sample_data()

    analysis_metadata = {
        'researcher': 'MFG_PDE User',
        'institution': 'Research Institution',
        'project': 'Mean Field Games Analysis',
        'experiment_id': 'MFG_001',
        'notes': 'Demonstration of interactive notebook reporting',
    }

    # Create the research report
    print("\nGenerating interactive research notebook...")

    try:
        result_paths = create_mfg_research_report(
            title="Interactive MFG Analysis: Particle-Collocation Method",
            solver_results=solver_results,
            problem_config=problem_config,
            output_dir="research_reports",
            export_html=True,
        )

        print(f"✓ Research notebook created: {result_paths['notebook']}")
        if 'html' in result_paths:
            print(f"✓ HTML export created: {result_paths['html']}")

        return result_paths

    except Exception as e:
        print(f"Error creating research report: {e}")
        import traceback

        traceback.print_exc()
        return None


def demonstrate_comparative_report():
    """Demonstrate creating a comparative analysis notebook."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION 2: Comparative Analysis Report")
    print("=" * 60)

    # Generate sample data for multiple methods
    solver_results_1, problem_config = generate_sample_data()

    # Create variations for comparison (simulated different methods)
    solver_results_2 = solver_results_1.copy()
    solver_results_2['U'] = solver_results_1['U'] * 1.02 + 0.01  # Slight variation
    solver_results_2['M'] = solver_results_1['M'] * 0.98  # Slight variation
    solver_results_2['convergence_info'] = {
        'converged': True,
        'iterations': 12,
        'final_error': 5.1e-6,
        'error_history': [1.2e-1, 6e-2, 2.5e-2, 1e-2, 4e-3, 1.8e-3, 7e-4, 3e-4, 1.2e-4, 5e-5, 2e-5, 5.1e-6],
        'solver_info': 'Alternative method simulation',
    }
    solver_results_2['solver_type'] = 'Alternative Method'

    solver_results_3 = solver_results_1.copy()
    solver_results_3['U'] = solver_results_1['U'] * 0.95 + np.random.normal(0, 0.001, solver_results_1['U'].shape)
    solver_results_3['M'] = solver_results_1['M'] * 1.05
    solver_results_3['convergence_info'] = {
        'converged': True,
        'iterations': 6,
        'final_error': 8.7e-7,
        'error_history': [8e-2, 3e-2, 1e-2, 3e-3, 8e-4, 8.7e-7],
        'solver_info': 'Fast convergence method simulation',
    }
    solver_results_3['solver_type'] = 'Fast Method'

    # Create comparative analysis
    results_dict = {
        'Particle-Collocation': solver_results_1,
        'Alternative Method': solver_results_2,
        'Fast Convergence Method': solver_results_3,
    }

    print("\nGenerating comparative analysis notebook...")

    try:
        result_paths = create_comparative_analysis(
            results_dict=results_dict,
            title="MFG Methods Comparison: Performance Analysis",
            output_dir="research_reports",
            export_html=True,
        )

        print(f"✓ Comparative notebook created: {result_paths['notebook']}")
        if 'html' in result_paths:
            print(f"✓ HTML export created: {result_paths['html']}")

        return result_paths

    except Exception as e:
        print(f"Error creating comparative report: {e}")
        import traceback

        traceback.print_exc()
        return None


def demonstrate_advanced_customization():
    """Demonstrate advanced notebook customization features."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION 3: Advanced Customization")
    print("=" * 60)

    # Generate data
    solver_results, problem_config = generate_sample_data()

    # Create custom sections
    custom_sections = [
        {
            'title': 'Economic Interpretation',
            'description': 'Analysis of the economic meaning of the MFG solution.',
            'markdown': '''
### Economic Context

This Mean Field Game represents a large population of economic agents making optimal decisions. Key insights:

- **Value Function**: Represents expected utility for each agent at position $x$ and time $t$
- **Population Density**: Shows the equilibrium distribution of agents in the economy
- **Nash Equilibrium**: The solution represents a Nash equilibrium where no agent can improve by deviating

#### Policy Implications

The results suggest optimal strategies for:
1. Resource allocation policies
2. Market regulation timing
3. Agent coordination mechanisms
''',
            'code': '''
# Economic analysis code
print("Economic Analysis:")
print("=" * 30)

if 'U' in locals():
    # Analyze value function properties
    value_range = np.max(U) - np.min(U)
    print(f"Value function range: {value_range:.4f}")

    # Find optimal regions (high value areas)
    optimal_threshold = np.percentile(U, 80)
    optimal_regions = np.where(U[-1, :] > optimal_threshold)[0]
    print(f"Optimal regions (80th percentile): {len(optimal_regions)} grid points")

    if 'x_grid' in locals():
        if len(optimal_regions) > 0:
            print(f"Optimal region span: x ∈ [{x_grid[optimal_regions[0]]:.3f}, {x_grid[optimal_regions[-1]]:.3f}]")

print("\\nPolicy recommendations based on solution characteristics...")
''',
        },
        {
            'title': 'Numerical Method Details',
            'description': 'Technical details of the numerical implementation.',
            'code': '''
# Technical implementation details
print("Numerical Method Implementation:")
print("=" * 40)

print("Discretization Parameters:")
print(f"- Spatial grid points: {len(x_grid) if 'x_grid' in locals() else 'N/A'}")
print(f"- Temporal grid points: {len(t_grid) if 't_grid' in locals() else 'N/A'}")
print(f"- Grid spacing dx: {x_grid[1] - x_grid[0]:.6f}" if 'x_grid' in locals() and len(x_grid) > 1 else "")
print(f"- Time step dt: {t_grid[1] - t_grid[0]:.6f}" if 't_grid' in locals() and len(t_grid) > 1 else "")

print("\\nSolver Configuration:")
if 'convergence_info' in locals() and isinstance(convergence_info, dict):
    for key, value in convergence_info.items():
        if key != 'error_history':
            print(f"- {key}: {value}")

print("\\nStability Analysis:")
if 'x_grid' in locals() and 't_grid' in locals() and len(x_grid) > 1 and len(t_grid) > 1:
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]
    cfl_number = dt / (dx**2)  # For diffusion-like equation
    print(f"- CFL-like number: {cfl_number:.6f}")
    if cfl_number < 0.5:
        print("  Status: ✓ Stable regime")
    else:
        print("  Status: ⚠ Check stability")
''',
        },
    ]

    # Create reporter with advanced settings
    reporter = MFGNotebookReporter(
        output_dir="research_reports", enable_latex=True, plotly_renderer="notebook", template_style="research"
    )

    print("\nGenerating advanced customized notebook...")

    try:
        notebook_path = reporter.create_research_report(
            title="Advanced MFG Analysis: Economic and Technical Perspectives",
            solver_results=solver_results,
            problem_config=problem_config,
            analysis_metadata={
                'research_focus': 'Economic applications of Mean Field Games',
                'technical_level': 'Advanced',
                'target_audience': 'Researchers and practitioners',
                'custom_analysis': 'Economic interpretation and numerical details',
            },
            custom_sections=custom_sections,
        )

        print(f"✓ Advanced notebook created: {notebook_path}")

        # Export to HTML
        try:
            html_path = reporter.export_to_html(notebook_path)
            print(f"✓ HTML export created: {html_path}")
        except Exception as e:
            print(f"HTML export failed: {e}")

        return notebook_path

    except Exception as e:
        print(f"Error creating advanced report: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Main demonstration function."""
    print("Interactive Research Notebook Generation Demo")
    print("=" * 60)
    print("This example demonstrates generating professional research reports")
    print("as interactive Jupyter notebooks with Plotly visualizations.")
    print()

    # Configure logging
    configure_logging(level="INFO", use_colors=True)
    logger = get_logger(__name__)

    # Check dependencies
    try:
        import nbformat

        import plotly

        print("✓ All required dependencies available")
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install nbformat jupyter plotly")
        return

    # Create output directory
    output_dir = Path("research_reports")
    output_dir.mkdir(exist_ok=True)

    results = {}

    # Run demonstrations
    try:
        # Demonstration 1: Single report
        result1 = demonstrate_single_report()
        if result1:
            results['single_report'] = result1

        # Demonstration 2: Comparative analysis
        result2 = demonstrate_comparative_report()
        if result2:
            results['comparative_report'] = result2

        # Demonstration 3: Advanced customization
        result3 = demonstrate_advanced_customization()
        if result3:
            results['advanced_report'] = result3

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback

        traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("DEMONSTRATION SUMMARY")
    print("=" * 60)

    if results:
        print("✓ Successfully generated research notebooks:")
        for report_type, paths in results.items():
            print(f"\n{report_type.replace('_', ' ').title()}:")
            if isinstance(paths, dict):
                for file_type, path in paths.items():
                    print(f"  {file_type}: {path}")
            else:
                print(f"  notebook: {paths}")

        print(f"\nAll reports saved in: {output_dir}")
        print("\nTo view the notebooks:")
        print("1. Install Jupyter: pip install jupyter")
        print("2. Start Jupyter: jupyter notebook")
        print("3. Navigate to the research_reports directory")
        print("4. Open any .ipynb file for interactive analysis")

        print("\nHTML files can be opened directly in any web browser")
        print("for viewing interactive plots without Jupyter.")

    else:
        print("ERROR: No reports were successfully generated")
        print("Check the error messages above for troubleshooting")

    print("\n" + "=" * 60)
    print("Notebook reporting demonstration completed!")


if __name__ == "__main__":
    main()
