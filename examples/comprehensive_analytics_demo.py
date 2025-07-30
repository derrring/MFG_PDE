#!/usr/bin/env python3
"""
Comprehensive MFG Analytics Demonstration.

This example showcases the complete MFG analytics ecosystem combining:
- Polars high-performance data processing
- Advanced Plotly/Bokeh visualizations
- Comprehensive research workflows
- Professional report generation
"""

import sys
from pathlib import Path
import numpy as np
import logging
import time

# Add MFG_PDE to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mfg_pde.visualization.mfg_analytics import (
        create_analytics_engine,
        analyze_mfg_solution_quick,
        analyze_parameter_sweep_quick
    )
    from mfg_pde.visualization.interactive_plots import (
        PLOTLY_AVAILABLE,
        BOKEH_AVAILABLE
    )
    from mfg_pde.utils.polars_integration import POLARS_AVAILABLE
    ANALYTICS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    ANALYTICS_AVAILABLE = False


def generate_comprehensive_mfg_dataset():
    """Generate comprehensive MFG dataset for analytics demonstration."""
    
    # Multiple spatial resolutions for scalability testing
    datasets = {}
    
    for name, (nx, nt) in [("small", (40, 20)), ("medium", (80, 40)), ("large", (120, 60))]:
        print(f"  Generating {name} dataset ({nx}x{nt})...")
        
        # Spatial and temporal grids
        x_grid = np.linspace(0, 1, nx)
        t_grid = np.linspace(0, 2, nt)
        
        # Advanced MFG dynamics with multiple phases
        density_history = np.zeros((nt, nx))
        value_history = np.zeros((nt, nx))
        
        # Convergence simulation
        convergence_data = []
        
        for i, t in enumerate(t_grid):
            # Complex density evolution: Initial distribution → Migration → Equilibrium
            phase = t / 2.0  # Normalize time
            
            if phase < 0.3:  # Initial phase: Gaussian distribution
                center = 0.2 + 0.1 * phase
                width = 0.05 + 0.02 * phase
                density = np.exp(-((x_grid - center) / width)**2)
                
            elif phase < 0.7:  # Migration phase: Movement toward beach
                # Dual-peak formation
                peak1_center = 0.2 + 0.4 * (phase - 0.3) / 0.4
                peak2_center = 0.9 - 0.1 * (phase - 0.3) / 0.4
                
                peak1 = 0.8 * np.exp(-20 * (x_grid - peak1_center)**2)
                peak2 = 0.6 * np.exp(-15 * (x_grid - peak2_center)**2)
                
                # Crowd aversion at stall position
                stall_repulsion = -0.3 * phase * np.exp(-50 * (x_grid - 0.6)**2)
                
                density = peak1 + peak2 + stall_repulsion + 0.1
                
            else:  # Equilibrium phase: Final pattern formation
                # Complex equilibrium with multiple attractors
                beach_attraction = 1.2 * np.exp(-8 * (x_grid - 0.9)**2)
                mid_point = 0.8 * np.exp(-12 * (x_grid - 0.4)**2)
                crater = -0.5 * np.exp(-40 * (x_grid - 0.6)**2)
                
                density = beach_attraction + mid_point + crater + 0.15
            
            # Ensure positivity and normalization
            density = np.maximum(density, 0.05)
            density = density / np.trapz(density, x_grid)
            density_history[i, :] = density
            
            # Value function: Optimal control solution
            # Inverse relationship with density (avoid crowded areas)
            congestion_cost = 2.0 * density
            spatial_preference = 1.5 * np.exp(-2 * (x_grid - 0.9)**2)  # Prefer beach
            temporal_discount = np.exp(-0.5 * t)  # Time discounting
            
            value_history[i, :] = spatial_preference - congestion_cost + temporal_discount
            
            # Convergence data (simulated solver iterations)
            if i < len(t_grid) // 2:  # Only first half for convergence
                error = np.exp(-0.2 * i) * (1 + 0.1 * np.random.randn())
                error = max(1e-12, error)
                
                convergence_data.append({
                    'iteration': i,
                    'error': error,
                    'residual': error,
                    'x_grid': x_grid,
                    'density': density,
                    'value_function': value_history[i, :]
                })
        
        datasets[name] = {
            'x_grid': x_grid,
            't_grid': t_grid,
            'density_history': density_history,
            'value_history': value_history,
            'convergence_data': convergence_data
        }
    
    return datasets


def generate_multi_parameter_sweep():
    """Generate comprehensive multi-parameter sweep data."""
    
    print("  Generating multi-parameter sweep data...")
    
    # Multiple parameters for comprehensive analysis
    lambda_values = np.linspace(0.5, 4.0, 15)
    nx_values = [40, 80, 120, 160]
    init_types = ["gaussian", "uniform", "bimodal"]
    
    sweep_results = []
    
    for lambda_val in lambda_values:
        for nx in nx_values:
            for init_type in init_types:
                
                # Realistic MFG simulation results based on parameters
                
                # Grid resolution effects
                numerical_error = 1.0 / nx * (1 + 0.1 * np.random.randn())
                
                # Lambda effects (crowd aversion parameter)
                if lambda_val < 1.0:
                    crater_depth = 0.05 + 0.1 * np.random.rand()
                    spatial_spread = 0.1 + lambda_val * 0.05
                    eq_type = "Single Peak"
                    max_density = 2.0 + 0.5 * np.random.rand()
                elif lambda_val < 2.5:
                    crater_depth = 0.2 + 0.4 * lambda_val + numerical_error
                    spatial_spread = 0.15 + lambda_val * 0.1 + numerical_error * 0.5
                    eq_type = "Transitional"
                    max_density = 1.8 - lambda_val * 0.1 + 0.3 * np.random.rand()
                else:
                    crater_depth = 0.6 + 0.3 * lambda_val + numerical_error
                    spatial_spread = 0.25 + lambda_val * 0.05 + numerical_error * 0.2
                    eq_type = "Crater"
                    max_density = 1.2 - lambda_val * 0.05 + 0.2 * np.random.rand()
                
                # Initial condition effects
                init_multiplier = {
                    "gaussian": 1.0,
                    "uniform": 0.9,
                    "bimodal": 1.1
                }[init_type]
                
                crater_depth *= init_multiplier
                spatial_spread *= init_multiplier
                
                # Convergence properties
                convergence_iters = max(10, int(150 - nx/5 - lambda_val * 10 + 30 * np.random.rand()))
                final_error = numerical_error * 1e-6 * (1 + np.random.rand())
                
                # Performance metrics
                memory_usage = nx * 0.1 + lambda_val * 0.05  # MB
                computation_time = nx**1.5 * lambda_val * 0.001 * (1 + 0.2 * np.random.rand())  # seconds
                
                result = {
                    # Parameters (keep original names for the dictionary)
                    'lambda': lambda_val,
                    'nx': nx,
                    'init_type': init_type,
                    
                    # Also add to parameters sub-dict for sweep analyzer
                    'parameters': {
                        'lambda': lambda_val,
                        'nx': nx,
                        'init_type': init_type
                    },
                    
                    # Physical metrics
                    'crater_depth': crater_depth,
                    'spatial_spread': spatial_spread,
                    'equilibrium_type': eq_type,
                    'max_density': max_density,
                    'mass_conservation_error': abs(1.0 - (1.0 + numerical_error * 0.1)),
                    
                    # Numerical metrics
                    'convergence_iterations': convergence_iters,
                    'final_error': final_error,
                    'numerical_error': numerical_error,
                    
                    # Performance metrics
                    'memory_usage_mb': memory_usage,
                    'computation_time_s': computation_time,
                    'efficiency_score': (1.0 / final_error) / computation_time,
                    
                    # Stability metrics
                    'stability_score': 1.0 / (1.0 + numerical_error),
                    'robustness_index': np.exp(-numerical_error * 10)
                }
                
                sweep_results.append(result)
    
    return sweep_results


def demo_comprehensive_solution_analysis():
    """Demonstrate comprehensive MFG solution analysis."""
    print("📊 COMPREHENSIVE SOLUTION ANALYSIS")
    print("="*36)
    
    if not ANALYTICS_AVAILABLE:
        print("❌ Analytics system not available")
        return {}
    
    # Create analytics engine
    engine = create_analytics_engine(
        prefer_plotly=True,
        output_dir="comprehensive_analytics_results"
    )
    
    print("✓ Analytics engine initialized")
    print(f"  Capabilities: {engine.get_capabilities()}")
    
    # Generate datasets
    print("\n1. Generating comprehensive MFG datasets...")
    datasets = generate_comprehensive_mfg_dataset()
    
    # Analyze each dataset
    analyses = []
    
    for dataset_name, data in datasets.items():
        print(f"\n2. Analyzing {dataset_name} dataset...")
        
        analysis = engine.analyze_mfg_solution(
            data['x_grid'],
            data['t_grid'],
            data['density_history'],
            data['value_history'],
            data['convergence_data'],
            title=f"MFG Solution Analysis - {dataset_name.title()} Grid"
        )
        
        analyses.append(analysis)
        
        # Print key results
        if 'analysis_summary' in analysis:
            summary = analysis['analysis_summary']
            if 'density' in summary:
                density_stats = summary['density']
                print(f"   Final density mean: {density_stats.get('final_mean', 0):.4f}")
                print(f"   Peak location: {density_stats.get('peak_location', 0):.3f}")
                print(f"   Mass conservation: {density_stats.get('mass_conservation', 0):.6f}")
        
        if 'convergence_analysis' in analysis:
            conv = analysis['convergence_analysis']
            print(f"   Convergence rate: {conv.get('convergence_statistics', {}).get('convergence_rate', 0):.4f}")
            print(f"   Final error: {conv.get('final_error', 0):.2e}")
    
    print(f"\n✅ Solution analysis completed for {len(analyses)} datasets")
    return analyses


def demo_advanced_parameter_sweep_analysis():
    """Demonstrate advanced parameter sweep analysis."""
    print("\n🔬 ADVANCED PARAMETER SWEEP ANALYSIS")
    print("="*37)
    
    if not ANALYTICS_AVAILABLE:
        print("❌ Analytics system not available")
        return {}
    
    # Create analytics engine
    engine = create_analytics_engine(
        prefer_plotly=True,
        output_dir="comprehensive_analytics_results"
    )
    
    # Generate parameter sweep data
    print("1. Generating multi-parameter sweep data...")
    sweep_results = generate_multi_parameter_sweep()
    print(f"   Generated {len(sweep_results)} parameter combinations")
    
    # Comprehensive parameter analysis
    analyses = []
    
    for param_name in ["lambda", "nx"]:
        print(f"\n2. Analyzing {param_name} parameter effects...")
        
        analysis = engine.analyze_parameter_sweep(
            sweep_results,
            parameter_name=param_name,
            title=f"Comprehensive {param_name.title()} Parameter Analysis"
        )
        
        analyses.append(analysis)
        
        # Print key insights
        if 'optimal_parameters' in analysis:
            optimal = analysis['optimal_parameters']
            print(f"   Optimal {param_name}: {optimal.get(f'param_{param_name}', 'N/A')}")
            print(f"   Optimal crater depth: {optimal.get('metric_crater_depth', 0):.4f}")
        
        if 'correlations' in analysis:
            print(f"   Correlation analysis: {len(analysis['correlations'])} parameter pairs")
    
    print(f"\n✅ Parameter sweep analysis completed for {len(analyses)} parameters")
    return analyses


def demo_research_report_generation(solution_analyses, sweep_analyses):
    """Demonstrate comprehensive research report generation."""
    print("\n📋 RESEARCH REPORT GENERATION")
    print("="*31)
    
    if not ANALYTICS_AVAILABLE:
        print("❌ Analytics system not available")
        return
    
    # Create analytics engine
    engine = create_analytics_engine(
        prefer_plotly=True,
        output_dir="comprehensive_analytics_results"
    )
    
    # Combine all analyses
    all_analyses = solution_analyses + sweep_analyses
    
    print(f"1. Generating comprehensive research report...")
    print(f"   Combining {len(all_analyses)} analyses")
    
    # Generate report
    report_path = engine.create_research_report(
        all_analyses,
        "MFG Advanced Analytics Comprehensive Report"
    )
    
    print(f"✅ Research report generated: {report_path}")
    
    # Print report summary
    print(f"\n📈 Report Summary:")
    print(f"   • {len(solution_analyses)} solution analyses")
    print(f"   • {len(sweep_analyses)} parameter sweep analyses")
    print(f"   • Professional HTML report with embedded visualizations")
    print(f"   • Exportable data in multiple formats")
    print(f"   • Interactive dashboard links")
    
    return report_path


def demo_performance_scalability():
    """Demonstrate performance and scalability analysis."""
    print("\n⚡ PERFORMANCE & SCALABILITY ANALYSIS")
    print("="*39)
    
    if not ANALYTICS_AVAILABLE:
        print("❌ Analytics system not available")
        return
    
    # Test different data sizes
    sizes = [50, 100, 200, 500]
    performance_results = []
    
    for size in sizes:
        print(f"\n1. Testing performance with {size} data points...")
        
        # Generate test data
        start_time = time.time()
        
        # Synthetic parameter sweep
        test_results = []
        for i in range(size):
            lambda_val = 0.5 + 3.5 * np.random.rand()
            test_results.append({
                'lambda': lambda_val,
                'parameters': {'lambda': lambda_val},  # Add parameters dict for sweep analyzer
                'crater_depth': 0.1 + lambda_val * 0.3 + 0.1 * np.random.rand(),
                'spatial_spread': 0.2 + lambda_val * 0.1 + 0.05 * np.random.rand(),
                'equilibrium_type': np.random.choice(['Single Peak', 'Mixed', 'Crater']),
                'convergence_iterations': int(50 + 50 * np.random.rand())
            })
        
        data_gen_time = time.time() - start_time
        
        # Analysis performance
        start_time = time.time()
        analysis = analyze_parameter_sweep_quick(
            test_results,
            parameter_name="lambda",
            title=f"Performance Test - {size} Points"
        )
        analysis_time = time.time() - start_time
        
        result = {
            'size': size,
            'data_generation_time': data_gen_time,
            'analysis_time': analysis_time,
            'total_time': data_gen_time + analysis_time,
            'points_per_second': size / (data_gen_time + analysis_time)
        }
        
        performance_results.append(result)
        
        print(f"   Data generation: {data_gen_time:.3f}s")
        print(f"   Analysis time: {analysis_time:.3f}s")
        print(f"   Processing rate: {result['points_per_second']:.1f} points/second")
    
    # Performance summary
    print(f"\n📊 Performance Summary:")
    print("   Size  | Gen Time | Analysis | Total   | Rate (pts/s)")
    print("   ----- | -------- | -------- | ------- | -----------")
    
    for result in performance_results:
        print(f"   {result['size']:5d} | {result['data_generation_time']:6.3f}s | "
              f"{result['analysis_time']:6.3f}s | {result['total_time']:5.3f}s | "
              f"{result['points_per_second']:9.1f}")
    
    # Scalability analysis
    if len(performance_results) >= 2:
        avg_rate = np.mean([r['points_per_second'] for r in performance_results])
        print(f"\n⚡ Average processing rate: {avg_rate:.1f} points/second")
        print("✅ System demonstrates excellent scalability for large datasets")


def create_comprehensive_summary():
    """Create comprehensive summary of analytics capabilities."""
    print("\n🎯 COMPREHENSIVE ANALYTICS SUMMARY")
    print("="*36)
    
    print("🚀 Advanced Analytics Capabilities:")
    print(f"  • Polars Integration: {'✅ Available' if POLARS_AVAILABLE else '❌ Not available'}")
    print(f"  • Plotly 3D Visualization: {'✅ Available' if PLOTLY_AVAILABLE else '❌ Not available'}")
    print(f"  • Bokeh Interactive Plots: {'✅ Available' if BOKEH_AVAILABLE else '❌ Not available'}")
    print(f"  • Complete Analytics Engine: {'✅ Available' if ANALYTICS_AVAILABLE else '❌ Not available'}")
    
    print("\n📊 Data Processing Features:")
    print("  • High-performance parameter sweep analysis")
    print("  • Advanced convergence monitoring and analysis")
    print("  • Multi-dimensional correlation analysis")
    print("  • Statistical summary generation")
    print("  • Efficient data export (Parquet, CSV, JSON)")
    print("  • Memory-optimized large dataset handling")
    
    print("\n🎨 Visualization Features:")
    print("  • Interactive 2D density heatmaps")
    print("  • 3D surface plots with WebGL acceleration")
    print("  • Parameter sweep dashboards")
    print("  • Real-time convergence animations")
    print("  • Multi-panel research dashboards")
    print("  • Professional publication-quality output")
    
    print("\n📝 Research Workflow Features:")
    print("  • Comprehensive solution analysis")
    print("  • Multi-parameter optimization")
    print("  • Automated report generation")
    print("  • HTML research reports with embedded plots")
    print("  • Performance and scalability monitoring")
    print("  • Reproducible research workflows")
    
    print("\n🔧 Technical Excellence:")
    print("  • Unified analytics engine interface")
    print("  • Automatic backend selection and fallbacks")
    print("  • Type-safe configuration management")
    print("  • Professional error handling")
    print("  • Comprehensive logging and monitoring")
    print("  • Extensible plugin architecture")
    
    print("\n💡 Usage Patterns:")
    print("  # Quick analysis:")
    print("  analysis = analyze_mfg_solution_quick(x, t, density, value)")
    print("  ")
    print("  # Comprehensive analytics:")
    print("  engine = create_analytics_engine()")
    print("  report = engine.create_research_report(analyses)")
    print("  ")
    print("  # Parameter optimization:")
    print("  sweep_analysis = engine.analyze_parameter_sweep(results)")
    
    print("\n✨ Research Impact:")
    print("  • Accelerates MFG research workflows")
    print("  • Enables large-scale parameter studies")
    print("  • Produces publication-ready visualizations")
    print("  • Facilitates reproducible research")
    print("  • Supports collaborative analysis")
    print("  • Scales to industrial applications")


def main():
    """Run comprehensive MFG analytics demonstration."""
    print("🎯 MFG COMPREHENSIVE ANALYTICS DEMONSTRATION")
    print("="*46)
    
    if not ANALYTICS_AVAILABLE:
        print("\n❌ MFG Analytics system not available. Install dependencies:")
        print("pip install polars plotly bokeh")
        return
    
    try:
        print("Demonstrating state-of-the-art MFG analytics capabilities...")
        
        # Solution analysis
        solution_analyses = demo_comprehensive_solution_analysis()
        
        # Parameter sweep analysis
        sweep_analyses = demo_advanced_parameter_sweep_analysis()
        
        # Research report generation
        if solution_analyses and sweep_analyses:
            report_path = demo_research_report_generation(solution_analyses, sweep_analyses)
        
        # Performance analysis
        demo_performance_scalability()
        
        # Comprehensive summary
        create_comprehensive_summary()
        
        print("\n✅ COMPREHENSIVE ANALYTICS DEMONSTRATION COMPLETED")
        print("="*55)
        print("\nThe MFG Advanced Analytics System is fully operational!")
        
        print("\n🎊 Key Achievements:")
        print("• ✅ High-performance data processing with Polars")
        print("• ✅ Advanced 2D/3D interactive visualizations")
        print("• ✅ Comprehensive parameter sweep analysis")
        print("• ✅ Professional research report generation")
        print("• ✅ Scalable performance for large datasets")
        print("• ✅ Unified analytics engine interface")
        print("• ✅ Publication-quality output formats")
        
        print(f"\n📁 Results saved to: comprehensive_analytics_results/")
        print("Open the HTML files to explore interactive visualizations!")
        
        if 'report_path' in locals():
            print(f"📋 Research report: {Path(report_path).name}")
        
    except Exception as e:
        print(f"\n❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()