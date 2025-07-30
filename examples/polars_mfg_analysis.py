#!/usr/bin/env python3
"""
Polars-powered MFG Analysis Example.

This example demonstrates high-performance data analysis for MFG simulations
using Polars, including parameter sweeps, convergence analysis, and 
efficient data export/import.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import logging

# Add MFG_PDE to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mfg_pde.utils.polars_integration import (
        create_mfg_dataframe,
        create_parameter_sweep_analyzer,
        create_time_series_analyzer,
        create_data_exporter,
        benchmark_polars_vs_pandas,
        POLARS_AVAILABLE
    )
    import polars as pl
except ImportError:
    POLARS_AVAILABLE = False
    print("Polars not available. Install with: pip install polars")


def generate_synthetic_mfg_results(n_runs: int = 100) -> list:
    """Generate synthetic MFG parameter sweep results for demonstration."""
    
    np.random.seed(42)
    results = []
    
    # Parameter ranges
    lambda_values = np.linspace(0.5, 5.0, 20)
    nx_values = [40, 80, 160, 320]
    init_types = ["gaussian", "uniform", "bimodal"]
    
    for i in range(n_runs):
        # Random parameter selection
        lambda_val = np.random.choice(lambda_values)
        nx = np.random.choice(nx_values)
        init_type = np.random.choice(init_types)
        
        # Simulate MFG results based on parameters
        # Higher lambda creates more dispersed patterns
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
        
        # Convergence metrics (higher resolution = better convergence)
        convergence_iterations = max(10, int(100 - nx/10 + 20 * np.random.rand()))
        final_error = 1e-8 * (1 + np.random.rand())
        
        # Synthetic final density
        x_grid = np.linspace(0, 1, nx)
        if eq_type == "Single Peak":
            final_density = 2.0 * np.exp(-8 * (x_grid - 0.6)**2)
        elif eq_type == "Mixed":
            peak1 = 1.2 * np.exp(-5 * (x_grid - 0.45)**2)
            peak2 = 1.3 * np.exp(-5 * (x_grid - 0.75)**2)
            final_density = peak1 + peak2 + 0.3
        else:  # Crater
            peak1 = 1.5 * np.exp(-3 * (x_grid - 0.3)**2)
            peak2 = 1.4 * np.exp(-3 * (x_grid - 0.9)**2)
            crater = -0.4 * np.exp(-10 * (x_grid - 0.6)**2)
            final_density = peak1 + peak2 + crater + 0.4
            final_density = np.maximum(final_density, 0.05)
        
        # Normalize
        final_density = final_density / np.trapz(final_density, x_grid)
        
        result = {
            "run_id": i,
            "parameters": {
                "lambda": lambda_val,
                "nx": nx,
                "init_type": init_type
            },
            "crater_depth": crater_depth,
            "spatial_spread": spatial_spread,
            "equilibrium_type": eq_type,
            "convergence_iterations": convergence_iterations,
            "final_error": final_error,
            "density_at_stall": final_density[np.argmin(np.abs(x_grid - 0.6))],
            "max_density": np.max(final_density),
            "x_grid": x_grid,
            "final_density": final_density
        }
        
        results.append(result)
    
    return results


def demo_parameter_sweep_analysis():
    """Demonstrate parameter sweep analysis with Polars."""
    print("🚀 PARAMETER SWEEP ANALYSIS WITH POLARS")
    print("="*45)
    
    if not POLARS_AVAILABLE:
        print("❌ Polars not available")
        return
    
    # Generate synthetic results
    print("1. Generating synthetic MFG parameter sweep results...")
    results = generate_synthetic_mfg_results(200)
    print(f"   Generated {len(results)} simulation results")
    
    # Create analyzer and DataFrame
    analyzer = create_parameter_sweep_analyzer()
    start_time = time.time()
    df = analyzer.create_sweep_dataframe(results)
    processing_time = time.time() - start_time
    
    print(f"✓ Created MFG DataFrame in {processing_time:.4f}s")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    
    # Show sample data
    print("\n2. Sample data:")
    print(df.head(3))
    
    # Analyze parameter effects
    print("\n3. Analyzing parameter effects...")
    parameter_cols = ["param_lambda", "param_nx"]
    metric_cols = ["metric_crater_depth", "metric_spatial_spread", "density_mean"]
    
    effects = analyzer.analyze_parameter_effects(df, parameter_cols, metric_cols)
    print(f"✓ Parameter effects analysis completed")
    print(f"  Found {len(effects)} parameter-metric combinations")
    
    # Show sample effects
    print("\nSample parameter effects:")
    sample_effects = effects.filter(pl.col("parameter") == "param_lambda").head(5)
    print(sample_effects)
    
    # Find optimal parameters
    print("\n4. Finding optimal parameters...")
    optimal = analyzer.find_optimal_parameters(df, "metric_crater_depth", minimize=True)
    print(f"✓ Optimal parameters for minimum crater depth:")
    print(f"  Lambda: {optimal.get('param_lambda', 'N/A'):.3f}")
    print(f"  Nx: {optimal.get('param_nx', 'N/A')}")
    print(f"  Crater depth: {optimal.get('metric_crater_depth', 'N/A'):.6f}")
    
    return df, analyzer


def demo_convergence_analysis():
    """Demonstrate convergence analysis with time series."""
    print("\n📈 CONVERGENCE ANALYSIS WITH TIME SERIES")
    print("="*42)
    
    if not POLARS_AVAILABLE:
        return
    
    # Generate synthetic convergence history
    print("1. Generating synthetic convergence data...")
    
    # Simulate different convergence patterns
    convergence_scenarios = {
        "fast_linear": {"rate": 0.8, "noise": 0.1},
        "slow_linear": {"rate": 0.95, "noise": 0.05},
        "superlinear": {"rate": 0.9, "noise": 0.02, "acceleration": 0.02}
    }
    
    analyzer = create_time_series_analyzer()
    convergence_results = {}
    
    for scenario_name, params in convergence_scenarios.items():
        print(f"   Analyzing {scenario_name} convergence...")
        
        # Generate convergence history
        max_iter = 100
        initial_error = 1.0
        history = []
        
        error = initial_error
        for i in range(max_iter):
            # Add noise
            noise = params["noise"] * np.random.randn()
            
            # Apply convergence rate
            if "acceleration" in params:
                # Superlinear convergence
                rate = params["rate"] - i * params["acceleration"] / max_iter
                rate = max(0.1, rate)  # Don't let it go too fast
            else:
                rate = params["rate"]
            
            error *= rate
            error = max(1e-12, error * (1 + noise))  # Add noise but keep positive
            
            history.append({
                "iteration": i,
                "error": error,
                "residual": error,
                "scenario": scenario_name,
                "relative_error": error / initial_error
            })
        
        # Create DataFrame and analyze
        conv_df = analyzer.create_convergence_dataframe(history)
        convergence_stats = analyzer.analyze_convergence_rate(conv_df)
        
        convergence_results[scenario_name] = {
            "dataframe": conv_df,
            "statistics": convergence_stats
        }
        
        print(f"     ✓ Convergence rate: {convergence_stats['convergence_rate']:.4f}")
        print(f"     ✓ Error reduction: {convergence_stats['error_reduction_factor']:.2e}")
        print(f"     ✓ R-squared: {convergence_stats['r_squared']:.4f}")
        
        # Detect plateau
        plateau_iter = analyzer.detect_convergence_plateau(conv_df)
        if plateau_iter:
            print(f"     ✓ Plateau detected at iteration {plateau_iter}")
        else:
            print(f"     ✓ No plateau detected")
    
    return convergence_results


def demo_data_export_import():
    """Demonstrate efficient data export and import."""
    print("\n💾 HIGH-PERFORMANCE DATA EXPORT/IMPORT")
    print("="*40)
    
    if not POLARS_AVAILABLE:
        return
    
    # Create sample data
    results = generate_synthetic_mfg_results(50)
    analyzer = create_parameter_sweep_analyzer()
    df = analyzer.create_sweep_dataframe(results)
    
    exporter = create_data_exporter()
    
    # Create output directory
    output_dir = Path("polars_demo_data")
    output_dir.mkdir(exist_ok=True)
    
    # Test different export formats
    formats = {
        "parquet": output_dir / "mfg_results.parquet",
        "csv": output_dir / "mfg_results.csv",
        "json": output_dir / "mfg_results.json"
    }
    
    export_times = {}
    file_sizes = {}
    
    for format_name, filepath in formats.items():
        print(f"1. Exporting to {format_name.upper()}...")
        
        start_time = time.time()
        if format_name == "parquet":
            exporter.export_to_parquet(df, filepath)
        elif format_name == "csv":
            exporter.export_to_csv(df, filepath)
        elif format_name == "json":
            exporter.export_to_json(df, filepath)
        
        export_times[format_name] = time.time() - start_time
        file_sizes[format_name] = filepath.stat().st_size / 1024  # KB
        
        print(f"   ✓ Export time: {export_times[format_name]:.4f}s")
        print(f"   ✓ File size: {file_sizes[format_name]:.1f} KB")
    
    # Test import performance
    print("\n2. Testing import performance...")
    import_times = {}
    
    # Test Parquet import (fastest)
    start_time = time.time()
    imported_df = exporter.load_from_parquet(formats["parquet"])
    import_times["parquet"] = time.time() - start_time
    
    print(f"   ✓ Parquet import time: {import_times['parquet']:.4f}s")
    print(f"   ✓ Imported shape: {imported_df.shape}")
    print(f"   ✓ Data integrity check: {imported_df.shape == df.shape}")
    
    # Performance summary
    print("\n3. Performance Summary:")
    print("   Format    | Export Time | Import Time | File Size")
    print("   --------- | ----------- | ----------- | ---------")
    for fmt in formats.keys():
        import_time = import_times.get(fmt, "N/A")
        print(f"   {fmt:9} | {export_times[fmt]:9.4f}s | {import_time:9} | {file_sizes[fmt]:7.1f} KB")
    
    return output_dir


def demo_performance_comparison():
    """Demonstrate Polars vs pandas performance."""
    print("\n⚡ PERFORMANCE COMPARISON: POLARS vs PANDAS")
    print("="*47)
    
    if not POLARS_AVAILABLE:
        return
    
    print("Running performance benchmarks...")
    
    # Test different data sizes
    data_sizes = [1000, 10000, 50000, 100000]
    benchmark_results = []
    
    for size in data_sizes:
        print(f"\n📊 Benchmarking with {size:,} records...")
        result = benchmark_polars_vs_pandas(size)
        benchmark_results.append(result)
        
        print(f"   Polars time: {result['polars_time']:.4f}s")
        if result['pandas_time']:
            print(f"   Pandas time: {result['pandas_time']:.4f}s")
            print(f"   Speedup: {result['speedup_factor']:.1f}x faster")
        else:
            print("   Pandas not available for comparison")
    
    # Create summary visualization
    if len(benchmark_results) > 1:
        create_performance_visualization(benchmark_results)
    
    return benchmark_results


def create_performance_visualization(benchmark_results):
    """Create visualization of performance comparison."""
    print("\n📈 Creating performance visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Polars vs Pandas Performance Comparison', fontsize=14, fontweight='bold')
    
    # Extract data
    sizes = [r['data_size'] for r in benchmark_results]
    polars_times = [r['polars_time'] for r in benchmark_results]
    pandas_times = [r['pandas_time'] for r in benchmark_results if r['pandas_time']]
    speedups = [r['speedup_factor'] for r in benchmark_results if r['speedup_factor']]
    
    # Plot 1: Execution times
    ax1 = axes[0]
    ax1.plot(sizes, polars_times, 'o-', linewidth=2, label='Polars', color='blue')
    if pandas_times:
        ax1.plot(sizes[:len(pandas_times)], pandas_times, 's-', linewidth=2, label='Pandas', color='orange')
    
    ax1.set_xlabel('Data Size (records)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup factor
    ax2 = axes[1]
    if speedups:
        ax2.plot(sizes[:len(speedups)], speedups, 'o-', linewidth=2, color='green')
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
        ax2.set_xlabel('Data Size (records)')
        ax2.set_ylabel('Speedup Factor (times faster)')
        ax2.set_title('Polars Speedup vs Pandas')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Pandas not available\nfor comparison', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Speedup Analysis')
    
    plt.tight_layout()
    
    output_path = "polars_performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Performance visualization saved: {output_path}")
    plt.close()


def create_analysis_summary(df, convergence_results, benchmark_results):
    """Create comprehensive analysis summary."""
    print("\n📋 COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*35)
    
    print("🚀 Polars Integration Capabilities:")
    print("   ✓ High-performance DataFrame operations")
    print("   ✓ Efficient parameter sweep analysis")
    print("   ✓ Advanced time series convergence analysis")
    print("   ✓ Multiple data export/import formats")
    print("   ✓ Memory-efficient large dataset handling")
    
    if df:
        print(f"\n📊 Parameter Sweep Analysis:")
        print(f"   • Processed {len(df)} simulation results")
        print(f"   • {len(df.columns)} data columns")
        print(f"   • Efficient groupby and aggregation operations")
    
    if convergence_results:
        print(f"\n📈 Convergence Analysis:")
        print(f"   • Analyzed {len(convergence_results)} convergence scenarios")
        print("   • Automatic convergence rate detection")
        print("   • Plateau detection capabilities")
        print("   • Linear regression fitting for rate analysis")
    
    if benchmark_results:
        avg_speedup = np.mean([r['speedup_factor'] for r in benchmark_results 
                              if r['speedup_factor']])
        if avg_speedup > 0:
            print(f"\n⚡ Performance Benefits:")
            print(f"   • Average speedup: {avg_speedup:.1f}x faster than pandas")
            print("   • Memory-efficient operations")
            print("   • Lazy evaluation capabilities")
            print("   • Optimized query execution")
    
    print(f"\n🔧 Integration Benefits:")
    print("   • Seamless NumPy array integration")
    print("   • Compatible with existing MFG workflows") 
    print("   • Professional data analysis capabilities")
    print("   • Scalable to large parameter sweeps")


def main():
    """Run complete Polars integration demonstration."""
    print("🏗️ MFG_PDE POLARS INTEGRATION DEMONSTRATION")
    print("="*48)
    
    if not POLARS_AVAILABLE:
        print("\n❌ Polars not available. Install with:")
        print("pip install polars")
        return
    
    try:
        print("Demonstrating high-performance data analysis with Polars...")
        
        # Parameter sweep analysis
        df, analyzer = demo_parameter_sweep_analysis()
        
        # Convergence analysis
        convergence_results = demo_convergence_analysis()
        
        # Data export/import
        output_dir = demo_data_export_import()
        
        # Performance comparison
        benchmark_results = demo_performance_comparison()
        
        # Summary
        create_analysis_summary(df, convergence_results, benchmark_results)
        
        print("\n✅ POLARS INTEGRATION DEMONSTRATION COMPLETED")
        print("="*47)
        print("\nPolars integration provides:")
        print("• 2-10x faster data processing than pandas")
        print("• Memory-efficient handling of large datasets")
        print("• Advanced analytics for MFG parameter sweeps")
        print("• Professional-grade data export/import")
        print("• Seamless integration with existing workflows")
        
        if output_dir and output_dir.exists():
            print(f"\nDemo data saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()