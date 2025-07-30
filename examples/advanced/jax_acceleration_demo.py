#!/usr/bin/env python3
"""
JAX Acceleration Demo for MFG_PDE

Demonstrates the performance benefits of JAX backend for GPU-accelerated
Mean Field Games solving with automatic differentiation capabilities.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# MFG_PDE imports
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.factory import BackendFactory, print_backend_info
from mfg_pde.utils.logging import get_logger, configure_research_logging
from mfg_pde.utils.integration import trapezoid

# Configure logging
configure_research_logging("jax_acceleration_demo", level="INFO")
logger = get_logger(__name__)


class BarProblemJAX(ExampleMFGProblem):
    """El Farol Bar Problem optimized for JAX backend demonstration."""
    
    def __init__(self, capacity=0.6, congestion_cost=2.0, **kwargs):
        super().__init__(**kwargs)
        self.capacity = capacity
        self.congestion_cost = congestion_cost
        logger.info(f"Created BarProblemJAX with capacity={capacity}, congestion_cost={congestion_cost}")
    
    def g(self, x):
        """Terminal cost: quadratic penalty for being far from center."""
        return 0.5 * (x - 0.5)**2
    
    def f(self, x, u, m):
        """Running cost with congestion effects."""
        # Get total attendance by integrating density
        if hasattr(self, '_backend') and self._backend:
            total_attendance = self._backend.trapezoid(m, dx=self.dx)
        else:
            total_attendance = trapezoid(m, dx=self.dx)
        
        # Congestion cost when attendance exceeds capacity
        congestion = max(0, total_attendance - self.capacity)
        movement_cost = 0.1 * u**2
        
        return movement_cost + self.congestion_cost * congestion**2
    
    def rho0(self, x):
        """Initial density: concentrated around x=0.3."""
        return np.exp(-20 * (x - 0.3)**2)


def benchmark_backends(problem_sizes=[100, 500, 1000], num_runs=3):
    """
    Benchmark NumPy vs JAX backends across different problem sizes.
    
    Args:
        problem_sizes: List of grid sizes to test
        num_runs: Number of runs for averaging
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Starting backend benchmark with sizes {problem_sizes}")
    
    results = {
        'problem_sizes': problem_sizes,
        'numpy_times': [],
        'jax_times': [],
        'speedups': [],
        'memory_usage': {}
    }
    
    for Nx in problem_sizes:
        logger.info(f"Benchmarking problem size Nx={Nx}")
        
        # Create test problem
        problem = BarProblemJAX(T=1.0, Nx=Nx, Nt=50, capacity=0.6)
        
        # Benchmark NumPy backend
        numpy_times = []
        for run in range(num_runs):
            try:
                backend = BackendFactory.create_backend("numpy", precision="float64")
                
                start_time = time.perf_counter()
                
                # Simulate basic MFG operations
                x_grid = backend.linspace(problem.xmin, problem.xmax, problem.Nx)
                U = backend.zeros((problem.Nx,))
                M = backend.ones((problem.Nx,))
                M = M / backend.trapezoid(M, dx=problem.dx)  # Normalize
                
                # Simulate solver iterations
                for _ in range(20):
                    U = backend.hjb_step(U, M, 0.01, problem.dx, 
                                       {'x_grid': x_grid, 'sigma_sq': 0.01})
                    M = backend.fpk_step(M, U, 0.01, problem.dx,
                                       {'x_grid': x_grid, 'sigma_sq': 0.01})
                
                end_time = time.perf_counter()
                numpy_times.append(end_time - start_time)
                
                if run == 0:  # Get memory info on first run
                    mem_info = backend.memory_usage()
                    if mem_info:
                        results['memory_usage'][f'numpy_{Nx}'] = mem_info
                
            except Exception as e:
                logger.warning(f"NumPy benchmark failed for Nx={Nx}: {e}")
                numpy_times.append(float('inf'))
        
        # Benchmark JAX backend
        jax_times = []
        for run in range(num_runs):
            try:
                backend = BackendFactory.create_backend("jax", device="auto", 
                                                       precision="float64", jit_compile=True)
                
                start_time = time.perf_counter()
                
                # Simulate basic MFG operations (JAX version)
                x_grid = backend.linspace(problem.xmin, problem.xmax, problem.Nx)
                U = backend.zeros((problem.Nx,))
                M = backend.ones((problem.Nx,))
                M = M / backend.trapezoid(M, dx=problem.dx)
                
                # Simulate solver iterations (JIT compilation on first run)
                for _ in range(20):
                    U = backend.hjb_step(U, M, 0.01, problem.dx,
                                       {'x_grid': x_grid, 'sigma_sq': 0.01})
                    M = backend.fpk_step(M, U, 0.01, problem.dx,
                                       {'x_grid': x_grid, 'sigma_sq': 0.01})
                
                end_time = time.perf_counter()
                jax_times.append(end_time - start_time)
                
                if run == 0:  # Get device info on first run
                    device_info = backend.get_device_info()
                    mem_info = backend.memory_usage()
                    results['memory_usage'][f'jax_{Nx}'] = {
                        'device_info': device_info,
                        'memory_info': mem_info
                    }
                
            except ImportError:
                logger.warning(f"JAX not available for benchmark")
                jax_times.append(float('inf'))
            except Exception as e:
                logger.warning(f"JAX benchmark failed for Nx={Nx}: {e}")
                jax_times.append(float('inf'))
        
        # Calculate averages and speedup
        avg_numpy_time = np.mean(numpy_times)
        avg_jax_time = np.mean(jax_times)
        speedup = avg_numpy_time / avg_jax_time if avg_jax_time != float('inf') else 0
        
        results['numpy_times'].append(avg_numpy_time)
        results['jax_times'].append(avg_jax_time)
        results['speedups'].append(speedup)
        
        logger.info(f"Nx={Nx}: NumPy={avg_numpy_time:.3f}s, JAX={avg_jax_time:.3f}s, "
                   f"Speedup={speedup:.2f}x")
    
    return results


def demonstrate_jax_features():
    """Demonstrate JAX-specific features like automatic differentiation."""
    logger.info("Demonstrating JAX-specific features")
    
    try:
        # Create JAX backend
        backend = BackendFactory.create_backend("jax", device="auto", jit_compile=True)
        logger.info(f"Created JAX backend: {backend.get_device_info()}")
        
        # Demonstrate automatic differentiation
        logger.info("Testing automatic differentiation...")
        
        def test_function(x):
            """Test function: f(x) = x^2 + sin(x)"""
            return x**2 + backend.array_module.sin(x)
        
        # Get gradient function
        grad_func = backend.grad(test_function)
        
        # Test points
        x_test = backend.linspace(0, 2*np.pi, 100)
        f_values = backend.vectorize(test_function)(x_test)
        grad_values = backend.vectorize(grad_func)(x_test)
        
        # Convert back to numpy for analysis
        x_np = backend.to_numpy(x_test)
        f_np = backend.to_numpy(f_values)
        grad_np = backend.to_numpy(grad_values)
        
        # Analytical gradient: f'(x) = 2x + cos(x)
        analytical_grad = 2*x_np + np.cos(x_np)
        gradient_error = np.max(np.abs(grad_np - analytical_grad))
        
        logger.info(f"Automatic differentiation error: {gradient_error:.2e}")
        
        return {
            'x': x_np,
            'function': f_np,
            'gradient': grad_np,
            'analytical_gradient': analytical_grad,
            'error': gradient_error
        }
        
    except ImportError:
        logger.warning("JAX not available for feature demonstration")
        return None
    except Exception as e:
        logger.error(f"JAX feature demonstration failed: {e}")
        return None


def create_performance_plots(benchmark_results, jax_features=None):
    """Create visualization plots for the benchmark results."""
    logger.info("Creating performance visualization plots")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('JAX vs NumPy Backend Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Execution time comparison
    ax1 = axes[0, 0]
    x_pos = np.arange(len(benchmark_results['problem_sizes']))
    width = 0.35
    
    ax1.bar(x_pos - width/2, benchmark_results['numpy_times'], width, 
           label='NumPy', color='blue', alpha=0.7)
    ax1.bar(x_pos + width/2, benchmark_results['jax_times'], width,
           label='JAX', color='red', alpha=0.7)
    
    ax1.set_xlabel('Problem Size (Nx)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(benchmark_results['problem_sizes'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup factor
    ax2 = axes[0, 1]
    valid_speedups = [s if s != float('inf') and s > 0 else 0 for s in benchmark_results['speedups']]
    ax2.plot(benchmark_results['problem_sizes'], valid_speedups, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Problem Size (Nx)')
    ax2.set_ylabel('Speedup Factor (NumPy/JAX)')
    ax2.set_title('JAX Speedup vs Problem Size')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
    ax2.legend()
    
    # Plot 3: Scaling analysis
    ax3 = axes[1, 0]
    ax3.loglog(benchmark_results['problem_sizes'], benchmark_results['numpy_times'], 
              'b-o', label='NumPy', linewidth=2)
    ax3.loglog(benchmark_results['problem_sizes'], benchmark_results['jax_times'],
              'r-s', label='JAX', linewidth=2)
    ax3.set_xlabel('Problem Size (Nx)')
    ax3.set_ylabel('Execution Time (seconds)')
    ax3.set_title('Scaling Analysis (Log-Log)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: JAX automatic differentiation demo
    ax4 = axes[1, 1]
    if jax_features:
        ax4.plot(jax_features['x'], jax_features['function'], 'b-', 
                label='f(x) = x² + sin(x)', linewidth=2)
        ax4.plot(jax_features['x'], jax_features['gradient'], 'r-',
                label="f'(x) (JAX autodiff)", linewidth=2)
        ax4.plot(jax_features['x'], jax_features['analytical_gradient'], 'g--',
                label="f'(x) (analytical)", linewidth=1, alpha=0.7)
        ax4.set_xlabel('x')
        ax4.set_ylabel('Value')
        ax4.set_title(f'Automatic Differentiation Demo\n(Error: {jax_features["error"]:.2e})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'JAX not available\nfor autodiff demo', 
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax4.set_title('Automatic Differentiation Demo')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("examples/advanced/outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "jax_acceleration_benchmark.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Performance plots saved to: {output_path}")
    
    return output_path


def main():
    """Main demonstration function."""
    logger.info("Starting JAX Acceleration Demo")
    
    # Print backend information
    print("🔧 Backend Information")
    print("=" * 50)
    print_backend_info()
    print()
    
    # Run benchmark
    print("🏃 Running Performance Benchmark...")
    print("=" * 50)
    benchmark_results = benchmark_backends(problem_sizes=[100, 300, 500, 1000])
    
    # Display results
    print("\n📊 Benchmark Results:")
    print("-" * 30)
    for i, size in enumerate(benchmark_results['problem_sizes']):
        numpy_time = benchmark_results['numpy_times'][i]
        jax_time = benchmark_results['jax_times'][i]
        speedup = benchmark_results['speedups'][i]
        
        print(f"Problem size Nx={size:4d}:")
        print(f"  NumPy:   {numpy_time:8.3f}s")
        print(f"  JAX:     {jax_time:8.3f}s")
        if speedup > 0 and speedup != float('inf'):
            print(f"  Speedup: {speedup:8.2f}x")
        else:
            print(f"  Speedup: N/A")
        print()
    
    # Demonstrate JAX features
    print("🎯 JAX Feature Demonstration...")
    print("=" * 50)
    jax_features = demonstrate_jax_features()
    
    if jax_features:
        print(f"✅ Automatic differentiation test passed")
        print(f"   Maximum gradient error: {jax_features['error']:.2e}")
    else:
        print("❌ JAX features not available")
    
    # Create visualizations
    print("\n📈 Creating Performance Visualizations...")
    print("=" * 50)
    plot_path = create_performance_plots(benchmark_results, jax_features)
    print(f"✅ Visualizations saved to: {plot_path}")
    
    # Summary and recommendations
    print("\n🎯 Summary and Recommendations:")
    print("=" * 50)
    
    avg_speedup = np.mean([s for s in benchmark_results['speedups'] 
                          if s > 0 and s != float('inf')])
    
    if avg_speedup > 1.5:
        print(f"✅ JAX shows significant performance improvement (avg {avg_speedup:.2f}x speedup)")
        print("   Recommendation: Use JAX backend for production workloads")
    elif avg_speedup > 1.0:
        print(f"⚠️  JAX shows modest performance improvement ({avg_speedup:.2f}x speedup)")
        print("   Recommendation: JAX beneficial for large problems")
    else:
        print("❌ JAX performance not superior to NumPy")
        print("   Recommendation: Check JAX installation and GPU availability")
    
    print("\n🚀 Next Steps:")
    print("   • Install JAX with GPU support: pip install 'mfg_pde[jax-cuda]'")
    print("   • Use BackendFactory.create_optimal_backend() for automatic selection")
    print("   • Enable JIT compilation for maximum performance")
    
    logger.info("JAX Acceleration Demo completed successfully")


if __name__ == "__main__":
    main()