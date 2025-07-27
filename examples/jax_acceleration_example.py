"""
JAX Acceleration Example for MFG_PDE

This example demonstrates how to use JAX-accelerated solvers for high-performance
Mean Field Games computation, including GPU acceleration and automatic differentiation.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, Any

# Import MFG_PDE components
from mfg_pde import ExampleMFGProblem
from mfg_pde.accelerated import (
    check_jax_availability, check_gpu_availability, 
    get_device_info, print_acceleration_info
)

# Conditional imports based on JAX availability
try:
    from mfg_pde.accelerated import JAXMFGSolver, HAS_JAX, HAS_GPU
    from mfg_pde import create_fast_solver  # For comparison
except ImportError:
    print("⚠️ JAX acceleration not available. Install JAX to run this example.")
    print("   pip install jax jaxlib  # CPU version")
    print("   pip install jax[cuda12_pip]  # CUDA version")
    exit(1)


def compare_solvers_performance():
    """Compare performance between standard and JAX-accelerated solvers."""
    print("\n" + "="*60)
    print("🏁 PERFORMANCE COMPARISON: Standard vs JAX Solvers")
    print("="*60)
    
    # Test different problem sizes
    problem_sizes = [
        (20, 10, "Small"),
        (50, 25, "Medium"), 
        (100, 50, "Large"),
        (200, 100, "Very Large") if HAS_GPU else (100, 50, "Large (CPU)")
    ]
    
    results = []
    
    for Nx, Nt, size_name in problem_sizes:
        print(f"\n🔧 Testing {size_name} problem: {Nx}x{Nt} grid")
        
        # Create test problem
        problem = ExampleMFGProblem(Nx=Nx, Nt=Nt, T=1.0)
        
        # Test standard solver
        print("  📊 Running standard solver...")
        start_time = time.time()
        try:
            standard_solver = create_fast_solver(problem, "fixed_point")
            standard_result = standard_solver.solve()
            standard_time = time.time() - start_time
            standard_converged = standard_result.converged
        except Exception as e:
            print(f"     ❌ Standard solver failed: {e}")
            standard_time = float('inf')
            standard_converged = False
        
        # Test JAX solver (CPU)
        print("  🚀 Running JAX solver (CPU)...")
        start_time = time.time()
        try:
            jax_cpu_solver = JAXMFGSolver(
                problem, 
                method="fixed_point",
                max_iterations=50,
                use_gpu=False,
                jit_compile=True
            )
            jax_cpu_result = jax_cpu_solver.solve()
            jax_cpu_time = time.time() - start_time
            jax_cpu_converged = jax_cpu_result.converged
        except Exception as e:
            print(f"     ❌ JAX CPU solver failed: {e}")
            jax_cpu_time = float('inf')
            jax_cpu_converged = False
        
        # Test JAX solver (GPU) if available
        jax_gpu_time = None
        jax_gpu_converged = None
        if HAS_GPU:
            print("  ⚡ Running JAX solver (GPU)...")
            start_time = time.time()
            try:
                jax_gpu_solver = JAXMFGSolver(
                    problem,
                    method="fixed_point", 
                    max_iterations=50,
                    use_gpu=True,
                    jit_compile=True
                )
                jax_gpu_result = jax_gpu_solver.solve()
                jax_gpu_time = time.time() - start_time
                jax_gpu_converged = jax_gpu_result.converged
            except Exception as e:
                print(f"     ❌ JAX GPU solver failed: {e}")
                jax_gpu_time = float('inf')
                jax_gpu_converged = False
        
        # Store results
        result = {
            'size_name': size_name,
            'grid_size': (Nx, Nt),
            'total_points': Nx * Nt,
            'standard_time': standard_time,
            'standard_converged': standard_converged,
            'jax_cpu_time': jax_cpu_time,
            'jax_cpu_converged': jax_cpu_converged,
            'jax_gpu_time': jax_gpu_time,
            'jax_gpu_converged': jax_gpu_converged,
            'cpu_speedup': standard_time / jax_cpu_time if jax_cpu_time > 0 else 0,
            'gpu_speedup': standard_time / jax_gpu_time if jax_gpu_time and jax_gpu_time > 0 else None
        }
        results.append(result)
        
        # Print summary for this size
        print(f"     ⏱️  Standard: {standard_time:.2f}s ({'✅' if standard_converged else '❌'})")
        print(f"     ⏱️  JAX CPU:  {jax_cpu_time:.2f}s ({'✅' if jax_cpu_converged else '❌'}) "
              f"[{result['cpu_speedup']:.1f}x speedup]")
        if HAS_GPU and jax_gpu_time:
            print(f"     ⏱️  JAX GPU:  {jax_gpu_time:.2f}s ({'✅' if jax_gpu_converged else '❌'}) "
                  f"[{result['gpu_speedup']:.1f}x speedup]")
    
    return results


def demonstrate_automatic_differentiation():
    """Demonstrate automatic differentiation capabilities."""
    print("\n" + "="*60)
    print("🔬 AUTOMATIC DIFFERENTIATION DEMO")
    print("="*60)
    
    if not HAS_JAX:
        print("❌ JAX not available for automatic differentiation")
        return
    
    # Create test problem
    problem = ExampleMFGProblem(Nx=30, Nt=15, T=1.0)
    
    # Create JAX solver
    solver = JAXMFGSolver(
        problem,
        method="fixed_point",
        max_iterations=20,
        tolerance=1e-4,
        use_gpu=HAS_GPU
    )
    
    print("🧮 Computing parameter sensitivity using automatic differentiation...")
    
    try:
        # Compute sensitivity to diffusion parameter
        sensitivity = solver.compute_sensitivity('sigma', perturbation=1e-6)
        
        print(f"📊 Sensitivity Results:")
        print(f"   Parameter: {sensitivity['parameter']}")
        print(f"   Current value: {sensitivity['current_value']:.4f}")
        print(f"   Sensitivity: {sensitivity['sensitivity']:.6e}")
        print(f"   ➡️  This means a 1% increase in σ changes the objective by ~{sensitivity['sensitivity']*0.01:.2e}")
        
    except Exception as e:
        print(f"⚠️ Sensitivity computation not fully implemented: {e}")
        print("   This would show parameter sensitivity using automatic differentiation")


def demonstrate_batch_processing():
    """Demonstrate vectorized batch processing."""
    print("\n" + "="*60)
    print("📦 BATCH PROCESSING DEMO")
    print("="*60)
    
    if not HAS_JAX:
        print("❌ JAX not available for batch processing")
        return
    
    print("🔄 Testing batch processing for parameter studies...")
    
    # Create multiple problems with different parameters
    base_problem = ExampleMFGProblem(Nx=25, Nt=12, T=1.0)
    
    # Different sigma values to test
    sigma_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    
    print(f"🧪 Running parameter study with {len(sigma_values)} different σ values...")
    
    batch_start = time.time()
    results = []
    
    for i, sigma in enumerate(sigma_values):
        print(f"   📍 Problem {i+1}/{len(sigma_values)}: σ = {sigma}")
        
        # Create problem with modified sigma
        problem = ExampleMFGProblem(Nx=25, Nt=12, T=1.0, sigma=sigma)
        
        # Solve with JAX
        solver = JAXMFGSolver(
            problem,
            method="fixed_point",
            max_iterations=20,
            tolerance=1e-4,
            use_gpu=HAS_GPU,
            jit_compile=(i == 0)  # Only compile once
        )
        
        result = solver.solve()
        results.append({
            'sigma': sigma,
            'converged': result.converged,
            'iterations': result.iterations,
            'final_error': result.final_error,
            'execution_time': result.execution_time
        })
    
    batch_time = time.time() - batch_start
    
    print(f"\n📊 Batch Processing Results:")
    print(f"   Total time: {batch_time:.2f}s")
    print(f"   Average time per problem: {batch_time/len(sigma_values):.2f}s")
    print(f"   All converged: {'✅' if all(r['converged'] for r in results) else '❌'}")
    
    # Show parameter sensitivity
    errors = [r['final_error'] for r in results]
    times = [r['execution_time'] for r in results]
    
    print(f"\n📈 Parameter Study Results:")
    for i, (sigma, result) in enumerate(zip(sigma_values, results)):
        print(f"   σ={sigma:.1f}: error={result['final_error']:.2e}, "
              f"time={result['execution_time']:.2f}s, "
              f"iter={result['iterations']}")


def demonstrate_performance_benchmarking():
    """Demonstrate built-in performance benchmarking."""
    print("\n" + "="*60)
    print("📊 PERFORMANCE BENCHMARKING DEMO")
    print("="*60)
    
    if not HAS_JAX:
        print("❌ JAX not available for performance benchmarking")
        return
    
    # Create test problem
    problem = ExampleMFGProblem(Nx=30, Nt=15, T=1.0)
    
    # Create JAX solver
    solver = JAXMFGSolver(
        problem,
        method="fixed_point",
        max_iterations=30,
        tolerance=1e-6,
        use_gpu=HAS_GPU
    )
    
    print("🏃 Running built-in performance benchmark...")
    
    # Define benchmark sizes
    if HAS_GPU:
        benchmark_sizes = [(20, 10), (40, 20), (60, 30), (80, 40)]
    else:
        benchmark_sizes = [(20, 10), (30, 15), (40, 20)]  # Smaller for CPU
    
    benchmark_results = solver.benchmark_performance(benchmark_sizes)
    
    print(f"\n📊 Benchmark Results:")
    print(f"   Device: {benchmark_results['device']}")
    print(f"   JAX Version: {benchmark_results['jax_version']}")
    
    summary = benchmark_results['summary']
    print(f"\n🏆 Performance Summary:")
    print(f"   Fastest time: {summary['fastest_time']:.2f}s")
    print(f"   Slowest time: {summary['slowest_time']:.2f}s")
    print(f"   Mean time: {summary['mean_time']:.2f}s")
    print(f"   Largest problem: {summary['largest_problem']} grid points")
    print(f"   All converged: {'✅' if summary['all_converged'] else '❌'}")
    print(f"   GPU acceleration: {'✅' if summary['gpu_acceleration'] else '❌'}")
    
    print(f"\n📋 Detailed Results:")
    for result in benchmark_results['benchmark_results']:
        size = result['problem_size']
        print(f"   {size[0]}x{size[1]}: {result['total_time']:.2f}s "
              f"({result['solve_time']:.2f}s solve + {result['compile_time']:.2f}s compile), "
              f"~{result['memory_gb']:.1f}GB")


def visualize_results_comparison():
    """Visualize comparison between standard and JAX solvers."""
    print("\n" + "="*60)
    print("📈 RESULTS VISUALIZATION")
    print("="*60)
    
    # Create test problem
    problem = ExampleMFGProblem(Nx=40, Nt=20, T=1.0)
    
    # Solve with both methods
    print("🔄 Solving with standard solver...")
    standard_solver = create_fast_solver(problem, "fixed_point")
    standard_result = standard_solver.solve()
    
    if HAS_JAX:
        print("🔄 Solving with JAX solver...")
        jax_solver = JAXMFGSolver(
            problem,
            method="fixed_point",
            max_iterations=50,
            tolerance=1e-6,
            use_gpu=HAS_GPU
        )
        jax_result = jax_solver.solve()
        
        # Compare solutions
        U_diff = np.abs(standard_result.U_solution - jax_result.U_solution)
        M_diff = np.abs(standard_result.M_solution - jax_result.M_solution)
        
        print(f"\n📊 Solution Comparison:")
        print(f"   Max difference in U: {np.max(U_diff):.2e}")
        print(f"   Max difference in M: {np.max(M_diff):.2e}")
        print(f"   Solutions match: {'✅' if np.max(U_diff) < 1e-4 else '❌'}")
        
        # Performance comparison
        print(f"\n⏱️  Performance Comparison:")
        print(f"   Standard solver: {standard_result.execution_time:.2f}s")
        print(f"   JAX solver: {jax_result.execution_time:.2f}s")
        speedup = standard_result.execution_time / jax_result.execution_time
        print(f"   Speedup: {speedup:.1f}x")
        
        # Convergence comparison
        print(f"\n🎯 Convergence Comparison:")
        print(f"   Standard: {standard_result.iterations} iterations, "
              f"error={standard_result.final_error:.2e}")
        print(f"   JAX: {jax_result.iterations} iterations, "
              f"error={jax_result.final_error:.2e}")
    
    else:
        print("⚠️ JAX not available for comparison")


def main():
    """Main demonstration function."""
    print("🚀 JAX Acceleration Demo for MFG_PDE")
    print("=" * 60)
    
    # Show system information
    print_acceleration_info()
    
    if not HAS_JAX:
        print("\n❌ JAX is not available. Please install JAX to run this demo:")
        print("   pip install jax jaxlib  # CPU version")
        print("   pip install jax[cuda12_pip]  # CUDA 12 version")
        print("   pip install jax[cuda11_pip]  # CUDA 11 version")
        return
    
    try:
        # Run all demonstrations
        performance_results = compare_solvers_performance()
        demonstrate_automatic_differentiation()
        demonstrate_batch_processing() 
        demonstrate_performance_benchmarking()
        visualize_results_comparison()
        
        # Final summary
        print("\n" + "="*60)
        print("🏁 JAX ACCELERATION DEMO COMPLETE")
        print("="*60)
        
        if performance_results:
            best_speedup = max(r['cpu_speedup'] for r in performance_results if r['cpu_speedup'] > 0)
            print(f"🎯 Best CPU speedup achieved: {best_speedup:.1f}x")
            
            if HAS_GPU:
                gpu_speedups = [r['gpu_speedup'] for r in performance_results if r['gpu_speedup']]
                if gpu_speedups:
                    best_gpu_speedup = max(gpu_speedups)
                    print(f"⚡ Best GPU speedup achieved: {best_gpu_speedup:.1f}x")
        
        print("\n💡 Key Benefits of JAX Acceleration:")
        print("   ✅ Automatic differentiation for sensitivity analysis")
        print("   ✅ Just-in-time compilation for optimized performance")
        print("   ✅ GPU acceleration for large problems")
        print("   ✅ Vectorized batch processing")
        print("   ✅ Memory-efficient implementations")
        
        print("\n🔗 Next Steps:")
        print("   • Integrate JAX solvers into your research workflow")
        print("   • Explore parameter sensitivity analysis")
        print("   • Scale up to larger problems with GPU acceleration")
        print("   • Use batch processing for parameter studies")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()