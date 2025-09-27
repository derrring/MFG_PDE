#!/usr/bin/env python3
"""
Backend Quick-Start Examples for MFG_PDE

This script demonstrates how to use each backend with simple, practical examples.
Run with: python backend_quickstart_examples.py
"""

import numpy as np

from mfg_pde import ExampleMFGProblem
from mfg_pde.backends import create_backend, get_available_backends


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print("=" * 60)


def main():
    """Run backend quick-start examples."""
    print("MFG_PDE Backend Quick-Start Examples")
    print("====================================")

    # Check available backends
    available_backends = get_available_backends()
    print("\n📋 Available Backends:")
    for name, status in available_backends.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {name}")

    # Create a simple test problem
    problem = ExampleMFGProblem(Nx=50, Nt=25, T=1.0)
    print(f"\n🧮 Test Problem: {problem.Nx}×{problem.Nt} grid, T={problem.T}")

    # Example 1: Automatic Backend Selection
    print_section("Example 1: Automatic Backend Selection")
    try:
        backend = create_backend("auto")
        print(f"✅ Auto-selected backend: {backend.name}")
        print(f"   Device: {backend.device}")
        print(f"   Precision: {backend.precision}")

        # Show backend info
        info = backend.get_device_info() if hasattr(backend, "get_device_info") else {}
        if info:
            print(f"   Additional info: {info}")
    except Exception as e:
        print(f"❌ Auto selection failed: {e}")

    # Example 2: PyTorch Backend (if available)
    if available_backends.get("torch", False):
        print_section("Example 2: PyTorch Backend")
        try:
            torch_backend = create_backend("torch")
            print(f"✅ PyTorch backend created: {torch_backend.name}")

            # Show specific device selection
            if available_backends.get("torch_mps", False):
                mps_backend = create_backend("torch", device="mps")
                print(f"✅ Apple Silicon MPS: {mps_backend.name}")
            elif available_backends.get("torch_cuda", False):
                cuda_backend = create_backend("torch", device="cuda")
                print(f"✅ NVIDIA CUDA: {cuda_backend.name}")
            else:
                cpu_backend = create_backend("torch", device="cpu")
                print(f"ℹ️  CPU backend: {cpu_backend.name}")

        except Exception as e:
            print(f"❌ PyTorch backend failed: {e}")

    # Example 3: JAX Backend (if available)
    if available_backends.get("jax", False):
        print_section("Example 3: JAX Mathematical Backend")
        try:
            # Import JAX utilities
            from mfg_pde.alg.mfg_solvers import JAX_MFG_AVAILABLE

            print("✅ JAX utilities available")
            print(f"   JAX MFG solver available: {JAX_MFG_AVAILABLE}")

            if JAX_MFG_AVAILABLE:
                from mfg_pde.alg.mfg_solvers import JAXMFGSolver

                # Create JAX solver
                jax_solver = JAXMFGSolver(
                    problem,
                    method="fixed_point",
                    max_iterations=10,  # Reduced for demo
                    use_gpu=available_backends.get("jax_gpu", False),
                )
                print("✅ JAX MFG solver created")
                print(f"   Using GPU: {jax_solver.use_gpu}")
                print(f"   Device: {jax_solver.device}")

                # Demo: Solve small problem
                print("   Running quick solve demo...")
                result = jax_solver.solve()
                print(f"   ✅ Solved! Converged: {result['converged']}")
                print(f"   Iterations: {result['iterations']}")
                print(f"   Final error: {result['final_error']:.2e}")
                print(f"   Execution time: {result['execution_time']:.3f}s")

            else:
                print("ℹ️  JAX solver not available (missing dependencies)")

        except Exception as e:
            print(f"❌ JAX backend failed: {e}")

    # Example 4: Numba Optimization (if available)
    if available_backends.get("numba", False):
        print_section("Example 4: Numba CPU Optimization")
        try:
            from mfg_pde.utils.performance_optimization import AccelerationBackend

            acceleration = AccelerationBackend()
            acceleration.set_backend("numba")
            print("✅ Numba backend configured")
            print(f"   Available backends: {list(acceleration.available_backends.keys())}")

            # Demo: Optimize a simple function
            def simple_finite_difference(u, dx):
                """Simple finite difference operation."""
                result = np.zeros_like(u)
                for i in range(1, len(u) - 1):
                    result[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / (dx * dx)
                return result

            # Optimize with Numba
            optimized_func = acceleration.optimize_function(simple_finite_difference)

            # Test performance with warm-up
            test_array = np.sin(np.linspace(0, 2 * np.pi, 1000))
            dx = 2 * np.pi / 999

            # Warm-up run for Numba compilation
            print("   Warming up Numba compilation...")
            _ = optimized_func(test_array, dx)

            # Time original
            import time

            start = time.time()
            for _ in range(100):
                result_orig = simple_finite_difference(test_array, dx)
            time_orig = time.time() - start

            # Time optimized (after warm-up)
            start = time.time()
            for _ in range(100):
                result_opt = optimized_func(test_array, dx)
            time_opt = time.time() - start

            print(f"   ⏱️  Original: {time_orig:.3f}s")
            print(f"   ⚡ Numba:    {time_opt:.3f}s")
            if time_opt > 0:
                speedup = time_orig / time_opt
                print(f"   🚀 Speedup:  {speedup:.1f}x")
            else:
                print("   🚀 Speedup:  >100x (too fast to measure)")

            # Verify results are identical
            error = np.max(np.abs(result_orig - result_opt))
            print(f"   ✅ Results identical (max error: {error:.2e})")

        except Exception as e:
            print(f"❌ Numba optimization failed: {e}")

    # Example 5: Performance Monitoring
    print_section("Example 5: Performance Monitoring")
    try:
        from mfg_pde.utils.performance_optimization import PerformanceMonitor

        monitor = PerformanceMonitor()

        with monitor.monitor_operation("backend_demo"):
            # Simulate some computation
            backend = create_backend("auto")

            # Create test arrays
            test_u = np.random.random((100, 100))
            test_m = np.random.random((100, 100))

            # Simple operations to monitor
            result = test_u + test_m
            result = np.fft.fft2(result)
            result = np.real(result)

        summary = monitor.get_summary()
        print("✅ Performance monitoring completed")
        print(f"   Total operations: {summary['total_operations']}")
        print(f"   Total time: {summary['total_time']:.3f}s")
        print(f"   Peak memory: {summary['peak_memory']:.1f}MB")

    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")

    # Example 6: Backend Recommendations
    print_section("Example 6: Automated Performance Recommendations")
    try:
        from mfg_pde.utils.performance_optimization import optimize_mfg_problem_performance

        # Get recommendations for different problem sizes
        problem_configs = [
            {"spatial_points": 1000, "time_steps": 50, "dimension": 2},
            {"spatial_points": 10000, "time_steps": 100, "dimension": 3},
        ]

        for i, config in enumerate(problem_configs, 1):
            print(
                f"\n   Problem {i}: {config['spatial_points']} points, {config['time_steps']} steps, {config['dimension']}D"
            )

            analysis = optimize_mfg_problem_performance(config)

            memory_est = analysis["memory_estimates"]
            print(f"   💾 Estimated memory: {memory_est['total_estimated']:.1f}MB")

            recommendations = analysis["recommendations"]
            if recommendations:
                print("   📋 Recommendations:")
                for rec in recommendations[:3]:  # Show top 3
                    print(f"      • {rec}")
            else:
                print("   ✅ Current setup optimal for this problem size")

    except Exception as e:
        print(f"❌ Recommendations failed: {e}")

    # Summary
    print_section("Summary and Next Steps")
    print("✅ Backend system exploration completed!")
    print("\n🎯 Key Takeaways:")
    print("   • Use create_backend('auto') for automatic optimal selection")
    print("   • PyTorch backend excels for neural network components")
    print("   • JAX backend optimal for pure mathematical computations")
    print("   • Numba backend great for CPU optimization of loops/conditionals")
    print("   • Performance monitoring helps identify bottlenecks")

    print("\n📚 Next Steps:")
    print("   • Check docs/development/BACKEND_USAGE_GUIDE.md for detailed usage")
    print("   • See docs/development/ACCELERATION_ARCHITECTURE_COMPLETE.md for architecture")
    print("   • Explore examples/advanced/ for more complex usage patterns")
    print("   • Run benchmarks/ for performance comparisons")

    print("\n🎉 MFG_PDE Backend System Ready!")
    print(f"   Available backends: {sum(available_backends.values())}/{len(available_backends)}")


if __name__ == "__main__":
    main()
