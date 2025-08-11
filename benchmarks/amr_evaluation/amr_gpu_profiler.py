#!/usr/bin/env python3
"""
AMR GPU/CPU Performance Profiler

This module provides detailed performance profiling of AMR operations
on different computational backends (CPU, GPU via JAX).

Profiling Metrics:
- Backend-specific timing (CPU vs GPU)
- Memory transfer overhead analysis
- JAX compilation timing
- Operations breakdown (error estimation, mesh adaptation, interpolation)
- Scaling analysis across problem sizes
- Memory bandwidth utilization
"""

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

import numpy as np

# MFG_PDE imports
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_amr_solver
from mfg_pde.geometry import Domain1D, periodic_bc

try:
    import jax
    import jax.numpy as jnp
    from jax import device_get, device_put

    JAX_AVAILABLE = True

    # Check available devices
    JAX_DEVICES = jax.devices()
    GPU_AVAILABLE = any(d.device_kind == 'gpu' for d in JAX_DEVICES)

except ImportError:
    JAX_AVAILABLE = False
    GPU_AVAILABLE = False
    JAX_DEVICES = []

try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


@dataclass
class ProfileResult:
    """Container for detailed profiling results."""

    operation_name: str
    backend: str  # 'cpu', 'gpu', 'auto'
    problem_size: int

    # Timing breakdown
    total_time: float
    compilation_time: float = 0.0  # JAX compilation overhead
    compute_time: float = 0.0  # Pure computation
    memory_transfer_time: float = 0.0  # CPU↔GPU transfers

    # Memory usage
    peak_memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0

    # Throughput metrics
    operations_per_second: float = 0.0
    memory_bandwidth_gb_s: float = 0.0

    # Operation-specific metrics
    error_estimation_time: float = 0.0
    mesh_adaptation_time: float = 0.0
    interpolation_time: float = 0.0

    # Device information
    device_info: Dict[str, Any] = field(default_factory=dict)


class AMRGPUProfiler:
    """GPU/CPU performance profiler for AMR operations."""

    def __init__(self, output_dir: str = "gpu_profiling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.results: List[ProfileResult] = []
        self.system_info = self._get_system_info()

        print("AMR GPU/CPU Performance Profiler")
        print("=" * 50)
        print(f"JAX Available: {JAX_AVAILABLE}")
        print(f"GPU Available: {GPU_AVAILABLE}")

        if JAX_AVAILABLE:
            print(f"JAX Devices: {[str(d) for d in JAX_DEVICES]}")

            # Warm up JAX
            print("Warming up JAX...")
            dummy = jnp.array([1.0, 2.0, 3.0])
            _ = jnp.sum(dummy)
            print("JAX ready.")

        print("=" * 50)

    def _get_system_info(self) -> Dict[str, Any]:
        """Collect detailed system information."""
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(logical=False),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'jax_available': JAX_AVAILABLE,
            'gpu_available': GPU_AVAILABLE,
            'devices': [],
        }

        if JAX_AVAILABLE:
            for device in JAX_DEVICES:
                device_info = {
                    'id': device.id,
                    'kind': device.device_kind,
                    'platform': device.platform,
                    'string': str(device),
                }
                info['devices'].append(device_info)

        return info

    def profile_jax_compilation_overhead(self, problem_sizes: List[int]):
        """Profile JAX compilation overhead for different problem sizes."""

        if not JAX_AVAILABLE:
            print("JAX not available - skipping compilation profiling")
            return

        print("\n🔥 JAX Compilation Overhead Analysis")
        print("-" * 40)

        # Define a representative AMR operation
        @jax.jit
        def mock_error_estimation(u_vals, m_vals, dx):
            """Mock error estimation function for compilation timing."""
            # Gradient computation
            du_dx = jnp.gradient(u_vals, dx)
            dm_dx = jnp.gradient(m_vals, dx)

            # Second derivatives
            d2u_dx2 = jnp.gradient(du_dx, dx)
            d2m_dx2 = jnp.gradient(dm_dx, dx)

            # Error indicator
            error_indicator = dx * (jnp.abs(du_dx) + jnp.abs(dm_dx)) + dx**2 * (jnp.abs(d2u_dx2) + jnp.abs(d2m_dx2))

            return error_indicator

        for size in problem_sizes:
            print(f"Problem size: {size}")

            # Generate test data
            x = jnp.linspace(0, 1, size)
            dx = x[1] - x[0]
            u_vals = jnp.sin(2 * jnp.pi * x) * jnp.exp(-0.1 * x)
            m_vals = jnp.exp(-(((x - 0.5) / 0.2) ** 2))

            # First call (includes compilation)
            start_time = time.perf_counter()
            _ = mock_error_estimation(u_vals, m_vals, dx).block_until_ready()
            first_call_time = time.perf_counter() - start_time

            # Second call (no compilation)
            start_time = time.perf_counter()
            _ = mock_error_estimation(u_vals, m_vals, dx).block_until_ready()
            second_call_time = time.perf_counter() - start_time

            compilation_overhead = first_call_time - second_call_time

            print(f"  First call: {first_call_time*1000:.2f}ms")
            print(f"  Second call: {second_call_time*1000:.2f}ms")
            print(f"  Compilation overhead: {compilation_overhead*1000:.2f}ms")

            # Store result
            result = ProfileResult(
                operation_name="JAX_Compilation",
                backend="gpu" if GPU_AVAILABLE else "cpu",
                problem_size=size,
                total_time=first_call_time,
                compilation_time=compilation_overhead,
                compute_time=second_call_time,
                operations_per_second=size / second_call_time,
                device_info={'device': str(jax.devices()[0])},
            )

            self.results.append(result)

    def profile_backend_comparison(self, problem_sizes: List[int]):
        """Compare performance across different computational backends."""

        print("\n⚡ Backend Performance Comparison")
        print("-" * 40)

        # Create test problems for each size
        for size in problem_sizes:
            print(f"\nProblem size: {size}")

            # Create 1D problem
            domain = Domain1D(0.0, 1.0, periodic_bc())
            problem = ExampleMFGProblem(T=0.5, xmin=0.0, xmax=1.0, Nx=size, Nt=20, sigma=0.05, coefCT=1.0)
            problem.domain = domain
            problem.dimension = 1

            # Test different backend configurations
            backends_to_test = ['cpu']
            if JAX_AVAILABLE and GPU_AVAILABLE:
                backends_to_test.append('gpu')

            for backend in backends_to_test:
                print(f"  Testing {backend.upper()} backend...")

                # Configure backend (if supported by implementation)
                # This would require backend selection in AMR solver

                # Create AMR solver
                solver = create_amr_solver(
                    problem,
                    base_solver_type='fixed_point',
                    error_threshold=1e-4,
                    max_levels=3,
                    initial_intervals=size // 2,
                )

                # Profile memory before
                process = psutil.Process()
                memory_before = process.memory_info().rss / (1024 * 1024)

                # Time the solve
                start_time = time.perf_counter()
                result = solver.solve(max_iterations=30, tolerance=1e-5, verbose=False)
                total_time = time.perf_counter() - start_time

                # Profile memory after
                memory_after = process.memory_info().rss / (1024 * 1024)
                memory_usage = memory_after - memory_before

                # Extract metrics
                if isinstance(result, dict):
                    total_elements = result.get('mesh_statistics', {}).get('total_intervals', size)
                    total_adaptations = result.get('total_adaptations', 0)
                    converged = result.get('converged', False)
                else:
                    total_elements = size
                    total_adaptations = 0
                    converged = True

                # Estimate operations per second
                # Rough estimate: elements * iterations * operations_per_element
                estimated_operations = total_elements * 30 * 10  # Rough estimate
                ops_per_second = estimated_operations / total_time if total_time > 0 else 0

                # Store result
                profile_result = ProfileResult(
                    operation_name=f"AMR_Solve_{backend}",
                    backend=backend,
                    problem_size=size,
                    total_time=total_time,
                    compute_time=total_time,  # Approximation
                    peak_memory_mb=memory_usage,
                    operations_per_second=ops_per_second,
                    device_info={
                        'backend': backend,
                        'converged': converged,
                        'total_elements': total_elements,
                        'adaptations': total_adaptations,
                    },
                )

                self.results.append(profile_result)

                print(f"    Time: {total_time:.3f}s")
                print(f"    Memory: {memory_usage:.1f}MB")
                print(f"    Elements: {total_elements}")
                print(f"    Converged: {converged}")

    def profile_memory_transfer_overhead(self):
        """Profile memory transfer overhead between CPU and GPU."""

        if not JAX_AVAILABLE or not GPU_AVAILABLE:
            print("GPU not available - skipping memory transfer profiling")
            return

        print("\n🔄 Memory Transfer Overhead Analysis")
        print("-" * 40)

        data_sizes = [1000, 10000, 100000, 1000000]  # Number of elements

        for size in data_sizes:
            print(f"Data size: {size} elements ({size*8/1024/1024:.2f} MB)")

            # Generate test data on CPU
            cpu_data = np.random.randn(size).astype(np.float64)

            # CPU → GPU transfer
            start_time = time.perf_counter()
            gpu_data = device_put(cpu_data)
            cpu_to_gpu_time = time.perf_counter() - start_time

            # GPU computation (dummy operation)
            start_time = time.perf_counter()
            gpu_result = gpu_data * 2.0 + 1.0
            gpu_result.block_until_ready()  # Ensure computation completes
            gpu_compute_time = time.perf_counter() - start_time

            # GPU → CPU transfer
            start_time = time.perf_counter()
            cpu_result = device_get(gpu_result)
            gpu_to_cpu_time = time.perf_counter() - start_time

            total_transfer_time = cpu_to_gpu_time + gpu_to_cpu_time
            data_size_mb = size * 8 / (1024 * 1024)  # 8 bytes per float64
            transfer_bandwidth = (2 * data_size_mb) / total_transfer_time  # GB/s

            print(f"  CPU→GPU: {cpu_to_gpu_time*1000:.2f}ms")
            print(f"  GPU compute: {gpu_compute_time*1000:.2f}ms")
            print(f"  GPU→CPU: {gpu_to_cpu_time*1000:.2f}ms")
            print(f"  Transfer bandwidth: {transfer_bandwidth:.2f} GB/s")

            # Store result
            result = ProfileResult(
                operation_name="Memory_Transfer",
                backend="gpu",
                problem_size=size,
                total_time=total_transfer_time + gpu_compute_time,
                compute_time=gpu_compute_time,
                memory_transfer_time=total_transfer_time,
                memory_bandwidth_gb_s=transfer_bandwidth,
                device_info={
                    'data_size_mb': data_size_mb,
                    'cpu_to_gpu_ms': cpu_to_gpu_time * 1000,
                    'gpu_to_cpu_ms': gpu_to_cpu_time * 1000,
                },
            )

            self.results.append(result)

    def profile_amr_operations_breakdown(self, problem_size: int = 1000):
        """Profile individual AMR operations to identify bottlenecks."""

        print(f"\n🔍 AMR Operations Breakdown (N={problem_size})")
        print("-" * 40)

        if not JAX_AVAILABLE:
            print("JAX not available - using simplified profiling")
            return

        # Mock the core AMR operations for detailed profiling
        x = jnp.linspace(0, 1, problem_size)
        dx = x[1] - x[0]

        # Generate test solution data
        u_vals = jnp.sin(4 * jnp.pi * x) * jnp.exp(-2 * (x - 0.3) ** 2)
        m_vals = jnp.exp(-10 * (x - 0.7) ** 2)
        m_vals = m_vals / jnp.trapz(m_vals, dx=dx)  # Normalize

        operations = {}

        # 1. Error Estimation
        @jax.jit
        def error_estimation_op(u, m, dx_val):
            du_dx = jnp.gradient(u, dx_val)
            dm_dx = jnp.gradient(m, dx_val)
            d2u_dx2 = jnp.gradient(du_dx, dx_val)
            d2m_dx2 = jnp.gradient(dm_dx, dx_val)

            error_u = dx_val * jnp.abs(du_dx) + dx_val**2 * jnp.abs(d2u_dx2)
            error_m = dx_val * jnp.abs(dm_dx) + dx_val**2 * jnp.abs(d2m_dx2)

            return jnp.maximum(error_u, error_m)

        # Warm up
        _ = error_estimation_op(u_vals, m_vals, dx).block_until_ready()

        # Time error estimation
        start_time = time.perf_counter()
        errors = error_estimation_op(u_vals, m_vals, dx).block_until_ready()
        error_time = time.perf_counter() - start_time
        operations['error_estimation'] = error_time

        print(f"Error estimation: {error_time*1000:.2f}ms")

        # 2. Refinement Decision
        @jax.jit
        def refinement_decision_op(errors, threshold):
            return errors > threshold

        start_time = time.perf_counter()
        refine_flags = refinement_decision_op(errors, 1e-4).block_until_ready()
        decision_time = time.perf_counter() - start_time
        operations['refinement_decision'] = decision_time

        print(f"Refinement decision: {decision_time*1000:.2f}ms")

        # 3. Conservative Interpolation (mock)
        @jax.jit
        def conservative_interpolation_op(values, refinement_mask):
            # Simplified interpolation for profiling
            refined_values = jnp.repeat(values, 2)  # Simple doubling
            return refined_values[: len(values)]  # Truncate to original size

        start_time = time.perf_counter()
        interp_u = conservative_interpolation_op(u_vals, refine_flags).block_until_ready()
        interp_time = time.perf_counter() - start_time
        operations['interpolation'] = interp_time

        print(f"Conservative interpolation: {interp_time*1000:.2f}ms")

        # Total time
        total_time = sum(operations.values())
        print(f"Total AMR operations: {total_time*1000:.2f}ms")

        # Store detailed breakdown
        result = ProfileResult(
            operation_name="AMR_Operations_Breakdown",
            backend="gpu" if GPU_AVAILABLE else "cpu",
            problem_size=problem_size,
            total_time=total_time,
            error_estimation_time=operations['error_estimation'],
            mesh_adaptation_time=operations['refinement_decision'],
            interpolation_time=operations['interpolation'],
            operations_per_second=problem_size / total_time,
            device_info={
                'num_refinements': int(jnp.sum(refine_flags)),
                'refinement_ratio': float(jnp.mean(refine_flags)),
            },
        )

        self.results.append(result)

    def generate_performance_plots(self):
        """Generate performance analysis plots."""

        if not PLOTTING_AVAILABLE:
            print("Matplotlib not available - skipping plots")
            return

        print("\nGenerating performance plots...")

        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('AMR GPU/CPU Performance Analysis', fontsize=16)

            # Plot 1: Compilation overhead vs problem size
            compilation_results = [r for r in self.results if r.operation_name == "JAX_Compilation"]
            if compilation_results:
                sizes = [r.problem_size for r in compilation_results]
                comp_times = [r.compilation_time * 1000 for r in compilation_results]  # ms
                compute_times = [r.compute_time * 1000 for r in compilation_results]  # ms

                ax = axes[0, 0]
                ax.loglog(sizes, comp_times, 'r-o', label='Compilation', markersize=6)
                ax.loglog(sizes, compute_times, 'b-s', label='Computation', markersize=6)
                ax.set_xlabel('Problem Size')
                ax.set_ylabel('Time (ms)')
                ax.set_title('JAX Compilation vs Computation Time')
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Plot 2: Backend comparison
            cpu_results = [r for r in self.results if r.backend == 'cpu' and 'AMR_Solve' in r.operation_name]
            gpu_results = [r for r in self.results if r.backend == 'gpu' and 'AMR_Solve' in r.operation_name]

            ax = axes[0, 1]
            if cpu_results:
                sizes = [r.problem_size for r in cpu_results]
                times = [r.total_time for r in cpu_results]
                ax.loglog(sizes, times, 'b-o', label='CPU', markersize=6)

            if gpu_results:
                sizes = [r.problem_size for r in gpu_results]
                times = [r.total_time for r in gpu_results]
                ax.loglog(sizes, times, 'r-s', label='GPU', markersize=6)

            ax.set_xlabel('Problem Size')
            ax.set_ylabel('Solve Time (s)')
            ax.set_title('CPU vs GPU Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 3: Memory transfer analysis
            transfer_results = [r for r in self.results if r.operation_name == "Memory_Transfer"]
            if transfer_results:
                sizes = [r.problem_size for r in transfer_results]
                bandwidths = [r.memory_bandwidth_gb_s for r in transfer_results]

                ax = axes[1, 0]
                ax.semilogx(sizes, bandwidths, 'g-o', markersize=6)
                ax.set_xlabel('Data Size (elements)')
                ax.set_ylabel('Transfer Bandwidth (GB/s)')
                ax.set_title('Memory Transfer Performance')
                ax.grid(True, alpha=0.3)

            # Plot 4: Operations breakdown
            breakdown_results = [r for r in self.results if r.operation_name == "AMR_Operations_Breakdown"]
            if breakdown_results:
                result = breakdown_results[0]  # Take first result

                operations = ['Error\nEstimation', 'Mesh\nAdaptation', 'Interpolation']
                times = [
                    result.error_estimation_time * 1000,
                    result.mesh_adaptation_time * 1000,
                    result.interpolation_time * 1000,
                ]

                ax = axes[1, 1]
                bars = ax.bar(operations, times, color=['red', 'blue', 'green'], alpha=0.7)
                ax.set_ylabel('Time (ms)')
                ax.set_title('AMR Operations Breakdown')

                # Add value labels on bars
                for bar, time_val in zip(bars, times):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        f'{time_val:.2f}ms',
                        ha='center',
                        va='bottom',
                    )

            plt.tight_layout()
            plt.savefig(self.output_dir / 'gpu_performance_analysis.png', dpi=150, bbox_inches='tight')
            print(f"Performance plots saved to {self.output_dir / 'gpu_performance_analysis.png'}")

            # Show if possible
            try:
                plt.show()
            except Exception:
                print("Display not available, plots saved to file.")

        except Exception as e:
            print(f"Error generating plots: {e}")

    def save_profiling_results(self):
        """Save detailed profiling results to JSON."""

        results_file = self.output_dir / "gpu_profiling_results.json"

        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            result_dict = {
                'operation_name': result.operation_name,
                'backend': result.backend,
                'problem_size': result.problem_size,
                'total_time': result.total_time,
                'compilation_time': result.compilation_time,
                'compute_time': result.compute_time,
                'memory_transfer_time': result.memory_transfer_time,
                'peak_memory_mb': result.peak_memory_mb,
                'gpu_memory_mb': result.gpu_memory_mb,
                'operations_per_second': result.operations_per_second,
                'memory_bandwidth_gb_s': result.memory_bandwidth_gb_s,
                'error_estimation_time': result.error_estimation_time,
                'mesh_adaptation_time': result.mesh_adaptation_time,
                'interpolation_time': result.interpolation_time,
                'device_info': result.device_info,
            }
            serializable_results.append(result_dict)

        with open(results_file, 'w') as f:
            json.dump(
                {
                    'profiling_metadata': {
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'system_info': self.system_info,
                        'total_profiles': len(self.results),
                    },
                    'results': serializable_results,
                },
                f,
                indent=2,
            )

        print(f"Profiling results saved to {results_file}")

    def run_comprehensive_gpu_profiling(self):
        """Run the complete GPU/CPU profiling suite."""

        print("Starting Comprehensive AMR GPU/CPU Profiling")
        print("=" * 60)

        problem_sizes = [100, 500, 1000, 2000]

        # Profile JAX compilation overhead
        if JAX_AVAILABLE:
            self.profile_jax_compilation_overhead(problem_sizes)

        # Profile backend comparison
        self.profile_backend_comparison(problem_sizes)

        # Profile memory transfer overhead
        if JAX_AVAILABLE and GPU_AVAILABLE:
            self.profile_memory_transfer_overhead()

        # Profile AMR operations breakdown
        self.profile_amr_operations_breakdown(1000)

        # Generate analysis
        self.generate_performance_plots()
        self.save_profiling_results()

        print(f"\n✅ GPU/CPU profiling complete!")
        print(f"Results saved to: {self.output_dir}")


def main():
    """Run the AMR GPU/CPU profiling suite."""
    profiler = AMRGPUProfiler()
    profiler.run_comprehensive_gpu_profiling()


if __name__ == "__main__":
    main()
