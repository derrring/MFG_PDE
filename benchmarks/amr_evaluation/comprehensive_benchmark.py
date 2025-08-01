#!/usr/bin/env python3
"""
Comprehensive AMR Benchmarking Suite

This is the main entry point for comprehensive AMR benchmarking that combines
all benchmark components: performance, accuracy, GPU profiling, memory analysis,
and real-world problem testing.

Features:
- Integrated benchmarking across all metrics
- Real-world problem validation
- Automated report generation
- Performance regression detection
- Publication-ready results
"""

import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Import all benchmark components
from amr_performance_benchmark import AMRBenchmarkSuite
from amr_accuracy_benchmark import AMRAccuracyBenchmark
from amr_gpu_profiler import AMRGPUProfiler
from amr_memory_profiler import AMRMemoryProfiler
from real_world_problems import RealWorldMFGProblems

# MFG_PDE imports
from mfg_pde.factory import create_solver, create_amr_solver


class ComprehensiveAMRBenchmark:
    """
    Master benchmarking suite that orchestrates all AMR performance analysis.
    
    This class coordinates:
    1. Performance benchmarking (timing, efficiency)
    2. Accuracy analysis (convergence, error metrics)
    3. GPU/CPU profiling (compilation, memory transfer)
    4. Memory usage analysis (scaling, leak detection)
    5. Real-world problem validation
    """
    
    def __init__(self, output_dir: str = "comprehensive_benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize all benchmark components
        self.performance_suite = AMRBenchmarkSuite(str(self.output_dir / "performance"))
        self.accuracy_suite = AMRAccuracyBenchmark(str(self.output_dir / "accuracy"))
        self.gpu_profiler = AMRGPUProfiler(str(self.output_dir / "gpu_profiling"))
        self.memory_profiler = AMRMemoryProfiler(str(self.output_dir / "memory_profiling"))
        self.real_world_problems = RealWorldMFGProblems()
        
        # Benchmark results storage
        self.benchmark_summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'performance': {},
            'accuracy': {},
            'gpu_profiling': {},
            'memory_analysis': {},
            'real_world_validation': {}
        }
        
        print("Comprehensive AMR Benchmark Suite")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"System: {self.performance_suite.system_info['platform']}")
        print(f"CPU cores: {self.performance_suite.system_info['cpu_count']}")
        print(f"JAX available: {self.performance_suite.system_info['jax_available']}")
        print(f"GPU available: {self.performance_suite.system_info['gpu_available']}")
        print("=" * 60)
    
    def run_performance_benchmarks(self):
        """Run comprehensive performance benchmarking."""
        print("\n🚀 Phase 1: Performance Benchmarking")
        print("-" * 50)
        
        # Run the main performance benchmark suite
        self.performance_suite.run_comprehensive_benchmark()
        
        # Extract summary metrics
        if self.performance_suite.results:
            amr_results = [r for r in self.performance_suite.results if r.amr_enabled]
            uniform_results = [r for r in self.performance_suite.results if not r.amr_enabled]
            
            if amr_results and uniform_results:
                avg_amr_time = np.mean([r.solve_time for r in amr_results])
                avg_uniform_time = np.mean([r.solve_time for r in uniform_results])
                avg_efficiency = np.mean([r.mesh_efficiency_ratio for r in amr_results if r.mesh_efficiency_ratio])
                
                self.benchmark_summary['performance'] = {
                    'avg_amr_solve_time': avg_amr_time,
                    'avg_uniform_solve_time': avg_uniform_time,
                    'speedup_ratio': avg_uniform_time / avg_amr_time if avg_amr_time > 0 else 0,
                    'avg_mesh_efficiency': avg_efficiency,
                    'total_benchmarks': len(self.performance_suite.results)
                }
                
                print(f"Performance Summary:")
                print(f"  Average AMR time: {avg_amr_time:.3f}s")
                print(f"  Average uniform time: {avg_uniform_time:.3f}s")
                print(f"  Speedup ratio: {avg_uniform_time/avg_amr_time:.2f}x")
                print(f"  Average mesh efficiency: {avg_efficiency:.3f}")
    
    def run_accuracy_benchmarks(self):
        """Run accuracy and convergence analysis."""
        print("\n🎯 Phase 2: Accuracy Analysis")
        print("-" * 50)
        
        # Run accuracy benchmark suite  
        self.accuracy_suite.run_comprehensive_accuracy_benchmark()
        
        # Extract accuracy metrics
        if self.accuracy_suite.results:
            amr_results = [r for r in self.accuracy_suite.results if 'AMR' in r.solver_name]
            uniform_results = [r for r in self.accuracy_suite.results if 'Uniform' in r.solver_name]
            
            if amr_results and uniform_results:
                avg_amr_error = np.mean([r.l2_error_u for r in amr_results if not np.isnan(r.l2_error_u)])
                avg_uniform_error = np.mean([r.l2_error_u for r in uniform_results if not np.isnan(r.l2_error_u)])
                
                self.benchmark_summary['accuracy'] = {
                    'avg_amr_l2_error': avg_amr_error,
                    'avg_uniform_l2_error': avg_uniform_error,
                    'accuracy_improvement': avg_uniform_error / avg_amr_error if avg_amr_error > 0 else 0,
                    'total_accuracy_tests': len(self.accuracy_suite.results)
                }
                
                print(f"Accuracy Summary:")
                print(f"  Average AMR L2 error: {avg_amr_error:.2e}")
                print(f"  Average uniform L2 error: {avg_uniform_error:.2e}")
                print(f"  Accuracy improvement: {avg_uniform_error/avg_amr_error:.2f}x better")
    
    def run_gpu_profiling(self):
        """Run GPU/CPU performance profiling."""
        print("\n⚡ Phase 3: GPU/CPU Profiling")
        print("-" * 50)
        
        # Run GPU profiling suite
        self.gpu_profiler.run_comprehensive_gpu_profiling()
        
        # Extract profiling metrics
        if self.gpu_profiler.results:
            compilation_results = [r for r in self.gpu_profiler.results if r.operation_name == "JAX_Compilation"]
            
            if compilation_results:
                avg_compilation_time = np.mean([r.compilation_time * 1000 for r in compilation_results])  # ms
                avg_compute_time = np.mean([r.compute_time * 1000 for r in compilation_results])  # ms
                
                self.benchmark_summary['gpu_profiling'] = {
                    'avg_compilation_time_ms': avg_compilation_time,
                    'avg_compute_time_ms': avg_compute_time,
                    'compilation_overhead_ratio': avg_compilation_time / avg_compute_time if avg_compute_time > 0 else 0,
                    'total_profiles': len(self.gpu_profiler.results)
                }
                
                print(f"GPU Profiling Summary:")
                print(f"  Average compilation time: {avg_compilation_time:.2f}ms")
                print(f"  Average compute time: {avg_compute_time:.2f}ms")
                print(f"  Compilation overhead: {avg_compilation_time/avg_compute_time:.2f}x")
    
    def run_memory_analysis(self):
        """Run memory usage analysis."""
        print("\n🧠 Phase 4: Memory Analysis")
        print("-" * 50)
        
        # Run memory profiling suite
        self.memory_profiler.run_comprehensive_memory_analysis()
        
        # Extract memory metrics
        if self.memory_profiler.results:
            scaling_results = [r for r in self.memory_profiler.results if "Scaling_Test" in r.problem_name]
            leak_results = [r for r in self.memory_profiler.results if r.problem_name == "Memory_Leak_Test"]
            
            if scaling_results:
                avg_memory_usage = np.mean([r.memory_increase_mb for r in scaling_results])
                avg_efficiency = np.mean([r.memory_efficiency for r in scaling_results if r.memory_efficiency > 0])
                
                memory_summary = {
                    'avg_memory_usage_mb': avg_memory_usage,
                    'avg_memory_efficiency': avg_efficiency,
                    'total_memory_tests': len(self.memory_profiler.results)
                }
                
                if leak_results:
                    leak_rate = leak_results[0].max_memory_growth_rate
                    memory_summary['memory_leak_rate_mb_per_iter'] = leak_rate
                    memory_summary['memory_leak_detected'] = abs(leak_rate) > 0.1
                
                self.benchmark_summary['memory_analysis'] = memory_summary
                
                print(f"Memory Analysis Summary:")
                print(f"  Average memory usage: {avg_memory_usage:.1f}MB")
                print(f"  Average efficiency: {avg_efficiency:.3f}")
                if leak_results:
                    print(f"  Memory leak rate: {leak_rate:.3f}MB/iter")
                    print(f"  Leak detected: {'Yes' if abs(leak_rate) > 0.1 else 'No'}")
    
    def run_real_world_validation(self):
        """Validate AMR on real-world problems."""
        print("\n🌍 Phase 5: Real-World Problem Validation")
        print("-" * 50)
        
        # Create real-world problems
        print("Creating real-world benchmark problems...")
        problem_suite = self.real_world_problems.create_benchmark_suite()
        
        validation_results = {}
        
        # Test each real-world problem
        for problem_name, problem in problem_suite.items():
            print(f"\nTesting {problem_name}...")
            
            try:
                # Test uniform solver
                uniform_solver = create_solver(problem, solver_type='fixed_point', preset='fast')
                uniform_start = time.perf_counter()
                uniform_result = uniform_solver.solve(max_iterations=50, tolerance=1e-5, verbose=False)
                uniform_time = time.perf_counter() - uniform_start
                
                # Test AMR solver
                amr_solver = create_amr_solver(
                    problem,
                    base_solver_type='fixed_point',
                    error_threshold=1e-4,
                    max_levels=4
                )
                amr_start = time.perf_counter()
                amr_result = amr_solver.solve(max_iterations=50, tolerance=1e-5, verbose=False)
                amr_time = time.perf_counter() - amr_start
                
                # Extract metrics
                uniform_converged = uniform_result.get('converged', False) if isinstance(uniform_result, dict) else True
                amr_converged = amr_result.get('converged', False) if isinstance(amr_result, dict) else True
                
                if isinstance(amr_result, dict) and 'mesh_statistics' in amr_result:
                    total_elements = amr_result['mesh_statistics'].get('total_intervals', problem.Nx)
                    efficiency = total_elements / problem.Nx
                else:
                    total_elements = problem.Nx
                    efficiency = 1.0
                
                validation_results[problem_name] = {
                    'uniform_time': uniform_time,
                    'amr_time': amr_time,
                    'speedup': uniform_time / amr_time if amr_time > 0 else 0,
                    'uniform_converged': uniform_converged,
                    'amr_converged': amr_converged,
                    'mesh_efficiency': efficiency,
                    'total_elements': total_elements,
                    'baseline_elements': problem.Nx
                }
                
                print(f"  Uniform: {uniform_time:.3f}s ({'✓' if uniform_converged else '✗'})")
                print(f"  AMR: {amr_time:.3f}s ({'✓' if amr_converged else '✗'})")
                print(f"  Speedup: {uniform_time/amr_time:.2f}x")
                print(f"  Efficiency: {efficiency:.3f} ({total_elements}/{problem.Nx} elements)")
                
            except Exception as e:
                print(f"  Error testing {problem_name}: {e}")
                validation_results[problem_name] = {'error': str(e)}
        
        self.benchmark_summary['real_world_validation'] = validation_results
        
        # Summary statistics
        successful_tests = [r for r in validation_results.values() if 'error' not in r]
        if successful_tests:
            avg_speedup = np.mean([r['speedup'] for r in successful_tests if r['speedup'] > 0])
            avg_efficiency = np.mean([r['mesh_efficiency'] for r in successful_tests])
            convergence_rate = np.mean([r['amr_converged'] for r in successful_tests])
            
            print(f"\nReal-World Validation Summary:")
            print(f"  Successful tests: {len(successful_tests)}/{len(validation_results)}")
            print(f"  Average speedup: {avg_speedup:.2f}x")
            print(f"  Average mesh efficiency: {avg_efficiency:.3f}")
            print(f"  AMR convergence rate: {convergence_rate:.1%}")
    
    def generate_comprehensive_report(self):
        """Generate master benchmark report combining all results."""
        print("\n📊 Generating Comprehensive Report")
        print("-" * 50)
        
        report_file = self.output_dir / "comprehensive_benchmark_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Comprehensive AMR Benchmark Report\n\n")
            f.write(f"**Generated**: {self.benchmark_summary['timestamp']}  \n")
            f.write(f"**System**: {self.performance_suite.system_info['platform']}  \n")
            f.write(f"**CPU**: {self.performance_suite.system_info['cpu_count']} cores  \n")
            f.write(f"**Memory**: {self.performance_suite.system_info['memory_total_gb']:.1f} GB  \n")
            f.write(f"**JAX**: {self.performance_suite.system_info['jax_available']}  \n")
            f.write(f"**GPU**: {self.performance_suite.system_info['gpu_available']}  \n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("This comprehensive benchmark evaluates AMR-enhanced MFG solvers across multiple dimensions:\n")
            f.write("performance, accuracy, GPU utilization, memory efficiency, and real-world applicability.\n\n")
            
            # Performance Summary
            if 'performance' in self.benchmark_summary and self.benchmark_summary['performance']:
                perf = self.benchmark_summary['performance']
                f.write("### Performance Results\n")
                f.write(f"- **Average Speedup**: {perf.get('speedup_ratio', 0):.2f}x over uniform grids\n")
                f.write(f"- **Mesh Efficiency**: {perf.get('avg_mesh_efficiency', 0):.3f} ")
                f.write(f"({100*(1-perf.get('avg_mesh_efficiency', 1)):.1f}% element reduction)\n")
                f.write(f"- **Total Benchmarks**: {perf.get('total_benchmarks', 0)}\n\n")
            
            # Accuracy Summary
            if 'accuracy' in self.benchmark_summary and self.benchmark_summary['accuracy']:
                acc = self.benchmark_summary['accuracy']
                f.write("### Accuracy Results\n")
                f.write(f"- **Accuracy Improvement**: {acc.get('accuracy_improvement', 0):.2f}x better than uniform\n")
                f.write(f"- **AMR L2 Error**: {acc.get('avg_amr_l2_error', 0):.2e}\n")
                f.write(f"- **Uniform L2 Error**: {acc.get('avg_uniform_l2_error', 0):.2e}\n\n")
            
            # GPU Profiling Summary
            if 'gpu_profiling' in self.benchmark_summary and self.benchmark_summary['gpu_profiling']:
                gpu = self.benchmark_summary['gpu_profiling']
                f.write("### GPU Performance\n")
                f.write(f"- **Compilation Overhead**: {gpu.get('compilation_overhead_ratio', 0):.2f}x compute time\n")
                f.write(f"- **Average Compilation**: {gpu.get('avg_compilation_time_ms', 0):.2f}ms\n")
                f.write(f"- **Average Compute**: {gpu.get('avg_compute_time_ms', 0):.2f}ms\n\n")
            
            # Memory Analysis Summary
            if 'memory_analysis' in self.benchmark_summary and self.benchmark_summary['memory_analysis']:
                mem = self.benchmark_summary['memory_analysis']
                f.write("### Memory Efficiency\n")
                f.write(f"- **Average Memory Usage**: {mem.get('avg_memory_usage_mb', 0):.1f}MB\n")
                f.write(f"- **Memory Efficiency**: {mem.get('avg_memory_efficiency', 0):.3f}\n")
                if 'memory_leak_detected' in mem:
                    leak_status = "Detected" if mem['memory_leak_detected'] else "None detected"
                    f.write(f"- **Memory Leaks**: {leak_status}\n")
                f.write("\n")
            
            # Real-World Validation Summary
            if 'real_world_validation' in self.benchmark_summary:
                real_world = self.benchmark_summary['real_world_validation']
                successful = [r for r in real_world.values() if 'error' not in r]
                
                f.write("### Real-World Problem Validation\n")
                f.write(f"- **Test Coverage**: {len(successful)}/{len(real_world)} problems passed\n")
                
                if successful:
                    avg_speedup = np.mean([r['speedup'] for r in successful if r['speedup'] > 0])
                    avg_efficiency = np.mean([r['mesh_efficiency'] for r in successful])
                    f.write(f"- **Average Speedup**: {avg_speedup:.2f}x\n")
                    f.write(f"- **Average Mesh Efficiency**: {avg_efficiency:.3f}\n")
                f.write("\n")
            
            # Detailed Results
            f.write("## Detailed Results\n\n")
            f.write("Detailed results are available in the following subdirectories:\n\n")
            f.write("- `performance/` - Performance benchmarking results\n")
            f.write("- `accuracy/` - Convergence and accuracy analysis\n")
            f.write("- `gpu_profiling/` - GPU/CPU performance profiling\n")
            f.write("- `memory_profiling/` - Memory usage analysis\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            # Generate data-driven recommendations
            perf = self.benchmark_summary.get('performance', {})
            if perf.get('avg_mesh_efficiency', 1) < 0.7:
                f.write("✅ **AMR Highly Effective**: Significant mesh reduction demonstrates strong efficiency gains.\n\n")
            
            acc = self.benchmark_summary.get('accuracy', {})
            if acc.get('accuracy_improvement', 0) > 1.2:
                f.write("✅ **AMR Accuracy Advantage**: Demonstrates superior solution accuracy.\n\n")
            
            mem = self.benchmark_summary.get('memory_analysis', {})
            if mem.get('memory_leak_detected', False):
                f.write("⚠️ **Memory Leak Attention**: Monitor memory usage in production deployments.\n\n")
            
            f.write("**General Guidelines**:\n")
            f.write("- Use AMR for problems with localized features or sharp gradients\n")
            f.write("- Monitor compilation overhead for small problems\n")
            f.write("- Consider memory usage for large-scale problems\n")
            f.write("- Validate convergence on new problem types\n")
        
        print(f"Comprehensive report generated: {report_file}")
        
        # Save benchmark summary as JSON
        summary_file = self.output_dir / "benchmark_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.benchmark_summary, f, indent=2, default=str)
        print(f"Benchmark summary saved: {summary_file}")
    
    def run_full_benchmark_suite(self):
        """Run the complete comprehensive benchmark suite."""
        print("Starting Comprehensive AMR Benchmark Suite")
        print("=" * 70)
        
        start_time = time.perf_counter()
        
        try:
            # Phase 1: Performance
            self.run_performance_benchmarks()
            
            # Phase 2: Accuracy
            self.run_accuracy_benchmarks()
            
            # Phase 3: GPU Profiling
            self.run_gpu_profiling()
            
            # Phase 4: Memory Analysis
            self.run_memory_analysis()
            
            # Phase 5: Real-World Validation
            self.run_real_world_validation()
            
            # Generate comprehensive report
            self.generate_comprehensive_report()
            
            total_time = time.perf_counter() - start_time
            
            print(f"\n✅ Comprehensive Benchmark Complete!")
            print(f"Total time: {total_time:.1f} seconds")
            print(f"Results saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"\n❌ Benchmark failed: {e}")
            raise


def main():
    """Run the comprehensive AMR benchmark suite."""
    benchmark_suite = ComprehensiveAMRBenchmark()
    benchmark_suite.run_full_benchmark_suite()


if __name__ == "__main__":
    main()