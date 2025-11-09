#!/usr/bin/env python3
"""
AMR Performance Benchmarking Suite

This module provides comprehensive benchmarking of AMR-enhanced solvers
vs uniform grid solvers across various MFG problem types and scales.

Performance Metrics:
- Solution accuracy (L2 error vs reference solution)
- Computational time (CPU/GPU)
- Memory usage
- Mesh efficiency (elements used vs uniform grid)
- Convergence behavior
"""

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

import numpy as np

# MFG_PDE imports
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_amr_solver, create_solver
from mfg_pde.geometry import Domain1D, periodic_bc

try:
    import matplotlib.pyplot as plt  # noqa: F401

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import jax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    problem_name: str
    problem_dimension: int
    problem_size: tuple[int, ...]  # (Nx,) for 1D, (Nx, Ny) for 2D

    # Solver information
    solver_type: str
    amr_enabled: bool
    base_solver_type: str | None = None

    # Performance metrics
    solve_time: float
    memory_usage_mb: float
    peak_memory_mb: float

    # Solution quality
    final_iterations: int
    converged: bool
    final_error: float | None = None

    # AMR-specific metrics
    total_elements: int | None = None
    max_refinement_level: int | None = None
    total_adaptations: int | None = None
    mesh_efficiency_ratio: float | None = None  # AMR elements / uniform elements

    # Solution accuracy (vs reference)
    l2_error_u: float | None = None
    l2_error_m: float | None = None
    max_error_u: float | None = None
    max_error_m: float | None = None

    # System information
    system_info: dict[str, Any] | None = None


class AMRBenchmarkSuite:
    """Comprehensive AMR performance benchmarking suite."""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.results: list[BenchmarkResult] = []
        self.system_info = self._get_system_info()

        print("AMR Performance Benchmarking Suite")
        print(f"Output directory: {self.output_dir}")
        print(f"System: {self.system_info['platform']} - {self.system_info['cpu_count']} cores")
        print(f"JAX available: {self.system_info['jax_available']}")
        print(f"GPU detected: {self.system_info['gpu_available']}")
        print("=" * 60)

    def _get_system_info(self) -> dict[str, Any]:
        """Collect system information for benchmarking context."""
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": platform.python_version(),
            "jax_available": JAX_AVAILABLE,
            "gpu_available": False,
        }

        if JAX_AVAILABLE:
            try:
                devices = jax.devices()
                gpu_devices = [d for d in devices if d.device_kind == "gpu"]
                info["gpu_available"] = len(gpu_devices) > 0
                info["gpu_count"] = len(gpu_devices)
                if gpu_devices:
                    info["gpu_types"] = [str(d) for d in gpu_devices]
            except Exception:
                pass

        return info

    def benchmark_problem(
        self,
        problem_generator,
        problem_name: str,
        solver_configs: list[dict[str, Any]],
        reference_solver_config: dict[str, Any] | None = None,
    ) -> list[BenchmarkResult]:
        """
        Benchmark a specific MFG problem with multiple solver configurations.

        Args:
            problem_generator: Function that creates the MFG problem
            problem_name: Descriptive name for the problem
            solver_configs: List of solver configurations to benchmark
            reference_solver_config: High-accuracy solver for error computation

        Returns:
            List of benchmark results
        """
        print(f"\nBenchmarking: {problem_name}")
        print("-" * 40)

        # Generate problem
        problem = problem_generator()

        # Compute reference solution if requested
        reference_solution = None
        if reference_solver_config:
            print("Computing high-accuracy reference solution...")
            ref_solver = self._create_solver(problem, reference_solver_config)
            ref_start = time.perf_counter()
            reference_solution = ref_solver.solve(verbose=False)
            ref_time = time.perf_counter() - ref_start
            print(f"  Reference solution computed in {ref_time:.2f}s")

        # Benchmark each solver configuration
        problem_results = []
        for i, config in enumerate(solver_configs):
            print(f"\nConfig {i + 1}/{len(solver_configs)}: {config.get('name', 'Unnamed')}")

            result = self._benchmark_single_config(problem, problem_name, config, reference_solution)

            problem_results.append(result)
            self.results.append(result)

            # Print summary
            self._print_config_summary(result)

        return problem_results

    def _create_solver(self, problem, config: dict[str, Any]):
        """Create solver from configuration."""
        if config.get("amr_enabled", False):
            return create_amr_solver(
                problem,
                base_solver_type=config.get("base_solver_type", "fixed_point"),
                error_threshold=config.get("error_threshold", 1e-4),
                max_levels=config.get("max_levels", 5),
                **config.get("amr_kwargs", {}),
            )
        else:
            return create_solver(
                problem,
                solver_type=config.get("solver_type", "fixed_point"),
                preset=config.get("preset", "balanced"),
                **config.get("solver_kwargs", {}),
            )

    def _benchmark_single_config(
        self, problem, problem_name: str, config: dict[str, Any], reference_solution=None
    ) -> BenchmarkResult:
        """Benchmark a single solver configuration."""

        # Create solver
        solver = self._create_solver(problem, config)

        # Memory before solving
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB

        # Solve with timing
        start_time = time.perf_counter()
        result = solver.solve(
            max_iterations=config.get("max_iterations", 100), tolerance=config.get("tolerance", 1e-6), verbose=False
        )
        solve_time = time.perf_counter() - start_time

        # Memory after solving
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_usage = memory_after - memory_before

        # Extract solution data
        if isinstance(result, dict):
            U = result.get("U")
            M = result.get("M")
            converged = result.get("converged", False)
            iterations = result.get("iterations", 0)
            final_error = result.get("final_error")
        else:
            # Handle tuple format (U, M, info)
            U, M, info = result
            converged = info.get("converged", False)
            iterations = info.get("iterations", 0)
            final_error = info.get("final_error")

        # Compute problem size
        if hasattr(problem, "ymin"):  # 2D problem
            problem_size = (problem.Nx, getattr(problem, "Ny", problem.Nx))
            problem_dimension = 2
        else:  # 1D problem
            problem_size = (problem.Nx,)
            problem_dimension = 1

        # AMR-specific metrics
        amr_enabled = config.get("amr_enabled", False)
        total_elements = None
        max_refinement_level = None
        total_adaptations = None
        mesh_efficiency_ratio = None

        if amr_enabled and isinstance(result, dict):
            if "mesh_statistics" in result:
                mesh_stats = result["mesh_statistics"]
                total_elements = mesh_stats.get(
                    "total_intervals", mesh_stats.get("total_triangles", mesh_stats.get("total_cells", None))
                )
                max_refinement_level = mesh_stats.get("max_level", 0)

            total_adaptations = result.get("total_adaptations", 0)

            # Compute mesh efficiency
            if total_elements is not None:
                uniform_elements = np.prod(problem_size)
                mesh_efficiency_ratio = total_elements / uniform_elements

        # Solution accuracy vs reference
        l2_error_u = None
        l2_error_m = None
        max_error_u = None
        max_error_m = None

        if reference_solution is not None:
            ref_U = reference_solution.get("U") if isinstance(reference_solution, dict) else reference_solution[0]
            ref_M = reference_solution.get("M") if isinstance(reference_solution, dict) else reference_solution[1]

            if U is not None and ref_U is not None and U.shape == ref_U.shape:
                l2_error_u = np.sqrt(np.mean((U - ref_U) ** 2))
                max_error_u = np.max(np.abs(U - ref_U))

            if M is not None and ref_M is not None and M.shape == ref_M.shape:
                l2_error_m = np.sqrt(np.mean((M - ref_M) ** 2))
                max_error_m = np.max(np.abs(M - ref_M))

        # Create benchmark result
        benchmark_result = BenchmarkResult(
            problem_name=problem_name,
            problem_dimension=problem_dimension,
            problem_size=problem_size,
            solver_type=config.get("name", "Unknown"),
            amr_enabled=amr_enabled,
            base_solver_type=config.get("base_solver_type") if amr_enabled else config.get("solver_type"),
            solve_time=solve_time,
            memory_usage_mb=memory_usage,
            peak_memory_mb=memory_after,
            final_iterations=iterations,
            converged=converged,
            final_error=final_error,
            total_elements=total_elements,
            max_refinement_level=max_refinement_level,
            total_adaptations=total_adaptations,
            mesh_efficiency_ratio=mesh_efficiency_ratio,
            l2_error_u=l2_error_u,
            l2_error_m=l2_error_m,
            max_error_u=max_error_u,
            max_error_m=max_error_m,
            system_info=self.system_info,
        )

        return benchmark_result

    def _print_config_summary(self, result: BenchmarkResult):
        """Print summary of benchmark result."""
        print(f"  Time: {result.solve_time:.3f}s")
        print(f"  Memory: {result.memory_usage_mb:.1f}MB")
        print(f"  Converged: {result.converged} ({result.final_iterations} iterations)")

        if result.amr_enabled:
            print(f"  AMR: {result.total_elements} elements (efficiency: {result.mesh_efficiency_ratio:.3f})")
            print(f"  Max level: {result.max_refinement_level}, Adaptations: {result.total_adaptations}")

        if result.l2_error_u is not None:
            print(f"  L2 Error U: {result.l2_error_u:.2e}")
        if result.l2_error_m is not None:
            print(f"  L2 Error M: {result.l2_error_m:.2e}")

    def run_comprehensive_benchmark(self):
        """Run comprehensive AMR benchmarking suite."""
        print("Starting Comprehensive AMR Benchmark Suite")
        print("=" * 60)

        # Benchmark 1: 1D Congestion Problem - Scale Analysis
        print("\n🔥 Benchmark 1: 1D Congestion Problem - Scale Analysis")

        def create_1d_congestion_problem(nx=64):
            domain = Domain1D(0.0, 2.0, periodic_bc())
            problem = ExampleMFGProblem(
                T=1.0,
                xmin=0.0,
                xmax=2.0,
                Nx=nx,
                Nt=50,
                sigma=0.05,
                coupling_coefficient=2.0,  # Sharp features  # Strong congestion
            )
            problem.domain = domain
            problem.dimension = 1
            return problem

        # Test different problem sizes
        for nx in [32, 64, 128]:
            solver_configs = [
                {"name": f"Uniform-{nx}", "solver_type": "fixed_point", "preset": "fast", "amr_enabled": False},
                {
                    "name": f"AMR-{nx}",
                    "amr_enabled": True,
                    "base_solver_type": "fixed_point",
                    "error_threshold": 1e-4,
                    "max_levels": 4,
                    "initial_intervals": nx // 2,
                },
            ]

            self.benchmark_problem(lambda n=nx: create_1d_congestion_problem(n), f"1D_Congestion_N{nx}", solver_configs)

        # Benchmark 2: 2D Problem with AMR
        print("\n🌐 Benchmark 2: 2D Problem Comparison")

        def create_2d_problem():
            problem = ExampleMFGProblem(
                T=1.0, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, Nx=32, Ny=32, Nt=30, sigma=0.1, coupling_coefficient=1.5
            )
            problem.dimension = 2
            return problem

        solver_configs_2d = [
            {"name": "Uniform-32x32", "solver_type": "fixed_point", "preset": "balanced", "amr_enabled": False},
            {
                "name": "AMR-2D",
                "amr_enabled": True,
                "base_solver_type": "fixed_point",
                "error_threshold": 1e-4,
                "max_levels": 3,
            },
        ]

        self.benchmark_problem(create_2d_problem, "2D_MFG_Problem", solver_configs_2d)

        # Save results
        self.save_results()

        # Generate report
        self.generate_performance_report()

        print("\n✅ Comprehensive benchmark complete!")
        print(f"Results saved to: {self.output_dir}")

    def save_results(self):
        """Save benchmark results to JSON."""
        results_file = self.output_dir / "benchmark_results.json"

        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            result_dict = {
                "problem_name": result.problem_name,
                "problem_dimension": result.problem_dimension,
                "problem_size": list(result.problem_size),
                "solver_type": result.solver_type,
                "amr_enabled": result.amr_enabled,
                "base_solver_type": result.base_solver_type,
                "solve_time": result.solve_time,
                "memory_usage_mb": result.memory_usage_mb,
                "peak_memory_mb": result.peak_memory_mb,
                "final_iterations": result.final_iterations,
                "converged": result.converged,
                "final_error": result.final_error,
                "total_elements": result.total_elements,
                "max_refinement_level": result.max_refinement_level,
                "total_adaptations": result.total_adaptations,
                "mesh_efficiency_ratio": result.mesh_efficiency_ratio,
                "l2_error_u": result.l2_error_u,
                "l2_error_m": result.l2_error_m,
                "max_error_u": result.max_error_u,
                "max_error_m": result.max_error_m,
                "system_info": result.system_info,
            }
            serializable_results.append(result_dict)

        with open(results_file, "w") as f:
            json.dump(
                {
                    "benchmark_metadata": {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "system_info": self.system_info,
                        "total_benchmarks": len(self.results),
                    },
                    "results": serializable_results,
                },
                f,
                indent=2,
            )

        print(f"Results saved to {results_file}")

    def generate_performance_report(self):
        """Generate comprehensive performance analysis report."""
        report_file = self.output_dir / "performance_report.md"

        with open(report_file, "w") as f:
            f.write("# AMR Performance Benchmark Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**System**: {self.system_info['platform']}  \n")
            f.write(f"**CPU**: {self.system_info['cpu_count']} cores  \n")
            f.write(f"**Memory**: {self.system_info['memory_total_gb']:.1f} GB  \n")
            f.write(f"**JAX**: {self.system_info['jax_available']}  \n")
            f.write(f"**GPU**: {self.system_info['gpu_available']}  \n\n")

            # Performance summary table
            f.write("## Performance Summary\n\n")
            f.write("| Problem | Solver | Time (s) | Memory (MB) | Elements | Efficiency | Converged |\n")
            f.write("|---------|--------|----------|-------------|----------|------------|----------|\n")

            for result in self.results:
                efficiency_str = f"{result.mesh_efficiency_ratio:.3f}" if result.mesh_efficiency_ratio else "N/A"
                elements_str = (
                    str(result.total_elements) if result.total_elements else f"{np.prod(result.problem_size)}"
                )

                f.write(
                    f"| {result.problem_name} | {result.solver_type} | "
                    f"{result.solve_time:.3f} | {result.memory_usage_mb:.1f} | "
                    f"{elements_str} | {efficiency_str} | {result.converged} |\n"
                )

            # AMR Analysis
            amr_results = [r for r in self.results if r.amr_enabled]
            uniform_results = [r for r in self.results if not r.amr_enabled]

            if amr_results and uniform_results:
                f.write("\n## AMR vs Uniform Grid Analysis\n\n")

                # Compare similar problems
                for amr_result in amr_results:
                    # Find corresponding uniform result
                    uniform_result = None
                    for uniform in uniform_results:
                        if uniform.problem_name.replace("Uniform-", "").replace(
                            "AMR-", ""
                        ) == amr_result.problem_name.replace("Uniform-", "").replace("AMR-", ""):
                            uniform_result = uniform
                            break

                    if uniform_result:
                        speedup = uniform_result.solve_time / amr_result.solve_time
                        memory_ratio = amr_result.memory_usage_mb / uniform_result.memory_usage_mb

                        f.write(f"### {amr_result.problem_name}\n")
                        f.write(f"- **Speedup**: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}\n")
                        f.write(f"- **Memory**: {memory_ratio:.2f}x {'more' if memory_ratio > 1 else 'less'}\n")
                        f.write(
                            f"- **Mesh Efficiency**: {amr_result.mesh_efficiency_ratio:.3f} "
                            f"({100 * (1 - amr_result.mesh_efficiency_ratio):.1f}% reduction)\n"
                        )
                        f.write(f"- **Max Refinement Level**: {amr_result.max_refinement_level}\n")
                        f.write(f"- **Total Adaptations**: {amr_result.total_adaptations}\n\n")

            f.write("\n## Recommendations\n\n")
            f.write("Based on benchmark results:\n\n")

            # Generate recommendations based on results
            if amr_results:
                avg_efficiency = np.mean([r.mesh_efficiency_ratio for r in amr_results if r.mesh_efficiency_ratio])
                if avg_efficiency < 0.7:
                    f.write(
                        "✅ **AMR Effective**: Average mesh reduction of "
                        f"{100 * (1 - avg_efficiency):.1f}% demonstrates significant efficiency gains.\n\n"
                    )
                else:
                    f.write(
                        "⚠️ **AMR Limited Benefit**: Consider adjusting error thresholds or problem characteristics.\n\n"
                    )

            f.write("**When to use AMR**:\n")
            f.write("- Problems with localized features or sharp gradients\n")
            f.write("- Large-scale problems where uniform grids are prohibitive\n")
            f.write("- When solution accuracy is more important than raw speed\n\n")

            f.write("**When to use uniform grids**:\n")
            f.write("- Small problems where setup overhead dominates\n")
            f.write("- Globally smooth solutions\n")
            f.write("- When simplicity is preferred over optimization\n")

        print(f"Performance report generated: {report_file}")


def main():
    """Run the AMR performance benchmark suite."""
    benchmark_suite = AMRBenchmarkSuite()
    benchmark_suite.run_comprehensive_benchmark()


if __name__ == "__main__":
    main()
