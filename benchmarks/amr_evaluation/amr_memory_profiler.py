#!/usr/bin/env python3
"""
AMR Memory Usage Profiler

This module provides detailed memory usage analysis for AMR-enhanced solvers,
including memory scaling, peak usage, memory efficiency, and memory leak detection.

Memory Metrics:
- Peak memory usage during solving
- Memory scaling with problem size
- Memory efficiency (useful memory vs overhead)
- Memory leak detection across multiple solves
- Memory allocation patterns
- Memory usage by component (mesh, solution, error estimation)
"""

import gc
import json
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import psutil

import matplotlib.pyplot as plt
import numpy as np

# MFG_PDE imports
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_amr_solver, create_solver
from mfg_pde.geometry import Domain1D, periodic_bc


@dataclass
class MemorySnapshot:
    """Container for memory usage snapshot."""

    timestamp: float
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    process_memory_mb: float
    peak_memory_mb: float = 0.0


@dataclass
class MemoryAnalysisResult:
    """Container for memory analysis results."""

    problem_name: str
    solver_type: str
    problem_size: tuple[int, ...]

    # Memory usage statistics
    initial_memory_mb: float
    peak_memory_mb: float
    final_memory_mb: float
    memory_increase_mb: float
    max_memory_growth_rate: float  # MB/s

    # Memory components breakdown
    mesh_memory_mb: float = 0.0
    solution_memory_mb: float = 0.0
    overhead_memory_mb: float = 0.0

    # Memory efficiency metrics
    theoretical_minimum_mb: float = 0.0
    memory_efficiency: float = 0.0  # useful_memory / total_memory
    memory_overhead_ratio: float = 0.0

    # Memory allocation info
    num_allocations: int = 0
    allocation_pattern: list[float] = None

    # Solve statistics
    solve_time: float = 0.0
    converged: bool = False
    total_elements: int = 0


class AMRMemoryProfiler:
    """Memory usage profiler for AMR operations."""

    def __init__(self, output_dir: str = "memory_profiling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.results: list[MemoryAnalysisResult] = []
        self.memory_snapshots: list[MemorySnapshot] = []

        # Start memory tracking
        tracemalloc.start()

        print("AMR Memory Usage Profiler")
        print("=" * 50)

        # Get system memory info
        memory = psutil.virtual_memory()
        print(f"System Memory: {memory.total / (1024**3):.2f} GB total")
        print(f"Available: {memory.available / (1024**3):.2f} GB")
        print("=" * 50)

    def take_memory_snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a snapshot of current memory usage."""

        # System memory
        memory = psutil.virtual_memory()

        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()

        snapshot = MemorySnapshot(
            timestamp=time.perf_counter(),
            total_memory_mb=memory.total / (1024 * 1024),
            used_memory_mb=memory.used / (1024 * 1024),
            available_memory_mb=memory.available / (1024 * 1024),
            process_memory_mb=process_memory.rss / (1024 * 1024),
        )

        self.memory_snapshots.append(snapshot)

        if label:
            print(f"Memory snapshot ({label}): {snapshot.process_memory_mb:.1f} MB")

        return snapshot

    def estimate_theoretical_memory_requirements(
        self, problem_size: tuple[int, ...], amr_enabled: bool = False, max_refinement_levels: int = 5
    ) -> float:
        """Estimate theoretical minimum memory requirements."""

        # Basic memory requirements (in MB)
        elements = np.prod(problem_size)

        # Each element needs storage for U, M values (8 bytes each = 16 bytes per element)
        solution_memory = elements * 16 / (1024 * 1024)

        if amr_enabled:
            # AMR overhead: tree structure, error estimates, refinement flags
            # Estimate 2x more elements on average due to refinement
            avg_refinement_factor = 2.0 ** (max_refinement_levels / 2)
            amr_elements = elements * avg_refinement_factor

            # AMR tree structure overhead (pointers, metadata)
            amr_overhead = amr_elements * 64 / (1024 * 1024)  # 64 bytes per element metadata

            total_memory = solution_memory * avg_refinement_factor + amr_overhead
        else:
            total_memory = solution_memory

        return total_memory

    def profile_memory_scaling(self):
        """Profile memory usage scaling with problem size."""

        print("\n📈 Memory Scaling Analysis")
        print("-" * 40)

        problem_sizes = [32, 64, 128, 256, 512]

        for size in problem_sizes:
            print(f"\nProblem size: {size}")

            # Create 1D problem
            domain = Domain1D(0.0, 1.0, periodic_bc())
            problem = ExampleMFGProblem(T=0.5, xmin=0.0, xmax=1.0, Nx=size, Nt=20, sigma=0.1, coefCT=1.0)
            problem.domain = domain
            problem.dimension = 1

            # Test both uniform and AMR solvers
            solver_configs = [
                {
                    "name": f"Uniform-{size}",
                    "create_solver": lambda p: create_solver(p, solver_type="fixed_point", preset="fast"),
                    "amr_enabled": False,
                },
                {
                    "name": f"AMR-{size}",
                    "create_solver": lambda p, s=size: create_amr_solver(
                        p,
                        base_solver_type="fixed_point",
                        error_threshold=1e-4,
                        max_levels=4,
                        initial_intervals=s // 2,
                    ),
                    "amr_enabled": True,
                },
            ]

            for config in solver_configs:
                print(f"  Testing {config['name']}...")

                # Force garbage collection before test
                gc.collect()

                # Take initial memory snapshot
                initial_snapshot = self.take_memory_snapshot("initial")

                # Create solver
                solver = config["create_solver"](problem)

                # Take snapshot after solver creation
                self.take_memory_snapshot("solver_created")

                # Monitor memory during solving
                memory_timeline = []

                def memory_monitor_callback(timeline=memory_timeline):
                    """Callback to monitor memory during solving."""
                    current_snapshot = self.take_memory_snapshot()
                    timeline.append(current_snapshot.process_memory_mb)

                # Solve with memory monitoring
                start_time = time.perf_counter()
                result = solver.solve(max_iterations=50, tolerance=1e-6, verbose=False)
                solve_time = time.perf_counter() - start_time

                # Take final memory snapshot
                final_snapshot = self.take_memory_snapshot("final")

                # Extract solution info
                if isinstance(result, dict):
                    converged = result.get("converged", False)
                    total_elements = result.get("mesh_statistics", {}).get("total_intervals", size)
                else:
                    converged = True
                    total_elements = size

                # Calculate memory metrics
                memory_increase = final_snapshot.process_memory_mb - initial_snapshot.process_memory_mb
                peak_memory = max(
                    [initial_snapshot.process_memory_mb, *memory_timeline, final_snapshot.process_memory_mb]
                )

                # Estimate theoretical minimum
                theoretical_min = self.estimate_theoretical_memory_requirements((size,), config["amr_enabled"], 4)

                # Calculate efficiency
                memory_efficiency = theoretical_min / memory_increase if memory_increase > 0 else 0.0
                overhead_ratio = (memory_increase - theoretical_min) / theoretical_min if theoretical_min > 0 else 0.0

                # Store results
                analysis_result = MemoryAnalysisResult(
                    problem_name=f"Scaling_Test_N{size}",
                    solver_type=config["name"],
                    problem_size=(size,),
                    initial_memory_mb=initial_snapshot.process_memory_mb,
                    peak_memory_mb=peak_memory,
                    final_memory_mb=final_snapshot.process_memory_mb,
                    memory_increase_mb=memory_increase,
                    max_memory_growth_rate=memory_increase / solve_time if solve_time > 0 else 0.0,
                    theoretical_minimum_mb=theoretical_min,
                    memory_efficiency=memory_efficiency,
                    memory_overhead_ratio=overhead_ratio,
                    allocation_pattern=memory_timeline,
                    solve_time=solve_time,
                    converged=converged,
                    total_elements=total_elements,
                )

                self.results.append(analysis_result)

                # Print summary
                print(f"    Memory increase: {memory_increase:.1f} MB")
                print(f"    Peak memory: {peak_memory:.1f} MB")
                print(f"    Theoretical min: {theoretical_min:.1f} MB")
                print(f"    Efficiency: {memory_efficiency:.2f}")
                print(f"    Elements: {total_elements}")

                # Clean up
                del solver
                gc.collect()
                time.sleep(0.1)  # Allow system to settle

    def profile_memory_leaks(self):
        """Profile for memory leaks across multiple solves."""

        print("\n🔍 Memory Leak Detection")
        print("-" * 40)

        # Create a moderate-size problem
        domain = Domain1D(0.0, 1.0, periodic_bc())
        problem = ExampleMFGProblem(T=0.5, xmin=0.0, xmax=1.0, Nx=128, Nt=20, sigma=0.1, coefCT=1.0)
        problem.domain = domain
        problem.dimension = 1

        num_iterations = 10
        memory_progression = []

        print(f"Running {num_iterations} consecutive solves...")

        for i in range(num_iterations):
            print(f"  Iteration {i + 1}/{num_iterations}")

            # Force garbage collection
            gc.collect()

            # Take memory snapshot before solve
            before_snapshot = self.take_memory_snapshot()

            # Create and solve
            solver = create_amr_solver(problem, base_solver_type="fixed_point", error_threshold=1e-4, max_levels=3)

            result = solver.solve(max_iterations=30, tolerance=1e-6, verbose=False)

            # Take memory snapshot after solve
            after_snapshot = self.take_memory_snapshot()

            memory_progression.append(
                {
                    "iteration": i + 1,
                    "before_mb": before_snapshot.process_memory_mb,
                    "after_mb": after_snapshot.process_memory_mb,
                    "increase_mb": after_snapshot.process_memory_mb - before_snapshot.process_memory_mb,
                }
            )

            # Clean up explicitly
            del solver, result
            gc.collect()

            print(f"    Memory: {before_snapshot.process_memory_mb:.1f} → {after_snapshot.process_memory_mb:.1f} MB")

        # Analyze memory leak pattern
        memory_increases = [entry["increase_mb"] for entry in memory_progression]
        cumulative_increase = sum(memory_increases)
        average_increase = np.mean(memory_increases)
        trend_slope = np.polyfit(range(len(memory_increases)), memory_increases, 1)[0]

        print("\nMemory Leak Analysis:")
        print(f"  Total memory increase: {cumulative_increase:.1f} MB")
        print(f"  Average per iteration: {average_increase:.1f} MB")
        print(f"  Trend slope: {trend_slope:.3f} MB/iteration")

        if abs(trend_slope) < 0.1:
            print("  ✅ No significant memory leak detected")
        elif trend_slope > 0.1:
            print("  ⚠️ Potential memory leak detected (increasing trend)")
        else:
            print("  📉 Memory usage decreasing over time (good garbage collection)")

        # Store leak analysis results
        leak_result = MemoryAnalysisResult(
            problem_name="Memory_Leak_Test",
            solver_type="AMR_Repeated",
            problem_size=(128,),
            initial_memory_mb=memory_progression[0]["before_mb"],
            peak_memory_mb=max(entry["after_mb"] for entry in memory_progression),
            final_memory_mb=memory_progression[-1]["after_mb"],
            memory_increase_mb=cumulative_increase,
            max_memory_growth_rate=trend_slope,
            allocation_pattern=[entry["after_mb"] for entry in memory_progression],
            num_allocations=num_iterations,
        )

        self.results.append(leak_result)

    def profile_amr_memory_components(self):
        """Profile memory usage of different AMR components."""

        print("\n🧩 AMR Memory Components Analysis")
        print("-" * 40)

        # Create test problem
        domain = Domain1D(0.0, 1.0, periodic_bc())
        problem = ExampleMFGProblem(
            T=0.5,
            xmin=0.0,
            xmax=1.0,
            Nx=256,
            Nt=20,
            sigma=0.05,
            coefCT=2.0,  # Sharp features for AMR
        )
        problem.domain = domain
        problem.dimension = 1

        # Memory checkpoints
        checkpoints = {}

        # Baseline memory
        gc.collect()
        checkpoints["baseline"] = self.take_memory_snapshot("baseline")

        # Create problem
        checkpoints["problem_created"] = self.take_memory_snapshot("problem_created")

        # Create AMR solver
        amr_solver = create_amr_solver(problem, base_solver_type="fixed_point", error_threshold=1e-4, max_levels=4)
        checkpoints["solver_created"] = self.take_memory_snapshot("solver_created")

        # Solve (this will trigger AMR operations)
        result = amr_solver.solve(max_iterations=50, tolerance=1e-6, verbose=False)
        checkpoints["solved"] = self.take_memory_snapshot("solved")

        # Calculate component memory usage
        problem_memory = checkpoints["problem_created"].process_memory_mb - checkpoints["baseline"].process_memory_mb
        solver_memory = (
            checkpoints["solver_created"].process_memory_mb - checkpoints["problem_created"].process_memory_mb
        )
        solve_memory = checkpoints["solved"].process_memory_mb - checkpoints["solver_created"].process_memory_mb

        print("Memory breakdown:")
        print(f"  Problem creation: {problem_memory:.1f} MB")
        print(f"  Solver initialization: {solver_memory:.1f} MB")
        print(f"  Solving (AMR operations): {solve_memory:.1f} MB")

        # Get AMR statistics
        if isinstance(result, dict):
            mesh_stats = result.get("mesh_statistics", {})
            total_elements = mesh_stats.get("total_intervals", 256)
            max_level = mesh_stats.get("max_level", 0)
            adaptations = result.get("total_adaptations", 0)
        else:
            total_elements = 256
            max_level = 0
            adaptations = 0

        print(f"  Final elements: {total_elements}")
        print(f"  Max refinement level: {max_level}")
        print(f"  Total adaptations: {adaptations}")

        # Estimate memory per element
        if total_elements > 0:
            memory_per_element = solve_memory * 1024 * 1024 / total_elements  # bytes per element
            print(f"  Memory per element: {memory_per_element:.0f} bytes")

        # Store component analysis
        component_result = MemoryAnalysisResult(
            problem_name="Component_Analysis",
            solver_type="AMR_Components",
            problem_size=(256,),
            initial_memory_mb=checkpoints["baseline"].process_memory_mb,
            peak_memory_mb=checkpoints["solved"].process_memory_mb,
            final_memory_mb=checkpoints["solved"].process_memory_mb,
            memory_increase_mb=checkpoints["solved"].process_memory_mb - checkpoints["baseline"].process_memory_mb,
            mesh_memory_mb=solver_memory,
            solution_memory_mb=solve_memory,
            overhead_memory_mb=problem_memory,
            total_elements=total_elements,
        )

        self.results.append(component_result)

    def generate_memory_plots(self):
        """Generate memory usage analysis plots."""

        print("\nGenerating memory analysis plots...")

        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("AMR Memory Usage Analysis", fontsize=16)

            # Plot 1: Memory scaling with problem size
            scaling_results = [r for r in self.results if "Scaling_Test" in r.problem_name]

            if scaling_results:
                uniform_results = [r for r in scaling_results if "Uniform" in r.solver_type]
                amr_results = [r for r in scaling_results if "AMR" in r.solver_type]

                ax = axes[0, 0]

                if uniform_results:
                    sizes = [r.problem_size[0] for r in uniform_results]
                    memory_usage = [r.memory_increase_mb for r in uniform_results]
                    ax.loglog(sizes, memory_usage, "b-o", label="Uniform Grid", markersize=6)

                if amr_results:
                    sizes = [r.problem_size[0] for r in amr_results]
                    memory_usage = [r.memory_increase_mb for r in amr_results]
                    ax.loglog(sizes, memory_usage, "r-s", label="AMR", markersize=6)

                ax.set_xlabel("Problem Size")
                ax.set_ylabel("Memory Usage (MB)")
                ax.set_title("Memory Scaling")
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Plot 2: Memory efficiency
            if scaling_results:
                ax = axes[0, 1]

                problem_sizes = list({r.problem_size[0] for r in scaling_results})
                problem_sizes.sort()

                uniform_efficiency = []
                amr_efficiency = []

                for size in problem_sizes:
                    uniform_res = [
                        r for r in scaling_results if r.problem_size[0] == size and "Uniform" in r.solver_type
                    ]
                    amr_res = [r for r in scaling_results if r.problem_size[0] == size and "AMR" in r.solver_type]

                    if uniform_res:
                        uniform_efficiency.append(uniform_res[0].memory_efficiency)
                    if amr_res:
                        amr_efficiency.append(amr_res[0].memory_efficiency)

                if uniform_efficiency:
                    ax.plot(
                        problem_sizes[: len(uniform_efficiency)],
                        uniform_efficiency,
                        "b-o",
                        label="Uniform",
                        markersize=6,
                    )
                if amr_efficiency:
                    ax.plot(problem_sizes[: len(amr_efficiency)], amr_efficiency, "r-s", label="AMR", markersize=6)

                ax.set_xlabel("Problem Size")
                ax.set_ylabel("Memory Efficiency")
                ax.set_title("Memory Efficiency vs Problem Size")
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Plot 3: Memory leak detection
            leak_results = [r for r in self.results if r.problem_name == "Memory_Leak_Test"]

            if leak_results and leak_results[0].allocation_pattern:
                ax = axes[1, 0]
                pattern = leak_results[0].allocation_pattern
                iterations = list(range(1, len(pattern) + 1))

                ax.plot(iterations, pattern, "g-o", markersize=6)
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Memory Usage (MB)")
                ax.set_title("Memory Leak Detection")
                ax.grid(True, alpha=0.3)

                # Add trend line
                if len(pattern) > 1:
                    z = np.polyfit(iterations, pattern, 1)
                    p = np.poly1d(z)
                    ax.plot(iterations, p(iterations), "r--", alpha=0.8, label=f"Trend: {z[0]:.2f} MB/iter")
                    ax.legend()

            # Plot 4: Memory components breakdown
            component_results = [r for r in self.results if r.problem_name == "Component_Analysis"]

            if component_results:
                result = component_results[0]

                ax = axes[1, 1]
                components = ["Mesh\nStructure", "Solution\nData", "Overhead"]
                memory_values = [result.mesh_memory_mb, result.solution_memory_mb, result.overhead_memory_mb]

                colors = ["blue", "red", "green"]
                bars = ax.bar(components, memory_values, color=colors, alpha=0.7)
                ax.set_ylabel("Memory Usage (MB)")
                ax.set_title("Memory Components Breakdown")

                # Add value labels
                for bar, value in zip(bars, memory_values, strict=False):
                    if value > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.1,
                            f"{value:.1f}MB",
                            ha="center",
                            va="bottom",
                        )

            plt.tight_layout()
            plt.savefig(self.output_dir / "memory_analysis.png", dpi=150, bbox_inches="tight")
            print(f"Memory plots saved to {self.output_dir / 'memory_analysis.png'}")

            # Show if possible
            try:
                plt.show()
            except Exception:
                print("Display not available, plots saved to file.")

        except Exception as e:
            print(f"Error generating plots: {e}")

    def generate_memory_report(self):
        """Generate comprehensive memory analysis report."""

        report_file = self.output_dir / "memory_analysis_report.md"

        with open(report_file, "w") as f:
            f.write("# AMR Memory Usage Analysis Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n\n")

            # System information
            memory = psutil.virtual_memory()
            f.write("## System Information\n\n")
            f.write(f"**Total System Memory**: {memory.total / (1024**3):.2f} GB  \n")
            f.write(f"**Available Memory**: {memory.available / (1024**3):.2f} GB  \n\n")

            # Memory scaling analysis
            scaling_results = [r for r in self.results if "Scaling_Test" in r.problem_name]

            if scaling_results:
                f.write("## Memory Scaling Analysis\n\n")
                f.write("| Problem Size | Solver | Memory (MB) | Peak (MB) | Efficiency | Elements |\n")
                f.write("|--------------|---------|-------------|-----------|------------|----------|\n")

                for result in scaling_results:
                    f.write(
                        f"| {result.problem_size[0]} | {result.solver_type} | "
                        f"{result.memory_increase_mb:.1f} | {result.peak_memory_mb:.1f} | "
                        f"{result.memory_efficiency:.3f} | {result.total_elements} |\n"
                    )

                # Calculate scaling factors
                uniform_results = [r for r in scaling_results if "Uniform" in r.solver_type]
                amr_results = [r for r in scaling_results if "AMR" in r.solver_type]

                if len(uniform_results) >= 2:
                    largest_uniform = max(uniform_results, key=lambda x: x.problem_size[0])
                    smallest_uniform = min(uniform_results, key=lambda x: x.problem_size[0])

                    size_ratio = largest_uniform.problem_size[0] / smallest_uniform.problem_size[0]
                    memory_ratio = largest_uniform.memory_increase_mb / smallest_uniform.memory_increase_mb

                    f.write(
                        f"\n**Uniform Grid Scaling**: {size_ratio:.1f}x size increase → "
                        f"{memory_ratio:.1f}x memory increase  \n"
                    )

                if len(amr_results) >= 2:
                    largest_amr = max(amr_results, key=lambda x: x.problem_size[0])
                    smallest_amr = min(amr_results, key=lambda x: x.problem_size[0])

                    size_ratio = largest_amr.problem_size[0] / smallest_amr.problem_size[0]
                    memory_ratio = largest_amr.memory_increase_mb / smallest_amr.memory_increase_mb

                    f.write(
                        f"**AMR Scaling**: {size_ratio:.1f}x size increase → {memory_ratio:.1f}x memory increase  \n\n"
                    )

            # Memory leak analysis
            leak_results = [r for r in self.results if r.problem_name == "Memory_Leak_Test"]

            if leak_results:
                result = leak_results[0]
                f.write("## Memory Leak Analysis\n\n")
                f.write(f"**Test**: {result.num_allocations} consecutive solves  \n")
                f.write(f"**Total Memory Increase**: {result.memory_increase_mb:.1f} MB  \n")
                f.write(f"**Growth Rate**: {result.max_memory_growth_rate:.3f} MB/iteration  \n")

                if abs(result.max_memory_growth_rate) < 0.1:
                    f.write("**Status**: ✅ No significant memory leak detected  \n\n")
                elif result.max_memory_growth_rate > 0.1:
                    f.write("**Status**: ⚠️ Potential memory leak detected  \n\n")
                else:
                    f.write("**Status**: 📉 Memory usage stable or decreasing  \n\n")

            # Memory efficiency summary
            if scaling_results:
                avg_uniform_efficiency = np.mean(
                    [r.memory_efficiency for r in uniform_results if r.memory_efficiency > 0]
                )
                avg_amr_efficiency = np.mean([r.memory_efficiency for r in amr_results if r.memory_efficiency > 0])

                f.write("## Memory Efficiency Summary\n\n")
                f.write(f"**Average Uniform Grid Efficiency**: {avg_uniform_efficiency:.3f}  \n")
                f.write(f"**Average AMR Efficiency**: {avg_amr_efficiency:.3f}  \n")

                if avg_amr_efficiency > avg_uniform_efficiency:
                    f.write(
                        f"**AMR Advantage**: {avg_amr_efficiency / avg_uniform_efficiency:.2f}x more efficient  \n\n"
                    )
                else:
                    f.write(
                        f"**Uniform Advantage**: {avg_uniform_efficiency / avg_amr_efficiency:.2f}x more efficient  \n\n"
                    )

            # Recommendations
            f.write("## Memory Usage Recommendations\n\n")

            # Analyze results to generate recommendations
            if scaling_results:
                large_problem_amr = [r for r in amr_results if r.problem_size[0] >= 256]
                if large_problem_amr:
                    avg_efficiency = np.mean([r.memory_efficiency for r in large_problem_amr])
                    if avg_efficiency > 0.5:
                        f.write("✅ **AMR Memory Efficient**: Good memory efficiency for large problems  \n")
                    else:
                        f.write("⚠️ **AMR Memory Overhead**: Consider optimizing AMR data structures  \n")

            if leak_results and leak_results[0].max_memory_growth_rate > 0.1:
                f.write("🔍 **Memory Leak Detected**: Investigate memory cleanup in AMR operations  \n")

            f.write("\n**General Guidelines**:  \n")
            f.write("- Monitor memory usage for problems larger than available RAM  \n")
            f.write("- Consider memory-efficient AMR thresholds for large-scale problems  \n")
            f.write("- Use garbage collection between consecutive solves in batch processing  \n")

        print(f"Memory analysis report generated: {report_file}")

    def save_memory_results(self):
        """Save memory analysis results to JSON."""

        results_file = self.output_dir / "memory_analysis_results.json"

        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            result_dict = {
                "problem_name": result.problem_name,
                "solver_type": result.solver_type,
                "problem_size": list(result.problem_size),
                "initial_memory_mb": result.initial_memory_mb,
                "peak_memory_mb": result.peak_memory_mb,
                "final_memory_mb": result.final_memory_mb,
                "memory_increase_mb": result.memory_increase_mb,
                "max_memory_growth_rate": result.max_memory_growth_rate,
                "mesh_memory_mb": result.mesh_memory_mb,
                "solution_memory_mb": result.solution_memory_mb,
                "overhead_memory_mb": result.overhead_memory_mb,
                "theoretical_minimum_mb": result.theoretical_minimum_mb,
                "memory_efficiency": result.memory_efficiency,
                "memory_overhead_ratio": result.memory_overhead_ratio,
                "num_allocations": result.num_allocations,
                "allocation_pattern": result.allocation_pattern,
                "solve_time": result.solve_time,
                "converged": result.converged,
                "total_elements": result.total_elements,
            }
            serializable_results.append(result_dict)

        with open(results_file, "w") as f:
            json.dump(
                {
                    "memory_analysis_metadata": {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "system_memory_gb": psutil.virtual_memory().total / (1024**3),
                        "total_analyses": len(self.results),
                    },
                    "results": serializable_results,
                },
                f,
                indent=2,
            )

        print(f"Memory analysis results saved to {results_file}")

    def run_comprehensive_memory_analysis(self):
        """Run the complete memory analysis suite."""

        print("Starting Comprehensive AMR Memory Analysis")
        print("=" * 60)

        # Run memory scaling analysis
        self.profile_memory_scaling()

        # Run memory leak detection
        self.profile_memory_leaks()

        # Run component analysis
        self.profile_amr_memory_components()

        # Generate analysis
        self.generate_memory_plots()
        self.generate_memory_report()
        self.save_memory_results()

        print("\n✅ Memory analysis complete!")
        print(f"Results saved to: {self.output_dir}")

        # Cleanup
        tracemalloc.stop()


def main():
    """Run the AMR memory analysis suite."""
    profiler = AMRMemoryProfiler()
    profiler.run_comprehensive_memory_analysis()


if __name__ == "__main__":
    main()
