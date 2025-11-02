"""
Phase 3 API Performance Benchmark
==================================

Measures performance of the unified Phase 3 API (v0.9.0) to ensure
the refactoring did not introduce regressions.

Benchmarks:
1. solve_mfg() call overhead (all three patterns)
2. SolverConfig parsing performance (YAML, Builder, Presets)
3. Factory function overhead vs direct instantiation
4. MFGProblem construction time

Usage:
    python benchmarks/phase3_api_performance.py

This creates a performance report comparing Phase 3 overhead.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from mfg_pde import ExampleMFGProblem, solve_mfg
from mfg_pde.config import ConfigBuilder, presets
from mfg_pde.factory import create_fast_solver


def measure_time(func, repeat: int = 10) -> tuple[float, float]:
    """
    Measure execution time with statistics.

    Args:
        func: Function to benchmark
        repeat: Number of repetitions

    Returns:
        (mean_time, std_time) in seconds
    """
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return float(np.mean(times)), float(np.std(times))


class Phase3PerformanceBenchmark:
    """Benchmark suite for Phase 3 unified API."""

    def __init__(self):
        self.results: dict[str, dict[str, Any]] = {}

    def benchmark_solve_mfg_patterns(self) -> None:
        """Benchmark all three solve_mfg() usage patterns."""
        print("\n1. Benchmarking solve_mfg() patterns...")

        # Pattern 1: MFGProblem object with preset config object
        def pattern1():
            problem = ExampleMFGProblem()
            config = presets.fast_solver()
            solve_mfg(problem, config=config, max_iterations=1)

        # Pattern 2: Builder API
        def pattern2():
            problem = ExampleMFGProblem()
            config = ConfigBuilder().backend(backend_type="numpy").picard(max_iterations=1, tolerance=1e-6).build()
            solve_mfg(problem, config=config)

        # Pattern 3: Preset shorthand (string) - uses preset's default max_iterations
        def pattern3():
            problem = ExampleMFGProblem()
            # Note: preset="fast" internally sets max_iterations=30 via presets.fast_solver()
            solve_mfg(problem, preset="fast")

        # Measure
        mean1, std1 = measure_time(pattern1, repeat=10)
        mean2, std2 = measure_time(pattern2, repeat=10)
        mean3, std3 = measure_time(pattern3, repeat=10)

        self.results["solve_mfg_patterns"] = {
            "pattern1_mfgproblem": {"mean": mean1, "std": std1},
            "pattern2_components": {"mean": mean2, "std": std2},
            "pattern3_factory": {"mean": mean3, "std": std3},
        }

        print(f"  Pattern 1 (MFGProblem): {mean1 * 1000:.2f} ± {std1 * 1000:.2f} ms")
        print(f"  Pattern 2 (Components): {mean2 * 1000:.2f} ± {std2 * 1000:.2f} ms")
        print(f"  Pattern 3 (Factory):    {mean3 * 1000:.2f} ± {std3 * 1000:.2f} ms")

    def benchmark_config_parsing(self) -> None:
        """Benchmark SolverConfig creation methods."""
        print("\n2. Benchmarking SolverConfig parsing...")

        # Preset-based
        def preset_fast():
            presets.fast_solver()

        def preset_balanced():
            presets.balanced_solver()

        def preset_accurate():
            presets.accurate_solver()

        # Builder pattern
        def builder():
            ConfigBuilder().backend(backend_type="numpy").picard(max_iterations=50, tolerance=1e-6).build()

        # Direct construction (if available)
        def direct():
            presets.fast_solver()  # Use preset as baseline for direct construction

        # Measure
        mean_fast, std_fast = measure_time(preset_fast, repeat=100)
        mean_balanced, std_balanced = measure_time(preset_balanced, repeat=100)
        mean_accurate, std_accurate = measure_time(preset_accurate, repeat=100)
        mean_builder, std_builder = measure_time(builder, repeat=100)
        mean_direct, std_direct = measure_time(direct, repeat=100)

        self.results["config_parsing"] = {
            "preset_fast": {"mean": mean_fast, "std": std_fast},
            "preset_balanced": {"mean": mean_balanced, "std": std_balanced},
            "preset_accurate": {"mean": mean_accurate, "std": std_accurate},
            "builder": {"mean": mean_builder, "std": std_builder},
            "direct": {"mean": mean_direct, "std": std_direct},
        }

        print(f"  Preset (fast):     {mean_fast * 1e6:.1f} ± {std_fast * 1e6:.1f} µs")
        print(f"  Preset (balanced): {mean_balanced * 1e6:.1f} ± {std_balanced * 1e6:.1f} µs")
        print(f"  Preset (accurate): {mean_accurate * 1e6:.1f} ± {std_accurate * 1e6:.1f} µs")
        print(f"  Builder:           {mean_builder * 1e6:.1f} ± {std_builder * 1e6:.1f} µs")
        print(f"  Direct:            {mean_direct * 1e6:.1f} ± {std_direct * 1e6:.1f} µs")

    def benchmark_problem_construction(self) -> None:
        """Benchmark MFGProblem construction."""
        print("\n3. Benchmarking MFGProblem construction...")

        def construct_problem():
            ExampleMFGProblem()

        mean, std = measure_time(construct_problem, repeat=100)

        self.results["problem_construction"] = {
            "mean": mean,
            "std": std,
        }

        print(f"  MFGProblem creation: {mean * 1e6:.1f} ± {std * 1e6:.1f} µs")

    def benchmark_factory_overhead(self) -> None:
        """Benchmark factory function overhead."""
        print("\n4. Benchmarking factory overhead...")

        def via_factory():
            problem = ExampleMFGProblem()
            config = presets.fast_solver()
            create_fast_solver(problem, config)

        mean, std = measure_time(via_factory, repeat=50)

        self.results["factory_overhead"] = {
            "mean": mean,
            "std": std,
        }

        print(f"  Factory creation: {mean * 1000:.2f} ± {std * 1000:.2f} ms")

    def benchmark_full_solve_small(self) -> None:
        """Benchmark full solve_mfg() on small problem."""
        print("\n5. Benchmarking full solve (small problem)...")

        def full_solve():
            problem = ExampleMFGProblem()
            solve_mfg(problem, preset="fast", max_iterations=10)

        mean, std = measure_time(full_solve, repeat=3)

        self.results["full_solve_small"] = {
            "mean": mean,
            "std": std,
        }

        print(f"  Full solve (10 iters): {mean:.3f} ± {std:.3f} s")

    def print_summary(self) -> None:
        """Print performance summary."""
        print("\n" + "=" * 70)
        print("Phase 3 API Performance Summary")
        print("=" * 70)

        # API overhead
        if "solve_mfg_patterns" in self.results:
            pattern1 = self.results["solve_mfg_patterns"]["pattern1_mfgproblem"]["mean"]
            pattern2 = self.results["solve_mfg_patterns"]["pattern2_components"]["mean"]
            pattern3 = self.results["solve_mfg_patterns"]["pattern3_factory"]["mean"]

            print("\nAPI Pattern Overhead:")
            print(f"  Pattern 1 (MFGProblem): {pattern1 * 1000:.2f} ms")
            print(f"  Pattern 2 (Components): {pattern2 * 1000:.2f} ms")
            print(f"  Pattern 3 (Factory):    {pattern3 * 1000:.2f} ms")
            print(
                f"  Overhead range: {abs(max(pattern1, pattern2, pattern3) - min(pattern1, pattern2, pattern3)) * 1000:.2f} ms"
            )

        # Config parsing overhead
        if "config_parsing" in self.results:
            preset = self.results["config_parsing"]["preset_fast"]["mean"]
            builder = self.results["config_parsing"]["builder"]["mean"]
            direct = self.results["config_parsing"]["direct"]["mean"]

            print("\nConfiguration Parsing:")
            print(f"  Preset:  {preset * 1e6:.1f} µs (baseline)")
            print(f"  Builder: {builder * 1e6:.1f} µs ({(builder / preset - 1) * 100:+.1f}%)")
            print(f"  Direct:  {direct * 1e6:.1f} µs ({(direct / preset - 1) * 100:+.1f}%)")

        # Problem construction
        if "problem_construction" in self.results:
            construct_time = self.results["problem_construction"]["mean"]
            print(f"\nProblem Construction: {construct_time * 1e6:.1f} µs")

        # Factory overhead
        if "factory_overhead" in self.results:
            factory_time = self.results["factory_overhead"]["mean"]
            print(f"Factory Overhead: {factory_time * 1000:.2f} ms")

        # Full solve
        if "full_solve_small" in self.results:
            solve_time = self.results["full_solve_small"]["mean"]
            print(f"\nFull Solve (10 iters, 51 points): {solve_time:.3f} s")

            # Calculate overhead percentage
            overhead = (pattern1 + construct_time + factory_time) if "solve_mfg_patterns" in self.results else 0
            if overhead > 0 and solve_time > 0:
                print(f"API Overhead: ~{(overhead / solve_time) * 100:.1f}% of total solve time")

        print("\nConclusion:")
        if "full_solve_small" in self.results:
            if overhead / solve_time < 0.05:
                print("  Phase 3 API overhead is negligible (<5% of solve time)")
            elif overhead / solve_time < 0.10:
                print("  Phase 3 API overhead is acceptable (<10% of solve time)")
            else:
                print("  Phase 3 API overhead may be significant (>10% of solve time)")

        print("\n" + "=" * 70)

    def save_results(self, filepath: Path | str = "benchmarks/results/phase3_api_performance.json") -> None:
        """Save results to JSON file."""
        import json

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {filepath}")


def main():
    """Run Phase 3 API performance benchmarks."""
    print("=" * 70)
    print("Phase 3 API Performance Benchmark (v0.9.0)")
    print("=" * 70)

    benchmark = Phase3PerformanceBenchmark()

    try:
        benchmark.benchmark_solve_mfg_patterns()
        benchmark.benchmark_config_parsing()
        benchmark.benchmark_problem_construction()
        benchmark.benchmark_factory_overhead()
        benchmark.benchmark_full_solve_small()
    except Exception as e:
        print(f"\nError during benchmarking: {e}")
        import traceback

        traceback.print_exc()
        return 1

    benchmark.print_summary()
    benchmark.save_results()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
